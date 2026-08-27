"""Frozen backbone loaders for SmolVLM-256M, M-CLIP, and the polarity head.

The loaders enforce the **training-free** spec scenario: every transformer
backbone is set to `eval()` with `requires_grad=False`. `assert_frozen` is the
single source of truth that callers and tests use to verify the guarantee.

We import torch / transformers at module level by design — the inference
pipeline always runs in an environment that has them. Tests that don't need
real weights inject `BackboneBundle`-shaped dummies instead of importing this
module.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .smolvlm_check import (
    GENERATION_SMOLVLM_TYPES,
    is_smolvlm_pipeline as _is_smolvlm_pipeline,
    smolvlm_can_caption as _smolvlm_can_caption,
    smolvlm_generation_module as _smolvlm_generation_module,
)

DEFAULT_SMOLVLM_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
# Emits a 512-d embedding aligned with OpenAI CLIP ViT-B/32's image tower.
DEFAULT_MCLIP_ID = "M-CLIP/XLM-Roberta-Large-Vit-B-32"
# Tweet-trained multilingual sentiment head. The older Persian
# `sentiment-snappfood` head was fitted on food-delivery reviews and mislabels
# ordinary captions (bare emotion nouns, mourning euphemisms, dismissive slang);
# see `scripts/bench_polarity_heads.py` for the head comparison.
DEFAULT_POLARITY_ID = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
# Kept as an alias so callers pinning the old keyword argument still work.
DEFAULT_PARSBERT_POLARITY_ID = DEFAULT_POLARITY_ID

# Legacy M-CLIP repos hold a bare transformer; their CLIP projection head ships
# separately as a pickle and is only loadable via `legacy_multilingual_clip`.
# Loading them through `pt_multilingual_clip` silently yields random weights.
_LEGACY_MCLIP_IDS: frozenset[str] = frozenset(
    {
        "M-CLIP/M-BERT-Distil-40",
        "M-CLIP/M-BERT-Base-ViT-B",
        "M-CLIP/M-BERT-Base-69",
        "M-CLIP/Swedish-500k",
        "M-CLIP/Swedish-2M",
    }
)

_TRANSFORMERS_CLIP_BY_DIM: dict[int, str] = {
    512: "openai/clip-vit-base-patch32",
    768: "openai/clip-vit-large-patch14",
}

# CLIP variant used for attention heatmaps (always ViT-B/32 vision blocks).
_CLIP_ATTN_ID = "openai/clip-vit-base-patch32"


@dataclass
class BackboneBundle:
    """Lazily-loaded frozen models needed by the inference pipeline."""

    smolvlm_model: Any
    smolvlm_processor: Any
    mclip_text: Any
    mclip_image: Any
    mclip_tokenizer: Any
    mclip_image_processor: Any
    # Sentiment-polarity head (`DEFAULT_POLARITY_ID`). Field names kept for
    # back-compat; the head is no longer necessarily a ParsBERT checkpoint.
    parsbert_polarity: Any
    parsbert_tokenizer: Any
    device: str
    mclip_embed_dim: int = 512
    mclip_vision_backend: str = "clip"
    # ViT-B/32 CLIP for patch attention when mclip_image is an open_clip RN50x4 tower.
    mclip_image_attn: Any | None = None
    mclip_image_attn_processor: Any | None = None


def assert_frozen(*modules: Any) -> None:
    """Spec scenario: every backbone parameter SHALL have `requires_grad=False`.

    Modules without a `parameters()` method (e.g. tokenizers, processors) are
    skipped silently.
    """
    for mod in modules:
        if mod is None or not hasattr(mod, "parameters"):
            continue
        if hasattr(mod, "training") and mod.training:
            raise RuntimeError(f"backbone {type(mod).__name__} is in train mode")
        for name, p in mod.named_parameters():
            if p.requires_grad:
                raise RuntimeError(
                    f"backbone {type(mod).__name__}.{name} has requires_grad=True"
                )


def _sanitize_embedding(vec: Any) -> np.ndarray:
    """Finite L2-safe vector for GDRM cosine features."""
    arr = np.nan_to_num(
        np.asarray(vec, dtype=np.float32).reshape(-1),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return arr


def _freeze(module: Any) -> Any:
    """Put `module` in eval mode and disable grad on every parameter."""
    if hasattr(module, "eval"):
        module.eval()
    if hasattr(module, "parameters"):
        for p in module.parameters():
            p.requires_grad_(False)
    return module


def _mclip_embed_dim(mclip_text: Any) -> int:
    return int(mclip_text.LinearTransformation.out_features)


def _resolve_mclip_source(mclip_id: str) -> str:
    """Prefer a pre-fetched local copy so runs work on a flaky connection.

    Populated by `scripts/fetch_mclip.py`; falls back to the Hub id.
    """
    local = Path("models") / mclip_id.replace("/", "--")
    if (local / "config.json").is_file() and (local / "pytorch_model.bin").is_file():
        return str(local)
    return mclip_id


@contextmanager
def _base_model_from_config_only():
    """Skip the redundant base-model weight download inside `MultilingualCLIP`.

    `MultilingualCLIP.__init__` calls `AutoModel.from_pretrained(modelBase)`,
    pulling a second multi-GB checkpoint whose weights are then fully overwritten
    by the M-CLIP state dict. Building the base from its config alone avoids the
    download; `_assert_mclip_weights_loaded` verifies nothing is left unfilled.
    """
    import transformers

    original = transformers.AutoModel.from_pretrained

    def from_config_instead(name_or_path, *args, **kwargs):
        config = transformers.AutoConfig.from_pretrained(name_or_path)
        return transformers.AutoModel.from_config(config)

    transformers.AutoModel.from_pretrained = from_config_instead
    try:
        yield
    finally:
        transformers.AutoModel.from_pretrained = original


def _cuda_gc() -> None:
    """Best-effort reclaim of CUDA memory between staged loads."""
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _maybe_half(module: Any, *, device: str) -> Any:
    import torch

    if str(device).startswith("cuda"):
        return module.half()
    return module


def _load_mclip_text(mclip_id: str, *, device: str, fp16: bool = True) -> Any:
    from multilingual_clip import pt_multilingual_clip
    from transformers import AutoTokenizer

    if mclip_id in _LEGACY_MCLIP_IDS:
        raise RuntimeError(
            f"{mclip_id!r} is a legacy M-CLIP repo that ships only a bare transformer; "
            f"its CLIP projection head is a separate pickle. Loading it here would "
            f"produce random weights and all-zero embeddings. "
            f"Use {DEFAULT_MCLIP_ID!r} instead."
        )

    source = _resolve_mclip_source(mclip_id)
    with _base_model_from_config_only():
        mclip_text, loading_info = pt_multilingual_clip.MultilingualCLIP.from_pretrained(
            source, output_loading_info=True
        )
    _assert_mclip_weights_loaded(source, loading_info)
    _freeze(mclip_text)
    mclip_text.to(device)
    if fp16:
        mclip_text = _maybe_half(mclip_text, device=device)
    tokenizer = AutoTokenizer.from_pretrained(source)
    return mclip_text, tokenizer


def _assert_mclip_weights_loaded(mclip_id: str, loading_info: dict) -> None:
    """Fail loudly when the checkpoint did not populate the M-CLIP text tower.

    A silent random init here is invisible downstream: NaN activations get
    flattened to zero vectors by `_sanitize_embedding`, which turns every GDRM
    cosine feature into a constant (Dsem == 1.0, Fvt == 0.0).
    """
    missing = [k for k in (loading_info.get("missing_keys") or []) if not k.endswith(".position_ids")]
    if missing:
        preview = ", ".join(sorted(missing)[:5])
        raise RuntimeError(
            f"M-CLIP checkpoint {mclip_id!r} left {len(missing)} weight(s) uninitialized "
            f"(e.g. {preview}). The text encoder would emit zero vectors. "
            f"Verify the repo exposes both `transformer.*` and `LinearTransformation.*` tensors."
        )


def _load_mclip_image_encoder(
    embed_dim: int, *, device: str, fp16: bool = True
) -> tuple[str, Any, Any, Any | None, Any | None]:
    """Return (backend, image_model, processor, attn_model, attn_processor)."""
    import torch
    from transformers import CLIPModel, AutoImageProcessor

    dtype = torch.float16 if fp16 and str(device).startswith("cuda") else torch.float32

    if embed_dim in _TRANSFORMERS_CLIP_BY_DIM:
        clip_id = _TRANSFORMERS_CLIP_BY_DIM[embed_dim]
        clip = CLIPModel.from_pretrained(
            clip_id, torch_dtype=dtype, attn_implementation="eager"
        )
        proc = AutoImageProcessor.from_pretrained(clip_id)
        _freeze(clip)
        clip.to(device)
        return "clip", clip, proc, None, None

    if embed_dim == 640:
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            "RN50x4", pretrained="openai"
        )
        _freeze(model)
        model.to(device)
        attn = CLIPModel.from_pretrained(
            _CLIP_ATTN_ID, torch_dtype=torch.float32, attn_implementation="eager"
        )
        attn_proc = AutoImageProcessor.from_pretrained(_CLIP_ATTN_ID)
        _freeze(attn)
        attn.to(device)
        return "open_clip", model, preprocess, attn, attn_proc

    raise RuntimeError(
        f"Unsupported M-CLIP embedding dimension {embed_dim}. "
        f"Supported: {sorted(_TRANSFORMERS_CLIP_BY_DIM)} or 640 (open_clip RN50x4)."
    )


def _load_smolvlm_captioner(
    smolvlm_id: str, *, torch_dtype: Any, device: str
) -> tuple[Any, Any]:
    """Return `(processor, captioner)` where captioner is a HF pipeline or gen model."""
    import torch
    from transformers import AutoProcessor, Idefics3ForConditionalGeneration, pipeline

    proc = AutoProcessor.from_pretrained(smolvlm_id)
    errors: list[str] = []

    try:
        captioner = pipeline(
            task="image-text-to-text",
            model=smolvlm_id,
            dtype=torch_dtype,
            device=device,
        )
        if _smolvlm_can_caption(captioner):
            if hasattr(captioner, "model"):
                _freeze(captioner.model)
            return proc, captioner
        errors.append(f"pipeline inner={type(getattr(captioner, 'model', captioner)).__name__}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"pipeline: {exc}")

    try:
        model = Idefics3ForConditionalGeneration.from_pretrained(smolvlm_id, dtype=torch_dtype)
        if type(model).__name__ in GENERATION_SMOLVLM_TYPES:
            model.to(device)
            return proc, model
        errors.append(f"Idefics3ForConditionalGeneration -> {type(model).__name__}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Idefics3ForConditionalGeneration: {exc}")

    raise RuntimeError(
        f"Could not load generation-capable SmolVLM from {smolvlm_id!r}. "
        f"Details: {'; '.join(errors)}"
    )


def load_backbones(
    *,
    device: str | None = None,
    smolvlm_id: str = DEFAULT_SMOLVLM_ID,
    mclip_id: str = DEFAULT_MCLIP_ID,
    parsbert_id: str = DEFAULT_PARSBERT_POLARITY_ID,
) -> BackboneBundle:
    """Load every backbone, freeze it, and return a `BackboneBundle`."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # --- SmolVLM-256M (captioner) ------------------------------------------
    smolvlm_proc, smolvlm = _load_smolvlm_captioner(
        smolvlm_id, torch_dtype=torch.float32, device=dev
    )
    if not _is_smolvlm_pipeline(smolvlm):
        _freeze(smolvlm)

    # --- M-CLIP (text + image, matched embedding dimension) ----------------
    mclip_text, mclip_tokenizer = _load_mclip_text(mclip_id, device=dev, fp16=str(dev).startswith("cuda"))
    mclip_dim = _mclip_embed_dim(mclip_text)
    backend, mclip_image, mclip_image_processor, mclip_attn, mclip_attn_proc = (
        _load_mclip_image_encoder(mclip_dim, device=dev, fp16=str(dev).startswith("cuda"))
    )

    # --- Polarity classifier -----------------------------------------------
    parsbert = AutoModelForSequenceClassification.from_pretrained(parsbert_id)
    _freeze(parsbert)
    parsbert.to(dev)
    _maybe_half(parsbert, device=dev)
    parsbert_tok = AutoTokenizer.from_pretrained(parsbert_id)

    frozen = [smolvlm, mclip_text, mclip_image, parsbert]
    if mclip_attn is not None:
        frozen.append(mclip_attn)
    assert_frozen(*frozen)
    bundle = BackboneBundle(
        smolvlm_model=smolvlm,
        smolvlm_processor=smolvlm_proc,
        mclip_text=mclip_text,
        mclip_image=mclip_image,
        mclip_tokenizer=mclip_tokenizer,
        mclip_image_processor=mclip_image_processor,
        parsbert_polarity=parsbert,
        parsbert_tokenizer=parsbert_tok,
        device=dev,
        mclip_embed_dim=mclip_dim,
        mclip_vision_backend=backend,
        mclip_image_attn=mclip_attn,
        mclip_image_attn_processor=mclip_attn_proc,
    )
    assert_mclip_embeddings_usable(bundle)
    return bundle


def _partial_bundle(device: str, **fields: Any) -> BackboneBundle:
    """A bundle with only the backbones one stage needs; the rest stay `None`."""
    defaults: dict[str, Any] = {
        "smolvlm_model": None,
        "smolvlm_processor": None,
        "mclip_text": None,
        "mclip_image": None,
        "mclip_tokenizer": None,
        "mclip_image_processor": None,
        "parsbert_polarity": None,
        "parsbert_tokenizer": None,
    }
    defaults.update(fields)
    return BackboneBundle(device=device, **defaults)


def resolve_device(device: str | None = None) -> str:
    import torch

    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def load_captioner_only(
    *, device: str | None = None, smolvlm_id: str = DEFAULT_SMOLVLM_ID
) -> BackboneBundle:
    """Bring up only SmolVLM, for the captioning stage."""
    import torch

    dev = resolve_device(device)
    dtype = torch.float16 if str(dev).startswith("cuda") else torch.float32
    proc, smolvlm = _load_smolvlm_captioner(
        smolvlm_id, torch_dtype=dtype, device=dev
    )
    if not _is_smolvlm_pipeline(smolvlm):
        _freeze(smolvlm)
    assert_frozen(smolvlm)
    return _partial_bundle(dev, smolvlm_model=smolvlm, smolvlm_processor=proc)


def load_mclip_text_only(
    *,
    device: str | None = None,
    mclip_id: str = DEFAULT_MCLIP_ID,
) -> BackboneBundle:
    """Bring up only the M-CLIP text tower.

    Defaults to **CPU**: XLM-Roberta-Large fp16 weights alone are ~1.04 GiB, so
    the text tower cannot satisfy the ≤1 GiB VRAM budget on GPU. Image / SmolVLM
    / ParsBERT stages still run on CUDA; the proposal explicitly allows CPU.
    """
    # Prefer CPU unless the caller forces a device (e.g. interactive dashboard).
    dev = "cpu" if device is None else resolve_device(device)
    use_fp16 = str(dev).startswith("cuda")
    mclip_text, mclip_tokenizer = _load_mclip_text(mclip_id, device=dev, fp16=use_fp16)
    assert_frozen(mclip_text)
    bundle = _partial_bundle(
        dev,
        mclip_text=mclip_text,
        mclip_tokenizer=mclip_tokenizer,
    )
    bundle.mclip_embed_dim = _mclip_embed_dim(mclip_text)
    assert_mclip_embeddings_usable(bundle)
    return bundle


def load_mclip_image_only(
    *,
    device: str | None = None,
    embed_dim: int = 512,
    with_attention_tower: bool = False,
) -> BackboneBundle:
    """Bring up only the CLIP image tower matched to the M-CLIP text embed dim."""
    dev = resolve_device(device)
    backend, mclip_image, mclip_proc, attn, attn_proc = _load_mclip_image_encoder(
        embed_dim, device=dev, fp16=True
    )
    if not with_attention_tower and attn is not None:
        attn, attn_proc = None, None
    frozen = [mclip_image] + ([attn] if attn is not None else [])
    assert_frozen(*frozen)
    bundle = _partial_bundle(
        dev,
        mclip_image=mclip_image,
        mclip_image_processor=mclip_proc,
    )
    bundle.mclip_embed_dim = embed_dim
    bundle.mclip_vision_backend = backend
    bundle.mclip_image_attn = attn
    bundle.mclip_image_attn_processor = attn_proc
    return bundle


def load_mclip_only(
    *,
    device: str | None = None,
    mclip_id: str = DEFAULT_MCLIP_ID,
    with_attention_tower: bool = False,
) -> BackboneBundle:
    """Bring up both M-CLIP towers (dashboard / interactive use).

    Staged extraction keeps the text tower on CPU and the image tower on CUDA
    so peak VRAM stays under 1 GiB. This helper still colocates both when a
    single interactive device is requested.
    """
    # Dashboard path: honor explicit/resolved device for both towers.
    text_device = resolve_device(device)
    text = load_mclip_text_only(device=text_device, mclip_id=mclip_id)
    try:
        image = load_mclip_image_only(
            device=device,
            embed_dim=text.mclip_embed_dim,
            with_attention_tower=with_attention_tower,
        )
    except Exception:
        release(text)
        raise
    # Merge into one bundle; drop the temporary image-only shell.
    text.mclip_image = image.mclip_image
    text.mclip_image_processor = image.mclip_image_processor
    text.mclip_vision_backend = image.mclip_vision_backend
    text.mclip_image_attn = image.mclip_image_attn
    text.mclip_image_attn_processor = image.mclip_image_attn_processor
    image.mclip_image = None
    image.mclip_image_processor = None
    image.mclip_image_attn = None
    image.mclip_image_attn_processor = None
    return text


def load_polarity_only(
    *, device: str | None = None, parsbert_id: str = DEFAULT_PARSBERT_POLARITY_ID
) -> BackboneBundle:
    """Bring up only the polarity head, for the sentiment stage."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dev = resolve_device(device)
    parsbert = AutoModelForSequenceClassification.from_pretrained(parsbert_id)
    _freeze(parsbert)
    parsbert.to(dev)
    _maybe_half(parsbert, device=dev)
    assert_frozen(parsbert)
    return _partial_bundle(
        dev,
        parsbert_polarity=parsbert,
        parsbert_tokenizer=AutoTokenizer.from_pretrained(parsbert_id),
    )


def release(bundle: BackboneBundle) -> None:
    """Drop every backbone reference in `bundle` and reclaim the CUDA cache."""
    import gc

    import torch

    smol = bundle.smolvlm_model
    if smol is not None:
        inner = getattr(smol, "model", None)
        if inner is not None:
            del inner
        if hasattr(smol, "destroy"):
            try:
                smol.destroy()
            except Exception:  # noqa: BLE001
                pass
        del smol

    for name in (
        "smolvlm_model",
        "smolvlm_processor",
        "mclip_text",
        "mclip_image",
        "mclip_tokenizer",
        "mclip_image_processor",
        "parsbert_polarity",
        "parsbert_tokenizer",
        "mclip_image_attn",
        "mclip_image_attn_processor",
    ):
        setattr(bundle, name, None)
    _cuda_gc()


def assert_mclip_embeddings_usable(bundle: BackboneBundle) -> None:
    """Probe the text tower so degenerate embeddings surface at load time.

    Two distinct sentences must yield finite, non-zero, non-identical vectors.
    """
    a = embed_text_mclip(bundle, "این عکس خیلی قشنگ است")
    b = embed_text_mclip(bundle, "ماشین در خیابان پارک شده بود")
    for name, vec in (("probe_a", a), ("probe_b", b)):
        if not np.all(np.isfinite(vec)):
            raise RuntimeError(f"M-CLIP text embedding {name} is not finite")
        if float(np.linalg.norm(vec)) == 0.0:
            raise RuntimeError(
                f"M-CLIP text embedding {name} is a zero vector; the text tower "
                f"did not load pretrained weights"
            )
    if np.allclose(a, b):
        raise RuntimeError(
            "M-CLIP text encoder returned identical embeddings for different "
            "sentences; the projection head is likely uninitialized"
        )


# --------------- Inference helpers (used by the pipeline) ---------------------


def caption_image(bundle: BackboneBundle, image) -> str:
    """Generate the objective description `T_hat` for an image using SmolVLM."""
    import torch

    proc = bundle.smolvlm_processor
    model = bundle.smolvlm_model

    if not _smolvlm_can_caption(model):
        proc, model = _load_smolvlm_captioner(
            DEFAULT_SMOLVLM_ID,
            torch_dtype=torch.float32,
            device=bundle.device,
        )
        bundle.smolvlm_processor = proc
        bundle.smolvlm_model = model

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        "In one or two sentences, describe this image. "
                        "State the facial expression explicitly: smiling/laughing, "
                        "sad/crying/frowning, or neutral."
                    ),
                },
            ],
        }
    ]
    prompt = proc.apply_chat_template(messages, add_generation_prompt=True)

    if _is_smolvlm_pipeline(model):
        with torch.no_grad():
            outputs = model(
                text=prompt,
                images=[image],
                max_new_tokens=64,
                return_full_text=False,
            )
        if isinstance(outputs, list) and outputs:
            text = outputs[0].get("generated_text", str(outputs[0]))
        elif isinstance(outputs, dict):
            text = outputs.get("generated_text", str(outputs))
        else:
            text = str(outputs)
    else:
        gen_model = _smolvlm_generation_module(model)
        if gen_model is None:
            raise RuntimeError(
                f"SmolVLM captioner is {type(model).__name__} and cannot generate text."
            )
        inputs = proc(text=prompt, images=[image], return_tensors="pt").to(bundle.device)
        with torch.no_grad():
            out = gen_model.generate(**inputs, max_new_tokens=64, do_sample=False)
        text = proc.batch_decode(
            out[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )[0]

    if "Assistant:" in text:
        text = text.split("Assistant:")[-1].strip()
    text = text.strip()
    if not text:
        text = "."
    return text


def embed_text_mclip(bundle: BackboneBundle, text: str) -> np.ndarray:
    """Embed `text` (any language M-CLIP supports) into the shared space."""
    import torch

    text = (text or "").strip() or "."
    tok = bundle.mclip_tokenizer
    model = bundle.mclip_text
    txt_tok = tok(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(bundle.device)
    with torch.no_grad():
        embs = model.transformer(**txt_tok)[0]
        att = txt_tok["attention_mask"]
        pooled = (embs * att.unsqueeze(2)).sum(dim=1) / att.sum(dim=1)[:, None]
        emb = model.LinearTransformation(pooled).cpu().numpy()[0]
    return _sanitize_embedding(emb)


def embed_image_mclip(bundle: BackboneBundle, image) -> np.ndarray:
    """Embed an image into the M-CLIP shared space (same dim as text embeddings)."""
    import torch

    if bundle.mclip_vision_backend == "open_clip":
        tensor = bundle.mclip_image_processor(image).unsqueeze(0).to(bundle.device)
        with torch.no_grad():
            feats = bundle.mclip_image.encode_image(tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return _sanitize_embedding(feats.cpu().numpy()[0])

    inputs = bundle.mclip_image_processor(images=image, return_tensors="pt").to(bundle.device)
    with torch.no_grad():
        out = bundle.mclip_image.get_image_features(**inputs).cpu().numpy()[0]
    return _sanitize_embedding(out)


def _clip_affect_context(bundle: BackboneBundle):
    """Cache CLIP smile/sad text embeddings on the bundle (one-time)."""
    cached = getattr(bundle, "_clip_affect", None)
    if cached is not None:
        return cached

    import torch
    from transformers import CLIPModel, CLIPProcessor

    from data.image_affect import CLIP_ID, NEG_PROMPTS, POS_PROMPTS

    device = bundle.device
    if bundle.mclip_vision_backend == "clip" and hasattr(bundle.mclip_image, "get_text_features"):
        model = bundle.mclip_image
    else:
        dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
        model = CLIPModel.from_pretrained(CLIP_ID, torch_dtype=dtype)
        _freeze(model)
        model.to(device)

    proc = CLIPProcessor.from_pretrained(CLIP_ID)
    prompts = list(POS_PROMPTS) + list(NEG_PROMPTS)
    text_inputs = proc(text=prompts, return_tensors="pt", padding=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    with torch.no_grad():
        text_emb = model.get_text_features(**text_inputs)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = model.logit_scale.exp().float()
    ctx = {
        "model": model,
        "proc": proc,
        "text_emb": text_emb,
        "scale": scale,
        "n_pos": len(POS_PROMPTS),
        "device": device,
    }
    bundle._clip_affect = ctx  # type: ignore[attr-defined]
    return ctx


def image_polarity_probs(bundle: BackboneBundle, image) -> np.ndarray:
    """CLIP smile-vs-sad as ``(p_negative, p_positive)`` for ``polarity_T_hat``.

    SmolVLM captions are too bland for image affect, so GDRM uses this CLIP
    channel as T̂ polarity while T̂ text still feeds Dsem / Fvt.
    """
    import torch

    from data.image_affect import polarity_vector

    ctx = _clip_affect_context(bundle)
    model, proc = ctx["model"], ctx["proc"]
    device, dtype = ctx["device"], next(model.parameters()).dtype
    inputs = proc(images=image, return_tensors="pt", padding=True)
    pixel = inputs["pixel_values"].to(device, dtype=dtype)
    with torch.no_grad():
        img_emb = model.get_image_features(pixel_values=pixel)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        sim = img_emb.float() @ ctx["text_emb"].float().T
        pos_logit = ctx["scale"] * sim[:, : ctx["n_pos"]].mean(dim=1)
        neg_logit = ctx["scale"] * sim[:, ctx["n_pos"] :].mean(dim=1)
        two = torch.stack([neg_logit, pos_logit], dim=1).softmax(dim=-1)[0]
    neg, pos = float(two[0]), float(two[1])
    return np.asarray(polarity_vector(pos, neg), dtype=np.float64)


def polarity_probs(bundle: BackboneBundle, text: str) -> np.ndarray:
    """Score Persian `text` and return ``(p_negative, p_positive)``.

    The vector is symmetric around the signed sentiment scalar, so
    :func:`inference.gdrm.polarity_scalar` stays ``p_pos - p_neg`` and
    ``Dsen`` reduces to the absolute difference of the two scalars.

    Two grammatical rewrites wrap the head, because sentiment models trained on
    review text mishandle both constructions regardless of vocabulary:

    - **negation** («ناراحت نیستم», «بد نبود») — score the de-negated claim and
      flip the sign, rather than trusting the head on the negated surface form.
    - **contrast** («… بود اما الان خوشحال است») — weight the clause after
      اما/ولی/لیکن, which carries the asserted sentiment.
    - **mourning / fixed formulas** («شادروان», «مرحوم», «خدا بیامرزدش»,
      «باور نکن», «به درک») — conventional phrases the head misreads
      (e.g. «شاد» inside «شادروان»).

    Negation and contrast use closed-class function words. The formula list is
    a short set of frozen idioms, not open sentiment vocabulary.
    """
    scalar = _contrast_aware_scalar(bundle, text)
    return _scalar_to_probs(scalar)


_POS_LABELS = frozenset({"HAPPY", "POSITIVE", "POS", "GOOD", "LOVE", "POS_LABEL", "LABEL_2"})
_NEG_LABELS = frozenset({"SAD", "NEGATIVE", "NEG", "BAD", "ANGRY", "NEG_LABEL", "LABEL_0"})

# Frozen formulas the head systematically misreads. Not a sentiment lexicon:
# funeral/condolence formulas, dismissive curses, and «I'm fine — don't believe it».
_FIXED_NEG_RE = re.compile(
    r"شادروان|مرحوم|فقید|مغفور|زنده‌?\s*یاد|"
    r"خدا\s*بیامرز|خدا\s*رحمت|طلب\s*مغفرت|"
    r"روحش(?:ون)?\s*شاد|روانش\s*شاد|"
    r"تسلیت|تعزیت|ترحیم|فاتحه|خاکسپاری|مجلس\s*ختم|"
    r"درگذشت|فوت\s*(?:کرد|کرده|نمود|شده)|رخت\s*بست|"
    r"به\s*رحمت\s*(?:خدا|ایزدی)|انا\s*لله|راجعون|"
    r"آسمانی\s*شد|دار\s*فانی|غم\s*آخر(?:ت|تان|تون)|"
    r"خدا\s*صبر|یادش(?:ان)?\s*گرامی|نور\s*به\s*قبر|قرین\s*رحمت|"
    r"رحمه?\s*الله|رحم[ةه]\s*الله|"
    r"به\s*درک|گور\s*پدر|برو\s*بمیر|"
    r"باور\s*نکن|هر\s*هر\s*هر|"
    r"\bRIP\b|rest\s+in\s+peace|condolences?",
    re.I,
)
_FIXED_NEG_SCALAR = -0.75
_CONTRAST_TAIL_WEIGHT = 0.75
# Negation rarely asserts the exact opposite, so damp the flipped score.
_NEGATION_FLIP = 0.85
# Below this, the de-negated claim is too vague to justify flipping the sign.
_NEGATION_MIN_AFFIRMATIVE = 0.5

# Closed-class negation cues (function words, never sentiment vocabulary).
# Bare «نه» only counts as its own clause (not inside «نه تنها / نه فقط»).
_NEGATION_RE = re.compile(
    r"(?:^|\s)(?:نیست\w*|نبود\w*|نباش\w*|نشد\w*|ندار\w*|نمی[\u200c\s]?\w+|هیچ|بدون)"
    r"(?=\s|$|[،,.!?؛;])"
    r"|(?:^|\s)نه(?=$|[،,.!?؛;])"
)

# Rewrites that strip the negative prefix so the head rates the bare claim.
_DENEGATION_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"نمی[\u200c\s]?"), "می‌"),
    (re.compile(r"\bنیستم\b"), "هستم"),
    (re.compile(r"\bنیستی\b"), "هستی"),
    (re.compile(r"\bنیستند\b"), "هستند"),
    (re.compile(r"\bنیست\b"), "است"),
    (re.compile(r"\bن(بود\w*|شد\w*|دار\w*|باش\w*|کرد\w*|رفت\w*)"), r"\1"),
    (re.compile(r"(?:^|\s)بدون\s"), " با "),
    (re.compile(r"(?:^|\s)هیچ\s"), " "),
)

_CONTRAST_RE = re.compile(r"(?:^|[\s،,؛;])(?:اما|ولی|لیکن)(?=[\s،,؛;]|$)")

# Spoken / clitic forms the frozen head misreads. Applied only on the polarity
# path (not mCLIP): «نبین که می‌خندم» is the standard wording of the same line.
_SPOKEN_FOR_POLARITY: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"نبین\s*که"), "نگاه نکن که"),
    (re.compile(r"خنده[\s\u200c]*داره\s*[؟!?]*"), "بامزه است"),
    (re.compile(r"خنده[\s\u200c]*دار(?=[\s؟!]|$)"), "بامزه"),
    (re.compile(r"خندانیم"), "خندانی هستم"),
    (re.compile(r"خندانم"), "میخندم"),
)


def spoken_for_polarity(text: str) -> str:
    """Map a few spoken smile-self lines to wording the polarity head knows."""
    out = text or ""
    for pattern, repl in _SPOKEN_FOR_POLARITY:
        out = pattern.sub(repl, out)
    return out


def has_negation(text: str) -> bool:
    """True if `text` carries an explicit Persian negation cue."""
    return bool(_NEGATION_RE.search(text or ""))


def denegate(text: str) -> str:
    """Rewrite a negated Persian clause into its affirmative form."""
    out = text or ""
    for pattern, repl in _DENEGATION_RULES:
        out = pattern.sub(repl, out)
    return re.sub(r"\s+", " ", out).strip()


def contrast_tail(text: str) -> str:
    """Clause after the last اما/ولی/لیکن, or `text` when there is no contrast."""
    t = text or ""
    matches = list(_CONTRAST_RE.finditer(t))
    if not matches:
        return t
    return t[matches[-1].end() :].strip() or t


def _scalar_to_probs(scalar: float) -> np.ndarray:
    s = float(np.clip(scalar, -1.0, 1.0))
    return np.array([(1.0 - s) / 2.0, (1.0 + s) / 2.0], dtype=np.float32)


def _head_scalar(bundle: BackboneBundle, text: str) -> float:
    """Raw signed sentiment of `text` from the frozen head, in ``[-1, +1]``."""
    import torch
    from torch.nn.functional import softmax

    if not (text or "").strip():
        return 0.0
    inputs = bundle.parsbert_tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    ).to(bundle.device)
    with torch.no_grad():
        logits = bundle.parsbert_polarity(**inputs).logits
    probs = softmax(logits.float(), dim=-1).cpu().numpy()[0]
    probs = np.nan_to_num(np.asarray(probs, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    total = float(probs.sum())
    if total > 0.0:
        probs = probs / total
    elif probs.size >= 2:
        probs = np.array([0.5, 0.5], dtype=np.float32)
    ordered = _canonicalize_polarity_probs(probs, bundle.parsbert_polarity)
    return float(ordered[1] - ordered[0])


def _clause_scalar(bundle: BackboneBundle, clause: str) -> float:
    """Sentiment of one clause, flipping the sign when it is negated."""
    raw = _head_scalar(bundle, clause)
    if not has_negation(clause):
        return raw
    rewritten = denegate(clause)
    if rewritten == clause:
        return raw
    affirmative = _head_scalar(bundle, rewritten)
    # A weak affirmative reading means the cue was not scoping sentiment (e.g.
    # the wish «کاش هیچ‌وقت نمی‌دیدمش»), so keep the head's own reading.
    if abs(affirmative) < _NEGATION_MIN_AFFIRMATIVE:
        return raw
    return -_NEGATION_FLIP * affirmative


def _contrast_aware_scalar(bundle: BackboneBundle, text: str) -> float:
    from data.preprocess import normalize_persian

    normalized = spoken_for_polarity(normalize_persian(text or "").strip())
    if not normalized:
        return 0.0
    if _FIXED_NEG_RE.search(normalized):
        return _FIXED_NEG_SCALAR
    whole = _clause_scalar(bundle, normalized)
    tail = contrast_tail(normalized)
    if tail == normalized:
        return whole
    w = _CONTRAST_TAIL_WEIGHT
    return w * _clause_scalar(bundle, tail) + (1.0 - w) * whole


def _canonicalize_polarity_probs(probs: np.ndarray, model: Any) -> np.ndarray:
    """Map classifier output to ``(p_neg, p_pos)`` using ``id2label`` when present.

    Heads with a neutral class (e.g. ``{negative, neutral, positive}``) drop the
    neutral mass, so the returned pair need not sum to 1; the signed difference
    ``p_pos - p_neg`` is what callers consume.
    """
    if probs.size < 2:
        raise ValueError("polarity vector must have at least 2 entries")
    id2label = getattr(getattr(model, "config", None), "id2label", None) or {}
    labels = {int(i): str(name).upper() for i, name in id2label.items()}
    pos_idx = next((i for i, name in labels.items() if name in _POS_LABELS), None)
    neg_idx = next((i for i, name in labels.items() if name in _NEG_LABELS), None)
    if pos_idx is not None and neg_idx is not None and pos_idx != neg_idx:
        return np.array([float(probs[neg_idx]), float(probs[pos_idx])], dtype=np.float32)
    # Fallback for unlabeled heads: legacy assumption (neg, pos) by index.
    return np.asarray(probs[:2], dtype=np.float32)
