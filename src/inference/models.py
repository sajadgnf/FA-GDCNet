"""Frozen backbone loaders for SmolVLM-256M, M-CLIP, and ParsBERT.

The loaders enforce the **training-free** spec scenario: every transformer
backbone is set to `eval()` with `requires_grad=False`. `assert_frozen` is the
single source of truth that callers and tests use to verify the guarantee.

We import torch / transformers at module level by design — the inference
pipeline always runs in an environment that has them. Tests that don't need
real weights inject `BackboneBundle`-shaped dummies instead of importing this
module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .smolvlm_check import (
    GENERATION_SMOLVLM_TYPES,
    is_smolvlm_pipeline as _is_smolvlm_pipeline,
    smolvlm_can_caption as _smolvlm_can_caption,
    smolvlm_generation_module as _smolvlm_generation_module,
)

DEFAULT_SMOLVLM_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
# Pairs with a 640-d CLIP space (RN50x4 via open_clip). Do not use vit-b/32 (512-d) here.
DEFAULT_MCLIP_ID = "M-CLIP/M-BERT-Distil-40"
DEFAULT_PARSBERT_POLARITY_ID = "HooshvareLab/bert-fa-base-uncased-sentiment-snappfood"

# (model_base, transformer_hidden, mclip_projection_dim) per HF M-CLIP model cards.
_MCLIP_CONFIGS: dict[str, tuple[str, int, int]] = {
    "M-CLIP/M-BERT-Distil-40": ("distilbert-base-multilingual-cased", 768, 640),
    "M-CLIP/M-BERT-Base-ViT-B": ("bert-base-multilingual-cased", 768, 640),
}

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


def _load_mclip_text(mclip_id: str, *, device: str) -> Any:
    from multilingual_clip import Config_MCLIP
    from multilingual_clip import pt_multilingual_clip
    from transformers import AutoTokenizer

    if mclip_id in _MCLIP_CONFIGS:
        model_base, transformer_dim, image_dim = _MCLIP_CONFIGS[mclip_id]
        config = Config_MCLIP.MCLIPConfig(
            modelBase=model_base,
            transformerDimSize=transformer_dim,
            imageDimSize=image_dim,
        )
        mclip_text = pt_multilingual_clip.MultilingualCLIP.from_pretrained(mclip_id, config=config)
    else:
        mclip_text = pt_multilingual_clip.MultilingualCLIP.from_pretrained(mclip_id)
    _freeze(mclip_text)
    mclip_text.to(device)
    tokenizer = AutoTokenizer.from_pretrained(mclip_id)
    return mclip_text, tokenizer


def _load_mclip_image_encoder(
    embed_dim: int, *, device: str
) -> tuple[str, Any, Any, Any | None, Any | None]:
    """Return (backend, image_model, processor, attn_model, attn_processor)."""
    import torch
    from transformers import CLIPModel, AutoImageProcessor

    if embed_dim in _TRANSFORMERS_CLIP_BY_DIM:
        clip_id = _TRANSFORMERS_CLIP_BY_DIM[embed_dim]
        clip = CLIPModel.from_pretrained(
            clip_id, torch_dtype=torch.float32, attn_implementation="eager"
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
    mclip_text, mclip_tokenizer = _load_mclip_text(mclip_id, device=dev)
    mclip_dim = _mclip_embed_dim(mclip_text)
    backend, mclip_image, mclip_image_processor, mclip_attn, mclip_attn_proc = (
        _load_mclip_image_encoder(mclip_dim, device=dev)
    )

    # --- ParsBERT polarity classifier --------------------------------------
    parsbert = AutoModelForSequenceClassification.from_pretrained(parsbert_id)
    _freeze(parsbert)
    parsbert.to(dev)
    parsbert_tok = AutoTokenizer.from_pretrained(parsbert_id)

    frozen = [smolvlm, mclip_text, mclip_image, parsbert]
    if mclip_attn is not None:
        frozen.append(mclip_attn)
    assert_frozen(*frozen)
    return BackboneBundle(
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
                {"type": "text", "text": "Describe this image objectively and concisely."},
            ],
        }
    ]
    prompt = proc.apply_chat_template(messages, add_generation_prompt=True)

    if _is_smolvlm_pipeline(model):
        with torch.no_grad():
            outputs = model(
                text=prompt,
                images=[image],
                max_new_tokens=48,
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
            out = gen_model.generate(**inputs, max_new_tokens=48, do_sample=False)
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


def polarity_probs(bundle: BackboneBundle, text: str) -> np.ndarray:
    """Run the ParsBERT polarity classifier on `text` (Persian)."""
    import torch
    from torch.nn.functional import softmax

    inputs = bundle.parsbert_tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    ).to(bundle.device)
    with torch.no_grad():
        logits = bundle.parsbert_polarity(**inputs).logits
    probs = softmax(logits, dim=-1).cpu().numpy()[0]
    probs = np.nan_to_num(np.asarray(probs, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    total = float(probs.sum())
    if total > 0.0:
        probs = probs / total
    elif probs.size >= 2:
        probs = np.array([0.5, 0.5], dtype=np.float32)
    return probs
