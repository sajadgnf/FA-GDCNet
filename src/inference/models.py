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

DEFAULT_SMOLVLM_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
DEFAULT_MCLIP_ID = "M-CLIP/M-BERT-Distil-40"
DEFAULT_PARSBERT_POLARITY_ID = "HooshvareLab/bert-fa-base-uncased-sentiment-snappfood"


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


def _freeze(module: Any) -> Any:
    """Put `module` in eval mode and disable grad on every parameter."""
    if hasattr(module, "eval"):
        module.eval()
    if hasattr(module, "parameters"):
        for p in module.parameters():
            p.requires_grad_(False)
    return module


def load_backbones(
    *,
    device: str | None = None,
    smolvlm_id: str = DEFAULT_SMOLVLM_ID,
    mclip_id: str = DEFAULT_MCLIP_ID,
    parsbert_id: str = DEFAULT_PARSBERT_POLARITY_ID,
) -> BackboneBundle:
    """Load every backbone, freeze it, and return a `BackboneBundle`."""
    import torch
    from multilingual_clip import pt_multilingual_clip
    from transformers import (
        AutoImageProcessor,
        AutoModel,
        AutoModelForSequenceClassification,
        AutoProcessor,
        AutoTokenizer,
    )

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # --- SmolVLM-256M (captioner) ------------------------------------------
    smolvlm_proc = AutoProcessor.from_pretrained(smolvlm_id)
    smolvlm = AutoModel.from_pretrained(smolvlm_id, torch_dtype=torch.float32)
    _freeze(smolvlm)
    smolvlm.to(dev)

    # --- M-CLIP (text + image) --------------------------------------------
    mclip_text = pt_multilingual_clip.MultilingualCLIP.from_pretrained(mclip_id)
    _freeze(mclip_text)
    mclip_text.to(dev)
    mclip_tokenizer = AutoTokenizer.from_pretrained(mclip_id)

    # The image side of M-CLIP-ViT-B-32 is the OpenAI CLIP ViT-B/32 vision tower.
    mclip_image = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
    _freeze(mclip_image)
    mclip_image.to(dev)
    mclip_image_processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # --- ParsBERT polarity classifier --------------------------------------
    parsbert = AutoModelForSequenceClassification.from_pretrained(parsbert_id)
    _freeze(parsbert)
    parsbert.to(dev)
    parsbert_tok = AutoTokenizer.from_pretrained(parsbert_id)

    assert_frozen(smolvlm, mclip_text, mclip_image, parsbert)
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
    )


# --------------- Inference helpers (used by the pipeline) ---------------------


def caption_image(bundle: BackboneBundle, image) -> str:
    """Generate the objective description `T_hat` for an image using SmolVLM."""
    import torch

    proc = bundle.smolvlm_processor
    model = bundle.smolvlm_model
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
    inputs = proc(text=prompt, images=[image], return_tensors="pt").to(bundle.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=48, do_sample=False)
    text = proc.batch_decode(out, skip_special_tokens=True)[0]
    # SmolVLM echoes the prompt; strip it.
    if "Assistant:" in text:
        text = text.split("Assistant:")[-1].strip()
    return text.strip()


def embed_text_mclip(bundle: BackboneBundle, text: str) -> np.ndarray:
    """Embed `text` (any language M-CLIP supports) into the shared space."""
    import torch

    with torch.no_grad():
        emb = bundle.mclip_text.forward([text], bundle.mclip_tokenizer).cpu().numpy()[0]
    return np.asarray(emb, dtype=np.float32)


def embed_image_mclip(bundle: BackboneBundle, image) -> np.ndarray:
    """Embed an image into the M-CLIP shared space via the CLIP vision tower."""
    import torch

    inputs = bundle.mclip_image_processor(images=image, return_tensors="pt").to(bundle.device)
    with torch.no_grad():
        out = bundle.mclip_image.get_image_features(**inputs).cpu().numpy()[0]
    return np.asarray(out, dtype=np.float32)


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
    return np.asarray(probs, dtype=np.float32)
