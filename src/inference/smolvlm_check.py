"""SmolVLM load-shape checks (no torch/transformers imports)."""

from __future__ import annotations

from typing import Any

GENERATION_SMOLVLM_TYPES = frozenset(
    {
        "Idefics3ForConditionalGeneration",
        "SmolVLMForConditionalGeneration",
    }
)


def is_smolvlm_pipeline(obj: Any) -> bool:
    return callable(obj) and hasattr(obj, "model") and "Pipeline" in type(obj).__name__


def smolvlm_generation_module(obj: Any) -> Any | None:
    """Return an object with `.generate`, never the Idefics3 vision backbone."""
    if type(obj).__name__ == "Idefics3Model":
        return None
    if callable(getattr(obj, "generate", None)):
        return obj
    return None


def smolvlm_can_caption(model: Any) -> bool:
    if is_smolvlm_pipeline(model):
        inner = getattr(model, "model", None)
        return inner is not None and type(inner).__name__ in GENERATION_SMOLVLM_TYPES
    return smolvlm_generation_module(model) is not None
