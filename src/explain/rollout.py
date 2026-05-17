"""Attention Rollout (Abnar & Zuidema, 2020) applied to ParsBERT / mCLIP.

Pure-numpy implementation of the layerwise multiplied attention matrices with
an identity-residual addition. The actual transformer call to gather per-layer
attention maps lives in `attention_from_text` (which is torch-dependent).

The rollout function itself is pure-numpy so it can be unit-tested directly.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def rollout(
    attention_per_layer: Iterable[np.ndarray],
    *,
    head_fusion: str = "mean",
    add_residual: bool = True,
) -> np.ndarray:
    """Compute the rolled-out attention map across transformer layers.

    Args:
        attention_per_layer: iterable of `(num_heads, seq_len, seq_len)` arrays
            in layer order. Heads are fused per `head_fusion`.
        head_fusion: one of `mean`, `max`, or `min`.
        add_residual: if True (default), add the identity to each layer's
            fused matrix and re-normalize rows so the rollout accounts for the
            residual stream.

    Returns:
        The final `(seq_len, seq_len)` rolled-out attention matrix.

    Raises:
        ValueError if no layers are supplied or `head_fusion` is unknown.
    """
    fuse = {
        "mean": lambda a: a.mean(axis=0),
        "max": lambda a: a.max(axis=0),
        "min": lambda a: a.min(axis=0),
    }
    if head_fusion not in fuse:
        raise ValueError(f"unknown head_fusion {head_fusion!r}; expected mean/max/min")

    layers = list(attention_per_layer)
    if not layers:
        raise ValueError("rollout: at least one attention layer is required")

    seq_len = layers[0].shape[-1]
    result: np.ndarray | None = None
    for layer in layers:
        fused = fuse[head_fusion](np.asarray(layer, dtype=np.float64))
        if add_residual:
            fused = fused + np.eye(seq_len, dtype=np.float64)
            row_sums = fused.sum(axis=-1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1.0, row_sums)
            fused = fused / row_sums
        result = fused if result is None else fused @ result
    assert result is not None
    return result


def token_scores_from_rollout(rollout_matrix: np.ndarray, *, cls_index: int = 0) -> np.ndarray:
    """Per-token attribution: the CLS row of the rollout matrix."""
    if rollout_matrix.ndim != 2:
        raise ValueError("rollout_matrix must be 2-D")
    if not 0 <= cls_index < rollout_matrix.shape[0]:
        raise ValueError(f"cls_index {cls_index} out of bounds for shape {rollout_matrix.shape}")
    return rollout_matrix[cls_index]


# ---------- Torch-dependent helpers (used by the dashboard) -------------------


def attention_from_text(bundle, text: str) -> tuple[list[str], np.ndarray]:
    """Run ParsBERT on `text` and return (tokens, rollout_token_scores)."""
    import torch  # lazy

    tok = bundle.parsbert_tokenizer
    enc = tok(text, return_tensors="pt", truncation=True, max_length=256).to(bundle.device)
    with torch.no_grad():
        out = bundle.parsbert_polarity(
            **enc,
            output_attentions=True,
        )
    # `out.attentions` is a tuple of (1, heads, seq, seq) tensors.
    layers = [a.squeeze(0).cpu().numpy() for a in out.attentions]
    matrix = rollout(layers)
    scores = token_scores_from_rollout(matrix, cls_index=0)
    tokens = tok.convert_ids_to_tokens(enc["input_ids"][0].cpu().tolist())
    return tokens, scores


def attention_from_image(bundle, image) -> np.ndarray:
    """Run the CLIP vision tower on `image` and return the rolled-out patch map.

    Returns a 2-D `(grid, grid)` numpy array suitable for `render_image.overlay`.
    """
    import torch  # lazy

    inputs = bundle.mclip_image_processor(images=image, return_tensors="pt").to(bundle.device)
    with torch.no_grad():
        out = bundle.mclip_image(**inputs, output_attentions=True)
    layers = [a.squeeze(0).cpu().numpy() for a in out.attentions]
    matrix = rollout(layers)
    # CLS row over patch tokens (drop the CLS column).
    patch_scores = matrix[0, 1:]
    grid = int(round(np.sqrt(patch_scores.size)))
    if grid * grid != patch_scores.size:
        # Pad to the next square if the encoder used a non-square patch count.
        side = int(np.ceil(np.sqrt(patch_scores.size)))
        padded = np.zeros(side * side, dtype=patch_scores.dtype)
        padded[: patch_scores.size] = patch_scores
        return padded.reshape(side, side)
    return patch_scores.reshape(grid, grid)
