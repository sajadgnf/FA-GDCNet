"""Generative Discrepancy Representation Module (GDRM).

Computes the three discrepancy signals defined in the spec and packs them into
a feature vector for the downstream sklearn classifier. All maths is pure
numpy so the module can be exhaustively unit-tested without torch.

Signals (per spec `multimodal-sentiment`):

- `Dsem = 1 - cos(mCLIP_text(T), mCLIP_text(T_hat))`
- `Dsen = |polarity(T) - polarity(T_hat)|`     (L1 over softmax distributions)
- `Fvt  = cos(mCLIP_image(I), mCLIP_text(T_hat))`

Auxiliary features added to the vector:
- `cos(mCLIP_text(T), mCLIP_image(I))`
- positive-polarity scalar of `T` and of `T_hat`

The polarity classifier is assumed to return probabilities for two classes
(`negative`, `positive`). We define the single-scalar polarity as
`p[positive] - p[negative]` so it sits in `[-1, +1]`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Default Fvt threshold below which the response is flagged `low_fidelity=true`.
DEFAULT_FVT_THRESHOLD: float = 0.2


def _as_vec(x) -> np.ndarray:
    arr = np.nan_to_num(
        np.asarray(x, dtype=np.float64).reshape(-1),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if arr.size == 0:
        raise ValueError("expected a non-empty vector")
    return arr


def _safe_norm(v: np.ndarray) -> float:
    n = float(np.linalg.norm(v))
    return n if np.isfinite(n) else 0.0


def cosine_similarity(a, b) -> float:
    """Return cosine similarity in [-1, 1]; returns 0.0 if either norm is zero."""
    va = _as_vec(a)
    vb = _as_vec(b)
    if va.shape != vb.shape:
        raise ValueError(f"cosine_similarity: shape mismatch {va.shape} vs {vb.shape}")
    na = _safe_norm(va)
    nb = _safe_norm(vb)
    if na == 0.0 or nb == 0.0:
        return 0.0
    sim = float(np.dot(va, vb) / (na * nb))
    return float(sim) if np.isfinite(sim) else 0.0


def cosine_distance(a, b) -> float:
    """Return 1 - cosine similarity; clamped to `[0, 2]`."""
    return float(np.clip(1.0 - cosine_similarity(a, b), 0.0, 2.0))


def polarity_scalar(probs) -> float:
    """Convert a `(p_neg, p_pos)` distribution to a scalar in `[-1, +1]`.

    Accepts either a 2-vector or a longer distribution; in the latter case the
    first two entries are taken as (neg, pos) which matches typical ParsBERT
    sentiment heads.
    """
    p = _as_vec(probs)
    if p.size < 2:
        raise ValueError("polarity vector must have at least 2 entries (neg, pos)")
    val = float(p[1] - p[0])
    return val if np.isfinite(val) else 0.0


def polarity_l1(probs_a, probs_b) -> float:
    """L1 distance between two polarity distributions (same length)."""
    a = _as_vec(probs_a)
    b = _as_vec(probs_b)
    if a.shape != b.shape:
        raise ValueError(f"polarity_l1: shape mismatch {a.shape} vs {b.shape}")
    return float(np.sum(np.abs(a - b)))


# ------------ Spec-named helpers (Dsem, Dsen, Fvt) ---------------------------


def compute_dsem(text_emb_T, text_emb_T_hat) -> float:
    """`Dsem = 1 - cos(mCLIP_text(T), mCLIP_text(T_hat))`."""
    return cosine_distance(text_emb_T, text_emb_T_hat)


def compute_dsen(polarity_T, polarity_T_hat) -> float:
    """`Dsen = L1(polarity(T), polarity(T_hat))`."""
    return polarity_l1(polarity_T, polarity_T_hat)


def compute_fvt(image_emb_I, text_emb_T_hat) -> float:
    """`Fvt = cos(mCLIP_image(I), mCLIP_text(T_hat))`."""
    return cosine_similarity(image_emb_I, text_emb_T_hat)


# ------------ Feature vector --------------------------------------------------

# Canonical feature order; downstream classifier columns must match.
FEATURE_NAMES: tuple[str, ...] = (
    "Dsem",
    "Dsen",
    "Fvt",
    "cos_TI",
    "polarity_T",
    "polarity_T_hat",
)


@dataclass
class DiscrepancyFeatures:
    """Six-dimensional feature vector consumed by the sklearn classifier."""

    Dsem: float
    Dsen: float
    Fvt: float
    cos_TI: float
    polarity_T: float
    polarity_T_hat: float

    def as_array(self) -> np.ndarray:
        arr = np.array(
            [self.Dsem, self.Dsen, self.Fvt, self.cos_TI, self.polarity_T, self.polarity_T_hat],
            dtype=np.float32,
        )
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    def as_dict(self) -> dict[str, float]:
        return {
            "Dsem": self.Dsem,
            "Dsen": self.Dsen,
            "Fvt": self.Fvt,
            "cos_TI": self.cos_TI,
            "polarity_T": self.polarity_T,
            "polarity_T_hat": self.polarity_T_hat,
        }


def build_feature_vector(
    *,
    text_emb_T,
    text_emb_T_hat,
    image_emb_I,
    polarity_probs_T,
    polarity_probs_T_hat,
) -> DiscrepancyFeatures:
    """Bundle of the spec-required features for one sample.

    All inputs are NumPy-array-like. mCLIP embeddings can have any (matching)
    dimensionality. Polarity probabilities are 2-vectors `(p_neg, p_pos)`.
    """
    return DiscrepancyFeatures(
        Dsem=compute_dsem(text_emb_T, text_emb_T_hat),
        Dsen=compute_dsen(polarity_probs_T, polarity_probs_T_hat),
        Fvt=compute_fvt(image_emb_I, text_emb_T_hat),
        cos_TI=cosine_similarity(text_emb_T, image_emb_I),
        polarity_T=polarity_scalar(polarity_probs_T),
        polarity_T_hat=polarity_scalar(polarity_probs_T_hat),
    )
