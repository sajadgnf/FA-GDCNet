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
- polarity scalar of `T` (text sentiment head) and of the image
  (`polarity_T_hat`: CLIP smile-vs-sad, not SmolVLM-caption polarity)
- `clash = -polarity_T * polarity_T_hat` (positive when text and face oppose;
  a linear head cannot learn that product from the two scalars alone)

`T̂` still feeds `Dsem` and `Fvt`. SmolVLM-256M captions under the 1 GiB budget
are often affect-free, so image polarity uses CLIP facial affect instead of
running the text head on `T̂`.

The polarity vectors are two-class (`negative`, `positive`). The scalar is
`p[positive] - p[negative]` in `[-1, +1]`.
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


def compute_clash(polarity_T: float, polarity_T_hat: float) -> float:
    """Opposite-sign polarities → positive clash; same-sign → negative.

    Linear classifiers see `polarity_T` and `polarity_T_hat` separately and
    cannot form this interaction unless it is an explicit column.
    """
    val = float(-float(polarity_T) * float(polarity_T_hat))
    return val if np.isfinite(val) else 0.0


def compute_fvt(image_emb_I, text_emb_T_hat) -> float:
    """`Fvt = cos(mCLIP_image(I), mCLIP_text(T_hat))`."""
    return cosine_similarity(image_emb_I, text_emb_T_hat)


# ------------ Feature vector --------------------------------------------------

# Canonical feature order; downstream classifier columns must match.
CORE_FEATURE_NAMES: tuple[str, ...] = (
    "Dsem",
    "Dsen",
    "Fvt",
    "cos_TI",
    "polarity_T",
    "polarity_T_hat",
)
FEATURE_NAMES: tuple[str, ...] = CORE_FEATURE_NAMES + ("clash",)
_N_CORE = len(CORE_FEATURE_NAMES)
_POLARITY_T_IDX = CORE_FEATURE_NAMES.index("polarity_T")
_POLARITY_HAT_IDX = CORE_FEATURE_NAMES.index("polarity_T_hat")


@dataclass
class DiscrepancyFeatures:
    """GDRM vector consumed by the sklearn classifier (core signals + clash)."""

    Dsem: float
    Dsen: float
    Fvt: float
    cos_TI: float
    polarity_T: float
    polarity_T_hat: float

    @property
    def clash(self) -> float:
        return compute_clash(self.polarity_T, self.polarity_T_hat)

    def as_array(self) -> np.ndarray:
        arr = np.array(
            [
                self.Dsem,
                self.Dsen,
                self.Fvt,
                self.cos_TI,
                self.polarity_T,
                self.polarity_T_hat,
                self.clash,
            ],
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
            "clash": self.clash,
        }


def features_from_array(x) -> DiscrepancyFeatures:
    """Rebuild features from a cache row (6 core columns, clash is derived)."""
    arr = np.nan_to_num(
        np.asarray(x, dtype=np.float64).reshape(-1),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if arr.size < _N_CORE:
        raise ValueError(f"expected at least {_N_CORE} feature columns, got {arr.size}")
    return DiscrepancyFeatures(
        Dsem=float(arr[0]),
        Dsen=float(arr[1]),
        Fvt=float(arr[2]),
        cos_TI=float(arr[3]),
        polarity_T=float(arr[4]),
        polarity_T_hat=float(arr[5]),
    )


def with_clash_column(X: np.ndarray) -> np.ndarray:
    """Append `clash` to a 6-column cache, or pass through a current matrix."""
    mat = np.asarray(X, dtype=np.float32)
    if mat.ndim != 2:
        raise ValueError(f"expected 2-d feature matrix, got shape {mat.shape}")
    n = len(FEATURE_NAMES)
    if mat.shape[1] == n:
        return mat
    if mat.shape[1] == _N_CORE:
        clash = -mat[:, _POLARITY_T_IDX] * mat[:, _POLARITY_HAT_IDX]
        clash = np.nan_to_num(clash, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return np.concatenate([mat, clash[:, None]], axis=1)
    raise ValueError(f"expected {_N_CORE} or {n} feature columns, got {mat.shape[1]}")


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
