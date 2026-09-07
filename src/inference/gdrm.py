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
  a linear head cannot learn that product from the two scalars alone).
  Weak / near-neutral polarities are zeroed: clash is only defined when both
  sides are committed, matching `pipeline.refine_label_for_polarity_conflict`.

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


# Same floors as `pipeline.refine_label_for_polarity_conflict` (clash_text / hat).
CLASH_TEXT_FLOOR = 0.20
CLASH_IMAGE_FLOOR = 0.15


def compute_clash(polarity_T: float, polarity_T_hat: float) -> float:
    """Opposite-sign polarities → positive clash; same-sign → negative.

    Linear classifiers see `polarity_T` and `polarity_T_hat` separately and
    cannot form this interaction unless it is an explicit column. Near-neutral
    text or face scores are not a clash (product of two weak signs is noise).
    """
    pt = float(polarity_T)
    ph = float(polarity_T_hat)
    if abs(pt) < CLASH_TEXT_FLOOR or abs(ph) < CLASH_IMAGE_FLOOR:
        return 0.0
    val = float(-pt * ph)
    return val if np.isfinite(val) else 0.0


def clash_column(polarity_T, polarity_T_hat) -> np.ndarray:
    """Vectorized ``compute_clash`` for a feature matrix."""
    pt = np.asarray(polarity_T, dtype=np.float64).reshape(-1)
    ph = np.asarray(polarity_T_hat, dtype=np.float64).reshape(-1)
    raw = -pt * ph
    weak = (np.abs(pt) < CLASH_TEXT_FLOOR) | (np.abs(ph) < CLASH_IMAGE_FLOOR)
    out = np.where(weak, 0.0, raw)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


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
_DSEN_IDX = CORE_FEATURE_NAMES.index("Dsen")


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
    """Append or refresh `clash` from the polarity columns (magnitude-gated)."""
    mat = np.asarray(X, dtype=np.float32)
    if mat.ndim != 2:
        raise ValueError(f"expected 2-d feature matrix, got shape {mat.shape}")
    n = len(FEATURE_NAMES)
    if mat.shape[1] not in (_N_CORE, n):
        raise ValueError(f"expected {_N_CORE} or {n} feature columns, got {mat.shape[1]}")
    clash = clash_column(mat[:, _POLARITY_T_IDX], mat[:, _POLARITY_HAT_IDX])
    if mat.shape[1] == _N_CORE:
        return np.concatenate([mat, clash[:, None]], axis=1)
    out = mat.copy()
    out[:, -1] = clash
    return out


def caption_polarity_scale(caption: str) -> float:
    """Hashtag walls and prompt-spam are not a committed caption polarity."""
    from data.preprocess import caption_is_thin, is_spam_caption

    if caption_is_thin(caption) or is_spam_caption(caption):
        return 0.0
    return 1.0


def apply_caption_guards(X: np.ndarray, captions: list[str]) -> np.ndarray:
    """Zero text polarity (and refresh Dsen/clash) on thin or spam captions."""
    mat = with_clash_column(X).copy()
    if len(captions) != mat.shape[0]:
        raise ValueError(f"caption count {len(captions)} != rows {mat.shape[0]}")
    scales = np.asarray(
        [caption_polarity_scale(c) for c in captions], dtype=np.float32
    )
    mat[:, _POLARITY_T_IDX] *= scales
    pt = mat[:, _POLARITY_T_IDX]
    ph = mat[:, _POLARITY_HAT_IDX]
    mat[:, _DSEN_IDX] = np.abs(pt - ph)
    return with_clash_column(mat)


def features_for_eval(X: np.ndarray, ids: list[str], dataset) -> np.ndarray:
    """Eval-time GDRM matrix: clash refresh + thin/spam caption guards."""
    from pathlib import Path

    from data.schema import iter_dataset

    by = {r.post_id: (r.caption or "") for r in iter_dataset(Path(dataset))}
    caps = [by.get(str(i), "") for i in ids]
    return apply_caption_guards(X, caps)


def guard_discrepancy_features(
    features: DiscrepancyFeatures, caption: str
) -> DiscrepancyFeatures:
    """Same thin/spam guard for a live pipeline vector."""
    if caption_polarity_scale(caption) == 1.0:
        return features
    pt = 0.0
    return DiscrepancyFeatures(
        Dsem=features.Dsem,
        Dsen=abs(pt - float(features.polarity_T_hat)),
        Fvt=features.Fvt,
        cos_TI=features.cos_TI,
        polarity_T=pt,
        polarity_T_hat=features.polarity_T_hat,
    )


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
