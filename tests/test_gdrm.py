"""Tests for the GDRM feature extraction (pure numpy)."""

from __future__ import annotations

import numpy as np
import pytest

from inference.gdrm import (
    DEFAULT_FVT_THRESHOLD,
    FEATURE_NAMES,
    build_feature_vector,
    compute_dsem,
    compute_dsen,
    compute_fvt,
    cosine_distance,
    cosine_similarity,
    polarity_l1,
    polarity_scalar,
)


def test_cosine_similarity_identical_vectors():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_opposite():
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    assert cosine_similarity(a, b) == pytest.approx(-1.0)


def test_cosine_similarity_zero_norm_returns_zero():
    a = np.zeros(4)
    b = np.array([1.0, 2.0, 3.0, 4.0])
    assert cosine_similarity(a, b) == 0.0


def test_cosine_similarity_all_nan_inputs_return_zero():
    a = np.array([np.nan, np.nan])
    b = np.array([1.0, 2.0])
    assert cosine_similarity(a, b) == 0.0


def test_cosine_similarity_shape_mismatch_raises():
    with pytest.raises(ValueError):
        cosine_similarity(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))


def test_cosine_distance_inverse_of_similarity():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert cosine_distance(a, b) == pytest.approx(1.0)


def test_polarity_l1_basic():
    a = np.array([0.7, 0.3])
    b = np.array([0.2, 0.8])
    assert polarity_l1(a, b) == pytest.approx(1.0)


def test_polarity_scalar_range():
    assert polarity_scalar(np.array([1.0, 0.0])) == pytest.approx(-1.0)
    assert polarity_scalar(np.array([0.0, 1.0])) == pytest.approx(1.0)
    assert polarity_scalar(np.array([0.5, 0.5])) == pytest.approx(0.0)


def test_compute_dsem_contradiction():
    """A clearly contradicting `T_hat` produces a large Dsem."""
    T_emb = np.array([1.0, 0.0, 0.0])
    T_hat_emb = np.array([0.0, 0.0, 1.0])
    assert compute_dsem(T_emb, T_hat_emb) == pytest.approx(1.0)


def test_compute_dsem_agreement():
    v = np.array([1.0, 2.0, 3.0])
    assert compute_dsem(v, v) == pytest.approx(0.0)


def test_compute_dsen_polarity_contradiction():
    pos_in_T = np.array([0.05, 0.95])
    neg_in_T_hat = np.array([0.95, 0.05])
    # L1 = |0.05-0.95| + |0.95-0.05| = 1.8
    assert compute_dsen(pos_in_T, neg_in_T_hat) == pytest.approx(1.8)


def test_compute_fvt_perfect_match():
    v = np.array([1.0, 0.0])
    assert compute_fvt(v, v) == pytest.approx(1.0)


def test_build_feature_vector_shape_and_names():
    T = np.array([1.0, 0.0, 0.0])
    Th = np.array([1.0, 0.0, 0.0])
    I = np.array([0.0, 1.0, 0.0])
    pT = np.array([0.2, 0.8])
    pTh = np.array([0.2, 0.8])

    f = build_feature_vector(
        text_emb_T=T,
        text_emb_T_hat=Th,
        image_emb_I=I,
        polarity_probs_T=pT,
        polarity_probs_T_hat=pTh,
    )
    arr = f.as_array()
    assert arr.shape == (6,)
    d = f.as_dict()
    assert tuple(d.keys()) == FEATURE_NAMES


def test_build_feature_vector_sarcasm_signature():
    """Sarcasm: positive Persian caption, but image content negative.

    Dsem high (caption embedding ≠ T_hat embedding), polarity scalar of T high,
    polarity scalar of T_hat low → Dsen also high. Fvt should still be high
    (T_hat does describe the image well).
    """
    T_emb = np.array([1.0, 0.0])
    T_hat_emb = np.array([-1.0, 0.0])  # opposite direction in shared space
    I_emb = np.array([-1.0, 0.0])
    p_T = np.array([0.05, 0.95])       # positive
    p_T_hat = np.array([0.92, 0.08])   # negative

    f = build_feature_vector(
        text_emb_T=T_emb,
        text_emb_T_hat=T_hat_emb,
        image_emb_I=I_emb,
        polarity_probs_T=p_T,
        polarity_probs_T_hat=p_T_hat,
    )
    assert f.Dsem > 0.9            # large semantic gap
    assert f.Dsen > 1.5            # large polarity flip
    assert f.Fvt > 0.9             # but description still describes the image
    assert f.polarity_T > 0.5
    assert f.polarity_T_hat < -0.5


def test_default_fvt_threshold_is_documented():
    assert 0.0 < DEFAULT_FVT_THRESHOLD < 1.0
