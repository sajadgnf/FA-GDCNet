"""Tests for the pure-numpy Attention Rollout core."""

from __future__ import annotations

import numpy as np
import pytest

from explain.rollout import rollout, token_scores_from_rollout


def _row_stochastic(rng: np.random.Generator, heads: int, seq: int) -> np.ndarray:
    raw = rng.random((heads, seq, seq))
    raw /= raw.sum(axis=-1, keepdims=True)
    return raw


def test_rollout_single_layer_with_residual_is_row_stochastic():
    rng = np.random.default_rng(0)
    layer = _row_stochastic(rng, heads=2, seq=4)
    out = rollout([layer], add_residual=True)
    row_sums = out.sum(axis=-1)
    assert np.allclose(row_sums, 1.0)


def test_rollout_multi_layer_is_row_stochastic():
    rng = np.random.default_rng(1)
    layers = [_row_stochastic(rng, heads=3, seq=5) for _ in range(4)]
    out = rollout(layers, add_residual=True)
    assert out.shape == (5, 5)
    assert np.allclose(out.sum(axis=-1), 1.0, atol=1e-6)


def test_rollout_no_layers_raises():
    with pytest.raises(ValueError):
        rollout([])


def test_rollout_unknown_fusion_raises():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        rollout([_row_stochastic(rng, 1, 3)], head_fusion="median")


def test_rollout_identity_is_no_op():
    seq = 4
    identity = np.eye(seq)[None, :, :]  # 1 head, seq×seq identity
    out = rollout([identity], add_residual=False)
    assert np.allclose(out, np.eye(seq))


def test_token_scores_from_rollout_picks_cls_row():
    matrix = np.array([[0.1, 0.2, 0.7], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]])
    scores = token_scores_from_rollout(matrix, cls_index=0)
    assert np.allclose(scores, matrix[0])


def test_token_scores_from_rollout_validates_index():
    matrix = np.eye(3)
    with pytest.raises(ValueError):
        token_scores_from_rollout(matrix, cls_index=5)
