"""Tests for `remap_rtl_indices` (pure stdlib)."""

from __future__ import annotations

import pytest

from explain.rtl import is_anchor, remap_rtl_indices, remap_rtl_pairs


def test_remap_reverses_interior_tokens():
    tokens = ["[CLS]", "سلام", "دنیا", "[SEP]"]
    scores = [0.0, 0.7, 0.2, 0.0]
    out_tokens, out_scores = remap_rtl_indices(tokens, scores)
    assert out_tokens == ["[CLS]", "دنیا", "سلام", "[SEP]"]
    assert out_scores == [0.0, 0.2, 0.7, 0.0]


def test_remap_preserves_score_values():
    """Spec scenario: scores are remapped, NEVER modified."""
    tokens = ["a", "b", "c"]
    scores = [0.11, 0.22, 0.33]
    _, out_scores = remap_rtl_indices(tokens, scores)
    assert sorted(out_scores) == pytest.approx(sorted(scores))


def test_remap_empty_input():
    assert remap_rtl_indices([], []) == ([], [])


def test_remap_no_anchors_full_reversal():
    tokens = ["a", "b", "c", "d"]
    scores = [1, 2, 3, 4]
    out_tokens, out_scores = remap_rtl_indices(tokens, scores)
    assert out_tokens == ["d", "c", "b", "a"]
    assert out_scores == [4, 3, 2, 1]


def test_remap_all_anchors_is_no_op():
    tokens = ["[CLS]", "[SEP]"]
    scores = [0.5, 0.5]
    assert remap_rtl_indices(tokens, scores) == (tokens, scores)


def test_remap_length_mismatch_raises():
    with pytest.raises(ValueError):
        remap_rtl_indices(["a", "b"], [0.1])


def test_remap_pairs_helper():
    pairs = [("[CLS]", 0.0), ("salam", 0.6), ("donya", 0.4), ("[SEP]", 0.0)]
    out = remap_rtl_pairs(pairs)
    assert [t for t, _ in out] == ["[CLS]", "donya", "salam", "[SEP]"]


def test_is_anchor_covers_known_special_tokens():
    for tok in ["[CLS]", "[SEP]", "<s>", "</s>", "<pad>"]:
        assert is_anchor(tok) is True
    assert is_anchor("سلام") is False
