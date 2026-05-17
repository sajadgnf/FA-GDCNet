"""Tests for Cohen's kappa and the IAA reporter."""

from __future__ import annotations

import pytest

from data.iaa import cohens_kappa, load_annotations, pairwise_kappa, render_iaa_report


def test_kappa_perfect_agreement():
    a = ["positive", "negative", "neutral"]
    b = ["positive", "negative", "neutral"]
    assert cohens_kappa(a, b) == pytest.approx(1.0)


def test_kappa_random_two_class():
    # Two raters each emit 50/50 of two labels and agree on half.
    a = ["positive", "positive", "negative", "negative"]
    b = ["positive", "negative", "positive", "negative"]
    # observed=0.5, expected=0.5  -> kappa=0
    assert cohens_kappa(a, b) == pytest.approx(0.0)


def test_kappa_degenerate_single_class():
    a = ["positive"] * 5
    b = ["positive"] * 5
    # Expected agreement is 1.0 → return 0.0 by convention instead of NaN.
    assert cohens_kappa(a, b) == 0.0


def test_kappa_length_mismatch_raises():
    with pytest.raises(ValueError):
        cohens_kappa(["positive"], ["positive", "negative"])


def test_kappa_empty_returns_zero():
    assert cohens_kappa([], []) == 0.0


def test_pairwise_kappa_only_pairs_with_overlap():
    annotator_labels = {
        "alice": {"p1": "positive", "p2": "negative", "p3": "neutral"},
        "bob": {"p1": "positive", "p2": "negative"},
        "carol": {"p4": "positive"},  # no overlap with alice / bob
    }
    pairs = pairwise_kappa(annotator_labels)
    # alice vs bob overlap on p1, p2 with perfect agreement.
    assert pairs[("alice", "bob")] == pytest.approx(1.0)
    # alice/carol and bob/carol have no overlap → not in result.
    assert ("alice", "carol") not in pairs
    assert ("bob", "carol") not in pairs


def test_load_annotations_groups_by_annotator():
    rows = [
        {"annotator_id": "alice", "post_id": "p1", "label": "positive"},
        {"annotator_id": "alice", "post_id": "p2", "label": "neutral"},
        {"annotator_id": "bob", "post_id": "p1", "label": "negative"},
        {"annotator_id": "bob", "post_id": "p2", "label": "bogus"},  # filtered out
    ]
    out = load_annotations(rows)
    assert out["alice"] == {"p1": "positive", "p2": "neutral"}
    assert out["bob"] == {"p1": "negative"}


def test_render_iaa_report_handles_no_overlap():
    out = render_iaa_report({}, annotator_labels={"alice": {"p1": "positive"}})
    assert "_No overlapping samples" in out
    assert "alice" in out


def test_render_iaa_report_lists_pair_kappas():
    annotator_labels = {
        "alice": {"p1": "positive", "p2": "negative"},
        "bob": {"p1": "positive", "p2": "negative"},
    }
    pairs = pairwise_kappa(annotator_labels)
    body = render_iaa_report(pairs, annotator_labels=annotator_labels)
    assert "alice" in body and "bob" in body
    assert "Kappa" in body
