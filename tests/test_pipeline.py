"""Tests for the inference Pipeline using injected fakes (no heavy deps)."""

from __future__ import annotations

import numpy as np
import pytest

from data.schema import LABELS
from inference.gdrm import build_feature_vector
from inference.pipeline import Pipeline


class _FakeBundle:
    """Stub that satisfies the duck-typed bundle interface used by tests."""


class _FakeSklearnModel:
    """sklearn-shaped fake with a `classes_` attribute and `predict_proba`."""

    def __init__(self, classes, fixed_proba):
        self.classes_ = np.asarray(classes, dtype=object)
        self._proba = np.asarray(fixed_proba, dtype=np.float32)

    def predict_proba(self, X):
        assert X.shape == (1, 6)
        return self._proba.reshape(1, -1)


def _make_clf_pack(*, target_label: str, confidence: float = 0.81) -> dict:
    """Build a clf_pack whose argmax lands on `target_label`."""
    proba = np.full(len(LABELS), (1.0 - confidence) / (len(LABELS) - 1), dtype=np.float32)
    proba[LABELS.index(target_label)] = confidence
    return {
        "model": _FakeSklearnModel(LABELS, proba),
        "classifier_name": "Fake",
        "feature_names": ["Dsem", "Dsen", "Fvt", "cos_TI", "polarity_T", "polarity_T_hat"],
        "label_order": list(LABELS),
    }


def _make_features(*, fvt: float = 0.7):
    return build_feature_vector(
        text_emb_T=np.array([1.0, 0.0]),
        text_emb_T_hat=np.array([1.0, 0.0]),
        image_emb_I=np.array([fvt, np.sqrt(max(0.0, 1.0 - fvt * fvt))]),
        polarity_probs_T=np.array([0.2, 0.8]),
        polarity_probs_T_hat=np.array([0.2, 0.8]),
    )


def test_predict_from_features_returns_argmax_label():
    clf_pack = _make_clf_pack(target_label="positive_sarcasm", confidence=0.81)
    pipeline = Pipeline(bundle=_FakeBundle(), clf_pack=clf_pack)
    pred = pipeline.predict_from_features(_make_features(fvt=0.7))
    assert pred.label == "positive_sarcasm"
    assert pred.confidence == pytest.approx(0.81)


def test_predict_from_features_low_fidelity_flag_set_when_fvt_below_tau():
    pack = _make_clf_pack(target_label="neutral")
    pipeline = Pipeline(bundle=_FakeBundle(), clf_pack=pack, fvt_threshold=0.3)
    low = pipeline.predict_from_features(_make_features(fvt=0.1))
    high = pipeline.predict_from_features(_make_features(fvt=0.8))
    assert low.low_fidelity is True
    assert high.low_fidelity is False


def test_predict_from_features_discrepancy_vector_keys():
    pack = _make_clf_pack(target_label="negative")
    pipeline = Pipeline(bundle=_FakeBundle(), clf_pack=pack)
    pred = pipeline.predict_from_features(_make_features())
    assert set(pred.discrepancy_vector) == {
        "Dsem", "Dsen", "Fvt", "cos_TI", "polarity_T", "polarity_T_hat"
    }


def test_predict_from_features_handles_classifier_with_subset_of_labels():
    """If the sklearn model never saw a class in training, its slot stays at 0."""
    proba = np.array([0.4, 0.6], dtype=np.float32)  # only 2 classes trained
    pack = {
        "model": _FakeSklearnModel(["positive", "negative"], proba),
        "classifier_name": "Fake",
        "feature_names": ["Dsem", "Dsen", "Fvt", "cos_TI", "polarity_T", "polarity_T_hat"],
        "label_order": list(LABELS),
    }
    pipeline = Pipeline(bundle=_FakeBundle(), clf_pack=pack)
    pred = pipeline.predict_from_features(_make_features())
    # argmax over the 5-vector should land on "negative".
    assert pred.label == "negative"


def test_prediction_as_dict_is_json_safe():
    pack = _make_clf_pack(target_label="positive")
    pipeline = Pipeline(bundle=_FakeBundle(), clf_pack=pack)
    pred = pipeline.predict_from_features(_make_features())
    import json
    body = json.dumps(pred.as_dict())
    assert "positive" in body
