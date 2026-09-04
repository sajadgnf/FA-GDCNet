"""Tests for the inference Pipeline using injected fakes (no heavy deps)."""

from __future__ import annotations

import numpy as np
import pytest

from data.schema import LABELS
from inference.gdrm import FEATURE_NAMES, build_feature_vector
from inference.pipeline import Pipeline


class _FakeBundle:
    """Stub that satisfies the duck-typed bundle interface used by tests."""


class _FakeSklearnModel:
    """sklearn-shaped fake with a `classes_` attribute and `predict_proba`."""

    def __init__(self, classes, fixed_proba):
        self.classes_ = np.asarray(classes, dtype=object)
        self._proba = np.asarray(fixed_proba, dtype=np.float32)

    def predict_proba(self, X):
        assert X.shape == (1, len(FEATURE_NAMES))
        return self._proba.reshape(1, -1)


def _make_clf_pack(*, target_label: str, confidence: float = 0.81) -> dict:
    """Build a clf_pack whose argmax lands on `target_label`."""
    proba = np.full(len(LABELS), (1.0 - confidence) / (len(LABELS) - 1), dtype=np.float32)
    proba[LABELS.index(target_label)] = confidence
    return {
        "model": _FakeSklearnModel(LABELS, proba),
        "classifier_name": "Fake",
        "feature_names": list(FEATURE_NAMES),
        "label_order": list(LABELS),
    }


def _make_features(
    *,
    fvt: float = 0.7,
    polarity_probs_T: np.ndarray | None = None,
    polarity_probs_T_hat: np.ndarray | None = None,
):
    return build_feature_vector(
        text_emb_T=np.array([1.0, 0.0]),
        text_emb_T_hat=np.array([1.0, 0.0]),
        image_emb_I=np.array([fvt, np.sqrt(max(0.0, 1.0 - fvt * fvt))]),
        polarity_probs_T=polarity_probs_T if polarity_probs_T is not None else np.array([0.2, 0.8]),
        polarity_probs_T_hat=(
            polarity_probs_T_hat if polarity_probs_T_hat is not None else np.array([0.2, 0.8])
        ),
    )


def test_predict_from_features_returns_argmax_label():
    clf_pack = _make_clf_pack(target_label="positive_sarcasm", confidence=0.81)
    pipeline = Pipeline(bundle=_FakeBundle(), clf_pack=clf_pack)
    # Negative caption vs positive description → positive_sarcasm.
    pred = pipeline.predict_from_features(
        _make_features(
            fvt=0.7,
            polarity_probs_T=np.array([0.85, 0.15]),
            polarity_probs_T_hat=np.array([0.15, 0.85]),
        )
    )
    assert pred.label == "positive_sarcasm"
    assert pred.confidence == pytest.approx(0.81)


def test_polarity_conflict_promotes_positive_sarcasm():
    """Negative caption + positive T̂ → positive_sarcasm (happy face, sad words)."""
    from inference.pipeline import refine_label_for_polarity_conflict
    from inference.gdrm import DiscrepancyFeatures

    feats = DiscrepancyFeatures(
        Dsem=0.4,
        Dsen=0.57,
        Fvt=0.3,
        cos_TI=0.2,
        polarity_T=-0.55,
        polarity_T_hat=0.50,
    )
    proba = np.full(len(LABELS), 0.05, dtype=np.float32)
    proba[LABELS.index("negative")] = 0.24
    label, conf = refine_label_for_polarity_conflict("negative", 0.24, proba, feats)
    assert label == "positive_sarcasm"
    assert conf >= 0.35


def test_polarity_conflict_promotes_negative_sarcasm():
    """Positive caption + negative T̂ → negative_sarcasm."""
    from inference.pipeline import refine_label_for_polarity_conflict
    from inference.gdrm import DiscrepancyFeatures

    feats = DiscrepancyFeatures(
        Dsem=0.5,
        Dsen=0.8,
        Fvt=0.4,
        cos_TI=0.2,
        polarity_T=0.7,
        polarity_T_hat=-0.5,
    )
    proba = np.full(len(LABELS), 0.05, dtype=np.float32)
    proba[LABELS.index("positive")] = 0.4
    label, _ = refine_label_for_polarity_conflict("positive", 0.4, proba, feats)
    assert label == "negative_sarcasm"


def test_clear_conflict_from_neutral_gets_stronger_confidence():
    """Laughing face + «ناراحت است» must not stay low-confidence neutral."""
    from inference.pipeline import refine_label_for_polarity_conflict
    from inference.gdrm import DiscrepancyFeatures

    feats = DiscrepancyFeatures(
        Dsem=0.31,
        Dsen=0.55,
        Fvt=0.3,
        cos_TI=0.19,
        polarity_T=-0.55,
        polarity_T_hat=0.50,
    )
    proba = np.full(len(LABELS), 0.05, dtype=np.float32)
    proba[LABELS.index("neutral")] = 0.258
    label, conf = refine_label_for_polarity_conflict("neutral", 0.258, proba, feats)
    assert label == "positive_sarcasm"
    assert conf >= 0.62


def test_polarity_agreement_lifts_neutral_to_positive():
    """Happy text + smiling description must not stay ~26% neutral."""
    from inference.pipeline import refine_label_for_polarity_conflict
    from inference.gdrm import DiscrepancyFeatures

    feats = DiscrepancyFeatures(
        Dsem=0.29,
        Dsen=0.22,
        Fvt=0.30,
        cos_TI=0.21,
        polarity_T=0.7153,
        polarity_T_hat=0.4975,
    )
    proba = np.full(len(LABELS), 0.05, dtype=np.float32)
    proba[LABELS.index("neutral")] = 0.259
    label, conf = refine_label_for_polarity_conflict("neutral", 0.259, proba, feats)
    assert label == "positive"
    assert conf >= 0.65


def test_polarity_agreement_ignores_strength_mismatch_dsen():
    """«خوشبخت» + smile: both positive; high Dsen from strength only."""
    from inference.pipeline import refine_label_for_polarity_conflict
    from inference.gdrm import DiscrepancyFeatures

    feats = DiscrepancyFeatures(
        Dsem=0.28,
        Dsen=0.4961,
        Fvt=0.30,
        cos_TI=0.22,
        polarity_T=0.9935,
        polarity_T_hat=0.4975,
    )
    proba = np.full(len(LABELS), 0.05, dtype=np.float32)
    proba[LABELS.index("neutral")] = 0.271
    label, conf = refine_label_for_polarity_conflict("neutral", 0.271, proba, feats)
    assert label == "positive"
    assert conf >= 0.65


def test_sad_text_bland_caption_demotes_false_sarcasm():
    """«من خوردم زمین» + somber photo: VLM near-neutral ≠ positive sarcasm."""
    from inference.pipeline import refine_label_for_polarity_conflict
    from inference.gdrm import DiscrepancyFeatures

    feats = DiscrepancyFeatures(
        Dsem=0.41,
        Dsen=0.96,
        Fvt=0.26,
        cos_TI=0.21,
        polarity_T=-0.9606,
        polarity_T_hat=0.0028,
    )
    proba = np.full(len(LABELS), 0.05, dtype=np.float32)
    proba[LABELS.index("positive_sarcasm")] = 0.389
    label, conf = refine_label_for_polarity_conflict(
        "positive_sarcasm", 0.389, proba, feats
    )
    assert label == "negative"
    assert conf >= 0.65


def test_weak_question_plus_smile_is_not_sarcasm():
    """«سر صبح زنگ میزنه؟» is only slightly negative because of «؟»."""
    from inference.pipeline import refine_label_for_polarity_conflict
    from inference.gdrm import DiscrepancyFeatures

    feats = DiscrepancyFeatures(
        Dsem=0.18,
        Dsen=1.10,
        Fvt=0.27,
        cos_TI=0.20,
        polarity_T=-0.1017,
        polarity_T_hat=0.9977,
    )
    proba = np.full(len(LABELS), 0.05, dtype=np.float32)
    proba[LABELS.index("positive_sarcasm")] = 0.693
    label, _conf = refine_label_for_polarity_conflict(
        "positive_sarcasm", 0.693, proba, feats
    )
    assert label == "positive"


def test_mock_laugh_plus_smile_is_positive_sarcasm():
    from inference.pipeline import refine_label_for_polarity_conflict
    from inference.gdrm import DiscrepancyFeatures

    feats = DiscrepancyFeatures(
        Dsem=0.15,
        Dsen=1.6,
        Fvt=0.27,
        cos_TI=0.22,
        polarity_T=-0.75,
        polarity_T_hat=0.9977,
    )
    proba = np.full(len(LABELS), 0.05, dtype=np.float32)
    proba[LABELS.index("positive")] = 0.55
    label, _conf = refine_label_for_polarity_conflict("positive", 0.55, proba, feats)
    assert label == "positive_sarcasm"


def test_mild_positive_text_plus_smile_is_not_sarcasm():
    """Dsen from intensity (0.23 vs 0.99) is not a polarity clash."""
    from inference.pipeline import refine_label_for_polarity_conflict
    from inference.gdrm import DiscrepancyFeatures

    feats = DiscrepancyFeatures(
        Dsem=0.25,
        Dsen=0.76,
        Fvt=0.27,
        cos_TI=0.18,
        polarity_T=0.2333,
        polarity_T_hat=0.9977,
    )
    proba = np.full(len(LABELS), 0.05, dtype=np.float32)
    proba[LABELS.index("positive_sarcasm")] = 0.499
    label, _conf = refine_label_for_polarity_conflict(
        "positive_sarcasm", 0.499, proba, feats
    )
    assert label == "positive"


def test_mourning_caption_smiling_photo_is_positive_sarcasm():
    """«شادروان» (forced negative) + laughing portrait → clash, not plain positive."""
    from inference.pipeline import refine_label_for_polarity_conflict
    from inference.gdrm import DiscrepancyFeatures

    feats = DiscrepancyFeatures(
        Dsem=0.15,
        Dsen=1.75,
        Fvt=0.27,
        cos_TI=0.22,
        polarity_T=-0.75,
        polarity_T_hat=0.997,
    )
    proba = np.full(len(LABELS), 0.05, dtype=np.float32)
    proba[LABELS.index("positive")] = 0.55
    label, conf = refine_label_for_polarity_conflict("positive", 0.55, proba, feats)
    assert label == "positive_sarcasm"
    assert conf >= 0.50


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
    assert set(pred.discrepancy_vector) == set(FEATURE_NAMES)


def test_predict_from_features_handles_classifier_with_subset_of_labels():
    """If the sklearn model never saw a class in training, its slot stays at 0."""
    proba = np.array([0.4, 0.6], dtype=np.float32)  # only 2 classes trained
    pack = {
        "model": _FakeSklearnModel(["positive", "negative"], proba),
        "classifier_name": "Fake",
        "feature_names": list(FEATURE_NAMES),
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
