"""End-to-end FA-GDCNet inference: `predict(text, image)`.

This is the spec-named entry point. It accepts a `(text, image)` pair and
returns a `Prediction` containing:

- `label`: one of `LABELS`
- `confidence`: probability of the chosen label in [0, 1]
- `discrepancy_vector`: dict view of the 6-feature GDRM output
- `low_fidelity`: True iff `Fvt < tau` (spec hallucination guard)

The backbones are loaded once and cached on the `Pipeline` instance so a
long-running dashboard or evaluation loop doesn't re-load weights per sample.

The module supports two construction modes:
- `Pipeline.from_pretrained(...)` — loads real backbones + classifier.
- `Pipeline(bundle=..., clf_pack=...)` — direct injection, used by tests.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from data.schema import LABELS

from .classifier import DEFAULT_CLF, load as load_clf, predict_proba
from .gdrm import DEFAULT_FVT_THRESHOLD, DiscrepancyFeatures, build_feature_vector

log = logging.getLogger(__name__)

_SARCASM = frozenset({"positive_sarcasm", "negative_sarcasm"})


def refine_label_for_polarity_conflict(
    label: str,
    confidence: float,
    proba: np.ndarray,
    features: DiscrepancyFeatures,
) -> tuple[str, float]:
    """Map plain sentiment → sarcasm subtype when T and T̂ polarities conflict.

    Project taxonomy (see ``scripts/proposal_demo.py`` feature templates):
    - ``positive_sarcasm``: positive caption vs negative description
    - ``negative_sarcasm``: negative caption vs positive description

    So «ناراحتی» + smiling face → ``negative_sarcasm``, not ``positive_sarcasm``.
    """
    p_t = float(features.polarity_T)
    p_th = float(features.polarity_T_hat)
    dsen = float(features.Dsen)
    text_neg, text_pos = p_t <= -0.05, p_t >= 0.05
    hat_neg, hat_pos = p_th <= -0.15, p_th >= 0.15
    if text_neg and hat_pos and dsen >= 0.35:
        target = "negative_sarcasm"
    elif text_pos and hat_neg and dsen >= 0.35:
        target = "positive_sarcasm"
    else:
        return label, confidence

    if label == target:
        return label, confidence
    # Override plain sentiment or the opposite sarcasm subtype.
    if label not in ("positive", "negative", "neutral") and label not in _SARCASM:
        return label, confidence

    t_idx = LABELS.index(target)
    target_p = float(proba[t_idx]) if t_idx < len(proba) else 0.0
    new_conf = max(target_p, min(confidence + 0.15, 0.85), 0.35)
    return target, float(min(new_conf, 0.99))


@dataclass
class Prediction:
    label: str
    confidence: float
    discrepancy_vector: dict[str, float]
    low_fidelity: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "discrepancy_vector": self.discrepancy_vector,
            "low_fidelity": self.low_fidelity,
        }


class Pipeline:
    def __init__(
        self,
        *,
        bundle: Any,
        clf_pack: dict,
        fvt_threshold: float = DEFAULT_FVT_THRESHOLD,
    ):
        self.bundle = bundle
        self.clf_pack = clf_pack
        self.fvt_threshold = fvt_threshold

    # ---- construction ------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        *,
        clf_path: Path = DEFAULT_CLF,
        fvt_threshold: float = DEFAULT_FVT_THRESHOLD,
        device: str | None = None,
    ) -> "Pipeline":
        from .models import load_backbones  # lazy

        bundle = load_backbones(device=device)
        clf_pack = load_clf(clf_path)
        return cls(bundle=bundle, clf_pack=clf_pack, fvt_threshold=fvt_threshold)

    # ---- core --------------------------------------------------------------

    def features_for(self, text: str, image: Any) -> DiscrepancyFeatures:
        """Compute the GDRM feature vector for one sample."""
        features, _ = self.features_and_caption(text, image)
        return features

    def features_and_caption(self, text: str, image: Any) -> tuple[DiscrepancyFeatures, str]:
        """Return GDRM features plus SmolVLM caption ``T̂`` (for explainability UI)."""
        from .models import (  # lazy
            caption_image,
            embed_image_mclip,
            embed_text_mclip,
            polarity_probs,
        )

        text_emb_T = embed_text_mclip(self.bundle, text)
        T_hat = caption_image(self.bundle, image)
        text_emb_T_hat = embed_text_mclip(self.bundle, T_hat)
        image_emb_I = embed_image_mclip(self.bundle, image)
        pol_T = polarity_probs(self.bundle, text)
        pol_T_hat = polarity_probs(self.bundle, T_hat)
        features = build_feature_vector(
            text_emb_T=text_emb_T,
            text_emb_T_hat=text_emb_T_hat,
            image_emb_I=image_emb_I,
            polarity_probs_T=pol_T,
            polarity_probs_T_hat=pol_T_hat,
        )
        return features, T_hat

    def explain(self, text: str, image: Any) -> tuple[Prediction, DiscrepancyFeatures, str]:
        """One-pass inference returning prediction, features, and image caption."""
        features, T_hat = self.features_and_caption(text, image)
        return self.predict_from_features(features), features, T_hat

    def predict_from_features(self, features: DiscrepancyFeatures) -> Prediction:
        """Run the classifier on an already-computed feature vector."""
        proba = predict_proba(self.clf_pack, features)
        idx = int(np.argmax(proba))
        label = LABELS[idx]
        confidence = float(proba[idx])
        label, confidence = refine_label_for_polarity_conflict(
            label, confidence, proba, features
        )
        low_fidelity = bool(features.Fvt < self.fvt_threshold)
        return Prediction(
            label=label,
            confidence=confidence,
            discrepancy_vector=features.as_dict(),
            low_fidelity=low_fidelity,
        )

    def predict(self, text: str, image: Any) -> Prediction:
        features = self.features_for(text, image)
        return self.predict_from_features(features)


# ---- functional sugar --------------------------------------------------------

_default_pipeline: Pipeline | None = None


def predict(text: str, image: Any) -> Prediction:
    """Spec entry point. Lazily constructs a process-wide singleton pipeline."""
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = Pipeline.from_pretrained()
    return _default_pipeline.predict(text, image)
