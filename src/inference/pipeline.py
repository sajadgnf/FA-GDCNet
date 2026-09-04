"""End-to-end FA-GDCNet inference: `predict(text, image)`.

This is the spec-named entry point. It accepts a `(text, image)` pair and
returns a `Prediction` containing:

- `label`: one of `LABELS`
- `confidence`: probability of the chosen label in [0, 1]
- `discrepancy_vector`: dict view of the GDRM output (core signals + clash)
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
_PLAIN = frozenset({"positive", "negative", "neutral"})


def _plain_from_strong_text(p_t: float) -> str | None:
    if p_t <= -0.35:
        return "negative"
    if p_t >= 0.35:
        return "positive"
    return None


def _promote_confidence(
    *,
    confidence: float,
    proba: np.ndarray,
    target: str,
    floor: float,
    bump: float = 0.25,
    strong_floor: float | None = None,
    p_t: float = 0.0,
) -> float:
    t_idx = LABELS.index(target)
    target_p = float(proba[t_idx]) if t_idx < len(proba) else 0.0
    new_conf = max(target_p, min(confidence + bump, 0.85), floor)
    if strong_floor is not None and abs(p_t) >= 0.55:
        new_conf = max(new_conf, strong_floor)
    return float(min(new_conf, 0.99))


def refine_label_for_polarity_conflict(
    label: str,
    confidence: float,
    proba: np.ndarray,
    features: DiscrepancyFeatures,
) -> tuple[str, float]:
    """Adjust the classifier label using T / T̂ polarity signals.

    1. Opposite polarities + enough Dsen → sarcasm subtype.
    2. Sarcasm without a *clear* opposite T̂ polarity → plain sentiment from T.
    3. Aligned clear polarities → plain positive/negative (lifts weak «neutral»).

    Project taxonomy (image/delivery polarity names the subtype):
    - ``positive_sarcasm``: negative caption vs positive image affect (CLIP)
      (sad/bitter words with a happy face — e.g. mourning formula + smile)
    - ``negative_sarcasm``: positive caption vs negative image affect (CLIP)
      (cheerful words with a bleak image)
    """
    p_t = float(features.polarity_T)
    p_th = float(features.polarity_T_hat)
    dsen = float(features.Dsen)
    text_neg, text_pos = p_t <= -0.05, p_t >= 0.05
    hat_neg, hat_pos = p_th <= -0.15, p_th >= 0.15
    dsen_conflict = dsen >= 0.25
    # Sarcasm subtypes need a *clear* caption polarity, not a question-mark nick.
    clash_text_neg, clash_text_pos = p_t <= -0.20, p_t >= 0.20

    # ---- conflict → sarcasm -------------------------------------------------
    target: str | None = None
    if clash_text_neg and hat_pos and dsen_conflict:
        target = "positive_sarcasm"
    elif clash_text_pos and hat_neg and dsen_conflict:
        target = "negative_sarcasm"

    if target is not None:
        if label == target:
            return label, confidence
        if label not in _PLAIN and label not in _SARCASM:
            return label, confidence
        new_conf = _promote_confidence(
            confidence=confidence, proba=proba, target=target, floor=0.35, bump=0.15
        )
        if abs(p_t) >= 0.35 and abs(p_th) >= 0.35:
            new_conf = max(new_conf, 0.62)
        elif abs(p_t) >= 0.2 and abs(p_th) >= 0.25:
            new_conf = max(new_conf, 0.50)
        return target, float(min(new_conf, 0.99))

    # Same-sign polarities are not a sarcasm subtype. Dsen can still be large
    # from intensity only (mild +0.23 text vs a 0.99 smile).
    if label == "positive_sarcasm" and not clash_text_neg:
        fallback = "positive" if (hat_pos or text_pos) else "neutral"
        return fallback, _promote_confidence(
            confidence=confidence, proba=proba, target=fallback, floor=0.45, bump=0.15
        )
    if label == "negative_sarcasm" and not clash_text_pos:
        fallback = "negative" if (hat_neg or text_neg) else "neutral"
        return fallback, _promote_confidence(
            confidence=confidence, proba=proba, target=fallback, floor=0.45, bump=0.15
        )

    # ---- unsupported sarcasm (e.g. sad text + bland VLM caption) ------------
    # Near-neutral T̂ is not evidence of irony; trust strong caption polarity.
    if label in _SARCASM:
        plain = _plain_from_strong_text(p_t)
        if plain is not None:
            return plain, _promote_confidence(
                confidence=confidence,
                proba=proba,
                target=plain,
                floor=0.55,
                bump=0.2,
                strong_floor=0.65,
                p_t=p_t,
            )

    # ---- agreement → plain sentiment (don't leave ~25% neutral) ------------
    # Same-sign clear polarities count as agreement even when Dsen is large
    # from *strength* mismatch (e.g. 0.99 vs 0.50 both positive).
    agree_pos = p_t >= 0.35 and p_th >= 0.25
    agree_neg = p_t <= -0.35 and p_th <= -0.25
    # Strong text + non-opposing (near-neutral) description also aligns.
    if not agree_pos and p_t >= 0.35 and abs(p_th) < 0.15:
        agree_pos = True
    if not agree_neg and p_t <= -0.35 and abs(p_th) < 0.15:
        agree_neg = True

    if agree_pos:
        target = "positive"
    elif agree_neg:
        target = "negative"
    else:
        return label, confidence

    if label == target:
        return label, float(max(confidence, 0.55))
    if label != "neutral":
        return label, confidence

    return target, _promote_confidence(
        confidence=confidence,
        proba=proba,
        target=target,
        floor=0.55,
        bump=0.25,
        strong_floor=0.65,
        p_t=p_t,
    )


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
            image_polarity_probs,
            polarity_probs,
        )

        text_emb_T = embed_text_mclip(self.bundle, text)
        T_hat = caption_image(self.bundle, image)
        text_emb_T_hat = embed_text_mclip(self.bundle, T_hat)
        image_emb_I = embed_image_mclip(self.bundle, image)
        pol_T = polarity_probs(self.bundle, text)
        pol_T_hat = image_polarity_probs(self.bundle, image)
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
