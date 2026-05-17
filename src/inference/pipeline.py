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
        return build_feature_vector(
            text_emb_T=text_emb_T,
            text_emb_T_hat=text_emb_T_hat,
            image_emb_I=image_emb_I,
            polarity_probs_T=pol_T,
            polarity_probs_T_hat=pol_T_hat,
        )

    def predict_from_features(self, features: DiscrepancyFeatures) -> Prediction:
        """Run the classifier on an already-computed feature vector."""
        proba = predict_proba(self.clf_pack, features)
        idx = int(np.argmax(proba))
        label = LABELS[idx]
        confidence = float(proba[idx])
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
