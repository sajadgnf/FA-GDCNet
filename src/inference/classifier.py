"""Lightweight discrepancy classifier with 5-fold stratified CV.

Loads the canonical labeled dataset, walks every record through the inference
pipeline to obtain GDRM feature vectors, then fits a `LogisticRegression`. If
the mean cross-validated Macro-F1 falls below 0.40, automatically retries with
`LinearSVC` and keeps whichever model is stronger.

The trained model is persisted to `artifacts/clf.joblib`. The same artifact is
loaded at inference time by `pipeline.predict`.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

from data.schema import LABELS, iter_dataset

from .gdrm import FEATURE_NAMES, DiscrepancyFeatures

log = logging.getLogger(__name__)

DEFAULT_DATASET = Path("datasets") / "persian_multimodal_irony.jsonl"
DEFAULT_FEATURES = Path("artifacts") / "features.npz"
DEFAULT_CLF = Path("artifacts") / "clf.joblib"
LOGREG_F1_FLOOR = 0.40


@dataclass
class TrainResult:
    classifier_name: str
    fold_macro_f1: list[float]
    mean_macro_f1: float
    std_macro_f1: float
    saved_to: Path


def _build_logreg() -> LogisticRegression:
    return LogisticRegression(
        class_weight="balanced",
        penalty="l2",
        solver="lbfgs",
        max_iter=2000,
        random_state=0,
    )


def _build_svc() -> CalibratedClassifierCV:
    # Wrap LinearSVC in CalibratedClassifierCV so we get a `predict_proba`
    # surface matching LogisticRegression — needed by the inference pipeline
    # to return a confidence score.
    return CalibratedClassifierCV(
        LinearSVC(class_weight="balanced", random_state=0, max_iter=5000),
        method="sigmoid",
        cv=3,
    )


def _cross_validated_macro_f1(
    X: np.ndarray,
    y: np.ndarray,
    *,
    build,
    n_splits: int = 5,
) -> tuple[list[float], np.ndarray]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    fold_scores: list[float] = []
    oof_preds = np.empty_like(y, dtype=object)
    for train_idx, test_idx in skf.split(X, y):
        clf = build()
        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict(X[test_idx])
        oof_preds[test_idx] = preds
        fold_scores.append(f1_score(y[test_idx], preds, average="macro", labels=list(LABELS)))
    return fold_scores, oof_preds


def features_from_records(records: Iterable[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Materialize the (X, y) numpy arrays for an already-feature-cached set.

    Each record SHOULD already contain `features` (a 6-list) and `label`.
    Used by the eval scripts which pre-compute features once and reuse them.
    """
    X_rows: list[np.ndarray] = []
    y_rows: list[str] = []
    for r in records:
        feats = r.get("features")
        if feats is None:
            raise KeyError(f"record {r.get('post_id')!r} lacks pre-computed `features`")
        X_rows.append(np.asarray(feats, dtype=np.float32))
        y_rows.append(r["label"])
    return np.vstack(X_rows), np.asarray(y_rows, dtype=object)


def compute_dataset_features(
    dataset_path: Path,
    *,
    cache_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Walk every dataset record through the inference backbones to get features.

    Imports the heavy backbones lazily so this module stays importable on
    machines without torch installed (e.g. when training is run from a
    different environment than the analysis notebooks).
    """
    from PIL import Image  # lazy

    from .models import (
        caption_image,
        embed_image_mclip,
        embed_text_mclip,
        load_backbones,
        polarity_probs,
    )

    bundle = load_backbones()

    rows: list[dict] = []
    X_rows: list[np.ndarray] = []
    y_rows: list[str] = []
    post_ids: list[str] = []
    for rec in iter_dataset(dataset_path):
        image = Image.open(rec.image_path).convert("RGB")
        text_emb_T = embed_text_mclip(bundle, rec.caption)
        T_hat = caption_image(bundle, image)
        text_emb_T_hat = embed_text_mclip(bundle, T_hat)
        image_emb_I = embed_image_mclip(bundle, image)
        pol_T = polarity_probs(bundle, rec.caption)
        pol_T_hat = polarity_probs(bundle, T_hat)

        from .gdrm import build_feature_vector

        feats = build_feature_vector(
            text_emb_T=text_emb_T,
            text_emb_T_hat=text_emb_T_hat,
            image_emb_I=image_emb_I,
            polarity_probs_T=pol_T,
            polarity_probs_T_hat=pol_T_hat,
        )
        rows.append({"post_id": rec.post_id, "label": rec.label, **feats.as_dict()})
        X_rows.append(feats.as_array())
        y_rows.append(rec.label)
        post_ids.append(rec.post_id)

    X = np.vstack(X_rows)
    y = np.asarray(y_rows, dtype=object)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            X=X,
            y=y,
            post_ids=np.asarray(post_ids, dtype=object),
            feature_names=np.asarray(FEATURE_NAMES, dtype=object),
        )
    return X, y, post_ids


def train(
    X: np.ndarray,
    y: np.ndarray,
    *,
    save_to: Path = DEFAULT_CLF,
) -> TrainResult:
    """Fit Logistic Regression; fall back to calibrated Linear SVM if weak."""
    log.info("running 5-fold CV with LogisticRegression on %d samples", len(y))
    fold_scores, _ = _cross_validated_macro_f1(X, y, build=_build_logreg)
    mean_f1 = float(np.mean(fold_scores))
    std_f1 = float(np.std(fold_scores))
    chosen_name = "LogisticRegression"
    chosen_build = _build_logreg

    if mean_f1 < LOGREG_F1_FLOOR:
        log.warning(
            "LogReg mean macro-F1=%.3f below floor %.2f; retrying with LinearSVC",
            mean_f1,
            LOGREG_F1_FLOOR,
        )
        svc_fold_scores, _ = _cross_validated_macro_f1(X, y, build=_build_svc)
        svc_mean = float(np.mean(svc_fold_scores))
        if svc_mean > mean_f1:
            chosen_name = "CalibratedLinearSVC"
            chosen_build = _build_svc
            fold_scores = svc_fold_scores
            mean_f1 = svc_mean
            std_f1 = float(np.std(svc_fold_scores))

    final_clf = chosen_build()
    final_clf.fit(X, y)
    save_to.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": final_clf,
            "classifier_name": chosen_name,
            "feature_names": list(FEATURE_NAMES),
            "label_order": list(LABELS),
        },
        save_to,
    )
    return TrainResult(
        classifier_name=chosen_name,
        fold_macro_f1=fold_scores,
        mean_macro_f1=mean_f1,
        std_macro_f1=std_f1,
        saved_to=save_to,
    )


def load(path: Path = DEFAULT_CLF) -> dict:
    """Reload a previously-trained classifier checkpoint."""
    return joblib.load(path)


def predict_proba(clf_pack: dict, features: DiscrepancyFeatures) -> np.ndarray:
    """Helper: get the per-class probabilities aligned with `LABELS`."""
    arr = features.as_array().reshape(1, -1)
    proba = clf_pack["model"].predict_proba(arr)[0]
    # `clf_pack["model"].classes_` may not equal LABELS verbatim — re-align.
    class_to_idx = {c: i for i, c in enumerate(clf_pack["model"].classes_)}
    out = np.zeros(len(LABELS), dtype=np.float32)
    for i, lbl in enumerate(LABELS):
        if lbl in class_to_idx:
            out[i] = proba[class_to_idx[lbl]]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit the discrepancy classifier.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--features-cache", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--save-to", type=Path, default=DEFAULT_CLF)
    parser.add_argument("--from-cache", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.from_cache and args.features_cache.exists():
        npz = np.load(args.features_cache, allow_pickle=True)
        X, y = npz["X"], npz["y"]
    else:
        X, y, _ = compute_dataset_features(args.dataset, cache_path=args.features_cache)

    result = train(X, y, save_to=args.save_to)
    print(json.dumps({
        "classifier": result.classifier_name,
        "mean_macro_f1": result.mean_macro_f1,
        "std_macro_f1": result.std_macro_f1,
        "fold_macro_f1": result.fold_macro_f1,
        "saved_to": str(result.saved_to),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
