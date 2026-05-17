"""Text-only ParsBERT baseline on the same 5-fold splits.

The frozen ParsBERT polarity head emits a 2-class (neg/pos) distribution. We
project that to the 5-class label space by using `argmax`-style mapping and
fitting a tiny LogisticRegression on the same fold splits, so the comparison
is apples-to-apples with the multimodal pipeline (same data, same folds).

Outputs `reports/baseline.csv`.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from data.schema import LABELS, iter_dataset

log = logging.getLogger(__name__)

DEFAULT_DATASET = Path("datasets") / "persian_multimodal_irony.jsonl"
DEFAULT_BASELINE_CSV = Path("reports") / "baseline.csv"
DEFAULT_BASELINE_FEATURES = Path("artifacts") / "baseline_features.npz"


def _compute_baseline_features(dataset: Path, cache: Path) -> tuple[np.ndarray, np.ndarray]:
    if cache.exists():
        npz = np.load(cache, allow_pickle=True)
        return npz["X"], npz["y"]

    from inference.models import load_backbones, polarity_probs

    bundle = load_backbones()
    X_rows: list[np.ndarray] = []
    y_rows: list[str] = []
    for rec in iter_dataset(dataset):
        probs = polarity_probs(bundle, rec.caption)
        X_rows.append(np.asarray(probs, dtype=np.float32))
        y_rows.append(rec.label)
    X = np.vstack(X_rows)
    y = np.asarray(y_rows, dtype=object)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X=X, y=y)
    return X, y


def evaluate(X: np.ndarray, y: np.ndarray) -> dict:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    accs: list[float] = []
    macros: list[float] = []
    per_class: dict[str, list[float]] = {lbl: [] for lbl in LABELS}
    for train_idx, test_idx in skf.split(X, y):
        clf = LogisticRegression(
            class_weight="balanced",
            penalty="l2",
            solver="lbfgs",
            max_iter=2000,
            random_state=0,
        )
        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict(X[test_idx])
        true = y[test_idx]
        accs.append(accuracy_score(true, preds))
        macros.append(f1_score(true, preds, average="macro", labels=list(LABELS)))
        f1s = f1_score(true, preds, average=None, labels=list(LABELS))
        for lbl, val in zip(LABELS, f1s):
            per_class[lbl].append(float(val))
    return {
        "accuracy": accs,
        "macro_f1": macros,
        "per_class_f1": per_class,
    }


def write_csv(result: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["fold", "accuracy", "macro_f1"] + [f"f1_{lbl}" for lbl in LABELS]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(len(result["accuracy"])):
            row = [i + 1, result["accuracy"][i], result["macro_f1"][i]]
            row += [result["per_class_f1"][lbl][i] for lbl in LABELS]
            w.writerow(row)
        mean = lambda xs: float(np.mean(xs)) if xs else 0.0
        std = lambda xs: float(np.std(xs)) if xs else 0.0
        summary = [
            "mean±std",
            f"{mean(result['accuracy']):.4f}±{std(result['accuracy']):.4f}",
            f"{mean(result['macro_f1']):.4f}±{std(result['macro_f1']):.4f}",
        ]
        for lbl in LABELS:
            summary.append(
                f"{mean(result['per_class_f1'][lbl]):.4f}±{std(result['per_class_f1'][lbl]):.4f}"
            )
        w.writerow(summary)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Unimodal ParsBERT baseline.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--features-cache", type=Path, default=DEFAULT_BASELINE_FEATURES)
    parser.add_argument("--out", type=Path, default=DEFAULT_BASELINE_CSV)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    X, y = _compute_baseline_features(args.dataset, args.features_cache)
    result = evaluate(X, y)
    out = write_csv(result, args.out)
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
