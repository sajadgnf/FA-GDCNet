"""5-fold stratified CV reporting (Accuracy, Macro-F1, per-class F1).

Writes `reports/metrics.csv`:
- one row per fold with Accuracy, Macro-F1, and per-class F1 columns;
- one final `mean ± std` row.

Per spec scenario *Meeting the accuracy hypothesis*, the report also appends a
trailing comment indicating whether the mean Accuracy on the sarcasm classes
crosses the 70 percent threshold.
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

from data.schema import LABELS

from inference.classifier import (
    DEFAULT_DATASET,
    DEFAULT_FEATURES,
    compute_dataset_features,
)

log = logging.getLogger(__name__)

DEFAULT_METRICS_CSV = Path("reports") / "metrics.csv"
SARCASM_LABELS = ("positive_sarcasm", "negative_sarcasm")
SARCASM_ACCURACY_FLOOR = 0.70


def _load_features(dataset: Path, cache: Path) -> tuple[np.ndarray, np.ndarray]:
    if cache.exists():
        npz = np.load(cache, allow_pickle=True)
        return npz["X"], npz["y"]
    X, y, _ = compute_dataset_features(dataset, cache_path=cache)
    return X, y


def _build_clf() -> LogisticRegression:
    return LogisticRegression(
        class_weight="balanced",
        penalty="l2",
        solver="lbfgs",
        max_iter=2000,
        random_state=0,
    )


def evaluate(X: np.ndarray, y: np.ndarray, *, n_splits: int = 5) -> dict:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    per_class_f1: dict[str, list[float]] = {lbl: [] for lbl in LABELS}
    accuracies: list[float] = []
    macro_f1s: list[float] = []
    sarcasm_accuracies: list[float] = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        clf = _build_clf()
        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict(X[test_idx])
        true = y[test_idx]
        accuracies.append(accuracy_score(true, preds))
        macro_f1s.append(f1_score(true, preds, average="macro", labels=list(LABELS)))
        f1_per_class = f1_score(true, preds, average=None, labels=list(LABELS))
        for lbl, val in zip(LABELS, f1_per_class):
            per_class_f1[lbl].append(float(val))

        sarcasm_mask = np.isin(true, SARCASM_LABELS)
        if sarcasm_mask.any():
            sarcasm_accuracies.append(
                accuracy_score(true[sarcasm_mask], preds[sarcasm_mask])
            )
        log.info("fold %d acc=%.3f macro_f1=%.3f", fold_idx, accuracies[-1], macro_f1s[-1])

    return {
        "folds": list(range(1, n_splits + 1)),
        "accuracy": accuracies,
        "macro_f1": macro_f1s,
        "per_class_f1": per_class_f1,
        "sarcasm_accuracy": sarcasm_accuracies,
    }


def write_csv(result: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["fold", "accuracy", "macro_f1"] + [f"f1_{lbl}" for lbl in LABELS]
    rows: list[list] = []
    for i, fold in enumerate(result["folds"]):
        row = [fold, result["accuracy"][i], result["macro_f1"][i]]
        row += [result["per_class_f1"][lbl][i] for lbl in LABELS]
        rows.append(row)

    def _stat(values: list[float]) -> tuple[float, float]:
        if not values:
            return 0.0, 0.0
        return float(np.mean(values)), float(np.std(values))

    acc_mu, acc_sd = _stat(result["accuracy"])
    f1_mu, f1_sd = _stat(result["macro_f1"])
    summary_row: list = ["mean±std", f"{acc_mu:.4f}±{acc_sd:.4f}", f"{f1_mu:.4f}±{f1_sd:.4f}"]
    for lbl in LABELS:
        mu, sd = _stat(result["per_class_f1"][lbl])
        summary_row.append(f"{mu:.4f}±{sd:.4f}")
    rows.append(summary_row)

    sarcasm_acc_mu = float(np.mean(result["sarcasm_accuracy"])) if result["sarcasm_accuracy"] else 0.0
    passes = sarcasm_acc_mu >= SARCASM_ACCURACY_FLOOR
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in rows:
            writer.writerow(r)
        # Trailing footer per spec scenario "Meeting the accuracy hypothesis".
        writer.writerow([])
        writer.writerow(["# sarcasm_mean_accuracy", f"{sarcasm_acc_mu:.4f}"])
        writer.writerow(["# meets_hypothesis_70pct", "true" if passes else "false"])
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="5-fold stratified CV metrics.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--features-cache", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--out", type=Path, default=DEFAULT_METRICS_CSV)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    X, y = _load_features(args.dataset, args.features_cache)
    result = evaluate(X, y)
    out = write_csv(result, args.out)
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
