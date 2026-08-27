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

from collections import Counter

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

from data.schema import LABELS

from inference.classifier import (
    DEFAULT_DATASET,
    DEFAULT_FEATURES,
    compute_dataset_features,
)

log = logging.getLogger(__name__)

DEFAULT_METRICS_CSV = Path("reports") / "metrics.csv"
DEFAULT_CONFUSION_PNG = Path("reports") / "confusion.png"
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

    dummy_accuracies: list[float] = []
    dummy_macro_f1s: list[float] = []
    oof_true: list[str] = []
    oof_pred: list[str] = []

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

        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X[train_idx], y[train_idx])
        dpred = dummy.predict(X[test_idx])
        dummy_accuracies.append(accuracy_score(true, dpred))
        dummy_macro_f1s.append(
            f1_score(true, dpred, average="macro", labels=list(LABELS), zero_division=0)
        )

        oof_true.extend(true.tolist())
        oof_pred.extend(preds.tolist())

        sarcasm_mask = np.isin(true, SARCASM_LABELS)
        if sarcasm_mask.any():
            sarcasm_accuracies.append(
                accuracy_score(true[sarcasm_mask], preds[sarcasm_mask])
            )
        log.info("fold %d acc=%.3f macro_f1=%.3f", fold_idx, accuracies[-1], macro_f1s[-1])

    counts = Counter(str(v) for v in y.tolist())
    return {
        "folds": list(range(1, n_splits + 1)),
        "accuracy": accuracies,
        "macro_f1": macro_f1s,
        "per_class_f1": per_class_f1,
        "sarcasm_accuracy": sarcasm_accuracies,
        "dummy_accuracy": dummy_accuracies,
        "dummy_macro_f1": dummy_macro_f1s,
        "oof_true": oof_true,
        "oof_pred": oof_pred,
        "n_samples": int(len(y)),
        "label_counts": {lbl: int(counts.get(lbl, 0)) for lbl in LABELS},
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
    dummy_acc_mu = float(np.mean(result.get("dummy_accuracy") or [0.0]))
    dummy_f1_mu = float(np.mean(result.get("dummy_macro_f1") or [0.0]))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in rows:
            writer.writerow(r)
        # Trailing footer per spec scenario "Meeting the accuracy hypothesis".
        writer.writerow([])
        writer.writerow(["# sarcasm_mean_accuracy", f"{sarcasm_acc_mu:.4f}"])
        writer.writerow(["# meets_hypothesis_70pct", "true" if passes else "false"])
        writer.writerow(["# dummy_majority_accuracy", f"{dummy_acc_mu:.4f}"])
        writer.writerow(["# dummy_majority_macro_f1", f"{dummy_f1_mu:.4f}"])
        writer.writerow(["# n_samples", str(result.get("n_samples", ""))])
        for lbl in LABELS:
            writer.writerow([f"# n_{lbl}", str((result.get("label_counts") or {}).get(lbl, ""))])
    return path


def write_confusion_png(y_true: list[str], y_pred: list[str], path: Path) -> Path:
    """Out-of-fold confusion matrix (rows = gold, columns = predicted)."""
    import matplotlib.pyplot as plt  # lazy

    path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(LABELS)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    ax.set_title("5-fold out-of-fold confusion")
    vmax = int(cm.max()) if cm.size else 0
    thresh = vmax / 2.0 if vmax else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(int(cm[i, j])),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9,
            )
    plt.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="5-fold stratified CV metrics.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--features-cache", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--out", type=Path, default=DEFAULT_METRICS_CSV)
    parser.add_argument("--confusion", type=Path, default=DEFAULT_CONFUSION_PNG)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    X, y = _load_features(args.dataset, args.features_cache)
    result = evaluate(X, y)
    out = write_csv(result, args.out)
    cm_path = write_confusion_png(result["oof_true"], result["oof_pred"], args.confusion)
    log.info("wrote %s and %s", out, cm_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
