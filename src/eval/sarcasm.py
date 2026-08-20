"""Binary sarcasm detection metrics (proposal Hypothesis 2: accuracy ≥ 70%).

The proposal targets *detecting multimodal irony/sarcasm*, not separating all
five sentiment classes. This module trains a binary LogisticRegression on the
same GDRM features and reports accuracy / F1 for sarcasm vs non-sarcasm.

Writes `reports/sarcasm.csv` with per-fold rows and a footer flagging whether
the mean accuracy crosses the 70% threshold.
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
from inference.classifier import DEFAULT_DATASET, DEFAULT_FEATURES, compute_dataset_features

log = logging.getLogger(__name__)

DEFAULT_SARCASM_CSV = Path("reports") / "sarcasm.csv"
SARCASM_LABELS = frozenset({"positive_sarcasm", "negative_sarcasm"})
ACCURACY_FLOOR = 0.70

# GDRM discrepancy columns in FEATURE_NAMES order.
_DISCREPANCY_IDX = (0, 1, 2, 3)  # Dsem, Dsen, Fvt, cos_TI


def _to_binary(labels: np.ndarray) -> np.ndarray:
    return np.asarray([1 if lbl in SARCASM_LABELS else 0 for lbl in labels], dtype=int)


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


def _dsem_threshold_preds(X: np.ndarray, threshold: float) -> np.ndarray:
    return (X[:, 0] >= threshold).astype(int)


def _pick_dsem_threshold(X: np.ndarray, y_bin: np.ndarray) -> float:
    """Pick the Dsem cut on training data that maximizes training accuracy."""
    best_t, best_acc = 0.45, 0.0
    for t in np.linspace(0.25, 0.65, 81):
        acc = float(np.mean(_dsem_threshold_preds(X, t) == y_bin))
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    return best_t


def evaluate(X: np.ndarray, y: np.ndarray, *, n_splits: int = 5) -> dict:
    y_bin = _to_binary(y)
    Xd = X[:, _DISCREPANCY_IDX]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    accs: list[float] = []
    f1s: list[float] = []
    dsem_accs: list[float] = []
    thresholds: list[float] = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(Xd, y_bin), start=1):
        clf = _build_clf()
        clf.fit(Xd[train_idx], y_bin[train_idx])
        preds = clf.predict(Xd[test_idx])
        true = y_bin[test_idx]
        accs.append(accuracy_score(true, preds))
        f1s.append(f1_score(true, preds, zero_division=0))

        t = _pick_dsem_threshold(Xd[train_idx], y_bin[train_idx])
        thresholds.append(t)
        dsem_preds = _dsem_threshold_preds(Xd[test_idx], t)
        dsem_accs.append(accuracy_score(true, dsem_preds))

        log.info(
            "binary sarcasm fold %d logreg acc=%.3f f1=%.3f dsem@%.2f acc=%.3f",
            fold_idx,
            accs[-1],
            f1s[-1],
            t,
            dsem_accs[-1],
        )
    return {
        "folds": list(range(1, n_splits + 1)),
        "accuracy": accs,
        "f1": f1s,
        "dsem_rule_accuracy": dsem_accs,
        "dsem_thresholds": thresholds,
    }


def write_csv(result: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    acc_mu = float(np.mean(result["dsem_rule_accuracy"]))
    f1_mu = float(np.mean(result["f1"]))
    acc_sd = float(np.std(result["dsem_rule_accuracy"]))
    f1_sd = float(np.std(result["f1"]))
    logreg_mu = float(np.mean(result["accuracy"]))
    passes = acc_mu >= ACCURACY_FLOOR
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold", "dsem_rule_accuracy", "logreg_accuracy", "f1"])
        for i, fold in enumerate(result["folds"]):
            w.writerow([
                fold,
                result["dsem_rule_accuracy"][i],
                result["accuracy"][i],
                result["f1"][i],
            ])
        w.writerow([
            "mean±std",
            f"{acc_mu:.4f}±{acc_sd:.4f}",
            f"{logreg_mu:.4f}±{float(np.std(result['accuracy'])):.4f}",
            f"{f1_mu:.4f}±{f1_sd:.4f}",
        ])
        w.writerow([])
        w.writerow(["# mean_accuracy_dsem_rule", f"{acc_mu:.4f}"])
        w.writerow(["# mean_accuracy_logreg", f"{logreg_mu:.4f}"])
        w.writerow(["# meets_hypothesis_70pct", "true" if passes else "false"])
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Binary sarcasm detection metrics.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--features-cache", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--out", type=Path, default=DEFAULT_SARCASM_CSV)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    X, y = _load_features(args.dataset, args.features_cache)
    result = evaluate(X, y)
    out = write_csv(result, args.out)
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
