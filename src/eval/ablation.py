"""Ablation study over GDRM signals.

Re-fits the lightweight classifier on each subset of GDRM signals from the
6-feature vector. Subsets (per spec scenario *Single-signal and pairwise runs*):

    {Dsem}, {Dsen}, {Fvt},
    {Dsem, Dsen}, {Dsem, Fvt}, {Dsen, Fvt},
    {Dsem, Dsen, Fvt}

Auxiliary features (`cos_TI`, `polarity_T`, `polarity_T_hat`) are kept in every
configuration so the ablation isolates the contribution of the three named
discrepancy signals (Dsem/Dsen/Fvt) rather than the auxiliary inputs.

Outputs:
- `reports/ablation.csv` with one row per configuration.
- `reports/ablation.png` bar chart of Macro-F1.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import logging
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from data.schema import LABELS
from inference.classifier import DEFAULT_DATASET, DEFAULT_FEATURES, compute_dataset_features
from inference.gdrm import FEATURE_NAMES

log = logging.getLogger(__name__)

DEFAULT_ABLATION_CSV = Path("reports") / "ablation.csv"
DEFAULT_ABLATION_PNG = Path("reports") / "ablation.png"

CORE_SIGNALS: tuple[str, ...] = ("Dsem", "Dsen", "Fvt")
AUX_SIGNALS: tuple[str, ...] = ("cos_TI", "polarity_T", "polarity_T_hat")


def _powerset(items: tuple[str, ...]) -> list[tuple[str, ...]]:
    out: list[tuple[str, ...]] = []
    for size in range(1, len(items) + 1):
        out.extend(itertools.combinations(items, size))
    return out


def _column_idx(feature_subset: tuple[str, ...]) -> list[int]:
    name_to_idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
    return [name_to_idx[n] for n in feature_subset]


def _eval(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    accs: list[float] = []
    f1s: list[float] = []
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
        accs.append(accuracy_score(y[test_idx], preds))
        f1s.append(f1_score(y[test_idx], preds, average="macro", labels=list(LABELS)))
    return float(np.mean(accs)), float(np.mean(f1s))


def run(X: np.ndarray, y: np.ndarray) -> list[dict]:
    aux_idx = _column_idx(AUX_SIGNALS)
    rows: list[dict] = []
    for subset in _powerset(CORE_SIGNALS):
        cols = sorted(_column_idx(subset) + aux_idx)
        Xs = X[:, cols]
        acc, f1 = _eval(Xs, y)
        rows.append(
            {
                "configuration": "+".join(subset),
                "n_features": len(cols),
                "mean_accuracy": acc,
                "mean_macro_f1": f1,
            }
        )
        log.info("subset=%s mean_macro_f1=%.4f", "+".join(subset), f1)
    return rows


def write_csv(rows: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["configuration", "n_features", "mean_accuracy", "mean_macro_f1"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fields})
    return path


def write_png(rows: list[dict], path: Path) -> Path:
    import matplotlib.pyplot as plt  # lazy

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [r["configuration"] for r in rows]
    values = [r["mean_macro_f1"] for r in rows]
    ax.bar(labels, values, color="#1f77b4")
    ax.set_ylabel("Mean Macro-F1 (5-fold)")
    ax.set_title("FA-GDCNet ablation over Dsem / Dsen / Fvt")
    ax.set_ylim(0, max(values) * 1.15 if values else 1)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GDRM ablation study.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--features-cache", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--csv", type=Path, default=DEFAULT_ABLATION_CSV)
    parser.add_argument("--png", type=Path, default=DEFAULT_ABLATION_PNG)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.features_cache.exists():
        npz = np.load(args.features_cache, allow_pickle=True)
        X, y = npz["X"], npz["y"]
    else:
        X, y, _ = compute_dataset_features(args.dataset, cache_path=args.features_cache)

    rows = run(X, y)
    write_csv(rows, args.csv)
    write_png(rows, args.png)
    log.info("wrote %s and %s", args.csv, args.png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
