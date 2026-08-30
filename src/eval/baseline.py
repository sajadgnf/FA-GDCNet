"""Text-only polarity baseline on the same labeled rows as the multimodal set.

The frozen head is ``cardiffnlp/twitter-xlm-roberta-base-sentiment`` (2-d),
not ParsBERT. The proposal named ParsBERT; this file documents the deviation.

A tiny LogisticRegression is fit on the same 5-fold splits as the multimodal
head. The cache is ignored when ``post_ids``/``y`` do not match the current
jsonl (same image-present filter).

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

from data.eval_set import is_eval_eligible, slice_records_from_cache
from data.schema import LABELS, iter_dataset

log = logging.getLogger(__name__)

DEFAULT_DATASET = Path("datasets") / "persian_multimodal_irony.jsonl"
DEFAULT_BASELINE_CSV = Path("reports") / "baseline.csv"
DEFAULT_BASELINE_SARCASM_CSV = Path("reports") / "baseline_sarcasm.csv"
DEFAULT_BASELINE_FEATURES = Path("artifacts") / "baseline_features.npz"

SARCASM_LABELS = frozenset({"positive_sarcasm", "negative_sarcasm"})


def _usable_records(dataset: Path) -> list:
    return [
        r
        for r in iter_dataset(dataset)
        if Path(r.image_path).is_file() and is_eval_eligible(r.annotators)
    ]


def _cache_matches_dataset(cache: Path, records: list) -> bool:
    if not cache.is_file():
        return False
    npz = np.load(cache, allow_pickle=True)
    if "post_ids" not in npz or "y" not in npz:
        return False
    cached = list(zip((str(x) for x in npz["post_ids"]), (str(x) for x in npz["y"])))
    want = [(r.post_id, r.label) for r in records]
    return cached == want


def _compute_baseline_features(dataset: Path, cache: Path) -> tuple[np.ndarray, np.ndarray]:
    records = _usable_records(dataset)
    sliced = slice_records_from_cache(cache, records)
    if sliced is not None:
        return sliced
    if cache.is_file():
        log.warning("baseline cache unusable vs %s; recomputing", dataset)

    from inference.models import load_polarity_only, polarity_probs

    bundle = load_polarity_only()
    X_rows: list[np.ndarray] = []
    y_rows: list[str] = []
    post_ids: list[str] = []
    for rec in records:
        probs = polarity_probs(bundle, rec.caption)
        X_rows.append(np.asarray(probs, dtype=np.float32))
        y_rows.append(rec.label)
        post_ids.append(rec.post_id)
    X = np.vstack(X_rows)
    y = np.asarray(y_rows, dtype=object)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X=X, y=y, post_ids=np.asarray(post_ids, dtype=object))
    log.info("baseline features: %d samples", X.shape[0])
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


def evaluate_binary_sarcasm(X: np.ndarray, y: np.ndarray) -> dict:
    y_bin = np.asarray([1 if lbl in SARCASM_LABELS else 0 for lbl in y], dtype=int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    accs: list[float] = []
    f1s: list[float] = []
    for train_idx, test_idx in skf.split(X, y_bin):
        clf = LogisticRegression(
            class_weight="balanced",
            penalty="l2",
            solver="lbfgs",
            max_iter=2000,
            random_state=0,
        )
        clf.fit(X[train_idx], y_bin[train_idx])
        preds = clf.predict(X[test_idx])
        true = y_bin[test_idx]
        accs.append(accuracy_score(true, preds))
        f1s.append(f1_score(true, preds, zero_division=0))
    return {"accuracy": accs, "f1": f1s}


def write_sarcasm_csv(result: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold", "accuracy", "f1"])
        for i in range(len(result["accuracy"])):
            w.writerow([i + 1, result["accuracy"][i], result["f1"][i]])
        w.writerow([
            "mean±std",
            f"{np.mean(result['accuracy']):.4f}±{np.std(result['accuracy']):.4f}",
            f"{np.mean(result['f1']):.4f}±{np.std(result['f1']):.4f}",
        ])
    return path


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
    sarcasm = evaluate_binary_sarcasm(X, y)
    sarcasm_out = write_sarcasm_csv(sarcasm, DEFAULT_BASELINE_SARCASM_CSV)
    log.info("wrote %s", sarcasm_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
