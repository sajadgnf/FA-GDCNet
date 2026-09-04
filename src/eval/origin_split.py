"""Retrain and score sarcasm metrics on organic vs crafted recaption rows.

Crafted rows have ``post_id`` prefix ``craft-``. They are human-labeled, but
the captions were written for clash, not scraped with the image (§5).

Two protocols:

* **retrain**: 5-fold CV using only that subset (no leakage from the other
  domain). This is the number that answers whether the Instagram-caption
  domain carries the result.
* **oof_slice**: existing mixed-train OOF predictions scored on each subset.
  Optimistic: other folds may still contain crafted clash.

Writes ``reports/origin_split.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from data.eval_set import slice_to_eval_set
from data.schema import LABELS
from eval.baseline import DEFAULT_BASELINE_FEATURES, evaluate as baseline_evaluate
from eval.metrics import evaluate as multiclass_evaluate
from eval.sarcasm import evaluate as binary_evaluate
from inference.classifier import DEFAULT_DATASET, DEFAULT_FEATURES
from inference.gdrm import with_clash_column

log = logging.getLogger(__name__)

DEFAULT_OUT = Path("reports") / "origin_split.json"
DEFAULT_OOF = Path("reports") / "oof_preds.jsonl"
SARCASM_LABELS = ("positive_sarcasm", "negative_sarcasm")
CRAFT_PREFIX = "craft-"
RQ2_FLOOR = 0.10


def is_crafted(post_id: str) -> bool:
    return str(post_id).startswith(CRAFT_PREFIX)


def _load_mm(dataset: Path, cache: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    npz = np.load(cache, allow_pickle=True)
    ids = [str(x) for x in npz["post_ids"].tolist()] if "post_ids" in npz else None
    X, y, ids, _ = slice_to_eval_set(
        dataset, with_clash_column(npz["X"]), npz["y"], ids
    )
    return X, y, list(ids)


def _align_baseline(
    ids: list[str], cache: Path
) -> tuple[np.ndarray, np.ndarray] | None:
    if not cache.is_file():
        return None
    npz = np.load(cache, allow_pickle=True)
    if "post_ids" not in npz:
        return None
    by_id = {str(p): i for i, p in enumerate(npz["post_ids"].tolist())}
    xs = []
    ys = []
    for pid in ids:
        idx = by_id.get(pid)
        if idx is None:
            return None
        xs.append(np.asarray(npz["X"][idx]))
        ys.append(str(npz["y"][idx]))
    return np.stack(xs, axis=0), np.asarray(ys, dtype=object)


def _sarcasm_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scores = f1_score(
        y_true, y_pred, average=None, labels=list(SARCASM_LABELS), zero_division=0
    )
    return float(np.mean(scores))


def _mean_std(xs: list[float]) -> dict:
    arr = np.asarray(xs, dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "std": None}
    return {"mean": float(arr.mean()), "std": float(arr.std())}


def _binary_summary(result: dict) -> dict:
    dummy = float(result.get("dummy_not_sarcasm_accuracy") or 0.0)
    clash_acc = float(np.mean(result["clash_rule_accuracy"]))
    return {
        "n_samples": int(result.get("n_samples") or 0),
        "n_sarcasm": int(result.get("n_sarcasm") or 0),
        "dummy_not_sarcasm_accuracy": dummy,
        "clash": {
            "accuracy": _mean_std(result["clash_rule_accuracy"]),
            "precision": _mean_std(result["clash_rule_precision"]),
            "recall": _mean_std(result["clash_rule_recall"]),
            "f1": _mean_std(result["clash_rule_f1"]),
            "pdf_accuracy_bar": clash_acc >= 0.70,
            "beats_dummy_accuracy": clash_acc > dummy + 1e-6,
        },
        "logreg": {
            "accuracy": _mean_std(result["accuracy"]),
            "precision": _mean_std(result["precision"]),
            "recall": _mean_std(result["recall"]),
            "f1": _mean_std(result["f1"]),
        },
    }


def _multiclass_summary(result: dict) -> dict:
    sarc = [
        float(np.mean(a))
        for a in zip(
            result["per_class_f1"]["positive_sarcasm"],
            result["per_class_f1"]["negative_sarcasm"],
        )
    ]
    return {
        "accuracy": _mean_std(result["accuracy"]),
        "macro_f1": _mean_std(result["macro_f1"]),
        "sarcasm_class_f1": _mean_std(sarc),
        "f1_positive_sarcasm": _mean_std(result["per_class_f1"]["positive_sarcasm"]),
        "f1_negative_sarcasm": _mean_std(result["per_class_f1"]["negative_sarcasm"]),
    }


def _can_stratify(y: np.ndarray, n_splits: int = 5) -> bool:
    counts = Counter(str(v) for v in y.tolist())
    return bool(counts) and min(counts.values()) >= n_splits


def _retrain_split(
    X: np.ndarray,
    y: np.ndarray,
    baseline: tuple[np.ndarray, np.ndarray] | None,
) -> dict:
    labels = Counter(str(v) for v in y.tolist())
    out: dict = {
        "n": int(len(y)),
        "n_sarcasm": int(sum(1 for v in y if str(v) in SARCASM_LABELS)),
        "label_counts": {k: int(labels.get(k, 0)) for k in LABELS},
        "binary": None,
        "multiclass": None,
        "unimodal_sarcasm_class_f1": None,
        "rq2_delta": None,
        "rq2_meets_10pp": None,
        "skip": None,
    }
    y_bin = np.asarray([1 if str(v) in SARCASM_LABELS else 0 for v in y], dtype=int)
    if int(y_bin.sum()) < 5 or int((1 - y_bin).sum()) < 5:
        out["skip"] = "binary: fewer than 5 sarcasm or 5 non-sarcasm rows"
        return out
    out["binary"] = _binary_summary(binary_evaluate(X, y))
    if _can_stratify(y):
        mc = multiclass_evaluate(X, y)
        out["multiclass"] = _multiclass_summary(mc)
        mm_sarc = out["multiclass"]["sarcasm_class_f1"]["mean"]
        if baseline is not None and _can_stratify(baseline[1]):
            b = baseline_evaluate(baseline[0], baseline[1])
            b_sarc = float(
                np.mean(
                    [
                        np.mean(pair)
                        for pair in zip(
                            b["per_class_f1"]["positive_sarcasm"],
                            b["per_class_f1"]["negative_sarcasm"],
                        )
                    ]
                )
            )
            out["unimodal_sarcasm_class_f1"] = b_sarc
            if mm_sarc is not None:
                delta = float(mm_sarc) - b_sarc
                out["rq2_delta"] = delta
                out["rq2_meets_10pp"] = delta >= RQ2_FLOOR
    else:
        out["skip"] = (
            "multiclass: a label has fewer than 5 rows; binary still ran"
        )
    return out


def _load_oof(path: Path) -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    if not path.is_file():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[str(row["post_id"])] = (str(row["gold"]), str(row["pred"]))
    return out


def _oof_slice(ids: list[str], oof: dict[str, tuple[str, str]]) -> dict | None:
    golds = []
    preds = []
    missing = 0
    for pid in ids:
        pair = oof.get(pid)
        if pair is None:
            missing += 1
            continue
        golds.append(pair[0])
        preds.append(pair[1])
    if not golds:
        return None
    g = np.asarray(golds, dtype=object)
    p = np.asarray(preds, dtype=object)
    g_bin = np.asarray([1 if x in SARCASM_LABELS else 0 for x in g], dtype=int)
    p_bin = np.asarray([1 if x in SARCASM_LABELS else 0 for x in p], dtype=int)
    return {
        "n": int(len(g)),
        "n_missing_oof": int(missing),
        "sarcasm_class_f1": _sarcasm_macro_f1(g, p),
        "binary_f1": float(
            f1_score(g_bin, p_bin, zero_division=0)
        ),
        "note": (
            "Mixed-train OOF. Crafted rows in other folds can still teach clash."
        ),
    }


def evaluate(
    *,
    dataset: Path,
    features: Path,
    baseline_cache: Path,
    oof_path: Path,
) -> dict:
    X, y, ids = _load_mm(dataset, features)
    crafted = np.array([is_crafted(p) for p in ids], dtype=bool)
    organic = ~crafted
    baseline_all = _align_baseline(ids, baseline_cache)
    oof = _load_oof(oof_path)

    def subset(mask: np.ndarray) -> tuple[
        np.ndarray, np.ndarray, list[str], tuple[np.ndarray, np.ndarray] | None
    ]:
        idx = np.flatnonzero(mask)
        sub_ids = [ids[i] for i in idx]
        b = None
        if baseline_all is not None:
            bX, by = baseline_all
            b = (bX[idx], by[idx])
        return X[idx], y[idx], sub_ids, b

    splits = {}
    for name, mask in (
        ("all", np.ones(len(ids), dtype=bool)),
        ("organic", organic),
        ("crafted", crafted),
    ):
        sX, sy, sids, b = subset(mask)
        log.info(
            "origin split %s n=%d sarcasm=%d",
            name,
            len(sy),
            int(sum(1 for v in sy if str(v) in SARCASM_LABELS)),
        )
        rec = _retrain_split(sX, sy, b)
        rec["oof_slice"] = _oof_slice(sids, oof)
        splits[name] = rec

    organic_bin = splits["organic"].get("binary") or {}
    crafted_bin = splits["crafted"].get("binary") or {}
    return {
        "craft_prefix": CRAFT_PREFIX,
        "splits": splits,
        "organic_clash_f1": (organic_bin.get("clash") or {}).get("f1"),
        "crafted_clash_f1": (crafted_bin.get("clash") or {}).get("f1"),
        "organic_rq2_delta": splits["organic"].get("rq2_delta"),
        "organic_rq2_meets_10pp": splits["organic"].get("rq2_meets_10pp"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Organic vs crafted sarcasm split.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--features-cache", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--baseline-cache", type=Path, default=DEFAULT_BASELINE_FEATURES)
    parser.add_argument("--oof", type=Path, default=DEFAULT_OOF)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    payload = evaluate(
        dataset=args.dataset,
        features=args.features_cache,
        baseline_cache=args.baseline_cache,
        oof_path=args.oof,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
