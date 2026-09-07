"""Binary sarcasm detection metrics (proposal Hypothesis 2: accuracy ≥ 70%).

The proposal targets *detecting multimodal irony/sarcasm*, not separating all
five sentiment classes. This module trains a binary LogisticRegression on the
GDRM features (including the polarity-product ``clash`` column) and reports
accuracy / F1 for sarcasm vs non-sarcasm.

The interpretable rule is a CV-tuned cut on ``clash`` that maximizes
**accuracy** (PDF §6.3(2) is an accuracy hypothesis). Precision / recall / F1
are still reported. A Dsem accuracy cut is written for the dummy-trap footnote:
Dsem is anti-correlated with face–caption sarcasm in this dataset.

Writes `reports/sarcasm.csv`. Footer ``pdf_accuracy_bar`` is the letter of
§6.3(2) (clash accuracy ≥ 70%). ``beats_dummy_accuracy`` is whether that
accuracy exceeds always-not-sarcasm. Those are different claims.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

from data.eval_set import slice_to_eval_set
from inference.classifier import DEFAULT_DATASET, DEFAULT_FEATURES, compute_dataset_features
from inference.gdrm import FEATURE_NAMES, features_for_eval, with_clash_column

log = logging.getLogger(__name__)

DEFAULT_SARCASM_CSV = Path("reports") / "sarcasm.csv"
SARCASM_LABELS = frozenset({"positive_sarcasm", "negative_sarcasm"})
ACCURACY_FLOOR = 0.70
DETECTION_F1_FLOOR = 0.40
_CLASH_IDX = FEATURE_NAMES.index("clash")
_DSEM_IDX = FEATURE_NAMES.index("Dsem")


def _to_binary(labels: np.ndarray) -> np.ndarray:
    return np.asarray([1 if lbl in SARCASM_LABELS else 0 for lbl in labels], dtype=int)


def _load_features(dataset: Path, cache: Path) -> tuple[np.ndarray, np.ndarray]:
    if cache.exists():
        npz = np.load(cache, allow_pickle=True)
        ids = [str(x) for x in npz["post_ids"].tolist()] if "post_ids" in npz else None
        X, y, ids, _ = slice_to_eval_set(
            dataset, with_clash_column(npz["X"]), npz["y"], ids
        )
        return features_for_eval(X, ids, dataset), y
    X, y, ids = compute_dataset_features(dataset, cache_path=cache)
    X, y, ids, _ = slice_to_eval_set(dataset, with_clash_column(X), y, list(ids))
    return features_for_eval(X, ids, dataset), y


def _build_clf() -> LogisticRegression:
    return LogisticRegression(
        class_weight="balanced",
        penalty="l2",
        solver="lbfgs",
        max_iter=2000,
        random_state=0,
    )


def _threshold_preds(scores: np.ndarray, threshold: float) -> np.ndarray:
    return (scores >= threshold).astype(int)


def pick_score_threshold(
    scores: np.ndarray,
    y_bin: np.ndarray,
    *,
    metric: str = "f1",
    low: float = -0.6,
    high: float = 0.8,
    n: int = 141,
) -> float:
    """Pick a cut on ``scores`` that maximizes training F1 (or accuracy)."""
    best_t = 0.0
    best_val = -1.0
    y_bin = np.asarray(y_bin, dtype=int)
    scores = np.asarray(scores, dtype=np.float64)
    for t in np.linspace(low, high, n):
        pred = _threshold_preds(scores, float(t))
        if metric == "accuracy":
            val = float(np.mean(pred == y_bin))
        else:
            val = float(f1_score(y_bin, pred, zero_division=0))
        if val > best_val:
            best_val = val
            best_t = float(t)
    return best_t


def evaluate(X: np.ndarray, y: np.ndarray, *, n_splits: int = 5) -> dict:
    y_bin = _to_binary(y)
    X = with_clash_column(X)
    clash = X[:, _CLASH_IDX]
    dsem = X[:, _DSEM_IDX]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    accs: list[float] = []
    f1s: list[float] = []
    precs: list[float] = []
    recs: list[float] = []
    clash_accs: list[float] = []
    clash_f1s: list[float] = []
    clash_precs: list[float] = []
    clash_recs: list[float] = []
    dsem_accs: list[float] = []
    dsem_f1s: list[float] = []
    clash_thresholds: list[float] = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y_bin), start=1):
        clf = _build_clf()
        clf.fit(X[train_idx], y_bin[train_idx])
        preds = clf.predict(X[test_idx])
        true = y_bin[test_idx]
        accs.append(accuracy_score(true, preds))
        f1s.append(f1_score(true, preds, zero_division=0))
        precs.append(precision_score(true, preds, zero_division=0))
        recs.append(recall_score(true, preds, zero_division=0))

        t_clash = pick_score_threshold(
            clash[train_idx], y_bin[train_idx], metric="accuracy"
        )
        clash_thresholds.append(t_clash)
        clash_pred = _threshold_preds(clash[test_idx], t_clash)
        clash_accs.append(accuracy_score(true, clash_pred))
        clash_f1s.append(f1_score(true, clash_pred, zero_division=0))
        clash_precs.append(precision_score(true, clash_pred, zero_division=0))
        clash_recs.append(recall_score(true, clash_pred, zero_division=0))

        t_dsem = pick_score_threshold(
            dsem[train_idx],
            y_bin[train_idx],
            metric="accuracy",
            low=0.25,
            high=0.65,
            n=81,
        )
        dsem_pred = _threshold_preds(dsem[test_idx], t_dsem)
        dsem_accs.append(accuracy_score(true, dsem_pred))
        dsem_f1s.append(f1_score(true, dsem_pred, zero_division=0))

        log.info(
            "binary sarcasm fold %d logreg acc=%.3f f1=%.3f clash@%.2f acc=%.3f f1=%.3f dsem-acc-cut acc=%.3f",
            fold_idx,
            accs[-1],
            f1s[-1],
            t_clash,
            clash_accs[-1],
            clash_f1s[-1],
            dsem_accs[-1],
        )
    dummy_acc = float(1.0 - np.mean(y_bin)) if len(y_bin) else 0.0
    return {
        "folds": list(range(1, n_splits + 1)),
        "accuracy": accs,
        "f1": f1s,
        "precision": precs,
        "recall": recs,
        "clash_rule_accuracy": clash_accs,
        "clash_rule_f1": clash_f1s,
        "clash_rule_precision": clash_precs,
        "clash_rule_recall": clash_recs,
        "dsem_rule_accuracy": dsem_accs,
        "dsem_rule_f1": dsem_f1s,
        "clash_thresholds": clash_thresholds,
        "dummy_not_sarcasm_accuracy": dummy_acc,
        "n_samples": int(len(y_bin)),
        "n_sarcasm": int(np.sum(y_bin)),
    }


def write_csv(result: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    clash_acc_mu = float(np.mean(result["clash_rule_accuracy"]))
    clash_f1_mu = float(np.mean(result["clash_rule_f1"]))
    clash_p_mu = float(np.mean(result["clash_rule_precision"]))
    clash_r_mu = float(np.mean(result["clash_rule_recall"]))
    dsem_acc_mu = float(np.mean(result["dsem_rule_accuracy"]))
    logreg_mu = float(np.mean(result["accuracy"]))
    f1_mu = float(np.mean(result["f1"]))
    p_mu = float(np.mean(result["precision"]))
    r_mu = float(np.mean(result["recall"]))
    dummy_acc = float(result.get("dummy_not_sarcasm_accuracy") or 0.0)
    pdf_bar = clash_acc_mu >= ACCURACY_FLOOR
    beats_dummy = clash_acc_mu > dummy_acc + 1e-6
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "fold",
                "clash_rule_accuracy",
                "clash_rule_precision",
                "clash_rule_recall",
                "clash_rule_f1",
                "dsem_rule_accuracy",
                "logreg_accuracy",
                "logreg_precision",
                "logreg_recall",
                "f1",
            ]
        )
        for i, fold in enumerate(result["folds"]):
            w.writerow(
                [
                    fold,
                    result["clash_rule_accuracy"][i],
                    result["clash_rule_precision"][i],
                    result["clash_rule_recall"][i],
                    result["clash_rule_f1"][i],
                    result["dsem_rule_accuracy"][i],
                    result["accuracy"][i],
                    result["precision"][i],
                    result["recall"][i],
                    result["f1"][i],
                ]
            )
        w.writerow(
            [
                "mean±std",
                f"{clash_acc_mu:.4f}±{float(np.std(result['clash_rule_accuracy'])):.4f}",
                f"{clash_p_mu:.4f}±{float(np.std(result['clash_rule_precision'])):.4f}",
                f"{clash_r_mu:.4f}±{float(np.std(result['clash_rule_recall'])):.4f}",
                f"{clash_f1_mu:.4f}±{float(np.std(result['clash_rule_f1'])):.4f}",
                f"{dsem_acc_mu:.4f}±{float(np.std(result['dsem_rule_accuracy'])):.4f}",
                f"{logreg_mu:.4f}±{float(np.std(result['accuracy'])):.4f}",
                f"{p_mu:.4f}±{float(np.std(result['precision'])):.4f}",
                f"{r_mu:.4f}±{float(np.std(result['recall'])):.4f}",
                f"{f1_mu:.4f}±{float(np.std(result['f1'])):.4f}",
            ]
        )
        w.writerow([])
        w.writerow(["# mean_accuracy_clash_rule", f"{clash_acc_mu:.4f}"])
        w.writerow(["# mean_precision_clash_rule", f"{clash_p_mu:.4f}"])
        w.writerow(["# mean_recall_clash_rule", f"{clash_r_mu:.4f}"])
        w.writerow(["# mean_f1_clash_rule", f"{clash_f1_mu:.4f}"])
        w.writerow(["# mean_accuracy_dsem_rule", f"{dsem_acc_mu:.4f}"])
        w.writerow(["# mean_accuracy_logreg", f"{logreg_mu:.4f}"])
        w.writerow(["# dummy_not_sarcasm_accuracy", f"{dummy_acc:.4f}"])
        w.writerow(["# pdf_accuracy_bar", "true" if pdf_bar else "false"])
        w.writerow(["# beats_dummy_accuracy", "true" if beats_dummy else "false"])
        # Letter of PDF §6.3(2): accuracy > 70%. Not detection vs the dummy.
        w.writerow(["# meets_hypothesis_70pct", "true" if pdf_bar else "false"])
        w.writerow(["# detection_f1_floor", f"{DETECTION_F1_FLOOR:.2f}"])
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
