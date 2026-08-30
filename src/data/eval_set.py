"""Which labeled rows may enter train / eval.

``weak-sarcasm-bootstrap`` rows are weak tags, not §5 gold. They are excluded
until a blind human pass records ``blind-relabel`` on that row.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .schema import DatasetRecord, iter_dataset
from .tags import BLIND_REVIEW_TAG, BOOTSTRAP_TAG

log = logging.getLogger(__name__)

DEFAULT_DATASET = Path("datasets") / "persian_multimodal_irony.jsonl"


def is_eval_eligible(annotators: list | None) -> bool:
    """True unless the row is bootstrap without a later blind review."""
    tags = [str(a) for a in (annotators or [])]
    if BOOTSTRAP_TAG not in tags:
        return True
    return BLIND_REVIEW_TAG in tags


def eligible_records(dataset: Path) -> list[DatasetRecord]:
    return [r for r in iter_dataset(dataset) if is_eval_eligible(r.annotators)]


def slice_to_eval_set(
    dataset: Path,
    X: np.ndarray,
    y: np.ndarray,
    post_ids: list[str] | None,
) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    """Drop ineligible cache rows. ``y`` is taken from jsonl, not the cache.

    Returns ``(X, y, post_ids, n_excluded_bootstrap)``.
    """
    if post_ids is None:
        raise ValueError(
            "feature cache has no post_ids; re-extract so eval can drop bootstrap rows"
        )
    if len(post_ids) != len(X) or len(post_ids) != len(y):
        raise ValueError(
            f"cache length mismatch X={len(X)} y={len(y)} post_ids={len(post_ids)}"
        )
    by_id = {r.post_id: r for r in iter_dataset(dataset)}
    keep_x: list[np.ndarray] = []
    keep_y: list[str] = []
    keep_ids: list[str] = []
    n_excluded = 0
    n_missing = 0
    for i, pid in enumerate(post_ids):
        rec = by_id.get(str(pid))
        if rec is None:
            n_missing += 1
            continue
        if not is_eval_eligible(rec.annotators):
            n_excluded += 1
            continue
        keep_x.append(X[i])
        keep_y.append(str(rec.label))
        keep_ids.append(str(pid))
    if n_missing:
        log.warning("eval set: %d cache ids missing from %s", n_missing, dataset)
    if n_excluded:
        log.info("eval set: excluded %d %s rows", n_excluded, BOOTSTRAP_TAG)
    if not keep_x:
        raise ValueError(f"no eval-eligible rows in cache vs {dataset}")
    return (
        np.stack(keep_x, axis=0),
        np.asarray(keep_y, dtype=object),
        keep_ids,
        n_excluded,
    )


def slice_records_from_cache(
    cache: Path,
    records: list[DatasetRecord],
    *,
    x_key: str = "X",
    y_key: str = "y",
) -> tuple[np.ndarray, np.ndarray] | None:
    """Index a feature cache by ``records`` order. Labels come from ``records``.

    Returns None when any record is missing from the cache (caller must recompute).
    Extra cache rows (e.g. bootstrap) are ignored.
    """
    if not cache.is_file() or not records:
        return None
    npz = np.load(cache, allow_pickle=True)
    if "post_ids" not in npz or x_key not in npz:
        return None
    by_id = {str(p): i for i, p in enumerate(npz["post_ids"].tolist())}
    xs: list[np.ndarray] = []
    ys: list[str] = []
    n_missing = 0
    for rec in records:
        idx = by_id.get(rec.post_id)
        if idx is None:
            n_missing += 1
            continue
        xs.append(np.asarray(npz[x_key][idx]))
        ys.append(str(rec.label))
    if n_missing:
        log.warning(
            "eval cache missing %d/%d eligible ids; using %d cached rows",
            n_missing,
            len(records),
            len(xs),
        )
    if not xs:
        return None
    return np.stack(xs, axis=0), np.asarray(ys, dtype=object)
