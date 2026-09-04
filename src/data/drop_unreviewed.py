"""Remove unlabeled sarcasm-candidate rows from gold and raw pools."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

from .enqueue_sarcasm import _load_jsonl, _save_jsonl, annotators_of, is_blind
from .relabel import read_ids_file
from .scrape import IGNORED_IDS_FILE, _load_ignored_shortcodes
from .tags import BOOTSTRAP_TAG

log = logging.getLogger(__name__)

DEFAULT_DATASET = Path("datasets") / "persian_multimodal_irony.jsonl"
DEFAULT_POOLS = (Path("datasets") / "raw" / "sarcasm.jsonl",)


@dataclass
class DropResult:
    dropped_gold: int = 0
    kept_gold: int = 0
    dropped_pool: int = 0
    ignored: int = 0
    images_deleted: int = 0
    ids: list[str] = field(default_factory=list)


def _append_ignored(shortcodes: set[str]) -> int:
    known = _load_ignored_shortcodes()
    new_ids = sorted(shortcodes - known)
    if not new_ids:
        return 0
    IGNORED_IDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with IGNORED_IDS_FILE.open("a", encoding="utf-8") as fh:
        for sc in new_ids:
            fh.write(sc + "\n")
    return len(new_ids)


def is_unreviewed_bootstrap(row: dict) -> bool:
    tags = annotators_of(row)
    return BOOTSTRAP_TAG in tags and not is_blind(row)


def drop_unreviewed(
    *,
    dataset: Path = DEFAULT_DATASET,
    ids: list[str] | None = None,
    pools: list[Path] | None = None,
    delete_images: bool = True,
    dry_run: bool = False,
) -> DropResult:
    """Drop placeholder-bootstrap rows that never got a 1–5 blind label.

    Existing human gold (even if skipped in a later queue) is kept.
    Dropped ids are added to the scrape ignore list so they are not re-imported.
    """
    gold = _load_jsonl(dataset)
    restrict = set(ids) if ids is not None else None
    result = DropResult()
    kept: list[dict] = []
    drop_ids: set[str] = set()
    drop_images: list[Path] = []

    for row in gold:
        pid = str(row.get("post_id") or "")
        if is_unreviewed_bootstrap(row) and (restrict is None or pid in restrict):
            drop_ids.add(pid)
            raw = str(row.get("image_path") or "").strip()
            if raw:
                drop_images.append(Path(raw))
            continue
        kept.append(row)

    result.ids = sorted(drop_ids)
    result.dropped_gold = len(drop_ids)
    result.kept_gold = len(kept)
    if dry_run:
        log.info("dry-run: would drop %d gold rows, keep %d", result.dropped_gold, result.kept_gold)
        return result

    _save_jsonl(dataset, kept)
    used_images = {str(Path(str(r.get("image_path") or ""))) for r in kept}
    if delete_images:
        for path in drop_images:
            if str(path) in used_images:
                continue
            if path.is_file():
                path.unlink()
                result.images_deleted += 1

    for pool in pools if pools is not None else list(DEFAULT_POOLS):
        if not pool.is_file():
            continue
        rows = _load_jsonl(pool)
        stay = [r for r in rows if str(r.get("post_id") or "") not in drop_ids]
        result.dropped_pool += len(rows) - len(stay)
        _save_jsonl(pool, stay)

    result.ignored = _append_ignored(drop_ids)
    log.info(
        "dropped %d unlabeled bootstrap from %s (kept %d, pool-%d, ignore+%d, images-%d)",
        result.dropped_gold,
        dataset,
        result.kept_gold,
        result.dropped_pool,
        result.ignored,
        result.images_deleted,
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Delete unlabeled bootstrap candidates from gold (keep human labels)."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--ids-file",
        type=Path,
        default=None,
        help="Only drop ids listed here (default: all unreviewed bootstrap).",
    )
    parser.add_argument(
        "--pool",
        action="append",
        type=Path,
        default=None,
        help="Raw pool JSONL to strip dropped ids from (repeatable).",
    )
    parser.add_argument("--keep-images", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if not args.dataset.is_file():
        print(f"dataset not found: {args.dataset}")
        return 1
    ids = read_ids_file(args.ids_file) if args.ids_file else None
    result = drop_unreviewed(
        dataset=args.dataset,
        ids=ids,
        pools=args.pool,
        delete_images=not args.keep_images,
        dry_run=args.dry_run,
    )
    print(
        f"dropped_gold={result.dropped_gold} kept_gold={result.kept_gold} "
        f"dropped_pool={result.dropped_pool} ignored={result.ignored} "
        f"images_deleted={result.images_deleted}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
