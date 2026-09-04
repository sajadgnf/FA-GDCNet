"""Queue scrape-pool posts for blind sarcasm review.

Does **not** assign sarcasm gold. ParsBERT/CLIP auto-labels are not §5 labels.

Sources, in review order:
1. Gold rows tagged ``weak-sarcasm-bootstrap`` that were never blindly reviewed
2. ``datasets/raw/archive/sarcasm_unlabeled.jsonl`` not already in gold
3. Other unlabeled pools whose caption passes the strict irony/clash heuristic
4. Existing non-sarcasm gold that matches the same strict heuristic

New rows get placeholder ``neutral`` + ``weak-sarcasm-bootstrap`` so eval
excludes them until ``blind-relabel``. Existing eval-eligible gold is only
tagged ``sarcasm-candidate``; labels are unchanged until a human 1–5.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from .relabel import write_ids_file
from .sarcasm_candidates import is_political_caption, is_sarcasm_candidate_caption
from .schema import LABELS
from .scrape import _load_ignored_shortcodes
from .tags import BLIND_REVIEW_TAG, BOOTSTRAP_TAG, CANDIDATE_TAG, CRAFTED_TAG

log = logging.getLogger(__name__)

DEFAULT_DATASET = Path("datasets") / "persian_multimodal_irony.jsonl"
DEFAULT_IDS = Path("datasets") / "sarcasm_candidate_ids.txt"
DEFAULT_UNFILTERED_POOLS = (Path("datasets") / "raw" / "archive" / "sarcasm_unlabeled.jsonl",)
DEFAULT_FILTERED_POOLS = (Path("datasets") / "raw" / "archive" / "hashtags_unlabeled.jsonl",)
PLACEHOLDER_LABEL = "neutral"
SARCASM = frozenset({"positive_sarcasm", "negative_sarcasm"})


@dataclass
class EnqueueResult:
    tagged_existing: int = 0
    imported: int = 0
    queued: int = 0
    skipped_no_image: int = 0
    skipped_already_blind: int = 0
    n_gold: int = 0
    ids: list[str] = field(default_factory=list)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def annotators_of(row: dict) -> list[str]:
    return [str(a) for a in (row.get("annotators") or [])]


def is_blind(row: dict) -> bool:
    return BLIND_REVIEW_TAG in annotators_of(row)


def is_bootstrap_pending(row: dict) -> bool:
    tags = annotators_of(row)
    return BOOTSTRAP_TAG in tags and BLIND_REVIEW_TAG not in tags


def tag_candidate(row: dict) -> bool:
    anns = annotators_of(row)
    if CANDIDATE_TAG in anns:
        return False
    anns.append(CANDIDATE_TAG)
    row["annotators"] = anns
    return True


def image_is_readable(image_path: str, *, roots: list[Path]) -> bool:
    raw = str(image_path or "").strip()
    if not raw:
        return False
    p = Path(raw)
    if p.is_file():
        return True
    for root in roots:
        if (root / p).is_file():
            return True
    return False


def _append_id(ids: list[str], seen: set[str], post_id: str) -> None:
    if post_id and post_id not in seen:
        seen.add(post_id)
        ids.append(post_id)


def _new_record(row: dict) -> dict:
    anns = [BOOTSTRAP_TAG, CANDIDATE_TAG]
    pid = str(row["post_id"])
    if pid.startswith("craft-") and CRAFTED_TAG not in anns:
        anns.append(CRAFTED_TAG)
    return {
        "post_id": pid,
        "caption": str(row.get("caption") or ""),
        "image_path": str(row.get("image_path") or ""),
        "label": PLACEHOLDER_LABEL,
        "annotators": anns,
        "kappa": None,
    }


def enqueue(
    *,
    dataset: Path = DEFAULT_DATASET,
    ids_path: Path = DEFAULT_IDS,
    unfiltered_pools: list[Path] | None = None,
    filtered_pools: list[Path] | None = None,
    include_gold_heuristic: bool = True,
    max_queue: int | None = None,
    from_pools_only: bool = False,
    dry_run: bool = False,
    roots: list[Path] | None = None,
) -> EnqueueResult:
    if PLACEHOLDER_LABEL not in LABELS:
        raise RuntimeError(f"placeholder {PLACEHOLDER_LABEL!r} is not a schema label")
    gold = _load_jsonl(dataset)
    search_roots = list(roots or [Path.cwd(), Path(__file__).resolve().parents[2]])
    result = EnqueueResult(n_gold=len(gold))
    by_id = {str(r.get("post_id")): r for r in gold if r.get("post_id")}
    ignored = _load_ignored_shortcodes()
    ordered_ids: list[str] = []
    seen: set[str] = set()

    def readable(row: dict) -> bool:
        ok = image_is_readable(str(row.get("image_path") or ""), roots=search_roots)
        if not ok:
            result.skipped_no_image += 1
        return ok

    # 1) Unreviewed bootstrap already in gold.
    if not from_pools_only:
        for row in gold:
            pid = str(row.get("post_id") or "")
            if not is_bootstrap_pending(row):
                continue
            if is_political_caption(str(row.get("caption") or "")):
                continue
            if not readable(row):
                continue
            if tag_candidate(row):
                result.tagged_existing += 1
            _append_id(ordered_ids, seen, pid)

    def import_pool(path: Path, *, require_heuristic: bool) -> None:
        for row in _load_jsonl(path):
            pid = str(row.get("post_id") or "")
            if not pid or pid in by_id or pid in ignored:
                continue
            caption = str(row.get("caption") or "")
            if len(caption.strip()) < 4:
                continue
            if is_political_caption(caption):
                continue
            if require_heuristic and not is_sarcasm_candidate_caption(
                caption, allow_weak_cues=False
            ):
                continue
            if not readable(row):
                continue
            rec = _new_record(row)
            by_id[pid] = rec
            gold.append(rec)
            result.imported += 1
            _append_id(ordered_ids, seen, pid)

    for path in unfiltered_pools if unfiltered_pools is not None else list(DEFAULT_UNFILTERED_POOLS):
        import_pool(path, require_heuristic=False)
    for path in filtered_pools if filtered_pools is not None else list(DEFAULT_FILTERED_POOLS):
        import_pool(path, require_heuristic=True)

    # 2) Existing gold that looks like irony but was never blindly reviewed.
    if include_gold_heuristic and not from_pools_only:
        for row in gold:
            pid = str(row.get("post_id") or "")
            if pid in seen:
                continue
            if is_blind(row):
                continue
            if str(row.get("label")) in SARCASM:
                continue
            if is_political_caption(str(row.get("caption") or "")):
                continue
            if not readable(row):
                continue
            if not is_sarcasm_candidate_caption(
                str(row.get("caption") or ""), allow_weak_cues=False
            ):
                continue
            if tag_candidate(row):
                result.tagged_existing += 1
            _append_id(ordered_ids, seen, pid)

    if max_queue is not None:
        ordered_ids = ordered_ids[: max(0, int(max_queue))]

    result.ids = ordered_ids
    result.queued = len(ordered_ids)
    if dry_run:
        log.info(
            "dry-run: would queue %d (tag %d existing, import %d)",
            result.queued,
            result.tagged_existing,
            result.imported,
        )
        return result

    _save_jsonl(dataset, gold)
    write_ids_file(
        ids_path,
        ordered_ids,
        header=(
            "Blind sarcasm-candidate queue. Labels are hidden; 1-5 required.\n"
            f"imported={result.imported} tagged_existing={result.tagged_existing} "
            f"queued={result.queued}\n"
            "New rows are placeholder-neutral + weak-sarcasm-bootstrap until "
            "blind-relabel.\n"
            f"python tasks.py relabel --only candidates --ids-file {ids_path}"
        ),
    )
    log.info(
        "queued %d candidates (%d imported, %d tagged) -> %s",
        result.queued,
        result.imported,
        result.tagged_existing,
        ids_path,
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Queue scrape-pool posts for blind sarcasm review (no auto gold)."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--ids-file", type=Path, default=DEFAULT_IDS)
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        dest="max_queue",
        help="Cap the review queue (default: all sourced candidates).",
    )
    parser.add_argument(
        "--no-gold-heuristic",
        action="store_true",
        help="Do not re-queue existing non-sarcasm gold that matches the caption heuristic.",
    )
    parser.add_argument(
        "--unfiltered-pool",
        action="append",
        type=Path,
        default=None,
        help="Raw JSONL to import without caption filter (repeatable).",
    )
    parser.add_argument(
        "--filtered-pool",
        action="append",
        type=Path,
        default=None,
        help="Raw JSONL to import only if the strict caption heuristic matches.",
    )
    parser.add_argument(
        "--from-pools-only",
        action="store_true",
        help="Do not re-queue existing gold/bootstrap; only import from pools.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if not args.dataset.is_file():
        print(f"dataset not found: {args.dataset}")
        return 1
    filtered = args.filtered_pool
    if args.from_pools_only and filtered is None:
        filtered = []
    result = enqueue(
        dataset=args.dataset,
        ids_path=args.ids_file,
        unfiltered_pools=args.unfiltered_pool,
        filtered_pools=filtered,
        include_gold_heuristic=not args.no_gold_heuristic,
        from_pools_only=args.from_pools_only,
        max_queue=args.max_queue,
        dry_run=args.dry_run,
    )
    print(
        f"queued={result.queued} imported={result.imported} "
        f"tagged_existing={result.tagged_existing} "
        f"skipped_no_image={result.skipped_no_image} gold={result.n_gold}"
    )
    if not args.dry_run:
        print("This is not sarcasm gold. Review with:")
        print(f"  python tasks.py relabel --only candidates --ids-file {args.ids_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
