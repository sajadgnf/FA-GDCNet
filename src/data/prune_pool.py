"""Archive unlabeled posts, remove them from an active pool, and block re-scrape."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

from . import scrape as scrape_mod

log = logging.getLogger(__name__)

DEFAULT_RAW_LOG = Path("datasets") / "annotations_raw.jsonl"
DEFAULT_ARCHIVE_DIR = Path("datasets") / "raw" / "archive"


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _labeled_ids(raw_log: Path, annotator: str) -> set[str]:
    return {
        row["post_id"]
        for row in _load_jsonl(raw_log)
        if row.get("annotator_id") == annotator
    }


def _append_ignored(shortcodes: set[str]) -> int:
    known = scrape_mod._load_ignored_shortcodes()
    new_ids = sorted(shortcodes - known)
    if not new_ids:
        return 0
    ignore_file = scrape_mod.IGNORED_IDS_FILE
    ignore_file.parent.mkdir(parents=True, exist_ok=True)
    with ignore_file.open("a", encoding="utf-8") as f:
        for sc in new_ids:
            f.write(sc + "\n")
    return len(new_ids)


def _move_image(src: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if src.resolve() == dest.resolve():
        return dest
    if dest.exists():
        src.unlink(missing_ok=True)
        return dest
    shutil.move(str(src), str(dest))
    return dest


def prune_pool(
    pool_path: Path,
    *,
    annotator: str,
    raw_log: Path = DEFAULT_RAW_LOG,
    archive_dir: Path = DEFAULT_ARCHIVE_DIR,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Move unlabeled rows to archive, shrink pool, add IDs to ignore list."""
    pool_rows = _load_jsonl(pool_path)
    if not pool_rows:
        log.info("pool %s is empty", pool_path)
        return 0, 0

    labeled = _labeled_ids(raw_log, annotator)
    kept: list[dict] = []
    archived: list[dict] = []

    pool_stem = pool_path.stem
    archive_jsonl = archive_dir / f"{pool_stem}_unlabeled.jsonl"
    archive_image_dir = archive_dir / pool_stem

    for row in pool_rows:
        post_id = row["post_id"]
        if post_id in labeled:
            kept.append(row)
            continue
        archived.append(row)

    if not archived:
        log.info("no unlabeled posts in %s for annotator %r", pool_path, annotator)
        return 0, len(kept)

    if dry_run:
        log.info(
            "dry-run: would archive %d, keep %d in %s",
            len(archived),
            len(kept),
            pool_path,
        )
        return len(archived), len(kept)

    archive_dir.mkdir(parents=True, exist_ok=True)
    archived_ids: set[str] = set()

    with archive_jsonl.open("a", encoding="utf-8") as out:
        for row in archived:
            image_path = Path(row["image_path"])
            if image_path.is_file():
                new_path = _move_image(image_path, archive_image_dir)
                row = {**row, "image_path": str(new_path)}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            archived_ids.add(row["post_id"])

    with pool_path.open("w", encoding="utf-8") as out:
        for row in kept:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    added = _append_ignored(archived_ids)
    log.info(
        "archived %d unlabeled posts to %s",
        len(archived),
        archive_jsonl,
    )
    log.info("kept %d labeled posts in %s", len(kept), pool_path)
    log.info("added %d IDs to %s", added, scrape_mod.IGNORED_IDS_FILE)
    return len(archived), len(kept)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Archive unlabeled pool rows and block them from future scrapes.",
    )
    parser.add_argument("--input", type=Path, required=True, help="Raw pool JSONL to prune.")
    parser.add_argument(
        "--annotator",
        default=os.environ.get("USER") or os.environ.get("USERNAME") or "anonymous",
    )
    parser.add_argument("--raw-log", type=Path, default=DEFAULT_RAW_LOG)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    archived, kept = prune_pool(
        args.input,
        annotator=args.annotator,
        raw_log=args.raw_log,
        archive_dir=args.archive_dir,
        dry_run=args.dry_run,
    )
    if archived == 0 and kept == 0 and not args.input.exists():
        log.error("pool not found: %s", args.input)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
