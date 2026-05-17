"""5-class CLI annotation tool with skip/undo and incremental IAA reporting.

Each labeling action emits ONE JSON line per record to two files:
- `datasets/persian_multimodal_irony.jsonl` (the canonical labeled dataset)
- `datasets/annotations_raw.jsonl` (the audit log: every annotator decision,
  including duplicates by different annotators; used to compute IAA).

The canonical dataset keeps a single record per `post_id` with all annotators
merged; the raw log preserves the per-annotator audit trail.

Image display is deliberately a no-op text path: we surface the image path so
the annotator can open it in their preferred viewer (instagram thumbnails are
PNG/JPG and OS file viewers handle them natively). For an in-terminal preview,
a future iteration could shell out to `kitty +kitten icat` or similar.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator

from .iaa import compute_and_write
from .schema import LABELS

log = logging.getLogger(__name__)

DEFAULT_DATASET = Path("datasets") / "persian_multimodal_irony.jsonl"
DEFAULT_RAW_LOG = Path("datasets") / "annotations_raw.jsonl"
DEFAULT_REPORT = Path("reports") / "iaa.md"

LABEL_KEYS: dict[str, str] = {str(i + 1): label for i, label in enumerate(LABELS)}


def _iter_input_pool(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _existing_audit(raw_log: Path) -> list[dict]:
    if not raw_log.exists():
        return []
    rows: list[dict] = []
    with raw_log.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _build_canonical(audit_rows: Iterable[dict], pool: dict[str, dict]) -> list[dict]:
    """Collapse the per-annotator audit log into one record per post_id.

    The chosen label is the majority vote, breaking ties by latest timestamp.
    """
    per_post: dict[str, list[dict]] = defaultdict(list)
    for row in audit_rows:
        per_post[row["post_id"]].append(row)
    out: list[dict] = []
    for post_id, rows in per_post.items():
        if post_id not in pool:
            continue
        counts: dict[str, int] = defaultdict(int)
        for r in rows:
            counts[r["label"]] += 1
        max_count = max(counts.values())
        candidates = [lbl for lbl, c in counts.items() if c == max_count]
        if len(candidates) == 1:
            label = candidates[0]
        else:
            label = sorted(
                (r for r in rows if r["label"] in candidates),
                key=lambda r: r["timestamp"],
                reverse=True,
            )[0]["label"]
        annotators = sorted({r["annotator_id"] for r in rows})
        out.append(
            {
                "post_id": post_id,
                "caption": pool[post_id]["caption"],
                "image_path": pool[post_id]["image_path"],
                "label": label,
                "annotators": annotators,
                "kappa": None,
            }
        )
    return out


def _persist_canonical(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _prompt_one(pool_entry: dict) -> str | None:
    """Return label key, "s" for skip, "u" for undo, or None for quit."""
    print()
    print("=" * 70)
    print(f"post_id : {pool_entry['post_id']}")
    print(f"image   : {pool_entry['image_path']}")
    print(f"caption : {pool_entry['caption']}")
    print("-" * 70)
    print("Choose a label:")
    for k, lbl in LABEL_KEYS.items():
        print(f"  {k}) {lbl}")
    print("  s) skip   u) undo last   q) quit")
    while True:
        try:
            choice = input("> ").strip().lower()
        except EOFError:
            return None
        if choice in LABEL_KEYS:
            return choice
        if choice in {"s", "u", "q"}:
            return choice
        print("invalid choice, try again.")


def _select_pool_path(args: argparse.Namespace) -> Path:
    if args.input:
        return Path(args.input)
    raw_dir = Path("datasets") / "raw"
    if not raw_dir.exists():
        return Path("datasets") / "raw" / "all.jsonl"
    jsonls = sorted(raw_dir.glob("*.jsonl"))
    if not jsonls:
        return raw_dir / "all.jsonl"
    if len(jsonls) == 1:
        return jsonls[0]
    print("Available raw pools:")
    for i, p in enumerate(jsonls):
        print(f"  [{i}] {p}")
    while True:
        try:
            idx = int(input("Select pool index: ").strip())
            return jsonls[idx]
        except (ValueError, IndexError):
            print("invalid index, try again.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="5-class Persian multimodal labeling tool.")
    parser.add_argument(
        "--annotator",
        default=os.environ.get("USER") or os.environ.get("USERNAME") or "anonymous",
    )
    parser.add_argument("--input", default=None, help="Path to a raw pool JSONL.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--raw-log", type=Path, default=DEFAULT_RAW_LOG)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    pool_path = _select_pool_path(args)
    pool_rows = list(_iter_input_pool(pool_path))
    pool: dict[str, dict] = {r["post_id"]: r for r in pool_rows}
    if not pool:
        print(f"no posts found in {pool_path}; run `tasks.py scrape` first.")
        return 1

    audit_rows = _existing_audit(args.raw_log)
    already_done = {r["post_id"] for r in audit_rows if r["annotator_id"] == args.annotator}
    pending = [r for r in pool_rows if r["post_id"] not in already_done]
    if not pending:
        print(f"annotator {args.annotator!r} has already labeled every post in this pool.")
        return 0

    print(f"loaded {len(pending)} pending posts for annotator {args.annotator!r}.")
    last_written_offset: int | None = None
    args.raw_log.parent.mkdir(parents=True, exist_ok=True)
    with args.raw_log.open("a", encoding="utf-8") as raw_out:
        for entry in pending:
            choice = _prompt_one(entry)
            if choice is None or choice == "q":
                break
            if choice == "s":
                continue
            if choice == "u":
                if last_written_offset is None:
                    print("nothing to undo.")
                    continue
                # Truncate the raw log to before the last write.
                raw_out.flush()
                with args.raw_log.open("rb+") as f:
                    f.seek(last_written_offset)
                    f.truncate()
                audit_rows = _existing_audit(args.raw_log)
                print("last record removed.")
                last_written_offset = None
                continue
            label = LABEL_KEYS[choice]
            row = {
                "post_id": entry["post_id"],
                "label": label,
                "annotator_id": args.annotator,
                "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            }
            line = json.dumps(row, ensure_ascii=False) + "\n"
            last_written_offset = raw_out.tell()
            raw_out.write(line)
            raw_out.flush()
            audit_rows.append(row)

    canonical = _build_canonical(audit_rows, pool)
    _persist_canonical(canonical, args.dataset)
    compute_and_write(args.raw_log, args.report)
    print(f"\nwrote {len(canonical)} records to {args.dataset}")
    print(f"wrote IAA report to {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
