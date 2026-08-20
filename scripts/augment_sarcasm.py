"""Add weak-labeled sarcasm candidates from the raw archive to the main dataset.

The archive pool (`datasets/raw/archive/sarcasm_unlabeled.jsonl`) holds posts
collected for irony cues but never manually labeled. We assign:

- `positive_sarcasm` when ParsBERT polarity favours positive
- `negative_sarcasm` otherwise

Only records with a readable image and a novel `post_id` are appended.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from data.schema import DatasetRecord, iter_dataset, write_dataset

log = logging.getLogger(__name__)

DEFAULT_DATASET = Path("datasets") / "persian_multimodal_irony.jsonl"
DEFAULT_POOL = Path("datasets") / "raw" / "archive" / "sarcasm_unlabeled.jsonl"


def _load_pool(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def augment(
    *,
    dataset_path: Path,
    pool_path: Path,
    dry_run: bool = False,
) -> int:
    existing = {r.post_id: r for r in iter_dataset(dataset_path)}
    pool = _load_pool(pool_path)

    from inference.models import load_polarity_only, polarity_probs

    bundle = load_polarity_only()
    added: list[DatasetRecord] = []
    for row in pool:
        post_id = str(row.get("post_id", ""))
        if not post_id or post_id in existing:
            continue
        image_path = str(row.get("image_path", ""))
        if not Path(image_path).is_file():
            continue
        caption = str(row.get("caption") or "").strip()
        if len(caption) < 4:
            continue

        probs = polarity_probs(bundle, caption)
        label = "positive_sarcasm" if float(probs[1]) >= float(probs[0]) else "negative_sarcasm"
        added.append(
            DatasetRecord(
                post_id=post_id,
                caption=caption,
                image_path=image_path,
                label=label,
                annotators=["weak-sarcasm-bootstrap"],
            )
        )

    if not added:
        log.info("no new sarcasm records to add")
        return 0

    log.info("adding %d weak-labeled sarcasm records", len(added))
    if dry_run:
        for r in added[:5]:
            log.info("  sample %s -> %s", r.post_id, r.label)
        return len(added)

    all_records = list(existing.values()) + added
    write_dataset(dataset_path, all_records)
    log.info("dataset now has %d records at %s", len(all_records), dataset_path)
    return len(added)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    augment(dataset_path=args.dataset, pool_path=args.pool, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
