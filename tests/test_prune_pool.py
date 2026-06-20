"""Tests for pool pruning / archive."""

from __future__ import annotations

import json
from pathlib import Path

from data.prune_pool import prune_pool


def test_prune_archives_unlabeled_and_updates_ignore(tmp_path: Path) -> None:
    pool = tmp_path / "hashtags.jsonl"
    raw_log = tmp_path / "annotations_raw.jsonl"
    archive_dir = tmp_path / "archive"
    ignore_file = tmp_path / "ignored_post_ids.txt"

    img_a = tmp_path / "hashtags" / "AAA.jpg"
    img_b = tmp_path / "hashtags" / "BBB.jpg"
    img_a.parent.mkdir(parents=True)
    img_a.write_bytes(b"x")
    img_b.write_bytes(b"y")

    pool.write_text(
        "\n".join([
            json.dumps({"post_id": "AAA", "caption": "a", "image_path": str(img_a)}),
            json.dumps({"post_id": "BBB", "caption": "b", "image_path": str(img_b)}),
        ])
        + "\n",
        encoding="utf-8",
    )
    raw_log.write_text(
        json.dumps({
            "post_id": "AAA",
            "label": "neutral",
            "annotator_id": "alice",
            "timestamp": "2026-01-01T00:00:00+00:00",
        })
        + "\n",
        encoding="utf-8",
    )

    import data.scrape as scrape_mod

    old_ignore = scrape_mod.IGNORED_IDS_FILE
    scrape_mod.IGNORED_IDS_FILE = ignore_file
    try:
        archived, kept = prune_pool(
            pool,
            annotator="alice",
            raw_log=raw_log,
            archive_dir=archive_dir,
        )
    finally:
        scrape_mod.IGNORED_IDS_FILE = old_ignore

    assert archived == 1
    assert kept == 1
    rows = [json.loads(l) for l in pool.read_text(encoding="utf-8").splitlines() if l]
    assert [r["post_id"] for r in rows] == ["AAA"]
    assert (archive_dir / "hashtags_unlabeled.jsonl").is_file()
    assert not img_b.exists()
    assert (archive_dir / "hashtags" / "BBB.jpg").is_file()
    assert "BBB" in ignore_file.read_text(encoding="utf-8")
