"""Tests for local pool import."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from data.import_pool import import_local_pool


def test_import_local_pool_skips_ignored(tmp_path: Path, monkeypatch) -> None:
    raw = tmp_path / "raw"
    inbox = raw / "inbox"
    inbox.mkdir(parents=True)
    ignore = raw / "ignored_post_ids.txt"
    ignore.write_text("seen123\n", encoding="utf-8")
    pool_jsonl = raw / "tags.jsonl"
    pool_jsonl.write_text(
        json.dumps(
            {
                "post_id": "seen123",
                "caption": "قبلا",
                "image_path": str(raw / "tags/seen123.jpg"),
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    Image.new("RGB", (32, 32), color="red").save(inbox / "seen123.jpg")
    Image.new("RGB", (32, 32), color="blue").save(inbox / "newpost.jpg")
    (inbox / "newpost.txt").write_text("پست جدید برای تست", encoding="utf-8")

    monkeypatch.setattr("data.import_pool.DEFAULT_RAW_DIR", raw)
    monkeypatch.setattr("data.scrape.DEFAULT_RAW_DIR", raw)
    monkeypatch.setattr("data.scrape.IGNORED_IDS_FILE", ignore)

    n = import_local_pool(inbox, pool_name="tags", out_dir=raw, require_face=False)
    assert n == 1
    rows = [json.loads(l) for l in pool_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 2
    assert rows[-1]["post_id"] == "newpost"
