"""Tests for embed-based link import."""

from __future__ import annotations

import json
from pathlib import Path

from data.scrape_embed import PostRef, import_from_links, load_shortcodes, normalize_post_refs
from data.scrape_embed import _caption_from_og_description


def test_load_shortcodes(tmp_path: Path) -> None:
    p = tmp_path / "links.txt"
    p.write_text(
        "# comment\n"
        "https://www.instagram.com/p/ABCdef12345/\n"
        "XYZ98765\n",
        encoding="utf-8",
    )
    assert load_shortcodes(p) == ["ABCdef12345", "XYZ98765"]


def test_caption_from_og_description() -> None:
    desc = '1,234 likes, 45 comments - user on January 1, 2024: "سلفی امروز"'
    assert _caption_from_og_description(desc) == "سلفی امروز"


def test_normalize_post_refs() -> None:
    refs = normalize_post_refs(["ABC", PostRef("XYZ", kind="reel")])
    assert refs == [PostRef("ABC", "p"), PostRef("XYZ", "reel")]


def test_import_from_links_mocked(tmp_path: Path, monkeypatch) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    links = raw / "links.txt"
    links.write_text("https://www.instagram.com/p/NEWPOST1/\n", encoding="utf-8")

    monkeypatch.setattr("data.scrape_embed.DEFAULT_RAW_DIR", raw)
    monkeypatch.setattr("data.scrape.DEFAULT_RAW_DIR", raw)
    monkeypatch.setattr(
        "data.scrape.IGNORED_IDS_FILE",
        raw / "ignored_post_ids.txt",
    )

    def fake_fetch(sc: str, *, timeout: float = 30.0) -> tuple[str, str]:
        return "یک پست تست فارسی", "https://example.com/img.jpg"

    def fake_download(url: str, dest: Path, *, timeout: float = 60.0) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"fake")

    monkeypatch.setattr("data.scrape_embed.fetch_embed_post", fake_fetch)
    monkeypatch.setattr("data.scrape_embed._download_image", fake_download)

    n = import_from_links(links, pool_name="tags", out_dir=raw, require_face=False)
    assert n == 1
    rows = [json.loads(l) for l in (raw / "tags.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["post_id"] == "NEWPOST1"
