"""Tests for hashtag file parsing (exact vs partial search)."""

from __future__ import annotations

from pathlib import Path

from data.scrape import HashtagSpec, _load_hashtag_specs


def test_load_hashtag_specs_exact_and_search(tmp_path: Path) -> None:
    p = tmp_path / "tags.txt"
    p.write_text(
        "# comment\n"
        "تیکه_سنگین\n"
        "?کنایه\n"
        "*sarcasm\n",
        encoding="utf-8",
    )
    specs = _load_hashtag_specs(p)
    assert specs == [
        HashtagSpec("تیکه_سنگین", search=False),
        HashtagSpec("کنایه", search=True),
        HashtagSpec("sarcasm", search=True),
    ]


def test_load_hashtag_specs_search_all(tmp_path: Path) -> None:
    p = tmp_path / "tags.txt"
    p.write_text("کنایه\nطنز\n", encoding="utf-8")
    specs = _load_hashtag_specs(p, search_all=True)
    assert all(spec.search for spec in specs)
    assert [spec.term for spec in specs] == ["کنایه", "طنز"]
