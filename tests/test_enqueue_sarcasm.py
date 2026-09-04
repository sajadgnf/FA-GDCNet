"""Queue scrape-pool posts for blind review without assigning sarcasm gold."""

import json
from pathlib import Path

from data.enqueue_sarcasm import PLACEHOLDER_LABEL, enqueue
from data.relabel import matching_indices
from data.tags import BOOTSTRAP_TAG, CANDIDATE_TAG


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


def test_enqueue_imports_archive_as_placeholder_not_sarcasm(tmp_path: Path):
    img = _touch(tmp_path / "a.jpg")
    gold = tmp_path / "gold.jsonl"
    _write_jsonl(
        gold,
        [
            {
                "post_id": "keep",
                "caption": "عکس امروز",
                "image_path": str(_touch(tmp_path / "keep.jpg")),
                "label": "positive",
                "annotators": ["sjjd6502"],
                "kappa": None,
            }
        ],
    )
    pool = tmp_path / "sarc.jsonl"
    _write_jsonl(
        pool,
        [
            {
                "post_id": "new1",
                "caption": "کپشن کنایه برای تست صف",
                "image_path": str(img),
            }
        ],
    )
    ids_path = tmp_path / "ids.txt"
    result = enqueue(
        dataset=gold,
        ids_path=ids_path,
        unfiltered_pools=[pool],
        filtered_pools=[],
        include_gold_heuristic=False,
        roots=[tmp_path],
    )
    assert result.imported == 1
    assert result.queued == 1
    rows = [json.loads(l) for l in gold.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_id = {r["post_id"]: r for r in rows}
    assert by_id["new1"]["label"] == PLACEHOLDER_LABEL
    assert by_id["new1"]["label"] not in {"positive_sarcasm", "negative_sarcasm"}
    assert BOOTSTRAP_TAG in by_id["new1"]["annotators"]
    assert CANDIDATE_TAG in by_id["new1"]["annotators"]
    assert "new1" in ids_path.read_text(encoding="utf-8")


def test_enqueue_bootstrap_pending_first_and_skips_blind(tmp_path: Path):
    boot_img = _touch(tmp_path / "b.jpg")
    blind_img = _touch(tmp_path / "c.jpg")
    gold = tmp_path / "gold.jsonl"
    _write_jsonl(
        gold,
        [
            {
                "post_id": "blind",
                "caption": "طنز تلخ باور نکن",
                "image_path": str(blind_img),
                "label": "positive",
                "annotators": ["sjjd6502", "blind-relabel"],
                "kappa": None,
            },
            {
                "post_id": "boot",
                "caption": "weak row",
                "image_path": str(boot_img),
                "label": "neutral",
                "annotators": [BOOTSTRAP_TAG],
                "kappa": None,
            },
        ],
    )
    result = enqueue(
        dataset=gold,
        ids_path=tmp_path / "ids.txt",
        unfiltered_pools=[],
        filtered_pools=[],
        include_gold_heuristic=True,
        roots=[tmp_path],
    )
    assert result.ids[0] == "boot"
    assert "blind" not in result.ids
    rows = [json.loads(l) for l in gold.read_text(encoding="utf-8").splitlines() if l.strip()]
    boot = next(r for r in rows if r["post_id"] == "boot")
    assert CANDIDATE_TAG in boot["annotators"]
    assert boot["label"] == "neutral"


def test_enqueue_filtered_pool_uses_strict_caption(tmp_path: Path):
    gold = tmp_path / "gold.jsonl"
    _write_jsonl(gold, [])
    keep = tmp_path / "keep.jsonl"
    drop = tmp_path / "also.jsonl"
    _write_jsonl(
        keep,
        [
            {
                "post_id": "hit",
                "caption": "حالم بده ولی دارم میخندم #طنز",
                "image_path": str(_touch(tmp_path / "hit.jpg")),
            }
        ],
    )
    _write_jsonl(
        drop,
        [
            {
                "post_id": "miss",
                "caption": 'گفت «سلام» و رفت #سلفی',
                "image_path": str(_touch(tmp_path / "miss.jpg")),
            }
        ],
    )
    result = enqueue(
        dataset=gold,
        ids_path=tmp_path / "ids.txt",
        unfiltered_pools=[],
        filtered_pools=[keep, drop],
        include_gold_heuristic=False,
        roots=[tmp_path],
    )
    assert result.ids == ["hit"]
    assert result.imported == 1


def test_from_pools_only_skips_existing_gold(tmp_path: Path):
    gold = tmp_path / "gold.jsonl"
    _write_jsonl(
        gold,
        [
            {
                "post_id": "boot",
                "caption": "weak row",
                "image_path": str(_touch(tmp_path / "b.jpg")),
                "label": "neutral",
                "annotators": [BOOTSTRAP_TAG],
                "kappa": None,
            }
        ],
    )
    pool = tmp_path / "sarc.jsonl"
    _write_jsonl(
        pool,
        [
            {
                "post_id": "fresh",
                "caption": "کنایه جدید",
                "image_path": str(_touch(tmp_path / "f.jpg")),
            }
        ],
    )
    result = enqueue(
        dataset=gold,
        ids_path=tmp_path / "ids.txt",
        unfiltered_pools=[pool],
        filtered_pools=[],
        from_pools_only=True,
        include_gold_heuristic=False,
        roots=[tmp_path],
    )
    assert result.ids == ["fresh"]
    assert result.imported == 1
    assert result.tagged_existing == 0


def test_enqueue_skips_political_even_from_unfiltered_pool(tmp_path: Path):
    gold = tmp_path / "gold.jsonl"
    _write_jsonl(gold, [])
    pool = tmp_path / "sarc.jsonl"
    _write_jsonl(
        pool,
        [
            {
                "post_id": "gov",
                "caption": "طنز تلخ قطعی برق #خمینی #رئیسی",
                "image_path": str(_touch(tmp_path / "gov.jpg")),
            },
            {
                "post_id": "ok",
                "caption": "کنایه به رفیق دورو",
                "image_path": str(_touch(tmp_path / "ok.jpg")),
            },
        ],
    )
    result = enqueue(
        dataset=gold,
        ids_path=tmp_path / "ids.txt",
        unfiltered_pools=[pool],
        filtered_pools=[],
        include_gold_heuristic=False,
        from_pools_only=True,
        roots=[tmp_path],
    )
    assert result.ids == ["ok"]
    assert result.imported == 1


def test_enqueue_skips_ignored_ids(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("data.enqueue_sarcasm._load_ignored_shortcodes", lambda: {"blocked"})
    gold = tmp_path / "gold.jsonl"
    _write_jsonl(gold, [])
    pool = tmp_path / "sarc.jsonl"
    img_ok = _touch(tmp_path / "ok.jpg")
    _write_jsonl(
        pool,
        [
            {
                "post_id": "blocked",
                "caption": "حالم بده ولی دارم میخندم",
                "image_path": str(_touch(tmp_path / "blocked.jpg")),
            },
            {
                "post_id": "fresh",
                "caption": "حالم بده ولی دارم میخندم",
                "image_path": str(img_ok),
            },
        ],
    )
    result = enqueue(
        dataset=gold,
        ids_path=tmp_path / "ids.txt",
        unfiltered_pools=[pool],
        filtered_pools=[],
        include_gold_heuristic=False,
        from_pools_only=True,
        roots=[tmp_path],
    )
    assert result.ids == ["fresh"]


def test_relabel_only_candidates():
    rows = [
        {"post_id": "a", "label": "neutral", "annotators": [CANDIDATE_TAG]},
        {"post_id": "b", "label": "positive_sarcasm", "annotators": ["sjjd6502"]},
        {
            "post_id": "c",
            "label": "neutral",
            "annotators": [CANDIDATE_TAG, "blind-relabel"],
        },
    ]
    pending = matching_indices(rows, only="candidates", pending_only=True)
    assert pending == [0]
    listed = matching_indices(rows, only="candidates", pending_only=False)
    assert listed == [0, 2]
