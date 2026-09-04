"""Drop unlabeled bootstrap candidates without touching human gold."""

import json
from pathlib import Path

from data.drop_unreviewed import drop_unreviewed
from data.tags import BOOTSTRAP_TAG, CANDIDATE_TAG


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )


def test_drop_unreviewed_keeps_human_gold(tmp_path: Path, monkeypatch) -> None:
    img_boot = tmp_path / "boot.jpg"
    img_keep = tmp_path / "keep.jpg"
    img_skip = tmp_path / "skip.jpg"
    img_boot.write_bytes(b"x")
    img_keep.write_bytes(b"y")
    img_skip.write_bytes(b"z")

    gold = tmp_path / "gold.jsonl"
    _write_jsonl(
        gold,
        [
            {
                "post_id": "boot",
                "caption": "c",
                "image_path": str(img_boot),
                "label": "neutral",
                "annotators": [BOOTSTRAP_TAG, CANDIDATE_TAG],
            },
            {
                "post_id": "labeled",
                "caption": "c",
                "image_path": str(img_keep),
                "label": "positive_sarcasm",
                "annotators": [BOOTSTRAP_TAG, CANDIDATE_TAG, "sjjd6502", "blind-relabel"],
            },
            {
                "post_id": "oldgold",
                "caption": "c",
                "image_path": str(img_skip),
                "label": "positive",
                "annotators": ["sjjd6502", "relabel", CANDIDATE_TAG],
            },
        ],
    )
    pool = tmp_path / "sarcasm.jsonl"
    _write_jsonl(
        pool,
        [
            {"post_id": "boot", "caption": "c", "image_path": str(img_boot)},
            {"post_id": "labeled", "caption": "c", "image_path": str(img_keep)},
        ],
    )
    ignore = tmp_path / "ignored_post_ids.txt"
    monkeypatch.setattr("data.drop_unreviewed.IGNORED_IDS_FILE", ignore)
    monkeypatch.setattr("data.scrape.IGNORED_IDS_FILE", ignore)

    result = drop_unreviewed(dataset=gold, pools=[pool], delete_images=True)
    assert result.dropped_gold == 1
    assert result.kept_gold == 2
    rows = [json.loads(l) for l in gold.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert {r["post_id"] for r in rows} == {"labeled", "oldgold"}
    pool_ids = [json.loads(l)["post_id"] for l in pool.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert pool_ids == ["labeled"]
    assert not img_boot.exists()
    assert img_keep.exists()
    assert "boot" in ignore.read_text(encoding="utf-8")


def test_drop_unreviewed_ids_file_restricts(tmp_path: Path, monkeypatch) -> None:
    gold = tmp_path / "gold.jsonl"
    _write_jsonl(
        gold,
        [
            {
                "post_id": "a",
                "caption": "c",
                "image_path": "",
                "label": "neutral",
                "annotators": [BOOTSTRAP_TAG],
            },
            {
                "post_id": "b",
                "caption": "c",
                "image_path": "",
                "label": "neutral",
                "annotators": [BOOTSTRAP_TAG],
            },
        ],
    )
    ignore = tmp_path / "ignored_post_ids.txt"
    monkeypatch.setattr("data.drop_unreviewed.IGNORED_IDS_FILE", ignore)
    monkeypatch.setattr("data.scrape.IGNORED_IDS_FILE", ignore)
    result = drop_unreviewed(dataset=gold, ids=["a"], pools=[], delete_images=False)
    assert result.dropped_gold == 1
    rows = [json.loads(l) for l in gold.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert [r["post_id"] for r in rows] == ["b"]
