"""Tests for the dataset schema and JSONL loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data.schema import (
    LABELS,
    DatasetRecord,
    DatasetSchemaError,
    iter_dataset,
    validate_record,
    write_dataset,
)


def test_labels_are_canonical_five():
    assert LABELS == (
        "positive",
        "negative",
        "neutral",
        "positive_sarcasm",
        "negative_sarcasm",
    )


def test_validate_record_accepts_minimum_fields():
    raw = {
        "post_id": "abc",
        "caption": "سلام",
        "image_path": "img.jpg",
        "label": "positive",
        "annotators": ["alice"],
    }
    # Should not raise.
    validate_record(raw)


def test_validate_record_rejects_missing_field():
    raw = {
        "post_id": "abc",
        "caption": "سلام",
        "image_path": "img.jpg",
        "label": "positive",
    }
    with pytest.raises(DatasetSchemaError) as exc:
        validate_record(raw, line_no=7)
    assert "annotators" in str(exc.value)
    assert "line 7" in str(exc.value)


def test_validate_record_rejects_unknown_label():
    raw = {
        "post_id": "abc",
        "caption": "سلام",
        "image_path": "img.jpg",
        "label": "tongue_in_cheek",
        "annotators": [],
    }
    with pytest.raises(DatasetSchemaError) as exc:
        validate_record(raw, line_no=3)
    assert "label" in str(exc.value)
    assert "tongue_in_cheek" in str(exc.value)


def test_iter_dataset_streams_records(tmp_path: Path):
    rows = [
        {
            "post_id": f"p{i}",
            "caption": "سلام",
            "image_path": f"img{i}.jpg",
            "label": LABELS[i % len(LABELS)],
            "annotators": ["alice"],
            "kappa": None,
        }
        for i in range(3)
    ]
    p = tmp_path / "data.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    loaded = list(iter_dataset(p))
    assert len(loaded) == 3
    assert all(isinstance(r, DatasetRecord) for r in loaded)
    assert loaded[0].post_id == "p0"


def test_iter_dataset_skips_blank_lines(tmp_path: Path):
    p = tmp_path / "data.jsonl"
    rec = {
        "post_id": "a",
        "caption": "x",
        "image_path": "i.jpg",
        "label": "neutral",
        "annotators": [],
    }
    p.write_text("\n" + json.dumps(rec) + "\n\n", encoding="utf-8")
    assert len(list(iter_dataset(p))) == 1


def test_iter_dataset_reports_line_for_bad_record(tmp_path: Path):
    p = tmp_path / "data.jsonl"
    good = {
        "post_id": "ok",
        "caption": "x",
        "image_path": "i.jpg",
        "label": "neutral",
        "annotators": [],
    }
    bad = dict(good)
    bad["label"] = "WAT"
    p.write_text(
        json.dumps(good) + "\n" + json.dumps(bad) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetSchemaError) as exc:
        list(iter_dataset(p))
    assert "line 2" in str(exc.value)


def test_iter_dataset_raises_on_invalid_json(tmp_path: Path):
    p = tmp_path / "data.jsonl"
    p.write_text("not-json\n", encoding="utf-8")
    with pytest.raises(DatasetSchemaError) as exc:
        list(iter_dataset(p))
    assert "line 1" in str(exc.value)


def test_write_dataset_roundtrip(tmp_path: Path):
    recs = [
        DatasetRecord(
            post_id=f"id{i}",
            caption="سلام",
            image_path=f"img{i}.jpg",
            label="neutral",
            annotators=["a"],
            kappa=0.83,
        )
        for i in range(2)
    ]
    p = tmp_path / "out.jsonl"
    assert write_dataset(p, recs) == 2
    back = list(iter_dataset(p))
    assert [r.post_id for r in back] == ["id0", "id1"]
    assert back[0].kappa == pytest.approx(0.83)
