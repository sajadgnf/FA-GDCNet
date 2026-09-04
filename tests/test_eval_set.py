"""Eval set excludes weak-sarcasm-bootstrap unless blindly reviewed."""

import json

import numpy as np

from data.eval_set import is_eval_eligible, slice_records_from_cache, slice_to_eval_set
from data.schema import DatasetRecord


def test_bootstrap_excluded_until_blind():
    assert is_eval_eligible(["weak-sarcasm-bootstrap"]) is False
    assert is_eval_eligible(["weak-sarcasm-bootstrap", "blind-relabel"]) is True
    assert is_eval_eligible(["sjjd6502", "proposal-retag"]) is True


def test_slice_drops_bootstrap_and_uses_jsonl_y(tmp_path):
    ds = tmp_path / "data.jsonl"
    rows = [
        {
            "post_id": "a",
            "caption": "x",
            "image_path": "i.jpg",
            "label": "negative",
            "annotators": ["sjjd6502"],
        },
        {
            "post_id": "b",
            "caption": "x",
            "image_path": "i.jpg",
            "label": "positive_sarcasm",
            "annotators": ["weak-sarcasm-bootstrap"],
        },
        {
            "post_id": "c",
            "caption": "x",
            "image_path": "i.jpg",
            "label": "positive",
            "annotators": ["weak-sarcasm-bootstrap", "blind-relabel"],
        },
    ]
    ds.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    X = np.arange(18, dtype=np.float32).reshape(3, 6)
    y = np.array(["positive", "positive_sarcasm", "neutral"], dtype=object)
    ids = ["a", "b", "c"]
    Xs, ys, kept, n_excl = slice_to_eval_set(ds, X, y, ids)
    assert n_excl == 1
    assert kept == ["a", "c"]
    assert list(ys) == ["negative", "positive"]
    assert Xs.shape == (2, 6)
    assert np.allclose(Xs[0], X[0])
    assert np.allclose(Xs[1], X[2])


def test_slice_records_skips_extra_cache_rows(tmp_path):
    cache = tmp_path / "feat.npz"
    np.savez_compressed(
        cache,
        X=np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float32),
        y=np.array(["positive", "positive", "negative"], dtype=object),
        post_ids=np.array(["a", "boot", "c"], dtype=object),
    )
    records = [
        DatasetRecord("a", "t", "i.jpg", "neutral", ["sjjd6502"]),
        DatasetRecord("c", "t", "i.jpg", "negative", ["sjjd6502"]),
    ]
    out = slice_records_from_cache(cache, records)
    assert out is not None
    X, y = out
    assert X.shape == (2, 2)
    assert list(y) == ["neutral", "negative"]
    assert float(X[0, 0]) == 1.0
    assert float(X[1, 0]) == 3.0


def test_slice_records_missing_id_returns_none(tmp_path):
    cache = tmp_path / "feat.npz"
    np.savez_compressed(
        cache,
        X=np.array([[1.0, 0.0]], dtype=np.float32),
        y=np.array(["positive"], dtype=object),
        post_ids=np.array(["a"], dtype=object),
    )
    records = [
        DatasetRecord("a", "t", "i.jpg", "positive", ["sjjd6502"]),
        DatasetRecord("new", "t", "i.jpg", "positive_sarcasm", ["sjjd6502"]),
    ]
    assert slice_records_from_cache(cache, records) is None
