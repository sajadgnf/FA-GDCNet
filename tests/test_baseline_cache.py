"""Baseline feature cache must match current jsonl ids and labels."""

from types import SimpleNamespace

import numpy as np

from eval.baseline import _cache_matches_dataset


def test_stale_cache_rejected(tmp_path):
    cache = tmp_path / "baseline_features.npz"
    np.savez_compressed(
        cache,
        X=np.zeros((2, 2), dtype=np.float32),
        y=np.array(["positive", "neutral"], dtype=object),
        post_ids=np.array(["a", "b"], dtype=object),
    )
    records = [
        SimpleNamespace(post_id="a", label="positive"),
        SimpleNamespace(post_id="b", label="negative"),
    ]
    assert _cache_matches_dataset(cache, records) is False
    records[1].label = "neutral"
    assert _cache_matches_dataset(cache, records) is True


def test_missing_post_ids_rejected(tmp_path):
    cache = tmp_path / "baseline_features.npz"
    np.savez_compressed(
        cache,
        X=np.zeros((1, 2), dtype=np.float32),
        y=np.array(["positive"], dtype=object),
    )
    records = [SimpleNamespace(post_id="a", label="positive")]
    assert _cache_matches_dataset(cache, records) is False
