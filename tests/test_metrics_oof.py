"""OOF predictions stay aligned with feature-cache row order."""

import numpy as np

from eval.metrics import evaluate, write_oof_jsonl


def test_oof_length_matches_n(tmp_path):
    rng = np.random.default_rng(0)
    y = np.array(
        ["positive"] * 16
        + ["negative"] * 8
        + ["neutral"] * 8
        + ["positive_sarcasm"] * 8
        + ["negative_sarcasm"] * 8,
        dtype=object,
    )
    X = rng.normal(size=(len(y), 6)).astype(np.float32)
    result = evaluate(X, y)
    assert len(result["oof_true"]) == len(y)
    assert len(result["oof_pred"]) == len(y)
    assert set(result["oof_true"]) <= set(y.tolist())
    ids = [f"p{i}" for i in range(len(y))]
    path = tmp_path / "oof.jsonl"
    write_oof_jsonl(ids, result["oof_true"], result["oof_pred"], path)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(y)
    assert '"post_id": "p0"' in lines[0]
