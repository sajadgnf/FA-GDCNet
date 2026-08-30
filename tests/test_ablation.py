"""Ablation includes a no-CLIP feature subset."""

import numpy as np

from eval.ablation import NO_CLIP, run
from inference.gdrm import FEATURE_NAMES


def test_no_clip_drops_hat_and_dsen():
    assert "polarity_T_hat" not in NO_CLIP
    assert "Dsen" not in NO_CLIP
    assert set(NO_CLIP) <= set(FEATURE_NAMES)


def test_run_includes_no_clip_row():
    rng = np.random.default_rng(0)
    y = np.array(
        ["positive"] * 10
        + ["negative"] * 10
        + ["neutral"] * 10
        + ["positive_sarcasm"] * 10
        + ["negative_sarcasm"] * 10,
        dtype=object,
    )
    X = rng.normal(size=(len(y), 6)).astype(np.float32)
    rows = run(X, y)
    names = [r["configuration"] for r in rows]
    assert "no_clip" in names
    assert "aux_only" in names
    assert "mean_sarcasm_f1" in rows[0]
