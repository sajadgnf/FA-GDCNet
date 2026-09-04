"""Assemble must score missing CLIP face affect before joining features."""

from __future__ import annotations

import json

import pytest

from data.schema import DatasetRecord
from inference.gdrm import polarity_scalar
from inference.stages import assemble


def test_assemble_scores_missing_clip_affect(tmp_path, monkeypatch):
    mclip = tmp_path / "mclip.jsonl"
    polarity = tmp_path / "polarity.jsonl"
    affect = tmp_path / "affect.jsonl"
    out = tmp_path / "features.npz"
    rec = DatasetRecord(
        "craft-vh000",
        "bitter caption",
        str(tmp_path / "face.jpg"),
        "positive_sarcasm",
        ["blind-relabel"],
    )
    mclip.write_text(
        json.dumps({"post_id": rec.post_id, "Dsem": 0.3, "Fvt": 0.5, "cos_TI": 0.9})
        + "\n",
        encoding="utf-8",
    )
    polarity.write_text(
        json.dumps(
            {
                "post_id": rec.post_id,
                "pol_T": [0.8, 0.2],
                "pol_T_hat": [0.5, 0.5],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    called: list[int] = []

    def fake_score(rows, *, cache, **_kwargs):
        called.append(len(rows))
        cache.write_text(
            json.dumps(
                {
                    "post_id": rec.post_id,
                    "pos": 0.9,
                    "neg": 0.1,
                    "hat": "pos",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {rec.post_id: {"pos": 0.9, "neg": 0.1, "hat": "pos"}}

    monkeypatch.setattr("data.image_affect.score_images", fake_score)
    X, y, ids = assemble(
        [rec], mclip=mclip, polarity=polarity, affect=affect, out=out
    )
    assert called == [1]
    assert ids == [rec.post_id]
    assert list(y) == ["positive_sarcasm"]
    assert float(X[0, 5]) == pytest.approx(polarity_scalar([0.1, 0.9]))
