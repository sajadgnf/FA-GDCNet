"""Organic vs crafted split retrains on each domain."""

import json

import numpy as np

from data.schema import LABELS, DatasetRecord, write_dataset
from eval.origin_split import evaluate, is_crafted
from inference.gdrm import FEATURE_NAMES


def test_is_crafted():
    assert is_crafted("craft-vh000") is True
    assert is_crafted("DGnvmEfK-aJ") is False


def test_retrain_split_separates_craft_clash(tmp_path):
    rng = np.random.default_rng(0)
    records = []
    Xs = []
    ys = []
    ids = []
    n_feat = len(FEATURE_NAMES)

    def row(*, clash: float, label: str) -> np.ndarray:
        x = rng.normal(0.0, 0.05, size=n_feat).astype(np.float32)
        x[4] = -0.8 if clash > 0 else 0.3
        x[5] = 0.8 if clash > 0 else 0.3
        x[6] = clash
        return x

    for lbl in LABELS:
        for i in range(8):
            pid = f"ig_{lbl}_{i}"
            records.append(
                DatasetRecord(pid, "c", "img.jpg", lbl, ["blind-relabel"])
            )
            ids.append(pid)
            ys.append(lbl)
            clash = 0.0
            Xs.append(row(clash=clash, label=lbl))
    for lbl in ("positive_sarcasm", "negative_sarcasm"):
        for i in range(8):
            pid = f"craft-x{i:03d}" if lbl.startswith("positive") else f"craft-y{i:03d}"
            records.append(
                DatasetRecord(pid, "c", "img.jpg", lbl, ["blind-relabel"])
            )
            ids.append(pid)
            ys.append(lbl)
            Xs.append(row(clash=0.7, label=lbl))
    for i in range(8):
        pid = f"craft-z{i:03d}"
        records.append(
            DatasetRecord(pid, "c", "img.jpg", "positive", ["blind-relabel"])
        )
        ids.append(pid)
        ys.append("positive")
        Xs.append(row(clash=0.0, label="positive"))

    ds = tmp_path / "data.jsonl"
    write_dataset(ds, records)
    feat = tmp_path / "feat.npz"
    X = np.stack(Xs)
    np.savez_compressed(
        feat,
        X=X,
        y=np.asarray(ys, dtype=object),
        post_ids=np.asarray(ids, dtype=object),
        feature_names=np.asarray(FEATURE_NAMES, dtype=object),
    )
    payload = evaluate(
        dataset=ds,
        features=feat,
        baseline_cache=tmp_path / "missing.npz",
        oof_path=tmp_path / "missing.jsonl",
    )
    org_f1 = payload["splits"]["organic"]["binary"]["clash"]["f1"]["mean"]
    craft_f1 = payload["splits"]["crafted"]["binary"]["clash"]["f1"]["mean"]
    assert payload["splits"]["organic"]["n_sarcasm"] == 16
    assert payload["splits"]["crafted"]["n_sarcasm"] == 16
    assert craft_f1 > org_f1
    assert payload["splits"]["organic"]["rq2_delta"] is None
    text = json.dumps(payload)
    assert "craft-" in text
