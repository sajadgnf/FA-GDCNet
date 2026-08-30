"""Human review recording and feature-label sync."""

from pathlib import Path

from data.relabel import (
    apply_review,
    format_item,
    is_reviewed,
    matching_indices,
    matching_indices_ordered,
    parse_choice,
    select_overlap_ids,
    sync_feature_labels,
    upsert_secondary,
)


def test_enter_is_not_a_label():
    assert parse_choice("") == "invalid"
    assert parse_choice("   ") == "invalid"


def test_parse_choice_keys():
    assert parse_choice("4") == "positive_sarcasm"
    assert parse_choice("q") == "quit"
    assert parse_choice("s") == "skip"


def test_prompt_hides_current_label():
    row = {
        "post_id": "abc",
        "caption": "hello",
        "image_path": "img.jpg",
        "label": "positive_sarcasm",
    }
    text = format_item(row, 1, 10)
    assert "positive_sarcasm" not in text
    assert "current=" not in text
    assert "abc" in text


def test_old_enter_confirm_is_not_blind_review():
    row = {"label": "positive_sarcasm", "annotators": ["sjjd6502", "relabel"]}
    assert is_reviewed(row) is False


def test_apply_review_records_blind_tag():
    row = {"label": "positive_sarcasm", "annotators": ["proposal-retag"]}
    changed = apply_review(row, "positive_sarcasm", "sjjd6502")
    assert changed is False
    assert is_reviewed(row)
    assert "sjjd6502" in row["annotators"]
    assert "blind-relabel" in row["annotators"]
    assert "relabel" not in row["annotators"]


def test_change_label_records_review():
    row = {"label": "positive_sarcasm", "annotators": ["proposal-retag"]}
    changed = apply_review(row, "positive", "sjjd6502")
    assert changed is True
    assert row["label"] == "positive"
    assert is_reviewed(row)


def test_pending_only_skips_blind_reviewed():
    rows = [
        {"post_id": "a", "label": "positive_sarcasm", "annotators": ["proposal-retag"]},
        {"post_id": "b", "label": "negative_sarcasm", "annotators": ["sjjd6502", "blind-relabel"]},
        {"post_id": "c", "label": "positive", "annotators": []},
        {"post_id": "d", "label": "negative_sarcasm", "annotators": ["sjjd6502", "relabel"]},
    ]
    pending = matching_indices(rows, only="sarcasm", pending_only=True)
    assert pending == [0, 3]
    all_sarc = matching_indices(rows, only="sarcasm", pending_only=False)
    assert all_sarc == [0, 1, 3]


def test_second_annotator_not_blocked_by_gold_blind():
    rows = [
        {"post_id": "a", "label": "positive_sarcasm", "annotators": ["sjjd6502", "blind-relabel"]},
        {"post_id": "b", "label": "positive", "annotators": []},
    ]
    pending = matching_indices_ordered(
        rows,
        id_order=["a", "b"],
        pending_only=True,
        already_done=set(),
        skip_if_gold_reviewed=False,
    )
    assert pending == [0, 1]


def test_select_overlap_includes_all_sarcasm(tmp_path: Path):
    rows = [
        {"post_id": f"s{i}", "label": "positive_sarcasm", "image_path": "x"}
        for i in range(3)
    ] + [
        {"post_id": f"p{i}", "label": "positive", "image_path": "x"}
        for i in range(8)
    ] + [
        {"post_id": f"n{i}", "label": "negative", "image_path": "x"}
        for i in range(4)
    ]
    ids = select_overlap_ids(rows, n_non_sarcasm=4, seed=0, require_image=False)
    assert {f"s{i}" for i in range(3)} <= set(ids)
    assert len(ids) == 7


def test_upsert_secondary_independent_of_gold(tmp_path: Path):
    path = tmp_path / "second.jsonl"
    upsert_secondary(path, post_id="a", annotator="bob", label="negative")
    upsert_secondary(path, post_id="a", annotator="bob", label="positive")
    text = path.read_text(encoding="utf-8")
    assert text.count("\n") == 1
    assert '"positive"' in text
    assert "bob" in text


def test_sync_feature_labels(tmp_path: Path):
    import json

    import numpy as np

    ds = tmp_path / "data.jsonl"
    rows = [
        {"post_id": "a", "label": "positive"},
        {"post_id": "b", "label": "negative_sarcasm"},
    ]
    ds.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    cache = tmp_path / "features.npz"
    np.savez_compressed(
        cache,
        X=np.zeros((2, 6), dtype=np.float32),
        y=np.array(["positive", "positive"], dtype=object),
        post_ids=np.array(["a", "b"], dtype=object),
    )
    changed = sync_feature_labels(ds, cache)
    assert changed == 1
    z = np.load(cache, allow_pickle=True)
    assert list(z["y"]) == ["positive", "negative_sarcasm"]
    assert z["X"].shape == (2, 6)
