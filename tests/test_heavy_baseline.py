"""Hypothesis 3 verdict and label parse — no VLM download."""

from eval.heavy_baseline import DEFAULT_MODEL, h3_verdict, parse_label, stratified_post_ids


class _Rec:
    def __init__(self, post_id, label):
        self.post_id = post_id
        self.label = label


def test_parse_label_finds_canonical_token():
    assert parse_label("positive_sarcasm") == "positive_sarcasm"
    assert parse_label("The label is NEGATIVE.") == "negative"
    assert parse_label("nope") is None
    assert parse_label("برچسب: کنایه مثبت") == "positive_sarcasm"
    assert parse_label("خنثی") == "neutral"


def test_default_heavy_model_is_loadable_smolvlm_family():
    assert "SmolVLM" in DEFAULT_MODEL
    assert "Qwen" not in DEFAULT_MODEL


def test_h3_not_run_when_vlm_failed():
    assert h3_verdict({"ran": False}) == "NOT_RUN"


def test_h3_pass_requires_all_three():
    ok = {
        "ran": True,
        "ours_under_1gib": True,
        "ours_faster": True,
        "accuracy_drop": 0.02,
    }
    assert h3_verdict(ok) == "PASS"
    assert h3_verdict({**ok, "accuracy_drop": 0.08}) == "FAIL"
    assert h3_verdict({**ok, "ours_faster": False}) == "FAIL"
    assert h3_verdict({**ok, "ours_under_1gib": False}) == "FAIL"


def test_oom_detector():
    from eval.heavy_baseline import _is_oom

    assert _is_oom(RuntimeError("CUDA out of memory"))
    assert not _is_oom(RuntimeError("file not found"))


def test_stratified_sample_size():
    recs = [_Rec(f"p{i}", lab) for i, lab in enumerate(["positive"] * 10 + ["negative"] * 10)]
    ids = stratified_post_ids(recs, 8, seed=0)
    assert len(ids) == 8
    assert len(set(ids)) == 8
