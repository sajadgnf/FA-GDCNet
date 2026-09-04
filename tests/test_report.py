"""Report builder uses PDF §6.3 names, dummy, and H3 NOT_RUN/FAIL/PASS."""

from pathlib import Path

from eval.report import render_report


def _write_min_inputs(tmp_path: Path) -> dict:
    metrics = tmp_path / "metrics.csv"
    metrics.write_text(
        "fold,accuracy,macro_f1,f1_positive,f1_negative,f1_neutral,"
        "f1_positive_sarcasm,f1_negative_sarcasm\n"
        "1,0.5,0.4,0.5,0.4,0.2,0.3,0.6\n"
        "mean±std,0.5000±0.0,0.4000±0.0,0.5000±0.0,0.4000±0.0,"
        "0.2000±0.0,0.3000±0.0,0.6000±0.0\n"
        "\n"
        "# dummy_majority_accuracy,0.5900\n"
        "# dummy_majority_macro_f1,0.1500\n"
        "# n_samples,10\n"
        "# n_positive,6\n"
        "# n_negative,2\n"
        "# n_neutral,1\n"
        "# n_positive_sarcasm,1\n"
        "# n_negative_sarcasm,0\n",
        encoding="utf-8",
    )
    baseline = tmp_path / "baseline.csv"
    baseline.write_text(
        "fold,accuracy,macro_f1,f1_positive,f1_negative,f1_neutral,"
        "f1_positive_sarcasm,f1_negative_sarcasm\n"
        "1,0.3,0.2,0.3,0.1,0.1,0.0,0.0\n",
        encoding="utf-8",
    )
    sarcasm = tmp_path / "sarcasm.csv"
    sarcasm.write_text(
        "fold,clash_rule_accuracy,clash_rule_f1,dsem_rule_accuracy,logreg_accuracy,f1\n"
        "mean±std,0.85±0.01,0.45±0.03,0.82±0.01,0.78±0.02,0.45±0.03\n"
        "\n"
        "# mean_accuracy_clash_rule,0.8500\n"
        "# mean_precision_clash_rule,0.5000\n"
        "# mean_recall_clash_rule,0.4000\n"
        "# mean_f1_clash_rule,0.4500\n"
        "# mean_accuracy_dsem_rule,0.8200\n"
        "# mean_accuracy_logreg,0.7800\n"
        "# dummy_not_sarcasm_accuracy,0.9000\n"
        "# pdf_accuracy_bar,true\n"
        "# beats_dummy_accuracy,false\n"
        "# meets_hypothesis_70pct,true\n",
        encoding="utf-8",
    )
    return {
        "metrics_csv": metrics,
        "baseline_csv": baseline,
        "profile_json": tmp_path / "missing.json",
        "profile_staged_json": tmp_path / "missing.json",
        "sarcasm_csv": sarcasm,
        "baseline_sarcasm_csv": tmp_path / "missing.csv",
        "ablation_png": tmp_path / "missing.png",
        "ablation_csv": tmp_path / "missing.csv",
        "confusion_png": tmp_path / "missing.png",
    }


def test_render_report_pdf_hypotheses_and_dummy(tmp_path: Path):
    kwargs = _write_min_inputs(tmp_path)
    heavy = tmp_path / "heavy_compare.json"
    heavy.write_text('{"ran": false, "h3": "NOT_RUN", "reason": "test"}\n', encoding="utf-8")
    body = render_report(**kwargs, heavy_compare_json=heavy)
    assert "0.5900" in body
    assert "Always-not-sarcasm dummy" in body
    assert "Hypothesis 3" in body
    assert "**NOT_RUN**" in body
    assert "Research question 2" in body
    assert "H1 memory < 1 GiB" in body
    assert "H2 sarcasm accuracy > 70%" in body
    assert "NO as detection" in body
    assert "Beats always-not-sarcasm accuracy" in body
    assert "H3 vs heavy model" in body
    assert "research question, not H3" in body
    assert "CLIP facial affect" in body
    assert "binary F1" in body
    assert "twitter-xlm-roberta" in body
    assert "ParsBERT" in body
    assert "Deviations from the proposal PDF" in body
    assert "reports/iaa.md" in body


def test_render_report_h3_fail_stamp(tmp_path: Path):
    kwargs = _write_min_inputs(tmp_path)
    heavy = tmp_path / "heavy_compare.json"
    heavy.write_text(
        '{"ran": true, "h3": "FAIL", "model": "x", "reason": "", '
        '"n_samples": 10, "heavy_accuracy": 0.9, "ours_accuracy": 0.5, '
        '"accuracy_drop": 0.4}\n',
        encoding="utf-8",
    )
    body = render_report(**kwargs, heavy_compare_json=heavy)
    assert "| H3 vs heavy model" in body
    assert "**FAIL**" in body


def test_origin_split_does_not_overwrite_rq2_delta(tmp_path: Path):
    kwargs = _write_min_inputs(tmp_path)
    origin = tmp_path / "origin_split.json"
    origin.write_text(
        '{"organic_rq2_delta": 0.11, "organic_rq2_meets_10pp": true, '
        '"splits": {'
        '"all": {"n": 10, "n_sarcasm": 2, "rq2_delta": 0.24, '
        '"binary": {"clash": {"precision": {"mean": 0.5}, "recall": {"mean": 0.5}, '
        '"f1": {"mean": 0.5}, "accuracy": {"mean": 0.8}, "beats_dummy_accuracy": false}}}, '
        '"organic": {"n": 8, "n_sarcasm": 1, "rq2_delta": 0.11, '
        '"binary": {"clash": {"precision": {"mean": 0.3}, "recall": {"mean": 0.3}, '
        '"f1": {"mean": 0.3}, "accuracy": {"mean": 0.9}, "beats_dummy_accuracy": false}}}, '
        '"crafted": {"n": 2, "n_sarcasm": 1, "rq2_delta": 0.99, '
        '"binary": {"clash": {"precision": {"mean": 0.9}, "recall": {"mean": 0.9}, '
        '"f1": {"mean": 0.9}, "accuracy": {"mean": 0.7}, "beats_dummy_accuracy": true}}}'
        "}}\n",
        encoding="utf-8",
    )
    heavy = tmp_path / "heavy_compare.json"
    heavy.write_text('{"ran": false, "h3": "NOT_RUN"}\n', encoding="utf-8")
    body = render_report(
        **kwargs, heavy_compare_json=heavy, origin_split_json=origin
    )
    assert "+45.0 pp" in body
    assert "+99.0 pp" not in body
    assert "Organic vs crafted" in body
