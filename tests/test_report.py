"""Report builder includes dummy baseline and CLIP defense notes."""

from pathlib import Path

from eval.report import render_report


def test_render_report_includes_dummy_and_clip_notes(tmp_path: Path):
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
        "fold,dsem_rule_accuracy,logreg_accuracy,f1\n"
        "mean±std,0.85±0.01,0.78±0.02,0.45±0.03\n"
        "\n"
        "# mean_accuracy_dsem_rule,0.8500\n"
        "# mean_accuracy_logreg,0.7800\n",
        encoding="utf-8",
    )
    body = render_report(
        metrics_csv=metrics,
        baseline_csv=baseline,
        profile_json=tmp_path / "missing.json",
        profile_staged_json=tmp_path / "missing.json",
        sarcasm_csv=sarcasm,
        baseline_sarcasm_csv=tmp_path / "missing.csv",
        ablation_png=tmp_path / "missing.png",
        ablation_csv=tmp_path / "missing.csv",
        confusion_png=tmp_path / "missing.png",
    )
    assert "0.5900" in body
    assert "CLIP facial affect" in body
    assert "`positive`" in body
    assert "binary F1" in body
