"""Compose the final Markdown report from metrics.csv / profile.json / baseline.csv.

Per spec scenario *Improvement margin check*, the report emits Δ on the
sarcasm-class F1 macro-average and flags whether the ≥10 percentage-point
hypothesis holds.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np

from data.schema import LABELS

log = logging.getLogger(__name__)

DEFAULT_METRICS = Path("reports") / "metrics.csv"
DEFAULT_BASELINE = Path("reports") / "baseline.csv"
DEFAULT_PROFILE = Path("reports") / "profile.json"
DEFAULT_PROFILE_STAGED = Path("reports") / "profile_staged.json"
DEFAULT_SARCASM = Path("reports") / "sarcasm.csv"
DEFAULT_BASELINE_SARCASM = Path("reports") / "baseline_sarcasm.csv"
DEFAULT_ABLATION_PNG = Path("reports") / "ablation.png"
DEFAULT_ABLATION_CSV = Path("reports") / "ablation.csv"
DEFAULT_CONFUSION_PNG = Path("reports") / "confusion.png"
DEFAULT_HEAVY = Path("reports") / "heavy_compare.json"
DEFAULT_ORIGIN_SPLIT = Path("reports") / "origin_split.json"
DEFAULT_OUT = Path("reports") / "REPORT.md"

SARCASM_LABELS = ("positive_sarcasm", "negative_sarcasm")
SARCASM_F1_DELTA_FLOOR = 0.10  # spec: ≥10 percentage points
SARCASM_ACC_FLOOR = 0.70


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def _per_fold_f1(rows: list[dict], lbl: str) -> list[float]:
    out: list[float] = []
    key = f"f1_{lbl}"
    for r in rows:
        fold = r.get("fold")
        if fold in (None, "", "mean±std") or not str(fold).isdigit():
            continue
        try:
            val = r.get(key)
            if val is None or val == "":
                continue
            out.append(float(val))
        except (KeyError, ValueError, TypeError):
            continue
    return out


def _sarcasm_macro_f1(rows: list[dict]) -> float:
    per_label = {lbl: _per_fold_f1(rows, lbl) for lbl in SARCASM_LABELS}
    if not all(per_label.values()):
        return 0.0
    folds = list(zip(*per_label.values()))
    macros = [float(np.mean(f)) for f in folds]
    return float(np.mean(macros)) if macros else 0.0


def _read_footer_value(rows: list[dict], key: str) -> str | None:
    for r in rows:
        fold = str(r.get("fold") or "")
        if fold == f"# {key}":
            for v in r.values():
                if v and str(v) != fold:
                    return str(v)
        if fold.startswith("#") and key in fold:
            vals = [str(v) for v in r.values() if v is not None and str(v) != fold]
            if vals:
                return vals[-1]
    return None


def _read_sarcasm_mean(rows: list[dict]) -> float:
    val = _read_footer_value(rows, "mean_accuracy_clash_rule")
    if val:
        return float(val)
    val = _read_footer_value(rows, "mean_accuracy_dsem_rule")
    if val:
        return float(val)
    for r in rows:
        if str(r.get("fold")) == "mean±std":
            try:
                col = (
                    r.get("clash_rule_accuracy")
                    or r.get("dsem_rule_accuracy")
                    or r.get("accuracy")
                )
                return float(str(col).split("±")[0])
            except (ValueError, TypeError, AttributeError):
                return 0.0
    return 0.0


def _fold_row(rows: list[dict], fold: str) -> dict | None:
    for r in rows:
        if str(r.get("fold") or "") == fold:
            return r
    return None


def _mean_cell(rows: list[dict], key: str) -> str:
    row = _fold_row(rows, "mean±std")
    if not row:
        return ""
    return str(row.get(key) or "")


def _sarcasm_binary_f1(rows: list[dict]) -> str:
    row = _fold_row(rows, "mean±std")
    if row and row.get("f1"):
        return str(row["f1"])
    return ""


def render_report(
    *,
    metrics_csv: Path,
    baseline_csv: Path,
    profile_json: Path,
    profile_staged_json: Path,
    sarcasm_csv: Path,
    baseline_sarcasm_csv: Path,
    ablation_png: Path,
    ablation_csv: Path = DEFAULT_ABLATION_CSV,
    confusion_png: Path = DEFAULT_CONFUSION_PNG,
    heavy_compare_json: Path = DEFAULT_HEAVY,
    origin_split_json: Path = DEFAULT_ORIGIN_SPLIT,
) -> str:
    metrics = _read_csv(metrics_csv)
    baseline = _read_csv(baseline_csv)
    ablation_rows = _read_csv(ablation_csv)
    profile = json.loads(profile_json.read_text(encoding="utf-8")) if profile_json.exists() else {}
    profile_staged = (
        json.loads(profile_staged_json.read_text(encoding="utf-8"))
        if profile_staged_json.exists()
        else {}
    )
    sarcasm_rows = _read_csv(sarcasm_csv)
    baseline_sarcasm_rows = _read_csv(baseline_sarcasm_csv)
    heavy = (
        json.loads(heavy_compare_json.read_text(encoding="utf-8"))
        if heavy_compare_json.exists()
        else {}
    )
    origin = (
        json.loads(origin_split_json.read_text(encoding="utf-8"))
        if origin_split_json.exists()
        else {}
    )

    dummy_acc = _read_footer_value(metrics, "dummy_majority_accuracy")
    dummy_f1 = _read_footer_value(metrics, "dummy_majority_macro_f1")
    n_samples = _read_footer_value(metrics, "n_samples")

    lines: list[str] = []
    lines.append("# FA-GDCNet — Final Report")
    lines.append("")
    lines.append("## Multimodal pipeline (5-fold CV)")
    lines.append("")
    if metrics:
        lines.append("| fold | accuracy | macro_f1 |")
        lines.append("| --- | --- | --- |")
        for r in metrics:
            fold = str(r.get("fold") or "")
            if not fold or fold.startswith("#") or fold == "mean±std":
                continue
            if not fold.isdigit():
                continue
            lines.append(f"| {r['fold']} | {r.get('accuracy','')} | {r.get('macro_f1','')} |")
        mean_acc = _mean_cell(metrics, "accuracy")
        mean_f1 = _mean_cell(metrics, "macro_f1")
        if mean_acc or mean_f1:
            lines.append(f"| mean±std | {mean_acc} | {mean_f1} |")
    else:
        lines.append("_metrics.csv not found_")
    lines.append("")
    if dummy_acc:
        lines.append(
            f"- Majority dummy (same folds): accuracy **{dummy_acc}**, "
            f"macro-F1 **{dummy_f1 or '—'}**. 5-class accuracy is not better than "
            f"this dummy when the dummy is higher."
        )
        lines.append("")

    if any(_read_footer_value(metrics, f"n_{lbl}") for lbl in LABELS) or n_samples:
        lines.append("### Label counts (evaluated set)")
        lines.append("")
        lines.append("| label | n |")
        lines.append("| --- | --- |")
        for lbl in LABELS:
            lines.append(f"| `{lbl}` | {_read_footer_value(metrics, f'n_{lbl}') or '—'} |")
        if n_samples:
            lines.append(f"| **total** | {n_samples} |")
        lines.append("")
        n_excl = _read_footer_value(metrics, "n_excluded_bootstrap")
        if n_excl and n_excl not in {"", "0"}:
            lines.append(
                f"Excluded **{n_excl}** `weak-sarcasm-bootstrap` rows from eval "
                "(unless later tagged `blind-relabel`)."
            )
            lines.append("")

    lines.append("### Per-class F1 (mean±std)")
    lines.append("")
    lines.append("| class | F1 |")
    lines.append("| --- | --- |")
    for lbl in LABELS:
        cell = _mean_cell(metrics, f"f1_{lbl}")
        lines.append(f"| `{lbl}` | {cell or '—'} |")
    lines.append("")

    lines.append("## Unimodal text-polarity baseline (same folds)")
    lines.append("")
    lines.append(
        "Head is `cardiffnlp/twitter-xlm-roberta-base-sentiment` (2-d), "
        "not ParsBERT. The proposal named ParsBERT; this is a deviation."
    )
    lines.append("")
    if baseline:
        lines.append("| fold | accuracy | macro_f1 |")
        lines.append("| --- | --- | --- |")
        for r in baseline:
            fold = str(r.get("fold") or "")
            if not fold or fold.startswith("#") or fold == "mean±std":
                continue
            if not fold.isdigit():
                continue
            lines.append(f"| {r['fold']} | {r.get('accuracy','')} | {r.get('macro_f1','')} |")
        b_acc = _mean_cell(baseline, "accuracy")
        b_f1 = _mean_cell(baseline, "macro_f1")
        if b_acc or b_f1:
            lines.append(f"| mean±std | {b_acc} | {b_f1} |")
    else:
        lines.append("_baseline.csv not found_")
    lines.append("")

    mm_sarcasm = _sarcasm_macro_f1(metrics)
    bs_sarcasm = _sarcasm_macro_f1(baseline)
    delta = mm_sarcasm - bs_sarcasm
    passes = delta >= SARCASM_F1_DELTA_FLOOR
    lines.append("## Research question 2 (not Hypothesis 3)")
    lines.append("")
    lines.append(
        "PDF §6.2 Q2: does Dsem+Dsen raise sarcasm detection by at least 10% "
        "versus a unimodal method? This is a research question. Hypothesis 3 "
        "in §6.3 is the heavy-model speed/accuracy comparison."
    )
    lines.append("")
    lines.append(
        f"- Multimodal sarcasm-F1 (macro of {', '.join(SARCASM_LABELS)}): **{mm_sarcasm:.4f}**"
    )
    lines.append(f"- Unimodal baseline sarcasm-F1: **{bs_sarcasm:.4f}**")
    lines.append(f"- Δ = **{delta:+.4f}** ({delta * 100:+.2f} percentage points)")
    lines.append(
        f"- Meets ≥10 pp vs unimodal: **{'YES' if passes else 'NO'}**"
    )
    lines.append("")

    mm_bin = _read_sarcasm_mean(sarcasm_rows)
    bs_bin = _read_sarcasm_mean(baseline_sarcasm_rows)
    clash_f1 = _read_footer_value(sarcasm_rows, "mean_f1_clash_rule")
    clash_p = _read_footer_value(sarcasm_rows, "mean_precision_clash_rule")
    clash_r = _read_footer_value(sarcasm_rows, "mean_recall_clash_rule")
    pdf_bar_flag = _read_footer_value(sarcasm_rows, "pdf_accuracy_bar")
    beats_dummy_flag = _read_footer_value(sarcasm_rows, "beats_dummy_accuracy")
    if pdf_bar_flag in ("true", "false"):
        pdf_bar = pdf_bar_flag == "true"
    else:
        pdf_bar = mm_bin >= SARCASM_ACC_FLOOR
    dummy_footer = _read_footer_value(sarcasm_rows, "dummy_not_sarcasm_accuracy")
    if beats_dummy_flag in ("true", "false"):
        beats_dummy = beats_dummy_flag == "true"
    else:
        try:
            beats_dummy = mm_bin > float(dummy_footer) + 1e-6 if dummy_footer else False
        except (TypeError, ValueError):
            beats_dummy = False
    lines.append("## Binary sarcasm detection (proposal Hypothesis 2)")
    lines.append("")
    clash_acc = _read_footer_value(sarcasm_rows, "mean_accuracy_clash_rule")
    lines.append(
        "- Clash rule (CV-tuned for accuracy, opposite text vs face polarity): "
        f"accuracy **{clash_acc or f'{mm_bin:.4f}'}**"
        + (f", precision **{clash_p}**" if clash_p else "")
        + (f", recall **{clash_r}**" if clash_r else "")
        + (f", F1 **{clash_f1}**" if clash_f1 else "")
    )
    dsem_acc = _read_footer_value(sarcasm_rows, "mean_accuracy_dsem_rule")
    if dsem_acc:
        lines.append(
            f"- Dsem accuracy cut (dummy-trap footnote, not the H2 detector): **{dsem_acc}**"
        )
    logreg_f1 = _sarcasm_binary_f1(sarcasm_rows)
    logreg_acc = _read_footer_value(sarcasm_rows, "mean_accuracy_logreg")
    if logreg_acc or logreg_f1:
        lines.append(
            "- LogReg on discrepancy features: "
            + (f"accuracy **{logreg_acc}**" if logreg_acc else "see `sarcasm.csv`")
            + (f", binary F1 **{logreg_f1}**" if logreg_f1 else "")
        )
    else:
        lines.append("- LogReg on discrepancy features: see `sarcasm.csv`")
    n_ps = _read_footer_value(metrics, "n_positive_sarcasm")
    n_ns = _read_footer_value(metrics, "n_negative_sarcasm")
    n_tot = _read_footer_value(metrics, "n_samples")
    dummy_not_sarc = dummy_footer or ""
    try:
        if not dummy_not_sarc and n_ps and n_ns and n_tot:
            dummy_not_sarc = f"{1.0 - (float(n_ps) + float(n_ns)) / float(n_tot):.4f}"
    except (TypeError, ValueError, ZeroDivisionError):
        dummy_not_sarc = dummy_not_sarc or ""
    lines.append(f"- Unimodal baseline binary accuracy: **{bs_bin:.4f}**")
    if dummy_not_sarc:
        lines.append(
            f"- Always-not-sarcasm dummy accuracy: **{dummy_not_sarc}** "
            f"(sarcasm-class F1 of this dummy is 0)."
        )
    lines.append(
        f"- PDF §6.3(2) letter (clash accuracy ≥70%): **{'YES' if pdf_bar else 'NO'}**"
    )
    lines.append(
        f"- Beats always-not-sarcasm accuracy: **{'YES' if beats_dummy else 'NO'}**"
    )
    lines.append("")

    if profile_staged:
        lines.append("## Staged inference profile (peak VRAM per backbone)")
        lines.append("")
        for stage in profile_staged.get("stages", []):
            device = stage.get("device")
            device_note = f" [{device}]" if device else ""
            lines.append(
                f"- `{stage.get('stage')}`{device_note}: peak **{stage.get('peak_memory_gib', 0.0):.3f} GiB**, "
                f"median **{stage.get('median_latency_s', 0.0)*1000:.0f} ms**/sample"
            )
            if stage.get("note"):
                lines.append(f"  - {stage['note']}")
        lines.append(
            f"- Combined peak (max stage): **{profile_staged.get('peak_memory_gib', 0.0):.3f} GiB**"
        )
        lines.append(
            f"- Staged under_1gib_budget: **{'YES' if profile_staged.get('under_1gib_budget') else 'NO'}**"
        )
        lines.append(
            f"- Staged median total latency: **{profile_staged.get('median_total_latency_s', 0.0)*1000:.0f} ms**/sample"
        )
        lines.append("")

    if profile:
        lines.append("## Full pipeline profile (all backbones resident)")
        lines.append("")
        lines.append(f"- Backend: `{profile.get('backend', '?')}`")
        lines.append(f"- Samples: `{profile.get('n_samples', '?')}`")
        lines.append(
            f"- Median latency: `{profile.get('median_latency_s', 0.0)*1000:.1f} ms`"
        )
        lines.append(
            f"- Peak memory: `{profile.get('peak_memory_gib', 0.0):.3f} GiB`"
        )
        lines.append(
            f"- under_1gib_budget: **{'YES' if profile.get('under_1gib_budget') else 'NO'}**"
        )
        lines.append("")

    if confusion_png.exists():
        lines.append("## Out-of-fold confusion")
        lines.append("")
        lines.append(f"![Confusion matrix]({confusion_png.name})")
        lines.append("")

    if ablation_png.exists() or ablation_rows:
        lines.append("## Ablation")
        lines.append("")
        if ablation_png.exists():
            lines.append(f"![Ablation Macro-F1]({ablation_png.name})")
            lines.append("")
        if ablation_rows:
            lines.append("| configuration | n_features | mean_macro_f1 | mean_sarcasm_f1 |")
            lines.append("| --- | --- | --- | --- |")
            for r in ablation_rows:
                lines.append(
                    f"| {r.get('configuration','')} | {r.get('n_features','')} | "
                    f"{r.get('mean_macro_f1','')} | {r.get('mean_sarcasm_f1','')} |"
                )
            lines.append("")
            lines.append(
                "`aux_only` is `cos_TI` + `polarity_T` + `polarity_T_hat` + `clash`. "
                "`no_clip` is `Dsem`+`Fvt`+`cos_TI`+`polarity_T` (drops CLIP "
                "`polarity_T_hat`, `Dsen`, and `clash`). PDF §8.3 required showing that "
                "Dsem and Dsen improve the final model. If `aux_only` matches or "
                "beats the full GDRM row, that contribution is **not shown**. "
                "If `no_clip` sarcasm-F1 collapses, subtype F1 depended on CLIP."
            )
            lines.append("")

    if origin.get("splits"):
        lines.append("## Organic vs crafted captions (domain split)")
        lines.append("")
        lines.append(
            "Retrain 5-fold CV on each subset. `craft-*` ids are recaptioned faces, "
            "not Instagram caption–image pairs. OOF slice is mixed-train and can leak."
        )
        lines.append("")
        lines.append(
            "| split | n | sarcasm | clash P / R / F1 | clash acc | beats dummy acc | RQ2 Δ |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for name in ("all", "organic", "crafted"):
            split = (origin.get("splits") or {}).get(name) or {}
            binary = split.get("binary") or {}
            clash = binary.get("clash") or {}
            f1 = (clash.get("f1") or {}).get("mean")
            p = (clash.get("precision") or {}).get("mean")
            r = (clash.get("recall") or {}).get("mean")
            acc = (clash.get("accuracy") or {}).get("mean")
            prf = (
                f"{p:.3f} / {r:.3f} / {f1:.3f}"
                if f1 is not None and p is not None and r is not None
                else "—"
            )
            acc_s = f"{acc:.4f}" if acc is not None else "—"
            beats = clash.get("beats_dummy_accuracy")
            beats_s = {True: "YES", False: "NO"}.get(beats, "—")
            split_delta = split.get("rq2_delta")
            delta_s = f"{split_delta:+.3f}" if split_delta is not None else "—"
            lines.append(
                f"| {name} | {split.get('n', '')} | {split.get('n_sarcasm', '')} | "
                f"{prf} | {acc_s} | {beats_s} | {delta_s} |"
            )
        lines.append("")
        org_delta = origin.get("organic_rq2_delta")
        org_ok = origin.get("organic_rq2_meets_10pp")
        if org_delta is not None:
            lines.append(
                f"- Organic-only RQ2 Δ: **{org_delta:+.4f}** "
                f"({'meets' if org_ok else 'does not meet'} ≥10 pp)."
            )
            lines.append("")

    h3 = str(heavy.get("h3") or "NOT_RUN")
    if heavy:
        lines.append("## Hypothesis 3 (heavy multimodal comparison)")
        lines.append("")
        lines.append(f"- Model: `{heavy.get('model', '?')}`")
        lines.append(f"- Ran: **{'YES' if heavy.get('ran') else 'NO'}**")
        if heavy.get("reason"):
            lines.append(f"- Reason: {heavy.get('reason')}")
        lines.append(f"- Samples: `{heavy.get('n_samples', 0)}`")
        lines.append(f"- Heavy 5-class accuracy: **{heavy.get('heavy_accuracy')}**")
        lines.append(f"- Ours OOF accuracy (same ids): **{heavy.get('ours_accuracy')}**")
        lines.append(f"- Accuracy drop (heavy − ours): **{heavy.get('accuracy_drop')}**")
        lines.append(f"- Heavy median latency: **{heavy.get('heavy_median_latency_s')}** s")
        lines.append(f"- Ours staged median latency: **{heavy.get('ours_median_latency_s')}** s")
        lines.append(f"- Heavy peak VRAM: **{heavy.get('heavy_peak_memory_gib')}** GiB")
        lines.append(f"- Ours staged peak VRAM: **{heavy.get('ours_peak_memory_gib')}** GiB")
        if heavy.get("note"):
            lines.append(f"- {heavy['note']}")
        lines.append(f"- Hypothesis 3: **{h3}**")
        n_h3 = heavy.get("n_samples") or 0
        try:
            n_h3_i = int(n_h3)
        except (TypeError, ValueError):
            n_h3_i = 0
        if h3 == "PARTIAL_LOCAL_STANDIN":
            lines.append(
                "- The comparison partner is a **local stand-in**, not the heavy class "
                "named in the proposal (§3.4: Flamingo / BLIP-2 / Idefics). A run against "
                "it cannot settle H3, whatever the three components show."
            )
        if h3 == "PARTIAL_ZERO_SHOT_BASELINE":
            lines.append(
                "- The comparison partner was prompted **zero-shot**, not adapted to the "
                "5-class task. Its accuracy is not an accuracy ceiling, so the "
                "drop-< 5% component is not a valid test of the light pipeline."
            )
        if h3 == "UNDERPOWERED" or heavy.get("underpowered") or n_h3_i < 30:
            lines.append(
                "- This run does **not** settle H3: sample size is too small "
                "(or the load failed on a larger n)."
            )
        lines.append("")

    lines.append("## Proposal hypotheses (PDF §6.3)")
    lines.append("")
    lines.append("| Item | Result |")
    lines.append("| --- | --- |")
    staged_ok = bool(profile_staged.get("under_1gib_budget"))
    lines.append(
        f"| H1 memory < 1 GiB (staged peak) | **{'YES' if staged_ok else 'NO'}** "
        f"({profile_staged.get('peak_memory_gib', 0.0):.2f} GiB)"
        if profile_staged
        else "| H1 memory < 1 GiB (staged peak) | _not measured_ |"
    )
    h2_note = f"letter {'YES' if pdf_bar else 'NO'} ({mm_bin:.1%})"
    if dummy_not_sarc:
        h2_note += f"; dummy {dummy_not_sarc}; beats dummy {'YES' if beats_dummy else 'NO'}"
    lines.append(
        f"| H2 sarcasm accuracy > 70% | **{'YES' if beats_dummy else 'NO as detection'}** "
        f"({h2_note}) |"
    )
    lines.append(f"| H3 vs heavy model (<1 GiB, faster, drop <5%) | **{h3}** |")
    lines.append(
        f"| RQ2 multimodal sarcasm F1 ≥10 pp vs unimodal | **{'YES' if passes else 'NO'}** "
        f"({delta*100:+.1f} pp) — research question, not H3 |"
    )
    lines.append("| Training-free backbones | YES (`assert_frozen`) |")
    lines.append("| §8.3 Dsem/Dsen improve the model | see ablation (null if aux_only ≈ full) |")
    lines.append("")

    lines.append("## Deviations from the proposal PDF")
    lines.append("")
    lines.append(
        "- **ParsBERT** is named for Dsen (§8.1, §9). The text polarity head is "
        "`cardiffnlp/twitter-xlm-roberta-base-sentiment`. Image polarity is CLIP "
        "smile/sad, not ParsBERT on SmolVLM `T̂`."
    )
    lines.append(
        "- **H1** is measured as staged extract (one backbone resident). The PDF "
        "does not define staging. Dashboard with all towers loaded exceeds 1 GiB."
    )
    lines.append(
        "- **§5** specifies Instagram text–image pairs. Crafted recaptions "
        "(`craft-*`) keep the photo and replace the caption."
    )
    lines.append(
        "- **§8.2 scenario 2** (implicit / common-knowledge sarcasm) is not evaluated."
    )
    lines.append(
        "- **H3** default local VLM is `HuggingFaceTB/SmolVLM-Instruct` (2.2B). "
        "`Qwen2-VL-2B-Instruct` crashed on this Windows host while loading shards."
    )
    lines.append("")

    lines.append("## How to read the scores")
    lines.append("")
    lines.append(
        "- **5-class quality** is reported as **macro-F1**, not accuracy. "
        "The labeled set is imbalanced (most posts are `positive`), so a majority "
        "dummy can beat overall accuracy while losing the rare classes."
    )
    lines.append(
        "- **Hypothesis 2** in the PDF is sarcasm **accuracy > 70%**. That bar can "
        "be met by always predicting not-sarcasm when sarcasm is rare. Detection "
        "evidence is precision/recall/F1 and whether accuracy beats the "
        "always-not-sarcasm dummy. A previous F1≥0.40 conjunct was not in the PDF."
    )
    lines.append(
        "- **`clash`** is zero unless `|polarity_T| ≥ 0.20` and `|polarity_T_hat| ≥ 0.15` "
        "(same floors as the inference polarity-conflict rule). Weak smile/sad scores "
        "are not counted as a text–image clash. Hashtag-only and prompt-spam captions "
        "have text polarity zeroed so they cannot fire clash. The clash cut is "
        "CV-tuned for **accuracy** (PDF §6.3(2))."
    )
    lines.append(
        "- **`polarity_T_hat`** in GDRM is **CLIP facial affect** (smile vs sad), "
        "not the polarity of the SmolVLM sentence. `T̂` still feeds `Dsem` and `Fvt`. "
        "Original GDCNet scores `T̂` with a text sentiment head; SmolVLM-256M captions "
        "under a 1 GiB budget are often bland, so CLIP zero-shot smile/sad is the "
        "visual-affect stand-in (facial cues are a documented substitute when captions "
        "omit expression)."
    )
    lines.append(
        "- **Sarcasm-subtype F1** can still use CLIP: `polarity_T_hat` is smile/sad "
        "in the feature vector. Kappa, if computed, is only on the overlap file "
        "vs gold `blind-relabel` rows (`reports/iaa.md`), not on the full eval set. "
        "Most non-sarcasm eval rows were not blindly relabeled."
    )
    lines.append(
        "- **Neutral F1** is the weakest class (thin captions / ads). It is not the "
        "sarcasm hypothesis; do not lead with it."
    )
    lines.append(
        "- **Taarof**, honorific mismatches, and purely cultural irony without a "
        "text–image polarity clash are **out of scope** of the 5-class GDRM contract."
    )
    lines.append("")

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compose the final Markdown report.")
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--profile-staged", type=Path, default=DEFAULT_PROFILE_STAGED)
    parser.add_argument("--sarcasm", type=Path, default=DEFAULT_SARCASM)
    parser.add_argument("--baseline-sarcasm", type=Path, default=DEFAULT_BASELINE_SARCASM)
    parser.add_argument("--ablation-png", type=Path, default=DEFAULT_ABLATION_PNG)
    parser.add_argument("--ablation-csv", type=Path, default=DEFAULT_ABLATION_CSV)
    parser.add_argument("--confusion", type=Path, default=DEFAULT_CONFUSION_PNG)
    parser.add_argument("--heavy", type=Path, default=DEFAULT_HEAVY)
    parser.add_argument("--origin-split", type=Path, default=DEFAULT_ORIGIN_SPLIT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    body = render_report(
        metrics_csv=args.metrics,
        baseline_csv=args.baseline,
        profile_json=args.profile,
        profile_staged_json=args.profile_staged,
        sarcasm_csv=args.sarcasm,
        baseline_sarcasm_csv=args.baseline_sarcasm,
        ablation_png=args.ablation_png,
        ablation_csv=args.ablation_csv,
        confusion_png=args.confusion,
        heavy_compare_json=args.heavy,
        origin_split_json=args.origin_split,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(body, encoding="utf-8")
    log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
