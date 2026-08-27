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
    val = _read_footer_value(rows, "mean_accuracy_dsem_rule")
    if val:
        return float(val)
    for r in rows:
        if str(r.get("fold")) == "mean±std":
            try:
                col = r.get("dsem_rule_accuracy") or r.get("accuracy")
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
            f"macro-F1 **{dummy_f1 or '—'}**. Lead with macro-F1, not accuracy."
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

    lines.append("### Per-class F1 (mean±std)")
    lines.append("")
    lines.append("| class | F1 |")
    lines.append("| --- | --- |")
    for lbl in LABELS:
        cell = _mean_cell(metrics, f"f1_{lbl}")
        lines.append(f"| `{lbl}` | {cell or '—'} |")
    lines.append("")

    lines.append("## Unimodal ParsBERT baseline (same folds)")
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
    lines.append("## Sarcasm-F1 improvement check")
    lines.append("")
    lines.append(
        f"- Multimodal sarcasm-F1 (macro of {', '.join(SARCASM_LABELS)}): **{mm_sarcasm:.4f}**"
    )
    lines.append(f"- Unimodal baseline sarcasm-F1: **{bs_sarcasm:.4f}**")
    lines.append(f"- Δ = **{delta:+.4f}** ({delta * 100:+.2f} percentage points)")
    lines.append(
        f"- Meets ≥10 pp hypothesis: **{'YES' if passes else 'NO'}**"
    )
    lines.append("")

    mm_bin = _read_sarcasm_mean(sarcasm_rows)
    bs_bin = _read_sarcasm_mean(baseline_sarcasm_rows)
    bin_delta = mm_bin - bs_bin
    bin_passes = mm_bin >= SARCASM_ACC_FLOOR
    lines.append("## Binary sarcasm detection (proposal Hypothesis 2)")
    lines.append("")
    lines.append(
        "- Dsem threshold rule (CV-tuned, interpretable): "
        f"**{_read_footer_value(sarcasm_rows, 'mean_accuracy_dsem_rule') or f'{mm_bin:.4f}'}**"
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
    lines.append(f"- Unimodal baseline binary accuracy: **{bs_bin:.4f}**")
    lines.append(f"- Meets ≥70% accuracy (Dsem rule): **{'YES' if bin_passes else 'NO'}**")
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
            lines.append("| configuration | n_features | mean_macro_f1 |")
            lines.append("| --- | --- | --- |")
            for r in ablation_rows:
                lines.append(
                    f"| {r.get('configuration','')} | {r.get('n_features','')} | "
                    f"{r.get('mean_macro_f1','')} |"
                )
            lines.append("")
            lines.append(
                "`aux_only` is `cos_TI` + `polarity_T` + `polarity_T_hat`. "
                "Dsen is partly redundant with those polarities, so core-signal "
                "deltas can be small. The proposal claim is **multimodal vs unimodal**, "
                "not Dsen vs aux."
            )
            lines.append("")

    lines.append("## Proposal claims checklist")
    lines.append("")
    lines.append("| Claim | Result |")
    lines.append("| --- | --- |")
    lines.append(f"| GDCNet-FA (Dsem/Dsen/Fvt) implemented | YES (see ablation) |")
    lines.append(f"| Training-free backbones | YES (`assert_frozen`) |")
    lines.append(f"| Binary sarcasm accuracy ≥ 70% (Dsem rule) | **{'YES' if bin_passes else 'NO'}** ({mm_bin:.1%}) |")
    lines.append(f"| Multimodal ≥10 pp over unimodal (sarcasm F1) | **{'YES' if passes else 'NO'}** ({delta*100:+.1f} pp) |")
    staged_ok = bool(profile_staged.get("under_1gib_budget"))
    lines.append(
        f"| Peak VRAM ≤ 1 GiB (staged) | **{'YES' if staged_ok else 'NO'}** "
        f"({profile_staged.get('peak_memory_gib', 0.0):.2f} GiB)"
        if profile_staged
        else "| Peak VRAM ≤ 1 GiB (staged) | _not measured_ |"
    )
    lines.append("")

    lines.append("## How to read the scores (defense notes)")
    lines.append("")
    lines.append(
        "- **5-class quality** is reported as **macro-F1**, not accuracy. "
        "The labeled set is imbalanced (most posts are `positive`), so a majority "
        "dummy can beat overall accuracy while losing the rare classes."
    )
    lines.append(
        "- **Binary Dsem accuracy** is the metric named in Hypothesis 2. "
        "Sarcasm is the minority class (~12%), so also report binary sarcasm F1; "
        "high accuracy alone does not mean sarcasm is detected that often."
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
        "- **Sarcasm-subtype F1** is an **upper bound** until the sarcasm gold labels "
        "are fully hand-reviewed: visual affect informed both the gold rule and `polarity_T_hat`."
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
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(body, encoding="utf-8")
    log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
