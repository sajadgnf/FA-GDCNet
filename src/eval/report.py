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
DEFAULT_ABLATION_PNG = Path("reports") / "ablation.png"
DEFAULT_OUT = Path("reports") / "REPORT.md"

SARCASM_LABELS = ("positive_sarcasm", "negative_sarcasm")
SARCASM_F1_DELTA_FLOOR = 0.10  # spec: ≥10 percentage points


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


def render_report(
    *,
    metrics_csv: Path,
    baseline_csv: Path,
    profile_json: Path,
    ablation_png: Path,
) -> str:
    metrics = _read_csv(metrics_csv)
    baseline = _read_csv(baseline_csv)
    profile = json.loads(profile_json.read_text(encoding="utf-8")) if profile_json.exists() else {}

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
    else:
        lines.append("_metrics.csv not found_")
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

    if profile:
        lines.append("## Inference profile")
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

    if ablation_png.exists():
        rel = ablation_png.name
        lines.append("## Ablation")
        lines.append("")
        lines.append(f"![Ablation Macro-F1]({rel})")
        lines.append("")

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compose the final Markdown report.")
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--ablation-png", type=Path, default=DEFAULT_ABLATION_PNG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    body = render_report(
        metrics_csv=args.metrics,
        baseline_csv=args.baseline,
        profile_json=args.profile,
        ablation_png=args.ablation_png,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(body, encoding="utf-8")
    log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
