"""Inter-annotator agreement: Cohen's kappa on overlapping samples.

This module is pure stdlib so it can be unit-tested without numpy.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .schema import LABELS


def cohens_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """Compute Cohen's kappa for two equal-length label sequences.

    Returns 0.0 when expected agreement is 1.0 (degenerate single-class case)
    rather than raising, so the labeling tool can degrade gracefully when an
    annotator labels everything the same way during bootstrap.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("label sequences must be the same length")
    n = len(labels_a)
    if n == 0:
        return 0.0

    observed = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n

    pa: dict[str, float] = defaultdict(float)
    pb: dict[str, float] = defaultdict(float)
    for a in labels_a:
        pa[a] += 1.0 / n
    for b in labels_b:
        pb[b] += 1.0 / n
    expected = sum(pa[k] * pb[k] for k in set(pa) | set(pb))
    if expected >= 1.0:
        return 0.0
    return (observed - expected) / (1.0 - expected)


def pairwise_kappa(annotator_labels: dict[str, dict[str, str]]) -> dict[tuple[str, str], float]:
    """Compute Cohen's kappa for every pair of annotators on their shared posts.

    Args:
        annotator_labels: mapping `annotator_id -> {post_id: label}`.

    Returns:
        mapping `(annotator_a, annotator_b) -> kappa` for annotators with at
        least one overlapping post. The pair tuple is sorted lexicographically.
    """
    annotators = sorted(annotator_labels.keys())
    out: dict[tuple[str, str], float] = {}
    for i in range(len(annotators)):
        for j in range(i + 1, len(annotators)):
            a, b = annotators[i], annotators[j]
            shared = sorted(set(annotator_labels[a]) & set(annotator_labels[b]))
            if not shared:
                continue
            labels_a = [annotator_labels[a][p] for p in shared]
            labels_b = [annotator_labels[b][p] for p in shared]
            out[(a, b)] = cohens_kappa(labels_a, labels_b)
    return out


def load_annotations(records: Iterable[dict]) -> dict[str, dict[str, str]]:
    """Group annotation records into `{annotator_id: {post_id: label}}`."""
    out: dict[str, dict[str, str]] = defaultdict(dict)
    for r in records:
        annotator = r.get("annotator_id") or "_unknown_"
        post_id = r["post_id"]
        label = r["label"]
        if label not in LABELS:
            continue
        out[annotator][post_id] = label
    return dict(out)


def compute_and_write(
    annotation_records_path: str | Path,
    report_path: str | Path,
) -> Path:
    """Helper used by the labeling CLI: read raw records, compute kappa, write report."""
    annotation_records_path = Path(annotation_records_path)
    report_path = Path(report_path)
    raw: list[dict] = []
    with annotation_records_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw.append(json.loads(line))
    annotator_labels = load_annotations(raw)
    kappas = pairwise_kappa(annotator_labels)
    report = render_iaa_report(kappas, annotator_labels=annotator_labels)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return report_path


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def gold_blind_records(dataset_path: Path, gold_annotator: str) -> list[dict]:
    """Independent labels from the canonical jsonl after a blind pass.

    Uses ``label`` only on rows tagged ``blind-relabel``. The jsonl
    ``annotators`` list is not treated as a second rater.
    """
    from .tags import BLIND_REVIEW_TAG

    out: list[dict] = []
    for row in _load_jsonl(dataset_path):
        tags = [str(a) for a in (row.get("annotators") or [])]
        if BLIND_REVIEW_TAG not in tags:
            continue
        label = row.get("label")
        if label not in LABELS:
            continue
        out.append(
            {
                "post_id": str(row["post_id"]),
                "annotator_id": gold_annotator,
                "label": label,
            }
        )
    return out


def render_iaa_report(
    kappas: dict[tuple[str, str], float],
    *,
    annotator_labels: dict[str, dict[str, str]],
) -> str:
    """Render a Markdown IAA report for `reports/iaa.md`."""
    lines: list[str] = ["# Inter-Annotator Agreement", "", "## Per-pair Cohen's kappa", ""]
    if not kappas:
        lines.append("_No overlapping samples between two independent annotators._")
        lines.append("")
        lines.append(
            "`annotators` on a canonical jsonl row is a tag list (who touched the "
            "row), not two independent labels. Kappa is computed only from "
            "blind-relabel gold vs `--second` (`datasets/iaa_second.jsonl`)."
        )
    else:
        lines.append("| Annotator A | Annotator B | Overlap | Kappa |")
        lines.append("| --- | --- | --- | --- |")
        for (a, b), k in sorted(kappas.items()):
            overlap = len(set(annotator_labels[a]) & set(annotator_labels[b]))
            lines.append(f"| {a} | {b} | {overlap} | {k:.4f} |")

    lines += ["", "## Per-annotator counts", "", "| Annotator | Labeled posts |", "| --- | --- |"]
    for ann, labels in sorted(annotator_labels.items()):
        lines.append(f"| {ann} | {len(labels)} |")
    return "\n".join(lines) + "\n"


def write_from_gold_and_second(
    *,
    dataset: Path,
    second: Path,
    report_path: Path,
    gold_annotator: str,
) -> Path:
    gold = gold_blind_records(dataset, gold_annotator)
    second_rows = _load_jsonl(second)
    annotator_labels = load_annotations(gold + second_rows)
    kappas = pairwise_kappa(annotator_labels)
    body = render_iaa_report(kappas, annotator_labels=annotator_labels)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(body, encoding="utf-8")
    return report_path


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Cohen's kappa from blind gold jsonl vs a second-annotator file."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets") / "persian_multimodal_irony.jsonl",
    )
    parser.add_argument(
        "--second",
        type=Path,
        default=Path("datasets") / "iaa_second.jsonl",
    )
    parser.add_argument("--gold-annotator", default="sjjd6502")
    parser.add_argument("--out", type=Path, default=Path("reports") / "iaa.md")
    args = parser.parse_args(argv)
    path = write_from_gold_and_second(
        dataset=args.dataset,
        second=args.second,
        report_path=args.out,
        gold_annotator=args.gold_annotator,
    )
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
