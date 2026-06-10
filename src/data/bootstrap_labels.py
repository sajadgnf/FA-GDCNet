"""Weak text-only labels for scraped raw pools (bootstrap before manual review).

Uses ParsBERT polarity on captions only. Sarcasm classes are never assigned
automatically — use `tasks.py label` to refine.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .schema import LABELS, DatasetRecord, iter_dataset, write_dataset

log = logging.getLogger(__name__)

DEFAULT_RAW = Path("datasets") / "raw" / "following.jsonl"
DEFAULT_DATASET = Path("datasets") / "persian_multimodal_irony.jsonl"

_POSITIVE = frozenset({"DELIGHT", "HAPPY", "POSITIVE", "positive"})
_NEGATIVE = frozenset({"SAD", "ANGRY", "NEGATIVE", "negative"})
_NEUTRAL = frozenset({"NEUTRAL", "neutral"})


def _load_id2label(model) -> dict[int, str]:
    cfg = model.config
    raw = getattr(cfg, "id2label", None) or {}
    return {int(k): str(v) for k, v in raw.items()}


def _weak_label_from_probs(probs, id2label: dict[int, str]) -> str:
    idx = int(probs.argmax())
    name = id2label.get(idx, str(idx)).upper()
    if name in _POSITIVE or "POS" in name or name in {"1"}:
        return "positive"
    if name in _NEGATIVE or "NEG" in name or "ANGR" in name or "SAD" in name:
        return "negative"
    if name in _NEUTRAL:
        return "neutral"
    # Snappfood 5-way: map extremes, middle -> neutral
    if idx == 0:
        return "negative"
    if idx >= 3:
        return "positive"
    return "neutral"


def _load_parsbert_polarity(*, device: str | None = None):
    """Load only ParsBERT for weak labeling (avoids loading SmolVLM/M-CLIP)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from inference.models import DEFAULT_PARSBERT_POLARITY_ID, _freeze

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(DEFAULT_PARSBERT_POLARITY_ID)
    _freeze(model)
    model.to(dev)
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_PARSBERT_POLARITY_ID)
    return model, tokenizer, dev


def _polarity_probs(model, tokenizer, device: str, text: str):
    import torch
    from torch.nn.functional import softmax

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    return softmax(logits, dim=-1).cpu().numpy()[0]


def _keyword_weak_label(caption: str) -> str:
    """Fallback when transformer load fails."""
    c = caption
    pos = ("خوشح", "عالی", "زیب", "دوست", "شاد", "❤", "🥰", "😍", "ممنون", "love")
    neg = ("ناراح", "غم", "بد", "عصب", "اشک", " hate", "angry", "sad")
    p = sum(1 for w in pos if w in c)
    n = sum(1 for w in neg if w in c)
    if p > n:
        return "positive"
    if n > p:
        return "negative"
    return "neutral"


def bootstrap_raw_pool(
    raw_path: Path,
    *,
    dataset_path: Path = DEFAULT_DATASET,
    annotator: str = "weak_parsbert",
    method: str = "auto",
) -> int:
    raw_rows: list[dict] = []
    with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_rows.append(json.loads(line))
    if not raw_rows:
        log.warning("no rows in %s", raw_path)
        return 0

    existing: dict[str, DatasetRecord] = {}
    if dataset_path.exists():
        for rec in iter_dataset(dataset_path):
            existing[rec.post_id] = rec

    use_keywords = method == "keywords"
    id2label: dict[int, str] = {}
    model = tokenizer = device = None
    if method in {"auto", "parsbert"} and not use_keywords:
        try:
            model, tokenizer, device = _load_parsbert_polarity()
            id2label = _load_id2label(model)
            log.info("weak labeling with ParsBERT on %d raw posts", len(raw_rows))
        except Exception as exc:  # noqa: BLE001
            log.warning("ParsBERT unavailable (%s); using keyword fallback", exc)
            use_keywords = True
    if use_keywords:
        annotator = "weak_keywords"
        log.info("weak labeling with keywords on %d raw posts", len(raw_rows))

    added = 0
    for row in raw_rows:
        post_id = row["post_id"]
        if post_id in existing:
            continue
        if use_keywords:
            label = _keyword_weak_label(row["caption"])
        else:
            probs = _polarity_probs(model, tokenizer, device, row["caption"])
            label = _weak_label_from_probs(probs, id2label)
        if label not in LABELS:
            label = "neutral"
        existing[post_id] = DatasetRecord(
            post_id=post_id,
            caption=row["caption"],
            image_path=row["image_path"],
            label=label,
            annotators=[annotator],
            kappa=None,
        )
        added += 1

    write_dataset(dataset_path, existing.values())
    log.info("added %d weak-labeled records (%d total in %s)", added, len(existing), dataset_path)
    return added


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Weak-label a raw scrape pool with ParsBERT.")
    parser.add_argument("--input", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--annotator", default="weak_parsbert")
    parser.add_argument(
        "--method",
        choices=("auto", "parsbert", "keywords"),
        default="auto",
        help="auto tries ParsBERT then keywords; keywords avoids torch.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if not args.input.is_file():
        raise SystemExit(f"raw pool not found: {args.input}")
    bootstrap_raw_pool(
        args.input,
        dataset_path=args.dataset,
        annotator=args.annotator,
        method=args.method,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
