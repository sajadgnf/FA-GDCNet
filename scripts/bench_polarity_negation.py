#!/usr/bin/env python3
"""Check whether de-negation rewriting improves the chosen sentiment head.

    python scripts/bench_polarity_negation.py

Negation is grammatical, not lexical: instead of listing phrases, rewrite the
caption to its affirmative form, score that with the *same* model, and flip the
sign. Reports raw vs de-negated accuracy on the shared benchmark splits.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (str(SRC), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from bench_polarity_heads import (  # noqa: E402
    HELDOUT_CASES,
    TUNE_CASES,
    denegate,
    has_negation,
)

MODEL_ID = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

_POS = {"POSITIVE", "POS", "LABEL_2", "HAPPY"}
_NEG = {"NEGATIVE", "NEG", "LABEL_0", "SAD"}


def _ok(score: float, want: int) -> bool:
    if want == 0:
        return abs(score) < 0.3
    return score * want > 0 and abs(score) >= 0.05


def main() -> int:
    import torch
    from torch.nn.functional import softmax
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval().to(device)
    id2label = {int(i): str(n).upper() for i, n in model.config.id2label.items()}

    def score(text: str) -> float:
        inputs = tok(
            text, return_tensors="pt", padding=True, truncation=True, max_length=256
        ).to(device)
        with torch.no_grad():
            probs = softmax(model(**inputs).logits, dim=-1).cpu().numpy()[0]
        pos = sum(float(probs[i]) for i, n in id2label.items() if n in _POS)
        neg = sum(float(probs[i]) for i, n in id2label.items() if n in _NEG)
        return pos - neg

    cases = [(t, w, "tune") for t, w in TUNE_CASES] + [
        (t, w, "held") for t, w in HELDOUT_CASES
    ]

    variants = {"raw": {"tune": 0, "held": 0}, "deneg-flip": {"tune": 0, "held": 0}}
    detail = []
    for text, want, split in cases:
        raw = score(text)
        neg_flag = has_negation(text)
        if neg_flag:
            rewritten = denegate(text)
            flipped = -0.85 * score(rewritten)
        else:
            rewritten, flipped = text, raw
        variants["raw"][split] += _ok(raw, want)
        variants["deneg-flip"][split] += _ok(flipped, want)
        detail.append((split, text, want, raw, flipped, neg_flag, rewritten))

    n_t, n_h = len(TUNE_CASES), len(HELDOUT_CASES)
    print(f"model: {MODEL_ID}\n")
    for name, hits in variants.items():
        total = hits["tune"] + hits["held"]
        print(f"{name:<12} tune {hits['tune']}/{n_t}  held {hits['held']}/{n_h}  total {total}/{n_t + n_h}")

    print("\nnegation cases only:")
    for split, text, want, raw, flipped, neg_flag, rewritten in detail:
        if neg_flag:
            print(
                f"  want {want:+d}  raw {raw:+.3f}  flip {flipped:+.3f}  "
                f"{text}   ->   {rewritten}"
            )

    print("\nremaining misses with de-negation:")
    for split, text, want, raw, flipped, neg_flag, _ in detail:
        if not _ok(flipped, want):
            tag = "H" if split == "held" else " "
            print(f"  {tag} {flipped:+.3f}  want {want:+d}  {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
