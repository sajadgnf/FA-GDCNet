#!/usr/bin/env python3
"""Score candidate sentiment heads on the polarity benchmark cases.

    python scripts/bench_polarity_models.py

Reuses the case lists from `bench_polarity_heads.py` so tuning and held-out
splits stay identical across experiments. Each head is collapsed to a signed
score in [-1, +1] via (p_pos - p_neg), independent of its own label names.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (str(SRC), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from bench_polarity_heads import HELDOUT_CASES, TUNE_CASES  # noqa: E402

CANDIDATES = [
    "HooshvareLab/bert-fa-base-uncased-sentiment-snappfood",
    "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    "tabularisai/multilingual-sentiment-analysis",
    "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
]

_POS = {
    "HAPPY", "POSITIVE", "POS", "LABEL_2", "DELIGHTED",
    "VERY POSITIVE", "VERY_POSITIVE",
}
_NEG = {
    "SAD", "NEGATIVE", "NEG", "LABEL_0", "FURIOUS", "ANGRY",
    "VERY NEGATIVE", "VERY_NEGATIVE",
}
_NEU = {"NEUTRAL", "LABEL_1"}


def _ok(score: float, want: int) -> bool:
    if want == 0:
        return abs(score) < 0.3
    return score * want > 0 and abs(score) >= 0.05


def main() -> int:
    import torch
    from torch.nn.functional import softmax
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cases = [(t, w, "tune") for t, w in TUNE_CASES] + [
        (t, w, "held") for t, w in HELDOUT_CASES
    ]
    n_tune = len(TUNE_CASES)
    n_held = len(HELDOUT_CASES)

    for model_id in CANDIDATES:
        print(f"\n=== {model_id} ===")
        try:
            tok = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForSequenceClassification.from_pretrained(model_id)
        except Exception as exc:  # noqa: BLE001
            print(f"  load failed: {type(exc).__name__}: {exc}")
            continue
        model.eval().to(device)
        id2label = {int(i): str(n).upper().replace("-", " ") for i, n in model.config.id2label.items()}
        unknown = [n for n in id2label.values() if n not in _POS | _NEG | _NEU]
        print(f"  id2label: {model.config.id2label}")
        if unknown:
            print(f"  skipped: unmapped labels {unknown}")
            del model
            continue

        hits = {"tune": 0, "held": 0}
        detail = []
        for text, want, split in cases:
            inputs = tok(
                text, return_tensors="pt", padding=True, truncation=True, max_length=256
            ).to(device)
            with torch.no_grad():
                probs = softmax(model(**inputs).logits, dim=-1).cpu().numpy()[0]
            pos = sum(float(probs[i]) for i, n in id2label.items() if n in _POS)
            neg = sum(float(probs[i]) for i, n in id2label.items() if n in _NEG)
            score = pos - neg
            ok = _ok(score, want)
            hits[split] += ok
            detail.append((split, ok, score, want, text))

        print(f"  tune {hits['tune']}/{n_tune}   held {hits['held']}/{n_held}   "
              f"total {hits['tune'] + hits['held']}/{n_tune + n_held}")
        for split, ok, score, want, text in detail:
            if not ok:
                tag = "H" if split == "held" else " "
                print(f"    {tag} MISS {score:+.3f}  want {want:+d}  {text}")

        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
