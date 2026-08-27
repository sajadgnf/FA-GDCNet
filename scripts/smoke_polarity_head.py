"""Smoke-test the dashboard path after a polarity-head swap.

Runs one real dataset post plus two hand-written captions through the full
pipeline (caption -> mCLIP -> polarity -> classifier -> label refinement) and
then through the text attention rollout, which is the part most sensitive to a
tokenizer / dtype change in the polarity head.

    python scripts/smoke_polarity_head.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")

from PIL import Image  # noqa: E402

from explain.rollout import attention_from_text  # noqa: E402
from inference.pipeline import Pipeline  # noqa: E402

DATASET = Path("datasets") / "persian_multimodal_irony.jsonl"
IMAGES = Path("datasets") / "raw" / "images"

PROBES = ("ناراحت است", "ناراحت نیستم", "به درک")


def _first_record_with_image() -> tuple[dict, Path]:
    with DATASET.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            path = IMAGES / f"{rec['post_id']}.jpg"
            if path.exists():
                return rec, path
    raise SystemExit("no dataset record with a local image")


def main() -> int:
    rec, image_path = _first_record_with_image()
    print(f"post {rec['post_id']}  gold={rec.get('label')}")

    pipe = Pipeline.from_pretrained()
    image = Image.open(image_path).convert("RGB")

    captions = [rec.get("text", "")[:120], *PROBES]
    for text in captions:
        if not text.strip():
            continue
        pred, feats, caption = pipe.explain(text, image)
        print()
        print(f"text:    {text[:70]}")
        print(f"caption: {caption[:70]}")
        print(
            f"  p_T={feats.polarity_T:+.3f}  p_That={feats.polarity_T_hat:+.3f}"
            f"  Dsem={feats.Dsem:.3f}  Dsen={feats.Dsen:.3f}"
        )
        print(f"  -> {pred.label}  conf={pred.confidence:.2f}")

    tokens, scores = attention_from_text(pipe.bundle, "ناراحت نیستم")
    print()
    print(f"rollout tokens: {tokens}")
    print(f"rollout scores: dtype={scores.dtype} min={scores.min():.4f} max={scores.max():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
