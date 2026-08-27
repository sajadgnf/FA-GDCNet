"""Zero-shot CLIP affect for dataset images (smile vs sad).

SmolVLM descriptions rarely mention sadness, so gold retagging uses CLIP
ViT-B/32 (already the M-CLIP vision tower) to score every photo once.

    python -m data.image_affect
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CLIP_ID = "openai/clip-vit-base-patch32"
CACHE = Path("artifacts") / "stages" / "image_affect.jsonl"

POS_PROMPTS = (
    "a photo of a person smiling with a happy face",
    "people laughing cheerfully",
    "a bright cheerful selfie with a big smile",
)
NEG_PROMPTS = (
    "a photo of a person looking sad or crying",
    "a distressed unhappy frowning face",
    "a gloomy bleak unhappy scene",
)
_POS_PROMPTS = POS_PROMPTS
_NEG_PROMPTS = NEG_PROMPTS

# 2-way (smile vs sad) probability needed to commit, after CLIP logit_scale.
_COMMIT = 0.58


def _index_jsonl(path: Path) -> dict:
    out: dict = {}
    if not path.is_file():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[str(row["post_id"])] = row
    return out


def polarity_vector(pos: float, neg: float) -> list[float]:
    """GDRM ``(p_negative, p_positive)`` from CLIP 2-way smile/sad scores."""
    return [float(neg), float(pos)]


def _hat_from_scores(pos: float, neg: float) -> str:
    """``pos``/``neg`` are 2-way probabilities (they sum to 1)."""
    if pos >= _COMMIT:
        return "pos"
    if neg >= _COMMIT:
        return "neg"
    return "neu"


def score_images(
    rows: list[dict],
    *,
    cache: Path = CACHE,
    device: str | None = None,
    batch_size: int = 16,
) -> dict:
    """Return ``{post_id: {hat, pos, neg}}``, writing/resuming ``cache``."""
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    done = _index_jsonl(cache)
    pending = [r for r in rows if str(r["post_id"]) not in done]
    if not pending:
        return done

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if str(dev).startswith("cuda") else torch.float32
    model = CLIPModel.from_pretrained(CLIP_ID, torch_dtype=dtype)
    proc = CLIPProcessor.from_pretrained(CLIP_ID)
    model.eval()
    model.to(dev)

    prompts = list(_POS_PROMPTS) + list(_NEG_PROMPTS)
    text_inputs = proc(text=prompts, return_tensors="pt", padding=True).to(dev)
    with torch.no_grad():
        text_emb = model.get_text_features(**text_inputs)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = model.logit_scale.exp().float()

    cache.parent.mkdir(parents=True, exist_ok=True)
    n_pos = len(_POS_PROMPTS)

    with cache.open("a", encoding="utf-8") as fh:
        for i in range(0, len(pending), batch_size):
            chunk = pending[i : i + batch_size]
            images = []
            keep = []
            for row in chunk:
                path = Path(str(row.get("image_path") or ""))
                if not path.is_file():
                    rec = {
                        "post_id": str(row["post_id"]),
                        "pos": 0.0,
                        "neg": 0.0,
                        "hat": "neu",
                        "missing": True,
                    }
                    done[rec["post_id"]] = rec
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    continue
                try:
                    images.append(Image.open(path).convert("RGB"))
                    keep.append(row)
                except OSError:
                    rec = {
                        "post_id": str(row["post_id"]),
                        "pos": 0.0,
                        "neg": 0.0,
                        "hat": "neu",
                        "missing": True,
                    }
                    done[rec["post_id"]] = rec
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if not images:
                continue
            inputs = proc(images=images, return_tensors="pt", padding=True)
            pixel = inputs["pixel_values"].to(dev, dtype=dtype)
            with torch.no_grad():
                img_emb = model.get_image_features(pixel_values=pixel)
                img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                sim = img_emb.float() @ text_emb.float().T
                pos_logit = scale * sim[:, :n_pos].mean(dim=1)
                neg_logit = scale * sim[:, n_pos:].mean(dim=1)
                two = torch.stack([neg_logit, pos_logit], dim=1).softmax(dim=-1)

            for row, p in zip(keep, two.cpu().tolist()):
                neg, pos = float(p[0]), float(p[1])
                rec = {
                    "post_id": str(row["post_id"]),
                    "pos": pos,
                    "neg": neg,
                    "hat": _hat_from_scores(pos, neg),
                }
                done[rec["post_id"]] = rec
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"image_affect {min(i + batch_size, len(pending))}/{len(pending)}", flush=True)

    return done


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets") / "persian_multimodal_irony.jsonl",
    )
    args = parser.parse_args(argv)
    rows: list[dict] = []
    with args.dataset.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    scored = score_images(rows)
    from collections import Counter

    print("hats", dict(Counter(r["hat"] for r in scored.values())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
