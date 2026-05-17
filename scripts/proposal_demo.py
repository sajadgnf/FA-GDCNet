#!/usr/bin/env python3
"""End-to-end demo of FA-GDCNet against the thesis proposal (no Instagram required).

Creates a small synthetic Persian multimodal dataset, trains the sklearn head on
GDRM-style features, runs eval/report tooling, and prints a proposal-fit checklist.

Usage (from repo root):
    python scripts/proposal_demo.py
    python scripts/proposal_demo.py --try-models   # also run one real backbone inference (downloads weights)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.schema import LABELS, write_dataset, DatasetRecord  # noqa: E402
from inference.classifier import train, DEFAULT_CLF, DEFAULT_FEATURES  # noqa: E402
from inference.gdrm import FEATURE_NAMES  # noqa: E402

DEMO_DIR = ROOT / "datasets" / "demo"
DEMO_JSONL = ROOT / "datasets" / "persian_multimodal_irony.jsonl"
DEMO_IMG_DIR = DEMO_DIR / "images"


def _make_image(path: Path, color: tuple[int, int, int], text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (224, 224), color=color)
    draw = ImageDraw.Draw(img)
    draw.text((10, 100), text[:20], fill=(255, 255, 255))
    img.save(path)


def bootstrap_demo_dataset(n_per_class: int = 12) -> Path:
    """Create labeled demo posts with Persian captions and solid-color images."""
    captions = {
        "positive": "چه روز زیبایی! خیلی خوشحالم.",
        "negative": "روز بدی بود، خیلی ناراحتم.",
        "neutral": "امروز هوا ابری است.",
        "positive_sarcasm": "چه روز عالی‌ای! (کنایه: متن مثبت، تصویر منفی)",
        "negative_sarcasm": "عالی شد! (کنایه: متن منفی ظاهری، تصویر شاد)",
    }
    colors = {
        "positive": (40, 180, 80),
        "negative": (180, 40, 40),
        "neutral": (120, 120, 120),
        "positive_sarcasm": (180, 40, 40),  # image contradicts positive text
        "negative_sarcasm": (40, 180, 80),
    }
    records: list[DatasetRecord] = []
    for label in LABELS:
        for i in range(n_per_class):
            pid = f"demo_{label}_{i}"
            img_path = DEMO_IMG_DIR / f"{pid}.jpg"
            _make_image(img_path, colors[label], label)
            records.append(
                DatasetRecord(
                    post_id=pid,
                    caption=captions[label],
                    image_path=str(img_path.relative_to(ROOT)),
                    label=label,
                    annotators=["demo"],
                    kappa=None,
                )
            )
    write_dataset(DEMO_JSONL, records)
    return DEMO_JSONL


def _feature_template(label: str, rng: np.random.Generator) -> np.ndarray:
    """Synthetic 6-D GDRM vector that separates classes for the demo."""
    noise = lambda s: float(rng.normal(0, s))
    base = {
        "positive": dict(Dsem=0.1, Dsen=0.1, Fvt=0.85, cos_TI=0.8, polarity_T=0.7, polarity_T_hat=0.65),
        "negative": dict(Dsem=0.15, Dsen=0.1, Fvt=0.8, cos_TI=0.75, polarity_T=-0.7, polarity_T_hat=-0.65),
        "neutral": dict(Dsem=0.2, Dsen=0.15, Fvt=0.75, cos_TI=0.5, polarity_T=0.05, polarity_T_hat=0.0),
        "positive_sarcasm": dict(Dsem=0.75, Dsen=0.9, Fvt=0.7, cos_TI=0.2, polarity_T=0.75, polarity_T_hat=-0.6),
        "negative_sarcasm": dict(Dsem=0.7, Dsen=0.85, Fvt=0.65, cos_TI=0.25, polarity_T=-0.5, polarity_T_hat=0.7),
    }[label]
    return np.array(
        [base[k] + noise(0.05) for k in FEATURE_NAMES],
        dtype=np.float32,
    )


def build_demo_features(dataset_path: Path) -> tuple[np.ndarray, np.ndarray]:
    from data.schema import iter_dataset

    rng = np.random.default_rng(42)
    X_rows, y_rows, post_ids = [], [], []
    for rec in iter_dataset(dataset_path):
        X_rows.append(_feature_template(rec.label, rng))
        y_rows.append(rec.label)
        post_ids.append(rec.post_id)
    return np.vstack(X_rows), np.asarray(y_rows, dtype=object), post_ids


def build_demo_baseline_features(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic 2-D ParsBERT polarity vectors for baseline eval without torch."""
    rng = np.random.default_rng(99)
    X_rows = []
    for label in y:
        if label in ("positive", "positive_sarcasm"):
            p = np.array([0.15, 0.85], dtype=np.float32) + rng.normal(0, 0.05, 2)
        elif label in ("negative", "negative_sarcasm"):
            p = np.array([0.85, 0.15], dtype=np.float32) + rng.normal(0, 0.05, 2)
        else:
            p = np.array([0.5, 0.5], dtype=np.float32) + rng.normal(0, 0.05, 2)
        p = np.clip(p, 0.01, 0.99)
        p /= p.sum()
        X_rows.append(p)
    return np.vstack(X_rows), y


def run_eval_tooling(X: np.ndarray, y: np.ndarray) -> None:
    import subprocess

    env = {**dict(__import__("os").environ), "PYTHONPATH": str(SRC)}
    baseline_cache = ROOT / "artifacts" / "baseline_features.npz"
    Xb, yb = build_demo_baseline_features(y)
    np.savez_compressed(baseline_cache, X=Xb, y=yb)

    steps = [
        [sys.executable, "-m", "eval.metrics"],
        [sys.executable, "-m", "eval.baseline", "--dataset", str(DEMO_JSONL)],
        [sys.executable, "-m", "eval.ablation"],
        [sys.executable, "-m", "eval.report"],
    ]
    for cmd in steps:
        print(f"\n$ {' '.join(cmd)}")
        rc = subprocess.call(cmd, cwd=str(ROOT), env=env)
        if rc != 0:
            print(f"warning: command exited {rc}")


def try_real_inference() -> None:
    """One sample through frozen backbones (downloads HuggingFace weights)."""
    from data.schema import iter_dataset
    from inference.pipeline import Pipeline

    rec = next(iter_dataset(DEMO_JSONL))
    image = Image.open(ROOT / rec.image_path).convert("RGB")
    print("\nLoading SmolVLM + M-CLIP + ParsBERT (first run downloads weights)...")
    pipeline = Pipeline.from_pretrained()
    pred = pipeline.predict(rec.caption, image)
    print("Real inference sample:")
    print(json.dumps(pred.as_dict(), indent=2, ensure_ascii=False))


def print_proposal_checklist(train_result) -> None:
    print("\n" + "=" * 70)
    print("PROPOSAL FIT CHECKLIST (FA-GDCNet thesis outline)")
    print("=" * 70)
    items = [
        ("Lightweight / training-free backbones", "Implemented in inference/models.py (frozen loaders)"),
        ("5-class output (incl. sarcasm variants)", f"Labels: {list(LABELS)}"),
        ("GDRM: Dsem, Dsen, Fvt", "inference/gdrm.py — verified by unit tests"),
        ("Sklearn head on discrepancy vector", f"Trained: {train_result.classifier_name}, Macro-F1={train_result.mean_macro_f1:.3f}"),
        ("VRAM < 1 GiB goal", "Run: python -m eval.profile after real inference"),
        ("RTL explainability", "explain/rtl.py + render_text.py — unit tested"),
        ("Instagram scrape + label tools", "data/scrape.py, data/label.py — ready; needs your credentials/data"),
        ("Eval: metrics, ablation, baseline, report", "See reports/ after this demo"),
        ("Manual 300–1000 labeled posts", "NOT DONE — demo uses synthetic data only"),
    ]
    for title, status in items:
        print(f"  [{'x' if 'NOT DONE' not in status else ' '}] {title}")
        print(f"      -> {status}")
    print("=" * 70)


def main() -> int:
    parser = argparse.ArgumentParser(description="FA-GDCNet proposal alignment demo")
    parser.add_argument(
        "--try-models",
        action="store_true",
        help="Run one real inference (downloads torch/transformers weights; slow)",
    )
    parser.add_argument("--skip-eval", action="store_true", help="Skip eval/*.py subprocesses")
    args = parser.parse_args()

    print("1) Bootstrap demo dataset (Persian captions + images)...")
    bootstrap_demo_dataset()
    print(f"   wrote {DEMO_JSONL} ({len(LABELS) * 12} samples)")

    print("2) Build synthetic GDRM feature cache (simulates backbone outputs)...")
    X, y, post_ids = build_demo_features(DEMO_JSONL)
    DEFAULT_FEATURES.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DEFAULT_FEATURES,
        X=X,
        y=y,
        post_ids=np.asarray(post_ids, dtype=object),
        feature_names=np.asarray(FEATURE_NAMES, dtype=object),
    )
    print(f"   wrote {DEFAULT_FEATURES}")

    print("3) Train sklearn classifier (5-fold CV)...")
    result = train(X, y, save_to=DEFAULT_CLF)
    print(
        f"   saved {result.saved_to} | {result.classifier_name} | "
        f"macro-F1={result.mean_macro_f1:.3f} ± {result.std_macro_f1:.3f}"
    )

    print("4) Demo predict() via saved classifier...")
    from inference.pipeline import Pipeline, Prediction
    from inference.classifier import load, predict_proba
    from inference.gdrm import build_feature_vector, DiscrepancyFeatures

    clf_pack = load(DEFAULT_CLF)

    class _MiniPipe:
        def __init__(self):
            self.clf_pack = clf_pack
            self.fvt_threshold = 0.2

        def predict_from_features(self, feats: DiscrepancyFeatures) -> Prediction:
            proba = predict_proba(self.clf_pack, feats)
            idx = int(np.argmax(proba))
            return Prediction(
                label=LABELS[idx],
                confidence=float(proba[idx]),
                discrepancy_vector=feats.as_dict(),
                low_fidelity=feats.Fvt < self.fvt_threshold,
            )

    pipe = _MiniPipe()
    sarcasm_feats = _feature_template("positive_sarcasm", np.random.default_rng(0))
    feat_dict = dict(zip(FEATURE_NAMES, map(float, sarcasm_feats), strict=True))
    pred = pipe.predict_from_features(DiscrepancyFeatures(**feat_dict))
    print("   sarcasm-shaped vector ->", json.dumps(pred.as_dict(), ensure_ascii=False))

    if not args.skip_eval:
        print("5) Run evaluation scripts (metrics, baseline, ablation, report)...")
        run_eval_tooling(X, y)

    if args.try_models:
        try_real_inference()

    print_proposal_checklist(result)
    print("\nNext steps for a real thesis run:")
    print("  python tasks.py scrape --hashtag <tag> --max-count 500")
    print("  python tasks.py label")
    print("  python tasks.py train    # uses real backbones + features")
    print("  python tasks.py eval")
    print("  python tasks.py dashboard")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
