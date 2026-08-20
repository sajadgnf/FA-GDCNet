"""Per-stage VRAM + latency profiler (proposal Hypothesis 1: peak ≤ 1 GiB).

Measures each extraction stage independently so peak memory reflects the
largest single backbone, not the sum of all three loaded at once.

M-CLIP is profiled as two sub-stages (text tower, then image tower) to match
the production extraction path in `inference.stages`.

Writes `reports/profile_staged.json` with per-stage peaks and a
`under_1gib_budget` flag based on max(stage peaks).
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from pathlib import Path

from data.schema import iter_dataset

log = logging.getLogger(__name__)

DEFAULT_DATASET = Path("datasets") / "persian_multimodal_irony.jsonl"
DEFAULT_OUT = Path("reports") / "profile_staged.json"
ONE_GIB = 1024**3


def _samples(dataset: Path, n: int) -> list[tuple[str, str]]:
    records = [r for r in iter_dataset(dataset) if Path(r.image_path).is_file()][:n]
    return [(r.caption, r.image_path) for r in records]


def _profile_caption(samples: list[tuple[str, str]]) -> dict:
    import torch
    from PIL import Image

    from inference import models

    bundle = models.load_captioner_only()
    torch.cuda.reset_peak_memory_stats()
    timings: list[float] = []
    try:
        for _caption, image_path in samples:
            with Image.open(image_path) as im:
                image = im.convert("RGB")
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            models.caption_image(bundle, image)
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - t0)
    finally:
        models.release(bundle)
    peak = int(torch.cuda.max_memory_allocated())
    return _stage_result("captions", timings, peak)


def _profile_mclip_text(samples: list[tuple[str, str]]) -> dict:
    import torch

    from inference import models

    # Text tower stays on CPU: fp16 XLM-R Large weights alone exceed 1 GiB VRAM.
    models._cuda_gc()
    bundle = models.load_mclip_text_only(device="cpu")
    torch.cuda.reset_peak_memory_stats()
    timings: list[float] = []
    try:
        for caption, _image_path in samples:
            t0 = time.perf_counter()
            _ = models.embed_text_mclip(bundle, caption)
            _ = models.embed_text_mclip(bundle, "a photo description")
            timings.append(time.perf_counter() - t0)
    finally:
        models.release(bundle)
    peak = int(torch.cuda.max_memory_allocated())
    result = _stage_result("mclip_text", timings, peak)
    result["device"] = "cpu"
    result["note"] = (
        "XLM-Roberta-Large text tower runs on CPU; weights alone are ~1.04 GiB in fp16"
    )
    return result


def _profile_mclip_image(samples: list[tuple[str, str]]) -> dict:
    import torch
    from PIL import Image

    from inference import models

    bundle = models.load_mclip_image_only()
    torch.cuda.reset_peak_memory_stats()
    timings: list[float] = []
    try:
        for _caption, image_path in samples:
            with Image.open(image_path) as im:
                image = im.convert("RGB")
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = models.embed_image_mclip(bundle, image)
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - t0)
    finally:
        models.release(bundle)
    peak = int(torch.cuda.max_memory_allocated())
    return _stage_result("mclip_image", timings, peak)


def _profile_polarity(samples: list[tuple[str, str]]) -> dict:
    import torch

    from inference import models

    bundle = models.load_polarity_only()
    torch.cuda.reset_peak_memory_stats()
    timings: list[float] = []
    try:
        for caption, _image_path in samples:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            models.polarity_probs(bundle, caption)
            models.polarity_probs(bundle, "generated caption text")
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - t0)
    finally:
        models.release(bundle)
    peak = int(torch.cuda.max_memory_allocated())
    return _stage_result("polarity", timings, peak)


def _stage_result(name: str, timings: list[float], peak_bytes: int) -> dict:
    med = statistics.median(timings) if timings else 0.0
    p95 = statistics.quantiles(timings, n=20)[18] if len(timings) >= 20 else max(timings, default=0.0)
    return {
        "stage": name,
        "n_samples": len(timings),
        "median_latency_s": med,
        "p95_latency_s": p95,
        "peak_memory_bytes": peak_bytes,
        "peak_memory_gib": peak_bytes / ONE_GIB,
        "under_1gib_budget": peak_bytes <= ONE_GIB,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile each extraction stage separately.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n", type=int, default=20, help="Samples per stage (keep small on 6GB GPUs).")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    import torch

    if not torch.cuda.is_available():
        log.error("CUDA required for staged VRAM profiling")
        return 1

    samples = _samples(args.dataset, args.n)
    if not samples:
        log.error("no usable samples in %s", args.dataset)
        return 1

    stages = [
        _profile_caption(samples),
        _profile_mclip_text(samples),
        _profile_mclip_image(samples),
        _profile_polarity(samples),
    ]
    peak_bytes = max(s["peak_memory_bytes"] for s in stages)
    total_latency = sum(s["median_latency_s"] for s in stages)
    result = {
        "backend": "cuda",
        "n_samples_per_stage": len(samples),
        "stages": stages,
        "peak_memory_bytes": peak_bytes,
        "peak_memory_gib": peak_bytes / ONE_GIB,
        "under_1gib_budget": peak_bytes <= ONE_GIB,
        "median_total_latency_s": total_latency,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log.info(
        "wrote %s (peak=%.2f GiB, under_1gib=%s)",
        args.out,
        result["peak_memory_gib"],
        result["under_1gib_budget"],
    )
    for s in stages:
        log.info(
            "  %s: %.3f GiB (ok=%s)",
            s["stage"],
            s["peak_memory_gib"],
            s["under_1gib_budget"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
