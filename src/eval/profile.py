"""Latency + memory profiler for the inference pipeline.

Records:
- median wall-clock latency over `--n` samples (default 100, spec minimum).
- `torch.cuda.max_memory_allocated()` on CUDA backends.
- peak RSS via `psutil` on CPU backends.

Writes `reports/profile.json` with a boolean `under_1gib_budget` flag per the
spec scenario *VRAM hypothesis check*.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from pathlib import Path

from data.schema import iter_dataset
from inference.pipeline import Pipeline

log = logging.getLogger(__name__)

DEFAULT_DATASET = Path("datasets") / "persian_multimodal_irony.jsonl"
DEFAULT_PROFILE_JSON = Path("reports") / "profile.json"
ONE_GIB = 1024 ** 3


def _is_cuda(pipeline: Pipeline) -> bool:
    dev = getattr(pipeline.bundle, "device", "") or ""
    return str(dev).startswith("cuda")


def _measure_cuda(pipeline: Pipeline, samples: list[tuple[str, str]]) -> dict:
    import torch  # lazy
    from PIL import Image

    torch.cuda.reset_peak_memory_stats()
    timings: list[float] = []
    for caption, image_path in samples:
        image = Image.open(image_path).convert("RGB")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        pipeline.predict(caption, image)
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - t0)
    peak_bytes = int(torch.cuda.max_memory_allocated())
    return {
        "backend": "cuda",
        "n_samples": len(timings),
        "median_latency_s": statistics.median(timings) if timings else 0.0,
        "p95_latency_s": (
            statistics.quantiles(timings, n=20)[18] if len(timings) >= 20 else max(timings, default=0.0)
        ),
        "peak_memory_bytes": peak_bytes,
        "peak_memory_gib": peak_bytes / ONE_GIB,
        "under_1gib_budget": peak_bytes <= ONE_GIB,
    }


def _measure_cpu(pipeline: Pipeline, samples: list[tuple[str, str]]) -> dict:
    import psutil
    from PIL import Image

    proc = psutil.Process()
    timings: list[float] = []
    peak_rss = 0
    for caption, image_path in samples:
        image = Image.open(image_path).convert("RGB")
        t0 = time.perf_counter()
        pipeline.predict(caption, image)
        timings.append(time.perf_counter() - t0)
        peak_rss = max(peak_rss, proc.memory_info().rss)
    return {
        "backend": "cpu",
        "n_samples": len(timings),
        "median_latency_s": statistics.median(timings) if timings else 0.0,
        "p95_latency_s": (
            statistics.quantiles(timings, n=20)[18] if len(timings) >= 20 else max(timings, default=0.0)
        ),
        "peak_memory_bytes": peak_rss,
        "peak_memory_gib": peak_rss / ONE_GIB,
        "under_1gib_budget": peak_rss <= ONE_GIB,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile FA-GDCNet latency + memory.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out", type=Path, default=DEFAULT_PROFILE_JSON)
    parser.add_argument("--n", type=int, default=100, help="Number of samples (spec minimum: 100).")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    records = list(iter_dataset(args.dataset))[: args.n]
    if not records:
        log.error("no dataset records at %s", args.dataset)
        return 1
    samples = [(r.caption, r.image_path) for r in records]
    pipeline = Pipeline.from_pretrained()
    measure = _measure_cuda if _is_cuda(pipeline) else _measure_cpu
    result = measure(pipeline, samples)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log.info("wrote %s (under_1gib_budget=%s)", args.out, result["under_1gib_budget"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
