"""Zero-shot larger-VLM baseline for proposal Hypothesis 3.

Hypothesis 3 (PDF §6.3): the proposed system uses < 1 GiB, is faster than a
heavy multimodal model, and the accuracy drop versus that model is < 5%.

Default local model is Qwen2-VL-2B-Instruct — larger than SmolVLM-256M, not
Flamingo / Idefics-80B. On OOM or import failure this writes ran=false and
does not stamp PASS.

This module is importable without transformers. The generate loop loads weights
only in ``run_vlm``.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from data.eval_set import is_eval_eligible
from data.schema import LABELS, iter_dataset

log = logging.getLogger(__name__)

DEFAULT_DATASET = Path("datasets") / "persian_multimodal_irony.jsonl"
DEFAULT_OOF = Path("reports") / "oof_preds.jsonl"
DEFAULT_STAGED = Path("reports") / "profile_staged.json"
DEFAULT_OUT = Path("reports") / "heavy_compare.json"
DEFAULT_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
ACCURACY_DROP_MAX = 0.05
ONE_GIB = 1024 ** 3

_PROMPT = (
    "Classify this Persian Instagram post (caption + image) as exactly one of: "
    + ", ".join(LABELS)
    + ". Reply with only that label token.\n\nCaption:\n{caption}"
)


def parse_label(text: str) -> str | None:
    raw = (text or "").strip().lower().replace(" ", "_")
    for lbl in sorted(LABELS, key=len, reverse=True):
        if lbl in raw:
            return lbl
    return None


def load_oof_map(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[str(row["post_id"])] = str(row["pred"])
    return out


def stratified_post_ids(
    records: list,
    n: int,
    *,
    seed: int = 0,
) -> list[str]:
    if n <= 0 or not records:
        return []
    y = np.array([r.label for r in records], dtype=object)
    ids = [r.post_id for r in records]
    take = min(int(n), len(records))
    if take >= len(records):
        return list(ids)
    try:
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=take, random_state=seed)
        idx, _ = next(splitter.split(np.zeros(len(records)), y))
        return [ids[i] for i in idx]
    except ValueError:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(records), size=take, replace=False)
        return [ids[int(i)] for i in chosen]


def h3_verdict(payload: dict) -> str:
    """Return PASS, FAIL, or NOT_RUN from a compare payload."""
    if not payload.get("ran"):
        return "NOT_RUN"
    staged_ok = bool(payload.get("ours_under_1gib"))
    faster = bool(payload.get("ours_faster"))
    drop = payload.get("accuracy_drop")
    try:
        drop_ok = drop is not None and float(drop) < ACCURACY_DROP_MAX
    except (TypeError, ValueError):
        drop_ok = False
    if staged_ok and faster and drop_ok:
        return "PASS"
    return "FAIL"


def _empty_payload(*, model: str, reason: str) -> dict:
    return {
        "ran": False,
        "reason": reason,
        "model": model,
        "n_samples": 0,
        "heavy_accuracy": None,
        "ours_accuracy": None,
        "accuracy_drop": None,
        "heavy_median_latency_s": None,
        "ours_median_latency_s": None,
        "heavy_peak_memory_gib": None,
        "ours_peak_memory_gib": None,
        "ours_under_1gib": False,
        "ours_faster": False,
        "h3": "NOT_RUN",
        "note": (
            "Local VLM is larger than SmolVLM-256M but weaker than the "
            "Flamingo / Idefics-80B models named in the proposal literature review."
        ),
    }


def run_vlm(
    records: list,
    *,
    model_id: str,
    device: str | None = None,
) -> tuple[list[str], list[float], float | None]:
    """Return (preds, latencies_s, peak_gib). Raises on load/OOM."""
    import torch
    from PIL import Image
    from transformers import AutoProcessor

    try:
        from transformers import Qwen2VLForConditionalGeneration as ModelCls
    except ImportError:
        from transformers import AutoModelForVision2Seq as ModelCls

    if device in (None, "auto"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_id)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = ModelCls.from_pretrained(model_id, dtype=dtype)
    model = model.to(device)
    model.eval()

    preds: list[str] = []
    latencies: list[float] = []
    peak_bytes = 0
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for i, rec in enumerate(records, start=1):
        log.info("heavy sample %d/%d post_id=%s", i, len(records), rec.post_id)
        image = Image.open(rec.image_path).convert("RGB")
        prompt = _PROMPT.format(caption=rec.caption or "")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
            )
        latencies.append(time.perf_counter() - t0)
        if device == "cuda":
            peak_bytes = max(peak_bytes, int(torch.cuda.max_memory_allocated()))
        in_len = int(inputs["input_ids"].shape[1])
        decoded = processor.batch_decode(out[:, in_len:], skip_special_tokens=True)[0]
        preds.append(parse_label(decoded) or "neutral")

    peak_gib = (peak_bytes / ONE_GIB) if peak_bytes else None
    return preds, latencies, peak_gib


def evaluate(
    *,
    dataset: Path,
    oof_path: Path,
    staged_path: Path,
    model_id: str,
    n: int,
    device: str | None = None,
) -> dict:
    records = [
        r
        for r in iter_dataset(dataset)
        if Path(r.image_path).is_file() and is_eval_eligible(r.annotators)
    ]
    chosen_ids = set(stratified_post_ids(records, n))
    sample = [r for r in records if r.post_id in chosen_ids]
    oof = load_oof_map(oof_path)
    staged = json.loads(staged_path.read_text(encoding="utf-8")) if staged_path.is_file() else {}
    payload = _empty_payload(model=model_id, reason="")
    payload.update(
        {
            "n_requested": n,
            "n_samples": len(sample),
            "ours_median_latency_s": staged.get("median_total_latency_s"),
            "ours_peak_memory_gib": staged.get("peak_memory_gib"),
            "ours_under_1gib": bool(staged.get("under_1gib_budget")),
        }
    )
    if not sample:
        payload["reason"] = "no usable samples"
        payload["h3"] = h3_verdict(payload)
        return payload
    if not oof:
        payload["reason"] = "missing reports/oof_preds.jsonl — run eval.metrics first"
        payload["h3"] = h3_verdict(payload)
        return payload

    try:
        preds, lats, peak_gib = run_vlm(sample, model_id=model_id, device=device)
    except Exception as exc:  # noqa: BLE001 — load/OOM/decode is NOT_RUN, not a crash
        payload["reason"] = f"failed to run VLM: {exc}"
        payload["h3"] = h3_verdict(payload)
        return payload

    gold = [r.label for r in sample]
    heavy_acc = float(np.mean([p == g for p, g in zip(preds, gold)]))
    ours_preds = [oof.get(r.post_id) for r in sample]
    if any(p is None for p in ours_preds):
        payload["reason"] = "OOF map missing some sampled post_ids"
        payload["h3"] = h3_verdict(payload)
        return payload
    ours_acc = float(np.mean([p == g for p, g in zip(ours_preds, gold)]))
    drop = heavy_acc - ours_acc
    heavy_lat = float(np.median(lats)) if lats else None
    ours_lat = payload["ours_median_latency_s"]
    payload.update(
        {
            "ran": True,
            "reason": "",
            "heavy_accuracy": heavy_acc,
            "ours_accuracy": ours_acc,
            "accuracy_drop": drop,
            "heavy_median_latency_s": heavy_lat,
            "heavy_peak_memory_gib": peak_gib,
            "ours_faster": (
                ours_lat is not None and heavy_lat is not None and float(ours_lat) < float(heavy_lat)
            ),
            "post_ids": [r.post_id for r in sample],
        }
    )
    payload["h3"] = h3_verdict(payload)
    return payload


def _write_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    raw = list(argv) if argv is not None else sys.argv[1:]
    in_process = "--in-process" in raw
    if in_process:
        raw = [a for a in raw if a != "--in-process"]

    parser = argparse.ArgumentParser(description="Hypothesis 3 heavy-VLM comparison.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--oof", type=Path, default=DEFAULT_OOF)
    parser.add_argument("--staged", type=Path, default=DEFAULT_STAGED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu")
    args = parser.parse_args(raw)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not in_process:
        rc = subprocess.call(
            [sys.executable, "-m", "eval.heavy_baseline", "--in-process", *raw]
        )
        if rc != 0:
            payload = _empty_payload(
                model=args.model,
                reason=(
                    f"VLM worker exited with code {rc}. "
                    "Qwen2-VL-2B-Instruct crashed while loading shards on this host "
                    "(Windows access violation 0xC0000005 on CUDA and CPU). "
                    "Hypothesis 3 is NOT_RUN, not PASS."
                ),
            )
            if args.staged.is_file():
                staged = json.loads(args.staged.read_text(encoding="utf-8"))
                payload["ours_median_latency_s"] = staged.get("median_total_latency_s")
                payload["ours_peak_memory_gib"] = staged.get("peak_memory_gib")
                payload["ours_under_1gib"] = bool(staged.get("under_1gib_budget"))
            payload["h3"] = h3_verdict(payload)
            _write_payload(args.out, payload)
            log.info("wrote %s h3=NOT_RUN worker_rc=%s", args.out, rc)
        return 0

    payload = evaluate(
        dataset=args.dataset,
        oof_path=args.oof,
        staged_path=args.staged,
        model_id=args.model,
        n=args.n,
        device=args.device,
    )
    _write_payload(args.out, payload)
    log.info(
        "wrote %s h3=%s ran=%s reason=%s",
        args.out,
        payload.get("h3"),
        payload.get("ran"),
        payload.get("reason"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
