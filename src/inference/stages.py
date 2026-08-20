"""Staged feature extraction: one backbone resident at a time.

Holding SmolVLM, mCLIP, and ParsBERT on the GPU simultaneously needs roughly
6 GB, which on a 6 GB card forces the driver to spill to host memory and
collapses throughput (observed: GPU utilisation ~40% and >20x slowdown).

Running one stage at a time makes peak memory track the largest single model
instead of their sum. Each stage appends to a JSONL checkpoint keyed by
`post_id`, so an interrupted run resumes instead of starting over.

Stages:
  1. captions   SmolVLM        -> generated description per image
  2. mclip      M-CLIP text then image towers (one at a time) -> Dsem, Fvt, cos_TI
  3. polarity   ParsBERT       -> polarity distributions for T and T_hat
  4. assemble   (no GPU)       -> artifacts/features.npz

Usage:
    python -m inference.stages                 # run all stages, resuming
    python -m inference.stages --stage captions
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np

from data.schema import DatasetRecord, iter_dataset

from .gdrm import FEATURE_NAMES, DiscrepancyFeatures, polarity_scalar

log = logging.getLogger(__name__)

DEFAULT_DATASET = Path("datasets") / "persian_multimodal_irony.jsonl"
STAGE_DIR = Path("artifacts") / "stages"
CAPTIONS_JSONL = STAGE_DIR / "captions.jsonl"
MCLIP_TEXT_JSONL = STAGE_DIR / "mclip_text.jsonl"
MCLIP_JSONL = STAGE_DIR / "mclip.jsonl"
POLARITY_JSONL = STAGE_DIR / "polarity.jsonl"
DEFAULT_FEATURES = Path("artifacts") / "features.npz"

PROGRESS_EVERY = 25


# --------------------------- checkpoint helpers ------------------------------


def _read_done(path: Path) -> dict[str, dict]:
    """Load a stage checkpoint as `{post_id: row}`; tolerate a truncated tail."""
    if not path.is_file():
        return {}
    done: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                log.warning("ignoring malformed checkpoint line in %s", path)
                continue
            pid = row.get("post_id")
            if pid:
                done[str(pid)] = row
    return done


class _Appender:
    """JSONL writer that flushes every record so progress survives a kill."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = path.open("a", encoding="utf-8")

    def write(self, row: dict) -> None:
        self._f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> "_Appender":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _usable_records(dataset: Path) -> list[DatasetRecord]:
    records: list[DatasetRecord] = []
    skipped = 0
    for rec in iter_dataset(dataset):
        if Path(rec.image_path).is_file():
            records.append(rec)
        else:
            skipped += 1
            log.warning("skipping %s: image not found at %s", rec.post_id, rec.image_path)
    if skipped:
        log.warning("skipped %d records with missing images", skipped)
    return records


def _pending(
    records: list[DatasetRecord], checkpoint: Path
) -> tuple[list[DatasetRecord], dict[str, dict]]:
    done = _read_done(checkpoint)
    todo = [r for r in records if r.post_id not in done]
    log.info(
        "%s: %d/%d already done, %d to process",
        checkpoint.name,
        len(done),
        len(records),
        len(todo),
    )
    return todo, done


def _run_stage(
    *,
    name: str,
    records: list[DatasetRecord],
    checkpoint: Path,
    load: Callable[[], Any],
    process: Callable[[Any, DatasetRecord], dict],
) -> None:
    """Load one backbone, process every pending record, then release the GPU."""
    from . import models

    todo, _ = _pending(records, checkpoint)
    if not todo:
        log.info("stage %s: nothing to do", name)
        return

    models._cuda_gc()
    bundle = load()
    try:
        with _Appender(checkpoint) as out:
            for i, rec in enumerate(todo, start=1):
                row = process(bundle, rec)
                row["post_id"] = rec.post_id
                out.write(row)
                if i % PROGRESS_EVERY == 0 or i == len(todo):
                    log.info("stage %s: %d/%d", name, i, len(todo))
    finally:
        models.release(bundle)
    log.info("stage %s complete -> %s", name, checkpoint)


# ------------------------------- stage 1 -------------------------------------


def stage_captions(records: list[DatasetRecord], *, checkpoint: Path = CAPTIONS_JSONL) -> None:
    from PIL import Image

    from . import models

    def process(bundle: Any, rec: DatasetRecord) -> dict:
        with Image.open(rec.image_path) as im:
            image = im.convert("RGB")
        return {"generated": models.caption_image(bundle, image)}

    _run_stage(
        name="captions",
        records=records,
        checkpoint=checkpoint,
        load=models.load_captioner_only,
        process=process,
    )


# ------------------------------- stage 2 -------------------------------------


def stage_mclip_text(
    records: list[DatasetRecord],
    *,
    checkpoint: Path = MCLIP_TEXT_JSONL,
    captions: Path = CAPTIONS_JSONL,
) -> None:
    """Embed captions with the M-CLIP text tower only (keeps VRAM ≤ one tower)."""
    from . import models

    generated = _require_captions(records, captions)

    def process(bundle: Any, rec: DatasetRecord) -> dict:
        emb_T = models.embed_text_mclip(bundle, rec.caption)
        emb_T_hat = models.embed_text_mclip(bundle, generated[rec.post_id])
        return {
            "emb_T": [float(x) for x in emb_T],
            "emb_T_hat": [float(x) for x in emb_T_hat],
        }

    _run_stage(
        name="mclip_text",
        records=records,
        checkpoint=checkpoint,
        # CPU: XLM-R Large fp16 weights alone are ~1.04 GiB (over the VRAM budget).
        load=lambda: models.load_mclip_text_only(device="cpu"),
        process=process,
    )


def stage_mclip_image(
    records: list[DatasetRecord],
    *,
    checkpoint: Path = MCLIP_JSONL,
    text_checkpoint: Path = MCLIP_TEXT_JSONL,
) -> None:
    """Embed images, then fold with cached text embeddings into Dsem/Fvt/cos_TI."""
    from PIL import Image

    from . import models
    from .gdrm import compute_dsem, compute_fvt, cosine_similarity

    text_rows = _read_done(text_checkpoint)
    missing = [r.post_id for r in records if r.post_id not in text_rows]
    if missing:
        raise RuntimeError(
            f"mclip_text stage incomplete: {len(missing)} record(s) missing from "
            f"{text_checkpoint} (e.g. {missing[:3]}). Run mclip text first."
        )

    def process(bundle: Any, rec: DatasetRecord) -> dict:
        row = text_rows[rec.post_id]
        emb_T = np.asarray(row["emb_T"], dtype=np.float32)
        emb_T_hat = np.asarray(row["emb_T_hat"], dtype=np.float32)
        with Image.open(rec.image_path) as im:
            image = im.convert("RGB")
        emb_I = models.embed_image_mclip(bundle, image)
        return {
            "Dsem": compute_dsem(emb_T, emb_T_hat),
            "Fvt": compute_fvt(emb_I, emb_T_hat),
            "cos_TI": cosine_similarity(emb_T, emb_I),
        }

    _run_stage(
        name="mclip_image",
        records=records,
        checkpoint=checkpoint,
        load=models.load_mclip_image_only,
        process=process,
    )


def stage_mclip(
    records: list[DatasetRecord],
    *,
    checkpoint: Path = MCLIP_JSONL,
    captions: Path = CAPTIONS_JSONL,
    text_checkpoint: Path = MCLIP_TEXT_JSONL,
) -> None:
    """Run M-CLIP as two sequential sub-stages so peak VRAM is one tower."""
    stage_mclip_text(records, checkpoint=text_checkpoint, captions=captions)
    stage_mclip_image(records, checkpoint=checkpoint, text_checkpoint=text_checkpoint)


# ------------------------------- stage 3 -------------------------------------


def stage_polarity(
    records: list[DatasetRecord],
    *,
    checkpoint: Path = POLARITY_JSONL,
    captions: Path = CAPTIONS_JSONL,
) -> None:
    from . import models

    generated = _require_captions(records, captions)

    def process(bundle: Any, rec: DatasetRecord) -> dict:
        pol_T = models.polarity_probs(bundle, rec.caption)
        pol_T_hat = models.polarity_probs(bundle, generated[rec.post_id])
        return {
            "pol_T": [float(x) for x in pol_T],
            "pol_T_hat": [float(x) for x in pol_T_hat],
        }

    _run_stage(
        name="polarity",
        records=records,
        checkpoint=checkpoint,
        load=models.load_polarity_only,
        process=process,
    )


# ------------------------------- stage 4 -------------------------------------


def _require_captions(records: list[DatasetRecord], captions: Path) -> dict[str, str]:
    done = _read_done(captions)
    missing = [r.post_id for r in records if r.post_id not in done]
    if missing:
        raise RuntimeError(
            f"caption stage incomplete: {len(missing)} record(s) missing from "
            f"{captions} (e.g. {missing[:3]}). Run the `captions` stage first."
        )
    return {pid: str(row.get("generated", "")) for pid, row in done.items()}


def assemble(
    records: list[DatasetRecord],
    *,
    mclip: Path = MCLIP_JSONL,
    polarity: Path = POLARITY_JSONL,
    out: Path = DEFAULT_FEATURES,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Join the stage checkpoints into the canonical `features.npz` cache."""
    m = _read_done(mclip)
    p = _read_done(polarity)

    X_rows: list[np.ndarray] = []
    y_rows: list[str] = []
    post_ids: list[str] = []
    for rec in records:
        mr, pr = m.get(rec.post_id), p.get(rec.post_id)
        if mr is None or pr is None:
            continue
        feats = DiscrepancyFeatures(
            Dsem=float(mr["Dsem"]),
            Dsen=float(np.sum(np.abs(np.asarray(pr["pol_T"]) - np.asarray(pr["pol_T_hat"])))),
            Fvt=float(mr["Fvt"]),
            cos_TI=float(mr["cos_TI"]),
            polarity_T=polarity_scalar(pr["pol_T"]),
            polarity_T_hat=polarity_scalar(pr["pol_T_hat"]),
        )
        X_rows.append(feats.as_array())
        y_rows.append(rec.label)
        post_ids.append(rec.post_id)

    if not X_rows:
        raise RuntimeError(
            "no complete records; run the captions, mclip, and polarity stages first"
        )

    X = np.vstack(X_rows)
    y = np.asarray(y_rows, dtype=object)
    constant = [FEATURE_NAMES[i] for i in range(X.shape[1]) if float(np.std(X[:, i])) == 0.0]
    if constant:
        log.error(
            "features %s are constant across all %d samples and contribute nothing; "
            "check that the corresponding encoder loaded its weights",
            ", ".join(constant),
            X.shape[0],
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        X=X,
        y=y,
        post_ids=np.asarray(post_ids, dtype=object),
        feature_names=np.asarray(FEATURE_NAMES, dtype=object),
    )
    log.info("assembled %d x %d features -> %s", X.shape[0], X.shape[1], out)
    return X, y, post_ids


# --------------------------------- CLI ---------------------------------------

STAGE_ORDER = ("captions", "mclip", "polarity", "assemble")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument(
        "--stage",
        choices=STAGE_ORDER,
        action="append",
        help="Run only these stages (repeatable). Default: all, in order.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    records = _usable_records(args.dataset)
    if not records:
        log.error("no usable records in %s", args.dataset)
        return 1

    wanted = tuple(args.stage) if args.stage else STAGE_ORDER
    for stage in STAGE_ORDER:
        if stage not in wanted:
            continue
        if stage == "captions":
            stage_captions(records)
        elif stage == "mclip":
            stage_mclip(records)
        elif stage == "polarity":
            stage_polarity(records)
        elif stage == "assemble":
            assemble(records, out=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
