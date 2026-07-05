"""Import locally saved Instagram posts into a raw labeling pool.

Use this when instaloader cannot reach Instagram's API but you can save images
from the browser (right-click → Save image, or a download helper).

Expected layout (pick one):

1) Folder of images named ``<post_id>.jpg`` with optional sidecar captions::

       datasets/raw/inbox/DZabc123.jpg
       datasets/raw/inbox/DZabc123.txt   # caption (UTF-8, one file)

2) Manifest JSONL (one object per line)::

       {"post_id": "DZabc123", "caption": "...", "image_path": "inbox/DZabc123.jpg"}

Rows are appended to ``datasets/raw/<pool>.jsonl`` with the same dedup / face /
ignore rules as ``data.scrape``.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

from .preprocess import is_persian_enough, is_spam_caption, preprocess_caption
from .scrape import (
    DEFAULT_RAW_DIR,
    IGNORED_IDS_FILE,
    _existing_shortcodes,
    _remember_ignored,
)

log = logging.getLogger(__name__)

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _load_caption_sidecar(image_path: Path) -> str:
    for ext in (".txt", ".caption"):
        sidecar = image_path.with_suffix(ext)
        if sidecar.is_file():
            return sidecar.read_text(encoding="utf-8").strip()
    return ""


def _load_manifest(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _iter_local_candidates(
    input_dir: Path,
    *,
    manifest: Path | None = None,
) -> list[tuple[str, str, Path]]:
    """Return (post_id, raw_caption, source_image_path) tuples."""
    out: list[tuple[str, str, Path]] = []
    if manifest is not None:
        for row in _load_manifest(manifest):
            post_id = str(row["post_id"]).strip()
            caption = str(row.get("caption") or "")
            image_path = Path(row["image_path"])
            if not image_path.is_absolute():
                image_path = (manifest.parent / image_path).resolve()
            out.append((post_id, caption, image_path))
        return out

    for path in sorted(input_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        post_id = path.stem
        caption = _load_caption_sidecar(path)
        out.append((post_id, caption, path))
    return out


def import_local_pool(
    input_dir: Path,
    *,
    pool_name: str = "manual",
    out_dir: Path = DEFAULT_RAW_DIR,
    require_face: bool = False,
    min_face_size: int = 40,
    manifest: Path | None = None,
) -> int:
    if manifest is None and not input_dir.is_dir():
        raise SystemExit(f"input directory not found: {input_dir}")

    if require_face:
        try:
            from .face_filter import has_face as _image_has_face
        except ImportError as exc:
            raise SystemExit(
                "Face filter requires opencv. Install with: pip install opencv-python-headless"
            ) from exc
    else:
        _image_has_face = None  # type: ignore[assignment,misc]

    jsonl_path = out_dir / f"{pool_name}.jsonl"
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    seen = _existing_shortcodes(jsonl_path)

    written = 0
    skipped_no_face = 0
    skipped_spam = 0
    with jsonl_path.open("a", encoding="utf-8") as out:
        for post_id, raw_caption, src in _iter_local_candidates(
            input_dir, manifest=manifest
        ):
            if post_id in seen:
                log.debug("skip duplicate %s", post_id)
                continue
            if not src.is_file():
                log.warning("skip %s: image not found at %s", post_id, src)
                continue

            caption = preprocess_caption(raw_caption)
            if caption and not is_persian_enough(caption):
                log.warning("skip %s: caption not Persian enough", post_id)
                continue
            if caption and is_spam_caption(caption):
                skipped_spam += 1
                _remember_ignored(post_id)
                seen.add(post_id)
                continue

            dest = image_dir / f"{post_id}{src.suffix.lower()}"
            if src.resolve() != dest.resolve():
                shutil.copy2(src, dest)

            if require_face and _image_has_face is not None:
                if not _image_has_face(dest, min_size=min_face_size):
                    skipped_no_face += 1
                    _remember_ignored(post_id)
                    seen.add(post_id)
                    dest.unlink(missing_ok=True)
                    log.debug("skip %s: no face detected", post_id)
                    continue

            row = {
                "post_id": post_id,
                "caption": caption,
                "image_path": str(dest),
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            seen.add(post_id)
            written += 1
            log.info("imported %s", post_id)

    if skipped_no_face:
        log.info(
            "skipped %d images with no face (IDs saved to %s)",
            skipped_no_face,
            IGNORED_IDS_FILE,
        )
    if skipped_spam:
        log.info("skipped %d spam captions (IDs saved to %s)", skipped_spam, IGNORED_IDS_FILE)
    log.info("imported %d posts into %s", written, jsonl_path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import local images into a raw labeling pool.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("datasets/raw/inbox"),
        help="Folder with <post_id>.jpg (+ optional <post_id>.txt caption).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional JSONL manifest with post_id, caption, image_path.",
    )
    parser.add_argument("--pool-name", default="manual")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument(
        "--require-face",
        action="store_true",
        help="Keep only images where OpenCV detects a frontal face.",
    )
    parser.add_argument("--min-face-size", type=int, default=40)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    import_local_pool(
        args.input_dir,
        pool_name=args.pool_name,
        out_dir=args.out_dir,
        require_face=args.require_face,
        min_face_size=args.min_face_size,
        manifest=args.manifest,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
