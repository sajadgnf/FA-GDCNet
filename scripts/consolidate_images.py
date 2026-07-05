"""One-off migration: merge per-pool image folders into a single images/ folder.

Moves every file from datasets/raw/{hashtags,sarcasm,profiles}/ into
datasets/raw/images/, then rewrites the ``image_path`` field in the canonical
dataset and all pool JSONL files to point at the new location.

Idempotent: safe to run more than once. Post IDs are globally unique, so image
filenames never collide across pools.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

RAW = Path("datasets") / "raw"
IMAGES = RAW / "images"
OLD_POOL_DIRS = ["hashtags", "sarcasm", "profiles"]

JSONL_FILES = [
    Path("datasets") / "persian_multimodal_irony.jsonl",
    RAW / "hashtags.jsonl",
    RAW / "sarcasm.jsonl",
    RAW / "profiles.jsonl",
]


def move_images() -> int:
    IMAGES.mkdir(parents=True, exist_ok=True)
    moved = 0
    for name in OLD_POOL_DIRS:
        src_dir = RAW / name
        if not src_dir.is_dir():
            continue
        for f in list(src_dir.iterdir()):
            if not f.is_file():
                continue
            dest = IMAGES / f.name
            if dest.exists():
                f.unlink()  # duplicate (same post id) — drop the copy
            else:
                shutil.move(str(f), str(dest))
                moved += 1
        # Remove the now-empty folder.
        try:
            src_dir.rmdir()
        except OSError:
            pass
    return moved


def _rewrite_path(p: str) -> str:
    if not p:
        return p
    name = Path(p.replace("\\", "/")).name
    return str(RAW / "images" / name)


def rewrite_jsonl(path: Path) -> int:
    if not path.is_file():
        return 0
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    changed = 0
    for row in rows:
        if "image_path" in row:
            new = _rewrite_path(row["image_path"])
            if new != row["image_path"]:
                row["image_path"] = new
                changed += 1
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
    return changed


def main() -> None:
    moved = move_images()
    print(f"moved {moved} image files into {IMAGES}")
    for jf in JSONL_FILES:
        n = rewrite_jsonl(jf)
        print(f"rewrote {n} image_path entries in {jf}")
    missing = 0
    for jf in JSONL_FILES:
        if not jf.is_file():
            continue
        for line in jf.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            ip = row.get("image_path")
            if ip and not Path(ip).is_file():
                missing += 1
    print(f"done. {missing} referenced image files not found on disk (pre-existing broken links)")


if __name__ == "__main__":
    main()
