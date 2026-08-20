"""Fetch the M-CLIP text encoder into a local model directory.

The checkpoint is ~2.1 GB and this network drops long transfers, so each file
is downloaded with HTTP Range resume and retried until complete. Re-running the
script continues where it left off instead of starting over.

Usage:  python scripts/fetch_mclip.py
"""

from __future__ import annotations

import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = "M-CLIP/XLM-Roberta-Large-Vit-B-32"
TARGET_DIR = Path("models") / "M-CLIP--XLM-Roberta-Large-Vit-B-32"
FILES = (
    "config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "pytorch_model.bin",
)
MAX_ATTEMPTS = 200
CHUNK = 1 << 18


def _remote_size(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "fa-gdcnet"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            n = r.headers.get("Content-Length")
            return int(n) if n else None
    except Exception:  # noqa: BLE001
        return None


def fetch(filename: str) -> None:
    url = f"https://huggingface.co/{REPO}/resolve/main/{filename}"
    dest = TARGET_DIR / filename
    part = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)

    total = _remote_size(url)
    if dest.is_file() and total is not None and dest.stat().st_size == total:
        print(f"HAVE {filename} ({total / 1048576:.1f} MB)", flush=True)
        return

    for attempt in range(1, MAX_ATTEMPTS + 1):
        have = part.stat().st_size if part.is_file() else 0
        if total is not None and have >= total:
            break
        headers = {"User-Agent": "fa-gdcnet"}
        if have:
            headers["Range"] = f"bytes={have}-"
        req = urllib.request.Request(url, headers=headers)
        t0 = time.time()
        start_have = have
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                mode = "ab" if have and r.status == 206 else "wb"
                if mode == "wb":
                    have = 0
                with part.open(mode) as f:
                    while True:
                        buf = r.read(CHUNK)
                        if not buf:
                            break
                        f.write(buf)
                        have += len(buf)
        except Exception as exc:  # noqa: BLE001
            rate = (have - start_have) / 1048576 / max(time.time() - t0, 1e-6)
            pct = f"{100 * have / total:.1f}%" if total else "?"
            print(
                f"RETRY {filename} attempt {attempt} at {pct} "
                f"({have / 1048576:.1f} MB, {rate:.2f} MB/s): {type(exc).__name__}",
                flush=True,
            )
            time.sleep(min(3 * attempt, 20))
            continue

        if total is None or have >= total:
            break
        print(
            f"SHORT {filename} attempt {attempt}: {have / 1048576:.1f}/"
            f"{total / 1048576:.1f} MB, resuming",
            flush=True,
        )
        time.sleep(2)
    else:
        raise RuntimeError(f"failed to download {filename} after {MAX_ATTEMPTS} attempts")

    if total is not None and part.stat().st_size != total:
        raise RuntimeError(
            f"{filename}: got {part.stat().st_size} bytes, expected {total}"
        )
    part.replace(dest)
    print(f"OK {filename} ({dest.stat().st_size / 1048576:.1f} MB)", flush=True)


def main() -> int:
    for name in FILES:
        fetch(name)
    print(f"DONE model ready at {TARGET_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
