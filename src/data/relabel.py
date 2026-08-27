"""Review and correct labels on the *existing* canonical dataset.

Unlike ``tasks.py label``, this never rebuilds the corpus from a raw pool. It
loads ``persian_multimodal_irony.jsonl``, shows one post at a time, opens the
image, and writes that row's ``label`` back in place.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .schema import LABELS

DEFAULT_DATASET = Path("datasets") / "persian_multimodal_irony.jsonl"
KEYS = {str(i + 1): label for i, label in enumerate(LABELS)}
SARCASM = frozenset({"positive_sarcasm", "negative_sarcasm"})


def _open_image(path: Path) -> None:
    if not path.is_file():
        print(f"  (image missing: {path})")
        return
    try:
        os.startfile(path)  # Windows
    except AttributeError:
        import subprocess

        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.Popen([opener, str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _load(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _save(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Correct labels on the existing dataset.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--only",
        choices=("all", "sarcasm", "positive", "negative", "neutral"),
        default="sarcasm",
        help="Which rows to review (default: sarcasm — highest impact).",
    )
    parser.add_argument("--start", type=int, default=0, help="Skip the first N matching rows.")
    args = parser.parse_args(argv)

    if not args.dataset.is_file():
        print(f"dataset not found: {args.dataset}")
        return 1

    rows = _load(args.dataset)
    indices = []
    for i, row in enumerate(rows):
        lab = str(row.get("label") or "")
        if args.only == "all":
            indices.append(i)
        elif args.only == "sarcasm" and lab in SARCASM:
            indices.append(i)
        elif lab == args.only:
            indices.append(i)

    todo = indices[max(args.start, 0) :]
    print(f"{len(todo)} posts to review (of {len(indices)} matching, {len(rows)} total).")
    print("Keys:  1 positive  2 negative  3 neutral  4 positive_sarcasm  5 negative_sarcasm")
    print("       Enter = keep current   s = skip   q = quit\n")

    changed = 0
    for n, idx in enumerate(todo, start=1):
        row = rows[idx]
        image = Path(str(row.get("image_path") or ""))
        print("=" * 70)
        print(f"[{n}/{len(todo)}]  post_id={row.get('post_id')}  current={row.get('label')}")
        print(f"image: {image}")
        print(f"caption: {row.get('caption')}")
        _open_image(image)
        try:
            choice = input("> ").strip().lower()
        except EOFError:
            break
        if choice in {"q", "quit"}:
            break
        if choice in {"", "s", "skip"}:
            continue
        if choice not in KEYS:
            print("invalid — use 1-5, Enter, s, or q")
            continue
        new = KEYS[choice]
        if new != row.get("label"):
            row["label"] = new
            annotators = list(row.get("annotators") or [])
            if "relabel" not in annotators:
                annotators.append("relabel")
            row["annotators"] = annotators
            _save(args.dataset, rows)
            changed += 1
            print(f"  -> {new}  (saved)")
        else:
            print("  unchanged")

    print(f"\ndone. corrected {changed} labels in {args.dataset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
