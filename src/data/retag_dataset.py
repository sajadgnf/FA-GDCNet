"""Retag the canonical dataset with proposal-aligned 5-class labels.

Uses frozen polarity vectors, SmolVLM descriptions, and CLIP visual affect.
Writes the jsonl in place after a ``.bak``.

    python -m data.retag_dataset --dry-run
    python -m data.retag_dataset
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from data.proposal_label import assign_proposal_label

DEFAULT_DATASET = Path("datasets") / "persian_multimodal_irony.jsonl"
CAPTIONS = Path("artifacts") / "stages" / "captions.jsonl"
POLARITY = Path("artifacts") / "stages" / "polarity.jsonl"
FEATURES = Path("artifacts") / "features.npz"
AFFECT = Path("artifacts") / "stages" / "image_affect.jsonl"


def _index_jsonl(path: Path, key: str = "post_id") -> dict:
    out: dict = {}
    if not path.is_file():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[str(row[key])] = row
    return out


def _load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def retag_rows(
    rows: list[dict],
    *,
    captions: dict,
    polarity: dict,
    affect: dict,
) -> tuple[list[dict], int, int]:
    changed = 0
    missing_pol = 0
    for row in rows:
        pid = str(row["post_id"])
        pol = polarity.get(pid)
        if pol is None:
            missing_pol += 1
        vis = affect.get(pid) or {}
        new = assign_proposal_label(
            str(row.get("caption") or ""),
            pol_T=None if pol is None else pol.get("pol_T"),
            pol_T_hat=None if pol is None else pol.get("pol_T_hat"),
            generated=str((captions.get(pid) or {}).get("generated") or ""),
            visual_hat=vis.get("hat"),
            visual_pos=vis.get("pos"),
            visual_neg=vis.get("neg"),
        )
        if new != row.get("label"):
            changed += 1
            row["label"] = new
            anns = list(row.get("annotators") or [])
            if "proposal-retag" not in anns:
                anns.append("proposal-retag")
            row["annotators"] = anns
    return rows, changed, missing_pol


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Do not run CLIP; use VLM wording + polarity only.",
    )
    parser.add_argument("--examples", type=int, default=6)
    args = parser.parse_args(argv)

    captions = _index_jsonl(CAPTIONS)
    polarity = _index_jsonl(POLARITY)
    rows = _load_rows(args.dataset)
    old = Counter(r.get("label") for r in rows)

    affect: dict = {}
    if not args.skip_images:
        from data.image_affect import score_images

        affect = score_images(rows, cache=AFFECT)

    # Copy so dry-run does not mutate until we decide to write.
    tagged = [dict(r) for r in rows]
    tagged, changed, missing_pol = retag_rows(
        tagged, captions=captions, polarity=polarity, affect=affect
    )
    new_c = Counter(r.get("label") for r in tagged)
    print(f"rows={len(rows)} changed={changed} missing_polarity={missing_pol}")
    print("before", dict(old))
    print("after ", dict(new_c))
    print("visual hats", dict(Counter((affect.get(str(r["post_id"])) or {}).get("hat") for r in rows)))

    # Show a few flips per target label so the run is auditable.
    flips: dict[str, list] = {}
    for old_row, new_row in zip(rows, tagged):
        if old_row.get("label") == new_row.get("label"):
            continue
        key = f"{old_row.get('label')} -> {new_row.get('label')}"
        flips.setdefault(key, []).append(new_row)
    print("flip kinds", {k: len(v) for k, v in sorted(flips.items(), key=lambda kv: -len(kv[1]))})
    n_ex = max(0, int(args.examples))
    for kind, items in sorted(flips.items(), key=lambda kv: -len(kv[1]))[:12]:
        print(f"\n# {kind} ({len(items)})")
        for row in items[:n_ex]:
            cap = (row.get("caption") or "").replace("\n", " ")[:110]
            vis = (affect.get(str(row["post_id"])) or {}).get("hat")
            print(f"  {row['post_id']} hat={vis} {cap}")

    if args.dry_run:
        return 0

    bak = args.dataset.with_suffix(".jsonl.bak")
    shutil.copy2(args.dataset, bak)
    print(f"backup {bak}")
    with args.dataset.open("w", encoding="utf-8") as fh:
        for row in tagged:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    if FEATURES.is_file():
        import numpy as np

        z = dict(np.load(FEATURES, allow_pickle=True))
        by_id = {str(r["post_id"]): r["label"] for r in tagged}
        if "post_ids" in z:
            z["y"] = np.array(
                [by_id.get(str(pid), y) for pid, y in zip(z["post_ids"], z["y"])],
                dtype=object,
            )
            np.savez_compressed(FEATURES, **z)
            print(f"updated y in {FEATURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
