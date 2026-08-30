"""Blind in-place review of the existing canonical dataset.

Unlike ``tasks.py label``, this never rebuilds the corpus from a raw pool.

The review is **blind**: the current label is not shown, and Enter cannot
confirm a machine tag. A kept CLIP/retag label is not a human label.

Primary annotator writes ``label`` + ``blind-relabel`` on the canonical jsonl.
A second annotator writes an independent file (``--out``) so kappa is defined.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

from .schema import LABELS
from .tags import BLIND_REVIEW_TAG

DEFAULT_DATASET = Path("datasets") / "persian_multimodal_irony.jsonl"
DEFAULT_FEATURES = Path("artifacts") / "features.npz"
DEFAULT_OVERLAP_IDS = Path("datasets") / "iaa_overlap_ids.txt"
DEFAULT_SECOND = Path("datasets") / "iaa_second.jsonl"
KEYS = {str(i + 1): label for i, label in enumerate(LABELS)}
SARCASM = frozenset({"positive_sarcasm", "negative_sarcasm"})
REVIEW_TAG = BLIND_REVIEW_TAG
DEFAULT_ANNOTATOR = "sjjd6502"
DEFAULT_N_NON_SARCASM = 100


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
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _save(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def annotators_of(row: dict) -> list[str]:
    return [str(a) for a in (row.get("annotators") or [])]


def is_reviewed(row: dict) -> bool:
    """True when this **blind** pass recorded an explicit 1–5 label."""
    return REVIEW_TAG in annotators_of(row)


def apply_review(row: dict, label: str, annotator: str) -> bool:
    """Set ``label`` and record ``annotator`` + ``blind-relabel``. Return True if label changed."""
    if label not in LABELS:
        raise ValueError(f"unknown label {label!r}")
    changed = str(row.get("label")) != label
    row["label"] = label
    anns = annotators_of(row)
    if annotator and annotator not in anns:
        anns.append(annotator)
    if REVIEW_TAG not in anns:
        anns.append(REVIEW_TAG)
    row["annotators"] = anns
    return changed


def parse_choice(raw: str) -> str:
    """Return ``quit``, ``skip``, a class name, or ``invalid``. Empty is invalid."""
    token = (raw or "").strip().lower()
    if token in {"q", "quit"}:
        return "quit"
    if token in {"s", "skip"}:
        return "skip"
    if token in KEYS:
        return KEYS[token]
    return "invalid"


def format_item(row: dict, n: int, total: int) -> str:
    """Prompt text with no current gold label."""
    image = row.get("image_path") or ""
    caption = row.get("caption") or ""
    return (
        f"[{n}/{total}]  post_id={row.get('post_id')}\n"
        f"image: {image}\n"
        f"caption: {caption}"
    )


def read_ids_file(path: Path) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        token = line.strip()
        if not token or token.startswith("#"):
            continue
        if token not in seen:
            seen.add(token)
            ids.append(token)
    return ids


def write_ids_file(path: Path, ids: list[str], *, header: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    if header:
        for hline in header.strip().splitlines():
            lines.append(hline if hline.startswith("#") else f"# {hline}")
    lines.extend(ids)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def select_overlap_ids(
    rows: list[dict],
    *,
    n_non_sarcasm: int = DEFAULT_N_NON_SARCASM,
    seed: int = 0,
    require_image: bool = True,
) -> list[str]:
    """All current sarcasm ids, plus a stratified sample of the rest."""
    usable: list[dict] = []
    for row in rows:
        if require_image and not Path(str(row.get("image_path") or "")).is_file():
            continue
        if not row.get("post_id"):
            continue
        usable.append(row)
    sarc = [r for r in usable if str(r.get("label")) in SARCASM]
    rest = [r for r in usable if str(r.get("label")) not in SARCASM]
    by_lab: dict[str, list[dict]] = defaultdict(list)
    for row in rest:
        by_lab[str(row.get("label") or "")].append(row)
    rng = random.Random(seed)
    for lab in by_lab:
        rng.shuffle(by_lab[lab])
    n_take = min(max(0, int(n_non_sarcasm)), len(rest))
    sampled: list[dict] = []
    classes = sorted(by_lab.keys())
    cursor = {c: 0 for c in classes}
    while len(sampled) < n_take and classes:
        progressed = False
        for lab in classes:
            i = cursor[lab]
            bucket = by_lab[lab]
            if i < len(bucket):
                sampled.append(bucket[i])
                cursor[lab] = i + 1
                progressed = True
                if len(sampled) >= n_take:
                    break
        if not progressed:
            break
    ordered = sarc + sampled
    out: list[str] = []
    seen: set[str] = set()
    for row in ordered:
        pid = str(row["post_id"])
        if pid not in seen:
            seen.add(pid)
            out.append(pid)
    return out


def matching_indices(
    rows: list[dict],
    *,
    only: str,
    pending_only: bool,
    ids: set[str] | None = None,
) -> list[int]:
    indices: list[int] = []
    for i, row in enumerate(rows):
        pid = str(row.get("post_id") or "")
        if ids is not None and pid not in ids:
            continue
        lab = str(row.get("label") or "")
        if only == "all":
            ok = True
        elif only == "sarcasm":
            ok = lab in SARCASM
        else:
            ok = lab == only
        if not ok:
            continue
        if pending_only and is_reviewed(row):
            continue
        indices.append(i)
    return indices


def matching_indices_ordered(
    rows: list[dict],
    *,
    id_order: list[str],
    pending_only: bool,
    already_done: set[str] | None = None,
    skip_if_gold_reviewed: bool = True,
) -> list[int]:
    """Walk ``id_order`` (overlap file) rather than jsonl order.

    Second-annotator mode must pass ``skip_if_gold_reviewed=False`` so A's
    ``blind-relabel`` tag does not hide the post from B.
    """
    by_id = {str(r.get("post_id")): i for i, r in enumerate(rows)}
    done = already_done if already_done is not None else set()
    indices: list[int] = []
    for pid in id_order:
        i = by_id.get(pid)
        if i is None:
            continue
        if pending_only:
            if pid in done:
                continue
            if skip_if_gold_reviewed and is_reviewed(rows[i]):
                continue
        indices.append(i)
    return indices


def secondary_done_ids(path: Path) -> set[str]:
    return {str(r.get("post_id")) for r in _load(path) if r.get("post_id")}


def upsert_secondary(
    path: Path,
    *,
    post_id: str,
    annotator: str,
    label: str,
    caption: str = "",
    image_path: str = "",
) -> None:
    if label not in LABELS:
        raise ValueError(f"unknown label {label!r}")
    rows = _load(path)
    found = False
    for row in rows:
        if str(row.get("post_id")) == post_id:
            row["annotator_id"] = annotator
            row["label"] = label
            row["caption"] = caption
            row["image_path"] = image_path
            found = True
            break
    if not found:
        rows.append(
            {
                "post_id": post_id,
                "annotator_id": annotator,
                "label": label,
                "caption": caption,
                "image_path": image_path,
            }
        )
    _save(path, rows)


def sync_feature_labels(
    dataset: Path,
    features: Path = DEFAULT_FEATURES,
) -> int:
    """Copy jsonl labels onto ``y`` in the feature cache. Does not change ``X``."""
    if not features.is_file():
        raise FileNotFoundError(f"features cache not found: {features}")
    import numpy as np

    rows = _load(dataset)
    by_id = {str(r["post_id"]): r["label"] for r in rows}
    z = dict(np.load(features, allow_pickle=True))
    if "post_ids" not in z:
        raise KeyError(f"{features} has no post_ids; cannot sync labels")
    old = [str(v) for v in z["y"].tolist()]
    new = [by_id.get(str(pid), y) for pid, y in zip(z["post_ids"], old)]
    changed = sum(a != b for a, b in zip(old, new))
    z["y"] = np.array(new, dtype=object)
    np.savez_compressed(features, **z)
    return changed


def _prompt_loop(
    rows: list[dict],
    todo: list[int],
    *,
    annotator: str,
    dataset: Path,
    secondary_out: Path | None,
) -> tuple[int, int]:
    print("Keys:  1 positive  2 negative  3 neutral  4 positive_sarcasm  5 negative_sarcasm")
    print("       1–5 required (current label is hidden).  s = skip   q = quit\n")
    recorded = 0
    changed = 0
    for n, idx in enumerate(todo, start=1):
        row = rows[idx]
        print("=" * 70)
        print(format_item(row, n, len(todo)))
        _open_image(Path(str(row.get("image_path") or "")))
        try:
            choice = parse_choice(input("> "))
        except EOFError:
            break
        if choice == "quit":
            break
        if choice == "skip":
            print("  skipped (not recorded)")
            continue
        if choice == "invalid":
            print("invalid — use 1-5, s, or q (Enter does not confirm a label)")
            continue
        if secondary_out is not None:
            upsert_secondary(
                secondary_out,
                post_id=str(row.get("post_id")),
                annotator=annotator,
                label=choice,
                caption=str(row.get("caption") or ""),
                image_path=str(row.get("image_path") or ""),
            )
            print(f"  recorded {choice} -> {secondary_out}")
        else:
            if apply_review(row, choice, annotator):
                changed += 1
            _save(dataset, rows)
            print(f"  recorded {choice}")
        recorded += 1
    return recorded, changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Blind 5-class review of the existing jsonl (no current label shown)."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--only",
        choices=("all", "sarcasm", "positive", "negative", "neutral"),
        default="sarcasm",
        help="Which rows to review when --ids-file is not set (default: sarcasm).",
    )
    parser.add_argument("--start", type=int, default=0, help="Skip the first N matching rows.")
    parser.add_argument(
        "--annotator",
        default=DEFAULT_ANNOTATOR,
        help="Human id recorded on each explicit 1–5 label.",
    )
    parser.add_argument(
        "--pending-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip rows already tagged blind-relabel (primary) or already in --out.",
    )
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Print pending counts and exit (no UI).",
    )
    parser.add_argument(
        "--ids-file",
        type=Path,
        default=None,
        help="Restrict to these post_ids (overlap list). Use --only all with this.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Second annotator: write independent labels here; do not change gold jsonl.",
    )
    parser.add_argument(
        "--export-overlap",
        type=Path,
        nargs="?",
        const=DEFAULT_OVERLAP_IDS,
        default=None,
        help="Write sarcasm + stratified non-sarcasm ids and exit.",
    )
    parser.add_argument(
        "--n-non-sarcasm",
        type=int,
        default=DEFAULT_N_NON_SARCASM,
        help="Non-sarcasm overlap sample size for --export-overlap (default 100).",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if not args.dataset.is_file():
        print(f"dataset not found: {args.dataset}")
        return 1

    rows = _load(args.dataset)

    if args.export_overlap is not None:
        ids = select_overlap_ids(
            rows,
            n_non_sarcasm=args.n_non_sarcasm,
            seed=args.seed,
        )
        n_sarc = sum(
            1
            for pid in ids
            for r in rows
            if str(r.get("post_id")) == pid and str(r.get("label")) in SARCASM
        )
        write_ids_file(
            args.export_overlap,
            ids,
            header=(
                f"IAA overlap: all sarcasm with images plus {args.n_non_sarcasm} "
                f"stratified non-sarcasm (seed={args.seed}).\n"
                f"sarcasm_ids={n_sarc} total_ids={len(ids)}\n"
                "Second annotator: python tasks.py relabel --only all "
                f"--ids-file {args.export_overlap} --out {DEFAULT_SECOND} "
                "--annotator PERSON2"
            ),
        )
        print(f"wrote {len(ids)} ids ({n_sarc} sarcasm) to {args.export_overlap}")
        return 0

    id_order: list[str] | None = None
    id_set: set[str] | None = None
    if args.ids_file is not None:
        if not args.ids_file.is_file():
            print(f"ids file not found: {args.ids_file}")
            return 1
        id_order = read_ids_file(args.ids_file)
        id_set = set(id_order)

    secondary_done: set[str] = set()
    if args.out is not None:
        secondary_done = secondary_done_ids(args.out)

    if id_order is not None:
        skip_gold = args.out is None
        done = secondary_done if args.out is not None else None
        indices = matching_indices_ordered(
            rows,
            id_order=id_order,
            pending_only=args.pending_only,
            already_done=done,
            skip_if_gold_reviewed=skip_gold,
        )
        listed = matching_indices_ordered(
            rows,
            id_order=id_order,
            pending_only=False,
            skip_if_gold_reviewed=False,
        )
        pending = matching_indices_ordered(
            rows,
            id_order=id_order,
            pending_only=True,
            already_done=done,
            skip_if_gold_reviewed=skip_gold,
        )
    else:
        listed = matching_indices(rows, only=args.only, pending_only=False, ids=id_set)
        if args.out is not None:
            pending = [
                i
                for i in listed
                if (not args.pending_only) or str(rows[i].get("post_id")) not in secondary_done
            ]
            indices = pending if args.pending_only else listed
        else:
            indices = matching_indices(
                rows, only=args.only, pending_only=args.pending_only, ids=id_set
            )
            pending = matching_indices(
                rows, only=args.only, pending_only=True, ids=id_set
            )

    if args.count_only:
        print(
            f"only={args.only} matching={len(listed)} pending={len(pending)} "
            f"reviewed={len(listed) - len(pending)} total_rows={len(rows)} "
            f"blind_tag={REVIEW_TAG}"
        )
        return 0

    todo = indices[max(args.start, 0) :]
    mode = f"secondary -> {args.out}" if args.out is not None else f"gold -> {args.dataset}"
    print(
        f"{len(todo)} posts to review ({mode}; of {len(listed)} listed, "
        f"{len(pending)} still pending, {len(rows)} total)."
    )
    recorded, changed = _prompt_loop(
        rows,
        todo,
        annotator=args.annotator,
        dataset=args.dataset,
        secondary_out=args.out,
    )
    print(
        f"\ndone. recorded {recorded} ({changed} gold label changes) in "
        f"{args.out or args.dataset}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
