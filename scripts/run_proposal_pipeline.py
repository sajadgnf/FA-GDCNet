"""End-to-end proposal pipeline: extract, train, eval, augment if needed.

Runs unattended. Re-invoking is safe: each stage resumes from its checkpoint.
"""

from __future__ import annotations

import csv
import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SARCASM_CSV = ROOT / "reports" / "sarcasm.csv"
ACC_FLOOR = 0.70


def _run(cmd: list[str]) -> int:
    logging.info(">>> %s", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(ROOT))


def _sarcasm_accuracy() -> float | None:
    if not SARCASM_CSV.is_file():
        return None
    with SARCASM_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0] == "# mean_accuracy":
                return float(row[1])
            if len(row) >= 2 and row[0] == "mean±std":
                return float(row[1].split("±")[0])
    return None


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    py = sys.executable

    if _run([py, "tasks.py", "extract"]) != 0:
        return 1
    if _run([py, "tasks.py", "finish"]) != 0:
        return 1

    acc = _sarcasm_accuracy()
    logging.info("binary sarcasm accuracy = %s", acc)
    if acc is not None and acc >= ACC_FLOOR:
        logging.info("proposal accuracy hypothesis met")
        return 0

    logging.info("accuracy below %.0f%% — augmenting sarcasm pool and re-running", ACC_FLOOR * 100)
    if _run([py, "tasks.py", "augment-sarcasm"]) != 0:
        return 1
    if _run([py, "tasks.py", "extract"]) != 0:
        return 1
    if _run([py, "tasks.py", "finish"]) != 0:
        return 1

    acc2 = _sarcasm_accuracy()
    logging.info("final binary sarcasm accuracy = %s", acc2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
