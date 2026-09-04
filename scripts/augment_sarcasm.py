"""Deprecated: weak ParsBERT sarcasm labels are not gold.

Use ``python tasks.py enqueue-sarcasm`` then blind review:

    python tasks.py relabel --only candidates --ids-file datasets/sarcasm_candidate_ids.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.enqueue_sarcasm import main

if __name__ == "__main__":
    raise SystemExit(main())
