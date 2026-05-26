"""Legacy Streamlit path. Prefer: python tasks.py dashboard"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

os.chdir(ROOT)
os.environ["PYTHONPATH"] = str(SRC) + os.pathsep + os.environ.get("PYTHONPATH", "")
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from explain.ui import run_dashboard_app

run_dashboard_app()
