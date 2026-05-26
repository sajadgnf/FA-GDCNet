#!/usr/bin/env python3
"""Launch the FA-GDCNet Streamlit dashboard."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
ENTRY = ROOT / "scripts" / "fa_gdcnet_dashboard.py"

os.chdir(ROOT)
os.environ["PYTHONPATH"] = str(SRC) + os.pathsep + os.environ.get("PYTHONPATH", "")
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from streamlit.web import cli as stcli

sys.argv = [
    "streamlit",
    "run",
    str(ENTRY),
    "--server.runOnSave",
    "true",
    "--server.fileWatcherType",
    "poll",
    "--browser.gatherUsageStats",
    "false",
]
raise SystemExit(stcli.main())
