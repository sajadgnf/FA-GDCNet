#!/usr/bin/env python3
"""Streamlit entry point — always use: python tasks.py dashboard"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

os.chdir(ROOT)
os.environ["PYTHONPATH"] = str(SRC) + os.pathsep + os.environ.get("PYTHONPATH", "")
sys.path.insert(0, str(SRC))

# Prove which file Streamlit is executing (check the terminal if the UI fails).
print(f"[FA-GDCNet] dashboard entry: {Path(__file__).resolve()}", flush=True)

from explain.ui import run_dashboard_app

run_dashboard_app()
