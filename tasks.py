"""Cross-platform task runner for the FA-GDCNet pipeline.

Usage:
    python tasks.py setup
    python tasks.py scrape --hashtag <tag> --max-count <n>
    python tasks.py label [--annotator <id>]
    python tasks.py train [--dataset <path>]
    python tasks.py eval
    python tasks.py dashboard
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"


def _venv_python() -> str:
    venv_py = ROOT / ".venv" / "Scripts" / "python.exe"
    return str(venv_py) if venv_py.is_file() else sys.executable


def _run(module: str, *args: str) -> int:
    cmd = [_venv_python(), "-m", module, *args]
    env_pythonpath = str(SRC)
    import os

    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = env_pythonpath + (os.pathsep + existing if existing else "")
    print(f"$ {' '.join(cmd)}")
    return subprocess.call(cmd, env=env)


def cmd_setup(_: argparse.Namespace) -> int:
    return subprocess.call(
        [sys.executable, "-m", "pip", "install", "-e", ".[dev]"],
        cwd=str(ROOT),
    )


def cmd_scrape(args: argparse.Namespace) -> int:
    extra: list[str] = ["--hashtag", args.hashtag, "--max-count", str(args.max_count)]
    if args.delay is not None:
        extra += ["--delay", str(args.delay)]
    if getattr(args, "username", None):
        extra += ["--username", args.username]
    if getattr(args, "password", None):
        extra += ["--password", args.password]
    if getattr(args, "session_user", None):
        extra += ["--session-user", args.session_user]
    return _run("data.scrape", *extra)


def cmd_label(args: argparse.Namespace) -> int:
    extra: list[str] = []
    if args.annotator:
        extra += ["--annotator", args.annotator]
    if args.input:
        extra += ["--input", args.input]
    return _run("data.label", *extra)


def cmd_train(args: argparse.Namespace) -> int:
    extra: list[str] = []
    if args.dataset:
        extra += ["--dataset", args.dataset]
    return _run("inference.classifier", *extra)


def cmd_eval(_: argparse.Namespace) -> int:
    rc = _run("eval.metrics")
    if rc != 0:
        return rc
    rc = _run("eval.profile")
    if rc != 0:
        return rc
    rc = _run("eval.ablation")
    if rc != 0:
        return rc
    rc = _run("eval.baseline")
    if rc != 0:
        return rc
    return _run("eval.report")


def cmd_dashboard(_: argparse.Namespace) -> int:
    launcher = ROOT / "scripts" / "run_dashboard.py"
    import os

    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(SRC) + (os.pathsep + existing if existing else "")
    return subprocess.call([_venv_python(), str(launcher)], cwd=str(ROOT), env=env)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("setup").set_defaults(func=cmd_setup)

    p_scrape = sub.add_parser("scrape")
    p_scrape.add_argument("--hashtag", required=True)
    p_scrape.add_argument("--max-count", type=int, default=200)
    p_scrape.add_argument("--delay", type=float, default=None)
    p_scrape.add_argument("--username", default=None, help="Instagram login (or INSTAGRAM_USERNAME).")
    p_scrape.add_argument("--password", default=None, help="Instagram password (or INSTAGRAM_PASSWORD).")
    p_scrape.add_argument("--session-user", default=None, help="Reuse instaloader saved session.")
    p_scrape.set_defaults(func=cmd_scrape)

    p_label = sub.add_parser("label")
    p_label.add_argument("--annotator", default=None)
    p_label.add_argument("--input", default=None)
    p_label.set_defaults(func=cmd_label)

    p_train = sub.add_parser("train")
    p_train.add_argument("--dataset", default=None)
    p_train.set_defaults(func=cmd_train)

    sub.add_parser("eval").set_defaults(func=cmd_eval)
    sub.add_parser("dashboard").set_defaults(func=cmd_dashboard)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
