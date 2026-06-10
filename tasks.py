"""Cross-platform task runner for the FA-GDCNet pipeline.

Usage:
    python tasks.py setup
    python tasks.py scrape --following --max-count <n>   # default source
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
    # Use ASCII-safe preview so Windows cp1252 consoles don't crash on Persian args.
    safe_preview = " ".join(ascii(part) for part in cmd)
    print(f"$ {safe_preview}")
    return subprocess.call(cmd, env=env)


def cmd_setup(_: argparse.Namespace) -> int:
    return subprocess.call(
        [sys.executable, "-m", "pip", "install", "-e", ".[dev]"],
        cwd=str(ROOT),
    )


def cmd_scrape_session(args: argparse.Namespace) -> int:
    """Import Instagram session from browser cookies (Firefox recommended)."""
    import subprocess as sp

    user = args.user
    session_file = Path(args.session_file).expanduser() if args.session_file else None
    if session_file is None:
        session_file = Path.home() / "AppData" / "Local" / "Instaloader" / f"session-{user}"
    session_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        _venv_python(),
        "-m",
        "instaloader",
        "--load-cookies",
        args.browser,
        "-f",
        str(session_file),
        "profile",
        user,
        "--no-posts",
        "--no-profile-pic",
    ]
    print(f"$ {' '.join(cmd)}")
    rc = sp.call(cmd, cwd=str(ROOT))
    if rc == 0 and session_file.is_file():
        print(f"Session ready: {session_file}")
        print(
            "Next (pick one source):\n"
            f"  .venv\\Scripts\\python.exe tasks.py scrape --following --max-count 200 "
            f"--session-user {user} --session-file \"{session_file}\"\n"
            f"  .venv\\Scripts\\python.exe tasks.py scrape --profile SOME_USER --max-count 50 "
            f"--session-user {user} --session-file \"{session_file}\""
        )
    return rc


def cmd_scrape(args: argparse.Namespace) -> int:
    extra: list[str] = ["--max-count", str(args.max_count)]
    has_source = bool(
        args.hashtag
        or getattr(args, "profile", None)
        or getattr(args, "profiles_file", None)
        or getattr(args, "following", False)
        or getattr(args, "followers_of", None)
    )
    if not has_source:
        extra.append("--following")
    if args.hashtag:
        extra += ["--hashtag", args.hashtag]
    if getattr(args, "profile", None):
        for handle in args.profile:
            extra += ["--profile", handle]
    if getattr(args, "profiles_file", None):
        extra += ["--profiles-file", args.profiles_file]
    if getattr(args, "following", False):
        extra.append("--following")
    if getattr(args, "followers_of", None):
        extra += ["--followers-of", args.followers_of]
    if getattr(args, "posts_per_profile", None) is not None:
        extra += ["--posts-per-profile", str(args.posts_per_profile)]
    if getattr(args, "max_profiles", None) is not None:
        extra += ["--max-profiles", str(args.max_profiles)]
    if getattr(args, "pool_name", None):
        extra += ["--pool-name", args.pool_name]
    if args.delay is not None:
        extra += ["--delay", str(args.delay)]
    if getattr(args, "username", None):
        extra += ["--username", args.username]
    if getattr(args, "password", None):
        extra += ["--password", args.password]
    if getattr(args, "session_user", None):
        extra += ["--session-user", args.session_user]
    if getattr(args, "session_file", None):
        extra += ["--session-file", args.session_file]
    return _run("data.scrape", *extra)


def cmd_bootstrap_labels(args: argparse.Namespace) -> int:
    extra: list[str] = []
    if args.input:
        extra += ["--input", args.input]
    if args.dataset:
        extra += ["--dataset", args.dataset]
    if getattr(args, "method", None):
        extra += ["--method", args.method]
    return _run("data.bootstrap_labels", *extra)


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
    scrape_src = p_scrape.add_mutually_exclusive_group(required=False)
    scrape_src.add_argument(
        "--hashtag",
        help="[legacy] Hashtag without # — mostly landscapes/ads; prefer --following.",
    )
    scrape_src.add_argument(
        "--profile",
        action="append",
        metavar="USERNAME",
        help="Personal account to scrape (repeatable; best for faces + daily captions).",
    )
    scrape_src.add_argument(
        "--profiles-file",
        help="File with one Instagram username per line.",
    )
    scrape_src.add_argument(
        "--following",
        action="store_true",
        help="Scrape posts from accounts your session user follows.",
    )
    scrape_src.add_argument(
        "--followers-of",
        metavar="USERNAME",
        help="Scrape posts from accounts that follow USERNAME.",
    )
    p_scrape.add_argument("--max-count", type=int, default=200)
    p_scrape.add_argument("--posts-per-profile", type=int, default=15)
    p_scrape.add_argument("--max-profiles", type=int, default=30)
    p_scrape.add_argument("--pool-name", default=None, help="Output JSONL basename override.")
    p_scrape.add_argument("--delay", type=float, default=None)
    p_scrape.add_argument("--username", default=None, help="Instagram login (or INSTAGRAM_USERNAME).")
    p_scrape.add_argument("--password", default=None, help="Instagram password (or INSTAGRAM_PASSWORD).")
    p_scrape.add_argument("--session-user", default=None, help="Reuse instaloader saved session.")
    p_scrape.add_argument("--session-file", default=None, help="Explicit path to instaloader session file.")
    p_scrape.set_defaults(func=cmd_scrape)

    p_sess = sub.add_parser("scrape-session", help="Import Instagram session from browser cookies.")
    p_sess.add_argument("--user", required=True, help="Instagram username (e.g. sjjd6502).")
    p_sess.add_argument("--browser", default="firefox", help="Browser for cookies (default: firefox).")
    p_sess.add_argument("--session-file", default=None, help="Where to save session file.")
    p_sess.set_defaults(func=cmd_scrape_session)

    p_label = sub.add_parser("label")
    p_label.add_argument("--annotator", default=None)
    p_label.add_argument("--input", default=None)
    p_label.set_defaults(func=cmd_label)

    p_boot = sub.add_parser(
        "label-bootstrap",
        help="Weak-label scraped raw pool with ParsBERT (positive/negative/neutral).",
    )
    p_boot.add_argument("--input", default=None, help="Raw JSONL (default: datasets/raw/following.jsonl).")
    p_boot.add_argument("--dataset", default=None, help="Output labeled dataset path.")
    p_boot.add_argument(
        "--method",
        choices=("auto", "parsbert", "keywords"),
        default="keywords",
        help="Weak label method (default: keywords, no GPU/torch needed).",
    )
    p_boot.set_defaults(func=cmd_bootstrap_labels)

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
