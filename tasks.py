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
        # No profile/hashtag targets — only import cookies and save session.
        # Adding "profile USER" triggers extra GraphQL calls that often fail on
        # filtered networks even when cookie login succeeded.
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
        or getattr(args, "hashtags_file", None)
        or getattr(args, "profile", None)
        or getattr(args, "profiles_file", None)
        or getattr(args, "following", False)
        or getattr(args, "followers_of", None)
    )
    if not has_source:
        extra.append("--following")
    if args.hashtag:
        extra += ["--hashtag", args.hashtag]
    if getattr(args, "hashtags_file", None):
        extra += ["--hashtags-file", args.hashtags_file]
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
    if getattr(args, "require_face", False):
        extra.append("--require-face")
    if getattr(args, "min_face_size", None) is not None:
        extra += ["--min-face-size", str(args.min_face_size)]
    if getattr(args, "request_timeout", None) is not None:
        extra += ["--request-timeout", str(args.request_timeout)]
    if getattr(args, "max_connection_attempts", None) is not None:
        extra += ["--max-connection-attempts", str(args.max_connection_attempts)]
    return _run("data.scrape", *extra)


def cmd_import_pool(args: argparse.Namespace) -> int:
    extra: list[str] = []
    if args.input_dir:
        extra += ["--input-dir", args.input_dir]
    if args.manifest:
        extra += ["--manifest", args.manifest]
    if args.pool_name:
        extra += ["--pool-name", args.pool_name]
    if getattr(args, "require_face", False):
        extra.append("--require-face")
    if getattr(args, "min_face_size", None) is not None:
        extra += ["--min-face-size", str(args.min_face_size)]
    return _run("data.import_pool", *extra)


def cmd_import_links(args: argparse.Namespace) -> int:
    extra: list[str] = ["--links", args.links]
    if args.pool_name:
        extra += ["--pool-name", args.pool_name]
    if getattr(args, "require_face", False):
        extra.append("--require-face")
    if getattr(args, "sarcasm_candidates", False):
        extra.append("--sarcasm-candidates")
    if getattr(args, "delay", None) is not None:
        extra += ["--delay", str(args.delay)]
    if getattr(args, "timeout", None) is not None:
        extra += ["--timeout", str(args.timeout)]
    return _run("data.scrape_embed", *extra)


def cmd_collect_hashtags(args: argparse.Namespace) -> int:
    extra: list[str] = ["--hashtags-file", args.hashtags_file]
    if args.pool_name:
        extra += ["--pool-name", args.pool_name]
    extra += ["--max-count", str(args.max_count)]
    if getattr(args, "require_face", False):
        extra.append("--require-face")
    if getattr(args, "sarcasm_candidates", False):
        extra.append("--sarcasm-candidates")
    if getattr(args, "delay", None) is not None:
        extra += ["--delay", str(args.delay)]
    if getattr(args, "timeout", None) is not None:
        extra += ["--timeout", str(args.timeout)]
    if getattr(args, "min_face_size", None) is not None:
        extra += ["--min-face-size", str(args.min_face_size)]
    if getattr(args, "headed", False):
        extra.append("--headed")
    if getattr(args, "scrolls", None) is not None:
        extra += ["--scrolls", str(args.scrolls)]
    if getattr(args, "page_timeout", None) is not None:
        extra += ["--page-timeout", str(args.page_timeout)]
    if getattr(args, "browser", None):
        extra += ["--browser", args.browser]
    if getattr(args, "hashtag_contains", False):
        extra.append("--hashtag-contains")
    if getattr(args, "max_search_tags", None) is not None:
        extra += ["--max-search-tags", str(args.max_search_tags)]
    if getattr(args, "login_wait", None) is not None:
        extra += ["--login-wait", str(args.login_wait)]
    return _run("data.collect_browser", *extra)


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


def cmd_relabel(args: argparse.Namespace) -> int:
    extra: list[str] = ["--only", args.only, "--annotator", args.annotator]
    if args.start:
        extra += ["--start", str(args.start)]
    if args.dataset:
        extra += ["--dataset", args.dataset]
    if not args.pending_only:
        extra.append("--no-pending-only")
    if getattr(args, "count_only", False):
        extra.append("--count-only")
    if getattr(args, "ids_file", None):
        extra += ["--ids-file", str(args.ids_file)]
    if getattr(args, "out", None):
        extra += ["--out", str(args.out)]
    if getattr(args, "export_overlap", None):
        extra += ["--export-overlap", str(args.export_overlap)]
    extra += ["--n-non-sarcasm", str(args.n_non_sarcasm), "--seed", str(args.seed)]
    return _run("data.relabel", *extra)


def cmd_enqueue_sarcasm(args: argparse.Namespace) -> int:
    extra: list[str] = []
    if args.dataset:
        extra += ["--dataset", args.dataset]
    if args.ids_file:
        extra += ["--ids-file", args.ids_file]
    if args.max_queue is not None:
        extra += ["--max", str(args.max_queue)]
    if args.no_gold_heuristic:
        extra.append("--no-gold-heuristic")
    if getattr(args, "from_pools_only", False):
        extra.append("--from-pools-only")
    for pool in getattr(args, "unfiltered_pool", None) or []:
        extra += ["--unfiltered-pool", str(pool)]
    if args.dry_run:
        extra.append("--dry-run")
    return _run("data.enqueue_sarcasm", *extra)


def cmd_iaa(args: argparse.Namespace) -> int:
    extra: list[str] = ["--gold-annotator", args.gold_annotator]
    if args.dataset:
        extra += ["--dataset", args.dataset]
    if args.second:
        extra += ["--second", args.second]
    if args.out:
        extra += ["--out", args.out]
    return _run("data.iaa", *extra)


def cmd_sync_labels(args: argparse.Namespace) -> int:
    extra: list[str] = []
    if args.dataset:
        extra += ["--dataset", args.dataset]
    if args.features_cache:
        extra += ["--features-cache", args.features_cache]
    return _run("data.sync_labels", *extra)


def cmd_retag(args: argparse.Namespace) -> int:
    extra: list[str] = []
    if args.dry_run:
        extra.append("--dry-run")
    if args.skip_images:
        extra.append("--skip-images")
    if args.dataset:
        extra += ["--dataset", args.dataset]
    return _run("data.retag_dataset", *extra)


def cmd_drop_unreviewed(args: argparse.Namespace) -> int:
    extra: list[str] = []
    if args.dataset:
        extra += ["--dataset", args.dataset]
    if args.ids_file:
        extra += ["--ids-file", args.ids_file]
    for pool in getattr(args, "pool", None) or []:
        extra += ["--pool", str(pool)]
    if args.keep_images:
        extra.append("--keep-images")
    if args.dry_run:
        extra.append("--dry-run")
    return _run("data.drop_unreviewed", *extra)


def cmd_craft_sarcasm(args: argparse.Namespace) -> int:
    extra: list[str] = ["--pool-name", args.pool_name]
    if args.out_dir:
        extra += ["--out-dir", args.out_dir]
    if args.timeout is not None:
        extra += ["--timeout", str(args.timeout)]
    if args.no_require_face:
        extra.append("--no-require-face")
    if getattr(args, "from_existing", False):
        extra.append("--from-existing")
    if getattr(args, "recaption_queue", False):
        extra.append("--recaption-queue")
    if getattr(args, "pending_only", False):
        extra.append("--pending-only")
    if getattr(args, "from_labeled", False):
        extra.append("--from-labeled")
    if getattr(args, "no_clip_gate", False):
        extra.append("--no-clip-gate")
    if getattr(args, "min_clip", None) is not None:
        extra += ["--min-clip", str(args.min_clip)]
    if getattr(args, "dataset", None):
        extra += ["--dataset", args.dataset]
    if getattr(args, "max_n", None) is not None:
        extra += ["--max", str(args.max_n)]
    if getattr(args, "replace", False):
        extra.append("--replace")
    return _run("data.craft_sarcasm", *extra)


def cmd_prune_pool(args: argparse.Namespace) -> int:
    extra: list[str] = ["--input", args.input]
    if args.annotator:
        extra += ["--annotator", args.annotator]
    if args.dry_run:
        extra.append("--dry-run")
    return _run("data.prune_pool", *extra)


def cmd_extract(args: argparse.Namespace) -> int:
    extra: list[str] = []
    if args.dataset:
        extra += ["--dataset", args.dataset]
    for stage in args.stage or []:
        extra += ["--stage", stage]
    return _run("inference.stages", *extra)


def cmd_train(args: argparse.Namespace) -> int:
    extra: list[str] = []
    if args.dataset:
        extra += ["--dataset", args.dataset]
    if getattr(args, "from_cache", False):
        extra.append("--from-cache")
    return _run("inference.classifier", *extra)


def cmd_eval(args: argparse.Namespace) -> int:
    rc = _run("eval.metrics")
    if rc != 0:
        return rc
    rc = _run("eval.sarcasm")
    if rc != 0:
        return rc
    rc = _run("eval.profile_staged")
    if rc != 0:
        return rc
    rc = _run("eval.ablation")
    if rc != 0:
        return rc
    rc = _run("eval.baseline")
    if rc != 0:
        return rc
    rc = _run("eval.origin_split")
    if rc != 0:
        return rc
    if getattr(args, "heavy", False):
        rc = _run("eval.heavy_baseline")
        if rc != 0:
            return rc
    return _run("eval.report")


def cmd_finish(_: argparse.Namespace) -> int:
    """Resume staged extraction, train classifier, run full eval suite."""
    rc = _run("inference.stages")
    if rc != 0:
        return rc
    rc = _run("inference.classifier", "--from-cache")
    if rc != 0:
        return rc
    return cmd_eval(_)


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
        help="Single hashtag without # — public posts from any account.",
    )
    scrape_src.add_argument(
        "--hashtags-file",
        help="File with hashtags (one per line) — discover posts beyond your account list.",
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
    p_scrape.add_argument(
        "--require-face",
        action="store_true",
        help="Skip images with no detected frontal face (landscapes, logos, etc.).",
    )
    p_scrape.add_argument("--min-face-size", type=int, default=40)
    p_scrape.add_argument("--delay", type=float, default=None)
    p_scrape.add_argument("--request-timeout", type=float, default=None)
    p_scrape.add_argument("--max-connection-attempts", type=int, default=None)
    p_scrape.add_argument("--username", default=None, help="Instagram login (or INSTAGRAM_USERNAME).")
    p_scrape.add_argument("--password", default=None, help="Instagram password (or INSTAGRAM_PASSWORD).")
    p_scrape.add_argument("--session-user", default=None, help="Reuse instaloader saved session.")
    p_scrape.add_argument("--session-file", default=None, help="Explicit path to instaloader session file.")
    p_scrape.set_defaults(func=cmd_scrape)

    p_import = sub.add_parser(
        "import-pool",
        help="Import locally saved images into a raw pool (no Instagram API).",
    )
    p_import.add_argument(
        "--input-dir",
        default="datasets/raw/inbox",
        help="Folder with <post_id>.jpg and optional <post_id>.txt caption.",
    )
    p_import.add_argument(
        "--manifest",
        default=None,
        help="Optional JSONL manifest (post_id, caption, image_path).",
    )
    p_import.add_argument("--pool-name", default="hashtags")
    p_import.add_argument(
        "--require-face",
        action="store_true",
        help="Keep only images with a detected frontal face.",
    )
    p_import.add_argument("--min-face-size", type=int, default=40)
    p_import.set_defaults(func=cmd_import_pool)

    p_links = sub.add_parser(
        "import-links",
        help="Import post URLs via embed pages (works when instaloader API times out).",
    )
    p_links.add_argument(
        "--links",
        default="datasets/raw/links.txt",
        help="File with instagram.com/p/... URLs or shortcodes, one per line.",
    )
    p_links.add_argument("--pool-name", default="hashtags")
    p_links.add_argument("--require-face", action="store_true")
    p_links.add_argument("--sarcasm-candidates", action="store_true")
    p_links.add_argument("--delay", type=float, default=2.0)
    p_links.add_argument("--timeout", type=float, default=30.0)
    p_links.set_defaults(func=cmd_import_links)

    p_collect = sub.add_parser(
        "collect-hashtags",
        help="Auto-collect hashtag posts via Firefox browser (no instaloader API).",
    )
    p_collect.add_argument("--hashtags-file", default="datasets/raw/hashtags.txt")
    p_collect.add_argument("--pool-name", default="hashtags")
    p_collect.add_argument("--max-count", type=int, default=30)
    p_collect.add_argument("--require-face", action="store_true")
    p_collect.add_argument(
        "--sarcasm-candidates",
        action="store_true",
        help="Skip captions with no polarity-clash cue (not irony hashtags alone).",
    )
    p_collect.add_argument("--delay", type=float, default=8.0)
    p_collect.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="Per-post fetch timeout in seconds (default 20).",
    )
    p_collect.add_argument("--min-face-size", type=int, default=40)
    p_collect.add_argument("--browser", default="firefox")
    p_collect.add_argument(
        "--headed",
        action="store_true",
        help="Show browser window (recommended first run).",
    )
    p_collect.add_argument("--scrolls", type=int, default=6)
    p_collect.add_argument("--page-timeout", type=int, default=120)
    p_collect.add_argument(
        "--hashtag-contains",
        action="store_true",
        help="Search tags whose name includes each line (partial match), not exact tag only.",
    )
    p_collect.add_argument(
        "--max-search-tags",
        type=int,
        default=15,
        help="Max tags per search term when using --hashtag-contains or ?prefix (default 15).",
    )
    p_collect.add_argument(
        "--login-wait",
        type=int,
        default=180,
        help="Seconds to wait for Instagram login when --headed. 0 = wait until login.",
    )
    p_collect.set_defaults(func=cmd_collect_hashtags)

    p_sess = sub.add_parser("scrape-session", help="Import Instagram session from browser cookies.")
    p_sess.add_argument("--user", required=True, help="Instagram username (e.g. sjjd6502).")
    p_sess.add_argument("--browser", default="firefox", help="Browser for cookies (default: firefox).")
    p_sess.add_argument("--session-file", default=None, help="Where to save session file.")
    p_sess.set_defaults(func=cmd_scrape_session)

    p_label = sub.add_parser("label")
    p_label.add_argument("--annotator", default=None)
    p_label.add_argument("--input", default=None)
    p_label.set_defaults(func=cmd_label)

    p_relabel = sub.add_parser(
        "relabel",
        help="Blind 5-class review (no current label shown; Enter does not confirm).",
    )
    p_relabel.add_argument(
        "--only",
        choices=("all", "sarcasm", "candidates", "positive", "negative", "neutral"),
        default="sarcasm",
    )
    p_relabel.add_argument("--start", type=int, default=0)
    p_relabel.add_argument("--dataset", default=None)
    p_relabel.add_argument("--annotator", default="sjjd6502")
    p_relabel.add_argument(
        "--pending-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p_relabel.add_argument("--count-only", action="store_true")
    p_relabel.add_argument(
        "--ids-file",
        default=None,
        help="Restrict to post_ids in this file (use with --only all for overlap).",
    )
    p_relabel.add_argument(
        "--out",
        default=None,
        help="Second annotator file; does not change gold jsonl.",
    )
    p_relabel.add_argument(
        "--export-overlap",
        nargs="?",
        const="datasets/iaa_overlap_ids.txt",
        default=None,
        help="Write sarcasm + stratified non-sarcasm ids and exit.",
    )
    p_relabel.add_argument("--n-non-sarcasm", type=int, default=100)
    p_relabel.add_argument("--seed", type=int, default=0)
    p_relabel.set_defaults(func=cmd_relabel)

    p_iaa = sub.add_parser(
        "iaa",
        help="Cohen's kappa from blind gold vs datasets/iaa_second.jsonl.",
    )
    p_iaa.add_argument("--dataset", default=None)
    p_iaa.add_argument("--second", default=None)
    p_iaa.add_argument("--gold-annotator", default="sjjd6502")
    p_iaa.add_argument("--out", default=None)
    p_iaa.set_defaults(func=cmd_iaa)

    p_sync = sub.add_parser(
        "sync-labels",
        help="Copy jsonl labels onto artifacts/features.npz y (does not recompute X).",
    )
    p_sync.add_argument("--dataset", default=None)
    p_sync.add_argument("--features-cache", default=None)
    p_sync.set_defaults(func=cmd_sync_labels)

    p_retag = sub.add_parser(
        "retag",
        help="Auto-retag the dataset to the proposal 5-class scheme (CLIP + polarity).",
    )
    p_retag.add_argument("--dataset", default=None)
    p_retag.add_argument("--dry-run", action="store_true")
    p_retag.add_argument("--skip-images", action="store_true")
    p_retag.set_defaults(func=cmd_retag)

    p_prune = sub.add_parser(
        "prune-pool",
        help="Archive unlabeled posts, remove from pool, add to scrape ignore list.",
    )
    p_prune.add_argument("--input", required=True, help="Raw pool JSONL (e.g. datasets/raw/hashtags.jsonl).")
    p_prune.add_argument("--annotator", default=None)
    p_prune.add_argument("--dry-run", action="store_true")
    p_prune.set_defaults(func=cmd_prune_pool)

    p_drop = sub.add_parser(
        "drop-unreviewed",
        help="Delete unlabeled bootstrap candidates from gold (keep human 1–5 labels).",
    )
    p_drop.add_argument("--dataset", default=None)
    p_drop.add_argument(
        "--ids-file",
        default=None,
        help="Only drop these ids (default: all unreviewed bootstrap).",
    )
    p_drop.add_argument(
        "--pool",
        action="append",
        default=None,
        help="Raw pool JSONL to strip (repeatable). Default: datasets/raw/sarcasm.jsonl",
    )
    p_drop.add_argument("--keep-images", action="store_true")
    p_drop.add_argument("--dry-run", action="store_true")
    p_drop.set_defaults(func=cmd_drop_unreviewed)

    p_craft = sub.add_parser(
        "craft-sarcasm",
        help="Download face photos and pair them with clash captions (still unlabeled).",
    )
    p_craft.add_argument("--out-dir", default="datasets/raw")
    p_craft.add_argument("--pool-name", default="crafted")
    p_craft.add_argument("--timeout", type=float, default=30.0)
    p_craft.add_argument("--no-require-face", action="store_true")
    p_craft.add_argument(
        "--from-existing",
        action="store_true",
        help="Recaption Instagram faces already on disk (no download).",
    )
    p_craft.add_argument("--replace", action="store_true")
    p_craft.add_argument(
        "--from-labeled",
        action="store_true",
        help="Recaption faces from labeled positive/negative gold with clash captions.",
    )
    p_craft.add_argument("--dataset", default="datasets/persian_multimodal_irony.jsonl")
    p_craft.add_argument("--max", type=int, default=120, dest="max_n")
    p_craft.add_argument("--no-clip-gate", action="store_true")
    p_craft.add_argument("--min-clip", type=float, default=None)
    p_craft.add_argument("--recaption-queue", action="store_true")
    p_craft.add_argument(
        "--pending-only",
        action="store_true",
        help="With --recaption-queue, do not rewrite already-labeled rows.",
    )
    p_craft.set_defaults(func=cmd_craft_sarcasm)

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

    p_extract = sub.add_parser(
        "extract",
        help="Staged feature extraction (one backbone at a time, resumable).",
    )
    p_extract.add_argument("--dataset", default=None)
    p_extract.add_argument(
        "--stage",
        action="append",
        choices=("captions", "mclip", "polarity", "assemble"),
        help="Run only these stages (repeatable).",
    )
    p_extract.set_defaults(func=cmd_extract)

    p_train = sub.add_parser("train")
    p_train.add_argument("--dataset", default=None)
    p_train.add_argument(
        "--from-cache",
        action="store_true",
        help="Fit from artifacts/features.npz without re-extracting backbones.",
    )
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("eval")
    p_eval.add_argument(
        "--heavy",
        action="store_true",
        help="Also run Qwen2-VL Hypothesis 3 (can crash or pin the GPU).",
    )
    p_eval.set_defaults(func=cmd_eval)

    sub.add_parser(
        "finish",
        help="Resume extraction, train classifier, run eval (proposal metrics).",
    ).set_defaults(func=cmd_finish)

    p_aug = sub.add_parser(
        "augment-sarcasm",
        help="Deprecated alias of enqueue-sarcasm (does not assign sarcasm gold).",
    )
    p_aug.add_argument("--dataset", default=None)
    p_aug.add_argument("--ids-file", default=None)
    p_aug.add_argument("--max", type=int, default=None, dest="max_queue")
    p_aug.add_argument("--no-gold-heuristic", action="store_true")
    p_aug.add_argument("--dry-run", action="store_true")
    p_aug.set_defaults(func=cmd_enqueue_sarcasm)

    p_enq = sub.add_parser(
        "enqueue-sarcasm",
        help="Queue scrape-pool posts for blind sarcasm review (no auto labels).",
    )
    p_enq.add_argument("--dataset", default=None)
    p_enq.add_argument("--ids-file", default=None)
    p_enq.add_argument("--max", type=int, default=None, dest="max_queue")
    p_enq.add_argument("--no-gold-heuristic", action="store_true")
    p_enq.add_argument("--from-pools-only", action="store_true")
    p_enq.add_argument(
        "--unfiltered-pool",
        action="append",
        default=None,
        help="Raw JSONL to import without caption filter.",
    )
    p_enq.add_argument("--dry-run", action="store_true")
    p_enq.set_defaults(func=cmd_enqueue_sarcasm)

    sub.add_parser("dashboard").set_defaults(func=cmd_dashboard)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
