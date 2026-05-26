"""Idempotent Instagram scraper that respects rate limits.

Persists `(post_id, caption, image_path)` rows to `datasets/raw/<hashtag>.jsonl`
with shortcode-based deduplication. The image bytes are stored locally only —
images are never re-uploaded or redistributed (see README "نکته حقوقی").

We import `instaloader` lazily so unit tests of the dedup/resume logic can run
without the network library installed.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from contextlib import suppress
from pathlib import Path
from typing import Iterable, Iterator, Protocol

from .preprocess import is_persian_enough, preprocess_caption

log = logging.getLogger(__name__)

DEFAULT_RAW_DIR = Path("datasets") / "raw"
DEFAULT_REQUEST_DELAY = 4.0  # seconds; safe default for unauthenticated instaloader.


class ScrapedPost(Protocol):
    """The minimum surface `_persist` needs from any scraper backend."""

    @property
    def shortcode(self) -> str: ...

    @property
    def caption(self) -> str | None: ...

    def download_image(self, target: Path) -> Path: ...


def _existing_shortcodes(jsonl_path: Path) -> set[str]:
    if not jsonl_path.exists():
        return set()
    seen: set[str] = set()
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            with suppress(json.JSONDecodeError, KeyError):
                seen.add(json.loads(line)["post_id"])
    return seen


def _persist(
    posts: Iterable[ScrapedPost],
    jsonl_path: Path,
    image_dir: Path,
    *,
    delay: float,
    max_count: int,
) -> int:
    seen = _existing_shortcodes(jsonl_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    with jsonl_path.open("a", encoding="utf-8") as out:
        for post in posts:
            if written >= max_count:
                break
            sc = post.shortcode
            if sc in seen:
                log.debug("skip duplicate %s", sc)
                continue
            raw_caption = post.caption or ""
            caption = preprocess_caption(raw_caption)
            if not is_persian_enough(caption):
                log.debug("skip non-persian %s", sc)
                continue
            image_path = post.download_image(image_dir / f"{sc}.jpg")
            row = {
                "post_id": sc,
                "caption": caption,
                "image_path": str(image_path),
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            seen.add(sc)
            written += 1
            time.sleep(delay)
    return written


# ---------- Instaloader backend ----------------------------------------------


def _build_instaloader():
    import instaloader

    return instaloader.Instaloader(
        download_pictures=False,
        download_videos=False,
        download_comments=False,
        save_metadata=False,
        compress_json=False,
        post_metadata_txt_pattern="",
    )


def _authenticate_instaloader(
    loader,
    *,
    username: str | None,
    password: str | None,
    session_user: str | None,
) -> str:
    """Log in or load a saved session. Returns the Instagram username in use."""
    import instaloader

    user = (session_user or username or os.environ.get("INSTAGRAM_USERNAME") or "").strip()
    if not user:
        raise SystemExit(
            "Instagram requires login for hashtag scraping (403 login_required).\n"
            "Provide credentials via:\n"
            "  --username YOUR_IG_USER --password YOUR_IG_PASS\n"
            "  --session-user YOUR_IG_USER  (after: instaloader --login YOUR_IG_USER)\n"
            "  env INSTAGRAM_USERNAME / INSTAGRAM_PASSWORD\n"
            "Or skip scraping and run: python scripts/proposal_demo.py"
        )

    pwd = password if password is not None else os.environ.get("INSTAGRAM_PASSWORD")

    # instaloader 4.15+: load_session_from_file(username, filename=None) — no session_dir kwarg.
    from instaloader.instaloader import get_default_session_filename

    custom_dir = os.environ.get("INSTALOADER_SESSION_DIR")
    if custom_dir:
        session_file = str(Path(custom_dir) / f"session-{user}")
    else:
        session_file = get_default_session_filename(user)

    try:
        loader.load_session_from_file(user, filename=session_file)
        log.info("loaded Instagram session for %s from %s", user, session_file)
        return user
    except (FileNotFoundError, instaloader.exceptions.ConnectionException, OSError) as exc:
        log.debug("no saved session for %s: %s", user, exc)

    if not pwd:
        raise SystemExit(
            f"No saved session for {user!r} and no password given.\n"
            "Either run once:\n"
            f"  instaloader --login {user}\n"
            "or pass --password / set INSTAGRAM_PASSWORD.\n"
            f"Default session path on this machine: {session_file}"
        )

    log.info("logging in to Instagram as %s ...", user)
    loader.login(user, pwd)
    loader.save_session_to_file(session_file)
    log.info("session saved to %s (reuse with --session-user %s)", session_file, user)
    return user


def _iter_instaloader_posts(
    hashtag: str,
    *,
    username: str | None = None,
    password: str | None = None,
    session_user: str | None = None,
) -> Iterator[ScrapedPost]:
    import instaloader

    loader = _build_instaloader()
    _authenticate_instaloader(
        loader,
        username=username,
        password=password,
        session_user=session_user,
    )

    class _IPost:
        def __init__(self, post: "instaloader.Post", ig_loader: "instaloader.Instaloader"):
            self._post = post
            self._loader = ig_loader

        @property
        def shortcode(self) -> str:
            return self._post.shortcode

        @property
        def caption(self) -> str | None:
            return self._post.caption

        def download_image(self, target: Path) -> Path:
            target.parent.mkdir(parents=True, exist_ok=True)
            self._loader.download_pic(self._post.url, target.stem, target.parent)
            # instaloader may add an extension; normalize to .jpg path we store in JSONL.
            if target.exists():
                return target
            for ext in (".jpg", ".jpeg", ".png", ".webp"):
                candidate = target.with_suffix(ext)
                if candidate.exists():
                    return candidate
            raise FileNotFoundError(f"image not saved for {self.shortcode} under {target.parent}")

    try:
        hashtag_obj = instaloader.Hashtag.from_name(loader.context, hashtag)
    except instaloader.exceptions.ConnectionException as exc:
        if "login_required" in str(exc).lower():
            raise SystemExit(
                "Instagram returned login_required. Use --username/--password or a saved session.\n"
                "See README section 'Instagram scraping (login required)'."
            ) from exc
        raise

    for post in hashtag_obj.get_posts():
        if post.is_video:
            continue
        yield _IPost(post, loader)


# ---------- CLI --------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scrape Persian Instagram posts by hashtag.")
    parser.add_argument("--hashtag", required=True, help="Hashtag without the leading #.")
    parser.add_argument("--max-count", type=int, default=200)
    parser.add_argument("--delay", type=float, default=DEFAULT_REQUEST_DELAY)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory where <hashtag>.jsonl and <hashtag>/ image folder live.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--username",
        default=None,
        help="Instagram username (or set INSTAGRAM_USERNAME).",
    )
    parser.add_argument(
        "--password",
        default=None,
        help="Instagram password (or set INSTAGRAM_PASSWORD). Prefer saved session instead.",
    )
    parser.add_argument(
        "--session-user",
        default=None,
        help="Load instaloader session for this user (run: instaloader --login USER).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    jsonl_path = args.out_dir / f"{args.hashtag}.jsonl"
    image_dir = args.out_dir / args.hashtag
    posts = _iter_instaloader_posts(
        args.hashtag,
        username=args.username,
        password=args.password,
        session_user=args.session_user,
    )
    n = _persist(
        posts,
        jsonl_path,
        image_dir,
        delay=args.delay,
        max_count=args.max_count,
    )
    log.info("wrote %d new posts to %s (already-seen ones skipped)", n, jsonl_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
