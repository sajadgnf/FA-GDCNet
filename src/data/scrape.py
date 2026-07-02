"""Idempotent Instagram scraper that respects rate limits.

Persists `(post_id, caption, image_path)` rows to `datasets/raw/<pool>.jsonl`
with shortcode-based deduplication. The image bytes are stored locally only —
images are never re-uploaded or redistributed (see README "نکته حقوقی").

Preferred sources for this project are **personal profiles** or the logged-in
user's **following feed** — real daily photos with Persian captions. Hashtag
scraping is kept for compatibility but tends to return landscapes, ads, and
generic stock-style posts (few faces, shallow captions).

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

from .preprocess import is_persian_enough, is_spam_caption, preprocess_caption

try:
    from .face_filter import has_face as _image_has_face
except ImportError:  # opencv optional in minimal installs
    _image_has_face = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

DEFAULT_RAW_DIR = Path("datasets") / "raw"
DEFAULT_REQUEST_DELAY = 4.0  # seconds; safe default for unauthenticated instaloader.
IGNORED_IDS_FILE = DEFAULT_RAW_DIR / "ignored_post_ids.txt"


def _load_ignored_shortcodes() -> set[str]:
    if not IGNORED_IDS_FILE.exists():
        return set()
    out: set[str] = set()
    with IGNORED_IDS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                out.add(line)
    return out


def _remember_ignored(shortcode: str) -> None:
    known = _load_ignored_shortcodes()
    if shortcode in known:
        return
    IGNORED_IDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with IGNORED_IDS_FILE.open("a", encoding="utf-8") as f:
        f.write(shortcode + "\n")


class ScrapedPost(Protocol):
    """The minimum surface `_persist` needs from any scraper backend."""

    @property
    def shortcode(self) -> str: ...

    @property
    def caption(self) -> str | None: ...

    def download_image(self, target: Path) -> Path: ...


def _existing_shortcodes(jsonl_path: Path) -> set[str]:
    seen = _load_ignored_shortcodes()
    if jsonl_path.exists():
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
    require_face: bool = False,
    min_face_size: int = 40,
) -> int:
    seen = _existing_shortcodes(jsonl_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    if require_face and _image_has_face is None:
        raise SystemExit(
            "Face filter requires opencv. Install with: pip install opencv-python-headless"
        )
    written = 0
    skipped_no_face = 0
    skipped_spam = 0
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
            if is_spam_caption(caption):
                skipped_spam += 1
                _remember_ignored(sc)
                log.debug("skip spam/bait caption %s", sc)
                continue
            try:
                image_path = post.download_image(image_dir / f"{sc}.jpg")
            except Exception as exc:  # noqa: BLE001
                log.warning("skip %s: image download failed (%s)", sc, exc)
                continue
            if require_face and not _image_has_face(image_path, min_size=min_face_size):
                skipped_no_face += 1
                _remember_ignored(sc)
                seen.add(sc)
                log.debug("skip %s: no face detected (added to ignore list)", sc)
                with suppress(OSError):
                    image_path.unlink()
                continue
            row = {
                "post_id": sc,
                "caption": caption,
                "image_path": str(image_path),
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            seen.add(sc)
            written += 1
            time.sleep(delay)
    if require_face and skipped_no_face:
        log.info(
            "skipped %d posts with no detected face (IDs saved to %s)",
            skipped_no_face,
            IGNORED_IDS_FILE,
        )
    if skipped_spam:
        log.info("skipped %d spam/bait captions (saved to %s)", skipped_spam, IGNORED_IDS_FILE)
    return written


# ---------- Instaloader backend ----------------------------------------------


def _build_instaloader(
    *,
    max_connection_attempts: int = 2,
    request_timeout: float = 45.0,
):
    import instaloader

    return instaloader.Instaloader(
        download_pictures=False,
        download_videos=False,
        download_comments=False,
        save_metadata=False,
        compress_json=False,
        post_metadata_txt_pattern="",
        max_connection_attempts=max_connection_attempts,
        request_timeout=request_timeout,
    )


def _is_ssl_or_network_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "ssl",
            "unexpected_eof",
            "connection aborted",
            "connection reset",
            "timed out",
            "max retries exceeded",
        )
    )


def _network_help_message() -> str:
    return (
        "Instagram API is unreachable from Python (SSL/timeout).\n"
        "Firefox can work while instaloader fails when VPN/proxy is browser-only.\n"
        "Try:\n"
        "  1) Enable a system-wide VPN (not only a browser extension)\n"
        "  2) Disable HTTPS scanning in antivirus / corporate proxy\n"
        "  3) Set HTTPS_PROXY to your VPN proxy, then retry\n"
        "  4) Refresh session: python tasks.py scrape-session --user YOUR_USER --browser firefox"
    )


def _verify_api_access(loader, session_user: str) -> None:
    """Fail fast when Instagram API cannot be reached (avoids hours of per-profile retries)."""
    import instaloader

    handle = session_user.strip().lstrip("@")
    if not handle:
        return
    try:
        _resolve_profile(loader, handle)
    except instaloader.exceptions.ProfileNotExistsException:
        return
    except instaloader.exceptions.ConnectionException as exc:
        if _is_ssl_or_network_error(exc):
            raise SystemExit(f"{_network_help_message()}\nDetails: {exc}") from exc
        raise


def _authenticate_instaloader(
    loader,
    *,
    username: str | None,
    password: str | None,
    session_user: str | None,
    session_file: str | None,
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

    if session_file:
        resolved_session_file = str(Path(session_file).expanduser())
    else:
        custom_dir = os.environ.get("INSTALOADER_SESSION_DIR")
        if custom_dir:
            resolved_session_file = str(Path(custom_dir) / f"session-{user}")
        else:
            resolved_session_file = get_default_session_filename(user)

    try:
        loader.load_session_from_file(user, filename=resolved_session_file)
        log.info("loaded Instagram session for %s from %s", user, resolved_session_file)
        return user
    except (FileNotFoundError, instaloader.exceptions.ConnectionException, OSError) as exc:
        log.debug("no saved session for %s: %s", user, exc)

    if not pwd:
        raise SystemExit(
            f"No saved session for {user!r} and no password given.\n"
            "Either run once:\n"
            f"  instaloader --login {user}\n"
            "Or export/import cookies and then reuse the produced session file.\n"
            "Then run this scraper with:\n"
            f"  --session-user {user} --session-file <path_to_session_file>\n"
            "or pass --password / set INSTAGRAM_PASSWORD.\n"
            f"Default session path on this machine: {resolved_session_file}"
        )

    log.info("logging in to Instagram as %s ...", user)
    try:
        loader.login(user, pwd)
    except instaloader.exceptions.ConnectionException as exc:
        msg = str(exc)
        if "unexpected null login result" in msg.lower():
            raise SystemExit(
                "Instagram rejected password login with 'Unexpected null login result'.\n"
                "Use a pre-saved session file instead of direct password login:\n"
                f"  python tasks.py scrape --hashtag <tag> --max-count <n> --session-user {user} "
                "--session-file <path_to_session_file>\n"
                "Tip: create the session via an interactive instaloader/browser-cookie flow first."
            ) from exc
        raise
    loader.save_session_to_file(resolved_session_file)
    log.info("session saved to %s (reuse with --session-user %s)", resolved_session_file, user)
    return user


class _InstaloaderPost:
    """Thin adapter from instaloader.Post to ScrapedPost."""

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
        from datetime import datetime, timezone

        target.parent.mkdir(parents=True, exist_ok=True)
        url = self._post.url
        stem = target.with_suffix("")
        mtime = getattr(self._post, "date_utc", None) or datetime.now(timezone.utc)
        # Instaloader 4.x: download_pic(filename_without_ext, url, mtime)
        self._loader.download_pic(str(stem), url, mtime)
        if target.exists():
            return target
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            candidate = stem.with_suffix(ext)
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"image not saved for {self.shortcode} under {target.parent}")


def _login_loader(
    *,
    username: str | None = None,
    password: str | None = None,
    session_user: str | None = None,
    session_file: str | None = None,
    max_connection_attempts: int = 2,
    request_timeout: float = 45.0,
    verify_api: bool = True,
):
    loader = _build_instaloader(
        max_connection_attempts=max_connection_attempts,
        request_timeout=request_timeout,
    )
    user = _authenticate_instaloader(
        loader,
        username=username,
        password=password,
        session_user=session_user,
        session_file=session_file,
    )
    if verify_api:
        _verify_api_access(loader, user)
    return loader


def _resolve_profile(loader, handle: str):
    """Load a Profile; prefer web_profile_info (search-based lookup often fails)."""
    import instaloader

    try:
        data = loader.context.get_iphone_json(
            "api/v1/users/web_profile_info/",
            {"username": handle},
        )
        user = data.get("data", {}).get("user")
        if user and user.get("username", "").lower() == handle.lower():
            return instaloader.Profile.from_iphone_struct(loader.context, user)
    except instaloader.exceptions.ConnectionException as exc:
        log.debug("web_profile_info failed for %s: %s", handle, exc)

    try:
        return instaloader.Profile.from_username(loader.context, handle)
    except instaloader.exceptions.ProfileNotExistsException:
        pass
    raise instaloader.exceptions.ProfileNotExistsException(
        f"Profile {handle} does not exist."
    )


def _iter_hashtag_posts(
    hashtag: str,
    loader,
) -> Iterator[ScrapedPost]:
    """Yield photo posts from a public hashtag feed."""
    import instaloader

    tag = hashtag.lstrip("#").strip().lower()
    if not tag:
        return

    # Avoid Hashtag.from_name() — it always hits i.instagram.com/web_info first.
    # get_posts_resumable() uses www.instagram.com GraphQL instead.
    hashtag_obj = instaloader.Hashtag(loader.context, {"name": tag})

    try:
        post_iter = hashtag_obj.get_posts_resumable()
    except instaloader.exceptions.QueryReturnedNotFoundException:
        log.warning("skip hashtag %s: not found", tag)
        return
    except instaloader.exceptions.ConnectionException as exc:
        if "login_required" in str(exc).lower():
            raise SystemExit(
                "Instagram returned login_required. Use --username/--password or a saved session.\n"
                "See README section 'Instagram scraping (login required)'."
            ) from exc
        log.warning("skip hashtag %s: connection error (%s)", tag, exc)
        return

    log.info("scraping hashtag #%s (www graphql)", tag)
    for post in post_iter:
        if post.is_video:
            continue
        yield _InstaloaderPost(post, loader)


def _iter_instaloader_posts(
    hashtag: str,
    *,
    username: str | None = None,
    password: str | None = None,
    session_user: str | None = None,
    session_file: str | None = None,
    max_connection_attempts: int = 2,
    request_timeout: float = 45.0,
) -> Iterator[ScrapedPost]:
    loader = _login_loader(
        username=username,
        password=password,
        session_user=session_user,
        session_file=session_file,
        max_connection_attempts=max_connection_attempts,
        request_timeout=request_timeout,
        verify_api=False,
    )
    yield from _iter_hashtag_posts(hashtag, loader)


class ProfileNetworkError(ConnectionError):
    """Raised when Instagram API cannot be reached for a profile lookup."""

    def __init__(self, handle: str, cause: BaseException):
        self.handle = handle
        super().__init__(str(cause))
        self.__cause__ = cause


def _iter_profile_posts(
    profile_name: str,
    loader,
    *,
    posts_per_profile: int | None = None,
) -> Iterator[ScrapedPost]:
    """Yield recent photo posts from one public (or followed) Instagram account."""
    import instaloader

    handle = profile_name.lstrip("@").strip()
    try:
        profile = _resolve_profile(loader, handle)
    except instaloader.exceptions.ProfileNotExistsException:
        log.warning("skip profile %s: not found", handle)
        return
    except instaloader.exceptions.ConnectionException as exc:
        if "login_required" in str(exc).lower():
            raise SystemExit(
                "Instagram returned login_required. Use --username/--password or a saved session."
            ) from exc
        if _is_ssl_or_network_error(exc):
            raise ProfileNetworkError(handle, exc) from exc
        log.warning("skip profile %s: connection error (%s)", handle, exc)
        return

    try:
        private = profile.is_private and not profile.followed_by_viewer
    except instaloader.exceptions.ConnectionException as exc:
        log.warning("skip profile %s: could not read privacy (%s)", handle, exc)
        return
    if private:
        log.warning("skip profile %s: private and not followed by session user", handle)
        return

    log.info("scraping @%s (%d posts)", handle, profile.mediacount)
    if profile.mediacount == 0:
        log.warning("skip profile %s: no posts", handle)
        return
    count = 0
    for post in profile.get_posts():
        if post.is_video:
            continue
        yield _InstaloaderPost(post, loader)
        count += 1
        if posts_per_profile is not None and count >= posts_per_profile:
            break


def _profiles_file_help() -> str:
    return (
        "Create a text file with one username per line, e.g. datasets/raw/accounts.txt:\n"
        "  friend_username\n"
        "  another_account\n"
        "Then run:\n"
        "  python tasks.py scrape --profiles-file datasets/raw/accounts.txt "
        "--session-user YOUR_USER --max-count 200"
    )


def _iter_following_posts(
    *,
    username: str | None = None,
    password: str | None = None,
    session_user: str | None = None,
    session_file: str | None = None,
    max_profiles: int = 30,
    posts_per_profile: int = 10,
) -> Iterator[ScrapedPost]:
    """Yield photo posts from accounts the logged-in user follows (daily-life feed)."""
    import instaloader
    from instaloader.exceptions import QueryReturnedBadRequestException

    loader = _login_loader(
        username=username,
        password=password,
        session_user=session_user,
        session_file=session_file,
    )
    viewer = session_user or username or os.environ.get("INSTAGRAM_USERNAME") or ""
    viewer = viewer.strip()
    if not viewer:
        raise SystemExit("--following requires --session-user (the account whose followees to scan).")

    profile = _resolve_profile(loader, viewer)
    try:
        followees_iter = profile.get_followees()
    except QueryReturnedBadRequestException as exc:
        raise SystemExit(
            "Instagram blocked the following-list API (400 invalid request).\n"
            "This is common in 2025–2026. Use a manual account list instead:\n\n"
            + _profiles_file_help()
        ) from exc

    if profile.followees == 0:
        raise SystemExit(
            f"@{viewer} has 0 followees visible to instaloader (empty list or session cannot read it).\n"
            "Log in on instagram.com in your browser, confirm you follow people, refresh the session:\n"
            "  python tasks.py scrape-session --user "
            f"{viewer}\n"
            "Or scrape specific accounts you care about:\n\n"
            + _profiles_file_help()
        )

    seen_profiles = 0
    for followee in followees_iter:
        if seen_profiles >= max_profiles:
            break
        seen_profiles += 1
        handle = followee.username
        if followee.is_private and not followee.followed_by_viewer:
            log.debug("skip private followee %s", handle)
            continue
        log.info("scraping followee @%s", handle)
        count = 0
        try:
            for post in followee.get_posts():
                if post.is_video:
                    continue
                yield _InstaloaderPost(post, loader)
                count += 1
                if count >= posts_per_profile:
                    break
        except instaloader.exceptions.ConnectionException as exc:
            log.warning("skip followee %s: %s", handle, exc)


def _iter_followers_of_posts(
    seed_profile: str,
    *,
    username: str | None = None,
    password: str | None = None,
    session_user: str | None = None,
    session_file: str | None = None,
    max_profiles: int = 30,
    posts_per_profile: int = 10,
) -> Iterator[ScrapedPost]:
    """Yield photo posts from accounts that follow ``seed_profile``."""
    import instaloader
    from instaloader.exceptions import QueryReturnedBadRequestException

    loader = _login_loader(
        username=username,
        password=password,
        session_user=session_user,
        session_file=session_file,
    )
    handle = seed_profile.lstrip("@").strip()
    viewer = (session_user or username or os.environ.get("INSTAGRAM_USERNAME") or "").strip()

    try:
        profile = _resolve_profile(loader, handle)
    except instaloader.exceptions.ProfileNotExistsException:
        raise SystemExit(f"Instagram profile @{handle} does not exist.") from None

    if profile.is_private and not profile.followed_by_viewer:
        raise SystemExit(
            f"@{handle} is private and your session user"
            + (f" (@{viewer})" if viewer else "")
            + " does not follow them — posts and follower lists are hidden.\n"
            "1) Follow @{handle} on instagram.com\n"
            "2) Refresh session: python tasks.py scrape-session --user YOUR_USER\n"
            "3) Retry this scrape command"
        )

    log.info("@%s has %d followers; sampling up to %d accounts", handle, profile.followers, max_profiles)
    try:
        followers_iter = profile.get_followers()
    except QueryReturnedBadRequestException as exc:
        raise SystemExit(
            f"Instagram blocked the follower-list API for @{handle}.\n"
            "Try scraping the account directly instead:\n"
            f"  python tasks.py scrape --profile {handle} --session-user YOUR_USER"
        ) from exc

    seen_profiles = 0
    for follower in followers_iter:
        if seen_profiles >= max_profiles:
            break
        seen_profiles += 1
        fan = follower.username
        log.info("scraping follower @%s of @%s", fan, handle)
        yield from _iter_profile_posts(
            fan,
            loader,
            posts_per_profile=posts_per_profile,
        )


def _load_profile_names(path: Path) -> list[str]:
    names: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            names.append(line.lstrip("@"))
    return names


def _load_hashtags(path: Path) -> list[str]:
    tags: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            tags.append(raw.lstrip("#"))
    return tags


def _iter_multi_hashtag_posts(
    hashtags: list[str],
    *,
    username: str | None = None,
    password: str | None = None,
    session_user: str | None = None,
    session_file: str | None = None,
    hashtag_pause: float = 10.0,
    max_connection_attempts: int = 2,
    request_timeout: float = 45.0,
) -> Iterator[ScrapedPost]:
    """Yield photo posts from public hashtag feeds (any account worldwide)."""
    loader = _login_loader(
        username=username,
        password=password,
        session_user=session_user,
        session_file=session_file,
        max_connection_attempts=max_connection_attempts,
        request_timeout=request_timeout,
        verify_api=False,
    )
    for i, tag in enumerate(hashtags):
        if i:
            time.sleep(hashtag_pause)
        yield from _iter_hashtag_posts(tag, loader)


def _iter_multi_profile_posts(
    profile_names: list[str],
    *,
    username: str | None = None,
    password: str | None = None,
    session_user: str | None = None,
    session_file: str | None = None,
    posts_per_profile: int = 15,
    profile_pause: float = 6.0,
    max_connection_attempts: int = 2,
    request_timeout: float = 45.0,
    max_profile_failures: int = 3,
) -> Iterator[ScrapedPost]:
    loader = _login_loader(
        username=username,
        password=password,
        session_user=session_user,
        session_file=session_file,
        max_connection_attempts=max_connection_attempts,
        request_timeout=request_timeout,
    )
    failures = 0
    for i, name in enumerate(profile_names):
        if i:
            time.sleep(profile_pause)
        try:
            yield from _iter_profile_posts(
                name,
                loader,
                posts_per_profile=posts_per_profile,
            )
            failures = 0
        except ProfileNetworkError as exc:
            failures += 1
            log.warning("skip profile %s: network error (%s)", exc.handle, exc)
            if failures >= max_profile_failures:
                raise SystemExit(
                    f"{_network_help_message()}\n"
                    f"Stopped after {failures} consecutive profile failures.\n"
                    f"Last error: {exc}"
                ) from exc

def _resolve_pool_name(args: argparse.Namespace) -> str:
    if args.pool_name:
        return args.pool_name
    if args.hashtag:
        return args.hashtag
    if getattr(args, "hashtags_file", None):
        return "hashtags"
    if args.following:
        return "following"
    if getattr(args, "followers_of", None):
        safe = args.followers_of.lstrip("@").replace(".", "_")
        return f"followers_of_{safe}"
    profiles = list(args.profile or [])
    if args.profiles_file:
        profiles.extend(_load_profile_names(Path(args.profiles_file)))
    if len(profiles) == 1:
        return f"profile_{profiles[0].lstrip('@')}"
    if profiles:
        return "profiles"
    raise SystemExit("Specify a source or omit all flags to default to --following.")


def _default_source(args: argparse.Namespace) -> None:
    if any((
        args.hashtag,
        getattr(args, "hashtags_file", None),
        args.profile,
        args.profiles_file,
        args.following,
        args.followers_of,
    )):
        return
    accounts = DEFAULT_RAW_DIR / "accounts.txt"
    if accounts.is_file() and _load_profile_names(accounts):
        args.profiles_file = accounts
        log.info("using default account list %s", accounts)
        return
    args.following = True


# ---------- CLI --------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Scrape Persian Instagram posts from personal profiles or your following feed "
        "(default). Hashtag mode is available but not recommended for daily-life multimodal data."
    )
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument(
        "--hashtag",
        help="Single hashtag without # — public posts from any account using that tag.",
    )
    source.add_argument(
        "--hashtags-file",
        type=Path,
        help="File with hashtags (one per line) — browse Instagram beyond your account list.",
    )
    source.add_argument(
        "--profile",
        action="append",
        metavar="USERNAME",
        help="Public personal account to scrape (repeatable). Best for daily selfies + captions.",
    )
    source.add_argument(
        "--profiles-file",
        type=Path,
        help="Text file with one @username per line (lines starting with # are ignored).",
    )
    source.add_argument(
        "--following",
        action="store_true",
        help="Scrape recent posts from accounts your session user follows.",
    )
    source.add_argument(
        "--followers-of",
        metavar="USERNAME",
        help="Scrape posts from accounts that follow USERNAME (seed account).",
    )
    parser.add_argument("--max-count", type=int, default=200)
    parser.add_argument(
        "--posts-per-profile",
        type=int,
        default=15,
        help="Cap posts taken from each profile (--profile / --following).",
    )
    parser.add_argument(
        "--max-profiles",
        type=int,
        default=30,
        help="How many profiles to visit with --following or --followers-of.",
    )
    parser.add_argument(
        "--pool-name",
        default=None,
        help="Output basename (default: hashtag, profile_<user>, profiles, or following).",
    )
    parser.add_argument("--delay", type=float, default=DEFAULT_REQUEST_DELAY)
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=45.0,
        help="Per-request timeout in seconds for Instagram API calls (default: 45).",
    )
    parser.add_argument(
        "--max-connection-attempts",
        type=int,
        default=2,
        help="Instaloader retries per API call (default: 2).",
    )
    parser.add_argument(
        "--require-face",
        action="store_true",
        help="Keep only images where OpenCV detects at least one frontal face.",
    )
    parser.add_argument(
        "--min-face-size",
        type=int,
        default=40,
        help="Minimum face box side in pixels for --require-face (default: 40).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory where <pool>.jsonl and <pool>/ image folder live.",
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
    parser.add_argument(
        "--session-file",
        default=None,
        help="Explicit path to a saved instaloader session file (optional).",
    )
    args = parser.parse_args(argv)
    _default_source(args)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    pool = _resolve_pool_name(args)
    jsonl_path = args.out_dir / f"{pool}.jsonl"
    image_dir = args.out_dir / pool

    auth = {
        "username": args.username,
        "password": args.password,
        "session_user": args.session_user,
        "session_file": args.session_file,
        "max_connection_attempts": args.max_connection_attempts,
        "request_timeout": args.request_timeout,
    }

    if args.hashtag:
        posts = _iter_instaloader_posts(args.hashtag, **auth)
    elif args.hashtags_file:
        tags = _load_hashtags(args.hashtags_file)
        if not tags:
            raise SystemExit(f"No hashtags in {args.hashtags_file}")
        posts = _iter_multi_hashtag_posts(tags, **auth)
    elif args.following:
        posts = _iter_following_posts(
            **auth,
            max_profiles=args.max_profiles,
            posts_per_profile=args.posts_per_profile,
        )
    elif args.followers_of:
        posts = _iter_followers_of_posts(
            args.followers_of,
            **auth,
            max_profiles=args.max_profiles,
            posts_per_profile=args.posts_per_profile,
        )
    else:
        profile_names = list(args.profile or [])
        if args.profiles_file:
            profile_names.extend(_load_profile_names(args.profiles_file))
        if not profile_names:
            raise SystemExit("No profiles given.")
        posts = _iter_multi_profile_posts(
            profile_names,
            **auth,
            posts_per_profile=args.posts_per_profile,
        )

    n = _persist(
        posts,
        jsonl_path,
        image_dir,
        delay=args.delay,
        max_count=args.max_count,
        require_face=args.require_face,
        min_face_size=args.min_face_size,
    )
    log.info("wrote %d new posts to %s (already-seen ones skipped)", n, jsonl_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
