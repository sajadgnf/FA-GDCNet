"""Collect hashtag posts via a real browser (Firefox cookies), then import via embed pages.

Uses Playwright to scroll Instagram hashtag explore pages like a normal user
(VPN + Firefox login apply). Post images/captions are downloaded through the
embed endpoint — no instaloader GraphQL / i.instagram.com API.

One-time setup::

    pip install playwright
    playwright install firefox

Usage::

    python tasks.py collect-hashtags --hashtags-file datasets/raw/hashtags.txt \\
        --max-count 30 --require-face --headed
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from .scrape import DEFAULT_RAW_DIR, _existing_shortcodes, _load_hashtags
from .scrape_embed import PostRef, import_shortcodes

log = logging.getLogger(__name__)


def _playwright_expires(raw) -> int:
    """Playwright accepts only -1 (session) or a positive Unix timestamp in seconds."""
    if raw is None:
        return -1
    try:
        exp = int(float(raw))
    except (TypeError, ValueError):
        return -1
    if exp <= 0:
        return -1
    # Firefox / browser_cookie3 often stores expiry in milliseconds.
    if exp > 10_000_000_000:
        exp //= 1000
    return exp if exp > 0 else -1


def _to_playwright_cookie(cookie) -> dict:
    same_site = getattr(cookie, "same_site", 0)
    if same_site == 1:
        ss = "Lax"
    elif same_site == 2:
        ss = "Strict"
    else:
        ss = "None"
    secure = bool(cookie.secure)
    if ss == "None" and not secure:
        ss = "Lax"
    return {
        "name": cookie.name,
        "value": cookie.value,
        "domain": cookie.domain,
        "path": cookie.path or "/",
        "expires": _playwright_expires(cookie.expires),
        "httpOnly": bool(getattr(cookie, "_rest", {}).get("HttpOnly", False)),
        "secure": secure,
        "sameSite": ss,
    }


def _playwright_cookies(browser: str = "firefox") -> list[dict]:
    import browser_cookie3

    loaders = {
        "firefox": browser_cookie3.firefox,
        "chrome": browser_cookie3.chrome,
        "edge": browser_cookie3.edge,
        "chromium": browser_cookie3.chromium,
    }
    loader = loaders.get(browser.lower())
    if loader is None:
        raise SystemExit(f"unsupported browser for cookies: {browser!r}")

    out: list[dict] = []
    seen_keys: set[tuple[str, str, str]] = set()
    for cookie in loader(domain_name=".instagram.com"):
        pw = _to_playwright_cookie(cookie)
        key = (pw["name"], pw["domain"], pw["path"])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        out.append(pw)
    if not out:
        raise SystemExit(
            f"No instagram.com cookies found in {browser!r}. "
            "Log in to Instagram in that browser first."
        )
    return out


def _goto_with_retry(
    page,
    url: str,
    *,
    page_timeout_ms: int = 120_000,
    attempts: int = 3,
) -> None:
    """Navigate with retries; Instagram often needs commit, not full domcontentloaded."""
    last_exc: Exception | None = None
    for attempt in range(attempts):
        for wait_until in ("commit", "domcontentloaded"):
            try:
                page.goto(url, wait_until=wait_until, timeout=page_timeout_ms)
                return
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                log.debug("goto %s wait=%s attempt %d failed: %s", url, wait_until, attempt + 1, exc)
        if attempt < attempts - 1:
            time.sleep(3 + attempt * 2)
    raise RuntimeError(f"navigation failed for {url}: {last_exc}") from last_exc


def _scroll_hashtag_page(
    page,
    tag: str,
    *,
    scrolls: int = 6,
    scroll_pause: float = 2.5,
    page_timeout_ms: int = 120_000,
) -> list[PostRef]:
    tag = tag.lstrip("#").strip().lower()
    url = f"https://www.instagram.com/explore/tags/{tag}/"
    found: list[PostRef] = []
    seen: set[str] = set()

    log.info("opening #%s (%s)", tag, url)
    _goto_with_retry(page, url, page_timeout_ms=page_timeout_ms)
    time.sleep(scroll_pause)

    for scroll in range(scrolls):
        refs = page.evaluate(
            """() => {
                const out = [];
                for (const a of document.querySelectorAll('a[href*="/p/"], a[href*="/reel/"]')) {
                    const m = a.href.match(/\\/(p|reel)\\/([A-Za-z0-9_-]+)/);
                    if (m) out.push({ kind: m[1], shortcode: m[2] });
                }
                return out;
            }"""
        )
        for item in refs:
            sc = str(item.get("shortcode", ""))
            kind = str(item.get("kind", "p"))
            if sc and sc not in seen:
                seen.add(sc)
                found.append(PostRef(shortcode=sc, kind=kind))
        log.info("#%s scroll %d/%d: %d links", tag, scroll + 1, scrolls, len(found))
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(scroll_pause)

    return found


def collect_shortcodes_from_hashtag(
    tag: str,
    *,
    browser: str = "firefox",
    headed: bool = False,
    scrolls: int = 6,
    scroll_pause: float = 2.5,
    page_timeout_ms: int = 60000,
) -> list[str]:
    """Scroll a hashtag explore page and return post shortcodes from the DOM."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise SystemExit(
            "Playwright is required for browser collection.\n"
            "Install with: pip install playwright && playwright install firefox"
        ) from exc

    with sync_playwright() as pw:
        launch = pw.firefox.launch(headless=not headed)
        context = launch.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            locale="fa-IR",
        )
        context.add_cookies(_playwright_cookies(browser))
        page = context.new_page()
        found = _scroll_hashtag_page(
            page,
            tag,
            scrolls=scrolls,
            scroll_pause=scroll_pause,
            page_timeout_ms=page_timeout_ms,
        )
        context.close()
        launch.close()

    return [ref.shortcode for ref in found]


def collect_hashtags_from_file(
    hashtags_file: Path,
    *,
    pool_name: str = "hashtags",
    out_dir: Path = DEFAULT_RAW_DIR,
    max_count: int = 30,
    require_face: bool = False,
    min_face_size: int = 40,
    delay: float = 2.0,
    timeout: float = 30.0,
    browser: str = "firefox",
    headed: bool = False,
    scrolls: int = 6,
    scroll_pause: float = 2.5,
    page_timeout_ms: int = 120_000,
    sarcasm_candidates: bool = False,
) -> int:
    tags = _load_hashtags(hashtags_file)
    if not tags:
        raise SystemExit(f"No hashtags in {hashtags_file}")

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise SystemExit(
            "Playwright is required for browser collection.\n"
            "Install with: pip install playwright && playwright install firefox"
        ) from exc

    jsonl_path = out_dir / f"{pool_name}.jsonl"
    blocked = _existing_shortcodes(jsonl_path)
    candidates: list[PostRef] = []
    seen_candidates: set[str] = set()

    with sync_playwright() as pw:
        launch = pw.firefox.launch(headless=not headed)
        context = launch.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            locale="fa-IR",
        )
        context.add_cookies(_playwright_cookies(browser))
        page = context.new_page()
        try:
            log.info("warming up instagram.com …")
            _goto_with_retry(page, "https://www.instagram.com/", page_timeout_ms=page_timeout_ms)
            time.sleep(scroll_pause)
        except Exception as exc:  # noqa: BLE001
            log.warning("instagram warm-up failed (%s) — continuing anyway", exc)

        consecutive_failures = 0
        for i, tag in enumerate(tags):
            if len(candidates) >= max_count * 3:
                break
            if consecutive_failures >= 5:
                log.error(
                    "Stopped after %d consecutive hashtag timeouts. "
                    "Check VPN, log into Instagram in %s, wait a few minutes, retry.",
                    consecutive_failures,
                    browser,
                )
                break
            if i:
                time.sleep(scroll_pause)
            try:
                found = _scroll_hashtag_page(
                    page,
                    tag,
                    scrolls=scrolls,
                    scroll_pause=scroll_pause,
                    page_timeout_ms=page_timeout_ms,
                )
            except Exception as exc:  # noqa: BLE001
                consecutive_failures += 1
                log.warning("skip hashtag %s: browser collection failed (%s)", tag, exc)
                time.sleep(scroll_pause * 2)
                continue
            consecutive_failures = 0
            for ref in found:
                if ref.shortcode in blocked or ref.shortcode in seen_candidates:
                    continue
                seen_candidates.add(ref.shortcode)
                candidates.append(ref)
            log.info("#%s yielded %d new candidates (%d total)", tag, len(found), len(candidates))

        if not candidates:
            context.close()
            launch.close()
            raise SystemExit(
                "Browser found no new post links. If every hashtag timed out, check VPN / "
                f"Firefox login and retry with --page-timeout 180. Otherwise try --headed or "
                "increase --scrolls."
            )

        written = import_shortcodes(
            candidates,
            pool_name=pool_name,
            out_dir=out_dir,
            require_face=require_face,
            min_face_size=min_face_size,
            delay=delay,
            timeout=timeout,
            max_count=max_count,
            page=page,
            browser_context=context,
            sarcasm_candidates=sarcasm_candidates,
        )

        context.close()
        launch.close()

    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Collect hashtag posts with Playwright + embed download (no API).",
    )
    parser.add_argument("--hashtags-file", type=Path, required=True)
    parser.add_argument("--pool-name", default="hashtags")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--max-count", type=int, default=30)
    parser.add_argument("--require-face", action="store_true")
    parser.add_argument("--min-face-size", type=int, default=40)
    parser.add_argument("--delay", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--browser", default="firefox")
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Show the browser window (useful if cookies/login fail headless).",
    )
    parser.add_argument("--scrolls", type=int, default=6)
    parser.add_argument("--scroll-pause", type=float, default=2.5)
    parser.add_argument(
        "--page-timeout",
        type=int,
        default=120,
        help="Navigation timeout per page in seconds (default 120).",
    )
    parser.add_argument(
        "--sarcasm-candidates",
        action="store_true",
        help="Skip captions with no irony/sarcasm text cues (plain selfies).",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    collect_hashtags_from_file(
        args.hashtags_file,
        pool_name=args.pool_name,
        out_dir=args.out_dir,
        max_count=args.max_count,
        require_face=args.require_face,
        min_face_size=args.min_face_size,
        delay=args.delay,
        timeout=args.timeout,
        browser=args.browser,
        headed=args.headed,
        scrolls=args.scrolls,
        scroll_pause=args.scroll_pause,
        page_timeout_ms=args.page_timeout * 1000,
        sarcasm_candidates=args.sarcasm_candidates,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
