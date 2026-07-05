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
import urllib.parse
from pathlib import Path

from collections.abc import Callable

from .scrape import DEFAULT_RAW_DIR, HashtagSpec, _existing_shortcodes, _load_hashtag_specs
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
    blocked: set[str] | None = None,
    on_batch: Callable[[list[PostRef], str], None] | None = None,
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
        batch_new: list[PostRef] = []
        for item in refs:
            sc = str(item.get("shortcode", ""))
            kind = str(item.get("kind", "p"))
            if sc and sc not in seen:
                seen.add(sc)
                ref = PostRef(shortcode=sc, kind=kind)
                found.append(ref)
                if blocked is None or sc not in blocked:
                    batch_new.append(ref)
        log.info("#%s scroll %d/%d: %d links (%d new this batch)", tag, scroll + 1, scrolls, len(found), len(batch_new))
        if on_batch and batch_new:
            on_batch(batch_new, url)
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(scroll_pause)

    return found


def _scroll_hashtag_and_import(
    page,
    tag: str,
    *,
    scrolls: int = 6,
    scroll_pause: float = 2.5,
    page_timeout_ms: int = 120_000,
    blocked: set[str],
    seen_candidates: set[str],
    pending_refs: list[PostRef],
    import_fn: Callable[[list[PostRef], str], int],
) -> tuple[list[PostRef], int]:
    """Scroll a hashtag page and import each batch while posts are still in the DOM."""
    imported = 0

    def on_batch(batch: list[PostRef], tag_url: str) -> None:
        nonlocal imported
        fresh = [
            ref
            for ref in batch
            if ref.shortcode not in blocked
            and ref.shortcode not in seen_candidates
        ]
        if not fresh:
            return
        for ref in fresh:
            seen_candidates.add(ref.shortcode)
            pending_refs.append(ref)
        n = import_fn(fresh, tag_url)
        imported += n

    found = _scroll_hashtag_page(
        page,
        tag,
        scrolls=scrolls,
        scroll_pause=scroll_pause,
        page_timeout_ms=page_timeout_ms,
        blocked=blocked,
        on_batch=on_batch,
    )
    return found, imported


def _search_hashtags_containing(
    page,
    query: str,
    *,
    max_tags: int = 15,
    page_timeout_ms: int = 120_000,
    scroll_pause: float = 2.5,
) -> list[str]:
    """Find Instagram tags whose name includes ``query`` (partial match)."""
    term = query.lstrip("#").strip()
    if not term:
        return []
    url = (
        "https://www.instagram.com/explore/search/keyword/?q="
        + urllib.parse.quote(term)
    )
    log.info("searching tags containing %r …", term)
    _goto_with_retry(page, url, page_timeout_ms=page_timeout_ms)
    time.sleep(scroll_pause)
    try:
        page.locator('a[href*="/explore/tags/"]').first.wait_for(
            state="attached",
            timeout=min(15_000, page_timeout_ms // 2),
        )
    except Exception:
        log.debug("no tag links visible yet for search %r", term)
    for _ in range(3):
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(0.8)
    needle = term.lower()
    tags: list[str] = page.evaluate(
        """(needle) => {
            const out = [];
            for (const a of document.querySelectorAll('a[href*="/explore/tags/"]')) {
                const m = a.href.match(/\\/explore\\/tags\\/([^/?#]+)/);
                if (!m) continue;
                const tag = decodeURIComponent(m[1]);
                if (tag.toLowerCase().includes(needle)) out.push(tag);
            }
            return [...new Set(out)];
        }""",
        needle,
    )
    if len(tags) > max_tags:
        tags = tags[:max_tags]
    return tags


def _resolve_hashtags(
    page,
    specs: list[HashtagSpec],
    *,
    max_search_tags: int = 15,
    page_timeout_ms: int = 120_000,
    scroll_pause: float = 2.5,
) -> list[str]:
    """Expand search specs to concrete tag names; keep exact specs as-is."""
    resolved: list[str] = []
    seen: set[str] = set()
    for spec in specs:
        if spec.search:
            found = _search_hashtags_containing(
                page,
                spec.term,
                max_tags=max_search_tags,
                page_timeout_ms=page_timeout_ms,
                scroll_pause=scroll_pause,
            )
            if not found:
                log.warning("no tags containing %r on Instagram — trying exact #%s", spec.term, spec.term)
                found = [spec.term]
            else:
                preview = ", ".join(f"#{t}" for t in found[:6])
                if len(found) > 6:
                    preview += f", … (+{len(found) - 6} more)"
                log.info("search %r → %d tags: %s", spec.term, len(found), preview)
        else:
            found = [spec.term]
        for tag in found:
            key = tag.lower()
            if key not in seen:
                seen.add(key)
                resolved.append(tag)
    return resolved


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
    hashtag_contains: bool = False,
    max_search_tags: int = 15,
) -> int:
    specs = _load_hashtag_specs(hashtags_file, search_all=hashtag_contains)
    if not specs:
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
    log.info(
        "%d post IDs already in pool/ignore list — only unseen hashtag links will be collected",
        len(blocked),
    )
    seen_candidates: set[str] = set()
    pending_refs: list[PostRef] = []
    written_total = 0

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

        tags = _resolve_hashtags(
            page,
            specs,
            max_search_tags=max_search_tags,
            page_timeout_ms=page_timeout_ms,
            scroll_pause=scroll_pause,
        )
        if not tags:
            context.close()
            launch.close()
            raise SystemExit(f"No hashtags resolved from {hashtags_file}")

        consecutive_failures = 0

        def import_batch(refs: list[PostRef], tag_url: str) -> int:
            nonlocal written_total
            if written_total >= max_count:
                return 0
            return import_shortcodes(
                refs,
                pool_name=pool_name,
                out_dir=out_dir,
                require_face=require_face,
                min_face_size=min_face_size,
                delay=delay,
                timeout=timeout,
                max_count=max_count - written_total,
                page=page,
                browser_context=context,
                sarcasm_candidates=sarcasm_candidates,
                prefer_modal=True,
                hashtag_url=tag_url,
            )

        for i, tag in enumerate(tags):
            if written_total >= max_count:
                break
            if len(seen_candidates) >= max_count * 3:
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
                found, written = _scroll_hashtag_and_import(
                    page,
                    tag,
                    scrolls=scrolls,
                    scroll_pause=scroll_pause,
                    page_timeout_ms=page_timeout_ms,
                    blocked=blocked,
                    seen_candidates=seen_candidates,
                    pending_refs=pending_refs,
                    import_fn=import_batch,
                )
            except Exception as exc:  # noqa: BLE001
                consecutive_failures += 1
                log.warning("skip hashtag %s: browser collection failed (%s)", tag, exc)
                time.sleep(scroll_pause * 2)
                continue
            consecutive_failures = 0
            written_total += written
            log.info(
                "#%s done: %d links seen, %d imported this tag (%d total)",
                tag,
                len(found),
                written,
                written_total,
            )
            if written_total >= max_count:
                break

        if written_total == 0 and pending_refs:
            pending_path = out_dir / f"{pool_name}_pending_links.txt"
            with pending_path.open("a", encoding="utf-8") as pf:
                for ref in pending_refs:
                    pf.write(f"https://www.instagram.com/{ref.kind}/{ref.shortcode}/\n")
            log.info(
                "Saved %d post links to %s — retry later with import-links",
                len(pending_refs),
                pending_path,
            )

        if written_total == 0 and not pending_refs:
            context.close()
            launch.close()
            raise SystemExit(
                "Browser found no new post links. If every hashtag timed out, check VPN / "
                f"Firefox login and retry with --page-timeout 180. Otherwise try --headed or "
                "increase --scrolls."
            )

        context.close()
        launch.close()

    return written_total


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
    parser.add_argument("--delay", type=float, default=8.0)
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
    parser.add_argument(
        "--hashtag-contains",
        action="store_true",
        help="Treat every line as a search term (tags whose name includes it), not exact tag.",
    )
    parser.add_argument(
        "--max-search-tags",
        type=int,
        default=15,
        help="Max tags to collect per search term (default 15).",
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
        hashtag_contains=args.hashtag_contains,
        max_search_tags=args.max_search_tags,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
