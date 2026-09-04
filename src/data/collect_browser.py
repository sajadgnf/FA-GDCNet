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
import json
import logging
import re
import sys
import time
import urllib.parse
from pathlib import Path

from collections.abc import Callable

from .preprocess import is_persian_enough, preprocess_caption
from .sarcasm_candidates import is_political_caption, is_sarcasm_candidate_caption
from .scrape import (
    DEFAULT_RAW_DIR,
    HashtagSpec,
    _existing_shortcodes,
    _jsonl_shortcodes,
    _load_hashtag_specs,
)
from .scrape_embed import FetchStall, PostRef, import_shortcodes, load_shortcodes, normalize_post_refs

log = logging.getLogger(__name__)

_SHORTCODE_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{8,15}$")
_CODE_IN_TEXT_RE = re.compile(r'"(?:code|shortcode)"\s*:\s*"([A-Za-z0-9_-]{8,15})"')
_HARVEST_URL_HINTS = (
    "graphql",
    "api/v1",
    "api/graphql",
    "/tags/",
    "web_info",
    "polaris",
    "fbsearch",
)
MAX_RELATED_TAGS = 40
IMPORT_BATCH = 30
PENDING_RETRY_BATCH = 80


def _rotate_pending_file(path: Path, attempted: list[str]) -> None:
    """Move attempted shortcodes to the end so the next wave tries new IDs."""
    attempted_set = {s for s in attempted if s}
    if not attempted_set or not path.is_file():
        return
    codes = load_shortcodes(path)
    head = [c for c in codes if c not in attempted_set]
    tail = [c for c in codes if c in attempted_set]
    path.write_text(
        "".join(f"https://www.instagram.com/p/{c}/\n" for c in head + tail),
        encoding="utf-8",
    )


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


_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:131.0) Gecko/20100101 Firefox/131.0"
)
DEFAULT_PROFILE_DIR = DEFAULT_RAW_DIR / "pw_ig_profile"
DEFAULT_STORAGE_STATE = DEFAULT_RAW_DIR / "ig_storage_state.json"


def caption_from_media_node(node: dict) -> str | None:
    cap = node.get("caption")
    if isinstance(cap, str) and cap.strip():
        return cap
    if isinstance(cap, dict):
        text = cap.get("text")
        if isinstance(text, str) and text.strip():
            return text
    edges = node.get("edge_media_to_caption")
    if isinstance(edges, dict):
        for edge in edges.get("edges") or []:
            inner = edge.get("node") if isinstance(edge, dict) else None
            if not isinstance(inner, dict):
                continue
            text = inner.get("text")
            if isinstance(text, str) and text.strip():
                return text
    return None


def json_caption_worth_fetching(caption: str | None, *, sarcasm_candidates: bool) -> bool:
    """If JSON already has a caption, drop English / non-irony posts before fetching."""
    if caption is None:
        return True
    text = preprocess_caption(caption)
    if not text or not is_persian_enough(text):
        return False
    if is_political_caption(text):
        return False
    if sarcasm_candidates and not is_sarcasm_candidate_caption(
        text, allow_weak_cues=False
    ):
        return False
    return True


def media_from_payload(obj: object) -> list[tuple[str, str, str | None]]:
    """Pull (shortcode, kind, caption-or-None) out of a GraphQL / api/v1 JSON tree."""
    found: list[tuple[str, str, str | None]] = []
    seen: set[str] = set()

    def walk(node: object) -> None:
        if isinstance(node, dict):
            code = node.get("code") or node.get("shortcode")
            if isinstance(code, str) and _SHORTCODE_TOKEN_RE.fullmatch(code) and code not in seen:
                product = str(node.get("product_type") or "")
                kind = "reel" if product.startswith("clips") else "p"
                seen.add(code)
                found.append((code, kind, caption_from_media_node(node)))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(obj)
    return found


def shortcodes_from_payload(obj: object) -> list[tuple[str, str]]:
    return [(code, kind) for code, kind, _caption in media_from_payload(obj)]


def media_from_response_text(body: str) -> list[tuple[str, str, str | None]]:
    if not body:
        return []
    stripped = body.lstrip()
    rows: list[tuple[str, str, str | None]] = []
    if stripped[:1] in "{[":
        try:
            rows = media_from_payload(json.loads(body))
        except json.JSONDecodeError:
            rows = []
    if rows:
        return rows
    seen: set[str] = set()
    out: list[tuple[str, str, str | None]] = []
    for match in _CODE_IN_TEXT_RE.finditer(body):
        code = match.group(1)
        if code not in seen:
            seen.add(code)
            out.append((code, "p", None))
    return out


def shortcodes_from_response_text(body: str) -> list[tuple[str, str]]:
    return [(code, kind) for code, kind, _caption in media_from_response_text(body)]


def _save_ig_storage(context) -> None:
    if not _has_sessionid(context):
        return
    DEFAULT_STORAGE_STATE.parent.mkdir(parents=True, exist_ok=True)
    context.storage_state(path=str(DEFAULT_STORAGE_STATE))
    log.info("saved Instagram session to %s", DEFAULT_STORAGE_STATE)


def _has_sessionid(context) -> bool:
    try:
        return any(c.get("name") == "sessionid" for c in context.cookies())
    except Exception:
        return False


def _merge_ig_cookies(context, browser: str) -> None:
    try:
        cookies = _playwright_cookies(browser)
        if cookies:
            context.add_cookies(cookies)
    except SystemExit:
        log.warning("No %s Instagram cookies to merge; log in in the opened window.", browser)
    except Exception as exc:  # noqa: BLE001
        log.debug("could not merge %s cookies: %s", browser, exc)


def _launch_ig_context(pw, *, headed: bool, browser: str, profile_dir: Path):
    """Firefox context for Instagram.

    Persistent Playwright profiles crash on this machine (Firefox exits 0).
    Use a normal launch plus ``ig_storage_state.json`` after the first headed login.
    """
    del profile_dir  # kept in the signature for callers
    launched = pw.firefox.launch(headless=not headed)
    context_kwargs: dict = {
        "user_agent": _UA,
        "locale": "fa-IR",
        "viewport": {"width": 1280, "height": 900},
    }
    if DEFAULT_STORAGE_STATE.is_file():
        context_kwargs["storage_state"] = str(DEFAULT_STORAGE_STATE)
        log.info("loading saved Instagram session from %s", DEFAULT_STORAGE_STATE)
    context = launched.new_context(**context_kwargs)
    _merge_ig_cookies(context, browser)
    page = context.new_page()
    return context, page, launched


def _wait_for_ig_session(context, *, headed: bool, seconds: int = 180) -> bool:
    if _has_sessionid(context):
        log.info("Instagram sessionid is present")
        _save_ig_storage(context)
        return True
    if not headed:
        log.warning(
            "No Instagram sessionid. Hashtag pages will be empty. "
            "Re-run with --headed and log in to Instagram in the Firefox window."
        )
        return False
    if seconds <= 0:
        log.warning(
            "No Instagram sessionid. Log in in the Firefox window. Waiting until login (no timeout)."
        )
    else:
        log.warning(
            "No Instagram sessionid. Log in in the Firefox window that opened. Waiting %d seconds.",
            seconds,
        )
    deadline = None if seconds <= 0 else time.time() + seconds
    n = 0
    while deadline is None or time.time() < deadline:
        if _has_sessionid(context):
            log.info("Instagram login detected")
            _save_ig_storage(context)
            return True
        n += 1
        if n % 6 == 0:
            log.warning("still waiting for Instagram login in the Firefox window…")
        time.sleep(5)
    log.warning("Still no sessionid; continuing anyway (pages may be empty)")
    return False


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


def _dom_post_refs(page) -> list[PostRef]:
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
    found: list[PostRef] = []
    seen: set[str] = set()
    for item in refs:
        sc = str(item.get("shortcode", ""))
        kind = str(item.get("kind", "p"))
        if sc and sc not in seen:
            seen.add(sc)
            found.append(PostRef(shortcode=sc, kind=kind))
    return found


def _related_hashtags(page, current: str, *, limit: int = 8) -> list[str]:
    current = current.lstrip("#").strip().lower()
    try:
        tags: list[str] = page.evaluate(
            """() => {
                const out = [];
                for (const a of document.querySelectorAll('a[href*="/explore/tags/"]')) {
                    const m = a.href.match(/\\/explore\\/tags\\/([^/?#]+)/);
                    if (!m) continue;
                    out.push(decodeURIComponent(m[1]));
                }
                return [...new Set(out)];
            }"""
        )
    except Exception:
        return []
    related: list[str] = []
    seen: set[str] = {current}
    for tag in tags:
        key = tag.lstrip("#").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        related.append(key)
        if len(related) >= limit:
            break
    return related


def _click_recent_tab(page) -> None:
    for name in ("Recent", "جدیدترین", "اخیر"):
        try:
            loc = page.get_by_role("tab", name=re.compile(name, re.I))
            if loc.count() == 0:
                continue
            loc.first.click(timeout=2500)
            log.info("clicked hashtag tab %r", name)
            time.sleep(1.2)
            return
        except Exception:
            continue


def _attach_shortcode_harvester(
    page,
    found: list[PostRef],
    seen: set[str],
    *,
    sarcasm_candidates: bool = False,
):
    def on_response(response) -> None:
        try:
            url = response.url
            if not any(hint in url for hint in _HARVEST_URL_HINTS):
                return
            if response.status != 200:
                return
            body = response.text()
        except Exception:
            return
        for sc, kind, caption in media_from_response_text(body):
            if sc in seen:
                continue
            if not json_caption_worth_fetching(caption, sarcasm_candidates=sarcasm_candidates):
                seen.add(sc)
                continue
            seen.add(sc)
            found.append(PostRef(shortcode=sc, kind=kind, caption=caption or ""))

    page.on("response", on_response)
    return on_response


def _hashtag_urls(tag: str) -> list[str]:
    quoted = urllib.parse.quote(tag)
    return [
        f"https://www.instagram.com/explore/tags/{quoted}/",
        f"https://www.instagram.com/explore/tags/{quoted}/?tab=recent",
    ]


def _scroll_hashtag_page(
    page,
    tag: str,
    *,
    scrolls: int = 6,
    scroll_pause: float = 2.5,
    page_timeout_ms: int = 120_000,
    blocked: set[str] | None = None,
    on_batch: Callable[[list[PostRef], str], None] | None = None,
    related_out: list[str] | None = None,
    sarcasm_candidates: bool = False,
) -> list[PostRef]:
    tag = tag.lstrip("#").strip()
    found: list[PostRef] = []
    seen: set[str] = set()
    emitted: set[str] = set()
    urls = _hashtag_urls(tag)
    handler = _attach_shortcode_harvester(
        page, found, seen, sarcasm_candidates=sarcasm_candidates
    )

    def emit(url: str) -> int:
        batch: list[PostRef] = []
        for ref in found:
            if ref.shortcode in emitted:
                continue
            if blocked is not None and ref.shortcode in blocked:
                emitted.add(ref.shortcode)
                continue
            emitted.add(ref.shortcode)
            batch.append(ref)
            if len(batch) >= IMPORT_BATCH:
                break
        if on_batch and batch:
            on_batch(batch, url)
        return len(batch)

    try:
        for url_i, url in enumerate(urls):
            log.info("opening #%s (%s)", tag, url)
            try:
                _goto_with_retry(page, url, page_timeout_ms=page_timeout_ms)
            except Exception as exc:  # noqa: BLE001
                log.warning("skip URL %s: %s", url, exc)
                continue
            time.sleep(scroll_pause)
            if url_i == 0:
                _click_recent_tab(page)
                if related_out is not None:
                    related_out.extend(_related_hashtags(page, tag))
            empty = 0
            for scroll in range(scrolls):
                before = len(found)
                for ref in _dom_post_refs(page):
                    if ref.shortcode not in seen:
                        seen.add(ref.shortcode)
                        found.append(ref)
                n_new = emit(url)
                log.info(
                    "#%s scroll %d/%d: %d links (%d new this batch)",
                    tag,
                    scroll + 1,
                    scrolls,
                    len(found),
                    n_new,
                )
                if n_new == 0 and len(found) == before:
                    empty += 1
                else:
                    empty = 0
                if empty >= 4 and scroll + 1 >= 6:
                    log.info("#%s stopping early after %d empty scrolls", tag, empty)
                    break
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                time.sleep(scroll_pause)
            emit(url)
            if len(found) >= 120:
                break
    finally:
        try:
            page.remove_listener("response", handler)
        except Exception:
            pass

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
    related_out: list[str] | None = None,
    sarcasm_candidates: bool = False,
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
        related_out=related_out,
        sarcasm_candidates=sarcasm_candidates,
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
        context, page, launched = _launch_ig_context(
            pw, headed=headed, browser=browser, profile_dir=DEFAULT_PROFILE_DIR
        )
        _wait_for_ig_session(context, headed=headed)
        found = _scroll_hashtag_page(
            page,
            tag,
            scrolls=scrolls,
            scroll_pause=scroll_pause,
            page_timeout_ms=page_timeout_ms,
        )
        context.close()
        if launched is not None:
            launched.close()

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
    login_wait: int = 180,
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
        context, page, launched = _launch_ig_context(
            pw, headed=headed, browser=browser, profile_dir=DEFAULT_PROFILE_DIR
        )
        try:
            try:
                log.info("warming up instagram.com …")
                _goto_with_retry(page, "https://www.instagram.com/", page_timeout_ms=page_timeout_ms)
                time.sleep(scroll_pause)
            except Exception as exc:  # noqa: BLE001
                log.warning("instagram warm-up failed (%s) — continuing anyway", exc)
            _wait_for_ig_session(context, headed=headed, seconds=login_wait)

            tags = _resolve_hashtags(
                page,
                specs,
                max_search_tags=max_search_tags,
                page_timeout_ms=page_timeout_ms,
                scroll_pause=scroll_pause,
            )
            if not tags:
                raise SystemExit(f"No hashtags resolved from {hashtags_file}")
            seen_tag_keys = {t.lstrip("#").strip().lower() for t in tags}

            consecutive_failures = 0

            def import_batch(
                refs: list[PostRef],
                tag_url: str,
                *,
                prefer_modal: bool = True,
                respect_ignore: bool = True,
                fetch_lightweight: bool = False,
            ) -> int:
                nonlocal written_total
                if written_total >= max_count:
                    return 0
                n = import_shortcodes(
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
                    prefer_modal=prefer_modal,
                    hashtag_url=tag_url,
                    respect_ignore=respect_ignore,
                    fetch_lightweight=fetch_lightweight,
                )
                written_total += n
                return n

            pending_path = out_dir / f"{pool_name}_pending_links.txt"
            if pending_path.is_file() and written_total < max_count:
                pending_from_file = normalize_post_refs(load_shortcodes(pending_path))
                already_in_pool = _jsonl_shortcodes(jsonl_path)
                fresh_pending = [
                    ref
                    for ref in pending_from_file
                    if ref.shortcode not in already_in_pool
                    and ref.shortcode not in seen_candidates
                ]
                if fresh_pending:
                    fresh_pending = fresh_pending[:PENDING_RETRY_BATCH]
                    log.info(
                        "importing %d saved pending links from %s before new hashtags",
                        len(fresh_pending),
                        pending_path,
                    )
                    try:
                        import_batch(
                            fresh_pending,
                            "https://www.instagram.com/",
                            prefer_modal=False,
                            respect_ignore=False,
                            fetch_lightweight=True,
                        )
                    except FetchStall as exc:
                        log.error(
                            "%s. Waiting 2 minutes, then continuing with hashtags.",
                            exc,
                        )
                        try:
                            _save_ig_storage(context)
                        except Exception:
                            pass
                        time.sleep(120)
                    _rotate_pending_file(
                        pending_path, [ref.shortcode for ref in fresh_pending]
                    )

            i = 0
            while i < len(tags):
                tag = tags[i]
                i += 1
                if written_total >= max_count:
                    break
                link_budget = max_count * (12 if require_face or sarcasm_candidates else 3)
                if len(seen_candidates) >= link_budget:
                    break
                if consecutive_failures >= 5:
                    log.error(
                        "Stopped after %d consecutive hashtag timeouts. "
                        "Check VPN, log into Instagram in %s, wait a few minutes, retry.",
                        consecutive_failures,
                        browser,
                    )
                    break
                if i > 1:
                    time.sleep(scroll_pause)
                related: list[str] = []
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
                        related_out=related,
                        sarcasm_candidates=sarcasm_candidates,
                    )
                except FetchStall as exc:
                    log.error(
                        "%s. Waiting 2 minutes, then continuing with the next hashtag "
                        "(browser stays open; session is kept).",
                        exc,
                    )
                    try:
                        _save_ig_storage(context)
                    except Exception:
                        pass
                    time.sleep(120)
                    consecutive_failures = 0
                    continue
                except Exception as exc:  # noqa: BLE001
                    consecutive_failures += 1
                    log.warning("skip hashtag %s: browser collection failed (%s)", tag, exc)
                    time.sleep(scroll_pause * 2)
                    continue
                consecutive_failures = 0
                log.info(
                    "#%s done: %d links seen, %d imported this tag (%d total)",
                    tag,
                    len(found),
                    written,
                    written_total,
                )
                for extra in related:
                    key = extra.lstrip("#").strip().lower()
                    if not key or key in seen_tag_keys:
                        continue
                    if len(tags) >= MAX_RELATED_TAGS:
                        break
                    seen_tag_keys.add(key)
                    tags.append(extra)
                    log.info("queued related #%s", extra)
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
                raise SystemExit(
                    "Browser found no new post links. Log into Instagram in the "
                    "Playwright Firefox window (your desktop Firefox cookies have no "
                    "sessionid), then re-run with --headed."
                )
        finally:
            try:
                _save_ig_storage(context)
            except Exception as exc:  # noqa: BLE001
                log.debug("could not save Instagram session: %s", exc)
            context.close()
            if launched is not None:
                launched.close()

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
    parser.add_argument(
        "--login-wait",
        type=int,
        default=180,
        help="Seconds to wait for Instagram login in --headed Firefox. 0 = wait until login.",
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
        login_wait=args.login_wait,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
