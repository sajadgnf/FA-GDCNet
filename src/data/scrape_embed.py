"""Import Instagram posts from post URLs using public embed pages (no GraphQL API).

When instaloader times out, browse hashtags in Firefox, copy post links into a
text file, then run::

    python tasks.py import-links --links datasets/raw/links.txt --pool-name hashtags --require-face
"""

from __future__ import annotations

import argparse
import html as html_mod
import json
import logging
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .preprocess import is_persian_enough, is_spam_caption, preprocess_caption
from .sarcasm_candidates import is_sarcasm_candidate_caption
from .scrape import (
    DEFAULT_RAW_DIR,
    IGNORED_IDS_FILE,
    _existing_shortcodes,
    _remember_ignored,
)

if TYPE_CHECKING:
    from playwright.sync_api import BrowserContext, Page

log = logging.getLogger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
_SHORTCODE_RE = re.compile(r"(?:instagram\.com/(?:p|reel|tv)/|/p/)([A-Za-z0-9_-]{5,})")
_CAPTION_RE = re.compile(r'class="Caption"[^>]*>(.*?)</span>', re.S)
_SCONTENT_RE = re.compile(
    r"https://[^\"'\s]*scontent[^\"'\s]+\.(?:jpg|webp)[^\"'\s]*",
    re.I,
)
_OG_DESC_CAPTION_RE = re.compile(
    r':\s*["\u201c](.+?)["\u201d]\s*$',
    re.S,
)
_OG_DESC_META_RE = re.compile(
    r'<meta\s+property="og:description"\s+content="([^"]*)"',
    re.I,
)
_OG_IMAGE_META_RE = re.compile(
    r'<meta\s+property="og:image"\s+content="([^"]*)"',
    re.I,
)
_FETCH_HTML_JS = """async ([url, timeoutMs]) => {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), timeoutMs);
    try {
        const resp = await fetch(url, { signal: ctrl.signal, credentials: 'include' });
        if (!resp.ok) throw new Error('HTTP ' + resp.status);
        return await resp.text();
    } finally {
        clearTimeout(timer);
    }
}"""
_MODAL_EXTRACT_JS = """() => {
    const root = document.querySelector('div[role="dialog"]')
        || document.querySelector('article[role="presentation"]')
        || document.querySelector('article');
    if (!root) return null;
    const images = [...root.querySelectorAll('img')]
        .map((img) => img.src)
        .filter((src) => src && (src.includes('scontent') || src.includes('cdninstagram'))
            && !src.includes('s150x150') && !src.includes('s100x100'));
    const ogDesc = document.querySelector('meta[property="og:description"]')?.content || null;
    const skip = /^(like|likes|comment|comments|share|save|follow|more|بیشتر|پسند|نظر|اشتراک|ذخیره|دنبال)$/i;
    let caption = '';
    const h1 = root.querySelector('h1');
    if (h1?.innerText?.trim()) caption = h1.innerText.trim();
    for (const el of root.querySelectorAll('h1, h2, span[dir="auto"], div[dir="auto"]')) {
        const text = (el.innerText || '').trim();
        if (!text || text.length < 4 || skip.test(text)) continue;
        if (text.length > caption.length) caption = text;
    }
    return { images, ogDesc, caption };
}"""
_CAPTION_JSON_JS = """() => {
    function walk(obj, depth) {
        if (!obj || depth > 16) return '';
        if (typeof obj !== 'object') return '';
        const edges = obj.edge_media_to_caption?.edges;
        if (Array.isArray(edges) && edges[0]?.node?.text) return edges[0].node.text;
        if (typeof obj.caption?.text === 'string' && obj.caption.text.length > 3)
            return obj.caption.text;
        if (typeof obj.text === 'string' && obj.text.length > 8) {
            const t = obj.__typename || '';
            if (t.includes('Caption') || t.includes('Comment')) return obj.text;
        }
        for (const v of Object.values(obj)) {
            const found = walk(v, depth + 1);
            if (found) return found;
        }
        return '';
    }
    for (const s of document.querySelectorAll('script[type="application/json"]')) {
        try {
            const found = walk(JSON.parse(s.textContent), 0);
            if (found) return found;
        } catch (_) {}
    }
    return '';
}"""
_POST_EXTRACT_JS = """() => {
    const ogImage = document.querySelector('meta[property="og:image"]')?.content || null;
    const ogDesc = document.querySelector('meta[property="og:description"]')?.content || null;
    const images = [...document.querySelectorAll('img')]
        .map((img) => img.src)
        .filter((src) => src && (src.includes('scontent') || src.includes('cdninstagram')));
    return { ogImage, ogDesc, images };
}"""


@dataclass(frozen=True)
class PostRef:
    shortcode: str
    kind: str = "p"


def post_ref(shortcode: str, kind: str = "p") -> PostRef:
    return PostRef(shortcode=shortcode, kind=kind)


def normalize_post_refs(items: list[str] | list[PostRef]) -> list[PostRef]:
    out: list[PostRef] = []
    for item in items:
        if isinstance(item, PostRef):
            out.append(item)
        else:
            out.append(PostRef(shortcode=item, kind="p"))
    return out


def _shortcode_from_line(line: str) -> str | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    m = _SHORTCODE_RE.search(line)
    if m:
        return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{5,}", line):
        return line
    return None


def load_shortcodes(path: Path) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            sc = _shortcode_from_line(line)
            if sc and sc not in seen:
                seen.add(sc)
                out.append(sc)
    return out


def _fetch_embed_html(shortcode: str, *, timeout: float = 30.0) -> str:
    url = f"https://www.instagram.com/p/{shortcode}/embed/captioned/"
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def _caption_from_og_description(desc: str) -> str:
    desc = desc.strip()
    if not desc:
        return ""
    m = _OG_DESC_CAPTION_RE.search(desc)
    if m:
        return html_mod.unescape(m.group(1)).strip()
    if ":" in desc:
        tail = desc.rsplit(":", 1)[-1].strip()
        return html_mod.unescape(tail.strip('"').strip("'")).strip()
    return html_mod.unescape(desc)


def _pick_image_url(urls: list[str]) -> str | None:
    if not urls:
        return None
    posts = [
        u
        for u in urls
        if "s100x100" not in u and "s150x150" not in u and "/t51.82787-19/" not in u
    ]
    if not posts:
        posts = urls
    posts.sort(key=len, reverse=True)
    return posts[0]


def _pick_image_url_from_browser(data: dict[str, Any]) -> str | None:
    urls: list[str] = []
    og = data.get("ogImage")
    if isinstance(og, str) and og:
        urls.append(og)
    for src in data.get("images") or []:
        if isinstance(src, str) and src:
            urls.append(src)
    return _pick_image_url(urls)


def _embed_url(shortcode: str, kind: str = "p") -> str:
    return f"https://www.instagram.com/{kind}/{shortcode}/embed/captioned/"


def _parse_caption_from_html(html: str) -> str:
    caption = _extract_caption(html)
    if not caption:
        m = _OG_DESC_META_RE.search(html)
        if m:
            caption = _caption_from_og_description(html_mod.unescape(m.group(1)))
    return caption


def _parse_post_html(html: str) -> tuple[str, str]:
    image_url = _pick_image_url_from_html(html)
    if not image_url:
        m = _OG_IMAGE_META_RE.search(html)
        if m:
            image_url = html_mod.unescape(m.group(1))
    if not image_url:
        raise RuntimeError("no image in html")
    caption = _parse_caption_from_html(html)
    return caption, image_url


def _caption_from_modal_data(data: dict[str, Any]) -> str:
    caption = str(data.get("caption") or "").strip()
    if not caption:
        caption = _caption_from_og_description(str(data.get("ogDesc") or ""))
    return caption


def _expand_modal_caption(page: Page) -> None:
    """Click Instagram's 'more' / 'بیشتر' to reveal full caption text."""
    for pattern in ("more", "بیشتر", "…"):
        try:
            btn = page.locator('div[role="dialog"] button, div[role="dialog"] span[role="button"]').filter(
                has_text=re.compile(pattern, re.I)
            ).first
            if btn.count():
                btn.click(timeout=1500)
                time.sleep(0.4)
        except Exception:
            pass


def _extract_caption_from_page(page: Page) -> str:
    caption = page.evaluate(_CAPTION_JSON_JS)
    if isinstance(caption, str) and caption.strip():
        return caption.strip()
    data = page.evaluate(_MODAL_EXTRACT_JS)
    if isinstance(data, dict):
        return _caption_from_modal_data(data)
    return ""


def _try_fetch_caption_only(
    page: Page,
    shortcode: str,
    *,
    kind: str = "p",
    timeout_ms: int = 120_000,
) -> str:
    """Fetch caption without navigating away (rate-limit friendly)."""
    caption = _extract_caption_from_page(page)
    if caption:
        return caption
    try:
        html = _fetch_html_inpage(page, _embed_url(shortcode, kind), timeout_ms=timeout_ms)
        caption = _parse_caption_from_html(html)
        if caption:
            return caption
    except Exception:
        pass
    return ""


def _fetch_html_inpage(page: Page, url: str, *, timeout_ms: int) -> str:
    html = page.evaluate(_FETCH_HTML_JS, [url, timeout_ms])
    if not isinstance(html, str) or not html.strip():
        raise RuntimeError(f"empty in-page fetch for {url}")
    return html


def _fetch_post_embed_inpage(
    page: Page,
    shortcode: str,
    *,
    kind: str = "p",
    timeout_ms: int = 120_000,
) -> tuple[str, str]:
    html = _fetch_html_inpage(page, _embed_url(shortcode, kind), timeout_ms=timeout_ms)
    return _parse_post_html(html)


def _fetch_post_embed_goto(
    page: Page,
    shortcode: str,
    *,
    kind: str = "p",
    timeout_ms: int = 120_000,
) -> tuple[str, str]:
    url = _embed_url(shortcode, kind)
    last_exc: Exception | None = None
    for wait_until in ("commit", "domcontentloaded"):
        try:
            page.goto(url, wait_until=wait_until, timeout=timeout_ms)
            last_exc = None
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
    if last_exc is not None:
        raise RuntimeError(f"embed navigation failed for {shortcode}: {last_exc}") from last_exc
    time.sleep(1)
    html = page.content()
    return _parse_post_html(html)


def _reload_hashtag_page(page: Page, hashtag_url: str, *, timeout_ms: int = 120_000) -> None:
    last_exc: Exception | None = None
    for wait_until in ("commit", "domcontentloaded"):
        try:
            page.goto(hashtag_url, wait_until=wait_until, timeout=timeout_ms)
            time.sleep(1.5)
            return
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
    raise RuntimeError(f"failed to reload hashtag page: {last_exc}") from last_exc


def _ensure_post_link_visible(
    page: Page,
    shortcode: str,
    *,
    kind: str = "p",
    hashtag_url: str | None = None,
    timeout_ms: int = 120_000,
) -> None:
    """Scroll the hashtag grid until the post thumbnail link is on screen."""
    href_part = f"/{kind}/{shortcode}"

    def _find_and_center() -> bool:
        return page.evaluate(
            """([part]) => {
                for (const a of document.querySelectorAll('a[href*="' + part + '"]')) {
                    a.scrollIntoView({ block: 'center', behavior: 'instant' });
                    const r = a.getBoundingClientRect();
                    if (r.width > 0 && r.height > 0) return true;
                }
                return false;
            }""",
            href_part,
        )

    def _scan_grid() -> bool:
        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(0.4)
        for _ in range(35):
            if _find_and_center():
                time.sleep(0.6)
                return True
            page.evaluate("window.scrollBy(0, window.innerHeight * 0.75)")
            time.sleep(0.35)
        return False

    if _find_and_center() or _scan_grid():
        return
    if hashtag_url:
        _reload_hashtag_page(page, hashtag_url, timeout_ms=timeout_ms)
        if _scan_grid():
            return
    raise RuntimeError(f"post {shortcode} not visible on page for modal open")


def _fetch_post_via_modal(
    page: Page,
    shortcode: str,
    *,
    kind: str = "p",
    timeout_ms: int = 120_000,
    hashtag_url: str | None = None,
) -> tuple[str, str]:
    _ensure_post_link_visible(
        page,
        shortcode,
        kind=kind,
        hashtag_url=hashtag_url,
        timeout_ms=timeout_ms,
    )
    href_part = f"/{kind}/{shortcode}"
    link = page.locator(f'a[href*="{href_part}"]').first
    try:
        link.wait_for(state="visible", timeout=min(8000, timeout_ms // 4))
    except Exception as exc:
        raise RuntimeError(f"post {shortcode} not visible on page for modal open") from exc
    time.sleep(0.6)
    link.click()
    try:
        page.wait_for_function(
            """() => {
                const root = document.querySelector('div[role="dialog"]') || document.querySelector('article');
                if (!root) return false;
                return [...root.querySelectorAll('img')].some(
                    (img) => img.src.includes('scontent') && !img.src.includes('s150x150')
                );
            }""",
            timeout=min(20_000, timeout_ms // 2),
        )
    except Exception:
        time.sleep(2)
    time.sleep(1.2)
    _expand_modal_caption(page)
    data = page.evaluate(_MODAL_EXTRACT_JS)
    if not data:
        page.keyboard.press("Escape")
        raise RuntimeError(f"modal did not open for {shortcode}")
    image_url = _pick_image_url([str(u) for u in data.get("images") or [] if isinstance(u, str)])
    if not image_url:
        page.keyboard.press("Escape")
        raise RuntimeError(f"no image in modal for {shortcode}")
    caption = _caption_from_modal_data(data)
    if not caption:
        caption = _extract_caption_from_page(page)
    if not caption:
        caption = _try_fetch_caption_only(page, shortcode, kind=kind, timeout_ms=timeout_ms)
    page.keyboard.press("Escape")
    time.sleep(1.2)
    return caption, image_url


def _fetch_post_full_goto(
    page: Page,
    shortcode: str,
    *,
    kind: str = "p",
    timeout_ms: int = 120_000,
) -> tuple[str, str]:
    url = f"https://www.instagram.com/{kind}/{shortcode}/"
    nav_error: Exception | None = None
    for attempt in range(3):
        for wait_until in ("commit", "domcontentloaded"):
            try:
                page.goto(url, wait_until=wait_until, timeout=timeout_ms)
                nav_error = None
                break
            except Exception as exc:  # noqa: BLE001
                nav_error = exc
        if nav_error is None:
            break
        time.sleep(3 + attempt * 2)
    if nav_error is not None:
        raise RuntimeError(f"browser navigation failed for {shortcode}: {nav_error}") from nav_error

    try:
        page.wait_for_function(
            """() => {
                const og = document.querySelector('meta[property="og:image"]');
                if (og?.content?.includes('scontent')) return true;
                return [...document.querySelectorAll('img')].some(
                    (img) => img.src.includes('scontent') && !img.src.includes('s150x150')
                );
            }""",
            timeout=15_000,
        )
    except Exception:
        time.sleep(2)
    data = page.evaluate(_POST_EXTRACT_JS)
    image_url = _pick_image_url_from_browser(data)
    if not image_url:
        raise RuntimeError(f"no image on post page for {shortcode}")
    caption = _caption_from_og_description(str(data.get("ogDesc") or ""))
    if not caption:
        caption = _extract_caption_from_page(page)
    return caption, image_url


def fetch_post_via_browser(
    page: Page,
    shortcode: str,
    *,
    kind: str = "p",
    timeout_ms: int = 120_000,
    prefer_modal: bool = False,
    lightweight_only: bool = False,
    hashtag_url: str | None = None,
) -> tuple[str, str]:
    """Return (caption, image_url) using a logged-in Playwright page.

    Tries lighter strategies first (modal / in-page embed fetch) because full
    ``/p/`` navigation often fails with NS_ERROR_NET_EMPTY_RESPONSE on VPN.
    """
    strategies: list[tuple[str, Any]] = []
    modal_kw = {"kind": kind, "timeout_ms": timeout_ms, "hashtag_url": hashtag_url}
    if prefer_modal:
        strategies.append(("modal", lambda: _fetch_post_via_modal(page, shortcode, **modal_kw)))
    if not lightweight_only:
        strategies.extend(
            [
                ("embed-fetch", lambda: _fetch_post_embed_inpage(page, shortcode, kind=kind, timeout_ms=timeout_ms)),
                ("embed-goto", lambda: _fetch_post_embed_goto(page, shortcode, kind=kind, timeout_ms=timeout_ms)),
            ]
        )
    elif not prefer_modal:
        strategies.append(
            ("embed-fetch", lambda: _fetch_post_embed_inpage(page, shortcode, kind=kind, timeout_ms=timeout_ms))
        )
    if not prefer_modal:
        strategies.append(("modal", lambda: _fetch_post_via_modal(page, shortcode, **modal_kw)))
    if not lightweight_only:
        strategies.append(("full-goto", lambda: _fetch_post_full_goto(page, shortcode, kind=kind, timeout_ms=timeout_ms)))

    errors: list[str] = []
    best_image: str | None = None
    for name, fn in strategies:
        try:
            caption, image_url = fn()
            if image_url and not best_image:
                best_image = image_url
            if caption.strip():
                log.debug("fetched %s via %s", shortcode, name)
                return caption, image_url
            if image_url:
                errors.append(f"{name}: empty caption")
                continue
            errors.append(f"{name}: no image")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {exc}")
            log.debug("fetch %s via %s failed: %s", shortcode, name, exc)
    if best_image:
        raise RuntimeError(f"got image but no caption for {shortcode}: {'; '.join(errors)}")
    raise RuntimeError(f"all fetch strategies failed for {shortcode}: {'; '.join(errors)}")


def _download_image_browser(page: Page, url: str, dest: Path, *, timeout: float = 60.0) -> None:
    """Download via in-page fetch so traffic uses the browser VPN route, not Playwright API."""
    data = page.evaluate(
        """async ([url, timeoutMs]) => {
            const ctrl = new AbortController();
            const timer = setTimeout(() => ctrl.abort(), timeoutMs);
            try {
                const resp = await fetch(url, { signal: ctrl.signal, credentials: 'omit' });
                if (!resp.ok) throw new Error('HTTP ' + resp.status);
                const buf = await resp.arrayBuffer();
                return Array.from(new Uint8Array(buf));
            } finally {
                clearTimeout(timer);
            }
        }""",
        [url, int(timeout * 1000)],
    )
    if not data:
        raise urllib.error.URLError(f"empty response for {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(bytes(data))


def _pick_image_url_from_html(html: str) -> str | None:
    urls = [html_mod.unescape(u) for u in _SCONTENT_RE.findall(html)]
    return _pick_image_url(urls)


def _extract_caption(html: str) -> str:
    m = _CAPTION_RE.search(html)
    if not m:
        return ""
    text = re.sub(r"<[^>]+>", " ", m.group(1))
    return html_mod.unescape(text).strip()


def _download_image(url: str, dest: Path, *, timeout: float = 60.0) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)


def fetch_embed_post(shortcode: str, *, timeout: float = 30.0) -> tuple[str, str]:
    """Return (caption, image_url) for a post shortcode via the public embed page."""
    html = _fetch_embed_html(shortcode, timeout=timeout)
    image_url = _pick_image_url_from_html(html)
    if not image_url:
        raise RuntimeError(f"no image URL in embed page for {shortcode}")
    return _extract_caption(html), image_url


def import_shortcodes(
    shortcodes: list[str] | list[PostRef],
    *,
    pool_name: str = "hashtags",
    out_dir: Path = DEFAULT_RAW_DIR,
    require_face: bool = False,
    min_face_size: int = 40,
    delay: float = 2.0,
    timeout: float = 30.0,
    max_count: int | None = None,
    page: Page | None = None,
    browser_context: BrowserContext | None = None,
    sarcasm_candidates: bool = False,
    prefer_modal: bool = False,
    hashtag_url: str | None = None,
) -> int:
    if require_face:
        try:
            from .face_filter import has_face as _image_has_face
        except ImportError as exc:
            raise SystemExit(
                "Face filter requires opencv. Install with: pip install opencv-python-headless"
            ) from exc
    else:
        _image_has_face = None  # type: ignore[assignment,misc]

    refs = normalize_post_refs(shortcodes)
    use_browser = page is not None
    if use_browser and browser_context is None:
        browser_context = page.context

    jsonl_path = out_dir / f"{pool_name}.jsonl"
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    seen = _existing_shortcodes(jsonl_path)

    written = 0
    skipped_no_face = 0
    skipped_no_caption = 0
    skipped_not_sarcasm = 0
    consecutive_fetch_failures = 0
    lightweight_fetch = False
    post_timeout_ms = max(int(timeout * 1000), 120_000)
    with jsonl_path.open("a", encoding="utf-8") as out:
        for i, ref in enumerate(refs):
            shortcode = ref.shortcode
            if max_count is not None and written >= max_count:
                break
            if shortcode in seen:
                log.debug("skip duplicate %s", shortcode)
                continue
            if i:
                time.sleep(delay)
            try:
                if use_browser and page is not None:
                    caption_raw, image_url = fetch_post_via_browser(
                        page,
                        shortcode,
                        kind=ref.kind,
                        timeout_ms=post_timeout_ms,
                        prefer_modal=prefer_modal,
                        lightweight_only=lightweight_fetch,
                        hashtag_url=hashtag_url,
                    )
                else:
                    caption_raw, image_url = fetch_embed_post(shortcode, timeout=timeout)
            except Exception as exc:  # noqa: BLE001 — playwright/network errors vary
                consecutive_fetch_failures += 1
                lightweight_fetch = consecutive_fetch_failures >= 2
                log.warning("skip %s: fetch failed (%s)", shortcode, exc)
                if consecutive_fetch_failures >= 3:
                    pause = min(60, 15 * consecutive_fetch_failures)
                    log.warning(
                        "Pausing %ds after %d consecutive fetch failures (rate limit? use --delay 8)",
                        pause,
                        consecutive_fetch_failures,
                    )
                    time.sleep(pause)
                if consecutive_fetch_failures >= 6:
                    log.error(
                        "Stopping import after %d consecutive failures — wait 10 min, retry with --delay 8",
                        consecutive_fetch_failures,
                    )
                    break
                time.sleep(delay)
                continue
            consecutive_fetch_failures = 0
            lightweight_fetch = False

            caption = preprocess_caption(caption_raw)
            if not caption:
                skipped_no_caption += 1
                log.warning("skip %s: empty caption", shortcode)
                time.sleep(delay / 2)
                continue
            if not is_persian_enough(caption):
                log.warning("skip %s: caption not Persian enough", shortcode)
                continue
            if is_spam_caption(caption):
                _remember_ignored(shortcode)
                seen.add(shortcode)
                continue
            if sarcasm_candidates and not is_sarcasm_candidate_caption(caption):
                skipped_not_sarcasm += 1
                log.debug("skip %s: caption lacks sarcasm/irony cues", shortcode)
                continue

            ext = ".webp" if ".webp" in image_url.split("?")[0].lower() else ".jpg"
            dest = image_dir / f"{shortcode}{ext}"
            try:
                if use_browser and page is not None:
                    _download_image_browser(page, image_url, dest, timeout=timeout * 2)
                else:
                    _download_image(image_url, dest, timeout=timeout * 2)
            except Exception as exc:  # noqa: BLE001 — playwright/network errors vary
                log.warning("skip %s: image download failed (%s)", shortcode, exc)
                continue

            if require_face and _image_has_face is not None:
                if not _image_has_face(dest, min_size=min_face_size):
                    skipped_no_face += 1
                    _remember_ignored(shortcode)
                    seen.add(shortcode)
                    dest.unlink(missing_ok=True)
                    log.info("skip %s: no face detected", shortcode)
                    continue

            row = {
                "post_id": shortcode,
                "caption": caption,
                "image_path": str(dest),
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            seen.add(shortcode)
            written += 1
            log.info("imported %s", shortcode)
            if use_browser:
                time.sleep(max(1.0, delay / 3))

    if skipped_no_caption:
        log.info("skipped %d posts with empty caption", skipped_no_caption)
    if skipped_no_face:
        log.info(
            "skipped %d posts with no face (IDs saved to %s)",
            skipped_no_face,
            IGNORED_IDS_FILE,
        )
    if skipped_not_sarcasm:
        log.info("skipped %d posts with no sarcasm cues in caption (plain selfies)", skipped_not_sarcasm)
    log.info("imported %d posts into %s", written, jsonl_path)
    return written


def import_from_links(
    links_path: Path,
    *,
    pool_name: str = "hashtags",
    out_dir: Path = DEFAULT_RAW_DIR,
    require_face: bool = False,
    min_face_size: int = 40,
    delay: float = 2.0,
    timeout: float = 30.0,
    max_count: int | None = None,
    sarcasm_candidates: bool = False,
) -> int:
    if not links_path.is_file():
        raise SystemExit(f"links file not found: {links_path}")

    shortcodes = load_shortcodes(links_path)
    if not shortcodes:
        raise SystemExit(f"no post URLs/shortcodes found in {links_path}")

    return import_shortcodes(
        shortcodes,
        pool_name=pool_name,
        out_dir=out_dir,
        require_face=require_face,
        min_face_size=min_face_size,
        delay=delay,
        timeout=timeout,
        max_count=max_count,
        sarcasm_candidates=sarcasm_candidates,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Import Instagram posts from /p/ URLs via embed pages (no API).",
    )
    parser.add_argument("--links", type=Path, required=True)
    parser.add_argument("--pool-name", default="hashtags")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--require-face", action="store_true")
    parser.add_argument("--sarcasm-candidates", action="store_true")
    parser.add_argument("--min-face-size", type=int, default=40)
    parser.add_argument("--delay", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    import_from_links(
        args.links,
        pool_name=args.pool_name,
        out_dir=args.out_dir,
        require_face=args.require_face,
        min_face_size=args.min_face_size,
        delay=args.delay,
        timeout=args.timeout,
        sarcasm_candidates=args.sarcasm_candidates,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
