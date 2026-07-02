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


def fetch_post_via_browser(
    page: Page,
    shortcode: str,
    *,
    kind: str = "p",
    timeout_ms: int = 60000,
) -> tuple[str, str]:
    """Return (caption, image_url) using a logged-in Playwright page."""
    url = f"https://www.instagram.com/{kind}/{shortcode}/"
    nav_error: Exception | None = None
    for attempt in range(3):
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            nav_error = None
            break
        except Exception as exc:  # noqa: BLE001 — playwright navigation errors vary
            nav_error = exc
            if attempt < 2:
                log.debug("retry navigation for %s (%s)", shortcode, exc)
                time.sleep(2 + attempt)
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
            timeout=15000,
        )
    except Exception:
        time.sleep(2)
    data = page.evaluate(_POST_EXTRACT_JS)
    image_url = _pick_image_url_from_browser(data)
    if not image_url:
        raise RuntimeError(f"no image on post page for {shortcode}")
    caption = _caption_from_og_description(str(data.get("ogDesc") or ""))
    return caption, image_url


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
    image_dir = out_dir / pool_name
    image_dir.mkdir(parents=True, exist_ok=True)
    seen = _existing_shortcodes(jsonl_path)

    written = 0
    skipped_no_face = 0
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
                        timeout_ms=int(timeout * 1000),
                    )
                else:
                    caption_raw, image_url = fetch_embed_post(shortcode, timeout=timeout)
            except Exception as exc:  # noqa: BLE001 — playwright/network errors vary
                log.warning("skip %s: fetch failed (%s)", shortcode, exc)
                continue

            caption = preprocess_caption(caption_raw)
            if caption and not is_persian_enough(caption):
                log.warning("skip %s: caption not Persian enough", shortcode)
                continue
            if caption and is_spam_caption(caption):
                _remember_ignored(shortcode)
                seen.add(shortcode)
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

    if skipped_no_face:
        log.info(
            "skipped %d posts with no face (IDs saved to %s)",
            skipped_no_face,
            IGNORED_IDS_FILE,
        )
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
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Import Instagram posts from /p/ URLs via embed pages (no API).",
    )
    parser.add_argument("--links", type=Path, required=True)
    parser.add_argument("--pool-name", default="hashtags")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--require-face", action="store_true")
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
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
