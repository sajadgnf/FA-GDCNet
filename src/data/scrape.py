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


def _iter_instaloader_posts(hashtag: str) -> Iterator[ScrapedPost]:
    import instaloader

    loader = instaloader.Instaloader(
        download_pictures=False,
        download_videos=False,
        download_comments=False,
        save_metadata=False,
        compress_json=False,
        post_metadata_txt_pattern="",
    )

    class _IPost:
        def __init__(self, post: "instaloader.Post"):
            self._post = post

        @property
        def shortcode(self) -> str:
            return self._post.shortcode

        @property
        def caption(self) -> str | None:
            return self._post.caption

        def download_image(self, target: Path) -> Path:
            import urllib.request

            target.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(self._post.url) as resp, target.open("wb") as f:
                f.write(resp.read())
            return target

    for post in instaloader.Hashtag.from_name(loader.context, hashtag).get_posts():
        if post.is_video:
            continue
        yield _IPost(post)


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
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    jsonl_path = args.out_dir / f"{args.hashtag}.jsonl"
    image_dir = args.out_dir / args.hashtag
    posts = _iter_instaloader_posts(args.hashtag)
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
