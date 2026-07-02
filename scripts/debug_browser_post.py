"""Debug browser-based post fetch (requires playwright + firefox)."""

from __future__ import annotations

import json
import sys
import time

from data.collect_browser import _playwright_cookies

sc = sys.argv[1] if len(sys.argv) > 1 else "DKIGkcViP-9"

from playwright.sync_api import sync_playwright

with sync_playwright() as pw:
    b = pw.firefox.launch(headless=True)
    ctx = b.new_context(
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        locale="fa-IR",
    )
    ctx.add_cookies(_playwright_cookies("firefox"))
    page = ctx.new_page()

    for path in [f"/p/{sc}/", f"/p/{sc}/embed/captioned/"]:
        url = "https://www.instagram.com" + path
        print("===", url)
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        time.sleep(4)
        data = page.evaluate(
            """() => {
            const og = document.querySelector('meta[property="og:image"]');
            const desc = document.querySelector('meta[property="og:description"]');
            const imgs = [...document.querySelectorAll('img')]
              .map(i => ({src: i.src, alt: (i.alt||'').slice(0,120)}))
              .filter(x => x.src.includes('scontent') || x.src.includes('cdninstagram'));
            return {
              ogImage: og?.content || null,
              description: desc?.content || null,
              imgs: imgs.slice(0, 5),
            };
        }"""
        )
        print(json.dumps(data, ensure_ascii=False, indent=2)[:2000])

    ctx.close()
    b.close()
