import instaloader
import re
import time

L = instaloader.Instaloader(max_connection_attempts=1, request_timeout=25)
L.load_session_from_file("sjjd6502")
s = L.context._session
for url in (
    "https://www.instagram.com/explore/tags/selfie/",
    "https://www.instagram.com/p/DZkb37qo3mG/",
):
    t = time.time()
    try:
        r = s.get(url, timeout=25)
        print(url, "status", r.status_code, "len", len(r.text), round(time.time() - t, 1), "s")
        scs = set(re.findall(r'"shortcode":"([A-Za-z0-9_-]{5,})"', r.text))
        if not scs:
            scs = set(re.findall(r"/p/([A-Za-z0-9_-]{5,})/", r.text))
        print("  shortcodes", len(scs), list(scs)[:5])
    except Exception as e:
        print(url, "FAIL", type(e).__name__, str(e)[:100])
