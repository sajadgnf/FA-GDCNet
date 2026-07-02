import re
import time
import urllib.request

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"
url = "https://www.instagram.com/explore/tags/selfie/"
req = urllib.request.Request(url, headers={"User-Agent": UA})
t = time.time()
raw = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")
print("status ok", len(raw), round(time.time() - t, 1), "s")
scs = set(re.findall(r"/p/([A-Za-z0-9_-]{5,})/", raw))
scs |= set(re.findall(r'"shortcode":"([A-Za-z0-9_-]{5,})"', raw))
print("shortcodes", len(scs), list(scs)[:10])
