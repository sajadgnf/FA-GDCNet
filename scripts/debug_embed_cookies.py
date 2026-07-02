import http.cookiejar
import re
import urllib.request

import browser_cookie3

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
sc = "DKIGkcViP-9"
url = f"https://www.instagram.com/p/{sc}/embed/captioned/"

jar = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
for c in browser_cookie3.firefox(domain_name=".instagram.com"):
    jar.set_cookie(c)

req = urllib.request.Request(url, headers={"User-Agent": UA})
raw = opener.open(req, timeout=25).read().decode("utf-8", "replace")
print("len", len(raw))
found = re.findall(r"https://[^\"']*scontent[^\"']+\.(?:jpg|webp)[^\"']*", raw, re.I)
print("scontent", len(found))
for u in found[:3]:
    print(" ", u[:120])
caps = re.findall(r'class="Caption"[^>]*>(.*?)</span>', raw, re.S)
print("caption blocks", len(caps))
