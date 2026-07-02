import html as html_mod
import re
import urllib.request

url = "https://www.instagram.com/p/DZkb37qo3mG/embed/captioned/"
raw = urllib.request.urlopen(url, timeout=25).read().decode("utf-8", "replace")
for pat in [
    r"https://[^\"']*scontent[^\"']+\.(?:jpg|webp)[^\"']*",
    r"https://[^\"']*fbcdn[^\"']+\.(?:jpg|webp)[^\"']*",
    r"display_url[^\"']*\"([^\"]+)\"",
]:
    found = re.findall(pat, raw)
    print(pat[:40], len(found))
    for u in found[:2]:
        print(" ", u[:140])

caps = re.findall(r'class="Caption"[^>]*>(.*?)</span>', raw, re.S)
if caps:
    text = re.sub(r"<[^>]+>", " ", caps[0])
    print("caption", html_mod.unescape(text.strip())[:300])
