import instaloader
import re
import json

L = instaloader.Instaloader()
L.load_session_from_file(
    "sjjd6502",
    filename=r"C:\Users\s.ganjifar\AppData\Local\Instaloader\session-sjjd6502",
)
ctx = L.context
url = "https://www.instagram.com/explore/tags/nature/"
resp = ctx._session.get(url, timeout=30)
print("status", resp.status_code, "len", len(resp.text))
text = resp.text
for pat in [
    r'"shortcode":"([A-Za-z0-9_-]{5,})"',
    r'"code":"([A-Za-z0-9_-]{5,})"',
    r'/p/([A-Za-z0-9_-]{5,})/',
]:
    scs = re.findall(pat, text)
    print(pat, len(scs), scs[:5])

# embedded JSON blobs
for m in re.finditer(r'<script type="application/json"[^>]*>(.*?)</script>', text, re.S):
    blob = m.group(1)
    if "shortcode" in blob or "xdt_location" in blob or "hashtag" in blob.lower():
        print("json_blob_len", len(blob), "snippet", blob[:200])
        try:
            data = json.loads(blob)
            print("json_keys", list(data.keys())[:10])
        except json.JSONDecodeError:
            pass
