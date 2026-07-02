import html as html_mod
import re
import sys
import urllib.request

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
shortcodes = sys.argv[1:] or ["DKIGkcViP-9", "DRaNSHBCC8Z", "DZkb37qo3mG"]

for sc in shortcodes:
    url = f"https://www.instagram.com/p/{sc}/embed/captioned/"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        raw = urllib.request.urlopen(req, timeout=25).read().decode("utf-8", "replace")
        print("===", sc, "len", len(raw))
        if "<title" in raw:
            i = raw.find("<title")
            print("title:", raw[i : raw.find("</title>", i) + 8])
        for pat in [
            r"https://[^\"']*scontent[^\"']+",
            r"https://[^\"']*cdninstagram[^\"']+",
            r"https://[^\"']*fbcdn[^\"']+",
            r'property="og:image" content="([^"]+)"',
            r'"display_url":"([^"]+)"',
            r'src="(https://[^"]+)"',
        ]:
            found = re.findall(pat, raw)
            if found:
                print(" ", pat[:45], "->", len(found))
                print("   ", found[0][:120])
    except Exception as e:
        print("===", sc, "ERROR", e)
