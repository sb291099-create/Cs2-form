"""Временная проверка доступности источников с серверов GitHub."""
import re
import time

import requests
from curl_cffi import requests as creq

URLS = [
    "https://www.hltv.org/results",
    "https://www.hltv.org/matches/2398718/legacy-vs-parivision-esl-pro-league-season-24",
    "https://www.hltv.org/matches",
    "https://www.hltv.org/rss/news",
    "https://liquipedia.net/counterstrike/api.php?action=query&list=recentchanges&rclimit=1&format=json",
]
s = creq.Session(impersonate="chrome")
for u in URLS:
    time.sleep(4)
    try:
        r = s.get(u, timeout=30)
        t = re.search(r"<title[^>]*>(.*?)</title>", r.text, re.S | re.I)
        info = (t.group(1).strip() if t else r.text[:80]).replace("\n", " ")[:90]
        extra = ""
        if "/matches/" in u and r.status_code == 200:
            extra = f" veto={'veto-box' in r.text} mapholder={r.text.count('mapholder')}"
        if "results" in u and r.status_code == 200:
            extra = f" result-con={r.text.count('result-con')}"
        print(f"::notice::{r.status_code} {u} «{info}»{extra}")
    except Exception as e:
        print(f"::notice::ERR {u} {e}")
