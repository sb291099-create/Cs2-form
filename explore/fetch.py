import gzip, json, sys, time, pathlib, urllib.parse, requests
UA = "cs2-form/0.1 (https://github.com/sb291099-create/Cs2-form)"
s = requests.Session(); s.headers.update({"User-Agent": UA, "Accept-Encoding": "gzip"})
API = "https://liquipedia.net/counterstrike/api.php"
out = pathlib.Path("explore/out"); out.mkdir(exist_ok=True)
titles = ["S-Tier_Tournaments", "A-Tier_Tournaments", "ESL/Pro_League/Season_24", "PARIVISION", "BLAST/Open/Fall/2026"]
for t in titles:
    time.sleep(3)
    r = s.get(API, params={"action": "query", "prop": "revisions", "rvprop": "content", "rvslots": "main", "titles": t, "format": "json", "redirects": 1})
    (out / (t.replace("/", "__") + ".json")).write_text(r.text)
    print(t, r.status_code, len(r.text))
time.sleep(3)
r = s.get(API, params={"action": "query", "list": "prefixsearch", "pssearch": "ESL/Pro League/Season 24", "pslimit": 50, "format": "json"})
(out / "prefix_epl24.json").write_text(r.text)
