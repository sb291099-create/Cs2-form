import time, pathlib, requests
UA = "cs2-form/0.1 (https://github.com/sb291099-create/Cs2-form)"
s = requests.Session(); s.headers.update({"User-Agent": UA, "Accept-Encoding": "gzip"})
API = "https://liquipedia.net/counterstrike/api.php"
out = pathlib.Path("explore/out"); out.mkdir(exist_ok=True)
def q(name, **params):
    time.sleep(3)
    r = s.get(API, params={"format": "json", **params})
    (out / name).write_text(r.text); print(name, r.status_code, len(r.text))
q("tiers.json", action="query", prop="revisions", rvprop="content", rvslots="main", titles="S-Tier Tournaments/Post 2023|A-Tier Tournaments/Post 2023|B-Tier Tournaments/Post 2023")
q("cologne_sub.json", action="query", list="allpages", apprefix="Intel Extreme Masters/2026/Cologne", aplimit=50)
q("cologne_sub2.json", action="query", list="prefixsearch", pssearch="IEM Cologne Major 2026", pslimit=20)
q("teamtpl.json", action="query", prop="revisions", rvprop="content", rvslots="main", titles="Template:Team/furia|Template:TeamShort/furia|Template:TeamPart/furia|Module:TeamTemplate/data")
q("expand.json", action="expandtemplates", text="{{Team|furia}} {{Team|parivision}} {{TeamOpponent|spirit}}", prop="wikitext")
