"""Сбор матчей, карт, составов и медиа-ссылок с Liquipedia через официальный MediaWiki API.

Правила API Liquipedia: свой User-Agent, gzip, не чаще 1 запроса в 2 секунды
(для action=parse/expandtemplates — 1 запрос в 30 секунд).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import date

import requests

API = "https://liquipedia.net/counterstrike/api.php"
USER_AGENT = "cs2-form/0.2 (https://github.com/sb291099-create/Cs2-form)"
MONTHS = {
    m: i
    for i, m in enumerate(
        [
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ],
        1,
    )
}


class Liquipedia:
    def __init__(self, delay: float = 2.5, parse_delay: float = 31.0):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"})
        self.delay, self.parse_delay = delay, parse_delay
        self._last = 0.0

    def _get(self, params: dict, slow: bool = False) -> dict:
        wait = self._last + (self.parse_delay if slow else self.delay) - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        for attempt in range(1, 4):
            r = self.s.get(API, params={"format": "json", **params}, timeout=60)
            self._last = time.monotonic()
            if r.status_code == 429:
                time.sleep(60 * attempt)
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError("Liquipedia: превышен лимит запросов")

    def expand(self, text: str) -> str:
        return self._get({"action": "expandtemplates", "text": text, "prop": "wikitext"}, slow=True)["expandtemplates"][
            "wikitext"
        ]

    def subpages(self, title: str) -> list[str]:
        d = self._get({"action": "query", "list": "allpages", "apprefix": title + "/", "aplimit": 100})
        return [p["title"] for p in d["query"]["allpages"]]

    def wikitext(self, titles: list[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for i in range(0, len(titles), 20):
            d = self._get(
                {
                    "action": "query",
                    "prop": "revisions",
                    "rvprop": "content",
                    "rvslots": "main",
                    "titles": "|".join(titles[i : i + 20]),
                    "redirects": 1,
                }
            )
            for p in d["query"].get("pages", {}).values():
                if "revisions" in p:
                    out[p["title"]] = p["revisions"][0]["slots"]["main"]["*"]
        return out

    def tournaments(self, start: date, end: date, tiers=(1, 2, 3)) -> list[tuple[str, int]]:
        """Список страниц турниров нужных тиров (1=S, 2=A, 3=B) за период."""
        text = "".join(
            f"<tier{t}>{{{{TournamentsList|tier={t}|tiertype=none|sdate={start}|edate={end}}}}}</tier{t}>"
            for t in tiers
        )
        html = self.expand(text)
        found: list[tuple[str, int]] = []
        for t in tiers:
            m = re.search(rf"<tier{t}>(.*?)</tier{t}>", html, re.S)
            for title in _tournament_links(m.group(1) if m else ""):
                if (title, t) not in found:
                    found.append((title, t))
        return found

    def team_names(self, ids: list[str]) -> dict[str, str]:
        names: dict[str, str] = {}
        for i in range(0, len(ids), 150):
            chunk = ids[i : i + 150]
            html = self.expand("".join(f"<t>{{{{Team|{x}}}}}</t>" for x in chunk))
            for x, part in zip(chunk, re.findall(r"<t>(.*?)</t>", html, re.S)):
                m = re.search(r'data-highlighting-class="([^"]+)"', part)
                names[x] = m.group(1) if m else x
        return names


def _tournament_links(html: str) -> list[str]:
    titles = []
    for t in re.findall(r"\[\[([^\]|#]+)(?:\|[^\]]*)?\]\]", html):
        t = t.strip().replace("_", " ")
        if t.startswith(("File:", "Category:", "Template:")) or ":" in t[:12]:
            continue
        if "/" in t and t not in titles:
            titles.append(t)
    return titles


# ---------- разбор wikitext ----------


def _blocks(text: str, opener: str) -> list[str]:
    """Все сбалансированные блоки {{opener ... }} в тексте."""
    out, i = [], 0
    pat = re.compile(r"\{\{\s*" + re.escape(opener) + r"\s*(?=[|\n}])")
    while True:
        m = pat.search(text, i)
        if not m:
            return out
        depth, j = 0, m.start()
        while j < len(text) - 1:
            two = text[j : j + 2]
            if two == "{{":
                depth += 1
                j += 2
                continue
            if two == "}}":
                depth -= 1
                j += 2
                if depth == 0:
                    break
                continue
            j += 1
        out.append(text[m.start() : j])
        i = j


def _param(block: str, name: str) -> str:
    m = re.search(r"\|\s*" + re.escape(name) + r"\s*=\s*([^|\n}]*)", block)
    return m.group(1).strip() if m else ""


def _int(v: str) -> int:
    v = re.sub(r"[^\d-]", "", v or "")
    return int(v) if v not in ("", "-") else 0


def parse_date(v: str) -> date | None:
    v = re.sub(r"\{\{.*?\}\}", "", v or "").strip()
    m = re.match(r"([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})", v)
    if m and m.group(1).lower() in MONTHS:
        return date(int(m.group(3)), MONTHS[m.group(1).lower()], int(m.group(2)))
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", v)
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return None


def _opponent(block: str, n: int) -> str:
    m = re.search(rf"\|\s*opponent{n}\s*=\s*\{{\{{\s*TeamOpponent\s*\|\s*([^}}|]+)", block)
    return m.group(1).strip().lower() if m else ""


def _map_score(mb: str) -> tuple[int, int] | None:
    s1, s2 = _param(mb, "score1"), _param(mb, "score2")
    if s1 and s2:
        return _int(s1), _int(s2)
    halves = ["t1t", "t1ct", "t2t", "t2ct"]
    if not all(_param(mb, h) for h in halves):
        return None
    a = _int(_param(mb, "t1t")) + _int(_param(mb, "t1ct"))
    b = _int(_param(mb, "t2t")) + _int(_param(mb, "t2ct"))
    for o in range(1, 10):
        if not _param(mb, f"o{o}t1t"):
            break
        a += _int(_param(mb, f"o{o}t1t")) + _int(_param(mb, f"o{o}t1ct"))
        b += _int(_param(mb, f"o{o}t2t")) + _int(_param(mb, f"o{o}t2ct"))
    return a, b


@dataclass
class ParsedPage:
    maps: list[dict] = field(default_factory=list)
    upcoming: list[dict] = field(default_factory=list)
    rosters: list[dict] = field(default_factory=list)
    media: list[dict] = field(default_factory=list)


def parse_page(title: str, text: str, tier: int, lan: bool | None = None) -> ParsedPage:
    out = ParsedPage()
    if lan is None:
        lan = _param(text, "type").lower() == "offline"
    event = (
        re.sub(r"\{\{DISPLAYTITLE:([^}]*)\}\}.*", r"\1", text[:300], flags=re.S)
        if "DISPLAYTITLE" in text[:300]
        else title
    )
    event = event.strip() or title
    for k, mb in enumerate(_blocks(text, "Match")):
        t1, t2 = _opponent(mb, 1), _opponent(mb, 2)
        d = parse_date(_param(mb, "date"))
        if not (t1 and t2 and d) or "tbd" in (t1, t2):
            continue
        hltv = _param(mb, "hltv")
        key = f"{title}#{k}"
        maps = _blocks(mb, "Map")
        bestof = _int(_param(mb, "bestof")) or len(maps) or 1
        if _param(mb, "finished").lower() != "true":
            out.upcoming.append(
                dict(
                    match_key=key,
                    date=d.isoformat(),
                    time=_param(mb, "date"),
                    team1=t1,
                    team2=t2,
                    bestof=bestof,
                    event=event,
                    tier=tier,
                    lan=lan,
                    hltv=hltv,
                )
            )
            continue
        for n, m in enumerate(maps, 1):
            name = _param(m, "map")
            if not name or _param(m, "finished").lower() != "true":
                continue
            sc = _map_score(m)
            if sc is None or sc[0] == sc[1]:
                continue
            first = _param(m, "t1firstside").lower()
            out.maps.append(
                dict(
                    map_id=f"{key}#{n}",
                    match_key=key,
                    date=d.isoformat(),
                    team1=t1,
                    team2=t2,
                    score1=sc[0],
                    score2=sc[1],
                    map=name.replace("Dust II", "Dust2"),
                    map_number=n,
                    bestof=bestof,
                    event=event,
                    tier=tier,
                    lan=lan,
                    hltv=hltv,
                    # стартовая сторона первой команды и раунды основного времени по сторонам
                    first1=first if first in ("ct", "t") else "",
                    ct1=_param(m, "t1ct"),
                    t1=_param(m, "t1t"),
                    ct2=_param(m, "t2ct"),
                    t2=_param(m, "t2t"),
                )
            )
    for opp in _blocks(text, "Opponent"):
        m = re.match(r"\{\{\s*Opponent\s*\|\s*([^|\n}]+)", opp)
        if not m:
            continue
        persons = re.findall(r"\{\{\s*Person\s*\|([^}]*)\}\}", opp)
        players = [p.split("|")[-1].strip() for p in persons if "role=" not in p]
        notes = re.sub(r"<ref.*?</ref>|<ref[^>]*/>", "", " ".join(_blocks(opp, "Notes")), flags=re.S)
        out.rosters.append(
            dict(
                event=event,
                page=title,
                team_name=m.group(1).strip(),
                players=",".join(players),
                notes=re.sub(r"\s+", " ", notes)[:500],
            )
        )
    for ml in _blocks(text, "ExternalMediaLink"):
        pre = text[max(0, text.find(ml) - 60) : text.find(ml)]
        team = re.findall(r"TeamIcon\|([^}]+)\}\}", pre)
        out.media.append(
            dict(
                event=event,
                date=_param(ml, "date"),
                title=_param(ml, "title"),
                link=_param(ml, "link"),
                team=(team[-1].lower() if team else ""),
                player=_param(ml, "player"),
                source=_param(ml, "of"),
            )
        )
    return out
