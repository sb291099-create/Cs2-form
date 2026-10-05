"""Сбор результатов карт и рейтинга команд с HLTV.

Источник результатов: https://www.hltv.org/stats/matches — каждая строка
таблицы это одна сыгранная карта (две команды, счёт по раундам, карта, турнир).
"""
from __future__ import annotations

import random
import re
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone

from bs4 import BeautifulSoup

BASE = "https://www.hltv.org"
PAGE_SIZE = 50


@dataclass
class MapResult:
    map_id: int
    date: str  # YYYY-MM-DD (UTC)
    team1_id: int
    team1: str
    team2_id: int
    team2: str
    score1: int
    score2: int
    map: str
    event: str


@dataclass
class RankedTeam:
    rank: int
    team: str
    points: int


class BlockedError(RuntimeError):
    """HLTV/Cloudflare не отдал страницу."""


class Fetcher:
    """HTTP-клиент, который выглядит как браузер и не торопится."""

    def __init__(self, delay: tuple[float, float] = (2.5, 5.0), retries: int = 3):
        from curl_cffi import requests

        self.session = requests.Session(impersonate="chrome")
        self.delay = delay
        self.retries = retries
        self._last = 0.0

    def get(self, path: str) -> str:
        url = path if path.startswith("http") else BASE + path
        for attempt in range(1, self.retries + 1):
            wait = self._last + random.uniform(*self.delay) - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            resp = self.session.get(url, timeout=30)
            if resp.status_code == 200 and not _is_challenge(resp.text):
                return resp.text
            if resp.status_code == 404:
                raise RuntimeError(f"404 для {url}")
            time.sleep(10 * attempt)
        raise BlockedError(f"HLTV не отдал {url} (последний статус {resp.status_code})")


def _is_challenge(html: str) -> bool:
    head = html[:5000].lower()
    return "just a moment" in head or "cf-chl" in head or "challenge-platform" in head


_ID_RE = re.compile(r"/(\d+)/")
_INT_RE = re.compile(r"-?\d+")


def _id_from_href(href: str) -> int:
    m = _ID_RE.search(href or "")
    if not m:
        raise ValueError(f"нет id в ссылке {href!r}")
    return int(m.group(1))


def _int(text: str) -> int:
    m = _INT_RE.search(text or "")
    if not m:
        raise ValueError(f"нет числа в {text!r}")
    return int(m.group(0))


def parse_map_results(html: str) -> list[MapResult]:
    soup = BeautifulSoup(html, "lxml")
    table = soup.select_one("table.matches-table")
    if table is None:
        raise ValueError("на странице нет таблицы matches-table (изменилась вёрстка HLTV?)")
    out: list[MapResult] = []
    for tr in table.select("tbody tr"):
        link = tr.select_one("td.date-col a")
        teams = tr.select("td.team-col")
        if link is None or len(teams) != 2:
            continue
        time_div = tr.select_one("td.date-col [data-unix]")
        if time_div is not None:
            ts = int(time_div["data-unix"]) / 1000
            day = datetime.fromtimestamp(ts, tz=timezone.utc).date()
        else:
            day = datetime.strptime(link.get_text(strip=True), "%d/%m/%y").date()
        t = []
        for td in teams:
            a = td.select_one("a")
            score = td.select_one(".score")
            t.append((_id_from_href(a["href"]), a.get_text(strip=True), _int(score.get_text())))
        map_name_el = tr.select_one(".dynamic-map-name-full") or tr.select_one("td.statsDetail")
        event_el = tr.select_one("td.event-col")
        out.append(
            MapResult(
                map_id=_id_from_href(link["href"].replace("/mapstatsid/", "/")),
                date=day.isoformat(),
                team1_id=t[0][0],
                team1=t[0][1],
                team2_id=t[1][0],
                team2=t[1][1],
                score1=t[0][2],
                score2=t[1][2],
                map=map_name_el.get_text(strip=True) if map_name_el else "",
                event=event_el.get_text(strip=True) if event_el else "",
            )
        )
    return out


def parse_ranking(html: str) -> list[RankedTeam]:
    soup = BeautifulSoup(html, "lxml")
    out = []
    for box in soup.select("div.ranked-team"):
        pos = box.select_one(".position")
        name = box.select_one(".teamLine .name") or box.select_one(".name")
        pts = box.select_one(".points")
        if not (pos and name):
            continue
        out.append(RankedTeam(_int(pos.get_text()), name.get_text(strip=True), _int(pts.get_text()) if pts else 0))
    return out


def fetch_map_results(
    fetcher: Fetcher, start: date, end: date, ranking_filter: str = "Top50", max_pages: int = 400
) -> list[MapResult]:
    results: list[MapResult] = []
    for page in range(max_pages):
        path = (
            f"/stats/matches?startDate={start.isoformat()}&endDate={end.isoformat()}"
            f"&rankingFilter={ranking_filter}&offset={page * PAGE_SIZE}"
        )
        rows = parse_map_results(fetcher.get(path))
        results.extend(rows)
        print(f"  страница {page + 1}: {len(rows)} карт", flush=True)
        if len(rows) < PAGE_SIZE:
            break
    return results


def fetch_ranking(fetcher: Fetcher) -> list[RankedTeam]:
    return parse_ranking(fetcher.get("/ranking/teams"))


def as_dicts(items) -> list[dict]:
    return [asdict(i) for i in items]
