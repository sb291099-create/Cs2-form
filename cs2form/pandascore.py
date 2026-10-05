"""Сбор результатов CS2 через официальный API PandaScore (https://developers.pandascore.co).

Каждая сыгранная карта (game) превращается в строку того же формата, что и у
HLTV-сборщика: две команды, счёт, карта, турнир. На бесплатном тарифе в играх
обычно нет счёта по раундам и названия карты — тогда счёт пишется как 1:0
победителю, а карта остаётся пустой.
"""

from __future__ import annotations

import time
from datetime import date

import requests

API = "https://api.pandascore.co"
GAME_PATH = "csgo"  # PandaScore держит CS2 под старым слагом csgo
DEFAULT_TIERS = ("s", "a", "b")


class PandaScoreError(RuntimeError):
    pass


class PandaScore:
    def __init__(self, token: str, delay: float = 0.5):
        if not token:
            raise PandaScoreError("нет токена PANDASCORE_TOKEN")
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {token}", "Accept": "application/json"})
        self.delay = delay

    def get(self, path: str, params: dict) -> list[dict]:
        for attempt in range(1, 4):
            time.sleep(self.delay)
            r = self.s.get(API + path, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(30 * attempt)
                continue
            if r.status_code != 200:
                raise PandaScoreError(f"{path}: HTTP {r.status_code} {r.text[:200]}")
            return r.json()
        raise PandaScoreError(f"{path}: лимит запросов (429)")

    def past_matches(self, start: date, end: date, max_pages: int = 200) -> list[dict]:
        out: list[dict] = []
        for page in range(1, max_pages + 1):
            batch = self.get(
                f"/{GAME_PATH}/matches/past",
                {
                    "range[begin_at]": f"{start.isoformat()}T00:00:00Z,{end.isoformat()}T23:59:59Z",
                    "sort": "-begin_at",
                    "page[size]": 100,
                    "page[number]": page,
                },
            )
            out.extend(batch)
            print(f"  страница {page}: {len(batch)} матчей", flush=True)
            if len(batch) < 100:
                break
        return out


def _tier(match: dict) -> str:
    return ((match.get("tournament") or {}).get("tier") or "").lower()


def games_to_rows(matches: list[dict], tiers: tuple[str, ...] | None = DEFAULT_TIERS) -> list[dict]:
    """Превращает матчи PandaScore в строки «одна карта = одна строка»."""
    rows = []
    for m in matches:
        if tiers and _tier(m) not in tiers:
            continue
        opps = [o.get("opponent") or {} for o in m.get("opponents") or []]
        if len(opps) != 2 or not all(o.get("id") for o in opps):
            continue
        t1, t2 = opps
        event = " ".join(
            x for x in ((m.get("league") or {}).get("name"), (m.get("serie") or {}).get("full_name")) if x
        ) or (m.get("tournament") or {}).get("name", "")
        for g in m.get("games") or []:
            winner = (g.get("winner") or {}).get("id")
            if not g.get("finished") or winner not in (t1["id"], t2["id"]) or g.get("forfeit"):
                continue
            s1, s2 = _round_score(g, t1["id"], t2["id"])
            if s1 is None:
                s1, s2 = (1, 0) if winner == t1["id"] else (0, 1)
            played = g.get("begin_at") or m.get("begin_at") or ""
            rows.append(
                dict(
                    map_id=g["id"],
                    date=played[:10],
                    team1_id=t1["id"],
                    team1=t1.get("name", ""),
                    team2_id=t2["id"],
                    team2=t2.get("name", ""),
                    score1=s1,
                    score2=s2,
                    map=((g.get("map") or {}).get("name") or ""),
                    event=event,
                    tier=_tier(m),
                )
            )
    return rows


def _round_score(game: dict, id1: int, id2: int):
    """Счёт по раундам, если тариф его отдаёт (поле teams[].score или rounds_score)."""
    scores = {}
    for t in game.get("teams") or []:
        tid = (t.get("team") or {}).get("id") or t.get("team_id")
        if tid is not None and t.get("score") is not None:
            scores[tid] = t["score"]
    for t in game.get("rounds_score") or []:
        if t.get("team_id") is not None and t.get("score") is not None:
            scores[t["team_id"]] = t["score"]
    if id1 in scores and id2 in scores and scores[id1] != scores[id2]:
        return int(scores[id1]), int(scores[id2])
    return None, None
