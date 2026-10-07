"""Кэфы букмекеров на CS2 через OddsPapi (https://oddspapi.io): Pinnacle, 1xBet, bet365 и ещё 150+ контор.

Бесплатный ключ даёт 250 запросов в месяц, поэтому кэфы берём раз в сутки (утренний запуск сбора)
и только на ближайшие матчи из data/upcoming.csv. Ключ — секрет ODDS_API_KEY в GitHub.
Каталог рынков и список матчей лежат в data/odds_raw/, разобранные кэфы — в data/odds.csv.gz.
Запуск: python -m cs2form.odds (python -m cs2form.odds --reparse — пересобрать из сохранённых ответов)
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

API = "https://api.oddspapi.io/v4"
SPORT_ID = 17  # Counter-Strike
DATA = Path(__file__).resolve().parent.parent / "data"
RAW = DATA / "odds_raw"
ODDS = DATA / "odds.csv.gz"
MAX_CALLS = 8  # за один запуск
MONTHLY_BUDGET = 240  # из 250 бесплатных, с запасом
MORNING_UTC = range(5, 9)  # плановый сбор в 06:00 UTC может стартовать с задержкой
MARKET_TYPES = {
    "moneyline",
    "spreads",
    "totals",
    "spreads-rounds",
    "totals-rounds",
    "teamtotals-rounds-team1",
    "teamtotals-rounds-team2",
}
# биржи, рынки предсказаний и копии линии Pinnacle: ставить там нельзя или это та же линия
SKIP_BOOKS = re.compile(
    r"(-ex$|^polymarket|^kalshi|^sharpxch|^4casters|^ps3838|^pin88|^asports\.bet|^bet487|^ole777|-spb$"
    r"|\.(de|es|fr|gr|it|nl|bg|ca|cz|dk|mx|pe|ro|be|se|ie|pl|rs|lv|lt|ar|br|au|co\.uk|com\.au|bet\.ar|bet\.br)$"
    r"|-nj$)"
)


def norm(name: str) -> str:
    s = str(name).lower()
    s = re.sub(r"\b(team|esports|gaming|club|gg)\b", " ", s)
    return re.sub(r"[^a-z0-9]+", "", s)


ALIASES = {"navi": "natusvincere", "1w": "1win", "faze": "fazeclan", "bb": "betboom"}


def same_team(a: str, b: str) -> bool:
    x, y = norm(a), norm(b)
    x, y = ALIASES.get(x, x), ALIASES.get(y, y)
    return bool(x) and (x == y or (min(len(x), len(y)) >= 4 and (x in y or y in x)))


def catalog(markets: list[dict]) -> dict[int, dict]:
    return {int(m["marketId"]): m for m in markets if isinstance(m, dict) and "marketId" in m}


def flatten(fixture: dict, odds: dict, cat: dict[int, dict]) -> list[dict]:
    """bookmakerOdds → markets → outcomes → players → price в плоские строки с названием рынка и линией."""
    rows = []
    for book, bo in (odds.get("bookmakerOdds") or {}).items():
        if SKIP_BOOKS.search(book) or not (bo or {}).get("bookmakerIsActive", True) or (bo or {}).get("suspended"):
            continue
        for mid, mk in ((bo or {}).get("markets") or {}).items():
            meta = cat.get(int(mid))
            if not meta or meta.get("marketType") not in MARKET_TYPES or not (mk or {}).get("marketActive", True):
                continue
            names = {int(o["outcomeId"]): str(o["outcomeName"]) for o in meta.get("outcomes", [])}
            for oid, oc in ((mk or {}).get("outcomes") or {}).items():
                pl = ((oc or {}).get("players") or {}).get("0") or {}
                if not pl.get("price") or pl.get("active") is False:
                    continue
                rows.append(
                    dict(
                        fixture_id=fixture.get("fixtureId"),
                        start=fixture.get("startTime", ""),
                        p1=fixture.get("participant1Name", ""),
                        p2=fixture.get("participant2Name", ""),
                        tournament=fixture.get("tournamentName", ""),
                        bookmaker=book,
                        market_id=int(mid),
                        market=meta.get("marketType"),
                        period=meta.get("period"),
                        line=float(meta.get("handicap") or 0.0),
                        outcome=names.get(int(oid), str(oid)),
                        price=float(pl["price"]),
                        main=bool(pl.get("mainLine")),
                    )
                )
    return rows


class Client:
    def __init__(self, key: str, state: dict):
        self.key, self.state, self.s = key, state, requests.Session()

    def get(self, path: str, **params):
        if self.state["used"] >= MONTHLY_BUDGET:
            raise RuntimeError(f"исчерпан месячный лимит запросов ({MONTHLY_BUDGET})")
        self.state["used"] += 1
        r = self.s.get(f"{API}{path}", params={"apiKey": self.key, **params}, timeout=30)
        if r.status_code == 429:
            time.sleep(min(60, int(r.headers.get("Retry-After", "5"))))
            r = self.s.get(f"{API}{path}", params={"apiKey": self.key, **params}, timeout=30)
        if r.status_code != 200:  # без URL: в нём ключ
            raise RuntimeError(f"{path}: HTTP {r.status_code} {r.text[:200]}".replace(self.key, "***"))
        return r.json()


def _save(name: str, obj) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / name).write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))


def _fixtures(resp) -> list[dict]:
    if isinstance(resp, dict):
        resp = resp.get("data") or resp.get("fixtures") or []
    return [f for f in resp if isinstance(f, dict)]


def match_upcoming(fixtures: list[dict], up: pd.DataFrame) -> list[tuple[dict, object]]:
    """Матчи OddsPapi, которые есть в нашем списке ближайших; S-tier и ранние первыми."""
    tier = {"s": 0, "a": 1, "b": 2}
    out = []
    for f in fixtures:
        p1, p2 = f.get("participant1Name", ""), f.get("participant2Name", "")
        for r in up.itertuples():
            if (same_team(p1, r.team1) and same_team(p2, r.team2)) or (
                same_team(p1, r.team2) and same_team(p2, r.team1)
            ):
                out.append((tier.get(str(r.tier), 3), str(f.get("startTime", "")), f, r))
                break
    out.sort(key=lambda x: (x[0], x[1]))
    return [(f, r) for _, _, f, r in out]


def write(rows: list[dict]) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(ODDS, index=False)


def reparse() -> int:
    """Пересобирает odds.csv.gz из сохранённых ответов (data/odds_raw/odds_*.json), без запросов к API."""
    cat = catalog(json.loads((RAW / "markets.json").read_text()))
    fx = {f.get("fixtureId"): f for f in _fixtures(json.loads((RAW / "fixtures.json").read_text()))}
    up = pd.read_csv(DATA / "upcoming.csv", dtype=str)
    keys = {f.get("fixtureId"): r.match_key for f, r in match_upcoming(list(fx.values()), up)}
    state = DATA / "odds_state.json"
    fetched = json.loads(state.read_text()).get("last", "") if state.exists() else ""
    rows = []
    for p in sorted(RAW.glob("odds_*.json")):
        od = json.loads(p.read_text())
        f = fx.get(od.get("fixtureId"), od)
        for row in flatten(f, od, cat):
            row.update(match_key=keys.get(od.get("fixtureId"), ""), fetched=fetched)
            rows.append(row)
    write(rows)
    print(f"строк {len(rows)}")
    return 0


def main() -> int:
    if "--reparse" in sys.argv:
        return reparse()
    key = os.environ.get("ODDS_API_KEY", "").strip()
    if not key:
        print("::notice::Кэфы не собираются: нет секрета ODDS_API_KEY", flush=True)
        return 0
    now = datetime.now(timezone.utc)
    forced = os.environ.get("ODDS_FORCE") == "1"
    if not forced and now.hour not in MORNING_UTC:
        print("Кэфы собираются только утренним запуском", flush=True)
        return 0
    state_path = DATA / "odds_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if not forced and state.get("ok") == now.date().isoformat():
        print("Кэфы сегодня уже скачаны", flush=True)
        return 0
    if state.get("month") != now.strftime("%Y-%m"):
        state = {"month": now.strftime("%Y-%m"), "used": 0}
    cl = Client(key, state)
    try:
        if not (RAW / "markets.json").exists():
            _save("markets.json", cl.get("/markets", sportId=SPORT_ID))
        cat = catalog(json.loads((RAW / "markets.json").read_text()))
        fx = _fixtures(
            cl.get(
                "/fixtures",
                sportId=SPORT_ID,
                hasOdds="true",
                **{"from": now.date().isoformat(), "to": (now.date() + timedelta(days=2)).isoformat()},
            )
        )
        _save("fixtures.json", fx)
        up = pd.read_csv(DATA / "upcoming.csv", dtype=str) if (DATA / "upcoming.csv").exists() else pd.DataFrame()
        picked = match_upcoming(fx, up) if not up.empty else []
        rows = []
        for f, r in picked[: MAX_CALLS - 2]:
            fid = f.get("fixtureId")
            od = cl.get("/odds", fixtureId=fid)
            if os.environ.get("ODDS_RAW") == "1":
                _save(f"odds_{fid}.json", od)
            for row in flatten(f, od, cat):
                row.update(match_key=r.match_key, fetched=now.isoformat(timespec="minutes"))
                rows.append(row)
        write(rows)
        if rows:
            state["ok"] = now.date().isoformat()
        print(
            f"::notice::Кэфы: матчей в OddsPapi {len(fx)}, совпало с нашими {len(picked)}, "
            f"скачано {min(len(picked), MAX_CALLS - 2)}, строк {len(rows)}, запросов за месяц {state['used']}",
            flush=True,
        )
    except Exception as e:  # noqa: BLE001
        print(f"::warning::Кэфы не скачались: {str(e).replace(key, '***')}", flush=True)
    finally:
        state["last"] = now.isoformat(timespec="minutes")
        state_path.write_text(json.dumps(state))
    return 0


if __name__ == "__main__":
    sys.exit(main())
