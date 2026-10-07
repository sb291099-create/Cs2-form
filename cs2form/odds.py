"""Кэфы букмекеров на CS2 через OddsPapi (https://oddspapi.io): Pinnacle, 1xBet и ещё 150+ контор.

Бесплатный ключ даёт 250 запросов в месяц, поэтому кэфы берём раз в сутки (утренний запуск сбора)
и только на ближайшие матчи из data/upcoming.csv. Ключ — секрет ODDS_API_KEY в GitHub.
Сырые ответы лежат в data/odds_raw/, разобранные кэфы — в data/odds.csv.
Запуск: python -m cs2form.odds
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
MAX_CALLS = 8  # за один запуск
MONTHLY_BUDGET = 240  # из 250 бесплатных, с запасом
MORNING_UTC = range(5, 9)  # плановый сбор в 06:00 UTC может стартовать с задержкой


def norm(name: str) -> str:
    s = str(name).lower()
    s = re.sub(r"\b(team|esports|gaming|club|gg)\b", " ", s)
    return re.sub(r"[^a-z0-9]+", "", s)


ALIASES = {"navi": "natusvincere", "1w": "1win", "faze": "fazeclan", "bb": "betboom"}


def same_team(a: str, b: str) -> bool:
    x, y = norm(a), norm(b)
    x, y = ALIASES.get(x, x), ALIASES.get(y, y)
    return bool(x) and (x == y or (min(len(x), len(y)) >= 4 and (x in y or y in x)))


def flatten(fixture_id, odds: dict) -> list[dict]:
    """bookmakerOdds → markets → outcomes → players → price в плоские строки."""
    rows = []
    for book, bo in (odds.get("bookmakerOdds") or {}).items():
        for mid, mk in ((bo or {}).get("markets") or {}).items():
            for oid, oc in ((mk or {}).get("outcomes") or {}).items():
                for _, pl in ((oc or {}).get("players") or {}).items():
                    price = (pl or {}).get("price")
                    if price:
                        rows.append(
                            dict(
                                fixture_id=fixture_id,
                                team1=odds.get("participant1Name", ""),
                                team2=odds.get("participant2Name", ""),
                                bookmaker=book,
                                market_id=str(mid),
                                outcome_id=str(oid),
                                price=float(price),
                                active=(pl or {}).get("active", True),
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


def main() -> int:
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
    if state.get("month") != now.strftime("%Y-%m"):
        state = {"month": now.strftime("%Y-%m"), "used": 0}
    cl = Client(key, state)
    try:
        if not (RAW / "markets.json").exists():
            _save("markets.json", cl.get("/markets", sportId=SPORT_ID))
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
        tier = {"s": 0, "a": 1, "b": 2}
        picked = []
        for f in fx:
            p1, p2 = f.get("participant1Name", ""), f.get("participant2Name", "")
            hit = [
                r
                for r in up.itertuples()
                if (same_team(p1, r.team1) and same_team(p2, r.team2))
                or (same_team(p1, r.team2) and same_team(p2, r.team1))
            ]
            if hit:
                picked.append((tier.get(str(hit[0].tier), 3), str(f.get("startTime", "")), f, hit[0]))
        picked.sort(key=lambda x: (x[0], x[1]))
        rows = []
        for old in RAW.glob("odds_*.json"):  # храним только последний сбор
            old.unlink()
        for _, _, f, r in picked[: MAX_CALLS - 2]:
            fid = f.get("fixtureId") or f.get("id")
            od = cl.get("/odds", fixtureId=fid)
            _save(f"odds_{fid}.json", od)
            for row in flatten(fid, od):
                row.update(
                    match_key=r.match_key, start=f.get("startTime", ""), fetched=now.isoformat(timespec="minutes")
                )
                rows.append(row)
        if rows:
            pd.DataFrame(rows).to_csv(DATA / "odds.csv", index=False)
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
