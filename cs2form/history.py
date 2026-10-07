"""Прошлые линии Pinnacle на матчи CS2 из OddsPapi — чтобы проверить модель на истории.

/v4/historical-odds бесплатен (не тратит месячный лимит), /v4/fixtures стоит 1 запрос на окно до 10 дней.
Для каждого прошлого матча из data/maps.csv берём линию Pinnacle на момент утренней загрузки (06:15 UTC в день
матча, как у ежедневного сбора) и последнюю перед началом. Результат: data/odds_history.csv.gz.
Запуск: python -m cs2form.history --days 90 [--raw 2]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from .odds import DATA, RAW, SPORT_ID, Client, _fixtures, catalog, same_team

HIST = DATA / "odds_history.csv.gz"
SAMPLES = DATA / "history_raw"
WINDOW_DAYS = 10  # больше /fixtures за раз не отдаёт
FETCH_AT = timedelta(hours=6, minutes=15)
BOOKMAKERS = "pinnacle"
PAUSE = 0.15  # у odds-эндпоинтов лимит 10 запросов в секунду


def wanted(meta: dict) -> bool:
    """Победа в матче и на картах, форы ±1.5 по картам, тотал карт 2.5."""
    t, per, h = meta.get("marketType"), meta.get("period"), float(meta.get("handicap") or 0)
    return (
        (t == "moneyline" and per in ("result", "p1", "p2", "p3"))
        or (t == "spreads" and per == "result" and abs(h) == 1.5)
        or (t == "totals" and per == "result" and h == 2.5)
    )


def _ts(x) -> pd.Timestamp | None:
    """createdAt (ISO) или changedAt / ключ (миллисекунды эпохи) → UTC."""
    if x is None:
        return None
    try:
        v = float(x)
        return pd.Timestamp(v / 1000 if v > 1e11 else v, unit="s", tz="UTC")
    except (TypeError, ValueError):
        t = pd.Timestamp(str(x))
        return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def timelines(obj, path=()) -> list[tuple[tuple, list[dict]]]:
    """Все ряды котировок в ответе: список или словарь записей с price, вместе с путём ключей до них."""
    out = []
    if isinstance(obj, list):
        if obj and all(isinstance(q, dict) and "price" in q for q in obj):
            return [(path, obj)]
        for i, q in enumerate(obj):
            out += timelines(q, path + (str(i),))
    elif isinstance(obj, dict):
        vals = list(obj.values())
        if vals and all(isinstance(q, dict) and "price" in q for q in vals):
            return [(path, [{**q, "_key": k} for k, q in obj.items()])]
        for k, v in obj.items():
            out += timelines(v, path + (str(k),))
    return out


def _after(path: tuple, name: str) -> str | None:
    return path[path.index(name) + 1] if name in path and path.index(name) + 1 < len(path) else None


def parse(hist: dict, cat: dict[int, dict], start: pd.Timestamp) -> list[dict]:
    """Кэф на утреннюю загрузку и последний до начала по каждому нужному исходу."""
    fetch = start.normalize() + FETCH_AT
    if fetch >= start:
        fetch = start - timedelta(hours=1)
    rows = []
    for path, quotes in timelines(hist):
        mid, oid = _after(path, "markets"), _after(path, "outcomes")
        book = _after(path, "bookmakerOdds") or _after(path, "bookmakers") or BOOKMAKERS
        if mid is None or oid is None or not mid.isdigit() or int(mid) not in cat:
            continue
        meta = cat[int(mid)]
        if not wanted(meta):
            continue
        names = {str(o["outcomeId"]): str(o["outcomeName"]) for o in meta.get("outcomes", [])}
        q = []
        for x in quotes:
            t = _ts(x.get("createdAt") or x.get("changedAt") or x.get("_key"))
            if t is not None and x.get("price") and x.get("active", True) is not False:
                q.append((t, float(x["price"])))
        q.sort()
        before = [p for t, p in q if t < start]
        if not before:
            continue
        at_fetch = [p for t, p in q if t <= fetch]
        rows.append(
            dict(
                book=book,
                market_id=int(mid),
                market=meta.get("marketType"),
                period=meta.get("period"),
                line=float(meta.get("handicap") or 0.0),
                outcome=names.get(oid, oid),
                price_fetch=at_fetch[-1] if at_fetch else before[0],
                price_close=before[-1],
                quotes=len(before),
            )
        )
    return rows


def match_past(fixtures: list[dict], maps: pd.DataFrame) -> list[tuple[dict, str, bool]]:
    """Прошлые матчи OddsPapi, которые есть в maps.csv: (матч, match_key, participant1 — это team1)."""
    games = maps.groupby("match_key").agg(date=("date", "min"), team1=("team1", "first"), team2=("team2", "first"))
    games["date"] = pd.to_datetime(games["date"])
    out = []
    for f in fixtures:
        start = _ts(f.get("startTime"))
        if start is None:
            continue
        day = start.tz_localize(None).normalize()
        near = games[(games["date"] - day).abs() <= pd.Timedelta(days=1)]
        p1, p2 = f.get("participant1Name", ""), f.get("participant2Name", "")
        for key, g in near.iterrows():
            if same_team(p1, g.team1) and same_team(p2, g.team2):
                out.append((f, key, True))
                break
            if same_team(p1, g.team2) and same_team(p2, g.team1):
                out.append((f, key, False))
                break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Прошлые линии Pinnacle на матчи из maps.csv")
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--raw", type=int, default=0, help="сколько сырых ответов сохранить для проверки разбора")
    args = ap.parse_args()
    key = os.environ.get("ODDS_API_KEY", "").strip()
    if not key:
        print("::notice::Нет секрета ODDS_API_KEY", flush=True)
        return 0
    now = datetime.now(timezone.utc)
    state_path = DATA / "odds_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if state.get("month") != now.strftime("%Y-%m"):
        state = {"month": now.strftime("%Y-%m"), "used": 0}
    cl = Client(key, state)
    maps = pd.read_csv(DATA / "maps.csv", dtype=str)
    cat = catalog(json.loads((RAW / "markets.json").read_text()))
    old = pd.read_csv(HIST) if HIST.exists() else pd.DataFrame()
    done = set(old["fixture_id"]) if len(old) else set()
    rows, matched, errors, fixtures = [], 0, 0, []
    try:
        acc = cl.get("/account", free=True)
        print(f"::notice::Аккаунт OddsPapi: {json.dumps(acc, ensure_ascii=False)[:300]}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"::warning::/account: {str(e).replace(key, '***')}", flush=True)
    try:
        end = now.date()
        for i in range(0, args.days, WINDOW_DAYS):
            to = end - timedelta(days=i)
            frm = max(to - timedelta(days=WINDOW_DAYS - 1), end - timedelta(days=args.days))
            fx = _fixtures(cl.get("/fixtures", sportId=SPORT_ID, **{"from": frm.isoformat(), "to": to.isoformat()}))
            fixtures += fx
            print(f"окно {frm}..{to}: матчей {len(fx)}", flush=True)
        past = match_past([f for f in fixtures if (_ts(f.get("startTime")) or now) < now], maps)
        matched = len(past)
        SAMPLES.mkdir(parents=True, exist_ok=True)
        for n, (f, match_key, p1_is_team1) in enumerate(past):
            fid = f.get("fixtureId")
            if fid in done:
                continue
            try:
                h = cl.get("/historical-odds", free=True, fixtureId=fid, bookmakers=BOOKMAKERS)
            except Exception as e:  # noqa: BLE001
                errors += 1
                print(f"::warning::{fid}: {str(e).replace(key, '***')[:200]}", flush=True)
                if errors >= 5:
                    break
                continue
            if n < args.raw:
                (SAMPLES / f"hist_{fid}.json").write_text(json.dumps(h, separators=(",", ":"))[:400_000])
            start = _ts(f.get("startTime"))
            for r in parse(h, cat, start):
                r.update(
                    fixture_id=fid,
                    match_key=match_key,
                    start=start.isoformat(),
                    p1=f.get("participant1Name", ""),
                    p2=f.get("participant2Name", ""),
                    p1_is_team1=p1_is_team1,
                    tournament=f.get("tournamentName", ""),
                )
                rows.append(r)
            time.sleep(PAUSE)
    except Exception as e:  # noqa: BLE001
        print(f"::warning::История кэфов прервалась: {str(e).replace(key, '***')}", flush=True)
    finally:
        state["last_history"] = now.isoformat(timespec="minutes")
        state_path.write_text(json.dumps(state))
    if fixtures and args.raw:
        (SAMPLES / "fixtures_sample.json").write_text(json.dumps(fixtures[:5], separators=(",", ":")))
    new = pd.DataFrame(rows)
    out = pd.concat([old, new], ignore_index=True) if len(old) else new
    if len(out):
        out.to_csv(HIST, index=False)
    print(
        f"::notice::История: матчей OddsPapi {len(fixtures)}, совпало с maps.csv {matched}, "
        f"новых строк {len(new)}, всего матчей с линией {out['fixture_id'].nunique() if len(out) else 0}, "
        f"ошибок {errors}, запросов за месяц {state['used']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
