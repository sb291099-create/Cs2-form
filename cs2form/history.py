"""Прошлые линии Pinnacle и других контор на матчи CS2 из OddsPapi — чтобы проверить модель и ставки на истории.

/v4/historical-odds бесплатен (не тратит месячный лимит), /v4/fixtures стоит 1 запрос на окно до 10 дней.
Для каждого прошлого матча из data/maps.csv берём линию на момент утренней загрузки (06:15 UTC в день
матча, как у ежедневного сбора) и последнюю перед началом. Результат: data/odds_history.csv.gz.
Запуск: python -m cs2form.history --days 90 [--raw 2]
Другие конторы — на матчи, уже скачанные с линией Pinnacle, без платных запросов:
python -m cs2form.history --books fonbet,marathonbet,1xbet,stake,bet365,melbet
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
BOOKS_PER_CALL = 3  # больше /historical-odds за раз не отдаёт
PAUSE = 0.15  # у odds-эндпоинтов лимит 10 запросов в секунду
RETRIES = 3


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
                opened=q[0][0].isoformat(),
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


def _flat(obj, prefix="") -> dict:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out.update(_flat(v, f"{prefix}{k}."))
        return out
    return {prefix.rstrip("."): obj}


def _is_quota(name: str) -> bool:
    n = name.lower()
    return any(w in n for w in ("request", "limit", "quota", "used", "remaining"))


def _get_hist(cl: Client, fid: str, books: str):
    """Ответ /historical-odds; None — у этих контор не было линии на матч. На 429 ждём и пробуем снова."""
    for i in range(RETRIES):
        try:
            return cl.get("/historical-odds", free=True, fixtureId=fid, bookmakers=books)
        except RuntimeError as e:
            if "HTTP 404" in str(e):
                return None
            if "HTTP 429" not in str(e) or i == RETRIES - 1:
                raise
            time.sleep(10 * (i + 1))
    return None


def _known(old: pd.DataFrame) -> list[tuple[dict, str, bool]]:
    """Матчи, уже скачанные с линией Pinnacle, в том же виде, что и match_past: без платного /fixtures."""
    out = []
    for r in old.drop_duplicates("fixture_id").itertuples():
        f = dict(
            fixtureId=r.fixture_id,
            startTime=r.start,
            participant1Name=r.p1,
            participant2Name=r.p2,
            tournamentName=r.tournament if isinstance(r.tournament, str) else "",
        )
        out.append((f, r.match_key, bool(r.p1_is_team1)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Прошлые линии контор на матчи из maps.csv")
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--raw", type=int, default=0, help="сколько сырых ответов сохранить для проверки разбора")
    ap.add_argument(
        "--books", default=BOOKMAKERS, help="конторы через запятую; кроме Pinnacle — на уже скачанные матчи"
    )
    ap.add_argument("--minutes", type=float, default=0, help="через сколько минут остановиться и сохранить скачанное")
    args = ap.parse_args()
    key = os.environ.get("ODDS_API_KEY", "").strip()
    if not key:
        print("::notice::Нет секрета ODDS_API_KEY", flush=True)
        return 0
    books = [b.strip().lower() for b in args.books.split(",") if b.strip()] or [BOOKMAKERS]
    chunks = [",".join(books[i : i + BOOKS_PER_CALL]) for i in range(0, len(books), BOOKS_PER_CALL)]
    deadline = time.monotonic() + args.minutes * 60 if args.minutes else None
    now = datetime.now(timezone.utc)
    state_path = DATA / "odds_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if state.get("month") != now.strftime("%Y-%m"):
        state = {"month": now.strftime("%Y-%m"), "used": 0}
    cl = Client(key, state)
    maps = pd.read_csv(DATA / "maps.csv", dtype=str)
    cat = catalog(json.loads((RAW / "markets.json").read_text()))
    old = pd.read_csv(HIST, dtype={"fixture_id": str, "match_key": str}) if HIST.exists() else pd.DataFrame()
    done = set(zip(old["fixture_id"], old["book"])) if len(old) else set()
    rows, matched, errors, missing, saved, fixtures, per_window, priced = [], 0, 0, 0, 0, [], [], []
    calls, stopped, quota = 0, False, ""
    try:  # только счётчики запросов: в ответе есть почта, а логи Actions публичные
        acc = cl.get("/account", free=True)
        nums = {k: v for k, v in _flat(acc).items() if isinstance(v, (int, float)) and _is_quota(k)}
        quota = ", ".join(f"{k}={v}" for k, v in list(nums.items())[:6])
    except Exception as e:  # noqa: BLE001
        print(f"::warning::/account: {str(e).replace(key, '***')[:200]}", flush=True)
    try:
        if books == [BOOKMAKERS]:
            end = now.date()
            for i in range(0, args.days, WINDOW_DAYS):
                to = end - timedelta(days=i)
                frm = max(to - timedelta(days=WINDOW_DAYS - 1), end - timedelta(days=args.days))
                window = cl.get("/fixtures", sportId=SPORT_ID, **{"from": frm.isoformat(), "to": to.isoformat()})
                fx = _fixtures(window)
                fixtures += fx
                per_window.append(len(fx))
            priced = [f for f in fixtures if (f.get("externalProviders") or {}).get("pinnacleId")]
            past = match_past([f for f in priced if (_ts(f.get("startTime")) or now) < now], maps)
        else:
            past = _known(old)
        matched = len(past)
        SAMPLES.mkdir(parents=True, exist_ok=True)
        for chunk in chunks:  # сначала первая тройка контор по всем матчам: при остановке по времени она полная
            for f, match_key, p1_is_team1 in past:
                fid = f.get("fixtureId")
                if all((fid, b) in done for b in chunk.split(",")):
                    continue
                if deadline and time.monotonic() > deadline:
                    stopped = True
                    break
                try:
                    calls += 1
                    h = _get_hist(cl, fid, chunk)
                except Exception as e:  # noqa: BLE001
                    errors += 1
                    print(f"::warning::{fid}: {str(e).replace(key, '***')[:200]}", flush=True)
                    if errors >= 5:
                        break
                    continue
                if h is None:
                    missing += 1
                    continue
                if saved < args.raw:
                    saved += 1
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
            if stopped or errors >= 5:
                break
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
    by_book = new.groupby("book")["fixture_id"].nunique().to_dict() if len(new) else {}
    total = out["fixture_id"].nunique() if len(out) else 0
    print(
        f"::notice::История ({args.books}): матчей OddsPapi {len(fixtures)} (по окнам {per_window}), "
        f"с линией Pinnacle {len(priced)}, к скачиванию {matched}, запросов истории {calls}, без линии {missing}, "
        f"новых строк {len(new)} (матчей по конторам {by_book}), всего матчей {total}, "
        f"ошибок {errors}{', остановлено по времени' if stopped else ''}, запросов за месяц {state['used']}"
        + (f"; OddsPapi: {quota}" if quota else ""),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
