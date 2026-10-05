"""Обновляет data/*.csv: результаты карт из PandaScore, рейтинг из HLTV.

Запуск: PANDASCORE_TOKEN=... python -m cs2form.update
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from .pandascore import PandaScore, PandaScoreError, games_to_rows
from .scraper import Fetcher, as_dicts, fetch_ranking

DATA = Path(__file__).resolve().parent.parent / "data"
MAPS_CSV = DATA / "maps.csv"
RANKING_CSV = DATA / "ranking.csv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=180, help="глубина истории при первом запуске")
    ap.add_argument("--overlap", type=int, default=3, help="сколько последних дней перекачивать")
    args = ap.parse_args()

    today = datetime.now(timezone.utc).date()
    old = pd.read_csv(MAPS_CSV) if MAPS_CSV.exists() else pd.DataFrame()
    if old.empty or "tier" not in old:
        old = pd.DataFrame()
        start = today - timedelta(days=args.days)
    else:
        start = date.fromisoformat(old["date"].max()) - timedelta(days=args.overlap)

    print(f"Матчи с {start} по {today}", flush=True)
    # ::notice:: / ::error:: превращаются в аннотации на странице запуска в GitHub
    try:
        matches = PandaScore(os.environ.get("PANDASCORE_TOKEN", "")).past_matches(start, today)
    except PandaScoreError as e:
        print(f"::error::PandaScore: {e}", flush=True)
        return 1
    new = pd.DataFrame(games_to_rows(matches))
    games = [g for m in matches for g in (m.get("games") or [])]
    if games:
        print(f"::notice::Поля карты в PandaScore: {sorted(games[0].keys())}", flush=True)
    if new.empty:
        print(f"::error::PandaScore вернул {len(matches)} матчей, но ни одной сыгранной карты тиров S/A/B", flush=True)
        return 1

    try:
        ranking = pd.DataFrame(as_dicts(fetch_ranking(Fetcher(retries=2))))
    except Exception as e:  # рейтинг HLTV — приятное дополнение, без него тоже работаем
        print(f"::warning::Рейтинг HLTV не скачался: {e}", flush=True)
        ranking = pd.DataFrame()

    maps = pd.concat([old, new]).drop_duplicates("map_id", keep="last").sort_values(["date", "map_id"])
    maps = maps[maps["date"] >= (today - timedelta(days=args.days)).isoformat()]
    DATA.mkdir(exist_ok=True)
    maps.to_csv(MAPS_CSV, index=False)
    if not ranking.empty:
        ranking.to_csv(RANKING_CSV, index=False)
    with_rounds = (new[["score1", "score2"]].max(axis=1) > 1).mean() * 100
    with_map = (new["map"].fillna("") != "").mean() * 100
    print(
        f"::notice::Готово: {len(matches)} матчей, {len(new)} карт (со счётом по раундам {with_rounds:.0f}%, "
        f"с названием карты {with_map:.0f}%), всего в базе {len(maps)}; рейтинг HLTV: {len(ranking)} команд",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
