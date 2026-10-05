"""Обновляет data/*.csv свежими данными с HLTV. Запуск: python -m cs2form.update"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from .scraper import BlockedError, Fetcher, as_dicts, fetch_map_results, fetch_ranking

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
    if old.empty:
        start = today - timedelta(days=args.days)
    else:
        start = date.fromisoformat(old["date"].max()) - timedelta(days=args.overlap)

    fetcher = Fetcher()
    print(f"Карты с {start} по {today}", flush=True)
    try:
        new = pd.DataFrame(as_dicts(fetch_map_results(fetcher, start, today)))
        ranking = pd.DataFrame(as_dicts(fetch_ranking(fetcher)))
    except BlockedError as e:
        print(f"ОШИБКА: {e}. Похоже, Cloudflare блокирует запросы с этого сервера.", file=sys.stderr)
        return 1

    maps = pd.concat([old, new]).drop_duplicates("map_id", keep="last").sort_values(["date", "map_id"])
    cutoff = (today - timedelta(days=args.days)).isoformat()
    maps = maps[maps["date"] >= cutoff]
    DATA.mkdir(exist_ok=True)
    maps.to_csv(MAPS_CSV, index=False)
    if not ranking.empty:
        ranking.to_csv(RANKING_CSV, index=False)
    print(f"Готово: {len(new)} карт скачано, всего {len(maps)}; рейтинг: {len(ranking)} команд")
    return 0


if __name__ == "__main__":
    sys.exit(main())
