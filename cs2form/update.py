"""Обновляет data/*.csv: карты, ближайшие матчи, составы и медиа из Liquipedia, рейтинг из HLTV.

Запуск: python -m cs2form.update [--full]
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from .liquipedia import Liquipedia, parse_page
from .scraper import Fetcher, as_dicts, fetch_ranking

DATA = Path(__file__).resolve().parent.parent / "data"
TIER_NAMES = {1: "s", 2: "a", 3: "b"}


def _merge(path: Path, new: pd.DataFrame, key, refreshed_pages: set[str] | None = None,
           keep_old: bool = True) -> pd.DataFrame:
    old = pd.read_csv(path, dtype=str) if path.exists() and keep_old else pd.DataFrame()
    if not old.empty and refreshed_pages is not None and "page" in old:
        old = old[~old["page"].isin(refreshed_pages)]  # перекачанные турниры заменяем целиком
    df = pd.concat([old, new.astype(str)]) if not new.empty else old
    return df.drop_duplicates(key, keep="last") if not df.empty else df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=180, help="глубина истории")
    ap.add_argument("--full", action="store_true", help="перекачать всю историю")
    args = ap.parse_args()

    today = datetime.now(timezone.utc).date()
    full = args.full or not (DATA / "maps.csv").exists() or "page" not in pd.read_csv(DATA / "maps.csv", nrows=1)
    # турниры могут длиться месяц, поэтому даже в ежедневном режиме смотрим 45 дней назад
    start = today - timedelta(days=args.days if full else 45)
    lp = Liquipedia()
    print(f"Турниры с {start} по {today + timedelta(days=14)} ({'полный' if full else 'ежедневный'} режим)", flush=True)
    tournaments = lp.tournaments(start, today + timedelta(days=14))
    print(f"::notice::Турниров найдено: {len(tournaments)}", flush=True)

    maps, upcoming, rosters, media = [], [], [], []
    refreshed: set[str] = set()
    for i, (title, tier) in enumerate(tournaments, 1):
        try:
            pages = [title] + lp.subpages(title)
            texts = lp.wikitext(pages)
        except Exception as e:  # noqa: BLE001
            print(f"  пропуск {title}: {e}", flush=True)
            continue
        main_text = texts.get(title, "")
        lan = "|type=offline" in main_text.lower().replace(" ", "")
        event = title
        for p, text in texts.items():
            parsed = parse_page(p, text, tier, lan=lan)
            if p == title:
                event = parsed.maps[0]["event"] if parsed.maps else title
            for rows, part in ((maps, parsed.maps), (upcoming, parsed.upcoming), (rosters, parsed.rosters),
                               (media, parsed.media)):
                for r in part:
                    r["page"] = title
                    r["event"] = event if p != title else r["event"]
                    rows.append(r)
        refreshed.add(title)
        print(f"  [{i}/{len(tournaments)}] {title}: страниц {len(texts)}, карт всего {len(maps)}", flush=True)

    ids = sorted({m["team1"] for m in maps + upcoming} | {m["team2"] for m in maps + upcoming})
    names_path = DATA / "team_names.csv"
    known = pd.read_csv(names_path, dtype=str).set_index("id")["name"].to_dict() if names_path.exists() else {}
    missing = [x for x in ids if x not in known]
    if missing:
        known.update(lp.team_names(missing))
    DATA.mkdir(exist_ok=True)
    pd.DataFrame(sorted(known.items()), columns=["id", "name"]).to_csv(names_path, index=False)

    def with_names(rows):
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df = df.rename(columns={"team1": "team1_id", "team2": "team2_id"})
        df["team1"] = df["team1_id"].map(lambda x: known.get(x, x))
        df["team2"] = df["team2_id"].map(lambda x: known.get(x, x))
        df["tier"] = df["tier"].map(TIER_NAMES)
        return df

    cutoff = (today - timedelta(days=args.days)).isoformat()
    all_maps = _merge(DATA / "maps.csv", with_names(maps), "map_id", refreshed, keep_old=not full)
    all_maps = all_maps[all_maps["date"] >= cutoff].sort_values(["date", "map_id"])
    all_maps.to_csv(DATA / "maps.csv", index=False)

    up = with_names(upcoming)
    if not up.empty:
        up = up[up["date"] >= today.isoformat()].sort_values(["date", "time"])
    up.to_csv(DATA / "upcoming.csv", index=False)

    rost = _merge(DATA / "rosters.csv", pd.DataFrame(rosters), ["page", "team_name"])
    rost.to_csv(DATA / "rosters.csv", index=False)
    med = _merge(DATA / "media.csv", pd.DataFrame(media), "link")
    med.to_csv(DATA / "media.csv", index=False)

    try:
        ranking = pd.DataFrame(as_dicts(fetch_ranking(Fetcher(retries=2))))
        if not ranking.empty:
            ranking.to_csv(DATA / "ranking.csv", index=False)
    except Exception as e:  # noqa: BLE001
        print(f"::warning::Рейтинг HLTV не скачался: {e}", flush=True)

    print(f"::notice::Готово: карт {len(all_maps)} (скачано {len(maps)}), ближайших матчей {len(up)}, "
          f"составов {len(rost)}, медиа-ссылок {len(med)}, команд {len(known)}", flush=True)
    return 0 if len(all_maps) else 1


if __name__ == "__main__":
    sys.exit(main())
