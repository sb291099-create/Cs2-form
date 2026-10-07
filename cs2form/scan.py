"""Поиск выгодных ставок: кэфы контор против честной цены Pinnacle по всем матчам, на которые скачаны кэфы.

Шанс для ставки — линия Pinnacle без маржи (если её нет — медиана контор), как в value: на истории модель
уступила рынку, поэтому её шанс показывается для справки (value.MODEL_WEIGHT). Перевес — насколько лучший
кэф крупных контор выше честной цены. Модельные шансы рынков серии считаются по прогнозу вето; рынки
отдельных карт — только когда вето известно (vetoes), потому что «первая карта» до вето — неизвестно какая.
Запуск: python -m cs2form.scan [--all] [--veto "PARIVISION=Ancient,Cache,Inferno"]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import markets, model, value
from .odds import DATA, ODDS, same_team

SHARP = "pinnacle"
PERIODS = {"p1": 0, "p2": 1, "p3": 2}
MAX_GAP = 0.12  # контора, чья линия без маржи дальше от эталонной, скорее всего перепутала исходы
NO_SHARP_CAP = 0.005  # без линии Pinnacle эталон — медиана мягких контор, ей верим меньше: не больше 0.5% банка
TOP_N = 3  # без крупных контор перевес считаем по третьему по величине кэфу: одна щедрая контора не решает
SHOW_BOOKS = {  # крупные конторы: перевес считаем по лучшему кэфу среди них, его реально получить
    "pinnacle": "Pinnacle",
    "fonbet": "Fonbet",
    "marathonbet": "Marathon",
    "stake": "Stake",
    "1xbet": "1xBet",
    "bet365": "bet365",
}


def _pairs(g: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Шансы исходов без маржи по каждой конторе, где есть оба исхода."""
    out = {}
    for book, b in g.groupby("bookmaker"):
        pr = b.drop_duplicates("outcome").set_index("outcome")["price"]
        if len(pr) == 2:
            inv = 1 / pr
            out[book] = (inv / inv.sum()).to_dict()
    return out


def _reference(pairs: dict[str, dict[str, float]]) -> dict[str, float]:
    if not pairs:
        return {}
    if SHARP in pairs:
        return pairs[SHARP]
    keys = next(iter(pairs.values()))
    return {k: float(np.median([q[k] for q in pairs.values() if k in q])) for k in keys}


def _unflip(g: pd.DataFrame, ml1: float | None) -> pd.DataFrame:
    """У части контор знак форы по картам перевёрнут: их «−1.5» на деле «+1.5». Такие линии разворачиваем:
    выиграть 2:0 не может быть вероятнее, чем выиграть матч, а взять хотя бы карту — менее вероятно."""
    sp = (g["market"] == "spreads") & (g["period"] == "result") & (g["line"] != 0)
    if ml1 is None or not sp.any():
        return g
    g = g.copy()
    for (book, line), b in g[sp].groupby(["bookmaker", "line"]):
        q = _pairs(b).get(book)
        if q and "1" in q and (line < 0) == (q["1"] > ml1):
            g.loc[b.index, "line"] = -line
    return g


def _clean(gm: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    """Эталонные шансы рынка (Pinnacle, иначе медиана) и строки только тех контор, чья линия с ними согласна."""
    pairs = _pairs(gm)
    ref = _reference(pairs)
    if not ref:
        return {}, gm.iloc[0:0]
    good = [b for b, q in pairs.items() if all(abs(q[k] - ref[k]) <= MAX_GAP for k in ref if k in q)]
    return ref, gm[gm["bookmaker"].isin(good)]


def _model_p(sim: markets.Sim, market: str, period: str, line: float, outcome: str, p1_is_a: bool) -> float | None:
    """Шанс исхода по симуляции; линии и исходы OddsPapi даны для participant1 (outcome «1»)."""
    ma, mb = (sim.maps_a, sim.maps_b) if p1_is_a else (sim.maps_b, sim.maps_a)
    if period == "result":
        if market == "moneyline":
            p1 = float((ma > mb).mean())
            return p1 if outcome == "1" else 1 - p1
        if market == "spreads":
            p1 = float((ma - mb + line > 0).mean())
            return p1 if outcome == "1" else float((mb - ma - line > 0).mean())
        if market == "totals":
            over = float((ma + mb > line).mean())
            return over if outcome == "Over" else float((ma + mb < line).mean())
        return None
    i = PERIODS.get(period)
    if i is None or i >= sim.ra.shape[1]:
        return None
    ra, rb = (sim.ra[:, i], sim.rb[:, i]) if p1_is_a else (sim.rb[:, i], sim.ra[:, i])
    played = ~np.isnan(ra)
    if played.mean() < 0.05:
        return None
    ra, rb = ra[played], rb[played]
    if market == "moneyline":
        p1 = float((ra > rb).mean())
        return p1 if outcome == "1" else 1 - p1
    if market == "spreads-rounds":
        return float((ra - rb + line > 0).mean()) if outcome == "1" else float((rb - ra - line > 0).mean())
    if market == "totals-rounds":
        return float((ra + rb > line).mean()) if outcome == "Over" else float((ra + rb < line).mean())
    if market in ("teamtotals-rounds-team1", "teamtotals-rounds-team2"):
        t = ra if market.endswith("team1") else rb
        return float((t > line).mean()) if outcome == "Over" else float((t < line).mean())
    return None


def _label(market: str, period: str, line: float, outcome: str, n1: str, n2: str, map_name: str = "") -> str:
    team = n1 if outcome == "1" else n2
    h = line if outcome == "1" else -line
    where = f" на карте {PERIODS[period] + 1}" + (f" ({map_name})" if map_name else "") if period in PERIODS else ""
    side = "больше" if outcome == "Over" else "меньше"
    if market == "moneyline":
        return f"Победа {team}" + where
    if market == "spreads":
        hint = {-1.5: " (2:0)", 1.5: " (хотя бы карта)"}.get(h, "")
        return f"Фора {team} {h:+.1f} по картам{hint}"
    if market == "totals":
        return f"Тотал карт {side} {line}"
    if market == "spreads-rounds":
        return f"Фора раундов {team} {h:+.1f}" + where
    if market == "totals-rounds":
        return f"Тотал раундов {side} {line}" + where
    if market.startswith("teamtotals-rounds"):
        t = n1 if market.endswith("team1") else n2
        return f"ИТ{'Б' if outcome == 'Over' else 'М'} {t} {line} раунда" + where
    return f"{market} {period} {line} {outcome}"


def _quotes(go: pd.DataFrame) -> str:
    pr = go.drop_duplicates("bookmaker").set_index("bookmaker")["price"]
    return ", ".join(f"{name} {pr[b]:.2f}" for b, name in SHOW_BOOKS.items() if b in pr)


def scan(
    odds: pd.DataFrame,
    upcoming: pd.DataFrame,
    mdl: model.MapModel,
    st: model.State,
    pool: list[str],
    table: markets.RoundTable,
    day: pd.Timestamp | None = None,
    vetoes: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Все исходы с кэфами: шанс модели и рынка, кэф «брать от», кэфы контор и перевес.
    Перевес и ставка считаются по лучшему кэфу крупных контор (price), лучший кэф вообще — для справки."""
    vetoes = vetoes or {}
    rows = []
    up = upcoming.drop_duplicates("match_key").set_index("match_key")
    for key, g in odds[odds["match_key"].isin(up.index)].groupby("match_key"):
        u = up.loc[key]
        p1_is_a = same_team(g["p1"].iloc[0], u["team1"]) or same_team(g["p2"].iloc[0], u["team2"])
        n1, n2 = (u["team1"], u["team2"]) if p1_is_a else (u["team2"], u["team1"])
        bo = int(float(u.get("bestof") or 3))
        veto = vetoes.get(key)
        d = day if day is not None else pd.Timestamp(u.get("date") or pd.Timestamp.now().normalize())
        fc = model.forecast(mdl, st, u["team1_id"], u["team2_id"], pool, d, bo, 1.0, veto)
        sim = markets.simulate([fc.map_p[m] for m in fc.played], table, bo)
        ml = _reference(_pairs(g[(g["market"] == "moneyline") & (g["period"] == "result")]))
        g = _unflip(g, ml.get("1"))
        for (market, period, line), gm in g.groupby(["market", "period", "line"]):
            if period != "result":
                if not veto:
                    continue
                gm = gm[gm["bookmaker"] == SHARP]  # рынки карт у мелких контор размечены вразнобой
            q, gm = _clean(gm)
            for outcome, go in gm.groupby("outcome"):
                p = _model_p(sim, market, period, float(line), str(outcome), p1_is_a)
                if p is None or str(outcome) not in q:
                    continue
                prices = go.sort_values("price", ascending=False)
                big = prices[prices["bookmaker"].isin(list(SHOW_BOOKS))]
                at = big.iloc[0] if len(big) else prices.iloc[min(TOP_N, len(prices)) - 1]
                price = float(at["price"])
                sharp = go.loc[go["bookmaker"] == SHARP, "price"]
                qo = q[str(outcome)]
                p_used = value.MODEL_WEIGHT * p + (1 - value.MODEL_WEIGHT) * qo
                min_odds = (1 + value.MIN_EDGE) / p_used
                edge = p_used * price - 1
                mp = fc.played[PERIODS[period]] if period in PERIODS and PERIODS[period] < len(fc.played) else ""
                side = str(outcome) in ("1", "2")  # исход за команду; иначе Over / Under
                has_sharp = bool((gm["bookmaker"] == SHARP).any())
                pick = (n1 if str(outcome) == "1" else n2) if side else str(outcome)
                pick_line = (float(line) if str(outcome) == "1" else 0.0 - float(line)) if side else float(line)
                rows.append(
                    dict(
                        match_key=key,
                        start=g["start"].iloc[0],
                        match=f"{u['team1']} – {u['team2']}",
                        market=_label(market, period, float(line), str(outcome), n1, n2, mp),
                        kind="серия" if period == "result" else "карта",
                        market_type=market,
                        period=period,
                        pick=pick,
                        pick_line=pick_line,
                        main=bool(go["main"].any()),
                        model_p=p,
                        market_p=qo,
                        p_used=p_used,
                        min_odds=min_odds,
                        sharp_price=float(sharp.iloc[0]) if len(sharp) else np.nan,
                        price=price,
                        price_book=at["bookmaker"],
                        best_price=float(prices["price"].iloc[0]),
                        best_book=prices["bookmaker"].iloc[0],
                        books=int(go["bookmaker"].nunique()),
                        books_ok=int(go.loc[go["price"] >= min_odds, "bookmaker"].nunique()),
                        quotes=_quotes(go),
                        edge=edge,
                        sharp=has_sharp,
                        stake=value._stake(price, edge) if has_sharp else min(value._stake(price, edge), NO_SHARP_CAP),
                    )
                )
    out = pd.DataFrame(rows)
    return out.sort_values("edge", ascending=False).reset_index(drop=True) if len(out) else out


def sharp_winner(odds: pd.DataFrame, name_a: str, name_b: str) -> float | None:
    """Шанс команды name_a на победу в матче по линии Pinnacle без маржи из скачанных кэфов, если она есть."""
    if odds.empty:
        return None
    ml = odds[(odds["bookmaker"] == SHARP) & (odds["market"] == "moneyline") & (odds["period"] == "result")]
    for (p1, p2), g in ml.groupby(["p1", "p2"]):
        q = _pairs(g).get(SHARP, {})
        if not {"1", "2"} <= set(q):
            continue
        if same_team(p1, name_a) and same_team(p2, name_b):
            return float(q["1"])
        if same_team(p1, name_b) and same_team(p2, name_a):
            return float(q["2"])
    return None


def prepare(data: Path = DATA) -> dict:
    """Данные и модель, как в приложении: канонические id команд, смены составов, обучение на истории."""
    maps = pd.read_csv(data / "maps.csv", dtype={"team1_id": str, "team2_id": str})
    maps["map"] = maps["map"].fillna("")
    names = pd.read_csv(data / "team_names.csv", dtype=str)
    ids = model.canonical_ids(maps, names)
    maps = model.canonicalize(maps, ids)
    rosters = pd.read_csv(data / "rosters.csv", dtype=str)
    changes = model.roster_changes(rosters, maps.groupby("page")["date"].min().to_dict(), model.name_to_id(names, ids))
    st, feat = model.build(maps, changes)
    warm = feat[feat["date"] >= feat["date"].min() + pd.Timedelta(days=45)]
    mdl = model.MapModel().fit(warm[model.FEATURES].values, warm["y"].values)
    upcoming = model.canonicalize(pd.read_csv(data / "upcoming.csv", dtype=str), ids)
    return dict(
        upcoming=upcoming, mdl=mdl, st=st, pool=model.active_pool(maps), table=markets.round_table(feat, maps, mdl)
    )


def parse_vetoes(items: list[str], upcoming: pd.DataFrame, pool: list[str]) -> dict[str, list[str]]:
    """«PARIVISION=Ancient,Cache,Inferno»: команда или ключ матча и карты в порядке игры."""
    by_lower = {m.lower(): m for m in pool}
    out = {}
    for item in items:
        who, _, maps_ = item.partition("=")
        key = next(
            (
                r.match_key
                for r in upcoming.itertuples()
                if who.strip() == r.match_key or same_team(who, r.team1) or same_team(who, r.team2)
            ),
            None,
        )
        veto = [by_lower.get(m.strip().lower(), m.strip()) for m in maps_.split(",") if m.strip()]
        if key and veto:
            out[key] = veto
    return out


def report(r: pd.DataFrame, show_all: bool = False) -> str:
    if r.empty:
        return "Нет матчей с кэфами."
    lines = []
    for match, g in r.sort_values(["start", "edge"], ascending=[True, False]).groupby("match", sort=False):
        lines.append(f"{match} · {str(g['start'].iloc[0])[:16].replace('T', ' ')} UTC")
        sel = g if show_all else g[g["edge"] >= value.MIN_EDGE]
        if sel.empty:
            lines.append("  без перевеса")
        for x in sel.itertuples():
            lines.append(
                f"  {x.market}: рынок {x.market_p * 100:.0f}% (модель {x.model_p * 100:.0f}%), "
                f"брать от {x.min_odds:.2f}; кэф {x.price:.2f}, перевес {x.edge * 100:+.0f}%, "
                f"ставка {x.stake * 100:.1f}%; от порога {x.books_ok} из {x.books}, лучший {x.best_price:.2f} "
                f"({x.best_book}); {x.quotes}"
            )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Выгодные ставки по скачанным кэфам")
    ap.add_argument("--all", action="store_true", help="показать все исходы, а не только с перевесом")
    ap.add_argument("--veto", action="append", default=[], help="«Команда=Карта1,Карта2,Карта3» для ставок на карты")
    args = ap.parse_args()
    if not ODDS.exists():
        print("Кэфов нет: data/odds.csv.gz ещё не скачан")
        return 0
    odds = pd.read_csv(ODDS)
    ctx = prepare()
    vetoes = parse_vetoes(args.veto, ctx["upcoming"], ctx["pool"])
    r = scan(odds, ctx["upcoming"], ctx["mdl"], ctx["st"], ctx["pool"], ctx["table"], vetoes=vetoes)
    fetched = str(odds["fetched"].max()) if "fetched" in odds else "?"
    print(f"Кэфы скачаны {fetched}, контор {odds['bookmaker'].nunique()}, матчей {odds['match_key'].nunique()}")
    print(report(r, args.all))
    return 0


if __name__ == "__main__":
    sys.exit(main())
