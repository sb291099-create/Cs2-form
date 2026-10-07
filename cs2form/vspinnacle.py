"""Ставки по кэфам выше честной цены Pinnacle на прошлых матчах: окупаются ли они.

Честная цена — линия Pinnacle без маржи на утреннюю загрузку (data/odds_history.csv.gz). Ставка — когда кэф
одной из контор выше неё хотя бы на min_edge, как в scan.scan без модели: тем же разворотом перевёрнутых фор
и отбором контор, чья линия согласна с эталоном. Берутся только линии, открытые к утренней загрузке.
Итог — по счёту серии из maps.csv; CLV — перевес той же ставки по линии Pinnacle на закрытии: на дистанции
он надёжнее итога, который на сотне ставок решает везение.
Запуск: python -m cs2form.vspinnacle
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import markets, model, scan, value
from .history import FETCH_AT, HIST
from .odds import DATA
from .vsmarket import logloss, save_section

SOFT = ("fonbet", "marathonbet", "1xbet", "stake", "bet365", "melbet")
EDGES = (0.0, 0.02, 0.03, 0.05, 0.08)


def results(maps: pd.DataFrame) -> pd.DataFrame:
    """Счёт сыгранных серий по картам: w1, w2 — карты team1 и team2 Liquipedia."""
    m = maps.assign(
        win1=(maps["score1"].astype(int) > maps["score2"].astype(int)).astype(int),
        win2=(maps["score2"].astype(int) > maps["score1"].astype(int)).astype(int),
    )
    g = m.groupby("match_key").agg(
        w1=("win1", "sum"), w2=("win2", "sum"), bestof=("bestof", "first"), tier=("tier", "first"), date=("date", "min")
    )
    g["bestof"] = g["bestof"].astype(float).astype(int)
    return g[g[["w1", "w2"]].max(axis=1) >= g["bestof"] // 2 + 1]


def fetch_time(start: pd.Timestamp) -> pd.Timestamp:
    t = start.normalize() + FETCH_AT
    return t if t < start else start - pd.Timedelta(hours=1)


def settle(market: str, line: float, outcome: str, a: int, b: int) -> float:
    """1 — выигрыш, 0 — проигрыш, 0.5 — возврат; a и b — карты participant1 и participant2 OddsPapi."""
    if market == "moneyline":
        d = a - b if outcome == "1" else b - a
    elif market == "spreads":
        d = a - b + line if outcome == "1" else b - a - line
    elif market == "totals":
        d = a + b - line if outcome == "Over" else line - a - b
    else:
        return np.nan
    return 1.0 if d > 0 else 0.5 if d == 0 else 0.0


def _main_fixtures(hist: pd.DataFrame) -> set:
    """Один матч иногда заведён в OddsPapi дважды: берём запись, где у Pinnacle больше котировок."""
    pin = hist[hist["book"] == scan.SHARP]
    q = pin.groupby(["match_key", "fixture_id"])["quotes"].sum().reset_index().sort_values("quotes")
    return set(q.drop_duplicates("match_key", keep="last")["fixture_id"])


def _pin_close(g: pd.DataFrame) -> dict:
    """Линия Pinnacle без маржи на закрытии по (рынок, линия)."""
    pin = g[g["book"] == scan.SHARP].rename(columns={"book": "bookmaker", "price_close": "price"})
    ml = scan._pairs(pin[pin["market"] == "moneyline"]).get(scan.SHARP, {})
    pin = scan._unflip(pin, ml.get("1"))  # знак форы бывает перевёрнут и у Pinnacle
    return {(mk, ln): scan._pairs(b).get(scan.SHARP, {}) for (mk, ln), b in pin.groupby(["market", "line"])}


def candidates(
    hist: pd.DataFrame, res: pd.DataFrame, books=SOFT, reference: str = scan.SHARP, when: str = "fetch"
) -> pd.DataFrame:
    """Лучший кэф среди books на каждый исход серии против эталона на утро, с итогом и перевесом на закрытии.
    reference = "pinnacle" — эталон Pinnacle; "median" — медиана контор без Pinnacle, как в scan, когда его нет."""
    h = hist[(hist["period"] == "result") & hist["fixture_id"].isin(_main_fixtures(hist))]
    h = h[h["match_key"].isin(res.index)]
    rows = []
    for fid, g in h.groupby("fixture_id"):
        start = pd.Timestamp(g["start"].iloc[0])
        opened = pd.to_datetime(g["opened"] if "opened" in g else pd.Series(None, g.index), utc=True, errors="coerce")
        o = g[opened.isna() | (opened <= fetch_time(start))]
        o = o.rename(columns={"book": "bookmaker", f"price_{when}": "price"})
        if reference != scan.SHARP:
            o = o[o["bookmaker"] != scan.SHARP]
        ml = scan._pairs(o[o["market"] == "moneyline"])
        if reference == scan.SHARP and scan.SHARP not in ml:
            continue
        o = scan._unflip(o, scan._reference(ml).get("1"))
        close = _pin_close(g)
        key = g["match_key"].iloc[0]
        r = res.loc[key]
        a, b = (r.w1, r.w2) if bool(g["p1_is_team1"].iloc[0]) else (r.w2, r.w1)
        for (market, line), gm in o.groupby(["market", "line"]):
            q, gm = scan._clean(gm)
            if not q or (reference == scan.SHARP and scan.SHARP not in set(gm["bookmaker"])):
                continue
            qc = close.get((market, line), {})
            for outcome, go in gm[gm["bookmaker"].isin(books)].groupby("outcome"):
                if str(outcome) not in q:
                    continue
                best = go.sort_values("price").iloc[-1]
                price = float(best["price"])
                rows.append(
                    dict(
                        match_key=key,
                        fixture_id=fid,
                        date=r.date,
                        tier=r.tier,
                        bestof=r.bestof,
                        market=market,
                        line=float(line),
                        outcome=str(outcome),
                        book=best["bookmaker"],
                        price=price,
                        q=q[str(outcome)],
                        edge=q[str(outcome)] * price - 1,
                        clv=qc[str(outcome)] * price - 1 if str(outcome) in qc else np.nan,
                        won=settle(market, float(line), str(outcome), a, b),
                    )
                )
    return pd.DataFrame(rows)


def summary(c: pd.DataFrame) -> dict:
    """Итог ставок по 1 единице: доля выигрышей, ROI с ошибкой, перевес на закрытии."""
    if c.empty:
        return dict(n=0)
    profit = np.where(c["won"] == 1, c["price"] - 1, np.where(c["won"] == 0.5, 0.0, -1.0))
    return dict(
        n=len(c),
        matches=int(c["match_key"].nunique()),
        won=round(float((c["won"] == 1).mean()), 3),
        price=round(float(c["price"].mean()), 2),
        edge=round(float(c["edge"].mean()), 3),
        roi=round(float(profit.mean()), 3),
        roi_se=round(float(profit.std(ddof=1) / np.sqrt(len(c))), 3) if len(c) > 1 else np.nan,
        clv=round(float(c["clv"].mean()), 3),
        clv_pos=round(float((c["clv"] > 0).mean()), 2),
    )


def one_per_match(c: pd.DataFrame) -> pd.DataFrame:
    """Как виртуальный банк: одна ставка на матч, с наибольшим перевесом."""
    return c.sort_values("edge", ascending=False).drop_duplicates("match_key")


def run(hist: pd.DataFrame | None = None, maps: pd.DataFrame | None = None, table=None) -> dict:
    hist = pd.read_csv(HIST, dtype={"match_key": str, "fixture_id": str}) if hist is None else hist
    maps = pd.read_csv(DATA / "maps.csv", dtype={"match_key": str}) if maps is None else maps
    res = results(maps)
    out = dict(books=hist.groupby("book")["fixture_id"].nunique().to_dict())
    for ref in (scan.SHARP, "median"):
        c = candidates(hist, res, reference=ref)
        if c.empty:
            out[ref] = {}
            continue
        out[ref] = {
            "все": {f"от {e:.0%}": summary(c[c["edge"] >= e]) for e in EDGES},
            "одна на матч": {f"от {e:.0%}": summary(one_per_match(c[c["edge"] >= e])) for e in EDGES},
            "по рынкам (от 3%)": {k: summary(x) for k, x in c[c["edge"] >= 0.03].groupby("market")},
            "по конторам (от 3%)": {k: summary(x) for k, x in c[c["edge"] >= 0.03].groupby("book")},
            "по уровню (от 3%)": {k: summary(x) for k, x in c[c["edge"] >= 0.03].groupby("tier")},
            "frame": c,
        }
    pin = hist[(hist["book"] == scan.SHARP) & (hist["market"] == "totals") & (hist["period"] == "result")]
    pin = pin[pin["fixture_id"].isin(_main_fixtures(hist)) & pin["match_key"].isin(res.index)]
    rows = []
    for key, g in pin.groupby("match_key"):
        pr = g.drop_duplicates("outcome").set_index("outcome")["price_fetch"]
        if {"Over", "Under"} <= set(pr.index) and res.loc[key, "bestof"] == 3:
            r = res.loc[key]
            rows.append(dict(match_key=key, price=pr["Under"], won=float(r.w1 + r.w2 == 2), edge=0.0, clv=np.nan))
    out["under_pinnacle"] = summary(pd.DataFrame(rows))
    if table is not None:
        out["derived"] = derived(hist, res, table)
    return out


FLAT = [0.5, 0.5, 0.5]  # на истории форма карт из модели не улучшала вывод рынков из линии на победу


def derived(hist: pd.DataFrame, res: pd.DataFrame, table) -> dict:
    """Проверка лесенки: 2:0 и тотал карт, выведенные из линии Pinnacle на победу, против линий Pinnacle на них.
    Шансы карт берутся ровными: линия на победу задаёт силу команд, а разброс счёта — история раундов."""
    h = hist[(hist["book"] == scan.SHARP) & (hist["period"] == "result")]
    h = h[h["fixture_id"].isin(_main_fixtures(hist)) & h["match_key"].isin(res.index)]
    rows = []
    for key, g in h.groupby("match_key"):
        r = res.loc[key]
        if r.bestof != 3:
            continue
        g = g.rename(columns={"book": "bookmaker", "price_fetch": "price"})
        ml = scan._pairs(g[g["market"] == "moneyline"]).get(scan.SHARP, {})
        g = scan._unflip(g, ml.get("1"))
        q = {(mk, ln): scan._pairs(b).get(scan.SHARP, {}) for (mk, ln), b in g.groupby(["market", "line"])}
        p20, p3 = q.get(("spreads", -1.5), {}).get("1"), q.get(("totals", 2.5), {}).get("Over")
        if not ml.get("1") or p20 is None or p3 is None:
            continue
        sim = markets.simulate(
            [model._shift(p, markets.market_shift(FLAT, 3, ml["1"])) for p in FLAT], table, 3, n=markets.N_SIM
        )
        a, b = (r.w1, r.w2) if bool(g["p1_is_team1"].iloc[0]) else (r.w2, r.w1)
        rows.append(
            dict(
                q20=p20,
                d20=float(((sim.maps_a == 2) & (sim.maps_b == 0)).mean()),
                y20=float(a == 2 and b == 0),
                q3=p3,
                d3=float((sim.maps_a + sim.maps_b == 3).mean()),
                y3=float(a + b == 3),
            )
        )
    d = pd.DataFrame(rows)
    if d.empty:
        return {}
    out = dict(n=len(d))
    for k in ("20", "3"):
        out[k] = dict(
            pinnacle=logloss(d[f"y{k}"], d[f"q{k}"]),
            derived=logloss(d[f"y{k}"], d[f"d{k}"]),
            actual=float(d[f"y{k}"].mean()),
            mean_pinnacle=float(d[f"q{k}"].mean()),
            mean_derived=float(d[f"d{k}"].mean()),
            gap=float((d[f"d{k}"] - d[f"q{k}"]).abs().mean()),
        )
    return out


def compact(out: dict, edge: float) -> dict:
    """Главное для приложения: одна ставка на матч при разных порогах, рынки и конторы при выбранном пороге."""
    v, m = out.get(scan.SHARP) or {}, out.get("median") or {}
    key = f"от {edge:.0%}"
    c = v.get("frame", pd.DataFrame())
    sel = one_per_match(c[c["edge"] >= edge]) if len(c) else c
    return dict(
        books=out["books"],
        thresholds=v.get("одна на матч", {}),
        by_market={k: summary(x) for k, x in sel.groupby("market")} if len(sel) else {},
        by_book={k: summary(x) for k, x in sel.groupby("book")} if len(sel) else {},
        median=(m.get("одна на матч") or {}).get(key, {}),
        derived=out.get("derived", {}),
        note=(
            f"Ставка — когда утренний кэф конторы выше честной цены Pinnacle хотя бы на {edge:.0%}, по одной на "
            "матч. «Перевес на закрытии» — тот же кэф против линии Pinnacle перед началом: на дистанции он "
            "надёжнее итога, потому что на сотне ставок итог решает везение. ± ошибка — разброс итога."
        ),
    )


def main() -> int:
    if not HIST.exists():
        print("Нет data/odds_history.csv.gz: сначала Actions → «Прошлые кэфы Pinnacle»")
        return 0
    out = run(table=scan.prepare()["table"] if "--derived" in sys.argv else None)
    if "--save" in sys.argv:
        save_section("value", compact(out, value.MIN_EDGE))
    print(f"Матчей по конторам: {out['books']}")
    for ref, name in ((scan.SHARP, "эталон Pinnacle"), ("median", "эталон — медиана контор без Pinnacle")):
        print(f"== {name}")
        for part, v in out[ref].items():
            if part == "frame":
                continue
            print(f"  {part}:")
            for k, s in v.items():
                print(f"    {k}: {s}")
    print(f"Всегда меньше 2.5 карт по Pinnacle утром: {out['under_pinnacle']}")
    if out.get("derived"):
        print(f"Лесенка из линии на победу против линий Pinnacle: {out['derived']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
