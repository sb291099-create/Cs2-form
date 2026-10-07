"""Модель против линии Pinnacle на прошлых матчах: кто точнее и сколько доверять модели.

Прогноз на каждый прошлый матч строится так же, как утром перед игрой: состояние команд на момент перед
первой картой, коэффициенты модели обучены только на картах до начала месяца матча, вето прогнозируется.
Линия рынка — Pinnacle без маржи на момент утренней загрузки и последняя перед началом (data/odds_history.csv.gz).
Запуск: python -m cs2form.vsmarket
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import model, scan, value
from .history import HIST
from .odds import DATA

WEIGHTS = np.round(np.arange(0, 1.01, 0.1), 1)


def load_maps() -> tuple[pd.DataFrame, dict]:
    maps = pd.read_csv(DATA / "maps.csv", dtype={"team1_id": str, "team2_id": str, "match_key": str})
    maps["map"] = maps["map"].fillna("")
    names = pd.read_csv(DATA / "team_names.csv", dtype=str)
    ids = model.canonical_ids(maps, names)
    maps = model.canonicalize(maps, ids)
    rosters = pd.read_csv(DATA / "rosters.csv", dtype=str)
    changes = model.roster_changes(rosters, maps.groupby("page")["date"].min().to_dict(), model.name_to_id(names, ids))
    return maps, changes


def monthly_models(feat: pd.DataFrame, months: list[pd.Timestamp], warmup_days: int = 45) -> dict:
    """Модель на каждый месяц проверки, обученная только на картах до его начала."""
    warm = feat[feat["date"] >= feat["date"].min() + pd.Timedelta(days=warmup_days)]
    out = {}
    for m in months:
        tr = warm[warm["date"] < m]
        out[m] = model.MapModel().fit(tr[model.FEATURES].values, tr["y"].values)
    return out


def prematch(maps: pd.DataFrame, changes: dict, models: dict, keys: set) -> pd.DataFrame:
    """Прогноз до матча по каждому матчу из keys: состояние перед первой картой, вето по прогнозу модели."""
    m = maps.copy()
    m["date"] = pd.to_datetime(m["date"])
    m = m.sort_values(["date", "match_key", "map_number"]).reset_index(drop=True)
    first = set(m.groupby("match_key").head(1).index)
    games = {k: g for k, g in m.groupby("match_key")}
    st = model.State()
    st.roster_change = changes
    pools: dict = {}
    rows = []
    for i, r in enumerate(m.itertuples(index=False)):
        st.apply_roster_changes(r.date)
        if i in first and r.match_key in keys:
            if r.date not in pools:
                recent = m[(m["date"] < r.date) & (m["date"] >= r.date - pd.Timedelta(days=60))]
                pools[r.date] = list(recent["map"].value_counts().head(7).index)
            mdl = models[r.date.to_period("M").to_timestamp()]
            g = games[r.match_key]
            bo = int(r.bestof)
            played = [mp for mp in g["map"] if mp]
            pool = list(dict.fromkeys(pools[r.date] + played))
            fc = model.forecast(mdl, st, r.team1_id, r.team2_id, pool, r.date, bo, 1.0)
            act = list(dict.fromkeys(played + fc.played))[:bo]
            fa = model.forecast(mdl, st, r.team1_id, r.team2_id, pool, r.date, bo, 1.0, act)
            s1, s2 = g["score1"].astype(int).values, g["score2"].astype(int).values
            w1, w2 = int((s1 > s2).sum()), int((s2 > s1).sum())
            sc = fc.scores
            rows.append(
                dict(
                    match_key=r.match_key,
                    date=r.date,
                    tier=getattr(r, "tier", ""),
                    team1=r.team1,
                    team2=r.team2,
                    bestof=bo,
                    p_model=fc.p_series,
                    p_model_veto=fa.p_series,
                    p20=sc.get((2, 0), np.nan) if bo == 3 else np.nan,
                    p02=sc.get((0, 2), np.nan) if bo == 3 else np.nan,
                    p_map1=fa.map_p.get(act[0], np.nan) if act else np.nan,
                    w1=w1,
                    w2=w2,
                    map1_win=int(s1[0] > s2[0]) if len(s1) else np.nan,
                )
            )
        model.update(st, r.team1_id, r.team2_id, r.map, r.date, int(r.score1), int(r.score2))
    return pd.DataFrame(rows)


def market_probs(hist: pd.DataFrame) -> pd.DataFrame:
    """Линия Pinnacle без маржи с точки зрения team1 Liquipedia: победа, 2:0, 0:2, тотал карт, первая карта."""
    h = hist[hist["book"] == "pinnacle"].copy()
    # один матч иногда заведён в OddsPapi дважды: берём запись, где больше котировок
    main = h.groupby(["match_key", "fixture_id"])["quotes"].sum().reset_index().sort_values("quotes")
    h = (
        h[h["fixture_id"].isin(main.drop_duplicates("match_key", keep="last")["fixture_id"])]
        if "fixture_id" in h
        else h
    )
    out = {}
    for key, gk in h.groupby("match_key"):
        flip = not bool(gk["p1_is_team1"].iloc[0])
        rec = out.setdefault(key, {})
        for col in ("price_fetch", "price_close"):
            when = col.split("_")[1]
            g = gk.rename(columns={"book": "bookmaker", col: "price"})
            # у части матчей OddsPapi знак форы Pinnacle перевёрнут, как и у других контор: разворачиваем
            ml = scan._pairs(g[(g["market"] == "moneyline") & (g["period"] == "result")]).get("pinnacle", {})
            g = scan._unflip(g, ml.get("1"))
            for (market, period, line), gm in g.groupby(["market", "period", "line"]):
                pr = gm.drop_duplicates("outcome").set_index("outcome")["price"]
                if len(pr) != 2:
                    continue
                inv = 1 / pr
                q = inv / inv.sum()
                if market == "moneyline" and period == "result" and {"1", "2"} <= set(q.index):
                    rec[f"q_{when}"] = q["2"] if flip else q["1"]
                    rec[f"price1_{when}"] = pr["2"] if flip else pr["1"]
                    rec[f"price2_{when}"] = pr["1"] if flip else pr["2"]
                elif market == "moneyline" and period == "p1" and {"1", "2"} <= set(q.index):
                    rec[f"qmap1_{when}"] = q["2"] if flip else q["1"]
                elif market == "spreads" and period == "result" and {"1", "2"} <= set(q.index):
                    # фора участника 1 −1.5: он выигрывает 2:0; +1.5: участник 2 выигрывает 2:0 с вероятностью 1 − q1
                    if line < 0:
                        rec[f"q{'02' if flip else '20'}_{when}"] = q["1"]
                    elif line > 0:
                        rec[f"q{'20' if flip else '02'}_{when}"] = 1 - q["1"]
                elif market == "totals" and period == "result" and {"Over", "Under"} <= set(q.index):
                    rec[f"q3_{when}"] = q["Over"]
    return pd.DataFrame.from_dict(out, orient="index").rename_axis("match_key").reset_index()


def logloss(y, p) -> float:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def blend_curve(y, p_model, q) -> pd.DataFrame:
    """Точность среднего w·модель + (1 − w)·рынок при разных w: w = 0 — чистый рынок, 1 — чистая модель."""
    return pd.DataFrame(
        [
            dict(
                w=w,
                logloss=logloss(y, w * p_model + (1 - w) * q),
                brier=float(np.mean((w * p_model + (1 - w) * q - y) ** 2)),
            )
            for w in WEIGHTS
        ]
    )


def bets(
    df: pd.DataFrame, w: float = value.MODEL_WEIGHT, min_edge: float = value.MIN_EDGE, when: str = "fetch"
) -> pd.DataFrame:
    """Ставки по нашим правилам на линии Pinnacle на утро: перевес от min_edge, на ту сторону, где он есть."""
    rows = []
    for r in df.itertuples():
        q = getattr(r, f"q_{when}")
        p = w * r.p_model + (1 - w) * q
        for side, ps, price, close in (
            (1, p, r.price1_fetch, r.price1_close),
            (2, 1 - p, r.price2_fetch, r.price2_close),
        ):
            edge = ps * price - 1
            if edge >= min_edge:
                won = (r.w1 > r.w2) if side == 1 else (r.w2 > r.w1)
                stake = value._stake(r.p_model if side == 1 else 1 - r.p_model, q if side == 1 else 1 - q, price, edge)
                rows.append(
                    dict(match_key=r.match_key, side=side, price=price, close=close, edge=edge, won=won, stake=stake)
                )
    return pd.DataFrame(rows)


def summary_bets(b: pd.DataFrame) -> dict:
    if b.empty:
        return dict(n=0)
    profit = np.where(b["won"], b["price"] - 1, -1.0)
    return dict(
        n=len(b),
        won=float(b["won"].mean()),
        roi=float(profit.mean()),
        roi_staked=float((profit * b["stake"]).sum() / b["stake"].sum()) if b["stake"].sum() else np.nan,
        clv=float((b["price"] / b["close"] - 1).mean()),
        avg_price=float(b["price"].mean()),
    )


def run() -> dict:
    hist = pd.read_csv(HIST, dtype={"match_key": str})
    maps, changes = load_maps()
    st, feat = model.build(maps, changes)
    keys = set(hist["match_key"])
    dates = pd.to_datetime(maps[maps["match_key"].isin(keys)]["date"])
    months = sorted({d.to_period("M").to_timestamp() for d in dates})
    pm = prematch(maps, changes, monthly_models(feat, months), keys)
    mk = market_probs(hist)
    df = pm.merge(mk, on="match_key", how="inner").dropna(subset=["q_fetch", "q_close"])
    df = df[df["bestof"] > 1]
    y = (df["w1"] > df["w2"]).astype(int).values
    res = dict(n=len(df), period=(df["date"].min(), df["date"].max()))
    res["logloss"] = {
        "монетка": logloss(y, np.full(len(y), 0.5)),
        "Pinnacle утром": logloss(y, df["q_fetch"]),
        "Pinnacle на закрытии": logloss(y, df["q_close"]),
        "модель": logloss(y, df["p_model"]),
        "модель при известном вето": logloss(y, df["p_model_veto"]),
        "среднее 50/50 (утро)": logloss(y, 0.5 * df["p_model"] + 0.5 * df["q_fetch"]),
    }
    res["curve_fetch"] = blend_curve(y, df["p_model"].values, df["q_fetch"].values)
    res["curve_close"] = blend_curve(y, df["p_model"].values, df["q_close"].values)
    res["bets"] = {
        f"w={w}, порог {e:.0%}": summary_bets(bets(df, w, e)) for w in (0.5, 0.3, 0.2, 0.1) for e in (0.05, 0.10)
    }
    res["by_tier"] = {
        t: dict(
            n=len(g),
            market=logloss((g["w1"] > g["w2"]).astype(int), g["q_fetch"]),
            model=logloss((g["w1"] > g["w2"]).astype(int), g["p_model"]),
        )
        for t, g in df.groupby("tier")
    }
    if "q3_fetch" in df:
        t3 = df.dropna(subset=["q3_fetch"])
        t3 = t3[t3["bestof"] == 3]
        y3 = ((t3["w1"] + t3["w2"]) == 3).astype(int).values
        p3 = (1 - t3["p20"] - t3["p02"]).values
        res["totals"] = dict(
            n=len(t3),
            actual=float(y3.mean()) if len(y3) else np.nan,
            market_mean=float(t3["q3_fetch"].mean()),
            model_mean=float(p3.mean()) if len(p3) else np.nan,
            market=logloss(y3, t3["q3_fetch"]),
            model=logloss(y3, p3),
        )
    if "q20_fetch" in df:
        t2 = df.dropna(subset=["q20_fetch"])
        t2 = t2[t2["bestof"] == 3]
        y2 = ((t2["w1"] == 2) & (t2["w2"] == 0)).astype(int).values
        res["two_nil"] = dict(n=len(t2), market=logloss(y2, t2["q20_fetch"]), model=logloss(y2, t2["p20"]))
    m1 = df.dropna(subset=["qmap1_close", "p_map1", "map1_win"]) if "qmap1_close" in df else df.iloc[0:0]
    if len(m1):
        res["map1"] = dict(
            n=len(m1),
            pinnacle=logloss(m1["map1_win"], m1["qmap1_close"]),
            model=logloss(m1["map1_win"], m1["p_map1"]),
            curve=blend_curve(m1["map1_win"].values, m1["p_map1"].values, m1["qmap1_close"].values),
        )
    res["frame"] = df
    return res


def main() -> int:
    if not HIST.exists():
        print("Нет data/odds_history.csv.gz: сначала Actions → «Прошлые кэфы Pinnacle»")
        return 0
    res = run()
    pd.set_option("display.width", 200)
    print(f"Матчей с линией Pinnacle и счётом: {res['n']}, {res['period'][0]:%d.%m} – {res['period'][1]:%d.%m}")
    for k, v in res["logloss"].items():
        print(f"  logloss {k}: {v:.4f}")
    print("Среднее модели и рынка (утро):")
    print(res["curve_fetch"].round(4).to_string(index=False))
    print("Ставки по правилам на утреннюю линию Pinnacle:")
    for k, v in res["bets"].items():
        print(f"  {k}: {v}")
    for k in ("by_tier", "totals", "two_nil"):
        if k in res:
            print(f"{k}: {res[k]}")
    if "map1" in res:
        print(
            f"Первая карта на закрытии: {res['map1']['n']} карт, Pinnacle {res['map1']['pinnacle']:.4f}, "
            f"модель {res['map1']['model']:.4f}"
        )
        print(res["map1"]["curve"].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
