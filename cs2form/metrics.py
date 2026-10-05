"""Метрики формы: Elo (общий и по картам), форма с затуханием, карт-пул, прогноз матча."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

BASE_ELO = 1500.0
K = 32.0
MAP_WEIGHT = 0.5  # насколько рейтинг на конкретной карте влияет на прогноз
FORM_HALF_LIFE_DAYS = 30.0
FORM_WINDOW_DAYS = 90


def expected(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))


def _margin_mult(round_diff: int) -> float:
    # Разгром 13:2 двигает рейтинг сильнее, чем 13:11 или овертайм.
    return 1.0 + math.log1p(abs(round_diff)) / 4.0


@dataclass
class EloResult:
    history: pd.DataFrame  # по строке на (карта, команда): рейтинг до и после, ожидание
    overall: dict[int, float]
    by_map: dict[tuple[int, str], float]


def run_elo(maps: pd.DataFrame) -> EloResult:
    """Прогоняет Elo по всем картам в хронологическом порядке."""
    overall: dict[int, float] = {}
    by_map: dict[tuple[int, str], float] = {}
    rows = []
    for m in maps.sort_values(["date", "map_id"]).itertuples(index=False):
        a, b = m.team1_id, m.team2_id
        ra, rb = overall.get(a, BASE_ELO), overall.get(b, BASE_ELO)
        has_map = isinstance(m.map, str) and m.map != ""
        ma = by_map.get((a, m.map), BASE_ELO) if has_map else BASE_ELO
        mb = by_map.get((b, m.map), BASE_ELO) if has_map else BASE_ELO
        e_a = map_win_prob(ra, rb, ma, mb)
        s_a = 1.0 if m.score1 > m.score2 else 0.0
        delta = K * _margin_mult(m.score1 - m.score2) * (s_a - e_a)
        overall[a], overall[b] = ra + delta, rb - delta
        if has_map:
            map_delta = K * (s_a - expected(ma, mb))
            by_map[(a, m.map)], by_map[(b, m.map)] = ma + map_delta, mb - map_delta
        for tid, team, opp_id, opp, r_before, r_after, exp, won, rf, ra_ in (
            (a, m.team1, b, m.team2, ra, overall[a], e_a, s_a, m.score1, m.score2),
            (b, m.team2, a, m.team1, rb, overall[b], 1 - e_a, 1 - s_a, m.score2, m.score1),
        ):
            rows.append(
                dict(
                    map_id=m.map_id,
                    date=m.date,
                    team_id=tid,
                    team=team,
                    opp_id=opp_id,
                    opp=opp,
                    map=m.map,
                    event=m.event,
                    rounds_for=rf,
                    rounds_against=ra_,
                    won=won,
                    expected=exp,
                    elo_before=r_before,
                    elo_after=r_after,
                )
            )
    hist = pd.DataFrame(rows)
    if not hist.empty:
        hist["date"] = pd.to_datetime(hist["date"])
    return EloResult(hist, overall, by_map)


def map_win_prob(ra: float, rb: float, ma: float, mb: float) -> float:
    """Вероятность победы A на карте: общий рейтинг + поправка за силу на этой карте."""
    return expected(ra + MAP_WEIGHT * (ma - BASE_ELO), rb + MAP_WEIGHT * (mb - BASE_ELO))


def form_table(elo: EloResult, as_of: pd.Timestamp | None = None, min_maps: int = 5) -> pd.DataFrame:
    """Сводка формы за последние FORM_WINDOW_DAYS дней.

    form — взвешенная по свежести разница «результат минус ожидание по Elo»,
    в процентах: +10 значит, что команда выигрывает на 10 п.п. чаще, чем
    предсказывал её рейтинг (т.е. на подъёме), с учётом силы соперников.
    """
    h = elo.history
    if h.empty:
        return pd.DataFrame()
    as_of = as_of or h["date"].max()
    recent = h[h["date"] > as_of - pd.Timedelta(days=FORM_WINDOW_DAYS)].copy()
    age = (as_of - recent["date"]).dt.days
    recent["w"] = 0.5 ** (age / FORM_HALF_LIFE_DAYS)
    recent["over"] = recent["won"] - recent["expected"]
    recent["rd"] = recent["rounds_for"] - recent["rounds_against"]
    recent["wx"] = recent["w"] * recent["over"]
    recent["ww"] = recent["w"] * recent["won"]
    g = recent.groupby("team_id")
    out = pd.DataFrame(
        {
            "team": g["team"].last(),
            "maps": g.size(),
            "winrate": g["won"].mean() * 100,
            "form": g["wx"].sum() / g["w"].sum() * 100,
            "weighted_winrate": g["ww"].sum() / g["w"].sum() * 100,
            "round_diff": g["rd"].mean(),
            "last_played": g["date"].max(),
        }
    )
    out["elo"] = out.index.map(lambda t: elo.overall.get(t, BASE_ELO))
    out["streak"] = out.index.map(lambda t: _streak(h[h["team_id"] == t]))
    out["last5"] = out.index.map(lambda t: _last_n(h[h["team_id"] == t], 5))
    out = out[out["maps"] >= min_maps]
    return out.sort_values("elo", ascending=False).reset_index()


def _streak(team_hist: pd.DataFrame) -> str:
    won = team_hist.sort_values(["date", "map_id"])["won"].tolist()
    if not won:
        return ""
    last, n = won[-1], 0
    for w in reversed(won):
        if w != last:
            break
        n += 1
    return f"{'W' if last else 'L'}{n}"


def _last_n(team_hist: pd.DataFrame, n: int) -> str:
    won = team_hist.sort_values(["date", "map_id"])["won"].tail(n).tolist()
    return "".join("W" if w else "L" for w in won)


def map_pool(elo: EloResult, team_id: int, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    h = elo.history
    as_of = as_of or h["date"].max()
    t = h[(h["team_id"] == team_id) & (h["date"] > as_of - pd.Timedelta(days=FORM_WINDOW_DAYS))]
    if t.empty:
        return pd.DataFrame(columns=["map", "maps", "winrate", "round_diff", "map_elo"])
    t = t.assign(rd=t["rounds_for"] - t["rounds_against"])
    g = t.groupby("map")
    out = pd.DataFrame({"maps": g.size(), "winrate": g["won"].mean() * 100, "round_diff": g["rd"].mean()})
    out["map_elo"] = [elo.by_map.get((team_id, m), BASE_ELO) for m in out.index]
    return out.sort_values("maps", ascending=False).reset_index()


def bo3_prob(p: float) -> float:
    """Вероятность выиграть серию до двух побед при одинаковом p на каждой карте."""
    return p * p * (3 - 2 * p)


def matchup(elo: EloResult, a: int, b: int, maps: list[str]) -> tuple[pd.DataFrame, float, float]:
    """Вероятности A на каждой карте и оценка для bo1/bo3.

    Без данных о вето карты взвешиваются по тому, как часто обе команды их играют.
    """
    ra, rb = elo.overall.get(a, BASE_ELO), elo.overall.get(b, BASE_ELO)
    pool_a = map_pool(elo, a).set_index("map")["maps"] if not elo.history.empty else pd.Series(dtype=float)
    pool_b = map_pool(elo, b).set_index("map")["maps"] if not elo.history.empty else pd.Series(dtype=float)
    rows = []
    for m in maps:
        p = map_win_prob(ra, rb, elo.by_map.get((a, m), BASE_ELO), elo.by_map.get((b, m), BASE_ELO))
        weight = float(pool_a.get(m, 0)) + float(pool_b.get(m, 0))
        rows.append(
            dict(map=m, prob_a=p * 100, maps_a=int(pool_a.get(m, 0)), maps_b=int(pool_b.get(m, 0)), weight=weight)
        )
    df = pd.DataFrame(rows, columns=["map", "prob_a", "maps_a", "maps_b", "weight"])
    if df["weight"].sum() > 0:
        p_map = (df["prob_a"] * df["weight"]).sum() / df["weight"].sum() / 100
    else:
        p_map = expected(ra, rb)
    return df.drop(columns="weight"), p_map * 100, bo3_prob(p_map) * 100
