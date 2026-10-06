"""Лесенка рынков на матч: победа, форы по картам, тоталы карт, точный счёт, тоталы и форы раундов, ИТБ.

Серия разыгрывается много раз: форма на день общая для всех карт (model.DAY_SIGMA), победитель каждой карты
по её шансу, а счёт карты берётся из истории — из карт, где у пары был похожий шанс и тот же победитель."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import model, value

N_SIM = 20000
BANDWIDTH = 0.25  # ширина окна по log-odds при подборе похожих карт из истории


@dataclass
class RoundTable:
    """Карты из истории, каждая дважды: с точки зрения первой и второй команды."""

    logit: np.ndarray
    a: np.ndarray
    b: np.ndarray
    bo3: np.ndarray  # карта из серии bo3
    full: np.ndarray  # серия дошла до третьей карты: в таких сериях карты ближе по счёту


def round_table(feat: pd.DataFrame, maps: pd.DataFrame, mdl: model.MapModel) -> RoundTable:
    p = np.clip(mdl.predict(feat[model.FEATURES].values), 1e-4, 1 - 1e-4)
    mm = maps.assign(_n=maps.groupby("match_key")["map_id"].transform("size")).set_index("map_id")
    sc = mm.loc[feat["map_id"].values]
    s1, s2 = sc["score1"].astype(int).values, sc["score2"].astype(int).values
    bo3 = sc["bestof"].astype(int).values == 3
    full = bo3 & (sc["_n"].values == 3)
    ok = (np.maximum(s1, s2) >= 13) & (s1 != s2)
    lg = np.log(p / (1 - p))[ok]

    def twice(x):
        return np.concatenate([x[ok], x[ok]])

    return RoundTable(
        np.concatenate([lg, -lg]),
        np.concatenate([s1[ok], s2[ok]]),
        np.concatenate([s2[ok], s1[ok]]),
        twice(bo3),
        twice(full),
    )


@dataclass
class Sim:
    maps_a: np.ndarray  # карт выиграно A в каждой симуляции
    maps_b: np.ndarray
    ra: np.ndarray  # раунды A на каждой карте (n × карт), NaN — карта не сыграна
    rb: np.ndarray
    bestof: int

    @property
    def win(self) -> np.ndarray:
        return self.maps_a > self.maps_b

    @property
    def rounds(self) -> np.ndarray:
        return np.nansum(self.ra + self.rb, axis=1)

    @property
    def diff(self) -> np.ndarray:
        return np.nansum(self.ra - self.rb, axis=1)


def simulate(map_p: list[float], table: RoundTable, bestof: int = 3, n: int = N_SIM, seed: int = 0) -> Sim:
    rng = np.random.default_rng(seed)
    need = bestof // 2 + 1
    k = len(map_p)
    z = rng.normal(0.0, model.DAY_SIGMA, n)
    wa, wb = np.zeros(n, int), np.zeros(n, int)
    alive, wins = np.zeros((n, k), bool), np.zeros((n, k), bool)
    for i, p in enumerate(map_p):
        alive[:, i] = (wa < need) & (wb < need)
        wins[:, i] = rng.random(n) < 1 / (1 + np.exp(-(model.day_logit(p) + z)))
        wa += alive[:, i] & wins[:, i]
        wb += alive[:, i] & ~wins[:, i]
    # счёт карты — из истории: похожий шанс, тот же победитель и (для bo3) та же длина серии
    if bestof == 3:
        full = wa + wb == 3
        groups = [(full, table.full), (~full, table.bo3 & ~table.full)]
    else:
        groups = [(np.ones(n, bool), np.ones(len(table.a), bool))]
    ra, rb = np.full((n, k), np.nan), np.full((n, k), np.nan)
    won_rows = table.a > table.b
    for i, p in enumerate(map_p):
        lg = math.log(max(p, 1e-4) / max(1 - p, 1e-4))
        w = np.exp(-0.5 * ((table.logit - lg) / BANDWIDTH) ** 2)
        for sims, rows in groups:
            for won in (True, False):
                mask = alive[:, i] & sims & (wins[:, i] == won)
                cand = np.flatnonzero(rows & (won_rows == won))
                if not mask.any() or not len(cand):
                    continue
                idx = rng.choice(cand, size=int(mask.sum()), p=w[cand] / w[cand].sum())
                ra[mask, i], rb[mask, i] = table.a[idx], table.b[idx]
    return Sim(wa, wb, ra, rb, bestof)


@dataclass
class Market:
    group: str
    name: str
    p: float  # шанс по модели

    @property
    def fair(self) -> float:
        return 1 / self.p if self.p > 0 else float("inf")

    @property
    def min_odds(self) -> float:
        return value.min_odds(self.p)


def _line(x: float) -> str:
    return f"{x:+.1f}" if x else "0"


def menu(
    sim: Sim,
    name_a: str,
    name_b: str,
    map_names: list[str] | None = None,
    rounds_line: float | None = None,
    hc_line: float = 4.5,
    map_total: float = 21.5,
    map_hc: float = 3.5,
    team_total: float = 9.5,
) -> list[Market]:
    """Все основные рынки серии и каждой карты с шансами по модели."""
    out: list[Market] = []
    g = "Матч"
    win = sim.win.mean()
    if sim.bestof > 1:
        need = sim.bestof // 2 + 1
        full = (sim.maps_a + sim.maps_b == sim.bestof).mean()
        a_clean, b_clean = (sim.maps_b == 0).mean(), (sim.maps_a == 0).mean()
        rl = rounds_line if rounds_line is not None else float(np.floor(np.median(sim.rounds))) + 0.5
        rd = sim.diff
        out += [
            Market(g, f"Победа {name_a}", win),
            Market(g, f"Победа {name_b}", 1 - win),
            Market(g, f"Фора {name_a} +1.5 по картам (ИТБ {name_a} 0.5 карты)", 1 - b_clean),
            Market(g, f"Фора {name_b} +1.5 по картам (ИТБ {name_b} 0.5 карты)", 1 - a_clean),
            Market(g, f"Фора {name_a} −1.5 по картам", a_clean),
            Market(g, f"Фора {name_b} −1.5 по картам", b_clean),
            Market(g, f"Тотал карт больше {sim.bestof - 0.5}", full),
            Market(g, f"Тотал карт меньше {sim.bestof - 0.5}", 1 - full),
        ]
        for x in range(need, -1, -1):
            for y in range(need + 1):
                if (x == need) != (y == need) and (x == need or y == need):
                    pr = ((sim.maps_a == x) & (sim.maps_b == y)).mean()
                    out.append(Market(g, f"Точный счёт {x}:{y}", pr))
        out += [
            Market(g, f"Тотал раундов больше {rl}", (sim.rounds > rl).mean()),
            Market(g, f"Тотал раундов меньше {rl}", (sim.rounds < rl).mean()),
            Market(g, f"Фора раундов {name_a} {_line(-hc_line)}", (rd > hc_line).mean()),
            Market(g, f"Фора раундов {name_b} {_line(hc_line)}", (rd < hc_line).mean()),
            Market(g, f"Фора раундов {name_a} {_line(hc_line)}", (rd > -hc_line).mean()),
            Market(g, f"Фора раундов {name_b} {_line(-hc_line)}", (rd < -hc_line).mean()),
        ]
    names = map_names or [f"{i + 1}" for i in range(sim.ra.shape[1])]
    for i, mp in enumerate(names[: sim.ra.shape[1]]):
        played = ~np.isnan(sim.ra[:, i])
        if played.mean() < 0.05:
            continue
        a, b = sim.ra[played, i], sim.rb[played, i]
        g = f"Карта {i + 1} ({mp})" if map_names else f"Карта {i + 1}"
        out += [
            Market(g, f"Победа {name_a}", (a > b).mean()),
            Market(g, f"Победа {name_b}", (b > a).mean()),
            Market(g, f"Тотал раундов больше {map_total}", (a + b > map_total).mean()),
            Market(g, f"Тотал раундов меньше {map_total}", (a + b < map_total).mean()),
            Market(g, f"Фора раундов {name_a} {_line(-map_hc)}", (a - b > map_hc).mean()),
            Market(g, f"Фора раундов {name_b} {_line(map_hc)}", (a - b < map_hc).mean()),
            Market(g, f"Фора раундов {name_a} {_line(map_hc)}", (a - b > -map_hc).mean()),
            Market(g, f"Фора раундов {name_b} {_line(-map_hc)}", (a - b < -map_hc).mean()),
            Market(g, f"ИТБ {name_a} {team_total} раунда", (a > team_total).mean()),
            Market(g, f"ИТБ {name_b} {team_total} раунда", (b > team_total).mean()),
            Market(g, "Овертайм", (np.maximum(a, b) > 13).mean()),
        ]
    return out


def market_shift(map_p: list[float], bestof: int, p_market: float) -> float:
    """Сдвиг log-odds всех карт, при котором шанс A на серию равен рыночному."""
    lo, hi = -6.0, 6.0
    need = bestof // 2 + 1
    for _ in range(50):
        mid = (lo + hi) / 2
        sc = model.score_distribution([model._shift(p, mid) for p in map_p], bestof)
        if sum(v for (x, _), v in sc.items() if x == need) < p_market:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def ladder(
    map_p: list[float],
    table: RoundTable,
    bestof: int,
    name_a: str,
    name_b: str,
    map_names: list[str] | None = None,
    odds_a: float | None = None,
    odds_b: float | None = None,
    **lines,
) -> pd.DataFrame:
    """Таблица рынков: шанс по модели, справедливый кэф и с какого кэфа брать.
    Если известны кэфы на победу, по ним оценивается, сколько букмекер, скорее всего, даст на остальные рынки."""
    mine = menu(simulate(map_p, table, bestof), name_a, name_b, map_names, **lines)
    rows = [
        {"Группа": m.group, "Рынок": m.name, "Модель %": m.p * 100, "Справедливый кэф": m.fair, "Брать от": m.min_odds}
        for m in mine
    ]
    df = pd.DataFrame(rows)
    if odds_a and odds_b and bestof > 1:
        margin = 1 / odds_a + 1 / odds_b - 1
        q = (1 / odds_a) / (1 / odds_a + 1 / odds_b)
        d = market_shift(map_p, bestof, q)
        theirs = menu(simulate([model._shift(p, d) for p in map_p], table, bestof), name_a, name_b, map_names, **lines)
        qs = np.array([m.p for m in theirs])
        exp_odds = 1 / np.maximum(qs * (1 + max(margin, 0.0)), 1e-9)
        df["Рынок %"] = qs * 100
        df["Ожидаемый кэф"] = exp_odds
        df["Перевес при нём %"] = [value.single(m.p, o, q_market=qm).edge * 100 for m, o, qm in zip(mine, exp_odds, qs)]
    return df
