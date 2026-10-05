"""Модель исхода карты и матча.

Идея: последовательно проходим все карты по времени, ведём рейтинги и историю,
и для каждой карты считаем признаки «до матча» (без заглядывания в будущее):

    elo        — разница общих Elo
    map_elo    — разница силы команд на этой конкретной карте
    form       — разница формы: насколько команда в последние 60 дней играла
                 лучше/хуже ожиданий, считая только матчи текущего состава
    map_exp    — разница опыта на карте (сколько раз играли её за 90 дней)
    new_roster — у кого состав сменился за последние 30 дней
    seed       — +1 для команды, записанной в сетке первой (на Liquipedia это обычно
                 посеянная выше команда; такие выигрывают 56% карт)

Затем логистическая регрессия переводит признаки в вероятность выиграть карту.
Для матча предсказываем вето (кто что выбьет и выберет) и считаем шанс серии.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd

BASE = 1500.0
K = 40.0
K_MAP = 24.0
FORM_DAYS = 60
FORM_HALF_LIFE = 21.0  # K и полураспад подобраны по проверке на истории
EXP_DAYS = 90
NEW_ROSTER_DAYS = 30
ROSTER_ELO_SHRINK = 0.15  # при смене игрока рейтинг сдвигается к среднему на 15% за каждого
FEATURES = ["elo", "map_elo", "form", "map_exp", "new_roster", "seed"]


def expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


@dataclass
class State:
    elo: dict = field(default_factory=lambda: defaultdict(lambda: BASE))
    map_elo: dict = field(default_factory=lambda: defaultdict(lambda: BASE))
    hist: dict = field(default_factory=lambda: defaultdict(list))  # team -> [(date, over_expectation)]
    map_dates: dict = field(default_factory=lambda: defaultdict(list))  # (team, map) -> [date]
    roster_change: dict = field(default_factory=dict)  # team -> [даты смены состава]
    roster_applied: set = field(default_factory=set)
    last_date: pd.Timestamp | None = None

    def last_change(self, team, day):
        dates = [d for d in self.roster_change.get(team, []) if d <= day]
        return max(dates) if dates else None

    def apply_roster_changes(self, day):
        """Сдвигает Elo к среднему, когда у команды меняется состав."""
        for team, changes in self.roster_change.items():
            for d, n in changes_with_counts(changes):
                if d <= day and (team, d) not in self.roster_applied:
                    self.roster_applied.add((team, d))
                    shrink = min(0.5, ROSTER_ELO_SHRINK * n)
                    self.elo[team] = self.elo[team] + (BASE - self.elo[team]) * shrink


def changes_with_counts(changes):
    # roster_change хранит даты; число заменённых игроков храним в атрибуте списка, если есть
    counts = getattr(changes, "counts", None) or [1] * len(changes)
    return list(zip(changes, counts))


class ChangeList(list):
    counts: list


def roster_changes(rosters: pd.DataFrame, event_dates: dict, name_to_id: dict) -> dict:
    """По составам на турнирах находит даты смены состава: team_id -> ChangeList[даты]."""
    if rosters is None or rosters.empty:
        return {}
    r = rosters.copy()
    r["date"] = r["page"].map(event_dates)
    r = r.dropna(subset=["date", "players"]).sort_values("date")
    out = {}
    for name, g in r.groupby("team_name"):
        tid = name_to_id.get(name, name.lower())
        prev, ch = None, ChangeList()
        ch.counts = []
        for row in g.itertuples():
            players = {p.strip().lower() for p in str(row.players).split(",") if p.strip()}
            if len(players) < 4:
                continue
            if prev is not None:
                diff = len(players - prev)
                if diff:
                    ch.append(pd.Timestamp(row.date))
                    ch.counts.append(diff)
            prev = players
        if ch:
            out[tid] = ch
    return out


def features(st: State, a, b, mp: str, day: pd.Timestamp, seed: float = 1.0) -> dict:
    def form(t):
        since = st.last_change(t, day)
        lo = day - pd.Timedelta(days=FORM_DAYS)
        if since is not None and since > lo:
            lo = since
        rows = [(d, x) for d, x in st.hist[t] if d > lo]
        if not rows:
            return 0.0
        w = np.array([0.5 ** ((day - d).days / FORM_HALF_LIFE) for d, _ in rows])
        x = np.array([v for _, v in rows])
        return float((w * x).sum() / (w.sum() + 2.0))  # +2: при малом числе карт форма тянется к нулю

    def exp(t):
        lo = day - pd.Timedelta(days=EXP_DAYS)
        return math.log1p(sum(1 for d in st.map_dates[(t, mp)] if d > lo))

    def fresh(t):
        c = st.last_change(t, day)
        return 1.0 if c is not None and (day - c).days <= NEW_ROSTER_DAYS else 0.0

    return {
        "elo": (st.elo[a] - st.elo[b]) / 100.0,
        "map_elo": ((st.map_elo[(a, mp)] - BASE) - (st.map_elo[(b, mp)] - BASE)) / 100.0,
        "form": form(a) - form(b),
        "map_exp": exp(a) - exp(b),
        "new_roster": fresh(a) - fresh(b),
        "seed": seed,
    }


def update(st: State, a, b, mp: str, day: pd.Timestamp, s1: int, s2: int):
    won = 1.0 if s1 > s2 else 0.0
    e = expected(st.elo[a], st.elo[b])
    margin = 1.0 + math.log1p(abs(s1 - s2)) / 4.0
    d = K * margin * (won - e)
    st.elo[a] += d
    st.elo[b] -= d
    em = expected(st.map_elo[(a, mp)], st.map_elo[(b, mp)])
    dm = K_MAP * (won - em)
    st.map_elo[(a, mp)] += dm
    st.map_elo[(b, mp)] -= dm
    st.hist[a].append((day, won - e))
    st.hist[b].append((day, (1 - won) - (1 - e)))
    st.map_dates[(a, mp)].append(day)
    st.map_dates[(b, mp)].append(day)
    st.last_date = day


def build(maps: pd.DataFrame, roster_change: dict | None = None) -> tuple[State, pd.DataFrame]:
    """Проходит все карты, возвращает итоговое состояние и таблицу признаков по каждой карте."""
    st = State()
    st.roster_change = roster_change or {}
    m = maps.copy()
    m["date"] = pd.to_datetime(m["date"])
    sort_cols = [c for c in ["date", "match_key", "map_number"] if c in m]
    rows = []
    for r in m.sort_values(sort_cols).itertuples(index=False):
        st.apply_roster_changes(r.date)
        f = features(st, r.team1_id, r.team2_id, r.map, r.date)
        f.update(
            map_id=r.map_id,
            date=r.date,
            team1_id=r.team1_id,
            team2_id=r.team2_id,
            map=r.map,
            y=1 if int(r.score1) > int(r.score2) else 0,
            elo_p=expected(st.elo[r.team1_id], st.elo[r.team2_id]),
        )
        rows.append(f)
        update(st, r.team1_id, r.team2_id, r.map, r.date, int(r.score1), int(r.score2))
    return st, pd.DataFrame(rows)


class MapModel:
    """Логистическая регрессия без свободного члена (модель симметрична: A против B = 1 − B против A)."""

    def __init__(self, l2: float = 1.0):
        self.l2 = l2
        self.w = np.zeros(len(FEATURES))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MapModel":
        # симметризуем выборку: каждая карта добавляется и с точки зрения второй команды
        X = np.vstack([X, -X])
        y = np.concatenate([y, 1 - y])
        w = np.zeros(X.shape[1])
        for _ in range(50):  # метод Ньютона
            p = 1 / (1 + np.exp(-X @ w))
            g = X.T @ (p - y) + self.l2 * w
            H = (X * (p * (1 - p))[:, None]).T @ X + self.l2 * np.eye(len(w))
            step = np.linalg.solve(H, g)
            w -= step
            if np.abs(step).max() < 1e-8:
                break
        self.w = w
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.asarray(X) @ self.w))


def _logloss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def backtest(feat: pd.DataFrame, train_share: float = 0.6, warmup_days: int = 45) -> dict:
    """Учим на первых 60% карт по времени, проверяем на оставшихся."""
    f = feat[feat["date"] >= feat["date"].min() + pd.Timedelta(days=warmup_days)].sort_values("date")
    cut = int(len(f) * train_share)
    tr, te = f.iloc[:cut], f.iloc[cut:]
    model = MapModel().fit(tr[FEATURES].values, tr["y"].values)
    p = model.predict(te[FEATURES].values)
    y = te["y"].values
    base = np.full(len(y), 0.5)
    return {
        "train_maps": len(tr),
        "test_maps": len(te),
        "test_from": te["date"].min(),
        "test_to": te["date"].max(),
        "logloss_model": _logloss(y, p),
        "logloss_elo": _logloss(y, te["elo_p"].values),
        "logloss_coin": _logloss(y, base),
        "acc_model": float(((p > 0.5) == y).mean()),
        "acc_elo": float(((te["elo_p"].values > 0.5) == y).mean()),
        "calibration": calibration(y, p),
        "weights": dict(zip(FEATURES, model.w.round(3))),
    }


def calibration(y, p, bins=(0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.01)) -> list[dict]:
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum():
            out.append(
                {
                    "bin": f"{lo:.0%}–{min(hi, 1):.0%}",
                    "n": int(m.sum()),
                    "pred": float(p[m].mean()),
                    "actual": float(y[m].mean()),
                }
            )
    return out


def active_pool(maps: pd.DataFrame, days: int = 60, size: int = 7) -> list[str]:
    m = maps[pd.to_datetime(maps["date"]) > pd.to_datetime(maps["date"]).max() - pd.Timedelta(days=days)]
    return list(m["map"].value_counts().head(size).index)


def map_probs(
    model: MapModel, st: State, a, b, pool: list[str], day: pd.Timestamp, seed: float = 1.0
) -> dict[str, float]:
    st.apply_roster_changes(day)
    X = np.array([[features(st, a, b, mp, day, seed)[k] for k in FEATURES] for mp in pool])
    return dict(zip(pool, model.predict(X)))


def comfort(st: State, t, pool, day, days=90) -> dict[str, int]:
    lo = day - pd.Timedelta(days=days)
    return {mp: sum(1 for d in st.map_dates[(t, mp)] if d > lo) for mp in pool}


def predict_veto(p: dict[str, float], comfort_a: dict, comfort_b: dict, bestof: int = 3) -> list[tuple[str, str, str]]:
    """Жадная симуляция вето. Команда сначала выбивает карты, которые не играет вовсе,
    потом самые невыгодные для себя; выбирает самые выгодные.
    Возвращает [(команда 'A'/'B', действие 'ban'/'pick'/'decider', карта)]."""
    left = dict(p)

    def score(team, mp):
        val = left[mp] if team == "A" else 1 - left[mp]
        played = (comfort_a if team == "A" else comfort_b).get(mp, 0)
        return val + (0.15 if played >= 3 else -0.25 if played == 0 else 0)

    order = {
        1: ["A-ban", "B-ban", "A-ban", "B-ban", "A-ban", "B-ban"],
        3: ["A-ban", "B-ban", "A-pick", "B-pick", "A-ban", "B-ban"],
        5: ["A-ban", "B-ban", "A-pick", "B-pick", "A-pick", "B-pick"],
    }[bestof if bestof in (1, 3, 5) else 3]
    steps = []
    for step in order:
        team, act = step.split("-")
        if len(left) <= 1:
            break
        pick = (max if act == "pick" else min)(left, key=lambda mp: score(team, mp))
        steps.append((team, act, pick))
        del left[pick]
    if left:
        steps.append(("-", "decider", next(iter(left))))
    return steps


def series_prob(map_p: list[float], bestof: int) -> float:
    need = bestof // 2 + 1
    n = len(map_p)
    total = 0.0
    # перебираем, какие карты выиграла A (карты после решающей не играются, но на вероятность это не влияет)
    for k in range(need, n + 1):
        for wins in combinations(range(n), k):
            pr = 1.0
            for i in range(n):
                pr *= map_p[i] if i in wins else 1 - map_p[i]
            total += pr
    return total


@dataclass
class MatchForecast:
    map_p: dict
    veto: list
    played: list
    p_series: float
    p_map_avg: float


def forecast(model: MapModel, st: State, a, b, pool, day, bestof: int = 3, seed: float = 1.0) -> MatchForecast:
    """seed=1 — a записана в сетке первой (как в ближайших матчах Liquipedia), 0 — порядок неизвестен."""
    p = map_probs(model, st, a, b, pool, day, seed)
    veto = predict_veto(p, comfort(st, a, pool, day), comfort(st, b, pool, day), bestof)
    played = [mp for _, act, mp in veto if act in ("pick", "decider")][:bestof]
    probs = [p[mp] for mp in played] or [float(np.mean(list(p.values())))]
    return MatchForecast(
        p, veto, played, series_prob(probs, bestof if len(probs) == bestof else 1), float(np.mean(probs))
    )
