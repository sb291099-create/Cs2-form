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
import re
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
K_PLAYER = 16.0  # шаг рейтинга игрока за карту: он переносит силу между командами при переходах
R_K = 0.06  # шаг рейтинга по разнице раундов (сам рейтинг — в раундах за карту)
R_HALF_LIFE = 120.0  # без игр рейтинг по раундам тянется к среднему
H2H_DAYS = 365
H2H_HALF_LIFE = 120.0
MAP_FORM_DAYS = 180
REST_CAP = 90
FEATURES = [
    "elo",
    "map_elo",
    "form",
    "map_exp",
    "new_roster",
    "seed",
    "rounds",  # рейтинг по разнице раундов: счёт карты говорит больше, чем сам факт победы
    "rounds_form",  # недавняя разница раундов сверх ожидания
    "h2h",  # личные встречи за год
    "h2h_map",  # личные встречи на этой карте
    "exp_all",  # сколько карт сыграно: у новых команд рейтинг ненадёжен
    "rest",  # дней с последней карты
    "players",  # средний рейтинг пяти игроков состава
    "players_vs_team",  # состав сильнее или слабее самой команды
    "lan_exp",  # опыт офлайна, только для матчей на LAN: новички на сцене там играют хуже
]


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
    rounds: dict = field(default_factory=lambda: defaultdict(float))  # team -> рейтинг в раундах за карту
    rounds_date: dict = field(default_factory=dict)  # team -> дата последнего затухания
    rounds_hist: dict = field(default_factory=lambda: defaultdict(list))  # team -> [(дата, раунды сверх ожидания)]
    games: dict = field(default_factory=lambda: defaultdict(int))  # team -> сыграно карт
    h2h: dict = field(default_factory=lambda: defaultdict(list))  # (a, b) -> [(дата, выиграл ли a, карта)]
    last_seen: dict = field(default_factory=dict)  # team -> дата последней карты
    player_elo: dict = field(default_factory=lambda: defaultdict(lambda: BASE))  # игрок -> рейтинг
    lineup: dict = field(default_factory=dict)  # team -> [(дата заявки, состав)]
    lan_games: dict = field(default_factory=lambda: defaultdict(int))  # team -> сыграно карт на LAN

    def last_change(self, team, day):
        dates = [d for d in self.roster_change.get(team, []) if d <= day]
        return max(dates) if dates else None

    def players(self, team, day) -> list[str]:
        """Последний известный состав команды на эту дату (турнирные заявки Liquipedia)."""
        rows = self.lineup.get(team)
        if not rows:
            return []
        known = [pl for d, pl in rows if d <= day]
        return known[-1] if known else rows[0][1]

    def squad_elo(self, team, day) -> float | None:
        pl = self.players(team, day)
        return float(np.mean([self.player_elo[p] for p in pl])) if pl else None

    def round_rating(self, team, day) -> float:
        """Рейтинг по раундам с затуханием: без игр сила команды неизвестна и тянется к среднему."""
        prev = self.rounds_date.get(team)
        if prev is not None and day > prev:
            self.rounds[team] *= 0.5 ** ((day - prev).days / R_HALF_LIFE)
        if prev is None or day > prev:
            self.rounds_date[team] = day
        return self.rounds[team]

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


def lineups(rosters: pd.DataFrame, event_dates: dict, name_to_id: dict) -> dict:
    """Составы с турниров: team_id -> [(дата турнира, пятёрка игроков)] по возрастанию даты."""
    if rosters is None or rosters.empty or "players" not in rosters:
        return {}
    r = rosters.dropna(subset=["players"]).copy()
    r["date"] = pd.to_datetime(r["page"].map(event_dates), errors="coerce")
    r = r.dropna(subset=["date"]).sort_values("date")
    out = defaultdict(list)
    for row in r.itertuples():
        pl = [p.strip() for p in str(row.players).split(",") if p.strip() and "=" not in p]
        if len(pl) >= 4:
            out[name_to_id.get(row.team_name, str(row.team_name).lower())].append((row.date, pl[:5]))
    return dict(out)


def rosters_state(maps: pd.DataFrame, team_names: pd.DataFrame, rosters: pd.DataFrame) -> tuple[dict, dict]:
    """Смены составов и известные составы команд: вместе, потому что считаются по одним и тем же заявкам."""
    if rosters is None or rosters.empty or "page" not in maps:
        return {}, {}
    dates = maps.groupby("page")["date"].min().to_dict()
    n2i = name_to_id(team_names, canonical_ids(maps, team_names))
    return roster_changes(rosters, dates, n2i), lineups(rosters, dates, n2i)


def canonical_ids(maps: pd.DataFrame, team_names: pd.DataFrame) -> dict[str, str]:
    """Один клуб в Liquipedia бывает записан разными шаблонами (spirit и team spirit):
    все id с одинаковым названием сводим к тому, у которого больше всего карт."""
    if team_names is None or team_names.empty:
        return {}
    count = pd.concat([maps["team1_id"], maps["team2_id"]]).value_counts()
    out = {}
    for _, g in team_names.dropna(subset=["name"]).groupby("name"):
        ids = list(g["id"])
        if len(ids) > 1:
            best = max(ids, key=lambda i: (count.get(i, 0), -len(str(i))))
            rb = _region(best)
            out.update({i: best for i in ids if i != best and (_region(i) is None or rb is None or _region(i) == rb)})
    return out


_REGIONS = {"russian": "ru", "american": "us", "mexican": "mx", "turkish": "tr", "brazilian": "br", "chinese": "cn"}


def _region(team_id) -> str | None:
    """Пометка страны в id Liquipedia: «players.br», «magic (russian team)». Разные страны — разные команды."""
    s = str(team_id)
    m = re.search(r"\((\w+) team\)", s)
    if m:
        return _REGIONS.get(m.group(1), m.group(1))
    m = re.search(r"\.([a-z]{2})$", s)
    return m.group(1) if m else None


def canonicalize(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    if df is None or df.empty or not mapping:
        return df
    df = df.copy()
    for c in ("team1_id", "team2_id"):
        if c in df:
            df[c] = df[c].map(lambda x: mapping.get(x, x))
    return df


def name_to_id(team_names: pd.DataFrame, mapping: dict[str, str]) -> dict[str, str]:
    if team_names is None or team_names.empty:
        return {}
    return {n: mapping.get(i, i) for i, n in zip(team_names["id"], team_names["name"])}


def features(st: State, a, b, mp: str, day: pd.Timestamp, seed: float = 1.0, lan: bool = False) -> dict:
    def form(t):
        since = st.last_change(t, day)
        lo = day - pd.Timedelta(days=FORM_DAYS)
        if since is not None and since > lo:
            lo = since
        rows = [(d, x) for d, x in st.hist[t] if lo < d <= day]
        if not rows:
            return 0.0
        w = np.array([0.5 ** ((day - d).days / FORM_HALF_LIFE) for d, _ in rows])
        x = np.array([v for _, v in rows])
        return float((w * x).sum() / (w.sum() + 2.0))  # +2: при малом числе карт форма тянется к нулю

    def exp(t):
        lo = day - pd.Timedelta(days=EXP_DAYS)
        return math.log1p(sum(1 for d in st.map_dates[(t, mp)] if lo < d <= day))

    def fresh(t):
        c = st.last_change(t, day)
        return 1.0 if c is not None and (day - c).days <= NEW_ROSTER_DAYS else 0.0

    def decayed(rows, half_life, prior=2.0):
        if not rows:
            return 0.0
        w = np.array([0.5 ** ((day - d).days / half_life) for d, _ in rows])
        return float((w * np.array([v for _, v in rows])).sum() / (w.sum() + prior))

    def rounds_form(t):
        since = st.last_change(t, day)
        lo = day - pd.Timedelta(days=FORM_DAYS)
        if since is not None and since > lo:
            lo = since
        return decayed([(d, x) for d, x in st.rounds_hist[t] if lo < d <= day], FORM_HALF_LIFE)

    def h2h(same_map):
        rows = [
            (d, w - 0.5)
            for d, w, m in st.h2h[(a, b)]
            if (day - d).days <= H2H_DAYS and d < day and (not same_map or m == mp)
        ]
        return decayed(rows, H2H_HALF_LIFE)

    def rest(t):
        seen = st.last_seen.get(t)
        days = (day - seen).days if seen is not None else REST_CAP
        return math.log1p(min(max(days, 0), REST_CAP))

    sa, sb = st.squad_elo(a, day), st.squad_elo(b, day)
    known = sa is not None and sb is not None
    return {
        "elo": (st.elo[a] - st.elo[b]) / 100.0,
        "map_elo": ((st.map_elo[(a, mp)] - BASE) - (st.map_elo[(b, mp)] - BASE)) / 100.0,
        "form": form(a) - form(b),
        "map_exp": exp(a) - exp(b),
        "new_roster": fresh(a) - fresh(b),
        "seed": seed,
        "rounds": (st.round_rating(a, day) - st.round_rating(b, day)) / 4.0,
        "rounds_form": (rounds_form(a) - rounds_form(b)) / 4.0,
        "h2h": h2h(False),
        "h2h_map": h2h(True),
        "exp_all": math.log1p(st.games[a]) - math.log1p(st.games[b]),
        "rest": rest(a) - rest(b),
        "players": (sa - sb) / 100.0 if known else 0.0,
        "players_known": 1.0 if known else 0.0,  # не признак (модель симметрична, вес вышел 0), только для справки
        "players_vs_team": ((sa - st.elo[a]) - (sb - st.elo[b])) / 100.0 if known else 0.0,
        "lan_exp": (math.log1p(st.lan_games[a]) - math.log1p(st.lan_games[b])) if lan else 0.0,
    }


def update(st: State, a, b, mp: str, day: pd.Timestamp, s1: int, s2: int, lan: bool = False):
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
    diff = max(-16, min(16, s1 - s2))
    over = diff - (st.round_rating(a, day) - st.round_rating(b, day))
    st.rounds[a] += R_K * over
    st.rounds[b] -= R_K * over
    st.rounds_hist[a].append((day, over))
    st.rounds_hist[b].append((day, -over))
    pa, pb = st.players(a, day), st.players(b, day)
    if pa and pb:
        ep = expected(st.squad_elo(a, day), st.squad_elo(b, day))
        dp = K_PLAYER * (won - ep)
        for p in pa:
            st.player_elo[p] += dp
        for p in pb:
            st.player_elo[p] -= dp
    st.h2h[(a, b)].append((day, won, mp))
    st.h2h[(b, a)].append((day, 1 - won, mp))
    st.games[a] += 1
    st.games[b] += 1
    if lan:
        st.lan_games[a] += 1
        st.lan_games[b] += 1
    st.last_seen[a] = st.last_seen[b] = day
    st.last_date = day


def build(
    maps: pd.DataFrame, roster_change: dict | None = None, lineup: dict | None = None
) -> tuple[State, pd.DataFrame]:
    """Проходит все карты, возвращает итоговое состояние и таблицу признаков по каждой карте."""
    st = State()
    st.roster_change = roster_change or {}
    st.lineup = lineup or {}
    m = maps.copy()
    m["date"] = pd.to_datetime(m["date"])
    sort_cols = [c for c in ["date", "match_key", "map_number"] if c in m]
    rows = []
    for r in m.sort_values(sort_cols).itertuples(index=False):
        st.apply_roster_changes(r.date)
        lan = bool(getattr(r, "lan", False))
        f = features(st, r.team1_id, r.team2_id, r.map, r.date, lan=lan)
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
        update(st, r.team1_id, r.team2_id, r.map, r.date, int(r.score1), int(r.score2), lan)
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
        "importance": importance(tr, te, _logloss(y, p)),
    }


def importance(tr: pd.DataFrame, te: pd.DataFrame, base: float) -> dict:
    """Вклад каждого признака: насколько хуже logloss, если убрать его одного.
    Признаки связаны между собой, поэтому вес в формуле сам по себе ни о чём не говорит, а это — говорит."""
    out = {}
    for f in FEATURES:
        cols = [c for c in FEATURES if c != f]
        m = MapModel().fit(tr[cols].values, tr["y"].values)
        out[f] = round(_logloss(te["y"].values, m.predict(te[cols].values)) - base, 4)
    return out


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
    model: MapModel, st: State, a, b, pool: list[str], day: pd.Timestamp, seed: float = 1.0, lan: bool = False
) -> dict[str, float]:
    st.apply_roster_changes(day)
    X = np.array([[features(st, a, b, mp, day, seed, lan)[k] for k in FEATURES] for mp in pool])
    return dict(zip(pool, model.predict(X)))


def comfort(st: State, t, pool, day, days=90) -> dict[str, int]:
    lo = day - pd.Timedelta(days=days)
    return {mp: sum(1 for d in st.map_dates[(t, mp)] if d > lo) for mp in pool}


def _side_rows(maps: pd.DataFrame) -> pd.DataFrame:
    if "first1" not in maps:
        return maps.iloc[0:0]
    m = maps[maps["first1"].isin(["ct", "t"])].copy()
    for c in ("ct1", "t1", "ct2", "t2"):
        m[c] = pd.to_numeric(m[c], errors="coerce")
    return m.dropna(subset=["ct1", "t1", "ct2", "t2"])


def side_stats(maps: pd.DataFrame, team, day, days: int = 90) -> dict[str, tuple[float, float, int]]:
    """Доля выигранных раундов команды за CT и за T на каждой карте (основное время) и число карт."""
    m = _side_rows(maps)
    m = m[pd.to_datetime(m["date"]) > pd.Timestamp(day) - pd.Timedelta(days=days)]
    out = {}
    for mp, g in m.groupby("map"):
        a, b = g[g["team1_id"] == team], g[g["team2_id"] == team]
        ct_w = a["ct1"].sum() + b["ct2"].sum()
        ct_l = a["t2"].sum() + b["t1"].sum()  # раунды соперника за T, пока команда за CT
        t_w = a["t1"].sum() + b["t2"].sum()
        t_l = a["ct2"].sum() + b["ct1"].sum()
        n = len(a) + len(b)
        if n:
            out[mp] = (ct_w / max(ct_w + ct_l, 1), t_w / max(t_w + t_l, 1), n)
    return out


def start_side_table(maps: pd.DataFrame) -> pd.DataFrame:
    """Как часто выигрывает команда, начавшая за CT, на решающих картах bo3 (сторона — ножом, без перекоса пика)."""
    m = _side_rows(maps)
    m = m[(m["bestof"].astype(int) == 3) & (m["map_number"].astype(int) == 3)]
    won = (m["score1"].astype(int) > m["score2"].astype(int)).astype(int)
    m = m.assign(start_ct_won=won.where(m["first1"] == "ct", 1 - won), ct_rounds=m["ct1"] + m["ct2"])
    g = m.groupby("map").agg(
        maps=("start_ct_won", "size"), start_ct_won=("start_ct_won", "mean"), ct_rounds=("ct_rounds", "sum")
    )
    total = m.groupby("map").apply(lambda x: (x["ct1"] + x["t1"] + x["ct2"] + x["t2"]).sum(), include_groups=False)
    g["ct_round_share"] = g["ct_rounds"] / total
    return g.drop(columns="ct_rounds").sort_values("maps", ascending=False)


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


MOMENTUM = 0.0  # отдельной инерции после карты нет: связь карт внутри серии объясняет DAY_SIGMA
DAY_SIGMA = 0.9  # разброс силы команды в день матча (log-odds), общий для всех карт серии; оценён по 1878 сериям bo3
_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(24)
_GH_W = _GH_W / _GH_W.sum()


def _shift(p: float, d: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return 1 / (1 + math.exp(-(math.log(p / (1 - p)) + d)))


def day_logit(p: float, sigma: float = DAY_SIGMA) -> float:
    """Log-odds карты «в среднем по дням», при котором с учётом разброса формы шанс карты остаётся равным p."""
    p = min(max(p, 1e-6), 1 - 1e-6)
    lo, hi = -15.0, 15.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if float((_GH_W / (1 + np.exp(-(mid + sigma * _GH_X)))).sum()) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def score_distribution(
    map_p: list[float], bestof: int, momentum: float = MOMENTUM, sigma: float = DAY_SIGMA
) -> dict[tuple[int, int], float]:
    """Вероятности точного счёта серии. Форма команды в день матча плавает одинаково для всех карт серии:
    поэтому 2:0 бывает чаще, чем при независимых картах, а шанс каждой отдельной карты не меняется."""
    need = bestof // 2 + 1
    out: dict[tuple[int, int], float] = {}
    nodes = list(zip(_GH_X * sigma, _GH_W)) if sigma > 0 else [(0.0, 1.0)]
    base = [day_logit(p, sigma) if sigma > 0 else math.log(max(p, 1e-6) / max(1 - p, 1e-6)) for p in map_p]

    def go(i, a, b, prob, last, ps):
        if a == need or b == need:
            out[(a, b)] = out.get((a, b), 0.0) + prob
            return
        p = ps[min(i, len(ps) - 1)]
        if last is not None and momentum:
            p = _shift(p, momentum if last else -momentum)
        go(i + 1, a + 1, b, prob * p, True, ps)
        go(i + 1, a, b + 1, prob * (1 - p), False, ps)

    for z, w in nodes:
        go(0, 0, 0, float(w), None, [1 / (1 + math.exp(-(x + z))) for x in base])
    return out


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
    scores: dict = field(default_factory=dict)

    @property
    def p_full_distance(self) -> float:
        """Шанс, что серия дойдёт до последней карты (для bo3 — до третьей)."""
        n = max(a + b for a, b in self.scores) if self.scores else 0
        return sum(v for (a, b), v in self.scores.items() if a + b == n)


def forecast(
    model: MapModel,
    st: State,
    a,
    b,
    pool,
    day,
    bestof: int = 3,
    seed: float = 1.0,
    maps: list[str] | None = None,
    lan: bool = False,
) -> MatchForecast:
    """seed=1 — a записана в сетке первой (как в ближайших матчах Liquipedia), 0 — порядок неизвестен.
    maps — карты по факту вето в порядке игры; без них вето прогнозируется."""
    p = map_probs(model, st, a, b, pool, day, seed, lan)
    veto = predict_veto(p, comfort(st, a, pool, day), comfort(st, b, pool, day), bestof)
    played = [mp for _, act, mp in veto if act in ("pick", "decider")][:bestof]
    if maps and len(maps) == bestof and all(mp in p for mp in maps):
        played = list(maps)
    probs = [p[mp] for mp in played] or [float(np.mean(list(p.values())))]
    bo = bestof if len(probs) == bestof else 1
    scores = score_distribution(probs, bo)
    need = bo // 2 + 1
    p_series = sum(v for (x, _), v in scores.items() if x == need)
    return MatchForecast(p, veto, played, p_series, float(np.mean(probs)), scores)
