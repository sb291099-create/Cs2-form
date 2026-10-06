"""Сравнение прогноза модели с кэфами букмекера: где есть перевес и сколько ставить."""

from __future__ import annotations

from dataclasses import dataclass

MIN_EDGE = 0.05  # перевес меньше 5% не берём: ошибка модели больше
KELLY_FRACTION = 0.25
MAX_STAKE = 0.02
MODEL_WEIGHT = 0.5  # шанс для ставки — среднее модели и рынка: рынок знает о стендинах и заменах
GAP_WARN = 0.12
DEFAULT_MARGIN = 0.06  # обычная маржа букмекера, когда известен кэф только на один исход


@dataclass
class Assessment:
    market_a: float  # шанс по кэфам без маржи
    market_b: float
    margin: float
    edge_a: float  # ожидаемая прибыль на 1 рубль ставки
    edge_b: float
    p_used: float  # шанс первой команды, по которому считается перевес
    pick: str  # "a", "b" или ""
    stake: float  # доля банка
    verdict: str


def assess(p_a: float, odds_a: float, odds_b: float, name_a: str = "A", name_b: str = "B") -> Assessment:
    inv_a, inv_b = 1 / odds_a, 1 / odds_b
    margin = inv_a + inv_b - 1
    market_a = inv_a / (inv_a + inv_b)
    p_used = MODEL_WEIGHT * p_a + (1 - MODEL_WEIGHT) * market_a
    edge_a, edge_b = p_used * odds_a - 1, (1 - p_used) * odds_b - 1
    pick, stake = "", 0.0
    if max(edge_a, edge_b) >= MIN_EDGE:
        pick = "a" if edge_a >= edge_b else "b"
        o, e = (odds_a, edge_a) if pick == "a" else (odds_b, edge_b)
        stake = min(MAX_STAKE, KELLY_FRACTION * e / (o - 1))
        if abs(p_a - market_a) > GAP_WARN:
            stake = min(stake, MAX_STAKE / 2)  # большой разрыв с рынком — чаще ошибка модели, ставка меньше
    fav_market = "a" if odds_a < odds_b else "b" if odds_b < odds_a else ""
    fav_model = "a" if p_a > 0.5 else "b"
    names = {"a": name_a, "b": name_b}
    if pick:
        role = "андердога" if fav_market and pick != fav_market else "фаворита"
        if not fav_market:
            role = "фаворита модели"
        verdict = (
            f"Ставка на {role} {names[pick]}: перевес {max(edge_a, edge_b) * 100:+.0f}%, "
            f"ставить {stake * 100:.1f}% банка."
        )
    elif not fav_market:
        verdict = (
            f"Кэфы равные, модель выделяет {names[fav_model]} ({max(p_a, 1 - p_a) * 100:.0f}%), "
            "но перевеса не хватает для ставки."
        )
    elif abs(p_a - market_a) <= GAP_WARN / 2:
        verdict = "Пропуск: модель согласна с рынком, перевеса нет."
    else:
        verdict = f"Пропуск: перевес {max(edge_a, edge_b) * 100:+.0f}% меньше порога {MIN_EDGE * 100:.0f}%."
    if abs(p_a - market_a) > GAP_WARN:
        verdict += (
            f" Модель расходится с рынком на {abs(p_a - market_a) * 100:.0f} п.п.: "
            "проверь составы и стендинов, рынок часто знает то, чего нет в статистике."
        )
    return Assessment(market_a, 1 - market_a, margin, edge_a, edge_b, p_used, pick, stake, verdict)


@dataclass
class Single:
    q: float  # шанс по рынку без маржи
    p_used: float
    edge: float
    stake: float


def _stake(p: float, q: float, odds: float, edge: float) -> float:
    if edge < MIN_EDGE or odds <= 1:
        return 0.0
    stake = min(MAX_STAKE, KELLY_FRACTION * edge / (odds - 1))
    return min(stake, MAX_STAKE / 2) if abs(p - q) > GAP_WARN else stake


def single(p: float, odds: float, q_market: float | None = None, margin: float = DEFAULT_MARGIN) -> Single:
    """Перевес для одного исхода (фора, тотал, счёт), когда известен только его кэф."""
    q = q_market if q_market is not None else 1 / (odds * (1 + margin))
    p_used = MODEL_WEIGHT * p + (1 - MODEL_WEIGHT) * q
    edge = p_used * odds - 1
    return Single(q, p_used, edge, _stake(p, q, odds, edge))


def min_odds(p: float, margin: float = DEFAULT_MARGIN) -> float:
    """С какого кэфа исход проходит порог перевеса, если шанс по модели p (рынок считается по самому кэфу)."""
    if p <= 0:
        return float("inf")
    return (1 + MIN_EDGE - (1 - MODEL_WEIGHT) / (1 + margin)) / (MODEL_WEIGHT * p)


def journal_summary(log) -> dict:
    """Итоги журнала ставок: только сыгранные ставки с ненулевой суммой."""
    bets = log[(log["stake"].astype(float) > 0) & log["status"].isin(["выигрыш", "проигрыш"])]
    stake = bets["stake"].astype(float)
    won = bets["status"] == "выигрыш"
    profit = (stake * (bets["odds"].astype(float) - 1)).where(won, -stake)
    return {
        "bets": len(bets),
        "wins": int(won.sum()),
        "staked": float(stake.sum()),
        "profit": float(profit.sum()),
        "roi": float(profit.sum() / stake.sum()) if len(bets) else 0.0,
        "pending": int(((log["stake"].astype(float) > 0) & ~log["status"].isin(["выигрыш", "проигрыш"])).sum()),
    }
