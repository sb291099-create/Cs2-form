"""Кэфы букмекера против честной цены: где есть перевес и сколько ставить.

Честная цена — линия Pinnacle без маржи. На 402 прошлых матчах она предсказывала точнее модели (logloss 0.60
против 0.67), и любая доля модели в среднем ухудшала прогноз (python -m cs2form.vsmarket), поэтому шанс для
ставки — линия рынка, а модель показывается для справки.
"""

from __future__ import annotations

from dataclasses import dataclass

MIN_EDGE = 0.05  # перевес меньше 5% не берём: ошибка модели больше
KELLY_FRACTION = 0.25
MAX_STAKE = 0.02
MODEL_WEIGHT = 0.0  # доля модели в шансе для ставки: на истории любая доля ухудшала прогноз рынка
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


def assess(
    p_a: float, odds_a: float, odds_b: float, name_a: str = "A", name_b: str = "B", q_ref: float | None = None
) -> Assessment:
    """Кэфы на победу против честной цены: q_ref — шанс первой команды по линии Pinnacle без маржи.
    Без q_ref перевес не считается: кэфы конторы не бывают выгоднее её собственной линии без маржи."""
    inv_a, inv_b = 1 / odds_a, 1 / odds_b
    margin = inv_a + inv_b - 1
    market_a = inv_a / (inv_a + inv_b)
    ref = q_ref if q_ref is not None else market_a
    p_used = MODEL_WEIGHT * p_a + (1 - MODEL_WEIGHT) * ref
    edge_a, edge_b = p_used * odds_a - 1, (1 - p_used) * odds_b - 1
    names = {"a": name_a, "b": name_b}
    pick, stake = "", 0.0
    if q_ref is not None and max(edge_a, edge_b) >= MIN_EDGE:
        pick = "a" if edge_a >= edge_b else "b"
        o, e, p = (odds_a, edge_a, p_used) if pick == "a" else (odds_b, edge_b, 1 - p_used)
        stake = _stake(o, e)
        verdict = (
            f"Ставка на {names[pick]}: кэф {o:.2f} выше честной цены Pinnacle {1 / p:.2f}, "
            f"перевес {e * 100:+.0f}%, ставить {stake * 100:.1f}% банка."
        )
    elif q_ref is None:
        verdict = (
            "Линии Pinnacle на этот матч нет, а без неё перевес не посчитать: на прошлых матчах модель "
            "уступила рынку, поэтому сама по себе поводом для ставки не служит."
        )
    else:
        verdict = (
            f"Пропуск: кэфы не выше честной цены Pinnacle. Ставка проходит от {min_odds(p_used):.2f} "
            f"на {name_a} или от {min_odds(1 - p_used):.2f} на {name_b}."
        )
    if abs(p_a - ref) > GAP_WARN:
        verdict += (
            f" Модель расходится с рынком на {abs(p_a - ref) * 100:.0f} п.п.: на прошлых матчах в таких "
            "случаях чаще был прав рынок."
        )
    return Assessment(market_a, 1 - market_a, margin, edge_a, edge_b, p_used, pick, stake, verdict)


@dataclass
class Single:
    q: float  # шанс по рынку без маржи
    p_used: float
    edge: float
    stake: float


def _stake(odds: float, edge: float) -> float:
    """Четверть Келли, не больше MAX_STAKE банка; ниже порога перевеса — ноль."""
    if edge < MIN_EDGE or odds <= 1:
        return 0.0
    return min(MAX_STAKE, KELLY_FRACTION * edge / (odds - 1))


def single(p: float, odds: float, q_market: float | None = None, margin: float = DEFAULT_MARGIN) -> Single:
    """Перевес для одного исхода (фора, тотал, счёт): q_market — честный шанс по линии рынка.
    Без него рынок считается по самому кэфу, и перевеса нет: это та же линия с маржой."""
    q = q_market if q_market is not None else 1 / (odds * (1 + margin))
    p_used = MODEL_WEIGHT * p + (1 - MODEL_WEIGHT) * q
    edge = p_used * odds - 1
    return Single(q, p_used, edge, _stake(odds, edge))


def min_odds(q: float) -> float:
    """С какого кэфа исход проходит порог перевеса, если честный шанс q."""
    return (1 + MIN_EDGE) / q if q > 0 else float("inf")


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
