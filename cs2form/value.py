"""Сравнение прогноза модели с кэфами букмекера: где есть перевес и сколько ставить."""

from __future__ import annotations

from dataclasses import dataclass

MIN_EDGE = 0.05  # перевес меньше 5% не берём: ошибка модели больше
KELLY_FRACTION = 0.25
MAX_STAKE = 0.02


@dataclass
class Assessment:
    market_a: float  # шанс по кэфам без маржи
    market_b: float
    margin: float
    edge_a: float  # ожидаемая прибыль на 1 рубль ставки
    edge_b: float
    pick: str  # "a", "b" или ""
    stake: float  # доля банка
    verdict: str


def assess(
    p_a: float, odds_a: float, odds_b: float, name_a: str = "A", name_b: str = "B"
) -> Assessment:
    inv_a, inv_b = 1 / odds_a, 1 / odds_b
    margin = inv_a + inv_b - 1
    market_a = inv_a / (inv_a + inv_b)
    edge_a, edge_b = p_a * odds_a - 1, (1 - p_a) * odds_b - 1
    pick, stake = "", 0.0
    if max(edge_a, edge_b) >= MIN_EDGE:
        pick = "a" if edge_a >= edge_b else "b"
        p, o, e = (p_a, odds_a, edge_a) if pick == "a" else (1 - p_a, odds_b, edge_b)
        stake = min(MAX_STAKE, KELLY_FRACTION * e / (o - 1))
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
    else:
        verdict = "Пропуск: модель согласна с рынком, перевеса нет."
    return Assessment(
        market_a, 1 - market_a, margin, edge_a, edge_b, pick, stake, verdict
    )
