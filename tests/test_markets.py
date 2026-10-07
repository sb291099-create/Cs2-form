import numpy as np
import pytest

from cs2form import markets, value


def _table():
    # равные пары: 13:11 и 13:7 в обе стороны, в сериях на две и на три карты
    a = np.array([13, 11, 13, 7] * 2)
    b = np.array([11, 13, 7, 13] * 2)
    full = np.array([True] * 4 + [False] * 4)
    return markets.RoundTable(np.zeros(8), a, b, np.ones(8, bool), full)


def test_simulate_even_series():
    sim = markets.simulate([0.5, 0.5, 0.5], _table(), 3, n=40000)
    assert sim.win.mean() == pytest.approx(0.5, abs=0.01)
    three = sim.maps_a + sim.maps_b == 3
    assert three.mean() == pytest.approx(0.426, abs=0.015)  # форма на день: до третьей карты реже, чем 50%
    assert np.isnan(sim.ra[~three, 2]).all() and not np.isnan(sim.ra[three, 2]).any()
    assert set(np.unique(sim.rounds[~three])) <= {40, 44, 48}


def test_menu_complements():
    sim = markets.simulate([0.6, 0.55, 0.5], _table(), 3, n=20000)
    m = {x.group + "|" + x.name: x.p for x in markets.menu(sim, "A", "B", ["Mirage", "Nuke", "Ancient"], 57.5)}
    assert m["Матч|Победа A"] + m["Матч|Победа B"] == pytest.approx(1)
    assert m["Матч|Фора A +1.5 по картам (ИТБ A 0.5 карты)"] + m["Матч|Фора B −1.5 по картам"] == pytest.approx(1)
    assert m["Матч|Тотал карт больше 2.5"] + m["Матч|Тотал карт меньше 2.5"] == pytest.approx(1)
    assert m["Матч|Тотал раундов больше 57.5"] + m["Матч|Тотал раундов меньше 57.5"] == pytest.approx(1)
    scores = [m[f"Матч|Точный счёт {s}"] for s in ("2:0", "2:1", "1:2", "0:2")]
    assert sum(scores) == pytest.approx(1)
    assert m["Карта 1 (Mirage)|Победа A"] > m["Карта 1 (Mirage)|Победа B"]


def test_ladder_with_odds_matches_market():
    df = markets.ladder([0.5, 0.5, 0.5], _table(), 3, "A", "B", odds_a=3.0, odds_b=1.36)
    row = df[(df["Группа"] == "Матч") & (df["Рынок"] == "Победа A")].iloc[0]
    q = (1 / 3.0) / (1 / 3.0 + 1 / 1.36)
    assert row["Рынок %"] == pytest.approx(q * 100, abs=1.5)
    assert row["Брать от"] == pytest.approx(value.min_odds(row["Рынок %"] / 100))  # от честной цены рынка, не модели
    assert {"Справедливый кэф", "Брать от", "Ожидаемый кэф"} <= set(df.columns)
    assert "Брать от" not in markets.ladder([0.5, 0.5, 0.5], _table(), 3, "A", "B").columns


def test_min_odds_is_threshold():
    for q in (0.25, 0.5, 0.7):
        o = value.min_odds(q)
        assert value.single(0.9, o, q_market=q).edge == pytest.approx(value.MIN_EDGE)  # модель не влияет
        assert value.single(0.9, o * 0.98, q_market=q).stake == 0
        assert value.single(0.9, o * 1.5).stake == 0  # без линии рынка перевеса нет
