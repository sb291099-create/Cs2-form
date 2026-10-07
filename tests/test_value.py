import pytest

from cs2form import value
from cs2form.value import assess


def test_price_above_pinnacle_fair_is_a_bet():
    r = assess(0.45, 1.4, 3.0, "Fav", "Dog", q_ref=0.6)  # честная цена Dog 2.50, у конторы 3.00
    assert r.pick == "b" and r.edge_b == pytest.approx(0.2)
    assert "Ставка на Dog" in r.verdict and "Pinnacle 2.50" in r.verdict and 0 < r.stake <= 0.02


def test_no_pinnacle_no_bet_whatever_the_model_says():
    r = assess(0.66, 1.85, 1.85, "M80", "TYLOO")
    assert r.pick == "" and r.stake == 0 and r.verdict.startswith("Линии Pinnacle")


def test_skip_below_fair_price_names_threshold():
    r = assess(0.5, 1.5, 2.4, "A", "B", q_ref=0.62)
    assert r.pick == "" and r.verdict.startswith("Пропуск")
    assert f"от {(1 + value.MIN_EDGE) / 0.62:.2f} на A" in r.verdict


def test_model_gap_is_only_a_warning():
    r = assess(0.73, 1.68, 2.15, "Legacy", "1win", q_ref=0.56)  # модель 73%, рынок 56%
    assert r.p_used == pytest.approx(0.56)
    assert "расходится с рынком на 17" in r.verdict


def test_journal_summary():
    import pandas as pd

    from cs2form.value import journal_summary

    log = pd.DataFrame(
        dict(
            odds=[2.8, 1.68, 2.0, 1.5],
            stake=[1, 1.5, 1, 0],
            status=["проигрыш", "выигрыш", "ждёт", "пропуск"],
        )
    )
    s = journal_summary(log)
    assert s["bets"] == 2 and s["wins"] == 1 and s["pending"] == 1
    assert abs(s["profit"] - (-1 + 1.5 * 0.68)) < 1e-9
