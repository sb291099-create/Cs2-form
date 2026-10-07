import numpy as np
import pandas as pd
import pytest

from cs2form import vsmarket


def _rows(market, period, line, prices):
    return [
        dict(
            match_key="EPL#31",
            book="pinnacle",
            market=market,
            period=period,
            line=line,
            outcome=o,
            price_fetch=pf,
            price_close=pc,
            p1_is_team1=False,  # в OddsPapi первым записан team2 с Liquipedia
        )
        for o, (pf, pc) in prices.items()
    ]


def test_market_probs_turns_line_to_team1():
    h = pd.DataFrame(
        _rows("moneyline", "result", 0.0, {"1": (1.43, 1.40), "2": (2.81, 2.90)})
        + _rows("spreads", "result", -1.5, {"1": (2.36, 2.36), "2": (1.598, 1.598)})
        + _rows("spreads", "result", 1.5, {"1": (1.151, 1.151), "2": (4.96, 4.96)})
        + _rows("totals", "result", 2.5, {"Over": (2.12, 2.12), "Under": (1.704, 1.704)})
    )
    r = vsmarket.market_probs(h).iloc[0]
    inv = lambda a, b: (1 / a) / (1 / a + 1 / b)  # noqa: E731
    assert r["q_fetch"] == pytest.approx(inv(2.81, 1.43))  # team1 — второй участник OddsPapi
    assert r["q_close"] == pytest.approx(inv(2.90, 1.40))
    assert (r["price1_fetch"], r["price2_fetch"]) == (2.81, 1.43)
    assert r["q02_fetch"] == pytest.approx(inv(2.36, 1.598))  # участник 1 OddsPapi (team2) выигрывает 2:0
    assert r["q20_fetch"] == pytest.approx(1 - inv(1.151, 4.96))
    assert r["q3_fetch"] == pytest.approx(inv(2.12, 1.704))


def test_bets_follow_rules_and_count_profit():
    df = pd.DataFrame(
        [
            dict(
                match_key="A",
                p_model=0.6,
                q_fetch=0.4,
                price1_fetch=2.4,
                price2_fetch=1.6,
                price1_close=2.2,
                price2_close=1.7,
                w1=2,
                w2=0,
            ),
            dict(
                match_key="B",
                p_model=0.5,
                q_fetch=0.5,
                price1_fetch=1.9,
                price2_fetch=1.9,
                price1_close=1.9,
                price2_close=1.9,
                w1=0,
                w2=2,
            ),
        ]
    )
    b = vsmarket.bets(df, w=0.5, min_edge=0.05)
    assert b[["match_key", "side"]].values.tolist() == [["A", 1]]  # в B перевеса нет
    s = vsmarket.summary_bets(b)
    assert (s["n"], s["roi"]) == (1, pytest.approx(1.4))
    assert s["clv"] == pytest.approx(2.4 / 2.2 - 1)


def test_blend_curve_endpoints():
    y = np.array([1, 0, 1, 1])
    c = vsmarket.blend_curve(y, np.array([0.9, 0.2, 0.8, 0.7]), np.full(4, 0.5)).set_index("w")
    assert c.loc[0.0, "logloss"] == pytest.approx(np.log(2))
    assert c.loc[1.0, "logloss"] < c.loc[0.0, "logloss"]
