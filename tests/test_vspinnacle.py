import pandas as pd
import pytest

from cs2form import vspinnacle

START = "2026-09-20T17:00:00+00:00"


def _rows(book, market, line, prices, opened="2026-09-20T05:00:00+00:00"):
    return [
        dict(
            fixture_id="f1",
            match_key="EPL#5",
            start=START,
            book=book,
            market=market,
            period="result",
            line=line,
            outcome=o,
            price_fetch=pf,
            price_close=pc,
            quotes=5,
            opened=opened,
            p1_is_team1=False,  # participant1 OddsPapi — team2 Liquipedia
        )
        for o, (pf, pc) in prices.items()
    ]


def _hist():
    rows = []
    rows += _rows("pinnacle", "moneyline", 0.0, {"1": (1.50, 1.40), "2": (2.70, 3.00)})
    rows += _rows("fonbet", "moneyline", 0.0, {"1": (1.45, 1.45), "2": (2.95, 2.95)})
    rows += _rows("1xbet", "moneyline", 0.0, {"1": (1.40, 1.40), "2": (3.20, 3.20)}, opened=START)  # открылась поздно
    # знак форы перевёрнут и у Pinnacle (на утро и на закрытии), и у Stake
    rows += _rows("pinnacle", "spreads", 1.5, {"1": (2.30, 2.10), "2": (1.62, 1.75)})
    rows += _rows("pinnacle", "spreads", -1.5, {"1": (1.18, 1.15), "2": (4.60, 5.30)})
    rows += _rows("stake", "spreads", 1.5, {"1": (2.50, 2.50), "2": (1.50, 1.50)})
    rows += _rows("stake", "spreads", -1.5, {"1": (1.15, 1.15), "2": (5.00, 5.00)})
    return pd.DataFrame(rows)


RES = pd.DataFrame(
    {"w1": [2], "w2": [0], "bestof": [3], "tier": ["s"], "date": ["2026-09-20"]}, index=pd.Index(["EPL#5"])
)


@pytest.mark.parametrize(
    "market,line,outcome,a,b,want",
    [
        ("moneyline", 0.0, "1", 2, 1, 1.0),
        ("moneyline", 0.0, "2", 2, 1, 0.0),
        ("spreads", -1.5, "1", 2, 1, 0.0),
        ("spreads", -1.5, "2", 2, 1, 1.0),
        ("spreads", 1.5, "2", 0, 2, 1.0),
        ("totals", 2.5, "Over", 2, 1, 1.0),
        ("totals", 2.5, "Under", 2, 1, 0.0),
    ],
)
def test_settle(market, line, outcome, a, b, want):
    assert vspinnacle.settle(market, line, outcome, a, b) == want


def test_results_keeps_finished_series_only():
    maps = pd.DataFrame(
        {
            "match_key": ["A", "A", "B"],
            "score1": [13, 9, 13],
            "score2": [7, 13, 4],
            "bestof": ["3", "3", "3"],
            "tier": ["s", "s", "b"],
            "date": ["2026-09-20"] * 3,
        }
    )
    r = vspinnacle.results(maps)
    assert list(r.index) == []  # A — 1:1, B — 1:0: обе серии не доиграны в данных
    maps.loc[len(maps)] = ["A", 13, 11, "3", "s", "2026-09-20"]
    assert vspinnacle.results(maps).loc["A", ["w1", "w2"]].tolist() == [2, 1]


def test_candidates_take_best_open_price_against_pinnacle():
    c = vspinnacle.candidates(_hist(), RES).set_index(["market", "line", "outcome"])
    ml = c.loc[("moneyline", 0.0, "2")]
    # 1xBet открылась позже утренней загрузки: берём Fonbet
    assert (ml["book"], ml["price"]) == ("fonbet", 2.95)
    q = (1 / 2.70) / (1 / 2.70 + 1 / 1.50)
    assert ml["edge"] == pytest.approx(q * 2.95 - 1)
    q_close = (1 / 3.00) / (1 / 3.00 + 1 / 1.40)
    assert ml["clv"] == pytest.approx(q_close * 2.95 - 1)
    assert ml["won"] == 1.0  # participant2 — team1 Liquipedia, выиграл 2:0
    # фора: после разворота «−1.5» участника 2 у Stake — 5.00, сверяем с развёрнутой линией Pinnacle
    hc = c.loc[("spreads", 1.5, "2")]
    assert (hc["book"], hc["price"]) == ("stake", 5.00)
    q20 = (1 / 4.60) / (1 / 4.60 + 1 / 1.18)
    assert hc["edge"] == pytest.approx(q20 * 5.0 - 1)
    q20_close = (1 / 5.30) / (1 / 5.30 + 1 / 1.15)
    assert hc["clv"] == pytest.approx(q20_close * 5.0 - 1)
    assert hc["won"] == 1.0


def test_summary_counts_roi_and_clv():
    c = pd.DataFrame(
        {"match_key": ["a", "b"], "price": [3.0, 2.0], "won": [1.0, 0.0], "edge": [0.05, 0.03], "clv": [0.1, -0.02]}
    )
    s = vspinnacle.summary(c)
    assert (s["n"], s["roi"], s["clv"], s["clv_pos"]) == (2, 0.5, 0.04, 0.5)
