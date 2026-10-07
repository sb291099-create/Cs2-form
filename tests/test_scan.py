from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cs2form import markets, model, scan, value

UP = pd.DataFrame(
    [
        dict(
            match_key="EPL#31",
            date="2026-10-07",
            team1="PARIVISION",
            team2="Natus Vincere",
            team1_id="parivision",
            team2_id="natus vincere",
            bestof="3",
        )
    ]
)


def _rows(book, market, line, prices):
    return [
        dict(
            match_key="EPL#31",
            start="2026-10-07T14:30:00.000Z",
            p1="Natus Vincere",  # в OddsPapi первым может стоять другой участник, чем на Liquipedia
            p2="Parivision",
            bookmaker=book,
            market=market,
            period="result",
            line=line,
            outcome=o,
            price=pr,
            main=True,
        )
        for o, pr in prices.items()
    ]


def _odds():
    rows = []
    rows += _rows("pinnacle", "moneyline", 0.0, {"1": 1.43, "2": 2.81})
    rows += _rows("fonbet", "moneyline", 0.0, {"1": 1.50, "2": 2.60})
    rows += _rows("mirror", "moneyline", 0.0, {"1": 2.81, "2": 1.43})  # перепутаны исходы
    rows += _rows("pinnacle", "spreads", -1.5, {"1": 2.36, "2": 1.598})
    rows += _rows("pinnacle", "spreads", 1.5, {"1": 1.151, "2": 4.96})
    rows += _rows("stake", "spreads", -1.5, {"1": 1.13, "2": 5.0})  # знак форы перевёрнут
    rows += _rows("stake", "spreads", 1.5, {"1": 2.5, "2": 1.48})
    return pd.DataFrame(rows)


def _sim():
    # PARIVISION (A): 2:0 в 3 из 10, 2:1 в 2, 1:2 в 2, 0:2 в 3
    a = np.array([2, 2, 2, 2, 2, 1, 1, 0, 0, 0])
    b = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 2])
    nan = np.full((10, 3), np.nan)
    return markets.Sim(a, b, nan, nan.copy(), 3)


def test_unflip_restores_handicap_sign():
    g = scan._unflip(_odds(), ml1=0.66)
    stake = g[(g["bookmaker"] == "stake")].set_index(["line", "outcome"])["price"]
    assert stake[(1.5, "2")] == 5.0  # PARIVISION −1.5 (2:0)
    assert stake[(-1.5, "2")] == 1.48  # PARIVISION +1.5
    pin = g[g["bookmaker"] == "pinnacle"].set_index(["line", "outcome"])["price"]
    assert pin[(1.5, "2")] == 4.96  # у Pinnacle знак верный, не трогаем


def test_unflip_fixes_own_team_handicap_records():
    # у Melbet фора записана для команды исхода: «−1.5 исход 2» — это 2:0 участника 2, то есть «+1.5» в нашей записи
    rows = _rows("pinnacle", "moneyline", 0.0, {"1": 1.571, "2": 2.43})
    rows += _rows("pinnacle", "spreads", -1.5, {"1": 2.66, "2": 1.487})
    rows += _rows("pinnacle", "spreads", 1.5, {"1": 1.201, "2": 4.55})
    rows += _rows("melbet", "moneyline", 0.0, {"1": 1.49, "2": 2.625})
    rows += _rows("melbet", "spreads", -1.5, {"1": 2.46, "2": 5.05})
    rows += _rows("melbet", "spreads", 1.5, {"1": 1.17, "2": 1.56})
    g = scan._unflip(pd.DataFrame(rows), ml1=0.611)
    mel = g[g["bookmaker"] == "melbet"].set_index(["line", "outcome"])["price"]
    assert (mel[(-1.5, "1")], mel[(-1.5, "2")]) == (2.46, 1.56)  # пара на фору участника 1 −1.5
    assert (mel[(1.5, "1")], mel[(1.5, "2")]) == (1.17, 5.05)
    for line in (-1.5, 1.5):  # сумма обратных кэфов пары — маржа конторы, всегда больше 1
        assert 1 / mel[(line, "1")] + 1 / mel[(line, "2")] > 1
    assert len(g) == len(rows)  # ничего не потеряли и не задвоили


def test_unflip_drops_contradictory_handicaps():
    rows = _rows("pinnacle", "moneyline", 0.0, {"1": 1.43, "2": 2.81})
    rows += _rows("odd", "spreads", -1.5, {"1": 2.0, "2": 2.0})
    rows += _rows("odd", "spreads", 1.5, {"1": 2.0, "2": 2.0})
    g = scan._unflip(pd.DataFrame(rows), ml1=0.66)
    assert len(g[g["bookmaker"] == "odd"]) == 4  # симметричные кэфы разворот не ломает
    rows += _rows("odd", "spreads", -1.5, {"1": 9.0, "2": 1.05})  # вторая, противоречивая запись той же линии
    g = scan._unflip(pd.DataFrame(rows), ml1=0.66)
    assert g[(g["bookmaker"] == "odd") & (g["line"] == -1.5)].empty


def test_clean_drops_book_far_from_pinnacle():
    o = _odds()
    ref, good = scan._clean(o[o["market"] == "moneyline"])
    assert ref["2"] == pytest.approx((1 / 2.81) / (1 / 2.81 + 1 / 1.43))
    assert set(good["bookmaker"]) == {"pinnacle", "fonbet"}


def test_scan_prices_and_orientation(monkeypatch):
    fc = SimpleNamespace(map_p={"Ancient": 0.5, "Cache": 0.5, "Inferno": 0.5}, played=["Ancient", "Cache", "Inferno"])
    monkeypatch.setattr(model, "forecast", lambda *a, **k: fc)
    monkeypatch.setattr(markets, "simulate", lambda *a, **k: _sim())
    r = scan.scan(_odds(), UP, None, None, [], None).set_index("market")

    win = r.loc["Победа PARIVISION"]
    q = (1 / 2.81) / (1 / 2.81 + 1 / 1.43)
    assert win["model_p"] == pytest.approx(0.5)
    assert win["market_p"] == pytest.approx(q)
    assert win["min_odds"] == pytest.approx((1 + value.MIN_EDGE) / q)  # модель не влияет
    assert (win["price"], win["books"]) == (2.81, 2)  # «mirror» с перепутанными исходами отброшен
    assert win["edge"] == pytest.approx(q * 2.81 - 1) and win["stake"] == 0
    assert bool(win["sharp"]) is True

    two_nil = r.loc["Фора PARIVISION -1.5 по картам (2:0)"]
    assert two_nil["model_p"] == pytest.approx(0.3)
    assert two_nil["price"] == 5.0 and "Stake 5.00" in two_nil["quotes"]
    one_map = r.loc["Фора PARIVISION +1.5 по картам (хотя бы карта)"]
    assert one_map["model_p"] == pytest.approx(0.7)
    assert r.loc["Победа Natus Vincere", "model_p"] == pytest.approx(0.5)
    assert "брать от" in scan.report(r.reset_index(), show_all=True)


def test_scan_bets_only_above_pinnacle_fair_price(monkeypatch):
    fc = SimpleNamespace(map_p={"Ancient": 0.5, "Cache": 0.5, "Inferno": 0.5}, played=["Ancient", "Cache", "Inferno"])
    monkeypatch.setattr(model, "forecast", lambda *a, **k: fc)
    monkeypatch.setattr(markets, "simulate", lambda *a, **k: _sim())
    odds = pd.concat([_odds(), pd.DataFrame(_rows("marathonbet", "moneyline", 0.0, {"1": 1.40, "2": 3.30}))])
    r = scan.scan(odds, UP, None, None, [], None).set_index("market")
    q = (1 / 2.81) / (1 / 2.81 + 1 / 1.43)
    pv = r.loc["Победа PARIVISION"]
    assert (pv["price"], pv["price_book"]) == (3.30, "marathonbet")
    assert pv["edge"] == pytest.approx(q * 3.30 - 1) and pv["stake"] > 0
    assert r.loc["Победа Natus Vincere", "stake"] == 0
    assert scan.sharp_winner(odds, "PARIVISION", "Natus Vincere") == pytest.approx(q)
    assert scan.sharp_winner(odds, "Natus Vincere", "PARIVISION") == pytest.approx(1 - q)
    assert scan.sharp_winner(odds, "M80", "Spirit") is None
    # без линии Pinnacle перевес считается по медиане контор, но ставка не предлагается
    soft = odds[odds["bookmaker"] != "pinnacle"]
    r2 = scan.scan(soft, UP, None, None, [], None).set_index("market")
    assert not bool(r2.loc["Победа PARIVISION", "sharp"]) and r2["stake"].max() == 0


def test_parse_vetoes_finds_match_and_maps():
    v = scan.parse_vetoes(["parivision=ancient, cache,INFERNO", "Nobody=Nuke"], UP, ["Ancient", "Cache", "Inferno"])
    assert v == {"EPL#31": ["Ancient", "Cache", "Inferno"]}
