import pandas as pd
import pytest

from cs2form import bank

NOW = pd.Timestamp("2026-10-07 09:00", tz="UTC")
MAPS = pd.DataFrame(
    {
        "match_key": ["EPL#31", "EPL#31", "EPL#32"],
        "team1": ["PARIVISION", "PARIVISION", "Team Spirit"],
        "team2": ["Natus Vincere", "Natus Vincere", "M80"],
        "score1": [13, 13, 13],
        "score2": [7, 9, 5],
        "bestof": [3, 3, 3],
    }
)


def _bet(**kw):
    base = dict(match_key="EPL#31", market="moneyline", line=0.0, pick="PARIVISION", stake=1000, odds=2.81)
    return pd.Series({**base, **kw})


@pytest.mark.parametrize(
    "kw, status",
    [
        (dict(), "выигрыш"),
        (dict(market="spreads", line=-1.5), "выигрыш"),
        (dict(market="spreads", line=1.5, pick="Natus Vincere"), "проигрыш"),
        (dict(market="totals", line=2.5, pick="Under"), "выигрыш"),
        (dict(market="totals", line=2.5, pick="Over"), "проигрыш"),
    ],
)
def test_outcome_by_series_score(kw, status):
    assert bank.outcome(_bet(**kw), MAPS)[0] == status


def test_outcome_waits_for_finished_series():
    assert bank.outcome(_bet(match_key="EPL#32", pick="M80"), MAPS) is None  # сыграна одна карта
    assert bank.outcome(_bet(match_key="EPL#99"), MAPS) is None


def test_settle_and_balance():
    bets = pd.DataFrame(
        [
            dict(match_key="EPL#31", market="spreads", line=-1.5, pick="PARIVISION", odds=5.05, stake=2000),
            dict(match_key="EPL#32", market="moneyline", line=0.0, pick="M80", odds=6.3, stake=3500),
        ]
    ).assign(status=bank.PENDING, payout=0, score="")
    done = bank.settle(bets, MAPS)
    assert done.loc[0, ["status", "payout", "score"]].tolist() == ["выигрыш", 10100, "2:0"]
    assert done.loc[1, "status"] == bank.PENDING
    b = bank.balance(done, 200_000)
    assert (b["bank"], b["in_play"], b["free"], b["won"]) == (208_100, 3500, 204_600, 1)
    assert b["roi"] == pytest.approx(8100 / 2000)


def _found(**kw):
    base = dict(
        match_key="A",
        start="2026-10-07T17:00:00.000Z",
        match="X – Y",
        market="Победа Y",
        kind="серия",
        market_type="moneyline",
        pick="Y",
        pick_line=0.0,
        model_p=0.27,
        market_p=0.16,
        p_used=0.218,
        price=6.3,
        price_book="fonbet",
        edge=0.374,
        stake=0.0176,
        sharp=True,
    )
    return {**base, **kw}


def test_choose_one_bet_per_open_match():
    found = pd.DataFrame(
        [
            _found(),
            _found(market="Фора Y +1.5", market_type="spreads", pick_line=1.5, p_used=0.45, price=2.44, edge=0.097),
            _found(match_key="B", start="2026-10-07T08:00:00.000Z"),  # уже начался
            _found(match_key="C"),  # на этот матч уже ставили
            _found(match_key="D", edge=0.03),  # перевес ниже порога
        ]
    )
    ledger = pd.DataFrame({"match_key": ["C"]})
    picks = bank.choose(found, ledger, "спокойно", NOW)
    assert picks[["match_key", "market"]].values.tolist() == [["A", "Победа Y"]]
    allin = bank.choose(found, ledger, "ва-банк", NOW)
    assert len(allin) == 1 and allin["fraction"].iloc[0] == 1.0


def test_place_rounds_stakes_within_free_bank():
    picks = pd.DataFrame([_found(fraction=0.0176), _found(match_key="E", fraction=0.6)])
    bets = bank.place(pd.DataFrame(columns=bank.COLUMNS), picks, {"start": 200_000}, NOW)
    assert bets["stake"].tolist() == [3500, 120_000]
    assert (bets["status"] == bank.PENDING).all() and bets["book"].iloc[0] == "Fonbet"
    more = bank.place(bets, pd.DataFrame([_found(match_key="F", fraction=0.6)]), {"start": 200_000}, NOW)
    assert more["stake"].iloc[-1] == 76_500  # свободно 200 000 − 123 500


def test_choose_skips_bets_without_pinnacle_line():
    found = pd.DataFrame([_found(sharp=False), _found(match_key="B")])
    picks = bank.choose(found, pd.DataFrame({"match_key": []}), "агрессивно", NOW)
    assert picks["match_key"].tolist() == ["B"]  # без линии Pinnacle перевес не проверить
    assert picks["fraction"].iloc[0] == pytest.approx(min((0.218 * 6.3 - 1) / 5.3, bank.FULL_KELLY_CAP))
