import json
from datetime import datetime, timezone

from cs2form import odds

CAT = odds.catalog(
    [
        {
            "marketId": 171,
            "marketType": "moneyline",
            "period": "result",
            "handicap": 0,
            "outcomes": [{"outcomeId": 171, "outcomeName": "1"}, {"outcomeId": 172, "outcomeName": "2"}],
        },
        {
            "marketId": 1709,
            "marketType": "spreads",
            "period": "result",
            "handicap": -1.5,
            "outcomes": [{"outcomeId": 1709, "outcomeName": "1"}, {"outcomeId": 1710, "outcomeName": "2"}],
        },
        {"marketId": 999, "marketType": "player-kills", "period": "result", "handicap": 0, "outcomes": []},
    ]
)
FIXTURE = {
    "fixtureId": "id7",
    "participant1Name": "Team Spirit",
    "participant2Name": "M80",
    "startTime": "2026-10-07T17:00:00.000Z",
    "tournamentName": "ESL Pro League",
}


def _book(prices: dict, **extra) -> dict:
    markets = {}
    for (mid, oid), price in prices.items():
        markets.setdefault(str(mid), {"outcomes": {}})["outcomes"][str(oid)] = {"players": {"0": {"price": price}}}
    return {"markets": markets, **extra}


def test_flatten_bookmaker_odds():
    raw = {
        "bookmakerOdds": {
            "pinnacle": _book({(171, 171): 1.18, (171, 172): 5.1, (1709, 1709): 1.55, (1709, 1710): 2.44}),
            "1xbet": _book({(171, 172): 5.6, (999, 1): 2.0}),
            "ps3838": _book({(171, 172): 5.1}),  # копия линии Pinnacle
            "closedbook": _book({(171, 172): 9.0}, bookmakerIsActive=False),
        }
    }
    rows = odds.flatten(FIXTURE, raw, CAT)
    assert len(rows) == 5
    assert {r["bookmaker"] for r in rows} == {"pinnacle", "1xbet"}
    m80 = next(r for r in rows if r["bookmaker"] == "1xbet")
    assert (m80["price"], m80["market"], m80["outcome"], m80["p2"]) == (5.6, "moneyline", "2", "M80")
    spread = next(r for r in rows if r["market"] == "spreads" and r["outcome"] == "2")
    assert (spread["line"], spread["price"]) == (-1.5, 2.44)


def test_same_team():
    assert odds.same_team("Natus Vincere", "NAVI")
    assert odds.same_team("Team Spirit", "Spirit")
    assert odds.same_team("9z Team", "9z")
    assert not odds.same_team("Team Spirit", "Team Falcons")


def test_main_skips_second_fetch_same_day(tmp_path, monkeypatch, capsys):
    today = datetime.now(timezone.utc).date().isoformat()
    (tmp_path / "odds_state.json").write_text(json.dumps({"month": today[:7], "used": 8, "ok": today}))
    monkeypatch.setattr(odds, "DATA", tmp_path)
    monkeypatch.setattr(odds, "MORNING_UTC", range(24))
    monkeypatch.setenv("ODDS_API_KEY", "secret-key")
    monkeypatch.delenv("ODDS_FORCE", raising=False)

    def no_calls(*a, **k):
        raise AssertionError("запрос к API не нужен")

    monkeypatch.setattr(odds.Client, "get", no_calls)
    assert odds.main() == 0
    assert "уже скачаны" in capsys.readouterr().out
