import pandas as pd

from cs2form import history

CAT = {
    171: {
        "marketType": "moneyline",
        "period": "result",
        "handicap": 0,
        "outcomes": [{"outcomeId": 171, "outcomeName": "1"}, {"outcomeId": 172, "outcomeName": "2"}],
    },
    999: {"marketType": "totals-kills", "period": "result", "handicap": 150.5, "outcomes": []},
}
START = pd.Timestamp("2026-09-20 17:00", tz="UTC")


def test_parse_list_timeline_takes_morning_and_closing_price():
    h = {
        "bookmakerOdds": {
            "pinnacle": {
                "markets": {
                    "171": {
                        "outcomes": {
                            "172": {
                                "players": {
                                    "0": [
                                        {"createdAt": "2026-09-19T10:00:00", "price": 5.0},
                                        {"createdAt": "2026-09-20T05:00:00", "price": 5.6},
                                        {"createdAt": "2026-09-20T12:00:00", "price": 6.1},
                                        {"createdAt": "2026-09-20T17:30:00", "price": 9.0},  # уже идёт матч
                                    ]
                                }
                            }
                        }
                    },
                    "999": {
                        "outcomes": {"1": {"players": {"0": [{"createdAt": "2026-09-20T05:00:00", "price": 1.9}]}}}
                    },
                }
            }
        }
    }
    rows = history.parse(h, CAT, START)
    assert len(rows) == 1
    r = rows[0]
    assert (r["book"], r["market"], r["outcome"]) == ("pinnacle", "moneyline", "2")
    assert (r["price_fetch"], r["price_close"], r["quotes"]) == (5.6, 6.1, 3)


def test_parse_dict_timeline_keyed_by_epoch_ms():
    ms = int(pd.Timestamp("2026-09-20 16:00", tz="UTC").timestamp() * 1000)
    h = {
        "bookmakers": {
            "pinnacle": {"markets": {"171": {"outcomes": {"171": {str(ms): {"price": 1.2, "changedAt": ms}}}}}}
        }
    }
    r = history.parse(h, CAT, START)[0]
    assert (r["outcome"], r["price_close"], r["price_fetch"]) == ("1", 1.2, 1.2)  # до утра котировок нет — берём первую


def test_match_past_finds_orientation():
    maps = pd.DataFrame({"match_key": ["EPL#5"], "date": ["2026-09-20"], "team1": ["Team Spirit"], "team2": ["M80"]})
    fx = [
        {
            "fixtureId": "a",
            "startTime": "2026-09-20T17:00:00.000Z",
            "participant1Name": "M80",
            "participant2Name": "Spirit",
        },
        {
            "fixtureId": "b",
            "startTime": "2026-09-25T17:00:00.000Z",
            "participant1Name": "M80",
            "participant2Name": "Spirit",
        },
    ]
    out = history.match_past(fx, maps)
    assert [(f["fixtureId"], k, p1) for f, k, p1 in out] == [("a", "EPL#5", False)]


def test_parse_keeps_book_from_multi_book_answer():
    q = [{"createdAt": "2026-09-20T05:00:00", "price": 1.8}]
    h = {
        "bookmakers": {
            b: {"markets": {"171": {"outcomes": {"171": {"players": {"0": q}}}}}} for b in ("fonbet", "1xbet")
        }
    }
    rows = history.parse(h, CAT, START)
    assert sorted(r["book"] for r in rows) == ["1xbet", "fonbet"]
    assert rows[0]["opened"].startswith("2026-09-20T05:00")


def test_known_rebuilds_fixtures_from_saved_history():
    old = pd.DataFrame(
        {
            "fixture_id": ["a", "a", "b"],
            "start": ["2026-09-20T17:00:00+00:00"] * 3,
            "p1": ["M80"] * 3,
            "p2": ["Spirit"] * 3,
            "tournament": ["EPL", "EPL", float("nan")],
            "match_key": ["EPL#5", "EPL#5", "EPL#6"],
            "p1_is_team1": [False, False, True],
        }
    )
    out = history._known(old)
    assert [(f["fixtureId"], k, p1, f["tournamentName"]) for f, k, p1 in out] == [
        ("a", "EPL#5", False, "EPL"),
        ("b", "EPL#6", True, ""),
    ]


def test_get_hist_retries_rate_limit_and_maps_404_to_none(monkeypatch):
    calls = []

    class Cl:
        def get(self, path, free=False, **params):
            calls.append(params["bookmakers"])
            if len(calls) == 1:
                raise RuntimeError("/historical-odds: HTTP 429 rate_limited")
            if params["fixtureId"] == "none":
                raise RuntimeError("/historical-odds: HTTP 404 No historical odds found.")
            return {"bookmakers": {}}

    monkeypatch.setattr(history.time, "sleep", lambda s: None)
    assert history._get_hist(Cl(), "a", "fonbet,1xbet") == {"bookmakers": {}}
    assert history._get_hist(Cl(), "none", "fonbet") is None
    assert calls == ["fonbet,1xbet", "fonbet,1xbet", "fonbet"]
