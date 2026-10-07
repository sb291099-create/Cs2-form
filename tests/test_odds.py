from cs2form import odds


def test_flatten_bookmaker_odds():
    raw = {
        "participant1Name": "Team Spirit",
        "participant2Name": "M80",
        "bookmakerOdds": {
            "pinnacle": {
                "markets": {
                    "171": {
                        "outcomes": {
                            "171": {"players": {"0": {"price": 1.18}}},
                            "172": {"players": {"0": {"price": 5.1}}},
                        }
                    }
                }
            },
            "1xbet": {"markets": {"171": {"outcomes": {"172": {"players": {"0": {"price": 5.6}}}}}}},
        },
    }
    rows = odds.flatten(7, raw)
    assert len(rows) == 3
    assert {r["bookmaker"] for r in rows} == {"pinnacle", "1xbet"}
    assert next(r for r in rows if r["bookmaker"] == "1xbet")["price"] == 5.6


def test_same_team():
    assert odds.same_team("Natus Vincere", "NAVI")
    assert odds.same_team("Team Spirit", "Spirit")
    assert odds.same_team("9z Team", "9z")
    assert not odds.same_team("Team Spirit", "Team Falcons")
