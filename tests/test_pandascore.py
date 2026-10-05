from cs2form.pandascore import games_to_rows


def match(tier="s", games=None):
    return {
        "id": 1,
        "begin_at": "2026-10-01T15:00:00Z",
        "tournament": {"tier": tier, "name": "Playoffs"},
        "league": {"name": "BLAST"},
        "serie": {"full_name": "Fall 2026"},
        "opponents": [{"opponent": {"id": 10, "name": "Vitality"}}, {"opponent": {"id": 20, "name": "MOUZ"}}],
        "games": games
        or [
            {"id": 101, "finished": True, "winner": {"id": 10}, "begin_at": "2026-10-01T15:05:00Z"},
            {"id": 102, "finished": True, "winner": {"id": 20}, "map": {"name": "Nuke"},
             "teams": [{"team": {"id": 10}, "score": 9}, {"team": {"id": 20}, "score": 13}]},
            {"id": 103, "finished": False, "winner": {"id": None}},
        ],
    }


def test_games_to_rows():
    rows = games_to_rows([match()])
    assert [r["map_id"] for r in rows] == [101, 102]
    assert (rows[0]["score1"], rows[0]["score2"], rows[0]["map"]) == (1, 0, "")
    assert (rows[1]["score1"], rows[1]["score2"], rows[1]["map"]) == (9, 13, "Nuke")
    assert rows[0]["date"] == "2026-10-01" and rows[0]["event"] == "BLAST Fall 2026"


def test_tier_filter():
    assert games_to_rows([match(tier="d")]) == []
    assert len(games_to_rows([match(tier="d")], tiers=None)) == 2
