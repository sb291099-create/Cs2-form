from pathlib import Path

from cs2form.liquipedia import parse_date, parse_page

WIKI = (Path(__file__).parent / "fixtures" / "epl24.wiki").read_text()


def test_parse_epl_page():
    p = parse_page("ESL/Pro League/Season 24", WIKI, 1)
    legacy_pv = [m for m in p.maps if m["team1"] == "legacy" and m["team2"] == "parivision"]
    assert [(m["map"], m["score1"], m["score2"]) for m in legacy_pv] == [
        ("Dust2", 13, 5),
        ("Inferno", 2, 13),
        ("Ancient", 12, 16),
    ]  # с овертаймом на Ancient
    assert all(m["lan"] and m["bestof"] == 3 and m["date"] == "2026-10-03" for m in legacy_pv)
    up = [u for u in p.upcoming if {u["team1"], u["team2"]} == {"parivision", "furia"}]
    assert len(up) == 1 and up[0]["hltv"] == "2398733"
    pv = [r for r in p.rosters if r["team_name"] == "PARIVISION"][0]
    assert "HObbit" in pv["players"] and "slaxejezzz" in pv["notes"]
    assert any(m["link"].startswith("https://www.hltv.org/") for m in p.media)


def test_parse_date():
    assert str(parse_date("October 5, 2026 - 11:00 {{Abbr/CEST}}")) == "2026-10-05"
    assert str(parse_date("2026-06-02")) == "2026-06-02"
