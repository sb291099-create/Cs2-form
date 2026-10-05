from pathlib import Path

from cs2form.scraper import _is_challenge, parse_map_results, parse_ranking

FIX = Path(__file__).parent / "fixtures"


def test_parse_map_results():
    rows = parse_map_results((FIX / "stats_matches.html").read_text())
    assert len(rows) == 2
    r = rows[0]
    assert (r.map_id, r.team1_id, r.team1, r.score1, r.team2_id, r.team2, r.score2) == (
        201234, 9565, "Vitality", 13, 4494, "MOUZ", 7)
    assert r.map == "Mirage" and r.event.startswith("BLAST") and r.date == "2025-10-04"
    assert rows[1].date == "2025-10-03" and rows[1].score2 == 16


def test_parse_ranking():
    teams = parse_ranking((FIX / "ranking.html").read_text())
    assert [(t.rank, t.team, t.points) for t in teams] == [(1, "Vitality", 1000), (2, "MOUZ", 812)]


def test_challenge_detection():
    assert _is_challenge("<html><title>Just a moment...</title>")
    assert not _is_challenge("<html><title>HLTV</title>")
