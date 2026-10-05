import pandas as pd
import pytest

from cs2form import metrics


def make_maps():
    rows, mid = [], 1
    # A стабильно бьёт B и C, B бьёт C; в конце C резко прибавляет.
    for day in range(1, 61):
        d = f"2025-08-{day:02d}" if day <= 31 else f"2025-09-{day - 31:02d}"
        for t1, n1, t2, n2, s1, s2 in [(1, "A", 2, "B", 13, 8), (1, "A", 3, "C", 13, 5), (2, "B", 3, "C", 13, 10)]:
            if day > 50 and t2 == 3:
                s1, s2 = s2, s1 + 3  # C выигрывает
            rows.append(
                dict(
                    map_id=mid,
                    date=d,
                    team1_id=t1,
                    team1=n1,
                    team2_id=t2,
                    team2=n2,
                    score1=s1,
                    score2=s2,
                    map="Mirage" if mid % 2 else "Nuke",
                    event="Test",
                )
            )
            mid += 1
    return pd.DataFrame(rows)


def test_elo_ordering_and_form():
    elo = metrics.run_elo(make_maps())
    t = metrics.form_table(elo).set_index("team")
    assert t.loc["A", "elo"] > t.loc["B", "elo"]
    # C на подъёме: выигрывает чаще, чем ожидал рейтинг
    assert t.loc["C", "form"] > 0 > t.loc["B", "form"]
    assert t.loc["C", "streak"].startswith("W")
    assert len(t.loc["A", "last5"]) == 5


def test_matchup_and_bo3():
    elo = metrics.run_elo(make_maps())
    per_map, p_map, p_bo3 = metrics.matchup(elo, 1, 2, ["Mirage", "Nuke", "Dust2"])
    assert p_map > 50 and p_bo3 > p_map  # фаворит в bo3 сильнее, чем на одной карте
    assert per_map.loc[per_map["map"] == "Dust2", "maps_a"].item() == 0
    assert metrics.bo3_prob(0.5) == pytest.approx(0.5)


def test_map_pool():
    elo = metrics.run_elo(make_maps())
    pool = metrics.map_pool(elo, 1)
    assert set(pool["map"]) == {"Mirage", "Nuke"} and pool["winrate"].min() > 80
