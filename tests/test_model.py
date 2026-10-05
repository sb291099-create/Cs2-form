import random

import numpy as np
import pandas as pd
import pytest

from cs2form import model

POOL = ["Ancient", "Anubis", "Cache", "Dust2", "Inferno", "Mirage", "Nuke"]


def synthetic(n=4000, seed=3):
    rnd = random.Random(seed)
    teams = [f"t{i}" for i in range(24)]
    base = {t: rnd.gauss(0, 1) for t in teams}
    by_map = {(t, m): rnd.gauss(0, 0.8) for t in teams for m in POOL}
    rows = []
    for k in range(n):
        a, b = rnd.sample(teams, 2)
        mp = rnd.choice(POOL)
        s = (base[a] + by_map[(a, mp)]) - (base[b] + by_map[(b, mp)])
        win = rnd.random() < 1 / (1 + np.exp(-s))
        lose = rnd.randint(3, 11)
        rows.append(
            dict(
                map_id=str(k),
                match_key=str(k),
                map_number=1,
                date=(pd.Timestamp("2026-04-01") + pd.Timedelta(hours=k)).date().isoformat(),
                team1_id=a,
                team2_id=b,
                score1=13 if win else lose,
                score2=lose if win else 13,
                map=mp,
            )
        )
    return pd.DataFrame(rows)


def test_model_beats_elo_on_map_specific_data():
    st, feat = model.build(synthetic())
    bt = model.backtest(feat, warmup_days=10)
    assert bt["logloss_model"] < bt["logloss_coin"]
    assert bt["logloss_model"] <= bt["logloss_elo"] + 0.005
    assert bt["weights"]["map_elo"] > 0


def test_series_prob():
    assert model.series_prob([0.5, 0.5, 0.5], 3) == pytest.approx(0.5)
    assert model.series_prob([0.6, 0.6, 0.6], 3) == pytest.approx(0.6**2 * (3 - 2 * 0.6))
    assert model.series_prob([0.7], 1) == pytest.approx(0.7)


def test_veto_picks_and_bans_sensibly():
    p = dict(zip(POOL, [0.2, 0.3, 0.45, 0.5, 0.55, 0.7, 0.8]))
    played = {m: 5 for m in POOL}
    veto = model.predict_veto(p, played, played, 3)
    assert [a for _, a, _ in veto] == ["ban", "ban", "pick", "pick", "ban", "ban", "decider"]
    assert veto[0][2] == "Ancient"  # A выбивает худшую для себя
    assert veto[1][2] == "Nuke"  # B выбивает лучшую для A
    assert veto[2][2] == "Mirage"  # A берёт лучшую из оставшихся
    # карта, которую команда не играет, выбивается первой
    veto2 = model.predict_veto(p, {**played, "Nuke": 0}, played, 3)
    assert veto2[0][2] == "Ancient" or veto2[0][2] == "Nuke"


def test_roster_change_detection():
    rosters = pd.DataFrame(
        [
            dict(page="E1", team_name="PARIVISION", players="a,b,c,d,e"),
            dict(page="E2", team_name="PARIVISION", players="a,b,c,d,f"),
        ]
    )
    ch = model.roster_changes(rosters, {"E1": "2026-08-01", "E2": "2026-10-01"}, {"PARIVISION": "parivision"})
    assert ch["parivision"] == [pd.Timestamp("2026-10-01")] and ch["parivision"].counts == [1]


def test_score_distribution_momentum():
    d0 = model.score_distribution([0.5, 0.5, 0.5], 3, momentum=0.0)
    assert d0[(2, 0)] == pytest.approx(0.25) and d0[(2, 1)] == pytest.approx(0.25)
    d = model.score_distribution([0.5, 0.5, 0.5], 3)
    assert sum(d.values()) == pytest.approx(1.0)
    assert d[(2, 0)] > 0.25  # с учётом инерции серий 2:0 больше
    fc = model.MatchForecast({}, [], [], 0.5, 0.5, d)
    assert fc.p_full_distance == pytest.approx(d[(2, 1)] + d[(1, 2)])


def test_forecast_uses_actual_veto():
    import pandas as pd

    from cs2form import model

    st = model.State()
    mdl = model.MapModel()
    mdl.w = np.zeros(len(model.FEATURES))
    pool = ["Ancient", "Mirage", "Inferno", "Nuke", "Dust2", "Anubis", "Cache"]
    fc = model.forecast(mdl, st, "a", "b", pool, pd.Timestamp("2026-10-05"), 3, 0.0, ["Ancient", "Mirage", "Inferno"])
    assert fc.played == ["Ancient", "Mirage", "Inferno"]
    assert abs(fc.p_series - 0.5) < 1e-9
