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


def test_score_distribution_day_form():
    d0 = model.score_distribution([0.5, 0.5, 0.5], 3, sigma=0.0)
    assert d0[(2, 0)] == pytest.approx(0.25) and d0[(2, 1)] == pytest.approx(0.25)
    d = model.score_distribution([0.5, 0.5, 0.5], 3)
    assert sum(d.values()) == pytest.approx(1.0)
    assert d[(2, 0)] > 0.25  # форма на день общая для серии: 2:0 чаще
    one = model.score_distribution([0.7], 1)
    assert one[(1, 0)] == pytest.approx(0.7, abs=1e-6)  # шанс отдельной карты не меняется
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


def test_side_stats():
    import pandas as pd

    from cs2form import model

    maps = pd.DataFrame(
        [
            dict(date="2026-10-01", team1_id="a", team2_id="b", map="Nuke", first1="ct", ct1=8, t1=5, ct2=7, t2=4),
            dict(date="2026-10-02", team1_id="c", team2_id="a", map="Nuke", first1="t", ct1=6, t1=2, ct2=6, t2=7),
        ]
    )
    ct, t, n = model.side_stats(maps, "a", "2026-10-05")["Nuke"]
    assert n == 2
    assert ct == pytest.approx((8 + 6) / (8 + 4 + 6 + 2))  # a за CT: 8 из 12 и 6 из 8
    assert t == pytest.approx((5 + 7) / (5 + 7 + 7 + 6))


def test_canonical_ids_merges_templates_but_not_countries():
    maps = pd.DataFrame(
        {
            "team1_id": ["spirit"] * 5 + ["team spirit", "magic.ru", "players.br"],
            "team2_id": ["x"] * 5 + ["y", "z", "w"],
        }
    )
    names = pd.DataFrame(
        {
            "id": ["spirit", "team spirit", "magic", "magic.ru", "players (russian team)", "players.br"],
            "name": ["Team Spirit", "Team Spirit", "Magic", "Magic", "Players", "Players"],
        }
    )
    ids = model.canonical_ids(maps, names)
    assert ids == {"team spirit": "spirit", "magic": "magic.ru"}  # российские и бразильские Players — разные
    assert model.canonicalize(maps, ids)["team1_id"].tolist()[5] == "spirit"
    assert model.name_to_id(names, ids)["Team Spirit"] == "spirit"


def test_lineups_and_rosters_state():
    maps = pd.DataFrame(
        {
            "page": ["E1", "E2"],
            "date": ["2026-08-01", "2026-10-01"],
            "team1_id": ["spirit", "spirit"],
            "team2_id": ["x", "x"],
        }
    )
    names = pd.DataFrame({"id": ["spirit"], "name": ["Team Spirit"]})
    rosters = pd.DataFrame(
        [
            dict(page="E1", team_name="Team Spirit", players="a,b,c,d,e"),
            dict(page="E2", team_name="Team Spirit", players="a,b,c,d,f,team=other"),
            dict(page="E2", team_name="Noname", players="g,h"),  # меньше четырёх — не состав
        ]
    )
    changes, squads = model.rosters_state(maps, names, rosters)
    assert changes["spirit"] == [pd.Timestamp("2026-10-01")]
    assert [d.date().isoformat() for d, _ in squads["spirit"]] == ["2026-08-01", "2026-10-01"]
    assert squads["spirit"][1][1] == ["a", "b", "c", "d", "f"]  # «team=other» — не игрок
    assert "noname" not in squads
    st = model.State()
    st.lineup = squads
    assert st.players("spirit", pd.Timestamp("2026-09-01")) == ["a", "b", "c", "d", "e"]
    assert st.players("spirit", pd.Timestamp("2026-07-01")) == ["a", "b", "c", "d", "e"]  # до первой заявки
    assert st.players("spirit", pd.Timestamp("2026-10-05"))[-1] == "f"
    assert st.players("x", pd.Timestamp("2026-10-05")) == []


def test_player_rating_follows_the_player_between_teams():
    day = pd.Timestamp("2026-09-01")
    st = model.State()
    st.lineup = {
        "strong": [(day, ["p1", "p2", "p3", "p4", "p5"])],
        "weak": [(day, ["w1", "w2", "w3", "w4", "w5"])],
        "new": [(day + pd.Timedelta(days=10), ["p1", "p2", "p3", "p4", "p5"])],
    }
    for i in range(10):
        model.update(st, "strong", "weak", "Nuke", day + pd.Timedelta(days=i), 13, 5)
    assert st.player_elo["p1"] > model.BASE > st.player_elo["w1"]
    # новая команда из тех же игроков: рейтинг состава высокий, хотя сама команда ещё без истории
    f = model.features(st, "new", "weak", "Nuke", day + pd.Timedelta(days=11))
    assert f["players"] > 0 and f["players_known"] == 1.0
    assert f["elo"] > 0  # хотя у самой команды истории нет
    # та же пятёрка в старой команде: рейтинг состава одинаков, но у новой команды нет наигранного Elo
    same = model.features(st, "new", "strong", "Nuke", day + pd.Timedelta(days=11))
    assert same["players"] == pytest.approx(0.0) and same["players_vs_team"] > 0
    assert model.features(st, "nobody", "weak", "Nuke", day)["players_known"] == 0.0


def test_round_rating_and_rest():
    day = pd.Timestamp("2026-09-01")
    st = model.State()
    for i in range(6):
        model.update(st, "a", "b", "Nuke", day + pd.Timedelta(days=i), 13, 4)
    assert st.round_rating("a", day + pd.Timedelta(days=6)) > 0
    f = model.features(st, "a", "b", "Nuke", day + pd.Timedelta(days=6))
    assert f["rounds"] > 0 and f["rounds_form"] > 0
    assert f["h2h"] > 0 and f["h2h_map"] > 0  # все личные встречи за «a»
    assert f["exp_all"] == 0.0 and f["rest"] == 0.0  # сыграли одинаково и в один день
    # без игр рейтинг по раундам тянется к нулю
    far = st.round_rating("a", day + pd.Timedelta(days=6 + int(model.R_HALF_LIFE)))
    assert 0 < far
    assert far == pytest.approx(st.rounds["a"])
    f2 = model.features(st, "a", "c", "Nuke", day + pd.Timedelta(days=30))
    assert f2["rest"] < 0  # «a» играла недавно, «c» не играла вовсе
