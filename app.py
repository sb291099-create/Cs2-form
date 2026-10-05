"""Дашборд формы команд CS2. Запуск: streamlit run app.py"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from cs2form import metrics

DATA = Path(__file__).parent / "data"

st.set_page_config(page_title="Форма команд CS2", page_icon="🎯", layout="wide")


@st.cache_data(ttl=3600)
def load():
    maps_path = DATA / "maps.csv"
    if not maps_path.exists():
        return None, None
    maps = pd.read_csv(maps_path)
    ranking_path = DATA / "ranking.csv"
    ranking = pd.read_csv(ranking_path) if ranking_path.exists() else None
    return maps, ranking


maps, ranking = load()
st.title("🎯 Форма команд CS2")
if maps is None or maps.empty:
    st.info("Данных пока нет: они появятся после первого запуска сбора с HLTV (GitHub Actions → «Обновить данные HLTV»).")
    st.stop()

elo = metrics.run_elo(maps)
table = metrics.form_table(elo)
if ranking is not None:
    table = table.merge(ranking[["team", "rank"]].rename(columns={"rank": "hltv_rank"}), on="team", how="left")
st.caption(f"Карт в базе: {len(maps)} · период {maps['date'].min()} — {maps['date'].max()} · источник: HLTV.org")

names = table.set_index("team_id")["team"].to_dict()
team_ids = list(table["team_id"])
all_maps = sorted(maps["map"].value_counts().head(7).index)

tab_rank, tab_team, tab_vs = st.tabs(["📊 Рейтинг формы", "🔎 Команда", "⚔️ Сравнение"])

with tab_rank:
    st.markdown(
        "**Elo** — сила команды с учётом соперников и счёта. "
        "**Форма** — насколько команда за последние 90 дней выигрывает чаще (+) или реже (−), "
        "чем ей предсказывал рейтинг; свежие карты весят больше (полураспад 30 дней)."
    )
    show = table.rename(
        columns={
            "team": "Команда", "hltv_rank": "HLTV #", "elo": "Elo", "form": "Форма",
            "maps": "Карт (90 дн.)", "winrate": "Винрейт %", "weighted_winrate": "Винрейт свеж. %",
            "round_diff": "Раунды ±", "last5": "Последние 5", "streak": "Серия",
        }
    )
    cols = ["Команда"] + (["HLTV #"] if "HLTV #" in show else []) + [
        "Elo", "Форма", "Винрейт %", "Винрейт свеж. %", "Раунды ±", "Карт (90 дн.)", "Последние 5", "Серия",
    ]
    st.dataframe(
        show[cols].style.format(
            {"Elo": "{:.0f}", "Форма": "{:+.1f}", "Винрейт %": "{:.0f}", "Винрейт свеж. %": "{:.0f}",
             "Раунды ±": "{:+.1f}", "HLTV #": "{:.0f}"}, na_rep="—"
        ).background_gradient(subset=["Форма"], cmap="RdYlGn", vmin=-20, vmax=20),
        hide_index=True, width="stretch", height=700,
    )

with tab_team:
    tid = st.selectbox("Команда", team_ids, format_func=names.get, key="team")
    row = table.set_index("team_id").loc[tid]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Elo", f"{row['elo']:.0f}")
    c2.metric("Форма", f"{row['form']:+.1f}")
    c3.metric("Винрейт (90 дн.)", f"{row['winrate']:.0f}%")
    c4.metric("Последние 5", row["last5"])
    h = elo.history[elo.history["team_id"] == tid]
    st.plotly_chart(
        px.line(h, x="date", y="elo_after", markers=True, hover_data=["opp", "map", "rounds_for", "rounds_against"],
                labels={"date": "", "elo_after": "Elo"}, title="Динамика Elo"),
        width="stretch",
    )
    st.subheader("Карт-пул (90 дней)")
    st.dataframe(
        metrics.map_pool(elo, tid).rename(columns={"map": "Карта", "maps": "Сыграно", "winrate": "Винрейт %",
                                                   "round_diff": "Раунды ±", "map_elo": "Elo на карте"})
        .style.format({"Винрейт %": "{:.0f}", "Раунды ±": "{:+.1f}", "Elo на карте": "{:.0f}"}),
        hide_index=True, width="stretch",
    )
    st.subheader("Последние карты")
    recent = h.sort_values("date", ascending=False).head(20)
    recent = recent.assign(Счёт=recent["rounds_for"].astype(str) + ":" + recent["rounds_against"].astype(str),
                           Итог=recent["won"].map({1.0: "✅", 0.0: "❌"}), Шанс=recent["expected"] * 100)
    st.dataframe(
        recent[["date", "opp", "map", "Счёт", "Итог", "Шанс", "event"]]
        .rename(columns={"date": "Дата", "opp": "Соперник", "map": "Карта", "Шанс": "Шанс по Elo %", "event": "Турнир"})
        .style.format({"Дата": lambda d: d.strftime("%d.%m.%Y"), "Шанс по Elo %": "{:.0f}"}),
        hide_index=True, width="stretch",
    )

with tab_vs:
    c1, c2 = st.columns(2)
    a = c1.selectbox("Команда 1", team_ids, index=0, format_func=names.get, key="a")
    b = c2.selectbox("Команда 2", team_ids, index=min(1, len(team_ids) - 1), format_func=names.get, key="b")
    if a == b:
        st.warning("Выбери две разные команды.")
    else:
        per_map, p_map, p_bo3 = metrics.matchup(elo, a, b, all_maps)
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{names[a]}: шанс на карту", f"{p_map:.0f}%")
        c2.metric(f"{names[a]}: шанс в bo3", f"{p_bo3:.0f}%")
        c3.metric(f"{names[b]}: шанс в bo3", f"{100 - p_bo3:.0f}%")
        st.caption("Справедливый коэффициент без маржи = 100 / шанс. Если у букмекера выше — это потенциальный value.")
        per_map = per_map.assign(fair_a=100 / per_map["prob_a"], fair_b=100 / (100 - per_map["prob_a"]))
        st.dataframe(
            per_map.rename(columns={"map": "Карта", "prob_a": f"Шанс {names[a]} %", "maps_a": f"Сыграно {names[a]}",
                                    "maps_b": f"Сыграно {names[b]}", "fair_a": f"Кф {names[a]}", "fair_b": f"Кф {names[b]}"})
            .style.format({f"Шанс {names[a]} %": "{:.0f}", f"Кф {names[a]}": "{:.2f}", f"Кф {names[b]}": "{:.2f}"}),
            hide_index=True, width="stretch",
        )
        h2h = maps[((maps.team1_id == a) & (maps.team2_id == b)) | ((maps.team1_id == b) & (maps.team2_id == a))]
        st.subheader(f"Личные встречи: {len(h2h)} карт")
        if not h2h.empty:
            st.dataframe(h2h.sort_values("date", ascending=False)[["date", "team1", "score1", "score2", "team2", "map", "event"]],
                         hide_index=True, width="stretch")
