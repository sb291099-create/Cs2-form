"""Дашборд формы команд CS2. Запуск: streamlit run app.py"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

import json


from cs2form import metrics, model, value

DATA = Path(__file__).parent / "data"

st.set_page_config(page_title="Форма команд CS2", page_icon="🎯", layout="wide")


def _read(name, **kw):
    path = DATA / name
    return pd.read_csv(path, **kw) if path.exists() else pd.DataFrame()


@st.cache_data(ttl=3600)
def load():
    maps = _read("maps.csv", dtype={"team1_id": str, "team2_id": str})
    if maps.empty:
        return None, None, {}
    maps["map"] = maps["map"].fillna("")
    ranking = _read("ranking.csv")
    extra = {n: _read(f"{n}.csv", dtype=str) for n in ("upcoming", "news", "rosters", "team_names")}
    return maps, (ranking if not ranking.empty else None), extra


@st.cache_resource(ttl=3600)
def deep_model(maps: pd.DataFrame, rosters: pd.DataFrame, team_names: pd.DataFrame):
    """Глубокая модель: Elo по картам, форма текущего состава, опыт на карте; обучение и проверка на истории."""
    if "page" in maps:
        event_dates = maps.groupby("page")["date"].min().to_dict()
        name_to_id = dict(zip(team_names["name"], team_names["id"])) if not team_names.empty else {}
        changes = model.roster_changes(rosters, event_dates, name_to_id)
    else:
        changes = {}
    state, feat = model.build(maps, changes)
    bt = model.backtest(feat)
    warm = feat[feat["date"] >= feat["date"].min() + pd.Timedelta(days=45)]
    mdl = model.MapModel().fit(warm[model.FEATURES].values, warm["y"].values)
    return state, mdl, bt, model.active_pool(maps), changes


maps, ranking, extra = load()
st.title("🎯 Форма команд CS2")
if maps is None or maps.empty:
    st.info("Данных пока нет: они появятся после первого запуска сбора (GitHub Actions → «Обновить данные»).")
    st.stop()

elo = metrics.run_elo(maps)
table = metrics.form_table(elo)
if ranking is not None:

    def _norm(name):
        return str(name).lower().replace("team ", "").replace(" esports", "").replace(" gaming", "").strip()

    hltv_rank = {_norm(t): r for t, r in zip(ranking["team"], ranking["rank"])}
    table["hltv_rank"] = table["team"].map(lambda t: hltv_rank.get(_norm(t)))
has_rounds = bool((maps[["score1", "score2"]].max(axis=1) > 1).any())
st.caption(
    f"Карт в базе: {len(maps)} · период {maps['date'].min()} — {maps['date'].max()} · "
    "карты и составы: Liquipedia, рейтинг и новости: HLTV.org"
)

names = table.set_index("team_id")["team"].to_dict()
team_ids = list(table["team_id"])
all_maps = sorted(m for m in maps["map"].value_counts().head(8).index if m)[:7]

has_maps = bool((maps["map"] != "").mean() > 0.5)
if has_maps:
    state, mdl, bt, pool, roster_changes = deep_model(maps, extra.get("rosters"), extra.get("team_names"))
today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()


def fair(p):
    return 100 / p if p > 0 else float("inf")


def render_value(p_a, name_a, name_b, key):
    """Кэфы букмекера → есть ли смысл ставить и на кого."""
    st.markdown("**Кэфы букмекера на победу в матче**")
    c1, c2, c3 = st.columns([1, 1, 3])
    oa = c1.number_input(name_a, min_value=1.0, value=None, step=0.01, format="%.2f", key=f"oa-{key}")
    ob = c2.number_input(name_b, min_value=1.0, value=None, step=0.01, format="%.2f", key=f"ob-{key}")
    if not (oa and ob and oa > 1 and ob > 1):
        c3.caption("Введи оба кэфа, и модель скажет, есть ли перевес и на чьей стороне.")
        return
    r = value.assess(p_a, oa, ob, name_a, name_b)
    c3.markdown(
        f"Рынок: {name_a} {r.market_a * 100:.0f}% · {name_b} {r.market_b * 100:.0f}% (маржа {r.margin * 100:.1f}%)  \n"
        f"Модель: {name_a} {p_a * 100:.0f}% · {name_b} {(1 - p_a) * 100:.0f}%  \n"
        f"Для ставки (среднее модели и рынка): {name_a} {r.p_used * 100:.0f}% · {name_b} {(1 - r.p_used) * 100:.0f}%  \n"
        f"Перевес: {name_a} {r.edge_a * 100:+.0f}% · {name_b} {r.edge_b * 100:+.0f}%"
    )
    (st.success if r.pick else st.info)(r.verdict)
    st.caption(
        "Перевес считается по среднему модели и рынка: рынок знает о заменах и стендинах, которых нет в статистике. "
        "Ставка только при перевесе от 5%. "
        "Размер — четверть Келли, не больше 2% банка."
    )


def render_forecast(a, b, name_a, name_b, bestof, seed=0.0):
    veto_maps = None
    if bestof > 1:
        veto_maps = st.multiselect(
            "Карты по факту вето (в порядке игры)",
            pool,
            max_selections=bestof,
            key=f"veto-{a}-{b}-{bestof}-{seed}",
            help=f"Выбери {bestof} карты, когда вето объявлено, и прогноз пересчитается под них.",
        )
    fc = model.forecast(mdl, state, a, b, pool, today, bestof, seed, veto_maps)
    if veto_maps and len(veto_maps) == bestof:
        st.caption("Прогноз посчитан по фактическому вето: " + ", ".join(veto_maps) + ".")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{name_a}: шанс на матч", f"{fc.p_series * 100:.0f}%")
    c2.metric(f"{name_b}: шанс на матч", f"{(1 - fc.p_series) * 100:.0f}%")
    c3.metric(f"Кф {name_a} без маржи", f"{fair(fc.p_series * 100):.2f}")
    c4.metric(f"Кф {name_b} без маржи", f"{fair((1 - fc.p_series) * 100):.2f}")
    if bestof > 1 and fc.scores:
        full = fc.p_full_distance
        cols = st.columns(len(fc.scores) + 2)
        for col, (sc, pr) in zip(cols, sorted(fc.scores.items(), key=lambda x: (-x[0][0], x[0][1]))):
            col.metric(f"Счёт {sc[0]}:{sc[1]}", f"{pr * 100:.0f}%")
        cols[-2].metric(
            f"Тотал Б {bestof - 0.5}",
            f"{fair(full * 100):.2f}",
            help=f"Шанс {full * 100:.0f}%",
        )
        cols[-1].metric(
            f"Тотал М {bestof - 0.5}",
            f"{fair((1 - full) * 100):.2f}",
            help=f"Шанс {(1 - full) * 100:.0f}%",
        )
        st.caption(
            "Счёт и тоталы учитывают инерцию: победитель карты чаще берёт и следующую. "
            "На истории до третьей карты доходят около 40% серий bo3."
        )
    render_value(fc.p_series, name_a, name_b, key=f"{a}-{b}-{bestof}-{seed}")
    left, right = st.columns([1, 1])
    with left:
        who = {"A": name_a, "B": name_b, "-": "—"}
        act = {"ban": "❌ убирает", "pick": "✅ выбирает", "decider": "🎲 десайдер"}
        if veto_maps and len(veto_maps) == bestof:
            st.markdown("**Карты матча**")
            rows = [{"Карта": m, f"Шанс {name_a} %": fc.map_p[m] * 100} for m in fc.played]
        else:
            st.markdown("**Прогноз вето**")
            rows = [
                {
                    "Команда": who[t],
                    "Действие": act[x],
                    "Карта": m,
                    f"Шанс {name_a} %": fc.map_p[m] * 100,
                }
                for t, x, m in fc.veto
            ]
        st.dataframe(
            pd.DataFrame(rows).style.format({f"Шанс {name_a} %": "{:.0f}"}),
            hide_index=True,
            width="stretch",
        )
    with right:
        st.markdown("**Шанс на каждой карте**")
        ca, cb = (
            model.comfort(state, a, pool, today),
            model.comfort(state, b, pool, today),
        )
        sa, sb = model.side_stats(maps, a, today), model.side_stats(maps, b, today)

        def sides(s, m):
            return f"{s[m][0] * 100:.0f} / {s[m][1] * 100:.0f}" if m in s else "—"

        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Карта": m,
                        f"Шанс {name_a} %": p * 100,
                        f"Сыграно {name_a}": ca[m],
                        f"Сыграно {name_b}": cb[m],
                        f"CT / T {name_a} %": sides(sa, m),
                        f"CT / T {name_b} %": sides(sb, m),
                        "Elo карты " + name_a: state.map_elo[(a, m)],
                        "Elo карты " + name_b: state.map_elo[(b, m)],
                    }
                    for m, p in sorted(fc.map_p.items(), key=lambda x: -x[1])
                ]
            )
            .style.format(
                {
                    f"Шанс {name_a} %": "{:.0f}",
                    "Elo карты " + name_a: "{:.0f}",
                    "Elo карты " + name_b: "{:.0f}",
                }
            )
            .background_gradient(subset=[f"Шанс {name_a} %"], cmap="RdYlGn", vmin=20, vmax=80),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "«Сыграно» — карт за 90 дней. Карты, которые команда не играет, вето считает её пермабаном. "
            "«CT / T» — доля раундов, выигранных за каждую сторону за 90 дней (основное время)."
        )
    for t, nm in ((a, name_a), (b, name_b)):
        ch = state.last_change(t, today)
        if ch is not None and (today - ch).days <= 60:
            st.info(f"У {nm} менялся состав {ch:%d.%m.%Y}: форма считается только по матчам нового состава.")
    return fc


tabs = ["📅 Матчи дня"] if has_maps else []
tabs += ["📊 Рейтинг формы", "🔎 Команда", "⚔️ Сравнение"] + (["🧪 Модель"] if has_maps else []) + ["📒 Журнал"]
tab_objs = st.tabs(tabs)
if has_maps:
    tab_today, tab_rank, tab_team, tab_vs, tab_model, tab_log = tab_objs
else:
    tab_rank, tab_team, tab_vs, tab_log = tab_objs

if has_maps:
    with tab_today:
        upcoming = extra.get("upcoming", pd.DataFrame())
        news = extra.get("news", pd.DataFrame())
        if upcoming.empty:
            st.info("Ближайших матчей в данных нет.")
        else:
            days = st.slider("Дней вперёд", 0, 7, 2)
            soon = upcoming[pd.to_datetime(upcoming["date"]) <= today + pd.Timedelta(days=days)]
            soon = soon.assign(_t=soon["tier"].map({"s": 0, "a": 1, "b": 2}).fillna(3)).sort_values(["date", "_t"])
            labels = [
                f"{r.date[8:]}.{r.date[5:7]} · {r.team1} — {r.team2} · {str(r.event).replace('_', ' ')} ({str(r.tier).upper()})"
                for r in soon.itertuples()
            ]
            if not labels:
                st.info("В выбранном окне матчей нет.")
            else:
                pick = st.selectbox("Матч", range(len(labels)), format_func=lambda i: labels[i])
                m = soon.iloc[pick]
                bo = int(float(m.get("bestof") or 3))
                st.subheader(f"{m['team1']} — {m['team2']}")
                st.caption(
                    f"{m['event']} · {str(m.get('time', '')).split('{')[0]} · bo{bo} · "
                    f"{'LAN' if str(m.get('lan')) == 'True' else 'онлайн'}"
                )
                render_forecast(m["team1_id"], m["team2_id"], m["team1"], m["team2"], bo, seed=1.0)
                st.caption(
                    f"{m['team1']} записана в сетке первой: на истории такие команды выигрывают чаще, "
                    "и модель это учитывает (около +5 п.п.)."
                )
                st.markdown("#### Новости и составы")
                nrow = news[news["match_key"] == m["match_key"]] if not news.empty else pd.DataFrame()
                if nrow.empty:
                    st.caption("Новостей по этому матчу ещё не собрано (собираются на 3 дня вперёд).")
                else:
                    n = nrow.iloc[-1]
                    if isinstance(n.get("summary"), str) and n["summary"]:
                        st.success(f"**Сводка Claude:** {n['summary']}")
                        for nm, fl in (
                            (m["team1"], n.get("team1_flags")),
                            (m["team2"], n.get("team2_flags")),
                        ):
                            if isinstance(fl, str) and fl:
                                st.warning(f"⚠️ {nm}: {fl}")
                    else:
                        st.caption("Сводка Claude появится после добавления ключа ANTHROPIC_API_KEY.")
                    for nm, col, rcol in (
                        (m["team1"], "news1", "roster1"),
                        (m["team2"], "news2", "roster2"),
                    ):
                        items = json.loads(n[col]) if isinstance(n.get(col), str) else []
                        roster = json.loads(n[rcol]) if isinstance(n.get(rcol), str) else {}
                        with st.expander(f"{nm}: {len(items)} новостей · состав: {roster.get('players', '—')}"):
                            if roster.get("notes"):
                                st.caption(roster["notes"])
                            for it in items:
                                st.markdown(f"- {it['date']} [{it['title']}]({it['link']}) · {it['source']}")

with tab_rank:
    st.markdown(
        "**Elo** — сила команды с учётом соперников и счёта. "
        "**Форма** — насколько команда за последние 90 дней выигрывает чаще (+) или реже (−), "
        "чем ей предсказывал рейтинг; свежие карты весят больше (полураспад 30 дней)."
    )
    show = table.rename(
        columns={
            "team": "Команда",
            "hltv_rank": "HLTV #",
            "elo": "Elo",
            "form": "Форма",
            "maps": "Карт (90 дн.)",
            "winrate": "Винрейт %",
            "weighted_winrate": "Винрейт свеж. %",
            "round_diff": "Раунды ±",
            "last5": "Последние 5",
            "streak": "Серия",
        }
    )
    cols = (
        ["Команда"]
        + (["HLTV #"] if "HLTV #" in show else [])
        + [
            "Elo",
            "Форма",
            "Винрейт %",
            "Винрейт свеж. %",
            "Раунды ±",
            "Карт (90 дн.)",
            "Последние 5",
            "Серия",
        ]
    )
    if not has_rounds:
        cols.remove("Раунды ±")
    st.dataframe(
        show[cols]
        .style.format(
            {
                "Elo": "{:.0f}",
                "Форма": "{:+.1f}",
                "Винрейт %": "{:.0f}",
                "Винрейт свеж. %": "{:.0f}",
                "Раунды ±": "{:+.1f}",
                "HLTV #": "{:.0f}",
            },
            na_rep="—",
        )
        .background_gradient(subset=["Форма"], cmap="RdYlGn", vmin=-20, vmax=20),
        hide_index=True,
        width="stretch",
        height=700,
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
        px.line(
            h,
            x="date",
            y="elo_after",
            markers=True,
            hover_data=["opp", "map", "rounds_for", "rounds_against"],
            labels={"date": "", "elo_after": "Elo"},
            title="Динамика Elo",
        ),
        width="stretch",
    )
    st.subheader("Карт-пул (90 дней)")
    pool = metrics.map_pool(elo, tid)
    pool = pool[pool["map"] != ""]
    if pool.empty:
        st.info("Источник пока не отдаёт названия карт, поэтому карт-пул недоступен.")
    else:
        st.dataframe(
            pool.rename(
                columns={
                    "map": "Карта",
                    "maps": "Сыграно",
                    "winrate": "Винрейт %",
                    "round_diff": "Раунды ±",
                    "map_elo": "Elo на карте",
                }
            ).style.format({"Винрейт %": "{:.0f}", "Раунды ±": "{:+.1f}", "Elo на карте": "{:.0f}"}),
            hide_index=True,
            width="stretch",
        )
    st.subheader("Последние карты")
    recent = h.sort_values("date", ascending=False).head(20)
    recent = recent.assign(
        Счёт=recent["rounds_for"].astype(str) + ":" + recent["rounds_against"].astype(str),
        Итог=recent["won"].map({1.0: "✅", 0.0: "❌"}),
        Шанс=recent["expected"] * 100,
    )
    st.dataframe(
        recent[["date", "opp", "map", "Счёт", "Итог", "Шанс", "event"]]
        .rename(
            columns={
                "date": "Дата",
                "opp": "Соперник",
                "map": "Карта",
                "Шанс": "Шанс по Elo %",
                "event": "Турнир",
            }
        )
        .style.format({"Дата": lambda d: d.strftime("%d.%m.%Y"), "Шанс по Elo %": "{:.0f}"}),
        hide_index=True,
        width="stretch",
    )

with tab_vs:
    c1, c2 = st.columns(2)
    a = c1.selectbox("Команда 1", team_ids, index=0, format_func=names.get, key="a")
    b = c2.selectbox(
        "Команда 2",
        team_ids,
        index=min(1, len(team_ids) - 1),
        format_func=names.get,
        key="b",
    )
    if a == b:
        st.warning("Выбери две разные команды.")
    elif has_maps:
        bo = st.radio(
            "Формат",
            [1, 3, 5],
            index=1,
            horizontal=True,
            format_func=lambda x: f"bo{x}",
        )
        render_forecast(a, b, names[a], names[b], bo)
    else:
        per_map, p_map, p_bo3 = metrics.matchup(elo, a, b, all_maps)
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{names[a]}: шанс на карту", f"{p_map:.0f}%")
        c2.metric(f"{names[a]}: шанс в bo3", f"{p_bo3:.0f}%")
        c3.metric(f"{names[b]}: шанс в bo3", f"{100 - p_bo3:.0f}%")
        st.caption("Справедливый коэффициент без маржи = 100 / шанс. Если у букмекера выше — это потенциальный value.")
        if per_map.empty:
            st.info("Названий карт в данных нет, поэтому шанс считается по общему Elo.")
        per_map = per_map.assign(fair_a=100 / per_map["prob_a"], fair_b=100 / (100 - per_map["prob_a"]))
        if not per_map.empty:
            st.dataframe(
                per_map.rename(
                    columns={
                        "map": "Карта",
                        "prob_a": f"Шанс {names[a]} %",
                        "maps_a": f"Сыграно {names[a]}",
                        "maps_b": f"Сыграно {names[b]}",
                        "fair_a": f"Кф {names[a]}",
                        "fair_b": f"Кф {names[b]}",
                    }
                ).style.format(
                    {
                        f"Шанс {names[a]} %": "{:.0f}",
                        f"Кф {names[a]}": "{:.2f}",
                        f"Кф {names[b]}": "{:.2f}",
                    }
                ),
                hide_index=True,
                width="stretch",
            )
    if a != b:
        h2h = maps[((maps.team1_id == a) & (maps.team2_id == b)) | ((maps.team1_id == b) & (maps.team2_id == a))]
        st.subheader(f"Личные встречи: {len(h2h)} карт")
        if not h2h.empty:
            st.dataframe(
                h2h.sort_values("date", ascending=False)[
                    ["date", "team1", "score1", "score2", "team2", "map", "event"]
                ],
                hide_index=True,
                width="stretch",
            )

if has_maps:
    with tab_model:
        st.markdown(
            "Модель учится на первых 60% карт по времени и проверяется на остальных, которых она не видела. "
            "**Log loss** — чем меньше, тем точнее вероятности (подбрасывание монеты = 0.693). "
            "Сравниваем с простым Elo без карт и состава."
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Log loss модели", f"{bt['logloss_model']:.3f}")
        c2.metric("Log loss простого Elo", f"{bt['logloss_elo']:.3f}")
        c3.metric("Угадано карт моделью", f"{bt['acc_model'] * 100:.1f}%")
        c4.metric("Угадано простым Elo", f"{bt['acc_elo'] * 100:.1f}%")
        st.caption(
            f"Проверка на {bt['test_maps']} картах с {bt['test_from']:%d.%m} по {bt['test_to']:%d.%m}, "
            f"обучение на {bt['train_maps']} картах."
        )
        st.markdown("**Калибровка:** когда модель говорит 60–70%, команда должна выигрывать примерно в 65% случаев.")
        cal = pd.DataFrame(bt["calibration"])
        st.dataframe(
            cal.rename(
                columns={
                    "bin": "Прогноз",
                    "n": "Карт",
                    "pred": "Средний прогноз",
                    "actual": "Реально выиграно",
                }
            ).style.format({"Средний прогноз": "{:.0%}", "Реально выиграно": "{:.0%}"}),
            hide_index=True,
            width="stretch",
        )
        st.markdown("**Веса признаков** (чем больше, тем сильнее влияет):")
        names_f = {
            "elo": "Общий Elo",
            "map_elo": "Elo на карте",
            "form": "Форма текущего состава",
            "map_exp": "Опыт на карте",
            "new_roster": "Свежая смена состава",
            "seed": "Первая в сетке (посев)",
        }
        st.dataframe(
            pd.DataFrame([{"Признак": names_f[k], "Вес": float(v)} for k, v in bt["weights"].items()]),
            hide_index=True,
        )
        st.caption(f"Активный пул карт: {', '.join(pool)}. Смен состава найдено: {len(roster_changes)} команд.")
        side = model.start_side_table(maps)
        if not side.empty:
            st.markdown(
                "**Стартовая сторона на решающих картах bo3** (сторону там решает нож, поэтому сравнение честное)"
            )
            st.dataframe(
                side.reset_index()
                .rename(
                    columns={
                        "map": "Карта",
                        "maps": "Карт",
                        "start_ct_won": "Начавшие за CT выиграли %",
                        "ct_round_share": "Раундов за CT %",
                    }
                )
                .assign(**{"Начавшие за CT выиграли %": lambda d: d["Начавшие за CT выиграли %"] * 100})
                .assign(**{"Раундов за CT %": lambda d: d["Раундов за CT %"] * 100})
                .style.format({"Начавшие за CT выиграли %": "{:.0f}", "Раундов за CT %": "{:.0f}"}),
                hide_index=True,
            )
            st.caption(
                "В среднем стартовая сторона карту не решает: начавшие за CT выигрывают около 51%. "
                "Отклонения на отдельных картах пока в пределах случайности, поэтому в модель они не добавлены."
            )

with tab_log:
    log_path = Path(__file__).parent / "data" / "bets_log.csv"
    if not log_path.exists():
        st.info("Журнал ставок пока пуст.")
    else:
        log = pd.read_csv(log_path)
        sm = value.journal_summary(log)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ставок сыграно", sm["bets"], help=f"Ждут результата: {sm['pending']}")
        c2.metric("Выиграно", f"{sm['wins']} из {sm['bets']}")
        c3.metric("Итог, % банка", f"{sm['profit']:+.2f}")
        c4.metric("ROI", f"{sm['roi'] * 100:+.0f}%")
        st.dataframe(
            log.rename(
                columns={
                    "date": "Дата",
                    "match": "Матч",
                    "market": "Рынок",
                    "pick": "Выбор",
                    "odds": "Кэф",
                    "stake": "Ставка, % банка",
                    "model_p": "Модель",
                    "market_p": "Рынок без маржи",
                    "status": "Итог",
                    "score": "Счёт",
                    "note": "Заметки",
                }
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Пропуски тоже записаны: по ним видно, не упускает ли модель выгодные ставки. "
            "Делать выводы о модели можно после 30–50 ставок, по нескольким матчам это шум."
        )
