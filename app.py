"""Дашборд формы команд CS2. Запуск: streamlit run app.py"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

import json


from cs2form import bank, markets, metrics, model, scan, value

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
    ids = model.canonical_ids(maps, extra["team_names"])
    maps, extra["upcoming"] = model.canonicalize(maps, ids), model.canonicalize(extra["upcoming"], ids)
    extra["ids"] = ids
    return maps, (ranking if not ranking.empty else None), extra


@st.cache_resource(ttl=3600)
def deep_model(maps: pd.DataFrame, rosters: pd.DataFrame, team_names: pd.DataFrame):
    """Глубокая модель: Elo по картам, форма текущего состава, опыт на карте; обучение и проверка на истории."""
    changes, squads = model.rosters_state(maps, team_names, rosters)
    state, feat = model.build(maps, changes, squads)
    bt = model.backtest(feat)
    warm = feat[feat["date"] >= feat["date"].min() + pd.Timedelta(days=45)]
    mdl = model.MapModel().fit(warm[model.FEATURES].values, warm["y"].values)
    return state, mdl, bt, model.active_pool(maps), changes, markets.round_table(feat, maps, mdl)


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
    state, mdl, bt, pool, roster_changes, round_tab = deep_model(maps, extra.get("rosters"), extra.get("team_names"))
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
        c3.caption("Введи оба кэфа, и я сравню их с честной ценой Pinnacle из утренних кэфов.")
        return
    q_ref = scan.sharp_winner(_read("odds.csv.gz"), name_a, name_b)
    r = value.assess(p_a, oa, ob, name_a, name_b, q_ref=q_ref)
    pin = (
        f"Pinnacle без маржи: {name_a} {q_ref * 100:.0f}% ({1 / q_ref:.2f}) · "
        f"{name_b} {(1 - q_ref) * 100:.0f}% ({1 / (1 - q_ref):.2f})  \n"
        f"Перевес: {name_a} {r.edge_a * 100:+.0f}% · {name_b} {r.edge_b * 100:+.0f}%  \n"
        if q_ref is not None
        else "Pinnacle: в утренних кэфах этого матча нет  \n"
    )
    c3.markdown(
        f"Твои кэфы без маржи: {name_a} {r.market_a * 100:.0f}% · {name_b} {r.market_b * 100:.0f}% "
        f"(маржа {r.margin * 100:.1f}%)  \n"
        + pin
        + f"Модель, для справки: {name_a} {p_a * 100:.0f}% · {name_b} {(1 - p_a) * 100:.0f}%"
    )
    (st.success if r.pick else st.info)(r.verdict)
    st.caption(
        f"Ставка — когда кэф выше честной цены Pinnacle хотя бы на {value.MIN_EDGE * 100:.0f}%. "
        "Модель в расчёт не входит: на 402 прошлых матчах линия Pinnacle угадывала точнее. "
        "Размер — четверть Келли, не больше 2% банка."
    )


@st.cache_data(ttl=3600, show_spinner="Считаю рынки…")
def markets_table(map_p, map_names, bestof, name_a, name_b, oa, ob, lines):
    return markets.ladder(list(map_p), round_tab, bestof, name_a, name_b, list(map_names), oa, ob, **dict(lines))


def render_markets(fc, name_a, name_b, bestof, key):
    """Лесенка рынков: если не победа, то фора, тоталы карт и раундов, ИТБ."""
    st.markdown("**Все рынки: форы, тоталы карт и раундов, ИТБ**")
    cols = st.columns(5)
    lines = {
        "rounds_line": cols[0].number_input(
            "Тотал раундов в матче", value=57.5, step=1.0, format="%.1f", key=f"rl-{key}"
        ),
        "hc_line": cols[1].number_input("Фора раундов в матче", value=4.5, step=1.0, format="%.1f", key=f"hl-{key}"),
        "map_total": cols[2].number_input(
            "Тотал раундов на карте", value=21.5, step=1.0, format="%.1f", key=f"mt-{key}"
        ),
        "map_hc": cols[3].number_input("Фора раундов на карте", value=3.5, step=1.0, format="%.1f", key=f"mh-{key}"),
        "team_total": cols[4].number_input("ИТБ раундов на карте", value=9.5, step=1.0, format="%.1f", key=f"tt-{key}"),
    }
    if bestof == 1:
        lines = {k: v for k, v in lines.items() if k in ("map_total", "map_hc", "team_total")}
    oa, ob = st.session_state.get(f"oa-{key}"), st.session_state.get(f"ob-{key}")
    odds = (oa, ob) if oa and ob and oa > 1 and ob > 1 else (None, None)
    names = [m for m in fc.played] if len(fc.played) == bestof else None
    df = markets_table(
        tuple(fc.map_p[m] for m in fc.played), tuple(names or []), bestof, name_a, name_b, *odds, tuple(lines.items())
    )
    df = df.assign(**{"Кэф букмекера": float("nan")})
    num = {c: st.column_config.NumberColumn(format="%.2f") for c in ("Справедливый кэф", "Брать от", "Ожидаемый кэф")}
    num.update({c: st.column_config.NumberColumn(format="%.0f") for c in ("Модель %", "Рынок %")})
    num["Кэф букмекера"] = st.column_config.NumberColumn(min_value=1.01, step=0.01, format="%.2f")
    edited = st.data_editor(
        df,
        column_config=num,
        disabled=[c for c in df.columns if c != "Кэф букмекера"],
        hide_index=True,
        width="stretch",
        key=f"mk-{key}",
    )
    got = edited[edited["Кэф букмекера"].notna()]
    if len(got) and "Рынок %" not in got:
        st.info("Сначала введи кэфы на победу (лучше Pinnacle): от них считается честная цена остальных рынков.")
        got = got.iloc[0:0]
    for _, r in got.iterrows():
        o = float(r["Кэф букмекера"])
        a = value.single(r["Модель %"] / 100, o, q_market=r["Рынок %"] / 100)
        text = f"{r['Группа']}: {r['Рынок']} по {o:.2f}: перевес к оценке {a.edge * 100:+.0f}%"
        if a.edge >= value.MIN_EDGE:
            st.success(f"{text}. Это оценка от кэфов на победу, а не линия Pinnacle: не больше 0.5% банка.")
        else:
            st.info(f"{text}, пропуск (нужно от {r['Брать от']:.2f}).")
    st.caption(
        "«Модель %» и «Справедливый кэф» — по модели, для справки: на прошлых матчах она уступила рынку. "
        "Когда введены кэфы на победу, «Рынок %» — шансы всех рынков при такой силе команд, как у букмекера, "
        f"а «Брать от» — кэф на {value.MIN_EDGE * 100:.0f}% выше этой честной цены. Это оценка: разброс счёта "
        "взят из истории (карты с похожим шансом пары, тем же победителем и той же длиной серии). "
        "«Ожидаемый кэф» — сколько, скорее всего, даст букмекер, если его линии согласованы с кэфами на победу. "
        "ИТБ 0.5 карты — то же, что фора +1.5 по картам."
    )


@st.cache_data(ttl=3600, show_spinner="Ищу выгодные ставки…")
def value_scan(fetched: str, n_maps: int, vetoes: tuple):
    """Прогноз модели против кэфов всех контор; fetched и n_maps — чтобы пересчитать при новых данных."""
    return scan.scan(_read("odds.csv.gz"), extra["upcoming"], mdl, state, pool, round_tab, vetoes=dict(vetoes))


def render_backtest():
    """Итоги проверки на прошлых матчах: модель против линии Pinnacle и ставки по кэфам выше её честной цены."""
    path = DATA / "backtest.json"
    if not path.exists():
        return
    bt_ = json.loads(path.read_text())
    m = bt_.get("model")
    if m:
        st.markdown(f"#### Модель против линии Pinnacle на прошлых матчах ({m['n']}, {m['since']} – {m['until']})")
        ll, acc, old = m["logloss"], m["accuracy"], m["old_rules"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pinnacle угадал победителя", f"{acc['Pinnacle утром'] * 100:.0f}%")
        c2.metric("Модель угадала", f"{acc['модель'] * 100:.0f}%")
        c3.metric("Log loss Pinnacle / модели", f"{ll['Pinnacle утром']:.3f} / {ll['модель']:.3f}")
        c4.metric("Старые правила, итог ставок", f"{old['roi'] * 100:+.0f}%", help=f"Ставок: {old['n']:.0f}")
        st.caption(
            "Прогноз модели строился так, как утром перед матчем: состояние команд до первой карты, вето по "
            "прогнозу, обучение только на картах до месяца матча. Линия Pinnacle — без маржи, на утреннюю "
            "загрузку кэфов. «Старые правила» — ставки по среднему модели и рынка с перевесом от 5% по кэфу "
            "Pinnacle. Поэтому шанс для ставки теперь — линия Pinnacle, а модель показывается для справки."
        )
    v = bt_.get("value")
    if v and v.get("thresholds"):
        st.markdown("#### Ставки по кэфам выше честной цены Pinnacle на тех же матчах")
        rows = [
            {
                "Перевес от": k,
                "Ставок": s.get("n", 0),
                "Выиграно %": s.get("won", 0) * 100,
                "Средний кэф": s.get("price"),
                "Итог на 1 ₽ %": s.get("roi", 0) * 100,
                "± ошибка %": (s.get("roi_se") or 0) * 100,
                "Перевес на закрытии %": (s.get("clv") or 0) * 100,
            }
            for k, s in v["thresholds"].items()
        ]
        st.dataframe(
            pd.DataFrame(rows),
            column_config={
                c: st.column_config.NumberColumn(format="%.1f")
                for c in ("Выиграно %", "Итог на 1 ₽ %", "± ошибка %", "Перевес на закрытии %")
            }
            | {"Средний кэф": st.column_config.NumberColumn(format="%.2f")},
            hide_index=True,
            width="stretch",
        )
        st.caption(v.get("note", ""))


def render_bank():
    """Виртуальный банк: ставки, которые Claude делает сам по сканеру, с итогами."""
    if not bank.BETS.exists():
        return
    rules, bets = bank.load_rules(), bank.load()
    b = bank.balance(bets, rules["start"])
    goal = rules.get("goal")
    st.markdown("#### 💰 Виртуальный банк: ставит Claude")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Банк", bank.money(b["bank"]), delta=bank.money(b["profit"], True) if b["settled"] else None)
    c2.metric("В игре", bank.money(b["in_play"]), help=f"Ставок ждут результата: {b['pending']}")
    c3.metric("Выиграно", f"{b['won']} из {b['settled']}")
    c4.metric("ROI", f"{b['roi'] * 100:+.0f}%")
    if goal:
        st.progress(
            min(max(b["bank"] / goal, 0.0), 1.0),
            text=f"Старт {bank.money(rules['start'])} · цель {bank.money(goal)} к {rules.get('until', '')} · "
            f"режим «{rules.get('mode', 'спокойно')}»",
        )
    if not bets.empty:
        view = bets.iloc[::-1].rename(
            columns={
                "start": "Начало, UTC",
                "match": "Матч",
                "bet": "Ставка",
                "odds": "Кэф",
                "book": "Контора",
                "stake": "Сумма",
                "status": "Итог",
                "score": "Счёт",
                "payout": "Выплата",
            }
        )
        st.dataframe(
            view[["Начало, UTC", "Матч", "Ставка", "Кэф", "Контора", "Сумма", "Итог", "Счёт", "Выплата"]],
            column_config={"Кэф": st.column_config.NumberColumn(format="%.2f")},
            hide_index=True,
            width="stretch",
        )
    st.caption(
        "Каждое утро Claude рассчитывает сыгранные ставки по счёту с Liquipedia и ставит на новые матчи: "
        "по одной ставке на матч, только где кэф выше честной цены Pinnacle, и из таких рынков — тот, "
        "что в среднем сильнее растит банк. Ставки 07.10 сделаны по старым правилам, со средним модели и рынка."
    )


def render_bets():
    """Выгодные ставки по скачанным кэфам: на победу, форы и тоталы карт, а при известном вето и на карты."""
    odds_df = _read("odds.csv.gz")
    if odds_df.empty:
        st.info(
            "Кэфы ещё не скачаны: они скачиваются каждое утро вместе с данными (GitHub Actions → «Обновить данные»)."
        )
        return
    fetched = str(odds_df["fetched"].max()) if "fetched" in odds_df else ""
    up = extra.get("upcoming", pd.DataFrame())
    have = up[up["match_key"].isin(set(odds_df["match_key"].dropna()))] if not up.empty else up
    st.caption(
        f"Кэфы {odds_df['bookmaker'].nunique()} контор на {len(have)} матчей, скачаны "
        f"{fetched[:16].replace('T', ' ')} UTC (OddsPapi). Скачиваются раз в сутки, утром."
    )
    vetoes = {}
    with st.expander("Вето, если уже известно: появятся ставки на отдельные карты (по линии Pinnacle)"):
        for r in have.itertuples():
            bo = int(float(r.bestof or 3))
            cols = st.columns([2] + [1] * bo)
            cols[0].markdown(f"{r.team1} — {r.team2}")
            veto = [
                cols[i + 1].selectbox(f"Карта {i + 1}", [""] + pool, key=f"veto-{r.match_key}-{i}") for i in range(bo)
            ]
            if all(veto):
                vetoes[r.match_key] = veto
    res = value_scan(fetched, len(maps), tuple(sorted((k, tuple(v)) for k, v in vetoes.items())))
    if res.empty:
        st.info("Матчей с кэфами в списке ближайших нет.")
        return
    show_all = st.toggle("Показать все рынки, а не только выгодные", key="bets-all")
    sel = res if show_all else res[res["edge"] >= value.MIN_EDGE]
    if sel.empty:
        st.info("Сейчас нет кэфов выше честной цены Pinnacle.")
    else:
        msk = pd.to_datetime(sel["start"], utc=True) + pd.Timedelta(hours=3)
        df = pd.DataFrame(
            {
                "Начало, МСК": msk.dt.strftime("%d.%m %H:%M"),
                "Матч": sel["match"],
                "Ставка": sel["market"],
                "Рынок %": sel["market_p"] * 100,
                "Модель %": sel["model_p"] * 100,
                "Брать от": sel["min_odds"],
                "Кэф": sel["price"],
                "Перевес %": sel["edge"] * 100,
                "Сумма, % банка": sel["stake"] * 100,
                "Крупные конторы": sel["quotes"],
                "Лучший кэф": sel["best_price"].map("{:.2f}".format) + " " + sel["best_book"],
                "Конторы от порога": sel["books_ok"].astype(str) + " из " + sel["books"].astype(str),
            }
        ).sort_values(["Начало, МСК", "Перевес %"], ascending=[True, False])
        num = {c: st.column_config.NumberColumn(format="%.0f") for c in ("Модель %", "Рынок %", "Перевес %")}
        num.update({c: st.column_config.NumberColumn(format="%.2f") for c in ("Брать от", "Кэф")})
        num["Сумма, % банка"] = st.column_config.NumberColumn(format="%.1f")
        st.dataframe(df, column_config=num, hide_index=True, width="stretch")
    no_value = sorted(set(res["match"]) - set(res.loc[res["edge"] >= value.MIN_EDGE, "match"]))
    if no_value and not show_all:
        st.caption("Без перевеса: " + "; ".join(no_value) + ".")
    st.caption(
        f"«Рынок %» — честный шанс по линии Pinnacle без маржи (если Pinnacle нет, медиана контор). «Брать от» — "
        f"кэф на {value.MIN_EDGE * 100:.0f}% выше этой честной цены: ниже него ставка не выгодна. «Кэф» — лучший у "
        "крупных контор (Fonbet, Marathon, Stake, 1xBet, bet365), перевес и сумма считаются по нему. Сумма — "
        "четверть Келли, не больше 2% банка. Если линии Pinnacle на матч нет, перевес считается по медиане "
        "контор и ставка не предлагается: на прошлых матчах такие ставки дали −32%. "
        "«Модель %» — для справки: на 402 прошлых матчах линия Pinnacle угадывала точнее, поэтому в расчёт "
        "модель не входит. Кэфы меняются: перед ставкой сверь свой с «Брать от»."
    )


def render_forecast(a, b, name_a, name_b, bestof, seed=0.0, lan=False):
    veto_maps = None
    if bestof > 1:
        veto_maps = st.multiselect(
            "Карты по факту вето (в порядке игры)",
            pool,
            max_selections=bestof,
            key=f"veto-{a}-{b}-{bestof}-{seed}",
            help=f"Выбери {bestof} карты, когда вето объявлено, и прогноз пересчитается под них.",
        )
    fc = model.forecast(mdl, state, a, b, pool, today, bestof, seed, veto_maps, lan)
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
            "Счёт и тоталы учитывают, что форма команды в день матча плавает: команда в ударе чаще забирает "
            "обе карты. На истории до третьей карты доходят 41% серий bo3, модель даёт столько же."
        )
    render_value(fc.p_series, name_a, name_b, key=f"{a}-{b}-{bestof}-{seed}")
    render_markets(fc, name_a, name_b, bestof, key=f"{a}-{b}-{bestof}-{seed}")
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


tabs = ["🔥 Ставки", "📅 Матчи дня"] if has_maps else []
tabs += ["📊 Рейтинг формы", "🔎 Команда", "⚔️ Сравнение"] + (["🧪 Модель"] if has_maps else []) + ["📒 Журнал"]
tab_objs = st.tabs(tabs)
if has_maps:
    tab_bets, tab_today, tab_rank, tab_team, tab_vs, tab_model, tab_log = tab_objs
else:
    tab_rank, tab_team, tab_vs, tab_log = tab_objs

if has_maps:
    with tab_bets:
        render_bank()
        st.markdown("#### 🔥 Выгодные ставки по кэфам")
        render_bets()

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
                render_forecast(
                    m["team1_id"],
                    m["team2_id"],
                    m["team1"],
                    m["team2"],
                    bo,
                    seed=1.0,
                    lan=str(m.get("lan")) == "True",
                )
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
        render_backtest()
        st.markdown("#### Точность модели по картам")
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
        st.markdown("**Что даёт каждый признак**: насколько хуже становится прогноз, если убрать его одного.")
        names_f = {
            "elo": "Общий Elo",
            "map_elo": "Elo на карте",
            "form": "Форма текущего состава",
            "map_exp": "Опыт на карте",
            "new_roster": "Свежая смена состава",
            "seed": "Первая в сетке (посев)",
            "rounds": "Рейтинг по разнице раундов",
            "rounds_form": "Разница раундов в последних матчах",
            "h2h": "Личные встречи за год",
            "h2h_map": "Личные встречи на этой карте",
            "exp_all": "Сколько карт сыграно всего",
            "rest": "Дней с последнего матча",
            "players": "Рейтинг пятёрки игроков",
            "players_known": "Составы известны",
            "lan_exp": "Опыт офлайна (на LAN)",
            "players_vs_team": "Состав сильнее самой команды",
        }
        imp = bt.get("importance", {})
        st.dataframe(
            pd.DataFrame(
                [
                    {"Признак": names_f.get(k, k), "Без него log loss хуже на": imp.get(k, 0.0), "Вес в формуле": v}
                    for k, v in bt["weights"].items()
                ]
            ).sort_values("Без него log loss хуже на", ascending=False),
            column_config={
                "Без него log loss хуже на": st.column_config.NumberColumn(format="%.4f"),
                "Вес в формуле": st.column_config.NumberColumn(format="%.3f"),
            },
            hide_index=True,
        )
        st.caption(
            "Признаки связаны между собой, поэтому отдельный вес в формуле сам по себе мало что значит: смотри "
            "первый столбец. Рейтинг игроков считается по турнирным составам Liquipedia: сила команды — среднее "
            "пяти рейтингов, поэтому при переходе игрок переносит силу с собой, а новая команда не начинает с нуля."
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
