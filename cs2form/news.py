"""Новости по командам ближайших матчей: лента HLTV + интервью с Liquipedia + заметки о составах.

Если задан ANTHROPIC_API_KEY, Claude делает по каждому матчу короткую сводку на русском
и расставляет флаги риска (замена, стендин, болезнь, визы, смена капитана и т. п.).
"""

from __future__ import annotations

import json
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import pandas as pd

RSS_URL = "https://www.hltv.org/rss/news"
MODEL = "claude-opus-5-5"
FLAGS = [
    "замена игрока",
    "стендин",
    "болезнь/травма",
    "визы/переезд",
    "смена капитана",
    "смена тренера",
    "уход игрока объявлен",
    "конфликт/давление",
    "мотивация/важность матча",
    "хорошая форма игрока",
]


def fetch_rss() -> pd.DataFrame:
    from curl_cffi import requests

    r = requests.get(RSS_URL, impersonate="chrome", timeout=30)
    r.raise_for_status()
    return parse_rss(r.text)


def parse_rss(xml: str) -> pd.DataFrame:
    root = ET.fromstring(xml)
    rows = []
    for it in root.iter("item"):
        pub = it.findtext("pubDate") or ""
        try:
            day = parsedate_to_datetime(pub).astimezone(timezone.utc).date().isoformat()
        except (TypeError, ValueError):
            day = ""
        rows.append(
            dict(
                date=day,
                title=(it.findtext("title") or "").strip(),
                description=re.sub(r"<[^>]+>", "", it.findtext("description") or "").strip()[:400],
                link=(it.findtext("link") or "").strip(),
                source="HLTV",
            )
        )
    return pd.DataFrame(rows)


def _aliases(team_id: str, name: str) -> list[str]:
    out = {name.lower(), str(team_id).lower()}
    for prefix in ("team ",):
        if name.lower().startswith(prefix):
            out.add(name.lower()[len(prefix) :])
    return [a for a in out if len(a) >= 3]


def team_news(team_id: str, name: str, rss: pd.DataFrame, media: pd.DataFrame, since: str) -> list[dict]:
    al = _aliases(team_id, name)
    pat = re.compile(r"\b(?:" + "|".join(re.escape(a) for a in al) + r")\b", re.I)
    items = []
    if rss is not None and not rss.empty:
        for r in rss[rss["date"] >= since].itertuples():
            if pat.search(f"{r.title} {r.description}"):
                items.append(dict(date=r.date, title=r.title, text=r.description, link=r.link, source="HLTV"))
    if media is not None and not media.empty:
        m = media[
            (media["date"].fillna("") >= since)
            & ((media["team"].fillna("") == str(team_id).lower()) | media["title"].fillna("").str.contains(pat))
        ]
        for r in m.itertuples():
            items.append(
                dict(
                    date=r.date,
                    title=r.title,
                    text=f"интервью: {r.player}",
                    link=r.link,
                    source=r.source or "Liquipedia",
                )
            )
    seen, out = set(), []
    for it in sorted(items, key=lambda x: x["date"], reverse=True):
        if it["link"] not in seen:
            seen.add(it["link"])
            out.append(it)
    return out[:15]


def latest_roster(team_name: str, rosters: pd.DataFrame) -> dict:
    if rosters is None or rosters.empty:
        return {}
    r = rosters[rosters["team_name"].str.lower() == team_name.lower()]
    if r.empty:
        return {}
    row = r.iloc[-1]
    notes = re.sub(r"\{\{player\|([^|}]+)[^}]*\}\}", r"\1", str(row.get("notes", "") or ""))
    notes = re.sub(r"\{\{|\}\}|'''|\[\[|\]\]|Notes\s*\|", " ", notes)
    return {"players": row["players"], "notes": re.sub(r"\s+", " ", notes).strip()}


SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "team1_flags": {"type": "array", "items": {"type": "string", "enum": FLAGS}},
        "team2_flags": {"type": "array", "items": {"type": "string", "enum": FLAGS}},
        "edge": {"type": "string", "enum": ["team1", "team2", "none"]},
    },
    "required": ["summary", "team1_flags", "team2_flags", "edge"],
    "additionalProperties": False,
}

SYSTEM = (
    "Ты аналитик CS2. По новостям, интервью и заметкам о составах двух команд перед матчем напиши сводку "
    "на русском, 2–4 предложения: только факты, которые могут повлиять на исход (замены, стендины, болезни, "
    "визы, смена ролей, уход игроков, заявления о форме). Не выдумывай: если существенного нет, так и скажи. "
    "Флаги ставь только при явном подтверждении в источниках. edge — у кого контекст лучше, или none."
)


def summarize(match: dict, n1: list, n2: list, r1: dict, r2: dict) -> dict:
    import anthropic

    client = anthropic.Anthropic()
    payload = {
        "match": f"{match['team1']} vs {match['team2']}, {match['event']}, {match['date']}",
        "team1": {"name": match["team1"], "roster": r1, "news": n1},
        "team2": {"name": match["team2"], "roster": r2, "news": n2},
    }
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=2000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        system=SYSTEM,
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("Claude отказался делать сводку")
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


def build_news(data: Path, days_ahead: int = 3, lookback: int = 14) -> pd.DataFrame:
    today = datetime.now(timezone.utc).date()
    since = (today - timedelta(days=lookback)).isoformat()
    rss_path = data / "rss.csv"
    rss_old = pd.read_csv(rss_path, dtype=str) if rss_path.exists() else pd.DataFrame()
    try:
        rss = pd.concat([rss_old, fetch_rss()]).drop_duplicates("link", keep="last")
        rss = rss[rss["date"] >= (today - timedelta(days=60)).isoformat()]
        rss.to_csv(rss_path, index=False)
    except Exception as e:  # noqa: BLE001
        print(f"::warning::Лента HLTV не скачалась: {e}", flush=True)
        rss = rss_old
    read = lambda n: pd.read_csv(data / n, dtype=str) if (data / n).exists() else pd.DataFrame()  # noqa: E731
    upcoming, media, rosters = read("upcoming.csv"), read("media.csv"), read("rosters.csv")
    if upcoming.empty:
        return pd.DataFrame()
    soon = upcoming[upcoming["date"] <= (today + timedelta(days=days_ahead)).isoformat()]
    old = read("news.csv")
    done = set(old["match_key"]) if not old.empty and "summary" in old else set()
    use_claude = bool(os.environ.get("ANTHROPIC_API_KEY"))
    rows = []
    for m in soon.to_dict("records"):
        n1 = team_news(m["team1_id"], m["team1"], rss, media, since)
        n2 = team_news(m["team2_id"], m["team2"], rss, media, since)
        r1, r2 = latest_roster(m["team1"], rosters), latest_roster(m["team2"], rosters)
        row = dict(
            match_key=m["match_key"],
            date=m["date"],
            team1=m["team1"],
            team2=m["team2"],
            news1=json.dumps(n1, ensure_ascii=False),
            news2=json.dumps(n2, ensure_ascii=False),
            roster1=json.dumps(r1, ensure_ascii=False),
            roster2=json.dumps(r2, ensure_ascii=False),
            summary="",
            team1_flags="",
            team2_flags="",
            edge="",
        )
        if use_claude and m["match_key"] not in done:
            try:
                s = summarize(m, n1, n2, r1, r2)
                row.update(
                    summary=s["summary"],
                    team1_flags="; ".join(s["team1_flags"]),
                    team2_flags="; ".join(s["team2_flags"]),
                    edge=s["edge"],
                )
            except Exception as e:  # noqa: BLE001
                print(f"::warning::Сводка для {m['team1']} — {m['team2']} не получилась: {e}", flush=True)
        elif m["match_key"] in done:
            prev = old[old["match_key"] == m["match_key"]].iloc[-1]
            row.update({k: prev.get(k, "") for k in ("summary", "team1_flags", "team2_flags", "edge")})
        rows.append(row)
    news = pd.DataFrame(rows)
    news.to_csv(data / "news.csv", index=False)
    print(
        f"::notice::Новости: {len(news)} матчей, со сводкой Claude: {(news['summary'].fillna('') != '').sum()}",
        flush=True,
    )
    return news
