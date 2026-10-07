"""Виртуальный банк, которым ставит Claude: правила в data/virtual_bank.json, ставки в data/virtual_bank.csv.

Каждое утро: рассчитать сыгранные ставки по счёту серий с Liquipedia (settle), затем поставить на матчи,
которые ещё не начались, по одной лучшей ставке на матч из сканера (place).
Запуск: python -m cs2form.bank [--settle] [--place]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import scan, value
from .odds import DATA, ODDS, same_team

BETS = DATA / "virtual_bank.csv"
RULES = DATA / "virtual_bank.json"
COLUMNS = [
    "placed",
    "start",
    "match_key",
    "match",
    "bet",
    "market",
    "line",
    "pick",
    "odds",
    "book",
    "model_p",
    "market_p",
    "stake",
    "status",
    "payout",
    "score",
]
PENDING = "ожидание"
FULL_KELLY_CAP = 0.10  # «агрессивно»: полный Келли, но не больше 10% банка на ставку


def load_rules() -> dict:
    return json.loads(RULES.read_text()) if RULES.exists() else {"start": 200_000, "mode": "спокойно"}


def load() -> pd.DataFrame:
    if not BETS.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(BETS, dtype={"match_key": str, "status": str, "score": str})


def save(bets: pd.DataFrame) -> None:
    bets[COLUMNS].to_csv(BETS, index=False)


def balance(bets: pd.DataFrame, start: float) -> dict:
    done = bets[bets["status"] != PENDING]
    profit = float((done["payout"].astype(float) - done["stake"].astype(float)).sum())
    in_play = float(bets.loc[bets["status"] == PENDING, "stake"].astype(float).sum())
    turnover = float(done["stake"].astype(float).sum())
    return dict(
        bank=start + profit,
        in_play=in_play,
        free=start + profit - in_play,
        profit=profit,
        roi=profit / turnover if turnover else 0.0,
        settled=len(done),
        won=int((done["status"] == "выигрыш").sum()),
        pending=int((bets["status"] == PENDING).sum()),
    )


def _series(g: pd.DataFrame) -> tuple[str, str, int, int, int]:
    s1, s2 = g["score1"].astype(int), g["score2"].astype(int)
    return g["team1"].iloc[0], g["team2"].iloc[0], int((s1 > s2).sum()), int((s2 > s1).sum()), int(g["bestof"].iloc[0])


def outcome(bet: pd.Series, maps: pd.DataFrame) -> tuple[str, str] | None:
    """Итог ставки на серию по картам матча или None, если серия ещё не сыграна."""
    g = maps[maps["match_key"] == bet["match_key"]]
    if g.empty:
        return None
    t1, t2, w1, w2, bo = _series(g)
    if max(w1, w2) < bo // 2 + 1:
        return None
    line = float(bet["line"])
    if bet["market"] == "totals":
        total = w1 + w2
        won = total > line if bet["pick"] == "Over" else total < line
        return ("выигрыш" if won else "проигрыш"), f"{w1}:{w2}"
    if not (same_team(bet["pick"], t1) or same_team(bet["pick"], t2)):
        return None  # команда записана иначе: рассчитать вручную
    mine, theirs = (w1, w2) if same_team(bet["pick"], t1) else (w2, w1)
    diff = mine - theirs + (line if bet["market"] == "spreads" else 0.0)
    status = "возврат" if diff == 0 else ("выигрыш" if diff > 0 else "проигрыш")
    return status, f"{mine}:{theirs}"


def settle(bets: pd.DataFrame, maps: pd.DataFrame) -> pd.DataFrame:
    bets = bets.copy()
    for i, b in bets[bets["status"] == PENDING].iterrows():
        res = outcome(b, maps)
        if res is None:
            continue
        status, score = res
        pay = {"выигрыш": float(b["stake"]) * float(b["odds"]), "возврат": float(b["stake"])}.get(status, 0.0)
        bets.loc[i, ["status", "payout", "score"]] = [status, round(pay), score]
    return bets


def _fraction(r: pd.Series, mode: str) -> float:
    if mode == "агрессивно":
        cap = FULL_KELLY_CAP if r.get("sharp", True) else FULL_KELLY_CAP / 4
        return min(max(0.0, (r["p_used"] * r["price"] - 1) / (r["price"] - 1)), cap)
    return float(r["stake"])


def choose(found: pd.DataFrame, bets: pd.DataFrame, mode: str, now: pd.Timestamp) -> pd.DataFrame:
    """По одной ставке на матч: ещё не начался, ставки на него не было, перевес от порога; больше всего по Келли."""
    if found.empty:
        return found
    ok = found[
        (found["edge"] >= value.MIN_EDGE)
        & (found["kind"] == "серия")
        & (pd.to_datetime(found["start"], utc=True) > now)
        & ~found["match_key"].isin(set(bets["match_key"]))
    ].copy()
    if ok.empty:
        return ok
    ok["fraction"] = [_fraction(r, mode) for _, r in ok.iterrows()]
    ok = ok[ok["fraction"] > 0]
    if mode == "ва-банк":  # весь свободный банк на один самый крупный кэф с перевесом
        return ok.sort_values("price", ascending=False).head(1).assign(fraction=1.0)
    # из лесенки рынков матча берём тот, что сильнее всего растит банк в среднем (рост логарифма банка)
    f, p, o = ok["fraction"], ok["p_used"], ok["price"]
    ok["growth"] = p * np.log1p(f * (o - 1)) + (1 - p) * np.log1p(-f)
    return ok.sort_values("growth", ascending=False).drop_duplicates("match_key")


def place(bets: pd.DataFrame, picks: pd.DataFrame, rules: dict, now: pd.Timestamp) -> pd.DataFrame:
    """Записывает ставки: сумма — доля банка на момент ставки, округлённая до 100, в пределах свободного банка."""
    bal = balance(bets, rules["start"])
    free, rows = bal["free"], []
    for _, r in picks.iterrows():
        stake = min(round(r["fraction"] * bal["bank"] / 100) * 100, int(free // 100) * 100)
        if stake < 100:
            continue
        free -= stake
        rows.append(
            dict(
                placed=now.strftime("%Y-%m-%d %H:%M"),
                start=str(r["start"])[:16].replace("T", " "),
                match_key=r["match_key"],
                match=r["match"],
                bet=r["market"],
                market=r["market_type"],
                line=r["pick_line"],
                pick=r["pick"],
                odds=r["price"],
                book=scan.SHOW_BOOKS.get(r["price_book"], r["price_book"]),
                model_p=round(float(r["model_p"]), 3),
                market_p=round(float(r["market_p"]), 3),
                stake=stake,
                status=PENDING,
                payout=0,
                score="",
            )
        )
    new = pd.DataFrame(rows, columns=COLUMNS)
    return pd.concat([bets, new], ignore_index=True) if len(bets) else new


def money(x: float, sign: bool = False) -> str:
    return (f"{x:+,.0f}" if sign else f"{x:,.0f}").replace(",", " ")


def summary(bets: pd.DataFrame, rules: dict) -> str:
    b = balance(bets, rules["start"])
    goal = rules.get("goal")
    lines = [
        f"Банк {money(b['bank'])} (старт {money(rules['start'])}"
        + (f", цель {money(goal)} к {rules.get('until', '')}" if goal else "")
        + f"), в игре {money(b['in_play'])}, рассчитано ставок {b['settled']}, выиграно {b['won']}, "
        f"итог {money(b['profit'], True)}, ROI {b['roi'] * 100:+.0f}%, режим «{rules.get('mode', 'спокойно')}»"
    ]
    for r in bets.itertuples():
        res = f"{r.status} {r.score}" if r.status != PENDING else PENDING
        lines.append(f"  {r.start} {r.match}: {r.bet} по {float(r.odds):.2f} ({r.book}), {money(r.stake)} → {res}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Виртуальный банк: расчёт и новые ставки")
    ap.add_argument("--settle", action="store_true", help="рассчитать сыгранные ставки")
    ap.add_argument("--place", action="store_true", help="поставить на матчи с перевесом")
    args = ap.parse_args()
    rules, bets = load_rules(), load()
    now = pd.Timestamp(datetime.now(timezone.utc))
    if args.settle:
        bets = settle(bets, pd.read_csv(DATA / "maps.csv", dtype={"match_key": str}))
    if args.place and ODDS.exists():
        ctx = scan.prepare()
        found = scan.scan(pd.read_csv(ODDS), ctx["upcoming"], ctx["mdl"], ctx["st"], ctx["pool"], ctx["table"])
        bets = place(bets, choose(found, bets, rules.get("mode", "спокойно"), now), rules, now)
    if args.settle or args.place:
        save(bets)
    print(summary(bets, rules))
    return 0


if __name__ == "__main__":
    sys.exit(main())
