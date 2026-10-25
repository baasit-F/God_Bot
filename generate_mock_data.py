"""Synthetic NFL league for testing the pipeline end to end.

The simulated world has a known ground truth so you can check the engine finds
real edges and does not invent fake ones:

* 32 teams, 18-week regular seasons (17 games + 1 bye between weeks 5 and 14),
  with Thursday, Sunday (1pm / 4pm / SNF) and Monday night windows.
* Team strengths drift week to week; margins are drawn from Normal(mu, 13.5),
  with about 1.8 points of home field.
* Rest edges: a team off its bye gains 2.5 points and a road team on a
  Thursday short week loses 2.5. The sharp market only gradually prices these
  in (10% at open, 35% at decision time, 80% at close). This planted edge is
  deliberately larger than anything you should expect in the real NFL market:
  with ~270 games a season, a smaller one cannot be learned from 5 seasons.
* The sharp book's line carries noise that shrinks toward the close; "sharp
  money" (handle % minus ticket %) leans toward the eventual correction.
* Soft books shade popular teams, add their own noise and charge 4-5% vig, so
  line shopping across them matters.
* Spreads and totals are quoted for every game. Spread lines follow the same
  market margin as the moneyline. Totals come from team offence, defence and
  pace; the market only partly prices pace (20% / 40% / 85%) and the lower
  scoring of Thursday short-week games, so recent plays-per-game carries a
  learnable totals edge. Soft books shade spreads toward popular teams and
  totals toward the over by half a point at a time.

Usage:
    python generate_mock_data.py                 # 5 past seasons + the current week
    python generate_mock_data.py --seasons 3 --today 2026-10-25
"""

from __future__ import annotations

import argparse
import math
from datetime import date

import numpy as np
import pandas as pd

from config import settings
from data.storage import Storage
from models.lines import selection_probs

TEAMS = [
    "Arizona Cardinals", "Atlanta Falcons", "Baltimore Ravens", "Buffalo Bills", "Carolina Panthers",
    "Chicago Bears", "Cincinnati Bengals", "Cleveland Browns", "Dallas Cowboys", "Denver Broncos",
    "Detroit Lions", "Green Bay Packers", "Houston Texans", "Indianapolis Colts", "Jacksonville Jaguars",
    "Kansas City Chiefs", "Las Vegas Raiders", "Los Angeles Chargers", "Los Angeles Rams", "Miami Dolphins",
    "Minnesota Vikings", "New England Patriots", "New Orleans Saints", "New York Giants", "New York Jets",
    "Philadelphia Eagles", "Pittsburgh Steelers", "San Francisco 49ers", "Seattle Seahawks",
    "Tampa Bay Buccaneers", "Tennessee Titans", "Washington Commanders",
]
SHARP_BOOK = "pinnacle"
SHARP_VIG = 0.025
SOFT_BOOKS = {"draftkings": 0.045, "fanduel": 0.042, "betmgm": 0.050, "caesars": 0.048}

WEEKS = 18
BYE_WEEKS = {5: 2, 6: 4, 7: 4, 8: 4, 9: 2, 10: 4, 11: 4, 12: 4, 13: 2, 14: 2}  # week -> teams on bye
MARGIN_SD = 13.5
HOME_FIELD = 1.8
BYE_BONUS = 2.5
ROAD_SHORT_WEEK_PENALTY = 2.5
REST_PRICED = {"open": 0.10, "decision": 0.35, "close": 0.80}
TOTAL_SD = 10.0
LEAGUE_TOTAL = 44.0
PACE_POINTS = 1.5            # points of total per sd of each team's pace
THURSDAY_TOTAL = -2.0        # short-week games score less
TOTAL_PRICED = {"open": 0.20, "decision": 0.40, "close": 0.85}
SNAPSHOT_OFFSETS = {"open": pd.Timedelta(hours=24 * 5), "decision": pd.Timedelta(hours=4), "close": pd.Timedelta(minutes=5)}
# Kickoff windows as hours after midnight of the US Eastern game date, in UTC (EDT).
THURSDAY_NIGHT, MONDAY_NIGHT, SUNDAY_NIGHT = 24.25, 24.25, 24.33
SUNDAY_SLOTS = [17.0] * 9 + [20.08, 20.08, 20.42, 20.42]


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _price(p: float, vig: float) -> float:
    return round(max(1.01, 1.0 / (p * (1.0 + vig))), 2)


def _half(x: float) -> float:
    return round(x * 2.0) / 2.0


def _line_quotes(market: str, first: str, second: str, line_first: float, mu: float, sd: float,
                 vig: float, noise: float, rng: np.random.Generator) -> list[tuple[str, float, float]]:
    """(selection, point, price) for both sides of a spread/total at `line_first`."""
    line_second = -line_first if market == "spreads" else line_first
    win, _, lose = selection_probs(market, np.array([first]), np.array([line_first]), mu, sd)
    p = float(np.clip(win[0] / (win[0] + lose[0]) + rng.normal(0, noise), 0.05, 0.95))
    return [(first, line_first, _price(p, vig)), (second, line_second, _price(1 - p, vig))]


def season_kickoff(year: int) -> pd.Timestamp:
    """Opening Thursday: the first Thursday on or after 5 September."""
    d = pd.Timestamp(year, 9, 5)
    return d + pd.Timedelta(days=(3 - d.weekday()) % 7)


def generate(store: Storage, seasons: int = 5, today: pd.Timestamp | None = None, seed: int = 42) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    today = pd.Timestamp(today or date.today()).normalize()
    current_year = today.year if today.month >= 8 else today.year - 1
    strength = dict(zip(TEAMS, rng.normal(0, 5.5, len(TEAMS))))
    popularity = dict(zip(TEAMS, rng.normal(0, 1, len(TEAMS))))
    offence = dict(zip(TEAMS, rng.normal(0, 1.8, len(TEAMS))))
    defence = dict(zip(TEAMS, rng.normal(0, 1.8, len(TEAMS))))
    pace = dict(zip(TEAMS, rng.normal(0, 1, len(TEAMS))))

    games, odds, sentiment, team_stats = [], [], [], []

    for year in range(current_year - seasons, current_year + 1):
        for t in TEAMS:  # off-season: regress and reshuffle (draft, free agency)
            strength[t] = 0.65 * strength[t] + rng.normal(0, 3.0)
            offence[t] = 0.6 * offence[t] + rng.normal(0, 1.45)
            defence[t] = 0.6 * defence[t] + rng.normal(0, 1.45)
            pace[t] = 0.6 * pace[t] + rng.normal(0, 0.8)
        last_played: dict[str, pd.Timestamp] = {}
        # Assign byes: each team gets exactly one bye in weeks 5-14.
        bye_order = list(rng.permutation(TEAMS))
        byes: dict[int, list[str]] = {}
        for week, n in BYE_WEEKS.items():
            byes[week], bye_order = bye_order[:n], bye_order[n:]

        for week in range(1, WEEKS + 1):
            thursday = season_kickoff(year) + pd.Timedelta(weeks=week - 1)
            if thursday > today:
                break
            for t in TEAMS:
                strength[t] += rng.normal(0, 0.5)
            playing = [t for t in rng.permutation(TEAMS) if t not in byes.get(week, [])]
            pairs = [(playing[i], playing[i + 1]) for i in range(0, len(playing), 2)]
            sunday, monday = thursday + pd.Timedelta(days=3), thursday + pd.Timedelta(days=4)
            slots = [(thursday, THURSDAY_NIGHT)]
            slots += [(sunday, h) for h in SUNDAY_SLOTS[: len(pairs) - 3]]
            slots += [(sunday, SUNDAY_NIGHT), (monday, MONDAY_NIGHT)]

            for idx, ((home, away), (day, hours)) in enumerate(zip(pairs, slots)):
                gid = f"MOCK-{year}-W{week:02d}-{idx:02d}"
                commence = (day + pd.Timedelta(hours=hours)).tz_localize("UTC")
                rest = {t: (day - last_played[t]).days if t in last_played else None for t in (home, away)}
                bye_h, bye_a = (int(rest[t] is not None and rest[t] >= 13) for t in (home, away))
                short_a = int(rest[away] is not None and rest[away] <= 5)
                last_played[home] = last_played[away] = day

                base_mu = strength[home] - strength[away] + HOME_FIELD
                rest_effect = BYE_BONUS * (bye_h - bye_a) - ROAD_SHORT_WEEK_PENALTY * short_a
                true_mu = base_mu + rest_effect

                # Sharp market: error shrinks toward the close, rest edges priced in gradually.
                e_open = rng.normal(0, 1.8)
                e_dec = 0.6 * e_open + rng.normal(0, 0.9)
                e_close = 0.35 * e_dec + rng.normal(0, 0.5)
                errs = {"open": e_open, "decision": e_dec, "close": e_close}
                m_mkt = {k: base_mu + REST_PRICED[k] * rest_effect + errs[k] for k in errs}
                p_mkt = {k: _phi(m / MARGIN_SD) for k, m in m_mkt.items()}

                # Totals: offence/defence fully priced, pace and short-week effects only partly.
                thursday_game = day.weekday() == 3
                base_total = LEAGUE_TOTAL + offence[home] + offence[away] - defence[home] - defence[away]
                hidden_total = PACE_POINTS * (pace[home] + pace[away]) + THURSDAY_TOTAL * thursday_game
                true_total = base_total + hidden_total
                t_open = rng.normal(0, 1.5)
                t_dec = 0.6 * t_open + rng.normal(0, 0.7)
                t_close = 0.35 * t_dec + rng.normal(0, 0.4)
                t_errs = {"open": t_open, "decision": t_dec, "close": t_close}
                t_mkt = {k: base_total + TOTAL_PRICED[k] * hidden_total + t_errs[k] for k in t_errs}

                final = day < today
                snaps = ["open", "decision", "close"] if final else ["open", "decision"]
                for snap in snaps:
                    ts = (commence - SNAPSHOT_OFFSETS[snap]).isoformat()
                    p = p_mkt[snap]
                    odds += [
                        (gid, SHARP_BOOK, "h2h", "home", _price(p, SHARP_VIG), None, ts),
                        (gid, SHARP_BOOK, "h2h", "away", _price(1 - p, SHARP_VIG), None, ts),
                    ]
                    spread_line, total_line = _half(-m_mkt[snap]), _half(t_mkt[snap])
                    for sel, pt, price in (
                        _line_quotes("spreads", "home", "away", spread_line, m_mkt[snap], MARGIN_SD, SHARP_VIG, 0.0, rng)
                        + _line_quotes("totals", "over", "under", total_line, t_mkt[snap], TOTAL_SD, SHARP_VIG, 0.0, rng)
                    ):
                        odds.append((gid, SHARP_BOOK, "spreads" if sel in ("home", "away") else "totals", sel, price, pt, ts))
                    for book, vig in SOFT_BOOKS.items():
                        ps = float(np.clip(p + 0.008 * (popularity[home] - popularity[away]) + rng.normal(0, 0.010), 0.03, 0.97))
                        odds += [
                            (gid, book, "h2h", "home", _price(ps, vig), None, ts),
                            (gid, book, "h2h", "away", _price(1 - ps, vig), None, ts),
                        ]
                        # Popular home side -> home line shaded worse; public overs -> total shaded up.
                        soft_spread = spread_line - _half(0.4 * (popularity[home] - popularity[away]) + rng.normal(0, 0.4))
                        soft_total = total_line + _half(rng.normal(0.2, 0.4))
                        for sel, pt, price in (
                            _line_quotes("spreads", "home", "away", soft_spread, m_mkt[snap], MARGIN_SD, vig, 0.010, rng)
                            + _line_quotes("totals", "over", "under", soft_total, t_mkt[snap], TOTAL_SD, vig, 0.010, rng)
                        ):
                            odds.append((gid, book, "spreads" if sel in ("home", "away") else "totals", sel, price, pt, ts))

                # Public tickets follow popularity and favourites; handle leans to the correction.
                fav = math.log(p_mkt["decision"] / (1 - p_mkt["decision"]))
                ticket = 1 / (1 + math.exp(-(0.6 * (popularity[home] - popularity[away]) + 1.2 * fav))) + rng.normal(0, 0.04)
                ticket = float(np.clip(ticket, 0.05, 0.95))
                sharp_signal = _phi((true_mu + 0.5 * e_close) / MARGIN_SD) - p_mkt["decision"]
                money = float(np.clip(ticket + 1.5 * sharp_signal + rng.normal(0, 0.04), 0.03, 0.97))
                sentiment.append((gid, (commence - SNAPSHOT_OFFSETS["decision"]).isoformat(), ticket, money))

                row = {
                    "game_id": gid, "sport": settings.sport_key, "season": str(year),
                    "commence_time": commence.isoformat(), "game_date": day.date().isoformat(),
                    "home_team": home, "away_team": away, "home_score": None, "away_score": None, "status": "scheduled",
                }
                if final:
                    raw = true_mu + rng.normal(0, MARGIN_SD)
                    margin = int(round(raw))
                    if margin == 0 and rng.random() > 0.15:  # most level games are settled in overtime
                        margin = 3 if raw >= 0 else -3
                    total = max(abs(margin) + 6, int(round(true_total + rng.normal(0, TOTAL_SD))))
                    hs = (total + margin) // 2
                    row.update(home_score=hs, away_score=hs - margin, status="final")
                    plays_h = 63 + 3 * pace[home] + rng.normal(0, 4)
                    plays_a = 63 + 3 * pace[away] + rng.normal(0, 4)
                    ypp_h = 5.4 + 0.045 * (row["home_score"] - 22) + rng.normal(0, 0.5)
                    ypp_a = 5.4 + 0.045 * (row["away_score"] - 22) + rng.normal(0, 0.5)
                    team_stats += [(gid, home, ypp_h, ypp_a, plays_h), (gid, away, ypp_a, ypp_h, plays_a)]
                games.append(row)

    store.reset()
    counts = {
        "games": store.upsert("games", pd.DataFrame(games)),
        "odds": store.upsert("odds", pd.DataFrame(odds, columns=["game_id", "bookmaker", "market", "selection", "price", "point", "snapshot_time"])),
        "sentiment": store.upsert("sentiment", pd.DataFrame(sentiment, columns=["game_id", "snapshot_time", "home_ticket_pct", "home_money_pct"])),
        "team_stats": store.upsert("team_stats", pd.DataFrame(team_stats, columns=["game_id", "team", "off_ypp", "def_ypp", "plays"])),
    }
    upcoming = sorted({g["game_date"] for g in games if g["status"] == "scheduled"})
    print(f"Synthetic data written to {store.db_path}")
    for table, n in counts.items():
        print(f"  {table:<11} {n:>9,} rows")
    n_up = sum(1 for g in games if g["status"] == "scheduled")
    print(f"  upcoming slate: {n_up} scheduled games on {', '.join(upcoming) or 'none (off-season)'}")
    return counts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seasons", type=int, default=5)
    p.add_argument("--today", default=date.today().isoformat())
    p.add_argument("--seed", type=int, default=settings.random_seed)
    p.add_argument("--db", default=None, help="SQLite path (default from config)")
    args = p.parse_args()
    generate(Storage(args.db), seasons=args.seasons, today=pd.Timestamp(args.today), seed=args.seed)


if __name__ == "__main__":
    main()
