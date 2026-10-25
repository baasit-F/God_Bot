"""Synthetic NBA-style league for testing the pipeline end to end.

The simulated world has a known ground truth so you can check the engine finds
real edges and does not invent fake ones:

* Team strengths drift day to day; results are drawn from Normal(margin, 12).
* Back-to-backs cost a team 2.5 points. The sharp market only gradually
  prices this in (30% at open, 60% at decision time, 85% at close), which is
  a genuine, learnable inefficiency.
* The sharp book's line carries noise that shrinks toward the close; "sharp
  money" (handle % minus ticket %) leans toward the eventual correction.
* Soft books shade popular teams, add their own noise and charge 4-5% vig, so
  line shopping across them matters.

Usage:
    python generate_mock_data.py                 # 3 past seasons + today's slate
    python generate_mock_data.py --seasons 2 --today 2026-10-25
"""

from __future__ import annotations

import argparse
import math
from datetime import date

import numpy as np
import pandas as pd

from config import settings
from data.storage import Storage

TEAMS = [
    "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets", "Chicago Bulls",
    "Cleveland Cavaliers", "Dallas Mavericks", "Denver Nuggets", "Detroit Pistons", "Golden State Warriors",
    "Houston Rockets", "Indiana Pacers", "Los Angeles Clippers", "Los Angeles Lakers", "Memphis Grizzlies",
    "Miami Heat", "Milwaukee Bucks", "Minnesota Timberwolves", "New Orleans Pelicans", "New York Knicks",
    "Oklahoma City Thunder", "Orlando Magic", "Philadelphia 76ers", "Phoenix Suns", "Portland Trail Blazers",
    "Sacramento Kings", "San Antonio Spurs", "Toronto Raptors", "Utah Jazz", "Washington Wizards",
]
SHARP_BOOK = "pinnacle"
SHARP_VIG = 0.025
SOFT_BOOKS = {"draftkings": 0.045, "fanduel": 0.042, "betmgm": 0.050, "caesars": 0.048}

MARGIN_SD = 12.0
HOME_ADV = 2.5
B2B_PENALTY = 2.5
B2B_PRICED = {"open": 0.30, "decision": 0.60, "close": 0.85}
SNAPSHOT_OFFSETS = {"open": pd.Timedelta(hours=24), "decision": pd.Timedelta(hours=4), "close": pd.Timedelta(minutes=5)}


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _price(p: float, vig: float) -> float:
    return round(max(1.01, 1.0 / (p * (1.0 + vig))), 2)


def _season_windows(today: pd.Timestamp, n_completed: int) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    """Completed Oct-Apr seasons, then the current season up to `today`."""
    current_start = today - pd.Timedelta(days=4)
    last_year = current_start.year - 1 if current_start >= pd.Timestamp(current_start.year, 4, 13) else current_start.year - 2
    windows = [
        (f"{y}-{str(y + 1)[2:]}", pd.Timestamp(y, 10, 22), pd.Timestamp(y + 1, 4, 12))
        for y in range(last_year - n_completed + 1, last_year + 1)
    ]
    windows.append((f"{current_start.year}-cur", current_start, today))
    return windows


def generate(store: Storage, seasons: int = 3, today: pd.Timestamp | None = None, seed: int = 42) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    today = pd.Timestamp(today or date.today()).normalize()
    strength = dict(zip(TEAMS, rng.normal(0, 4.5, len(TEAMS))))
    popularity = dict(zip(TEAMS, rng.normal(0, 1, len(TEAMS))))
    last_played: dict[str, pd.Timestamp] = {}

    games, odds, sentiment, team_stats = [], [], [], []

    for season, start, end in _season_windows(today, seasons):
        for t in TEAMS:  # off-season: regress and reshuffle a bit
            strength[t] = 0.7 * strength[t] + rng.normal(0, 2.0)
        last_played.clear()

        for day in pd.date_range(start, end, freq="D"):
            for t in TEAMS:
                strength[t] += rng.normal(0, 0.08)
            n_games = int(rng.integers(9, 13)) if day == today else int(rng.integers(4, 13))
            # Rested teams are ~5x likelier to be scheduled, giving an NBA-like b2b rate.
            yesterday = day - pd.Timedelta(days=1)
            weights = np.array([0.2 if last_played.get(t) == yesterday else 1.0 for t in TEAMS])
            order = list(rng.choice(TEAMS, size=2 * n_games, replace=False, p=weights / weights.sum()))
            for idx in range(n_games):
                home, away = order[2 * idx], order[2 * idx + 1]
                gid = f"MOCK-{day:%Y%m%d}-{idx:02d}"
                commence = (day + pd.Timedelta(hours=float(rng.choice([23.0, 23.5, 24.0, 25.0, 26.0])))).tz_localize("UTC")
                b2b_h = int(last_played.get(home) == day - pd.Timedelta(days=1))
                b2b_a = int(last_played.get(away) == day - pd.Timedelta(days=1))
                last_played[home] = last_played[away] = day

                base_mu = strength[home] - strength[away] + HOME_ADV
                fatigue = B2B_PENALTY * (b2b_a - b2b_h)
                true_mu = base_mu + fatigue

                # Sharp market: error shrinks toward the close, fatigue priced in gradually.
                e_open = rng.normal(0, 1.6)
                e_dec = 0.6 * e_open + rng.normal(0, 0.8)
                e_close = 0.35 * e_dec + rng.normal(0, 0.5)
                errs = {"open": e_open, "decision": e_dec, "close": e_close}
                p_mkt = {k: _phi((base_mu + B2B_PRICED[k] * fatigue + errs[k]) / MARGIN_SD) for k in errs}

                final = day < today
                snaps = ["open", "decision", "close"] if final else ["open", "decision"]
                for snap in snaps:
                    ts = (commence - SNAPSHOT_OFFSETS[snap]).isoformat()
                    p = p_mkt[snap]
                    odds += [
                        (gid, SHARP_BOOK, "h2h", "home", _price(p, SHARP_VIG), ts),
                        (gid, SHARP_BOOK, "h2h", "away", _price(1 - p, SHARP_VIG), ts),
                    ]
                    for book, vig in SOFT_BOOKS.items():
                        ps = float(np.clip(p + 0.008 * (popularity[home] - popularity[away]) + rng.normal(0, 0.010), 0.03, 0.97))
                        odds += [
                            (gid, book, "h2h", "home", _price(ps, vig), ts),
                            (gid, book, "h2h", "away", _price(1 - ps, vig), ts),
                        ]

                # Public tickets follow popularity and favourites; handle leans to the correction.
                fav = math.log(p_mkt["decision"] / (1 - p_mkt["decision"]))
                ticket = 1 / (1 + math.exp(-(0.6 * (popularity[home] - popularity[away]) + 1.2 * fav))) + rng.normal(0, 0.04)
                ticket = float(np.clip(ticket, 0.05, 0.95))
                sharp_signal = _phi((true_mu + 0.5 * e_close) / MARGIN_SD) - p_mkt["decision"]
                money = float(np.clip(ticket + 0.8 * sharp_signal + rng.normal(0, 0.06), 0.03, 0.97))
                sentiment.append((gid, (commence - SNAPSHOT_OFFSETS["decision"]).isoformat(), ticket, money))

                row = {
                    "game_id": gid, "sport": settings.sport_key, "season": season,
                    "commence_time": commence.isoformat(), "game_date": day.date().isoformat(),
                    "home_team": home, "away_team": away, "home_score": None, "away_score": None, "status": "scheduled",
                }
                if final:
                    margin = true_mu + rng.normal(0, MARGIN_SD)
                    margin = round(margin) or (1 if margin >= 0 else -1)
                    total = rng.normal(224, 16)
                    hs = int(round((total + margin) / 2))
                    row.update(home_score=hs, away_score=hs - margin, status="final")
                    pace = rng.normal(99, 3)
                    team_stats += [
                        (gid, home, row["home_score"] / pace * 100, row["away_score"] / pace * 100, pace),
                        (gid, away, row["away_score"] / pace * 100, row["home_score"] / pace * 100, pace),
                    ]
                games.append(row)

    store.reset()
    counts = {
        "games": store.upsert("games", pd.DataFrame(games)),
        "odds": store.upsert("odds", pd.DataFrame(odds, columns=["game_id", "bookmaker", "market", "selection", "price", "snapshot_time"])),
        "sentiment": store.upsert("sentiment", pd.DataFrame(sentiment, columns=["game_id", "snapshot_time", "home_ticket_pct", "home_money_pct"])),
        "team_stats": store.upsert("team_stats", pd.DataFrame(team_stats, columns=["game_id", "team", "off_rating", "def_rating", "pace"])),
    }
    n_today = sum(1 for g in games if g["status"] == "scheduled")
    print(f"Synthetic data written to {store.db_path}")
    for table, n in counts.items():
        print(f"  {table:<11} {n:>9,} rows")
    print(f"  today's slate ({today.date()}): {n_today} scheduled games")
    return counts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seasons", type=int, default=3)
    p.add_argument("--today", default=date.today().isoformat())
    p.add_argument("--seed", type=int, default=settings.random_seed)
    p.add_argument("--db", default=None, help="SQLite path (default from config)")
    args = p.parse_args()
    generate(Storage(args.db), seasons=args.seasons, today=pd.Timestamp(args.today), seed=args.seed)


if __name__ == "__main__":
    main()
