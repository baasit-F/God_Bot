"""ETL jobs that move connector output into local storage.

    ingest_live_odds()      today's slate + current prices (run several times a day
                            to build a line-movement history)
    ingest_results(day)     final scores for a day, joined onto stored games
    backfill_odds(...)      historical odds snapshots for offline backtesting
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pandas as pd

from data.connectors.base import OddsConnector, StatsConnector
from data.storage import Storage


def ingest_live_odds(connector: OddsConnector, store: Storage) -> tuple[int, int]:
    games, odds = connector.fetch_current()
    # Never overwrite final scores of games we already know about.
    known = store.read("games")
    if not known.empty and not games.empty:
        games = games[~games["game_id"].isin(known.loc[known["status"] == "final", "game_id"])]
    return store.upsert("games", games), store.upsert("odds", odds)


def ingest_results(connector: StatsConnector, store: Storage, day: date) -> int:
    results, team_stats = connector.fetch_results(day)
    if results.empty:
        return 0
    games = store.read("games", "game_date = ?", (day.isoformat(),))
    merged = games.drop(columns=["home_score", "away_score"]).merge(
        results, on=["game_date", "home_team", "away_team"], how="inner"
    )
    merged["status"] = "final"
    store.upsert("games", merged[games.columns])
    if not team_stats.empty:
        store.upsert("team_stats", team_stats)
    return len(merged)


def backfill_odds(
    connector: OddsConnector,
    store: Storage,
    start: datetime,
    end: datetime,
    step: timedelta = timedelta(hours=6),
) -> int:
    """Pull historical snapshots every `step` between start and end (UTC)."""
    written = 0
    t = start.astimezone(timezone.utc)
    while t <= end:
        games, odds = connector.fetch_historical(t)
        store.upsert("games", games)
        written += store.upsert("odds", odds)
        t += step
    return written


def backfill_results(connector: StatsConnector, store: Storage, start: date, end: date) -> int:
    total = 0
    for day in pd.date_range(start, end, freq="D"):
        total += ingest_results(connector, store, day.date())
    return total
