"""SQLite storage layer (with Parquet export) for offline backtesting.

Tables
------
games       one row per fixture; scores are NULL until the game is final
odds        every price snapshot per (game, book, selection, snapshot_time)
sentiment   public betting splits per snapshot (ticket % and money %)
team_stats  per-team, per-game box score metrics from stats connectors
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd

from config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    game_id        TEXT PRIMARY KEY,
    sport          TEXT NOT NULL,
    season         TEXT,
    commence_time  TEXT NOT NULL,          -- ISO-8601 UTC
    game_date      TEXT NOT NULL,          -- YYYY-MM-DD
    home_team      TEXT NOT NULL,
    away_team      TEXT NOT NULL,
    home_score     INTEGER,
    away_score     INTEGER,
    status         TEXT NOT NULL DEFAULT 'scheduled'   -- scheduled | final
);
CREATE INDEX IF NOT EXISTS ix_games_date ON games(game_date);

CREATE TABLE IF NOT EXISTS odds (
    game_id        TEXT NOT NULL,
    bookmaker      TEXT NOT NULL,
    market         TEXT NOT NULL DEFAULT 'h2h',
    selection      TEXT NOT NULL,          -- home | away | draw
    price          REAL NOT NULL,          -- decimal odds
    snapshot_time  TEXT NOT NULL,          -- ISO-8601 UTC
    PRIMARY KEY (game_id, bookmaker, market, selection, snapshot_time)
);
CREATE INDEX IF NOT EXISTS ix_odds_game ON odds(game_id);

CREATE TABLE IF NOT EXISTS sentiment (
    game_id        TEXT NOT NULL,
    snapshot_time  TEXT NOT NULL,
    home_ticket_pct REAL,                  -- share of bets on home (0-1)
    home_money_pct  REAL,                  -- share of handle on home (0-1)
    PRIMARY KEY (game_id, snapshot_time)
);

CREATE TABLE IF NOT EXISTS team_stats (
    game_id        TEXT NOT NULL,
    team           TEXT NOT NULL,
    off_rating     REAL,
    def_rating     REAL,
    pace           REAL,
    PRIMARY KEY (game_id, team)
);
"""

TABLES = ("games", "odds", "sentiment", "team_stats")


class Storage:
    """Thin wrapper around a SQLite file with DataFrame helpers."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path or settings.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------ write
    def upsert(self, table: str, df: pd.DataFrame) -> int:
        """INSERT OR REPLACE a DataFrame into `table`. Returns rows written."""
        if table not in TABLES:
            raise ValueError(f"Unknown table {table!r}")
        if df.empty:
            return 0
        cols = list(df.columns)
        placeholders = ",".join("?" for _ in cols)
        sql = f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) VALUES ({placeholders})"
        rows = [tuple(None if pd.isna(v) else v for v in row) for row in df.itertuples(index=False)]
        with self.connect() as conn:
            conn.executemany(sql, rows)
        return len(rows)

    def reset(self) -> None:
        with self.connect() as conn:
            for table in TABLES:
                conn.execute(f"DELETE FROM {table}")

    # ------------------------------------------------------------------- read
    def read(self, table: str, where: str = "", params: tuple = ()) -> pd.DataFrame:
        if table not in TABLES:
            raise ValueError(f"Unknown table {table!r}")
        sql = f"SELECT * FROM {table}" + (f" WHERE {where}" if where else "")
        with self.connect() as conn:
            return pd.read_sql_query(sql, conn, params=params)

    def games(self) -> pd.DataFrame:
        df = self.read("games")
        df["commence_time"] = pd.to_datetime(df["commence_time"], utc=True)
        df["game_date"] = pd.to_datetime(df["game_date"])
        return df.sort_values("commence_time").reset_index(drop=True)

    def odds(self) -> pd.DataFrame:
        df = self.read("odds")
        df["snapshot_time"] = pd.to_datetime(df["snapshot_time"], utc=True)
        return df

    def sentiment(self) -> pd.DataFrame:
        df = self.read("sentiment")
        df["snapshot_time"] = pd.to_datetime(df["snapshot_time"], utc=True)
        return df

    def team_stats(self) -> pd.DataFrame:
        return self.read("team_stats")

    # ---------------------------------------------------------------- parquet
    def export_parquet(self, out_dir: Path | str | None = None) -> list[Path]:
        """Dump every table to Parquet for fast columnar offline analysis."""
        out = Path(out_dir or settings.parquet_dir)
        out.mkdir(parents=True, exist_ok=True)
        written = []
        for table in TABLES:
            path = out / f"{table}.parquet"
            self.read(table).to_parquet(path, index=False)
            written.append(path)
        return written
