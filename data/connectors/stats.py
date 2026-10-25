"""Statistics connectors.

`ESPNStatsConnector` uses ESPN's public scoreboard JSON (no key required).
`SportsDataIOConnector` is the same contract for a keyed provider. Both return
results that the ETL joins onto games by (game_date, home_team, away_team).
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from config import settings
from data.connectors.base import ConnectorError, HttpConnector, StatsConnector

TEAM_STATS_COLUMNS = ["game_id", "team", "off_ypp", "def_ypp", "plays"]

# Map provider-specific team names/abbreviations to The-Odds-API full names.
# ESPN's NFL display names already match; SportsDataIO uses abbreviations.
TEAM_ALIASES: dict[str, str] = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens", "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers", "CHI": "Chicago Bears", "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys", "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars", "KC": "Kansas City Chiefs",
    "LV": "Las Vegas Raiders", "LAC": "Los Angeles Chargers", "LAR": "Los Angeles Rams", "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings", "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers", "SF": "San Francisco 49ers",
    "SEA": "Seattle Seahawks", "TB": "Tampa Bay Buccaneers", "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}

ESPN_SPORT_PATHS: dict[str, str] = {
    "basketball_nba": "basketball/nba",
    "americanfootball_nfl": "football/nfl",
    "baseball_mlb": "baseball/mlb",
    "icehockey_nhl": "hockey/nhl",
}


def canonical_team(name: str) -> str:
    return TEAM_ALIASES.get(name, name)


class ESPNStatsConnector(StatsConnector):
    def __init__(self, sport_key: str | None = None) -> None:
        sport_key = sport_key or settings.sport_key
        if sport_key not in ESPN_SPORT_PATHS:
            raise ConnectorError(f"No ESPN mapping for sport {sport_key!r}")
        self.http = HttpConnector(f"https://site.api.espn.com/apis/site/v2/sports/{ESPN_SPORT_PATHS[sport_key]}")

    def fetch_results(self, day: date) -> tuple[pd.DataFrame, pd.DataFrame]:
        payload = self.http.get_json("scoreboard", {"dates": day.strftime("%Y%m%d")})
        return self.normalize(payload, day)

    @staticmethod
    def normalize(payload: dict[str, Any], day: date) -> tuple[pd.DataFrame, pd.DataFrame]:
        results = []
        for ev in payload.get("events", []):
            comp = ev["competitions"][0]
            if not comp.get("status", ev.get("status", {})).get("type", {}).get("completed", False):
                continue
            sides = {c["homeAway"]: c for c in comp["competitors"]}
            results.append(
                {
                    "game_date": day.isoformat(),
                    "home_team": canonical_team(sides["home"]["team"]["displayName"]),
                    "away_team": canonical_team(sides["away"]["team"]["displayName"]),
                    "home_score": int(sides["home"]["score"]),
                    "away_score": int(sides["away"]["score"]),
                }
            )
        # The scoreboard has no advanced box-score metrics; team_stats stays empty
        # here and is filled by richer providers (SportsDataIO / nflverse play-by-play).
        return pd.DataFrame(results), pd.DataFrame(columns=TEAM_STATS_COLUMNS)


class SportsDataIOConnector(StatsConnector):
    """SportsDataIO NFL scores + team game stats (requires SPORTSDATAIO_API_KEY)."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or settings.sportsdataio_api_key
        if not self.api_key:
            raise ConnectorError("SPORTSDATAIO_API_KEY is not set")
        self.http = HttpConnector("https://api.sportsdata.io/v3/nfl/scores/json")

    def fetch_results(self, day: date) -> tuple[pd.DataFrame, pd.DataFrame]:
        stamp = day.strftime("%Y-%b-%d").upper()
        games = self.http.get_json(f"ScoresByDate/{stamp}", {"key": self.api_key})
        results = [
            {
                "game_date": day.isoformat(),
                "home_team": canonical_team(g["HomeTeam"]),
                "away_team": canonical_team(g["AwayTeam"]),
                "home_score": g["HomeScore"],
                "away_score": g["AwayScore"],
            }
            for g in games
            if g.get("Status") in ("Final", "F/OT")
        ]
        # Yards-per-play needs the TeamGameStatsByDate endpoint; map OffensiveYardsPerPlay
        # and the opponent's value into TEAM_STATS_COLUMNS once game_ids are joined.
        return pd.DataFrame(results), pd.DataFrame(columns=TEAM_STATS_COLUMNS)
