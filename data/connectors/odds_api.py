"""The-Odds-API v4 connector (https://the-odds-api.com/liveapi/guides/v4/).

Also works for OpticOdds-style payloads after a small mapping, because both
expose events -> bookmakers -> markets -> outcomes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from config import settings
from data.connectors.base import ConnectorError, HttpConnector, OddsConnector


class TheOddsAPIConnector(OddsConnector):
    def __init__(
        self,
        api_key: str | None = None,
        sport_key: str | None = None,
        regions: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.api_key = api_key or settings.odds_api_key
        if not self.api_key:
            raise ConnectorError("ODDS_API_KEY is not set (see .env.example)")
        self.sport_key = sport_key or settings.sport_key
        self.regions = regions or settings.regions
        self.http = HttpConnector(base_url or settings.odds_api_base_url)

    def _params(self, **extra: Any) -> dict[str, Any]:
        return {
            "apiKey": self.api_key,
            "regions": self.regions,
            "markets": "h2h",
            "oddsFormat": "decimal",
            "dateFormat": "iso",
            **extra,
        }

    def fetch_current(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        events = self.http.get_json(f"sports/{self.sport_key}/odds", self._params())
        return self.normalize(events, snapshot=datetime.now(timezone.utc))

    def fetch_historical(self, snapshot: datetime) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Historical snapshot (paid plans). Call repeatedly to build line history."""
        payload = self.http.get_json(
            f"historical/sports/{self.sport_key}/odds",
            self._params(date=snapshot.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")),
        )
        ts = pd.Timestamp(payload.get("timestamp", snapshot)).to_pydatetime()
        return self.normalize(payload.get("data", []), snapshot=ts)

    def fetch_scores(self, days_from: int = 3) -> pd.DataFrame:
        """Final scores for recently completed games (keyed by Odds-API event id)."""
        events = self.http.get_json(
            f"sports/{self.sport_key}/scores", {"apiKey": self.api_key, "daysFrom": days_from, "dateFormat": "iso"}
        )
        rows = []
        for ev in events:
            if not ev.get("completed") or not ev.get("scores"):
                continue
            scores = {s["name"]: int(s["score"]) for s in ev["scores"]}
            rows.append(
                {
                    "game_id": ev["id"],
                    "home_score": scores.get(ev["home_team"]),
                    "away_score": scores.get(ev["away_team"]),
                    "status": "final",
                }
            )
        return pd.DataFrame(rows)

    def normalize(self, events: list[dict[str, Any]], snapshot: datetime) -> tuple[pd.DataFrame, pd.DataFrame]:
        games, odds = [], []
        snap = pd.Timestamp(snapshot)
        snap_iso = (snap.tz_localize("UTC") if snap.tzinfo is None else snap.tz_convert("UTC")).isoformat()
        for ev in events:
            commence = pd.Timestamp(ev["commence_time"])
            games.append(
                {
                    "game_id": ev["id"],
                    "sport": ev.get("sport_key", self.sport_key),
                    "season": str(commence.year if commence.month >= 8 else commence.year - 1),
                    "commence_time": commence.isoformat(),
                    "game_date": commence.tz_convert("America/New_York").date().isoformat(),
                    "home_team": ev["home_team"],
                    "away_team": ev["away_team"],
                    "status": "scheduled",
                }
            )
            for book in ev.get("bookmakers", []):
                for market in book.get("markets", []):
                    if market.get("key") != "h2h":
                        continue
                    for outcome in market.get("outcomes", []):
                        name = outcome["name"]
                        selection = "home" if name == ev["home_team"] else "away" if name == ev["away_team"] else "draw"
                        odds.append(
                            {
                                "game_id": ev["id"],
                                "bookmaker": book["key"],
                                "market": "h2h",
                                "selection": selection,
                                "price": float(outcome["price"]),
                                "snapshot_time": snap_iso,
                            }
                        )
        return pd.DataFrame(games), pd.DataFrame(odds)
