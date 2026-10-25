"""Market sentiment: public betting splits.

There is no free, official API for ticket/handle splits; most people use a paid
feed (Action Network, Sports Insights, OddsJam). `JSONSplitsConnector` reads any
endpoint returning a list of {game_id, home_ticket_pct, home_money_pct} objects,
so wiring a provider in is a matter of setting its URL and key.

Line movement (sharp vs. public) does not need this feed: it is derived from the
odds snapshots in `models.features` (open -> decision-time move at the sharp
book, and the gap between sharp and soft book prices).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from config import settings
from data.connectors.base import ConnectorError, HttpConnector, SentimentConnector


class JSONSplitsConnector(SentimentConnector):
    def __init__(self, url: str, api_key: str | None = None) -> None:
        if not url:
            raise ConnectorError("Sentiment feed URL is not set")
        self.http = HttpConnector(url)
        self.api_key = api_key or settings.sentiment_api_key

    def fetch_splits(self) -> pd.DataFrame:
        rows = self.http.get_json("", {"apiKey": self.api_key} if self.api_key else None)
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df["snapshot_time"] = datetime.now(timezone.utc).isoformat()
        for col in ("home_ticket_pct", "home_money_pct"):
            # Accept both 0-1 and 0-100 conventions.
            df[col] = df[col].astype(float).where(df[col].astype(float) <= 1.0, df[col].astype(float) / 100.0)
        return df[["game_id", "snapshot_time", "home_ticket_pct", "home_money_pct"]]
