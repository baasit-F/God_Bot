"""Connector interfaces.

Each connector turns a third-party API response into DataFrames that match the
tables in `data.storage`. Swapping providers (e.g. The-Odds-API -> OpticOdds,
ESPN -> SportsDataIO) means writing one new class; nothing downstream changes.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Any

import pandas as pd
import requests


class ConnectorError(RuntimeError):
    pass


class HttpConnector:
    """Shared HTTP plumbing: session reuse, retries with backoff, timeouts."""

    def __init__(self, base_url: str, timeout: float = 20.0, max_retries: int = 3) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise ConnectorError(f"HTTP {resp.status_code} from {url}")
                resp.raise_for_status()
                return resp.json()
            except (requests.RequestException, ConnectorError) as exc:
                last_error = exc
                time.sleep(2**attempt)
        raise ConnectorError(f"GET {url} failed after {self.max_retries} attempts: {last_error}")


class OddsConnector(ABC):
    """Returns (games, odds) DataFrames in storage schema."""

    @abstractmethod
    def fetch_current(self) -> tuple[pd.DataFrame, pd.DataFrame]: ...

    @abstractmethod
    def fetch_historical(self, snapshot: datetime) -> tuple[pd.DataFrame, pd.DataFrame]: ...


class StatsConnector(ABC):
    """Returns (results, team_stats) DataFrames for games on a given day."""

    @abstractmethod
    def fetch_results(self, day: date) -> tuple[pd.DataFrame, pd.DataFrame]: ...


class SentimentConnector(ABC):
    """Returns a sentiment DataFrame (public ticket % / money % splits)."""

    @abstractmethod
    def fetch_splits(self) -> pd.DataFrame: ...
