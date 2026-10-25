"""Slate prediction engine: train on everything final before the slate, score the slate.

NFL slates span Thursday-Monday, so `days` widens the window (e.g. Sunday + Monday)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config import settings
from data.storage import Storage
from models.engine import MarketModels
from models.features import build_feature_frame, build_quotes
from models.scoring import select_picks, size_picks


@dataclass
class DailyPicks:
    day: pd.Timestamp
    picks: pd.DataFrame
    opportunities: pd.DataFrame
    trained_on: int


def run_daily(
    store: Storage, day: pd.Timestamp, bankroll: float | None = None, model_type: str | None = None, days: int = 1
) -> DailyPicks:
    bankroll = settings.starting_bankroll if bankroll is None else bankroll
    games = store.games()
    last_day = day + pd.Timedelta(days=days - 1)
    games = games[games["game_date"] <= last_day]
    # decision_hours_before=0: use the latest pre-game snapshot available now.
    odds = store.odds()
    feats = build_feature_frame(games, odds, store.sentiment(), store.team_stats(), decision_hours_before=0)
    quotes = build_quotes(games, odds, decision_hours_before=0)

    history = feats[(feats["game_date"] < day) & (feats["status"] == "final")]
    slate = feats[feats["game_date"].between(day, last_day) & (feats["status"] != "final")]
    if len(history) < settings.min_train_games:
        raise RuntimeError(f"Only {len(history)} finished games before {day.date()}; need {settings.min_train_games}")
    if slate.empty:
        return DailyPicks(day, pd.DataFrame(), pd.DataFrame(), len(history))

    models = MarketModels(model_type=model_type).fit(history)
    opp = models.opportunities(slate, quotes)
    picks = size_picks(select_picks(opp), bankroll)
    return DailyPicks(day, picks, opp, len(history))
