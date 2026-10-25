"""Daily prediction engine: train on everything final before today, score today's slate."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config import settings
from data.storage import Storage
from models.features import build_feature_frame
from models.model import WinProbModel
from models.scoring import build_opportunities, select_picks, size_picks


@dataclass
class DailyPicks:
    day: pd.Timestamp
    picks: pd.DataFrame
    opportunities: pd.DataFrame
    trained_on: int


def run_daily(store: Storage, day: pd.Timestamp, bankroll: float | None = None, model_type: str | None = None) -> DailyPicks:
    bankroll = settings.starting_bankroll if bankroll is None else bankroll
    games = store.games()
    games = games[games["game_date"] <= day]
    # decision_hours_before=0: use the latest pre-game snapshot available now.
    feats = build_feature_frame(games, store.odds(), store.sentiment(), store.team_stats(), decision_hours_before=0)

    history = feats[(feats["game_date"] < day) & feats["home_win"].notna()]
    slate = feats[(feats["game_date"] == day) & (feats["status"] != "final")]
    if len(history) < settings.min_train_games:
        raise RuntimeError(f"Only {len(history)} finished games before {day.date()}; need {settings.min_train_games}")
    if slate.empty:
        return DailyPicks(day, pd.DataFrame(), pd.DataFrame(), len(history))

    model = WinProbModel(model_type).fit(history)
    opp = build_opportunities(slate, model.predict_home(slate))
    picks = size_picks(select_picks(opp), bankroll)
    return DailyPicks(day, picks, opp, len(history))
