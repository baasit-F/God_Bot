"""Walk-forward backtester.

For every betting day D (chronologically):
  1. (Re)train the moneyline, margin and total models on final games with game_date < D, every
     `retrain_every_days` days. Training never sees day D or later.
  2. Predict day D's games from features that only use pre-D results and
     decision-time odds snapshots (see models.features).
  3. Filter to Top Line Picks, size with fractional Kelly on the bankroll at the
     start of day D, then settle at the end of the day.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import settings
from models.engine import MarketModels
from models.model import evaluate_probs
from models.scoring import select_picks, size_picks


@dataclass
class BacktestResult:
    bets: pd.DataFrame
    daily: pd.DataFrame
    predictions: pd.DataFrame
    retrain_dates: list[pd.Timestamp] = field(default_factory=list)
    starting_bankroll: float = 0.0


def run_walk_forward(
    features: pd.DataFrame,
    quotes: pd.DataFrame | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    bankroll: float | None = None,
    model_type: str | None = None,
    verbose: bool = True,
) -> BacktestResult:
    bankroll0 = settings.starting_bankroll if bankroll is None else bankroll
    df = features[features["status"] == "final"].copy()
    quotes = quotes if quotes is not None else pd.DataFrame(columns=["game_id", "market"])
    dates = sorted(df["game_date"].unique())

    model: MarketModels | None = None
    last_train: pd.Timestamp | None = None
    retrains: list[pd.Timestamp] = []
    bankroll_now = bankroll0
    all_bets, daily_rows, preds = [], [], []

    for day in dates:
        day = pd.Timestamp(day)
        if (start is not None and day < start) or (end is not None and day > end):
            continue
        history = df[df["game_date"] < day]
        if len(history) < settings.min_train_games:
            continue
        if model is None or (day - last_train).days >= settings.retrain_every_days:
            model = MarketModels(model_type=model_type).fit(history)
            last_train = day
            retrains.append(day)
            if verbose:
                print(f"  [{day.date()}] retrained on {len(history):,} games (bankroll {bankroll_now:,.0f})")

        today = df[df["game_date"] == day]
        preds.append(model.predictions(today))
        picks = size_picks(select_picks(model.opportunities(today, quotes)), bankroll_now)
        pnl = 0.0
        if not picks.empty:
            picks["bankroll_before"] = bankroll_now
            # won: 1 = win, 0 = loss, 0.5 = push (moneyline tie or landing on a whole-number line).
            picks["profit"] = np.select(
                [picks["won"] == 1.0, picks["won"] == 0.5], [picks["stake"] * (picks["odds"] - 1.0), 0.0], -picks["stake"]
            )
            # CLV: how much better our price was than the de-vigged closing line.
            picks["clv"] = picks["odds"] * picks["p_close"] - 1.0
            pnl = float(picks["profit"].sum())
            all_bets.append(picks)

        daily_rows.append({"date": day, "n_bets": len(picks), "staked": float(picks["stake"].sum()) if len(picks) else 0.0,
                           "pnl": pnl, "bankroll_start": bankroll_now, "bankroll_end": bankroll_now + pnl})
        bankroll_now += pnl

    bets = pd.concat(all_bets, ignore_index=True) if all_bets else pd.DataFrame()
    return BacktestResult(
        bets=bets,
        daily=pd.DataFrame(daily_rows),
        predictions=pd.concat(preds, ignore_index=True) if preds else pd.DataFrame(),
        retrain_dates=retrains,
        starting_bankroll=bankroll0,
    )


def model_vs_market(predictions: pd.DataFrame) -> list[dict[str, object]]:
    """Out-of-sample accuracy of each model against the sharp market (lower is better)."""
    rows: list[dict[str, object]] = []
    decided = predictions[predictions["home_win"].isin([0.0, 1.0]) & predictions["p_model_home"].notna()]
    if not decided.empty:
        y = decided["home_win"].to_numpy()
        m, k = evaluate_probs(y, decided["p_model_home"].to_numpy()), evaluate_probs(y, decided["p_market_home"].to_numpy())
        rows.append({"market": "Moneyline", "metric": "Brier", "model": m["brier"], "sharp": k["brier"], "games": len(decided)})
    for prefix, actual, label in (("spread", "actual_margin", "Spread"), ("total", "actual_total", "Total")):
        col = f"{prefix}_mu_model"
        if col not in predictions:
            continue
        d = predictions[predictions[col].notna() & predictions[actual].notna()]
        if d.empty:
            continue
        rmse = lambda pred: float(np.sqrt(np.mean((d[actual] - pred) ** 2)))  # noqa: E731
        rows.append({"market": label, "metric": "RMSE (pts)", "model": rmse(d[col]), "sharp": rmse(d[f"{prefix}_mu"]), "games": len(d)})
    return rows
