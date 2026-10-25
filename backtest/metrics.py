"""Performance metrics for a backtest.

ROI    profit / starting bankroll (bankroll growth)
Yield  profit / total amount staked (return per unit wagered)
Sharpe annualised mean / std of daily bankroll returns on betting days
MaxDD  largest peak-to-trough fall of the bankroll, in percent
CLV    average edge of our price over the de-vigged sharp closing line
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.walkforward import BacktestResult


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    return float(((equity - peak) / peak).min())


def sharpe_ratio(daily: pd.DataFrame) -> float:
    active = daily[daily["n_bets"] > 0]
    if len(active) < 2:
        return 0.0
    rets = active["pnl"] / active["bankroll_start"]
    if rets.std(ddof=1) == 0:
        return 0.0
    span_years = max((daily["date"].max() - daily["date"].min()).days / 365.25, 1 / 365.25)
    per_year = len(active) / span_years
    return float(rets.mean() / rets.std(ddof=1) * np.sqrt(per_year))


def summarize(result: BacktestResult) -> dict[str, float]:
    bets, daily = result.bets, result.daily
    start = result.starting_bankroll
    if bets.empty:
        return {"bets": 0}
    profit = float(bets["profit"].sum())
    staked = float(bets["stake"].sum())
    equity = pd.concat([pd.Series([start]), daily["bankroll_end"]], ignore_index=True)
    return {
        "bets": int(len(bets)),
        "betting_days": int((daily["n_bets"] > 0).sum()),
        "win_rate": _win_rate(bets["won"]),
        "pushes": int((bets["won"] == 0.5).sum()),
        "avg_odds": float(bets["odds"].mean()),
        "total_staked": staked,
        "profit": profit,
        "final_bankroll": start + profit,
        "roi": profit / start,
        "yield": profit / staked if staked else 0.0,
        "sharpe": sharpe_ratio(daily),
        "max_drawdown": max_drawdown(equity),
        "avg_ev": float(bets["ev"].mean()),
        "avg_vos": float(bets["vos"].mean()),
        "avg_clv": float(bets["clv"].mean()) if "clv" in bets and bets["clv"].notna().any() else float("nan"),
        "avg_stake_pct": float(bets["stake_pct"].mean()),
    }


def by_period(bets: pd.DataFrame, freq: str = "M") -> pd.DataFrame:
    """Bets, win rate, profit and yield per calendar period."""
    if bets.empty:
        return pd.DataFrame()
    period = pd.to_datetime(bets["game_date"]).dt.to_period(freq)
    grp = bets.groupby(period)
    out = pd.DataFrame(
        {"bets": grp.size(), "win_rate": grp["won"].apply(_win_rate), "staked": grp["stake"].sum(), "profit": grp["profit"].sum()}
    )
    out["yield"] = out["profit"] / out["staked"]
    return out


def by_tier(bets: pd.DataFrame, column: str = "tier") -> pd.DataFrame:
    """Bets, win rate, profit and yield per tier (or any other column, e.g. market)."""
    if bets.empty:
        return pd.DataFrame()
    grp = bets.groupby(column)
    out = pd.DataFrame({"bets": grp.size(), "win_rate": grp["won"].apply(_win_rate), "staked": grp["stake"].sum(),
                        "profit": grp["profit"].sum(), "avg_clv": grp["clv"].mean()})
    out["yield"] = out["profit"] / out["staked"]
    return out


def _win_rate(won: pd.Series) -> float:
    """Wins / decided bets (pushes excluded)."""
    decided = won[won != 0.5]
    return float((decided == 1.0).mean()) if len(decided) else float("nan")
