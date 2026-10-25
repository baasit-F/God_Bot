"""Equity curve, drawdown and rolling-yield charts (Matplotlib, saved as PNG)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from backtest.metrics import by_period
from backtest.walkforward import BacktestResult


def plot_report(result: BacktestResult, out_dir: Path, rolling_bets: int = 100) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    daily, bets = result.daily, result.bets
    equity = daily.set_index("date")["bankroll_end"]
    drawdown = (equity / equity.cummax() - 1.0) * 100

    fig, axes = plt.subplots(4, 1, figsize=(12, 13), gridspec_kw={"height_ratios": [3, 1.3, 1.6, 1.6]})
    fig.suptitle("ValuePicks Bot - Walk-Forward Backtest", fontsize=15, fontweight="bold")

    ax = axes[0]
    ax.plot(equity.index, equity.values, color="#1f6feb", lw=1.8)
    ax.axhline(result.starting_bankroll, color="#888", lw=0.8, ls="--")
    ax.set_ylabel("Bankroll")
    ax.set_title("Equity curve")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.fill_between(drawdown.index, drawdown.values, 0, color="#d1242f", alpha=0.35)
    ax.set_ylabel("Drawdown %")
    ax.set_title("Drawdown")
    ax.grid(alpha=0.3)

    ax = axes[2]
    if not bets.empty:
        b = bets.sort_values("commence_time")
        roll = b["profit"].rolling(rolling_bets).sum() / b["stake"].rolling(rolling_bets).sum() * 100
        ax.plot(pd.to_datetime(b["game_date"]), roll, color="#8250df", lw=1.2)
        ax.axhline(0, color="#888", lw=0.8)
    ax.set_ylabel("Yield %")
    ax.set_title(f"Rolling yield ({rolling_bets} bets)")
    ax.grid(alpha=0.3)

    ax = axes[3]
    monthly = by_period(bets, "M")
    if not monthly.empty:
        x = monthly.index.to_timestamp()
        colors = ["#1a7f37" if v >= 0 else "#d1242f" for v in monthly["yield"]]
        ax.bar(x, monthly["yield"] * 100, width=20, color=colors)
        ax.axhline(0, color="#888", lw=0.8)
    ax.set_ylabel("Yield %")
    ax.set_title("Monthly yield")
    ax.grid(alpha=0.3, axis="y")

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.tight_layout()
    path = out_dir / "backtest_report.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
