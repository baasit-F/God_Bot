"""Probability maths for line markets (spreads and totals).

Both markets are bets on whether an integer score quantity X lands above or
below a line:

    spreads  X = home_score - away_score   home -3.5 wins if X > 3.5
                                           away +3.5 wins if X < 3.5
    totals   X = home_score + away_score   over 44.5 wins if X > 44.5

X is modelled as a discretised Normal(mu, sd). On a whole-number line, the
probability mass at exactly the line is a push (stake refunded).

The sharp book's line and de-vigged price pin down the market's implied mean
(`implied_mean`), so model and market probabilities can be compared at any
soft book's line, even when books hang different numbers.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

LINE_MARKETS = ("spreads", "totals")
MARKET_SIDES = {"h2h": ("home", "away"), "spreads": ("home", "away"), "totals": ("over", "under")}


def above_probs(mu, sd: float, threshold) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """P(X > t), P(X == t), P(X < t) for integer X ~ discretised Normal(mu, sd)."""
    mu, t = np.broadcast_arrays(np.asarray(mu, float), np.asarray(threshold, float))
    whole = np.isclose(t, np.round(t))
    hi = np.where(whole, t + 0.5, t)
    lo = np.where(whole, t - 0.5, t)
    p_above = 1.0 - ndtr((hi - mu) / sd)
    p_below = ndtr((lo - mu) / sd)
    return p_above, np.clip(1.0 - p_above - p_below, 0.0, 1.0), p_below


def implied_mean(threshold, p_above_no_push, sd: float, iters: int = 60) -> np.ndarray:
    """Mean of X at which P(X > t | no push) equals the given probability."""
    t = np.asarray(threshold, float)
    target = np.asarray(p_above_no_push, float)
    lo, hi = t - 80.0, t + 80.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        pa, _, pb = above_probs(mid, sd, t)
        too_low = pa / np.maximum(pa + pb, 1e-12) < target
        lo, hi = np.where(too_low, mid, lo), np.where(too_low, hi, mid)
    return 0.5 * (lo + hi)


def selection_probs(market: str, side, line, mu, sd: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(win, push, lose) for a selection at its own quoted line.

    `line` is the selection's point: the home or away handicap for spreads, the
    total for totals.
    """
    side = np.asarray(side)
    line = np.asarray(line, float)
    if market == "spreads":
        # home covers when X > -line_home; away covers when X < line_away
        threshold = np.where(side == "home", -line, line)
        above, push, below = above_probs(mu, sd, threshold)
        win_on_above = side == "home"
    elif market == "totals":
        above, push, below = above_probs(mu, sd, line)
        win_on_above = side == "over"
    else:
        raise ValueError(f"{market!r} is not a line market")
    return np.where(win_on_above, above, below), push, np.where(win_on_above, below, above)


def market_mean(market: str, home_or_over_line, p_home_or_over: np.ndarray, sd: float) -> np.ndarray:
    """Implied mean of X from the sharp line and its de-vigged home/over probability."""
    line = np.asarray(home_or_over_line, float)
    threshold = -line if market == "spreads" else line
    return implied_mean(threshold, p_home_or_over, sd)


def format_point(point: float, signed: bool) -> str:
    text = f"{point:+.1f}" if signed else f"{point:.1f}"
    return text.replace(".0", "") if text.endswith(".0") else text
