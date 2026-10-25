"""Remove the bookmaker margin (vig) from a set of mutually exclusive prices.

All functions take decimal odds for every outcome of one market and return fair
probabilities that sum to 1.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def implied_probs(odds: Sequence[float]) -> np.ndarray:
    o = np.asarray(odds, dtype=float)
    if np.any(o <= 1.0):
        raise ValueError(f"Decimal odds must be > 1.0, got {odds}")
    return 1.0 / o


def overround(odds: Sequence[float]) -> float:
    """Book margin, e.g. 0.045 for a 4.5% hold."""
    return float(implied_probs(odds).sum() - 1.0)


def devig_proportional(odds: Sequence[float]) -> np.ndarray:
    """Multiplicative method: scale implied probabilities so they sum to 1."""
    p = implied_probs(odds)
    return p / p.sum()


def devig_shin(odds: Sequence[float], tol: float = 1e-12, max_iter: int = 200) -> np.ndarray:
    """Shin (1993) method.

    Models the margin as protection against a share `z` of insiders, which
    assigns more of the vig to longshots (the favourite-longshot bias). Solves
    for z by bisection so that the fair probabilities sum to 1.
    """
    pi = implied_probs(odds)
    total = pi.sum()
    if total <= 1.0:
        return pi / total  # no margin to remove (or an arb): fall back

    def fair(z: float) -> np.ndarray:
        return (np.sqrt(z * z + 4.0 * (1.0 - z) * pi * pi / total) - z) / (2.0 * (1.0 - z))

    lo, hi = 0.0, 0.5
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if fair(mid).sum() > 1.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    p = fair(0.5 * (lo + hi))
    return p / p.sum()


def devig_power(odds: Sequence[float], tol: float = 1e-12, max_iter: int = 200) -> np.ndarray:
    """Power method: p_i = pi_i ** k with k chosen so that sum(p) = 1."""
    pi = implied_probs(odds)
    if pi.sum() <= 1.0:
        return pi / pi.sum()
    lo, hi = 1.0, 10.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if (pi**mid).sum() > 1.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    p = pi ** (0.5 * (lo + hi))
    return p / p.sum()


METHODS: dict[str, Callable[[Sequence[float]], np.ndarray]] = {
    "proportional": devig_proportional,
    "shin": devig_shin,
    "power": devig_power,
}


def devig(odds: Sequence[float], method: str = "shin") -> np.ndarray:
    try:
        return METHODS[method](odds)
    except KeyError as exc:
        raise ValueError(f"Unknown devig method {method!r}; choose from {sorted(METHODS)}") from exc
