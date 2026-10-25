"""Edge, Value-to-Odds Score (VOS), confidence tiers and pick filtering.

    EV  = P_model * Odds - 1
    VOS = ((P_model - P_market) / P_market * edge_scale) * log10(Odds) * Tier

`edge_scale` (default 100) expresses the relative edge in percent; see config.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from config import settings

TIER_MULTIPLIERS: dict[str, float] = {"A": 1.0, "B": 0.75, "C": 0.5}


def expected_value(p_model: float | np.ndarray, odds: float | np.ndarray) -> float | np.ndarray:
    return p_model * odds - 1.0


def vos_score(
    p_model: float | np.ndarray,
    p_market: float | np.ndarray,
    odds: float | np.ndarray,
    tier_mult: float | np.ndarray,
    edge_scale: float | None = None,
) -> float | np.ndarray:
    scale = settings.vos_edge_scale if edge_scale is None else edge_scale
    return ((p_model - p_market) / p_market) * scale * np.log10(odds) * tier_mult


def confidence_tier(has_sharp: bool, min_games_played: float, sharp_move_toward: float) -> str:
    """A/B/C tier from three data-quality and market-agreement checks.

    +1  a sharp-book line exists at decision time (otherwise we de-vig consensus)
    +1  both teams have >= `min_games_for_tier` games this season (form features are mature)
    +1  the sharp line has not moved against the selection since open
    """
    points = int(bool(has_sharp)) + int(min_games_played >= settings.min_games_for_tier) + int(sharp_move_toward >= -0.005)
    return "A" if points == 3 else "B" if points == 2 else "C"


def kelly_fraction(p: float, odds: float) -> float:
    """Full-Kelly fraction of bankroll for decimal `odds` (0 if no edge)."""
    b = odds - 1.0
    if b <= 0:
        return 0.0
    return max((b * p - (1.0 - p)) / b, 0.0)


def stake_size(
    p: float,
    odds: float,
    bankroll: float,
    fraction: float | None = None,
    max_pct: float | None = None,
) -> float:
    """Fractional Kelly with a hard cap per pick."""
    frac = settings.kelly_fraction if fraction is None else fraction
    cap = settings.max_stake_pct if max_pct is None else max_pct
    return bankroll * min(frac * kelly_fraction(p, odds), cap)


def blend_with_market(p_model: np.ndarray, p_market: np.ndarray, weight: float | None = None) -> np.ndarray:
    """Logit-space shrinkage of model probabilities toward the market."""
    w = settings.model_weight if weight is None else weight
    lm = np.log(p_market / (1 - p_market))
    lp = np.log(np.clip(p_model, 1e-6, 1 - 1e-6) / (1 - np.clip(p_model, 1e-6, 1 - 1e-6)))
    return 1.0 / (1.0 + np.exp(-(lm + w * (lp - lm))))


def build_opportunities(df: pd.DataFrame, p_home: np.ndarray) -> pd.DataFrame:
    """Expand each game into home/away opportunities with EV, VOS and tier."""
    p_home = blend_with_market(np.asarray(p_home), df["p_sharp_home"].to_numpy())
    rows = []
    for (_, g), ph in zip(df.iterrows(), p_home):
        for side, p_model, p_mkt, sign in (
            ("home", ph, g["p_sharp_home"], 1.0),
            ("away", 1.0 - ph, 1.0 - g["p_sharp_home"], -1.0),
        ):
            odds = float(g[f"best_{side}_price"])
            close_mkt = g.get("close_p_sharp_home")
            tier = confidence_tier(g["has_sharp"], g.get("min_games_played", 0) or 0, sign * g["sharp_move"])
            rows.append(
                {
                    "game_id": g["game_id"],
                    "game_date": g["game_date"],
                    "commence_time": g["commence_time"],
                    "fixture": f"{g['away_team']} @ {g['home_team']}",
                    "selection": g[f"{side}_team"],
                    "side": side,
                    "book": g[f"best_{side}_book"],
                    "odds": odds,
                    "p_model": float(p_model),
                    "p_market": float(p_mkt),
                    "p_close": float(close_mkt if side == "home" else 1.0 - close_mkt) if pd.notna(close_mkt) else math.nan,
                    "tier": tier,
                    "won": (g["home_win"] if side == "home" else 1.0 - g["home_win"]) if pd.notna(g["home_win"]) else math.nan,
                }
            )
    opp = pd.DataFrame(rows)
    if opp.empty:
        return opp
    opp["ev"] = expected_value(opp["p_model"], opp["odds"])
    opp["tier_mult"] = opp["tier"].map(TIER_MULTIPLIERS)
    opp["vos"] = vos_score(opp["p_model"], opp["p_market"], opp["odds"], opp["tier_mult"])
    return opp


def select_picks(opp: pd.DataFrame, max_per_day: int | None = None) -> pd.DataFrame:
    """Apply the Top Line Picks filters and rank by VOS (best side per game only)."""
    if opp.empty:
        return opp
    mask = (
        (opp["vos"] > settings.min_vos)
        & (opp["ev"] > settings.min_ev)
        & opp["odds"].between(settings.min_odds, settings.max_odds)
    )
    picks = opp[mask].sort_values("vos", ascending=False)
    picks = picks.drop_duplicates("game_id")
    limit = settings.max_picks_per_day if max_per_day is None else max_per_day
    return picks.groupby("game_date", group_keys=False).head(limit).reset_index(drop=True)


def size_picks(picks: pd.DataFrame, bankroll: float) -> pd.DataFrame:
    """Kelly stakes per pick, scaled down if the day's exposure cap is hit."""
    picks = picks.copy()
    if picks.empty:
        picks["stake"] = []
        return picks
    picks["kelly_full"] = [kelly_fraction(p, o) for p, o in zip(picks["p_model"], picks["odds"])]
    picks["stake"] = [stake_size(p, o, bankroll) for p, o in zip(picks["p_model"], picks["odds"])]
    cap = bankroll * settings.max_daily_exposure_pct
    total = picks["stake"].sum()
    if total > cap > 0:
        picks["stake"] *= cap / total
    picks["stake_pct"] = picks["stake"] / bankroll
    return picks
