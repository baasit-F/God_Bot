"""Edge, Value-to-Odds Score (VOS), confidence tiers and pick filtering.

    EV  = P_win * Odds - 1 - P_push * (Odds - 1)   (= P_model * Odds - 1 with no push)
    VOS = ((P_model - P_market) / P_market * edge_scale) * log10(Odds) * Tier

For spreads and totals on whole-number lines a push refunds the stake, so EV and
Kelly use win/push/lose probabilities, while VOS compares win probabilities
conditional on no push (P_model, P_market). `edge_scale` (default 100)
expresses the relative edge in percent; see config.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from config import settings
from models.features import LINE_PREFIX
from models.lines import format_point, selection_probs

TIER_MULTIPLIERS: dict[str, float] = {"A": 1.0, "B": 0.75, "C": 0.5}


LINE_MOVE_TOLERANCE_PTS = 0.25


def expected_value(p_win, odds, p_push=0.0):
    """EV per unit staked; a push returns the stake (no profit, no loss)."""
    p_lose = 1.0 - p_win - p_push
    return p_win * (odds - 1.0) - p_lose


def vos_score(
    p_model: float | np.ndarray,
    p_market: float | np.ndarray,
    odds: float | np.ndarray,
    tier_mult: float | np.ndarray,
    edge_scale: float | None = None,
) -> float | np.ndarray:
    scale = settings.vos_edge_scale if edge_scale is None else edge_scale
    return ((p_model - p_market) / p_market) * scale * np.log10(odds) * tier_mult


def confidence_tier(has_sharp: bool, min_games_played: float, moved_against: bool) -> str:
    """A/B/C tier from three data-quality and market-agreement checks.

    +1  a sharp-book line exists at decision time (otherwise we de-vig consensus)
    +1  both teams have >= `min_games_for_tier` games this season (form features are mature)
    +1  the sharp line has not moved against the selection since open
    """
    points = int(bool(has_sharp)) + int(min_games_played >= settings.min_games_for_tier) + int(not moved_against)
    return "A" if points == 3 else "B" if points == 2 else "C"


def kelly_fraction(p_win: float, odds: float, p_push: float = 0.0) -> float:
    """Full-Kelly fraction of bankroll for decimal `odds` (0 if no edge).

    With pushes: f* = (b * p_win - p_lose) / b.
    """
    b = odds - 1.0
    if b <= 0:
        return 0.0
    return max((b * p_win - (1.0 - p_win - p_push)) / b, 0.0)


def stake_size(
    p_win: float,
    odds: float,
    bankroll: float,
    fraction: float | None = None,
    max_pct: float | None = None,
    p_push: float = 0.0,
) -> float:
    """Fractional Kelly with a hard cap per pick."""
    frac = settings.kelly_fraction if fraction is None else fraction
    cap = settings.max_stake_pct if max_pct is None else max_pct
    return bankroll * min(frac * kelly_fraction(p_win, odds, p_push), cap)


def blend_with_market(p_model: np.ndarray, p_market: np.ndarray, weight: float | None = None) -> np.ndarray:
    """Logit-space shrinkage of model probabilities toward the market."""
    w = settings.model_weight if weight is None else weight
    lm = np.log(p_market / (1 - p_market))
    lp = np.log(np.clip(p_model, 1e-6, 1 - 1e-6) / (1 - np.clip(p_model, 1e-6, 1 - 1e-6)))
    return 1.0 / (1.0 + np.exp(-(lm + w * (lp - lm))))


def build_opportunities(df: pd.DataFrame, p_home: np.ndarray) -> pd.DataFrame:
    """Expand each game into moneyline home/away opportunities with EV, VOS and tier."""
    p_home = blend_with_market(np.asarray(p_home), df["p_sharp_home"].to_numpy())
    rows = []
    for (_, g), ph in zip(df.iterrows(), p_home):
        for side, p_model, p_mkt, sign in (
            ("home", ph, g["p_sharp_home"], 1.0),
            ("away", 1.0 - ph, 1.0 - g["p_sharp_home"], -1.0),
        ):
            odds = float(g[f"best_{side}_price"])
            close_mkt = g.get("close_p_sharp_home")
            tier = confidence_tier(g["has_sharp"], g.get("min_games_played", 0) or 0, sign * g["sharp_move"] < -0.005)
            rows.append(
                {
                    "game_id": g["game_id"],
                    "game_date": g["game_date"],
                    "commence_time": g["commence_time"],
                    "fixture": f"{g['away_team']} @ {g['home_team']}",
                    "market": "h2h",
                    "selection": f"{g[f'{side}_team']} ML",
                    "side": side,
                    "line": math.nan,
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
    opp["p_win"], opp["p_push"] = opp["p_model"], 0.0
    opp["ev"] = expected_value(opp["p_win"], opp["odds"])
    return _finish(opp)


def _finish(opp: pd.DataFrame) -> pd.DataFrame:
    opp["tier_mult"] = opp["tier"].map(TIER_MULTIPLIERS)
    opp["vos"] = vos_score(opp["p_model"], opp["p_market"], opp["odds"], opp["tier_mult"])
    return opp


def _settle_line(market: str, side: pd.Series, point: pd.Series, actual: pd.Series) -> pd.Series:
    """1 win / 0.5 push / 0 loss / NaN unsettled, for spreads and totals."""
    if market == "spreads":
        diff = np.where(side == "home", actual + point, -actual + point)
    else:
        diff = np.where(side == "over", actual - point, point - actual)
    return pd.Series(np.select([diff > 0, diff == 0, diff < 0], [1.0, 0.5, 0.0], np.nan), index=side.index)


def build_line_opportunities(df: pd.DataFrame, quotes: pd.DataFrame, mu_model: np.ndarray, market: str) -> pd.DataFrame:
    """Spread or total opportunities: every bettable quote priced under the model,
    keeping the best-EV book per selection.

    `mu_model` is the model's mean home margin (spreads) or total (totals) for
    each row of `df`.
    """
    prefix = LINE_PREFIX[market]
    sd = settings.margin_sd if market == "spreads" else settings.total_sd
    actual_col = "actual_margin" if market == "spreads" else "actual_total"
    cols = ["game_id", "game_date", "commence_time", "home_team", "away_team", "min_games_played", actual_col,
            f"{prefix}_mu", f"close_{prefix}_mu", f"{prefix}_has_sharp", f"{prefix}_move"]
    g = df[cols].copy()
    g["mu_model"] = np.asarray(mu_model, float)
    q = quotes[quotes["market"] == market].merge(g, on="game_id")
    if q.empty:
        return pd.DataFrame()

    side, point, odds = q["selection"].to_numpy(), q["point"].to_numpy(float), q["price"].to_numpy(float)
    win, push, lose = selection_probs(market, side, point, q["mu_model"].to_numpy(), sd)
    m_win, _, m_lose = selection_probs(market, side, point, q[f"{prefix}_mu"].to_numpy(), sd)
    c_win, _, c_lose = selection_probs(market, side, point, q[f"close_{prefix}_mu"].to_numpy(), sd)

    toward_first = np.isin(side, ["home", "over"])
    move = np.where(toward_first, 1.0, -1.0) * q[f"{prefix}_move"].to_numpy(float)
    if market == "spreads":
        label = np.where(side == "home", q["home_team"], q["away_team"])
        selection = [f"{t} {format_point(p, signed=True)}" for t, p in zip(label, point)]
    else:
        selection = [f"{s.title()} {format_point(p, signed=False)}" for s, p in zip(side, point)]

    opp = pd.DataFrame(
        {
            "game_id": q["game_id"],
            "game_date": q["game_date"],
            "commence_time": q["commence_time"],
            "fixture": q["away_team"] + " @ " + q["home_team"],
            "market": market,
            "selection": selection,
            "side": side,
            "line": point,
            "book": q["bookmaker"],
            "odds": odds,
            "p_win": win,
            "p_push": push,
            "p_model": win / np.maximum(win + lose, 1e-12),
            "p_market": m_win / np.maximum(m_win + m_lose, 1e-12),
            "p_close": c_win / np.maximum(c_win + c_lose, 1e-12),
            "tier": [
                confidence_tier(hs, mg, mv < -LINE_MOVE_TOLERANCE_PTS)
                for hs, mg, mv in zip(q[f"{prefix}_has_sharp"].fillna(False), q["min_games_played"].fillna(0), move)
            ],
            "won": _settle_line(market, q["selection"], q["point"], q[actual_col]),
        }
    )
    opp["ev"] = expected_value(opp["p_win"], opp["odds"], opp["p_push"])
    opp = opp.sort_values("ev", ascending=False).drop_duplicates(["game_id", "side"]).sort_index()
    return _finish(opp.reset_index(drop=True))


def select_picks(opp: pd.DataFrame, max_per_day: int | None = None) -> pd.DataFrame:
    """Apply the Top Line Picks filters and rank by VOS.

    One side per game per market. Unless `allow_correlated`, moneyline and spread
    picks on the same game compete and only the higher VOS survives.
    """
    if opp.empty:
        return opp
    mask = (
        (opp["vos"] > settings.min_vos)
        & (opp["ev"] > settings.min_ev)
        & opp["odds"].between(settings.min_odds, settings.max_odds)
    )
    picks = opp[mask].sort_values("vos", ascending=False)
    picks = picks.drop_duplicates(["game_id", "market"])
    if not settings.allow_correlated:
        group = np.where(picks["market"] == "totals", "totals", "side")
        picks = picks.assign(_group=group).drop_duplicates(["game_id", "_group"]).drop(columns="_group")
    limit = settings.max_picks_per_day if max_per_day is None else max_per_day
    return picks.groupby("game_date", group_keys=False).head(limit).reset_index(drop=True)


def size_picks(picks: pd.DataFrame, bankroll: float) -> pd.DataFrame:
    """Kelly stakes per pick, scaled down if the day's exposure cap is hit."""
    picks = picks.copy()
    if picks.empty:
        picks["stake"] = []
        return picks
    args = list(zip(picks["p_win"], picks["odds"], picks["p_push"]))
    picks["kelly_full"] = [kelly_fraction(p, o, push) for p, o, push in args]
    picks["stake"] = [stake_size(p, o, bankroll, p_push=push) for p, o, push in args]
    cap = bankroll * settings.max_daily_exposure_pct
    total = picks["stake"].sum()
    if total > cap > 0:
        picks["stake"] *= cap / total
    picks["stake_pct"] = picks["stake"] / bankroll
    return picks
