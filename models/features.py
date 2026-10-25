"""Leakage-free feature engineering.

Two families of features, both built so that a game's row only uses
information that existed when the bot had to decide:

Market features  come from the latest odds snapshot at or before the decision
                 time (`commence_time - decision_hours_before`). Closing prices
                 are kept in `close_*` columns for CLV reporting only and are
                 never in FEATURES.
Team features    come from a single chronological pass that processes one
                 calendar day at a time: every game on day D is featurised from
                 team state built from games strictly before D, then the state
                 is updated with day D's results.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import settings
from models.devig import devig

FEATURES: list[str] = [
    "sharp_logit",          # de-vigged sharp-book home prob at decision time (logit)
    "sharp_move",           # sharp home prob change from open to decision
    "sharp_soft_gap",       # sharp minus soft-book consensus home prob
    "money_ticket_gap",     # home handle % minus home ticket % (sharp money tell)
    "elo_logit",            # Elo win probability (logit), home field included
    "margin_diff",          # avg point margin over the form window, home minus away
    "ypp_diff",             # rolling (off - def) yards per play, home minus away
    "rest_diff",            # home rest days minus away rest days (capped at 14)
    "home_off_bye",         # home team coming off its bye week
    "away_off_bye",
    "home_short_week",      # home team on short rest (Thursday after Sunday)
    "away_short_week",
]

REST_CAP_DAYS = 14
BYE_REST_DAYS = 13
SHORT_WEEK_DAYS = 5


def _logit(p: np.ndarray | float) -> np.ndarray | float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# --------------------------------------------------------------------------- market
def _book_probs(snap: pd.DataFrame, method: str) -> pd.DataFrame:
    """De-vig each (game, book) two-way market from a latest-snapshot frame."""
    wide = snap.pivot_table(index=["game_id", "bookmaker"], columns="selection", values="price", aggfunc="last")
    wide = wide.dropna(subset=["home", "away"])
    probs = np.array([devig([h, a], method) for h, a in zip(wide["home"], wide["away"])]) if len(wide) else np.empty((0, 2))
    out = wide.reset_index()[["game_id", "bookmaker", "home", "away"]].copy()
    out["p_home"] = probs[:, 0] if len(probs) else []
    return out


def _latest(odds: pd.DataFrame, cutoff: pd.Series) -> pd.DataFrame:
    """Latest snapshot per (game, book, selection) with snapshot_time <= cutoff[game]."""
    df = odds.merge(cutoff.rename("cutoff"), left_on="game_id", right_index=True)
    df = df[df["snapshot_time"] <= df["cutoff"]]
    df = df.sort_values("snapshot_time")
    return df.groupby(["game_id", "bookmaker", "selection"], as_index=False).last()


def _earliest(odds: pd.DataFrame) -> pd.DataFrame:
    return odds.sort_values("snapshot_time").groupby(["game_id", "bookmaker", "selection"], as_index=False).first()


def build_market_features(
    games: pd.DataFrame,
    odds: pd.DataFrame,
    sentiment: pd.DataFrame | None = None,
    decision_hours_before: float | None = None,
    sharp_book: str | None = None,
    bettable_books: list[str] | None = None,
    method: str | None = None,
) -> pd.DataFrame:
    """One row per game with sharp/soft probabilities, best prices and splits.

    `decision_hours_before=0` uses the latest pre-game snapshot (daily mode).
    """
    hours = settings.decision_hours_before if decision_hours_before is None else decision_hours_before
    sharp = sharp_book or settings.sharp_book
    books = bettable_books or settings.bettable_books
    method = method or settings.devig_method
    odds = odds[odds["market"] == "h2h"]

    g = games.set_index("game_id")
    decision = g["commence_time"] - pd.Timedelta(hours=hours)
    out = pd.DataFrame(index=g.index)

    dec = _book_probs(_latest(odds, decision), method)
    opn = _book_probs(_earliest(odds[odds["game_id"].isin(g.index)]), method)
    cls = _book_probs(_latest(odds, g["commence_time"]), method)

    def sharp_prob(frame: pd.DataFrame) -> pd.Series:
        s = frame[frame["bookmaker"] == sharp].set_index("game_id")["p_home"]
        # Fall back to the all-book consensus if the sharp book is missing.
        consensus = frame.groupby("game_id")["p_home"].mean()
        return s.reindex(consensus.index).fillna(consensus)

    out["p_sharp_home"] = sharp_prob(dec)
    out["has_sharp"] = dec[dec["bookmaker"] == sharp].set_index("game_id")["p_home"].reindex(out.index).notna()
    out["p_sharp_home_open"] = sharp_prob(opn)
    out["close_p_sharp_home"] = sharp_prob(cls)

    soft = dec[dec["bookmaker"].isin(books)]
    out["p_soft_home"] = soft.groupby("game_id")["p_home"].mean()

    for side in ("home", "away"):
        best = soft.loc[soft.groupby("game_id")[side].idxmax()] if len(soft) else soft
        out[f"best_{side}_price"] = best.set_index("game_id")[side]
        out[f"best_{side}_book"] = best.set_index("game_id")["bookmaker"]

    out["sharp_logit"] = _logit(out["p_sharp_home"])
    out["sharp_move"] = (out["p_sharp_home"] - out["p_sharp_home_open"]).fillna(0.0)
    out["sharp_soft_gap"] = (out["p_sharp_home"] - out["p_soft_home"]).fillna(0.0)

    out["money_ticket_gap"] = 0.0
    if sentiment is not None and not sentiment.empty:
        sent = _latest_sentiment(sentiment, decision)
        out["money_ticket_gap"] = (sent["home_money_pct"] - sent["home_ticket_pct"]).reindex(out.index).fillna(0.0)
        out["home_ticket_pct"] = sent["home_ticket_pct"].reindex(out.index)

    return out.reset_index()


def _latest_sentiment(sentiment: pd.DataFrame, cutoff: pd.Series) -> pd.DataFrame:
    df = sentiment.merge(cutoff.rename("cutoff"), left_on="game_id", right_index=True)
    df = df[df["snapshot_time"] <= df["cutoff"]].sort_values("snapshot_time")
    return df.groupby("game_id").last()


# ----------------------------------------------------------------------------- team
@dataclass
class TeamState:
    elo: float = 1500.0
    season: str | None = None
    last_date: pd.Timestamp | None = None
    games_in_season: int = 0
    margins: deque = field(default_factory=lambda: deque(maxlen=settings.form_window))
    ypp_net: deque = field(default_factory=lambda: deque(maxlen=settings.form_window))


def _elo_prob(home: TeamState, away: TeamState) -> float:
    return 1.0 / (1.0 + 10 ** (-((home.elo + settings.elo_home_adv) - away.elo) / 400.0))


def build_team_features(games: pd.DataFrame, team_stats: pd.DataFrame | None = None) -> pd.DataFrame:
    """Elo / form / rest features, featurising each day before learning from it."""
    stats = {}
    if team_stats is not None and not team_stats.empty:
        ts = team_stats.dropna(subset=["off_ypp", "def_ypp"])
        stats = {(r.game_id, r.team): r.off_ypp - r.def_ypp for r in ts.itertuples()}

    state: dict[str, TeamState] = defaultdict(TeamState)
    rows = []
    games = games.sort_values("commence_time")

    for day, day_games in games.groupby("game_date", sort=True):
        # 1) featurise every game of the day from pre-day state
        for gm in day_games.itertuples():
            h, a = state[gm.home_team], state[gm.away_team]
            for team in (h, a):
                if team.season is not None and team.season != gm.season:
                    team.elo = 1500.0 + (team.elo - 1500.0) * (1 - settings.elo_season_regress)
                    team.games_in_season = 0
                    team.margins.clear()
                    team.ypp_net.clear()
                    team.last_date = None
                team.season = gm.season

            elo_p = _elo_prob(h, a)
            # Week 1 counts as fully rested (off-season), not as coming off a bye.
            rest_h = min((day - h.last_date).days, REST_CAP_DAYS) if h.last_date is not None else None
            rest_a = min((day - a.last_date).days, REST_CAP_DAYS) if a.last_date is not None else None
            rows.append(
                {
                    "game_id": gm.game_id,
                    "elo_prob": elo_p,
                    "elo_logit": float(_logit(elo_p)),
                    "margin_diff": (np.mean(h.margins) if h.margins else 0.0) - (np.mean(a.margins) if a.margins else 0.0),
                    "ypp_diff": (np.mean(h.ypp_net) if h.ypp_net else 0.0) - (np.mean(a.ypp_net) if a.ypp_net else 0.0),
                    "rest_diff": (rest_h or 7) - (rest_a or 7),
                    "home_off_bye": int(rest_h is not None and rest_h >= BYE_REST_DAYS),
                    "away_off_bye": int(rest_a is not None and rest_a >= BYE_REST_DAYS),
                    "home_short_week": int(rest_h is not None and rest_h <= SHORT_WEEK_DAYS),
                    "away_short_week": int(rest_a is not None and rest_a <= SHORT_WEEK_DAYS),
                    "min_games_played": min(h.games_in_season, a.games_in_season),
                }
            )

        # 2) learn from the day's final results
        for gm in day_games.itertuples():
            if gm.status != "final" or pd.isna(gm.home_score):
                continue
            h, a = state[gm.home_team], state[gm.away_team]
            margin = float(gm.home_score - gm.away_score)
            expected = _elo_prob(h, a)
            actual = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
            mov_mult = math.log(abs(margin) + 1.0) * (2.2 / ((abs(h.elo - a.elo) * 0.001) + 2.2))
            delta = settings.elo_k * mov_mult * (actual - expected)
            h.elo += delta
            a.elo -= delta
            h.margins.append(margin)
            a.margins.append(-margin)
            if (gm.game_id, gm.home_team) in stats:
                h.ypp_net.append(stats[(gm.game_id, gm.home_team)])
            if (gm.game_id, gm.away_team) in stats:
                a.ypp_net.append(stats[(gm.game_id, gm.away_team)])
            for team in (h, a):
                team.last_date = day
                team.games_in_season += 1

    return pd.DataFrame(rows)


def build_feature_frame(
    games: pd.DataFrame,
    odds: pd.DataFrame,
    sentiment: pd.DataFrame | None = None,
    team_stats: pd.DataFrame | None = None,
    decision_hours_before: float | None = None,
) -> pd.DataFrame:
    """Join game metadata, market features, team features and the target."""
    market = build_market_features(games, odds, sentiment, decision_hours_before=decision_hours_before)
    team = build_team_features(games, team_stats)
    df = games.merge(market, on="game_id", how="left").merge(team, on="game_id", how="left")
    # 1 = home win, 0 = away win, 0.5 = tie (a push on a two-way moneyline).
    df["home_win"] = np.where(
        df["status"] == "final",
        np.sign(df["home_score"] - df["away_score"]).map({1.0: 1.0, -1.0: 0.0, 0.0: 0.5}),
        np.nan,
    )
    df = df.dropna(subset=["p_sharp_home", "best_home_price", "best_away_price"])
    df[FEATURES] = df[FEATURES].fillna(0.0)
    return df.sort_values("commence_time").reset_index(drop=True)
