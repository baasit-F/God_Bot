import math

import numpy as np
import pandas as pd
import pytest

from backtest import walkforward
from data.storage import Storage
from generate_mock_data import generate
from models.devig import devig, devig_proportional, devig_shin, overround
from models.features import FEATURES, SPREAD_FEATURES, TOTAL_FEATURES, build_feature_frame, build_quotes
from models.lines import above_probs, market_mean, selection_probs
from models.scoring import _settle_line, expected_value, kelly_fraction, stake_size, vos_score


# ------------------------------------------------------------------ de-vigging
@pytest.mark.parametrize("method", ["proportional", "shin", "power"])
def test_devig_sums_to_one(method):
    p = devig([1.50, 2.70], method)
    assert p.sum() == pytest.approx(1.0)
    assert (p > 0).all()


def test_shin_assigns_more_margin_to_longshot():
    odds = [1.25, 4.20]
    prop, shin = devig_proportional(odds), devig_shin(odds)
    assert shin[0] > prop[0]  # favourite fairer under Shin
    assert shin[1] < prop[1]


def test_no_vig_market_is_unchanged():
    p = devig_shin([2.0, 2.0])
    assert p == pytest.approx([0.5, 0.5])
    assert overround([2.0, 2.0]) == pytest.approx(0.0)


# --------------------------------------------------------------- edge & sizing
def test_ev_and_kelly():
    assert expected_value(0.55, 2.0) == pytest.approx(0.10)
    assert kelly_fraction(0.55, 2.0) == pytest.approx(0.10)
    assert kelly_fraction(0.40, 2.0) == 0.0


def test_stake_is_capped():
    # Full Kelly 0.6 -> quarter Kelly 15% -> hard cap 3%
    assert stake_size(0.80, 2.0, 10_000, fraction=0.25, max_pct=0.03) == pytest.approx(300.0)
    assert stake_size(0.52, 2.0, 10_000, fraction=0.25, max_pct=0.03) == pytest.approx(10_000 * 0.25 * 0.04)


def test_vos_formula():
    # 10% relative edge, odds 2.0, tier A, edge in percent
    expected = 10.0 * math.log10(2.0) * 1.0
    assert vos_score(0.55, 0.50, 2.0, 1.0, edge_scale=100) == pytest.approx(expected)


# ---------------------------------------------------------------- line markets
def test_whole_number_line_has_push_mass():
    above, push, below = above_probs(3.0, 13.5, 3)
    assert push > 0.02
    assert above + push + below == pytest.approx(1.0)
    assert above_probs(3.0, 13.5, 3.5)[1] == pytest.approx(0.0)


def test_implied_mean_round_trips_through_line_price():
    # A sharp line of home -6.5 at a fair 55% cover price implies a mean margin above 6.5.
    mu = market_mean("spreads", np.array([-6.5]), np.array([0.55]), 13.5)[0]
    win, push, lose = selection_probs("spreads", np.array(["home"]), np.array([-6.5]), mu, 13.5)
    assert mu > 6.5
    assert win[0] / (win[0] + lose[0]) == pytest.approx(0.55, abs=1e-6)


def test_spread_sides_are_complementary():
    win, push, lose = selection_probs("spreads", np.array(["home", "away"]), np.array([-3.0, 3.0]), 2.0, 13.5)
    assert win[0] == pytest.approx(lose[1])
    assert push[0] == pytest.approx(push[1])


def test_totals_over_under_complementary():
    win, push, lose = selection_probs("totals", np.array(["over", "under"]), np.array([44.5, 44.5]), 46.0, 10.0)
    assert win[0] == pytest.approx(lose[1])
    assert win[0] > 0.5


def test_line_settlement():
    side = pd.Series(["home", "away", "home", "over", "under"])
    point = pd.Series([-3.0, 3.0, -3.5, 44.0, 44.0])
    actual = pd.Series([3.0, 3.0, 4.0, 45.0, 45.0])
    assert _settle_line("spreads", side[:3], point[:3], actual[:3]).tolist() == [0.5, 0.5, 1.0]
    assert _settle_line("totals", side[3:], point[3:], actual[3:]).tolist() == [1.0, 0.0]


def test_push_aware_ev_and_kelly():
    # 50% win, 5% push, 45% lose at 2.0: EV = 0.5 - 0.45 = +5%; Kelly = (1*0.5 - 0.45)/1
    assert expected_value(0.50, 2.0, 0.05) == pytest.approx(0.05)
    assert kelly_fraction(0.50, 2.0, 0.05) == pytest.approx(0.05)


# -------------------------------------------------------------- data leakage
@pytest.fixture(scope="module")
def mock_store(tmp_path_factory):
    store = Storage(tmp_path_factory.mktemp("db") / "test.db")
    generate(store, seasons=2, today=pd.Timestamp("2026-10-25"), seed=7)
    return store


def test_features_ignore_future_results(mock_store):
    games, odds = mock_store.games(), mock_store.odds()
    base = build_feature_frame(games, odds, mock_store.sentiment(), mock_store.team_stats())

    cutoff = games["game_date"].sort_values().iloc[len(games) // 2]
    tampered = games.copy()
    future = tampered["game_date"] >= cutoff
    tampered.loc[future, ["home_score", "away_score"]] = tampered.loc[future, ["away_score", "home_score"]].values
    after = build_feature_frame(tampered, odds, mock_store.sentiment(), mock_store.team_stats())

    # Every feature of every game up to and including the cutoff day is unchanged.
    cols = sorted(set(FEATURES + SPREAD_FEATURES + TOTAL_FEATURES))
    a = base[base["game_date"] <= cutoff].set_index("game_id")[cols]
    b = after[after["game_date"] <= cutoff].set_index("game_id")[cols]
    pd.testing.assert_frame_equal(a, b)


def test_market_features_ignore_post_decision_odds(mock_store):
    games, odds = mock_store.games(), mock_store.odds()
    base = build_feature_frame(games, odds, decision_hours_before=3)
    commence = games.set_index("game_id")["commence_time"]
    late = odds["snapshot_time"] > odds["game_id"].map(commence) - pd.Timedelta(hours=3)
    shocked = odds.copy()
    shocked.loc[late, "price"] = 1.01 + np.random.default_rng(0).random(late.sum()) * 10
    shocked.loc[late, "point"] += 7.0  # move every post-decision spread/total line
    after = build_feature_frame(games, shocked, decision_hours_before=3)
    cols = ["sharp_logit", "sharp_move", "sharp_soft_gap", "best_home_price", "best_away_price",
            "spread_mu", "spread_move", "total_mu", "total_move"]
    pd.testing.assert_frame_equal(base.set_index("game_id")[cols], after.set_index("game_id")[cols])
    key = ["game_id", "market", "bookmaker", "selection"]
    qa = build_quotes(games, odds, decision_hours_before=3).sort_values(key).reset_index(drop=True)
    qb = build_quotes(games, shocked, decision_hours_before=3).sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(qa, qb)


def test_walk_forward_trains_only_on_past(mock_store, monkeypatch):
    games, odds = mock_store.games(), mock_store.odds()
    feats = build_feature_frame(games, odds, mock_store.sentiment(), mock_store.team_stats())
    seen: list[pd.Timestamp] = []
    real_fit = walkforward.MarketModels.fit

    def spy_fit(self, df):
        seen.append(df["game_date"].max())
        return real_fit(self, df)

    monkeypatch.setattr(walkforward.MarketModels, "fit", spy_fit)
    result = walkforward.run_walk_forward(feats, build_quotes(games, odds), verbose=False)
    assert result.retrain_dates
    assert set(result.bets["market"]) == {"h2h", "spreads", "totals"}
    for train_max, retrain_day in zip(seen, result.retrain_dates):
        assert train_max < retrain_day
