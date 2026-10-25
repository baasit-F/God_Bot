# ValuePicks Bot (NFL)

A modular Python framework for backtesting and weekly picks in **NFL moneyline, spread and total**
markets. It ingests odds, stats and betting-split data, removes the bookmaker margin from sharp-book
prices in each market, models win probability, final margin and final total as adjustments to the
sharp line, scores each opportunity with a **Value-to-Odds Score (VOS)**, and sizes stakes with
fractional Kelly. The walk-forward backtester never lets a prediction
see data from its own game day or later.

> Ships with a synthetic NFL data generator so you can run the whole pipeline before you plug in
> API keys. The bundled results come from that simulated market. They say nothing about real-world
> profitability.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python cli.py generate-mock                    # 5 seasons + the current week, 1,467 games, 3 markets
python cli.py backtest                         # walk-forward backtest -> reports/backtest_report.png
python cli.py daily-picks --days 2             # Top Line Picks per market for Sunday + Monday night
pytest                                         # devig/Kelly/VOS/push maths + data-leakage tests
```

`daily-picks` scores the slate starting at `--date` (default today) and covering `--days` days.
Run it on Thursday with `--days 1` for Thursday night, then on Sunday with `--days 2`.

## Project layout

```
config.py                  every threshold, book list, model and Elo setting (env / .env overridable)
cli.py                     generate-mock | backtest | daily-picks | ingest-odds | ingest-results | backfill | export-parquet
generate_mock_data.py      synthetic NFL league with a known ground truth
data/
  storage.py               SQLite schema (games, odds, sentiment, team_stats) + Parquet export
  etl.py                   live ingest, results join, historical backfill
  connectors/
    base.py                connector interfaces + HTTP retry/backoff
    odds_api.py            The-Odds-API v4 (americanfootball_nfl h2h/spreads/totals: current, historical, scores)
    stats.py               ESPN NFL scoreboard (keyless) and SportsDataIO NFL
    sentiment.py           generic JSON public-splits feed (ticket % / handle %)
models/
  devig.py                 proportional, Shin and power de-vigging
  lines.py                 spread/total probabilities: implied means, push mass, cover odds
  features.py              leakage-free market features (all 3 markets) + team features
  model.py                 moneyline: market-offset logistic regression (or LightGBM)
  line_model.py            spreads/totals: market-offset ridge on margin and total
  engine.py                trains and scores every enabled market
  scoring.py               push-aware EV, VOS, confidence tiers, Kelly sizing, pick filters
  daily.py                 slate prediction engine
backtest/
  walkforward.py           walk-forward loop, retrained weekly
  metrics.py               ROI, yield, win rate, Sharpe, max drawdown, CLV
  plots.py                 equity curve, drawdown, rolling and monthly yield
tests/test_core.py
```

## How it works

**1. De-vigging, per market.** Each two-way market at the sharp book (Pinnacle by default) is
de-vigged on its own: home/away for the moneyline and the spread, over/under for the total. Shin's
method is the default. It assigns more of the margin to longshots, which corrects the
favourite-longshot bias. Set `VP_DEVIG_METHOD=proportional` to use the multiplicative method instead.

- **Moneyline:** the de-vigged home probability is $P_{market}$.
- **Spreads and totals:** books hang different numbers, so a de-vigged price is only meaningful at
  its own line. `models/lines.py` treats the final margin (or total) as an integer from a
  discretised Normal with sd 13.5 (or 10) and solves for the mean that makes the sharp line's
  de-vigged price fair. That implied mean, `spread_mu` or `total_mu`, can then price any book's line:
  $P_{market}$ at FanDuel's −3 comes from Pinnacle's −3.5. On a whole-number line, the probability
  of landing exactly on it is a push.

**2. Features** (`models/features.py`)

| Feature | Source | Meaning |
|---|---|---|
| `sharp_logit` | odds | de-vigged sharp home probability at decision time (the model's offset) |
| `sharp_move` | odds | sharp-line movement from open to decision |
| `sharp_soft_gap` | odds | sharp probability minus the soft-book consensus |
| `money_ticket_gap` | splits | handle % minus ticket % (a sign of sharp money) |
| `elo_logit` | results | NFL Elo: K=20, 48 points of home field, MOV multiplier, ⅓ regression each off-season |
| `margin_diff`, `ypp_diff` | results / stats | last-5-game point margin and net yards per play |
| `rest_diff`, `*_off_bye`, `*_short_week` | schedule | bye weeks and Thursday short weeks |

The spread model uses `spread_move`, `elo_spread_gap` (Elo-implied margin minus the market's), form,
rest and sharp-money features. The total model uses `total_move`, `total_form_gap` (recent game
totals minus the market's) and `plays_form` (pace).

**3. Model** (`models/model.py`). The model learns adjustments on top of the sharp line:

$$\text{logit}(p_{home}) = \text{logit}(P_{market}) + \beta_0 + \beta \cdot x$$

It is fitted with an L2 penalty (`VP_MODEL_L2`, default 100), so with little data it stays close to
the market instead of drifting toward 50%. This matters in the NFL, where 272 games a season is a
small sample. A plain regularised logistic regression pulls every probability toward a coin flip,
which makes every underdog look like value. `VP_MODEL=lightgbm` boosts from the same offset.
`VP_MODEL_WEIGHT` (default 1.0) adds further shrinkage toward the market.

Spreads and totals use the same idea in points (`models/line_model.py`):
$E[\text{margin}] = \text{spread\_mu} + \beta \cdot x$, fitted as a ridge regression on the
residual. The penalties are `VP_SPREAD_MODEL_L2=100` and `VP_TOTAL_MODEL_L2=500`, chosen on the 2024
mock season.

**Guardrail.** No model may disagree with the sharp line by more than 0.30 log-odds (moneyline) or
2.5 points (spread and total), set by `VP_MAX_*_DEV`. Without the cap, an early-season model trained
on about 300 games put one spread 6 points off the market, because two rare short-week flags
lined up.

**4. Scoring and filtering**

- $EV = P_{win}(\text{Odds} - 1) - P_{lose}$. A push returns the stake, so with no push this is
  $P_{model} \times \text{Odds} - 1$. The moneyline uses the best price across `VP_BETTABLE_BOOKS`.
  For spreads and totals every book's line and price is priced under the model, and the best EV wins.
- $VOS = \left(\frac{P_{model} - P_{market}}{P_{market}} \times 100\right) \times \log_{10}(\text{Odds}) \times \text{Tier}$
- Confidence tier A/B/C (×1.0 / 0.75 / 0.5) earns one point for each of: a sharp line exists, both
  teams have played at least 4 games this season, and the sharp line has not moved against the pick.
- For spreads and totals, VOS compares win probabilities that exclude pushes. EV and Kelly do
  account for pushes.
- A **Top Line Pick** needs VOS > 1.2, EV > +2.5% and odds between 1.30 and 6.00. Each market keeps
  at most one side per game, and picks are ranked by VOS within each market.
- A moneyline and a spread on the same game are highly correlated, so only the higher-VOS of the two
  is kept (`VP_ALLOW_CORRELATED=false`). Totals are independent.

**5. Sizing.** Stake = bankroll × min(0.25 × Kelly, 3%), where Kelly with pushes is
$(b\,P_{win} - P_{lose}) / b$. If one day's picks across all markets add up to more than 15% of the
bankroll, every stake is scaled down to fit. Moneyline ties and landing on a whole-number line are
refunded as pushes.

### VOS scaling

The relative-edge term is multiplied by 100 (`VP_VOS_EDGE_SCALE`). Without that, a 10% relative
edge at odds of 2.00 scores 0.10 × 0.301 = 0.03, and VOS > 1.2 would need a relative edge of about
400%. With the scale, a 4% relative edge at 2.00 with tier A scores exactly 1.2. Set
`VP_VOS_EDGE_SCALE=1` to apply the formula to raw fractions.

## Zero data leakage

- **Team features** come from a single pass over the calendar. All of day D's games are built from
  state as it stood before D, so the 4pm and Sunday-night games never see the 1pm results.
- **Market features** use the latest snapshot at or before `kickoff − VP_DECISION_HOURS` (3h).
  Closing prices sit in `close_*` columns, which are used only to report CLV.
- **Walk-forward:** the model is retrained weekly on final games dated strictly before the betting
  day.
- `tests/test_core.py` checks all three properties. It flips future results, scrambles every price
  posted after the decision time and moves every line by 7 points, and spies on the training data.

## Sample output (synthetic data)

`python cli.py daily-picks --date 2026-10-25 --days 2` prints one table per market. The spread
table, for example:

```
| Rank | Fixture                               | Selection                           | Odds | Model Prob | Devig Prob |  EV % | VOS Score | Recommended Stake |
|-----:|:--------------------------------------|:------------------------------------|-----:|-----------:|-----------:|------:|----------:|------------------:|
|    1 | New England Patriots @ Chicago Bears  | New England Patriots +1.5 (caesars) | 2.04 |      51.8% |      47.0% | +5.8% |  2.37 (B) |      $139 (1.39%) |
|    2 | Miami Dolphins @ Minnesota Vikings    | Miami Dolphins +5.5 (betmgm)        | 2.05 |      52.0% |      47.6% | +6.5% |  2.16 (B) |      $156 (1.56%) |
|    3 | Buffalo Bills @ Indianapolis Colts    | Buffalo Bills -3.5 (fanduel)        | 2.00 |      52.8% |      48.4% | +5.7% |  2.07 (B) |      $142 (1.42%) |
```

Backtest on synthetic data, 2022 to 2026: 719 bets, yield +10.0%, average beat of the closing line
+4.4%, Sharpe 1.30, max drawdown −26.9%.

| Market | Bets | Yield | Avg CLV | Out of sample: model vs sharp line |
|---|--:|--:|--:|---|
| Moneyline | 142 | +11.5% | +5.3% | Brier 0.2347 vs 0.2347 |
| Spread | 267 | +11.2% | +1.9% | RMSE 13.91 vs 13.93 pts |
| Total | 310 | +8.1% | +6.2% | RMSE 9.96 vs 9.99 pts |

Full output is in [reports/sample_backtest_output.md](reports/sample_backtest_output.md) and
[reports/sample_daily_picks.md](reports/sample_daily_picks.md).

## Going live

1. `cp .env.example .env` and set `ODDS_API_KEY`.
2. Build history: `python cli.py backfill --start 2024-09-05 --end 2025-01-06 --step-hours 12`.
   This needs a historical plan on The-Odds-API, and each snapshot costs credits.
3. Each game week:
   - Run `python cli.py ingest-odds` at several points (Tuesday open, Sunday morning, about 3h
     before kickoff) so line movement is recorded.
   - After each game day, run `python cli.py ingest-results --date YYYY-MM-DD`.
   - Run `python cli.py daily-picks --days 2` on Sunday morning, and `--days 1` on Thursday.
4. Re-run `python cli.py backtest` on the real data before trusting any threshold.

To swap in a different provider, subclass `OddsConnector`, `StatsConnector` or `SentimentConnector`
in `data/connectors/` and return the same columns. Nothing downstream changes.

## Caveats

- **The NFL sample is small.** At 272 games a season, the per-season yield in the backtest swings
  between about −30% and +27% even though the edge is real and planted. Judge a model on closing-line
  value and Brier score against the market, not on one season's profit.
- The synthetic market's edges are deliberately large so the engine can find them:
  - bye weeks and road short weeks are worth 2.5 points and only 35% priced at decision time;
  - team pace is only 40% priced into totals;
  - sharp money leans toward the close;
  - soft books shade popular teams and overs.

  Edges this large would be learnable from 5 seasons; real markets offer far smaller ones.
- The tier rules (A/B/C) do not yet predict results on the mock data. Tier B outperformed tier A.
  Re-derive them from real data.
- Margins and totals are modelled as smooth discretised Normals. Real NFL margins pile up on 3 and 7,
  so pushes and half-point value around key numbers are underestimated. An empirical margin
  distribution is the natural next step.
- This is research software, not financial advice. Check the law where you live before betting.
