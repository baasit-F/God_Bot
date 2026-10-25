# ValuePicks Bot

A modular Python framework for backtesting and making daily picks in sports betting markets. It
ingests odds, stats and betting-split data, removes the bookmaker margin from sharp-book prices,
models win probability, scores each daily opportunity with a **Value-to-Odds Score (VOS)**, and sizes
stakes with fractional Kelly. The walk-forward backtester never lets a prediction see data from its
own match day or later.

> Ships with a synthetic data generator so you can run the whole pipeline before you plug in API keys.
> The bundled results come from that simulated market. They say nothing about real-world profitability.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python cli.py generate-mock          # ~4,300 synthetic games, 127k odds snapshots, today's slate
python cli.py backtest               # walk-forward backtest -> reports/backtest_report.png
python cli.py daily-picks            # today's Top Line Picks
pytest                               # devig/Kelly/VOS maths + data-leakage tests
```

## Project layout

```
config.py                  every threshold, book list and API key (env / .env overridable)
cli.py                     generate-mock | backtest | daily-picks | ingest-odds | ingest-results | backfill | export-parquet
generate_mock_data.py      synthetic league with a known ground truth
data/
  storage.py               SQLite schema (games, odds, sentiment, team_stats) + Parquet export
  etl.py                   live ingest, results join, historical backfill
  connectors/
    base.py                connector interfaces + HTTP retry/backoff
    odds_api.py            The-Odds-API v4 (current, historical, scores)
    stats.py               ESPN scoreboard (keyless) and SportsDataIO
    sentiment.py           generic JSON public-splits feed (ticket % / handle %)
models/
  devig.py                 proportional, Shin and power de-vigging
  features.py              leakage-free market + team features
  model.py                 calibrated logistic regression (or LightGBM)
  scoring.py               EV, VOS, confidence tiers, Kelly sizing, pick filters
  daily.py                 daily prediction engine
backtest/
  walkforward.py           walk-forward loop with periodic retraining
  metrics.py               ROI, yield, win rate, Sharpe, max drawdown, CLV
  plots.py                 equity curve, drawdown, rolling and monthly yield
tests/test_core.py
```

## How it works

**1. De-vigging.** The sharp book (Pinnacle by default) is de-vigged with Shin's method. Shin
assigns more of the margin to longshots, which corrects the favourite-longshot bias. The result is
$P_{market}$. Set `VP_DEVIG_METHOD=proportional` to use the multiplicative method instead.

**2. Features** (`models/features.py`)

| Feature | Source | Meaning |
|---|---|---|
| `sharp_logit` | odds | de-vigged sharp home probability at decision time |
| `sharp_move` | odds | sharp-line movement from open to decision |
| `sharp_soft_gap` | odds | sharp probability minus the soft-book consensus |
| `money_ticket_gap` | splits | handle % minus ticket % (a sign of sharp money) |
| `elo_logit` | results | margin-of-victory Elo with home advantage |
| `margin_diff_10`, `net_rating_diff_10` | results / stats | 10-game form |
| `rest_diff`, `home_b2b`, `away_b2b` | schedule | fatigue |

**3. Model.** A standardised logistic regression, sigmoid-calibrated with `CalibratedClassifierCV`.
The market probability is one of its inputs, so the model learns where to deviate from the line.
The prediction is then shrunk toward the market in logit space with
$p = m + w\,(p_{model} - m)$ (`VP_MODEL_WEIGHT`, default 0.5). This counters the winner's curse of
betting only where the model disagrees most.

**4. Scoring and filtering**

- $EV = P_{model} \times \text{Odds} - 1$, using the best price across `VP_BETTABLE_BOOKS`
- $VOS = \left(\frac{P_{model} - P_{market}}{P_{market}} \times 100\right) \times \log_{10}(\text{Odds}) \times \text{Tier}$
- Confidence tier A/B/C (×1.0 / 0.75 / 0.5) earns one point for each of: a sharp line exists, both
  teams have played at least 10 games this season, and the sharp line has not moved against the pick.
- A **Top Line Pick** needs VOS > 1.2, EV > +2.5% and odds between 1.30 and 6.00. Only the best side
  of each game qualifies, and picks are ranked by VOS.

**5. Sizing.** Stake = bankroll × min(0.25 × Kelly, 3%). If the day's picks add up to more than 15%
of the bankroll, every stake is scaled down to fit.

### VOS scaling

The relative-edge term is multiplied by 100 (`VP_VOS_EDGE_SCALE`). Without that, a 10% relative
edge at odds of 2.00 scores 0.10 × 0.301 = 0.03, and VOS > 1.2 would need a relative edge of about
400%. With the scale, a 4% relative edge at 2.00 with tier A scores exactly 1.2. Set
`VP_VOS_EDGE_SCALE=1` to apply the formula to raw fractions.

## Zero data leakage

- **Team features** come from a single pass over the calendar. All of day D's games are built from
  state as it stood before D, and the state takes in D's results only afterwards.
- **Market features** use the latest snapshot at or before `commence − VP_DECISION_HOURS` (3h).
  Closing prices sit in `close_*` columns, which are used only to report CLV.
- **Walk-forward:** the model is retrained every 14 days on final games dated strictly before the
  betting day.
- `tests/test_core.py` checks all three properties: it flips future results, scrambles odds posted
  after the decision time, and spies on the training data.

## Sample output (synthetic data)

`python cli.py daily-picks --date 2026-10-25`

```
| Rank | Fixture                          | Selection                   | Odds | Model Prob | Devig Prob |   EV % | VOS Score | Recommended Stake |
|-----:|:---------------------------------|:----------------------------|-----:|-----------:|-----------:|-------:|----------:|------------------:|
|    1 | Milwaukee Bucks @ Denver Nuggets | Denver Nuggets (draftkings) | 5.04 |      21.9% |      20.5% | +10.5% |  2.47 (C) |       $65 (0.65%) |
|    2 | Dallas Mavericks @ Phoenix Suns  | Phoenix Suns (draftkings)   | 2.76 |      39.9% |      36.3% | +10.2% |  2.22 (C) |      $144 (1.44%) |
```

Full output: [reports/sample_daily_picks.md](reports/sample_daily_picks.md) and
[reports/sample_backtest_output.md](reports/sample_backtest_output.md). Charts are in
`reports/backtest_report.png`.

## Going live

1. `cp .env.example .env` and set `ODDS_API_KEY`.
2. Build history: `python cli.py backfill --start 2024-10-22 --end 2025-04-13`. This needs a
   historical plan on The-Odds-API, and each snapshot costs credits, so use `--step-hours` to choose
   how often prices are sampled.
3. Every day: run `python cli.py ingest-odds` several times (open → decision) so line movement is
   recorded, run `python cli.py ingest-results` for yesterday's scores, then run
   `python cli.py daily-picks`.
4. Re-run `python cli.py backtest` on the real data before trusting any threshold.

To swap in a different provider, subclass `OddsConnector`, `StatsConnector` or `SentimentConnector`
in `data/connectors/` and return the same columns. Nothing downstream changes.

## Caveats

- The synthetic market has deliberate inefficiencies baked in, so the bot has something to find:
  back-to-backs are under-priced and sharp money leans toward the closing line. Real markets are
  much harder.
- Choose `VP_MODEL_WEIGHT` and the thresholds on a validation period, not on the period you report.
- This is research software, not financial advice. Check the law where you live before betting.
