# ValuePicks Bot (NFL)

A modular Python framework for backtesting and weekly picks in **NFL moneyline** markets. It ingests
odds, stats and betting-split data, removes the bookmaker margin from sharp-book prices, models win
probability as adjustments to the sharp line, scores each opportunity with a **Value-to-Odds Score
(VOS)**, and sizes stakes with fractional Kelly. The walk-forward backtester never lets a prediction
see data from its own game day or later.

> Ships with a synthetic NFL data generator so you can run the whole pipeline before you plug in
> API keys. The bundled results come from that simulated market. They say nothing about real-world
> profitability.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python cli.py generate-mock                    # 5 seasons + the current week, 1,467 games
python cli.py backtest                         # walk-forward backtest -> reports/backtest_report.png
python cli.py daily-picks --days 2             # Top Line Picks for Sunday + Monday night
pytest                                         # devig/Kelly/VOS maths + data-leakage tests
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
    odds_api.py            The-Odds-API v4 (americanfootball_nfl: current, historical, scores)
    stats.py               ESPN NFL scoreboard (keyless) and SportsDataIO NFL
    sentiment.py           generic JSON public-splits feed (ticket % / handle %)
models/
  devig.py                 proportional, Shin and power de-vigging
  features.py              leakage-free market + team features
  model.py                 market-offset logistic regression (or LightGBM with init_score)
  scoring.py               EV, VOS, confidence tiers, Kelly sizing, pick filters
  daily.py                 slate prediction engine
backtest/
  walkforward.py           walk-forward loop, retrained weekly
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
| `sharp_logit` | odds | de-vigged sharp home probability at decision time (the model's offset) |
| `sharp_move` | odds | sharp-line movement from open to decision |
| `sharp_soft_gap` | odds | sharp probability minus the soft-book consensus |
| `money_ticket_gap` | splits | handle % minus ticket % (a sign of sharp money) |
| `elo_logit` | results | NFL Elo: K=20, 48 points of home field, MOV multiplier, ⅓ regression each off-season |
| `margin_diff`, `ypp_diff` | results / stats | last-5-game point margin and net yards per play |
| `rest_diff`, `*_off_bye`, `*_short_week` | schedule | bye weeks and Thursday short weeks |

**3. Model** (`models/model.py`). The model learns adjustments on top of the sharp line:

$$\text{logit}(p_{home}) = \text{logit}(P_{market}) + \beta_0 + \beta \cdot x$$

It is fitted with an L2 penalty (`VP_MODEL_L2`, default 100), so with little data it stays close to
the market instead of drifting toward 50%. This matters in the NFL, where 272 games a season is a
small sample. A plain regularised logistic regression pulls every probability toward a coin flip,
which makes every underdog look like value. `VP_MODEL=lightgbm` boosts from the same offset.
`VP_MODEL_WEIGHT` (default 1.0) adds further shrinkage toward the market.

**4. Scoring and filtering**

- $EV = P_{model} \times \text{Odds} - 1$, using the best price across `VP_BETTABLE_BOOKS`
- $VOS = \left(\frac{P_{model} - P_{market}}{P_{market}} \times 100\right) \times \log_{10}(\text{Odds}) \times \text{Tier}$
- Confidence tier A/B/C (×1.0 / 0.75 / 0.5) earns one point for each of: a sharp line exists, both
  teams have played at least 4 games this season, and the sharp line has not moved against the pick.
- A **Top Line Pick** needs VOS > 1.2, EV > +2.5% and odds between 1.30 and 6.00. Only the best side
  of each game qualifies, and picks are ranked by VOS.

**5. Sizing.** Stake = bankroll × min(0.25 × Kelly, 3%). If one day's picks add up to more than 15%
of the bankroll, every stake is scaled down to fit. Ties are refunded as pushes.

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
- `tests/test_core.py` checks all three properties: it flips future results, scrambles odds posted
  after the decision time, and spies on the training data.

## Sample output (synthetic data)

`python cli.py daily-picks --date 2026-10-25 --days 2`

```
| Rank | Fixture                                   | Selection                      | Odds | Model Prob | Devig Prob |  EV % | VOS Score | Recommended Stake |
|-----:|:------------------------------------------|:-------------------------------|-----:|-----------:|-----------:|------:|----------:|------------------:|
|    1 | New Orleans Saints @ Miami Dolphins       | Miami Dolphins (fanduel)       | 2.10 |      51.4% |      46.3% | +8.0% |  2.64 (B) |      $181 (1.81%) |
|    2 | Minnesota Vikings @ Cincinnati Bengals    | Cincinnati Bengals (fanduel)   | 2.79 |      36.8% |      34.5% | +2.6% |  2.16 (B) |       $36 (0.36%) |
|    3 | Philadelphia Eagles @ San Francisco 49ers | San Francisco 49ers (fanduel)  | 2.25 |      47.6% |      45.3% | +7.1% |  1.75 (A) |      $141 (1.41%) |
|    4 | Arizona Cardinals @ Las Vegas Raiders     | Arizona Cardinals (draftkings) | 3.77 |      27.4% |      26.7% | +3.3% |  1.59 (A) |       $29 (0.29%) |
```

Backtest on synthetic data, 2022 to 2026: 410 bets, yield +3.4%, average beat of the closing line
+3.9%, Sharpe 0.42, max drawdown −27.8%. The model's out-of-sample Brier score is 0.2246, against
0.2248 for the market. Full output is in [reports/sample_backtest_output.md](reports/sample_backtest_output.md)
and [reports/sample_daily_picks.md](reports/sample_daily_picks.md).

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
- The synthetic market's edge is deliberately large so the engine can find it. Bye weeks and road
  short weeks are worth 2.5 points and only 35% priced at decision time, and sharp money leans toward
  the close. Edges this large would be learnable from 5 seasons; real markets offer far smaller ones.
- The tier rules (A/B/C) do not yet predict results on the mock data. Tier B outperformed tier A.
  Re-derive them from real data.
- Only moneylines are modelled. Spreads and totals are the main NFL markets; `models/devig.py` and
  the scoring work unchanged, but the model needs a margin target.
- This is research software, not financial advice. Check the law where you live before betting.
