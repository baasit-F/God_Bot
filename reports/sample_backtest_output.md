Building leakage-free features for 1,467 games ...
Running walk-forward backtest (retrain every 7d, decision at T-3h, de-vig=shin) ...

## Backtest Summary  (2022-09-19 -> 2026-10-22)

| Metric                 |                    Value |
|:-----------------------|-------------------------:|
| Bets placed            |       410  over 143 days |
| Win rate               |                    47.4% |
| Average odds           |                     2.49 |
| Total staked           |                  $94,390 |
| Net profit             |                   $3,250 |
| Final bankroll         | $13,250  (start $10,000) |
| ROI (on bankroll)      |                   +32.5% |
| Yield (on turnover)    |                   +3.44% |
| Sharpe (annualised)    |                     0.42 |
| Max drawdown           |                   -27.8% |
| Avg EV at bet time     |                  +10.22% |
| Avg CLV vs sharp close |                   +3.86% |
| Avg stake              |        1.81% of bankroll |

## Out-of-sample calibration (all games, lower is better)

| Source |  Brier | Log loss |
|:-------|-------:|---------:|
| Model  | 0.2246 |   0.6391 |
| Market | 0.2248 |   0.6399 |

## By confidence tier

| Tier | Bets | Win % |  Profit |   Yield |
|:-----|-----:|------:|--------:|--------:|
| A    |  220 | 43.0% | $-3,595 |  -7.57% |
| B    |  161 | 53.4% |  $6,405 | +16.42% |
| C    |   29 | 48.3% |    $440 |  +5.58% |

## By year

| Year | Bets | Win % |  Profit |   Yield |
|:-----|-----:|------:|--------:|--------:|
| 2022 |   58 | 48.3% |   $-535 |  -5.85% |
| 2023 |  100 | 41.0% | $-1,433 |  -8.72% |
| 2024 |  108 | 58.8% |  $6,642 | +27.28% |
| 2025 |   99 | 47.5% |  $2,670 |  +8.78% |
| 2026 |   45 | 33.3% | $-4,093 | -29.17% |

Charts:  reports/backtest_report.png
Bet log: reports/backtest_bets.csv
