Building leakage-free features for 4,267 games ...
Running walk-forward backtest (retrain every 14d, decision at T-3h, de-vig=shin) ...

## Backtest Summary  (2023-12-10 -> 2026-10-24)

| Metric                 |                    Value |
|:-----------------------|-------------------------:|
| Bets placed            |     1,255  over 432 days |
| Win rate               |                    36.1% |
| Average odds           |                     3.11 |
| Total staked           |                 $157,994 |
| Net profit             |                   $7,331 |
| Final bankroll         | $17,331  (start $10,000) |
| ROI (on bankroll)      |                   +73.3% |
| Yield (on turnover)    |                   +4.64% |
| Sharpe (annualised)    |                     0.72 |
| Max drawdown           |                   -44.7% |
| Avg EV at bet time     |                   +8.39% |
| Avg CLV vs sharp close |                   +3.11% |
| Avg stake              |        1.05% of bankroll |

## Out-of-sample calibration (all games, lower is better)

| Source |  Brier | Log loss |
|:-------|-------:|---------:|
| Model  | 0.2296 |   0.6508 |
| Market | 0.2291 |   0.6496 |

## By confidence tier

| Tier | Bets | Win % | Profit |   Yield |
|:-----|-----:|------:|-------:|--------:|
| A    |  667 | 36.6% |  $-904 |  -1.05% |
| B    |  543 | 35.7% | $8,068 | +12.16% |
| C    |   45 | 33.3% |   $166 |  +3.05% |

## By year

| Year | Bets | Win % | Profit |   Yield |
|:-----|-----:|------:|-------:|--------:|
| 2023 |   89 | 40.4% | $1,806 | +12.23% |
| 2024 |  551 | 34.8% |  $-175 |  -0.31% |
| 2025 |  416 | 36.1% |   $970 |  +1.69% |
| 2026 |  199 | 37.7% | $4,731 | +16.58% |

Charts:  reports/backtest_report.png
Bet log: reports/backtest_bets.csv
