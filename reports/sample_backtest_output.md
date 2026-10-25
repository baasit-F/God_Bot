Building leakage-free features for 1,467 games ...
Running walk-forward backtest (markets h2h, spreads, totals; retrain every 7d, decision at T-3h, de-vig=shin) ...

## Backtest Summary  (2022-09-19 -> 2026-10-22)

| Metric                 |                    Value |
|:-----------------------|-------------------------:|
| Bets placed            |       719  over 194 days |
| Win rate               |                    52.6% |
| Average odds           |                     2.14 |
| Total staked           |                 $233,583 |
| Net profit             |                  $23,287 |
| Final bankroll         | $33,287  (start $10,000) |
| ROI (on bankroll)      |                  +232.9% |
| Yield (on turnover)    |                   +9.97% |
| Sharpe (annualised)    |                     1.30 |
| Max drawdown           |                   -26.9% |
| Avg EV at bet time     |                   +7.97% |
| Avg CLV vs sharp close |                   +4.42% |
| Avg stake              |        1.78% of bankroll |

## Out-of-sample accuracy: model vs sharp line (lower is better)

| Market    | Metric     |   Model | Sharp line | Games |
|:----------|:-----------|--------:|-----------:|------:|
| Moneyline | Brier      |  0.2347 |     0.2347 | 1,147 |
| Spread    | RMSE (pts) | 13.9098 |    13.9310 | 1,151 |
| Total     | RMSE (pts) |  9.9623 |     9.9899 | 1,151 |

## By market

| Market    | Bets | Win % |  Profit |   Yield | Avg CLV |
|:----------|-----:|------:|--------:|--------:|--------:|
| Moneyline |  142 | 41.8% |  $4,466 | +11.48% |  +5.33% |
| Spread    |  267 | 55.4% | $11,001 | +11.19% |  +1.89% |
| Total     |  310 | 55.3% |  $7,820 |  +8.11% |  +6.19% |

## By confidence tier

| Tier | Bets | Win % |  Profit |   Yield |
|:-----|-----:|------:|--------:|--------:|
| A    |  423 | 51.6% | $11,201 |  +8.65% |
| B    |  264 | 55.0% | $12,111 | +13.31% |
| C    |   32 | 46.7% |    $-25 |  -0.19% |

## By year

| Year | Bets | Win % | Profit |   Yield |
|:-----|-----:|------:|-------:|--------:|
| 2022 |  116 | 54.4% | $2,263 | +10.68% |
| 2023 |  190 | 53.2% | $3,934 |  +6.89% |
| 2024 |  175 | 50.0% | $3,669 |  +6.85% |
| 2025 |  164 | 51.2% | $6,363 |  +9.97% |
| 2026 |   74 | 57.7% | $7,058 | +18.65% |

Charts:  reports/backtest_report.png
Bet log: reports/backtest_bets.csv
