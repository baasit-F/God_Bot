"""ValuePicks Bot command line.

    python cli.py generate-mock            # synthetic data for offline testing
    python cli.py backtest                 # walk-forward backtest + report
    python cli.py daily-picks --days 2     # Top Line Picks for Sunday + Monday night
    python cli.py ingest-odds              # pull live odds (needs ODDS_API_KEY)
    python cli.py ingest-results --date 2026-10-19
    python cli.py export-parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from config import PROJECT_ROOT, settings
from data.storage import Storage
from models.engine import MARKET_LABELS


# ---------------------------------------------------------------------- formatting
def md_table(headers: list[str], rows: list[list[str]], align: list[str] | None = None) -> str:
    align = align or ["l"] * len(headers)
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)]

    def fmt(cells: list[str]) -> str:
        return "| " + " | ".join(c.rjust(w) if a == "r" else c.ljust(w) for c, w, a in zip(cells, widths, align)) + " |"

    sep = "|" + "|".join(("-" * (w + 1) + ":") if a == "r" else (":" + "-" * (w + 1)) for w, a in zip(widths, align)) + "|"
    return "\n".join([fmt(headers), sep, *(fmt(r) for r in rows)])


def _pct(x: float, digits: int = 1, signed: bool = False) -> str:
    return f"{x * 100:+.{digits}f}%" if signed else f"{x * 100:.{digits}f}%"


def _rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)


def _money(x: float) -> str:
    return f"${x:,.0f}"


# ------------------------------------------------------------------------ commands
def cmd_generate_mock(args: argparse.Namespace) -> None:
    from generate_mock_data import generate

    generate(Storage(), seasons=args.seasons, today=pd.Timestamp(args.today), seed=args.seed)


def cmd_backtest(args: argparse.Namespace) -> None:
    from backtest.metrics import by_period, by_tier, summarize
    from backtest.plots import plot_report
    from backtest.walkforward import model_vs_market, run_walk_forward
    from models.features import build_feature_frame, build_quotes

    store = Storage()
    games = store.games()
    if games.empty:
        sys.exit("No games in the database. Run `python cli.py generate-mock` or ingest data first.")

    print(f"Building leakage-free features for {len(games):,} games ...")
    odds = store.odds()
    feats = build_feature_frame(games, odds, store.sentiment(), store.team_stats())
    quotes = build_quotes(games, odds)
    print(f"Running walk-forward backtest (markets {', '.join(settings.markets)}; retrain every "
          f"{settings.retrain_every_days}d, decision at T-{settings.decision_hours_before:g}h, de-vig={settings.devig_method}) ...")
    result = run_walk_forward(
        feats,
        quotes,
        start=pd.Timestamp(args.start) if args.start else None,
        end=pd.Timestamp(args.end) if args.end else None,
        bankroll=args.bankroll,
        model_type=args.model,
        verbose=args.verbose,
    )
    s = summarize(result)
    if not s.get("bets"):
        print("No bets passed the filters. Try lowering VP_MIN_VOS / VP_MIN_EV.")
        return

    period = f"{result.daily['date'].min().date()} -> {result.daily['date'].max().date()}"
    print(f"\n## Backtest Summary  ({period})\n")
    rows = [
        ["Bets placed", f"{s['bets']:,}  over {s['betting_days']} days"],
        ["Win rate", _pct(s["win_rate"])],
        ["Average odds", f"{s['avg_odds']:.2f}"],
        ["Total staked", _money(s["total_staked"])],
        ["Net profit", _money(s["profit"])],
        ["Final bankroll", f"{_money(s['final_bankroll'])}  (start {_money(result.starting_bankroll)})"],
        ["ROI (on bankroll)", _pct(s["roi"], signed=True)],
        ["Yield (on turnover)", _pct(s["yield"], 2, signed=True)],
        ["Sharpe (annualised)", f"{s['sharpe']:.2f}"],
        ["Max drawdown", _pct(s["max_drawdown"])],
        ["Avg EV at bet time", _pct(s["avg_ev"], 2, signed=True)],
        ["Avg CLV vs sharp close", _pct(s["avg_clv"], 2, signed=True)],
        ["Avg stake", _pct(s["avg_stake_pct"], 2) + " of bankroll"],
    ]
    print(md_table(["Metric", "Value"], rows, ["l", "r"]))

    mvm = model_vs_market(result.predictions)
    print("\n## Out-of-sample accuracy: model vs sharp line (lower is better)\n")
    print(md_table(["Market", "Metric", "Model", "Sharp line", "Games"],
                   [[r["market"], r["metric"], f"{r['model']:.4f}", f"{r['sharp']:.4f}", f"{r['games']:,}"] for r in mvm],
                   ["l", "l", "r", "r", "r"]))

    markets = by_tier(result.bets, "market")
    print("\n## By market\n")
    print(md_table(["Market", "Bets", "Win %", "Profit", "Yield", "Avg CLV"],
                   [[MARKET_LABELS.get(str(r.Index), str(r.Index)), f"{r.bets:,}", _pct(r.win_rate), _money(r.profit),
                     _pct(r.yield_, 2, True), _pct(r.avg_clv, 2, True)]
                    for r in markets.rename(columns={"yield": "yield_"}).itertuples()], ["l", "r", "r", "r", "r", "r"]))

    tiers = by_tier(result.bets)
    print("\n## By confidence tier\n")
    print(md_table(["Tier", "Bets", "Win %", "Profit", "Yield"],
                   [[str(r.Index), f"{r.bets:,}", _pct(r.win_rate), _money(r.profit), _pct(r.yield_, 2, True)]
                    for r in tiers.rename(columns={"yield": "yield_"}).itertuples()], ["l", "r", "r", "r", "r"]))

    seasons = by_period(result.bets, "Y")
    print("\n## By year\n")
    print(md_table(["Year", "Bets", "Win %", "Profit", "Yield"],
                   [[str(r.Index), f"{r.bets:,}", _pct(r.win_rate), _money(r.profit), _pct(r.yield_, 2, True)]
                    for r in seasons.rename(columns={"yield": "yield_"}).itertuples()], ["l", "r", "r", "r", "r"]))

    out = settings.reports_dir
    out.mkdir(parents=True, exist_ok=True)
    result.bets.to_csv(out / "backtest_bets.csv", index=False)
    result.daily.to_csv(out / "backtest_daily.csv", index=False)
    (out / "backtest_summary.json").write_text(json.dumps(s, indent=2, default=float))
    if not args.no_plot:
        path = plot_report(result, out)
        print(f"\nCharts:  {_rel(path)}")
    print(f"Bet log: {_rel(out / 'backtest_bets.csv')}")


def cmd_daily_picks(args: argparse.Namespace) -> None:
    from models.daily import run_daily


    day = pd.Timestamp(args.date) if args.date else pd.Timestamp(date.today())
    bankroll = args.bankroll if args.bankroll is not None else settings.starting_bankroll
    result = run_daily(Storage(), day, bankroll, args.model, args.days)
    picks = result.picks

    span = f"{day:%a %d %b %Y}" + (f" - {day + pd.Timedelta(days=args.days - 1):%a %d %b %Y}" if args.days > 1 else "")
    print(f"\n# ValuePicks Bot - NFL Top Line Picks - {span}\n")
    opp = result.opportunities
    n_games = opp["game_id"].nunique() if not opp.empty else 0
    print(f"Models trained on {result.trained_on:,} completed games. Slate: {n_games} games, "
          f"{len(opp)} selections screened across {', '.join(MARKET_LABELS[m] for m in settings.markets)}.")
    print(f"Filters: VOS > {settings.min_vos}, EV > {_pct(settings.min_ev, 1, True)}, "
          f"odds {settings.min_odds:.2f}-{settings.max_odds:.2f}. "
          f"Sizing: {settings.kelly_fraction:g}x Kelly, cap {_pct(settings.max_stake_pct, 0)}/pick, "
          f"bankroll {_money(bankroll)}.\n")
    if picks.empty:
        print("No opportunities passed the filters on this slate. Sitting out is a position.")
        return

    for market in settings.markets:
        mp = picks[picks["market"] == market]
        print(f"## {MARKET_LABELS[market]}\n")
        if mp.empty:
            print("No picks.\n")
            continue
        rows = [
            [
                str(i + 1),
                p.fixture,
                f"{p.selection} ({p.book})",
                f"{p.odds:.2f}",
                _pct(p.p_model),
                _pct(p.p_market),
                _pct(p.ev, 1, True),
                f"{p.vos:.2f} ({p.tier})",
                f"{_money(p.stake)} ({_pct(p.stake_pct, 2)})",
            ]
            for i, p in enumerate(mp.itertuples())
        ]
        print(md_table(
            ["Rank", "Fixture", "Selection", "Odds", "Model Prob", "Devig Prob", "EV %", "VOS Score", "Recommended Stake"],
            rows, ["r", "l", "l", "r", "r", "r", "r", "r", "r"],
        ))
        print()
    print(f"Total exposure: {_money(picks['stake'].sum())} ({_pct(picks['stake_pct'].sum(), 2)} of bankroll) "
          f"across {len(picks)} picks. VOS tier in brackets (A=1.0, B=0.75, C=0.5).")
    if {"spreads", "totals"} & set(picks["market"]):
        print("Spread/total probabilities exclude pushes; EV and stakes account for them.")

    out = settings.reports_dir
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"picks_{day:%Y-%m-%d}.csv"
    picks.to_csv(path, index=False)
    print(f"Saved: {_rel(path)}")


def cmd_ingest_odds(args: argparse.Namespace) -> None:
    from data.connectors.odds_api import TheOddsAPIConnector
    from data.etl import ingest_live_odds

    g, o = ingest_live_odds(TheOddsAPIConnector(), Storage())
    print(f"Upserted {g} games and {o} odds rows.")


def cmd_ingest_results(args: argparse.Namespace) -> None:
    from data.connectors.stats import ESPNStatsConnector
    from data.etl import ingest_results

    day = date.fromisoformat(args.date) if args.date else date.today() - timedelta(days=1)
    n = ingest_results(ESPNStatsConnector(), Storage(), day)
    print(f"Marked {n} games final for {day}.")


def cmd_backfill(args: argparse.Namespace) -> None:
    from data.connectors.odds_api import TheOddsAPIConnector
    from data.connectors.stats import ESPNStatsConnector
    from data.etl import backfill_odds, backfill_results

    store = Storage()
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    to_dt = lambda d: datetime(d.year, d.month, d.day, tzinfo=timezone.utc)  # noqa: E731
    n = backfill_odds(TheOddsAPIConnector(), store, to_dt(start), to_dt(end), timedelta(hours=args.step_hours))
    print(f"Backfilled {n} odds rows.")
    print(f"Marked {backfill_results(ESPNStatsConnector(), store, start, end)} games final.")


def cmd_export_parquet(args: argparse.Namespace) -> None:
    for path in Storage().export_parquet():
        print(f"Wrote {path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="valuepicks", description="ValuePicks Bot - backtesting & daily picks engine")
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate-mock", help="generate synthetic data into the SQLite DB")
    g.add_argument("--seasons", type=int, default=5, help="completed seasons to simulate before the current one")
    g.add_argument("--today", default=date.today().isoformat(), help="date of the simulated upcoming slate")
    g.add_argument("--seed", type=int, default=settings.random_seed)
    g.set_defaults(func=cmd_generate_mock)

    b = sub.add_parser("backtest", help="run the walk-forward backtest")
    b.add_argument("--start", help="first betting date (YYYY-MM-DD)")
    b.add_argument("--end", help="last betting date (YYYY-MM-DD)")
    b.add_argument("--bankroll", type=float, default=None)
    b.add_argument("--model", choices=["logistic", "lightgbm"], default=None)
    b.add_argument("--no-plot", action="store_true")
    b.add_argument("--verbose", action="store_true", help="log each retrain")
    b.set_defaults(func=cmd_backtest)

    d = sub.add_parser("daily-picks", help="score a slate and print Top Line Picks")
    d.add_argument("--date", help="first slate date (default: today)")
    d.add_argument("--days", type=int, default=1, help="slate length in days, e.g. 2 for Sunday + Monday night")
    d.add_argument("--bankroll", type=float, default=None)
    d.add_argument("--model", choices=["logistic", "lightgbm"], default=None)
    d.set_defaults(func=cmd_daily_picks)

    sub.add_parser("ingest-odds", help="fetch current odds from The-Odds-API").set_defaults(func=cmd_ingest_odds)

    r = sub.add_parser("ingest-results", help="fetch final scores from ESPN")
    r.add_argument("--date", help="YYYY-MM-DD (default: yesterday)")
    r.set_defaults(func=cmd_ingest_results)

    bf = sub.add_parser("backfill", help="historical odds + results for a date range")
    bf.add_argument("--start", required=True)
    bf.add_argument("--end", required=True)
    bf.add_argument("--step-hours", type=float, default=6.0)
    bf.set_defaults(func=cmd_backfill)

    sub.add_parser("export-parquet", help="dump all tables to Parquet").set_defaults(func=cmd_export_parquet)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
