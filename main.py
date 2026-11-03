#!/usr/bin/env python3
"""TrendWarden - command line entry point.

    python main.py backtest            # test the strategy on historical data
    python main.py optimise            # parameter search with an unseen test period
    python main.py discover            # test every strategy on unseen data, pick the most robust
    python main.py paper               # trade live prices with simulated money
    python main.py live --confirm-live # trade real money (read the README first)
    python main.py status              # show saved paper/live state
    python main.py reset-halt --mode paper
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from bot.config import load_config

log = logging.getLogger("trendwarden")


def setup_logging(verbose: bool, logfile: str | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if logfile:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(logfile))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        handlers=handlers,
    )


def load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


def get_data(cfg, args) -> dict[str, pd.DataFrame]:
    from bot.data import coverage, fetch_history, load_csv, make_market, synthetic_ohlcv

    if args.synthetic:
        log.info("Using SYNTHETIC data (for demo/testing only)")
        return {s: synthetic_ohlcv(n=4000, seed=i + 1, freq=cfg.timeframe) for i, s in enumerate(cfg.symbols)}
    if args.csv:
        if len(cfg.symbols) != 1:
            raise SystemExit("--csv works with a single symbol; set one symbol in the config")
        return {cfg.symbols[0]: load_csv(args.csv)}
    ex = make_market(cfg)
    data = {}
    for s in cfg.symbols:
        log.info("Downloading %d days of %s %s candles from %s...", cfg.history_days, s, cfg.timeframe,
                 "MetaTrader 5" if cfg.platform == "mt5" else cfg.exchange)
        data[s] = fetch_history(ex, s, cfg.timeframe, cfg.history_days)
        log.info("  %d candles (%s -> %s)", len(data[s]), data[s].index[0], data[s].index[-1])
        cov = coverage(data[s], cfg.timeframe)
        if cov < 0.8:
            log.warning("  %s history has big GAPS: only %.0f%% of the expected candles are there. Results will be "
                        "unreliable. In MT5: Tools > Options > Charts > Max bars in chart = Unlimited, restart, "
                        "open a %s %s chart and scroll back (Home key) to load full history, then run again.",
                        s, cov * 100, s, cfg.timeframe.upper())
            if hasattr(ex, "gold_symbols"):
                others = [g for g in ex.gold_symbols() if g != s]
                if others:
                    log.warning("  Other gold symbols on this account: %s. Some trade only part of the day. "
                                "Try one, e.g. `symbols: [%s]` in config.yaml.", ", ".join(others[:8]),
                                "XAUUSD" if "XAUUSD" in others else others[0])
    return data


def sim_kwargs(cfg) -> dict:
    x = cfg.execution
    kw = dict(starting_cash=x.starting_cash, fee_rate=x.fee_rate, slippage_rate=x.slippage_rate,
              leverage=x.leverage, min_qty=x.min_qty, qty_step=x.qty_step)
    ng = cfg.intel["news_guard"]
    if ng.enabled and ng.in_backtest:
        from bot.intel.news_guard import NewsGuard
        kw["entry_filter"] = NewsGuard(ng).backtest_filter(cfg.intel["data_timezone"])
    return kw


def cmd_backtest(cfg, args) -> None:
    from bot.backtest import run_backtest

    data = get_data(cfg, args)
    res = run_backtest(data, cfg.strategy_name, cfg.strategy_params, cfg.risk, cfg.timeframe,
                       **sim_kwargs(cfg))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res.equity.to_csv(out / "equity.csv")
    res.trades.to_csv(out / "trades.csv", index=False)

    print("\n=== TrendWarden backtest:", cfg.strategy_name, "|", ", ".join(cfg.symbols), cfg.timeframe, "===")
    for k, v in res.metrics.items():
        print(f"  {k:<20} {v}")
    print("  --- equal-weight buy & hold over the same period ---")
    for k, v in res.benchmark.items():
        print(f"  {k:<20} {v}")
    print(f"\nEquity curve and trade list saved to {out}/")
    if args.plot:
        plot(res, out / "equity.png")


def plot(res, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install matplotlib for --plot")
        return
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    res.equity.plot(ax=a1, color="#1f6feb")
    a1.set_title("Equity")
    a1.grid(alpha=0.3)
    ((res.equity / res.equity.cummax() - 1) * 100).plot(ax=a2, color="#d1242f")
    a2.set_title("Drawdown %")
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"Chart saved to {path}")


DEFAULT_GRIDS = {
    "trend_momentum": {"fast_ema": [12, 21, 34], "slow_ema": [55, 89], "adx_min": [15, 20, 25]},
    "mean_reversion": {"bb_period": [20, 30], "bb_std": [2.0, 2.5], "rsi_entry": [25, 30, 35]},
    "breakout": {"entry_period": [20, 40, 55], "exit_period": [10, 20]},
    "supertrend": {"st_period": [7, 10, 14], "st_mult": [2.0, 3.0]},
    "macd_trend": {"fast": [8, 12], "slow": [21, 26], "below_zero_only": [True, False]},
    "trend_pullback": {"fast_ema": [20, 50], "rsi_low": [35, 40, 45]},
    "squeeze_breakout": {"kc_mult": [1.25, 1.5], "lookback": [4, 8]},
    "ensemble": {"min_votes": [2, 3]},
}


def cmd_discover(cfg, args) -> None:
    import yaml

    from bot.backtest import discover

    names = args.strategies.split(",") if args.strategies else list(DEFAULT_GRIDS)
    unknown = [n for n in names if n not in DEFAULT_GRIDS]
    if unknown:
        raise SystemExit(f"Unknown strategies: {unknown}. Choose from {list(DEFAULT_GRIDS)}")
    grids = {n: DEFAULT_GRIDS[n] for n in names}
    data = get_data(cfg, args)
    print(f"\nWalk-forward testing {len(grids)} strategies x {args.folds} unseen windows "
          f"on {', '.join(cfg.symbols)} {cfg.timeframe}. This can take a few minutes...")
    results = discover(data, grids, cfg.risk, cfg.timeframe, folds=args.folds, min_trades=args.min_trades,
                       progress=lambda n: print(f"  testing {n}...", flush=True), **sim_kwargs(cfg))

    rows = [{"strategy": r.strategy, "robust": "YES" if r.robust else "no", **r.summary,
             "why_not": "; ".join(r.reasons)} for r in results]
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    print("\n=== Strategy leaderboard (results on data each strategy was NOT tuned on) ===\n")
    print(table.to_string(index=False))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "discover.csv", index=False)
    best = results[0]
    if args.synthetic:
        print("\n(Synthetic data: this only shows the tool works. Run without --synthetic on real prices.)")
        return
    if not best.robust:
        other = ", ".join(t for t in ("15m", "1h", "4h") if t != cfg.timeframe)
        print("\nNo strategy passed the robustness checks on this market/timeframe. That is useful to know:"
              f"\n- don't trade it live; try another timeframe ({other}) or more history,"
              "\n- or keep paper trading and re-run discover later.")
        return
    raw = yaml.safe_load(Path(args.config).read_text()) or {}
    raw["strategy"] = {"name": best.strategy, "params": best.params}
    raw["live_enabled"] = False
    (out / "best_config.yaml").write_text(
        "# Generated by `python main.py discover`. Review, then paper trade it before going live.\n"
        + yaml.safe_dump(raw, sort_keys=False))
    print(f"\nBest robust strategy: {best.strategy} {best.params}")
    print(f"Saved a ready-to-use config to {out / 'best_config.yaml'} (live trading left OFF).")
    print("Next: copy it over config.yaml and run `python main.py paper` for a few weeks.")


def cmd_optimise(cfg, args) -> None:
    from bot.backtest import optimise

    grid = json.loads(args.grid) if args.grid else DEFAULT_GRIDS[cfg.strategy_name]
    data = get_data(cfg, args)
    report = optimise(data, cfg.strategy_name, grid, cfg.strategy_params, cfg.risk, cfg.timeframe,
                      train_frac=args.train, min_trades=args.min_trades,
                      **sim_kwargs(cfg))
    if report.empty:
        print("No parameter set produced enough trades. Try more history or a lower --min-trades.")
        return
    pd.set_option("display.width", 200)
    print("\nTop configurations (IS = in-sample/training, OOS = out-of-sample/unseen):\n")
    print(report.to_string(index=False))
    print("\nPrefer configs that stay profitable OOS with similar Sharpe - big IS/OOS gaps mean overfitting.")


def cmd_intel(cfg, args) -> None:
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    if args.what == "news":
        from bot.intel.news_guard import NewsGuard

        ng = cfg.intel["news_guard"]
        guard = NewsGuard(ng)
        print(f"News guard is {'ON' if ng.enabled else 'OFF'} in your config "
              f"(pause {ng.minutes_before} min before to {ng.minutes_after} min after each event).\n")
        print("Upcoming scheduled events (your local time):")
        for t, name in guard.events_between(now, now + timedelta(days=args.days)):
            print(f"  {t.astimezone():%a %d %b %H:%M}  {name}")
        guard.poll_headlines(now)
        hb = guard.headline_block(now)
        print(f"\nBreaking-headline pause right now: {hb or 'none'}")
        print(f"Scheduled pause right now: {guard.scheduled_block(now) or 'none'}")
    elif args.what == "smart-money":
        from bot.intel.smart_money import SmartMoney, render

        sm = SmartMoney(cfg.intel["smart_money"], cfg.state_dir)
        report = sm.refresh(force=args.refresh or sm.stale())
        print(render(report) if report else "No report yet (see the warning above).")
    elif args.what == "congress":
        from dataclasses import replace

        from bot.intel.congress import CongressWatch

        watch = CongressWatch(replace(cfg.intel["congress"], enabled=True), cfg.state_dir)
        found = watch.check(now)
        print(f"\n{len(found)} gold-related Congress trade(s) in new filings." if found else
              "\nNo new gold-related Congress trades.")


def cmd_copy(cfg, args) -> None:
    from bot import copytrade as ct
    from bot.alpaca import AlpacaError

    try:
        if args.action == "plan":
            ct.plan(args.investor, args.top, args.budget, use_alpaca=not args.no_alpaca)
        elif args.action == "rebalance":
            ct.rebalance(args.sell_untracked)
        elif args.action == "execute":
            placed = ct.execute()
            print(f"\n{len(placed)} order(s) placed.")
        elif args.action == "status":
            from bot.alpaca import Alpaca

            info = ct.sync_from_alpaca(Alpaca())
            print(f"Alpaca: equity ${info['equity']:,.2f}, cash ${info['cash']:,.2f}, {info['positions']} positions "
                  "(saved to trader/portfolio.json)")
    except (ct.CopyError, AlpacaError) as exc:
        raise SystemExit(str(exc)) from None


def cmd_run(cfg, args, mode: str) -> None:
    from bot.engine import Engine

    if mode == "live":
        if not cfg.live_enabled:
            raise SystemExit("Live trading is disabled. Set `live_enabled: true` in your config first.")
        if not args.confirm_live:
            raise SystemExit("Refusing to trade real money without --confirm-live.")
        venue = "MetaTrader 5" if cfg.platform == "mt5" else cfg.exchange
        log.warning("LIVE TRADING on %s%s", venue, " (TESTNET)" if cfg.testnet else "")
    Engine(cfg, mode).run()


def cmd_status(cfg, args) -> None:
    for mode in ("paper", "live"):
        p = Path(cfg.state_dir) / f"{mode}_state.json"
        if not p.exists():
            continue
        s = json.loads(p.read_text())
        print(f"\n[{mode}] saved {s.get('saved_at')}")
        if "broker" in s:
            print(f"  cash: {s['broker']['cash']:.2f}")
        r = s["risk"]
        print(f"  peak equity: {r['peak_equity']:.2f} | halted: {r['halted']} {r.get('halt_reason', '')}")
        for sym, pos in s["positions"].items():
            print(f"  {sym}: qty {pos['qty']:.6g} entry {pos['entry_price']:.6g} stop {pos['stop']:.6g} "
                  f"target {pos['take_profit']}")
        t = Path(cfg.state_dir) / f"{mode}_trades.csv"
        if t.exists():
            trades = pd.read_csv(t)
            print(f"  closed trades: {len(trades)} | total PnL {trades.pnl.sum():+.2f}")


def cmd_reset_halt(cfg, args) -> None:
    p = Path(cfg.state_dir) / f"{args.mode}_state.json"
    if not p.exists():
        raise SystemExit(f"No state file at {p}")
    s = json.loads(p.read_text())
    s["risk"].update(halted=False, halt_reason="", peak_equity=0.0)
    p.write_text(json.dumps(s, indent=2))
    print(f"Kill switch reset for {args.mode}. Peak equity will restart from current equity.")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="trendwarden", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def data_args(p):
        p.add_argument("--csv", help="Use a local OHLCV CSV instead of downloading")
        p.add_argument("--synthetic", action="store_true", help="Use generated data (offline demo)")

    bt = sub.add_parser("backtest")
    data_args(bt)
    bt.add_argument("--out", default="results")
    bt.add_argument("--plot", action="store_true")

    op = sub.add_parser("optimise", aliases=["optimize"])
    data_args(op)
    op.add_argument("--grid", help='JSON, e.g. \'{"fast_ema":[12,21],"slow_ema":[55,89]}\'')
    op.add_argument("--train", type=float, default=0.7)
    op.add_argument("--min-trades", type=int, default=10)

    dc = sub.add_parser("discover", help="walk-forward test all strategies and pick the most robust")
    data_args(dc)
    dc.add_argument("--folds", type=int, default=4)
    dc.add_argument("--min-trades", type=int, default=10)
    dc.add_argument("--strategies", help="comma-separated subset, e.g. supertrend,macd_trend")
    dc.add_argument("--out", default="results")

    it = sub.add_parser("intel", help="news guard / smart money / Congress reports")
    it.add_argument("what", choices=["news", "smart-money", "congress"])
    it.add_argument("--days", type=int, default=14, help="news: how far ahead to list events")
    it.add_argument("--refresh", action="store_true", help="smart-money: re-download now")

    cp = sub.add_parser("copy", help="copy a famous investor's US stock portfolio via Alpaca")
    cp.add_argument("action", choices=["plan", "rebalance", "execute", "status"])
    cp.add_argument("--investor", default="berkshire", help="name (buffett, ackman, dalio...) or SEC CIK")
    cp.add_argument("--top", type=int, default=5)
    cp.add_argument("--budget", type=float, help="dollars to invest (saved in trader/config.json)")
    cp.add_argument("--no-alpaca", action="store_true", help="plan without connecting to Alpaca")
    cp.add_argument("--sell-untracked", action="store_true", help="rebalance: also sell stocks not in the targets")

    sub.add_parser("paper")
    lv = sub.add_parser("live")
    lv.add_argument("--confirm-live", action="store_true")
    sub.add_parser("status")
    rh = sub.add_parser("reset-halt")
    rh.add_argument("--mode", choices=["paper", "live"], required=True)

    args = ap.parse_args(argv)
    load_env()
    cfg = load_config(args.config)
    logfile = f"{cfg.state_dir}/{args.cmd}.log" if args.cmd in ("paper", "live") else None
    setup_logging(args.verbose, logfile)
    if args.cmd in ("backtest", "optimise", "optimize", "discover") and not args.verbose:
        logging.getLogger("bot.trader").setLevel(logging.WARNING)  # per-trade lines only with -v

    if args.cmd == "backtest":
        cmd_backtest(cfg, args)
    elif args.cmd in ("optimise", "optimize"):
        cmd_optimise(cfg, args)
    elif args.cmd == "intel":
        cmd_intel(cfg, args)
    elif args.cmd == "copy":
        cmd_copy(cfg, args)
    elif args.cmd == "discover":
        cmd_discover(cfg, args)
    elif args.cmd in ("paper", "live"):
        cmd_run(cfg, args, args.cmd)
    elif args.cmd == "status":
        cmd_status(cfg, args)
    elif args.cmd == "reset-halt":
        cmd_reset_halt(cfg, args)


if __name__ == "__main__":
    main()
