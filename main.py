#!/usr/bin/env python3
"""TrendWarden - command line entry point.

    python main.py backtest            # test the strategy on historical data
    python main.py optimise            # walk-forward parameter search
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
    from bot.data import fetch_history, load_csv, make_market, synthetic_ohlcv

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
    return data


def cmd_backtest(cfg, args) -> None:
    from bot.backtest import run_backtest

    data = get_data(cfg, args)
    res = run_backtest(data, cfg.strategy_name, cfg.strategy_params, cfg.risk, cfg.timeframe,
                       cfg.execution.starting_cash, cfg.execution.fee_rate, cfg.execution.slippage_rate)
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
}


def cmd_optimise(cfg, args) -> None:
    from bot.backtest import optimise

    grid = json.loads(args.grid) if args.grid else DEFAULT_GRIDS[cfg.strategy_name]
    data = get_data(cfg, args)
    report = optimise(data, cfg.strategy_name, grid, cfg.strategy_params, cfg.risk, cfg.timeframe,
                      train_frac=args.train, min_trades=args.min_trades,
                      starting_cash=cfg.execution.starting_cash, fee_rate=cfg.execution.fee_rate,
                      slippage_rate=cfg.execution.slippage_rate)
    if report.empty:
        print("No parameter set produced enough trades. Try more history or a lower --min-trades.")
        return
    pd.set_option("display.width", 200)
    print("\nTop configurations (IS = in-sample/training, OOS = out-of-sample/unseen):\n")
    print(report.to_string(index=False))
    print("\nPrefer configs that stay profitable OOS with similar Sharpe - big IS/OOS gaps mean overfitting.")


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
    if args.cmd in ("backtest", "optimise", "optimize") and not args.verbose:
        logging.getLogger("bot.trader").setLevel(logging.WARNING)  # per-trade lines only with -v

    if args.cmd == "backtest":
        cmd_backtest(cfg, args)
    elif args.cmd in ("optimise", "optimize"):
        cmd_optimise(cfg, args)
    elif args.cmd in ("paper", "live"):
        cmd_run(cfg, args, args.cmd)
    elif args.cmd == "status":
        cmd_status(cfg, args)
    elif args.cmd == "reset-halt":
        cmd_reset_halt(cfg, args)


if __name__ == "__main__":
    main()
