"""Paper and live trading loop.

Every `poll_seconds` the engine:
1. checks the latest price of each open position against its stop / target
   (stops are managed by the bot, so it must keep running while positions are open);
2. when a new candle has closed, runs the strategy on it through the same
   Trader logic the backtester uses;
3. applies account-level risk checks and saves its state to disk, so a restart
   resumes exactly where it left off.
"""
from __future__ import annotations

import json
import logging
import signal
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from .broker import LiveBroker, SimBroker
from .config import Config
from .data import drop_unclosed, make_market, to_frame
from .notify import make_notifier
from .intel.hub import Intel
from .risk import Position, RiskManager, RiskState
from .trader import Trader

log = logging.getLogger(__name__)


class Engine:
    def __init__(self, cfg: Config, mode: str):
        if mode not in ("paper", "live"):
            raise ValueError("mode must be 'paper' or 'live'")
        self.cfg = cfg
        self.mode = mode
        self.strategy = cfg.strategy()
        self.state_path = Path(cfg.state_dir) / f"{mode}_state.json"
        self.trades_path = Path(cfg.state_dir) / f"{mode}_trades.csv"
        self.running = True
        self.last_bar: dict[str, str] = {}

        if mode == "live":
            self.exchange = make_market(cfg, auth=True)
            if cfg.platform == "mt5":
                from .mt5 import MT5Broker
                broker = MT5Broker(self.exchange.mt5, int(cfg.mt5.get("magic", 240611)),
                                   int(cfg.mt5.get("deviation", 30)), cfg.execution.fee_rate)
            else:
                broker = LiveBroker(self.exchange, cfg.quote_currency, cfg.execution.fee_rate)
        else:
            self.exchange = make_market(cfg)  # market data only, simulated money
            x = cfg.execution
            broker = SimBroker(x.starting_cash, x.fee_rate, x.slippage_rate, x.leverage, x.min_qty, x.qty_step)

        self.trader = Trader(broker, RiskManager(cfg.risk), make_notifier(f"TrendWarden {mode}"))
        self.intel = Intel(cfg.intel, cfg.state_dir, self.trader.notify)
        self._last_block: str | None = None
        self._load_state()

    # persistence
    def _load_state(self) -> None:
        if not self.state_path.exists():
            return
        s = json.loads(self.state_path.read_text())
        self.trader.positions = {k: Position.from_dict(v) for k, v in s["positions"].items()}
        self.trader.entry_fees = s.get("entry_fees", {})
        self.trader.risk.state = RiskState.from_dict(s["risk"])
        self.last_bar = s.get("last_bar", {})
        if self.mode == "paper" and "broker" in s:
            self.trader.broker.load(s["broker"])
        log.info("Resumed from %s with %d open position(s)", self.state_path, len(self.trader.positions))
        if self.trader.risk.state.halted:
            log.warning("Bot is HALTED (%s). Delete 'halted' in %s or run `reset-halt` to resume trading.",
                        self.trader.risk.state.halt_reason, self.state_path)

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        s = {
            "positions": {k: p.to_dict() for k, p in self.trader.positions.items()},
            "entry_fees": self.trader.entry_fees,
            "risk": self.trader.risk.state.to_dict(),
            "last_bar": self.last_bar,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        if self.mode == "paper":
            s["broker"] = self.trader.broker.to_dict()
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(s, indent=2))
        tmp.replace(self.state_path)  # atomic: never leaves a half-written state file

        if self.trader.trades:
            df = pd.DataFrame([t.__dict__ for t in self.trader.trades])
            df.to_csv(self.trades_path, mode="a", header=not self.trades_path.exists(), index=False)
            self.trader.trades.clear()

    # market data
    def _candles(self, symbol: str) -> pd.DataFrame:
        limit = min(self.strategy.warmup + 50, 1000)
        rows = self.exchange.fetch_ohlcv(symbol, self.cfg.timeframe, limit=limit)
        return drop_unclosed(to_frame(rows), self.cfg.timeframe, self.exchange.milliseconds())

    def _prices(self) -> dict[str, float]:
        prices = {}
        for symbol in self.cfg.symbols:
            t = self.exchange.fetch_ticker(symbol)
            prices[symbol] = float(t.get("last") or t.get("close"))
        return prices

    # main loop
    def step(self) -> None:
        now = datetime.now(timezone.utc)
        ts = now.isoformat(timespec="seconds")
        prices = self._prices()

        # 0) Positions the broker closed on its own (server-side stop / target hit).
        broker = self.trader.broker
        if hasattr(broker, "position_open"):
            for symbol, pos in list(self.trader.positions.items()):
                if not broker.position_open(symbol):
                    since = datetime.fromisoformat(pos.entry_time) - timedelta(days=1)
                    price = broker.last_exit_price(symbol, since) or pos.stop
                    reason = "take_profit" if pos.take_profit and price >= pos.take_profit else "stop"
                    self.trader.record_external_close(symbol, price, ts, f"{reason} (broker)")

        # 1) Intrabar stop / take-profit checks on live prices.
        for symbol, pos in list(self.trader.positions.items()):
            px = prices[symbol]
            if px <= pos.stop:
                self.trader.close(symbol, px, ts, "stop")
            elif pos.take_profit is not None and px >= pos.take_profit:
                self.trader.close(symbol, px, ts, "take_profit")

        # 1b) News guard / smart money / Congress alerts.
        if self.intel.active:
            self.intel.update(now)
            block = self.intel.block_reason(now)
            if block != self._last_block:
                if block:
                    log.warning("New entries paused: %s", block)
                    self.trader.notify(f"New entries paused: {block}")
                else:
                    log.info("New entries allowed again")
                self._last_block = block
            self.trader.entry_block = block
            if self.intel.news_pause_started(now):
                self.trader.protect_breakeven(prices, ts)

        # 2) Strategy on newly closed candles.
        for symbol in self.cfg.symbols:
            df = self._candles(symbol)
            if len(df) < self.strategy.warmup:
                log.warning("%s: only %d candles, need %d for warm-up", symbol, len(df), self.strategy.warmup)
                continue
            last_closed = df.index[-1].isoformat()
            if symbol not in self.last_bar:
                # First run: don't act on a signal that may be hours old.
                self.last_bar[symbol] = last_closed
                log.info("%s: starting from candle %s", symbol, last_closed)
                continue
            if last_closed <= self.last_bar[symbol]:
                continue
            sig = self.strategy.generate(df)
            new_bars = sig[sig.index > pd.Timestamp(self.last_bar[symbol])]
            for bar_ts, bar in new_bars.iterrows():
                self.trader.on_bar(symbol, bar, bar_ts.isoformat(), prices)
            self.last_bar[symbol] = last_closed

        # 3) Account-level risk.
        equity = self.trader.after_bar(prices, now.date(), ts)
        open_pos = ", ".join(f"{s} stop {p.stop:.6g}" for s, p in self.trader.positions.items()) or "none"
        log.info("equity %.2f %s | open: %s", equity, self.cfg.quote_currency, open_pos)
        self._save_state()

    def run(self) -> None:
        def stop(*_):
            log.info("Shutdown requested - finishing current step and saving state...")
            self.running = False

        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        log.info("TrendWarden %s mode | %s | %s | %s | strategy=%s",
                 self.mode.upper(), self.cfg.exchange, ",".join(self.cfg.symbols),
                 self.cfg.timeframe, self.strategy.name)
        self.trader.notify(f"Started in {self.mode} mode on {self.cfg.exchange}: {', '.join(self.cfg.symbols)}")

        failures = 0
        while self.running:
            try:
                self.step()
                failures = 0
            except Exception as exc:  # network blips, exchange maintenance, ...
                failures += 1
                wait = min(300, self.cfg.execution.poll_seconds * 2 ** failures)
                log.exception("Step failed (%d in a row), retrying in %ss: %s", failures, wait, exc)
                if failures == 5:
                    self.trader.notify(f"5 consecutive errors, still retrying: {exc}")
                self._sleep(wait)
                continue
            self._sleep(self.cfg.execution.poll_seconds)

        self._save_state()
        if self.trader.positions:
            log.warning("Stopped with open positions: %s. Their stops are NOT active while the bot is off.",
                        ", ".join(self.trader.positions))
        self.trader.notify("Stopped.")

    def _sleep(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while self.running and time.monotonic() < end:
            time.sleep(min(1.0, end - time.monotonic()))
