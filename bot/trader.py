"""The trading core: turns strategy signals into orders under the risk rules.

The backtester and the live/paper engine both drive this same class one closed
bar at a time, so a strategy behaves identically in testing and in production.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Callable

import pandas as pd

from .risk import Position, RiskManager

log = logging.getLogger(__name__)


@dataclass
class Trade:
    symbol: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    qty: float
    pnl: float           # net of fees
    r_multiple: float    # pnl relative to the initial risk
    reason: str


class Trader:
    def __init__(self, broker, risk: RiskManager, notify: Callable[[str], None] | None = None):
        self.broker = broker
        self.risk = risk
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []
        self.entry_fees: dict[str, float] = {}
        self.notify = notify or (lambda msg: None)

    # ----- accounting ----------------------------------------------------
    def equity(self, prices: dict[str, float]) -> float:
        value = sum(p.qty * prices.get(s, p.entry_price) for s, p in self.positions.items())
        return self.broker.get_cash() + value

    # ----- orders --------------------------------------------------------
    def open(self, symbol: str, price: float, atr: float, ts: str, prices: dict[str, float]) -> None:
        equity = self.equity(prices)
        ok, why = self.risk.can_open(symbol, equity, len(self.positions))
        if not ok:
            log.info("%s %s: entry signal skipped (%s)", ts, symbol, why)
            return
        qty = self.risk.size(equity, self.broker.get_cash(), price, atr, self.broker.fee_rate)
        if qty <= 0:
            return
        fill = self.broker.buy(symbol, qty, price)
        if fill is None or fill.qty <= 0:
            return
        pos = self.risk.new_position(symbol, fill.qty, fill.price, atr, ts)
        self.positions[symbol] = pos
        self.entry_fees[symbol] = fill.fee
        msg = (f"BUY {symbol} qty={fill.qty:.6g} @ {fill.price:.6g} | stop {pos.stop:.6g}"
               + (f" | target {pos.take_profit:.6g}" if pos.take_profit else ""))
        log.info("%s %s", ts, msg)
        self.notify(msg)

    def close(self, symbol: str, price: float, ts: str, reason: str) -> Trade | None:
        pos = self.positions.get(symbol)
        if pos is None:
            return None
        fill = self.broker.sell(symbol, pos.qty, price)
        if fill is None or fill.qty <= 0:
            log.error("%s %s: failed to close position (%s)", ts, symbol, reason)
            return None
        entry_fee = self.entry_fees.pop(symbol, 0.0)
        pnl = (fill.price - pos.entry_price) * fill.qty - fill.fee - entry_fee
        risk_cash = pos.initial_risk * pos.qty
        trade = Trade(symbol, pos.entry_time, ts, pos.entry_price, fill.price, fill.qty, pnl,
                      pnl / risk_cash if risk_cash else 0.0, reason)
        self.trades.append(trade)
        del self.positions[symbol]
        self.risk.record_close(symbol, pnl)
        msg = f"SELL {symbol} @ {fill.price:.6g} ({reason}) | PnL {pnl:+.2f} ({trade.r_multiple:+.2f}R)"
        log.info("%s %s", ts, msg)
        self.notify(msg)
        return trade

    def flatten(self, prices: dict[str, float], ts: str, reason: str) -> None:
        for symbol in list(self.positions):
            self.close(symbol, prices[symbol], ts, reason)

    # ----- per-bar logic -------------------------------------------------
    def on_bar(self, symbol: str, bar: pd.Series, ts: str, prices: dict[str, float]) -> None:
        """Process one *closed* bar for one symbol. `bar` must include the
        strategy columns (entry, exit, atr)."""
        self.risk.tick_cooldowns(symbol)
        pos = self.positions.get(symbol)
        atr = float(bar["atr"]) if pd.notna(bar["atr"]) else 0.0

        if pos is not None:
            hit = self.risk.check_exit(pos, bar["open"], bar["low"], bar["high"])
            if hit:
                self.close(symbol, hit[1], ts, hit[0])
            elif bool(bar["exit"]):
                self.close(symbol, bar["close"], ts, "signal")
            else:
                self.risk.trail(pos, bar["close"], atr)
        elif bool(bar["entry"]) and atr > 0:
            self.open(symbol, bar["close"], atr, ts, prices)

    def after_bar(self, prices: dict[str, float], today: date, ts: str) -> float:
        """Account-level checks once all symbols for a timestamp are processed."""
        equity = self.equity(prices)
        if self.risk.update_equity(equity, today):
            msg = f"KILL SWITCH: {self.risk.state.halt_reason}. Closing all positions and halting."
            log.critical(msg)
            self.notify(msg)
            self.flatten(prices, ts, "kill_switch")
            equity = self.equity(prices)
        return equity
