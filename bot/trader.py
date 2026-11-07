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
        self.entry_block: str | None = None  # set by the news guard / smart-money filter
        self.notify = notify or (lambda msg: None)

    # accounting
    def equity(self, prices: dict[str, float]) -> float:
        if hasattr(self.broker, "account_equity"):  # live margin accounts (MT5) report equity directly
            return self.broker.account_equity()
        if hasattr(self.broker, "equity_at"):
            return self.broker.equity_at(prices)
        value = sum(p.qty * prices.get(s, p.entry_price) for s, p in self.positions.items())
        return self.broker.get_cash() + value

    # orders
    def open(self, symbol: str, price: float, atr: float, ts: str, prices: dict[str, float]) -> None:
        if self.entry_block:
            log.info("%s %s: entry signal skipped (%s)", ts, symbol, self.entry_block)
            return
        equity = self.equity(prices)
        ok, why = self.risk.can_open(symbol, equity, len(self.positions))
        if not ok:
            log.info("%s %s: entry signal skipped (%s)", ts, symbol, why)
            return
        fee_rate = self.broker.fee_rate
        qty = self.risk.size(equity, self.broker.get_cash(), price, atr, fee_rate)
        min_qty = self.broker.min_qty(symbol) if hasattr(self.broker, "min_qty") else 0.0
        if min_qty and qty < min_qty:
            # Too small for the broker's minimum trade size (typical for small accounts).
            risk = self.risk.risk_pct(min_qty, equity, price, atr, fee_rate)
            cap = self.risk.cfg.min_lot_max_risk_pct
            if cap > 0 and risk <= cap:
                log.warning("%s %s: small account - using the minimum trade size; this trade risks %.1f%% "
                            "of equity (limit %.1f%%)", ts, symbol, risk, cap)
                qty = min_qty
            else:
                log.info("%s %s: entry skipped - the minimum trade size would risk %.1f%% of equity "
                         "(allowed: %s)", ts, symbol, risk, f"{cap}%" if cap > 0 else "min_lot_max_risk_pct is off")
                return
        if qty <= 0:
            return
        fill = self.broker.buy(symbol, qty, price)
        if fill is None or fill.qty <= 0:
            return
        pos = self.risk.new_position(symbol, fill.qty, fill.price, atr, ts)
        self.positions[symbol] = pos
        if hasattr(self.broker, "protect"):  # put the stop/target on the broker's server
            self.broker.protect(symbol, pos.stop, pos.take_profit)
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
        return self._record(symbol, fill.price, fill.qty, fill.fee, ts, reason)

    def record_external_close(self, symbol: str, price: float, ts: str, reason: str) -> Trade | None:
        """The broker closed the position itself (e.g. its server-side stop hit)."""
        pos = self.positions.get(symbol)
        if pos is None:
            return None
        return self._record(symbol, price, pos.qty, 0.0, ts, reason)

    def _record(self, symbol: str, price: float, qty: float, fee: float, ts: str, reason: str) -> Trade:
        pos = self.positions[symbol]
        entry_fee = self.entry_fees.pop(symbol, 0.0)
        pnl = (price - pos.entry_price) * qty - fee - entry_fee
        risk_cash = pos.initial_risk * pos.qty
        trade = Trade(symbol, pos.entry_time, ts, pos.entry_price, price, qty, pnl,
                      pnl / risk_cash if risk_cash else 0.0, reason)
        self.trades.append(trade)
        del self.positions[symbol]
        self.risk.record_close(symbol, pnl)
        msg = f"SELL {symbol} @ {price:.6g} ({reason}) | PnL {pnl:+.2f} ({trade.r_multiple:+.2f}R)"
        log.info("%s %s", ts, msg)
        self.notify(msg)
        return trade

    def protect_breakeven(self, prices: dict[str, float], ts: str) -> None:
        """Move the stop of every trade that is in profit up to its entry price."""
        for symbol, pos in self.positions.items():
            px = prices.get(symbol)
            if px is not None and px > pos.entry_price and pos.stop < pos.entry_price:
                pos.stop = pos.entry_price
                log.info("%s %s: news pause - stop moved to break-even %.6g", ts, symbol, pos.stop)
                if hasattr(self.broker, "protect"):
                    self.broker.protect(symbol, pos.stop, pos.take_profit)

    def flatten(self, prices: dict[str, float], ts: str, reason: str) -> None:
        for symbol in list(self.positions):
            self.close(symbol, prices[symbol], ts, reason)

    # per-bar logic
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
                old_stop = pos.stop
                self.risk.trail(pos, bar["close"], atr)
                if pos.stop > old_stop and hasattr(self.broker, "protect"):
                    self.broker.protect(symbol, pos.stop, pos.take_profit)
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
