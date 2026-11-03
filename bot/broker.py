"""Order execution.

* SimBroker   - simulated fills with fees and slippage (backtests and paper trading)
* LiveBroker  - real market orders on an exchange through ccxt

Both expose the same small interface so the trading core doesn't care which
one it is talking to.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class Fill:
    qty: float
    price: float  # average fill price
    fee: float    # in quote currency


class SimBroker:
    """Simulated broker.

    * leverage == 1: cash account (crypto spot) - buying costs the full price.
    * leverage > 1:  margin/CFD account (e.g. gold on MT5) - buying only ties up
      margin (value / leverage); profit or loss is settled when the trade closes.
    `min_qty` / `qty_step` mimic the broker's minimum and step size (in units,
    e.g. ounces of gold: 0.01 lot of GOLD = 1 oz).
    """

    def __init__(self, cash: float, fee_rate: float, slippage_rate: float,
                 leverage: float = 1.0, min_qty: float = 0.0, qty_step: float = 0.0):
        self.cash = float(cash)
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.leverage = max(1.0, float(leverage))
        self.min_qty_units = float(min_qty)
        self.qty_step = float(qty_step)
        self.holdings: dict[str, float] = {}
        self.avg_price: dict[str, float] = {}

    @property
    def margin_mode(self) -> bool:
        return self.leverage > 1

    def min_qty(self, symbol: str) -> float:
        return self.min_qty_units

    def _round(self, qty: float) -> float:
        if self.qty_step > 0:
            qty = math.floor(qty / self.qty_step + 1e-9) * self.qty_step
        return qty

    def used_margin(self) -> float:
        return sum(q * self.avg_price[s] for s, q in self.holdings.items()) / self.leverage

    def get_cash(self) -> float:
        """Spot: cash. Margin: buying power (free margin x leverage)."""
        if self.margin_mode:
            return max(0.0, self.cash - self.used_margin()) * self.leverage
        return self.cash

    def equity_at(self, prices: dict[str, float]) -> float:
        if self.margin_mode:
            return self.cash + sum(q * (prices.get(s, self.avg_price[s]) - self.avg_price[s])
                                   for s, q in self.holdings.items())
        return self.cash + sum(q * prices.get(s, self.avg_price[s]) for s, q in self.holdings.items())

    def buy(self, symbol: str, qty: float, price: float) -> Fill | None:
        qty = self._round(qty)
        if qty <= 0 or qty < self.min_qty_units - 1e-12:
            return None
        fill_price = price * (1 + self.slippage_rate)
        value = qty * fill_price
        fee = value * self.fee_rate
        if self.margin_mode:
            if value / self.leverage + fee > self.cash - self.used_margin() + 1e-9:
                return None  # not enough free margin
            self.cash -= fee
        else:
            if value + fee > self.cash + 1e-9:
                return None
            self.cash -= value + fee
        held = self.holdings.get(symbol, 0.0)
        self.avg_price[symbol] = (held * self.avg_price.get(symbol, 0.0) + value) / (held + qty)
        self.holdings[symbol] = held + qty
        return Fill(qty, fill_price, fee)

    def sell(self, symbol: str, qty: float, price: float) -> Fill | None:
        held = self.holdings.get(symbol, 0.0)
        qty = min(qty, held)
        if qty <= 0:
            return None
        fill_price = price * (1 - self.slippage_rate)
        proceeds = qty * fill_price
        fee = proceeds * self.fee_rate
        if self.margin_mode:
            self.cash += qty * (fill_price - self.avg_price[symbol]) - fee
        else:
            self.cash += proceeds - fee
        self.holdings[symbol] = held - qty
        if self.holdings[symbol] <= 1e-12:
            del self.holdings[symbol]
            del self.avg_price[symbol]
        return Fill(qty, fill_price, fee)

    def to_dict(self) -> dict:
        return {"cash": self.cash, "holdings": self.holdings, "avg_price": self.avg_price}

    def load(self, d: dict) -> None:
        self.cash = float(d["cash"])
        self.holdings = {k: float(v) for k, v in d.get("holdings", {}).items()}
        self.avg_price = {k: float(v) for k, v in d.get("avg_price", {}).items()}
        for k in self.holdings:  # state files from older versions
            self.avg_price.setdefault(k, 0.0)


class LiveBroker:
    """Places real market orders. Every order is checked against the exchange's
    minimum size and rounded to its precision before it is sent."""

    def __init__(self, exchange, quote: str, fee_rate: float):
        self.ex = exchange
        self.quote = quote
        self.fee_rate = fee_rate
        self.ex.load_markets()

    def get_cash(self) -> float:
        bal = self.ex.fetch_balance()
        return float(bal.get("free", {}).get(self.quote) or 0.0)

    def base_balance(self, symbol: str) -> float:
        base = self.ex.market(symbol)["base"]
        bal = self.ex.fetch_balance()
        return float(bal.get("free", {}).get(base) or 0.0)

    def _amount(self, symbol: str, qty: float, price: float) -> float | None:
        market = self.ex.market(symbol)
        amount = float(self.ex.amount_to_precision(symbol, qty))
        limits = market.get("limits", {})
        min_amt = (limits.get("amount") or {}).get("min") or 0
        min_cost = (limits.get("cost") or {}).get("min") or 0
        if amount <= 0 or amount < min_amt or amount * price < min_cost:
            log.warning("%s: order of %s too small for exchange limits (min amount %s, min cost %s)",
                        symbol, qty, min_amt, min_cost)
            return None
        return amount

    def _fill(self, order: dict, fallback_price: float) -> Fill:
        oid = order.get("id")
        if oid and (order.get("filled") is None or order.get("average") is None):
            try:
                order = self.ex.fetch_order(oid, order.get("symbol"))
            except Exception as exc:  # some exchanges don't support fetch_order
                log.debug("fetch_order failed: %s", exc)
        qty = float(order.get("filled") or order.get("amount") or 0)
        price = float(order.get("average") or order.get("price") or fallback_price)
        fee_info = order.get("fee") or {}
        if fee_info.get("cost") is not None and fee_info.get("currency") == self.quote:
            fee = float(fee_info["cost"])
        else:
            fee = qty * price * self.fee_rate
        return Fill(qty, price, fee)

    def buy(self, symbol: str, qty: float, price: float) -> Fill | None:
        amount = self._amount(symbol, qty, price)
        if amount is None:
            return None
        log.info("LIVE BUY %s %s @ ~%s", amount, symbol, price)
        return self._fill(self.ex.create_market_buy_order(symbol, amount), price)

    def sell(self, symbol: str, qty: float, price: float) -> Fill | None:
        # Never sell more than we actually hold (fees may have been taken in the base coin).
        qty = min(qty, self.base_balance(symbol))
        amount = self._amount(symbol, qty, price)
        if amount is None:
            return None
        log.info("LIVE SELL %s %s @ ~%s", amount, symbol, price)
        return self._fill(self.ex.create_market_sell_order(symbol, amount), price)
