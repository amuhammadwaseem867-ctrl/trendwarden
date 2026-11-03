"""Order execution.

* SimBroker   - simulated fills with fees and slippage (backtests and paper trading)
* LiveBroker  - real market orders on an exchange through ccxt

Both expose the same small interface so the trading core doesn't care which
one it is talking to.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class Fill:
    qty: float
    price: float  # average fill price
    fee: float    # in quote currency


class SimBroker:
    def __init__(self, cash: float, fee_rate: float, slippage_rate: float):
        self.cash = float(cash)
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.holdings: dict[str, float] = {}

    def get_cash(self) -> float:
        return self.cash

    def buy(self, symbol: str, qty: float, price: float) -> Fill | None:
        fill_price = price * (1 + self.slippage_rate)
        cost = qty * fill_price
        fee = cost * self.fee_rate
        if qty <= 0 or cost + fee > self.cash + 1e-9:
            return None
        self.cash -= cost + fee
        self.holdings[symbol] = self.holdings.get(symbol, 0.0) + qty
        return Fill(qty, fill_price, fee)

    def sell(self, symbol: str, qty: float, price: float) -> Fill | None:
        held = self.holdings.get(symbol, 0.0)
        qty = min(qty, held)
        if qty <= 0:
            return None
        fill_price = price * (1 - self.slippage_rate)
        proceeds = qty * fill_price
        fee = proceeds * self.fee_rate
        self.cash += proceeds - fee
        self.holdings[symbol] = held - qty
        if self.holdings[symbol] <= 1e-12:
            del self.holdings[symbol]
        return Fill(qty, fill_price, fee)

    def to_dict(self) -> dict:
        return {"cash": self.cash, "holdings": self.holdings}

    def load(self, d: dict) -> None:
        self.cash = float(d["cash"])
        self.holdings = {k: float(v) for k, v in d.get("holdings", {}).items()}


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
