"""Minimal Alpaca Trading API client (standard library only) for stock copy-trading.

Keys come from the environment: ALPACA_API_KEY / ALPACA_API_SECRET.
Paper trading is the default; ALPACA_PAPER=false switches to the live API.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL = "https://api.alpaca.markets"


class AlpacaError(Exception):
    pass


class Alpaca:
    def __init__(self, key: str | None = None, secret: str | None = None, paper: bool | None = None):
        self.key = key or os.getenv("ALPACA_API_KEY")
        self.secret = secret or os.getenv("ALPACA_API_SECRET")
        if paper is None:
            paper = os.getenv("ALPACA_PAPER", "true").strip().lower() not in ("false", "0", "no")
        self.paper = paper
        self.base = PAPER_URL if paper else LIVE_URL
        if not (self.key and self.secret):
            raise AlpacaError("Set ALPACA_API_KEY and ALPACA_API_SECRET in .env "
                              "(Alpaca dashboard > API Keys; use your PAPER account's keys first).")

    def _req(self, method: str, path: str, body: dict | None = None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method, headers={
            "APCA-API-KEY-ID": self.key, "APCA-API-SECRET-KEY": self.secret,
            "Content-Type": "application/json", "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise AlpacaError(f"Alpaca {method} {path} failed: HTTP {exc.code} {detail}") from None
        except urllib.error.URLError as exc:
            raise AlpacaError(f"Could not reach Alpaca: {exc.reason}") from None

    def account(self) -> dict:
        return self._req("GET", "/v2/account")

    def positions(self) -> list[dict]:
        return self._req("GET", "/v2/positions")

    def clock(self) -> dict:
        return self._req("GET", "/v2/clock")

    def place_order(self, order: dict) -> dict:
        return self._req("POST", "/v2/orders", order)

    def get_order(self, order_id: str) -> dict:
        return self._req("GET", f"/v2/orders/{order_id}")

    def wait_fill(self, order_id: str, seconds: float = 20) -> dict:
        """Poll until the order is filled (or stops changing); market orders
        outside market hours stay 'accepted' until the open."""
        end = time.monotonic() + seconds
        order = self.get_order(order_id)
        while order.get("status") not in ("filled", "canceled", "rejected", "expired") and time.monotonic() < end:
            time.sleep(1)
            order = self.get_order(order_id)
        return order
