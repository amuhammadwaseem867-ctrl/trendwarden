"""Trading strategies.

A strategy turns a candle DataFrame into one with these extra columns:

* ``entry`` (bool)  - open a long position at the close of this bar
* ``exit``  (bool)  - close any open long position at the close of this bar
* ``atr``   (float) - volatility used by the risk layer for stops and sizing

Signals are computed only from data available at the close of each bar, so the
backtester and the live engine see exactly the same thing (no look-ahead).
The bot is long-only (spot): it never shorts and never uses leverage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from . import indicators as ta


@dataclass
class Strategy:
    params: dict[str, Any] = field(default_factory=dict)
    name: str = "base"
    defaults: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = set(self.params) - set(self.defaults)
        if unknown:
            raise ValueError(f"Unknown params for {self.name}: {sorted(unknown)}")
        self.p = {**self.defaults, **self.params}

    @property
    def warmup(self) -> int:
        """Bars needed before signals are meaningful."""
        return 50

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover
        raise NotImplementedError


class TrendMomentum(Strategy):
    """Trend following: EMA crossover, filtered by a long-term trend EMA,
    ADX trend strength and an RSI cap so it doesn't buy into exhaustion."""

    def __init__(self, params: dict | None = None):
        super().__init__(
            params=params or {},
            name="trend_momentum",
            defaults=dict(
                fast_ema=21, slow_ema=55, trend_ema=200,
                rsi_period=14, rsi_max_entry=70,
                adx_period=14, adx_min=20, atr_period=14,
            ),
        )

    @property
    def warmup(self) -> int:
        return int(self.p["trend_ema"]) + 10

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.p
        out = df.copy()
        fast = ta.ema(out["close"], p["fast_ema"])
        slow = ta.ema(out["close"], p["slow_ema"])
        trend = ta.ema(out["close"], p["trend_ema"])
        r = ta.rsi(out["close"], p["rsi_period"])
        strength = ta.adx(out, p["adx_period"])
        out["atr"] = ta.atr(out, p["atr_period"])

        cross_up = (fast > slow) & (fast.shift(1) <= slow.shift(1))
        cross_down = (fast < slow) & (fast.shift(1) >= slow.shift(1))
        out["entry"] = (
            cross_up
            & (out["close"] > trend)
            & (strength >= p["adx_min"])
            & (r <= p["rsi_max_entry"])
        )
        out["exit"] = cross_down | (out["close"] < trend)
        return out


class MeanReversion(Strategy):
    """Buy oversold dips in an up-trend: close below the lower Bollinger band
    with low RSI while price is above the long-term trend EMA. Exit at the mean."""

    def __init__(self, params: dict | None = None):
        super().__init__(
            params=params or {},
            name="mean_reversion",
            defaults=dict(
                bb_period=20, bb_std=2.0, rsi_period=14, rsi_entry=30,
                trend_ema=200, atr_period=14,
            ),
        )

    @property
    def warmup(self) -> int:
        return int(self.p["trend_ema"]) + 10

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.p
        out = df.copy()
        lower, mid, _ = ta.bollinger(out["close"], p["bb_period"], p["bb_std"])
        r = ta.rsi(out["close"], p["rsi_period"])
        trend = ta.ema(out["close"], p["trend_ema"])
        out["atr"] = ta.atr(out, p["atr_period"])
        out["entry"] = (out["close"] < lower) & (r <= p["rsi_entry"]) & (out["close"] > trend)
        out["exit"] = out["close"] >= mid
        return out


class Breakout(Strategy):
    """Donchian channel breakout (turtle-style): buy a close above the N-bar
    high, exit on a close below the M-bar low."""

    def __init__(self, params: dict | None = None):
        super().__init__(
            params=params or {},
            name="breakout",
            defaults=dict(entry_period=55, exit_period=20, atr_period=20, trend_ema=200),
        )

    @property
    def warmup(self) -> int:
        return max(int(self.p["entry_period"]), int(self.p["trend_ema"])) + 10

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.p
        out = df.copy()
        _, upper = ta.donchian(out, p["entry_period"])
        lower, _ = ta.donchian(out, p["exit_period"])
        trend = ta.ema(out["close"], p["trend_ema"])
        out["atr"] = ta.atr(out, p["atr_period"])
        out["entry"] = (out["close"] > upper) & (out["close"] > trend)
        out["exit"] = out["close"] < lower
        return out


STRATEGIES = {
    "trend_momentum": TrendMomentum,
    "mean_reversion": MeanReversion,
    "breakout": Breakout,
}


def build_strategy(name: str, params: dict | None = None) -> Strategy:
    try:
        return STRATEGIES[name](params)
    except KeyError:
        raise ValueError(f"Unknown strategy '{name}'. Choose from: {', '.join(STRATEGIES)}") from None
