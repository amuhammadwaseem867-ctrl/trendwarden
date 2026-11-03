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
        out["bull"] = (fast > slow) & (out["close"] > trend)
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


def _trend_filter(close: pd.Series, period: int) -> pd.Series:
    """close above its EMA, or always True when the filter is disabled (period 0)."""
    if not period:
        return pd.Series(True, index=close.index)
    return close > ta.ema(close, period)


class Supertrend(Strategy):
    """Supertrend (ATR-based trend line). Buys when the trend flips up while
    price is above the long-term EMA; sells when it flips down."""

    def __init__(self, params: dict | None = None):
        super().__init__(params=params or {}, name="supertrend",
                         defaults=dict(st_period=10, st_mult=3.0, trend_ema=200, atr_period=14))

    @property
    def warmup(self) -> int:
        return int(self.p["trend_ema"]) + 10

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.p
        out = df.copy()
        _, direction = ta.supertrend(out, p["st_period"], p["st_mult"])
        trend_ok = _trend_filter(out["close"], p["trend_ema"])
        out["atr"] = ta.atr(out, p["atr_period"])
        flip_up = (direction > 0) & (direction.shift(1) < 0)
        out["entry"] = flip_up & trend_ok
        out["exit"] = direction < 0
        out["bull"] = (direction > 0) & trend_ok
        return out


class MacdTrend(Strategy):
    """MACD crosses above its signal line while price is above the trend EMA
    and MACD is below zero (buying the start of a new leg, not the top)."""

    def __init__(self, params: dict | None = None):
        super().__init__(params=params or {}, name="macd_trend",
                         defaults=dict(fast=12, slow=26, signal=9, trend_ema=200,
                                       below_zero_only=True, atr_period=14))

    @property
    def warmup(self) -> int:
        return int(self.p["trend_ema"]) + 10

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.p
        out = df.copy()
        line, sig, _ = ta.macd(out["close"], p["fast"], p["slow"], p["signal"])
        trend_ok = _trend_filter(out["close"], p["trend_ema"])
        out["atr"] = ta.atr(out, p["atr_period"])
        cross_up = (line > sig) & (line.shift(1) <= sig.shift(1))
        cross_down = (line < sig) & (line.shift(1) >= sig.shift(1))
        zero_ok = (line < 0) if p["below_zero_only"] else True
        out["entry"] = cross_up & trend_ok & zero_ok
        out["exit"] = cross_down
        out["bull"] = (line > sig) & trend_ok
        return out


class TrendPullback(Strategy):
    """Buy the dip inside an established up-trend: fast EMA above slow EMA,
    RSI pulls back below `rsi_low`, then turns back up through it."""

    def __init__(self, params: dict | None = None):
        super().__init__(params=params or {}, name="trend_pullback",
                         defaults=dict(fast_ema=50, slow_ema=200, rsi_period=14, rsi_low=40,
                                       rsi_exit=75, atr_period=14))

    @property
    def warmup(self) -> int:
        return int(self.p["slow_ema"]) + 10

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.p
        out = df.copy()
        fast = ta.ema(out["close"], p["fast_ema"])
        slow = ta.ema(out["close"], p["slow_ema"])
        r = ta.rsi(out["close"], p["rsi_period"])
        out["atr"] = ta.atr(out, p["atr_period"])
        uptrend = (fast > slow) & (out["close"] > slow)
        turn_up = (r > p["rsi_low"]) & (r.shift(1) <= p["rsi_low"])
        out["entry"] = uptrend & turn_up
        out["exit"] = (out["close"] < slow) | (r >= p["rsi_exit"])
        out["bull"] = uptrend
        return out


class SqueezeBreakout(Strategy):
    """Volatility squeeze: Bollinger Bands contract inside the Keltner Channel
    (quiet market), then price breaks out upward as volatility returns."""

    def __init__(self, params: dict | None = None):
        super().__init__(params=params or {}, name="squeeze_breakout",
                         defaults=dict(bb_period=20, bb_std=2.0, kc_period=20, kc_mult=1.5,
                                       lookback=6, trend_ema=100, atr_period=14))

    @property
    def warmup(self) -> int:
        return max(int(self.p["trend_ema"]), int(self.p["bb_period"]), int(self.p["kc_period"])) + 10

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.p
        out = df.copy()
        bb_lo, _, bb_hi = ta.bollinger(out["close"], p["bb_period"], p["bb_std"])
        kc_lo, kc_mid, kc_hi = ta.keltner(out, p["kc_period"], p["kc_mult"])
        trend_ok = _trend_filter(out["close"], p["trend_ema"])
        out["atr"] = ta.atr(out, p["atr_period"])
        squeeze = (bb_lo > kc_lo) & (bb_hi < kc_hi)
        recently_squeezed = squeeze.astype(float).rolling(p["lookback"], min_periods=1).max().shift(1) == 1
        out["entry"] = recently_squeezed & ~squeeze & (out["close"] > kc_hi) & trend_ok
        out["exit"] = out["close"] < kc_mid
        out["bull"] = (out["close"] > kc_mid) & trend_ok
        return out


class Ensemble(Strategy):
    """Committee vote: each member strategy says whether it is bullish. Buy
    when at least `min_votes` agree, sell when support drops below that.
    Fewer, higher-conviction trades than any single member."""

    def __init__(self, params: dict | None = None):
        super().__init__(params=params or {}, name="ensemble",
                         defaults=dict(members=["trend_momentum", "supertrend", "macd_trend"],
                                       min_votes=2, member_params={}, atr_period=14))
        bad = [m for m in self.p["members"] if m in ("ensemble", "mean_reversion", "breakout")
               or m not in STRATEGIES]
        if bad:
            raise ValueError(f"Ensemble members must be trend strategies with a bull state, got {bad}")
        self.members = [build_strategy(m, self.p["member_params"].get(m)) for m in self.p["members"]]

    @property
    def warmup(self) -> int:
        return max(m.warmup for m in self.members)

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        votes = sum(m.generate(df)["bull"].fillna(False).astype(int) for m in self.members)
        bull = votes >= self.p["min_votes"]
        out["atr"] = ta.atr(out, self.p["atr_period"])
        out["entry"] = bull & ~bull.shift(1, fill_value=False)
        out["exit"] = ~bull
        out["bull"] = bull
        out["votes"] = votes
        return out


STRATEGIES = {
    "trend_momentum": TrendMomentum,
    "mean_reversion": MeanReversion,
    "breakout": Breakout,
    "supertrend": Supertrend,
    "macd_trend": MacdTrend,
    "trend_pullback": TrendPullback,
    "squeeze_breakout": SqueezeBreakout,
    "ensemble": Ensemble,
}


def build_strategy(name: str, params: dict | None = None) -> Strategy:
    try:
        return STRATEGIES[name](params)
    except KeyError:
        raise ValueError(f"Unknown strategy '{name}'. Choose from: {', '.join(STRATEGIES)}") from None
