"""Technical indicators implemented with pandas (no TA-Lib dependency).

All smoothing that the original authors defined with Wilder's method (RSI, ATR,
ADX) uses an EMA with alpha = 1 / period, which matches TradingView and most
charting packages.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def _wilder(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = _wilder(delta.clip(lower=0.0), period)
    loss = _wilder(-delta.clip(upper=0.0), period)
    rs = gain / loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # No losses in the window -> RSI is 100 by definition.
    return out.where(loss != 0.0, 100.0).where(gain.notna())


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return _wilder(true_range(df), period)


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr = _wilder(true_range(df), period)
    plus_di = 100.0 * _wilder(plus_dm, period) / tr
    minus_di = 100.0 * _wilder(minus_dm, period) / tr
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return _wilder(dx, period)


def bollinger(close: pd.Series, period: int = 20, std: float = 2.0):
    mid = sma(close, period)
    dev = close.rolling(period, min_periods=period).std(ddof=0)
    return mid - std * dev, mid, mid + std * dev


def donchian(df: pd.DataFrame, period: int):
    """Highest high / lowest low of the *previous* `period` bars (excludes the current bar)."""
    upper = df["high"].rolling(period, min_periods=period).max().shift(1)
    lower = df["low"].rolling(period, min_periods=period).min().shift(1)
    return lower, upper


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, sig, line - sig


def keltner(df: pd.DataFrame, period: int = 20, mult: float = 1.5):
    mid = ema(df["close"], period)
    rng = atr(df, period)
    return mid - mult * rng, mid, mid + mult * rng


def supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0):
    """Supertrend line and direction (+1 up-trend, -1 down-trend)."""
    a = atr(df, period).to_numpy()
    hl2 = ((df["high"] + df["low"]) / 2).to_numpy()
    close = df["close"].to_numpy()
    n = len(df)
    upper = hl2 + mult * a
    lower = hl2 - mult * a
    fu, fl = upper.copy(), lower.copy()
    direction = np.zeros(n)
    line = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(a[i]):
            continue
        if i == 0 or np.isnan(a[i - 1]):
            direction[i] = 1
            line[i] = fl[i]
            continue
        # Bands only tighten while price stays on their side.
        fu[i] = upper[i] if (upper[i] < fu[i - 1] or close[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lower[i] if (lower[i] > fl[i - 1] or close[i - 1] < fl[i - 1]) else fl[i - 1]
        if direction[i - 1] >= 0:
            direction[i] = -1 if close[i] < fl[i] else 1
        else:
            direction[i] = 1 if close[i] > fu[i] else -1
        line[i] = fl[i] if direction[i] > 0 else fu[i]
    return pd.Series(line, index=df.index), pd.Series(direction, index=df.index)
