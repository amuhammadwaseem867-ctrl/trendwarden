"""Market data: download candle history from an exchange, cache it, or generate
synthetic data for offline testing."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def make_exchange(name: str, testnet: bool = False, api_key: str | None = None,
                  secret: str | None = None, password: str | None = None):
    import ccxt  # imported lazily so backtests on CSV data don't need network access

    try:
        cls = getattr(ccxt, name)
    except AttributeError:
        raise ValueError(f"Unknown exchange '{name}'. See ccxt.exchanges for the list.") from None
    params = {"enableRateLimit": True}
    if api_key:
        params.update(apiKey=api_key, secret=secret)
        if password:
            params["password"] = password
    ex = cls(params)
    if testnet:
        ex.set_sandbox_mode(True)
    return ex


def make_market(cfg, auth: bool = False):
    """Market-data/exchange object for the configured platform.
    `auth=True` logs in with credentials from the environment (needed for live)."""
    import os

    if cfg.platform == "mt5":
        from .mt5 import MT5Market, connect

        mt5 = connect(cfg.mt5.get("path"), os.getenv("MT5_LOGIN"), os.getenv("MT5_PASSWORD"),
                      os.getenv("MT5_SERVER"))
        return MT5Market(mt5, cfg.symbols)
    if auth:
        key, secret = os.getenv("EXCHANGE_API_KEY"), os.getenv("EXCHANGE_API_SECRET")
        if not (key and secret):
            raise SystemExit("Live mode needs EXCHANGE_API_KEY and EXCHANGE_API_SECRET in .env")
        return make_exchange(cfg.exchange, cfg.testnet, key, secret, os.getenv("EXCHANGE_API_PASSWORD"))
    return make_exchange(cfg.exchange, cfg.testnet)


def to_frame(rows: list[list]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates("timestamp").set_index("timestamp").sort_index()
    df.index = df.index.as_unit("ns")  # one time unit everywhere (pandas 3 varies it)
    return df.astype(float)


def timeframe_ms(timeframe: str) -> int:
    units = {"m": 60, "h": 3600, "d": 86400, "w": 604800}
    return int(timeframe[:-1]) * units[timeframe[-1]] * 1000


def fetch_history(exchange, symbol: str, timeframe: str, days: int,
                  cache_dir: str | Path | None = "data") -> pd.DataFrame:
    """Download `days` of closed candles, paging through the exchange API.
    Results are cached as CSV and only the missing tail is re-downloaded."""
    step = timeframe_ms(timeframe)
    now = exchange.milliseconds()
    since = now - days * 86_400_000
    cache = None
    cached = None
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        safe = symbol.replace("/", "-").replace(":", "_")
        cache = Path(cache_dir) / f"{exchange.id}_{safe}_{timeframe}.csv"
        if cache.exists():
            try:
                cached = load_csv(cache)
            except Exception as exc:  # corrupt / old-format cache: just re-download
                log.warning("Ignoring unreadable cache %s (%s)", cache, exc)
                cached = None
            if cached is not None and len(cached) and coverage(cached, timeframe) < 0.8:
                log.info("Cached %s history has gaps - downloading it again", symbol)
                cached = None
            if cached is not None and len(cached) and cached.index[0].value // 1_000_000 <= since + step:
                since = cached.index[-1].value // 1_000_000 + step
            else:
                cached = None

    rows: list[list] = []
    cursor = since
    while cursor < now - step:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        nxt = batch[-1][0] + step
        if nxt <= cursor:
            break
        cursor = nxt
        log.debug("%s: fetched %d candles up to %s", symbol, len(rows), pd.to_datetime(cursor, unit="ms"))
        time.sleep(exchange.rateLimit / 1000)

    df = to_frame(rows) if rows else pd.DataFrame(
        columns=COLUMNS[1:], dtype=float, index=pd.DatetimeIndex([], tz="UTC", name="timestamp").as_unit("ns"))
    if cached is not None:
        df = pd.concat([cached, df])
        df = df[~df.index.duplicated(keep="last")].sort_index()
    df = drop_unclosed(df, timeframe, now)
    df = df[df.index >= pd.to_datetime(now - days * 86_400_000, unit="ms", utc=True)]
    if cache is not None and len(df):
        df.to_csv(cache)
    return df


def coverage(df: pd.DataFrame, timeframe: str) -> float:
    """Share of the expected candles that are present (1.0 = no gaps).
    Weekends are excluded, so 24/5 markets like gold/forex score ~1.0 when complete."""
    if len(df) < 2:
        return 0.0
    span_ms = (df.index[-1] - df.index[0]).total_seconds() * 1000
    expected = span_ms / timeframe_ms(timeframe)
    weekday_share = (df.index.dayofweek < 5).mean()
    if weekday_share > 0.97:  # market closed at weekends
        expected *= 5 / 7 * (23 / 24)
    return min(1.0, len(df) / max(expected, 1))


def drop_unclosed(df: pd.DataFrame, timeframe: str, now_ms: int) -> pd.DataFrame:
    """Remove the candle that is still forming (its close isn't final yet)."""
    if df.empty:
        return df
    last_open_ms = df.index[-1].value // 1_000_000
    if last_open_ms + timeframe_ms(timeframe) > now_ms:
        return df.iloc[:-1]
    return df


def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    ts = df["timestamp"]
    if pd.api.types.is_numeric_dtype(ts):  # works with pandas 2 and 3 string columns
        df["timestamp"] = pd.to_datetime(ts, unit="ms", utc=True)
    else:
        df["timestamp"] = pd.to_datetime(ts, utc=True)
    df = df.set_index("timestamp").sort_index()
    df.index = df.index.as_unit("ns")
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def synthetic_ohlcv(n: int = 3000, seed: int = 7, start_price: float = 100.0,
                    freq: str = "1h", regime_len: int = 400) -> pd.DataFrame:
    """Random-walk candles with alternating trending / ranging regimes.
    Used for tests and offline demos - not a model of any real market."""
    rng = np.random.default_rng(seed)
    drift = np.repeat(rng.choice([-0.0008, 0.0, 0.0012], size=n // regime_len + 1), regime_len)[:n]
    rets = drift + rng.normal(0, 0.008, n)
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[start_price], close[:-1]])
    spread = np.abs(rng.normal(0, 0.004, n)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    idx = pd.date_range("2024-01-01", periods=n, freq=freq[:-1] + "min" if freq.endswith("m") else freq, tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": rng.uniform(10, 100, n)},
        index=idx,
    ).rename_axis("timestamp")
