"""Load and validate the YAML configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .risk import RiskConfig
from .intel.hub import parse as parse_intel
from .strategies import build_strategy


@dataclass
class ExecutionConfig:
    fee_pct: float = 0.1          # taker fee per side, in percent
    slippage_pct: float = 0.05    # assumed adverse slippage per fill, in percent
    poll_seconds: int = 30        # live/paper: how often to check prices
    starting_cash: float = 10_000 # backtest/paper starting balance (quote currency)
    # Backtest/paper only (live MT5 reads these from the broker):
    leverage: float = 1.0         # 1 = cash account (crypto spot); >1 = margin/CFD account
    min_qty: float = 0.0          # broker's minimum trade size in units (gold: 1 oz = 0.01 lot)
    qty_step: float = 0.0         # size increment in units (gold: 1 oz)

    @property
    def fee_rate(self) -> float:
        return self.fee_pct / 100

    @property
    def slippage_rate(self) -> float:
        return self.slippage_pct / 100


@dataclass
class Config:
    platform: str = "ccxt"        # ccxt (crypto exchanges) | mt5 (MetaTrader 5: XM, gold, forex)
    exchange: str = "binance"
    testnet: bool = False
    symbols: list[str] = field(default_factory=lambda: ["BTC/USDT"])
    timeframe: str = "1h"
    history_days: int = 365
    strategy_name: str = "trend_momentum"
    strategy_params: dict = field(default_factory=dict)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    state_dir: str = "state"
    live_enabled: bool = False
    mt5: dict = field(default_factory=dict)  # path, magic, deviation
    intel: dict = field(default_factory=lambda: parse_intel(None))

    def strategy(self):
        return build_strategy(self.strategy_name, self.strategy_params)

    @property
    def quote_currency(self) -> str:
        if self.platform == "mt5":
            return "account currency"
        quotes = {s.split("/")[1].split(":")[0] for s in self.symbols}
        if len(quotes) != 1:
            raise ValueError(f"All symbols must share one quote currency, got {sorted(quotes)}")
        return quotes.pop()


def load_config(path: str | Path) -> Config:
    text = Path(path).read_text(encoding="utf-8")
    try:
        raw = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f" on line {mark.line + 1}" if mark else ""
        line = text.splitlines()[mark.line].strip() if mark and mark.line < len(text.splitlines()) else ""
        hint = ""
        if "\\" in line and '"' in line:
            hint = ("\nWindows paths with backslashes must use SINGLE quotes, e.g.\n"
                    "    path: 'C:\\Program Files\\XM Global MT5\\terminal64.exe'")
        raise SystemExit(f"{path} has a formatting mistake{where}:\n    {line}\n{exc.problem if hasattr(exc, 'problem') else exc}"
                         f"{hint}\nAlso check that indentation uses spaces (not tabs) and lines up.") from None
    strat = raw.get("strategy", {}) or {}
    exe = raw.get("execution", {}) or {}
    unknown_exe = set(exe) - set(ExecutionConfig.__dataclass_fields__)
    if unknown_exe:
        raise ValueError(f"Unknown execution settings: {sorted(unknown_exe)}")

    platform = raw.get("platform", "ccxt")
    if platform not in ("ccxt", "mt5"):
        raise ValueError("platform must be 'ccxt' or 'mt5'")
    cfg = Config(
        platform=platform,
        exchange=raw.get("exchange", "binance"),
        testnet=bool(raw.get("testnet", False)),
        symbols=list(raw.get("symbols", ["BTC/USDT"])),
        timeframe=raw.get("timeframe", "1h"),
        history_days=int(raw.get("history_days", 365)),
        strategy_name=strat.get("name", "trend_momentum"),
        strategy_params=strat.get("params", {}) or {},
        risk=RiskConfig.from_dict(raw.get("risk")),
        execution=ExecutionConfig(**exe),
        state_dir=raw.get("state_dir", "state"),
        live_enabled=bool(raw.get("live_enabled", False)),
        mt5=raw.get("mt5", {}) or {},
        intel=parse_intel(raw.get("intel")),
    )
    if not cfg.symbols:
        raise ValueError("At least one symbol is required")
    cfg.strategy()          # validates strategy name and params
    _ = cfg.quote_currency  # validates symbols
    return cfg
