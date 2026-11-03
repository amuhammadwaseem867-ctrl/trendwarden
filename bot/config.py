"""Load and validate the YAML configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .risk import RiskConfig
from .strategies import build_strategy


@dataclass
class ExecutionConfig:
    fee_pct: float = 0.1          # taker fee per side, in percent
    slippage_pct: float = 0.05    # assumed adverse slippage per fill, in percent
    poll_seconds: int = 30        # live/paper: how often to check prices
    starting_cash: float = 10_000 # backtest/paper starting balance (quote currency)

    @property
    def fee_rate(self) -> float:
        return self.fee_pct / 100

    @property
    def slippage_rate(self) -> float:
        return self.slippage_pct / 100


@dataclass
class Config:
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

    def strategy(self):
        return build_strategy(self.strategy_name, self.strategy_params)

    @property
    def quote_currency(self) -> str:
        quotes = {s.split("/")[1].split(":")[0] for s in self.symbols}
        if len(quotes) != 1:
            raise ValueError(f"All symbols must share one quote currency, got {sorted(quotes)}")
        return quotes.pop()


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    strat = raw.get("strategy", {}) or {}
    exe = raw.get("execution", {}) or {}
    unknown_exe = set(exe) - set(ExecutionConfig.__dataclass_fields__)
    if unknown_exe:
        raise ValueError(f"Unknown execution settings: {sorted(unknown_exe)}")

    cfg = Config(
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
    )
    if not cfg.symbols:
        raise ValueError("At least one symbol is required")
    cfg.strategy()          # validates strategy name and params
    _ = cfg.quote_currency  # validates symbols
    return cfg
