"""Risk management: position sizing, stops, and account-level circuit breakers.

The same RiskManager is used by the backtester and the live engine.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date


@dataclass
class RiskConfig:
    risk_per_trade_pct: float = 1.0      # % of equity lost if the stop is hit
    max_position_pct: float = 25.0       # max notional of one position, % of equity
    max_open_positions: int = 3
    max_daily_loss_pct: float = 3.0      # stop opening trades for the rest of the UTC day
    max_drawdown_pct: float = 15.0       # kill switch: flatten everything and halt
    atr_stop_mult: float = 2.0           # initial stop = entry - mult * ATR
    atr_trail_mult: float = 3.0          # trailing stop = highest close - mult * ATR (0 = off)
    reward_risk: float = 2.5             # take profit at R multiple (0 = off)
    cooldown_bars: int = 3               # bars to wait after a losing trade on a symbol

    @classmethod
    def from_dict(cls, d: dict | None) -> "RiskConfig":
        d = dict(d or {})
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown risk settings: {sorted(unknown)}")
        cfg = cls(**d)
        if not 0 < cfg.risk_per_trade_pct <= 10:
            raise ValueError("risk_per_trade_pct must be between 0 and 10")
        if cfg.atr_stop_mult <= 0:
            raise ValueError("atr_stop_mult must be positive")
        return cfg


@dataclass
class Position:
    symbol: str
    qty: float
    entry_price: float
    stop: float
    take_profit: float | None
    highest: float
    entry_time: str
    initial_risk: float  # price distance entry -> initial stop

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict) -> "Position":
        return cls(**d)


@dataclass
class RiskState:
    peak_equity: float = 0.0
    day: str = ""
    day_start_equity: float = 0.0
    halted: bool = False                 # kill switch tripped (manual reset required)
    halt_reason: str = ""
    cooldown: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict) -> "RiskState":
        return cls(**d)


class RiskManager:
    def __init__(self, cfg: RiskConfig, state: RiskState | None = None):
        self.cfg = cfg
        self.state = state or RiskState()

    # ----- account level -------------------------------------------------
    def update_equity(self, equity: float, today: date) -> bool:
        """Track peak / daily equity. Returns True if the kill switch just tripped."""
        s = self.state
        s.peak_equity = max(s.peak_equity, equity)
        if s.day != today.isoformat():
            s.day = today.isoformat()
            s.day_start_equity = equity
        if not s.halted and s.peak_equity > 0:
            dd = (s.peak_equity - equity) / s.peak_equity * 100
            if dd >= self.cfg.max_drawdown_pct:
                s.halted = True
                s.halt_reason = f"max drawdown {dd:.1f}% >= {self.cfg.max_drawdown_pct}%"
                return True
        return False

    def daily_loss_hit(self, equity: float) -> bool:
        s = self.state
        if s.day_start_equity <= 0:
            return False
        loss = (s.day_start_equity - equity) / s.day_start_equity * 100
        return loss >= self.cfg.max_daily_loss_pct

    def can_open(self, symbol: str, equity: float, open_positions: int) -> tuple[bool, str]:
        if self.state.halted:
            return False, f"halted: {self.state.halt_reason}"
        if self.daily_loss_hit(equity):
            return False, "daily loss limit reached"
        if open_positions >= self.cfg.max_open_positions:
            return False, "max open positions"
        if self.state.cooldown.get(symbol, 0) > 0:
            return False, "cooldown after loss"
        return True, ""

    def tick_cooldowns(self, symbol: str) -> None:
        if self.state.cooldown.get(symbol, 0) > 0:
            self.state.cooldown[symbol] -= 1

    def record_close(self, symbol: str, pnl: float) -> None:
        if pnl < 0 and self.cfg.cooldown_bars > 0:
            self.state.cooldown[symbol] = self.cfg.cooldown_bars

    # ----- trade level ---------------------------------------------------
    def size(self, equity: float, cash: float, price: float, atr: float, fee_rate: float) -> float:
        """Quantity such that hitting the stop loses ~risk_per_trade_pct of equity."""
        if not (atr > 0 and price > 0 and math.isfinite(atr)):
            return 0.0
        stop_dist = self.cfg.atr_stop_mult * atr
        risk_cash = equity * self.cfg.risk_per_trade_pct / 100
        # Include round-trip fees in the per-unit loss.
        per_unit_loss = stop_dist + 2 * fee_rate * price
        qty = risk_cash / per_unit_loss
        max_notional = min(equity * self.cfg.max_position_pct / 100, cash / (1 + fee_rate))
        return max(0.0, min(qty, max_notional / price))

    def new_position(self, symbol: str, qty: float, fill: float, atr: float, ts: str) -> Position:
        dist = self.cfg.atr_stop_mult * atr
        tp = fill + self.cfg.reward_risk * dist if self.cfg.reward_risk > 0 else None
        return Position(symbol, qty, fill, fill - dist, tp, fill, ts, dist)

    def trail(self, pos: Position, close: float, atr: float) -> None:
        """Ratchet the stop up behind price. Stops never move down."""
        pos.highest = max(pos.highest, close)
        if self.cfg.atr_trail_mult > 0 and atr > 0:
            pos.stop = max(pos.stop, pos.highest - self.cfg.atr_trail_mult * atr)

    @staticmethod
    def check_exit(pos: Position, open_: float, low: float, high: float) -> tuple[str, float] | None:
        """Did this bar touch the stop or take-profit? If both, assume the stop
        (the conservative choice when intrabar order is unknown). A gap through
        a level fills at the open, not at the level."""
        if low <= pos.stop:
            return "stop", min(open_, pos.stop)
        if pos.take_profit is not None and high >= pos.take_profit:
            return "take_profit", max(open_, pos.take_profit)
        return None
