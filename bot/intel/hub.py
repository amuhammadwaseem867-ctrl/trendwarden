"""Runs the intelligence features alongside the paper/live engine."""
from __future__ import annotations

import logging
from datetime import datetime

from .congress import CongressConfig, CongressWatch
from .news_guard import NewsGuard, NewsGuardConfig
from .smart_money import SmartMoney, SmartMoneyConfig

log = logging.getLogger(__name__)


def parse(raw: dict | None) -> dict:
    raw = dict(raw or {})
    unknown = set(raw) - {"news_guard", "smart_money", "congress", "data_timezone"}
    if unknown:
        raise ValueError(f"Unknown intel settings: {sorted(unknown)}")
    return {
        "news_guard": NewsGuardConfig.from_dict(raw.get("news_guard")),
        "smart_money": SmartMoneyConfig.from_dict(raw.get("smart_money")),
        "congress": CongressConfig.from_dict(raw.get("congress")),
        # Time zone of the candle timestamps, for the news schedule in backtests.
        # XM's MT5 server runs on GMT+2 (GMT+3 in summer): Europe/Athens matches it.
        "data_timezone": raw.get("data_timezone", "UTC"),
    }


class Intel:
    def __init__(self, cfg: dict, state_dir: str, notify=None):
        self.cfg = cfg
        self.news = NewsGuard(cfg["news_guard"]) if cfg["news_guard"].enabled else None
        self.smart = SmartMoney(cfg["smart_money"], state_dir) if cfg["smart_money"].enabled else None
        self.congress = CongressWatch(cfg["congress"], state_dir, notify) if cfg["congress"].enabled else None

    @property
    def active(self) -> bool:
        return any((self.news, self.smart, self.congress))

    def update(self, now: datetime) -> None:
        """Network polling. Failures are logged and never stop trading."""
        for name, fn in (("news", lambda: self.news and self.news.poll_headlines(now)),
                         ("smart money", lambda: self.smart and self.smart.refresh()),
                         ("congress", lambda: self.congress and self.congress.due(now) and self.congress.check(now))):
            try:
                fn()
            except Exception as exc:
                log.warning("Intel (%s) failed: %s", name, exc)

    def block_reason(self, now: datetime) -> str | None:
        reasons = []
        if self.news:
            r = self.news.block_reason(now)
            if r:
                reasons.append(f"news guard: {r}")
        if self.smart:
            r = self.smart.block_reason()
            if r:
                reasons.append(r)
        return "; ".join(reasons) or None

    def news_pause_started(self, now: datetime) -> bool:
        return bool(self.news) and self.news.cfg.protect_profits and \
            self.news.pause_started(self.news.block_reason(now) is not None)
