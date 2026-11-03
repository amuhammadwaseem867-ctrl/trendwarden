"""News guard: no new trades around the events that move gold the most.

Two sources:
1. A schedule of big US releases (known in advance, so it also works in
   backtests): US jobs report (first Friday, 8:30 New York time), Fed rate
   decisions (14:00 New York time, dates from the config) and any extra
   dates you add (e.g. CPI).
2. Breaking headlines for gold (Yahoo Finance RSS via the trader-news tool):
   a fresh headline about the Fed, inflation, jobs, war, sanctions... pauses
   new entries for a while. Live/paper only.

When a pause starts, open trades that are in profit can have their stop moved
up to break-even (`protect_profits`).
"""
from __future__ import annotations

import logging
import re
import urllib.parse
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone

from .skills import load

log = logging.getLogger(__name__)

# Fed (FOMC) rate decision days for 2026, 14:00 New York time. Check them at
# https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm and add
# later years in the config (intel.news_guard.fomc_dates).
FOMC_2026 = ["2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
             "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09"]

HIGH_IMPACT = re.compile(
    r"\bFed\b|\bFOMC\b|\bPowell\b|\brate (cut|hike|decision)s?\b|\binterest rates?\b|\bCPI\b|"
    r"\binflation\b|\bpayrolls?\b|\bjobs report\b|\bunemployment\b|\bPCE\b|\bGDP\b|\bwar\b|"
    r"\bmissile|\binvasion\b|\bsanctions?\b|\btariffs?\b|\bdefault\b|\bbank (run|collapse|failure)\b|"
    r"\bemergency\b|\bcentral bank\b",
    re.I,
)


def _ny():
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo("America/New_York")
    except Exception as exc:  # Windows without the tzdata package
        raise SystemExit("Time zone data missing. Run: pip install tzdata") from exc


@dataclass
class NewsGuardConfig:
    enabled: bool = False
    jobs_report: bool = True                # first Friday of each month, 08:30 New York
    fomc_dates: list[str] = field(default_factory=lambda: list(FOMC_2026))
    extra_events: list[str] = field(default_factory=list)  # "YYYY-MM-DD HH:MM" New York time, e.g. CPI
    minutes_before: int = 30
    minutes_after: int = 90
    headlines: bool = True                  # live/paper only
    headline_symbols: list[str] = field(default_factory=lambda: ["GC=F", "GLD"])
    headline_pause_minutes: int = 60
    headline_poll_minutes: int = 5
    protect_profits: bool = True            # move stops of winning trades to break-even
    in_backtest: bool = True                # apply the schedule in backtests/discover

    @classmethod
    def from_dict(cls, d: dict | None) -> "NewsGuardConfig":
        d = dict(d or {})
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown intel.news_guard settings: {sorted(unknown)}")
        # YAML turns unquoted 2026-01-28 into a date object: accept both.
        for key in ("fomc_dates", "extra_events"):
            if key in d:
                d[key] = [str(v) for v in (d[key] or [])]
        cfg = cls(**d)
        for s in cfg.fomc_dates:
            date.fromisoformat(s[:10])
        for s in cfg.extra_events:
            datetime.strptime(s, "%Y-%m-%d %H:%M")
        return cfg


def first_friday(year: int, month: int) -> date:
    d = date(year, month, 1)
    return d + timedelta(days=(4 - d.weekday()) % 7)


class NewsGuard:
    def __init__(self, cfg: NewsGuardConfig):
        self.cfg = cfg
        self.ny = _ny()
        self._headline_until: datetime | None = None
        self._headline_reason = ""
        self._last_poll: datetime | None = None
        self._seen: set[str] = set()
        self._was_paused = False

    # ----- scheduled events ---------------------------------------------
    def events_between(self, start: datetime, end: datetime) -> list[tuple[datetime, str]]:
        """Scheduled events (UTC) in [start, end]."""
        out = []
        if self.cfg.jobs_report:
            y, m = start.year, start.month
            while (y, m) <= (end.year, end.month):
                d = first_friday(y, m)
                out.append((datetime.combine(d, time(8, 30), self.ny), "US jobs report"))
                y, m = (y + 1, 1) if m == 12 else (y, m + 1)
        for s in self.cfg.fomc_dates:
            out.append((datetime.combine(date.fromisoformat(s), time(14, 0), self.ny), "Fed rate decision"))
        for s in self.cfg.extra_events:
            out.append((datetime.strptime(s, "%Y-%m-%d %H:%M").replace(tzinfo=self.ny), "scheduled event"))
        out = [(t.astimezone(timezone.utc), name) for t, name in out]
        return [(t, n) for t, n in out if start <= t <= end]

    def scheduled_block(self, now: datetime) -> str | None:
        before = timedelta(minutes=self.cfg.minutes_before)
        after = timedelta(minutes=self.cfg.minutes_after)
        for t, name in self.events_between(now - after - timedelta(days=1), now + before + timedelta(days=1)):
            if t - before <= now <= t + after:
                return f"{name} at {t.astimezone(self.ny):%H:%M} New York"
        return None

    def backtest_filter(self, tz_name: str = "UTC"):
        """A function bar_timestamp -> reason|None for the backtester. Bar times
        are interpreted in `tz_name` (MT5 bars are in the broker's server time)."""
        from zoneinfo import ZoneInfo

        tz = ZoneInfo(tz_name)
        before = timedelta(minutes=self.cfg.minutes_before)
        after = timedelta(minutes=self.cfg.minutes_after)

        cache: dict = {}

        def check(ts) -> str | None:
            if ts in cache:
                return cache[ts]
            cache[ts] = result = _check(ts)
            return result

        def _check(ts) -> str | None:
            t = ts.to_pydatetime().replace(tzinfo=tz).astimezone(timezone.utc)
            for ev, name in self.events_between(t - after - timedelta(hours=6), t + before + timedelta(hours=6)):
                if ev - before <= t <= ev + after:
                    return name
            return None

        return check

    # ----- headlines ------------------------------------------------------
    def poll_headlines(self, now: datetime) -> None:
        if not self.cfg.headlines:
            return
        if self._last_poll and now - self._last_poll < timedelta(minutes=self.cfg.headline_poll_minutes):
            return
        self._last_poll = now
        news = load("trader-news", "news")
        fetcher = news.Fetcher()
        for sym in self.cfg.headline_symbols:
            try:
                data = fetcher.get(news.YAHOO % urllib.parse.quote(sym), f"yahoo-{sym}.xml")
                items = news.parse_rss(data, sym)
            except Exception as exc:
                log.warning("News guard: could not read headlines for %s: %s", sym, exc)
                continue
            for it in items:
                if it["id"] in self._seen or it["time"] is None:
                    continue
                self._seen.add(it["id"])
                age = now - it["time"]
                window = timedelta(minutes=self.cfg.headline_pause_minutes)
                if age < window and HIGH_IMPACT.search(f"{it['title']} {it.get('summary') or ''}"):
                    until = it["time"] + window
                    if not self._headline_until or until > self._headline_until:
                        self._headline_until = until
                        self._headline_reason = f"headline: {it['title'][:90]}"
                        log.warning("News guard: high-impact headline (%s) - no new trades until %s UTC",
                                    it["title"][:120], until.strftime("%H:%M"))

    def headline_block(self, now: datetime) -> str | None:
        if self._headline_until and now < self._headline_until:
            return self._headline_reason
        return None

    # ----- combined -------------------------------------------------------
    def block_reason(self, now: datetime) -> str | None:
        if not self.cfg.enabled:
            return None
        return self.scheduled_block(now) or self.headline_block(now)

    def pause_started(self, paused: bool) -> bool:
        """True exactly once when a pause begins."""
        started = paused and not self._was_paused
        self._was_paused = paused
        return started
