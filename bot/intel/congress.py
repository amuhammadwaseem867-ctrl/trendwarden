"""Congress trades: alert when a member of the House discloses a trade in a
gold ETF or gold miner (via the trader-pelosi tool, House Clerk PTRs).

Reading the PDFs needs `pdftotext` (Poppler). Without it you still get an alert
for each new filing, with a link to read it yourself.
These disclosures arrive days or weeks after the trade. Information only:
alerts never change the bot's trades.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from .skills import load

log = logging.getLogger(__name__)

GOLD_TICKERS = {
    "GLD", "IAU", "GLDM", "IAUM", "SGOL", "BAR", "AAAU", "OUNZ", "PHYS",      # gold ETFs / trusts
    "GDX", "GDXJ", "RING", "SGDM",                                             # miner ETFs
    "NEM", "GOLD", "AEM", "FNV", "WPM", "RGLD", "KGC", "AU", "AGI", "HMY", "BTG", "IAG", "EGO", "GFI",
}
GOLD_WORDS = re.compile(r"\bgold\b|\bbullion\b", re.I)


@dataclass
class CongressConfig:
    enabled: bool = False
    poll_hours: float = 6
    members: list[str] = field(default_factory=list)   # empty = everyone
    extra_tickers: list[str] = field(default_factory=list)
    max_pdfs_per_check: int = 25

    @classmethod
    def from_dict(cls, d: dict | None) -> "CongressConfig":
        d = dict(d or {})
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown intel.congress settings: {sorted(unknown)}")
        return cls(**d)


def gold_rows(report: dict, extra: set[str] = frozenset()) -> list[dict]:
    tickers = GOLD_TICKERS | {t.upper() for t in extra}
    return [r for r in report.get("rows", [])
            if (r.get("ticker") or "").upper() in tickers or GOLD_WORDS.search(r.get("asset") or "")]


class CongressWatch:
    def __init__(self, cfg: CongressConfig, state_dir: str, notify=None):
        self.cfg = cfg
        self.state_path = str(Path(state_dir) / "congress.json")
        self.pdf_dir = str(Path(state_dir) / "congress_pdfs")
        self.notify = notify or (lambda msg: None)
        self._last: datetime | None = None
        self._warned_pdftotext = False

    def due(self, now: datetime) -> bool:
        return self.cfg.enabled and (self._last is None or now - self._last >= timedelta(hours=self.cfg.poll_hours))

    def check(self, now: datetime | None = None, today: date | None = None) -> list[dict]:
        """Look for new PTRs; returns the gold-related trades found (and alerts on them)."""
        now = now or datetime.now(timezone.utc)
        today = today or now.date()
        self._last = now
        ptr = load("trader-pelosi", "ptr")
        state = ptr.load_state(self.state_path)
        try:
            rows = ptr.select(ptr.load_index(today.year), members=self.cfg.members or None, kind="P")
        except Exception as exc:
            log.warning("Congress: could not read the House disclosure index (%s)", exc)
            return []
        new = ptr.new_since(rows, state, today)
        found = []
        for r in new[: self.cfg.max_pdfs_per_check]:
            try:
                _, report = ptr._fetch_and_read(r["doc_id"], r["year"], self.pdf_dir, filed=r["filed"])
            except Exception as exc:
                log.warning("Congress: could not read filing %s by %s (%s)", r["doc_id"], r["name"], exc)
                continue
            state.setdefault("seen", []).append(r["doc_id"])
            if report is None:
                if not self._warned_pdftotext:
                    log.warning("Congress: install Poppler (pdftotext) to scan filings for gold trades. "
                                "Until then you only get a link for each new filing.")
                    self._warned_pdftotext = True
                log.info("Congress: new filing by %s (%s) %s", r["name"], r["filed"], r["url"])
                continue
            for row in gold_rows(report, set(self.cfg.extra_tickers)):
                row = dict(row, member=r["name"], filed=r["filed"], url=r["url"])
                found.append(row)
                msg = (f"Congress gold trade: {r['name']} - {row.get('type_name') or row.get('type')} "
                       f"{row.get('asset')} ({row.get('ticker') or '-'}) {row.get('amount') or ''} "
                       f"on {row.get('date')}, disclosed {r['filed']}. {r['url']}")
                log.warning(msg)
                self.notify(msg)
        state["last_check"] = now.isoformat(timespec="seconds")
        ptr.save_state(self.state_path, state)
        return found
