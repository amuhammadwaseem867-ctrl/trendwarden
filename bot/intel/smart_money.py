"""Smart money: are big funds adding to or cutting their gold ETF holdings?

Reads each fund's two latest 13F filings on SEC EDGAR (via the trader-copycat
tool) and compares their physical-gold ETF shares (SPDR Gold, iShares Gold,
abrdn, Sprott, GraniteShares...).

Limits, stated plainly: 13Fs are quarterly snapshots filed up to 45 days after
the quarter ends, so this is a SLOW signal about positioning weeks or months
ago. It is used as a background bias, never as an entry trigger.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .skills import load

log = logging.getLogger(__name__)

GOLD_ETF_ISSUER = re.compile(
    r"SPDR GOLD|ISHARES GOLD|ABRDN GOLD|ABERDEEN STANDARD GOLD|ABRDN PRECIOUS|SPROTT PHYSICAL GOLD|"
    r"GRANITESHARES GOLD|GOLDMAN SACHS PHYSICAL GOLD|\bGOLD (TR|TRUST|SHS|ETF)\b|\bGOLD MINISHARES\b",
    re.I,
)
MINERS = re.compile(r"MINERS|MINING|\bCORP\b|\bINC\b|\bLTD\b", re.I)

DEFAULT_FUNDS = {
    "0001350694": "Bridgewater (Ray Dalio)",
    "0001029160": "Soros Fund Management",
    "0001536411": "Duquesne (Druckenmiller)",
    "0001037389": "Renaissance Technologies",
    "0001791786": "Elliott Investment Management",
    "0001040273": "Third Point (Dan Loeb)",
    "0001656456": "Appaloosa (David Tepper)",
}


@dataclass
class SmartMoneyConfig:
    enabled: bool = False
    mode: str = "info"            # info = report only | filter = no new buys while bearish
    funds: dict = field(default_factory=lambda: dict(DEFAULT_FUNDS))
    refresh_days: int = 7

    @classmethod
    def from_dict(cls, d: dict | None) -> "SmartMoneyConfig":
        d = dict(d or {})
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown intel.smart_money settings: {sorted(unknown)}")
        cfg = cls(**d)
        if cfg.mode not in ("info", "filter"):
            raise ValueError("intel.smart_money.mode must be 'info' or 'filter'")
        return cfg


def is_gold_etf(position: dict) -> bool:
    name = f"{position.get('issuer', '')} {position.get('title', '')}"
    return (not position.get("put_call") and position.get("share_type", "SH") == "SH"
            and bool(GOLD_ETF_ISSUER.search(name)) and not MINERS.search(position.get("issuer", "")))


def gold_shares(rows: list[dict]) -> tuple[float, float]:
    """(shares, value) of physical gold ETFs in one 13F information table."""
    edgar = load("trader-copycat", "edgar13f")
    positions, _ = edgar.aggregate(rows)
    gold = [p for p in positions if is_gold_etf(p)]
    return sum(p["shares"] for p in gold), sum(p["value"] for p in gold)


def classify(now_shares: float, before_shares: float) -> str:
    if now_shares == before_shares:
        return "none" if now_shares == 0 else "unchanged"
    if before_shares == 0:
        return "new"
    if now_shares == 0:
        return "sold out"
    change = (now_shares - before_shares) / before_shares
    if abs(change) < 0.05:
        return "unchanged"
    return "added" if change > 0 else "trimmed"


def summarise(funds: list[dict]) -> dict:
    up = sum(f["status"] in ("new", "added") for f in funds)
    down = sum(f["status"] in ("sold out", "trimmed") for f in funds)
    bias = "bullish" if up > down else "bearish" if down > up else "neutral"
    return {"bias": bias, "adding": up, "cutting": down, "funds": funds}


def fetch(cfg: SmartMoneyConfig, user_agent: str | None) -> dict:
    edgar = load("trader-copycat", "edgar13f")
    user_agent = edgar.user_agent(user_agent, config_path=None)  # rejects the example placeholder
    if not user_agent:
        raise RuntimeError("SEC requires a contact: set SEC_USER_AGENT=\"Your Name you@email.com\" in .env")
    fetcher = edgar.Fetcher(user_agent)
    funds = []
    for cik, label in cfg.funds.items():
        try:
            meta, rows, _ = edgar.load_quarter(fetcher, cik)
            prev_meta, prev_rows, _ = edgar.load_quarter(fetcher, cik, before=meta["period"])
        except Exception as exc:
            log.warning("Smart money: %s skipped (%s)", label, exc)
            continue
        now_sh, now_val = gold_shares(rows)
        before_sh, _ = gold_shares(prev_rows)
        funds.append({"fund": label, "cik": cik, "period": meta["period"], "filed": meta["filed"],
                      "gold_etf_shares": now_sh, "gold_etf_value": now_val,
                      "previous_shares": before_sh, "status": classify(now_sh, before_sh)})
    result = summarise(funds)
    result["checked_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return result


class SmartMoney:
    def __init__(self, cfg: SmartMoneyConfig, state_dir: str):
        self.cfg = cfg
        self.path = Path(state_dir) / "smart_money.json"
        self.report: dict | None = None
        if self.path.exists():
            try:
                self.report = json.loads(self.path.read_text())
            except ValueError:
                self.report = None

    def stale(self) -> bool:
        if not self.report:
            return True
        checked = datetime.fromisoformat(self.report["checked_at"])
        return datetime.now(timezone.utc) - checked > timedelta(days=self.cfg.refresh_days)

    def refresh(self, force: bool = False) -> dict | None:
        if not (force or self.stale()):
            return self.report
        try:
            self.report = fetch(self.cfg, os.getenv("SEC_USER_AGENT"))
        except Exception as exc:
            log.warning("Smart money: refresh failed (%s); keeping the last report", exc)
            return self.report
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.report, indent=2))
        log.info("Smart money: gold ETF bias %s (%d funds adding, %d cutting)",
                 self.report["bias"].upper(), self.report["adding"], self.report["cutting"])
        return self.report

    def block_reason(self) -> str | None:
        if not (self.cfg.enabled and self.cfg.mode == "filter" and self.report):
            return None
        if self.report["bias"] == "bearish":
            return (f"smart money bearish on gold ({self.report['cutting']} funds cutting gold ETFs, "
                    f"{self.report['adding']} adding)")
        return None


def render(report: dict) -> str:
    lines = [f"Gold ETF positioning of big funds (13F): bias {report['bias'].upper()} "
             f"- {report['adding']} adding, {report['cutting']} cutting  (checked {report['checked_at']})", ""]
    for f in report["funds"]:
        lines.append(f"  {f['fund']:<34} {f['status']:<10} {f['gold_etf_shares']:>14,.0f} shares "
                     f"(was {f['previous_shares']:,.0f})  quarter {f['period']}, filed {f['filed']}")
    lines.append("\n13F filings are quarterly and up to 45 days old: a slow background signal, not a trade trigger.")
    return "\n".join(lines)
