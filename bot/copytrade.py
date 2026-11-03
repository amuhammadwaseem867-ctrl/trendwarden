"""Stock copy-trading: copy a famous investor's 13F portfolio into an Alpaca
account, using the bundled Trader skill pack's tools for every step:

    plan       trader-copycat (13F -> targets)  ->  trader-budget (sizing)
               ->  trader-risk (limits)          ->  you review the orders
    rebalance  trader-rebalance (drift back to the targets) -> trader-risk
    execute    every order that PASSED risk is shown as a ticket and placed
               on Alpaca only after you type yes for THAT order (trader-broker
               records your exact reply). Paper trading by default.

All files live in trader/ (git-ignored), exactly where the skills expect them,
so you can also drive the same portfolio with /trader-* in Claude Code.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

from .alpaca import Alpaca, AlpacaError
from .intel.skills import REPO_ROOT, load, script

TRADER = REPO_ROOT / "trader"

DEFAULT_CONFIG = {
    "budget": 1000,
    "cash_buffer_pct": 10,
    "max_position_pct": 25,
    "max_order_usd": 2000,
    "min_order_usd": 5,
    "max_orders_per_day": 10,
    "fractional": True,
    "paper": True,
    "blocked": [],
    "rebalance_band_pct": 5,
    "allow_options": False,
    "allow_margin": False,
    "allow_short": False,
}


class CopyError(Exception):
    pass


def run_tool(skill: str, module: str, *args: str, check: bool = True) -> int:
    """Run one of the skill pack's CLI tools from the repo root (where trader/ lives)."""
    cmd = [sys.executable, str(script(skill, module)), *args]
    code = subprocess.run(cmd, cwd=REPO_ROOT).returncode
    if check and code not in (0, 1):  # 1 = "something blocked / nothing to do" for several tools
        raise CopyError(f"{skill} failed (exit {code})")
    return code


def ensure_config(budget: float | None = None) -> Path:
    TRADER.mkdir(exist_ok=True)
    path = TRADER / "config.json"
    cfg = json.loads(path.read_text()) if path.exists() else dict(DEFAULT_CONFIG)
    for k, v in DEFAULT_CONFIG.items():
        cfg.setdefault(k, v)
    if budget is not None:
        cfg["budget"] = budget
    ua = os.getenv("SEC_USER_AGENT")
    if ua and not cfg.get("sec_user_agent"):
        cfg["sec_user_agent"] = ua
    path.write_text(json.dumps(cfg, indent=2) + "\n")
    return path


def find_investor(name: str) -> dict:
    edgar = load("trader-copycat", "edgar13f")
    if name.replace("-", "").isdigit():
        return {"cik": edgar.cik10(name), "name": name}
    hits = edgar.find_bundled(name)
    if not hits:
        raise CopyError(f"No well-known filer matches '{name}'. Find its CIK with:\n"
                        f"  python .claude/skills/trader-copycat/edgar13f.py find \"{name}\" --sec\n"
                        f"then run: python main.py copy plan --investor <CIK>")
    return hits[0]


def sync_from_alpaca(api: Alpaca) -> dict:
    """Write trader/portfolio.json and trader/prices.json from the Alpaca account,
    keeping any tax lots trader-tax already recorded."""
    acct = api.account()
    positions = api.positions()
    path = TRADER / "portfolio.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    lots = {p["ticker"]: p.get("lots") for p in old.get("positions", []) if p.get("lots")}
    pf = {"cash": float(acct["cash"]),
          "positions": [{"ticker": p["symbol"], "shares": float(p["qty"]),
                         **({"lots": lots[p["symbol"]]} if p["symbol"] in lots else {})}
                        for p in positions if p.get("side", "long") == "long"],
          "source": f"alpaca {'paper' if api.paper else 'LIVE'} account {acct.get('account_number', '')}",
          "as_of": date.today().isoformat()}
    TRADER.mkdir(exist_ok=True)
    path.write_text(json.dumps(pf, indent=2) + "\n")
    prices = {"as_of": date.today().isoformat(), "source": "alpaca positions",
              "prices": {p["symbol"]: float(p["current_price"]) for p in positions}}
    (TRADER / "prices.json").write_text(json.dumps(prices, indent=2) + "\n")
    return {"equity": float(acct["equity"]), "cash": float(acct["cash"]), "positions": len(pf["positions"])}


def plan(investor: str, top: int, budget: float | None, use_alpaca: bool = True) -> None:
    filer = find_investor(investor)
    print(f"\n== 1/3  Copying {filer.get('name')} ({filer.get('person', '')}) - latest 13F, top {top}\n")
    ensure_config(budget)
    run_tool("trader-copycat", "edgar13f", "copy", filer["cik"], "--top", str(top))
    print("\n== 2/3  Sizing to your budget (trader/config.json)\n")
    run_tool("trader-budget", "size", "--write")
    if use_alpaca:
        try:
            info = sync_from_alpaca(Alpaca())
            print(f"\nAlpaca account: equity ${info['equity']:,.2f}, cash ${info['cash']:,.2f}, "
                  f"{info['positions']} positions")
        except AlpacaError as exc:
            print(f"\n(Alpaca not connected: {exc}\n Risk is checked against a fresh account funded with the budget.)")
    print("\n== 3/3  Risk check\n")
    run_tool("trader-risk", "check")
    print("\nReview the orders above. To place the ones that PASSED: python main.py copy execute")


def rebalance(sell_untracked: bool = False) -> None:
    sync_from_alpaca(Alpaca())
    args = ["--write"] + (["--sell-untracked"] if sell_untracked else [])
    if run_tool("trader-rebalance", "rebalance", *args) == 1:
        print("\nNothing has drifted past the band - no trades needed.")
        return
    run_tool("trader-risk", "check")
    print("\nTo place the orders that PASSED: python main.py copy execute")


def execute(ask=input) -> list[dict]:
    """Show each order; place it on Alpaca only after an explicit yes for that order."""
    ticket = load("trader-broker", "ticket")
    api = Alpaca()
    mode = "PAPER" if api.paper else "LIVE - REAL MONEY"
    cfg = json.loads((TRADER / "config.json").read_text())
    if bool(cfg.get("paper", True)) != api.paper:
        raise CopyError(f"trader/config.json says paper={cfg.get('paper', True)} but your Alpaca keys are for a "
                        f"{'paper' if api.paper else 'LIVE'} account (ALPACA_PAPER). Make them match first.")
    if not (TRADER / "orders.json").exists():
        raise CopyError("No orders yet. Run: python main.py copy plan --investor buffett")
    orders_doc = json.loads((TRADER / "orders.json").read_text())
    orders = orders_doc.get("orders", orders_doc) if isinstance(orders_doc, dict) else orders_doc
    clock = api.clock()
    if not clock.get("is_open"):
        print(f"Note: the US market is closed; market orders will fill at the next open ({clock.get('next_open')}).")
    placed = []
    for i in range(1, len(orders) + 1):
        shown = subprocess.run([sys.executable, str(script("trader-broker", "ticket")), "show",
                                "--index", str(i), "--json"], cwd=REPO_ROOT, capture_output=True, text=True)
        try:
            t = json.loads(shown.stdout)
        except ValueError:
            print(shown.stdout or shown.stderr)
            continue
        t = t[0] if isinstance(t, list) else t
        print(f"\n[{mode}] Order {i}/{len(orders)}: {t.get('plain')}  -  {t.get('status')}")
        if not t.get("ready"):
            print("  Skipped.")
            continue
        reply = ask(f"  Type 'yes' to place this {t.get('plain')} order, anything else to skip: ").strip()
        code = run_tool("trader-broker", "ticket", "approve", "--index", str(i), "--reply", reply, check=False)
        if code != 0:
            run_tool("trader-broker", "ticket", "decline", "--index", str(i), check=False)
            print("  Not placed.")
            continue
        try:
            res = api.place_order(ticket.tool_input(orders[i - 1]))
            fill = api.wait_fill(res["id"])
        except AlpacaError as exc:
            run_tool("trader-broker", "ticket", "result", "--index", str(i), "--status", "failed",
                     "--note", str(exc)[:200], check=False)
            print(f"  FAILED: {exc}")
            continue
        args = ["result", "--index", str(i), "--status", "placed", "--broker-id", res["id"]]
        if fill.get("status") == "filled":
            args += ["--filled-qty", str(fill["filled_qty"]), "--filled-price", str(fill["filled_avg_price"])]
        run_tool("trader-broker", "ticket", *args, check=False)
        print(f"  Placed on Alpaca ({fill.get('status')}), order id {res['id']}")
        placed.append(fill)
    return placed
