"""News guard, smart money, Congress alerts and stock copy-trading (network faked)."""
import json
from datetime import date, datetime, timedelta, timezone


import pandas as pd
import pytest

from bot.backtest import run_backtest
from bot.broker import SimBroker
from bot.data import synthetic_ohlcv
from bot.intel import congress as cg
from bot.intel import smart_money as sm
from bot.intel.hub import Intel, parse
from bot.intel.news_guard import NewsGuard, NewsGuardConfig, first_friday
from bot.intel.skills import load
from bot.risk import RiskConfig, RiskManager
from bot.trader import Trader

UTC = timezone.utc


# news guard
def test_first_friday():
    assert first_friday(2026, 11) == date(2026, 11, 6)
    assert first_friday(2026, 5) == date(2026, 5, 1)


def test_schedule_blocks_jobs_report_and_fed():
    g = NewsGuard(NewsGuardConfig(enabled=True))
    assert g.block_reason(datetime(2026, 11, 6, 13, 30, tzinfo=UTC))           # 08:30 New York (EST)
    assert g.block_reason(datetime(2026, 11, 6, 15, 30, tzinfo=UTC)) is None
    assert "Fed" in g.block_reason(datetime(2026, 10, 28, 18, 0, tzinfo=UTC))   # 14:00 New York (EDT)
    assert NewsGuard(NewsGuardConfig(enabled=False)).block_reason(datetime(2026, 11, 6, 13, 30, tzinfo=UTC)) is None


def test_extra_events_and_backtest_filter_timezone():
    g = NewsGuard(NewsGuardConfig(enabled=True, jobs_report=False, fomc_dates=[],
                                  extra_events=["2026-11-12 08:30"]))
    f = g.backtest_filter("Europe/Athens")                                      # broker server time
    assert f(pd.Timestamp("2026-11-12 15:30", tz="UTC"))                       # 13:30 UTC
    assert f(pd.Timestamp("2026-11-12 13:30", tz="UTC")) is None               # that's 11:30 UTC


def test_headlines_pause_on_high_impact_news(monkeypatch):
    news = load("trader-news", "news")
    now = datetime(2026, 11, 3, 15, 0, tzinfo=UTC)
    fresh = (now - timedelta(minutes=10)).strftime("%a, %d %b %Y %H:%M:%S +0000")
    old = (now - timedelta(hours=5)).strftime("%a, %d %b %Y %H:%M:%S +0000")
    rss = f"""<rss><channel>
      <item><title>Powell signals surprise rate cut</title><link>a</link><guid>1</guid><pubDate>{fresh}</pubDate></item>
      <item><title>Gold miners hold steady</title><link>b</link><guid>2</guid><pubDate>{fresh}</pubDate></item>
      <item><title>Inflation report shocks markets</title><link>c</link><guid>3</guid><pubDate>{old}</pubDate></item>
    </channel></rss>""".encode()
    monkeypatch.setattr(news.Fetcher, "get", lambda self, url, name, sec=False: rss)
    g = NewsGuard(NewsGuardConfig(enabled=True, headline_symbols=["GC=F"]))
    g.poll_headlines(now)
    assert "Powell" in g.block_reason(now)
    assert g.block_reason(now + timedelta(minutes=55)) is None                 # 60 min after the headline


def test_backtest_respects_entry_filter():
    df = synthetic_ohlcv(n=2500, seed=3)
    free = run_backtest({"A": df}, "breakout", {}, RiskConfig(), "1h")
    blocked = run_backtest({"A": df}, "breakout", {}, RiskConfig(), "1h", entry_filter=lambda ts: "news")
    assert free.metrics["trades"] > 0 and blocked.metrics["trades"] == 0


def test_entry_block_and_breakeven():
    t = Trader(SimBroker(10_000, 0, 0), RiskManager(RiskConfig()))
    t.entry_block = "news guard: test"
    t.open("A", 100, atr=2, ts="t", prices={"A": 100})
    assert not t.positions
    t.entry_block = None
    t.open("A", 100, atr=2, ts="t", prices={"A": 100})
    entry = t.positions["A"].entry_price
    t.protect_breakeven({"A": entry + 5}, "t")
    assert t.positions["A"].stop == entry


def test_intel_config_parsing():
    cfg = parse({"news_guard": {"enabled": True, "fomc_dates": [date(2026, 12, 9)]},
                 "data_timezone": "Europe/Athens"})
    assert cfg["news_guard"].fomc_dates == ["2026-12-09"]                     # YAML date objects accepted
    with pytest.raises(ValueError):
        parse({"news_guard": {"extra_events": ["12 Nov 2026"]}})
    assert cfg["news_guard"].enabled and not cfg["smart_money"].enabled
    with pytest.raises(ValueError):
        parse({"news_gaurd": {}})
    with pytest.raises(ValueError):
        parse({"smart_money": {"mode": "yolo"}})
    assert not Intel(parse(None), "state").active


# smart money
def _row(issuer, title, shares, cusip):
    return {"issuer": issuer, "title": title, "cusip": cusip, "figi": "", "put_call": "", "share_type": "SH",
            "value": shares * 200.0, "shares": float(shares)}


def test_gold_etf_detection():
    assert sm.is_gold_etf({"issuer": "SPDR GOLD TR", "title": "GOLD SHS", "put_call": "", "share_type": "SH"})
    assert sm.is_gold_etf({"issuer": "ISHARES GOLD TR", "title": "ISHARES NEW", "put_call": "", "share_type": "SH"})
    assert not sm.is_gold_etf({"issuer": "BARRICK GOLD CORP", "title": "COM", "put_call": "", "share_type": "SH"})
    assert not sm.is_gold_etf({"issuer": "GOLDMAN SACHS GROUP INC", "title": "COM", "put_call": "", "share_type": "SH"})
    assert not sm.is_gold_etf({"issuer": "SPDR GOLD TR", "title": "GOLD SHS", "put_call": "PUT", "share_type": "SH"})


def test_classify_and_bias():
    assert sm.classify(110, 100) == "added" and sm.classify(80, 100) == "trimmed"
    assert sm.classify(100, 0) == "new" and sm.classify(0, 100) == "sold out"
    assert sm.classify(102, 100) == "unchanged" and sm.classify(0, 0) == "none"
    rep = sm.summarise([{"status": "added"}, {"status": "new"}, {"status": "trimmed"}])
    assert rep["bias"] == "bullish" and rep["adding"] == 2


def test_smart_money_fetch_and_filter(monkeypatch, tmp_path):
    edgar = load("trader-copycat", "edgar13f")
    quarters = {
        ("A", None): [_row("SPDR GOLD TR", "GOLD SHS", 1000, "1"), _row("APPLE INC", "COM", 50, "2")],
        ("A", "2026-06-30"): [_row("SPDR GOLD TR", "GOLD SHS", 2000, "1")],
        ("B", None): [_row("ISHARES GOLD TR", "SHS", 0, "3")],
        ("B", "2026-06-30"): [_row("ISHARES GOLD TR", "SHS", 500, "3")],
    }

    def load_quarter(fetcher, cik, period=None, accession=None, before=None):
        rows = quarters[(cik, before)]
        return {"period": "2026-06-30" if before is None else "2026-03-31", "filed": "2026-08-14"}, rows, []

    monkeypatch.setattr(edgar, "load_quarter", load_quarter)
    monkeypatch.setenv("SEC_USER_AGENT", "Test User test@example.org")
    cfg = sm.SmartMoneyConfig(enabled=True, mode="filter", funds={"A": "Fund A", "B": "Fund B"})
    s = sm.SmartMoney(cfg, str(tmp_path))
    report = s.refresh(force=True)
    assert report["bias"] == "bearish" and report["cutting"] == 2
    assert "bearish" in s.block_reason()
    assert (tmp_path / "smart_money.json").exists() and not sm.SmartMoney(cfg, str(tmp_path)).stale()
    assert "Fund A" in sm.render(report)
    s.cfg.mode = "info"
    assert s.block_reason() is None


# congress
def test_congress_alerts_on_gold_trades(monkeypatch, tmp_path):
    ptr = load("trader-pelosi", "ptr")
    index = [{"doc_id": "1", "type": "P", "name": "Hon. A", "last": "A", "first": "", "year": 2026,
              "filed": "2026-11-01", "url": "u1"},
             {"doc_id": "2", "type": "P", "name": "Hon. B", "last": "B", "first": "", "year": 2026,
              "filed": "2026-11-02", "url": "u2"}]
    reports = {"1": {"rows": [{"asset": "SPDR Gold Shares", "ticker": "GLD", "type": "P", "type_name": "purchase",
                               "amount": "$1,001 - $15,000", "date": "2026-10-20"},
                              {"asset": "Apple Inc", "ticker": "AAPL", "type": "S"}]},
               "2": {"rows": [{"asset": "Microsoft", "ticker": "MSFT", "type": "P"}]}}
    monkeypatch.setattr(ptr, "load_index", lambda year, offline=None: index)
    monkeypatch.setattr(ptr, "_fetch_and_read", lambda doc, year, d, filed=None: ("x.pdf", reports[doc]))
    alerts = []
    w = cg.CongressWatch(cg.CongressConfig(enabled=True), str(tmp_path), alerts.append)
    found = w.check(datetime(2026, 11, 3, tzinfo=UTC))
    assert [f["ticker"] for f in found] == ["GLD"] and "Hon. A" in alerts[0]
    assert w.check(datetime(2026, 11, 3, 12, tzinfo=UTC)) == []                # each filing alerts once


def test_congress_without_pdftotext(monkeypatch, tmp_path):
    ptr = load("trader-pelosi", "ptr")
    monkeypatch.setattr(ptr, "load_index", lambda year, offline=None: [
        {"doc_id": "9", "type": "P", "name": "Hon. C", "last": "C", "first": "", "year": 2026,
         "filed": "2026-11-01", "url": "u9"}])
    monkeypatch.setattr(ptr, "_fetch_and_read", lambda *a, **k: ("x.pdf", None))
    assert cg.CongressWatch(cg.CongressConfig(enabled=True), str(tmp_path)).check(datetime(2026, 11, 3, tzinfo=UTC)) == []


# stock copy-trading
class FakeAlpaca:
    paper = True

    def __init__(self, *a, **k):
        self.orders = []

    def clock(self):
        return {"is_open": True}

    def account(self):
        return {"cash": "1000", "equity": "1000", "account_number": "PA1"}

    def positions(self):
        return []

    def place_order(self, order):
        self.orders.append(order)
        return {"id": f"ord{len(self.orders)}"}

    def wait_fill(self, oid):
        return {"status": "filled", "filled_qty": "1.5", "filled_avg_price": "100"}


def test_copy_trading_places_only_confirmed_orders(monkeypatch, tmp_path):
    from bot import copytrade as ct

    monkeypatch.setattr(ct, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(ct, "TRADER", tmp_path / "trader")
    fake = FakeAlpaca()
    monkeypatch.setattr(ct, "Alpaca", lambda *a, **k: fake)
    ct.ensure_config(budget=1000)
    ct.run_tool("trader-budget", "size", "--weights", "AAPL=60,KO=40", "--write")
    ct.sync_from_alpaca(fake)
    ct.run_tool("trader-risk", "check")
    replies = iter(["yes", "no thanks"])
    placed = ct.execute(ask=lambda prompt: next(replies))
    assert len(placed) == 1 and len(fake.orders) == 1
    assert fake.orders[0]["symbol"] == "AAPL" and fake.orders[0]["side"] == "buy"
    log = json.loads((tmp_path / "trader" / "state" / "broker.json").read_text())
    assert [o["status"] for o in log["orders"]] == ["approved", "placed", "declined"]


def test_copy_refuses_paper_live_mismatch(monkeypatch, tmp_path):
    from bot import copytrade as ct

    monkeypatch.setattr(ct, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(ct, "TRADER", tmp_path / "trader")
    live = FakeAlpaca()
    live.paper = False
    monkeypatch.setattr(ct, "Alpaca", lambda *a, **k: live)
    ct.ensure_config(budget=1000)
    with pytest.raises(ct.CopyError, match="paper"):
        ct.execute(ask=lambda p: "yes")


def test_find_investor():
    from bot import copytrade as ct
    assert ct.find_investor("buffett")["cik"] == "0001067983"
    assert ct.find_investor("1067983")["cik"] == "0001067983"
    with pytest.raises(ct.CopyError):
        ct.find_investor("nobody-at-all-xyz")


def test_paper_engine_pauses_entries_during_news(monkeypatch, tmp_path):
    from bot import engine as eng
    from bot.config import Config, ExecutionConfig
    from test_bot import FakeExchange

    df = synthetic_ohlcv(n=2500, seed=3)
    fake = FakeExchange(df)
    monkeypatch.setattr(eng, "make_market", lambda *a, **k: fake)

    class AlwaysPaused:
        active = True

        def __init__(self, *a, **k):
            pass

        def update(self, now):
            pass

        def block_reason(self, now):
            return "news guard: test event"

        def news_pause_started(self, now):
            return False

    monkeypatch.setattr(eng, "Intel", AlwaysPaused)
    cfg = Config(symbols=["A/USDT"], timeframe="1h", strategy_name="breakout",
                 strategy_params={"entry_period": 20, "exit_period": 10, "trend_ema": 50},
                 execution=ExecutionConfig(starting_cash=10_000), state_dir=str(tmp_path))
    e = eng.Engine(cfg, "paper")
    for _ in range(300):
        fake.i += 1
        e.step()
    assert not e.trader.positions and e.trader.broker.cash == 10_000
