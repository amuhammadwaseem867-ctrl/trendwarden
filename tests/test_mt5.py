"""MT5 integration tested against a fake MetaTrader5 module (the real package
only installs on Windows next to a running terminal)."""
import sys
import types
from types import SimpleNamespace as NS

import numpy as np
import pytest

from bot.data import synthetic_ohlcv


class FakeMT5(types.ModuleType):
    TIMEFRAME_M1, TIMEFRAME_H1, TIMEFRAME_H4, TIMEFRAME_D1 = 1, 16385, 16388, 16408
    TRADE_ACTION_DEAL, TRADE_ACTION_SLTP = 1, 6
    ORDER_TYPE_BUY, ORDER_TYPE_SELL = 0, 1
    ORDER_TIME_GTC = 0
    ORDER_FILLING_FOK, ORDER_FILLING_IOC, ORDER_FILLING_RETURN = 0, 1, 2
    TRADE_RETCODE_DONE = 10009
    DEAL_ENTRY_OUT = 1
    ACCOUNT_TRADE_MODE_DEMO = 0

    def __init__(self):
        super().__init__("MetaTrader5")
        self.df = synthetic_ohlcv(n=600, seed=5, start_price=2400, freq="1h")
        self.i = 500
        self.positions = []
        self.requests = []
        self.equity = 100_000.0
        self.algo = True
        self.closed = []

    # connection / account
    def initialize(self, **kw): return True
    def last_error(self): return (1, "ok")
    def account_info(self):
        return NS(login=123, server="XMGlobal-MT5 Demo", leverage=500, equity=self.equity,
                  margin_free=self.equity, currency="USD", trade_mode=0)
    def terminal_info(self): return NS(trade_allowed=self.algo)

    # symbols
    def symbol_info(self, s):
        if s != "GOLD":
            return None
        return NS(name="GOLD", visible=True, volume_min=0.01, volume_max=50, volume_step=0.01,
                  trade_contract_size=100, filling_mode=1, digits=2, point=0.01)
    def symbol_select(self, s, on): return True
    def symbols_get(self, group): return [NS(name="GOLD")]
    def symbol_info_tick(self, s):
        ts = self.df.index[self.i]
        px = float(self.df["open"].iloc[self.i])
        return NS(bid=px, ask=px + 0.3, time=int(ts.timestamp()) + 1, time_msc=(int(ts.timestamp()) + 1) * 1000)

    # data
    def _rates(self, part):
        arr = np.zeros(len(part), dtype=[("time", "i8"), ("open", "f8"), ("high", "f8"), ("low", "f8"),
                                         ("close", "f8"), ("tick_volume", "i8")])
        arr["time"] = [int(t.timestamp()) for t in part.index]
        for c in ("open", "high", "low", "close"):
            arr[c] = part[c].values
        arr["tick_volume"] = 1
        return arr
    def copy_rates_from_pos(self, s, tf, start, count):
        return self._rates(self.df.iloc[max(0, self.i + 1 - count): self.i + 1])
    def copy_rates_range(self, s, tf, a, b):
        part = self.df.iloc[: self.i + 1]
        return self._rates(part[(part.index >= a) & (part.index <= b)])

    # trading
    def positions_get(self, symbol=None): return tuple(self.positions)
    def order_send(self, req):
        self.requests.append(req)
        if req["action"] == self.TRADE_ACTION_DEAL and req["type"] == self.ORDER_TYPE_BUY:
            self.positions.append(NS(ticket=len(self.requests) + 100, volume=req["volume"], magic=req["magic"], sl=0.0, tp=0.0))
        elif req["action"] == self.TRADE_ACTION_DEAL:
            self.positions = [p for p in self.positions if p.ticket != req["position"]]
        elif req["action"] == self.TRADE_ACTION_SLTP:
            for p in self.positions:
                p.sl, p.tp = req["sl"], req["tp"]
        return NS(retcode=self.TRADE_RETCODE_DONE, price=req.get("price", 0), comment="done")
    def history_deals_get(self, a, b, group=None):
        old = NS(magic=240611, entry=1, price=1.0, position_id=1, time_msc=1)  # some earlier trade
        return [old] + [NS(magic=240611, entry=1, price=p.sl, position_id=p.ticket, time_msc=2)
                        for p in self.closed]


@pytest.fixture
def mt5(monkeypatch):
    fake = FakeMT5()
    monkeypatch.setitem(sys.modules, "MetaTrader5", fake)
    return fake


def test_lot_sizing_rounds_down_and_respects_minimum(mt5):
    from bot.mt5 import MT5Broker
    b = MT5Broker(mt5)
    fill = b.buy("GOLD", qty=3.789, price=2400)          # 3.789 oz = 0.03789 lots -> 0.03
    assert mt5.requests[-1]["volume"] == pytest.approx(0.03)
    assert fill.qty == pytest.approx(3.0)
    assert b.buy("GOLD", qty=0.5, price=2400) is None    # 0.005 lots < 0.01 minimum


def test_server_side_stop_and_close(mt5):
    from bot.mt5 import MT5Broker
    b = MT5Broker(mt5)
    b.buy("GOLD", 10, 2400)
    b.protect("GOLD", 2370.123, 2475.0)
    assert mt5.positions[0].sl == 2370.12 and mt5.positions[0].tp == 2475.0
    assert b.position_open("GOLD")
    b.sell("GOLD", 10, 2410)
    assert not b.position_open("GOLD")


def test_algo_trading_off_is_refused(mt5):
    from bot.mt5 import MT5Broker
    mt5.algo = False
    with pytest.raises(SystemExit):
        MT5Broker(mt5)


def test_unknown_symbol_lists_gold_names(mt5):
    from bot.mt5 import MT5Market
    with pytest.raises(SystemExit, match="GOLD"):
        MT5Market(mt5, ["XAUUSD"])


def test_market_data_and_history(mt5):
    from bot.data import fetch_history
    from bot.mt5 import MT5Market
    m = MT5Market(mt5, ["GOLD"])
    df = fetch_history(m, "GOLD", "1h", days=10, cache_dir=None)
    assert len(df) > 100 and df.index[-1] == mt5.df.index[mt5.i - 1]  # forming candle dropped
    assert m.fetch_ticker("GOLD")["last"] == pytest.approx(mt5.df["open"].iloc[mt5.i])


def test_live_engine_with_mt5(mt5, tmp_path, monkeypatch):
    from bot import engine as eng
    from bot.config import Config, ExecutionConfig
    from bot.risk import RiskConfig

    cfg = Config(platform="mt5", symbols=["GOLD"], timeframe="1h", strategy_name="breakout",
                 strategy_params={"entry_period": 20, "exit_period": 10, "trend_ema": 50},
                 risk=RiskConfig(risk_per_trade_pct=1.0, max_position_pct=200), execution=ExecutionConfig(fee_pct=0, slippage_pct=0),
                 state_dir=str(tmp_path), live_enabled=True)
    e = eng.Engine(cfg, "live")
    e.step()
    saw_stop_on_server = False
    for _ in range(95):
        mt5.i += 1
        e.step()
        if mt5.positions:
            saw_stop_on_server |= mt5.positions[0].sl > 0
            # simulate the broker's stop being hit while the bot isn't looking
            if mt5.i % 7 == 0:
                mt5.closed += mt5.positions
                mt5.positions.clear()
    assert saw_stop_on_server
    import pandas as pd
    trades = pd.read_csv(tmp_path / "live_trades.csv")
    broker_closed = trades[trades.reason.str.contains("broker")]
    assert len(broker_closed)
    # exit price comes from this position's own deal (its server stop), not an older trade
    assert (broker_closed.exit_price > 100).all()
    assert (broker_closed.r_multiple.abs() < 3).all()


def test_connect_tries_found_terminals_and_explains_failure(mt5, monkeypatch, tmp_path):
    from bot import mt5 as m
    exe = tmp_path / "terminal64.exe"
    exe.write_text("")
    tried = []

    def init(**kw):
        tried.append(kw.get("path"))
        return kw.get("path") == str(exe)

    monkeypatch.setattr(mt5, "initialize", init)
    monkeypatch.setattr(m, "find_terminals", lambda: [str(exe)])
    assert m.connect() is mt5
    assert tried == [None, str(exe)]  # auto-detect first, then the found terminal

    monkeypatch.setattr(m, "find_terminals", lambda: [])
    with pytest.raises(SystemExit, match="terminal64.exe"):
        m.connect()


def test_min_qty_and_margin_check(mt5):
    from bot.mt5 import MT5Broker
    b = MT5Broker(mt5)
    assert b.min_qty("GOLD") == pytest.approx(1.0)          # 0.01 lot x 100 oz
    mt5.order_calc_margin = lambda t, s, lots, px: lots * 100 * px / 500
    mt5.equity = 80
    assert b.buy("GOLD", 1, 2400) is not None               # needs $4.80 margin
    mt5.positions.clear()
    assert b.buy("GOLD", 50, 2400) is None                  # needs $240 margin, only $80


def test_history_when_symbol_has_no_tick_yet(mt5, monkeypatch):
    """Freshly added symbols report tick time 0; that used to make the start
    date negative and crash on Windows (OSError: [Errno 22])."""
    from bot.data import fetch_history
    from bot.mt5 import MT5Market, _utc
    m = MT5Market(mt5, ["GOLD"])
    monkeypatch.setattr(mt5, "symbol_info_tick", lambda s: NS(bid=0.0, ask=0.0, time=0, time_msc=0))
    assert m.milliseconds() > 1_600_000_000_000          # falls back to the latest bar
    df = fetch_history(m, "GOLD", "1h", days=10, cache_dir=None)
    assert len(df) > 100
    assert _utc(-5).year == 1970                          # never raises


def test_waits_for_mt5_to_finish_loading_history(mt5, monkeypatch):
    """MT5 returns partial history on the first request while it syncs."""
    import time as _time
    from bot.mt5 import MT5Market
    monkeypatch.setattr(_time, "sleep", lambda s: None)
    real = mt5.copy_rates_range
    calls = {"n": 0}

    def growing(s, tf, a, b):
        calls["n"] += 1
        full = real(s, tf, a, b)
        return full[-min(len(full), 50 * calls["n"]):]   # 50, 100, 150 ... candles

    monkeypatch.setattr(mt5, "copy_rates_range", growing)
    m = MT5Market(mt5, ["GOLD"])
    rows = m.fetch_ohlcv("GOLD", "1h", since=int(mt5.df.index[0].timestamp() * 1000))
    assert len(rows) == mt5.i + 1            # got everything, not just the first 50
    assert "GOLD" in m.gold_symbols()


def test_ipc_timeout_retries_then_explains(mt5, monkeypatch):
    import time as _time
    from bot import mt5 as m
    monkeypatch.setattr(_time, "sleep", lambda s: None)
    calls = []

    def init(**kw):
        calls.append(kw)
        return len(calls) == 2           # first attempt times out, retry works

    monkeypatch.setattr(mt5, "initialize", init)
    monkeypatch.setattr(mt5, "last_error", lambda: (-10005, "IPC timeout"))
    monkeypatch.setattr(m, "find_terminals", lambda: [])
    assert m.connect() is mt5 and len(calls) == 2 and calls[0]["timeout"] == 30_000

    calls.clear()
    monkeypatch.setattr(mt5, "initialize", lambda **kw: calls.append(kw) or False)
    monkeypatch.setattr(m, "find_terminals", lambda: ["a/terminal64.exe", "b/terminal64.exe"])
    monkeypatch.setattr("os.path.exists", lambda p: True)
    with pytest.raises(SystemExit, match="Administrator"):
        m.connect()
    assert len(calls) == 4      # 3 places tried, only ONE retry in total (not one per place)
