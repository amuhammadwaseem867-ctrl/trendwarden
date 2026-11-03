import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from bot import indicators as ta
from bot.backtest import run_backtest
from bot.broker import SimBroker
from bot.config import Config, ExecutionConfig
from bot.data import synthetic_ohlcv, timeframe_ms
from bot.risk import Position, RiskConfig, RiskManager
from bot.strategies import STRATEGIES, build_strategy


@pytest.fixture(scope="module")
def df():
    return synthetic_ohlcv(n=2500, seed=3)


# ---------------------------------------------------------------- indicators
def test_rsi_bounds(df):
    r = ta.rsi(df["close"]).dropna()
    assert len(r) and r.between(0, 100).all()


def test_atr_positive(df):
    assert (ta.atr(df).dropna() > 0).all()


def test_donchian_excludes_current_bar(df):
    _, upper = ta.donchian(df, 20)
    assert upper.iloc[100] == df["high"].iloc[80:100].max()


# ---------------------------------------------------------------- risk
def test_size_risks_configured_fraction():
    rm = RiskManager(RiskConfig(risk_per_trade_pct=1.0, atr_stop_mult=2.0, max_position_pct=100))
    qty = rm.size(equity=10_000, cash=10_000, price=100, atr=2.5, fee_rate=0.0)
    assert qty * 2 * 2.5 == pytest.approx(100)  # loses 1% of 10k at the stop


def test_size_capped_by_max_position():
    rm = RiskManager(RiskConfig(risk_per_trade_pct=5.0, max_position_pct=10))
    qty = rm.size(10_000, 10_000, price=100, atr=0.1, fee_rate=0.001)
    assert qty * 100 <= 1_000 + 1e-6


def test_trailing_stop_only_moves_up():
    rm = RiskManager(RiskConfig(atr_trail_mult=2.0))
    pos = Position("X", 1, 100, 95, None, 100, "t", 5)
    rm.trail(pos, 120, 2)
    assert pos.stop == 116
    rm.trail(pos, 110, 2)
    assert pos.stop == 116


def test_gap_down_fills_at_open():
    pos = Position("X", 1, 100, 95, 110, 100, "t", 5)
    assert RiskManager.check_exit(pos, open_=90, low=88, high=92) == ("stop", 90)
    assert RiskManager.check_exit(pos, open_=96, low=94, high=111) == ("stop", 95)  # both hit -> stop
    assert RiskManager.check_exit(pos, open_=112, low=111, high=115) == ("take_profit", 112)


def test_kill_switch_and_daily_limit():
    rm = RiskManager(RiskConfig(max_drawdown_pct=10, max_daily_loss_pct=3))
    assert not rm.update_equity(10_000, date(2024, 1, 1))
    assert rm.can_open("X", 9_600, 0) == (False, "daily loss limit reached")
    assert rm.update_equity(8_900, date(2024, 1, 2))
    assert rm.can_open("X", 8_900, 0)[0] is False


# ---------------------------------------------------------------- backtest
@pytest.mark.parametrize("name", list(STRATEGIES))
def test_backtest_runs_and_accounts_cleanly(df, name):
    res = run_backtest({"A/USDT": df}, name, {}, RiskConfig(), "1h", 10_000, 0.001, 0.0005)
    m = res.metrics
    assert m["trades"] > 0
    # All positions are closed at the end, so final equity = start + sum of trade PnL.
    assert m["final_equity"] == pytest.approx(10_000 + res.trades.pnl.sum(), abs=0.01)
    assert m["max_drawdown_pct"] <= 0


def test_no_lookahead(df):
    """Changing future candles must not change any decision made before them."""
    cut = 1800
    base = run_backtest({"A/USDT": df}, "trend_momentum", {}, RiskConfig(), "1h")
    tampered = df.copy()
    tampered.iloc[cut:, :4] *= np.linspace(0.5, 2.0, len(df) - cut)[:, None]
    alt = run_backtest({"A/USDT": tampered}, "trend_momentum", {}, RiskConfig(), "1h")
    cutoff = df.index[cut].isoformat()
    a = base.trades[base.trades.exit_time < cutoff]
    b = alt.trades[alt.trades.exit_time < cutoff]
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_multi_symbol_respects_max_positions():
    data = {f"S{i}/USDT": synthetic_ohlcv(n=2000, seed=i) for i in range(5)}
    res = run_backtest(data, "breakout", {}, RiskConfig(max_open_positions=2), "1h")
    t = res.trades
    events = sorted([(r.entry_time, 1) for r in t.itertuples()] + [(r.exit_time, -1) for r in t.itertuples()],
                    key=lambda e: (e[0], e[1]))
    open_count = peak = 0
    for _, d in events:
        open_count += d
        peak = max(peak, open_count)
    assert peak <= 2


def test_unknown_strategy_param_rejected():
    with pytest.raises(ValueError):
        build_strategy("trend_momentum", {"fats_ema": 3})


def test_sim_broker_fees():
    b = SimBroker(1000, fee_rate=0.01, slippage_rate=0.0)
    f = b.buy("X", 5, 100)
    assert f.fee == pytest.approx(5) and b.cash == pytest.approx(495)
    assert b.buy("X", 10, 100) is None  # not enough cash
    b.sell("X", 5, 110)
    assert b.cash == pytest.approx(495 + 550 - 5.5)


# ---------------------------------------------------------------- engine (paper) with a fake exchange
class FakeExchange:
    id = "fake"
    rateLimit = 0

    def __init__(self, df, tf="1h"):
        self.df = df
        self.step = timeframe_ms(tf)
        self.i = 400  # number of candles that have closed so far

    def milliseconds(self):
        # "now" is just after the close of candle i-1 (candle i is still forming)
        return self.df.index[self.i].value // 1_000_000 + 1000

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=500):
        part = self.df.iloc[max(0, self.i + 1 - limit): self.i + 1]
        return [[ts.value // 1_000_000, *row] for ts, row in zip(part.index, part.values.tolist())]

    def fetch_ticker(self, symbol):
        return {"last": float(self.df["open"].iloc[self.i])}


def test_paper_engine_runs_and_resumes(tmp_path, monkeypatch, df):
    from bot import engine as eng

    fake = FakeExchange(df)
    monkeypatch.setattr(eng, "make_exchange", lambda *a, **k: fake)
    cfg = Config(symbols=["A/USDT"], timeframe="1h", strategy_name="breakout",
                 strategy_params={"entry_period": 20, "exit_period": 10, "trend_ema": 50},
                 execution=ExecutionConfig(starting_cash=10_000), state_dir=str(tmp_path))
    e = eng.Engine(cfg, "paper")
    e.step()  # first run only records the starting candle
    assert e.last_bar["A/USDT"] == df.index[fake.i - 1].isoformat()
    opened = False
    for _ in range(600):
        fake.i += 1
        e.step()
        opened |= bool(e.trader.positions)
    assert opened
    assert (tmp_path / "paper_trades.csv").exists()

    # Restart: state is restored exactly.
    e2 = eng.Engine(cfg, "paper")
    assert e2.trader.broker.cash == pytest.approx(e.trader.broker.cash)
    assert set(e2.trader.positions) == set(e.trader.positions)
    assert e2.last_bar == e.last_bar
