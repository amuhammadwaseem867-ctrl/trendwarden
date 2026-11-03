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


@pytest.mark.parametrize("name", list(STRATEGIES))
def test_no_lookahead(df, name):
    """Changing future candles must not change any decision made before them."""
    cut = 1800
    base = run_backtest({"A/USDT": df}, name, {}, RiskConfig(), "1h")
    tampered = df.copy()
    tampered.iloc[cut:, :4] *= np.linspace(0.5, 2.0, len(df) - cut)[:, None]
    alt = run_backtest({"A/USDT": tampered}, name, {}, RiskConfig(), "1h")
    cutoff = df.index[cut].isoformat()
    a = base.trades[base.trades.exit_time < cutoff]
    b = alt.trades[alt.trades.exit_time < cutoff]
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)


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
    monkeypatch.setattr(eng, "make_market", lambda *a, **k: fake)
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


# ---------------------------------------------------------------- small accounts / margin
def test_margin_broker_accounting():
    b = SimBroker(80, fee_rate=0.0, slippage_rate=0.0, leverage=100, min_qty=1, qty_step=1)
    f = b.buy("GOLD", 1.7, 2400)              # rounds down to 1 oz; margin 24
    assert f.qty == 1 and b.cash == 80
    assert b.equity_at({"GOLD": 2390}) == pytest.approx(70)
    assert b.buy("GOLD", 3, 2400) is None      # would need 72 more margin, only 56 free
    b.sell("GOLD", 1, 2410)
    assert b.cash == pytest.approx(90) and not b.holdings
    assert SimBroker(80, 0, 0, leverage=20).buy("GOLD", 1, 2400) is None  # margin 120 > 80


def _small_trader(cap):
    from bot.trader import Trader
    broker = SimBroker(80, fee_rate=0.0, slippage_rate=0.0, leverage=100, min_qty=1, qty_step=1)
    return Trader(broker, RiskManager(RiskConfig(risk_per_trade_pct=2, min_lot_max_risk_pct=cap)))


def test_small_account_uses_minimum_size_within_cap():
    t = _small_trader(cap=10)
    t.open("GOLD", 2400, atr=3, ts="t", prices={"GOLD": 2400})   # 1 oz x $6 stop = 7.5% of $80
    assert t.positions["GOLD"].qty == 1


@pytest.mark.parametrize("cap", [0, 5])
def test_small_account_skips_when_minimum_too_risky(cap):
    t = _small_trader(cap)
    t.open("GOLD", 2400, atr=3, ts="t", prices={"GOLD": 2400})
    assert not t.positions


def test_small_account_backtest_accounts_cleanly():
    gold = synthetic_ohlcv(n=3000, seed=11, start_price=300, freq="15m")  # calmer than synthetic 2400
    res = run_backtest({"GOLD": gold}, "trend_momentum", {},
                       RiskConfig(risk_per_trade_pct=2, min_lot_max_risk_pct=10, atr_stop_mult=1.5,
                                  max_drawdown_pct=90, max_daily_loss_pct=90),
                       "15m", starting_cash=80, fee_rate=0, slippage_rate=0.0002,
                       leverage=100, min_qty=1, qty_step=1)
    assert res.metrics["trades"] > 0
    assert (res.trades.qty == res.trades.qty.round()).all()          # whole ounces only
    assert res.metrics["final_equity"] == pytest.approx(80 + res.trades.pnl.sum(), abs=0.01)


# ---------------------------------------------------------------- new strategies / discovery
def test_supertrend_follows_trend():
    idx = pd.date_range("2024-01-01", periods=300, freq="1h", tz="UTC")
    up = np.linspace(100, 200, 150)
    down = np.linspace(200, 120, 150)
    close = np.concatenate([up, down])
    df_ = pd.DataFrame({"open": close, "high": close + 0.5, "low": close - 0.5, "close": close, "volume": 1}, index=idx)
    _, direction = ta.supertrend(df_, 10, 3.0)
    assert direction.iloc[140] == 1 and direction.iloc[-1] == -1


def test_macd_matches_definition(df):
    line, sig, hist = ta.macd(df["close"])
    expected = ta.ema(df["close"], 12) - ta.ema(df["close"], 26)
    pd.testing.assert_series_equal(line, expected)
    assert (hist.dropna() == (line - sig).dropna()).all()


def test_ensemble_needs_votes(df):
    one = build_strategy("ensemble", {"min_votes": 1}).generate(df)
    all3 = build_strategy("ensemble", {"min_votes": 3}).generate(df)
    assert one["bull"].sum() >= all3["bull"].sum()
    assert (all3.loc[all3["bull"], "votes"] == 3).all()
    with pytest.raises(ValueError):
        build_strategy("ensemble", {"members": ["mean_reversion"]})


def test_walk_forward_and_discover(df):
    from bot.backtest import discover, walk_forward
    wf = walk_forward({"A/USDT": df}, "supertrend", {"st_mult": [2.0, 3.0]}, {}, RiskConfig(), "1h", folds=3)
    assert len(wf.folds) == 3 and "oos_return_pct" in wf.summary
    assert wf.robust == (not wf.reasons)
    ranked = discover({"A/USDT": df}, {"supertrend": {"st_mult": [3.0]}, "macd_trend": {"fast": [12]}},
                      RiskConfig(), "1h", folds=2, min_trades=1)
    assert {r.strategy for r in ranked} == {"supertrend", "macd_trend"}
    assert [r.robust for r in ranked] == sorted([r.robust for r in ranked], reverse=True)


def test_coverage_detects_gaps():
    from bot.data import coverage
    full = synthetic_ohlcv(n=2000, seed=1)
    assert coverage(full, "1h") > 0.95
    assert coverage(full.iloc[::5], "1h") < 0.3
    weekdays = full[full.index.dayofweek < 5]
    assert coverage(weekdays, "1h") > 0.95  # weekend closure is not a gap


def test_history_cache_roundtrip_and_gappy_cache_redownloaded(tmp_path):
    from bot.data import fetch_history, load_csv
    full = synthetic_ohlcv(n=3000, seed=4)

    class Ex:
        id = "fake"
        rateLimit = 0
        calls = []

        def milliseconds(self):
            return full.index[-1].value // 1_000_000 + 3_600_000 + 1000

        def fetch_ohlcv(self, symbol, tf, since=None, limit=1000):
            self.calls.append(since)
            ms = np.array([t.value // 1_000_000 for t in full.index])  # unit-independent
            part = full[ms >= since].iloc[:limit]
            return [[t.value // 1_000_000, *r] for t, r in zip(part.index, part.values.tolist())]

    ex = Ex()
    first = fetch_history(ex, "A/B", "1h", days=100, cache_dir=tmp_path)
    cache = next(tmp_path.glob("*.csv"))
    reloaded = load_csv(cache)                      # string timestamps under pandas 3
    pd.testing.assert_index_equal(reloaded.index, first.index)

    first.iloc[::4].to_csv(cache)                   # simulate an old gappy download
    ex.calls.clear()
    again = fetch_history(ex, "A/B", "1h", days=100, cache_dir=tmp_path)
    assert len(again) == len(first)                 # gaps were refilled from the exchange
