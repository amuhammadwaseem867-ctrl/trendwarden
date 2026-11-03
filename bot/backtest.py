"""Event-driven backtester, performance metrics and a walk-forward optimiser."""
from __future__ import annotations

import itertools
import logging
import math
from dataclasses import asdict, dataclass, fields

import numpy as np
import pandas as pd

from .broker import SimBroker
from .data import timeframe_ms
from .risk import RiskConfig, RiskManager
from .strategies import build_strategy
from .trader import Trade, Trader

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: pd.DataFrame
    metrics: dict
    benchmark: dict


def run_backtest(data: dict[str, pd.DataFrame], strategy_name: str, strategy_params: dict,
                 risk_cfg: RiskConfig, timeframe: str, starting_cash: float = 10_000,
                 fee_rate: float = 0.001, slippage_rate: float = 0.0005,
                 start: pd.Timestamp | None = None, end: pd.Timestamp | None = None,
                 leverage: float = 1.0, min_qty: float = 0.0, qty_step: float = 0.0,
                 entry_filter=None) -> BacktestResult:
    """Simulate the strategy bar by bar.

    Signals are computed on the full history (they only use past data), and
    trading happens only between `start` and `end`. That lets the out-of-sample
    test use earlier bars to warm up indicators without trading on them.
    """
    strategy = build_strategy(strategy_name, strategy_params)
    prepared = {s: strategy.generate(df) for s, df in data.items()}
    timeline = sorted(set().union(*(df.index for df in prepared.values())))
    if start is not None:
        timeline = [t for t in timeline if t >= start]
    if end is not None:
        timeline = [t for t in timeline if t < end]

    broker = SimBroker(starting_cash, fee_rate, slippage_rate, leverage, min_qty, qty_step)
    trader = Trader(broker, RiskManager(risk_cfg))
    prices: dict[str, float] = {}
    curve = []

    cols = ["open", "high", "low", "close", "entry", "exit", "atr"]
    rows = {s: df[cols].to_dict("index") for s, df in prepared.items()}  # fast per-bar lookup
    for ts in timeline:
        ts_str = ts.isoformat()
        if entry_filter is not None:
            trader.entry_block = entry_filter(ts)
        for symbol, bars in rows.items():
            bar = bars.get(ts)
            if bar is None:
                continue
            prices[symbol] = float(bar["close"])
            trader.on_bar(symbol, bar, ts_str, prices)
        curve.append((ts, trader.after_bar(prices, ts.date(), ts_str)))

    if timeline:
        trader.flatten(prices, timeline[-1].isoformat(), "end_of_test")
        curve[-1] = (timeline[-1], trader.equity(prices))

    equity = pd.Series(dict(curve), dtype=float, name="equity")
    trades = pd.DataFrame([asdict(t) for t in trader.trades],
                          columns=[f.name for f in fields(Trade)])
    metrics = compute_metrics(equity, trades, timeframe, starting_cash)
    metrics["halted"] = trader.risk.state.halted
    bench = buy_and_hold(prepared, timeline, timeframe)
    return BacktestResult(equity, trades, metrics, bench)


def periods_per_year(timeframe: str) -> float:
    return 365 * 86_400_000 / timeframe_ms(timeframe)  # crypto trades 24/7


def compute_metrics(equity: pd.Series, trades: pd.DataFrame, timeframe: str, starting_cash: float) -> dict:
    if equity.empty:
        return {"trades": 0}
    rets = equity.pct_change().dropna()
    span_years = (equity.index[-1] - equity.index[0]).total_seconds() / (365.25 * 86400)
    if span_years > 0.05:
        years = span_years
        ppy = len(equity) / span_years  # actual bars per year (markets with weekends/closures)
    else:
        ppy = periods_per_year(timeframe)
        years = max(len(equity) / ppy, 1e-9)
    final = float(equity.iloc[-1])
    total = final / starting_cash - 1
    std = rets.std(ddof=1)
    downside = rets[rets < 0].std(ddof=1)
    dd = equity / equity.cummax() - 1

    m = {
        "start": str(equity.index[0]),
        "end": str(equity.index[-1]),
        "final_equity": round(final, 2),
        "total_return_pct": round(total * 100, 2),
        "cagr_pct": round(((final / starting_cash) ** (1 / years) - 1) * 100, 2) if final > 0 else -100.0,
        "max_drawdown_pct": round(float(dd.min()) * 100, 2),
        "sharpe": round(float(rets.mean() / std * math.sqrt(ppy)), 2) if std and std > 0 else 0.0,
        "sortino": round(float(rets.mean() / downside * math.sqrt(ppy)), 2) if downside and downside > 0 else 0.0,
        "trades": int(len(trades)),
    }
    if len(trades):
        wins = trades[trades.pnl > 0]
        losses = trades[trades.pnl <= 0]
        gross_loss = -losses.pnl.sum()
        m.update(
            win_rate_pct=round(len(wins) / len(trades) * 100, 1),
            profit_factor=round(wins.pnl.sum() / gross_loss, 2) if gross_loss > 0 else float("inf"),
            avg_r=round(float(trades.r_multiple.mean()), 2),
            best_trade=round(float(trades.pnl.max()), 2),
            worst_trade=round(float(trades.pnl.min()), 2),
        )
    calmar = m["cagr_pct"] / abs(m["max_drawdown_pct"]) if m["max_drawdown_pct"] < 0 else 0.0
    m["calmar"] = round(calmar, 2)
    return m


def buy_and_hold(prepared: dict[str, pd.DataFrame], timeline: list, timeframe: str) -> dict:
    """Equal-weight buy-and-hold of the same symbols over the same period."""
    if not timeline:
        return {}
    closes = pd.DataFrame({s: df["close"] for s, df in prepared.items()})
    closes = closes.loc[(closes.index >= timeline[0]) & (closes.index <= timeline[-1])].ffill().dropna()
    if closes.empty:
        return {}
    curve = (closes / closes.iloc[0]).mean(axis=1)
    dd = curve / curve.cummax() - 1
    return {
        "total_return_pct": round(float(curve.iloc[-1] - 1) * 100, 2),
        "max_drawdown_pct": round(float(dd.min()) * 100, 2),
    }


# ----------------------------------------------------------------------------
# Walk-forward optimisation
# ----------------------------------------------------------------------------
def score(metrics: dict, min_trades: int) -> float:
    """Rank by risk-adjusted return; reject configs with too few trades to trust."""
    if metrics.get("trades", 0) < min_trades:
        return -np.inf
    return metrics["sharpe"]


def optimise(data: dict[str, pd.DataFrame], strategy_name: str, grid: dict[str, list],
             base_params: dict, risk_cfg: RiskConfig, timeframe: str, train_frac: float = 0.7,
             min_trades: int = 10, top: int = 5, **bt_kwargs) -> pd.DataFrame:
    """Grid-search parameters on the first `train_frac` of the data, then report
    how the best candidates did on the unseen remainder. A config that only
    shines in-sample is probably overfit - trust the out-of-sample columns."""
    timeline = sorted(set().union(*(df.index for df in data.values())))
    split = timeline[int(len(timeline) * train_frac)]
    keys = list(grid)
    combos = list(itertools.product(*(grid[k] for k in keys)))
    log.info("Optimising %d combinations; train < %s <= test", len(combos), split)

    rows = []
    for values in combos:
        params = {**base_params, **dict(zip(keys, values))}
        try:
            res = run_backtest(data, strategy_name, params, risk_cfg, timeframe, end=split, **bt_kwargs)
        except ValueError as exc:
            log.debug("skip %s: %s", params, exc)
            continue
        rows.append((score(res.metrics, min_trades), params, res.metrics))

    rows.sort(key=lambda r: r[0], reverse=True)
    report = []
    for s, params, ins in rows[:top]:
        if not np.isfinite(s):
            continue
        oos = run_backtest(data, strategy_name, params, risk_cfg, timeframe, start=split, **bt_kwargs).metrics
        report.append({
            **{k: params[k] for k in keys},
            "is_sharpe": ins["sharpe"], "is_return_pct": ins["total_return_pct"],
            "is_maxdd_pct": ins["max_drawdown_pct"], "is_trades": ins["trades"],
            "oos_sharpe": oos["sharpe"], "oos_return_pct": oos["total_return_pct"],
            "oos_maxdd_pct": oos["max_drawdown_pct"], "oos_trades": oos["trades"],
        })
    return pd.DataFrame(report)


# ----------------------------------------------------------------------------
# Walk-forward strategy discovery
# ----------------------------------------------------------------------------
@dataclass
class WalkForwardResult:
    strategy: str
    params: dict            # best params on the most recent training window
    folds: list[dict]       # out-of-sample metrics for each test window
    summary: dict
    robust: bool
    reasons: list[str]


def _best_params(data, strategy_name, grid, base_params, risk_cfg, timeframe, end, min_trades, bt_kwargs):
    keys = list(grid)
    best, best_score = None, -np.inf
    for values in itertools.product(*(grid[k] for k in keys)) if keys else [()]:
        params = {**base_params, **dict(zip(keys, values))}
        try:
            res = run_backtest(data, strategy_name, params, risk_cfg, timeframe, end=end, **bt_kwargs)
        except ValueError:
            continue
        sc = score(res.metrics, min_trades)
        if sc > best_score:
            best, best_score = params, sc
    return best


def walk_forward(data: dict[str, pd.DataFrame], strategy_name: str, grid: dict[str, list],
                 base_params: dict, risk_cfg: RiskConfig, timeframe: str, folds: int = 4,
                 first_train_frac: float = 0.5, min_trades: int = 5, **bt_kwargs) -> WalkForwardResult:
    """Anchored walk-forward test, the way the strategy would really be used:
    tune on all data up to a date, trade the next unseen window, move forward,
    repeat. Only the unseen windows count towards the result."""
    timeline = sorted(set().union(*(df.index for df in data.values())))
    first = int(len(timeline) * first_train_frac)
    bounds = np.linspace(first, len(timeline), folds + 1).astype(int)
    fold_results, params = [], None
    for k in range(folds):
        test_start = timeline[bounds[k]]
        test_end = timeline[bounds[k + 1]] if bounds[k + 1] < len(timeline) else None
        params = _best_params(data, strategy_name, grid, base_params, risk_cfg, timeframe,
                              test_start, max(1, min_trades // 2), bt_kwargs)
        if params is None:
            fold_results.append({"fold": k + 1, "return_pct": 0.0, "trades": 0, "sharpe": 0.0,
                                 "max_drawdown_pct": 0.0, "profit_factor": 0.0, "note": "no params traded"})
            continue
        m = run_backtest(data, strategy_name, params, risk_cfg, timeframe,
                         start=test_start, end=test_end, **bt_kwargs).metrics
        fold_results.append({"fold": k + 1, "return_pct": m["total_return_pct"], "trades": m["trades"],
                             "sharpe": m["sharpe"], "max_drawdown_pct": m["max_drawdown_pct"],
                             "profit_factor": m.get("profit_factor", 0.0)})

    rets = [f["return_pct"] for f in fold_results]
    compounded = (np.prod([1 + r / 100 for r in rets]) - 1) * 100
    trades = sum(f["trades"] for f in fold_results)
    positive = sum(r > 0 for r in rets)
    worst_dd = min(f["max_drawdown_pct"] for f in fold_results)
    summary = {
        "oos_return_pct": round(float(compounded), 2),
        "folds_positive": f"{positive}/{folds}",
        "worst_fold_pct": round(min(rets), 2),
        "avg_sharpe": round(float(np.mean([f["sharpe"] for f in fold_results])), 2),
        "worst_drawdown_pct": round(worst_dd, 2),
        "oos_trades": trades,
    }
    reasons = []
    if compounded <= 0:
        reasons.append("lost money on unseen data")
    if positive < math.ceil(folds / 2):  # must make money in at least half the windows
        reasons.append("profitable in too few test windows")
    if trades < min_trades:
        reasons.append(f"only {trades} trades - too few to trust")
    if summary["avg_sharpe"] <= 0:
        reasons.append("negative risk-adjusted return")
    return WalkForwardResult(strategy_name, params or base_params, fold_results, summary,
                             robust=not reasons, reasons=reasons)


def discover(data: dict[str, pd.DataFrame], grids: dict[str, dict], risk_cfg: RiskConfig, timeframe: str,
             folds: int = 4, min_trades: int = 10, progress=None, **bt_kwargs) -> list[WalkForwardResult]:
    """Walk-forward test every strategy and rank them. Robust ones first, then
    by out-of-sample return per unit of drawdown."""
    results = []
    for name, grid in grids.items():
        if progress:
            progress(name)
        results.append(walk_forward(data, name, grid, {}, risk_cfg, timeframe, folds=folds,
                                    min_trades=min_trades, **bt_kwargs))

    def rank(r: WalkForwardResult):
        s = r.summary
        return (r.robust, s["oos_return_pct"] / max(1.0, abs(s["worst_drawdown_pct"])))

    return sorted(results, key=rank, reverse=True)
