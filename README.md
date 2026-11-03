# TrendWarden

An automated crypto trading bot with strict risk controls. You test a strategy on years of history, check that it holds up on data it was never tuned on, paper trade it with live prices, and only then switch to real money.

> **Read this first.** No trading bot guarantees profit, and most strategies lose money after fees. TrendWarden is designed to limit losses, not to promise gains. Past backtest results do not predict future returns. Only trade money you can afford to lose.

## What it does

- **Strategies** (`bot/strategies.py`)
  - `trend_momentum`: EMA crossover, only trading in the direction of the 200-EMA trend, filtered by ADX trend strength and an RSI cap.
  - `mean_reversion`: buys oversold Bollinger Band dips in an up-trend and exits at the mean.
  - `breakout`: Donchian channel breakouts (turtle-style).
- **Risk management** (`bot/risk.py`)
  - Position size is set so that a stop-loss costs about 1% of your account (configurable), fees included.
  - ATR stop-loss, trailing stop and take-profit on every trade.
  - Caps on position size and number of open positions, plus a cooldown after a loss.
  - **Daily loss limit**: no new trades for the rest of the day after -3%.
  - **Kill switch**: at -15% from peak equity it closes everything and halts until you reset it manually.
- **Realistic backtester**: fees, slippage, gaps through stops, and the stop assumed to hit first when a candle touches both stop and target. There's no look-ahead (a test checks this). Results are compared with buy & hold.
- **Walk-forward optimiser**: tunes parameters on the first 70% of history, then shows how they did on the unseen 30%.
- **Paper and live trading** on 100+ crypto exchanges through [ccxt](https://github.com/ccxt/ccxt) (Binance, Kraken, Bybit, OKX, Coinbase, KuCoin and more), **and on MetaTrader 5 brokers such as XM, for gold, forex and other CFDs**. See [Trading gold on XM](#trading-gold-on-xm-metatrader-5).
- **Crash-safe**: state is saved after every step and restored on restart. It retries network errors with backoff.
- **Telegram alerts** (optional) for every trade and kill-switch event.

It trades **long only (buys, never sells short)**. On crypto exchanges it trades spot with no leverage. On MT5 it trades CFDs; position size is still set by the stop-loss and your risk %.

## Setup

Requires Python 3.10+.

```bash
git clone https://github.com/amuhammadwaseem867-ctrl/trendwarden.git
cd trendwarden
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp config.example.yaml config.yaml
```

## The safe path to live trading

### 1. Backtest

```bash
python main.py backtest --plot
```

This downloads `history_days` of candles for your symbols, then prints return, CAGR, max drawdown, Sharpe/Sortino, win rate, profit factor and average R, compared with buy & hold. The equity curve, trade list and chart are saved to `results/`.

Try other strategies, symbols and timeframes by editing `config.yaml`. To test offline, use `--synthetic` for generated data or `--csv file.csv` for your own data.

### 2. Optimise without fooling yourself

```bash
python main.py optimise
python main.py optimise --grid '{"fast_ema":[12,21,34],"slow_ema":[55,89]}'
```

Focus on the **OOS** (out-of-sample) columns. Pick settings that are still profitable on unseen data and that don't change much between neighbouring values. A big gap between in-sample and out-of-sample results means the settings are overfit.

### 3. Paper trade for at least a few weeks

```bash
python main.py paper
python main.py status      # in another terminal
```

This uses live prices and simulated money. No API keys are needed. Stop it with Ctrl+C; it resumes where it left off.

### 4. Go live with small size

1. On your exchange, create an API key with **trade permission only**. **Never enable withdrawals**, and restrict the key to your IP address if the exchange allows it.
2. `cp .env.example .env` and fill in `EXCHANGE_API_KEY` and `EXCHANGE_API_SECRET`.
3. Optionally, set `testnet: true` first to trade on the exchange's sandbox.
4. Set `live_enabled: true` in `config.yaml`, then:

```bash
python main.py live --confirm-live
```

Start with a small balance in that exchange account. The bot sizes trades from the actual balance in the quote currency (for example USDT).

**Important:** stop-losses are managed by the bot, not placed on the exchange. **Keep the bot running whenever positions are open.** Run it on a small always-on server (a VPS) using `tmux`, `screen`, or a systemd service.

## Trading gold on XM (MetaTrader 5)

The bot drives the MetaTrader 5 desktop terminal through MetaQuotes' official Python package. This **only works on Windows** and needs an XM **MT5** account. XM MT4 accounts are not supported, so open an MT5 account (demo or real) in the XM Members Area if you only have MT4.

### Setup (once)

1. Install **MetaTrader 5 from XM** and log in to your XM MT5 account. Start with a **demo** account.
2. In MT5, click the **Algo Trading** button in the toolbar so it turns green.
3. Optional, for long backtests: go to *Tools → Options → Charts* and set **Max bars in chart** to *Unlimited*, then restart MT5.
4. In the bot folder:

```bash
pip install -r requirements.txt
copy config.gold-xm.yaml config.yaml
```

5. Optionally, put your MT5 login in `.env` (`MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER`, e.g. `XMGlobal-MT5 2`). If you leave these empty, the bot uses whatever account is logged in to the open terminal.

### Run it

```bash
python main.py backtest --plot      # uses gold history from your MT5 terminal
python main.py optimise
python main.py paper                # live XM gold prices, simulated money
python main.py live --confirm-live  # after setting live_enabled: true (use a DEMO account first)
```

Keep MT5 open while the bot runs. **Run `live` on an XM demo account first.** It places real orders on the demo server, so you can see exactly how the bot behaves.

### How it differs from crypto

- **Stops live on XM's server.** Every order is sent with its stop-loss and take-profit attached, and the trailing stop is moved on the server as price rises. Positions stay protected even if your PC or the bot goes offline. When the server closes a position, the bot detects it and records the trade.
- **Orders are tagged.** The bot only manages positions it opened, identified by the `magic` number in the config. Your manual trades in the same account are left alone.
- **Lot sizes:** the bot reads the contract size, minimum lot and lot step from XM (for example 1 lot of GOLD = 100 oz, minimum 0.01 lot = 1 oz). It always rounds **down**. If 1% risk works out to less than the minimum lot, it **skips the trade** rather than risking more than you set. In practice that means you need roughly **$3,000–5,000** for 1% risk on 1-hour gold, and less on a lower timeframe or at a higher risk %. The log tells you when a trade was skipped for this reason.
- **Costs:** the spread is modelled with `slippage_pct`. **Overnight swap** charges are *not* included in backtests, so long-held positions will cost a bit more in reality.
- **Weekends:** gold closes from Friday night to Sunday night. Prices can gap at the open, and a stop-loss then fills at the gap price, not the stop level. The backtester models these gaps.

## Commands

| Command | What it does |
|---|---|
| `python main.py backtest [--plot] [--synthetic \| --csv F]` | Historical test |
| `python main.py optimise [--grid JSON] [--train 0.7]` | Walk-forward parameter search |
| `python main.py paper` | Paper trading on live prices |
| `python main.py live --confirm-live` | Real-money trading |
| `python main.py status` | Open positions, cash, PnL, halt state |
| `python main.py reset-halt --mode paper\|live` | Re-arm after the kill switch |

Add `-v` for detailed logs and `-c other.yaml` to use another config. Paper and live logs are written to `state/`.

## Telegram alerts (optional)

Create a bot with [@BotFather](https://t.me/BotFather), get your chat id, and set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env`.

## Project layout

```
main.py              command line
config.example.yaml  all settings, with comments
bot/indicators.py    EMA, RSI, ATR, ADX, Bollinger, Donchian
bot/strategies.py    the three strategies (add your own here)
bot/risk.py          sizing, stops, daily limit, kill switch
bot/trader.py        shared trading logic used by backtest AND live
bot/backtest.py      backtester, metrics, walk-forward optimiser
bot/broker.py        simulated and live (ccxt) order execution
bot/mt5.py           MetaTrader 5 data + orders (XM, gold, forex)
config.gold-xm.yaml  ready-made config for gold on XM
bot/engine.py        paper/live loop with state persistence
bot/data.py          candle download, caching, synthetic data
tests/               pytest suite (run: python -m pytest)
```

### Adding a strategy

Subclass `Strategy` in `bot/strategies.py`. Its `generate(df)` must add `entry`, `exit` and `atr` columns, using only data up to each bar's close. Register it in `STRATEGIES`. The backtester, optimiser and live engine pick it up automatically.
