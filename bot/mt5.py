"""MetaTrader 5 support (XM and any other MT5 broker): gold, forex, indices, CFDs.

Requires Windows, the MetaTrader 5 desktop terminal installed and logged in,
and `pip install MetaTrader5`. The Python package talks to the running terminal.

* MT5Market - market data with the same small interface the bot uses for ccxt
              exchanges (fetch_ohlcv / fetch_ticker / milliseconds), so the
              backtester and engine work unchanged.
* MT5Broker - real orders. Stop-loss and take-profit are placed ON THE BROKER'S
              SERVER with each order and the trailing stop is pushed to the
              server as it moves, so positions stay protected even if the bot
              or your PC goes offline.

Note: MT5 reports candle times in the broker's server time zone, not UTC. All
"now" comparisons use the server clock (latest tick time) so this is consistent.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone

from .broker import Fill

log = logging.getLogger(__name__)

_TIMEFRAMES = {
    "1m": "TIMEFRAME_M1", "5m": "TIMEFRAME_M5", "15m": "TIMEFRAME_M15", "30m": "TIMEFRAME_M30",
    "1h": "TIMEFRAME_H1", "4h": "TIMEFRAME_H4", "1d": "TIMEFRAME_D1", "1w": "TIMEFRAME_W1",
}


def _mt5():
    try:
        import MetaTrader5 as mt5
    except ImportError:
        raise SystemExit(
            "The MetaTrader5 package is not installed. It only works on Windows:\n"
            "    pip install MetaTrader5\n"
            "and the MetaTrader 5 terminal must be installed and running."
        ) from None
    return mt5


def find_terminals() -> list[str]:
    """Look for installed MT5 terminals (terminal64.exe) in the usual places.
    XM installs to e.g. C:\\Program Files\\XM Global MT5\\terminal64.exe."""
    import glob
    import os

    roots = []
    for var in ("ProgramFiles", "ProgramW6432", "ProgramFiles(x86)", "LOCALAPPDATA", "APPDATA"):
        if os.getenv(var):
            roots.append(os.getenv(var))
    for drive in "CDEF":
        roots += [f"{drive}:\\Program Files", f"{drive}:\\Program Files (x86)", f"{drive}:\\"]
    found: list[str] = []
    for root in roots:
        for pattern in ("*\\terminal64.exe", "Programs\\*\\terminal64.exe"):
            for f in glob.glob(os.path.join(root, pattern)):
                if f not in found:
                    found.append(f)
    # Prefer XM's terminal if several brokers' terminals are installed.
    found.sort(key=lambda f: (0 if "xm" in f.lower() else 1, f))
    return found


def connect(path: str | None = None, login: str | None = None, password: str | None = None,
            server: str | None = None):
    """Start/attach to the MT5 terminal. If login details are omitted, the
    account already logged in to the terminal is used."""
    import os

    mt5 = _mt5()
    creds = {}
    if login:
        creds.update(login=int(login), password=password or "", server=server or "")

    path = path or os.getenv("MT5_PATH")
    candidates = [path] if path else [None] + find_terminals()
    errors = []
    for candidate in candidates:
        kwargs = dict(creds)
        if candidate:
            if not os.path.exists(candidate):
                errors.append(f"{candidate}: file does not exist")
                continue
            kwargs["path"] = candidate
        if mt5.initialize(**kwargs):
            if candidate:
                log.info("Using MT5 terminal at %s", candidate)
            break
        errors.append(f"{candidate or 'auto-detect'}: {mt5.last_error()}")
    else:
        raise SystemExit(
            "Could not connect to MetaTrader 5.\n  " + "\n  ".join(errors) + "\n\n"
            "Fix:\n"
            "  1. Install MetaTrader 5 from XM (an MT5 account - MT4 does not work) and open it.\n"
            "  2. Find terminal64.exe: right-click the MT5 desktop shortcut > Open file location.\n"
            "  3. Put that path in config.yaml under mt5:, in single quotes, e.g.\n"
            "       path: 'C:\\Program Files\\XM Global MT5\\terminal64.exe'\n"
            "     (or set MT5_PATH in .env), then run the command again.\n"
            "  Also check you installed 64-bit Python (python -c \"import struct; print(struct.calcsize('P')*8)\" -> 64)."
        )
    acct = mt5.account_info()
    if acct is None:
        raise SystemExit(f"MetaTrader 5 is not logged in to an account: {mt5.last_error()}. "
                         "Log in inside the terminal (File > Login to Trade Account) and retry.")
    log.info("Connected to MT5 account %s on %s (%s, leverage 1:%s, equity %.2f %s)",
             acct.login, acct.server, "DEMO" if _is_demo(mt5, acct) else "REAL",
             acct.leverage, acct.equity, acct.currency)
    return mt5


def _is_demo(mt5, acct) -> bool:
    return getattr(acct, "trade_mode", None) == getattr(mt5, "ACCOUNT_TRADE_MODE_DEMO", 0)


def ensure_symbol(mt5, symbol: str):
    info = mt5.symbol_info(symbol)
    if info is None:
        guesses = [s.name for s in (mt5.symbols_get("*GOLD*,*XAU*") or [])][:10]
        hint = f" Gold symbols on this account: {', '.join(guesses)}" if guesses else ""
        raise SystemExit(f"Symbol '{symbol}' not found on this MT5 account.{hint}")
    if not info.visible and not mt5.symbol_select(symbol, True):
        raise SystemExit(f"Could not add '{symbol}' to Market Watch: {mt5.last_error()}")
    return mt5.symbol_info(symbol)


class MT5Market:
    id = "mt5"
    rateLimit = 0

    def __init__(self, mt5, symbols: list[str]):
        self.mt5 = mt5
        self.symbols = symbols
        for s in symbols:
            ensure_symbol(mt5, s)

    def _tf(self, timeframe: str):
        try:
            return getattr(self.mt5, _TIMEFRAMES[timeframe])
        except KeyError:
            raise ValueError(f"Timeframe {timeframe} not supported on MT5; use one of {list(_TIMEFRAMES)}") from None

    def milliseconds(self) -> int:
        """Current broker server time (from the latest tick), in ms."""
        times = []
        for s in self.symbols:
            tick = self.mt5.symbol_info_tick(s)
            if tick is not None:
                times.append(int(getattr(tick, "time_msc", 0) or tick.time * 1000))
        if not times:
            return int(datetime.now(timezone.utc).timestamp() * 1000)
        return max(times)

    def fetch_ohlcv(self, symbol: str, timeframe: str, since: int | None = None, limit: int = 500):
        tf = self._tf(timeframe)
        if since is None:
            rates = self.mt5.copy_rates_from_pos(symbol, tf, 0, limit)
        else:
            start = datetime.fromtimestamp(since / 1000, tz=timezone.utc)
            end = datetime.fromtimestamp(self.milliseconds() / 1000 + 86_400, tz=timezone.utc)
            rates = self.mt5.copy_rates_range(symbol, tf, start, end)
        if rates is None:
            raise RuntimeError(f"MT5 returned no data for {symbol} {timeframe}: {self.mt5.last_error()}")
        return [[int(r["time"]) * 1000, float(r["open"]), float(r["high"]), float(r["low"]),
                 float(r["close"]), float(r["tick_volume"])] for r in rates]

    def fetch_ticker(self, symbol: str) -> dict:
        tick = self.mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"No price for {symbol}: {self.mt5.last_error()}")
        # Long positions are valued and closed at the bid.
        return {"last": float(tick.bid), "bid": float(tick.bid), "ask": float(tick.ask)}


class MT5Broker:
    """Market orders on an MT5 account. Quantities from the bot are in units of
    the underlying (ounces for gold) and converted to lots here."""

    def __init__(self, mt5, magic: int = 240611, deviation: int = 30, fee_rate: float = 0.0):
        self.mt5 = mt5
        self.magic = magic
        self.deviation = deviation
        self.fee_rate = fee_rate
        self.tickets: dict[str, int] = {}  # symbol -> MT5 position ticket
        term = mt5.terminal_info()
        if term is not None and not term.trade_allowed:
            raise SystemExit("Algo Trading is switched OFF in the MT5 terminal. "
                             "Click the 'Algo Trading' button in the toolbar so it turns green.")

    # ----- account -------------------------------------------------------
    def account_equity(self) -> float:
        return float(self.mt5.account_info().equity)

    def get_cash(self) -> float:
        """Buying power: free margin times leverage (CFDs are margined)."""
        a = self.mt5.account_info()
        return float(a.margin_free) * float(a.leverage or 1)

    # ----- helpers -------------------------------------------------------
    def _position(self, symbol: str):
        for p in self.mt5.positions_get(symbol=symbol) or ():
            if p.magic == self.magic:
                self.tickets[symbol] = p.ticket
                return p
        return None

    def min_qty(self, symbol: str) -> float:
        """Broker's minimum trade size in units (e.g. 0.01 lot x 100 oz = 1 oz of gold)."""
        info = ensure_symbol(self.mt5, symbol)
        return float(info.volume_min) * float(info.trade_contract_size)

    def position_open(self, symbol: str) -> bool:
        return self._position(symbol) is not None

    def _filling(self, info):
        mode = int(getattr(info, "filling_mode", 0))
        if mode & 1:
            return self.mt5.ORDER_FILLING_FOK
        if mode & 2:
            return self.mt5.ORDER_FILLING_IOC
        return self.mt5.ORDER_FILLING_RETURN

    def _lots(self, info, qty: float) -> float:
        lots = qty / info.trade_contract_size
        step = info.volume_step
        lots = math.floor(lots / step + 1e-9) * step  # round DOWN: never risk more than planned
        lots = min(lots, info.volume_max)
        return round(lots, 8)

    def _send(self, request: dict):
        result = self.mt5.order_send(request)
        if result is None:
            log.error("order_send failed: %s", self.mt5.last_error())
            return None
        if result.retcode != self.mt5.TRADE_RETCODE_DONE:
            log.error("Order rejected (retcode %s): %s", result.retcode, result.comment)
            return None
        return result

    def _price_ok(self, info, price: float) -> float:
        return round(price, int(info.digits))

    # ----- orders --------------------------------------------------------
    def buy(self, symbol: str, qty: float, price: float, sl: float | None = None,
            tp: float | None = None) -> Fill | None:
        info = ensure_symbol(self.mt5, symbol)
        lots = self._lots(info, qty)
        if lots < info.volume_min:
            log.warning("%s: planned size %.4f lots is below the broker minimum %.2f lots - trade skipped. "
                        "Your account is too small for this risk setting.",
                        symbol, qty / info.trade_contract_size, info.volume_min)
            return None
        tick = self.mt5.symbol_info_tick(symbol)
        calc = getattr(self.mt5, "order_calc_margin", None)
        if calc is not None:
            margin = calc(self.mt5.ORDER_TYPE_BUY, symbol, lots, tick.ask)
            free = float(self.mt5.account_info().margin_free)
            if margin is not None and margin > free * 0.9:  # keep a 10% buffer
                log.warning("%s: not enough free margin for %s lots (needs %.2f, free %.2f) - trade skipped",
                            symbol, lots, margin, free)
                return None
        req = {
            "action": self.mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": lots,
            "type": self.mt5.ORDER_TYPE_BUY, "price": tick.ask, "deviation": self.deviation,
            "magic": self.magic, "comment": "TrendWarden", "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self._filling(info),
        }
        if sl:
            req["sl"] = self._price_ok(info, sl)
        if tp:
            req["tp"] = self._price_ok(info, tp)
        log.info("MT5 BUY %s lots %s @ ~%s", lots, symbol, tick.ask)
        res = self._send(req)
        if res is None:
            return None
        self._position(symbol)  # remember the new position's ticket
        return Fill(lots * info.trade_contract_size, float(res.price or tick.ask), 0.0)

    def protect(self, symbol: str, sl: float, tp: float | None) -> None:
        """Set/move the server-side stop-loss and take-profit."""
        pos = self._position(symbol)
        if pos is None:
            return
        info = self.mt5.symbol_info(symbol)
        new_sl = self._price_ok(info, sl)
        new_tp = self._price_ok(info, tp) if tp else 0.0
        if abs(new_sl - pos.sl) < info.point and abs(new_tp - pos.tp) < info.point:
            return
        res = self._send({"action": self.mt5.TRADE_ACTION_SLTP, "symbol": symbol,
                          "position": pos.ticket, "sl": new_sl, "tp": new_tp})
        if res is not None:
            log.info("%s: server stop moved to %s", symbol, new_sl)

    def sell(self, symbol: str, qty: float, price: float) -> Fill | None:
        """Close the bot's position in `symbol`."""
        pos = self._position(symbol)
        if pos is None:
            log.warning("%s: no open MT5 position to close", symbol)
            return None
        info = self.mt5.symbol_info(symbol)
        tick = self.mt5.symbol_info_tick(symbol)
        res = self._send({
            "action": self.mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": pos.volume,
            "type": self.mt5.ORDER_TYPE_SELL, "position": pos.ticket, "price": tick.bid,
            "deviation": self.deviation, "magic": self.magic, "comment": "TrendWarden close",
            "type_time": self.mt5.ORDER_TIME_GTC, "type_filling": self._filling(info),
        })
        if res is None:
            return None
        log.info("MT5 CLOSE %s lots %s @ %s", pos.volume, symbol, res.price)
        return Fill(pos.volume * info.trade_contract_size, float(res.price or tick.bid), 0.0)

    def last_exit_price(self, symbol: str, since: datetime) -> float | None:
        """Price at which the broker closed our position (e.g. its stop-loss hit)."""
        deals = self.mt5.history_deals_get(since, datetime.now(timezone.utc) + timedelta(days=2),
                                           group=f"*{symbol}*") or ()
        out = getattr(self.mt5, "DEAL_ENTRY_OUT", 1)
        ticket = self.tickets.get(symbol)
        exits = [d for d in deals if d.entry == out and
                 (getattr(d, "position_id", None) == ticket if ticket else d.magic == self.magic)]
        exits.sort(key=lambda d: getattr(d, "time_msc", 0))
        return float(exits[-1].price) if exits else None
