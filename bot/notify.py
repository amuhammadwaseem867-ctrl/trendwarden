"""Optional Telegram alerts for trades and kill-switch events.

Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env to enable. Failures to
send are logged and never interrupt trading.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.request

log = logging.getLogger(__name__)


def make_notifier(prefix: str = "TrendWarden"):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not (token and chat_id):
        return lambda msg: None

    url = f"https://api.telegram.org/bot{token}/sendMessage"

    def send(msg: str) -> None:
        body = json.dumps({"chat_id": chat_id, "text": f"[{prefix}] {msg}"}).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=10).read()
        except Exception as exc:
            log.warning("Telegram notification failed: %s", exc)

    return send
