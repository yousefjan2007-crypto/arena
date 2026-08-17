# VENDORED from ~/signal_lab/alerts.py @ de99b61 on 2026-08-17.
# Byte-identical to that source below this header. The public cloud runner
# has no sibling checkouts. DO NOT EDIT — re-vendor from the source instead.
"""
Alert delivery — lifted from vrp_backtest/monitor.py so signals reach your phone
through the channels you already use: macOS Notification Center, ntfy.sh (push),
and Telegram. SSL uses certifi (same fix as monitor.py and universe.py).

Credentials come from config.load_credentials() — never hardcoded here.
"""
from __future__ import annotations

import ssl
import subprocess
import sys
import time
import urllib.parse
import urllib.request

import certifi
import pandas as pd

import config

_SSL_CTX = ssl.create_default_context(cafile=certifi.where())


def _send_with_retries(name: str, req: urllib.request.Request, attempts: int = 3) -> None:
    """Alerts are rare and the scripts one-shot, so a transient network blip
    (e.g. a DNS drop on wake-from-sleep) would otherwise lose the alert forever."""
    for i in range(1, attempts + 1):
        try:
            urllib.request.urlopen(req, timeout=10, context=_SSL_CTX)
            return
        except Exception as e:
            print(f"[{name} error] attempt {i}/{attempts}: {e}", file=sys.stderr)
            if i < attempts:
                time.sleep(2 * i)


def macos_notify(title: str, body: str) -> None:
    title = title.replace('"', '\\"')
    body = body.replace('"', '\\"').replace("\n", " — ")
    script = f'display notification "{body}" with title "{title}" sound name "Glass"'
    subprocess.run(["osascript", "-e", script], check=False)


def ntfy_notify(topic: str, title: str, body: str) -> None:
    if not topic:
        return
    req = urllib.request.Request(f"https://ntfy.sh/{topic}", data=body.encode("utf-8"),
                                 method="POST")
    req.add_header("Title", title)
    req.add_header("Priority", "high")
    req.add_header("Tags", "chart_with_upwards_trend,bell")
    _send_with_retries("ntfy", req)


def telegram_notify(bot_token: str, chat_id: str, title: str, body: str) -> None:
    if not bot_token or not chat_id:
        return

    def esc(s):
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    text = f"<b>{esc(title)}</b>\n<pre>{esc(body)}</pre>"
    data = urllib.parse.urlencode({"chat_id": str(chat_id), "text": text,
                                   "parse_mode": "HTML"}).encode("utf-8")
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    _send_with_retries("telegram", urllib.request.Request(url, data=data))


def format_signals(signals: pd.DataFrame, asof) -> tuple[str, str]:
    """Build a compact alert title + body from a ranked signal table."""
    longs = signals[signals["side"] == "BUY"]
    shorts = signals[signals["side"] == "SHORT"]
    title = f"signal_lab {pd.Timestamp(asof).date()}: {len(longs)} long / {len(shorts)} short"

    def fmt_row(r):
        return (f"{r.side:5s} {r.symbol:6s} conv {r.conviction:+.2f} "
                f"[{r.horizon}] @ {r.entry:.2f}  stop {r.stop:.2f}")

    lines = ["LONGS:"] + [fmt_row(r) for r in longs.itertuples()]
    lines += ["", "SHORTS:"] + [fmt_row(r) for r in shorts.itertuples()]
    return title, "\n".join(lines)


def send_all(title: str, body: str, dry_run: bool = True) -> None:
    """Send to every configured channel. dry_run just prints to console."""
    print(f"\n=== ALERT {'(DRY RUN — not sent)' if dry_run else '(SENDING)'} ===")
    print(title)
    print(body)
    if dry_run:
        return
    creds = config.load_credentials()
    macos_notify(title, body)
    ntfy_notify(creds.get("ntfy_topic"), title, body)
    tg = creds.get("telegram") or {}
    telegram_notify(tg.get("bot_token"), tg.get("chat_id"), title, body)


if __name__ == "__main__":
    demo = pd.DataFrame([
        {"side": "BUY", "symbol": "NVDA", "conviction": 0.34, "horizon": "med", "entry": 170.2, "stop": 158.0},
        {"side": "SHORT", "symbol": "XOM", "conviction": -0.28, "horizon": "long", "entry": 95.1, "stop": 101.5},
    ])
    t, b = format_signals(demo, "2026-06-03")
    send_all(t, b, dry_run=True)
    print("\n(creds present: %s)" % {k: bool(v) for k, v in config.load_credentials().items()})
