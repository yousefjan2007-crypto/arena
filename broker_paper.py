"""
The Alpaca PAPER adapter — the only file in arena that can place an order at all.

`paper=True` IS HARDCODED, one line, no branch. This file is structurally
incapable of reaching a live account: there is no mode argument, no config knob
and no environment variable that flips it, and the credential names it reads
(ALPACA_PAPER_KEY / ALPACA_PAPER_SECRET) are deliberately NOT signal_lab's
ALPACA_API_KEY / ALPACA_SECRET_KEY, so a live-account credential already sitting
in this workspace's environment cannot be picked up here by accident. Going live
is a human decision made somewhere else, with some other tool (docs/DESIGN.md,
"Graduation ladder": "The system never self-starts live").

INTERFACE — the shape signal_lab/broker.py established, in shares rather than
notional, because the sandbox decides in whole shares and the paper account has
to mirror the sandbox:

    place(symbol, side, qty, client_order_id=None) -> dict
                                          market-on-open, whole shares; the
                                          client_order_id is the broker-enforced
                                          double-entry guard
    close(symbol)            -> dict
    position(symbol)         -> dict | None
    positions()              -> {symbol: signed int shares}
    account()                -> dict | None
    clock()                  -> dict | None   the EXCHANGE clock, not this box's
    fills_since(iso_date)    -> [dict]    closed orders, for reconciliation

ORDERS ARE MARKET-ON-OPEN (TimeInForce.OPG), and that is not a preference: the
sandbox fills every decision at the next open (env.py's one clock), so an order
type that fills anywhere else would make the sim-shadow comparison measure the
order type instead of the strategy. `qty`, never `notional` — the simulator
already did the whole-share rounding, and handing Alpaca a dollar amount would
let it round differently. Alpaca accepts OPG orders only up to ~09:28 ET, which
is why paper.yml runs at 13:00 UTC / 09:00 ET; an order submitted outside that
window comes back as an exception from submit_order and the caller records it as
a rejection rather than pretending it was placed.

UNARMED IS A FIRST-CLASS STATE, not an error. With no credentials the adapter
still constructs, reports `credentialed = False`, and answers every WRITE with
{"executed": False, "reason": "no_credentials"}; the READS answer emptily
(positions() -> {}, account() -> None, fills_since() -> []). An empty book from
an unarmed broker means "unknown", NOT "flat", so a caller diffing against it
must consult `.credentialed` and say which basis it used — run_paper.py does.
Nothing here raises at import time, and `alpaca` itself is imported lazily
inside the two methods that need it, so this module is importable (and testable)
on a machine that has never installed alpaca-py.
"""
from __future__ import annotations

import os

import config                       # FIRST: puts the siblings on sys.path

MODE = "paper"
UNARMED_REASON = "no_credentials"

# Terminal statuses that mean the order is NOT working. submit_order does not
# raise for all of these — an order can come back already rejected — so "no
# exception" is not the same as "an order exists", and place() reports which.
DEAD_STATUSES = frozenset({"rejected", "canceled", "expired", "suspended",
                           "stopped", "done_for_day", "replaced"})

# The environment path (GitHub Actions repo secrets -> workflow env) and the local
# path (arena/config.local.json, gitignored). Never a literal in source.
ENV_KEY, ENV_SECRET = "ALPACA_PAPER_KEY", "ALPACA_PAPER_SECRET"
LOCAL_KEY, LOCAL_SECRET = "alpaca_paper_key", "alpaca_paper_secret"


def credentials() -> dict:
    """{"key", "secret", "source"} — environment first, config.local.json second.

    Same chain as alerts_arena.credentials(), for the same reason: a GitHub runner
    has no local file and this Mac has no repo secrets. Either half may be absent;
    the caller is unarmed unless BOTH resolve. Values are never printed anywhere
    in this module — only their presence.
    """
    key = os.environ.get(ENV_KEY)                                   # io-boundary
    secret = os.environ.get(ENV_SECRET)                             # io-boundary
    source = "environment" if (key and secret) else ""
    if not (key and secret):
        local = {}
        try:
            local = config.load_credentials()
        except Exception:
            local = {}
        key = key or local.get(LOCAL_KEY)
        secret = secret or local.get(LOCAL_SECRET)
        if key and secret:
            source = "config.local.json"
    return {"key": key, "secret": secret, "source": source}


class AlpacaPaperBroker:
    """Alpaca paper trading. Constructs in every case; submits only when armed."""

    mode = MODE

    def __init__(self, key: str = None, secret: str = None):
        if key is None and secret is None:
            cr = credentials()
        else:
            cr = {"key": key, "secret": secret, "source": "explicit"}
        self._key, self._secret = cr["key"], cr["secret"]
        self.source = cr["source"]
        self.credentialed = bool(self._key and self._secret)
        self._client = None

    def __repr__(self) -> str:
        return "<AlpacaPaperBroker mode=%s credentialed=%s source=%s>" % (
            self.mode, self.credentialed, self.source or "none")

    # ── the client ─────────────────────────────────────────────────────────────
    @property
    def client(self):
        """The alpaca-py TradingClient, built once, lazily.

        Lazy so that an unarmed run — the entire pre-graduation life of this
        project — never needs alpaca-py installed at all, and so that importing
        this module can never fail on a machine that lacks it.
        """
        if self._client is None:
            if not self.credentialed:
                raise RuntimeError(
                    "no Alpaca paper credentials: set %s and %s, or put %s / %s in "
                    "arena/config.local.json. This adapter constructs unarmed on "
                    "purpose; reaching for the client is the caller's mistake."
                    % (ENV_KEY, ENV_SECRET, LOCAL_KEY, LOCAL_SECRET))
            from alpaca.trading.client import TradingClient
            # paper=True, hardcoded, no branch above it and no argument that can
            # reach it. See this module's docstring.
            self._client = TradingClient(self._key, self._secret, paper=True)
        return self._client

    def _refused(self, action: str, **extra) -> dict:
        out = {"broker": "alpaca_paper", "mode": self.mode, "action": action,
               "executed": False, "reason": UNARMED_REASON}
        out.update(extra)
        return out

    # ── writes ─────────────────────────────────────────────────────────────────
    def place(self, symbol: str, side: str, qty: int, client_order_id: str = None) -> dict:
        """One market-on-open order for `qty` WHOLE shares. `side` is buy | sell.

        Raises on a non-integral or non-positive quantity rather than rounding:
        the sandbox already decided the share count, and an adapter that quietly
        changed it would break the one thing the paper stage measures.

        `client_order_id` IS THE DOUBLE-ENTRY GUARD, and it is the only one that
        works across machines. An OPG order does not fill until the next opening
        auction, so between two runs of the same session the account still reads
        flat and the caller would compute — and send — the identical order a
        second time. Alpaca rejects a duplicate client_order_id outright, so
        deriving it from (session date, symbol, side, qty) makes the submission
        idempotent AT THE BROKER, where a local check cannot reach. run_paper
        also checks its own ledger; this is the layer that survives a lost ledger,
        a second machine, or a re-run against a runner that never pushed.
        """
        side = str(side).lower()
        if side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell', got %r" % side)
        if int(qty) != qty or int(qty) <= 0:
            raise ValueError("qty must be a positive whole number of shares, got %r" % qty)
        qty = int(qty)
        if not self.credentialed:
            return self._refused("place", symbol=symbol, side=side, qty=qty,
                                 client_order_id=client_order_id)

        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest
        # MarketOrderRequest fixes type=OrderType.MARKET; OPG is the market-on-open
        # time-in-force. Both halves are the env's fill model, stated to the broker.
        req = MarketOrderRequest(symbol=symbol, qty=qty,
                                 side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                                 time_in_force=TimeInForce.OPG,
                                 client_order_id=client_order_id)
        order = self.client.submit_order(req)
        # order.status is REPORTED, never assumed: submit_order can return an order
        # that is already rejected, and a caller that recorded "submitted" off the
        # absence of an exception would have a ledger row for a position it does
        # not hold. See DEAD_STATUSES.
        status = _enum_value(order.status)
        return {"broker": "alpaca_paper", "mode": self.mode, "action": "place",
                "symbol": symbol, "side": side, "qty": qty, "executed": True,
                "order_id": str(order.id), "status": status,
                "client_order_id": str(order.client_order_id or client_order_id or ""),
                "live": status not in DEAD_STATUSES}

    def close(self, symbol: str) -> dict:
        """Flatten one symbol. Alpaca closes at market, so this is an exit only —
        the ordinary path is place(), because the sandbox's exits fill at the open
        like everything else."""
        if not self.credentialed:
            return self._refused("close", symbol=symbol)
        order = self.client.close_position(symbol)
        return {"broker": "alpaca_paper", "mode": self.mode, "action": "close",
                "symbol": symbol, "executed": True, "order_id": str(order.id),
                "status": _enum_value(order.status)}

    # ── reads ──────────────────────────────────────────────────────────────────
    def account(self):
        """{"equity", "last_equity", "cash", "buying_power", "status"} or None.

        `last_equity` is the account's equity at the previous session's close, so
        equity/last_equity - 1 is the realised return of the session just ended —
        which is the number the sim-shadow is compared against, read pre-open in
        one call rather than reconstructed from two.
        """
        if not self.credentialed:
            return None
        a = self.client.get_account()
        return {"equity": float(a.equity), "last_equity": float(a.last_equity),
                "cash": float(a.cash), "buying_power": float(a.buying_power),
                "status": _enum_value(a.status)}

    def clock(self):
        """The EXCHANGE clock: {"is_open", "timestamp", "next_open", "next_close"}.

        Not this machine's clock and not a compute input — it answers one
        question, "is the session I am about to measure actually over", and the
        only correct source for it is the venue. run_paper's shadow row compares
        the account's `equity/last_equity` against a full simulated trading day,
        and that comparison is only true PRE-OPEN. Mid-session the account holds a
        partial day and the row would be a permanent, wrong record. So the clock
        is read, and a session that is open is skipped rather than mismeasured.
        """
        if not self.credentialed:
            return None
        c = self.client.get_clock()                                 # io-boundary
        return {"is_open": bool(c.is_open), "timestamp": c.timestamp,
                "next_open": c.next_open, "next_close": c.next_close}

    def positions(self) -> dict:
        """{symbol: signed whole shares}. EMPTY MEANS UNKNOWN when unarmed."""
        if not self.credentialed:
            return {}
        out = {}
        for p in self.client.get_all_positions():
            out[p.symbol] = int(float(p.qty))     # signed: Alpaca reports shorts negative
        return out

    def position(self, symbol: str):
        """One position, or None when flat (or unarmed)."""
        if not self.credentialed:
            return None
        try:
            p = self.client.get_open_position(symbol)
        except Exception:
            return None                           # Alpaca 404s a flat symbol
        return {"symbol": symbol, "qty": int(float(p.qty)),
                "avg_entry_price": float(p.avg_entry_price),
                "market_value": float(p.market_value),
                "unrealized_plpc": float(p.unrealized_plpc)}

    def fills_since(self, iso_date: str) -> list:
        """Closed orders from `iso_date` (YYYY-MM-DD) onward, oldest first.

        The reconciliation input: one dict per closed order with date (the fill
        date, or the submission date for an order that never filled), symbol,
        side, qty FILLED, fill price, order id and terminal status. Orders that
        expired or were cancelled come back too, with qty 0 — a paper session
        that submitted an order which never filled is a fact the ledger has to
        carry, not an absence.

        No wall clock: the caller supplies the anchor, which comes from the
        ledger's own last recorded date.
        """
        if not self.credentialed:
            return []
        import datetime as _dt
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        after = _dt.datetime.strptime(str(iso_date)[:10], "%Y-%m-%d")   # parse, not now()
        req = GetOrdersRequest(status=QueryOrderStatus.CLOSED, after=after,
                               direction="asc", limit=500)
        out = []
        for o in self.client.get_orders(filter=req):
            when = o.filled_at or o.submitted_at or o.created_at
            out.append({"date": str(when.date()) if when is not None else "",
                        "symbol": o.symbol,
                        "side": _enum_value(o.side),
                        "qty": int(float(o.filled_qty or 0)),
                        "fill_px": float(o.filled_avg_price) if o.filled_avg_price else float("nan"),
                        "order_id": str(o.id),
                        "status": _enum_value(o.status)})
        return out


def _enum_value(value):
    """alpaca-py returns enums for status/side; the ledger stores their strings."""
    return getattr(value, "value", value)


def get_broker(key: str = None, secret: str = None) -> AlpacaPaperBroker:
    """The only constructor arena calls. There is no live variant to choose."""
    return AlpacaPaperBroker(key, secret)


if __name__ == "__main__":
    # OFFLINE by construction: with no credentials nothing is imported from
    # alpaca-py and no socket is opened. With credentials present this touches the
    # account-read endpoint once (an I/O boundary, printed as such) and places NO
    # order — this smoke test must be safe to run on any machine, any time of day.
    b = get_broker()
    print("arena paper broker")
    print("  adapter    : %s" % b)
    print("  paper=True : hardcoded in broker_paper.AlpacaPaperBroker.client — there "
          "is no argument, knob or env var that can make this adapter live")
    print("  credentials: %s / %s (env) -> %s / %s (config.local.json); present: %s%s"
          % (ENV_KEY, ENV_SECRET, LOCAL_KEY, LOCAL_SECRET, b.credentialed,
             "  [source: %s]" % b.source if b.credentialed else ""))
    print("  execution  : config.EXECUTION_MODE = %r" % config.EXECUTION_MODE)

    if not b.credentialed:
        print("  UNARMED — every write refuses, every read answers empty:")
        print("    place  : %s" % b.place("SPY", "buy", 3))
        print("    close  : %s" % b.close("SPY"))
        print("    account: %s   positions: %s   clock: %s   fills_since: %s"
              % (b.account(), b.positions(), b.clock(), b.fills_since("2026-01-02")))
        print("    An empty book here means UNKNOWN, not flat: run_paper.py checks "
              ".credentialed before it diffs anything against it.")
        # The refusal contract, asserted rather than described.
        for result in (b.place("SPY", "buy", 3), b.close("SPY")):
            assert result["executed"] is False and result["reason"] == UNARMED_REASON
        assert b.positions() == {} and b.account() is None and b.clock() is None \
            and b.fills_since("2026-01-02") == []
        print("  smoke      : PASS (unarmed contract holds; alpaca-py never imported)")
    else:
        acct = b.account()                                          # io-boundary
        clk = b.clock()                                             # io-boundary
        print("  ARMED — account read (no order placed):")
        print("    equity $%.2f | last_equity $%.2f | cash $%.2f | buying power $%.2f"
              % (acct["equity"], acct["last_equity"], acct["cash"], acct["buying_power"]))
        print("    status %s | %d open position(s)" % (acct["status"], len(b.positions())))
        print("    exchange clock: %s (next open %s)"
              % ("OPEN" if clk["is_open"] else "closed", clk["next_open"]))
        print("  smoke      : PASS (account reachable; nothing was submitted)")

    # Whole shares are a contract, not a convenience.
    for bad in (0, -3, 2.5):
        try:
            b.place("SPY", "buy", bad)
            raise AssertionError("qty %r was accepted" % bad)
        except ValueError:
            pass
    try:
        b.place("SPY", "hold", 1)
        raise AssertionError("side 'hold' was accepted")
    except ValueError:
        pass
    print("  guards     : non-integral, non-positive and unknown-side orders all refused")
