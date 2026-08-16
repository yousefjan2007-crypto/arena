"""
MarketEnv — the daily-bar sandbox market. One clock, one account, honest costs.

THE CLOCK (there is only one, and everything in arena obeys it):

    close of day t          the agent sees features/prices through t and the
                            account state, and emits target weights
    open of day t+1         the env converts weight deltas into whole-share
                            orders and fills them at open +- half-spread +-
                            slippage, charging commission
    close of day t+1        borrow on short market value, margin interest on
                            negative cash, then mark to market
    reward                  log(equity[t+1] / equity[t]), net of everything

Nothing can be decided and executed on the same bar, so same-day round trips are
structurally impossible (config.NO_INTRADAY_EXITS) and the account stays PDT-safe.

WHAT THE ENV ENFORCES (not the strategy — a genome cannot cheat these):
whole shares; orders below config.MIN_POSITION_USD dropped (full exits exempt,
or a sub-minimum position could never be closed); per-name weight cap; max
open positions; pro-rata scale-down when gross or net leverage is exceeded;
no orders in a symbol with no bar at t+1. Caps are enforced on the decision
reference (close t, equity t) — the basis the order was sized on. Prices moving
overnight can carry a name past its cap afterwards; that is drift, not a breach,
and the env does not chase it intraday. It does, on the next decision, place
whatever reducing trades the caps require even when they are smaller than
MIN_POSITION_USD (reason "risk_cap"): risk limits outrank the dust filter, or a
drifted position could sit outside its cap indefinitely.

DECISION LOG (Phase 3 passes the real logger; None disables it). Any object with
`.append(row_dict)` works. One row per fill, per forced action, and per financing
charge, so the equity path is fully replayable from the log alone:

    date            date of the cash flow: the fill date (open t+1), or the mark
                    date for financing. The decision behind it was made at the
                    previous close.
    symbol          ticker, or "__CASH__" for account-level financing rows
    side            "buy" | "sell" | "carry"
    shares          SIGNED share delta (+bought, -sold); 0 on carry rows
    fill_px         execution price per share; NaN on carry rows
    commission      dollars, >= 0
    spread_cost     dollars, >= 0  (half-spread paid, |shares| x per-share spread)
    slippage        dollars, >= 0
    borrow          dollars, >= 0  (financing charged on carry rows)
    weight_before   position weight before the fill, at the decision reference
    weight_after    position weight after the fill, at the decision reference
    reason          "rebalance" (or whatever the strategy layer labels it:
                    stop/regime/derisk), "risk_cap", "data_end", "borrow", "margin"

Replay: cash = START_CASH; for each row cash -= shares*fill_px + commission +
borrow, and shares[symbol] += shares. spread_cost/slippage are attribution of
the fill price, already inside fill_px — do not subtract them again.

SIMULATOR LIMITATIONS (stated because they bias results):
  • Delisting is not modeled as a corporate event. When a held symbol has no bar
    at t+1 the position is force-closed at the last available close, cost-free,
    logged with reason "data_end". Real delistings gap and cost money, so short
    books look slightly better here than they would live.
  • Per-share frictions (commission_per_share, half_spread_floor) are charged on
    SPLIT-ADJUSTED prices, because that is what the shared cache stores. A 1995
    AAPL bar reads $0.28 rather than ~$40, so the $/share floor is ~176bp there
    instead of 2.5bp: pre-2005 costs are materially overstated for names that
    split a lot. The milestone below prints the cost share of gross P&L so the
    distortion is always visible.
  • `rng` is stored and unused. It reserves the seed stream for a future
    stochastic fill model (partial fills, queue position); today every fill is
    deterministic given the bars, so results are reproducible without it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config                       # FIRST: puts the siblings on sys.path


@dataclass
class CostModel:
    """Per-fill and per-day frictions. `stress_mult` scales EVERY component
    linearly (0 = frictionless reference, 2 = the G7 double-cost stress)."""

    commission_per_share: float = config.COMMISSION_PER_SHARE
    commission_min: float = config.COMMISSION_MIN
    half_spread_bps: float = config.HALF_SPREAD_BPS
    half_spread_floor: float = config.HALF_SPREAD_FLOOR
    slippage_bps: float = config.SLIPPAGE_BPS
    borrow_annual: float = config.BORROW_ANNUAL
    margin_annual: float = config.MARGIN_ANNUAL
    stress_mult: float = 1.0

    def spread_per_share(self, px: float) -> float:
        """Half-spread in $/share: the bps term, floored in cents (the floor is
        what a small account actually pays on a cheap stock)."""
        return max(px * self.half_spread_bps / 1e4, self.half_spread_floor) * self.stress_mult

    def slippage_per_share(self, px: float) -> float:
        return px * self.slippage_bps / 1e4 * self.stress_mult

    def commission(self, shares: float) -> float:
        return max(self.commission_min,
                   self.commission_per_share * abs(shares)) * self.stress_mult

    def daily_borrow(self, short_mv: float) -> float:
        return short_mv * self.borrow_annual / config.TRADING_DAYS_YEAR * self.stress_mult

    def daily_margin(self, debit: float) -> float:
        return debit * self.margin_annual / config.TRADING_DAYS_YEAR * self.stress_mult


class MarketEnv:
    """Replays a MarketData window as a tradable account.

    env = MarketEnv(market, CostModel(), start, end, rng)
    obs = env.reset()
    obs, reward, done, info = env.step(target_w)   # target_w: weight of EQUITY
                                                   # per symbol, at close t
    """

    def __init__(self, market, cost=None, start=None, end=None, rng=None,
                 decision_log=None):
        self.market = market
        self.cost = cost if cost is not None else CostModel()
        self.rng = rng if rng is not None else np.random.default_rng(config.SEED)
        self.decision_log = decision_log

        self.i0 = market.pos(start, "left") if start is not None else 0
        self.i1 = market.pos(end, "right") if end is not None else len(market) - 1
        if self.i1 <= self.i0:
            raise ValueError("empty window: start %s is not before end %s"
                             % (market.dates[self.i0].date(), market.dates[self.i1].date()))
        self.n_symbols = len(market.symbols)
        self.t = self.i0
        self.done = True                       # must reset() before stepping

    # ── episode control ────────────────────────────────────────────────────────
    def reset(self) -> dict:
        self.t = self.i0
        self.cash = float(config.START_CASH)
        self.shares = np.zeros(self.n_symbols, dtype=np.float64)
        self.days_held = np.zeros(self.n_symbols, dtype=np.int64)
        self.equity = self.cash
        self.peak_equity = self.equity
        self.done = False
        self.totals = {"commission": 0.0, "spread_cost": 0.0, "slippage": 0.0,
                       "borrow": 0.0, "margin": 0.0}
        return self._obs()

    def _obs(self) -> dict:
        t = self.t
        return {
            "t": t,
            "date": self.market.dates[t],
            "features": {k: v[t] for k, v in self.market.features.items()},
            "close": self.market.close[t],
            "position_shares": self.shares.copy(),
            "cash": self.cash,
            "equity": self.equity,
            "drawdown": self.equity / self.peak_equity - 1.0,
            "days_held": self.days_held.copy(),
        }

    def _position_value(self, i: int) -> float:
        """Marked value of the book at close i. Held symbols always have a close
        there — the force-close rule guarantees it."""
        px = self.market.close[i]
        return float(np.sum(np.where(self.shares != 0.0, self.shares * px, 0.0)))

    # ── order construction ─────────────────────────────────────────────────────
    def _constrain(self, w, close_t, rejected):
        """Apply the weight-level constraints. Returns (weights, leverage_scale)."""
        w = np.where(np.isfinite(w), w, 0.0)
        # No close at t means the symbol is not listed yet: nothing to size against.
        # It cannot be a held position — those were force-closed when their data ended.
        w = np.where(np.isfinite(close_t) & (close_t > 0.0), w, 0.0)

        cap = config.MAX_NAME_WEIGHT
        over = np.abs(w) > cap
        for j in np.flatnonzero(over):
            rejected.append((self.market.symbols[j], "name_cap"))
        w = np.clip(w, -cap, cap)

        live = np.flatnonzero(w != 0.0)
        if len(live) > config.MAX_POSITIONS:
            keep = live[np.argsort(-np.abs(w[live]), kind="stable")[:config.MAX_POSITIONS]]
            drop = np.setdiff1d(live, keep)
            for j in drop:
                rejected.append((self.market.symbols[j], "position_cap"))
            w = np.where(np.isin(np.arange(len(w)), keep), w, 0.0)

        scale = 1.0
        gross, net = float(np.abs(w).sum()), float(w.sum())
        if gross > config.MAX_GROSS_LEV:
            scale = min(scale, config.MAX_GROSS_LEV / gross)
        if abs(net) * scale > config.MAX_NET_LEV:
            scale = min(scale, config.MAX_NET_LEV / abs(net))
        return w * scale, scale

    def _target_shares(self, w, close_t, equity_t):
        raw = np.zeros(self.n_symbols)
        ok = np.isfinite(close_t) & (close_t > 0.0)
        raw[ok] = w[ok] * equity_t / close_t[ok]
        if not config.WHOLE_SHARES:
            return raw
        # Truncate toward zero: rounding up could breach a cap the weights were
        # just clipped to. The epsilon absorbs the float round-trip of an
        # unchanged book (shares*px/equity*equity/px can land a hair under
        # `shares`, which would otherwise shed a share every single day).
        return np.trunc(raw + np.sign(raw) * config.SHARE_ROUND_EPS)

    def _trim_to_leverage(self, tgt, close_t, equity_t):
        """Whole-share rounding of a SHORT reduces |short|, which can push net
        leverage back over its cap. Cut whole shares off the largest position on
        the offending side until the actual share book fits. Deterministic,
        bounded, and only ever reduces |shares|."""
        px = np.where(np.isfinite(close_t), close_t, 0.0)
        tol = config.ACCOUNT_TOL
        gross_cap = config.MAX_GROSS_LEV * equity_t
        net_cap = config.MAX_NET_LEV * equity_t
        for _ in range(config.ENV_MAX_TRIM_ITERS):
            val = tgt * px
            gross, net = float(np.abs(val).sum()), float(val.sum())
            if gross > gross_cap + tol:
                excess = gross - gross_cap
                j = int(np.argmax(np.abs(val)))
            elif abs(net) > net_cap + tol:
                excess = abs(net) - net_cap
                side = 1.0 if net > 0 else -1.0     # shrink the heavy side
                j = int(np.argmax(np.where(np.sign(val) == side, np.abs(val), -1.0)))
            else:
                break
            if tgt[j] == 0.0 or px[j] <= 0.0:
                break
            # cut whole shares in one go — one share at a time would need
            # thousands of passes on a low-priced (split-adjusted) symbol
            cut = min(abs(tgt[j]), np.ceil(excess / px[j]))
            tgt[j] -= np.sign(tgt[j]) * cut
        return tgt

    def _repair_book(self, close_t, equity_t):
        """The smallest reducing trades that put the REALIZED book back inside the
        caps. Needed because the dust filter can strand a position outside a cap:
        a name that drifted to 21% wants a trim worth less than MIN_POSITION_USD,
        the order gets dropped, and the breach persists. Risk limits outrank the
        dust filter, so those repairs are executed unfiltered."""
        px = np.where(np.isfinite(close_t), close_t, 0.0)
        cap_usd = config.MAX_NAME_WEIGHT * equity_t
        want = self.shares.copy()
        over = np.abs(want * px) > cap_usd + config.ACCOUNT_TOL
        if over.any():
            allowed = np.trunc(cap_usd / np.where(px > 0.0, px, np.inf))
            want = np.where(over, np.sign(want) * allowed, want)
        return self._trim_to_leverage(want, close_t, equity_t)

    def _execute(self, tgt, ctx, reason: str, dust_filter: bool = True):
        """Trade the book toward `tgt` at the next open. `ctx` carries the row of
        prices and the day's accumulators; see step()."""
        for j in np.flatnonzero(tgt != self.shares):
            held, want = self.shares[j], tgt[j]
            sym = self.market.symbols[j]
            if not ctx["tradable"][j]:
                ctx["rejected"].append((sym, "no_data_next_open"))
                continue
            delta = want - held
            px_close = float(ctx["close_t"][j])
            exiting = (want == 0.0) and (held != 0.0)
            if dust_filter and not exiting and abs(delta) * px_close < config.MIN_POSITION_USD:
                ctx["rejected"].append((sym, "below_min_usd"))
                continue

            px_open = float(ctx["open_next"][j])
            spread_ps = self.cost.spread_per_share(px_open)
            slip_ps = self.cost.slippage_per_share(px_open)
            fill_px = px_open + (1.0 if delta > 0 else -1.0) * (spread_ps + slip_ps)
            commission = self.cost.commission(delta)
            self.cash -= delta * fill_px + commission
            self.shares[j] = want

            row = self._row(ctx["date_next"], sym, delta, fill_px, commission,
                            spread_ps * abs(delta), slip_ps * abs(delta), 0.0,
                            held * px_close / ctx["equity_t"],
                            want * px_close / ctx["equity_t"], reason)
            ctx["day"]["commission"] += commission
            ctx["day"]["spread_cost"] += row["spread_cost"]
            ctx["day"]["slippage"] += row["slippage"]
            ctx["fills"].append(row)
            self._log(row)

    # ── the step ───────────────────────────────────────────────────────────────
    def step(self, target_w, reason: str = "rebalance"):
        """Trade toward `target_w` (weight of equity per symbol, decided at close
        t) and advance one bar. Returns (obs, reward, done, info)."""
        if self.done:
            raise RuntimeError("step() after the episode ended — call reset() first")
        w = np.asarray(target_w, dtype=np.float64).ravel()
        if w.shape != (self.n_symbols,):
            raise ValueError("target_w has shape %s, expected (%d,)"
                             % (w.shape, self.n_symbols))

        t = self.t
        mkt = self.market
        close_t = mkt.close[t]
        open_next = mkt.open[t + 1]
        close_next = mkt.close[t + 1]
        date_next = mkt.dates[t + 1]
        equity_t = self.equity
        sign_before = np.sign(self.shares)          # captured pre-fill: days_held
        rejected: list = []                         # must restart when a name flips side
        fills: list = []
        day = {"commission": 0.0, "spread_cost": 0.0, "slippage": 0.0,
               "borrow": 0.0, "margin": 0.0}

        # 1. Symbols whose data ends here: force-close at the last available close.
        gone = (self.shares != 0.0) & ~np.isfinite(close_next)
        for j in np.flatnonzero(gone):
            delta = -self.shares[j]
            px = float(close_t[j])
            self.cash += -delta * px                    # cost-free, see limitations
            row = self._row(mkt.dates[t], mkt.symbols[j], delta, px, 0.0, 0.0, 0.0, 0.0,
                            self.shares[j] * px / equity_t, 0.0, "data_end")
            self.shares[j] = 0.0
            fills.append(row)
            self._log(row)

        # 2. Weight constraints, then whole-share targets on the surviving book.
        w, lev_scale = self._constrain(w, close_t, rejected)
        tgt = self._target_shares(w, close_t, equity_t)
        tgt = self._trim_to_leverage(tgt, close_t, equity_t)

        # 3. Orders: everything is decided on close-t information, filled at open t+1.
        ctx = {"close_t": close_t, "open_next": open_next, "date_next": date_next,
               "tradable": np.isfinite(open_next) & np.isfinite(close_next),
               "equity_t": equity_t, "day": day, "fills": fills, "rejected": rejected}
        self._execute(tgt, ctx, reason)
        self._execute(self._repair_book(close_t, equity_t), ctx, "risk_cap",
                      dust_filter=False)

        # 4. Financing, charged on the post-fill book marked at close t+1.
        short_mv = float(np.sum(np.where(self.shares < 0.0, -self.shares * close_next, 0.0)))
        borrow = self.cost.daily_borrow(short_mv)
        margin = self.cost.daily_margin(max(0.0, -self.cash))
        self.cash -= borrow + margin
        day["borrow"], day["margin"] = borrow, margin
        for amount, tag in ((borrow, "borrow"), (margin, "margin")):
            if amount > 0.0:
                self._log(self._row(date_next, "__CASH__", 0.0, float("nan"), 0.0, 0.0,
                                    0.0, amount, float("nan"), float("nan"), tag,
                                    side="carry"))

        # 5. Mark to market at close t+1.
        self.t = t + 1
        equity_next = self.cash + self._position_value(self.t)
        # A wiped-out account cannot compound; floor the ratio so reward stays finite.
        reward = float(np.log(max(equity_next, config.ACCOUNT_TOL) / equity_t))
        self.equity = equity_next
        self.peak_equity = max(self.peak_equity, equity_next)
        is_open = self.shares != 0.0
        self.days_held = np.where(is_open,
                                  np.where(np.sign(self.shares) == sign_before,
                                           self.days_held + 1, 1), 0)
        for k in day:
            self.totals[k] += day[k]
        self.done = (self.t >= self.i1) or (equity_next <= 0.0)

        if config.ENV_CHECK_INVARIANTS:
            self._assert_invariants(day, close_t, equity_t)

        info = {"fills": fills, "commissions": day["commission"],
                "spread_cost": day["spread_cost"], "slippage": day["slippage"],
                "borrow": day["borrow"], "margin": day["margin"],
                "rejected_orders": rejected, "leverage_scaled": lev_scale}
        return self._obs(), reward, self.done, info

    # ── logging & invariants ───────────────────────────────────────────────────
    @staticmethod
    def _row(date, symbol, shares, fill_px, commission, spread_cost, slippage, borrow,
             weight_before, weight_after, reason, side=None):
        if side is None:
            side = "buy" if shares > 0 else "sell"
        return {"date": date, "symbol": symbol, "side": side, "shares": float(shares),
                "fill_px": float(fill_px), "commission": float(commission),
                "spread_cost": float(spread_cost), "slippage": float(slippage),
                "borrow": float(borrow), "weight_before": float(weight_before),
                "weight_after": float(weight_after), "reason": reason}

    def _log(self, row) -> None:
        if self.decision_log is not None:
            self.decision_log.append(row)

    def _assert_invariants(self, day, close_t, equity_t) -> None:
        """close_t/equity_t are the DECISION reference the orders were sized on —
        the only basis on which the caps are a statement about the env's behavior
        rather than about overnight price moves."""
        tol = config.ACCOUNT_TOL
        identity = abs(self.equity - (self.cash + self._position_value(self.t)))
        assert identity <= tol, "equity identity broken by %.3e" % identity
        assert all(v >= 0.0 for v in day.values()), "negative cost component: %s" % day
        if config.WHOLE_SHARES:
            assert np.all(self.shares == np.trunc(self.shares)), "fractional shares"
        held = self.shares != 0.0
        assert np.all(np.isfinite(self.market.close[self.t][held])), "position without a price"
        val = np.where(held, self.shares * close_t, 0.0)
        n_open = int(np.count_nonzero(self.shares))
        assert n_open <= config.MAX_POSITIONS, "%d open positions" % n_open
        assert np.max(np.abs(val), initial=0.0) <= config.MAX_NAME_WEIGHT * equity_t + tol, \
            "name weight %.4f over cap" % (np.max(np.abs(val), initial=0.0) / equity_t)
        assert np.abs(val).sum() <= config.MAX_GROSS_LEV * equity_t + tol, \
            "gross leverage %.4f over cap" % (np.abs(val).sum() / equity_t)
        assert abs(val.sum()) <= config.MAX_NET_LEV * equity_t + tol, \
            "net leverage %.4f over cap" % (val.sum() / equity_t)


if __name__ == "__main__":
    # Milestone smoke test: a hard-coded momentum book over the whole cache.
    # These are DEMO parameters, not system knobs — Phase 2's genomes own the
    # real strategy space. This proves the sandbox runs and accounts honestly;
    # it is NOT a strategy claim (see the honesty line printed below).
    import datafeed

    TOP_N, LOOKBACK, SKIP, REBAL = 10, 126, 21, 21

    wanted = ["SPY", "AAPL", "MSFT", "JPM", "KO", "GE", "XOM", "PG", "JNJ", "WMT",
              "MRK", "INTC", "IBM", "T", "DIS", "CVX", "MCD", "HD", "CAT", "MMM",
              "BAC", "C", "AXP", "PFE", "CSCO", "ADBE", "AMGN", "HON", "LOW", "TXN"]
    syms = datafeed.in_cache(wanted)
    missing = sorted(set(wanted) - set(syms))
    md = datafeed.load_market(syms, start=config.DATA_START)

    log: list = []
    env = MarketEnv(md, CostModel(), rng=np.random.default_rng(config.SEED),
                    decision_log=log)
    obs = env.reset()
    n = len(md.symbols)
    close = md.close
    equity = [obs["equity"]]
    w = np.zeros(n)
    done = False
    while not done:
        t = obs["t"]
        if t >= LOOKBACK + SKIP and (t - LOOKBACK - SKIP) % REBAL == 0:
            mom = close[t - SKIP] / close[t - LOOKBACK - SKIP] - 1.0
            mom = np.where(np.isfinite(mom) & np.isfinite(close[t]), mom, -np.inf)
            pick = np.argsort(-mom, kind="stable")[:TOP_N]
            pick = [j for j in pick if np.isfinite(mom[j])]
            w = np.zeros(n)
            if pick:
                w[pick] = 1.0 / len(pick)          # long-only, equal weight
        else:
            px = close[t]                          # hold: restate the current book
            w = np.where(np.isfinite(px), env.shares * px / obs["equity"], 0.0)
        obs, reward, done, info = env.step(w)
        equity.append(obs["equity"])

    eq = np.array(equity)
    peak = np.maximum.accumulate(eq)
    costs = env.totals
    total_cost = sum(costs.values())
    gross_pnl = (eq[-1] - eq[0]) + total_cost
    years = len(eq) / config.TRADING_DAYS_YEAR

    print("arena sandbox milestone — top-%d 126d/skip-21 momentum, monthly, long-only"
          % TOP_N)
    print("  symbols     : %d of %d requested%s"
          % (len(md.symbols), len(wanted), "  (missing: %s)" % ", ".join(missing) if missing else ""))
    print("  window      : %s -> %s  (%d bars, %.1f years)"
          % (md.dates[0].date(), md.dates[-1].date(), len(md), years))
    print("  data_hash   : %s" % md.data_hash)
    print("  start cash  : $%12.2f" % eq[0])
    print("  final equity: $%12.2f" % eq[-1])
    print("  total return: %11.1f%%   (%.2f%% / yr compounded)"
          % (100 * (eq[-1] / eq[0] - 1), 100 * ((eq[-1] / eq[0]) ** (1 / years) - 1)))
    print("  max drawdown: %11.1f%%" % (100 * (eq / peak - 1).min()))
    print("  costs       : commission $%.0f  spread $%.0f  slippage $%.0f  "
          "borrow $%.0f  margin $%.0f" % (costs["commission"], costs["spread_cost"],
                                          costs["slippage"], costs["borrow"], costs["margin"]))
    print("                total $%.0f = %.1f%% of gross P&L  (%d log rows)"
          % (total_cost, 100 * total_cost / gross_pnl if gross_pnl else float("nan"), len(log)))

    # Invariant check, recomputed from scratch rather than trusted.
    pos_val = float(np.sum(np.where(env.shares != 0.0, env.shares * close[env.t], 0.0)))
    identity = abs(env.equity - (env.cash + pos_val))
    integral = bool(np.all(env.shares == np.trunc(env.shares)))
    n_open = int(np.count_nonzero(env.shares))
    gross = float(np.sum(np.abs(np.where(env.shares != 0.0, env.shares * close[env.t], 0.0))))
    print("  invariants  : equity identity %.2e | whole shares %s | %d/%d positions | "
          "gross %.2fx" % (identity, integral, n_open, config.MAX_POSITIONS,
                           gross / env.equity))
    ok = identity <= config.ACCOUNT_TOL and integral and n_open <= config.MAX_POSITIONS
    print("  result      :", "PASS" if ok else "FAIL")
    print("\n  Sandbox output is a claim about the past, not a guarantee — and not")
    print("  financial advice. This book is a fixed smoke test, never selected or")
    print("  tuned; the universe is today's survivors, so long results flatter.")
