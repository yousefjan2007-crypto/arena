"""
arena/verify.py — the test suite. Nothing gets scheduled until this is green.

docs/DESIGN.md lists eleven checks; the phases fill them in. Implemented here
(Phase 1, the sandbox):

  3. Accounting fuzz     500 seeded random-target steps: the equity identity
                         holds, limits hold post-fill, shares stay integral,
                         costs stay non-negative, and the equity path replays
                         exactly from the decision log alone.
  4. Fill timing         a decision at close t fills at open t+1, at exactly the
                         configured price adjustment, correct sign each way, and
                         never on the same bar it was decided.
  6. Cost linearity      the same trade stream at stress_mult 0/1/2 costs
                         exactly 0/c/2c per component, and the frictionless
                         equity path dominates the costed ones pointwise.
 10. No wall-clock       source scan: nothing reads the system clock (calendar
                         `now`, epoch seconds) outside run_*.py I/O boundaries.
                         The literal call spellings are deliberately absent from
                         this file's prose — the scanner is dumb on purpose.

Still to come: 1 planted leak (Phase 3), 2 determinism (Phase 4), 5 streaming
purge (Phase 3), 7 gates (Phase 5), 8 trial ledger (Phase 3), 9 genome ops
(Phase 2), 11 PBO sanity (Phase 3).

Everything here runs on a synthetic, seeded market. verify.py NEVER touches the
network or the cache: the test suite must give the same answer on a plane, on a
GitHub runner, and after a bad yfinance day.
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

import config                       # FIRST: puts the siblings on sys.path
import datafeed
from env import CostModel, MarketEnv

ok = True


def check(name, cond, detail=""):
    global ok
    cond = bool(cond)
    ok = ok and cond
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")
    return cond


# ── synthetic market (deterministic; no network, no cache) ─────────────────────
def synthetic_market(n_days: int = 520, n_syms: int = 8, seed: int = None):
    """A seeded GBM market with overnight gaps, so open != close and the fill
    model is actually exercised. Price levels differ per symbol so whole-share
    rounding bites differently across names."""
    rng = np.random.default_rng(config.SEED if seed is None else seed)
    dates = pd.bdate_range("2000-01-03", periods=n_days)      # fixed, not wall-clock
    logret = rng.normal(0.0002, 0.015, size=(n_days, n_syms))
    levels = 20.0 * (1.0 + np.arange(n_syms))                 # $20 .. $160
    close = levels * np.exp(np.cumsum(logret, axis=0))
    overnight = rng.normal(0.0, 0.006, size=(n_days, n_syms))
    open_ = np.vstack([close[:1], close[:-1]]) * np.exp(overnight)
    volume = rng.uniform(1e6, 5e6, size=(n_days, n_syms))
    symbols = ["SYN%d" % i for i in range(n_syms)]
    return datafeed.MarketData(dates, symbols, open_, close, volume)


def flat_market(n_days: int = 60, n_syms: int = 10, price: float = 5.0):
    """Constant prices, no gaps: the only thing that can move leverage is the
    account itself (whole-share rounding, costs), which is exactly what the
    boundary-churn check needs to isolate. The default price divides START_CASH
    evenly across n_syms, so an equal-weight book leaves NO rounding residue and
    the frictions alone push cash negative — the condition that puts a book that
    targets exactly the net cap a hair over it."""
    dates = pd.bdate_range("2000-01-03", periods=n_days)
    px = np.full((n_days, n_syms), price)
    volume = np.full((n_days, n_syms), 1e6)
    return datafeed.MarketData(dates, ["FLT%d" % i for i in range(n_syms)],
                               px.copy(), px.copy(), volume)


def replay_equity(log, market, upto_t: int):
    """Rebuild (cash, shares) from the decision log alone — the only evidence a
    real audit would have. Carry rows charge financing; fill rows move shares."""
    cash = float(config.START_CASH)
    shares = np.zeros(len(market.symbols))
    index = {s: j for j, s in enumerate(market.symbols)}
    for row in log:
        if row["side"] == "carry":
            cash -= row["borrow"]
            continue
        cash -= row["shares"] * row["fill_px"] + row["commission"]
        shares[index[row["symbol"]]] += row["shares"]
    px = market.close[upto_t]
    equity = cash + float(np.sum(np.where(shares != 0.0, shares * px, 0.0)))
    return cash, shares, equity


# ── 3. accounting fuzz ─────────────────────────────────────────────────────────
def test_accounting():
    print("3. Accounting fuzz (500 seeded random-target steps)")
    market = synthetic_market()
    rng = np.random.default_rng(config.SEED)
    log: list = []
    env = MarketEnv(market, CostModel(), rng=np.random.default_rng(config.SEED),
                    decision_log=log)
    obs = env.reset()
    n = len(market.symbols)

    worst_identity = 0.0
    limit_breaches = []
    negative_costs = 0
    fractional = 0
    n_steps = 0
    saw = {"below_min_usd": 0, "name_cap": 0, "position_cap": 0, "scaled": 0, "short": 0}

    while not env.done and n_steps < 500:
        t0, eq0 = obs["t"], obs["equity"]
        w = np.zeros(n)
        k = int(rng.integers(0, n + 1))
        if k:
            idx = rng.choice(n, size=k, replace=False)
            # wide enough that name caps and leverage scaling both fire often
            w[idx] = rng.normal(0.0, 0.22, size=k)            # long AND short
        obs, reward, done, info = env.step(w)
        n_steps += 1

        # equity identity, recomputed independently of the env's own bookkeeping
        px = market.close[obs["t"]]
        marked = float(np.sum(np.where(env.shares != 0.0, env.shares * px, 0.0)))
        worst_identity = max(worst_identity, abs(obs["equity"] - (env.cash + marked)))

        # limits on the post-fill book, at the reference the orders were sized on
        val = np.where(env.shares != 0.0, env.shares * market.close[t0], 0.0)
        band = 1.0 + config.LEV_EPS          # gross/net keep a hysteresis band
        if np.max(np.abs(val), initial=0.0) > config.MAX_NAME_WEIGHT * eq0 + config.ACCOUNT_TOL:
            limit_breaches.append(("name", t0))
        if np.abs(val).sum() > config.MAX_GROSS_LEV * band * eq0 + config.ACCOUNT_TOL:
            limit_breaches.append(("gross", t0))
        if abs(val.sum()) > config.MAX_NET_LEV * band * eq0 + config.ACCOUNT_TOL:
            limit_breaches.append(("net", t0))
        if np.count_nonzero(env.shares) > config.MAX_POSITIONS:
            limit_breaches.append(("positions", t0))

        if not np.all(env.shares == np.trunc(env.shares)):
            fractional += 1
        for key in ("commissions", "spread_cost", "slippage", "borrow", "margin"):
            if info[key] < 0.0:
                negative_costs += 1
        for _sym, why in info["rejected_orders"]:
            if why in saw:
                saw[why] += 1
        saw["scaled"] += int(info["leverage_scaled"] < 1.0)
        saw["short"] += int(np.any(env.shares < 0))

    check("ran 500 steps", n_steps == 500, "%d steps, %d log rows" % (n_steps, len(log)))
    check("equity == cash + shares.close every step", worst_identity <= config.ACCOUNT_TOL,
          "worst |error| = %.2e" % worst_identity)
    check("limits hold post-fill (name/gross/net/count)", not limit_breaches,
          "%d breaches" % len(limit_breaches))
    check("shares stay integral", fractional == 0)
    check("every cost component >= 0", negative_costs == 0)
    check("total costs > 0 (the fuzz actually traded)",
          sum(env.totals.values()) > 0.0,
          "commission $%.0f spread $%.0f slip $%.0f borrow $%.2f margin $%.2f"
          % (env.totals["commission"], env.totals["spread_cost"], env.totals["slippage"],
             env.totals["borrow"], env.totals["margin"]))
    check("constraint paths exercised", saw["short"] and saw["scaled"] and saw["below_min_usd"],
          "shorts %d, lev-scaled %d, dust-rejected %d, name-capped %d, count-capped %d"
          % (saw["short"], saw["scaled"], saw["below_min_usd"], saw["name_cap"],
             saw["position_cap"]))

    # Boundary churn: a book sitting exactly ON the net cap must not be dragged
    # around by rounding. Go fully invested at 1.0 on a flat market, then hold it
    # (restating the book's own weights, the way a between-rebalance strategy
    # does). Costs put cash slightly negative, so net sits a hair above 1.0 for
    # the rest of the run — inside LEV_EPS, so nothing should scale or repair.
    flat = flat_market()
    fenv = MarketEnv(flat, CostModel())
    fobs = fenv.reset()
    nf = len(flat.symbols)
    scaled = repairs = 0
    for k in range(40):
        px = flat.close[fobs["t"]]
        w = (np.full(nf, 1.0 / nf) if k == 0
             else fenv.shares * px / fobs["equity"])          # hold what we hold
        fobs, _, fdone, finfo = fenv.step(w)
        scaled += int(finfo["leverage_scaled"] < 1.0)
        repairs += sum(1 for r in finfo["fills"] if r["reason"] == "risk_cap")
        if fdone:
            break
    net = float(np.sum(fenv.shares * flat.close[fenv.t])) / fenv.equity
    check("a 1.0-gross book on a flat market never self-deleverages",
          scaled == 0 and repairs == 0 and net > config.MAX_NET_LEV,
          "net settled at %.5f (cash $%.2f), %d scale events, %d risk_cap trades"
          % (net, fenv.cash, scaled, repairs))

    # The 8-symbol market cannot reach MAX_POSITIONS, so the count cap gets its
    # own short run on a wider one.
    wide = synthetic_market(n_days=40, n_syms=2 * config.MAX_POSITIONS)
    wenv = MarketEnv(wide, CostModel())
    wobs = wenv.reset()
    capped = 0
    for _ in range(30):
        wobs, _, wdone, winfo = wenv.step(np.full(len(wide.symbols), 0.05))
        capped += sum(1 for _s, why in winfo["rejected_orders"] if why == "position_cap")
        if wdone:
            break
    check("max positions enforced on a %d-symbol market" % len(wide.symbols),
          np.count_nonzero(wenv.shares) <= config.MAX_POSITIONS and capped > 0,
          "%d open, %d count-cap rejections" % (np.count_nonzero(wenv.shares), capped))

    r_cash, r_shares, r_equity = replay_equity(log, market, env.t)
    check("decision log replays the share book", np.array_equal(r_shares, env.shares))
    check("decision log replays cash", abs(r_cash - env.cash) <= config.ACCOUNT_TOL,
          "%.10f vs %.10f" % (r_cash, env.cash))
    check("decision log replays equity", abs(r_equity - env.equity) <= config.ACCOUNT_TOL,
          "$%.6f vs $%.6f" % (r_equity, env.equity))


# ── 4. fill timing ─────────────────────────────────────────────────────────────
def test_fill_timing():
    print("\n4. Fill timing (decide at close t, fill at open t+1)")
    market = synthetic_market(n_days=40, n_syms=4)
    cost = CostModel()
    log: list = []
    env = MarketEnv(market, cost, decision_log=log)
    obs = env.reset()
    n = len(market.symbols)

    # BUY: one name, decided at close t0.
    t0 = obs["t"]
    w = np.zeros(n)
    w[1] = config.MAX_NAME_WEIGHT
    shares_before = env.shares.copy()
    obs, _, _, info = env.step(w)

    check("no position change on the decision bar", np.array_equal(shares_before, np.zeros(n)))
    check("exactly one fill", len(info["fills"]) == 1)
    row = info["fills"][0]
    px_open = float(market.open[t0 + 1, 1])
    expected = px_open * (1.0 + (config.HALF_SPREAD_BPS + config.SLIPPAGE_BPS) / 1e4)
    check("fill dated t+1, not t", row["date"] == market.dates[t0 + 1],
          "%s (decided %s)" % (row["date"].date(), market.dates[t0].date()))
    check("buy fills at open + half-spread + slippage", abs(row["fill_px"] - expected) < 1e-12,
          "%.10f vs %.10f (open %.10f)" % (row["fill_px"], expected, px_open))
    check("buy fill is above the open", row["fill_px"] > px_open)
    check("shares sized off close t, not open t+1",
          row["shares"] == np.trunc(config.MAX_NAME_WEIGHT * config.START_CASH
                                    / market.close[t0, 1] + config.SHARE_ROUND_EPS),
          "%.0f shares" % row["shares"])
    check("commission = max(min, bps x notional)",
          abs(row["commission"] - max(config.COMMISSION_MIN,
                                      row["shares"] * px_open * config.COMMISSION_BPS / 1e4)
              ) < 1e-12,
          "$%.4f on $%.0f notional" % (row["commission"], row["shares"] * px_open))

    # SELL: close the same position at the next decision.
    t1 = obs["t"]
    obs, _, _, info = env.step(np.zeros(n))
    check("exactly one closing fill", len(info["fills"]) == 1)
    srow = info["fills"][0]
    px_open = float(market.open[t1 + 1, 1])
    expected = px_open * (1.0 - (config.HALF_SPREAD_BPS + config.SLIPPAGE_BPS) / 1e4)
    check("sell fills at open - half-spread - slippage", abs(srow["fill_px"] - expected) < 1e-12,
          "%.10f vs %.10f (open %.10f)" % (srow["fill_px"], expected, px_open))
    check("sell fill is below the open", srow["fill_px"] < px_open)
    check("position is flat after the exit", np.count_nonzero(env.shares) == 0)

    # obs contract: days_held counts bars since the position opened, and restarts
    # when a name flips side (a flip is a new position, not an older one).
    short_w = np.zeros(n)
    short_w[1] = -config.MAX_NAME_WEIGHT
    obs, _, _, _ = env.step(short_w)
    d1 = obs["days_held"][1]
    obs, _, _, _ = env.step(short_w)
    d2 = obs["days_held"][1]
    obs, _, _, _ = env.step(-short_w)
    check("days_held counts bars held, resets on a side flip",
          d1 == 1 and d2 == 2 and obs["days_held"][1] == 1,
          "short %d -> %d, flipped long -> %d" % (d1, d2, obs["days_held"][1]))

    # No same-day round trips: never a buy and a sell in one name on one date.
    for _ in range(20):
        if env.done:
            break
        env.step(np.where(np.arange(n) == (env.t % n), config.MAX_NAME_WEIGHT, 0.0))
    seen = {}
    same_day_roundtrip = 0
    for r in log:
        if r["side"] == "carry":
            continue
        key = (r["date"], r["symbol"])
        if key in seen and np.sign(seen[key]) != np.sign(r["shares"]):
            same_day_roundtrip += 1
        seen[key] = r["shares"]
    check("no same-day round trips", same_day_roundtrip == 0,
          "%d fills across %d dates" % (len(seen), len({d for d, _ in seen})))
    check("NO_INTRADAY_EXITS is structural, not optional", config.NO_INTRADAY_EXITS)


# ── 6. cost linearity ──────────────────────────────────────────────────────────
def test_cost_linearity():
    print("\n6. Cost linearity (stress_mult 0 / 1 / 2)")
    market = synthetic_market(n_days=140, n_syms=6)
    n = len(market.symbols)
    rng = np.random.default_rng(config.SEED)

    # A fixed SHARE plan, replayed identically at every stress level. Driving the
    # env with weights instead would let the diverging equity paths change the
    # share counts, and then the cost streams would not be comparable at all.
    # Sized off START_CASH (not the run's own equity) and kept well inside every
    # cap, so no constraint clips — clipping is equity-dependent and would
    # de-synchronise the three runs just as badly.
    plan = []
    held = np.zeros(n)
    for step in range(120):
        if step % 10 == 0:
            frac = rng.normal(0.0, 0.03, size=n)               # ~3% of the account per name
            held = np.trunc(frac * config.START_CASH / market.close[step])
        plan.append(held.copy())

    runs = {}
    for mult in (0.0, 1.0, 2.0):
        env = MarketEnv(market, CostModel(stress_mult=mult))
        obs = env.reset()
        equity = [obs["equity"]]
        for target_shares in plan:
            px = market.close[obs["t"]]
            obs, _, done, _ = env.step(target_shares * px / obs["equity"])
            equity.append(obs["equity"])
            if done:
                break
        runs[mult] = (dict(env.totals), np.array(equity))

    base = runs[1.0][0]
    zero = runs[0.0][0]
    double = runs[2.0][0]
    check("stress 0 is exactly frictionless", all(v == 0.0 for v in zero.values()),
          str({k: round(v, 12) for k, v in zero.items()}))
    for comp in ("commission", "spread_cost", "slippage", "borrow"):
        check("%s doubles exactly at 2x" % comp,
              abs(double[comp] - 2.0 * base[comp]) < 1e-9 and base[comp] > 0.0,
              "$%.6f -> $%.6f" % (base[comp], double[comp]))
    # Margin interest is charged on the realized cash balance, which necessarily
    # differs once costs differ, so it is the one component that cannot be
    # exactly linear. This plan never runs a debit, so all three read zero.
    check("margin excluded from linearity (no debit in this plan)",
          zero["margin"] == base["margin"] == double["margin"] == 0.0)

    e0, e1, e2 = runs[0.0][1], runs[1.0][1], runs[2.0][1]
    check("frictionless equity >= costed equity pointwise",
          np.all(e0 >= e1 - 1e-9) and np.all(e1 >= e2 - 1e-9),
          "final $%.2f >= $%.2f >= $%.2f" % (e0[-1], e1[-1], e2[-1]))


# ── 10. no wall-clock in compute paths ─────────────────────────────────────────
def test_no_wallclock():
    print("\n10. No wall-clock outside run_*.py I/O boundaries")
    pattern = re.compile(r"datetime\.now|Timestamp\.now|time\.time\(")
    offenders = []
    scanned = 0
    for path in sorted(glob.glob(os.path.join(config.ROOT, "*.py"))):
        name = os.path.basename(path)
        if name.startswith("run_"):
            continue                      # one-shot entry points: I/O boundary by design
        scanned += 1
        with open(path) as f:
            for lineno, line in enumerate(f, 1):
                if pattern.search(line) and "# io-boundary" not in line:
                    offenders.append("%s:%d" % (name, lineno))
    check("no wall-clock in %d scanned modules" % scanned, not offenders,
          ", ".join(offenders) if offenders else "clean")


def main() -> int:
    test_accounting()
    test_fill_timing()
    test_cost_linearity()
    test_no_wallclock()
    print("\nVERIFY:", "ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
