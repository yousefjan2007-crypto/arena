"""
The daily pre-open paper session: reconcile yesterday, check the arming gate,
compute today's orders — and, only if armed and explicitly asked, submit them.

    python3 run_paper.py                 dry: compute, log intended orders, submit nothing
    python3 run_paper.py --submit        submit — IF the arming gate is closed
    python3 run_paper.py --send          deliver the alerts instead of printing them

THE ARMING GATE IS THE POINT OF THIS FILE. docs/DESIGN.md's graduation ladder says
paper trading begins when "the same champion survives 3 consecutive weekly deep
evals", and that sentence is the difference between a system that trades what it
proved and one that trades whatever it happened to be holding. So submission is
allowed only when ALL FOUR of these hold (arming_status below, and it prints every
one of them whether they pass or fail):

  1. a champion pointer exists at all;
  2. state/deepeval_history.csv records that hash passing ALL TEN GATES in a
     complete run — the `candidates` field's `hash:1:` form, read where the
     evidence actually lives rather than inferred;
  3. there are at least config.PAPER_ARM_CONSECUTIVE complete deep-eval entries;
  4. the latest config.PAPER_ARM_CONSECUTIVE of them all name that hash as
     `champion_hash_after` — it held the pointer through every one of them.

Anything less and the session computes its orders, logs them as INTENDED, submits
nothing, and says which of the four failed. `--dry` (the default) refuses on top
of that, so a fully armed system still needs a deliberate `--submit`.

WHAT THE SIM-SHADOW IS, AND WHERE THE SEAM RUNS. Each session builds a FRESH
StrategyAgent on today's market and replays it from the paper stage's own start
date, with the sandbox's own account and the sandbox's own fills at the next
open. That episode is the shadow: the decision handed to the broker is the
sandbox's decision, taken on the SANDBOX's book. The ORDERS are the difference
between that decision and the BROKER's book, which is a different book — real
fills, real slippage, real equity — and the two drift apart. That drift is not a
defect to be engineered away; it is the quantity the paper stage exists to
measure, and it lands in state/paper_shadow.csv as tracking_error_bps. One
consequence to be explicit about: a stop is evaluated against the sandbox's entry
price, not the paper account's fill price.

The shadow episode starts at the FIRST DATE IN state/paper_ledger.csv, not in
1995, for two reasons: the paper account started flat with config.START_CASH on
that date and the shadow has to start where the thing it shadows started, and a
full-history episode with real refits is hours of compute that a pre-open session
does not have. Model families still fit on all history available at each refit —
the agent's design matrix is the whole panel; only the ACCOUNT starts late.

NO WALL CLOCK IN ANY COMPUTED QUANTITY. Like run_generation and run_deepeval this
file is an I/O boundary: it parses arguments, reads cache mtimes for the staleness
guard, and talks to a broker. Every such call is commented `io-boundary`. Nothing
it measures is derived from the clock — the session's date is the market's last
bar, and the reconciliation anchor is the ledger's own last date, so two runs on
the same data produce the same rows and the ledger append is idempotent.

THIS FILE NEVER GOES LIVE. It refuses to run at all under
config.EXECUTION_MODE == "live" (that is the only place the word appears here),
the broker it imports hardcodes paper=True, and the go-live evidence table it
prints is addressed to a human. The system never self-starts live.
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np
import pandas as pd

import config                       # FIRST: puts the siblings on sys.path
import alerts_arena
import broker_paper
import datafeed
import evolution
import features as arena_features
import registry
import run_deepeval
import run_generation
from env import CostModel, MarketEnv
from strategy import StrategyAgent

LEDGER_FILE = "paper_ledger.csv"
SHADOW_FILE = "paper_shadow.csv"

# `date` is the DECISION date for an intended/submitted row (the last bar the
# decision was taken on; the order fills at the NEXT open) and the FILL date for a
# reconciled row. The two are one bar apart on purpose — that is the env's clock,
# and collapsing them would hide the only thing this ledger is for.
# `status`: intended | submitted | rejected | <Alpaca's terminal status, e.g.
# filled, canceled, expired>. A row is never rewritten; a decision that changed
# within one session appends a second row and the LAST row for a (date, symbol)
# wins.
LEDGER_COLUMNS = ("date", "symbol", "side", "qty", "fill_px", "order_id", "status",
                  "genome_hash", "reason")

# One row per session that had a paper account to compare against. sim_net is the
# shadow episode's net return for that bar; paper_net is the account's own
# realised return over the same session (equity / last_equity - 1, read pre-open
# in one call). tracking_error_bps is (paper - sim) in basis points.
SHADOW_COLUMNS = ("date", "genome_hash", "sim_net", "paper_net", "tracking_error_bps",
                  "fill_slippage_bps", "n_fills", "paper_equity", "paper_last_equity",
                  "sim_equity")


def ledger_path(state_dir=None) -> str:
    return os.path.join(state_dir or config.STATE_DIR, LEDGER_FILE)


def shadow_path(state_dir=None) -> str:
    return os.path.join(state_dir or config.STATE_DIR, SHADOW_FILE)


# ── the arming gate ────────────────────────────────────────────────────────────
def complete_entries(rows) -> list:
    """The deep-eval entries the graduation trigger counts, oldest first.

    Two filters, both stated by run_deepeval.HISTORY_COLUMNS' own comment.
    `complete = 0` means the time budget stopped that run before every candidate
    was measured, and a half-measured week is not a week the champion survived, so
    those rows are dropped rather than counted or treated as a break in the
    streak. And the file holds ONE ROW PER RUN, not per week — re-running a
    generation appends another row — so entries are grouped by `generation` and
    the LAST row for each generation stands, that being the decision that was
    actually left in place.
    """
    by_gen = {}
    for row in rows:
        if str(row.get("complete", "")).strip() not in ("1", "True", "true"):
            continue
        try:
            gen = int(row.get("generation"))
        except (TypeError, ValueError):
            continue
        by_gen[gen] = row                      # later row for a generation wins
    return [by_gen[g] for g in sorted(by_gen)]


def gates_passed_in(row, ghash: str) -> bool:
    """Did `ghash` pass all ten gates in this entry?

    The `candidates` field packs `hash:all_pass:failed` per candidate, ";"
    separated — run_deepeval's own idiom, and a compound field a reader must SPLIT
    rather than compare. `all_pass` is the middle part.
    """
    for item in str(row.get("candidates", "")).split(";"):
        parts = item.split(":")
        if len(parts) >= 2 and parts[0] == ghash:
            return parts[1] == "1"
    return False


def arming_status(rows, champion_hash, n_required: int = None) -> dict:
    """Is run_paper allowed to submit? Pure: rows in, verdict out.

    Returns {"armed", "reason", "checks": [(name, passed, detail)], "entries",
    "n_required"}. Every check is reported whether it passed or failed, because
    "UNARMED" with no reason is exactly the kind of shrug this project treats as a
    bug — and because the four checks are the operational reading of DESIGN's one
    sentence about graduation, so they should be legible to someone holding only
    that sentence.
    """
    n = int(config.PAPER_ARM_CONSECUTIVE if n_required is None else n_required)
    entries = complete_entries(rows)
    latest = entries[-n:] if n > 0 else []
    checks = []

    has_champ = bool(champion_hash)
    checks.append(("a champion pointer exists", has_champ,
                   champion_hash if has_champ else
                   "state/champion.json does not exist — nothing has ever passed the "
                   "ten gates, so there is nothing to trade"))

    passed = has_champ and any(gates_passed_in(r, champion_hash) for r in entries)
    checks.append(("deepeval_history records it passing all ten gates", passed,
                   "found in the candidates field of a complete entry" if passed else
                   "no complete deep-eval entry records %s with all_pass=1"
                   % (champion_hash or "(no champion)")))

    enough = len(entries) >= n
    checks.append(("at least %d complete deep-eval entries" % n, enough,
                   "%d complete entr%s on record (generations %s)"
                   % (len(entries), "y" if len(entries) == 1 else "ies",
                      ", ".join(str(r["generation"]) for r in entries[-n:]) or "none")))

    held = bool(has_champ and enough
                and all(r.get("champion_hash_after") == champion_hash for r in latest))
    checks.append(("it held the pointer through the latest %d" % n, held,
                   " -> ".join("gen %s: %s" % (r["generation"],
                                               r.get("champion_hash_after") or "(none)")
                               for r in latest) or "no entries to check"))

    armed = all(c[1] for c in checks)
    reason = ("armed: %s survived %d consecutive complete deep evals"
              % (champion_hash, n) if armed else
              "; ".join(name for name, passed_, _d in checks if not passed_))
    return {"armed": armed, "reason": reason, "checks": checks,
            "entries": entries, "latest": latest, "n_required": n}


# ── append-only state files ────────────────────────────────────────────────────
def _read_csv(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def read_ledger(state_dir=None) -> list:
    return _read_csv(ledger_path(state_dir))


def read_shadow(state_dir=None) -> list:
    return _read_csv(shadow_path(state_dir))


def _append(path: str, columns, rows) -> int:
    """Append rows, refusing to append into a file of another shape.

    csv.writer appends POSITIONALLY, so a file written under an older column list
    would take today's fields silently misaligned. run_deepeval.append_history
    makes the same check for the same reason: a misaligned record of what was
    ordered is worse than no record.
    """
    if not rows:
        return 0
    new = not os.path.exists(path)
    if not new:
        with open(path, newline="") as f:
            header = next(csv.reader(f), [])
        if tuple(header) != tuple(columns):
            raise ValueError(
                "%s has columns\n    %s\nbut this version writes\n    %s\n"
                "Appending would misalign every field. Move the old file aside or "
                "migrate it; nothing was written."
                % (path, ",".join(header), ",".join(columns)))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(columns)
        for row in rows:
            w.writerow([row.get(c, "") for c in columns])
    return len(rows)


def _row_key(row) -> tuple:
    return tuple(str(row.get(c, "")) for c in LEDGER_COLUMNS)


def append_ledger(rows, state_dir=None) -> dict:
    """Append order rows idempotently. Returns {"written", "skipped"}.

    TWO KEYS, because the two kinds of row mean different things.

      a FILL row is identified by its `order_id`: the broker's own identifier for
      an event that happened once. Re-reading the same closed order on a later
      session must not double-count it, and the reconciliation window deliberately
      overlaps (it is anchored on a DATE) so that nothing is missed.

      an INTENDED/SUBMITTED row has no order id, so it is identified by its whole
      content. Re-running a session on the same data recomputes the same decision
      and writes nothing — the idempotency this file's whole re-runnability rests
      on. A decision that genuinely CHANGED within one session (the cache
      refreshed between two runs) differs in some field and is appended as the
      second decision it is; the last row for a (date, symbol) is the live one.
    """
    existing = read_ledger(state_dir)
    seen_ids = {r.get("order_id") for r in existing if r.get("order_id")}
    seen_rows = {_row_key(r) for r in existing}
    fresh, skipped = [], 0
    for row in rows:
        oid = str(row.get("order_id", "") or "")
        key = _row_key(row)
        if (oid and oid in seen_ids) or (not oid and key in seen_rows):
            skipped += 1
            continue
        fresh.append(row)
        seen_ids.add(oid) if oid else None
        seen_rows.add(key)
    return {"written": _append(ledger_path(state_dir), LEDGER_COLUMNS, fresh),
            "skipped": skipped}


def append_shadow(row, state_dir=None) -> bool:
    """One shadow row per session date. A re-run on the same bar writes nothing."""
    if any(r.get("date") == str(row.get("date")) for r in read_shadow(state_dir)):
        return False
    return bool(_append(shadow_path(state_dir), SHADOW_COLUMNS, [row]))


# ── the sandbox's decision for the next open ───────────────────────────────────
def replay_book(log, market):
    """(shares, cash) from the decision log alone — env.py's documented replay.

    The env keeps its book inside an object that run_episode does not return, and
    the paper session needs the sandbox's terminal positions to evaluate stops.
    Rebuilding them from the log rather than reaching into the env is the same
    audit an outsider could do, and run_paper prints how far the replayed equity
    lands from the episode's own, which is a live check that the log explains the
    numbers beside it.
    """
    cash = float(config.START_CASH)
    shares = np.zeros(len(market.symbols))
    index = {s: j for j, s in enumerate(market.symbols)}
    for row in log:
        if row["side"] == "carry":
            cash -= row["borrow"]
            continue
        cash -= row["shares"] * row["fill_px"] + row["commission"]
        shares[index[row["symbol"]]] += row["shares"]
    return shares, cash


def _last_fit_step(fit_audit, market, i0: int):
    """The episode step at which the model was last refitted, or None."""
    if not fit_audit:
        return None
    idx = int(market.dates.searchsorted(fit_audit[-1]["fit_date"]))
    return idx - i0


def live_decision(agent, market, shares, equity, peak_equity, i0: int, fit_audit=None) -> dict:
    """The target weights the sandbox would carry into the NEXT open.

    This is the one bar run_episode cannot reach: its loop decides at bar t and
    fills at t+1, so it stops deciding at the second-to-last bar, while a paper
    session at 09:00 ET is standing exactly on the last bar with the next open
    ahead of it. So the loop body is executed once more here — and ONLY the loop
    body: every quantity comes from the agent's own methods (`_scores`,
    `_targets`, `_stopped`, `_overlay_scale`, `_fit`), so the decision the broker
    is handed cannot drift from the decision the sandbox scores. Nothing about
    portfolio construction, sizing, stops or regime overlays is restated here;
    verify.py pins the equivalence directly (test 12).

    The cadences are recovered rather than re-derived: `step` is bars since the
    episode's start (run_episode's own counter), the genome rebalances when
    `step % rebalance_days == 0` and refits when the model is `refit_days` steps
    old, and the last fit's step comes from the fit audit.
    """
    t = len(market) - 1
    step = t - i0
    genome = agent.genome
    out = {"t": t, "date": market.dates[t], "step": step, "refitted": False,
           "rebalanced": False}

    if agent.is_model:
        last_fit = _last_fit_step(fit_audit, market, i0)
        if last_fit is None or step - last_fit >= genome.signal.refit_days:
            out["refitted"] = bool(agent._fit(t, fit_audit))            # noqa: SLF001

    if step % genome.portfolio.rebalance_days == 0:
        tradable = np.isfinite(market.close[t]) & (market.close[t] > 0.0)
        agent._target_w = agent._targets(                               # noqa: SLF001
            t, agent._scores(t, tradable), tradable)                    # noqa: SLF001
        out["rebalanced"] = True

    hit = agent._stopped(t, shares)                                     # noqa: SLF001
    if hit.any():
        agent._target_w = np.where(hit, 0.0, agent._target_w)           # noqa: SLF001
    scale, why = agent._overlay_scale(t, equity / peak_equity - 1.0)    # noqa: SLF001

    causes = (["stop"] if hit.any() else [])
    if why is not None:
        causes.append(why)
    out["target_w"] = agent._target_w * scale                           # noqa: SLF001
    out["scale"] = scale
    out["reason"] = "+".join(causes) if causes else "rebalance"
    out["n_stopped"] = int(hit.sum())
    return out


def share_targets(env, target_w, close_t, equity: float):
    """Weights -> the whole-share book the env would hold, by the env's own rules.

    The three calls are MarketEnv's, in MarketEnv's order: clip to the per-name
    cap and the position count, truncate to whole shares, then trim whatever the
    rounding pushed back over a leverage cap. Calling them rather than restating
    them is the point — the live order and the simulated order have to round the
    same way or the tracking error measures arithmetic instead of execution.
    """
    rejected = []
    w, lev_scale = env._constrain(np.asarray(target_w, float), close_t, rejected)  # noqa: SLF001
    tgt = env._target_shares(w, close_t, equity)                        # noqa: SLF001
    tgt = env._trim_to_leverage(tgt, close_t, equity)                   # noqa: SLF001
    return tgt, w, lev_scale, rejected


def order_deltas(env, market, target, current, close_t, equity: float, reason: str) -> dict:
    """The orders that take `current` to `target`, with the env's dust filter.

    Mirrors MarketEnv._execute's two passes: the ordinary trades, where an order
    worth less than config.MIN_POSITION_USD is dropped as dust (a full EXIT is
    exempt, or a sub-minimum position could never be closed), and then the
    risk-cap repair, which is executed UNFILTERED because a position that drifted
    outside a cap must not be left there by the dust rule. Both passes are the
    env's, and the second one calls env._repair_book directly.
    """
    orders, dropped = [], []

    def _emit(tgt_book, book, why, dust):
        for j in np.flatnonzero(tgt_book != book):
            sym = market.symbols[j]
            px = float(close_t[j])
            if not np.isfinite(px) or px <= 0.0:
                dropped.append((sym, "no_price"))
                continue
            delta = float(tgt_book[j] - book[j])
            exiting = (tgt_book[j] == 0.0) and (book[j] != 0.0)
            if dust and not exiting and abs(delta) * px < config.MIN_POSITION_USD:
                dropped.append((sym, "below_min_usd"))
                continue
            orders.append({"symbol": sym, "side": "buy" if delta > 0 else "sell",
                           "qty": int(abs(delta)), "reason": why,
                           "shares_before": int(book[j]), "shares_after": int(tgt_book[j])})
            book[j] = tgt_book[j]

    book = np.asarray(current, dtype=np.float64).copy()
    _emit(np.asarray(target, dtype=np.float64), book, reason, True)
    env.shares = book.copy()                    # the repair reads the POST-fill book
    _emit(env._repair_book(close_t, equity), book, "risk_cap", False)   # noqa: SLF001
    return {"orders": orders, "dropped": dropped, "book_after": book}


# ── the shadow ─────────────────────────────────────────────────────────────────
def fill_slippage_bps(fills, market) -> float:
    """Median SIGNED adverse slippage of a session's fills, against that day's open.

    Positive means the paper account did worse than the open it was aiming at —
    paid more buying, received less selling — which is the direction that costs
    money and the one DESIGN's go-live criterion bounds. A fill on a date the
    cache has no bar for is skipped rather than guessed at.
    """
    dates = {str(d.date()): i for i, d in enumerate(market.dates)}
    index = {s: j for j, s in enumerate(market.symbols)}
    slips = []
    for f in fills:
        i, j = dates.get(str(f.get("date"))), index.get(f.get("symbol"))
        px = float(f.get("fill_px") or float("nan"))
        if i is None or j is None or not np.isfinite(px) or int(f.get("qty") or 0) <= 0:
            continue
        open_px = float(market.open[i, j])
        if not np.isfinite(open_px) or open_px <= 0.0:
            continue
        sign = 1.0 if str(f.get("side")).lower() == "buy" else -1.0
        slips.append(sign * (px - open_px) / open_px * 1e4)
    return float(np.median(slips)) if slips else float("nan")


def _float(value, default=float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out


def breach_spell_start(shadow_rows, limit: float):
    """The date the CURRENT unbroken run of tracking-error breaches began, or None.

    This is the anti-spam key, and it is what makes "alert on breaches" mean one
    message per divergence rather than one per day inside a five-day one. It moves
    only when a NEW spell starts, so a spell that ends and a later one that begins
    are two different states and both are heard.
    """
    start = None
    for row in shadow_rows:
        te = abs(_float(row.get("tracking_error_bps")))
        if np.isfinite(te) and te > limit:
            start = start or row.get("date")
        else:
            start = None
    return start


def evidence_table(shadow_rows) -> dict:
    """DESIGN's go-live criteria, measured — never acted on.

    "Go-live recommendation requires: median |fill slippage| < 10 bps,
    corr(daily paper, sim-shadow) >= 0.80, cumulative paper return inside the
    shadow's bootstrap 90% band, no kill-switch trips in 60 days." The cumulative
    band is the one criterion NOT computed here: it needs the bootstrap machinery
    over a paper history that does not exist yet, and reporting a number for it
    before there is one would be the opposite of what this table is for. Its row
    says so.
    """
    sim = np.array([_float(r.get("sim_net")) for r in shadow_rows])
    paper = np.array([_float(r.get("paper_net")) for r in shadow_rows])
    slip = np.array([_float(r.get("fill_slippage_bps")) for r in shadow_rows])
    te = np.array([abs(_float(r.get("tracking_error_bps"))) for r in shadow_rows])
    both = np.isfinite(sim) & np.isfinite(paper)

    corr = float("nan")
    if both.sum() >= 3 and np.std(sim[both]) > 0 and np.std(paper[both]) > 0:
        corr = float(np.corrcoef(sim[both], paper[both])[0, 1])
    med_slip = float(np.median(np.abs(slip[np.isfinite(slip)]))) \
        if np.isfinite(slip).any() else float("nan")
    recent = te[np.isfinite(te)][-60:]
    trips = int((recent > config.PAPER_MAX_TE_BPS).sum())

    rows = [("paper days", len(shadow_rows), ">= %d" % config.PAPER_MIN_DAYS,
             len(shadow_rows) >= config.PAPER_MIN_DAYS),
            ("corr(paper, sim-shadow)", corr, ">= %.2f" % config.PAPER_MIN_CORR,
             np.isfinite(corr) and corr >= config.PAPER_MIN_CORR),
            ("median |fill slippage| bps", med_slip, "< %.0f" % config.PAPER_MAX_SLIPPAGE_BPS,
             np.isfinite(med_slip) and med_slip < config.PAPER_MAX_SLIPPAGE_BPS),
            ("tracking trips in last 60", trips, "== 0", trips == 0),
            ("cumulative inside bootstrap band", float("nan"),
             "not computed before %d days" % config.PAPER_MIN_DAYS, False)]
    return {"rows": rows, "all_pass": all(r[3] for r in rows),
            "corr": corr, "median_slippage_bps": med_slip, "trips": trips,
            "n_days": len(shadow_rows)}


# ── inputs ─────────────────────────────────────────────────────────────────────
def load_market():
    """Today's market and panel, cache-first and OFFLINE.

    The same six lines run_generation and run_deepeval open with, deliberately not
    factored out of either (see run_deepeval.load_market). No --refresh option
    exists here on purpose: the generation job refreshes the cache twice a day, a
    network fetch inside the order path is a failure mode with a market open
    behind it, and a session that cannot see yesterday's bar should refuse rather
    than reach for one.
    """
    universe = config.import_sibling("universe", config.SIGNAL_LAB)
    wanted = universe.build_universe()[0]
    symbols = datafeed.in_cache(wanted)[:config.UNIVERSE_SIZE]
    market = datafeed.load_market(symbols, start=config.DATA_START)
    arena_features.build_features(market)
    return market, len(wanted)


def shadow_episode(entry, market, ledger_rows):
    """Replay the champion from the paper stage's start to today's last bar.

    Returns (agent, episode, shares, equity, peak, i0, fit_audit). The start date
    is the first date any row of state/paper_ledger.csv carries — the day this
    account began — and for a first-ever session, where there is no such row, the
    market's second-to-last bar, giving the minimum one-step episode MarketEnv
    accepts. That first session's book is flat either way: nothing has been
    ordered yet, and the orders it produces are diffed against the broker.
    """
    dates = sorted({r.get("date") for r in ledger_rows if r.get("date")})
    start = pd.Timestamp(dates[0]) if dates else market.dates[-2]
    start = min(start, market.dates[-2])       # a 1-bar window is not an episode
    agent = StrategyAgent(evolution.entry_genome(entry), market, CostModel())
    log, fit_audit = [], []
    episode = agent.run_episode(env_start=start, decision_log=log, fit_audit=fit_audit)
    shares, cash = replay_book(log, market)
    equity = cash + float(np.sum(np.where(shares != 0.0, shares * market.close[-1], 0.0)))
    peak = float(np.max(agent._equity))                                 # noqa: SLF001
    i0 = int(market.pos(start, "left"))
    return {"agent": agent, "episode": episode, "shares": shares, "cash": cash,
            "equity": equity, "peak": peak, "i0": i0, "fit_audit": fit_audit,
            "start": start, "log_rows": len(log),
            "drift": abs(equity - float(agent._equity[-1]))}            # noqa: SLF001


# ── the run ────────────────────────────────────────────────────────────────────
def print_honesty(armed: bool, submitted: int) -> None:
    print("\n  The paper account is a MEASUREMENT, not a result. Ten gates and a")
    print("  six-year vault are evidence about the past; the only thing that has")
    print("  never been simulated is what a broker actually fills, which is what")
    print("  this stage buys and why it runs for %d trading days before anyone"
          % config.PAPER_MIN_DAYS)
    print("  reads the table. Going live is a human decision: nothing in this repo")
    print("  writes EXECUTION_MODE, and the broker adapter hardcodes paper=True.")
    if not armed:
        print("  Nothing was submitted: the arming gate has NOT closed.%s"
              % ("" if submitted == 0 else
                 " (and yet %d order(s) went out — that is a bug)" % submitted))
    print("  %s" % alerts_arena.SURVIVORSHIP_LINE)
    print("  %s" % alerts_arena.HONESTY_LINE)


def main() -> int:
    ap = argparse.ArgumentParser(                                       # io-boundary
        description="daily paper session: reconcile, check the arming gate, order")
    ap.add_argument("--dry", action="store_true",
                    help="the default: compute and LOG intended orders, submit "
                         "nothing (implied whenever --submit is absent)")
    ap.add_argument("--submit", action="store_true",
                    help="actually submit market-on-open orders to the Alpaca "
                         "PAPER account — still refused unless the arming gate is "
                         "closed")
    ap.add_argument("--send", action="store_true",
                    help="deliver the alerts; without it the exact text is printed")
    args = ap.parse_args()
    submit_wanted = args.submit and not args.dry

    t_start = time.time()                                               # io-boundary
    print("arena paper session%s" % ("" if submit_wanted else "  (DRY — nothing will be submitted)"))

    # The one mention of the live mode in this file, and it is a refusal.
    if config.EXECUTION_MODE == "live":
        print("  ABORT: config.EXECUTION_MODE is 'live'. run_paper.py is the PAPER "
              "stage and its broker hardcodes paper=True, so it cannot act on that "
              "setting and will not pretend to. Live execution is a separate, "
              "human-driven decision with a tool this repo does not contain.")
        return 1

    market, n_wanted = load_market()
    cache_age, bar_age = run_generation.data_staleness(market)          # io-boundary
    session_date = market.dates[-1]
    print("  data      : %d/%d symbols cached, last bar %s (%.1f days old), cache "
          "%.1f days old" % (len(market.symbols), n_wanted, session_date.date(),
                             bar_age, cache_age))
    if max(cache_age, bar_age) > config.MAX_DATA_STALENESS_DAYS:
        print("  ABORT: data is stale; limit is %d days. Orders sized on a market "
              "that no longer exists are worse than no orders."
              % config.MAX_DATA_STALENESS_DAYS)
        title, body = alerts_arena.data_stale_summary(
            cache_age, bar_age, session_date.date(), config.MAX_DATA_STALENESS_DAYS,
            job="paper session")
        alerts_arena.send_transition("paper-data", {"status": "stale-data",
                                                   "date": str(session_date.date())},
                                     title, body, dry_run=not args.send)
        return 1

    broker = broker_paper.get_broker()
    print("  broker    : %s  (%s)"
          % (broker, "credentialed — reads and writes are live against the PAPER account"
             if broker.credentialed else
             "no credentials: reads answer EMPTY and every write refuses"))

    champ = registry.champion()
    history = run_deepeval.read_history()
    arm = arming_status(history, champ[0] if champ else None)

    # ── 1. reconcile the previous session ─────────────────────────────────────
    print("\n  ── reconcile %s" % ("─" * 62))
    prior = read_ledger()
    fills, appended = [], {"written": 0, "skipped": 0}
    if not prior:
        print("  nothing to reconcile: state/%s does not exist yet, so this is the "
              "first paper session on record." % LEDGER_FILE)
    elif not broker.credentialed:
        print("  %d ledger row(s) on record, but the broker is unarmed — there is no "
              "account to read fills from. Reconciliation resumes when credentials "
              "are present." % len(prior))
    else:
        anchor = max(r["date"] for r in prior if r.get("date"))
        fills = broker.fills_since(anchor)                              # io-boundary
        appended = append_ledger([{"date": f["date"], "symbol": f["symbol"],
                                   "side": f["side"], "qty": f["qty"],
                                   "fill_px": f["fill_px"], "order_id": f["order_id"],
                                   "status": f["status"],
                                   "genome_hash": champ[0] if champ else "",
                                   "reason": "fill"} for f in fills])
        print("  fills     : %d closed order(s) since %s -> %d new ledger row(s), %d "
              "already recorded (order_id is the key)"
              % (len(fills), anchor, appended["written"], appended["skipped"]))

    # ── 2. the arming gate ────────────────────────────────────────────────────
    print("\n  ── arming gate %s" % ("─" * 59))
    print("  verdict   : %s" % ("ARMED" if arm["armed"] else "UNARMED"))
    for name, passed, detail in arm["checks"]:
        print("    [%s] %-48s %s" % ("ok" if passed else "--", name, detail))
    print("  rule      : the same champion must survive %d consecutive COMPLETE deep "
          "evals\n              (config.PAPER_ARM_CONSECUTIVE; docs/DESIGN.md "
          "\"Graduation ladder\")" % arm["n_required"])

    # ── 3. today's decision ───────────────────────────────────────────────────
    print("\n  ── decision for the next open %s" % ("─" * 44))
    exit_code = 0
    orders, decision, shadow, submitted = [], None, None, 0
    if champ is None:
        print("  NO CHAMPION. state/champion.json does not exist: no candidate has "
              "ever passed all ten gates, so there is no genome whose targets could "
              "be computed. Nothing is intended, nothing is logged, and that is the "
              "honest state of this system rather than an error.")
    else:
        entry = registry.artifact_entry(champ[0])
        if entry is None:
            print("  ABORT: champion %s has no artifact — it cannot be re-simulated, "
                  "and an order sized from a genome the registry cannot produce is "
                  "an order nothing stands behind." % champ[0])
            return 1
        genome = evolution.entry_genome(entry)
        print("  champion  : %s  %s" % (champ[0], genome.describe()[:70]))
        shadow = shadow_episode(entry, market, prior)
        ep = shadow["episode"]
        print("  shadow    : episode %s -> %s (%d bars, %d fits, %d log rows); "
              "replayed equity differs from the episode's by $%.2e"
              % (shadow["start"].date(), ep["dates"][-1].date(), len(ep["dates"]),
                 len(shadow["fit_audit"]), shadow["log_rows"], shadow["drift"]))
        print("  sandbox   : equity $%.2f, %d position(s), drawdown %.2f%%"
              % (shadow["equity"], int(np.count_nonzero(shadow["shares"])),
                 100 * (shadow["equity"] / shadow["peak"] - 1.0)))

        decision = live_decision(shadow["agent"], market, shadow["shares"],
                                 shadow["equity"], shadow["peak"], shadow["i0"],
                                 shadow["fit_audit"])
        close_t = market.close[-1]
        current = broker.positions()                                    # io-boundary
        basis_equity = shadow["equity"]
        basis = "the sandbox's own equity (no paper account to read)"
        if broker.credentialed:
            acct = broker.account()                                     # io-boundary
            basis_equity, basis = acct["equity"], "the paper account's equity"
        book = np.zeros(len(market.symbols))
        for sym, qty in current.items():
            if sym in market.symbols:
                book[market.symbols.index(sym)] = float(qty)

        env = MarketEnv(market, CostModel())
        tgt, w, lev_scale, rejected = share_targets(env, decision["target_w"],
                                                    close_t, basis_equity)
        plan = order_deltas(env, market, tgt, book, close_t, basis_equity,
                            decision["reason"])
        orders = plan["orders"]
        print("  targets   : %d name(s) at gross %.2fx / net %+.2fx%s, reason %s%s"
              % (int(np.count_nonzero(w)), float(np.abs(w).sum()), float(w.sum()),
                 "" if lev_scale == 1.0 else " (leverage-scaled %.3f)" % lev_scale,
                 decision["reason"],
                 ", refit today" if decision.get("refitted") else ""))
        print("  basis     : $%.2f — %s%s"
              % (basis_equity, basis,
                 "" if broker.credentialed else
                 "; the broker's empty book means UNKNOWN, so these orders are "
                 "stated against a FLAT account"))
        print("  orders    : %d intended (%d dropped: %s)"
              % (len(orders), len(plan["dropped"]),
                 ", ".join("%s %s" % (s, why) for s, why in plan["dropped"][:6]) or "none"))
        for o in orders[:20]:
            print("      %-6s %-6s %5d sh  @ ~$%8.2f  %s -> %s  [%s]"
                  % (o["side"], o["symbol"], o["qty"],
                     float(close_t[market.symbols.index(o["symbol"])]),
                     o["shares_before"], o["shares_after"], o["reason"]))
        if len(orders) > 20:
            print("      ... and %d more" % (len(orders) - 20))

        # ── 4. submit, or say why not ─────────────────────────────────────────
        rows, rejected_orders = [], 0
        for o in orders:
            status, order_id = "intended", ""
            if arm["armed"] and submit_wanted:
                try:
                    res = broker.place(o["symbol"], o["side"], o["qty"])  # io-boundary
                except Exception as exc:                # a broker refusal is a FACT
                    status, order_id = "rejected", ""
                    rejected_orders += 1
                    print("      REJECTED %s %s x%d: %s"
                          % (o["side"], o["symbol"], o["qty"],
                             ("%s: %s" % (type(exc).__name__, exc))[:160]))
                else:
                    if res.get("executed"):
                        status, order_id = "submitted", res.get("order_id", "")
                        submitted += 1
                    else:
                        status = "rejected"
                        rejected_orders += 1
                        print("      REFUSED  %s %s x%d: %s"
                              % (o["side"], o["symbol"], o["qty"], res.get("reason")))
            rows.append({"date": str(session_date.date()), "symbol": o["symbol"],
                         "side": o["side"], "qty": o["qty"], "fill_px": "",
                         "order_id": order_id, "status": status,
                         "genome_hash": champ[0], "reason": o["reason"]})
        wrote = append_ledger(rows)
        print("  ledger    : %d row(s) appended, %d already on record (%s)"
              % (wrote["written"], wrote["skipped"],
                 "status=submitted" if submitted else "status=intended"))
        if not arm["armed"]:
            print("              UNARMED, so every row above is INTENDED and no order "
                  "reached a broker: %s." % arm["reason"])
        elif not submit_wanted:
            print("              armed, but this is a dry run — pass --submit to place "
                  "these orders.")
        if rejected_orders:
            print("  %d order(s) were rejected by the broker. This session's book is "
                  "NOT the book it computed, and the shadow will show it."
                  % rejected_orders)
            exit_code = 1

    # ── 5. the shadow row ─────────────────────────────────────────────────────
    print("\n  ── sim-shadow %s" % ("─" * 60))
    shadow_rows = read_shadow()
    if shadow is None or not broker.credentialed:
        print("  no row: a shadow compares the paper account against the sandbox, and "
              "there is %s." % ("no champion to simulate" if shadow is None
                                else "no paper account to read"))
    else:
        acct = broker.account()                                         # io-boundary
        ep = shadow["episode"]
        if ep["dates"][-1] != session_date:
            raise RuntimeError(
                "the shadow episode ends at %s but the market's last bar is %s — the "
                "sim return and the paper return would be dated differently, which "
                "is the one thing a tracking error may not be."
                % (ep["dates"][-1].date(), session_date.date()))
        sim_net = float(ep["daily_net"][-1])
        paper_net = acct["equity"] / acct["last_equity"] - 1.0 if acct["last_equity"] else float("nan")
        te_bps = (paper_net - sim_net) * 1e4
        slip = fill_slippage_bps(fills, market)
        row = {"date": str(session_date.date()), "genome_hash": champ[0],
               "sim_net": repr(sim_net), "paper_net": repr(paper_net),
               "tracking_error_bps": repr(te_bps), "fill_slippage_bps": repr(slip),
               "n_fills": len(fills), "paper_equity": repr(acct["equity"]),
               "paper_last_equity": repr(acct["last_equity"]),
               "sim_equity": repr(shadow["equity"])}
        fresh = append_shadow(row)
        shadow_rows = read_shadow()
        print("  %s  sim %+.3f%%  paper %+.3f%%  tracking %+.1f bps  median fill "
              "slippage %s bps  (%s)"
              % (row["date"], 100 * sim_net, 100 * paper_net, te_bps,
                 "n/a" if not np.isfinite(slip) else "%+.1f" % slip,
                 "appended" if fresh else "already recorded for this bar"))

        if abs(te_bps) > config.PAPER_MAX_TE_BPS:
            spell = breach_spell_start(shadow_rows, config.PAPER_MAX_TE_BPS)
            title, body = alerts_arena.paper_summary(
                "tracking-breach", champion=champ[0], date=row["date"],
                sim_net=sim_net, paper_net=paper_net, tracking_error_bps=te_bps,
                slippage_bps=slip,
                detail="Limit is %.0f bps/day (config.PAPER_MAX_TE_BPS). This spell "
                       "began %s; DESIGN's kill switch is 20 consecutive days."
                       % (config.PAPER_MAX_TE_BPS, spell))
            alerts_arena.send_transition(
                "paper-tracking", {"champion": champ[0], "status": "tracking-breach",
                                   "spell": spell or ""}, title, body,
                dry_run=not args.send)

    # ── 6. the go-live evidence table (read by a human, acted on by nobody) ───
    if shadow_rows:
        table = evidence_table(shadow_rows)
        print("\n  ── go-live evidence (docs/DESIGN.md ladder) %s" % ("─" * 29))
        for name, value, threshold, passed in table["rows"]:
            shown = value if isinstance(value, int) else (
                "n/a" if not np.isfinite(value) else "%.3f" % value)
            print("    [%s] %-34s %10s   needs %s"
                  % ("ok" if passed else "--", name, shown, threshold))
        if table["all_pass"]:
            title, body = alerts_arena.paper_summary(
                "go-live-candidate", champion=champ[0] if champ else None,
                n_days=table["n_days"], slippage_bps=table["median_slippage_bps"],
                detail="corr %.2f, %d tracking trips in the last 60 sessions."
                       % (table["corr"], table["trips"]))
            alerts_arena.send_transition(
                "paper-golive", {"champion": champ[0] if champ else "",
                                 "status": "go-live-candidate"},
                title, body, dry_run=not args.send)

    # ── 7. the arming transition ──────────────────────────────────────────────
    if arm["armed"]:
        # The gate is about the CHAMPION, so it can close while the machine still
        # has no keys. Saying so is the difference between "trading has started"
        # and "trading is now permitted", and only one of those is true here.
        title, body = alerts_arena.paper_summary(
            "armed", champion=champ[0], arm_consecutive=arm["n_required"],
            n_days=config.PAPER_MIN_DAYS,
            detail="" if broker.credentialed else
                   "NOTE: no Alpaca paper credentials are configured on this "
                   "machine, so nothing can be submitted yet. The gate is open; "
                   "the account is not.")
        alerts_arena.send_transition(
            "paper-arming", {"champion": champ[0], "status": "armed",
                             "broker": "credentialed" if broker.credentialed else "none"},
            title, body, dry_run=not args.send)

    print("\n  wall clock: %.1f s" % (time.time() - t_start))           # io-boundary
    print_honesty(arm["armed"], submitted)
    return exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except run_generation.IdentityDrift as exc:
        # The same guard the other two entry points take. Nothing in this file
        # writes a trial row, so it can only arrive from a shared helper — but a
        # provenance failure must never exit 0 anywhere in this project.
        print("\nABORT (IdentityDrift): %s" % exc)
        raise SystemExit(1)
