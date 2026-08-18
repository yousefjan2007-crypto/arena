"""
One generation of the arena, one process, one exit code. The nightly entry point.

    python3 run_generation.py --init --pop 64      seed a population and run gen 0
    python3 run_generation.py                      run the next generation
    python3 run_generation.py --no-evolve          evaluate and stop before breeding

THE FULL PATH: load the population -> F0-screen all of it -> fully evaluate (F1)
the top SCREEN_FRAC plus every elite -> write the Tier-A returns matrix -> breed
the next population -> update the hall of fame -> print an honest summary -> exit.
Phase 5 adds the gates, Phase 6 the alert and the cron.

ONE-SHOT BY DESIGN. Fetch, evaluate, append, print, exit — no loop, no daemon.
The scheduler (launchd, then GitHub Actions) re-runs it; anti-spam and resume come
from what is already on disk (the ledger is idempotent by genome hash, generation
and fidelity), never from in-process state. That is the pattern every scheduled
job in this workspace uses, and it is what makes a killed run harmless.

A GENERATION MAY SPAN SESSIONS, ON PURPOSE. One hgb genome costs ~21 minutes at
F1 where a momentum rule costs 4 seconds, so an hgb-heavy population can outlast
any single session window. The answer is NOT to cap, trim or reweight the slate
for runtime — that would quietly bias the search toward cheap families and make
every reported number a claim about what fit in the budget rather than about what
works. Instead:

  • each F1 episode is checkpointed to state/tmp_gen_<N>/<hash>.npz BY THE WORKER
    THAT COMPUTED IT, the moment it finishes;
  • config.GEN_TIME_BUDGET_MIN is checked between genome evaluations, and when it
    is exhausted the run stops, keeps everything finished, and exits 0 saying
    "resumable";
  • the next run reloads those files instead of re-simulating, and F0 results are
    reloaded from the trial ledger the same way;
  • EVOLUTION ONLY HAPPENS WHEN THE WHOLE GENERATION IS EVALUATED. A population
    bred from a half-scored generation would select on which genomes happened to
    be cheap, which is the same bias by a slower route.

The generation counter lives in population.json, so a re-run after a completed
generation simply proceeds with the next one.

WALL-CLOCK LIVES HERE AND NOWHERE ELSE. This file is an I/O boundary: it reads
cache mtimes to decide whether the data is too stale to evaluate, it enforces the
time budget above, and it times itself so the human reading the output knows what
a generation costs. Nothing it measures feeds a simulated quantity — verify.py's
scanner skips run_*.py for exactly this reason, so every clock call below is
commented as the boundary it is.

PARALLELISM. joblib over genomes with config.N_JOBS workers. OMP_NUM_THREADS is
pinned to 1 at the top of this file, BEFORE the import chain reaches sklearn:
the workers inherit the environment, so each genome gets one core instead of
fourteen threads fighting over the same fourteen cores. The panel is passed as an
argument and joblib memory-maps the arrays inside it, so the 120-symbol feature
panel is written to disk once and shared, not copied per worker.
"""
from __future__ import annotations

import os

# BEFORE anything imports sklearn (config -> ... -> strategy does): one BLAS/OpenMP
# thread per worker. setdefault, so an operator who exports it wins.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse                                                       # noqa: E402
import contextlib                                                     # noqa: E402
import csv                                                            # noqa: E402
import glob                                                           # noqa: E402
import json                                                           # noqa: E402
import math                                                           # noqa: E402
import shutil                                                         # noqa: E402
import time                                                           # noqa: E402

import numpy as np                                                    # noqa: E402
import pandas as pd                                                   # noqa: E402
from joblib import Parallel, delayed                                  # noqa: E402

import config                       # FIRST: puts the siblings on sys.path  # noqa: E402
import alerts_arena                                                   # noqa: E402
import datafeed                                                       # noqa: E402
import evaluate                                                       # noqa: E402
import evolution                                                      # noqa: E402
import features as arena_features                                     # noqa: E402
import genome as gn                                                   # noqa: E402
import ledger                                                         # noqa: E402
import registry                                                       # noqa: E402
from env import CostModel                                             # noqa: E402

POPULATION_FILE = "population.json"


class IdentityDrift(RuntimeError):
    """The inputs moved under a generation that was already partly on disk.

    Both of this run's persistence layers are deliberately write-once — the ledger
    is idempotent on (genome hash, generation, fidelity) and the returns matrix
    refuses to overwrite a generation — and both of those keys are BLIND TO
    IDENTITY. That is exactly right for a resumed or re-raced run, where the work
    being re-offered is the same work. It is exactly wrong when the data cache
    refreshed between two runs of the same generation: everything gets re-simulated
    under the new bars, breeding selects on the new scores, and then the new ledger
    rows are dropped as duplicates while the old returns matrix is handed back as
    though it were this run's. The population would then have been bred from
    numbers that appear nowhere on disk, and nothing about the files would look
    wrong.

    So the run stops instead. The fix is a human decision, not a default: either
    let the generation stand as it was evaluated (restore the data the row names)
    or start the generation over under a new number. A dead run is cheap; a
    provenance that lies is not, and this project's whole claim is that the ledger
    says what actually happened.
    """


# ── data freshness (I/O boundary: the only place a wall clock is consulted) ────
def data_staleness(market) -> tuple:
    """(cache age in days, last-bar age in days) — both against the wall clock.

    Two different questions, both worth asking: the cache file's mtime says when
    the data was last refreshed, and the last bar's date says how old the newest
    price is. A cache refreshed this morning that still ends last Tuesday is stale
    in the way that matters, and a stale mtime with fresh bars is only a weekend.
    """
    now = time.time()                                          # io-boundary
    mtime = os.path.getmtime(datafeed._cache_path(config.BENCHMARK))   # noqa: SLF001
    cache_age = (now - mtime) / 86400.0
    bar_age = (now - market.dates[-1].timestamp()) / 86400.0
    return cache_age, bar_age


# ── cache refresh (the one network step, and the only place that takes it) ─────
@contextlib.contextmanager
def _force_fetch():
    """Make sell_in_may's fetcher treat every cached file as stale for the moment.

    The mirror image of features._cache_only(), and it exists for a cloud-specific
    reason: `data.py` decides freshness from the CSV's MTIME, and a git checkout
    stamps every file with the time of the checkout. On a runner the cache is
    therefore always "fresh" and would never be refreshed at all, however old the
    bars inside it are.
    """
    saved = config.CACHE_MAX_AGE_DAYS
    config.CACHE_MAX_AGE_DAYS = -1.0          # nothing is fresher than -1 days old
    try:
        yield
    finally:
        config.CACHE_MAX_AGE_DAYS = saved


def _rel_diff(a: float, b: float) -> float:
    """|a-b| relative to the larger magnitude. 0 vs 0 is 0, not a division.

    A cell that is absent on one side and present on the other is a REPAIR, not a
    rounding difference, and returns inf so the tolerance gate takes it as a real
    restatement. Without that, NaN would propagate through every comparison as
    "not greater than the tolerance" and the repair would be silently discarded.
    """
    a_ok, b_ok = a == a, b == b                 # NaN is the only value != itself
    if not (a_ok and b_ok):
        return 0.0 if not (a_ok or b_ok) else float("inf")
    scale = max(abs(a), abs(b))
    if scale == 0.0:
        return 0.0
    return abs(a - b) / scale


def _parse_cache_lines(payload: bytes) -> tuple:
    """(header, {date: (raw line, [floats])}) for a cache CSV, or (None, {})."""
    try:
        text = payload.decode()
    except Exception:
        return None, {}
    lines = text.splitlines()
    if not lines:
        return None, {}
    rows = {}
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split(",")
        try:
            rows[parts[0]] = (line, [float(v) if v != "" else float("nan")
                                     for v in parts[1:]])
        except ValueError:
            return None, {}                 # unparseable: treat the file as opaque
    return lines[0], rows


def _tolerance_gate(path: str, cached: bytes, tol: float) -> dict:
    """Keep the CACHED bytes for every row the refetch did not really change.

    THE PROBLEM THIS SOLVES. yfinance recomputes its entire auto-adjusted history
    in reduced precision on every fetch, so a refetch minutes later — with no new
    bar, no dividend, nothing — restates every row by ~3e-7 relative. Measured on
    the first two cloud runs, 75 minutes apart: 13,114 of AAPL's 15,918 lines
    changed, and the whole 78 MB cache was rewritten twice a day. At the scheduled
    cadence that is repository growth measured in gigabytes per year, for a
    difference no computation in this project can resolve.

    THE RULE. Every date present in both the cached and the fetched frame is
    compared column by column. If EVERY shared row agrees within `tol` relative,
    the cached bytes are kept verbatim (full precision, not rounded — rounding
    would only move the jitter to a different boundary) and only genuinely new
    dates are appended, taken verbatim from the fetch. If ANY row exceeds `tol`, a
    real adjustment event has happened — a split, a dividend, a vendor correction —
    and the fetched file is accepted whole and the event is logged loudly.

    Rows the fetch DROPPED are kept from the cache and counted: a short fetch is a
    bad data day, and this project's stated posture is that a bad data day degrades
    to stale-cache rather than to loss.

    WHAT THIS IS NOT. It is not an attempt at permanent cross-machine byte parity.
    Two machines that refresh at different times will each cross an adjustment
    event on their own schedule and restate independently, so their caches diverge
    for real reasons and stay diverged. The doctrine is unchanged: results are
    per-platform, the cloud is canonical, and `data_hash` on every row records
    exactly which bytes a result was produced from.

    Returns a decision dict; writes `path` only when it has something better to
    put there.
    """
    with open(path, "rb") as f:
        fetched = f.read()
    if not cached:
        return {"action": "new-file", "restated": 0, "added": 0, "max_rel": 0.0}

    old_head, old_rows = _parse_cache_lines(cached)
    new_head, new_rows = _parse_cache_lines(fetched)
    if old_head is None or new_head is None or old_head != new_head or not new_rows:
        return {"action": "accept-unparseable", "restated": 0, "added": 0, "max_rel": 0.0}

    shared = [d for d in new_rows if d in old_rows]
    worst, n_over = 0.0, 0
    for d in shared:
        old_v, new_v = old_rows[d][1], new_rows[d][1]
        if len(old_v) != len(new_v):
            return {"action": "accept-shape", "restated": len(shared), "added": 0,
                    "max_rel": float("inf")}
        row_worst = max((_rel_diff(a, b) for a, b in zip(old_v, new_v)), default=0.0)
        worst = max(worst, row_worst)
        if row_worst >= tol:
            n_over += 1

    if n_over:
        return {"action": "restated", "restated": n_over, "added": 0, "max_rel": worst}

    added = [d for d in new_rows if d not in old_rows]
    dropped = [d for d in old_rows if d not in new_rows]
    if not added:
        with open(path, "wb") as f:                 # nothing new: cache unchanged
            f.write(cached)
        return {"action": "unchanged", "restated": 0, "added": 0,
                "dropped": len(dropped), "max_rel": worst}

    # Cached bytes verbatim, then the new dates verbatim from the fetch. Sorted by
    # date because datafeed._clean_frame refuses an unsorted cache.
    body = [old_rows[d][0] for d in sorted(old_rows)] + \
           [new_rows[d][0] for d in sorted(added)]
    with open(path, "wb") as f:
        f.write(("\n".join([old_head] + sorted(body, key=lambda ln: ln.split(",")[0]))
                 + "\n").encode())
    return {"action": "appended", "restated": 0, "added": len(added),
            "dropped": len(dropped), "max_rel": worst}


def new_refresh_stat(n_wanted: int = 0) -> dict:
    """The empty tally refresh_cache accumulates into.

    Two maxima, never one: the largest drift among the files that were KEPT is the
    measurement that says the tolerance is set sensibly, and folding a restated
    symbol's much larger drift into it would report a number ABOVE the tolerance
    under a label claiming everything stayed below it.
    """
    return {"wanted": int(n_wanted), "ok": 0, "empty": 0, "error": 0,
            "unchanged": 0, "appended": 0, "restated": 0, "new": 0,
            "bars_added": 0, "rows_restated": 0, "max_rel_kept": 0.0,
            "max_rel_restated": 0.0, "restated_symbols": [],
            "shape_changed": 0, "shape_symbols": []}


def note_decision(stat: dict, symbol: str, d: dict) -> dict:
    """Fold one _tolerance_gate decision into the tally. Pure; verify.py calls it.

    `inf` is the row-SHAPE anomaly (a column appeared or vanished), not a measured
    drift, and it never reaches either maximum. Coercing it to 0.0 there — which
    this function used to do inline — reported the loudest event the gate can see
    as the quietest one.
    """
    shape_change = d["max_rel"] == float("inf")
    worst = 0.0 if shape_change else d["max_rel"]
    if shape_change:
        stat["shape_changed"] += 1
        stat["shape_symbols"].append(symbol)
    if d["action"] == "unchanged":
        stat["unchanged"] += 1
        stat["max_rel_kept"] = max(stat["max_rel_kept"], worst)
    elif d["action"] == "appended":
        stat["appended"] += 1
        stat["bars_added"] += d["added"]
        stat["max_rel_kept"] = max(stat["max_rel_kept"], worst)
    elif d["action"] == "new-file":
        stat["new"] += 1
    else:
        stat["restated"] += 1
        stat["rows_restated"] += d["restated"]
        stat["restated_symbols"].append(symbol)
        stat["max_rel_restated"] = max(stat["max_rel_restated"], worst)
    return stat


def refresh_cache(symbols, verbose=True) -> dict:
    """Re-download the price cache for `symbols` + the macro tickers. Network!

    THIS IS THE ONLY NETWORK CALL IN THE WHOLE RUN, and it happens strictly before
    the market is loaded — never inside an evaluation path, where two runs of the
    same generation could otherwise see different data (datafeed's module
    docstring, and the reason load_market defaults to offline).

    The macro tickers are not optional. features.py builds the panel through
    signal_lab's macro frame, which reads ^TNX ^IRX ^VIX DX-Y.NYB HYG LQD from this
    same cache; leaving them behind would extend the equity bars past the macro
    bars and quietly fill the tail of every macro column with NaN.

    A failed download is not fatal — the point of a cache-first design is that a
    bad yfinance day degrades to stale data, which the staleness guard then judges
    on its own terms.

    EVERY WRITE GOES THROUGH THE TOLERANCE GATE (see _tolerance_gate): the fetcher
    overwrites the CSV, and this function then decides whether that overwrite said
    anything. Below config.REFRESH_REL_TOL it did not, and the cached bytes are
    put back so the repository does not carry a 78 MB rewrite twice a day for a
    difference no computation can resolve.
    """
    macro = config.import_sibling("macro", config.SIGNAL_LAB)
    wanted = list(dict.fromkeys(list(symbols) + list(macro.MACRO_TICKERS.values())))
    tol = float(config.REFRESH_REL_TOL)
    stat = new_refresh_stat(len(wanted))
    with _force_fetch():
        for sym in wanted:
            path = datafeed._cache_path(sym)                        # noqa: SLF001
            cached = b""
            if os.path.exists(path):
                with open(path, "rb") as f:
                    cached = f.read()
            try:
                df = datafeed._read_symbol(sym, refresh=True)       # noqa: SLF001
            except Exception:
                stat["error"] += 1
                continue
            if not len(df):
                stat["empty"] += 1
                if cached and not os.path.exists(path):
                    with open(path, "wb") as f:                     # never lose history
                        f.write(cached)
                continue
            stat["ok"] += 1
            if not os.path.exists(path):
                continue                        # fetch succeeded but wrote nothing
            note_decision(stat, sym, _tolerance_gate(path, cached, tol))
    if verbose:
        print("  refresh   : %d symbols fetched (%d empty, %d errored) into %s"
              % (stat["ok"], stat["empty"], stat["error"],
                 os.path.relpath(config.CACHE_DIR, config.ROOT)
                 if config.CACHE_DIR.startswith(config.ROOT) else config.CACHE_DIR))
        print("  tolerance : %d kept unchanged, %d appended-to (%d new bars), "
              "%d seeded, %d RESTATED — largest drift among the files KEPT %.2e "
              "(tol %.0e)"
              % (stat["unchanged"], stat["appended"], stat["bars_added"], stat["new"],
                 stat["restated"], stat["max_rel_kept"], tol))
        if stat["shape_symbols"]:
            print("  SHAPE     : %d symbol(s) came back with a different row shape "
                  "(a column appeared or vanished), so the drift is not a number — "
                  "reported as shape-changed, not as 0.0 — and the fetch was "
                  "accepted whole: %s"
                  % (stat["shape_changed"], ", ".join(sorted(stat["shape_symbols"])[:12])))
        if stat["restated_symbols"]:
            # Loud on purpose: a restatement moves the prices this arena scores
            # genomes on. It should never be a silent line.
            print("  RESTATED  : %d symbol(s) exceeded the tolerance (largest drift "
                  "%.2e over %d row(s)) and were rewritten whole: %s — a split, "
                  "dividend or vendor correction landed. An equity moves data_hash; "
                  "a macro ticker moves panel_hash only (data_hash never sees the "
                  "six macro series). Prior evaluations are a different vintage "
                  "from here on."
                  % (stat["restated"], stat["max_rel_restated"], stat["rows_restated"],
                     ", ".join(sorted(stat["restated_symbols"])[:12])
                     + (" ..." if len(stat["restated_symbols"]) > 12 else "")))
    return stat


def generation_in_flight(generation, state_dir=None) -> bool:
    """True when this generation already has evaluations on disk.

    A refresh is only safe BETWEEN generations. Refreshing under a generation that
    is half-evaluated moves data_hash and panel_hash, which makes every checkpoint
    unusable, every already-ledgered row a different vintage, and the next
    _append_trial an IdentityDrift abort (see that class's docstring). On a laptop
    that is a loud stop a human resolves; on a twice-daily cloud schedule it is a
    generation that can never finish, because every session would refresh again.

    So the runner refreshes when it is starting fresh work and resumes on the
    committed cache when it is finishing old work. Nothing is lost: the next
    generation gets the new bars.
    """
    if generation is None:
        return False
    if glob.glob(os.path.join(tmp_dir(generation, state_dir), "*.npz")):
        return True
    path = ledger.ledger_path(state_dir)
    if not os.path.exists(path):
        return False
    with open(path, newline="") as f:
        return any(int(row["generation"]) == int(generation) for row in csv.DictReader(f))


# ── population ─────────────────────────────────────────────────────────────────
def pending_generation(state_dir=None):
    """The generation population.json points at, or None if there is no population.

    Read before the market is loaded, because both the refresh guard above and the
    alert's anti-spam key need to know which generation this run is about.
    """
    path = os.path.join(state_dir or config.STATE_DIR, POPULATION_FILE)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return int(json.load(f)["generation"])
    except Exception:
        return None


def alert(kind: str, generation, status: str, title: str, body: str, send: bool) -> bool:
    """One alert, suppressed unless (champion, generation, status) has moved."""
    champ = registry.champion()
    state = {"champion": champ[0] if champ else "",
             "generation": None if generation is None else int(generation),
             "status": status}
    return alerts_arena.send_transition(kind, state, title, body, dry_run=not send)


def seed_population(n: int, feature_names, generation: int = 0) -> list:
    """`n` uniformly drawn genomes as population entries, each from its own
    reproducible stream.

    child_rng(SEED, generation, "genesis", i) rather than one shared rng: drawing
    from a single stream would make genome i depend on how many genomes were drawn
    before it, so `--pop 16` and `--pop 64` would share nothing. This way the
    first 16 of a 64-genome seed are the same 16.
    """
    return [evolution.make_entry(
        gn.random_genome(gn.child_rng(config.SEED, generation, "genesis", i), feature_names),
        "seed", "", generation) for i in range(n)]


def save_population(entries, generation: int, state_dir=None) -> str:
    """state/population.json — a POINTER to the live generation, like champion.json.

    This is one of the two files a generation rewrites (the hall of fame is the
    other). It is not the record: every genome in it was appended to the trial
    ledger when it was evaluated, and the ledger is what nothing may rewrite. The
    file says so in its own `note` field, because everything else under state/ is
    append-only and a reader is entitled to know which files are which.
    """
    path = os.path.join(state_dir or config.STATE_DIR, POPULATION_FILE)
    payload = {"note": evolution.POPULATION_NOTE,
               "generation": int(generation),
               "size": len(entries),
               "op_counts": evolution.op_counts(entries),
               "entries": list(entries)}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1, sort_keys=True)
    os.replace(tmp, path)
    return path


def load_population(state_dir=None) -> tuple:
    """(entries, generation). A Phase-3 file with bare genomes and no lineage still
    loads — its entries become "seed" entries of their own generation, which is
    exactly what they were."""
    path = os.path.join(state_dir or config.STATE_DIR, POPULATION_FILE)
    if not os.path.exists(path):
        raise SystemExit("no %s — run once with --init to seed a population" % path)
    with open(path) as f:
        payload = json.load(f)
    generation = int(payload["generation"])
    if "entries" in payload:
        return list(payload["entries"]), generation
    return ([evolution.make_entry(gn.from_dict(d), "seed", "", generation)
             for d in payload["genomes"]], generation)


# ── F1 checkpoints (state/tmp_gen_<N>/<hash>.npz) ─────────────────────────────
# What a finished F1 episode has to carry into the returns matrix, the ledger row
# and the summary. Everything is pre-vault: `vault_daily_net` is deliberately NOT
# checkpointed — the same rule ledger.write_returns_matrix enforces, so no vault
# day is ever written under state/ by an evaluation path.
_F1_ARRAYS = ("daily_net", "daily_gross", "turnover", "costs")
_F1_SCALARS = ("score", "sharpe_prevault", "n_days_prevault", "n_features",
               "first_active", "n_fits", "regime_finite_frac")
_F1_INTS = ("n_days_prevault", "n_features", "first_active", "n_fits")


def tmp_dir(generation: int, state_dir=None) -> str:
    return os.path.join(state_dir or config.STATE_DIR, "tmp_gen_%04d" % int(generation))


def _tmp_path(generation: int, ghash: str, state_dir=None) -> str:
    return os.path.join(tmp_dir(generation, state_dir), "%s.npz" % ghash)


def save_f1_checkpoint(generation: int, entry: dict, res: dict, identity, state_dir=None) -> str:
    """Persist one finished F1 episode, written by the worker that computed it.

    In the worker rather than the parent so that a run stopped by the time budget
    keeps every episode that FINISHED, not just the ones the parent had gotten
    around to collecting — with eight workers and 20-minute hgb episodes that is
    up to seven episodes saved per stop.

    `identity` is (data_hash, panel_hash, config_hash): a checkpoint is only
    reusable if the market, the panel and the settings are the ones it was
    computed under, so it is stored alongside and checked on the way back in.

    THE WHOLE POPULATION ENTRY IS STORED, not just the hash. A checkpoint is
    evidence that an evaluation HAPPENED, and an evaluation that happened has to
    be able to reach the trial ledger even if the run that computed it never got
    that far and its population.json was overwritten in the meantime —
    sweep_orphan_checkpoints is what does that, and it needs the genome body, the
    lineage and the identity the episode was computed under.
    """
    ghash = entry["hash"]
    path = _tmp_path(generation, ghash, state_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {k: np.asarray(res[k], dtype=np.float64) for k in _F1_ARRAYS}
    payload["dates"] = (pd.DatetimeIndex(res["dates"]).values
                        .astype("datetime64[ns]").astype(np.int64))
    for k in _F1_SCALARS:
        payload[k] = np.float64(res[k])
    payload["identity"] = np.array(list(identity) + [ghash], dtype="U32")
    payload["entry"] = np.array(json.dumps(entry, sort_keys=True))
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, path)                    # never leave a half-written checkpoint
    return path


def load_f1_checkpoint(generation: int, ghash: str, identity, state_dir=None):
    """The episode back, or None if there isn't a usable one.

    None — not an exception — for every failure mode, because all of them mean the
    same thing to the caller: simulate it again. A killed run can leave a
    truncated file, and the identity check catches a checkpoint written before the
    data, the panel or a config knob moved.

    PHASE 5, READ THIS BEFORE TRUSTING A RESUMED RESULT. What comes back is
    _F1_ARRAYS + _F1_SCALARS and nothing else: there is no `fit_audit` (so the
    streaming-purge evidence for that episode is not here — verify test 5's
    invariant cannot be re-checked from a checkpoint) and no `vault_daily_net`
    (deliberately never written under state/ by an evaluation path). A gate that
    needs either — G3's vault confirmation, any fit-time audit — must RE-SIMULATE
    the genome rather than read this dict, which is what gate G1's "incumbent
    re-simulated fresh" rule already requires for its own reasons.
    """
    path = _tmp_path(generation, ghash, state_dir)
    if not os.path.exists(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            if [str(v) for v in z["identity"]] != [str(v) for v in identity] + [ghash]:
                return None
            res = {k: z[k].astype(np.float64) for k in _F1_ARRAYS}
            res["dates"] = pd.DatetimeIndex(z["dates"].astype("datetime64[ns]"))
            for k in _F1_SCALARS:
                res[k] = float(z[k])
    except Exception:
        return None
    for k in _F1_INTS:
        res[k] = int(res[k])
    # npz has no None: strategy.py returns None for "this genome has no regime
    # filter", which np.float64 stores as NaN. The fraction it otherwise holds is a
    # mean of booleans and can never be NaN, so the mapping back is unambiguous —
    # and a resumed result must mean exactly what a fresh one means.
    if not np.isfinite(res["regime_finite_frac"]):
        res["regime_finite_frac"] = None
    res["resumed"] = True
    return res


def sweep_orphan_checkpoints(generation: int, state_dir=None, verbose=True) -> dict:
    """Ledger every checkpointed F1 episode that never made it onto the ledger.

    AN EVALUATION THAT HAPPENED HAS TO COUNT. n_trials() is the N that deflates
    every Sharpe this project reports, and it is the count of genomes the search
    LOOKED at — so an episode that was simulated, checkpointed, and then orphaned
    (the run died before appending; the next run was `--init`; the population file
    moved on) is a trial the ledger is missing. Dropping it does not just lose a
    record, it makes DSR optimistic, which is the one direction this project is
    not allowed to be wrong in.

    Orphans are ledgered under THEIR OWN recorded identity — the data, panel and
    config hashes stored in the checkpoint, not this run's — because that is what
    the episode was actually computed under, and a row claiming otherwise would be
    exactly the provenance lie IdentityDrift exists to prevent.

    A directory for a generation the arena has already passed can never be
    resumed (the counter only moves forward), so it is removed once swept — UNLESS
    something in it could not be read. An unreadable checkpoint is left alone
    rather than deleted (see the handler below); deleting the directory around it
    would do exactly what the handler refuses to do, and the episodes it holds
    would be gone from the only place they were ever recorded, undercounting
    n_trials() and making every DSR in the project optimistic. Such a directory is
    kept, reported loudly, and left for a human.
    """
    root = state_dir or config.STATE_DIR
    stat = {"appended": 0, "already": 0, "unreadable": 0, "removed": 0, "kept": 0}
    for d in sorted(glob.glob(os.path.join(root, "tmp_gen_*"))):
        try:
            gen = int(os.path.basename(d).rsplit("_", 1)[-1])
        except ValueError:
            continue
        unreadable_here = 0
        for path in sorted(glob.glob(os.path.join(d, "*.npz"))):
            try:
                with np.load(path, allow_pickle=False) as z:
                    entry = json.loads(str(z["entry"]))
                    ident = [str(v) for v in z["identity"]]
                    score = float(z["score"])
                    sharpe = float(z["sharpe_prevault"])
                    n_days = int(float(z["n_days_prevault"]))
            except Exception:
                # Truncated, or written before the entry was stored. Either way it
                # cannot be turned into an honest row; it is still usable as a
                # resume checkpoint, so it is left alone rather than deleted.
                stat["unreadable"] += 1
                unreadable_here += 1
                continue
            if _ledger_row(gen, entry["hash"], "F1", state_dir) is not None:
                stat["already"] += 1
                continue
            ledger.record_trial(gen, evolution.entry_genome(entry), "F1", score, sharpe,
                                n_days, ident[0], ident[1],
                                parent_hash=evolution.parent_field(entry["parent_hash"]),
                                birth_gen=entry["birth_gen"], state_dir=state_dir,
                                config_hash=ident[2])
            stat["appended"] += 1
            if verbose:
                print("  orphan    : ledgered %s (generation %d, F1, SR %+.2f) from a "
                      "checkpoint no run ever recorded" % (entry["hash"], gen, sharpe))
        if gen < int(generation) and unreadable_here:
            # The handler above leaves an unreadable checkpoint ALONE; deleting the
            # directory it sits in would undo that decision by another route, and
            # the episodes it holds have no row anywhere else.
            stat["kept"] += 1
            if verbose:
                print("  orphan    : KEPT %s — %d checkpoint(s) there could not be "
                      "read, so they were never ledgered. Removing the directory "
                      "would destroy the only record that those episodes ran and "
                      "leave n_trials() undercounting, which makes every DSR "
                      "optimistic. Inspect or delete it by hand."
                      % (os.path.basename(d), unreadable_here))
        elif gen < int(generation):
            shutil.rmtree(d, ignore_errors=True)
            stat["removed"] += 1
    if verbose and stat["removed"]:
        print("  orphan    : removed %d checkpoint dir(s) for generations already "
              "passed" % stat["removed"])
    return stat


def _ledger_row(generation: int, ghash: str, fidelity: str, state_dir=None):
    """The ledger row for one (genome, generation, fidelity), or None."""
    path = ledger.ledger_path(state_dir)
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if (row["genome_hash"] == ghash and row["fidelity"] == fidelity
                    and int(row["generation"]) == int(generation)):
                return row
    return None


def _row_identity(row) -> tuple:
    return (row["data_hash"], row["panel_hash"], row["config_hash"])


def _append_trial(generation: int, entry, res, fidelity: str, sharpe, n_days,
                  identity, state_dir=None) -> bool:
    """Append one evaluation, and refuse to let a duplicate key swallow new work.

    record_trial returning False means "already on record", which is the whole
    point of a resumable, re-raceable job — but its key does not include the
    identity columns. A False for a result THIS RUN COMPUTED, against a row
    written under different inputs, is the one case where idempotency is silently
    destructive: the numbers in memory are what breeding is about to select on,
    and the row on disk describes a different market. See IdentityDrift.
    """
    wrote = ledger.record_trial(generation, evolution.entry_genome(entry), fidelity,
                                res["score"], sharpe, n_days, identity[0], identity[1],
                                parent_hash=evolution.parent_field(entry["parent_hash"]),
                                birth_gen=entry["birth_gen"], state_dir=state_dir)
    if wrote or res.get("resumed"):
        return wrote
    prior = _ledger_row(generation, entry["hash"], fidelity, state_dir)
    if prior is not None and _row_identity(prior) == tuple(identity):
        # Same genome, same generation, same inputs: this run re-simulated work
        # that was already recorded (checkpoints removed by hand, say). The row
        # on disk says what this run would have written, so keeping it is honest.
        return False
    raise IdentityDrift(
        "generation %d already has a %s row for %s, recorded under different "
        "inputs than this run computed it with:\n"
        "    on disk : data %s | panel %s | config %s\n"
        "    this run: data %s | panel %s | config %s\n"
        "The ledger is idempotent on (genome, generation, fidelity) and would "
        "have dropped the row this run just produced, while the population was "
        "bred from it. Either restore the inputs the row names, or evaluate this "
        "population as a new generation."
        % ((generation, fidelity, entry["hash"])
           + (_row_identity(prior) if prior is not None
              else ("?", "?", "? (no row found — the ledger changed mid-run)"))
           + tuple(identity)))


def _ledger_f0(generation: int, identity, state_dir=None) -> dict:
    """F0 scores already on record for this generation, hash -> result.

    The F0 checkpoint is the trial ledger itself — the screen's whole output is one
    row, so writing a second copy of it would be inventing a record that can
    disagree with the one that counts. Only rows carrying this run's data, panel
    AND config hash are reusable; anything else is re-screened.

    Read with csv rather than ledger.read_ledger() for one specific reason: scores
    are stored as repr(float) and Python's float() round-trips that exactly, while
    pandas' fast float converter can land a unit in the last place away. A 1-ULP
    difference is invisible everywhere except in a tie, where it would flip a rank
    between a fresh run and a resumed one.
    """
    path = ledger.ledger_path(state_dir)
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            same = (row["data_hash"], row["panel_hash"], row["config_hash"]) == tuple(identity)
            if row["fidelity"] != "F0" or int(row["generation"]) != int(generation) or not same:
                continue
            out[row["genome_hash"]] = {"score": float(row["score"]),
                                       "era_sharpes": None,
                                       "n_features": int(row["n_features"]),
                                       "n_days": int(row["n_days"]),
                                       "resumed": True}
    return out


# ── workers (module level: joblib has to be able to pickle them) ──────────────
def _screen_one(entry, era_markets, cost):
    t0 = time.time()                                           # io-boundary
    res = evaluate.screen(evolution.entry_genome(entry), None, cost, era_markets=era_markets)
    return entry["hash"], res, time.time() - t0                # io-boundary


def _full_one(entry, market, cost, generation, identity, state_dir):
    t0 = time.time()                                           # io-boundary
    res = evaluate.full_eval(evolution.entry_genome(entry), market, cost)
    res["n_fits"] = len(res["fit_audit"])
    save_f1_checkpoint(generation, entry, res, identity, state_dir)
    return entry["hash"], res, time.time() - t0                # io-boundary


def _describe(res: dict) -> str:
    if res.get("era_sharpes"):
        return "eras " + "/".join("%+.2f" % s for s in res["era_sharpes"])
    if "sharpe_prevault" in res:
        return "SR %+.2f over %d days" % (res["sharpe_prevault"], res["n_days_prevault"])
    return "resumed from the ledger"


def _run_stage(fn, entries, args, n_jobs, label, deadline, on_result, verbose) -> tuple:
    """Dispatch one fidelity across workers. Returns (rows, exhausted).

    A generation runs for many minutes; a silent terminal is indistinguishable
    from a hung one, so results are printed as they arrive rather than collected
    in silence. Order is submission order, so the printout is stable across runs
    even though the workers are not.

    THE BUDGET IS CHECKED BETWEEN GENOMES, never inside one: a partially simulated
    episode is not evidence of anything, and interrupting one would leave the
    account state of a half-run book on disk. On exhaustion the joblib generator is
    dropped, which cancels whatever is still queued; anything already finished in a
    worker has already written its own checkpoint.
    """
    rows, exhausted = [], False
    par = Parallel(n_jobs=n_jobs, batch_size=1, return_as="generator")
    results = par(delayed(fn)(e, *args) for e in entries)
    try:
        for i, (_ghash, res, secs) in enumerate(results):
            entry = entries[i]
            rows.append((entry, res, secs))
            if on_result is not None:
                on_result(entry, res)
            if verbose:
                print("    %s %2d/%-2d  %-13s %s  score %+7.3f  %-34s %6.1fs"
                      % (label, i + 1, len(entries), entry["genome"]["signal"]["family"],
                         entry["hash"], res["score"], _describe(res), secs), flush=True)
            if deadline is not None and time.time() > deadline:     # io-boundary
                exhausted = (i + 1) < len(entries)
                if exhausted:
                    break
    finally:
        del results                    # dropping the generator cancels the queue
    return rows, exhausted


def _family_timings(rows) -> str:
    """Seconds per family, for the genomes this run actually simulated. Reused
    checkpoints carry no timing and are counted separately, not as instant work."""
    by = {}
    for entry, _res, secs in rows:
        if secs > 0:
            by.setdefault(entry["genome"]["signal"]["family"], []).append(secs)
    if not by:
        return "nothing simulated this run (all reused from disk)"
    return "  ".join("%s %d x %.0fs" % (fam, len(v), float(np.mean(v)))
                     for fam, v in sorted(by.items(), key=lambda kv: -np.mean(kv[1])))


# ── the generation ─────────────────────────────────────────────────────────────
def run_generation(market, entries, generation: int, cost=None, n_jobs=1, evolve=True,
                   deadline=None, state_dir=None, verbose=True) -> dict:
    """Evaluate one generation and, if it completed, breed the next.

    Pure of the CLI, the network and the cache: verify.py test 2 drives this exact
    function over a synthetic market, which is what makes the determinism proof a
    proof about the production path rather than about a copy of it.

    `deadline` is an absolute wall-clock instant (seconds) or None for unlimited.
    Returns a dict with "complete" False when the budget ran out — the caller
    prints, exits 0, and the next run picks up the checkpoints.
    """
    cost = cost if cost is not None else CostModel()
    identity = (market.data_hash, market.panel_hash, config.config_hash())
    out = {"generation": generation, "complete": False, "stage": "F0",
           "f0": [], "f1": [], "npz": None, "entries_next": None, "hof": None,
           "scores": {}, "f0_secs": 0.0, "f1_secs": 0.0, "resumed_f0": 0,
           "resumed_f1": 0, "orphans": {}}

    # Before anything else: an evaluation that happened but was never recorded
    # would undercount n_trials and make every DSR in the project optimistic.
    out["orphans"] = sweep_orphan_checkpoints(generation, state_dir, verbose)

    # ── F0: screen everything on three pre-vault eras ─────────────────────────
    timing: dict = {}                  # hash -> seconds THIS run spent on it
    have = _ledger_f0(generation, identity, state_dir)
    # ONE EPISODE PER HASH, the same rule F1's finalist list follows. evolution
    # dedups the population it breeds, but a random seed draw can collide, and two
    # slots holding the same genome must not buy two identical screen episodes:
    # the ledger's (hash, generation, fidelity) key keeps one row and the second
    # episode is pure spend. `have` is keyed by hash too, so every entry — copies
    # included — still gets its result below.
    todo, queued = [], set()
    for e in entries:
        if e["hash"] not in have and e["hash"] not in queued:
            todo.append(e)
            queued.add(e["hash"])
    out["resumed_f0"] = sum(1 for e in entries if e["hash"] in have)
    if verbose:
        print("\n  F0 screen : %d genomes x %d eras, %d-symbol point-in-time universes, "
              "%d workers%s" % (len(entries), len(config.SCREEN_ERAS),
                                config.SCREEN_UNIVERSE_N, n_jobs,
                                "" if not out["resumed_f0"] else
                                "  (%d already on the ledger)" % out["resumed_f0"]))
    t0 = time.time()                                           # io-boundary
    if todo:
        eras = evaluate.screen_markets(market)
        if verbose:
            for (start, end, sub) in eras:
                print("      era %s..%s  %d symbols (panel %s)"
                      % (start[:7], end[:7], len(sub.symbols), sub.panel_hash))

        def _record_f0(entry, res):
            _append_trial(generation, entry, res, "F0",
                          float(np.mean(res["era_sharpes"])), res["n_days"],
                          identity, state_dir)

        fresh, exhausted = _run_stage(_screen_one, todo, (eras, cost), n_jobs, "F0",
                                      deadline, _record_f0, verbose)
        # The era panels have done their job; do not ship 3 more copies of the
        # panel to every F1 worker.
        market._era_memo = None                                # noqa: SLF001
        for entry, res, secs in fresh:
            have[entry["hash"]] = res
            timing[entry["hash"]] = secs
        if exhausted:
            out["f0"] = [(e, have[e["hash"]], timing.get(e["hash"], 0.0))
                         for e in entries if e["hash"] in have]
            out["f0_secs"] = time.time() - t0                  # io-boundary
            return out
    out["f0_secs"] = time.time() - t0                          # io-boundary
    out["f0"] = [(e, have[e["hash"]], timing.get(e["hash"], 0.0)) for e in entries]

    # ── who gets a full evaluation ────────────────────────────────────────────
    # The top SCREEN_FRAC by screen score, PLUS every elite. Elites are carried
    # verbatim precisely because they were the best measurement available, and a
    # measurement made two generations ago on less data is not current: their F1
    # score has to be re-earned every generation, whatever the screen thinks. The
    # ledger's (hash, generation, fidelity) key keeps that to one row per
    # generation rather than a duplicate.
    n_full = max(1, int(math.ceil(config.SCREEN_FRAC * len(entries))))
    ranked = sorted(out["f0"], key=lambda row: (-row[1]["score"], row[0]["hash"]))
    keep = {row[0]["hash"] for row in ranked[:n_full]}
    keep |= {e["hash"] for e in entries if e["op"] == "elite"}
    # One entry per HASH: evolution dedups the population it breeds, but a random
    # seed draw can collide, and two slots holding the same genome must not buy two
    # identical episodes (the ledger would keep one row and the second would be
    # pure waste).
    finalists, taken = [], set()
    for row in ranked:
        if row[0]["hash"] in keep and row[0]["hash"] not in taken:
            finalists.append(row[0])
            taken.add(row[0]["hash"])

    out["stage"] = "F1"
    if verbose:
        print("\n  F1 full   : %d of %d (top %d by F0 score + %d elites), %s -> %s, "
              "full %d-symbol universe"
              % (len(finalists), len(entries), n_full, len(keep) - n_full,
                 market.dates[0].date(), config.VAULT_START, len(market.symbols)))

    done, todo = {}, []
    for entry in finalists:
        res = load_f1_checkpoint(generation, entry["hash"], identity, state_dir)
        if res is None:
            todo.append(entry)
        else:
            done[entry["hash"]] = res
    out["resumed_f1"] = len(done)
    if verbose and done:
        print("      resuming %d checkpointed episodes from %s"
              % (len(done), os.path.relpath(tmp_dir(generation, state_dir),
                                            state_dir or config.STATE_DIR)))

    def _record_f1(entry, res):
        _append_trial(generation, entry, res, "F1", res["sharpe_prevault"],
                      res["n_days_prevault"], identity, state_dir)

    t1 = time.time()                                           # io-boundary
    fresh, exhausted = _run_stage(_full_one, todo, (market, cost, generation, identity,
                                                    state_dir),
                                  n_jobs, "F1", deadline, _record_f1, verbose)
    out["f1_secs"] = time.time() - t1                          # io-boundary
    for entry, res, secs in fresh:
        done[entry["hash"]] = res
        timing[entry["hash"]] = secs
    # A resumed checkpoint still owes the ledger its row: the run that computed it
    # may have been killed between the worker's write and the parent's append.
    for entry in finalists:
        res = done.get(entry["hash"])
        if res is not None and res.get("resumed"):
            _record_f1(entry, res)
    out["f1"] = [(e, done[e["hash"]], timing.get(e["hash"], 0.0))
                 for e in finalists if e["hash"] in done]
    if exhausted:
        return out

    # ── the generation is complete: artifact, then breed ──────────────────────
    out["complete"] = True
    out["stage"] = "done"
    # write_returns_matrix is write-once, which is what makes a resumed run safe
    # and what makes a re-run under CHANGED inputs dangerous: it would hand back
    # the old generation's matrix while this run's freshly simulated returns —
    # the ones breeding is about to select on — went nowhere. Only a matrix whose
    # stored identity matches may be reused, and only if this run had nothing new
    # to contribute to it.
    npz_path = ledger.returns_path(generation, state_dir)
    if fresh and os.path.exists(npz_path):
        prior = ledger.load_returns_matrix(generation, state_dir)
        prior_id = (prior["data_hash"], prior["panel_hash"], prior["config_hash"])
        if prior_id != tuple(identity):
            raise IdentityDrift(
                "generation %d already has %s, and this run simulated %d episode(s) "
                "afresh under different inputs:\n"
                "    on disk : data %s | panel %s | config %s\n"
                "    this run: data %s | panel %s | config %s\n"
                "The artifact is write-once, so the file that would be handed to "
                "PBO and DSR is the old one while the population was bred from the "
                "new numbers. Either restore the inputs the artifact names, or "
                "evaluate this population as a new generation."
                % ((generation, os.path.basename(npz_path), len(fresh))
                   + tuple("(unrecorded)" if not v else v for v in prior_id)
                   + tuple(identity)))
    out["npz"] = ledger.write_returns_matrix(
        generation, {e["hash"]: done[e["hash"]] for e in finalists},
        state_dir=state_dir, identity=identity)

    # Selection score: F1 where the genome earned one, the screen's number where it
    # did not (evolution.next_generation documents what that mixes).
    scores = {e["hash"]: res["score"] for e, res, _s in out["f0"]}
    scores.update({e["hash"]: done[e["hash"]]["score"] for e in finalists if e["hash"] in done})
    out["scores"] = scores

    if evolve:
        out["entries_next"] = evolution.next_generation(entries, scores, generation,
                                                        market.feature_names)
        save_population(out["entries_next"], generation + 1, state_dir)
        out["hof"] = evolution.update_hall_of_fame(
            evolution.load_hall_of_fame(state_dir),
            [(e, done[e["hash"]]) for e in finalists if e["hash"] in done],
            generation)
        evolution.save_hall_of_fame(out["hof"], state_dir)
        # Only now: the checkpoints are spent, the artifact is written and the
        # population has moved on. --no-evolve leaves them in place on purpose —
        # that generation is still open, and its next run should not pay twice.
        shutil.rmtree(tmp_dir(generation, state_dir), ignore_errors=True)
    return out


def _print_summary(res: dict, market, entries, n_jobs: int, elapsed: float,
                   rows_before: int) -> None:
    """Everything below is PRE-VAULT; the vault has not been read."""
    generation = res["generation"]
    f1 = res["f1"]
    print("\n  ── generation %d summary %s" % (generation, "─" * 44))
    if f1:
        sr = np.array([r["sharpe_prevault"] for _e, r, _s in f1])
        best_i = int(np.argmax(sr))
        print("  F1 net Sharpe (pre-vault): best %+.2f (%s, %s)  median %+.2f  worst %+.2f"
              % (sr[best_i], f1[best_i][0]["genome"]["signal"]["family"],
                 f1[best_i][0]["hash"], float(np.median(sr)), sr.min()))
        print("  scored days per F1 genome: %d-%d (out of %d bars before the vault)"
              % (min(r["n_days_prevault"] for _e, r, _s in f1),
                 max(r["n_days_prevault"] for _e, r, _s in f1),
                 int(market.dates.searchsorted(pd.Timestamp(config.VAULT_START)))))
    print("  trial ledger             : %d distinct genomes (%d new), %d refits audited"
          % (ledger.n_trials(), ledger.n_trials() - rows_before,
             sum(r.get("n_fits", 0) for _e, r, _s in f1)))
    print("  vault accesses           : %d  (this run made none — F0/F1 never read it)"
          % ledger.vault_trials())
    if res["npz"]:
        print("  returns matrix           : %s" % os.path.relpath(res["npz"], config.ROOT))
    print("  wall clock               : %.1f min total — F0 %.1f min, F1 %.1f min, "
          "%d workers%s" % (elapsed, res["f0_secs"] / 60.0, res["f1_secs"] / 60.0, n_jobs,
                            "" if not (res["resumed_f0"] or res["resumed_f1"]) else
                            "  (%d F0 + %d F1 reused, not re-simulated)"
                            % (res["resumed_f0"], res["resumed_f1"])))
    print("  F0 cost by family        : %s" % _family_timings(res["f0"]))
    print("  F1 cost by family        : %s" % _family_timings(f1))
    # Extrapolate from THIS run's family mix — the only honest basis available,
    # and a wide one: an hgb genome refitting every 63 days costs two orders of
    # magnitude more than a momentum rule, so a population that drifts toward hgb
    # drifts toward the budget. Wall clock is bounded BELOW by two things and it
    # is the larger that binds: the total work spread over the workers, and the
    # single slowest genome (which no amount of parallelism divides).
    timed = [(rows, count) for rows, count in
             ((res["f0"], config.POP_SIZE),
              (f1, math.ceil(config.SCREEN_FRAC * config.POP_SIZE)))
             if any(s > 0 for _e, _r, s in rows)]
    if timed:
        projected, slowest = 0.0, 0.0
        for rows, count in timed:
            secs = [s for _e, _r, s in rows if s > 0]
            work = sum(secs) * (count / max(len(secs), 1)) / config.N_JOBS
            projected += max(work, max(secs)) / 60.0
            slowest = max(slowest, max(secs) / 60.0)
        print("  projected at POP_SIZE=%d : ~%.0f min/generation on %d workers "
              "(budget %d min)%s — extrapolated from this run's family mix; the "
              "slowest single genome (%.0f min) is a floor no parallelism removes"
              % (config.POP_SIZE, projected, config.N_JOBS, config.GEN_TIME_BUDGET_MIN,
                 "" if projected <= config.GEN_TIME_BUDGET_MIN else "  ** OVER BUDGET: "
                 "the generation will span sessions, which is what the checkpoints "
                 "are for **", slowest))

    if res["entries_next"] is not None:
        ops = evolution.op_counts(res["entries_next"])
        print("  %-25s: %s -> %d distinct genomes carried into %s"
              % ("bred generation %d" % (generation + 1),
                 ", ".join("%s %d" % kv for kv in sorted(ops.items())),
                 len({e["hash"] for e in res["entries_next"]}), POPULATION_FILE))
        # Against the parent population only — a mutant that coincides with a
        # genome from five generations ago is new HERE, and the ledger is where
        # "ever evaluated" is answered (n_trials above).
        fresh_hashes = {e["hash"] for e in res["entries_next"]} - {e["hash"] for e in entries}
        print("  %-25s: %d of %d slots hold a genome the parent population did not"
              % ("search progress", len(fresh_hashes), len(res["entries_next"])))
    if res["hof"]:
        print("  %-25s: %d records, best:"
              % ("hall of fame (top %d)" % config.HOF_SIZE, len(res["hof"])))
        for r in res["hof"][:3]:
            print("      %s %-13s SR %+.2f  gen %d  born gen %d via %s"
                  % (r["hash"], r["family"], r["sharpe_prevault"], r["generation"],
                     r["birth_gen"], r["op"]))


def _summary_alert(res: dict, generation: int, status: str, budget_min: float) -> tuple:
    """The nightly one-liner, from the numbers the run just printed.

    An incomplete generation is a DIFFERENT state from a complete one, which is
    what keeps the anti-spam honest: a second session that is still grinding
    through the same generation says nothing, and the session that finishes it
    speaks.
    """
    sr = [r["sharpe_prevault"] for _e, r, _s in res["f1"]]
    champ = registry.champion()
    detail = ("%d F0 screens + %d F1 episodes this run." % (len(res["f0"]), len(res["f1"])))
    if status != "complete":
        detail += (" The %.0f-min budget was spent during %s; the work is "
                   "checkpointed and the next session resumes it. Nothing was bred — "
                   "breeding from a half-evaluated generation would select on which "
                   "genomes happened to be cheap." % (budget_min, res["stage"]))
    return alerts_arena.generation_summary(
        generation=generation, n_evaluated=len(res["f1"]), n_trials=ledger.n_trials(),
        best_sharpe=max(sr) if sr else None,
        median_sharpe=float(np.median(sr)) if sr else None,
        champion="%s (promoted through all ten gates)" % champ[0] if champ else None,
        status=status, platform=ledger.platform_tag(), detail=detail)


def main() -> int:
    ap = argparse.ArgumentParser(description="run one arena generation")   # io-boundary
    ap.add_argument("--init", action="store_true",
                    help="seed a fresh random population (overwrites population.json)")
    ap.add_argument("--no-evolve", action="store_true",
                    help="evaluate and persist, but do not breed the next population")
    ap.add_argument("--pop", type=int, default=None,
                    help="population size; --init only (default config.POP_SIZE)")
    ap.add_argument("--jobs", type=int, default=config.N_JOBS, help="worker processes")
    ap.add_argument("--budget-min", type=float, default=float(config.GEN_TIME_BUDGET_MIN),
                    help="wall-clock budget in minutes; the run checkpoints and exits 0 "
                         "when it is spent (default config.GEN_TIME_BUDGET_MIN)")
    ap.add_argument("--refresh", action="store_true",
                    help="re-download the price cache before evaluating (the cloud "
                         "path; skipped automatically when the generation is already "
                         "mid-flight, which a refresh would break)")
    ap.add_argument("--send", action="store_true",
                    help="deliver the summary alert; without it the exact text is "
                         "printed and nothing is sent")
    args = ap.parse_args()

    t_start = time.time()                                      # io-boundary
    deadline = t_start + args.budget_min * 60.0                # io-boundary
    print("arena generation runner")

    # 1. Universe and data. Cache-first and offline: a download inside an
    #    evaluation path would let two runs of the same generation disagree.
    #    (build_universe reads signal_lab's cached S&P table; with
    #    USE_LIVE_SP500_LIST off it only supplies sector anchors, and a missing
    #    cache degrades to the static fallback rather than failing.)
    # Before anything reads the ledger: a union merge (see .gitattributes) can
    # leave a row twice, and n_trials() is the N every deflated Sharpe here is
    # deflated by.
    ledger.dedup_ledger()

    universe = config.import_sibling("universe", config.SIGNAL_LAB)
    wanted = universe.build_universe()[0]

    pending = 0 if args.init else pending_generation()
    if args.refresh:
        if generation_in_flight(pending):
            print("  refresh   : SKIPPED — generation %s is mid-flight. New bars would "
                  "move data_hash under work already on disk, orphaning every "
                  "checkpoint and stopping the run on IdentityDrift; this session "
                  "finishes on the committed cache and the next generation takes "
                  "the new data." % pending)
        else:
            # The symbols ALREADY cached, not everything the universe wants: the
            # committed cache is what defines this arena's symbol set, and pulling a
            # newly-listed name into it mid-stream would move data_hash for a reason
            # that has nothing to do with new bars.
            refresh_cache(datafeed.in_cache(wanted) or wanted)

    symbols = datafeed.in_cache(wanted)[:config.UNIVERSE_SIZE]
    market = datafeed.load_market(symbols, start=config.DATA_START)

    cache_age, bar_age = data_staleness(market)
    if max(cache_age, bar_age) > config.MAX_DATA_STALENESS_DAYS:
        print("  ABORT: data is stale — cache %.1f days old, last bar %s (%.1f days "
              "old); limit is %d. Refresh the cache before evaluating; a stale "
              "sandbox scores genomes on a market that no longer exists."
              % (cache_age, market.dates[-1].date(), bar_age,
                 config.MAX_DATA_STALENESS_DAYS))
        if generation_in_flight(pending):
            # The two safety rules meet here and deadlock: --refresh refuses to
            # move data_hash under a half-evaluated generation, and this check
            # refuses to evaluate on the cache that refusal preserved. Neither is
            # wrong and neither can break the tie, so the runner names the only
            # exit — a human — instead of retrying twice a day forever.
            print("  DEADLOCK: generation %s is MID-FLIGHT, and the two safety rules "
                  "now cancel each other — --refresh refuses to move data_hash under "
                  "a half-evaluated generation (it skips the download), and this "
                  "staleness check refuses to evaluate on the cache that refusal "
                  "preserved. The scheduled job repeats this forever; it ends only "
                  "when a human ends it, one of two ways: "
                  "FINISH it — run without --refresh and with enough budget to "
                  "complete generation %s on the committed cache, after which the "
                  "next session refreshes normally; or ABANDON it — refresh the "
                  "cache by hand and bump population.json's \"generation\" so this "
                  "population is evaluated as a NEW generation (what IdentityDrift "
                  "tells you to do), leaving %s and every existing ledger row "
                  "exactly where they are. Do not delete ledger rows: n_trials() "
                  "counts looks the search really took."
                  % (pending, pending, os.path.relpath(tmp_dir(pending), config.ROOT)))
        title, body = alerts_arena.data_stale_summary(
            cache_age, bar_age, market.dates[-1].date(),
            config.MAX_DATA_STALENESS_DAYS, job="generation")
        alert("generation", pending, "stale-data", title, body, args.send)
        return 1

    arena_features.build_features(market)
    print("  data      : %d/%d symbols cached, %s -> %s (%d bars)"
          % (len(symbols), len(wanted), market.dates[0].date(), market.dates[-1].date(),
             len(market)))
    print("  staleness : cache %.1fd, last bar %.1fd (limit %dd)"
          % (cache_age, bar_age, config.MAX_DATA_STALENESS_DAYS))
    print("  identity  : data %s | panel %s | config %s | %s"
          % (market.data_hash, market.panel_hash, config.config_hash(),
             ledger.platform_tag()))

    # 2. Population.
    if args.init:
        entries = seed_population(args.pop or config.POP_SIZE, market.feature_names,
                                  generation=0)
        generation = 0
        save_population(entries, generation)
        print("  population: seeded %d genomes at generation %d (%d distinct hashes)"
              % (len(entries), generation, len({e["hash"] for e in entries})))
    else:
        if args.pop is not None:
            print("  note      : --pop is only honoured with --init; the population file "
                  "carries its own size (%s)" % POPULATION_FILE)
        entries, generation = load_population()
        ops = evolution.op_counts(entries)
        print("  population: loaded %d genomes at generation %d (%s)"
              % (len(entries), generation, ", ".join("%s %d" % kv for kv in sorted(ops.items()))))

    n_jobs = max(1, min(args.jobs, len(entries)))
    rows_before = ledger.n_trials()
    print("  budget    : %.0f min from now, checked between genome evaluations"
          % args.budget_min)

    try:
        res = run_generation(market, entries, generation, cost=CostModel(), n_jobs=n_jobs,
                             evolve=not args.no_evolve, deadline=deadline, verbose=True)
    except IdentityDrift as exc:
        # Exit 1, loudly. A scheduled job that fails is visible; a scheduled job
        # that quietly writes a ledger describing a different run than the one it
        # bred from is not, and this project's entire claim is that the ledger
        # says what happened.
        print("\n  ABORT: inputs changed under an evaluated generation\n")
        for line in str(exc).splitlines():
            print("  %s" % line)
        title, body = alerts_arena.generation_summary(
            generation, 0, ledger.n_trials(), status="identity-drift",
            platform=ledger.platform_tag(),
            detail="Inputs moved under a generation that was already partly on disk. "
                   "Nothing was bred and nothing was overwritten; this needs a human "
                   "(restore the data the ledger row names, or evaluate the "
                   "population as a new generation).")
        alert("generation", generation, "identity-drift", title, body, args.send)
        return 1

    elapsed = (time.time() - t_start) / 60.0                   # io-boundary
    if not res["complete"]:
        print("\n  ── generation %d INCOMPLETE %s" % (generation, "─" * 40))
        print("  the %.0f-min budget was spent during %s; %d F0 and %d F1 evaluations "
              "are on disk" % (args.budget_min, res["stage"], len(res["f0"]), len(res["f1"])))
        print("  resumable — rerun to continue gen %d. Nothing was bred: a population "
              "selected from a half-evaluated generation would be selecting on which "
              "genomes happened to be cheap." % generation)
        print("  elapsed   : %.1f min" % elapsed)
        title, body = _summary_alert(res, generation, "incomplete", args.budget_min)
        alert("generation", generation, "incomplete", title, body, args.send)
        return 0

    _print_summary(res, market, entries, n_jobs, elapsed, rows_before)
    title, body = _summary_alert(res, generation, "complete", args.budget_min)
    alert("generation", generation, "complete", title, body, args.send)
    if args.no_evolve:
        print("  --no-evolve: stopped before breeding; generation %d stays open and its "
              "checkpoints are kept, so the next run resumes rather than re-simulates."
              % generation)

    print("\n  Every Sharpe above is PRE-VAULT and UNGATED: these genomes were")
    print("  screened and ranked, not promoted. The population was searched, so the")
    print("  best of it is biased upward by construction — that is what the trial")
    print("  ledger, DSR, PBO, the vault and the gate stack exist to discount.")
    print("  Survivorship: the universe is today's S&P membership, so long results")
    print("  flatter and short results understate. Sandbox output is a claim about")
    print("  the past, not a guarantee — and not financial advice.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
