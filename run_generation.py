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
import csv                                                            # noqa: E402
import json                                                           # noqa: E402
import math                                                           # noqa: E402
import shutil                                                         # noqa: E402
import time                                                           # noqa: E402

import numpy as np                                                    # noqa: E402
import pandas as pd                                                   # noqa: E402
from joblib import Parallel, delayed                                  # noqa: E402

import config                       # FIRST: puts the siblings on sys.path  # noqa: E402
import datafeed                                                       # noqa: E402
import evaluate                                                       # noqa: E402
import evolution                                                      # noqa: E402
import features as arena_features                                     # noqa: E402
import genome as gn                                                   # noqa: E402
import ledger                                                         # noqa: E402
from env import CostModel                                             # noqa: E402

POPULATION_FILE = "population.json"


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


# ── population ─────────────────────────────────────────────────────────────────
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


def save_f1_checkpoint(generation: int, ghash: str, res: dict, identity, state_dir=None) -> str:
    """Persist one finished F1 episode, written by the worker that computed it.

    In the worker rather than the parent so that a run stopped by the time budget
    keeps every episode that FINISHED, not just the ones the parent had gotten
    around to collecting — with eight workers and 20-minute hgb episodes that is
    up to seven episodes saved per stop.

    `identity` is (data_hash, panel_hash, config_hash): a checkpoint is only
    reusable if the market, the panel and the settings are the ones it was
    computed under, so it is stored alongside and checked on the way back in.
    """
    path = _tmp_path(generation, ghash, state_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {k: np.asarray(res[k], dtype=np.float64) for k in _F1_ARRAYS}
    payload["dates"] = (pd.DatetimeIndex(res["dates"]).values
                        .astype("datetime64[ns]").astype(np.int64))
    for k in _F1_SCALARS:
        payload[k] = np.float64(res[k])
    payload["identity"] = np.array(list(identity) + [ghash], dtype="U32")
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
    res["resumed"] = True
    return res


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
            if (row["fidelity"] != "F0" or int(row["generation"]) != int(generation)
                    or (row["data_hash"], row["panel_hash"], row["config_hash"]) != tuple(identity)):
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
    save_f1_checkpoint(generation, entry["hash"], res, identity, state_dir)
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
           "scores": {}, "f0_secs": 0.0, "f1_secs": 0.0, "resumed_f0": 0, "resumed_f1": 0}

    # ── F0: screen everything on three pre-vault eras ─────────────────────────
    timing: dict = {}                  # hash -> seconds THIS run spent on it
    have = _ledger_f0(generation, identity, state_dir)
    todo = [e for e in entries if e["hash"] not in have]
    out["resumed_f0"] = len(entries) - len(todo)
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
            ledger.record_trial(generation, evolution.entry_genome(entry), "F0", res["score"],
                                float(np.mean(res["era_sharpes"])), res["n_days"],
                                identity[0], identity[1],
                                parent_hash=evolution.parent_field(entry["parent_hash"]),
                                birth_gen=entry["birth_gen"], state_dir=state_dir)

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
    finalists = [row[0] for row in ranked if row[0]["hash"] in keep]

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
        ledger.record_trial(generation, evolution.entry_genome(entry), "F1", res["score"],
                            res["sharpe_prevault"], res["n_days_prevault"],
                            identity[0], identity[1],
                            parent_hash=evolution.parent_field(entry["parent_hash"]),
                            birth_gen=entry["birth_gen"], state_dir=state_dir)

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
    out["npz"] = ledger.write_returns_matrix(
        generation, {e["hash"]: done[e["hash"]] for e in finalists}, state_dir=state_dir)

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
        print("  bred generation %-9d: %s -> %d distinct genomes carried into %s"
              % (generation + 1, ", ".join("%s %d" % kv for kv in sorted(ops.items())),
                 len({e["hash"] for e in res["entries_next"]}), POPULATION_FILE))
        fresh_hashes = {e["hash"] for e in res["entries_next"]} - {e["hash"] for e in entries}
        print("  %-25s: %d of %d slots are genomes the arena has never evaluated"
              % ("search progress", len(fresh_hashes), len(res["entries_next"])))
    if res["hof"]:
        print("  hall of fame (top %d)     : %d records, best:" % (config.HOF_SIZE, len(res["hof"])))
        for r in res["hof"][:3]:
            print("      %s %-13s SR %+.2f  gen %d  born gen %d via %s"
                  % (r["hash"], r["family"], r["sharpe_prevault"], r["generation"],
                     r["birth_gen"], r["op"]))


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
    args = ap.parse_args()

    t_start = time.time()                                      # io-boundary
    deadline = t_start + args.budget_min * 60.0                # io-boundary
    print("arena generation runner")

    # 1. Universe and data. Cache-first and offline: a download inside an
    #    evaluation path would let two runs of the same generation disagree.
    #    (build_universe reads signal_lab's cached S&P table; with
    #    USE_LIVE_SP500_LIST off it only supplies sector anchors, and a missing
    #    cache degrades to the static fallback rather than failing.)
    universe = config.import_sibling("universe", config.SIGNAL_LAB)
    wanted = universe.build_universe()[0]
    symbols = datafeed.in_cache(wanted)[:config.UNIVERSE_SIZE]
    market = datafeed.load_market(symbols, start=config.DATA_START)

    cache_age, bar_age = data_staleness(market)
    if max(cache_age, bar_age) > config.MAX_DATA_STALENESS_DAYS:
        print("  ABORT: data is stale — cache %.1f days old, last bar %s (%.1f days "
              "old); limit is %d. Refresh the cache before evaluating; a stale "
              "sandbox scores genomes on a market that no longer exists."
              % (cache_age, market.dates[-1].date(), bar_age,
                 config.MAX_DATA_STALENESS_DAYS))
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

    res = run_generation(market, entries, generation, cost=CostModel(), n_jobs=n_jobs,
                         evolve=not args.no_evolve, deadline=deadline, verbose=True)

    elapsed = (time.time() - t_start) / 60.0                   # io-boundary
    if not res["complete"]:
        print("\n  ── generation %d INCOMPLETE %s" % (generation, "─" * 40))
        print("  the %.0f-min budget was spent during %s; %d F0 and %d F1 evaluations "
              "are on disk" % (args.budget_min, res["stage"], len(res["f0"]), len(res["f1"])))
        print("  resumable — rerun to continue gen %d. Nothing was bred: a population "
              "selected from a half-evaluated generation would be selecting on which "
              "genomes happened to be cheap." % generation)
        print("  elapsed   : %.1f min" % elapsed)
        return 0

    _print_summary(res, market, entries, n_jobs, elapsed, rows_before)
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
