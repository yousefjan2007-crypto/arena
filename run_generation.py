"""
One generation of the arena, one process, one exit code. The nightly entry point.

    python3 run_generation.py --init --no-evolve --pop 16

WHAT THIS PHASE DOES: seed a population, screen it (F0), fully evaluate the
survivors (F1), record every evaluation in the trial ledger, write the Tier-A
returns matrix, print an honest summary, exit. `--no-evolve` stops there; Phase 4
adds the breeding step, Phase 5 the gates, Phase 6 the alert and the cron.

ONE-SHOT BY DESIGN. Fetch, evaluate, append, print, exit — no loop, no daemon.
The scheduler (launchd, then GitHub Actions) re-runs it; anti-spam and resume come
from what is already on disk (the ledger is idempotent by genome hash and
generation), never from in-process state. That is the pattern every scheduled job
in this workspace uses, and it is what makes a killed run harmless.

WALL-CLOCK LIVES HERE AND NOWHERE ELSE. This file is an I/O boundary: it reads
cache mtimes to decide whether the data is too stale to evaluate, and it times
itself so the human reading the output knows what a generation costs. Nothing it
measures feeds a simulated quantity — verify.py's scanner skips run_*.py for
exactly this reason, so every clock call below is commented as the boundary it is.

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
import json                                                           # noqa: E402
import math                                                           # noqa: E402
import time                                                           # noqa: E402

import numpy as np                                                    # noqa: E402
import pandas as pd                                                   # noqa: E402
from joblib import Parallel, delayed                                  # noqa: E402

import config                       # FIRST: puts the siblings on sys.path  # noqa: E402
import datafeed                                                       # noqa: E402
import evaluate                                                       # noqa: E402
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
    """`n` uniformly drawn genomes, each from its own reproducible stream.

    child_rng(SEED, generation, "genesis", i) rather than one shared rng: drawing
    from a single stream would make genome i depend on how many genomes were drawn
    before it, so `--pop 16` and `--pop 64` would share nothing. This way the
    first 16 of a 64-genome seed are the same 16.
    """
    return [gn.random_genome(gn.child_rng(config.SEED, generation, "genesis", i),
                             feature_names)
            for i in range(n)]


def save_population(genomes, generation: int, state_dir=None) -> str:
    """state/population.json — a POINTER to the live generation, like champion.json.

    This is the one file a generation rewrites. It is not the record: every genome
    in it was appended to the trial ledger when it was evaluated, and the ledger is
    what nothing may rewrite.
    """
    path = os.path.join(state_dir or config.STATE_DIR, POPULATION_FILE)
    payload = {"generation": int(generation),
               "genomes": [g.to_dict() for g in genomes],
               "hashes": [g.hash() for g in genomes]}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1, sort_keys=True)
    os.replace(tmp, path)
    return path


def load_population(state_dir=None) -> tuple:
    path = os.path.join(state_dir or config.STATE_DIR, POPULATION_FILE)
    if not os.path.exists(path):
        raise SystemExit("no %s — run once with --init to seed a population" % path)
    with open(path) as f:
        payload = json.load(f)
    return [gn.from_dict(d) for d in payload["genomes"]], int(payload["generation"])


# ── workers (module level: joblib has to be able to pickle them) ──────────────
def _screen_one(genome, era_markets, cost):
    t0 = time.time()                                           # io-boundary
    res = evaluate.screen(genome, None, cost, era_markets=era_markets)
    return genome.hash(), res, time.time() - t0                # io-boundary


def _full_one(genome, market, cost):
    t0 = time.time()                                           # io-boundary
    res = evaluate.full_eval(genome, market, cost)
    return genome.hash(), res, time.time() - t0                # io-boundary


def _run_parallel(fn, genomes, shared, cost, n_jobs, label):
    """Dispatch one fidelity across workers, printing each genome as it lands.

    A generation runs for many minutes; a silent terminal is indistinguishable
    from a hung one, so results are printed as they arrive rather than collected
    in silence. Order is submission order, so the printout is stable across runs
    even though the workers are not.
    """
    out = []
    par = Parallel(n_jobs=n_jobs, batch_size=1, return_as="generator")
    for i, (ghash, res, secs) in enumerate(
            par(delayed(fn)(g, shared, cost) for g in genomes)):
        g = genomes[i]
        extra = ("eras " + "/".join("%+.2f" % s for s in res["era_sharpes"])
                 if "era_sharpes" in res
                 else "SR %+.2f over %d days" % (res["sharpe_prevault"],
                                                 res["n_days_prevault"]))
        print("    %s %2d/%-2d  %-13s %s  score %+7.3f  %-34s %6.1fs"
              % (label, i + 1, len(genomes), g.signal.family, ghash,
                 res["score"], extra, secs), flush=True)
        out.append((g, res, secs))
    return out


def _family_timings(rows) -> str:
    by = {}
    for g, _res, secs in rows:
        by.setdefault(g.signal.family, []).append(secs)
    return "  ".join("%s %d x %.0fs" % (fam, len(v), float(np.mean(v)))
                     for fam, v in sorted(by.items(), key=lambda kv: -np.mean(kv[1])))


def main() -> int:
    ap = argparse.ArgumentParser(description="run one arena generation")   # io-boundary
    ap.add_argument("--init", action="store_true",
                    help="seed a fresh random population (overwrites population.json)")
    ap.add_argument("--no-evolve", action="store_true",
                    help="evaluate and persist, but do not breed (Phase 4 adds breeding)")
    ap.add_argument("--pop", type=int, default=config.POP_SIZE, help="population size")
    ap.add_argument("--jobs", type=int, default=config.N_JOBS, help="worker processes")
    args = ap.parse_args()
    if not args.no_evolve:
        raise SystemExit("--no-evolve is required at this phase: evolution.py is Phase 4")

    t_start = time.time()                                      # io-boundary
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
        genomes = seed_population(args.pop, market.feature_names, generation=0)
        generation = 0
        save_population(genomes, generation)
        print("  population: seeded %d genomes at generation %d (%d distinct hashes)"
              % (len(genomes), generation, len({g.hash() for g in genomes})))
    else:
        genomes, generation = load_population()
        print("  population: loaded %d genomes at generation %d" % (len(genomes), generation))

    cost = CostModel()
    n_jobs = max(1, min(args.jobs, len(genomes)))
    rows_before = ledger.n_trials()

    # 3. F0 screen — everything, on three pre-vault eras.
    eras = evaluate.screen_markets(market)
    print("\n  F0 screen : %d genomes x %d eras, %d-symbol point-in-time universes, "
          "%d workers" % (len(genomes), len(eras), config.SCREEN_UNIVERSE_N, n_jobs))
    for (start, end, sub) in eras:
        print("      era %s..%s  %d symbols (panel %s)"
              % (start[:7], end[:7], len(sub.symbols), sub.panel_hash))
    t_f0 = time.time()                                         # io-boundary
    screened = _run_parallel(_screen_one, genomes, eras, cost, n_jobs, "F0")
    f0_secs = time.time() - t_f0                               # io-boundary

    for g, res, _s in screened:
        ledger.record_trial(generation, g, "F0", res["score"],
                            float(np.mean(res["era_sharpes"])), res["n_days"],
                            market.data_hash, market.panel_hash)

    # The era panels have done their job; do not ship 3 more copies of the panel
    # to every F1 worker.
    market._era_memo = None

    # 4. F1 — the survivors, full history, their own genes.
    n_full = max(1, int(math.ceil(config.SCREEN_FRAC * len(genomes))))
    ranked = sorted(screened, key=lambda row: (-row[1]["score"], row[0].hash()))
    finalists = [row[0] for row in ranked[:n_full]]
    print("\n  F1 full   : top %d of %d by F0 score, %s -> %s, full %d-symbol universe"
          % (n_full, len(genomes), market.dates[0].date(), config.VAULT_START,
             len(market.symbols)))
    t_f1 = time.time()                                         # io-boundary
    full = _run_parallel(_full_one, finalists, market, cost, n_jobs, "F1")
    f1_secs = time.time() - t_f1                               # io-boundary

    for g, res, _s in full:
        ledger.record_trial(generation, g, "F1", res["score"], res["sharpe_prevault"],
                            res["n_days_prevault"], market.data_hash, market.panel_hash)
    npz = ledger.write_returns_matrix(generation, {g.hash(): res for g, res, _s in full})

    # 5. Summary. Every number below is pre-vault; the vault has not been read.
    sr = np.array([res["sharpe_prevault"] for _g, res, _s in full])
    best_i = int(np.argmax(sr))
    audits = sum(len(res["fit_audit"]) for _g, res, _s in full)
    elapsed = (time.time() - t_start) / 60.0                   # io-boundary

    print("\n  ── generation %d summary %s" % (generation, "─" * 44))
    print("  F1 net Sharpe (pre-vault): best %+.2f (%s, %s)  median %+.2f  worst %+.2f"
          % (sr[best_i], full[best_i][0].signal.family, full[best_i][0].hash(),
             float(np.median(sr)), sr.min()))
    print("  scored days per F1 genome: %d-%d (out of %d bars before the vault)"
          % (min(r["n_days_prevault"] for _g, r, _s in full),
             max(r["n_days_prevault"] for _g, r, _s in full),
             int(market.dates.searchsorted(pd.Timestamp(config.VAULT_START)))))
    print("  trial ledger             : %d distinct genomes (%d new), %d refits audited"
          % (ledger.n_trials(), ledger.n_trials() - rows_before, audits))
    print("  vault accesses           : %d  (this run made none — F0/F1 never read it)"
          % ledger.vault_trials())
    print("  returns matrix           : %s" % os.path.relpath(npz, config.ROOT))
    print("  wall clock               : %.1f min total — F0 %.1f min, F1 %.1f min, "
          "%d workers" % (elapsed, f0_secs / 60.0, f1_secs / 60.0, n_jobs))
    print("  F0 cost by family        : %s" % _family_timings(screened))
    print("  F1 cost by family        : %s" % _family_timings(full))
    # Extrapolate from THIS run's family mix — the only honest basis available,
    # and a wide one: an hgb genome refitting every 63 days costs two orders of
    # magnitude more than a momentum rule, so a population that drifts toward hgb
    # drifts toward the budget. Wall clock is bounded BELOW by two things and it
    # is the larger that binds: the total work spread over the workers, and the
    # single slowest genome (which no amount of parallelism divides — this run
    # spent 21 of its 26 minutes with six of eight workers idle behind one hgb).
    projected = 0.0
    for rows, count in ((screened, config.POP_SIZE),
                        (full, math.ceil(config.SCREEN_FRAC * config.POP_SIZE))):
        secs = [s for _g, _r, s in rows]
        work = sum(secs) * (count / max(len(rows), 1)) / config.N_JOBS
        projected += max(work, max(secs)) / 60.0
    print("  projected at POP_SIZE=%d : ~%.0f min/generation on %d workers "
          "(budget %d min)%s — extrapolated from this run's family mix; the "
          "slowest single genome (%.0f min) is a floor no parallelism removes"
          % (config.POP_SIZE, projected, config.N_JOBS, config.GEN_TIME_BUDGET_MIN,
             "" if projected <= config.GEN_TIME_BUDGET_MIN else "  ** OVER BUDGET **",
             max(s for _g, _r, s in full) / 60.0))

    if args.no_evolve:
        print("  --no-evolve: stopping before the breeding step (Phase 4).")

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
