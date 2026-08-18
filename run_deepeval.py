"""
The weekly deep evaluation: F2 on the hall-of-fame leaders, ten gates, and either
a promotion or an honest refusal. One process, one exit code.

    python3 run_deepeval.py --dry              measure and print, write no decision
    python3 run_deepeval.py                    measure, decide, promote if it passes
    python3 run_deepeval.py --rollback <hash>  repoint the champion at a prior hash

THE WHOLE POINT OF THIS FILE IS TO MAKE PROMOTION EXPENSIVE. Nightly generations
search; this decides, and everything about it is arranged so that the decision
cannot be flattered:

  • EVERY PARTY IS RE-SIMULATED FRESH, incumbent included, on today's market and
    today's settings. Not read from a checkpoint (run_generation.load_f1_checkpoint
    documents why a resumed dict cannot answer this — no fit audit, no vault
    series), and never compared against a stored number from an older run. That is
    gate G1's like-for-like guarantee, and it is the failure that promoted
    signal_lab's champion off a 27-symbol run against 164-symbol challengers.
  • THE VAULT IS READ HERE AND NOWHERE ELSE, and every read appends a row to
    state/vault_access.csv (reason "gate_eval") so gate G3's DSR can be deflated
    by how often the vault has been consulted. Asking twice IS two looks.
  • THE DRY RUN STILL PAYS THE HONESTY TAX. `--dry` suppresses the DECISION
    writes — the champion pointer, the artifacts, the deep-eval history — but it
    still appends the F2 trial rows and the vault accesses, because the
    evaluation genuinely happened and the ledger's whole claim is that it says
    what happened. A dry run that quietly did not count would make every DSR in
    the project optimistic, which is the one direction this project may not be
    wrong in.
  • A REFUSAL IS A RESULT. The gate table is printed in full either way, with
    every value beside its threshold, so "it missed on G5 by 0.02" is visible
    rather than collapsed into "no promotion".

WHAT F2 IS (docs/DESIGN.md "Weekly F2 deep eval"), per candidate:
    DSR             deflated_sharpe(daily_net, ledger.dsr_trial_sharpes()) and
                    nothing else — the accessor fixes both the units and N.
                    ledger.f1_sharpes() is the documented trap; it is not used here.
    vault           vault Sharpe and vault DSR at N = vault_trials (a counted read)
    PBO             CSCV S=16 over the latest generation's pre-vault returns matrix
    CPCV            C(8,2) = 28 combinatorial purged paths with real refits
    bootstrap       5,000-resample block-bootstrap CI of the net Sharpe
    stress          the whole episode again at 2x costs with borrow tripled
    regime slices   cumulative net return through four crisis windows
    ruin MC         P(drawdown > 40% in two years), GARCH-t and block bootstrap

WALL-CLOCK LIVES HERE, as it does in run_generation.py: this is an I/O boundary.
It reads cache mtimes for the staleness guard, enforces the time budget, and times
itself so the human reading the output knows what a deep eval costs. Nothing it
measures feeds a simulated quantity, and every call below is commented as the
boundary it is.
"""
from __future__ import annotations

import os

# BEFORE the import chain reaches sklearn: one BLAS/OpenMP thread per worker.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse                                                      # noqa: E402
import csv                                                           # noqa: E402
import glob                                                          # noqa: E402
import math                                                          # noqa: E402
import time                                                          # noqa: E402

import numpy as np                                                   # noqa: E402
import pandas as pd                                                  # noqa: E402
from joblib import Parallel, delayed                                 # noqa: E402

import config                       # FIRST: puts the siblings on sys.path  # noqa: E402
import alerts_arena                                                  # noqa: E402
import datafeed                                                      # noqa: E402
import evaluate                                                      # noqa: E402
import evolution                                                     # noqa: E402
import features as arena_features                                    # noqa: E402
import gates                                                         # noqa: E402
import genome as gn                                                  # noqa: E402
import ledger                                                        # noqa: E402
import registry                                                      # noqa: E402
import reports as arena_reports                                      # noqa: E402
import run_generation                                                # noqa: E402
from env import CostModel                                            # noqa: E402
from strategy import StrategyAgent                                   # noqa: E402

HISTORY_FILE = "deepeval_history.csv"
# `candidates` packs one entry per candidate as hash:all_pass:failed, ";"-separated,
# with the failed gates "+"-joined ("35d8...:0:G2+G5") — the same compound-field
# idiom the decision log uses, so a reader splits rather than compares.
# `gates_failed` is the top-ranked candidate's failures, the per-candidate detail
# being in `candidates`. `complete` is 0 when the time budget stopped the run
# before every candidate was evaluated.
# THIS FILE HOLDS DECISIONS ONLY. A --dry run appends NOTHING here (its output is
# its record), so every row describes a run that was allowed to change the
# champion — there is no "dry" column to filter on, because there is nothing to
# filter out. One row per RUN, not per week: re-running a generation appends
# another row, so Phase 7's graduation trigger ("the same champion survives 3
# consecutive weekly deep evals") groups by `generation` and counts only
# `complete=1` rows.
# `ledger_drift` names any party whose F2 trial row was already on record under an
# EARLIER data/panel/config vintage, so the older row still feeds G2/G3's
# trial-Sharpe dispersion (see _ledger_f2). Empty is the normal case.
HISTORY_COLUMNS = ("generation", "champion_hash_before", "champion_hash_after",
                   "promoted", "n_candidates", "candidates", "gates_failed",
                   "ledger_drift", "complete", "data_hash", "panel_hash",
                   "config_hash", "platform")


def history_path(state_dir=None) -> str:
    return os.path.join(state_dir or config.STATE_DIR, HISTORY_FILE)


# ── inputs ─────────────────────────────────────────────────────────────────────
def load_market():
    """Today's market and panel, cache-first and offline.

    The same six lines run_generation.main() opens with, deliberately not factored
    out of it: that file is the nightly path and this one is the weekly path, and a
    shared helper would let a change made for one silently retune the other. What
    IS shared is the staleness rule itself (run_generation.data_staleness), because
    two different answers to "is this data too old to act on" would be a bug.
    """
    universe = config.import_sibling("universe", config.SIGNAL_LAB)
    wanted = universe.build_universe()[0]
    symbols = datafeed.in_cache(wanted)[:config.UNIVERSE_SIZE]
    market = datafeed.load_market(symbols, start=config.DATA_START)
    arena_features.build_features(market)
    return market, len(wanted)


def latest_generation(state_dir=None) -> int:
    """The last generation with a returns matrix on disk — the cohort PBO reads it,
    and the F2 ledger rows belong to it. NOT population.json's counter: that points
    at the generation about to be evaluated, which has no cohort yet."""
    root = os.path.join(state_dir or config.STATE_DIR, "returns")
    gens = []
    for path in glob.glob(os.path.join(root, "gen_*.npz")):
        try:
            gens.append(int(os.path.basename(path)[4:-4]))
        except ValueError:
            continue
    if gens:
        return max(gens)
    return int(run_generation.load_population(state_dir)[1])


def load_parties(state_dir=None, artifact_dir=None) -> tuple:
    """(candidate entries, incumbent entry or None).

    Candidates are the top config.DEEPEVAL_CANDIDATES hall-of-fame records that
    are not already the champion — the hall is ranked by pre-vault Sharpe, which
    is a SELECTION statistic and exactly why everything below re-measures them.
    The incumbent comes from its immutable artifact when it has one (the registry
    is self-sufficient: a champion can be re-simulated from artifacts/ alone), and
    from the hall of fame only as a fallback.
    """
    hof = evolution.load_hall_of_fame(state_dir)
    champ = registry.champion(state_dir)
    champ_hash = champ[0] if champ else None

    def _entry(rec):
        return {"genome": rec["genome"], "hash": rec["hash"], "op": rec.get("op", ""),
                "parent_hash": rec.get("parent_hash", ""),
                "birth_gen": int(rec.get("birth_gen", 0))}

    candidates = [_entry(r) for r in hof
                  if r["hash"] != champ_hash][:config.DEEPEVAL_CANDIDATES]
    incumbent = None
    if champ_hash:
        incumbent = registry.artifact_entry(champ_hash, artifact_dir)
        if incumbent is None:
            match = [r for r in hof if r["hash"] == champ_hash]
            if not match:
                raise SystemExit(
                    "champion %s has neither an artifact nor a hall-of-fame record: "
                    "it cannot be re-simulated, and gate G1 forbids comparing a "
                    "candidate against a stored number." % champ_hash)
            incumbent = _entry(match[0])
    return candidates, incumbent


# ── workers (module level: joblib has to be able to pickle them) ───────────────
def _simulate(entry, market, cost, label):
    t0 = time.time()                                           # io-boundary
    res = evaluate.full_eval(evolution.entry_genome(entry), market, cost)
    res["n_fits"] = len(res["fit_audit"])
    return entry["hash"], label, res, time.time() - t0         # io-boundary


def _vault_dates(market, n_expected: int) -> pd.DatetimeIndex:
    """The dates behind evaluate.full_eval's `vault_daily_net`.

    full_eval returns the vault series and NOT its dates — the vault leaves that
    function through exactly one key, which is the invariant that makes
    `grep vault_` a complete audit. The dates are recoverable from the calendar:
    an episode's returns are dated at bars 1..n (bar 0 is the reset), so the vault
    rows are the episode dates on or after VAULT_START. Derived, then ASSERTED
    against the length of the series it is labelling — a silent off-by-one here
    would mislabel every vault return by a day.
    """
    episode = market.dates[1:]
    vault = episode[episode >= pd.Timestamp(config.VAULT_START)]
    if len(vault) != n_expected:
        raise RuntimeError("vault calendar (%d bars from %s) does not match the "
                           "simulated vault series (%d) — refusing to label it"
                           % (len(vault), config.VAULT_START, n_expected))
    return vault


# ── the F2 battery for one candidate ───────────────────────────────────────────
def f2_metrics(entry, res, market, cost, generation, index, cohort, incumbent_res=None,
               n_jobs=1, state_dir=None, verbose=True) -> dict:
    """Everything gates.py reads about one candidate, measured here.

    `res` is this run's FRESH F1 result for the genome; `cohort` is the loaded
    returns matrix for the generation (PBO's input); `incumbent_res` is the
    incumbent's fresh F1 result, or None when nothing has ever been promoted.

    Every random draw comes from genome.child_rng(SEED, generation, tag, index) —
    per-candidate streams, so candidate 2's bootstrap does not depend on how many
    numbers candidate 1 happened to draw.
    """
    ghash = entry["hash"]
    genome = evolution.entry_genome(entry)
    net = res["daily_net"]
    out = {"hash": ghash,
           "role": "candidate",
           "identity": tuple(res["identity"]),
           "window": (str(res["dates"][0].date()), str(res["dates"][-1].date())),
           "resimulated": True,
           "sharpe": evaluate.sharpe(net),
           "n_days_prevault": int(res["n_days_prevault"]),
           "family": genome.signal.family}

    # G2 — the ONLY way this project computes a DSR. dsr_trial_sharpes() returns
    # one daily-unit Sharpe per genome ever ledgered, so both the units and N are
    # fixed at the accessor; f1_sharpes() is annualised and F1-only and would
    # silently make G2 unpassable (see its docstring).
    trials = ledger.dsr_trial_sharpes(state_dir)
    dsr = evaluate.deflated_sharpe(net, trials)
    out["dsr"] = dsr.get("dsr")
    out["dsr_n_trials"] = dsr.get("n_trials")
    out["dsr_detail"] = dsr

    # G3 — the vault. One counted access per candidate, before it is read.
    ledger.record_vault_access(ghash, "gate_eval", state_dir)
    vault_net = res["vault_daily_net"]
    vault_dsr = evaluate.deflated_sharpe(vault_net, ledger.vault_trial_sharpes(state_dir))
    out["vault_sharpe"] = evaluate.sharpe(vault_net)
    out["vault_dsr"] = vault_dsr.get("dsr")
    out["vault_trials"] = ledger.vault_trials(state_dir)
    out["vault_days"] = int(np.size(vault_net))

    # G4 — a COHORT statistic, and only for a member of that cohort.
    out["pbo"], pbo_note = cohort_pbo(cohort, ghash)
    out["pbo_in_cohort"] = out["pbo"] is not None
    out["pbo_note"] = pbo_note                   # gates.py prints this when pbo is None
    out["pbo_detail"] = dict(cohort, note=pbo_note)
    if verbose and out["pbo"] is None:
        print("    PBO      : NOT MEASURABLE — %s. G4 fails: an unmeasured gate is "
              "not a passed gate." % pbo_note)

    # G5 — 28 combinatorial purged paths with real refits. The expensive one.
    if verbose:
        print("    CPCV     : %d paths over %d pre-vault blocks, %d workers "
              "(real refits per path)"
              % (math.comb(config.CPCV_GROUPS, config.CPCV_K), config.CPCV_GROUPS,
                 n_jobs), flush=True)
    cpcv = evaluate.cpcv_paths(genome, market, cost, n_jobs=n_jobs, verbose=verbose)
    out["cpcv_frac_positive"] = cpcv["frac_positive"]
    out["cpcv_median_sharpe"] = cpcv["median_sharpe"]
    out["cpcv_n_paths"] = cpcv["n_paths"]
    out["cpcv_detail"] = cpcv

    # G6 — block bootstrap of the pre-vault net Sharpe.
    lo, hi = evaluate.bootstrap_sharpe_ci(
        net, alpha=1.0 - config.GATE_BOOT_CI,
        rng=gn.child_rng(config.SEED, generation, "boot", index))
    out["boot_ci_lo"], out["boot_ci_hi"] = lo, hi
    out["boot_iters"] = config.BOOT_ITERS

    # G8 — crisis windows. Two of the four are vault years, so the slices are read
    # off the pre-vault AND vault series together; that is the same single counted
    # access recorded above, not a second one.
    full_net = np.concatenate([net, vault_net])
    full_dates = res["dates"].append(_vault_dates(market, len(vault_net)))
    out["regime_slices"] = evaluate.regime_slices(full_net, full_dates)
    out["regime_days"] = evaluate.regime_slice_days(full_dates)

    # G10 — ruin. Seeded per candidate, two engines, the worse of the two.
    ruin = evaluate.ruin_mc(net, rng=gn.child_rng(config.SEED, generation, "ruin", index))
    out["p_ruin"] = ruin["p_ruin"]
    out["ruin_detail"] = ruin

    # G9 — only measurable against an incumbent.
    if incumbent_res is not None:
        wins = evaluate.rolling_window_wins(net, res["dates"],
                                            incumbent_res["daily_net"],
                                            incumbent_res["dates"])
        out["rolling_win_frac"] = wins["win_frac"]
        out["rolling_n_windows"] = wins["n_windows"]
        out["rolling_detail"] = wins
    return out


def cohort_pbo(cohort: dict, ghash: str) -> tuple:
    """(PBO for this candidate, why) — None unless the candidate IS in the cohort.

    CSCV asks a question about ONE cohort: if you pick the in-sample best of these
    N configurations, does it stay above the pack out of sample? The candidate has
    to be one of the N. It often is not: candidates come from the ALL-TIME hall of
    fame while the returns matrix belongs to the last evaluated generation, so a
    leader from four generations ago is simply absent from it. Handing that
    candidate the cohort's PBO would be reporting evidence about somebody else's
    selection under its name, so it gets None instead — and gates.py's rule that
    an unmeasurable gate is a FAILED gate does the rest. The honest fix is to
    re-evaluate the genome inside a current cohort, not to borrow a number.
    """
    if cohort.get("pbo") is None:
        return None, cohort.get("note") or "no cohort matrix for this generation"
    if ghash not in (cohort.get("hashes") or ()):
        return None, ("%s is not one of the %d genomes in the generation-%s returns "
                      "matrix, so that cohort's PBO is not evidence about it"
                      % (ghash, len(cohort.get("hashes") or ()),
                         cohort.get("generation", "?")))
    return cohort["pbo"], "cohort of %d genomes, %s splits" % (
        len(cohort["hashes"]), cohort.get("n_splits"))


def print_candidate(m: dict) -> None:
    """The measurements behind the gate table, in the order they were taken."""
    stress_sharpe = m["sharpe_stress"]
    print("    DSR      : %s at n_trials=%s (pre-vault SR %+.2f over %d days, "
          "sr0 %s)" % (m["dsr"], m["dsr_n_trials"], m["sharpe"], m["n_days_prevault"],
                       m["dsr_detail"].get("sr0_threshold")))
    print("    vault    : SR %+.2f over %d days, DSR %s at N=%d accesses"
          % (m["vault_sharpe"], m["vault_days"], m["vault_dsr"], m["vault_trials"]))
    print("    PBO      : %s (%s)" % (m["pbo"], m["pbo_detail"].get("note")))
    c = m["cpcv_detail"]
    print("    CPCV     : %d paths, %.0f%% positive, median SR %+.2f (min %+.2f, "
          "max %+.2f), %d refits total"
          % (c["n_paths"], 100 * c["frac_positive"], c["median_sharpe"],
             min(c["path_sharpes"]), max(c["path_sharpes"]), c["total_fits"]))
    print("    bootstrap: %.0f%% CI [%+.2f, %+.2f] of the annualised net Sharpe"
          % (100 * config.GATE_BOOT_CI, m["boot_ci_lo"], m["boot_ci_hi"]))
    print("    stress   : SR %+.2f at %.0fx costs / %.0fx borrow (%.0f%% of base)"
          % (stress_sharpe, config.GATE_STRESS_MULT, config.GATE_BORROW_STRESS_MULT
             * config.GATE_STRESS_MULT,
             100 * stress_sharpe / m["sharpe"] if m["sharpe"] > 0 else float("nan")))
    print("    regimes  : %s"
          % "  ".join("%s %s (%d d)" % (w[0][:7],
                                        "n/a" if not np.isfinite(v) else "%+.1f%%" % (100 * v),
                                        d)
                      for w, v, d in zip(config.GATE_REGIME_WINDOWS,
                                         m["regime_slices"], m["regime_days"])))
    r = m["ruin_detail"]
    print("    ruin MC  : P(DD>%.0f%% in %dy) garch-t %.3f | bootstrap %.3f -> %.3f (%s)"
          % (100 * config.GATE_RUIN_DD, config.RUIN_MC_YEARS, r["p_ruin_garch_t"],
             r["p_ruin_bootstrap"], r["p_ruin"], r["note"]))
    if "rolling_win_frac" in m:
        print("    vs champ : wins %.0f%% of %d rolling %d-year windows"
              % (100 * m["rolling_win_frac"], m["rolling_n_windows"],
                 config.GATE_ROLLING_WINDOW_YEARS))


# ── persistence ────────────────────────────────────────────────────────────────
def append_history(row: dict, state_dir=None) -> str:
    """Append one decision row, refusing to append into a file of another shape.

    csv.writer appends positionally, so a file written by an older column list
    would take today's fields silently misaligned — every value shifted one column
    left from the first dropped name, and nothing in the file to say so. The
    header is therefore checked before the write and a mismatch raises: this file
    is the record Phase 7's graduation trigger reads, and a misaligned record is
    worse than a missing one.
    """
    path = history_path(state_dir)
    new = not os.path.exists(path)
    if not new:
        with open(path, newline="") as f:
            header = next(csv.reader(f), [])
        if tuple(header) != HISTORY_COLUMNS:
            raise ValueError(
                "%s has columns\n    %s\nbut this version writes\n    %s\n"
                "Appending would misalign every field. Move the old file aside "
                "(it is a record of decisions made under a different schema) or "
                "migrate it; nothing was written."
                % (path, ",".join(header), ",".join(HISTORY_COLUMNS)))
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(HISTORY_COLUMNS)
        w.writerow([row.get(c, "") for c in HISTORY_COLUMNS])
    return path


def read_history(state_dir=None) -> list:
    path = history_path(state_dir)
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _ledger_f2(generation: int, entry, res, identity, state_dir=None) -> dict:
    """Append one F2 trial row. Returns a record of what happened:

        {"status": "wrote" | "already" | "drift", "hash", "generation",
         "prior_identity", "identity", "message"}

    "drift" = this generation already has an F2 row for this genome, recorded
    under DIFFERENT inputs (the data cache refreshed, or a config knob moved). The
    ledger is idempotent on (genome, generation, fidelity), so the older row
    stands and this run's row is not written.

    WHAT THAT COSTS, STATED PRECISELY — the earlier version of this docstring said
    "nothing selects on an F2 row", which is wrong in the way that matters.
    Nothing is BRED from it (run_generation selects on F0/F1 scores) and the gate
    decision is made on this run's fresh simulation, not on the ledger. But
    ledger.dsr_trial_sharpes() and vault_trial_sharpes() return each genome's
    BEST-FIDELITY row — F2 outranks F1 outranks F0 — so a stale F2 row IS the
    value those accessors hand to G2 and G3, and it moves `var(all_sharpes)`,
    which is what sets the sr0 threshold both gates are measured against. The
    direction is uncontrolled and it is not always conservative: understating the
    dispersion LOWERS sr0 and therefore EASES G2. The review measured the
    sensitivity on this ledger — halving 2 of the 21 trial values lifts a
    candidate's DSR from 0.027 to 0.046 — so today the effect is bounded far below
    the 0.95 threshold, but it is a real gate input going stale, not a cosmetic
    record-keeping wrinkle. Hence: printed at the point it happens, stored in the
    artifact's metrics beside dsr_detail, and carried as a column in
    state/deepeval_history.csv, so an artifact reader sees it without the job log.

    WHY NOT RAISE, AS run_generation DOES. There a dropped duplicate is fatal —
    the population was bred from numbers that would then appear nowhere on disk
    (IdentityDrift's docstring). Here, aborting would mean a deep eval can never
    run twice on one generation, which is exactly what happens when the data
    refreshes between two Saturdays without a new generation in between: the
    weekly job would simply stop working, having measured everything correctly.

    THE ALTERNATIVE THAT WAS AVAILABLE AND DECLINED, so Phase 6+ does not have to
    rediscover it: keep-newer-by-append. Add the identity triple to record_trial's
    idempotency key, and the newer row appends instead of being dropped; the
    accessors need no change, because their stable sort on
    ["genome_hash", "_rank", "generation"] + tail(1) already prefers the last row
    written for a genome, and n_trials counts DISTINCT genome hashes so it stays
    put. That is the better answer to the staleness above. It was declined here
    because record_trial's key is Phase-4 idempotency semantics that run_generation
    depends on for resume and for its IdentityDrift guard, and widening it is a
    change to that contract rather than to this file. It belongs in a task that
    can re-run the Phase-4 determinism and resume proofs.
    """
    wrote = ledger.record_trial(generation, evolution.entry_genome(entry), "F2",
                                res["score"], res["sharpe_prevault"],
                                res["n_days_prevault"], identity[0], identity[1],
                                parent_hash=evolution.parent_field(entry["parent_hash"]),
                                birth_gen=entry["birth_gen"], state_dir=state_dir)
    out = {"hash": entry["hash"], "generation": int(generation),
           "identity": tuple(str(v) for v in identity), "prior_identity": None,
           "message": ""}
    if wrote:
        return dict(out, status="wrote")
    prior = run_generation._ledger_row(generation, entry["hash"], "F2", state_dir)  # noqa: SLF001
    if prior is None:
        # record_trial said "already there" and the file says otherwise: the ledger
        # changed under this run (a concurrent writer, a hand edit, a truncated
        # file). Unlike a drift, THE ROW IS SIMPLY LOST — there is no older row
        # standing in for it — so this is an error, exactly as run_generation
        # treats the same case.
        raise run_generation.IdentityDrift(
            "the trial ledger reports an existing F2 row for %s at generation %d "
            "but no such row can be read back. The ledger changed under this run; "
            "this evaluation would be recorded nowhere at all. Nothing was "
            "promoted." % (entry["hash"], generation))
    prior_id = run_generation._row_identity(prior)                       # noqa: SLF001
    if prior_id == tuple(identity):
        return dict(out, status="already")
    return dict(out, status="drift", prior_identity=prior_id,
                message=("%s: generation %d already has an F2 row from an earlier "
                         "vintage (data %s | panel %s | config %s); this run computed "
                         "it under (data %s | panel %s | config %s). The older row "
                         "stands and still feeds the DSR trial-Sharpe dispersion."
                         % ((entry["hash"], generation) + prior_id
                            + tuple(str(v) for v in identity))))


def store(entry, res, metrics, gate_report, decisions=None, artifact_dir=None,
          state_dir=None) -> str:
    """One immutable artifact for one party.

    An artifact carries the vault segment of the return series, so STORING one is
    itself a vault touch and gets its own logged access. The candidates already
    have a "gate_eval" row from f2_metrics — this adds a row, not a new genome, so
    it cannot change vault_trials — but the incumbent is only ever re-simulated,
    never batteried, and its vault rows would otherwise reach disk with nothing in
    state/vault_access.csv to show it. `grep record_vault_access` has to be a
    complete answer to "who touched the vault", including the writer.
    """
    if res.get("vault_dates") is not None and len(res["vault_dates"]):
        ledger.record_vault_access(entry["hash"], "artifact_store", state_dir)
    payload = dict(metrics or {})
    payload.pop("hash", None)
    if gate_report is not None:
        payload["gate_report"] = gate_report
    return registry.store_artifact(entry, res, payload or None, decisions,
                                   artifact_dir=artifact_dir)


# ── alerting ───────────────────────────────────────────────────────────────────
def champ_before_hash(state_dir=None) -> str:
    champ = registry.champion(state_dir)
    return champ[0] if champ else ""


def _alert(generation, status: str, title: str, body: str, send: bool) -> bool:
    """One alert, suppressed unless (champion, generation, status) has moved.

    Same key as the generation job's, so the two anti-spam memories live side by
    side in state/alert_state.json under their own kinds. A weekly refusal that
    refuses the same candidates for the same reasons on the same generation is
    not news the second time; a promotion always is, because the champion moved.
    """
    state = {"champion": champ_before_hash(),
             "generation": None if generation is None else int(generation),
             "status": status}
    return alerts_arena.send_transition("deepeval", state, title, body, dry_run=not send)


# ── the run ────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(                                   # io-boundary
        description="weekly deep evaluation: F2 battery, ten gates, promote or refuse")
    ap.add_argument("--dry", action="store_true",
                    help="measure and print, but write no DECISION: no promotion, no "
                         "artifacts, no deep-eval history row. The trial ledger and "
                         "the vault-access log are still appended — the evaluation "
                         "happened, and an uncounted look would make every DSR "
                         "optimistic")
    ap.add_argument("--jobs", type=int, default=config.N_JOBS, help="worker processes")
    ap.add_argument("--budget-min", type=float,
                    default=float(config.DEEPEVAL_TIME_BUDGET_MIN),
                    help="wall-clock budget in minutes (default "
                         "config.DEEPEVAL_TIME_BUDGET_MIN); the run stops between "
                         "candidates and refuses rather than half-evaluating one")
    ap.add_argument("--rollback", metavar="HASH", default=None,
                    help="repoint champion.json at a prior artifact hash and exit; "
                         "appends a rollback row to champion_history.csv")
    ap.add_argument("--reason", default="operator rollback",
                    help="the note recorded beside a --rollback row")
    ap.add_argument("--send", action="store_true",
                    help="deliver the decision alert; without it the exact text is "
                         "printed and nothing is sent")
    ap.add_argument("--no-report", action="store_true",
                    help="skip building output/report_gen<N>.md at the end")
    args = ap.parse_args()

    if args.rollback:
        return do_rollback(args.rollback, args.reason)

    t_start = time.time()                                           # io-boundary
    deadline = t_start + args.budget_min * 60.0                     # io-boundary
    print("arena weekly deep evaluation%s" % ("  (DRY RUN)" if args.dry else ""))

    # Before any gate reads the ledger: a union merge (see .gitattributes) can
    # leave a row twice, and G2/G3 are deflated by exactly that count.
    ledger.dedup_ledger()

    market, n_wanted = load_market()
    cache_age, bar_age = run_generation.data_staleness(market)
    if max(cache_age, bar_age) > config.MAX_DATA_STALENESS_DAYS:
        print("  ABORT: data is stale — cache %.1f days old, last bar %s (%.1f days "
              "old); limit is %d. A promotion decided on a market that no longer "
              "exists is worse than no decision."
              % (cache_age, market.dates[-1].date(), bar_age,
                 config.MAX_DATA_STALENESS_DAYS))
        title, body = alerts_arena.data_stale_summary(
            cache_age, bar_age, market.dates[-1].date(),
            config.MAX_DATA_STALENESS_DAYS, job="deep eval")
        _alert(latest_generation(), "stale-data", title, body, args.send)
        return 1

    # getattr, not market.panel_hash: features.py attaches it ad hoc, and a None
    # here has to reach gate G1 as a G1 FAILURE rather than as an exception.
    identity = (market.data_hash, getattr(market, "panel_hash", None),
                config.config_hash())
    generation = latest_generation()
    print("  data      : %d/%d symbols cached, %s -> %s (%d bars), cache %.1fd old"
          % (len(market.symbols), n_wanted, market.dates[0].date(),
             market.dates[-1].date(), len(market), cache_age))
    print("  identity  : data %s | panel %s | config %s | %s"
          % (identity + (ledger.platform_tag(),)))
    print("  generation: %d (the last with a returns matrix — the PBO cohort)" % generation)

    candidates, incumbent = load_parties()
    if not candidates:
        print("  nothing to evaluate: the hall of fame holds no genome that is not "
              "already the champion. Run a generation first.")
        title, body = alerts_arena.deepeval_summary(
            generation, [], champion_before=champ_before_hash(), status="nothing-to-evaluate",
            platform=ledger.platform_tag(),
            detail="The hall of fame holds no genome that is not already the "
                   "champion, so there was nothing to put through the gates.")
        _alert(generation, "nothing-to-evaluate", title, body, args.send)
        return 0
    champ_before = registry.champion()
    print("  parties   : %d candidate(s) from the hall of fame%s"
          % (len(candidates),
             ", champion %s re-simulated fresh" % champ_before[0] if champ_before
             else ", no champion yet (G9 is skipped, margin 0)"))
    for i, e in enumerate(candidates):
        print("      cand %d  %s  %s" % (i + 1, e["hash"],
                                         evolution.entry_genome(e).describe()[:78]))

    # ── 1. fresh F1 for every party, base costs and stressed ──────────────────
    base_cost = CostModel()
    # stress_mult scales EVERY friction including borrow, so a borrow_annual
    # tripled here is 6x the base borrow charge after the multiplier — DESIGN's
    # "2x cost stress + 3x borrow" read literally, and the harsher reading of it.
    stress_cost = CostModel(borrow_annual=config.BORROW_ANNUAL * config.GATE_BORROW_STRESS_MULT,
                            stress_mult=config.GATE_STRESS_MULT)
    parties = list(candidates) + ([incumbent] if incumbent is not None else [])
    # The incumbent is re-simulated at BASE costs only: no gate reads a stressed
    # incumbent (G7 asks whether the CANDIDATE survives doubled frictions), and an
    # episode nothing reads is an episode nobody should pay for.
    jobs = [(e, base_cost, "base") for e in parties] + \
           [(e, stress_cost, "stress") for e in candidates]
    print("\n  F1 fresh  : %d parties at base costs + %d candidates at %.0fx costs = "
          "%d full episodes on %d workers"
          % (len(parties), len(candidates), config.GATE_STRESS_MULT, len(jobs),
             min(args.jobs, len(jobs))))

    fresh: dict = {}
    par = Parallel(n_jobs=max(1, min(args.jobs, len(jobs))), batch_size=1,
                   return_as="generator")
    results = par(delayed(_simulate)(e, market, c, label) for e, c, label in jobs)
    try:
        for ghash, label, res, secs in results:
            fresh[(ghash, label)] = res
            print("      %-6s %s  SR %+.2f over %d days  %d refits  %6.1fs"
                  % (label, ghash, res["sharpe_prevault"], res["n_days_prevault"],
                     res["n_fits"], secs), flush=True)
    finally:
        del results

    counts = {"wrote": 0, "already": 0, "drift": 0}
    drifted: dict = {}                 # hash -> the drift record, stored and reported
    for entry in parties:
        res = fresh[(entry["hash"], "base")]
        res["identity"] = identity
        res["generation"] = generation
        res["vault_dates"] = _vault_dates(market, len(res["vault_daily_net"]))
        rec = _ledger_f2(generation, entry, res, identity)
        counts[rec["status"]] += 1
        if rec["status"] == "drift":
            drifted[entry["hash"]] = rec
            print("      ledger  : KEPT THE OLDER ROW — %s" % rec["message"])
    print("      ledger  : %d F2 row(s) appended, %d already on record, %d kept from "
          "an earlier vintage (n_trials now %d)"
          % (counts["wrote"], counts["already"], counts["drift"], ledger.n_trials()))
    if drifted:
        print("                Those genomes' BEST-FIDELITY rows are the older "
              "vintage's, so that is the value dsr_trial_sharpes() feeds to G2 and "
              "vault_trial_sharpes() to G3 — it moves the trial-Sharpe dispersion "
              "that sets sr0, in an uncontrolled direction (understated dispersion "
              "EASES G2). Recorded in each artifact's metrics and in the "
              "deepeval_history ledger_drift column, not just here.")

    # ── 2. the cohort PBO, once ───────────────────────────────────────────────
    cohort = load_cohort(generation, identity)

    # ── 3. the battery, per candidate ─────────────────────────────────────────
    inc_res = fresh[(incumbent["hash"], "base")] if incumbent is not None else None
    inc_metrics = None
    if incumbent is not None:
        inc_metrics = {"hash": incumbent["hash"],
                       # No F2 battery is run on the incumbent: the gates ask
                       # whether the CHALLENGER is real, and all the incumbent has
                       # to supply is a like-for-like Sharpe (G1) and a series to
                       # be beaten over rolling windows (G9).
                       "role": "incumbent (re-simulated for G1 and G9)",
                       "identity": identity,
                       "window": (str(inc_res["dates"][0].date()),
                                  str(inc_res["dates"][-1].date())),
                       "resimulated": True,
                       "sharpe": evaluate.sharpe(inc_res["daily_net"]),
                       "ledger_drift": drifted.get(incumbent["hash"])}

    measured, reports, complete = [], {}, True
    for i, entry in enumerate(candidates):
        if time.time() > deadline:                                  # io-boundary
            complete = False
            print("\n  budget spent (%.0f min) before candidate %d — stopping. A "
                  "half-measured candidate is not evidence, and nothing is promoted "
                  "on a partial battery." % (args.budget_min, i + 1))
            break
        print("\n  ── candidate %d/%d  %s  %s"
              % (i + 1, len(candidates), entry["hash"],
                 evolution.entry_genome(entry).describe()[:60]))
        res = fresh[(entry["hash"], "base")]
        m = f2_metrics(entry, res, market, base_cost, generation, i, cohort,
                       incumbent_res=inc_res, n_jobs=args.jobs)
        m["sharpe_stress"] = evaluate.sharpe(fresh[(entry["hash"], "stress")]["daily_net"])
        # Stored beside dsr_detail, because it is a caveat ON the DSR: this
        # genome's trial-Sharpe contribution came from an older vintage's row.
        m["ledger_drift"] = drifted.get(entry["hash"])
        print_candidate(m)
        report = gates.evaluate_gates(m, inc_metrics)
        print()
        for line in gates.gate_table(report):
            print(line)
        # The two counts this candidate's DSRs were deflated by. They are part of
        # the artifact's eval key (registry.eval_key) because they are the one
        # thing in the record that a re-run can honestly change: another deep eval
        # is another look at the vault, so the same candidate scores a slightly
        # lower vault DSR next time, and that belongs beside the first result
        # rather than on top of it.
        res["eval_tag"] = "trials%s.vault%s" % (m["dsr_n_trials"], m["vault_trials"])
        measured.append((entry, res, m))
        reports[entry["hash"]] = report

    # ── 4. the decision ───────────────────────────────────────────────────────
    # Only one genome can be the champion, so if both candidates clear all ten
    # gates the better pre-vault Sharpe takes it, with the hash breaking a tie:
    # the decision is then a function of the numbers, never of dict order.
    passing = [row for row in measured if reports[row[0]["hash"]]["all_pass"]]
    winner = max(passing, key=lambda row: (row[2]["sharpe"], row[0]["hash"])) \
        if passing else None
    champ_after = champ_before[0] if champ_before else ""

    print("\n  ── decision %s" % ("─" * 62))
    if winner is None:
        print("  REFUSED: no candidate passed all ten gates. The champion is "
              "unchanged (%s)." % (champ_before[0] if champ_before else "still none"))
        for e, _res, _m in measured:
            print("      %s failed %s" % (e["hash"], ", ".join(reports[e["hash"]]["failed"])))
    else:
        w_entry, w_res, w_m = winner
        print("  PROMOTE: %s passed all ten gates (pre-vault SR %+.2f, vault SR %+.2f, "
              "DSR %s)" % (w_entry["hash"], w_m["sharpe"], w_m["vault_sharpe"], w_m["dsr"]))
        if not args.dry:
            champ_after = w_entry["hash"]

    # ── 5. persistence ────────────────────────────────────────────────────────
    if args.dry:
        print("\n  --dry: no artifacts, no champion move, and NO deep-eval history "
              "row (that file holds decisions; this run made none). The %d F2 ledger "
              "rows and %d vault access(es) it made WERE written — the evaluation "
              "happened." % (len(parties), len(measured)))
    else:
        decisions_for = winner[0]["hash"] if winner else (
            measured[0][0]["hash"] if measured else None)
        for e, res, m in measured:
            dec = None
            if e["hash"] == decisions_for and time.time() < deadline:   # io-boundary
                dec = tier_b_decisions(e, market, base_cost, res)
            store(e, res, m, reports[e["hash"]], dec)
        if incumbent is not None:
            store(incumbent, inc_res, inc_metrics, None)
        print("\n  artifacts : %d stored under %s"
              % (len(measured) + (1 if incumbent is not None else 0),
                 os.path.relpath(registry.genomes_dir(), config.ROOT)))
        if winner is not None:
            meta = registry.promote(winner[0]["hash"], generation,
                                    reports[winner[0]["hash"]],
                                    note="deep eval generation %d" % generation)
            print("  champion  : %s -> %s (history row appended)"
                  % (meta["previous_hash"] or "(none)", meta["hash"]))
        path = append_history({
            "generation": generation,
            "champion_hash_before": champ_before[0] if champ_before else "",
            "champion_hash_after": champ_after,
            "promoted": int(winner is not None),
            "n_candidates": len(measured),
            "candidates": ";".join(
                "%s:%d:%s" % (e["hash"], int(reports[e["hash"]]["all_pass"]),
                              "+".join(reports[e["hash"]]["failed"]) or "none")
                for e, _r, _m in measured),
            "gates_failed": "+".join(reports[measured[0][0]["hash"]]["failed"])
            if measured else "",
            # hash:prior_data|prior_panel|prior_config per drifted party, so the
            # caveat on G2/G3's dispersion is in the decision record itself.
            "ledger_drift": ";".join(
                "%s:%s" % (h, "|".join(rec["prior_identity"]))
                for h, rec in sorted(drifted.items())),
            "complete": int(complete),
            "data_hash": identity[0], "panel_hash": identity[1] or "",
            "config_hash": identity[2], "platform": ledger.platform_tag()},
        )
        print("  history   : appended to %s" % os.path.relpath(path, config.ROOT))

    elapsed = (time.time() - t_start) / 60.0                        # io-boundary
    print("\n  wall clock: %.1f min (%d workers) — CPCV dominates: %d paths x %d "
          "episodes per candidate" % (elapsed, args.jobs,
                                      measured[0][2]["cpcv_n_paths"] if measured else 0,
                                      config.CPCV_K))

    # ── 6. the report, then the alert ─────────────────────────────────────────
    # The report is built from what is now on disk, so it has to come after the
    # artifacts and the history row — and before the alert, so a failure to draw a
    # chart cannot swallow the decision notification.
    if not args.no_report:
        try:
            path = arena_reports.build_report(generation)
            print("  report    : %s" % os.path.relpath(path, config.ROOT))
        except Exception as exc:                    # a report is a rendering of a
            # decision that has already been made and persisted; failing to draw it
            # must not turn a completed deep eval into a failed job.
            print("  report    : FAILED to build (%s: %s) — the decision above still "
                  "stands and is on disk" % (type(exc).__name__, str(exc)[:160]))

    top = measured[0][2] if measured else {}
    title, body = alerts_arena.deepeval_summary(
        generation=generation,
        candidates=[(e["hash"], bool(reports[e["hash"]]["all_pass"]),
                     list(reports[e["hash"]]["failed"]))
                    for e, _r, _m in measured] if measured else [],
        promoted=(winner[0]["hash"] if winner and not args.dry else None),
        champion_before=champ_before[0] if champ_before else None,
        dsr=top.get("dsr"), n_trials=top.get("dsr_n_trials"),
        vault_trials=top.get("vault_trials"),
        status=("promoted" if winner else ("incomplete" if not complete else "refused")),
        platform=ledger.platform_tag(),
        detail=("DRY RUN: no artifacts, no champion move, no history row — but the "
                "F2 ledger rows and vault accesses WERE written, because the "
                "evaluation happened." if args.dry else ""))
    status = ("promoted:%s" % winner[0]["hash"] if winner and not args.dry
              else ("incomplete" if not complete
                    else "refused:%s" % "+".join(
                        sorted({g for e, _r, _m in measured
                                for g in reports[e["hash"]]["failed"]}))))
    _alert(generation, status, title, body, args.send)

    print_honesty(complete)
    return 0


def load_cohort(generation: int, identity) -> dict:
    """CSCV PBO over the generation's pre-vault returns matrix (gate G4).

    The matrix is an artifact of the generation that produced it, so its stored
    identity can legitimately differ from this run's — a config knob added since,
    or a data refresh. That does NOT invalidate the statistic (PBO asks whether
    picking the in-sample best of that cohort survived out of sample, and those
    are the numbers that cohort was selected on), but it is printed rather than
    swallowed, because a reader comparing PBO against the other gates is entitled
    to know the cohort is not this run's simulation.

    The cohort's genome hashes come back with it: cohort_pbo() will only give this
    number to a candidate that is actually in the matrix.
    """
    try:
        mat = ledger.load_returns_matrix(generation)
    except FileNotFoundError:
        print("\n  PBO       : no returns matrix for generation %d — G4 cannot be "
              "measured and will fail" % generation)
        return {"pbo": None, "n_splits": 0, "hashes": [], "generation": generation,
                "note": "no returns matrix for generation %d" % generation}
    R = np.asarray(mat["daily_net"], dtype=np.float64)
    out = evaluate.pbo_cscv(R, S=config.PBO_SPLITS)
    out["hashes"] = list(mat["hashes"])
    out["generation"] = generation
    stored = (mat["data_hash"], mat["panel_hash"], mat["config_hash"])
    drift = [n for n, a, b in zip(("data", "panel", "config"), stored, identity)
             if a and a != str(b)]
    out["cohort_identity"] = stored
    out["cohort_drift"] = drift
    print("\n  PBO cohort: generation %d matrix, %d days x %d genomes -> PBO %s "
          "over %d splits" % (generation, R.shape[0], R.shape[1], out["pbo"],
                              out["n_splits"]))
    if drift:
        print("      NOTE: the cohort was written under a different %s hash than "
              "this run's. PBO still describes that cohort's own selection; every "
              "other gate below is measured on this run's fresh simulation."
              % " and ".join(drift))
    return out


def tier_b_decisions(entry, market, cost, res) -> list:
    """Re-run the winner's episode with the decision log on (docs/DESIGN.md Tier B).

    A second episode rather than logging every party's: the log is position-level
    and runs to tens of thousands of rows, and only the genome the report is about
    needs one. It covers the WHOLE episode, including the warm-up bars before
    scoring starts — the artifact's return series is the scored window, and the
    log is what produced it, warm-up included. The replay is CHECKED against the
    metrics episode over the scored window — same
    seed, same market, same costs, so the pre-vault series must come back
    identical — because a decision log that does not explain the numbers stored
    beside it is worse than no log at all.
    """
    log: list = []
    episode = StrategyAgent(evolution.entry_genome(entry), market, cost).run_episode(
        decision_log=log)
    first = int(res["first_active"])
    vault_i = int(episode["dates"].searchsorted(pd.Timestamp(config.VAULT_START)))
    replayed = episode["daily_net"][first:vault_i]
    if replayed.shape != res["daily_net"].shape or not np.allclose(replayed,
                                                                   res["daily_net"]):
        raise RuntimeError(
            "the decision-logged replay of %s does not reproduce its own scored "
            "series (%d vs %d bars) — the log would not explain the metrics stored "
            "with it" % (entry["hash"], replayed.size, res["daily_net"].size))
    return log


def do_rollback(ghash: str, reason: str) -> int:
    """Repoint the champion at a prior artifact. DESIGN: "Rollback = repoint to any
    prior hash" — the same mechanics as a promotion, logged the same way, and the
    reason is recorded verbatim."""
    print("arena champion rollback")
    art = registry.load_artifact(ghash)
    if art is None:
        print("  ABORT: no artifact for %s. A rollback may only point at a genome "
              "the registry can still produce the evidence for." % ghash)
        return 1
    current = registry.champion()
    print("  current   : %s" % (current[0] if current else "(none)"))
    # The row records the generation the rollback HAPPENED at, not the one the old
    # champion was promoted at: arena's clock is the generation counter, and a
    # history row nobody can place in time is not an audit trail.
    meta = registry.rollback(ghash, reason, generation=latest_generation())
    print("  rolled to : %s (%s)" % (meta["hash"], reason))
    print("  history   : %d rows in %s"
          % (len(registry.champion_history()),
             os.path.relpath(registry.history_path(), config.ROOT)))
    for row in registry.champion_history()[-3:]:
        print("      gen %-4s %-12s -> %-12s %-13s %s"
              % (row["generation"], row["prev_hash"] or "(none)", row["new_hash"],
                 row["reason"], row["note"]))
    print("\n  The pointer moved; nothing else did. Every artifact is immutable, the")
    print("  trial ledger is unchanged, and the genome this now points at is exactly")
    print("  the one whose evidence is stored beside it.")
    return 0


def print_honesty(complete: bool) -> None:
    if not complete:
        print("\n  This run did NOT evaluate every candidate (time budget). The row it")
        print("  wrote says so, and Phase 7's graduation trigger counts only complete runs.")
    print("\n  Ten gates passed is EVIDENCE, NOT A GUARANTEE. DSR and PBO correct for")
    print("  the searches this ledger knows about; evolutionary trials are correlated,")
    print("  so the correction is optimistic by construction. The vault is six years")
    print("  of held-out data, not a future. Survivorship: the universe is today's")
    print("  S&P membership, so long results flatter and short results understate.")
    print("  A backtested edge is a claim about the past — not a guarantee, and not")
    print("  financial advice.")


if __name__ == "__main__":
    raise SystemExit(main())
