"""
The trial ledger: every genome arena ever evaluated, appended and never rewritten.

    record_trial(...)          one row per (genome, generation, fidelity)
    n_trials()                 distinct genomes ever evaluated  -> DSR's N
    dsr_trial_sharpes()        one DAILY Sharpe per genome      -> DSR's all_sharpes
    trial_sr_std()             empirical spread of F1 Sharpes   -> reports
    record_vault_access(...)   every look at post-2020 data, counted
    write_returns_matrix(...)  the Tier-A (days x genomes) artifact DSR/PBO read

THIS FILE IS THE ANSWER TO SIGNAL_LAB'S TWO WORST FAILURES (docs/DESIGN.md "Why
shaped this way"): a censored ledger that recorded only the alerts it liked, and a
deflated-Sharpe call with `n_trials=50` hardcoded next to a search that had tried
thousands. So:

  • EVERY evaluation is recorded, at its fidelity, whatever the result. A screen
    counts as a trial — it exerted selection, and a multiple-testing correction
    that ignores the trials you threw away is not a correction.
  • n_trials() is READ FROM THIS FILE. No caller may pass a count.
  • Rows are append-only. record_trial() is idempotent on
    (genome_hash, generation, fidelity) so a killed run that resumes re-appends
    nothing, and a lost race between the Mac booster and the cloud runner wastes
    compute rather than corrupting the count.

WHAT MAKES TWO ROWS COMPARABLE (gate G1's raw material) is carried on every row:
  data_hash    the equity bars (symbols, calendar ends, per-symbol bar count/last
               close) — datafeed's digest.
  panel_hash   data AND features AND the six macro series, features.py's digest.
               THIS is the like-for-like key: data_hash never sees macro, so two
               rows can agree on it and still have been scored on different
               inputs.
  config_hash  every simulation setting arena declares (config.config_hash).
  platform     machine + OS. Determinism is guaranteed per-platform only — BLAS
               differences can flip low-order bits between Apple Silicon and the
               x86 runner — so a cross-platform comparison must be visible as one.

The counterpart file, `state/vault_access.csv`, is the honesty tax on the vault:
Phase 5's gates are the only code allowed to read post-VAULT_START days, and each
read appends a row here so vault-DSR can be deflated by how often the vault has
been consulted. It is deliberately NOT idempotent — asking twice IS two looks.
"""
from __future__ import annotations

import csv
import os
import platform as _platform
import sys

import numpy as np
import pandas as pd

import config                       # FIRST: puts the siblings on sys.path

LEDGER_COLUMNS = ("generation", "genome_hash", "fidelity", "family", "horizon",
                  "n_features", "score", "sharpe_prevault", "n_days", "data_hash",
                  "panel_hash", "config_hash", "platform", "parent_hash", "birth_gen")
VAULT_COLUMNS = ("genome_hash", "reason")
FIDELITIES = ("F0", "F1", "F2")

# (genome_hash, generation, fidelity) already on disk, per ledger path. Read once
# per process; every append updates it, so a 64-genome generation costs one read.
_KEY_CACHE: dict = {}


def platform_tag() -> str:
    """machine + OS, e.g. "arm64darwin". An I/O boundary (it describes the host,
    not the simulation) and the reason a ledger row can be trusted to compare
    against another one."""
    return "%s%s" % (_platform.machine(), sys.platform)


# ── paths (resolved per call, so a test can point config.STATE_DIR at a tmpdir) ─
def ledger_path(state_dir=None) -> str:
    return os.path.join(state_dir or config.STATE_DIR, "trial_ledger.csv")


def vault_path(state_dir=None) -> str:
    return os.path.join(state_dir or config.STATE_DIR, "vault_access.csv")


def returns_path(generation: int, state_dir=None) -> str:
    return os.path.join(state_dir or config.STATE_DIR, "returns",
                        "gen_%04d.npz" % int(generation))


def _append(path: str, columns, row) -> None:
    """Append one row, writing the header if the file is new. Nothing here ever
    opens a file for anything but 'a'."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(columns)
        w.writerow(row)


def read_ledger(state_dir=None) -> pd.DataFrame:
    """The whole ledger, or an empty frame with the right columns."""
    path = ledger_path(state_dir)
    if not os.path.exists(path):
        return pd.DataFrame(columns=list(LEDGER_COLUMNS))
    return pd.read_csv(path, dtype={"genome_hash": str, "parent_hash": str,
                                    "data_hash": str, "panel_hash": str,
                                    "config_hash": str})


def _keys(path: str) -> set:
    if path not in _KEY_CACHE:
        keys = set()
        if os.path.exists(path):
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    keys.add((row["genome_hash"], int(row["generation"]), row["fidelity"]))
        _KEY_CACHE[path] = keys
    return _KEY_CACHE[path]


def forget_cache() -> None:
    """Drop the in-process key index (a test that swaps STATE_DIR calls this)."""
    _KEY_CACHE.clear()


# ── trials ─────────────────────────────────────────────────────────────────────
def record_trial(generation: int, genome, fidelity: str, score: float,
                 sharpe_prevault: float, n_days: int, data_hash: str,
                 panel_hash: str, parent_hash: str = "", birth_gen=None,
                 state_dir=None, config_hash: str = None) -> bool:
    """Append one evaluation. True if written, False if it was already there.

    `sharpe_prevault` is the pre-vault net Sharpe for an F1/F2 row; for an F0 row
    it is the mean era Sharpe (the screen's own number) — trial_sr_std() only ever
    reads F1 rows, so the two never get mixed into one dispersion estimate.

    FALSE IS NOT ALWAYS BENIGN, AND THE CALLER OWNS THE DIFFERENCE. Idempotency is
    what makes a resumed or re-raced run harmless, but the key is only (hash,
    generation, fidelity) — it does NOT include the identity columns. So a row
    that was written under one data_hash blocks a row for the same genome and
    generation computed under another, and the ledger keeps the older provenance
    while the newer numbers are the ones in memory. Re-recording work that was
    already recorded is fine; discarding a freshly computed evaluation is not, and
    run_generation raises on exactly that case rather than letting the ledger's
    identity columns quietly describe a different run than the one that happened.

    `config_hash` defaults to the settings this process is running under. It is a
    parameter only so an orphaned checkpoint can be ledgered under the settings it
    was actually computed with (run_generation.sweep_orphan_checkpoints).
    """
    if fidelity not in FIDELITIES:
        raise ValueError("fidelity must be one of %s, got %r" % (FIDELITIES, fidelity))
    path = ledger_path(state_dir)
    ghash = genome.hash() if hasattr(genome, "hash") else str(genome)
    key = (ghash, int(generation), fidelity)
    if key in _keys(path):
        return False

    sig = genome.signal
    _append(path, LEDGER_COLUMNS, [
        int(generation), ghash, fidelity, sig.family, int(sig.horizon),
        len(sig.features), repr(float(score)), repr(float(sharpe_prevault)),
        int(n_days), data_hash, panel_hash,
        config_hash or config.config_hash(), platform_tag(),
        parent_hash or "", "" if birth_gen is None else int(birth_gen)])
    _keys(path).add(key)
    return True


def n_trials(state_dir=None) -> int:
    """Distinct genomes ever evaluated at ANY fidelity — the N that deflates every
    Sharpe this project reports. Screens included: a genome that was screened and
    dropped still consumed a look at the data."""
    df = read_ledger(state_dir)
    return int(df["genome_hash"].nunique()) if len(df) else 0


def f1_sharpes(state_dir=None) -> np.ndarray:
    """Every F1 pre-vault Sharpe on record, ANNUALISED. **NOT DSR-ready.**

    This is the reporting series: the numbers as stored, in the units a human
    reads. Passing it to deflated_sharpe as `all_sharpes` is wrong twice over —
    the units are annualised where that function works in daily ones, and the
    count is F1 rows where the multiple-testing N is every genome ever tried.
    Use dsr_trial_sharpes() for that; this one is for reports and for
    trial_sr_std's empirical dispersion.
    """
    df = read_ledger(state_dir)
    if not len(df):
        return np.array([], dtype=np.float64)
    vals = pd.to_numeric(df.loc[df["fidelity"] == "F1", "sharpe_prevault"],
                         errors="coerce").to_numpy(dtype=np.float64)
    return vals[np.isfinite(vals)]


def dsr_trial_sharpes(state_dir=None) -> np.ndarray:
    """deflated_sharpe's `all_sharpes`: one DAILY Sharpe per genome ever ledgered.

    Two couplings live here, and both are the reason this exists as its own
    function rather than as something a caller assembles at the call site.

    UNITS. deflated_sharpe computes `sr = mean/std` of a DAILY return series and
    compares it against a threshold built from `var(all_sharpes)`. This ledger
    stores annualised Sharpes. Hand it the stored numbers and the threshold comes
    out sqrt(TRADING_DAYS_YEAR) = 15.9x too wide, sr0 becomes unreachable, and
    gate G2 rejects every candidate forever while looking perfectly healthy. So
    the conversion happens here, once, where the units are known.

    N. The multiple-testing count must be every genome the search ever LOOKED at,
    not every genome that survived to a full evaluation. A genome screened at F0
    and dropped still exerted selection — it is why the survivor looked good — and
    a correction that ignores the trials you threw away is not a correction. So
    this returns exactly one value per distinct genome_hash, and
    `len(dsr_trial_sharpes()) == n_trials()` holds by construction (verify #8).

    Each genome contributes its BEST-FIDELITY row (F2 > F1 > F0, latest
    generation on a tie): the most informative estimate on record for that
    genome. A row whose Sharpe is unreadable contributes 0.0 rather than being
    dropped — "no evidence" is what evaluate.sharpe already returns for an
    unscoreable episode, and dropping it would quietly shrink N.
    """
    df = read_ledger(state_dir)
    if not len(df):
        return np.array([], dtype=np.float64)
    order = df.assign(_rank=df["fidelity"].map({f: i for i, f in enumerate(FIDELITIES)}))
    order = order.sort_values(["genome_hash", "_rank", "generation"], kind="mergesort")
    best = order.groupby("genome_hash", sort=True).tail(1)
    vals = pd.to_numeric(best["sharpe_prevault"], errors="coerce").to_numpy(dtype=np.float64)
    vals = np.where(np.isfinite(vals), vals, 0.0)
    return vals / np.sqrt(config.TRADING_DAYS_YEAR)


def trial_sr_std(state_dir=None, daily: bool = False) -> float:
    """Empirical spread of the trial Sharpes — futures_bot's approach, never a
    hardcoded constant.

    Under config.TRIAL_SR_STD_MIN_ROWS F1 rows the sample std is itself noise, so
    it falls back to 1/sqrt(n_trials): the standard error of a Sharpe estimated
    from a single trial's worth of evidence, which shrinks as the search grows and
    so keeps the correction conservative early on rather than confidently wrong.

    UNITS — THE TRAP THAT WOULD SILENTLY BREAK GATE G2. This ledger stores
    ANNUALISED Sharpes, because that is what evaluate.sharpe returns and what a
    human reads. deflated_sharpe() works in DAILY Sharpe units: it computes
    mean/std of a daily return series and compares it to a threshold built from
    var(all_sharpes). Feed it annualised dispersion and sigma comes out sqrt(252)
    = 15.9x too wide, sr0 becomes unreachable, and G2 rejects every candidate
    forever while looking like it is working. `daily=True` returns the same
    dispersion in deflated_sharpe's units.

    This is the F1 dispersion, for reports and for the thin-ledger fallback. The
    canonical G2 call is `deflated_sharpe(daily_net, dsr_trial_sharpes())`, which
    also fixes N to n_trials(); do not rebuild `all_sharpes` from this function.
    """
    s = f1_sharpes(state_dir)
    scale = np.sqrt(config.TRADING_DAYS_YEAR) if daily else 1.0
    if len(s) >= config.TRIAL_SR_STD_MIN_ROWS:
        sd = float(np.std(s, ddof=1))
        if np.isfinite(sd) and sd > 0.0:
            return sd / scale
    return 1.0 / np.sqrt(max(n_trials(state_dir), 1)) / scale


# ── vault access (Phase 5 calls this; it is counted, not free) ─────────────────
def record_vault_access(genome_hash: str, reason: str, state_dir=None) -> None:
    """Log one look at post-VAULT_START data. Deliberately NOT idempotent."""
    _append(vault_path(state_dir), VAULT_COLUMNS, [genome_hash, reason])


def vault_trial_sharpes(state_dir=None) -> np.ndarray:
    """Gate G3's `all_sharpes`: one DAILY Sharpe per genome that has SEEN the vault.

    The vault counterpart of dsr_trial_sharpes, and it exists for the same reason —
    so that no caller assembles a DSR input at the call site and gets the units or
    the count wrong. DESIGN's G3 says "vault DSR >= 0.90 at N = vault_trials", so
    the length of what comes back is vault_trials() by construction, exactly as
    dsr_trial_sharpes' length is n_trials().

    THE VALUES ARE PRE-VAULT SHARPES, and that is deliberate. Nothing in this
    project stores a per-genome VAULT Sharpe — writing one would make the vault a
    number selection could rank on, which is the whole thing the vault prevents.
    What deflated_sharpe needs from `all_sharpes` is a DISPERSION (how far the best
    of N tries strays from zero by luck) and a COUNT; the trial-Sharpe spread is
    the honest estimate of the first, and the vault-access count is exactly the
    second. A genome that has looked at the vault but somehow has no ledger row
    contributes 0.0 rather than being dropped — dropping it would shrink N, which
    is the one direction this correction may not be wrong in.
    """
    path = vault_path(state_dir)
    if not os.path.exists(path):
        return np.array([], dtype=np.float64)
    with open(path, newline="") as f:
        seen = sorted({row["genome_hash"] for row in csv.DictReader(f)})
    df = read_ledger(state_dir)
    by_hash = {}
    if len(df):
        order = df.assign(_rank=df["fidelity"].map({f: i for i, f in enumerate(FIDELITIES)}))
        order = order.sort_values(["genome_hash", "_rank", "generation"], kind="mergesort")
        best = order.groupby("genome_hash", sort=True).tail(1)
        vals = pd.to_numeric(best["sharpe_prevault"], errors="coerce").to_numpy(dtype=np.float64)
        by_hash = dict(zip(best["genome_hash"], np.where(np.isfinite(vals), vals, 0.0)))
    out = np.array([by_hash.get(h, 0.0) for h in seen], dtype=np.float64)
    return out / np.sqrt(config.TRADING_DAYS_YEAR)


def vault_trials(state_dir=None) -> int:
    """Distinct genomes that have ever been shown the vault — the N that deflates
    the vault Sharpe in gate G3."""
    path = vault_path(state_dir)
    if not os.path.exists(path):
        return 0
    with open(path, newline="") as f:
        return len({row["genome_hash"] for row in csv.DictReader(f)})


# ── Tier-A returns matrix (docs/DESIGN.md "Decision logging") ──────────────────
_MATRIX_KEYS = ("daily_net", "daily_gross", "turnover", "costs")


def write_returns_matrix(generation: int, results, state_dir=None, identity=None) -> str:
    """Write state/returns/gen_<NNNN>.npz: four (days x genomes) float32 matrices.

    `results` maps genome_hash -> an evaluate.full_eval result (a dict, or any
    sequence of (hash, result) pairs). Columns are ordered by genome hash so the
    file is a function of its contents, not of the order the workers finished in.

    Rows are the UNION of the genomes' scored dates, NaN where a genome was not
    active (families warm up at different bars; NaN says "not scored", 0.0 would
    say "scored, flat" and quietly drag every statistic toward zero).

    PRE-VAULT ONLY — asserted here, not assumed. full_eval already returns
    pre-vault arrays under these keys; this is the second lock on the door,
    because this file is exactly what a PBO or DSR run would consume.

    `identity` is (data_hash, panel_hash, config_hash) and is STORED IN THE FILE.
    Without it the artifact is a matrix of numbers with no record of what produced
    them: two runs of the same generation under different data would write
    interchangeable-looking files, and a downstream reader (PBO, DSR, the gates)
    could not tell that the matrix and the ledger rows it joins against describe
    different worlds. Optional only so the Phase-3 smoke tests still call it.

    Write-once: an existing generation file is left alone and its path returned,
    so a resumed run neither duplicates nor rewrites history. WRITE-ONCE IS NOT A
    MERGE — if the caller has computed a generation afresh under new inputs, the
    file already on disk is the OLD one and returning it silently would hand back
    an artifact that disagrees with the caller's own numbers. run_generation
    checks for exactly that before calling here, and refuses.
    """
    path = returns_path(generation, state_dir)
    if os.path.exists(path):
        return path
    pairs = results.items() if hasattr(results, "items") else results
    items = sorted(pairs, key=lambda kv: kv[0])          # by hash: never by finish order
    if not items:
        raise ValueError("no results to write for generation %s" % generation)

    vault = pd.Timestamp(config.VAULT_START)
    dates = pd.DatetimeIndex(sorted({d for _h, r in items for d in r["dates"]}))
    if len(dates) and dates[-1] >= vault:
        raise ValueError("returns matrix for generation %s reaches the vault (%s >= %s)"
                         % (generation, dates[-1].date(), vault.date()))

    hashes = [h for h, _r in items]
    mats = {k: np.full((len(dates), len(items)), np.nan, dtype=np.float32)
            for k in _MATRIX_KEYS}
    for col, (_h, res) in enumerate(items):
        rows = dates.get_indexer(pd.DatetimeIndex(res["dates"]))
        for k in _MATRIX_KEYS:
            mats[k][rows, col] = np.asarray(res[k], dtype=np.float32)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, generation=np.int64(generation),
                        dates=dates.values.astype("datetime64[ns]").astype(np.int64),
                        hashes=np.array(hashes, dtype="U12"),
                        identity=np.array(list(identity) if identity else ["", "", ""],
                                          dtype="U32"), **mats)
    os.replace(tmp, path)                    # never leave a half-written artifact
    return path


def load_returns_matrix(generation: int, state_dir=None) -> dict:
    """Read back what write_returns_matrix wrote; dates come back as a DatetimeIndex.

    `data_hash`/`panel_hash`/`config_hash` are empty strings for a matrix written
    before the identity was stored. A consumer that joins this against the ledger
    (gate G1's like-for-like check) must treat empty as "unknown provenance", not
    as "matches".
    """
    path = returns_path(generation, state_dir)
    with np.load(path, allow_pickle=False) as z:
        ident = [str(v) for v in z["identity"]] if "identity" in z.files else ["", "", ""]
        out = {"generation": int(z["generation"]),
               "dates": pd.DatetimeIndex(z["dates"].astype("datetime64[ns]")),
               "hashes": [str(h) for h in z["hashes"]],
               "data_hash": ident[0], "panel_hash": ident[1], "config_hash": ident[2]}
        for k in _MATRIX_KEYS:
            out[k] = z[k]
    return out


if __name__ == "__main__":
    # Smoke test: a full round trip in a temp directory. Touches no real state.
    import shutil
    import tempfile
    from dataclasses import replace

    import genome as gn

    tmp = tempfile.mkdtemp(prefix="arena_ledger_")
    saved = config.STATE_DIR
    config.STATE_DIR = tmp
    forget_cache()
    try:
        rng = np.random.default_rng(config.SEED)
        lib = tuple("feat_%02d" % i for i in range(20))
        genomes = [gn.random_genome(rng, lib) for _ in range(5)]
        dates = pd.bdate_range("2010-01-04", periods=300)

        print("arena trial ledger  (temp state dir: %s)" % tmp)
        written = 0
        results = {}
        for i, g in enumerate(genomes):
            net = rng.normal(0.0004, 0.01, len(dates))
            res = {"dates": dates, "daily_net": net, "daily_gross": net + 8e-5,
                   "turnover": np.abs(rng.normal(0.1, 0.02, len(dates))),
                   "costs": np.abs(rng.normal(1.2, 0.3, len(dates)))}
            results[g.hash()] = res
            for fid, score in (("F0", 0.3 + 0.1 * i), ("F1", 0.2 + 0.1 * i)):
                written += record_trial(0, g, fid, score, score, len(dates),
                                        "deadbeefdeadbeef", "cafebabecafebabe")
        again = sum(record_trial(0, g, f, 0.0, 0.0, 1, "deadbeefdeadbeef",
                                 "cafebabecafebabe")
                    for g in genomes for f in ("F0", "F1"))
        print("  rows        : %d written, %d on an identical re-run (idempotent)"
              % (written, again))
        print("  n_trials    : %d distinct genomes | trial_sr_std %.4f annual / %.4f daily"
              % (n_trials(), trial_sr_std(), trial_sr_std(daily=True)))
        _dsr = dsr_trial_sharpes()
        print("  DSR input   : %d trials (== n_trials: %s), daily units, best fidelity "
              "per genome, max %.4f (%.3f annualised)"
              % (len(_dsr), len(_dsr) == n_trials(), _dsr.max(),
                 _dsr.max() * np.sqrt(config.TRADING_DAYS_YEAR)))

        # A mutant of an existing genome, with lineage.
        child = gn.mutate(genomes[0], rng, lib)
        record_trial(1, child, "F1", 0.55, 0.55, 300, "deadbeefdeadbeef",
                     "cafebabecafebabe", parent_hash=genomes[0].hash(), birth_gen=1)
        print("  lineage row : %s <- parent %s (gen 1)" % (child.hash(), genomes[0].hash()))

        path = write_returns_matrix(0, results)
        back = load_returns_matrix(0)
        same = all(np.allclose(back["daily_net"][:, j],
                               np.float32(results[h]["daily_net"]), equal_nan=True)
                   for j, h in enumerate(back["hashes"]))
        print("  returns npz : %s  %d days x %d genomes, %.1f kB, round-trip %s"
              % (os.path.basename(path), len(back["dates"]), len(back["hashes"]),
                 os.path.getsize(path) / 1024.0, "OK" if same else "MISMATCH"))
        print("  columns     : sorted by hash -> %s" % ", ".join(back["hashes"][:3]))

        for g in genomes[:2]:
            record_vault_access(g.hash(), "smoke_test")
        record_vault_access(genomes[0].hash(), "smoke_test")   # not idempotent: 2 looks
        _vs = vault_trial_sharpes()
        print("  vault       : %d distinct genomes have seen post-%s data "
              "(3 accesses logged); G3 deflates at N=%d (== vault_trials: %s)"
              % (vault_trials(), config.VAULT_START, len(_vs),
                 len(_vs) == vault_trials()))
        with open(ledger_path()) as f:
            print("  header      : %s" % f.readline().strip())
    finally:
        config.STATE_DIR = saved
        forget_cache()
        shutil.rmtree(tmp, ignore_errors=True)
