"""
arena/verify.py — the test suite. Nothing gets scheduled until this is green.

docs/DESIGN.md lists eleven checks; the phases fill them in. Implemented here:

  1. Planted leak        two leaks, planted through the real env + eval path, and
                         the honest answer to each: an act-time clairvoyant column
                         (which NO backtest engine can defend, ours included — the
                         defense is upstream, in how features are built) and a
                         fit-time contamination (overlapping labels + a memorising
                         model), which the streaming purged walk-forward DOES
                         collapse from IC 0.40 to IC 0.03. See test_planted_leak's
                         docstring: the two are different failures and conflating
                         them is how a project ends up trusting the wrong defense.
  2. Determinism         two full generations (F0 + F1 + breed) of an 8-genome
                         population, run twice into two state directories on a
                         synthetic market: identical population hashes each
                         generation, byte-identical returns .npz artifacts,
                         byte-identical trial ledgers and hall of fame.
  3. Accounting fuzz     500 seeded random-target steps: the equity identity
                         holds, limits hold post-fill, shares stay integral,
                         costs stay non-negative, and the equity path replays
                         exactly from the decision log alone.
  4. Fill timing         a decision at close t fills at open t+1, at exactly the
                         configured price adjustment, correct sign each way, and
                         never on the same bar it was decided.
  5. Streaming purge     every model family's fit audit: no training label
                         resolves later than embargo trading days before the bar
                         it was fitted on, and a refit is audited every single
                         time one happens.
  6. Cost linearity      the same trade stream at stress_mult 0/1/2 costs
                         exactly 0/c/2c per component, and the frictionless
                         equity path dominates the costed ones pointwise.
  7. Gates               the ten-gate stack on synthetic metric dicts: an
                         all-pass report promotes (and moves the champion pointer
                         with a history row), every single-gate violation blocks
                         on its own, a tie goes to the incumbent, a data_hash
                         mismatch fails G1 whatever the scores say, and no
                         incumbent skips G9 while the other nine still bind.
                         Plus: a cross-family challenger, whose scored window
                         STARTS later because scoring begins at its own first
                         active bar, still passes G1 (the window END and the
                         identity triple are what must match) while a moved end
                         or a moved data_hash still fails it; a rejected artifact
                         store writes nothing (so the original evaluation can
                         still re-store as a no-op); a candidate outside the PBO
                         cohort gets no PBO and fails G4 unmeasured; regime
                         slices pass by absence; the bootstrap CI is reproducible
                         from a fixed rng; and the report of the week AFTER a
                         promotion prints the promotion's own gate table, labelled
                         with the generation it was run in, instead of ten "not
                         evaluated" rows and a G4 failure nobody measured.
  8. Trial ledger        k evaluations write exactly k rows, an identical re-run
                         writes none, DSR falls monotonically as the trial count
                         grows, and every vault access is counted.
  9. Genome ops          20,000 seeded mutations and crossovers stay inside
                         BOUNDS, keep n_long+n_short >= 3 and sorted/deduped
                         3-15 feature subsets (empty for rule families), survive
                         encode -> decode -> hash unchanged, and derive child
                         streams reproducibly from (SEED, generation, parent).
 10. No wall-clock       source scan: nothing reads the system clock (calendar
                         `now`, epoch seconds) outside run_*.py I/O boundaries.
                         The literal call spellings are deliberately absent from
                         this file's prose — the scanner is dumb on purpose.
 11. PBO sanity          CSCV on pure noise reports a high probability of
                         backtest overfitting; plant one genuinely persistent
                         column and it reports a low one, having found it.

All eleven of docs/DESIGN.md's checks are now implemented. Phase 7 adds a
twelfth, for the stage where a mistake stops being hypothetical:

 12. Paper stage         the arming gate as a truth table over synthetic
                         deep-eval history (three consecutive complete entries
                         under one champion arm it; a broken streak, another
                         champion, failed gates, or a budget-truncated run do
                         not); the weights -> whole-share -> delta conversion
                         against a fake position book (flips, dust, exits,
                         truncation, the unfiltered risk-cap repair); the paper
                         ledger's idempotency by order id and by content; and
                         the equivalence the whole design rests on — the orders
                         the live path computes at the last bar ARE the fills
                         the ordinary sandbox episode makes on the next one.

Everything here runs on a synthetic, seeded market. verify.py NEVER touches the
network or the cache: the test suite must give the same answer on a plane, on a
GitHub runner, and after a bad yfinance day.
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import tempfile

import numpy as np
import pandas as pd

import config                       # FIRST: puts the siblings on sys.path
import datafeed
import evaluate
import evolution
import features as arena_features
import gates
import genome as gn
import ledger
import broker_paper
import registry
import reports
import run_deepeval
import run_generation
import run_paper
from env import CostModel, MarketEnv
from strategy import StrategyAgent
from strategy import sharpe as report_sharpe      # NaN on degenerate input, unlike
                                                  # evaluate.sharpe's selection 0.0

ok = True


def check(name, cond, detail=""):
    global ok
    cond = bool(cond)
    ok = ok and cond
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")
    return cond


# ── synthetic market (deterministic; no network, no cache) ─────────────────────
def synthetic_market(n_days: int = 520, n_syms: int = 8, seed: int = None,
                     drift: float = 0.0002):
    """A seeded GBM market with overnight gaps, so open != close and the fill
    model is actually exercised. Price levels differ per symbol so whole-share
    rounding bites differently across names. `drift=0.0` gives pure noise: no
    true edge exists in it, so any measured skill is measurement error."""
    rng = np.random.default_rng(config.SEED if seed is None else seed)
    dates = pd.bdate_range("2000-01-03", periods=n_days)      # fixed, not wall-clock
    logret = rng.normal(drift, 0.015, size=(n_days, n_syms))
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


# ── 1. planted leak ────────────────────────────────────────────────────────────
# Independent markets the fit-time-contamination half of test 1 is repeated on.
# Five, because the residual it measures is sampling noise: one draw cannot
# separate "the firewall holds" from "this seed was kind".
LEAK_SEEDS = 5


def attach_features(market, panel: dict) -> None:
    """Attach a TEST-ONLY feature panel by hand.

    features.py is deliberately not involved: it needs the price cache and
    signal_lab's builder, and verify.py must give the same answer offline. It is
    also what makes the planted leak below possible AT ALL — see the structural
    check in test_planted_leak: nothing on the production path can construct a
    forward-looking column, so the leak has to be injected from outside.
    """
    market.features.clear()
    for name in sorted(panel):
        arr = np.ascontiguousarray(panel[name], dtype=np.float32)
        arr.setflags(write=False)
        market.features[name] = arr
    market.feature_names = tuple(sorted(panel))
    # The panel identity every ledger row and every checkpoint is keyed on. Test
    # panels get a real one for the same reason production ones do: two results are
    # comparable only if they were computed on the same inputs.
    market.panel_hash = arena_features.panel_hash(market.data_hash, market.feature_names,
                                                  market.features)


def _benign(market) -> dict:
    """Three ordinary backward-looking columns: 21-day momentum, 21-day realised
    vol, 5-day reversal. Each is a function of bars <= t, so none can leak
    anything — and three is the BOUNDS minimum, so a genome built on them alone is
    a legal genome."""
    close = pd.DataFrame(market.close)
    return {"benign_mom21": close.pct_change(21).to_numpy(),
            "benign_rv21": np.log(close).diff().rolling(21).std().to_numpy(),
            "benign_rev5": -close.pct_change(5).to_numpy()}


def _genome(family, horizon, features, params, rebalance=1):
    """A minimal, overlay-free genome: whatever skill shows up is the signal's."""
    return gn.Genome(
        signal=gn.SignalGene(family=family, horizon=horizon, refit_days=252,
                             features=tuple(sorted(features)), params=tuple(sorted(params))),
        portfolio=gn.PortfolioGene(n_long=3, n_short=3, weighting="equal", gross=1.0,
                                   vol_target=None, rebalance_days=rebalance),
        risk=gn.RiskGene(stop_loss=None, trail_stop=None, regime_filter=None,
                         regime_scale=0.0, dd_limit=None))


def _predict(model, family, X):
    if family == "ridge":
        return model.predict(X)
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    return proba[:, classes.index(1)] - proba[:, classes.index(-1)]


def NAIVE_EVALUATOR_ic(agent) -> float:
    """THE NAIVE EVALUATOR — what a careless backtest does, implemented here so
    the comparison is like-for-like.

    Same pipeline, same features, same labels as the arena path. The ONLY
    difference is the fitting discipline: this fits ONE model on EVERY labelled
    row — the act window included, no purge, no embargo — and then scores its
    predictions on those same rows. That is a pooled in-sample score, the
    shuffled-KFold family of mistake, and it is exactly how a strategy that has
    memorised its own test set comes back looking brilliant.
    """
    family = agent.genome.signal.family
    y_mat = agent._y if family == "ridge" else agent._y_class       # noqa: SLF001
    X = agent._X.reshape(-1, agent._X.shape[2])                     # noqa: SLF001
    realized = agent._y.reshape(-1)                                 # noqa: SLF001
    rows = np.flatnonzero(np.isfinite(realized) & np.isfinite(X).any(axis=1))
    model = agent._pipeline()                                       # noqa: SLF001
    Xr = np.asarray(X[rows], dtype=np.float64)
    model.fit(Xr, y_mat.reshape(-1)[rows])
    return float(np.corrcoef(_predict(model, family, Xr), realized[rows])[0, 1])


def arena_streaming_ic(agent, market) -> dict:
    """The ARENA PATH's act-time information coefficient.

    Replays run_episode's own refit cadence and, at each bar, correlates the
    prediction the strategy would have ACTED on with the return that bar actually
    went on to earn. No model is ever asked about a bar it was fitted after, which
    is the whole difference from the naive evaluator above.

    Returns the pooled correlation, the mean per-date cross-sectional correlation
    (the quant-standard IC, and the one that is not dominated by a handful of
    volatile dates), and the count behind them.
    """
    agent._model = None                                             # noqa: SLF001
    refit = agent.genome.signal.refit_days
    h = agent.genome.signal.horizon
    fwd = agent._y                                                  # noqa: SLF001
    last_fit, first, preds, realz = None, None, [], []
    for t in range(len(market.dates)):
        if last_fit is None or t - last_fit >= refit:
            if agent._fit(t, None):                                 # noqa: SLF001
                last_fit, first = t, (first if first is not None else t)
        if agent._model is None or t >= len(market.dates) - h:      # noqa: SLF001
            continue
        tradable = np.isfinite(market.close[t]) & (market.close[t] > 0.0)
        s = agent._scores(t, tradable)                              # noqa: SLF001
        keep = np.isfinite(s) & np.isfinite(fwd[t])
        preds.append(s[keep])
        realz.append(fwd[t][keep])
    p, r = np.concatenate(preds), np.concatenate(realz)
    daily = [float(np.corrcoef(a, b)[0, 1]) for a, b in zip(preds, realz)
             if len(a) > 2 and np.std(a) > 0 and np.std(b) > 0]
    return {"ic": float(np.corrcoef(p, r)[0, 1]), "daily_ic": float(np.mean(daily)),
            "n": len(p), "n_dates": len(daily), "first_fit": first}


def _purge_holds(market, audit) -> bool:
    """Every audited fit trained only on labels that had resolved at least
    WF_EMBARGO_DAYS trading days before the fit."""
    for row in audit:
        gap = (int(market.dates.searchsorted(row["fit_date"]))
               - int(market.dates.searchsorted(row["max_t1_used"])))
        if gap < config.WF_EMBARGO_DAYS:
            return False
    return bool(audit)


def test_planted_leak():
    """DESIGN check 1. Two DIFFERENT leaks, because they have different defenses
    and only one of them is the walk-forward's job.

    (a) ACT-TIME CLAIRVOYANCE. A column whose value at bar t IS bar t's future.
        The naive evaluator is fooled (IC ~ 1.0) and SO IS THE ARENA PATH — and
        that is the honest result, not a bug to assert away: purging decides which
        rows a model may be FITTED on, and no purge can help once the future is
        sitting in the row the strategy ACTS on. Every backtest engine ever
        written has this property. The defense is upstream and structural: arena
        computes no feature of its own. features.py hands the job to signal_lab's
        asof-truncated builder, which only ever does rolling, expanding, or
        POSITIVE shifts (past -> present), so a forward-looking column cannot be
        built there in the first place — asserted below by source scan, with
        point-in-time invariance itself proven by
        signal_lab/verify.py::test_pointintime (a cached-data test; this file
        stays offline).

        This test was originally specified to assert that the arena path
        COLLAPSES this leak to |Sharpe| < 1. It does not, measurably: it earns a
        Sharpe near 18 on a pure-noise market, and the fitted ridge puts all its
        weight on the poisoned column. Asserting the collapse would have been
        asserting something false about our own defenses — the one failure mode
        this project cannot afford — so the check pins the real behaviour instead,
        and will fail loudly if anyone ever changes it.

    (b) FIT-TIME CONTAMINATION — what the purge and the embargo actually exist
        for. A feature with ZERO true predictive power (a random walk independent
        of returns) plus heavily overlapping 21-day labels plus a model with
        enough capacity to memorise. Pooled in-sample scoring calls that skill
        (IC ~0.49 on every one of five independent markets); the streaming purged
        walk-forward collapses it to noise (IC mean -0.008, never worse than
        0.07, and a traded Sharpe within two standard errors of zero). Nothing
        about the market changed between those two numbers — only the discipline.

    WHY THE RESIDUAL IN (b) IS NOT EXACTLY ZERO, and why the bounds are what they
    are. 550 scored days is a finite sample: a Sharpe estimated from it has a
    standard error near sqrt(252/550) = 0.68 even when the truth is zero, and the
    6,300 act-time predictions behind the IC contain only ~26 independent 21-day
    label windows. So the honest claim is "indistinguishable from zero", not
    "zero", and it is tested as such — averaged over five markets, with per-market
    bounds loose enough to survive an unlucky draw. The single-seed thresholds this
    phase was briefed with (|IC| < 0.05, |Sharpe| < 1) happen to hold at the
    canonical seed but are not properties of the firewall: on other seeds this same
    zero-skill setup produced |IC| up to 0.15 and |Sharpe| up to 1.16. Pinning a
    test to the kind draw is the failure this project exists to avoid.
    """
    print("1. Planted leak through the env + eval path")

    # ── (a) act-time clairvoyance ─────────────────────────────────────────────
    h = 5
    market = synthetic_market(n_days=1600, n_syms=12, drift=0.0)   # no true edge
    close = market.close
    leak = np.full(close.shape, np.nan)
    leak[:-h] = np.log(close[h:] / close[:-h])          # bar t's own future, exactly
    panel = _benign(market)
    panel["leak_fwd"] = leak
    attach_features(market, panel)

    genome = _genome("ridge", h, tuple(panel), (("alpha", 1.0),))
    agent = StrategyAgent(genome, market)
    naive_ic = NAIVE_EVALUATOR_ic(agent)
    stream = arena_streaming_ic(agent, market)
    res = evaluate.full_eval(genome, market)
    coef = dict(zip(genome.signal.features,
                    agent._model.named_steps["clf"].coef_))        # noqa: SLF001

    check("(a) naive evaluator is fooled by the planted future", naive_ic > 0.30,
          "pooled in-sample IC = %.4f on a market with no true edge" % naive_ic)
    check("(a) purge held at FIT time even so", _purge_holds(market, res["fit_audit"]),
          "%d refits, every training label resolved >= %d trading days before its fit"
          % (len(res["fit_audit"]), config.WF_EMBARGO_DAYS))
    benign_max = max(abs(v) for k, v in coef.items() if k != "leak_fwd")
    check("(a) act-time clairvoyance is NOT defended by the purge (known, documented)",
          stream["ic"] > 0.90 and res["sharpe_prevault"] > 3.0,
          "streaming act-time IC = %.4f over %d predictions -> pre-vault net Sharpe "
          "%.2f; ridge puts %.3f on leak_fwd and at most %.1e on any benign column"
          % (stream["ic"], stream["n"], res["sharpe_prevault"], coef["leak_fwd"],
             benign_max))

    # The real defense against (a): such a column cannot be built by the code that
    # builds arena's features. Scan the production path for the only transforms
    # that can move the future backwards in time.
    forward = re.compile(r"shift\(\s*-|center\s*=\s*True|\[::-1\]|\.bfill\(|backfill")
    scanned, offenders = [], []
    for project, name in ((config.SIGNAL_LAB, "features.py"), (config.SIGNAL_LAB, "macro.py"),
                          (config.ROOT, "features.py")):
        tag = "%s/%s" % (os.path.basename(project), name)
        scanned.append(tag)
        with open(os.path.join(project, name)) as f:
            for lineno, line in enumerate(f, 1):
                if forward.search(line):
                    offenders.append("%s:%d" % (tag, lineno))
    check("(a) production feature path cannot build a forward-looking column",
          not offenders,
          ", ".join(offenders) if offenders else
          "no backward shift / centred window / reversal in %s (leak_fwd is "
          "test-injected; PIT invariance itself is proven by signal_lab's "
          "test_pointintime, which needs the cache)" % ", ".join(scanned))

    # ── (b) fit-time contamination — the leak the firewall is FOR ─────────────
    # Run it on LEAK_SEEDS independent markets, not one. The residual here is
    # sampling noise, and a single draw of it cannot tell "the firewall works"
    # from "this seed was kind": across five markets the streaming IC lands
    # anywhere in +-0.07 and the traded Sharpe anywhere in +-0.9, while the naive
    # evaluator sits at 0.48-0.50 every single time. Asserting the one-seed number
    # would be fitting the test to its own draw — the exact habit this file exists
    # to catch.
    trials = []
    for k in range(LEAK_SEEDS):
        seed = config.SEED + k
        market2 = synthetic_market(n_days=1600, n_syms=12, seed=seed, drift=0.0)
        rng = np.random.default_rng(seed + 7)
        # Persistent random walks, independent of returns: zero true skill, but each
        # time-neighbourhood has a distinctive value, so a model with capacity can
        # memorise which neighbourhood it is in — and the 21-day labels of adjacent
        # rows overlap by 20 days, so memorising the neighbourhood memorises the label.
        attach_features(market2, {"persist_%d" % i:
                                  np.cumsum(rng.normal(size=market2.close.shape), axis=0) / 10.0
                                  for i in range(3)})
        genome2 = _genome("hgb", 21, market2.feature_names,
                          (("learning_rate", 0.1), ("max_depth", 4), ("max_iter", 150),
                           ("min_samples_leaf", 100)), rebalance=5)
        agent2 = StrategyAgent(genome2, market2)
        naive2 = NAIVE_EVALUATOR_ic(agent2)
        stream2 = arena_streaming_ic(agent2, market2)
        res2 = evaluate.full_eval(genome2, market2)
        # A Sharpe estimated from n daily observations has standard error
        # ~sqrt(252/n) even when the true Sharpe is zero. 550 days -> 0.68, so
        # "indistinguishable from zero" means inside a couple of those, not "small".
        se = np.sqrt(config.TRADING_DAYS_YEAR / max(res2["n_days_prevault"], 1))
        trials.append({"naive": naive2, "ic": stream2["ic"], "daily_ic": stream2["daily_ic"],
                       "net": res2["sharpe_prevault"],
                       "gross": report_sharpe(res2["daily_gross"]),
                       "t": res2["sharpe_prevault"] / se, "n": res2["n_days_prevault"]})

    naive_ic2 = np.array([t["naive"] for t in trials])
    ic2 = np.array([t["ic"] for t in trials])
    net2 = np.array([t["net"] for t in trials])
    gross2 = np.array([t["gross"] for t in trials])
    tstat2 = np.array([t["t"] for t in trials])

    check("(b) naive evaluator is fooled by overlapping labels, on every market",
          naive_ic2.min() > 0.30,
          "pooled in-sample IC %.3f-%.3f across %d ZERO-skill markets"
          % (naive_ic2.min(), naive_ic2.max(), LEAK_SEEDS))
    check("(b) streaming purged walk-forward collapses it to zero on average",
          abs(ic2.mean()) < 0.05 and np.abs(ic2).max() < 0.10,
          "act-time IC mean %+.4f (per-market %s), daily-IC mean %+.4f"
          % (ic2.mean(), " ".join("%+.3f" % v for v in ic2),
             float(np.mean([t["daily_ic"] for t in trials]))))
    check("(b) the collapse is the discipline, not the market",
          (naive_ic2 / np.abs(ic2)).min() > 4.0,
          "same market, same pipeline, same labels, only the fitting rule differs: "
          "naive is %.0fx the act-time IC at worst" % (naive_ic2 / np.abs(ic2)).min())
    check("(b) and the traded result stays indistinguishable from noise",
          np.abs(net2).max() < 1.0 and np.abs(gross2).max() < 1.0 and np.abs(tstat2).max() < 2.0,
          "pre-vault net Sharpe %s (worst %.2f standard errors from zero on %d days), "
          "gross |max| %.2f" % (" ".join("%+.2f" % v for v in net2),
                                np.abs(tstat2).max(), trials[0]["n"], np.abs(gross2).max()))


# ── 2. determinism ─────────────────────────────────────────────────────────────
# A short synthetic history: the eras and the vault below are chosen to fit inside
# it, and WF_MIN_TRAIN_DAYS is shortened so the model families can actually reach
# their first fit. The arena's real settings assume 30 years; asking for them here
# would mean a 30-year synthetic market and a test nobody runs. What is under test
# is reproducibility, and that is scale-free.
DET_DAYS = 760                                   # 2000-01-03 .. 2002-11
DET_POP = 8
DET_GENS = 2
DET_MAX_RUNS = 60                                # bound on the one-genome-per-run leg
DET_CONFIG = {
    "SCREEN_ERAS": [("2000-11-01", "2001-04-30"),
                    ("2001-05-01", "2001-10-31"),
                    ("2001-11-01", "2002-04-30")],
    "VAULT_START": "2002-07-01",
    "WF_MIN_TRAIN_DAYS": 80,
}


def determinism_market():
    """A 10-symbol synthetic market with a panel rich enough that EVERY genome
    BOUNDS can draw is runnable: the benchmark symbol (spy_200dma needs it), a
    `vix_pct` column (vix_pct_80 needs it), and the seasonal column (seasonal_rule
    scores on it directly). Otherwise a random population would raise the moment
    the draw picked one of those genes, and the test would be measuring which
    genomes happened to be legal.

    Every column is backward-looking — rolling or shifted, never centred — because
    a leak here would not fail this test (both runs would leak identically) but
    would quietly make the numbers it prints meaningless.
    """
    market = synthetic_market(n_days=DET_DAYS, n_syms=10, seed=config.SEED)
    market.symbols[0] = config.BENCHMARK
    panel = _benign(market)
    close = pd.DataFrame(market.close)
    # A per-date "fear" level: trailing percentile rank of cross-sectional realised
    # vol, broadcast to every symbol the way features.py broadcasts a macro series.
    rv = np.log(close).diff().std(axis=1).rolling(21).mean()
    vix = rv.rolling(252, min_periods=21).rank(pct=True)
    panel["vix_pct"] = np.repeat(vix.to_numpy()[:, None], len(market.symbols), axis=1)
    # A seasonal score: the month-of-year mean return computed on PRIOR years only
    # (expanding, shifted), which is what signal_lab's _expanding_seasonal does and
    # why the panel's seasonal column is not a whole-sample statistic.
    ret = close.pct_change()
    month = pd.Series(market.dates.month, index=ret.index)
    seasonal = np.full(ret.shape, np.nan)
    for m in range(1, 13):
        rows = np.flatnonzero(month.to_numpy() == m)
        if len(rows) < 2:
            continue
        prior = ret.iloc[rows].expanding().mean().shift(1).to_numpy()
        seasonal[rows] = prior
    panel[arena_features.SEASONAL_COL] = seasonal
    attach_features(market, panel)
    return market


def _pair_run(market, state_dir: str) -> dict:
    """Seed a population and run DET_GENS complete generations into `state_dir`."""
    ledger.forget_cache()
    saved = config.STATE_DIR
    config.STATE_DIR = state_dir              # panel/artifact paths, for anything
    try:                                      # that resolves them lazily
        entries = run_generation.seed_population(DET_POP, market.feature_names, generation=0)
        run_generation.save_population(entries, 0, state_dir=state_dir)
        pops, best = [], []
        for _ in range(DET_GENS):
            entries, generation = run_generation.load_population(state_dir=state_dir)
            res = run_generation.run_generation(
                market, entries, generation, cost=CostModel(), n_jobs=1, evolve=True,
                deadline=None, state_dir=state_dir, verbose=False)
            if not res["complete"]:
                raise AssertionError("generation %d did not complete" % generation)
            pops.append([e["hash"] for e in res["entries_next"]])
            best.append(max(r["sharpe_prevault"] for _e, r, _s in res["f1"]))
        return {"pops": pops, "best": best,
                "ops": evolution.op_counts(res["entries_next"]),
                "n_trials": ledger.n_trials(state_dir), "hof": res["hof"]}
    finally:
        config.STATE_DIR = saved
        ledger.forget_cache()


def _chunked_run(market, state_dir: str) -> dict:
    """The same two generations, evaluated one genome per run.

    `deadline=0.0` is an instant in 1970, so the budget is spent the moment the
    first genome of a stage lands and every run stops after exactly one
    evaluation, checkpoints it and exits incomplete. Re-running to completion
    therefore crosses every resume boundary there is: mid-F0, the F0 -> F1
    handover, and mid-F1. A wall clock would make this test time-dependent; a
    deadline in the past makes it a pure function.
    """
    ledger.forget_cache()
    saved = config.STATE_DIR
    config.STATE_DIR = state_dir
    try:
        entries = run_generation.seed_population(DET_POP, market.feature_names, generation=0)
        run_generation.save_population(entries, 0, state_dir=state_dir)
        pops, runs, stops = [], 0, 0
        for _ in range(DET_GENS):
            while True:
                runs += 1
                if runs > DET_MAX_RUNS:
                    raise AssertionError("chunked run did not converge in %d runs"
                                         % DET_MAX_RUNS)
                entries, generation = run_generation.load_population(state_dir=state_dir)
                res = run_generation.run_generation(
                    market, entries, generation, cost=CostModel(), n_jobs=1, evolve=True,
                    deadline=0.0, state_dir=state_dir, verbose=False)
                if res["complete"]:
                    pops.append([e["hash"] for e in res["entries_next"]])
                    break
                stops += 1
        return {"pops": pops, "runs": runs, "stops": stops}
    finally:
        config.STATE_DIR = saved
        ledger.forget_cache()


def _read(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def test_determinism():
    """DESIGN check 2. The claim the whole project rests on: a generation is a
    pure function of (SEED, generation, population, data), so any two runs of the
    same generation produce the same genomes, the same simulated returns and the
    same ledger — on this platform (BLAS differences across architectures can flip
    low-order bits, which is why ledger rows are platform-tagged).

    It is proved by BYTES, not by tolerances. A tolerance-based comparison would
    pass while a shared mutable rng silently reordered the search, which is the
    exact failure evolution.py's per-slot streams exist to prevent.
    """
    print("\n2. Determinism (two generations, run twice, compared byte for byte)")
    saved_cfg = {k: getattr(config, k) for k in DET_CONFIG}
    dirs = [tempfile.mkdtemp(prefix="arena_verify_det%d_" % i) for i in (1, 2, 3)]
    try:
        for k, v in DET_CONFIG.items():
            setattr(config, k, v)
        market = determinism_market()
        a = _pair_run(market, dirs[0])
        b = _pair_run(market, dirs[1])

        # The run has to have DONE something, or every equality below is vacuous.
        check("the pair-run actually ran a search",
              a["n_trials"] >= DET_POP + 1 and len(a["pops"]) == DET_GENS
              and sum(a["ops"].get(op, 0) for op in ("mutate", "crossover")) > 0,
              "%d distinct genomes ledgered over %d generations of %d; generation %d "
              "was bred %s" % (a["n_trials"], DET_GENS, DET_POP, DET_GENS,
                               ", ".join("%s %d" % kv for kv in sorted(a["ops"].items()))))

        same_pops = a["pops"] == b["pops"]
        check("identical population hashes after every generation", same_pops,
              "gen %s" % " | ".join("%d: %s..." % (i + 1, ",".join(h[:6] for h in p[:3]))
                                    for i, p in enumerate(a["pops"])))
        check("identical best pre-vault Sharpe per generation",
              all(x == y for x, y in zip(a["best"], b["best"])),
              "best F1 SR by generation: %s"
              % ", ".join("%+.6f" % s for s in a["best"]))

        npz = ["returns/gen_%04d.npz" % g for g in range(DET_GENS)]
        diffs = [n for n in npz
                 if _read(os.path.join(dirs[0], n)) != _read(os.path.join(dirs[1], n))]
        check("byte-identical returns artifacts", not diffs,
              "%s differ" % ", ".join(diffs) if diffs else
              "%s (%s)" % (", ".join(npz),
                           ", ".join("%.1f kB" % (os.path.getsize(os.path.join(dirs[0], n))
                                                  / 1024.0) for n in npz)))

        led = [_read(os.path.join(d, "trial_ledger.csv")) for d in dirs[:2]]
        rows = led[0].decode().strip().splitlines()
        check("byte-identical trial ledger (every column)", led[0] == led[1],
              "%d rows incl. header, %d bytes; platform %s"
              % (len(rows), len(led[0]), ledger.platform_tag()))

        hof = [_read(os.path.join(d, evolution.HOF_FILE)) for d in dirs[:2]]
        check("byte-identical hall of fame", hof[0] == hof[1] and bool(a["hof"]),
              "%d records, best %s SR %+.4f"
              % (len(a["hof"]), a["hof"][0]["hash"], a["hof"][0]["sharpe_prevault"]))

        # Determinism must not be the trivial kind: if evolution never changed the
        # population, two identical runs would prove nothing about the operators.
        moved = sum(1 for x, y in zip(a["pops"][0], a["pops"][1]) if x != y)
        check("the population actually moved between generations", moved > 0,
              "%d of %d slots differ between generation 1 and 2" % (moved, DET_POP))

        # The artifact has to say what produced it, or a matrix written under one
        # market is indistinguishable from one written under another.
        mat = ledger.load_returns_matrix(0, dirs[0])
        check("returns artifact carries its own identity",
              (mat["data_hash"], mat["panel_hash"], mat["config_hash"])
              == (market.data_hash, market.panel_hash, config.config_hash()),
              "data %s | panel %s | config %s stored in gen_0000.npz"
              % (mat["data_hash"], mat["panel_hash"], mat["config_hash"]))

        # ── the interrupted leg ───────────────────────────────────────────────
        # Same two generations, but every run is stopped by the time budget after
        # one genome and resumed. Determinism has to survive being cut in half:
        # the checkpoints, the ledger-resumed F0 scores and the fresh episodes
        # must reassemble into the same generation, byte for byte, as the run that
        # was never interrupted.
        c = _chunked_run(market, dirs[2])
        check("a run stopped by the budget resumes to the same population",
              c["pops"] == a["pops"],
              "%d runs, %d of them stopped mid-generation, to reach the same %d "
              "generations" % (c["runs"], c["stops"], DET_GENS))
        for name in [n for n in npz] + ["trial_ledger.csv", evolution.HOF_FILE,
                                        run_generation.POPULATION_FILE]:
            check("resumed == uninterrupted: %s" % name,
                  _read(os.path.join(dirs[0], name)) == _read(os.path.join(dirs[2], name)),
                  "%d bytes" % os.path.getsize(os.path.join(dirs[2], name)))
        check("no checkpoint directory survives a completed generation",
              not glob.glob(os.path.join(dirs[2], "tmp_gen_*")),
              "state/tmp_gen_* removed once the population advanced")
    finally:
        for k, v in saved_cfg.items():
            setattr(config, k, v)
        for d in dirs:
            shutil.rmtree(d, ignore_errors=True)


# ── 3. accounting fuzz ─────────────────────────────────────────────────────────
def test_accounting():
    print("\n3. Accounting fuzz (500 seeded random-target steps)")
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
    # the rest of the run — inside LEV_EPS, so nothing should trade.
    # Two things this test has to get right or it passes for the wrong reason:
    #   • count EVERY reducing fill, not just the ones tagged "risk_cap" — the
    #     leverage trim issues its cuts under the strategy's own reason tag;
    #   • run it where LEV_EPS*equity comfortably exceeds MIN_POSITION_USD, or the
    #     churn order dies in the dust filter and the breach hides at small size.
    big_cash = 2_000_000.0
    assert config.LEV_EPS * big_cash > config.MIN_POSITION_USD, "test cannot bind"
    saved_cash = config.START_CASH
    config.START_CASH = big_cash
    try:
        flat = flat_market()
        fenv = MarketEnv(flat, CostModel())
        fobs = fenv.reset()
        nf = len(flat.symbols)
        scaled = reducing = 0
        for k in range(40):
            px = flat.close[fobs["t"]]
            w = (np.full(nf, 1.0 / nf) if k == 0
                 else fenv.shares * px / fobs["equity"])      # hold what we hold
            fobs, _, fdone, finfo = fenv.step(w)
            scaled += int(finfo["leverage_scaled"] < 1.0)
            reducing += sum(1 for r in finfo["fills"]
                            if abs(r["weight_after"]) < abs(r["weight_before"]))
            if fdone:
                break
        net = float(np.sum(fenv.shares * flat.close[fenv.t])) / fenv.equity
        end_cash = fenv.cash
    finally:
        config.START_CASH = saved_cash
    check("a book held exactly at the net cap never self-deleverages",
          scaled == 0 and reducing == 0 and net > config.MAX_NET_LEV,
          "$%.0fk account: net settled at %.5f (cash $%.2f), %d scale events, "
          "%d reducing trades" % (big_cash / 1e3, net, end_cash, scaled, reducing))

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


# ── 5. streaming purge ─────────────────────────────────────────────────────────
def test_streaming_purge():
    """DESIGN check 5, for every family that fits anything.

    The streaming walk-forward already guarantees that no data after the fit bar
    exists — the simulator has not reached it. The one leak that survives that is
    the LABEL WINDOW: a row dated t carries a label that only resolves at t+h, so
    training on it at bar f < t+h would be training on the future. strategy._fit
    purges those rows and embargoes another WF_EMBARGO_DAYS on top. This test
    proves it from the outside, from the audit trail rather than from the code.
    """
    print("\n5. Streaming purge (fit audit: labels resolved, then embargoed)")
    market = synthetic_market(n_days=1600, n_syms=12, drift=0.0)
    attach_features(market, _benign(market))
    n_bars = len(market.dates)

    for family, params in (("ridge", (("alpha", 1.0),)),
                           ("logistic", (("C", 1.0),)),
                           ("hgb", (("learning_rate", 0.1), ("max_depth", 2),
                                    ("max_iter", 100), ("min_samples_leaf", 200)))):
        genome = _genome(family, 21, market.feature_names, params, rebalance=5)
        audit: list = []
        StrategyAgent(genome, market).run_episode(fit_audit=audit)

        worst_gap, bad = None, []
        for row in audit:
            gap = (int(market.dates.searchsorted(row["fit_date"]))
                   - int(market.dates.searchsorted(row["max_t1_used"])))
            worst_gap = gap if worst_gap is None else min(worst_gap, gap)
            if gap < config.WF_EMBARGO_DAYS:
                bad.append("%s: label closed %d bars before fit"
                           % (row["fit_date"].date(), gap))
            for key in ("fit_date", "max_t1_used", "n_rows", "family", "horizon",
                        "embargo_days"):
                if key not in row:
                    bad.append("audit row missing %s" % key)

        # Every refit that happened is audited: fits land on a fixed cadence from
        # the first one, and the last one is within one cadence of the episode end.
        bars = [int(market.dates.searchsorted(r["fit_date"])) for r in audit]
        cadence = genome.signal.refit_days
        spaced = all(b - a == cadence for a, b in zip(bars, bars[1:]))
        complete = bool(bars) and bars[-1] + cadence > n_bars - 2

        check("%-8s every training label resolved >= %d trading days pre-fit"
              % (family, config.WF_EMBARGO_DAYS), not bad and worst_gap is not None,
              "; ".join(bad[:3]) if bad else
              "%d refits, tightest gap %d bars (embargo %d + horizon %d), %d train rows max"
              % (len(audit), worst_gap, config.WF_EMBARGO_DAYS, genome.signal.horizon,
                 max(r["n_rows"] for r in audit)))
        check("%-8s an audit row exists for every refit" % family, spaced and complete,
              "fits at bars %s of %d, cadence %d" % (bars, n_bars, cadence))

    # Rule families fit nothing, so they must not produce audit rows at all.
    rule = gn.Genome(signal=gn.SignalGene("mom_rule", 21, 252, (),
                                          (("lookback", 63), ("skip", 5))),
                     portfolio=gn.PortfolioGene(3, 3, "equal", 1.0, None, 5),
                     risk=gn.RiskGene(None, None, None, 0.0, None))
    rule_audit: list = []
    StrategyAgent(rule, market).run_episode(fit_audit=rule_audit)
    check("rule families fit nothing and audit nothing", not rule_audit,
          "%d audit rows from mom_rule" % len(rule_audit))


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


# ── 7. gates ───────────────────────────────────────────────────────────────────
# One candidate that clears every gate with room to spare, and an incumbent it
# beats. Every case below is this dict with ONE field moved, so a failure names
# exactly the rule that broke rather than a soup of interacting numbers.
PASSING_CAND = {
    "hash": "cand00000001",
    "identity": ("data0000deadbeef", "panel000cafebabe", "cfg00000feedface"),
    "window": ("1999-01-04", "2019-12-31"),
    "resimulated": True,
    "sharpe": 1.10, "dsr": 0.98, "dsr_n_trials": 812,
    "vault_sharpe": 0.65, "vault_dsr": 0.94, "vault_trials": 6,
    "pbo": 0.12,
    "cpcv_frac_positive": 0.82, "cpcv_median_sharpe": 0.55, "cpcv_n_paths": 28,
    "boot_ci_lo": 0.31, "boot_ci_hi": 1.88,
    "sharpe_stress": 0.74,
    "regime_slices": [0.12, -0.03, 0.08, -0.02],
    "rolling_win_frac": 0.71, "rolling_n_windows": 45,
    "p_ruin": 0.02,
}
PASSING_INC = dict(PASSING_CAND, hash="incum0000001", sharpe=0.80)

# (gate, the single field that violates it) — one per gate, all ten.
GATE_VIOLATIONS = [
    ("G1", {"identity": ("OTHERdata0000000", "panel000cafebabe", "cfg00000feedface")}),
    ("G2", {"dsr": config.GATE_MIN_DSR - 0.001}),
    ("G3", {"vault_dsr": config.GATE_VAULT_MIN_DSR - 0.001}),
    ("G4", {"pbo": config.GATE_MAX_PBO + 0.001}),
    ("G5", {"cpcv_frac_positive": config.GATE_CPCV_MIN_POS_FRAC - 0.001}),
    ("G6", {"boot_ci_lo": 0.0}),
    ("G7", {"sharpe_stress": (config.GATE_STRESS_MIN_SR_RATIO - 0.01) * 1.10}),
    ("G8", {"regime_slices": [config.GATE_REGIME_MAX_LOSS - 0.001, -0.03, 0.08, -0.02]}),
    ("G9", {"rolling_win_frac": config.GATE_ROLLING_WIN_FRAC - 0.001}),
    ("G10", {"p_ruin": config.GATE_RUIN_MAX_PROB}),
]


def test_gates():
    """DESIGN check 7. The decision stack is a pure function of two metric dicts,
    so it can be tested exhaustively without a market — and it is: all ten gates,
    one violation at a time, plus the three rules that are easy to get subtly
    wrong (ties, a missing incumbent, and an identity mismatch outranking every
    score in the report)."""
    print("\n7. Promotion gates (G1-G10, pure functions of two metric dicts)")

    report = gates.evaluate_gates(PASSING_CAND, PASSING_INC)
    check("a candidate clearing every threshold passes all ten", report["all_pass"],
          "%d gates, none failed; ties_to_incumbent=%s"
          % (report["n_gates"], report["ties_to_incumbent"]))
    check("the report covers exactly DESIGN's ten gates",
          list(report["gates"]) and set(report["gates"]) == set(gates.GATE_ORDER)
          and len(gates.GATE_ORDER) == 10,
          ", ".join(gates.GATE_ORDER))

    blocked = []
    for gid, patch in GATE_VIOLATIONS:
        rep = gates.evaluate_gates(dict(PASSING_CAND, **patch), PASSING_INC)
        blocked.append((gid, rep["failed"] == [gid]))
    check("every single-gate violation blocks promotion, alone",
          all(good for _g, good in blocked),
          "; ".join("%s %s" % (g, "blocks" if good else "DID NOT BLOCK ALONE")
                    for g, good in blocked))

    # A tie is not a win: the candidate matches the incumbent exactly, and again
    # at exactly the required margin. Neither may promote.
    tie = gates.evaluate_gates(dict(PASSING_CAND, sharpe=PASSING_INC["sharpe"]), PASSING_INC)
    on_margin = gates.evaluate_gates(
        dict(PASSING_CAND, sharpe=PASSING_INC["sharpe"] + config.GATE_BEAT_SR_MARGIN),
        PASSING_INC)
    check("a tie goes to the incumbent (G9 is a strict beat)",
          not tie["gates"]["G9"]["pass"] and not on_margin["gates"]["G9"]["pass"]
          and not tie["all_pass"] and not on_margin["all_pass"],
          "equal Sharpe -> G9 %s; exactly incumbent+%.2f -> G9 %s"
          % ("FAIL" if not tie["gates"]["G9"]["pass"] else "pass",
             config.GATE_BEAT_SR_MARGIN,
             "FAIL" if not on_margin["gates"]["G9"]["pass"] else "pass"))

    # G1 outranks the scoreboard: a spectacular candidate measured on other data
    # is not a candidate, it is an anecdote.
    spectacular = dict(PASSING_CAND, sharpe=9.9, dsr=0.999, vault_sharpe=3.0,
                       vault_dsr=0.99, pbo=0.0, cpcv_frac_positive=1.0,
                       cpcv_median_sharpe=2.5, boot_ci_lo=2.0, sharpe_stress=9.0,
                       p_ruin=0.0, rolling_win_frac=1.0,
                       identity=("DIFFERENTdata001",) + PASSING_CAND["identity"][1:])
    rep = gates.evaluate_gates(spectacular, PASSING_INC)
    check("a data_hash mismatch fails G1 whatever the scores are",
          not rep["all_pass"] and rep["failed"] == ["G1"],
          "nine gates pass on numbers measured against other data; G1: %s"
          % rep["gates"]["G1"]["detail"])
    resumed = gates.evaluate_gates(PASSING_CAND, dict(PASSING_INC, resimulated=False))
    check("...and so does an incumbent that was not re-simulated fresh",
          not resumed["all_pass"] and resumed["failed"] == ["G1"],
          resumed["gates"]["G1"]["detail"][:88])

    # CROSS-FAMILY DETHRONEMENT. evaluate.full_eval starts scoring at each genome's
    # own first active bar, so a rule family (bar WF_MIN_TRAIN_DAYS) and a model
    # family (the bar after its first refit, weeks later and different per horizon)
    # produce scored slices that BEGIN on different dates over the identical
    # market. If G1 compared window starts, the first promotion would lock every
    # other family out permanently — the gate would be enforcing family, not
    # evidence. The end, and the identity triple that pins the data span, are what
    # must match; G9 already intersects the two calendars before counting a window.
    cross = dict(PASSING_CAND, window=("1999-06-14", "2019-12-31"))
    rep_cross = gates.evaluate_gates(cross, PASSING_INC)
    end_moved = gates.evaluate_gates(dict(PASSING_CAND, window=("1999-01-04", "2019-06-28")),
                                     PASSING_INC)
    cross_data = gates.evaluate_gates(
        dict(cross, identity=("OTHERdata0000000",) + PASSING_CAND["identity"][1:]),
        PASSING_INC)
    check("a challenger whose scored window STARTS later still passes G1 (and promotes)",
          rep_cross["gates"]["G1"]["pass"] and rep_cross["all_pass"]
          and "START on different bars" in rep_cross["gates"]["G1"]["detail"],
          rep_cross["gates"]["G1"]["detail"][:104])
    check("...but a different window END is not like-for-like and fails G1",
          not end_moved["gates"]["G1"]["pass"] and end_moved["failed"] == ["G1"]
          and "window end" in end_moved["gates"]["G1"]["detail"],
          end_moved["gates"]["G1"]["detail"][:88])
    check("...and a different data_hash fails it whatever the starts are",
          not cross_data["gates"]["G1"]["pass"] and cross_data["failed"] == ["G1"]
          and "data_hash" in cross_data["gates"]["G1"]["detail"],
          cross_data["gates"]["G1"]["detail"][:88])
    no_inc_window = gates.evaluate_gates(PASSING_CAND, dict(PASSING_INC, window=()))
    check("...and an incumbent with no readable window fails G1 rather than skipping it",
          not no_inc_window["gates"]["G1"]["pass"] and no_inc_window["failed"] == ["G1"],
          no_inc_window["gates"]["G1"]["detail"][:88])

    # No incumbent: G9 is skipped (nothing to beat) and G1 has nothing to compare
    # against, so it degrades to "is this candidate's own identity complete". The
    # other EIGHT must still bind exactly as they do with a champion present.
    solo = gates.evaluate_gates(PASSING_CAND, None)
    solo_broken = [gid for gid, patch in GATE_VIOLATIONS if gid not in ("G1", "G9")
                   and gates.evaluate_gates(dict(PASSING_CAND, **patch), None)["failed"] != [gid]]
    check("with no incumbent G9 is skipped and the other gates still bind",
          solo["all_pass"] and solo["gates"]["G9"]["pass"] and not solo_broken,
          "G9: %s" % solo["gates"]["G9"]["detail"])

    # features.py attaches panel_hash to the MarketData ad hoc, so a caller
    # reading it with getattr can hold None. That must fail G1 — with or without
    # an incumbent — rather than compare equal to another missing hash.
    blind = dict(PASSING_CAND, identity=(PASSING_CAND["identity"][0], None,
                                         PASSING_CAND["identity"][2]))
    check("a missing panel_hash fails G1 even when there is no incumbent",
          not gates.evaluate_gates(blind, None)["gates"]["G1"]["pass"]
          and not gates.evaluate_gates(blind, PASSING_INC)["gates"]["G1"]["pass"],
          gates.evaluate_gates(blind, None)["gates"]["G1"]["detail"])

    # An unmeasurable gate is a failed gate — never a skipped one.
    for missing in ("dsr", "pbo", "p_ruin", "boot_ci_lo"):
        rep = gates.evaluate_gates({k: v for k, v in PASSING_CAND.items() if k != missing},
                                   PASSING_INC)
        if rep["all_pass"]:
            break
    else:
        missing = None
    check("a missing measurement fails its gate rather than skipping it",
          missing is None, "dsr / pbo / p_ruin / boot_ci_lo each removed in turn")

    # The report a passing candidate produces has to be able to MOVE the pointer,
    # and a failing one has to be refused by the registry itself.
    tmp = tempfile.mkdtemp(prefix="arena_gates_")
    try:
        good = gates.evaluate_gates(PASSING_CAND, PASSING_INC)
        bad = gates.evaluate_gates(dict(PASSING_CAND, dsr=0.1), PASSING_INC)
        registry.promote(PASSING_INC["hash"], 3, None, "seed the pointer", state_dir=tmp)
        registry.promote(PASSING_CAND["hash"], 4, good, "gates passed", state_dir=tmp)
        h, meta = registry.champion(tmp)
        rows = registry.champion_history(tmp)
        refused = False
        try:
            registry.promote("nevernevernev", 5, bad, state_dir=tmp)
        except ValueError:
            refused = True
        registry.rollback(PASSING_INC["hash"], "verify rollback", state_dir=tmp)
        back, _m = registry.champion(tmp)
        after = registry.champion_history(tmp)
        check("an all-pass report promotes, a failing one is refused, rollback restores",
              h == PASSING_CAND["hash"] and meta["previous_hash"] == PASSING_INC["hash"]
              and len(rows) == 2 and refused and back == PASSING_INC["hash"]
              and len(after) == 3 and after[-1]["reason"] == "rollback",
              "%d pointer moves, each with a history row; failing report refused: %s"
              % (len(after), refused))

        # A REJECTED STORE MUST CHANGE NOTHING. Storing changed bytes under the
        # same eval key has to raise — and leave no file behind, or the original
        # evaluation's own byte-identical re-store would be illegal from then on
        # and the weekly job would abort mid-persistence needing hand repair.
        arts = os.path.join(tmp, "artifacts")
        rng = np.random.default_rng(config.SEED)
        lib = tuple("feat_%02d" % i for i in range(12))
        g = gn.random_genome(rng, lib)
        entry = {"genome": g.to_dict(), "hash": g.hash(), "op": "mutate",
                 "parent_hash": "", "birth_gen": 0}
        dates = pd.bdate_range("2000-01-03", periods=200)
        base = {"dates": dates, "daily_net": rng.normal(0.0004, 0.01, len(dates)),
                "daily_gross": rng.normal(0.0005, 0.01, len(dates)),
                "turnover": np.abs(rng.normal(0.1, 0.01, len(dates))),
                "costs": np.abs(rng.normal(1.0, 0.1, len(dates))),
                "identity": ("data0000deadbeef", "panel000cafebabe", "cfg00000feedface"),
                "generation": 4, "score": 0.5, "sharpe_prevault": 0.6,
                "n_days_prevault": len(dates), "n_features": len(g.signal.features),
                "first_active": 0, "regime_finite_frac": None, "n_fits": 0}
        first = registry.store_artifact(entry, base, {"dsr": 0.9}, artifact_dir=arts)
        listing = sorted(os.listdir(first))
        rejected = False
        try:
            registry.store_artifact(entry, dict(base, daily_net=base["daily_net"] * 1.5),
                                    {"dsr": 0.9}, artifact_dir=arts)
        except registry.ImmutableArtifact:
            rejected = True
        orphans = sorted(os.listdir(first))
        registry.store_artifact(entry, base, {"dsr": 0.9}, artifact_dir=arts)   # no-op
        check("a rejected store writes nothing, and the original still re-stores",
              rejected and orphans == listing and sorted(os.listdir(first)) == listing,
              "changed series under the same eval key raised; %d files before, %d "
              "after the rejection, %d after the identical re-store"
              % (len(listing), len(orphans), len(os.listdir(first))))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # G4's cohort membership: CSCV is a statement about ONE cohort's selection, and
    # a candidate that is not in that cohort's returns matrix (an all-time
    # hall-of-fame leader from an older generation) may not borrow its number.
    cohort = {"pbo": 0.10, "n_splits": 12870, "generation": 7,
              "hashes": ["aaaaaaaaaaaa", PASSING_CAND["hash"]]}
    inside, note_in = run_deepeval.cohort_pbo(cohort, PASSING_CAND["hash"])
    outside, note_out = run_deepeval.cohort_pbo(cohort, "notinthecohort")
    empty, _n = run_deepeval.cohort_pbo({"pbo": None, "hashes": []}, PASSING_CAND["hash"])
    rep_in = gates.evaluate_gates(dict(PASSING_CAND, pbo=inside), PASSING_INC)
    rep_out = gates.evaluate_gates(dict(PASSING_CAND, pbo=outside, pbo_note=note_out),
                                   PASSING_INC)
    check("a candidate outside the PBO cohort gets no PBO, and G4 fails unmeasured",
          inside == 0.10 and outside is None and empty is None
          and rep_in["all_pass"] and rep_out["failed"] == ["G4"],
          "in cohort -> %.2f (G4 %s) | absent -> %s (G4 %s): %s"
          % (inside, "PASS" if rep_in["gates"]["G4"]["pass"] else "FAIL", outside,
             "PASS" if rep_out["gates"]["G4"]["pass"] else "FAIL", note_out[:60]))
    # ...and the table says WHY, rather than showing a bare failure a reader would
    # have to go to the job log to explain.
    g4_line = [ln for ln in gates.gate_table(rep_out) if ln.strip().startswith("G4")][0]
    check("...and the gate table carries the reason, not the generic CSCV note",
          "NOT MEASURABLE" in g4_line and "not one of the" in g4_line
          and "COHORT statistic" not in g4_line, g4_line.strip()[:104])

    # The decision record's schema is checked before an append: csv.writer appends
    # positionally, so a file from an older column list would silently misalign
    # every field of every future row.
    tmp2 = tempfile.mkdtemp(prefix="arena_hist_")
    try:
        row = {c: "x" for c in run_deepeval.HISTORY_COLUMNS}
        run_deepeval.append_history(row, state_dir=tmp2)
        run_deepeval.append_history(row, state_dir=tmp2)
        good = len(run_deepeval.read_history(tmp2)) == 2
        with open(run_deepeval.history_path(tmp2), "w") as f:
            f.write(",".join(run_deepeval.HISTORY_COLUMNS[:-3]) + "\n")   # an older schema
        refused = False
        try:
            run_deepeval.append_history(row, state_dir=tmp2)
        except ValueError:
            refused = True
        check("appending to a deep-eval history of another shape fails loudly",
              good and refused,
              "two rows appended and read back; a %d-column file refused a "
              "%d-column row" % (len(run_deepeval.HISTORY_COLUMNS) - 3,
                                 len(run_deepeval.HISTORY_COLUMNS)))
    finally:
        shutil.rmtree(tmp2, ignore_errors=True)

    # An F2 row whose key is taken by an EARLIER vintage is kept (and reported);
    # one that record_trial claims exists but that cannot be read back is LOST,
    # and that is an error, not a drift.
    tmp3 = tempfile.mkdtemp(prefix="arena_f2_")
    try:
        ledger.forget_cache()
        g2 = gn.random_genome(np.random.default_rng(7), tuple("f%02d" % i for i in range(12)))
        e2 = {"genome": g2.to_dict(), "hash": g2.hash(), "op": "seed",
              "parent_hash": "", "birth_gen": 0}
        r2 = {"score": 0.4, "sharpe_prevault": 0.5, "n_days_prevault": 300}
        # The config hash on a row is always the WRITING process's own
        # (ledger.record_trial takes only the data and panel hashes), so a
        # like-for-like identity here has to carry this process's.
        ident_a = ("dataA", "panelA", config.config_hash())
        ident_b = ("dataB", "panelB", config.config_hash())
        first = run_deepeval._ledger_f2(3, e2, r2, ident_a, state_dir=tmp3)   # noqa: SLF001
        same = run_deepeval._ledger_f2(3, e2, r2, ident_a, state_dir=tmp3)    # noqa: SLF001
        drift = run_deepeval._ledger_f2(3, e2, r2, ident_b, state_dir=tmp3)   # noqa: SLF001
        os.remove(ledger.ledger_path(tmp3))      # the row record_trial remembers vanishes
        lost = False
        try:
            run_deepeval._ledger_f2(3, e2, r2, ident_b, state_dir=tmp3)       # noqa: SLF001
        except run_generation.IdentityDrift:
            lost = True
        check("an F2 row taken by an earlier vintage drifts; an unreadable one raises",
              first["status"] == "wrote" and same["status"] == "already"
              and drift["status"] == "drift"
              and drift["prior_identity"] == ident_a and lost,
              "wrote -> already -> drift (prior %s, kept), and a row that cannot be "
              "read back raises instead of pretending an older row stands"
              % "|".join(drift["prior_identity"]))
    finally:
        ledger.forget_cache()
        shutil.rmtree(tmp3, ignore_errors=True)

    # ── the two F2 inputs whose rules are not obvious from their signatures ────
    # A series that covers the 2000-02 and 2008-09 windows and stops long before
    # the two vault-era ones: exactly the shape a pre-vault-only evaluation has.
    dates = pd.bdate_range("1999-01-04", periods=3000)      # ends 2010: no 2020, no 2022
    net = np.full(len(dates), 0.0002)
    vals = evaluate.regime_slices(net, dates)
    days = evaluate.regime_slice_days(dates)
    absent = [i for i, v in enumerate(vals) if np.isnan(v)]
    rep = gates.evaluate_gates(dict(PASSING_CAND, regime_slices=vals), PASSING_INC)
    check("G8 passes by absence: an uncovered window is NaN, and NaN is not a failure",
          absent == [2, 3] and days[2] == 0 and days[3] == 0 and rep["gates"]["G8"]["pass"],
          "windows %s uncovered by a series ending %s; G8 %s"
          % (absent, dates[-1].date(), rep["gates"]["G8"]["detail"]))
    hard = list(vals)
    hard[0] = config.GATE_REGIME_MAX_LOSS - 0.01
    check("...but a COVERED window below the floor still fails G8",
          not gates.evaluate_gates(dict(PASSING_CAND, regime_slices=hard),
                                   PASSING_INC)["gates"]["G8"]["pass"],
          "slice 0 at %.0f%% (floor %.0f%%)"
          % (100 * hard[0], 100 * config.GATE_REGIME_MAX_LOSS))

    rng_net = np.random.default_rng(config.SEED).normal(0.0006, 0.01, 1200)
    ci_a = evaluate.bootstrap_sharpe_ci(rng_net, n=400, rng=np.random.default_rng(99))
    ci_b = evaluate.bootstrap_sharpe_ci(rng_net, n=400, rng=np.random.default_rng(99))
    ci_c = evaluate.bootstrap_sharpe_ci(rng_net, n=400, rng=np.random.default_rng(100))
    check("the bootstrap CI is reproducible from a fixed rng (and only from it)",
          ci_a == ci_b and ci_a != ci_c and ci_a[0] < ci_a[1],
          "same seed [%+.3f, %+.3f] twice; a different seed gives [%+.3f, %+.3f]"
          % (ci_a[0], ci_a[1], ci_c[0], ci_c[1]))
    short = evaluate.bootstrap_sharpe_ci(rng_net[:config.SHARPE_MIN_OBS - 1], n=100,
                                         rng=np.random.default_rng(1))
    check("...and a series too short to have a Sharpe returns NaN, which fails G6",
          all(np.isnan(v) for v in short)
          and not gates.evaluate_gates(dict(PASSING_CAND, boot_ci_lo=short[0]),
                                       PASSING_INC)["gates"]["G6"]["pass"],
          "%d observations -> (nan, nan)" % (config.SHARPE_MIN_OBS - 1))

    # ── the report may not claim a gate verdict that was never reached ────────
    # THE WEEK AFTER A PROMOTION, the newest record on the champion's artifact is
    # not an evaluation: run_deepeval re-simulates the incumbent for G1 and G9 and
    # stores that four-key result (role, identity, window, Sharpe). No battery
    # runs on it and no gates are run against it — the ten gates are what a
    # CHALLENGER faces. Reported as-is, that record renders ten "not evaluated"
    # gate rows and the sentence "gate G4 fails on that alone", about gates nobody
    # ran, on the champion's own page. So the report falls back to the evidence
    # that promoted the genome and labels which generation it came from.
    tmp4 = tempfile.mkdtemp(prefix="arena_report_")
    try:
        arts4, out4 = os.path.join(tmp4, "artifacts"), os.path.join(tmp4, "output")
        rng = np.random.default_rng(config.SEED)
        g = gn.random_genome(rng, tuple("feat_%02d" % i for i in range(12)))
        entry = {"genome": g.to_dict(), "hash": g.hash(), "op": "mutate",
                 "parent_hash": "", "birth_gen": 0}
        dates = pd.bdate_range("2000-01-03", periods=400)
        f1 = {"dates": dates, "daily_net": rng.normal(0.0006, 0.01, len(dates)),
              "daily_gross": rng.normal(0.0007, 0.01, len(dates)),
              "turnover": np.abs(rng.normal(0.1, 0.01, len(dates))),
              "costs": np.abs(rng.normal(1.0, 0.1, len(dates))),
              "identity": PASSING_CAND["identity"], "generation": 4,
              "score": 0.5, "sharpe_prevault": 1.10, "n_days_prevault": len(dates),
              "n_features": len(g.signal.features), "first_active": 0,
              "regime_finite_frac": None, "n_fits": 0}
        promo_report = gates.evaluate_gates(PASSING_CAND, PASSING_INC)
        battery = dict(PASSING_CAND, hash=entry["hash"], family="momentum",
                       pbo_in_cohort=True, pbo_note="cohort of 40 genomes, 16 splits",
                       n_days_prevault=len(dates), vault_days=0,
                       gate_report=promo_report)
        registry.store_artifact(entry, f1, battery, artifact_dir=arts4)
        registry.promote(entry["hash"], 4, promo_report, "gates passed", state_dir=tmp4)
        # ...and the following week, the incumbent's bare re-simulation.
        resim = {"hash": entry["hash"], "role": "incumbent (re-simulated for G1 and G9)",
                 "identity": PASSING_CAND["identity"], "resimulated": True,
                 "window": (str(dates[0].date()), str(dates[-1].date())),
                 "sharpe": 1.07}
        registry.store_artifact(entry, dict(f1, generation=5), resim, artifact_dir=arts4)

        subject = reports.report_subject(5, state_dir=tmp4, artifact_dir=arts4)
        # verify.py never reads the shared price cache (module docstring); the only
        # line in build_report that would is the benchmark curve, which already
        # degrades to "absent" on a cache miss. The stub IS that miss path.
        real_curve = reports.benchmark_curve
        reports.benchmark_curve = lambda dates: pd.Series(dtype=float)
        try:
            path = reports.build_report(5, state_dir=tmp4, artifact_dir=arts4,
                                        output_dir=out4)
        finally:
            reports.benchmark_curve = real_curve
        with open(path) as f:
            text = f.read()
        g4_row = [ln for ln in text.splitlines() if ln.startswith("| G4 ")]
        check("a champion's re-simulation week reports the PROMOTION gates, labelled",
              subject.get("gates_generation") == 4
              and (subject.get("record") or {}).get("gate_report") is not None
              and "gates were last run in generation 4" in text
              and "not evaluated" not in text
              and len(g4_row) == 1 and "PASS" in g4_row[0],
              "subject falls back to the generation-%s record; G4 row: %s"
              % (subject.get("gates_generation"), g4_row[0].strip() if g4_row else "MISSING"))
        check("...and it does not claim gate G4 failed on a battery nobody ran",
              "gate G4 fails on that alone" not in text
              and "re-simulated for G1 and G9" in text
              and "1.10" in text,          # the promotion week's numbers, not blanks
              "the false PBO sentence is absent and the promotion evidence is quoted")

        # The same fallback must NOT fire when the newest record is a real
        # evaluation: a refused candidate's own gate report is the one to print.
        plain = reports.report_subject(4, state_dir=tmp4, artifact_dir=arts4)
        check("...and a week whose newest record IS an evaluation is untouched",
              "resim_record" not in plain and "gates_generation" not in plain
              and (plain["record"].get("gate_report") or {}).get("all_pass"),
              "generation 4 reports its own battery, with no fallback label")
    finally:
        shutil.rmtree(tmp4, ignore_errors=True)


# ── 8. trial ledger ────────────────────────────────────────────────────────────
def test_trial_ledger():
    """DESIGN check 8. The ledger is the input to every multiple-testing
    correction this project makes, so it has to be complete (k evaluations, k
    rows), idempotent (a resumed run re-appends nothing), and it has to actually
    move the statistic it feeds."""
    print("\n8. Trial ledger (append-only, idempotent, and it feeds DSR)")
    tmp = tempfile.mkdtemp(prefix="arena_verify_ledger_")
    saved = config.STATE_DIR
    config.STATE_DIR = tmp
    ledger.forget_cache()
    try:
        rng = np.random.default_rng(config.SEED)
        genomes = [gn.random_genome(rng, FEATURE_LIB) for _ in range(6)]
        # Mixed fidelities on purpose, including one genome that was screened and
        # dropped (F0 only) — it still exerted selection and still has to count.
        screened_only = gn.random_genome(rng, FEATURE_LIB)
        evals = [(g, fid) for g in genomes for fid in ("F0", "F1")] + [(screened_only, "F0")]
        written = sum(ledger.record_trial(0, g, fid, 0.4 + 0.02 * i, 0.3 + 0.02 * i,
                                          1200, "d" * 16, "p" * 16)
                      for i, (g, fid) in enumerate(evals))
        rows = len(ledger.read_ledger())
        check("k evaluations -> exactly k rows", written == len(evals) == rows,
              "%d evaluations, %d written, %d rows on disk" % (len(evals), written, rows))

        # A resumed run re-evaluates and re-records the same work: nothing new.
        again = sum(ledger.record_trial(0, g, fid, 0.0, 0.0, 1, "d" * 16, "p" * 16)
                    for g, fid in evals)
        # ...and a fresh process sees the same file, so the index must survive a
        # cold start too, not just the in-memory set.
        ledger.forget_cache()
        cold = sum(ledger.record_trial(0, g, fid, 0.0, 0.0, 1, "d" * 16, "p" * 16)
                   for g, fid in evals)
        check("identical re-run appends nothing (hot and cold)",
              again == 0 and cold == 0 and len(ledger.read_ledger()) == rows,
              "%d + %d new rows, still %d on disk" % (again, cold, len(ledger.read_ledger())))
        n_genomes = len(genomes) + 1                 # + the screened-and-dropped one
        check("n_trials counts distinct genomes, not rows",
              ledger.n_trials() == n_genomes,
              "%d genomes over %d rows (screens count: they exerted selection)"
              % (ledger.n_trials(), rows))

        # DSR's N must be every genome the search looked at, not every genome that
        # survived to F1 — otherwise the correction ignores exactly the trials that
        # made the survivor look good. Coupled by construction, pinned here.
        dsr_s = ledger.dsr_trial_sharpes()
        f1_s = ledger.f1_sharpes()
        best = {g.hash(): 0.3 + 0.02 * i for i, (g, fid) in enumerate(evals) if fid == "F1"}
        best[screened_only.hash()] = 0.3 + 0.02 * (len(evals) - 1)     # F0 is its best
        expect = np.array(sorted(best.values())) / np.sqrt(config.TRADING_DAYS_YEAR)
        check("dsr_trial_sharpes: one value per genome, N == n_trials()",
              len(dsr_s) == ledger.n_trials() == n_genomes and len(f1_s) == len(genomes),
              "%d DSR trials (= n_trials) vs %d F1 rows; the screened-and-dropped "
              "genome is in the first and not the second" % (len(dsr_s), len(f1_s)))
        check("dsr_trial_sharpes: best fidelity per genome, in DAILY units",
              np.allclose(np.sort(dsr_s), expect),
              "F1 row wins over the genome's F0 row; annualised /sqrt(%d) "
              "(max %.4f daily vs %.4f annualised)"
              % (config.TRADING_DAYS_YEAR, dsr_s.max(), f1_s.max()))
        check("trial_sr_std falls back below %d F1 rows" % config.TRIAL_SR_STD_MIN_ROWS,
              abs(ledger.trial_sr_std() - 1.0 / np.sqrt(n_genomes)) < 1e-12,
              "%d F1 rows -> 1/sqrt(%d) = %.4f"
              % (len(ledger.f1_sharpes()), n_genomes, ledger.trial_sr_std()))
        # The ledger stores ANNUALISED Sharpes; deflated_sharpe works in DAILY
        # ones. Get that wrong and gate G2 rejects everything forever while
        # looking healthy, so the conversion is pinned here.
        check("trial_sr_std(daily=True) is deflated_sharpe's unit",
              abs(ledger.trial_sr_std(daily=True)
                  - ledger.trial_sr_std() / np.sqrt(config.TRADING_DAYS_YEAR)) < 1e-12,
              "%.4f annualised -> %.4f daily"
              % (ledger.trial_sr_std(), ledger.trial_sr_std(daily=True)))

        # DSR must fall as the search widens. Hold the trial-Sharpe DISPERSION
        # exactly fixed and grow only the count, since that is the property being
        # tested — a resampled set would move sigma and N at once.
        ret = rng.normal(0.0006, 0.01, 2000)
        z = rng.normal(size=8192)
        z = (z - z.mean()) / z.std(ddof=1)              # unit dispersion, every prefix
        dsrs = []
        for n in (2, 4, 8, 32, 128, 1024, 8192):
            zz = z[:n]
            zz = (zz - zz.mean()) / zz.std(ddof=1)
            dsrs.append(evaluate.deflated_sharpe(ret, 0.05 * zz + 0.02)["dsr"])
        check("DSR is monotone non-increasing in the trial count",
              all(b <= a + 1e-12 for a, b in zip(dsrs, dsrs[1:])) and dsrs[-1] < dsrs[0],
              "N=2 -> %.3f ... N=8192 -> %.3f" % (dsrs[0], dsrs[-1]))

        # Vault access: counted, and deliberately not idempotent.
        check("vault starts unread", ledger.vault_trials() == 0)
        ledger.record_vault_access(genomes[0].hash(), "gate_G3")
        ledger.record_vault_access(genomes[1].hash(), "gate_G3")
        ledger.record_vault_access(genomes[0].hash(), "gate_G3")     # a second look
        check("record_vault_access increments vault_trials",
              ledger.vault_trials() == 2,
              "3 accesses by 2 distinct genomes -> vault_trials = %d"
              % ledger.vault_trials())

        # Every ledger row stamps config_hash, so a knob missing from that digest
        # is a row claiming like-for-like when it is not. The dangerous class is
        # the names arena REDECLARES on top of the sell_in_may re-export with the
        # SAME value (TRADING_DAYS_YEAR = 252, SEED = 12345): invisible to any
        # value-based test, and TRADING_DAYS_YEAR scales every Sharpe on record.
        sm = config.import_sibling("config", config.SELL_IN_MAY)
        shadowed = sorted(n for n in config._DECLARED_HERE            # noqa: SLF001
                          if n in vars(sm) and n not in config._CONFIG_HASH_SKIP  # noqa: SLF001
                          and config._canon(getattr(config, n)) is not None)      # noqa: SLF001
        covered = dict(config.config_hash_items())
        missing = [n for n in shadowed if n not in covered]
        check("config_hash covers every knob arena redeclares over a sibling's",
              not missing and "TRADING_DAYS_YEAR" in covered,
              "missing: %s" % ", ".join(missing) if missing else
              "%d shadowed (%s) all in the %d-setting digest %s"
              % (len(shadowed), ", ".join(shadowed), len(covered), config.config_hash()))

        # The returns matrix refuses to store anything the vault owns.
        dates = pd.bdate_range("2019-11-01", periods=60)             # crosses 2020-01-01
        bad = {genomes[0].hash(): {"dates": dates,
                                   "daily_net": np.zeros(60), "daily_gross": np.zeros(60),
                                   "turnover": np.zeros(60), "costs": np.zeros(60)}}
        try:
            ledger.write_returns_matrix(9999, bad)
            refused = False
        except ValueError:
            refused = True
        check("returns matrix refuses vault days", refused,
              "a %s..%s window is rejected before it can be written"
              % (dates[0].date(), dates[-1].date()))

        # ── the union-merge repair ────────────────────────────────────────────
        # .gitattributes gives state/*.csv git's union merge driver, because these
        # files are EOF-appended and the default driver turns every two-writer
        # append into a conflict an unattended runner cannot resolve — and an
        # unresolvable conflict costs a whole session's ledger rows. Union's price
        # is that a line both sides appended can survive twice, and that a HEADER
        # can be carried down mid-file when both sides created the file.
        #
        # Both are repaired at startup, and both have to be, for different reasons:
        # a duplicate row inflates n_trials (which deflates DSR — wrong direction,
        # but conservative), while a repeated header makes pandas read the column
        # names as data and silently degrades every numeric column to dtype object.
        path = ledger.ledger_path()
        original = open(path).read().splitlines()
        header, body = original[0], original[1:]
        merged = [header] + body[:3] + [header] + body[3:] + [body[-1]]   # 1 header + 1 dup
        with open(path, "w") as f:
            f.write("\n".join(merged) + "\n")
        before_dtype = pd.read_csv(path)["generation"].dtype
        removed = ledger.dedup_ledger(verbose=False)
        after = open(path).read().splitlines()
        after_dtype = pd.read_csv(path)["generation"].dtype
        check("dedup_ledger removes a union merge's duplicate row AND repeated header",
              removed == 2 and after == original
              and after.count(header) == 1 and len(after) == len(original),
              "%d line(s) removed from a %d-line merged file; %d rows restored "
              "byte-for-byte, header appears %d time(s)"
              % (removed, len(merged), len(after) - 1, after.count(header)))
        check("...and the repaired ledger reads back with numeric dtypes",
              before_dtype == object and after_dtype != object,
              "generation dtype %s before the repair (every numeric column degraded, "
              "and n_trials would count a row that is not a trial) -> %s after"
              % (before_dtype, after_dtype))
        check("dedup_ledger is a no-op on a clean ledger",
              ledger.dedup_ledger(verbose=False) == 0
              and open(path).read().splitlines() == original,
              "nothing removed on the second pass; an append-only file is only ever "
              "rewritten when a merge actually damaged it")

        # A row differing in ANY column is a different evaluation and must survive:
        # only byte-identical rows are merge artifacts.
        near = body[-1].rsplit(",", 1)[0] + ",99"
        with open(path, "w") as f:
            f.write("\n".join(original + [near]) + "\n")
        check("a near-duplicate row is NOT removed",
              ledger.dedup_ledger(verbose=False) == 0
              and len(open(path).read().splitlines()) == len(original) + 1,
              "one column differs, so it is a different evaluation and stays")
        with open(path, "w") as f:
            f.write("\n".join(original) + "\n")
        ledger.forget_cache()

        # ── the refresh tolerance gate's shape anomaly ────────────────────────
        # A refetch that comes back with a different row SHAPE (a column appeared
        # or vanished) has no numeric drift at all. The gate says so with inf, and
        # the tally must keep that out of both maxima: folding it in as 0.0 would
        # report the loudest event this gate can see as the quietest one.
        cache_tmp = tempfile.mkdtemp(prefix="arena_verify_tolgate_")
        try:
            csv_path = os.path.join(cache_tmp, "X.csv")
            cached = b"date,open,close\n2020-01-02,1.0,1.0\n2020-01-03,2.0,2.0\n"
            with open(csv_path, "wb") as f:
                f.write(cached)
            with open(csv_path, "w") as f:                 # the refetch grew a column
                f.write("date,open,close\n2020-01-02,1.0,1.0\n2020-01-03,2.0,2.0,9.0\n")
            d = run_generation._tolerance_gate(          # noqa: SLF001
                csv_path, cached, config.REFRESH_REL_TOL)
            stat = run_generation.note_decision(run_generation.new_refresh_stat(1), "X", d)
            check("a changed row shape reports shape-changed, never a drift of 0.0",
                  d["action"] == "accept-shape" and d["max_rel"] == float("inf")
                  and stat["shape_changed"] == 1 and stat["shape_symbols"] == ["X"]
                  and stat["max_rel_kept"] == 0.0 and stat["max_rel_restated"] == 0.0,
                  "action=%s max_rel=%s -> shape_changed=%d, and it stays out of both "
                  "maxima (kept %.2e, restated %.2e) rather than being logged as 0.0"
                  % (d["action"], d["max_rel"], stat["shape_changed"],
                     stat["max_rel_kept"], stat["max_rel_restated"]))

            # The ordinary branches still reach the right maximum, so the exclusion
            # above is specific to the anomaly and not a hole.
            kept = run_generation.note_decision(
                run_generation.new_refresh_stat(1), "Y",
                {"action": "unchanged", "max_rel": 3.15e-07, "restated": 0, "added": 0})
            restated = run_generation.note_decision(
                run_generation.new_refresh_stat(1), "Z",
                {"action": "restated", "max_rel": 2e-02, "restated": 7, "added": 0})
            check("...while a kept file and a restated one land in separate maxima",
                  kept["max_rel_kept"] == 3.15e-07 and kept["max_rel_restated"] == 0.0
                  and restated["max_rel_restated"] == 2e-02
                  and restated["max_rel_kept"] == 0.0
                  and restated["restated_symbols"] == ["Z"],
                  "sub-tolerance drift %.2e -> max_rel_kept; a %.0e restatement -> "
                  "max_rel_restated, never mixed into one number"
                  % (kept["max_rel_kept"], restated["max_rel_restated"]))
        finally:
            shutil.rmtree(cache_tmp, ignore_errors=True)
    finally:
        config.STATE_DIR = saved
        ledger.forget_cache()
        shutil.rmtree(tmp, ignore_errors=True)


# ── 9. genome operators ────────────────────────────────────────────────────────
# A stand-in feature library: verify.py never touches the cache, and the operators
# only ever treat these as opaque names.
FEATURE_LIB = tuple("feat_%02d" % i for i in range(37))


def _violations(g, lib) -> list:
    """Every way a genome can be out of spec. Empty list = legal genome."""
    B = gn.BOUNDS
    s, p, r = g.signal, g.portfolio, g.risk
    bad = []
    if s.family not in B["families"]:
        bad.append("family=%s" % s.family)
        return bad                                  # the rest is family-relative
    if s.horizon not in B["horizon"]:
        bad.append("horizon=%s" % s.horizon)
    if s.refit_days not in B["refit_days"]:
        bad.append("refit_days=%s" % s.refit_days)

    grids = B["params"][s.family]
    if set(dict(s.params)) != set(grids):
        bad.append("param keys %s" % sorted(dict(s.params)))
    for k, v in s.params:
        if k in grids and v not in grids[k]:
            bad.append("%s=%s" % (k, v))
    if tuple(s.params) != tuple(sorted(s.params)):
        bad.append("params unsorted")

    lo, hi = B["n_features"]
    if s.family in B["model_families"]:
        if not (lo <= len(s.features) <= hi):
            bad.append("n_features=%d" % len(s.features))
        if list(s.features) != sorted(set(s.features)):
            bad.append("features unsorted or duplicated")
        if any(f not in lib for f in s.features):
            bad.append("feature outside the library")
    elif s.features:
        bad.append("rule family carries %d features" % len(s.features))

    for name, value in (("n_long", p.n_long), ("n_short", p.n_short),
                        ("weighting", p.weighting), ("gross", p.gross),
                        ("vol_target", p.vol_target), ("rebalance_days", p.rebalance_days)):
        if value not in B["portfolio"][name]:
            bad.append("%s=%s" % (name, value))
    if p.n_long + p.n_short < B["min_positions"]:
        bad.append("n_long+n_short=%d" % (p.n_long + p.n_short))

    for name, value in (("stop_loss", r.stop_loss), ("trail_stop", r.trail_stop),
                        ("regime_filter", r.regime_filter), ("regime_scale", r.regime_scale),
                        ("dd_limit", r.dd_limit)):
        if value not in B["risk"][name]:
            bad.append("%s=%s" % (name, value))
    return bad


def test_genome_ops():
    print("\n9. Genome operators (20,000 seeded mutations + crossovers)")
    rng = np.random.default_rng(config.SEED)
    lib = FEATURE_LIB

    pool = [gn.random_genome(rng, lib) for _ in range(64)]
    illegal, broken_roundtrip, clones = [], [], 0
    families = {}
    n_mut = n_cross = 0
    g = pool[0]
    for i in range(10_000):
        parent = g
        g = gn.mutate(g, rng, lib)
        n_mut += 1
        clones += int(g.hash() == parent.hash())
        mate = pool[int(rng.integers(len(pool)))]
        child = gn.crossover(g, mate, rng)
        n_cross += 1
        for who in (g, child):
            v = _violations(who, lib)
            if v and len(illegal) < 5:
                illegal.append("%s: %s" % (who.hash(), ", ".join(v)))
            if gn.from_dict(who.to_dict()).hash() != who.hash() and len(broken_roundtrip) < 5:
                broken_roundtrip.append(who.hash())
            families[who.signal.family] = families.get(who.signal.family, 0) + 1
        g = child if i % 3 == 0 else g              # let crossover children breed too

    check("%d mutations + %d crossovers all inside BOUNDS" % (n_mut, n_cross),
          not illegal, "; ".join(illegal) if illegal else
          "%d genomes checked across %d families" % (2 * n_mut, len(families)))
    check("encode -> decode -> hash is identity", not broken_roundtrip,
          "; ".join(broken_roundtrip) if broken_roundtrip else "on every genome")
    check("every family reachable by mutation", len(families) == len(gn.BOUNDS["families"]),
          " ".join("%s=%d" % kv for kv in sorted(families.items())))
    check("mutation rarely returns the parent unchanged", clones / n_mut < 0.05,
          "%d clones in %d mutations (%.2f%%)" % (clones, n_mut, 100 * clones / n_mut))

    # Fixed inputs -> fixed stream, in this process and in any other. The golden
    # value pins the derivation itself: change how child seeds are built and every
    # lineage in the ledger silently becomes irreproducible.
    a1 = gn.child_rng(config.SEED, 7, "abcdef123456", 3).random()
    a2 = gn.child_rng(config.SEED, 7, "abcdef123456", 3).random()
    b1 = gn.child_rng(config.SEED, 7, "abcdef123456", 4).random()
    c1 = gn.child_rng(config.SEED, 8, "abcdef123456", 3).random()
    check("child_rng reproducible from (SEED, generation, parent, idx)",
          a1 == a2 and a1 != b1 and a1 != c1,
          "idx 3 -> %.9f twice; idx 4 -> %.9f; gen 8 -> %.9f" % (a1, b1, c1))
    check("stable_hash is process-independent (golden value)",
          gn.stable_hash(config.SEED, 7, "abcdef123456", 3) == 2198565245971884535,
          "%d" % gn.stable_hash(config.SEED, 7, "abcdef123456", 3))

    # The operators must not need a legal genome to hand back a legal one.
    wrecked = gn.Genome(
        signal=gn.SignalGene(family="hgb", horizon=99, refit_days=7,
                             features=("feat_00", "feat_00", "nope"),
                             params=(("max_depth", 12),)),
        portfolio=gn.PortfolioGene(n_long=0, n_short=1, weighting="nope", gross=9.0,
                                   vol_target=0.99, rebalance_days=3),
        risk=gn.RiskGene(stop_loss=0.99, trail_stop=0.99, regime_filter="nope",
                         regime_scale=9.0, dd_limit=0.99))
    fixed = gn._repair(wrecked, lib)                # noqa: SLF001 — the repair IS the test
    check("repair drags an out-of-spec genome back inside BOUNDS",
          not _violations(fixed, lib), "; ".join(_violations(fixed, lib)) or fixed.describe())


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


# ── 11. PBO sanity ─────────────────────────────────────────────────────────────
def test_pbo_sanity():
    """DESIGN check 11. CSCV asks: when you pick the in-sample winner out of a
    cohort, does it stay above the pack out of sample? On pure noise the answer is
    "no, about half the time" — the winner was a winner because it was lucky — and
    PBO should be high. Give the cohort one column with a real, persistent edge and
    PBO should drop AND that column should be the one being picked."""
    print("\n11. PBO sanity (CSCV on noise vs. a planted persistent edge)")
    rng = np.random.default_rng(config.SEED)
    R = rng.normal(0.0, 0.01, size=(500, 20))          # 500 days, 20 zero-skill configs

    noise = evaluate.pbo_cscv(R, S=config.PBO_SPLITS)
    check("pure noise reports a high probability of backtest overfitting",
          noise["pbo"] >= 0.4,
          "PBO = %.3f over %d splits; the IS-best config's median OOS Sharpe is %s"
          % (noise["pbo"], noise["n_splits"], noise["median_oos_sharpe_of_is_best"]))

    planted = 7
    R2 = R.copy()
    R2[:, planted] += 0.0025                           # a persistent edge, every day
    out = evaluate.pbo_cscv(R2, S=config.PBO_SPLITS)
    wins = sum(int(np.nanargmax(evaluate._sharpe_cols(tr))) == planted    # noqa: SLF001
               for tr, _te in evaluate._cscv_splits(R2, config.PBO_SPLITS))  # noqa: SLF001
    total = sum(1 for _ in evaluate._cscv_splits(R2, config.PBO_SPLITS))     # noqa: SLF001
    check("a real edge drives PBO down", out["pbo"] < 0.2,
          "PBO = %.3f, %.0f%% of splits keep the IS-best positive OOS"
          % (out["pbo"], 100 * out["frac_oos_positive"]))
    check("...and the planted column is the one being selected",
          wins > 0.5 * total,
          "column %d wins %d of %d in-sample splits" % (planted, wins, total))


# ── 12. the paper stage ────────────────────────────────────────────────────────
def _hist_row(generation, after, candidates, complete=1, before=""):
    """One state/deepeval_history.csv row, in the real schema's shape."""
    row = {c: "" for c in run_deepeval.HISTORY_COLUMNS}
    row.update({"generation": str(generation), "champion_hash_before": before,
                "champion_hash_after": after, "candidates": candidates,
                "complete": str(complete), "n_candidates": "1"})
    return row


class FakeBroker:
    """The broker's read surface, with a book. No alpaca-py, no network, no keys.

    verify.py never touches the network, so the adapter under test here is the
    INTERFACE run_paper depends on — positions() as a signed whole-share dict —
    rather than Alpaca's wire format. The real adapter's unarmed contract is
    checked separately below, and it is checkable precisely because an unarmed
    AlpacaPaperBroker imports nothing.
    """

    mode = "paper"

    def __init__(self, book=None, credentialed=True, is_open=False, status="accepted"):
        self.book = dict(book or {})
        self.credentialed = credentialed
        self.is_open = is_open
        self.status = status
        self.placed = []

    def positions(self):
        return dict(self.book) if self.credentialed else {}

    def clock(self):
        return None if not self.credentialed else {"is_open": self.is_open}

    def place(self, symbol, side, qty, client_order_id=None):
        self.placed.append((symbol, side, qty, client_order_id))
        return {"executed": True, "order_id": "fake-%d" % len(self.placed),
                "status": self.status, "client_order_id": client_order_id,
                "live": self.status not in broker_paper.DEAD_STATUSES}


def evidence_rows(sim_series):
    """A synthetic paper history that tracks its shadow, as evidence_table sees it.

    The go-live alert can only be trusted if it can FIRE, and it can only be shown
    to fire against a history that passes all five criteria. Built here rather
    than waiting 126 real sessions: paper returns are the sim's with a small
    execution wobble, slippage is inside the bound, tracking never trips.
    """
    rng = np.random.default_rng(config.SEED + 1)
    wobble = rng.normal(0.0, 0.00005, len(sim_series))       # ~0.5 bp/day of drift
    rows = []
    for i, s in enumerate(sim_series):
        p = float(s + wobble[i])
        rows.append({"date": "d%03d" % i, "sim_net": repr(float(s)),
                     "paper_net": repr(p),
                     "tracking_error_bps": repr((p - float(s)) * 1e4),
                     "fill_slippage_bps": repr(float(rng.normal(0.0, 2.0))),
                     "n_fills": 1})
    return run_paper.evidence_table(rows)


def test_paper_stage():
    """Phase 7. The three things that decide whether an order exists and what it
    is: the arming gate, the weights -> whole-share -> delta conversion, and the
    ledger's idempotency. Plus the one equivalence the whole design rests on —
    that the decision handed to a broker is the decision the sandbox scores.

    Offline by construction: a synthetic market, a fake broker, and synthetic
    deep-eval history. No alpaca-py import happens anywhere in this test.
    """
    print("\n12. Paper stage (arming gate, share deltas, ledger, live/sim parity)")

    # ── (a) the arming gate, as a truth table ─────────────────────────────────
    X, Z = "champ0000001", "other0000002"
    passed = "%s:1:none" % X
    armed_rows = [_hist_row(1, X, passed), _hist_row(2, X, "cand1:0:G5"),
                  _hist_row(3, X, "cand2:0:G2+G4")]
    cases = [
        ("3 consecutive complete entries, same champion, gates on record",
         armed_rows, X, True),
        ("a broken streak (another genome held the pointer in the middle)",
         [_hist_row(1, X, passed), _hist_row(2, Z, "%s:1:none" % Z), _hist_row(3, X, "")],
         X, False),
        ("the latest entries name a DIFFERENT champion",
         armed_rows, Z, False),
        ("the champion's own deep eval failed gates",
         [_hist_row(g, X, "%s:0:G2+G4" % X) for g in (1, 2, 3)], X, False),
        ("only 2 complete entries (the third ran out of budget)",
         [_hist_row(1, X, passed), _hist_row(2, X, ""), _hist_row(3, X, "", complete=0)],
         X, False),
        ("no champion pointer at all", armed_rows, None, False),
    ]
    for name, rows, champ, expect in cases:
        verdict = run_paper.arming_status(rows, champ)
        check("arming: %s -> %s" % (name, "ARMED" if expect else "unarmed"),
              verdict["armed"] == expect,
              verdict["reason"][:96])

    # A generation re-run appends a SECOND row; the trigger counts generations,
    # not rows, and the later row for a generation is the decision that stands.
    rerun = armed_rows + [_hist_row(3, Z, "%s:1:none" % Z)]
    check("arming: a re-run of a generation replaces that generation's entry",
          run_paper.complete_entries(rerun)[-1]["champion_hash_after"] == Z
          and len(run_paper.complete_entries(rerun)) == 3
          and not run_paper.arming_status(rerun, X)["armed"],
          "4 rows -> 3 entries, the last of generation 3 wins, and the champion is "
          "no longer armed")

    # The real adapter, unarmed: it constructs, refuses every write, and reads
    # EMPTY — which run_paper must never read as "flat".
    b = broker_paper.AlpacaPaperBroker(key=None, secret=None)
    refusals = [b.place("SPY", "buy", 3), b.close("SPY")]
    check("unarmed AlpacaPaperBroker refuses every write and imports no alpaca",
          all(r["executed"] is False and r["reason"] == broker_paper.UNARMED_REASON
              for r in refusals)
          and b.positions() == {} and b.account() is None
          and b.fills_since("2020-01-02") == []
          and "alpaca" not in __import__("sys").modules,
          "place/close -> %s; positions {} means UNKNOWN" % refusals[0]["reason"])

    # ── (b) weights -> whole shares -> order deltas ───────────────────────────
    # One flat $50 market so every number below is checkable by hand:
    # equity $15,000, per-name cap 20% = $3,000 = 60 shares, dust floor $500 = 10 sh.
    px, equity = 50.0, 15000.0
    market = flat_market(n_days=6, n_syms=6, price=px)
    env = MarketEnv(market, CostModel())
    close_t = market.close[-1]
    current = np.array([60.0, 0.0, 2.0, 60.0, 69.0, 0.0])
    target_w = np.array([-0.20, 0.20, 0.0, 0.199, 0.20, 0.0])
    tgt, w, _scale, _rej = run_paper.share_targets(env, target_w, close_t, equity)
    plan = run_paper.order_deltas(env, market, tgt, current, close_t, equity, "rebalance")
    got = {o["symbol"]: (o["side"], o["qty"], o["reason"]) for o in plan["orders"]}
    dropped = dict(plan["dropped"])

    check("a long -> short flip is ONE order for the whole distance",
          got.get("FLT0") == ("sell", 120, "rebalance"),
          "+60 shares to a -20%% target (-60 shares) -> %s" % (got.get("FLT0"),))
    check("a new position rounds to whole shares, truncated toward zero",
          got.get("FLT1") == ("buy", 60, "rebalance") and tgt[3] == 59.0,
          "20%% of $15,000 at $50 -> 60 sh; 19.9%% -> $%.0f/$50 = 59.7 -> %d sh"
          % (0.199 * equity, int(tgt[3])))
    check("a full exit is exempt from the dust filter",
          got.get("FLT2") == ("sell", 2, "rebalance"),
          "a $100 position is below the $%.0f minimum and still gets closed"
          % config.MIN_POSITION_USD)
    check("a sub-minimum REBALANCE is dropped as dust",
          "FLT3" not in got and dropped.get("FLT3") == "below_min_usd",
          "1 share = $50 < $%.0f, so the 60 -> 59 trim is not worth trading"
          % config.MIN_POSITION_USD)
    check("...but a position outside the per-name cap is repaired unfiltered",
          got.get("FLT4") == ("sell", 9, "risk_cap"),
          "69 sh = $3,450 = 23.0% of equity; the 9-share trim is $450 of dust and "
          "risk limits outrank the dust filter")
    book = plan["book_after"]
    val = book * px
    check("the resulting book is integral and inside every env cap",
          np.all(book == np.trunc(book))
          and np.max(np.abs(val)) <= config.MAX_NAME_WEIGHT * equity + 1e-9
          and np.abs(val).sum() <= config.MAX_GROSS_LEV * equity * (1 + config.LEV_EPS)
          and abs(val.sum()) <= config.MAX_NET_LEV * equity * (1 + config.LEV_EPS),
          "gross %.3fx net %+.3fx, max name %.1f%%"
          % (np.abs(val).sum() / equity, val.sum() / equity,
             100 * np.max(np.abs(val)) / equity))
    fake = FakeBroker({"FLT0": 60, "FLT2": 2, "FLT3": 60, "FLT4": 69})
    check("the same deltas come out of a broker's position dict",
          {s: int(q) for s, q in fake.positions().items() if q}
          == {"FLT0": 60, "FLT2": 2, "FLT3": 60, "FLT4": 69}
          and FakeBroker({}, credentialed=False).positions() == {},
          "an unarmed broker reports {} — unknown, not flat")

    # ── (c) the paper ledger is append-only AND idempotent ────────────────────
    tmp = tempfile.mkdtemp(prefix="arena_verify_paper_")
    try:
        rows = [{"date": "2026-08-17", "symbol": o["symbol"], "side": o["side"],
                 "qty": o["qty"], "fill_px": "", "order_id": "",
                 "client_order_id": "", "status": "intended", "broker_status": "",
                 "genome_hash": X, "reason": o["reason"]} for o in plan["orders"]]
        first = run_paper.append_ledger(rows, state_dir=tmp)
        second = run_paper.append_ledger(rows, state_dir=tmp)
        check("re-running a session on the same bar appends no duplicate intended rows",
              first["written"] == len(rows) and second["written"] == 0
              and second["skipped"] == len(rows)
              and len(run_paper.read_ledger(tmp)) == len(rows),
              "%d rows written, then %d written / %d skipped"
              % (first["written"], second["written"], second["skipped"]))

        fill = {"date": "2026-08-18", "symbol": "FLT1", "side": "buy", "qty": 60,
                "fill_px": 50.02, "order_id": "abc-123", "client_order_id": "",
                "status": "filled", "broker_status": "filled",
                "genome_hash": X, "reason": "fill"}
        run_paper.append_ledger([fill], state_dir=tmp)
        again = run_paper.append_ledger([dict(fill, status="filled")], state_dir=tmp)
        check("a fill is identified by its order_id, so an overlapping "
              "reconciliation window double-counts nothing",
              again["written"] == 0 and again["skipped"] == 1,
              "the same closed order read twice writes one row")

        # THE SUBMITTED -> FILL SEQUENCE. A submitted row already carries the
        # broker's order id, so an order-id key scoped to the whole file would
        # recognise the next morning's FILL of that order as "already recorded"
        # and drop it — leaving no fill price, no fill date and no terminal status
        # for exactly the orders this system places, with the reconcile line
        # reporting the loss as correct dedup. The key is scoped to fill rows.
        sub = {"date": "2026-08-19", "symbol": "FLT5", "side": "buy", "qty": 20,
               "fill_px": "", "order_id": "ord-777",
               "client_order_id": "arena-2026-08-19-FLT5-buy-20",
               "status": "submitted", "broker_status": "accepted",
               "genome_hash": X, "reason": "rebalance"}
        run_paper.append_ledger([sub], state_dir=tmp)
        its_fill = {"date": "2026-08-20", "symbol": "FLT5", "side": "buy", "qty": 20,
                    "fill_px": 50.11, "order_id": "ord-777", "client_order_id": "",
                    "status": "filled", "broker_status": "filled",
                    "genome_hash": X, "reason": "fill"}
        landed = run_paper.append_ledger([its_fill], state_dir=tmp)
        twice = run_paper.append_ledger([its_fill], state_dir=tmp)
        back = [r for r in run_paper.read_ledger(tmp) if r["order_id"] == "ord-777"]
        check("the realized fill of a SUBMITTED order is recorded, not eaten by its "
              "own order id",
              landed["written"] == 1 and twice["written"] == 0 and len(back) == 2
              and {r["status"] for r in back} == {"submitted", "filled"}
              and [r["fill_px"] for r in back if r["status"] == "filled"] == ["50.11"],
              "order ord-777 has both rows: submitted 2026-08-19 (no price) and "
              "filled 2026-08-20 @ 50.11; a second read of the fill writes nothing")

        # The double-entry guard, local half.
        already = run_paper.submitted_today(run_paper.read_ledger(tmp), "2026-08-19")
        check("a symbol already submitted for this session is recognised",
              set(already) == {"FLT5"} and already["FLT5"]["qty"] == "20"
              and run_paper.submitted_today(run_paper.read_ledger(tmp), "2026-08-21") == {},
              "an OPG order rests until the auction, so positions() is unchanged "
              "and the recomputed order would DOUBLE the position")
        check("...and the broker-enforced half is deterministic in the decision",
              run_paper.client_order_id("2026-08-19", "FLT5", "buy", 20)
              == "arena-2026-08-19-FLT5-buy-20"
              and run_paper.client_order_id("2026-08-19", "FLT5", "buy", 20)
              == run_paper.client_order_id("2026-08-19", "FLT5", "BUY", 20.0)
              and run_paper.client_order_id("2026-08-19", "FLT5", "buy", 21)
              != run_paper.client_order_id("2026-08-19", "FLT5", "buy", 20),
              "Alpaca rejects a duplicate client_order_id, so this survives a lost "
              "ledger and a second machine; a different size is a different order")

        changed = [dict(rows[0], qty=int(rows[0]["qty"]) + 5)]
        check("a decision that genuinely CHANGED within a session is appended, not "
              "swallowed",
              run_paper.append_ledger(changed, state_dir=tmp)["written"] == 1,
              "same date and symbol, different size -> a second row; the last row "
              "for a (date, symbol) is the live one")

        shadow = {"date": "2026-08-18", "genome_hash": X, "sim_net": "0.001",
                  "paper_net": "0.0007", "tracking_error_bps": "-3.0",
                  "fill_slippage_bps": "1.5", "n_fills": 1, "paper_equity": "15010",
                  "paper_last_equity": "15000", "sim_equity": "15015"}
        check("one shadow row per session date",
              run_paper.append_shadow(shadow, state_dir=tmp)
              and not run_paper.append_shadow(shadow, state_dir=tmp)
              and len(run_paper.read_shadow(tmp)) == 1,
              "a re-run on the same bar re-measures nothing")

        # A file written under an older column list must not be appended into.
        with open(run_paper.ledger_path(tmp), "w") as f:
            f.write("date,symbol,qty\n2026-08-17,FLT1,60\n")
        try:
            run_paper.append_ledger([fill], state_dir=tmp)
            refused = False
        except ValueError:
            refused = True
        check("appending into a differently-shaped ledger is refused, not misaligned",
              refused, "csv.writer appends positionally; a silent shift would "
                       "misreport every order ever placed")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # The shadow may only be measured once the session it measures has ENDED.
    for name, clock, expect in [("the market is open", {"is_open": True}, False),
                                ("the market is closed", {"is_open": False}, True),
                                ("the clock could not be read", None, False)]:
        session_ok, why = run_paper.shadow_session_ok(clock)
        check("shadow row when %s -> %s" % (name, "written" if expect else "SKIPPED"),
              session_ok == expect, (why or "pre-open: a full session to compare")[:88])
    check("an unarmed broker has no exchange clock either",
          FakeBroker({}, credentialed=False).clock() is None
          and not run_paper.shadow_session_ok(FakeBroker({}, credentialed=False).clock())[0],
          "no account, no clock, no row — rather than a row measuring a partial day")

    # The go-live band: DESIGN's fourth criterion, computed rather than stubbed.
    rng = np.random.default_rng(config.SEED)
    sim_series = rng.normal(0.0004, 0.008, 300)
    inside = run_paper.cumulative_band(sim_series,
                                       float(np.prod(1.0 + sim_series) - 1.0))
    outside = run_paper.cumulative_band(sim_series, -0.95)
    short_series = run_paper.cumulative_band(sim_series[:10], 0.01)
    check("the cumulative paper return is tested against the shadow's bootstrap band",
          inside["inside"] is True and outside["inside"] is False
          and short_series["inside"] is None,
          "band [%+.3f, %+.3f] over %d block resamples: the shadow's own cumulative "
          "is inside, -95%% is outside, and too few days answers None (not False)"
          % (inside["lo"], inside["hi"], config.BOOT_ITERS))
    table = evidence_rows(sim_series)
    check("...so the go-live path is reachable rather than dead code",
          table["all_pass"] is True and len(table["rows"]) == 5
          and all(r[3] for r in table["rows"]),
          "a synthetic 300-session paper history that tracks its shadow passes all "
          "five criteria — the alert can fire, which is the only way it can be tested")

    # Breach spells: the anti-spam key moves only when a NEW divergence starts.
    limit = config.PAPER_MAX_TE_BPS
    series = [("d1", 2.0), ("d2", limit + 5), ("d3", limit + 9), ("d4", 1.0),
              ("d5", limit + 1)]
    srows = [{"date": d, "tracking_error_bps": str(v)} for d, v in series]
    check("a breach spell is keyed on the day it began, so one divergence is one alert",
          run_paper.breach_spell_start(srows, limit) == "d5"
          and run_paper.breach_spell_start(srows[:3], limit) == "d2"
          and run_paper.breach_spell_start(srows[:1], limit) is None,
          "d2-d3 is one spell, d4 clears it, d5 opens a new one that speaks again")

    # ── (d) the live decision IS the sandbox's decision ───────────────────────
    # run_episode decides at bar t and fills at t+1, so it stops deciding one bar
    # before the market ends — and a paper session stands exactly on that last bar
    # with the next open ahead of it. live_decision executes that one extra loop
    # iteration. If it drifted from the loop by so much as a rounding rule, the
    # tracking error would measure arena's own arithmetic instead of the broker's
    # execution, so the equivalence is pinned end to end: the orders the live path
    # produces at the second-to-last bar must BE the fills the ordinary episode
    # makes at the last one.
    full = synthetic_market(n_days=320, n_syms=6, seed=config.SEED)
    g = _genome("mom_rule", 21, [], (("lookback", 21), ("skip", 0)), rebalance=1)
    start = full.dates[200]

    # ONE reference episode over the whole window. It is forward-only and
    # deterministic, so its fills on bar T are exactly the fills an episode that
    # ENDED at T would have made — which is what the live path has to reproduce
    # from the market truncated one bar earlier.
    log = []
    StrategyAgent(g, full, CostModel()).run_episode(
        env_start=start, env_end=full.dates[-1], decision_log=log)
    by_date = {}
    for row in log:
        if row["side"] == "carry":
            continue
        by_date.setdefault(row["date"], {})
        by_date[row["date"]][row["symbol"]] = \
            by_date[row["date"]].get(row["symbol"], 0) + row["shares"]

    # Six bars, taken from the end and deliberately mixed: bars the sandbox
    # traded on and bars it did not, because "the live path also orders NOTHING
    # when the sandbox would have" is half the claim.
    candidates = [t for t in range(len(full) - 12, len(full))]
    traded = agreed = 0
    mismatch = None
    for T in candidates:
        short = datafeed.MarketData(full.dates[:T], full.symbols, full.open[:T],
                                    full.close[:T], full.volume[:T])
        agent = StrategyAgent(g, short, CostModel())
        sim_log, audit = [], []
        agent.run_episode(env_start=start, env_end=short.dates[-1],
                          decision_log=sim_log, fit_audit=audit)
        shares, cash = run_paper.replay_book(sim_log, short)
        eq = cash + float(np.sum(np.where(shares != 0.0, shares * short.close[-1], 0.0)))
        peak = float(np.max(agent._equity))                           # noqa: SLF001
        live = run_paper.live_decision(agent, short, shares, eq, peak,
                                       int(short.pos(start, "left")), audit)
        env2 = MarketEnv(short, CostModel())
        tgt2, _w2, _s2, _r2 = run_paper.share_targets(env2, live["target_w"],
                                                      short.close[-1], eq)
        plan2 = run_paper.order_deltas(env2, short, tgt2, shares, short.close[-1], eq,
                                       live["reason"])
        live_orders = {o["symbol"]: (1 if o["side"] == "buy" else -1) * o["qty"]
                       for o in plan2["orders"]}
        want = {s: int(round(v)) for s, v in by_date.get(full.dates[T], {}).items() if v}
        traded += bool(want)
        if live_orders == want:
            agreed += 1
        elif mismatch is None:
            mismatch = "%s: live %s vs episode %s" % (full.dates[T].date(),
                                                      sorted(live_orders.items()),
                                                      sorted(want.items()))
        last = (eq, float(agent._equity[-1]))                         # noqa: SLF001
    check("the live decision reproduces the sandbox's own next-bar orders exactly",
          agreed == len(candidates) and traded >= 2,
          "%d/%d bars agree, %d of them with actual trades%s"
          % (agreed, len(candidates), traded, "" if mismatch is None else
             " — first mismatch %s" % mismatch))
    check("...and the log replay agrees with the episode's own equity",
          abs(last[0] - last[1]) <= config.ACCOUNT_TOL * 10,
          "replayed $%.2f vs episode $%.2f" % last)


def main() -> int:
    test_planted_leak()
    test_determinism()
    test_accounting()
    test_fill_timing()
    test_streaming_purge()
    test_cost_linearity()
    test_gates()
    test_trial_ledger()
    test_genome_ops()
    test_no_wallclock()
    test_pbo_sanity()
    test_paper_stage()
    print("\nVERIFY:", "ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
