"""
The honest-evaluation ladder: F0 screen, F1 full walk-forward, the F2 battery, and
the two multiple-testing statistics the gates read.

    screen(genome, market, cost)     -> {"score", "era_sharpes", "n_features", "n_days"}
    full_eval(genome, market, cost)  -> {"score", "sharpe_prevault", "daily_net", ...}
    deflated_sharpe(returns, all_sharpes)
    pbo_cscv(R, S)

    # F2 (Phase 5) — one candidate at a time, read by gates.py:
    bootstrap_sharpe_ci(daily_net, ...)        -> (lo, hi)          G6
    cpcv_paths(genome, market, cost, ...)      -> 28 path Sharpes   G5
    regime_slices(daily_net, dates)            -> 4 cumulative returns  G8
    ruin_mc(daily_net, rng)                    -> P(deep drawdown)  G10
    rolling_window_wins(cand, inc, ...)        -> win fraction      G9

THE LADDER EXISTS BECAUSE COMPUTE IS THE BINDING CONSTRAINT. A full-history
episode for one hgb genome costs minutes; a 64-genome population cannot pay that
64 times a night. F0 buys a cheap RANKING (three five-year eras, a 60-symbol
point-in-time universe, weekly rebalancing, yearly refits) and F1 pays for a real
MEASUREMENT on the survivors. F0's number is therefore a screen, never a result:
it is recorded at fidelity="F0" in the trial ledger — where it still counts as a
trial, because it still exerted selection — and nothing promotes on it.

THE VAULT. Fitness and selection may only see days before config.VAULT_START.
This module is where that is enforced, structurally:

  • F0 cannot reach the vault at all: every era in config.SCREEN_ERAS ends before
    VAULT_START, asserted at import (`_assert_eras_prevault`), not assumed.
  • F1 simulates the whole history but SPLITS the daily series at VAULT_START and
    returns the two halves under different names. Every unprefixed array it hands
    back (`daily_net`, `daily_gross`, `turnover`, `costs`, `dates`) is PRE-VAULT
    ONLY. The vault leaves this function through exactly one key,
    `vault_daily_net`, and nothing in this file reads it.

  GREP-ABLE INVARIANT: search the repo for `vault_daily_net` / `vault_`. Outside
  run_deepeval.py (the gate runner, which logs every access through
  ledger.record_vault_access before reading a single vault day) every hit must be
  a store or a pass-through — never an input to a comparison, a sort, a Sharpe, or
  a score. If that stops being true, the vault is gone and the last six years of
  data stop being evidence.

WHY F0 AND F1 CAN DISAGREE, AND WHY THAT IS NOT A BUG: F0 overrides the genome's
rebalance and refit cadence and trades a different universe over different years.
It is a different question asked of the same genome. The ledger records the
genome's true hash on both rows precisely so the disagreement stays visible.

deflated_sharpe() and pbo_cscv() (with their helpers _sharpe_cols and
_cscv_splits) are COPIED VERBATIM from ~/futures_bot/evaluate.py:200-278 — pure
functions, no futures_bot import (its `config` module would clash with arena's on
sys.path). Bailey/Borwein/López de Prado/Zhu. futures_bot's known flaw — selecting
in-sample on GROSS P&L and reporting OOS net — is not inherited: everything here
scores `daily_net`.
"""
from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import kurtosis, norm, skew

import config                       # FIRST: puts the siblings on sys.path
import datafeed
import features as arena_features
from env import CostModel
from strategy import StrategyAgent


def _assert_eras_prevault() -> None:
    """The F0 screen must be structurally incapable of touching the vault."""
    vault = pd.Timestamp(config.VAULT_START)
    for start, end in config.SCREEN_ERAS:
        if pd.Timestamp(end) >= vault:
            raise ValueError("SCREEN_ERAS entry %s..%s reaches the vault (%s): the F0 "
                             "screen may only see pre-vault days" % (start, end, config.VAULT_START))


_assert_eras_prevault()


# ── the selection statistic ────────────────────────────────────────────────────
def sharpe(daily_net) -> float:
    """Annualised Sharpe of a daily NET return series, for SELECTION.

    Returns 0.0 — not NaN — on anything degenerate: fewer than
    config.SHARPE_MIN_OBS finite observations, zero variance, or a series that is
    all NaN. Two reasons, both deliberate:

      • Selection has to order candidates. A NaN sorts unpredictably and an
        exception kills a generation over one bad genome; 0.0 says "no evidence"
        and, since promotion needs a positive Sharpe through ten gates, a 0 can
        never promote anything. The failure mode is a missed candidate, which is
        the honest direction to fail in.
      • The screening windows are SHORT (a five-year era, minus whatever warm-up
        a model family needs before its first fit). Sixty days is already thin
        for a Sharpe; below that the number is noise wearing a decimal point.

    strategy.sharpe() is the REPORTING counterpart and returns NaN instead — it
    is read by humans, who should see "undefined", not "zero".
    """
    r = np.asarray(daily_net, dtype=np.float64).ravel()
    r = r[np.isfinite(r)]
    if len(r) < config.SHARPE_MIN_OBS:
        return 0.0
    sd = float(r.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0.0:
        return 0.0
    return float(r.mean() / sd * np.sqrt(config.TRADING_DAYS_YEAR))


def _first_active(dates, fit_audit, is_model: bool, warmup: int) -> int:
    """First index of `dates` whose return is worth scoring.

    Model families: everything before the FIRST successful refit is a flat book —
    `_scores` returns NaN until a model exists — and scoring those days would
    shrink a Sharpe by roughly sqrt(active fraction) for a mechanical reason,
    penalising exactly the families that need a training window. So scoring starts
    the bar after the first fit, read off the episode's own audit rather than
    computed from a rule that could drift away from strategy.py's.

    Rule families: they trade from bar one, so `warmup` is the caller's choice of
    a comparable start (WF_MIN_TRAIN_DAYS on the full history, 0 inside an F0 era
    — where a model can already fit at the era's first bar off the pre-era
    history, so a rule needs no extra handicap).
    """
    # Only the MAIN book gates scoring: a bear engine's fits (signal_bear) are
    # interleaved in the same audit, and a bear book that fits first must not
    # start the scoring clock for a main model that cannot act yet.
    fit_audit = [r for r in fit_audit if r.get("engine", "main") == "main"]
    if not is_model:
        return int(min(warmup, len(dates)))
    if not fit_audit:
        return len(dates)                    # never fitted: nothing here is evidence
    return int(dates.searchsorted(fit_audit[0]["fit_date"], side="right"))


# ── F0: the screen ─────────────────────────────────────────────────────────────
def _era_universe(market, era_start, n: int) -> list:
    """The `n` most liquid symbols AS OF `era_start` — point-in-time, by trailing
    median dollar volume over the bars strictly BEFORE the era's first bar.

    Ranking on the full sample instead (the obvious shortcut, and what the Phase-2
    milestone does for its one-off print) would hand the 1997 era the names that
    turned out to be liquid in 2026 — a survivorship leak layered on top of the
    one this project already discloses. The median, not the mean, so one halted or
    one frenzied session cannot buy a symbol its way in.

    The benchmark is always included: it is the anchor for the spy_200dma regime
    gene, and StrategyAgent refuses to run such a genome without it.
    """
    i0 = market.pos(era_start, "left")
    lo = max(0, i0 - config.SCREEN_LIQUIDITY_DAYS)
    win = market.dollar_vol[lo:i0]
    finite = np.isfinite(win)
    enough = finite.sum(axis=0) >= config.SCREEN_LIQUIDITY_MIN_BARS
    med = np.full(win.shape[1], -np.inf)
    if enough.any():
        med[enough] = np.nanmedian(win[:, enough], axis=0)
    order = [j for j in np.argsort(-med, kind="stable")[:n] if np.isfinite(med[j])]
    keep = {market.symbols[j] for j in order}
    if config.BENCHMARK in market.symbols:
        keep.add(config.BENCHMARK)
    return sorted(keep)


def subset_market(market, symbols):
    """The same market with a column subset — same calendar, same bars, same panel.

    The feature columns are SLICED from the parent panel, not rebuilt. Two
    consequences worth stating out loud:
      • Nothing here is a look-ahead: every panel value is point-in-time by
        construction (features.py delegates to signal_lab's asof-truncated
        builder), and taking a subset of columns cannot make a row see its future.
      • The cross-sectional columns (xs_mom_63, xs_rs_spy) are still ranked over
        the FULL universe the panel was built from, not over the 60 names the
        screen trades. That is more information than a 60-symbol panel would
        carry, and it is the same information in F0 and F1 — but it does mean an
        F0 era is not identical to "arena with a 60-symbol universe".
    """
    wanted = set(symbols)
    idx = [j for j, s in enumerate(market.symbols) if s in wanted]
    sub = datafeed.MarketData(market.dates, [market.symbols[j] for j in idx],
                              market.open[:, idx], market.close[:, idx],
                              market.volume[:, idx])
    names = getattr(market, "feature_names", ())
    if names:
        for name in names:
            arr = market.features[name][:, idx]
            arr.setflags(write=False)        # history is immutable, features too
            sub.features[name] = arr
        sub.feature_names = names
        sub.panel_hash = arena_features.panel_hash(sub.data_hash, names, sub.features)
    return sub


def screen_markets(market) -> list:
    """One (era_start, era_end, sub-market) per config.SCREEN_ERAS.

    Memoised on the MarketData instance, keyed by the era/universe settings, for
    the same reason strategy.py memoises its rolling statistics: this depends on
    the bars and the config, never on the genome, and a 64-genome screen would
    otherwise rebuild and re-hash it 64 times. run_generation builds them once in
    the parent so the workers receive three 60-symbol panels instead of the whole
    120-symbol one.

    The sub-market keeps the FULL date range on purpose: the episode window is the
    era, but a model refitting at the era's first bar trains on everything before
    it, which is where a five-year era gets a four-year training set from.
    """
    key = (tuple(tuple(e) for e in config.SCREEN_ERAS), config.SCREEN_UNIVERSE_N)
    memo = getattr(market, "_era_memo", None)
    if memo is not None and memo[0] == key:
        return memo[1]
    out = []
    for start, end in config.SCREEN_ERAS:
        syms = _era_universe(market, start, config.SCREEN_UNIVERSE_N)
        out.append((start, end, subset_market(market, syms)))
    market._era_memo = (key, out)
    return out


def _screen_genome(genome):
    """The genome as F0 simulates it: same signal, coarser cadence.

    The OVERRIDES ARE THE SCREEN'S, NOT THE GENOME'S. The returned object is a
    throwaway; its hash is never recorded anywhere. The trial ledger carries the
    real genome's hash on the F0 row, because the real genome is what was screened
    and what the selection acted on.
    """
    p = genome.portfolio
    return replace(
        genome,
        signal=replace(genome.signal, refit_days=config.SCREEN_REFIT_DAYS),
        portfolio=replace(p, rebalance_days=max(config.SCREEN_MIN_REBALANCE_DAYS,
                                                p.rebalance_days)))


def screen(genome, market, cost=None, era_markets=None) -> dict:
    """F0. Mean net Sharpe across the three eras, minus the complexity tax.

    `market` is the full panel; pass `era_markets` from screen_markets() instead
    when they are already built (run_generation does, so the workers are shipped
    three 60-symbol panels rather than the whole one), in which case `market` is
    unused and may be None.

    Each era is one episode of the coarsened genome over that era's window, on
    that era's point-in-time 60-symbol universe. An era whose scored window is too
    short to mean anything contributes 0.0 (see `sharpe`) rather than a lucky
    number — which is how a model family that could not fit inside an era (the
    1997 era only reaches WF_MIN_TRAIN_DAYS of history in 1999) is treated as
    "no evidence" rather than "no skill".

    Score = min(era Sharpes) − PARSIMONY_PENALTY × n_features — the WORST era,
    so a screen cannot be carried by one golden regime. The tax is on the
    GENOME's feature count, not the screened copy's — they are the same, but the
    distinction is the rule: F0 may change how a genome is simulated, never what
    it is.
    """
    cost = cost if cost is not None else CostModel()
    eras = era_markets if era_markets is not None else screen_markets(market)
    coarse = _screen_genome(genome)

    era_sharpes, n_days = [], 0
    for start, end, sub in eras:
        audit: list = []
        res = StrategyAgent(coarse, sub, cost).run_episode(env_start=start, env_end=end,
                                                           fit_audit=audit)
        first = _first_active(res["dates"], audit, genome.is_model, 0)
        scored = res["daily_net"][first:]
        era_sharpes.append(sharpe(scored))
        n_days += len(scored)

    n_features = len(genome.signal.features)
    return {"score": float(np.min(era_sharpes)) - config.PARSIMONY_PENALTY * n_features,
            "era_sharpes": [float(s) for s in era_sharpes],
            "n_features": n_features,
            "n_days": int(n_days)}


# ── F1: the full anchored walk-forward ─────────────────────────────────────────
def robust_score(daily_net, dates, n_features) -> float:
    """Selection currency for F1: the FITNESS_QUANTILE quantile of rolling
    FITNESS_WINDOW_DAYS-day Sharpes (stepped FITNESS_WINDOW_STEP), minus the
    parsimony tax; falls back to the full-span Sharpe when fewer than 4 windows
    fit. A strategy carried by one golden regime scores its bad quartile — which
    is what gates G8/G9 will measure anyway, so selection now optimizes what the
    gates test. `sharpe_prevault` is unchanged: reports and the hall of fame
    keep the plain full-span number."""
    r = np.asarray(daily_net, dtype=np.float64)
    w, s = int(config.FITNESS_WINDOW_DAYS), int(config.FITNESS_WINDOW_STEP)
    srs = [sharpe(r[i:i + w]) for i in range(0, max(0, len(r) - w) + 1, s)
           if i + w <= len(r)]
    base = (float(np.quantile(srs, config.FITNESS_QUANTILE))
            if len(srs) >= 4 else sharpe(r))
    return base - config.PARSIMONY_PENALTY * int(n_features)


def full_eval(genome, market, cost=None) -> dict:
    """F1. One full-history episode, the genome's own genes, split at the vault.

    The episode runs DATA_START -> end of the calendar over the full universe, with
    the genome's real rebalance and refit cadence. Its daily series is then cut in
    two at config.VAULT_START:

      returned unprefixed   PRE-VAULT, from the first scored bar to the vault.
                            This is the only thing that may be scored, ranked,
                            selected on, or written to a returns matrix.
      vault_daily_net       everything from the vault onward. Returned so Phase 5's
                            gate stack can ask for it through a counted access —
                            and read by nothing else, ever.

    `score` = robust_score(pre-vault daily_net) — the bad quartile of rolling
    3-year Sharpes minus the parsimony tax, the same currency as the F0 score so
    the two ladders are at least comparable in units.

    Scoring starts at the episode's own first active bar (see `_first_active`), so
    a model's warm-up and a rule's WF_MIN_TRAIN_DAYS handicap are both read off
    the run rather than hardcoded to a calendar date.
    """
    cost = cost if cost is not None else CostModel()
    audit: list = []
    res = StrategyAgent(genome, market, cost).run_episode(fit_audit=audit)

    dates = res["dates"]
    first = _first_active(dates, audit, genome.is_model, config.WF_MIN_TRAIN_DAYS)
    vault_i = int(dates.searchsorted(pd.Timestamp(config.VAULT_START)))
    pre = slice(first, vault_i)                     # scored
    post = slice(max(first, vault_i), None)         # stored, never scored

    sr = sharpe(res["daily_net"][pre])
    n_features = len(genome.signal.features)
    return {"score": robust_score(res["daily_net"][pre], dates[pre], n_features),
            "sharpe_prevault": sr,
            "n_days_prevault": int(max(0, vault_i - first)),
            "daily_net": res["daily_net"][pre],
            "daily_gross": res["daily_gross"][pre],
            "turnover": res["turnover"][pre],
            "costs": res["costs"][pre],
            "dates": dates[pre],
            "vault_daily_net": res["daily_net"][post],
            "fit_audit": audit,
            "n_features": n_features,
            "first_active": int(first),
            "regime_finite_frac": res["regime_finite_frac"]}


# ── F2: the deep-evaluation battery (docs/DESIGN.md "Weekly F2 deep eval") ─────
# NOTHING HERE MAY TOUCH THE VAULT ON ITS OWN. Every function below takes a return
# series it is handed; run_deepeval.py is the only caller, and it logs a
# vault-access row (ledger.record_vault_access, reason="gate_eval") for every
# post-VAULT_START day it passes in. Keeping the accounting at the call site is
# what makes `grep record_vault_access` a complete answer to "who looked".
def bootstrap_sharpe_ci(daily_net, n=None, block=None, alpha=0.05, rng=None) -> tuple:
    """Gate G6. Circular block bootstrap CI of the ANNUALISED net Sharpe.

    Blocks, not single days: daily strategy returns are autocorrelated (a held
    book, a vol-target overlay and a drawdown brake all persist across days), and
    an iid bootstrap would resample that structure away and report a confidence
    interval far too narrow. `block` is config.BOOT_BLOCK (~a month). CIRCULAR so
    every observation has the same chance of being drawn — a non-circular block
    bootstrap under-samples both ends of the series.

    This is the plain PERCENTILE interval, not BCa: the gate only reads the lower
    bound's SIGN, and a bias-corrected interval would move it by less than the
    Monte Carlo noise of 5,000 resamples. Returns (nan, nan) on a series too short
    or too degenerate to have a Sharpe at all — which fails G6, the honest
    direction.

    Deterministic given `rng` (run_deepeval derives one per candidate from
    genome.child_rng(SEED, generation, "boot", i)).
    """
    n = int(config.BOOT_ITERS if n is None else n)
    block = int(config.BOOT_BLOCK if block is None else block)
    rng = rng if rng is not None else np.random.default_rng(config.SEED)

    r = np.asarray(daily_net, dtype=np.float64).ravel()
    r = r[np.isfinite(r)]
    T = r.size
    if T < config.SHARPE_MIN_OBS or r.std(ddof=1) <= 0.0:
        return float("nan"), float("nan")

    block = max(1, min(block, T))
    n_blocks = int(np.ceil(T / block))
    offsets = np.arange(block)
    ann = np.sqrt(config.TRADING_DAYS_YEAR)
    out = np.empty(n, dtype=np.float64)
    # Chunked so the index matrix stays a few MB whatever BOOT_ITERS x T is.
    chunk = max(1, int(2_000_000 // max(n_blocks * block, 1)))
    for lo in range(0, n, chunk):
        m = min(chunk, n - lo)
        starts = rng.integers(T, size=(m, n_blocks, 1))
        idx = (starts + offsets).reshape(m, -1)[:, :T] % T
        sample = r[idx]
        sd = sample.std(axis=1, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[lo:lo + m] = np.where(sd > 0, sample.mean(axis=1) / sd * ann, np.nan)
    out = out[np.isfinite(out)]
    if out.size == 0:
        return float("nan"), float("nan")
    return (float(np.quantile(out, alpha / 2.0)),
            float(np.quantile(out, 1.0 - alpha / 2.0)))


def cpcv_blocks(market, n_groups=None) -> list:
    """The pre-vault SCORING span cut into `n_groups` contiguous blocks.

    [(first_bar, last_bar), ...] as inclusive market-date indices. The span starts
    where full_eval starts scoring a rule family (config.WF_MIN_TRAIN_DAYS bars
    into the episode, i.e. market bar WF_MIN_TRAIN_DAYS + 1 — an episode's first
    return is dated at bar 1) and ends at config.VAULT_START. Genome-independent
    on purpose: the 28 paths of two candidates have to be the same 28 windows, or
    comparing their path Sharpes compares calendars instead of strategies.
    """
    n_groups = int(config.CPCV_GROUPS if n_groups is None else n_groups)
    start = config.WF_MIN_TRAIN_DAYS + 1
    end = int(market.dates.searchsorted(pd.Timestamp(config.VAULT_START)))
    if end - start < n_groups * config.SHARPE_MIN_OBS:
        raise ValueError("pre-vault span %d..%d is too short for %d CPCV blocks of at "
                         "least %d bars" % (start, end, n_groups, config.SHARPE_MIN_OBS))
    return [(int(g[0]), int(g[-1])) for g in
            np.array_split(np.arange(start, end), n_groups)]


def cpcv_mask(market, blocks, test_pair, embargo=None) -> np.ndarray:
    """The `fit_date_mask` for one CPCV path: train on the other blocks only.

    True exactly on the non-test blocks, minus config.WF_EMBARGO_DAYS on BOTH
    sides of every test block. Two exclusions, two different leaks:

      • the blocks themselves, because training on a test block is the leak CPCV
        exists to measure;
      • the embargo band, because a label that RESOLVES inside a test block
        carries that block's returns even though its feature row sits outside it.
        strategy.py's mask rule requires both the row's date and its t1 to be
        mask-True, which closes the same hole from the label side.

    Everything outside the scoring span (the pre-1999 warm-up, and every vault
    day) is False. So a model family trains ONLY on the six training blocks: an
    early test block has little or no training data before it and its model stays
    flat, which biases those paths toward zero. That bias is DOWNWARD — the gate
    can only be made harder to pass by it — and the alternative (letting a path
    train on pre-span history that other paths' test blocks sit inside) would
    quietly restore the leak.
    """
    embargo = int(config.WF_EMBARGO_DAYS if embargo is None else embargo)
    mask = np.zeros(len(market.dates), dtype=bool)
    for k, (a, b) in enumerate(blocks):
        if k not in test_pair:
            mask[a:b + 1] = True
    for k in test_pair:
        a, b = blocks[k]
        mask[max(0, a - embargo):min(len(mask), b + 1 + embargo)] = False
    return mask


def _cpcv_one(genome, market, cost, blocks, pair) -> dict:
    """One combinatorial purged path: two test blocks, each simulated by an agent
    that may only have fitted on the other six. Module level so joblib can pickle it."""
    mask = cpcv_mask(market, blocks, pair)
    nets, dates, fits = [], [], 0
    for k in sorted(pair):                       # chronological: blocks are ordered
        a, b = blocks[k]
        audit: list = []
        res = StrategyAgent(genome, market, cost).run_episode(
            env_start=market.dates[a], env_end=market.dates[b],
            fit_audit=audit, fit_date_mask=mask)
        nets.append(res["daily_net"])
        dates.append(res["dates"])
        fits += len(audit)
    net = np.concatenate(nets)
    return {"pair": tuple(int(k) for k in pair),
            "sharpe": sharpe(net),
            "cum_return": float(np.prod(1.0 + net[np.isfinite(net)]) - 1.0),
            "n_days": int(net.size),
            "n_fits": int(fits),
            "start": dates[0][0], "end": dates[-1][-1]}


def cpcv_paths(genome, market, cost=None, n_groups=None, k=None, n_jobs=1,
               verbose=False) -> dict:
    """Gate G5. Combinatorial purged cross-validation with REAL refits.

    C(n_groups, k) = C(8, 2) = 28 paths. Each path takes two of the eight
    pre-vault blocks as its test set and lets the genome fit on the other six
    (see cpcv_mask); the path's daily series is the two test blocks concatenated
    in date order, and the path Sharpe is evaluate.sharpe of that series.

    WHAT THIS ANSWERS THAT A WALK-FORWARD CANNOT: F1 produces ONE path through
    history, so its Sharpe has no dispersion to report and a single lucky regime
    can carry it. 28 paths built from different train/test partitions give a
    distribution — DESIGN's G5 asks that 70% of them are net-positive and that the
    MEDIAN path clears 0.30, which a strategy that works in one era and not the
    others cannot do.

    EXPENSIVE BY CONSTRUCTION: 2 x 28 = 56 episodes with refits, per candidate.
    joblib over paths; `verbose` prints each path as it lands, because a silent
    hour is indistinguishable from a hung one.

    Rule families never fit, so the mask is inert for them and their paths measure
    sub-period consistency rather than a train/test split — stated in the report
    rather than papered over.
    """
    cost = cost if cost is not None else CostModel()
    k = int(config.CPCV_K if k is None else k)
    blocks = cpcv_blocks(market, n_groups)
    pairs = list(itertools.combinations(range(len(blocks)), k))

    rows = []
    par = Parallel(n_jobs=max(1, int(n_jobs)), batch_size=1, return_as="generator")
    results = par(delayed(_cpcv_one)(genome, market, cost, blocks, p) for p in pairs)
    try:
        for i, row in enumerate(results):
            rows.append(row)
            if verbose:
                print("      path %2d/%d  blocks %s  %s..%s  %5d days  SR %+6.2f  "
                      "return %+7.1f%%  %d fits"
                      % (i + 1, len(pairs), "+".join(str(b) for b in row["pair"]),
                         row["start"].date(), row["end"].date(), row["n_days"],
                         row["sharpe"], 100 * row["cum_return"], row["n_fits"]),
                      flush=True)
    finally:
        del results
    rows.sort(key=lambda r: r["pair"])

    srs = np.array([r["sharpe"] for r in rows], dtype=np.float64)
    return {"path_sharpes": [float(s) for s in srs],
            "pairs": [r["pair"] for r in rows],
            # "net-positive" is read on the Sharpe (equivalently: a positive mean
            # daily net return). The cumulative-return count is reported beside it
            # because compounding can separate the two on a volatile path.
            "frac_positive": float(np.mean(srs > 0.0)) if srs.size else 0.0,
            "frac_return_positive": float(np.mean([r["cum_return"] > 0 for r in rows]))
            if rows else 0.0,
            "median_sharpe": float(np.median(srs)) if srs.size else 0.0,
            "n_paths": len(rows),
            "n_blocks": len(blocks),
            "blocks": [(str(market.dates[a].date()), str(market.dates[b].date()))
                       for a, b in blocks],
            "total_fits": int(sum(r["n_fits"] for r in rows)),
            "days_per_path": int(np.median([r["n_days"] for r in rows])) if rows else 0}


def regime_slices(daily_net, dates, windows=None) -> list:
    """Gate G8. Cumulative NET return inside each crisis window.

    One entry per config.GATE_REGIME_WINDOWS slice, in order, or NaN where the
    series has no day inside it at all. PASS-BY-ABSENCE IS DELIBERATE AND IT IS
    THE CALLER'S RULE TO APPLY (gates.py): a genome whose scored history stops in
    2019 has said nothing about 2020, and inventing a failure there would be as
    dishonest as inventing a pass. What makes that safe is that run_deepeval hands
    this function the pre-vault AND vault series together — under a counted vault
    access — so in production every slice IS covered, and absence only shows up on
    a short synthetic series (verify test 7) or a genome that never traded then.

    A partially covered slice is scored on the days it has, and run_deepeval
    prints the day count beside each number so a thin slice is visible as one.
    """
    windows = windows if windows is not None else config.GATE_REGIME_WINDOWS
    r = np.asarray(daily_net, dtype=np.float64).ravel()
    idx = pd.DatetimeIndex(dates)
    out = []
    for start, end in windows:
        sel = (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))
        vals = r[sel.to_numpy() if hasattr(sel, "to_numpy") else sel]
        vals = vals[np.isfinite(vals)]
        out.append(float(np.prod(1.0 + vals) - 1.0) if vals.size else float("nan"))
    return out


def regime_slice_days(dates, windows=None) -> list:
    """How many scored days fall in each slice — the coverage behind regime_slices."""
    windows = windows if windows is not None else config.GATE_REGIME_WINDOWS
    idx = pd.DatetimeIndex(dates)
    return [int(((idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))).sum())
            for a, b in windows]


def _garch_t_params(r: np.ndarray) -> dict:
    """Fit GARCH(1,1) with Student-t errors to a daily return series.

    `arch` is imported HERE, not at module top: it pulls in statsmodels, and every
    F0/F1 worker imports this module while none of them needs a volatility model.

    Returns the parameters plus the last conditional variance and residual (the
    state a forecast continues from), all in PERCENT units — arch is numerically
    much better behaved on returns scaled to percent, and the caller divides back.
    On a fit that fails or comes back degenerate the caller falls back to an iid
    Student-t with the same moments, and says so in the result.
    """
    from arch import arch_model                # noqa: PLC0415 — see docstring
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = arch_model(r * 100.0, mean="Constant", vol="Garch", p=1, q=1,
                         dist="t").fit(disp="off", show_warning=False)
    p = fit.params
    out = {"mu": float(p["mu"]), "omega": float(p["omega"]),
           "alpha": float(p["alpha[1]"]), "beta": float(p["beta[1]"]),
           "nu": float(p["nu"]),
           "var_last": float(fit.conditional_volatility[-1] ** 2),
           "resid_last": float(np.asarray(fit.resid)[-1])}
    ok = (np.isfinite(list(out.values())).all() and out["omega"] > 0
          and out["alpha"] >= 0 and out["beta"] >= 0
          and out["alpha"] + out["beta"] < 1.0 and out["nu"] > 2.0)
    out["ok"] = bool(ok)
    return out


def _max_drawdowns(paths: np.ndarray) -> np.ndarray:
    """Worst peak-to-trough drawdown of each simulated path (a negative number)."""
    eq = np.cumprod(1.0 + paths, axis=1)
    return (eq / np.maximum.accumulate(eq, axis=1) - 1.0).min(axis=1)


def ruin_mc(daily_net, rng=None, paths=None, years=None, dd=None, block=None) -> dict:
    """Gate G10. P(drawdown deeper than `dd` within `years`), two ways.

    TWO ENGINES BECAUSE ONE MODEL IS AN OPINION:
      garch_t    a GARCH(1,1)-t fitted to this strategy's own daily net returns,
                 simulated forward. Volatility clusters and the tails are fat, so
                 a quiet backtest cannot promise a quiet future.
      bootstrap  a circular block bootstrap (block = config.BOOT_BLOCK) of the
                 realised returns themselves — no distribution assumed at all,
                 and the drawdown-generating autocorrelation preserved.

    The reported `p_ruin` is the WORSE of the two, not the average: the engines
    disagree exactly where the tail is, and a gate that averages away the
    pessimistic model is a gate that fails in the expensive direction. Both
    numbers are returned and the report prints both.

    ATTRIBUTION: sell_in_may/montecarlo.py has seeded GARCH-t and block-bootstrap
    engines, and they are NOT reused here — they return terminal PRICES
    (`S0 * exp(sum of h daily draws)`) because an option payoff only needs the
    endpoint, and a drawdown needs the whole path. The two engines below are the
    same idea (standardised t shocks scaled by sqrt((nu-2)/nu); circular blocks)
    re-expressed as paths, ~20 lines, rather than a widening of that module's
    contract for one caller in another project.

    Deterministic given `rng` (run_deepeval derives child_rng(SEED, generation,
    "ruin", i) per candidate).
    """
    rng = rng if rng is not None else np.random.default_rng(config.SEED)
    n = int(config.RUIN_MC_PATHS if paths is None else paths)
    horizon = int(config.TRADING_DAYS_YEAR * (config.RUIN_MC_YEARS if years is None else years))
    dd = float(config.GATE_RUIN_DD if dd is None else dd)
    block = int(config.BOOT_BLOCK if block is None else block)

    r = np.asarray(daily_net, dtype=np.float64).ravel()
    r = r[np.isfinite(r)]
    if r.size < config.SHARPE_MIN_OBS or r.std(ddof=1) <= 0.0:
        return {"p_ruin": float("nan"), "p_ruin_garch_t": float("nan"),
                "p_ruin_bootstrap": float("nan"), "n_paths": 0, "horizon_days": horizon,
                "dd_threshold": dd, "garch_ok": False,
                "note": "series too short or degenerate to simulate"}

    try:
        par = _garch_t_params(r)
    except Exception as exc:                    # a failed fit is a fallback, not a crash
        par = {"ok": False, "error": str(exc)[:120]}
    if not par.get("ok"):
        # iid Student-t with the series' own moments: fatter-tailed than a normal,
        # no volatility clustering. Flagged, because it is the weaker model.
        nu = 6.0
        z = rng.standard_t(nu, size=(n, horizon)) * np.sqrt((nu - 2.0) / nu)
        garch = r.mean() + r.std(ddof=1) * z
    else:
        nu = par["nu"]
        scale = np.sqrt((nu - 2.0) / nu)        # unit-variance standardised t
        var = np.full(n, par["omega"] + par["alpha"] * par["resid_last"] ** 2
                      + par["beta"] * par["var_last"])
        garch = np.empty((n, horizon))
        for t in range(horizon):
            eps = np.sqrt(var) * rng.standard_t(nu, size=n) * scale
            garch[:, t] = (par["mu"] + eps) / 100.0        # percent -> return
            var = par["omega"] + par["alpha"] * eps ** 2 + par["beta"] * var

    T = r.size
    n_blocks = int(np.ceil(horizon / max(1, min(block, T))))
    starts = rng.integers(T, size=(n, n_blocks, 1))
    idx = (starts + np.arange(min(block, T))).reshape(n, -1)[:, :horizon] % T
    boot = r[idx]

    p_g = float(np.mean(_max_drawdowns(garch) < -dd))
    p_b = float(np.mean(_max_drawdowns(boot) < -dd))
    return {"p_ruin": max(p_g, p_b), "p_ruin_garch_t": p_g, "p_ruin_bootstrap": p_b,
            "n_paths": n, "horizon_days": horizon, "dd_threshold": dd,
            "garch_ok": bool(par.get("ok")),
            "note": "worse of two engines" if par.get("ok") else
                    "GARCH fit unusable (%s); iid Student-t fallback"
                    % par.get("error", "degenerate parameters")}


def _shared_calendar(cand_net, cand_dates, inc_net, inc_dates) -> tuple:
    """(candidate, incumbent, shared dates) restricted to the calendar BOTH were
    scored on. The one place the two halves of gate G9 agree on what "same period"
    means, so they cannot drift apart."""
    a = pd.Series(np.asarray(cand_net, dtype=np.float64),
                  index=pd.DatetimeIndex(cand_dates))
    b = pd.Series(np.asarray(inc_net, dtype=np.float64), index=pd.DatetimeIndex(inc_dates))
    common = a.index.intersection(b.index)
    return a.reindex(common).to_numpy(), b.reindex(common).to_numpy(), common


def shared_span_sharpes(cand_net, cand_dates, inc_net, inc_dates) -> dict:
    """Gate G9's FIRST half: both parties' Sharpe over the days they share.

    G1 pins the two runs to the same market and the same last bar, but NOT to the
    same first bar — evaluate.full_eval starts each genome at its own first active
    bar, and after the window-END fix a challenger whose history begins in 2015 is
    like-for-like with a champion scored from 1999. Comparing each party's OWN
    full-window Sharpe would then hand the shorter party every era the longer one
    had to survive and it did not: a 2015-start candidate is measured across a
    single bull run, the incumbent across the dot-com bust, 2008 and 2020, and the
    margin gate reads the difference as skill. THE DIFFERENCE IT WOULD BE READING
    IS THE CALENDAR.

    So both Sharpes are computed on the intersection, exactly as
    rolling_window_wins already does for the other half of G9. Fewer than
    config.SHARPE_MIN_OBS shared days is not a thin comparison, it is no
    comparison: both come back NaN, which gates.py counts as unmeasured, and an
    unmeasured gate is a failed gate.
    """
    a, b, common = _shared_calendar(cand_net, cand_dates, inc_net, inc_dates)
    out = {"n_common_days": int(len(common)),
           "start": str(common[0].date()) if len(common) else None,
           "end": str(common[-1].date()) if len(common) else None}
    if len(common) < config.SHARPE_MIN_OBS:
        return dict(out, cand_sharpe=float("nan"), inc_sharpe=float("nan"),
                    note="only %d shared day(s), fewer than the %d a Sharpe needs: "
                         "the margin is NOT MEASURABLE on a shared calendar"
                         % (len(common), config.SHARPE_MIN_OBS))
    return dict(out, cand_sharpe=sharpe(a), inc_sharpe=sharpe(b),
                note="both Sharpes measured on the %d days both parties were "
                     "scored (%s .. %s)"
                     % (len(common), out["start"], out["end"]))


def rolling_window_wins(cand_net, cand_dates, inc_net, inc_dates, years=None) -> dict:
    """Gate G9's second half: the share of rolling windows the candidate wins.

    Both series are restricted to the dates they SHARE — a candidate cannot claim
    a win over years the incumbent was not scored on — then cut into overlapping
    windows of `years` trading years, stepped a quarter at a time (a derived step,
    not a new knob: TRADING_DAYS_YEAR // 4). A window is won only if the
    candidate's Sharpe is STRICTLY greater; a tie goes to the incumbent, which is
    the same doctrine gates.py applies to every comparison.
    """
    years = int(config.GATE_ROLLING_WINDOW_YEARS if years is None else years)
    a, b, common = _shared_calendar(cand_net, cand_dates, inc_net, inc_dates)

    width = years * config.TRADING_DAYS_YEAR
    step = max(1, config.TRADING_DAYS_YEAR // 4)
    if len(common) < width:
        return {"win_frac": 0.0, "n_windows": 0, "n_common_days": int(len(common)),
                "note": "fewer than %d shared days: no %d-year window exists" % (width, years)}
    wins = []
    for lo in range(0, len(common) - width + 1, step):
        wins.append(sharpe(a[lo:lo + width]) > sharpe(b[lo:lo + width]))
    return {"win_frac": float(np.mean(wins)), "n_windows": len(wins),
            "n_common_days": int(len(common)),
            "note": "%d-year windows stepped %d days, strict wins only" % (years, step)}


# ── multiple-testing statistics ────────────────────────────────────────────────
# Copied verbatim from ~/futures_bot/evaluate.py:200-278 (Bailey & Lopez de Prado
# DSR; Bailey/Borwein/LdP/Zhu CSCV PBO). Kept byte-for-byte so the two projects
# cannot silently drift apart; arena feeds them NET returns only.
def _sharpe_cols(X):
    """Per-column (per-config) Sharpe over rows; NaN where std is 0."""
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0, ddof=1)
    sd = np.where(sd == 0, np.nan, sd)
    return mu / sd


def deflated_sharpe(returns, all_sharpes):
    """Bailey & López de Prado Deflated Sharpe Ratio. all_sharpes = the per-config
    daily Sharpes across the whole trial set (multiple-testing correction)."""
    r = np.asarray(returns, float)
    r = r[~np.isnan(r)]
    T = r.size
    if T < 3 or r.std(ddof=1) == 0:
        return {"dsr": None, "reason": "insufficient/degenerate returns"}
    sr = float(r.mean() / r.std(ddof=1))
    g3 = float(skew(r))
    g4 = float(kurtosis(r, fisher=False))            # non-excess (normal == 3)
    s = np.asarray(all_sharpes, float)
    s = s[~np.isnan(s)]
    N = max(int(s.size), 1)
    var_sr = float(np.var(s, ddof=1)) if N > 1 else 0.0
    emc = 0.5772156649015329                          # Euler-Mascheroni
    if N > 1 and var_sr > 0:
        z1 = norm.ppf(1.0 - 1.0 / N)
        z2 = norm.ppf(1.0 - 1.0 / (N * np.e))
        sr0 = np.sqrt(var_sr) * ((1 - emc) * z1 + emc * z2)
    else:
        sr0 = 0.0
    denom = np.sqrt(max(1e-12, 1 - g3 * sr + ((g4 - 1) / 4.0) * sr ** 2))
    dsr = float(norm.cdf((sr - sr0) * np.sqrt(T - 1) / denom))
    return {"sharpe_daily": round(sr, 4),
            "sharpe_annual": round(sr * np.sqrt(252), 3),
            "sr0_threshold": round(float(sr0), 4),
            "dsr": round(dsr, 4), "n_trials": N, "T": T,
            "skew": round(g3, 3), "kurtosis": round(g4, 3)}


def _cscv_splits(R, S):
    R = np.asarray(R, float)
    T = R.shape[0]
    groups = [g for g in np.array_split(np.arange(T), S) if len(g) > 0]
    S = len(groups)
    if S % 2 == 1:
        groups = groups[:-1]
        S -= 1
    half = S // 2
    for combo in itertools.combinations(range(S), half):
        train = np.concatenate([groups[i] for i in combo])
        test = np.concatenate([groups[i] for i in range(S) if i not in combo])
        yield R[train], R[test]


def pbo_cscv(R, S=10):
    """Probability of Backtest Overfitting (Bailey/Borwein/LdP/Zhu, 2017).
    Fraction of train/test splits where the IS-best config lands below the OOS median."""
    R = np.asarray(R, float)
    N = R.shape[1]
    lambdas, oos_best = [], []
    for tr, te in _cscv_splits(R, S):
        is_s = _sharpe_cols(tr)
        oos_s = _sharpe_cols(te)
        nstar = int(np.nanargmax(is_s))
        order = np.argsort(np.nan_to_num(oos_s, nan=-1e18))
        rank = int(np.where(order == nstar)[0][0]) + 1     # 1=worst .. N=best
        omega = min(max(rank / (N + 1), 1e-6), 1 - 1e-6)
        lambdas.append(np.log(omega / (1 - omega)))
        oos_best.append(oos_s[nstar])
    lambdas = np.array(lambdas)
    oos_best = np.array(oos_best, float)
    oos_best = oos_best[~np.isnan(oos_best)]
    return {"pbo": round(float((lambdas <= 0).mean()), 3),
            "n_splits": int(lambdas.size),
            "median_oos_sharpe_of_is_best": round(float(np.median(oos_best)), 4)
            if oos_best.size else None,
            "frac_oos_positive": round(float((oos_best > 0).mean()), 3)
            if oos_best.size else None}


if __name__ == "__main__":
    # Smoke test: the two statistics on toy arrays, no market, no cache, no network.
    rng = np.random.default_rng(config.SEED)

    print("arena evaluation ladder")
    print("  eras       : %s  (all pre-vault: %s)"
          % (", ".join("%s..%s" % (a[:4], b[:4]) for a, b in config.SCREEN_ERAS),
             config.VAULT_START))
    print("  sharpe()   : %d obs -> %.2f | %d obs -> %.2f (below SHARPE_MIN_OBS=%d)"
          % (400, sharpe(rng.normal(0.0008, 0.01, 400)),
             30, sharpe(rng.normal(0.0008, 0.01, 30)), config.SHARPE_MIN_OBS))

    # DSR: the same return series gets less impressive the more candidates were tried.
    ret = rng.normal(0.0006, 0.01, 2000)
    print("  deflated_sharpe on one fixed 2000-day series, trial-Sharpe spread held")
    print("    %-10s %-10s %-10s" % ("n_trials", "sr0", "dsr"))
    base = rng.normal(0.0, 1.0, 4096)
    base = (base - base.mean()) / base.std(ddof=1)          # unit dispersion, fixed
    for n in (1, 8, 64, 512, 4096):
        d = deflated_sharpe(ret, 0.05 * base[:n] + 0.02)
        print("    %-10d %-10s %-10s" % (n, d["sr0_threshold"], d["dsr"]))

    # PBO: pure noise should look overfit; a genuinely persistent column should not.
    R = rng.normal(0.0, 0.01, size=(500, 20))
    print("  pbo_cscv(500 x 20 pure noise, S=16)      : %s"
          % pbo_cscv(R, S=config.PBO_SPLITS)["pbo"])
    R2 = R.copy()
    R2[:, 7] += 0.0025                                       # one real, persistent edge
    out = pbo_cscv(R2, S=config.PBO_SPLITS)
    print("  pbo_cscv(same + one persistent column)   : %s  (median OOS SR of IS-best %s)"
          % (out["pbo"], out["median_oos_sharpe_of_is_best"]))

    # The F2 battery on toy series (cpcv_paths needs a market — the milestone and
    # run_deepeval exercise it; everything else here is a pure function).
    print("\n  F2 battery (toy series; gates.py reads exactly these numbers)")
    good = rng.normal(0.0008, 0.009, 3000)                  # ~SR 1.4 annualised
    flat = rng.normal(0.0000, 0.009, 3000)
    for label, series in (("SR %+.2f" % sharpe(good), good), ("SR %+.2f" % sharpe(flat), flat)):
        lo, hi = bootstrap_sharpe_ci(series, n=1000, rng=np.random.default_rng(config.SEED))
        ruin = ruin_mc(series, rng=np.random.default_rng(config.SEED))
        print("    %-9s  boot 95%% CI [%+.2f, %+.2f]  G6 %-4s | P(DD>%.0f%%) garch %.3f "
              "bootstrap %.3f -> %.3f  G10 %s"
              % (label, lo, hi, "pass" if lo > 0 else "FAIL", 100 * config.GATE_RUIN_DD,
                 ruin["p_ruin_garch_t"], ruin["p_ruin_bootstrap"], ruin["p_ruin"],
                 "pass" if ruin["p_ruin"] < config.GATE_RUIN_MAX_PROB else "FAIL"))
    # Determinism: same rng in, same interval out (verify test 7 pins this too).
    same = (bootstrap_sharpe_ci(good, n=500, rng=np.random.default_rng(7))
            == bootstrap_sharpe_ci(good, n=500, rng=np.random.default_rng(7)))
    print("    bootstrap CI reproducible from a fixed rng: %s" % same)

    dates = pd.bdate_range("1999-01-04", periods=3000)       # covers 2000-02 and 2008-09
    slices = regime_slices(good, dates)
    days = regime_slice_days(dates)
    print("    regime slices : %s"
          % "  ".join("%s %s (%d d)" % (w[0][:7],
                                        "n/a" if np.isnan(v) else "%+.1f%%" % (100 * v), d)
                      for w, v, d in zip(config.GATE_REGIME_WINDOWS, slices, days)))
    print("      NaN slices are PASS-BY-ABSENCE: the series said nothing about that")
    print("      window, and inventing a failure there is as dishonest as a pass —")
    print("      bounded by gates.py at GATE_REGIME_MIN_COVERED of %d windows, so"
          % config.GATE_REGIME_MIN_COVERED)
    print("      absence may not be what carries G8.")
    wins = rolling_window_wins(good, dates, flat, dates)
    print("    G9 rolling    : candidate wins %.0f%% of %d %d-year windows (%s)"
          % (100 * wins["win_frac"], wins["n_windows"], config.GATE_ROLLING_WINDOW_YEARS,
             wins["note"]))
    # G9's other half on ASYMMETRIC spans — the case the gate exists for: a
    # challenger that only ran for the tail of the incumbent's history.
    tail = dates[-600:]
    span = shared_span_sharpes(good[-600:], tail, flat, dates)
    print("    G9 margin     : candidate %+.2f vs incumbent %+.2f on %d shared days "
          "(%s .. %s); the incumbent's OWN full window is %+.2f"
          % (span["cand_sharpe"], span["inc_sharpe"], span["n_common_days"],
             span["start"], span["end"], sharpe(flat)))
    print("\n  DSR and PBO correct for SEARCH, not for a wrong sandbox. They are")
    print("  evidence about the past, not a guarantee — and not financial advice.")
