"""
The honest-evaluation ladder: F0 screen, F1 full walk-forward, and the two
multiple-testing statistics the gates will read.

    screen(genome, market, cost)     -> {"score", "era_sharpes", "n_features", "n_days"}
    full_eval(genome, market, cost)  -> {"score", "sharpe_prevault", "daily_net", ...}
    deflated_sharpe(returns, all_sharpes)
    pbo_cscv(R, S)

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
  gates.py (Phase 5, which counts every access through ledger.record_vault_access)
  every hit must be a store or a pass-through — never an input to a comparison, a
  sort, a Sharpe, or a score. If that stops being true, the vault is gone and the
  last six years of data stop being evidence.

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

import numpy as np
import pandas as pd
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
    from dataclasses import replace
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

    Score = mean(era Sharpes) − PARSIMONY_PENALTY × n_features. The tax is on the
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
    return {"score": float(np.mean(era_sharpes)) - config.PARSIMONY_PENALTY * n_features,
            "era_sharpes": [float(s) for s in era_sharpes],
            "n_features": n_features,
            "n_days": int(n_days)}


# ── F1: the full anchored walk-forward ─────────────────────────────────────────
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

    `score` = pre-vault net Sharpe − PARSIMONY_PENALTY × n_features, the same
    currency as the F0 score so the two ladders are at least comparable in units.

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
    return {"score": sr - config.PARSIMONY_PENALTY * n_features,
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
    print("\n  DSR and PBO correct for SEARCH, not for a wrong sandbox. They are")
    print("  evidence about the past, not a guarantee — and not financial advice.")
