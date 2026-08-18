# arena — an evolving strategy arena

A population of complete trading strategies competes in a sandbox that replays
~30 years of daily US equity bars. Each generation, winners breed mutated
offspring and losers are culled. A strategy is promoted to *champion* only by
passing ten anti-luck statistical gates — and so far, **none has**.

That last sentence is the point of this repository. It is a machine for
producing honest refusals as readily as promotions, and everything below exists
to make a false positive expensive.

**This is not investment advice, not a signal service, and not a product.** It
is a research system whose entire output is a probabilistic claim about the past.

---

## What it actually does

```
run_generation.py     nightly: refresh data -> screen 64 genomes -> full-history
                      evaluate the survivors -> breed -> append -> alert -> exit
run_deepeval.py       weekly: re-simulate every party fresh -> the F2 battery ->
                      ten gates -> promote or refuse -> report -> alert -> exit
```

A **genome** is a whole strategy: a signal (momentum / mean-reversion /
seasonal rule, or ridge / logistic / gradient-boosting model), a portfolio
construction (how many longs and shorts, weighting, gross exposure, vol target,
rebalance cadence) and a risk overlay (stops, trailing stops, regime filter,
drawdown limit). Its sha256 is its identity, and every evaluation of it — at any
fidelity, ever — is appended to a trial ledger that nothing may rewrite — with one
bounded exception: `ledger.dedup_ledger` removes byte-identical merge artifacts
(a row a union merge appended twice, a header line it carried into the middle of
the file), never a row that differs by so much as one character, and it prints
what it removed and why.

The **sandbox** charges honest small-account costs. You decide at the close of
day `t`; you fill at the open of `t+1` at `open ± half-spread ± slippage`, plus
commission with a $1 minimum, daily borrow on short market value and margin
interest on negative cash. Whole shares. No same-day round trips (PDT-safe under
$25K). The accounting identity `equity == cash + Σ shares·close` is asserted
every single step, and the equity path is replayable from the decision log.

## How the honesty machinery works

The failure mode this project is built against is not "the code has a bug". It
is "the backtest is beautiful and the money is gone". Five defenses, each aimed
at a specific way that happens:

**The vault.** All fitness and selection uses out-of-sample days *before
2020-01-01 only*. Everything from 2020 onward is touched exclusively by the
weekly promotion gates, and every single access is counted in
`state/vault_access.csv`. Selection therefore cannot overfit the recent six
years, because it has never seen them. Asking twice is two looks, and the second
look is deflated by the first.

**The trial ledger is the DSR input.** A Deflated Sharpe Ratio corrects a
Sharpe for how many strategies you tried before finding it — so it is only as
honest as that count. Here `n_trials` is the number of distinct genomes ever
evaluated, screens included (a screen exerts selection pressure, so excluding it
would flatter every number downstream), and the deflation uses the *empirical*
spread of trial Sharpes from that ledger. It is never a hardcoded constant.
**Every report prints DSR with its N attached.**

**Purged, embargoed cross-validation.** Overlapping forward-return labels let a
training row peek at its own test period. Training rows whose label window
overlaps a test block are purged, and a buffer around the boundary is embargoed.
The full-history evaluation is *streaming* — a refit at date `d` can only use
rows whose labels resolved before `d − 21 days` — and every refit writes an audit
row proving it.

**CPCV and PBO.** Combinatorial purged cross-validation runs 28 train/test path
combinations with real refits per candidate. CSCV estimates the Probability of
Backtest Overfitting: given that you picked the in-sample winner out of a cohort,
how often does it fall below the pack out of sample? Above 0.20 and the
candidate is refused.

**The gate stack.** Ten gates, all of which must pass; ties go to the incumbent:

| | gate | threshold |
|---|---|---|
| G1 | like-for-like — same data, panel and settings; incumbent re-simulated fresh | identical hashes |
| G2 | DSR, pre-vault | ≥ 0.95 at the full ledger N |
| G3 | vault confirmation | vault Sharpe > 0 and vault DSR ≥ 0.90 |
| G4 | PBO (CSCV, S=16) | ≤ 0.20 |
| G5 | CPCV, 28 paths | ≥ 70% net-positive, median path Sharpe ≥ 0.30 |
| G6 | bootstrap 95% CI of net Sharpe | lower bound > 0 |
| G7 | 2× cost stress (borrow 3×) | Sharpe > 0 and ≥ 0.5× base |
| G8 | four crisis windows | ≥3 of 4 covered by the scored span, none worse than −30%, ≥3 of 4 above −5% |
| G9 | beats the incumbent | +0.15 Sharpe **on the calendar both were scored on** and ≥60% of rolling 3-year windows |
| G10 | ruin Monte Carlo | P(drawdown > 40% in 2 years) < 5% |

G8 and G9 are worded that way because a challenger may legitimately have less
history than the champion (G1 compares the window's END, not its start, or the
first promotion would lock every other strategy family out forever). A genome
first active in 2015 has no dot-com bust, no 2008 and no 2020 in its record —
which flatters its Sharpe and empties its crisis slices at the same time. So the
margin is measured only over days both parties actually traded, and a span that
misses most of the crisis windows fails G8 as unmeasurable rather than passing
by absence.

The gates have already refused. `state/deepeval_history.csv` records the
refusals with the failing gate IDs, and the latest report shows every value
beside its threshold.

## Running it locally

Python 3.9, no virtualenv, no build system. Every module has a `__main__` smoke
test that prints a sanity check.

```bash
pip install -r requirements.txt

python3 verify.py                    # the test suite — the gate for scheduling anything
python3 config.py                    # which sibling modules are in play, and the config hash
python3 features.py                  # panel identity + live-vs-vendored parity check
python3 alerts_arena.py              # both alert formats, dry
python3 reports.py --gen 1           # rebuild a report from what is on disk

python3 run_generation.py --init     # seed a population and run generation 0
python3 run_generation.py            # run the next generation (dry alert)
python3 run_deepeval.py --dry        # the battery and the gates, writing no decision
python3 run_deepeval.py --rollback <hash>    # repoint the champion at a prior artifact
```

`verify.py` runs entirely on a synthetic seeded market — no network, no cache —
so it gives the same answer on a plane, on a GitHub runner, and after a bad
yfinance day. It must be green before anything is scheduled.

### The vendored siblings

arena grew out of two sibling projects on the author's machine (`sell_in_may`,
`signal_lab`) and reuses their point-in-time feature panel and price cache. A
public runner checks out one repository and has neither, so the seven modules
the import graph actually reaches are copied under `vendor/`, byte-identical
below a provenance header. `config.py` imports the live checkouts when they
exist and the vendored copies otherwise; `ARENA_FORCE_VENDOR=1` forces the
vendored path on a machine that has both.

The standing proof that the runner runs this code is a **byte comparison**:
`python3 config.py` compares every file under `vendor/` against its source and
says so. `python3 features.py` adds an end-to-end check — both modes must build
the same feature panel, compared by `panel_hash` — but that one is **conditional
on the two caches holding the same data vintage**, and they only do until the
first cloud refresh. yfinance restates its whole adjusted history on every fetch,
so the sibling cache and arena's committed `data/cache` diverge for real reasons;
once they have, the check reports `NOT COMPARABLE` with both data hashes rather
than a failure that would say nothing about the code.

```bash
python3 config.py              # byte-compares every vendored file (always valid)
python3 features.py            # panel_hash parity, while the caches agree
ARENA_FORCE_VENDOR=1 python3 verify.py
```

To re-vendor after a sibling changes, copy the file back and re-stamp the
three-line header with `git -C <sibling> rev-parse --short HEAD`. Do not edit
the copies in place.

## The schedule

Everything runs on GitHub Actions, on public-repo runners, with the laptop off.
The repository is the source of truth: each run checks out state, computes,
appends, and commits the result back.

| workflow | when | what |
|---|---|---|
| `generation.yml` | 02:30 and 14:30 UTC, daily | one generation of search |
| `deepeval.yml` | Saturday 12:00 UTC | the F2 battery, the gates, the report |
| `paper.yml` | dispatch only — **not armed** | Phase 7, at paper graduation |

Twice daily is the honest reading of "as often as possible": the data changes
once per trading day, so beyond about two generations a day the extra runs
mostly inflate the trial count — which *deflates* every deflated Sharpe — without
exploring meaningfully more. Both scheduled workflows share one concurrency
group so they can never overlap on state.

One consequence of that group is worth knowing: **GitHub keeps only one pending
run per concurrency group.** If a run is executing and two more are triggered,
the first of those waits and the second silently *replaces* it — the replaced run
is cancelled before it starts and never appears as a failure. A generation can
therefore be skipped without anything looking wrong. It is harmless by design
(nothing is lost, the next run resumes or moves on) but it means the run count
is not a reliable measure of how many generations were attempted; the trial
ledger is.

Alerts go to Telegram/ntfy through repo secrets. Nothing sensitive is in the
repository; `config.local.json` is gitignored and no credential is ever printed,
only its presence.

## Limitations — the parts that would embarrass this project if left unsaid

- **Survivorship bias.** The universe is a snapshot of *today's* S&P 500
  membership, so every company that failed out of the index is missing from all
  of these backtests. Long results come out optimistic and short results
  pessimistic. Disclosed, not solved; delisted-inclusive (CRSP-class) data is the
  named upgrade path.
- **Non-stationarity.** 1990s microstructure is simulated with modern costs, and
  an edge that existed decades ago may simply have been arbitraged away. The
  vault, the regime gate and the paper stage are defenses, not proofs.
- **Multiple testing survives the gates.** Evolutionary trials are *correlated*,
  and correlated trials deflate less than independent ones would — so the ledger
  DSR is an optimistic correction, and PBO does not cover designer-level choices
  at all. Treat the gates as risk reduction. The only accumulating true
  out-of-sample is vault → paper → live.
- **Small-account sensitivity.** A $1 minimum commission and whole-share
  rounding are 5–10 bps each way on small positions. The 2× cost-stress gate and
  the per-report "cost share of gross" line are what keep that visible.
- **The cost model is proportional (bps), not per-share.** The price history is
  split-adjusted, so a $/share friction would silently become hundreds of bps on
  1990s adjusted prices of high-split names. Real 1990s spreads were wider than
  modern bps, so any edge that survives *only* at modern costs is suspect by
  construction.
- **Determinism is per-platform.** Seeded everywhere, and identical run to run on
  one machine — but BLAS floating-point ordering differs between Apple Silicon
  and x86 Linux, so a Mac result and a runner result can disagree in the low
  bits. Ledger rows are platform-tagged and the cloud is canonical.
- **yfinance fragility.** Cache-first, and the runner keeps its own committed
  copy of the cache, so a bad data day degrades to stale-cache rather than
  failure. Past five days stale, the run aborts with an alert instead of scoring
  genomes on a market that no longer exists.
- **This repository is public on purpose,** including the champion genome and
  every ledger. At $10–25K nobody can profitably front-run it, and the honesty
  machinery is worth more with the working shown.

**Positive sandbox alpha — even DSR-corrected, PBO-checked and vault-confirmed —
is a probabilistic claim about the past. It is evidence, not a guarantee, and it
is not financial advice.**

---

Design document, in full: [`docs/DESIGN.md`](docs/DESIGN.md) — the sandbox clock,
the genome space, the evaluation ladder, the gate thresholds, the graduation
ladder to paper and live trading, and the eleven checks `verify.py` implements.
