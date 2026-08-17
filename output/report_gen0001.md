# arena — generation 1

**No champion.** Nothing has passed all ten promotion gates, so this system currently recommends nothing and holds nothing.

This report is about `9353d613c7d3`, the leading candidate of the last deep evaluation, **which the gates REFUSED** (failed G2+G4+G8). It is shown because a refusal with its numbers attached is more useful than a blank page — not because it is close to being a champion.

| | |
|---|---|
| data as of | 2026-08-17 |
| evaluation window | 1998-12-30 .. 2019-12-31 |
| vault window | 2020-01-02 .. 2026-08-17 |
| identity | data `77d8d9caea8a4ec0` · panel `586c15467c68bf0f` · config `1434ad9ce10cb1f6` |
| platform | x86_64darwin |
| family | seasonal_rule |

## Equity, rolling Sharpe, drawdown

![report_gen0001_equity.png](report_gen0001_equity.png)

*Net equity against SPY buy-and-hold. The strategy line is net of every modelled friction; the benchmark line is not, which flatters the benchmark and is the harder comparison to win.*

![report_gen0001_rolling_sharpe.png](report_gen0001_rolling_sharpe.png)

*Rolling 3-year net Sharpe. Flat stretches below zero are what a single headline Sharpe hides.*

![report_gen0001_drawdown.png](report_gen0001_drawdown.png)

*Drawdown from the running peak, strategy and benchmark.*

*Everything left of the dotted vault line was available to selection; everything right of it was touched only by the promotion gates.*

## Headline numbers

| statistic | value |
|---|---|
| pre-vault net Sharpe | +0.897 |
| pre-vault days scored | 5285 |
| vault net Sharpe | +0.974 |
| vault days | 1664 |
| Sharpe at 2x costs | +0.761 |
| bootstrap 95% CI of net Sharpe | [+0.458, +1.356] |
| CPCV paths net-positive | 100% of 28 (median path SR +0.73) |
| P(drawdown > 40% in 2 years) | 0.020 |

### Deflated Sharpe and PBO

- **DSR 0.1142 at N = 21 ledger trials, 2 vault trials** (sr0 threshold 0.0731, T = 5285 days, skew -0.009, kurtosis 5.512). The deflation uses the EMPIRICAL spread of trial Sharpes from the ledger, never a hardcoded count.
- Vault DSR **0.9899** at N = 2 vault trials — the count of times any candidate has been shown the post-2020-01-01 data at all.
- **PBO 0.507** (CSCV, cohort of 7 genomes, 12870 splits) — computed over this generation's returns matrix.
- DSR is an OPTIMISTIC correction here: evolutionary trials are correlated, and correlated trials deflate less than independent ones would. The vault and the paper stage sit above it for exactly that reason.

> **Ledger drift on this genome.** Its best-fidelity ledger row is from an earlier vintage (data `c8b7f490b91efedc`), so the trial-Sharpe dispersion that sets the DSR threshold is that vintage's, in an uncontrolled direction. Recorded here rather than buried in a log.

## The ten promotion gates

| gate | what it asks | value | threshold | |
|---|---|---|---|---|
| G1 | like-for-like | 77d8d9caea8a4ec0 / 586c15467c68bf0f / 1434ad9ce10cb1f6 | complete identity | PASS |
| G2 | DSR (pre-vault) | 0.114 | 0.950 | **FAIL** |
| G3 | vault confirmation | 0.974 / 0.990 | 0.000 / 0.900 | PASS |
| G4 | PBO (CSCV) | 0.507 | 0.200 | **FAIL** |
| G5 | CPCV 28 paths | 1.000 / 0.735 | 0.700 / 0.300 | PASS |
| G6 | bootstrap Sharpe CI | 0.458 | 0.000 | PASS |
| G7 | 2x cost stress | 0.761 / 0.848 | 0.000 / 0.500 | PASS |
| G8 | regime slices | -0.287 / 1 | -0.300 / 3 | **FAIL** |
| G9 | beats incumbent | n/a | 0.000 | PASS |
| G10 | ruin MC | 0.020 | 0.050 | PASS |

Deep-eval history row: promoted=0, candidates evaluated=2, gates failed=G2+G4+G8, complete=1.

Ledger drift recorded for this decision: `35d85a0408d2:c8b7f490b91efedc|dc1e8bed731b9232|6c8ac09323e93676;9353d613c7d3:c8b7f490b91efedc|dc1e8bed731b9232|6c8ac09323e93676`.

*All ten must pass; ties go to the incumbent. A single failure is a refusal, and a refusal is the system working.*

## Crisis regimes (gate G8)

| window | net return | days | |
|---|---:|---:|---|
| 2000-03-01 .. 2002-10-31 | -28.7% | 671 | under the soft floor (-5%) |
| 2008-09-01 .. 2009-03-31 | -19.8% | 146 | under the soft floor (-5%) |
| 2020-01-01 .. 2020-06-30 | +5.2% | 125 |  |
| 2022-01-01 .. 2022-12-31 | -18.2% | 251 | under the soft floor (-5%) |

*Windows are drawdown LEGS — peak to trough, not peak to recovery. A window containing the rebound measures the wrong thing: 2008-01..2009-12 comes out positive for a book that was destroyed in the autumn of 2008.*

## Costs

- Gross cumulative return **+1409.2%**, net **+1042.7%** over 5285 pre-vault days.
- Cost drag **133 bps/year** of equity; mean daily turnover **6.5%** of equity.
- Total frictions paid: **$28,117** on a $15,000 account.
- **Cost share of gross: 9.5%**

## Trial ledger

- **21 distinct genomes** have been evaluated at some fidelity and are on the ledger. That is the N every deflated Sharpe above is deflated by.
- **2 vault accesses** logged. Every look at post-2020-01-01 data is counted, including the ones that only stored an artifact.
- Screens count as trials. They exert selection pressure, so excluding them would make every DSR on this page optimistic.

## Hall of fame (top 10, pre-vault and ungated)

| # | genome | family | pre-vault SR | gen | born | op | parent |
|---|--------|--------|-------------:|----:|-----:|----|--------|
| 1 | `9353d613c7d3` | seasonal_rule | +0.896 | 1 | 1 | mutate | `b5efe2549ec6` |
| 2 | `35d85a0408d2` | seasonal_rule | +0.783 | 1 | 1 | mutate | `b5efe2549ec6` |
| 3 | `b5efe2549ec6` | seasonal_rule | +0.783 | 1 | 0 | elite | — |
| 4 | `699945479b17` | seasonal_rule | +0.782 | 1 | 1 | immigrant | — |
| 5 | `3e1b9c004175` | ridge | +0.735 | 0 | 0 | seed | — |
| 6 | `a46d3cf1ca37` | seasonal_rule | +0.734 | 1 | 1 | crossover | `b454e881a1fc` x `b454e881a1fc` |
| 7 | `b454e881a1fc` | seasonal_rule | +0.727 | 1 | 0 | elite | — |
| 8 | `ea46fc6d7a36` | mom_rule | +0.687 | 1 | 0 | elite | — |
| 9 | `7588d7dcc554` | hgb | +0.654 | 0 | 0 | seed | — |
| 10 | `77f8a5cf2db7` | mom_rule | +0.646 | 0 | 0 | seed | — |

*Lineage columns are how a genome got here: which operator made it, from which parent, in which generation.*

## Simulation vs paper trading

**Paper trading has not started.** The graduation rule is three consecutive weekly deep evals kept by the same champion (docs/DESIGN.md, "Graduation ladder"); there is no champion at all. Until then every number in this report is a simulation of the past, and the sim-vs-realised overlay this section will hold does not exist.

---

## Limitations — read these with every number above

- **Survivorship bias.** The universe is a snapshot of today's index membership,
  so every company that failed out of it is absent from all of these backtests:
  long results come out optimistic and short results pessimistic. Disclosed, not
  solved; delisted-inclusive (CRSP-class) data is the named upgrade path.
- **Non-stationarity.** 1990s microstructure is simulated with modern costs, and
  a decades-old edge may simply have been arbitraged away since. The vault, the
  regime gate and the paper stage are defenses, not proofs.
- **Multiple testing survives the gates.** Evolutionary trials are correlated, so
  the ledger DSR under-deflates; PBO does not cover designer-level choices at all.
  The only accumulating true out-of-sample is vault -> paper -> live.
- **Small-account sensitivity.** The $1 minimum commission and whole-share
  rounding are 5-10 bps each way on small positions; the 2x cost-stress gate and
  the cost-share line above are what keep that visible.
- **The cost model is proportional (bps), not per-share.** Prices are
  split-adjusted, so a $/share friction would silently become hundreds of bps on
  1990s adjusted prices. Real 1990s spreads were wider than modern bps: any edge
  that survives only at modern costs is suspect by construction.

**Survivorship: the universe is today's S&P 500 membership, so long results flatter and short results understate.**

**Backtest alpha is a claim about the past, not a guarantee — not financial advice.**
