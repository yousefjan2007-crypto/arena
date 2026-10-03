# arena — generation 98

**No champion.** Nothing has passed all ten promotion gates, so this system currently recommends nothing and holds nothing.

This report is about `de2ba6a15d3d`, the leading candidate of the last deep evaluation, **which the gates REFUSED** (failed G2+G4+G8). It is shown because a refusal with its numbers attached is more useful than a blank page — not because it is close to being a champion.

| | |
|---|---|
| data as of | 2026-10-02 |
| evaluation window | 1998-12-30 .. 2022-12-30 |
| vault window | 2023-01-03 .. 2026-10-02 |
| identity | data `1d643a8361862e07` · panel `a1beb11ead2d1033` · config `dd373ebf180943a7` |
| platform | x86_64linux (vendored siblings) |
| family | seasonal_rule |

## Equity, rolling Sharpe, drawdown

![report_gen0098_equity.png](report_gen0098_equity.png)

*Net equity against SPY buy-and-hold. The strategy line is net of every modelled friction; the benchmark line is not, which flatters the benchmark and is the harder comparison to win.*

![report_gen0098_rolling_sharpe.png](report_gen0098_rolling_sharpe.png)

*Rolling 3-year net Sharpe. Flat stretches below zero are what a single headline Sharpe hides.*

![report_gen0098_drawdown.png](report_gen0098_drawdown.png)

*Drawdown from the running peak, strategy and benchmark.*

*Everything left of the dotted vault line was available to selection; everything right of it was touched only by the promotion gates.*

## Headline numbers

| statistic | value |
|---|---|
| pre-vault net Sharpe | +1.336 |
| pre-vault days scored | 6041 |
| vault net Sharpe | +1.196 |
| vault days | 941 |
| Sharpe at 2x costs | +1.183 |
| bootstrap 95% CI of net Sharpe | [+0.961, +1.717] |
| CPCV paths net-positive | 100% of 28 (median path SR +1.23) |
| P(drawdown > 40% in 2 years) | 0.010 |

### Deflated Sharpe and PBO

- **DSR 0.0000 at N = 3810 ledger trials, 13 vault trials** (sr0 threshold 0.1492, T = 6041 days, skew 0.470, kurtosis 10.015). The deflation uses the EMPIRICAL spread of trial Sharpes from the ledger, never a hardcoded count.
- Vault DSR **0.9594** at N = 13 vault trials — the count of times any candidate has been shown the post-2023-01-01 data at all.
- **PBO n/a** (CSCV, de2ba6a15d3d is not one of the 32 genomes in the generation-98 returns matrix, so that cohort's PBO is not evidence about it) — **this candidate is not in the cohort matrix, so PBO is unmeasured and gate G4 fails on that alone.**
- DSR is an OPTIMISTIC correction here: evolutionary trials are correlated, and correlated trials deflate less than independent ones would. The vault and the paper stage sit above it for exactly that reason.

## The ten promotion gates

| gate | what it asks | value | threshold | |
|---|---|---|---|---|
| G1 | like-for-like | 1d643a8361862e07 / a1beb11ead2d1033 / dd373ebf180943a7 | complete identity | PASS |
| G2 | DSR (pre-vault) | 0.000 | 0.950 | **FAIL** |
| G3 | vault confirmation | 1.196 / 0.959 | 0.000 / 0.900 | PASS |
| G4 | PBO (CSCV) | n/a | 0.200 | **FAIL** |
| G5 | CPCV 28 paths | 1.000 / 1.234 | 0.700 / 0.300 | PASS |
| G6 | bootstrap Sharpe CI | 0.961 | 0.000 | PASS |
| G7 | 2x cost stress | 1.183 / 0.885 | 0.000 / 0.500 | PASS |
| G8 | regime slices | -0.198 / 2 | -0.300 / 3 | **FAIL** |
| G9 | beats incumbent | n/a | 0.000 | PASS |
| G10 | ruin MC | 0.010 | 0.050 | PASS |

Deep-eval history row: promoted=0, candidates evaluated=2, gates failed=G2+G4+G8, complete=1.

*All ten must pass; ties go to the incumbent. A single failure is a refusal, and a refusal is the system working.*

## Crisis regimes (gate G8)

| window | net return | days | |
|---|---:|---:|---|
| 2000-03-01 .. 2002-10-31 | +29.6% | 671 |  |
| 2008-09-01 .. 2009-03-31 | -8.8% | 146 | under the soft floor (-5%) |
| 2020-01-01 .. 2020-06-30 | +10.4% | 125 |  |
| 2022-01-01 .. 2022-12-31 | -19.8% | 251 | under the soft floor (-5%) |

*Windows are drawdown LEGS — peak to trough, not peak to recovery. A window containing the rebound measures the wrong thing: 2008-01..2009-12 comes out positive for a book that was destroyed in the autumn of 2008.*

## Costs

- Gross cumulative return **+45042.8%**, net **+24372.4%** over 6041 pre-vault days.
- Cost drag **256 bps/year** of equity; mean daily turnover **18.1%** of equity.
- Total frictions paid: **$731,033** on a $15,000 account.
- **Cost share of gross: 9.4%**

## Trial ledger

- **3810 distinct genomes** have been evaluated at some fidelity and are on the ledger. That is the N every deflated Sharpe above is deflated by.
- **14 vault accesses** logged. Every look at post-2023-01-01 data is counted, including the ones that only stored an artifact.
- Screens count as trials. They exert selection pressure, so excluding them would make every DSR on this page optimistic.

## Hall of fame (top 10, pre-vault and ungated)

| # | genome | family | pre-vault SR | gen | born | op | parent |
|---|--------|--------|-------------:|----:|-----:|----|--------|
| 1 | `de2ba6a15d3d` | seasonal_rule | +1.384 | 97 | 95 | elite | `446f1cea63b6` |
| 2 | `f53489855b6f` | seasonal_rule | +1.384 | 96 | 96 | crossover | `de2ba6a15d3d` x `b2d9e1eabddc` |
| 3 | `94266f26903a` | seasonal_rule | +1.370 | 91 | 90 | elite | `b74051ecbc38` |
| 4 | `9bb8a901be40` | seasonal_rule | +1.370 | 91 | 91 | mutate | `930e54f99672` |
| 5 | `b74051ecbc38` | seasonal_rule | +1.370 | 91 | 85 | elite | `cc6c5546e8bc` |
| 6 | `d158a2601fea` | seasonal_rule | +1.370 | 91 | 91 | mutate | `c87630d5180f` |
| 7 | `92d3d4a15474` | seasonal_rule | +1.350 | 98 | 98 | crossover | `150e6e6da39b` x `2e205f0c0d2d` |
| 8 | `33244dc372cd` | seasonal_rule | +1.350 | 98 | 97 | elite | `446f1cea63b6` x `31abe48461ff` |
| 9 | `307c9ba00a6e` | seasonal_rule | +1.340 | 98 | 98 | crossover | `2d5910ff86dd` x `31abe48461ff` |
| 10 | `446f1cea63b6` | seasonal_rule | +1.336 | 98 | 98 | crossover | `446f1cea63b6` x `206e0326b880` |

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
