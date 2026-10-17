# arena — generation 125

**No champion.** Nothing has passed all ten promotion gates, so this system currently recommends nothing and holds nothing.

This report is about `b2f25a455c8c`, the leading candidate of the last deep evaluation, **which the gates REFUSED** (failed G2+G4). It is shown because a refusal with its numbers attached is more useful than a blank page — not because it is close to being a champion.

| | |
|---|---|
| data as of | 2026-10-16 |
| evaluation window | 1998-12-30 .. 2022-12-30 |
| vault window | 2023-01-03 .. 2026-10-16 |
| identity | data `5078f86a6fe70a30` · panel `21970453cc8ac3fb` · config `dd373ebf180943a7` |
| platform | x86_64linux (vendored siblings) |
| family | seasonal_rule |

## Equity, rolling Sharpe, drawdown

![report_gen0125_equity.png](report_gen0125_equity.png)

*Net equity against SPY buy-and-hold. The strategy line is net of every modelled friction; the benchmark line is not, which flatters the benchmark and is the harder comparison to win.*

![report_gen0125_rolling_sharpe.png](report_gen0125_rolling_sharpe.png)

*Rolling 3-year net Sharpe. Flat stretches below zero are what a single headline Sharpe hides.*

![report_gen0125_drawdown.png](report_gen0125_drawdown.png)

*Drawdown from the running peak, strategy and benchmark.*

*Everything left of the dotted vault line was available to selection; everything right of it was touched only by the promotion gates.*

## Headline numbers

| statistic | value |
|---|---|
| pre-vault net Sharpe | +1.385 |
| pre-vault days scored | 6041 |
| vault net Sharpe | +1.474 |
| vault days | 951 |
| Sharpe at 2x costs | +1.224 |
| bootstrap 95% CI of net Sharpe | [+1.014, +1.770] |
| CPCV paths net-positive | 100% of 28 (median path SR +1.30) |
| P(drawdown > 40% in 2 years) | 0.010 |

### Deflated Sharpe and PBO

- **DSR 0.0000 at N = 5292 ledger trials, 16 vault trials** (sr0 threshold 0.1557, T = 6041 days, skew 0.266, kurtosis 7.844). The deflation uses the EMPIRICAL spread of trial Sharpes from the ledger, never a hardcoded count.
- Vault DSR **0.9888** at N = 16 vault trials — the count of times any candidate has been shown the post-2023-01-01 data at all.
- **PBO n/a** (CSCV, b2f25a455c8c is not one of the 32 genomes in the generation-125 returns matrix, so that cohort's PBO is not evidence about it) — **this candidate is not in the cohort matrix, so PBO is unmeasured and gate G4 fails on that alone.**
- DSR is an OPTIMISTIC correction here: evolutionary trials are correlated, and correlated trials deflate less than independent ones would. The vault and the paper stage sit above it for exactly that reason.

## The ten promotion gates

| gate | what it asks | value | threshold | |
|---|---|---|---|---|
| G1 | like-for-like | 5078f86a6fe70a30 / 21970453cc8ac3fb / dd373ebf180943a7 | complete identity | PASS |
| G2 | DSR (pre-vault) | 0.000 | 0.950 | **FAIL** |
| G3 | vault confirmation | 1.474 / 0.989 | 0.000 / 0.900 | PASS |
| G4 | PBO (CSCV) | n/a | 0.200 | **FAIL** |
| G5 | CPCV 28 paths | 1.000 / 1.297 | 0.700 / 0.300 | PASS |
| G6 | bootstrap Sharpe CI | 1.014 | 0.000 | PASS |
| G7 | 2x cost stress | 1.224 / 0.883 | 0.000 / 0.500 | PASS |
| G8 | regime slices | -0.164 / 3 | -0.300 / 3 | PASS |
| G9 | beats incumbent | n/a | 0.000 | PASS |
| G10 | ruin MC | 0.010 | 0.050 | PASS |

Deep-eval history row: promoted=0, candidates evaluated=2, gates failed=G2+G4, complete=1.

*All ten must pass; ties go to the incumbent. A single failure is a refusal, and a refusal is the system working.*

## Crisis regimes (gate G8)

| window | net return | days | |
|---|---:|---:|---|
| 2000-03-01 .. 2002-10-31 | +60.4% | 671 |  |
| 2008-09-01 .. 2009-03-31 | +4.9% | 146 |  |
| 2020-01-01 .. 2020-06-30 | +11.8% | 125 |  |
| 2022-01-01 .. 2022-12-31 | -16.4% | 251 | under the soft floor (-5%) |

*Windows are drawdown LEGS — peak to trough, not peak to recovery. A window containing the rebound measures the wrong thing: 2008-01..2009-12 comes out positive for a book that was destroyed in the autumn of 2008.*

## Costs

- Gross cumulative return **+82427.2%**, net **+40247.8%** over 6041 pre-vault days.
- Cost drag **299 bps/year** of equity; mean daily turnover **21.4%** of equity.
- Total frictions paid: **$1,453,736** on a $15,000 account.
- **Cost share of gross: 10.0%**

## Trial ledger

- **5292 distinct genomes** have been evaluated at some fidelity and are on the ledger. That is the N every deflated Sharpe above is deflated by.
- **17 vault accesses** logged. Every look at post-2023-01-01 data is counted, including the ones that only stored an artifact.
- Screens count as trials. They exert selection pressure, so excluding them would make every DSR on this page optimistic.

## Hall of fame (top 10, pre-vault and ungated)

| # | genome | family | pre-vault SR | gen | born | op | parent |
|---|--------|--------|-------------:|----:|-----:|----|--------|
| 1 | `b2f25a455c8c` | seasonal_rule | +1.468 | 123 | 123 | mutate | `9519a24196a6` |
| 2 | `c03f8cadda5a` | seasonal_rule | +1.447 | 123 | 123 | mutate | `159b669aabbc` |
| 3 | `e424b80b412c` | seasonal_rule | +1.422 | 113 | 113 | mutate | `ec98db8dbd80` |
| 4 | `abd250c7ce1d` | seasonal_rule | +1.421 | 124 | 124 | mutate | `800e765f95d6` |
| 5 | `2709c5c68243` | seasonal_rule | +1.417 | 121 | 121 | mutate | `10208e494415` |
| 6 | `74e0f2adc61d` | seasonal_rule | +1.415 | 117 | 111 | elite | `75ef2484d297` x `3bd8908e2e0e` |
| 7 | `b37d771eaeeb` | seasonal_rule | +1.413 | 109 | 109 | crossover | `dd03715ec671` x `0b25006900ce` |
| 8 | `419cac1894c5` | seasonal_rule | +1.411 | 124 | 124 | mutate | `9519a24196a6` |
| 9 | `4a0c5960fc18` | seasonal_rule | +1.406 | 117 | 117 | crossover | `e702cfebbddb` x `6691a659dba8` |
| 10 | `68cc6ed79b20` | seasonal_rule | +1.384 | 125 | 120 | elite | `79817fa42195` |

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
