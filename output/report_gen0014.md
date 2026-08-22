# arena — generation 14

**No champion.** Nothing has passed all ten promotion gates, so this system currently recommends nothing and holds nothing.

This report is about `18beaccc7ce0`, the leading candidate of the last deep evaluation, **which the gates REFUSED** (failed G2+G4). It is shown because a refusal with its numbers attached is more useful than a blank page — not because it is close to being a champion.

| | |
|---|---|
| data as of | 2026-08-21 |
| evaluation window | 1998-12-30 .. 2019-12-31 |
| vault window | 2020-01-02 .. 2026-08-21 |
| identity | data `d3fdf1e8706240f5` · panel `d9088aa5ba5177b8` · config `08e08b3bdbf78812` |
| platform | x86_64linux (vendored siblings) |
| family | seasonal_rule |

## Equity, rolling Sharpe, drawdown

![report_gen0014_equity.png](report_gen0014_equity.png)

*Net equity against SPY buy-and-hold. The strategy line is net of every modelled friction; the benchmark line is not, which flatters the benchmark and is the harder comparison to win.*

![report_gen0014_rolling_sharpe.png](report_gen0014_rolling_sharpe.png)

*Rolling 3-year net Sharpe. Flat stretches below zero are what a single headline Sharpe hides.*

![report_gen0014_drawdown.png](report_gen0014_drawdown.png)

*Drawdown from the running peak, strategy and benchmark.*

*Everything left of the dotted vault line was available to selection; everything right of it was touched only by the promotion gates.*

## Headline numbers

| statistic | value |
|---|---|
| pre-vault net Sharpe | +1.233 |
| pre-vault days scored | 5285 |
| vault net Sharpe | +1.146 |
| vault days | 1668 |
| Sharpe at 2x costs | +1.158 |
| bootstrap 95% CI of net Sharpe | [+0.775, +1.683] |
| CPCV paths net-positive | 100% of 28 (median path SR +1.17) |
| P(drawdown > 40% in 2 years) | 0.000 |

### Deflated Sharpe and PBO

- **DSR 0.0117 at N = 131 ledger trials, 3 vault trials** (sr0 threshold 0.1087, T = 5285 days, skew 0.239, kurtosis 7.646). The deflation uses the EMPIRICAL spread of trial Sharpes from the ledger, never a hardcoded count.
- Vault DSR **0.9914** at N = 3 vault trials — the count of times any candidate has been shown the post-2020-01-01 data at all.
- **PBO 0.865** (CSCV, cohort of 6 genomes, 12870 splits) — computed over the returns matrix of generation 14, the cohort this record belongs to.
- DSR is an OPTIMISTIC correction here: evolutionary trials are correlated, and correlated trials deflate less than independent ones would. The vault and the paper stage sit above it for exactly that reason.

## The ten promotion gates

| gate | what it asks | value | threshold | |
|---|---|---|---|---|
| G1 | like-for-like | d3fdf1e8706240f5 / d9088aa5ba5177b8 / 08e08b3bdbf78812 | complete identity | PASS |
| G2 | DSR (pre-vault) | 0.012 | 0.950 | **FAIL** |
| G3 | vault confirmation | 1.146 / 0.991 | 0.000 / 0.900 | PASS |
| G4 | PBO (CSCV) | 0.865 | 0.200 | **FAIL** |
| G5 | CPCV 28 paths | 1.000 / 1.169 | 0.700 / 0.300 | PASS |
| G6 | bootstrap Sharpe CI | 0.775 | 0.000 | PASS |
| G7 | 2x cost stress | 1.158 / 0.939 | 0.000 / 0.500 | PASS |
| G8 | regime slices | -0.210 / 3 | -0.300 / 3 | PASS |
| G9 | beats incumbent | n/a | 0.000 | PASS |
| G10 | ruin MC | 0.000 | 0.050 | PASS |

Deep-eval history row: promoted=0, candidates evaluated=2, gates failed=G2+G4, complete=1.

*All ten must pass; ties go to the incumbent. A single failure is a refusal, and a refusal is the system working.*

## Crisis regimes (gate G8)

| window | net return | days | |
|---|---:|---:|---|
| 2000-03-01 .. 2002-10-31 | +6.1% | 671 |  |
| 2008-09-01 .. 2009-03-31 | +0.0% | 146 |  |
| 2020-01-01 .. 2020-06-30 | +0.3% | 125 |  |
| 2022-01-01 .. 2022-12-31 | -21.0% | 251 | under the soft floor (-5%) |

*Windows are drawdown LEGS — peak to trough, not peak to recovery. A window containing the rebound measures the wrong thing: 2008-01..2009-12 comes out positive for a book that was destroyed in the autumn of 2008.*

## Costs

- Gross cumulative return **+4418.8%**, net **+3484.6%** over 5285 pre-vault days.
- Cost drag **110 bps/year** of equity; mean daily turnover **6.9%** of equity.
- Total frictions paid: **$71,215** on a $15,000 account.
- **Cost share of gross: 5.7%**

## Trial ledger

- **131 distinct genomes** have been evaluated at some fidelity and are on the ledger. That is the N every deflated Sharpe above is deflated by.
- **4 vault accesses** logged. Every look at post-2020-01-01 data is counted, including the ones that only stored an artifact.
- Screens count as trials. They exert selection pressure, so excluding them would make every DSR on this page optimistic.

## Hall of fame (top 10, pre-vault and ungated)

| # | genome | family | pre-vault SR | gen | born | op | parent |
|---|--------|--------|-------------:|----:|-----:|----|--------|
| 1 | `18beaccc7ce0` | seasonal_rule | +1.233 | 14 | 13 | elite | `487279b1d7ad` |
| 2 | `487279b1d7ad` | seasonal_rule | +1.233 | 14 | 9 | elite | `19389d18f6e0` |
| 3 | `70979f35f561` | seasonal_rule | +1.233 | 14 | 10 | elite | `487279b1d7ad` |
| 4 | `07c92d843434` | seasonal_rule | +1.216 | 13 | 11 | elite | `19389d18f6e0` |
| 5 | `ecdd3719a12d` | seasonal_rule | +1.216 | 14 | 14 | crossover | `07c92d843434` x `18beaccc7ce0` |
| 6 | `0c85f7d6d52d` | seasonal_rule | +1.213 | 12 | 12 | mutate | `521733d9920e` |
| 7 | `19389d18f6e0` | seasonal_rule | +1.213 | 12 | 12 | crossover | `19389d18f6e0` x `487279b1d7ad` |
| 8 | `521733d9920e` | seasonal_rule | +1.212 | 11 | 11 | crossover | `487279b1d7ad` x `19389d18f6e0` |
| 9 | `2a1654d862e1` | seasonal_rule | +1.196 | 14 | 14 | mutate | `487279b1d7ad` |
| 10 | `59eea337e2ba` | seasonal_rule | +1.196 | 13 | 13 | mutate | `487279b1d7ad` |

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
