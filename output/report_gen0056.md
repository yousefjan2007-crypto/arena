# arena — generation 56

**No champion.** Nothing has passed all ten promotion gates, so this system currently recommends nothing and holds nothing.

This report is about `33420b3d8d1b`, the leading candidate of the last deep evaluation, **which the gates REFUSED** (failed G2+G4+G8). It is shown because a refusal with its numbers attached is more useful than a blank page — not because it is close to being a champion.

| | |
|---|---|
| data as of | 2026-09-11 |
| evaluation window | 1998-12-30 .. 2022-12-30 |
| vault window | 2023-01-03 .. 2026-09-11 |
| identity | data `d1fd771212e4e0e5` · panel `6b488317bbbec2fa` · config `dd373ebf180943a7` |
| platform | x86_64linux (vendored siblings) |
| family | seasonal_rule |

## Equity, rolling Sharpe, drawdown

![report_gen0056_equity.png](report_gen0056_equity.png)

*Net equity against SPY buy-and-hold. The strategy line is net of every modelled friction; the benchmark line is not, which flatters the benchmark and is the harder comparison to win.*

![report_gen0056_rolling_sharpe.png](report_gen0056_rolling_sharpe.png)

*Rolling 3-year net Sharpe. Flat stretches below zero are what a single headline Sharpe hides.*

![report_gen0056_drawdown.png](report_gen0056_drawdown.png)

*Drawdown from the running peak, strategy and benchmark.*

*Everything left of the dotted vault line was available to selection; everything right of it was touched only by the promotion gates.*

## Headline numbers

| statistic | value |
|---|---|
| pre-vault net Sharpe | +1.327 |
| pre-vault days scored | 6041 |
| vault net Sharpe | +1.549 |
| vault days | 926 |
| Sharpe at 2x costs | +1.142 |
| bootstrap 95% CI of net Sharpe | [+0.932, +1.714] |
| CPCV paths net-positive | 100% of 28 (median path SR +1.17) |
| P(drawdown > 40% in 2 years) | 0.005 |

### Deflated Sharpe and PBO

- **DSR 0.0000 at N = 1621 ledger trials, 9 vault trials** (sr0 threshold 0.1382, T = 6041 days, skew 0.148, kurtosis 6.052). The deflation uses the EMPIRICAL spread of trial Sharpes from the ledger, never a hardcoded count.
- Vault DSR **0.9940** at N = 9 vault trials — the count of times any candidate has been shown the post-2023-01-01 data at all.
- **PBO n/a** (CSCV, 33420b3d8d1b is not one of the 32 genomes in the generation-56 returns matrix, so that cohort's PBO is not evidence about it) — **this candidate is not in the cohort matrix, so PBO is unmeasured and gate G4 fails on that alone.**
- DSR is an OPTIMISTIC correction here: evolutionary trials are correlated, and correlated trials deflate less than independent ones would. The vault and the paper stage sit above it for exactly that reason.

## The ten promotion gates

| gate | what it asks | value | threshold | |
|---|---|---|---|---|
| G1 | like-for-like | d1fd771212e4e0e5 / 6b488317bbbec2fa / dd373ebf180943a7 | complete identity | PASS |
| G2 | DSR (pre-vault) | 0.000 | 0.950 | **FAIL** |
| G3 | vault confirmation | 1.549 / 0.994 | 0.000 / 0.900 | PASS |
| G4 | PBO (CSCV) | n/a | 0.200 | **FAIL** |
| G5 | CPCV 28 paths | 1.000 / 1.173 | 0.700 / 0.300 | PASS |
| G6 | bootstrap Sharpe CI | 0.932 | 0.000 | PASS |
| G7 | 2x cost stress | 1.142 / 0.860 | 0.000 / 0.500 | PASS |
| G8 | regime slices | -0.200 / 2 | -0.300 / 3 | **FAIL** |
| G9 | beats incumbent | n/a | 0.000 | PASS |
| G10 | ruin MC | 0.005 | 0.050 | PASS |

Deep-eval history row: promoted=0, candidates evaluated=2, gates failed=G2+G4+G8, complete=1.

*All ten must pass; ties go to the incumbent. A single failure is a refusal, and a refusal is the system working.*

## Crisis regimes (gate G8)

| window | net return | days | |
|---|---:|---:|---|
| 2000-03-01 .. 2002-10-31 | +96.2% | 671 |  |
| 2008-09-01 .. 2009-03-31 | -7.3% | 146 | under the soft floor (-5%) |
| 2020-01-01 .. 2020-06-30 | +13.2% | 125 |  |
| 2022-01-01 .. 2022-12-31 | -20.0% | 251 | under the soft floor (-5%) |

*Windows are drawdown LEGS — peak to trough, not peak to recovery. A window containing the rebound measures the wrong thing: 2008-01..2009-12 comes out positive for a book that was destroyed in the autumn of 2008.*

## Costs

- Gross cumulative return **+21121.6%**, net **+11185.4%** over 6041 pre-vault days.
- Cost drag **264 bps/year** of equity; mean daily turnover **18.3%** of equity.
- Total frictions paid: **$497,091** on a $15,000 account.
- **Cost share of gross: 11.2%**

## Trial ledger

- **1621 distinct genomes** have been evaluated at some fidelity and are on the ledger. That is the N every deflated Sharpe above is deflated by.
- **10 vault accesses** logged. Every look at post-2023-01-01 data is counted, including the ones that only stored an artifact.
- Screens count as trials. They exert selection pressure, so excluding them would make every DSR on this page optimistic.

## Hall of fame (top 10, pre-vault and ungated)

| # | genome | family | pre-vault SR | gen | born | op | parent |
|---|--------|--------|-------------:|----:|-----:|----|--------|
| 1 | `33420b3d8d1b` | seasonal_rule | +1.327 | 50 | 50 | mutate | `1aec4f9c8871` |
| 2 | `1aec4f9c8871` | seasonal_rule | +1.307 | 49 | 49 | mutate | `84c31eead7a7` |
| 3 | `84c31eead7a7` | seasonal_rule | +1.299 | 50 | 50 | mutate | `4c930c7c5ac4` |
| 4 | `f92d4442d863` | seasonal_rule | +1.299 | 55 | 55 | mutate | `1e5d16cade27` |
| 5 | `8c4bf0eb0bee` | seasonal_rule | +1.299 | 49 | 49 | mutate | `c78907493761` |
| 6 | `c78907493761` | seasonal_rule | +1.299 | 48 | 48 | mutate | `f96cf3d59188` |
| 7 | `cbd61111ceb8` | seasonal_rule | +1.299 | 56 | 56 | crossover | `304e921105f9` x `1e5d16cade27` |
| 8 | `8e38a1585a4e` | seasonal_rule | +1.263 | 51 | 51 | mutate | `84c31eead7a7` |
| 9 | `f6534cfed801` | seasonal_rule | +1.262 | 23 | 23 | crossover | `4b899e1cb71e` x `4b899e1cb71e` |
| 10 | `a0a007afe3f1` | seasonal_rule | +1.262 | 27 | 16 | elite | `487279b1d7ad` |

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
