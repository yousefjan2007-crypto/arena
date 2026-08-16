# `arena/` — Self-Improving Trading System (Evolving Strategy Arena)

## Context

The user wants a self-improving algorithm that teaches itself to trade by
interacting with a sandbox market built from years of historical data, aiming
for positive alpha and Sharpe. Decisions confirmed with the user:

- **Market:** daily US stocks, long/short.
- **Learner:** an **evolving strategy arena** — a population of complete
  strategy variants (signal + sizing + entry/exit/risk rules) competes in a
  sandbox replaying ~30 years of daily data; each generation, winners spawn
  mutated offspring, losers are culled; champions are promoted only through
  anti-luck statistical gates. Deep RL deferred (Python 3.9 blocks torch/gym),
  but the env exposes a clean reset/step interface so an RL species can be
  added later.
- **Lifecycle:** research sandbox → graduation gates → Alpaca paper → real
  money ($10K–$25K), with learning continuing at every stage (challengers keep
  evolving and must dethrone the live champion through the same gates).
- **Account realism:** sandbox charges honest small-account costs
  (commission/spread/slippage/borrow/margin), whole shares, position limits,
  no intraday round-trips (PDT-safe under $25K).

New standalone project `~/arena/` (own git repo), reusing siblings by the
established path-injection pattern.

- **Execution home (user decision):** a **public GitHub repo** running on
  **GitHub Actions cron** — the `valuation-monitor` pattern (repo is the
  source of truth; each run pulls state, computes, commits results back), so
  everything runs 24/7 with the laptop off, at $0 (public repos get unlimited
  standard-runner minutes; the private-repo 2,000-min pool shared with
  outreach-finder is untouched). The Mac remains a dev/verify machine and an
  optional booster (pull → run extra generations → push).

## Why shaped this way (exploration findings, condensed)

`signal_lab` is already a self-improving *prediction* system, and all three of
its learning loops are inert — each failure is a design requirement here:

1. No hypothesis generator (fixed architecture refit weekly, must beat champ
   by +0.10 Sharpe from ~18 monthly obs) → arena needs a real search space AND
   ~5,300 daily OOS observations (full-history walk-forward, not the last 378
   days that `cv.py`'s `first=` logic limits signal_lab to).
2. Censored ledger (only top-N alerts; 2 resolved trades ever) → arena logs
   EVERY simulated decision, attributable to a genome hash.
3. `champion.joblib` overwritten in place, `model_version` constant → arena
   registry is immutable/content-hashed with full lineage + rollback.
4. Champion promoted from a 27-symbol `--quick` run compared against
   164-symbol challengers; stale stored Sharpe → gates compare like-for-like:
   same data hash, same window, incumbent re-simulated fresh.
5. DSR `n_trials=50` hardcoded; "PBO" is a 6-fold IC-sign count; real
   `CombinatorialPurgedCV` unused → arena wires DSR to an actual trial ledger
   and computes CSCV PBO.
6. No market sim anywhere in the workspace (futures_bot pre-resolves trades
   before account logic; nothing gym-like exists) → the daily-bar step env is
   a new component.

**Reused directly:** `signal_lab/cv.py` (purge/embargo correctness contract),
`signal_lab/features.py` + `macro.py` (point-in-time panel; `_expanding_seasonal`),
`signal_lab/universe.py`, `signal_lab/alerts.py`, `sell_in_may/data.py` (cache),
`sell_in_may/montecarlo.py` (4 seeded synthetic-path engines for stress MC).
**Copied with attribution** (importing futures_bot would clash on `config`):
`futures_bot/evaluate.py` `deflated_sharpe` (:208, empirical trial-std) and
`pbo_cscv` (:254) — ~120 lines of pure functions. Avoid its known flaw
(selecting in-sample on gross but scoring OOS net).

**Deps:** Python 3.9.5; numpy/pandas/scipy/sklearn/statsmodels/arch/yfinance/
matplotlib/joblib only. No torch/gym/xgboost — don't require them.
**Conventions held everywhere:** `np.random.default_rng(config.SEED)` only; no
wall-clock in compute paths; CSV/JSON/joblib storage; launchd one-shot jobs;
certifi SSL for urllib; secrets via env → `config.local.json` →
`vrp_backtest/monitor_config.json`.

## Phase 0 (prerequisite): repair the shared data cache

`signal_lab/deepvalue/dv_monitor.py` lines 163 and 187 call
`smdata.fetch_history(tk, start=pd.Timestamp.now() - DateOffset(years=6))`;
`sell_in_may/data.py:fetch_history` unconditionally overwrites the cache CSV →
23+ megacaps (AAPL MSFT AMZN GOOGL META NVDA JPM KO DIS T F NKE CVS CCJ,
BTC-USD, ES=F NQ=F GC=F CL=F SI=F, PYPL SOFI AMD…) hold only ~6 years from
2020-08-14, while SPY et al. go back to 1995.

Fix: at both call sites fetch full history (`start="1995-01-01"`) and slice
locally to the trailing `HIST_YEARS+1` window **anchored to the data's last
date** (`price.index.max() - DateOffset(years=HIST_YEARS+1)`), not the wall
clock — preserves dv_monitor behavior exactly and removes the workspace's one
no-wall-clock violation in a compute path. Then delete + refetch the truncated
CSVs (regenerate the list by scanning cache start-dates > 1996 for non-crypto).

Verify: start-date scan shows every equity ≥ its listing date or 1995;
`python3 signal_lab/deepvalue/dv_monitor.py` dry run output unchanged;
`python3 signal_lab/verify.py` still passes.

## Project layout

```
arena/
  config.py         # ALL knobs; path-injects sell_in_may + signal_lab (superset
                    # re-export, ROOT restored after — the documented clobber trap);
                    # rule: every arena module imports config FIRST
  datafeed.py       # cache-first loader -> MarketData (aligned numpy arrays); integrity checks
  features.py       # adapter over signal_lab features.build_panel + macro; joblib-cached by data_hash
  env.py            # MarketEnv reset/step sandbox (account, next-open fills, costs, limits, logging)
  genome.py         # Genome dataclasses, bounds, canonical JSON, sha256 hash, mutate/crossover
  strategy.py       # Genome -> StrategyAgent (signal model + sizing + exits; purged in-sim refits)
  evaluate.py       # fidelity ladder F0/F1/F2 + deflated_sharpe/pbo_cscv (copied from futures_bot)
  evolution.py      # generation cycle: elitism, tournament, mutate/crossover, immigrants, hall of fame
  ledger.py         # append-only trial ledger (EVERY genome), vault-access counter, DSR inputs
  registry.py       # immutable artifacts keyed by genome hash; champion pointer; lineage; rollback
  gates.py          # promotion/graduation gate stack (pure functions: metrics -> gate report)
  reports.py        # weekly honest report (md+png)
  alerts_arena.py   # wrapper over signal_lab/alerts.py + vrp secrets chain
  broker_paper.py   # Alpaca paper adapter (signal_lab/broker.py place/close/position shape)
  run_generation.py # NIGHTLY one-shot: refresh data -> F0 -> F1 -> evolve -> persist -> alert -> exit
  run_deepeval.py   # WEEKLY one-shot: F2 -> gates -> maybe promote -> report -> alert -> exit
  run_paper.py      # DAILY one-shot (paper stage): champion targets -> orders -> sim-shadow compare
  verify.py         # test suite below; must pass before anything is scheduled
  state/            # population.json, trial_ledger.csv, hall_of_fame.json, champion.json,
                    # champion_history.csv, paper_ledger.csv, returns/gen_<NNNN>.npz
  artifacts/genomes/<hash12>/  # genome.json, metrics.json, daily_returns.csv.gz,
                               # decisions.csv.gz (F2+), model.joblib
  output/           # report_gen<NNNN>.md, *.png
  requirements.txt  # EXACT pins matching the Mac (numpy 2.0.2, pandas 2.3.3,
                    # scikit-learn 1.6.1, scipy, statsmodels, arch, yfinance,
                    # matplotlib, joblib) + python 3.9 via actions/setup-python
  .github/workflows/  # generation.yml, deepeval.yml, paper.yml (cron; see Scheduling)
```

Everything in `state/`+`artifacts/` is append-only or immutable; only
`champion.json` (a pointer) is rewritten, and every rewrite is appended to
`champion_history.csv`.

Key `config.py` defaults (all thresholds live here — no hidden module
constants): `SEED=12345`; `DATA_START="1995-01-01"`; `UNIVERSE_SIZE=120`;
`VAULT_START="2020-01-01"`; `START_CASH=15_000` (mid of user's $10–25K,
conservative); `MAX_GROSS_LEV=1.5, MAX_NET_LEV=1.0, MAX_POSITIONS=20,
MAX_NAME_WEIGHT=0.20, MIN_POSITION_USD=500, WHOLE_SHARES=True`;
`COMMISSION_BPS=0.5, COMMISSION_MIN=1.00, HALF_SPREAD_BPS=2.5,
LEV_EPS=0.01, SLIPPAGE_BPS=2.0, BORROW_ANNUAL=0.01,
MARGIN_ANNUAL=0.065, NO_INTRADAY_EXITS=True`; `WF_MIN_TRAIN_DAYS=1008,
WF_EMBARGO_DAYS=21`; the four `CV_*` constants signal_lab's `cv.py` import
needs; `CPCV_GROUPS=8, CPCV_K=2, PBO_SPLITS=16, BOOT_ITERS=5000,
BOOT_BLOCK=21`; `POP_SIZE=64, ELITE_N=4, IMMIGRANT_N=4, TOURNAMENT_K=4,
SCREEN_FRAC=0.5, PARSIMONY_PENALTY=0.01, GEN_TIME_BUDGET_MIN=180, N_JOBS=8`;
gate thresholds (table below); `EXECUTION_MODE="sandbox"` ("paper"/"live" —
live is a human-only flip).

## Sandbox env (`env.py`)

**Timing (one clock):** at close of day `t` the agent sees features through
`t` (asof-truncated), prices ≤ `t`, and account state; it emits target
weights. Env converts weight deltas to whole-share orders **filled at open of
`t+1`** at `open ± half_spread ± slippage`, charges commission, daily borrow
on short market value, margin interest on negative cash, then marks to market
at close `t+1`. Reward = `log(equity[t+1]/equity[t])` net. Stops evaluated at
close, exit at next open. No same-day round trips.

```python
class MarketEnv:
    def __init__(self, market: MarketData, cost: CostModel, start, end, rng,
                 decision_log=None): ...
    def reset(self) -> dict   # obs: t, date, features row, close row,
                              # position_shares, cash, equity, drawdown, days_held
    def step(self, target_w: np.ndarray) -> (obs, reward, done, info)
    # info: fills, commissions, spread_cost, slippage, borrow, margin, rejected_orders
```

Env (not the strategy) enforces: whole shares; drop orders < `MIN_POSITION_USD`;
per-name cap; pro-rata scale-down if gross/net leverage exceeded; reject
shorts on symbols with no data at `t+1` (IPO/delist edges); max positions.
`CostModel.stress_mult` scales every friction (used by the 2× stress gate).

**Invariants (debug-asserted every step, tested):** `equity == cash +
Σ shares·close` to 1e-6; costs ≥ 0; limits hold post-fill; integral shares;
equity path fully replayable from the decision log.

**Decision logging (log everything — the signal_lab lesson):**
- Tier A, every genome every day: `state/returns/gen_<NNNN>.npz` — (days ×
  genomes) matrices of net return, gross return, turnover, costs, indexed by
  genome hash (~2 MB/generation). This is the DSR/PBO input artifact.
- Tier B, F2 finalists + champion: position-level `decisions.csv.gz`
  (date, symbol, side, shares, fill_px, each cost component, weights,
  reason ∈ {rebalance, stop, regime, derisk}), rows carry genome + config hash.

## Genome / strategy space (`genome.py`, `strategy.py`)

Three frozen dataclass gene blocks; sha256(canonical JSON)[:12] is the identity:

- **SignalGene:** `family ∈ {mom_rule, meanrev_rule, seasonal_rule, ridge,
  logistic, hgb}`, `horizon ∈ {5,10,21,63}`, `features` (3–15 of the ~35-col
  point-in-time library; model families only), family-bounded `params`
  (e.g. hgb: depth {2,3,4}, lr {.03,.05,.1}, iters {100..300},
  min_leaf {100..400}, `random_state=SEED`), `refit_days ∈ {63,126,252}`.
  Model families use the signal_lab plumbing contract (Pipeline step named
  "clf", predict_proba over {-1,0,+1}, sample_weight, X.values) so models
  stay reusable both ways.
- **PortfolioGene:** `n_long 0–12, n_short 0–8 (sum ≥ 3)`, `weighting ∈
  {equal, score, inv_vol}`, `gross ∈ {0.6,0.8,1.0,1.3}`, `vol_target ∈
  {None,.10,.15,.20}`, `rebalance_days ∈ {1,5,21}`.
- **RiskGene:** `stop_loss ∈ {None,.05,.10,.15}`, `trail_stop ∈ {None,.10,.20}`,
  `regime_filter ∈ {None, spy_200dma, vix_pct_80}` with `regime_scale ∈ {0,.5}`,
  `dd_limit ∈ {None,.10,.15,.20}` (halve gross until half-recovered).

**Operators (deterministic):** child rng = `default_rng(stable_hash(SEED,
generation, parent_hash, child_idx))`. Mutation: jitter one param to a grid
neighbor (p=.6) / add-drop-swap a feature (.3) / resample a gene block (.15) /
family hop (.05). Crossover: uniform at gene-block level, 25% of offspring.
Bounds re-clamped; encode→decode→hash round-trip is identity (tested).
Effective space ~10⁶–10⁸ — searchable by 64 genomes × ~40 evals/night.

## Evolution + honest evaluation (`evaluate.py`, `evolution.py`, `ledger.py`)

**The vault (structural anti-snooping):** all fitness/selection uses OOS days
**before `VAULT_START` (2020-01-01) only**. 2020→present is touched solely by
weekly promotion gates; every vault access increments `vault_trials` and
vault-DSR is deflated by that count. Selection can't overfit the recent 6
years; paper/live extends the vault forward.

**Nightly generation (`run_generation.py`):**
1. Data refresh via `sell_in_may/data.py` (cache-first; abort with alert if
   staleness > 5 days). Rebuild feature panel (joblib-cached by data_hash).
2. **F0 screen (all 64):** episodes on 3 disjoint 5-year eras
   (1997–2001, 2007–2011, 2015–2019), 60-symbol subuniverse, weekly rebalance,
   coarse refits. Score = mean net Sharpe across eras − parsimony·n_features.
   Every screened genome → trial ledger (fidelity=F0).
3. **F1 full eval (top 32 + elites):** one anchored streaming walk-forward
   episode 1995→present, full universe, genome's own cadence, refits every
   `refit_days` on rows with `t1 ≤ refit_date − 21d embargo` (purging inherent
   to streaming — nothing after the fit date exists yet; fit-audit recorded).
   Scoring window ≈ 1999→2020: **~5,300 daily OOS observations**. Vault days
   simulated but stored separately, never scored. Ledger rows (fidelity=F1).
4. **Evolve:** elitism (4) → tournament(k=4) parents → offspring 75% mutants /
   25% crossover + 4 random immigrants → next 64. Hall of fame (top-10
   all-time, lineage).
5. Persist (population, returns npz, ledger, HoF) → one-line Telegram → exit.
   Checkpointed per phase; killed runs resume; ledger idempotent by
   (genome hash, generation).

**Trial ledger = DSR input:** `n_trials()` = distinct genome hashes ever
evaluated (screens count — they exert selection); `trial_sr_std()` = empirical
std of F1 pre-vault Sharpes (futures_bot approach; never a hardcoded count).
Reports state the caveat: evolutionary trials are correlated → DSR is an
optimistic correction — which is why the vault and paper stage sit above it.

**Compute:** joblib parallelism, OMP pinned to 1/worker, panel shared via
memmap; HGB ≈ 20–40s/fit × ~55 refits worst case. On the primary GitHub
runner (ubuntu-latest, 4 vCPU → `N_JOBS=4`): F0 ≈ 50 min, F1 ≈ 2–3 h →
`GEN_TIME_BUDGET_MIN` set to fit the session window, checkpoint/resume
carries any overflow to the next session. On the Mac booster (14 cores,
`N_JOBS=8`) a full generation is ~1.5–2.5 h. Note: determinism is guaranteed
per-platform (BLAS float differences between Apple Silicon and x86 Linux can
flip low-order bits); once scheduled, **the cloud is canonical** and ledger
rows are platform-tagged.

**Weekly F2 deep eval (`run_deepeval.py`)** on hall-of-fame leaders +
re-simulated champion: CSCV PBO (S=16) on the cohort's pre-vault returns
matrix; per-candidate CPCV with real refits (8C2 = 28 paths, top 1–2 only);
block-bootstrap Sharpe CI (5000×, block 21); 2× cost stress + 3× borrow;
regime slices; ruin MC on `sell_in_may/montecarlo.py` GARCH-t + block-bootstrap
synthetic paths (200 seeded, 2-yr horizon).

## Promotion gates (`gates.py`) — all must pass; ties → incumbent

| # | Gate | Threshold |
|---|------|-----------|
| G1 | Like-for-like | identical data_hash + cost-config hash + window; incumbent re-simulated fresh |
| G2 | DSR (pre-vault) | ≥ 0.95 at n_trials = full ledger, empirical trial_sr_std |
| G3 | Vault confirmation | vault net Sharpe > 0 AND vault DSR ≥ 0.90 at N = vault_trials |
| G4 | PBO (CSCV, S=16) | ≤ 0.20 |
| G5 | CPCV 28 paths | ≥ 70% net-positive; median path Sharpe ≥ 0.30 |
| G6 | Bootstrap 95% CI of net Sharpe | lower bound > 0 |
| G7 | 2× cost stress | Sharpe > 0 and ≥ 0.5× base |
| G8 | Regime slices (2000–02, 2008–09, 2020H1, 2022) | no slice < −30%; ≥3 of 4 > −5% |
| G9 | Beats incumbent | net Sharpe ≥ incumbent+0.15 same-window AND wins ≥60% of rolling 3-yr windows (skip if none) |
| G10 | Ruin MC | P(DD > 40% in 2 yrs) < 5% |

Promotion repoints `champion.json` to the immutable artifact (lineage: parent
hash, mutation, birth generation, eval window, gate report). Rollback =
repoint to any prior hash.

## Graduation ladder (learning never stops)

- **Sandbox → paper:** same champion survives 3 consecutive weekly deep-evals,
  then `run_paper.py` is scheduled (Alpaca paper, whole shares, MOO orders at
  next open — mirrors the env's fill model).
- **Paper (≥126 trading days):** daily sim-shadow vs paper-fill ledger.
  Go-live *recommendation* requires: median |fill slippage| < 10 bps,
  corr(daily paper, sim-shadow) ≥ 0.80, cumulative paper return inside the
  shadow's bootstrap 90% band, no kill-switch trips in 60 days. **The system
  never self-starts live** — it emits a GO-LIVE-candidate alert with the
  evidence table; a human flips `EXECUTION_MODE`.
- **Live ($10–25K):** same loop. Kill-switches (auto-flatten → demote to paper
  → alert): live DD > 15% from live peak; tracking error > 25 bps/day for 20
  days; data staleness > 3 days (hold + alert only).
- **Post-live:** nightly evolution continues; a challenger replaces the live
  genome only via G1–G10 against the re-evaluated live champion PLUS ≥60 days
  paper-shadowing alongside live with better net performance and in-bound
  tracking error.

## Scheduling & reporting (GitHub Actions, public repo)

The `valuation-monitor` pattern, hardened: each workflow checks out the repo,
`pip install -r requirements.txt` (Python 3.9 via actions/setup-python, exact
version pins), runs the one-shot script (fetch → evaluate → append → alert →
exit; anti-spam via last-logged-state), and **commits `state/` + `artifacts/`
+ `output/` back** with `git pull --rebase` retry. An Actions `concurrency:`
group per workflow prevents overlapping runs; the Mac booster uses plain git
(pull → run → push; a rejected push aborts cleanly — ledger writes are
idempotent by genome hash + generation so a lost race wastes nothing).

- `generation.yml` — **twice daily, 7 days/week** (e.g. 02:30 and 14:30 UTC),
  up to ~4 h/session on the 4-vCPU runner. This is the honest reading of "as
  often as possible": beyond ~2 generations/day on data that only changes
  once per trading day, extra runs mostly inflate the trial count (which
  deflates DSR) without exploring meaningfully more — ~14 generations/week is
  already 3× the original nightly plan. Weekend runs are pure search on
  static data.
- `deepeval.yml` — Sat 12:00 UTC (F2 + gates + weekly report; up to 6 h).
- `paper.yml` — weekdays 13:00 UTC (09:00 ET), only once the paper stage
  begins: submits MOO orders to Alpaca paper well before the 09:30 open
  auction, tolerant of Actions cron jitter.

Free-tier fit: public-repo standard runners are free/unlimited; jobs stay
under the 6 h cap; daily state commits keep the repo active (scheduled
workflows are auto-disabled only after 60 days of inactivity). Secrets
(Telegram/ntfy, later Alpaca paper keys) live in Actions repo secrets — the
outreach-finder pattern; nothing sensitive in the repo. Repo growth policy:
the permanent record is the (small) trial ledger CSV + hall-of-fame +
champion artifacts; per-generation returns `.npz` older than ~90 generations
are pruned by the nightly job.

Nightly Telegram one-liner (gen #, pop evaluated, ledger N, best/median F1
Sharpe, champion status). Weekly report md+png: champion equity vs SPY,
rolling 3-yr Sharpe, drawdown, gates table with actual numbers, **DSR with
trial count printed**, PBO, HoF lineage, cost-stress + regime tables, and
(once paper starts) sim-vs-realized overlay. Mandatory footer: survivorship
disclosure + "backtest alpha is a claim about the past, not a guarantee — not
financial advice."

## verify.py (must pass before anything is scheduled)

1. **Planted leak through the full env+eval path** (amended in Phase 3 —
   the original wording asserted the impossible): the streaming purged
   walk-forward defends against FIT-time leakage, not act-time leakage. A
   label-overlap leak that inflates a naive pooled evaluator (IC ≈ 0.5)
   must collapse to ≈ 0 through the arena's streaming purged path. A
   feature that *is* the future return (`leak_fwd`) wins at act time no
   matter how the model was fit — the test pins that behavior explicitly
   and documents the structural defense: act-time-leaky features cannot
   exist in the production panel (PIT builder, proven upstream by
   `signal_lab/verify.py::test_pointintime`); the arena's test-only
   injection path is the only way to create one.
2. **Determinism:** 2 generations on a synthetic 10-symbol market, same SEED,
   run twice → identical genome hashes, byte-identical returns .npz, identical
   ledger.
3. **Accounting fuzz:** 500 seeded random-action steps → equity identity to
   1e-6, limits hold, integral shares, costs ≥ 0, replay-from-log equality.
4. **Fill timing:** close-t decision fills exactly at open t+1 with correct
   spread/slippage sign; no same-day exits.
5. **Streaming purge:** every training row's `t1 ≤ refit_date − embargo` in
   the fit audit, all families.
6. **Cost linearity:** stress_mult 0/1/2 → costs 0/c/2c; zero-cost equity ≥
   costed equity pointwise.
7. **Gates:** all-pass promotes; each single violation blocks; tie →
   incumbent; data_hash mismatch forces incumbent re-eval.
8. **Trial ledger:** k evals → exactly k rows (idempotent on resume); DSR
   monotone non-increasing in N; vault access increments vault_trials.
9. **Genome ops:** 10,000 seeded mutations/crossovers in bounds;
   encode→decode→hash identity; child rng reproducible from
   (SEED, gen, parent_hash).
10. **No wall-clock:** source scan rejects `now()`/`time.time` outside
    `run_*.py` I/O boundaries.
11. **PBO sanity:** pure noise → PBO ≥ 0.4; planted persistent-skill column →
    low PBO, column selected.

## Build order (each phase ends runnable)

- **Phase 0 — data repair** (touches signal_lab): dv_monitor fetch fix (data-
  anchored local slice), delete+refetch truncated CSVs. Verify: start-date
  scan; dv_monitor dry-run unchanged; `signal_lab/verify.py` passes.
- **Phase 1 — env:** config.py, datafeed.py, env.py + verify tests 3/4/6/10.
  Milestone: `python3 arena/env.py` runs a hard-coded momentum book 1995→2026,
  prints net equity/costs/invariants.
- **Phase 2 — genome+strategy:** features.py, genome.py, strategy.py + test 9.
  Milestone: `python3 arena/strategy.py` evaluates one seed genome per family,
  prints pre-vault net Sharpes.
- **Phase 3 — evaluation of a fixed population:** evaluate.py, ledger.py,
  returns artifact + tests 1/5/8/11. Milestone:
  `python3 arena/run_generation.py --init --no-evolve` on 16 seeded genomes.
- **Phase 4 — evolution:** evolution.py, HoF, checkpoint/resume + test 2.
  Milestone: 3 consecutive generations show ledger growth + improvement.
- **Phase 5 — registry+gates:** registry.py, gates.py, run_deepeval.py +
  test 7. Milestone: a candidate promoted or honestly refused with a full
  gate report; rollback demonstrated.
- **Phase 6 — cloud scheduler+reports:** reports.py, alerts_arena.py, GitHub
  repo creation (public), `.github/workflows/*.yml`, requirements.txt,
  Actions secrets. Milestone: `workflow_dispatch` manual trigger runs a full
  generation on the runner, commits state back, sends the (dry) alert;
  anti-spam verified by immediate re-trigger; cron enabled after.
- **Phase 7 — paper:** broker_paper.py, run_paper.py, tracking ledger.
  Milestone: `--dry` logs intended orders; then paper keys via the standard
  secrets chain.

## Verification (end-to-end)

- `python3 arena/verify.py` green (the 11 tests above) — the gate for
  scheduling anything.
- `python3 signal_lab/verify.py` + `python3 sell_in_may/verify.py` still green
  after Phase 0 (shared cache/config untouched behavior).
- Determinism pair-run: `run_generation.py` twice from a copied `state/` →
  identical outputs (same platform).
- Cloud parity: `workflow_dispatch` run on the runner completes, commits
  state, and its ledger rows carry the platform tag; second dispatch is a
  clean no-op/append (no duplicate trials).
- Milestone commands per phase above; `git init` + commit per phase.

## Risks & honest limitations (stated in README + every report)

- **Survivorship bias:** universe = today's S&P membership → long results
  optimistic, short results pessimistic. Disclosed, not solved; headline
  claims restricted accordingly; delisted-inclusive data (CRSP-class) is the
  named upgrade path.
- **Non-stationarity:** 1990s microstructure simulated with modern costs;
  decades-old edges may be arbitraged away. Vault + regime gates + paper stage
  are defenses, not proofs.
- **Multiple testing survives the gates:** correlated evolutionary trials →
  ledger DSR under-deflates; PBO doesn't cover designer-level choices. Treated
  as risk reduction; the only accumulating true OOS is vault → paper → live.
- **Small-account sensitivity:** $1 min commissions + whole-share rounding are
  5–10 bps each way on small positions; 2× stress gate + per-report "cost
  share of gross" line keep it visible. PDT avoided structurally.
- **Cost model is proportional (bps), not per-share** (amended in Phase 1):
  the price history is split-adjusted, so a $/share friction silently
  becomes hundreds of bps on 1990s adjusted prices of high-split names — a
  cross-sectional bias concentrated in the early screen eras. All frictions
  are therefore bps of notional (commission floored at $1). Real 1990s
  spreads were wider than modern bps; the 2× cost-stress gate (G7) is the
  standing defense, and any edge that survives only at modern costs is
  suspect by construction.
- **yfinance fragility:** cache-first; abort (with alert) rather than run >5
  days stale; certifi SSL idiom for any direct urllib. The cloud runner keeps
  its own committed cache copy so a bad yfinance day degrades to stale-cache,
  not failure.
- **Public repo (user-accepted):** code, champion genome, and paper ledger are
  world-readable. Harmless at $10–25K scale (nobody can profitably front-run
  it), and the repo can be flipped private — or live-stage ledgers moved out —
  at go-live if desired (accepting the private-minute cadence limits then).
- **Core honesty statement (verbatim in outputs):** positive sandbox alpha —
  even DSR/PBO/vault-gated — is a probabilistic claim about the past. Evidence,
  not a guarantee, and not financial advice.
