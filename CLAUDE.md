# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this workspace is

An **evolutionary strategy search** over ~30 years of daily US-equity bars. The system breeds genomes (rule families + model families, plus two adaptive genes: `train_window` bounds a model's refits to trailing labels, and `signal_bear` is an optional second book scored while the regime filter reads off), evaluates each through three fidelity levels (F0 screen, F1 walk-forward, F2 battery), and applies ten anti-luck gates. Ten gates mean a champion often refuses to form—refusals are the system working, not a defect. State lives in the repo itself; GitHub Actions is the canonical runner (cloud results are canonical; determinism is per-platform only—BLAS differs between Apple Silicon and x86). The vault (days ≥ `config.VAULT_START`, currently 2023-01-01, moved from 2020-01-01 on 2026-08-30) is gates-only and tracked with meticulous audit logging. F1 selection scores `evaluate.robust_score` (the bad quartile of rolling 3-year Sharpes); the hall of fame still ranks plain `sharpe_prevault`.

## Commands

**Test suite** (the gate for scheduling anything):
```bash
python3 verify.py                                          # 12 design checks, ~3–6 min, fully offline
python3 -c "import verify; verify.test_gates(); raise SystemExit(0 if verify.ok else 1)"  # single test
```
The twelve test functions: `test_planted_leak` (two planted leaks, act-time vs fit-time), `test_determinism` (two full generations, byte-identical artifacts), `test_accounting` (equity identity, limits, integral shares), `test_fill_timing` (decide t, fill t+1, correct spread/slippage), `test_streaming_purge` (label embargo audit, every refit), `test_cost_linearity` (stress 0/1/2 costs 0/c/2c), `test_gates` (all-pass promotes, single violation blocks, data_hash mismatch fails G1), `test_trial_ledger` (k evals = k rows, DSR monotonic), `test_genome_ops` (mutations stay in bounds, dedup hashes), `test_no_wallclock` (no clock reads outside run_*.py), `test_pbo_sanity` (noise reports high PBO; plant signal, low), `test_paper_stage` (arming truth table, position book, ledger idempotency).

**Entry points:**
```bash
python3 run_generation.py                                  # breed one generation (dry)
python3 run_generation.py --init --pop 64                  # seed a fresh population
python3 run_generation.py --refresh --send                 # cloud path: fetch live cache + breed + alert
python3 run_generation.py --no-evolve --jobs 4             # evaluate, do not breed
python3 run_deepeval.py --dry                              # gates only (ledger rows still append)
python3 run_deepeval.py --send                             # run gates + alert
python3 run_deepeval.py --rollback HASH --reason TEXT      # repoint champion.json + history row
python3 run_paper.py                                       # dry: compute, log, submit nothing
python3 run_paper.py --submit                              # submit if arming gate closed
python3 reports.py --gen N                                 # rebuild report from disk, recompute nothing
```

**Module smoke tests** (all modules have `__main__`):
- Pure/fast (seconds; `evaluate.py` ~10s): `python3 config.py`, `genome.py`, `evolution.py`, `gates.py`, `evaluate.py`.
- Cache-needing/slow: `datafeed.py`, `features.py`, `env.py`, `strategy.py`.

**Environment knobs** (only three are identity-safe, in `_CONFIG_HASH_SKIP`; the rest change config_hash):
- `ARENA_N_JOBS` — worker processes (default 8). Scales parallel evaluation.
- `GEN_TIME_BUDGET_MIN` — wall-clock budget in minutes (default 180). Checkpoints and exits 0 when spent.
- `DEEPEVAL_TIME_BUDGET_MIN` — deep-eval budget in minutes (default 360). Stops between candidates and refuses rather than half-evaluate one.
- `ARENA_FORCE_VENDOR=1` — switch to vendored siblings + `arena/data` cache on a machine that has both. Changes what data is read; data/panel hashes can move. Used to prove vendor/live agreement.

**Dependencies:** Python 3.9, system `python3`, exact pins in `requirements.txt` (closed set—no torch/xgboost/optuna). `requirements-paper.txt` adds alpaca-py; only `paper.yml` installs it.

## Architecture

**Pipeline:** `config` (imports first everywhere; wires sibling paths via `import_sibling`) → `datafeed`/`features` (panel + 3 hashes) → `genome`/`strategy`/`env` (decide at close t, fill at open t+1; equity identity asserted at every step in env.py) → `evaluate` (F0 screen / F1 full-history walk-forward / F2 weekly battery) → `evolution` (elite/mutate/crossover/immigrant per-slot child_rng streams; no niching or family cap—exact-hash dedup + 4 immigrants/gen is the entire diversity mechanism) → `ledger` (append-only trials = DSR's N; screens count) → `gates` (G1–G10 pure arithmetic, all thresholds in config) → `registry` (content-addressed immutable artifacts; champion.json pointer + champion_history.csv) → `run_*` orchestrators → `reports`/`alerts_arena`.

**Identity triple for G1:** Gate G1 gates on a three-part key: `data_hash` (equity bars—symbols, calendar ends, per-symbol bar count/last close from datafeed.py) ⊂ `panel_hash` (data + features + six macro series—the real like-for-like key; macro tickers invisible to data_hash, so two rows agree on data_hash but diverge on macro) + `config_hash`. Every row carries all three; G1 blocks a promotion unless panel_hash and config_hash match the incumbent's, regardless of score. A moved end date or a macro ticker change fails G1 even if the genome scores higher.

**The vault:** Days ≥ `config.VAULT_START` (2023-01-01 since 2026-08-30; the bump released 2020–2022, then held by 4 logged gate looks, into selection — revisit annually by hand in config.py) are gates-only; earlier days are development. F0 cannot reach the vault (every era in `config.SCREEN_ERAS` ends before VAULT_START, asserted at module import). F1 returns pre-vault and vault halves under different names; vault data leaves `evaluate.full_eval` through exactly one key (`vault_daily_net`) and nothing in that module reads it. Enforced by key-name firewall + three hard raises + a **grep-able invariant**: outside `run_deepeval.py` every `vault_` hit must be a store or pass-through. Every look appends to `state/vault_access.csv` (deliberately non-idempotent—asking twice IS two looks in the audit log; `vault_trials()` counts distinct genomes for the vault-DSR N).

**State taxonomy:** Append-only records (`trial_ledger.csv`, `vault_access.csv`, `deepeval_history.csv`, `champion_history.csv`) are the source of truth—a row, once appended, never changes or moves. Rewritable pointers (`population.json`, `hall_of_fame.json`, `champion.json`, `alert_state.json`) are the decision state—one entry at a time, rewritten per run. Regenerable caches (`state/panel_cache/`, `output/`) are ephemeral; a missing cache rebuilds on next access. `ledger.dedup_ledger` may remove only byte-identical union-merge artifacts, only in the trial ledger, because `record_trial` is idempotent on (hash, generation, fidelity)—two identical rows cannot be two real evaluations.

**Ledger and DSR:** Every genome ever evaluated at any fidelity—including screens—counts in DSR's N. `n_trials()` reads the count from `trial_ledger.csv`; no caller may pass one. `dsr_trial_sharpes()` returns one **daily-unit** Sharpe per distinct genome_hash and is the only valid G2 input—hand `deflated_sharpe` annualised numbers and the sr0 threshold comes out ~16× too wide, making G2 unpassable forever. Reports print annualised Sharpes; gates consume daily ones.

**Scheduling:** `generation.yml` 2×/day (cloud runner + Mac booster), `deepeval.yml` Sat 12:00 UTC (cloud runner only), `paper.yml` cron commented out until a champion survives `PAPER_ARM_CONSECUTIVE=3` consecutive complete deep evals; all three share `concurrency: arena-state` group to serialize writes. Runners commit state back to the repo (exit 75 = lost push race, considered a cloud failure not a code defect).

## Traps

1. **Determinism:** Never a shared/global RNG—per-slot `genome.child_rng` streams derived reproducibly (evolution.py ~35–42). No wall-clock in compute paths; verify's scanner exempts `run_*.py` and honors `# io-boundary` comments (a convention, not a check).

2. **Units (the silent G2 killer):** G2 must use `ledger.dsr_trial_sharpes()` (daily), never `f1_sharpes()` (annualised)—the mismatch makes G2 reject everything forever while looking healthy (ledger.py:308).

3. **config.py geometry:** The `_INHERITED = dict(globals())` marker at ~line 120 decides identity membership—assignments above never reach `config_hash`; new UPPERCASE scalars below it change the hash automatically (aborting in-flight generations via `IdentityDrift`—intended). Dict/set-valued settings are silently invisible to the hash; new env-driven knobs must go into `_CONFIG_HASH_SKIP` or Mac/runner split identities.

4. **Ledger:** `record_trial` returning False is not always benign—same key under a different identity triple means `IdentityDrift`. Never dedup `vault_access.csv`, `deepeval_history.csv`, or `champion_history.csv`—under-counting vault looks makes every DSR optimistic, the one forbidden direction.

5. **.gitattributes union merge on `state/*.csv` is load-bearing** for the two-writer (Mac + runner) setup. `state/tmp_gen_*/` is deliberately committed (checkpoint/resume on ephemeral runners); `state/panel_cache/` is deliberately ignored.

6. **Rule-family scores are centred** (`pct_rank − 0.5`, not raw 0..1)—do not "correct" to raw rank, or every rule genome becomes silently long-only (strategy.py:13).

7. **Registry is write-once** (ImmutableArtifact); bump `METRICS_SCHEMA` in the same commit as any eval-record shape change.

8. **vendor/ copies must be byte-identical** below the 3-line header (`python3 config.py` proves it)—**currently drifted** (sell_in_may/config.py, data.py); re-vendor before trusting a cloud run. Never edit vendored files in place.

9. **Fill prices already include spread/slippage**—do not subtract them again (env.py:55).

10. **Paper safety:** `paper=True` hardcoded, `ALPACA_PAPER_*` credential names (distinct), `--dry` beats `--submit`, arming gate reads `deepeval_history.csv`.

## Ethos

Preserve the honest-refusal posture: screens count as trials (DSR uses the real ledger N), every report prints DSR with its N, gates refuse when the evidence does not support promotion, "claim about the past, not a guarantee / not financial advice" framing stays in reports and alerts. The system trades what it proved or trades nothing.
