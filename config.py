"""
Central configuration for arena — the evolving strategy arena.

TWO RULES THIS FILE EXISTS TO ENFORCE:

  1. EVERY arena module imports `config` FIRST, before importing anything from a
     sibling project. Importing this module is what puts ~/sell_in_may and
     ~/signal_lab on sys.path; `import data` / `import macro` only resolve after.

  2. This config is a SUPERSET of the sibling configs, and it is the config the
     siblings will see. arena runs sibling modules in-process, so when
     sell_in_may/data.py or signal_lab/{features,macro,universe,cv,alerts}.py do
     `import config`, Python hands them THIS module (arena's directory is
     sys.path[0]). Every `config.X` any of those six modules reads must therefore
     be defined here. `python3 config.py` proves it by importing all six.
     The re-export of sell_in_may's config below clobbers ROOT/OUTPUT_DIR with
     that project's paths — the documented trap — so arena's ROOT is RESTORED
     immediately after, before any arena path is derived from it.

TWO HOMES, ONE ANSWER. arena runs on this Mac (siblings on disk) and on a public
GitHub runner (no siblings at all), so rule 1 has to resolve twice. `VENDORED`
picks: live sibling checkouts when ~/sell_in_may and ~/signal_lab exist, the
byte-identical copies under vendor/ otherwise. The data the siblings read moves
with them — CACHE_DIR and DATA_DIR point at the live sibling caches on the Mac
and at arena's own committed data/ on the runner. `ARENA_FORCE_VENDOR=1` forces
the vendored path on a machine that has both, which is the only way to prove the
two agree (features.py's smoke asserts panel_hash equality across the pair).

Secrets are never hardcoded: load_credentials() reads a gitignored local file,
then borrows the existing Telegram/ntfy secrets from vrp_backtest.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import sys

# ── locate sibling projects ────────────────────────────────────────────────────
HOME = os.path.expanduser("~")
VRP_BACKTEST = os.path.join(HOME, "vrp_backtest")
ROOT = os.path.dirname(os.path.abspath(__file__))
VENDOR_DIR = os.path.join(ROOT, "vendor")

# The runner has no siblings, so it uses the copies under vendor/. Forcing the
# vendored path on a machine that HAS siblings is how the two are proven equal:
# same panel_hash, same config_hash, same everything a result is recorded under.
FORCE_VENDOR = os.environ.get("ARENA_FORCE_VENDOR", "") not in ("", "0")   # io-boundary
_LIVE_SIBLINGS = all(os.path.isdir(os.path.join(HOME, p))
                     for p in ("sell_in_may", "signal_lab"))
VENDORED = FORCE_VENDOR or not _LIVE_SIBLINGS
_HOME_OF_SIBLINGS = VENDOR_DIR if VENDORED else HOME
SELL_IN_MAY = os.path.join(_HOME_OF_SIBLINGS, "sell_in_may")
SIGNAL_LAB = os.path.join(_HOME_OF_SIBLINGS, "signal_lab")

# Appended (not prepended) so arena's own modules win every name clash, while the
# siblings' internal `import config` still lands on this superset.
for _p in (SELL_IN_MAY, SIGNAL_LAB):
    if _p not in sys.path:
        sys.path.append(_p)


def import_sibling(module: str, project: str):
    """Import a sibling project's module from an explicit path, under its own key.

    arena shares module names with signal_lab (features, ledger, registry,
    evaluate, verify...), and arena's directory wins on sys.path, so a plain
    `import features` inside arena would import arena's own file. This loads the
    sibling's file directly so both can live in one process. Modules loaded this
    way still resolve their own `import config` to THIS module.
    """
    key = "_%s_%s" % (os.path.basename(project), module)
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, os.path.join(project, module + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod          # register before exec so circular imports work
    spec.loader.exec_module(mod)
    return mod


# ── re-export sell_in_may/config.py so the reused modules find every attribute ──
_sm = import_sibling("config", SELL_IN_MAY)
for _k, _v in vars(_sm).items():
    if not _k.startswith("__"):
        globals()[_k] = _v

# The re-export above clobbered ROOT/OUTPUT_DIR with sell_in_may's paths. Restore
# arena's ROOT before deriving any arena directory from it — including CACHE_DIR
# on the next line, which is arena's own directory in vendored mode and would
# otherwise land inside vendor/sell_in_may/.
ROOT = os.path.dirname(os.path.abspath(__file__))

# On the Mac: share sell_in_may's price cache — arena reads the same CSVs and
# writes none of its own. On the runner there is no such cache, so arena keeps its
# OWN committed copy (docs/DESIGN.md, yfinance-fragility risk: "the cloud runner
# keeps its own committed cache copy so a bad yfinance day degrades to stale-cache,
# not failure"), and the workflow commits back whatever a refresh changed.
#
# Assigned ABOVE the _INHERITED marker on purpose: every name up here is snapshotted
# at its final value, so none of them can reach config_hash(). A machine's cache
# location is not part of a result's identity, and letting it in would give the Mac
# and the runner different config hashes for identical settings — exactly the
# like-for-like failure gate G1 exists to catch.
CACHE_DIR = os.path.join(ROOT, "data", "cache") if VENDORED else _sm.CACHE_DIR

# Everything the re-export brought in, NAME -> VALUE. config_hash() (bottom of this
# file) hashes what arena itself declares BELOW this line, not a sibling project's
# option-pricing knobs — sell_in_may bumps TARGET_YEAR every January, and an arena
# result must not stop being like-for-like because of it.
#
# The values, not just the names: a bare name set silently drops every knob arena
# REDECLARES (the name is in both), and three are — OUTPUT_DIR, SEED and
# TRADING_DAYS_YEAR. Losing TRADING_DAYS_YEAR from the digest is the dangerous one:
# it scales every Sharpe in the ledger and the daily/annual conversion in
# ledger.trial_sr_std, so changing it would change every score while the config
# hash stayed put — precisely the like-for-like failure gate G1 exists to catch.
_INHERITED = dict(globals())

# ── superset: attributes the reused sibling modules read as config.X ───────────
# signal_lab/features.py
MOMENTUM_LOOKBACKS = [5, 10, 21, 63, 126, 252]
RSI_PERIOD = 14
MIN_HISTORY_DAYS = 400              # symbol must have this much data to enter the panel
MIN_DOLLAR_VOL = 5_000_000.0        # liquidity gate (avg daily $ volume)

# signal_lab/universe.py
ETFS = [BENCHMARK] + SECTOR_ETFS    # noqa: F821 (from the sell_in_may re-export)
CRYPTO = []                         # arena is stocks-only: no weekend calendar to fold in
# Off on purpose. A live fetch inside an evaluation path is non-reproducible, and
# universe.py writes what it fetches to DATA_DIR/sp500.csv — which points at
# signal_lab's data dir, so an arena run would overwrite a sibling's cache. The
# static fallback below is the deterministic choice; point-in-time membership is
# the named upgrade path (see DESIGN risks: survivorship bias).
USE_LIVE_SP500_LIST = False
LARGECAP_FALLBACK = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "JPM", "V",
    "UNH", "XOM", "MA", "JNJ", "HD", "PG", "COST", "ABBV", "MRK", "CVX",
    "ADBE", "CRM", "AMD", "NFLX", "PEP", "KO", "WMT", "BAC", "TMO", "ACN",
    "MCD", "CSCO", "ABT", "LIN", "DHR", "INTC", "QCOM", "TXN", "WFC", "PM",
    "CAT", "INTU", "VZ", "AMGN", "IBM", "GE", "NOW", "UBER", "SPGI", "GS",
    "ISRG", "PFE", "HON", "BKNG", "AMAT", "NEE", "RTX", "LOW", "UNP", "BLK",
    "ELV", "C", "PLD", "SYK", "BSX", "MU", "ADP", "DE", "LRCX", "MDT",
    "CB", "GILD", "TJX", "VRTX", "REGN", "PGR", "ETN", "SCHW", "BMY", "MMC",
    "KLAC", "PANW", "CI", "SO", "MO", "DUK", "ZTS", "BX", "SNPS", "CDNS",
    "ITW", "SHW", "APH", "ICE", "PYPL", "MCK", "PH", "CME", "USB", "AON",
]
# universe.py caches the S&P 500 membership table here. Pointed at signal_lab's
# data dir on purpose: the table is already cached there, so arena resolves the
# universe offline instead of re-fetching Wikipedia. arena's own writes all go to
# STATE_DIR / ARTIFACT_DIR / OUTPUT_DIR below.
#
# Vendored (the runner): arena's own data/, holding a byte copy of that same
# sp500.csv. It is not a nicety — the table supplies each equity's sector anchor,
# the anchor decides the rs_sector feature, and a missing table would build a
# DIFFERENT panel while every other input matched. Committed, therefore, and in
# the config_hash's skip list because it is a path, not a setting.
DATA_DIR = os.path.join(ROOT, "data") if VENDORED else os.path.join(SIGNAL_LAB, "data")

# signal_lab/cv.py (its four splitter constants)
CV_N_SPLITS = 6
CV_TEST_DAYS = 63
CV_EMBARGO_FRAC = 0.01
CV_MIN_TRAIN_DAYS = 504

# ── arena paths ────────────────────────────────────────────────────────────────
STATE_DIR = os.path.join(ROOT, "state")          # population, ledger, champion pointer
ARTIFACT_DIR = os.path.join(ROOT, "artifacts")   # immutable per-genome artifacts
OUTPUT_DIR = os.path.join(ROOT, "output")        # reports + charts
for _d in (STATE_DIR, ARTIFACT_DIR, OUTPUT_DIR):
    os.makedirs(_d, exist_ok=True)

# ── data window & universe ─────────────────────────────────────────────────────
SEED = _sm.SEED                     # 12345 — the single reproducibility seed
# Pinned here rather than inherited: borrow and margin are priced per trading day
# (annual rate / this), and that must not silently follow a sibling project's edit.
TRADING_DAYS_YEAR = 252
DATA_START = "1995-01-01"           # start of the replayed sandbox history
UNIVERSE_SIZE = 120                 # symbols carried into the full-fidelity evaluation
VAULT_START = "2020-01-01"          # days >= this are the vault: gates only, never fitness

# ── account & constraints (small-account realism) ──────────────────────────────
START_CASH = 15_000.0               # mid of the user's $10-25K, conservative
MAX_GROSS_LEV = 1.5                 # sum |weight|
MAX_NET_LEV = 1.0                   # |sum weight|
MAX_POSITIONS = 20
MAX_NAME_WEIGHT = 0.20
MIN_POSITION_USD = 500.0            # orders smaller than this are dropped (exits exempt)
WHOLE_SHARES = True
NO_INTRADAY_EXITS = True            # PDT-safe: decide at close t, fill at open t+1
# Hysteresis band on the leverage caps: the risk repair only fires past
# cap*(1+LEV_EPS), so a strategy targeting exactly the cap does not churn a share
# back and forth every day on rounding and cost drift. Repairs still trim all the
# way back to the strict cap.
LEV_EPS = 0.01

# ── frictions (charged on every simulated fill) ────────────────────────────────
# All proportional (bps of notional). Nothing is quoted per share: the cache
# stores split-adjusted prices, so a $/share figure means something different in
# 1995 than today, while a bps figure is scale-invariant.
COMMISSION_BPS = 0.5
COMMISSION_MIN = 1.00               # dominates below a ~$20k order — small-account reality
HALF_SPREAD_BPS = 2.5
SLIPPAGE_BPS = 2.0
BORROW_ANNUAL = 0.01                # on short market value
MARGIN_ANNUAL = 0.065               # on negative cash

# ── env numerics ───────────────────────────────────────────────────────────────
ENV_CHECK_INVARIANTS = True         # assert the accounting identity every step
ACCOUNT_TOL = 1e-6                  # equity identity tolerance (dollars)
SHARE_ROUND_EPS = 1e-9              # see env._target_shares: float round-trip guard
ENV_MAX_TRIM_ITERS = 64             # bound on the whole-share leverage repair loop

# ── walk-forward / cross-validation ────────────────────────────────────────────
WF_MIN_TRAIN_DAYS = 1008            # ~4 years before the first live decision
WF_EMBARGO_DAYS = 21
CPCV_GROUPS = 8                     # 8C2 = 28 combinatorial purged paths
CPCV_K = 2
PBO_SPLITS = 16                     # CSCV S
BOOT_ITERS = 5000                   # block bootstrap resamples
BOOT_BLOCK = 21                     # block length (~1 month) preserves autocorrelation

# ── evolution ──────────────────────────────────────────────────────────────────
# POP_SIZE is the size a population is SEEDED at (run_generation --init, whose
# --pop overrides it). After that the population file carries its own size and
# evolution.next_generation keeps it: a 12-genome arena stays 12 genomes, because
# silently inflating a small run to 64 would multiply its cost by five between one
# generation and the next.
POP_SIZE = 64
ELITE_N = 4
IMMIGRANT_N = 4
TOURNAMENT_K = 4
SCREEN_FRAC = 0.5                   # fraction of the population surviving the F0 screen
PARSIMONY_PENALTY = 0.01            # Sharpe penalty per feature (complexity tax)
DEDUP_MAX_TRIES = 10                # re-mutations before a colliding child is replaced
                                    # by a fresh immigrant (evolution.next_generation)
HOF_SIZE = 10                       # hall-of-fame depth (top-N all-time by pre-vault Sharpe)
# ELITE_N + IMMIGRANT_N are absolute counts sized for POP_SIZE=64 (8 slots of 64).
# On a small run they would reserve the whole population — 4+4 of an 8-genome test
# population breeds nothing at all — so evolution.slot_counts caps the two carried
# groups at this fraction and scales them down proportionally. A no-op at POP_SIZE.
EVOLVE_MAX_CARRY_FRAC = 0.5
# The three knobs below are HOW FAST, NOT WHAT — they are in _CONFIG_HASH_SKIP, so
# none of them can move a result's identity, which is exactly why they are the
# three the environment is allowed to override. The runner has 4 vCPUs and a 6-hour
# job ceiling; this Mac has 14 cores and no ceiling. Same settings, same hashes,
# different machines. (io-boundary: environment read, no compute path involved.)
GEN_TIME_BUDGET_MIN = float(os.environ.get("GEN_TIME_BUDGET_MIN") or 180)
N_JOBS = int(os.environ.get("ARENA_N_JOBS") or 8)

# ── the evaluation ladder (evaluate.py) ────────────────────────────────────────
# F0 is a SCREEN, not a measurement: three disjoint five-year eras, a smaller
# point-in-time universe, and coarse cadences, chosen so a 64-genome population
# can be ranked in minutes rather than a day. Every era ends before VAULT_START —
# evaluate.py asserts it rather than trusting this comment.
SCREEN_ERAS = [("1997-01-01", "2001-12-31"),      # dot-com run-up and bust
               ("2007-01-01", "2011-12-31"),      # GFC and the recovery
               ("2015-01-01", "2019-12-31")]      # low-vol grind + 2018 shakeout
SCREEN_UNIVERSE_N = 60              # symbols per era, ranked point-in-time (see below)
SCREEN_REFIT_DAYS = 252             # F0 forces yearly refits whatever the genome asks
SCREEN_MIN_REBALANCE_DAYS = 5       # ...and at most weekly rebalancing
SCREEN_LIQUIDITY_DAYS = 252         # trailing window for the era's dollar-volume rank
SCREEN_LIQUIDITY_MIN_BARS = 21      # a symbol needs this many bars in it to be ranked
# Below this many scored days a Sharpe is noise, and evaluate.sharpe returns 0.0
# ("no evidence") rather than a number selection could act on.
SHARPE_MIN_OBS = 60
# Empirical trial-Sharpe dispersion needs a sample; under this many F1 rows the
# ledger falls back to 1/sqrt(n_trials) (ledger.trial_sr_std explains why).
TRIAL_SR_STD_MIN_ROWS = 8
# run_generation refuses to evaluate on a cache older than this (calendar days).
MAX_DATA_STALENESS_DAYS = 5
# Tolerance gate on cache writes (run_generation._tolerance_gate). yfinance
# recomputes its whole auto-adjusted history in reduced precision on every fetch,
# so a refetch with no new bar still restates every row by ~3e-7 relative. Below
# this bound the CACHED bytes are kept; above it, a real adjustment event has
# happened and the restatement is accepted whole. 1e-5 sits two orders above the
# jitter and orders below any real split or dividend adjustment.
REFRESH_REL_TOL = 1e-5

# Genome operators (docs/DESIGN.md "Operators"). These are INDEPENDENT per-move
# probabilities, not a distribution — they sum past 1 on purpose, so one child can
# carry two moves. genome.mutate() forces a single weighted move when none fire.
P_MUT_PARAM = 0.60                  # jitter one param to a grid neighbour
P_MUT_FEATURE = 0.30                # add / drop / swap one feature
P_MUT_BLOCK = 0.15                  # resample a whole gene block
P_MUT_FAMILY = 0.05                 # hop to another signal family
CROSSOVER_FRAC = 0.25               # share of offspring bred by crossover
MUT_MAX_TRIES = 4                   # bound on re-drawing a mutation that changed nothing
                                    # (a genome sitting on the n_long+n_short floor can
                                    # have single steps undone by the repair)

# ── strategy mechanics (strategy.py) ───────────────────────────────────────────
# The genome owns WHICH overlay is on and how hard; these are the fixed mechanics
# every genome shares, so they belong here rather than in the search space.
REALIZED_VOL_DAYS = 21              # window for inv_vol weighting and vol targeting
VOL_TARGET_MIN = 0.25               # vol targeting may not shrink gross below this...
VOL_TARGET_MAX = 2.0                # ...nor inflate it beyond this
TREND_MA_DAYS = 200                 # the 200DMA behind spy_200dma and the trend gate
VIX_PCT_LIMIT = 0.80                # panel vix_pct above this is the fear regime
DD_DERISK_SCALE = 0.5               # a dd_limit breach halves gross...
DD_RECOVER_FRAC = 0.5               # ...until drawdown recovers inside this fraction of it
LOGIT_MAX_ITER = 200                # lbfgs iterations for the logistic family

# ── promotion gates (docs/DESIGN.md table G1-G10) ──────────────────────────────
GATE_MIN_DSR = 0.95                 # G2  pre-vault deflated Sharpe
GATE_VAULT_MIN_DSR = 0.90           # G3  vault Sharpe > 0 and vault DSR >= this
GATE_MAX_PBO = 0.20                 # G4
GATE_CPCV_MIN_POS_FRAC = 0.70       # G5  fraction of the 28 paths that must be net-positive
GATE_CPCV_MIN_MEDIAN_SR = 0.30      # G5
GATE_BOOT_CI = 0.95                 # G6  bootstrap CI level; lower bound must exceed 0
GATE_STRESS_MULT = 2.0              # G7  cost stress multiplier
GATE_STRESS_MIN_SR_RATIO = 0.5      # G7  stressed Sharpe >= this x base Sharpe
GATE_BORROW_STRESS_MULT = 3.0       # G7  borrow-only stress
# G8  the four crisis slices, dated NARROWLY — peak to trough, not peak to
# recovery. A window that contains the rebound measures the wrong thing: the 2008
# calendar years 2008-01..2009-12 come out POSITIVE for a long-biased book that
# was destroyed in the autumn of 2008 and carried up by 2009, and a gate cannot
# ask "how bad was the crisis" with a number like that. These are the drawdown
# legs: the dot-com slide to its October-2002 low, the Lehman quarter to the March
# 2009 bottom, the COVID crash and its first-half round trip, and the 2022 repricing.
GATE_REGIME_WINDOWS = [
    ("2000-03-01", "2002-10-31"),
    ("2008-09-01", "2009-03-31"),
    ("2020-01-01", "2020-06-30"),
    ("2022-01-01", "2022-12-31"),
]
GATE_REGIME_MAX_LOSS = -0.30        # G8  no slice may lose more than this
GATE_REGIME_SOFT_LOSS = -0.05       # G8  at least GATE_REGIME_MIN_OK slices above this
GATE_REGIME_MIN_OK = 3
# G8  and at least this many of the windows must actually be COVERED by the
# scored span. An uncovered slice passes by absence — failing a genome for a
# crisis it never traded would be inventing evidence — but absence may not be
# what earns the quorum above. Without this floor a candidate whose history
# begins in 2015 waives the dot-com and 2008 windows for free and clears G8 on
# COVID and 2022 alone, which is the gate agreeing it survived crises it never
# saw. Three of four is the same bar GATE_REGIME_MIN_OK sets: one window may be
# missing, not most of them.
GATE_REGIME_MIN_COVERED = 3
GATE_BEAT_SR_MARGIN = 0.15          # G9  challenger must beat the incumbent by this
GATE_ROLLING_WIN_FRAC = 0.60        # G9  and win this share of rolling windows
GATE_ROLLING_WINDOW_YEARS = 3
GATE_RUIN_DD = 0.40                 # G10 drawdown defining ruin
GATE_RUIN_MAX_PROB = 0.05           # G10 P(ruin within RUIN_MC_YEARS) must be below this
RUIN_MC_PATHS = 200                 # per engine (GARCH-t and block bootstrap)
RUIN_MC_YEARS = 2

# ── weekly deep evaluation (run_deepeval.py) ───────────────────────────────────
# How many hall-of-fame leaders get the F2 battery each week. It is small because
# F2 is expensive (CPCV alone is 2 x C(8,2) = 56 episodes with real refits per
# candidate) and because every candidate shown the vault deflates the vault DSR of
# every candidate after it — the honesty tax is paid in the gate, so the number of
# looks belongs in the config digest rather than on a command line.
DEEPEVAL_CANDIDATES = 2
# The Actions deepeval.yml window (6 h); env-overridable for the same reason as
# GEN_TIME_BUDGET_MIN above, and skip-listed for the same reason.
DEEPEVAL_TIME_BUDGET_MIN = float(os.environ.get("DEEPEVAL_TIME_BUDGET_MIN") or 360)

# ── repo growth policy (docs/DESIGN.md "Repo growth policy") ───────────────────
# The permanent record is the trial ledger, the hall of fame and the champion
# artifacts. The per-generation returns matrices are ~400 kB each and exist to
# feed the cohort PBO of a RECENT generation, so the workflow prunes the ones
# older than this. Skip-listed: deleting a file that nothing reads changes no
# number, and a repo that grew without bound would eventually make every cloud
# checkout slower than the run it enables.
RETURNS_KEEP_GENERATIONS = 90

# ── execution ──────────────────────────────────────────────────────────────────
# "sandbox" -> "paper" -> "live". Going live is a human-only flip; nothing in this
# repo ever writes "live" here.
EXECUTION_MODE = "sandbox"

# ── the paper stage (run_paper.py; docs/DESIGN.md "Graduation ladder") ─────────
# PAPER_ARM_CONSECUTIVE is the arming gate: run_paper.py may submit an order only
# when the SAME champion has come through this many consecutive COMPLETE deep
# evals still holding the pointer, and deepeval_history records it passing all ten
# gates. Below it, the run computes and logs its intended orders and submits
# nothing. The other four are the go-live EVIDENCE thresholds DESIGN names — they
# decide what a paper session reports, never what it does, because "the system
# never self-starts live": a human reads the table and flips EXECUTION_MODE.
# The count INCLUDES the deep eval that promoted it (that entry names it
# champion_hash_after too), so arming happens two deep evals after promotion;
# run_paper.arming_status says why, and what the stricter reading would cost.
PAPER_ARM_CONSECUTIVE = 3           # consecutive complete deep evals the champion must survive
PAPER_MIN_DAYS = 126                # trading days of paper before go-live is even discussable
PAPER_MAX_TE_BPS = 25.0             # daily |paper - sim| that trips the tracking alert
PAPER_MIN_CORR = 0.80               # corr(daily paper, sim-shadow) required
PAPER_MAX_SLIPPAGE_BPS = 10.0       # median |fill slippage| required


# ── configuration identity (gate G1 / the trial ledger) ────────────────────────
# Two results are comparable only if they were produced under the same rules. The
# DATA identity is datafeed's data_hash and features' panel_hash; this is the other
# half — the SETTINGS identity, recorded on every ledger row.
#
# COVERS every UPPERCASE name whose value is a scalar or a nested list/tuple of
# scalars and that is ARENA'S, by any of three tests:
#   1. this file assigns it below the _INHERITED marker (_DECLARED_HERE, a source
#      scan) — the account, every friction, the leverage and position caps, the
#      walk-forward and CV constants, the whole evolution and evaluation ladder,
#      the gate thresholds, the universe lists, EXECUTION_MODE;
#   2. it is not in the inherited snapshot at all;
#   3. its value differs from the inherited one (a knob arena changed at runtime).
# Plus _CONFIG_HASH_EXTRA, for inherited names that bind the simulation anyway.
#
# Test 1 is not redundant with test 3, and assuming it was is how TRADING_DAYS_YEAR
# fell out of this digest once already: arena redeclares `TRADING_DAYS_YEAR = 252`
# and `SEED = 12345`, which are exactly the values sell_in_may already held. A
# value diff sees no change and drops them — yet they are arena's knobs, arena is
# free to change either tomorrow, and TRADING_DAYS_YEAR scales every Sharpe on
# record. The scanner behind test 1 is deliberately dumb (one regex, module level,
# same idiom as verify.py's wall-clock scan) so that it cannot be clever and wrong.
#
# DOES NOT COVER: filesystem paths and machine identity (a result is not different
# because it ran in a different directory), the parallelism/time budget knobs (how
# fast, not what), the invariant-assertion toggle (it can only raise, never change
# a number), sell_in_may's own knobs, and anything not JSON-scalar-shaped. It also
# cannot see a CODE change — that is git's job, not this hash's.
#
# Nor does it cover a knob whose whole effect is ALREADY RECORDED IN data_hash.
# This digest exists to catch settings that change results INVISIBLY; a setting
# that can only change which bars land in the cache is visible by construction,
# because data_hash is computed from the bars actually used and is stamped on
# every ledger row, artifact and history row. REFRESH_REL_TOL is the one such
# knob (RETURNS_KEEP_GENERATIONS is the weaker case: it deletes files nothing
# reads). The alternative — digesting it — would invalidate like-for-like against
# every row on record to describe a difference those rows already carry.
#
# Nor the PAPER_* knobs, and that one is worth stating in full because
# EXECUTION_MODE — an execution setting — IS in the digest. Two reasons they are
# not. (1) Nothing in the simulation reads them: `grep PAPER_` finds run_paper.py
# and verify.py and nothing in datafeed/env/strategy/evaluate/gates/ledger, so
# they cannot move a single number a ledger row records. (2) Digesting them would
# have BROKEN THE RUNNING SYSTEM the moment they were added: config_hash is
# stamped on every ledger row and returns matrix, and run_generation raises
# IdentityDrift — by design, fatally — when a generation already partly on disk is
# resumed under a different identity. A knob that changes no result must not be
# able to stop the nightly job. EXECUTION_MODE stays in because it has been in
# every hash on record since Phase 1; removing it now would cost exactly what
# adding these would.
_CONFIG_HASH_SKIP = frozenset({
    "STATE_DIR", "ARTIFACT_DIR", "OUTPUT_DIR", "DATA_DIR",   # where files live
    "N_JOBS", "GEN_TIME_BUDGET_MIN", "DEEPEVAL_TIME_BUDGET_MIN",   # how fast, not what
    "RETURNS_KEEP_GENERATIONS",                     # what is kept, not what was computed
    "REFRESH_REL_TOL",                              # already visible in data_hash (above)
    "ENV_CHECK_INVARIANTS",                         # asserts, never arithmetic
    "PAPER_ARM_CONSECUTIVE", "PAPER_MIN_DAYS", "PAPER_MAX_TE_BPS",   # the paper stage:
    "PAPER_MIN_CORR", "PAPER_MAX_SLIPPAGE_BPS",                      # see above
})
# Inherited, never redeclared here, but the simulation reads them. (SEED is also
# caught by _DECLARED_HERE now; it stays listed because losing it would be silent.)
_CONFIG_HASH_EXTRA = ("SEED", "BENCHMARK")


def _declared_here() -> frozenset:
    """UPPERCASE names this FILE assigns at module level below the _INHERITED
    marker — i.e. arena's own knobs, whatever their values happen to equal.

    Reads its own source rather than reasoning about values, because a
    redeclaration that repeats the inherited value is invisible to every
    value-based test (see the comment above). Module level only: the `^` anchor
    excludes the indented assignments inside the __main__ block, and leading
    underscores are filtered by config_hash_items.
    """
    with open(os.path.abspath(__file__)) as f:
        lines = f.read().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("_INHERITED"))
    assign = re.compile(r"^([A-Z][A-Z0-9_]*)\s*(?::[^=]+)?=[^=]")
    return frozenset(m.group(1) for m in
                     (assign.match(line) for line in lines[start + 1:]) if m)


_DECLARED_HERE = _declared_here()


def _canon(value):
    """Canonical text for a config value, or None if it is not hash-shaped."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return repr(value)                  # repr round-trips floats exactly in py3
    if isinstance(value, (list, tuple)):
        parts = [_canon(v) for v in value]
        return None if any(p is None for p in parts) else "[%s]" % ",".join(parts)
    return None


def config_hash_items() -> list:
    """The exact (name, canonical value) pairs config_hash() digests, sorted."""
    out = []
    for name in sorted(set(globals()) | set(_CONFIG_HASH_EXTRA)):
        if name.startswith("_") or not name.isupper() or name in _CONFIG_HASH_SKIP:
            continue
        value = globals().get(name)
        canon = _canon(value)
        if canon is None:
            continue                        # not JSON-scalar-shaped: nothing to hash
        arenas = (name in _DECLARED_HERE                       # 1. declared here
                  or name not in _INHERITED                    # 2. not inherited
                  or _canon(_INHERITED[name]) != canon         # 3. changed since
                  or name in _CONFIG_HASH_EXTRA)
        if arenas:
            out.append((name, canon))
    return out


def config_hash() -> str:
    """sha256 over the sorted (name, value) list above, first 16 hex chars."""
    h = hashlib.sha256()
    for name, canon in config_hash_items():
        h.update(("%s=%s\n" % (name, canon)).encode())
    return h.hexdigest()[:16]


# ── credential loading (no secrets in source) ──────────────────────────────────
def load_credentials() -> dict:
    """Load notifier + API secrets without hardcoding them.

    Priority: arena/config.local.json -> vrp_backtest/monitor_config.json (reuse
    the existing Telegram/ntfy secrets) -> environment. Same chain as signal_lab,
    solana_screener and robinhood_screener. Returns ntfy_topic, telegram
    {bot_token, chat_id}, anthropic_api_key.
    """
    creds = {"ntfy_topic": None, "telegram": {}, "anthropic_api_key": None}

    local = os.path.join(ROOT, "config.local.json")
    if os.path.exists(local):
        try:
            with open(local) as f:
                creds.update(json.load(f))
        except Exception:
            pass

    mon = os.path.join(VRP_BACKTEST, "monitor_config.json")
    if os.path.exists(mon):
        try:
            with open(mon) as f:
                m = json.load(f)
            if not creds.get("ntfy_topic"):
                creds["ntfy_topic"] = m.get("ntfy_topic")
            if not creds.get("telegram"):
                creds["telegram"] = m.get("telegram", {})
        except Exception:
            pass

    if not creds.get("anthropic_api_key"):
        creds["anthropic_api_key"] = os.environ.get("ANTHROPIC_API_KEY")
    return creds


if __name__ == "__main__":
    print("arena config")
    print("  ROOT       :", ROOT)
    print("  siblings   : %s (%s)%s"
          % ("VENDORED" if VENDORED else "live", os.path.relpath(SELL_IN_MAY, ROOT)
             if VENDORED else SELL_IN_MAY,
             "  [ARENA_FORCE_VENDOR]" if FORCE_VENDOR else
             ("" if _LIVE_SIBLINGS else "  [no sibling checkouts on this machine]")))
    print("  CACHE_DIR  :", CACHE_DIR)
    print("  DATA_DIR   :", DATA_DIR)
    print("  SEED       :", SEED, "| data from", DATA_START, "| vault from", VAULT_START)
    print("  account    : $%.0f  gross<=%.1f net<=%.1f  max %d names @ %.0f%%"
          % (START_CASH, MAX_GROSS_LEV, MAX_NET_LEV, MAX_POSITIONS, 100 * MAX_NAME_WEIGHT))

    # The superset contract: every sibling module arena reuses must find each
    # config.X it reads in THIS module. Import each one, then scan its source for
    # `config.<attr>` — importing alone would miss attributes read at call time.
    print("  superset check (siblings resolving `import config` to arena's):")
    for _mod, _proj in [("data", SELL_IN_MAY), ("features", SIGNAL_LAB), ("macro", SIGNAL_LAB),
                        ("universe", SIGNAL_LAB), ("cv", SIGNAL_LAB), ("alerts", SIGNAL_LAB)]:
        _path = os.path.join(_proj, _mod + ".py")
        import_sibling(_mod, _proj)
        # run as a script this file is "__main__", so the siblings' `import config`
        # loads a second instance of it — assert it is at least the same FILE.
        assert os.path.abspath(sys.modules["config"].__file__) == os.path.abspath(__file__), \
            "arena config was shadowed by another config.py on sys.path"
        with open(_path) as _f:
            _attrs = sorted(set(re.findall(r"\bconfig\.([A-Za-z_]\w*)", _f.read())))
        _missing = [a for a in _attrs if a not in globals()]
        assert not _missing, "%s reads config.%s which arena/config.py does not define" % (
            _mod, ", config.".join(_missing))
        print("    [ok] %-9s <- %-11s %2d config attrs resolved"
              % (_mod, os.path.basename(_proj), len(_attrs)))

    # THE VENDORING CONTRACT, checked rather than asserted in a comment: every file
    # under vendor/ must be byte-identical to its source below the three-line
    # header. This is the real proof that the runner runs the same code as this
    # Mac — features.py's panel comparison is the end-to-end check on top of it,
    # and that one can only run while both caches still hold the same vintage.
    _VENDOR_HEADER_LINES = 3
    if os.path.isdir(VENDOR_DIR):
        _drift, _n = [], 0
        for _proj in sorted(os.listdir(VENDOR_DIR)):
            for _name in sorted(os.listdir(os.path.join(VENDOR_DIR, _proj))):
                if not _name.endswith(".py"):
                    continue
                _live = os.path.join(HOME, _proj, _name)
                if not os.path.exists(_live):
                    continue                    # no sibling checkout: nothing to compare
                _n += 1
                with open(os.path.join(VENDOR_DIR, _proj, _name), "rb") as _f:
                    _body = _f.read().split(b"\n", _VENDOR_HEADER_LINES)[-1]
                with open(_live, "rb") as _f:
                    if _f.read() != _body:
                        _drift.append("%s/%s" % (_proj, _name))
        print("  vendored   : %d file(s) byte-identical to their source%s"
              % (_n, "" if not _drift else
                 " EXCEPT %s — re-vendor before trusting a cloud run" % ", ".join(_drift)))

    _items = config_hash_items()
    print("  config_hash: %s  (%d settings: %s ...)"
          % (config_hash(), len(_items), ", ".join(n for n, _ in _items[:5])))
    # The knobs arena redeclares on top of the re-export are the ones a name-only
    # or value-only test drops. Print them, with whether the digest sees them.
    _shadowed = sorted(n for n in _DECLARED_HERE
                       if n in _INHERITED and n not in _CONFIG_HASH_SKIP)
    _covered = dict(_items)
    print("  shadowed   : %s" % ", ".join(
        "%s=%s%s" % (n, _covered.get(n, "?"), "" if n in _covered else " MISSING")
        for n in _shadowed))

    _c = load_credentials()
    print("  creds      : ntfy=%s telegram=%s anthropic_key=%s" % (
        bool(_c["ntfy_topic"]),
        bool(_c["telegram"].get("bot_token") if _c["telegram"] else False),
        bool(_c["anthropic_api_key"])))
