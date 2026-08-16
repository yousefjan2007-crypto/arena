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

Secrets are never hardcoded: load_credentials() reads a gitignored local file,
then borrows the existing Telegram/ntfy secrets from vrp_backtest.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

# ── locate sibling projects ────────────────────────────────────────────────────
HOME = os.path.expanduser("~")
SELL_IN_MAY = os.path.join(HOME, "sell_in_may")
SIGNAL_LAB = os.path.join(HOME, "signal_lab")
VRP_BACKTEST = os.path.join(HOME, "vrp_backtest")
ROOT = os.path.dirname(os.path.abspath(__file__))

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

# Share sell_in_may's price cache — arena reads the same CSVs, writes none of its own.
CACHE_DIR = _sm.CACHE_DIR

# The re-export above clobbered ROOT/OUTPUT_DIR with sell_in_may's paths. Restore
# arena's ROOT before deriving any arena directory from it.
ROOT = os.path.dirname(os.path.abspath(__file__))

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
DATA_DIR = os.path.join(SIGNAL_LAB, "data")

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
POP_SIZE = 64
ELITE_N = 4
IMMIGRANT_N = 4
TOURNAMENT_K = 4
SCREEN_FRAC = 0.5                   # fraction of the population surviving the F0 screen
PARSIMONY_PENALTY = 0.01            # Sharpe penalty per feature (complexity tax)
GEN_TIME_BUDGET_MIN = 180
N_JOBS = 8                          # 8 on the Mac booster; the cloud runner sets 4

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
GATE_REGIME_WINDOWS = [             # G8  the four crisis slices
    ("2000-01-01", "2002-12-31"),
    ("2008-01-01", "2009-12-31"),
    ("2020-01-01", "2020-06-30"),
    ("2022-01-01", "2022-12-31"),
]
GATE_REGIME_MAX_LOSS = -0.30        # G8  no slice may lose more than this
GATE_REGIME_SOFT_LOSS = -0.05       # G8  at least GATE_REGIME_MIN_OK slices above this
GATE_REGIME_MIN_OK = 3
GATE_BEAT_SR_MARGIN = 0.15          # G9  challenger must beat the incumbent by this
GATE_ROLLING_WIN_FRAC = 0.60        # G9  and win this share of rolling windows
GATE_ROLLING_WINDOW_YEARS = 3
GATE_RUIN_DD = 0.40                 # G10 drawdown defining ruin
GATE_RUIN_MAX_PROB = 0.05           # G10 P(ruin within RUIN_MC_YEARS) must be below this
RUIN_MC_PATHS = 200
RUIN_MC_YEARS = 2

# ── execution ──────────────────────────────────────────────────────────────────
# "sandbox" -> "paper" -> "live". Going live is a human-only flip; nothing in this
# repo ever writes "live" here.
EXECUTION_MODE = "sandbox"


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
    print("  CACHE_DIR  :", CACHE_DIR)
    print("  SEED       :", SEED, "| data from", DATA_START, "| vault from", VAULT_START)
    print("  account    : $%.0f  gross<=%.1f net<=%.1f  max %d names @ %.0f%%"
          % (START_CASH, MAX_GROSS_LEV, MAX_NET_LEV, MAX_POSITIONS, 100 * MAX_NAME_WEIGHT))

    # The superset contract: every sibling module arena reuses must find each
    # config.X it reads in THIS module. Import each one, then scan its source for
    # `config.<attr>` — importing alone would miss attributes read at call time.
    import re
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

    _c = load_credentials()
    print("  creds      : ntfy=%s telegram=%s anthropic_key=%s" % (
        bool(_c["ntfy_topic"]),
        bool(_c["telegram"].get("bot_token") if _c["telegram"] else False),
        bool(_c["anthropic_api_key"])))
