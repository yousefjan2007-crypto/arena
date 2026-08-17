# VENDORED from ~/sell_in_may/config.py @ sha256:e8226fcfba04 on 2026-08-17. (source is not a git repo; content digest instead)
# Byte-identical to that source below this header. The public cloud runner
# has no sibling checkouts. DO NOT EDIT — re-vendor from the source instead.
"""
Central configuration for the "Sell in May" seasonal short engine.

Everything tunable lives here so the rest of the code reads cleanly. Constants
are grouped by what they control. Read top-to-bottom for a quick tour of the
assumptions baked into the engine.
"""
from __future__ import annotations
import os

# ── universe ──────────────────────────────────────────────────────────────────
# SPY + the 11 S&P 500 sector SPDRs. The "Sell in May" / Halloween effect is an
# index/sector-level phenomenon, which is why we screen these rather than single
# stocks. Inception dates matter: the newer ETFs simply have shorter samples.
BENCHMARK = "SPY"
SECTOR_ETFS = [
    "XLK",   # Technology
    "XLF",   # Financials
    "XLE",   # Energy
    "XLV",   # Health Care
    "XLI",   # Industrials
    "XLY",   # Consumer Discretionary
    "XLP",   # Consumer Staples
    "XLU",   # Utilities
    "XLB",   # Materials
    "XLRE",  # Real Estate (since 2015)
    "XLC",   # Communication Services (since 2018)
]
UNIVERSE = [BENCHMARK] + SECTOR_ETFS

# Deep-history proxies for the seasonality test (free, long samples).
SP500_INDEX = "^GSPC"        # S&P 500 price index, daily since 1950
VIX = "^VIX"                 # implied-vol regime
RISK_FREE_TICKER = "^IRX"    # 13-week T-bill discount rate (annual %, /100 to use)
FALLBACK_RISK_FREE = 0.043   # used if ^IRX fetch fails

# ── seasonality definition ──────────────────────────────────────────────────
# Classic Halloween convention: "summer" = May..Oct (weak), "winter" = Nov..Apr.
SUMMER_MONTHS = {5, 6, 7, 8, 9, 10}
WINTER_MONTHS = {11, 12, 1, 2, 3, 4}
TARGET_MONTH = 6             # June 2026 is the month we are trading
TARGET_YEAR = 2026

# ── statistics ────────────────────────────────────────────────────────────────
NEWEY_WEST_LAGS = 21         # ~1 trading month of HAC lags
BOOTSTRAP_ITERS = 5000       # block-bootstrap resamples for CIs / reality check
BLOCK_LEN = 21               # block length (~1 month) to preserve autocorrelation
ALPHA = 0.05                 # significance level
OOS_SPLIT_DATE = "2010-01-01"  # walk-forward train/test boundary
SEED = 12345                 # reproducibility (NO Date.now/random anywhere)

# ── Monte Carlo ─────────────────────────────────────────────────────────────
N_PATHS = 100_000            # paths per engine
TRADING_DAYS_YEAR = 252
MC_ENGINES = ["gbm", "garch_t", "bootstrap", "merton"]

# ── options / trade construction ──────────────────────────────────────────────
ACCOUNT_SIZE = 5_000.0       # your defined-risk account
RISK_PCT = 0.02              # max 2% of account at risk per trade -> $100 cap
KELLY_FRACTION = 0.25        # fractional Kelly reported alongside the fixed cap
CONTRACT_MULTIPLIER = 100    # 1 option contract = 100 shares
TARGET_DTE = 38             # aim ~30-45 days to expiry so seasonal drift can act
DTE_TOLERANCE = 12           # acceptable +/- window when matching a real expiry

# Strike selection by delta for the bear put spread (and friends).
LONG_PUT_DELTA = -0.40       # buy the ~40-delta put (near the money)
SHORT_PUT_DELTA = -0.20      # sell the ~20-delta put (further OTM)

# ── io ────────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(ROOT, "cache")
OUTPUT_DIR = os.path.join(ROOT, "output")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# How stale cached price data may be before we re-download (days).
CACHE_MAX_AGE_DAYS = 1
