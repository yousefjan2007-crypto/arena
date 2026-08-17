# VENDORED from ~/sell_in_may/data.py @ sha256:cbb56bb5eedc on 2026-08-17. (source is not a git repo; content digest instead)
# Byte-identical to that source below this header. The public cloud runner
# has no sibling checkouts. DO NOT EDIT — re-vendor from the source instead.
"""
Data layer: fetch and cache everything the engine needs.

Design choices:
- Per-ticker downloads (not a single multi-ticker call) so we keep clean OHLCV
  columns and avoid yfinance's MultiIndex column surprises.
- Local CSV cache keyed by ticker; re-downloaded only when older than
  CACHE_MAX_AGE_DAYS. Keeps repeated runs fast and offline-friendly.
- Every network call is wrapped so a single flaky source never kills the run;
  optional sources (Ken French) degrade gracefully to None.
"""
from __future__ import annotations
import os
import time
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

import config

warnings.filterwarnings("ignore", category=FutureWarning)


# ── cache helpers ──────────────────────────────────────────────────────────────
def _cache_path(name: str) -> str:
    safe = name.replace("^", "_idx_").replace("/", "_")
    return os.path.join(config.CACHE_DIR, f"{safe}.csv")


def _is_fresh(path: str) -> bool:
    if not os.path.exists(path):
        return False
    age_days = (time.time() - os.path.getmtime(path)) / 86400.0
    return age_days <= config.CACHE_MAX_AGE_DAYS


def _load_cache(path: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df if len(df) else None
    except Exception:
        return None


# ── price history ──────────────────────────────────────────────────────────────
def fetch_history(ticker: str, start: str = "1990-01-01") -> pd.DataFrame:
    """Daily OHLCV for one ticker, auto-adjusted, cached locally.

    Returns a DataFrame indexed by date with lowercase columns
    open/high/low/close/volume. Empty DataFrame on total failure.
    """
    path = _cache_path(ticker)
    if _is_fresh(path):
        cached = _load_cache(path)
        if cached is not None:
            return cached

    try:
        raw = yf.download(
            ticker, start=start, auto_adjust=True,
            progress=False, multi_level_index=False,
        )
    except Exception as exc:  # network / parsing failure
        print(f"  [warn] download failed for {ticker}: {exc}")
        cached = _load_cache(path)
        return cached if cached is not None else pd.DataFrame()

    if raw is None or len(raw) == 0:
        cached = _load_cache(path)
        return cached if cached is not None else pd.DataFrame()

    raw = raw.rename(columns=str.lower)
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in raw.columns]
    df = raw[keep].dropna(how="all")
    df.index.name = "date"
    try:
        df.to_csv(path)
    except Exception:
        pass
    return df


def get_universe_history(start: str = "1990-01-01") -> dict[str, pd.DataFrame]:
    """OHLCV for SPY + every sector ETF. Skips tickers that return nothing."""
    out: dict[str, pd.DataFrame] = {}
    for tk in config.UNIVERSE:
        df = fetch_history(tk, start=start)
        if len(df):
            out[tk] = df
            print(f"  {tk:5s} {len(df):>6,} rows  {df.index[0].date()} -> {df.index[-1].date()}")
        else:
            print(f"  {tk:5s} [skip] no data")
    return out


def get_index_history() -> pd.DataFrame:
    """Deep-history S&P 500 index (^GSPC) for the long-sample seasonality test."""
    return fetch_history(config.SP500_INDEX, start="1950-01-01")


def get_vix() -> pd.DataFrame:
    return fetch_history(config.VIX, start="1990-01-01")


def get_risk_free() -> float:
    """Latest 13-week T-bill rate as a decimal (e.g. 0.043). Falls back if absent."""
    df = fetch_history(config.RISK_FREE_TICKER, start="2015-01-01")
    if len(df) and "close" in df.columns and np.isfinite(df["close"].iloc[-1]):
        return float(df["close"].iloc[-1]) / 100.0
    return config.FALLBACK_RISK_FREE


def get_fama_french_industries() -> pd.DataFrame | None:
    """Ken French 12-industry daily portfolios (since 1926). Optional / best-effort.

    Returns a DataFrame of daily simple returns (decimal) or None if the source
    is unavailable. Used only to corroborate the seasonal effect on a much longer
    sample than the ETFs allow.
    """
    try:
        from pandas_datareader import data as web
    except Exception:
        print("  [info] pandas_datareader unavailable; skipping Ken French data")
        return None
    try:
        ds = web.DataReader("12_Industry_Portfolios_Daily", "famafrench", start="1926-01-01")
        df = ds[0].copy()              # [0] = average value-weighted daily returns (%)
        df = df.replace([-99.99, -999], np.nan) / 100.0
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as exc:
        print(f"  [info] Ken French fetch failed ({exc}); continuing without it")
        return None


# ── option chains ──────────────────────────────────────────────────────────────
def get_spot(ticker: str) -> float:
    df = fetch_history(ticker, start="2024-01-01")
    return float(df["close"].iloc[-1]) if len(df) else float("nan")


def get_option_chain(ticker: str, target_dte: int = config.TARGET_DTE):
    """Live option chain for the expiry closest to target_dte.

    Returns dict with keys: expiry (str), dte (int), spot (float),
    calls (DataFrame), puts (DataFrame). Returns None if no usable chain.
    Each leg DataFrame carries strike, bid, ask, mid, impliedVolatility.
    """
    try:
        tk = yf.Ticker(ticker)
        expiries = tk.options
    except Exception as exc:
        print(f"  [warn] no option expiries for {ticker}: {exc}")
        return None
    if not expiries:
        return None

    today = pd.Timestamp.now().normalize()
    dtes = {e: (pd.Timestamp(e) - today).days for e in expiries}
    # closest expiry to the target horizon, ignoring already-expired ones
    valid = {e: d for e, d in dtes.items() if d > 0}
    if not valid:
        return None
    expiry = min(valid, key=lambda e: abs(valid[e] - target_dte))

    try:
        chain = tk.option_chain(expiry)
    except Exception as exc:
        print(f"  [warn] chain fetch failed for {ticker} {expiry}: {exc}")
        return None

    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in ["bid", "ask", "lastPrice", "impliedVolatility", "strike"]:
            if col not in df.columns:
                df[col] = np.nan
        # mid price; fall back to lastPrice when quotes are crossed/empty
        mid = (df["bid"] + df["ask"]) / 2.0
        mid = mid.where((df["bid"] > 0) & (df["ask"] > 0), df["lastPrice"])
        df["mid"] = mid
        return df[["strike", "bid", "ask", "lastPrice", "mid", "impliedVolatility"]]

    spot = get_spot(ticker)
    return {
        "expiry": expiry,
        "dte": valid[expiry],
        "spot": spot,
        "calls": _clean(chain.calls),
        "puts": _clean(chain.puts),
    }


if __name__ == "__main__":
    print("Fetching universe ...")
    hist = get_universe_history()
    print(f"\nRisk-free rate: {get_risk_free():.4f}")
    print(f"Index rows: {len(get_index_history()):,}")
    ff = get_fama_french_industries()
    print(f"Ken French industries: {'loaded ' + str(ff.shape) if ff is not None else 'unavailable'}")
    ch = get_option_chain("XLE")
    if ch:
        print(f"XLE chain: expiry {ch['expiry']} ({ch['dte']}d), spot {ch['spot']:.2f}, "
              f"{len(ch['puts'])} puts")
