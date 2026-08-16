"""
Market data: the shared price cache -> one aligned MarketData block.

Everything downstream (env, features, evaluation) reads bars from a MarketData:
dense float64 matrices shaped (n_dates, n_symbols), all on ONE trading calendar,
with NaN meaning "this symbol was not listed / had no bar that day". Aligning
once, up front, is what lets the sandbox step forward by integer index instead of
re-joining DataFrames 8,000 times.

Design choices:
  • Cache-first by default and OFFLINE by default. The bars come from
    sell_in_may/data.py's CSV cache (fetching is that module's job, never
    re-implemented here). `load_market(..., refresh=True)` is the ONLY path that
    may touch the network; evaluation loops must never take it, or two runs of
    the same generation could see different data.
  • The trading calendar is SPY's own index (config.BENCHMARK) clipped to
    [start, end] — a real exchange calendar, not a generated bdate_range, so
    holidays and half-days are exactly right. Every symbol is reindexed onto it.
  • Integrity is checked on load and raises rather than silently repairing:
    bad prices in the sandbox mean fabricated alpha.
  • `data_hash` identifies the exact data a result was produced from (gate G1
    compares it). It hashes: the symbol list, the first and last calendar date,
    and per symbol its non-NaN row count and last close (rounded to 6dp). Cheap
    to compute, but a changed cache, window, or symbol set all move it.

`features` is an empty dict at this phase; Phase 2's features.py fills it with
the point-in-time panel keyed by column name.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

import config                    # FIRST: this is what puts the siblings on sys.path
import data as smdata            # sell_in_may/data.py — the shared CSV cache

PRICE_COLUMNS = ["open", "close", "volume"]


class MarketData:
    """Daily bars for a fixed symbol set, aligned on one trading calendar.

    Attributes:
      dates       DatetimeIndex (n_dates,) — the trading calendar, ascending
      symbols     list[str] (n_symbols,) — sorted, deduplicated
      open/close  float64 (n_dates, n_symbols), NaN where the symbol had no bar
      volume      float64 shares
      dollar_vol  float64 close * volume (liquidity proxy)
      features    dict[str, np.ndarray] — empty until Phase 2
      data_hash   str — see the module docstring
    """

    def __init__(self, dates, symbols, open_, close, volume, features=None):
        self.dates = pd.DatetimeIndex(dates)
        self.symbols = list(symbols)
        self.open = np.asarray(open_, dtype=np.float64)
        self.close = np.asarray(close, dtype=np.float64)
        self.volume = np.asarray(volume, dtype=np.float64)
        self.dollar_vol = self.close * self.volume
        self.features = dict(features) if features else {}
        _check_integrity(self)
        self.data_hash = _hash_market(self)
        # History is immutable once loaded: a strategy handed an obs row must not
        # be able to rewrite the past it is being scored against.
        for arr in (self.open, self.close, self.volume, self.dollar_vol):
            arr.setflags(write=False)

    def __len__(self) -> int:
        return len(self.dates)

    @property
    def shape(self) -> tuple:
        return self.close.shape

    def pos(self, when, side: str = "left") -> int:
        """Calendar index for a date. side="left" -> first date >= when,
        side="right" -> last date <= when. Used to turn a start/end into ints."""
        ts = pd.Timestamp(when)
        if side == "left":
            i = int(self.dates.searchsorted(ts, side="left"))
            return min(i, len(self.dates) - 1)
        i = int(self.dates.searchsorted(ts, side="right")) - 1
        return max(i, 0)

    def __repr__(self) -> str:
        return "MarketData(%d dates %s..%s, %d symbols, hash=%s)" % (
            len(self.dates), self.dates[0].date(), self.dates[-1].date(),
            len(self.symbols), self.data_hash)


# ── integrity ──────────────────────────────────────────────────────────────────
def _check_integrity(md: MarketData) -> None:
    """Raise ValueError on anything that would make the sandbox lie."""
    if len(md.dates) == 0:
        raise ValueError("empty trading calendar")
    if md.dates.has_duplicates:
        raise ValueError("duplicate dates in the trading calendar")
    if not md.dates.is_monotonic_increasing:
        raise ValueError("trading calendar is not monotonically increasing")
    for name in ("open", "close", "volume"):
        arr = getattr(md, name)
        if arr.shape != (len(md.dates), len(md.symbols)):
            raise ValueError("%s has shape %s, expected %s"
                             % (name, arr.shape, (len(md.dates), len(md.symbols))))
    for name in ("open", "close"):
        arr = getattr(md, name)
        bad = np.isfinite(arr) & (arr <= 0.0)
        if bad.any():
            j = int(np.argmax(bad.any(axis=0)))
            raise ValueError("non-positive %s price for %s" % (name, md.symbols[j]))
    if np.nanmin(np.where(np.isfinite(md.volume), md.volume, 0.0)) < 0.0:
        raise ValueError("negative volume")
    # A bar is either fully there or fully absent: a cell with a close but no open
    # cannot be traded next morning, and one with an open but no close cannot be
    # marked. Either would quietly corrupt the accounting.
    mismatch = np.isfinite(md.open) != np.isfinite(md.close)
    if mismatch.any():
        j = int(np.argmax(mismatch.any(axis=0)))
        i = int(np.argmax(mismatch[:, j]))
        raise ValueError("open/close presence mismatch for %s on %s"
                         % (md.symbols[j], md.dates[i].date()))


def _hash_market(md: MarketData) -> str:
    h = hashlib.sha256()
    h.update(("|".join(md.symbols)).encode())
    h.update(("%s|%s" % (md.dates[0].date(), md.dates[-1].date())).encode())
    for j, sym in enumerate(md.symbols):
        col = md.close[:, j]
        ok = np.isfinite(col)
        n = int(ok.sum())
        last = float(col[ok][-1]) if n else float("nan")
        h.update(("%s:%d:%.6f" % (sym, n, last)).encode())
    return h.hexdigest()[:16]


# ── cache access (sell_in_may/data.py owns the fetching and the file layout) ────
def _cache_path(symbol: str) -> str:
    return smdata._cache_path(symbol)     # noqa: SLF001 — the layout lives there, not here


def in_cache(symbols) -> list[str]:
    """Subset of `symbols` that already has a cached CSV. Never touches the network."""
    return [s for s in symbols if os.path.exists(_cache_path(s))]


def _read_symbol(symbol: str, refresh: bool) -> pd.DataFrame:
    """OHLCV for one symbol. refresh=True delegates to sell_in_may's fetcher (which
    may download); refresh=False reads the cached CSV and nothing else."""
    if refresh:
        return smdata.fetch_history(symbol, start=config.DATA_START)
    path = _cache_path(symbol)
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df if len(df) else pd.DataFrame()


def _clean_frame(symbol: str, df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in PRICE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("%s cache is missing column(s) %s" % (symbol, missing))
    if df.index.has_duplicates:
        raise ValueError("%s cache has duplicate dates (e.g. %s)"
                         % (symbol, df.index[df.index.duplicated()][0].date()))
    if not df.index.is_monotonic_increasing:
        raise ValueError("%s cache dates are not sorted" % symbol)
    return df[PRICE_COLUMNS].astype(float)


# ── loader ─────────────────────────────────────────────────────────────────────
def load_market(symbols, start=None, end=None, refresh: bool = False) -> MarketData:
    """Build a MarketData for `symbols` over [start, end] from the shared cache.

    symbols   iterable of tickers; deduplicated and SORTED, so the data hash does
              not depend on the caller's ordering.
    start/end date-like; default to config.DATA_START and the end of the calendar.
    refresh   True lets sell_in_may/data.py re-download stale symbols. Leave False
              inside anything that must be reproducible.

    Raises ValueError if a requested symbol has no cached data at all, or if the
    bars fail an integrity check. Symbols that are simply not listed during the
    window are kept as all-NaN columns (correct: they were untradable then).
    """
    symbols = sorted(set(symbols))
    if not symbols:
        raise ValueError("no symbols requested")

    bench = config.BENCHMARK
    frames = {}
    for sym in symbols + ([bench] if bench not in symbols else []):
        df = _read_symbol(sym, refresh)
        if not len(df):
            raise ValueError("no cached data for %s (run with refresh=True to fetch)" % sym)
        frames[sym] = _clean_frame(sym, df)

    # The calendar is the benchmark's actual trading days, clipped to the window.
    cal = frames[bench].index
    lo = pd.Timestamp(start if start is not None else config.DATA_START)
    hi = pd.Timestamp(end) if end is not None else cal[-1]
    dates = cal[(cal >= lo) & (cal <= hi)]
    if len(dates) == 0:
        raise ValueError("no trading days between %s and %s" % (lo.date(), hi.date()))

    n, m = len(dates), len(symbols)
    open_ = np.full((n, m), np.nan)
    close = np.full((n, m), np.nan)
    volume = np.full((n, m), np.nan)
    for j, sym in enumerate(symbols):
        aligned = frames[sym].reindex(dates)     # NaN on days the symbol had no bar
        open_[:, j] = aligned["open"].to_numpy(dtype=np.float64)
        close[:, j] = aligned["close"].to_numpy(dtype=np.float64)
        volume[:, j] = aligned["volume"].to_numpy(dtype=np.float64)

    return MarketData(dates, symbols, open_, close, volume)


if __name__ == "__main__":
    syms = ["SPY", "AAPL", "MSFT", "JPM", "XOM"]
    md = load_market(syms, start=config.DATA_START)
    print("arena datafeed")
    print("  requested   :", ", ".join(syms))
    print("  ", md)
    print("  shape       : %d dates x %d symbols" % md.shape)
    print("  calendar    : %s -> %s" % (md.dates[0].date(), md.dates[-1].date()))
    print("  data_hash   :", md.data_hash)
    for j, s in enumerate(md.symbols):
        col = md.close[:, j]
        ok = np.isfinite(col)
        print("    %-5s %5d bars  %s -> %s  last close %8.2f  avg $vol %6.1fM"
              % (s, ok.sum(), md.dates[ok][0].date(), md.dates[ok][-1].date(),
                 col[ok][-1], np.nanmean(md.dollar_vol[:, j]) / 1e6))
    print("  features    : %d (Phase 2 fills these)" % len(md.features))
