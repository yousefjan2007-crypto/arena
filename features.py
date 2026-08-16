"""
Feature adapter: signal_lab's point-in-time panel -> arena's (dates x symbols) grid.

arena does NOT compute features. The point-in-time discipline that makes a
backtest honest already exists, tested, in signal_lab/features.py (asof
truncation, rolling/expanding/shift only, an expanding same-month seasonal that
excludes the current month) and signal_lab/macro.py. Re-deriving any of it here
would be a second implementation to keep correct, and the second one always
rots. This module's whole job is SHAPE: take that long (date, symbol) DataFrame
and lay each column out as the dense `(n_dates, n_symbols)` float32 matrix the
sandbox steps through by integer index.

    build_features(market, asof) -> None      # fills market.features in place

What is dropped, and why they are not features:
  _dollar_vol   signal_lab's liquidity helper (it drops it from model input too)
  asset_class   a raw category code (0/1/2); an unordered label, not a magnitude

What is KEPT that may look odd: the macro columns (us10y, vix, vix_pct, ...) are
one value per DATE, broadcast identically across every symbol. They carry no
cross-sectional information, but they are exactly what a regime-conditional model
needs — and strategy.py's `vix_pct_80` regime filter reads `vix_pct` from here.

OFFLINE BY CONTRACT. signal_lab's macro frame fetches its tickers through
sell_in_may/data.py, which re-downloads any cache file older than
config.CACHE_MAX_AGE_DAYS. A download inside an evaluation path would make two
runs of the same generation see different data, so `_cache_only()` pins that
staleness bound for the duration of the build: refreshing the cache is
run_generation.py's job, before anything gets evaluated.

CACHING. The pivoted result is joblib-cached under state/panel_cache/ keyed by
(market.data_hash, asof). Every build round-trips through the cache file before
returning, so the arrays a fresh build hands back are byte-identical to the ones
a later run loads.

IDENTITY: `market.panel_hash` IS THE LIKE-FOR-LIKE KEY FOR GATE G1, NOT
`data_hash`. datafeed's `data_hash` covers the equity bars only — the symbol
list, the calendar ends, each symbol's bar count and last close. The panel also
depends on six macro tickers (^TNX ^IRX ^VIX DX-Y.NYB HYG LQD) that data_hash
never sees, so two runs could agree on data_hash and still have been scored on
different features. `panel_hash` closes that: sha256 over data_hash followed by
every (feature name, array bytes) pair in canonical name order, so it is a
digest of exactly what the strategy was handed — data AND features AND macro.
G1 must compare panel_hash; data_hash stays as it is and keeps its own meaning.

NaN bytes are canonicalised before hashing (every NaN, whatever payload or sign
the producing libm chose, is written as one bit pattern; infinities are left
alone), so the digest cannot differ over a NaN payload two platforms disagree
about. Float VALUES can still differ across platforms — that is the standing
per-platform determinism caveat in DESIGN, and the cloud is canonical.

Why (data_hash, asof) is still a sufficient CACHE key even though it is an
insufficient G1 key: the cache is a memo, not an identity. A macro-only cache
restatement could serve a panel built from slightly older macro bars — and
because panel_hash is computed from the arrays actually attached, whether they
came from disk or from a fresh build, that panel is still recorded for exactly
what it is. The cache can only make two runs agree, never disagree silently.
"""
from __future__ import annotations

import contextlib
import hashlib
import os

import joblib
import numpy as np
import pandas as pd

import config                       # FIRST: puts the siblings on sys.path
import datafeed

# Loaded by explicit path: a plain `import features` inside arena imports THIS file.
_sl_features = config.import_sibling("features", config.SIGNAL_LAB)
_sl_universe = config.import_sibling("universe", config.SIGNAL_LAB)

# Panel columns that are not model input. Same intent as signal_lab's
# feature_columns(), plus asset_class (a category code, not a magnitude).
NON_FEATURES = ("_dollar_vol", "asset_class")

# signal_lab/features.py::_expanding_seasonal writes its output here. strategy.py's
# seasonal_rule family scores off this column and refuses to substitute anything
# else, so the name is part of the interface between the two modules.
SEASONAL_COL = "seasonality"

PANEL_CACHE_DIR = os.path.join(config.STATE_DIR, "panel_cache")


# ── offline guard ──────────────────────────────────────────────────────────────
@contextlib.contextmanager
def _cache_only():
    """Make every sibling `fetch_history` call read the cache and never download.

    sell_in_may/data.py decides freshness from config.CACHE_MAX_AGE_DAYS, and the
    siblings resolve `import config` to arena's config — so raising it here is
    enough to keep macro.py (the one fetcher on this path) offline.
    """
    saved = config.CACHE_MAX_AGE_DAYS
    config.CACHE_MAX_AGE_DAYS = float("inf")
    try:
        yield
    finally:
        config.CACHE_MAX_AGE_DAYS = saved


# ── inputs to signal_lab's builder ─────────────────────────────────────────────
def _history(symbols) -> dict:
    """OHLCV frames straight from the shared CSV cache, never the network.

    The benchmark is always included even when it is not a tradable symbol here:
    build_panel takes its trading calendar from it and measures rs_spy against it.
    """
    wanted = list(symbols)
    if config.BENCHMARK not in wanted:
        wanted.append(config.BENCHMARK)
    hist = {}
    for sym in wanted:
        df = datafeed._read_symbol(sym, refresh=False)      # noqa: SLF001 — same project
        if len(df):
            hist[sym] = datafeed._clean_frame(sym, df)      # noqa: SLF001
    if config.BENCHMARK not in hist:
        raise ValueError("no cached data for the benchmark %s" % config.BENCHMARK)
    return hist


def _meta(symbols) -> dict:
    """asset_class + sector anchor per symbol, from signal_lab's static universe.

    USE_LIVE_SP500_LIST is off (see config.py), so this is a deterministic, offline
    lookup. Anything the table does not know is treated as an equity anchored to
    SPY — the same default universe.py applies to an unmapped sector.
    """
    _, known = _sl_universe.build_universe()
    out = {}
    for sym in symbols:
        out[sym] = known.get(sym, {"asset_class": "equity", "sector_etf": config.BENCHMARK})
    return out


# ── panel -> grid ──────────────────────────────────────────────────────────────
def _to_grid(panel: pd.DataFrame, market) -> tuple:
    """Lay every feature column out as (n_dates, n_symbols) float32.

    Sorted column order is canonical: genome feature subsets are stored as names
    and hashed, so a run that reordered these would score the same genome
    differently. Rows the panel has no entry for (symbol not listed yet, symbol
    below the history minimum) stay NaN — models impute, rules skip.
    """
    names = tuple(sorted(c for c in panel.columns if c not in NON_FEATURES))
    date_pos = market.dates.get_indexer(panel.index.get_level_values("date"))
    col_of = {s: j for j, s in enumerate(market.symbols)}
    sym_pos = np.fromiter((col_of.get(s, -1) for s in panel.index.get_level_values("symbol")),
                          dtype=np.int64, count=len(panel))
    keep = (date_pos >= 0) & (sym_pos >= 0)
    di, sj = date_pos[keep], sym_pos[keep]

    shape = (len(market.dates), len(market.symbols))
    grid = {}
    for col in names:
        arr = np.full(shape, np.nan, dtype=np.float32)
        arr[di, sj] = panel[col].to_numpy(dtype=np.float64)[keep]
        grid[col] = arr
    return names, grid


def panel_hash(data_hash: str, names, grid) -> str:
    """Content digest of the data AND the features built from it — see the module
    docstring: this, not data_hash, is what gate G1 compares.

    Canonical feature-name order (so the digest does not depend on dict order) and
    canonicalised NaN bytes (so it does not depend on which NaN bit pattern the
    producing library chose). Infinities are hashed as they are — an inf carries
    real information about a feature and should move the digest.
    """
    h = hashlib.sha256()
    h.update(data_hash.encode())
    for name in names:
        arr = grid[name]
        clean = arr.copy()
        np.copyto(clean, np.array(np.nan, dtype=arr.dtype), where=np.isnan(clean))
        h.update(name.encode())
        h.update(clean.tobytes())
    return h.hexdigest()[:16]


def _cache_file(data_hash: str, asof: pd.Timestamp) -> str:
    return os.path.join(PANEL_CACHE_DIR, "panel_%s_%s.joblib" % (data_hash, asof.strftime("%Y%m%d")))


def _load(path: str, market):
    """Cached payload if it matches this market exactly, else None."""
    if not os.path.exists(path):
        return None
    try:
        payload = joblib.load(path)
    except Exception:
        return None                                  # corrupt/half-written: rebuild
    if payload.get("symbols") != list(market.symbols) or payload.get("n_dates") != len(market.dates):
        return None
    return payload


def build_features(market, asof=None) -> None:
    """Fill `market.features` and `market.feature_names` for `market`, as of `asof`.

    asof   date-like; defaults to the market's last bar. Truncating the sibling
           builder at asof is what makes the panel point-in-time — nothing after
           asof exists while any feature is computed.

    Mutates the MarketData in place and returns None: the panel is ~35 arrays of
    (n_dates x n_symbols), and every consumer wants them attached to the bars they
    line up with rather than passed around separately.
    """
    asof = pd.Timestamp(asof) if asof is not None else market.dates[-1]
    os.makedirs(PANEL_CACHE_DIR, exist_ok=True)
    path = _cache_file(market.data_hash, asof)

    payload = _load(path, market)
    if payload is None:
        with _cache_only():
            hist = _history(market.symbols)
            panel = _sl_features.build_panel(hist, _meta(list(hist)), asof=asof)
        names, grid = _to_grid(panel, market)
        payload = {"feature_names": names, "grid": grid, "symbols": list(market.symbols),
                   "n_dates": len(market.dates), "data_hash": market.data_hash,
                   "asof": asof}
        tmp = path + ".tmp"
        joblib.dump(payload, tmp)
        os.replace(tmp, path)                        # never leave a half-written cache
        # Hand back exactly what a later run will load, not the pre-serialisation
        # objects — "the cache is byte-stable" is then true by construction.
        payload = joblib.load(path)

    market.features.clear()
    for name, arr in payload["grid"].items():
        arr.setflags(write=False)                    # history is immutable, features too
        market.features[name] = arr
    market.feature_names = payload["feature_names"]
    # Computed from the arrays actually attached, cache hit or miss, so it always
    # describes what the strategy was really handed. This is the G1 identity.
    market.panel_hash = panel_hash(market.data_hash, market.feature_names, market.features)


if __name__ == "__main__":
    syms = datafeed.in_cache(_sl_universe.build_universe()[0])[:20]
    md = datafeed.load_market(syms, start=config.DATA_START)
    build_features(md)

    n_dates, n_syms = md.shape
    print("arena feature adapter")
    print("  market      :", md)
    print("  features    : %d columns x (%d dates x %d symbols) float32"
          % (len(md.feature_names), n_dates, n_syms))
    print("  3 names     :", ", ".join(md.feature_names[:3]),
          "... (seasonal column: %s%s)"
          % (SEASONAL_COL, "" if SEASONAL_COL in md.feature_names else " MISSING"))
    print("  dropped     :", ", ".join(NON_FEATURES))
    print("  data_hash   : %s   (equity bars only — NOT the G1 identity)" % md.data_hash)
    print("  panel_hash  : %s   (data + features + macro — gate G1 compares this)"
          % md.panel_hash)
    print("  cache       :", os.path.relpath(_cache_file(md.data_hash, md.dates[-1]), config.ROOT))

    dense = {k: float(np.isfinite(v).mean()) for k, v in md.features.items()}
    thin = sorted(dense.items(), key=lambda kv: kv[1])[:3]
    print("  coverage    : %.1f%% of cells finite overall; thinnest %s"
          % (100 * np.mean(list(dense.values())),
             ", ".join("%s %.0f%%" % (k, 100 * v) for k, v in thin)))

    # Byte-stability: a fresh recompute must equal what came back from the cache,
    # or two runs of the same generation would score the same genome differently.
    with _cache_only():
        panel = _sl_features.build_panel(_history(md.symbols), _meta(md.symbols),
                                         asof=md.dates[-1])
    names, grid = _to_grid(panel, md)
    same = (names == md.feature_names
            and all(grid[k].tobytes() == md.features[k].tobytes() for k in names)
            and panel_hash(md.data_hash, names, grid) == md.panel_hash)
    print("  byte-stable : %s (recomputed panel vs cached load, incl. panel_hash)"
          % ("PASS" if same else "FAIL"))
