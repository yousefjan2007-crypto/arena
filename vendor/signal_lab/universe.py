# VENDORED from ~/signal_lab/universe.py @ de99b61 on 2026-08-17.
# Byte-identical to that source below this header. The public cloud runner
# has no sibling checkouts. DO NOT EDIT — re-vendor from the source instead.
"""
Universe definition: which symbols we track, what asset class each is, and which
sector SPDR anchors each equity for relative-strength features.

Universe = SPY + 11 sector SPDRs + ~100 liquid large-caps + BTC/ETH.

Large-cap membership and GICS sectors come from the live S&P 500 list (Wikipedia)
when reachable, cached to data/sp500.csv; otherwise we fall back to a curated
static list. Sector -> SPDR mapping powers the relative-strength features in
features.py. History is fetched through the existing sell_in_may data layer.
"""
from __future__ import annotations

import io
import os
import ssl
import urllib.request

import certifi
import pandas as pd

import config                       # sets sys.path so the next import works
import data as smdata               # sell_in_may/data.py

_SSL_CTX = ssl.create_default_context(cafile=certifi.where())   # same fix as monitor.py

# GICS sector -> sector SPDR ETF (the relative-strength anchor).
GICS_TO_SPDR = {
    "Information Technology": "XLK",
    "Financials": "XLF",
    "Energy": "XLE",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Utilities": "XLU",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}

_SP500_CACHE = os.path.join(config.DATA_DIR, "sp500.csv")


def get_sp500_table() -> pd.DataFrame | None:
    """Symbol + GICS sector for current S&P 500, from Wikipedia (cached). None on failure."""
    if os.path.exists(_SP500_CACHE):
        try:
            return pd.read_csv(_SP500_CACHE)
        except Exception:
            pass
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15, context=_SSL_CTX) as resp:
            html = resp.read().decode("utf-8")
        tables = pd.read_html(io.StringIO(html))
        df = tables[0][["Symbol", "GICS Sector"]].rename(
            columns={"Symbol": "symbol", "GICS Sector": "sector"})
        df["symbol"] = df["symbol"].str.replace(".", "-", regex=False)  # BRK.B -> BRK-B
        df.to_csv(_SP500_CACHE, index=False)
        return df
    except Exception as exc:
        print(f"  [info] S&P 500 list unavailable ({exc}); using static fallback")
        return None


def build_universe() -> tuple[list[str], dict[str, dict]]:
    """Return (symbols, meta) where meta[sym] = {asset_class, sector_etf}."""
    sp = get_sp500_table()
    sector_lookup: dict[str, str] = {}
    if sp is not None:
        for _, row in sp.iterrows():
            sector_lookup[row["symbol"]] = GICS_TO_SPDR.get(row["sector"], "SPY")

    # large-cap selection: live list capped to a liquid core, else static fallback
    if sp is not None and config.USE_LIVE_SP500_LIST:
        # keep names that intersect our curated liquid core, then top up from the list
        core = [s for s in config.LARGECAP_FALLBACK if s in set(sp["symbol"])]
        extra = [s for s in sp["symbol"] if s not in set(core)]
        largecaps = (core + extra)[:150]
    else:
        largecaps = list(config.LARGECAP_FALLBACK)

    meta: dict[str, dict] = {}
    for s in config.ETFS:
        meta[s] = {"asset_class": "etf", "sector_etf": s if s != "SPY" else "SPY"}
    for s in config.CRYPTO:
        meta[s] = {"asset_class": "crypto", "sector_etf": None}
    for s in largecaps:
        meta[s] = {"asset_class": "equity", "sector_etf": sector_lookup.get(s, "SPY")}

    # de-dupe while preserving order: ETFs first (anchors), then equities, then crypto
    symbols = list(dict.fromkeys(config.ETFS + largecaps + config.CRYPTO))
    return symbols, meta


def load_history(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV for every symbol; drop those with too little history."""
    hist: dict[str, pd.DataFrame] = {}
    for s in symbols:
        start = "2015-01-01" if s.endswith("-USD") else "1995-01-01"
        df = smdata.fetch_history(s, start=start)
        if len(df) >= config.MIN_HISTORY_DAYS:
            hist[s] = df
    return hist


if __name__ == "__main__":
    syms, meta = build_universe()
    n_eq = sum(1 for m in meta.values() if m["asset_class"] == "equity")
    n_etf = sum(1 for m in meta.values() if m["asset_class"] == "etf")
    n_cr = sum(1 for m in meta.values() if m["asset_class"] == "crypto")
    print(f"Universe: {len(syms)} symbols  ({n_etf} ETFs, {n_eq} equities, {n_cr} crypto)")
    # show a few sector mappings
    for s in ["AAPL", "JPM", "XOM", "BTC-USD", "XLK"]:
        if s in meta:
            print(f"  {s:8s} class={meta[s]['asset_class']:7s} anchor={meta[s]['sector_etf']}")
