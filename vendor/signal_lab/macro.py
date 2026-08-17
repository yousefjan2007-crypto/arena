# VENDORED from ~/signal_lab/macro.py @ de99b61 on 2026-08-17.
# Byte-identical to that source below this header. The public cloud runner
# has no sibling checkouts. DO NOT EDIT — re-vendor from the source instead.
"""
Macro regime features — market-wide context broadcast to every symbol.

The per-symbol features in features.py are cross-sectional (who's strong vs weak).
These are the opposite: one value per *date*, shared by all symbols, describing the
macro regime the whole market is in. The model can then learn regime-conditional
behavior (e.g. momentum works differently when the curve is inverted or credit is
widening). Most useful for the medium/long horizons.

Sources are deliberately DAILY, MARKET-BASED series (yfinance), not lagged economic
releases (CPI, payrolls) — the latter publish weeks after their observation date and
joining them by observation date is a look-ahead leak. Everything here is known at
that day's close:

  us10y        10-year Treasury yield (^TNX) — level + 21d change
  curve_10y3m  yield-curve slope, 10y minus 3m (^TNX − ^IRX) — inversion = recession signal
  vix          implied vol level, 21d change, and trailing-252d percentile (fear regime)
  dxy_mom      US dollar index momentum (DX-Y.NYB) — strong USD pressures risk assets
  credit_mom   HYG/LQD relative momentum — risk appetite (falling = credit stress / risk-off)

All transforms are rolling/backward-looking, so a value on date t uses only data <= t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
import data as smdata

MACRO_TICKERS = {"tnx": "^TNX", "irx": "^IRX", "vix": "^VIX",
                 "dxy": "DX-Y.NYB", "hyg": "HYG", "lqd": "LQD"}

MACRO_COLUMNS = ["us10y", "us10y_chg21", "curve_10y3m", "curve_chg21",
                 "vix", "vix_chg21", "vix_pct", "dxy_mom21", "credit_mom21"]


def _close(ticker: str) -> pd.Series:
    df = smdata.fetch_history(ticker, start="2000-01-01")
    return df["close"].astype(float) if len(df) else pd.Series(dtype=float)


def macro_frame(asof: pd.Timestamp | None = None) -> pd.DataFrame:
    """Daily macro feature frame indexed by date. Point-in-time as of `asof`."""
    s = {k: _close(v) for k, v in MACRO_TICKERS.items()}
    idx = s["vix"].index if len(s["vix"]) else None
    if idx is None:
        return pd.DataFrame(columns=MACRO_COLUMNS)

    def align(x):
        return x.reindex(x.index.union(idx)).ffill().reindex(idx)

    tnx, irx, vix = s["tnx"], align(s["irx"]), s["vix"]
    dxy, hyg, lqd = align(s["dxy"]), align(s["hyg"]), align(s["lqd"])

    out = pd.DataFrame(index=idx)
    out["us10y"] = tnx
    out["us10y_chg21"] = tnx.diff(21)
    out["curve_10y3m"] = tnx - irx                      # >0 normal, <0 inverted
    out["curve_chg21"] = out["curve_10y3m"].diff(21)
    out["vix"] = vix
    out["vix_chg21"] = vix.diff(21)
    out["vix_pct"] = vix.rolling(252).rank(pct=True)    # trailing-year fear percentile
    out["dxy_mom21"] = dxy / dxy.shift(21) - 1.0
    out["credit_mom21"] = (hyg / hyg.shift(21)) - (lqd / lqd.shift(21))   # risk-appetite proxy

    out = out.dropna(how="all")
    if asof is not None:
        out = out[out.index <= pd.Timestamp(asof)]
    return out


def attach(panel: pd.DataFrame, asof: pd.Timestamp | None = None) -> pd.DataFrame:
    """Broadcast macro columns onto a (date, symbol) panel by matching the date level."""
    mf = macro_frame(asof=asof)
    if len(mf) == 0:
        for c in MACRO_COLUMNS:
            panel[c] = np.nan
        return panel
    dates = panel.index.get_level_values("date")
    for c in MACRO_COLUMNS:
        panel[c] = dates.map(mf[c]).to_numpy(dtype=float)
    return panel


if __name__ == "__main__":
    mf = macro_frame(asof="2026-06-03")
    print(f"Macro frame: {mf.shape[0]} dates x {mf.shape[1]} cols, "
          f"{mf.index[0].date()} -> {mf.index[-1].date()}")
    print("\nLatest macro regime:")
    print(mf.iloc[-1].round(3).to_string())
    inv = mf["curve_10y3m"].iloc[-1]
    print(f"\nYield curve 10y-3m = {inv:.2f} ({'INVERTED' if inv < 0 else 'normal'}); "
          f"VIX percentile = {mf['vix_pct'].iloc[-1]:.2f}")
