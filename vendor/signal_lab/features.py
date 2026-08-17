# VENDORED from ~/signal_lab/features.py @ de99b61 on 2026-08-17.
# Byte-identical to that source below this header. The public cloud runner
# has no sibling checkouts. DO NOT EDIT — re-vendor from the source instead.
"""
Point-in-time feature panel — the input to every model.

THE ONE RULE: every feature value on the row dated `t` must be computable using
only data with timestamp <= t. Violate it and the backtest invents profits that
don't exist live. Defenses baked in here:
  • `asof` truncates each price series BEFORE any feature is computed.
  • Only rolling / expanding / shift operations — never centered windows, never
    whole-sample scaling (scalers live inside the model Pipeline, refit per fold).
  • The seasonal feature uses an EXPANDING same-month mean that excludes the
    current month, so it can never see the future (the classic leak when people
    reuse sell_in_may's whole-sample seasonality output).
  • Crypto is reindexed onto the equity trading calendar (weekend moves fold into
    the next session) so features and labels share one clock.

Output: a long DataFrame MultiIndexed by (date, symbol), one row per symbol/day,
numeric feature columns + an `asset_class` code. Cross-sectional features (ranks,
relative strength) are computed per-date so they are inherently point-in-time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
import macro

ASSET_CLASS_CODE = {"equity": 0, "etf": 1, "crypto": 2}


# ── per-symbol time-series features ─────────────────────────────────────────────
def _symbol_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"].astype(float)
    out = pd.DataFrame(index=df.index)

    logret = np.log(close / close.shift(1))

    # A. momentum at multiple lookbacks (log returns)
    for L in config.MOMENTUM_LOOKBACKS:
        out[f"mom_{L}"] = np.log(close / close.shift(L))
    out["mom_12_1"] = close.shift(21) / close.shift(252) - 1.0   # 12m skip last 1m

    # B. mean-reversion / oscillators
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(com=config.RSI_PERIOD - 1, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=config.RSI_PERIOD - 1, adjust=False).mean()
    out["rsi"] = 100 - 100 / (1 + gain / loss)
    roll20 = close.rolling(20)
    out["bb_z"] = (close - roll20.mean()) / roll20.std()
    out["rev_1"] = -logret                                       # 1d reversal
    out["rev_5"] = -(close / close.shift(5) - 1.0)               # 5d reversal

    # C. volatility regime (rolling realized vol; no per-row GARCH for speed)
    ann = np.sqrt(config.TRADING_DAYS_YEAR)
    rv21 = logret.rolling(21).std() * ann
    rv63 = logret.rolling(63).std() * ann
    rv252 = logret.rolling(252).std() * ann
    out["rv21"] = rv21
    out["vol_regime"] = rv21 / rv252                             # high vs own history
    out["vol_of_vol"] = rv21.rolling(21).std()

    # D. trend / distance from moving averages
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    out["px_ema50"] = close / ema50 - 1.0
    out["px_ema200"] = close / ema200 - 1.0
    out["ema50_200"] = ema50 / ema200 - 1.0
    out["ema_cross"] = np.sign(ema9 - ema21)

    # G. volume / liquidity
    dvol = close * df["volume"].astype(float)
    out["dvol_z"] = (dvol - dvol.rolling(63).mean()) / dvol.rolling(63).std()
    out["vol_ratio"] = df["volume"] / df["volume"].rolling(21).mean()
    out["_dollar_vol"] = dvol.rolling(21).mean()                 # liquidity gate (not a feature)

    # F. expanding same-month seasonality (point-in-time)
    out["seasonality"] = _expanding_seasonal(close)
    return out


def _expanding_seasonal(close: pd.Series) -> pd.Series:
    """Mean monthly return for this row's calendar month, using only PRIOR years.

    Implementation: monthly returns -> within each month-of-year, expanding mean
    shifted by 1 (excludes the current observation) -> mapped back to daily rows by
    (year, month). Never uses the current or any future month.
    """
    monthly = close.resample("ME").last().pct_change().dropna()
    if len(monthly) < 13:
        return pd.Series(np.nan, index=close.index)
    mdf = monthly.to_frame("r")
    mdf["month"] = mdf.index.month
    mdf["year"] = mdf.index.year
    mdf["seas"] = mdf.groupby("month")["r"].transform(lambda s: s.expanding().mean().shift(1))
    lookup = {(int(r.year), int(r.month)): r.seas for r in mdf.itertuples()}
    ym = list(zip(close.index.year, close.index.month))
    return pd.Series([lookup.get(k, np.nan) for k in ym], index=close.index)


# ── crypto onto the equity calendar ─────────────────────────────────────────────
def _align_crypto(df: pd.DataFrame, equity_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Reindex a 7-day crypto series onto the equity trading-day grid (ffill), so
    weekend moves fold into the next session and all symbols share one clock."""
    return df.reindex(df.index.union(equity_index)).ffill().reindex(equity_index)


# ── panel assembly ──────────────────────────────────────────────────────────────
def build_panel(hist: dict[str, pd.DataFrame], meta: dict[str, dict],
                asof: pd.Timestamp | None = None) -> pd.DataFrame:
    """Build the long (date, symbol) feature panel as of `asof`."""
    if asof is not None:
        asof = pd.Timestamp(asof)
        hist = {s: df[df.index <= asof] for s, df in hist.items()}

    equity_index = hist[config.BENCHMARK].index

    # per-symbol feature frames
    frames = {}
    for s, df in hist.items():
        if meta.get(s, {}).get("asset_class") == "crypto":
            df = _align_crypto(df, equity_index)
        if len(df.dropna(how="all")) < config.MIN_HISTORY_DAYS:
            continue
        f = _symbol_features(df)
        f["asset_class"] = ASSET_CLASS_CODE[meta[s]["asset_class"]]
        frames[s] = f

    panel = pd.concat(frames, names=["symbol", "date"]).swaplevel().sort_index()

    # ── cross-sectional features (per date), fully vectorized ──
    date_idx = panel.index.get_level_values("date")
    sym_idx = panel.index.get_level_values("symbol")
    mom63 = panel["mom_63"]

    # relative strength vs SPY: subtract SPY's 63d momentum on the same date
    spy_mom = frames[config.BENCHMARK]["mom_63"]
    panel["rs_spy"] = mom63.values - date_idx.map(spy_mom).to_numpy(dtype=float)

    # relative strength vs each symbol's sector anchor (reindex panel by (date, anchor))
    anchor_for = {s: (meta[s].get("sector_etf") if meta[s].get("sector_etf") in frames
                      else config.BENCHMARK) for s in frames}
    anchor_index = pd.MultiIndex.from_arrays(
        [date_idx, [anchor_for[s] for s in sym_idx]], names=["date", "symbol"])
    anchor_vals = mom63.reindex(anchor_index).to_numpy(dtype=float)
    panel["rs_sector"] = mom63.values - anchor_vals

    # cross-sectional ranks (per date) — dollar-neutral, regime-robust
    g = panel.groupby(level="date")
    for col in ["mom_21", "mom_63", "mom_126", "rs_spy", "rs_sector"]:
        panel[f"xs_{col}"] = g[col].rank(pct=True)

    # ── macro regime features (one value per date, broadcast to all symbols) ──
    panel = macro.attach(panel, asof=asof)

    return panel


def feature_columns(panel: pd.DataFrame) -> list[str]:
    """Model input columns: everything numeric except internal/leakage helpers."""
    drop = {"_dollar_vol"}
    return [c for c in panel.columns if c not in drop]


def liquidity_mask(panel: pd.DataFrame) -> pd.Series:
    """Boolean mask of rows that clear the dollar-volume liquidity gate."""
    return panel["_dollar_vol"] >= config.MIN_DOLLAR_VOL


if __name__ == "__main__":
    import universe
    syms, meta = universe.build_universe()
    subset = ["SPY", "XLK", "XLF", "XLE", "AAPL", "MSFT", "JPM", "XOM", "BTC-USD", "ETH-USD"]
    syms = [s for s in subset if s in meta]
    hist = universe.load_history(syms)
    panel = build_panel(hist, meta, asof="2026-05-29")
    print(f"Panel shape: {panel.shape}  rows over {panel.index.get_level_values('date').nunique()} dates")
    last_date = panel.index.get_level_values("date").max()
    snap = panel.xs(last_date, level="date")
    cols = ["mom_21", "mom_63", "rsi", "rv21", "vol_regime", "px_ema200",
            "seasonality", "rs_spy", "xs_mom_63"]
    print(f"\nFeature snapshot on {last_date.date()}:")
    print(snap[cols].round(3).to_string())
    print(f"\n# feature columns: {len(feature_columns(panel))}, liquid rows: {int(liquidity_mask(panel).sum())}/{len(panel)}")
