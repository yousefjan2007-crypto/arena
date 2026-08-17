# VENDORED from ~/signal_lab/cv.py @ de99b61 on 2026-08-17.
# Byte-identical to that source below this header. The public cloud runner
# has no sibling checkouts. DO NOT EDIT — re-vendor from the source instead.
"""
Leakage-free cross-validation — the firewall of the whole system.

With forward-return labels, a training row dated `t` whose label window [t, t1]
overlaps the test period has effectively seen the test outcome. Training on it and
scoring on the test set is leakage: the backtest looks brilliant and live trading
loses money. Two mechanisms stop it (López de Prado):

  PURGE   — drop any train row whose label-end t1 >= the test block's start date.
  EMBARGO — additionally drop a buffer of train dates just before the test block,
            because serial correlation contaminates the boundary even when labels
            don't literally overlap.

Both splitters operate on UNIQUE DATES, then map back to all rows on those dates,
so an entire cross-section moves together (you never train on AAPL@t while testing
MSFT@t). Walk-forward (not shuffled K-fold) because the live use is strictly forward.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator

import config


def _date_arrays(X, t1):
    row_dates = pd.DatetimeIndex(X.index.get_level_values("date")).values.astype("datetime64[ns]")
    t1v = pd.to_datetime(pd.Series(np.asarray(t1)).values).values.astype("datetime64[ns]")
    all_dates = np.array(sorted(pd.DatetimeIndex(X.index.get_level_values("date")).unique()),
                         dtype="datetime64[ns]")
    return row_dates, t1v, all_dates


class PurgedWalkForwardCV(BaseCrossValidator):
    """Expanding walk-forward CV with label purging + embargo.

    Parameters
    ----------
    t1 : array-like of label-end dates, aligned row-for-row with X (REQUIRED).
    n_splits, test_size, embargo_frac, min_train_size : see config defaults.
    """

    def __init__(self, t1, n_splits=config.CV_N_SPLITS, test_size=config.CV_TEST_DAYS,
                 embargo_frac=config.CV_EMBARGO_FRAC, min_train_size=config.CV_MIN_TRAIN_DAYS):
        self.t1 = np.asarray(t1)
        self.n_splits = n_splits
        self.test_size = test_size
        self.embargo_frac = embargo_frac
        self.min_train_size = min_train_size

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        row_dates, t1v, all_dates = _date_arrays(X, self.t1)
        n_dates = len(all_dates)
        embargo = int(np.ceil(self.embargo_frac * n_dates))
        pos = np.arange(len(X))
        first = max(self.min_train_size, n_dates - self.n_splits * self.test_size)
        for k in range(self.n_splits):
            ts = first + k * self.test_size
            te = min(ts + self.test_size, n_dates)
            if ts >= n_dates or te - ts < 2:
                break
            test_start = all_dates[ts]
            test_end = all_dates[te - 1]
            embargo_date = all_dates[max(0, ts - embargo)]

            test_mask = (row_dates >= test_start) & (row_dates <= test_end)
            # train: strictly before the embargo buffer AND label closes before test
            train_mask = (row_dates < embargo_date) & (t1v < test_start)
            if train_mask.sum() == 0 or test_mask.sum() == 0:
                continue
            yield pos[train_mask], pos[test_mask]


class CombinatorialPurgedCV(BaseCrossValidator):
    """Combinatorial purged CV (CPCV) — many train/test path combinations, used to
    estimate the Probability of Backtest Overfitting. Splits the timeline into
    `n_groups` contiguous blocks and tests every combination of `k_test` blocks,
    purging+embargoing the train side against each test block."""

    def __init__(self, t1, n_groups=6, k_test=2, embargo_frac=config.CV_EMBARGO_FRAC):
        self.t1 = np.asarray(t1)
        self.n_groups = n_groups
        self.k_test = k_test
        self.embargo_frac = embargo_frac

    def get_n_splits(self, X=None, y=None, groups=None):
        from math import comb
        return comb(self.n_groups, self.k_test)

    def split(self, X, y=None, groups=None):
        row_dates, t1v, all_dates = _date_arrays(X, self.t1)
        n_dates = len(all_dates)
        embargo = int(np.ceil(self.embargo_frac * n_dates))
        pos = np.arange(len(X))
        bounds = np.linspace(0, n_dates, self.n_groups + 1).astype(int)
        blocks = [(bounds[i], bounds[i + 1]) for i in range(self.n_groups)]

        for combo in itertools.combinations(range(self.n_groups), self.k_test):
            test_mask = np.zeros(len(X), dtype=bool)
            test_intervals = []
            for gi in combo:
                a, b = blocks[gi]
                ds, de = all_dates[a], all_dates[min(b, n_dates - 1)]
                test_mask |= (row_dates >= ds) & (row_dates <= de)
                test_intervals.append((a, b))
            # train = everything not in test, purged+embargoed around every test block
            train_mask = ~test_mask
            for (a, b) in test_intervals:
                ts_date = all_dates[a]
                te_idx = min(b + embargo, n_dates - 1)
                te_date = all_dates[te_idx]
                # purge: label must not overlap this test block's start
                train_mask &= ~((t1v >= ts_date) & (row_dates < ts_date))
                # embargo: drop train rows just after this test block
                train_mask &= ~((row_dates > all_dates[min(b, n_dates - 1)]) & (row_dates <= te_date))
            if train_mask.sum() and test_mask.sum():
                yield pos[train_mask], pos[test_mask]


if __name__ == "__main__":
    # smoke test on a small synthetic panel
    dates = pd.bdate_range("2020-01-01", periods=800)
    syms = ["A", "B", "C"]
    idx = pd.MultiIndex.from_product([dates, syms], names=["date", "symbol"])
    X = pd.DataFrame({"f": np.arange(len(idx))}, index=idx)
    h = 21
    t1 = pd.Series(idx.get_level_values("date")).groupby(
        pd.Series(idx.get_level_values("symbol"))).shift  # placeholder
    # simple t1 = date + h business days
    t1 = (pd.DatetimeIndex(idx.get_level_values("date")) + pd.offsets.BDay(h)).values

    cv = PurgedWalkForwardCV(t1, n_splits=5, test_size=63, min_train_size=200)
    print("PurgedWalkForwardCV folds:")
    for i, (tr, te) in enumerate(cv.split(X)):
        tr_dates = pd.DatetimeIndex(X.index.get_level_values("date")[tr])
        te_dates = pd.DatetimeIndex(X.index.get_level_values("date")[te])
        gap = (te_dates.min() - tr_dates.max()).days
        print(f"  fold {i}: train={len(tr):5d} (-> {tr_dates.max().date()})  "
              f"test={len(te):4d} ({te_dates.min().date()}..{te_dates.max().date()})  gap={gap}d")
    cpcv = CombinatorialPurgedCV(t1, n_groups=6, k_test=2)
    print(f"\nCombinatorialPurgedCV: {cpcv.get_n_splits()} combinations")
