"""
Genome -> StrategyAgent: the thing that actually trades a genome in the sandbox.

    result = StrategyAgent(genome, market, cost).run_episode(start, end)
    # {"dates", "daily_net", "daily_gross", "turnover", "costs"}

The agent decides; env.py enforces. Weights leave this module as intentions —
whole shares, per-name caps, leverage, the dust filter and the fills are the env's
business, and a genome cannot talk its way past any of them.

FOUR THINGS DECIDED HERE, EACH WITH A REASON:

1. RULE SCORES ARE CENTRED — `pct_rank - 0.5`, NOT a raw 0..1 percentile rank —
   so that zero means "the median name today" and the sign of a score is a
   direction rather than an artefact of the scale. DO NOT "correct" this back to
   a raw rank: the selection rule below never shorts a positive score, so on a
   raw rank every rule genome would be silently long-only, and meanrev's trend
   gate ("only names below their 200DMA get a short score") would be unreachable
   code. Centring is also what makes score-proportional weighting meaningful — on
   a raw rank a 99th-percentile name would weigh barely more than a 51st. The
   model families need no centring: a predicted return and a P(top) − P(bottom)
   spread are already signed. Selection then requires score > 0 to go long and
   < 0 to go short — the "no forced sign flips" rule, which also drops gated-out
   names (score exactly 0).

2. HOLDING RESTATES THE INTENDED WEIGHTS, NOT THE BOOK. Between rebalances the
   agent re-sends the weights it last chose (minus anything stopped out, times
   the current overlay scale). Re-sending the *realised* book instead — the
   obvious alternative — would compound every overlay: a regime filter at 0.5
   would cut the book by half again the next day, and again the day after. The
   cost of restating intentions is a trickle of drift-correcting orders, which
   the env's MIN_POSITION_USD filter absorbs at this account size.

3. REFITS ARE PURGED AND EMBARGOED AT FIT TIME. A model refitted at bar `f`
   trains only on rows whose label has already RESOLVED with room to spare:
       row index t is trainable  <=>  t + horizon <= f - WF_EMBARGO_DAYS
   The streaming walk-forward means nothing after `f` exists yet, so this is the
   only leak left to close — the label window itself. Every fit appends
   (fit_date, max_t1_used, n_rows) to `fit_audit` so verify.py can prove it from
   the outside rather than take this docstring's word for it.

4. NO VAULT LOGIC LIVES HERE. The agent simulates whatever window it is handed;
   which days may be scored is evaluate.py's decision, not the simulator's.

Everything the agent reads at bar t is backward-looking: features are asof-built
by features.py, realised vol and moving averages are trailing windows, and the
label matrix is only ever indexed below the purge boundary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config                       # FIRST: puts the siblings on sys.path
import features as arena_features
from env import CostModel, MarketEnv
from genome import BOUNDS


# ── trailing statistics (backward windows only) ────────────────────────────────
def _cached_roll(market, kind: str, window: int) -> np.ndarray:
    """A trailing statistic of `market.close`, computed once per market.

    Every genome on a market gets the same realised-vol and moving-average
    matrices — they depend on the bars and the window, never on the genome — so
    Phase 3's F0 screen would otherwise rebuild them once per agent (64 genomes x
    3 eras). The memo hangs off the MarketData instance rather than a module
    global so it dies with the market it describes; it is safe because a
    MarketData's price arrays are read-only from construction.
    """
    memo = getattr(market, "_roll_memo", None)
    if memo is None:
        memo = {}
        market._roll_memo = memo
    key = (kind, window)
    if key not in memo:
        fn = _rolling_std if kind == "vol" else _rolling_mean
        arr = fn(market.close, window)
        arr.setflags(write=False)
        memo[key] = arr
    return memo[key]


def _rolling_std(mat: np.ndarray, window: int) -> np.ndarray:
    """Trailing std of daily log returns, annualised. Row t uses rows <= t."""
    with np.errstate(divide="ignore", invalid="ignore"):
        logret = np.log(mat[1:] / mat[:-1])
    logret = np.vstack([np.full((1, mat.shape[1]), np.nan), logret])
    out = pd.DataFrame(logret).rolling(window).std().to_numpy()
    return out * np.sqrt(config.TRADING_DAYS_YEAR)


def _rolling_mean(mat: np.ndarray, window: int) -> np.ndarray:
    return pd.DataFrame(mat).rolling(window).mean().to_numpy()


def _pct_rank(row: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Cross-sectional percentile rank in [0, 1] over `valid`, NaN elsewhere."""
    out = np.full(row.shape, np.nan)
    idx = np.flatnonzero(valid)
    if len(idx) == 0:
        return out
    if len(idx) == 1:
        out[idx] = 0.5
        return out
    order = np.argsort(row[idx], kind="stable")
    ranks = np.empty(len(idx), dtype=np.float64)
    ranks[order] = np.arange(len(idx), dtype=np.float64)
    out[idx] = ranks / (len(idx) - 1.0)
    return out


def _date_level(arr: np.ndarray) -> np.ndarray:
    """Collapse a macro column (one value per date, broadcast across symbols) to a
    per-date vector, taking the first symbol that actually has the row."""
    finite = np.isfinite(arr)
    first = finite.argmax(axis=1)
    vals = arr[np.arange(arr.shape[0]), first]
    return np.where(finite.any(axis=1), vals, np.nan)


class StrategyAgent:
    """One genome, one market, one cost model. `run_episode` is the whole API."""

    def __init__(self, genome, market, cost=None, cfg=config):
        self.genome = genome
        self.market = market
        self.cost = cost if cost is not None else CostModel()
        self.cfg = cfg
        self.n = len(market.symbols)
        self.close = market.close
        self._model = None

        sig = genome.signal
        self.is_model = genome.is_model
        self.params = sig.pdict

        # Trailing stats every family may need: never per bar, and — since they
        # depend only on the bars — never per genome either. See _cached_roll.
        self._vol = _cached_roll(market, "vol", cfg.REALIZED_VOL_DAYS)
        self._ma = _cached_roll(market, "ma", cfg.TREND_MA_DAYS)

        self._seasonal = None
        if sig.family == "seasonal_rule":
            col = arena_features.SEASONAL_COL
            if col not in getattr(market, "feature_names", ()):  # never substitute
                raise ValueError("seasonal_rule needs the '%s' feature column; "
                                 "run features.build_features(market) first" % col)
            self._seasonal = market.features[col]

        self._spy = None
        if genome.risk.regime_filter == "spy_200dma":
            if cfg.BENCHMARK not in market.symbols:
                raise ValueError("regime filter spy_200dma needs %s in the universe"
                                 % cfg.BENCHMARK)
            self._spy = market.symbols.index(cfg.BENCHMARK)
        self._vix_pct = None
        if genome.risk.regime_filter == "vix_pct_80":
            col = "vix_pct"
            if col not in getattr(market, "feature_names", ()):
                raise ValueError("regime filter vix_pct_80 needs the '%s' column" % col)
            self._vix_pct = _date_level(market.features[col])

        # Model families: the design matrix and the labels, laid out once.
        self._X = self._y = self._y_class = None
        if self.is_model:
            missing = [f for f in sig.features if f not in getattr(market, "feature_names", ())]
            if missing:
                raise ValueError("features not in this market's panel: %s" % ", ".join(missing))
            self._X = np.stack([market.features[f] for f in sig.features], axis=2)
            self._y = self._forward_logret(sig.horizon)
            if sig.family in ("logistic", "hgb"):
                self._y_class = self._terciles(self._y)

    # ── labels ─────────────────────────────────────────────────────────────────
    def _forward_logret(self, h: int) -> np.ndarray:
        """log(close[t+h] / close[t]); the final h rows are unlabelled (NaN)."""
        out = np.full(self.close.shape, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[:-h] = np.log(self.close[h:] / self.close[:-h])
        return np.where(np.isfinite(out), out, np.nan)

    @staticmethod
    def _terciles(y: np.ndarray) -> np.ndarray:
        """Per-date cross-sectional terciles of the forward return: +1 top, -1
        bottom, 0 middle. A transform of the LABEL only — the purge decides which
        rows may be trained on."""
        out = np.zeros(y.shape, dtype=np.int8)
        for i in range(y.shape[0]):
            valid = np.isfinite(y[i])
            if valid.sum() < 3:
                continue
            r = _pct_rank(y[i], valid)
            out[i] = np.where(np.isfinite(r) & (r >= 2 / 3), 1,
                              np.where(np.isfinite(r) & (r <= 1 / 3), -1, 0))
        return out

    # ── model ──────────────────────────────────────────────────────────────────
    def _pipeline(self):
        """Fresh Pipeline per refit — a reused one would carry the previous fold's
        imputation medians and scaling into the next fit. The final step is named
        "clf" and takes clf__sample_weight, matching signal_lab's plumbing so a
        model trained here stays usable there.

        keep_empty_features is load-bearing: a feature that is all-NaN over the
        training window (a symbol set with no macro history yet) would otherwise be
        DROPPED by the imputer, and the fitted model would expect fewer columns
        than the live cross-section hands it.
        """
        family, p = self.genome.signal.family, self.params
        steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
        if family in ("ridge", "logistic"):
            steps.append(("scale", StandardScaler()))
        if family == "ridge":
            clf = Ridge(alpha=p["alpha"], random_state=self.cfg.SEED)
        elif family == "logistic":
            clf = LogisticRegression(C=p["C"], max_iter=self.cfg.LOGIT_MAX_ITER,
                                     random_state=self.cfg.SEED)
        else:
            clf = HistGradientBoostingClassifier(
                learning_rate=p["learning_rate"], max_depth=p["max_depth"],
                max_iter=p["max_iter"], min_samples_leaf=p["min_samples_leaf"],
                random_state=self.cfg.SEED)
        return Pipeline(steps + [("clf", clf)])

    def _fit(self, f: int, fit_audit) -> bool:
        """Refit at bar f on resolved, embargoed labels only. False = not yet."""
        h = self.genome.signal.horizon
        last = f - self.cfg.WF_EMBARGO_DAYS - h        # the purge boundary, in bars
        if last < 0 or last + 1 < self.cfg.WF_MIN_TRAIN_DAYS:
            return False

        y_mat = self._y if self.genome.signal.family == "ridge" else self._y_class
        X = self._X[:last + 1].reshape(-1, self._X.shape[2])
        y = y_mat[:last + 1].reshape(-1)
        labelled = np.isfinite(self._y[:last + 1]).reshape(-1)
        usable = labelled & np.isfinite(X).any(axis=1)
        rows = np.flatnonzero(usable)
        if len(rows) < self.cfg.WF_MIN_TRAIN_DAYS:
            return False
        if y_mat is not self._y and len(np.unique(y[rows])) < 2:
            return False                                # a classifier needs 2 classes

        model = self._pipeline()
        model.fit(np.asarray(X[rows], dtype=np.float64), y[rows])
        self._model = model

        if fit_audit is not None:
            max_row = int(rows[-1] // self._X.shape[1])          # last training DATE
            fit_audit.append({"fit_date": self.market.dates[f],
                              "max_t1_used": self.market.dates[max_row + h],
                              "n_rows": int(len(rows)),
                              "family": self.genome.signal.family,
                              "horizon": h,
                              "embargo_days": self.cfg.WF_EMBARGO_DAYS})
        return True

    # ── scores ─────────────────────────────────────────────────────────────────
    def _scores(self, t: int, tradable: np.ndarray) -> np.ndarray:
        family, p = self.genome.signal.family, self.params
        px = self.close

        if family == "mom_rule":
            skip, look = p["skip"], p["lookback"]
            if t - skip - look < 0:
                return np.full(self.n, np.nan)
            with np.errstate(divide="ignore", invalid="ignore"):
                raw = px[t - skip] / px[t - skip - look] - 1.0
            return _pct_rank(raw, tradable & np.isfinite(raw)) - 0.5

        if family == "meanrev_rule":
            look = p["lookback"]
            if t - look < 0:
                return np.full(self.n, np.nan)
            with np.errstate(divide="ignore", invalid="ignore"):
                raw = -(px[t] / px[t - look] - 1.0)
            score = _pct_rank(raw, tradable & np.isfinite(raw)) - 0.5
            if p["trend_gate"]:
                # Only names above their 200DMA may be bought, only names below may
                # be sold: a reversal bet against the primary trend is the losing half.
                above = px[t] > self._ma[t]
                score = np.where((score > 0) & above, score,
                                 np.where((score < 0) & (px[t] < self._ma[t]), score, 0.0))
            return score

        if family == "seasonal_rule":
            return np.where(tradable, self._seasonal[t], np.nan).astype(np.float64)

        if self._model is None:
            return np.full(self.n, np.nan)              # nothing fitted yet: stay flat
        X = np.asarray(self._X[t], dtype=np.float64)
        keep = tradable & np.isfinite(X).any(axis=1)
        out = np.full(self.n, np.nan)
        if not keep.any():
            return out
        if family == "ridge":
            out[keep] = self._model.predict(X[keep])
        else:
            proba = self._model.predict_proba(X[keep])
            classes = list(self._model.classes_)
            up = proba[:, classes.index(1)] if 1 in classes else 0.0
            dn = proba[:, classes.index(-1)] if -1 in classes else 0.0
            out[keep] = up - dn
        return out

    # ── portfolio construction ─────────────────────────────────────────────────
    def _targets(self, t: int, scores: np.ndarray, tradable: np.ndarray) -> np.ndarray:
        p = self.genome.portfolio
        w = np.zeros(self.n)
        valid = tradable & np.isfinite(scores)
        idx = np.flatnonzero(valid)
        if len(idx) == 0:
            return w
        order = idx[np.argsort(-scores[idx], kind="stable")]
        longs = [j for j in order[:p.n_long] if scores[j] > 0.0]
        taken = set(longs)
        shorts = [j for j in order[::-1][:p.n_short] if scores[j] < 0.0 and j not in taken]
        if not longs and not shorts:
            return w

        raw = self._raw_weights(t, scores, longs + shorts)
        n_l, n_s = len(longs), len(shorts)
        # Split the gross budget by the counts, so net is exactly the long-minus-short
        # tilt the genome asked for: net = gross * (n_l - n_s) / (n_l + n_s).
        for side, names, budget in ((1.0, longs, p.gross * n_l / (n_l + n_s)),
                                    (-1.0, shorts, p.gross * n_s / (n_l + n_s))):
            if not names:
                continue
            r = np.array([raw[j] for j in names], dtype=np.float64)
            total = r.sum()
            r = r / total if total > 0 else np.full(len(names), 1.0 / len(names))
            w[names] = side * budget * r
        return w * self._vol_target_scale()

    def _raw_weights(self, t: int, scores: np.ndarray, names: list) -> dict:
        mode = self.genome.portfolio.weighting
        if mode == "equal":
            return {j: 1.0 for j in names}
        if mode == "score":
            mag = {j: abs(float(scores[j])) for j in names}
            return mag if sum(mag.values()) > 0 else {j: 1.0 for j in names}
        vol = self._vol[t]
        good = [vol[j] for j in names if np.isfinite(vol[j]) and vol[j] > 0]
        fill = float(np.median(good)) if good else 1.0        # unknown vol -> typical vol
        return {j: 1.0 / (vol[j] if np.isfinite(vol[j]) and vol[j] > 0 else fill) for j in names}

    def _vol_target_scale(self) -> float:
        """Scale gross toward the genome's volatility target, using the strategy's
        OWN realised vol. Identity until there is a full window of equity history."""
        target = self.genome.portfolio.vol_target
        if target is None or len(self._equity) <= self.cfg.REALIZED_VOL_DAYS:
            return 1.0
        eq = np.asarray(self._equity[-(self.cfg.REALIZED_VOL_DAYS + 1):], dtype=np.float64)
        r = np.log(eq[1:] / eq[:-1])
        realized = float(np.std(r, ddof=1) * np.sqrt(self.cfg.TRADING_DAYS_YEAR))
        if not np.isfinite(realized) or realized <= 0:
            return 1.0
        return float(np.clip(target / realized, self.cfg.VOL_TARGET_MIN, self.cfg.VOL_TARGET_MAX))

    # ── risk overlays (checked every bar, not only on rebalances) ──────────────
    def _stopped(self, t: int, shares: np.ndarray) -> np.ndarray:
        r = self.genome.risk
        if r.stop_loss is None and r.trail_stop is None:
            return np.zeros(self.n, dtype=bool)
        px = self.close[t]
        long_ = (shares > 0) & np.isfinite(px)
        short = (shares < 0) & np.isfinite(px)
        hit = np.zeros(self.n, dtype=bool)
        if r.stop_loss is not None:
            hit |= long_ & (px <= self._entry * (1.0 - r.stop_loss))
            hit |= short & (px >= self._entry * (1.0 + r.stop_loss))
        if r.trail_stop is not None:
            hit |= long_ & (px <= self._extreme * (1.0 - r.trail_stop))
            hit |= short & (px >= self._extreme * (1.0 + r.trail_stop))
        return hit

    def _overlay_scale(self, t: int, drawdown: float):
        """(gross multiplier, names of the overlays holding it down, "+"-joined).

        A MISSING regime reading is treated as "regime fine" — the safe direction,
        but not a free one, and callers must know it. macro.py fetches ^VIX from
        2000 and vix_pct is a 252-bar trailing rank, so `vix_pct_80` reads NaN for
        every bar before ~2001: on DESIGN's 1997-2001 F0 era that gene is
        structurally inert and scores exactly like `regime_filter=None`. The same
        holds, more briefly, for spy_200dma over its first 200 bars. run_episode
        reports `regime_finite_frac` so evaluation can see it rather than
        mis-attribute a dead gene as a neutral one.
        """
        r = self.genome.risk
        scale, active = 1.0, []

        if self._regime_off(t):
            scale, active = r.regime_scale, ["regime"]

        if r.dd_limit is not None:
            dd = -min(drawdown, 0.0)
            if self._derisked and dd < r.dd_limit * self.cfg.DD_RECOVER_FRAC:
                self._derisked = False
            elif not self._derisked and dd > r.dd_limit:
                self._derisked = True
            if self._derisked:
                scale *= self.cfg.DD_DERISK_SCALE
                active.append("derisk")
        return scale, ("+".join(active) if active else None)

    def _regime_off(self, t: int) -> bool:
        r = self.genome.risk
        if r.regime_filter == "spy_200dma":
            spy, ma = self.close[t, self._spy], self._ma[t, self._spy]
            return bool(np.isfinite(spy) and np.isfinite(ma) and spy < ma)
        if r.regime_filter == "vix_pct_80":
            v = self._vix_pct[t]
            return bool(np.isfinite(v) and v > self.cfg.VIX_PCT_LIMIT)
        return False

    def _regime_finite_frac(self, i0: int, n_steps: int):
        """Fraction of the episode's decision bars on which the genome's regime
        input was actually readable. None when the genome has no regime filter.
        Reported, not acted on — see _overlay_scale."""
        r = self.genome.risk
        if r.regime_filter is None or n_steps <= 0:
            return None
        sl = slice(i0, i0 + n_steps)
        if r.regime_filter == "spy_200dma":
            ok = np.isfinite(self.close[sl, self._spy]) & np.isfinite(self._ma[sl, self._spy])
        else:
            ok = np.isfinite(self._vix_pct[sl])
        return float(np.mean(ok))

    # ── position bookkeeping (entry price / extreme close per spell) ───────────
    def _track_spells(self, t_next: int, before: np.ndarray, after: np.ndarray, fills) -> None:
        fill_px = {}
        for row in fills:
            fill_px.setdefault(row["symbol"], row["fill_px"])
        px = self.close[t_next]
        for j in np.flatnonzero(before != after):
            sym = self.market.symbols[j]
            if after[j] == 0.0:
                self._entry[j] = self._extreme[j] = np.nan
            elif np.sign(before[j]) != np.sign(after[j]):      # a new spell (incl. flips)
                self._entry[j] = fill_px.get(sym, px[j])
                self._extreme[j] = px[j]
        live = after != 0.0
        long_ = live & (after > 0)
        short = live & (after < 0)
        start = live & ~np.isfinite(self._extreme)
        self._extreme = np.where(start, px, self._extreme)
        self._extreme = np.where(long_ & np.isfinite(px), np.fmax(self._extreme, px),
                                 np.where(short & np.isfinite(px),
                                          np.fmin(self._extreme, px), self._extreme))
        self._extreme = np.where(live, self._extreme, np.nan)

    # ── the episode ────────────────────────────────────────────────────────────
    def run_episode(self, env_start=None, env_end=None, decision_log=None, fit_audit=None):
        """Drive a MarketEnv from reset to done and return the daily series.

        Returns dates / daily_net / daily_gross / turnover / costs, each of length
        (bars - 1): one entry per env step, dated by the bar the return was marked
        on. daily_gross is the same path with the day's frictions added back, so
        gross − net is exactly the cost drag. Plus `regime_finite_frac`: how much
        of the episode the genome's regime input was readable at all (None when it
        has no regime filter) — see _overlay_scale for why that is not decoration.

        DECISION-LOG TAGGING IS TRANSITION-ONLY, and Phase 5 must not read it as a
        per-trade overlay flag. A bar is tagged with the overlays that MOVED on it
        ("regime", "derisk", "stop", or a "+"-join like "stop+regime"); trades
        executed on later bars while the book is still held down carry the plain
        "rebalance" tag, because nothing about the overlay changed to cause them.
        To know whether a given trade happened under an overlay, replay the tags
        forward from the last transition — one flag per fill is not what is stored.
        """
        env = MarketEnv(self.market, self.cost, start=env_start, end=env_end,
                        rng=np.random.default_rng(self.cfg.SEED), decision_log=decision_log)
        obs = env.reset()

        self._model = None
        self._equity = [obs["equity"]]
        self._entry = np.full(self.n, np.nan)
        self._extreme = np.full(self.n, np.nan)
        self._derisked = False
        self._target_w = np.zeros(self.n)
        last_fit = None
        last_scale = 1.0

        rebalance_days = self.genome.portfolio.rebalance_days
        refit_days = self.genome.signal.refit_days
        dates, net, gross, turn, cost_path = [], [], [], [], []

        step = 0
        while not env.done:
            t = obs["t"]
            if self.is_model and (last_fit is None or step - last_fit >= refit_days):
                if self._fit(t, fit_audit):
                    last_fit = step

            if step % rebalance_days == 0:
                tradable = np.isfinite(self.close[t]) & (self.close[t] > 0.0)
                self._target_w = self._targets(t, self._scores(t, tradable), tradable)

            hit = self._stopped(t, env.shares)
            if hit.any():
                self._target_w = np.where(hit, 0.0, self._target_w)

            scale, why = self._overlay_scale(t, obs["drawdown"])
            # Every cause acting on THIS bar, "+"-joined ("stop+regime"). A stop
            # must not swallow a coincident overlay transition: the tag is the
            # only record that the overlay moved, and suppressing it here would
            # erase that transition from the log permanently — the later bars see
            # no change to report. "rebalance" is also the right tag between
            # rebalances, where the agent restates its standing target and any
            # fill is drift correction toward it rather than a new decision.
            causes = (["stop"] if hit.any() else [])
            if why is not None and scale != last_scale:
                causes.append(why)
            reason = "+".join(causes) if causes else "rebalance"
            last_scale = scale

            before = env.shares.copy()
            equity_t = obs["equity"]
            obs, _reward, _done, info = env.step(self._target_w * scale, reason=reason)
            self._track_spells(obs["t"], before, env.shares, info["fills"])

            day_cost = (info["commissions"] + info["spread_cost"] + info["slippage"]
                        + info["borrow"] + info["margin"])
            traded = sum(abs(r["shares"] * r["fill_px"]) for r in info["fills"])
            equity_next = obs["equity"]
            self._equity.append(equity_next)
            dates.append(obs["date"])
            net.append(equity_next / equity_t - 1.0)
            gross.append((equity_next + day_cost) / equity_t - 1.0)
            turn.append(traded / equity_t)
            cost_path.append(day_cost)
            step += 1

        return {"dates": pd.DatetimeIndex(dates),
                "daily_net": np.asarray(net, dtype=np.float64),
                "daily_gross": np.asarray(gross, dtype=np.float64),
                "turnover": np.asarray(turn, dtype=np.float64),
                "costs": np.asarray(cost_path, dtype=np.float64),
                "regime_finite_frac": self._regime_finite_frac(env.i0, step)}


def sharpe(returns: np.ndarray) -> float:
    """Annualised Sharpe of a daily return series (excess of zero — the sandbox
    holds no cash sleeve to earn the bill rate)."""
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(config.TRADING_DAYS_YEAR))


if __name__ == "__main__":
    # Phase 2 milestone: one SEED genome per family, full history, pre-vault net
    # Sharpe. These genomes are drawn from the seeded rng, not chosen — no genome
    # here was selected for its result, which is the only reason printing them is
    # honest. Selection is Phase 3+, behind the vault and the gate stack.
    import time
    from dataclasses import replace

    import datafeed
    import genome as gn

    universe = config.import_sibling("universe", config.SIGNAL_LAB)
    syms = datafeed.in_cache(universe.build_universe()[0])
    md = datafeed.load_market(syms, start=config.DATA_START)

    # Top ~60 by average dollar volume over the WHOLE window, counting days a
    # symbol did not trade as zero — so the cut favours names with deep history
    # rather than a recent listing with one huge year.
    liquidity = np.nan_to_num(md.dollar_vol).mean(axis=0)
    keep = [md.symbols[j] for j in np.argsort(-liquidity)[:60]]
    if config.BENCHMARK not in keep:
        keep.append(config.BENCHMARK)
    md = datafeed.load_market(keep, start=config.DATA_START)
    arena_features.build_features(md)

    vault = md.dates.searchsorted(pd.Timestamp(config.VAULT_START))
    print("arena strategy milestone — one seed genome per family")
    print("  market      : %d symbols, %s -> %s (%d bars), hash %s"
          % (len(md.symbols), md.dates[0].date(), md.dates[-1].date(), len(md), md.data_hash))
    print("  features    : %d columns | pre-vault window ends %s (%d bars scored)"
          % (len(md.feature_names), config.VAULT_START, vault))
    print()
    print("  %-13s %-12s %8s %8s %8s %8s %7s %6s  %s"
          % ("family", "hash", "netSR", "grossSR", "return", "maxDD", "turn/d", "secs", "genome"))

    for family in BOUNDS["families"]:
        rng = np.random.default_rng(config.SEED)
        while True:                       # first seeded draw of this family
            g = gn.random_genome(rng, md.feature_names)
            if g.signal.family == family:
                break
        if g.is_model:                    # yearly refits keep the milestone in minutes
            g = replace(g, signal=replace(g.signal, refit_days=252))

        # io-boundary: the elapsed seconds are printed and nothing else — no
        # simulated quantity is derived from them, so the episode stays replayable.
        t0 = time.time()                   # io-boundary
        audit: list = []
        res = StrategyAgent(g, md).run_episode(fit_audit=audit)
        secs = time.time() - t0            # io-boundary

        pre = res["dates"] < pd.Timestamp(config.VAULT_START)
        eq = np.cumprod(1.0 + res["daily_net"][pre])
        dd = float((eq / np.maximum.accumulate(eq) - 1.0).min())
        print("  %-13s %-12s %8.2f %8.2f %+7.1f%% %7.1f%% %6.3f %6.1f  %s"
              % (family, g.hash(), sharpe(res["daily_net"][pre]),
                 sharpe(res["daily_gross"][pre]), 100 * (eq[-1] - 1.0), 100 * dd,
                 float(res["turnover"][pre].mean()), secs, g.describe()[:58]))
        rff = res["regime_finite_frac"]
        if rff is not None:
            print("  %-13s   regime input %-11s readable on %.0f%% of episode bars"
                  % ("", g.risk.regime_filter, 100 * rff))
        if audit:
            worst = max(audit, key=lambda a: a["max_t1_used"])
            print("  %-13s   %d refits, purge holds: last label used %s closed %d bars "
                  "before its fit date %s"
                  % ("", len(audit), worst["max_t1_used"].date(),
                     md.dates.searchsorted(worst["fit_date"])
                     - md.dates.searchsorted(worst["max_t1_used"]),
                     worst["fit_date"].date()))

    print("\n  Pre-vault Sharpes above are UNSELECTED single draws on today's S&P")
    print("  survivors, gross of tax and of any 1990s spread wider than a few bps.")
    print("  Sandbox output is a claim about the past, not a guarantee — and not")
    print("  financial advice.")
