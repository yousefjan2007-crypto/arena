"""
The strategy genome: what a candidate IS, and the operators that breed new ones.

A genome is a complete strategy — signal, portfolio construction, risk overlays —
in three frozen blocks. Nothing else in arena decides how to trade: strategy.py
reads a genome and does what it says, so the search space is exactly what BOUNDS
below declares and a result is always attributable to twelve characters.

    genome.hash()  = sha256(canonical JSON)[:12]      the identity everywhere:
                     artifacts/genomes/<hash12>/, the trial ledger, the champion
                     pointer, lineage. Two genomes that trade identically MUST
                     hash identically, so encode -> decode -> hash is identity
                     (verify.py test 9) and the field order in the JSON is fixed
                     by sort_keys, never by declaration order.

BOUNDS is the whole space, in one dict, so evolution can enumerate it and
verify.py can assert against it. Every grid is an ORDERED tuple: mutation's
"jitter to a grid neighbour" is one step along that order, which is a true
neighbour for the numeric grids and a deterministic local move for the
categorical ones.

INVARIANTS held by construction and re-imposed after every operator (`_repair`):
  • n_long + n_short >= BOUNDS["min_positions"] — a book of one or two names is a
    coin flip, not a strategy, and it makes the cross-sectional scores meaningless.
  • `features` is sorted, deduplicated, 3-15 entries, drawn from the live feature
    library — and EMPTY for the three rule families, which do not fit anything.
    Sorted because the hash must not depend on the order a subset was drawn in.
  • every param is a member of its family's grid.

DETERMINISM: `child_rng(SEED, generation, parent_hash, child_idx)` derives an
offspring's stream from a sha256 of those four values. Python's built-in hash()
is salted per process and would silently break reproducibility across runs.

Inert genes are allowed on purpose: `regime_scale` does nothing while
`regime_filter` is None, so two such genomes hash differently but trade the same.
Collapsing them would force a scale on every filter that switches on. The cost is
a slightly inflated trial count, which deflates DSR — an error in the honest
direction.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace

import numpy as np

import config                       # FIRST: puts the siblings on sys.path

# ── the space ──────────────────────────────────────────────────────────────────
BOUNDS = {
    "families": ("mom_rule", "meanrev_rule", "seasonal_rule", "ridge", "logistic", "hgb"),
    # The families that fit a model, and therefore the only ones that carry features.
    "model_families": ("ridge", "logistic", "hgb"),
    "horizon": (5, 10, 21, 63),                  # trading days the signal predicts
    "refit_days": (63, 126, 252),                # walk-forward refit cadence
    "n_features": (3, 15),                       # inclusive, model families only
    "min_positions": 3,                          # n_long + n_short floor
    # Family parameter grids. Names are the sklearn kwargs for the model families,
    # so strategy.py can pass them straight through with no translation table.
    "params": {
        "mom_rule": {"lookback": (21, 63, 126, 252), "skip": (0, 5, 21)},
        "meanrev_rule": {"lookback": (5, 10, 21), "trend_gate": (False, True)},
        "seasonal_rule": {},                     # the panel's seasonal column IS the score
        "ridge": {"alpha": (0.1, 1.0, 10.0, 100.0)},
        "logistic": {"C": (0.01, 0.1, 1.0, 10.0)},
        "hgb": {"learning_rate": (0.03, 0.05, 0.1), "max_depth": (2, 3, 4),
                "max_iter": (100, 150, 200, 250, 300),
                "min_samples_leaf": (100, 200, 300, 400)},
    },
    "portfolio": {
        "n_long": tuple(range(0, 13)),
        "n_short": tuple(range(0, 9)),
        "weighting": ("equal", "score", "inv_vol"),
        "gross": (0.6, 0.8, 1.0, 1.3),
        "vol_target": (None, 0.10, 0.15, 0.20),
        "rebalance_days": (1, 5, 21),
    },
    "risk": {
        "stop_loss": (None, 0.05, 0.10, 0.15),
        "trail_stop": (None, 0.10, 0.20),
        "regime_filter": (None, "spy_200dma", "vix_pct_80"),
        "regime_scale": (0.0, 0.5),              # gross multiplier while the filter is on
        "dd_limit": (None, 0.10, 0.15, 0.20),
    },
}


# ── gene blocks ────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SignalGene:
    """What produces the cross-sectional score. `params` is a sorted tuple of
    (name, value) pairs rather than a dict so the block stays immutable and
    canonically ordered — dicts are neither."""

    family: str
    horizon: int
    refit_days: int
    features: tuple = ()
    params: tuple = ()

    @property
    def pdict(self) -> dict:
        return dict(self.params)


@dataclass(frozen=True)
class PortfolioGene:
    n_long: int
    n_short: int
    weighting: str
    gross: float
    vol_target: object          # float or None
    rebalance_days: int


@dataclass(frozen=True)
class RiskGene:
    stop_loss: object           # float or None
    trail_stop: object          # float or None
    regime_filter: object       # str or None
    regime_scale: float
    dd_limit: object            # float or None


@dataclass(frozen=True)
class Genome:
    signal: SignalGene
    portfolio: PortfolioGene
    risk: RiskGene

    def to_dict(self) -> dict:
        return to_dict(self)

    def canonical_json(self) -> str:
        return canonical_json(self)

    def hash(self) -> str:
        return hashlib.sha256(canonical_json(self).encode()).hexdigest()[:12]

    @property
    def is_model(self) -> bool:
        return self.signal.family in BOUNDS["model_families"]

    def describe(self) -> str:
        s, p, r = self.signal, self.portfolio, self.risk
        bits = ["%s h%d" % (s.family, s.horizon)]
        if self.is_model:
            bits.append("refit %dd, %d features" % (s.refit_days, len(s.features)))
        if s.params:
            bits.append(" ".join("%s=%s" % kv for kv in s.params))
        bits.append("%dL/%dS %s gross %.1f%s rebal %dd"
                    % (p.n_long, p.n_short, p.weighting, p.gross,
                       "" if p.vol_target is None else " volt %.2f" % p.vol_target,
                       p.rebalance_days))
        risk = [n for n, v in (("stop %s" % r.stop_loss, r.stop_loss),
                               ("trail %s" % r.trail_stop, r.trail_stop),
                               ("%s x%.1f" % (r.regime_filter, r.regime_scale), r.regime_filter),
                               ("dd %s" % r.dd_limit, r.dd_limit)) if v is not None]
        bits.append(", ".join(risk) if risk else "no overlays")
        return " | ".join(bits)


# ── canonical encoding ─────────────────────────────────────────────────────────
def to_dict(g: Genome) -> dict:
    """Plain JSON-able types only: tuples become lists, params become a mapping."""
    return {
        "signal": {"family": g.signal.family, "horizon": g.signal.horizon,
                   "refit_days": g.signal.refit_days,
                   "features": list(g.signal.features),
                   "params": dict(g.signal.params)},
        "portfolio": {"n_long": g.portfolio.n_long, "n_short": g.portfolio.n_short,
                      "weighting": g.portfolio.weighting, "gross": g.portfolio.gross,
                      "vol_target": g.portfolio.vol_target,
                      "rebalance_days": g.portfolio.rebalance_days},
        "risk": {"stop_loss": g.risk.stop_loss, "trail_stop": g.risk.trail_stop,
                 "regime_filter": g.risk.regime_filter, "regime_scale": g.risk.regime_scale,
                 "dd_limit": g.risk.dd_limit},
    }


def from_dict(d: dict) -> Genome:
    s, p, r = d["signal"], d["portfolio"], d["risk"]
    return Genome(
        signal=SignalGene(family=s["family"], horizon=int(s["horizon"]),
                          refit_days=int(s["refit_days"]),
                          features=tuple(s["features"]),
                          params=tuple(sorted(s["params"].items()))),
        portfolio=PortfolioGene(n_long=int(p["n_long"]), n_short=int(p["n_short"]),
                                weighting=p["weighting"], gross=p["gross"],
                                vol_target=p["vol_target"],
                                rebalance_days=int(p["rebalance_days"])),
        risk=RiskGene(stop_loss=r["stop_loss"], trail_stop=r["trail_stop"],
                      regime_filter=r["regime_filter"], regime_scale=r["regime_scale"],
                      dd_limit=r["dd_limit"]))


def canonical_json(g: Genome) -> str:
    """sort_keys makes the encoding independent of declaration order; the compact
    separators keep the hashed bytes free of formatting choices."""
    return json.dumps(to_dict(g), sort_keys=True, separators=(",", ":"))


def stable_hash(*parts) -> int:
    """A 64-bit hash that is the same in every process, unlike Python's hash()."""
    payload = "|".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def child_rng(seed: int, generation: int, parent_hash: str, child_idx: int):
    """The offspring's own random stream, reproducible from its coordinates."""
    return np.random.default_rng(stable_hash(seed, generation, parent_hash, child_idx))


# ── repair (every operator ends here) ──────────────────────────────────────────
def _pick(rng, grid):
    return grid[int(rng.integers(len(grid)))]


def _snap(value, grid):
    """Nearest grid member. Numeric grids snap by distance; anything else falls
    back to the declared default (the first entry) when the value is unknown."""
    if value in grid:
        return value
    numeric = [v for v in grid if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if isinstance(value, (int, float)) and not isinstance(value, bool) and numeric:
        return min(numeric, key=lambda v: abs(v - value))
    return grid[0]


def _repair_features(family, features, feature_names):
    if family not in BOUNDS["model_families"]:
        return ()                                   # rules fit nothing
    lo, hi = BOUNDS["n_features"]
    known = tuple(feature_names) if feature_names is not None else None
    keep = sorted(set(features) if known is None else {f for f in features if f in known})
    if known is not None and len(keep) < lo:
        # Deterministic top-up: the first unused names in canonical order. Only
        # reachable when an operator hands over a subset the library cannot serve.
        keep = sorted(set(keep) | set([f for f in known if f not in keep][:lo - len(keep)]))
    return tuple(keep[:hi])


def _repair(g: Genome, feature_names=None) -> Genome:
    s, p, r = g.signal, g.portfolio, g.risk

    family = s.family if s.family in BOUNDS["families"] else BOUNDS["families"][0]
    grids = BOUNDS["params"][family]
    params = dict(s.params)
    params = {k: _snap(params[k], grids[k]) for k in grids if k in params}
    for k in grids:                                  # a family hop can leave a gap
        if k not in params:
            params[k] = grids[k][len(grids[k]) // 2]
    signal = SignalGene(family=family,
                        horizon=_snap(s.horizon, BOUNDS["horizon"]),
                        refit_days=_snap(s.refit_days, BOUNDS["refit_days"]),
                        features=_repair_features(family, s.features, feature_names),
                        params=tuple(sorted(params.items())))

    pg = BOUNDS["portfolio"]
    n_long = int(np.clip(p.n_long, pg["n_long"][0], pg["n_long"][-1]))
    n_short = int(np.clip(p.n_short, pg["n_short"][0], pg["n_short"][-1]))
    # Grow the side that already dominates, so repairing a short-tilted genome
    # does not quietly turn it long.
    while n_long + n_short < BOUNDS["min_positions"]:
        if n_short > n_long and n_short < pg["n_short"][-1]:
            n_short += 1
        elif n_long < pg["n_long"][-1]:
            n_long += 1
        else:
            n_short += 1
    portfolio = PortfolioGene(n_long=n_long, n_short=n_short,
                              weighting=_snap(p.weighting, pg["weighting"]),
                              gross=_snap(p.gross, pg["gross"]),
                              vol_target=_snap(p.vol_target, pg["vol_target"]),
                              rebalance_days=_snap(p.rebalance_days, pg["rebalance_days"]))

    rg = BOUNDS["risk"]
    risk = RiskGene(stop_loss=_snap(r.stop_loss, rg["stop_loss"]),
                    trail_stop=_snap(r.trail_stop, rg["trail_stop"]),
                    regime_filter=_snap(r.regime_filter, rg["regime_filter"]),
                    regime_scale=_snap(r.regime_scale, rg["regime_scale"]),
                    dd_limit=_snap(r.dd_limit, rg["dd_limit"]))
    return Genome(signal=signal, portfolio=portfolio, risk=risk)


# ── operators ──────────────────────────────────────────────────────────────────
def _random_features(rng, family, feature_names):
    if family not in BOUNDS["model_families"]:
        return ()
    lo, hi = BOUNDS["n_features"]
    names = list(feature_names)
    k = int(rng.integers(lo, min(hi, len(names)) + 1))
    return tuple(sorted(names[i] for i in rng.choice(len(names), size=k, replace=False)))


def _random_params(rng, family):
    return tuple(sorted((k, _pick(rng, grid)) for k, grid in BOUNDS["params"][family].items()))


def _random_signal(rng, feature_names, family=None):
    family = _pick(rng, BOUNDS["families"]) if family is None else family
    return SignalGene(family=family,
                      horizon=_pick(rng, BOUNDS["horizon"]),
                      refit_days=_pick(rng, BOUNDS["refit_days"]),
                      features=_random_features(rng, family, feature_names),
                      params=_random_params(rng, family))


def _random_portfolio(rng):
    pg = BOUNDS["portfolio"]
    return PortfolioGene(n_long=_pick(rng, pg["n_long"]), n_short=_pick(rng, pg["n_short"]),
                         weighting=_pick(rng, pg["weighting"]), gross=_pick(rng, pg["gross"]),
                         vol_target=_pick(rng, pg["vol_target"]),
                         rebalance_days=_pick(rng, pg["rebalance_days"]))


def _random_risk(rng):
    rg = BOUNDS["risk"]
    return RiskGene(stop_loss=_pick(rng, rg["stop_loss"]), trail_stop=_pick(rng, rg["trail_stop"]),
                    regime_filter=_pick(rng, rg["regime_filter"]),
                    regime_scale=_pick(rng, rg["regime_scale"]),
                    dd_limit=_pick(rng, rg["dd_limit"]))


def random_genome(rng, feature_names, family=None) -> Genome:
    """A uniformly drawn member of BOUNDS — the immigrant operator, and the seed
    population. `family` forces the signal family (stratified immigrants)."""
    return _repair(Genome(signal=_random_signal(rng, feature_names, family=family),
                          portfolio=_random_portfolio(rng),
                          risk=_random_risk(rng)), feature_names)


def _tunables(g: Genome) -> list:
    """Every single scalar a jitter could step, as (block, field, grid)."""
    out = [("signal", "horizon", BOUNDS["horizon"]),
           ("signal", "refit_days", BOUNDS["refit_days"])]
    for name, grid in sorted(BOUNDS["params"][g.signal.family].items()):
        out.append(("param", name, grid))
    for name, grid in sorted(BOUNDS["portfolio"].items()):
        out.append(("portfolio", name, grid))
    for name, grid in sorted(BOUNDS["risk"].items()):
        out.append(("risk", name, grid))
    return out


def _jitter_param(g: Genome, rng) -> Genome:
    """Move ONE scalar one step along its grid — the local search move."""
    tunables = _tunables(g)
    block, name, grid = tunables[int(rng.integers(len(tunables)))]
    current = (g.signal.pdict.get(name) if block == "param"
               else getattr(getattr(g, block), name))
    i = grid.index(current) if current in grid else len(grid) // 2
    step = 1 if (i == 0) else (-1 if i == len(grid) - 1 else int(rng.choice([-1, 1])))
    value = grid[i + step]
    if block == "param":
        params = dict(g.signal.params)
        params[name] = value
        return replace(g, signal=replace(g.signal, params=tuple(sorted(params.items()))))
    if block == "signal":
        return replace(g, signal=replace(g.signal, **{name: value}))
    return replace(g, **{block: replace(getattr(g, block), **{name: value})})


def _mutate_features(g: Genome, rng, feature_names) -> Genome:
    """Add / drop / swap one feature. No-op for the rule families, which have none."""
    if not g.is_model:
        return g
    lo, hi = BOUNDS["n_features"]
    have = list(g.signal.features)
    pool = [f for f in feature_names if f not in have]
    moves = (["add"] if len(have) < hi and pool else []) + \
            (["drop"] if len(have) > lo else []) + (["swap"] if pool else [])
    if not moves:
        return g
    move = moves[int(rng.integers(len(moves)))]
    if move == "add":
        have.append(pool[int(rng.integers(len(pool)))])
    elif move == "drop":
        have.pop(int(rng.integers(len(have))))
    else:
        have[int(rng.integers(len(have)))] = pool[int(rng.integers(len(pool)))]
    return replace(g, signal=replace(g.signal, features=tuple(sorted(set(have)))))


def _resample_block(g: Genome, rng, feature_names) -> Genome:
    """Redraw one whole block. The signal block keeps its family — swapping the
    family is a separate, rarer move."""
    which = int(rng.integers(3))
    if which == 0:
        return replace(g, signal=_random_signal(rng, feature_names, family=g.signal.family))
    if which == 1:
        return replace(g, portfolio=_random_portfolio(rng))
    return replace(g, risk=_random_risk(rng))


def _family_hop(g: Genome, rng, feature_names) -> Genome:
    """Jump to another family, keeping horizon and cadence; params and features
    must be redrawn because they mean nothing outside their own family."""
    others = [f for f in BOUNDS["families"] if f != g.signal.family]
    family = others[int(rng.integers(len(others)))]
    return replace(g, signal=SignalGene(
        family=family, horizon=g.signal.horizon, refit_days=g.signal.refit_days,
        features=_random_features(rng, family, feature_names),
        params=_random_params(rng, family)))


def _apply(fns, genome: Genome, rng, feature_names) -> Genome:
    out = genome
    for fn in fns:
        out = fn(out, rng) if fn is _jitter_param else fn(out, rng, feature_names)
    return _repair(out, feature_names)


def mutate(genome: Genome, rng, feature_names) -> Genome:
    """The four DESIGN moves, each an independent draw at its config probability.

    They are independent (the probabilities sum past 1), so a child can carry two
    moves — intended, not a bug. What is NOT intended is a child identical to its
    parent, and there are two ways to get one: ~23% of draws fire nothing at all,
    and the feature move is a no-op on the rule families, which carry no features.
    Re-evaluating a clone burns a population slot and inflates the trial count
    (which deflates DSR for nothing), so a mutation that changed nothing is
    retried with one forced move, weighted the same way, then with plain param
    jitters. Even a jitter can be undone: a genome sitting exactly on the
    n_long+n_short floor has a step down that position count repaired straight
    back. Hence a bounded retry rather than one guaranteed move.
    """
    moves = ((config.P_MUT_PARAM, _jitter_param),
             (config.P_MUT_FEATURE, _mutate_features),
             (config.P_MUT_BLOCK, _resample_block),
             (config.P_MUT_FAMILY, _family_hop))
    out = _apply([fn for p, fn in moves if rng.random() < p], genome, rng, feature_names)
    if out != genome:
        return out
    weights = np.array([p for p, _ in moves], dtype=float)
    forced = moves[int(rng.choice(len(moves), p=weights / weights.sum()))][1]
    out = _apply([forced], genome, rng, feature_names)
    for _ in range(config.MUT_MAX_TRIES):
        if out != genome:
            break
        out = _apply([_jitter_param], genome, rng, feature_names)
    return out


def crossover(a: Genome, b: Genome, rng) -> Genome:
    """Uniform crossover at the gene-BLOCK level: a whole signal/portfolio/risk
    block is inherited intact. Blending inside a block (half of one signal, half
    of another) would mostly produce incoherent strategies.

    A uniform draw copies all three blocks from one parent 2 times in 8. The
    operator does not fake diversity by forcing a swap — evolution.py sees the
    duplicate hash and can cull it, which is the honest place to handle it.
    """
    child = Genome(signal=a.signal if rng.random() < 0.5 else b.signal,
                   portfolio=a.portfolio if rng.random() < 0.5 else b.portfolio,
                   risk=a.risk if rng.random() < 0.5 else b.risk)
    return _repair(child)


if __name__ == "__main__":
    # DEMO feature names — a real run passes market.feature_names from features.py.
    DEMO = ("bb_z", "credit_mom21", "ema50_200", "mom_12_1", "mom_21", "mom_63",
            "mom_126", "px_ema200", "rev_5", "rsi", "rs_spy", "rv21", "seasonality",
            "us10y", "vix_pct", "vol_regime", "xs_mom_63", "xs_rs_spy")

    rng = np.random.default_rng(config.SEED)
    g = random_genome(rng, DEMO)
    print("arena genome")
    print("  random   %s  %s" % (g.hash(), g.describe()))
    if g.is_model:
        print("           features: %s" % ", ".join(g.signal.features))
    m = mutate(g, rng, DEMO)
    print("  mutant   %s  %s" % (m.hash(), m.describe()))
    other = random_genome(rng, DEMO)
    c = crossover(g, other, rng)
    src = "".join("A" if getattr(c, b) == getattr(g, b) else "B"
                  for b in ("signal", "portfolio", "risk"))
    print("  parent B %s  %s" % (other.hash(), other.describe()))
    print("  crossed  %s  %s" % (c.hash(), c.describe()))
    print("           blocks signal/portfolio/risk from parent %s" % "/".join(src))

    print("  json     %s" % g.canonical_json()[:96] + " ...")
    print("  round-trip identity: %s" % (from_dict(g.to_dict()).hash() == g.hash()))
    a1 = child_rng(config.SEED, 3, g.hash(), 7).random()
    a2 = child_rng(config.SEED, 3, g.hash(), 7).random()
    b1 = child_rng(config.SEED, 3, g.hash(), 8).random()
    print("  child_rng reproducible: %s (idx 7 -> %.6f twice, idx 8 -> %.6f)"
          % (a1 == a2 and a1 != b1, a1, b1))
