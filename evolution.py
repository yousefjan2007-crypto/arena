"""
The breeding step: how one evaluated population becomes the next one.

    next_generation(pop_entries, scores, generation, feature_names) -> [entry]
    update_hall_of_fame(hof, f1_results, generation)               -> [record]

EVERY ENTRY CARRIES ITS LINEAGE. A population is not a bag of genomes, it is a
list of dicts::

    {"genome": <genome.to_dict()>, "hash": "0f2c...", "op": "elite|mutate|
     crossover|immigrant|seed", "parent_hash": "" | "abc..." | ["abc","def"],
     "birth_gen": 3}

so that a champion three phases from now can be traced back to the random draw it
descended from, without joining anything. `op` describes how the entry entered
THIS generation, not how the genome was first created: an immigrant that survives
is an "elite" the next generation, and its birth op is recoverable from its
`birth_gen` row in the trial ledger.

WHERE THE SELECTION PRESSURE COMES FROM, AND ITS ONE HONEST WART. `scores` maps
genome hash -> the score the genome was last measured at, which is its F1 score
if it reached F1 and its F0 screen score otherwise (run_generation only sends the
top SCREEN_FRAC to F1). Elitism therefore ranks F1 numbers against F0 numbers,
and the two are NOT the same measurement — F0 trades three five-year eras on a
60-symbol universe at a forced weekly cadence, F1 trades the full history on the
genome's own cadence. They share units (Sharpe minus the parsimony tax) and
nothing else. Two consequences, both accepted deliberately:

  • an elite is always an F1 genome in practice, because F1 is a harder test and
    a genome that never reached it was already ranked below half its generation;
  • a tournament can still pit an F0 score against an F1 one. That is the point:
    selection pressure exists at BOTH fidelities, so a genome the screen liked is
    allowed to breed even in the generation it was not fully measured in.

DETERMINISM IS THE WHOLE CONTRACT OF THIS FILE. Every stochastic draw comes from
`genome.child_rng(SEED, generation, tag, slot)` — a stream derived from the
slot's own coordinates. There is deliberately NO shared mutable rng: with one,
the genome bred into slot 7 would depend on how many draws slots 0-6 happened to
make, so adding a family that mutates differently, or reordering the loop, would
silently change every later slot. Per-slot streams make the whole generation a
pure function of (SEED, generation, parent hashes, scores) — which is what
verify.py test 2 proves by running two generations twice and diffing the bytes.

DEDUP. A population with the same genome twice pays twice to learn one thing, and
inflates the trial count (which deflates DSR) for nothing. Crossover is the main
source — a uniform block draw copies one parent outright 2 times in 8 — so a
colliding child is re-mutated up to config.DEDUP_MAX_TRIES times and, if it is
still a duplicate, replaced by a fresh immigrant. The re-mutation keeps the
slot's `op` and lineage: it was still bred by that operator from that parent, it
was just pushed off a hash that was already taken.
"""
from __future__ import annotations

import json
import os

import config                       # FIRST: puts the siblings on sys.path
import genome as gn

HOF_FILE = "hall_of_fame.json"

# Both JSON files this module and run_generation.py rewrite say so in their own
# header, because state/ is otherwise append-only and a reader is entitled to know
# which files are records and which are caches.
POPULATION_NOTE = ("REWRITTEN every generation: this is the generation POINTER, "
                   "not a record. Every genome in it is in the trial ledger, which "
                   "is append-only; this file is regenerable from that ledger.")
HOF_NOTE = ("REWRITTEN every generation: a ranked CACHE of the best pre-vault "
            "Sharpes on record, regenerable from the trial ledger and the returns "
            "artifacts. Scores are the LATEST evaluation of each genome, not its "
            "best ever — see evolution.update_hall_of_fame.")


# ── entries ────────────────────────────────────────────────────────────────────
def make_entry(genome, op: str, parent_hash="", birth_gen: int = 0) -> dict:
    """One population slot, with everything needed to explain where it came from."""
    return {"genome": genome.to_dict(), "hash": genome.hash(), "op": op,
            "parent_hash": parent_hash, "birth_gen": int(birth_gen)}


def entry_genome(entry: dict):
    return gn.from_dict(entry["genome"])


def parent_field(parent_hash) -> str:
    """The ledger's single-column rendering of a lineage: a crossover's two parents
    join with '+' so one CSV cell can still hold the whole story."""
    if isinstance(parent_hash, (list, tuple)):
        return "+".join(str(p) for p in parent_hash)
    return str(parent_hash or "")


def op_counts(entries) -> dict:
    out = {}
    for e in entries:
        out[e["op"]] = out.get(e["op"], 0) + 1
    return out


# ── slot arithmetic ────────────────────────────────────────────────────────────
def slot_counts(n_pop: int) -> tuple:
    """(elites, immigrants, offspring) for a population of `n_pop`.

    THE POPULATION KEEPS ITS OWN SIZE. config.POP_SIZE is the size a population is
    SEEDED at; after that the incoming population's length is the target, so a
    12-genome arena stays 12 genomes instead of quintupling its own cost between
    one generation and the next.

    ELITE_N and IMMIGRANT_N are absolute counts tuned for POP_SIZE=64, where they
    reserve 8 slots of 64. Applied literally to a 12-genome test run they reserve 8
    of 12, and to an 8-genome one they reserve all of it — a "generation" that
    breeds nothing at all. So the two carried groups are capped together at
    config.EVOLVE_MAX_CARRY_FRAC of the population and scaled down proportionally
    when they exceed it. At POP_SIZE this is a no-op (8 <= 32); it only ever binds
    on the small runs, and it binds the same way in both of a determinism pair-run.
    """
    if n_pop <= 0:
        return 0, 0, 0
    n_elite = min(config.ELITE_N, n_pop)
    n_imm = min(config.IMMIGRANT_N, n_pop - n_elite)
    cap = int(n_pop * config.EVOLVE_MAX_CARRY_FRAC)
    if n_elite + n_imm > cap:
        share = cap / float(n_elite + n_imm)
        # +0.5 rather than round(): banker's rounding would send a .5 to the even
        # number, which is a coin flip nobody asked for inside a determinism proof.
        n_elite = int(n_elite * share + 0.5)
        n_imm = int(n_imm * share + 0.5)
    n_elite = min(max(1, n_elite), n_pop)          # never lose the best genome
    n_imm = max(0, min(n_imm, n_pop - n_elite))
    return n_elite, n_imm, n_pop - n_elite - n_imm


# ── selection ──────────────────────────────────────────────────────────────────
def _score_of(entry: dict, scores: dict) -> float:
    """A genome with no score at all cannot win anything.

    -inf, not 0.0: a genome whose evaluation never landed (a killed run, a family
    that raised) must not be able to out-rank a measured genome that merely scored
    badly. It can still be drawn into a tournament and lose there, which is the
    correct amount of influence for "no evidence".
    """
    v = scores.get(entry["hash"])
    return float(v) if v is not None else float("-inf")


def _rank(entries, scores) -> list:
    """Best first. Ties break on the hash so the order is a function of the
    population, never of the order the workers happened to finish in."""
    return sorted(entries, key=lambda e: (-_score_of(e, scores), e["hash"]))


def tournament(entries, scores, rng):
    """Draw config.TOURNAMENT_K entries WITH replacement, return the best.

    With replacement because sampling without it turns into "rank the whole
    population" as K approaches the population size — and on the small runs it
    would guarantee the best genome is in every tournament, which is elitism
    wearing a tournament's clothes.
    """
    k = max(1, min(config.TOURNAMENT_K, len(entries)))
    drawn = [entries[int(i)] for i in rng.integers(len(entries), size=k)]
    return _rank(drawn, scores)[0]


# ── the generation step ────────────────────────────────────────────────────────
def next_generation(pop_entries, scores, generation: int, feature_names) -> list:
    """The population for `generation + 1`.

    `pop_entries` is the population that was just evaluated (population.json's
    entries) and `scores` maps genome hash -> its score at the best fidelity it
    reached this generation. Nothing here touches disk, the clock, or the market:
    given the same three arguments it returns the same list, on any machine.

    Composition (see slot_counts for how the counts are derived):
      elites      the top scorers, carried VERBATIM — same genome, same hash, and
                  their original parent_hash/birth_gen, because surviving does not
                  re-birth a genome. They are re-evaluated next generation anyway;
                  a stored score is never what promotes anything.
      offspring   each slot is a crossover of two tournament parents with
                  probability config.CROSSOVER_FRAC, otherwise a mutant of one.
      immigrants  fresh uniform draws from BOUNDS — the diversity floor that keeps
                  a converged population from searching a single ridge forever.
    """
    entries = list(pop_entries)
    if not entries:
        raise ValueError("next_generation needs a non-empty population")
    n_elite, n_imm, n_off = slot_counts(len(entries))
    birth_gen = int(generation) + 1

    # Distinct hashes: a population that somehow holds one genome twice must not
    # spend two elite slots carrying the same strategy into the next generation.
    ranked, out, seen = _rank(entries, scores), [], set()
    for e in ranked:
        if len(out) >= n_elite:
            break
        if e["hash"] not in seen:
            out.append(dict(e, op="elite"))
            seen.add(e["hash"])

    for slot in range(n_off):
        # Two streams per slot, both derived from the slot's own coordinates: one
        # to CHOOSE (parents, and which operator), one to BREED. Splitting them
        # keeps the choice reproducible even if an operator's internal number of
        # draws changes.
        pick = gn.child_rng(config.SEED, generation, "tournament", slot)
        p1 = tournament(entries, scores, pick)
        use_crossover = pick.random() < config.CROSSOVER_FRAC
        p2 = tournament(entries, scores, pick) if use_crossover else None

        rng = gn.child_rng(config.SEED, generation, p1["hash"], slot)
        if use_crossover:
            child = gn.crossover(entry_genome(p1), entry_genome(p2), rng)
            op, parent = "crossover", [p1["hash"], p2["hash"]]
        else:
            child = gn.mutate(entry_genome(p1), rng, feature_names)
            op, parent = "mutate", p1["hash"]

        for _ in range(config.DEDUP_MAX_TRIES):
            if child.hash() not in seen:
                break
            # Re-mutate the CHILD, not the parent: the collision says this exact
            # point in the space is already taken, and mutating away from it is a
            # shorter walk than re-drawing the parent's neighbourhood.
            child = gn.mutate(child, rng, feature_names)
        if child.hash() in seen:
            # Ten mutations could not escape: the neighbourhood is saturated. A
            # fresh immigrant is a slot spent on new information rather than on
            # re-evaluating a genome the population already has.
            child = _fresh(generation, "dedup", slot, seen, feature_names)
            op, parent = "immigrant", ""

        out.append(make_entry(child, op, parent, birth_gen))
        seen.add(child.hash())

    for slot in range(n_imm):
        g = _fresh(generation, "imm", slot, seen, feature_names)
        out.append(make_entry(g, "immigrant", "", birth_gen))
        seen.add(g.hash())
    return out


def _fresh(generation: int, tag: str, slot: int, seen: set, feature_names):
    """A uniform draw from BOUNDS that is not already in the new population.

    Retries advance the SAME per-slot stream rather than reseeding, so the draw
    stays a function of (SEED, generation, tag, slot) and of how full the
    population already was — both of which are identical in a re-run. A collision
    that survives DEDUP_MAX_TRIES is accepted: at that point the duplicate costs
    one wasted evaluation (the ledger is idempotent, so it costs nothing else) and
    looping forever would cost the generation.
    """
    rng = gn.child_rng(config.SEED, generation, tag, slot)
    g = gn.random_genome(rng, feature_names)
    for _ in range(config.DEDUP_MAX_TRIES):
        if g.hash() not in seen:
            break
        g = gn.random_genome(rng, feature_names)
    return g


# ── hall of fame ───────────────────────────────────────────────────────────────
def hof_path(state_dir=None) -> str:
    return os.path.join(state_dir or config.STATE_DIR, HOF_FILE)


def load_hall_of_fame(state_dir=None) -> list:
    path = hof_path(state_dir)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f).get("hall_of_fame", [])


def save_hall_of_fame(hof, state_dir=None) -> str:
    path = hof_path(state_dir)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"note": HOF_NOTE, "size": len(hof), "hall_of_fame": hof},
                  f, indent=1, sort_keys=True)
    os.replace(tmp, path)              # never leave a half-written pointer
    return path


def update_hall_of_fame(hof, f1_results, generation: int) -> list:
    """Top config.HOF_SIZE genomes all-time by PRE-VAULT net Sharpe.

    `f1_results` is an iterable of (population entry, full_eval result) pairs — the
    genomes that reached F1 this generation. Pairs rather than a hash -> result
    mapping because the hall of fame stores lineage, and a result dict does not
    know its own parents.

    THE STORED SCORE IS THE LATEST ONE, NOT THE BEST ONE. A genome that is
    re-evaluated (every elite is, every generation) has its record REPLACED, and
    `generation` records when that happened. The alternative — keeping each
    genome's best-ever Sharpe — lets one lucky evaluation squat in the hall
    forever, and re-evaluating on more data is exactly how a stale flattering
    number gets found out. Ranking a max-over-evaluations would also be a second
    hidden selection on top of the one the ledger already counts.

    The hall is a CACHE (see HOF_NOTE): dropping it and rebuilding it from the
    ledger loses nothing but the genome bodies, which live in the ledger's
    generation + hash and in the population files.
    """
    records = {r["hash"]: dict(r) for r in hof}
    for entry, res in (f1_results.items() if hasattr(f1_results, "items") else f1_results):
        records[entry["hash"]] = {
            "hash": entry["hash"],
            "genome": entry["genome"],
            "op": entry["op"],
            "parent_hash": entry["parent_hash"],
            "birth_gen": int(entry["birth_gen"]),
            "family": entry["genome"]["signal"]["family"],
            "sharpe_prevault": float(res["sharpe_prevault"]),
            "score": float(res["score"]),
            "n_days_prevault": int(res.get("n_days_prevault", 0)),
            "generation": int(generation),          # last evaluated, not first seen
        }
    ranked = sorted(records.values(),
                    key=lambda r: (-float(r["sharpe_prevault"]), r["hash"]))
    return ranked[:config.HOF_SIZE]


if __name__ == "__main__":
    # Smoke test: breed a 16-genome population twice from identical inputs and
    # prove the two are the same population. No disk, no market, no network.
    import numpy as np

    DEMO = ("bb_z", "credit_mom21", "ema50_200", "mom_12_1", "mom_21", "mom_63",
            "mom_126", "px_ema200", "rev_5", "rsi", "rs_spy", "rv21", "seasonality",
            "us10y", "vix_pct", "vol_regime", "xs_mom_63", "xs_rs_spy")
    GEN = 3

    seed_pop = [make_entry(gn.random_genome(gn.child_rng(config.SEED, 0, "genesis", i), DEMO),
                           "seed", "", 0) for i in range(16)]
    # Fake scores: a fixed function of the hash, so the ranking is arbitrary but
    # identical in both runs — the rng is what is under test here, not the market.
    scores = {e["hash"]: (int(e["hash"][:6], 16) % 1000) / 500.0 - 1.0 for e in seed_pop}

    print("arena evolution")
    ne, ni, no = slot_counts(len(seed_pop))
    print("  slots      : %d elites + %d offspring + %d immigrants = %d  "
          "(ELITE_N=%d IMMIGRANT_N=%d, cap %.0f%% of the population)"
          % (ne, no, ni, len(seed_pop), config.ELITE_N, config.IMMIGRANT_N,
             100 * config.EVOLVE_MAX_CARRY_FRAC))

    a = next_generation(seed_pop, scores, GEN, DEMO)
    b = next_generation(seed_pop, scores, GEN, DEMO)
    ha, hb = [e["hash"] for e in a], [e["hash"] for e in b]
    print("  reproducible: %s  (%d slots, same order, same hashes)"
          % (ha == hb, len(ha)))
    print("  distinct    : %d of %d hashes (dedup bound %d re-mutations)"
          % (len(set(ha)), len(ha), config.DEDUP_MAX_TRIES))
    print("  op counts   : %s" % ", ".join("%s %d" % kv for kv in sorted(op_counts(a).items())))

    best = _rank(seed_pop, scores)[0]
    print("  elite       : %s carried verbatim (score %+.3f), lineage op=%s birth_gen=%d"
          % (a[0]["hash"], scores[a[0]["hash"]], a[0]["op"], a[0]["birth_gen"]))
    print("                same genome as the top of the parent population: %s"
          % (a[0]["hash"] == best["hash"]))
    xover = [e for e in a if e["op"] == "crossover"]
    if xover:
        print("  crossover   : %s <- %s" % (xover[0]["hash"], " + ".join(xover[0]["parent_hash"])))
    mut = [e for e in a if e["op"] == "mutate"]
    if mut:
        print("  mutant      : %s <- %s" % (mut[0]["hash"], mut[0]["parent_hash"]))

    # A different generation number must breed a different population from the same
    # parents: the generation is part of every child's coordinates.
    c = next_generation(seed_pop, scores, GEN + 1, DEMO)
    moved = sum(1 for x, y in zip(ha, [e["hash"] for e in c]) if x != y)
    print("  gen %d vs %d : %d of %d slots differ (elites are the same by design)"
          % (GEN, GEN + 1, moved, len(ha)))

    # Hall of fame: two generations of fake F1 results, latest score wins.
    hof = update_hall_of_fame([], [(e, {"sharpe_prevault": scores[e["hash"]],
                                        "score": scores[e["hash"]] - 0.01,
                                        "n_days_prevault": 5000})
                                   for e in seed_pop[:8]], GEN)
    print("  hall of fame: %d records (HOF_SIZE=%d), best %s %s SR %+.3f"
          % (len(hof), config.HOF_SIZE, hof[0]["hash"], hof[0]["family"],
             hof[0]["sharpe_prevault"]))
    top = hof[0]
    hof2 = update_hall_of_fame(hof, [(seed_pop[[e["hash"] for e in seed_pop].index(top["hash"])],
                                      {"sharpe_prevault": top["sharpe_prevault"] - 0.5,
                                       "score": 0.0, "n_days_prevault": 5200})], GEN + 1)
    now = [r for r in hof2 if r["hash"] == top["hash"]][0]
    print("  re-evaluated: %s %+.3f -> %+.3f at generation %d (latest, not best)"
          % (top["hash"], top["sharpe_prevault"], now["sharpe_prevault"], now["generation"]))
    assert np.isclose(now["sharpe_prevault"], top["sharpe_prevault"] - 0.5)
