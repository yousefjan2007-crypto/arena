"""
The immutable registry: what arena evaluated, what it promoted, and how to undo it.

    store_artifact(entry, f1_result, f2_metrics=None, decisions=None) -> path
    champion()                            -> (hash, meta) | None
    promote(hash, generation, gate_report) -> meta
    rollback(hash, reason)                 -> meta

THIS FILE IS THE ANSWER TO SIGNAL_LAB'S THIRD FAILURE (docs/DESIGN.md "Why shaped
this way"): `champion.joblib` overwritten in place, `model_version` constant, and
therefore no way to say what the live model was three weeks ago, what evidence
promoted it, or how to put the old one back. Here:

  • an artifact is CONTENT-ADDRESSED by the genome hash and WRITE-ONCE. Storing
    the same thing twice is a silent no-op; storing something DIFFERENT under a
    hash that already exists raises ImmutableArtifact rather than overwriting.
  • champion.json is a POINTER — the one file this project rewrites — and every
    rewrite appends a row to champion_history.csv. promote() and rollback() are
    the same mechanics with different reasons, so a rollback is as ordinary and
    as auditable as a promotion.
  • the pointer rewrite is verified by READING IT BACK: a champion.json whose
    hash has no matching history row is the corruption this file exists to
    prevent, so it is checked rather than assumed.

A REJECTED STORE CHANGES NOTHING. store_artifact plans every file and checks every
immutability rule BEFORE it writes the first byte, so "same hash + same content →
silent no-op" still holds after a store was refused. The alternative was found the
hard way: writing the return series first and raising on the metrics conflict
afterwards left an orphan that made the original evaluation's own re-store illegal
from then on.

THREE DOCUMENTED EXCEPTIONS TO "WRITE-ONCE", ALL BECAUSE THE SAME GENOME IS
EVALUATED AGAIN EVERY WEEK, ON NEW DATA:

  metrics.json is MERGE-APPEND. It may gain keys; an existing key may never
  change. Every evaluation is filed under its own eval key —
  (generation, data_hash, panel_hash, config_hash[, a caller tag]) — so next
  week's numbers land
  beside this week's instead of on top of them, and re-running the SAME week
  reproduces byte-identical numbers (everything upstream is seeded) and writes
  nothing. A different number under the SAME eval key is a real corruption and
  still raises.

  daily_returns/decisions follow the same rule by FILE: the first evaluation
  writes `daily_returns.csv.gz`, a later evaluation under a different eval key
  writes `daily_returns_<key8>.csv.gz`, and the eval record in metrics.json names
  its own file. Rewriting an existing file with different bytes still raises.

  genome.json's `lineage` is a LIST, appended to when a genome is re-discovered
  by a different parent (which is a fact about the search, not a conflict). The
  `genome` body itself is immutable — it is what the hash means, so a mismatch
  there is a hash collision and raises.

NO WALL CLOCK. Nothing here is timestamped: `generation` is arena's clock, and a
file whose bytes depend on when it was written could never be compared for
equality, which is the whole immutability mechanism. gzip members are written
with mtime=0 for the same reason.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os

import config                       # FIRST: puts the siblings on sys.path

GENOME_FILE = "genome.json"
METRICS_FILE = "metrics.json"
RETURNS_FILE = "daily_returns.csv.gz"
DECISIONS_FILE = "decisions.csv.gz"
CHAMPION_FILE = "champion.json"
HISTORY_FILE = "champion_history.csv"

HISTORY_COLUMNS = ("generation", "prev_hash", "new_hash", "reason", "note")
RETURN_COLUMNS = ("date", "segment", "net", "gross", "turnover", "costs")
# env.py's decision-log row, in its documented order. `reason` is a "+"-joined
# compound ("stop+regime") — a reader must split it, never compare it for equality.
DECISION_COLUMNS = ("date", "symbol", "side", "shares", "fill_px", "commission",
                    "spread_cost", "slippage", "borrow", "weight_before",
                    "weight_after", "reason")

CHAMPION_NOTE = ("REWRITTEN on every promotion or rollback: this is the champion "
                 "POINTER, not a record. Every rewrite appends a row to "
                 "champion_history.csv, and the artifact it points at is immutable.")
METRICS_NOTE = ("MERGE-APPEND only: an evaluation is filed under its own eval key "
                "(schema version + generation + data/panel/config hash) and existing "
                "keys are never rewritten. See registry.py's docstring.")

# The SHAPE of an eval record, not its numbers. Bump it in the same commit as any
# change to what store_artifact puts in `record` below.
#
# WHY IT IS IN THE KEY. The eval key promises "two evaluations sharing this key
# must agree number for number", and a record-shape change breaks that promise
# without changing a single number: the old record and the new one differ, so
# _plan_metrics raises ImmutableArtifact and a weekly job stops until a human
# moves files aside by hand. That happened once already — Phase 5 added
# `ledger_drift` and had to preserve artifacts/genomes.schema_pre_ledger_drift/ —
# and config_hash cannot prevent it, because config.py says of itself that it
# "cannot see a CODE change". So the schema version rides in the tag, which
# exists precisely for "something outside (generation, identity) that the record
# depends on". A schema change now files as a NEW evaluation beside the old one,
# which is what it is.
#   1  Phase 5 and earlier (unversioned keys — no `s<N>.` prefix)
#   2  filed_seq added (Phase 6)
METRICS_SCHEMA = 2


class ImmutableArtifact(RuntimeError):
    """A stored artifact was asked to change.

    The registry is the only claim this project makes about the past that is not
    reconstructible from the ledger: it holds the actual return series a gate
    decision was made on. If that can be rewritten, a champion's evidence can be
    edited after the fact and nothing in the repo would show it. So the write
    fails and a human decides — exactly like run_generation's IdentityDrift, and
    for the same reason.
    """


# ── paths (resolved per call, so a test can point config.* at a tmpdir) ────────
def genomes_dir(artifact_dir=None) -> str:
    return os.path.join(artifact_dir or config.ARTIFACT_DIR, "genomes")


def artifact_path(ghash: str, artifact_dir=None) -> str:
    return os.path.join(genomes_dir(artifact_dir), str(ghash))


def champion_path(state_dir=None) -> str:
    return os.path.join(state_dir or config.STATE_DIR, CHAMPION_FILE)


def history_path(state_dir=None) -> str:
    return os.path.join(state_dir or config.STATE_DIR, HISTORY_FILE)


# ── write-once primitives ──────────────────────────────────────────────────────
def _atomic_write(path: str, payload: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(payload)
    os.replace(tmp, path)              # never leave a half-written artifact


def _gzip_bytes(payload: bytes) -> bytes:
    """Deterministic gzip: mtime=0, so the same rows always give the same file."""
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        gz.write(payload)
    return buf.getvalue()


def _read_payload(path: str) -> bytes:
    """File contents, decompressed for .gz — the comparison has to be about the
    ROWS, not about which gzip level or header the writing process chose."""
    with open(path, "rb") as f:
        raw = f.read()
    return gzip.decompress(raw) if path.endswith(".gz") else raw


def _plan_write(path: str, payload: bytes, kind: str) -> bool:
    """Would writing `payload` (the UNCOMPRESSED content, whatever the extension)
    here be legal, and is it necessary? True = write it, False = already there.

    CHECKS ONLY — IT WRITES NOTHING, and that is the whole point. store_artifact
    plans every file before it writes any of them, because a rejected write used to
    poison the artifact: the returns file went down, then the metrics conflict
    raised, and the orphaned file left behind made even the ORIGINAL evaluation's
    byte-identical re-store illegal from then on. "Same hash + same content →
    silent no-op" has to survive a rejected write, so nothing may be written until
    everything is known to be writable.
    """
    if os.path.exists(path):
        if _read_payload(path) == payload:
            return False
        raise ImmutableArtifact(
            "%s already exists with different content:\n    %s\n"
            "Artifacts are write-once — the evidence a gate decision was made on "
            "may not be edited after the fact. If the inputs really moved, the "
            "evaluation belongs to a new eval key (see registry.py), not to this "
            "file." % (kind, path))
    return True


def _commit(plan: list) -> None:
    """Write the planned (path, payload) pairs. Every check already passed."""
    for path, payload in plan:
        _atomic_write(path, _gzip_bytes(payload) if path.endswith(".gz") else payload)


def _json_bytes(obj) -> bytes:
    return (json.dumps(obj, indent=1, sort_keys=True) + "\n").encode()


# ── artifact ───────────────────────────────────────────────────────────────────
def eval_key(generation: int, identity, tag: str = "") -> str:
    """The coordinates of ONE evaluation: which generation, on which data, panel
    and settings. Two evaluations sharing this key must agree number for number —
    everything upstream is seeded — so it is the key metrics.json merges on.

    `tag` exists because that promise is only true for numbers computed from the
    market. Some are not: a deflated Sharpe is deflated by the TRIAL LEDGER, and
    the ledger grows — re-running the same deep eval is genuinely another look at
    the vault, so the same candidate honestly scores a slightly lower vault DSR
    the second time. The caller (run_deepeval, via `f1_result["eval_tag"]`) puts
    the counts its numbers depend on in here, so a second look is filed as a
    second evaluation instead of colliding with the first. Numbers that depend on
    NOTHING but the market never need it.

    The record SCHEMA version leads the tag, always (see METRICS_SCHEMA): a
    reshaped record is a different record, and it must not collide with the old
    shape under the same key.
    """
    parts = (int(generation),) + tuple(str(v) for v in identity)
    stamped = "s%d%s" % (METRICS_SCHEMA, "." + tag if tag else "")
    return "%04d|%s|%s|%s" % parts + "|%s" % stamped


def _key8(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:8]


def _fmt(value) -> str:
    """repr() for floats: it round-trips exactly, and an artifact that does not
    round-trip cannot be compared for equality on a re-store."""
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _returns_csv(f1_result) -> str:
    """date, segment, net, gross, turnover, costs — pre-vault AND vault rows.

    THE VAULT ROWS CARRY `net` AND NOTHING ELSE, on purpose. evaluate.full_eval
    lets the vault out through exactly one key (`vault_daily_net`) so that the
    grep-able vault invariant stays checkable; widening that surface to carry
    gross/turnover/costs would buy four columns of an artifact at the cost of the
    one structural guarantee this project's honesty rests on. The vault dates come
    from the caller (run_deepeval derives them from the market calendar and
    asserts the length), because full_eval does not return them either.
    """
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow(RETURN_COLUMNS)
    dates = f1_result["dates"]
    for i in range(len(dates)):
        w.writerow([str(dates[i].date()), "prevault",
                    _fmt(float(f1_result["daily_net"][i])),
                    _fmt(float(f1_result["daily_gross"][i])),
                    _fmt(float(f1_result["turnover"][i])),
                    _fmt(float(f1_result["costs"][i]))])
    vault_net = f1_result.get("vault_daily_net")
    vault_dates = f1_result.get("vault_dates")
    if vault_net is not None and vault_dates is not None:
        if len(vault_net) != len(vault_dates):
            raise ValueError("vault_dates (%d) does not line up with vault_daily_net (%d)"
                             % (len(vault_dates), len(vault_net)))
        for i in range(len(vault_dates)):
            w.writerow([str(vault_dates[i].date()), "vault",
                        _fmt(float(vault_net[i])), "", "", ""])
    return out.getvalue()


def _decisions_csv(decisions, ghash: str, identity) -> str:
    """Tier B (docs/DESIGN.md "Decision logging"): one row per fill or financing
    charge, each carrying the genome and config hash it belongs to."""
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow(DECISION_COLUMNS + ("genome_hash", "config_hash"))
    for row in decisions:
        w.writerow([_fmt(row.get(c)) for c in DECISION_COLUMNS]
                   + [ghash, str(identity[2])])
    return out.getvalue()


def _lineage(entry: dict) -> dict:
    return {"op": entry.get("op", ""),
            "parent_hash": entry.get("parent_hash", ""),
            "birth_gen": int(entry.get("birth_gen", 0))}


def _plan_genome(path: str, entry: dict):
    """genome.json — the body is immutable, the lineage list is append-only.
    Returns the payload to write, or None when the file already says this.

    A genome re-discovered from a different parent is not a conflict: it is the
    same strategy reached twice, which is a fact about the search worth keeping.
    A different genome BODY under the same hash would be a sha256 collision, and
    raises. Checks only — see _plan_write.
    """
    ghash = entry["hash"]
    body = entry["genome"]
    line = _lineage(entry)
    payload = {"hash": ghash, "genome": body, "lineage": [line]}
    if os.path.exists(path):
        current = json.loads(_read_payload(path).decode())
        if current["genome"] != body:
            raise ImmutableArtifact(
                "genome.json for %s holds a different genome body — the hash is "
                "sha256 of the canonical genome, so this is a collision or a "
                "corrupted artifact: %s" % (ghash, path))
        if line in current["lineage"]:
            return None                              # same genome, same story
        payload = dict(current, lineage=current["lineage"] + [line])
    return _json_bytes(payload)


def _filed_seq(evals: dict) -> int:
    """The next filing sequence number for this artifact.

    WHY A COUNTER AND NOT THE KEY ORDER. "Which of these evaluations is the most
    recent" has no answer in the file itself: metrics.json is written with
    sort_keys=True, so read-back order is LEXICOGRAPHIC, and the eval key ends in
    the trial counts the record's DSRs were deflated by. `trials9.vault3` sorts
    above `trials21.vault2`, so key order ranks an EARLIER, LESS DEFLATED record
    as the latest — and a report quoting it would be quoting the more flattering
    of two numbers while calling it the newest. So filing order is recorded
    explicitly, at store time, and readers take the max.

    Monotonic per artifact, assigned only to records that are actually new (see
    _plan_metrics), so a re-store stays the byte-identical no-op it has to be.
    """
    return 1 + max([int(v.get("filed_seq", 0)) for v in evals.values()] or [0])


def _same_record(a: dict, b: dict) -> bool:
    """Record equality for the immutability guard, ignoring the filing counter.

    filed_seq is bookkeeping the STORE assigns, not a measurement the caller
    supplies, so comparing it would make every re-store look like a conflict and
    turn the idempotent no-op into a raise.
    """
    return {k: v for k, v in a.items() if k != "filed_seq"} == \
           {k: v for k, v in b.items() if k != "filed_seq"}


def _plan_metrics(path: str, ghash: str, key: str, record: dict):
    """metrics.json — merge-append on the eval key (see the module docstring).
    Returns the payload to write, or None when this evaluation is already filed.
    Checks only — see _plan_write."""
    payload = {"hash": ghash, "note": METRICS_NOTE, "evals": {}}
    if os.path.exists(path):
        payload = json.loads(_read_payload(path).decode())
        payload.setdefault("evals", {})
    prior = payload["evals"].get(key)
    if prior is not None:
        if _same_record(prior, record):
            return None
        raise ImmutableArtifact(
            "metrics.json for %s already holds different numbers under eval key\n"
            "    %s\n"
            "Same generation, same data, same panel, same settings — so the two "
            "results should be identical and one of them is wrong. Nothing was "
            "written: %s" % (ghash, key, path))
    payload["evals"][key] = dict(record, filed_seq=_filed_seq(payload["evals"]))
    payload["note"] = METRICS_NOTE
    return _json_bytes(payload)


def _target(root: str, base: str, ext: str, key: str, payload: bytes) -> str:
    """Where this evaluation's series file goes.

    The FIRST evaluation owns the plain name the layout in docs/DESIGN.md
    promises (`daily_returns.csv.gz`); an evaluation that disagrees with it — a
    later week, on refreshed data — gets a name of its own keyed by its eval key,
    because both series are true and neither may overwrite the other. Re-storing
    an evaluation lands back on the file it already wrote, so it stays a no-op.
    """
    plain = os.path.join(root, base + ext)
    keyed = os.path.join(root, "%s_%s%s" % (base, _key8(key), ext))
    if os.path.exists(keyed):
        return keyed
    if not os.path.exists(plain) or _read_payload(plain) == payload:
        return plain
    return keyed


def store_artifact(genome_entry: dict, f1_result: dict, f2_metrics=None,
                   decisions=None, artifact_dir=None) -> str:
    """Write artifacts/genomes/<hash12>/ for one evaluated genome. Returns the dir.

    `genome_entry`  a population/hall-of-fame entry: hash, genome, op,
                    parent_hash, birth_gen.
    `f1_result`     an evaluate.full_eval result PLUS two keys the caller owns:
                      identity    (data_hash, panel_hash, config_hash)
                      generation  which generation this evaluation belongs to
                    optionally `vault_dates` alongside `vault_daily_net` (see
                    _returns_csv for why the dates are the caller's), and
                    optionally `eval_tag` (see eval_key).
    `f2_metrics`    the F2 battery for this evaluation, gate report included
                    under "gate_report" when there is one. None for an F1-only
                    store — the record then says so rather than inventing nulls.
    `decisions`     Tier B decision-log rows (the winner and the champion get
                    these; a whole population would be gigabytes).

    PLAN EVERYTHING, THEN WRITE. Every immutability check below runs before the
    first byte is written, so a rejected store leaves the directory exactly as it
    found it. It did not always: the returns file was written first and the
    metrics conflict raised afterwards, and the orphan left behind made the
    ORIGINAL evaluation's own byte-identical re-store illegal from then on — a
    weekly job that could only be repaired by hand. Nothing here is allowed to
    half-happen.
    """
    ghash = genome_entry["hash"]
    identity = tuple(str(v) for v in f1_result["identity"])
    generation = int(f1_result["generation"])
    key = eval_key(generation, identity, str(f1_result.get("eval_tag", "")))
    root = artifact_path(ghash, artifact_dir)
    # No mkdir here: _atomic_write creates the directory for the first file it
    # writes, so a first-ever store that is REJECTED leaves not even an empty
    # directory behind — the same "a rejected store changes nothing" rule, applied
    # to the container as well as the contents.

    plan = []                                    # (path, uncompressed payload)
    genome_payload = _plan_genome(os.path.join(root, GENOME_FILE), genome_entry)
    if genome_payload is not None:
        plan.append((os.path.join(root, GENOME_FILE), genome_payload))

    returns_payload = _returns_csv(f1_result).encode()
    returns_file = _target(root, "daily_returns", ".csv.gz", key, returns_payload)
    if _plan_write(returns_file, returns_payload, "daily returns"):
        plan.append((returns_file, returns_payload))

    decisions_file = None
    if decisions is not None:
        dec_payload = _decisions_csv(decisions, ghash, identity).encode()
        decisions_file = _target(root, "decisions", ".csv.gz", key, dec_payload)
        if _plan_write(decisions_file, dec_payload, "decision log"):
            plan.append((decisions_file, dec_payload))

    dates = f1_result["dates"]
    f2 = dict(f2_metrics) if f2_metrics else None
    gate_report = f2.pop("gate_report", None) if f2 else None
    record = {
        "generation": generation,
        "identity": {"data_hash": identity[0], "panel_hash": identity[1],
                     "config_hash": identity[2]},
        "window": [str(dates[0].date()), str(dates[-1].date())] if len(dates) else [],
        "vault_window": ([str(f1_result["vault_dates"][0].date()),
                          str(f1_result["vault_dates"][-1].date())]
                         if f1_result.get("vault_dates") is not None
                         and len(f1_result["vault_dates"]) else []),
        "returns_file": os.path.basename(returns_file),
        "decisions_file": os.path.basename(decisions_file) if decisions_file else None,
        "f1": {k: _jsonable(f1_result.get(k)) for k in
               ("score", "sharpe_prevault", "n_days_prevault", "n_features",
                "first_active", "regime_finite_frac", "n_fits")},
        "f2": _jsonable(f2),
        "gate_report": _jsonable(gate_report),
    }
    record = json.loads(json.dumps(record))       # one canonical JSON shape to compare
    metrics_payload = _plan_metrics(os.path.join(root, METRICS_FILE), ghash, key, record)
    if metrics_payload is not None:
        plan.append((os.path.join(root, METRICS_FILE), metrics_payload))

    _commit(plan)                                 # every check passed; now write
    return root


def _jsonable(value):
    """numpy scalars/arrays -> plain Python, so a stored record compares equal to
    the one a later run builds (json.dumps of a np.float64 and of a float agree,
    but the round-tripped objects would not)."""
    if value is None or isinstance(value, (bool, str, int, float)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    return str(value)


def load_artifact(ghash: str, artifact_dir=None) -> dict:
    """{"hash", "genome", "lineage", "evals"} for a stored genome, or None."""
    root = artifact_path(ghash, artifact_dir)
    gpath = os.path.join(root, GENOME_FILE)
    if not os.path.exists(gpath):
        return None
    out = json.loads(_read_payload(gpath).decode())
    mpath = os.path.join(root, METRICS_FILE)
    out["evals"] = (json.loads(_read_payload(mpath).decode()).get("evals", {})
                    if os.path.exists(mpath) else {})
    return out


def artifact_entry(ghash: str, artifact_dir=None) -> dict:
    """A stored genome back as a population entry (the shape every arena module
    passes around), so the champion can be re-simulated from the registry alone."""
    art = load_artifact(ghash, artifact_dir)
    if art is None:
        return None
    line = (art.get("lineage") or [{}])[0]
    return {"genome": art["genome"], "hash": art["hash"],
            "op": line.get("op", "champion"), "parent_hash": line.get("parent_hash", ""),
            "birth_gen": int(line.get("birth_gen", 0))}


# ── the champion pointer ───────────────────────────────────────────────────────
def champion(state_dir=None):
    """(hash, meta) for the current champion, or None if nothing was ever promoted."""
    path = champion_path(state_dir)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        meta = json.load(f)
    return meta["hash"], meta


def champion_history(state_dir=None) -> list:
    """Every pointer move ever made, oldest first."""
    path = history_path(state_dir)
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _repoint(ghash: str, generation: int, reason: str, note: str, gate_report,
             state_dir=None) -> dict:
    """Rewrite champion.json and append the history row. Never called directly.

    THE READ-BACK IS THE POINT. Two files have to agree — the pointer and the
    log — and the only failure this project cannot tolerate is a pointer that
    moved without the log saying so. So both are written, then both are read
    back, and a disagreement raises instead of being reported as success.
    """
    prev = champion(state_dir)
    prev_hash = prev[0] if prev else ""
    meta = {"note": CHAMPION_NOTE,
            "hash": str(ghash),
            "generation": int(generation),
            "reason": reason,
            "detail": note,
            "previous_hash": prev_hash,
            "artifact": os.path.join("artifacts", "genomes", str(ghash)),
            "gate_report": _jsonable(gate_report)}

    path = history_path(state_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(HISTORY_COLUMNS)
        w.writerow([int(generation), prev_hash, str(ghash), reason, note])
    _atomic_write(champion_path(state_dir), _json_bytes(meta))

    back = champion(state_dir)
    rows = champion_history(state_dir)
    if back is None or back[0] != str(ghash) or not rows \
            or rows[-1]["new_hash"] != str(ghash) or rows[-1]["reason"] != reason:
        raise RuntimeError(
            "champion pointer and history disagree after writing %s: pointer=%s, "
            "last history row=%s. A pointer that moved without a history row is "
            "the one corruption this registry exists to prevent."
            % (ghash, back[0] if back else None, rows[-1] if rows else None))
    return meta


def promote(ghash: str, generation: int, gate_report, note: str = "",
            state_dir=None) -> dict:
    """Repoint the champion at `ghash` because it passed every gate.

    The gate report is stored WITH the pointer, not just in the artifact: the
    question "why is this the champion" must be answerable from the one file that
    says which genome is live, without joining anything.
    """
    if gate_report is not None and not gate_report.get("all_pass", False):
        raise ValueError("promote() refuses a gate report that did not pass: "
                         "failed %s" % ", ".join(gate_report.get("failed", [])))
    return _repoint(ghash, generation, "gates_passed", note, gate_report, state_dir)


def rollback(ghash: str, reason: str, generation: int = None, state_dir=None) -> dict:
    """Repoint the champion at any prior hash. Same mechanics as promote().

    A rollback carries no gate report — it is a human decision to stop trusting
    the current champion, not a claim that the old one just passed something —
    and `reason` is recorded verbatim beside the row.
    """
    if generation is None:
        prev = champion(state_dir)
        generation = int(prev[1].get("generation", 0)) if prev else 0
    return _repoint(ghash, generation, "rollback", reason, None, state_dir)


if __name__ == "__main__":
    # Smoke test: a store / promote / rollback round trip in temp directories.
    # Touches no real state, no market, no network.
    import shutil
    import tempfile

    import numpy as np
    import pandas as pd

    import genome as gn

    tmp = tempfile.mkdtemp(prefix="arena_registry_")
    state, arts = os.path.join(tmp, "state"), os.path.join(tmp, "artifacts")
    os.makedirs(state)
    try:
        rng = np.random.default_rng(config.SEED)
        lib = tuple("feat_%02d" % i for i in range(20))
        dates = pd.bdate_range("2000-01-03", periods=500)
        vault_dates = pd.bdate_range("2020-01-02", periods=120)
        identity = ("dead" * 4, "beef" * 4, "cafe" * 4)

        def result(g, seed_shift=0.0):
            net = rng.normal(0.0005 + seed_shift, 0.01, len(dates))
            return {"dates": dates, "daily_net": net, "daily_gross": net + 1e-4,
                    "turnover": np.abs(rng.normal(0.1, 0.02, len(dates))),
                    "costs": np.abs(rng.normal(1.5, 0.4, len(dates))),
                    "vault_daily_net": rng.normal(0.0004, 0.011, len(vault_dates)),
                    "vault_dates": vault_dates,
                    "identity": identity, "generation": 7,
                    "score": 0.61, "sharpe_prevault": 0.72, "n_days_prevault": 500,
                    "n_features": 5, "first_active": 1008, "regime_finite_frac": None,
                    "n_fits": 12}

        entries = [{"genome": gn.random_genome(rng, lib).to_dict(), "hash": "",
                    "op": "mutate", "parent_hash": "abc123abc123", "birth_gen": 6}
                   for _ in range(2)]
        for e in entries:
            e["hash"] = gn.from_dict(e["genome"]).hash()

        print("arena registry  (temp: %s)" % tmp)
        res_a, res_b = result(entries[0]), result(entries[1], 0.0002)
        gates = {"all_pass": True, "failed": [], "gates": {"G2": {"pass": True}}}
        decisions = [{"date": dates[0].date(), "symbol": "SPY", "side": "buy",
                      "shares": 10, "fill_px": 100.5, "commission": 1.0,
                      "spread_cost": 0.25, "slippage": 0.2, "borrow": 0.0,
                      "weight_before": 0.0, "weight_after": 0.07,
                      "reason": "rebalance"}]
        path_a = store_artifact(entries[0], res_a, {"dsr": 0.97, "gate_report": gates},
                                decisions, artifact_dir=arts)
        path_b = store_artifact(entries[1], res_b, {"dsr": 0.41}, artifact_dir=arts)
        print("  stored     : %s  (%s)"
              % (os.path.relpath(path_a, tmp), ", ".join(sorted(os.listdir(path_a)))))
        print("  stored     : %s" % os.path.relpath(path_b, tmp))

        # Immutability: identical re-store is silent; a changed series raises.
        store_artifact(entries[0], res_a, {"dsr": 0.97, "gate_report": gates},
                       decisions, artifact_dir=arts)
        before = sorted(os.listdir(path_a))
        moved = dict(res_a, daily_net=res_a["daily_net"] + 1e-6)
        try:
            store_artifact(entries[0], moved, {"dsr": 0.97, "gate_report": gates},
                           decisions, artifact_dir=arts)
            raised = "NO — the registry let history be rewritten"
        except ImmutableArtifact:
            raised = "ImmutableArtifact"
        print("  re-store   : identical -> silent no-op | changed returns -> %s" % raised)
        # ...and the rejection left NOTHING behind, so the original evaluation can
        # still be re-stored as the no-op it is.
        after = sorted(os.listdir(path_a))
        store_artifact(entries[0], res_a, {"dsr": 0.97, "gate_report": gates},
                       decisions, artifact_dir=arts)
        print("  rejected   : wrote no file (%s), and the original re-stores fine: %s"
              % ("clean" if after == before
                 else "ORPHANS: %s" % (set(after) - set(before)),
                 sorted(os.listdir(path_a)) == before))
        assert after == before, "a rejected store left an orphan file behind"

        # A new week (new data hash, one more bar, slightly different numbers)
        # files a second eval beside the first instead of overwriting it.
        next_week = dict(res_a, identity=("f00d" * 4, "beef" * 4, "cafe" * 4),
                         generation=8, daily_net=res_a["daily_net"] * 1.01)
        store_artifact(entries[0], next_week, {"dsr": 0.93}, artifact_dir=arts)
        art = load_artifact(entries[0]["hash"], artifact_dir=arts)
        print("  eval keys  : %d filed under one hash (%s)"
              % (len(art["evals"]), ", ".join(sorted(
                  os.path.basename(v["returns_file"]) for v in art["evals"].values()))))

        print("  champion   : %s (nothing promoted yet)" % champion(state))
        promote(entries[0]["hash"], 7, gates, "first champion", state_dir=state)
        promote(entries[1]["hash"], 8, gates, "challenger dethroned it", state_dir=state)
        h, meta = champion(state)
        print("  promoted   : %s <- %s (gen %d, %s)"
              % (h, meta["previous_hash"], meta["generation"], meta["reason"]))
        rollback(entries[0]["hash"], "smoke-test rollback", state_dir=state)
        h2, meta2 = champion(state)
        print("  rolled back: %s (was %s), reason %s / %s"
              % (h2, meta2["previous_hash"], meta2["reason"], meta2["detail"]))
        print("  history    : %d rows" % len(champion_history(state)))
        for row in champion_history(state):
            print("      gen %-3s %-12s -> %-12s %-13s %s"
                  % (row["generation"], row["prev_hash"] or "(none)", row["new_hash"],
                     row["reason"], row["note"]))
        assert h2 == entries[0]["hash"], "rollback did not restore the prior pointer"
        assert len(champion_history(state)) == 3
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
