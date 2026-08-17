"""
The promotion gate stack: ten conditions, all of which must pass, ties to the
incumbent.

    report = evaluate_gates(cand, inc, cfg=config)
    for line in gate_table(report): print(line)

PURE FUNCTIONS OF TWO DICTS. This module does no I/O, runs no simulation, reads
no market and touches no clock: it takes the metrics run_deepeval.py measured for
the candidate and for the incumbent, and returns a report. That is what makes the
decision stack testable without a market (verify.py test 7 drives it on synthetic
dicts), and it is why the expensive, fallible half — re-simulating both parties,
paying for CPCV, counting vault accesses — lives in the runner where it can be
audited separately from the arithmetic that acts on it.

THE THRESHOLDS ARE docs/DESIGN.md's TABLE, AND THEY LIVE IN config.py. Nothing
here holds a number.

| G1  | like-for-like    | identical data/panel/config hash + window; incumbent re-simulated fresh |
| G2  | DSR (pre-vault)  | >= GATE_MIN_DSR at n_trials = the full ledger  |
| G3  | vault            | vault Sharpe > 0 AND vault DSR >= GATE_VAULT_MIN_DSR at N = vault_trials |
| G4  | PBO (CSCV S=16)  | <= GATE_MAX_PBO                               |
| G5  | CPCV 28 paths    | >= GATE_CPCV_MIN_POS_FRAC positive, median SR >= GATE_CPCV_MIN_MEDIAN_SR |
| G6  | bootstrap CI     | lower bound > 0                               |
| G7  | 2x cost stress   | stressed Sharpe > 0 and >= GATE_STRESS_MIN_SR_RATIO x base |
| G8  | regime slices    | none below GATE_REGIME_MAX_LOSS; >= GATE_REGIME_MIN_OK above GATE_REGIME_SOFT_LOSS |
| G9  | beats incumbent  | Sharpe > incumbent + GATE_BEAT_SR_MARGIN AND wins >= GATE_ROLLING_WIN_FRAC of rolling windows |
| G10 | ruin MC          | P(DD > GATE_RUIN_DD in RUIN_MC_YEARS) < GATE_RUIN_MAX_PROB |

TIES GO TO THE INCUMBENT — the report says so in `ties_to_incumbent`, and it is
implemented rather than asserted: every "beats" comparison is STRICT, a rolling
window is won only outright, and a missing or unreadable number is a FAILURE, not
a skip. A gate that cannot be evaluated has not been passed. The single exception
is written into DESIGN itself: with no incumbent (nothing has ever been promoted)
G9 has nothing to beat and is skipped with margin 0.

PASS-BY-ABSENCE, THE ONE PLACE THE RULE ABOVE IS INVERTED, IS G8's REGIME SLICES.
A NaN slice means the scored series contains no day of that crisis at all. Failing
a candidate for a window it never traded would be inventing evidence, so a NaN
counts as satisfying both halves of G8 — and run_deepeval prints the day count
beside each slice so an absent window is visible rather than silently generous.
"""
from __future__ import annotations

import math

import config                       # FIRST: puts the siblings on sys.path

GATE_ORDER = ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10")
GATE_NAMES = {"G1": "like-for-like", "G2": "DSR (pre-vault)", "G3": "vault confirmation",
              "G4": "PBO (CSCV)", "G5": "CPCV 28 paths", "G6": "bootstrap Sharpe CI",
              "G7": "2x cost stress", "G8": "regime slices", "G9": "beats incumbent",
              "G10": "ruin MC"}


def _num(value):
    """A usable float, or None. None/NaN/unparseable all mean "not measured",
    and every gate treats that as a failure."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else out


def _gate(gid: str, passed: bool, value, threshold, detail: str = "") -> dict:
    return {"gate": gid, "name": GATE_NAMES[gid], "pass": bool(passed),
            "value": value, "threshold": threshold, "detail": detail}


def _identity(d) -> tuple:
    """(data_hash, panel_hash, config_hash) from a metrics dict, as strings.

    `panel_hash` is the like-for-like key, not data_hash (features.py's docstring
    explains why: the panel depends on six macro series data_hash never sees).
    features.py attaches it to the MarketData ad hoc, so a caller reading it with
    getattr can legitimately arrive here holding None — which is exactly why an
    empty component fails G1 instead of comparing equal to another empty one.
    """
    ident = (d or {}).get("identity")
    if isinstance(ident, dict):
        ident = (ident.get("data_hash"), ident.get("panel_hash"), ident.get("config_hash"))
    if ident is None:
        return ("", "", "")
    return tuple("" if v is None else str(v) for v in tuple(ident)[:3])


def _window(d) -> tuple:
    win = (d or {}).get("window") or ()
    return tuple(str(v) for v in tuple(win)[:2])


# ── the ten gates ──────────────────────────────────────────────────────────────
def _g1(cand, inc) -> dict:
    """Like-for-like. Everything below is a comparison of numbers, and a
    comparison of numbers produced under different data, a different feature
    panel, different settings or a different window is not evidence — it is the
    exact failure that promoted signal_lab's champion off a 27-symbol run against
    164-symbol challengers (docs/DESIGN.md "Why shaped this way", item 4)."""
    c_id, c_win = _identity(cand), _window(cand)
    missing = [n for n, v in zip(("data_hash", "panel_hash", "config_hash"), c_id) if not v]
    if missing or len(c_win) != 2 or not all(c_win):
        return _gate("G1", False, c_id + c_win, "complete identity",
                     "candidate identity incomplete: %s"
                     % (", ".join(missing) if missing else "no evaluation window"))
    for party, label in ((cand, "candidate"), (inc, "incumbent")):
        if party is not None and not party.get("resimulated", True):
            return _gate("G1", False, label, "re-simulated fresh",
                         "%s was not re-simulated on this run: a stored or resumed "
                         "result is not like-for-like (see load_f1_checkpoint)" % label)
    if inc is None:
        return _gate("G1", True, c_id, "complete identity",
                     "no incumbent: nothing to compare against, identity complete")

    i_id, i_win = _identity(inc), _window(inc)
    diffs = [n for n, a, b in zip(("data_hash", "panel_hash", "config_hash"), c_id, i_id)
             if a != b]
    if c_win != i_win:
        diffs.append("window")
    return _gate("G1", not diffs, c_id + c_win, i_id + i_win,
                 "identical data, panel, settings and window" if not diffs
                 else "differs on %s — the incumbent must be re-simulated on this "
                      "run's inputs before the two numbers mean anything"
                      % ", ".join(diffs))


def _g2(cand, cfg) -> dict:
    dsr = _num(cand.get("dsr"))
    return _gate("G2", dsr is not None and dsr >= cfg.GATE_MIN_DSR, dsr, cfg.GATE_MIN_DSR,
                 "deflated at n_trials=%s, empirical trial-Sharpe spread"
                 % cand.get("dsr_n_trials", "?"))


def _g3(cand, cfg) -> dict:
    sr = _num(cand.get("vault_sharpe"))
    dsr = _num(cand.get("vault_dsr"))
    passed = (sr is not None and sr > 0.0 and dsr is not None
              and dsr >= cfg.GATE_VAULT_MIN_DSR)
    return _gate("G3", passed, (sr, dsr), (0.0, cfg.GATE_VAULT_MIN_DSR),
                 "vault Sharpe must be positive and its DSR deflated at "
                 "N=%s vault accesses" % cand.get("vault_trials", "?"))


def _g4(cand, cfg) -> dict:
    """CSCV over the generation's pre-vault returns matrix — a COHORT statistic.

    When there is no number, the note says WHY there is no number (the caller puts
    it in `pbo_note`): "unmeasured" and "measured at 0.51" both fail, but only one
    of them is about the candidate, and a gate table that could not tell them apart
    would send a reader hunting through the job log.
    """
    pbo = _num(cand.get("pbo"))
    if pbo is None:
        return _gate("G4", False, None, cfg.GATE_MAX_PBO,
                     "NOT MEASURABLE — %s. An unmeasured gate is not a passed gate."
                     % (cand.get("pbo_note") or "no PBO was supplied"))
    return _gate("G4", pbo <= cfg.GATE_MAX_PBO, pbo, cfg.GATE_MAX_PBO,
                 "CSCV over the generation's pre-vault returns matrix (S=%d): a "
                 "COHORT statistic, not a per-genome one" % cfg.PBO_SPLITS)


def _g5(cand, cfg) -> dict:
    frac = _num(cand.get("cpcv_frac_positive"))
    med = _num(cand.get("cpcv_median_sharpe"))
    passed = (frac is not None and frac >= cfg.GATE_CPCV_MIN_POS_FRAC
              and med is not None and med >= cfg.GATE_CPCV_MIN_MEDIAN_SR)
    return _gate("G5", passed, (frac, med),
                 (cfg.GATE_CPCV_MIN_POS_FRAC, cfg.GATE_CPCV_MIN_MEDIAN_SR),
                 "%s combinatorial purged paths, real refits" % cand.get("cpcv_n_paths", "?"))


def _g6(cand, cfg) -> dict:
    lo = _num(cand.get("boot_ci_lo"))
    return _gate("G6", lo is not None and lo > 0.0, lo, 0.0,
                 "lower bound of the %.0f%% block-bootstrap interval, %s resamples"
                 % (100 * cfg.GATE_BOOT_CI, cand.get("boot_iters", cfg.BOOT_ITERS)))


def _g7(cand, cfg) -> dict:
    """Costs doubled and borrow tripled. A base Sharpe that is not positive fails
    here rather than dividing by it: a ratio against a non-positive denominator is
    not a number this gate can act on, and such a candidate has already failed G6."""
    base = _num(cand.get("sharpe"))
    stressed = _num(cand.get("sharpe_stress"))
    ratio = (stressed / base) if (base is not None and base > 0.0
                                  and stressed is not None) else None
    passed = (stressed is not None and stressed > 0.0
              and ratio is not None and ratio >= cfg.GATE_STRESS_MIN_SR_RATIO)
    return _gate("G7", passed, (stressed, ratio),
                 (0.0, cfg.GATE_STRESS_MIN_SR_RATIO),
                 "stress_mult %.1f and borrow x%.0f against base SR %s"
                 % (cfg.GATE_STRESS_MULT, cfg.GATE_BORROW_STRESS_MULT,
                    "n/a" if base is None else "%+.2f" % base))


def _g8(cand, cfg) -> dict:
    """Crisis windows. NaN = the series never traded that window = pass by absence
    (see the module docstring); every covered slice still has to clear the hard
    floor, and at least GATE_REGIME_MIN_OK of the four must clear the soft one."""
    raw = cand.get("regime_slices")
    if raw is None or len(raw) != len(cfg.GATE_REGIME_WINDOWS):
        return _gate("G8", False, raw, cfg.GATE_REGIME_MAX_LOSS,
                     "expected one cumulative return per %d configured windows"
                     % len(cfg.GATE_REGIME_WINDOWS))
    vals = [_num(v) for v in raw]
    worst = [v for v in vals if v is not None]
    n_ok = sum(1 for v in vals if v is None or v > cfg.GATE_REGIME_SOFT_LOSS)
    n_absent = sum(1 for v in vals if v is None)
    passed = (all(v >= cfg.GATE_REGIME_MAX_LOSS for v in worst)
              and n_ok >= cfg.GATE_REGIME_MIN_OK)
    return _gate("G8", passed, (min(worst) if worst else None, n_ok),
                 (cfg.GATE_REGIME_MAX_LOSS, cfg.GATE_REGIME_MIN_OK),
                 "%d of %d windows covered; %d above %.0f%%, worst %s"
                 % (len(worst), len(vals), n_ok, 100 * cfg.GATE_REGIME_SOFT_LOSS,
                    "n/a" if not worst else "%+.1f%%" % (100 * min(worst))))


def _g9(cand, inc, cfg) -> dict:
    """Beats the incumbent, or there is no reason to change anything.

    Both halves are STRICT wins. DESIGN writes the margin with a >=, but the gate
    is named "beats" and the standing rule is that ties go to the incumbent, so a
    candidate that lands exactly on incumbent + margin has not beaten it. The
    rolling-window fraction keeps its >= (it is a proportion threshold, not a
    head-to-head), and the individual windows behind it are already strict wins —
    see evaluate.rolling_window_wins.
    """
    if inc is None:
        return _gate("G9", True, None, 0.0,
                     "skipped: nothing has ever been promoted, so the margin is 0 "
                     "(docs/DESIGN.md G9)")
    cand_sr, inc_sr = _num(cand.get("sharpe")), _num(inc.get("sharpe"))
    frac = _num(cand.get("rolling_win_frac"))
    if cand_sr is None or inc_sr is None:
        return _gate("G9", False, (cand_sr, inc_sr), cfg.GATE_BEAT_SR_MARGIN,
                     "a missing Sharpe on either side cannot beat anything")
    beats = cand_sr > inc_sr + cfg.GATE_BEAT_SR_MARGIN
    wins = frac is not None and frac >= cfg.GATE_ROLLING_WIN_FRAC
    return _gate("G9", beats and wins, (cand_sr - inc_sr, frac),
                 (cfg.GATE_BEAT_SR_MARGIN, cfg.GATE_ROLLING_WIN_FRAC),
                 "candidate %+.2f vs incumbent %+.2f over %s rolling %d-year windows"
                 % (cand_sr, inc_sr, cand.get("rolling_n_windows", "?"),
                    cfg.GATE_ROLLING_WINDOW_YEARS))


def _g10(cand, cfg) -> dict:
    p = _num(cand.get("p_ruin"))
    return _gate("G10", p is not None and p < cfg.GATE_RUIN_MAX_PROB, p,
                 cfg.GATE_RUIN_MAX_PROB,
                 "P(drawdown > %.0f%% within %d years), worse of GARCH-t and block "
                 "bootstrap over %d paths each"
                 % (100 * cfg.GATE_RUIN_DD, cfg.RUIN_MC_YEARS, cfg.RUIN_MC_PATHS))


def evaluate_gates(cand: dict, inc=None, cfg=config) -> dict:
    """Run the ten gates. `cand` and `inc` are metric dicts; `inc` None = no champion.

    Expected keys on each party (anything missing fails its gate):
        identity            (data_hash, panel_hash, config_hash) — or a dict of them
        window              (start, end) of the SCORED pre-vault span
        resimulated         False if the numbers came from a stored/resumed result
        sharpe              pre-vault annualised net Sharpe        G7, G9
        dsr, dsr_n_trials   pre-vault deflated Sharpe               G2
        vault_sharpe, vault_dsr, vault_trials                       G3
        pbo                                                         G4
        cpcv_frac_positive, cpcv_median_sharpe, cpcv_n_paths        G5
        boot_ci_lo, boot_ci_hi                                      G6
        sharpe_stress                                               G7
        regime_slices       one cumulative return per config window G8
        rolling_win_frac    share of rolling windows won vs `inc`   G9
        p_ruin                                                      G10
    """
    gates = {"G1": _g1(cand, inc), "G2": _g2(cand, cfg), "G3": _g3(cand, cfg),
             "G4": _g4(cand, cfg), "G5": _g5(cand, cfg), "G6": _g6(cand, cfg),
             "G7": _g7(cand, cfg), "G8": _g8(cand, cfg),
             "G9": _g9(cand, inc, cfg), "G10": _g10(cand, cfg)}
    failed = [gid for gid in GATE_ORDER if not gates[gid]["pass"]]
    return {"candidate": cand.get("hash"),
            "incumbent": (inc or {}).get("hash"),
            "gates": gates,
            "order": list(GATE_ORDER),
            "failed": failed,
            "all_pass": not failed,
            "n_gates": len(GATE_ORDER),
            # Not decoration: it is the doctrine every comparison above implements
            # (strict "beats", a tie is not a win, an unmeasurable gate is a fail).
            "ties_to_incumbent": True}


# ── reporting ──────────────────────────────────────────────────────────────────
def _fmt(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (tuple, list)):
        return " / ".join(_fmt(v) for v in value)
    if isinstance(value, float):
        return "n/a" if math.isnan(value) else "%.3f" % value
    return str(value)


def _cell(value, width: int = 26) -> str:
    """One table cell. G1's identity tuples are far longer than any column, and a
    table that wraps is a table nobody reads — the untruncated values stay in the
    report dict (and so in the artifact), which is what a reader who needs the
    full hashes should be looking at anyway."""
    text = _fmt(value)
    return text if len(text) <= width else text[:width - 1] + "~"


def gate_table(report: dict) -> list:
    """The gate table as lines of text: value vs threshold, per gate. Every report
    this project prints shows all ten, passed or failed — a refusal that only
    listed what broke would hide how close the rest came."""
    row = "  %-4s %-20s %-7s %-26s %-26s %s"
    lines = [row % ("gate", "what it asks", "verdict", "value", "threshold", "note"),
             "  " + "-" * 90]
    for gid in report["order"]:
        g = report["gates"][gid]
        lines.append(row % (gid, g["name"], "PASS" if g["pass"] else "FAIL",
                            _cell(g["value"]), _cell(g["threshold"]), g["detail"]))
    lines.append("  " + "-" * 90)
    lines.append("  verdict: %s"
                 % ("ALL TEN PASS — promotion is allowed"
                    if report["all_pass"] else
                    "REFUSED — %s did not pass" % ", ".join(report["failed"])))
    return lines


if __name__ == "__main__":
    # Smoke test: a passing candidate, then one violation at a time. Pure dicts —
    # no market, no cache, no network, and no simulation anywhere in this file.
    PASSING = {
        "hash": "cand00000001",
        "identity": ("data0000", "panel000", "cfg00000"),
        "window": ("1999-01-04", "2019-12-31"),
        "sharpe": 1.10, "dsr": 0.98, "dsr_n_trials": 812,
        "vault_sharpe": 0.65, "vault_dsr": 0.94, "vault_trials": 6,
        "pbo": 0.12,
        "cpcv_frac_positive": 0.82, "cpcv_median_sharpe": 0.55, "cpcv_n_paths": 28,
        "boot_ci_lo": 0.31, "boot_ci_hi": 1.88, "boot_iters": 5000,
        "sharpe_stress": 0.74,
        "regime_slices": [0.12, -0.03, 0.08, -0.02],
        "rolling_win_frac": 0.71, "rolling_n_windows": 45,
        "p_ruin": 0.02,
    }
    INCUMBENT = dict(PASSING, hash="incum0000001", sharpe=0.80)

    print("arena promotion gates (docs/DESIGN.md G1-G10)\n")
    report = evaluate_gates(PASSING, INCUMBENT)
    for line in gate_table(report):
        print(line)

    print("\n  one violation at a time (everything else held at the passing values):")
    breaks = [("G1", {"identity": ("OTHER", "panel000", "cfg00000")}),
              ("G2", {"dsr": config.GATE_MIN_DSR - 0.01}),
              ("G3", {"vault_sharpe": -0.01}),
              ("G4", {"pbo": config.GATE_MAX_PBO + 0.01}),
              ("G5", {"cpcv_median_sharpe": config.GATE_CPCV_MIN_MEDIAN_SR - 0.01}),
              ("G6", {"boot_ci_lo": 0.0}),
              ("G7", {"sharpe_stress": 0.4 * PASSING["sharpe"]}),
              ("G8", {"regime_slices": [-0.31, -0.03, 0.08, -0.02]}),
              ("G9", {"sharpe": INCUMBENT["sharpe"] + config.GATE_BEAT_SR_MARGIN}),
              ("G10", {"p_ruin": config.GATE_RUIN_MAX_PROB})]
    for gid, patch in breaks:
        rep = evaluate_gates(dict(PASSING, **patch), INCUMBENT)
        print("    %-4s broken -> all_pass %-5s  failed %s"
              % (gid, rep["all_pass"], ", ".join(rep["failed"])))

    solo = evaluate_gates(PASSING, None)
    print("\n  no incumbent : all_pass %s, G9 %s (%s)"
          % (solo["all_pass"], "PASS" if solo["gates"]["G9"]["pass"] else "FAIL",
             solo["gates"]["G9"]["detail"]))
    tie = evaluate_gates(dict(PASSING, sharpe=INCUMBENT["sharpe"]), INCUMBENT)
    print("  exact tie    : G9 %s — ties go to the incumbent, always"
          % ("PASS" if tie["gates"]["G9"]["pass"] else "FAIL"))
