"""
The weekly honest report: markdown + charts, built from what is already on disk.

`build_report(generation)` -> output/report_gen<NNNN>.md plus three PNGs. It reads
the immutable artifacts, the deep-eval history, the trial ledger and the hall of
fame; it computes nothing new about a strategy and re-simulates nothing. That is
deliberate — a report that recomputed its own numbers could disagree with the
gate report that actually decided the promotion, and then two documents in this
repository would describe two different runs.

WHAT IT REFUSES TO DO. Report a champion that does not exist. Until a candidate
clears all ten gates there is no champion, and the report says so in those words
rather than quietly promoting the best-looking genome to the top of a page. When
there is no champion, the SUBJECT of the report is the most recent deep eval's
leading candidate — and every chart title, every table header and the prose
around them carry the word REFUSED, because that is what happened to it. The
alternative (skip the charts entirely) would make the honest state of the system
invisible, and the state this project is most likely to be in for a long while
is "nothing has passed yet".

IT ALSO REFUSES TO REPORT GATES NOBODY RAN. The champion's newest record is
usually not an evaluation at all — run_deepeval re-simulates the incumbent every
week for G1 and G9 and stores that four-key result — so the report falls back to
the evidence that promoted the genome and says which generation it is from
(_promotion_evidence, resim_note). Reporting the re-simulation as-is would print
ten "not evaluated" gate rows and a claim that gate G4 failed, in the first
report after every promotion.

Everything user-facing carries docs/DESIGN.md's mandatory footer verbatim: the
survivorship disclosure and "backtest alpha is a claim about the past, not a
guarantee — not financial advice."

NO WALL CLOCK, not even for the report date. The report is dated by the last bar
of the data it describes, which is the only date that means anything about its
contents — and it keeps `python3 reports.py --gen N` reproducible, so two runs
over an unchanged state/ produce the same file.

    python3 reports.py                # the latest generation with a deep eval
    python3 reports.py --gen 1        # a specific one
"""
from __future__ import annotations

import csv
import gzip
import json
import os

import matplotlib
matplotlib.use("Agg")                  # headless: no display on a GitHub runner
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                     # noqa: E402
import pandas as pd                    # noqa: E402

import config                          # FIRST: puts the siblings on sys.path  # noqa: E402
import datafeed                                                               # noqa: E402
import evolution                                                              # noqa: E402
import gates                                                                  # noqa: E402
import ledger                                                                 # noqa: E402
import registry                                                               # noqa: E402
from alerts_arena import HONESTY_LINE, SURVIVORSHIP_LINE                      # noqa: E402

# docs/DESIGN.md, "Risks & honest limitations" — reproduced in full at the foot of
# every report, because a reader who only ever sees this page has to see them too.
FOOTER = """---

## Limitations — read these with every number above

- **Survivorship bias.** The universe is a snapshot of today's index membership,
  so every company that failed out of it is absent from all of these backtests:
  long results come out optimistic and short results pessimistic. Disclosed, not
  solved; delisted-inclusive (CRSP-class) data is the named upgrade path.
- **Non-stationarity.** 1990s microstructure is simulated with modern costs, and
  a decades-old edge may simply have been arbitraged away since. The vault, the
  regime gate and the paper stage are defenses, not proofs.
- **Multiple testing survives the gates.** Evolutionary trials are correlated, so
  the ledger DSR under-deflates; PBO does not cover designer-level choices at all.
  The only accumulating true out-of-sample is vault -> paper -> live.
- **Small-account sensitivity.** The $1 minimum commission and whole-share
  rounding are 5-10 bps each way on small positions; the 2x cost-stress gate and
  the cost-share line above are what keep that visible.
- **The cost model is proportional (bps), not per-share.** Prices are
  split-adjusted, so a $/share friction would silently become hundreds of bps on
  1990s adjusted prices. Real 1990s spreads were wider than modern bps: any edge
  that survives only at modern costs is suspect by construction.

**%s**

**%s**
""" % (SURVIVORSHIP_LINE, HONESTY_LINE)


# ── locating what to report on ─────────────────────────────────────────────────
def latest_reported_generation(state_dir=None) -> int:
    """The most recent generation a deep eval ran on, else the last returns matrix."""
    import run_deepeval
    hist = run_deepeval.read_history(state_dir)
    if hist:
        return max(int(r["generation"]) for r in hist)
    return run_deepeval.latest_generation(state_dir)


def _eval_for(art: dict, generation: int) -> dict:
    """The artifact's evaluation record for `generation` — the LAST one FILED if
    the same generation was evaluated more than once (a second look at the vault
    is a later, more deflated, and therefore more honest record than the first).

    Ranked by the explicit `filed_seq` the registry stamps at store time, because
    nothing else in the file can answer the question. metrics.json is written with
    sort_keys=True, so both the on-disk order and json.loads' insertion order are
    LEXICOGRAPHIC — and the eval key ends in the trial counts, where `trials9`
    outranks `trials21`. Either would hand back the earlier, less deflated record
    and this report would quote the more flattering number as the newest.

    Records written before Phase 6 carry no filed_seq and rank 0. Where every
    candidate is pre-Phase-6 the question is genuinely unanswerable from the file,
    and the last in iteration order is returned as it always was.
    """
    evals = art.get("evals") or {}
    matching = [v for k, v in evals.items()
                if k.split("|")[0] == "%04d" % int(generation)]
    if not matching:
        matching = list(evals.values())
    if not matching:
        return {}
    return max(enumerate(matching), key=lambda iv: (int(iv[1].get("filed_seq", 0)), iv[0]))[1]


def _promotion_evidence(subject: dict) -> dict:
    """A champion subject whose newest record is a bare RE-SIMULATION, refilled
    with the evidence that actually promoted it. Returns `subject`, unchanged
    unless it has to change.

    run_deepeval re-simulates the incumbent every week — that is gate G1's whole
    point — and stores the result as an evaluation record of its own carrying four
    keys: role, identity, window and Sharpe. No battery is run on it (the gates ask
    whether a CHALLENGER is real; nothing is asked of the champion but a
    like-for-like Sharpe and a series to be beaten), so the record has no gate
    report and no cohort statistic. Reported as-is it produces a page that claims
    things that never happened: ten "not evaluated" gate rows, and — because
    `pbo_in_cohort` is absent rather than False — the sentence "this candidate is
    not in the cohort matrix, so PBO is unmeasured and gate G4 fails on that
    alone", about a gate no one ran. That page is guaranteed the first Saturday
    after any promotion.

    The real evidence is on disk twice: the promotion-week eval record inside the
    artifact, and champion.json's own copy of the gate report registry.promote was
    handed. The first is preferred (it carries the battery as well as the verdict);
    the pointer's copy is the fallback when that record is gone. Either way the
    generation the gates ran in is returned as `gates_generation` and the
    re-simulation as `resim_record`, so the report can say which week is which.

    WHICH GENERATION TO LOOK IN is the pointer's `gates_generation` — when the
    evidence was measured — not its `generation`, which is when the pointer last
    moved. The two differ after a rollback: the pointer moves this week and the
    gates it points at were run whenever that genome was promoted. An explicit
    None means the registry has no promotion on record for this hash, and that is
    NOT a reason to fall back to `generation` — a battery labelled with the wrong
    week is the mislabelling this function exists to refuse.
    """
    record = subject.get("record") or {}
    if record.get("gate_report"):
        return subject                       # a real evaluation: nothing to fall back to
    meta = subject.get("champion_meta") or {}
    gen = meta["gates_generation"] if "gates_generation" in meta else meta.get("generation")
    promo = _eval_for(subject["artifact"], gen) if gen is not None else {}
    if not promo.get("gate_report") or int(promo.get("generation", -1)) != int(gen):
        # _eval_for falls back to "any record" when the generation has none, so a
        # record that is not FROM the promotion generation is not evidence about
        # this promotion and is refused here rather than mislabelled below.
        if not meta.get("gate_report"):
            return subject                   # nothing honest to show; claim nothing
        promo = dict(record, gate_report=meta["gate_report"])
    subject["record"] = promo
    subject["resim_record"] = record
    subject["gates_generation"] = gen
    return subject


def report_subject(generation: int, state_dir=None, artifact_dir=None) -> dict:
    """Who this report is about: {hash, role, artifact, record}, or None.

    role  "champion"  the promoted genome — the pointer says so
          "refused"   the leading candidate of the last deep eval, which the
                      gates turned down. Named in full so that nothing downstream
                      can present it as a champion by accident.

    A champion subject may carry two extra keys, `resim_record` and
    `gates_generation`: see _promotion_evidence for the week in which the newest
    record on file is a re-simulation rather than an evaluation.
    """
    import run_deepeval
    champ = registry.champion(state_dir)
    if champ is not None:
        art = registry.load_artifact(champ[0], artifact_dir)
        if art is not None:
            return _promotion_evidence(
                {"hash": champ[0], "role": "champion", "artifact": art,
                 "record": _eval_for(art, generation), "champion_meta": champ[1]})

    hist = [r for r in run_deepeval.read_history(state_dir)
            if int(r["generation"]) == int(generation)]
    for row in reversed(hist):
        for spec in (row.get("candidates") or "").split(";"):
            ghash = spec.split(":")[0]
            art = registry.load_artifact(ghash, artifact_dir) if ghash else None
            if art is not None:
                return {"hash": ghash, "role": "refused", "artifact": art,
                        "record": _eval_for(art, generation), "history_row": row}
    return None


# ── series ─────────────────────────────────────────────────────────────────────
def load_returns(ghash: str, record: dict, artifact_dir=None) -> pd.DataFrame:
    """The artifact's daily return series: date, segment, net, gross, turnover, costs.

    `segment` is "prevault" or "vault"; the vault rows carry net and nothing else
    (registry._returns_csv explains why), so gross-dependent lines below are
    pre-vault only and say so.
    """
    name = record.get("returns_file") or registry.RETURNS_FILE
    path = os.path.join(registry.artifact_path(ghash, artifact_dir), name)
    if not os.path.exists(path):
        return pd.DataFrame(columns=list(registry.RETURN_COLUMNS))
    with gzip.open(path, "rt") as f:
        rows = list(csv.DictReader(f))
    df = pd.DataFrame(rows)
    if not len(df):
        return df
    df["date"] = pd.to_datetime(df["date"])
    for col in ("net", "gross", "turnover", "costs"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def benchmark_curve(dates) -> pd.Series:
    """SPY buy-and-hold over the same dates, from the shared cache. Empty on miss.

    Buy-and-hold, not a traded benchmark: no costs, no rebalancing, nothing to
    tune. It is the thing an investor could have done instead, which is the only
    comparison that answers "was any of this worth doing".
    """
    try:
        px = datafeed._read_symbol(config.BENCHMARK, refresh=False)   # noqa: SLF001
    except Exception:
        return pd.Series(dtype=float)
    if not len(px) or "close" not in px.columns:
        return pd.Series(dtype=float)
    close = px["close"].astype(float).reindex(pd.DatetimeIndex(dates)).ffill()
    if not np.isfinite(close.to_numpy()).any():
        return pd.Series(dtype=float)
    first = close.dropna().iloc[0]
    return close / first


def equity_curve(net) -> np.ndarray:
    return np.cumprod(1.0 + np.asarray(net, dtype=np.float64))


def drawdown(curve) -> np.ndarray:
    curve = np.asarray(curve, dtype=np.float64)
    peak = np.maximum.accumulate(np.where(np.isfinite(curve), curve, -np.inf))
    return curve / peak - 1.0


def rolling_sharpe(net, window_years=None) -> np.ndarray:
    """Annualised Sharpe over a trailing window. NaN until the window fills —
    a partial window is a different statistic, not an early answer."""
    years = config.GATE_ROLLING_WINDOW_YEARS if window_years is None else window_years
    w = int(years * config.TRADING_DAYS_YEAR)
    s = pd.Series(np.asarray(net, dtype=np.float64))
    mu = s.rolling(w).mean()
    sd = s.rolling(w).std(ddof=1)
    return (mu / sd * np.sqrt(config.TRADING_DAYS_YEAR)).to_numpy()


# ── charts ─────────────────────────────────────────────────────────────────────
def _label(subject: dict) -> str:
    return "%s %s" % ("champion" if subject["role"] == "champion" else "REFUSED candidate",
                      subject["hash"])


def _vault_line(ax, dates) -> None:
    v = pd.Timestamp(config.VAULT_START)
    if len(dates) and dates[0] <= v <= dates[-1]:
        ax.axvline(v, color="0.4", linestyle=":", linewidth=1)
        ax.text(v, ax.get_ylim()[1], " vault", va="top", ha="left", fontsize=7, color="0.4")


def draw_charts(subject: dict, df: pd.DataFrame, generation: int, output_dir=None) -> list:
    """Three PNGs: equity vs SPY, rolling 3-yr Sharpe, drawdown. Returns filenames."""
    out = output_dir or config.OUTPUT_DIR
    os.makedirs(out, exist_ok=True)
    stem = "report_gen%04d" % int(generation)
    label = _label(subject)
    dates = pd.DatetimeIndex(df["date"])
    net = df["net"].to_numpy(dtype=np.float64)
    curve = equity_curve(net)
    bench = benchmark_curve(dates)
    written = []

    # 1. equity vs SPY buy-and-hold
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(dates, curve, linewidth=1.2, label="%s (net of costs)" % label)
    if len(bench):
        ax.plot(dates, bench.to_numpy(), linewidth=1.0, color="0.55",
                label="%s buy & hold (no costs)" % config.BENCHMARK)
    ax.set_yscale("log")
    ax.set_ylabel("growth of 1 (log)")
    ax.set_title("arena gen %d — %s vs %s buy & hold" % (generation, label, config.BENCHMARK))
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)
    _vault_line(ax, dates)
    written.append(_save(fig, out, stem + "_equity.png"))

    # 2. rolling Sharpe
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(dates, rolling_sharpe(net), linewidth=1.0)
    ax.axhline(0.0, color="0.4", linewidth=0.8)
    ax.set_ylabel("Sharpe")
    ax.set_title("rolling %d-year net Sharpe — %s" % (config.GATE_ROLLING_WINDOW_YEARS, label))
    ax.grid(alpha=0.25)
    _vault_line(ax, dates)
    written.append(_save(fig, out, stem + "_rolling_sharpe.png"))

    # 3. drawdown
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.fill_between(dates, 100 * drawdown(curve), 0.0, alpha=0.65, linewidth=0)
    if len(bench):
        ax.plot(dates, 100 * drawdown(bench.to_numpy()), linewidth=0.9, color="0.55",
                label="%s" % config.BENCHMARK)
        ax.legend(loc="lower left", fontsize=8)
    ax.set_ylabel("drawdown %")
    ax.set_title("drawdown — %s" % label)
    ax.grid(alpha=0.25)
    _vault_line(ax, dates)
    written.append(_save(fig, out, stem + "_drawdown.png"))
    return written


def _save(fig, out_dir: str, name: str) -> str:
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, name), dpi=110)
    plt.close(fig)
    return name


# ── tables ─────────────────────────────────────────────────────────────────────
def _f(value, fmt="%.4f") -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "n/a" if value is None else str(value)
    return "n/a" if x != x else fmt % x


def _first(*values):
    """The first value that was actually MEASURED, or None.

    `a or b` would fall through a 0.0, and 0.0 is a measurement: a Sharpe of
    exactly zero is what evaluate.sharpe returns for a series too short or too
    degenerate to have one, and reporting the F1 number in its place would quote a
    different quantity under the same row label.
    """
    for value in values:
        if value is not None:
            return value
    return None


def _battery_ran(f2: dict) -> bool:
    """Did an F2 battery produce this record, or is it a bare re-simulation?

    `pbo_in_cohort` is written for every candidate the battery ran on — including
    the ones whose PBO it could not measure, where it is False — so its PRESENCE,
    not its truth, is what separates an evaluation from run_deepeval's four-key
    incumbent re-simulation. See _promotion_evidence.
    """
    return bool(f2) and "pbo_in_cohort" in f2


def resim_note(subject: dict) -> list:
    """The label that stops a re-simulation week from reading as a fresh verdict.

    Empty for every subject whose newest record is a real evaluation — which is
    every report except the ones that follow a promotion.
    """
    if not subject.get("resim_record"):
        return []
    resim = subject["resim_record"] or {}
    win = resim.get("window") or []
    sharpe = (resim.get("f2") or {}).get("sharpe")
    gen = subject.get("gates_generation", "?")
    return ["> **Champion re-simulated for G1 and G9 this week; the gates were last "
            "run in generation %s.** The weekly deep eval re-runs the incumbent on "
            "today's data to supply a like-for-like Sharpe (G1) and a series for "
            "challengers to beat (G9). It runs no battery and no gates against the "
            "champion itself — the ten gates are what a CHALLENGER faces — so every "
            "number and every gate row below is the evidence that promoted this "
            "genome in generation %s, not a fresh verdict on it. This week's "
            "re-simulation: %s, pre-vault net Sharpe %s."
            % (gen, gen, " .. ".join(win) if win else "window not recorded",
               _f(sharpe, "%+.3f")),
            ""]


def gate_rows(record: dict) -> list:
    """Markdown rows for the ten gates: value vs threshold, from the stored report.

    Read from the artifact, not recomputed. The gate report in the artifact is the
    one the promotion decision was actually made on; a freshly computed table could
    differ from it and there would be no way to tell which page was lying.
    """
    report = record.get("gate_report") or {}
    stored = report.get("gates") or {}
    rows = []
    for gid in gates.GATE_ORDER:
        g = stored.get(gid)
        if g is None:
            rows.append("| %s | %s | — | — | not evaluated |"
                        % (gid, gates.GATE_NAMES.get(gid, "")))
            continue
        rows.append("| %s | %s | %s | %s | %s |"
                    % (gid, g.get("name", gates.GATE_NAMES.get(gid, "")),
                       gates._fmt(g.get("value")),            # noqa: SLF001
                       gates._fmt(g.get("threshold")),        # noqa: SLF001
                       "PASS" if g.get("pass") else "**FAIL**"))
    return rows


def regime_rows(record: dict) -> list:
    f2 = record.get("f2") or {}
    slices = f2.get("regime_slices") or []
    days = f2.get("regime_days") or []
    rows = []
    for i, (lo, hi) in enumerate(config.GATE_REGIME_WINDOWS):
        val = slices[i] if i < len(slices) else None
        n = days[i] if i < len(days) else None
        note = ""
        if not n:
            note = "no bars in window"
        elif val is not None and val == val:
            if val < config.GATE_REGIME_MAX_LOSS:
                note = "**below the hard floor (%.0f%%)**" % (100 * config.GATE_REGIME_MAX_LOSS)
            elif val < config.GATE_REGIME_SOFT_LOSS:
                note = "under the soft floor (%.0f%%)" % (100 * config.GATE_REGIME_SOFT_LOSS)
        rows.append("| %s .. %s | %s | %s | %s |"
                    % (lo, hi,
                       "n/a" if val is None or val != val else "%+.1f%%" % (100 * val),
                       "—" if n is None else "%d" % n, note))
    return rows


def _parents(value) -> str:
    """A parent field renders as one hash, or as both parents of a crossover
    (evolution stores a list there), never as a repr'd Python list."""
    if not value:
        return "—"
    if isinstance(value, (list, tuple)):
        return " x ".join("`%s`" % v for v in value) or "—"
    return "`%s`" % value


def hof_rows(state_dir=None) -> list:
    hof = evolution.load_hall_of_fame(state_dir)
    rows = []
    for i, r in enumerate(hof[:config.HOF_SIZE], 1):
        rows.append("| %d | `%s` | %s | %+.3f | %d | %d | %s | %s |"
                    % (i, r.get("hash", ""), r.get("family", ""),
                       float(r.get("sharpe_prevault", float("nan"))),
                       int(r.get("generation", 0)), int(r.get("birth_gen", 0)),
                       r.get("op", ""), _parents(r.get("parent_hash"))))
    return rows


def cost_lines(df: pd.DataFrame) -> list:
    """Cost share of gross — DESIGN's standing small-account tripwire.

    Pre-vault rows only: the artifact stores gross/turnover/costs for those alone,
    so the vault segment cannot answer this question and is not asked.
    """
    pre = df[df["segment"] == "prevault"]
    if not len(pre):
        return ["No pre-vault rows in the artifact — cost share cannot be computed."]
    net = pre["net"].to_numpy(dtype=np.float64)
    gross = pre["gross"].to_numpy(dtype=np.float64)
    drag = gross - net                                # exactly the day's friction
    ann_bps = 1e4 * np.nanmean(drag) * config.TRADING_DAYS_YEAR
    sum_gross = float(np.nansum(gross))
    share = (float(np.nansum(drag)) / sum_gross) if sum_gross > 0 else float("nan")
    lines = [
        "- Gross cumulative return **%+.1f%%**, net **%+.1f%%** over %d pre-vault days."
        % (100 * (np.nanprod(1.0 + gross) - 1.0), 100 * (np.nanprod(1.0 + net) - 1.0),
           len(pre)),
        "- Cost drag **%.0f bps/year** of equity; mean daily turnover **%.1f%%** of equity."
        % (ann_bps, 100 * float(np.nanmean(pre["turnover"].to_numpy(dtype=np.float64)))),
        "- Total frictions paid: **${:,.0f}** on a ${:,.0f} account.".format(
            float(np.nansum(pre["costs"].to_numpy())), config.START_CASH),
    ]
    lines.append("- **Cost share of gross: %s**%s"
                 % ("%.1f%%" % (100 * share) if share == share else "n/a",
                    "" if share == share else
                    " (gross return over the window is not positive, so the ratio "
                    "would not mean what it says)"))
    return lines


# ── the report ─────────────────────────────────────────────────────────────────
def build_report(generation=None, state_dir=None, artifact_dir=None,
                 output_dir=None) -> str:
    """Write output/report_gen<NNNN>.md (+ PNGs). Returns the markdown path."""
    import run_deepeval

    generation = int(generation if generation is not None
                     else latest_reported_generation(state_dir))
    out_dir = output_dir or config.OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "report_gen%04d.md" % generation)

    subject = report_subject(generation, state_dir, artifact_dir)
    history = [r for r in run_deepeval.read_history(state_dir)
               if int(r["generation"]) == generation]
    row = history[-1] if history else None

    L = ["# arena — generation %d" % generation, ""]

    if subject is None:
        L += ["## No evaluated genome to report on yet",
              "",
              "There is no champion, and no deep-eval artifact for generation %d. "
              "That is the honest state of the system: generations have been run "
              "and scored pre-vault, but nothing has been through the ten "
              "promotion gates, so there is nothing here that would survive being "
              "called a result." % generation,
              ""]
        L += _ledger_section(state_dir)
        L += ["", "## Hall of fame (pre-vault, ungated)", "",
              "| # | genome | family | pre-vault SR | gen | born | op | parent |",
              "|---|--------|--------|-------------:|----:|-----:|----|--------|"]
        L += hof_rows(state_dir) or ["| — | — | — | — | — | — | — | — |"]
        L += ["", "*Ranked by pre-vault Sharpe. The ranking is the SEARCH's own "
              "opinion of itself and is biased upward by construction.*", ""]
        L += [FOOTER]
        with open(path, "w") as f:
            f.write("\n".join(L))
        return path

    record = subject["record"] or {}
    f2 = record.get("f2") or {}
    f1 = record.get("f1") or {}
    df = load_returns(subject["hash"], record, artifact_dir)
    window = record.get("window") or []
    asof = str(pd.DatetimeIndex(df["date"])[-1].date()) if len(df) else (
        window[1] if len(window) > 1 else "unknown")

    # ── header ────────────────────────────────────────────────────────────────
    if subject["role"] == "champion":
        L += ["**Champion `%s`** — promoted through all ten gates." % subject["hash"], ""]
        L += resim_note(subject)
    else:
        L += ["**No champion.** Nothing has passed all ten promotion gates, so this "
              "system currently recommends nothing and holds nothing.", "",
              "This report is about `%s`, the leading candidate of the last deep "
              "evaluation, **which the gates REFUSED** (%s). It is shown because a "
              "refusal with its numbers attached is more useful than a blank page — "
              "not because it is close to being a champion."
              % (subject["hash"],
                 "failed " + (row.get("gates_failed") or "?") if row else "see the table below"),
              ""]

    ident = record.get("identity") or {}
    L += ["| | |", "|---|---|",
          "| data as of | %s |" % asof,
          "| evaluation window | %s |" % (" .. ".join(window) if window else "n/a"),
          "| vault window | %s |" % (" .. ".join(record.get("vault_window") or []) or "n/a"),
          "| identity | data `%s` · panel `%s` · config `%s` |"
          % (ident.get("data_hash", "?"), ident.get("panel_hash", "?"),
             ident.get("config_hash", "?")),
          "| platform | %s%s |" % (ledger.platform_tag(),
                                   " (vendored siblings)" if config.VENDORED else ""),
          "| family | %s |" % (f2.get("family") or "?"),
          ""]

    # ── charts ────────────────────────────────────────────────────────────────
    if len(df):
        pngs = draw_charts(subject, df, generation, out_dir)
        L += ["## Equity, rolling Sharpe, drawdown", ""]
        for name, caption in zip(pngs, (
                "Net equity against %s buy-and-hold. The strategy line is net of "
                "every modelled friction; the benchmark line is not, which flatters "
                "the benchmark and is the harder comparison to win." % config.BENCHMARK,
                "Rolling %d-year net Sharpe. Flat stretches below zero are what a "
                "single headline Sharpe hides." % config.GATE_ROLLING_WINDOW_YEARS,
                "Drawdown from the running peak, strategy and benchmark.")):
            L += ["![%s](%s)" % (name, name), "", "*%s*" % caption, ""]
        L += ["*Everything left of the dotted vault line was available to selection; "
              "everything right of it was touched only by the promotion gates.*", ""]

    # ── headline numbers ──────────────────────────────────────────────────────
    L += ["## Headline numbers", "",
          "| statistic | value |", "|---|---|",
          "| pre-vault net Sharpe | %s |"
          % _f(_first(f2.get("sharpe"), f1.get("sharpe_prevault")), "%+.3f"),
          "| pre-vault days scored | %s |" % (f2.get("n_days_prevault") or f1.get("n_days_prevault") or "n/a"),
          "| vault net Sharpe | %s |" % _f(f2.get("vault_sharpe"), "%+.3f"),
          "| vault days | %s |" % (f2.get("vault_days") or "n/a"),
          "| Sharpe at %.0fx costs | %s |" % (config.GATE_STRESS_MULT, _f(f2.get("sharpe_stress"), "%+.3f")),
          "| bootstrap %.0f%% CI of net Sharpe | [%s, %s] |"
          % (100 * config.GATE_BOOT_CI, _f(f2.get("boot_ci_lo"), "%+.3f"),
             _f(f2.get("boot_ci_hi"), "%+.3f")),
          "| CPCV paths net-positive | %s of %s (median path SR %s) |"
          % ("%.0f%%" % (100 * f2["cpcv_frac_positive"]) if f2.get("cpcv_frac_positive") is not None else "n/a",
             f2.get("cpcv_n_paths", "n/a"), _f(f2.get("cpcv_median_sharpe"), "%+.2f")),
          "| P(drawdown > %.0f%% in %d years) | %s |"
          % (100 * config.GATE_RUIN_DD, config.RUIN_MC_YEARS, _f(f2.get("p_ruin"), "%.3f")),
          ""]

    # DSR is never quoted without its N, and PBO never without its cohort.
    dsr_detail = f2.get("dsr_detail") or {}
    L += ["### Deflated Sharpe and PBO", "",
          "- **DSR %s at N = %s ledger trials, %s vault trials** (sr0 threshold %s, "
          "T = %s days, skew %s, kurtosis %s). The deflation uses the EMPIRICAL "
          "spread of trial Sharpes from the ledger, never a hardcoded count."
          % (_f(f2.get("dsr"), "%.4f"), f2.get("dsr_n_trials", "?"),
             f2.get("vault_trials", "?"), _f(dsr_detail.get("sr0_threshold"), "%.4f"),
             dsr_detail.get("T", "?"), _f(dsr_detail.get("skew"), "%.3f"),
             _f(dsr_detail.get("kurtosis"), "%.3f")),
          "- Vault DSR **%s** at N = %s vault trials — the count of times any "
          "candidate has been shown the post-%s data at all."
          % (_f(f2.get("vault_dsr"), "%.4f"), f2.get("vault_trials", "?"),
             config.VAULT_START),
          # THREE STATES, not two. "In the cohort" and "outside it" are both
          # results of a battery that RAN; a record with no battery behind it at
          # all (the champion's weekly re-simulation) has neither, and printing
          # "G4 fails on that alone" about it would invent a failed gate.
          "- **PBO %s** (CSCV, %s) — %s"
          % (_f(f2.get("pbo"), "%.3f"), f2.get("pbo_note", "cohort unknown"),
             ("computed over the returns matrix of generation %s, the cohort this "
              "record belongs to." % record.get("generation", "?")
              if f2.get("pbo_in_cohort")
              else "**this candidate is not in the cohort matrix, so PBO is unmeasured "
                   "and gate G4 fails on that alone.**") if _battery_ran(f2)
             else "no battery was run on this record, so there is no PBO to report "
                  "and no G4 verdict in it either way."),
          "- DSR is an OPTIMISTIC correction here: evolutionary trials are "
          "correlated, and correlated trials deflate less than independent ones "
          "would. The vault and the paper stage sit above it for exactly that reason.",
          ""]

    if f2.get("ledger_drift"):
        d = f2["ledger_drift"]
        L += ["> **Ledger drift on this genome.** Its best-fidelity ledger row is "
              "from an earlier vintage (data `%s`), so the trial-Sharpe dispersion "
              "that sets the DSR threshold is that vintage's, in an uncontrolled "
              "direction. Recorded here rather than buried in a log."
              % (d.get("prior_identity") or ["?"])[0], ""]

    # ── the gates ─────────────────────────────────────────────────────────────
    L += ["## The ten promotion gates", "",
          "| gate | what it asks | value | threshold | |",
          "|---|---|---|---|---|"]
    L += gate_rows(record)
    if row:
        L += ["", "Deep-eval history row: promoted=%s, candidates evaluated=%s, "
              "gates failed=%s, complete=%s."
              % (row.get("promoted"), row.get("n_candidates"),
                 row.get("gates_failed") or "none", row.get("complete"))]
        if row.get("ledger_drift"):
            L += ["", "Ledger drift recorded for this decision: `%s`." % row["ledger_drift"]]
    L += ["", "*All ten must pass; ties go to the incumbent. A single failure is a "
          "refusal, and a refusal is the system working.*", ""]

    # ── regime slices and costs ───────────────────────────────────────────────
    L += ["## Crisis regimes (gate G8)", "",
          "| window | net return | days | |", "|---|---:|---:|---|"]
    L += regime_rows(record)
    L += ["", "*Windows are drawdown LEGS — peak to trough, not peak to recovery. A "
          "window containing the rebound measures the wrong thing: 2008-01..2009-12 "
          "comes out positive for a book that was destroyed in the autumn of 2008.*",
          "",
          "## Costs", ""]
    L += cost_lines(df)
    L += [""]

    # ── ledger, hall of fame, paper ───────────────────────────────────────────
    L += _ledger_section(state_dir)
    L += ["", "## Hall of fame (top %d, pre-vault and ungated)" % config.HOF_SIZE, "",
          "| # | genome | family | pre-vault SR | gen | born | op | parent |",
          "|---|--------|--------|-------------:|----:|-----:|----|--------|"]
    L += hof_rows(state_dir) or ["| — | — | — | — | — | — | — | — |"]
    L += ["", "*Lineage columns are how a genome got here: which operator made it, "
          "from which parent, in which generation.*", "",
          "## Simulation vs paper trading", "",
          "**Paper trading has not started.** The graduation rule is three "
          "consecutive weekly deep evals kept by the same champion (docs/DESIGN.md, "
          "\"Graduation ladder\"); there is %s. Until then every number in this "
          "report is a simulation of the past, and the sim-vs-realised overlay this "
          "section will hold does not exist."
          % ("no champion at all" if subject["role"] != "champion"
             else "a champion, but not yet the three-week record"),
          "", FOOTER]

    with open(path, "w") as f:
        f.write("\n".join(L))
    return path


def _ledger_section(state_dir=None) -> list:
    n = ledger.n_trials(state_dir)
    v = ledger.vault_trials(state_dir)
    return ["## Trial ledger", "",
            "- **%d distinct genomes** have been evaluated at some fidelity and are "
            "on the ledger. That is the N every deflated Sharpe above is deflated by." % n,
            "- **%d vault accesses** logged. Every look at post-%s data is counted, "
            "including the ones that only stored an artifact." % (v, config.VAULT_START),
            "- Screens count as trials. They exert selection pressure, so excluding "
            "them would make every DSR on this page optimistic."]


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="build the arena report")   # io-boundary
    ap.add_argument("--gen", type=int, default=None,
                    help="generation to report on (default: the latest with a deep eval)")
    args = ap.parse_args(argv)

    path = build_report(args.gen)
    print("arena report")
    print("  wrote     : %s" % os.path.relpath(path, config.ROOT))
    for name in sorted(os.listdir(config.OUTPUT_DIR)):
        if name.startswith(os.path.basename(path)[:-3]) and name.endswith(".png"):
            print("  chart     : %s" % os.path.join(
                os.path.relpath(config.OUTPUT_DIR, config.ROOT), name))
    with open(path) as f:
        text = f.read()
    print("  size      : %d lines, %.1f kB" % (text.count("\n") + 1, len(text) / 1024.0))
    print("  footer    : %s" % ("present" if HONESTY_LINE in text and
                                SURVIVORSHIP_LINE in text else "MISSING"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
