"""
Alert delivery for arena — one wrapper over signal_lab's notifier, plus the two
things a scheduled cloud job needs that a laptop job does not.

WHAT IS REUSED. The transport (macOS Notification Center, ntfy.sh, Telegram, all
with certifi SSL and retries) is signal_lab/alerts.py, imported through
config.import_sibling so it resolves to the live checkout on this Mac and to
vendor/signal_lab/alerts.py on the runner. Nothing about HTTP is re-implemented
here; the only transport decision this module makes is skipping the osascript
branch off macOS, because an Actions runner has no Notification Center and
subprocess would just print a traceback into the log twice a day.

WHAT IS NEW.

  1. Env-first credentials. GitHub Actions injects secrets as environment
     variables, and the Mac chain (arena/config.local.json ->
     vrp_backtest/monitor_config.json) does not exist on a runner. credentials()
     reads the environment FIELD BY FIELD and falls back to the local chain for
     whatever the environment did not supply, so a half-configured machine sends
     on the channels it can and stays quiet on the rest.

  2. State-transition anti-spam. These jobs run twice a day, 7 days a week, and
     most runs have nothing new to say: the same generation still mid-flight, the
     same candidate refused by the same three gates. state/alert_state.json
     remembers, per alert kind, the STATE that was last delivered — (champion,
     generation, status) — and an identical state sends nothing. This is the
     house pattern (vrp_backtest, deepvalue): anti-spam by comparing against the
     last logged state, never by a timer or an in-process loop, because the
     scripts are one-shot and remember nothing between invocations.

     A dry run never writes that file. Otherwise a `--send`-less rehearsal would
     silently swallow the next real alert, and the one thing an alerting path
     must not do is go quiet for a reason nobody chose.

The composers are pure functions of already-computed numbers — no I/O, no
config reads beyond thresholds — so they can be exercised without a market, a
population, or a network. `python3 alerts_arena.py` prints both formats dry,
including the suppression behaviour, against a temporary state directory.

EVERY BODY ENDS WITH THE HONESTY LINE. Not decoration: an alert is the only part
of this system most people will ever read, and a Sharpe number with no caveat
attached is a claim this project does not make.
"""
from __future__ import annotations

import json
import os
import sys

import config                       # FIRST: puts the siblings on sys.path

# The transport, from whichever signal_lab is in play (live on the Mac, vendored
# on the runner). Loaded by explicit path because arena has no `alerts` of its own
# to clash with, but the sibling's `import config` must still land on arena's.
_alerts = config.import_sibling("alerts", config.SIGNAL_LAB)

ALERT_STATE_FILE = "alert_state.json"

ALERT_STATE_NOTE = ("REWRITTEN on every delivered alert: the last state announced "
                    "per alert kind, so a repeated state stays quiet. Not a record "
                    "of anything — the trial ledger, the deep-eval history and "
                    "champion_history.csv are where decisions live.")

# docs/DESIGN.md, "Risks & honest limitations" — the core honesty statement and the
# survivorship disclosure, verbatim in every user-facing output.
HONESTY_LINE = ("Backtest alpha is a claim about the past, not a guarantee — "
                "not financial advice.")
SURVIVORSHIP_LINE = ("Survivorship: the universe is today's S&P 500 membership, so "
                     "long results flatter and short results understate.")


# ── credentials ────────────────────────────────────────────────────────────────
def credentials() -> dict:
    """Notifier secrets, environment first, local chain second, field by field.

    TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID / NTFY_TOPIC are the Actions path (repo
    secrets -> workflow env). config.load_credentials() is the Mac path. Neither
    is required: a channel with no credential is simply not used.

    Returns the shape signal_lab's alerts.py expects, and NEVER logs a value.
    """
    env_topic = os.environ.get("NTFY_TOPIC")                        # io-boundary
    env_token = os.environ.get("TELEGRAM_BOT_TOKEN")                # io-boundary
    env_chat = os.environ.get("TELEGRAM_CHAT_ID")                   # io-boundary

    local = {"ntfy_topic": None, "telegram": {}}
    if not (env_topic and env_token and env_chat):
        try:
            local = config.load_credentials()
        except Exception:
            pass
    tg = local.get("telegram") or {}
    return {"ntfy_topic": env_topic or local.get("ntfy_topic"),
            "telegram": {"bot_token": env_token or tg.get("bot_token"),
                         "chat_id": env_chat or tg.get("chat_id")}}


def channels(creds=None) -> list:
    """Names of the channels that would actually receive a send. Never values."""
    creds = creds if creds is not None else credentials()
    tg = creds.get("telegram") or {}
    out = []
    if sys.platform == "darwin":
        out.append("macos")
    if creds.get("ntfy_topic"):
        out.append("ntfy")
    if tg.get("bot_token") and tg.get("chat_id"):
        out.append("telegram")
    return out


# ── delivery ───────────────────────────────────────────────────────────────────
class NoChannel(RuntimeError):
    """A real send was asked for and there was nowhere to send it.

    Raised only when the caller opts in (`require_delivery=True`), because most
    alerts would rather be silent than fail a job that otherwise succeeded. The
    push-failure notice is the exception: it exists to make a lost session loud,
    so a version of it that prints "delivered to: nothing" and exits 0 is the
    failure it was written to prevent.
    """


def send_all(title: str, body: str, dry_run: bool = True,
             require_delivery: bool = False) -> list:
    """Print the exact text, then deliver it unless this is a dry run.

    Returns the list of channels delivered to (empty on a dry run). The printing
    is not a debug aid — a dry run's whole output IS the alert, and the scheduled
    job's log is where a human checks what was said.

    `require_delivery` turns "nowhere to send it" into a NoChannel exception. See
    that class: it is for the alerts whose whole purpose is to be heard.
    """
    print("\n=== ALERT %s ===" % ("(DRY RUN — not sent)" if dry_run else "(SENDING)"))
    print(title)
    print(body)
    if dry_run:
        return []

    creds = credentials()
    used = channels(creds)
    if not used and require_delivery:
        raise NoChannel(
            "no alert channel is configured, so this message reached nobody. On a "
            "GitHub runner that means the three secrets did not reach this step — "
            "step-level `env:` does not carry between steps, so they must be set "
            "at JOB level. Names checked: NTFY_TOPIC, TELEGRAM_BOT_TOKEN, "
            "TELEGRAM_CHAT_ID (values are never read into a log).")
    if "macos" in used:
        # Off macOS there is no Notification Center; osascript would fail on every
        # scheduled run and say nothing useful when it did.
        _alerts.macos_notify(title, body)
    if "ntfy" in used:
        _alerts.ntfy_notify(creds["ntfy_topic"], title, body)
    if "telegram" in used:
        _alerts.telegram_notify(creds["telegram"]["bot_token"],
                                creds["telegram"]["chat_id"], title, body)
    print("  delivered to: %s" % (", ".join(used) if used else
                                  "nothing (no channel is configured on this machine)"))
    return used


# ── state-transition anti-spam ─────────────────────────────────────────────────
def alert_state_path(state_dir=None) -> str:
    return os.path.join(state_dir or config.STATE_DIR, ALERT_STATE_FILE)


def load_alert_state(state_dir=None) -> dict:
    """The whole file, or an empty dict. A corrupt file reads as empty: the
    failure mode of forgetting is a duplicate alert, and the failure mode of
    raising is a scheduled job that dies at the last step with its work done."""
    path = alert_state_path(state_dir)
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            payload = json.load(f)
        return payload.get("kinds", {})
    except Exception:
        return {}


def _save_alert_state(kinds: dict, state_dir=None) -> str:
    path = alert_state_path(state_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"note": ALERT_STATE_NOTE, "kinds": kinds}, f, indent=1, sort_keys=True)
    os.replace(tmp, path)
    return path


def is_new_state(kind: str, state: dict, state_dir=None) -> bool:
    """True when `state` differs from the one last DELIVERED under `kind`."""
    last = load_alert_state(state_dir).get(kind)
    return json.dumps(last, sort_keys=True) != json.dumps(state, sort_keys=True)


def send_transition(kind: str, state: dict, title: str, body: str,
                    dry_run: bool = True, state_dir=None) -> bool:
    """Deliver only on a state transition. Returns True if an alert was delivered.

    `state` is the small dict that decides sameness — champion, generation, status
    — and nothing that moves on its own (no timestamps, no elapsed seconds), or
    every run would look like a transition and the anti-spam would be decorative.
    """
    new = is_new_state(kind, state, state_dir)
    if not new:
        print("\n=== ALERT SUPPRESSED (%s) — state unchanged since the last one sent: %s"
              % (kind, json.dumps(state, sort_keys=True)))
        return False
    send_all(title, body, dry_run=dry_run)
    if dry_run:
        print("  (dry run: state/%s not updated, so the real send is still armed)"
              % ALERT_STATE_FILE)
        return False
    kinds = load_alert_state(state_dir)
    kinds[kind] = state
    _save_alert_state(kinds, state_dir)
    return True


# ── composers (pure) ───────────────────────────────────────────────────────────
def _sr(value) -> str:
    """A Sharpe, or a dash. None and NaN both mean "no number", and printing
    'nan' in an alert reads as a bug rather than as an absence."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "  n/a"
    return "  n/a" if f != f else "%+.2f" % f


def generation_summary(generation: int, n_evaluated: int, n_trials: int,
                       best_sharpe=None, median_sharpe=None, champion=None,
                       status: str = "complete", platform: str = "",
                       detail: str = "") -> tuple:
    """The nightly one-liner of docs/DESIGN.md, plus the caveat it needs.

    status  "complete" | "incomplete" | "stale-data" | "identity-drift" — the
            kill-state half of the anti-spam key. A generation that is still
            mid-flight after a second session is the SAME state and says nothing;
            the same generation reaching "complete" is a transition and does.
    """
    head = "arena gen %d — %s" % (generation, status)
    title = "%s: %d evaluated, best SR %s" % (head, n_evaluated, _sr(best_sharpe))

    lines = ["gen %d  |  %d genomes evaluated this run  |  ledger %d distinct trials"
             % (generation, n_evaluated, n_trials)]
    if best_sharpe is not None or median_sharpe is not None:
        lines.append("F1 net Sharpe (pre-vault): best %s   median %s"
                     % (_sr(best_sharpe), _sr(median_sharpe)))
    lines.append("champion: %s" % (champion or "none yet — no candidate has passed "
                                               "all ten gates"))
    if platform:
        lines.append("platform: %s" % platform)
    if detail:
        lines.append("")
        lines.append(detail)
    lines += ["",
              "Pre-vault and ungated: the population was SEARCHED, so its best is "
              "biased upward by construction — that is what the trial ledger, DSR, "
              "PBO, the vault and the gate stack exist to discount.",
              SURVIVORSHIP_LINE,
              HONESTY_LINE]
    return title, "\n".join(lines)


def deepeval_summary(generation: int, candidates, promoted=None, champion_before=None,
                     dsr=None, n_trials=None, vault_trials=None, status: str = "refused",
                     platform: str = "", detail: str = "") -> tuple:
    """The weekly decision, promotion or refusal, with the failed gates named.

    candidates  [(hash, passed: bool, [failed gate ids])] in the order evaluated.
                A refusal that does not say WHICH gates refused is a shrug, and
                the gates are the entire reason this project's numbers are worth
                anything.
    """
    if promoted:
        title = "arena deep eval gen %d: PROMOTED %s" % (generation, promoted)
    elif status == "nothing-to-evaluate":
        # NOT a refusal. Nothing was put to the gates, so nothing was turned down,
        # and a title saying "REFUSED — 0 candidates" would report a decision that
        # was never made.
        title = ("arena deep eval gen %d: nothing to evaluate — no decision made"
                 % generation)
    elif status == "incomplete":
        title = "arena deep eval gen %d: INCOMPLETE — nothing promoted" % generation
    else:
        title = ("arena deep eval gen %d: REFUSED — %d candidate(s), 0 promoted"
                 % (generation, len(candidates)))

    lines = []
    for ghash, passed, failed in candidates:
        lines.append("%s  %s" % (ghash, "PASSED all ten gates" if passed else
                                 "failed %s" % ("+".join(failed) or "?")))
    if dsr is not None:
        # DSR is never quoted without its N: it is a correction FOR the number of
        # trials, so the number of trials is half the statistic.
        lines.append("DSR %s at N=%s ledger trials%s"
                     % (dsr, n_trials if n_trials is not None else "?",
                        ", %s vault trials" % vault_trials if vault_trials is not None else ""))
    lines.append("champion: %s%s"
                 % (champion_before or "none",
                    " -> %s" % promoted if promoted else " (unchanged)"))
    if platform:
        lines.append("platform: %s" % platform)
    if detail:
        lines.append("")
        lines.append(detail)
    lines += ["",
              "Passing the gates is risk REDUCTION, not proof: correlated "
              "evolutionary trials leave the ledger DSR under-deflated, and PBO "
              "does not cover designer-level choices.",
              SURVIVORSHIP_LINE,
              HONESTY_LINE]
    return title, "\n".join(lines)


def paper_summary(status: str, champion=None, date=None, sim_net=None, paper_net=None,
                  tracking_error_bps=None, slippage_bps=None, n_days=None,
                  arm_consecutive=None, detail: str = "") -> tuple:
    """The two things the paper stage has to say out loud (docs/DESIGN.md ladder).

    status  "armed"           the arming gate closed for the first time: this
                              champion has survived PAPER_ARM_CONSECUTIVE
                              consecutive complete deep evals and run_paper.py
                              may now submit orders. A state transition worth a
                              phone buzz precisely because nothing else in the
                              system announces that trading became possible.
            "tracking-breach" a session's |paper - sim-shadow| exceeded
                              config.PAPER_MAX_TE_BPS. Sent once per breach
                              SPELL, not once per day inside one (run_paper keys
                              the anti-spam on the spell's first date), because a
                              five-day divergence is one piece of news.
            "go-live-candidate"
                              every measurable criterion in DESIGN's ladder is
                              met. A RECOMMENDATION AND NOTHING ELSE: the body
                              says in as many words that the system cannot act
                              on it, because a human flipping EXECUTION_MODE is
                              the entire safety property here.

    Pure: every number is passed in.
    """
    ghash = champion or "?"
    if status == "armed":
        title = "arena paper stage ARMED: %s" % ghash
        lines = ["champion %s has held the pointer through %s consecutive complete "
                 "deep evals, and deepeval_history records it passing all ten gates."
                 % (ghash, arm_consecutive if arm_consecutive is not None else "?"),
                 "run_paper.py may now submit market-on-open orders to the ALPACA "
                 "PAPER account. No real money is involved and none can be: the "
                 "adapter hardcodes paper=True.",
                 "",
                 "This is the start of measurement, not a verdict. The paper stage "
                 "runs for at least %s trading days before the evidence table is "
                 "even worth reading, and going live is a human decision nothing "
                 "here can make." % (n_days if n_days is not None else "?")]
    elif status == "go-live-candidate":
        title = "arena paper: GO-LIVE CANDIDATE %s — a human decision" % ghash
        lines = ["%s has now met every measurable criterion in the graduation "
                 "ladder over %s paper sessions: median |fill slippage| %s bps, "
                 "tracking inside tolerance, and a paper path that tracks its "
                 "sim-shadow." % (ghash, n_days if n_days is not None else "?",
                                  _bps(slippage_bps)),
                 "",
                 "NOTHING HAS BEEN DONE. This system does not go live by itself and "
                 "has no code that could: the only broker adapter it owns hardcodes "
                 "paper=True, and EXECUTION_MODE is a setting a person edits. Read "
                 "the evidence table in the session log before deciding anything.",
                 "",
                 "The criteria are necessary, not sufficient. They say the executed "
                 "book resembled the simulated one — not that the edge is real."]
    else:
        title = ("arena paper: tracking error %s bps%s"
                 % (_bps(tracking_error_bps), " on %s" % date if date else ""))
        lines = ["champion %s%s" % (ghash, " | %s" % date if date else ""),
                 "sim-shadow %s   paper %s   difference %s bps"
                 % (_pct(sim_net), _pct(paper_net), _bps(tracking_error_bps))]
        if slippage_bps is not None:
            lines.append("median fill slippage %s bps vs the day's open" % _bps(slippage_bps))
        lines += ["",
                  "The paper account and the simulator were handed the same target "
                  "weights on the same day and did not end it in the same place. "
                  "That gap is the whole point of the paper stage: it is the part "
                  "of the backtest that was never true."]
    if detail:
        lines += ["", detail]
    lines += ["", SURVIVORSHIP_LINE, HONESTY_LINE]
    return title, "\n".join(lines)


def _bps(value) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if f != f else "%+.1f" % f


def _pct(value) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if f != f else "%+.3f%%" % (100.0 * f)


def push_failure_summary(job: str, attempts: int, ref: str = "", run_url: str = "") -> tuple:
    """A cloud session that computed real work and could not push it home.

    Loud by construction and NOT subject to the anti-spam: the repository is the
    only place an Actions runner's work survives, so a failed push means ledger
    rows for evaluations that actually happened have been lost. "Every genome
    evaluation of any fidelity appends to the trial ledger" is a constraint of
    this project — a silent violation of it is the worst outcome available here,
    because the lost rows are what deflate every Sharpe the system reports.
    """
    title = "arena %s: STATE NOT PUSHED after %d attempts" % (job, attempts)
    body = "\n".join([
        "The run completed and could not push its state back%s."
        % (" to %s" % ref if ref else ""),
        "",
        "The runner is ephemeral, so that work is GONE: any trial-ledger rows it "
        "wrote are lost, and n_trials() is now lower than the number of genomes "
        "this search has actually looked at — which makes every deflated Sharpe "
        "reported from here optimistic until those genomes are evaluated again.",
        "",
        "They will be: the ledger is idempotent on (genome, generation, fidelity), "
        "so re-running the generation re-appends them. Nothing on the remote is "
        "wrong; it is only missing.",
    ] + (["", "Run log: %s" % run_url] if run_url else []) + ["", HONESTY_LINE])
    return title, body


def data_stale_summary(cache_age: float, bar_age: float, last_bar, limit: int,
                       job: str = "generation") -> tuple:
    """The abort DESIGN asks for by name: 'abort with alert if staleness > 5 days'."""
    title = "arena %s ABORTED: data is %.1f days stale" % (job, max(cache_age, bar_age))
    body = "\n".join([
        "cache %.1f days old, last bar %s (%.1f days old); limit is %d."
        % (cache_age, last_bar, bar_age, limit),
        "Nothing was evaluated. A stale sandbox scores genomes on a market that no "
        "longer exists, which is worse than not scoring them at all.",
        "",
        HONESTY_LINE])
    return title, body


if __name__ == "__main__":
    import argparse
    import tempfile

    ap = argparse.ArgumentParser(description="arena alert smoke / push-failure notice")
    ap.add_argument("--push-failed", action="store_true",
                    help="compose and deliver the commit-back failure notice "
                         "(the workflows call this when the push loop gives up)")
    ap.add_argument("--job", default="generation", help="which job failed to push")
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--ref", default="")
    ap.add_argument("--run-url", default="")
    ap.add_argument("--send", action="store_true", help="deliver; default is dry")
    ap.add_argument("--channels", action="store_true",
                    help="print which channels resolve here and exit (NAMES only — "
                         "no credential value is ever printed)")
    _args = ap.parse_args()                                          # io-boundary

    if _args.channels:
        _c = credentials()
        _ch = channels(_c)
        print("channels: %s" % (", ".join(_ch) or "NONE"))
        print("presence: ntfy_topic=%s telegram_bot_token=%s telegram_chat_id=%s"
              % (bool(_c["ntfy_topic"]), bool(_c["telegram"].get("bot_token")),
                 bool(_c["telegram"].get("chat_id"))))
        raise SystemExit(0 if _ch else 1)

    if _args.push_failed:
        _t, _b = push_failure_summary(_args.job, _args.attempts, _args.ref, _args.run_url)
        # send_all, not send_transition: a lost session is never "the same state as
        # last time", and the state file it would consult is on the disk that is
        # about to be discarded.
        #
        # require_delivery with --send: this notice exists to be HEARD. Printing it
        # to a log nobody is watching and exiting 0 would reproduce, one layer up,
        # exactly the silent loss it is reporting.
        try:
            send_all(_t, _b, dry_run=not _args.send, require_delivery=_args.send)
        except NoChannel as exc:
            print("ALERT UNDELIVERABLE: %s" % exc)
            raise SystemExit(2)
        raise SystemExit(0)

    print("arena alerts")
    creds = credentials()
    print("  channels    : %s" % (", ".join(channels(creds)) or "none configured"))
    print("  credentials : ntfy=%s telegram_token=%s telegram_chat=%s  (presence only, "
          "never values)" % (bool(creds["ntfy_topic"]), bool(creds["telegram"]["bot_token"]),
                             bool(creds["telegram"]["chat_id"])))

    t1, b1 = generation_summary(generation=2, n_evaluated=12, n_trials=84,
                                best_sharpe=0.83, median_sharpe=0.21, champion=None,
                                status="complete", platform="x86_64linux")
    t2, b2 = deepeval_summary(generation=2,
                              candidates=[("9353d613c7d3", False, ["G2", "G4", "G8"]),
                                          ("35d85a0408d2", False, ["G2", "G4", "G8"])],
                              promoted=None, champion_before=None, dsr="0.046",
                              n_trials=84, vault_trials=4, status="refused",
                              platform="x86_64linux")
    t3, b3 = data_stale_summary(6.2, 6.0, "2026-08-11", config.MAX_DATA_STALENESS_DAYS)
    t4, b4 = paper_summary("armed", champion="35d85a0408d2",
                           arm_consecutive=config.PAPER_ARM_CONSECUTIVE,
                           n_days=config.PAPER_MIN_DAYS)
    t5, b5 = paper_summary("tracking-breach", champion="35d85a0408d2", date="2026-08-17",
                           sim_net=0.0041, paper_net=-0.0002,
                           tracking_error_bps=-43.0, slippage_bps=6.2)

    tmp = tempfile.mkdtemp(prefix="arena_alert_")
    print("\n--- composer 1: the nightly generation line ---")
    send_all(t1, b1, dry_run=True)
    print("\n--- composer 2: the weekly deep-eval decision ---")
    send_all(t2, b2, dry_run=True)
    print("\n--- composer 3: the staleness abort ---")
    send_all(t3, b3, dry_run=True)
    print("\n--- composer 4: the paper stage arming ---")
    send_all(t4, b4, dry_run=True)
    print("\n--- composer 5: a paper tracking-error breach ---")
    send_all(t5, b5, dry_run=True)

    # Anti-spam, end to end, against a throwaway state dir: a first state is
    # delivered, the identical state is silent, a moved state speaks again.
    print("\n--- state-transition anti-spam (delivery simulated: no channel is "
          "contacted because this run writes state only) ---")
    state_a = {"champion": "", "generation": 2, "status": "incomplete"}
    state_b = {"champion": "", "generation": 2, "status": "complete"}
    kinds = load_alert_state(tmp)
    kinds["generation"] = state_a
    _save_alert_state(kinds, tmp)
    same = is_new_state("generation", state_a, tmp)
    moved = is_new_state("generation", state_b, tmp)
    print("  after recording %s:" % json.dumps(state_a, sort_keys=True))
    print("    identical state is new? %s   (expected False — the re-trigger stays quiet)"
          % same)
    print("    completed state is new? %s   (expected True  — the transition speaks)"
          % moved)
    print("  smoke: %s" % ("PASS" if (not same and moved) else "FAIL"))
    print("  state file  : %s" % alert_state_path(tmp))
    for line in (b1, b2, b3, b4, b5):
        assert line.rstrip().endswith(HONESTY_LINE), "a body reached a channel without the footer"
    print("  every body ends with the honesty line: PASS")
