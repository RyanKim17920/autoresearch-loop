"""loop.digest -- v3: :func:`write_report` (below) writes the 3-HOUR MILESTONE DIGEST from each meta report (the meta cycle
calls it; ``<state>/digests/digest-<ts>.md`` + ``<state>/digest.md``): new incumbent, tests, best posterior, top new
results, alerts. The v2 DAILY digest (``numbers`` / ``tick``, read by the ``digest`` CLI only) follows.

v2 daily digest (no LLM): once per ``EVERY_S`` the driver writes
``<state_dir>/digest.md`` (and ``digests/digest-<date>.md``) and logs a ``digest`` event with the numbers:

* milestones: confirmations, audits, incumbent changes, Labless submissions, LLM pauses;
* LLM (Opus) calls by role: research sessions started (and how many ended in an outage), reviews by trigger,
  scout runs, playbook consolidations, skipped duplicate jobs (loop.redundancy);
* GPU-h by lane (runs finished in the window): ideas, controls, audit / re-mint, design, tuning / ladders,
  backfill, eval-only;
* idle: GPU-h the slots could have run but did not (capacity x hours - used, approximate);
* operator proposals still open;
* the data bank (loop.databank): registered datasets and PRUNE CANDIDATES (no open / confirmed node trains on
  them, ready more than ``databank.retain_days`` ago) -- listed, never deleted."""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from loop.contracts import NodeStatus, Stage, Strategy

K_DIGEST = "digest.last"            # {"ts": last digest}
EVERY_S = 86400.0
_CONTROL = {Stage.BASELINE.value, Stage.PARENT.value, Stage.CALIBRATE.value}
_HELD_OUT = {Stage.AUDIT.value, Stage.TIEBREAK.value, Stage.REMINT.value}


def lane(run: Any, node: Any) -> str:
    """The GPU lane a run belongs to."""
    from loop.evalonly import is_eval_run
    notes = getattr(node, "notes", "") or ""
    if is_eval_run(run):
        return "eval-only"
    if run.stage.value in _CONTROL:
        return "controls"
    if run.stage.value in _HELD_OUT:
        return "audit / re-mint"
    if "tune=" in notes or "tune_of=" in notes:
        return "tuning / ladders"
    if node is not None and node.strategy is Strategy.DESIGN:
        return "design / legs"
    if "backfill_of=" in notes or "retry_of=" in notes:
        return "backfill"
    return "ideas"


def numbers(ctx: Any, store: Any, now: float, window: float = EVERY_S) -> dict[str, Any]:
    """Every number the digest prints, over ``[now - window, now)``."""
    from loop import selftune as T
    lo = now - window
    ev = [e for e in store.events() if lo <= float(e.get("ts") or 0.0) < now]
    kinds = Counter(e["event"] for e in ev)
    to = Counter(e.get("to") for e in ev if e["event"] == "node_transition")
    nodes = {n.id: n for n in store.nodes()}
    finished = {e.get("run_id") for e in ev if e["event"] == "run_finished"}
    gpu: Counter[str] = Counter()
    for r in store.runs():
        if r.id in finished and r.gpu_h:
            gpu[lane(r, nodes.get(r.node_id))] += float(r.gpu_h)
    used = sum(gpu.values())
    reviews = Counter(r.trigger for r in store.reviews() if lo <= float(r.created_ts or 0.0) < now)
    cur = int((store.get_state("meta.playbook") or {}).get("version") or 0)        # loop.meta.K_PLAYBOOK
    pbs = sum(1 for k in range(1, cur + 1) if (pb := store.get_state(f"meta.playbook.v{k}")) is not None
              and lo <= float(pb.get("ts") or 0.0) < now)
    from loop import databank as DB
    reg = DB.book(store)["registered"] if getattr(getattr(ctx.cfg, "databank", None), "enabled", False) else {}
    outage = sum(e["event"] == "research_failed" and str(e.get("status")) == "backend_unavailable" for e in ev)
    return {
        "window_h": window / 3600.0,
        "milestones": {"confirmed": to.get(NodeStatus.CONFIRMED.value, 0),
                       "audits": [f"{e.get('node_id')} {e.get('verdict')}" for e in ev if e["event"] == "audit"],
                       "incumbent": [f"{e.get('previous')} -> {e.get('node_id')}" for e in ev if e["event"] == "incumbent"],
                       "labless": [f"{e.get('node_id')} ({e.get('tier')})" for e in ev if e["event"] == "labless_submitted"],
                       "llm_pauses": [str(e.get("kind")) for e in ev if e["event"] == "llm_paused"],
                       "closed": sum(to.values()) - to.get(NodeStatus.CONFIRMED.value, 0)},
        "llm_calls": {"researcher": kinds.get("research_started", 0), "researcher_outage": outage,
                      **{f"review:{k}": v for k, v in sorted(reviews.items())}, "scout": kinds.get("scout_run", 0),
                      "consolidation": pbs, "skipped_redundant": kinds.get("llm_job_skipped", 0)},
        "gpu_h": {k: round(v, 1) for k, v in gpu.most_common()},
        "gpu_h_used": round(used, 1),
        "idle_gpu_h": round(max(0.0, ctx.cfg.capacity * window / 3600.0 - used), 1),
        "proposals_open": len(T.open_proposals(store)),
        "databank": {"registered": sorted(reg), "prune_candidates": DB.prune_candidates(ctx, store, now)},
    }


def render(n: dict[str, Any], now: float) -> str:
    m, c = n["milestones"], n["llm_calls"]
    day = time.strftime("%Y-%m-%d %H:%MZ", time.gmtime(now))
    line = lambda xs: ", ".join(xs) or "none"                       # noqa: E731
    return "\n".join([
        f"# Loop digest {day} (last {n['window_h']:.0f} h, computed)", "",
        "## Milestones",
        f"- confirmed: {m['confirmed']}; other closes: {m['closed']}",
        f"- audits: {line(m['audits'])}", f"- incumbent: {line(m['incumbent'])}",
        f"- Labless: {line(m['labless'])}", f"- LLM pauses: {line(m['llm_pauses'])}", "",
        "## LLM calls by role",
        *(f"- {k}: {v}" for k, v in c.items()), "",
        "## GPU-h by lane (runs finished in the window)",
        *(f"- {k}: {v}" for k, v in n["gpu_h"].items()),
        f"- total {n['gpu_h_used']}; idle ~{n['idle_gpu_h']} GPU-h of the slots", "",
        f"## Operator proposals open: {n['proposals_open']}", "",
        *([f"## Data bank: {line(n['databank']['registered'])} registered",
           f"- prune candidates (unreferenced, old; never deleted): {line(n['databank']['prune_candidates'])}", ""]
          if n.get("databank", {}).get("registered") or n.get("databank", {}).get("prune_candidates") else [])])


def tick(ctx: Any, store: Any, now: float, force: bool = False) -> list[str]:
    """Write the digest when one is due (never fails the loop)."""
    if not force and now - float((store.get_state(K_DIGEST) or {}).get("ts", -1e18)) < EVERY_S:
        return []
    store.set_state(K_DIGEST, {"ts": now})
    try:
        n = numbers(ctx, store, now)
        text = render(n, now)
        out = Path(ctx.state_dir) / "digest.md"
        tmp = out.with_suffix(".md.tmp")
        tmp.write_text(text)
        tmp.rename(out)                                          # atomic on POSIX (same directory)
        arch = Path(ctx.state_dir) / "digests" / f"digest-{time.strftime('%Y-%m-%d', time.gmtime(now))}.md"
        arch.parent.mkdir(parents=True, exist_ok=True)
        arch.write_text(text)
    except Exception as exc:  # noqa: BLE001 - a report is not worth a tick
        store.log("digest_error", error=repr(exc)[:300])
        return [f"digest failed: {exc!r}"[:200]]
    store.log("digest", **n)
    c = n["llm_calls"]
    return [f"digest written ({out}): {n['milestones']['confirmed']} confirmed, LLM {sum(v for k, v in c.items() if k not in ('researcher_outage', 'skipped_redundant'))} calls, "
            f"GPU-h {n['gpu_h_used']} used / ~{n['idle_gpu_h']} idle, {n['proposals_open']} proposals open"]


# ---------------------------------------------------------------------------------------------- v3: the 3-hour digest
K_DIGEST_V3 = "digest.v3"            # {"ts", "incumbent"}: the previous digest's incumbent, to name a change


def report_alerts(rep: Mapping[str, Any]) -> list[str]:
    """What needs a look in one meta report (a dict: ``asdict(MetaReport)`` or its ``.json``)."""
    out: list[str] = []
    try:
        from loop.core.meta import guard_bounds
        gb, g = guard_bounds(), {**{k: (rep.get("calibration") or {}).get(k) for k in ("slope", "coverage80", "rho")},
                                 **{k: (rep.get("tests") or {}).get(k) for k in ("false_pass", "harm_rate", "type1_sim")}}
        out += [f"guard {k} {v:.3g} outside [{gb[k][0]}, {gb[k][1]}]" for k, v in g.items()
                if v is not None and k in gb and not gb[k][0] <= v <= gb[k][1]]
    except Exception:  # noqa: BLE001 - alerts are best effort
        pass
    out += [f"not computed: {k}" for k in (rep.get("errors") or {})]
    if (llm := rep.get("llm") or {}).get("failed"):
        out.append(f"{int(llm['failed'])} LLM session(s) failed in the window")
    if (rep.get("plateau") or {}).get("active"):
        out.append("plateau ACTIVE")
    if beats := (rep.get("bar") or {}).get("beats"):
        out.append("beats the Labless bar: " + "; ".join(beats))
    if fp := (rep.get("fnf") or {}).get("pause"):
        out.append(f"fork-and-flush paused: {fp}")
    for r in ((rep.get("components") or {}).get("rows") or []):
        if r.get("verdict") == "harmful":
            out.append(f"component {r['component']} audited HARMFUL ({r.get('effect')}+-{r.get('se')}, n={r.get('n')})")
    return out


def render_report(rep: Mapping[str, Any], prev_incumbent: str = "") -> str:
    """The compact digest text of one meta report."""
    ts = float(rep.get("ts") or 0.0)
    inc, t, bar = rep.get("incumbent") or "?", rep.get("tests") or {}, rep.get("bar") or {}
    since = float(rep.get("since") or 0.0)
    sw = rep.get("switches") or []
    moved = f"{prev_incumbent} -> {inc}" if prev_incumbent and prev_incumbent != inc else ""
    best = str(bar.get("best") or "none")
    alerts = report_alerts(rep)
    return "\n".join([
        f"# Loop digest {time.strftime('%Y-%m-%d %H:%MZ', time.gmtime(ts))} (window {max(0.0, ts - since) / 3600.0:.1f} h, from the meta report)",
        "",
        "## Incumbent",
        f"- {inc}" + (f"; CHANGED {moved}" if moved else "") + ("; switches in window: " + "; ".join(sw) if sw else ""),
        "",
        "## Tests",
        f"- closed {t.get('closed', 0)} ({t.get('window', 0)} in window), passed {t.get('passed', 0)} "
        f"(rate {float(t.get('pass_rate', 0.0)):.2f}), mean seeds {t.get('mean_n', 0)}",
        "",
        "## Best posterior (external bar)",
        f"- {best}",
        *([f"- {bar['text']}"] if bar.get("text") else []),
        "",
        "## Top new results (best of window, unadjusted: selected on the window's test verdicts, not shrunk)",
        *([f"- {x}" for x in (rep.get("top_new") or [])] or ["- none"]),
        "",
        "## Alerts",
        *([f"- {a}" for a in alerts] or ["- none"]),
        ""])


def write_report(state_dir: str | Path, store: Any, rep: Mapping[str, Any]) -> Path:
    """Write ``digests/digest-<ts>.md`` and ``digest.md`` under ``state_dir`` for one meta report."""
    prev = (store.get_state(K_DIGEST_V3) or {}).get("incumbent", "")
    text = render_report(rep, str(prev or ""))
    from loop import localruns as LR           # the manual-run import's summary (state set only when it is on)
    if lines := LR.digest_lines(store):
        text = text + "\n".join(["## Already measured locally (operator's manual runs; single seed, not CRN-paired)",
                                 *lines, ""])
    ts = float(rep.get("ts") or time.time())
    d = Path(state_dir)
    arch = d / "digests" / f"digest-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime(ts))}.md"
    arch.parent.mkdir(parents=True, exist_ok=True)
    arch.write_text(text)
    tmp = d / "digest.md.tmp"
    tmp.write_text(text)
    tmp.replace(d / "digest.md")
    store.set_state(K_DIGEST_V3, {"ts": ts, "incumbent": rep.get("incumbent") or ""})
    store.log("digest", v=3, path=str(arch), incumbent=rep.get("incumbent") or "")
    return arch
