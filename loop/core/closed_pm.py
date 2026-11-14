"""loop.core.closed_pm -- the computed POSTMORTEM of every closed node (10-22; no LLM).

The Opus postmortem was removed on 10-06 (3b748f8) and v3 wrote none since; only :mod:`loop.core.struct_tune` wrote a
computed one, for its ladders. A node that reaches a CLOSED derived status (screened out, failed its test,
confirmed, crashed, void, vetoed) now gets one cheap row in ``postmortems`` from facts the ledger already holds:

- the node's own panel increment (posterior m +- sd over n paired seeds) and each axis's paired delta,
- the engagement diagnostic's outcome (did the mechanism's declared quantity move along its prediction),
- the tuning points tried on it (ladder / tuning-pass / re-run children) and how many finished,
- a TYPED outcome: ``harmful_engaged`` (the mechanism moved its diagnostic and the panel fell), ``no_engage`` (the
  diagnostic did not move or reversed: the idea was not really tested), ``axis_tradeoff`` (one axis paid, another
  was charged), ``under_tuned`` (a structural change closed negative before its tuning ladder), ``noise`` (inside the
  noise), plus ``positive_unconfirmed`` / ``harmful`` / ``confirmed`` / ``test_fail`` / ``crashed`` / ``void`` /
  ``unmeasured``.

The row is labelled COMPUTED (``mechanism_reasoning`` starts with :data:`MARK`): briefs, the review and the ``history``
tool show it as measured fact, never under the 'written by earlier LLM sessions' disclaimer. The tick calls
:func:`write` once per tick through ``Ideation.ideate`` (it is a no-op once every closed node has its row)."""

from __future__ import annotations

import json
import math
import re
from typing import Any, Mapping, Sequence

from loop.contracts import LeverTags, NodeStatus, Postmortem

MARK = "COMPUTED FACTS (loop.core.closed_pm, no LLM)"
K_PM = "ideation.closed_pm"          # {node: [status, pm id]}
LIMIT = 20                          # rows written per tick (the 10-06 .. 10-22 backlog drains over a few ticks)
Z_STRONG = 2.0                      # |m| / sd for a clear own effect
Z_AXIS = 1.5                        # an axis delta counts as paid / charged beyond this many se
TYPES = ("harmful_engaged", "harmful", "no_engage", "axis_tradeoff", "under_tuned", "noise", "positive_unconfirmed",
         "confirmed", "test_fail", "crashed", "void", "unmeasured")
CLOSED = {NodeStatus.SCREENED_OUT: "screened_out", NodeStatus.NOT_CONFIRMED: "test_fail",
          NodeStatus.CONFIRMED: "confirmed", NodeStatus.CRASHED: "crashed", NodeStatus.VOID: "void",
          NodeStatus.VETOED: "vetoed", NodeStatus.AUDIT_FAILED: "audit_failed"}
_LINK = re.compile(r"(tune_of|block_of|backrun_of|expose_of|retry_of|ladder)=([A-Za-z0-9_@]+)")
HINTS = {
    "harmful_engaged": "the mechanism possibly caused the fall: it moved its diagnostic and the panel fell (a correlation "
                       "at this dose and sign, not an isolated cause); a smaller dose, the opposite sign or a different "
                       "carrier would be new information",
    "harmful": "a clear own loss with no engagement measurement: the cause is unknown; read the per-axis deltas",
    "no_engage": "the diagnostic possibly did not move (or moved the wrong way): the idea may not have been tested; check "
                 "the implementation or its dose so the declared quantity moves before judging the idea",
    "axis_tradeoff": "a trade, not a free gain: {trade}; keep the paying axis and neutralise the charged one",
    "under_tuned": "closed before its tuning ladder (fewer than the required tuned points): any causal reading of its "
                   "result is only possible; re-tune (lr / layer decay / dose) before calling it dead",
    "noise": "inside the noise: more seeds or a bigger dose; it is neither a win nor a refutation",
    "positive_unconfirmed": "positive on screen seeds but not confirmed: selection-shrunk; needs fresh seeds",
    "confirmed": "confirmed on fresh seeds",
    "test_fail": "passed screening but failed the fresh-seed test: the screen gain was possibly selection / carrier",
    "crashed": "never measured: every run crashed; fix the implementation, the idea is open",
    "void": "never measured: its runs were void; the idea is open",
    "unmeasured": "no seed finished: nothing is known about the idea",
}


def is_computed(pm: Any) -> bool:
    return str(getattr(pm, "mechanism_reasoning", "") or "").startswith(MARK)


def classify(status: str, *, m: float | None, sd: float | None, engaged: str | None, axes: Mapping[str, Any],
             tuned: int, structural: bool, need: int = 3) -> tuple[str, str]:
    """(type, trade text): the typed outcome. Pure: every input is passed in."""
    if status in ("crashed", "void", "confirmed") or status == "test_fail" and not (structural and tuned < need):
        return status, ""
    if m is None:
        return "unmeasured", ""
    z = m / sd if sd and sd > 0 else (0.0 if m == 0 else math.copysign(9.0, m))
    if structural and tuned < need and (z <= 0 or status == "test_fail"):
        return "under_tuned", ""    # first: a structural change with < ``need`` tuned points has no causal reading yet
    up = sorted(((a, d / s) for a, (d, s, _n) in axes.items() if s > 0 and d / s >= Z_AXIS), key=lambda x: -x[1])
    down = sorted(((a, d / s) for a, (d, s, _n) in axes.items() if s > 0 and d / s <= -Z_AXIS), key=lambda x: x[1])
    if engaged == "engaged" and z <= -Z_STRONG:
        return "harmful_engaged", ""
    if up and down and z < Z_STRONG:
        return "axis_tradeoff", f"paid on {', '.join(a for a, _ in up[:2])}; charged on {', '.join(a for a, _ in down[:2])}"
    if engaged in ("did not engage", "reversed"):
        return "no_engage", ""
    if abs(z) < Z_STRONG:
        return "noise", ""
    return ("harmful", "") if z < 0 else ("positive_unconfirmed", "")


def children_of(led: Any) -> dict[str, set[str]]:
    """target node -> the tuning / ladder / re-run children booked on it (by the notes the tuners write)."""
    out: dict[str, set[str]] = {}
    for nid, node in led.nodes.items():
        notes = f" {getattr(node, 'notes', '') or ''}"
        port = " struct_retro=1" in notes and " struct_tune=" not in notes   # a retro port carries ``ladder=<old>``: a
        for kind, t in _LINK.findall(notes):                                  # re-run on another base, no tuned point
            if port and kind == "ladder":
                continue
            if t in led.nodes and t != nid:
                out.setdefault(t, set()).add(nid)
    return out


def facts(ctx: Any, store: Any, v: Any, nid: str, status: str, eng: Any, kids: Sequence[str],
          by: Mapping[str, Sequence[Any]]) -> dict[str, Any]:
    """The computed numbers behind one row."""
    from loop.agents import history as HX
    from loop.core import struct_tune as STT
    b = v.beliefs.get(nid)
    axes = {}
    from loop.project import AXES
    for a in AXES:
        try:
            d = HX.axis_delta(v, nid, a)
        except Exception:  # noqa: BLE001 - one axis with no pairs never stops a row
            d = None
        if d is not None and math.isfinite(d[0]):
            axes[a] = d
    sn = store.get_node(nid)
    fin = [c for c in kids if by.get(c)]
    best = max(((c, v.beliefs[c].m) for c in fin if c in v.beliefs), key=lambda x: x[1], default=None)
    return {"m": b.m if b else None, "sd": math.sqrt(max(0.0, b.v)) if b else None, "n": b.n if b else 0,
            "seeds": sorted({o.seed for o in by.get(nid, ())}), "axes": axes,
            "engaged": getattr(eng, "status", None), "z": getattr(eng, "z", None),
            "diag": getattr(eng, "diagnostic", ""), "tuned": len(kids), "tuned_done": len(fin), "best_tuned": best,
            "structural": bool(sn is not None and STT.structural_hint(sn))}


def explain_text(status: str, kind: str, f: Mapping[str, Any]) -> str:
    parts = [f"store status {status}, outcome type {kind}."]
    if f["m"] is not None:
        sd = f["sd"] or 0.0
        parts.append(f"Own panel increment over its parent {f['m']:+.4f} +- {sd:.4f} (z {f['m'] / sd:+.1f}) over {f['n']} "
                     f"paired seed(s) {f['seeds']}." if sd > 0 else
                     f"Own panel increment {f['m']:+.4f} over {f['n']} seed(s) {f['seeds']} (no sd yet).")
    if f["axes"]:
        parts.append("Per-axis paired delta: " + ", ".join(
            f"{a} {d:+.4f} +- {s:.4f} ({d / s:+.1f}se, n={n})" if s > 0 else f"{a} {d:+.4f} (n={n})"
            for a, (d, s, n) in f["axes"].items()) + ".")
    if f["engaged"]:
        parts.append(f"Engagement diagnostic {f['diag'] or '?'}: {f['engaged']}"
                     + (f" (z {f['z']:+.1f})" if f.get("z") is not None else "") + ".")
    else:
        parts.append("Engagement: no declared measurable diagnostic.")
    parts.append(f"Tuning points tried: {f['tuned']} ({f['tuned_done']} finished"
                 + (f"; best {f['best_tuned'][0]} {f['best_tuned'][1]:+.4f}" if f["best_tuned"] else "") + ").")
    return " ".join(parts)


def write(ctx: Any, store: Any, v: Any, engs: Sequence[Any], now: float, limit: int = LIMIT) -> list[str]:
    """One COMPUTED postmortem row per closed node that has none for its status (newest nodes first, at most ``limit``
    per tick); a node with any stored postmortem (an old Opus one, a struct ladder's) is left alone unless its status
    changed since this module wrote it. Returns the tick's act lines."""
    led = v.led
    if not hasattr(led, "by_node") or not hasattr(store, "put_postmortem"):
        return []
    done = dict(store.get_state(K_PM) or {})
    have = {pm.node_id for pm in store.postmortems()}
    todo = []
    for nid, node in led.nodes.items():
        sn = store.get_node(nid)
        kind = CLOSED.get(getattr(sn, "status", None)) if sn is not None else None
        if kind is None:
            continue
        prev = done.get(nid)
        if (prev and prev[0] == kind) or (nid in have and not prev):
            continue
        todo.append((node.created_ts or 0.0, nid, kind))
    if not todo:
        return []
    by, kids = led.by_node(), children_of(led)
    emap = {e.node: e for e in engs}
    acts: list[str] = []
    for _, nid, status in sorted(todo, reverse=True)[:limit]:
        try:
            f = facts(ctx, store, v, nid, status, emap.get(nid), sorted(kids.get(nid, ())), by)
            kind, trade = classify(status, m=f["m"] if f["n"] else None, sd=f["sd"], engaged=f["engaged"],
                                   axes=f["axes"], tuned=f["tuned_done"], structural=f["structural"])
            z = (f["m"] / f["sd"]) if f["m"] is not None and f["sd"] else 0.0
            verdict = {"crashed": "crash", "void": "void", "confirmed": "win"}.get(
                kind, "loss" if z <= -Z_STRONG or kind in ("harmful_engaged", "harmful") else "null")
            pid = f"pm-calc-{nid}-{status}"
            store.put_postmortem(Postmortem(
                id=pid, task_id=store.task_id, node_id=nid, verdict=verdict,
                explanation=f"[{kind}] " + explain_text(status, kind, f),
                mechanism_reasoning=f"{MARK}: a typed outcome from the ledger's own numbers; it says nothing about "
                                    f"WHY beyond what the numbers show.",
                levers=LeverTags(), suggested_next=HINTS[kind].format(trade=trade), created_ts=now,
                under_tuned=False, tune_knobs=json.dumps([])))
            done[nid] = [status, pid]
            acts.append(f"computed postmortem {nid}: {kind}")
        except Exception as exc:  # noqa: BLE001 - one node never stops the rest
            store.log("closed_pm_error", node_id=nid, error=repr(exc)[:300])
            done[nid] = [status, ""]
    store.set_state(K_PM, done)
    store.log("closed_pm", n=len(acts), left=max(0, len(todo) - limit))
    return [f"computed postmortems: {len(acts)} written ({max(0, len(todo) - limit)} left)"] if acts else []
