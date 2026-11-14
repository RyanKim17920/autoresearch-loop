"""loop.tune — MOONSHOT TUNING. A structural change (a moonshot, or an idea-bank idea) is screened at the
incumbent's hyperparameters, so a new objective / loss / masking scheme can look null only because its own
knobs (a new loss weight, mask ratio, schedule) or the sensitive shared ones (lr, warmup) were never retuned
for it (of 177 moonshots 10 got a pass, almost all at one dose; dose response is real: KDE concentration an
inverted U, iBOT weight .2/.3 negative, salient tau 1.5 good and .75 / 3 null, xscale .5 good and 1.5 bad).

THE DOSE LADDER (a CODE change: :func:`code_change`). A closed screened-out / not-confirmed code-changing moonshot,
explore or idea-bank node with shrunk P(own gain > 0) >= ``moonshot_tune.ladder_p`` (no confirm-seed requirement)
gets ONE ladder: its own dose knob (:func:`dose_knob`: the researcher's declared ``dose_knob``, else a knob its diff
introduced / changed) at 1/2x and 2x (1x = the node itself, already run) on the node's OWN screen seeds, each point
paired against the node's runs there (its parent controls already cancel), LR and every shared knob fixed: ~4 runs.
The dose response is fitted hierarchically (:func:`dose_fit`: quadratic in log2-dose per mechanism, the curvature
shrunk toward the population mean of every finished ladder by a DerSimonian-Laird tau; an argmax at the top / bottom
rung with a predicted gain > 0 first adds the next rung, :func:`_edge_rung`); the argmax-posterior dose,
when it is not 1x and its predicted gain over 1x is > 0, becomes a tuned node that takes promote / confirm / audit
on FRESH seeds (its own block, never the ladder's: the selection never touches the verdict). The ladder rows and the
fit are recorded in the tune history. A node built on ANOTHER carrier than the incumbent (root, an older confirmed
node) is BACKRUN on the incumbent (operator 2026-10-03): its change grafted onto the incumbent's commit (the
eval-only lane's 3-way graft) is the ladder's 1x point, screened against the incumbent's controls beside its 1/2x
and 2x points (~6 runs + 2 incumbent controls), and its dose is picked on gain over the incumbent, 1x included;
a graft that does not apply ladders on its own carrier. A backrun keeps the incumbent it started on. A node with NO
dose knob (its strength a hard-coded constant, or a data-sampling change's reweighting strength) first gets a KNOB
EXPOSURE (:func:`requests`, a side-slot researcher request on its 1x commit): the constant becomes a config key whose
default is the current value, declared as the dose knob; verified identical at 1x, the 1x measurements stand and the
points are built on the exposed commit beside them, else the exposed commit is a new 1x point on the same block. At most ``max_ladders_inflight`` run at once, one per mechanism (a retried dead
mechanism's cluster, :func:`mech_key`), most promising first (so the backlog runs P >= 0.7 first); scarce mode starts
only P >= ``ladder_scarce_p`` ones.

THE PASS (any qualifying structural node; a NUMBERS-ONLY one behind :func:`gate`'s fresh-confirm-seed rule, a MAJOR
one -- :func:`major_change` -- as well as its dose ladder, never both at once):
A closed structural node that qualifies (:func:`classify`) gets ONE tuning pass, run by the driver
beside everything else (at most ``moonshot_tune.max_concurrent`` at once; separate id namespace ``t<b>``, so
normal DESIGN batches are never blocked):

* budget: ``max_trials_major`` variants for a PROMISING-ROUGH node (a MAJOR code change with shrunk
  P(U>0) >= ``rough_min_p``: it reached ~baseline, it just needs its hyperparameters), else
  ``max_trials_minor``. A trial is one variant node on top of the moonshot's commit (its parent), screened on
  ONE shared seed block and paired against the moonshot's own PARENT runs at those seeds (the moonshot is the
  control: every trial differs from it only by the variant, so the pairing is exact);
* phase 1: a small 2-level design (loop.design's machinery and analysis) over the moonshot's own new knobs
  (its postmortem's ``tune_knobs``, else the config keys it changed, stepped to a softer dose) plus at most
  one shared knob (``moonshot_tune.shared``): a full 2^2 or a 2^(3-1) fraction (3 trials), or one factor at
  a time when the budget is 2;
* phase 2: the remaining trials one at a time, each proposed by the researcher (a ``tune`` brief with the
  phase-1 effects, every trial so far and the moonshot's diff); the pass stops early once a trial clears
  the promote bar or two consecutive trials fall below the best so far;
* the best trial that beat the untuned moonshot becomes a normal node (strategy moonshot, notes
  ``tune_of=<id>``) that takes promote / confirm / audit on FRESH seeds: the best-of-k selection bias never
  reaches a verdict, because the tuning seeds are asserted disjoint from every promote / confirm / audit seed.

Credit: the pass's GPU-h is charged to the moonshot arm (the tuned node's bandit reward divides by it; a pass
without a tuned node records a zero-reward moonshot pull). The quality table groups it as
``moonshot_tuned``."""

from __future__ import annotations

import dataclasses
import json
import math
import re
import statistics
import time
from statistics import NormalDist
from pathlib import Path
from typing import Any, Mapping, Sequence

from loop import components as CG
from loop import design as D
from loop import seeds as S
from loop.agents.researcher import diff_hash
from loop.contracts import (CARRIER, CLOSED_STATUSES, DESIGN_FAMILY, PANEL, ROOT_NODE_ID, ArmPull, AxisPrediction,
                            CandidateRequest, HypothesisKind, HypothesisRecord, Node, NodeStatus, Operator,
                            Stage, Strategy, design_point)
from loop.select.bandit import tune_gpu_h
from loop.spec import LoopContext

K_TUNE = "tune"     # {"seq": n, "current": {batch: pass}, "batches": [closed passes], "tuned": [node ids]}
_CLOSES = (NodeStatus.SCREENED_OUT, NodeStatus.VETOED, NodeStatus.NOT_CONFIRMED)
_CONFIG_SUFFIXES = (".yaml", ".yml", ".json", ".toml", ".cfg", ".ini")
#: 2-level rows (+1 = high), row 0 = the moonshot itself (every knob at its own value)
_FULL2 = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
_HALF3 = [(-1, -1, -1), (-1, 1, 1), (1, -1, 1), (1, 1, -1)]      # C = -AB: resolution III, 3 trials
_OFAT2 = [(-1, -1), (1, -1), (-1, 1)]                             # one factor at a time, 2 trials


def _note(node: Node, key: str) -> str:
    m = re.search(rf"(?:^|\s){key}=(\S+)", node.notes or "")
    return m.group(1) if m else ""


def structural(node: Node) -> bool:
    """A moonshot or an idea-bank idea (never a tuned node or a trial itself)."""
    return (node.strategy is Strategy.MOONSHOT or bool(_note(node, "bank"))) and not design_point(node) \
        and not _note(node, "tune_of") and not _note(node, "tune")


def book(store: Any) -> dict[str, Any]:
    return {"seq": 0, "current": {}, "batches": [], "tuned": [], "laddered": [], **(store.get_state(K_TUNE) or {})}


def latest_postmortem(store: Any, node_id: str) -> Any:
    return max(store.postmortems(node_id=node_id), key=lambda pm: pm.created_ts or 0.0, default=None)


def fresh_postmortem(store: Any, node: Node) -> Any:
    """The latest postmortem if it explains the node's CURRENT close (None for e.g. the crash postmortem of
    a node the OOM retry reopened, which then scored and closed again)."""
    pm = latest_postmortem(store, node.id)
    want = {NodeStatus.CRASHED: ("crash",), NodeStatus.VOID: ("void",)}.get(node.status, ("loss", "null"))
    return pm if pm is not None and pm.verdict in want else None


def postmortem_knobs(pm: Any) -> list[dict[str, Any]]:
    try:
        rows = json.loads(pm.tune_knobs) if pm is not None and pm.tune_knobs else []
    except (TypeError, ValueError):
        return []
    return [r for r in rows if isinstance(r, dict) and r.get("key")] if isinstance(rows, list) else []


def diff_stats(text: str) -> dict[str, Any]:
    """Code (non-config) lines changed, files added, defs / classes added — from a unified diff."""
    code = new_files = defs = 0
    path, is_code = "", False
    for line in text.splitlines():
        if line.startswith("--- "):
            new_files += line.strip() == "--- /dev/null"
            continue
        if line.startswith("+++ "):
            path = line[4:].strip().removeprefix("b/")
            is_code = not path.endswith(_CONFIG_SUFFIXES)
            continue
        if is_code and line[:1] in "+-" and line[1:].strip():
            code += 1
            defs += line.startswith("+") and bool(re.match(r"\s*(def|class)\s", line[1:]))
    return {"code_lines": code, "new_files": new_files, "defs": defs}


def is_major(ctx: LoopContext, store: Any, node: Node) -> bool:
    """MAJOR: the diff adds / changes code (a new file, a def / class, or >= major_code_lines code lines);
    a node whose diff is unreadable is MAJOR iff it changed no config value (a code-only change)."""
    try:
        text = Path(node.diff_path).read_text(errors="replace") if node.diff_path else ""
    except OSError:
        text = ""
    if not text:
        return not store.get_state(f"config_diff:{node.id}")
    st = diff_stats(text)
    return bool(st["new_files"] or st["defs"] or st["code_lines"] >= ctx.cfg.moonshot_tune.major_code_lines)


MAJOR_TYPES = ("loss_objective", "architecture_addon")      # a new objective / loss, an architecture change


def major_change(ctx: LoopContext, store: Any, node: Node) -> bool:
    """A MAJOR change for the tuning budget: a code change (:func:`is_major`), an init / architecture swap
    (``init=`` in its notes) or an objective / architecture mechanism (loop.novelty mechanism type)."""
    from loop.novelty import node_type
    return is_major(ctx, store, node) or bool(_note(node, "init")) or node_type(node) in MAJOR_TYPES


def focus_match(ctx: LoopContext, node: Node) -> bool:
    """The node's mechanism tag shares a word (>= 4 letters, prefix match) with the operator focus."""
    focus = re.findall(r"[a-z]{4,}", (ctx.cfg.operator_focus or "").lower())
    return any(f.startswith(t) or t.startswith(f) for t in re.findall(r"[a-z]{4,}", node.mechanism_tag.lower())
               for f in focus)


def p_gain(post: Any) -> float:
    """Shrunk P(increment > 0): the node vs the PARENT it was built on. (P(U>0) is vs root: ~1 for every child
    of a confirmed incumbent, so it cannot tell a near miss from a loss there; for a child of root they agree.)"""
    if getattr(post, "gain_p", None) is not None:          # the selection-aware EB mixture (loop/world/mixture)
        return post.gain_p
    m, sd = post.increment.mean, post.increment.sd
    return NormalDist().cdf(m / sd) if sd > 0 else 0.5 + 0.5 * ((m > 0) - (m < 0))


def classify(ctx: LoopContext, store: Any, snapshot: Any, node: Node) -> dict[str, Any] | None:
    """A closed structural node that qualifies for its tuning pass -> {reason, budget, major, rough, promise};
    None otherwise (P = shrunk P(gain over its parent > 0), :func:`p_gain`). Screened out with P in
    [p_lo, p_hi] or a postmortem saying under_tuned;
    NOT_CONFIRMED (it passed promote, then failed fresh seeds); VETOED only when borderline (P >= veto_min_p,
    ONE vetoed axis, its posterior harm within veto_max_sigmas axis sigmas): a clearly terrible node stays dead."""
    mt = ctx.cfg.moonshot_tune
    post = snapshot.nodes.get(node.id) if snapshot is not None else None
    if not structural(node) or node.status not in _CLOSES or post is None or post.utility.n == 0:
        return None                     # a code change gets this pass as well as its dose ladder (never at once)
    pm, p = fresh_postmortem(store, node), p_gain(post)
    under = bool(pm is not None and pm.under_tuned)          # a stored (historical) postmortem's call
    if node.status is NodeStatus.VETOED:
        if p < mt.veto_min_p or len(post.veto_axes) != 1:
            return None
        axis = post.veto_axes[0]
        q, sd = post.axes.get(axis), snapshot.sigma.get(axis) or ctx.cfg.axis_sd.get(axis, 0.0)
        if q is None or q.mean < -mt.veto_max_sigmas * sd:
            return None
        reason = f"vetoed on {axis} but borderline ({q.mean:+.4f} vs {sd:.4f} sigma, P(gain>0)={p:.2f})"
    elif node.status is NodeStatus.SCREENED_OUT:
        if not (mt.p_lo <= p <= mt.p_hi or under):
            return None
        reason = f"screened out at P(gain>0)={p:.2f}" + (" (postmortem: under-tuned)" if under else "")
    else:
        reason = f"not confirmed on fresh seeds (P(gain>0)={p:.2f})"
    major = major_change(ctx, store, node)      # was hard-coded False (915bbb7f): the 5-trial budget never ran
    rough = major and p >= mt.rough_min_p
    focus = focus_match(ctx, node)
    from loop.core import bold as BOLD
    if (room := BOLD.tune_room(ctx, store, node)) is not None:   # v3-c8b: a bold node's family shares ONE budget
        if room <= 0:
            return None
        # v3 review S2: not forced rough / major whatever p is -- the usual rule, within the unit's room
        return {"reason": reason + f" (bold: {room} of its tuning trials left)",
                "budget": min(room, mt.max_trials_major if rough else mt.max_trials_minor), "major": major,
                "rough": rough, "focus": focus, "p": p, "promise": p + 0.25 * focus + 0.25 * under + 0.1 * rough}
    return {"reason": reason, "budget": mt.max_trials_major if rough else mt.max_trials_minor,
            "major": major, "rough": rough, "focus": focus, "p": p,
            "promise": p + 0.25 * focus + 0.25 * under + 0.1 * rough}


def busy_families(store: Any, skip: str = "") -> dict[str, str]:
    """normalized mechanism tag -> the node holding it in tuning-or-confirm: a CONFIRMING node, a running pass's base,
    or a tuned node still live (``skip``: a node never counted against itself)."""
    from loop.novelty import norm_tag
    bases = {c["base"] for c in book(store)["current"].values()}
    out: dict[str, str] = {}
    for n in sorted(store.nodes(), key=lambda n: (n.created_ts, n.id)):
        if n.id != skip and (n.status is NodeStatus.CONFIRMING or n.id in bases
                             or _note(n, "tune_of") and n.status not in CLOSED_STATUSES) and norm_tag(n.mechanism_tag):
            out.setdefault(norm_tag(n.mechanism_tag), n.id)
    return out


def gate(ctx: LoopContext, store: Any, snapshot: Any, node: Node, busy: Mapping[str, str] | None = None) -> str:
    """Why ``node`` may NOT be tuned ('' = it may): a NUMBERS-ONLY node's OWN increment on its fresh CONFIRM seeds
    alone (the confirm e-value's numbers, WorldSnapshot.own_confirm) must reach ``moonshot_tune.min_confirm_sigmas``
    (1) panel sigma -- selection-seed promise is the winner's curse (live: ~70 GPU-h of passes, 6 tuned nodes, all
    screened out); a CODE change never needs it (its ladder's selection seeds never reach its verdict, and the gate
    made every screened-out code change permanently ineligible) -- and no other node of its family (normalized
    tag) may be in tuning-or-confirm (:func:`busy_families`)."""
    from loop.novelty import norm_tag
    m, n = (getattr(snapshot, "own_confirm", None) or {}).get(node.id, (0.0, 0))
    sig = float(snapshot.sigma.get(PANEL, ctx.cfg.panel_sd))
    if (k := ctx.cfg.moonshot_tune.min_confirm_sigmas) > 0 and (not n or m < k * sig) \
            and not major_change(ctx, store, node):
        return f"own confirm increment {m:+.4f} on {n} fresh seed(s) < {k:g} sigma ({k * sig:.4f})"
    busy = busy_families(store, skip=node.id) if busy is None else busy
    if (other := busy.get(norm_tag(node.mechanism_tag))) and other != node.id:
        return f"family {norm_tag(node.mechanism_tag)} already in tuning/confirm ({other})"
    return ""


def eligible(ctx: LoopContext, store: Any, snapshot: Any, now: float) -> list[tuple[Node, dict[str, Any]]]:
    """Qualifying closed structural nodes without a pass yet, most promising first by the node's OWN increment
    (shrunk P(gain > 0) over the ancestor it is measured against, then its mean -- never U vs root, which every
    carrier copy inherits), plus operator focus match, a (stored) postmortem saying under-tuned, promising-rough."""
    from loop import campaign as CP
    b = book(store)
    done = set(b["tuned"]) | {c["base"] for c in b["current"].values()}
    out = []
    for n in store.nodes(status=list(_CLOSES)):
        if n.id in done or CP.claims(ctx, store, n):    # a laddered node (``tuning=``) still gets its pass, after
            continue                                     # its ladder; a campaign's node is the campaign's alone
        if (c := classify(ctx, store, snapshot, n)) is not None:
            out.append((n, c))
    out.sort(key=lambda nc: (-nc[1]["promise"], -snapshot.nodes[nc[0].id].increment.mean, nc[0].created_ts, nc[0].id))
    from loop.novelty import norm_tag
    busy, kept = busy_families(store), []
    for n, c in out:                                  # the gate; then one per family (the most promising)
        if not gate(ctx, store, snapshot, n, {k: v for k, v in busy.items() if v != n.id}):
            kept.append((n, c))
            busy.setdefault(norm_tag(n.mechanism_tag), n.id)
    return kept


# ---------------------------------------------------------------------------------------- knobs
def dose_keys(store: Any, node: Node, base: Mapping[str, Any]) -> set[str]:
    """The node's MECHANISM-DOSE knobs: the config keys its own diff changed (``config_diff:<id>``) plus base keys
    whose leaf (>= 4 chars, so never ``lr``) its diff's added code lines name (a knob the new code reads)."""
    keys = set(store.get_state(f"config_diff:{node.id}") or {})
    added = " ".join(l[1:] for l in _read(node.diff_path).splitlines() if l.startswith("+") and not l.startswith("+++"))
    return keys | {k for k in base if len(leaf := k.rsplit(".", 1)[-1]) >= 4
                   and re.search(rf"(?<![A-Za-z0-9_]){re.escape(leaf)}(?![A-Za-z0-9_])", added)}


def pick_knobs(ctx: LoopContext, store: Any, node: Node, text: str, base: Mapping[str, Any],
               pm: Any, first: Sequence[Mapping[str, Any]] = ()) -> list[dict[str, Any]]:
    """<= 2 of the moonshot's own MECHANISM-DOSE knobs (:func:`dose_keys`: its (stored) postmortem's
    tune_knobs on them, then the config keys it changed, a numeric one stepped DOWN by design.step: a new term that
    loses usually overpowers the recipe) plus <= 1 shared knob (a postmortem's knob outside the dose set,
    else ``moonshot_tune.shared``: high = value x factor); each low = the moonshot's value. A design always tunes
    >= 1 dose knob: with none usable it is [] (phase 2 researcher variants, told to tune the dose), never an
    lr-only design (9/20 under-tuned postmortems named dino.lr and the passes all tried lr x1.25)."""
    own: list[dict[str, Any]] = []
    shared: list[dict[str, Any]] = []
    dose, seen = dose_keys(store, node, base), D.screened_out(store)

    def add(into: list, key: str, low: Any, high: Any) -> None:     # never a level screened out 3x before
        kb = {"key": key, "low": low, "high": high, "levers": {}}
        if high is not None and key not in {k["key"] for k in own + shared} and D._knob_ok(ctx, text, base, kb) \
                and D.fresh_level(store, kb, seen):
            into.append(kb)
    for r in [*first, *postmortem_knobs(pm)]:           # given knobs, then the postmortem's
        add(own if str(r["key"]) in dose else shared, str(r["key"]), r.get("low", base.get(str(r["key"]))), r.get("high"))
    for key in sorted(dose, key=lambda k: (k not in (store.get_state(f"config_diff:{node.id}") or {}), k)):
        if key in base and D._num(base[key]):
            add(own, key, base[key], D._step(base[key], -1, ctx.cfg.design.step))
    for key, f in ctx.cfg.moonshot_tune.shared.items():
        if not shared and key in base and D._num(base[key]):
            add(shared, key, base[key], D._step(base[key], 1 if f > 1 else -1, f if f > 1 else 1 / f))
    return own[:2] + shared[:1] if own else []


def rows_for(k: int, budget: int) -> list[tuple[int, ...]]:
    """Phase-1 rows over the first k' <= k knobs (row 0 = the moonshot): 3 trials when the budget allows
    (2^2, or the 2^(3-1) fraction for 3 knobs), else one factor at a time over 2 knobs, else 1 knob."""
    if k == 0:
        return []
    if k == 1 or budget < 2:
        return [(-1,), (1,)]
    if budget >= 3:
        return _HALF3 if k >= 3 else _FULL2
    return _OFAT2


# ------------------------------------------------------------------------------------ the pass
def tick(ctx: LoopContext, store: Any, adapter: Any, snapshot: Any, *, new_id: Any, now: float | None = None,
         inflight: Sequence[str] = (), scarce: bool = False) -> list[str]:
    """Advance every running pass; start passes for the most promising qualifying nodes while fewer than
    ``max_concurrent`` rule passes run (all passes <= :func:`safety_cap`).
    ``inflight``: request ids the driver's researchers are still working on. ``scarce`` (the driver's scarce mode):
    running passes only advance -- no new pass starts (their trials still finish); a dose
    ladder still starts for P(own gain > 0) >= ``ladder_scarce_p`` (:func:`_ladders`)."""
    mt, now = ctx.cfg.moonshot_tune, time.time() if now is None else now
    if not mt.enabled or snapshot is None:
        return []
    acts: list[str] = []
    acts += reopen_void_ladders(ctx, store, now, new_id)
    for bid in list(book(store)["current"]):
        bk = book(store)
        cur = bk["current"][bid]
        try:
            acts += _advance(ctx, store, snapshot, bk, cur, now, set(inflight), new_id, adapter)
        except Exception as exc:  # noqa: BLE001 — a git / adapter failure closes the pass, never the loop
            _close(store, bk, cur, now, error=repr(exc)[:300])
            store.log("tune_error", batch=cur["batch"], node_id=cur["base"], error=repr(exc)[:500])
            acts.append(f"tune t{cur['batch']} error: {exc!r}"[:300])
    acts += _ladders(ctx, store, adapter, snapshot, new_id, now, scarce)
    if scarce:
        return acts
    cur_all = book(store)["current"].values()
    free = min(mt.max_concurrent - sum(not c.get("granted") and c.get("kind") != "ladder" for c in cur_all),
               safety_cap(ctx) - len(cur_all))
    for node, info in eligible(ctx, store, snapshot, now)[:max(0, free)]:
        store.log("tune", node_id=node.id, queued=True, **{
            k: v for k, v in info.items() if k != "promise"}, promise=round(info["promise"], 4))
        acts.append(f"tune queued {node.id} ({node.mechanism_tag}): "
                    f"{info['reason']}; budget {info['budget']} trial(s)"
                    + (" [major, promising-rough]" if info["rough"] else " [major]" if info["major"] else ""))
        try:
            acts += _start(ctx, store, adapter, node, info, new_id, now)
        except Exception as exc:  # noqa: BLE001
            store.log("tune_error", node_id=node.id, error=repr(exc)[:500])
            acts.append(f"tune {node.id} could not start: {exc!r}"[:300])
    return acts


def safety_cap(ctx: LoopContext) -> int:
    """All tuning passes together: at most half the preemptible slots (their trials run nowhere else)."""
    pre = sum(p.slots for p in ctx.cfg.pools if p.preemptible) or ctx.cfg.capacity
    return max(1, pre // 2)


def _start(ctx: LoopContext, store: Any, adapter: Any, node: Node, info: Mapping[str, Any], new_id: Any,
           now: float) -> list[str]:
    bk = book(store)
    b = bk["seq"] + 1
    tag = f"t{b}"
    bk["seq"], bk["tuned"] = b, bk["tuned"] + [node.id]              # ONE pass per node, even on failure
    cur = {"batch": b, "base": node.id, "budget": int(info["budget"]), "major": info["major"],
           "rough": info["rough"], "reason": info["reason"], "knobs": [], "rows": [], "points": {"0": node.id},
           "trials": [], "phase": 1, "pending": "", "fails": 0, "effects": [], "started": now,
           "granted": bool(info.get("granted"))}
    bk["current"][str(b)] = cur
    store.set_state(K_TUNE, bk)
    store.update_node(node.id, notes=f"{node.notes or ''} tuning={tag}".strip())
    with D._Tree(ctx, store, adapter, f"tune{b}", base_id=node.id) as t:
        knobs = pick_knobs(ctx, store, node, t.text, t.base, fresh_postmortem(store, node),
                           first=info.get("knobs") or ())
        rows = rows_for(len(knobs), cur["budget"])
        knobs = knobs[:len(rows[0])] if rows else []
        for i, row in enumerate(rows[1:], start=1):
            nid = t.node({kb["key"]: kb["high"] if x > 0 else kb["low"] for kb, x in zip(knobs, row)},
                         knobs=knobs, tag=f"{tag}pt{i:02d}", operator=None, new_id=new_id, notes=f"tune={tag}")
            if nid and nid != node.id:
                cur["points"][str(i)] = nid
                cur["trials"].append(nid)
    cur.update(knobs=knobs, rows=[list(r) for r in rows], phase=1 if cur["trials"] else 2)
    bk = book(store)
    bk["current"][str(b)] = cur
    store.set_state(K_TUNE, bk)
    store.log("tune_started", batch=b, node_id=node.id, knobs=knobs, trials=cur["trials"], budget=cur["budget"])
    return [f"tune {tag} started for {node.id}: budget {cur['budget']}; phase 1 {len(cur['trials'])} config "
            f"trial(s) over {[kb['key'] for kb in knobs] or 'no usable knob'} on one shared block vs {node.id}"]


def delta(ctx: LoopContext, store: Any, base_id: str, nid: str) -> tuple[float, float, int]:
    """(mean, se, n) of a trial's panel minus the moonshot's PARENT run at the same seed."""
    ctl = {r.seed: y for r in store.runs(node_id=base_id, stage=Stage.PARENT)
           if (y := D.panel_of(ctx, r)) is not None}
    ds = [y - ctl[r.seed] for r in store.runs(node_id=nid, stage=Stage.SCREEN)
          if r.seed in ctl and (y := D.panel_of(ctx, r)) is not None]
    if not ds:
        return 0.0, 0.0, 0
    return statistics.fmean(ds), (statistics.stdev(ds) / len(ds) ** 0.5 if len(ds) > 1 else 0.0), len(ds)


def tuning_seeds(store: Any, cur: Mapping[str, Any]) -> set[int]:
    return {r.seed for nid in cur["trials"] for r in store.runs(node_id=nid, stage=Stage.SCREEN)}


def _stop(ctx: LoopContext, store: Any, snapshot: Any, cur: dict) -> str:
    """Why the pass ends now ('' = propose another trial)."""
    pr = ctx.cfg.promotion
    for nid in cur["trials"]:                  # the screen bar AND the promote bar (it would beat the incumbent)
        post = snapshot.nodes.get(nid)
        if post is not None and post.utility.n and post.p_gate >= pr.p_promote \
                and post.p_beats_incumbent >= pr.p_confirm:
            return (f"{nid} clears the promote bar (P(U>0)={post.p_positive:.2f}, P(gain>0)={post.p_gain:.2f}, "
                    f"P(U>U_inc)={post.p_beats_incumbent:.2f})")
    if len(cur["trials"]) >= cur["budget"]:
        return f"budget spent ({len(cur['trials'])}/{cur['budget']} trials)"
    if cur["fails"] >= ctx.cfg.moonshot_tune.max_research_fails:
        return f"{cur['fails']} variant proposals failed"
    best, worse = 0.0, 0                        # the untuned moonshot is the bar to beat
    n1 = sum(nid in cur["points"].values() for nid in cur["trials"])
    for i, nid in enumerate(cur["trials"]):
        d, _, n = delta(ctx, store, cur["base"], nid)
        if i >= n1:                             # phase-2 trials: consecutive ones below the best so far
            worse = worse + 1 if (not n or d < best) else 0
        if n:
            best = max(best, d)
    return "two consecutive trials fell below the best so far" if worse >= 2 else ""


def _advance(ctx: LoopContext, store: Any, snapshot: Any, bk: dict, cur: dict, now: float,
             inflight: set[str], new_id: Any, adapter: Any = None) -> list[str]:
    tag, acts = f"t{cur['batch']}", []
    if cur.get("kind") == "ladder":
        if not cur["knob"] and "expose" not in cur and ctx.cfg.moonshot_tune.expose:
            cur["expose"] = "want"              # a knob-less ladder started before exposures existed (backfill)
            _save(store, bk, cur)
        if cur.get("expose") and (cur["expose"] != "done" or not cur["trials"]):
            return _expose(ctx, store, adapter, bk, cur, now, inflight, new_id)
        if any(n is None or n.status not in CLOSED_STATUSES for n in map(store.get_node, cur["trials"])):
            return []
        if (acts := _retry_void_points(ctx, store, cur, now, new_id)) or (
                acts := _edge_rung(ctx, store, adapter, snapshot, cur, new_id)):
            _save(store, bk, cur)
            return acts
        return _finish_ladder(ctx, store, snapshot, bk, cur, now, new_id)
    if cur["pending"]:
        if cur["pending"] in inflight:
            return []
        cur.update(pending="", fails=cur["fails"] + 1)              # the proposal produced no trial
        store.log("tune_proposal_failed", batch=cur["batch"], node_id=cur["base"], fails=cur["fails"])
        acts.append(f"tune {tag}: variant proposal produced no trial ({cur['fails']} failed)")
    nodes = [store.get_node(i) for i in cur["trials"]]
    if any(n is None or n.status not in CLOSED_STATUSES for n in nodes):
        _save(store, bk, cur)
        return acts
    if cur["phase"] == 1:
        ana = D.analyze([kb["key"] for kb in cur["knobs"]], cur["rows"], D._obs(ctx, store, cur), ctx.cfg.design.z)
        cur.update(phase=2, effects=ana["effects"], n=ana["n"], df=ana["df"])
        acts.append(f"tune {tag} phase 1 done: " + ("; ".join(f"{e['key']} {D.fmt_effect(e)}" for e in ana["effects"])
                                                      or "no knob"))
    if why := _stop(ctx, store, snapshot, cur):
        return acts + _finish(ctx, store, snapshot, bk, cur, now, why, new_id)
    cur["want"] = True                          # the driver's next researcher slot proposes a variant
    _save(store, bk, cur)
    return acts


def outage(store: Any, request_id: str) -> str:
    """A tuning / exposure request an outage ate (backend_unavailable) is NOT a failed proposal: asked again, no
    ``fails`` count ('' when no pass waits on it)."""
    bk = book(store)
    cur = next((c for c in bk["current"].values() if c["pending"] == request_id), None)
    if cur is None:
        return ""
    cur["pending"] = ""
    if cur.get("expose") == "pending":
        cur["expose"] = "want"
    _save(store, bk, cur)
    return f"t{cur['batch']}"


def withdraw(store: Any, batch: Any, why: str, now: float) -> None:
    """Close a running pass without a tuned node (no new trial; its in-flight trials finish unread)."""
    bk = book(store)
    if (cur := bk["current"].get(str(batch))) is not None:
        gpu = tune_gpu_h(store, cur["base"])         # the pass still costs the moonshot arm
        store.record_pull(ArmPull(task_id=store.task_id, arm=Strategy.MOONSHOT, node_id=f"{cur['base']}#tune",
                                  reward=0.0, gpu_h=gpu, ts=now))
        _close(store, bk, cur, now, why=f"withdrawn: {why}", tuned="", best="", gpu_h=gpu)
        store.log("tune_withdrawn", batch=cur["batch"], node_id=cur["base"], why=why)


def _save(store: Any, bk: dict, cur: dict) -> None:
    bk["current"][str(cur["batch"])] = cur
    store.set_state(K_TUNE, bk)


def _close(store: Any, bk: dict, cur: dict, now: float, **result: Any) -> None:
    bk["current"].pop(str(cur["batch"]), None)
    bk["batches"] = bk["batches"] + [{**cur, **result, "want": False, "closed": now}]
    store.set_state(K_TUNE, bk)


def _finish(ctx: LoopContext, store: Any, snapshot: Any, bk: dict, cur: dict, now: float, why: str,
            new_id: Any) -> list[str]:
    tag, base = f"t{cur['batch']}", store.get_node(cur["base"])
    plan = ctx.cfg.seeds
    seeds = tuning_seeds(store, cur)
    if clash := seeds & {int(s) for s in (*plan.promote, *plan.confirm_pool, *plan.audit)}:
        raise ValueError(f"tuning seeds {sorted(clash)} overlap promote/confirm/audit seeds: no fresh-seed verdict")
    scored = [(nid, *delta(ctx, store, cur["base"], nid)) for nid in cur["trials"]]
    best = max((s for s in scored if s[3]), key=lambda s: s[1], default=None)
    gpu = tune_gpu_h(store, cur["base"])
    result: dict[str, Any] = {"why": why, "tuned": "", "best": best[0] if best else "", "gpu_h": gpu,
                              "deltas": {nid: [d, se, n] for nid, d, se, n in scored}, "seeds": sorted(seeds)}
    acts = [f"tune {tag} for {cur['base']} ends: {why}; " + (", ".join(
        f"{nid} {d:+.4f}±{se:.4f}" for nid, d, se, n in scored if n) or "no scored trial")]
    from loop.core import bold as BOLD0
    if best is not None and best[1] > 0 and (live := BOLD0.tuned_live(store, base)):
        acts.append(f"tune {tag}: {base.id}'s bold family already has tuned node {live} in flight: at most one TEST-bound "
                    f"tuned node per family at a time, {best[0]} is not promoted")
        store.log("tune_tuned_held", node_id=base.id, best=best[0], live=live)
        best = None
    if best is not None and best[1] > 0:
        result["tuned"] = _tuned_node(ctx, store, snapshot, base, store.get_node(best[0]), cur, best[1], new_id, now)
        acts.append(f"tune {tag}: tuned node {result['tuned']} = {best[0]} (+{best[1]:.4f} over {base.id} on the "
                    f"tuning block) -> promote/confirm/audit on fresh seeds")
    else:                                      # nothing to promote: the pass still costs the moonshot arm
        store.record_pull(ArmPull(task_id=ctx.task_id, arm=Strategy.MOONSHOT, node_id=f"{cur['base']}#tune",
                                  reward=0.0, gpu_h=gpu, ts=now))
        acts.append(f"tune {tag}: no trial beat {base.id}; {gpu:.2f} GPU-h charged to the moonshot arm")
    _close(store, bk, cur, now, **result)
    store.log("tune_done", batch=cur["batch"], node_id=cur["base"], **{k: v for k, v in result.items() if k != "deltas"})
    return acts


def _tuned_node(ctx: LoopContext, store: Any, snapshot: Any, base: Node, best: Node, cur: Mapping[str, Any],
                gain: float, new_id: Any, now: float, parent: str = "") -> str:
    """The best trial's commit as a normal moonshot node beside the untuned one (same parent; a backrun's: the
    incumbent it ran on, ``parent``): it screens on its own block, then promote / confirm / audit — every one on
    seeds the tuning never used."""
    nid, tag = new_id(), f"t{cur['batch']}"
    cfg_diff = {**(store.get_state(f"config_diff:{base.id}") or {}), **(store.get_state(f"config_diff:{best.id}") or {})}
    post = snapshot.nodes.get(base.id)
    hyp = HypothesisRecord(
        id=f"h-{nid}", task_id=ctx.task_id, kind=HypothesisKind.TRIAL, family=base.family,
        mechanism_tag=f"{base.mechanism_tag}_tuned", created_ts=now, levers=base.levers,
        claim=f"{base.mechanism_tag} ({base.id}) was judged at the incumbent's hyperparameters; retuned "
              f"({best.notes or best.id}) it beat its untuned self by {gain:+.4f} on the tuning block",
        predictions={PANEL: AxisPrediction(1, max(1e-4, (post.increment.mean if post else 0.0) + gain))},
        falsification="the tuned node does not beat its parent on fresh seeds")
    store.put_hypothesis(hyp)
    diffs = Path(ctx.state_dir) / "diffs"
    diffs.mkdir(parents=True, exist_ok=True)
    text = "".join(_read(p) for p in dict.fromkeys((base.diff_path, best.diff_path)))
    if not best.diff_path and cfg_diff:
        text += f"# tuning trial {best.id} (config on top of {base.id}): {store.get_state(f'config_diff:{best.id}')}\n"
    (diffs / f"{nid}.diff").write_text(text)
    fams = _note(base, "fams")
    store.put_node(Node(
        id=nid, task_id=ctx.task_id, commit=best.commit, parent_id=parent or base.parent_id or ROOT_NODE_ID,
        strategy=Strategy.MOONSHOT, operator=Operator.IMPROVE, family=base.family,
        mechanism_tag=hyp.mechanism_tag, diff_hash=diff_hash(f"tuned:{best.commit}"), status=NodeStatus.SCREENING,
        island=base.island, diff_path=str(diffs / f"{nid}.diff"), hypothesis_id=hyp.id, levers=base.levers,
        created_ts=now, updated_ts=now,
        notes=f"tune_of={base.id} tune={tag} best={best.id} brief=tune" + (f" fams={fams}" if fams else "")))
    store.link(hyp.id, nid, sign=1, auto=False)
    if cfg_diff:
        store.set_state(f"config_diff:{nid}", cfg_diff)
    store.log("node_transition", node_id=nid, frm="new", to=NodeStatus.SCREENING.value,
              reason=f"tuned {base.id} via {tag} (best trial {best.id})")
    return nid


def _read(path: str) -> str:
    try:
        return Path(path).read_text(errors="replace") if path else ""
    except OSError:
        return ""


# ------------------------------------------------------------------ phase 2: researcher proposals
def requests(ctx: LoopContext, store: Any, snapshot: Any, limit: int, scarce: bool = False) -> list[CandidateRequest]:
    """The next variant proposal of each pass that wants one (<= ``limit``; none in ``scarce`` mode): an IMPROVE on
    the moonshot with a ``tune`` brief; and each ladder's knob exposure (``expose`` brief, scarce mode too: a waiting
    ladder holds a max_ladders_inflight slot). Marked pending here; :func:`adopt` turns the admitted candidate into a
    trial / the exposed commit."""
    out: list[CandidateRequest] = []
    bk = book(store)
    for cur in bk["current"].values():
        if len(out) < limit and cur.get("expose") == "want":      # a ladder's knob exposure, on its 1x commit
            base = store.get_node(cur["base"])
            req = CandidateRequest(id=f"expose-t{cur['batch']}-{cur['fails'] + 1}", task_id=ctx.task_id,
                                   strategy=Strategy.DESIGN, operator=Operator.IMPROVE, island=base.island,
                                   parent_ids=(cur.get("at") or base.id,), brief=expose_brief(ctx, store, cur),
                                   brief_kind="expose", family_hint=base.family)
            cur.update(pending=req.id, expose="pending")
            out.append(req)
            continue
        if scarce or len(out) >= limit or not cur.get("want") or cur["pending"]:
            continue
        base = store.get_node(cur["base"])
        k = len(cur["trials"]) + cur["fails"] + 1
        req = CandidateRequest(id=f"tune-t{cur['batch']}-{k}", task_id=ctx.task_id, strategy=Strategy.DESIGN,
                               operator=Operator.IMPROVE, island=base.island, parent_ids=(base.id,),
                               brief=tune_brief(ctx, store, snapshot, cur), brief_kind="tune", family_hint=base.family)
        cur.update(pending=req.id, want=False)
        out.append(req)
    if out:
        store.set_state(K_TUNE, bk)
    return out


def tune_brief(ctx: LoopContext, store: Any, snapshot: Any, cur: Mapping[str, Any]) -> str:
    base = store.get_node(cur["base"])
    hyp = store.get_hypothesis(base.hypothesis_id or "")
    post = snapshot.nodes.get(base.id) if snapshot is not None else None
    left = cur["budget"] - len(cur["trials"])
    lines = [f"TUNING PASS t{cur['batch']} for {base.id} ({base.family}/{base.mechanism_tag}), trial "
             f"{len(cur['trials']) + 1} of {cur['budget']} ({left} left). {cur['reason']}.",
             f"Claim: {hyp.claim[:600] if hyp else '?'}",
             "It was screened at the incumbent's hyperparameters. Keep ITS MECHANISM exactly; change only its "
             "dose / hyperparameters so it gets a fair chance. EVERY variant must change at least one of its OWN "
             "mechanism-dose knobs (loss weight, mask ratio, schedule, temperature, ...: "
             f"{sorted(dose_keys(store, base, {})) or 'none exposed yet: expose one from its code'}); the shared "
             "sensitive ones (lr, warmup_fraction) only beside such a dose change, never alone. A knob "
             "hard-coded in the moonshot's code may be exposed as a config value or changed in place. Your diff "
             "is ON TOP of the moonshot (parent); make exactly ONE variant, different from every trial below."]
    if post is not None:
        lines.append(f"Untuned: U={post.utility.mean:+.4f}±{post.utility.sd:.4f}, P(U>0)={post.p_positive:.2f}, "
                     f"axes " + ", ".join(f"{a}={q.mean:+.4f}" for a, q in post.axes.items()))
    if cur["effects"]:
        lines.append("Phase-1 design (effect = high - low on the panel vs the moonshot, * = |effect| >= z se):")
        lines += [f"  {kb['key']}: {kb['low']!r} -> {kb['high']!r}: {D.fmt_effect(e)}"
                  for kb, e in zip(cur["knobs"], cur["effects"])]
    lines.append("Trials so far (panel delta vs the untuned moonshot on the shared tuning seeds):")
    for nid in cur["trials"]:
        n = store.get_node(nid)
        d, se, k = delta(ctx, store, cur["base"], nid)
        cfg = store.get_state(f"config_diff:{nid}") or {}
        lines.append(f"  {nid}: {cfg or (n.notes if n else '')} -> " + (f"{d:+.4f}±{se:.4f} (n={k})" if k else "not scored"))
    if not cur["trials"]:
        lines.append("  (none)")
    return "\n".join(lines)


def adopt(store: Any, request: CandidateRequest, node: Node) -> Node:
    """The driver's intake of a ``tune`` request: an admitted candidate becomes the pass's next trial (a config
    point on the shared block: strategy design, no operator, tag ``t<b>pt<i>``, screening at once); a rejected
    one counts as a failed proposal."""
    if request.brief_kind not in ("tune", "expose"):
        return node
    bk = book(store)
    cur = next((c for c in bk["current"].values() if c["pending"] == request.id), None)
    if cur is None:                             # its pass was withdrawn meanwhile: never a stray normal node
        return dataclasses.replace(node, status=NodeStatus.REJECTED, reject_reason="tune_pass_closed",
                                   notes=f"{node.notes} tune=closed")
    cur["pending"] = ""
    if request.brief_kind == "expose":          # the exposed commit: a carrier, never run unless not identical
        rejected = node.status is NodeStatus.REJECTED
        cur.update(expose="rejected" if rejected else "got", exposed="" if rejected else node.id,
                   expose_why=node.reject_reason if rejected else "")
        _save(store, bk, cur)
        return dataclasses.replace(node, notes=f"{node.notes} tune=t{cur['batch']} expose_of={cur['base']}") \
            if rejected else dataclasses.replace(
                node, strategy=Strategy.DESIGN, operator=None, family=DESIGN_FAMILY, mechanism_tag=f"t{cur['batch']}ex",
                status=NodeStatus.REJECTED, reject_reason=CARRIER,
                notes=f"{node.notes} tune=t{cur['batch']} expose_of={cur['base']}")
    if node.status is NodeStatus.REJECTED:
        cur["fails"] += 1
        _save(store, bk, cur)
        return dataclasses.replace(node, notes=f"{node.notes} tune=t{cur['batch']}")
    i = max(int(k) for k in cur["points"]) + 1 + sum(t not in cur["points"].values() for t in cur["trials"])
    cur["trials"] = cur["trials"] + [node.id]
    _save(store, bk, cur)
    return dataclasses.replace(node, strategy=Strategy.DESIGN, operator=None, family=DESIGN_FAMILY,
                               mechanism_tag=f"t{cur['batch']}pt{i:02d}", status=NodeStatus.SCREENING,
                               notes=f"{node.notes} tune=t{cur['batch']}")


def card(store: Any, node_id: str) -> str:
    """The tuning line of a node's dashboard card ('' if none): its pass."""
    bk = book(store)
    out = []
    for c in [*bk["batches"], *bk["current"].values()]:
        if c["base"] == node_id and c.get("kind") == "ladder":
            out.append(f"dose ladder t{c['batch']} {c.get('knob') or '-'}={c.get('value')}: " + (
                f"{c.get('why', '')}; tuned node {c.get('tuned') or 'none'}" if "closed" in c else "running"))
        elif c["base"] == node_id:
            out.append(f"tuning pass t{c['batch']} ({c['reason']}): {len(c['trials'])}/{c['budget']} trials"
                       + (f"; {c.get('why') or c.get('error', '')}; tuned node {c.get('tuned') or 'none'}"
                          if "closed" in c else f"; running, phase {c['phase']}"))
    return " | ".join(out)


def lines(store: Any) -> str:
    """One line per pass: running (phase, trials) or closed (why, best, tuned node)."""
    bk, out = book(store), []
    for c in bk["batches"]:
        if c.get("kind") == "ladder":
            out.append(f"ladder t{c['batch']} {c['base']} {c.get('knob') or '-'}={c.get('value')}: {c.get('why', '')}; "
                       f"tuned node {c.get('tuned') or 'none'}")
            continue
        out.append(f"tune t{c['batch']} {c['base']}: {c.get('why') or c.get('error', '?')}; best {c.get('best') or '-'}"
                   f"; tuned node {c.get('tuned') or 'none'}; {len(c['trials'])}/{c['budget']} trials")
    for c in bk["current"].values():
        if c.get("kind") == "ladder":
            out.append(f"ladder t{c['batch']} {c['base']} running: {c.get('knob')}={c.get('value')} at "
                       f"{sorted(c.get('doses', {}))} log2-dose, {len(c['trials'])} point(s)"
                       + (f", knob exposure {c['expose']}" if c.get("expose", "done") != "done" else ""))
            continue
        out.append(f"tune t{c['batch']} {c['base']} running: phase {c['phase']}, {len(c['trials'])}/{c['budget']} "
                   f"trials"
                   + (f", proposal {c['pending']} pending" if c["pending"] else ""))
    return "\n".join(out)


# ---------------------------------------------------------------------------------- the dose ladder
LADDER_TRIALS = 3          # a dose ladder's trials at most at its start: a backrun graft + its 1/2x and 2x rungs
_BOUNDED = re.compile(r"(ratio|prob|frac|fraction|momentum|ema|beta|rate|decay|_p)$")
B_PRIOR_VAR = 1.0          # the dose-response slope's prior variance: flat on the panel scale


def code_change(ctx: LoopContext, store: Any, node: Node) -> bool:
    """A CODE-changing moonshot / explore / idea-bank node (:func:`is_major`), never a design point, a trial or a
    tuned node: dose-ladder territory."""
    return (node.strategy in (Strategy.MOONSHOT, Strategy.EXPLORE) or bool(_note(node, "bank"))) \
        and not design_point(node) and not _note(node, "tune_of") and not _note(node, "tune") \
        and is_major(ctx, store, node)


def _vs_control(ctx: LoopContext, store: Any, node_id: str, ctl: str) -> list[float]:
    """The node's screen panels minus ``ctl``'s run at each seed (its control stage first, then a reused one)."""
    order = (Stage.BASELINE, Stage.PARENT, *S.REUSE_STAGES)
    cs: dict[int, float] = {}
    for r in sorted(store.runs(node_id=ctl), key=lambda r: order.index(r.stage) if r.stage in order else 99):
        if r.stage in order and (y := D.panel_of(ctx, r)) is not None:
            cs.setdefault(r.seed, y)
    return [y - cs[s] for s, y in screen_panels(ctx, store, node_id).items() if s in cs]


def screen_panels(ctx: LoopContext, store: Any, node_id: str) -> dict[int, float]:
    """seed -> the node's scored SCREEN panel."""
    return {r.seed: y for r in store.runs(node_id=node_id, stage=Stage.SCREEN) if (y := D.panel_of(ctx, r)) is not None}


def ladder_info(ctx: LoopContext, store: Any, snapshot: Any, node: Node) -> dict[str, Any] | None:
    """A closed (screened-out / not-confirmed) code change with scored screen seeds and shrunk P(own gain > 0) >=
    ``moonshot_tune.ladder_p`` -> {reason, p}; None otherwise. No confirm-seed requirement."""
    post = snapshot.nodes.get(node.id) if snapshot is not None else None
    bar = ctx.cfg.ideation.base_ladder_p if " brief=base" in f" {node.notes or ''}" else ctx.cfg.moonshot_tune.ladder_p
    if node.status not in (NodeStatus.SCREENED_OUT, NodeStatus.NOT_CONFIRMED) or post is None or not post.utility.n \
            or (p := p_gain(post)) < bar or not code_change(ctx, store, node) \
            or not screen_panels(ctx, store, node.id):
        return None
    from loop.novelty import node_type
    if node_type(node) in ctx.cfg.moonshot_tune.ladder_skip_types:       # an hparam-like mechanism: no ladder
        return None
    return {"reason": f"{node.status.value} code change at P(own gain>0)={p:.2f}: its dose was never searched", "p": p}


def mech_key(ctx: LoopContext, store: Any, node: Node) -> str:
    """The mechanism for one-ladder-per-mechanism: its retried dead cluster's tag (loop.novelty.dead_clusters: a
    dead mechanism tried n times is laddered once, at its most promising member), else its normalized tag."""
    from loop.novelty import dead_clusters, norm_tag
    return next((tag for tag, ids in dead_clusters(ctx, store) if node.id in ids), norm_tag(node.mechanism_tag))


def eligible_ladders(ctx: LoopContext, store: Any, snapshot: Any, now: float) -> list[tuple[Node, dict[str, Any]]]:
    """Ladder-eligible nodes (:func:`ladder_info`) without a ladder yet, highest P first (the backlog: P >= 0.7, then
    >= 0.5, ...), one per mechanism (:func:`mech_key`) ever (a node with no dose knob does not use it up), none while
    its family is in tuning-or-confirm. With
    ``expose`` on, a node whose ladder closed knob-less before exposures existed is BACKFILLED (laddered again)."""
    from loop.novelty import norm_tag
    from loop import campaign as CP
    bk, mt = book(store), ctx.cfg.moonshot_tune
    running = [c["base"] for c in bk["current"].values()]
    lad = [c for c in bk["batches"] if c.get("kind") == "ladder"]
    redo = {c["base"] for c in lad if not c.get("knob")} - {c["base"] for c in lad if c.get("knob") or "expose" in c} \
        if mt.expose else set()
    out = []
    from loop.core import bold as BOLD
    for n in store.nodes(status=[NodeStatus.SCREENED_OUT, NodeStatus.NOT_CONFIRMED]):
        if n.id in bk["laddered"] and n.id not in redo or n.id in running or CP.claims(ctx, store, n):
            continue
        if (room := BOLD.tune_room(ctx, store, n)) is not None and room < LADDER_TRIALS:
            continue                                   # v3-c8b: a bold family without room for a whole ladder
        if (c := ladder_info(ctx, store, snapshot, n)) is not None:
            out.append((n, c))
    out.sort(key=lambda nc: (-nc[1]["p"], nc[0].created_ts, nc[0].id))
    ran = [c["base"] for c in lad if c["trials"] and (c.get("knob") or not mt.expose)]   # a no-knob node: not its mechanism's ladder
    from loop.variants import bases as variant_bases
    out = [(n, c) for n, c in out if n.id not in variant_bases(store)]   # its variants already searched the dose
    done = {mech_key(ctx, store, b) for i in [*ran, *running] if (b := store.get_node(i)) is not None}
    busy, kept = busy_families(store), []
    for n, c in out:
        if (k := mech_key(ctx, store, n)) in done or busy.get(norm_tag(n.mechanism_tag), n.id) != n.id:
            continue
        done.add(k)
        kept.append((n, c))
    return kept


def ladder_levels(key: str, v: Any) -> list[Any]:
    """1/2x and 2x of ``v`` (an int rounds; a bounded knob -- ratio / prob / momentum ... -- in (0, 1) goes halfway to
    1 instead of past it); a level equal to ``v``, zero or across zero is dropped."""
    lo, hi = v / 2, v * 2
    if 0 < v < 1 <= hi and _BOUNDED.search(key.rsplit(".", 1)[-1]):
        hi = (1 + v) / 2
    lv = [round(lo), round(hi)] if isinstance(v, int) else [float(f"{lo:.4g}"), float(f"{hi:.4g}")]
    return [x for x in lv if x != v and x and (x > 0) == (v > 0)]


def dose_knob(ctx: LoopContext, store: Any, node: Node, text: str,
              base: Mapping[str, Any]) -> tuple[str, Any, list[Any]] | None:
    """(key, value, ladder levels) of the node's ONE dose knob: the researcher's declared ``dose=`` key, else a
    config key its diff introduced / changed (a postmortem tune knob among them first); numeric, non-zero, settable
    (design._knob_ok), never a shared knob (``moonshot_tune.shared``) or an ``lr``. None when it has none: a
    pre-existing key its code merely reads (train.global_views, train.batch_size, dino.layerwise_decay on the live
    backlog) is a SHARED knob, never its dose."""
    own = store.get_state(f"config_diff:{node.id}") or {}
    pm = [str(r["key"]) for r in postmortem_knobs(fresh_postmortem(store, node)) if str(r["key"]) in own]
    for key in dict.fromkeys([_note(node, "dose"), *pm, *sorted(own)]):
        v = base.get(key)
        if not key or key in ctx.cfg.moonshot_tune.shared or key.rsplit(".", 1)[-1] == "lr" or not D._num(v) or not v:
            continue
        lv = [x for x in ladder_levels(key, v) if D._knob_ok(ctx, text, base, {"key": key, "low": v, "high": x})]
        if lv:
            return key, v, lv
    return None


def dose_fit(points: Sequence[tuple[float, float, float]], prior: tuple[float, float]) -> dict[str, float]:
    """Quadratic dose response through 1x: r(x) = b x + c x^2 (x = log2-dose vs the node's own, r = panel delta vs
    the node at 1x, ``points`` = (x, r, se)), the slope flat, the curvature ~ N(prior mean, prior tau^2):
    the posterior (b, c, var c) and, with >= 2 distinct doses, the raw (unshrunk) curvature (c_raw, v_raw) the
    population prior is fitted from."""
    def solve(pc: float) -> tuple[float, float, float] | None:
        a11 = 1.0 / B_PRIOR_VAR + sum(x * x / se ** 2 for x, _, se in points)
        a12 = sum(x ** 3 / se ** 2 for x, _, se in points)
        a22 = pc + sum(x ** 4 / se ** 2 for x, _, se in points)
        r1 = sum(r * x / se ** 2 for x, r, se in points)
        r2 = pc * prior[0] + sum(r * x * x / se ** 2 for x, r, se in points)
        det = a11 * a22 - a12 * a12
        if det <= 0.0:
            return None
        return (a22 * r1 - a12 * r2) / det, (a11 * r2 - a12 * r1) / det, a11 / det
    b, c, vc = solve(1.0 / prior[1] ** 2) or (0.0, prior[0], prior[1] ** 2)
    out = {"b": b, "c": c, "vc": vc}
    if len({x for x, _, _ in points}) >= 2 and (raw := solve(0.0)) is not None:
        out.update(c_raw=raw[1], v_raw=raw[2])
    return out


def curvature_prior(ctx: LoopContext, store: Any) -> tuple[float, float]:
    """(mean, tau) of the dose-response curvature over every finished ladder's raw fit: the DerSimonian-Laird tau
    (world.shrinkage, the prior's floor / cap) and the precision-weighted mean; cold (< 2 ladders): (0, tau0)."""
    from loop.world.shrinkage import dersimonian_laird
    pr = ctx.cfg.prior
    rows = [(f["c_raw"], f["v_raw"]) for c in book(store)["batches"]
            if c.get("kind") == "ladder" and "c_raw" in (f := c.get("fit") or {})]
    tau = dersimonian_laird([rows], pr.tau_floor, pr.tau_max)
    if tau is None:
        return 0.0, pr.tau0
    w = [1.0 / (v + tau * tau) for _, v in rows]
    return sum(wi * c for wi, (c, _) in zip(w, rows)) / sum(w), tau


def _ladders(ctx: LoopContext, store: Any, adapter: Any, snapshot: Any, new_id: Any, now: float,
             scarce: bool) -> list[str]:
    """Start dose ladders while fewer than ``max_ladders_inflight`` run (scarce mode: P >= ``ladder_scarce_p`` only)."""
    mt, acts = ctx.cfg.moonshot_tune, []
    free = mt.max_ladders_inflight - sum(c.get("kind") == "ladder" for c in book(store)["current"].values())
    for node, info in eligible_ladders(ctx, store, snapshot, now) if free > 0 else []:
        if free <= 0:
            break
        if scarce and info["p"] < mt.ladder_scarce_p:
            continue
        store.log("ladder", node_id=node.id, queued=True, p=round(info["p"], 4), reason=info["reason"])
        try:
            got = _start_ladder(ctx, store, adapter, snapshot, node, info, new_id, now)
        except Exception as exc:  # noqa: BLE001 — a git / adapter failure closes the ladder, never the loop
            store.log("tune_error", node_id=node.id, error=repr(exc)[:500])
            got = [f"ladder {node.id} could not start: {exc!r}"[:300]]
        acts += got
        free -= any(c.get("kind") == "ladder" and c["base"] == node.id for c in book(store)["current"].values())
    return acts


def _backrun(ctx: LoopContext, store: Any, adapter: Any, snapshot: Any, node: Node, cur: dict, new_id: Any,
             now: float) -> str:
    """The ladder's 1x node: ``node`` itself when it is built on the incumbent (or no graft applies), else its
    change grafted onto the incumbent -- a screen-only point (parent: the incumbent) on a block of its own."""
    inc = getattr(snapshot, "incumbent_id", "") or ROOT_NODE_ID
    parent, onto = store.get_node(node.parent_id or ROOT_NODE_ID), store.get_node(inc)
    if parent is None or onto is None or parent.id == inc or not hasattr(adapter, "graft"):
        return node.id
    if prev := next((c["at"] for c in reversed(book(store)["batches"]) if c["base"] == node.id
                     and c.get("on") == inc and c.get("at") == (c.get("doses") or {}).get("+0")), ""):
        cur.update(at=prev, on=inc, doses={"+0": prev}, trials=[prev])   # a backfill: its 1x graft already ran
        return prev
    commit = adapter.graft(parent.commit, node.commit, onto.commit, f"bk{cur['batch']}-{node.id}")
    if not commit:
        cur["graft"] = f"did not apply on {inc}"
        store.log("ladder_graft_failed", node_id=node.id, onto=inc)
        return node.id
    gid, tag = new_id(), f"t{cur['batch']}"
    store.put_node(Node(
        id=gid, task_id=ctx.task_id, commit=commit, parent_id=inc, strategy=Strategy.DESIGN, operator=None,
        family=DESIGN_FAMILY, mechanism_tag=f"{tag}pt00", diff_hash=diff_hash(f"backrun:{commit}"),
        status=NodeStatus.SCREENING, island=node.island, diff_path=node.diff_path, levers=node.levers,
        created_ts=now, updated_ts=now, notes=f"tune={tag} backrun_of={node.id} ladder=+0"))
    store.set_state(f"config_diff:{gid}", dict(store.get_state(f"config_diff:{node.id}") or {}))
    store.log("node_transition", node_id=gid, frm="new", to=NodeStatus.SCREENING.value,
              reason=f"backrun of {node.id} ({parent.id}) on incumbent {inc} via {tag}")
    cur.update(at=gid, on=inc, doses={"+0": gid}, trials=[gid])
    return gid


def _start_ladder(ctx: LoopContext, store: Any, adapter: Any, snapshot: Any, node: Node, info: Mapping[str, Any],
                  new_id: Any, now: float) -> list[str]:
    bk = book(store)
    b = bk["seq"] + 1
    tag = f"t{b}"
    bk["seq"], bk["laddered"] = b, bk["laddered"] + [node.id]          # ONE ladder per node, even on failure
    cur = {"batch": b, "base": node.id, "kind": "ladder", "budget": 0, "major": True, "rough": False,
           "reason": info["reason"], "p": info["p"], "knob": "", "value": None, "doses": {}, "points": {"0": node.id},
           "trials": [], "knobs": [], "rows": [], "phase": 1, "pending": "", "fails": 0, "effects": [],
           "started": now, "granted": False, "seeds": sorted(screen_panels(ctx, store, node.id))}
    bk["current"][str(b)] = cur
    store.set_state(K_TUNE, bk)
    at = _backrun(ctx, store, adapter, snapshot, node, cur, new_id, now)
    _dose_points(ctx, store, adapter, cur, node, at, new_id)
    if not cur["knob"] and ctx.cfg.moonshot_tune.expose:
        cur["expose"] = "want"                  # no dose knob: a researcher exposes the hard-coded strength first
    cur["budget"] = len(cur["trials"])
    bk = book(store)
    if not cur["trials"] and not cur.get("expose"):
        why = "no dose knob (none declared or changed by its diff)" if not cur["knob"] else "no ladder point admitted"
        _close(store, bk, cur, now, why=why, tuned="", best="", gpu_h=0.0)
        store.log("ladder_done", batch=b, node_id=node.id, why=why)
        return [f"ladder {tag} for {node.id}: {why}"]
    bk["current"][str(b)] = cur
    store.set_state(K_TUNE, bk)
    store.update_node(node.id, notes=f"{node.notes or ''} tuning={tag}".strip())
    store.log("ladder_started", batch=b, node_id=node.id, knob=cur["knob"], value=cur["value"], doses=cur["doses"],
              seeds=cur["seeds"], on=cur.get("on", ""), graft=cur.get("graft", ""))
    where = (f"BACKRUN on incumbent {cur['on']} (graft {cur['at']}, a fresh block, vs the incumbent's controls)"
             if cur.get("on") else f"on its own screen seeds {cur['seeds']}, paired vs itself"
             + (f" (graft {cur['graft']})" if cur.get("graft") else ""))
    return [f"ladder {tag} for {node.id} ({node.mechanism_tag}, {info['reason']}): {cur['knob'] or 'no dose knob'}"
            f"={cur['value']} at log2-dose {sorted(cur['doses'])} {where}"
            + ("; knob exposure requested" if cur.get("expose") else "")]


def _dose_points(ctx: LoopContext, store: Any, adapter: Any, cur: dict, node: Node, tree_id: str,
                 new_id: Any) -> None:
    """The ladder's 1/2x and 2x config points of ``node``'s dose knob, written on ``tree_id``'s commit, on the 1x
    node's block (``at``); a point built on an identical exposure (``tree_id`` != ``at``) is parented on ``at``."""
    at, tag = cur.get("at") or cur["base"], f"t{cur['batch']}"
    with D._Tree(ctx, store, adapter, f"tune{cur['batch']}", base_id=tree_id) as t:
        if (got := dose_knob(ctx, store, node, t.text, t.base)) is None:
            return
        key, v, levels = got
        cur.update(knob=key, value=v)
        for i, lv in enumerate(levels, start=1):
            x = f"{math.log2(lv / v):+.3g}"
            nid = t.node({key: lv}, knobs=[{"key": key, "low": v, "high": lv, "levers": {}}], tag=f"{tag}pt{i:02d}",
                         operator=None, new_id=new_id, notes=f"tune={tag} block_of={at} ladder={x}")
            if nid and nid != tree_id:
                if tree_id != at:
                    store.update_node(nid, parent_id=at)
                cur["doses"][x] = nid
                cur["trials"].append(nid)


def expose_brief(ctx: LoopContext, store: Any, cur: Mapping[str, Any]) -> str:
    base = store.get_node(cur["base"])
    hyp = store.get_hypothesis(base.hypothesis_id or "")
    return "\n".join([
        f"KNOB EXPOSURE for dose ladder t{cur['batch']} of {base.id} ({base.family}/{base.mechanism_tag}). "
        f"{cur['reason']}.", f"Claim: {hyp.claim[:600] if hyp else '?'}",
        "Its mechanism's strength is hard-coded (no config key sets it), so its dose could not be searched. Change "
        "NOTHING about what the code computes: move the ONE constant that sets the mechanism's strength (a loss "
        "weight, temperature, mask ratio, mixing coefficient, ...; for a DATA-SAMPLING change the sampling strength, "
        "e.g. the power / temperature / mixing fraction of its reweighting, where 0 would mean no reweighting) into "
        "the config as ONE new key whose default is the CURRENT value, read it where the constant was, and declare "
        "it as hypothesis.dose_knob (never lr, warmup or a shared knob). At the default the run must be numerically "
        "identical to the parent (same ops in the same order, same RNG draws). Smoke it and check that; set "
        "hypothesis.identical_at_default true ONLY when you verified it (false: the new commit is re-measured at 1x). "
        "The ladder then runs it at 1/2x and 2x. Keep the parent's hypothesis, family and mechanism_tag.",
        "The key MUST be written into the config file (a code default alone cannot be set by a ladder point), hold "
        "ONE number, and match a key pattern of a family whose changes need no init check (prefer the moonshot's "
        f"{base.family}): " + ", ".join(f"{g} ({f})" for f, r in ctx.cfg.families.items() if not r.init_check
                                        for g in r.keys if "*" in g) + "."]
        + ([f"The previous exposure failed: {cur['expose_why']}."] if cur.get("expose_why") else []))


def _expose(ctx: LoopContext, store: Any, adapter: Any, bk: dict, cur: dict, now: float, inflight: set[str],
            new_id: Any) -> list[str]:
    """A ladder waiting on its knob exposure: retried up to ``max_research_fails``; got: identical at 1x (declared,
    and no other config key changed) -> the points are built on the exposed commit beside the measured 1x node;
    else the exposed commit is a new 1x point (screened on the 1x node's block). Failed: a backrun keeps its 1x
    point, any other ladder closes."""
    tag, st = f"t{cur['batch']}", cur["expose"]
    if st == "done" or st == "want" or (st == "pending" and cur["pending"] in inflight):
        if st == "done":                      # exposed but no point and no 1x: nothing to run
            _close(store, bk, cur, now, why="the exposure gave no usable dose knob", tuned="", best="", gpu_h=0.0)
            return [f"ladder {tag} for {cur['base']}: the exposure gave no usable dose knob"]
        return []
    if st in ("pending", "rejected"):         # the request produced nothing, or a rejected candidate
        cur.update(pending="", fails=cur["fails"] + 1)
        why = cur.get("expose_why") or "no candidate"
        store.log("ladder_expose_failed", batch=cur["batch"], node_id=cur["base"], fails=cur["fails"], why=why)
        if cur["fails"] < ctx.cfg.moonshot_tune.max_research_fails:
            cur["expose"] = "want"
            _save(store, bk, cur)
            return [f"ladder {tag}: knob exposure failed ({why}); asking again ({cur['fails']} failed)"]
        cur["expose"] = "done"
        _save(store, bk, cur)
        if not cur["trials"]:
            _close(store, bk, cur, now, why=f"knob exposure failed {cur['fails']}x ({why})", tuned="", best="",
                   gpu_h=0.0)
        return [f"ladder {tag} for {cur['base']}: knob exposure failed {cur['fails']}x ({why})"
                + ("; the 1x backrun continues" if cur["trials"] else "")]
    ex = store.get_node(cur["exposed"])
    key = _note(ex, "dose")
    same = " identical=1" in f" {ex.notes}" and set(store.get_state(f"config_diff:{ex.id}") or {}) <= {key}
    was = cur.get("at")                       # a backrun's graft; None: the node itself
    old = was or cur["base"]
    if not same:                              # a new 1x point: tried first so a no-knob exposure never runs
        cur["at"] = ex.id
    _dose_points(ctx, store, adapter, cur, ex, ex.id, new_id)
    if not cur["knob"]:
        cur.pop("at", None) if was is None else cur.update(at=was)
        # an exposure whose declared key no ladder point can set is a FAILED exposure, asked again with the why
        # (live 2026-10-04: 6 of 14 exposures read data.<new key> as a code default -- in no config file, no family)
        store.update_node(ex.id, reject_reason="expose_unusable")
        cur.update(expose="rejected", exposed="", expose_why=f"declared dose knob {key or '(none)'!r} is not a single "
                   "number in the config file under a key of the moonshot's family, so no ladder point can set it")
        return _expose(ctx, store, adapter, bk, cur, now, inflight, new_id)
    if not same:
        parent = cur.get("on") or store.get_node(cur["base"]).parent_id or ROOT_NODE_ID
        store.update_node(ex.id, status=NodeStatus.SCREENING, reject_reason="", parent_id=parent,
                          notes=f"{ex.notes} block_of={old} ladder=+0")
        cur["doses"]["+0"] = ex.id
        cur["trials"].append(ex.id)
    cur.update(expose="done", identical=same, budget=len(cur["trials"]))
    _save(store, bk, cur)
    store.log("ladder_exposed", batch=cur["batch"], node_id=cur["base"], exposed=ex.id, knob=cur["knob"],
              identical=same, doses=cur["doses"])
    if not cur["trials"]:
        return _expose(ctx, store, adapter, bk, cur, now, inflight, new_id)
    return [f"ladder {tag} for {cur['base']}: exposed {cur['knob'] or 'no usable knob'}={cur['value']} in {ex.id} "
            + ("(identical at 1x: the 1x measurements stand)" if same else "(not identical: re-measured at 1x)")
            + f"; log2-dose {sorted(cur['doses'])}"]


def _retry_void_points(ctx: LoopContext, store: Any, cur: dict, now: float, new_id: Any) -> list[str]:
    """A dose point (never the 1x) that ended VOID is re-run ONCE -- a fresh node, same commit and config, on the
    1x block -- before the ladder fits (live 2026-10-03: t17/t31/t34 closed on their 1x alone, every 1/2x and 2x
    point void:partner)."""
    at, tag, acts = cur.get("at") or cur["base"], f"t{cur['batch']}", []
    for x, nid in sorted(cur["doses"].items()):
        n = store.get_node(nid)
        if nid == at or n is None or n.status is not NodeStatus.VOID or nid in cur.get("retried", []):
            continue
        rid = new_id()
        store.put_node(dataclasses.replace(n, id=rid, status=NodeStatus.SCREENING, reject_reason="", created_ts=now,
                                           updated_ts=now, notes=f"{n.notes} retry_of={nid}"))
        store.set_state(f"config_diff:{rid}", dict(store.get_state(f"config_diff:{nid}") or {}))
        cur["doses"][x] = rid
        cur["trials"] = [rid if t == nid else t for t in cur["trials"]]
        cur["retried"] = cur.get("retried", []) + [nid, rid]      # once: the retry itself is never retried
        store.log("node_transition", node_id=rid, frm="new", to=NodeStatus.SCREENING.value,
                  reason=f"retry of void ladder point {nid} ({tag} log2-dose {x})")
        acts.append(f"ladder {tag} for {cur['base']}: log2-dose {x} point {nid} was void; retried once as {rid}")
    return acts


def reopen_void_ladders(ctx: LoopContext, store: Any, now: float, new_id: Any) -> list[str]:
    """A CLOSED ladder with a void dose point never retried (:func:`_retry_void_points` did not exist yet) is
    re-opened with that point retried; it fits again when the retries close."""
    bk, acts = book(store), []
    for c in [c for c in bk["batches"] if c.get("kind") == "ladder" and not c.get("error")]:
        cur = {k: v for k, v in c.items() if k not in ("why", "tuned", "best", "gpu_h", "rows", "fit", "deltas",
                                                          "want", "closed")}
        if not (got := _retry_void_points(ctx, store, cur, now, new_id)):
            continue
        bk["batches"] = [b for b in bk["batches"] if b is not c]
        bk["current"][str(cur["batch"])] = cur
        store.set_state(K_TUNE, bk)
        store.drop_pulls(f"{cur['base']}#tune")       # its close's charge: paid again at the new close
        store.log("ladder_reopened", batch=cur["batch"], node_id=cur["base"], doses=cur["doses"])
        acts += got + [f"ladder t{cur['batch']} for {cur['base']} re-opened (was: {c.get('why', '')})"]
    return acts


def _edge_rung(ctx: LoopContext, store: Any, adapter: Any, snapshot: Any, cur: dict, new_id: Any) -> list[str]:
    """RANGE EDGE: when the dose fit's argmax is the ladder's top (or bottom) rung with a predicted gain over 1x > 0,
    the next rung (its dose x2 again, or /2) joins the 1x block before the ladder fits, within the knob's hard bounds
    (loop.extrapolate.inside) -- a dose response still rising at the ladder's edge is followed."""
    from loop.extrapolate import inside
    from loop.core import bold as BOLD
    if (base := store.get_node(cur["base"])) is not None and (room := BOLD.tune_room(ctx, store, base)) is not None \
            and room < 1:
        return []                                      # v3-c8b: the bold family's tuning budget is spent (an edge rung is a trial)
    at, key, v = cur.get("at") or cur["base"], cur.get("knob"), cur.get("value")
    if not key or adapter is None or not D._num(v) \
            or cur.get("edges", 0) >= getattr(getattr(ctx.cfg, "extrapolate", None), "edge_max", 0):
        return []
    ys, sig = screen_panels(ctx, store, at), float(snapshot.sigma.get(PANEL, ctx.cfg.panel_sd))
    rows = {}
    for x, nid in cur["doses"].items():
        if nid != at and (ds := [y - ys[s] for s, y in screen_panels(ctx, store, nid).items() if s in ys]):
            rows[float(x)] = (statistics.fmean(ds), sig / len(ds) ** 0.5)
    if not rows:
        return []
    fit = dose_fit([(x, r[0], r[1]) for x, r in rows.items()], curvature_prior(ctx, store))
    pred = {x: fit["b"] * x + fit["c"] * x * x for x in rows}
    best = max(pred, key=lambda x: (pred[x], x))
    if pred[best] <= 0.0 or best not in (max(rows), min(rows)) or best == 0.0:
        return []
    nx = best + (1.0 if best > 0 else -1.0)
    lv = round(v * 2 ** nx) if isinstance(v, int) else float(f"{v * 2 ** nx:.4g}")
    tag, xs = f"t{cur['batch']}", f"{nx:+.3g}"
    if xs in cur["doses"] or lv == v or not inside(ctx, key, lv, v):
        return []
    tree = cur.get("exposed") or at                 # an identical exposure: the knob lives on its commit
    with D._Tree(ctx, store, adapter, f"tune{cur['batch']}e{xs}", base_id=tree) as t:
        if not D._knob_ok(ctx, t.text, t.base, {"key": key, "low": v, "high": lv}):
            return []
        nid = t.node({key: lv}, knobs=[{"key": key, "low": v, "high": lv, "levers": {}}], tag=f"{tag}pt{len(cur['doses']) + 1:02d}",
                     operator=None, new_id=new_id, notes=f"tune={tag} block_of={at} ladder={xs}")
    if not nid or nid == tree:
        return []
    if tree != at:
        store.update_node(nid, parent_id=at)
    cur["doses"][xs], cur["trials"] = nid, cur["trials"] + [nid]
    cur["budget"], cur["edges"] = len(cur["trials"]), cur.get("edges", 0) + 1
    store.log("ladder_edge", batch=cur["batch"], node_id=cur["base"], knob=key, dose=xs, level=lv, point=nid)
    return [f"ladder {tag} for {cur['base']}: the fit's best rung log2-dose {best:+.3g} ({key}) predicted "
            f"{pred[best]:+.4f} over 1x at the ladder's edge -> next rung {key}={lv} (log2-dose {xs}) as {nid}"]


def _finish_ladder(ctx: LoopContext, store: Any, snapshot: Any, bk: dict, cur: dict, now: float,
                   new_id: Any) -> list[str]:
    """Fit the ladder (:func:`dose_fit`, curvature prior :func:`curvature_prior`) and graduate the argmax-posterior
    dose when it is not 1x and beats 1x; the rows, the fit and the seeds go to the tune history."""
    tag, base = f"t{cur['batch']}", store.get_node(cur["base"])
    at = cur.get("at") or base.id                 # the 1x node the points pair against (a backrun: the graft)
    ys = screen_panels(ctx, store, at)
    cur["seeds"] = sorted(ys)
    plan = ctx.cfg.seeds
    if clash := set(cur["seeds"]) & {int(s) for s in (*plan.promote, *plan.confirm_pool, *plan.audit)}:
        raise ValueError(f"ladder seeds {sorted(clash)} overlap promote/confirm/audit seeds: no fresh-seed verdict")
    sig = float(snapshot.sigma.get(PANEL, ctx.cfg.panel_sd))
    rows = {}
    for x, nid in cur["doses"].items():
        ds = [y - ys[s] for s, y in screen_panels(ctx, store, nid).items() if s in ys] if nid != at else \
            _vs_control(ctx, store, at, cur.get("on") or ROOT_NODE_ID)      # a backrun's 1x: gain over the incumbent
        if ds:
            rows[x] = [statistics.fmean(ds), sig / len(ds) ** 0.5, len(ds), nid]
    prior = curvature_prior(ctx, store)
    pts = {x: r for x, r in rows.items() if r[3] != at}
    fit = dose_fit([(float(x), r[0], r[1]) for x, r in pts.items()], prior)
    pred = {x: fit["b"] * float(x) + fit["c"] * float(x) ** 2 for x in pts}
    if cur.get("on"):            # a backrun: every dose (1x too) on its gain over the incumbent; no 1x row: no pick
        g0 = rows.get("+0", [None])[0]
        pred = {} if g0 is None else {"+0": g0, **{x: g0 + p for x, p in pred.items()}}
    best = max(pred, key=lambda x: (pred[x], x), default="")
    gpu = tune_gpu_h(store, base.id)
    result: dict[str, Any] = {"why": "", "tuned": "", "best": rows[best][3] if best else "", "gpu_h": gpu,
                              "rows": rows, "fit": {**fit, "prior": list(prior), "pred": pred},
                              "deltas": {r[3]: r[:3] for r in rows.values()}}
    line = ", ".join(f"{x}: {r[0]:+.4f}±{r[1]:.4f} (fit {pred[x]:+.4f})" for x, r in sorted(rows.items())) or "no scored point"
    over = f"the incumbent {cur['on']}" if cur.get("on") else "1x"
    if best and pred[best] > 0.0:
        result["why"] = f"dose {cur['knob'] or '-'} x{2 ** float(best):.3g} predicted {pred[best]:+.4f} over {over}"
        result["tuned"] = _tuned_node(ctx, store, snapshot, base, store.get_node(rows[best][3]), cur, pred[best],
                                      new_id, now, parent=cur.get("on", ""))
    else:
        result["why"] = (f"no dose predicted above the incumbent {cur['on']}" if cur.get("on")
                         else "1x (the node's own dose) is the argmax: no better dose")
        store.record_pull(ArmPull(task_id=ctx.task_id, arm=Strategy.MOONSHOT, node_id=f"{base.id}#tune",
                                  reward=0.0, gpu_h=gpu, ts=now))
    _close(store, bk, cur, now, **result)
    store.log("ladder_done", batch=cur["batch"], node_id=base.id, why=result["why"], tuned=result["tuned"],
              rows=rows, fit=result["fit"])
    return [f"ladder {tag} for {base.id} ({cur['knob']}={cur['value']}; curvature prior {prior[0]:+.4f}±{prior[1]:.4f}): "
            f"{line} -> {result['why']}" + (f"; tuned node {result['tuned']} -> promote/confirm/audit on fresh seeds"
                                             if result["tuned"] else "")]
