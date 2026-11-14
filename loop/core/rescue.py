"""loop.core.rescue -- the MOONSHOT RESCUE (operator 11-10: "tune structural moonshots before judging, <= 5 trials for
promising major changes, retro-revisit closed ones, fix-and-continue crashes"). Live: a node sparse_upcycled_moe_mlp
-.052 and a node hemi_swap_detect -.046 were each judged on ONE untuned seed with no bug check; a node / a node / a node /
a node / a node sat flat on one seed of an old parent; a node natinit_sincos_pos3d was rejected ``bold_knob_only`` untested.

It reuses the loop's machinery: the researcher's DEBUG operator (a session on the node's commit, judged against its
control), ``cutover.port_node`` (the change grafted onto the incumbent), the structural ladder of
:mod:`loop.core.struct_tune` (config-only children tagged ``ladder=``) and the held / futility exemptions those carry.

WHICH NODES (:func:`candidates`): a bold / structural node (:func:`loop.core.struct_tune.structural_hint`) of the current
base, not the incumbent, not a ladder child / port / program arm, derived SCREENING (or CRASHED: every run failed, or
a REJECTED one the knob re-judge restored), with no run in flight and <= ``moonshot_rescue_max_seeds`` paired seeds.
Its evidence is its RAW paired gain over its parent on the shared done seeds (``se = panel_sd / sqrt(n)``):

* CATASTROPHIC (mean < -``moonshot_cata_k`` x panel_sd or < -``moonshot_cata_abs``), or all-crashed: ONE researcher
  DEBUG session (:func:`debug_sessions`, request ``req-v3-fix-dbg-<node>-1``): it reads the diff and the run logs
  (output dirs, log tails: loss curve, NaN, grad norms) for an implementation bug. A fix is a child (re-parented
  onto the node's control: its diff is judged against it), which is then rescued like any candidate; no child = "no
  bug found" (closed). A debug child is never debugged again (max 1 debug per lineage).
* NOT CLEARLY NEGATIVE (mean > -``moonshot_rescue_neg_se`` x se): RESCUED (:func:`step`). Its parent is not the
  incumbent: its change (over its parent) is grafted onto the incumbent (``cutover.port_node``, tagged ``ladder=<node>
  struct_retro=1 moonshot_rescue=<node>``: program priority, futility-exempt below 3 seeds); its parent is the
  incumbent: the node itself is tagged and tuned in place. ``struct_tune.struct_ladders`` then builds <=
  ``moonshot_rescue_rungs`` config children over the 1-2 knobs its researcher declared (``components`` lo / hi; none:
  the struct_tune default -- its own dose keys, else the LR x weight-decay grid), outside ``struct_ladder_max_open``.
  The port and every rung are held at ``moonshot_rescue_seeds`` screen seeds (:func:`hold`) until each has them; the
  best of the port + rungs (shifted to the incumbent) is released to the normal screen-to-5 path, the rest stay held.
  Port + rungs <= 5 trials.
* clearly negative: booked ``-: ...`` (never revisited).

BUDGET (:func:`step`): <= ``moonshot_rescue_max`` undecided rescues and <= ``moonshot_debug_max_open`` debug sessions
pending / in flight (a debug costs a researcher session, no GPU), each chosen by
PRIOR PLAUSIBILITY (the bold lane's EB prior cell mean + the researcher's predicted median, the slate plausibility as
tie-break; nodes never tuned before go first), never by the 1-seed delta. No new booking while rescue runs hold >=
``moonshot_rescue_gpu_share`` of the capacity (and the rescue's queued nodes then wait). A test block is never held.

STATE ``ideation.moonshot_rescue`` {node: {phase debug|ported|done, ...} | "-: why"}: a node is booked ONCE (restarts
re-read it), so a node is never enqueued twice. RETRO (:func:`retro`, once per deploy): the knob judge's past
``bold_knob_only`` rejections whose diff adds code (``bold.code_override``) are restored (a node), and the backlog of
closed / stalled moonshots is listed (event ``moonshot_rescue_retro``) -- the per-tick pass enqueues them under the cap.

EVENTS ``moonshot_debug`` / ``moonshot_rescue``; ``loop status`` shows :func:`lines`."""

from __future__ import annotations

import math
import re
import time
from typing import Any, Mapping, Sequence

K_RESCUE = "ideation.moonshot_rescue"
K_RETRO = "ideation.moonshot_rescue_retro"     # {ts, version, rejudged, backlog}: the once-per-deploy retro pass
TAG = " moonshot_rescue="
DEBUG_TAG = " debug_of="
DEAD = frozenset({"crash", "void"})
LIVE_RUN = frozenset({"running", "submitting"})


def opt(ctx: Any, name: str) -> Any:
    from loop.core.levers import opt as o
    return o(ctx, name)


def on(ctx: Any) -> bool:
    return bool(opt(ctx, "moonshot_rescue"))


def _notes(node: Any) -> str:
    return f" {getattr(node, 'notes', '') or ''} "


def debug_prefix() -> str:
    from loop.core.ideation import REPAIR_PREFIX        # req-v3-fix-: a repair (cluster-exempt, never a duplicate)
    return REPAIR_PREFIX + "dbg-"


def book(store: Any) -> dict[str, Any]:
    return dict(store.get_state(K_RESCUE) or {})


def _status(s: Any) -> str:
    s = getattr(s, "status", s)
    return str(getattr(s, "name", None) or getattr(s, "value", None) or s).upper()


def incumbent(statuses: Mapping[str, Any]) -> str:
    from loop.contracts import ROOT_NODE_ID
    return next((n for n, s in statuses.items() if _status(s) == "INCUMBENT"), ROOT_NODE_ID)


def is_open(rec: Any) -> bool:
    """A booking that takes a rescue slot: a debug pending / in flight, or a rescue whose ladder is not decided."""
    return isinstance(rec, Mapping) and (
        (rec.get("phase") == "debug" and rec.get("status") in ("pending", "started"))
        or rec.get("phase") in ("ported", "port_session"))


def claimed(store: Any) -> set[str]:
    """Nodes the rescue decides (struct_tune never ladders them in place): every node it booked as a dict, the ports
    it made and the debug children."""
    out: set[str] = set()
    for nid, rec in book(store).items():
        if isinstance(rec, Mapping):
            out.add(nid)
            out |= {x for x in (rec.get("child"),) if x}
    return out


def owed(store: Any) -> dict[str, str]:
    """node -> why it may not be CLOSED as negative yet (``struct_tune.held``: no negative cluster evidence, no early
    kill): a node under debug, a rescued original and its port until the rescue's ladder is decided. Never raises."""
    out: dict[str, str] = {}
    try:
        for nid, rec in book(store).items():
            if not is_open(rec):
                continue
            if rec.get("phase") == "debug":
                out[nid] = "moonshot debug in flight (loop.core.rescue)"
            else:
                out[nid] = f"moonshot rescue: ported as {rec.get('port') or '?'}, its ladder is undecided"
                if rec.get("port"):
                    out[rec["port"]] = f"moonshot rescue of {nid}: its ladder is undecided"
    except Exception:  # noqa: BLE001 - a broken book holds nothing
        return {}
    return out


def origin(node: Any) -> str:
    m = re.search(r"(?:^|\s)moonshot_rescue=(\S+)", _notes(node))
    return m.group(1) if m else ""


def spec_of(store: Any, node: Any) -> dict[str, Any]:
    """The rescue ladder's explicit points (struct_tune ``explicit_points``) of a rescued node."""
    rec = book(store).get(origin(node))
    return dict(rec.get("spec") or {}) if isinstance(rec, Mapping) else {}


# ---------------------------------------------------------------------------------------------- evidence / ranking
def evidence(led: Any, nid: str, parent: str | None = None) -> dict[str, Any]:
    """The node's RAW paired gain over ``parent`` (default its parent) on the done seeds both ran (official panel)."""
    from loop.core.data import PANEL
    from loop.core.model import seed_means
    n = led.nodes.get(nid)
    ref = parent or (n.parent if n is not None else None)
    by = led.by_node()
    c, p = seed_means(by.get(nid, ()), PANEL), seed_means(by.get(ref, ()) if ref else (), PANEL)
    d = {s: c[s] - p[s] for s in sorted(c.keys() & p.keys())}
    return {"vs": ref or "", "n": len(d), "mean": (sum(d.values()) / len(d)) if d else None,
            "seeds": {str(s): round(x, 6) for s, x in d.items()}}


def panel_sd(ctx: Any) -> float | None:
    sd = getattr(getattr(ctx, "cfg", None), "panel_sd", None)
    try:
        sd = float(sd)
    except (TypeError, ValueError):
        return None
    return sd if math.isfinite(sd) and sd > 0 else None


def verdict(ctx: Any, ev: Mapping[str, Any], crashed: bool = False) -> tuple[str, str]:
    """('debug' | 'rescue' | 'negative' | 'wait', why) of one candidate's evidence."""
    n, m, sd = int(ev.get("n") or 0), ev.get("mean"), panel_sd(ctx)
    if n == 0 or m is None:
        return ("debug", "every run crashed") if crashed else ("wait", "no paired seed yet")
    k, cabs = float(opt(ctx, "moonshot_cata_k")), float(opt(ctx, "moonshot_cata_abs"))
    if (sd is not None and m < -k * sd) or m < -cabs:
        bar = f"-{k:g} x panel_sd {sd:.4f} or " if sd is not None else ""
        return "debug", f"catastrophic first screen {m:+.4f} on {n} seed(s) (< {bar}-{cabs:g})"
    se = (sd / math.sqrt(n)) if sd is not None else None
    z = float(opt(ctx, "moonshot_rescue_neg_se"))
    if se is None or m > -z * se:
        return "rescue", f"not clearly negative: {m:+.4f} on {n} seed(s)" + (f" > -{z:g} se ({se:.4f})" if se else "")
    return "negative", f"clearly negative: {m:+.4f} on {n} seed(s) <= -{z:g} se ({se:.4f})"


def plausibility(store: Any, post: Any, nid: str) -> tuple[float, float]:
    """(prior score, tie-break): the bold lane's EB prior cell mean for the node + the researcher's predicted median
    gain (``ideation:<node>`` prediction / mechanism_prediction), and the slate plausibility of the built candidate."""
    pm = 0.0
    try:
        pm = float(post.prior.cell(post.cell(nid))[0])
    except Exception:  # noqa: BLE001 - no prior view: the researcher's own expectation alone
        pm = 0.0
    plan = store.get_state(f"ideation:{nid}") or {}
    med = 0.0
    for key in ("prediction", "mechanism_prediction"):
        p = plan.get(key) if isinstance(plan, Mapping) else None
        if isinstance(p, Mapping) and isinstance(p.get("median", p.get("size")), (int, float)):
            med = float(p.get("median", p.get("size")))
            break
    pl = 0.0
    try:
        sl = plan.get("slate") or {}
        rows = sl.get("rows") or []
        ch = int(sl.get("chosen") or 0)
        pl = float(next((r.get("plausibility") for r in rows if int(r.get("i", -1)) == ch), 0.0) or 0.0)
    except Exception:  # noqa: BLE001 - no slate
        pl = 0.0
    return pm + med, pl


def _structural(node: Any) -> bool:
    from loop.core import struct_tune as STT
    n = _notes(node)
    if STT.is_child(node) or (TAG in n and origin(node) != node.id) or " program_arm=1" in n or " local_confirm=" in n \
            or re.search(r"(?:^|\s)(?:wave|point)=", n):
        return False
    if getattr(node, "port_of", "") or " port_of=" in n:
        return False
    return DEBUG_TAG in n or STT.structural_hint(node)


def _active(led: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in led.runs:
        if r.status in LIVE_RUN:
            out[r.node] = out.get(r.node, 0) + 1
    return out


def _chain(led: Any, inc: str) -> set[str]:
    out, cur = set(), inc
    while cur and cur not in out:
        out.add(cur)
        n = led.nodes.get(cur)
        cur = n.parent if n is not None else None
    return out


def candidates(ctx: Any, store: Any, led: Any, statuses: Mapping[str, Any], post: Any, inc: str,
               now: float | None = None) -> list[dict[str, Any]]:
    """The not-yet-booked bold / structural nodes with a decision (module docstring), in PRIORITY order: never-tuned
    nodes first, then by prior plausibility (:func:`plausibility`). ``verdict`` 'wait' ones are left out."""
    from loop.core import struct_tune as STT
    bk, active, anc = book(store), _active(led), _chain(led, inc)
    sbk, by = STT.book(store), led.by_node()
    now = time.time() if now is None else now
    need, sseeds = int(opt(ctx, "struct_min_points")), int(opt(ctx, "struct_min_seeds"))
    hold_s = float(opt(ctx, "struct_hold_h")) * 3600.0
    cap = int(opt(ctx, "moonshot_rescue_max_seeds"))
    stall_s = float(opt(ctx, "moonshot_rescue_stall_h")) * 3600.0
    submitted: dict[str, float] = {}
    for r in led.runs:
        submitted[r.node] = max(submitted.get(r.node, -math.inf), float(r.submitted_ts or 0.0))
    out = []
    for nid, node in led.nodes.items():
        if nid in bk or nid in anc or node.base != led.epoch or not _structural(node) or active.get(nid):
            continue
        st = _status(statuses.get(nid, ""))
        sn = store.get_node(nid)
        rejudged = sn is not None and " knob_rejudged=1" in _notes(sn)
        fix = DEBUG_TAG in _notes(node) or (sn is not None and DEBUG_TAG in _notes(sn))
        if st not in ("SCREENING", "CRASHED") and not (rejudged and st in ("VOID", "QUEUED", "")) \
                and not (fix and st in ("QUEUED", "")):
            continue
        if (last := submitted.get(nid)) is not None and now - last < stall_s:
            continue                                       # 11-11 review: mid-screen (a node / a node): not stalled yet
        srec = sbk.get(nid)
        if isinstance(srec, Mapping) and srec.get("children") and node.parent in (inc, led.port(inc)):
            continue                                       # tuned in place on the incumbent already: its ladder judges
        if isinstance(srec, Mapping) and srec.get("children") \
                and STT.ladder_state(store, srec, by, need, sseeds, hold_s, now)[2]:
            continue                                       # its own ladder is still running: wait for it
        if isinstance(srec, Mapping) and srec.get("children") and not generic_ladder(ctx, store, srec):
            continue                                       # tuned on its own knobs already: its ladder judged it
        ev = evidence(led, nid)
        if ev["n"] > cap:
            continue                                       # judged on > cap seeds: the normal path decides
        if rejudged and ev["n"] == 0:
            v, why = "rescue", "never tested (bold_knob_only re-judged structural)"
        elif fix and ev["n"] == 0 and st != "CRASHED":
            v, why = "rescue", "a debug fix, never screened (ported before it screens)"
        else:
            v, why = verdict(ctx, ev, crashed=st == "CRASHED")
        if v == "wait":
            continue
        if v == "debug" and (DEBUG_TAG in _notes(node) or not opt(ctx, "moonshot_debug")):
            v, why = "negative", why + ("; already a debug fix" if DEBUG_TAG in _notes(node) else "; debug is off")
        if v == "rescue" and node.parent in (inc, led.port(inc)) and srec is None and not rejudged:
            continue                                       # on the incumbent, not skipped: struct_tune ladders it
        score, pl = plausibility(store, post, nid)
        tuned = isinstance(srec, Mapping) and bool(srec.get("children"))
        out.append({"node": nid, "verdict": v, "why": why, "ev": ev, "score": round(score, 6), "plaus": pl,
                    "tuned": tuned, "parent": node.parent})
    out.sort(key=lambda c: (c["verdict"] == "negative", c["tuned"], -c["score"], -c["plaus"], c["node"]))
    return out


def generic_ladder(ctx: Any, store: Any, srec: Mapping[str, Any]) -> bool:
    """A finished in-place struct_tune ladder that only moved the GENERIC optimizer grid (struct_tune_knobs backbone /
    fallback grid + extra keys: LR mult, weight decay, warmup), not the node's own knobs -- e.g. a node's muon_lr_mult x
    weight_decay ladder on root. Such a node may still be rescued (ranked last); one tuned on its own knobs may not."""
    from loop.core import struct_tune as STT
    generic: set[str] = set()
    for kind in ("backbone", "other", "loss"):
        spec = STT.knob_spec(ctx, kind)
        for blk in (spec, dict(spec.get("fallback") or {})):
            generic |= {str(r.get("key")) for r in [*(blk.get("grid") or ()), *(blk.get("extra") or ())]
                        if isinstance(r, Mapping) and r.get("key")}
    keys = {k for c in srec.get("children") or () for k in (store.get_state(f"config_diff:{c}") or {})}
    return bool(keys) and keys <= generic


# ---------------------------------------------------------------------------------------------- the rescue ladder spec
def settable(ctx: Any, key: str) -> bool:
    """A declared knob the rescue ladder may move: not a protected key (``spec.surface.protected_keys``: eval / seeds /
    budget caps). Unlike ``extrapolate._allowed`` it needs no family prefix: a bold node's own new keys (nanobrain
    ``model_kwargs.*``) belong to no configured family, which is why struct_tune fell back to the LR x wd grid."""
    root = key.split("[", 1)[0]
    prot = tuple(getattr(getattr(getattr(ctx, "spec", None), "surface", None), "protected_keys", ()) or ())
    return bool(root) and not any(root == pk or root.startswith(pk + ".") for pk in prot)


def ladder_spec(ctx: Any, store: Any, nids: Sequence[str], cfg: Mapping[str, Any]) -> dict[str, Any]:
    """{points: [...]} over the 1-2 knobs the researcher declared (``components[*].knobs`` {key, lo, hi, scale}) that the
    rescued config sets: two knobs -> each at lo and hi; one -> lo, hi and the two midpoints to the current value (log
    or linear per its scale); none -> {} (struct_tune's default ladder). <= ``moonshot_rescue_rungs`` points."""
    from loop.core.struct_tune import BACKBONE_KEYS, _num
    knobs = []
    for nid in nids:
        plan = store.get_state(f"ideation:{nid}") or {}
        for c in (plan.get("components") or []) if isinstance(plan, Mapping) else []:
            for k in (c.get("knobs") or []) if isinstance(c, Mapping) else []:
                if isinstance(k, Mapping) and k.get("key") and _num(k.get("lo")) and _num(k.get("hi")) \
                        and _num(cfg.get(k["key"])) and settable(ctx, str(k["key"])) and k["key"] not in BACKBONE_KEYS \
                        and k["key"] not in [x["key"] for x in knobs]:
                    knobs.append(dict(k))
        if knobs:
            break
    knobs, budget = knobs[:2], int(opt(ctx, "moonshot_rescue_rungs"))
    pts: list[dict[str, Any]] = []
    for k in knobs:
        cur, lo, hi = float(cfg[k["key"]]), float(k["lo"]), float(k["hi"])
        vals = [lo, hi]
        if len(knobs) == 1:
            logs = str(k.get("scale") or "lin") == "log" and min(lo, hi, cur) > 0
            mid = (lambda a, b: math.sqrt(a * b)) if logs else (lambda a, b: (a + b) / 2)
            vals = [lo, hi, mid(lo, cur), mid(cur, hi)]
        for v in vals:
            if abs(v - cur) > 1e-12 * max(1.0, abs(cur)) and {k["key"]: v} not in pts:
                pts.append({k["key"]: v})
    return {"points": pts[:budget]} if pts else {}


# ---------------------------------------------------------------------------------------------- the tick step
def gpu_runs(store: Any, led: Any) -> tuple[int, set[str]]:
    """(runs in flight on rescue nodes, the rescue's node ids: originals' ports, their rungs, in-place nodes)."""
    from loop.core import struct_tune as STT
    sbk, fam = STT.book(store), set()
    for nid, rec in book(store).items():
        if isinstance(rec, Mapping) and rec.get("port"):
            fam.add(rec["port"])
            sr = sbk.get(rec["port"])
            fam |= set(sr.get("children") or ()) if isinstance(sr, Mapping) else set()
    return sum(1 for r in led.runs if r.status in LIVE_RUN and r.node in fam), fam


def share_full(ctx: Any, store: Any, led: Any, planned: int = 0) -> tuple[bool, int, int]:
    """(full, rescue runs in flight, cap): full when the in-flight rescue runs plus ``planned`` (a new booking's first
    screen: ``moonshot_rescue_seeds``) exceed ``moonshot_rescue_gpu_share`` of the capacity; ``planned`` 0: at the cap."""
    n, _ = gpu_runs(store, led)
    capacity = int(getattr(getattr(ctx, "cfg", None), "capacity", 0) or 0)
    cap = max(int(opt(ctx, "moonshot_rescue_seeds")), int(float(opt(ctx, "moonshot_rescue_gpu_share")) * capacity)) \
        if capacity > 0 else 0                            # at least one rescue's first screen fits
    return (cap > 0 and (n + planned > cap if planned else n >= cap)), n, cap


def rescue_one(ctx: Any, store: Any, adapter: Any, led: Any, nid: str, inc: str, now: float, *,
               src: str | None = None, why: str = "", score: float = 0.0, dry: bool = False) -> tuple[dict | str, str]:
    """Port ``nid`` (its change over ``src``, default its parent) onto the incumbent, or tag it in place when its parent
    IS the incumbent; returns (booking, action line)."""
    from loop import cutover as CO
    from loop.core import struct_tune as STT
    sn = store.get_node(nid)
    if sn is None:
        return "-: not in the store", f"moonshot rescue {nid}: not in the store"
    onto = led.port(inc) or inc
    src = src or sn.parent_id or ""
    tags = f"ladder={nid}{STT.RETRO_TAG}{TAG}{nid}"
    cfg_ids = [nid]
    if DEBUG_TAG in _notes(sn):
        cfg_ids.append(re.search(r"(?:^|\s)debug_of=(\S+)", _notes(sn)).group(1))
    if dry:
        return {"phase": "ported", "port": "?", "ts": now}, (
            f"moonshot rescue {nid}: would " + ("tune it in place" if src == onto else f"port it onto {onto}") + f" ({why})")
    if src == onto:                                      # on the incumbent already: tuned in place
        if TAG not in _notes(sn):
            store.update_node(nid, notes=f"{sn.notes or ''} {tags}".strip())
        if str(getattr(sn.status, "value", sn.status)) == "rejected":    # a re-judged knob rejection: screened now
            from loop.contracts import NodeStatus
            store.update_node(nid, status=NodeStatus.QUEUED, reject_reason="")
        sbk = STT.book(store)
        if isinstance(sbk.get(nid), str):                 # a skipped first shot: the rescue's ladder replaces the skip
            sbk.pop(nid, None)
            store.set_state(STT.K_STRUCT, sbk)
        pid, line = nid, f"moonshot rescue {nid}: tuned in place on the incumbent ({why})"
    else:
        got = CO.port_node(store, adapter, led, nid, now, onto=onto, src=src, tag=tags, lane_tags=False)
        if not got.node or got.how not in ("applied", "exists"):
            fn = store.get_node(got.node) if got.node else None
            if fn is not None and TAG in _notes(fn):      # a failed graft is no rescue node (it holds the TARGET's
                store.update_node(fn.id, notes=" ".join(  # commit: 11-11 the leader re-rooted onto such a copy)
                    t for t in (fn.notes or "").split() if t not in tags.split()))
            if str(got.how).startswith("conflict"):        # fix-and-continue: one researcher port session
                return _port_session(store, nid, onto, src, got, now)
            return f"-: no port onto {onto} ({got.how})", f"moonshot rescue {nid}: no port onto {onto} ({got.how})"
        pid, line = got.node, f"moonshot rescue {nid}: ported onto {onto} as {got.node} ({why})"
    cfg: dict[str, Any] = {}
    from loop.core.struct_tune import _cfg_at
    for c in (store.get_node(pid).commit, sn.commit):   # the port's config, else the original's (its own keys)
        try:
            cfg = _cfg_at(adapter, c)
            if cfg:
                break
        except Exception:  # noqa: BLE001 - unreadable: the next one, else the struct_tune default ladder
            cfg = {}
    rec = {"phase": "ported", "port": pid, "onto": onto, "src": src, "ts": now, "why": why, "score": score,
           "spec": ladder_spec(ctx, store, cfg_ids, cfg)}
    return rec, line + (f"; ladder over {[next(iter(p)) for p in rec['spec']['points']]}" if rec["spec"] else
                        "; ladder: the struct_tune default")


def _port_session(store: Any, nid: str, onto: str, src: str, got: Any, now: float) -> tuple[dict, str]:
    """A conflicting graft: a researcher port session, QUEUED here (``queued``) and handed to the tick's port sessions
    by :func:`debug_sessions` only under the rescue's researcher-slot cap (11-11 review: live tick 1 took all 4 slots)."""
    return ({"phase": "port_session", "queued": True, "onto": onto, "src": src, "ts": now, "how": got.how},
            f"moonshot rescue {nid}: graft onto {onto} conflicts ({got.how}): a researcher port session will "
            "re-implement it (rescue researcher-slot cap)")


def _adopt_port_session(store: Any, nid: str, rec: Mapping[str, Any], acts: list[str], now: float = 0.0,
                        hold_s: float = float("inf")) -> dict | str | None:
    """A port session's node (``request=req-v3-port-<nid>``) becomes the rescue's port (tagged); a failed session closes."""
    from loop.core.ideation import PORT_PREFIX
    from loop.core import struct_tune as STT
    from loop.tick import Tick
    want = f"request={PORT_PREFIX}{nid}"
    for n in store.nodes():
        if want in (n.notes or "").split():
            if TAG not in _notes(n):
                store.update_node(n.id, notes=f"{n.notes} ladder={nid}{STT.RETRO_TAG}{TAG}{nid}")
            acts.append(f"moonshot rescue {nid}: its port session made {n.id}")
            return {**{k: v for k, v in rec.items() if k != "queued"}, "phase": "ported", "port": n.id}
    st = (store.get_state(Tick.K_PORT_SESSIONS) or {}).get(nid) or {}
    if st.get("status") == "failed":
        return f"-: port session failed ({rec.get('how')})"
    if now - float(rec.get("ts") or now) > hold_s:     # 11-11 review: never a booking that waits forever
        return f"-: port session timed out after {hold_s / 3600.0:g} h ({rec.get('how')})"
    return None


def decide(ctx: Any, store: Any, led: Any, post: Any, nid: str, rec: Mapping[str, Any], inc: str, now: float
           ) -> dict | None:
    """The finished rescue's booking (phase done, ``best``), or None while its port / ladder still runs.

    11-11 review: (a) never before struct_ladders has SEEN the port (a ladder booked, or a "-:" booking / its
    not-structural memo): the port's seeds landing in one tick no longer decide the rescue on the port alone while 4
    rungs get built behind it; (b) only points at the full ``moonshot_rescue_seeds`` compete (a stale rescue: any
    point with a seed); (c) the hold clock starts when the PORT exists, not at the booking."""
    from loop.core import model as M
    from loop.core import struct_tune as STT
    pid = rec.get("port") or ""
    seeds = int(opt(ctx, "moonshot_rescue_seeds"))
    hold_s = float(opt(ctx, "moonshot_rescue_hold_h")) * 3600.0
    pn = store.get_node(pid)
    t0 = float(getattr(pn, "created_ts", 0.0) or 0.0) or float(rec.get("ts") or now)
    stale = now - t0 > hold_s
    sr = STT.book(store).get(pid)
    seen = sr is not None or pid in (store.get_state(STT.K_NONSTRUCT) or {})
    by = led.by_node()
    pdead = pn is None or str(getattr(pn.status, "value", pn.status)).lower() in ("rejected", "void", "crashed")
    if isinstance(sr, Mapping):
        _, _, running = STT.ladder_state(store, sr, by, int(opt(ctx, "struct_min_points")), seeds, hold_s, now)
        kids = list(sr.get("children") or ())
        if running and not stale:
            return None
    else:                    # no ladder (none buildable / not structural): the port alone, at its seeds
        kids = []
        if not pdead and not stale and (not seen or len(by.get(pid) or ()) < seeds):
            return None
    need = 1 if stale else seeds
    rows = []
    for c in [pid, *kids]:
        if c not in getattr(post, "est", {}) or len(by.get(c) or ()) < need:
            continue
        try:
            m, v = post.belief(c)
            dm, dv = post.shift(c, M.COMPOSITE, inc) if inc in getattr(post, "by", {}) else (0.0, 0.0)
        except Exception:  # noqa: BLE001 - not shiftable: its own frame
            continue
        rows.append((c, m + dm, math.sqrt(max(v + dv, 0.0)), len(by.get(c) or ())))
    best = max(rows, key=lambda r: r[1], default=None)
    return {**rec, "phase": "done", "done_ts": now, "best": best[0] if best else "", "k": len([pid, *kids]),
            "points": [pid, *kids], "table": [[c, round(m, 5), round(sd, 5), n] for c, m, sd, n in rows],
            "outcome": ("stale" if stale else "port dead" if pdead and not rows else "decided")}


def close_points(store: Any, nid: str, rec: Mapping[str, Any], acts: list[str]) -> list[str]:
    """11-11 review: a decided rescue's NON-best points are CLOSED for good (stored ``screened_out`` -- REJECTED when the
    point never ran, so it derives VOID and leaves the queue -- with a reason and ``rescue_closed=1``: never futility-
    exempt, never an open port, never re-queued); the best is tagged ``rescue_best_of=<K>`` so its test entry is
    max-of-K shrunk for the K points it was picked from (acquire.beta_for / struct_k)."""
    from loop.contracts import NodeStatus
    best, closed = rec.get("best") or "", []
    for c in rec.get("points") or ():
        n = store.get_node(c)
        if n is None:
            continue
        if c == best:
            if " rescue_best_of=" not in _notes(n):
                store.update_node(c, notes=f"{n.notes or ''} rescue_best_of={int(rec.get('k') or 1)}".strip())
            continue
        if " rescue_closed=1" in _notes(n):
            continue
        ran = bool(store.runs(node_id=c)) if hasattr(store, "runs") else True
        dead = str(getattr(n.status, "value", n.status)) in ("rejected", "void", "crashed")
        store.update_node(c, notes=f"{n.notes or ''} rescue_closed=1".strip(),
                          **({} if dead else {"status": NodeStatus.SCREENED_OUT if ran else NodeStatus.REJECTED,
                                               "reject_reason": f"rescue_closed: not the best point of the {nid} rescue "
                                                                f"(best {best or 'none'}; not a measured failure)"}))
        closed.append(c)
    if closed:
        acts.append(f"moonshot rescue {nid}: closed {closed} (not its best point)")
    return closed


def step(ctx: Any, store: Any, adapter: Any, led: Any, post: Any, statuses: Mapping[str, Any], now: float,
         dry: bool = False) -> list[str]:
    """One tick of the rescue (called from ``program.ladders`` BEFORE ``struct_ladders``): the retro pass once, finished
    rescues decided, port sessions adopted, then new debugs / rescues booked in priority order while slots are open."""
    if not on(ctx) or post is None or not getattr(led, "epoch", "") or not hasattr(led, "runs") \
            or not hasattr(led, "by_node"):
        return []
    acts: list[str] = [] if dry else retro(ctx, store, led, statuses, post, now, adapter)
    bk, inc = book(store), incumbent(statuses)
    changed = False
    for nid, rec in list(bk.items()):
        if not isinstance(rec, Mapping) or dry:
            continue
        if rec.get("phase") == "booking":                # a tick died mid-booking: re-planned (the port is idempotent)
            bk.pop(nid, None)
            changed = True
        elif rec.get("phase") == "port_session":
            got = _adopt_port_session(store, nid, rec, acts, now,
                                      float(opt(ctx, "moonshot_rescue_hold_h")) * 3600.0)
            if got is not None:
                bk[nid], changed = got, True
        elif rec.get("phase") == "ported":
            got = decide(ctx, store, led, post, nid, rec, inc, now)
            if got is not None:
                bk[nid], changed = got, True
                store.set_state(K_RESCUE, bk)              # booked before the nodes are touched (idempotent close)
                close_points(store, nid, got, acts)
                store.log("moonshot_rescue", node_id=nid, phase="decided", port=rec.get("port"), best=got["best"],
                          table=got["table"], outcome=got["outcome"])
                acts.append(f"moonshot rescue {nid}: {got['outcome']} -- best {got['best'] or 'none'} "
                            f"{got['table']} (released to the normal screen path, max-of-{got['k']} shrunk; the rest "
                            "closed)")
    if changed:
        store.set_state(K_RESCUE, bk)
    slots = int(opt(ctx, "moonshot_rescue_max")) - sum(1 for r in bk.values() if is_open(r) and r["phase"] != "debug")
    dslots = int(opt(ctx, "moonshot_debug_max_open")) - sum(1 for r in bk.values() if is_open(r) and r["phase"] == "debug")
    _, n_gpu, cap_gpu = share_full(ctx, store, led)
    planned = 0                                           # the first screens of this tick's bookings (GPU share)
    cands = candidates(ctx, store, led, statuses, post, inc, now)
    for c in cands:
        nid = c["node"]
        if c["verdict"] == "negative":
            if not dry:
                bk[nid] = f"-: {c['why']}"
                store.log("moonshot_rescue", node_id=nid, phase="skip", why=c["why"], evidence=c["ev"])
            continue
        full = cap_gpu > 0 and n_gpu + planned + int(opt(ctx, "moonshot_rescue_seeds")) > cap_gpu
        if c["verdict"] == "debug" and dslots <= 0 or c["verdict"] == "rescue" and (slots <= 0 or full):
            continue                                      # waits (unbooked) for a slot / the GPU share
        if c["verdict"] == "debug":
            rec: dict | str = {"phase": "debug", "status": "pending", "ts": now, "why": c["why"], "evidence": c["ev"],
                               "score": c["score"]}
            line = f"moonshot debug {nid}: queued ({c['why']}; prior {c['score']:+.4f})"
            if not dry:
                store.log("moonshot_debug", node_id=nid, phase="queued", why=c["why"], evidence=c["ev"],
                          score=c["score"])
        else:
            if not dry:                                   # booked BEFORE the port / tag is written (a crash between
                bk[nid] = {"phase": "booking", "ts": now}   # them leaves a booking, never an orphan tag)
                store.set_state(K_RESCUE, bk)
            fsn = store.get_node(nid)
            fsrc = None
            if fsn is not None and DEBUG_TAG in _notes(fsn):
                fsrc = fsn.parent_id                        # a debug fix: its whole change over the broken node's control
            rec, line = rescue_one(ctx, store, adapter, led, nid, inc, now, src=fsrc, why=c["why"], score=c["score"],
                                   dry=dry)
            if not isinstance(rec, str):
                planned += int(opt(ctx, "moonshot_rescue_seeds"))
            if not dry:
                store.log("moonshot_rescue", node_id=nid, phase="booked", why=c["why"], evidence=c["ev"],
                          score=c["score"], booking=rec if isinstance(rec, str) else
                          {k: rec.get(k) for k in ("phase", "port", "onto", "spec")})
        acts.append(line if not dry else f"(dry) {line}")
        if not dry:
            bk[nid] = rec
            store.set_state(K_RESCUE, bk)
        if is_open(rec):
            if rec["phase"] == "debug":
                dslots -= 1
            else:
                slots -= 1
    if not dry:
        store.set_state(K_RESCUE, bk)
    full = cap_gpu > 0 and n_gpu + planned + int(opt(ctx, "moonshot_rescue_seeds")) > cap_gpu
    waiting = [f"{c['node']}:{c['verdict']}" for c in cands if c["verdict"] != "negative" and c["node"] not in bk]
    if waiting:
        acts.append(f"moonshot rescue: {len(waiting)} waiting (rescues: moonshot_rescue_max "
                    f"{opt(ctx, 'moonshot_rescue_max')}" + (f", GPU share full {n_gpu}/{cap_gpu} runs" if full else "")
                    + f"; debugs: moonshot_debug_max_open {opt(ctx, 'moonshot_debug_max_open')}): {waiting[:10]}")
    return acts


# ---------------------------------------------------------------------------------------------- debug sessions
def debug_brief(store: Any, led: Any, nid: str, rec: Mapping[str, Any]) -> str:
    ev = rec.get("evidence") or {}
    per = ", ".join(f"{s}: {float(x):+.4f}" for s, x in (ev.get("seeds") or {}).items()) or "none"
    try:
        tails = {r.id: (r.log_tail or "") for r in store.runs(node_id=nid)}
    except Exception:  # noqa: BLE001 - no run rows: the ledger's output dirs only
        tails = {}
    runs = [r for r in getattr(led, "runs", ()) if r.node == nid]
    where = "\n".join(f"  run {r.id} ({r.status}, seed {r.seed}): output {r.output_dir or '?'}"
                      + (f"\n    log tail: {tails[r.id][-1500:]}" if tails.get(r.id) else "") for r in runs[:3]) \
        or "  (none)"
    return (f"DEBUG {nid} (moonshot rescue). Its first screen was CATASTROPHIC: {rec.get('why')}. Paired panel gain "
            f"vs {ev.get('vs') or 'its parent'} per seed: {per}. A drop this large usually means an IMPLEMENTATION "
            "BUG, not a bad idea. Read the parent diff below and the runs' logs in their output dirs (the training log: "
            "loss curve, NaN / inf, exploding or vanishing grad norms, a loss term that dominates or never moves, "
            "diagnostics / metrics files; a summary.json per run):\n" + where + "\nLook for: a sign / scale / shape "
            "error, a term applied at eval or to the wrong tokens, a frozen or never-updated parameter group, a "
            "broken init, a mask leaking or inverted, a missing normalisation, a schedule that never turns on / off. "
            "If you FIND a bug: fix it, keep the SAME mechanism and hypothesis (no new idea, no knob search), smoke "
            "it, and say what the bug was in the handoff. If you find NO bug: change NOTHING and finish (an empty "
            "diff records 'no bug found'). probe.py, benchmarking/ and other locked paths stay untouched.")


DEBUG_TRIES = 2                 # a debug session lost to infra (no verdict) is re-issued once; "no bug found" never is
INFRA_LOST = ("backend_unavailable", "restart", "intake_error", "smoke_infra", "smoke_starved")


def _infra_lost(store: Any, request: str) -> bool:
    """The session ended with no verdict of its own (an infra failure event for ``request``), not "no bug found"."""
    try:
        evs = [*store.events("research_error"), *store.events("research_failed")]
    except Exception:  # noqa: BLE001 - no event view: its end is its verdict
        return False
    return any(e.get("request_id") == request and (e.get("event") == "research_error" or e.get("status") in INFRA_LOST)
               for e in evs)


def llm_in_flight(store: Any, bk: Mapping[str, Any], live: set[str]) -> int:
    """Researcher sessions the rescue holds now: debugs started (or just harvested, not yet reconciled) and rescue port
    sessions handed to the tick (pending or started in ``ports.sessions``)."""
    from loop.tick import Tick
    ps = store.get_state(Tick.K_PORT_SESSIONS) or {}
    n = sum(1 for r in bk.values() if isinstance(r, Mapping) and r.get("phase") == "debug" and r.get("status") == "started")
    n += sum(1 for nid, r in bk.items() if isinstance(r, Mapping) and r.get("phase") == "port_session"
             and not r.get("queued") and (ps.get(nid) or {}).get("status") in ("pending", "started"))
    return n


def debug_sessions(tick: Any, led: Any, acts: list[str]) -> None:
    """The rescue's RESEARCHER sessions (debugs and conflicting-graft ports), at most ``moonshot_rescue_llm_slots`` in
    flight and only while more than ``moonshot_rescue_llm_reserve`` researcher slots are free (11-11 review: live tick 1
    took all 4 slots). A finished debug either left a node (``request=<id>``: the fix -- re-parented onto the node's
    control and tagged ``debug_of=``; held until the per-tick pass rescues it under the rescue cap / GPU share) or none
    ('no bug found', closed). One debug per node: re-issued once only when lost to infra (``research_error`` / an infra
    ``research_failed``). A queued rescue port session is handed to the tick's ``ports.sessions`` (started there)."""
    from loop.contracts import CandidateRequest, Operator, Strategy
    from loop.tick import Tick
    ctx, store = tick.ctx, tick.store
    if not on(ctx):
        return
    bk = book(store)
    if not any(isinstance(r, Mapping) and r.get("phase") in ("debug", "port_session") for r in bk.values()):
        return
    dry = getattr(tick, "dry", False)
    changed = False
    research = getattr(tick, "research", None)
    live = {r.id for r in [*getattr(research, "inflight", {}).values(), *getattr(research, "requeue", [])]} \
        if research is not None else set()
    cutoff = tick._tick_ts if getattr(tick, "_tick_ts", None) is not None else -float("inf")
    made = {t[len("request="):]: n for n in store.nodes() for t in (n.notes or "").split() if t.startswith("request=")}
    dbg = {n: r for n, r in bk.items() if isinstance(r, Mapping) and r.get("phase") == "debug"}
    for nid, rec in dbg.items():
        if rec.get("status") != "started" or rec.get("request") in live or float(rec.get("ts", 0)) >= cutoff or dry:
            continue
        child = made.get(rec.get("request") or "")
        if child is None and int(rec.get("tries") or 1) < DEBUG_TRIES and _infra_lost(store, rec.get("request") or ""):
            bk[nid] = {**rec, "status": "pending"}         # lost to infra (no verdict): ONE more session
            store.log("moonshot_debug", node_id=nid, phase="retry", request=rec.get("request"))
            acts.append(f"moonshot debug {nid}: its session was lost to infra; re-queued once")
        elif child is None:
            bk[nid] = {**rec, "status": "no_bug", "done_ts": tick.clock()}
            store.log("moonshot_debug", node_id=nid, phase="done", outcome="no_bug_found", request=rec.get("request"))
            acts.append(f"moonshot debug {nid}: no bug found (the session left no fix); closed")
        else:
            orig = store.get_node(nid)
            ctl = orig.parent_id if orig is not None else None
            ok = child.status.value != "rejected"
            bk[nid] = {**rec, "status": "fixed" if ok else "fix_rejected", "child": child.id, "done_ts": tick.clock()}
            store.set_state(K_RESCUE, bk)                   # booked before the child is touched
            if DEBUG_TAG not in _notes(child):
                store.update_node(child.id, parent_id=ctl or child.parent_id,
                                  notes=f"{child.notes or ''}{DEBUG_TAG}{nid}".strip())
            store.log("moonshot_debug", node_id=nid, phase="done", outcome="fixed" if ok else "fix_rejected",
                      child=child.id, request=rec.get("request"))
            acts.append(f"moonshot debug {nid}: fixed as {child.id} (re-parented onto its control {ctl}; rescued "
                        "under the rescue cap)" if ok else f"moonshot debug {nid}: its fix {child.id} was rejected")
        changed = True
    if changed:
        store.set_state(K_RESCUE, bk)
    # start new sessions: debugs pending + queued port sessions, best prior first, under the researcher-slot cap
    want = sorted([(n, r) for n, r in bk.items() if isinstance(r, Mapping) and (
        (r.get("phase") == "debug" and r.get("status") == "pending") or (r.get("phase") == "port_session"
                                                                           and r.get("queued")))],
                  key=lambda kv: -float(kv[1].get("score") or 0.0))
    room = int(opt(ctx, "moonshot_rescue_llm_slots")) - llm_in_flight(store, bk, live)
    reserve = int(opt(ctx, "moonshot_rescue_llm_reserve"))
    for nid, rec in want:
        kind = "DEBUG" if rec.get("phase") == "debug" else "port"
        if dry:
            acts.append(f"would start a researcher {kind} session for {nid} (moonshot rescue)"
                        + ("" if room > 0 else " (waits: moonshot_rescue_llm_slots)"))
            room -= 1
            continue
        free = research.free() if research is not None else 0
        if room <= 0 or free <= reserve or store.get_node(nid) is None:
            break
        if kind == "port":
            ps = dict(store.get_state(Tick.K_PORT_SESSIONS) or {})
            if nid not in ps:
                ps[nid] = {"status": "pending", "parent": rec.get("onto"), "anchor": rec.get("src"), "how": rec.get("how")}
                store.set_state(Tick.K_PORT_SESSIONS, ps)
            bk[nid] = {k: v for k, v in rec.items() if k != "queued"}
            store.set_state(K_RESCUE, bk)
            acts.append(f"moonshot rescue {nid}: port session handed to the researcher (starts next tick)")
        else:
            tries = int(rec.get("tries") or 0) + 1
            req = CandidateRequest(id=f"{debug_prefix()}{nid}-{tries}", task_id=ctx.task_id, strategy=Strategy.MOONSHOT,
                                   operator=Operator.DEBUG, island="main", parent_ids=(nid,),
                                   brief=debug_brief(store, led, nid, rec), brief_kind="repair")
            bk[nid] = {**rec, "status": "started", "request": req.id, "ts": tick.clock(), "tries": tries}
            store.set_state(K_RESCUE, bk)                   # booked BEFORE the session starts (a crash never doubles it)
            research.start(req, None, lane="repair")
            store.log("moonshot_debug", node_id=nid, phase="started", request=req.id)
            acts.append(f"researcher started: {req.id} DEBUG of {nid} (moonshot rescue: {rec.get('why')})")
        room -= 1


# ---------------------------------------------------------------------------------------------- holds
def hold(ctx: Any, store: Any, led: Any) -> set[str]:
    """Nodes that get NO more runs now (the tick adds them to ``core.held``): a rescued original (its port measures it
    now), an undecided rescue's port / rung at ``moonshot_rescue_seeds`` live-or-done seeds, a decided rescue's points
    other than its best, and -- while the GPU share is full -- every rescue node with nothing in flight."""
    from loop.core import struct_tune as STT
    if not on(ctx):
        return set()
    seeds, sbk = int(opt(ctx, "moonshot_rescue_seeds")), STT.book(store)
    used: dict[str, int] = {}
    for r in led.runs:
        if r.status not in DEAD:
            used[r.node] = used.get(r.node, 0) + 1
    full, _, _ = share_full(ctx, store, led)
    active = _active(led)
    bk = book(store)
    dbg = f"request={debug_prefix()}"
    ran = {r.node for r in led.runs}
    out: set[str] = {nid for nid, n in led.nodes.items()       # a debug fix: rescued (ported) before it screens
                     if (dbg in (n.notes or "") or DEBUG_TAG in _notes(n)) and nid not in bk and nid not in ran}
    for nid, rec in bk.items():
        if not isinstance(rec, Mapping) or rec.get("phase") not in ("ported", "done"):
            continue
        pid = rec.get("port") or ""
        if pid and pid != nid:
            out.add(nid)
        sr = sbk.get(pid)
        pts = [pid, *((sr.get("children") or ()) if isinstance(sr, Mapping) else ())]
        for c in pts:
            if not c:
                continue
            if rec.get("phase") == "done":
                if c != rec.get("best"):
                    out.add(c)
            elif used.get(c, 0) >= seeds or (full and not active.get(c)):
                out.add(c)
    return out


def seed_caps(ctx: Any, store: Any, led: Any) -> dict[str, int]:
    """node -> runs it may still get (the tick trims a screen / seed / fill action to it): an UNDECIDED rescue's port
    and rungs get at most ``moonshot_rescue_seeds`` live-or-done runs (11-11 review: a 0-run point took a normal
    ``idea_runs`` = 3 screen, 15 runs per rescue)."""
    from loop.core import struct_tune as STT
    if not on(ctx):
        return {}
    seeds, sbk = int(opt(ctx, "moonshot_rescue_seeds")), STT.book(store)
    used: dict[str, int] = {}
    for r in led.runs:
        if r.status not in DEAD:
            used[r.node] = used.get(r.node, 0) + 1
    out: dict[str, int] = {}
    for nid, rec in book(store).items():
        if not isinstance(rec, Mapping) or rec.get("phase") != "ported" or not rec.get("port"):
            continue
        sr = sbk.get(rec["port"])
        for c in [rec["port"], *((sr.get("children") or ()) if isinstance(sr, Mapping) else ())]:
            out[c] = max(0, seeds - used.get(c, 0))
    return out


# ---------------------------------------------------------------------------------------------- retro (once per deploy)
def parent_src(adapter: Any, commit: str) -> Any:
    """``path -> the parent commit's source`` for the AST comparison (as ``Ideation._parent_src``); None without one."""
    show, repo = getattr(adapter, "_show", None), getattr(adapter, "repo", None)
    if not (commit and show and repo):
        return None
    return lambda rel: show(repo, commit, rel)


def rejudge_knob(ctx: Any, store: Any, now: float, adapter: Any = None) -> list[str]:
    """The knob judge's past ``bold_knob_only`` rejections whose diff adds code (``bold.code_override``) are restored:
    tagged ``knob_rejudged=1`` (still REJECTED: never screened on its old parent) for :func:`step` to port onto the
    incumbent (a node already on the incumbent is QUEUED in place) under the rescue cap."""
    from pathlib import Path
    from loop.contracts import NodeStatus
    from loop.core import bold as BOLD
    acts = []
    for n in store.nodes(status=[NodeStatus.REJECTED]):
        if not str(n.reject_reason or "").startswith("bold_knob_only: the knob judge") or " knob_rejudged=1" in _notes(n) \
                or " brief=bold" not in _notes(n):
            continue
        try:
            diff = Path(n.diff_path).read_text() if n.diff_path else ""
        except OSError:
            diff = ""
        par = store.get_node(n.parent_id or "")
        why = BOLD.code_override(ctx, diff, parent_src(adapter, getattr(par, "commit", "")))   # vs the parent's source
        if not why:
            continue
        store.update_node(n.id, reject_reason=f"bold_knob_only re-judged structural ({why[:120]}): a moonshot rescue "
                          "ports it onto the incumbent (loop.core.rescue)", notes=f"{n.notes or ''} knob_rejudged=1".strip())
        store.log("moonshot_rescue", node_id=n.id, phase="knob_rejudged", why=why, was=n.reject_reason[:300])
        acts.append(f"moonshot rescue: {n.id} {n.mechanism_tag} re-judged structural ({why}); restored")
    return acts


def code_version() -> str:
    """The deploy's code version for the retro pass: a hash of the rescue + knob-rule sources (a redeploy that changes
    them re-runs the retro; a plain restart does not)."""
    import hashlib
    from pathlib import Path
    h = hashlib.sha1()
    for f in ("rescue.py", "bold.py", "struct_tune.py"):
        try:
            h.update((Path(__file__).parent / f).read_bytes())
        except OSError:
            pass
    return h.hexdigest()[:12]


def retro(ctx: Any, store: Any, led: Any, statuses: Mapping[str, Any], post: Any, now: float,
          adapter: Any = None) -> list[str]:
    """Once per DEPLOY (``K_RETRO`` keyed on :func:`code_version`): restore the knob-judge rejections that change code,
    and list the backlog of closed / stalled moonshots in the order the per-tick pass enqueues them (event
    ``moonshot_rescue_retro``)."""
    ver = code_version()
    if (store.get_state(K_RETRO) or {}).get("version") == ver:
        return []
    acts = rejudge_knob(ctx, store, now, adapter)
    cands = candidates(ctx, store, led, statuses, post, incumbent(statuses))
    backlog = [[c["node"], c["verdict"], c["score"], c["ev"]["n"],
                None if c["ev"]["mean"] is None else round(c["ev"]["mean"], 5)] for c in cands]
    store.set_state(K_RETRO, {"ts": now, "version": ver, "rejudged": [a.split()[2] for a in acts], "backlog": backlog})
    store.log("moonshot_rescue_retro", backlog=backlog, rejudged=len(acts))
    if any(c["verdict"] != "negative" for c in cands):
        acts.append(f"moonshot rescue retro: {sum(c['verdict'] != 'negative' for c in cands)} closed / stalled moonshot(s) "
                    f"to debug or rescue (<= {opt(ctx, 'moonshot_rescue_max')} at once): "
                    + ", ".join(f"{c['node']}:{c['verdict']}" for c in cands if c["verdict"] != "negative")[:400])
    return acts


# ---------------------------------------------------------------------------------------------- status
def lines(store: Any, ctx: Any = None) -> list[str]:
    """``loop status`` lines: open debugs / rescues, finished ones, and the skip count."""
    bk = book(store)
    if not bk and not store.get_state(K_RETRO):
        return []
    opn = {n: r for n, r in bk.items() if is_open(r)}
    done = {n: r for n, r in bk.items() if isinstance(r, Mapping) and not is_open(r)}
    skip = sum(1 for r in bk.values() if isinstance(r, str))
    cap = opt(ctx, "moonshot_rescue_max") if ctx is not None else "?"
    dcap = opt(ctx, "moonshot_debug_max_open") if ctx is not None else "?"
    nd = sum(1 for r in opn.values() if r.get("phase") == "debug")
    out = [f"moonshot rescue: {len(opn) - nd}/{cap} rescues + {nd}/{dcap} debugs open, {len(done)} finished, {skip} "
           "skipped (clearly negative / no port)"]
    for n, r in sorted(opn.items()):
        if r.get("phase") == "debug":
            out.append(f"  {n} DEBUG {r.get('status')} ({r.get('why', '')[:90]})")
        else:
            out.append(f"  {n} RESCUE {r.get('phase')} -> {r.get('port') or '?'} onto {r.get('onto') or '?'}"
                       + (f", ladder over {[next(iter(p)) for p in (r.get('spec') or {}).get('points') or []]}"
                          if (r.get("spec") or {}).get("points") else ""))
    for n, r in sorted(done.items()):
        if r.get("phase") == "debug":
            out.append(f"  {n} debug {r.get('status')}" + (f" -> {r['child']}" if r.get("child") else ""))
        else:
            out.append(f"  {n} rescue {r.get('outcome', r.get('phase'))}: best {r.get('best') or 'none'} "
                       f"{r.get('table') or ''}")
    return out
