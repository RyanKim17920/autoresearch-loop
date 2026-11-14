"""loop.legs — the LEG FACTORIAL on the stack carrier (operator / stacking audit 2026-10-04).

The old stack builder (loop.select.stack_plan: the top-P legs of ONE carrier crossed by a researcher, screened on 2
seeds) built 9 stacks with 1 real gain: it reused the same legs in every stack, only combined legs of one carrier,
missed crossover legs that contain other legs, stacked same-subsystem samplers that cancel, rebuilt a node from a node
instead of adding to it, and let 2-seed screens kill the stacks. It is retired while this is on (``legs.enabled``).

ONE batch per carrier (the incumbent when the batch starts; it keeps that carrier):

1. LEGS (:func:`candidates`): closed nodes with pooled P(own gain > 0) >= ``legs.min_p`` over >= ``legs.min_seeds``
   scored seeds, built on ANY established carrier (a CONFIRMED node or root) or on the carrier itself; crossovers and
   stacks are never legs themselves (their constituents are judged on their own); a leg one of whose constituents is
   already in the carrier, or (built elsewhere) setting a config key the carrier already sets or sharing a mechanism
   word with a carrier leg, is dropped; the best leg per SUBSYSTEM (:func:`subsystem`), at most ``legs.max_new`` and ``legs.max_factors`` (8) factors in all.
2. BRANCH (a researcher side request, brief ``legs``): ONE commit on the carrier where every new leg is grafted behind
   a 0/1 config flag (key containing ``leg_<id>``, default 0) and every carrier leg whose strength is hard-coded is
   exposed as a strength multiplier (``leg_<id>``, default 1 = current, 0 = off); identical to the carrier at the
   defaults (declared ``identical_at_default``, else the carrier's own runs are not used as a design row).
3. TWO DESIGNS on the branch, config-only points, ONE run each on ONE seed, blocked 4 points per seed (the per-seed
   intercept of loop.design.analyze absorbs the block): the LEG FACTORIAL (every new leg off/on, every carrier leg
   off/current -- the carrier's drop-one-leg ablation in context -- and every shared knob the carrier retuned at its
   parent's / its own value): a two-level resolution-IV design (odd words of 4-6 base factors: 16 runs for <= 8
   factors, 32 for 9, 64 on 16 seeds for >= 10), SE ~ sigma / sqrt(runs / 4); and the DOSE
   design (every carrier leg's strength x0.5 / x2, the shared knobs as above; 16 runs, its 4 seeds shared with the
   factorial's controls). Points run on preemptible slots under ``design.max_share``; the carrier's own run at each
   seed is the paired control and, when the branch is identical, one more design row (its corner / centre).
4. Each design's fitted main-effects optimum (every factor at its better level) becomes ONE node when its predicted
   gain over the carrier is > 0: parent the carrier, a fresh screen block, NO screen gate (``legs_of=``: the design
   was the selection) -> 2 screen + 3 confirm fresh seeds paired against the carrier, then the held-out audit."""

from __future__ import annotations

import dataclasses
import itertools
import math
import re
import time
from typing import Any, Mapping, Sequence

from loop import design as D
from loop.agents.researcher import set_yaml_value
from loop.contracts import (CARRIER, CLOSED_STATUSES, DESIGN_FAMILY, PANEL, ROOT_NODE_ID, AxisPrediction,
                            CandidateRequest, HypothesisKind, HypothesisRecord, Node, NodeStatus, Operator, Stage,
                            Strategy, design_point)
from loop.spec import LoopContext

K_LEGS = "legs"     # {"seq": n, "current": batch | None, "batches": [closed batches]}
K_SWAP = "legs.swaps"   # queued child of the incumbent -> its swap node ('' = considered, none built)
MAX_FACTORS = 16    # hard cap (legs.max_factors pre-screens to 8); >= 10 factors run 64 runs on 16 seeds
BLOCK = 4           # points per seed
#: (subsystem, mechanism-tag pattern), first match wins; else the node's mtype=, else its family
SUBSYSTEMS: tuple[tuple[str, str], ...] = (
    ("teacher", r"teacher|\bema\b|momentum|centering"),
    ("ibot_proto", r"ibot|proto|koleo|memax|equipart"),
    ("predictor", r"predictor|pred_width|bottleneck"),
    ("sampling", r"sampl|coverage|balanc|curricul|tissue_aware"),
    ("augmentation", r"aug|rot90|\bd4\b|crop|jitter|elastic|blur|stain|flip"),
    ("regularizer", r"sigreg|isotrop|regulari|weight_decay|drop_path|var_hinge|variance|uniformity|dispersion"),
    ("schedule", r"ramp|schedul|warmup|latephase|\blr\b|lion|adam|lawa|freeze"),
    ("masked_modeling", r"contour|\bmim\b|masked_"),
    ("dense_objective", r"densecl|xview|dense"),
    ("fino", r"fino|subtype|patient"),
    ("cls", r"\bcls\b|cls_"),
    ("jepa", r"jepa|xscale|salient|attmask"),
)
_MTYPE = {"data_sampling": "sampling", "augmentation": "augmentation", "regularization": "regularizer",
          "training_dynamics": "schedule", "architecture_addon": "architecture", "loss_objective": "objective"}
#: words too generic to say two legs share a mechanism
_GENERIC = frozenset({"jepa", "ijepa", "tune", "tuned", "loss", "patch", "head", "sampl", "stack", "objective",
                      "local", "global", "weight", "late", "phase"})


def _note(node: Node | None, key: str) -> str:
    m = re.search(rf"(?:^|\s){key}=(\S+)", (node.notes or "") if node is not None else "")
    return m.group(1) if m else ""


def book(store: Any) -> dict[str, Any]:
    return {"seq": 0, "current": None, "batches": [], **(store.get_state(K_LEGS) or {})}


# ------------------------------------------------------------------------------------------------ legs
def subsystem(node: Node) -> str:
    tag = (node.mechanism_tag or "").lower()
    for name, pat in SUBSYSTEMS:
        if re.search(pat, tag) or re.search(pat, tag.replace("_", " ")):
            return name
    return _MTYPE.get(_note(node, "mtype"), node.family or "other")


def constituents(store: Any, nid: str, seen: set[str] | None = None) -> set[str]:
    """The single-change legs a node is made of: a crossover's pair / a stack's legs, expanded; else itself."""
    seen = set() if seen is None else seen
    if nid in seen:
        return set()
    seen.add(nid)
    n = store.get_node(nid)
    parts = [*(_note(n, "stack_legs").split(",") if _note(n, "stack_legs") else ()),
             *(_note(n, "legs_on").split(",") if _note(n, "legs_on") else ()),      # a graduated combination
             *(_note(n, "xover").split(",") if _note(n, "xover") else ())]
    if not parts:
        return {nid}
    return set().union(*(constituents(store, p, seen) for p in parts if p))


def ancestors(store: Any, nid: str) -> list[str]:
    out, n = [], store.get_node(nid)
    while n is not None and n.parent_id and n.parent_id not in out:
        out.append(n.parent_id)
        n = store.get_node(n.parent_id)
    return out


_SYNONYMS = ((r"cross_?scale", "xscale"), (r"slide_?bal", "slide_balanced"))


def _words(store: Any, nids: Sequence[str]) -> set[str]:
    from loop.novelty import tag_words

    def canon(tag: str) -> str:
        for pat, to in _SYNONYMS:
            tag = re.sub(pat, to, tag)
        return tag
    return {w for i in nids if (n := store.get_node(i)) is not None
            for w in tag_words(canon((n.mechanism_tag or "").lower()))} - _GENERIC


def own_legs(store: Any, carrier: str) -> list[str]:
    """The carrier's own legs: its constituents (a stack / crossover), else the carrier itself."""
    return sorted(constituents(store, carrier) - {carrier}) or [carrier]


#: a leg's statuses: closed, or in its fresh-seed confirmation (early-confirmed / banked: well measured, not CONFIRMED)
LEG_STATUSES = (NodeStatus.SCREENED_OUT, NodeStatus.NOT_CONFIRMED, NodeStatus.CONFIRMED, NodeStatus.CONFIRMING,
                NodeStatus.AUDIT_FAILED)


def promising(ctx: LoopContext, store: Any, snapshot: Any, n: Node, carrier: str = "") -> str:
    """Why ``n`` is a WELL-MEASURED PROMISING change ('' = it is not; CONFIRMED or not -- a +0.002 real effect is what
    stacking is for, and the factorial's shrinkage + null gate guard the winner's curse): pooled P(own gain > 0) >=
    ``legs.min_p`` over >= ``legs.min_seeds`` scored seeds; banked (loop.world.banked); >= ``legs.fresh_seeds`` own
    fresh confirm seeds with a positive mean; or (``legs.port_legs``) a backfill port onto ``carrier`` whose own
    increment there is positive."""
    from loop.select import STACK_MIN_SEEDS, pooled_gain
    from loop.world import scored_seeds
    cfg, post = ctx.cfg.legs, snapshot.nodes.get(n.id)
    if post is None or post.vetoed or design_point(n):
        return ""
    p = pooled_gain(snapshot, n.id, float(snapshot.sigma.get(PANEL, ctx.cfg.panel_sd)))
    oc = (getattr(snapshot, "own_confirm", None) or {}).get(n.id)
    if p >= cfg.min_p and (seeds := scored_seeds(store, n.id)) >= max(cfg.min_seeds, STACK_MIN_SEEDS):
        return f"P {p:.2f} over {seeds} seeds"
    if n.id in set(getattr(snapshot, "banked", ()) or ()):
        return "banked"
    if cfg.fresh_seeds > 0 and oc is not None and oc[1] >= cfg.fresh_seeds and oc[0] > 0:
        return f"{oc[1]} fresh confirm seeds {oc[0]:+.4f}"
    if cfg.port_legs and carrier and n.parent_id == carrier and _note(n, "backfill_of") \
            and post.increment.mean > 0 and scored_seeds(store, n.id) >= 2:
        return f"port of {_note(n, 'backfill_of')} on {carrier} {post.increment.mean:+.4f}"
    return ""


def pool(ctx: LoopContext, store: Any, snapshot: Any) -> list[str]:
    """The well-measured promising NON-confirmed nodes (:func:`promising`): the winner-cross pool's addition to the
    CONFIRMED winners (crossed with the stack head only, like a banked node)."""
    if snapshot is None:
        return []
    return sorted(n.id for n in store.nodes(status=[s for s in LEG_STATUSES if s is not NodeStatus.CONFIRMED])
                  if n.id != ROOT_NODE_ID and promising(ctx, store, snapshot, n))


def candidates(ctx: LoopContext, store: Any, snapshot: Any, carrier: str) -> tuple[list[dict], list[dict]]:
    """(legs picked, legs dropped with why), each {id, p, seeds, subsystem, parent, mechanism, why_leg}."""
    from loop.select import built_from, change_set, pooled_gain
    from loop.world import held_back, scored_seeds
    cfg, sigma = ctx.cfg.legs, float(snapshot.sigma.get(PANEL, ctx.cfg.panel_sd))
    held = held_back(store)
    ok = {ROOT_NODE_ID} | {n.id for n in store.nodes(status=[NodeStatus.CONFIRMED]) if n.id not in held} \
        | set(getattr(snapshot, "banked", ()) or ())
    mine = own_legs(store, carrier)
    own = set(mine) | built_from(store, carrier) | {carrier}        # every leg / source it was built from
    own_keys = {k for i in mine for k in (store.get_state(f"config_diff:{i}") or {})} \
        | {s["key"] for s in shared_knobs(ctx, store, carrier)} | set(change_set(store, store.get_node(carrier))[0])
    own_words = _words(store, mine)
    tried = {r["id"]: f"L{b['batch']}" for b in book(store)["batches"]           # on it, or on a carrier with the
             if b["carrier"] == carrier or own_legs(store, b["carrier"]) == mine     # same legs (a node = a node's 4)
             for r in b.get("legs") or ()}
    picked: dict[str, dict] = {}
    dropped: list[dict] = []
    for n in store.nodes(status=list(LEG_STATUSES)):
        post = snapshot.nodes.get(n.id)
        if n.id in own or design_point(n) or post is None or (n.parent_id or ROOT_NODE_ID) not in ok | {carrier} \
                or n.strategy is Strategy.DESIGN or _note(n, "legs_of"):
            continue
        if not (good := promising(ctx, store, snapshot, n, carrier)):
            continue
        p = pooled_gain(snapshot, n.id, sigma)
        row = {"id": n.id, "p": round(p, 3), "seeds": scored_seeds(store, n.id), "subsystem": subsystem(n),
               "parent": n.parent_id or ROOT_NODE_ID, "mechanism": n.mechanism_tag, "why_leg": good}
        parts = constituents(store, n.id)
        on_carrier = carrier in ancestors(store, n.id)
        if parts != {n.id}:
            why = f"a crossover / stack of {sorted(parts)}: its constituents are judged on their own"
        elif parts & own:
            why = f"already in the carrier ({sorted(parts & own)})"
        elif n.id in tried or _note(n, "backfill_of") in tried:
            why = f"already a leg of round {tried.get(n.id) or tried[_note(n, 'backfill_of')]} on this carrier"
        elif not on_carrier and (k := set(store.get_state(f"config_diff:{n.id}") or {}) & own_keys):
            why = f"sets key(s) the carrier already sets {sorted(k)}"
        elif not on_carrier and (w := _words(store, [n.id]) & own_words):
            why = f"overlaps a carrier leg's mechanism ({sorted(w)})"
        else:
            why = ""
        if why:
            dropped.append({**row, "why": why})
            continue
        best = picked.get(row["subsystem"])
        if best is None or (p, n.id) > (best["p"], best["id"]):
            if best is not None:
                dropped.append({**best, "why": f"subsystem {best['subsystem']}: {n.id} has the higher P"})
            picked[row["subsystem"]] = row
        else:
            dropped.append({**row, "why": f"subsystem {row['subsystem']}: {best['id']} has the higher P"})
    legs = sorted(picked.values(), key=lambda r: (-r["p"], r["id"]))
    cap = new_cap(ctx, len(mine) + len(shared_knobs(ctx, store, carrier)))
    for r in legs[max(0, cap):]:                         # pre-screened by their own measured P (legs is P-sorted)
        dropped.append({**r, "why": f"over the {cap}-leg cap (legs.max_factors {cfg.max_factors} in all, "
                                     f">= legs.min_new {cfg.min_new} new)"})
    return legs[:max(0, cap)], sorted(dropped, key=lambda r: (-r["p"], r["id"]))


def new_cap(ctx: LoopContext, fixed: int) -> int:
    """New legs a round takes next to ``fixed`` carrier legs + shared knobs: ``legs.max_factors`` in all, but never
    under ``legs.min_new`` (a round needs that many; live 10-05..10-07: 8 - 4 carrier legs - 1 knob = 3 < 4 deadlocked
    every further round on a node) and never past MAX_FACTORS."""
    cfg = ctx.cfg.legs
    return max(0, min(cfg.max_new, MAX_FACTORS - fixed, max(min(cfg.max_factors, MAX_FACTORS) - fixed, cfg.min_new)))


def _cfg(store: Any, nid: str) -> dict[str, Any]:
    from loop.select.intake import K_BASE, parent_config
    return {**(store.get_state(K_BASE) or {}), **parent_config(store, nid)}


def shared_knobs(ctx: LoopContext, store: Any, carrier: str) -> list[dict[str, Any]]:
    """Knobs the carrier RETUNED (a numeric root key whose value differs from its parent's): low = the parent's."""
    from loop.select.intake import K_BASE
    base = store.get_state(K_BASE) or {}
    c, p = _cfg(store, carrier), _cfg(store, (store.get_node(carrier).parent_id or ROOT_NODE_ID))
    return [{"key": k, "low": p[k], "high": c[k], "kind": "shared", "leg": "", "levers": {}} for k in sorted(c)
            if k in base and k in p and D._num(c[k]) and D._num(p[k]) and c[k] != p[k]]


def dose_key(store: Any, leg: str, carrier: str) -> str:
    """A carrier leg's own strength key: its declared ``dose=``, else the one numeric non-zero key its diff added
    (not a root knob: that is a shared retune)."""
    from loop.select.intake import K_BASE
    n, base, c = store.get_node(leg), store.get_state(K_BASE) or {}, _cfg(store, carrier)
    if (d := _note(n, "dose")) and d in c and D._num(c[d]):
        return d
    keys = [k for k in (store.get_state(f"config_diff:{leg}") or {}) if k not in base and D._num(c.get(k)) and c.get(k)]
    return keys[0] if len(keys) == 1 else ""


# ------------------------------------------------------------------------------------------------ designs
def odd_words(k: int) -> tuple[int, list[tuple[int, ...]], list[tuple[int, ...]]]:
    """(base factors b, factor words, rows): the 2^b full factorial's odd-length words as factor columns -- the
    16-run Plackett-Burman (2^(15-11)) folded over is the 32-run case; resolution IV (two factors' product is an even
    word, never a factor). b = 4 for <= 8 factors, 5 for 9, 6 (64 runs on 16 seeds) for >= 10: a saturated 32-run
    design leaves too few df and an se too large for 10-16 factors."""
    if not 1 <= k <= MAX_FACTORS:
        raise ValueError(f"{k} factors: 1..{MAX_FACTORS}")
    b = 4 if k <= 8 else 5 if k <= 9 else 6
    words = [w for r in range(1, b + 1, 2) for w in itertools.combinations(range(b), r)][:k]
    rows = []
    for r in range(2 ** b):
        x = [1 if r >> (b - 1 - j) & 1 else -1 for j in range(b)]
        rows.append(tuple(math.prod(x[i] for i in w) for w in words))
    return b, words, rows


def block_of(b: int, r: int) -> int:
    """Row r's block (4 rows each): the signs of the even words AB, AC(, AD) -- every block contrast is an even
    word, so no block is ever confounded with a main effect."""
    x = [1 if r >> (b - 1 - j) & 1 else -1 for j in range(b)]
    return sum((x[0] * x[g] > 0) << i for i, g in enumerate(range(1, b - 1)))


def _fresh_seeds(ctx: LoopContext, store: Any, n: int, purpose: str = "legs") -> list[int]:
    """``n`` never-used seeds from the rotation pool (loop.remint.draw). Not max-seed + 1: that range is where the
    screen blocks grow, so a block assigned but not yet launched (or the graduate's own fresh block) could land on
    the factorial's seeds."""
    from loop.remint import draw
    return draw(ctx, store, n, purpose)


def factors(ctx: LoopContext, store: Any, cur: Mapping[str, Any], branch_cfg: Mapping[str, Any]) -> dict[str, list]:
    """{"legs": factorial factors, "dose": dose-design factors, "missing": legs the branch has no key for}. Levels
    come from the contract, not the branch's committed values: a new leg's flag 0 / 1, a carrier leg's ``leg_<id>``
    multiplier 1 (a dose key: the carrier's own value). Live L3 (branch a node): the researcher's time limit fired
    before it reset its smoke values (3 new flags at 1, a node's multiplier at 2.0), so every new leg read as 'no key'
    and a node's 'current' level was 2.0."""
    keyed = {i: next((k for k in branch_cfg if re.search(rf"(?:^|[._]){re.escape('leg_' + i)}$", k)), "")
             for i in [*(r["id"] for r in cur["legs"]), *cur["own"]]}
    car = _cfg(store, cur["carrier"])
    fac, dose, missing = [], [], []
    for r in cur["legs"]:
        if (k := keyed[r["id"]]) and isinstance(branch_cfg.get(k), (int, float)):
            fac.append({"key": k, "low": type(branch_cfg[k])(0), "high": type(branch_cfg[k])(1), "kind": "new",
                        "leg": r["id"], "levers": {}})
        else:
            missing.append(r["id"])
    for leg in cur["own"]:
        k = keyed[leg] or dose_key(store, leg, cur["carrier"])
        v = branch_cfg.get(k)
        v = type(v)(1) if keyed[leg] and D._num(v) else car.get(k, v)
        if not k or not D._num(v) or not v:
            missing.append(leg)
            continue
        zero = type(v)(0)
        fac.append({"key": k, "low": zero, "high": v, "kind": "own", "leg": leg, "levers": {}})
        if (lv := dose_levels(k, v, bool(keyed[leg]))) is not None:
            dose.append({"key": k, "low": lv[0], "high": lv[1], "kind": "own", "leg": leg, "levers": {}})
    for s in cur["shared"]:
        fac.append(dict(s))
        dose.append(dict(s))
    return {"legs": fac, "dose": dose, "missing": missing}


def dose_levels(key: str, v: Any, multiplier: bool) -> tuple[Any, Any] | None:
    """A carrier leg's x0.5 / x2 dose levels around its current ``v`` (the design's centre, coded 0). A ``leg_<id>``
    strength multiplier is continuous by contract (0.5 = half strength) even when the YAML wrote it as the int 1: float
    0.5 v / 2 v. Any other int knob halves to ``v // 2``; when that equals ``v`` (v = 1) the low level IS the carrier
    and coding it -1 fakes an effect and miscodes the carrier row (live L1/L2: a node / a node low = high/2 = 1) -> None,
    the factor is left out of the dose design."""
    if multiplier or isinstance(v, float):
        return float(v) * 0.5, float(v) * 2.0
    low = max(1, v // 2)
    return None if low == v else (low, v * 2)


def corner(fac: Sequence[Mapping[str, Any]], dose: bool) -> list[int]:
    """The carrier's own row: a new leg off (-1), a carrier leg on (+1; the dose design's centre 0), a shared knob at
    the carrier's value (+1)."""
    return [-1 if f["kind"] == "new" else 0 if dose and f["kind"] == "own" else 1 for f in fac]


# ------------------------------------------------------------------------------------------------ the batch
def carriers(ctx: LoopContext, store: Any, snapshot: Any) -> list[str]:
    """The incumbent, then (``legs.challenger_p`` > 0) every STRONG CHALLENGER -- CONFIRMED, not held back, not vetoed,
    P(U > U_inc) >= legs.challenger_p (live a node 0.94) -- best first."""
    from loop.world import held_back
    inc, cp = getattr(snapshot, "incumbent_id", ROOT_NODE_ID), ctx.cfg.legs.challenger_p
    held = held_back(store) if cp > 0 else set()
    ch = sorted((-snapshot.nodes[n.id].p_beats_incumbent, n.id) for n in store.nodes(status=[NodeStatus.CONFIRMED])
                if cp > 0 and n.id not in (inc, ROOT_NODE_ID) and n.id not in held and n.id in snapshot.nodes
                and not snapshot.nodes[n.id].vetoed and snapshot.nodes[n.id].p_beats_incumbent >= cp)
    return ([inc] if inc != ROOT_NODE_ID else []) + [i for _, i in ch]


def plan(ctx: LoopContext, store: Any, snapshot: Any, now: float | None = None) -> dict[str, Any] | None:
    """What a new batch would start with now: on the first of :func:`carriers` that is DUE (None: off, a batch
    running, or none due). A carrier's first round is due at once (a challenger's once 2 legs qualify); a further
    round once ``legs.min_new`` new legs (capped by :func:`new_cap`) qualify, or -- the CADENCE -- once 2 do and
    ``legs.round_hours`` passed since its last round (live 10-05..10-07: no round for 2 days)."""
    if not ctx.cfg.legs.enabled or snapshot is None or book(store)["current"] is not None:
        return None
    now = time.time() if now is None else now
    for i, cid in enumerate(carriers(ctx, store, snapshot)):
        if (p := _plan_on(ctx, store, snapshot, cid, now, challenger=i > 0)) is not None:
            return p
    return None


def _plan_on(ctx: LoopContext, store: Any, snapshot: Any, cid: str, now: float, challenger: bool) -> dict | None:
    cfg, past = ctx.cfg.legs, [b for b in book(store)["batches"] if b["carrier"] == cid]
    rounds = len(past)
    legs, dropped = candidates(ctx, store, snapshot, cid)
    mine, shared = own_legs(store, cid), shared_knobs(ctx, store, cid)
    need = max(2, min(cfg.min_new, new_cap(ctx, len(mine) + len(shared))))
    last = max((float(b.get("closed") or b.get("started") or 0.0) for b in past), default=0.0)
    if not rounds and not challenger:
        why = "first round on the incumbent"
    elif not rounds and len(legs) >= 2:
        why = f"strong challenger (P(U>U_inc) {snapshot.nodes[cid].p_beats_incumbent:.2f})"
    elif rounds and len(legs) >= need:
        why = f"{len(legs)} new legs >= {need}"
    elif rounds and len(legs) >= 2 and cfg.round_hours > 0 and now - last >= 3600.0 * cfg.round_hours:
        why = f"cadence: {(now - last) / 3600.0:.0f} h since its last round"
    else:
        return None
    k = len(legs) + len(mine) + len(shared)
    runs = 2 ** odd_words(k)[0] if k else 0
    return {"carrier": cid, "round": rounds + 1, "why": why, "legs": legs, "dropped": dropped, "own": mine,
            "shared": shared,
            "expose": [i for i in mine if not dose_key(store, i, cid)],
            "runs": runs, "seeds": runs // BLOCK, "dose_runs": 16 if ctx.cfg.legs.dose else 0}


def tick(ctx: LoopContext, store: Any, adapter: Any, snapshot: Any, *, new_id: Any, now: float | None = None,
         scarce: bool = False, inflight: Sequence[str] = ()) -> list[str]:
    """Start a batch when one is due (:func:`plan`; also in scarce mode: one batch is all stacking holds), wait for
    its branch, design it, and analyse it once every point closed. ``inflight``: request ids the driver's researchers
    still work on. Swap tests pause in scarce mode."""
    now = time.time() if now is None else now
    if not ctx.cfg.legs.enabled or snapshot is None:
        return []
    acts = swaps(ctx, store, adapter, snapshot, new_id=new_id, now=now) if not scarce else []
    bk = book(store)
    cur = bk["current"]
    try:
        if cur is None:
            if (p := plan(ctx, store, snapshot, now)) is None:
                return acts
            if not p["legs"] and not p["expose"] and len(p["own"]) + len(p["shared"]) < 2:
                return acts
            b = bk["seq"] + 1
            cur = {"batch": b, "carrier": p["carrier"], "round": p["round"], "legs": p["legs"], "dropped": p["dropped"], "own": p["own"],
                   "shared": p["shared"], "phase": "build", "want": True, "pending": "", "fails": 0, "branch": "",
                   "started": now, "designs": {}}
            bk.update(seq=b, current=cur)
            store.set_state(K_LEGS, bk)
            store.log("legs_started", batch=b, carrier=p["carrier"], legs=[r["id"] for r in p["legs"]],
                      own=p["own"], shared=[s["key"] for s in p["shared"]], why=p["why"])
            return acts + [f"leg factorial L{b} on carrier {p['carrier']} (round {p['round']}; {p['why']}): "
                           f"{len(p['legs'])} new leg(s) "
                    f"{[r['id'] + '/' + r['subsystem'] for r in p['legs']]}, carrier legs {p['own']}, shared "
                    f"{[s['key'] for s in p['shared']]}; a researcher builds the flagged branch first"]
        if cur["phase"] == "build":
            return acts + _build(ctx, store, adapter, bk, cur, new_id, now, set(inflight))
        if cur["phase"] == "run":
            return acts + _finish(ctx, store, adapter, bk, cur, new_id, now, snapshot)
    except Exception as exc:  # noqa: BLE001 -- a git / adapter failure closes the batch, never the loop
        if cur is None:
            store.log("legs_error", error=repr(exc)[:500])
            return [f"leg factorial plan error: {exc!r}"[:300]]
        _close(store, bk, cur, now, error=repr(exc)[:300])
        store.log("legs_error", batch=cur["batch"], error=repr(exc)[:500])
        return acts + [f"leg factorial L{cur['batch']} error: {exc!r}"[:300]]
    return acts


def _close(store: Any, bk: dict, cur: dict, now: float, **result: Any) -> None:
    bk.update(current=None, batches=bk["batches"] + [{**cur, **result, "want": False, "closed": now}])
    store.set_state(K_LEGS, bk)


def _build(ctx: LoopContext, store: Any, adapter: Any, bk: dict, cur: dict, new_id: Any, now: float,
           inflight: set[str]) -> list[str]:
    """The branch arrived: design both batches on it. A rejected branch, or a request that ended without one
    (adopt clears ``pending``; an outage re-wants it), is asked again up to ``moonshot_tune.max_research_fails``."""
    tag = f"L{cur['batch']}"
    lost = cur["pending"] and cur["pending"] not in inflight
    if cur.get("got") == "rejected" or lost:
        cur.update(fails=cur["fails"] + 1, got="", pending="", why=cur.get("why") if not lost else "no candidate")
        if cur["fails"] >= ctx.cfg.moonshot_tune.max_research_fails:
            _close(store, bk, cur, now, why=f"branch build failed {cur['fails']}x ({cur.get('why', '')})")
            return [f"leg factorial {tag}: branch build failed {cur['fails']}x; closed"]
        cur["want"] = True
        store.set_state(K_LEGS, bk)
        return [f"leg factorial {tag}: branch build failed ({cur.get('why', '') or 'no candidate'}); asking again"]
    if not cur["branch"]:
        return []
    br = store.get_node(cur["branch"])
    with D._Tree(ctx, store, adapter, f"legs{cur['batch']}", base_id=br.id) as t:
        f = factors(ctx, store, cur, t.base)
        if len(f["legs"]) < 2:
            _close(store, bk, cur, now, why=f"the branch exposes {len(f['legs'])} factor(s) (missing {f['missing']})")
            return [f"leg factorial {tag}: too few factors on branch {br.id} (missing {f['missing']}); closed"]
        reset = {x["key"]: v for x, c in zip(f["legs"], corner(f["legs"], False))
                 if t.base.get(x["key"]) != (v := x["high"] if c > 0 else x["low"])}
        for k, v in reset.items():          # the branch's base = the carrier's corner: its row IS the carrier's control
            t.text = set_yaml_value(t.text, k, v)
            t.base[k] = v
        identical = " identical=1" in f" {br.notes}"
        seeds = _fresh_seeds(ctx, store, 2 ** odd_words(len(f["legs"]))[0] // BLOCK, f"legs {tag}")
        cur.update(identical=identical, missing=f["missing"], seeds=seeds, reset=reset)
        for name, fac in (("legs", f["legs"]), ("dose", f["dose"] if ctx.cfg.legs.dose else [])):
            if len(fac) < 2:
                continue
            b, _, rows = odd_words(len(fac))
            points: dict[str, str] = {}
            for i, row in enumerate(rows):
                s = seeds[block_of(b, i)]
                nid = t.node({kb["key"]: kb["high"] if x > 0 else kb["low"] for kb, x in zip(fac, row)}, knobs=fac,
                             tag=f"{tag}{name[0]}pt{i:02d}", operator=None, new_id=new_id,
                             notes=f"legs={tag} seed1={s}")
                if nid == br.id:                       # the carrier's own row: its control run is that point
                    points[str(i)] = cur["carrier"]
                elif nid:
                    store.update_node(nid, parent_id=cur["carrier"])
                    points[str(i)] = nid
            cur["designs"][name] = {"factors": fac, "rows": [list(r) for r in rows], "points": points,
                                    "corner": corner(fac, name == "dose"), "b": b}
    cur["phase"] = "run"
    store.set_state(K_LEGS, bk)
    store.log("legs_designed", batch=cur["batch"], branch=br.id, identical=identical, seeds=seeds,
              **{n: {"factors": [x["key"] for x in d["factors"]], "points": len(d["points"])}
                 for n, d in cur["designs"].items()})
    return [f"leg factorial {tag} on {cur['carrier']} via branch {br.id}: " + "; ".join(
        f"{n} design {len(d['factors'])} factors x {len(d['rows'])} runs ({len(d['points'])} points)"
        for n, d in cur["designs"].items()) + f"; 1 run per point, {BLOCK} per seed on seeds {seeds}"
        + ("" if identical else "; branch NOT declared identical: the carrier's runs are not a design row")
        + (f"; no key for {f['missing']}" if f["missing"] else "")
        + (f"; branch config reset to the carrier's levels {reset}" if reset else "")]


def _obs(ctx: LoopContext, store: Any, cur: Mapping[str, Any], d: Mapping[str, Any]) -> list[tuple[int, int, float]]:
    """(row, seed, panel) of every finished point; the carrier's control at each seed as the extra corner row
    (index len(rows)) when the branch is identical."""
    out, seeds = [], set()
    for i, nid in d["points"].items():
        if nid == cur["carrier"]:
            continue
        for r in store.runs(node_id=nid, stage=Stage.SCREEN):
            if (y := D.panel_of(ctx, r)) is not None:
                out.append((int(i), int(r.seed), y))
                seeds.add(int(r.seed))
    if cur.get("identical"):
        for r in store.runs(node_id=cur["carrier"], stage=Stage.PARENT):
            if int(r.seed) in seeds and (y := D.panel_of(ctx, r)) is not None:
                out.append((len(d["rows"]), int(r.seed), y))
    return out


def se_floor(ana: dict[str, Any], sigma: float | None, z: float) -> float:
    """Floor every estimable effect's se at the panel-sigma-implied ``2 sigma / sqrt(N)`` (N scored runs): the
    residual-sd se of a 16-run design has ~11 df and runs small by chance (live L2 dose: salient tau -0.0044 starred
    on a residual se well under the panel's). The residual se is kept as ``se_resid``. Returns the floor."""
    if not sigma or not ana.get("n"):
        return 0.0
    floor = 2.0 * float(sigma) / math.sqrt(ana["n"])
    for e in ana["effects"]:
        if e["estimable"] and e["se"] < floor:
            e.update(se_resid=e["se"], se=floor, significant=abs(e["effect"]) >= z * floor)
    return floor


def analyse(ctx: LoopContext, store: Any, cur: Mapping[str, Any], d: Mapping[str, Any],
            sigma: float | None = None) -> dict[str, Any]:
    """loop.design.analyze over the rows (+ the corner), each se floored at the panel ``sigma`` (:func:`se_floor`);
    the fitted optimum (each factor at its better level) and its predicted gain over the carrier's own row."""
    fac = d["factors"]
    rows = [*map(tuple, d["rows"]), tuple(d["corner"])]
    ana = D.analyze([f["key"] for f in fac], rows, _obs(ctx, store, cur, d), ctx.cfg.design.z)
    se_floor(ana, sigma, ctx.cfg.design.z)
    tau = shrink(ana["effects"])
    best, gain, raw, xs = {}, 0.0, 0.0, []
    for f, e, c in zip(fac, ana["effects"], d["corner"]):
        if not e["estimable"]:
            best[f["key"]] = f["high"] if c > 0 else f["low"] if c < 0 else None
            xs.append(c)
            continue
        xr = 1 if e["effect"] > 1e-12 else -1 if e["effect"] < -1e-12 else c    # the raw (unshrunk) optimum
        raw += e["effect"] / 2 * (xr - c)
        flip = ctx.cfg.legs.flip_p
        x = 1 if e["p_pos"] >= flip and c < 1 else -1 if 1 - e["p_pos"] >= flip and c > -1 else c
        best[f["key"]] = f["high"] if x > 0 else f["low"] if x < 0 else None
        gain += e["shrunk"] / 2 * (x - c)
        xs.append(x)
    # the combination of ONLY the individually significant legs (|z| >= design.z), each moved to its better level
    sig, sig_gain = dict(best), 0.0
    for f, e, c in zip(fac, ana["effects"], d["corner"]):
        if e["estimable"]:
            x = (1 if e["effect"] > 0 and c < 1 else -1 if e["effect"] < 0 and c > -1 else c) \
                if e.get("significant") else c
            sig[f["key"]] = f["high"] if x > 0 else f["low"] if x < 0 else None
            sig_gain += e["shrunk"] / 2 * (x - c)
    return {**ana, "best": best, "gain": gain, "raw_gain": raw, "tau": tau, "x": xs, "sig_best": sig,
            "sig_gain": sig_gain}


def shrink(effects: Sequence[dict[str, Any]]) -> float:
    """Empirical-Bayes (positive-part James-Stein) main effects, in place: ``e_j ~ N(theta_j, se_j^2)``, ``theta_j ~
    N(0, tau^2)``, ``tau^2 = max(0, mean(e_j^2 - se_j^2))``; each effect gets ``shrunk`` = B_j e_j, ``shrunk_sd`` =
    sqrt(B_j) se_j and ``p_pos`` = P(theta_j > 0) (0.5 when tau = 0: the spread is pure noise), B_j = tau^2 / (tau^2 +
    se_j^2). The optimum of RAW effects is the max of k noisy draws (live L1: spread = noise, tau 0, predicted +0.0040,
    measured ~0); only a shrunk effect that is credibly non-zero moves a factor off the carrier. Returns tau."""
    est = [e for e in effects if e["estimable"]]
    if not est:
        return 0.0
    tau2 = max(0.0, sum(e["effect"] ** 2 - e["se"] ** 2 for e in est) / len(est))
    for e in est:
        v = e["se"] ** 2
        b = tau2 / (tau2 + v) if tau2 + v > 0 else 0.0
        m, sd = b * e["effect"], math.sqrt(b * v)
        e.update(shrunk=m, shrunk_sd=sd, p_pos=0.5 * math.erfc(-m / sd / math.sqrt(2.0)) if sd > 0 else 0.5)
    return math.sqrt(tau2)


def null_max_q(effects: Sequence[Mapping[str, Any]], corner: Sequence[int], q: float, seed: int = 0,
               sims: int = 4000) -> float:
    """The ``q`` quantile of the raw main-effects optimum's predicted gain under the GLOBAL NULL (every true effect 0,
    each estimate N(0, se_j), independent: the mains are orthogonal): what picking each factor's better level from
    pure noise 'gains'. Per factor the optimum takes ``max_x g/2 (x - c) = (|g| - g c) / 2``."""
    import random
    rng, ses = random.Random(seed), [(e["se"], c) for e, c in zip(effects, corner) if e["estimable"] and e["se"]]
    draws = sorted(sum((abs(g) - g * c) / 2 for s, c in ses for g in (rng.gauss(0.0, s),)) for _ in range(sims))
    return draws[min(sims - 1, int(q * sims))] if ses else 0.0


def confirm_mde(ctx: LoopContext, store: Any, snapshot: Any) -> float:
    """The confirm MDE: delta_1 of the confirm E-value at sigma_ucb, the graduate's screen + planned confirm seeds."""
    from loop.world.evalues import STATE_KEY, confirm_delta, sigma_ucb
    cfg, sp = ctx.cfg, ctx.cfg.seeds
    sigma = float(snapshot.sigma.get(PANEL, cfg.panel_sd)) if snapshot is not None else cfg.panel_sd
    row = (store.get_state(STATE_KEY) or {}).get(PANEL) or {}
    sig = sigma_ucb(row["sigma"], row["df_eff"]) if "sigma" in row and "df_eff" in row else sigma
    n_scr, cap = len(sp.screen), sp.max_seeds_per_node
    n = n_scr + (max(1, min(cfg.promotion.confirm_max_seeds, cap - n_scr)) if cap > 0 else cfg.promotion.confirm_max_seeds)
    return confirm_delta(sig, cfg.promotion.alpha_confirm, n)


def shrunk_bar(ctx: LoopContext, store: Any, snapshot: Any) -> float:
    """The bar a SHRUNK estimate alone graduates at (``promotion.shrunk_grad_mde`` x :func:`confirm_mde`; inf = off):
    the selection-corrected estimate goes to fresh confirmation, which decides, without a second null-max gate."""
    f = float(getattr(ctx.cfg.promotion, "shrunk_grad_mde", 0.0) or 0.0)
    return f * confirm_mde(ctx, store, snapshot) if f > 0 else math.inf


def graduation(ctx: LoopContext, store: Any, snapshot: Any, d: Mapping[str, Any], ana: Mapping[str, Any],
               seed: int = 0) -> dict[str, Any]:
    """Graduate a design's optimum when its SHRUNK predicted gain >= :func:`shrunk_bar`, or (the null rule) when it is
    >= the confirm MDE AND its RAW main-effects optimum beats the ``legs.null_q`` quantile of the simulated global-null
    max (live L1: +0.0040 sat at the null's 32nd percentile). Fresh confirmation decides either way. When the shrunk
    optimum fails both, the combination of only the individually significant legs (``sig_best``) graduates if its
    shrunk gain clears the bar. {ok, why, reason, levels ('optimum' | 'significant'), mde, bar, null_q, raw, gain}."""
    mde, bar = confirm_mde(ctx, store, snapshot), shrunk_bar(ctx, store, snapshot)
    q = ctx.cfg.legs.null_q
    nq = null_max_q(ana["effects"], d["corner"], q, seed)
    gain, raw = ana["gain"], ana["raw_gain"]
    why = [w for w, bad in ((f"shrunk gain {gain:+.5f} < confirm MDE {mde:.5f}", gain < mde),
                            (f"raw optimum {raw:+.5f} <= global-null max q{q:g} {nq:+.5f}", raw <= nq)) if bad]
    out = {"mde": mde, "bar": bar if bar < math.inf else None, "null_q": nq, "raw": raw, "gain": gain, "levels": "optimum", "why": "", "ok": True}
    if not why:
        return {**out, "reason": f"raw optimum {raw:+.5f} > null q{q:g} {nq:+.5f}"}
    if gain >= bar:
        return {**out, "reason": f"graduate on shrunk {gain:+.5f} >= bar {bar:.5f} (null gate not passed)"}
    sg = float(ana.get("sig_gain") or 0.0)
    if sg >= bar and ana.get("sig_best") is not None:
        return {**out, "gain": sg, "levels": "significant",
                "reason": f"graduate the significant legs only on shrunk {sg:+.5f} >= bar {bar:.5f} (null gate not passed)"}
    return {**out, "ok": False, "reason": "",
            "why": "; ".join(why) + (f"; shrunk {gain:+.5f} < bar {bar:.5f}" if bar < math.inf else "")}


def grad_levels(ana: Mapping[str, Any], gate: Mapping[str, Any]) -> dict[str, Any]:
    """The graduate's factor levels: the shrunk optimum, or the significant-legs-only combination."""
    lv = ana["sig_best"] if gate.get("levels") == "significant" else ana["best"]
    return {k: v for k, v in lv.items() if v is not None}


def design_closed(store: Any, cur: Mapping[str, Any], d: Mapping[str, Any]) -> bool:
    """Every point of design ``d`` closed (the carrier's own row is its control run, never a point)."""
    return all((n := store.get_node(i)) is not None and n.status in CLOSED_STATUSES
               for i in d["points"].values() if i != cur["carrier"])


def leg_point(node: Node | None) -> bool:
    """A leg-factorial point (``legs=L<b>`` in its notes; its graduated optimum carries ``legs_of=`` too)."""
    return node is not None and node.operator is None and f" {node.notes or ''}".find(" legs=L") >= 0 \
        and "legs_of=" not in (node.notes or "")


def stragglers(store: Any) -> set[str]:
    """The current batch's open points of a design not yet analysed with at most one block (BLOCK) left open: the
    driver runs them HIGH (primary allowed, moved off a pending preemptible slot) -- the whole analysis waits on
    them (live L1: 2 dose points pending at low QoS, SLURM's expected start ~9 h out)."""
    cur, out = book(store)["current"], set()
    if cur is None or cur.get("phase") != "run":
        return out
    for name, d in cur.get("designs", {}).items():
        if name in (cur.get("result") or {}):
            continue
        open_ = [i for i in d["points"].values() if i != cur["carrier"]
                 and not ((n := store.get_node(i)) is not None and n.status in CLOSED_STATUSES)]
        if len(open_) <= BLOCK:
            out.update(open_)
    return out


def _finish(ctx: LoopContext, store: Any, adapter: Any, bk: dict, cur: dict, new_id: Any, now: float,
            snapshot: Any = None) -> list[str]:
    """Analyse each design AS SOON AS its own points closed (its optimum goes to confirm then), the batch closing
    once every design is analysed: one design never waits on the other's stragglers (live L1: the 32-point legs
    design closed 01:35Z and sat unanalysed behind two pending dose points SLURM expected to start ~12 h later)."""
    tag, acts, result = f"L{cur['batch']}", [], cur.setdefault("result", {})
    for name, d in cur["designs"].items():
        if name in result or not design_closed(store, cur, d):
            continue
        ana = analyse(ctx, store, cur, d, float(snapshot.sigma.get(PANEL, ctx.cfg.panel_sd)) if snapshot is not None
                      else ctx.cfg.panel_sd)
        lines = [f"{f.get('leg') or f['key']} {D.fmt_effect(e)}" for f, e in zip(d["factors"], ana["effects"])]
        res: dict[str, Any] = {"effects": ana["effects"], "interactions": ana["interactions"], "n": ana["n"],
                               "df": ana["df"], "best": ana["best"], "gain": ana["gain"], "node": "",
                               "raw_gain": ana["raw_gain"], "tau": ana["tau"]}
        gate = graduation(ctx, store, snapshot, d, ana, cur["batch"])
        levels = grad_levels(ana, gate)
        res.update(gate=gate)
        if gate["gain"] > 0 and levels and gate["ok"]:
            res["node"] = _graduate(ctx, store, adapter, cur, name, levels, gate["gain"], new_id, now)
        result[name] = res
        store.log("legs_analysed", batch=cur["batch"], design=name, gain=ana["gain"], node=res["node"], gate=gate,
                  effects={f.get("leg") or f["key"]: e.get("effect") for f, e in zip(d["factors"], ana["effects"])})
        acts.append(f"leg factorial {tag} {name} design ({ana['n']} runs, df {ana['df']}): " + "; ".join(lines)
                    + f" -> optimum predicted {ana['gain']:+.5f} over {cur['carrier']} (EB-shrunk, tau "
                      f"{ana['tau']:.5f}; raw main-effects optimum {ana['raw_gain']:+.5f})"
                    + (f": node {res['node']} ({gate['reason']}) -> confirm / audit on fresh seeds" if res["node"] else
                       f": no graduate ({gate['why']})" if not gate["ok"] else ": nothing to send"))
    if len(result) < len(cur["designs"]):
        if acts:
            store.set_state(K_LEGS, bk)
        return acts
    _close(store, bk, cur, now)
    return acts


def on_legs(cur: Mapping[str, Any], fac: Sequence[Mapping[str, Any]], levels: Mapping[str, Any]) -> list[str]:
    """The legs a combination keeps ON (its ``legs_on``, read by :func:`constituents`: the next round on it then
    treats them as the carrier's own legs, never as new ones): a carrier leg unless set to 0, a new leg when set to anything but its 0 (off)."""
    on = set(cur["own"])
    for f in fac:
        if not f.get("leg") or f["key"] not in levels:
            continue
        (on.add if levels[f["key"]] not in (0, 0.0, False) else on.discard)(f["leg"])
    return sorted(on)


def _graduate(ctx: LoopContext, store: Any, adapter: Any, cur: Mapping[str, Any], name: str,
              levels: Mapping[str, Any], gain: float, new_id: Any, now: float) -> str:
    """The optimum as a node on the branch, parent the carrier, a fresh block, no screen gate (``legs_of=``)."""
    tag = f"L{cur['batch']}{name[0]}best"
    hyp = HypothesisRecord(
        id="", task_id=ctx.task_id, kind=HypothesisKind.TRIAL, family=DESIGN_FAMILY, mechanism_tag=tag,
        claim=f"leg factorial L{cur['batch']} ({name}) optimum on {cur['carrier']}: {dict(levels)}",
        predictions={PANEL: AxisPrediction(1, max(1e-4, gain))},
        falsification=f"it does not beat {cur['carrier']} on fresh seeds")
    on = on_legs(cur, cur["designs"][name]["factors"], levels)
    with D._Tree(ctx, store, adapter, f"legs{cur['batch']}-{name}", base_id=cur["branch"]) as t:
        nid = t.node(dict(levels), knobs=cur["designs"][name]["factors"], tag=tag, operator=Operator.DRAFT,
                     new_id=new_id, hyp=hyp, notes=f"legs_of={cur['carrier']} legs=L{cur['batch']}"
                     + (f" legs_on={','.join(on)}" if on else ""))
    if not nid or nid == cur["branch"]:
        return ""
    store.update_node(nid, parent_id=cur["carrier"])
    return nid


# ------------------------------------------------------------------------------------------------ swap tests
def _contribution(snapshot: Any, nid: str) -> float | None:
    """A leg's own gain: its fresh confirm seeds' mean when it has any, else its posterior increment."""
    if (oc := (getattr(snapshot, "own_confirm", None) or {}).get(nid)) is not None:
        return float(oc[0])
    p = snapshot.nodes.get(nid)
    return p.increment.mean if p is not None else None


def swaps(ctx: LoopContext, store: Any, adapter: Any, snapshot: Any, *, new_id: Any, now: float) -> list[str]:
    """SWAP TESTS: a queued child X of the incumbent that declares (``conflicts_with=``) a conflict with an incumbent
    leg L whose own gain is under ``legs.swap_max_sigmas`` sigma also runs as X with L turned OFF (L's config key --
    its ``leg_<L>`` flag / dose key -- at 0; a config-only child, parent the incumbent): ``I(X, L) = inc(X) -
    inc(swap) - e_L``, and the swap may simply win. Each X is considered once."""
    cfg, cid = ctx.cfg.legs, getattr(snapshot, "incumbent_id", ROOT_NODE_ID)
    if cfg.swap_max_sigmas <= 0 or snapshot is None or cid == ROOT_NODE_ID:
        return []
    before = store.get_state(K_SWAP) or {}
    done, acts = dict(before), []
    mine = [i for i in own_legs(store, cid) if i != cid]
    sigma = float(snapshot.sigma.get(PANEL, ctx.cfg.panel_sd))
    for n in store.nodes(status=[NodeStatus.QUEUED, NodeStatus.SCREENING]):
        if n.id in done or (n.parent_id or ROOT_NODE_ID) != cid or n.strategy is Strategy.DESIGN or not mine \
                or _note(n, "legs_of") or _note(n, "swap_of"):
            continue
        done[n.id] = ""
        said = set(_note(n, "conflicts_with").split(","))
        for leg in (leg for leg in mine if leg in said):
            e_l = _contribution(snapshot, leg)
            if e_l is None or abs(e_l) > cfg.swap_max_sigmas * sigma:
                continue
            cfg_x = _cfg(store, n.id)
            key = next((k for k in cfg_x if re.search(rf"(?:^|[._]){re.escape('leg_' + leg)}$", k)), "") \
                or dose_key(store, leg, cid)
            v = cfg_x.get(key)
            if not key or not D._num(v) or not v:
                continue
            with D._Tree(ctx, store, adapter, f"swap-{n.id}-{leg}", base_id=n.id) as t:
                sid = t.node({key: type(v)(0)}, knobs=[{"key": key, "low": type(v)(0), "high": v, "levers": {}}],
                             tag=f"{n.id}_without_{leg}", operator=Operator.DRAFT, new_id=new_id,
                             notes=f"swap_of={n.id} swap_off={leg}")
            if not sid or sid == n.id:
                continue
            store.update_node(sid, parent_id=cid)
            done[n.id] = sid
            store.log("legs_swap", node_id=n.id, leg=leg, swap=sid, key=key, declared=True, leg_gain=e_l)
            acts.append(f"swap test {sid}: {n.id} with incumbent leg {leg} OFF ({key}=0; "
                        f"declared conflict, {leg}'s own gain {e_l:+.4f})")
            break
    if done != before:
        store.set_state(K_SWAP, done)
    return acts


# ------------------------------------------------------------------------------------------------ researcher
def requests(ctx: LoopContext, store: Any, snapshot: Any, limit: int) -> list[CandidateRequest]:
    bk = book(store)
    cur = bk["current"]
    if limit <= 0 or cur is None or cur["phase"] != "build" or not cur["want"] or cur["pending"]:
        return []
    car = store.get_node(cur["carrier"])
    req = CandidateRequest(id=f"legs-L{cur['batch']}-{cur['fails'] + 1}", task_id=ctx.task_id,
                           strategy=Strategy.DESIGN, operator=Operator.IMPROVE, island=car.island,
                           parent_ids=(car.id,), brief=brief(ctx, store, cur), brief_kind="legs",
                           family_hint=car.family)
    cur.update(pending=req.id, want=False)
    store.set_state(K_LEGS, bk)
    return [req]


def brief(ctx: LoopContext, store: Any, cur: Mapping[str, Any]) -> str:
    keys = ", ".join(f"{g} ({f})" for f, r in ctx.cfg.families.items() if not r.init_check for g in r.keys if "*" in g)
    lines = [f"LEG FACTORIAL L{cur['batch']}: build ONE branch on the carrier {cur['carrier']} (your parent) where "
             "every leg below can be switched by config, so a designed batch can turn each on / off in every "
             "combination. At the defaults the run MUST be numerically identical to the carrier (same ops, same "
             "order, same RNG draws): smoke it and set hypothesis.identical_at_default true only when you verified it.",
             "Every key you add: ONE number written into the config file, its name ENDING in leg_<node id> (e.g. "
             "<section>.<prefix>leg_n0123), matching a key pattern of a family whose changes need no init check: "
             f"{keys}. Change no other config value.", "NEW LEGS -- graft each one's change onto the carrier behind "
             "its flag (0 = off = the carrier exactly, the default; 1 = the leg exactly as it was measured, its own "
             "config values hard-coded or as extra keys read only when the flag is 1). Read each leg's diff "
             "(path below; or git diff <its parent commit> <its commit>):"]
    for r in cur["legs"]:
        n = store.get_node(r["id"])
        p = store.get_node(r["parent"])
        lines.append(f"  {r['id']} [{r['subsystem']}] {r['mechanism']} (P(own gain>0)={r['p']} over {r['seeds']} "
                     f"seeds; built on {r['parent']}): diff {n.diff_path or '-'}; commits {p.commit if p else '?'}.."
                     f"{n.commit}; config {store.get_state(f'config_diff:{r['id']}') or {}}")
    expose = [i for i in cur["own"] if not dose_key(store, i, cur["carrier"])]
    if expose:
        lines.append("CARRIER LEGS whose strength is hard-coded -- expose each as ONE strength multiplier key ending "
                     "in leg_<id> (default 1 = the carrier exactly; 0 = that leg OFF exactly, as if it was never "
                     "added; 0.5 / 2 = half / double strength, e.g. a loss weight x it, a sampling reweighting "
                     "exponent x it):")
        lines += [f"  {i} {store.get_node(i).mechanism_tag}: diff {store.get_node(i).diff_path or '-'}" for i in expose]
    keep = [i for i in cur["own"] if i not in expose]
    if keep:
        lines.append("Carrier legs already set by a config key (0 turns them off; do NOT add a key for them): "
                     + ", ".join(f"{i}={dose_key(store, i, cur['carrier'])}" for i in keep) + ".")
    lines.append("Commit the config file AT THE DEFAULTS (every new flag 0, every multiplier 1): reset every value "
                 "you changed to smoke before you finish.")
    lines.append("A leg you cannot graft cleanly (a conflict, an init-incompatible change): leave it out and say "
                 "which in the handoff. Keep the carrier's hypothesis family; mechanism_tag leg_branch_L"
                 f"{cur['batch']}.")
    if cur.get("why"):
        lines.append(f"The previous attempt failed: {cur['why']}.")
    return "\n".join(lines)


def adopt(store: Any, request: CandidateRequest, node: Node) -> Node:
    """The branch: a never-run CARRIER node (rejected, the points are built on it); a rejected one is retried."""
    if request.brief_kind != "legs":
        return node
    bk = book(store)
    cur = bk["current"]
    if cur is None or cur["pending"] != request.id:
        return dataclasses.replace(node, status=NodeStatus.REJECTED, reject_reason="legs_batch_closed")
    cur["pending"] = ""
    if node.status is NodeStatus.REJECTED:
        cur.update(got="rejected", why=node.reject_reason)
        store.set_state(K_LEGS, bk)
        return dataclasses.replace(node, notes=f"{node.notes} legs=L{cur['batch']}")
    cur.update(branch=node.id, got="ok")
    store.set_state(K_LEGS, bk)
    return dataclasses.replace(node, strategy=Strategy.DESIGN, operator=None, family=DESIGN_FAMILY,
                               mechanism_tag=f"L{cur['batch']}branch", status=NodeStatus.REJECTED,
                               reject_reason=CARRIER, notes=f"{node.notes} legs=L{cur['batch']}")


def outage(store: Any, request_id: str) -> str:
    bk = book(store)
    cur = bk["current"]
    if cur is None or cur["pending"] != request_id:
        return ""
    cur.update(pending="", want=True)
    store.set_state(K_LEGS, bk)
    return f"L{cur['batch']}"


def hb(store: Any, now: float | None = None) -> str:
    """The heartbeat's stacking field: ``legs=L4(run on a node: 6 legs, 24/32 runs)``, else the last round's age."""
    bk, now = book(store), time.time() if now is None else now
    if (c := bk["current"]) is not None:
        pts = [store.get_node(i) for d in c.get("designs", {}).values() for i in d["points"].values() if i != c["carrier"]]
        done = sum(n is not None and n.status in CLOSED_STATUSES for n in pts)
        return (f"legs=L{c['batch']}({c['phase']} on {c['carrier']}: {len(c['legs'])} legs"
                + (f", {done}/{len(pts)} runs" if pts else "") + ")")
    if not bk["batches"]:
        return "legs=none"
    b = bk["batches"][-1]
    return f"legs=idle(L{b['batch']} on {b['carrier']} closed {(now - float(b.get('closed', now))) / 3600.0:.0f} h ago)"


def lines(store: Any) -> str:
    bk, out = book(store), []
    for c in bk["batches"]:
        res = c.get("result") or {}
        out.append(f"legs L{c['batch']} (round {c.get('round', 1)}) on {c['carrier']}: " + (
            c.get("why") or c.get("error") or "; ".join(
                f"{n} optimum {r['gain']:+.5f}" + (f" (main effects alone {r['main_gain']:+.5f})" if "main_gain" in r else "")
                + f" -> {r['node'] or 'none'}" for n, r in res.items())))
    if (c := bk["current"]) is not None:
        pts = [store.get_node(i) for d in c.get("designs", {}).values() for i in d["points"].values() if i != c["carrier"]]
        out.append(f"legs L{c['batch']} (round {c.get('round', 1)}) on {c['carrier']} running: {c['phase']}"
                   + (f", branch {c['branch']}" if c["branch"] else f", branch request {c['pending'] or 'wanted'}")
                   + (f", {sum(bool(n) and n.status in CLOSED_STATUSES for n in pts)}/{len(pts)} points closed" if pts else "")
                   + "".join(f"; {n} design analysed: optimum {r['gain']:+.5f} -> {r['node'] or 'none'}"
                             for n, r in (c.get("result") or {}).items())
                   + f"; legs {[r['id'] for r in c['legs']]}, carrier legs {c['own']}")
    if sw := {x: s for x, s in (store.get_state(K_SWAP) or {}).items() if s}:
        out.append("swap tests (idea -> idea with the conflicting incumbent leg off): "
                   + ", ".join(f"{x}->{s}" for x, s in sw.items()))
    return "\n".join(out)


__all__ = ["K_LEGS", "K_SWAP", "design_closed", "leg_point", "stragglers", "on_legs", "optimum", "swaps", "adopt", "analyse", "block_of", "book", "candidates", "factors", "odd_words", "outage", "plan",
           "carriers", "hb", "new_cap", "pool", "promising", "requests", "subsystem", "tick"]
