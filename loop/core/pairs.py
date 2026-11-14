"""loop.core.pairs — CODE-PICKED crossover pairs (v3-c6a; GEAR, arXiv 2605.13874: complementary pairs chosen by code
became elites 71% of the time, pairs chosen by the LLM 14%).

The planner reads the tick's posterior (the VISIBLE ledger's: ``loop.core.holdout``) and picks partner pairs to merge
onto the incumbent:

- LEGS   nodes with an own-increment estimate on >= ``pairs.min_seeds`` seeds, not FAILED / VOID / CRASHED (derived)
         nor VETOED / REJECTED (store), not a wave point, not already in the incumbent (its lineage, or every key of a
         config-only leg already at the leg's value), whose PARENT is sound (its own CUPED increment over its parent
         >= -``pairs.parent_tol``: a leg that only recovers a broken parent's loss transfers nothing), of the current
         base or -- ``pairs.old_epoch`` -- an older one (a port in the current base stands in for its original when it
         has its own estimate; an old-epoch original needs >= ``pairs.old_min_seeds`` seeds: an n=2 screen of an old
         base is mostly selection noise, and its shrunk mean counts at ``pairs.old_w``: older-base evidence, as the
         acquisition's prior discounts it). Never a COMBINATION (a crossover / stack / merged pair / factorial point:
         its own increment carries its legs' gains).
         A COMPOSITE leg has a shrunk own composite mean (``post.belief``) m > 0 with m >= ``pairs.leg_z`` x its sd.
         An AXIS leg (operator 10-13: discovery built in -- e.g. fewer local crops: rob +.016 at a small composite
         loss) has >= ``pairs.axis_min_seeds`` seeds, a shrunk composite >= -``pairs.axis_leg_tol`` and one axis
         (not in ``pairs.axis_exclude``: the visible survival axis is ~pure noise) whose own gain clears a
         BONFERRONI z over every candidate node x axis tested (one-sided ``pairs.axis_alpha``; never below
         ``pairs.axis_leg_z``): ~z 4 on the live ledger, so an axis leg is a real axis effect, not the max of noise.
- CONFIG a leg's config MOVES over its parent ({key: {from, to}}, read from git: ``git diff --name-only`` says
         whether the change touches only the config file; cached in state ``pairs.config_moves``). An unread leg
         (``pairs.config_reads_per_tick`` git reads per tick, best legs first) is :data:`UNREAD` -- never confused with
         "has code": a pair with an unread leg is planned but never merged, ported or briefed until it is read.
         TRANSFER (an old-epoch leg's absolute values are not its measured change): only the keys whose incumbent
         value equals the leg's PARENT value move (the same move on the incumbent); the other keys are dropped, and a
         leg with nothing left is no leg. Config legs whose moves overlap (item Jaccard >= ``pairs.dedup_jaccard``:
         design siblings a node / a node / a node, one colour-jitter halving) are ONE mechanism: the best-z one stays.
- AXES   each leg's per-axis own increment (CUPED over its parent, :func:`loop.core.model.cuped`, the axis's noise),
         shrunk by min(the posterior's own composite shrinkage, max(0, 1 - 1/z^2)), times the axis's weight in the
         SCREENING composite (``post.weights``): ``g[a]``. Its DOMINANT axis is the largest ``g[a]``.
- RANK   by the STACK-PRIOR prediction pred = ``acquire.stack_mean`` of the two shrunk legs (min(sum, max(.74 x best,
         .24 x sum + .0013)): stacks are sub-additive; it must be > ``pairs.min_pred``), optionally corrected for the
         WINNER'S CURSE: value = pred - ``pairs.curse`` x sqrt(2 ln N) x sqrt(sdA^2 + sdB^2) (default curse 0, v3
         review K5: the legs are posterior means already -- a curse on top double-shrinks), plus ``pairs.coverage_w``
         x coverage; a leg appears in at most ``pairs.leg_cap`` ranked pairs
         (coverage = sum_a max(gA, gB, 0) - max(sum gA+, sum gB+): what the union buys beyond the better leg). The
         repair term (one leg gains what the other loses) is logged, not ranked on: it double-counts the coverage. A
         pair needs DIFFERENT dominant axes, different mechanism groups (``acquire.same_mechanism``; no config key in
         common), at least one composite leg, an axis leg that repairs its partner's loss on its axis, and must never
         have been used (``K_USED``: merged, briefed, unbuilt or failed out).
- OUTPUT both legs config-only: a CONFIG-ONLY MERGE node on the incumbent (``factorial.build_point``), QUEUED
         (``pairs.builds_per_tick``, at most ``pairs.max_open`` still screening). A build failure backs off
         exponentially (``pairs.backoff_s`` x 2^(k-1)); ``pairs.max_failures`` failures drop the pair (marked
         ``failed``) -- only a deterministic "no change" marks it ``unbuilt`` at once. Otherwise a crossover brief names
         the exact pair: :func:`next_pair` (the ideation side; NO side effect) and :func:`claim` at the session's start
         (atomic under the store's write lock). An old-epoch CODE leg of a top-``pairs.top_k`` pair is ported first
         (``Tick.port_graft``, ``pairs.ports_per_tick``, at most ``pairs.max_open_ports`` ports still screening),
         booked in ``K_PORTED`` only once its port node exists, then paired once its port is measured.

Nothing is named or hardcoded: the 10-13 acceptance case (4 global crops x fewer local crops) is a test fixture."""

from __future__ import annotations

import hashlib
import logging
import math
import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from loop import project as P
from loop.core.data import AXES, OFFICIAL_WEIGHTS, Ledger

logger = logging.getLogger(__name__)

K_USED = "pairs.used"          # {"a|b": {"how": merge | brief | unbuilt | failed, "node": id, "ts": t}}
K_CFG = "pairs.config_moves"   # {node: {key: {"from": parent value, "to": value}} | null}: null = has code / no move
K_PLAN = "pairs.plan"          # the last top pairs (logged when they change)
PLAN_EPS = 0.0005             # a logged pair's pred must move this much (or the top-k set change) to be re-logged


def plan_moved(prev: Any, top: Sequence[Mapping[str, Any]], eps: float = PLAN_EPS) -> bool:
    """The top pairs changed materially since the last logged plan ``prev``: another top-k set (order aside) or a
    pred moved by more than ``eps`` (live 10-14: the same 3 lines every tick on a 1e-5 drift)."""
    if not isinstance(prev, list):
        return True
    old = {(t.get("a"), t.get("b")): float(t.get("pred") or 0.0) for t in prev if isinstance(t, Mapping)}
    new = {(t["a"], t["b"]): float(t["pred"] or 0.0) for t in top}
    return set(old) != set(new) or any(abs(new[k] - old[k]) > eps for k in new)
K_PORTED = "pairs.ported"      # old-epoch code legs whose port node the planner made: {leg: {for, port, ts}}
K_FAIL = "pairs.failures"      # {"pair:a|b" | "port:x" | "cfg:x": {n, next, why}}: retryable failures, backed off
TAG = "pair_of="
DEAD = ("FAILED", "VOID", "CRASHED")
DEAD_STORE = ("vetoed", "rejected")
OPEN = ("QUEUED", "SCREENING", "")


class _Unread:
    """A leg whose config diff was not read yet (the per-tick git budget): neither config-only nor code."""

    def __repr__(self) -> str:
        return "UNREAD"


UNREAD: Any = _Unread()
_UNKNOWN = object()           # a move whose parent value is unknown (a state ``config_diff`` with new values only)


@dataclass(frozen=True)
class Leg:
    node: str                    # the node as the ledger knows it (an old-epoch original, or a current one)
    src: str                     # the node whose estimate is read (its current-epoch port when measured)
    m: float                     # shrunk own composite mean (post.belief)
    sd: float
    n: int
    g: Mapping[str, float]       # per-axis shrunk own gain x screening weight
    raw: Mapping[str, float]     # per-axis own increment (axis units)
    z: Mapping[str, float]       # per-axis z
    kind: str                    # composite | axis
    base: str
    old: bool                    # an older base epoch than the posterior's
    cfg: Any = None              # {key: value} the leg transfers (config-only) | None (code) | UNREAD
    dropped: tuple[str, ...] = ()  # config keys not transferred (the incumbent no longer has the leg's parent value)

    @property
    def dom(self) -> str:
        return max(self.g, key=lambda a: (self.g[a], a)) if self.g else ""

    @property
    def strong(self) -> str:
        """The axis an AXIS leg is a leg for (its largest z among the eligible axes)."""
        return max(self.z, key=lambda a: (self.z[a], a)) if self.z else ""

    @property
    def zc(self) -> float:
        return self.m / self.sd if self.sd > 0 else (math.inf if self.m > 0 else 0.0)


@dataclass(frozen=True)
class Pair:
    a: str
    b: str
    comp: float                  # coverage + repair (logged only)
    coverage: float
    repair: float
    pred: float                  # additive: sum of the shrunk composite means
    dom: tuple[str, str]
    kinds: tuple[str, str]
    config: Mapping[str, Any] | None = None   # the merged config levels (both legs config-only), else None
    why: str = ""
    needs_port: tuple[str, ...] = field(default=())
    value: float = 0.0           # the ranking value: pred - the winner's-curse correction + coverage bonus
    unread: bool = False         # a leg's config diff is not read yet: never merged, ported or briefed

    @property
    def key(self) -> str:
        return pair_key(self.a, self.b)


def pair_key(a: str, b: str) -> str:
    return "|".join(sorted((a, b)))


def _opt(ctx: Any, name: str, default: Any) -> Any:
    return getattr(getattr(getattr(ctx, "cfg", None), "pairs", None), name, default)


def _status(statuses: Mapping[str, Any], nid: str) -> str:
    s = statuses.get(nid)
    s = getattr(s, "status", s)
    return str(getattr(s, "name", None) or getattr(s, "value", None) or s or "").upper()


def complementarity(ga: Mapping[str, float], gb: Mapping[str, float]) -> tuple[float, float, float]:
    """(comp, coverage, repair) of two per-axis weighted gain vectors (see the module docstring)."""
    axes = set(ga) | set(gb)
    pa = {a: max(0.0, ga.get(a, 0.0)) for a in axes}
    pb = {a: max(0.0, gb.get(a, 0.0)) for a in axes}
    la = {a: max(0.0, -ga.get(a, 0.0)) for a in axes}
    lb = {a: max(0.0, -gb.get(a, 0.0)) for a in axes}
    coverage = sum(max(pa[a], pb[a]) for a in axes) - max(sum(pa.values()), sum(pb.values()))
    repair = sum(min(pa[a], lb[a]) + min(pb[a], la[a]) for a in axes)
    return coverage + repair, coverage, repair


def curse_z(n: int) -> float:
    """The expected max of ``n`` standard-normal draws, ~sqrt(2 ln n) (0 for n <= 1)."""
    return math.sqrt(2.0 * math.log(n)) if n > 1 else 0.0


def bonferroni_z(m: int, alpha: float) -> float:
    """The one-sided z of a Bonferroni test over ``m`` hypotheses at family-wise ``alpha``."""
    if m <= 0 or not 0 < alpha < 1:
        return 0.0
    return statistics.NormalDist().inv_cdf(1.0 - alpha / m)


# -- config moves -------------------------------------------------------------------------------------------------
def moves_of(diff: Any) -> Any:
    """``{key: (from, to)}`` of a config-of value: UNREAD / None as they are; a ``{key: {from, to}}`` book entry as
    its moves; a plain ``{key: value}`` (a state ``config_diff``: new values only) with an unknown parent value."""
    if diff is UNREAD or diff is None:
        return diff
    if not isinstance(diff, Mapping) or not diff:
        return None
    out = {}
    for k, v in diff.items():
        if isinstance(v, Mapping) and set(v) == {"from", "to"}:
            out[k] = (v["from"], v["to"])
        else:
            out[k] = (_UNKNOWN, v)
    return out


def transfer(moves: Mapping[str, tuple[Any, Any]], inc_cfg: Mapping[str, Any] | None
             ) -> tuple[dict[str, Any], tuple[str, ...], bool]:
    """(the keys -> values a config leg moves on the incumbent, the keys dropped, whether the incumbent already holds
    every move). A key moves only while the incumbent still has the leg's PARENT value (else the absolute value would
    be a different move; an unknown parent value moves as before); a key the incumbent already holds at the leg's
    value is no move."""
    if inc_cfg is None:
        return {k: t for k, (_, t) in moves.items()}, (), False
    eff, dropped = {}, []
    held = all(inc_cfg.get(k) == t for k, (_, t) in moves.items())
    for k, (f, t) in moves.items():
        if inc_cfg.get(k) == t:
            continue
        if f is _UNKNOWN or inc_cfg.get(k) == f:
            eff[k] = t
        else:
            dropped.append(k)
    return eff, tuple(sorted(dropped)), held


def _items(moves: Mapping[str, Any]) -> set[tuple[str, str]]:
    return {(k, repr(v)) for k, v in moves.items()}


def jaccard(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    ia, ib = _items(a), _items(b)
    return len(ia & ib) / len(ia | ib) if ia | ib else 0.0


def dedup(legs: Sequence[Leg], thr: float) -> tuple[list[Leg], dict[str, str]]:
    """(the legs with config-move siblings collapsed to the best-z one, dropped -> kept). Only config legs collapse."""
    if thr <= 0:
        return list(legs), {}
    keep: list[Leg] = []
    gone: dict[str, str] = {}
    for leg in sorted(legs, key=lambda l: (-l.zc, -l.n, l.node)):
        if isinstance(leg.cfg, Mapping) and leg.cfg:
            twin = next((k for k in keep if isinstance(k.cfg, Mapping) and k.cfg and jaccard(k.cfg, leg.cfg) >= thr),
                        None)
            if twin is not None:
                gone[leg.node] = twin.node
                continue
        keep.append(leg)
    return keep, gone


# -- per-axis gains -----------------------------------------------------------------------------------------------
_AXES_CACHE: dict[tuple, tuple[dict[str, float], dict[str, float], dict[str, float]]] = {}


def _axes(post: Any, led: Ledger, src: str, shrink_c: float = 1.0
          ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """(weighted shrunk gains, raw increments, z) per axis of ``src``'s own increment over its parent (cached while
    neither node gains an observation)."""
    node = led.nodes[src]
    key = (src, node.base, len(post.by.get(src, ())), len(post.by.get(node.parent, ())), round(shrink_c, 6))
    if key not in _AXES_CACHE:
        if len(_AXES_CACHE) > 20000:
            _AXES_CACHE.clear()
        _AXES_CACHE[key] = _axes_fit(post, led, src, shrink_c)
    return _AXES_CACHE[key]


def _axes_fit(post: Any, led: Ledger, src: str, shrink_c: float) -> tuple[dict[str, float], dict[str, float],
                                                                       dict[str, float]]:
    from loop.core import model as M
    node = led.nodes[src]
    nz, wz = post.fit(node.base, src not in getattr(post, "near", ()) or bool(node.port_of))
    g, raw, z = {}, {}, {}
    for a in AXES:
        if a not in getattr(nz, "sigma2", {}):
            continue
        e = M.cuped(post.by[src], post.by[node.parent], a, nz, wz)
        if e is None or e.var <= 0:
            continue
        za = e.delta / math.sqrt(e.var)
        shrink = min(shrink_c, max(0.0, 1.0 - 1.0 / (za * za))) if za else 0.0
        raw[a], z[a] = e.delta, za
        g[a] = float((getattr(post, "weights", None) or OFFICIAL_WEIGHTS).get(a, 0.0)) * e.delta * shrink
    return g, raw, z


# -- legs ---------------------------------------------------------------------------------------------------------
@dataclass
class _Cand:
    nid: str
    orig: str
    node: Any
    m: float
    sd: float
    n: int
    delta: float
    old: bool


def legs(post: Any, led: Ledger, ctx: Any, statuses: Mapping[str, Any], *, store: Any = None, incumbent: str = "",
         inc_cfg: Mapping[str, Any] | None = None, config_of: Callable[[str], Any] | None = None,
         info: dict[str, Any] | None = None) -> list[Leg]:
    """The legs of this posterior (module docstring), best first. ``info`` (optional) gets the counts: candidates,
    the axis-leg z bar, the deduped siblings, the unread legs."""
    from loop.core import model as M
    from loop.core.acquire import legs_of, wave_node
    if post is None:
        return []
    min_n, old_n = int(_opt(ctx, "min_seeds", 2)), int(_opt(ctx, "old_min_seeds", 3))
    leg_z, old_ok = float(_opt(ctx, "leg_z", 1.0)), bool(_opt(ctx, "old_epoch", True))
    ax_z0, ax_tol = float(_opt(ctx, "axis_leg_z", 2.5)), float(_opt(ctx, "axis_leg_tol", 0.002))
    ax_n, ax_alpha = int(_opt(ctx, "axis_min_seeds", 3)), float(_opt(ctx, "axis_alpha", 0.05))
    ax_ex = _opt(ctx, "axis_exclude", None)
    ax_skip = set(P.axis_exclude() if ax_ex is None else ax_ex)     # None: the project's (nanopath: surv)
    old_z, ptol = float(_opt(ctx, "old_epoch_min_z", 0.0)), float(_opt(ctx, "parent_tol", 0.004))
    old_w, combos = float(_opt(ctx, "old_w", 1.0)), bool(_opt(ctx, "combo_legs", False))
    inside = set(legs_of(led, incumbent)) | ({incumbent} if incumbent else set())
    inside |= {led.nodes[x].port_of for x in list(inside) if x in led.nodes and led.nodes[x].port_of}
    ported = {n.port_of: k for k, n in led.nodes.items() if n.port_of and n.base == post.epoch}
    dead_store = _dead_ids(store)
    cands: list[_Cand] = []
    for nid, est in post.est.items():
        node = led.nodes.get(nid)
        if node is None or node.parent is None or wave_node(node) or node.lane in ("baseline",):
            continue
        orig = node.port_of or nid                       # a port stands for its original
        if orig in inside or nid in inside or _status(statuses, nid) in DEAD or _status(statuses, orig) in DEAD:
            continue
        src_node = led.nodes.get(orig) or node
        if not combos and _combo(src_node):
            continue                                     # a combination's own increment carries its legs' gains
        if node.base != post.epoch and (not old_ok or orig in ported and ported[orig] in post.est):
            continue                                     # an old node whose port is measured: the port is the leg
        if orig in dead_store or nid in dead_store:
            continue
        c = est.get(M.COMPOSITE)
        if c is None or c.n < min_n:
            continue
        pc = (post.est.get(node.parent) or {}).get(M.COMPOSITE)
        if pc is not None and pc.delta < -ptol:
            continue                                     # its parent lost ground: the leg's gain is a recovery
        m, v = post.belief(nid)
        old = node.base != post.epoch and not node.port_of
        if old and c.n < old_n:
            continue                                     # an old-epoch n=2 screen: mostly selection noise
        cands.append(_Cand(nid, orig, node, m, math.sqrt(max(v, 0.0)), c.n, c.delta, old))
    n_ax = len([a for a in AXES if a not in ax_skip])
    ax_bar = max(ax_z0, bonferroni_z(len(cands) * n_ax, ax_alpha)) if ax_z0 > 0 else math.inf
    picked: list[tuple[_Cand, str, tuple]] = []
    why_not: dict[str, int] = {}
    for c in cands:
        if c.m > 0 and c.m >= leg_z * c.sd and (not c.old or c.m >= old_z * c.sd):
            picked.append((c, "composite", _axes(post, led, c.nid, min(1.0, c.m / c.delta) if c.delta > 0 else 1.0)))
        elif ax_z0 > 0 and c.n >= ax_n and c.m >= -ax_tol:
            g, raw, z = _axes(post, led, c.nid)
            zz = {a: x for a, x in z.items() if a not in ax_skip}
            if not zz or max(zz.values()) < ax_bar:
                why_not["axis z < bar"] = why_not.get("axis z < bar", 0) + 1
                continue
            picked.append((c, "axis", (g, raw, zz)))
        else:
            k = "shrunk m <= 0" if c.m <= 0 else "m < leg_z sd"
            why_not[k] = why_not.get(k, 0) + 1
    picked.sort(key=lambda t: (-(t[0].m / t[0].sd if t[0].sd > 0 else 0.0), t[0].nid))   # best legs read first
    out: dict[str, Leg] = {}
    unread = 0
    for c, kind, (g, raw, z) in picked:
        cfg: Any = None
        dropped: tuple[str, ...] = ()
        if config_of is not None:
            mv = moves_of(config_of(c.orig))
            if mv is UNREAD:
                cfg, unread = UNREAD, unread + 1
            elif mv:
                cfg, dropped, held = transfer(mv, inc_cfg)
                if held:
                    continue                             # the incumbent already holds every key of it
                if not cfg:
                    continue                             # nothing of the move transfers: no leg
        # v3 review K5: older-base evidence is discounted ONCE -- post.belief already holds its V / OLD_EPOCH_W;
        # pairs.old_w (default 1) multiplies on top only when set below 1
        m_eff = c.m * old_w if c.old and old_w < 1.0 else c.m
        leg = Leg(c.orig, c.nid, m_eff, c.sd, c.n, g, raw, z, kind, c.node.base, c.old, cfg, dropped)
        if c.orig not in out or (leg.base == post.epoch, leg.n) > (out[c.orig].base == post.epoch, out[c.orig].n):
            out[c.orig] = leg
    kept, gone = dedup(list(out.values()), float(_opt(ctx, "dedup_jaccard", 0.5)))
    if info is not None:
        # v3-c7a A9: why candidates are no legs, and the current-epoch near misses (the shrunk composite z)
        near = sorted((c for c in cands if not c.old and c.node.base == post.epoch and c.sd > 0),
                      key=lambda c: -c.m / c.sd)[:3]
        info.update(candidates=len(cands), axis_z=round(ax_bar, 3) if math.isfinite(ax_bar) else None,
                    siblings=gone, unread=unread, legs=len(kept), why_not=why_not,
                    near_miss=[[c.nid, round(c.m, 5), round(c.sd, 5), c.n] for c in near])
    return sorted(kept, key=lambda l: (-l.m, l.node))


COMBO = re.compile(r"\b(xover|stack_legs|pair_of|legs)=")


def _combo(node: Any) -> bool:
    """A COMBINATION node (a crossover / stack / merged pair / factorial point: a second parent, the crossover or
    factorial lane, or an ``xover= / stack_legs= / pair_of= / legs=`` tag): its own increment over its parent carries
    its legs' gains (live 10-13: old crossovers of the confirmed a node onto root topped the plan at +.027), so it is no
    leg (``pairs.combo_legs``)."""
    return bool(node.second_parent) or node.lane in ("crossover", "factorial") or bool(COMBO.search(node.notes or ""))


def _dead_ids(store: Any) -> set[str]:
    """VETOED / REJECTED node ids in the store (one query)."""
    if store is None:
        return set()
    try:
        from loop.contracts import NodeStatus
        return {n.id for n in store.nodes(status=[NodeStatus(s) for s in DEAD_STORE])}
    except Exception:  # noqa: BLE001 - unknown: none
        return set()


# -- pairs --------------------------------------------------------------------------------------------------------
def plan(post: Any, led: Ledger, ctx: Any, statuses: Mapping[str, Any], *, store: Any = None, incumbent: str = "",
         inc_cfg: Mapping[str, Any] | None = None, config_of: Callable[[str], Any] | None = None,
         used: Mapping[str, Any] | None = None, info: dict[str, Any] | None = None) -> list[Pair]:
    """Every admissible pair, best first (module docstring)."""
    from loop.core.acquire import same_mechanism
    ls = legs(post, led, ctx, statuses, store=store, incumbent=incumbent, inc_cfg=inc_cfg, config_of=config_of,
              info=info)
    used = used if used is not None else ((store.get_state(K_USED) or {}) if store is not None else {})
    min_pred = float(_opt(ctx, "min_pred", 0.0))
    cov_w = float(_opt(ctx, "coverage_w", 0.25))
    ok_pairs: list[tuple[Leg, Leg]] = []
    why_not: dict[str, int] = {}

    def no(why: str) -> None:
        why_not[why] = why_not.get(why, 0) + 1
    for i, a in enumerate(ls):
        for b in ls[i + 1:]:
            if pair_key(a.node, b.node) in used:
                no("used")
                continue
            if a.dom == b.dom or not a.dom or not b.dom:
                no("same dominant axis")
                continue
            if a.kind == "axis" and b.kind == "axis":
                no("two axis legs")
                continue
            if stack_pred(a, b) <= min_pred:
                no("pred <= min_pred")
                continue
            da, db = a.cfg, b.cfg
            if isinstance(da, Mapping) and isinstance(db, Mapping) and set(da) & set(db):
                no("a shared config key")
                continue                                 # one config key: one mechanism (and not mergeable)
            try:
                if same_mechanism(led, a.node, b.node, ref=incumbent or None):
                    no("same mechanism")
                    continue
            except Exception:  # noqa: BLE001 - no grouping: distinct
                pass
            ok = True
            for x, y in ((a, b), (b, a)):              # an axis leg must repair an axis the other leg loses
                if x.kind == "axis" and not y.g.get(x.strong, 0.0) < 0 < x.g.get(x.strong, 0.0):
                    ok = False
            if ok:
                ok_pairs.append((a, b))
            else:
                no("axis leg repairs no axis the other loses")
    # v3-c7a A9: the winner's curse is the max over the INDEPENDENT legs the admissible pairs draw on (a leg in many
    # pairs is one draw, not many), each pair corrected by its own posterior sd
    n_eff = len({x.node for ab in ok_pairs for x in ab})
    cz = float(_opt(ctx, "curse", 0.0)) * curse_z(n_eff)
    if info is not None:
        info.update(curse_n=n_eff, pair_why_not=why_not,
                    leg_list=[(x.node, x.kind, x.dom, round(x.m, 5), round(x.sd, 5), bool(x.old)) for x in ls])
    out = []
    for a, b in ok_pairs:
        da, db = a.cfg, b.cfg
        pred = stack_pred(a, b)
        comp, cov, rep = complementarity(a.g, b.g)
        value = pred - cz * math.sqrt(a.sd * a.sd + b.sd * b.sd) + cov_w * cov
        unread = da is UNREAD or db is UNREAD
        merged = None
        if isinstance(da, Mapping) and isinstance(db, Mapping) and da and db and inc_cfg is not None \
                and all(k in inc_cfg for k in (*da, *db)):
            merged = {**da, **db}
        port = tuple(x.node for x in (a, b) if x.old and x.cfg is None)
        why = (f"{a.node} ({a.kind}, dom {a.dom} {_fmt(a)}) x {b.node} ({b.kind}, dom {b.dom} {_fmt(b)}): "
               f"pred {pred:+.5f} (stack prior of the shrunk legs: min(sum {a.m + b.m:+.5f}, max(.74 x best, "
               f".24 x sum + .0013)), ranked {value:+.5f}"
               + (f" after the winner's-curse correction (max of {n_eff} legs)" if cz > 0 else "")
               + f"; coverage {cov:+.5f}"
               + "".join(f"; {x.node} keeps {list(x.cfg)} (dropped {list(x.dropped)}: the incumbent moved them)"
                         for x in (a, b) if x.dropped and isinstance(x.cfg, Mapping)))
        out.append(Pair(a.node, b.node, comp, cov, rep, pred, (a.dom, b.dom), (a.kind, b.kind), merged,
                        why, port, value, unread))
    out.sort(key=lambda p: (-p.value, -p.pred, p.key))
    cap = int(_opt(ctx, "leg_cap", 2))
    if cap > 0:                                      # v3 review K5: one strong leg must not fill the whole top
        seen: dict[str, int] = {}
        kept = []
        for p in out:
            if seen.get(p.a, 0) >= cap or seen.get(p.b, 0) >= cap:
                continue
            seen[p.a], seen[p.b] = seen.get(p.a, 0) + 1, seen.get(p.b, 0) + 1
            kept.append(p)
        if info is not None:
            info.update(leg_capped=len(out) - len(kept))
        out = kept
    return out


def stack_pred(a: Leg, b: Leg) -> float:
    """A pair's predicted gain: the acquisition's STACK prior over its two shrunk legs (``acquire.stack_mean``,
    ledger 10-12: stack ~ .24 x the sum of legs + .0013, .74 x the best leg; 0% beat the sum) -- never the additive
    sum (v3 review K5)."""
    from loop.core.acquire import stack_mean
    return stack_mean([(a.m, a.node), (b.m, b.node)])


def _fmt(l: Leg) -> str:
    return " ".join(f"{a} {l.raw[a]:+.4f}" for a in AXES if a in l.raw and abs(l.z.get(a, 0.0)) >= 1.5) or "-"


# -- the ideation side's calls ------------------------------------------------------------------------------------
_BOUND: dict[str, Any] = {}


def bind(ctx: Any, store: Any, *, statuses: Mapping[str, Any] | None = None,
         config_of: Callable[[str], Any] | None = None, inc_cfg: Mapping[str, Any] | None = None,
         post: Any = None, pairs: Sequence[Pair] | None = None) -> None:
    """What :func:`next_pair` needs beyond the view (the tick binds it each tick, swapped in whole: a reader holds
    its own reference): the context, the store, the statuses, a READ-ONLY config lookup (cached diffs; UNREAD
    otherwise), the incumbent's config and -- ``post`` / ``pairs`` -- this tick's planned pairs."""
    global _BOUND
    _BOUND = dict(ctx=ctx, store=store, statuses=dict(statuses or {}), config_of=config_of, inc_cfg=inc_cfg,
                  post=post, pairs=tuple(pairs) if pairs is not None else None)


def briefable(p: Pair) -> bool:
    """A pair a crossover brief may take: not a config-only merge (the tick builds those), no leg to port first,
    every leg's config read."""
    return p.config is None and not p.needs_port and not p.unread


def next_pair(view: Any, *, mark: bool = False, exclude: Sequence[str] = ()) -> tuple[str, str, str] | None:
    """The best unused briefable pair (:func:`briefable`) for a crossover brief: ``(a, b, why)``; None when there is
    none. NO side effect by default (a dry run, a shadow or a re-plan never burns a pair): the session's start
    :func:`claim` s it. ``exclude``: pair keys already held by queued briefs. ``mark``: claim it now (tests / a
    synchronous caller). ``view``: ``loop.core.ideation.View``."""
    b = _BOUND
    ctx, store = b.get("ctx"), b.get("store")
    post = getattr(view, "post", None)
    if post is None or not bool(_opt(ctx, "enabled", True)):
        return None
    used = (store.get_state(K_USED) or {}) if store is not None else {}
    if b.get("pairs") is not None and b.get("post") is post:
        got = [p for p in b["pairs"] if p.key not in used]
    else:
        got = plan(post, view.led, ctx, b.get("statuses") or {}, store=store, incumbent=view.incumbent,
                   inc_cfg=b.get("inc_cfg"), config_of=b.get("config_of"), used=used)
    skip = set(exclude)
    for p in got:
        if not briefable(p) or p.key in skip:
            continue
        if mark and store is not None and not claim(store, p.a, p.b, "brief", ""):
            continue
        return p.a, p.b, p.why
    return None


def claim(store: Any, a: str, b: str, how: str, node: str, ts: float | None = None, **extra: Any) -> bool:
    """Mark the pair used, atomically (under the store's write lock when it has one): False when it already was."""
    key = pair_key(a, b)

    def fn(cur: Any) -> tuple[Any, bool]:
        cur = dict(cur or {})
        if key in cur:
            return None, False
        cur[key] = {"how": how, "node": node, "ts": ts, **extra}
        return cur, True
    upd = getattr(store, "update_state", None)
    ok = upd(K_USED, fn) if callable(upd) else _plain_update(store, K_USED, fn)
    if ok:
        store.log("pair_used", a=a, b=b, how=how, node=node, **extra)
    return bool(ok)


def release(store: Any, a: str, b: str, why: str = "", node: str | None = None) -> None:
    """Give a claimed brief pair back (its session never started); ``node``: only the claim made by that request."""
    key = pair_key(a, b)

    def fn(cur: Any) -> tuple[Any, bool]:
        cur = dict(cur or {})
        rec = cur.get(key) or {}
        if rec.get("how") != "brief" or (node is not None and rec.get("node") != node):
            return None, False
        cur.pop(key)
        return cur, True
    upd = getattr(store, "update_state", None)
    if (upd(K_USED, fn) if callable(upd) else _plain_update(store, K_USED, fn)):
        store.log("pair_released", a=a, b=b, why=why[:200])


def _plain_update(store: Any, key: str, fn: Callable[[Any], tuple[Any, Any]]) -> Any:
    new, res = fn(store.get_state(key))
    if new is not None:
        store.set_state(key, new)
    return res


def mark_used(store: Any, a: str, b: str, how: str, node: str, ts: float | None = None) -> None:
    """Mark the pair used (overwrites: the tick's own merge / unbuilt / failed records)."""
    key = pair_key(a, b)

    def fn(cur: Any) -> tuple[Any, None]:
        cur = dict(cur or {})
        cur[key] = {"how": how, "node": node, "ts": ts}
        return cur, None
    upd = getattr(store, "update_state", None)
    upd(K_USED, fn) if callable(upd) else _plain_update(store, K_USED, fn)
    store.log("pair_used", a=a, b=b, how=how, node=node)


# -- the tick's side -----------------------------------------------------------------------------------------------
class Planner:
    """Each tick: the pairs of the posterior; config-only merges built on the incumbent, old-epoch code legs ported;
    :func:`bind` for the ideation side; the top pairs logged (``pairs_plan``) when they change."""

    def __init__(self, ctx: Any, store: Any, adapter: Any, clock: Callable[[], float]) -> None:
        self.ctx, self.store, self.adapter, self.clock = ctx, store, adapter, clock
        self._book: dict[str, Any] | None = None     # the last FULLY BUILT config book (swapped in whole)
        self.info: dict[str, Any] = {}

    @property
    def on(self) -> bool:
        return bool(_opt(self.ctx, "enabled", True))

    def cached(self, nid: str) -> Any:
        """The ideation side's read-only lookup: the last built book's entry, else UNREAD (no git read off-tick)."""
        book = self._book
        return UNREAD if book is None else book.get(nid, UNREAD)

    def _reader(self, book: dict[str, Any], fails: dict[str, Any], budget: list[int]) -> Callable[[str], Any]:
        """This tick's lookup: ``book`` (cached in state), else up to ``budget[0]`` git reads (best legs ask first),
        else UNREAD. A read that raises is retried on later ticks (backed off); after ``pairs.max_failures`` the leg
        is booked as code (no merge) -- never a permanent verdict from one git hiccup."""
        def config_of(nid: str) -> Any:
            if nid in book:
                return book[nid]
            if budget[0] <= 0 or not self._due(fails, f"cfg:{nid}"):
                return UNREAD
            budget[0] -= 1
            try:
                got = self._config_diff(nid)
            except Exception as exc:  # noqa: BLE001 - git trouble: retried later, bounded
                if self._fail(fails, f"cfg:{nid}", repr(exc)):
                    book[nid] = None
                    return None
                return UNREAD
            book[nid] = got
            return got
        return config_of

    def _config_diff(self, nid: str) -> dict[str, Any] | None:
        """``nid``'s config moves over its parent ({key: {from, to}}) when its change touches only the config file;
        None when it has code (or no config move); raises when git cannot say (retried)."""
        from loop.adapters import flatten
        n = self.store.get_node(nid)
        p = self.store.get_node(n.parent_id) if n is not None and n.parent_id else None
        if n is None or p is None or not n.commit or not p.commit:
            return None
        if not hasattr(self.adapter, "config_at"):
            got = dict(self.store.get_state(f"config_diff:{nid}") or {})
            return got or None                         # new values only: their parent value is unknown
        cfg = getattr(self.adapter, "config_rel", None)
        if hasattr(self.adapter, "changed_files"):
            files = list(self.adapter.changed_files(p.commit, n.commit))
        elif hasattr(self.adapter, "diff_commits"):
            import re
            files = re.findall(r"^diff --git a/(\S+) b/", self.adapter.diff_commits(p.commit, n.commit) or "", re.M)
        else:
            got = dict(self.store.get_state(f"config_diff:{nid}") or {})
            return got or None
        if not files or [f for f in files if (f != str(cfg) if cfg else not f.endswith((".yaml", ".yml")))]:
            return None
        a, b = flatten(self.adapter.config_at(p.commit)), flatten(self.adapter.config_at(n.commit))
        if not a or not b:
            raise RuntimeError(f"config unreadable at {p.commit[:12]} / {n.commit[:12]}")
        diff = {k: {"from": a.get(k), "to": b.get(k)} for k in set(a) | set(b) if a.get(k) != b.get(k)}
        return diff or None

    # -- retryable failures ------------------------------------------------------------------------------------
    def _due(self, fails: Mapping[str, Any], key: str) -> bool:
        rec = fails.get(key)
        return rec is None or float(rec.get("next") or 0.0) <= self.clock()

    def _fail(self, fails: dict[str, Any], key: str, why: str) -> bool:
        """Count one failure of ``key`` (next try after ``pairs.backoff_s`` x 2^(k-1)); True once it reached
        ``pairs.max_failures`` (the caller gives up on it)."""
        rec = dict(fails.get(key) or {})
        n = int(rec.get("n", 0)) + 1
        base = float(_opt(self.ctx, "backoff_s", 600.0))
        fails[key] = {"n": n, "next": self.clock() + base * 2 ** (n - 1), "why": why[:300]}
        self.store.log("pairs_failure", key=key, n=n, why=why[:300])
        return n >= int(_opt(self.ctx, "max_failures", 3))

    # -- the tick ----------------------------------------------------------------------------------------------
    def step(self, tick: Any, led: Ledger, post: Any, statuses: Mapping[str, Any], incumbent: str,
             acts: list[str]) -> list[Pair]:
        from loop.adapters import flatten
        if not self.on or post is None:
            return []
        inc = self.store.get_node(incumbent)
        try:
            inc_cfg = flatten(self.adapter.config_at(inc.commit)) if inc is not None and hasattr(
                self.adapter, "config_at") else None
        except Exception:  # noqa: BLE001 - no incumbent config: no merges, briefs only
            inc_cfg = None
        book = dict(self.store.get_state(K_CFG) or {})
        fails = dict(self.store.get_state(K_FAIL) or {})
        fails0, n0 = dict(fails), len(book)
        budget = [int(_opt(self.ctx, "config_reads_per_tick", 60))]
        info: dict[str, Any] = {}
        pairs = plan(post, led, self.ctx, statuses, store=self.store, incumbent=incumbent, inc_cfg=inc_cfg,
                     config_of=self._reader(book, fails, budget), info=info)
        if len(book) != n0:
            self.store.set_state(K_CFG, book)
        self._book = book                               # swapped in whole: the ideation side reads it lock-free
        bind(self.ctx, self.store, statuses=statuses, config_of=self.cached, inc_cfg=inc_cfg, post=post, pairs=pairs)
        info.update(n_pairs=len(pairs), unread_pairs=sum(1 for p in pairs if p.unread),
                    needs_port=sum(1 for p in pairs if p.needs_port), reads=int(_opt(self.ctx, "config_reads_per_tick", 60)) - budget[0])
        self.info = info
        k = int(_opt(self.ctx, "top_k", 5))
        top = [{"a": p.a, "b": p.b, "pred": round(p.pred, 6), "value": round(p.value, 6),
                "coverage": round(p.coverage, 6), "merge": p.config is not None, "port": list(p.needs_port),
                "unread": p.unread, "why": p.why} for p in pairs[:k]]
        if plan_moved(self.store.get_state(K_PLAN), top):
            self.store.set_state(K_PLAN, top)
            self.store.log("pairs_plan", pairs=top, **{x: y for x, y in info.items() if x != "siblings"})
            for t in top[:3]:
                acts.append(f"pair {t['a']} x {t['b']}: {t['why']}"[:400])
        if not getattr(tick, "dry", False):
            acts += self._ports(tick, led, pairs[:k], statuses, fails)
            acts += self._merges(led, statuses, incumbent, inc, pairs, fails)
        if fails != fails0:
            self.store.set_state(K_FAIL, fails)
        return pairs

    def _open(self, led: Ledger, statuses: Mapping[str, Any]) -> int:
        return sum(1 for k, n in led.nodes.items() if TAG in (n.notes or "") and _status(statuses, k) in OPEN)

    def _merges(self, led: Ledger, statuses: Mapping[str, Any], incumbent: str, inc: Any, pairs: Sequence[Pair],
                fails: dict[str, Any] | None = None) -> list[str]:
        from loop.core import factorial as F
        if inc is None:
            return []
        fails = {} if fails is None else fails
        room = min(int(_opt(self.ctx, "builds_per_tick", 1)), int(_opt(self.ctx, "max_open", 2)) - self._open(led, statuses))
        out = []
        for p in pairs:
            if room <= 0:
                break
            if p.config is None or p.unread or not self._due(fails, f"pair:{p.key}"):
                continue
            cfg = dict(p.config)
            knobs = [F.Knob(k, None, v, "lin") for k, v in cfg.items()]
            try:
                sha, err = F.build_point(self.adapter, inc.commit, cfg, knobs, {}, f"pair-{p.a}-{p.b}",
                                         getattr(self.adapter, "config_rel", "config.yaml"))
            except Exception as exc:  # noqa: BLE001 - a build fault is retried (backed off), never quarantines the site
                sha, err = "", f"raised {exc!r}"
            if sha and sha == inc.commit or (not sha and not err):
                mark_used(self.store, p.a, p.b, "unbuilt", "", self.clock())    # deterministic: no change
                out.append(f"pair {p.a} x {p.b}: not built (no change)"[:300])
                continue
            if not sha:
                gave_up = self._fail(fails, f"pair:{p.key}", err)
                if gave_up:
                    mark_used(self.store, p.a, p.b, "failed", "", self.clock())
                out.append(f"pair {p.a} x {p.b}: build failed ({err}){'; given up' if gave_up else '; retried later'}"[:300])
                continue
            nid = self._child(inc, sha, p, cfg)
            mark_used(self.store, p.a, p.b, "merge", nid, self.clock())
            out.append(f"pair merge {nid} = {incumbent} + {p.a} + {p.b} (config {cfg}; pred {p.pred:+.5f})"[:400])
            room -= 1
        return out

    def _child(self, inc: Any, sha: str, p: Pair, cfg: Mapping[str, Any]) -> str:
        from loop.contracts import Node as StoreNode, NodeStatus, Operator, Strategy
        now = self.clock()
        seq = len(self.store.nodes())
        while self.store.get_node(f"n{seq:04d}") is not None:
            seq += 1
        nid = f"n{seq:04d}"
        a, b = self.store.get_node(p.a), self.store.get_node(p.b)
        fam = next((x.family for x in (a, b) if x is not None and x.family), "crossover")
        self.store.put_node(StoreNode(
            id=nid, task_id=inc.task_id, commit=sha, parent_id=inc.id, strategy=Strategy.CROSSOVER,
            operator=Operator.CROSSOVER, family=fam, mechanism_tag=f"pair_{p.a}_{p.b}",
            diff_hash=hashlib.sha256(f"{inc.commit}\n{sha}".encode()).hexdigest(), status=NodeStatus.QUEUED,
            second_parent_id=p.b, created_ts=now, updated_ts=now,
            notes=f"{TAG}{p.a},{p.b} pairs pred={p.pred:.6f} value={p.value:.6f} dom={p.dom[0]}+{p.dom[1]}"))
        self.store.set_state(f"config_diff:{nid}", dict(cfg))
        self.store.log("pair_merge", node_id=nid, a=p.a, b=p.b, pred=p.pred, value=p.value, comp=p.comp,
                       config=dict(cfg), why=p.why)
        return nid

    def open_ports(self, statuses: Mapping[str, Any], led: Ledger) -> int:
        """Planner-made ports still screening (their node QUEUED / SCREENING)."""
        book = self.store.get_state(K_PORTED) or {}
        return sum(1 for rec in book.values() if isinstance(rec, Mapping) and rec.get("port")
                   and rec["port"] in led.nodes and _status(statuses, rec["port"]) in OPEN)

    def _ports(self, tick: Any, led: Ledger, pairs: Sequence[Pair], statuses: Mapping[str, Any] | None = None,
               fails: dict[str, Any] | None = None) -> list[str]:
        """Old-epoch code legs of the TOP pairs: ported first (``Tick.port_graft``), paired once measured. At most
        ``pairs.ports_per_tick`` per tick and ``pairs.max_open_ports`` still screening; booked once the port node
        exists (a raising port is retried, backed off)."""
        from loop import cutover as CO
        statuses = statuses or {}
        fails = {} if fails is None else fails
        n = min(int(_opt(self.ctx, "ports_per_tick", 1)),
                int(_opt(self.ctx, "max_open_ports", 2)) - self.open_ports(statuses, led))
        out: list[str] = []
        if n <= 0 or not hasattr(tick, "port_graft"):
            return out
        book = dict(self.store.get_state(K_PORTED) or {})
        for p in pairs:
            if p.unread:
                continue
            for x in p.needs_port:
                if n <= 0 or x in book or not self._due(fails, f"port:{x}"):
                    continue
                try:
                    got = tick.port_graft(x, led)
                except Exception as exc:  # noqa: BLE001 - retried later (backed off), bounded
                    if self._fail(fails, f"port:{x}", repr(exc)):
                        book[x] = {"for": p.key, "ts": self.clock(), "port": "", "failed": repr(exc)[:200]}
                        self.store.set_state(K_PORTED, book)
                    out.append(f"port {x} for pair {p.key} raised: {exc!r}"[:300])
                    n -= 1
                    continue
                out += got
                pid = CO.port_id(x, led.epoch) if led.epoch else ""
                if pid and self.store.get_node(pid) is not None:
                    book[x] = {"for": p.key, "ts": self.clock(), "port": pid}
                    self.store.set_state(K_PORTED, book)
                n -= 1
        return out
