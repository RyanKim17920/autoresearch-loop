"""Paired deltas vs each node's control and the lazy noise refit (DESIGN §3.1-3.2). The control is
root's run at the seed (root child) or the parent's ``parent``-stage run (else its own done run at the seed);
voids never count.

Every child paired against one control RUN shares that run's own deviation ``b_c ~ N(0, sigma^2/2)``
(half the paired-delta variance). :func:`correct` subtracts its shrunk leave-one-out estimate from each
child's delta and carries ``Var(b_c | siblings)`` in the delta's variance, so siblings are not treated
as independent evidence of the control's luck (DESIGN §3.1)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from kernel.stats import paired_delta
from loop.contracts import PANEL, ROOT_NODE_ID, Node, NodeStatus, Run, RunStatus, Stage
from loop.spec import LoopContext

SELECTION_STAGES = (Stage.SCREEN, Stage.PROMOTE)
STATE_KEY = "world.noise"
# node -> [ancestor ids, nearest first]: the nodes its commit CONTAINS (git ancestors strictly below it, down to its
# declared parent) by effective-config distance; the first ESTABLISHED one is what its own increment is measured against
MEASURE_KEY = "world.measure_parent"
CAL_KEY = "calibration"                  # {"df": ...} written by loop.calibrate
_DEAD = (NodeStatus.VOID, NodeStatus.CRASHED)
OUTLIER_SIGMAS = 3.0                     # a sibling delta beyond this many sigma never estimates a control's offset


def targets(ctx: LoopContext) -> tuple[str, ...]:
    return (PANEL,) + tuple(ctx.cfg.axes)


def target_values(ctx: LoopContext, scores: Mapping[str, float],
                  base: Mapping[str, float]) -> dict[str, float]:
    """Per-target paired delta; the panel only when every axis is present."""
    pd = paired_delta(scores, base, ctx.cfg.weights)
    out = {a: pd.deltas[a] for a in ctx.cfg.axes if a in pd.deltas}
    if len(out) == len(ctx.cfg.axes):
        out[PANEL] = pd.aggregate
    return out


_REUSE = (Stage.PARENT, Stage.CONFIRM, Stage.PROMOTE, Stage.SCREEN)   # loop.seeds.REUSE_STAGES (non-root parent)


def controls(node: Node, root: Mapping[int, Run], parent_runs: Sequence[Run]) -> dict[int, Run]:
    """seed -> the control run a node is paired against (see module doc): the parent's ``parent`` run, else its own
    finished confirm / promote / screen run at that seed (the same commit and seed: reused, never relaunched)."""
    if (node.parent_id or ROOT_NODE_ID) == ROOT_NODE_ID:
        return dict(root)
    out: dict[int, Run] = {}
    for r in sorted((r for r in parent_runs if r.stage in _REUSE), key=lambda r: _REUSE.index(r.stage)):
        out.setdefault(r.seed, r)
    return out


Gauss = tuple[float, float]          # (mean, var)


@dataclass(frozen=True)
class Obs:
    stage: Stage
    control: str                                # the control run's id, shared by its other children
    delta: Mapping[str, float]                  # target -> raw paired delta
    resumed: bool = False                       # either run resumed from latest.pt: not an exact replicate


@dataclass(frozen=True)
class NodeData:
    node: Node
    obs: Sequence[Obs]                          # raw, by seed (a confirming node's confirm runs too)
    all_seeds: Mapping[str, Sequence[float]]    # target -> raw delta per any paired seed (noise)
    first_submitted_ts: float
    selection: Mapping[str, Sequence[Gauss]] = field(default_factory=dict)   # target -> corrected
    confirm: Sequence[Gauss] = ()               # corrected panel delta per confirm seed, by seed

    def n(self, target: str) -> int:
        return len(self.selection.get(target, ()))

    def est(self, target: str) -> Gauss:
        """Precision-weighted mean of the corrected selection deltas and its variance."""
        return pool(self.selection[target])


def resume_var(ctx: LoopContext, sigma: Mapping[str, float], t: str) -> float:
    """Extra variance of a paired delta whose run (or control) RESUMED from latest.pt (loop.resume): train.py
    restarts the loader permutation and reseeds the RNG, so it is no longer its seed's exact replicate and the
    pairing no longer cancels the seed component. Seed-level: ``noise.seed_transfer_sd`` on the panel (an axis:
    scaled by sigma_t / sigma_panel), else the target's own sigma."""
    s = getattr(getattr(ctx, "noise", None), "seed_transfer_sd", None)
    if isinstance(s, (int, float)) and not isinstance(s, bool) and s > 0 and sigma.get(PANEL):
        return (float(s) * sigma[t] / sigma[PANEL]) ** 2
    return sigma[t] ** 2


def pool(xs: Sequence[Gauss]) -> Gauss:
    w = sum(1.0 / v for _, v in xs)
    return sum(x / v for x, v in xs) / w, 1.0 / w


def shared_offset(sibs: Sequence[Gauss], sigma: float) -> Gauss:
    """Posterior of a control run's deviation ``b ~ N(0, sigma^2/2)`` from its other children's
    residuals ``y_j ~ N(b, sigma^2/2 + v_j)``; no siblings -> the prior (no free lunch)."""
    half = sigma * sigma / 2.0
    p = 1.0 / half + sum(1.0 / (half + v) for _, v in sibs)
    return sum(y / (half + v) for y, v in sibs) / p, 1.0 / p


@dataclass(frozen=True)
class Ledger:
    nodes: Mapping[str, Node]
    data: Mapping[str, NodeData]                # scored non-root nodes
    measure: Mapping[str, str] = field(default_factory=dict)   # node -> its measure parent (when not its parent)
    offset: Mapping[str, Mapping[str, Gauss]] = field(default_factory=dict)   # node -> target -> U_m - U_parent
    twins: Mapping[str, frozenset[str]] = field(default_factory=dict)   # node -> near-dup nodes on a shared control


def parent_of(ledger: Ledger, nid: str) -> str:
    """The node a node's OWN increment is over: its measure parent (a re-rooted crossover / copy that contains a
    deeper ancestor's whole change is measured against that ancestor, never credited with its gain), else its
    declared parent (root when none)."""
    return ledger.measure.get(nid) or (ledger.nodes[nid].parent_id if nid in ledger.nodes else None) or ROOT_NODE_ID


AUDIT_KEY = "driver.audit"               # loop.labless.K_AUDIT: {passed: [...], ...}


def measure_parents(store: Any, nodes: Mapping[str, Node]) -> dict[str, str]:
    """node -> the first ESTABLISHED entry of its MEASURE_KEY list (a CONFIRMED node -- the incumbent among them --
    or one that passed the held-out audit: its gain is already credited), when that is not its declared parent. A
    composition of unconfirmed legs keeps its declared parent: nothing it carries was credited before."""
    out, passed = {}, set((store.get_state(AUDIT_KEY) or {}).get("passed") or ())
    for nid, cands in (store.get_state(MEASURE_KEY) or {}).items():
        node = nodes.get(nid)
        m = next((c for c in cands or () if c in nodes and c != nid
                  and (nodes[c].status is NodeStatus.CONFIRMED or c in passed)), None) if node is not None else None
        if m is not None and m != (node.parent_id or ROOT_NODE_ID):
            out[nid] = m
    return out


def rebase(ledger: Ledger, offset: Mapping[str, Mapping[str, Gauss]]) -> Ledger:
    """Each re-measured node's corrected deltas (vs its paired control, its declared parent's runs) minus
    ``U_measure - U_parent``, its variance added: the node's increment over its measure parent."""
    data = {}
    for nid, nd in ledger.data.items():
        off = offset.get(nid)
        if not off:
            data[nid] = nd
            continue
        sel = {t: [(x - off[t][0], v + off[t][1]) for x, v in xs] if t in off else list(xs)
               for t, xs in nd.selection.items()}
        o = off.get(PANEL, (0.0, 0.0))
        data[nid] = replace(nd, selection=sel, confirm=[(x - o[0], v + o[1]) for x, v in nd.confirm])
    return Ledger(ledger.nodes, data, ledger.measure, dict(offset), ledger.twins)


def read_ledger(ctx: LoopContext, store: Any) -> Ledger:
    nodes = {n.id: n for n in store.nodes()}
    first: dict[str, float] = {}
    meas: dict[str, list[Run]] = {}
    for r in store.runs():
        if r.submitted_ts is not None:
            first[r.node_id] = min(first.get(r.node_id, math.inf), r.submitted_ts)
        if r.status is RunStatus.DONE and r.scores and r.stage not in (Stage.AUDIT, Stage.TIEBREAK, Stage.REMINT):
            meas.setdefault(r.node_id, []).append(r)
    root: dict[int, Run] = {}
    for r in meas.get(ROOT_NODE_ID, []):
        if r.stage in (Stage.BASELINE, Stage.CALIBRATE) and (
                r.seed not in root or r.stage is Stage.BASELINE):
            root[r.seed] = r
    data: dict[str, NodeData] = {}
    for nid, runs in meas.items():
        node = nodes.get(nid)
        if nid == ROOT_NODE_ID or node is None or node.status in _DEAD:
            continue
        ctl = controls(node, root, meas.get(node.parent_id or ROOT_NODE_ID, []))
        obs: list[Obs] = []
        every: dict[str, list[float]] = {}
        for r in sorted(runs, key=lambda r: (r.seed, r.stage.value)):
            if r.stage is Stage.PARENT or r.seed not in ctl:
                continue
            obs.append(Obs(r.stage, ctl[r.seed].id, target_values(ctx, r.scores, ctl[r.seed].scores),
                           bool(r.resumed or ctl[r.seed].resumed)))
            if r.stage is Stage.CONFIRM and node.status is NodeStatus.CONFIRMING or obs[-1].resumed:
                continue                # never feeds the sigma its own confirmation uses (D3.7); nor a resumed run
            for t, x in obs[-1].delta.items():
                every.setdefault(t, []).append(x)
        # a CONFIRM-ONLY node (a campaign / tune graduate, loop.campaign: no screen / promote seed of its own) is
        # scored too: its confirm increments are its whole verdict (live 2026-10-08: a node ran 3 confirm seeds,
        # each paired vs a DONE root run, and closed "seeds spent E=1 n=0" -- it was never in the ledger)
        if any(o.stage in (*SELECTION_STAGES, Stage.CONFIRM) and PANEL in o.delta for o in obs):
            data[nid] = NodeData(node, obs, every, first.get(nid, math.inf))
    return Ledger(nodes, data, measure_parents(store, nodes), twins=twins(ctx, store, nodes, data))


def twins(ctx: LoopContext, store: Any, nodes: Mapping[str, Node], data: Mapping[str, NodeData]) -> dict[str, frozenset[str]]:
    """node -> the near-duplicates among the nodes sharing one of its control runs: the same normalized tag
    (loop.novelty.norm_tag) or a mechanism cosine >= ``novelty.mech_dup_cos``. A near-dup's delta carries the
    same effect, not the control's luck: a node (slide-balanced sampling) read sibling a node's (the same idea)
    shared control as lucky and shrank its gain to ~0."""
    from loop.novelty import mech_vectors, norm_tag
    by_ctl: dict[str, set[str]] = {}
    for nid, nd in data.items():
        for o in nd.obs:
            by_ctl.setdefault(o.control, set()).add(nid)
    thr, vec, out = ctx.cfg.novelty.mech_dup_cos, None, {}
    seen: set[tuple[str, str]] = set()
    for ids in by_ctl.values():
        for a in ids:
            for b in ids:
                if a >= b or (a, b) in seen:
                    continue
                seen.add((a, b))
                ta, tb = norm_tag(nodes[a].mechanism_tag), norm_tag(nodes[b].mechanism_tag)
                dup = bool(ta) and ta == tb
                if not dup and thr > 0:
                    vec = mech_vectors(store) if vec is None else vec
                    va, vb = vec.get(a, {}), vec.get(b, {})
                    dup = sum(x * vb.get(w, 0.0) for w, x in va.items()) >= thr
                if dup:
                    out.setdefault(a, set()).add(b)
                    out.setdefault(b, set()).add(a)
    return {k: frozenset(v) for k, v in out.items()}


def offset_ok(ledger: Ledger, nid: str, j: str, y: float, sigma: float) -> bool:
    """Whether sibling ``j``'s raw delta ``y`` on a control ``nid`` shares may estimate that control's offset
    for ``nid``: never its own, a harm-VETOED node's, a near-duplicate's (:func:`twins`), nor one beyond
    ``OUTLIER_SIGMAS * sigma`` (a collapsed / broken run). Live: a node's -0.066 on control 412 read a node's raw
    +0.0014 as +0.013; vetoed a node's -0.015 / -0.019 lifted a node likewise."""
    n = ledger.nodes.get(j)
    return j != nid and abs(y) <= OUTLIER_SIGMAS * sigma and j not in ledger.twins.get(nid, ()) \
        and (n is None or n.status is not NodeStatus.VETOED)


def offset_rise(ctx: LoopContext, ledger: Ledger, sigma: Mapping[str, float]) -> dict[str, float]:
    """node -> how much :func:`offset_ok` RAISED its panel control-offset estimates (summed over its selection
    deltas, so lowered its corrected deltas) vs the rule before it (every other sibling counts); for the
    one-time re-gate (loop.driver._regate) of nodes that passed a gate on a broken / vetoed / twin sibling."""
    by_ctl, out = sibling_groups(ledger), {}
    tau2 = sibling_tau(ctx, by_ctl, sigma)[PANEL] ** 2
    for nid, nd in ledger.data.items():
        for o in nd.obs:
            if o.stage not in SELECTION_STAGES or PANEL not in o.delta:
                continue
            ys = by_ctl.get((o.control, PANEL), ())
            new = shared_offset([(y, tau2) for j, y in ys if offset_ok(ledger, nid, j, y, sigma[PANEL])], sigma[PANEL])
            old = shared_offset([(y, tau2) for j, y in ys if j != nid], sigma[PANEL])
            out[nid] = out.get(nid, 0.0) + new[0] - old[0]
    return out


def sibling_groups(ledger: Ledger) -> dict[tuple[str, str], list[tuple[str, float]]]:
    """(control run id, target) -> [(node id, raw paired delta)] of every obs paired against it."""
    by_ctl: dict[tuple[str, str], list[tuple[str, float]]] = {}
    for nid, nd in ledger.data.items():
        for o in nd.obs:
            for t, x in o.delta.items() if not o.resumed else ():     # a resumed delta says little about b_c
                by_ctl.setdefault((o.control, t), []).append((nid, x))
    return by_ctl


def sibling_tau(ctx: LoopContext, groups: Mapping[tuple[str, str], Sequence[tuple[str, float]]],
                sigma: Mapping[str, float]) -> dict[str, float]:
    """target -> sd of siblings' own effects: the pooled spread of deltas that share a control run
    minus their own-run variance sigma^2/2 (tau0 until ``prior.noise_min_df`` df exist)."""
    p, ss, df = ctx.cfg.prior, {}, {}
    for (_, t), ys in groups.items():
        if len(ys) >= 2:
            m = sum(y for _, y in ys) / len(ys)
            ss[t] = ss.get(t, 0.0) + sum((y - m) ** 2 for _, y in ys)
            df[t] = df.get(t, 0) + len(ys) - 1
    return {t: math.sqrt(max(p.tau_floor ** 2, ss[t] / df[t] - s * s / 2.0))
            if df.get(t, 0) >= p.noise_min_df else p.tau0 for t, s in sigma.items()}


def correct(ctx: LoopContext, ledger: Ledger, sigma: Mapping[str, float]) -> Ledger:
    """Fill each node's corrected ``selection`` / ``confirm``: ``d - b_hat_c(-i)`` with variance
    ``sigma^2/2 + Var(b_hat_c(-i))``. A sibling's delta is ``b_c`` plus its own effect, taken as
    ``N(0, tau^2)`` (:func:`sibling_tau`); a declared family prior is NOT used as its mean (the
    architecture -0.02 would read three flat architecture siblings as a +0.02 control)."""
    by_ctl = sibling_groups(ledger)
    tau = sibling_tau(ctx, by_ctl, sigma)
    out = {}
    for nid, nd in ledger.data.items():
        sel: dict[str, list[Gauss]] = {}
        conf: list[Gauss] = []
        for o in nd.obs:
            for t, x in o.delta.items():
                sibs = [(y, tau[t] ** 2) for j, y in by_ctl.get((o.control, t), ())
                        if offset_ok(ledger, nid, j, y, sigma[t])]
                b, vb = shared_offset(sibs, sigma[t])
                g = (x - b, sigma[t] ** 2 / 2.0 + vb + (resume_var(ctx, sigma, t) if o.resumed else 0.0))
                if o.stage in SELECTION_STAGES:
                    sel.setdefault(t, []).append(g)
                elif o.stage is Stage.CONFIRM and t == PANEL:
                    conf.append(g)
        out[nid] = replace(nd, selection=sel, confirm=conf)
    return Ledger(ledger.nodes, out, ledger.measure, ledger.offset, ledger.twins)


def fit_noise(ctx: LoopContext, store: Any, ledger: Ledger) -> dict[str, float]:
    """Pooled within-node variance blended with the calibration prior (D3.2).

    ``sigma`` stays the declared value until ``prior.noise_min_df`` within-node df
    exist; ``df_eff`` is the df behind the sigma in force (its upper bound, D3.7).
    Cached in store state; returns target -> sigma.
    """
    df0 = max(1, int((store.get_state(CAL_KEY) or {}).get("df", len(ctx.cfg.seeds.calibration_set) - 1)))
    out: dict[str, dict[str, float]] = {}
    for t in targets(ctx):
        s0, ss, df = float(ctx.cfg.sigma(t)), 0.0, 0
        for nd in ledger.data.values():
            xs = nd.all_seeds.get(t, ())
            if len(xs) >= 2:
                m = sum(xs) / len(xs)
                ss += sum((x - m) ** 2 for x in xs)
                df += len(xs) - 1
        sigma, df_eff = s0, df0
        if df >= ctx.cfg.prior.noise_min_df:
            sigma, df_eff = math.sqrt((df0 * s0 * s0 + ss) / (df0 + df)), df0 + df
        out[t] = {"sigma": sigma, "sigma0": s0, "df": float(df), "df_eff": float(df_eff)}
    if store.get_state(STATE_KEY) != out:
        store.set_state(STATE_KEY, out)
    return {t: row["sigma"] for t, row in out.items()}
