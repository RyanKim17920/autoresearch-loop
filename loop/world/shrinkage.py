"""Hierarchical normal-normal shrinkage of increments (DESIGN §3.3-3.4): node -> family -> prior, exact
Gaussian belief propagation; ``U_i = U_parent + eps_i``. (There is no tag level: pooling nodes of one mechanism
tag had no out-of-time skill, loop.world.predictive.) A variance of 0 is a
point mass and ``inf`` is no message; :func:`combine` handles both."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from loop.contracts import ROOT_NODE_ID, Posterior, VarianceComponents
from loop.spec import LoopContext
from loop.world.noise import Ledger, parent_of

INF = math.inf
Gauss = tuple[float, float]          # (mean, var)
FLAT: Gauss = (0.0, INF)


def combine(a: Gauss, b: Gauss) -> Gauss:
    """Product of two Gaussian densities."""
    (ma, va), (mb, vb) = a, b
    if va == 0.0 or vb == INF:
        return a
    if vb == 0.0 or va == INF:
        return b
    p = 1.0 / va + 1.0 / vb
    return ((ma / va + mb / vb) / p, 1.0 / p)


def combine_all(items: Iterable[Gauss], start: Gauss = FLAT) -> Gauss:
    for g in items:
        start = combine(start, g)
    return start


def widen(g: Gauss, extra_var: float) -> Gauss:
    return (g[0], g[1] + extra_var)


@dataclass(frozen=True)
class TreeFit:
    eps: Mapping[str, Gauss]              # node -> posterior of eps_i
    eps_prior: Mapping[str, Gauss]        # node -> cavity prior of eps_i (without its own e_i)
    fam_cavity: Mapping[str, Gauss]       # node -> its family's mean without the node
    families: Mapping[str, Gauss]


def tree_bp(obs: Mapping[str, tuple[str, float, float]],
            family_priors: Mapping[str, Gauss], tau1: float) -> TreeFit:
    """``obs``: node -> (family, e_i, v_i); ``family_priors``: family -> (m_f, tau3_f^2). A node's own prior
    (``eps_prior``) is its family's mean WITHOUT the node's own message (prefix / suffix products)."""
    t1 = tau1 * tau1
    by_family: dict[str, list[str]] = {f: [] for f in family_priors}
    for nid, (f, _, _) in obs.items():
        by_family[f].append(nid)
    msg = {nid: (e, v + t1) for nid, (_, e, v) in obs.items()}
    families, eps, eps_prior, fam_cavity = {}, {}, {}, {}
    for f, nids in by_family.items():
        pre = [family_priors[f]]
        for nid in nids:
            pre.append(combine(pre[-1], msg[nid]))
        families[f] = pre[-1]
        suf = FLAT
        for i in range(len(nids) - 1, -1, -1):
            nid = nids[i]
            cavity = combine(pre[i], suf)
            suf = combine(suf, msg[nid])
            fam_cavity[nid], eps_prior[nid] = cavity, widen(cavity, t1)
            eps[nid] = combine(eps_prior[nid], obs[nid][1:])
    return TreeFit(eps, eps_prior, fam_cavity, families)


def dersimonian_laird(groups: Iterable[Iterable[Gauss]], floor: float, cap: float) -> float | None:
    """Pooled DL tau; ``None`` when no group has >= 2 finite-variance members."""
    num = den = 0.0
    seen = False
    for members in groups:
        xs = [(x, v) for x, v in members if 0.0 < v < INF]
        if len(xs) < 2:
            continue
        seen = True
        w = [1.0 / v for _, v in xs]
        sw = sum(w)
        xbar = sum(wi * x for wi, (x, _) in zip(w, xs)) / sw
        num += sum(wi * (x - xbar) ** 2 for wi, (x, _) in zip(w, xs)) - (len(xs) - 1)
        den += sw - sum(wi * wi for wi in w) / sw
    if not seen or den <= 0.0:
        return None
    return min(cap, max(floor, math.sqrt(max(floor * floor, num / den))))


def refit_components(obs: Mapping[str, tuple[str, float, float]], ctx: LoopContext,
                     n_scored: int, current: VarianceComponents) -> VarianceComponents:
    """DL per level. A level DL cannot estimate keeps its current tau (tau0 when cold),
    never the floor, which would silently switch shrinkage to maximal. ``tau_tag`` is fixed 0 (no tag level)."""
    p = ctx.cfg.prior

    def dl(groups: Iterable, cur: float) -> float:
        tau = dersimonian_laird(groups, p.tau_floor, p.tau_max)
        return tau if tau is not None else (p.tau0 if current.cold else cur)

    by_family: dict[str, list[Gauss]] = {}
    for f, e, v in obs.values():
        by_family.setdefault(f, []).append((e, v))
    tau1 = dl(by_family.values(), current.tau_node)
    fam_est = [combine_all((e, v + tau1 * tau1) for e, v in xs)
               for f, xs in by_family.items() if f not in p.families]
    return VarianceComponents(tau_node=tau1, tau_tag=0.0, tau_family=dl([fam_est], current.tau_family),
                              n_scored_at_fit=n_scored, cold=False)


def cold_components(ctx: LoopContext) -> VarianceComponents:
    return VarianceComponents(tau_node=ctx.cfg.prior.tau0, tau_tag=0.0, tau_family=0.0,
                              n_scored_at_fit=0, cold=True)


def family_prior(ctx: LoopContext, family: str, comps: VarianceComponents) -> Gauss:
    """Declared prior (``loop.prior.families``) or N(0, tau_family^2)."""
    fp = ctx.cfg.prior.families.get(family)
    return (float(fp.mean), float(fp.sd) ** 2) if fp else (0.0, comps.tau_family ** 2)


def predict(ctx: LoopContext, comps: VarianceComponents, family_means: Mapping[str, Posterior],
            family: str) -> Gauss:
    """Increment predictive for an unrun node of ``family``: the family mean (else its prior) widened by tau_node."""
    f = family_means.get(family)
    base = (f.mean, f.var) if f is not None else family_prior(ctx, family, comps)
    return widen(base, comps.tau_node ** 2)


def posterior(g: Gauss, n: int = 0) -> Posterior:
    return Posterior(mean=g[0], sd=math.sqrt(max(0.0, g[1])), n=n)


@dataclass
class TargetFit:
    target: str
    sigma: float
    comps: VarianceComponents
    tree: TreeFit
    obs: dict[str, tuple[str, float, float]]
    utility: dict[str, Gauss]
    family_means: dict[str, Posterior]


def fit_target(ctx: LoopContext, store: Any, target: str, ledger: Ledger,
               sigma: float) -> TargetFit:
    """Shrink one target's increments, refitting variance components on schedule."""
    p, key = ctx.cfg.prior, f"world.variance.{target}"
    scored = {nid: nd for nid, nd in ledger.data.items() if nd.n(target) > 0}
    obs = {nid: (nd.node.family, *nd.est(target)) for nid, nd in scored.items()}
    stored = store.get_state(key)
    comps = VarianceComponents(**stored) if stored else cold_components(ctx)
    if len(obs) < p.refit_min:
        comps = cold_components(ctx)
    elif comps.cold or len(obs) - comps.n_scored_at_fit >= p.refit_every:
        comps = refit_components(obs, ctx, len(obs), comps)
    comps = dataclasses.replace(comps, tau_tag=0.0)        # a state written before the tag level was removed
    if stored != (now := comps.__dict__):
        store.set_state(key, now)
    priors = {f: family_prior(ctx, f, comps) for f in {o[0] for o in obs.values()}}
    tree = tree_bp(obs, priors, comps.tau_node)
    fit = TargetFit(target, sigma, comps, tree, obs, {}, {f: posterior(x) for f, x in tree.families.items()})
    for nid in sorted(scored, key=lambda n: (ledger.nodes[n].created_ts, n)):   # parents first
        up = parent_utility(ctx, fit, ledger, parent_of(ledger, nid))
        fit.utility[nid] = (up[0] + tree.eps[nid][0], up[1] + tree.eps[nid][1])
    return fit


def parent_utility(ctx: LoopContext, fit: TargetFit, ledger: Ledger, nid: str,
                   depth: int = 0) -> Gauss:
    """U of ``nid``: 0 for root, its posterior if scored, else its parent's U plus the family predictive."""
    if nid == ROOT_NODE_ID or depth > 64 or nid not in ledger.nodes:
        return (0.0, 0.0)
    if nid in fit.utility:
        return fit.utility[nid]
    node = ledger.nodes[nid]
    up = parent_utility(ctx, fit, ledger, parent_of(ledger, nid), depth + 1)
    m, v = predict(ctx, fit.comps, fit.family_means, node.family)
    return (up[0] + m, up[1] + v)
