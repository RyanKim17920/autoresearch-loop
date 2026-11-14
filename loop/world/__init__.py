"""loop.world — the COMPUTED world model (DESIGN §3). ``fit(store)`` -> WorldSnapshot: noise,
shrinkage per target, utility, veto, incumbent, lever trends, hypothesis e-values. No LLM writes here."""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping

from loop.contracts import (PANEL, ROOT_NODE_ID, LeverTags, NodePosterior, NodeStatus, Posterior, RunStatus,
                            Stage, WorldSnapshot)
from loop.spec import LoopContext
from loop.world import evalues, mixture, noise, predictive, shrinkage
from loop.world.evalues import LinkIndex, link_meta, phi
from loop.world.shrinkage import Gauss, TargetFit, combine, parent_utility, posterior

K_REFROZEN = "world.refreeze_confirmed"   # nodes the driver's one-time refreeze confirmed (rule applied after the data)
K_INC = "world.incumbent"                 # the last fit's incumbent (a twin swap needs the noise rule, below)
K_AUDIT = "driver.audit"                  # loop.labless.K_AUDIT: {passed, failed, queue, current, gate}
K_REMINT = "driver.remint"                # loop.remint.K_REMINT: {"rounds": [{inc, nodes, seeds, closed?, ...}]}

__all__ = ["ComputedWorldModel", "LinkIndex", "StateOverlay", "banked", "fresh_paired", "held_back", "link_meta", "scored_seeds", "veto_sigmas"]


class StateOverlay:
    """A store view whose state writes and log lines stay in memory (``status``, ``--dry-run``)."""

    def __init__(self, store: Any) -> None:
        self._store, self._state = store, {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)

    def get_state(self, key: str, default: Any = None) -> Any:
        return self._state[key] if key in self._state else self._store.get_state(key, default)

    def set_state(self, key: str, value: Any) -> None:
        self._state[key] = value

    def log(self, *_: Any, **__: Any) -> None:
        pass


def held_back(store: Any) -> set[str]:
    """Refreeze-confirmed nodes not yet past the held-out audit: never the incumbent, a stack head or a carrier."""
    passed = set((store.get_state(K_AUDIT) or {}).get("passed", []))
    return {n for n in store.get_state(K_REFROZEN) or [] if n not in passed}


def audited_estimate(store: Any, nid: str, conf: Mapping[str, Posterior]) -> Gauss:
    """(mean, var): the held-out audit's frozen numbers when the node passed one, else its confirm-only estimate."""
    g = ((store.get_state(K_AUDIT) or {}).get("gate") or {}).get(nid) or {}
    if g.get("verdict") == "passed" and g.get("mean") is not None and g.get("se") is not None:
        return float(g["mean"]), float(g["se"]) ** 2
    return conf[nid].mean, conf[nid].var


def audit_paired(store: Any, a: str, b: str) -> Gauss | None:
    """(mean, variance of the mean) of ``a - b`` on the held-out audit seeds both frozen gates share: each seed's
    panel delta vs root with its offset added back (the shared root run cancels); None under 2 shared seeds."""
    gate = (store.get_state(K_AUDIT) or {}).get("gate") or {}
    raw = lambda g: {s: x + float((g.get("offsets") or {}).get(s, 0.0))     # noqa: E731
                     for s, x in (g.get("deltas") or {}).items()}
    ra, rb = raw(gate.get(a) or {}), raw(gate.get(b) or {})
    ds = [ra[s] - rb[s] for s in ra if s in rb]
    if len(ds) < 2:
        return None
    m = sum(ds) / len(ds)
    return m, sum((d - m) ** 2 for d in ds) / (len(ds) - 1) / len(ds)


#: stages whose runs are a node's FRESH (verdict) seeds: never a selection (screen / promote) seed
FRESH_STAGES = frozenset({Stage.CONFIRM, Stage.PARENT, Stage.AUDIT, Stage.TIEBREAK, Stage.REMINT})


def fresh_paired(ctx: LoopContext, store: Any, a: str, b: str, strict: bool = False) -> Gauss | None:
    """(mean, variance of the mean) of ``a - b`` paired run against run over EVERY fresh seed both ran (confirm /
    parent / audit stages, :data:`FRESH_STAGES`; a seed either node SELECTED on -- screen / promote -- never counts);
    None under 2 shared seeds. The audit seeds alone are 4: a node vs a node had 7 fresh shared seeds. ``strict``: a
    seed either node CONFIRMED on is a selection seed too (it decided CONFIRMED: the re-mint, loop.remint)."""
    pick = (Stage.SCREEN, Stage.PROMOTE, Stage.CONFIRM) if strict else (Stage.SCREEN, Stage.PROMOTE)

    def panel(nid: str) -> tuple[dict[int, float], set[int]]:
        ys: dict[int, list[float]] = {}
        sel: set[int] = set()
        for r in store.runs(node_id=nid):
            if r.stage in pick:
                sel.add(int(r.seed))
            elif r.stage in FRESH_STAGES and r.status is RunStatus.DONE and all(x in r.scores for x in ctx.cfg.axes):
                ys.setdefault(int(r.seed), []).append(sum(ctx.cfg.weights[x] * r.scores[x] for x in ctx.cfg.axes))
        return {s: sum(v) / len(v) for s, v in ys.items()}, sel
    (ya, sa), (yb, sb) = panel(a), panel(b)
    ds = [ya[s] - yb[s] for s in ya if s in yb and s not in sa | sb]
    if len(ds) < 2:
        return None
    m = sum(ds) / len(ds)
    return m, sum((d - m) ** 2 for d in ds) / (len(ds) - 1) / len(ds)


def reminted(store: Any, a: str, b: str) -> bool:
    """Both nodes ran in a closed re-mint round (loop.remint): their comparison then counts no seed either was
    selected on (screen / promote / confirm), and so always includes that newest never-used block."""
    return any(r.get("closed") and a in r["nodes"] and b in r["nodes"]
               for r in (store.get_state(K_REMINT) or {}).get("rounds") or [])


def prob_greater(g: Gauss, threshold: float = 0.0) -> float:
    """P(X > threshold) for X ~ N(g); a point mass gives 0/1 (ties -> 0)."""
    m, v = g
    return (1.0 if m > threshold else 0.0) if v <= 0.0 else phi((m - threshold) / v ** 0.5)


def veto_sigmas(ctx: LoopContext, u: Gauss) -> float:
    """How far (axis sigmas) an axis must drop to veto: the benchmark is the panel average, so a node
    whose panel mean is positive is vetoed only for a catastrophic single-axis drop."""
    return ctx.cfg.veto.catastrophic_sigmas if u[0] > 0.0 else ctx.cfg.veto.harm_sigmas


def scored_seeds(store: Any, node_id: str) -> int:
    """The node's own scored lifecycle runs (screen / promote / confirm; never the held-out audit)."""
    return sum(r.status is RunStatus.DONE and bool(r.scores) for r in store.runs(node_id=node_id)
               if r.stage in (Stage.SCREEN, Stage.PROMOTE, Stage.CONFIRM))


def banked(ctx: LoopContext, store: Any, nodes: Mapping[str, NodePosterior],
           own_confirm: Mapping[str, Posterior]) -> tuple[str, ...]:
    """Confirm-and-condense: nodes that join the CONFIRMED winners in the winner-cross stack pool -- not
    CONFIRMED, but >= promotion.bank_min_seeds scored seeds (they reached fresh-seed confirmation), never
    vetoed, with P(own panel increment over the parent > 0) >= promotion.bank_p on its CONFIRM seeds alone
    (``own_confirm``, >= promotion.bank_min_confirm of them: the selection seeds' p_gain banked a node with confirm
    seeds -0.0020 / -0.0051) and a positive confirm-only U. Stack-building only: never graduates or the incumbent."""
    pr = ctx.cfg.promotion
    out = []
    for n in store.nodes(status=[NodeStatus.NOT_CONFIRMED, NodeStatus.CONFIRMING]) if pr.bank_min_seeds > 0 else []:
        p, own = nodes.get(n.id), own_confirm.get(n.id)
        if p is not None and own is not None and not p.vetoed and own.n >= pr.bank_min_confirm \
                and prob_greater((own.mean, own.var)) >= pr.bank_p \
                and (p.confirm_estimate or p.utility).mean > 0.0 and scored_seeds(store, n.id) >= pr.bank_min_seeds:
            out.append(n.id)
    return tuple(sorted(out))


def fresh_vs_root(ctx: LoopContext, store: Any, ledger: noise.Ledger, sigma: Mapping[str, float]) -> dict[str, list[Gauss]]:
    """node (with confirm seeds) -> its panel delta vs root's run at each FRESH seed (:data:`FRESH_STAGES`) the ledger
    does not already pair (no confirm / screen / promote run of the node at that seed) and root ran (baseline /
    calibrate / held-out audit); variance the larger of sigma^2 and the node's own spread of them (3+; these pairs
    span days: a node sd 0.0053 vs sigma 0.0033), plus the resume variance when either run resumed. The parent /
    audit / tiebreak runs of a carrier (a node: 16 such seeds next to 3 confirm seeds) are fresh evidence of its U."""
    want = {nid for nid, nd in ledger.data.items() if nd.confirm}
    root: dict[int, Any] = {}
    by: dict[str, list[Any]] = {}
    for r in store.runs():
        if r.status is not RunStatus.DONE or not r.scores:
            continue
        if r.node_id == ROOT_NODE_ID and r.stage in (Stage.BASELINE, Stage.CALIBRATE, Stage.AUDIT):
            if r.seed not in root or r.stage is Stage.BASELINE:
                root[r.seed] = r
        elif r.node_id in want:
            by.setdefault(r.node_id, []).append(r)
    s2, rv = sigma[PANEL] ** 2, noise.resume_var(ctx, sigma, PANEL)
    out: dict[str, list[Gauss]] = {}
    for nid, runs in by.items():
        paired = {r.seed for r in runs if r.stage not in FRESH_STAGES or r.stage is Stage.CONFIRM}
        seen: set[int] = set()
        for r in runs:
            if r.stage in FRESH_STAGES and r.seed in root and r.seed not in paired | seen:
                d = noise.target_values(ctx, r.scores, root[r.seed].scores).get(PANEL)
                if d is not None:
                    seen.add(r.seed)
                    out.setdefault(nid, []).append((d, rv if r.resumed or root[r.seed].resumed else 0.0))
        if xs := out.get(nid):
            m = sum(x for x, _ in xs) / len(xs)
            v = max(s2, sum((x - m) ** 2 for x, _ in xs) / (len(xs) - 1)) if len(xs) >= 3 else s2
            out[nid] = [(x, v + e) for x, e in xs]
    return out


def _confirm_estimates(ctx: LoopContext, ledger: noise.Ledger, fit: TargetFit, fresh: Mapping[str, list[Gauss]] | None = None
                       ) -> tuple[dict[str, Posterior], dict[str, Posterior]]:
    """(U, own eps) from fresh seeds alone (winner's curse, D3.6): eps shrunk by its cavity prior from the confirm
    increments, on top of the parent's fresh-seed U (root: 0); then every other fresh seed the node ran against a
    root run (``fresh``, :func:`fresh_vs_root`) -- a direct U measurement, so the prior's weight shrinks with all
    the node's fresh seeds, never 1..3 confirm seeds alone (2026-10-06: a node read +0.0049 on its one confirm seed
    beside 15 root-paired fresh seeds at +0.0090). ``own`` stays confirm-only (banking)."""
    out: dict[str, Posterior] = {ROOT_NODE_ID: Posterior(0.0, 0.0)}
    own: dict[str, Posterior] = {}
    fresh = fresh or {}
    for nid in sorted(ledger.data, key=lambda n: (ledger.nodes[n].created_ts, n)):
        cs = ledger.data[nid].confirm
        if not cs:
            continue
        pid = ledger.nodes[nid].parent_id or ROOT_NODE_ID
        up = out.get(pid) or posterior(fit.utility.get(pid, (0.0, 0.0)))
        prior = fit.tree.eps_prior.get(nid)       # a confirm-only node (campaign graduate) is not in the tree:
        prior = prior or shrinkage.predict(ctx, fit.comps, fit.family_means, ledger.nodes[nid].family)  # its family's
        eps = combine(prior, noise.pool(cs))
        u, xs = (up.mean + eps[0], up.var + eps[1]), fresh.get(nid) or []
        out[nid], own[nid] = posterior(combine(u, noise.pool(xs)) if xs else u, len(cs) + len(xs)), posterior(eps, len(cs))
    return out, own


def beats_gauss(p: NodePosterior, parent_id: str, inc: str, conf: Mapping[str, Posterior],
                own_confirm: Posterior | None, parent_u: Posterior) -> Gauss:
    """U_i - U_inc without the inherited-carrier bias: the parent's confirm-only estimate (its U when it has
    no confirm seeds) + the node's OWN paired increment (confirm-only when it has confirm seeds) - the
    incumbent's confirm-only estimate. A child of the incumbent: just its own increment (the shared term
    cancels, mean and variance), never its full-seed U against the incumbent's confirm-only U."""
    own = own_confirm or p.increment
    if parent_id == inc:
        return own.mean, own.var
    base, ci = conf.get(parent_id) or parent_u, conf[inc]
    return base.mean + own.mean - ci.mean, base.var + own.var + ci.var


def build_posteriors(ctx: LoopContext, ledger: noise.Ledger, fits: Mapping[str, TargetFit],
                     sigma: Mapping[str, float], own: Mapping[str, Gauss] | None = None,
                     blocked: frozenset[str] | set[str] = frozenset(), hold: Any = None,
                     mix: Mapping[str, tuple[tuple[eb.Comp, ...], tuple[eb.Comp, ...], float]] | None = None,
                     fresh: Mapping[str, list[Gauss]] | None = None) -> tuple[dict[str, NodePosterior], str]:
    """Posteriors, the incumbent, and P(U > U_inc). The incumbent is the CONFIRMED,
    non-vetoed node with the best confirm-only estimate; P(U > U_inc) is :func:`beats_gauss`
    (confirm-only on both sides of the shared carrier, D3.6). ``own``: a re-measured node's increment over its
    measure parent (what it is credited with); its U, confirm estimate and P(U>U_inc) are what it scores.
    ``blocked`` nodes are never the incumbent; ``hold(inc, conf, eligible)`` may keep the previous one. ``mix``: node ->
    (EB posterior components of its increment over its declared parent, of its own increment, the prior P(own gain
    > 0) of its generation) (loop.world.mixture; the
    panel fit's eps are then their moments): P(U>0), P(gain>0) and a selection-seed P(U>U_inc) are exact under it."""
    mix = mix or {}
    panel, (conf, own_conf) = fits[PANEL], _confirm_estimates(ctx, ledger, fits[PANEL], fresh)
    own = own or {}
    zero = Posterior(0.0, 0.0)
    nodes = {ROOT_NODE_ID: NodePosterior(ROOT_NODE_ID, zero, zero, {a: zero for a in ctx.cfg.axes},
                                         0.0, 0.0, False, (), zero)}
    for nid, u in panel.utility.items():
        n = ledger.data[nid].n
        axes = {a: fits[a].utility[nid] for a in ctx.cfg.axes if nid in fits[a].utility}
        veto = tuple(a for a, g in axes.items() if 1.0 - prob_greater(g, -veto_sigmas(ctx, u) * sigma[a])
                     >= ctx.cfg.veto.prob)
        m, e = mix.get(nid), panel.tree.eps[nid]
        pu = prob_greater(u) if m is None else mixture.prob_greater(m[0], (u[0] - e[0], max(0.0, u[1] - e[1])))
        nodes[nid] = NodePosterior(nid, posterior(own.get(nid, e), n(PANEL)),
                                   posterior(u, n(PANEL)), {a: posterior(g, n(a)) for a, g in axes.items()},
                                   pu, 0.0, bool(veto), veto, conf.get(nid),
                                   *((None, None, None) if m is None else (mixture.prob_greater(m[1]), m[2], mixture.expected_gain(m[1]))))
    for nid, c in conf.items():        # a confirm-only node (campaign graduate, no selection seed): its fresh seeds
        if nid not in nodes:           # alone; P(U>U_inc) below from its confirm-only increment (loop.seeds.step)
            nodes[nid] = NodePosterior(nid, own_conf[nid], c, {}, prob_greater((c.mean, c.var)), 0.0, False, (), c)
    eligible =[nid for nid, p in nodes.items() if nid in conf and nid != ROOT_NODE_ID
                and not p.vetoed and ledger.nodes[nid].status is NodeStatus.CONFIRMED and nid not in blocked]
    inc = max(eligible, key=lambda k: (conf[k].mean, k), default=ROOT_NODE_ID)
    inc = inc if conf[inc].mean > 0.0 else ROOT_NODE_ID
    inc = hold(inc, conf, eligible) if hold is not None else inc
    out = {}
    for nid, p in nodes.items():
        if nid in (ROOT_NODE_ID, inc):
            out[nid] = p
            continue
        pid = ledger.nodes[nid].parent_id or ROOT_NODE_ID                  # what it scores: its declared carrier
        raw = dataclasses.replace(p, increment=posterior(panel.tree.eps[nid], p.increment.n)) if nid in own else p
        pu = nodes[pid].utility if pid in nodes else zero
        g = beats_gauss(raw, pid, inc, conf, own_conf.get(nid), pu)
        if (m := mix.get(nid)) is not None and own_conf.get(nid) is None:     # selection seeds only: the mixture
            base, ci = (zero, zero) if pid == inc else (conf.get(pid) or pu, conf[inc])
            pb = mixture.prob_greater(m[0], (base.mean - ci.mean, base.var + ci.var))
        else:
            pb = prob_greater(g)
        out[nid] = NodePosterior(**{**p.__dict__, "p_beats_incumbent": pb})
    return out, inc


def measure_offsets(ctx: LoopContext, ledger: noise.Ledger,
                    fits: Mapping[str, TargetFit]) -> dict[str, dict[str, Gauss]]:
    """node -> target -> ``U_measure - U_parent`` (variance: the difference of theirs, floored at 0) from a fit
    on DECLARED parents, for every re-measured node: what its paired deltas carry that is not its own change."""
    plain, out = dataclasses.replace(ledger, measure={}), {}
    for nid, m in ledger.measure.items():
        if nid not in ledger.data:
            continue
        p = ledger.nodes[nid].parent_id or ROOT_NODE_ID
        out[nid] = {}
        for t, f in fits.items():
            um, up = parent_utility(ctx, f, plain, m), parent_utility(ctx, f, plain, p)
            out[nid][t] = (um[0] - up[0], max(0.0, um[1] - up[1]))
    return out


def diag_features(store: Any) -> Any:
    """node -> its checkpoint diagnostics' derived quantities (loop.diagnostics.node_features), when that layer and
    its table exist; else None (the predictive model's diagnostics group is then all-missing and drops out)."""
    try:
        from loop import diagnostics as DG
    except ImportError:
        return None
    fn = getattr(DG, "node_features", None)
    if fn is None:
        return None
    cache: dict[str, Any] = {}

    def get(nid: str) -> Any:
        if nid not in cache:
            try:
                cache[nid] = fn(store, nid)
            except Exception:               # noqa: BLE001 - a missing table / row is no diagnostic, never a failed fit
                cache[nid] = None
        return cache[nid]
    return get


def refit_utility(ctx: Any, fit: TargetFit, ledger: noise.Ledger, eps: Mapping[str, Gauss]) -> TargetFit:
    """A copy of the panel ``TargetFit`` whose scored nodes' eps are ``eps`` (their gate posterior moments) and whose
    utilities ``U_i = U_parent + eps_i`` are recomputed on them, parents first (loop.world.shrinkage.fit_target)."""
    out = dataclasses.replace(fit, tree=dataclasses.replace(fit.tree, eps={**fit.tree.eps, **eps}), utility={})
    for nid in sorted(fit.utility, key=lambda n: (ledger.nodes[n].created_ts, n)):
        up, e = parent_utility(ctx, out, ledger, noise.parent_of(ledger, nid)), out.tree.eps[nid]
        out.utility[nid] = (up[0] + e[0], up[1] + e[1])
    return out


class ComputedWorldModel:
    def __init__(self, ctx: LoopContext) -> None:
        self.ctx = ctx

    def fit(self, store: Any, persist: bool = True) -> WorldSnapshot:
        ctx, store = self.ctx, store if persist else StateOverlay(store)
        if store.task_id != ctx.task_id:
            raise ValueError(f"store is for task {store.task_id!r}, model for {ctx.task_id!r}")
        ledger = noise.read_ledger(ctx, store)
        sigma = noise.fit_noise(ctx, store, ledger)
        ledger = noise.correct(ctx, ledger, sigma)
        plain = dataclasses.replace(ledger, measure={})
        scores = fits = {t: shrinkage.fit_target(ctx, store, t, plain, sigma[t]) for t in noise.targets(ctx)}
        if ledger.measure:   # own increments over each node's measure parent; U and P(U>U_inc) stay as scored
            ledger = noise.rebase(ledger, measure_offsets(ctx, ledger, scores))
            fits = {t: shrinkage.fit_target(ctx, store, t, ledger, sigma[t]) for t in noise.targets(ctx)}
        wm = predictive.fit(ctx, store, plain, ledger, sigma, diag=diag_features(store)) \
            if ctx.cfg.prior.model == "predictive" else None
        mix, mixes, raw = self.gates(wm)
        if mix:          # the gate posterior replaces the tree's for every scored node (loop.world.predictive)
            scores = {**scores, PANEL: refit_utility(ctx, scores[PANEL], plain,
                                                     {nid: mixture.moments(c[0]) for nid, c in mix.items()})}
        prev = store.get_state(K_INC)
        fresh = fresh_vs_root(ctx, store, plain, sigma)

        def hold(inc: str, conf: Mapping[str, Posterior], eligible: list[str]) -> str:
            """The incumbent moves only on evidence. Against an AUDITED incumbent: a non-audited challenger never
            displaces it (it is audited first: every confirmed node is), and EVERY audited node (best audited
            estimate first; 2026-10-04: a node's lucky audit seed 203 made it the best audited estimate, so a node,
            +0.0021 paired over 10 fresh seeds at 2.1 sd, was never tested) or the confirm leader displaces it
            only when their PAIRED difference over every fresh seed both ran (:func:`fresh_paired`:
            audit + confirm + parent seeds, never a selection seed; the audit seeds alone (:func:`audit_paired`) when
            they share < 2) beats 2x its sd, whatever their relatedness (a node took a node's place on U +0.0077 vs
            +0.0076 while losing on all four shared audit seeds, -0.0029 mean). Two nodes RE-MINTED together
            (:func:`reminted`, loop.remint) pair only on seeds neither was selected on, confirm seeds included: the
            incumbency must survive never-used seeds. Against an unaudited one, a twin of / a node containing
            or contained by it needs its audited estimate to beat by 2x the combined sd (a node is built from
            a node: a higher point estimate on the same recipe is noise)."""
            from loop.select import related                  # loop.select imports this package
            passed = set((store.get_state(K_AUDIT) or {}).get("passed") or ())
            aud = [n for n in eligible if n in passed]

            def beats(c: str, cur: str) -> bool:
                if cur == ROOT_NODE_ID or c == cur:
                    return c != cur
                if cur in passed:
                    if c not in passed:
                        return False
                    (m1, v1), (m0, v0) = audited_estimate(store, c, conf), audited_estimate(store, cur, conf)
                    m, v = (reminted(store, c, cur) and fresh_paired(ctx, store, c, cur, strict=True)
                            or fresh_paired(ctx, store, c, cur) or audit_paired(store, c, cur) or (m1 - m0, v1 + v0))
                    return m > 0.0 and m >= 2.0 * v ** 0.5
                if not related(store, c, cur, ctx.cfg.novelty.major_ratio):
                    return True
                (m1, v1), (m0, v0) = audited_estimate(store, c, conf), audited_estimate(store, cur, conf)
                return m1 - m0 >= 2.0 * (v1 + v0) ** 0.5
            if inc == ROOT_NODE_ID:
                return inc
            order = sorted(aud, key=lambda k: (-audited_estimate(store, k, conf)[0], k))
            best = order[0] if order else None
            cur = prev if prev in eligible else best or inc
            for c in ((*order, inc) if cur in passed else (inc,)):
                cur = c if c is not None and beats(c, cur) else cur
            return cur
        nodes, incumbent = build_posteriors(ctx, plain, scores, sigma,
                                            {nid: mixture.moments(mix[nid][1]) if nid in mix else fits[PANEL].tree.eps[nid]
                                             for nid in ledger.offset if nid in fits[PANEL].tree.eps},
                                            held_back(store), hold, mix, fresh)
        if incumbent != prev:
            store.log("incumbent", node_id=incumbent, previous=prev or "")
        store.set_state(K_INC, incumbent)
        for node in ledger.nodes.values():            # freeze at confirm start (D3.7)
            if node.status is NodeStatus.CONFIRMING:
                evalues.freeze_confirm(ctx, store, node, nodes)
        return WorldSnapshot(
            task_id=ctx.task_id, n_scored=len(fits[PANEL].obs), sigma=sigma,
            variance={t: f.comps for t, f in fits.items()}, nodes=nodes, incumbent_id=incumbent,
            family_means=fits[PANEL].family_means,
            levers=wm.lever_effects() if wm else (),
            hypotheses=evalues.all_evidence(ctx, store, ledger, fits),
            banked=banked(ctx, store, nodes, _confirm_estimates(ctx, plain, scores[PANEL])[1]),
            measured={nid: (m, *ledger.offset.get(nid, {}).get(PANEL, (0.0, 0.0))) for nid, m in ledger.measure.items()},
            own_confirm={nid: (sum(x for x, _ in nd.confirm) / len(nd.confirm)
                               - ledger.offset.get(nid, {}).get(PANEL, (0.0, 0.0))[0], len(nd.confirm))
                         for nid, nd in plain.data.items() if nd.confirm},
            eb={g: (m.prior, m.n) for g, m in mixes.items()}, eb_raw=raw if mix else {}, wm=wm)

    def gates(self, wm: predictive.Fit | None) -> tuple[dict, dict, dict]:
        """(node -> (gate posterior components of its increment over its declared parent, of its own increment, its
        prior P(own gain > 0)), gate cell -> prior Mixture, node -> (cell, own selection-seed increment, its variance))
        from the predictive fit (loop.world.predictive: the gate prior per cell, conditioned as its out-of-time
        evaluation chose); all empty without one (the tree's estimates stand). A node's posterior conditions on all its
        seeds (selection + confirm); ``raw`` (selection only) is what loop.select.pooled_gain re-pools."""
        if wm is None or not wm.gate:
            return {}, {}, {}
        mix, raw, used = {}, {}, {}
        for nid, row in wm.rows.items():
            g = wm.gate_prior(row)
            sel = row.own or row.sel
            if g is None or sel is None or row.sel is None:
                continue
            cell, prior = g
            mix[nid] = (mixture.posterior(prior, row.dec_all or row.sel), mixture.posterior(prior, row.own_all or sel),
                        mixture.prob_greater(prior.prior))
            raw[nid], used[cell] = (cell, *sel), prior
        return mix, used, raw

    def predictive(self, snapshot: WorldSnapshot, *, family: str, tag: str,
                   levers: LeverTags, node: Any = None) -> Posterior:
        """An UNRUN candidate's own increment. With the predictive model in force and the candidate ``node`` given:
        its fresh-scale prior from its creation-time features (loop.world.predictive.Fit.candidate); else the tree's
        family -> prior (D3.4)."""
        wm = getattr(snapshot, "wm", None)
        if wm is not None and node is not None:
            m, v = wm.candidate(node)
            return Posterior(mean=m, sd=max(v, 0.0) ** 0.5, n=0)
        m, v = shrinkage.predict(self.ctx, snapshot.variance[PANEL], snapshot.family_means, family)
        return Posterior(mean=m, sd=v ** 0.5, n=0)

    def confirm_evalue(self, snapshot: WorldSnapshot, node_id: str, store: Any,
                       ledger: noise.Ledger | None = None) -> tuple[float, int]:
        """(E, n confirm seeds) from the frozen (delta_1, sigma): the one machinery every
        caller uses. ``ledger``: the corrected ledger, when the caller scores many nodes at once."""
        ledger = ledger or noise.correct(self.ctx, noise.read_ledger(self.ctx, store), snapshot.sigma)
        node = ledger.nodes[node_id]
        d1, sig = evalues.freeze_confirm(self.ctx, store, node, snapshot.nodes)
        off = snapshot.measured[node_id][1] if node_id in snapshot.measured else 0.0   # over its measure parent
        nd = ledger.data.get(node_id)
        xs, rs = evalues.confirm_increments(nd) if nd is not None else ([], [])   # + a legs_of graduate's screen seeds
        cs = [x - off for x in xs]
        rv = noise.resume_var(self.ctx, snapshot.sigma, PANEL)       # a resumed confirm run: seed-level noise
        ex = [rv if r else 0.0 for r in rs]
        return evalues.lr_evalue(cs, d1, sig, ex), len(cs)

