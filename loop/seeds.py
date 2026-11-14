"""loop.seeds — pure promotion / transition logic and seed sets (docs/loop/DESIGN.md §5). No I/O:
:func:`step` says which transition fires now and which runs are still missing."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from loop.contracts import (ACTIVE_STATUSES, ROOT_NODE_ID, Node, NodePosterior, NodeStatus, Run,
                            RunStatus, Stage, VoidType, design_point)
from loop.spec import LoopConfig, SeedPlan
from loop.world.evalues import confirm_delta, phi
from loop.world.noise import shared_offset

CONTROL_STAGES = frozenset({Stage.BASELINE, Stage.PARENT, Stage.AUDIT, Stage.TIEBREAK, Stage.REMINT})
#: a parent's DONE runs of these stages (in this order after its control stage) serve as a child's control at their seed
#: (loop.world.noise.controls pairs the same way): same commit, same seed -- no fresh parent run is launched
REUSE_STAGES = (Stage.BASELINE, Stage.PARENT, Stage.CALIBRATE, Stage.CONFIRM, Stage.PROMOTE, Stage.SCREEN)
EARLY_CONFIRM_SEEDS = 3                # fresh confirm seeds an early confirmation needs (else the screen-out stands)
STACK_SEEDS = tuple(range(105, 110))   # fresh confirm seeds a stack may add: its legs spent the pool (below 205's blocks)


@dataclass(frozen=True)
class RunNeed:
    node_id: str
    stage: Stage
    seed: int
    for_node: str = ""        # the comparison that needs it

    @property
    def run_id(self) -> str:
        return Run.make_id(self.node_id, self.stage, self.seed)


@dataclass(frozen=True)
class Decision:
    status: NodeStatus | None = None
    fields: Mapping[str, Any] = field(default_factory=dict)
    needs: tuple[RunNeed, ...] = ()
    reason: str = ""


def _block_base(plan: SeedPlan) -> int:
    return 1 + max((*plan.screen, *plan.promote, *plan.confirm_pool, *plan.audit, *plan.calibration))


def screen_block(plan: SeedPlan, block: int) -> tuple[int, ...]:
    if block == 0:
        return tuple(plan.screen)
    base, n = _block_base(plan), len(plan.screen)
    return tuple(range(base + (block - 1) * n, base + block * n))


def promote_block(plan: SeedPlan, block: int) -> tuple[int, ...]:
    """``n_promote`` consecutive seeds of the ``promote`` pool, rotating per block, so blocks do not
    all pair their promotions against one control run."""
    k, pool = plan.n_promote, plan.promote
    return tuple(pool[(block * k + i) % len(pool)] for i in range(k))


def assign_blocks(nodes: Iterable[Node], plan: SeedPlan, runs: Iterable[Run] = ()) -> dict[str, int]:
    """node id -> screen block; pure and append-stable (creation order). A node that already has
    a screen run keeps the block its seeds say (so a changed ``block_children`` never moves it).
    Every point of one designed batch (tag ``d<b>pt<i>``) shares ONE block, however many there are. A dose-ladder
    point (note ``block_of=<base>``, loop.tune) runs on its base's OWN block; a tuning / leg-factorial graduate
    (``tune_of=`` / ``legs_of=``) on a block of its own, never one its selection ran on."""
    base, n = _block_base(plan), len(plan.screen)
    have = {r.node_id: 0 if r.seed in plan.screen else (r.seed - base) // n + 1
            for r in runs if r.stage is Stage.SCREEN and (r.seed in plan.screen or r.seed >= base)}
    used = set(have.values())
    out: dict[str, int] = {}
    cur: dict[str, list[int]] = {}                 # parent -> [block, children in it]
    fresh = 0
    for n_ in sorted(nodes, key=lambda n: (n.created_ts, n.id)):
        if n_.id == ROOT_NODE_ID or n_.status is NodeStatus.REJECTED:
            continue
        notes = f" {n_.notes or ''}"
        if (of := notes.partition(" block_of=")[2].split(" ")[0]) and of in out:
            out[n_.id] = out[of]
            continue
        if (" tune_of=" in notes or " legs_of=" in notes) and n_.id not in have:
            while fresh in used:
                fresh += 1
            used.add(fresh)
            out[n_.id] = fresh
            continue
        pt = design_point(n_)
        pid = "design:" + n_.mechanism_tag.partition("pt")[0] if pt else n_.parent_id or ROOT_NODE_ID
        slot = cur.get(pid)
        if n_.id in have:
            if slot is None or slot[0] != have[n_.id]:
                slot = cur[pid] = [have[n_.id], 0]
        elif slot is None or (slot[1] >= plan.block_children and not pt):
            while fresh in used:
                fresh += 1
            slot = cur[pid] = [fresh, 0]
            used.add(fresh)
        slot[1] += 1
        out[n_.id] = slot[0]
    return out


def bayes_factor(p: float, prior: float) -> float:
    """Posterior odds of a gain over its prior odds: > 1 when the node's own seeds made a gain MORE likely."""
    p, prior = min(max(p, 1e-12), 1.0 - 1e-12), min(max(prior, 1e-12), 1.0 - 1e-12)
    return (p / (1.0 - p)) / (prior / (1.0 - prior))


def promote_ok(post: NodePosterior, promo: Any) -> bool:
    """The screen / promote gate: P(U>0) and P(own gain>0) >= p_promote, or (``promotion.voi_gain`` > 0) the
    value-of-information rule -- E[max(own gain, 0)] under the EB posterior >= voi_gain with P(U>0) >= p_promote: a
    +0.004 screen whose P(gain>0) the prior holds at 0.25 is still worth 3 confirm seeds."""
    if post.p_gate >= promo.p_promote:
        return True
    return (promo.voi_gain > 0.0 and post.gain_plus is not None and post.gain_plus >= promo.voi_gain
            and post.p_positive >= promo.p_promote)


def confirm_power(post: NodePosterior, sigma: float, alpha: float, n: int) -> float:
    """Predictive probability that ``n`` fresh confirm seeds lift the frozen LR e-value (loop.world.evalues.lr_evalue at
    delta_1 = confirm_delta(sigma, alpha, n)) to 1/alpha. E reaches 1/alpha exactly when the mean confirm increment
    reaches delta_1, so given a true gain g, P = Phi((g - delta_1) sqrt(n) / sigma). Under the EB posterior the gain is
    positive with P(gain>0), at its conditional mean E[gain+] / P(gain>0), and otherwise taken as 0 (no worse)."""
    d1, z = confirm_delta(sigma, alpha, n), math.sqrt(max(1, n)) / sigma
    p, plus = post.p_gain, post.gain_plus
    g = plus / p if plus is not None and p > 0.0 else max(post.increment.mean, 0.0)
    return p * phi((g - d1) * z) + (1.0 - p) * phi(-d1 * z)


def early_confirm_bf(post: NodePosterior, cfg: LoopConfig, raw: float | None, own_runs: Iterable[Run],
                     sigma: float | None = None) -> float | None:
    """EARLY CONFIRMATION eligibility of a screen below the gate: positive raw mean, P(gain>0) between futile and the
    gate, its seeds made a gain more likely than the generation's prior (BF > 1), >= EARLY_CONFIRM_SEEDS confirm
    seeds left under the 5-seed cap -- 2 seeds cannot show a +0.002 carrier gain, so it goes to fresh confirm seeds
    (not more selection-biased screen seeds that left 1 confirm seed) -- and (``promotion.early_confirm_power`` > 0,
    ``sigma`` the confirm sigma) a predictive probability >= early_confirm_power that those seeds reach the confirm bar
    (:func:`confirm_power`; live 2026-10-07: BF 1.3-2.1 at E[gain+] 0.0002-0.0007 sent a node/1/4, a node, a node to 3
    confirm seeds at ~5-8% power). Its Bayes factor, or None."""
    promo = cfg.promotion
    n = planned_confirm(cfg, own_runs)
    if not promo.early_confirm or raw is None or raw <= 0.0 or post.gain_prior is None \
            or post.p_gain < promo.screen_futility_p * post.gain_prior \
            or n < min(EARLY_CONFIRM_SEEDS, promo.confirm_max_seeds):
        return None
    if promo.early_confirm_power > 0.0 and sigma is not None \
            and confirm_power(post, sigma, promo.alpha_confirm, n) < promo.early_confirm_power:
        return None
    bf = bayes_factor(post.p_gain, post.gain_prior)
    return bf if bf > 1.0 else None


def reachable(e_value: float, frozen: Sequence[float], left: int, alpha: float) -> bool:
    """Whether ``left`` more confirm seeds, each at the frozen delta_1 + 1 sigma, could lift E to 1/alpha."""
    d1, sig = float(frozen[0]), float(frozen[1])
    best = math.log(max(e_value, 1e-300)) + left * (d1 * (d1 + sig) - d1 * d1 / 2.0) / (sig * sig)
    return best >= math.log(1.0 / alpha)


def planned_confirm(cfg: LoopConfig, own_runs: Iterable[Run]) -> int:
    """Confirm seeds the node may use: ``promotion.confirm_max_seeds`` within ``seeds.max_seeds_per_node`` less the
    selection (screen / promote) seeds it ran -- a node that promoted on a seed before the promote stage was
    dropped keeps 5 in all (2 confirm seeds), a new one gets 3."""
    n, cap = cfg.promotion.confirm_max_seeds, cfg.seeds.max_seeds_per_node
    used = len({(r.stage, r.seed) for r in own_runs
                if r.stage in (Stage.SCREEN, Stage.PROMOTE) and r.status is RunStatus.DONE})
    return max(1, min(n, cap - used)) if cap > 0 else n


def control_of(node: Node) -> tuple[str, Stage]:
    pid = node.parent_id or ROOT_NODE_ID
    return (ROOT_NODE_ID, Stage.BASELINE) if pid == ROOT_NODE_ID else (pid, Stage.PARENT)


def is_retryable(run: Run, infra_retry: int) -> bool:
    """An infra void while attempts are left; any other failure of a control/audit run once
    (a crash is deterministic: retrying it eight times only burns GPU). A run the loop cancelled never is."""
    if run.void_type is VoidType.CANCELLED:
        return False
    if run.status is RunStatus.VOID and run.void_type in (None, VoidType.INFRA):
        return run.attempt < infra_retry
    return (run.status in (RunStatus.VOID, RunStatus.CRASH) and run.stage in CONTROL_STAGES
            and run.attempt < min(1, infra_retry))


def run_failed(run: Run, infra_retry: int) -> bool:
    return run.status in (RunStatus.VOID, RunStatus.CRASH) and not is_retryable(run, infra_retry)


def _state(run: Run | None, infra_retry: int) -> str:
    if run is None:
        return "missing"
    if run.status is RunStatus.DONE:
        return "done"
    return "failed" if run_failed(run, infra_retry) else "pending"


def _index(runs: Iterable[Run]) -> dict[tuple[Stage, int], Run]:
    return {(r.stage, int(r.seed)): r for r in runs}


def _why(run: Run) -> str:
    return "crash" if run.status is RunStatus.CRASH else f"void:{(run.void_type or VoidType.INFRA).value}"


def _fail(why: str) -> Decision:
    return Decision(NodeStatus.CRASHED if why == "crash" else NodeStatus.VOID, reason=why)


def step(node: Node, *, own_runs: Sequence[Run], control_runs: Sequence[Run],
         post: NodePosterior | None, cfg: LoopConfig, sigma_panel: float, infra_retry: int,
         confirm: tuple[float, int] | None = None, screen: Sequence[int] | None = None,
         promote: Sequence[int] | None = None, confirm_rot: int = 0, sigma_ucb: float | None = None,
         frozen: Sequence[float] | None = None, confirm_pool: Sequence[int] | None = None,
         raw: float | None = None, extra: int = 0) -> Decision:
    """One lifecycle evaluation. ``screen`` / ``promote`` are the node's block seeds; ``confirm`` is the
    world model's (E, n confirm seeds); ``post`` None = not scored yet (needs still reported). ``confirm_rot``:
    the node's rank among its confirming siblings (same control): its next confirm seed is the
    ``confirm_rot``-th still-unused pool seed, so siblings confirm on different seeds / control runs. ``sigma_ucb``:
    the frozen confirm sigma bound the alternative is set from (None: ``sigma_panel``). ``frozen``: the node's frozen
    confirm (delta_1, sigma) -- the confirm FUTILITY stop: after a confirm seed, if E could not reach 1/alpha_confirm
    even with every remaining planned seed at delta_1 + 1 sigma, the node is NOT_CONFIRMED now (its open runs
    cancelled) instead of spending them. ``confirm_pool``: the seeds it may confirm on (None: ``seeds.confirm_pool``;
    a stack's: none its legs ran, loop.driver._stack_pool). ``raw``: its raw paired own-increment mean (an early
    confirmation needs > 0).
    ``extra``: confirm seeds granted past ``planned_confirm`` (a backfill re-confirmation, loop.backfill)."""
    if node.status not in ACTIVE_STATUSES:
        return Decision()
    own, ctl = _index(own_runs), _index(control_runs)
    pid, cstage = control_of(node)
    promo, needs = cfg.promotion, []

    def reused(seed: int) -> Run | None:
        """The parent's own scored DONE run at ``seed`` (any REUSE_STAGES stage) when it has no control-stage run."""
        return next((r for st in REUSE_STAGES if st is not cstage and (r := ctl.get((st, seed))) is not None
                     and r.status is RunStatus.DONE and r.scores), None)

    def paired(seed: int) -> str:
        """State of the control at ``seed`` (recording a need or a failure); a reused parent run is 'done' -- also
        when the control-stage run failed (live 2026-10-03: a ladder's 1x design point closed on its own screen and
        its close cancelled the duplicate PARENT runs, voiding every dose point paired on them). While the parent's
        own run at ``seed`` is in flight it is the control: no duplicate control run is launched."""
        st = _state(ctl.get((cstage, seed)), infra_retry)
        if st in ("missing", "failed") and reused(seed) is not None:
            return "done"
        if st == "missing" and any(st_ is not cstage and _state(ctl.get((st_, seed)), infra_retry) == "pending"
                                   for st_ in REUSE_STAGES):
            return "pending"
        if st == "missing":
            needs.append(RunNeed(pid, cstage, seed, node.id))
        return st

    if node.status in (NodeStatus.SCREENING, NodeStatus.PROMOTING):
        screening = node.status is NodeStatus.SCREENING
        stage = Stage.SCREEN if screening else Stage.PROMOTE
        done, scored = True, 0
        for seed in map(int, (screen or cfg.seeds.screen) if screening else (promote or cfg.seeds.promote)):
            run = own.get((stage, seed))
            st = _state(run, infra_retry)
            if st == "failed":
                return _fail(_why(run))
            if st == "missing":
                needs.append(RunNeed(node.id, stage, seed, node.id))
            cst = paired(seed)
            if cst == "failed":
                return _fail(f"void:partner {Run.make_id(pid, cstage, seed)}")
            done, scored = done and st == cst == "done", scored + (st == cst == "done")
        prescreened = " legs_of=" in f" {node.notes or ''}"      # loop.legs: the factorial was the selection
        # futile: P(gain>0) under screen_futility_p -- under the EB mixture (loop/world/mixture), screen_futility_p x its
        # PRIOR P(gain>0): a carrier child's prior is ~0.03 (pi0 0.90), so the absolute bar would end every screen
        # before its first seed; relative, it ends one whose own seeds made a real gain 10x less likely than before
        futile = promo.screen_futility_p * (post.gain_prior if post is not None and post.gain_prior is not None else 1.0)
        if screening and not done and scored and post is not None and not design_point(node) and not prescreened \
                and post.p_gain < futile:
            return Decision(NodeStatus.SCREENED_OUT, reason=f"futile at screen: P(gain>0)={post.p_gain:.3f} "
                                                            f"< {futile:.3f} after {scored} seed(s)")
        if done and design_point(node):
            return Decision(NodeStatus.SCREENED_OUT, reason="design point (screen only)")
        if not done or post is None:
            return Decision(needs=tuple(needs))
        if post.vetoed:
            return Decision(NodeStatus.VETOED, reason=f"harm veto on {list(post.veto_axes)}")
        gate = f"P(U>0)={post.p_positive:.3f} P(gain>0)={post.p_gain:.3f}"
        gate += f" E[gain+]={post.gain_plus:.4f}" if post.gain_plus is not None else ""
        if screening and prescreened:            # no 2-seed screen gate: its screen seeds are fresh paired evidence
            alt = confirm_delta(sigma_ucb or sigma_panel, promo.alpha_confirm, planned_confirm(cfg, own_runs) + scored)
            return Decision(NodeStatus.CONFIRMING, {"confirm_alt": alt}, reason=f"{gate} (legs_of: no screen gate; "
                                                                                f"its {scored} screen seeds count in E)")
        ok = promote_ok(post, promo)                     # its OWN gain too, not only the carrier's U
        if screening and not ok and (bf := early_confirm_bf(post, cfg, raw, own_runs, sigma_ucb or sigma_panel)) is not None:
            n_c = planned_confirm(cfg, own_runs)
            alt = confirm_delta(sigma_ucb or sigma_panel, promo.alpha_confirm, n_c)
            pw = confirm_power(post, sigma_ucb or sigma_panel, promo.alpha_confirm, n_c)
            return Decision(NodeStatus.CONFIRMING, {"confirm_alt": alt},
                            reason=f"{gate} (early confirmation: BF={bf:.2f} power={pw:.2f})")
        if screening and (not ok or cfg.seeds.promote_stage):
            return Decision(NodeStatus.PROMOTING if ok else NodeStatus.SCREENED_OUT, reason=gate)
        why = (f"{gate} " if screening else "") + f"P(U>U_inc)={post.p_beats_incumbent:.3f}"   # no promote stage
        if post.p_beats_incumbent < promo.p_confirm:
            return Decision(NodeStatus.SCREENED_OUT, reason=why)
        alt = confirm_delta(sigma_ucb or sigma_panel, promo.alpha_confirm, planned_confirm(cfg, own_runs))
        return Decision(NodeStatus.CONFIRMING, {"confirm_alt": alt}, reason=why)

    # CONFIRMING: pool seeds fresh for the node, sequentially -- those where the parent already has a DONE run (reused as
    # the control) first. Reusing the parent's own CONFIRM-seed runs biases the child CONSERVATIVELY: the parent was
    # confirmed partly because those seeds came in high for it, so the child-minus-parent delta there reads low.
    pool = sorted((int(s) for s in (cfg.seeds.confirm_pool if confirm_pool is None else confirm_pool)
                   if (Stage.SCREEN, int(s)) not in own and (Stage.PROMOTE, int(s)) not in own),
                  key=lambda s: (cstage, s) not in ctl and reused(s) is None)
    used = [s for s in pool if (Stage.CONFIRM, s) in own]
    waiting = False
    for seed in used:
        run = own[(Stage.CONFIRM, seed)]
        if run_failed(run, infra_retry):
            if run.status is RunStatus.CRASH:
                return _fail("crash")
            if run.void_type in (VoidType.EVAL, VoidType.TOO_SLOW):     # a verdict, not bad luck
                return Decision(NodeStatus.NOT_CONFIRMED, reason=f"void:eval on confirm seed {seed}")
            continue                               # spent infra retries on this seed; move on
        st = paired(seed)
        if st == "failed":
            return _fail(f"void:partner {Run.make_id(pid, cstage, seed)}")
        waiting = waiting or st != "done" or run.status is not RunStatus.DONE
    if waiting:
        return Decision(needs=tuple(needs))
    e_value, n = confirm if confirm is not None else (1.0, 0)
    fields = {"confirm_e": float(e_value)} if n else {}
    if n and post is not None and post.vetoed:
        return Decision(NodeStatus.VETOED, fields, reason=f"harm veto on {list(post.veto_axes)}")
    # a legs_of graduate's screen seeds count in E (loop.world.evalues.confirm_increments), but its verdict needs >= 1
    # confirm-stage run of its own
    ran = any(own[(Stage.CONFIRM, s)].status is RunStatus.DONE for s in used)
    if n and e_value >= 1.0 / promo.alpha_confirm and (ran or " legs_of=" not in f" {node.notes or ''}"):
        if " campaign_of=" in f" {node.notes or ''}" and post is not None and post.p_beats_incumbent < promo.p_confirm:
            return Decision(NodeStatus.NOT_CONFIRMED, fields, reason=f"E={e_value:.3g} n={n} beats its parent but not "
                                                                    f"the incumbent: P(U>U_inc)={post.p_beats_incumbent:.3f}")
        return Decision(NodeStatus.CONFIRMED, fields, reason=f"E={e_value:.3g} n={n}")
    if n and e_value <= promo.futility_e:
        return Decision(NodeStatus.NOT_CONFIRMED, fields, reason=f"futile E={e_value:.3g} n={n}")
    remaining, planned = [s for s in pool if (Stage.CONFIRM, s) not in own], planned_confirm(cfg, own_runs) + extra
    left = min(planned - len(used), len(remaining))
    if n and frozen and left > 0 and e_value > 0:
        d1, sig = float(frozen[0]), float(frozen[1])
        if not reachable(e_value, (d1, sig), left, promo.alpha_confirm):
            best = math.log(e_value) + left * (d1 * (d1 + sig) - d1 * d1 / 2.0) / (sig * sig)
            return Decision(NodeStatus.NOT_CONFIRMED, fields,
                            reason=f"futile: E={e_value:.3g} n={n} reaches at most {math.exp(min(700.0, best)):.3g} < "
                                   f"{1.0 / promo.alpha_confirm:g} with {left} more seed(s) at delta_1+1 sigma")
    if len(used) >= planned or not remaining:
        return Decision(NodeStatus.NOT_CONFIRMED, fields, reason=f"seeds spent E={e_value:.3g} n={n}")
    nxt = remaining[confirm_rot % len(remaining)]     # live: a node / a node / a node all on seed 101 vs one a node run
    needs.append(RunNeed(node.id, Stage.CONFIRM, nxt, node.id))
    paired(nxt)
    return Decision(fields=fields, needs=tuple(needs))


def _panel(cfg: LoopConfig, run: Run, base: Run) -> float | None:
    if all(a in run.scores and a in base.scores for a in cfg.axes):
        return sum(cfg.weights[a] * (run.scores[a] - base.scores[a]) for a in cfg.axes)
    return None


def audit_offsets(node_id: str, by_node: Mapping[str, Sequence[Run]], cfg: LoopConfig,
                  estimates: Mapping[str, tuple[float, float]], sigma: float,
                  seeds: Sequence[int] | None = None) -> dict[int, float]:
    """audit seed -> shrunk deviation of root's audit run there (every audit on these seeds pairs against the
    same root runs), from the OTHER audited nodes' deltas minus their estimated gain (mean, var). ``seeds``: the
    audit's own seed set (loop.remint rotation; None: ``seeds.audit``)."""
    roots, out = _index(by_node.get(ROOT_NODE_ID, [])), {}
    for seed in map(int, cfg.seeds.audit if seeds is None else seeds):
        r0 = roots.get((Stage.AUDIT, seed))
        if r0 is None or r0.status is not RunStatus.DONE:
            continue
        sibs = []
        for nid, runs in by_node.items():
            r = _index(runs).get((Stage.AUDIT, seed))
            if (nid not in (node_id, ROOT_NODE_ID) and nid in estimates and r is not None
                    and r.status is RunStatus.DONE and (c := _panel(cfg, r, r0)) is not None):
                sibs.append((c - estimates[nid][0], estimates[nid][1]))
        out[seed] = shared_offset(sibs, sigma)[0]
    return out


def audit_deltas(node_runs: Sequence[Run], root_runs: Sequence[Run], *, cfg: LoopConfig,
                 offsets: Mapping[int, float] | None = None, seeds: Sequence[int] | None = None) -> dict[int, float]:
    """audit seed -> the node's panel delta vs root there minus root's shared deviation, for every
    seed both runs finished: exactly the values :func:`audit_step` weighs."""
    mine, roots, out = _index(node_runs), _index(root_runs), {}
    for seed in map(int, cfg.seeds.audit if seeds is None else seeds):
        a, b = mine.get((Stage.AUDIT, seed)), roots.get((Stage.AUDIT, seed))
        if (a is not None and b is not None and a.status is RunStatus.DONE and b.status is RunStatus.DONE
                and (c := _panel(cfg, a, b)) is not None):
            out[seed] = c - (offsets or {}).get(seed, 0.0)
    return out


def audit_gate(cs: Sequence[float], *, verdict: str, e_value: float, delta: float, sigma: float,
               alpha: float, offsets: Mapping[int, float] | None = None, seeds: Sequence[int] = (),
               ts: float | None = None, source: str = "gate") -> dict[str, Any]:
    """The audit verdict's own numbers, frozen when it is made (the Labless note prints these;
    a later re-fit of the shared-control correction never rewrites them)."""
    cs = list(cs)
    se = statistics.stdev(cs) / len(cs) ** 0.5 if len(cs) > 1 else None
    return {"verdict": verdict, "mean": statistics.fmean(cs) if cs else None, "se": se, "n": len(cs),
            "e": float(e_value), "alpha": float(alpha), "delta": delta, "sigma": sigma,
            "deltas": {str(s): c for s, c in zip(seeds, cs)} if seeds else {},
            "offsets": {str(k): v for k, v in (offsets or {}).items()}, "ts": ts, "source": source}


def audit_step(node_id: str, node_runs: Sequence[Run], root_runs: Sequence[Run], *,
               cfg: LoopConfig, delta: float, sigma: float, infra_retry: int,
               offsets: Mapping[int, float] | None = None, resume_v: float = 0.0,
               seeds: Sequence[int] | None = None) -> tuple[str, float, tuple[RunNeed, ...]]:
    """Held-out audit of ``node_id`` vs root -> (verdict, E, needs). ``verdict`` is ''
    while runs are outstanding, else 'passed' iff the likelihood ratio of the frozen
    claimed gain ``delta`` over 0 reaches ``1/alpha_audit`` on >= 2 seeds. ``offsets`` (seed ->
    root audit run's shared deviation, :func:`audit_offsets`) is subtracted from each delta; the
    frozen ``sigma`` bound still covers the corrected delta's variance (<= sigma^2). A seed where either run
    resumed from latest.pt (non-exact) weighs with ``sigma^2 + resume_v`` (seed-level noise). ``seeds``: the audit's
    own seed set (loop.remint rotation; None: ``seeds.audit``)."""
    mine, roots = _index(node_runs), _index(root_runs)
    needs, open_ = [], False
    seeds = list(map(int, cfg.seeds.audit if seeds is None else seeds))
    for seed in seeds:
        for nid, runs in ((node_id, mine), (ROOT_NODE_ID, roots)):
            st = _state(runs.get((Stage.AUDIT, seed)), infra_retry)
            if st == "missing":
                needs.append(RunNeed(nid, Stage.AUDIT, seed, node_id))
            open_ = open_ or st in ("missing", "pending")
    if open_:
        return "", 1.0, tuple(needs)
    cs = audit_deltas(node_runs, root_runs, cfg=cfg, offsets=offsets, seeds=seeds)
    res = {s for s in cs if mine[(Stage.AUDIT, s)].resumed or roots[(Stage.AUDIT, s)].resumed}
    e_value = math.exp(min(700.0, sum((c * delta - delta * delta / 2) / (sigma ** 2 + (resume_v if s in res else 0.0))
                                      for s, c in cs.items())))
    ok = len(cs) >= 2 and e_value >= 1.0 / cfg.promotion.alpha_audit
    return ("passed" if ok else "failed"), e_value, ()


def stage_needs(node_id: str, stage: Stage, seeds: Iterable[int], runs: Sequence[Run], infra_retry: int,
                for_node: str = "") -> tuple[list[RunNeed], bool]:
    """(the node's runs at ``stage`` still to launch on ``seeds``, whether any is missing / in flight)."""
    own, needs, open_ = _index(runs), [], False
    for seed in map(int, seeds):
        st = _state(own.get((stage, seed)), infra_retry)
        if st == "missing":
            needs.append(RunNeed(node_id, stage, seed, for_node or node_id))
        open_ = open_ or st in ("missing", "pending")
    return needs, open_

