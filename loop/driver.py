"""loop.driver — the plain-Python loop (docs/loop/DESIGN.md §6): tick, slots, resume, the held-out
audit. No LLM here; all state lives in the store, so a new Driver on the same database resumes."""

from __future__ import annotations

import dataclasses
import json
import math
import re
import sys
import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

from loop import backfill as BF
from loop import campaign as CP
from loop import components as CG
from loop import databank as DB
from loop import dashboard
from loop import digest as DJ
from loop import extrapolate as EX
from loop import meta as ML
from loop import novelty as NV
from loop import scout as SC
from loop import design as D
from loop import resume as RS
from loop import evalonly as EO
from loop import diagnostics as DG
from loop import prune as PR
from loop import redundancy as RD
from loop import seeds as S
from loop import selftune as T
from loop import legs as LG
from loop import regrad as GR
from loop import shift as SH
from loop.world import mixture as MX
from loop.world import predictive as WM
from loop import remint as RM
from loop import runner as R
from loop.runner import (COURTESY_EVERY_S, COURTESY_FLOOR, COURTESY_MARGIN, COURTESY_RELEASE_S,  # noqa: F401
                         COURTESY_STEP, COURTESY_STEP_AFTER_S, COURTESY_STEP_EVERY_S, K_COURTESY, K_LLM_PAUSE,
                         K_STARVED, LLM_PROBE_MAX_S, LLM_PROBE_S, MAX_RESTARTS, SMOKE_SWEEP_S, STARVE_S,
                         STARVED_PENDING, Courtesy, nap, utc_now)
from loop import tune as MT
from loop import variants as VR
from loop.agents.analyst import FP_TRIGGER
from loop.agents.llm import BackendUnavailable
from loop.labless import K_AUDIT, LablessSubmitter, backfill_audit_gate, gate_state
from loop.contracts import (ACTIVE_STATUSES, CLOSED_STATUSES, K_EPOCH, K_INFLIGHT, PANEL, ROOT_NODE_ID,
                            K_STALE_META, CandidateRequest, ExecutionAdapter, HypothesisKind, current_reviews,
                            Node, NodeStatus, Operator, ResearchResult, Run,
                            RunStatus, SmokeResult, Stage, Strategy, VoidType,
                            WorldSnapshot, design_point, is_oom, tune_control, with_oom_note)
from loop.spec import LoopContext, build, encode
from loop.select import bank as B
from loop.select.bandit import K_KIND_PULLS
from loop.select import STACK_MAX, crossover_levers, stack_plan, winner_pairs
from loop.select import intake as IN
from loop.select.intake import common_ancestor, compound_sources, compound_split, derive_families
from loop.world import K_REFROZEN, LinkIndex, StateOverlay, link_meta
from loop.world.evalues import CONFIRM_KEY, confirm_delta, sigma_ucb
from loop.world import noise as WN
from loop.world.noise import MEASURE_KEY, STATE_KEY as NOISE_KEY

_ACTIVE_RUN = (RunStatus.SUBMITTING, RunStatus.RUNNING)
_EVAL_CLOSES = CLOSED_STATUSES - {NodeStatus.REJECTED}
_TIER = {Stage.BASELINE: 0, Stage.PARENT: 0, Stage.CALIBRATE: 0, Stage.CONFIRM: 1,
         Stage.AUDIT: 1, Stage.TIEBREAK: 1, Stage.REMINT: 1, Stage.PROMOTE: 2, Stage.SCREEN: 3}
#: SCARCE MODE (slurm.scarce_gpus): {"on", "since", "avg", "starved", "samples": [[ts, running GPUs], ...]}
K_SCARCE = "runbook.scarce"
SCARCE_WINDOW_S, SCARCE_SAMPLE_S = 7200.0, 60.0
CONTROL_ONLY = frozenset({Stage.BASELINE, Stage.PARENT})   # control runs: at most half of primary while candidates wait
LLM_TRIES = 3             # failed attempts per review / consolidation before moving on
K_REVIEW = "driver.review"            # {"evals", "confirmations"} at the last stored review
K_LAST_CONFIRM = "driver.evals_at_last_confirm"
K_FP = "driver.first_principles"      # {"ts", "big"}: last first-principles review (done or given up)
K_OUTAGE_REPAIR = "driver.outage_repair"    # once: what the 2026-10-03 account outages were counted for, undone
K_DEBUG = "driver.debug"              # {crashed node id: request id}: its one deterministic DEBUG request
K_WINX_TRIES = "driver.winner_cross_tries"   # {"a+b": non-backend failures retried}
K_WINX = "driver.winner_cross"        # {"a+b": request id}: the one standing crossover of two confirmed winners
K_STACK = "driver.stacks"            # {"a+b+c": request id}: the one stack of those unconfirmed small signals
K_REGATE = "driver.regate_carrier"    # set once the live nodes were re-gated under the inherited-carrier fix
K_REGATE_OFFSET = "driver.regate_offset"   # ... and once under the robust control offsets (noise.offset_ok)
K_REINTAKE = "driver.reintake_effective_config"   # the REINTAKE_V the QUEUED nodes were last re-checked under
REINTAKE_V = 4           # 2: an exact 2x move is minor (loop.novelty.minor): a node / a node / a node off a node;
                         # 3: dead mechanisms of any age (loop.novelty.dead); 4: unreachable_schedule
K_RELABEL = "driver.relabel_cancels"  # set once the loop's own past cancels were relabelled void/cancelled
K_REMEASURE = "driver.remeasure"      # set once the live / confirmed nodes were re-gated on re-measured own increments
K_REFREEZE = "driver.refreeze_delta1"  # set once the frozen confirm alternatives were re-set to confirm_delta
K_MECH = "driver.mechanism_pooled"     # {norm tag: [node ids]}: members whose confirm seeds a pooled graduation spent
K_FP_FORCE = "driver.fp_force"        # a measurement-epoch bump asks for a first-principles review now
K_TUNE_GATE = "driver.tune_gate"      # set once the running tuning passes were re-checked under loop.tune.gate
K_HIT = "driver.hit_own_increment"     # set once the playbook built on the old hit (P(U>0)) was retired
K_READMIT = "driver.readmit_low_value"  # {"nodes": [...]}: set once the low_value rejects the exempt gate passes were requeued
READMIT_SINCE = 1791345720.0            # 2026-10-07T04:02Z: the deploy whose value gate rejected moonshots / init swaps
_REOPEN = {Stage.SCREEN: NodeStatus.SCREENING, Stage.PROMOTE: NodeStatus.PROMOTING,
           Stage.CONFIRM: NodeStatus.CONFIRMING}
MAX_SMOKE_REISSUES = 3    # re-issues of a request whose smoke never got a slot (backend unavailable, not a verdict)
K_SMOKE_REPAIR = "driver.smoke_timeout_repair"   # set once the nodes rejected on a smoke queue timeout were re-issued
SMOKE_UNAVAILABLE_REASON = "smoke_unavailable"  # reject_reason of such a node: never paid, never a verdict
REISSUE_MARK = "\n\n=== RE-ISSUED: the previous attempt's smoke job never got a GPU slot ==="
HIGH_P = 0.85             # a screen of a node this likely to beat root is HIGH importance



def pool_use(ctx: LoopContext, runs: Sequence[Run], pending: Any = None, starved: Any = None) -> str:
    """``primary 3/8 (1 running, 2 pending), idle 5/16 (preemptible)``: active runs per SLURM pool; with
    ``pending`` (run ids SLURM held PENDING) split into running / pending; ``starved``: K_STARVED."""
    pools, held = ctx.cfg.pools, set(pending or ())
    act = [r for r in runs if r.status in _ACTIVE_RUN]
    used = Counter(r.pool or pools[0].name for r in act)
    pend = Counter(r.pool or pools[0].name for r in act if r.id in held)
    return ", ".join(f"{p.name} {used[p.name]}/{p.slots}"
                     + (f" ({used[p.name] - pend[p.name]} running, {pend[p.name]} pending)" if pending is not None else "")
                     + (" (preemptible)" if p.preemptible else "")
                     + (" STARVED" if p.name in (starved or {}) and time.time() - starved[p.name] > STARVE_S else "")
                     for p in pools)


def scarce_note(store: Any) -> str:
    """``scarce=on(2.6 GPUs avg 2h; idle STARVED) `` while SCARCE MODE is in force, else ''."""
    sc = store.get_state(K_SCARCE) or {}
    return (f"scarce=on({float(sc.get('avg', 0.0)):.1f} GPUs avg 2h" + (f"; {','.join(sc['starved'])} STARVED"
            if sc.get("starved") else "") + ") ") if sc.get("on") else ""


def hb_line(ctx: LoopContext, store: Any, phase: str,
            snapshot: WorldSnapshot | None = None, researching: int = 0) -> str:
    """One cheap status line (a subset of ``loop status``): the log is otherwise silent while jobs run."""
    runs, nodes = store.runs(), store.nodes()
    n = lambda st: sum(r.status is st for r in runs)      # noqa: E731
    active = sum(r.status in _ACTIVE_RUN for r in runs)
    inc = snapshot.nodes.get(snapshot.incumbent_id) if snapshot else None
    est = (inc.confirm_estimate or inc.utility) if inc else None
    best = max(((p.utility.mean, i) for i, p in (snapshot.nodes if snapshot else {}).items()
                if i != ROOT_NODE_ID), default=None)          # gate posterior mean (loop.world.predictive), not a lucky screen's
    conf = {x.id for x in nodes if x.status is NodeStatus.CONFIRMED}
    fresh = max(((p.confirm_estimate.mean, i) for i, p in (snapshot.nodes if snapshot else {}).items()
                 if i in conf and p.confirm_estimate is not None), default=None)    # fresh confirm seeds only
    mix = getattr(snapshot, "eb", None) or {}
    try:
        from loop.agents.llm import SdkLLMClient
        llm = SdkLLMClient(ctx.spec).tier("researcher")
    except Exception:  # noqa: BLE001 - a heartbeat never fails the loop
        llm = "?"
    court, held = store.get_state(K_COURTESY) or {}, set(store.get_state(T.K_PENDING) or ())
    pend = sum(r.status is RunStatus.RUNNING and r.id in held for r in runs)
    if (lp := store.get_state(K_LLM_PAUSE) or {}).get("active"):
        llm = f"{llm}(PAUSED:{lp['kind']}_until_{utc_now(lp['until'])})"
    return (f"hb phase={phase} slots={active}/{ctx.cfg.capacity} "
            f"[{pool_use(ctx, runs, held, store.get_state(K_STARVED) or {})}] "
            + (f"courtesy_backoff=on(cap {court.get('cap')}; {','.join(court.get('jobs', []))} pending) "
               if court.get("on") else "") + scarce_note(store) + f"running={n(RunStatus.RUNNING) - pend} pending={pend} "
            f"queued={sum(x.status is NodeStatus.QUEUED for x in nodes)} done={n(RunStatus.DONE)} "
            f"void={n(RunStatus.VOID)} researching={researching} llm={llm} {CP.note(store)}"
            + (f"{LG.hb(store)} " if ctx.cfg.legs.enabled else "") +
            f"incumbent={snapshot.incumbent_id if snapshot else ROOT_NODE_ID}({(est.mean if est else 0.0):+.4f}) "
            f"best_post={'-' if best is None else f'{best[0]:+.4f}({best[1]})'} "
            f"best_fresh={'-' if fresh is None else f'{fresh[0]:+.4f}({fresh[1]})'} "
            + (f"prior_gain={'/'.join(f'{g}:{MX.prob_greater(mix[g][0]):.2f}' for g in sorted(mix) if g != 'all')} "
               if mix else "") +
            f"confirmed={sum(x.status is NodeStatus.CONFIRMED for x in nodes)} "
            f"resumed={sum(r.resumed > 0 for r in runs if r.status is RunStatus.DONE)}/{n(RunStatus.DONE)} done "
            f"{(T.preempt_line(ctx, store, time.time()) + ' ').lstrip()}labless: {gate_state(ctx, store)} proposals: {len(T.open_proposals(store))} awaiting operator")


def research_starts(queued: int, in_flight: int, target: int, qmax: int, max_concurrent: int) -> int:
    """How many bandit researchers to start this tick. A researcher in flight is future supply (~0.75 of a node:
    some are rejected), so one starts whenever queued + 0.75 x in_flight < target (nodes, see
    Driver._queue_marks), within max_concurrent and qmax, and at most max(2, ceil(max_concurrent / 4)) per tick. The
    old rule (start only while queued < low water, ignoring researchers in flight) started all 16 together;
    they finished together, overfilled the queue, then sat idle while it drained (live 2026-10-04: queued=0 in
    94/130 heartbeats). Counting in-flight supply and capping per-tick starts staggers starts continuously."""
    n, cap = 0, max(2, -(-max_concurrent // 4))
    while n < cap and queued + 0.75 * (in_flight + n) < target and queued + in_flight + n < qmax \
            and in_flight + n < max_concurrent:
        n += 1
    return n


class RunBook(R.RunBook):
    """The old lanes' run routing on top of :class:`loop.runner.RunBook` (the plumbing): importance from the
    snapshot / design points / tuning controls / leg and campaign stragglers, the controls' half of primary, scarce
    mode, and the old OOM / cancel eligibility. Shared by the driver and calibrate."""

    def __init__(self, ctx: LoopContext, store: Any, adapter: ExecutionAdapter, *,
                 clock: Callable[[], float] = time.time) -> None:
        super().__init__(ctx, store, adapter, clock=clock)
        self.p_positive: Callable[[str], float] = lambda _nid: 0.0     # the driver wires its snapshot

    def high(self, node: Node | None, stage: Stage | None, seed: int | None = None) -> bool:
        """HIGH importance: every non-screen run (controls, calibrate, promote, confirm, audit), a
        stacked design node's screens, and screens of a node with P(U>0) >= HIGH_P. LOW: other
        screens, design points, tuning controls no trial waits on yet (and smoke jobs). A leg-factorial point among
        its design's last block of open points is HIGH (loop.legs.stragglers: the analysis waits on it)."""
        if self._leg_straggler(node):
            return True
        if node is not None and (design_point(node) or self._low_control(node, stage, seed)):
            return False
        if stage is not Stage.SCREEN:
            return True
        return node is not None and (node.strategy is Strategy.DESIGN or self.p_positive(node.id) >= HIGH_P)

    def pool(self, node: Node | None = None, stage: Stage | None = None, seed: int | None = None) -> str:
        """Where a run goes. HIGH: a primary (non-preemptible) pool, else a preemptible one, else the
        first. LOW: a preemptible pool, else primary only while no HIGH run waits ('' = it waits). A
        design point runs only on a preemptible pool ('' = none free: it never takes a primary slot), and a
        designed batch's points hold at most ``design.max_share`` of the slots (tuning trials excepted). An
        eval-only run (loop.evalonly: a ~23 min refit + probe on a trained checkpoint, launched as PROMOTE /
        CONFIRM / AUDIT) runs only on a preemptible pool ('' = none free)."""
        pools = self.ctx.cfg.pools
        if seed is not None and seed >= EO.EO_SEED and any(p.preemptible for p in pools):
            return next(iter(self._open(True)), "")
        if node and self._batch_point(node) and (self.scarce() or sum(
                self._batch_point(self.store.get_node(r.node_id)) for r in self.active()) >= self._design_cap()):
            return ""                                   # scarce mode pauses design batches
        if not any(p.preemptible for p in pools):
            return next(iter(self._open(False)), pools[0].name)
        # design points (but a leg factorial's stragglers) and tuning controls (loop.tune): preemptible only
        if node and (design_point(node) and not self._leg_straggler(node) or self._low_control(node, stage, seed)):
            return next(iter(self._open(True)), "")
        if self.high(node, stage, seed):
            if stage in CONTROL_ONLY and not self.serves_high(node, seed) \
                    and any(self.controls_capped(p.name) for p in pools if not p.preemptible):
                prim = [p for p in self._open(False) if not self.controls_capped(p)]   # primary's candidate
                return next(iter(prim + self._open(True)), "")                        # half: idle, else wait
            return next(iter(self._open(False) + self._open(True)), pools[0].name)
        return next(iter(self._open(True) + ([] if self.high_demand else self._open(False))), "")

    def serves_high(self, node: Node | None, seed: int | None) -> bool:
        """A control run at a confirm-pool seed while a child of ``node`` is CONFIRMING: it pairs a HIGH confirm run
        and is placed like one (never held to the controls' half of primary; live: 0 GPU-h reached confirm)."""
        if node is None or seed is None or int(seed) not in {int(s) for s in (*self.ctx.cfg.seeds.confirm_pool,
                                                                              *S.STACK_SEEDS)} | RM.confirm_seeds(
                                                                                  self.store, node.id):
            return False
        return any((n.parent_id or ROOT_NODE_ID) == node.id and n.id != node.id
                   for n in self.store.nodes(status=[NodeStatus.CONFIRMING]))

    def controls_capped(self, pool: str) -> bool:
        """Control runs (parent / baseline) hold >= max(1, slots // 2) of non-preemptible ``pool`` while candidate
        runs wait PENDING elsewhere: the rest is reserved for candidates (live: primary held 8/8 controls while
        every candidate screen starved on idle)."""
        cfg = next((p for p in self.ctx.cfg.pools if p.name == pool), None)
        if cfg is None or cfg.preemptible:
            return False
        first, runs = self.ctx.cfg.pools[0].name, self.active()
        if not any(r.stage not in CONTROL_ONLY and (r.pool or first) != pool and r.id in self.pending for r in runs):
            return False
        return sum((r.pool or first) == pool and r.stage in CONTROL_ONLY for r in runs) >= max(1, cfg.slots // 2)

    def _low_control(self, node: Node, stage: Stage | None, seed: int | None) -> bool:
        """A tuning control (:func:`tune_control`) no trial waits on yet: once a trial finished its own run on
        ``seed``, the control at that seed blocks the trial's verdict and is HIGH (live: a node:parent:390 at nice
        100, 6 restarts, pending while a node had both seeds done)."""
        if not tune_control(node, stage):
            return False
        kids = {n.id for n in self.store.nodes() if n.parent_id == node.id}
        return seed is None or not any(r.seed == int(seed) and r.node_id in kids
                                       for r in self.store.runs(status=[RunStatus.DONE]))

    @staticmethod
    def _batch_point(node: Node | None) -> bool:
        """A designed batch's point held to ``design.max_share`` and paused in scarce mode: not a tuning trial, not a
        leg-factorial point (loop.legs: the main stacking path; live L1 waited 3 h behind batch 7's points for the
        shared 9-slot share, then trickled ~9 at a time for 7.5 h)."""
        return node is not None and design_point(node) and " tune=" not in f" {node.notes or ''}" \
            and not LG.leg_point(node)

    def _leg_straggler(self, node: Node | None) -> bool:
        """A leg factorial's or a campaign round's straggler (loop.legs / loop.campaign ``stragglers``): HIGH."""
        return LG.leg_point(node) and node.id in LG.stragglers(self.store) \
            or CP.campaign_point(node) and node.id in CP.stragglers(self.store)

    def _design_cap(self) -> int:
        return max(1, int(self.ctx.cfg.design.max_share * self.ctx.cfg.capacity))

    def promote_stuck(self) -> list[str]:
        """A HIGH run SLURM still holds PENDING on a preemptible pool moves to a free primary slot:
        scancel + the same launch there (a free retry: same row and attempt, no infra_retry). One SLURM
        already requeued ``slurm.repool_restarts`` times moves to primary even when it is full (it queues
        there next instead of being preempted again; at most :meth:`_overflow_cap` of ours queue there) -- on the requeue event itself (``repool_due``), not only
        if a poll happens to catch it PENDING (live: a HIGH screen requeued 15x stayed on idle). Any run, LOW
        too, requeued ``slurm.repool_restarts_any`` times moves likewise (live: a LOW design point 37x) -- but a
        LOW run only into a FREE primary slot: queued PENDING on a full primary it held a slot (AssocGrpGRES)."""
        prim = next((p.name for p in self.ctx.cfg.pools if not p.preemptible), "")
        repool, out = getattr(self.adapter, "repool", None), []
        pre = {p.name for p in self.ctx.cfg.pools if p.preemptible}
        stuck, any_n = [], self.ctx.cfg.slurm.repool_restarts_any
        starved = {p for p in pre if self.starved(p)}
        for run in self.store.runs(status=[RunStatus.RUNNING]) if repool and pre else ():
            node = self.store.get_node(run.node_id)
            hi = self.high(node, run.stage, run.seed)
            # a LOW screen pending on a STARVED pool moves too (live: 41/43 jobs PENDING, idle ran none) -- never a
            # design point: a batch's or a tuning pass's (``tune=``) trials stay preemptible-only (live: t10 / t12
            # trials a node / a node repooled onto primary)
            low_screen = run.pool in starved and run.id in self.pending and run.stage is Stage.SCREEN \
                and not design_point(node)
            if (run.id in self.pending or run.id in self.repool_due) and run.pool in pre and not EO.is_eval_run(run) \
                    and (hi or low_screen or self.restarts.get(run.id, 0) >= any_n > 0):
                stuck.append(((_TIER.get(run.stage, 3), not self._controls_done(node, run), -self.p_positive(run.node_id),
                               run.submitted_ts or 0.0), run, hi))
        first = self.ctx.cfg.pools[0].name
        backlog = sum((r.pool or first) == prim and r.id in self.pending for r in self.active()) if prim else 0
        for _, run, hi in sorted(stuck, key=lambda x: x[0]):
            again = hi and self.restarts.get(run.id, 0) >= self.ctx.cfg.slurm.repool_restarts > 0
            short = run.pool in starved and prim and self._primary_queue_short(prim, run.pool)
            # a move, not a new job (our pending count is unchanged): a free slot ignores max_pending_per_pool. Onto a
            # FULL primary only while its own queue is under the overflow cap, moves this pass counted (live 10-08: one
            # preemption wave requeued 8 HIGH screens at once -> primary 16/8, 8 pending AssocGrpGRES behind 8 running
            # for a run length; the rest stay pending on idle, still due, and move as primary's queue drains)
            free_dest = next(iter(self._open(False, gate=False)), "")
            over = bool(prim) and (again or short) and backlog < self._overflow_cap(prim)
            if not (dest := free_dest or (prim if over else "")) \
                    or run.stage in CONTROL_ONLY and self.controls_capped(dest) \
                    and not self.serves_high(node, run.seed):                  # the candidates' half
                continue
            if (line := self._move(run, dest, hi, again)) is not None:
                backlog += not free_dest                 # it queues on the full primary
                out.append(line)
        return out

    def _controls_done(self, node: Node | None, run: Run) -> bool:
        """A screen's block control (its parent's control run at the same seed) is DONE: its verdict is one run
        away (moved first)."""
        if node is None or run.stage is not Stage.SCREEN:
            return True
        return any(r.seed == run.seed and r.stage in S.CONTROL_STAGES
                   for r in self.store.runs(node_id=node.parent_id or ROOT_NODE_ID, status=[RunStatus.DONE]))

    def _cancel_keeps(self, node_id: str) -> Callable[[Run], bool]:
        """A PARENT-stage run an active child still pairs against is its control, never cancelled (cancelling it
        voided the child: void:partner)."""
        kids = any(n.parent_id == node_id for n in self.store.nodes(status=ACTIVE_STATUSES))
        return lambda run: kids and run.stage is Stage.PARENT

    def _oom_ok(self, run: Run, node: Node) -> bool:
        return node.id == ROOT_NODE_ID or run.stage in S.CONTROL_STAGES or node.status in ACTIVE_STATUSES

    def poll(self) -> list[Run]:
        finished = super().poll()
        self._scarce_step()
        return finished

    def scarce(self) -> bool:
        """SCARCE MODE in force (K_SCARCE, set at each poll by :meth:`_scarce_step`)."""
        return bool((self.store.get_state(K_SCARCE) or {}).get("on"))

    def _scarce_step(self) -> None:
        """Sample the GPUs SLURM runs for us (at most every SCARCE_SAMPLE_S, kept SCARCE_WINDOW_S); SCARCE MODE is on
        while their mean is below ``slurm.scarce_gpus`` or a preemptible pool is STARVED (0 = off). Logs
        scarce_on / scarce_off on a change."""
        now, thr = self.clock(), float(self.ctx.cfg.slurm.scarce_gpus)
        was = self.store.get_state(K_SCARCE) or {}
        samples = [s for s in was.get("samples", []) if now - float(s[0]) <= SCARCE_WINDOW_S]
        if not samples or now - float(samples[-1][0]) >= SCARCE_SAMPLE_S:
            samples.append([now, sum(r for r, _ in (self.pool_states or {}).values())])
        avg = sum(float(n) for _, n in samples) / len(samples)
        starved = sorted(p.name for p in self.ctx.cfg.pools if p.preemptible and self.starved(p.name))
        on = thr > 0 and (avg < thr or bool(starved))
        st = {"on": on, "since": was.get("since") if on == bool(was.get("on")) else now, "avg": round(avg, 3),
              "starved": starved, "samples": samples}
        if on != bool(was.get("on")):
            self.store.log("scarce_on" if on else "scarce_off", avg_gpus=round(avg, 3), threshold=thr, starved=starved)
            self.acts.append(f"scarce mode {'ON' if on else 'off'}: {avg:.1f} running GPUs avg over "
                             f"{SCARCE_WINDOW_S / 3600:.0f} h (threshold {thr:g})"
                             + (f", {','.join(starved)} STARVED" if starved else "")
                             + ("; design batches and tuning paused, new screens on <= 2 carriers" if on else ""))
        if st != was:
            self.store.set_state(K_SCARCE, st)


class Driver:
    def __init__(self, ctx: LoopContext, store: Any, adapter: ExecutionAdapter, world: Any,
                 selector: Any, researcher: Any, analyst: Any, *, scout: Any = None,
                 clock: Callable[[], float] = time.time,
                 executor: Callable[..., Any] = ThreadPoolExecutor,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        if store.task_id != ctx.task_id:
            raise ValueError(f"store is for task {store.task_id!r}, context for {ctx.task_id!r}")
        self.ctx, self.store, self._spec_cfg = ctx, store, ctx.cfg
        ctx.spec_cfg = ctx.cfg
        self.adapter, self.world, self.selector = adapter, world, selector
        self.researcher, self.analyst = researcher, analyst
        self.clock, self.sleep, self._reaped, self._restored = clock, sleep, False, False
        self.book = RunBook(ctx, store, adapter, clock=clock)
        self.book.p_positive = lambda nid: (p.p_gate if self.snapshot is not None
                                            and (p := self.snapshot.nodes.get(nid)) else 0.0)
        self.courtesy = Courtesy(ctx, store, adapter, self.book, clock)
        self._gate_backfilled = False
        self._ds_cache: dict[str, str] = {}     # node id -> its data.dataset_dir (loop.databank.gate)
        self._db_lines: list[str] = []          # dataset_spec requests' outcomes (printed with the intake)
        if hasattr(adapter, "pool_picker"):           # smoke jobs are placed (and counted) like runs
            adapter.pool_picker = self.book.smoke_pool
        self.pool = executor(max_workers=max(self.cfg.researcher.max_concurrent, T.MAX_CONCURRENT) + 2)
        self.snapshot: WorldSnapshot | None = None
        self._dirty = True
        self._research: dict[Future, CandidateRequest] = {}
        self._restarts: dict[str, int] = {}     # request id -> times re-issued after a restart
        self._requeue: list[CandidateRequest] = []  # restart re-issues waiting for a slot under max_concurrent
        self._draining = False                  # stop requested: no new LLM job while in-flight research finishes
        self._smoke_reissues: dict[str, int] = {}   # request id -> times re-issued after a smoke queue timeout
        self._research_paused = False               # RunBook.saturated() held new researchers at the last fill
        self._headroom_seen: list[tuple[float, int]] = []   # (ts, RunBook.headroom()) over one research lead time
        self._side: set[str] = set()            # request ids from the driver's own sources (researcher.max_side)
        self._stack: set[str] = set()           # ... of its STACKING sources (researcher.max_stack, not max_side)
        self._review: tuple[Future, str, dict] | None = None
        self._smoke: Future | None = None       # a designed batch's heaviest-point memory smoke
        self._smoke_swept: float | None = None  # when orphan smoke jobs were last scancelled
        self.scout = scout                       # loop.scout.Scout (None: no research scout)
        self._scout: tuple[Future, str, Any] | None = None
        self._consolidation: tuple[Future, str] | None = None   # (job, the evidence it saw: loop.redundancy)
        self._oom_retro = False
        self._fails: dict[str, int] = {}
        self.labless = LablessSubmitter(ctx, store, self.pool, clock)
        self.evalonly = EO.EvalOnly(ctx, store, adapter, self.book, self._new_id, clock)
        self.pruner = PR.Pruner(ctx, store, clock, adapter, background=True)   # off the tick: minutes of git
        self._tune()

    @property
    def cfg(self) -> Any:
        return self.ctx.cfg

    def _tune(self) -> None:
        """Effective config = spec + self-tune overrides, installed on the shared context (every
        component reads ``ctx.cfg``)."""
        if (eff := T.effective(self._spec_cfg, self.store)) != self.ctx.cfg:
            self.ctx.cfg = eff

    # ------------------------------------------------------------------ public
    def tick(self, dry_run: bool = False) -> list[str]:
        if dry_run:
            return self._plan_only()
        acts: list[str] = []
        self._tune()
        if not self._reaped:                        # once per process: no researcher is alive yet
            acts += [f"reaped orphan worktree {p}" for p in self.adapter.reap_orphans()]
            self._reaped = True
            if hasattr(self.adapter, "base_config"):   # intake compares EFFECTIVE configs against it
                try:
                    if base := self.adapter.base_config():
                        self.store.set_state(IN.K_BASE, base)
                except Exception as exc:  # noqa: BLE001 - intake falls back to the diffs' removed lines
                    self.store.log("base_config_failed", error=repr(exc)[:300])
        if self.store.get_node(ROOT_NODE_ID) is None:
            acts.append(f"root node at {self.store.ensure_root(self.adapter.root_commit()).commit}")
        if not self._restored:                      # once per process, before K_INFLIGHT is rewritten
            self._restored = True
            self._restore_inflight(acts)
            self._repair_smoke_timeouts(acts)
            self._repair_outage(acts)
        self._sweep_smokes(acts)
        self._harvest(acts)
        self._llm_gate(acts)
        self._measure()
        self._sync_inflight()
        if not self._oom_retro:                     # once per process (deploy): OOM crashes before the ladder
            self._oom_retro = True
            self._oom_reopen(acts)
        acts += RS.tick(self.ctx, self.store, self.book, self.adapter)     # before any (re)submit: resumable runs
        acts += self.book.resolve_submitting()
        finished = self.book.poll()
        acts += self.book.acts
        self.book.acts = []
        acts += [f"run {r.id} -> {r.status.value}" + (f" ({r.void_type.value})" if r.void_type else "")
                 for r in finished]
        before = self.snapshot
        if finished or self._dirty or self.snapshot is None:
            self._refit()
        if self.snapshot is not None and not self.store.get_state(K_REGATE):
            self._regate(acts)
        if self.snapshot is not None and not self.store.get_state(K_REGATE_OFFSET):
            rise = WN.offset_rise(self.ctx, WN.read_ledger(self.ctx, self.store), self.snapshot.sigma)
            self._regate(acts, K_REGATE_OFFSET, "robust control offsets", lambda nid: rise.get(nid, 0.0) > 1e-9)
        if self.snapshot is not None and self.store.get_state(K_REINTAKE) != REINTAKE_V:   # True: v1
            self._reintake(acts)
        if not self.store.get_state(K_RELABEL):
            self._relabel_cancels(acts)
        if self.snapshot is not None and not self.store.get_state(K_REMEASURE):
            self._remeasure(acts)
        if self.snapshot is not None and not self.store.get_state(K_REFREEZE):
            self._refreeze(acts)
        if self.snapshot is not None and not self.store.get_state(K_TUNE_GATE):
            self._tune_gate(acts)
        if getattr(self.snapshot, "wm", None) is not None and self.store.get_state(K_READMIT) is None:
            self._readmit_low_value(acts)
        if not self.store.get_state(K_HIT):         # once: a hit is now an OWN increment > 0 (loop.meta._Facts.hit)
            acts += [x for x in [ML.retire_playbook(self.store, self.clock(), "hit redefined as own increment > 0")] if x]
            self.store.set_state(K_HIT, True)
        self._epoch(acts)
        if ks := SH.tick(self.ctx, self.store, self.adapter, self.snapshot, new_id=self._new_id, now=self.clock()):
            acts += ks                                 # an incumbent change: its knob shift, re-based queued nodes
            self._dirty = True
        if bf := BF.tick(self.ctx, self.store, self.adapter, self.snapshot, new_id=self._new_id, now=self.clock(),
                         reopen=lambda n, st, why: self._set_status(n, st, why, acts)):
            acts += bf
            self._refit()
        self._transitions(acts, before)
        self._mech_pool(acts)
        acts += ML.mark_stale(self.ctx, self.store, self.snapshot, now=self.clock())
        self._triggers(acts)
        self._meta_jobs(acts)
        scarce = self.book.scarce()                 # scarce mode: no new design batch / tuning pass / trial
        dlane = scarce or not self._lane_gate(       # a new design batch only when the model expects value
            "design", [n for n in self.store.nodes() if design_point(n)],
            str(len((self.store.get_state(D.K_DESIGN) or {}).get("batches") or [])))
        if design := D.tick(self.ctx, self.store, self.adapter, self.snapshot, evals=self._n_evals(),
                            evals_at_confirm=int(self.store.get_state(K_LAST_CONFIRM, 0)), new_id=self._new_id,
                            paused=dlane):
            acts += design
            self._dirty = True
        self._design_smoke(acts)
        if db := DB.tick(self.ctx, self.store, self.adapter, self.snapshot, new_id=self._new_id, now=self.clock()):
            acts += db                                 # the data bank: register / build / adopt
            self._dirty = True
        if tuned := MT.tick(self.ctx, self.store, self.adapter, self.snapshot, new_id=self._new_id,
                            now=self.clock(), inflight=[r.id for r in self._research.values()], scarce=scarce):
            acts += tuned
            self._dirty = True
        if var := VR.tick(self.ctx, self.store, self.adapter, self.snapshot, new_id=self._new_id, now=self.clock(),
                          scarce=scarce):                      # multi-variant ideas + their branched plans
            acts += var
            self._dirty = True
        if camp := CP.tick(self.ctx, self.store, self.adapter, self.snapshot, new_id=self._new_id, now=self.clock(),
                           scarce=scarce):                     # difficult-tune campaigns (computed rounds)
            acts += camp
            self._dirty = True
        if ext := EX.tick(self.ctx, self.store, self.adapter, self.snapshot, new_id=self._new_id, now=self.clock(),
                          scarce=scarce):                      # follow the winning directions on the incumbent
            acts += ext
            self._dirty = True
        if legs :=LG.tick(self.ctx, self.store, self.adapter, self.snapshot, new_id=self._new_id, now=self.clock(),
                           scarce=scarce, inflight=[r.id for r in self._research.values()]):
            acts += legs
            self._dirty = True
        if regrad := GR.once(self.ctx, self.store, self.adapter, self.snapshot, new_id=self._new_id, now=self.clock()):
            acts += regrad                             # one-shot: the shrunk-graduation rule on the last 48 h
            self._dirty = True
        self._courtesy(acts)
        if lane := self.evalonly.tick(self.snapshot):      # readout-only candidates: scored on trained checkpoints
            acts += lane
            self._dirty = True
        first = self._audit(acts)
        rm, rm_needs = RM.tick(self.ctx, self.store, self.snapshot, self.book.infra_retry, now=self.clock())
        acts += rm
        self._fill(acts, first=[*first, *rm_needs])
        acts += self.labless.step(self._audit_book()["passed"], self.snapshot)
        if self.ctx.cfg.diagnostics.enabled:
            try:
                if n := DG.ingest(self.ctx, self.store):
                    self.store.log("diagnostics_ingested", n=n)
                acts += DG.bank(self.ctx, self.store, self.snapshot)   # computed bank ideas from the incumbent's flags
            except Exception as exc:  # noqa: BLE001
                self.store.log("diagnostics_error", error=repr(exc)[:500])
        acts += self.pruner.tick(self.snapshot)
        acts += DJ.tick(self.ctx, self.store, self.clock())     # <state_dir>/digest.md once a day
        return acts + CG.take_lines()

    def run(self, max_ticks: int | None = None, stop: Callable[[], bool] = lambda: False) -> None:
        """Tick until ``max_ticks`` or ``stop()``; a stop lands between ticks (jobs keep running). With research in
        flight a stop DRAINS first: no new LLM job, ticks go on (harvest, intake, polls) until every session is
        harvested or ``researcher.drain_min`` passed; what is left is re-issued after the restart (queued)."""
        n, drain_until = 0, None
        try:
            next_hb = self.clock()
            while max_ticks is None or n < max_ticks:
                if stop() and drain_until is None:
                    if not self._research or self.cfg.researcher.drain_min <= 0:
                        break
                    self._draining, drain_until = True, self.clock() + 60.0 * self.cfg.researcher.drain_min
                    print(f"{utc_now(self.clock())} stop requested: draining {len(self._research)} research "
                          f"session(s) for <= {self.cfg.researcher.drain_min:g} min; no new LLM job", flush=True)
                if drain_until is not None and (not self._research or self.clock() >= drain_until):
                    print(f"{utc_now(self.clock())} drain done: {len(self._research)} research session(s) left "
                          "(re-issued after the restart)", flush=True)
                    break
                for line in self.tick():
                    print(f"{utc_now(self.clock())} {line}", flush=True)
                if self.cfg.dashboard.every_ticks > 0 and n % self.cfg.dashboard.every_ticks == 0:
                    self.write_dashboard()
                if self.cfg.heartbeat.log_every_min > 0 and self.clock() >= next_hb:
                    print(utc_now(self.clock()), hb_line(self.ctx, self.store, "loop", self.snapshot, len(self._research)), flush=True)
                    next_hb = self.clock() + 60 * self.cfg.heartbeat.log_every_min
                n += 1
                if max_ticks is None or n < max_ticks:      # draining: stop() stays set, so nap on its own clock
                    nap(min(self.cfg.heartbeat.poll_s, 15.0) if drain_until is not None else self.cfg.heartbeat.poll_s,
                        (lambda: False) if drain_until is not None else stop, self.sleep)
        finally:
            self.pruner.stop()                       # a background prune pass ends between git commands
            self.pool.shutdown(wait=False, cancel_futures=True)

    def write_dashboard(self) -> None:
        """<state_dir>/dashboard/index.html from the tick's already-fitted snapshot (atomic; never
        fails the loop)."""
        try:
            dashboard.write(self.ctx, self.store, self.snapshot)
        except Exception as exc:  # noqa: BLE001 - a page is not worth a tick
            print(f"{utc_now(self.clock())} dashboard write failed: {type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)

    def drain(self) -> None:
        """Wait for in-flight LLM work (tests / clean shutdown); never kills it."""
        for fut in [*self._research, *([self._review[0]] if self._review else []),
                    *([self._scout[0]] if self._scout else []),
                    *([self._consolidation[0]] if self._consolidation else [])]:
            try:
                fut.result()
            except Exception:  # noqa: BLE001 - reported when harvested
                pass

    def labless_ready(self, node_id: str) -> bool:
        """Only a node that passed the held-out audit may go to Labless; asking queues one (every
        tick asks about the incumbent, so a newly confirmed incumbent is audited at once)."""
        a = self._audit_book()
        if node_id in a["passed"]:
            return True
        if node_id not in a["failed"] + a["queue"]:
            a["queue"].append(node_id)
            self.store.set_state(K_AUDIT, a)
        return False

    # --------------------------------------------------------------- plumbing
    def _refit(self) -> WorldSnapshot:
        metas = self.store.hypotheses(kind=HypothesisKind.META)       # D3.9: auto-link trials
        index = LinkIndex(self.store) if metas else None
        for h in metas:
            link_meta(self.store, h, index)
        self.snapshot, self._dirty = self.world.fit(self.store), False
        return self.snapshot

    def _sigma_p(self) -> float:
        return float(self.snapshot.sigma.get(PANEL, self.cfg.panel_sd)) if self.snapshot \
            else self.cfg.panel_sd

    def _runs_by_node(self) -> dict[str, list[Run]]:
        out: dict[str, list[Run]] = {}
        for r in self.store.runs():
            out.setdefault(r.node_id, []).append(r)
        return out

    def _decide(self, node: Node, by_node: Mapping[str, list[Run]],
                blocks: Mapping[str, int]) -> S.Decision:
        snap = self.snapshot
        own = by_node.get(node.id, [])
        confirm = None
        if node.status is NodeStatus.CONFIRMING and snap is not None and any(
                r.stage is Stage.CONFIRM and r.status is RunStatus.DONE for r in own):
            confirm = self.world.confirm_evalue(snap, node.id, self.store)
        rot, fresh = 0, RM.confirm_block(self.ctx, self.store, node, own, self.clock())
        if fresh is not None:                        # a fresh confirm block (loop.remint.confirm_block)
            rot = fresh[1]
        elif node.status is NodeStatus.CONFIRMING:   # siblings on one control confirm on different seeds
            sibs = sorted((n.created_ts, n.id) for n in self.store.nodes()
                          if n.parent_id == node.parent_id and n.confirm_alt is not None)
            rot = next((i for i, (_, nid) in enumerate(sibs) if nid == node.id), 0)
        row = (self.store.get_state(NOISE_KEY) or {}).get(PANEL)
        return S.step(node, own_runs=own, control_runs=by_node.get(S.control_of(node)[0], []),
                      post=snap.nodes.get(node.id) if snap else None, cfg=self.cfg,
                      sigma_panel=self._sigma_p(), infra_retry=self.book.infra_retry,
                      confirm=confirm, screen=(int(one),) if (one := LG._note(node, "seed1"))   # a leg-factorial point
                      else S.screen_block(self.cfg.seeds, blocks.get(node.id, 0)),
                      promote=S.promote_block(self.cfg.seeds, blocks.get(node.id, 0)), confirm_rot=rot,
                      sigma_ucb=sigma_ucb(row["sigma"], row["df_eff"]) if row else None,
                      frozen=(self.store.get_state(CONFIRM_KEY) or {}).get(node.id) if confirm else None,
                      confirm_pool=fresh[0] if fresh is not None else self._stack_pool(node, by_node),
                      raw=raw[1] if snap and (raw := snap.eb_raw.get(node.id)) else None,
                      extra=BF.extra_confirm(self.store, node.id))

    def _stack_pool(self, node: Node, by_node: Mapping[str, list[Run]]) -> list[int] | None:
        """A stack's confirm seeds (its legs in ``stack_legs=``): the pool and S.STACK_SEEDS less every seed a leg
        ran a scored run on -- its verdict never rests on a seed a leg was selected on; None for any other node."""
        if not (m := re.search(r"\bstack_legs=(\S+)", node.notes or "")):
            return None
        used = {r.seed for leg in m.group(1).split(",") for r in by_node.get(leg, ())
                if r.stage in (Stage.SCREEN, Stage.PROMOTE, Stage.CONFIRM)}
        return [int(s) for s in (*self.cfg.seeds.confirm_pool, *S.STACK_SEEDS) if int(s) not in used]

    def _backend(self) -> str:
        """The researcher's LLM backend (quality metrics group by it); '?' when unknown."""
        tier = getattr(getattr(self.researcher, "llm", None), "tier", None)
        try:
            return str(tier("researcher")) if callable(tier) else "?"
        except Exception:  # noqa: BLE001
            return "?"

    def _n_evals(self) -> int:
        return len(self.store.nodes(status=_EVAL_CLOSES))

    def _set_status(self, node: Node, status: NodeStatus, reason: str, acts: list[str],
                    **fields: Any) -> Node:
        new = self.store.update_node(node.id, status=status, **fields)
        self.store.log("node_transition", node_id=node.id, frm=node.status.value,
                       to=status.value, reason=reason)
        acts.append(f"node {node.id}: {node.status.value} -> {status.value} ({reason})")
        self._dirty = True
        return new

    def _value_gate(self, node: Node, acts: list[str]) -> bool:
        """The SCREEN value gate: a queued node whose predicted P(fresh own gain > MDE) (the predictive model, from its
        creation-time features) is under ``prior.wm_value_p`` is REJECTED ``low_value`` instead of screened, except on
        the exploration slot (``prior.wm_explore``, deterministic per node). Side lanes with their own budgets
        (design points, leg-factorial and tuning nodes) pass, and so does a node the model cannot judge
        (:meth:`_gate_exempt`)."""
        wm = getattr(self.snapshot, "wm", None) if self.snapshot is not None else None
        notes = f" {node.notes or ''}"
        if wm is None or design_point(node) or " legs_of=" in notes or " legs=" in notes or " tune_of=" in notes:
            return True
        if exempt := self._gate_exempt(wm, node):
            self.store.log("wm_gate", node_id=node.id, lane="screen", passed=True, why=exempt)
            return True
        ok, why = WM.gate_value(wm, self.cfg, node.id, wm.value(node))
        self.store.log("wm_gate", node_id=node.id, lane="screen", passed=ok, why=why)
        if not ok:
            self._set_status(node, NodeStatus.REJECTED, f"value gate: {why}", acts, reject_reason="low_value")
        return ok

    def _gate_exempt(self, wm: Any, node: Node) -> str:
        """Why the screen value gate does not judge ``node`` ("" when it does): a moonshot or an init swap (novel by
        intent), or thin predictor support (< ``prior.wm_gate_min_support`` scored nodes share its family, mechanism
        type or init) -- the fit extrapolates to ~0 there; the gate prunes near-duplicates of well-measured ones."""
        if node.strategy is Strategy.MOONSHOT and not re.search(r"(?:^|\s)brief=bold(?:\s|$)", node.notes or ""):
            return "exempt: moonshot"      # v3 review S2: a BOLD node is gated like any idea (thin support still exempts it)
        if re.search(r"(?:^|\s)init=\S", node.notes or ""):
            return "exempt: init swap"
        k = getattr(self.cfg.prior, "wm_gate_min_support", 0)
        n = wm.support(node) if k > 0 and hasattr(wm, "support") else k
        return f"exempt: thin support ({n} < {k} scored nodes share its family / mechanism type / init)" if n < k else ""

    def _readmit_low_value(self, acts: list[str]) -> None:
        """Once: every node the screen gate rejected ``low_value`` since READMIT_SINCE that :meth:`_gate_exempt` now
        passes is QUEUED again (moonshots / init swaps / thin-support nodes the old gate killed, e.g. a node)."""
        wm = getattr(self.snapshot, "wm", None)
        back = []
        for node in self.store.nodes(status=[NodeStatus.REJECTED]):
            if node.reject_reason == "low_value" and node.updated_ts >= READMIT_SINCE and wm is not None \
                    and (why := self._gate_exempt(wm, node)):
                self._set_status(node, NodeStatus.QUEUED, f"readmitted: value gate {why}", acts, reject_reason="")
                back.append(node.id)
        self.store.log("readmit_low_value", nodes=back, since=READMIT_SINCE)
        self.store.set_state(K_READMIT, {"nodes": back})

    def _lane_gate(self, lane: str, members: Sequence[Node], key: str) -> bool:
        """A side lane (a design batch, ...) starts a new round only when the predictive model expects one more of its
        nodes, built on the incumbent, to be worth it (:meth:`loop.world.predictive.Fit.lane_value`), or on the
        exploration slot for this round ``key``. Logged when the verdict changes."""
        wm = getattr(self.snapshot, "wm", None) if self.snapshot is not None else None
        if wm is None:
            return True
        p, n = wm.lane_value(members, parent=self.snapshot.incumbent_id)
        ok, why = WM.gate_value(wm, self.cfg, f"{lane}:{key}", p) if n else (True, "no history")
        st = self.store.get_state("wm.lane_gate") or {}
        if st.get(lane) != ok:
            self.store.set_state("wm.lane_gate", {**st, lane: ok})
            self.store.log("wm_gate", lane=lane, passed=ok, why=why, n=n, key=key)
        return ok

    def _reintake(self, acts: list[str]) -> None:
        """Once per REINTAKE_V (a change to the intake rules): every QUEUED node is re-checked by the config rules
        (sweep_only / below_noise against its nearest relative, near_dup_config) and the mechanism rule, oldest first; one that
        fails is rejected with the rule as its reason (a sweep's knobs go to the design pool)."""
        for node in sorted(self.store.nodes(status=[NodeStatus.QUEUED]), key=lambda n: (n.created_ts, n.id)):
            rule, fix = IN.recheck(self.ctx, self.store, self.snapshot, node, forward=True)
            if not rule:
                continue
            self._set_status(node, NodeStatus.REJECTED, f"reintake {rule}: {fix[:200]}", acts, reject_reason=rule)
            self.store.log("intake", node_id=node.id, request_id="reintake", admitted=False, reason=rule)
        self.store.set_state(K_REINTAKE, REINTAKE_V)

    def _relabel_cancels(self, acts: list[str]) -> None:
        """Once: an infra void recorded AFTER its node closed was the loop's own cancel (a close / regate
        scancels the node's open runs; before void/cancelled existed the next poll called it infra)."""
        closed = {}
        for e in self.store.events("node_transition"):
            if e.get("to") in {s.value for s in CLOSED_STATUSES}:
                closed[e.get("node_id")] = e["ts"]
        fin = {e.get("run_id"): e["ts"] for e in self.store.events("run_finished")}
        n = 0
        for run in self.store.runs(status=[RunStatus.VOID]):
            node = self.store.get_node(run.node_id)
            if run.void_type is VoidType.INFRA and node is not None and node.status in CLOSED_STATUSES \
                    and run.node_id in closed and fin.get(run.id, -1.0) >= closed[run.node_id]:
                self.store.update_run(run.id, void_type=VoidType.CANCELLED)
                n += 1
        if n:
            self.store.log("relabel_cancels", n=n)
            acts.append(f"relabelled {n} past infra void(s) as cancelled (the loop's own cancels)")
        self.store.set_state(K_RELABEL, True)

    def _dup_of(self, node: Node, nodes: Sequence[Node]) -> Node | None:
        """An earlier node with the same commit / commit tree (whatever its parent): ``node`` is a copy of it."""
        trees = self.store.get_state(IN.K_TREES) or {}
        mine = trees.get(node.commit)
        return next((m for m in nodes if (m.created_ts, m.id) < (node.created_ts, node.id) and m.id != ROOT_NODE_ID
                     and not design_point(m) and m.status is not NodeStatus.REJECTED
                     and (m.commit == node.commit or mine and trees.get(m.commit) == mine)), None)

    def _remeasure(self, acts: list[str]) -> None:
        """Once (own increments re-measured against each node's nearest contained ancestor): a QUEUED copy of an
        earlier node is rejected exact_dup; a live copy is screened out; a PROMOTING / CONFIRMING node below its
        gate on the re-measured increment is screened out (as _regate); a CONFIRMED node that is a copy, or whose
        confirm E on its own increment is now below 1/alpha_confirm, is NOT_CONFIRMED -- its runs (the held-out
        audit too) cancelled, its audit and unsent Labless entries dropped, its bandit credit re-paid at the new
        status and the stall clock (K_LAST_CONFIRM) put back to the last remaining confirmation. Then the
        measurement epoch is bumped (stale reviews / playbook / overrides, see loop.meta.on_epoch)."""
        snap, pr = self.snapshot, self.cfg.promotion
        nodes = sorted(self.store.nodes(), key=lambda n: (n.created_ts, n.id))
        live = (NodeStatus.QUEUED, NodeStatus.SCREENING, NodeStatus.PROMOTING, NodeStatus.CONFIRMING, NodeStatus.CONFIRMED)
        demoted, changed = [], []
        for node in [n for n in nodes if n.status in live]:
            dup, meas, post = self._dup_of(node, nodes), snap.measured.get(node.id), snap.nodes.get(node.id)
            if dup is None and (meas is None or post is None):
                continue
            why = f"duplicate of {dup.id} (same commit tree)" if dup else \
                f"own increment {post.increment.mean:+.4f} over {meas[0]} (contained ancestor)"
            if node.status is NodeStatus.QUEUED:
                if dup:
                    self._set_status(node, NodeStatus.REJECTED, f"remeasure exact_dup: {why}", acts, reject_reason="exact_dup")
                continue
            if node.status is NodeStatus.CONFIRMED:
                e, _ = self.world.confirm_evalue(snap, node.id, self.store)
                if dup or e < 1.0 / pr.alpha_confirm:
                    self._set_status(node, NodeStatus.NOT_CONFIRMED, f"remeasure: {why}; confirm E={e:.3g}", acts,
                                     confirm_e=float(e))
                    demoted.append(node.id)
                continue
            p, bar = (post.p_gate, pr.p_promote) if node.status is NodeStatus.PROMOTING else \
                (post.p_beats_incumbent, pr.p_confirm) if node.status is NodeStatus.CONFIRMING and post else (1.0, 0.0)
            if dup or p < bar:
                self._set_status(node, NodeStatus.SCREENED_OUT, f"remeasure: {why}" + ("" if dup else f" gate {p:.3f}"), acts)
                changed.append(node.id)
        for nid in changed + demoted:
            self.book.cancel(nid, "remeasure")
        if demoted:
            a = self._audit_book()
            if (a.get("current") or {}).get("node") in demoted:
                a["current"] = None
            a["queue"] = [n for n in a["queue"] if n not in demoted]
            self.store.set_state(K_AUDIT, a)
            from loop.labless import EXP, K_LABLESS
            lb = self.store.get_state(K_LABLESS, {}) or {}
            self.store.set_state(K_LABLESS, {k: e for k, e in lb.items()
                                             if k.removesuffix(EXP) not in demoted or e.get("submission_id")})
            from loop.select.bandit import K_KIND_PULLS
            self.store.set_state(K_KIND_PULLS, [k for k in self.store.get_state(K_KIND_PULLS, []) or []
                                                if k.get("node_id") not in demoted])
            for nid in demoted:
                self.store.drop_pulls(nid)
            self.store.set_state(K_LAST_CONFIRM, self._evals_at_last_confirm())
        self.store.log("remeasure", demoted=demoted, screened_out=changed,
                       measured={k: v[0] for k, v in snap.measured.items()})
        ML.bump_epoch(self.store, "own increments re-measured against the nearest contained ancestor "
                                  f"(demoted {demoted or 'none'})", now=self.clock())
        self.store.set_state(K_REMEASURE, True)

    def _refreeze(self, acts: list[str]) -> None:
        """Once (the confirm alternative is now data-independent, loop.world.evalues.confirm_delta): every
        CONFIRMING / NOT_CONFIRMED node's frozen delta_1 is re-set to confirm_delta at its frozen sigma bound and
        planned confirm seeds (it never depended on the confirm data, so the e-value stays valid); a NOT_CONFIRMED
        node that closed on its confirm seeds (seeds spent / futile; never a remeasure or eval void), not vetoed,
        whose E now reaches 1/alpha_confirm is CONFIRMED (reason 'refreeze', its bandit credit re-paid) and listed under
        loop.world.K_REFROZEN: it becomes the incumbent, a stack head or a carrier only once it passes the held-out
        audit (loop.world.held_back). CONFIRMED nodes keep theirs."""
        pr, snap, by_node = self.cfg.promotion, self.snapshot, self._runs_by_node()
        book, why = dict(self.store.get_state(CONFIRM_KEY) or {}), {}
        for e in self.store.events("node_transition"):
            why[e.get("node_id")] = str(e.get("reason") or "")
        for node in self.store.nodes(status=[NodeStatus.CONFIRMING, NodeStatus.NOT_CONFIRMED]):
            if node.id not in book:
                continue
            d1 = confirm_delta(book[node.id][1], pr.alpha_confirm, S.planned_confirm(self.cfg, by_node.get(node.id, [])))
            book[node.id] = [d1, book[node.id][1]]
            self.store.update_node(node.id, confirm_alt=d1)
        self.store.set_state(CONFIRM_KEY, book)
        up, led = [], WN.correct(self.ctx, WN.read_ledger(self.ctx, self.store), snap.sigma)
        for node in self.store.nodes(status=[NodeStatus.NOT_CONFIRMED]):
            post = snap.nodes.get(node.id)
            if node.id not in book or post is None or post.vetoed \
                    or not why.get(node.id, "").startswith(("seeds spent", "futile E")):
                continue
            e, n = self.world.confirm_evalue(snap, node.id, self.store, led)
            if n and e >= 1.0 / pr.alpha_confirm:
                new = self._set_status(node, NodeStatus.CONFIRMED, f"refreeze: delta_1={book[node.id][0]:.4f} "
                                                                   f"E={e:.3g} n={n}", acts, confirm_e=float(e))
                self.store.drop_pulls(node.id)
                self.selector.on_confirmed(self.store, snap, new)
                up.append(node.id)
        if up:              # the rule changed after the data: incumbent / head / carrier only after the audit
            self.store.set_state(K_LAST_CONFIRM, self._n_evals())
            self.store.set_state(K_REFROZEN, up)
        self.store.log("refreeze_delta1", confirmed=up, nodes=len(book))
        self.store.set_state(K_REFREEZE, True)

    def _tune_gate(self, acts: list[str]) -> None:
        """Once (loop.tune.gate: own CONFIRM-seed increment >= 1 sigma, one family in tuning-or-confirm): a running
        pass whose base fails it is withdrawn -- no new trial -- and its trials no run of which SLURM has started are
        closed with their PENDING runs cancelled; runs already running finish. Oldest pass first keeps its family."""
        bk = MT.book(self.store)
        for cur in sorted(bk["current"].values(), key=lambda c: c["batch"]):
            base = self.store.get_node(cur["base"])
            others = {k: v for k, v in MT.busy_families(self.store, skip=cur["base"]).items()
                      if v not in {c["base"] for c in MT.book(self.store)["current"].values() if c["batch"] > cur["batch"]}}
            if base is None or not (why := MT.gate(self.ctx, self.store, self.snapshot, base, others)):
                continue
            MT.withdraw(self.store, cur["batch"], why, self.clock())
            closed = []
            for nid in cur["trials"]:
                node = self.store.get_node(nid)
                live = [r for r in self.store.runs(node_id=nid, status=[RunStatus.RUNNING]) if r.id not in self.book.pending]
                if node is None or node.status not in ACTIVE_STATUSES or live:
                    continue
                self._set_status(node, NodeStatus.SCREENED_OUT, f"tuning pass t{cur['batch']} withdrawn: {why}", acts)
                self.book.cancel(nid, "tuning pass withdrawn", only_pending=True)
                closed.append(nid)
            acts.append(f"tune t{cur['batch']} ({cur['base']}) withdrawn: {why}; closed pending trial(s) "
                        f"{closed or 'none'}" + (f"; proposal {cur['pending']} will be rejected" if cur["pending"] else ""))
        self.store.set_state(K_TUNE_GATE, True)

    def _epoch(self, acts: list[str]) -> None:
        """A measurement-epoch bump not yet handled: loop.meta.on_epoch withdraws the playbook and the self-tune
        overrides built on the corrected numbers, and a first-principles review is due at once."""
        e = self.store.get_state(K_EPOCH) or {}
        if self.snapshot is None or int(e.get("epoch", 0)) <= int(e.get("handled", 0)):
            return
        acts += ML.on_epoch(self.ctx, self.store, self.snapshot, now=self.clock())
        self._tune()
        self.store.set_state(K_FP_FORCE, True)
        acts.append(f"measurement epoch {e['epoch']}: reviews before it stop steering; first-principles review due")

    def _evals_at_last_confirm(self) -> int:
        """The closed-node count when the last still-CONFIRMED node was confirmed (from the transition log)."""
        conf, closes, last = {n.id for n in self.store.nodes(status=[NodeStatus.CONFIRMED])}, set(), 0
        for e in self.store.events("node_transition"):
            if e.get("to") in {s.value for s in _EVAL_CLOSES}:
                closes.add(e.get("node_id"))
            if e.get("to") == NodeStatus.CONFIRMED.value and e.get("node_id") in conf:
                last = len(closes)
        return last

    # ------------------------------------------------------------ transitions
    def _regate(self, acts: list[str], key: str = K_REGATE, why: str = "inherited-carrier fix",
                only: Callable[[str], bool] = lambda _nid: True) -> None:
        """Once per ``key`` (the inherited-carrier fix; the robust control offsets, for the nodes whose offset
        estimate the fix raised: ``only``): a PROMOTING node below the screen gate (P(U>0) and its OWN
        P(gain>0)) or a CONFIRMING node below the promote gate (the corrected P(U>U_inc)) passed on a lead
        it does not have; it is screened out as those gates would have (its open runs cancelled)."""
        pr = self.cfg.promotion
        for node in self.store.nodes(status=[NodeStatus.PROMOTING, NodeStatus.CONFIRMING]):
            post = self.snapshot.nodes.get(node.id)
            if post is None or not post.utility.n or not only(node.id):
                continue
            p, bar, what = (post.p_gate, pr.p_promote, "P(U>0)&P(gain>0)") if node.status is NodeStatus.PROMOTING \
                else (post.p_beats_incumbent, pr.p_confirm, "P(U>U_inc)")
            if p < bar:
                self._set_status(node, NodeStatus.SCREENED_OUT, f"regate ({why}): {what}={p:.3f}", acts)
                self.book.cancel(node.id, "regate")
        self.store.set_state(key, True)

    def _transitions(self, acts: list[str], before: WorldSnapshot | None) -> None:
        blocks = S.assign_blocks(self.store.nodes(), self.cfg.seeds, self.store.runs())
        for _ in range(4):                       # a transition can enable the next one
            changed = False
            by_node = self._runs_by_node()
            for node in self.store.nodes(status=ACTIVE_STATUSES):
                d = self._decide(node, by_node, blocks)
                fields = {k: v for k, v in d.fields.items() if getattr(node, k) != v}
                if d.status is None:
                    if fields:
                        self.store.update_node(node.id, **fields)
                    continue
                new = self._set_status(node, d.status, d.reason, acts, **fields)
                changed = True
                if new.status in CLOSED_STATUSES:        # its other runs are evidence of nothing now
                    self.book.cancel(node.id, d.reason)
                if new.status is NodeStatus.CONFIRMED:
                    self.store.set_state(K_LAST_CONFIRM, self._n_evals())
                    self.selector.on_confirmed(self.store, self.snapshot, new)
            if not changed:
                break
            self._refit()
        after, paid = self.snapshot, {p.node_id for p in self.store.pulls()}
        for node in self.store.nodes(status=CLOSED_STATUSES) if after else ():
            if node.id not in paid and node.reject_reason != SMOKE_UNAVAILABLE_REASON:   # no verdict, no credit
                pull = self.selector.reward(self.store, before or after, after, node)
                self.store.record_pull(pull)
                acts.append(f"reward {pull.arm.value} {pull.reward:+.4g} for {node.id}")

    def _mech_pool(self, acts: list[str]) -> None:
        """Mechanism-level pooled confirmation: CONFIRMING / NOT_CONFIRMED siblings sharing a normalized mechanism tag
        (loop.novelty.norm_tag) multiply their confirm-seed e-values (independent fresh seeds, already control-
        corrected) against the mechanism null; at 1/alpha_confirm the member with the best confirm-only estimate
        (> 0) is CONFIRMED ('mechanism-pooled'). Each seed counts once: a pooled graduation spends every member
        (K_MECH), and a node already CONFIRMED on its own never pools. Never vetoed or design points."""
        snap, pr = self.snapshot, self.cfg.promotion
        if snap is None:
            return
        spent = dict(self.store.get_state(K_MECH) or {})
        used, groups, led = {n for v in spent.values() for n in v}, {}, None
        for n in self.store.nodes(status=[NodeStatus.CONFIRMING, NodeStatus.NOT_CONFIRMED]):
            p, tag = snap.nodes.get(n.id), NV.norm_tag(n.mechanism_tag)
            if not (n.id in used or p is None or p.vetoed or design_point(n) or not tag):
                groups.setdefault(tag, []).append(n)
        for tag, nodes in sorted(groups.items()):
            if len(nodes) < 2:
                continue
            led = led or WN.correct(self.ctx, WN.read_ledger(self.ctx, self.store), snap.sigma)
            mem = [(n, e, (snap.nodes[n.id].confirm_estimate or snap.nodes[n.id].utility).mean)
                   for n in nodes for e, k in [self.world.confirm_evalue(snap, n.id, self.store, led)] if k]
            prod = math.prod(e for _, e, _ in mem)
            best = max(mem, key=lambda m: (m[2], m[0].id))
            if len(mem) < 2 or prod < 1.0 / pr.alpha_confirm or best[2] <= 0.0:
                continue
            ids = [m[0].id for m in mem]
            new = self._set_status(best[0], NodeStatus.CONFIRMED, f"mechanism-pooled: E={prod:.3g} over "
                                                                  f"{','.join(ids)} ({tag})", acts, confirm_e=float(prod))
            self.book.cancel(new.id, "mechanism-pooled")
            self.store.drop_pulls(new.id)              # its credit is paid (again) at the new status
            self.store.set_state(K_LAST_CONFIRM, self._n_evals())
            self.selector.on_confirmed(self.store, snap, new)
            spent[tag] = ids
            self.store.set_state(K_MECH, spent)
            self.store.log("mechanism_pooled", node_id=new.id, tag=tag, members=ids, e_value=prod)

    # --------------------------------------------------------------- triggers
    def _triggers(self, acts: list[str]) -> None:
        """Reviews only: a closed node's explanation is computed on demand (loop.explain), never an LLM call."""
        if self._llm_paused() or self._llm_probing():
            return
        if self._review is None and (trigger := FP_TRIGGER if self._fp_due() else self._review_due()):
            self.start_review(trigger, acts)

    def _big_reviews(self) -> int:
        return sum(r.trigger != FP_TRIGGER for r in self.store.reviews())

    def _fp_due(self) -> bool:
        """A first-principles review every ``fp_every_reviews`` big reviews, or after
        ``fp_stall_closes`` closes without a confirmation; never within ``fp_min_hours`` of the last."""
        rc, last = self.cfg.review, self.store.get_state(K_FP, {})
        if self.store.get_state(K_FP_FORCE):                  # a measurement-epoch bump: no waiting
            return True
        if self.clock() - float(last.get("ts", -1e18)) < 3600 * rc.fp_min_hours:
            return False
        return (0 < rc.fp_every_reviews <= self._big_reviews() - int(last.get("big", 0))
                or 0 < rc.fp_stall_closes <= self._n_evals() - int(self.store.get_state(K_LAST_CONFIRM, 0)))

    def _review_due(self) -> str:
        """A STALL review only (cadence reviews are gone): ``stall_evals`` closes since both the last review and the
        last confirmation, at most once per ``stall_min_hours``."""
        rc, last = self.cfg.review, self.store.get_state(K_REVIEW, {})
        evals = self._n_evals()
        since_eval = evals - int(last.get("evals", 0))
        since_conf = evals - int(self.store.get_state(K_LAST_CONFIRM, 0))
        if min(since_conf, since_eval) < rc.stall_evals:
            return ""
        ts = last.get("ts") or max((r.created_ts or 0.0 for r in self.store.reviews()), default=0.0)
        return "stall" if self.clock() - float(ts) >= 3600 * rc.stall_min_hours else ""

    def start_review(self, trigger: str, acts: list[str] | None = None) -> None:
        snap = self.snapshot or self._refit()
        marker = {"evals": self._n_evals(), "trigger": trigger, "ts": self.clock(),
                  "confirmations": len(self.store.nodes(status=[NodeStatus.CONFIRMED])),
                  "evidence": RD.evidence_key(self.store, snap)}
        if trigger == FP_TRIGGER:                    # the causal component audit runs with each FP review
            self.store.set_state(K_FP_FORCE, False)
            try:
                done = CG.audit(self.ctx, self.store, snap, self.clock())
            except Exception as exc:  # noqa: BLE001 - analytics never blocks the review
                done = [f"component audit failed: {exc!r}"[:200]]
                self.store.log("component_audit_error", error=repr(exc)[:500])
            if acts is not None:
                acts += done
        if trigger in ("stall", FP_TRIGGER) and RD.same_input(self.store, "review", marker["evidence"]):
            self.store.set_state(K_REVIEW, {**marker, "skipped": True})       # loop.redundancy: nothing new to
            if trigger == FP_TRIGGER:                                         # review; the window counts as done
                self.store.set_state(K_FP, {"ts": self.clock(), "big": self._big_reviews()})
            line = RD.skipped(self.store, f"review:{trigger}", marker["evidence"])
            if acts is not None:
                acts.append(line)
            return
        self._review = (self.pool.submit(self.analyst.big_review, self.store, snap, trigger=trigger),
                        trigger, marker)
        if acts is not None:
            acts.append(f"big review started ({trigger})")

    def _llm_paused(self) -> bool:
        return getattr(self, "_draining", False) or R.llm_paused(self.store)

    def _llm_probing(self) -> bool:
        return R.llm_probing(self.store)

    def _probe_ok(self) -> None:
        R.llm_probe_ok(self.store)

    def _llm_gate(self, acts: list[str]) -> None:
        R.llm_gate(self.store, self.clock(), acts)

    def _llm_failed(self, key: str, exc: BaseException) -> bool:
        """Count a failed LLM job (an outage is not counted: it waits); True once spent."""
        if not isinstance(exc, BackendUnavailable):
            self._fails[key] = self._fails.get(key, 0) + 1
        return self._fails.get(key, 0) >= LLM_TRIES

    # ------------------------------------------------------------ held-out audit
    def _audit_book(self) -> dict[str, Any]:
        return {"passed": [], "failed": [], "queue": [], "current": None, "gate": {},
                **self.store.get_state(K_AUDIT, {})}

    def _audit(self, acts: list[str]) -> list[S.RunNeed]:
        """The queued nodes (the incumbent whenever it is an unaudited confirmed node) vs root on
        the audit seeds; a failed audit demotes the chain to the last audited ancestor. With ``seeds.audit_rotate``
        each audit draws its own never-used seeds (loop.remint) and root AND the audited incumbent (``vs``, stage
        remint) run on them too, so the incumbent comparison stays paired on seeds neither node was selected on."""
        snap = self.snapshot
        if not self._gate_backfilled:                # verdicts from before the gate numbers were stored
            self._gate_backfilled = True
            try:
                acts += backfill_audit_gate(self.ctx, self.store, snap)
            except Exception as exc:  # noqa: BLE001 - a note detail never stops the audit
                self.store.log("audit_gate_backfill_failed", error=repr(exc)[:300])
        for nid in (snap.incumbent_id, *sorted(n.id for n in self.store.nodes(status=[NodeStatus.CONFIRMED]))):
            if nid != ROOT_NODE_ID:                  # every confirmed node is audited, the incumbent first
                self.labless_ready(nid)
        a = self._audit_book()
        if a["current"] is None:
            done = a["passed"] + a["failed"]
            todo = [n for n in a["queue"] if n not in done and n in snap.nodes]   # the incumbent jumps the queue
            pick = snap.incumbent_id if snap.incumbent_id in todo else next(iter(todo), None)
            post = snap.nodes.get(pick) if pick else None
            if post is None:
                return []
            claim = (post.confirm_estimate or post.utility).mean
            row = self.store.get_state(NOISE_KEY)[PANEL]      # the frozen sigma bound, as confirm
            inc = snap.incumbent_id
            vs = inc if inc not in (ROOT_NODE_ID, pick) and inc in a["passed"] else None
            seeds = (RM.draw(self.ctx, self.store, len(self.cfg.seeds.audit), f"audit {pick}", self.clock())
                     if self.cfg.seeds.audit_rotate else list(map(int, self.cfg.seeds.audit)))
            a["current"] = {"node": pick, "sigma": sigma_ucb(row["sigma"], row["df_eff"]),
                            "delta": max(claim, self.cfg.promotion.delta_min_sigmas * self._sigma_p()),
                            "seeds": seeds, "vs": vs}
            self.store.set_state(K_AUDIT, a)
            acts.append(f"audit started for {pick} (claimed gain {claim:+.5f}) on seeds {seeds}"
                        + (f", incumbent {vs} paired on them" if vs else ""))
        cur, by_node = a["current"], self._runs_by_node()
        seeds, vs = [int(s) for s in cur.get("seeds") or self.cfg.seeds.audit], cur.get("vs")
        est = {k: ((q := p.confirm_estimate or p.utility).mean, q.sd ** 2) for k, p in snap.nodes.items()}
        offsets = S.audit_offsets(cur["node"], by_node, self.cfg, est, self._sigma_p(), seeds=seeds)
        verdict, e_value, needs = S.audit_step(
            cur["node"], by_node.get(cur["node"], []), by_node.get(ROOT_NODE_ID, []), cfg=self.cfg,
            delta=cur["delta"], sigma=cur["sigma"], infra_retry=self.book.infra_retry, offsets=offsets,
            resume_v=WN.resume_var(self.ctx, snap.sigma, PANEL), seeds=seeds)
        vneeds, vopen = S.stage_needs(vs, Stage.REMINT, seeds, by_node.get(vs, []), self.book.infra_retry,
                                      cur["node"]) if vs else ([], False)
        if not verdict or vopen:
            return [*needs, *vneeds]
        cs = S.audit_deltas(by_node.get(cur["node"], []), by_node.get(ROOT_NODE_ID, []), cfg=self.cfg,
                            offsets=offsets, seeds=seeds)
        a["current"], a["queue"] = None, [n for n in a["queue"] if n != cur["node"]]
        a[verdict].append(cur["node"])
        a.setdefault("gate", {})[cur["node"]] = {**S.audit_gate(          # frozen: the note prints these
            list(cs.values()), seeds=list(cs), verdict=verdict, e_value=e_value, delta=cur["delta"],
            sigma=cur["sigma"], alpha=self.cfg.promotion.alpha_audit, offsets=offsets, ts=self.clock()),
            "audit_seeds": seeds, "vs": vs}
        self.store.set_state(K_AUDIT, a)
        self.store.log("audit", node_id=cur["node"], verdict=verdict, e_value=e_value)
        acts.append(f"audit {cur['node']}: {verdict} (E={e_value:.3g})")
        nid = cur["node"] if verdict == "failed" else None
        while nid and nid != ROOT_NODE_ID and nid not in a["passed"]:
            node = self.store.get_node(nid)
            if node is not None and node.status is NodeStatus.CONFIRMED:
                self._set_status(node, NodeStatus.AUDIT_FAILED, f"audit_failed {cur['node']}", acts)
            nid = node.parent_id if node is not None else None
        return []

    # ---------------------------------------------------------------- harvest
    def _meta_jobs(self, acts: list[str]) -> None:
        """Start the RESEARCH SCOUT when due (loop.scout) and the meta-learning CONSOLIDATION after the reviews
        that call for it (loop.meta.consolidation_due); both run in the pool, the harvest writes."""
        now, snap = self.clock(), self.snapshot
        if self._llm_paused() or self._llm_probing():
            return
        if self.scout is not None and self._scout is None:
            marker = int(self.store.get_state(K_LAST_CONFIRM, 0))
            if (trigger := SC.due(self.ctx, self.store, now, evals_since_confirm=self._n_evals() - marker,
                                  confirm_marker=marker)) \
                    and not CG.use(self.ctx, self.store, "scout", "run", f"run@{now:.0f}"):
                SC.write(self.ctx, self.store, {"trigger": trigger, "ts": now, "ok": True, "ideas": [],
                                                "error": "ablated (component audit)"}, confirm_marker=marker)
            elif trigger and trigger != "first" and RD.same_input(self.store, "scout", key := RD.evidence_key(
                    self.store, snap)):
                SC.write(self.ctx, self.store, {"trigger": trigger, "ts": now, "ok": True, "ideas": [],
                                                "error": "skipped: no new evidence (loop.redundancy)"}, confirm_marker=marker)
                acts.append(RD.skipped(self.store, "scout", key))
            elif trigger:
                self._scout = (self.pool.submit(self.scout.run, self.store, snap, trigger=trigger, now=now),
                               trigger, marker)
                acts.append(f"research scout started ({trigger})")
        if self._consolidation is None and hasattr(self.analyst, "consolidate") and snap is not None \
                and ML.consolidation_due(self.store, self.cfg, FP_TRIGGER):
            if RD.same_input(self.store, "consolidation", key := RD.evidence_key(self.store, snap)):
                old = ML.playbook(self.store)                 # nothing new to condense: the window counts as done
                self.store.set_state(ML.K_PLAYBOOK, {**old, "ts": now, "reviews_at": len(self.store.reviews())})
                acts.append(RD.skipped(self.store, "consolidation", key))
            else:
                self._consolidation = (self.pool.submit(self.analyst.consolidate, self.store, snap), key)
                acts.append("meta-learning consolidation started")

    def _harvest_meta(self, acts: list[str]) -> None:
        if self._scout is not None and self._scout[0].done():
            fut, trigger, marker = self._scout
            self._scout = None
            try:
                res = fut.result()
            except Exception as exc:  # noqa: BLE001 - recorded as a failed run: retried after RETRY_FAILED_S
                res = {"trigger": trigger, "ts": self.clock(), "ok": False, "error": repr(exc)[:300], "ideas": []}
            run = SC.write(self.ctx, self.store, res, confirm_marker=marker)
            if run.get("ok"):
                RD.remember(self.store, "scout", RD.evidence_key(self.store, self.snapshot), self.clock())
            acts.append(f"research scout ({trigger}): {run['written']} idea(s) written, {run['dup']} dup, "
                        f"{run['refused']} refused, {run.get('searches') or 0} searches"
                        + ("" if run.get("ok") else f" FAILED: {str(run.get('error'))[:120]}"))
        if self._consolidation is not None and self._consolidation[0].done():
            (fut, key), self._consolidation = self._consolidation, None
            try:
                pb = fut.result()
            except Exception as exc:  # noqa: BLE001
                self.store.log("consolidation_error", error=repr(exc)[:500])
                if self._llm_failed("consolidate", exc):     # give up on this window
                    self._fails.pop("consolidate")
                    old = ML.playbook(self.store)
                    self.store.set_state(ML.K_PLAYBOOK, {**old, "ts": self.clock(),
                                                         "reviews_at": len(self.store.reviews())})
                return
            RD.remember(self.store, "consolidation", key, self.clock())
            old = ML.playbook(self.store)
            if RD.same_rules(old.get("rules") or [], pb.get("rules") or []):     # a near-copy: no new version
                self.store.set_state(ML.K_PLAYBOOK, {**old, "ts": self.clock(), "reviews_at": len(self.store.reviews())})
                self.store.log("playbook_unchanged", version=old.get("version"))
                acts.append(f"playbook v{old.get('version')} kept: the consolidation restated its rules")
                return
            ML.save_playbook(self.store, pb)
            acts.append(f"playbook v{pb['version']}: {len(pb['rules'])} rule(s), {len(pb.get('dropped', []))} "
                        f"dropped (invented numbers), {len(pb.get('retired', []))} retired")

    def _harvest(self, acts: list[str]) -> None:
        self._harvest_meta(acts)
        for fut in [f for f in self._research if f.done()]:
            req = self._research.pop(fut)
            if hasattr(self.adapter, "cancel_smokes"):     # its researcher ended: a smoke job left queued would
                try:                                       # start later on a reaped worktree (no .venv) and
                    if ids := self.adapter.cancel_smokes(f"loop-{req.id}"):     # waste a slot
                        acts.append(f"scancelled smoke job(s) {','.join(ids)} left by {req.id}")
                except Exception as exc:  # noqa: BLE001
                    self.store.log("cancel_smokes_failed", request_id=req.id, error=repr(exc)[:300])
            try:
                res = fut.result()
                if res.status != "backend_unavailable":
                    self._probe_ok()
                acts.append(self._intake(req, res))
                acts += [x for x in self._db_lines if x]
                self._db_lines = []
            except BackendUnavailable:                 # the backend is down: keep the work, retry next tick
                self._research[fut] = req
                acts.append(f"research {req.id} finished; intake waits for the LLM backend")
            except Exception as exc:  # noqa: BLE001
                self.store.log("research_error", request_id=req.id, error=repr(exc)[:500])
                acts.append(f"research {req.id} raised {exc!r}"[:200])
        if self._review is not None and self._review[0].done():
            fut, trigger, marker = self._review
            self._review = None
            try:
                report, metas = fut.result()        # "evals" = the closes the triggers count (n_scored is not)
                acts += self.persist_review(dataclasses.replace(report, n_evals_at=int(marker["evals"])), metas)
            except Exception as exc:  # noqa: BLE001 - not done: re-triggered next tick
                self.store.log("review_error", trigger=trigger, error=repr(exc)[:500])
                if self._llm_failed("review", exc):          # give up on this window
                    self._fails.pop("review")
                    self.store.set_state(K_REVIEW, {**marker, "failed": True})
                    if trigger == FP_TRIGGER:
                        self.store.set_state(K_FP, {"ts": self.clock(), "big": self._big_reviews()})
                return
            self.store.set_state(K_REVIEW, marker)
            RD.remember(self.store, "review", marker.get("evidence", ""), self.clock())
            if trigger == FP_TRIGGER:
                self.store.set_state(K_FP, {"ts": self.clock(), "big": self._big_reviews()})
            acts.append(f"review {report.id} ({trigger}): {len(metas)} meta-hypothesis(es)")

    def persist_review(self, report: Any, metas: Sequence[Any]) -> list[str]:
        """Store a review; its self-tune asks are applied (whitelisted) or become operator proposals. A proposal
        restating an earlier one is dropped and a meta-hypothesis restating a live one merged into it
        (loop.redundancy)."""
        now, tuned = self.clock(), []
        if report.selftune:
            tuned, props = T.apply(self.store, self._spec_cfg, json.loads(report.selftune), source=report.id, now=now)
            report = dataclasses.replace(report, proposals=(*report.proposals, *props))
            self._tune()
        kept, dropped = RD.dedupe_proposals(self.store, report.proposals)
        for text, oid, c in dropped:
            self.store.log("proposal_deduped", review_id=report.id, same_as=oid, cos=c, text=text[:300])
        stale = self.store.get_state(K_STALE_META) or {}
        live = [h for h in self.store.hypotheses(kind=HypothesisKind.META) if h.id not in stale
                and (h.review_id or "") in {r.id for r in current_reviews(self.store)}]
        metas, into = RD.merge_metas(self.store, metas, live)
        for mid, oid in into.items():
            self.store.log("meta_merged", review_id=report.id, meta_id=mid, into=oid)
        if dropped or into:
            report = dataclasses.replace(
                report, proposals=tuple(kept),
                meta_hypothesis_ids=tuple(dict.fromkeys(into.get(m, m) for m in report.meta_hypothesis_ids)),
                probe_briefs=tuple((into.get(m, m) if m else m, b) for m, b in report.probe_briefs))
        for h in metas:
            if self.store.get_hypothesis(h.id) is None:
                self.store.put_hypothesis(dataclasses.replace(
                    h, kind=HypothesisKind.META, review_id=report.id, created_ts=h.created_ts or now))
        self.store.put_review(dataclasses.replace(report, created_ts=report.created_ts or now))
        ML.record(self.store, "first_principles" if report.trigger == FP_TRIGGER else "review", report.id,
                    review=report.id, metas=[h.id for h in metas], probes=len(report.probe_briefs),
                    selftune=tuned, design=bool(report.design))
        if report.design:
            D.request(self.store, json.loads(report.design))
        self._dirty = True
        return tuned

    def _new_id(self) -> str:
        seq = len(self.store.nodes())                   # nodes are never deleted
        while self.store.get_node(f"n{seq:04d}") is not None:
            seq += 1
        return f"n{seq:04d}"

    def _trees(self, *commits: str) -> None:
        """K_TREES: the git tree of every node commit and ``commits`` (intake's exact_dup on code + config)."""
        fn, have = getattr(self.adapter, "trees", None), self.store.get_state(IN.K_TREES) or {}
        todo = [c for c in (*(n.commit for n in self.store.nodes()), *commits) if c and c not in have]
        if fn is None or not todo:
            return
        try:
            self.store.set_state(IN.K_TREES, {**have, **fn(todo)})
        except Exception as exc:  # noqa: BLE001 - commit equality still catches a re-rooted copy
            self.store.log("trees_failed", error=repr(exc)[:300])

    def _measure(self) -> None:
        """MEASURE_KEY: for each new node, the nodes its commit contains above its declared parent's (git
        ancestry: a crossover / copy re-rooted on root that carries the incumbent's whole stack contains it),
        nearest effective config first (fewest differing keys, then the latest): the world measures its OWN
        increment against the first ESTABLISHED one (confirmed or audit-passed: its gain already credited), so no
        node is credited with a carrier it merely copied, while a composition of unconfirmed legs keeps its parent."""
        fn = getattr(self.adapter, "ancestry", None)
        book = dict(self.store.get_state(MEASURE_KEY) or {})
        nodes = self.store.nodes()
        todo = [n for n in nodes if n.id not in book and n.id != ROOT_NODE_ID and not design_point(n)
                and n.status is not NodeStatus.REJECTED]
        if fn is None or not todo:
            return
        base, cfgs = self.store.get_state(IN.K_BASE) or {}, {}
        eff = lambda nid: cfgs.setdefault(nid, {**base, **IN.parent_config(self.store, nid)})   # noqa: E731
        by_commit: dict[str, list[Node]] = {}
        for m in nodes:
            if m.id != ROOT_NODE_ID and not design_point(m) and m.status is not NodeStatus.REJECTED:
                by_commit.setdefault(m.commit, []).append(m)
        for n in todo:
            parent = self.store.get_node(n.parent_id or ROOT_NODE_ID)
            try:
                commits = fn(n.commit, parent.commit) if parent is not None else []
            except Exception as exc:  # noqa: BLE001 - measured against its declared parent meanwhile
                self.store.log("measure_failed", node_id=n.id, error=repr(exc)[:300])
                continue
            cands = [m for c in commits for m in by_commit.get(c, ()) if (m.created_ts, m.id) < (n.created_ts, n.id)]
            book[n.id] = [m.id for m in sorted(cands, key=lambda m: (len(IN.config_gaps(eff(m.id), eff(n.id))),
                                                                     -m.created_ts, m.id))]
        if book != (self.store.get_state(MEASURE_KEY) or {}):
            self.store.set_state(MEASURE_KEY, book)
            self._dirty = True

    def _reissue(self, req: CandidateRequest, cand: Any, why: str) -> CandidateRequest:
        """Re-submit ``req`` (same id, so a tuning pass / bank booking still matches) with the previous
        attempt's diff and hypothesis appended to its brief: a smoke that never got a slot is no verdict."""
        brief = (req.brief or "").split(REISSUE_MARK)[0] + REISSUE_MARK + f"\n({why[:200]}) It says NOTHING about " \
            "the idea. Re-apply the previous attempt's change below (fix only what your own smoke shows broken), " \
            "smoke it, and finish with the same hypothesis unless the smoke changes your mind."
        if cand is not None:
            h = cand.hypothesis
            brief += (f"\nPrevious hypothesis ({h.family}/{h.mechanism_tag}): {h.claim[:1500]}"
                      f"\nPrevious diff:\n{(cand.diff_text or '')[:20000]}")
        req = dataclasses.replace(req, brief=brief)
        self._smoke_reissues[req.id] = self._smoke_reissues.get(req.id, 0) + 1
        self._research[self.pool.submit(self.researcher.research, req, self.store,
                                        self.snapshot or self._refit())] = req
        self._sync_inflight()
        self.store.log("research_reissued", request_id=req.id, reason=why[:300],
                       reissues=self._smoke_reissues[req.id])
        self.store.log("research_started", request_id=req.id, strategy=req.strategy.value,
                       operator=req.operator.value, parents=list(req.parent_ids), brief_kind=req.brief_kind,
                       backend=self._backend(), meta_id=req.meta_hypothesis_id, reissued=True)
        return req

    def _intake(self, req: CandidateRequest, result: ResearchResult) -> str:
        cand = result.candidate
        if result.status == "backend_unavailable" and result.error.startswith(SMOKE_UNAVAILABLE_REASON) \
                and self._smoke_reissues.get(req.id, 0) < MAX_SMOKE_REISSUES:
            self._reissue(req, cand, result.error)    # no node, no reward: the cluster had no slot
            return f"research {req.id}: smoke got no slot in time; re-issued ({self._smoke_reissues[req.id]}/" \
                   f"{MAX_SMOKE_REISSUES})"
        if cand is not None:
            self._trees(cand.commit)
        ds = (result.plan or {}).get("dataset_spec") if result.status != "backend_unavailable" else None
        if ds is not None and cand is None:              # a dataset request without a node: the request's own
            self._db_lines.append(DB.request(self.ctx, self.store, req.id, ds))
        if cand is None or result.status == "backend_unavailable":   # an outage is nobody's node
            book = self.store.get_state(K_DEBUG) or {}
            if result.status == "backend_unavailable" and req.parent_ids and book.get(req.parent_ids[0]) == req.id:
                book.pop(req.parent_ids[0])                        # a deterministic debug is re-asked
                self.store.set_state(K_DEBUG, book)
            wx, key = self.store.get_state(K_WINX) or {}, "+".join(req.parent_ids[:2])
            if wx.get(key) == req.id and (result.status == "backend_unavailable" or self._winx_retry(key)):
                wx.pop(key)                                        # so is a winner crossover (and one that
                self.store.set_state(K_WINX, wx)                   # failed for another reason, once)
            st, key = self.store.get_state(K_STACK) or {}, "+".join(sorted(req.parent_ids))
            if st.get(key) == req.id and (result.status == "backend_unavailable" or self._winx_retry(key)):
                st.pop(key)                                        # and a stack
                self.store.set_state(K_STACK, st)
            self.store.log("research_failed", request_id=req.id, status=result.status,
                           error=result.error[:500])
            if result.status == "backend_unavailable":       # no attempt: nothing is spent or counted
                back = [MT.outage(self.store, req.id), LG.outage(self.store, req.id),
                        *getattr(self.selector, "unserve_probe", lambda *_: [])(self.store, req.id)]
                if back := [x for x in back if x]:
                    return f"research {req.id} produced nothing (backend_unavailable): {', '.join(back)} asked again"
            return f"research {req.id} produced nothing ({result.status})"
        snap = self.snapshot or self._refit()
        reason = self.selector.admit(self.store, snap, cand, req.operator, None, req.strategy) if result.status == "ok" \
            else result.status
        admitted = not reason
        node_id, now, hyp = self._new_id(), self.clock(), cand.hypothesis
        parent = self.store.get_node(req.parent_ids[0] if req.parent_ids else ROOT_NODE_ID)
        crashed = parent if req.operator is Operator.DEBUG and parent is not None else None
        if crashed is not None:           # a fix is compared with the crashed node's own control
            parent = self.store.get_node(crashed.parent_id or ROOT_NODE_ID)
        xover = req.operator is Operator.CROSSOVER and len(req.parent_ids) > 1
        if xover:                         # a crossover is paired against its parents' common ancestor
            parent = self.store.get_node(common_ancestor(self.store, *req.parent_ids[:2]))
            if not hyp.levers.directions:
                hyp = dataclasses.replace(hyp, levers=crossover_levers(self.store, req.parent_ids[:2]))
        hyp_id = hyp.id or f"h-{node_id}"
        if self.store.get_hypothesis(hyp_id) is None:
            self.store.put_hypothesis(dataclasses.replace(hyp, id=hyp_id, created_ts=hyp.created_ts or now))
        diff_path = ""
        if cand.diff_text:
            p = Path(self.ctx.state_dir) / "diffs" / f"{node_id}.diff"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(cand.diff_text)
            diff_path = str(p)
        self.store.put_node(LG.adopt(self.store, req, MT.adopt(self.store, req, Node(
            # a tuning / exposure / leg-branch proposal: its pass / ladder / leg factorial
            id=node_id, task_id=self.ctx.task_id, commit=cand.commit,
            parent_id=parent.id if parent else ROOT_NODE_ID, strategy=req.strategy,
            operator=req.operator, family=hyp.family, mechanism_tag=hyp.mechanism_tag,
            diff_hash=cand.diff_hash, status=NodeStatus.QUEUED if admitted else NodeStatus.REJECTED,
            island=req.island, second_parent_id=crashed.id if crashed else
            req.parent_ids[1] if len(req.parent_ids) > 1 else None,
            diff_path=diff_path, hypothesis_id=hyp_id, levers=hyp.levers,
            debug_depth=crashed.debug_depth + 1 if crashed else 0,
            embedding=self.selector.embedding_of(cand) if admitted else None,
            reject_reason=reason, created_ts=now, updated_ts=now,
            notes=f"request={req.id} brief={req.brief_kind} backend={self._backend()}"
                  + (f" web=1 searches={result.web_searches} fetches={result.web_fetches}" if result.web else " web=0")
                  + (f" bank={req.bank_id}" if req.bank_id else "")
                  + (f" idea={result.plan['idea_id']}" if result.plan.get("idea_id") else "")
                  + (" fp=1" if result.plan.get("first_principles") else "")
                  + f" mtype={NV.mechanism_type(result.mechanism_type, cand.changed_paths, cand.diff_text, hyp.mechanism_tag, hyp.family)}"
                  + (f" dose={result.dose_knob}" if result.dose_knob else "")
                  + (f" init={''.join(str(to.get('source') or to.get('type') or '?').split())}"   # an init swap
                     if (to := (getattr(cand.init_compat, "checks", None) or {}).get("init_swap", {}).get("to")) else "")
                  + (" identical=1" if result.identical else "")
                  + (f" stacks_with={','.join(result.stacks_with)}" if result.stacks_with else "")
                  + (f" conflicts_with={','.join(result.conflicts_with)}" if result.conflicts_with else "")
                  + (f" persona={req.persona}" if req.persona else "")
                  + (f" xover={req.parent_ids[0]},{req.parent_ids[1]}" if xover else "")
                  + (f" stack_legs={','.join(req.parent_ids)}"
                     if req.id in (self.store.get_state(K_STACK) or {}).values() else "")
                  + f" fams={'+'.join(derive_families(self.ctx, cand)[0]) or '-'}"
                  + (f" compound={split[0]}+{split[1]}"
                     + (f" stack_from={','.join(src)}"
                        if (src := compound_sources(self.ctx, self.store, snap, cand, split[0])) else "")
                     if admitted and not xover and req.strategy is not Strategy.MOONSHOT
                     and (split := compound_split(self.ctx, self.store, snap, cand)) else "")))))
        if admitted and cand.config_diff:
            self.store.set_state(f"config_diff:{node_id}", dict(cand.config_diff))
        if ds is not None and admitted:     # its dataset_spec: built for it (it waits while it is built)
            self._db_lines.append(DB.request(self.ctx, self.store, node_id, ds))
        if result.plan:                     # first-principles record, idea id, variants + branched plan
            self.store.set_state(f"ideation:{node_id}", dict(result.plan))
            if admitted and (got := VR.register(self.ctx, self.store, node_id, result.plan, snap)):
                self.store.log("variants_planned", node_id=node_id, detail=got[:300])
        self.store.link(hyp_id, node_id, sign=1, auto=False)
        meta = self.store.get_hypothesis(req.meta_hypothesis_id or "")
        if meta is not None and admitted:
            if req.brief_kind in ("probe", "moonshot") and req.operator is not Operator.CROSSOVER:
                # made from the meta's own review probe brief: an EXPLICIT link (D3.9)
                dot = sum(d * meta.levers.directions.get(l, 0) for l, d in hyp.levers.directions.items())
                self.store.link(meta.id, node_id, sign=-1 if dot < 0 else 1, auto=False)
            link_meta(self.store, meta)
            if node_id not in {n for n, _ in self.store.links(meta.id)}:
                self.store.log("probe_unlinked", node_id=node_id, meta_id=req.meta_hypothesis_id)
        self.store.put_handoff(dataclasses.replace(cand.handoff, id=f"ho-{node_id}", node_id=node_id,
                                                   created_ts=cand.handoff.created_ts or now))
        self.store.log("intake", node_id=node_id, request_id=req.id, admitted=admitted, reason=reason,
                       **({"init_compat": str(cand.init_compat.reason)[:600]}
                          if reason in ("init_incompatible", "init_illegal", "over_flop_cap")
                          and getattr(cand, "init_compat", None) is not None else {}))
        self._dirty = True
        if reason in ("sweep_only", "below_noise"):    # operator separation: the knobs go to a design batch
            keys = IN.redirect(self.ctx, self.store, cand, node_id, {node_id}, now, request_id=req.id)
            return f"redirected {node_id} from {req.id}: numeric-only {keys} -> design knob pool"
        if not admitted:
            return f"rejected {node_id} from {req.id}: {reason}"
        return f"queued {node_id} ({hyp.family}/{hyp.mechanism_tag}) from {req.id}"

    # -------------------------------------------------------- cluster courtesy
    def _courtesy(self, acts: list[str]) -> None:
        self.courtesy.step(acts)

    # ------------------------------------------------------------ OOM / crashes
    def _oom_reopen(self, acts: list[str]) -> None:
        """A node that closed CRASHED on a GPU OOM with its adapter's ladder not spent (it crashed
        before the ladder existed) is reopened at the stage it crashed in: the crash run becomes an owed
        infra retry at the next step (idea nodes, and the running designed batch's points)."""
        live = D.current_points(self.store)
        for node in self.store.nodes(status=[NodeStatus.CRASHED]):
            if node.id == ROOT_NODE_ID or (design_point(node) and node.id not in live):
                continue
            crashes = [r for r in self.store.runs(node_id=node.id) if r.status is RunStatus.CRASH]
            crash = crashes[0] if crashes else None
            if crash is None or crash.stage not in _REOPEN or not all(is_oom(r.log_tail) for r in crashes) \
                    or not (step := self.book.next_mitigation(node, crash)):
                continue
            for r in crashes:
                self.store.update_run(r.id, status=RunStatus.VOID, void_type=VoidType.INFRA, attempt=0, mitigation=step)
            node = self.store.update_node(node.id, notes=with_oom_note(node.notes, step))
            self._set_status(node, _REOPEN[crash.stage], f"oom retry at {step} ({crash.id})", acts)
            self.store.log("run_oom_retry", run_id=crash.id, node_id=node.id, mitigation=step, reopened=True)

    def _debug_requests(self) -> list[CandidateRequest]:
        """Fix-and-continue: every non-design node closed CRASHED (its own run crashed, after the OOM
        ladder) gets ONE DEBUG request at once, outside the bandit, while its lineage is below
        ``researcher.max_debug``; the researcher sees the crash log tail and the diff. An outage frees
        the slot for a retry."""
        book, out = self.store.get_state(K_DEBUG) or {}, []
        room = self._side_room()
        for node in self.store.nodes(status=[NodeStatus.CRASHED]):
            if len(out) >= room:
                break
            if node.id in book or node.id == ROOT_NODE_ID or node.strategy is Strategy.DESIGN \
                    or node.debug_depth >= self.cfg.researcher.max_debug:
                continue
            req = self.selector.debug_request(self.store, self.snapshot, node)
            book[node.id] = req.id
            self.store.set_state(K_DEBUG, book)
            out.append(req)
        return out

    def _queue_marks(self) -> tuple[int, int]:
        """(target, max) for the idea queue, in NODES: target = max(queue_low_water, free_slots // 2) -- a queued
        node needs ~2 screen runs (+ maybe controls), so free_slots // 2 nodes fill the free slots (RunBook.headroom,
        not the pending-gated free()); the max leaves room for the researchers in flight. See
        :func:`research_starts` for how it is used.

        The headroom is the MAX seen over the last research lead time (``researcher.moonshot_timeout_min``), not the
        instant one: a researcher started now delivers its node ~35 min later, while a pool's PENDING spell (the
        ``max_pending_per_pool`` gate zeroes its headroom) usually clears in minutes. Sized on the instant value the
        queue target fell to ``queue_low_water`` whenever 2 of our runs pended on idle; live 2026-10-05 15:23-16:13:
        researchers 12 -> 0, then SLURM granted the pending runs, the queue emptied and 40 slots waited ~45 min on
        a research pipeline restarted from zero (16:30: queued=0, 16 running). A pool pending past the window
        still shrinks the target."""
        i, now, h = self.cfg.intake, self.clock(), self.book.headroom()
        win = 60.0 * self.cfg.researcher.moonshot_timeout_min
        self._headroom_seen = [(t, x) for t, x in self._headroom_seen if now - t < win] + [(now, h)]
        target = max(i.queue_low_water, max(x for _, x in self._headroom_seen) // 2)
        return target, max(i.queue_max, target + self.cfg.researcher.max_concurrent)

    def _stack_room(self) -> int:
        """Researcher slots left for STACKING (winner crosses, the stack builder, the leg-factorial branch): their own
        ``researcher.max_stack``, never shared with ``max_side`` (live 10-05..10-07: max_side 1, held by debug fixes /
        tuning / ports, starved the winner crosses and the leg branch)."""
        n = sum(r.id in self._stack for r in self._research.values())
        return max(0, min(self.cfg.researcher.max_stack - n, self.cfg.researcher.max_concurrent - len(self._research)))

    def _side_room(self) -> int:
        """Researcher slots left for the driver's own request sources (debug fixes, winner crosses, tuning
        variants): together they hold at most ``researcher.max_side``, so the bandit always gets turns."""
        side = sum(r.id in self._side and r.id not in self._stack for r in self._research.values())
        return max(0, min(self.cfg.researcher.max_side - side, self.cfg.researcher.max_concurrent - len(self._research)))

    def _winx_retry(self, key: str) -> bool:
        """A non-backend failure frees the pair ONCE; the second failure keeps it booked (never a loop)."""
        tries = self.store.get_state(K_WINX_TRIES) or {}
        if tries.get(key, 0) >= 1:
            return False
        self.store.set_state(K_WINX_TRIES, {**tries, key: tries.get(key, 0) + 1})
        return True

    def _winner_cross_requests(self) -> list[CandidateRequest]:
        """Standing exploitation: every untried pair of CONFIRMED winners (loop.select.winner_pairs, most
        complementary first) gets ONE crossover request, outside the bandit, within ``researcher.max_stack``;
        a merge that confirms joins the pool (compounding), and so does a well-measured promising node (loop.legs.pool:
        crossed with the stack head only, like a banked one). An outage frees the pair."""
        room = self._stack_room()
        if room <= 0 or self.snapshot is None:
            return []
        book = self.store.get_state(K_WINX) or {}
        skip = {frozenset(k.split("+")) for k in book} | {
            frozenset(r.parent_ids[:2]) for r in self._research.values() if r.operator is Operator.CROSSOVER}
        out = [self.selector.winner_cross_request(self.store, self.snapshot, pair)
               for pair in winner_pairs(self.store, self.snapshot, skip, self.cfg.promotion.head_p,
                                        self.cfg.novelty.major_ratio, self.cfg.promotion.leg_p,
                                        pool=LG.pool(self.ctx, self.store, self.snapshot))[:room]]
        if out:
            self.store.set_state(K_WINX, {**book, **{"+".join(r.parent_ids): r.id for r in out}})
        return out

    def _stack_requests(self) -> list[CandidateRequest]:
        """Stacking unconfirmed small signals (loop.select.stack_plan): each new stack gets ONE crossover request of
        its legs on their carrier, outside the bandit, within ``researcher.max_stack``, at most STACK_MAX stacks in
        flight (researching, queued or unresolved). An outage frees it."""
        room = self._stack_room()
        if room <= 0 or self.snapshot is None or self.cfg.legs.enabled:    # retired by the leg factorial (loop.legs)
            return []
        book = self.store.get_state(K_STACK) or {}
        live = sum(r.id in book.values() for r in self._research.values()) + sum(
            "stack_legs=" in (n.notes or "") and n.status not in CLOSED_STATUSES for n in self.store.nodes())
        out = [self.selector.stack_request(self.store, self.snapshot, legs) for legs in stack_plan(
            self.store, self.snapshot, self._sigma_p(), {tuple(sorted(k.split("+"))) for k in book})
               [:max(0, min(room, STACK_MAX - live))]]
        if out:
            self.store.set_state(K_STACK, {**book, **{"+".join(sorted(r.parent_ids)): r.id for r in out}})
        return out

    def _design_smoke(self, acts: list[str]) -> None:
        """The running batch's heaviest point is memory-smoked before any point launches (4.6); an
        OOM climbs the adapter's ladder, and the step that passes is applied to every point."""
        for _ in range(8):
            if self._smoke is not None:
                if not self._smoke.done():
                    return
                fut, self._smoke = self._smoke, None
                try:
                    res = fut.result()
                except Exception as exc:  # noqa: BLE001 - a smoke that cannot run releases the batch
                    res = SmokeResult(False, f"smoke raised {exc!r}"[:300])
                pt = D.smoke_due(self.store)
                acts += D.smoke_done(self.store, res, self.book.ladder(self.store.get_node(pt[0]) if pt else None))
                self._dirty = True
            if (due := D.smoke_due(self.store)) is None:
                return
            node = self.store.get_node(due[0])
            if node is None:    # the heaviest point vanished: release the batch rather than raise every tick
                self.store.log("design_smoke", point=due[0], result="missing_node")
                acts += D.smoke_done(self.store, SmokeResult(False, f"smoke node {due[0]} no longer exists"), [])
                self._dirty = True
                return
            self._smoke = self.pool.submit(D.smoke_point, self.adapter, node.commit, node.id, due[1])
            acts.append(f"design memory smoke of heaviest point {node.id}" + (f" with {due[1]}" if due[1] else ""))

    # ------------------------------------------------------------- slot filling
    def _needs(self, nodes: Sequence[Node], by_node: Mapping[str, list[Run]]) -> list[S.RunNeed]:
        blocks, snap, out = S.assign_blocks(self.store.nodes(), self.cfg.seeds, self.store.runs()), self.snapshot, []
        share = self._block_sharers(blocks)
        for node in nodes:
            post = snap.nodes.get(node.id) if snap else None
            p = post.p_beats_incumbent if post else 0.0
            out += [((_TIER[n.stage], -p, node.created_ts, node.id), n)
                    for n in self._decide(node, by_node, blocks).needs
                    if not self._hold_control(n, node, blocks, share, by_node)]
        return [n for _, n in sorted(out, key=lambda x: x[0])]

    def _block_sharers(self, blocks: Mapping[str, int]) -> dict[tuple[str, int], int]:
        """(parent, screen block) -> its live children (queued / screening): who a control run there would serve."""
        out: dict[tuple[str, int], int] = {}
        for n in self.store.nodes(status=[NodeStatus.QUEUED, NodeStatus.SCREENING]):
            if n.id in blocks and not design_point(n):
                k = (n.parent_id or ROOT_NODE_ID, blocks[n.id])
                out[k] = out.get(k, 0) + 1
        return out

    def _hold_control(self, need: S.RunNeed, node: Node, blocks: Mapping[str, int],
                      share: Mapping[tuple[str, int], int], by_node: Mapping[str, list[Run]]) -> bool:
        """CONTROL SHARING (``seeds.control_wait_h``): hold a SCREENING node's missing screen-block control run while
        it would serve fewer than 2 live children of the parent and the node's first own screen run is younger than
        the wait; the node's own runs go ahead. A tuning / leg-factorial graduate (``tune_of=`` / ``legs_of=``) screens
        on a block of its own (loop.seeds.assign_blocks): no sibling can ever share it, so it never waits (live
        2026-10-08: a node, the L4 graduate on a node, held its a node controls 3 h for nobody)."""
        wait = self.cfg.seeds.control_wait_h
        notes = f" {node.notes or ''}"
        if wait <= 0 or node.status is not NodeStatus.SCREENING or need.node_id == node.id or design_point(node) \
                or " tune_of=" in notes or " legs_of=" in notes \
                or need.stage not in (Stage.BASELINE, Stage.PARENT) or node.id not in blocks \
                or need.seed not in S.screen_block(self.cfg.seeds, blocks[node.id]):
            return False
        if share.get((node.parent_id or ROOT_NODE_ID, blocks[node.id]), 0) >= 2:
            return False
        first = min((r.submitted_ts for r in by_node.get(node.id, ()) if r.stage is Stage.SCREEN and r.submitted_ts),
                    default=None)
        return first is None or self.clock() - first < 3600.0 * wait

    def _fill(self, acts: list[str], first: Sequence[S.RunNeed] = ()) -> None:
        book = self.book
        acts += book.promote_stuck()                 # stuck HIGH runs take free primary slots first
        free = book.free()
        by_node = self._runs_by_node()
        taken = {r.id for rs in by_node.values() for r in rs}
        nodes = {n.id: n for n in self.store.nodes()}
        owed = [(run, nodes[run.node_id]) for run in self.store.runs(status=[RunStatus.VOID, RunStatus.CRASH])
                if run.node_id in nodes and S.is_retryable(run, book.infra_retry)
                and (run.stage in S.CONTROL_STAGES or nodes[run.node_id].status in ACTIVE_STATUSES)]
        held = D.held(self.store) | DB.gate(         # a batch waiting on its memory smoke launches nothing; nor
            self.ctx, self.store, [n for n in nodes.values() if n.status in ACTIVE_STATUSES  # a node whose dataset
                                   or n.status is NodeStatus.QUEUED],                     # is building (it waits
            lambda n, why: self._set_status(n, NodeStatus.VOID, why, acts), self._ds_cache)    # in the queue)
        needs = [n for n in (*first, *self._needs([n for n in nodes.values() if n.status in ACTIVE_STATUSES], by_node))
                 if n.for_node not in held]
        # HIGH demand: HIGH runs owed / needed now, or SLURM-pending anywhere; LOW runs then keep off primary
        book.high_demand = sum(book.high(o, r.stage, r.seed) for r, o in owed) + sum(
            book.high(nodes[n.node_id], n.stage, n.seed) for n in needs if n.run_id not in taken and n.node_id in nodes) + sum(
            book.high(nodes.get(r.node_id), r.stage, r.seed) for r in self.store.runs(status=[RunStatus.RUNNING])
            if r.id in book.pending)

        def take(need: S.RunNeed) -> None:
            nonlocal free
            if free > 0 and need.run_id not in taken and need.node_id in nodes \
                    and book.pool(nodes[need.node_id], need.stage, need.seed):     # a LOW run / design point may wait
                taken.add(need.run_id)
                free -= 1
                if (run := book.launch(nodes[need.node_id], need.stage, need.seed)) is not None:
                    acts.append(f"submit {need.run_id} -> {run.status.value}")

        for run, owner in owed:                      # owed slots first
            if free > 0 and book.pool(owner, run.stage, run.seed):
                free -= 1
                acts.append(f"retry {run.id} -> {book.retry(run, owner).status.value}")
        for need in needs:
            take(need)
        queued = [n for n in nodes.values() if n.status is NodeStatus.QUEUED]
        choice, key = free > 0 and len(queued) >= 2, f"rank@{self.clock():.3f}"   # a ranking decides something
        lev = not choice or CG.use(self.ctx, self.store, "lever_prior", "rank", key)
        heads = []
        for head in self.selector.rank_queue(self.store, self.snapshot, queued, lev) if queued else ():
            if free <= 0:
                break
            if head.id in held:
                continue
            if not self._value_gate(head, acts):         # predicted not worth a screen (loop.world.predictive)
                continue
            nodes[head.id] = self._set_status(head, NodeStatus.SCREENING, "thompson head", acts)
            heads.append(head.id)
            for need in self._needs([nodes[head.id]], by_node):
                take(need)
        if choice and heads:                         # the heads a ranking launched are its outcome units
            CG.attach(self.store, "lever_prior", key, heads)
        if book.saturated():                         # no smoke could start soon: start no researcher
            if not self._research_paused:
                self.store.log("research_paused", pools=dict(book.pool_states or {}))
                acts.append("research paused: every pool full with our own runs pending; resumes when a slot frees")
            self._research_paused = True
            self._sync_inflight()
            return
        if self._research_paused:
            self.store.log("research_resumed", pools=dict(book.pool_states or {}))
            acts.append("research resumed: a slot is free")
        self._research_paused = False
        if self._llm_paused():
            return
        if self._llm_probing():                      # one probe session, nothing else, until a call works again
            if not self._research and (req := self._requeue.pop(0) if self._requeue
                                       else self.selector.next_request(self.store, self.snapshot)) is not None:
                self._research[self.pool.submit(self.researcher.research, req, self.store, self.snapshot)] = req
                self._sync_inflight()
                self.store.log("research_started", request_id=req.id, strategy=req.strategy.value,
                               operator=req.operator.value, parents=list(req.parent_ids), brief_kind=req.brief_kind,
                               backend=self._backend(), meta_id=req.meta_hypothesis_id, probe=True)
                acts.append(f"researcher started: {req.id} (LLM probe after a pause)")
            return
        while self._requeue and len(self._research) < self.cfg.researcher.max_concurrent:   # restart re-issues first
            req = self._requeue.pop(0)
            self._research[self.pool.submit(self.researcher.research, req, self.store, self.snapshot)] = req
            self.store.log("research_started", request_id=req.id, strategy=req.strategy.value,
                           operator=req.operator.value, parents=list(req.parent_ids), brief_kind=req.brief_kind,
                           backend=self._backend(), meta_id=req.meta_hypothesis_id, restarted=True)
            acts.append(f"researcher re-issued after restart: {req.id} {req.strategy.value}/{req.operator.value}")
        self._sync_inflight()
        if self._requeue:                            # every slot is taken by re-issues: nothing new this tick
            return
        for req in self._debug_requests():
            self._research[self.pool.submit(self.researcher.research, req, self.store, self.snapshot)] = req
            self._side.add(req.id)
            self.store.log("research_started", request_id=req.id, strategy=req.strategy.value,
                           operator=req.operator.value, parents=list(req.parent_ids), brief_kind=req.brief_kind,
                           backend=self._backend(), meta_id=None, deterministic_debug=True)
            acts.append(f"researcher started: {req.id} debug of crashed {req.parent_ids[0]} (fix-and-continue)")
        # The leg branch goes BEFORE the winner crosses: it is ONE request that unblocks a whole factorial, and the
        # crosses (a standing stream, every freed max_stack slot refilled the same tick) took every slot first:
        # live 10-07 17:54-22:00Z L4 on a node sat want=True / pending='' while 9 crosses a node x .. started.
        for req in LG.requests(self.ctx, self.store, self.snapshot, limit=self._stack_room()):  # the leg branch FIRST
            self._research[self.pool.submit(self.researcher.research, req, self.store, self.snapshot)] = req
            self._side.add(req.id)
            self._stack.add(req.id)
            self.store.log("research_started", request_id=req.id, strategy=req.strategy.value,
                           operator=req.operator.value, parents=list(req.parent_ids), brief_kind=req.brief_kind,
                           backend=self._backend(), meta_id=None)
            acts.append(f"researcher started: {req.id} leg-factorial branch on {req.parent_ids[0]}")
        for req in self._winner_cross_requests():
            self._research[self.pool.submit(self.researcher.research, req, self.store, self.snapshot)] = req
            self._side.add(req.id)
            self._stack.add(req.id)
            self.store.log("research_started", request_id=req.id, strategy=req.strategy.value,
                           operator=req.operator.value, parents=list(req.parent_ids), brief_kind=req.brief_kind,
                           backend=self._backend(), meta_id=req.meta_hypothesis_id, winner_cross=True)
            self.store.log("winner_cross", request_id=req.id, parents=list(req.parent_ids))
            acts.append(f"winner_cross queued {req.parent_ids[0]} x {req.parent_ids[1]} ({req.id})")
        for req in self._stack_requests():
            self._research[self.pool.submit(self.researcher.research, req, self.store, self.snapshot)] = req
            self._side.add(req.id)
            self._stack.add(req.id)
            self.store.log("research_started", request_id=req.id, strategy=req.strategy.value,
                           operator=req.operator.value, parents=list(req.parent_ids), brief_kind=req.brief_kind,
                           backend=self._backend(), meta_id=req.meta_hypothesis_id, stack=True)
            acts.append(f"stack queued {' x '.join(req.parent_ids)} ({req.id})")
        for req in MT.requests(self.ctx, self.store, self.snapshot, limit=self._side_room(), scarce=book.scarce()):
            self._research[self.pool.submit(self.researcher.research, req, self.store, self.snapshot)] = req
            self._side.add(req.id)
            self.store.log("research_started", request_id=req.id, strategy=req.strategy.value,
                           operator=req.operator.value, parents=list(req.parent_ids), brief_kind=req.brief_kind,
                           backend=self._backend(), meta_id=None)
            acts.append(f"researcher started: {req.id} tune/{req.operator.value} parents={list(req.parent_ids)}")
        for req in BF.port_requests(self.ctx, self.store, self.snapshot, self.selector, self._side_room(),
                                    {r.id for r in self._research.values()}):          # lowest priority
            self._research[self.pool.submit(self.researcher.research, req, self.store, self.snapshot)] = req
            self._side.add(req.id)
            self.store.log("research_started", request_id=req.id, strategy=req.strategy.value,
                           operator=req.operator.value, parents=list(req.parent_ids), brief_kind=req.brief_kind,
                           backend=self._backend(), meta_id=None, port_of=req.parent_ids[1])
            acts.append(f"researcher started: {req.id} backfill port of {req.parent_ids[1]} onto {req.parent_ids[0]}")
        self._sync_inflight()
        queued_n = len(self.store.nodes(status=[NodeStatus.QUEUED]))
        target, qmax = self._queue_marks()
        for _ in range(research_starts(queued_n, len(self._research), target, qmax,
                                       self.cfg.researcher.max_concurrent)):
            if (req := self.selector.next_request(self.store, self.snapshot)) is None:
                return
            self._research[self.pool.submit(self.researcher.research, req, self.store, self.snapshot)] = req
            self._sync_inflight()                    # the next request sees this one's family
            self.store.log("research_started", request_id=req.id, strategy=req.strategy.value,
                           operator=req.operator.value, parents=list(req.parent_ids),
                           brief_kind=req.brief_kind, backend=self._backend(),
                           meta_id=req.meta_hypothesis_id)
            acts.append(f"researcher started: {req.id} {req.strategy.value}/{req.operator.value} "
                        f"parents={list(req.parent_ids)}")

    def _sweep_smokes(self, acts: list[str]) -> None:
        self._smoke_swept = R.sweep_smokes(self.adapter, self.store, acts, self.clock(), self._smoke_swept)

    def _restore_inflight(self, acts: list[str]) -> None:
        """Requests the previous process left in flight (K_INFLIGHT) died with it: each is logged
        ``research_failed status=restart`` and QUEUED for re-issue (same id, brief and arm; a fresh session) up to
        MAX_RESTARTS times: a restart is not the request's attempt. The queue (``_requeue``) starts under
        ``researcher.max_concurrent`` like any session, ahead of new work (live 09-29..10-06: 217 were re-issued at
        once, uncapped, after restarts). A queued re-issue that never started is no new restart. A winner cross
        whose pair is no longer eligible (the stack head moved) is dropped and its booking freed."""
        live = {r.id for r in [*self._research.values(), *self._requeue]}
        pairs = None
        for e in self.store.get_state(K_INFLIGHT) or []:
            if e["id"] in live:
                continue
            again = "req" in e and (e.get("queued") or e.get("restarts", 0) < MAX_RESTARTS)
            req = build(CandidateRequest, e["req"], "driver.inflight.req") if again else None
            if req is not None and req.brief_kind == "tune":    # its pass was closed, or fails the gate (withdrawn
                cur = next((c for c in MT.book(self.store)["current"].values() if c["pending"] == req.id), None)
                base = self.store.get_node(cur["base"]) if cur is not None else None     # this tick: _tune_gate)
                if base is None or MT.gate(self.ctx, self.store, self.snapshot or self._refit(), base):
                    again, req = False, None
            key = "+".join(req.parent_ids[:2]) if req is not None else ""
            if req is not None and e.get("side") and (self.store.get_state(K_WINX) or {}).get(key) == req.id:
                if pairs is None:
                    snap = self.snapshot or self._refit()
                    pairs = {frozenset(p) for p in winner_pairs(self.store, snap, head_p=self.cfg.promotion.head_p,
                                                                major=self.cfg.novelty.major_ratio,
                                                                leg_p=self.cfg.promotion.leg_p,
                                                                pool=LG.pool(self.ctx, self.store, self.snapshot))}
                if frozenset(req.parent_ids[:2]) not in pairs:
                    self.store.set_state(K_WINX, {k: v for k, v in (self.store.get_state(K_WINX) or {}).items()
                                                  if k != key})
                    again = False
            if hasattr(self.adapter, "cancel_smokes"):   # its orphaned smoke job would recreate the worktree
                try:
                    if ids := self.adapter.cancel_smokes(f"loop-{e['id']}"):
                        acts.append(f"scancelled orphaned smoke job(s) {','.join(ids)} of {e['id']}")
                except Exception as exc:  # noqa: BLE001 - prepare_worktree still verifies the dest is gone
                    self.store.log("cancel_smokes_failed", request_id=e["id"], error=repr(exc)[:300])
            if not e.get("queued"):
                self.store.log("research_failed", request_id=e["id"], status="restart", reissued=again,
                               error="the loop restarted while this request was in flight")
            if not again:
                acts.append(f"research {e['id']} lost to a restart (not re-issued)")
                continue
            self._restarts[req.id] = e.get("restarts", 0) + (0 if e.get("queued") else 1)
            if e.get("smoke_reissues"):
                self._smoke_reissues[req.id] = int(e["smoke_reissues"])
            if e.get("side"):
                self._side.add(req.id)
            if e.get("stack"):
                self._stack.add(req.id)
            if req.bank_id:                          # its logged failure must not re-serve the idea
                B.mark(self.store, req.bank_id, req.id)
            self._requeue.append(req)                 # started by _fill under max_concurrent (and never while paused)
            acts.append(f"researcher re-issue queued after restart: {req.id} {req.strategy.value}/{req.operator.value}")
        self._sync_inflight()

    def _repair_smoke_timeouts(self, acts: list[str]) -> None:
        """Once (K_SMOKE_REPAIR): a node REJECTED ``smoke_failed`` with no commit whose smoke job never started
        (its smoke dir has no ``out/``: a queue timeout, rc 124) was no verdict. It is relabelled
        ``smoke_unavailable`` (never paid), its bandit credit is dropped, and its request is re-issued with the
        stored diff and hypothesis. A tuning proposal's counted failure is given back instead (the pass proposes
        its next variant itself)."""
        if self.store.get_state(K_SMOKE_REPAIR):
            return
        smoke_root, started = Path(self.ctx.state_dir) / "smoke", {}
        for e in self.store.events("research_started"):
            started[e.get("request_id")] = e
        kinds, fixed = self.store.get_state(K_KIND_PULLS, []) or [], []
        tune_bk = None
        for n in self.store.nodes(status=[NodeStatus.REJECTED]):
            m = re.search(r"\brequest=(\S+)", n.notes or "")
            d = smoke_root / f"loop-{m.group(1)}" if m else None
            if n.reject_reason != "smoke_failed" or n.commit or d is None or not (d / "smoke.yaml").exists() \
                    or (d / "out").exists():
                continue
            rid, ev = m.group(1), started.get(m.group(1)) or {}
            self.store.update_node(n.id, reject_reason=SMOKE_UNAVAILABLE_REASON)
            self.store.drop_pulls(n.id)
            kinds = [k for k in kinds if k.get("node_id") != n.id]
            note = lambda k: (mm.group(1) if (mm := re.search(rf"\b{k}=(\S+)", n.notes or "")) else "")  # noqa: E731
            if note("brief") == "tune":
                tune_bk = tune_bk or MT.book(self.store)
                cur = next((c for c in tune_bk["current"].values()
                            if rid.startswith(f"tune-t{c['batch']}-") and c["fails"] > 0), None)
                if cur is not None:
                    cur["fails"] -= 1
                fixed.append(f"{n.id} ({rid}: tuning failure given back)" if cur else f"{n.id} ({rid}: pass over)")
                continue
            if any(r.id == rid for r in self._research.values()):
                fixed.append(f"{n.id} ({rid} already in flight)")
                continue
            bank = note("bank")
            entry = next((e for e in B.load(self.cfg.idea_banks, self.store) if e["id"] == bank), None) if bank else None
            h = self.store.get_hypothesis(n.hypothesis_id or "")
            req = CandidateRequest(
                id=rid, task_id=self.ctx.task_id, strategy=n.strategy, operator=n.operator or Operator.DRAFT,
                island=n.island, parent_ids=tuple(ev.get("parents") or [n.parent_id or ROOT_NODE_ID]),
                brief=B.text(entry) if entry else (h.claim if h else ""), brief_kind=note("brief"),
                family_hint=n.family, meta_hypothesis_id=ev.get("meta_id"), bank_id=bank, persona=note("persona"))
            diff = Path(n.diff_path).read_text() if n.diff_path and Path(n.diff_path).exists() else ""
            cand = SimpleNamespace(hypothesis=h, diff_text=diff) if h else None
            self._reissue(req, cand, f"{SMOKE_UNAVAILABLE_REASON}: {n.id} was rejected on a smoke queue timeout")
            fixed.append(f"{n.id} -> re-issued {rid}")
        if tune_bk is not None:
            self.store.set_state(MT.K_TUNE, tune_bk)
        self.store.set_state(K_KIND_PULLS, kinds)
        self.store.set_state(K_SMOKE_REPAIR, {"ts": self.clock(), "nodes": fixed})
        if fixed:
            self._dirty = True
            acts.append("smoke-timeout repair: " + "; ".join(fixed))

    def _repair_outage(self, acts: list[str]) -> None:
        """Once (K_OUTAGE_REPAIR): undo what account outages (2026-10-03: a logged-out CLI ~10:30-19:30Z, the weekly
        limit 22:00-24:00Z) were counted for before an outage was recognised (``outage_requests``: the latest attempt
        ended backend_unavailable). A dose ladder whose knob exposure 'failed' only on such requests is re-opened
        (closed 'knob exposure failed Nx') or given its exposure back (a running 1x-only backrun), fails 0; a running
        tuning pass gets those failures back; review probes they served are served again. Bank ideas, component
        decisions and the quality table read ``outage_requests`` live; arm / kind pulls are node-keyed and an outage
        made no node, so no bandit reward was paid."""
        if self.store.get_state(K_OUTAGE_REPAIR):
            return
        from loop.contracts import outage_requests
        out, bk, fixed = outage_requests(self.store), MT.book(self.store), []
        ate = lambda c: c["fails"] > 0 and all(f"expose-t{c['batch']}-{k}" in out    # noqa: E731
                                               for k in range(1, c["fails"] + 1))
        for c in list(bk["batches"]):
            if c.get("kind") == "ladder" and str(c.get("why", "")).startswith("knob exposure failed") and ate(c) \
                    and str(c["batch"]) not in bk["current"]:
                bk["batches"].remove(c)
                bk["current"][str(c["batch"])] = {**{k: v for k, v in c.items()
                                                     if k not in ("why", "tuned", "best", "gpu_h", "closed")},
                                                  "fails": 0, "expose": "want", "pending": ""}
                fixed.append(f"t{c['batch']} ({c['base']}) re-opened: its {c['fails']} exposure failures were outages")
        for c in bk["current"].values():
            if c.get("kind") == "ladder" and c.get("expose") == "done" and not c.get("exposed") and ate(c):
                fixed.append(f"t{c['batch']} ({c['base']}) exposure asked again: {c['fails']} outage failure(s)")
                c.update(fails=0, expose="want", pending="")
            elif c.get("kind") != "ladder" and (n := sum(r.startswith(f"tune-t{c['batch']}-") for r in out)):
                c["fails"] = max(0, c["fails"] - n)
                fixed.append(f"t{c['batch']} ({c['base']}): {n} outage proposal failure(s) given back")
        self.store.set_state(MT.K_TUNE, bk)
        probes = [k for r in sorted(out) for k in getattr(self.selector, "unserve_probe", lambda *_: [])(self.store, r)]
        if probes:
            fixed.append(f"{len(probes)} review probe(s) served again: {', '.join(probes)}")
        self.store.set_state(K_OUTAGE_REPAIR, {"ts": self.clock(), "outage_requests": len(out), "fixed": fixed})
        self.store.log("outage_repair", outage_requests=len(out), fixed=fixed)
        if fixed:
            self._dirty = True
            acts.append("outage repair: " + "; ".join(fixed))

    def _sync_inflight(self) -> None:
        """K_INFLIGHT: the sessions in flight, then the queued restart re-issues (``queued``: never started)."""
        cur = [{"id": r.id, "strategy": r.strategy.value, "family": r.family_hint,
                "parents": list(r.parent_ids), "req": encode(r), "restarts": self._restarts.get(r.id, 0),
                "side": r.id in self._side, "stack": r.id in self._stack, "smoke_reissues": self._smoke_reissues.get(r.id, 0),
                **({"queued": True} if q else {})}
               for rs, q in ((self._research.values(), False), (self._requeue, True)) for r in rs]
        if self.store.get_state(K_INFLIGHT) != cur:
            self.store.set_state(K_INFLIGHT, cur)

    def _plan_only(self) -> list[str]:
        """``--dry-run``: what the next tick would do; nothing is submitted or stored, no LLM call."""
        real, self.store = self.store, StateOverlay(self.store)
        self._tune()
        try:
            return self._plan()
        finally:
            self.store = real

    def _plan(self) -> list[str]:
        acts: list[str] = []
        if self.store.get_node(ROOT_NODE_ID) is None:
            acts.append(f"would create root node at {self.adapter.root_commit()}")
        else:
            self.snapshot = self.world.fit(self.store)
        acts += [f"would poll {r.id} ({r.status.value})" for r in self.book.active()]
        by_node, blocks = self._runs_by_node(), S.assign_blocks(self.store.nodes(), self.cfg.seeds, self.store.runs())
        active = self.store.nodes(status=ACTIVE_STATUSES)
        for node in active:
            if (d := self._decide(node, by_node, blocks)).status is not None:
                acts.append(f"would move {node.id}: {node.status.value} -> {d.status.value} ({d.reason})")
        acts += [f"would submit {n.run_id} (for {n.for_node})" for n in self._needs(active, by_node)]
        if (lp := LG.plan(self.ctx, self.store, self.snapshot)) is not None:
            acts.append(f"would start leg factorial round {lp.get('round', 1)} on carrier {lp['carrier']}: {len(lp['legs'])} new leg(s) "
                        + ", ".join(f"{r['id']}[{r['subsystem']}] P={r['p']}" for r in lp["legs"])
                        + f"; carrier legs {lp['own']} (expose {lp['expose']}); shared "
                        + f"{[(s['key'], s['low'], s['high']) for s in lp['shared']]}; {lp['runs']}-run factorial on "
                        + f"{lp['seeds']} seeds + {lp['dose_runs']}-run dose design; dropped "
                        + "; ".join(f"{r['id']} ({r['why']})" for r in lp["dropped"]))
        if self.snapshot is not None:
            q = BF.plan(self.ctx, self.store, self.snapshot, self.clock())
            room = max(0, min(self.cfg.backfill.per_tick, self.cfg.backfill.max_active - len(BF.active(self.store))))
            acts += [f"would backfill {i.line()}" if self.cfg.backfill.enabled and k < room else f"backfill queue: {i.line()}"
                     for k, i in enumerate(q[:10])] + ([f"backfill queue: ... {len(q) - 10} more"] if len(q) > 10 else [])
        acts += DB.plan(self.ctx, self.store, self.snapshot)
        if (cur := self._audit_book()["current"]) is not None:
            acts.append(f"audit {cur['node']} running on seeds {cur.get('seeds') or list(self.cfg.seeds.audit)}"
                        + (f", incumbent {cur['vs']} paired" if cur.get("vs") else ""))
        elif self.cfg.seeds.audit_rotate:             # StateOverlay: the draw stays in memory
            acts.append(f"the next audit would draw never-used seeds "
                        f"{RM.draw(self.ctx, self.store, len(self.cfg.seeds.audit), 'dry-run', self.clock())}")
        if self.snapshot is not None:                # StateOverlay: the re-mint book stays in memory
            rm, rm_needs = RM.tick(self.ctx, self.store, self.snapshot, self.book.infra_retry, now=self.clock())
            acts += [f"would {a}" for a in rm] + [f"would submit {n.run_id} (re-mint)" for n in rm_needs
                                                   if self.store.get_run(n.run_id) is None]
        if self.snapshot is not None and (due := self._review_due()):
            acts.append(f"would start a big review ({due})")
        queued = len(self.store.nodes(status=[NodeStatus.QUEUED]))
        if research_starts(queued, 0, *self._queue_marks(), self.cfg.researcher.max_concurrent):
            acts.append(f"would start a researcher (queued={queued})")
        return acts + [f"dry run: nothing submitted, no LLM calls (slots {self.cfg.capacity}: {pool_use(self.ctx, self.store.runs())}, "
                       f"active {len(self.book.active())})"]
