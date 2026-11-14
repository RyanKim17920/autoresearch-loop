"""loop.runner — the v3 PLUMBING (docs/loop-core-v3.md §1): run rows, pools, SLURM, LLM sessions. No decisions.

``RunBook``   the only caller of ``adapter.submit`` / ``find`` / ``poll`` / ``cancel`` / ``repool``: slots per pool,
              importance routing (HIGH: primary first; LOW: preemptible first), the max-pending gate, starved
              pools, the OOM ladder (adapter rungs + activation checkpointing), retries, repooling of stuck runs
              (the full-primary overflow cap, 7b94306), and ``guard`` -- the seed-pool check before EVERY submit.
``Courtesy``  the cluster-courtesy backoff for the preemptible pools while a protected job (qwen) is PENDING.
``Research``  researcher sessions in a thread pool: start / harvest / persist in flight (K_INFLIGHT) / re-issue
              after a restart, and the LLM pause gate (account outages, the SDK breaker).
plus ``sweep_smokes`` (orphan smoke jobs), ``repair_outage`` (once: what outages were counted for), the data bank
tick (register / build), and ``IncumbentLabless`` (the incumbent only, once per switch).

The old ``loop.driver`` subclasses :class:`RunBook` (its importance / routing rules ride on the old lanes) and
imports ``Courtesy`` / ``nap`` / ``utc_now`` / the LLM gate from here; everything else it keeps its own copy of
until the cutover deletes it."""

from __future__ import annotations

import dataclasses
import os
import socket
import sqlite3
import sys
import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Sequence

from loop import databank as DB
from loop import infra as INFRA
from loop.agents import llm as LA
from loop.agents.briefs import in_prompt_dir
from loop.agents.llm import BackendUnavailable
from loop.contracts import (K_INFLIGHT, ROOT_NODE_ID, CandidateRequest, ExecutionAdapter, JobHandle, Node, Run,
                            RunStatus, Stage, SubmissionRejected, VoidType, is_oom, next_mitigation, oom_note,
                            outage_requests, stronger, with_oom_note)
from loop.spec import LoopContext, build, encode

ACTIVE_RUN = (RunStatus.SUBMITTING, RunStatus.RUNNING)
#: a preemptible pool SLURM has run 0 of our (pending) jobs on for this long is STARVED: no new jobs go there
#: beyond STARVED_PENDING pending ones
STARVE_S, STARVED_PENDING = 1800.0, 3
K_TEST_WAIT = "runbook.test_primary_wait"   # {"node:stage:seed": ts a TEST-stage run first found no free primary slot}
K_STALE_MOVED = "runbook.stale_moved"       # [run id]: runs stale-pending already moved once (never moved twice)
K_TEST_SPILLED = "runbook.test_spilled"     # [run id]: TEST-stage runs that spilled onto a preemptible pool after the wait
K_TEST_RESERVE = "runbook.test_reserve"     # v3-c6a: primary slots reserved for owed TEST runs this tick (the tick writes it)
K_CANCEL_RETRY = "runbook.cancel_retry"      # {run id: reason}: cancels whose scancel was not confirmed (retried per tick)
K_PEND_SINCE = "runbook.pending_since"   # {run_id: ts the run was first seen PENDING (the stale-pending clock)}
K_STARVED = "runbook.starved_since"   # {pool: ts since SLURM ran 0 of our jobs there while some were pending}
K_PENDING = "runbook.pending"         # run ids SLURM held PENDING at the last poll (`status`); = loop.selftune's
#: the OOM rung after the adapter's own: ``train.activation_checkpointing``. Since the crn-knobs base (3144ffa /
#: 065b87f, the v3 epoch) the first-step FlopCounterMode excludes the checkpoint recompute, so it no longer moves the
#: FLOP-keyed schedules or the FLOP stop (on older bases it did: +1/3 FLOPs/step). Still the last rung (slower); the
#: run's ``mitigation`` records it
ACT_CKPT = "activation_checkpointing"
# v3 review O3: stages the preemption backoff never drops -- a PARENT row (a wave's control, a paired parent seed), the
# BASELINE, an AUDIT, a PROMOTE (the TEST stages are exempt too): each is the reference something else pairs against
BACKOFF_EXEMPT = (Stage.PARENT, Stage.BASELINE, Stage.AUDIT, Stage.PROMOTE)
COURTESY_EVERY_S = 120.0  # squeue for the protected jobs at most this often
COURTESY_MARGIN = COURTESY_FLOOR = 4   # backoff: preemptible cap starts at held preemptible - 4, floor 4
# The protected job (qwen, QOS bottom) is started and PREEMPTED again within minutes, dozens of times a
# day (503066: 45 restarts on 2026-09-30); each start took it out of PENDING for one probe, which lifted
# the backoff and refilled every preemptible slot. The backoff now lifts only after the protected jobs
# stayed out of PENDING this long, and its cap never rises while it is on.
COURTESY_RELEASE_S = 1200.0
COURTESY_STEP_AFTER_S, COURTESY_STEP_EVERY_S, COURTESY_STEP = 3600.0, 1800.0, 4   # still pending: tighten
K_COURTESY = "driver.courtesy"        # {"on", "jobs", "cap"}: cluster-courtesy backoff in force
K_LLM_PAUSE = "driver.llm_pause"      # {active, kind, since, until, resets_at, probes, probing, ok_mark}: no new LLM
#                                       job while active; while probing (after a resume) one researcher and nothing else
LLM_PROBE_S, LLM_PROBE_MAX_S = 900.0, 3600.0   # an account refusal with no known reset: retried after 15 min, x2, <= 1 h
SMOKE_SWEEP_S = 600.0     # orphan smoke jobs (no live smoke() call waits on them) are scancelled this often
MAX_RESTARTS = 3          # re-issues of a request a restart killed (a restart is not the request's attempt)
K_OUTAGE_REPAIR = "runner.outage_repair"
K_LABLESS = "runner.labless"          # {node: {run_id, tries, next_ts, submission_id, error}}
LABLESS_BACKOFF_S = 3600.0


class SubmitDeferred(RuntimeError):
    """A prepared row :meth:`RunBook.submit_prepared` did not submit (tick budget spent / an sbatch hung): the row
    stays ``submitting`` and :meth:`RunBook.resolve_submitting` adopts or submits it next tick."""


def hung(exc: BaseException) -> bool:
    """The adapter's submit failed because the sbatch call itself timed out (``SubprocessRunner``: rc 124,
    'timeout after Ns'): it may still have gone through, so its row is reconciled, never voided."""
    msg = str(exc)
    return isinstance(exc, TimeoutError) or "rc=124" in msg or "timeout after" in msg


def utc_now(ts: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def nap(seconds: float, stop: Callable[[], bool], sleep: Callable[[float], None] = time.sleep,
        step: float = 1.0) -> None:
    """Sleep ``seconds`` in <= ``step`` slices, returning within one slice of ``stop()``."""
    while seconds > 0 and not stop():
        sleep(min(step, seconds))
        seconds -= step


class Watchdog:
    """The systemd watchdog without a dependency: ``WATCHDOG=1`` (and ``READY=1`` / ``STATUS=``) datagrams to
    ``$NOTIFY_SOCKET`` (the unit sets ``WatchdogSec`` + ``NotifyAccess=all``: ``uv run`` makes python a child of the
    unit's main PID), at most every ``every_s``; plus the mtime / content of ``path`` (``<state>/loop.heartbeat``),
    for a timer check where the unit has no watchdog. Silent without a socket; a send error is never fatal."""

    def __init__(self, path: str | os.PathLike[str] | None = None, *, every_s: float = 10.0,
                 clock: Callable[[], float] = time.time, env: Any = None) -> None:
        env = os.environ if env is None else env
        addr = env.get("NOTIFY_SOCKET", "")
        self.addr = "\0" + addr[1:] if addr.startswith("@") else addr
        self.path, self.every_s, self.clock, self.last = path, every_s, clock, -1e18
        self.sent: list[str] = []                       # what was sent (tests)

    @property
    def enabled(self) -> bool:
        return bool(self.addr)

    def notify(self, msg: str) -> bool:
        if not self.addr:
            return False
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sk:
                sk.connect(self.addr)
                sk.sendall(msg.encode())
            self.sent.append(msg)
            return True
        except OSError:
            return False

    def ping(self, status: str = "", force: bool = False) -> None:
        now = self.clock()
        if not force and now - self.last < self.every_s:
            return
        self.last = now
        self.notify("WATCHDOG=1" + (f"\nSTATUS={status[:200]}" if status else ""))
        if self.path:
            try:
                tmp = f"{self.path}.tmp"
                with open(tmp, "w") as fh:
                    fh.write(f"{now:.0f} {utc_now(now)} {status[:200]}\n")
                os.replace(tmp, self.path)
            except OSError:
                pass


class RunBook:
    """Run-row bookkeeping: the only caller of ``adapter.submit`` / ``find`` / ``poll`` / ``cancel``. The routing
    hooks (:meth:`high`, :meth:`pool`, :meth:`promote_stuck`, :meth:`_oom_ok`, :meth:`_cancel_keeps`) are the v3
    rules here; ``loop.driver.RunBook`` overrides them with the old lanes' rules."""

    def __init__(self, ctx: LoopContext, store: Any, adapter: ExecutionAdapter, *,
                 clock: Callable[[], float] = time.time, extra_rungs: Sequence[str] = (),
                 guard: Callable[[Run], None] | None = None) -> None:
        self.ctx, self.store, self.adapter, self.clock = ctx, store, adapter, clock
        self.infra_retry = int(ctx.spec.caps.infra_retry)
        self.extra_rungs = tuple(extra_rungs)   # OOM rungs after the adapter's own (v3: ACT_CKPT)
        self.guard = guard                      # raises to refuse a submit (v3: core.seeds.check_submit)
        self.importance: Callable[[Node | None, Stage | None, int | None], bool] = \
            lambda _n, stage, _s: stage is not Stage.SCREEN       # v3: a run a verdict waits on is HIGH
        self.pending: set[str] = set()          # running rows SLURM still held PENDING at the last poll
        #: pool -> (rows SLURM ran, rows it held PENDING) at the last poll; None before the first one. Rebound,
        #: never mutated: the researchers' threads read it (smoke_pool)
        self.pool_states: dict[str, tuple[int, int]] | None = None
        #: pool -> our smokes PENDING there at the last poll (:meth:`limits` holds new submits to it); rebound, never mutated
        self.smoke_pending: dict[str, int] = {}
        self.restarts: dict[str, int] = {}      # running row -> SLURM requeues of its job (preemptions)
        self.repool_due: set[str] = set()       # rows SLURM just requeued >= slurm.repool_restarts times
        self.fill_only: set[tuple[str, int]] = set()   # (node, seed) of this tick's idle-fill runs: preemptible first, else a free primary slot (slurm.fill_on_primary)
        self._fill_runs: set[str] | None = None  # run ids launched as idle fill (event ``run_fill``; read once)
        self.high_demand = 0                   # HIGH runs waiting (set by the slot fill)
        #: v3-c6a: TEST-stage runs owed (an open TEST block's run not launched yet; the tick sets it before the poll):
        #: that many primary slots are RESERVED for them -- no other run takes them (``slurm.test_reserve``)
        self.test_reserve = 0
        self.courtesy_cap: int | None = None    # total preemptible slots while a protected job waits
        self.acts: list[str] = []               # OOM relaunches since the caller last drained them
        #: pool -> {limit, others, loop, effective} (``adapter.pool_capacity``), read once per poll (= per tick);
        #: None = not read yet this tick. {} = no association limit known (the declared slots stand)
        self._cap: dict[str, dict[str, int]] | None = None
        self._cap_err = False
        self._free: dict[str, int] | None = None   # partition -> startable GPUs (adapter.free_gpus), read once per poll
        self._free_err = False
        self._kept: dict[str, int] = {}         # idle churn: resumable fill run -> restarts when it was last kept

    def active(self) -> list[Run]:
        return self.store.runs(status=ACTIVE_RUN)

    def _smoke_home(self) -> str:
        """Where the smoke reserve sits and a smoke queues: the first preemptible pool SLURM keeps up with
        (:meth:`smoke_blocked`), else the primary (non-preemptible) one -- a smoke never queues behind our own
        PENDING runs (live 10-13: research smokes starved behind our queued preemptible runs)."""
        pools = self.ctx.cfg.pools
        if self.smoke_on_primary():
            return next(p.name for p in pools if not p.preemptible)
        return next((p.name for p in pools if p.preemptible and not self.smoke_blocked(p.name)),
                    next((p.name for p in pools if not p.preemptible), pools[0].name))

    def smoke_on_primary(self) -> bool:
        """``slurm.smoke_pool: primary`` and a primary (non-preemptible) pool exists: smokes queue there (11-08)."""
        return str(getattr(self.ctx.cfg.slurm, "smoke_pool", "auto") or "auto").lower() == "primary" \
            and any(not p.preemptible for p in self.ctx.cfg.pools)

    def smoke_blocked(self, pool: str) -> bool:
        """A preemptible pool a smoke must not queue on: SLURM runs none of our jobs there (:meth:`idle_dead`), or
        holds more than ``slurm.smoke_skip_pending`` (0: ANY) of ours PENDING there -- a smoke would wait behind them
        (same user / account / QoS: priority order; ``scontrol top`` is not enabled for users on this cluster).
        Unknown before the first poll: not blocked. (c4j review M3 raised the default to 8 because 0 sent every smoke
        to primary; since then ``slurm.smoke_primary_max`` caps primary smokes, and 8 never triggered at the live
        6-8 pending, so smokes queued behind our own LOW runs again: v3-c5 R3 restores 0.)"""
        st = self.pool_states
        if st is None or not any(p.name == pool and p.preemptible for p in self.ctx.cfg.pools):
            return False
        if self._assoc_full():
            return False         # primary would hold the smoke at AssocGrpGRES: idle is the better queue (10-21)
        run_n, pend_n = st.get(pool, (0, 0))
        return run_n == 0 or pend_n > int(getattr(self.ctx.cfg.slurm, "smoke_skip_pending", 0))

    def diag_jobs(self) -> int:
        """Feature-diagnostic jobs (``loop.agents.diagnose``, on the primary pool) presumed alive: submitted, no report
        yet, and younger than twice their time limit (``ideation.diagnose_job_min``, + 30 min of queueing). From the
        store state, no squeue: :meth:`used` runs many times a tick."""
        try:
            from loop.agents.diagnose import K_JOBS
            jobs = self.store.get_state(K_JOBS) or {}
        except Exception:  # noqa: BLE001 - no diagnose state: none
            return 0
        lim = 2 * 60.0 * float(getattr(getattr(self.ctx.cfg, "ideation", None), "diagnose_job_min", 15) or 15) + 1800.0
        now, n = self.clock(), 0
        for j in jobs.values() if isinstance(jobs, dict) else ():
            if not isinstance(j, dict) or not j.get("job") or now - float(j.get("ts") or 0.0) > lim:
                continue
            n += not (j.get("out") and os.path.isfile(str(j["out"])))
        return n

    def used(self) -> Counter:
        """Slots held per pool, as runs see them: active runs plus the adapter's in-flight smoke jobs and the live
        feature-diagnostic jobs (:meth:`diag_jobs`, on primary), and on the smoke home pool at least
        ``slurm.smoke_reserve`` slots held for smokes (live: 6 smoke jobs PENDING behind our own 40 runs left the
        researchers idle). The reserve sits on PRIMARY only while a smoke is actually in flight (queued or running):
        an idle reserve there kept a primary slot empty for hours (c4j review M3); else on the first preemptible
        pool."""
        pools, smokes = self.ctx.cfg.pools, getattr(self.adapter, "smokes", None)
        used = Counter(r.pool or pools[0].name for r in self.active())
        used.update({k: v for k, v in (smokes or {}).items() if v > 0})
        if (nd := self.diag_jobs()) > 0:
            used[next((p.name for p in pools if not p.preemptible), pools[0].name)] += nd
        if smokes is not None and any(p.preemptible for p in pools) and self.ctx.cfg.slurm.smoke_reserve > 0:
            home = self._smoke_home()
            if not any(p.name == home and p.preemptible for p in pools) and sum(max(0, v) for v in smokes.values()) <= 0 \
                    and not (self.smoke_on_primary() and getattr(self.ctx.cfg.slurm, "smoke_primary_hold", False)):
                home = next(p.name for p in pools if p.preemptible)    # no smoke waiting: never idle a primary slot
            used[home] += max(0, self.ctx.cfg.slurm.smoke_reserve - max(0, smokes.get(home, 0)))
        return used

    def high(self, node: Node | None, stage: Stage | None, seed: int | None = None) -> bool:
        """HIGH importance (primary first): ``importance`` -- by default every run but a screen."""
        return self.importance(node, stage, seed)

    def capacity(self) -> dict[str, dict[str, int]]:
        """The association-limited pools' real capacity (``adapter.pool_capacity``: limit, GPUs held by our OTHER
        jobs, the loop's own, effective), cached for the tick (:meth:`poll` clears it). {} without the probe, with
        ``slurm.effective_capacity`` off, or when it fails (the previous reading is kept on a failure)."""
        if not getattr(self.ctx.cfg.slurm, "effective_capacity", True) or \
                not callable(getattr(self.adapter, "pool_capacity", None)):
            return {}
        if self._cap is None:
            try:
                self._cap = {k: dict(v) for k, v in (self.adapter.pool_capacity() or {}).items()}
                self._cap_err = False
            except Exception as exc:  # noqa: BLE001 - unknown is not a verdict: the declared slots stand
                if not self._cap_err:
                    self.store.log("capacity_probe_failed", error=repr(exc)[:300])
                self._cap_err = True
                self._cap = {}
        return self._cap

    def free_gpus(self, pool: str) -> int | None:
        """GPUs the partition of ``pool`` can start a job on now (``adapter.free_gpus``, cached for the tick). None = no
        probe (switched off, no adapter probe, or the partition is unknown: no new-submit cap); a probe failure gives
        ``slurm.max_new_per_tick`` (the small constant), logged once per outage."""
        sl = self.ctx.cfg.slurm
        if not getattr(sl, "free_gpu_probe", True) or not callable(getattr(self.adapter, "free_gpus", None)):
            return None
        if self._free is None:
            try:
                self._free = {k: int(v) for k, v in (self.adapter.free_gpus() or {}).items()}
                self._free_err = False
            except Exception as exc:  # noqa: BLE001 - unknown is not a verdict: a small constant
                if not self._free_err:
                    self.store.log("free_gpu_probe_failed", error=repr(exc)[:300])
                self._free_err = True
                self._free = {}
        if self._free_err:
            return int(getattr(sl, "max_new_per_tick", 4))
        return self._free.get(next((p.partition for p in self.ctx.cfg.pools if p.name == pool), ""))

    def cap_note(self) -> str:
        """`` cap=primary:2/8(other 6)`` for the heartbeat: effective slots of every association-limited pool, the
        limit, and the GPUs our other jobs hold; '' when no pool is cut."""
        parts = [f"{n}:{c['effective']}/{c['limit']}(other {c['others']})" for n, c in self.capacity().items()
                 if c["effective"] < next((p.slots for p in self.ctx.cfg.pools if p.name == n), 0)]
        return " cap=" + ",".join(parts) if parts else ""

    def _primary_capped(self) -> bool:
        """An association-limited primary pool is cut below its declared slots by OTHER jobs of the user (the operator's
        sweeps hold the GrpTRES budget): a slot freed there is taken by those jobs' own pending queue, so waiting for it
        is no plan. False without a capacity reading (the declared slots stand)."""
        cap = self.capacity()
        return bool(cap) and any(not p.preemptible and p.name in cap and int(cap[p.name]["effective"]) < p.slots
                                 for p in self.ctx.cfg.pools)

    def _assoc_full(self) -> bool:
        """An association-limited primary pool has no free effective slot (runs + smokes >= its limit): a smoke or
        run queued there would wait at AssocGrpGRES, so smokes and screens spill to the idle pool (operator 10-21).
        Never calls :meth:`used` (it is read by :meth:`smoke_blocked`, which :meth:`used` calls)."""
        cap = self.capacity()
        if not cap:
            return False
        lim, smokes = self.limits(gate=False), getattr(self.adapter, "smokes", None) or {}
        first = self.ctx.cfg.pools[0].name
        runs = Counter(r.pool or first for r in self.active())
        return any(p.name in cap and not p.preemptible
                   and runs[p.name] + max(0, smokes.get(p.name, 0)) >= lim.get(p.name, p.slots)
                   for p in self.ctx.cfg.pools)

    def limits(self, gate: bool = True) -> dict[str, int]:
        """Slots per pool in force: as declared, the preemptible pools cut to ``courtesy_cap`` in total, and
        (``gate``) on a pool where ``slurm.max_pending_per_pool`` of our training runs were PENDING at the last poll,
        cut to those SLURM ran plus that many (smoke jobs and the smoke reserve not counted against it). A pool
        where fewer were PENDING is not cut: only PENDING jobs count against the cap, so a pool that starts our jobs
        at once fills in one tick (live 2026-10-05: cutting every pool to running + 2 let 2 jobs start per tick,
        about what finished, so 29 of 64 GPUs sat idle for hours with 19 nodes queued and 0 of ours pending)."""
        left, out = self.courtesy_cap, {}
        for p in self.ctx.cfg.pools:
            out[p.name] = p.slots if not p.preemptible or left is None else min(p.slots, left)
            if p.preemptible and left is not None:
                left -= out[p.name]
            if p.preemptible and self.starved(p.name):     # keep at most a few jobs pending there
                out[p.name] = min(out[p.name], (self.pool_states or {}).get(p.name, (0, 0))[0] + STARVED_PENDING)
        for name, c in self.capacity().items():          # the association's GPU cap less our other jobs' GPUs
            if name in out:
                out[name] = min(out[name], int(c["effective"]))
        if gate and self.smoke_pending:                  # smokes go first: no new submit to a pool where one waits
            used = self.used()
            for name, n in self.smoke_pending.items():
                if n > 0 and name in out:
                    out[name] = min(out[name], used[name])
        cap, st = self.ctx.cfg.slurm.max_pending_per_pool, self.pool_states
        if gate and cap > 0 and st is not None:          # unknown before the first poll: no gate
            used, first = self.used(), self.ctx.cfg.pools[0].name
            runs = Counter(r.pool or first for r in self.active())
            for p in self.ctx.cfg.pools:                 # used - runs = smoke jobs + the smoke reserve
                run_n, pend_n = st.get(p.name, (0, 0))
                # Runs launched since the poll are ones SLURM has not reported running, i.e. PENDING until proven
                # otherwise, whether or not the pool held pending jobs at the poll. A pool with nothing pending used to
                # fill its whole declared width in ONE tick (live 10-22 10:39Z: 50 screens to an idle pool declared 56
                # with ~3 GPUs free; 43 sat PENDING, aged the queue, starved smokes). NEW submits per tick are now
                # capped to what the partition can start: max(2, free GPUs); max_new_per_tick when the probe fails.
                free = self.free_gpus(p.name)
                if free is not None:
                    new = max(cap - pend_n, max(2, free)) if pend_n < cap else 0
                    out[p.name] = min(out[p.name], (run_n + pend_n + new if new else run_n + cap)
                                      + used[p.name] - runs[p.name])
                    continue
                if pend_n > 0:                           # no probe: SLURM already queues ours there, so do this tick's new ones
                    pend_n = max(pend_n, runs[p.name] - run_n)
                if pend_n < cap:
                    continue                             # SLURM keeps up there: no gate
                out[p.name] = min(out[p.name], run_n + cap + used[p.name] - runs[p.name])
        return out

    def starved(self, pool: str) -> bool:
        """SLURM has run none of our jobs on preemptible ``pool`` for STARVE_S while some were pending."""
        since = (self.store.get_state(K_STARVED) or {}).get(pool)
        return since is not None and self.clock() - float(since) > STARVE_S

    def free(self) -> int:
        lim, used = self.limits(), self.used()
        return sum(max(0, n - used[k]) for k, n in lim.items())

    def idle(self) -> int:
        """GPU slots that would otherwise idle, for scaling the research sessions (``Research.limit``): the
        :meth:`free` slots plus those our idle-fill / relax runs hold (running or PENDING: preemptible runs with no
        decision value, the first thing a new idea's screen should displace). Our other PENDING runs count as used --
        they have value and wait for SLURM, not for ideas (v3-c5 A3: free() alone was > 12 on 2 of 128 live ticks
        because fill held the slack, so ~4.7 sessions ran under ``ideation.llm_slots_max`` 8)."""
        fills = self.fill_runs()
        return self.free() + sum(1 for r in self.active() if r.id in fills)

    def headroom(self) -> int:
        """Slots our runs could fill soon, for sizing the idea queue: the ungated free slots of every pool SLURM
        keeps up with (fewer than ``slurm.max_pending_per_pool`` of our runs PENDING there), none elsewhere. Not
        :meth:`free`: on a pool holding max_pending_per_pool of our runs PENDING the gate cuts it, and runs submitted
        since the last poll are counted (live 2026-10-04: 35 of 64 idle, 0 pending, free() = 4 -> queue target 2 ->
        ~3 researchers instead of ~16, under the old gate)."""
        lim, used = self.limits(gate=False), self.used()
        cap, st = self.ctx.cfg.slurm.max_pending_per_pool, self.pool_states
        return sum(max(0, n - used[k]) for k, n in lim.items()
                   if cap <= 0 or st is None or st.get(k, (0, 0))[1] < cap)

    def saturated(self) -> bool:
        """No free slot anywhere and every pool holds >= ``researcher.pause_pending`` of our runs PENDING, or runs
        none of ours while some wait (a starved pool): a new researcher's smoke could not start soon (live
        2026-10-02: 15 researchers' smokes queued past SMOKE_WAIT_S behind 23 of our own pending runs). Unknown
        before the first poll: not saturated."""
        n, st = self.ctx.cfg.researcher.pause_pending, self.pool_states
        if n <= 0 or st is None or self.free() > 0:
            return False
        return all(p >= n or (p > 0 and r == 0) for r, p in (st.get(q.name, (0, 0)) for q in self.ctx.cfg.pools))

    def reserved(self) -> int:
        """Primary slots held for owed TEST runs (:attr:`test_reserve`; 0 when ``slurm.test_reserve`` is off or no
        preemptible pool exists -- then primary is the only home of everything)."""
        if not getattr(self.ctx.cfg.slurm, "test_reserve", True) or self.test_reserve <= 0 \
                or not any(p.preemptible for p in self.ctx.cfg.pools):
            return 0
        return int(self.test_reserve)

    def _open(self, preemptible: bool, gate: bool = True, test: bool = False) -> list[str]:
        """Pools of that kind with a free slot. A non-TEST request (``test`` False) sees the primary pools' free slots
        minus :meth:`reserved` (v3-c6a: an owed TEST run is first on primary, before every other kind)."""
        lim, used = self.limits(gate), self.used()
        left = 0 if preemptible or test else self.reserved()
        out = []
        for p in self.ctx.cfg.pools:
            if p.preemptible is not preemptible:
                continue
            free = lim[p.name] - used[p.name]
            take = min(left, max(0, free))
            left -= take
            if free - take > 0:
                out.append(p.name)
        return out

    def test_room(self, node: Node | None, stage: Stage | None, seed: int | None) -> int:
        """Free slots on the pool :meth:`pool` would place this TEST-stage run on now (0: none takes it): a primary
        pool's free slots counted WITH the owed-TEST reserve (it is held for exactly these runs), a spill pool's
        free slots. ``Tick._launch`` places a test's candidate + control of one seed only when the pair fits whole.
        Like :meth:`pool` it starts the run's TEST spill-wait clock (``K_TEST_WAIT``) on its first call."""
        got = self.pool(node, stage, seed)
        if not got:
            return 0
        lim, used = self.limits(), self.used()
        return max(0, lim.get(got, 0) - used.get(got, 0))

    def test_primary_only(self, stage: Stage | None) -> bool:
        """A TEST-stage run (``loop.tick.TEST_STAGES``: CONFIRM / REMINT on fresh test seeds) waits for a primary
        slot instead of spilling onto a preemptible pool (``slurm.test_primary_only``, default on): it is never
        resumed after a preemption, so every preemption restarts it from step 0 (live 10-13: 20 preemptions /
        10.2 GPU-h lost in 4 h). Only while a primary (non-preemptible) pool exists -- else nothing could run it."""
        if stage is None or not getattr(self.ctx.cfg.slurm, "test_primary_only", True):
            return False
        from loop.tick import TEST_STAGES                # lazy: loop.tick imports this module
        return stage in TEST_STAGES and any(not p.preemptible for p in self.ctx.cfg.pools)

    def _test_spill(self, node: Node | None, stage: Stage | None, seed: int | None) -> str:
        """No free primary slot for a TEST-stage run: '' (it waits) until it has waited ``slurm.test_primary_wait_s``
        (default 3 h; 0 = never spill), then a free preemptible pool. The wait starts the first time this (node,
        stage, seed) found no primary slot, kept in the store state (``K_TEST_WAIT``: survives a restart; cleared
        when the run launches)."""
        wait = float(getattr(self.ctx.cfg.slurm, "test_primary_wait_s", 3 * 3600.0))
        if node is None or seed is None or wait <= 0:
            return ""
        key, now = f"{node.id}:{getattr(stage, 'value', stage)}:{seed}", self.clock()
        book = dict(self.store.get_state(K_TEST_WAIT) or {})
        since = book.setdefault(key, now)
        if since == now:
            self.store.set_state(K_TEST_WAIT, book)
        if now - float(since) < wait:
            return ""
        return next(iter(self._open(True)), "")

    def prune_test_wait(self, live: set[tuple[str, int]]) -> int:
        """Drop the TEST-wait entries (``K_TEST_WAIT``) whose (node, seed) is no longer in an open TEST block (the
        block closed before the run launched): a stale wait would let a later run of that key spill at once."""
        book = dict(self.store.get_state(K_TEST_WAIT) or {})
        keep = {k: v for k, v in book.items()
                if (lambda p: len(p) == 3 and p[2].lstrip("-").isdigit() and (p[0], int(p[2])) in live)(k.rsplit(":", 2))}
        if len(keep) != len(book):
            self.store.set_state(K_TEST_WAIT, keep)
        return len(book) - len(keep)

    def spilled(self) -> set[str]:
        """TEST-stage run ids that took a preemptible pool after waiting for a primary one: the only TEST-stage
        runs resumed after a preemption (``loop.tick`` ``no_resume`` exempts them)."""
        return set(self.store.get_state(K_TEST_SPILLED) or [])

    def _note_test_pool(self, run: Run) -> None:
        """A TEST-stage run launching on a preemptible pool spilled there (``pool``): remember it and arm its
        resume (``adapter.resume_runs``); a launch clears its wait either way."""
        if not self.test_primary_only(run.stage):
            return
        if self.test_reserve > 0:                     # this owed run now holds its slot (counted in used())
            self.test_reserve -= 1
        book = dict(self.store.get_state(K_TEST_WAIT) or {})
        if book.pop(f"{run.node_id}:{getattr(run.stage, 'value', run.stage)}:{run.seed}", None) is not None:
            self.store.set_state(K_TEST_WAIT, book)
        self._mark_spilled(run)

    def _mark_spilled(self, run: Run) -> None:
        """``run`` (a TEST-stage run) sits on a preemptible pool: remember it as spilled and arm its resume."""
        pre = {p.name for p in self.ctx.cfg.pools if p.preemptible}
        sp = self.spilled()
        if run.pool in pre and run.id not in sp:
            self.store.set_state(K_TEST_SPILLED, sorted(sp | {run.id}))
            self.store.log("test_spilled", run_id=run.id, pool=run.pool)
            sp.add(run.id)
        if run.id in sp and isinstance(getattr(self.adapter, "resume_runs", None), set):
            self.adapter.resume_runs.add(run.id)

    def pool(self, node: Node | None = None, stage: Stage | None = None, seed: int | None = None) -> str:
        """Where a run goes. HIGH: a primary (non-preemptible) pool, else a preemptible one, else the first. LOW: a
        preemptible pool, else primary only while no HIGH run waits ('' = it waits). An idle-fill run
        (``fill_only``): a preemptible pool, else (``slurm.fill_on_primary``, no HIGH run waiting) a free primary slot beyond the owed-TEST reserve, else nothing. A TEST-stage
        run (:meth:`test_primary_only`): a primary pool or nothing ('' = it waits for a primary slot), but a
        preemptible one after ``slurm.test_primary_wait_s`` (:meth:`_test_spill`; then resumed after a preemption)."""
        pools = self.ctx.cfg.pools
        if not any(p.preemptible for p in pools):
            return next(iter(self._open(False)), pools[0].name)
        if self.test_primary_only(stage):
            if prim := next(iter(self._open(False, test=True)), ""):
                return prim
            if getattr(self.ctx.cfg.slurm, "test_spill_capped", True) and self._primary_capped():
                # 10-21: primary is cut by the operator's other jobs (effective capacity): a TEST run would wait out
                # test_primary_wait_s for a slot those jobs refill the moment it frees -> spill to idle now (resumed
                # after a preemption like any spilled TEST run, see _note_test_pool)
                return next(iter(self._open(True)), "")
            return self._test_spill(node, stage, seed)
        if node is not None and seed is not None and (node.id, seed) in self.fill_only:
            # 10-22: a fill run with no open preemptible slot (the idle pool at its pending gate) takes a free PRIMARY slot
            # (``_open(False)`` already keeps the owed-TEST reserve back) unless a HIGH run waits for one
            on_primary = getattr(self.ctx.cfg.slurm, "fill_on_primary", True) and not self.high_demand
            return next(iter(self._open(True) + (self._open(False) if on_primary else [])), "")
        if self.high(node, stage, seed):              # nowhere free: queue on the first pool (a preemptible one while
            home = pools[0].name                      # primary slots are reserved for owed TEST runs)
            if self.reserved() and not next(p for p in pools if p.name == home).preemptible:
                home = next(p.name for p in pools if p.preemptible)
            return next(iter(self._open(False) + self._open(True)), home)
        return next(iter(self._open(True) + ([] if self.high_demand else self._open(False))), "")

    def idle_dead(self, pool: str) -> bool:
        """A preemptible pool SLURM runs none of our jobs on at the last poll (pending-aware: a PENDING row is
        not running). Unknown before the first poll."""
        st = self.pool_states
        return st is not None and st.get(pool, (0, 0))[0] == 0

    def smoke_pool(self) -> str:
        """Where a smoke job goes: a pool with a free slot (the smoke reserve counts as free), preemptible
        first; else it queues on the smoke home pool, where it holds a slot runs then do not refill (on primary
        while the courtesy backoff bars new preemptible jobs). A preemptible pool SLURM runs 0 of our jobs on, or
        holds our runs PENDING on (:meth:`smoke_blocked`), is skipped: the smoke goes to primary, where the smoke
        reserve then sits (live: low-QoS smokes queued past SMOKE_WAIT_S behind our own pending runs)."""
        lim, smokes = self.limits(gate=False), getattr(self.adapter, "smokes", None) or {}
        runs = Counter(r.pool or self.ctx.cfg.pools[0].name for r in self.active())
        prim = next((p.name for p in self.ctx.cfg.pools if not p.preemptible), self.ctx.cfg.pools[0].name)
        pools = [p for p in sorted(self.ctx.cfg.pools, key=lambda p: not p.preemptible)
                 if not (p.preemptible and self.smoke_blocked(p.name))]
        if self.smoke_on_primary():                           # queue on primary, free or not: limits() then holds
            got = prim                                        # new primary submits while it is PENDING (11-08)
        elif len(pools) < len(self.ctx.cfg.pools):            # a blocked preemptible pool: primary, free or not
            got = next((p.name for p in pools if runs[p.name] + max(0, smokes.get(p.name, 0)) < lim[p.name]), prim)
        else:
            got = next((p.name for p in pools if runs[p.name] + max(0, smokes.get(p.name, 0)) < lim[p.name]),
                       self.ctx.cfg.pools[0].name if self.courtesy_cap is not None else self._smoke_home())
        if self.reserved() and any(p.name == got and not p.preemptible for p in self.ctx.cfg.pools) \
                and runs[got] + max(0, smokes.get(got, 0)) + self.reserved() >= lim[got]:
            got = self._least_pending_preemptible() or got       # v3-c6a: primary's last free slots are a test's
        return self._cap_primary_smokes(got, smokes)

    def _least_pending_preemptible(self) -> str:
        pre = [p.name for p in self.ctx.cfg.pools if p.preemptible]
        st = self.pool_states or {}
        return min(pre, key=lambda n: (n in st and st[n][0] == 0, st.get(n, (0, 0))[1], pre.index(n))) if pre else ""

    def _cap_primary_smokes(self, got: str, smokes: Any) -> str:
        """At most ``slurm.smoke_primary_max`` (2; 0 = no cap) smokes in flight on the primary (non-preemptible) pool:
        past it a smoke queues on the preemptible pool holding the fewest of our runs PENDING (live c4j review: every
        smoke landed on primary and held its slots). Primary stays the answer when no preemptible pool exists."""
        cap = int(getattr(self.ctx.cfg.slurm, "smoke_primary_max", 2) or 0)
        pools = self.ctx.cfg.pools
        if cap <= 0 or not any(p.name == got and not p.preemptible for p in pools) or smokes.get(got, 0) < cap:
            return got
        pre = [p.name for p in pools if p.preemptible]
        if not pre:
            return got
        st = self.pool_states or {}
        return min(pre, key=lambda n: (n in st and st[n][0] == 0, st.get(n, (0, 0))[1], pre.index(n)))

    def promote_stuck(self) -> list[str]:
        """A HIGH run SLURM holds PENDING on a preemptible pool moves to a free primary slot (scancel + the same
        launch there: a free retry, same row and attempt). One SLURM already requeued ``slurm.repool_restarts`` times
        moves to primary even when it is full, while fewer than :meth:`_overflow_cap` of ours queue there (moves this
        pass counted: live 10-08 one preemption wave put 8 HIGH screens behind 8 running on an 8-slot primary). Any
        run requeued ``slurm.repool_restarts_any`` times moves into a FREE primary slot only."""
        sl, pools = self.ctx.cfg.slurm, self.ctx.cfg.pools
        prim = next((p.name for p in pools if not p.preemptible), "")
        pre = {p.name for p in pools if p.preemptible}
        if not (getattr(self.adapter, "repool", None) and pre and prim):
            return []
        stuck = []
        for run in self.store.runs(status=[RunStatus.RUNNING]):
            if run.pool not in pre or not (run.id in self.pending or run.id in self.repool_due):
                continue
            hi = self.high(self.store.get_node(run.node_id), run.stage, run.seed)
            if hi or self.restarts.get(run.id, 0) >= sl.repool_restarts_any > 0:
                stuck.append((not hi, run.submitted_ts or 0.0, run.id, run, hi))      # HIGH first, oldest first
        first, out = pools[0].name, []
        backlog = sum((r.pool or first) == prim and r.id in self.pending for r in self.active())
        for *_, run, hi in sorted(stuck, key=lambda x: x[:3]):
            again = hi and self.restarts.get(run.id, 0) >= sl.repool_restarts > 0 and not self.reserved() \
                and not self._primary_capped()        # a capped primary would hold it at AssocGrpGRES (10-21)
            free_dest = next(iter(self._open(False, gate=False)), "")      # never a slot reserved for a TEST run
            if not (dest := free_dest or (prim if again and backlog < self._overflow_cap(prim) else "")):
                continue
            if (line := self._move(run, dest, hi, again)) is not None:
                backlog += not free_dest                 # it queues on the full primary
                out.append(line)
        return out

    def _note_pending(self) -> None:
        """The stale-pending clock (``K_PEND_SINCE``): when each run was first seen PENDING; a run no longer pending
        (started, finished, moved) is dropped. Persisted: a restart keeps the wait."""
        book = dict(self.store.get_state(K_PEND_SINCE) or {})
        now = self.clock()
        new = {rid: book.get(rid, now) for rid in self.pending}
        if new != book:
            self.store.set_state(K_PEND_SINCE, new)

    def stale_pending(self) -> list[str]:
        """A run PENDING longer than ``slurm.stale_pending_min`` (90; 0 = off) for a reason in
        ``slurm.stale_pending_reasons`` (a prefix: AssocGrpGRES, QOSMax..., a pool that cannot schedule it, not
        plain Priority) is MOVED to a pool with a free effective slot (:meth:`_move`: a free retry, same attempt) or,
        with none, its job is scancelled (the loop's own, by id: ``adapter.cancel``) and the row returns to the queue
        as an unspent INFRA void (attempt - 1: no retry is burnt) that ``Tick._retries`` re-submits once a slot is
        free. Its slot is released either way. TEST-stage runs and factorial wave points are left alone (they wait
        for primary on purpose). Extends the repool / preempt-backoff machinery, never a second path (live 10-21:
        screens and smokes sat at AssocGrpGRES for 30-120 min)."""
        sl = self.ctx.cfg.slurm
        stale_s = 60.0 * float(getattr(sl, "stale_pending_min", 0.0) or 0.0)
        reasons = tuple(getattr(sl, "stale_pending_reasons", ()) or ())
        reason_of = getattr(self.adapter, "pending_reason", None)
        if stale_s <= 0 or not reasons or reason_of is None or not self.pending:
            return []
        from loop.tick import TEST_STAGES                # lazy: loop.tick imports this module
        since, now, out = dict(self.store.get_state(K_PEND_SINCE) or {}), self.clock(), []
        can_move = bool(getattr(self.adapter, "repool", None))
        pre_names = {p.name for p in self.ctx.cfg.pools if p.preemptible}
        capped = self._primary_capped()
        for run in self.store.runs(status=[RunStatus.RUNNING]):
            t0 = since.get(run.id)
            if run.id not in self.pending or t0 is None or now - float(t0) < stale_s or self._wave_point(run):
                continue
            test = run.stage in TEST_STAGES
            # a TEST run waits for primary on purpose -- unless primary is cut by the operator's other jobs (it would
            # sit at AssocGrpGRES for good): then it MOVES to a preemptible pool (never cancelled, never re-queued)
            if test and not (capped and run.pool not in pre_names and can_move):
                continue
            handle = JobHandle(run.id, run.job_id, run.output_dir, run.submitted_ts or 0.0)
            try:
                why = str(reason_of(handle) or "")
            except Exception as exc:  # noqa: BLE001 - unknown is not a verdict
                self.store.log("stale_pending_probe_failed", run_id=run.id, error=repr(exc)[:300])
                continue
            if not why.startswith(reasons):
                continue                                 # plain Priority / Resources: it will start
            wait_min = (now - float(t0)) / 60.0
            hi = self.high(self.store.get_node(run.node_id), run.stage, run.seed)
            # preemptible pools first for a LOW run, primary first for a HIGH one; never the pool that cannot take it
            order = [True] if test else [True, False] if not hi else [False, True]
            moved = set(self.store.get_state(K_STALE_MOVED) or [])
            # a run stale-pending already moved once is never moved again (primary <-> idle ping-pong): it is
            # cancelled back into the queue instead, where the normal placement decides
            dest = next((d for pre in order for d in self._open(pre, gate=False) if d != run.pool), "") \
                if can_move and run.id not in moved else ""
            if test and not dest:
                continue                                 # no preemptible slot either: it keeps its place
            if dest and not self._cancel_pending(handle, release=False):     # repool needs the receipt
                continue                                 # it started (or SLURM would not say): leave it alone
            if dest and test:
                self._mark_spilled(dataclasses.replace(run, pool=dest))
            if dest and (line := self._move(run, dest, hi, False)) is not None:
                self.store.log("run_stale_pending", run_id=run.id, action="moved", frm=run.pool, to=dest,
                               reason=why, waited_min=round(wait_min))
                self._drop_pending(run.id)
                live = {r.id for r in self.active()}
                self.store.set_state(K_STALE_MOVED, sorted((moved & live) | {run.id}))
                out.append(f"stale pending: {run.id} waited {wait_min:.0f} min at {why} on {run.pool} -> {line}")
                continue
            if not self._cancel_pending(handle):
                continue                                 # it started (or SLURM would not say): leave it alone
            self._drop_pending(run.id)
            self.restarts.pop(run.id, None)
            # an unspent INFRA void: attempt - 1 (may be -1) so the tick's retry (attempt + 1) is the SAME attempt, and the
            # cancelled job's receipt is set aside (adapter.release_receipt) so that submit launches a NEW job: the
            # adapter returns the receipt's job for an attempt it already launched ("never twice"), which would re-attach
            # the row to the cancelled one
            self.store.update_run(run.id, status=RunStatus.VOID, void_type=VoidType.INFRA, attempt=run.attempt - 1)
            self.store.log("run_stale_pending", run_id=run.id, action="cancelled", frm=run.pool, reason=why,
                           waited_min=round(wait_min))
            out.append(f"stale pending: {run.id} waited {wait_min:.0f} min at {why} on {run.pool}, no pool could "
                       f"take it -- scancelled, back in the queue (slot released)")
        return out

    def _cancel_pending(self, handle: JobHandle, release: bool = True) -> bool:
        """scancel the loop's own job by id ONLY while SLURM still holds it PENDING (``scancel --state=PENDING``: a job that
        started in the window since the poll is left running -- the adapter's confirm raises when it is still live) and set
        its receipt aside. False = not cancelled (logged ``cancel_failed``); the next poll sees whatever SLURM holds."""
        try:
            try:
                self.adapter.cancel(handle, only_pending=True)
            except TypeError:                            # an adapter without the flag: a plain cancel
                self.adapter.cancel(handle)
        except Exception as exc:  # noqa: BLE001 - the next poll sees whatever SLURM holds
            self.store.log("cancel_failed", run_id=handle.run_id, error=repr(exc)[:500])
            return False
        if release and (rel := getattr(self.adapter, "release_receipt", None)) is not None:
            try:
                rel(handle)
            except Exception as exc:  # noqa: BLE001 - the retry then reuses the receipt (logged)
                self.store.log("release_receipt_failed", run_id=handle.run_id, error=repr(exc)[:300])
        return True

    def _drop_pending(self, rid: str) -> None:
        self.pending.discard(rid)
        self.repool_due.discard(rid)
        if rid in (book := dict(self.store.get_state(K_PEND_SINCE) or {})):
            book.pop(rid)
            self.store.set_state(K_PEND_SINCE, book)

    def _move(self, run: Run, dest: str, hi: bool, again: bool) -> str | None:
        """scancel ``run``'s job and launch the same row on ``dest`` (adapter.repool); None if that failed."""
        try:
            handle = self.adapter.repool(JobHandle(run.id, run.job_id, run.output_dir, run.submitted_ts or 0.0),
                                         dataclasses.replace(run, pool=dest))
        except Exception as exc:  # noqa: BLE001 - the next poll sees whatever SLURM holds
            self.store.log("repool_failed", run_id=run.id, error=repr(exc)[:500])
            return None
        self.repool_due.discard(run.id)
        if run.id in (book := dict(self.store.get_state(K_PEND_SINCE) or {})):    # a new job: the stale clock restarts
            book.pop(run.id)
            self.store.set_state(K_PEND_SINCE, book)
        self.store.update_run(run.id, pool=dest, job_id=handle.job_id, output_dir=handle.output_dir)
        self._nice(dataclasses.replace(run, pool=dest), handle)      # a new job: sbatch gave it nice 0
        self.store.log("run_repooled", run_id=run.id, frm=run.pool, to=dest, job_id=handle.job_id,
                       restarts=self.restarts.pop(run.id, 0))
        return (f"repool {run.id}: pending on {run.pool} -> {dest} as job {handle.job_id} ({'HIGH' if hi else 'LOW'}"
                + (", preempted again and again)" if again else ")"))

    def _overflow_cap(self, prim: str) -> int:
        """How many of our runs may queue PENDING on a full primary pool (moved there by :meth:`promote_stuck`):
        max(1, slots // 4) -- a short queue that starts within a fraction of a run length."""
        return max(1, next((p.slots for p in self.ctx.cfg.pools if p.name == prim), 0) // 4)

    def _primary_queue_short(self, prim: str, frm: str) -> bool:
        """Primary's own SLURM queue is shorter than the starved pool's, and short (< half its slots)."""
        st = self.pool_states or {}
        p_pend, f_pend = st.get(prim, (0, 0))[1], st.get(frm, (0, 0))[1]
        slots = next((p.slots for p in self.ctx.cfg.pools if p.name == prim), 0)
        return p_pend < f_pend and p_pend < max(1, slots // 2)

    # ------------------------------------------------------------------ the OOM ladder
    def ladder(self, node: Node | None) -> tuple[str, ...]:
        """The OOM mitigation ladder for ``node``'s commit: the adapter's rungs, then ``extra_rungs`` (() = none)."""
        fn = getattr(self.adapter, "oom_ladder", None)
        if fn is None or node is None:
            return ()
        try:
            own = tuple(fn(node.commit))
        except Exception as exc:  # noqa: BLE001 - no ladder is not a verdict
            self.store.log("oom_ladder_failed", node_id=node.id, error=repr(exc)[:300])
            return ()
        return own + tuple(r for r in self.extra_rungs if r not in own)

    def next_mitigation(self, node: Node, run: Run) -> str:
        return next_mitigation(self.ladder(node), run.mitigation, oom_note(node))

    def oom_retry(self, run: Run, node: Node, step: str) -> Run:
        """Relaunch the SAME row at the next OOM ladder step (no new node). The step is stored on the run and as
        ``oom=<step>`` in the node's notes, so its later runs start there."""
        node = self.store.update_node(node.id, notes=with_oom_note(node.notes, step))
        self.store.log("run_oom_retry", run_id=run.id, node_id=node.id, mitigation=step, attempt=run.attempt + 1)
        self.acts.append(f"oom {run.id}: relaunched with {step} (attempt {run.attempt + 1})")
        run = self.store.update_run(run.id, status=RunStatus.SUBMITTING, job_id="", void_type=None,
                                    attempt=run.attempt + 1, mitigation=step)
        return self._submit(run, node)

    # ------------------------------------------------------------------ submit / poll / cancel
    def launch(self, node: Node, stage: Stage, seed: int) -> Run | None:
        """Write the row ``submitting``, then submit; None if the row already existed (or no pool
        takes it now: a design point / LOW run waiting for a slot)."""
        run = self.prepare_launch(node, stage, seed)
        return self._submit(run, node) if run is not None else None

    def prepare_launch(self, node: Node, stage: Stage, seed: int) -> Run | None:
        """:meth:`launch` up to the submit: the pool, the ``submitting`` row (slots are counted from here on), the
        TEST-pool and idle-fill notes. None: the row already existed or no pool takes it now."""
        if not (pool := self.pool(node, stage, seed)):
            return None
        run = Run(id=Run.make_id(node.id, stage, seed), task_id=self.ctx.task_id, node_id=node.id,
                  seed=int(seed), stage=stage, status=RunStatus.SUBMITTING,
                  submitted_ts=self.clock(), pool=pool, mitigation=stronger(self.ladder(node), oom_note(node)))
        try:
            self.store.put_run(run)
        except sqlite3.IntegrityError:
            self.store.log("double_launch_refused", run_id=run.id)
            return None
        self._note_test_pool(run)
        if (node.id, seed) in self.fill_only:        # idle fill: remembered (restart cap, the meta's fill bucket)
            self.store.log("run_fill", run_id=run.id, node_id=node.id, pool=pool)
            self.fill_runs().add(run.id)
        return run

    def submit_prepared(self, pairs: Sequence[tuple[Run, Node]],
                        stop: Callable[[], bool] | None = None, min_n: int = 0) -> list[Run | BaseException]:
        """Submit ``pairs`` (rows from :meth:`prepare_launch`): the seed guard and the store writes on this thread, the
        adapter submits (node worktree + sbatch) on up to ``slurm.submit_workers`` threads when the adapter allows
        it (``concurrent_submit``; live 10-13: a tick's serial submits took minutes), in chunks of that many. Before
        each chunk ``stop()`` (the tick's ``over_budget``) is asked, and after a chunk in which an sbatch HUNG
        (:func:`hung`: the runner's timeout) no further chunk starts (c4j review S2: a hanging slurmctld held the
        tick for workers x 300 s per chunk). The rows not submitted stay ``submitting`` (their slots held) and get
        :class:`SubmitDeferred`; the hung row stays ``submitting`` too (its sbatch may have gone through): each tick's
        :meth:`resolve_submitting` adopts the job by its receipt / job name or submits it then. Per pair, in order:
        the Run (RUNNING, or VOID when refused / failed), the ``SubmissionRejected`` it raised (its row is void), or a
        ``SubmitDeferred``. ``stop()`` is only asked once ``min_n`` rows were submitted (the tick's min_submits floor)."""
        workers = int(getattr(self.ctx.cfg.slurm, "submit_workers", 1) or 1)
        if workers <= 1 or len(pairs) < 2 or not getattr(self.adapter, "concurrent_submit", False):
            workers = 1
        res: list[Any] = [None] * len(pairs)
        todo = list(range(len(pairs)))
        halt = ""

        def one(i: int) -> JobHandle | BaseException:
            run, node = pairs[i]
            try:
                return self.adapter.submit(run=run, commit=node.commit)
            except Exception as exc:  # noqa: BLE001 - handled on the tick thread
                return exc
        ex = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
        try:
            while todo:
                if not halt and stop is not None and sum(r is not None for r in res) >= min_n:
                    try:
                        halt = "tick budget spent" if stop() else ""
                    except Exception:  # noqa: BLE001 - a broken budget probe never stops submits
                        halt = ""
                if halt:
                    for i in todo:
                        res[i] = SubmitDeferred(f"{pairs[i][0].id}: not submitted ({halt}); resolved next tick")
                    self.store.log("submit_deferred", run_ids=[pairs[i][0].id for i in todo], reason=halt)
                    break
                chunk, todo = [], list(todo)
                while todo and len(chunk) < workers:
                    i = todo.pop(0)
                    if (refused := self._guard_refused(pairs[i][0])) is not None:
                        res[i] = refused
                    else:
                        chunk.append(i)
                got = list(ex.map(one, chunk)) if ex is not None else [one(i) for i in chunk]
                for i, h in zip(chunk, got):
                    if isinstance(h, BaseException) and not isinstance(h, SubmissionRejected) and hung(h):
                        self.store.log("submit_hung", run_id=pairs[i][0].id, error=repr(h)[:500])
                        res[i] = SubmitDeferred(f"{pairs[i][0].id}: sbatch hung ({h}); resolved next tick")
                        halt = "an sbatch hung"
                        continue
                    try:
                        res[i] = self._submitted(pairs[i][0], h)
                    except SubmissionRejected as exc:
                        res[i] = exc
        finally:
            if ex is not None:
                ex.shutdown(wait=True)
        return res

    def fill_runs(self) -> set[str]:
        """Run ids launched as idle fill (the ``run_fill`` events, read once per process)."""
        if self._fill_runs is None:
            try:
                self._fill_runs = {str(e.get("run_id")) for e in self.store.events("run_fill")}
            except Exception:  # noqa: BLE001 - a store without events: only this process's fills are known
                self._fill_runs = set()
        return self._fill_runs

    def idle_churn(self) -> list[str]:
        """Idle-pool churn (``acquire.idle_max_restarts``, 0 = off): an idle-fill run SLURM preempted that many times
        on a preemptible pool moves to a FREE primary slot (same row, same attempt: keeps its latest.pt) while no HIGH
        run waits (``high_demand``), else it is dropped (void / CANCELLED: the loop's own decision; ``acquire.idle_fill``
        never refills that (node, seed)). Live 10-12: 198 preemptions of idle-pool runs lost 34 GPU-h in 4 h.
        A RESUMABLE run (its latest.pt armed: a requeue resumes it, losing only the steps since the last save) at least
        ``acquire.idle_keep_frac`` (.5) through training is never dropped: it stays where it is (live 10-13: 16 of 18
        dropped fill runs were ~98% done, armed and resumed -- each drop threw away ~1 GPU-h of finished training)."""
        aq = getattr(self.ctx.cfg, "acquire", None)
        cap = int(getattr(aq, "idle_max_restarts", 2))
        keep_frac = float(getattr(aq, "idle_keep_frac", 0.5))
        pools = self.ctx.cfg.pools
        pre = {p.name for p in pools if p.preemptible}
        if cap <= 0 or not pre or not self.restarts:
            return []
        fills, out = self.fill_runs(), []
        for run in self.store.runs(status=[RunStatus.RUNNING]):
            if run.id not in fills or run.pool not in pre or self.restarts.get(run.id, 0) < cap:
                continue
            if self._wave_point(run):     # a relax wave's point: its wave waits on it -- a cancel leaves a zombie
                continue
            # a free primary slot under the gated limits, and only while no HIGH run waits for one (a test's runs
            # must never lose their slot to a fill run); else the run is dropped
            dest = next(iter(self._open(False)), "") if getattr(self.adapter, "repool", None) \
                and not self.high_demand else ""
            if dest and (line := self._move(run, dest, False, True)) is not None:
                out.append(f"idle churn: {line}")
                continue
            if self._resumable_progress(run, keep_frac):    # resumes from latest.pt: a requeue loses ~a save
                continue
            try:
                self.adapter.cancel(JobHandle(run.id, run.job_id, run.output_dir, run.submitted_ts or 0.0))
            except Exception as exc:  # noqa: BLE001 - the next poll sees whatever SLURM holds
                self.store.log("cancel_failed", run_id=run.id, error=repr(exc)[:500])
                continue
            n = self.restarts.pop(run.id, 0)
            self.pending.discard(run.id)
            self.repool_due.discard(run.id)
            self.store.update_run(run.id, status=RunStatus.VOID, void_type=VoidType.CANCELLED)
            self.store.log("run_cancelled", run_id=run.id, node_id=run.node_id,
                           reason=f"idle fill preempted {n}x (cap {cap}), no free primary slot"
                           + (" (HIGH runs waiting)" if self.high_demand else ""))
            out.append(f"idle churn: dropped {run.id} (preempted {n}x on {run.pool}, no free primary slot)")
        return out + self._preempt_backoff(fills, pre, keep_frac)

    def _preempt_backoff(self, fills: set[str], pre: set[str], keep_frac: float) -> list[str]:
        """v3-c7a A8, the idle churn for every OTHER run (``acquire.preempt_max_restarts``, 3; 0 = off): a non-fill
        run preempted that often on a preemptible pool moves to a FREE primary slot while no HIGH run waits; with
        primary full it is dropped (same ``idle fill preempted`` accounting) only when the preemptions are pure loss:
        it does not resume (no armed latest.pt: every requeue restarts it), or after twice the cap it is still below
        ``acquire.idle_keep_frac`` of training. Never a TEST-stage run, a :data:`BACKOFF_EXEMPT` stage, a wave
        point, a resumable run not yet at its first save, or a run whose progress the adapter cannot read. v3 review
        O3: a drop is an INFRA void re-submitted at once (attempt + 1) on a primary pool, never a final CANCELLED (live 10-14: 189 preemptions on 66 idle-pool screen runs, all armed, all finished)."""
        aq = getattr(self.ctx.cfg, "acquire", None)
        cap = int(getattr(aq, "preempt_max_restarts", 3))
        rs = getattr(self.adapter, "resume_state", None)
        if cap <= 0 or not pre or not self.restarts:
            return []
        from loop.tick import TEST_STAGES                # lazy: loop.tick imports this module
        keep = {*TEST_STAGES, *BACKOFF_EXEMPT}
        out = []
        # 11-10 (nanobrain, primary 17/8): the drop below re-queues on primary however full it is -- 6 dropped screens
        # sat PENDING at AssocGrpGRES behind 8 running. Same bound as promote_stuck: a drop only while fewer than
        # _overflow_cap of ours queue on primary and primary is not association-capped; else it stays on its pool
        prim0 = next((p.name for p in self.ctx.cfg.pools if not p.preemptible), "")
        first = self.ctx.cfg.pools[0].name if self.ctx.cfg.pools else ""
        backlog = sum((r.pool or first) == prim0 and r.id in self.pending for r in self.active())
        for run in self.store.runs(status=[RunStatus.RUNNING]):
            n = self.restarts.get(run.id, 0)
            if run.id in fills or run.pool not in pre or n < cap or run.stage in keep or self._wave_point(run):
                continue
            dest = next(iter(self._open(False)), "") if getattr(self.adapter, "repool", None) \
                and not self.high_demand else ""
            if dest and (line := self._move(run, dest, False, True)) is not None:
                out.append(f"preempt backoff: {line}")
                continue
            if rs is None or self.reserved() > 0:
                # progress unknown: never dropped. 11-10 (nanobrain test-a node): owed TEST runs hold primary -- the
                # drop below re-queues the run on primary however full it is, and SLURM then starts it in the next
                # freed primary slot ahead of the test (4 screens re-queued there in the block's first hour while
                # its pairs waited); it stays on its preemptible pool until the test's runs are placed
                continue
            try:
                st = rs(JobHandle(run.id, run.job_id, run.output_dir, run.submitted_ts or 0.0)) or {}
            except Exception:  # noqa: BLE001 - progress unknown: never dropped
                continue
            armed, frac = bool(st.get("armed")), float(st.get("fraction") or 0.0)
            if armed and (n < 2 * cap or frac >= keep_frac):
                continue                                 # it resumes and moves forward: let it finish
            if not armed and st.get("resumable"):
                continue                                 # v3 review O3: resumable, its first save not yet reached
            if not prim0 or backlog >= self._overflow_cap(prim0) or self._primary_capped():
                continue                                 # primary's queue is full: it stays (a SLURM requeue in place is
                #   the same restart a cancel + idle re-submit would be); the _move above takes the next FREE primary slot
            backlog += 1
            try:
                self.adapter.cancel(JobHandle(run.id, run.job_id, run.output_dir, run.submitted_ts or 0.0))
            except Exception as exc:  # noqa: BLE001 - the next poll sees whatever SLURM holds
                self.store.log("cancel_failed", run_id=run.id, error=repr(exc)[:500])
                continue
            self.restarts.pop(run.id, 0)
            self.pending.discard(run.id)
            self.repool_due.discard(run.id)
            why = "not resumable" if not armed else f"{frac:.0%} through after {n} restarts"
            was = run.pool
            # v3 review O3: no verdict on the idea -- an INFRA void re-submitted at once as the next attempt QUEUED
            # on a primary pool (never preempted again), not a final CANCELLED that loses the seed
            run = self.store.update_run(run.id, status=RunStatus.VOID, void_type=VoidType.INFRA)
            self.store.log("run_cancelled", run_id=run.id, node_id=run.node_id,
                           reason=f"idle fill preempted {n}x (non-fill {getattr(run.stage, 'value', run.stage)}, "
                                  f"cap {cap}, {why}), no free primary slot -- re-queued on primary"
                           + (" (HIGH runs waiting)" if self.high_demand else ""))
            prim = next((p.name for p in self.ctx.cfg.pools if not p.preemptible), "")
            node = self.store.get_node(run.node_id)
            if prim and node is not None:
                try:
                    run = self.retry(run, node, pool=prim)
                except Exception as exc:  # noqa: BLE001 - the INFRA void stays for the tick's retries
                    self.store.log("preempt_backoff_retry_failed", run_id=run.id, error=repr(exc)[:500])
            out.append(f"preempt backoff: re-queued {run.id} on {run.pool} (preempted {n}x on {was}, {why}, "
                       f"attempt {run.attempt})")
        return out

    def _wave_point(self, run: Run) -> bool:
        """``run`` is a factorial wave's point (``core.factorial.is_point``): the idle churn never cancels it (v3-c5
        A2: a churn-cancelled relax-wave point left its wave open with a void point -- a zombie holding the carrier's
        one-open-wave slot). It stays preemptible-only and resumes from its latest.pt like any requeued run."""
        from loop.core.factorial import is_point         # lazy: keeps the runner's import graph flat
        try:
            node = self.store.get_node(run.node_id)
        except Exception:  # noqa: BLE001 - unknown node: not a point
            return False
        return node is not None and is_point(getattr(node, "notes", "") or "")

    def _resumable_progress(self, run: Run, keep_frac: float) -> bool:
        """``run`` resumes from an armed latest.pt and is >= ``keep_frac`` through training (``adapter.resume_state``;
        False without one): idle churn keeps it (logged ``idle_churn_kept`` once per restart count)."""
        rs = getattr(self.adapter, "resume_state", None)
        if rs is None or keep_frac > 1.0:
            return False
        try:
            st = rs(JobHandle(run.id, run.job_id, run.output_dir, run.submitted_ts or 0.0))
        except Exception:  # noqa: BLE001 - unknown progress: the old rule
            return False
        if not (st.get("armed") and float(st.get("fraction") or 0.0) >= keep_frac):
            return False
        n = self.restarts.get(run.id, 0)
        if self._kept.get(run.id) != n:
            self._kept[run.id] = n
            self.store.log("idle_churn_kept", run_id=run.id, node_id=run.node_id, restarts=n,
                           fraction=round(float(st.get("fraction") or 0.0), 3), resumed=int(st.get("resumed") or 0))
        return True

    def retry(self, run: Run, node: Node, pool: str = "") -> Run:
        """Re-submit the SAME row as the next attempt; ``submitted_ts`` stays the first. ``pool``: where it must
        go (the preemption backoff's primary queue), else the usual choice."""
        run = self.store.update_run(run.id, status=RunStatus.SUBMITTING, job_id="",
                                    void_type=None, attempt=run.attempt + 1,
                                    pool=pool or self.pool(node, run.stage, run.seed) or self._retry_home(run),
                                    mitigation=stronger(self.ladder(node), run.mitigation, oom_note(node)))
        self.store.log("run_retry", run_id=run.id, attempt=run.attempt)
        self._note_test_pool(run)
        return self._submit(run, node)

    def _retry_home(self, run: Run) -> str:
        """Where a retry no pool takes now queues: its own pool, but a TEST-stage run's primary, never preemptible."""
        if self.test_primary_only(run.stage) and run.pool not in {p.name for p in self.ctx.cfg.pools
                                                                  if not p.preemptible}:
            return next(p.name for p in self.ctx.cfg.pools if not p.preemptible)
        if self.reserved() and not self.test_primary_only(run.stage) \
                and any(p.name == run.pool and not p.preemptible for p in self.ctx.cfg.pools):
            return self._least_pending_preemptible() or run.pool   # v3-c6a: no queueing on primary ahead of a test
        return run.pool

    def _guard_refused(self, run: Run) -> Run | None:
        """The seed-pool check (before EVERY submit, retries too): None = may go, else the row voided CANCELLED."""
        if self.guard is None:
            return None
        try:
            self.guard(run)
        except Exception as exc:  # noqa: BLE001 - a refused run is the loop's own cancel, never retried
            self.store.log("submit_refused", run_id=run.id, node_id=run.node_id, seed=run.seed,
                           error=repr(exc)[:500])
            print(f"{utc_now(self.clock())} ALERT: submit of {run.id} refused: {exc}", file=sys.stderr, flush=True)
            return self.store.update_run(run.id, status=RunStatus.VOID, void_type=VoidType.CANCELLED)
        return None

    def _submit(self, run: Run, node: Node) -> Run:
        if (refused := self._guard_refused(run)) is not None:
            return refused
        try:
            handle: JobHandle | BaseException = self.adapter.submit(run=run, commit=node.commit)
        except Exception as exc:  # noqa: BLE001 - classified by _submitted
            handle = exc
        return self._submitted(run, handle)

    def _submitted(self, run: Run, handle: JobHandle | BaseException) -> Run:
        """The submit's outcome on the store: attached, or an infra void (``SubmissionRejected``: re-raised)."""
        if isinstance(handle, SubmissionRejected):
            self.store.update_run(run.id, status=RunStatus.VOID, void_type=VoidType.INFRA,
                                  attempt=max(run.attempt, int(self.infra_retry) + 1))
            raise handle                             # deterministic: stop, do not retry
        if isinstance(handle, BaseException):        # a failed submit is an infra void
            self.store.log("submit_failed", run_id=run.id, error=repr(handle)[:500])
            return self.store.update_run(run.id, status=RunStatus.VOID, void_type=VoidType.INFRA)
        return self._attach(run, handle)

    def _attach(self, run: Run, handle: JobHandle) -> Run:
        self.store.log("run_submitted", run_id=run.id, job_id=handle.job_id)
        self._nice(run, handle)
        return self.store.update_run(run.id, status=RunStatus.RUNNING, job_id=handle.job_id,
                                     output_dir=handle.output_dir)

    def _nice(self, run: Run, handle: JobHandle) -> None:
        """A LOW run on a preemptible pool gets ``slurm.low_nice``; every other run ``slurm.run_nice`` (both 0 =
        off by default: nice ranks a job below every other user's too)."""
        sl, pre = self.ctx.cfg.slurm, {p.name for p in self.ctx.cfg.pools if p.preemptible}
        low = sl.low_nice > 0 and run.pool in pre \
            and not self.high(self.store.get_node(run.node_id), run.stage, run.seed)
        nice = sl.low_nice if low else sl.run_nice
        if nice > 0 and hasattr(self.adapter, "nice"):
            try:
                self.adapter.nice(handle, nice)
            except Exception as exc:  # noqa: BLE001 - priority is a courtesy, never a verdict
                self.store.log("nice_failed", run_id=run.id, error=repr(exc)[:300])

    def resolve_submitting(self) -> list[str]:
        """``submitting`` rows left by a crash: adopt the job the adapter's receipts know (``find``), else
        (idempotently) submit."""
        out = []
        for run in self.store.runs(status=[RunStatus.SUBMITTING]):
            try:
                handle = self.adapter.find(run)
            except Exception as exc:  # noqa: BLE001
                self.store.log("find_failed", run_id=run.id, error=repr(exc)[:500])
                continue
            if handle is not None:
                self._attach(run, handle)
                out.append(f"resume: found job {handle.job_id} for {run.id}")
            elif (node := self.store.get_node(run.node_id)) is not None:
                if (refused := self._guard_refused(run)) is not None:
                    out.append(f"resume: {run.id} refused by the seed guard -> {refused.status.value}")
                    continue
                try:
                    h: JobHandle | BaseException = self.adapter.submit(run=run, commit=node.commit)
                except Exception as exc:  # noqa: BLE001 - classified below
                    h = exc
                if isinstance(h, BaseException) and not isinstance(h, SubmissionRejected) and hung(h):
                    # sbatch hangs (slurmctld): the row stays submitting, the rest wait for the next tick
                    self.store.log("submit_hung", run_id=run.id, error=repr(h)[:500])
                    out.append(f"resume: sbatch hung on {run.id}; the remaining submitting rows wait a tick")
                    break
                try:
                    self._submitted(run, h)
                except SubmissionRejected:
                    pass
                out.append(f"resume: resubmitted {run.id}")
        return out

    def _cancel_keeps(self, node_id: str) -> Callable[[Run], bool]:
        """Runs of ``node_id`` a cancel leaves running (v3: none)."""
        return lambda _run: False

    def cancel(self, node_id: str, reason: str = "node closed", only_pending: bool = False) -> None:
        """Cancel a node's running jobs (``only_pending``: only those SLURM still holds PENDING); each is recorded
        void/CANCELLED at once (the loop's own decision: never an infra void, never retried)."""
        keep = self._cancel_keeps(node_id)
        for run in self.store.runs(node_id=node_id, status=[RunStatus.RUNNING]):
            if only_pending and run.id not in self.pending or keep(run):
                continue
            self.cancel_run(run, reason)

    def cancel_run(self, run: Run, reason: str) -> bool:
        """Cancel ONE run's job by its job id (adapter.cancel: scancel <job>) and record it void/CANCELLED; False
        when the cancel failed or was not confirmed (the adapter checks scancel's rc and squeue): the row stays
        RUNNING (never VOID with a live job behind it) and is retried every tick (:meth:`retry_cancels`)."""
        retry = dict(self.store.get_state(K_CANCEL_RETRY) or {})
        try:
            self.adapter.cancel(JobHandle(run.id, run.job_id, run.output_dir, run.submitted_ts or 0.0))
        except Exception as exc:  # noqa: BLE001 - still running: retried next tick
            self.store.log("cancel_failed", run_id=run.id, error=repr(exc)[:500])
            if run.id not in retry:
                self.store.set_state(K_CANCEL_RETRY, {**retry, run.id: reason[:300]})
            return False
        if retry.pop(run.id, None) is not None:
            self.store.set_state(K_CANCEL_RETRY, retry)
        self.pending.discard(run.id)
        self.restarts.pop(run.id, None)
        self.store.update_run(run.id, status=RunStatus.VOID, void_type=VoidType.CANCELLED)
        self.store.log("run_cancelled", run_id=run.id, node_id=run.node_id, reason=reason[:300])
        return True

    def retry_cancels(self) -> list[str]:
        """Retry the cancels :meth:`cancel_run` could not confirm; a run no longer RUNNING is dropped."""
        out = []
        for rid, reason in dict(self.store.get_state(K_CANCEL_RETRY) or {}).items():
            run = self.store.get_run(rid)
            if run is None or run.status is not RunStatus.RUNNING:
                left = dict(self.store.get_state(K_CANCEL_RETRY) or {})
                left.pop(rid, None)
                self.store.set_state(K_CANCEL_RETRY, left)
            elif self.cancel_run(run, reason):
                out.append(f"cancel retried: {rid}")
        return out

    def _oom_ok(self, run: Run, node: Node) -> bool:
        """An OOM of ``run`` climbs the ladder (v3: any run; the old driver: a live node's or a control's)."""
        return True

    def poll(self) -> list[Run]:
        """Poll every running run; return the ones that reached a terminal state."""
        finished = []
        self._cap = None                                     # the capacity probe is read once per tick
        self._free = None                                    # ... and the free-GPU probe
        for run in self.store.runs(status=[RunStatus.RUNNING]):
            handle = JobHandle(run.id, run.job_id, run.output_dir, run.submitted_ts or 0.0)
            try:
                st = self.adapter.poll(handle)
            except Exception as exc:  # noqa: BLE001 - a poll error is not a verdict
                self.store.log("poll_failed", run_id=run.id, error=repr(exc)[:500])
                continue
            resumes = st.resumed > run.resumed       # this requeue resumes from latest.pt: its progress is kept
            if resumes:                              # non-exact from now on (loop.world.noise.resume_var)
                run = self.store.update_run(run.id, resumed=int(st.resumed))
            if st.state in ACTIVE_RUN:
                (self.pending.add if st.detail == "PENDING" else self.pending.discard)(run.id)
                if st.restarts:
                    self.restarts[run.id] = st.restarts
                for _ in range(st.requeued):         # SLURM requeued it under the same job id: preempted
                    self.store.log("run_preempted", run_id=run.id, pool=run.pool, gpu_h=float(st.gpu_h) / st.requeued,
                                   detail=f"slurm requeue (restarts {st.restarts})", seeded=True, resumed=resumes)
                if st.requeued and st.restarts >= self.ctx.cfg.slurm.repool_restarts > 0:
                    self.repool_due.add(run.id)
                if "free retry" in st.detail:       # SLURM took the machine: lost time, a requeue
                    self.store.log("run_preempted", run_id=run.id, pool=run.pool, gpu_h=float(st.gpu_h),
                                   detail=st.detail[:200])
                continue
            self.pending.discard(run.id)
            self.restarts.pop(run.id, None)
            self.repool_due.discard(run.id)
            void_type = (st.void_type or VoidType.INFRA) if st.state is RunStatus.VOID else None
            if st.state is not RunStatus.DONE:
                try:                     # a failed job may still hold its GPU
                    self.adapter.cancel(handle)
                except Exception as exc:  # noqa: BLE001
                    self.store.log("cancel_failed", run_id=run.id, error=repr(exc)[:500])
            if st.state is not RunStatus.DONE and void_type in (None, VoidType.INFRA) and is_oom(st.log_tail) \
                    and (node := self.store.get_node(run.node_id)) is not None and self._oom_ok(run, node) \
                    and (step := self.next_mitigation(node, run)):     # OOM: mitigate before any verdict
                self.store.update_run(run.id, gpu_h=float(st.gpu_h), log_tail=st.log_tail[-4000:])
                try:
                    self.oom_retry(self.store.get_run(run.id), node, step)
                except SubmissionRejected:
                    pass                                 # the row is void (deterministic); next poll skips it
                continue
            finished.append(self.store.update_run(
                run.id, status=st.state, void_type=void_type, scores=dict(st.scores),
                aggregate=st.aggregate, gpu_h=float(st.gpu_h), log_tail=st.log_tail[-4000:]))
            self.store.log("run_finished", run_id=run.id, status=st.state.value,
                           void_type=void_type.value if void_type else None, detail=st.detail[:500])
            if st.state is RunStatus.DONE and (post := getattr(self.adapter, "postprocess", None)) is not None:
                try:                                 # loop.diagnostics: the run's checkpoint features (detached, nice 19)
                    post(finished[-1])
                except Exception as exc:  # noqa: BLE001 - diagnostics never stop the loop
                    self.store.log("diagnostics_error", run_id=run.id, error=repr(exc)[:500])
            if void_type is VoidType.TOO_SLOW and run.node_id == ROOT_NODE_ID:   # baseline ~1 h
                self.store.log("ALERT_root_timeout", run_id=run.id, detail=st.detail[:500])
                print(f"{utc_now()} ALERT: root run {run.id} hit the time limit ({st.detail}): infra, not the "
                      f"candidate -- check the cluster", file=sys.stderr, flush=True)
        if (held := sorted(self.pending)) != (self.store.get_state(K_PENDING) or []):
            self.store.set_state(K_PENDING, held)            # `status` shows pending vs running
        self._note_pending()
        states: dict[str, list[int]] = {p.name: [0, 0] for p in self.ctx.cfg.pools}
        for r in self.store.runs(status=[RunStatus.RUNNING]):
            states.setdefault(r.pool or self.ctx.cfg.pools[0].name, [0, 0])[r.id in self.pending] += 1
        self.pool_states = {k: (v[0], v[1]) for k, v in states.items()}
        try:
            self.smoke_pending = dict(getattr(self.adapter, "smokes_pending", None) and self.adapter.smokes_pending() or {})
        except Exception:  # noqa: BLE001 - the hold is a courtesy; a squeue failure never stops a poll
            self.smoke_pending = {}
        book = dict(self.store.get_state(K_STARVED) or {})
        for p in self.ctx.cfg.pools:
            run_n, pend_n = self.pool_states.get(p.name, (0, 0))
            if p.preemptible and run_n == 0 and pend_n > 0:
                book.setdefault(p.name, self.clock())
            elif p.preemptible and run_n > 0:
                book.pop(p.name, None)
        if book != (self.store.get_state(K_STARVED) or {}):
            self.store.set_state(K_STARVED, book)
            self.store.log("pool_starved", pools=book)
        return finished


class Courtesy:
    """Cluster-courtesy backoff for the preemptible pools, shared by the driver and calibrate (every
    preemptible submission goes through ``RunBook.pool`` / ``smoke_pool``, which honour
    ``book.courtesy_cap``; ``step`` sets it)."""

    def __init__(self, ctx: LoopContext, store: Any, adapter: Any, book: "RunBook",
                 clock: Callable[[], float] = time.time) -> None:
        self.ctx, self.store, self.adapter, self.book, self.clock = ctx, store, adapter, book, clock
        self.next_probe, self.jobs = 0.0, []

    def step(self, acts: list[str]) -> None:
        """While a protected job (the operator's shared qwen server) is PENDING, the preemptible pools
        are capped: no NEW preemptible job until it has started for good. The cap is monotone while the
        backoff is on: it starts at max(4, held - 4), follows held preemptible down (a finished job is
        never replaced) and never rises; after 60 min still pending it steps down 4 every 30 min to the
        floor 4. The backoff lifts only once every protected job stayed out of PENDING for
        COURTESY_RELEASE_S (a start that is preempted minutes later is not a start). Running jobs are
        never cancelled for it. squeue at most every COURTESY_EVERY_S; an error keeps the state. State
        lives in the store (K_COURTESY), so a driver restart resumes the same backoff and cap."""
        names, probe = tuple(self.ctx.cfg.slurm.protect_jobs), getattr(self.adapter, "pending_jobs", None)
        now = self.clock()
        if names and probe is not None and now >= self.next_probe:
            self.next_probe = now + COURTESY_EVERY_S
            try:
                self.jobs = list(probe(names))
            except Exception as exc:  # noqa: BLE001 - unknown is not a verdict
                self.store.log("courtesy_probe_failed", error=repr(exc)[:300])
        if not names or probe is None:
            self.jobs = []
        pre = {p.name for p in self.ctx.cfg.pools if p.preemptible}
        held = sum(n for k, n in self.book.used().items() if k in pre)
        was = self.store.get_state(K_COURTESY) or {}
        st = dict(was) if was.get("on") and names and probe is not None and pre else {}
        jobs = list(self.jobs)
        if jobs and pre and not st:                                  # backoff starts
            cap = max(COURTESY_FLOOR, held - COURTESY_MARGIN)
            st = {"on": True, "jobs": jobs, "cap": cap, "cap0": cap, "held0": held, "since": now,
                  "next_step": now + COURTESY_STEP_AFTER_S, "clear_since": None}
            self.store.log("courtesy_backoff", on=True, jobs=jobs, cap=cap, held=held)
            acts.append(f"courtesy_backoff on: {','.join(jobs)} PENDING -> preemptible cap {cap} "
                        f"(held {held}); no new preemptible jobs")
        elif st and jobs:                                            # still (or again) pending
            if st.get("clear_since") is not None:
                self.store.log("courtesy_backoff_held", jobs=jobs, cap=st["cap"],
                               out_of_pending_s=now - float(st["clear_since"]))
                acts.append(f"courtesy_backoff stays on: {','.join(jobs)} PENDING again after "
                            f"{(now - float(st['clear_since'])) / 60:.0f} min out of it (cap {st['cap']})")
            st.update(jobs=jobs, clear_since=None)
        elif st:                                                     # not pending: lift only after a grace
            if st.get("clear_since") is None:
                st["clear_since"] = now
                self.store.log("courtesy_backoff_grace", jobs=st.get("jobs", []), cap=st["cap"])
                acts.append(f"courtesy_backoff grace: {','.join(st.get('jobs', []))} left PENDING; cap "
                            f"{st['cap']} held until it stays out of PENDING {COURTESY_RELEASE_S / 60:.0f} min")
            elif now - float(st["clear_since"]) >= COURTESY_RELEASE_S:
                self.store.log("courtesy_backoff", on=False, jobs=st.get("jobs", []), cap=None, held=held,
                               minutes_on=(now - float(st.get("since", now))) / 60)
                acts.append("courtesy_backoff off")
                st = {}
        if st:                                                       # monotone: never above the last cap
            cap = min(int(st["cap"]), max(COURTESY_FLOOR, held))
            if st.get("clear_since") is None and now >= float(st.get("next_step", now + 1)):
                st["next_step"] = now + COURTESY_STEP_EVERY_S
                if cap > COURTESY_FLOOR:
                    new = max(COURTESY_FLOOR, cap - COURTESY_STEP)
                    self.store.log("courtesy_step", jobs=st["jobs"], cap_from=cap, cap_to=new, held=held,
                                   pending_min=(now - float(st["since"])) / 60)
                    acts.append(f"courtesy_backoff step: {','.join(st['jobs'])} still PENDING after "
                                f"{(now - float(st['since'])) / 60:.0f} min -> preemptible cap {cap} -> {new} "
                                f"(held {held})")
                    cap = new
            st["cap"] = cap
        self.book.courtesy_cap = st["cap"] if st else None
        cur = st or {"on": False, "jobs": [], "cap": None}
        if cur != {k: v for k, v in was.items() if k != "ts"}:
            self.store.set_state(K_COURTESY, {**cur, "ts": now})


# ---------------------------------------------------------------------------------------------- the LLM gate
def llm_paused(store: Any) -> bool:
    return bool((store.get_state(K_LLM_PAUSE) or {}).get("active"))


def llm_probe_ok(store: Any) -> None:
    if (p := store.get_state(K_LLM_PAUSE) or {}).get("probing"):
        store.set_state(K_LLM_PAUSE, {**p, "probing": False})
        store.log("llm_probe_ok", kind=p.get("kind"))


def llm_probing(store: Any) -> bool:
    """After a pause: one research session probes the backend; every other LLM source waits until a Claude call
    works again (loop.agents.llm.last_success, or a harvested result that is no outage)."""
    p = store.get_state(K_LLM_PAUSE) or {}
    if not p.get("probing"):
        return False
    if LA.last_success() > float(p.get("ok_mark", 0.0)):     # a Claude call worked since the resume
        llm_probe_ok(store)
        return False
    return True


def llm_gate(store: Any, now: float, acts: list[str]) -> None:
    """An account refusal (usage limit / logged out: :mod:`loop.agents.llm` OUTAGE_KINDS) or a tripped SDK
    circuit breaker (kind 'sdk': :func:`loop.agents.llm.note_failure`) pauses every NEW LLM job until its reset
    (+1 min); with no known reset, for 15 min, doubling to <= 1 h while each resume (the probe: ONE research
    session, :func:`llm_probing`) is refused again. In-flight jobs finish; their outage results are never counted."""
    out, p = LA.take_outage(), store.get_state(K_LLM_PAUSE) or {}
    if out is not None and not p.get("active"):
        again = bool(p) and now - float(p.get("until", -1e18)) < LLM_PROBE_MAX_S   # the probe was refused
        probes = int(p.get("probes", 0)) + 1 if again else 0
        r = out.get("resets_at")
        until = float(r) + 60.0 if r and float(r) > now else now + min(LLM_PROBE_S * 2 ** probes, LLM_PROBE_MAX_S)
        p = {"active": True, "kind": out["kind"], "since": p["since"] if again else now, "until": until,
             "resets_at": r, "probes": probes}
        store.set_state(K_LLM_PAUSE, p)
        store.log("llm_paused", kind=p["kind"], until=until, resets_at=r, probes=probes)
        acts.append(f"llm_paused {p['kind']} until {utc_now(until)}"
                    + ("" if r else f" (reset unknown: probe {probes + 1})") + ": no new LLM job")
    elif p.get("active") and now >= float(p["until"]):
        store.set_state(K_LLM_PAUSE, {**p, "active": False, "probing": True, "resumed": now,
                                      "ok_mark": LA.last_success()})
        store.log("llm_resumed", kind=p["kind"], paused_s=round(now - float(p["since"])), probes=p["probes"])
        acts.append(f"llm_resumed after a {p['kind']} pause of {(now - float(p['since'])) / 3600:.1f} h")


def cancel_smokes(adapter: Any, store: Any, request_id: str = "") -> list[str]:
    """scancel the smoke jobs of ``request_id`` (every one of this loop's with ''), logged on failure."""
    if not hasattr(adapter, "cancel_smokes"):
        return []
    try:
        return list(adapter.cancel_smokes(f"loop-{request_id}") if request_id else adapter.cancel_smokes())
    except Exception as exc:  # noqa: BLE001 - the next sweep tries again
        store.log("cancel_smokes_failed", request_id=request_id, error=repr(exc)[:300])
        return []


def sweep_smokes(adapter: Any, store: Any, acts: list[str], now: float, last: float | None) -> float | None:
    """At start (deploy) and every SMOKE_SWEEP_S: scancel this loop's smoke jobs no live smoke() call waits on
    (their owner is gone: queued, nothing would ever time them out). Returns when the sweep last ran."""
    if not hasattr(adapter, "cancel_smokes") or (last is not None and now - last < SMOKE_SWEEP_S):
        return last
    if ids := cancel_smokes(adapter, store):
        acts.append(f"scancelled orphaned smoke job(s) {','.join(ids)}")
        store.log("smokes_swept", jobs=ids)
    return now


def repair_outage(store: Any, now: float, acts: list[str],
                  on_outage: Callable[[set[str]], list[str]] = lambda _ids: []) -> None:
    """Once (K_OUTAGE_REPAIR): the requests whose latest attempt ended backend_unavailable (an account outage) are
    no verdict; ``on_outage`` undoes what they were counted for (v3: nothing yet -- credit is paid from test blocks
    only, so an outage request never earned or lost any)."""
    if store.get_state(K_OUTAGE_REPAIR):
        return
    out = outage_requests(store)
    fixed = on_outage(out)
    store.set_state(K_OUTAGE_REPAIR, {"ts": now, "outage_requests": len(out), "fixed": fixed})
    store.log("outage_repair", outage_requests=len(out), fixed=fixed)
    if fixed:
        acts.append("outage repair: " + "; ".join(fixed))


# ---------------------------------------------------------------------------------------------- research sessions
class Research:
    """Researcher sessions in ``pool`` (one future each). ``K_INFLIGHT`` lists the sessions in flight then the
    queued restart re-issues, in the old driver's format (a cutover re-issues the old driver's sessions too)."""

    def __init__(self, ctx: LoopContext, store: Any, adapter: Any, researcher: Any, pool: Any,
                 clock: Callable[[], float] = time.time) -> None:
        self.ctx, self.store, self.adapter, self.researcher, self.pool, self.clock = \
            ctx, store, adapter, researcher, pool, clock
        self.inflight: dict[Future, CandidateRequest] = {}
        self.requeue: list[CandidateRequest] = []   # restart re-issues waiting for a slot under max_concurrent
        self.restarts: dict[str, int] = {}          # request id -> times re-issued after a restart
        self.unavailable: dict[str, int] = {}       # request id -> sessions that raised BackendUnavailable
        self.draining = False                       # stop requested: no new session
        self.disk_low = ""                          # why new sessions are held (the tick's disk guard), '' = not
        self.breaker = ""                           # why new sessions are held (loop.breaker tripped), '' = not
        #: request id -> the prompt dir its session renders from (loop.core.meta: the active version / A/B arm)
        self.prompt_dir: Callable[[str], Any] | None = None
        self.gpu_free = 0                           # free GPU slots this tick (the tick sets it): :meth:`limit`
        #: seconds the smoke gate still holds (the tick sets it from the adapter): infra re-issues wait it out
        self.smoke_hold: Callable[[], float] = lambda: 0.0
        self.infra_wait: set[str] = set()           # request ids re-issued after an infra smoke failure
        #: (state, why) of the smoke queue (the tick sets it from the adapter): ``closed`` starts no session, ``probe`` one
        self.smoke_capacity: Callable[[], tuple[str, str]] = lambda: ("open", "")

    def limit(self) -> int:
        """Sessions allowed at once: ``researcher.max_concurrent``, raised to ``ideation.llm_slots_max`` while more
        than ``ideation.llm_scale_free_slots`` (2) GPU slots would otherwise idle (``gpu_free``: the tick sets it from
        ``RunBook.idle`` -- free slots plus those held by idle-fill / relax runs; 10-13 review: ideas are what fills
        the GPUs)."""
        base = int(self.ctx.cfg.researcher.max_concurrent)
        ic = getattr(self.ctx.cfg, "ideation", None)
        hi, thr = int(getattr(ic, "llm_slots_max", 8)), int(getattr(ic, "llm_scale_free_slots", 2))
        return max(base, hi) if self.gpu_free > thr else base

    def free(self) -> int:
        """Sessions that may start now (0 while the LLM is paused / probing with one in flight, draining, or the
        disk guard holds new sessions)."""
        if self.researcher is None:
            return 0
        return self.capacity()

    def capacity(self) -> int:
        """:meth:`free` without the researcher check: what the LLM gate allows now (the shadow reports it)."""
        if self.draining or self.disk_low or self.breaker or llm_paused(self.store):
            return 0
        if llm_probing(self.store):
            return 0 if self.inflight else 1
        cap = self.smoke_capacity()[0]
        if cap == "closed":                         # 10-22: no session whose smoke cannot get a slot within its wait
            return 0
        if cap == "probe":
            return 0 if self.inflight else min(1, max(0, self.limit()))
        return max(0, self.limit() - len(self.inflight))

    def start(self, req: CandidateRequest, *args: Any, **log: Any) -> None:
        """Start ``researcher.research(req, store, *args)`` in the pool (prompts from ``prompt_dir(req.id)``).
        ``args`` default to ``(None,)``: ``LLMResearcher.research(request, store, snapshot)`` needs the snapshot
        positionally and v3 has none."""
        args = args or (None,)
        d = self.prompt_dir(req.id) if self.prompt_dir is not None else None
        self.inflight[self.pool.submit(in_prompt_dir, d, self.researcher.research, req, self.store, *args)] = req
        self.sync()
        self.store.log("research_started", request_id=req.id, strategy=req.strategy.value,
                       operator=req.operator.value, parents=list(req.parent_ids), brief_kind=req.brief_kind,
                       meta_id=req.meta_hypothesis_id, **({"prompt_dir": str(d)} if d else {}), **log)

    def start_requeued(self, *args: Any) -> list[str]:
        """Restart re-issues (and the one retry of a BackendUnavailable session) first, under the free session
        count; ``args`` default to ``(None,)`` (the snapshot ``LLMResearcher.research`` takes positionally)."""
        out, held = [], []
        while self.requeue and self.free() > 0:
            req = self.requeue.pop(0)
            if req.id in self.infra_wait and self.smoke_hold() > 0:    # the environment is still down: no LLM session yet
                held.append(req)
                continue
            self.infra_wait.discard(req.id)
            self.start(req, *(args or (None,)), restarted=True)
            out.append(f"researcher re-issued after restart: {req.id} {req.strategy.value}/{req.operator.value}")
        if held:
            self.requeue[:0] = held
            self.sync()
        return out

    def reissue_infra(self, req: CandidateRequest, res: Any) -> str:
        """A session whose smoke died of the ENVIRONMENT (loop.infra): no node, no reward, no 'produced nothing' -- the
        request is re-issued (same id) with its previous hypothesis and diff in the brief, up to MAX_INFRA_REISSUES
        times; the new session starts once the smoke gate lets a smoke through."""
        book = dict(self.store.get_state(INFRA.K_INFRA_REISSUES) or {})
        n = int(book.get(req.id, 0))
        again = n < INFRA.MAX_INFRA_REISSUES
        self.store.log("research_failed", request_id=req.id, status="smoke_infra", reissued=again,
                       error=str(getattr(res, "error", ""))[:500])
        if not again:
            return f"research {req.id}: smoke died of the environment {n} times; dropped (not re-issued again)"
        book[req.id] = n + 1
        self.store.set_state(INFRA.K_INFRA_REISSUES, book)
        self.requeue.append(INFRA.reissue_brief(req, getattr(res, "candidate", None), str(getattr(res, "error", ""))))
        self.infra_wait.add(req.id)
        self.sync()
        self.store.log("research_reissued", request_id=req.id, reason="smoke_infra", reissues=n + 1)
        return f"research {req.id}: smoke died of the environment (no verdict); re-issued ({n + 1}/{INFRA.MAX_INFRA_REISSUES})"

    def reissue_starved(self, req: CandidateRequest, res: Any) -> str:
        """A session whose smoke queue-timed-out (no slot, no verdict): its hypothesis and diff go into a re-issue of the
        same request instead of being dropped (10-22: four finished drafts vanished as 'produced nothing'). The new session
        starts when ``smoke_capacity`` lets one; up to MAX_STARVE_REISSUES per request."""
        book = dict(self.store.get_state(INFRA.K_STARVE_REISSUES) or {})
        n = int(book.get(req.id, 0))
        again = n < INFRA.MAX_STARVE_REISSUES
        self.store.log("research_failed", request_id=req.id, status="smoke_starved", reissued=again,
                       error=str(getattr(res, "error", ""))[:500])
        if not again:
            return f"research {req.id}: smoke starved {n} times; dropped (not re-issued again)"
        book[req.id] = n + 1
        self.store.set_state(INFRA.K_STARVE_REISSUES, book)
        self.requeue.append(INFRA.reissue_brief(req, getattr(res, "candidate", None), str(getattr(res, "error", ""))))
        self.sync()
        self.store.log("research_reissued", request_id=req.id, reason="smoke_starved", reissues=n + 1)
        return f"research {req.id}: smoke got no slot in time (no verdict); re-issued with its diff ({n + 1}/{INFRA.MAX_STARVE_REISSUES})"

    def harvest(self, acts: list[str]) -> list[tuple[CandidateRequest, Any]]:
        """The finished sessions' (request, result); a backend outage keeps the result for the next tick."""
        done = []
        for fut in [f for f in self.inflight if f.done()]:
            req = self.inflight.pop(fut)
            if ids := cancel_smokes(self.adapter, self.store, req.id):   # a smoke left queued would start on a
                acts.append(f"scancelled smoke job(s) {','.join(ids)} left by {req.id}")   # reaped worktree
            try:
                res = fut.result()
            except BackendUnavailable as exc:          # the backend is down: release the slot, retry ONCE
                n = self.unavailable[req.id] = self.unavailable.get(req.id, 0) + 1
                self.store.log("research_backend_unavailable", request_id=req.id, tries=n, error=repr(exc)[:300])
                if n <= 1:
                    self.requeue.append(req)           # re-issued by start_requeued once the LLM gate allows
                    acts.append(f"research {req.id}: LLM backend unavailable; slot released, retried once")
                else:
                    self.store.log("research_failed", request_id=req.id, status="backend_unavailable",
                                   error=repr(exc)[:500])
                    acts.append(f"research {req.id}: LLM backend unavailable again; dropped")
                continue
            except Exception as exc:  # noqa: BLE001
                self.store.log("research_error", request_id=req.id, error=repr(exc)[:500])
                acts.append(f"research {req.id} raised {exc!r}"[:200])
                continue
            if getattr(res, "status", "") != "backend_unavailable":
                llm_probe_ok(self.store)
            done.append((req, res))
        self.sync()
        return done

    def restore(self, acts: list[str]) -> None:
        """Requests the previous process left in flight (K_INFLIGHT) died with it: each is logged
        ``research_failed status=restart`` and queued for re-issue (same id, brief; a fresh session) up to
        MAX_RESTARTS times; its orphaned smoke jobs are scancelled. A queued re-issue that never started is no new
        restart."""
        live = {r.id for r in [*self.inflight.values(), *self.requeue]}
        for e in self.store.get_state(K_INFLIGHT) or []:
            if e["id"] in live:
                continue
            again = "req" in e and (e.get("queued") or e.get("restarts", 0) < MAX_RESTARTS)
            req = build(CandidateRequest, e["req"], "runner.inflight.req") if again else None
            if ids := cancel_smokes(self.adapter, self.store, e["id"]):
                acts.append(f"scancelled orphaned smoke job(s) {','.join(ids)} of {e['id']}")
            if not e.get("queued"):
                self.store.log("research_failed", request_id=e["id"], status="restart", reissued=bool(req),
                               error="the loop restarted while this request was in flight")
            if req is None:
                acts.append(f"research {e['id']} lost to a restart (not re-issued)")
                continue
            self.restarts[req.id] = e.get("restarts", 0) + (0 if e.get("queued") else 1)
            self.requeue.append(req)
            acts.append(f"researcher re-issue queued after restart: {req.id} {req.strategy.value}/{req.operator.value}")
        self.sync()

    def sync(self) -> None:
        """K_INFLIGHT: the sessions in flight, then the queued restart re-issues (``queued``: never started)."""
        cur = [{"id": r.id, "strategy": r.strategy.value, "family": r.family_hint, "parents": list(r.parent_ids),
                "req": encode(r), "restarts": self.restarts.get(r.id, 0), **({"queued": True} if q else {})}
               for rs, q in ((self.inflight.values(), False), (self.requeue, True)) for r in rs]
        if self.store.get_state(K_INFLIGHT) != cur:
            self.store.set_state(K_INFLIGHT, cur)

    def wait(self) -> None:
        """Block until every session finished (tests / a clean shutdown); never kills one."""
        for fut in list(self.inflight):
            try:
                fut.result()
            except Exception:  # noqa: BLE001 - reported when harvested
                pass


# ---------------------------------------------------------------------------------------------- data bank, Labless
def databank_tick(ctx: LoopContext, store: Any, adapter: Any, now: float) -> list[str]:
    """Register ready datasets and settle / launch builds. No adoption here: adopting a dataset was a campaign
    on the incumbent (an old lane); in v3 it becomes a factorial knob (docs/loop-core-v3.md §5)."""
    try:
        return DB.tick(ctx, store, adapter, None, new_id=None, now=now)
    except Exception as exc:  # noqa: BLE001 - the data bank never stops the loop
        store.log("databank_error", error=repr(exc)[:500])
        return [f"databank tick failed: {exc!r}"[:300]]


#: the "labless: incumbent ... not submitted" line (and its event) at most this often per incumbent (10-22: one per new TEST seed)
BELOW_BAR_LOG_S = 3600.0


class IncumbentLabless:
    """Labless gets the incumbent only, once per incumbent: its latest DONE run (a test-block run on a fresh
    seed), one submission in flight in ``pool``, retried hourly on failure. State: K_LABLESS.
    With a bar configured (``labless.leader_median`` / ``bar_node``; operator 2026-10-13) an incumbent is submitted only
    once it BEATS THE BAR (``loop.core.bar``: P(median of 3 fresh seeds >= leader_median + promote_margin) >=
    promote_p on >= min_test_seeds fresh TEST seeds) on
    ``data``; until then it waits (re-judged every tick as its seeds accrue; logged ``labless_below_bar`` once per
    seed count and at most hourly). No ``data`` with a bar configured: nothing is submitted."""

    def __init__(self, ctx: LoopContext, store: Any, pool: Any, clock: Callable[[], float] = time.time,
                 submit: Callable[..., str] | None = None) -> None:
        from loop.labless import submit_run
        self.ctx, self.store, self.pool, self.clock = ctx, store, pool, clock
        self.submit = submit or submit_run
        self.fut: tuple[Future, str] | None = None

    def below_bar(self, incumbent: str, data: Any, book: dict[str, Any]) -> str | None:
        """'' when the incumbent beats the bar (or no bar is configured), else why it waits (None: already said)."""
        lc = self.ctx.cfg.labless
        from loop.core import bar as BR
        if not BR.configured(lc):
            return ""
        if data is None:
            return None
        bar = BR.bar_from(data, lc)
        b = BR.beat(data, bar, incumbent) if bar is not None else None
        if b is not None and b.beats:
            return ""
        e = book.get(incumbent, {})
        seen = b.n_test if b is not None else 0
        now = self.clock()
        if e.get("below_bar_n") == seen or now - float(e.get("below_bar_ts", -BELOW_BAR_LOG_S)) < BELOW_BAR_LOG_S:
            return None                  # said for this seed count, or said within the hour (a new TEST seed every few min)
        book[incumbent] = {**e, "below_bar_n": seen, "below_bar_ts": now}
        self.store.set_state(K_LABLESS, book)
        why = BR.beat_text(b, bar) if b is not None else ("bar not measurable" if bar is None else "no current-epoch run")
        self.store.log("labless_below_bar", node_id=incumbent, bar=lc.bar_node, n=seen,
                       p=round(b.p_test, 3) if b is not None and b.p_test is not None else None, d=round(b.d, 5) if b is not None else None,
                       need=round(b.need, 5) if b is not None else None, target=round(bar.target, 5) if bar else None)
        return f"labless: incumbent {incumbent} not submitted, does not beat {BR.bar_text(bar) if bar else lc.bar_node}: {why}"

    def step(self, incumbent: str | None, data: Any = None) -> list[str]:
        from loop.labless import run_name
        book, now = dict(self.store.get_state(K_LABLESS) or {}), self.clock()
        if self.fut is not None:
            if not self.fut[0].done():
                return []
            (fut, nid), self.fut = self.fut, None
            e = book[nid]
            try:
                e.update(submission_id=fut.result(), error="")
                self.store.log("labless_submitted", node_id=nid, tier="incumbent", submission_id=e["submission_id"])
                msg = f"labless: incumbent {nid} submitted, submission_id={e['submission_id']}"
            except Exception as exc:  # noqa: BLE001 - logged, retried after the backoff
                e.update(error=repr(exc)[:500], next_ts=now + LABLESS_BACKOFF_S)
                self.store.log("labless_failed", node_id=nid, tier="incumbent", tries=e["tries"], error=e["error"])
                msg = f"labless: incumbent {nid} failed (try {e['tries']})"
            self.store.set_state(K_LABLESS, book)
            return [msg]
        if not self.ctx.cfg.labless.enabled or not incumbent or incumbent == ROOT_NODE_ID:
            return []
        e = book.get(incumbent, {})
        if e.get("submission_id") or e.get("duplicate") or now < float(e.get("next_ts", 0.0)):
            return []
        if getattr(self.ctx.cfg.labless, "dedupe_submitted", True):   # loop.leader: never a second submission
            from loop import leader as LD
            if dup := LD.submitted(self.ctx, self.store, incumbent):
                book[incumbent] = {**e, "duplicate": dup}
                self.store.set_state(K_LABLESS, book)
                self.store.log("labless_duplicate", node_id=incumbent, tier="incumbent", prior=dup)
                return [f"labless: incumbent {incumbent} not submitted: already on Labless ({dup})"]
        if (wait := self.below_bar(incumbent, data, book)) != "":
            return [wait] if wait else []
        runs = sorted((r for r in self.store.runs(node_id=incumbent, status=[RunStatus.DONE]) if r.output_dir),
                      key=lambda r: (r.submitted_ts or 0.0, r.id))
        if not runs:
            return []
        run = runs[-1]
        book[incumbent] = {**e, "run_id": run.id, "tries": int(e.get("tries", 0)) + 1, "submission_id": None,
                           "next_ts": now + LABLESS_BACKOFF_S}   # set up front: a crash mid-call backs off too
        self.store.set_state(K_LABLESS, book)
        self.store.log("labless_submit", node_id=incumbent, run_id=run.id, tries=book[incumbent]["tries"],
                       tier="incumbent")
        from loop.core import bar as BR
        lc = self.ctx.cfg.labless
        bar = (f"Labless median≥{BR._f(sum(BR.leader_margin(lc)))}" if BR.leader_margin(lc)[0] is not None
               else lc.bar_node) if BR.configured(lc) else ""
        from loop.labless import submit_run
        self.fut = (self.pool.submit(self.submit, self.ctx, run.output_dir, run_name(incumbent, None, "", "v3"),
                                     f"v3 incumbent {incumbent}: passed the fresh-seed test"
                                     + (f" and beats the bar {bar}" if bar else ""),
                                     # the {node} of labless.push.ref; injected test submitters keep their 4 args
                                     **({"node_id": incumbent} if self.submit is submit_run else {})), incumbent)
        return [f"labless: submitting incumbent {incumbent} from {run.id}"]
