"""loop.tick — one v3 tick (docs/loop-core-v3.md §1): poll -> derive -> decide -> acquire -> launch -> ideate -> meta.

Every decision comes from a :class:`Core`; the tick only wires it to the plumbing (``loop.runner``), writes the seed
blocks the core asks for, and enforces ``check_submit`` before every submit (``RunBook.guard``: launches, retries,
OOM relaunches, crash-adopted rows). Statuses are derived each tick, never stored.

- derive: ``status.derive`` over the ledger and the ``seed_blocks``.
- decide: ``status.decide_open`` on the open TEST blocks: PASS / FAIL close the block and FREEZE its verdict (with the
  σ_d / ρ / dfs / change class used: ``seed_blocks.verdicts``; replay never re-judges it), a PASS that switches
  re-issues its seeds as the new incumbent's SELECT block; EXTEND (once per block, cap K_MAX) draws ``more``
  never-used seeds into it; a pair whose run is dead (crashed, or a void that is not retried) gets a replacement seed.
  A closed block without frozen verdicts (historical) is frozen once with its legacy judgement.
- launch: owed infra retries, then the runs every open test block still needs, then the ranked actions:
  ``test`` -> a new TEST block (``seeds.draw_test``) and CONFIRM (candidate) + REMINT (incumbent) runs on it,
  ``seed`` -> one SCREEN run on a SELECT seed (the parent's first, CRN), ``rebase`` -> BASELINE runs of the new base
  root and the incumbent's port, ``wave`` -> its points at their block seeds (``core.factorial.wave_runs``);
  ``screen`` -> the first SCREEN runs of a QUEUED node (an intake idea, an x*), on its parent's CRN seeds;
  ``port`` (and an unported incumbent) -> the ``port`` hook: ``loop.cutover.port_node`` grafts the old node's change
  onto the current base (a ``port_of`` node) and its screening runs launch in the same tick; a conflicting graft is
  a REJECTED port node (never offered again) and gets ONE researcher port session (``req-v3-port-<old>``).
- ideate / intake: :class:`loop.core.ideation.Ideation` by default (step 10): one brief per ``idea`` action (shadow:
  the briefs it would start), a research result -> a node; it reads the tick's core posterior (``Tick.post``).
  meta: :class:`loop.core.meta.Meta` by default (step 11): the 3-hour report / review / acceptance, and its
  ``guard`` -- a LOCKED file that differs from the deployed HEAD trips the breaker: nothing launches, no session.

:class:`ModuleCore` is the real core (``loop.core``); :class:`StubCore` is the inert stand-in and default seed picker."""

from __future__ import annotations

import collections
import inspect
import math
import json
import os
import re
import shutil
import sqlite3
import sys
import time
from types import SimpleNamespace
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from loop import infra as INFRA
from loop import resume as RS
from loop import runner as R
from loop.contracts import ROOT_NODE_ID, Run, RunStatus, Stage, SubmissionRejected, VoidType
from loop.core import data as CD

from loop.spec import LoopContext

#: stages of the runs an action launches: selection seeds are SCREEN; a fresh-seed test runs the candidate (CONFIRM)
#: and the incumbent (REMINT) on the same TEST seeds; the re-baseline after a base change is BASELINE
SEED_STAGE, TEST_STAGE, TEST_CONTROL_STAGE, REBASE_STAGE = Stage.SCREEN, Stage.CONFIRM, Stage.REMINT, Stage.BASELINE
CLOSED = frozenset({"FAILED", "VOID", "CRASHED"})   # derived statuses whose infra voids are not retried
INCUMBENT = "INCUMBENT"
#: TEST-stage runs (CONFIRM / REMINT on test seeds) are never resumed from latest.pt after a preemption: a resumed
#: pair is not exact, and the simulation lost the test's power to preemption + resume; they restart from step 0
#: (and go to the primary pool first: ``RunBook.high`` -- every stage but SCREEN)
TEST_STAGES = frozenset({TEST_STAGE, TEST_CONTROL_STAGE})
#: below this many GB free on the state / worktree filesystem no new research session starts (an alert is logged)
DISK_FLOOR_GB = float(os.environ.get("LOOP_DISK_FLOOR_GB", "30"))
#: a call site (port / wave / intake / ...) that failed this many times in a row is QUARANTINED: skipped for good
#: (until the operator clears ``tick.quarantine``); a quarantined port is recorded as a REJECTED port node
QUARANTINE_AFTER = 3
K_ERRORS, K_QUARANTINE = "tick.errors", "tick.quarantine"
K_WAVE_BUILD = "factorial.building"     # the wave being materialised: a crash mid-build is resumed, never re-drawn


def code_rev() -> str:
    """The git revision this process runs ('' when it cannot be read): a quarantine records it, and a restart on a
    DIFFERENT revision releases the sites quarantined by the old code (live 10-21 23:06: ``queue.age_out`` was
    quarantined by a bug fixed at 23:00 by a deploy -- and stayed disabled, since the entry only left by hand)."""
    import subprocess
    try:
        r = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "--short=12", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:  # noqa: BLE001 - no git: nothing is ever auto-released
        return ""


@dataclass(frozen=True)
class Action:
    """One ranked unit of work (mirrors ``loop.core.acquire.Action``, docs §3)."""
    kind: str               # idea | seed | wave | test | port | rebase
    node: str | None
    lane: str
    gpu_h: float
    value: float
    runs: int = 0
    why: str = ""
    pre: bool = False       # the relax pass (acquire.rank): preemptible-only


LOCAL_SEEDS_RE = re.compile(r"(?:^|\s)local_seeds=([0-9,]+)")


def local_seeds(notes: str) -> set[int]:
    """The seeds a manual run's confirm node may never screen on (notes ``local_seeds=1,7338``: its source runs')."""
    m = LOCAL_SEEDS_RE.search(notes or "")
    return {int(x) for x in m.group(1).split(",") if x} if m else set()


@dataclass(frozen=True)
class RunSpec:
    node: str
    stage: Stage
    seed: int


class Core(Protocol):
    """What the tick needs from loop.core (``blocks``: ``store.seed_blocks()``, re-read whenever the tick writes)."""

    def noise(self, data: CD.Ledger) -> Any: ...

    def derive(self, data: CD.Ledger, blocks: Sequence[Any], noise: Any) -> Mapping[str, Any]: ...

    def decide(self, data: CD.Ledger, blocks: Sequence[Any], noise: Any, incumbent: str) -> list[tuple[Any, str, Any]]:
        """(open TEST block, candidate, verdict with ``.kind`` pass|fail|extend|open and ``.more``)."""

    def posterior(self, data: CD.Ledger, blocks: Sequence[Any]) -> Any:
        """This tick's posterior (``acquire.Posterior``), shared by ``rank`` and ideation; None without data."""

    def rank(self, data: CD.Ledger, blocks: Sequence[Any], statuses: Mapping[str, Any], post: Any,
             free_slots: int, llm_free: int) -> list[Any]: ...

    def open_test(self, node: str, incumbent: str, data: CD.Ledger, blocks: Sequence[Any], now: float,
                  cfg: Any) -> Any:
        """A new TEST block for ``node`` vs ``incumbent`` (None: the core keeps no blocks, or one is open)."""

    def draw(self, k: int, data: CD.Ledger, blocks: Sequence[Any], cfg: Any) -> tuple[int, ...]: ...

    def runs_for(self, action: Any, data: CD.Ledger, blocks: Sequence[Any], statuses: Mapping[str, Any],
                 cfg: Any) -> list[RunSpec]: ...

    def check_submit(self, run: Run, blocks: Sequence[Any]) -> None: ...

    def migrate(self, store: Any, data: CD.Ledger) -> int: ...


def _submitted(run: Any) -> str:
    """The submit line's state: a RUNNING row right after a submit is only queued in SLURM (it may sit PENDING)."""
    if getattr(run, "status", None) is RunStatus.RUNNING:
        return f"submitted (job {run.job_id})" if getattr(run, "job_id", "") else "submitted"
    return str(getattr(getattr(run, "status", None), "value", ""))


def status_name(status: Any) -> str:
    """'INCUMBENT' for Status.INCUMBENT, 'incumbent', or an object with ``.status``."""
    s = getattr(status, "status", status)
    return str(getattr(s, "name", None) or getattr(s, "value", None) or s).upper()


def T_K(core: Any) -> int:
    """The test's default seeds (``test.K``): a block holding more was extended already and never is again."""
    from loop.core import test as T
    return int(getattr(getattr(core, "costs", None), "k_test", T.K) or T.K)


K_NODE_SYNC = "tick.node_status_sync"    # {node_id: the stored status the tick wrote}: the nodes whose stored status it owns
#: derived status -> the stored ``NodeStatus`` it is written as. PASSED / INCUMBENT have NO entry on purpose: the only
#: stored equivalent is CONFIRMED, which sends the node to Labless (audited / exploratory tiers), pins its checkpoints
#: for good (prune.hard) and blocks its dataset from the databank's retention list -- a derived status must never grant that
SYNC_TO = {"QUEUED": "queued", "SCREENING": "screening", "TESTING": "confirming", "FAILED": "not_confirmed",
           "VOID": "void", "CRASHED": "crashed"}
#: stored statuses the sync may overwrite: the v3 live ones. Every other stored status (rejected, vetoed, baseline,
#: confirmed, screened_out, audit_failed, eval_*, a void another path set) is an explicit act and stays; one the sync
#: itself wrote (``K_NODE_SYNC``) follows the derivation from then on, so void -> screening flaps are not stuck
SYNC_FROM = frozenset({"queued", "screening", "promoting", "confirming"})
#: derived statuses that CLOSE a node: an incremental sync writing one stamps ``updated_ts = now`` (prune's grace for a
#: newly failed node starts at close). The first mass pass (> SYNC_EVENT_MAX) keeps the old timestamps: re-stamping ~3000
#: historical closes would restart their grace.
SYNC_TERMINAL = frozenset({"not_confirmed", "void", "crashed"})
SYNC_EVENT_MAX = 20        # more changes than this in one tick (the first pass: ~3000) log ONE node_status_sync summary event

K_TICK_INC = "tick.incumbent"     # the status-derived incumbent of the last tick (the heartbeat's; meta's bar section uses it)


def incumbent_of(statuses: Mapping[str, Any]) -> str:
    return next((n for n, s in statuses.items() if status_name(s) == INCUMBENT), ROOT_NODE_ID)


def member_fault(run: Any) -> bool:
    """A dead run its node is answerable for: a crash, an eval / too_slow void. Not an infra void (preemption, node
    failure: retried, then re-drawn) nor a loop cancel."""
    return run is not None and (run.status is RunStatus.CRASH or run.status is RunStatus.VOID
                                and run.void_type not in (None, VoidType.INFRA, VoidType.CANCELLED))


_EXC = re.compile(r"^(?:[\w.]+\.)?(\w*(?:Error|Exception|Interrupt|Exit))\b", re.M)
#: v3-c7a A4: the LOCKED probe turns sksurv's Coxnet ConvergenceWarning into an exit-1 probe crash: the candidate's
#: features are too wide / ill-conditioned for the survival probe (live 10-13: 23 such crashes on 19 nodes; a node's
#: 2080-dim covariance block) -- an IDEA defect to repair, not an infra flake and not a judged FAIL
ILL_CONDITIONED = "crash:probe_ill_conditioned"
_ILL = re.compile(r"ConvergenceWarning|\bcoxnet\b|Coxnet", re.I)


def crash_class(run: Any) -> str:
    """A short class of a dead run: ``void:<type>``, or ``crash:oom`` / ``crash:probe_ill_conditioned`` (the probe's
    survival fit did not converge: :data:`ILL_CONDITIONED`) / ``crash:probe`` (the probe subprocess failed) /
    ``crash:nan`` / ``crash:<last exception name>`` / ``crash:unknown`` from its log tail."""
    if run is None:
        return "missing"
    if run.status is RunStatus.VOID:
        return f"void:{getattr(run.void_type, 'value', None) or 'infra'}"
    from loop.contracts import is_oom
    tail = run.log_tail or ""
    if is_oom(tail):
        return "crash:oom"
    if "probe.py" in tail and _ILL.search(tail):
        return ILL_CONDITIONED
    if "probe.py" in tail:
        return "crash:probe"
    if "resume_shim" in tail:                       # the eval-only lane's resume shim (loop.evalonly)
        return "crash:shim"
    if re.search(r"latest\.pt|Error\(s\) in loading state_dict|train\.resume", tail):
        return "crash:checkpoint"                   # a resumed / saved checkpoint, not the candidate's code
    if re.search(r"\bnan\b|non-finite", tail, re.I):
        return "crash:nan"
    exc = _EXC.findall(tail)
    return f"crash:{exc[-1]}" if exc else "crash:unknown"


#: v3-c5b S3: crash classes that are the loop's / the eval lane's, never the member's (re-drawn, not counted)
INFRA_CRASH = frozenset({"crash:shim", "crash:checkpoint"})
#: ... and the only crash classes an EVAL-ONLY run (parent checkpoint + candidate readout) is answerable for
EVAL_RUN_FAULT = frozenset({"crash:probe", "crash:nan", "crash:oom", ILL_CONDITIONED})
K_PROBE_RETRY = "test.probe_retried"             # run ids of TEST-block crash:probe runs retried once on their seed


def test_fault(run: Any, *, eval_run: bool = False, pending_retry: bool = False) -> bool:
    """A dead TEST-block run that counts against its member (v3-c5b S3): :func:`member_fault`, except a too_slow void
    (the machine / queue, re-drawn), a shim / checkpoint crash, an eval-only run's crash outside the probe / nan / oom
    classes (the eval lane's own error) and a crash:probe still owed its one same-seed retry (``pending_retry``)."""
    if not member_fault(run) or run.status is RunStatus.VOID and run.void_type is VoidType.TOO_SLOW:
        return False
    cls = crash_class(run)
    if cls in INFRA_CRASH or eval_run and run.status is RunStatus.CRASH and cls not in EVAL_RUN_FAULT:
        return False
    return not (cls == "crash:probe" and pending_retry)


def _tally(classes: Sequence[str]) -> str:
    """'crash:probe x3, void:eval' -- the classes of a member's crashes, most frequent first."""
    n: dict[str, int] = {}
    for k in classes:
        n[k] = n.get(k, 0) + 1
    return ", ".join(f"{k} x{v}" if v > 1 else k for k, v in sorted(n.items(), key=lambda kv: (-kv[1], kv[0])))


class StubCore:
    """Inert core: nothing derived, nothing ranked, no seed blocks, every submit allowed. Its ``runs_for`` picks
    seeds without blocks: a ``seed`` action takes the next of its parent's seeds it has not run (CRN), else its
    lowest unused seed from 1 (siblings share seeds); a ``test`` takes k never-used seeds >= ``seeds.fresh_base``
    (its own earlier ones first) for the candidate and the incumbent."""

    TEST_K = 5

    def noise(self, data: CD.Ledger) -> Any:
        return None

    def derive(self, data: CD.Ledger, blocks: Sequence[Any], noise: Any) -> Mapping[str, Any]:
        return {}

    def decide(self, data: CD.Ledger, blocks: Sequence[Any], noise: Any, incumbent: str) -> list[tuple[Any, str, Any]]:
        return []

    def posterior(self, data: CD.Ledger, blocks: Sequence[Any]) -> Any:
        return None

    def rank(self, data: CD.Ledger, blocks: Sequence[Any], statuses: Mapping[str, Any], post: Any,
             free_slots: int, llm_free: int) -> list[Any]:
        return []

    def open_test(self, node: str, incumbent: str, data: CD.Ledger, blocks: Sequence[Any], now: float,
                  cfg: Any) -> Any:
        return None

    def draw(self, k: int, data: CD.Ledger, blocks: Sequence[Any], cfg: Any) -> tuple[int, ...]:
        start = max([r.seed for r in data.runs if r.seed >= cfg.seeds.fresh_base] or [cfg.seeds.fresh_base - 1]) + 1
        return tuple(range(start, start + k))

    def check_submit(self, run: Run, blocks: Sequence[Any]) -> None:
        return None

    def migrate(self, store: Any, data: CD.Ledger) -> int:
        return 0

    def runs_for(self, action: Any, data: CD.Ledger, blocks: Sequence[Any], statuses: Mapping[str, Any],
                 cfg: Any) -> list[RunSpec]:
        fresh = int(cfg.seeds.fresh_base)
        if action.kind == "seed" and action.node in data.nodes:
            mine = {r.seed for r in data.runs if r.node == action.node}
            parent = data.nodes[action.node].parent or ROOT_NODE_ID
            crn = sorted({r.seed for r in data.runs if r.node == parent and r.seed < fresh} - mine)
            seed = crn[0] if crn else min(set(range(1, len(mine) + 2)) - mine)
            return [RunSpec(action.node, SEED_STAGE, seed)]
        if action.kind == "test" and action.node in data.nodes:
            inc = incumbent_of(statuses)
            have = sorted({r.seed for r in data.runs if r.node == action.node and r.stage == TEST_STAGE.value
                           and r.seed >= fresh})              # an open block keeps its seeds, topped up to k
            seeds = [*have, *self.draw(max(0, self.TEST_K - len(have)), data, blocks, cfg)]
            return [x for s in seeds for x in (RunSpec(action.node, TEST_STAGE, s), RunSpec(inc, TEST_CONTROL_STAGE, s))]
        return []


class ModuleCore(StubCore):
    """The loop.core modules: ``model``/``backtest.fit_at`` (noise), ``status.derive`` / ``decide_open``,
    ``acquire.ledger_options`` + ``rank``, ``seeds`` (blocks, draws, ``check_submit``, the one-time migration)."""

    missing: list[str] = []

    def __init__(self, costs: Any = None) -> None:
        from loop.core import acquire, backtest, seeds, status
        self.A, self.S, self.ST, self.fit_at = acquire, seeds, status, backtest.fit_at
        self.costs = costs or acquire.Costs()
        self.harm: dict[str, Any] = {}               # rank's harm_q / harm_h: the meta-cycle's TUNABLE config
        self.floors: dict[str, float] = {}           # rank keyword overrides (floor / wave_floor), e.g. in tests
        self.near: frozenset[str] = frozenset()      # nodes ONE config key from their parent, config only (classify)
        # v3-c5 I5: pure eval-only readout nodes (loop.evalonly.Lane.pure) and their pair-noise floor / min dfs
        self.evalonly: frozenset[str] = frozenset()
        self.eo_floor, self.eo_min_df = 0.0, 4
        #: v3-c5b M1: every run with an eval plan (eval-only runs + null controls): left out of every noise fit
        self.eo_runs: frozenset[str] = frozenset()
        self.near_pair: Callable[[str, str], bool] = lambda c, i: False
        self.notes: list[str] = []                   # lines for the tick log (Tick.step takes them after rank)
        self._floor_seen: set[tuple[str, Any]] | None = None   # gap-floor drops already logged (K_FLOOR_SEEN)
        self._store: Any = None                      # the tick's store (configure): persisted acquisition state
        #: 11-11 review: node -> runs it may still get this tick (a moonshot rescue's port / rung: <= its rescue seeds)
        self.seed_caps: dict[str, int] = {}

    K_TARGET = "acquire.target_active"   # the target-mode hysteresis state (acquire.target_active)
    K_FLOOR_SEEN = "acquire.floor_seen"  # [[kind, node], ...]: gap-floor drops already logged (once per candidate)
    CHURN_REASON = "idle fill preempted"  # run_cancelled reason prefix of the idle churn (runner.RunBook.idle_churn)

    def configure(self, cfg: Any, store: Any = None) -> None:
        """The operator keys of the acquisition (``loop.acquire`` / ``loop.status`` / the ``loop.labless`` target);
        ``store``: where the acquisition's state persists across restarts (target mode, logged gap-floor drops)."""
        self.A.configure(cfg)
        self.stack_priority = bool(getattr(getattr(cfg, "factorial", None), "stack_priority", True))
        self.alloc_cfg = getattr(cfg, "alloc", None)  # v3-c7i: the GPU split / early kills (loop.core.alloc)
        self._cfg = cfg
        if store is not None:
            self._store = store

    def _state(self, key: str, default: Any = None) -> Any:
        try:
            return self._store.get_state(key, default) if self._store is not None else default
        except Exception:  # noqa: BLE001 - no state: the defaults
            return default

    def _put(self, key: str, value: Any) -> None:
        if self._store is not None:
            try:
                self._store.set_state(key, value)
            except Exception:  # noqa: BLE001 - not persisted: re-derived next tick
                pass

    def _target_mode(self, data: CD.Ledger, incumbent: str) -> None:
        """Target mode with hysteresis (``acquire.target_active``), its state persisted in ``K_TARGET``: entered when
        the gap > its se, left when it falls below -se. Sets ``acquire.TARGET_ACTIVE`` for this tick's ranking."""
        A = self.A
        got = A.gap_se(data, incumbent) if A.target_on() else None
        if got is None:
            A.TARGET_ACTIVE = None
            return
        was = self._state(self.K_TARGET)
        now = A.target_active(*got, bool(was) if was is not None else False)
        if now != was:
            self._put(self.K_TARGET, now)
            self.notes.append(f"target mode {'on' if now else 'off'}: gap {got[0]:+.5f} (se {got[1]:.5f})")
        A.TARGET_ACTIVE = now

    stack_priority: bool = True              # factorial.stack_priority (configure)

    def stacking(self) -> set[str]:
        """Carriers whose bank's wave stacks >= 2 positive legs (``stacking`` diff knobs among its chosen knobs)."""
        from loop.core import factorial as F
        return {c for c, bank in (self.knobs or {}).items()
                if sum(1 for b in F.choose(bank) if b.source == "stacking" and b.knob.scale == "diff") >= 2}

    def _stack_first(self, opts: Sequence[Any], out: list[Any], ranked: Callable[..., list[Any]],
                     free_slots: int | None = None) -> list[Any]:
        """``factorial.stack_priority``: a stacking wave (a crossover of >= 2 positive legs) goes before a relax-pass
        (below-floor) wave of plain knobs. When rank left a stacking wave out, rank again with the plain waves it did
        not pick and the relax waves it picked removed, and the relax pass allowed one wave (even at
        ``acquire.relax_max_waves`` 0: only a stacking wave can take it); kept when that ranking holds the stacking
        wave. A stacking wave picked in the relax pass runs on primary slots (a crossover of measured gains is no
        idle fill: 10-13, relax wave points were churned to death).

        Slot-starved (``free_slots`` known): a wave is ranked only when ALL its runs fit the free slots, which on a
        saturated cluster (live 10-13: gpu_free 0-2 on ~90% of ticks; a 7-knob stacking wave needs 16) is ~never --
        the 1-run seeds / screens take every slot as it frees. When the stacking wave would be ranked at full size
        (value, hard caps, tag caps: a ranking of it alone with its runs as the slots) it takes the slots the
        primary-pass (above-floor) actions leave, the relax-pass picks giving way; ``Tick._launch`` submits what fits
        now and an open wave's never-ran points go first on the next ticks (``_act`` needs one free slot)."""
        if not self.stack_priority or not (stack := self.stacking()):
            return out
        got = {a.node for a in out if a.kind == "wave"}
        want = sorted(o.node for o in opts if o.kind == "wave" and o.node in stack and o.node not in got)
        if want:
            bump = {a.node for a in out if a.kind == "wave" and getattr(a, "pre", False) and a.node not in stack}
            keep = got - bump
            pool = [o for o in opts if not (o.kind == "wave" and o.node not in stack and o.node not in keep)]
            trial = ranked(pool, relax_wave=1)
            if not any(a.kind == "wave" and a.node in stack for a in trial) and free_slots:
                trial = self._stack_starved(pool, stack, ranked, free_slots) or trial
            if any(a.kind == "wave" and a.node in stack for a in trial):
                out = trial
                self.notes.append(f"stack priority: the stacking wave on {', '.join(want)} ranked first"
                                  + (f" (relax wave(s) on {', '.join(sorted(bump))} give way)" if bump else ""))
        # v3-c5b S1: the primary path keeps the wave floor -- a stacking wave the relax pass picked runs on primary
        # slots only when it clears WAVE_FLOOR against the whole option set; below it, it stays preemptible
        return [replace(a, pre=False, why=f"{a.why} [stacking: primary]")
                if a.kind == "wave" and a.node in stack and getattr(a, "pre", False) and self._wave_clears(opts, a.node)
                else a for a in out]

    def _wave_clears(self, opts: Sequence[Any], node: str) -> bool:
        """The wave on ``node`` clears the primary pass's wave floor (``acquire.floors``: ``wave_floor`` x the best
        screen's value/GPU-h) among ALL of ``opts`` (options, or already-scored actions) -- ranked alone, a wave's
        floor is 0 (no screen to compare)."""
        try:
            acts = list(opts) if all(hasattr(o, "value") and not hasattr(o, "m") for o in opts) \
                else self.A.score(opts, **self.harm)
        except Exception:  # noqa: BLE001 - no floor computable: not on primary
            return False

        def rate(a: Any) -> float:
            return a.value / a.gpu_h if a.gpu_h > 0 else math.inf
        wf = float({"wave_floor": self.A.WAVE_FLOOR, **self.floors}["wave_floor"])
        ref = max([rate(a) for a in acts if a.kind in ("screen", "idea") and a.value > 0], default=0.0)
        scr = max([rate(a) for a in acts if a.kind == "screen" and a.value > 0], default=ref)
        a = next((a for a in acts if a.kind == "wave" and a.node == node), None)
        return a is not None and a.value > 0 and not getattr(a, "below", False) and rate(a) >= wf * scr

    def _stack_starved(self, pool: Sequence[Any], stack: set[str], ranked: Callable[..., list[Any]],
                       free_slots: int) -> list[Any]:
        """:meth:`_stack_first`'s slot-starved case: the primary-pass actions of ``pool`` without the stacking wave,
        then the stacking wave (best m) on the slots they leave (>= 1); [] when it is not rankable at full size or no
        slot is left."""
        waves = [o for o in pool if o.kind == "wave" and o.node in stack]
        if not waves:
            return []
        w = max(waves, key=lambda o: getattr(o, "m", 0.0))
        if not self._wave_clears(pool, w.node):          # S1: primary slots only above the wave floor
            return []
        lone = next((a for a in ranked([w], relax_wave=1, slots=w.runs) if a.kind == "wave"), None)
        if lone is None:                                 # no value / a hard or tag cap: not a slot question
            return []
        prim = [a for a in ranked([o for o in pool if o is not w], relax_wave=0) if not getattr(a, "pre", False)]
        room = free_slots - sum(a.runs for a in prim if a.kind != "idea")
        if room <= 0:
            return []
        return [*(a for a in prim if a.kind != "idea"),
                replace(lone, why=f"{lone.why} [{room} of {w.runs} points now, the rest queued first]"),
                *(a for a in prim if a.kind == "idea")]

    def _churned(self, data: CD.Ledger) -> set[tuple[str, int]]:
        """(node, seed) of the idle-fill runs the idle churn cancelled (``run_cancelled`` events: persisted)."""
        try:
            ids = {str(e.get("run_id")) for e in self._store.events("run_cancelled")
                   if str(e.get("reason", "")).startswith(self.CHURN_REASON)} if self._store is not None else set()
        except Exception:  # noqa: BLE001 - a store without events
            ids = set()
        return {(r.node, r.seed) for r in data.runs if r.id in ids}

    def fit_data(self, data: CD.Ledger) -> CD.Ledger:
        """``data`` without the eval-only runs (``eo_runs``, v3-c5b M1): what every noise fit / pair pool reads."""
        eo = self.eo_runs
        return replace(data, obs=tuple(o for o in data.obs if o.run_id not in eo)) if eo else data

    def _fit(self, data: CD.Ledger, parents: Any = None) -> Any:
        """Noise with ρ from the (child, parent) pairs of ``parents`` (default: every child / parent pair); never
        from an eval-only run (:meth:`fit_data`)."""
        data = self.fit_data(data)
        try:
            return self.fit_at(data, data.change_parents() if parents is None else parents)[0]
        except (ValueError, ZeroDivisionError):
            return None

    def classify(self, near: frozenset[str], near_pair: Callable[[str, str], bool]) -> None:
        """The change classes (tick: ``Tick._classify`` from the adapter's configs and diffs): ``near`` = nodes one
        config key from their parent with no code change; ``near_pair(c, i)`` = c and i are one config key apart."""
        self.near, self.near_pair = frozenset(near), near_pair

    def noise(self, data: CD.Ledger) -> Any:
        """Panel / axis / composite noise over every epoch (only the legacy replay of not-yet-frozen historical
        blocks judges with it); None without data."""
        return self._fit(data) if self.fit_data(data).obs else None

    def derive(self, data: CD.Ledger, blocks: Sequence[Any], noise: Any) -> Mapping[str, Any]:
        if noise is None and not any(b.verdicts is not None for b in blocks if b.pool == CD.TEST):
            return {}
        return self.ST.derive(data, blocks, noise)

    def pair_fit(self, data: CD.Ledger, c: str, i: str, block: Any) -> Any:
        """The noise c vs i is judged with: ``status.pair_fit`` (near only for a one-config-key pair with enough
        dfs, else far; the candidate's own runs and the block's seeds left out)."""
        return self.ST.pair_fit(self.fit_data(data), c, set(getattr(block, "seeds", ())), self.near, self.near_pair(c, i),
                                lambda led, pairs: self._fit(led, pairs))

    def decide(self, data: CD.Ledger, blocks: Sequence[Any], noise: Any, incumbent: str) -> list[tuple[Any, ...]]:
        """The open blocks' verdicts, each pair judged with its own :meth:`pair_fit` (falls back to ``noise``)."""
        if noise is None or not any(b.pool == CD.TEST and b.open for b in blocks):
            return []
        fits: dict[tuple[str, str, str], Any] = {}

        def noise_for(c: str, i: str, b: Any) -> Any:
            if (c, i, b.id) not in fits:
                fits[c, i, b.id] = self.pair_fit(data, c, i, b) or self.ST.PairFit(noise, False, "history")
            return fits[c, i, b.id]
        return self.ST.decide_open(data, blocks, noise, incumbent, noise_for)

    def posterior(self, data: CD.Ledger, blocks: Sequence[Any]) -> Any:
        if not data.obs:
            return None
        import inspect
        kw = {"near": self.near} if "near" in inspect.signature(self.A.Posterior).parameters else {}
        if self.evalonly and "evalonly" in inspect.signature(self.A.Posterior).parameters:   # v3-c5 I5
            kw.update(evalonly=self.evalonly, eo_floor=self.eo_floor, eo_min_df=self.eo_min_df)
        if self.eo_runs and "eo_runs" in inspect.signature(self.A.Posterior).parameters:    # v3-c5b M1
            kw["eo_runs"] = self.eo_runs
        try:
            return self.A.Posterior(data, blocks, **kw)
        except (ValueError, ZeroDivisionError):
            return None

    def rank(self, data: CD.Ledger, blocks: Sequence[Any], statuses: Mapping[str, Any], post: Any,
             free_slots: int, llm_free: int) -> list[Any]:
        self.preempt, self._gains = {}, {}           # this rank's (never a stale tick's) preemption / paired view
        if post is None or not statuses:
            return []
        dropped: list = []
        self._target_mode(data, incumbent_of(statuses))
        if hasattr(self.A, "CLOSED_WAVES"):          # a wave with a recorded result is closed (acquire's wave caps)
            from loop.core import factorial as F
            self.A.CLOSED_WAVES = frozenset(w for w in F.open_waves(data)
                                            if self._state(f"wave:{w}:result") is not None)
        opts = self.A.ledger_options(data, blocks, statuses, incumbent_of(statuses), self.costs, post,
                                     knobs=self.knobs, dropped=dropped)
        if self._floor_seen is None:
            self._floor_seen = {(str(k), n) for k, n in self._state(self.K_FLOOR_SEEN) or []}
        new = [(kind, node, ub) for kind, node, ub in dropped if (kind, node) not in self._floor_seen]
        for kind, node, ub in new:                     # the gap floor (target mode), each candidate logged once
            self._floor_seen.add((kind, node))
            frac = getattr(self.A, "SEED_FLOOR_FRAC", 0.0) if kind == "seed" else self.A.GAP_FLOOR_FRAC
            self.notes.append(f"gap_floor: dropped {kind} {node} ub={ub:+.5f} < {frac:g} x gap")
        if new:
            self._put(self.K_FLOOR_SEEN, sorted([k, n] for k, n in self._floor_seen))
        if self.held:                                # loop.databank: a node waits for its dataset (prepare)
            opts = [o for o in opts if o.node not in self.held]
        killed = self._killed()                      # v3-c7i I9: screening ended early: no more seeds / screens
        if killed:
            opts = [o for o in opts if not (o.kind in ("seed", "screen") and o.node in killed)]
        self._split(data, statuses, post, free_slots)   # v3-c7i I2: the LLM-idea units' GPU share (soft cap)
        if self.inc_gains:                           # v3 review K2: a wave on a carrier below the incumbent is
            from dataclasses import replace as _rep     # priced over the incumbent (its x* starts that far behind)
            opts = [_rep(o, m=o.m + self.inc_gains[o.node]) if o.kind == "wave" and o.node in self.inc_gains
                    and self.inc_gains[o.node] < 0 else o for o in opts]
        self.last_opts = list(opts)
        if self.test_backoff:                        # v3-c5b S3: a control-unscorable close backs its candidates off
            opts = [o for o in opts if not (o.kind == "test" and o.node in self.test_backoff)]
        if self.prior is not None:                   # v3-c6i: the pairwise pre-screen orders the screen options
            from loop.core import prescreen as PS
            try:
                opts = PS.nudge(opts, *self.prior())
            except Exception:  # noqa: BLE001 - the prior never stops a rank
                pass
        kw = {}
        if "relax_caps" in inspect.signature(self.A.rank).parameters:    # open relax waves count against the cap
            from loop.core import factorial as F
            pre = sum(1 for w in F.open_waves(data) if (self._state(f"wave:{w}") or {}).get("pre")
                      and self._state(f"wave:{w}:result") is None)
            kw["relax_caps"] = {"wave": max(0, getattr(self.A, "RELAX_MAX_WAVES", 1) - pre)}

        prefer: set[str] = set()
        if self.alloc_cfg is not None and getattr(self.alloc_cfg, "priority_categories", ()):
            from loop.core import alloc as _AL                              # 11-09: bold screens before the others
            prefer = _AL.priority_nodes(self.alloc_cfg, data, statuses)

        # 11-10: no second test opens on partial slots while a TEST block is open (Tick._launch submits owed runs first)
        # (review: and at most one test opened on partial slots at a time -- any open TEST block turns it off; a test
        # that fits whole still opens)
        owed_test = any(b.pool == CD.TEST and b.open for b in blocks)
        # 11-10 (16:38Z: a re-root port, paired gain -.0008, opened the test over a node, +.0013 / P(promote) .84, on a
        # composite-EI tie; its block then held off a node's for hours): the tests are ordered by the promotion bar's
        # belief, and an open block none of whose runs started gives way to a clearly stronger candidate
        test_order, self.preempt = self._test_order(data, opts, incumbent_of(statuses)), {}
        if owed_test and test_order:
            self.preempt = self._preemptible(data, blocks, opts, test_order)
            owed_test = not self.preempt

        def ranked(opts: Sequence[Any], relax_wave: int | None = None, slots: int | None = None) -> list[Any]:
            k = dict(kw)
            if relax_wave is not None and "relax_caps" in k:   # stack priority: the relax-pass wave cap it may use
                k["relax_caps"] = {**k["relax_caps"], "wave": relax_wave}
            b = self.A.budgets(data, statuses, free_slots, self.costs)
            if self.group_soft and "groups" in inspect.signature(self.A.rank).parameters:
                b = {**b, "soft": {**b.get("soft", {}), **self.group_soft}}
                k["groups"] = self.groups
            if self.reserve and "reserve" in inspect.signature(self.A.rank).parameters:
                k["reserve"] = self.reserve
            if "priority" in inspect.signature(self.A.rank).parameters and hasattr(self.A, "program_nodes"):
                k["priority"] = self.A.program_nodes(data, statuses)      # 10-21: operator-program arms first
            if prefer and "prefer" in inspect.signature(self.A.rank).parameters:
                k["prefer"] = prefer
            if owed_test and "partial_test" in inspect.signature(self.A.rank).parameters:
                k["partial_test"] = False                 # one half-launched test at a time (acquire.rank)
            elif self.preempt and "partial_test" in inspect.signature(self.A.rank).parameters:
                k["partial_test"] = frozenset(v[1] for v in self.preempt.values())   # the challenger's seat only
            if test_order and "test_order" in inspect.signature(self.A.rank).parameters:
                k["test_order"] = test_order              # TEST slots: best paired gain first (acquire.rank)
            return self.A.rank(opts, free_slots if slots is None else slots, llm_free, **self.harm,
                               **{"floor": self.A.VALUE_FLOOR, "wave_floor": self.A.WAVE_FLOOR, **self.floors},
                               **b, tags=self.tags, tag_left=self.tag_left, **k)
        out = self._stack_first(opts, ranked(opts), ranked, free_slots)
        if self.preempt and not any(a.kind == "test" and a.node in {v[1] for v in self.preempt.values()} for a in out):
            # review: the partial seat re-opened for the CHALLENGER only -- not placed (a floor, a cap, no fit), the
            # holder keeps its block and no other test takes a second partial seat
            self.preempt, owed_test = {}, True
            out = self._stack_first(opts, ranked(opts), ranked, free_slots)
        left = free_slots - sum(a.runs for a in out if a.kind != "idea")
        if left > 0 and isinstance(post, self.A.Posterior) and not any(o.kind == "rebase" for o in opts):
            spent = {}                                   # what rank's actions took of each branch's cap
            for a in out:
                if (t := self.tags.get(a.node or "")) is not None and a.kind != "test":
                    spent[t] = spent.get(t, 0.0) + a.gpu_h
            out += self.A.idle_fill(data, blocks, statuses, incumbent_of(statuses), post, left,
                                    {a.node for a in out if a.node} | set(self.held) | killed, self.costs, tags=self.tags,
                                    tag_left={t: x - spent.get(t, 0.0) for t, x in self.tag_left.items()},
                                    churned=self._churned(data),
                                    **({"prefer": prefer} if prefer and "prefer" in
                                       inspect.signature(self.A.idle_fill).parameters else {}))
        return out

    # -- 11-10: which candidate a TEST slot goes to --------------------------------------------------------------
    preempt: Mapping[str, tuple[str, str, bool]] = {}   # block id -> (its candidate, the stronger one, started)
    _gains: Mapping[str, tuple[float, float, int]] = {}   # this rank's paired view (node -> gain, var, n)
    PREEMPT_P = 0.8                      # P(challenger's paired gain > the holder's) a withdrawal needs
    MAX_STARTED_TESTS = 2                # a challenger may open beside at most 1 started TEST block

    def _paired(self, data: CD.Ledger, inc: str | None) -> Mapping[str, tuple[float, float, int]]:
        """node -> (EB-shrunk paired gain over ``inc``, its variance, paired seeds): ``bar.paired``, the belief the
        promotion bar (P(promote)) reads; {} without one."""
        from loop.core import bar as BR
        if not inc or not hasattr(data, "in_epoch"):
            return {}
        eb = int(getattr(getattr(self._cfg, "labless", None), "eb_min_nodes", BR.EB_MIN_NODES))
        try:
            pv = BR.paired(data, data.port(inc) or inc, eb_min_nodes=eb)
        except Exception:  # noqa: BLE001 - no paired view: the value order stands
            return {}
        return dict(pv.gains) if pv is not None else {}

    def _test_order(self, data: CD.Ledger, opts: Sequence[Any], inc: str | None) -> dict[str, float]:
        """``acquire.rank``'s ``test_order``: every TEST option's EB-shrunk paired gain over the incumbent (its own
        composite belief when it does not pair); kept for :meth:`_preemptible` in ``self._gains``."""
        tests = [o for o in opts if o.kind == "test" and o.node]
        self._gains = self._paired(data, inc) if tests else {}
        if not self._gains:
            return {}
        self._gains = {n: g for n, g in self._gains.items() if math.isfinite(g[0]) and math.isfinite(g[1])}
        if not self._gains:
            return {}
        out = {o.node: float(self._gains[o.node][0]) if o.node in self._gains else float(o.m) for o in tests}
        return {n: m for n, m in out.items() if math.isfinite(m)}

    def _preemptible(self, data: CD.Ledger, blocks: Sequence[Any], opts: Sequence[Any],
                     order: Mapping[str, float]) -> dict[str, tuple[str, str, bool]]:
        """{block: (holder, challenger, started)} when the best TEST option outside the open TEST blocks is clearly
        stronger than every holder on the paired view (its EB gain > 0 and P(its gain > the holder's) >=
        ``PREEMPT_P``) and at most ``MAX_STARTED_TESTS`` - 1 of them started (a run on any of its seeds). The
        challenger then gets the partial-test seat; ``Tick._preempt_tests`` withdraws the UNSTARTED blocks (a seat
        reservation, nothing measured) once rank actually placed its test -- a started block keeps running beside
        it (at most ``MAX_STARTED_TESTS`` tests then, and ``Tick._launch`` owes the stronger one's runs first)."""
        open_t = [b for b in blocks if b.pool == CD.TEST and b.open]
        if not open_t:
            return {}
        used = {r.seed for r in data.runs}
        started = {b.id: bool(used.intersection(b.seeds)) for b in open_t}
        if sum(started.values()) >= self.MAX_STARTED_TESTS:
            return {}
        held = {c for b in open_t for c in b.node_ids[:-1]}
        cands = [n for n in order if n not in held and n in self._gains]
        if not cands:
            return {}
        ch = max(cands, key=lambda n: order[n])
        if any(order[n] > order[ch] for n in order if n not in held and n not in self._gains):
            return {}                                 # an unpaired test outranks it: no clear challenger
        mc, vc, _ = self._gains[ch]
        if mc <= 0:
            return {}
        out = {}
        for b in open_t:
            for h in b.node_ids[:-1]:
                if h not in self._gains:                  # a holder the paired view cannot judge keeps its seat
                    return {}
                mh, vh, _ = self._gains[h]
                sd = math.sqrt(max(vc + vh, 0.0))
                p = 0.5 * (1 + math.erf((mc - mh) / (sd * math.sqrt(2)))) if sd > 0 else float(mc > mh)
                if p < self.PREEMPT_P:
                    return {}
            out[b.id] = (",".join(b.node_ids[:-1]), ch, started[b.id])
        return out

    # -- v3-c7i: budget allocation (loop.core.alloc) -------------------------------------------------------------
    alloc_cfg: Any = None
    _cfg: Any = None
    groups: Mapping[str, str] = {}           # node -> llm / code / other (its unit's group)
    group_soft: Mapping[str, int] = {}       # "group:llm" -> the most LLM-unit runs to launch now (rank's soft cap)
    split: Mapping[str, float] = {}          # the GPU split behind it (alloc.gpu_split + slots)
    last_opts: list[Any] = []                # this tick's options (the backlog age-out reads their screen values)
    inc_gains: Mapping[str, float] = {}      # node -> its shrunk paired gain over the incumbent (bar.paired)
    reserve: tuple[int, frozenset[str]] | None = None   # v3 review K2: (slots, discovery nodes) rank holds for them
    K_SPLIT = "alloc.gpu_split"

    def _killed(self) -> frozenset[str]:
        from loop.core import alloc as AL
        ac = self.alloc_cfg
        if ac is None or not (ac.enabled and ac.early_kill):
            return frozenset()
        return frozenset(self._state(AL.K_EARLY) or {})

    def _split(self, data: CD.Ledger, statuses: Mapping[str, Any], post: Any, free_slots: int) -> None:
        """``groups`` / ``group_soft`` for this tick's rank: LLM-idea units' runs (in flight + launched) may hold at
        most ``alloc.gpu_split`` share x the GPU slots; logged (``gpu_split``) when it changes."""
        from loop.core import alloc as AL
        self.groups, self.group_soft, self.split, self.inc_gains, self.reserve = {}, {}, {}, {}, None
        ac = self.alloc_cfg
        if ac is None or not ac.enabled or post is None:
            return
        from types import SimpleNamespace
        inc = incumbent_of(statuses) if statuses else None
        tab, pooled, sp = AL.table(SimpleNamespace(cfg=self._cfg), data, AL.post_beliefs(post), incumbent=inc)
        self.inc_gains = dict(AL.inc_gains(data, inc) or {}) if inc else {}
        self.groups = AL.groups(data)
        live = [r for r in data.runs if r.status in self.A.LIVE and r.stage not in AL.TEST_STAGES]
        llm_live = sum(1 for r in live if self.groups.get(r.node) == "llm")
        total = free_slots + len(live)
        cap = int(sp["share"] * total)
        self.group_soft = {"group:llm": max(0, cap - llm_live)}
        # v3 review K2: a HARD reservation for seed 1-2 screens of NEW LLM-idea nodes on the incumbent (discovery
        # never starved by waves / repairs / seed extensions)
        frac = float(getattr(ac, "discovery_reserve", 0.25) or 0.0)
        done: dict[str, int] = {}
        for o in data.obs:
            if o.stage not in AL.TEST_STAGES:
                done[o.node] = done.get(o.node, 0) + 1
        disc = frozenset(n for n, nd in data.nodes.items() if inc and nd.parent == inc and nd.base == data.epoch
                         and self.groups.get(n) == "llm" and done.get(n, 0) < 2
                         and "opposite_of=" not in (nd.notes or "")      # v3-c8a: a mirror arm is not a new idea
                         and status_name(statuses.get(n, "")) in ("QUEUED", "SCREENING"))
        d_live = sum(1 for r in live if r.node in disc)
        want = math.ceil(frac * total) if frac > 0 else 0
        self.reserve = (max(0, want - d_live), disc) if want else None
        self.split = {"share": round(sp["share"], 4), "llm_rate": float(f"{sp['llm_rate']:.4g}"),
                      "code_rate": float(f"{sp['code_rate']:.4g}"), "slots": total, "llm_cap_runs": cap,
                      "llm_live": llm_live, "ci": [round(sp.get("ci_lo", 0.0), 3), round(sp.get("ci_hi", 0.0), 3)],
                      "point": round(sp.get("point", 0.0), 3), "reserve": max(0, want - d_live),
                      "reserve_slots": want, "discovery_live": d_live, "discovery_nodes": len(disc)}
        key = {k: self.split[k] for k in ("share", "llm_cap_runs")}
        if self._state(self.K_SPLIT) != key:
            self._put(self.K_SPLIT, key)
            if self._store is not None:
                self._store.log("gpu_split", **self.split)

    # -- factorial knob banks (docs §5 "Knob sources") ----------------------------------------------------------
    knobs: Mapping[str, list[Any]] = {}
    held: frozenset[str] = frozenset()       # nodes waiting for their data-bank dataset (prepare -> rank)
    test_backoff: frozenset[str] = frozenset()   # S3: candidates of a control-unscorable close, inside test.retest_backoff_h
    prior: Any = None                        # v3-c6i: () -> (pre-screen book, weight); Ideation._ps_tick sets it
    tags: Mapping[str, str] = {}             # node -> its open fork-and-flush branch (prepare -> rank: core.fnf caps)
    tag_left: Mapping[str, float] = {}       # branch -> GPU-h left under its cap
    K_BANK = "factorial.bank:"          # + carrier: the one-time retro seed {knobs, lines, tau2, used, v}
    K_NEXT = "factorial.next:"          # + carrier: the next ranges a closed wave left (the edge rule)
    K_USED = "factorial.used:"          # + carrier: signatures of the knobs its waves consumed (BankKnob.signature)
    RETRO_V = 2                         # the retro seed's version: v2 = trimmed to factorial.RETRO_K knobs |z| >= RETRO_Z

    def prepare(self, tick: "Tick", data: CD.Ledger, statuses: Mapping[str, Any], post: Any) -> list[str]:
        """This tick's knob banks (carrier -> ``factorial.BankKnob`` list, best first) for ``rank``'s wave options, and
        the nodes held for their dataset (``loop.databank.gate``). The incumbent's carrier: its retro wave first (the
        retro seed trimmed to <= 4 knobs at |z| >= 3, a wave of its own), then positive stacking candidates on it
        (ports included), the ranges a closed wave left, the diagnostics flags' registered knobs and the plateau
        review's knobs; a positive live node of the current base gets ONE wave over its researcher-declared knobs.
        A knob a wave consumed on a carrier is never offered there again (``K_USED``, written with the wave)."""
        from loop.core import factorial as F
        from loop.core import fnf as FN
        self.knobs, acts = {}, []
        self.held = self._held(tick, statuses, acts) | self._split_held(tick, data, statuses, acts)
        st = tick.store.get_state(FN.K_FNF)
        self.tags, self.tag_left = FN.budgets(st, data)
        if post is None or not statuses:
            return acts
        car = data.port(incumbent_of(statuses)) or incumbent_of(statuses)
        if st and st.get("open") and FN.track_incumbent(st := dict(st), car, tick.clock()):
            tick.store.set_state(FN.K_FNF, st)              # the open round follows an incumbent switch
            acts.append(f"fnf round {st['open']['id']}: incumbent now {car}")
        if car in post.by and car in data.nodes and data.nodes[car].base == post.epoch:
            if bank := self._carrier_bank(tick, data, post, car, acts, statuses):
                self.knobs[car] = bank
        for nid in post.est:
            node = data.nodes[nid]
            if node.base != post.epoch or nid == car or F.is_point(node.notes) \
                    or status_name(statuses.get(nid, "")) != "SCREENING":
                continue
            plan = tick.store.get_state(f"ideation:{nid}") or {}
            if " brief=program" in f" {node.notes}":    # a program: its <= 5-trial dose ladder (core.program)
                continue
            if plan.get("knobs") and post.belief(nid)[0] > 0:
                t2 = F.factorial_tau2(post.prior, nid)
                bank = F.unused(self._valid(tick, nid, [F.BankKnob(k, 0.0, t2, "declared")
                                                        for k in F.declared(plan["knobs"])],
                                            self._config(tick, nid), acts),
                                tick.store.get_state(self.K_USED + nid) or ())
                if bank:
                    self.knobs[nid] = bank
        if getattr(getattr(tick.ctx.cfg, "factorial", None), "prereq_ports", True):
            self.prereqs(tick, data, statuses, post, acts)
        return acts

    def _held(self, tick: "Tick", statuses: Mapping[str, Any], acts: list[str]) -> frozenset[str]:
        """QUEUED nodes whose ``data.dataset_dir`` is a data-bank dataset not ready yet: they wait (no screen, no
        launch); one whose build failed or that nobody requested is voided with the reason (``loop.databank.gate``)."""
        from loop import databank as DB
        from loop.contracts import NodeStatus
        if not getattr(getattr(tick.ctx.cfg, "databank", None), "enabled", False):
            return frozenset()
        store = tick.store
        queued = [n for nid, s in statuses.items() if status_name(s) == "QUEUED" and (n := store.get_node(nid))]

        def void(n: Any, why: str) -> None:
            store.update_node(n.id, status=NodeStatus.VOID, reject_reason=why)
            store.log("databank_void", node_id=n.id, reason=why)
            acts.append(f"void {n.id}: {why}")
        cache = self.__dict__.setdefault("_ds_cache", {})
        return frozenset(DB.gate(tick.ctx, store, queued, void, cache))

    def _split_held(self, tick: "Tick", data: CD.Ledger, statuses: Mapping[str, Any], acts: list[str]) -> frozenset[str]:
        """v3-c8a B3: QUEUED mixed-change nodes wait while their per-group parts screen (``loop.core.mixed.held``:
        the parts are built on first sight). Never raises (a failure holds nothing)."""
        from loop.core import mixed as MX
        out: frozenset = frozenset()
        ide = getattr(getattr(tick, "intake", None), "__self__", None)
        if hasattr(ide, "knob_holds"):               # v3 review M3: bold nodes waiting for a knob verdict (fails closed)
            out = ide.knob_holds(statuses, tick.clock(), acts)
        try:
            return out | MX.held(tick.ctx, tick.store, tick.adapter, data, statuses, tick.clock(), acts)
        except Exception as exc:  # noqa: BLE001 - the split never stops the tick
            tick.store.log("mixed_split_error", error=repr(exc)[:500])
            return out

    def _carrier_bank(self, tick: "Tick", data: CD.Ledger, post: Any, car: str, acts: list[str],
                      statuses: Mapping[str, Any] | None = None) -> list[Any]:
        from loop import diagnostics as DG
        from loop.core import factorial as F
        store = tick.store
        cfg = self._config(tick, car)
        seed = store.get_state(self.K_BANK + car)
        if seed is None or (not seed.get("used") and seed.get("v") != self.RETRO_V):
            seed = self._retro_seed(tick, data, post, car, cfg)
            store.set_state(self.K_BANK + car, seed)
            store.log("knob_bank", carrier=car, knobs=[b["knob"]["key"] for b in seed["knobs"]], tau2=seed["tau2"],
                      lines=seed["lines"])
            acts.append(f"knob bank {car}: {len(seed['knobs'])} retro knobs "
                        f"({', '.join(b['knob']['key'] for b in seed['knobs'])}) from {seed['effects']} historical "
                        f"main effects (derivation: event knob_bank)")
        t2 = float(seed["tau2"]) or F.factorial_tau2(post.prior, car)
        used = store.get_state(self.K_USED + car) or ()
        if not seed.get("used") and (retro := F.unused(self._valid(
                tick, car, [F.BankKnob.from_json(b) for b in seed["knobs"]], cfg, acts), used)):
            return retro                                   # the retro wave: its own <= 4 knobs, 8 points
        from loop.core.model import COMPOSITE
        fcfg = getattr(tick.ctx.cfg, "factorial", None)
        min_n = int(getattr(fcfg, "stack_min_seeds", 2))
        leg_z = float(getattr(fcfg, "stack_leg_z", 1.0))
        leg_z_min = min(leg_z, float(getattr(fcfg, "stack_leg_z_min", 0.75)))
        sts = statuses if statuses is not None else (getattr(tick, "statuses", None) or {})
        cands: list[tuple[float, Any]] = []
        for nid, n in data.nodes.items():                  # stacking: positive candidates on the carrier, ports too
            if n.parent != car or n.base != post.epoch or F.is_point(n.notes) or "for_wave=" in n.notes \
                    or "port_failed" in n.notes or n.notes.startswith("factorial_of="):
                continue
            if self._dead_leg(tick, nid, sts):            # v3-c6a: a FAILED test / a VETOED node is never a leg
                continue
            src = nid if nid in post.est else n.port_of if n.port_of in post.est else None
            if src is None or (b := post.belief(src))[0] <= 0:
                continue
            est = (post.est.get(src) or {}).get(COMPOSITE) if isinstance(post.est, Mapping) else None
            if est is not None and int(getattr(est, "n", min_n)) < min_n:   # a leg: positive on >= min_n seeds
                continue
            z = b[0] / math.sqrt(b[1]) if b[1] > 0 else math.inf
            cands.append((z, F.BankKnob(F.Knob(nid, 0, 1, "diff"), b[0], b[1], "stacking")))
        # S1: a leg's own posterior mean >= stack_leg_z x its sd; v3-c6a: fewer than 2 such legs -> relaxed to
        # stack_leg_z_min; still fewer than 2: no stacking legs (a 1-leg "stack" is no stack)
        legs = [kb for z, kb in cands if z >= leg_z]
        if len(legs) < 2:
            legs = [kb for z, kb in cands if z >= leg_z_min]
            if len(legs) >= 2 and leg_z_min < leg_z:
                acts.append(f"knob bank {car}: < 2 stacking legs at z >= {leg_z:g}; relaxed to {leg_z_min:g} "
                            f"({', '.join(kb.knob.key for kb in legs)})")
        if len(legs) < 2:
            legs = []
        bank = self._independent(data, car, legs)
        bank += [F.BankKnob(F.Knob(**k), 0.0, t2, "next") for k in store.get_state(self.K_NEXT + car) or ()]
        try:
            flags = [f["flag"] for f in DG.bottlenecks(store, car)
                     or DG.bottlenecks(store, data.nodes[car].port_of or car)]
        except Exception as exc:  # noqa: BLE001 - no diagnostics flags is not a reason to stop the tick
            store.log("tick_call_error", key=f"bottlenecks:{car}", error=repr(exc)[:500],
                      traceback=traceback.format_exc()[-3000:])
            acts.append(f"knob bank {car}: diagnostics flags unreadable ({exc!r}); flag knobs left out"[:300])
            flags = []
        bank += self._extrapolate_knobs(tick, data, post, car, cfg, t2, acts)   # v3-c7a A5: follow the gradient
        registry = getattr(getattr(tick.ctx.cfg, "factorial", None), "flag_knobs", {}) or {}
        for kb in F.flagged(flags, registry, cfg):
            if not cfg or F._same(cfg.get(kb.key), kb.lo):
                bank.append(F.BankKnob(kb, 0.0, t2, "flag"))
        from loop.core.ideation import K_PLATEAU          # the plateau review's dose questions (its hypotheses' knobs)
        for h in (store.get_state(K_PLATEAU) or {}).get("hypotheses") or ():
            for kb in F.declared(h.get("knobs") if isinstance(h, Mapping) else ()):
                if not cfg or kb.key not in cfg or F._same(cfg.get(kb.key), kb.lo):
                    bank.append(F.BankKnob(kb, 0.0, t2, "plateau"))
        return F.unused(self._valid(tick, car, bank, cfg, acts), used)

    def _one_key_move(self, tick: "Tick", nid: str, parent: str, cache: dict[str, bool]
                      ) -> tuple[str, Any, Any] | None:
        """(key, parent value, node value) when ``nid`` is ONE config key from ``parent`` and no other file."""
        if not tick._one_key(nid, parent, cache):
            return None
        a, b = self._config(tick, parent), self._config(tick, nid)
        keys = [k for k in set(a) | set(b) if a.get(k) != b.get(k)]
        return (keys[0], a.get(keys[0]), b.get(keys[0])) if len(keys) == 1 else None

    def _extrapolate_knobs(self, tick: "Tick", data: CD.Ledger, post: Any, car: str, cfg: Mapping[str, Any],
                           t2: float, acts: list[str]) -> list[Any]:
        """FOLLOW THE GRADIENT on the v3 carrier (v3-c7a A5; ``extrapolate.enabled``): every WINNING one-config-key
        move measured on the carrier -- its own children, the carrier's own last step, and (old epoch, via the port)
        the children / own step of the node it is the port of -- whose shrunk own increment is > 0 at z >=
        ``extrapolate.min_z`` becomes a knob carrier value -> one step past the winner (b + (b - a) in the key's
        scale, ``loop.extrapolate.meta``, inside its hard bounds), strongest first, one per key, at most
        ``extrapolate.max_inflight``. An old-epoch move transfers only while the carrier still holds its start (the
        knob is then the move itself, a -> b: its port) or its end value (one step past it); else it was a
        different move. The state ``extrapolate`` (``v3``) re-arms on an incumbent or
        epoch change (``extrapolate_rearmed``) and is refreshed every pass (10-14: last written 5 days earlier,
        every entry still on a node)."""
        from loop import extrapolate as EX
        from loop.core import factorial as F
        ex = getattr(tick.ctx.cfg, "extrapolate", None)
        if ex is None or not getattr(ex, "enabled", False) or not cfg:
            return []
        store, now = tick.store, tick.clock()
        bk = EX.book(store)
        v3 = dict(bk.get("v3") or {})
        if v3.get("incumbent") != car or v3.get("epoch") != data.epoch:
            store.log("extrapolate_rearmed", incumbent=car, epoch=data.epoch, previous=v3.get("incumbent", ""),
                      previous_epoch=v3.get("epoch", ""))
            acts.append(f"extrapolate: re-armed on {car} (epoch {data.epoch[:7]}; was {v3.get('incumbent') or 'none'})")
            v3 = {"incumbent": car, "epoch": data.epoch, "armed_ts": now, "rearmed": int(v3.get("rearmed", 0)) + 1}
        orig = data.nodes[car].port_of if car in data.nodes else ""
        heads = [h for h in (car, orig) if h and h in data.nodes]
        cands: list[tuple[str, str]] = []               # (winner, the node it moved from)
        for nid, n in data.nodes.items():
            if n.parent in heads and nid not in heads and not F.is_point(n.notes) and "for_wave=" not in n.notes \
                    and not n.notes.startswith("factorial_of=") and not n.second_parent:
                cands.append((nid, n.parent))
        for h in heads:                                  # the carrier's own winning step(s)
            if data.nodes[h].parent is not None and not data.nodes[h].port_of:
                cands.append((h, data.nodes[h].parent))
        cache = dict(store.get_state(tick.K_CHANGE) or {})
        n0, min_z, out = len(cache), float(getattr(ex, "min_z", 1.0)), {}
        for nid, par in cands:
            if nid not in getattr(post, "est", {}) or par is None:
                continue
            try:
                m, v = post.belief(nid)
            except (KeyError, ValueError):
                continue
            sd = math.sqrt(max(v, 0.0))
            if m <= 0 or (sd > 0 and m / sd < min_z):
                continue
            mv = self._one_key_move(tick, nid, par, cache)
            if mv is None:
                continue
            key, a, b = mv
            c = cfg.get(key)
            if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in (a, b, c)) or a == b:
                continue
            if data.nodes[nid].base != data.epoch and not (F._same(c, a) or F._same(c, b)):
                continue                                 # the carrier moved the key elsewhere: not the same move
            meta = EX.meta(tick.ctx, key, b, [a, c])
            if meta is None or (meta["scale"] == "log" and min(a, b) <= 0):
                continue
            old = data.nodes[nid].base != data.epoch
            # an old-epoch winner the carrier has not taken is first PORTED as its own move (a -> b on the carrier);
            # a move already on the carrier (or measured on it) continues one step past the winner
            hi = b if old and F._same(c, a) else EX.from_u(meta, EX.to_u(meta, b) + (EX.to_u(meta, b) -
                                                                                      EX.to_u(meta, a)), c)
            if F._same(hi, c) or (F._same(hi, b) and not old) or not EX.inside(tick.ctx, key, hi, b, [a, c]):
                continue
            z = m / sd if sd > 0 else math.inf
            why = (f"{nid} on {par}: {key} {a}->{b} {m:+.4f} (z {z:.1f}"
                   + (", old epoch via the port" if data.nodes[nid].base != data.epoch else "") + f"); next {c}->{hi}")
            kb = F.BankKnob(F.Knob(key, c, hi, "log" if meta["scale"] == "log" else "lin"), m, t2, "extrapolate", why,
                            z if math.isfinite(z) else 0.0)
            if key not in out or z > out[key][0]:
                out[key] = (z, kb)
        if len(cache) != n0:
            store.set_state(tick.K_CHANGE, cache)
        knobs = [kb for _, kb in sorted(out.values(), key=lambda t: -t[0])][:max(0, int(getattr(ex, "max_inflight", 4)))]
        v3.update(ts=now, knobs=[{"key": k.knob.key, "lo": k.knob.lo, "hi": k.knob.hi, "why": k.why} for k in knobs])
        store.set_state(EX.K_EXT, {**bk, "v3": v3})
        return knobs

    @staticmethod
    def _dead_leg(tick: "Tick", nid: str, statuses: Mapping[str, Any]) -> bool:
        """``nid`` FAILED its test (derived status) or is VETOED / REJECTED in the store: never a stacking leg."""
        if status_name(statuses.get(nid, "")) in ("FAILED", "VOID", "CRASHED"):
            return True
        try:
            n = tick.store.get_node(nid)
        except Exception:  # noqa: BLE001 - unknown: not dead
            return False
        return n is not None and str(getattr(n.status, "value", n.status)).lower() in ("vetoed", "rejected")

    # ------------------------------------------------------------------ missing prerequisites (v3-c7a A6)
    K_PREREQ = "prereq.ports"     # "<carrier>|<key>" -> {status waiting|open|done|failed, key, carrier, knobs, ports,
    #                               source, port, why, ts}: a knob / port that needs a config key the carrier lacks
    PREREQ_MAX_OPEN = 1           # prerequisite ports in flight at once (the rest wait their turn)
    PREREQ_SCAN = 60              # newest ledger nodes read for a witness of the key when no config_diff names it
    _MISSING = re.compile(r"^the carrier config has no (\S+)$")

    @classmethod
    def note_missing(cls, tick: "Tick", car: str, key: str, *, knob: Any = None, port: str = "",
                     witness: str = "") -> bool:
        """Book a missing prerequisite: ``car``'s config lacks ``key`` that a dropped knob (``knob``) or a failed port
        (``port``, the old node) needs. One record per (carrier, key), the knobs / ports that wait on it gathered;
        a ``failed`` record is not re-opened. True when the record is new."""
        book = dict(tick.store.get_state(cls.K_PREREQ) or {})
        rid = f"{car}|{key}"
        rec = dict(book.get(rid) or {"status": "waiting", "key": key, "carrier": car, "knobs": [], "ports": [],
                                     "witness": [], "ts": tick.clock()})
        new = rid not in book
        if knob is not None and (js := knob.to_json()) not in rec["knobs"]:
            rec["knobs"] = [*rec["knobs"], js]
        if port and port not in rec["ports"]:
            rec["ports"] = [*rec["ports"], port]
        if witness and witness not in rec["witness"]:
            rec["witness"] = [*rec["witness"], witness]
        if rec == book.get(rid):
            return False
        book[rid] = rec
        tick.store.set_state(cls.K_PREREQ, book)
        if new:
            tick.store.log("prereq_needed", carrier=car, key=key, knob=getattr(getattr(knob, "knob", None), "key", ""),
                           port_of=port)
        return new

    @classmethod
    def note_dropped(cls, tick: "Tick", car: str, dropped: Sequence[tuple[Any, str]]) -> None:
        """The drops of ``factorial.check_knobs`` whose reason is a key the carrier lacks: booked as prerequisites."""
        for b, why in dropped:
            if (m := cls._MISSING.match(str(why))) is not None:
                cls.note_missing(tick, car, m.group(1), knob=b)

    @staticmethod
    def conflict_keys(how: str) -> list[str]:
        """The config keys of a graft's ``conflict: <cfg>:<key>, ...`` reason (the adapter's ``merge_config`` clash)."""
        if not str(how).startswith("conflict"):
            return []
        return [t.rsplit(":", 1)[-1].strip() for t in str(how).split(":", 1)[-1].split(",") if t.strip()]

    def _cfg_of(self, tick: "Tick", nid: str) -> dict[str, Any]:
        cache = self.__dict__.setdefault("_cfg_cache", {})
        node = tick.store.get_node(nid)
        c = getattr(node, "commit", "") or ""
        if c not in cache:
            cache[c] = self._config(tick, nid) if c else {}
        return cache[c]

    def introducer(self, tick: "Tick", data: CD.Ledger, key: str, witness: Sequence[str] = ()) -> str | None:
        """The ledger node whose own diff introduced config ``key``: from a node that holds it (the record's
        witnesses, nodes whose ``config_diff:<id>`` names it, else the newest ``PREREQ_SCAN`` nodes), up its parent
        chain to the first node whose parent lacks it. The earliest-made introducer of a non-dead witness wins;
        None when no node holds the key or the base root itself does."""
        store = tick.store
        found: dict[str, float] = {}

        def climb(w: str) -> None:
            cur, seen = w, set()
            while cur in data.nodes and cur not in seen:
                seen.add(cur)
                par = data.nodes[cur].parent
                if par is None:
                    return                               # the root holds it: no change introduced it
                if key not in self._cfg_of(tick, par):
                    found.setdefault(cur, data.nodes[cur].created_ts)
                    return
                cur = par
        cands = [w for w in witness if w in data.nodes]
        cands += [nid for nid in data.nodes if key in (store.get_state(f"config_diff:{nid}") or {})]
        if not cands:
            newest = sorted(data.nodes.values(), key=lambda n: -n.created_ts)[:self.PREREQ_SCAN]
            cands = [n.id for n in newest if key in self._cfg_of(tick, n.id)][:5]
        for w in cands[:8]:
            if key in self._cfg_of(tick, w):
                climb(w)
        live = {n: ts for n, ts in found.items() if not self._dead_leg(tick, n, {})}
        pick = live or found
        return min(pick, key=lambda n: (pick[n], n)) if pick else None

    def prereqs(self, tick: "Tick", data: CD.Ledger, statuses: Mapping[str, Any], post: Any,
                acts: list[str]) -> None:
        """Advance the prerequisite book (:meth:`note_missing`): an ``open`` record's port measured alive (a
        posterior) -> ``done``, its waiting knobs a bank on the port (the dependent wave retries on top of it) and its
        waiting ports re-grafted onto it; a dead / rejected port -> ``failed``. Then at most ``PREREQ_MAX_OPEN`` waiting
        records open: the key's introducer (:meth:`introducer`) ported onto the record's carrier as its own QUEUED
        node (``prereq_for=<key>``; screened like any port), never a wave point."""
        from loop import cutover as CO
        from loop.core import factorial as F
        store = tick.store
        book = dict(store.get_state(self.K_PREREQ) or {})
        if not book:
            return
        changed = False
        for rid, rec in sorted(book.items()):
            if rec.get("status") not in ("open", "done"):
                continue
            port = rec.get("port", "")
            node = store.get_node(port) if port else None
            if rec["status"] == "open":
                if node is None or self._dead_leg(tick, port, statuses):
                    book[rid] = {**rec, "status": "failed", "why": f"prerequisite port {port or '?'} dead"}
                    store.log("prereq_failed", carrier=rec["carrier"], key=rec["key"], port=port,
                              why=book[rid]["why"])
                    acts.append(f"prerequisite {rec['key']} on {rec['carrier']}: port {port} dead -> given up")
                    changed = True
                    continue
                if post is None or port not in post.est:
                    continue                              # not measured yet
                rec = book[rid] = {**rec, "status": "done", "ts": tick.clock()}
                changed = True
                store.log("prereq_done", carrier=rec["carrier"], key=rec["key"], port=port,
                          knobs=[k["knob"]["key"] for k in rec["knobs"]], ports=rec["ports"])
                acts.append(f"prerequisite {rec['key']}: {port} measured; {len(rec['knobs'])} knob(s) / "
                            f"{len(rec['ports'])} port(s) retried on top of it")
                for old in rec["ports"]:
                    try:
                        p = CO.port_node(store, tick.adapter, data, old, tick.clock(), onto=port,
                                         tag=f"after_prereq={port}")
                        acts.append(f"port {old} onto prerequisite {port}: {p.how}" + (f" -> {p.node}" if p.node
                                                                                       else ""))
                    except Exception as exc:  # noqa: BLE001 - logged; the record stays done
                        store.log("tick_call_error", key=f"prereq_port:{old}~{port}", error=repr(exc)[:300])
            if rec["status"] == "done" and rec["knobs"] and port in data.nodes \
                    and not self._dead_leg(tick, port, statuses) and data.nodes[port].base == getattr(post, "epoch",
                                                                                                      None):
                bank = F.unused(self._valid(tick, port, [F.BankKnob.from_json(k) for k in rec["knobs"]],
                                            self._config(tick, port), acts),
                                store.get_state(self.K_USED + port) or ())
                if bank:
                    self.knobs[port] = [*self.knobs.get(port, []), *bank]
        n_open = sum(1 for r in book.values() if r.get("status") == "open")
        for rid, rec in sorted(book.items(), key=lambda kv: (float(kv[1].get("ts", 0)), kv[0])):
            if n_open >= self.PREREQ_MAX_OPEN:
                break
            if rec.get("status") != "waiting":
                continue
            changed = True
            car, key = rec["carrier"], rec["key"]
            if key in self._cfg_of(tick, car):
                book[rid] = {**rec, "status": "done", "port": car, "why": "the carrier holds it now"}
                continue
            src = self.introducer(tick, data, key, rec.get("witness", ()))
            if src is None:
                book[rid] = {**rec, "status": "failed", "why": "no ledger node introduced it"}
                store.log("prereq_failed", carrier=car, key=key, why=book[rid]["why"])
                acts.append(f"prerequisite {key} on {car}: no ledger node introduced it -> given up")
                continue
            try:
                p, what = self.prereq_patch(tick, data, post, src, car, key)
            except Exception as exc:  # noqa: BLE001 - the record fails, the tick goes on
                p, what = CO.Ported(src, "", car, "", f"raised {exc!r}"[:200]), ""
            if p.how == "exists" or p.ok:
                book[rid] = {**rec, "status": "open", "source": src, "port": p.node, "ts": tick.clock(), "what": what}
                n_open += 1
                store.log("prereq_port", carrier=car, key=key, source=src, port=p.node, how=p.how, what=what)
                acts.append(f"prerequisite {key} on {car}: {what} of {src} (introduced it) -> {p.node} ({p.how})")
            elif p.how.startswith("refused"):
                book[rid] = {**rec, "status": "failed", "source": src, "why": p.how}
                store.log("prereq_failed", carrier=car, key=key, source=src, why=p.how)
                acts.append(f"prerequisite {key} on {car}: {p.how} -> given up")
            else:
                book[rid] = {**rec, "status": "failed", "source": src, "port": p.node, "why": f"port {p.how}"}
                store.log("prereq_failed", carrier=car, key=key, source=src, why=f"port {p.how}")
                acts.append(f"prerequisite {key} on {car}: port of {src} failed ({p.how}) -> given up")
        if changed:
            store.set_state(self.K_PREREQ, book)

    @staticmethod
    def codefined(key: str, introduced: Sequence[str]) -> list[str]:
        """``key`` plus the keys its introducer added WITH it that it depends on: NEW keys of the same config block
        (``train.schedule_clock`` -> the other new ``train.*`` keys: the schedule clock's), or -- a top-level key -- of
        the same leaf stem; a key the introducer only moved (it existed before: global_views 2 -> 4) is never one."""
        def block(k: str) -> str:
            return k.rsplit(".", 1)[0] if "." in k else ""

        def stem(k: str) -> str:
            return k.rsplit(".", 1)[-1].split("_")[0]
        out = [key]
        for k in sorted(introduced):
            if k != key and block(k) == block(key) and (block(key) or len(stem(key)) >= 4 and stem(k) == stem(key)):
                out.append(k)
        return out

    def prereq_patch(self, tick: "Tick", data: CD.Ledger, post: Any, src: str, car: str, key: str) -> tuple[Any, str]:
        """(Ported, what) of a missing prerequisite (v3 review K3; 10-14: the whole introducer a node -- global_views 4
        + the schedule clock, -.0044 -- was grafted for one key): the missing key and its co-defined keys
        (:meth:`codefined`) as a CONFIG PATCH on the carrier (``adapter.graft_keys``) when the introducer's change is
        config-only; else -- the key may need its code -- the whole introducer, unless its own increment is < 0
        (refused). The node is tagged ``prereq_of=<src> prereq_for=<key>``, never ``port_of``."""
        from loop import cutover as CO
        from loop.contracts import Node as StoreNode, NodeStatus
        store, adapter = tick.store, tick.adapter
        sn, cn = store.get_node(src), store.get_node(car)
        par = store.get_node(sn.parent_id) if sn is not None and sn.parent_id else None
        if sn is None or cn is None or par is None:
            return CO.Ported(src, "", car, "", "no anchor"), ""
        cfg_src, cfg_par = self._cfg_of(tick, src), self._cfg_of(tick, par.id)
        keys = self.codefined(key, [k for k in cfg_src if k not in cfg_par])
        try:
            code = [f for f in adapter.changed_files(par.commit, sn.commit) if f != getattr(adapter, "config_rel", "config.yaml")]
        except Exception:  # noqa: BLE001 - unreadable: not separable
            code = ["?"]
        tag = f"prereq_of={src} prereq_for={key} prereq_keys={','.join(keys)}"
        if not code and hasattr(adapter, "graft_keys"):
            nid = f"{src}.pq@{data.epoch[:7]}"
            nid = nid if store.get_node(nid) is None or store.get_node(nid).parent_id == car else f"{nid}~{car}"
            if (got := store.get_node(nid)) is not None:
                return CO.Ported(src, nid, car, par.id, "exists", got.commit), "key patch"
            sha, how = adapter.graft_keys(sn.commit, keys, cn.commit, f"pq-{src}-{car}-{key}")
            if how != "applied" or not sha:
                return CO.Ported(src, "", car, par.id, f"key patch {how}"), "key patch"
            now = tick.clock()
            store.put_node(StoreNode(
                id=nid, task_id=store.task_id, commit=sha, parent_id=car, strategy=sn.strategy, operator=sn.operator,
                family=sn.family, mechanism_tag=sn.mechanism_tag, diff_hash=f"prereq-{sha[:12]}-{src}",
                status=NodeStatus.QUEUED, hypothesis_id=sn.hypothesis_id, levers=sn.levers, created_ts=now,
                updated_ts=now, notes=f"{tag} port_anchor={par.id}"))
            store.set_state(f"config_diff:{nid}", {k: cfg_src[k] for k in keys})
            return CO.Ported(src, nid, car, par.id, "applied", sha), f"key patch {keys}"
        inc = None
        if post is not None and src in getattr(post, "est", {}):
            try:
                inc = float(post.belief(src)[0])
            except Exception:  # noqa: BLE001 - unknown increment: not refused
                inc = None
        if inc is not None and inc < 0:
            return CO.Ported(src, "", car, par.id, f"refused: {src}'s change is not separable "
                                                   f"({len(code)} code file(s)) and its own increment {inc:+.4f} < 0"), ""
        return CO.port_node(store, adapter, data, src, tick.clock(), onto=car, tag=tag.split(" ", 1)[1],
                            rel="prereq_of"), "whole change"

    K_DROPPED = "factorial.dropped:"    # + carrier: signatures of the knobs check_knobs dropped (logged once)

    def _valid(self, tick: "Tick", car: str, bank: Sequence[Any], cfg: Mapping[str, Any], acts: list[str]
               ) -> list[Any]:
        """``bank`` checked against ``car``'s config (``factorial.check_knobs``: a config key the carrier lacks, a level
        its type can't take or levels that collapse -> dropped; int / bool levels coerced). A drop is logged once
        (``knob_dropped`` + an act line); the rest of the bank goes on (10-13: one absent key voided all of w007)."""
        from loop.core import factorial as F
        kept, dropped = F.check_knobs(bank, cfg)
        if dropped:
            self.note_dropped(tick, car, dropped)        # v3-c7a A6: a key the carrier lacks -> a prerequisite port
            seen = set(tick.store.get_state(self.K_DROPPED + car) or ())
            new = [(b, why) for b, why in dropped if b.signature() not in seen]
            for b, why in new:
                tick.store.log("knob_dropped", carrier=car, key=b.knob.key, lo=b.knob.lo, hi=b.knob.hi,
                               source=b.source, why=why)
                acts.append(f"knob bank {car}: {b.source} knob {b.knob.key} dropped ({why})"[:300])
            if new:
                tick.store.set_state(self.K_DROPPED + car, sorted(seen | {b.signature() for b, _ in new}))
        return kept

    def _independent(self, data: CD.Ledger, car: str, legs: Sequence[Any]) -> list[Any]:
        """The stacking legs of ``car``, one per mechanism (``acquire.mechanisms`` over the legs beyond the carrier's:
        a shared idea leg or a specific family), its best prior first: a wave slot spent on a 2nd leg of one
        mechanism buys nothing ``acquire.stack_mean`` would credit."""
        if len(legs) < 2:
            return list(legs)
        try:
            mech = self.A.mechanisms(data, [b.knob.key for b in legs], ref=car)
        except Exception:  # noqa: BLE001 - no grouping: every leg is its own mechanism
            return list(legs)
        best: dict[str, Any] = {}
        for b in sorted(legs, key=lambda b: (-b.m, b.knob.key)):
            best.setdefault(mech.get(b.knob.key, b.knob.key), b)
        keep = set(map(id, best.values()))
        return [b for b in legs if id(b) in keep]

    @staticmethod
    def _config(tick: "Tick", nid: str) -> dict[str, Any]:
        node = tick.store.get_node(nid)
        try:
            return dict(tick.adapter.config_at(node.commit)) if node is not None and hasattr(tick.adapter,
                                                                                            "config_at") else {}
        except Exception:  # noqa: BLE001 - an unreadable config: config knobs are left out, diff knobs stay
            return {}

    def _retro_seed(self, tick: "Tick", data: CD.Ledger, post: Any, car: str, cfg: Mapping[str, Any]) -> dict:
        """The one-time retro seed of ``car``'s knob bank: core.factorial's retro over the ledger's historical
        factorials (old epoch noise), restricted to the carrier's old lineage (its original, then its ancestors)."""
        from loop.core import factorial as F
        orig = data.nodes[car].port_of or car
        lineage, cur = [], orig
        while cur is not None and cur not in lineage and cur in data.nodes:
            lineage.append(cur)
            cur = data.nodes[cur].parent
        waves = F.hist_waves(tick.store.get_state("legs"), tick.store.get_state("design"))
        base = data.nodes[orig].base
        try:
            nz, wz = post.fit(base)
            eff = F.retro(data, waves, nz, wz)
        except (KeyError, ValueError, ZeroDivisionError):
            eff = []
        tau2 = F.retro_tau2(eff) if eff else 0.0
        w = 1.0 if base == data.epoch else self.A.OLD_EPOCH_W         # older-base evidence: a discounted prior
        bank, lines = F.retro_bank(eff, lineage, cfg, tau2, w=w) if eff else ([], [])
        keep = F.retro_trim(bank)
        lines = [f"retro over {len(waves)} historical waves, {len(eff)} main effects, factorial tau "
                 f"{tau2 ** .5:.4f}; evidence weight {w}; lineage {' > '.join(lineage)}; carrier config "
                 f"{'read' if cfg else 'UNREADABLE (config knobs left out)'}"] + lines + [
            f"retro wave: {len(keep)} of {len(bank)} knobs (|z| >= {F.RETRO_Z:g}, the {F.RETRO_K} largest): "
            + (", ".join(f"{b.knob.key} (z {b.z:.1f})" for b in keep) or "none")
            + "".join(f"; {b.knob.key} (z {b.z:.1f}) left out" for b in bank if b not in keep)]
        return {"knobs": [b.to_json() for b in keep], "lines": lines, "tau2": tau2, "effects": len(eff),
                "used": False, "v": self.RETRO_V}

    def open_test(self, node: str, incumbent: str, data: CD.Ledger, blocks: Sequence[Any], now: float,
                  cfg: Any) -> Any:
        if any(b.pool == CD.TEST and b.open and node in b.node_ids for b in blocks):
            return None
        return self.S.SeedBlock(f"test-{node}-{int(now)}", CD.TEST, self.draw(self.costs.k_test, data, blocks, cfg),
                                (node, incumbent), f"test {node}", now)

    def draw(self, k: int, data: CD.Ledger, blocks: Sequence[Any], cfg: Any) -> tuple[int, ...]:
        return self.S.draw_test(k, {r.seed for r in data.runs}, blocks, int(cfg.seeds.fresh_base))

    def check_submit(self, run: Run, blocks: Sequence[Any]) -> None:
        self.S.check_submit(run.node_id, run.seed, blocks)

    def migrate(self, store: Any, data: CD.Ledger) -> int:
        return self.S.migrate(store, data)

    def _select(self, seed: int, blocks: Sequence[Any]) -> bool:
        return self.S.pool_of(seed, blocks) == CD.SELECT

    def screen_specs(self, node: str, parent: str | None, k: int, data: CD.Ledger, blocks: Sequence[Any],
                     cfg: Any) -> list[RunSpec]:
        """``k`` SCREEN runs of ``node``: its parent's done SELECT seeds it lacks first (CRN), then the lowest SELECT
        seeds below ``fresh_base`` it has not run. THE seed cap's chokepoint (every seed / screen / fill / port screen
        comes through here): never more than the cap (``acquire.seed_cap``: ``Costs.max_seeds`` within
        ``seeds.max_seeds_per_node``) DISTINCT live-or-done screen seeds per node (``acquire.screen_seeds``)."""
        fresh, mine = int(cfg.seeds.fresh_base), {r.seed for r in data.runs if r.node == node}
        sp_cap = int(getattr(getattr(cfg, "seeds", None), "max_seeds_per_node", 0) or 0)
        cap = min(self.costs.max_seeds, sp_cap) if sp_cap > 0 else self.costs.max_seeds
        k = min(k, max(0, cap - len(self.A.screen_seeds(data.runs, node).get(node, ()))))
        n = data.nodes.get(node) if hasattr(data.nodes, "get") else None
        excl = local_seeds(getattr(n, "notes", "") or "")
        crn = sorted({r.seed for r in data.runs if r.node == parent and r.status == "done"} - mine - excl)
        ok = [s for s in crn if self._select(s, blocks)][:k]
        for s in range(1, fresh):
            if len(ok) >= k:
                break
            if s not in mine and s not in ok and s not in excl and self._select(s, blocks):
                ok.append(s)
        out = [RunSpec(node, SEED_STAGE, s) for s in ok]
        if excl and parent:
            # 11-09 review of 182a2ab: a manual run's confirm node screens only on seeds NONE of its source runs used
            # (the root replicate on such a seed is the reference the arm was SELECTED against; same-seed correlation
            # ~.6 keeps the selection noise), so its fresh seeds may be ones the root never ran: the root's control
            # run on each goes with it (CRN pairing; the launch drops a row that already exists)
            live = {r.seed for r in data.runs if r.node == parent and r.status not in ("crash", "void")}
            pn = data.nodes.get(parent) if hasattr(data.nodes, "get") else None
            stage = Stage.BASELINE if pn is None or getattr(pn, "parent", None) is None else Stage.PARENT
            out += [RunSpec(parent, stage, s) for s in ok if s not in live]
        return out

    def runs_for(self, action: Any, data: CD.Ledger, blocks: Sequence[Any], statuses: Mapping[str, Any],
                 cfg: Any) -> list[RunSpec]:
        specs = self._runs_for(action, data, blocks, statuses, cfg)
        cap = (getattr(self, "seed_caps", None) or {}).get(getattr(action, "node", None) or "")
        if cap is None or action.kind not in ("seed", "screen", "fill"):
            return specs
        own = [sp for sp in specs if sp.node == action.node][:max(0, int(cap))]   # a rescue point: <= its rescue seeds
        keep = {sp.seed for sp in own}
        return [sp for sp in specs if sp in own or (sp.node != action.node and sp.seed in keep)]

    def _runs_for(self, action: Any, data: CD.Ledger, blocks: Sequence[Any], statuses: Mapping[str, Any],
                  cfg: Any) -> list[RunSpec]:
        fresh = int(cfg.seeds.fresh_base)
        if action.kind == "seed" and action.node in data.nodes:
            return self.screen_specs(action.node, data.nodes[action.node].parent, 1, data, blocks, cfg)
        if action.kind == "screen" and action.node in data.nodes:
            k = self.costs.idea_runs                     # a partial screen (acquire.rank 4th pass) is fewer runs
            if 0 < int(getattr(action, "runs", 0) or 0) < k:
                k = int(action.runs)
            return self.screen_specs(action.node, data.nodes[action.node].parent, k, data, blocks, cfg)
        if action.kind == "fill" and action.node in data.nodes:     # idle slots (acquire.idle_fill): CRN seeds first
            return self.screen_specs(action.node, data.nodes[action.node].parent, action.runs, data, blocks, cfg)
        if action.kind == "test":
            out = []
            for b in blocks:
                if b.pool == CD.TEST and b.open and action.node in b.node_ids[:-1]:
                    confirmed = {(r.node, r.seed) for r in data.runs if r.stage == TEST_STAGE.value}
                    out += [RunSpec(n, TEST_STAGE if n != b.node_ids[-1] or (n, s) in confirmed  # re-paired control
                                    else TEST_CONTROL_STAGE, s)
                            for s in b.seeds for n in (action.node, b.node_ids[-1])]
            k = int(getattr(action, "runs", 0) or 0)     # a partial test (acquire.rank): only the pairs it was given
            return out[:k // 2 * 2] if 0 < k < len(out) else out
        if action.kind == "wave":                    # a wave (core.factorial): its points at their block seeds
            from loop.core import factorial
            return [RunSpec(n, SEED_STAGE, s) for n, s in factorial.wave_runs(action.node, data)] + [
                RunSpec(n, Stage.PARENT, s) for n, s in factorial.wave_controls(data, action.node)]
        if action.kind == "rebase":
            k = self.costs.rebase_seeds
            seeds = [s for s in range(1, fresh) if self._select(s, blocks)][:k]
            nodes = [n for n in (data.base_root, data.port(action.node)) if n is not None]
            return [RunSpec(n, REBASE_STAGE, s) for n in nodes for s in seeds]
        return []


@dataclass
class TickReport:
    lines: list[str]
    statuses: Mapping[str, Any] = field(default_factory=dict)
    actions: list[Any] = field(default_factory=list)
    launched: list[str] = field(default_factory=list)


Hook = Callable[..., list[str]]


def default_port(tick: "Tick", action: Any, data: CD.Ledger) -> list[str]:
    """The old-epoch node's port on the current base (:meth:`Tick.port_graft`)."""
    return tick.port_graft(action.node, data)


class Tick:
    """One v3 tick per :meth:`step`. ``db_path`` is the store's SQLite file (core.data reads it read-only).
    ``dry``: no data-bank builds, no Labless, no research sessions (shadow mode; pair it with a dry-run adapter)."""

    def __init__(self, ctx: LoopContext, store: Any, adapter: Any, core: Core, *, db_path: str | Path,
                 researcher: Any = None, executor: Callable[..., Any] = ThreadPoolExecutor,
                 clock: Callable[[], float] = time.time, ideate: Hook | None = None, meta: Hook | None = None,
                 intake: Callable[[Any, Any], str] | None = None, port: Hook | None = None,
                 dry: bool = False, port_sessions: bool = True) -> None:
        self.ctx, self.store, self.adapter, self.core, self.clock = ctx, store, adapter, core, clock
        self.db_path, self.dry = Path(db_path), dry
        self.book = R.RunBook(ctx, store, adapter, clock=clock, extra_rungs=(R.ACT_CKPT,), guard=self._guard)
        self.courtesy = R.Courtesy(ctx, store, adapter, self.book, clock)
        self.pool = executor(max_workers=max(1, ctx.cfg.researcher.max_concurrent,
                                             int(getattr(ctx.cfg.ideation, "llm_slots_max", 8))) + 2)
        self.research = R.Research(ctx, store, adapter, researcher, self.pool, clock)
        from loop.breaker import Breaker
        self.breaker = Breaker(ctx, store, code_rev=code_rev, clock=clock)
        self.research.breaker = self.breaker.tripped()     # a trip survives a plain restart (released by check())
        self.selfcheck_hold = ""                            # 11-08: loop.selfcheck failed at startup -> research held
        self.labless = R.IncumbentLabless(ctx, store, self.pool, clock)
        self._spec_labless = getattr(ctx.cfg, "labless", None)   # the configured bar (loop.leader re-bases ctx.cfg's)
        # 11-08 research features (each default off: loop.board / loop.root_audit / scout.v3)
        from loop import board as BD, localruns as LR, root_audit as RA
        self.board = BD.Board(ctx, store, self.pool, clock)
        self.local_runs = LR.LocalRuns(ctx, store, self.pool, clock)
        self.root_audit = RA.RootAudit(ctx, store, lambda: getattr(getattr(self.research, "researcher", None), "llm", None),
                                       self.pool, clock)
        self._scout: tuple[Any, str] | None = None
        if ideate is None or intake is None:
            from loop.core.ideation import Ideation
            ide = Ideation(ctx, store, clock=clock)
            ideate, intake = ideate or ide.ideate, intake or ide.intake
        for h in (ideate, intake):                     # v3-c7i: ideation's intake reads configs (knob keys)
            if (o := getattr(h, "__self__", None)) is not None and getattr(o, "adapter", "-") is None:
                o.adapter = adapter
        self.ideate: Hook = ideate                                                     # step 10
        if meta is None:                                                               # step 11: core.meta
            from loop.core.meta import Meta
            meta = Meta(ctx, store, clock=clock, dry=dry)
        self.meta: Hook = meta
        self.port: Hook = port or default_port                                         # step 12
        self.port_sessions = port_sessions             # a conflicting port may get ONE researcher session
        self._ported: list[tuple[str, str]] = []       # (port node, its parent) made this tick: screened right away
        self.intake = intake
        if hasattr(adapter, "pool_picker"):            # smoke jobs are placed (and counted) like runs
            adapter.pool_picker = self.book.smoke_pool
        if hasattr(adapter, "smoke_capacity"):         # no session starts whose smoke cannot get a slot (10-22)
            self.research.smoke_capacity = adapter.smoke_capacity
        if hasattr(adapter, "smoke_hold_wait"):        # infra re-issues wait out the smoke gate's hold (loop.infra)
            self.research.smoke_hold = adapter.smoke_hold_wait
        self._started, self._swept, self._blocks = False, None, None
        self._migrated: int | None = None              # historical seed blocks written once (core.seeds.migrate)
        self.post: Any = None                          # this tick's core posterior (ideation reads it)
        self.submits = 0                               # submits the guard let through (tests)
        self.data: CD.Ledger | None = None             # this tick's ledger view (the heartbeat reads it)
        self.statuses: Mapping[str, Any] = {}
        self.phase = ""                                # where the tick is (a tick_error names it)
        self._tick_ts: float | None = None             # this tick's clock
        self._doing: str | None = None                 # the call site in progress (an uncaught error quarantines it)
        self.disk: dict[str, float] = {}               # filesystem -> GB free at the last disk check
        if hasattr(adapter, "resume_ok"):              # TEST-stage runs restart from step 0 when preempted
            adapter.no_resume = lambda run: run.stage in TEST_STAGES and run.id not in self.book.spilled()
        from loop.core.pairs import Planner            # v3-c6a: code-picked crossover pairs
        self.pairs = Planner(ctx, store, adapter, clock)
        from loop.core.holdout import Holdout          # v3-c6a: the blind holdout (visible ledger + audit)
        self.holdout = Holdout(ctx, store)
        self.vis: CD.Ledger | None = None              # this tick's visible ledger (selection reads it)
        from loop.prune import Pruner                  # housekeeping (see _prune)
        self.pruner = Pruner(ctx, store, clock, adapter, background=executor is ThreadPoolExecutor)

    # ------------------------------------------------------------------ seed blocks + the guard
    def blocks(self) -> list[Any]:
        if self._blocks is None:
            self._blocks = list(self.store.seed_blocks()) if hasattr(self.store, "seed_blocks") else []
        return self._blocks

    def _guard(self, run: Run) -> None:
        self.core.check_submit(run, self.blocks())     # raises: the runbook voids the row, never submits it
        self.submits += 1

    def _put_block(self, block: Any, acts: list[str]) -> None:
        self.store.put_seed_block(block)
        if block.pool == CD.TEST:                      # core.data's TEST pool reads seeds_drawn events
            self.store.log("seeds_drawn", purpose=f"confirm {block.node_ids[0]}", seeds=list(block.seeds),
                           block=block.id)
        acts.append(f"seed block {block.id} {block.pool} {list(block.seeds)} for {', '.join(block.node_ids)}")
        self._blocks = None

    def _log_test_draw(self, block: Any, seeds: Sequence[int], why: str) -> None:
        """Seeds added to an open TEST block (an EXTEND, a dead pair's replacement) are TEST seeds too: core.data
        labels a seed TEST only through a ``seeds_drawn`` event of TEST purpose (live 10-13: a node had 8 test seeds,
        the heartbeat's P_test / gap-from-test-seeds saw the first 4)."""
        if seeds:
            self.store.log("seeds_drawn", purpose=f"confirm {block.node_ids[0]}", seeds=list(seeds), block=block.id,
                           why=why)

    def _preempt_tests(self, actions: Sequence[Any], now: float, acts: list[str]) -> None:
        """``ModuleCore.preempt`` (11-10): once rank placed the stronger candidate's TEST, each open TEST block none
        of whose runs started is withdrawn -- closed now with no verdict (``verdicts=[]``: no judgement, its
        candidate back to SCREENING, its seeds stay owned, never reused), logged ``test_preempted`` -- so the
        stronger test opens this tick on the slots rank gave it."""
        placed = {a.node for a in actions if a.kind == "test"}
        for bid, (holder, ch, started) in dict(self.core.preempt).items():
            if ch not in placed or started:           # a started block runs on beside the stronger test
                if ch in placed:
                    acts.append(f"test {ch} opens beside started block {bid} ({holder}): clearly stronger")
                continue
            b = next((x for x in self.blocks() if x.id == bid and x.open), None)
            if b is None:
                continue
            g = getattr(self.core, "_gains", {}) or {}
            gh, gc = g.get(b.node_ids[0], (math.nan,))[0], g.get(ch, (math.nan,))[0]
            self._update_block(bid, acts, f"withdrawn unstarted (paired gain {holder} {gh:+.4f} < {ch} {gc:+.4f}; "
                               f"{ch}'s test takes its place)", closed_ts=now, verdicts=())
            self.store.log("test_preempted", block=bid, node_id=holder, by=ch, gain=gh, by_gain=gc,
                           seeds=list(b.seeds))
            self._blocks = None
        self.core.preempt = {}

    def _update_block(self, block_id: str, acts: list[str], note: str, **fields: Any) -> None:
        self.store.update_seed_block(block_id, **fields)
        acts.append(f"seed block {block_id}: {note}")
        self._blocks = None

    # ------------------------------------------------------------------ the tick
    def step(self) -> TickReport:
        acts: list[str] = []
        now, cfg = self.clock(), self.ctx.cfg
        self._blocks, self._doing, self._tick_ts = None, None, now   # blocks: re-read once per tick
        self.phase_s, self._t0 = {}, time.monotonic()                 # this tick's per-phase seconds, its start
        self._lock0 = self._lock_stats(reset_max=True)                # v3-c6a: the store's write-lock wait this tick
        self._wpts = 0                                                # wave points attempted this tick (min_wave_points)
        self.phase = "start"
        if not self._started:
            self._start(acts, now)
        # 1. poll: adopt / poll / repool / courtesy / smokes / LLM gate / disk guard / harvest research
        self.phase = "poll"
        self._reserve_tests(acts)                     # v3-c6a: owed TEST runs hold primary before the poll's repools
        acts += RS.tick(self.ctx, self.store, self.book, self.adapter)
        acts += self.book.resolve_submitting()
        finished = self.book.poll()
        acts += self.book.acts
        self.book.acts = []
        acts += [f"run {r.id} -> {r.status.value}" + (f" ({r.void_type.value})" if r.void_type else "")
                 for r in finished]
        acts += self.book.promote_stuck()
        acts += self.book.stale_pending()            # 10-21: pending past slurm.stale_pending_min at a limit: move / return
        acts += self.book.idle_churn()               # idle fill preempted too often: free primary or drop
        self.courtesy.step(acts)
        self._swept = R.sweep_smokes(self.adapter, self.store, acts, now, self._swept)
        self.guarded("infra.smoke", "surface the smoke gate / re-issue outage-rejected ideas", lambda: self._infra(now, acts), acts)
        R.llm_gate(self.store, now, acts)
        from loop import diagnostics as DG0          # v3 review M2: full-train runs costing far more than their FLOP charge
        acts += self.guarded("flop.audit", "audit the FLOP accounting of finished runs",
                             lambda: DG0.flop_audit(self.store, now), acts, default=[]) or []
        self._disk_guard(now, acts)
        self.research.breaker = self.guarded("research.breaker", "release a tripped research breaker",
                                             lambda: self.breaker.check(acts), acts, default=self.research.breaker) \
            or self.selfcheck_hold or ""
        self.phase = "intake"
        acts += [self._intake(req, res) for req, res in self.research.harvest(acts)]
        # 2. derive, 3. decide the open tests (then derive again if a block changed)
        self.phase = "derive"
        try:
            data = CD.load(self.db_path, self.ctx.task_id)
        except Exception as exc:  # noqa: BLE001 - no view of the ledger: nothing is decided or launched this tick
            self._call_error("ledger.load", "load the ledger view", exc, acts, quarantine=False)
            return TickReport(acts)
        if self._migrated is None:
            self._migrated = self.core.migrate(self.store, data)
            if self._migrated:
                acts.append(f"seed_blocks: {self._migrated} historical blocks migrated")
                self._blocks = None
        self._void_blocks(data, acts)
        self._eo_runs(data, acts)                    # v3-c5b M1: eval-only runs never enter a noise fit
        noise = self.core.noise(data)
        statuses = self.core.derive(data, self.blocks(), noise)
        self.phase = "decide"
        if self._decide(data, noise, statuses, now, acts):
            data = CD.load(self.db_path, self.ctx.task_id)
            statuses = self.core.derive(data, self.blocks(), noise)
        if self.guarded("labless.leader_reroot", "re-root onto our validated Labless leader (``loop.leader``)",
                        lambda: self._leader_reroot(data, statuses, now, acts), acts, default=False):
            data = CD.load(self.db_path, self.ctx.task_id)
            statuses = self.core.derive(data, self.blocks(), noise)
        if self.guarded("carriers.sync", "register TEST-pass incumbents as carriers",
                        lambda: self._sync_carriers(data, noise, statuses, acts), acts, default=False):
            data = CD.load(self.db_path, self.ctx.task_id)
            statuses = self.core.derive(data, self.blocks(), noise)
        self.data, self.statuses = data, statuses
        # our own submission leads Labless: re-base the bar's leader_median (loop.leader); a failure keeps the last bar
        acts += self.guarded("labless.leader", "re-base the bar on our Labless leader (``loop.leader``)",
                             lambda: self._leader(data, now), acts, default=[]) or []
        cfg = self.ctx.cfg
        self.guarded("nodes.sync_status", "write the derived statuses back to the stored nodes",
                     lambda: self._sync_node_status(statuses, now, acts), acts)
        if (inc_now := incumbent_of(statuses)) and self.store.get_state(K_TICK_INC) != inc_now:
            self.store.set_state(K_TICK_INC, inc_now)      # the meta report's bar section ranks against THIS incumbent
        # v3-c6a: the blind holdout -- selection reads the VISIBLE ledger (held-out probe tasks removed from every axis;
        # the official panel unchanged); the TEST verdicts above, the Labless bar and the heartbeat read the full one
        self.phase = "holdout"
        vis = self.vis = self.guarded("holdout.visible", "the holdout's visible ledger",
                                      lambda: self.holdout.visible(data), acts, default=data) or data
        self.guarded("holdout.audit", "the holdout audit", lambda: self.holdout.audit_tick(data, self.blocks(), acts, vis, now),
                     acts)
        # 4. acquire (shadow: the LLM gate's real capacity -- a pause shows as 0, not as free sessions)
        self.phase = "acquire"
        # slots that would otherwise idle (free + held by idle-fill / relax runs): more sessions (ideation.llm_*)
        self.research.gpu_free = self.book.idle() if hasattr(self.book, "idle") else self.book.free()
        llm_free = self.research.free() if not self.dry else self.research.capacity()
        if self.dry and llm_free <= 0 and (why := self.llm_state()) != "ok":
            acts.append(f"LLM {why}: no research session would start")
        held_eo = self.guarded("evalonly.prepare", "eval-only noise / harm guard",   # v3-c5 I5 (loop.evalonly)
                               lambda: self._eo_prepare(vis, acts), acts, default=()) or ()
        self.post = self.core.posterior(vis, self.blocks())
        if self.post is not None:
            self._close_waves(vis, acts)
        if hasattr(self.core, "prepare"):            # knob banks (reads configs): one that raises = no wave this tick
            acts += self.guarded("core.prepare", "prepare the knob banks",
                                 lambda: self.core.prepare(self, vis, statuses, self.post), acts, default=[]) or []
        if self.post is not None:                    # v3-c6a: code-picked crossover pairs (loop.core.pairs)
            self.guarded("pairs.plan", "plan the crossover pairs", lambda: self.pairs.step(
                self, vis, self.post, statuses, incumbent_of(statuses), acts), acts)
            acts += self.guarded("alloc.graft", "graft the code-picked pairs (v3-c7i I5)",      # before the LLM brief
                                 lambda: self._graft_pairs(vis, statuses, now), acts, default=[]) or []
        if held_eo and hasattr(self.core, "held"):   # harm-guarded eval-only readouts: no more runs, no test
            self.core.held = frozenset(self.core.held) | frozenset(held_eo)
        if hasattr(self.core, "held"):               # 11-10 moonshot rescue: rungs at their seeds, rescued originals
            from loop.core import rescue as RESCUE
            self.core.held = frozenset(self.core.held) | frozenset(self.guarded(
                "rescue.hold", "the moonshot rescue's held nodes", lambda: RESCUE.hold(self.ctx, self.store, vis), acts,
                default=()) or ())
            if hasattr(self.core, "seed_caps"):      # 11-11 review: a rescue point never gets more than its seeds
                self.core.seed_caps = self.guarded("rescue.seed_caps", "the moonshot rescue's per-point run caps",
                                                   lambda: RESCUE.seed_caps(self.ctx, self.store, vis), acts,
                                                   default={}) or {}
        if hasattr(self.core, "configure"):          # the target and the operator acquisition keys (+ their state)
            self.core.configure(cfg, store=self.store)
        if hasattr(self.core, "test_backoff"):        # v3-c5b S3: no fresh test inside a control-unscorable backoff
            self.core.test_backoff = self._test_backoff(now)
        actions = list(self.core.rank(vis, self.blocks(), statuses, self.post, self.book.free(), llm_free))
        acts += self.guarded("alloc.backlog", "age out the QUEUED backlog (v3-c7i I2)",
                             lambda: self._age_out(vis, statuses, actions, now), acts, default=[]) or []
        acts += self.guarded("alloc.prune_stale", "prune stale QUEUED tweaks of flat families (11-09)",
                             lambda: self._prune_stale(vis, statuses, actions, now), acts, default=[]) or []
        if notes := getattr(self.core, "notes", None):
            for line in notes:
                self.store.log("acquire_note", text=line[:500])
            acts += notes
            self.core.notes = []
        # a LOCKED file changed since the deploy (core.meta.Locks): the breaker -- no launch, no new session
        breach = getattr(self.meta, "guard", lambda _t: None)(self)
        if breach:
            acts.append(breach)
        # 5. launch: owed infra retries, the runs open tests still need, then the ranked runs while slots are free
        self.phase = "launch"
        launched = [] if breach else self._retries(statuses, acts)
        if getattr(self.core, "preempt", None):      # 11-10: an unstarted TEST block gives way to a stronger test
            if breach or self.book.free() <= 0:       # nothing would open in its place this tick
                self.core.preempt = {}
            else:
                self.guarded("test.preempt", "withdraw an unstarted TEST block for a stronger candidate",
                             lambda: self._preempt_tests(actions, now, acts), acts)
        launched += [] if breach else self._launch(actions, data, statuses, now, acts)
        # 6. ideate, 7. meta
        self.phase = "ideate"
        if not breach:
            acts += self.research.start_requeued()
            self._port_sessions(acts)
            self.guarded("reroot.cross", "re-root crossover sessions",                       # operator 11-10
                         lambda: self._reroot_cross_sessions(acts), acts)
            self.guarded("repair.sessions", "probe ill-conditioning repair sessions",          # v3-c7a A4
                         lambda: self._repair_sessions(data, acts), acts)
            self.guarded("moonshot.debug", "moonshot rescue DEBUG sessions (``loop.core.rescue``)",   # 11-10
                         lambda: __import__("loop.core.rescue", fromlist=["x"]).debug_sessions(self, data, acts), acts)
            acts += self.guarded("fnf.pause_expiry", "the fork-and-flush pause expiry", self._fnf_pause_expiry, acts,
                                 default=[]) or []
            acts += self.ideate(self, [a for a in actions if a.kind == "idea"], vis, statuses)
        self.phase = "meta"
        # v3-c6r H2: the meta report / A/B replay the LLM reviewer reads is the VISIBLE ledger, its panel the visible
        # composite (holdout.visible_panel); the official numbers stay with the TEST bar and the heartbeat
        meta_led = self.data if self.vis is None or self.vis is self.data else self._meta_ledger()
        acts += self.meta(self, meta_led, statuses)
        if not self.dry:
            self.phase = "housekeeping"
            acts += R.databank_tick(self.ctx, self.store, self.adapter, now)
            acts += self.guarded("databank.runs", "run the banked datasets on the incumbent", lambda: R.DB.run_unrun(
                self.ctx, self.store, self.adapter, data, incumbent_of(statuses), now), acts, default=[]) or []
            acts += self.guarded("program.ladders", "build the program dose ladders",
                                 lambda: self._ladders(vis, statuses, now), acts, default=[]) or []
            acts += self.guarded("program.arms", "materialise the operator programs' eval-only arms",   # v3-c6r O1
                                 lambda: self._arms(statuses, now), acts, default=[]) or []
            acts += self.labless.step(incumbent_of(statuses), data)
            self.guarded("board.step", "the competition board ingest (``loop.board``)",
                         lambda: self.board.step(acts), acts)
            self.guarded("local_runs.step", "import the operator's manual runs (``loop.localruns``)",
                         lambda: self.local_runs.step(acts), acts)
            acts += self.guarded("local_runs.confirms", "build the promising manual runs' confirm nodes",
                                 lambda: self.local_runs.confirm(self.adapter), acts, default=[]) or []
            if self.llm_state() == "ok":
                self.guarded("root_audit.step", "the root assumption ledger (``loop.root_audit``)",
                             lambda: self.root_audit.step(acts), acts)
                self.guarded("scout.v3", "the research scout (``scout.v3``)", lambda: self._scout_v3(acts), acts)
            acts += self._prune(data, statuses)
            acts += self.guarded("queue.age_out", "age out QUEUED nodes that never ran",
                                 lambda: self._age_out_queued(data, statuses, now), acts, default=[]) or []
            self.guarded("worktree.prewarm", "pre-create the run worktrees of queued nodes",
                         lambda: self._prewarm(statuses, acts), acts)
            self.guarded("dashboard.render", "render the dashboard page (``dashboard.every_ticks``)",
                         lambda: self._dashboard(statuses, now, acts), acts)
        self.phase = "done"
        self._lock_time()
        return TickReport(acts, statuses, actions, launched)

    def _scout_v3(self, acts: list[str]) -> None:
        """``scout.v3`` (11-08: the scout was wired only into the v2 driver): the web research scout every
        ``scout.every_hours`` in the pool; its cited ideas go to its bank, which v3 briefs serve (``ideation.bank_v3``)."""
        from loop import scout as SC
        sc = self.ctx.cfg.scout
        if not (sc.enabled and getattr(sc, "v3", False)):
            return
        now = self.clock()
        if self._scout is not None and self._scout[0].done():
            (fut, trigger), self._scout = self._scout, None
            try:
                res = fut.result()
            except Exception as exc:  # noqa: BLE001 - a failed session is a failed run (retried after an hour)
                res = {"trigger": trigger, "ts": now, "ok": False, "error": repr(exc)[:300], "ideas": []}
            run = SC.write(self.ctx, self.store, res)
            acts.append(f"research scout ({trigger}): {run['written']} idea(s) written, {run['dup']} dup, "
                        f"{run['refused']} refused" + ("" if res.get("ok") else f" (FAILED: {str(res.get('error'))[:80]})"))
        if self._scout is not None:
            return
        llm = getattr(getattr(self.research, "researcher", None), "llm", None)
        from loop import board as BD
        if BD.enabled(self.ctx) and not BD.digest(self.store):   # the scout reads the board digest: wait for the first
            return
        if llm is None or not (trigger := SC.due(self.ctx, self.store, now, evals_since_confirm=0, confirm_marker=None)):
            return
        self._scout = (self.pool.submit(SC.Scout(self.ctx, llm).run, self.store, None, trigger=trigger, now=now), trigger)
        acts.append(f"research scout started ({trigger})")

    def _age_out_queued(self, data: CD.Ledger, statuses: Mapping[str, Any], now: float) -> list[str]:
        """``acquire.queued_max_age_h``: QUEUED nodes that never ran and sat longer than that are marked aged out
        (REJECTED / ``dropped_backlog: aged_out_queued``; recoverable with ``acquire.restore_aged``), not deleted, so the backlog does not
        starve fresh nodes and program arms (live 10-21: 320 stored-queued nodes, some from 10-08)."""
        A = getattr(self.core, "A", None)          # the core's acquire module (a Tick has no ``A`` of its own)
        if A is None or not hasattr(A, "aged_queued") or self.dry:
            return []
        touched = {n.id: float(n.updated_ts or 0.0) for n in self.store.nodes()}
        ids = A.aged_queued(data, statuses, now, touched=touched)
        done = A.age_out(self.store, ids, now)
        return [f"queue: aged out {len(done)} QUEUED node(s) older than {A.QUEUED_MAX_AGE_H:g} h that never ran "
                f"({', '.join(done[:6])}{' ...' if len(done) > 6 else ''}; recoverable)"] if done else []

    def _dashboard_spawn(self, cmd: list[str]) -> Any:
        """The render child: ``loop dashboard`` in its own NICED process (the page is ~50 s of CPU / 26 MB: in-process it
        would eat the tick's time box and hold the GIL)."""
        import subprocess
        return subprocess.Popen(["nice", "-n", "10", *cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                stdin=subprocess.DEVNULL)

    def _dashboard(self, statuses: Mapping[str, Any], now: float, acts: list[str]) -> bool:
        """Every ``dashboard.every_ticks`` ticks (and at least ``min_interval_s`` apart) render
        ``<state_dir>/dashboard/index.html`` -- the file ``loop serve`` serves (it went stale 10-08: loop.dashboard was
        reachable only from the CLI). The page reads the stored node status, which v3 never wrote PASSED / INCUMBENT
        into, so the tick's DERIVED status of every node whose stored one is the sync's to overwrite is stored first
        (``dashboard.K_STATUS``) and the page shows it. One render at a time; True when one was started."""
        from loop import dashboard as DASH
        cfg = getattr(self.ctx.cfg, "dashboard", None)
        every = int(getattr(cfg, "every_ticks", 0) or 0)
        spec = str(getattr(self.ctx, "spec_path", "") or "")
        if every <= 0 or self.dry or not spec:
            return False
        st = dict(self.store.get_state(DASH.K_RENDER) or {})
        st["ticks"] = int(st.get("ticks", 0)) + 1
        if (proc := self.__dict__.get("_dash_proc")) is not None:
            if (rc := proc.poll()) is None:                  # still rendering: never two at once
                self.store.set_state(DASH.K_RENDER, st)
                return False
            self._dash_proc = None
            if rc:
                self.store.log("dashboard_error", rc=rc)
                acts.append(f"dashboard: the render child exited {rc}")
        if st["ticks"] < every or now - float(st.get("ts", 0.0)) < float(getattr(cfg, "min_interval_s", 0.0) or 0.0):
            self.store.set_state(DASH.K_RENDER, st)
            return False
        owned = dict(self.store.get_state(K_NODE_SYNC) or {})
        over = {}
        for node in self.store.nodes():
            name = status_name(statuses.get(node.id)) if node.id in statuses else ""
            if name in ("PASSED", "INCUMBENT") and node.id != ROOT_NODE_ID and (
                    node.status.value in SYNC_FROM or owned.get(node.id) == node.status.value):
                over[node.id] = name.lower()
        self.store.set_state(DASH.K_STATUS, over)
        self._dash_proc = self._dashboard_spawn([sys.executable, "-m", "loop", "dashboard", "--task", spec,
                                                 "--state-dir", str(self.ctx.state_dir)])
        self.store.set_state(DASH.K_RENDER, {"ticks": 0, "ts": now})
        return True

    PREWARM_MAX = 12     # node worktrees one background prewarm creates at most

    def _retry_controls(self, data: CD.Ledger, acts: list[str]) -> list[str]:
        """v3 review O4: an open wave's control seed whose carrier PARENT row is dead (crash / void, so
        ``factorial.wave_controls`` lists it again but the row exists and no spec can launch it) is re-submitted as
        the same row's next attempt (``book.retry``) while attempts are left and slots are free."""
        from loop.core import factorial as F
        out = []
        for wid in self._open_waves(data):
            for n, s in F.wave_controls(data, wid):
                run = self.store.get_run(Run.make_id(n, Stage.PARENT, s))
                if run is None or run.status not in (RunStatus.CRASH, RunStatus.VOID) or self.book.free() <= 0 \
                        or run.attempt >= self.book.infra_retry or self.dry:
                    continue
                if (node := self.store.get_node(n)) is None:
                    continue
                run = self.book.retry(run, node)
                self.store.log("wave_control_retry", run_id=run.id, wave=wid, attempt=run.attempt)
                acts.append(f"wave {wid}: control {run.id} re-submitted (attempt {run.attempt}) -> {run.status.value}")
                out.append(run.id)
        return out

    def _prewarm(self, statuses: Mapping[str, Any], acts: list[str]) -> None:
        """Live 10-14 (tick 138, 238 s): launch.submit took 193 s, ~25 s per NEW node's `git worktree add` (serial
        under the adapter's worktree lock). The QUEUED nodes' run worktrees are created on ONE background thread
        (``adapter.warm_worktree``) between ticks, newest first, so their first submit finds them; a thread still
        running is left alone. No-op for an adapter without it."""
        import threading
        warm = getattr(self.adapter, "warm_worktree", None)
        if warm is None or (th := getattr(self, "_warm_th", None)) is not None and th.is_alive():
            return
        if n := getattr(self, "_warmed", 0):
            self._warmed = 0
            acts.append(f"worktree prewarm: {n} queued node worktree(s) created between ticks")
        todo, since = [], self.clock() - 6 * 3600.0       # the fresh nodes only: an old backlog may never screen
        for nid, st in statuses.items():
            if status_name(st) != "QUEUED":
                continue
            node = self.store.get_node(nid)
            if node is not None and node.commit and float(node.created_ts or 0.0) >= since:
                todo.append((float(node.created_ts or 0.0), nid, node.commit))
        todo = sorted(todo, reverse=True)[:self.PREWARM_MAX]
        if not todo:
            return

        def run() -> None:
            n = sum(bool(warm(nid, commit)) for _, nid, commit in todo)
            self._warmed = getattr(self, "_warmed", 0) + n
        self._warm_th = threading.Thread(target=run, name="worktree-prewarm", daemon=True)
        self._warm_th.start()

    def _lock_stats(self, reset_max: bool = False) -> dict[str, float]:
        try:
            return dict(self.store.lock_stats(reset_max=reset_max)) if hasattr(self.store, "lock_stats") else {}
        except Exception:  # noqa: BLE001 - a store without stats
            return {}

    def _lock_time(self) -> None:
        """``phase_s`` store.lock_wait (seconds every thread waited for the store's write lock this tick) and
        store.lock_max (the longest single wait); the heartbeat shows them with the phases."""
        now, t0 = self._lock_stats(), getattr(self, "_lock0", None) or {}
        if not now:
            return
        ps = self.__dict__.setdefault("phase_s", {})
        ps["store.lock_wait"] = float(now.get("lock_wait_s", 0.0)) - float(t0.get("lock_wait_s", 0.0))
        ps["store.lock_max"] = float(now.get("lock_wait_max_s", 0.0))

    def _graft_pairs(self, data: CD.Ledger, statuses: Mapping[str, Any], now: float) -> list[str]:
        """v3-c7i I5 (loop.core.alloc.graft_pairs): the planner's code-leg pairs grafted by code; a conflict is
        flagged for the LLM crossover brief. Shadow: listed only."""
        from types import SimpleNamespace
        from loop.core import alloc as AL
        view = SimpleNamespace(post=self.post, led=data, incumbent=incumbent_of(statuses))
        return AL.graft_pairs(self.ctx, self.store, self.adapter, view, statuses, now, dry=self.dry)

    def _age_out(self, data: CD.Ledger, statuses: Mapping[str, Any], actions: list[Any], now: float) -> list[str]:
        """v3-c7i I2: QUEUED nodes older than ``alloc.drop_age_h`` whose screen value / GPU-h is outside the best
        ``alloc.backlog_keep`` are dropped (REJECTED 'dropped_backlog', logged; shadow: listed only)."""
        from loop.core import alloc as AL
        ac = AL.cfg_of(self.ctx)
        opts = [o for o in getattr(self.core, "last_opts", []) or [] if o.kind == "screen" and o.node]
        if not ac.enabled:
            return []
        A = getattr(self.core, "A", None)
        scored = A.score(opts) if (A is not None and opts) else []
        rates = {a.node: a.rate for a in scored if a.node}
        bl = AL.backlog(data, statuses, now)
        # v3 review K3: a prerequisite port is never aged out (a wave's knobs / a port wait on it)
        pre = {n for n in bl["ages"] if "prereq_for=" in (data.nodes[n].notes or "") if n in data.nodes}
        if A is not None and hasattr(A, "program_child"):    # 10-22: program arms / ladder doses are run on purpose
            pre |= {n for n in bl["ages"] if n in data.nodes and A.program_child(data.nodes[n])}
        if getattr(ac, "keep_bank_nodes", False) and A is not None and hasattr(A, "operator_bank_node"):
            # 11-09 review of 182a2ab: backlog_keep 20 dropped operator-bank / root-audit arms (not program_child)
            pre |= {n for n in bl["ages"] if n in data.nodes and A.operator_bank_node(data.nodes[n])}
        held = (*(getattr(self.core, "held", ()) or ()), *sorted(pre))
        rows = AL.age_out(ac, bl, rates, held) if opts else []
        try:                                  # nodes an open seed block waits on are never starved out from under it
            blocked = {n for b in self.store.seed_blocks() if getattr(b, "open", False) for n in b.node_ids}
        except Exception:  # noqa: BLE001 - no block view
            blocked = set()
        touched = {}                          # a restored node (acquire.restore_aged) gets a fresh window
        for ev in self.store.events("queued_restored"):
            touched[str(ev.get("node_id"))] = max(float(ev.get("ts") or 0.0), touched.get(str(ev.get("node_id")), 0.0))
        gone0 = {r[0] for r in rows}
        starved = [r for r in AL.starved(ac, bl, data.nodes, {*held, *blocked, *gone0}, touched, now)]
        if not rows and not starved:
            return []
        if self.dry:
            out = []
            if rows:
                out.append(f"backlog: would drop {len(rows)} aged QUEUED node(s) of {bl['n']} (oldest {max(r[1] for r in rows):.0f} h): "
                           + ", ".join(r[0] for r in rows[:12]))
            if starved:
                out.append(f"backlog: would drop {len(starved)} starved zero-run QUEUED node(s) of {bl['n']} (oldest "
                           f"{max(r[1] for r in starved):.0f} h): " + ", ".join(r[0] for r in starved[:12]))
            return out
        gone = gone0 | {r[0] for r in starved}
        if isinstance(actions, list):                  # a dropped node is not screened this tick either
            actions[:] = [a for a in actions if getattr(a, "node", None) not in gone]
        return (AL.drop(self.store, rows, int(ac.backlog_keep), now) if rows else []) \
            + AL.drop_starved(self.store, starved, float(ac.drop_age_h), now)

    def _prune_stale(self, data: CD.Ledger, statuses: Mapping[str, Any], actions: list[Any], now: float) -> list[str]:
        """11-09 ``alloc.prune_stale_family`` (loop.core.alloc.stale_family): QUEUED nodes of a mechanism cluster whose
        measured members all came in under the promote need are REJECTED ``pruned_stale_family`` (dry: listed only).
        Off (no key) = nothing read. Never a node with a run, an open seed block, a held / confirm / program node."""
        from loop.core import alloc as AL
        ac = AL.cfg_of(self.ctx)
        if not ac.enabled or getattr(ac, "prune_stale_family", None) is None or self.post is None \
                or not hasattr(self.post, "beliefs"):
            return []
        beliefs = AL.post_beliefs(self.post)           # paired only (a seed offset is no gain)
        try:
            blocked = {n for b in self.store.seed_blocks() if getattr(b, "open", False) for n in b.node_ids}
        except Exception:  # noqa: BLE001 - no block view: refuse rather than prune under an open block
            return []
        touched: dict[str, float] = {}
        for ev in self.store.events("queued_restored"):
            touched[str(ev.get("node_id"))] = max(float(ev.get("ts") or 0.0), touched.get(str(ev.get("node_id")), 0.0))
        launching = {getattr(a, "node", None) for a in actions or () if getattr(a, "kind", "") != "idea"}
        exempt = {*(getattr(self.core, "held", ()) or ()), *blocked, *(n for n in launching if n)}
        protect = AL.protected(data, incumbent_of(statuses))
        revived = [] if self.dry else AL.revive_pruned(self.ctx, self.store, data, self.post, beliefs, now, protect)
        rows = AL.stale_family(self.ctx, self.store, data, statuses, self.post, beliefs, now, exempt=exempt,
                               protect=protect, touched=touched)
        if not rows:
            return revived
        if self.dry:
            return [f"backlog: would prune {len(rows)} stale QUEUED node(s) of flat families: "
                    + ", ".join(f"{r['node']} ({r['cluster']} {r['label']}, best {r['best'][1]:+.4f})" for r in rows[:12])]
        return revived + AL.prune_stale(self.store, rows, now)

    def _arms(self, statuses: Mapping[str, Any], now: float) -> list[str]:
        from loop.core.program import arms
        return arms(self.ctx, self.store, self.adapter, self.store.get_node(incumbent_of(statuses)), now)

    def _ladders(self, data: CD.Ledger, statuses: Mapping[str, Any], now: float) -> list[str]:
        from loop.core.program import ladders           # a program node's small dose ladder (10-13)
        return ladders(self.ctx, self.store, self.adapter, data, self.post, statuses, now)

    # ------------------------------------------------------------------ guarded call sites + quarantine
    def quarantined(self, key: str) -> bool:
        return key in (self.store.get_state(K_QUARANTINE) or {})

    def quarantine(self, key: str, reason: str, acts: list[str]) -> None:
        """Skip ``key`` for good (until the operator removes it from state ``tick.quarantine``)."""
        q = dict(self.store.get_state(K_QUARANTINE) or {})
        if key in q:
            return
        q[key] = {"reason": reason[:500], "ts": self.clock(), "rev": code_rev()}
        self.store.set_state(K_QUARANTINE, q)
        self.store.log("quarantine", key=key, reason=reason[:500])
        acts.append(f"ALERT QUARANTINED {key}: {reason}"[:400])
        print(f"{R.utc_now(self.clock())} ALERT: quarantined {key} after repeated failures: {reason}"[:600],
              file=sys.stderr, flush=True)

    def _call_error(self, key: str, what: str, exc: BaseException, acts: list[str], *, quarantine: bool = True,
                    on_quarantine: Callable[[str], None] | None = None) -> None:
        """Log a failed call site (``tick_call_error`` with the traceback); the QUARANTINE_AFTER-th failure in a row
        of the same ``key`` quarantines it (``on_quarantine(reason)``: e.g. a REJECTED port node)."""
        errs = dict(self.store.get_state(K_ERRORS) or {})
        msg = repr(exc)[:500]
        n = int((errs.get(key) or {}).get("n", 0)) + 1
        errs[key] = {"n": n, "error": msg, "ts": self.clock()}
        self.store.set_state(K_ERRORS, errs)
        self.store.log("tick_call_error", key=key, what=what, n=n, error=msg, phase=self.phase,
                       traceback=traceback.format_exc()[-3000:])
        acts.append(f"ERROR {what} [{key}, failure {n}]: {msg}"[:400])
        if quarantine and n >= QUARANTINE_AFTER:
            self.quarantine(key, f"{what}: {msg}", acts)
            if on_quarantine is not None:
                try:
                    on_quarantine(f"{what}: {msg}")
                except Exception as exc2:  # noqa: BLE001
                    self.store.log("tick_call_error", key=f"{key}.quarantine", error=repr(exc2)[:500])

    def release_stale_quarantine(self, acts: list[str], rev: str | None = None) -> list[str]:
        """Once per process: a site quarantined by an OLDER revision (its entry's ``rev`` differs from this process's)
        is released -- the deploy may have fixed it; if it still fails it is quarantined again after
        QUARANTINE_AFTER failures. An entry WITHOUT a ``rev`` (written before the field existed) is stale by definition --
        no code that wrote it can be named, and it would otherwise persist across every deploy (10-22) -- so the first
        start of a process whose revision CAN be read releases it too. A process whose revision cannot be read releases
        nothing: only the operator does."""
        rev = code_rev() if rev is None else rev
        q = dict(self.store.get_state(K_QUARANTINE) or {})
        gone = [k for k, v in q.items() if rev and (v or {}).get("rev") != rev]
        if not gone:
            return []
        errs = dict(self.store.get_state(K_ERRORS) or {})
        self.store.set_state(K_QUARANTINE, {k: v for k, v in q.items() if k not in gone})
        self.store.set_state(K_ERRORS, {k: v for k, v in errs.items() if k not in gone})
        self.store.log("quarantine_released", keys=gone, rev=rev)
        acts.append(f"quarantine released on new code {rev}: {', '.join(gone)} (re-quarantined if they fail again)")
        return gone

    def _call_ok(self, key: str) -> None:
        errs = self.store.get_state(K_ERRORS) or {}
        if key in errs:
            self.store.set_state(K_ERRORS, {k: v for k, v in errs.items() if k != key})

    def _leader_reroot(self, data: Any, statuses: Mapping[str, Any], now: float, acts: list[str]) -> bool:
        """``labless.leader_reroot``: our VALIDATED Labless leader becomes the incumbent at once, through the usual
        path -- a frozen TEST pass (tier ``leader``) over the incumbent: its own open TEST block is closed with it
        (else a closed seedless block holds it), its seeds re-mint as the new incumbent's SELECT block, and status
        replay / stale-control re-pairing / carriers / re-root ports follow. Once per leader run; never when it is
        already the incumbent or an ancestor of it. True when a block was written (the caller re-derives)."""
        from loop import leader as LD
        from loop.core.seeds import SeedBlock
        inc = incumbent_of(statuses)
        plan = LD.reroot_plan(self.ctx, self.store, data, inc)
        if plan is None:
            return False
        rid, nid, why = plan
        if self.dry:
            acts.append(f"{nid}: would LEADER RE-ROOT over {inc} ({why})")
            return False
        own = [b for b in self.blocks() if b.pool == CD.TEST and b.open and b.node_ids and b.node_ids[0] == nid]
        try:                                           # its paired gain over the incumbent (reporting only)
            from loop.core import bar as BR
            pv = BR.paired(data, inc)
            g = pv.gains.get(nid) if pv is not None else None
        except Exception:  # noqa: BLE001
            g = None
        at = now + 1e-3                                # its own replay group, after this tick's TEST verdicts
        rec = {"node": nid, "vs": inc, "kind": "pass", "e": 0.0, "n": int(g[2]) if g else 0,
               "mean": float(g[0]) if g else 0.0, "reason": f"leader re-root: {why}", "harm": [],
               "sd": math.sqrt(g[1]) if g and g[1] >= 0 else 0.0, "legacy": False, "tier": "leader", "ts": at,
               "labless_run": rid}
        if own:
            b = max(own, key=lambda x: (x.drawn_ts, x.id))
            self._update_block(b.id, acts, f"{nid} PASS (leader re-root: {why}): closed", closed_ts=at,
                               verdicts=(rec,))
            seeds = tuple(b.seeds)
        else:
            self.store.put_seed_block(SeedBlock(f"leader-{nid}-{int(now)}", CD.TEST, (), (nid, inc),
                                                f"leader re-root {nid}", now, at, (rec,)))
            self._blocks = None
            seeds = tuple(sorted({o.seed for o in data.in_epoch().by_node().get(nid, ())}))
        if seeds:
            self._put_block(SeedBlock(f"remint-{nid}-{int(now)}", CD.SELECT, seeds, (nid,), f"remint {nid}", now),
                            acts)
        self.store.set_state(LD.K_REROOT, {"run_id": rid, "node": nid, "previous": inc, "ts": now})
        self.store.log("leader_reroot", node_id=nid, previous=inc, run_id=rid, why=why)
        acts.append(f"{nid}: LEADER RE-ROOT over {inc} ({why}): every delta is now measured against {nid}")
        return True

    def _leader(self, data: Any, now: float) -> list[str]:
        """``loop.leader``: judge who leads from the stored board rows, then install the effective leader_median."""
        from loop import leader as LD
        lines = LD.sync(self.ctx, self.store, data, now, spec_lc=self._spec_labless)
        self.ctx.cfg = LD.effective(self.ctx.cfg, self.store, self._spec_labless)
        return lines

    def guarded(self, key: str, what: str, fn: Callable[[], Any], acts: list[str], default: Any = None, *,
                on_quarantine: Callable[[str], None] | None = None) -> Any:
        """``fn()``, or ``default`` when it raises (logged; repeated -> quarantined) or ``key`` is quarantined."""
        if self.quarantined(key):
            return default
        outer, self._doing = self._doing, key
        try:
            out = fn()
        except Exception as exc:  # noqa: BLE001 - one call site never stops the tick
            self._call_error(key, what, exc, acts, on_quarantine=on_quarantine)
            return default
        finally:
            self._doing = outer
        self._call_ok(key)
        return out

    def _infra(self, now: float, acts: list[str]) -> None:
        """The smoke gate (loop.infra): one ``ALERT_smoke_infra`` event + line per trip; the operator's re-issue drop file."""
        if self.dry:
            return
        gate = getattr(self.adapter, "smoke_gate", None)
        st = gate.status() if gate is not None else {}
        if st.get("tripped") and st.get("trip_id") != self.store.get_state(INFRA.K_INFRA_ALERTED):
            self.store.set_state(INFRA.K_INFRA_ALERTED, st.get("trip_id"))
            self.store.log("ALERT_smoke_infra", reason=str(st.get("reason", ""))[:400], sig=st.get("sig", ""),
                           hold_until=st.get("hold_until"), backoff=st.get("backoff"))
            acts.append(f"smoke INFRA hold: {str(st.get('reason', ''))[:300]} (new smokes held {st.get('backoff', 0):.0f}s, "
                        "then one canary; ideas are kept and re-issued, not rejected)")
        acts += INFRA.reissue_from_file(self.ctx, self.store, self.research, now)

    def _intake(self, req: Any, res: Any) -> str:
        """A harvested session -> a node; an intake that raises is logged (``research_failed``), never fatal."""
        if INFRA.is_infra_unavailable(res):              # the smoke died of the environment: no node, re-issue the request
            return self.research.reissue_infra(req, res)
        if INFRA.is_smoke_starved(res):                  # the smoke queue timed out: no node, the draft is kept and re-issued
            return self.research.reissue_starved(req, res)
        try:
            out = self.intake(req, res)
        except Exception as exc:  # noqa: BLE001
            self.store.log("tick_call_error", key=f"intake:{req.id}", what="intake", error=repr(exc)[:500],
                           phase=self.phase, traceback=traceback.format_exc()[-3000:])
            self.store.log("research_failed", request_id=req.id, status="intake_error", error=repr(exc)[:500])
            out = f"ERROR intake of {req.id}: {exc!r}; no node"[:400]
            # review fix: an intake that raised is a session that produced nothing -- the breaker counts it
            Tick._breaker_record(self, SimpleNamespace(status="failed", error=f"intake_error: {exc!r}"[:300]), out, [])
            return out
        extra: list[str] = []
        Tick._breaker_record(self, res, str(out or ""), extra)
        return "\n".join([str(out), *extra]) if extra else out

    def _breaker_record(self, res: Any, out: str, extra: list[str]) -> None:
        """Count one harvested session in the research breaker (loop.breaker); its trip / release sets the hold."""
        if (brk := getattr(self, "breaker", None)) is None:   # 11-08: the LLM-spend circuit breaker (loop.breaker)
            return
        try:
            brk.record(res, out, extra)
            self.research.breaker = brk.tripped() or getattr(self, "selfcheck_hold", "") or ""
        except Exception as exc:  # noqa: BLE001 - a breaker fault never loses the intake
            brk.store.log("tick_call_error", key="research.breaker", what="breaker.record", error=repr(exc)[:500])

    # ------------------------------------------------------------------ disk guard, prune, heartbeat
    def _disk_paths(self) -> list[Path]:
        out = [Path(self.ctx.state_dir)] if getattr(self.ctx, "state_dir", "") else []
        wt = getattr(self.adapter, "worktrees", None)
        return out + ([Path(wt)] if isinstance(wt, (str, Path)) else [])

    def _disk_guard(self, now: float, acts: list[str]) -> None:
        """Below DISK_FLOOR_GB free on the state / worktree filesystem: no new research session (an alert, a
        ``disk_low`` event on each change) and the prune pass is due at once; lifted once space is back."""
        free = {}
        for path in self._disk_paths():
            try:
                free[str(path)] = shutil.disk_usage(path if path.exists() else path.parent).free / 1e9
            except OSError:
                continue
        self.disk = free
        floor = float(getattr(getattr(self.ctx, "cfg", None), "disk_floor_gb", 0.0) or 0.0) or DISK_FLOOR_GB
        low = {p: g for p, g in free.items() if g < floor}
        why = ", ".join(f"{p} {g:.0f} GB free" for p, g in low.items())
        if why and not self.research.disk_low:
            self.store.log("disk_low", free_gb=free, floor_gb=floor)
            acts.append(f"ALERT disk low ({why} < {floor:g} GB): no new research session; prune due now")
            print(f"{R.utc_now(now)} ALERT: disk low: {why} < {floor:g} GB", file=sys.stderr, flush=True)
            from loop.prune import K_PRUNE
            self.store.set_state(K_PRUNE, {"ts": 0.0})
        elif not why and self.research.disk_low:
            self.store.log("disk_ok", free_gb=free, floor_gb=floor)
            acts.append("disk space back above the floor: research sessions may start")
        self.research.disk_low = why

    def prune_view(self, data: CD.Ledger, statuses: Mapping[str, Any]) -> Any:
        """The snapshot ``loop.prune`` reads in v3: the incumbent (a model), the derived-closed nodes, and what open
        nodes / blocks / waves / ports reference (kept whatever their store status)."""
        from types import SimpleNamespace
        from loop.core import factorial as F
        closed = {n for n, st in statuses.items() if status_name(st) in CLOSED}
        # v3 (10-19): a SCREENING node stays SCREENING for life, so it is NOT kept merely for being open (that held
        # 2357 nodes / 1.9 TB of latest.pt): ``screening`` goes to loop.prune.classify, which settles those no open
        # work, run or grace window holds. ``keep`` = everything else not closed + what open work references.
        screening = {n for n, st in statuses.items() if status_name(st) == "SCREENING"}
        keep = {n for n, st in statuses.items() if status_name(st) not in CLOSED and n not in screening}
        opened = {n for b in self.blocks() if b.open for n in b.node_ids}
        for wid, car in self._open_waves(data).items():
            opened |= {car, *(n for n, _ in F.wave_points(data, wid).values())}
        for rec in (self.store.get_state(self.K_PORT_SESSIONS) or {}).values():
            opened |= {x for x in (rec.get("parent"), rec.get("anchor")) if x}
        queued = {n for n, st in statuses.items() if status_name(st) == "QUEUED"}
        keep |= opened
        keep |= self._prune_models(data, statuses)
        keep |= {data.nodes[n].parent for n in list(keep) if n in data.nodes and data.nodes[n].parent}
        # v3-c8a B6: ``queued`` / ``open`` -- what still launches -- decide which node WORKTREES stay (loop.prune
        # .worktree_nodes); ``keep`` keeps checkpoints
        opened |= {data.nodes[n].parent for n in list(opened | queued) if n in data.nodes and data.nodes[n].parent}
        return SimpleNamespace(incumbent_id=incumbent_of(statuses), banked=(), closed=closed - keep, keep=keep,
                               screening=screening - keep, queued=queued, open=opened)

    def _prune_models(self, data: CD.Ledger, statuses: Mapping[str, Any]) -> set[str]:
        """Nodes whose checkpoints must outlive their screening although no open work references them (10-20 review of
        08876d4: its first pass would have deleted 23 buildable alt-base candidates -- the top three by posterior
        among them -- and ``altbase.buildable`` needs a checkpoint, so each would have left the lane for good):
        the in-flight / re-queued research sessions' parents (their child resumes the parent's checkpoint
        eval-only), every past incumbent, the bar node, and the ``prune.alt_top`` best sound, standing current-epoch
        nodes by posterior PLUS every such node within one posterior sd of the Nth's value (hysteresis: nodes at
        noise level of the cut are not deleted by ranking jitter) -- the alt-base sources: cluster / top-outside
        picks and the previous epoch's ports. The posterior is the live tick's, built on the holdout-visible ledger."""
        from loop.contracts import K_INFLIGHT
        out: set[str] = set()
        for e in self.store.get_state(K_INFLIGHT) or []:
            out |= {str(x) for x in e.get("parents") or [] if x}
        out |= {x for row in getattr(data, "incumbents", ()) or () for x in row[1:] if isinstance(x, str) and x}
        out |= {str(getattr(self.ctx.cfg.labless, "bar_node", "") or "")}
        out |= {q for n in list(out) if n in data.nodes and (q := data.port(n))}
        try:
            from types import SimpleNamespace as NS
            from loop.core import altbase as AB
            beliefs = self.post.beliefs() if self.post is not None else {}
            v = NS(beliefs={n: NS(m=m, v=var, n=k) for n, (m, var, k) in beliefs.items()}, post=self.post)
            ms, zz = int(AB.opt(self.ctx, "alt_base_min_seeds")), float(AB.opt(self.ctx, "alt_base_z"))
            ok, sds = {}, {}
            for n in data.nodes:
                if n in beliefs and data.nodes[n].base == data.epoch and not AB._skip_node(data, n) \
                        and (got := AB.standing(v, n, ms, zz)) is not None and AB.sound(self.store, statuses, n, self.ctx):
                    ok[n] = got[0]
                    sds[n] = got[1]
            top = max(0, int(self.ctx.cfg.prune.alt_top))
            ranked = sorted(ok, key=lambda n: (-ok[n], n))
            out |= set(ranked[:top])
            if top and len(ranked) >= top:
                edge = ranked[top - 1]
                out |= {n for n in ranked[top:] if ok[n] >= ok[edge] - sds[edge]}
        except Exception as exc:  # noqa: BLE001 - the prune view never fails on a heuristic
            self.store.log("prune_models_error", error=repr(exc)[:300])
        out |= self._prune_challengers(data, statuses)
        return {n for n in out if n in data.nodes}

    def _prune_challengers(self, data: CD.Ledger, statuses: Mapping[str, Any]) -> set[str]:
        """10-22 ops review: the live challengers whose checkpoints must outlive their screening. The alt-base ranking
        above orders nodes by their OWN increment over the parent, so a strong child of a strong parent (a node: post ~.6823,
        data echoing x2) ranks low and settled; here the order is the ABSOLUTE composite level of the node's own screening
        seeds: the best ``prune.challenger_top`` live (SCREENING / PASSED / TESTING) current-epoch nodes, every one within
        one sd of the best non-incumbent, and every node in TEST (a wave point is a knob design, not a challenger). A view without a posterior protects only the TEST nodes."""
        from loop.core import model as M
        from loop.core.acquire import wave_node
        live = {n for n, st in statuses.items() if status_name(st) in ("SCREENING", "PASSED", "TESTING")
                and n in data.nodes and data.nodes[n].base == data.epoch and not wave_node(data.nodes[n])}
        out = {n for n in live if status_name(statuses[n]) == "TESTING"}
        inc, top = incumbent_of(statuses), max(0, int(getattr(self.ctx.cfg.prune, "challenger_top", 8)))
        try:
            post = self.post
            noise, w = post.fit(data.epoch)
            lvl: dict[str, tuple[float, float]] = {}
            for n in live - {inc}:
                if post.by.get(n):
                    mean, k = M.parent_mean(post.by[n], M.COMPOSITE, w)
                    lvl[n] = (mean, math.sqrt(max(0.0, float(noise.sigma2[M.COMPOSITE])) / max(1, k)))
            ranked = sorted(lvl, key=lambda n: (-lvl[n][0], n))
            if ranked and top:
                best_m, best_sd = lvl[ranked[0]]
                out |= set(ranked[:top]) | set([n for n in ranked[top:] if lvl[n][0] >= best_m - best_sd][:3 * top])
        except Exception as exc:  # noqa: BLE001 - the prune view never fails on a heuristic
            self.store.log("prune_challengers_error", error=repr(exc)[:300])
        return out

    def _prune(self, data: CD.Ledger, statuses: Mapping[str, Any]) -> list[str]:
        """``loop.prune`` (periodic, ``prune.every_min``; a background thread live): checkpoints of closed runs,
        stale research branches, node worktrees of closed / void nodes -- never anything an open node, block,
        wave or port references."""
        acts: list[str] = []
        out = self.guarded("prune", "prune pass", lambda: self.pruner.tick(self.prune_view(data, statuses)), acts,
                           default=[])
        return acts + list(out or [])

    def llm_state(self) -> str:
        """ok | paused <kind> until <utc> | probing | draining | held (disk low)."""
        p = self.store.get_state(R.K_LLM_PAUSE) or {}
        if p.get("active"):
            return f"paused {p.get('kind')} until {R.utc_now(float(p.get('until', 0)))}"
        if self.research.draining:
            return "draining"
        if self.research.disk_low:
            return "held (disk low)"
        if getattr(self, "selfcheck_hold", ""):
            return "held (selfcheck failed)"
        if self.research.breaker:
            return "held (research breaker)"
        return "probing" if p.get("probing") else "ok"

    def _gap_hb(self) -> str:
        """`` gap=+.0042±.0011 mode=target|incumbent``: the distance to the external target (acquire.gap_se) and the
        acquisition mode in force (its hysteresis state); '' without a target or a core that has one."""
        A = getattr(self.core, "A", None)
        if A is None or self.data is None or not A.target_on():
            return ""

        def f(x: float, sign: bool = False) -> str:
            t = f"{x:+.4f}" if sign else f"{x:.4f}"
            return t.replace("0.", ".", 1)
        try:
            got = A.gap_se(self.data, incumbent_of(self.statuses))
        except Exception as exc:  # noqa: BLE001 - the heartbeat never raises
            return f" gap=? ({type(exc).__name__})"
        mode = "target" if A.TARGET_ACTIVE else "incumbent"
        return f" gap={f(got[0], True)}±{f(got[1])} mode={mode}" if got is not None else f" gap=? mode={mode}"

    def heartbeat(self, n: int, dur: float, error: str = "") -> str:
        """One ``hb`` line: tick, duration, free GPU slots, research in flight / free, open tests / waves / port
        sessions, incumbent, LLM pause / breaker, lock guard, disk free; ``error``: the tick raised."""
        try:
            blocks, gpu = self.blocks(), self.book.free()
        except Exception:  # noqa: BLE001 - the heartbeat never raises
            blocks, gpu = [], -1
        tests = sum(1 for b in blocks if b.pool == CD.TEST and b.open)
        waves = len(self._open_waves(self.data)) if self.data is not None else 0
        ports = self.store.get_state(self.K_PORT_SESSIONS) or {}
        pend = sum(1 for r in ports.values() if r.get("status") == "pending")
        started = sum(1 for r in ports.values() if r.get("status") == "started")
        lock = getattr(self.meta, "lock_state", lambda: "n/a")()
        q = len(self.store.get_state(K_QUARANTINE) or {})
        disk = " ".join(f"{Path(p).name or p}={g:.0f}GB" for p, g in self.disk.items()) or "?"
        lc, bar = self.ctx.cfg.labless, ""
        from loop.core import bar as BR
        if BR.configured(lc) and self.data is not None:     # the external bar (loop.core.bar), the best candidate
            try:
                commits = {n.id: n.commit for n in self.store.nodes() if getattr(n, "commit", "")}
                bar = " " + BR.summary(self.data, lc, commits, incumbent_of(self.statuses))
            except Exception as exc:  # noqa: BLE001 - the heartbeat never raises
                bar = f" bar={lc.bar_node} (failed: {exc!r})"[:120]
        self._pool_sanity()
        return (f"hb tick={n} dur={dur:.1f}s gpu_free={gpu}{self._pools_hb()}{self._alloc_hb()} research={len(self.research.inflight)} in flight/"
                f"{self.research.free()} free (+{len(self.research.requeue)} queued){self._throttle_hb()}{self._smokecap_hb()} open_tests={tests} "
                f"open_waves={waves} port_sessions={pend} pending/{started} started "
                f"incumbent={incumbent_of(self.statuses)}{bar}{self._gap_hb()} llm={self.llm_state()} lock={lock} disk={disk}"
                + (f" quarantined={q}" if q else "") + self.timing_text() + (f" TICK_ERROR {error}" if error else ""))

    def _throttle_hb(self) -> str:
        """`` throttle=off|on(why; trickle k)``: the brief throttle's last verdict (``alloc.backlog_max`` /
        ``backlog_age_h``; ``backlog_trickle`` sessions may still run while it holds)."""
        try:
            from loop.core import alloc as AL
            why = str(self.store.get_state(AL.K_THROTTLE) or "")
            k = int(getattr(AL.cfg_of(self.ctx), "backlog_trickle", 0) or 0)
        except Exception:  # noqa: BLE001 - the heartbeat never raises
            return ""
        return f" throttle=on({why[:60]}; trickle {k})" if why else " throttle=off"

    def _smokecap_hb(self) -> str:
        """`` smokecap=open|probe|closed[(why)]``: whether a new research session may start (``adapter.smoke_capacity``)."""
        try:
            st, why = self.research.smoke_capacity()
        except Exception:  # noqa: BLE001 - the heartbeat never raises
            return ""
        return f" smokecap={st}" + (f"({why[:70]})" if why else "")

    def _pool_sanity(self) -> None:
        """One ALERT when our active TRAINING runs pass TWICE the slots in force (``limits(gate=False)``) summed over
        the pools (live 10-21: ``idle:2r/51p 54/5``, a burst past the cap); re-armed once back in range. Smoke jobs, the
        smoke reserve and the smoke-pending gate are not the cap's (11-09 00:16Z false alarm: ``primary:7r/0p 11/8,
        idle:0r/9p 12/3`` = 16 runs + 7 smoke slots vs a smoke-gated 11)."""
        try:                                 # training runs only: smoke jobs / the smoke reserve sit outside the cap
            used, lim = len(self.book.active()), sum(self.book.limits(gate=False).values())
        except Exception:  # noqa: BLE001 - the heartbeat never raises
            return
        over = used > 2 * max(lim, 1)
        if over and not getattr(self, "_pool_alerted", False):
            print(f"{R.utc_now(self.clock())} ALERT: pool runs held {used} > 2x the {lim} slots in force -- jobs queued past the pool cap "
                  f"(stale-pending / burst submit); check squeue", file=sys.stderr, flush=True)
        self._pool_alerted = over

    def _alloc_hb(self) -> str:
        """`` alloc=llm<live>/<cap>runs share<s> throttle=<why>``: v3-c7i's GPU split in force this tick (LLM-idea
        unit runs live vs the soft cap) and the brief throttle verdict ('' when the allocator is off / not run)."""
        try:
            sp = getattr(self.core, "split", None) or {}
            why = self.store.get_state("alloc.throttle") or ""
        except Exception:  # noqa: BLE001 - the heartbeat never raises
            return ""
        out = (f" alloc=llm{sp.get('llm_live', '?')}/{sp.get('llm_cap_runs', '?')}runs share{sp.get('share', 0):.2f}"
               if sp else "")
        return out + (f" throttle={str(why).split(':')[0][:40]}" if why else "")

    def _pools_hb(self) -> str:
        """`` pools=<pool>:<running>r/<pending>p <used>/<limit> ...``: per pool, our runs SLURM reported running /
        pending at the last poll (``?`` before the first) and the slots actually held against the limit in force."""
        try:
            st, used, lim = self.book.pool_states, self.book.used(), self.book.limits()
            note = self.book.cap_note()
        except Exception:  # noqa: BLE001 - the heartbeat never raises
            return ""
        parts = []
        for p in self.ctx.cfg.pools:
            r, q = (st or {}).get(p.name, ("?", "?")) if st is not None else ("?", "?")
            parts.append(f"{p.name}:{r}r/{q}p {used.get(p.name, 0)}/{lim.get(p.name, p.slots)}")
        return (" pools=" + ",".join(parts) + note) if parts else ""

    def _start(self, acts: list[str], now: float) -> None:
        """Once per process: reap orphan worktrees, the root node, re-issue the research a restart killed,
        the one-time outage repair."""
        self._started = True
        if not self.dry:
            self.release_stale_quarantine(acts)
        acts += [f"reaped orphan worktree {p}" for p in getattr(self.adapter, "reap_orphans", lambda: [])()]
        acts += [f"reaped orphan graft worktree {p}" for p in self._reap_grafts()]
        if self.store.get_node(ROOT_NODE_ID) is None:
            acts.append(f"root node at {self.store.ensure_root(self.adapter.root_commit()).commit}")
        self.research.restore(acts)
        R.repair_outage(self.store, now, acts)

    def _reap_grafts(self) -> list[str]:
        """``<worktrees>/_graft/*`` a killed graft left behind (a graft reaps its own; at start none is running)."""
        wt, reap = getattr(self.adapter, "worktrees", None), getattr(self.adapter, "reap_worktree", None)
        if not isinstance(wt, (str, Path)) or reap is None:
            return []
        from loop.contracts import Worktree
        out = []
        for d in sorted((Path(wt) / "_graft").glob("*")):
            try:
                reap(Worktree(str(d), "", ""))
                out.append(str(d))
            except Exception as exc:  # noqa: BLE001 - the prune pass / next start tries again
                self.store.log("tick_call_error", key="reap_grafts", error=repr(exc)[:300])
        return out

    def _void_blocks(self, data: CD.Ledger, acts: list[str]) -> None:
        """Historical blocks migrated before ``seeds.VOID_NO_RUNS`` that are open with no run on any seed (a draw never
        run): closed as VOID at their draw, logged ``seed_block_void`` -- not left open forever, not ignored."""
        used = {r.seed for r in data.runs}
        for b in self.blocks():
            if b.open and b.id.startswith("hist-") and not used.intersection(b.seeds):
                self._update_block(b.id, acts, f"VOID ({b.purpose}: no run on {list(b.seeds)}): closed",
                                   closed_ts=b.drawn_ts)
                self.store.log("seed_block_void", block=b.id, pool=b.pool, seeds=list(b.seeds), purpose=b.purpose,
                               reason="open historical block with no runs on its seeds: closed as VOID at its draw")

    # ------------------------------------------------------------------ ports (cutover step 12)
    K_PORT_SESSIONS = "ports.sessions"             # old id -> {status pending|started|failed, request, parent, how, tries}
    PORT_SESSION_TRIES = 3                         # a port session that ends without a node is re-issued this often

    def port_graft(self, old: str, data: CD.Ledger) -> list[str]:
        """Port ``old`` onto the current base (``loop.cutover.port_node``): applied -> a QUEUED ``port_of`` node whose
        screening runs this tick launches; conflict -> a REJECTED port node (not offered again) + one researcher
        session (queued until a researcher slot is free); a change the base already holds -> REJECTED, no session."""
        from loop import cutover as CO
        if not hasattr(self.adapter, "graft_port"):
            return [f"would port {old} onto base {data.epoch[:7]} (the adapter cannot graft)"]
        p = CO.port_node(self.store, self.adapter, data, old, self.clock())
        if p.how == "exists":
            return []
        if p.ok:
            self._ported.append((p.node, p.parent))
            return [f"port {old} -> {p.node} on {p.parent} @ {p.commit[:7]} (graft of {p.anchor}..{old})"]
        out = [f"port {old} onto {p.parent or 'base ' + data.epoch[:7]}: {p.how}" + (f" -> REJECTED {p.node}" if p.node
                                                                                    else "")]
        if p.parent:                                    # v3-c7a A6: a key the base lacks -> its prerequisite first
            self._missing_prereqs(p.parent, p.how, port=old, witness=old)
        if p.how.startswith("conflict") and self.port_sessions:
            book = dict(self.store.get_state(self.K_PORT_SESSIONS) or {})
            if old not in book:
                book[old] = {"status": "pending", "parent": p.parent, "anchor": p.anchor, "how": p.how}
                self.store.set_state(self.K_PORT_SESSIONS, book)
                out.append(f"port session for {old} queued (one researcher session re-implements it on {p.parent})")
        return out

    def _alt_root_ports(self, data: CD.Ledger, statuses: Mapping[str, Any], now: float, acts: list[str]) -> list[RunSpec]:
        """A root-based child (alt_src=root) with a positive own increment over root (m > 0, m > 1 se, enough paired
        seeds) is ported onto the incumbent through the port machinery (its OWN change over root, ``onto`` the
        incumbent; ``port_src=alt_root``), <= 1 open at a time; shadow: listed only. Returns its screening runs."""
        from loop import cutover as CO
        from loop.core import altbase as AB
        inc = incumbent_of(statuses)
        k = int(getattr(getattr(self.core, "costs", None), "idea_runs", 3))
        due = AB.port_due(self.ctx, self.store, data, self.post, statuses, inc, k)
        specs: list[RunSpec] = []
        for old in due:
            if self.dry:
                acts.append(f"would auto-port root child {old} onto the incumbent {inc} (port_src=alt_root)")
                continue
            if not hasattr(self.adapter, "graft_port"):
                continue
            p = CO.port_node(self.store, self.adapter, data, old, now, onto=inc, tag=AB.PORT_TAG)
            if p.how == "exists":
                continue
            acts.append(f"alt-root port {old} -> {p.node or '-'} on {inc}: {p.how}")
            self.store.log("alt_root_port", node_id=old, port=p.node, onto=inc, how=p.how)
            if p.ok and hasattr(self.core, "screen_specs"):
                specs += self.core.screen_specs(p.node, p.parent, k, data, self.blocks(), self.ctx.cfg)
        return specs

    def _same_mech(self, inc: str, old: str) -> Any:
        """``nid -> why | ''``: ``nid``'s change is the new incumbent's MECHANISM (same family, and its change over
        ``old`` shares a non-config source file with ``inc``'s) -- e.g. a second sampler on a sampling incumbent."""
        si, so = self.store.get_node(inc), self.store.get_node(old)
        cf = getattr(self.adapter, "changed_files", None)
        conf = (".yaml", ".yml", ".json", ".txt", ".md")

        def files(a: Any, b: Any) -> set[str]:
            try:
                return {f for f in (cf(a.commit, b.commit) if cf else ()) if not f.endswith(conf)}
            except Exception:  # noqa: BLE001 - unknown files: judged by family alone
                return set()
        mine = files(so, si) if si is not None and so is not None else set()

        def why(nid: str) -> str:
            n = self.store.get_node(nid)
            if n is None or si is None or not si.family or n.family != si.family:
                return ""
            both = mine & files(so, n) if mine else set()
            if mine and not both:
                return ""
            return (f"same mechanism as {inc} (family {si.family}"
                    + (f"; both change {', '.join(sorted(both))}" if both else "") + ")")
        return why

    def _reroot_ports(self, data: CD.Ledger, statuses: Mapping[str, Any], now: float, acts: list[str]) -> list[RunSpec]:
        """RE-ROOT PORTS (``test.reroot_port``; operator 11-10): after an incumbent switch, the old incumbent's best
        descendants (:func:`altbase.reroot_plan`) are grafted -- their whole change over the OLD incumbent -- onto the
        new one (``port_src=reroot``), at most ``reroot_port_k`` per switch, booked in ``reroot.ports``. Every open
        re-root port's missing screen runs (up to ``idea_runs``) are returned each tick, so they go ahead of the ranked
        queue until screened. Shadow (dry): listed only."""
        from loop import cutover as CO
        from loop.core import altbase as AB
        tc = getattr(self.ctx.cfg, "test", None)
        if not getattr(tc, "reroot_port", False):
            return []
        inc = incumbent_of(statuses)
        old = AB.previous_incumbent(data, inc)
        k = int(getattr(getattr(self.core, "costs", None), "idea_runs", 3))
        book = dict(self.store.get_state(AB.K_REROOT) or {})
        ent = dict(book.get(inc) or {"from": old, "ts": now, "ported": {}, "skipped": {}})
        specs: list[RunSpec] = []
        ent.setdefault("failed", {})
        ent.setdefault("crossed", {})
        room = int(tc.reroot_port_k) - len(ent["ported"])
        cross_room = (int(getattr(tc, "reroot_cross_k", 2)) - sum(
            1 for v in ent["crossed"].values() if not isinstance(v, dict) or v.get("status") != "contained")) \
            if getattr(tc, "reroot_cross", False) else 0
        open_skips = [n for n in ent["skipped"] if n not in ent["crossed"] and n not in (ent.get("cross_declined") or {})]
        if old and old != inc and (room > 0 or (cross_room > 0 and open_skips)):   # no plan once all is decided
            cands: dict[str, Any] = {}
            picks, skipped = AB.reroot_plan(
                data, self.store, statuses, inc, old, p_min=float(tc.reroot_port_p),
                min_seeds=int(tc.reroot_port_min_seeds),
                eb_min_nodes=int(getattr(getattr(self.ctx.cfg, "labless", None), "eb_min_nodes", 5)),
                same_mech=self._same_mech(inc, old), ctx=self.ctx, crossable=cands)
            done_lin = {v.get("lineage") for v in ent["ported"].values() if isinstance(v, dict)}

            def save() -> None:
                if not self.dry:
                    book[inc] = ent
                    self.store.set_state(AB.K_REROOT, book)
            for nid, why in skipped.items():
                if nid not in ent["skipped"]:
                    ent["skipped"][nid] = why
                    acts.append(f"re-root port: skip {nid} ({why})")
                    self.store.log("reroot_port_skip", node_id=nid, onto=inc, frm=old, why=why)
                    save()
            if cross_room > 0:              # a skip that is not really the same mechanism: crossed, not dropped
                self._reroot_cross(data, inc, old, ent, cands, done_lin, cross_room, now, acts, save)
            for pk in picks:
                if room <= 0:
                    break
                if pk.node in ent["ported"] or pk.node in ent["failed"] or pk.lineage in done_lin:
                    continue
                if self.dry or not hasattr(self.adapter, "graft_port"):
                    acts.append(f"would re-root port {pk.node} onto {inc} (gain over {old} {pk.m:+.5f}, P {pk.p:.2f}, "
                                f"{pk.n} seeds)")
                    continue
                try:
                    p = CO.port_node(self.store, self.adapter, data, pk.node, now, onto=inc, tag=AB.REROOT_TAG,
                                     src=old, lane_tags=False)
                except Exception as exc:  # noqa: BLE001 - one bad graft never stops the others (review 11-10)
                    ent["failed"][pk.node] = f"error: {exc!r}"[:300]
                    self.store.log("reroot_port_error", node_id=pk.node, onto=inc, error=repr(exc)[:300])
                    save()
                    continue
                sn = self.store.get_node(p.node) if p.node else None
                ok = p.how == "applied" or (p.how == "exists" and sn is not None
                                            and getattr(sn.status, "value", str(sn.status)) in ("queued", "screening"))
                acts.append(f"re-root port {pk.node} (lineage {pk.lineage}; +{pk.m:.5f} over {old}, P {pk.p:.2f}, "
                            f"{pk.n} seeds) -> {p.node or '-'} on {inc}: {p.how}")
                self.store.log("reroot_port", node_id=pk.node, port=p.node, onto=inc, frm=old, how=p.how,
                               m=pk.m, p=pk.p, n=pk.n, lineage=pk.lineage)
                if not ok:                  # contained / conflict: no K slot spent, the lineage's next best may go
                    ent["failed"][pk.node] = p.how[:300]
                    if p.how.startswith("conflict"):
                        self._missing_prereqs(inc, p.how, port=pk.node)
                    save()
                    continue
                ent["ported"][pk.node] = {"port": p.node, "how": "applied", "lineage": pk.lineage,
                                          "m": round(pk.m, 6), "p": round(pk.p, 4), "n": pk.n, "ts": now}
                done_lin.add(pk.lineage)
                room -= 1
                save()
        if self.dry or not hasattr(self.core, "screen_specs"):
            return specs
        opened = [v.get("port") if isinstance(v, dict) and v.get("how") == "applied" else ""
                  for v in ent["ported"].values()]
        opened += [v.get("node") or "" for v in ent["crossed"].values() if isinstance(v, dict)]   # re-root crossovers
        for pid in opened:                                  # every open re-root port's missing screens, each tick
            sn = self.store.get_node(pid) if pid else None
            if sn is None or status_name(statuses.get(pid, "")) in CLOSED \
                    or getattr(sn.status, "value", str(sn.status)) not in ("queued", "screening"):
                continue
            have = {r.seed for r in self.store.runs(node_id=pid) if r.stage is SEED_STAGE}   # a void retries in place
            if len(have) < k:
                specs += self.core.screen_specs(pid, inc, k - len(have), data, self.blocks(), self.ctx.cfg)
        return specs

    REROOT_XOVER_TRIES = 3                      # a re-root crossover session that ends without a node is re-issued

    def _reroot_cross(self, data: CD.Ledger, inc: str, old: str, ent: dict[str, Any], cands: Mapping[str, Any],
                      done_lin: set, room: int, now: float, acts: list[str], save: Any) -> None:
        """RE-ROOT CROSSOVERS (``test.reroot_cross``): a re-root port candidate skipped only as the new incumbent's
        family + shared file (``reroot.ports`` ``skipped``; its evidence cleared the port bar: ``cands``) that is NOT the
        same mechanism (:func:`altbase.true_same`) is crossed with ``inc``: the candidate's whole change over ``old``
        is grafted onto ``inc`` -- clean: a QUEUED code-made crossover node (parent ``inc``, second parent the
        candidate); a conflict: ONE researcher crossover session (:meth:`_reroot_cross_sessions`) that merges both.
        At most ``room`` (``reroot_cross_k`` minus those booked), best EB gain first, one per lineage (a ported lineage
        included), booked in ``ent['crossed']`` BEFORE anything is made (a restart never makes a second). The skips of a
        re-root made before this deploy are crossed on the next tick the same way (catch-up). Shadow: listed only."""
        import hashlib
        from loop.contracts import Node as StoreNode, NodeStatus, Operator, Strategy
        from loop.core import altbase as AB
        from loop.core.alloc import _next_id
        from loop.core.ideation import REROOT_XOVER_PREFIX
        crossed, declined = ent["crossed"], ent.setdefault("cross_declined", {})
        lins = set(done_lin) | {v.get("lineage") for v in crossed.values() if isinstance(v, dict)}
        for pk in sorted(cands.values(), key=lambda x: (-x.m, x.node)):
            if pk.node in crossed or pk.node not in ent["skipped"]:
                continue
            if pk.lineage in lins or (same := AB.true_same(data, self.store, inc, pk.node)):
                if pk.node not in declined and not self.dry:     # decided: never re-planned for it
                    declined[pk.node] = f"lineage {pk.lineage} crossed / ported" if pk.lineage in lins else same
                    save()
                continue
            if room <= 0:                                        # cap reached: undecided ones wait (no slot)
                continue
            line = (f"re-root crossover {inc} x {pk.node} (lineage {pk.lineage}; +{pk.m:.5f} over {old}, P {pk.p:.2f}, "
                    f"{pk.n} seeds; port skipped: {str(ent['skipped'].get(pk.node, ''))[:120]})")
            lins.add(pk.lineage)
            room -= 1
            if self.dry or not hasattr(self.adapter, "graft_port"):
                acts.append("would make " + line)
                continue
            rec: dict[str, Any] = {"lineage": pk.lineage, "m": round(pk.m, 6), "p": round(pk.p, 4), "n": pk.n,
                                   "from": old, "ts": now, "status": "pending", "node": "",
                                   "request": f"{REROOT_XOVER_PREFIX}{inc}-{pk.node}"}
            made = next((n.id for n in self.store.nodes() if n.parent_id == inc and n.second_parent_id == pk.node
                         and f"{AB.REROOT_CROSS_TAG}={pk.node}" in (n.notes or "")), "")
            si, so, sc = self.store.get_node(inc), self.store.get_node(old), self.store.get_node(pk.node)
            how = "exists" if made else ""
            if not made and si is not None and so is not None and sc is not None:
                try:
                    sha, how = self.adapter.graft_port(so.commit, sc.commit, si.commit, f"rx-{inc}-{pk.node}")
                except Exception as exc:  # noqa: BLE001 - a graft error: the researcher session merges it
                    sha, how = "", f"error: {exc!r}"[:300]
                if how == "applied" and sha and sha != si.commit:
                    made = _next_id(self.store)
                    self.store.put_node(StoreNode(
                        id=made, task_id=si.task_id, commit=sha, parent_id=inc, strategy=Strategy.CROSSOVER,
                        operator=Operator.CROSSOVER, family=sc.family or si.family or "crossover",
                        mechanism_tag=f"pair_{inc}_{pk.node}",
                        diff_hash=hashlib.sha256(f"{si.commit}\n{sha}".encode()).hexdigest(),
                        status=NodeStatus.QUEUED, second_parent_id=pk.node, created_ts=now, updated_ts=now,
                        notes=f"pair_of={inc},{pk.node} graft_xover {AB.REROOT_CROSS_TAG}={pk.node} reroot_from={old}"))
                elif how == "contained":
                    rec.update(status="contained", how=how)
                    room += 1                           # inc already holds it: no crossover slot spent
            if made:
                rec.update(status="done", node=made, how=how or "graft")
            elif rec["status"] == "pending":
                rec["how"] = (how or "no graft")[:300]
            crossed[pk.node] = rec
            save()
            self.store.log("reroot_port_cross", node_id=pk.node, onto=inc, frm=old, cross=rec["node"],
                           status=rec["status"], how=str(rec.get("how", ""))[:300], m=pk.m, p=pk.p, n=pk.n,
                           lineage=pk.lineage)
            acts.append(line + (f" -> crossover node {made} (code graft)" if made and how != "exists" else
                                f" -> {made} (exists)" if made else
                                " -> already in the incumbent" if rec["status"] == "contained" else
                                f" -> researcher crossover session queued (graft: {rec['how'][:120]})"))

    def _reroot_cross_brief(self, inc: str, cand: str, old: str, rec: Mapping[str, Any], why: str) -> str:
        si, sc, so = self.store.get_node(inc), self.store.get_node(cand), self.store.get_node(old)
        try:
            diff = self.adapter.diff_commits(so.commit, sc.commit)[:6000] \
                if hasattr(self.adapter, "diff_commits") and so is not None and sc is not None else ""
        except Exception:  # noqa: BLE001 - the brief without the diff
            diff = ""
        ti, tc = getattr(si, "mechanism_tag", "") or "?", getattr(sc, "mechanism_tag", "") or "?"
        return (f"CROSSOVER (re-root): merge {cand} ({tc}) into the incumbent {inc} ({ti}). {cand} was built on the "
                f"previous incumbent {old} and gained {float(rec.get('m') or 0):+.4f} over it (P(gain>0) "
                f"{float(rec.get('p') or 0):.2f} on {rec.get('n')} paired seeds). Its port onto {inc} was skipped "
                f"({why}), but the two are DIFFERENT mechanisms and plausibly stack; its code graft onto {inc} did not "
                f"apply ({str(rec.get('how') or '')[:200]}).\n"
                f"Start from {inc}'s code and config and add {cand}'s mechanism on top -- its whole change over {old}, "
                f"below. Where both touch the same code, make both mechanisms work together: keep {inc}'s mechanism "
                f"intact, do not just pick one side, and add nothing else.\n"
                + (f"The change ({old}..{cand}):\n```diff\n{diff}\n```" if diff else ""))

    def _reroot_cross_sessions(self, acts: list[str]) -> None:
        """Start the queued re-root crossover sessions (``reroot.ports`` ``crossed``, status pending) of the CURRENT
        incumbent while researcher slots are free -- one request id per (incumbent, candidate), so a restart never
        starts a second; a started session lost without a node is re-issued up to REROOT_XOVER_TRIES times, a node made
        from it (``request=<id>``) is booked. Shadow: listed only."""
        from loop.contracts import CandidateRequest, Operator, Strategy
        from loop.core import altbase as AB
        inc = incumbent_of(self.statuses) if self.statuses else ""
        book = dict(self.store.get_state(AB.K_REROOT) or {})
        ent = dict(book.get(inc) or {}) if inc else {}
        crossed = dict(ent.get("crossed") or {})
        if not crossed:
            return
        changed = False
        live = {r.id for r in [*self.research.inflight.values(), *self.research.requeue]}
        cutoff = self._tick_ts if self._tick_ts is not None else -float("inf")
        made = {}
        if any(isinstance(r, dict) and r.get("status") in ("pending", "started") for r in crossed.values()):
            made = {t[len("request="):]: n.id for n in self.store.nodes() for t in (n.notes or "").split()
                    if t.startswith("request=")}
        for cand, rec in list(crossed.items()):
            if not isinstance(rec, dict) or rec.get("status") not in ("pending", "started"):
                continue
            rid = str(rec.get("request") or "")
            if rid in made:                            # its session made the node (a restart may have lost the mark)
                crossed[cand] = {**rec, "status": "done", "node": made[rid]}
                self.store.log("reroot_cross_made", node_id=made[rid], cand=cand, onto=inc, request_id=rid)
                changed = True
                continue
            if rec["status"] == "started":
                if rid in live or float(rec.get("started", -float("inf"))) >= cutoff:
                    continue
                if int(rec.get("tries", 1)) >= self.REROOT_XOVER_TRIES:
                    crossed[cand] = {**rec, "status": "failed"}
                    self.store.log("reroot_cross_failed", node_id=cand, onto=inc, request_id=rid,
                                   tries=int(rec.get("tries", 1)))
                    acts.append(f"re-root crossover session {inc} x {cand} failed {self.REROOT_XOVER_TRIES} times")
                    changed = True
                    continue
                rec = {**rec, "status": "pending"}
                crossed[cand], changed = rec, True
            if self.dry:
                acts.append(f"would start a re-root crossover session {inc} x {cand}"
                            + ("" if self.research.capacity() > 0 else f" (waits: LLM {self.llm_state()})"))
                continue
            if self.research.free() <= 0:
                break
            if rid in live:
                continue
            req = CandidateRequest(id=rid, task_id=self.ctx.task_id, strategy=Strategy.CROSSOVER,
                                   operator=Operator.CROSSOVER, island="main", parent_ids=(inc, cand),
                                   brief=self._reroot_cross_brief(inc, cand, str(rec.get("from") or ent.get("from")),
                                                                  rec, str((ent.get("skipped") or {}).get(cand, ""))),
                                   brief_kind="crossover")
            self.research.start(req, None, lane="port")
            crossed[cand] = {**rec, "status": "started", "started": self.clock(),
                             "tries": int(rec.get("tries", 0)) + 1}
            changed = True
            acts.append(f"researcher started: {rid} re-root crossover {inc} x {cand}")
        if changed and not self.dry:
            cur = dict(self.store.get_state(AB.K_REROOT) or {})
            e = dict(cur.get(inc) or ent)
            e["crossed"] = {**dict(e.get("crossed") or {}), **crossed}
            cur[inc] = e
            self.store.set_state(AB.K_REROOT, cur)

    def _missing_prereqs(self, car: str, how: str, *, knob: Any = None, port: str = "", witness: str = "") -> None:
        """A graft onto ``car`` that conflicted on config keys ``car`` lacks (``ModuleCore.conflict_keys``): each
        booked as a prerequisite (``ModuleCore.note_missing``) for the knob / port that waits on it."""
        keys = ModuleCore.conflict_keys(how)
        if not keys or not getattr(getattr(self.ctx.cfg, "factorial", None), "prereq_ports", True):
            return
        cfg = ModuleCore._config(self, car)
        for k in keys:
            if cfg and k not in cfg:
                ModuleCore.note_missing(self, car, k, knob=knob, port=port, witness=witness)

    def reject_port(self, old: str, data: CD.Ledger, reason: str) -> None:
        """A port that kept raising (quarantined): a REJECTED port node with the reason, so it is not offered again."""
        from loop import cutover as CO
        from loop.contracts import Node, NodeStatus
        nid = CO.port_id(old, data.epoch)
        o, root = self.store.get_node(old), self.store.get_node(data.base_root) if data.base_root else None
        if self.store.get_node(nid) is not None or o is None or root is None:
            return
        now = self.clock()
        self.store.put_node(Node(id=nid, task_id=self.ctx.task_id, commit=root.commit, parent_id=root.id,
                                 strategy=o.strategy, operator=o.operator, family=o.family,
                                 mechanism_tag=o.mechanism_tag, diff_hash=f"port-error-{data.epoch[:12]}-{old}",
                                 status=NodeStatus.REJECTED, hypothesis_id=o.hypothesis_id, levers=o.levers,
                                 reject_reason=f"port error: {reason}"[:500], created_ts=now, updated_ts=now,
                                 notes=f"port_of={old} port_failed port_error"))
        self.store.log("port_conflict", node_id=old, port=nid, reason=f"port error: {reason}"[:500])

    def _reconcile_port_sessions(self, book: dict[str, Any], acts: list[str]) -> bool:
        """A started port session that is neither in flight nor queued and left no node (``request=<id>``) was
        lost (an error, a failed result, a restart past MAX_RESTARTS, a BackendUnavailable twice): pending again,
        up to PORT_SESSION_TRIES sessions, then failed."""
        live = {r.id for r in [*self.research.inflight.values(), *self.research.requeue]}
        cutoff = self._tick_ts if self._tick_ts is not None else -float("inf")   # started this tick: not lost yet
        started = {old: rec for old, rec in book.items() if rec.get("status") == "started"
                   and rec.get("request") not in live and float(rec.get("ts", -float("inf"))) < cutoff}
        if not started:
            return False
        made = {t[len("request="):] for n in self.store.nodes() for t in (n.notes or "").split()
                if t.startswith("request=")}
        changed = False
        for old, rec in started.items():
            if rec.get("request") in made:
                book[old] = {**rec, "status": "done"}
            elif int(rec.get("tries", 1)) < self.PORT_SESSION_TRIES:
                book[old] = {**rec, "status": "pending", "tries": int(rec.get("tries", 1))}
                self.store.log("port_session_retry", node_id=old, request_id=rec.get("request"),
                               tries=int(rec.get("tries", 1)))
                acts.append(f"port session for {old} ended without a node: re-queued "
                            f"(try {int(rec.get('tries', 1)) + 1}/{self.PORT_SESSION_TRIES})")
            else:
                book[old] = {**rec, "status": "failed"}
                self.store.log("port_session_failed", node_id=old, request_id=rec.get("request"),
                               tries=int(rec.get("tries", 1)))
                acts.append(f"port session for {old} failed {self.PORT_SESSION_TRIES} times: given up")
            changed = True
        return changed

    def _port_sessions(self, acts: list[str]) -> None:
        """Start the queued researcher port sessions while researcher slots are free (shadow: listed only); a
        session lost without a node is re-issued (:meth:`_reconcile_port_sessions`)."""
        from loop import cutover as CO
        from loop.contracts import CandidateRequest, Operator, Strategy
        from loop.core.ideation import PORT_PREFIX
        book = dict(self.store.get_state(self.K_PORT_SESSIONS) or {})
        changed = False if self.dry else self._reconcile_port_sessions(book, acts)
        for old, rec in book.items():
            if rec.get("status") != "pending":
                continue
            if self.dry:
                acts.append(f"would start a researcher port session for {old} on {rec['parent']}"
                            + ("" if self.research.capacity() > 0 else f" (waits: LLM {self.llm_state()})"))
                continue
            if self.research.free() <= 0:
                break
            p = CO.Ported(old, "", rec["parent"], rec["anchor"], rec["how"])
            req = CandidateRequest(id=f"{PORT_PREFIX}{old}", task_id=self.ctx.task_id, strategy=Strategy.EXPLOIT,
                                   operator=Operator.IMPROVE, island="main", parent_ids=(rec["parent"],),
                                   brief=CO.port_brief(self.store, self.adapter, p), brief_kind="port")
            self.research.start(req, None, lane="port")
            book[old] = {**rec, "status": "started", "request": req.id, "ts": self.clock(),
                         "tries": int(rec.get("tries", 0)) + 1 if "request" in rec else 1}
            changed = True
            acts.append(f"researcher started: {req.id} port of {old} onto {rec['parent']}")
        if changed:
            self.store.set_state(self.K_PORT_SESSIONS, book)

    # ------------------------------------------------------------------ probe ill-conditioning repairs (v3-c7a A4)
    K_REPAIR = "repair.sessions"     # node -> {status pending|started|done|failed, request, seeds, evidence, ts, tries}
    K_NO_REPAIR = "repair.not_booked"   # node -> ts: ill-conditioned, but its scorable evidence mean <= 0 (logged once)
    REPAIR_MAX_OPEN = 2              # repair sessions in flight at once (ideation.repair_max_open)
    REPAIR_TRIES = 2

    K_ILL_FLAKE = "repair.ill_flakes"   # {run id: ts}: ill-conditioned crashes judged a FLAKE (re-run / re-drawn once)
    ILL_MIN_CRASHES = 2                 # v3 review K4: an idea defect needs >= this many ill-conditioned crashes ...
    ILL_MIN_FRAC = 1.0 / 3.0            # ... making >= this share of the node's finished runs (or a readout diff)

    def _ill_nodes(self, data: CD.Ledger) -> dict[str, list[int]]:
        """Current-epoch nodes (not the incumbent, not a wave point) with a crash classed :data:`ILL_CONDITIONED`:
        node -> the seeds it crashed on (crashes already judged a flake left out)."""
        from loop.core.factorial import is_point
        inc = incumbent_of(self.statuses) if self.statuses else ""
        flakes = self.store.get_state(self.K_ILL_FLAKE) or {}
        out: dict[str, list[int]] = {}
        for r in self.store.runs(status=[RunStatus.CRASH]):
            n = data.nodes.get(r.node_id)
            if n is None or n.base != data.epoch or r.node_id in (inc, data.base_root) or is_point(n.notes) \
                    or r.id in flakes:
                continue
            if crash_class(r) == ILL_CONDITIONED:
                out.setdefault(r.node_id, []).append(int(r.seed))
        return out

    def _readout_diff(self, data: CD.Ledger, node: str) -> bool:
        """The node's change touches the probe's features: an eval-readout family node, or a diff adding a wide /
        second-order feature block (:data:`_WIDE`)."""
        fam = getattr(getattr(self.ctx.cfg, "evalonly", None), "family", "") or "eval_readout"
        n = data.nodes.get(node)
        if n is not None and n.family == fam:
            return True
        sn = self.store.get_node(node)
        try:
            text = Path(sn.diff_path).read_text() if sn is not None and sn.diff_path else ""
        except OSError:
            text = ""
        return any(line.startswith("+") and self._WIDE.search(line) for line in text.splitlines())

    def ill_defect(self, data: CD.Ledger, node: str, runs: Sequence[Any] | None = None) -> tuple[bool, str]:
        """(an IDEA defect?, why): v3 review K4 (10-14: 8 of 9 flagged nodes had crashed ONCE, a node@ among them) --
        >= ILL_MIN_CRASHES ill-conditioned crashes making >= ILL_MIN_FRAC of its finished (done + crashed) runs on
        this epoch (``runs``: of these runs only -- a TEST block's), or a readout / probe-feature diff; otherwise a
        flake (the seed is re-run / re-drawn)."""
        flakes = self.store.get_state(self.K_ILL_FLAKE) or {}
        mine = [r for r in (self.store.runs(node_id=node) if runs is None else runs)
                if r is not None and r.status in (RunStatus.DONE, RunStatus.CRASH) and r.id not in flakes]
        ill = sum(1 for r in mine if r.status is RunStatus.CRASH and crash_class(r) == ILL_CONDITIONED)
        frac = ill / len(mine) if mine else 0.0
        if self._readout_diff(data, node):
            return True, f"{ill} ill-conditioned crash(es), a readout / probe-feature diff"
        if ill >= self.ILL_MIN_CRASHES and frac >= self.ILL_MIN_FRAC - 1e-9:
            return True, f"{ill} ill-conditioned crashes, {frac:.0%} of {len(mine)} finished runs"
        return False, f"{ill} ill-conditioned crash(es), {frac:.0%} of {len(mine)} finished runs: a flake"

    def _flake(self, run_ids: Sequence[str]) -> None:
        book = dict(self.store.get_state(self.K_ILL_FLAKE) or {})
        book.update({r: self.clock() for r in run_ids})
        self.store.set_state(self.K_ILL_FLAKE, book)

    def evidence(self, data: CD.Ledger, node: str, vs: str | None = None, seeds: Sequence[int] | None = None
                 ) -> dict[str, Any]:
        """The scorable seeds' evidence of ``node``: its paired panel gain over ``vs`` (default its parent) on the
        seeds both scored (``seeds``: only those), per seed."""
        from loop.core.model import seed_means
        n = data.nodes.get(node)
        ref = vs or (n.parent if n is not None else None)
        by = data.by_node()
        c, p = seed_means(by.get(node, ()), CD.PANEL), seed_means(by.get(ref, ()) if ref else (), CD.PANEL)
        d = {s: c[s] - p[s] for s in sorted(c.keys() & p.keys()) if seeds is None or s in seeds}
        return {"vs": ref or "", "n": len(d), "mean": round(sum(d.values()) / len(d), 6) if d else None,
                "seeds": {str(s): round(x, 6) for s, x in d.items()}}

    def book_repair(self, data: CD.Ledger, node: str, seeds: Sequence[int], acts: list[str], *, source: str,
                    evidence: Mapping[str, Any] | None = None) -> bool:
        """Book ONE repair session for ``node`` (pending until a researcher slot and the open cap allow); its scorable
        seeds' evidence is kept with it (``probe_ill_conditioned`` event). False when it is booked already."""
        book = dict(self.store.get_state(self.K_REPAIR) or {})
        if node in book:
            return False
        ev = dict(evidence or self.evidence(data, node))
        if ev.get("mean") is None or float(ev["mean"]) <= 0:      # v3 review K4: a repair only for a promising idea
            seen = dict(self.store.get_state(self.K_NO_REPAIR) or {})
            if node not in seen:
                seen[node] = self.clock()
                self.store.set_state(self.K_NO_REPAIR, seen)
                self.store.log("probe_ill_no_repair", node_id=node, seeds=sorted({int(x) for x in seeds}),
                               source=source, evidence=ev)
                acts.append(f"{node}: probe ill-conditioned on seeds {sorted({int(x) for x in seeds})}, but its "
                            f"scorable evidence (n {ev.get('n', 0)}, mean {ev.get('mean')}) is not positive: no repair")
            return False
        book[node] = {"status": "pending", "seeds": sorted({int(x) for x in seeds}), "evidence": ev,
                      "ts": self.clock(), "source": source}
        self.store.set_state(self.K_REPAIR, book)
        self.store.log("probe_ill_conditioned", node_id=node, seeds=book[node]["seeds"], source=source, evidence=ev)
        acts.append(f"{node}: probe ill-conditioned (survival probe did not converge on seeds {book[node]['seeds']}) "
                    f"-> idea defect, repair session queued (scorable evidence {ev.get('n', 0)} seeds, mean "
                    f"{ev.get('mean')})")
        return True

    def _ill_flake_rerun(self, node: str, why: str, acts: list[str]) -> None:
        """A screen's ill-conditioned crash judged a FLAKE (:meth:`ill_defect`): the crashed run is re-run once as
        its next attempt (its seed re-drawn by the SLURM-side nondeterminism; a second crash makes it a defect),
        logged ``probe_ill_flake``; shadow: listed only."""
        flakes = self.store.get_state(self.K_ILL_FLAKE) or {}
        sn = self.store.get_node(node)
        for r in self.store.runs(node_id=node, status=[RunStatus.CRASH]):
            if r.id in flakes or crash_class(r) != ILL_CONDITIONED or r.stage in TEST_STAGES:
                continue
            if self.dry:
                acts.append(f"{node}: probe ill-conditioned once ({why}): would re-run {r.id}")
                continue
            if sn is None or self.book.free() <= 0:
                return
            self._flake([r.id])
            got = self.guarded(f"run:{r.id}", f"re-run {r.id} (probe ill-conditioned flake)",
                               lambda r=r: self.book.retry(r, sn), acts)
            self.store.log("probe_ill_flake", node_id=node, run_id=r.id, why=why)
            acts.append(f"{node}: probe ill-conditioned once ({why}) -> re-run {r.id}"
                        + (f" -> {got.status.value}" if got is not None else ""))

    K_WIDTH = "repair.width_warned"  # nodes the pre-launch feature-width heuristic looked at (once each)
    #: an added line building a second-order / wide feature block (covariance, outer / Gram / bilinear products, the
    #: upper triangle of a d x d matrix): what made a node's 2080-dim block crash the survival probe
    _WIDE = re.compile(r"torch\.cov\b|\.cov\(|\bouter\(|triu_indices|\bgram\b|bilinear|"
                       r"einsum\([\"'][^\"']*,[^\"']*->[^\"']*(?:ij|jk|cd)[\"']|transpose\(-1, -2\)\)", re.I)

    def _width_warn(self, data: CD.Ledger, acts: list[str]) -> None:
        """Pre-launch heuristic (cheap, advisory): a current-epoch eval-readout node with no run yet whose diff adds
        a second-order / wide feature block (:data:`_WIDE`) is flagged once (``probe_width_warning``): its features
        may be too wide / ill-conditioned for the locked survival probe."""
        fam = getattr(getattr(self.ctx.cfg, "evalonly", None), "family", "") or "eval_readout"
        seen = set(self.store.get_state(self.K_WIDTH) or ())
        ran = {r.node for r in data.runs}
        new = []
        for nid, n in data.nodes.items():
            if nid in seen or nid in ran or n.base != data.epoch or n.family != fam:
                continue
            new.append(nid)
            node = self.store.get_node(nid)
            try:
                text = Path(node.diff_path).read_text() if node is not None and node.diff_path else ""
            except OSError:
                text = ""
            hits = sorted({m.group(0) for line in text.splitlines() if line.startswith("+")
                           for m in [self._WIDE.search(line)] if m})
            if hits:
                self.store.log("probe_width_warning", node_id=nid, hits=hits[:5])
                acts.append(f"{nid}: WARNING its readout adds a wide / second-order feature block ({', '.join(hits[:3])})"
                            " -- the locked survival probe may not converge (probe ill-conditioned)"[:300])
        if new:
            self.store.set_state(self.K_WIDTH, sorted(seen | set(new)))

    def _repair_throttle(self, data: CD.Ledger) -> str:
        """The v3-c7i brief throttle (``alloc.throttle`` on the QUEUED backlog) applied to the repair sessions too:
        '' (none) when the allocator is off or the backlog is fine."""
        from loop.core import alloc as AL
        ac = AL.cfg_of(self.ctx)
        if not ac.enabled or not self.statuses:
            return ""
        try:
            return AL.throttle(ac, AL.backlog(data, self.statuses, self.clock(), AL.screenable(self.core)))
        except Exception:  # noqa: BLE001 - a throttle that cannot be computed never blocks a repair
            return ""

    @staticmethod
    def repair_brief(node: str, rec: Mapping[str, Any]) -> str:
        ev = rec.get("evidence") or {}
        per = ", ".join(f"{s}: {float(x):+.4f}" for s, x in (ev.get("seeds") or {}).items()) or "none"
        return (f"REPAIR {node} (probe ill-conditioned). Its runs crashed in the LOCKED probe on seeds "
                f"{rec.get('seeds')}: the survival probe's Coxnet (sksurv) did not converge (ConvergenceWarning -> "
                "exit 1). The features this change exposes to the probe are too WIDE or ILL-CONDITIONED for the Cox "
                "model (e.g. a large covariance / outer-product / concatenated block: one node's 2080-dim covariance "
                "block crashed the same way). Keep the mechanism, fix the conditioning: reduce the added feature "
                "dimensions (project / pool / PCA it to at most the incumbent's feature width), normalise the added "
                "block (standardise or layer-norm it, scale it to the existing features), or keep it out of the "
                "exposed probe features. probe.py and benchmarking/ are LOCKED: never edit them.\n"
                f"Evidence from its SCORABLE seeds (paired panel gain vs {ev.get('vs') or 'its parent'}): "
                f"n {ev.get('n', 0)}, mean {ev.get('mean')}; per seed {per}. The repaired child keeps this evidence: "
                "it is measured against this node.")

    def _repair_sessions(self, data: CD.Ledger, acts: list[str]) -> None:
        """Book a repair for every ill-conditioned node, then start pending sessions while researcher slots are free
        and fewer than ``ideation.repair_max_open`` are in flight (shadow: listed only). A started session that left
        no node and is no longer in flight is re-issued up to REPAIR_TRIES times, then failed."""
        from loop.contracts import CandidateRequest, Operator, Strategy
        from loop.core.ideation import REPAIR_PREFIX
        self._width_warn(data, acts)
        for node, seeds in self._ill_nodes(data).items():
            real, why = self.ill_defect(data, node)
            if real:
                self.book_repair(data, node, seeds, acts, source="screen")
                continue
            self._ill_flake_rerun(node, why, acts)
        book = dict(self.store.get_state(self.K_REPAIR) or {})
        if not book:
            return
        changed = False
        live = {r.id for r in [*self.research.inflight.values(), *self.research.requeue]}
        made = {t[len("request="):] for n in self.store.nodes() for t in (n.notes or "").split()
                if t.startswith("request=")}
        cutoff = self._tick_ts if self._tick_ts is not None else -float("inf")
        for node, rec in book.items():
            if rec.get("status") != "started" or rec.get("request") in live or float(rec.get("ts", 0)) >= cutoff:
                continue
            ok = rec.get("request") in made
            again = not ok and int(rec.get("tries", 1)) < self.REPAIR_TRIES
            book[node] = {**rec, "status": "done" if ok else "pending" if again else "failed"}
            changed = True
        cap = int(getattr(getattr(self.ctx.cfg, "ideation", None), "repair_max_open", self.REPAIR_MAX_OPEN))
        open_ = sum(1 for r in book.values() if r.get("status") == "started")
        why = self._repair_throttle(data)
        for node, rec in sorted(book.items(), key=lambda kv: float(kv[1].get("ts", 0))):
            if rec.get("status") != "pending" or open_ >= cap:
                continue
            if why:                      # merged c7a x c7i: a repair is a brief like any other (v3-c7i I2 throttle)
                acts.append(f"repair session for {node} held: briefs throttled ({why})")
                continue
            if self.dry:
                acts.append(f"would start a researcher repair session for {node} (probe ill-conditioned)")
                open_ += 1
                continue
            if self.research.free() <= 0 or self.store.get_node(node) is None:
                break
            req = CandidateRequest(id=f"{REPAIR_PREFIX}{node}-{int(rec.get('tries', 0)) + 1}",
                                   task_id=self.ctx.task_id, strategy=Strategy.EXPLOIT, operator=Operator.DEBUG,
                                   island="main", parent_ids=(node,), brief=self.repair_brief(node, rec),
                                   brief_kind="repair_probe")
            self.research.start(req, None, lane="repair")
            book[node] = {**rec, "status": "started", "request": req.id, "ts": self.clock(),
                          "tries": int(rec.get("tries", 0)) + 1}
            open_, changed = open_ + 1, True
            acts.append(f"researcher started: {req.id} repair of {node} (probe ill-conditioned)")
        if changed:
            self.store.set_state(self.K_REPAIR, book)

    # ------------------------------------------------------------------ factorial waves (docs §5)
    K_WAVES = "factorial.next_wave"

    def _wave(self, carrier: str, data: CD.Ledger, now: float, acts: list[str], pre: bool = False) -> list[RunSpec]:
        """Design and materialise the wave of ``carrier``'s best knobs (``factorial.choose`` over the core's bank):
        an old-epoch diff knob is first ported onto the carrier (its own change; ``for_wave=<id>``), a knob whose
        graft fails (or raises) is dropped; the two highest-prior knobs are placed so their interaction is not the block
        contrast (``factorial.arrange``); one fresh SELECT seed per block of 8 (a ``wave-<id>`` SELECT block); the
        points are nodes tagged ``wave=<id> point=..``; their SCREEN runs at their block seed are returned.

        Crash-safe, ONE mechanism: the build is journalled first (``factorial.building``: id, carrier, knobs, seeds,
        diffs, the ``used`` marks), the points are built (git only), then :meth:`_finish_wave` writes the wave
        counter, its seed block, ``wave:<id>``, the point nodes, the bank's ``used`` marks (every knob chosen, built
        or dropped), the retro seed's flag, ``wave_started`` and the journal's clearing in ONE transaction. A crash
        before that commit leaves only the journal: the next tick resumes it (:meth:`_resume_wave`), never re-draws a
        new wave on the same bank. ``pre``: a relax-pass wave (``acquire.rank``): journalled and kept in ``wave:<id>``
        (``pre``) so every one of its points -- launched now or resumed on a later tick -- is preemptible-only
        (``RunBook.fill_only``; 10-13: w002's resumed points took 5 of 8 primary slots)."""
        from dataclasses import replace
        from loop import cutover as CO
        from loop.core import factorial as F
        if self.store.get_state(K_WAVE_BUILD):          # one build at a time: the half-built one first
            return self._resume_wave(data, now, acts)
        # the knobs a wave already consumed are out even when this action was ranked before that wave committed
        # (a resumed build earlier in the same tick): never a second wave on the same bank
        bank = F.unused(list(getattr(self.core, "knobs", {}).get(carrier) or ()),
                        self.store.get_state(ModuleCore.K_USED + carrier) or ())
        # every config knob checked against the carrier's config right before the build: one bad knob is dropped
        # (marked used, logged), never a wave whose every point fails (10-13: w007 built 0 of 16)
        bank, bad = F.check_knobs(bank, ModuleCore._config(self, carrier))
        for b, why in bad:
            self.store.log("knob_dropped", carrier=carrier, key=b.knob.key, lo=b.knob.lo, hi=b.knob.hi,
                           source=b.source, why=why, at="wave build")
            acts.append(f"wave on {carrier}: {b.source} knob {b.knob.key} dropped ({why})"[:300])
        ModuleCore.note_dropped(self, carrier, bad)       # v3-c7a A6: a missing key -> a prerequisite port
        if bad:
            self.store.set_state(ModuleCore.K_USED + carrier, sorted(
                {*(self.store.get_state(ModuleCore.K_USED + carrier) or ()), *(b.signature() for b, _ in bad)}))
        if not bank:
            acts.append(f"wave on {carrier}: its bank is consumed")
            return []
        n = int(self.store.get_state(self.K_WAVES) or 0) + 1
        wid, final, diffs, chosen = f"w{n:03d}", [], {}, F.choose(bank)
        for b in chosen:
            if b.knob.scale != "diff":
                final.append(b)
                continue
            nid = b.knob.key
            if nid not in data.nodes:
                continue
            if data.nodes[nid].base != data.epoch:
                try:
                    p = CO.port_node(self.store, self.adapter, data, nid, now, onto=carrier, tag=f"for_wave={wid}")
                except Exception as exc:  # noqa: BLE001 - the knob is dropped, the wave goes on
                    self.store.log("tick_call_error", key=f"port:{nid}~{carrier}", what="wave knob port",
                                   error=repr(exc)[:500], traceback=traceback.format_exc()[-3000:])
                    acts.append(f"wave {wid}: knob {nid} dropped (its port onto {carrier} raised {exc!r})"[:300])
                    continue
                if p.how not in ("applied", "exists"):
                    acts.append(f"wave {wid}: knob {nid} dropped (its port onto {carrier}: {p.how})")
                    self._missing_prereqs(carrier, p.how, knob=b, witness=nid)
                    continue
                acts.append(f"wave {wid}: knob {nid} via its port {p.node} on {carrier}")
                nid = p.node
            node = self.store.get_node(nid)
            diffs[nid] = (self.store.get_node(node.parent_id).commit, node.commit)
            final.append(replace(b, knob=F.Knob(nid, 0, 1, "diff")))
        used_key = ModuleCore.K_USED + carrier
        used = sorted({*(self.store.get_state(used_key) or ()), *(b.signature() for b in chosen)})
        if not final:
            self.store.set_state(used_key, used)          # nothing buildable: never offered again on this carrier
            acts.append(f"wave on {carrier}: no buildable knob")
            return []
        final = F.arrange(final)
        seeds = self.core.draw(F.n_blocks(len(final)), data, self.blocks(), self.ctx.cfg)
        job = {"wave": wid, "n": n, "carrier": carrier, "bank": [b.to_json() for b in final], "seeds": list(seeds),
               "blocks": list(F.matrix(len(final))[1]),
               "diffs": {k: list(v) for k, v in diffs.items()}, "used": used, "ts": now, "tries": 0, "pre": bool(pre)}
        self.store.set_state(K_WAVE_BUILD, job)         # the journal first: from here on a crash is resumed
        return self._finish_wave(job, now, acts)

    def _resume_wave(self, data: CD.Ledger, now: float, acts: list[str]) -> list[RunSpec]:
        """Finish a wave whose build a crash interrupted (state ``factorial.building``); given up after
        QUARANTINE_AFTER resumes (``wave_abandoned``; its knobs are marked used so the bank never re-offers them)."""
        from loop.core import factorial as F
        job = self.store.get_state(K_WAVE_BUILD)
        if not job:
            return []
        tries = int(job.get("tries", 0)) + (0 if job.get("paused") else 1)    # a time-boxed pause is no crash
        stall, cap = int(job.get("stall", 0)), int(self._hb("wave_pause_quarantine", 6))
        if job.get("paused") and cap > 0 and stall >= cap:     # paused tick after tick with no new point: quarantine
            with F.atomic(self.store):
                if job.get("used") is not None:
                    self.store.set_state(ModuleCore.K_USED + job["carrier"], job["used"])
                self.store.set_state(K_WAVE_BUILD, None)
                self.store.log("wave_quarantined", wave=job["wave"], carrier=job["carrier"], stall=stall,
                               paused=int(job.get("paused_n", 0)), built=len(job.get("built") or {}))
            acts.append(f"ALERT wave {job['wave']} on {job['carrier']}: build QUARANTINED after {stall} paused ticks "
                        f"with no new point ({len(job.get('built') or {})} built)")
            return []
        if tries > QUARANTINE_AFTER:
            with F.atomic(self.store):
                if job.get("used") is not None:
                    self.store.set_state(ModuleCore.K_USED + job["carrier"], job["used"])
                self.store.set_state(K_WAVE_BUILD, None)
                self.store.log("wave_abandoned", wave=job["wave"], carrier=job["carrier"], tries=tries - 1)
            acts.append(f"ALERT wave {job['wave']} on {job['carrier']}: build abandoned after {tries - 1} resumes")
            return []
        job = {**job, "tries": tries, "paused": False,
               "paused_n": int(job.get("paused_n", 0)) + (1 if job.get("paused") else 0)}   # paused tries counted
        self.store.set_state(K_WAVE_BUILD, job)
        acts.append(f"wave {job['wave']} on {job['carrier']}: resuming a half-built wave (resume {tries}, "
                    f"paused {job['paused_n']}, stalled {stall})")
        return self._finish_wave(job, now, acts)

    def _finish_wave(self, job: Mapping[str, Any], now: float, acts: list[str]) -> list[RunSpec]:
        """The journalled build ``job``: every point is built (git only; a build that raises is a failed point),
        then ONE transaction writes the wave counter, the seed block, ``wave:<id>``, the point nodes,
        ``wave:<id>:bank``, the ``used`` marks, the retro seed's flag and ``wave_started``, and clears the journal."""
        from loop.core import factorial as F
        from loop.core.seeds import SeedBlock
        wid, carrier, n = job["wave"], job["carrier"], int(job["n"])
        final = [F.BankKnob.from_json(b) for b in job["bank"]]
        diffs = {k: tuple(v) for k, v in (job.get("diffs") or {}).items()}
        seeds = tuple(int(x) for x in job["seeds"])
        cfg_rel = getattr(self.adapter, "config_rel", "config.yaml")
        try:                                            # O1: a journal of an older design version (no ``blocks``)
            wave = F.design(wid, carrier, [b.knob for b in final], seeds, blocks=job.get("blocks"))
        except ValueError as exc:                       # ... is never a crash to quarantine: given up, logged
            with F.atomic(self.store):
                if job.get("used") is not None:
                    self.store.set_state(ModuleCore.K_USED + carrier, job["used"])
                self.store.set_state(K_WAVE_BUILD, None)
                self.store.log("wave_design_error", wave=wid, carrier=carrier, seeds=list(seeds), error=str(exc)[:300])
            acts.append(f"ALERT wave {wid} on {carrier}: its journal does not fit this design version ({exc}); "
                        "build given up (knobs marked used)")
            return []
        rows = F.matrix(len(final))
        if len(wave.points) == len(rows[0]) and len(set(wave.blocks)) < len(set(rows[1])):
            acts.append(f"wave {wid}: resumed on the journal's single-block layout (seeds {list(seeds)})")
        if (dups := len(rows[0]) - len(wave.points)):         # v3-c7a A2: byte-identical replicates left out
            acts.append(f"wave {wid}: {dups} duplicate point(s) (same levels, same block seed) left out")
        t_w = time.monotonic()
        try:
            got = self._build_points(wave, diffs, cfg_rel, job)
        finally:
            self._time("launch.waves", t_w)
        if got is None:                                 # time-boxed: the journal keeps the built points
            acts.append(f"wave {wid} on {carrier}: build paused (tick budget), {len(job.get('built') or {})}"
                        f"/{len(wave.points)} points built; resumed next tick")
            return []
        have, built, fails = got
        cnode = self.store.get_node(carrier)
        with F.atomic(self.store):
            if int(self.store.get_state(self.K_WAVES) or 0) < n:
                self.store.set_state(self.K_WAVES, n)
            if not any(b.id == f"wave-{wid}" for b in self.blocks()):
                self._put_block(SeedBlock(f"wave-{wid}", CD.SELECT, seeds, (carrier,), f"wave {wid} on {carrier}",
                                          float(job.get("ts", now))), acts)
            self.store.set_state(f"wave:{wid}", {**F.to_json(wave), "pre": bool(job.get("pre"))})
            made = [have[i] for i in sorted(have)]
            k = 0
            for i, sha in built:
                while self.store.get_node(f"{wid}.p{k:02d}") is not None:
                    k += 1
                made.append(F._put(self.store, f"{wid}.p{k:02d}", sha, cnode, now,
                                   f"{F.point_tag(wave, i)} {F._levels(wave.points[i])}", None))
            self.store.set_state(f"wave:{wid}:bank", [b.to_json() for b in final])
            if job.get("used") is not None:
                self.store.set_state(ModuleCore.K_USED + carrier, job["used"])
            if any(b.source == "retro" for b in final):     # the retro seed is consumed by the first wave
                bk = dict(self.store.get_state(ModuleCore.K_BANK + carrier) or {})
                if bk:
                    self.store.set_state(ModuleCore.K_BANK + carrier, {**bk, "used": wid})
            self.store.log("wave_started", wave=wid, carrier=carrier, seeds=list(seeds), points=made, failed=fails,
                           knobs=[{"key": b.knob.key, "lo": b.knob.lo, "hi": b.knob.hi, "scale": b.knob.scale,
                                   "m": b.m, "source": b.source} for b in final])
            self.store.set_state(K_WAVE_BUILD, None)        # in the same commit: the wave is whole
        acts.append(f"wave {wid} on {carrier}: {len(final)} knobs ({', '.join(f'{b.knob.key} {b.source}' for b in final)}),"
                    f" {len(made)}/{len(wave.points)} points, seeds {list(seeds)}"
                    + (f"; {len(fails)} point(s) left out: {fails[0]}" if fails else ""))
        out = []
        for nid in made:
            t = F._TAG.search(self.store.get_node(nid).notes)
            out.append(RunSpec(nid, SEED_STAGE, int(t.group(4))))
        if made:                                        # v3-c7a A2: one carrier control per block seed (pairing)
            have = {r.seed for r in self.store.runs(node_id=carrier)}
            ctl = [RunSpec(carrier, Stage.PARENT, s) for s in seeds if s not in have]
            if ctl:
                acts.append(f"wave {wid}: carrier control {carrier} on {[c.seed for c in ctl]}")
            out += ctl
        if job.get("pre"):                              # a relax wave: preemptible-only, now and when resumed
            self.book.fill_only |= {(s.node, s.seed) for s in out}
        return out

    def _build_points(self, wave: Any, diffs: Mapping[str, tuple[str, str]], cfg_rel: str,
                      job: Mapping[str, Any] | None = None
                      ) -> tuple[dict[int, str], list[tuple[int, str]], list[str]] | None:
        """Git only, no store writes but error events: the points already in the store (by their ``wave=<id>
        point=<i>`` tag) are kept, the others built (``factorial.build_point``; a build that raises is a failed point
        with its reason). Returns (kept: point -> node, built: [(point, sha)], failures). With the build journal
        ``job``: points it already built (``built``: point -> sha) are reused, and past the tick budget
        (:meth:`over_budget`) the journal is updated with what was built and None returned (resumed next tick)."""
        from loop.core import factorial as F
        have: dict[int, str] = {}
        for nd in self.store.nodes():
            if (t := F._TAG.search(nd.notes or "")) and t.group(1) == wave.id:
                have[int(t.group(2))] = nd.id
        carrier = self.store.get_node(wave.carrier)
        done = {int(k): v for k, v in ((job or {}).get("built") or {}).items()}
        built, fails, new = [], [], 0
        for i, levels in enumerate(wave.points):
            if i in have:
                continue
            if i in done:
                built.append((i, done[i]))
                continue
            # past the budget the build pauses -- but only once this tick attempted >= min_wave_points points (a
            # tick whose budget is gone before the launch phase still makes progress: never starved)
            if job is not None and self.over_budget() and self.__dict__.get("_wpts", 0) >= self._hb("min_wave_points", 1):
                stall = 0 if new else int(job.get("stall", 0)) + 1     # paused ticks running with no new point
                self.store.set_state(K_WAVE_BUILD, {**job, "built": {str(k): v for k, v in done.items()},
                                                    "paused": True, "stall": stall})
                if isinstance(job, dict):
                    job["built"], job["stall"] = {str(k): v for k, v in done.items()}, stall
                return None
            self.__dict__["_wpts"] = self.__dict__.get("_wpts", 0) + 1
            try:
                sha, why = F.build_point(self.adapter, carrier.commit, levels, wave.knobs, diffs, f"{wave.id}-p{i}",
                                         cfg_rel)
            except Exception as exc:  # noqa: BLE001 - one point that does not build is left out
                self.store.log("tick_call_error", key=f"build_point:{wave.id}.p{i}", what="build a wave point",
                               error=repr(exc)[:500], traceback=traceback.format_exc()[-3000:])
                sha, why = "", f"raised {exc!r}"[:300]
            if sha:
                built.append((i, sha))
                done[i] = sha
                new += 1
            else:
                fails.append(f"{wave.id} point {i}: {why}")
        return have, built, fails

    def _close_waves(self, data: CD.Ledger, acts: list[str]) -> None:
        """A wave whose every point has a final run is analysed once: main effects (``factorial.estimate``, the
        factorial prior of its carrier), x* as a QUEUED node (never the best cell), and the next ranges (the edge
        rule) left for the carrier's next wave; ``wave:<id>:result`` + a ``wave_result`` event."""
        from loop.core import factorial as F
        waves: dict[str, list[str]] = {}
        for n in data.nodes.values():
            if (t := F._TAG.search(n.notes)):
                waves.setdefault(t.group(1), []).append(n.id)
        for wid, pts in waves.items():       # each wave guarded: an x* build that raises never stops the tick
            self.guarded(f"wave.close:{wid}", f"close wave {wid}",
                         lambda wid=wid, pts=pts: self._close_wave(wid, pts, data, acts), acts)
        building = (self.store.get_state(K_WAVE_BUILD) or {}).get("wave")
        for i in range(1, int(self.store.get_state(self.K_WAVES) or 0) + 1):    # waves that built no point
            wid = f"w{i:03d}"
            if wid in waves or wid == building or self.store.get_state(f"wave:{wid}:result") is not None \
                    or (js := self.store.get_state(f"wave:{wid}")) is None:
                continue
            self.guarded(f"wave.rearm:{wid}", f"re-arm the empty wave {wid}",
                         lambda wid=wid, js=js: self._rearm(wid, js, data, acts), acts)

    def _open_waves(self, data: CD.Ledger) -> dict[str, str]:
        """``factorial.open_waves`` without the waves whose result is recorded (``wave:<id>:result``: closed)."""
        from loop.core import factorial as F
        ow = F.open_waves(data)
        return F.open_waves(data, closed=[w for w in ow if self.store.get_state(f"wave:{w}:result") is not None])

    def _fcfg(self, key: str, default: Any) -> Any:
        """``factorial.<key>`` of the loop config, else ``default``."""
        return getattr(getattr(self.ctx.cfg, "factorial", None), key, default)

    def _rearm(self, wid: str, js: Mapping[str, Any], data: CD.Ledger, acts: list[str]) -> None:
        """A wave that built NO point (10-13: w007, a 4-leg stack, 0/16 -- one plateau knob's key was absent from
        the carrier config, so every point failed; its knobs stayed consumed and the stack never ran): closed as
        void (``wave:<id>:result``), and its knobs the carrier's config takes (``factorial.check_knobs``) are no
        longer marked used, so the carrier's next wave offers them again -- each at most ``factorial.rearm_max``
        times (``factorial.rearm:<carrier>``); the bad knobs stay consumed. ``wave_rearmed`` event + an act line."""
        from loop.core import factorial as F
        wave = F.from_json(js)
        car = wave.carrier
        bank = [F.BankKnob.from_json(b) for b in self.store.get_state(f"wave:{wid}:bank") or ()] \
            or [F.BankKnob(k, 0.0, 0.0, "wave") for k in wave.knobs]
        ok, bad = F.check_knobs(bank, ModuleCore._config(self, car))
        cap = int(self._fcfg("rearm_max", 1))
        counts = dict(self.store.get_state(f"factorial.rearm:{car}") or {})
        used = set(self.store.get_state(ModuleCore.K_USED + car) or ())
        freed, spent = [], []
        for b in ok:
            sigs = {b.signature()}
            if b.knob.scale == "diff" and (nd := data.nodes.get(b.knob.key)) is not None and nd.port_of:
                sigs.add(json.dumps([nd.port_of, b.knob.lo, b.knob.hi], sort_keys=True, default=str))
            if int(counts.get(b.signature(), 0)) >= cap:
                spent.append(b.knob.key)
                continue
            counts[b.signature()] = int(counts.get(b.signature(), 0)) + 1
            used -= sigs
            freed.append(b.knob.key)
        res = {"points": 0, "observed": 0, "void": "no point built", "rearmed": freed,
               "dropped": [{"key": b.knob.key, "why": why} for b, why in bad], "spent": spent}
        with F.atomic(self.store):
            self.store.set_state(ModuleCore.K_USED + car, sorted(used))
            self.store.set_state(f"factorial.rearm:{car}", counts)
            self.store.set_state(f"wave:{wid}:result", res)
            self.store.log("wave_rearmed", wave=wid, carrier=car, **res)
        acts.append(f"wave {wid} on {car}: built no point -> VOID; knobs re-armed for the next wave: "
                    f"{', '.join(freed) or 'none'}" + "".join(f"; {b.knob.key} stays consumed ({why})" for b, why in bad)
                    + (f"; {', '.join(spent)} re-armed {cap}x already" if spent else ""))

    def _churn_ids(self) -> set[str]:
        """Run ids the idle churn cancelled (``run_cancelled`` events, reason ``idle fill preempted ..``)."""
        try:
            return {str(e.get("run_id")) for e in self.store.events("run_cancelled")
                    if str(e.get("reason", "")).startswith(ModuleCore.CHURN_REASON)}
        except Exception:  # noqa: BLE001 - a store without events
            return set()

    def _churn_due(self, wid: str, pts: Sequence[str], data: CD.Ledger, ids: set[str]) -> list[str]:
        """Run ids of ``wid``'s points the idle churn cancelled that still have a resubmit left
        (``factorial.churn_resubmit_max``; counted in ``wave:<id>:resub``) -- none once the wave has a result."""
        cap = int(self._fcfg("churn_resubmit_max", 1))
        if cap <= 0 or not ids or self.store.get_state(f"wave:{wid}:result") is not None:
            return []
        n = self.store.get_state(f"wave:{wid}:resub") or {}
        mine = set(pts)
        return [r.id for r in data.runs if r.node in mine and r.id in ids and r.status == "void"
                and int(n.get(r.id, 0)) < cap]

    def _resubmit_churned(self, data: CD.Ledger, acts: list[str]) -> list[str]:
        """The churned points of every wave without a result, resubmitted (``RunBook.retry``: the same row, the next
        attempt) while GPU slots are free and resubmits are left -- not fill-only: the preemptible pool is what
        killed them. Their wave closes once they finish (or their resubmits are spent: abandoned, logged)."""
        from loop.core import factorial as F
        ids = self._churn_ids()
        if not ids:
            return []
        waves: dict[str, list[str]] = {}
        for nd in data.nodes.values():
            if (t := F._TAG.search(nd.notes)):
                waves.setdefault(t.group(1), []).append(nd.id)
        out = []
        for wid, pts in sorted(waves.items()):
            for rid in self._churn_due(wid, pts, data, ids):
                if self.book.free() <= 0:
                    return out
                run = self.store.get_run(rid)
                node = self.store.get_node(run.node_id) if run is not None else None
                if run is None or node is None or run.status is not RunStatus.VOID:
                    continue
                n = dict(self.store.get_state(f"wave:{wid}:resub") or {})
                n[rid] = int(n.get(rid, 0)) + 1
                self.store.set_state(f"wave:{wid}:resub", n)
                self.book.fill_only.discard((run.node_id, run.seed))
                got = self.guarded(f"run:{rid}", f"resubmit the churned wave point {rid}",
                                   lambda run=run, node=node: self.book.retry(run, node), acts)
                if got is not None:
                    self.store.log("wave_point_resubmitted", wave=wid, run_id=rid, n=n[rid])
                    acts.append(f"wave {wid}: churned point {rid} resubmitted ({n[rid]}/"
                                f"{int(self._fcfg('churn_resubmit_max', 1))}) -> {got.status.value}")
                    out.append(got.id)
        return out

    def _close_wave(self, wid: str, pts: list[str], data: CD.Ledger, acts: list[str]) -> None:
        """One complete wave's analysis (:meth:`_close_waves`). Raises before its result is written are retried next
        tick (an x* already built is reused: ``<id>.x``), then quarantined (``wave.close:<id>``)."""
        from loop.core import factorial as F
        if self.store.get_state(f"wave:{wid}:result") is not None or (js := self.store.get_state(f"wave:{wid}")) \
                is None:
            return
        runs = [self.store.get_run(r.id) for r in data.runs if r.node in pts]
        if {r.node_id for r in runs if r is not None} != set(pts) or any(
                r is not None and r.status not in (RunStatus.DONE, RunStatus.CRASH) and not self._dead(r)
                for r in runs):
            return
        ids = self._churn_ids()
        if self._churn_due(wid, pts, data, ids):        # a churned point gets its resubmit first (_launch)
            return
        for r in runs:                                  # the rest: abandoned explicitly, the wave closes without them
            if r is not None and r.status is RunStatus.VOID and r.id in ids:
                self.store.log("wave_point_abandoned", wave=wid, run_id=r.id, reason="idle churn, resubmits spent")
                acts.append(f"wave {wid}: point {r.id} abandoned (cancelled by the idle churn, resubmits spent)")
        wave, post = F.from_json(js), self.post
        ys = F.point_values(data, wid, post.weights)
        res: dict[str, Any] = {"points": len(pts), "observed": len(ys)}
        if len(ys) < len(wave.knobs) + 2:
            res["void"] = f"{len(ys)} of {len(pts)} points observed: too few for {len(wave.knobs)} main effects"
            acts.append(f"wave {wid}: VOID ({res['void']})")
        else:
            t2 = float((self.store.get_state(ModuleCore.K_BANK + wave.carrier) or {}).get("tau2") or 0) \
                or F.factorial_tau2(post.prior, wave.carrier)
            eff = F.estimate(wave, ys, post.noise, t2, far=post.fit(post.epoch, True)[0]
                             if hasattr(post, "fit") else None)
            cfg = ModuleCore._config(self, wave.carrier)
            res["effects"] = [{"key": e.key, "beta": e.beta, "se": e.se, "shrunk": e.shrunk} for e in eff]
            nxt = F.next_point(wave, eff, cfg)
            if nxt is not None:
                diffs = {k.key: (self.store.get_node(self.store.get_node(k.key).parent_id).commit,
                                 self.store.get_node(k.key).commit) for k in wave.knobs if k.scale == "diff"}
                xid, why = (f"{wid}.x", "") if self.store.get_node(f"{wid}.x") is not None else \
                    F.materialize_next(self.store, self.adapter, wave, nxt, diffs,
                                       getattr(self.adapter, "config_rel", "config.yaml"), lambda: f"{wid}.x",
                                       self.clock())
                res["next"] = {"node": xid, "why": why, "levels": nxt.levels, "m": nxt.m, "v": nxt.v}
            nk = [dict(key=k.key, lo=k.lo, hi=k.hi, scale=k.scale) for k in F.next_knobs(wave, eff)
                  if k.scale != "diff"]
            self.store.set_state(ModuleCore.K_NEXT + wave.carrier, nk)
            acts.append(f"wave {wid} closed: " + ", ".join(f"{e.key} {e.beta:+.4f} (z {e.z:+.1f})" for e in eff)
                        + (f"; x* {res['next']['node'] or 'not built: ' + res['next']['why']} prior "
                           f"{nxt.m:+.4f}" if nxt is not None else "; no knob moves (x* = the carrier)"))
        self.store.set_state(f"wave:{wid}:result", res)
        self.store.log("wave_result", wave=wid, carrier=wave.carrier, **res)

    def _dead(self, run: Any) -> bool:
        """A run that will never give its pair: crashed, or a void the runbook does not retry."""
        return run is not None and (run.status is RunStatus.CRASH or run.status is RunStatus.VOID and (
            run.void_type not in (None, VoidType.INFRA) or run.attempt >= self.book.infra_retry))

    K_CHANGE = "core.change"                       # "c|i" -> True when c and i are one config key apart, config only

    def _one_key(self, c: str, i: str, cache: dict[str, bool]) -> bool:
        """c and i differ in exactly ONE config key and no other file (the adapter's ``config_at`` + ``diff_commits``;
        an adapter without them, an unreadable config or a failed diff: False -- the far class)."""
        key = f"{c}|{i}"
        if key in cache:
            return cache[key]
        ok = False
        nc, ni = self.store.get_node(c), self.store.get_node(i)
        if nc is not None and ni is not None and hasattr(self.adapter, "config_at") \
                and hasattr(self.adapter, "diff_commits") and nc.commit and ni.commit:
            try:
                a, b = self.adapter.config_at(ni.commit), self.adapter.config_at(nc.commit)
                keys = [k for k in set(a) | set(b) if a.get(k) != b.get(k)]
                files = re.findall(r"^diff --git a/(\S+) b/", self.adapter.diff_commits(ni.commit, nc.commit) or "",
                                   re.M)
                cfg = getattr(self.adapter, "config_rel", None)
                code = [f for f in files if (f != str(cfg) if cfg else not f.endswith((".yaml", ".yml")))]
                ok = bool(a) and bool(b) and len(keys) == 1 and not code
            except Exception:  # noqa: BLE001 - unclassifiable: the far class
                ok = False
        cache[key] = ok
        return ok

    def _classify(self, data: CD.Ledger) -> None:
        """The change classes the core judges and CUPEDs with: every current-epoch node vs its parent, and every open
        test's candidate vs its control (cached in state ``core.change``: commits never change)."""
        if not hasattr(self.core, "classify"):
            return
        cache = dict(self.store.get_state(self.K_CHANGE) or {})
        n0 = len(cache)
        near = {k for k, n in data.nodes.items() if n.parent is not None and n.base == data.epoch and not n.port_of
                and self._one_key(k, n.parent, cache)}
        for b in self.blocks():
            if b.pool == CD.TEST and b.open:
                for c in b.node_ids[:-1]:
                    self._one_key(c, b.node_ids[-1], cache)
        if len(cache) != n0:
            self.store.set_state(self.K_CHANGE, cache)
        self.core.classify(frozenset(near), lambda c, i: bool(cache.get(f"{c}|{i}")))

    def _freeze(self, data: CD.Ledger, noise: Any, acts: list[str]) -> bool:
        """Closed TEST blocks without frozen verdicts (historical ones; a block closed outside ``_decide``): their
        judgement made ONCE now and stored; replay never re-judges them."""
        from loop.core import status as ST
        if not hasattr(self.core, "ST"):
            return False
        frozen = ST.freeze(data, self.blocks(), noise)
        for bid, recs in frozen.items():
            said = ", ".join(str(r["node"]) + " " + str(r["kind"]) for r in recs) or "none"
            self._update_block(bid, acts, f"verdicts frozen ({said})", verdicts=recs)
        return bool(frozen)

    def _stale_controls(self, data: CD.Ledger, inc: str, now: float, acts: list[str]) -> bool:
        """Open TEST blocks whose control (their last member) is no longer the incumbent (nor its port) can switch
        nothing (status.replay). Re-paired against the incumbent when it has a live run on EVERY seed of the block
        (same seeds: CRN), else closed as ``stale_control`` (no verdict: the candidate is re-testable). The runs
        that no longer pair are cancelled by job id (``RunBook.cancel_run``): PENDING ones always, RUNNING ones when
        ``test.stale_cancel_running``. True if a block changed.

        Stale = the control's PORT in the current base epoch is not the incumbent's (``Ledger.port``): a base-epoch
        change of the SAME incumbent (control and incumbent port to one another) cancels nothing. At an epoch switch
        (the control belongs to an older base) RUNNING runs are never cancelled: they finish (their observations
        still count in the ledger). SUBMITTING rows (no job id yet, a submit may be in flight: scancel would miss
        it) are likewise never cancelled -- CHOICE: let them finish and log ``stale_run_kept``."""
        acts.extend(self.book.retry_cancels())        # cancels the adapter could not confirm last tick
        port = getattr(data, "port", lambda *_: None)
        here = [h for h in (inc, port(inc)) if h]
        if not here or inc == ROOT_NODE_ID:
            return False
        epoch = getattr(data, "epoch", None)
        nodes = getattr(data, "nodes", {}) or {}
        run_all = bool(getattr(getattr(self.ctx.cfg, "test", None), "stale_cancel_running", True))
        live = {(r.node, r.seed) for r in data.runs if r.status not in ("crash", "void")}
        changed = False
        for b in [b for b in self.blocks() if b.pool == CD.TEST and b.open]:
            ctrl = b.node_ids[-1] if b.node_ids else None
            if ctrl is None or ctrl in here or port(ctrl) in here:
                continue
            switch = epoch is not None and getattr(nodes.get(ctrl), "base", epoch) != epoch
            new = next((h for h in here if all((h, s) in live for s in b.seeds)), None)
            cands = [n for n in b.node_ids[:-1] if n not in here]
            if new is not None and cands:
                drop = [ctrl]
                self._update_block(b.id, acts, f"control {ctrl} -> {new} (incumbent switch; {new} ran every seed: "
                                   "re-paired)", node_ids=(*cands, new))
                self.store.log("test_repaired", block=b.id, old=ctrl, new=new)
            else:
                drop = list(b.node_ids)
                self._update_block(b.id, acts, f"stale_control: {ctrl} is no longer the incumbent ({inc}): closed",
                                   closed_ts=now, verdicts=())
                self.store.log("test_stale_control", block=b.id, control=ctrl, incumbent=inc,
                               candidates=list(b.node_ids[:-1]))
            seeds = set(b.seeds)
            for n in drop:
                for run in self.store.runs(node_id=n, status=[RunStatus.SUBMITTING]):
                    if run.seed in seeds and run.stage in TEST_STAGES:
                        self.store.log("stale_run_kept", run=run.id, block=b.id, status="submitting")
                for run in self.store.runs(node_id=n, status=[RunStatus.RUNNING]):
                    if switch and run.seed in seeds and run.stage in TEST_STAGES:
                        self.store.log("stale_run_kept", run=run.id, block=b.id, status="running")
                        continue                                   # an epoch switch: running runs finish
                    if run.seed in seeds and run.stage in TEST_STAGES \
                            and (run_all or run.id in self.book.pending):
                        if self.book.cancel_run(run, f"stale_control: block {b.id} ({ctrl} no longer incumbent)"):
                            acts.append(f"cancelled {run.id} (stale control)")
            changed = True
        return changed

    def _test_stage(self, n: str, b: Any, s: int) -> Stage:
        """The stage of member ``n``'s run on seed ``s`` of TEST block ``b``: candidates CONFIRM, the control REMINT
        -- unless the control was re-paired in (``_stale_controls``) and already ran it as a CONFIRM."""
        if n != b.node_ids[-1]:
            return TEST_STAGE
        return TEST_STAGE if self.store.get_run(Run.make_id(n, TEST_STAGE, s)) is not None else TEST_CONTROL_STAGE

    K_LOCAL = "test.local_looks"     # {block: {str(n): "ok" | "no"}}: the local tier's ONE decision per look (persisted)

    def _local_reroot(self, data: CD.Ledger, b: Any, c: str, v: Any, kind: str, here: set[str],
                      acts: list[str]) -> tuple[Any, dict[str, Any]] | None:
        """THE LOCAL-INCUMBENT TIER (``test.local_reroot``; operator 11-09 "reroot on free wins"): (the verdict turned
        PASS, the provenance fields frozen into it) when candidate ``c``'s TEST block is at a look the gs2 test left
        undecided -- every planned seed in and EXTEND (look 1), or waiting for an extension slot, or an extended block
        the tier never judged at look 1 (judged once on its K seeds while the extension runs) -- or FAILed at its
        FINAL look (look 2) without harm, its control is the incumbent, and :func:`status.local_reroot` finds it a free
        win. Never a look-1 FAIL (futility is binding and priced into c2). ONE decision per look (block, n), persisted
        (``K_LOCAL``): the EB prior moves as other screens land, and re-deciding the same data each tick would be
        optional stopping. The PASS switches the incumbent through the usual machinery (frozen verdict, replay, the
        remint SELECT block, the carrier, stale controls). It submits nothing itself: the incumbent-only Labless
        submitter still sends an incumbent only once it beats the (unchanged) bar. None: not judged / no. Any error
        is logged and read as "no" (the tick never stops on it)."""
        tc = getattr(self.ctx.cfg, "test", None)
        if not getattr(tc, "local_reroot", False) or kind == "pass" or not b.node_ids or b.node_ids[-1] not in here:
            return None
        n = int(getattr(v, "n", 0))
        if getattr(v, "harm", ()):
            return None                               # a harmed axis vetoes
        from loop.core import test as T
        looks = dict(self.store.get_state(self.K_LOCAL) or {})
        if str(n) in (looks.get(b.id) or {}):
            return None                               # decided at this look already (a "no" stands)
        # a block EXTENDED before the tier looked at its look 1 (the tier off / deployed mid-test): look 1 is judged
        # once, late, on exactly its K seeds while the extension's seeds still run -- the same single look
        catch_up = kind == "open" and n == T.K and len(b.seeds) > T.K
        if n < len(b.seeds) and not catch_up:
            return None                               # a look needs every planned seed in
        final = len(b.seeds) > T.K or n >= T.K_MAX
        if kind == "open" and int(getattr(v, "more", 0)) != 0 and not catch_up:
            return None
        if kind not in ("extend", "open", "fail") or (kind == "fail" and not final):
            return None
        try:
            from loop.core import status as ST
            lr = ST.local_reroot(data, c, b.node_ids[-1], b.seeds, p_min=float(tc.local_reroot_p),
                                 min_fresh=int(tc.local_reroot_min_fresh), min_gain=float(tc.local_reroot_min_gain),
                                 fresh_z=float(getattr(tc, "local_reroot_fresh_z", 1.0)),
                                 sd_fresh=float(getattr(v, "sd", 0.0) or 0.0) or None,
                                 eb_min_nodes=int(getattr(getattr(self.ctx.cfg, "labless", None), "eb_min_nodes", 5)))
            out = replace(v, kind=T.Kind.PASS, reason="local re-root", more=0) if lr.ok else None
        except Exception as exc:  # noqa: BLE001 - the tier is an add-on: an error is a "no", never a stopped tick
            self.store.log("local_reroot_error", block=b.id, node_id=c, error=repr(exc)[:500])
            acts.append(f"{c}: local re-root tier error ({exc!r})"[:300])
            return None
        looks[b.id] = {**(looks.get(b.id) or {}), str(n): "ok" if lr.ok else "no"}
        self.store.set_state(self.K_LOCAL, looks)
        if not lr.ok:
            self.store.log("local_reroot_no", block=b.id, node_id=c, gs2=kind, n=n, why=lr.why)
            acts.append(f"{c}: local re-root tier: no ({lr.why}); gs2 {kind.upper()}")
            return None
        why = f"{lr.why} (gs2 {kind.upper()}: {getattr(v, 'reason', '')})"
        fields = {"tier": "local", "local_p": round(lr.p, 4), "local_gain": round(lr.gain, 6),
                  "local_sd": round(lr.sd, 6), "local_n_pair": lr.n_pair, "local_n_fresh": lr.n_fresh,
                  "local_fresh_mean": round(lr.fresh_mean, 6), "gs2_kind": kind, "local_why": why}
        return replace(out, reason=f"local re-root: {why}"), fields

    def _decide(self, data: CD.Ledger, noise: Any, statuses: Mapping[str, Any], now: float,
                acts: list[str]) -> bool:
        """Apply the open tests' verdicts (each frozen into its block as it closes); replace the dead pairs' seeds.
        Several passes against the incumbent in one tick: only the best (mean, then E) re-mints -- the others are
        PASSED and re-testable (``status.retest_candidates``). True if a block changed."""
        from loop.core import status as ST
        changed = self._freeze(data, noise, acts)
        self._classify(data)
        inc = incumbent_of(statuses)
        if statuses:
            changed = self._stale_controls(data, inc, now, acts) or changed
        passes = []
        here = {inc, *(p for p in (getattr(data, "port", lambda *_: None)(inc),) if p)}
        for item in self.core.decide(data, self.blocks(), noise, inc):
            b, c, v = item[:3]
            fit = item[3] if len(item) > 3 else None
            kind = str(getattr(v.kind, "value", v.kind)).lower()
            # review 11-10 S8: guarded -- an error anywhere in the tier is gs2 alone, never a stopped tick
            local = self.guarded("test.local_reroot", "the local-incumbent re-root tier (``test.local_reroot``)",
                                 lambda: self._local_reroot(data, b, c, v, kind, here, acts), acts, default=None)
            if local is not None:
                v, kind = local[0], "pass"
            if kind in ("pass", "fail"):
                rec = ST.record(c, b.node_ids[-1], v, fit, ts=now) if hasattr(v, "sd") else None
                if rec is not None and local is not None:
                    rec.update(local[1])
                self._update_block(b.id, acts, f"{c} {kind.upper()} ({v.reason}; n={v.n}, mean {v.mean:+.5f}, "
                                   f"E={v.e:.3g}" + (f", sd_d {rec['sd']:.5f} {rec.get('fit', '')}" if rec else "")
                                   + "): closed", closed_ts=now, **({"verdicts": (rec,)} if rec else {}))
                self.store.log("test_verdict", block=b.id, node_id=c, verdict=kind, e=v.e, n=v.n, mean=v.mean,
                               **({k: rec.get(k) for k in ("sd", "rho", "df", "near", "fit", "resumed")} if rec else {}),
                               **({"tier": "local"} if local is not None else {}))
                if local is not None:
                    self.store.log("local_reroot", block=b.id, node_id=c, previous=b.node_ids[-1], **local[1])
                    acts.append(f"{c}: LOCAL RE-ROOT over {b.node_ids[-1]} (free win; Labless only once it beats the bar): "
                                f"{local[1]['local_why']}")
                if kind == "pass":
                    passes.append((b, c, v))
                changed = True
            elif kind == "extend" and v.more > 0 and len(b.seeds) <= T_K(self.core):
                more = self.core.draw(v.more, data, self.blocks(), self.ctx.cfg)
                self._update_block(b.id, acts, f"{c} EXTEND by {list(more)} ({v.reason})",
                                   seeds=(*b.seeds, *more))
                self._log_test_draw(b, more, "extend")
                changed = True
        win = [(b, c, v) for b, c, v in passes if b.node_ids[-1] in here]
        def rank(x: Any) -> tuple:                     # a gs2 PASS outranks a local-tier one (status.replay too)
            return (not str(getattr(x[2], "reason", "")).startswith("local re-root"), x[2].mean, x[2].e, x[0].id)
        if win:                                        # its seeds become the new incumbent's SELECT (CRN) block
            from loop.core.seeds import SeedBlock
            b, c, v = max(win, key=rank)
            self._put_block(SeedBlock(f"remint-{c}-{int(now)}", CD.SELECT, tuple(b.seeds), (c,),
                                      f"remint {c}", now), acts)
        for b, c, _ in passes:
            if not win or c != max(win, key=rank)[1]:
                acts.append(f"{c} PASSED without a switch (vs {b.node_ids[-1]}): re-testable against the incumbent")
        for b in [b for b in self.blocks() if b.pool == CD.TEST and b.open]:
            changed = self._dead_pairs(b, data, now, acts) or changed
        return changed

    def _fnf_pause_expiry(self) -> list[str]:
        """``ideation.fnf_pause_max_d``: a fork-and-flush pause that nothing can lift grants one round (alloc)."""
        from loop.core import alloc as AL
        from loop.core.levers import opt
        if self.dry:
            return []
        return AL.fnf_pause_expiry(self.store, self.clock(), float(opt(self.ctx, "fnf_pause_max_d")))

    def _sync_node_status(self, statuses: Mapping[str, Any], now: float, acts: list[str]) -> int:
        """v3 derives every status each tick and used to store none (7 days: ONE ``node_transition``; every recent node
        read queued / screening), so each stored-status consumer -- ``loop status``, the dashboard, quality, scout,
        explain, the ideation queue lists, prune's LIVE set, the databank's busy / retention -- saw a stale lifecycle.
        After derive, write each node's derived status back to ``nodes.status`` when it changed (``SYNC_TO``), but only
        over a live stored status or one this sync wrote (``SYNC_FROM`` / ``K_NODE_SYNC``): explicit terminal acts
        (rejected / vetoed / baseline / confirmed / screened_out ...) and PASSED / INCUMBENT are never touched.
        ``updated_ts`` is preserved (queue age windows and prune's grace read it). One ``node_transition`` per change,
        or ONE ``node_status_sync`` summary when more than ``SYNC_EVENT_MAX`` change (the first pass; and no per-node
        ``close`` timestamps for prune's grace). Returns how many were written."""
        owned = dict(self.store.get_state(K_NODE_SYNC) or {})
        moves: list[tuple[Any, str]] = []
        for node in self.store.nodes():
            st = statuses.get(node.id)
            to = SYNC_TO.get(status_name(st)) if st is not None else None
            if to is None or to == node.status.value or node.id == ROOT_NODE_ID:
                continue
            if node.status.value in SYNC_FROM or owned.get(node.id) == node.status.value:
                moves.append((node, to))
        if not moves:
            return 0
        from loop.contracts import NodeStatus
        for node, to in moves:
            # a TERMINAL derived status starts prune's grace for a newly failed node AT CLOSE (updated_ts = now); a live
            # one keeps its ts (queue age windows read it)
            self.store.update_node(node.id, status=NodeStatus(to),
                                   updated_ts=now if to in SYNC_TERMINAL and len(moves) <= SYNC_EVENT_MAX
                                   else node.updated_ts)
            owned[node.id] = to
        self.store.set_state(K_NODE_SYNC, owned)
        if len(moves) > SYNC_EVENT_MAX:
            flips = collections.Counter(f"{n.status.value}->{to}" for n, to in moves)
            self.store.log("node_status_sync", n=len(moves), flips=dict(flips))
            acts.append(f"node status sync: {len(moves)} stored statuses brought to the derived ones "
                        f"({', '.join(f'{k} {v}' for k, v in flips.most_common())})")
        else:
            for node, to in moves:
                self.store.log("node_transition", node_id=node.id, frm=node.status.value, to=to, reason="derived status")
        return len(moves)

    def _sync_carriers(self, data: CD.Ledger, noise: Any, statuses: Mapping[str, Any], acts: list[str]) -> bool:
        """A v3 TEST PASS that switches the incumbent is a CARRIER from the moment its block closed, as an audit pass /
        an ``incumbent`` event of the old path is (``data.load`` reads ``carrier`` events with their ``at``): children
        of the incumbent sit in its carrier cell of the prior, not in the root's (10-14: a node switched by
        TEST and logged neither, so its children were root-cell). Every switch of the replayed chain not yet a carrier
        is logged once (``carrier`` node_id / previous / block / at / source=test_pass); cheap when the derived
        incumbent is a carrier already. True when one was logged (the caller reloads the ledger)."""
        from loop.core import status as ST
        known = {c for _, c in data.carriers}
        inc = incumbent_of(statuses) if statuses else ""
        if not inc or inc in known or inc in (CD.ROOT, data.base_root) or noise is None:
            return False
        logged = False
        for j in ST.replay(data, self.blocks(), noise):
            if j.switched and j.node not in known:
                known.add(j.node)
                self.store.log("carrier", node_id=j.node, previous=j.vs, block=j.block, at=float(j.ts),
                               source="test_pass")
                acts.append(f"{j.node}: TEST-pass incumbent (block {j.block}) registered as a carrier")
                logged = True
        return logged

    def _dead_pairs(self, b: Any, data: CD.Ledger, now: float, acts: list[str]) -> bool:
        """The dead pairs of open TEST block ``b``. An INFRA death (an infra void past its retries, a loop cancel, a
        too_slow void, a shim / checkpoint crash, an eval-only run's eval-lane error: :func:`test_fault`) is no one's
        fault: its seed is re-drawn, without limit. A MEMBER death (a crash, an eval void) is counted per side over
        EVERY seed the block ever drew, every tick (v3-c5b M4: the tally ran only when a current seed was dead, so
        a node's 4 probe crashes on since-replaced seeds never closed its block): below ``test.max_candidate_crashes``
        its seed is re-drawn too; at it the block closes -- the candidate's crashes as FAIL ``unscorable``, the
        control's (the incumbent's: not the candidate's fault) with NO verdict, the candidate re-testable after
        ``test.retest_backoff_h``. A crash:probe on a current seed is first retried ONCE on the same seed (S3: a probe
        flake is not the candidate's); only the retry's crash counts (and replaces the seed). A crash:probe on a seed
        replaced before that rule (never retried) still counts. Either close logs the crash classes and cancels the
        candidate's running runs of the block. True if the block changed."""
        runs = {(n, s): self.store.get_run(Run.make_id(n, self._test_stage(n, b, s), s))
                for n in b.node_ids for s in b.seeds}
        retried = set(self.store.get_state(K_PROBE_RETRY) or ())
        waiting, n_retried = set(), len(retried)
        for (n, s), r in sorted(runs.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            if r is None or r.status is not RunStatus.CRASH or crash_class(r) != "crash:probe" or r.id in retried:
                continue
            node = self.store.get_node(n)
            if node is not None and self.book.free() > 0 and (got := self.guarded(
                    f"run:{r.id}", f"retry {r.id} (crash:probe, once)", lambda r=r, node=node: self.book.retry(r, node),
                    acts)) is not None:
                retried.add(r.id)
                runs[(n, s)] = got
                self.store.log("test_probe_retry", block=b.id, run_id=r.id, node_id=n, seed=s)
                acts.append(f"retry {r.id} once on its seed (crash:probe; block {b.id}) -> {got.status.value}")
            else:
                waiting.add(s)                           # no slot yet: neither re-drawn nor counted
        if len(retried) != n_retried:
            self.store.set_state(K_PROBE_RETRY, sorted(retried))
        dead = [s for s in b.seeds if s not in waiting and any(self._dead(runs[(n, s)]) for n in b.node_ids)]
        cap = int(getattr(getattr(self.ctx.cfg, "test", None), "max_candidate_crashes", 2))
        ever = set(b.seeds)
        for e in self.store.events("seeds_drawn"):
            if e.get("block") == b.id:
                ever.update(int(s) for s in e.get("seeds") or ())
        ctrl, cands = b.node_ids[-1], list(b.node_ids[:-1])
        eo = getattr(self.core, "eo_runs", frozenset()) or frozenset()
        faults: dict[str, list[tuple[str, str]]] = {}            # member -> [(run id, crash class)]
        forgiven = self.store.get_state(self.K_ILL_FLAKE) or {}   # v3 review K4: ill-conditioned flakes, re-drawn
        for n in b.node_ids:
            for s in sorted(ever):
                r = runs.get((n, s)) or self.store.get_run(Run.make_id(n, self._test_stage(n, b, s), s))
                if self._dead(r) and r.id not in forgiven and test_fault(r, eval_run=r.id in eo or self._planned(r),
                                                pending_retry=s in b.seeds and r.id not in retried):
                    faults.setdefault(n, []).append((r.id, crash_class(r)))
        ctrl_bad = bool(cands) and cap > 0 and len(faults.get(ctrl, ())) >= cap
        bad = [] if ctrl_bad else [c for c in cands if cap > 0 and len(faults.get(c, ())) >= cap]   # control first
        if not bad and not ctrl_bad:
            if not dead:
                return False
            new = self.core.draw(len(dead), data, self.blocks(), self.ctx.cfg)
            why = "; ".join(f"{s}: " + ",".join(f"{n} {crash_class(runs[(n, s)])}" for n in b.node_ids
                                                if self._dead(runs[(n, s)])) for s in dead)
            self._update_block(b.id, acts, f"dead pairs on {dead} replaced by {list(new)} ({why})",
                               seeds=(*[s for s in b.seeds if s not in dead], *new))
            self._log_test_draw(b, new, "replace dead pairs")
            return True
        from loop.core import status as ST
        from loop.core import test as T
        classes = {n: _tally([k for _, k in f]) for n, f in faults.items()}
        recs, ill, flake = [], [], []
        flakes = self.store.get_state(self.K_ILL_FLAKE) or {}
        for c in [c for c in bad if all(k == ILL_CONDITIONED for _, k in faults[c])]:
            # v3-c7a A4: every crash an ill-conditioned probe -- an idea defect (v3 review K4: >= 2 such crashes and
            # >= 1/3 of its finished runs, or a readout diff; once per block a flake is re-drawn instead): no FAIL, a
            # repair session that keeps the block's scorable seeds' evidence when that is positive (else a FAIL);
            # the candidate backs off retesting (test.ill_backoff_h)
            mine = [runs.get((c, s)) or self.store.get_run(Run.make_id(c, self._test_stage(c, b, s), s))
                    for s in sorted(ever)]
            real, why = self.ill_defect(data, c, mine)
            if not real and not any(str(f).startswith(f"{b.id}|{c}|") for f in flakes):
                flake.append((c, why))
                continue
            ok = [s for s in b.seeds if all(runs[(n, s)] is not None and runs[(n, s)].status is RunStatus.DONE
                                            for n in (c, ctrl))]
            ev = self.evidence(data, c, ctrl, ok)
            if ev.get("mean") is None or float(ev["mean"]) <= 0:
                continue                                 # no promising evidence: the candidate's crashes are a FAIL
            self.book_repair(data, c, [int(r.split(":")[-1]) for r, _ in faults[c] if r.split(":")[-1].isdigit()],
                             acts, source=f"test {b.id}", evidence=ev)
            self.store.log("test_ill_conditioned", block=b.id, node_id=c, control=ctrl, candidates=[c],
                           crashes=[i for i, _ in faults[c]], evidence=ev,
                           backoff_until=now + 3600.0 * float(getattr(getattr(self.ctx.cfg, "test", None),
                                                                      "ill_backoff_h", 72.0)))
            ill.append(c)
        if flake and not ctrl_bad and len(flake) == len(bad):
            # every candidate fault a flake: forgiven ONCE per block -- the crashes no longer count, their seeds
            # are re-drawn like any dead pair
            self._flake([*(i for c, _ in flake for i, _ in faults[c]), *(f"{b.id}|{c}|" for c, _ in flake)])
            gone = sorted({s for c, _ in flake for s in b.seeds if self._dead(runs[(c, s)])})
            if gone:
                new = self.core.draw(len(gone), data, self.blocks(), self.ctx.cfg)
                self._update_block(b.id, acts, "probe ill-conditioned flake(s) " + "; ".join(
                    f"{c} ({w})" for c, w in flake) + f": seeds {gone} re-drawn as {list(new)}",
                    seeds=(*[s for s in b.seeds if s not in gone], *new))
                self._log_test_draw(b, new, "replace ill-conditioned flake")
            for c, w in flake:
                self.store.log("probe_ill_flake", block=b.id, node_id=c, why=w, crashes=[i for i, _ in faults[c]])
            return True
        bad = [c for c in bad if c not in ill]
        for c in bad:                    # the candidate's own crashes: a FAIL (judged; re-tested only as a new node)
            ok = {s for s in b.seeds if all(runs[(n, s)] is not None and runs[(n, s)].status is RunStatus.DONE
                                            for n in (c, ctrl))}
            p = ST.paired(data, c, ctrl, ok) if ok else None
            n_ok = len(p.panel) if p is not None else 0
            mean = sum(p.panel) / n_ok if n_ok else 0.0
            v = T.Verdict(T.Kind.FAIL, 0.0, n_ok, mean, f"unscorable: {len(faults[c])} candidate crashes "
                                                          f"({classes[c]})")
            recs.append(ST.record(c, ctrl, v, ts=now))
            self.store.log("test_verdict", block=b.id, node_id=c, verdict="fail", e=0.0, n=n_ok, mean=mean,
                           unscorable=True, crashes=[i for i, _ in faults[c]], crash_class=classes[c])
        if bad or ill:
            note = "; ".join([*(f"{c} FAIL (unscorable: {len(faults[c])} crashes, {classes[c]})" for c in bad),
                              *(f"{c} NO VERDICT (probe ill-conditioned: {len(faults[c])} crashes; repair queued)"
                                for c in ill)])
        else:
            note = f"unscorable (control): {ctrl} {len(faults[ctrl])} crashes ({classes[ctrl]}), no verdict"
            self.store.log("test_unscorable_control", block=b.id, control=ctrl, candidates=cands,
                           crashes=[i for i, _ in faults[ctrl]], crash_class=classes[ctrl],
                           backoff_until=now + 3600.0 * float(getattr(getattr(self.ctx.cfg, "test", None),
                                                                      "retest_backoff_h", 12.0)))
        self._update_block(b.id, acts, f"{note}: closed", closed_ts=now, verdicts=tuple(recs))
        seeds = set(b.seeds)
        for c in cands:                  # its remaining runs of this block pair with nothing now
            for run in self.store.runs(node_id=c, status=[RunStatus.RUNNING]):
                if run.seed in seeds and run.stage in TEST_STAGES \
                        and self.book.cancel_run(run, f"test block {b.id} closed unscorable"):
                    acts.append(f"cancelled {run.id} (block {b.id} unscorable)")
        return True

    def test_owed(self) -> int:
        """TEST-stage runs the open TEST blocks still owe: a (member, seed) of an open TEST block with no run row, or
        whose row is an infra void with attempts left (re-submitted). Candidates run CONFIRM, the control REMINT."""
        n = 0
        for b in self.blocks():
            if b.pool != CD.TEST or not b.open or not b.node_ids:
                continue
            for node in b.node_ids:
                stage = TEST_CONTROL_STAGE if node == b.node_ids[-1] else TEST_STAGE
                for seed in b.seeds:
                    r = self.store.get_run(Run.make_id(node, stage, seed))
                    n += r is None or (r.status is RunStatus.VOID and r.void_type in (None, VoidType.INFRA)
                                       and r.attempt < self.book.infra_retry)
        return n

    def _reserve_tests(self, acts: list[str]) -> None:
        """v3-c6a (live 10-13: a node's TEST block waited 3 h for primary -- the poll's repools of preempted screens /
        wave points and idle-churn moves took every primary slot that freed, before the launch phase reached the
        owed test runs): the owed TEST runs RESERVE that many primary slots for this tick (``RunBook.reserved``;
        ``slurm.test_reserve``). The count is in state ``runbook.test_reserve`` (diagnose defers its primary job)."""
        try:
            n = self.test_owed() if getattr(self.ctx.cfg.slurm, "test_reserve", True) else 0
        except Exception as exc:  # noqa: BLE001 - no reserve this tick (logged)
            self._call_error("test.reserve", "count the owed TEST runs", exc, acts, quarantine=False)
            n = 0
        n = self._reserve_timeout(n, acts)
        self.book.test_reserve = n
        if (self.store.get_state(R.K_TEST_RESERVE) or 0) != n:
            self.store.set_state(R.K_TEST_RESERVE, n)
            if n:
                acts.append(f"test reserve: {n} owed TEST run(s) hold primary first")

    K_RESERVE_SINCE = "runbook.test_reserve_since"     # {n, since, released}: the reserve's timeout (v3-c6r M1)

    def _reserve_timeout(self, n: int, acts: list[str]) -> int:
        """v3-c6r M1: the reserve is RELEASED (0) once the owed count has not moved for ``slurm.test_reserve_max_h``
        hours (the owed TEST runs cannot launch -- a missing source, a held node --, so the held primary slots would
        idle forever); logged ``test_reserve_released``. Any change of the owed count re-arms it."""
        max_h = float(getattr(self.ctx.cfg.slurm, "test_reserve_max_h", 3.0) or 0.0)
        now = self._tick_ts if self._tick_ts is not None else self.clock()
        st = self.store.get_state(self.K_RESERVE_SINCE) or {}
        if n <= 0 or max_h <= 0:
            if st:
                self.store.set_state(self.K_RESERVE_SINCE, None)
            return n
        if int(st.get("n", -1)) != n:
            self.store.set_state(self.K_RESERVE_SINCE, {"n": n, "since": now, "released": False})
            return n
        if st.get("released"):
            return 0
        if now - float(st.get("since") or now) >= 3600.0 * max_h:
            self.store.set_state(self.K_RESERVE_SINCE, {**st, "released": True})
            self.store.log("test_reserve_released", owed=n, held_h=round((now - float(st["since"])) / 3600.0, 2))
            acts.append(f"test reserve released: {n} owed TEST run(s) could not launch for {max_h:g} h "
                        "(primary slots returned until the owed count changes)")
            return 0
        return n

    def _infra_void_only(self, nid: str, statuses: Mapping[str, Any]) -> bool:
        """``nid`` derives VOID only because every run of it is an infra void (no verdict: "a void is not dead"), and
        its stored status is a live one or the sync's own (never an explicit act: rejected / vetoed / ...). Live
        11-11: a node's one screen voided on submit_failed (a half-created worktree), derived VOID, and CLOSED kept
        ``_retries`` from ever re-submitting it although it had all its infra retries left. Each run's own
        ``infra_retry`` cap still applies in ``_retries``: at the cap the node stays VOID. A node closed on purpose
        (``rescue_closed=1``: a rescue point that never ran; ``port_failed``: a failed graft) is never revived."""
        if status_name(statuses.get(nid, "")) != "VOID" or (node := self.store.get_node(nid)) is None:
            return False
        notes = f" {node.notes or ''} "
        if " rescue_closed=1" in notes or "port_failed" in notes:
            return False
        owned = (self.store.get_state(K_NODE_SYNC) or {}).get(nid)
        if node.status.value not in SYNC_FROM and owned != node.status.value:
            return False
        runs = self.store.runs(node_id=nid)
        return bool(runs) and all(r.status is RunStatus.VOID and r.void_type in (None, VoidType.INFRA) for r in runs)

    def _retries(self, statuses: Mapping[str, Any], acts: list[str]) -> list[str]:
        """Infra voids are no verdict: the same row is re-submitted while attempts are left (a cancel never)."""
        out, infra_only, reopened = [], {}, set()
        for run in self.store.runs(status=[RunStatus.VOID]):
            if run.void_type not in (None, VoidType.INFRA) or run.attempt >= self.book.infra_retry \
                    or self.book.free() <= 0:
                continue
            if status_name(statuses.get(run.node_id, "")) in CLOSED:
                if run.node_id not in infra_only:
                    infra_only[run.node_id] = self._infra_void_only(run.node_id, statuses)
                if not infra_only[run.node_id]:
                    continue
            if (node := self.store.get_node(run.node_id)) is None or not self.book.pool(node, run.stage, run.seed):
                continue
            try:
                run = self.book.retry(run, node)
            except SubmissionRejected as exc:          # deterministic: the row is void for good; the rest go on
                acts.append(f"retry {run.id} rejected: {exc}"[:300])
                continue
            acts.append(f"retry {run.id} -> {run.status.value}")
            out.append(run.id)
            if infra_only.get(run.node_id) and run.status is not RunStatus.VOID and run.node_id not in reopened:
                self._reopen(node)
                reopened.add(run.node_id)
        return out

    def _reopen(self, node: Any) -> None:
        """An infra-only VOID node whose row was re-submitted is live again: its stored status goes back to
        ``screening`` at once (owned by the sync, ``K_NODE_SYNC``; its ``updated_ts`` kept) so rescue, struct_tune
        and leader -- which read the stored status -- stop treating it as dead before the next derive."""
        from loop.contracts import NodeStatus
        self.store.update_node(node.id, status=NodeStatus.SCREENING, updated_ts=node.updated_ts)
        owned = dict(self.store.get_state(K_NODE_SYNC) or {})
        owned[node.id] = NodeStatus.SCREENING.value
        self.store.set_state(K_NODE_SYNC, owned)
        self.store.log("node_transition", node_id=node.id, frm=node.status.value, to=NodeStatus.SCREENING.value,
                       reason="infra retry re-submitted")

    def _launch(self, actions: Sequence[Any], data: CD.Ledger, statuses: Mapping[str, Any], now: float,
                acts: list[str]) -> list[str]:
        """The specs of every action (each action guarded: one that raises is skipped with a logged reason and,
        repeated, quarantined), then the submits while slots are free."""
        cfg = self.ctx.cfg
        owed = [Action("test", c, "", 0.0, 0.0) for b in self.blocks() if b.pool == CD.TEST and b.open
                for c in b.node_ids[:-1]]
        g = getattr(self.core, "_gains", None) or {}  # 11-10: the stronger test's owed runs first (stable otherwise)
        owed.sort(key=lambda a: -g[a.node][0] if a.node in g and math.isfinite(g[a.node][0]) else math.inf)
        specs: list[RunSpec] = []
        self.book.fill_only = set()
        if hasattr(self.book, "prune_test_wait"):      # a closed TEST block's waits go with it
            self.book.prune_test_wait({(n, int(s)) for b in self.blocks() if b.pool == CD.TEST and b.open
                                       for n in b.node_ids for s in b.seeds})
        from loop.core import factorial as F
        self._ported = []

        def act(group: Sequence[Any]) -> None:
            for a in group:
                if a.kind == "idea":
                    continue
                specs.extend(self.guarded(f"action:{a.kind}:{a.node}", f"{a.kind} {a.node}",
                                          lambda a=a: self._act(a, data, statuses, now, acts), acts, default=[]) or [])
        # v3-c5b M2: the order the slots go in -- (1) the runs open TESTs owe, (2) the primary-pass ranked actions,
        # (3) the never-ran points of open / half-built waves, at most factorial.resume_points_per_tick per tick
        # (live 10-13: 16 leftover stacking points went ahead of owed test runs every tick), (4) the relax pass and
        # the idle fill (preemptible)
        # a half-built wave's build is finished FIRST (a ranked wave action would otherwise resume it a 2nd time this
        # tick, counted as a crash); its points launch right after the primary pass (the build's own launch)
        resumed = list(self.guarded("wave.resume", "resume a half-built wave",
                                    lambda: self._resume_wave(data, now, acts), acts, default=[]) or [])
        act(owed)
        specs += self.guarded("alt_root.port", "auto-port a positive root child onto the incumbent",
                              lambda: self._alt_root_ports(data, statuses, now, acts), acts, default=[]) or []
        specs += self.guarded("reroot.port", "port the old incumbent's best children onto the new incumbent",
                              lambda: self._reroot_ports(data, statuses, now, acts), acts, default=[]) or []
        late = [a for a in actions if a.kind == "fill" or getattr(a, "pre", False)]
        act([a for a in actions if a not in late])
        specs += resumed                               # the resumed build's own launch: not capped
        left: list[RunSpec] = []
        pre_pts: set[tuple[str, int]] = set()
        for wid in self._open_waves(data):             # an open wave's points that never ran (a crash-stopped launch)
            pts = [RunSpec(n, SEED_STAGE, s) for n, s in F.wave_runs(wid, data)] + [
                RunSpec(n, Stage.PARENT, s) for n, s in F.wave_controls(data, wid)]   # v3-c7a A2: its controls
            if pts and (self.store.get_state(f"wave:{wid}") or {}).get("pre"):   # a relax wave stays preemptible
                pre_pts |= {(s.node, s.seed) for s in pts}
            left += pts
        have = set(specs)
        left = [s for s in dict.fromkeys(left) if s not in have
                and self.store.get_run(Run.make_id(s.node, s.stage, s.seed)) is None]
        self.guarded("wave.controls", "re-submit dead wave control rows",                       # v3 review O4
                     lambda: self._retry_controls(data, acts), acts)
        cap = max(0, int(self._fcfg("resume_points_per_tick", 4)))
        if len(left) > cap:
            acts.append(f"open waves: {cap} of {len(left)} never-ran point(s) this tick (factorial.resume_points_per_tick)")
        self.book.fill_only |= {(s.node, s.seed) for s in left[:cap] if (s.node, s.seed) in pre_pts}
        specs += left[:cap]
        act(late)
        specs = [s for s in dict.fromkeys(specs) if self.store.get_run(Run.make_id(s.node, s.stage, s.seed)) is None]
        routed = self.guarded("evalonly.route", "route readout-only runs onto parent checkpoints",   # v3-c5 I5
                              lambda: self._eo().route(data, specs, acts), acts, default=None)
        specs = specs if routed is None else routed
        nodes = {s.node: self.store.get_node(s.node) for s in specs}
        self.book.high_demand = sum(self.book.high(nodes[s.node], s.stage, s.seed) for s in specs)
        t_sub = time.monotonic()
        out, ready = [], []
        # 11-10 (nanobrain test-a node): an open TEST's runs of one seed (candidate CONFIRM + control REMINT) not yet
        # placed go WHOLE -- a lone candidate on the last free primary slot left its control for a later slot (root
        # 100001 still owed 40 min after a node's confirm 100001 started). The reserve keeps the slot meanwhile.
        group, held, placed, seen = self._test_groups(specs), set(), set(), set()
        whole = hasattr(self.book, "test_room") and hasattr(self.book, "prepare_launch") \
            and self.book.reserved() > 0          # review: once the reserve is released, one by one as before
        todo, i = list(specs), 0
        while i < len(todo):                      # the rows (slot accounting) serially, then the submits batched
            s, i = todo[i], i + 1
            if s in seen or s in held:
                continue
            if self.book.free() <= 0:
                break
            g = group.get(s, ())
            if whole and len(g) > 1 and s not in placed and all(nodes.get(m.node) is not None for m in g) \
                    and self.book.test_primary_only(s.stage):
                # every member asks for its pool (review: each one's spill-wait clock starts together); the group goes
                # only when all land on ONE pool with room for all of them, and its other members are prepared right
                # after this one (no other spec takes the slots in between)
                rooms = [self.book.test_room(nodes[m.node], m.stage, m.seed) for m in g]
                pools = {self.book.pool(nodes[m.node], m.stage, m.seed) for m in g}
                if len(pools) != 1 or "" in pools or min(rooms) < len(g):
                    held |= set(g)
                    if (s.node, s.seed) not in self._pair_wait:      # said once per wait, not every tick
                        acts.append(f"test pair {s.node}:{s.stage.value}:{s.seed}: {len(g)} run(s) wait for "
                                    f"{len(g)} free slots together")
                    continue
                placed |= set(g)
                todo[i:i] = [m for m in g if m != s]
            seen.add(s)
            # time-box: the rest is re-ranked next tick -- after >= min_submits (a late tick still launches)
            if s not in placed and self.over_budget() and len(ready) + len(out) >= self._hb("min_submits", 8):
                acts.append(f"tick budget ({self._budget():.0f} s) spent: {len(todo) - i + 1} run(s) deferred")
                break
            if (node := nodes[s.node]) is None:
                acts.append(f"skip {s.node}:{s.stage.value}:{s.seed}: unknown node")
                continue
            key = f"run:{s.node}:{s.stage.value}:{s.seed}"
            if not hasattr(self.book, "prepare_launch"):
                if (run := self.guarded(key, f"submit {key[4:]}", lambda s=s, node=node: self._submit(node, s, acts),
                                        acts)) is not None:
                    acts.append(f"submit {run.id} -> {_submitted(run)}")
                    out.append(run.id)
                continue
            if (row := self.guarded(key, f"submit {key[4:]}", lambda s=s, node=node: self.book.prepare_launch(
                    node, s.stage, s.seed), acts)) is not None:
                ready.append((key, s, node, row))
        self._pair_wait = frozenset((s.node, s.seed) for s in held)
        if ready:
            got = self.guarded("submit.batch", "submit the prepared runs",
                               lambda: self.book.submit_prepared(
                                   [(row, node) for _, _, node, row in ready], stop=self.over_budget,
                                   min_n=max(0, self._hb("min_submits", 8) - len(out))), acts,
                               default=None)
            for (key, s, _, row), run in zip(ready, got or [None] * len(ready)):
                if isinstance(run, SubmissionRejected):
                    acts.append(f"submit {s.node}:{s.stage.value}:{s.seed} rejected: {run}"[:300])
                elif isinstance(run, BaseException) or run is None:
                    continue
                else:
                    self._call_ok(key)
                    acts.append(f"submit {run.id} -> {_submitted(run)}")
                    out.append(run.id)
        # the slots still free after every ranked action: churned points of waves without a result get them
        out += self.guarded("wave.resubmit", "resubmit churned wave points",
                            lambda: self._resubmit_churned(data, acts), acts, default=[]) or []
        self._time("launch.submit", t_sub)
        return out

    _pair_wait: frozenset[tuple[str, int]] = frozenset()     # TEST pairs held back last tick (logged once per wait)

    def _test_groups(self, specs: Sequence[RunSpec]) -> dict[RunSpec, tuple[RunSpec, ...]]:
        """spec -> the specs of its (open TEST block, seed) group in ``specs`` (TEST stages only, no run row yet):
        the candidate(s) and the control of one seed, placed whole by :meth:`_launch`."""
        try:
            blocks = [b for b in self.blocks() if b.pool == CD.TEST and b.open and b.node_ids]
        except Exception:  # noqa: BLE001 - no grouping: the runs go one by one as before
            return {}
        key = {(n, int(sd)): (b.id, int(sd)) for b in blocks for n in b.node_ids for sd in b.seeds}
        by: dict[tuple[str, int], list[RunSpec]] = {}
        for s in specs:
            if s.stage in TEST_STAGES and (k := key.get((s.node, int(s.seed)))) is not None:
                by.setdefault(k, []).append(s)
        return {s: tuple(g) for g in by.values() for s in g}

    # -- eval-only readout lane (v3-c5 I5, loop.evalonly.Lane) --------------------------------------------------------
    def _eo(self) -> Any:
        if getattr(self, "_eo_lane", None) is None:
            from loop.evalonly import Lane
            self._eo_lane = Lane(self.ctx, self.store, self.adapter, self.clock)
        return self._eo_lane

    def _test_backoff(self, now: float) -> frozenset[str]:
        """Candidates of a TEST closed for an unscorable CONTROL less than ``test.retest_backoff_h`` ago (S3: the
        incumbent kept crashing; re-testing at once re-ran into the same control)."""
        h = float(getattr(getattr(self.ctx.cfg, "test", None), "retest_backoff_h", 12.0))
        out = set()
        for kind in ("test_unscorable_control", "test_ill_conditioned"):    # v3-c7a A4: the repair waits first
            for e in self.store.events(kind):
                until = float(e.get("backoff_until") or (float(e.get("ts") or 0.0) + 3600.0 * h))
                if until > now:
                    out.update(str(c) for c in e.get("candidates") or ())
        return frozenset(out)

    def _planned(self, run: Any) -> bool:
        """``run`` ran (or was routed) with an eval plan (a crashed eval-only run has no obs: not in ``eo_runs``)."""
        return hasattr(self.adapter, "eval_plan") and self._eo().plan_of(run.node_id, run.stage, run.seed) is not None

    def _eo_runs(self, data: CD.Ledger, acts: list[str]) -> None:
        """The core's ``eo_runs`` (v3-c5b M1): the runs with an eval plan, left out of every noise fit / pair pool.
        On a failure the previous tick's set stands (runs only gain plans before they launch)."""
        if hasattr(self.core, "eo_runs") and hasattr(self.adapter, "eval_plan"):
            got = self.guarded("evalonly.runs", "find the eval-only runs", lambda: self._eo().eval_runs(data), acts)
            if got is not None:
                self.core.eo_runs = frozenset(got)

    def _meta_ledger(self) -> CD.Ledger:
        from loop.core import holdout as HO
        try:
            return HO.visible_panel(self.vis)
        except Exception as exc:  # noqa: BLE001 - the visible ledger itself, panel official
            self.store.log("tick_call_error", key="holdout.visible_panel", error=repr(exc)[:300])
            return self.vis

    def _eo_prepare(self, data: CD.Ledger, acts: list[str]) -> frozenset[str]:
        """The pure eval-only nodes -> the core's posterior (their own pair noise); returns the harm-held nodes."""
        from loop.evalonly import floor_sd
        # S5: the harm guard acts before TEST -- open TEST blocks' members and the incumbent are left alone
        protect = {n for b in self.blocks() if b.pool == CD.TEST and b.open for n in b.node_ids}
        protect.add(incumbent_of(getattr(self, "statuses", None) or {}))
        pure, harm = self._eo().prepare(data, acts, protect, full=self.data)
        if hasattr(self.core, "evalonly"):
            self.core.evalonly, self.core.eo_floor = pure, floor_sd(self.ctx)
            self.core.eo_min_df = int(self.ctx.cfg.evalonly.noise_min_df)
        return frozenset(harm)

    def _hb(self, name: str, default: int) -> int:
        """``heartbeat.<name>`` (an int floor of the time box), else ``default``."""
        return int(getattr(self.ctx.cfg.heartbeat, name, default))

    def _budget(self) -> float:
        return float(getattr(self.ctx.cfg.heartbeat, "tick_budget_s", 0.0) or 0.0)

    def over_budget(self) -> bool:
        """This tick has run longer than ``heartbeat.tick_budget_s`` (0 = no budget): wave builds pause (journalled,
        resumed next tick) and no more runs are submitted (re-ranked next tick). False outside a tick."""
        t0, b = getattr(self, "_t0", None), self._budget()
        return t0 is not None and b > 0 and time.monotonic() - t0 > b

    def _time(self, key: str, t0: float) -> None:
        """Add the seconds since ``t0`` to ``phase_s[key]`` (a sub-phase: launch.waves, launch.submit)."""
        ps = self.__dict__.setdefault("phase_s", {})
        ps[key] = ps.get(key, 0.0) + time.monotonic() - t0

    @property
    def phase(self) -> str:
        return self.__dict__.get("_phase", "")

    @phase.setter
    def phase(self, value: str) -> None:
        """Where the tick is (a tick_error names it); the time spent in the previous phase goes to ``phase_s``."""
        t, prev, t0 = time.monotonic(), self.__dict__.get("_phase", ""), self.__dict__.get("_phase_t0")
        ps = self.__dict__.setdefault("phase_s", {})
        if prev and t0 is not None and prev != "done":
            ps[prev] = ps.get(prev, 0.0) + t - t0
        self.__dict__["_phase"], self.__dict__["_phase_t0"] = value, t

    def timing_text(self) -> str:
        """`` phase_s=poll:11,intake:35,launch:400,launch.waves:390,...`` (phases >= .5 s; '' when none)."""
        ps = getattr(self, "phase_s", None) or {}
        items = [f"{k}:{v:.0f}" for k, v in ps.items() if v >= 0.5]
        return f" phase_s={','.join(items)}" if items else ""

    def _submit(self, node: Any, s: RunSpec, acts: list[str]) -> Run | None:
        try:
            return self.book.launch(node, s.stage, s.seed)
        except SubmissionRejected as exc:              # deterministic: the row is void for good; the rest go on
            acts.append(f"submit {s.node}:{s.stage.value}:{s.seed} rejected: {exc}"[:300])
            return None

    def _act(self, a: Any, data: CD.Ledger, statuses: Mapping[str, Any], now: float, acts: list[str]) -> list[RunSpec]:
        """The run specs of one ranked (or owed) action; the wave / port hooks guarded on their own keys."""
        cfg, inc = self.ctx.cfg, incumbent_of(statuses)
        if a.kind == "wave":
            if self.over_budget() and not self.store.get_state(K_WAVE_BUILD) \
                    and self.__dict__.get("_wpts", 0) >= self._hb("min_wave_points", 1):
                acts.append(f"wave on {a.node}: not started (tick budget spent); re-ranked next tick")
                return []
            if a.node in data.nodes and self.book.free() > 0:
                pre = bool(getattr(a, "pre", False))  # acquire.rank's relax pass: preemptible-only (persisted)
                return self.guarded(f"wave:{a.node}", f"build a wave on {a.node}",
                                    lambda: self._wave(a.node, data, now, acts, pre), acts, default=[]) or []
            return []
        specs: list[RunSpec] = []
        if a.kind == "port" or a.kind == "rebase" and data.port(a.node) is None:
            self._ported = []
            acts += self.guarded(f"port:{a.node}", f"port {a.node}", lambda: self.port(self, a, data), acts,
                                 default=[], on_quarantine=lambda why: self.reject_port(a.node, data, why)) or []
            k = getattr(getattr(self.core, "costs", None), "idea_runs", 3) if a.kind == "port" else 0
            for node, parent in self._ported if k and hasattr(self.core, "screen_specs") else ():
                specs += self.core.screen_specs(node, parent, k, data, self.blocks(), cfg)
            self._ported = []
        if a.kind == "test" and a.node in data.nodes and self.book.free() > 0:
            if (blk := self.core.open_test(a.node, inc, data, self.blocks(), now, cfg)) is not None:
                self._put_block(blk, acts)
        if a.kind in ("seed", "screen", "test", "rebase"):
            own = self.core.runs_for(a, data, self.blocks(), statuses, cfg)
            if getattr(a, "pre", False):             # acquire.rank's relax pass: below a floor -> preemptible-only
                self.book.fill_only |= {(s.node, s.seed) for s in own}
            specs += own
        if a.kind == "fill":                         # idle slots: preemptible-only (RunBook.pool), launched last
            fill = self.core.runs_for(a, data, self.blocks(), statuses, cfg)
            self.book.fill_only |= {(s.node, s.seed) for s in fill}
            specs += fill
        return specs

    def close(self) -> None:
        # R4: the ideation's brief workers first (they query the store, which closes right after a Tick does)
        if callable(stop := getattr(getattr(self.ideate, "__self__", None), "close", None)):
            stop()
        self.pool.shutdown(wait=False, cancel_futures=True)


# ---------------------------------------------------------------------------------------------- shadow mode
class DryRunAdapter:
    """No sbatch: a submit records a fake job that never finishes; a poll reports every job still running (the
    live copy's real jobs too); find / cancel / repool are no-ops. The root commit is the store's.

    Grafts never write: with ``repo`` (the task's git repo, read with ``git show`` / ``git diff`` only) a graft is the
    config merge of ``loop.adapters.merge_config`` -- a key the carrier moved elsewhere is a conflict, a change it
    already holds is ``contained`` -- and an applied one is a fake ``dry-<hash>`` commit (code-level 3-way conflicts
    are not detected in shadow); without ``repo`` every graft applies. ``config_at`` reads a commit's config."""

    def __init__(self, store: Any, clock: Callable[[], float] = time.time, *, repo: str | Path | None = None,
                 config_rel: str = "configs/main.yaml") -> None:
        self.store, self.clock, self.submitted = store, clock, []
        self.repo = Path(repo) if repo is not None and Path(repo).exists() else None
        self.config_rel, self.grafts = config_rel, []

    def _git(self, *args: str) -> str | None:
        import subprocess
        if self.repo is None:
            return None
        r = subprocess.run(["git", "-C", str(self.repo), *args], capture_output=True, text=True, timeout=60)
        return r.stdout if r.returncode == 0 else None

    def config_at(self, commit: str) -> dict[str, Any]:
        import yaml
        from loop.adapters import flatten
        text = self._git("show", f"{commit}:{self.config_rel}") if commit and not commit.startswith("dry-") else None
        try:
            return flatten(yaml.safe_load(text or "") or {})
        except yaml.YAMLError:
            return {}

    def changed_files(self, a: str, b: str) -> list[str]:
        """``git diff --name-only`` (read-only); raises without the repo / on a git failure."""
        if a.startswith("dry-") or b.startswith("dry-"):
            raise RuntimeError("a shadow-built commit has no tree")
        out = self._git("diff", "--name-only", a, b)
        if out is None:
            raise RuntimeError(f"git diff --name-only {a[:12]} {b[:12]} failed (or no repo)")
        return [x for x in out.split("\n") if x.strip()]

    def graft_port(self, parent_commit: str, commit: str, onto: str, name: str) -> tuple[str, str]:
        import hashlib
        from loop.adapters import merge_config
        fake = "dry-" + hashlib.sha1(f"{parent_commit}{commit}{onto}".encode()).hexdigest()[:12]
        self.grafts.append((name, parent_commit, commit, onto))
        if onto == parent_commit:
            return commit, "applied"
        if self.repo is None or any(self._git("cat-file", "-e", f"{c}^{{commit}}") is None
                                    for c in (parent_commit, commit, onto) if not c.startswith("dry-")):
            return fake, "applied"
        put, _, clash = merge_config(*(self.config_at(c) for c in (parent_commit, commit, onto)))
        if clash:
            return "", "conflict: " + ", ".join(f"{self.config_rel}:{k}" for k in clash)
        code = [n for n in (self._git("diff", "--name-only", parent_commit, commit) or "").split() if n != self.config_rel]
        return ("", "contained") if not code and not put else (fake, "applied")

    def diff_commits(self, a: str, b: str) -> str:
        return self._git("diff", a, b) or ""

    def graft_keys(self, commit: str, keys: Sequence[str], onto: str, name: str) -> tuple[str, str]:
        """Shadow: a key patch is a fake ``dry-<hash>`` commit unless ``onto`` already holds every key's value."""
        import hashlib
        self.grafts.append((name, "", commit, onto))
        fs, fo = self.config_at(commit), self.config_at(onto)
        if fs and fo and all(k in fo and fo[k] == fs.get(k) for k in keys):
            return "", "contained"
        return "dry-" + hashlib.sha1(f"keys{commit}{sorted(keys)}{onto}".encode()).hexdigest()[:12], "applied"

    def dry_build(self, carrier_commit: str, levels: Mapping[str, Any], knobs: Sequence[Any],
                  diffs: Mapping[str, tuple[str, str]], name: str) -> tuple[str, str]:
        """A wave point checked, not built: each diff knob that is on must graft onto the carrier."""
        import hashlib
        for kb in knobs:
            if kb.scale == "diff" and levels.get(kb.key) == kb.hi:
                sha, how = self.graft_port(*diffs[kb.key], carrier_commit, f"{name}-{kb.key}")
                if not sha:
                    return "", f"{kb.key}: {how}"
        return "dry-" + hashlib.sha1(f"{carrier_commit}{sorted(map(str, levels.items()))}".encode()).hexdigest()[:12], ""

    def file_at(self, commit: str, rel: str) -> str | None:
        """``rel`` at ``commit`` (``git show``, read-only); None for a shadow-built commit / without the repo."""
        return self._git("show", f"{commit}:{rel}") if commit and not commit.startswith("dry-") else None

    def commit_files(self, parent_commit: str, files: Mapping[str, str], *, message: str, branch: str = "") -> str:
        """Shadow: a fake ``dry-<hash>`` commit (nothing written); the edits were applied by the caller."""
        import hashlib
        self.grafts.append((message[:60], parent_commit, "", ""))
        return "dry-" + hashlib.sha1(f"{parent_commit}{sorted(files.items())}".encode()).hexdigest()[:12]

    def root_commit(self) -> str:
        root = self.store.get_node(ROOT_NODE_ID)
        return root.commit if root is not None else "dry-root"

    def submit(self, *, run: Run, commit: str) -> Any:
        from loop.contracts import JobHandle
        self.submitted.append(run.id)
        return JobHandle(run.id, f"dry-{len(self.submitted)}", "", self.clock())

    def poll(self, handle: Any) -> Any:
        from loop.contracts import JobStatus
        return JobStatus(RunStatus.RUNNING, detail="dry-run")

    def find(self, run: Run) -> Any:
        return None

    def cancel(self, handle: Any) -> None:
        return None


def refresh(live: str | Path, copy: str | Path) -> None:
    """Copy the live ledger into ``copy`` through SQLite's online backup (a consistent snapshot; the live file is
    opened read-only, never written)."""
    src = sqlite3.connect(f"file:{Path(live).resolve()}?mode=ro", uri=True)
    dst = sqlite3.connect(str(copy))
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()


def shadow_record(n: int, report: TickReport, now: float, top: int = 20) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for s in report.statuses.values():
        counts[status_name(s)] = counts.get(status_name(s), 0) + 1
    return {"tick": n, "ts": now, "incumbent": incumbent_of(report.statuses), "statuses": counts,
            "actions": [{"kind": a.kind, "node": a.node, "lane": a.lane, "gpu_h": a.gpu_h, "value": a.value,
                         "runs": getattr(a, "runs", 0), "why": getattr(a, "why", "")}
                        for a in report.actions[:top]], "would_submit": report.launched, "lines": report.lines}


def shadow(ctx: LoopContext, db: str | Path, core: Core, *, refresh_from: str | Path | None = None,
           refresh_s: float = 3600.0, max_ticks: int | None = None, every_s: float = 60.0,
           stop: Callable[[], bool] = lambda: False, clock: Callable[[], float] = time.time,
           sleep: Callable[[float], None] = time.sleep, out: Callable[[str], None] = print) -> int:
    """Ticks against the ledger COPY ``db`` with a :class:`DryRunAdapter`: nothing is submitted, no LLM session
    starts, no data-bank build or Labless submission. Each tick's derived statuses and ranked actions are printed
    and appended to ``<db>.shadow.jsonl``. ``refresh_from``: the copy is re-taken from that live ledger before the
    first tick and every ``refresh_s``. Returns the number of ticks run."""
    from loop.store import SqliteStore
    db, log = Path(db), Path(f"{db}.shadow.jsonl")
    n, last_refresh, tick, store = 0, None, None, None
    try:
        while (max_ticks is None or n < max_ticks) and not stop():
            now = clock()
            if refresh_from is not None and (last_refresh is None or now - last_refresh >= refresh_s):
                if store is not None:
                    tick.close()
                    store.close()
                    store = None
                refresh(refresh_from, db)
                last_refresh = now
                out(f"{R.utc_now(now)} shadow: copy refreshed from {refresh_from}")
            if store is None:
                store = SqliteStore(db, ctx.task_id, weights=ctx.cfg.weights)
                tick = Tick(ctx, store, DryRunAdapter(store, clock, repo=getattr(ctx.spec, "repo", None)), core,
                            db_path=db, clock=clock, dry=True)
            rec = shadow_record(n, tick.step(), now)
            with log.open("a") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
            out(f"{R.utc_now(now)} shadow tick {n}: incumbent={rec['incumbent']} statuses={rec['statuses']} "
                f"actions={[(a['kind'], a['node'], round(a['value'], 6)) for a in rec['actions'][:5]]} "
                f"would_submit={len(rec['would_submit'])}")
            n += 1
            if max_ticks is None or n < max_ticks:
                R.nap(every_s, stop, sleep)
    finally:
        if store is not None:
            tick.close()
            store.close()
    return n


# ---------------------------------------------------------------------------------------------- live (`loop run`)
#: a tick that raised is retried after poll_s * 2**(k-1) (k = tick errors in a row), at most this long
TICK_BACKOFF_MAX_S = 1800.0
#: this many tick errors in a row (of any kind): :class:`TickFatal` -- ``loop run`` exits non-zero through
#: ``os._exit`` and systemd (Restart=always) starts a fresh process
FATAL_STREAK = 10


class TickFatal(RuntimeError):
    """The live loop gives up on this process (repeated tick errors): exit, let systemd restart it."""


def live(ctx: LoopContext, db: str | Path, core: Core, adapter: Any, researcher: Any, *,
         max_ticks: int | None = None, stop: Callable[[], bool] = lambda: False, clock: Callable[[], float] = time.time,
         sleep: Callable[[float], None] = time.sleep, out: Callable[[str], None] = print,
         watchdog: Any = None, selfcheck: bool = False, **tick_kw: Any) -> int:
    """The live v3 loop (``python -m loop run``): one :meth:`Tick.step` every ``heartbeat.poll_s`` on the ledger ``db``
    with the real adapter and researcher. A stop lands between ticks (SLURM jobs keep running; they are re-adopted on
    restart); with research in flight it DRAINS first -- no new session, ticks go on until every session is harvested
    or ``researcher.drain_min`` passed (what is left is re-issued after the restart). Returns the ticks run.

    No zombie: a tick that raises is logged ``tick_error`` (phase, call site, traceback) and retried after an
    exponential backoff; the same failure QUARANTINE_AFTER times in a row at one call site (``Tick._doing``)
    quarantines that site; FATAL_STREAK errors in a row raise :class:`TickFatal` (the caller exits, systemd
    restarts). ``watchdog`` (default :class:`loop.runner.Watchdog`: sd_notify WATCHDOG=1 + ``<state>/loop.heartbeat``)
    is pinged after every tick and through the naps; a ``hb`` line (:meth:`Tick.heartbeat`) is printed and logged
    every ``heartbeat.log_every_min`` (every tick when 0) and after every failed tick.

    ``selfcheck`` (11-08, ``loop run``): :func:`loop.selfcheck.gate` runs first -- a failure is an ``ALERT_selfcheck``
    and holds every NEW research session for this process's life (runs, tests and harvests go on)."""
    from loop.store import SqliteStore
    store = SqliteStore(db, ctx.task_id, weights=ctx.cfg.weights)
    tick = Tick(ctx, store, adapter, core, db_path=db, researcher=researcher, clock=clock, **tick_kw)
    if selfcheck:
        from loop import selfcheck as SC
        try:
            hold = SC.gate(ctx, store, adapter, db=db, out=out)
        except Exception as exc:  # noqa: BLE001 - a crashing preflight holds research (fail closed), never the loop
            hold = f"selfcheck crashed: {type(exc).__name__}: {str(exc)[:300]}"
            store.log("ALERT_selfcheck", failed=["crash"], detail={"crash": hold})
            print(f"ALERT {hold} -- no research session is issued", file=sys.stderr, flush=True)
        tick.selfcheck_hold = hold
        tick.research.breaker = tick.research.breaker or hold
    state = getattr(ctx, "state_dir", "") or ""
    wd = watchdog if watchdog is not None else R.Watchdog(Path(state) / "loop.heartbeat" if state else None,
                                                          clock=clock)
    wd.notify("READY=1")
    n, drain_until, poll = 0, None, ctx.cfg.heartbeat.poll_s
    hb_every, last_hb = 60.0 * float(ctx.cfg.heartbeat.log_every_min), -float("inf")
    streak, same, last_sig = 0, 0, None

    def snooze(dt: float) -> None:
        sleep(dt)
        wd.ping()

    try:
        while max_ticks is None or n < max_ticks:
            if stop() and drain_until is None:
                if not tick.research.inflight or ctx.cfg.researcher.drain_min <= 0:
                    break
                tick.research.draining, drain_until = True, clock() + 60.0 * ctx.cfg.researcher.drain_min
                out(f"{R.utc_now(clock())} stop requested: draining {len(tick.research.inflight)} research "
                    f"session(s) for <= {ctx.cfg.researcher.drain_min:g} min; no new LLM job")
            if drain_until is not None and (not tick.research.inflight or clock() >= drain_until):
                out(f"{R.utc_now(clock())} drain done: {len(tick.research.inflight)} research session(s) left "
                    "(re-issued after the restart)")
                break
            t0, err = clock(), ""
            try:
                rep = tick.step()
            except Exception as exc:  # noqa: BLE001 - the loop never dies of one tick
                tb = traceback.format_exc()
                doing, phase = tick._doing, tick.phase
                sig = (type(exc).__name__, str(exc)[:200], phase, doing)
                same = same + 1 if sig == last_sig else 1
                last_sig, streak = sig, streak + 1
                err = (f"{type(exc).__name__} in {phase}" + (f" at {doing}" if doing else "") + f": {exc}")[:400]
                try:
                    store.log("tick_error", tick=n, phase=phase, doing=doing, error=repr(exc)[:500], streak=streak,
                              same=same, traceback=tb[-6000:])
                except Exception:  # noqa: BLE001 - the store itself may be what failed
                    pass
                out(f"{R.utc_now(clock())} tick_error ({streak} in a row, {same} identical): {err}\n{tb.rstrip()}")
                if doing and same >= QUARANTINE_AFTER:
                    lines: list[str] = []
                    try:
                        tick.quarantine(doing, err, lines)
                    except Exception:  # noqa: BLE001
                        lines.append(f"ALERT could not quarantine {doing}")
                    for line in lines:
                        out(f"{R.utc_now(clock())} {line}")
                    same, last_sig = 0, None
                if streak >= FATAL_STREAK:
                    raise TickFatal(f"{streak} tick errors in a row; last: {err}") from exc
            else:
                streak, same, last_sig = 0, 0, None
                for line in rep.lines:
                    out(f"{R.utc_now(clock())} {line}")
            dur = clock() - t0
            wd.ping(status=f"tick {n}" + (" ERROR" if err else ""), force=True)
            if err or clock() - last_hb >= hb_every:
                try:
                    hb = tick.heartbeat(n, dur, err)
                    store.log("hb", line=hb)
                except Exception as exc:  # noqa: BLE001 - the heartbeat never stops the loop
                    hb = f"hb tick={n} dur={dur:.1f}s (heartbeat failed: {exc!r})"
                out(f"{R.utc_now(clock())} {hb}")
                last_hb = clock()
            n += 1
            if max_ticks is None or n < max_ticks:
                wait = poll if not streak else min(poll * 2 ** (streak - 1), TICK_BACKOFF_MAX_S)
                R.nap(min(wait, 15.0) if drain_until is not None else wait,
                      (lambda: False) if drain_until is not None else stop, snooze)
    finally:
        wd.notify("STOPPING=1")
        tick.pruner.stop(timeout=30.0)          # a background prune pass ends between two git commands
        tick.close()
        store.close()
    return n
