"""loop.runner: the v3 plumbing -- OOM ladder + activation checkpointing, the submit guard, v3 routing and the
full-primary overflow cap, research sessions across a restart, incumbent-only Labless; and that the old driver's
RunBook (a subclass) keeps its own rules."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from loop import runner as R
from loop.adapters.sim import SimAdapter
from loop.contracts import (K_INFLIGHT, ROOT_NODE_ID, CandidateRequest, JobHandle, JobStatus, Node, NodeStatus,
                            Operator, RunStatus, Stage, Strategy, VoidType)
from loop.spec import PoolCfg, SlurmCfg, encode

from .conftest import InlineExecutor


def _pools(ctx, primary=4, idle=4, **slurm):
    return dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, slurm=SlurmCfg(pools=(
        PoolCfg("primary", slots=primary), PoolCfg("idle", slots=idle, preemptible=True)), **slurm)))


def _node(store, nid, commit="c"):
    n = Node(id=nid, task_id=store.task_id, commit=commit, parent_id=ROOT_NODE_ID, strategy=Strategy.EXPLORE,
             operator=Operator.DRAFT, family="f", mechanism_tag="m", diff_hash=f"h{nid}", status=NodeStatus.QUEUED)
    store.put_node(n)
    return n


class _Pending:
    """Fake SLURM: every job PENDING; repool recorded."""

    def __init__(self):
        self.jobs, self.moved, self.submitted = 0, [], []

    def submit(self, *, run, commit):
        self.jobs += 1
        self.submitted.append(run.id)
        return JobHandle(run.id, str(self.jobs), f"/o/{run.id}", 0.0)

    def poll(self, handle):
        return JobStatus(RunStatus.RUNNING, detail="PENDING")

    def cancel(self, handle):
        pass

    def repool(self, handle, run):
        self.jobs += 1
        self.moved.append((run.id, run.pool))
        return JobHandle(run.id, str(self.jobs), handle.output_dir, 0.0)


class _ShortLadder(SimAdapter):
    """The sim with the nanopath adapter's ladder (allocator only): the runner adds the activation-checkpoint rung."""

    def oom_ladder(self, commit):
        return ("expandable_segments",)


def test_ladder_appends_activation_checkpointing_after_the_adapters_rungs(ctx, store):
    from loop.driver import RunBook as OldBook
    adapter = _ShortLadder(ctx)
    node = _node(store, "n1", adapter.root_commit())
    assert R.RunBook(ctx, store, adapter, extra_rungs=(R.ACT_CKPT,)).ladder(node) == \
        ("expandable_segments", R.ACT_CKPT)
    assert OldBook(ctx, store, adapter).ladder(node) == ("expandable_segments",)     # the old driver: unchanged
    assert R.RunBook(ctx, store, SimAdapter(ctx), extra_rungs=(R.ACT_CKPT,)).ladder(node) == \
        ("expandable_segments", R.ACT_CKPT)                                          # no duplicate rung


def test_an_oom_climbs_to_activation_checkpointing_and_then_finishes(ctx, store):
    adapter = _ShortLadder(ctx, duration_ticks=1)
    adapter.oom_above = 10                                          # epochs 10 at root: > 10 OOMs
    commit = adapter.commit(_wt(adapter, {"train": {"epochs": 20}}), message="big")
    node = _node(store, "n1", commit)
    book = R.RunBook(ctx, store, adapter, extra_rungs=(R.ACT_CKPT,))
    book.launch(node, Stage.SCREEN, 1)
    for _ in range(4):
        book.poll()
    run = store.get_run("n1:screen:1")
    assert run.status is RunStatus.DONE and run.mitigation == R.ACT_CKPT and run.attempt == 2
    assert "oom=activation_checkpointing" in store.get_node("n1").notes
    assert any("relaunched with activation_checkpointing" in a for a in book.acts)


def _wt(adapter, cfg_update):
    import yaml
    from pathlib import Path
    wt = adapter.prepare_worktree(parent_commit=adapter.root_commit(), name="w")
    p = Path(wt.path) / "config.yaml"
    cfg = yaml.safe_load(p.read_text())
    for k, v in cfg_update.items():
        cfg.setdefault(k, {}).update(v)
    p.write_text(yaml.safe_dump(cfg))
    return wt


def test_the_guard_refuses_a_submit_and_voids_the_row_unsubmitted(ctx, store):
    adapter = _Pending()

    def guard(run):
        if run.seed >= 100_000 and run.stage is Stage.SCREEN:
            raise ValueError(f"test seed {run.seed} outside its block")
    book = R.RunBook(_pools(ctx), store, adapter, guard=guard)
    node = _node(store, "n1")
    assert book.launch(node, Stage.SCREEN, 5).status is RunStatus.RUNNING
    bad = book.launch(node, Stage.SCREEN, 100_003)
    assert bad.status is RunStatus.VOID and bad.void_type is VoidType.CANCELLED
    assert adapter.submitted == ["n1:screen:5"]
    assert [e["run_id"] for e in store.events("submit_refused")] == ["n1:screen:100003"]


def test_v3_routing_high_takes_primary_low_takes_preemptible(ctx, store):
    book = R.RunBook(_pools(ctx, primary=1, idle=2), store, _Pending())
    node = _node(store, "n1")
    assert book.pool(node, Stage.CONFIRM, 100_000) == "primary"     # a test run: HIGH
    assert book.pool(node, Stage.SCREEN, 1) == "idle"               # a selection seed: LOW
    book.launch(node, Stage.SCREEN, 1)
    book.launch(node, Stage.SCREEN, 2)
    book.high_demand = 1
    assert book.pool(node, Stage.SCREEN, 3) == ""                   # LOW never takes primary while HIGH waits
    book.high_demand = 0
    assert book.pool(node, Stage.SCREEN, 3) == "primary"


def test_promote_stuck_moves_high_to_a_free_primary_and_caps_the_full_primary_queue(ctx, store):
    ctx = _pools(ctx, primary=8, idle=16, repool_restarts=2)
    adapter = _Pending()
    book = R.RunBook(ctx, store, adapter)
    node = _node(store, "n1")
    for s in range(8):                                               # primary full, all running
        book.launch(node, Stage.BASELINE, 100_000 + s)
    book.poll()
    book.pending.clear()
    high = [book.launch(node, Stage.BASELINE, 100_100 + s) for s in range(8)]  # spill onto idle, HIGH (not TEST)
    assert {r.pool for r in high} == {"idle"}
    book.poll()                                                      # everything PENDING
    book.pending -= {r.id for r in store.runs() if r.pool == "primary"}
    for r in high:                                                   # preempted again and again
        book.restarts[r.id] = 2
        book.repool_due.add(r.id)
    lines = book.promote_stuck()
    assert len(lines) == book._overflow_cap("primary") == 2          # 7b94306: max(1, 8 // 4) onto a full primary
    assert all("preempted again and again" in x for x in lines)


def _slurm_key(ctx, **kw):
    """ctx with extra slurm keys (a namespace copy while loop.spec.SlurmCfg does not declare them yet)."""
    sl = ctx.cfg.slurm
    try:
        return dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, slurm=dataclasses.replace(sl, **kw)))
    except TypeError:
        ns = SimpleNamespace(**{f.name: getattr(sl, f.name) for f in dataclasses.fields(sl)}, **kw)
        return dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, slurm=ns))


def test_test_stage_runs_wait_for_primary_and_never_spill_onto_preemptible(ctx, store):
    book = R.RunBook(_pools(ctx, primary=1, idle=4), store, _Pending())
    node = _node(store, "n1")
    first = book.launch(node, Stage.CONFIRM, 100_000)
    assert first.pool == "primary"                                    # primary free: a test run takes it
    assert book.pool(node, Stage.CONFIRM, 100_001) == ""              # primary full: CONFIRM waits ...
    assert book.pool(node, Stage.REMINT, 100_001) == ""               # ... and so does REMINT
    assert book.launch(node, Stage.REMINT, 100_001) is None and store.get_run("n1:remint:100001") is None
    assert book.pool(node, Stage.BASELINE, 1) == "idle"               # a non-TEST HIGH run still spills
    assert book.pool(node, Stage.SCREEN, 1) == "idle"
    book.fill_only = {("n1", 100_001)}                                # an idle-fill key never moves a test run off
    assert book.pool(node, Stage.CONFIRM, 100_001) == ""
    book.fill_only = set()
    store.update_run(first.id, status=RunStatus.DONE)                 # a primary slot frees up
    assert book.launch(node, Stage.REMINT, 100_001).pool == "primary"
    # a retry no pool takes now queues on primary, never on its old preemptible pool
    stale = dataclasses.replace(first, id="n1:confirm:100009", pool="idle", stage=Stage.CONFIRM)
    assert book._retry_home(stale) == "primary"
    assert book._retry_home(dataclasses.replace(stale, stage=Stage.SCREEN)) == "idle"


def test_test_primary_only_off_restores_the_spill_and_no_primary_keeps_the_old_rule(ctx, store):
    book = R.RunBook(_slurm_key(_pools(ctx, primary=1, idle=4), test_primary_only=False), store, _Pending())
    node = _node(store, "n1")
    book.launch(node, Stage.CONFIRM, 100_000)
    assert book.pool(node, Stage.CONFIRM, 100_001) == "idle"          # the old rule: HIGH spills
    only_pre = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, slurm=SlurmCfg(pools=(
        PoolCfg("idle", slots=2, preemptible=True),))))
    book2 = R.RunBook(only_pre, store, _Pending())                    # no primary pool: never deadlock
    assert not book2.test_primary_only(Stage.CONFIRM)
    assert book2.pool(_node(store, "n2"), Stage.CONFIRM, 100_002) == "idle"


def test_research_sessions_harvest_persist_and_are_reissued_after_a_restart(ctx, store):
    started = []

    class Researcher:
        def research(self, req, store_, *args):
            started.append(req.id)
            return SimpleNamespace(status="ok", req=req.id)
    req = CandidateRequest(id="r1", task_id=ctx.task_id, strategy=Strategy.EXPLORE, operator=Operator.DRAFT,
                           island="main", parent_ids=(ROOT_NODE_ID,), brief="b")
    rs = R.Research(ctx, store, SimAdapter(ctx), Researcher(), InlineExecutor())
    rs.start(req)
    assert [e["id"] for e in store.get_state(K_INFLIGHT)] == ["r1"]
    acts = []
    assert [(q.id, r.req) for q, r in rs.harvest(acts)] == [("r1", "r1")] and store.get_state(K_INFLIGHT) == []
    store.set_state(K_INFLIGHT, [{"id": "r2", "req": encode(dataclasses.replace(req, id="r2")), "restarts": 0},
                                 {"id": "r3", "req": encode(dataclasses.replace(req, id="r3")),
                                  "restarts": R.MAX_RESTARTS}])
    fresh = R.Research(ctx, store, SimAdapter(ctx), Researcher(), InlineExecutor())   # a new process
    fresh.restore(acts)
    assert [r.id for r in fresh.requeue] == ["r2"] and "research r3 lost to a restart (not re-issued)" in acts
    assert fresh.start_requeued() and started == ["r1", "r2"]
    assert [e["request_id"] for e in store.events("research_failed")] == ["r2", "r3"]


def test_research_free_respects_the_llm_pause_and_the_probe(ctx, store):
    rs = R.Research(ctx, store, SimAdapter(ctx), object(), InlineExecutor())
    assert rs.free() == ctx.cfg.researcher.max_concurrent
    store.set_state(R.K_LLM_PAUSE, {"active": True, "kind": "usage", "until": 1e18})
    assert rs.free() == 0
    store.set_state(R.K_LLM_PAUSE, {"active": False, "probing": True, "ok_mark": 1e18})
    assert rs.free() == 1                                             # one probe session


def test_incumbent_labless_submits_each_incumbent_once(ctx, store):
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(
        ctx.cfg, labless=dataclasses.replace(ctx.cfg.labless, enabled=True)))
    calls = []
    lab = R.IncumbentLabless(ctx, store, InlineExecutor(), clock=lambda: 1000.0,
                             submit=lambda ctx_, out, name, text: calls.append((out, name)) or "sub-1")
    assert lab.step(ROOT_NODE_ID) == [] and lab.step("n1") == []      # root never; no done run yet
    node = _node(store, "n1")
    book = R.RunBook(ctx, store, _Pending())
    book.launch(node, Stage.CONFIRM, 100_000)
    store.update_run("n1:confirm:100000", status=RunStatus.DONE)
    assert lab.step("n1") == ["labless: submitting incumbent n1 from n1:confirm:100000"]
    assert lab.step("n1") == ["labless: incumbent n1 submitted, submission_id=sub-1"]
    assert lab.step("n1") == [] and len(calls) == 1
    assert store.get_state(R.K_LABLESS)["n1"]["submission_id"] == "sub-1"


def test_repair_outage_runs_once(ctx, store):
    acts = []
    R.repair_outage(store, 1.0, acts, on_outage=lambda ids: ["fixed"])
    R.repair_outage(store, 2.0, acts, on_outage=lambda ids: pytest.fail("ran twice"))
    assert acts == ["outage repair: fixed"] and store.get_state(R.K_OUTAGE_REPAIR)["ts"] == 1.0


def test_smoke_reserve_and_smokes_move_to_primary_while_our_runs_pend_preemptible(ctx, store):
    """10-13: research smokes starved behind our own PENDING preemptible runs. A preemptible pool holding our runs
    PENDING (> slurm.smoke_skip_pending) is skipped by smokes; the smoke reserve sits on primary only while a smoke is
    in flight (c4j review M3: an idle reserve there kept a primary slot empty)."""
    from collections import Counter
    from types import SimpleNamespace
    from loop.spec import PoolCfg, SlurmCfg
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, slurm=SlurmCfg(pools=(
        PoolCfg("primary", slots=4), PoolCfg("idle", slots=8, preemptible=True)), smoke_skip_pending=0)))
    smokes = Counter()
    book = R.RunBook(ctx, store, SimpleNamespace(smokes=smokes))
    assert book.used() == Counter(idle=1) and book.smoke_pool() == "idle"        # before a poll: preemptible
    book.pool_states = {"primary": (4, 0), "idle": (6, 2)}
    assert book.smoke_blocked("idle") and not book.smoke_blocked("primary")
    assert book.used() == Counter(idle=1) and book.smoke_pool() == "primary"      # no smoke waiting: reserve stays
    smokes["idle"] = 1                                                            # a smoke in flight: reserve on primary
    assert book.used() == Counter(primary=1, idle=1)
    smokes.clear()
    book.pool_states = {"primary": (4, 0), "idle": (6, 0)}
    assert book.used() == Counter(idle=1) and book.smoke_pool() == "idle"
    ctx2 = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, slurm=dataclasses.replace(
        ctx.cfg.slurm, smoke_skip_pending=3)))
    book2 = R.RunBook(ctx2, store, SimpleNamespace(smokes=Counter()))
    book2.pool_states = {"primary": (4, 0), "idle": (6, 2)}
    assert not book2.smoke_blocked("idle") and book2.smoke_pool() == "idle"


def test_submit_prepared_runs_adapter_submits_concurrently_and_keeps_outcomes_in_order(ctx, store):
    """10-13 tick timing: a tick's submits were serial (node worktree + sbatch each). Rows are prepared serially (slot
    accounting), the adapter submits run on slurm.submit_workers threads, the store writes stay on this thread."""
    import threading
    import time as _t
    from loop.contracts import SubmissionRejected

    class _Slow(_Pending):
        concurrent_submit = True

        def __init__(self):
            super().__init__()
            self.live = self.peak = 0
            self.lock = threading.Lock()

        def submit(self, *, run, commit):
            with self.lock:
                self.live += 1
                self.peak = max(self.peak, self.live)
            _t.sleep(0.05)
            with self.lock:
                self.live -= 1
            if run.node_id == "bad":
                raise RuntimeError("sbatch: socket timed out")
            if run.node_id == "rej":
                raise SubmissionRejected("invalid account")
            return super().submit(run=run, commit=commit)

    c = _pools(ctx, primary=8, idle=0, submit_workers=4)
    adapter = _Slow()
    book = R.RunBook(c, store, adapter)
    names = ["a", "b", "bad", "c", "rej", "d"]
    rows = [(book.prepare_launch(_node(store, n), Stage.SCREEN, 11), store.get_node(n)) for n in names]
    assert all(r.status is RunStatus.SUBMITTING for r, _ in rows) and book.free() == 2    # slots held up front
    got = book.submit_prepared(rows)
    assert adapter.peak > 1                                                           # concurrent
    assert [g.status if not isinstance(g, BaseException) else type(g).__name__ for g in got] == [
        RunStatus.RUNNING, RunStatus.RUNNING, RunStatus.VOID, RunStatus.RUNNING, "SubmissionRejected", RunStatus.RUNNING]
    assert store.get_run("rej:screen:11").status is RunStatus.VOID and [e["run_id"] for e in store.events(
        "submit_failed")] == ["bad:screen:11"]
    serial = R.RunBook(_pools(ctx, primary=8, idle=0, submit_workers=1), store, _Pending())
    one = serial.submit_prepared([(serial.prepare_launch(_node(store, "e"), Stage.SCREEN, 11), store.get_node("e"))])
    assert one[0].status is RunStatus.RUNNING
