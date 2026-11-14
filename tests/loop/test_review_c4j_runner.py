"""c4j pre-deploy review, runner: M3 (smoke routing: skip_pending default 8, <= smoke_primary_max smokes on primary,
the reserve on primary only while a smoke waits, diagnose jobs hold a primary slot) and S2 (chunked submits stop on
the tick budget and after a hung sbatch; the rows left ``submitting`` are reconciled next tick)."""

from __future__ import annotations

import dataclasses
import time as _t
from collections import Counter
from types import SimpleNamespace

from loop import runner as R
from loop.contracts import JobHandle, RunStatus, Stage
from loop.spec import PoolCfg, SlurmCfg

from .test_runner import _node, _Pending


def _ctx(ctx, primary=4, idle=8, **slurm):
    return dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, slurm=SlurmCfg(pools=(
        PoolCfg("primary", slots=primary), PoolCfg("idle", slots=idle, preemptible=True)), **slurm)))


def test_smoke_skip_pending_8_and_a_few_pending_runs_do_not_push_smokes_to_primary(ctx, store):
    """c4j M3's skip_pending 8 (v3-c5 R3 restores the default 0: tests/loop/test_c5_fill_slots.py)."""
    assert SlurmCfg().smoke_skip_pending == 0 and SlurmCfg().smoke_primary_max == 2
    book = R.RunBook(_ctx(ctx, smoke_skip_pending=8), store, SimpleNamespace(smokes=Counter()))
    book.pool_states = {"primary": (4, 0), "idle": (6, 3)}
    assert not book.smoke_blocked("idle") and book.smoke_pool() == "idle"
    book.pool_states = {"primary": (4, 0), "idle": (6, 9)}
    assert book.smoke_blocked("idle")


def test_at_most_smoke_primary_max_smokes_on_primary(ctx, store):
    smokes = Counter()
    book = R.RunBook(_ctx(ctx, primary=8), store, SimpleNamespace(smokes=smokes))
    book.pool_states = {"primary": (0, 0), "idle": (6, 20)}                       # idle blocked -> primary
    assert book.smoke_pool() == "primary"
    smokes["primary"] = 2
    assert book.smoke_pool() == "idle"                                            # cap reached: preemptible
    book2 = R.RunBook(_ctx(ctx, primary=8, smoke_primary_max=0), store, SimpleNamespace(smokes=smokes))
    book2.pool_states = {"primary": (0, 0), "idle": (6, 20)}
    assert book2.smoke_pool() == "primary"                                        # 0 = no cap


def test_live_diagnose_jobs_hold_a_primary_slot(ctx, store, tmp_path):
    from loop.agents.diagnose import K_JOBS
    now = _t.time()
    book = R.RunBook(_ctx(ctx), store, SimpleNamespace(smokes=Counter()), clock=lambda: now)
    done = tmp_path / "f.json"
    done.write_text("{}")
    store.set_state(K_JOBS, {"a": {"job": "1", "ts": now - 60, "out": str(tmp_path / "x.json")},   # alive
                             "b": {"job": "2", "ts": now - 60, "out": str(done)},                   # reported
                             "c": {"job": "3", "ts": now - 86400, "out": str(tmp_path / "y.json")}})  # stale
    assert book.diag_jobs() == 1 and book.used()["primary"] == 1


class _Hang(_Pending):
    concurrent_submit = True

    def __init__(self, hang=()):
        super().__init__()
        self.hang = set(hang)

    def submit(self, *, run, commit):
        if run.node_id in self.hang:
            raise RuntimeError("sbatch launch failed (rc=124): timeout after 300s")
        return super().submit(run=run, commit=commit)


def test_submit_prepared_aborts_after_a_hung_sbatch_and_the_rows_are_reconciled_next_tick(ctx, store):
    adapter = _Hang(hang={"b"})
    book = R.RunBook(_ctx(ctx, primary=8, idle=0, submit_workers=2), store, adapter)
    names = ["a", "b", "c", "d", "e"]
    rows = [(book.prepare_launch(_node(store, n), Stage.SCREEN, 11), store.get_node(n)) for n in names]
    got = book.submit_prepared(rows)
    kinds = [g.status if not isinstance(g, BaseException) else type(g).__name__ for g in got]
    assert kinds == [RunStatus.RUNNING, "SubmitDeferred", "SubmitDeferred", "SubmitDeferred", "SubmitDeferred"]
    assert adapter.submitted == ["a:screen:11"]                                       # nothing after the hung chunk
    assert all(store.get_run(f"{n}:screen:11").status is RunStatus.SUBMITTING for n in "bcde")   # slots held
    adapter.hang.clear()                                                      # next tick: slurmctld answers
    found = {"b"}                                                             # the hung sbatch had gone through

    def find(run):
        return JobHandle(run.id, "77", f"/o/{run.id}", 0.0) if run.node_id in found else None
    adapter.find = find
    acts = book.resolve_submitting()
    assert any("found job 77" in a for a in acts)
    assert all(store.get_run(f"{n}:screen:11").status is RunStatus.RUNNING for n in "bcde")
    assert sorted(adapter.submitted) == [f"{n}:screen:11" for n in "acde"]                  # b adopted, never submitted twice


def test_submit_prepared_stops_between_chunks_when_the_tick_budget_is_spent(ctx, store):
    adapter = _Hang()
    book = R.RunBook(_ctx(ctx, primary=8, idle=0, submit_workers=2), store, adapter)
    rows = [(book.prepare_launch(_node(store, n), Stage.SCREEN, 11), store.get_node(n)) for n in "abcde"]
    calls = []
    got = book.submit_prepared(rows, stop=lambda: calls.append(1) or len(calls) > 1)
    assert [isinstance(g, R.SubmitDeferred) for g in got] == [False, False, True, True, True]
    assert store.events("submit_deferred")[-1]["run_ids"] == ["c:screen:11", "d:screen:11", "e:screen:11"]
    serial = R.RunBook(_ctx(ctx, primary=8, idle=0, submit_workers=1), store, _Hang())
    rows = [(serial.prepare_launch(store.get_node(n), Stage.SCREEN, 12), store.get_node(n)) for n in "ab"]
    got = serial.submit_prepared(rows, stop=lambda: True)
    assert all(isinstance(g, R.SubmitDeferred) for g in got)


def test_resolve_submitting_stops_at_a_hung_sbatch(ctx, store):
    adapter = _Hang(hang={"a"})
    adapter.find = lambda run: None
    book = R.RunBook(_ctx(ctx, primary=8, idle=0), store, adapter)
    for n in "ab":
        book.prepare_launch(_node(store, n), Stage.SCREEN, 11)
    acts = book.resolve_submitting()
    assert any("hung" in a for a in acts) and adapter.submitted == []
    assert all(store.get_run(f"{n}:screen:11").status is RunStatus.SUBMITTING for n in "ab")
