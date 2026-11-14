"""c3j pre-deploy review fixes: stale_control (port-aware, epoch switch keeps running runs), the TEST-stage primary
wait -> preemptible spill with resume, and a scancel that is confirmed before the row goes VOID."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

from loop import runner as R
from loop import tick as TK
from loop.adapters.nanopath_v2 import NanopathV2Adapter
from loop.contracts import JobHandle, RunStatus, Stage
from loop.core import data as CD

from .test_runner import _node, _Pending, _pools, _slurm_key


# ---------------------------------------------------------------- 4: stale_control
def _stale_self(runs, blocks, cancelled):
    logs = []
    store = SimpleNamespace(runs=lambda node_id=None, status=None: [r for r in runs if r.node_id == node_id
                                                                    and r.status in status],
                            log=lambda ev, **kw: logs.append((ev, kw)))
    book = SimpleNamespace(pending=set(), retry_cancels=lambda: [],
                           cancel_run=lambda run, why: cancelled.append(run.id) or True)
    me = SimpleNamespace(ctx=SimpleNamespace(cfg=SimpleNamespace(test=SimpleNamespace(stale_cancel_running=True))),
                         store=store, book=book, blocks=lambda: blocks, closed=[], logs=logs)
    me._update_block = lambda bid, acts, why, **kw: me.closed.append((bid, why, kw))
    return me


def _led(epoch, nodes, port):
    return SimpleNamespace(epoch=epoch, nodes={k: SimpleNamespace(base=v) for k, v in nodes.items()},
                           port=port, runs=[])


def _blk():
    return SimpleNamespace(id="test-1", pool=CD.TEST, open=True, node_ids=("cand", "ctrl_old"), seeds=(7,))


def _run(node, status):
    return SimpleNamespace(id=f"{node}:confirm:7", node_id=node, seed=7, stage=Stage.CONFIRM, status=status)


def test_same_incumbent_ported_to_a_new_base_epoch_is_not_stale():
    cancelled = []
    runs = [_run("ctrl_old", RunStatus.RUNNING)]
    me = _stale_self(runs, [_blk()], cancelled)
    led = _led("B2", {"ctrl_old": "B1", "ctrl_new": "B2"}, lambda n, b=None: "ctrl_new" if n in ("ctrl_old", "ctrl_new") else n)
    assert TK.Tick._stale_controls(me, led, "ctrl_new", 1.0, []) is False
    assert not me.closed and not cancelled


def test_epoch_switch_with_a_different_incumbent_closes_but_keeps_running_and_submitting_runs():
    cancelled = []
    runs = [_run("ctrl_old", RunStatus.RUNNING), _run("cand", RunStatus.SUBMITTING)]
    me = _stale_self(runs, [_blk()], cancelled)
    led = _led("B2", {"ctrl_old": "B1", "other": "B2"}, lambda n, b=None: n if n == "other" else None)
    assert TK.Tick._stale_controls(me, led, "other", 1.0, []) is True
    assert me.closed and "stale_control" in me.closed[0][1]
    assert cancelled == []                                           # nothing cancelled at an epoch switch
    assert {kw["status"] for ev, kw in me.logs if ev == "stale_run_kept"} == {"running", "submitting"}


def test_same_epoch_incumbent_switch_still_cancels_running_runs_but_never_submitting():
    cancelled = []
    runs = [_run("ctrl_old", RunStatus.RUNNING), _run("cand", RunStatus.SUBMITTING)]
    me = _stale_self(runs, [_blk()], cancelled)
    led = _led("B1", {"ctrl_old": "B1", "other": "B1"}, lambda n, b=None: n)
    assert TK.Tick._stale_controls(me, led, "other", 1.0, []) is True
    assert cancelled == ["ctrl_old:confirm:7"]


# ---------------------------------------------------------------- 5: primary wait -> spill with resume
def test_test_run_spills_to_preemptible_after_the_primary_wait_and_only_it_resumes(ctx, store):
    t = [1000.0]
    c = _slurm_key(_pools(ctx, primary=1, idle=4), test_primary_wait_s=100.0)
    adapter = _Pending()
    adapter.resume_runs = set()
    book = R.RunBook(c, store, adapter, clock=lambda: t[0])
    node = _node(store, "n1")
    first = book.launch(node, Stage.CONFIRM, 100_000)
    assert first.pool == "primary" and book.spilled() == set()
    assert book.pool(node, Stage.CONFIRM, 100_001) == ""             # waiting: wait starts now
    t[0] += 99
    assert book.pool(node, Stage.CONFIRM, 100_001) == ""
    t[0] += 2
    assert book.pool(node, Stage.CONFIRM, 100_001) == "idle"        # waited >= 100 s: may spill
    spilled = book.launch(node, Stage.CONFIRM, 100_001)
    assert spilled.pool == "idle" and book.spilled() == {spilled.id}
    assert spilled.id in adapter.resume_runs and first.id not in adapter.resume_runs
    assert book.store.get_state(R.K_TEST_WAIT) == {}                 # launch cleared the wait
    # the wait survives a restart (a new RunBook on the same store)
    book2 = R.RunBook(c, store, _Pending(), clock=lambda: t[0])
    assert book2.pool(node, Stage.REMINT, 5) == ""
    t[0] += 101
    assert book2.pool(node, Stage.REMINT, 5) == "idle"


def test_primary_wait_zero_never_spills(ctx, store):
    t = [0.0]
    book = R.RunBook(_slurm_key(_pools(ctx, primary=1, idle=4), test_primary_wait_s=0.0), store, _Pending(),
                     clock=lambda: t[0])
    node = _node(store, "n1")
    book.launch(node, Stage.CONFIRM, 1)
    t[0] += 10 ** 7
    assert book.pool(node, Stage.CONFIRM, 2) == ""


def test_tick_no_resume_exempts_spilled_test_runs(ctx, store):
    book = R.RunBook(_pools(ctx), store, _Pending())
    store.set_state(R.K_TEST_SPILLED, ["n1:confirm:5"])
    no_resume = lambda run: run.stage in TK.TEST_STAGES and run.id not in book.spilled()  # noqa: E731  (tick.py's rule)
    assert not no_resume(SimpleNamespace(stage=Stage.CONFIRM, id="n1:confirm:5"))
    assert no_resume(SimpleNamespace(stage=Stage.CONFIRM, id="n1:confirm:6"))


# ---------------------------------------------------------------- 8: confirmed cancel
class _Slurm:
    def __init__(self, scancel_rc, squeue):
        self.scancel_rc, self.squeue, self.calls = scancel_rc, squeue, []

    def run(self, argv, **kw):
        self.calls.append(argv[0])
        if argv[0] == "scancel":
            return self.scancel_rc, "", "denied" if self.scancel_rc else ""
        return self.squeue


def _adapter(runner):
    a = object.__new__(NanopathV2Adapter)
    a.runner = runner
    a._rel_of = lambda h: "rel"
    a._receipt = lambda rel: {}
    a._job_id = lambda rel, rec: "42"
    return a


H = JobHandle("r", "42", "/o", 0.0)


def test_adapter_cancel_confirms_the_job_is_gone_or_ending():
    _adapter(_Slurm(0, (0, "", ""))).cancel(H)                                  # gone
    _adapter(_Slurm(1, (1, "", "slurm_load_jobs error: Invalid job id specified"))).cancel(H)   # finished already
    _adapter(_Slurm(0, (0, "COMPLETING\n", ""))).cancel(H)                      # ending


def test_adapter_cancel_raises_when_the_job_is_still_live():
    import pytest
    with pytest.raises(RuntimeError):
        _adapter(_Slurm(1, (0, "RUNNING\n", ""))).cancel(H)
    with pytest.raises(RuntimeError):
        _adapter(_Slurm(0, (0, "RUNNING\n", ""))).cancel(H)


def test_failed_cancel_keeps_the_row_running_and_is_retried_next_tick(ctx, store):
    class Flaky(_Pending):
        ok = False

        def cancel(self, handle):
            if not self.ok:
                raise RuntimeError("scancel not confirmed")

    ad = Flaky()
    book = R.RunBook(_pools(ctx), store, ad)
    node = _node(store, "n1")
    run = book.launch(node, Stage.SCREEN, 1)
    assert book.cancel_run(run, "why") is False
    assert store.get_run(run.id).status is RunStatus.RUNNING
    assert book.retry_cancels() == []                                           # still failing: still RUNNING
    ad.ok = True
    assert book.retry_cancels() == [f"cancel retried: {run.id}"]
    got = store.get_run(run.id)
    assert got.status is RunStatus.VOID and got.void_type.name == "CANCELLED"
    assert (store.get_state(R.K_CANCEL_RETRY) or {}) == {}


def test_a_closed_test_blocks_waits_are_pruned(ctx, store):
    """A TEST-stage wait (K_TEST_WAIT) whose block closed before the run launched is dropped (else a later run of the
    same key would spill at once)."""
    t = [1000.0]
    c = _slurm_key(_pools(ctx, primary=1, idle=4), test_primary_wait_s=100.0)
    book = R.RunBook(c, store, _Pending(), clock=lambda: t[0])
    node = _node(store, "n1")
    book.launch(node, Stage.CONFIRM, 100_000)
    assert book.pool(node, Stage.CONFIRM, 100_001) == "" and book.pool(node, Stage.CONFIRM, 100_002) == ""
    assert len(store.get_state(R.K_TEST_WAIT)) == 2
    assert book.prune_test_wait({("n1", 100_001)}) == 1
    assert list(store.get_state(R.K_TEST_WAIT)) == ["n1:confirm:100001"]
    assert book.prune_test_wait(set()) == 1 and store.get_state(R.K_TEST_WAIT) == {}
