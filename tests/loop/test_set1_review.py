"""Adversarial review of the set-1 harness commits (2026-10-21): TEST runs under an association cap, the stale-pending
release (atomic cancel, same-attempt retry, receipt), the capacity probe's backoff, the generic-stem stoplist of the
negative-family block, shadow filler words and the aging guards."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from loop import runner as R
from loop.contracts import JobHandle, JobStatus, Run, RunStatus, Stage, VoidType

from .test_adapters import SlurmRunner, base_repo, make_np, np_ctx  # noqa: F401  (fixtures)
from .test_runner import _node, _pools
from .test_set1_harness import _Cap, _Held, _acct_ctx, _stale_book

CUT = {"primary": {"limit": 8, "others": 8, "loop": 0, "effective": 0}}       # the sweeps hold every GPU of the cap


# ---------------------------------------------------------------- 1: TEST runs spill promptly when primary is cut
def test_a_test_run_spills_to_idle_at_once_when_primary_is_cut_by_other_jobs(ctx, store):
    node = _node(store, "n1")
    book = R.RunBook(_pools(ctx, primary=6, idle=4), store, _Cap(CUT))
    book.poll()
    assert book.pool(node, Stage.CONFIRM, 100_100) == "idle"          # not after test_primary_wait_s (3 h)
    run = book.launch(node, Stage.CONFIRM, 100_100)
    assert run.pool == "idle" and run.id in book.spilled()             # remembered: resumed after a preemption
    assert store.events("test_spilled")[-1]["run_id"] == run.id


def test_a_test_run_still_waits_for_primary_without_a_cut_or_with_the_switch_off(ctx, store):
    node = _node(store, "n1")
    roomy = R.RunBook(_pools(ctx, primary=1, idle=4), store, _Cap({}))      # no association cap known
    roomy.launch(_node(store, "n0"), Stage.CONFIRM, 100_100)               # takes the only primary slot
    assert roomy.pool(node, Stage.CONFIRM, 100_101) == ""                   # waits (3 h), as before
    off = R.RunBook(_pools(ctx, primary=6, idle=4, test_spill_capped=False), store, _Cap(CUT))
    off.poll()
    assert off.pool(node, Stage.CONFIRM, 100_100) == ""


def test_a_cut_primary_with_a_free_effective_slot_still_takes_the_test_first(ctx, store):
    some = {"primary": {"limit": 8, "others": 6, "loop": 0, "effective": 2}}
    book = R.RunBook(_pools(ctx, primary=6, idle=4), store, _Cap(some))
    book.poll()
    assert book.pool(_node(store, "n1"), Stage.CONFIRM, 100_100) == "primary"


def test_a_stale_pending_test_run_on_a_cut_primary_moves_to_idle_and_is_never_cancelled(ctx, store):
    adapter = _Held()
    adapter.caps = CUT
    book, adapter, t = _stale_book(ctx, store, adapter=adapter, primary=2, idle=2, stale_pending_min=90)
    node = _node(store, "n1")
    run = store.put_run(Run(id=Run.make_id("n1", Stage.CONFIRM, 100_100), task_id=store.task_id, node_id="n1", seed=100_100,
                            stage=Stage.CONFIRM, status=RunStatus.RUNNING, job_id="9", output_dir="/o/x", pool="primary",
                            submitted_ts=1000.0)) or store.get_run(Run.make_id("n1", Stage.CONFIRM, 100_100))
    book.poll()
    t[0] += 60 * 100
    out = book.stale_pending()
    assert len(out) == 1 and adapter.moved == [(run.id, "idle")]             # moved (its pending job cancelled first, by id)
    assert adapter.cancelled == ["9"] and store.get_run(run.id).status is RunStatus.RUNNING     # ... never voided
    assert run.id in book.spilled()                                          # resumed after its first preemption
    moved = store.get_run(run.id)
    assert moved.pool == "idle" and moved.status is RunStatus.RUNNING and moved.attempt == run.attempt
    # no preemptible slot either: it keeps its place, nothing cancelled
    full, adapter2, t2 = _stale_book(ctx, store, adapter=_Held(), primary=2, idle=0, stale_pending_min=90)
    adapter2.caps = CUT
    full.poll()
    t2[0] += 60 * 100
    store.update_run(run.id, pool="primary", status=RunStatus.RUNNING)
    full.poll()
    t2[0] += 60 * 100
    assert full.stale_pending() == [] and not adapter2.cancelled and store.get_run(run.id).status is RunStatus.RUNNING


# ---------------------------------------------------------------- 3: stale pending -- atomic, same attempt
class _Atomic(_Held):
    """cancel(only_pending) that finds the job already RUNNING (it started in the window since the poll)."""

    def __init__(self):
        super().__init__()
        self.started, self.released, self.only = False, [], []

    def cancel(self, handle, only_pending=False):
        self.only.append(only_pending)
        if self.started:
            raise RuntimeError("scancel 1 not confirmed (state=['RUNNING'])")
        self.cancelled.append(handle.job_id)

    def release_receipt(self, handle):
        self.released.append(handle.job_id)


def test_a_job_that_started_in_the_window_is_not_cancelled(ctx, store):
    book, adapter, t = _stale_book(ctx, store, adapter=_Atomic(), primary=1, idle=1, stale_pending_min=90)
    node = _node(store, "n1")
    a, b = book.launch(node, Stage.BASELINE, 100_000), book.launch(node, Stage.BASELINE, 100_001)
    book.poll()
    t[0] += 60 * 100
    adapter.started = True
    assert book.stale_pending() == []                                       # SLURM would not cancel a RUNNING job
    assert store.get_run(a.id).status is RunStatus.RUNNING and not adapter.cancelled and not adapter.released
    assert adapter.only and all(adapter.only)                               # always scancel --state=PENDING
    assert len(store.events("cancel_failed")) >= 1


def test_the_cancelled_row_is_retried_as_the_same_attempt_with_its_receipt_released(ctx, store):
    book, adapter, t = _stale_book(ctx, store, adapter=_Atomic(), primary=1, idle=1, stale_pending_min=90)
    node = _node(store, "n1")
    a, b = book.launch(node, Stage.BASELINE, 100_000), book.launch(node, Stage.BASELINE, 100_001)
    assert (a.attempt, a.pool) == (0, "primary")
    book.poll()
    t[0] += 60 * 100
    book.stale_pending()
    r = store.get_run(a.id)
    assert r.status is RunStatus.VOID and r.void_type is VoidType.INFRA and r.attempt == -1    # unspent
    assert adapter.released == ["1"] and adapter.cancelled[0] == "1"
    again = book.retry(r, node)                                              # what Tick._retries does
    assert again.attempt == 0                                                # the SAME attempt: no retry burnt


def test_the_real_adapter_cancels_only_pending_and_sets_the_receipt_aside(np_ctx, base_repo):   # noqa: F811
    adapter, runner = make_np(np_ctx, base_repo, runner=SlurmRunner({"squeue": (0, "", "")}))
    rel = "n1/screen_s11_a0"
    rec = {"run_id": "n1:screen:11", "attempt": 0, "job_id": "55", "output_dir": str(adapter.runs / rel),
           "submitted_ts": 1.0}
    adapter._save(rel, rec)
    h = JobHandle("n1:screen:11", "55", str(adapter.runs / rel), 1.0)
    adapter.cancel(h, only_pending=True)
    assert ["scancel", "--state=PENDING", "55"] in runner.argvs("scancel")
    run = Run(id="n1:screen:11", task_id="t", node_id="n1", seed=11, stage=Stage.SCREEN, status=RunStatus.RUNNING)
    assert adapter.find(run) is not None                                    # submit would return the dead job
    adapter.release_receipt(h)
    assert adapter.find(run) is None                                        # ... now it launches a new one
    kept = list((Path(adapter.state) / "receipts/n1").glob("screen_s11_a0.released-*.json"))
    assert len(kept) == 1 and json.loads(kept[0].read_text())["job_id"] == "55"        # nothing deleted


def test_a_started_job_makes_the_real_cancel_raise(np_ctx, base_repo):   # noqa: F811
    adapter, runner = make_np(np_ctx, base_repo, runner=SlurmRunner({"squeue": (0, "RUNNING\n", "")}))
    rel = "n1/screen_s11_a0"
    adapter._save(rel, {"job_id": "55", "output_dir": str(adapter.runs / rel), "submitted_ts": 1.0})
    import pytest
    with pytest.raises(RuntimeError, match="not confirmed"):
        adapter.cancel(JobHandle("r", "55", str(adapter.runs / rel), 1.0), only_pending=True)


# ---------------------------------------------------------------- 2: the capacity probe never stalls the loop
def test_a_failed_probe_is_not_repeated_every_tick_and_has_a_short_timeout(np_ctx, base_repo):   # noqa: F811
    now = [1000.0]
    adapter, runner = make_np(_acct_ctx(np_ctx), base_repo, runner=SlurmRunner({"sacctmgr": (124, "", "timeout")}),
                              clock=lambda: now[0])
    assert adapter.pool_capacity() == {}
    n = len(runner.argvs("sacctmgr"))
    import loop.adapters.nanopath_v2 as NP
    assert n == 1 and NP.CAP_PROBE_TIMEOUT_S <= 30
    now[0] += 60
    assert adapter.pool_capacity() == {} and len(runner.argvs("sacctmgr")) == n      # backed off 5 min
    now[0] += 300
    adapter.pool_capacity()
    assert len(runner.argvs("sacctmgr")) == n + 1


def test_an_empty_or_account_level_limit_means_no_cap_and_never_zero_slots(np_ctx, base_repo):   # noqa: F811
    for rows in ("main|\n", "", "main|cpu=100\n", "root|gres/gpu=8\n"):      # empty / other limit / other account
        adapter, _ = make_np(_acct_ctx(np_ctx), base_repo, runner=SlurmRunner({
            "sacctmgr": (0, rows, ""), "squeue": (0, "", "")}))
        assert adapter.pool_capacity() == {}


def test_the_idle_pool_always_keeps_its_declared_slots_under_a_zero_primary(ctx, store):
    book = R.RunBook(_pools(ctx, primary=6, idle=4), store, _Cap(CUT))
    book.poll()
    lim = book.limits(gate=False)
    assert lim == {"primary": 0, "idle": 4}                                  # no deadlock: idle still takes work
    assert book.pool(_node(store, "n1"), Stage.BASELINE, 100_000) == "idle"


# ---------------------------------------------------------------- 5: generic stems never make a negative family
def _tn(nid, tag):
    return SimpleNamespace(id=nid, family="f", mechanism_tag=tag)


def test_generic_stems_make_no_term_family_but_specific_ones_do(ctx):
    from loop.core import alloc as AL
    tags = {"a": "coding_rate_a", "b": "coding_rate_b", "c": "learned_rate_c", "d": "rate_match_d", "e": "rate_e",
            "p1": "position_head_a", "p2": "position_pred_b", "p3": "rel_position_c", "p4": "position_scale_d",
            "p5": "abs_position_e"}
    nodes = [_tn(k, v) for k, v in tags.items()] + [_tn(f"z{i}", f"unrelated_thing_{i}") for i in range(80)]
    beliefs = {k: (-0.01, 2) for k in tags}
    fams = AL.term_families(ctx, nodes, beliefs)
    assert not any(w.startswith("rate") for w in fams)                       # 'rate' = coding / learning / sampling rate
    pos = next(f for w, f in fams.items() if w.startswith("positi"))
    assert pos.neg_family and pos.n == 5                                     # a specific stem still blocks
    assert AL.generic_stem(ctx, "project") and AL.generic_stem(ctx, "backbon") and not AL.generic_stem(ctx, pos.stem)


# ---------------------------------------------------------------- 6: program shadows ignore boilerplate words
def test_arm_stems_drop_boilerplate_so_two_filler_words_cannot_shadow(ctx):
    from loop.core import program as PG
    arm = {"id": "A1", "note": "default dose knob; only the grad never changes; same step count", "components": [],
           "keys": ["dino.init_align_weight"]}
    stems = PG.arm_stems(arm)
    assert not stems & {"dose", "knob", "default", "grad", "never", "same", "step", "count", "dino"}
    assert "init" in stems or any(s.startswith("init") for s in stems)


# ---------------------------------------------------------------- 4: aging never touches a node with a run or an open block
def test_age_out_skips_nodes_with_a_run_row_or_an_open_block(ctx, store):
    from loop.core import acquire as A
    for nid in ("n1", "n2", "n3"):
        _node(store, nid)
        store.update_node(nid, created_ts=1000.0)
    store.put_run(Run(id=Run.make_id("n2", Stage.SCREEN, 11), task_id=store.task_id, node_id="n2", seed=11,
                      stage=Stage.SCREEN, status=RunStatus.RUNNING, pool="primary", submitted_ts=1.0))
    from loop.core.seeds import SeedBlock  # noqa: F401
    blk = SimpleNamespace(open=True, node_ids=("n3",))
    store.seed_blocks = lambda: [blk]                                        # n3 sits in an open block
    done = A.age_out(store, ["n1", "n2", "n3"], now=1000.0 + 200 * 3600)
    assert done == ["n1"]
