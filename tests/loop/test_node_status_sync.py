"""10-22: v3 derives every status each tick and stored none (7 days: one ``node_transition``). The tick now writes the
derived status back to ``nodes.status`` over live stored statuses only, never over an explicit terminal act and never
PASSED / INCUMBENT (the stored CONFIRMED would grant Labless / prune / databank effects), with one summary event on the
first mass pass."""

from __future__ import annotations

import time

from loop import tick as TK
from loop.contracts import NodeStatus
from loop.core import data as CD

from .test_runner import _node
from .test_set1_harness import _Held


def _tick(ctx, store, tmp_path):
    return TK.Tick(ctx, store, _Held(), TK.ModuleCore(), db_path=tmp_path / "loop.db", clock=time.time)


def _st(tick, **by):
    S = tick.core.ST.Status
    return {nid: S[name.upper()] for nid, name in by.items()}


def test_the_first_mass_pass_writes_one_summary_event_and_keeps_updated_ts(ctx, store, tmp_path):
    tick = _tick(ctx, store, tmp_path)
    ids = [f"n{i}" for i in range(30)]
    for nid in ids:
        _node(store, nid)
        store.update_node(nid, updated_ts=1234.0)
    acts: list[str] = []
    assert tick._sync_node_status(_st(tick, **{nid: "screening" for nid in ids}), time.time(), acts) == 30
    assert all(store.get_node(nid).status is NodeStatus.SCREENING for nid in ids)
    assert all(store.get_node(nid).updated_ts == 1234.0 for nid in ids)       # queue age / prune grace read it
    assert len(store.events("node_status_sync")) == 1 and not store.events("node_transition")
    assert len(acts) == 1 and "queued->screening 30" in acts[0]
    assert tick._sync_node_status(_st(tick, **{nid: "screening" for nid in ids}), time.time(), []) == 0   # idempotent
    tick.close()


def test_a_few_changes_log_one_node_transition_each(ctx, store, tmp_path):
    tick = _tick(ctx, store, tmp_path)
    _node(store, "n1")
    _node(store, "n2")
    tick._sync_node_status(_st(tick, n1="testing", n2="failed"), time.time(), [])
    assert store.get_node("n1").status is NodeStatus.CONFIRMING and store.get_node("n2").status is NodeStatus.NOT_CONFIRMED
    ev = {e["node_id"]: (e["frm"], e["to"]) for e in store.events("node_transition")}
    assert ev == {"n1": ("queued", "confirming"), "n2": ("queued", "not_confirmed")}
    tick.close()


def test_explicit_terminal_statuses_and_passed_incumbent_are_never_overwritten(ctx, store, tmp_path):
    tick = _tick(ctx, store, tmp_path)
    for nid, st in (("conf", NodeStatus.CONFIRMED), ("base", NodeStatus.BASELINE), ("veto", NodeStatus.VETOED),
                    ("rej", NodeStatus.REJECTED), ("out", NodeStatus.SCREENED_OUT)):
        _node(store, nid)
        store.update_node(nid, status=st)
    _node(store, "passed")
    _node(store, "inc")
    sts = _st(tick, conf="failed", base="screening", veto="screening", rej="void", out="screening",
              passed="passed", inc="incumbent")
    assert tick._sync_node_status(sts, time.time(), []) == 0
    assert {n.id: n.status for n in store.nodes()} == {
        "conf": NodeStatus.CONFIRMED, "base": NodeStatus.BASELINE, "veto": NodeStatus.VETOED,
        "rej": NodeStatus.REJECTED, "out": NodeStatus.SCREENED_OUT, "passed": NodeStatus.QUEUED,
        "inc": NodeStatus.QUEUED}
    tick.close()


def test_a_node_the_sync_closed_follows_the_derivation_back_open(ctx, store, tmp_path):
    tick = _tick(ctx, store, tmp_path)
    _node(store, "n1")
    tick._sync_node_status(_st(tick, n1="void"), time.time(), [])
    assert store.get_node("n1").status is NodeStatus.VOID
    tick._sync_node_status(_st(tick, n1="screening"), time.time(), [])          # a retried run: not stuck 'void'
    assert store.get_node("n1").status is NodeStatus.SCREENING
    _node(store, "n2")
    store.update_node("n2", status=NodeStatus.VOID)                              # a void another path set: not ours
    tick._sync_node_status(_st(tick, n2="screening"), time.time(), [])
    assert store.get_node("n2").status is NodeStatus.VOID
    tick.close()


def test_a_sync_never_pins_or_unpins_a_confirmed_checkpoint_in_prune(ctx, store, tmp_path):
    from loop import prune as P
    from types import SimpleNamespace
    tick = _tick(ctx, store, tmp_path)
    _node(store, "c")
    store.update_node("c", status=NodeStatus.CONFIRMED)
    _node(store, "q")
    snap = SimpleNamespace(incumbent_id="", banked=(), closed={"q"}, keep=set(), screening=set(), queued=set(), open=set())
    before = P.classify(ctx, store, snap, time.time() + 1e7)
    tick._sync_node_status(_st(tick, c="failed", q="failed"), time.time(), [])
    after = P.classify(ctx, store, snap, time.time() + 1e7)
    assert "c" in before[1] and "c" in after[1] and before[1] == after[1]       # models (hard) unchanged
    assert "c" in after[0]
    tick.close()


def test_the_tick_step_wires_the_sync_after_derive(ctx, store, tmp_path):
    import inspect
    src = inspect.getsource(TK.Tick.step)
    assert src.index("self.statuses = data, statuses") < src.index("_sync_node_status")


def test_a_terminal_derived_status_stamps_updated_ts_so_prune_grace_starts_at_close(ctx, store, tmp_path):
    tick = _tick(ctx, store, tmp_path)
    _node(store, "n1")
    _node(store, "n2")
    store.update_node("n1", updated_ts=1234.0)
    store.update_node("n2", updated_ts=1234.0)
    t0 = time.time()
    tick._sync_node_status(_st(tick, n1="failed", n2="screening"), t0, [])
    assert store.get_node("n1").updated_ts == t0 and store.get_node("n2").updated_ts == 1234.0   # live keeps its ts
    tick.close()
