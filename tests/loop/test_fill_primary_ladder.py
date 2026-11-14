"""10-22 throughput fixes. (1) 12:13-13:23Z two effective primary slots sat empty for 70 min with 14 screenable QUEUED nodes
waiting: a screen is 3 runs and ``rank`` placed it only whole (2 slots free: nothing), the relax pass then gave the slots to
preemptible-only seeds, and the idle pool was at its pending gate. Now a screen takes the slots left, and a fill / relax run
takes a free primary slot when no preemptible pool has room (never the owed-TEST reserve, never while a HIGH run waits).
(2) 8 program dose nodes waiting their turn held the brief throttle's median queue age ~20 h: program arms and ladder doses
are screened first, left out of that median, and never dropped as starved."""

from __future__ import annotations

import dataclasses
import time

from loop import runner as R
from loop import tick as TK
from loop.contracts import Stage
from loop.core import acquire as A
from loop.core import alloc as AL
from loop.core import data as CD
from loop.core.acquire import Option
from loop.spec import AllocCfg

from .test_runner import _node, _pools
from .test_set1_harness import _Held

S2 = 0.003 ** 2
H = 3600.0


def _real_tick(ctx, store, tmp_path):
    return TK.Tick(ctx, store, _Held(), TK.ModuleCore(), db_path=tmp_path / "loop.db", clock=time.time)


# ---------------------------------------------------------------- (1a) rank: a screen takes the slots left
def _screen(node="s", value_m=0.004, runs=3):
    return Option("screen", node, "explore", value_m, 0.002 ** 2, S2 / 3, float(runs), runs)


def test_a_screen_that_does_not_fit_whole_takes_the_two_free_slots():
    acts = [a for a in A.rank([_screen()], free_slots=2, llm_free=0) if a.kind != "idea"]
    assert [(a.kind, a.node, a.runs, a.pre) for a in acts] == [("screen", "s", 2, False)]     # was: nothing / a relax seed
    assert abs(acts[0].gpu_h - 2.0) < 1e-9 and "partial: 2 of 3" in acts[0].why


def test_a_screen_that_fits_is_whole_and_partial_never_overbooks():
    acts = [a for a in A.rank([_screen("a"), _screen("b")], free_slots=4, llm_free=0) if a.kind != "idea"]
    assert [(a.node, a.runs) for a in acts] == [("a", 3), ("b", 1)]                         # the 4th slot is b's first seed
    assert sum(a.runs for a in acts) == 4
    assert [a.runs for a in A.rank([_screen()], free_slots=3, llm_free=0)] == [3]
    assert not [a for a in A.rank([_screen()], free_slots=0, llm_free=0) if a.kind != "idea"]


def test_the_partial_screen_beats_a_relax_seed_for_the_slots_and_respects_the_screen_cap(monkeypatch):
    monkeypatch.setattr(A, "FILL_DROP_BELOW", False)
    seed = Option("seed", "x", "explore", 0.0002, 0.0005 ** 2, S2, 1.0, 1, below=True) if "below" in Option.__dataclass_fields__ \
        else Option("seed", "x", "explore", 0.0002, 0.0005 ** 2, S2, 1.0, 1)
    got = [a for a in A.rank([_screen(), seed], free_slots=2, llm_free=0) if a.kind != "idea"]
    assert got[0].kind == "screen" and got[0].runs == 2 and not got[0].pre
    assert not [a for a in A.rank([_screen()], free_slots=2, llm_free=0, caps={"screen": 0}) if a.kind != "idea"]


def test_the_real_core_launches_only_the_partial_screens_runs(ctx, store, tmp_path):
    _node(store, "n1")
    tick = _real_tick(ctx, store, tmp_path)
    data = CD.load(tmp_path / "loop.db", ctx.task_id)
    full = A.Action("screen", "n1", "explore", 3.0, 1e-5, 3, "")
    part = dataclasses.replace(full, runs=2)
    assert len(tick.core.runs_for(full, data, tick.blocks(), {}, ctx.cfg)) == 3
    assert len(tick.core.runs_for(part, data, tick.blocks(), {}, ctx.cfg)) == 2
    tick.close()


# ---------------------------------------------------------------- (1b) runbook: a fill run takes a free primary slot
def _book(ctx, store, primary=8, idle=2, **slurm):
    book = R.RunBook(_pools(ctx, primary=primary, idle=idle, **slurm), store, _Held(), clock=lambda: 1000.0)
    return book


def test_a_fill_run_takes_free_primary_when_the_idle_pool_is_full(ctx, store):
    book = _book(ctx, store)
    node = _node(store, "n1")
    book.fill_only = {("n1", 100_001)}
    book.used = lambda: {"primary": 6, "idle": 2}                 # live 10-22: primary 6/8, idle at its pending gate
    book.limits = lambda gate=True: {"primary": 8, "idle": 2}
    assert book.pool(node, Stage.SCREEN, 100_001) == "primary"
    book.used = lambda: {"primary": 6, "idle": 0}
    assert book.pool(node, Stage.SCREEN, 100_001) == "idle"        # a preemptible slot is still preferred


def test_a_fill_run_never_takes_the_owed_test_reserve_or_a_slot_a_high_run_waits_for(ctx, store):
    book = _book(ctx, store)
    node = _node(store, "n1")
    book.fill_only = {("n1", 100_001)}
    book.used = lambda: {"primary": 6, "idle": 2}
    book.limits = lambda gate=True: {"primary": 8, "idle": 2}
    book.test_reserve = 2                                          # two primary slots owed to TEST runs
    assert book.reserved() == 2 and book.pool(node, Stage.SCREEN, 100_001) == ""
    book.test_reserve = 1                                          # one of the 2 free is still the test's
    assert book.pool(node, Stage.SCREEN, 100_001) == "primary"
    book.used = lambda: {"primary": 7, "idle": 2}
    assert book.pool(node, Stage.SCREEN, 100_001) == ""
    book.test_reserve, book.used = 0, (lambda: {"primary": 6, "idle": 2})
    book.high_demand = 1
    assert book.pool(node, Stage.SCREEN, 100_001) == ""


def test_the_fill_on_primary_switch_restores_preemptible_only(ctx, store):
    book = _book(ctx, store, fill_on_primary=False)
    node = _node(store, "n1")
    book.fill_only = {("n1", 100_001)}
    book.used = lambda: {"primary": 6, "idle": 2}
    book.limits = lambda gate=True: {"primary": 8, "idle": 2}
    assert book.pool(node, Stage.SCREEN, 100_001) == ""


# ---------------------------------------------------------------- (2) program doses: priority, throttle, drop
def _ladder_nodes(store, now):
    """n1 an idea (30 h old), n2 / n3 ladder doses (30 h old), n4 a tune node whose notes read ``ladder=+0`` (30 h old)."""
    for nid in ("n1", "n2", "n3", "n4", "n5"):
        _node(store, nid)
        store.update_node(nid, created_ts=now - 30 * H)
    store.update_node("n2", notes="ladder=n9 dose=data.x=0.4 component=0")
    store.update_node("n3", notes="ladder=n9 loo_key=data.y")
    store.update_node("n4", notes="tune=t1 block_of=n8 ladder=+0")
    store.update_node("n5", notes="program_arm=1")


def test_program_child_is_an_arm_or_a_program_ladder_dose_not_a_tune_node(ctx, store, tmp_path):
    now = time.time()
    _ladder_nodes(store, now)
    data = CD.load(tmp_path / "loop.db", ctx.task_id)
    got = {n for n, node in data.nodes.items() if A.program_child(node)}
    assert got == {"n2", "n3", "n5"}
    st = {n: TK.ModuleCore().ST.Status.QUEUED for n in data.nodes}
    assert A.program_nodes(data, st) == {"n2", "n3", "n5"}          # ladder doses get the program-arm priority
    assert A.aged_queued(data, st, now + 500 * H) == ["n1", "n4"]   # ... and are never aged out


def test_ladder_doses_do_not_hold_the_throttle_but_idea_nodes_still_do(ctx, store, tmp_path):
    now = time.time()
    _ladder_nodes(store, now)
    data = CD.load(tmp_path / "loop.db", ctx.task_id)
    Q = TK.ModuleCore().ST.Status.QUEUED
    st = {n: Q for n in ("n2", "n3", "n5")}
    bl = AL.backlog(data, st, now)
    assert bl["n"] == 3 and bl["n_program"] == 3 and bl["median_age_h"] == 0.0     # was 30 h
    assert AL.throttle(AllocCfg(), bl) == ""                                       # the throttle lifts
    st["n1"] = Q                                                                   # one 30 h-old idea node: still throttled
    bl = AL.backlog(data, st, now)
    assert bl["median_age_h"] > 29 and "median queued age" in AL.throttle(AllocCfg(), bl)
    assert AL.backlog(data, st, now, screenable={"n1", "n2"})["n_screenable"] == 1


def test_starved_and_age_out_spare_ladder_doses(ctx, store, tmp_path):
    now = time.time()
    _ladder_nodes(store, now)
    data = CD.load(tmp_path / "loop.db", ctx.task_id)
    Q = TK.ModuleCore().ST.Status.QUEUED
    st = {n: Q for n in data.nodes}
    bl = AL.backlog(data, st, now)
    assert [n for n, _ in AL.starved(AllocCfg(), bl, data.nodes, now=now)] == ["n1", "n4"]


def test_the_real_tick_drops_the_starved_idea_but_keeps_the_ladder_doses(ctx, store, tmp_path):
    """Through ``Tick._age_out`` (the real wiring): only the idea node n1 and the tune node n4 are starved out."""
    now = time.time()
    _ladder_nodes(store, now)
    tick = _real_tick(ctx, store, tmp_path)
    data = CD.load(tmp_path / "loop.db", ctx.task_id)
    Q = tick.core.ST.Status.QUEUED
    statuses = {nid: Q for nid in data.nodes}
    out = tick._age_out(data, statuses, [], now)
    assert any("n1" in a for a in out) and not any(n in " ".join(out) for n in ("n2 ", "n3 ", "n5 "))
    from loop.contracts import NodeStatus
    assert store.get_node("n1").status is NodeStatus.REJECTED
    assert all(store.get_node(n).status is NodeStatus.QUEUED for n in ("n2", "n3", "n5"))
    tick.close()
