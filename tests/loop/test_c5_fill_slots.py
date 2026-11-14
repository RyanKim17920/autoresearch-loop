"""v3-c5 fill / slot policy (A2, A3 slot part, R3).

A2: relax / idle fill produced no decision value (live epoch 065b87f: 251 fill GPU-h, 24% preempted, 110 of 127 filled
nodes at a posterior <= 0 over the incumbent). Target mode: no relax waves by default, below-seed-floor seeds are
DROPPED from relax and from idle fill (b) (slots left idle instead), and a relax wave's points are never cancelled by
the idle churn (a zombie wave). A3: research sessions scale to ``ideation.llm_slots_max`` when slots would otherwise
idle (free + held by fill runs; threshold 2). R3: smokes skip a preemptible pool holding ANY of our runs PENDING."""

from __future__ import annotations

import dataclasses
from collections import Counter
from types import SimpleNamespace

from loop import runner as R
from loop.contracts import ROOT_NODE_ID, JobStatus, Node, NodeStatus, Operator, RunStatus, Stage, Strategy
from loop.core import acquire as A
from loop.core.acquire import Option
from loop.spec import AcquireCfg, IdeationCfg, PoolCfg, SlurmCfg

from .core.conftest import sample, sample_db  # noqa: F401 - fixtures
from .test_driver import _pnode, _PoolAdapter, _pools_ctx

S2 = 0.003 ** 2


# ---------------------------------------------------------------------------------------------- A2: config defaults
def test_c5_defaults():
    aq = AcquireCfg()
    assert aq.relax_max_waves == 0 and aq.fill_drop_below is True
    assert A.RELAX_MAX_WAVES == 0 and A.FILL_DROP_BELOW is True
    assert SlurmCfg().smoke_skip_pending == 0 and SlurmCfg().smoke_primary_max == 2
    assert IdeationCfg().llm_slots_max == 8 and IdeationCfg().llm_scale_free_slots == 2


def test_configure_reads_fill_drop_below():
    keys = dict(target_mode="auto", fresh_shrink=0.55, gap_floor_frac=0.5, stack_best_leg_frac=0.74,
                stack_sum_frac=0.24, stack_intercept=0.0013, untested_var_mult=2.0, idle_max_restarts=2,
                idle_max_run_frac=1.5)
    try:
        A.configure(SimpleNamespace(acquire=SimpleNamespace(**keys, fill_drop_below=False)))
        assert A.FILL_DROP_BELOW is False
        A.configure(SimpleNamespace(acquire=SimpleNamespace(**keys)))      # an older config keeps the value in force
        assert A.FILL_DROP_BELOW is False
    finally:
        A.FILL_DROP_BELOW = True


# ---------------------------------------------------------------------------------------------- A2: relax pass
def _below_world(gap=0.0062):
    seeds = [Option("seed", f"n{i}", "explore", 0.0002 + 0.00001 * i, 0.0005 ** 2, S2, 1.0, 1) for i in range(10)]
    wave = Option("wave", "c", "factorial", 0.0004, 0.0006 ** 2, S2 / 16, 8.0, 8)
    scr = Option("screen", "s", "explore", -0.001, 0.002 ** 2, S2 / 3, 3.0, 3)       # queued: a first screen
    return A.target_aware([*seeds, wave, scr], gap, [])


def test_target_mode_relax_drops_below_floor_seeds_and_leaves_slots_idle():
    opts = _below_world()
    assert all(o.below for o in opts if o.kind == "seed")
    acts = A.rank(opts, free_slots=40, llm_free=0)
    assert not [a for a in acts if a.kind == "seed"]                     # no below-floor seed: no decision value
    assert not [a for a in acts if a.kind == "wave"]                     # no relax wave (RELAX_MAX_WAVES 0)
    assert sum(a.runs for a in acts) < 40                                # slots left idle on purpose


def test_relax_still_screens_queued_nodes_and_the_key_restores_the_old_rule(monkeypatch):
    opts = _below_world()
    acts = A.rank(opts, free_slots=40, llm_free=0)
    assert {a.kind for a in acts} <= {"screen"}                          # a first screen is information on an idea
    monkeypatch.setattr(A, "FILL_DROP_BELOW", False)
    old = A.rank(opts, free_slots=40, llm_free=0)
    assert len([a for a in old if a.kind == "seed"]) == 10 and all(a.pre for a in old if a.kind == "seed")


def test_relax_wave_only_when_the_operator_opens_the_cap():
    scr = Option("screen", "s", "explore", 0.004, 0.002 ** 2, S2 / 3, 3.0, 3)
    weak = Option("wave", "c0", "factorial", -0.004, 0.0005 ** 2, S2 / 16, 8.0, 8)
    assert not [a for a in A.rank([scr, weak], 64, 0) if a.kind == "wave"]
    assert [a.node for a in A.rank([scr, weak], 64, 0, relax_caps={"wave": 1}) if a.kind == "wave"] == ["c0"]


def test_ranked_waves_come_before_relax_and_fill():
    """A wave above the floors (e.g. a stacking wave) is in the ranked pass, ahead of every relax action."""
    good = Option("wave", "stack", "factorial", 0.01, 0.002 ** 2, S2 / 16, 8.0, 8)
    seeds = [Option("seed", f"n{i}", "explore", 0.0002, 0.0005 ** 2, S2, 1.0, 1) for i in range(20)]
    acts = A.rank(A.target_aware([good, *seeds], 0.0062, []), free_slots=10, llm_free=0)
    assert acts and acts[0].node == "stack" and not acts[0].pre


# ---------------------------------------------------------------------------------------------- A2: idle fill (b)
def test_idle_fill_b_drops_nodes_under_the_seed_floor(sample_db, monkeypatch):  # noqa: F811
    from loop.core import model as Mo

    from .core.test_acquire_floors import _fill_world
    led, blocks, st, inc, post = _fill_world(sample_db)
    monkeypatch.setattr(Mo, "NEAR_MIN_DF", 0)                            # no far calibration: (b) gets the slots
    b = lambda acts: {a.node for a in acts if a.why.startswith(("idle (b)", "idle (d)"))}  # noqa: E731
    seeded = b(A.idle_fill(led, blocks, st, inc, post, 200, set(), A.Costs()))
    assert seeded                                                        # incumbent mode: no floor
    monkeypatch.setattr(A, "target_on", lambda: True)
    monkeypatch.setattr(A, "gap_of", lambda *a, **k: 1.0)                # a gap no node can reach
    acts = A.idle_fill(led, blocks, st, inc, post, 200, set(), A.Costs())
    assert b(acts) == set()                                              # dropped, not ordered last
    assert any(a.why.startswith("idle (c)") for a in acts)               # queued screens are unaffected
    monkeypatch.setattr(A, "gap_of", lambda *a, **k: -1.0)               # every node clears the floor
    assert b(A.idle_fill(led, blocks, st, inc, post, 200, set(), A.Costs())) == seeded
    monkeypatch.setattr(A, "gap_of", lambda *a, **k: 1.0)
    monkeypatch.setattr(A, "FILL_DROP_BELOW", False)                     # the key off: the old order-only rule
    assert b(A.idle_fill(led, blocks, st, inc, post, 200, set(), A.Costs())) == seeded


# ---------------------------------------------------------------------------------------------- A2: churn vs waves
def test_idle_churn_never_cancels_a_relax_wave_point(ctx, store):
    ctx = _pools_ctx(ctx, primary=1, idle=3)
    adapter = _PoolAdapter()
    book = R.RunBook(ctx, store, adapter)
    busy = _pnode(store, "busy")
    pt = Node(id="w1.p00", task_id=store.task_id, commit="c", parent_id=ROOT_NODE_ID, strategy=Strategy.EXPLORE,
              operator=Operator.DRAFT, family="design",
              mechanism_tag="m", diff_hash="hw", status=NodeStatus.SCREENING, notes="wave=w1 point=0 block=0 seed=1")
    store.put_node(pt)
    fill = _pnode(store, "f1")
    book.launch(busy, Stage.CONFIRM, 1)                                  # the only primary slot: no move possible
    book.fill_only = {("w1.p00", 1), ("f1", 1)}
    book.launch(pt, Stage.SCREEN, 1)
    book.launch(fill, Stage.SCREEN, 1)
    adapter.poll = lambda h: JobStatus(RunStatus.RUNNING, detail="RUNNING", restarts=2, requeued=1)
    book.poll()
    acts = book.idle_churn()
    assert len(acts) == 1 and "f1:screen:1" in acts[0]                   # the plain fill run is still dropped
    assert store.get_run("w1.p00:screen:1").status is RunStatus.RUNNING  # the wave point is kept
    assert book.idle_churn() == []


# ---------------------------------------------------------------------------------------------- A3: research slots
def _research(gpu, max_concurrent=4, **ic):
    rs = R.Research.__new__(R.Research)
    cfg = SimpleNamespace(researcher=SimpleNamespace(max_concurrent=max_concurrent),
                          ideation=dataclasses.replace(IdeationCfg(), **ic))
    rs.ctx, rs.gpu_free = SimpleNamespace(cfg=cfg), gpu
    return rs


def test_sessions_scale_to_llm_slots_max_when_a_few_slots_would_idle():
    assert _research(0).limit() == 4 and _research(2).limit() == 4      # strictly more than llm_scale_free_slots
    assert _research(3).limit() == 8                                     # live: 1-4 free most ticks -> 8, not 4
    assert _research(3, llm_scale_free_slots=12).limit() == 4            # the old threshold


def test_runbook_idle_counts_fill_runs_as_idle_and_other_pending_runs_as_used(ctx, store):
    ctx = _pools_ctx(ctx, primary=2, idle=6, smoke_reserve=0)
    book = R.RunBook(ctx, store, _PoolAdapter())
    nodes = [_pnode(store, f"n{i}") for i in range(5)]
    book.fill_only = {("n3", 1), ("n4", 1)}
    for n in nodes:
        book.launch(n, Stage.SCREEN, 1)                                  # every job PENDING (the fake SLURM)
    assert book.free() == 3                                              # 8 slots - 5 of ours
    assert book.idle() == 3 + 2                                          # + the two fill runs (no decision value)


def test_tick_feeds_research_the_idle_count():
    import inspect

    from loop import tick as T
    src = inspect.getsource(T)
    assert "self.research.gpu_free = self.book.idle()" in src


# ---------------------------------------------------------------------------------------------- R3: smokes
def _sctx(ctx, **slurm):
    return dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, slurm=SlurmCfg(pools=(
        PoolCfg("primary", slots=8), PoolCfg("idle", slots=56, preemptible=True)), **slurm)))


def test_any_pending_run_of_ours_blocks_a_pool_for_smokes_primary_still_capped(ctx, store):
    smokes = Counter()
    book = R.RunBook(_sctx(ctx), store, SimpleNamespace(smokes=smokes))
    book.pool_states = {"primary": (8, 0), "idle": (40, 1)}
    assert book.smoke_blocked("idle") and book.smoke_pool() == "primary"   # ONE pending: skip (8 never triggered)
    book.pool_states = {"primary": (8, 0), "idle": (40, 7)}               # the live 6-8 pending
    assert book.smoke_blocked("idle")
    smokes["primary"] = 2                                                  # smoke_primary_max reached
    assert book.smoke_pool() == "idle"
    book.pool_states = {"primary": (8, 0), "idle": (40, 0)}
    smokes.clear()
    assert not book.smoke_blocked("idle") and book.smoke_pool() == "idle"


def test_budgets_skip_waves_whose_result_is_recorded(monkeypatch):
    from loop.core import data as D
    from loop.core import factorial as F
    nodes = {"root": D.Node("root", None, None, "baseline", "baseline", "baseline", "root", 0.0, "", "", "B"),
             **{f"w00{i}.p00": D.Node(f"w00{i}.p00", "root", None, "factorial", "design", "factorial", "root", 0.0,
                                      "", f"wave=w00{i} point=0 block=0 seed=7", "B") for i in range(1, 3)}}
    led = D.Ledger(nodes, (), (), frozenset(), ((0.0, "root"),), bases=((0.0, "root", "B"),))
    assert len(F.open_waves(led)) == 2
    monkeypatch.setattr(A, "CLOSED_WAVES", frozenset({"w001"}))         # its result recorded: closed (tick state)
    assert A.budgets(led, {}, 10, SimpleNamespace(idea_runs=3))["caps"]["wave"] == A.MAX_OPEN_WAVES - 1
