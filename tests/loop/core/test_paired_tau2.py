"""11-10 nanobrain test-n0012: (1) the paired posterior's EB scale (bar.tau2_robust) on a heteroscedastic set of
gains -- the old (1.4826 MAD)² − mean V floored τ² and shrank n0012's +.0060 (5/5 paired seeds) to -.0004; (2) an
open TEST's candidate + control of one seed are placed whole; (3) the preemption backoff never re-queues a screen on
primary while owed TEST runs hold it."""

from __future__ import annotations

import dataclasses
from statistics import NormalDist
from types import SimpleNamespace

from loop.contracts import RunStatus, Stage
from loop.core import bar as BR
from loop.core import data as CD
from loop.core import model as M
from loop.core.data import AXES, Ledger, Node, Obs

Q = [NormalDist().inv_cdf((i + 0.5) / 17) for i in range(17)]       # 17 evenly spaced normal quantiles


def _old(ms, vs):
    return max(M.SPREADS["mad"](ms) - sum(vs) / len(vs), M.TAU2_MIN)


LIVE = [(-0.019815, 2.537e-05), (-0.016330, 2.537e-05), (-0.015588, 2.537e-05), (-0.014020, 1.756e-05),
        (-0.012640, 1.143e-05), (-0.011604, 6.656e-06), (-0.011318, 2.537e-05), (-0.011155, 1.566e-05),
        (-0.010812, 2.537e-05), (-0.008670, 2.537e-05), (-0.008327, 6.510e-06), (-0.008225, 1.199e-05),
        (-0.006414, 2.537e-05), (-0.005585, 4.568e-06), (-0.005027, 1.203e-05), (-0.004963, 7.019e-06),
        (-0.004744, 2.537e-05), (-0.004062, 1.216e-05), (-0.003915, 5.390e-06), (-0.003290, 7.600e-06),
        (-0.002924, 7.258e-06), (-0.002767, 1.320e-05), (-0.002737, 2.537e-05), (-0.002405, 2.537e-05),
        (-0.002213, 1.143e-05), (-0.001658, 2.537e-05), (-0.001598, 3.938e-06), (-0.001593, 1.172e-05),
        (-0.001449, 4.603e-06), (-0.001432, 4.791e-06), (-0.001101, 1.376e-05), (-0.000962, 1.586e-05),
        (-0.000943, 1.685e-05), (-0.000919, 2.537e-05), (-0.000583, 1.139e-05), (-0.000471, 2.537e-05),
        (-0.000403, 2.537e-05), (-0.000055, 1.157e-05), (0.000404, 1.139e-05), (0.000467, 1.023e-05),
        (0.000480, 4.729e-06), (0.000718, 4.632e-06), (0.000824, 4.624e-06), (0.000901, 4.661e-06),
        (0.001241, 1.147e-05), (0.001390, 2.537e-05), (0.001637, 1.313e-05), (0.002657, 4.218e-06),
        (0.002943, 2.537e-05), (0.003542, 1.128e-05), (0.003556, 2.537e-05), (0.006050, 4.545e-06)]


def test_live_11_10_gains_the_old_estimator_floored_now_keep_n0012():
    """The 51 raw paired gains (m, V) of the live 11-10 06:40Z ledger (n0012 last): the old (1.4826 MAD)² − mean V is
    negative (floored at TAU2_MIN, scale .0003) and n0012 shrank to -.0004; the robust Paule–Mandel τ is ~.0025 and
    n0012 keeps ~+.0042."""
    ms, vs = [m for m, _ in LIVE], [v for _, v in LIVE]
    assert _old(ms, vs) == M.TAU2_MIN
    t2 = BR.tau2_robust(ms, vs)
    assert 0.002 ** 2 < t2 < 0.003 ** 2
    loc = M._median(ms)
    assert M.posterior(loc, M.TAU2_MIN, M.Estimate(0.00605, 4.545e-06, 5, 5))[0] < 0      # the bug
    assert 0.0035 < M.posterior(loc, t2, M.Estimate(0.00605, 4.545e-06, 5, 5))[0] < 0.0050


def test_tau2_robust_recovers_a_known_spread_under_heteroscedastic_noise():
    tau, ms, vs = 0.0025, [], []
    for v in (5e-6, 1.3e-5, 2.5e-5):
        ms += [-0.0016 + z * (tau * tau + v) ** 0.5 for z in Q]
        vs += [v] * len(Q)
    assert 0.002 ** 2 < BR.tau2_robust(ms, vs) < 0.003 ** 2


def test_tau2_robust_is_the_floor_on_pure_noise_and_degenerate_input():
    ms = [z * 0.002 for z in Q]
    assert BR.tau2_robust(ms, [0.002 ** 2] * len(Q)) < 0.0005 ** 2       # V explains every bit of the spread
    assert BR.tau2_robust([0.0], [1e-6]) == M.TAU2_MIN
    assert BR.tau2_robust([0.001] * 5, [1e-6] * 5) == M.TAU2_MIN
    assert BR.tau2_robust([0.001, float("nan"), 0.002], [1e-6, 1e-6, float("inf")]) == M.TAU2_MIN


def _n(nid, parent, base="B"):
    return Node(nid, parent, None, "explore", "f", "explore", "root", 0.0, "", "", base, "")


def _o(node, seed, y, base="B", stage="screen"):
    return Obs(f"{node}:{stage}:{seed}", node, seed, stage, "select", {a: y for a in AXES}, y, 1.0, 0.0, 0, None, base)


def test_paired_keeps_a_5_of_5_paired_gain_among_many_one_seed_screens():
    """root on 6 seeds; n0012 +.006 on 5/5 paired seeds; 16 nodes at 5 seeds and 34 at 1-2 seeds around -.0016."""
    root = {1: .5479, 2: .5434, 3: .5476, 8: .5470, 2026: .5445, 7338: .5521}
    d12 = {1: .0014, 2: .0094, 3: .0042, 8: .0114, 2026: .0039}
    nodes = {"root": _n("root", None), "n0012": _n("n0012", "root")}
    obs = [_o("root", s, y, stage="baseline") for s, y in root.items()]
    obs += [_o("n0012", s, root[s] + d) for s, d in d12.items()]
    seeds5, k = [1, 2, 3, 8, 2026], 0
    for i in range(50):
        nid = f"x{i:02d}"
        nodes[nid] = _n(nid, "root")
        g = -0.0016 + 0.0025 * Q[i % 17]
        use = seeds5 if i < 16 else seeds5[i % 5:i % 5 + 1 + (i % 2)]
        for s in use:
            k += 1
            obs.append(_o(nid, s, root[s] + g + 0.004 * Q[k % 17]))
    led = Ledger(nodes, (), tuple(obs), frozenset(), ((0.0, "root"),), bases=((0.0, "root", "B"),))
    pv = BR.paired(led, "root")
    m, v, n = pv.gains["n0012"]
    assert n == 5 and pv.prior is not None and pv.prior[1] > 0.001 ** 2
    assert m > 0.003                                                     # was -.0004 at the τ² floor


# -- whole TEST pairs (Tick._launch) --------------------------------------------------------------------------------
def _launch_env(ctx, tmp_path, monkeypatch, cands, free):
    from loop import tick as TK
    from loop.core.seeds import SeedBlock
    from tests.loop.test_port import _two_epochs
    sim, store = _two_epochs(ctx, tmp_path)
    big = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, slots=16))
    tick = TK.Tick(big, store, sim, TK.ModuleCore(), db_path=tmp_path / "loop.db", clock=lambda: 50.0)
    led = CD.load(tmp_path / "loop.db", ctx.task_id)
    for c in cands:
        if store.get_node(c) is None:
            store.put_node(dataclasses.replace(store.get_node("inc@new"), id=c))
    store.put_seed_block(SeedBlock("t", CD.TEST, (100_000, 100_001), (*cands, "root"), "test", 40.0, None))
    tick._blocks = None
    # ModuleCore.runs_for's order: per candidate (cand s, control s) over the seeds; the shared control deduped later
    specs = [TK.RunSpec(n, TK.TEST_STAGE if n != "root" else TK.TEST_CONTROL_STAGE, sd)
             for c in cands for sd in (100_000, 100_001) for n in (c, "root")]
    monkeypatch.setattr(tick, "_act", lambda a, *_: specs if a.kind == "test" else [])
    monkeypatch.setattr(tick, "_eo", lambda: SimpleNamespace(route=lambda data, sp, acts: sp))
    book, made = tick.book, []
    real = book.prepare_launch

    def prep(node, stage, seed):
        made.append((node.id, seed))
        return real(node, stage, seed)
    monkeypatch.setattr(book, "prepare_launch", prep)
    monkeypatch.setattr(book, "reserved", lambda: 10)
    monkeypatch.setattr(book, "test_primary_only", lambda stage: True)
    monkeypatch.setattr(book, "pool", lambda node, stage, seed: "default")
    monkeypatch.setattr(book, "test_room", lambda node, stage, seed: free["n"] - len(made))
    return TK, tick, store, led, made


def test_an_open_tests_candidate_and_control_of_one_seed_launch_whole(ctx, tmp_path, monkeypatch):
    free = {"n": 1}
    TK, tick, store, led, made = _launch_env(ctx, tmp_path, monkeypatch, ("inc@new",), free)
    acts: list[str] = []
    assert tick._launch([], led, {}, 50.0, acts) == [] and made == []      # 1 slot: no lone candidate
    assert sum("wait for 2 free slots together" in a for a in acts) == 2
    acts = []
    assert tick._launch([], led, {}, 51.0, acts) == [] and not any("together" in a for a in acts)   # said once
    free["n"] = 3                                                        # room for one pair only
    got = tick._launch([], led, {}, 52.0, [])
    assert sorted(got) == ["inc@new:confirm:100000", "root:remint:100000"]
    tick.close()
    store.close()


def test_two_candidates_one_control_never_split_a_seed_group(ctx, tmp_path, monkeypatch):
    """review: specs A1 C1 A2 C2 B1 B2 -> groups {A1 C1 B1} / {A2 C2 B2}, not adjacent; with 5 free only the first
    group goes, all 3 of its runs (the old check-then-launch-later let A2 C2 take B1's promised slots)."""
    free = {"n": 5}
    TK, tick, store, led, made = _launch_env(ctx, tmp_path, monkeypatch, ("inc@new", "cand2"), free)
    tick._launch([], led, {}, 50.0, [])
    assert sorted(made) == [("cand2", 100_000), ("inc@new", 100_000), ("root", 100_000)]
    tick.close()
    store.close()


def test_a_released_reserve_places_test_runs_one_by_one(ctx, tmp_path, monkeypatch):
    free = {"n": 1}
    TK, tick, store, led, made = _launch_env(ctx, tmp_path, monkeypatch, ("inc@new",), free)
    monkeypatch.setattr(tick.book, "reserved", lambda: 0)
    tick._launch([], led, {}, 50.0, [])
    assert made and made[0] == ("inc@new", 100_000)
    tick.close()
    store.close()


def test_test_room_counts_the_reserved_primary_slots_for_the_tests_own_runs(ctx, store):
    from loop import runner as R
    from tests.loop.test_runner import _node, _Pending, _pools
    book = R.RunBook(_pools(ctx, primary=4, idle=8), store, _Pending())
    node = _node(store, "n1")
    book.launch(node, Stage.BASELINE, 1)
    book.launch(node, Stage.BASELINE, 2)
    book.test_reserve = 2
    assert book._open(False) == [] and book.test_room(node, Stage.CONFIRM, 100_000) == 2
    store.update_run("n1:baseline:2", status=RunStatus.DONE)
    assert book.test_room(node, Stage.CONFIRM, 100_000) == 3


# -- the preemption backoff vs owed TEST runs -----------------------------------------------------------------------
def test_preempt_backoff_never_requeues_on_primary_while_test_runs_are_owed(ctx, store):
    from loop.contracts import JobStatus
    from loop.runner import RunBook
    from tests.loop.test_driver import _pnode, _PoolAdapter, _pools_ctx
    ctx = _pools_ctx(ctx, primary=1, idle=5)
    adapter = _PoolAdapter()
    adapter.resume_state = lambda h: {"armed": False, "fraction": 0.0}
    book = RunBook(ctx, store, adapter)
    book.launch(_pnode(store, "busy"), Stage.CONFIRM, 1)                # primary full
    book.launch(_pnode(store, "cold"), Stage.SCREEN, 1)
    adapter.poll = lambda h: JobStatus(RunStatus.RUNNING, detail="RUNNING", restarts=3, requeued=1)
    book.poll()
    book.test_reserve = 2                                                # an open test owes 2 runs
    assert book.idle_churn() == []
    r = store.get_run("cold:screen:1")
    assert r.pool == "idle" and r.status is RunStatus.RUNNING and r.attempt == 0
    book.test_reserve = 0                                                # nothing owed: the O3 re-queue as before
    acts = book.idle_churn()
    assert acts and store.get_run("cold:screen:1").pool == "primary"
