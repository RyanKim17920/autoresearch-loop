"""11-09 seed cap + futility stop (loop.core.acquire.screen_seeds / seed_cap / futile, ModuleCore.screen_specs).

Live nanobrain 11-08 17:55Z..00:43Z: n0001 got 10 distinct screen seeds, n0004 7, n0002 6 under a 5-seed operator cap.
The seed option was gated on the node's SCORED seed count (done runs), so a node with 3 done + 3 in flight kept
getting seeds; and incumbent mode has no gap floors, so a flat node (P(gain > 0) ~ .5) looked worth resolving."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from loop import tick as TK
from loop.core import acquire as A
from loop.core import data as D
from loop.core import model as M

from .test_acquire_floors import _fill_world


def _run(node, seed, status="done", stage="screen"):
    return D.Run(f"{node}:{stage}:{seed}", node, seed, stage, status, 0.0, 0.0, 0.0, "", 0)


def test_screen_seeds_count_distinct_live_or_done_screen_seeds():
    runs = [_run("a", 1), _run("a", 2, "running"), _run("a", 2, "running"), _run("a", 3, "crash"),
            _run("a", 4, "void"), _run("a", 9, stage="parent"), _run("b", 1, "pending")]
    assert A.screen_seeds(runs) == {"a": {1, 2}, "b": {1}}
    assert A.screen_seeds(runs, "b") == {"b": {1}}


def test_seed_cap_takes_the_operator_cap(monkeypatch):
    assert A.seed_cap(A.Costs()) == A.Costs().max_seeds
    monkeypatch.setattr(A, "SEED_CAP", 3)
    assert A.seed_cap(A.Costs()) == 3
    monkeypatch.setattr(A, "SEED_CAP", 9)
    assert A.seed_cap(A.Costs()) == 5                              # Costs.max_seeds still bounds


def test_configure_reads_the_futility_keys_and_the_seed_cap(monkeypatch):
    for k in ("FUTILITY_P", "FUTILITY_MIN_SEEDS", "FUTILITY_GAP_FRAC", "FUTILITY_DELTA", "SEED_CAP"):
        monkeypatch.setattr(A, k, getattr(A, k))
    from loop.spec import AcquireCfg, SeedPlan
    A.configure(SimpleNamespace(acquire=AcquireCfg(), seeds=SeedPlan(max_seeds_per_node=4)))
    assert A.FUTILITY_P == 0.02 and A.SEED_CAP == 4                # 11-10: a generic default (.02) for every task
    A.configure(SimpleNamespace(acquire=AcquireCfg(futility_p=0.05, futility_min_seeds=2, futility_delta=0.001)))
    assert (A.FUTILITY_P, A.FUTILITY_MIN_SEEDS, A.FUTILITY_DELTA) == (0.05, 2, 0.001)


def test_futile_needs_min_seeds_and_a_tiny_probability(monkeypatch):
    assert A.futile(0.0, 0.002 ** 2, 9, 0.004) is None            # off by default
    monkeypatch.setattr(A, "FUTILITY_P", 0.05)
    monkeypatch.setattr(A, "FUTILITY_MIN_SEEDS", 3)
    p = A.futile(0.0, 0.0019 ** 2, 6, 0.0039)                      # live n0001: delta ~0, sd .0019, bar .0039
    assert p is not None and p == pytest.approx(0.0201, abs=0.002)
    assert A.futile(0.0, 0.0019 ** 2, 2, 0.0039) is None          # too few seeds to judge
    assert A.futile(0.003, 0.0019 ** 2, 6, 0.0039) is None        # P ~ .32: a live idea keeps its seeds
    assert A.futile(-0.01, 0.0, 6, 0.0) == 0.0                     # degenerate sd


def test_futility_bar_is_a_gap_fraction_else_the_absolute_delta(monkeypatch):
    monkeypatch.setattr(A, "FUTILITY_GAP_FRAC", 0.33)
    monkeypatch.setattr(A, "FUTILITY_DELTA", 0.002)
    monkeypatch.setattr(A, "gap_of", lambda *a, **k: 0.012)
    assert A.futility_bar(None, "root") == pytest.approx(0.00396)
    monkeypatch.setattr(A, "gap_of", lambda *a, **k: None)
    assert A.futility_bar(None, "root") == 0.002


def _seeded(opts):
    return {o.node for o in opts if o.kind == "seed"}


def test_futility_drops_seed_options_and_idle_fill_b_of_flat_nodes(sample_db, monkeypatch):
    led, blocks, st, inc, post = _fill_world(sample_db)
    monkeypatch.setattr(M, "NEAR_MIN_DF", 0)                       # no far calibration: (b) gets the slots
    base = _seeded(A.ledger_options(led, blocks, st, inc, post=post))
    fill_b = lambda: {a.node for a in A.idle_fill(led, blocks, st, inc, post, 200, set(), A.Costs())  # noqa: E731
                      if a.why.startswith("idle (b)")}
    b0 = fill_b()
    assert base and b0
    monkeypatch.setattr(A, "FUTILITY_P", 0.5)
    monkeypatch.setattr(A, "FUTILITY_MIN_SEEDS", 1)
    monkeypatch.setattr(A, "futility_bar", lambda *a, **k: 1.0)    # a bar no node can reach: every node is futile
    assert _seeded(A.ledger_options(led, blocks, st, inc, post=post)) == set()
    assert fill_b() == set()
    acts = A.idle_fill(led, blocks, st, inc, post, 200, set(), A.Costs())
    assert any(a.why.startswith("idle (c)") for a in acts)         # the slots go to the queued screens
    monkeypatch.setattr(A, "futility_bar", lambda *a, **k: -1.0)   # every node clears it: unchanged
    assert _seeded(A.ledger_options(led, blocks, st, inc, post=post)) == base and fill_b() == b0


def test_in_flight_screen_seeds_count_toward_the_seed_option_cap(sample_db, monkeypatch):
    led, blocks, st, inc, post = _fill_world(sample_db)
    base = _seeded(A.ledger_options(led, blocks, st, inc, post=post))
    nid = sorted(base)[0]
    busy = tuple(_run(nid, 500 + i, "running") for i in range(A.Costs().max_seeds))
    led2 = dataclasses.replace(led, runs=led.runs + busy)
    assert nid not in _seeded(A.ledger_options(led2, blocks, st, inc, post=post))
    monkeypatch.setattr(A, "SEED_CAP", 2)                           # the operator cap binds below Costs.max_seeds
    led3 = dataclasses.replace(led, runs=led.runs + busy[:2])
    assert nid not in _seeded(A.ledger_options(led3, blocks, st, inc, post=post))


def test_screen_specs_never_pass_the_cap_whatever_the_path():
    core = TK.ModuleCore()
    core._select = lambda s, blocks: True
    nodes = {"r": SimpleNamespace(parent=None), "n": SimpleNamespace(parent="r")}
    runs = [_run("r", s) for s in (1, 2026, 7338)] + [_run("n", 1), _run("n", 2026), _run("n", 7338),
                                                       _run("n", 2, "crash"), _run("n", 3, "running")]
    data = SimpleNamespace(nodes=nodes, runs=runs)
    cfg = SimpleNamespace(seeds=SimpleNamespace(fresh_base=100000, max_seeds_per_node=5))
    specs = core.screen_specs("n", "r", 3, data, [], cfg)            # 4 live / done (the crash holds none): 1 left
    assert [s.seed for s in specs] == [4]
    runs.append(_run("n", 4, "pending"))
    assert core.screen_specs("n", "r", 3, data, [], cfg) == []
    cfg.seeds.max_seeds_per_node = 0                                 # no operator cap: Costs.max_seeds (5) still binds
    assert core.screen_specs("n", "r", 3, data, [], cfg) == []


def test_a_test_that_does_not_fit_whole_opens_on_the_free_slots():
    """11-10 live nanobrain: n0012 (5 seeds, test entry ok) ranked first but a test is 2 x k_test = 10 runs and the
    saturated cluster never had 10 free slots at once, so no test ever opened (seed_blocks empty)."""
    S2 = 0.003 ** 2
    test = A.Option("test", "t", "explore", 0.004, 0.002 ** 2, S2 / 5, 10.0, 10)
    seed = A.Option("seed", "s", "explore", 0.002, 0.002 ** 2, S2, 1.0, 1)
    got = A.rank([test, seed], 3, 0)
    assert [(a.kind, a.node, a.runs) for a in got][:1] == [("test", "t", 2)] and "partial" in got[0].why   # whole pairs
    assert [(a.kind, a.runs) for a in A.rank([test, seed], 12, 0)][:1] == [("test", 10)]    # fits whole
    assert not any(a.kind == "test" for a in A.rank([test, seed], 1, 0))     # < TEST_MIN_SLOTS: no pair fits
    assert not any(a.kind == "test" for a in A.rank([test, seed], 3, 0, partial_test=False))
    two = A.rank([test, dataclasses.replace(test, node="u"), seed], 6, 0)
    assert sum(a.kind == "test" for a in two) == 1                            # one partial test per rank


def test_idle_fill_screens_queued_nodes_before_seeding_nodes_behind_the_incumbent(sample_db):
    from .test_acquire_floors import _fill_world
    led, blocks, st, inc, post = _fill_world(sample_db)
    acts = A.idle_fill(led, blocks, st, inc, post, 200, set(), A.Costs())
    order = [a.why[:8] for a in acts]
    assert order == sorted(order)
    for a in acts:
        if a.why.startswith("idle (b)"):
            m = post.belief(a.node)[0] + post.shift(a.node, M.COMPOSITE, inc)[0]
            assert m > 0                                                       # (b): ahead of the incumbent only
        if a.why.startswith("idle (d)"):
            m = post.belief(a.node)[0] + post.shift(a.node, M.COMPOSITE, inc)[0]
            assert m <= 0


def test_struct_ladder_cell_tags_name_the_cell():
    from loop.core.struct_tune import cell_tag
    assert cell_tag("block_lr_mult=0.5", 0) == "block_lr_mult_0p5"
    assert cell_tag("layer_decay=0.95", 3) == "layer_decay_0p95"
    assert cell_tag("muon_lr_mult:hi,weight_decay:lo", 1) == "muon_lr_mult_hi_weight_decay_lo"
    assert cell_tag("natinit.block_lr_mult=1.5", 2) == "natinit_block_lr_mult_1p5"
    assert cell_tag("", 2) == "s2"


def test_ladder_points_are_exempt_from_futility_below_three_seeds():
    lad = SimpleNamespace(notes="ladder=n0001 struct_tune=other+operator cell=layer_decay=0.9")
    assert A.futility_exempt(lad, 2) and not A.futility_exempt(lad, 3)
    assert not A.futility_exempt(SimpleNamespace(notes="brief=explore"), 2)
