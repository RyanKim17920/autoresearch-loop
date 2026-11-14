"""loop.remint: seed rotation (never-used seed blocks for every audit / re-mint) and the incumbent RE-MINT (the
incumbent must survive a never-used block, compared on seeds neither node was selected on).
Run: uv run pytest tests/loop -q --confcutdir=tests/loop"""

from __future__ import annotations

import dataclasses

from loop import remint as RM
from loop import seeds as S
from loop.contracts import NodeStatus, Run, RunStatus, Stage
from loop.world import ComputedWorldModel, fresh_paired, reminted
from loop.world import K_AUDIT
from tests.loop.test_world import BASE, Lab


def _setup(ctx, store):
    """inc (audited incumbent) vs kid (audited): kid +0.0018 on 6 shared fresh seeds (z ~1.4); inc has 5
    more finished fresh runs (PARENT at 105..109) and selection runs at 11 / 12 (kid screens there too)."""
    lab = Lab(ctx, store, noise=False)
    for nid, fam in (("inc", "optimization"), ("kid", "regularization")):
        lab.node(nid, status=NodeStatus.CONFIRMED, tag=nid, family=fam)
        store.set_state(f"config_diff:{nid}", {f"k.{nid}": 1})
        lab.measure(nid, {a: 0.004 for a in BASE}, (11, 12))
    for s, d in zip((101, 102, 201, 202, 203, 204), (0.005, -0.002, 0.004, -0.001, 0.005, 0.0)):
        st = Stage.AUDIT if s > 200 else Stage.CONFIRM
        lab.measure("inc", {a: 0.004 for a in BASE}, (s,), stage=st)
        lab.measure("kid", {a: 0.004 + d for a in BASE}, (s,), stage=st)
    lab.measure("inc", {a: 0.004 for a in BASE}, (105, 106, 107, 108, 109), stage=Stage.PARENT)
    gate = {"verdict": "passed", "mean": 0.006, "se": 0.002, "deltas": {}, "offsets": {}}
    store.set_state(K_AUDIT, {"passed": ["inc", "kid"], "gate": {"inc": gate, "kid": {**gate, "mean": 0.005}}})
    store.set_state("world.incumbent", "inc")
    return lab


def _ctx(ctx, hours=24.0):
    return dataclasses.replace(ctx, cfg=dataclasses.replace(
        ctx.cfg, promotion=dataclasses.replace(ctx.cfg.promotion, remint_hours=hours)))


def test_draw_hands_out_never_used_seeds_and_never_the_same_twice(ctx, store):
    base = ctx.cfg.seeds.fresh_base
    store.ensure_root("base")
    store.put_run(Run(id=Run.make_id("root", Stage.SCREEN, base + 1), task_id=ctx.task_id, node_id="root",
                      seed=base + 1, stage=Stage.SCREEN, status=RunStatus.RUNNING))      # a run took base + 1
    a = RM.draw(ctx, store, 4, "audit x", now=1.0)
    b = RM.draw(ctx, store, 4, "audit y", now=2.0)                  # no run yet on a: still reserved
    assert a == [base, base + 2, base + 3, base + 4] and b == [base + 5, base + 6, base + 7, base + 8]
    assert [d["purpose"] for d in store.get_state(RM.K_DRAWN)] == ["audit x", "audit y"]
    assert not set(a + b) & RM.used_seeds(ctx, store) - set(a + b) - {base + 1}


def test_stage_needs_reports_missing_and_open_runs():
    runs = [Run(id=Run.make_id("inc", Stage.REMINT, 7), task_id="t", node_id="inc", seed=7, stage=Stage.REMINT,
                status=RunStatus.RUNNING)]
    needs, open_ = S.stage_needs("inc", Stage.REMINT, (7, 8), runs, 2, "kid")
    assert [(n.node_id, n.stage, n.seed, n.for_node) for n in needs] == [("inc", Stage.REMINT, 8, "kid")] and open_
    done = [dataclasses.replace(runs[0], status=RunStatus.DONE)]
    assert S.stage_needs("inc", Stage.REMINT, (7,), done, 2) == ([], False)


def test_remint_runs_incumbent_and_challenger_on_a_fresh_block_then_the_rule_counts_no_selection_seed(ctx, store):
    ctx = _ctx(ctx)
    lab = _setup(ctx, store)               # kid +0.0018 over inc on confirm 101/102 + audit 201..204 (z ~1.4)
    snap = ComputedWorldModel(ctx).fit(store)
    assert snap.incumbent_id == "inc" and RM.due(ctx, store, snap, 0.0) == "new incumbent"
    assert RM.due(_ctx(ctx, 0.0), store, snap, 0.0) == ""                       # off
    acts, needs = RM.tick(ctx, store, snap, infra_retry=2, now=10.0)
    seeds = store.get_state(RM.K_REMINT)["rounds"][0]["seeds"]
    assert min(seeds) >= ctx.cfg.seeds.fresh_base and len(seeds) == len(ctx.cfg.seeds.audit)
    assert sorted((n.node_id, n.seed) for n in needs) == sorted((nid, s) for nid in ("inc", "kid") for s in seeds)
    assert all(n.stage is Stage.REMINT for n in needs) and "re-mint (new incumbent)" in acts[0]
    assert RM.due(ctx, store, snap, 1e9) == ""                                  # one round at a time
    for s, d in zip(seeds, (0.004, 0.006, 0.005, 0.007)):                       # kid clearly better on fresh seeds
        lab.measure("inc", {a: 0.004 for a in BASE}, (s,), stage=Stage.REMINT)
        lab.measure("kid", {a: 0.004 + d for a in BASE}, (s,), stage=Stage.REMINT)
    acts, needs = RM.tick(ctx, store, snap, infra_retry=2, now=20.0)
    assert needs == [] and "done" in acts[0] and reminted(store, "kid", "inc")
    strict, loose = fresh_paired(ctx, store, "kid", "inc", strict=True), fresh_paired(ctx, store, "kid", "inc")
    assert strict[0] > loose[0]            # confirm 101 / 102 (selection) dropped; the new block counts
    snap = ComputedWorldModel(ctx).fit(store)
    assert snap.incumbent_id == "kid"
    assert RM.due(ctx, store, snap, 30.0) == "new incumbent"                    # the new incumbent is re-minted
    store.set_state(RM.K_REMINT, {"rounds": [{**r, "inc": "kid"} for r in store.get_state(RM.K_REMINT)["rounds"]]})
    assert RM.due(ctx, store, snap, 20.0 + 3600 * 23) == "" and RM.due(ctx, store, snap, 20.0 + 3600 * 24)


def test_an_incumbent_that_holds_on_the_fresh_block_stays(ctx, store):
    ctx = _ctx(ctx)
    lab = _setup(ctx, store)
    snap = ComputedWorldModel(ctx).fit(store)
    _, needs = RM.tick(ctx, store, snap, infra_retry=2, now=10.0)
    for n in needs:                                                             # a dead heat on the block
        lab.measure(n.node_id, {a: 0.004 + (0.001 if n.seed % 2 else -0.001) * (n.node_id == "kid")
                                for a in BASE}, (n.seed,), stage=Stage.REMINT)
    RM.tick(ctx, store, snap, infra_retry=2, now=20.0)
    assert ComputedWorldModel(ctx).fit(store).incumbent_id == "inc"


def test_no_round_without_an_audited_challenger(ctx, store):
    ctx = _ctx(ctx)
    _setup(ctx, store)
    store.update_node("kid", status=NodeStatus.NOT_CONFIRMED)
    snap = ComputedWorldModel(ctx).fit(store)
    assert RM.tick(ctx, store, snap, infra_retry=2, now=10.0) == ([], [])


def test_confirm_blocks_are_fresh_shared_by_at_most_block_children_siblings_and_spare_old_confirmations(ctx, store):
    """Live 2026-10-05: 90 confirmations on 14 / 101-104 (per-seed mean paired delta +0.0034 at 14 to -0.0025 at 104;
    4 of 6 CONFIRMED on 14). Each parent's confirming children draw never-used blocks, <= block_children per block,
    concurrent siblings on different first seeds; a node already confirming on the fixed pool finishes there."""
    from tests.loop.test_driver import _node
    store.ensure_root("base")
    cap, k = ctx.cfg.seeds.block_children, max(ctx.cfg.promotion.confirm_max_seeds, ctx.cfg.seeds.block_children)
    kids = [dataclasses.replace(_node(NodeStatus.CONFIRMING, parent="p1", nid=f"k{i}"), task_id=ctx.task_id)
            for i in range(cap + 1)]
    got = [RM.confirm_block(ctx, store, n, [], now=1.0) for n in kids]
    first = got[0][0]
    assert len(first) == k and min(first) >= ctx.cfg.seeds.fresh_base
    assert [g[0] for g in got[:cap]] == [first] * cap and [g[1] for g in got[:cap]] == list(range(cap))
    assert got[cap][1] == 0 and not set(got[cap][0]) & set(first)                 # the 5th: a new fresh block
    assert RM.confirm_block(ctx, store, kids[1], [], now=2.0) == got[1]            # stable
    assert RM.confirm_seeds(store, "p1") == set(first) | set(got[cap][0])
    firsts = [S.step(n, own_runs=[], control_runs=[], post=None, cfg=ctx.cfg, sigma_panel=0.004, infra_retry=2,
                     confirm_rot=g[1], confirm_pool=g[0]).needs for n, g in zip(kids[:cap], got)]
    assert len({ns[0].seed for ns in firsts}) == cap                                # concurrent: different seeds
    assert all((ns[1].node_id, ns[1].stage, ns[1].seed) == ("p1", Stage.PARENT, ns[0].seed) for ns in firsts)
    old = dataclasses.replace(kids[0], id="old")
    legacy = [Run(id=Run.make_id("old", Stage.CONFIRM, 101), task_id=ctx.task_id, node_id="old", seed=101,
                  stage=Stage.CONFIRM, status=RunStatus.DONE)]
    assert RM.confirm_block(ctx, store, old, legacy) is None                       # old confirmation untouched
    off = dataclasses.replace(ctx, cfg=dataclasses.replace(
        ctx.cfg, seeds=dataclasses.replace(ctx.cfg.seeds, confirm_rotate=False)))
    assert RM.confirm_block(off, store, dataclasses.replace(kids[0], id="new"), []) is None
