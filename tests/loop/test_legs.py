"""loop.legs: the LEG FACTORIAL on the stack carrier -- the resolution-IV blocked design, leg picking (pooled P, one
per subsystem, crossover expansion, overlap with the carrier), the researcher branch, the points (one run on one seed,
4 per seed, paired against the carrier), the analysis and the optimum's no-screen-gate path to confirm.
Run: uv run pytest tests/loop -q --confcutdir=tests/loop"""

from __future__ import annotations

import dataclasses
import itertools
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from loop import design as D
from loop import legs as LG
from loop import seeds as S
from loop.agents.researcher import diff_hash
from loop.contracts import (PANEL, ROOT_NODE_ID, Node, NodePosterior, NodeStatus, Operator, Posterior, Run,
                            RunStatus, Stage, Strategy, design_point)
from loop.spec import LegsCfg


@pytest.fixture
def sim(ctx, store):
    from loop.adapters.sim import SimAdapter
    return SimAdapter(ctx)


def _on(ctx, **kw):
    return dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, legs=LegsCfg(enabled=True, **kw)))


# ------------------------------------------------------------------------------------------------ the design
@pytest.mark.parametrize("k", [3, 5, 8, 9, 12, 16])
def test_odd_word_designs_are_resolution_iv_and_blocks_never_confound_a_main_effect(k):
    b, words, rows = LG.odd_words(k)
    assert len(rows) == (16 if k <= 8 else 32 if k == 9 else 64) and len(words) == k
    cols = [tuple(r[j] for r in rows) for j in range(k)]
    assert all(sum(c) == 0 for c in cols)                                    # balanced
    for a, c in itertools.combinations(range(k), 2):
        assert sum(x * y for x, y in zip(cols[a], cols[c])) == 0             # orthogonal mains
        prod = tuple(x * y for x, y in zip(cols[a], cols[c]))
        assert all(prod not in (m, tuple(-v for v in m)) for m in cols)      # no main aliased with a 2FI
    blocks = [LG.block_of(b, i) for i in range(len(rows))]
    assert sorted(set(blocks)) == list(range(len(rows) // LG.BLOCK))
    assert all(blocks.count(x) == LG.BLOCK for x in set(blocks))
    for c in cols:                                                           # every block balanced in every factor
        assert all(sum(c[i] for i in range(len(rows)) if blocks[i] == x) == 0 for x in set(blocks))


def test_the_blocked_design_recovers_every_main_effect_under_large_per_seed_offsets():
    k = 16
    b, _, rows = LG.odd_words(k)
    rng = random.Random(3)
    eff = [rng.uniform(-0.004, 0.004) for _ in range(k)]
    seed_off = {x: rng.uniform(-0.05, 0.05) for x in range(16)}
    obs = [(i, LG.block_of(b, i), 0.6 + seed_off[LG.block_of(b, i)] + sum(e / 2 * x for e, x in zip(eff, row))
            + rng.gauss(0, 1e-6)) for i, row in enumerate(rows)]
    ana = D.analyze([f"k{j}" for j in range(k)], rows, obs, z=2.0)
    assert [e["effect"] for e in ana["effects"]] == pytest.approx(eff, abs=1e-5) and ana["df"] == 64 - 16 - 16


def test_subsystems():
    n = lambda tag, notes="", fam="training": Node(  # noqa: E731
        id="x", task_id="t", commit="c", parent_id="root", strategy=Strategy.EXPLORE, operator=None, family=fam,
        mechanism_tag=tag, diff_hash="h", status=NodeStatus.NOT_CONFIRMED, notes=notes)
    got = {t: LG.subsystem(n(t)) for t in (
        "tissue_coverage_power05_floored", "sigreg_patch_isotropy", "masked_contour_modeling",
        "jepa_predictor_bottleneck", "fino_patient_positive_bank", "zeroinit_residual_cls_branch",
        "latephase_cls_jepa_reweight", "rot90_d4_invariance", "densecl_xview_corr", "xscale_pred_width_4x",
        "salient_ibot_protos_capacity", "teacher_momentum_horizon_fix", "slide_balanced_sampling")}
    assert got == {"tissue_coverage_power05_floored": "sampling", "sigreg_patch_isotropy": "regularizer",
                   "masked_contour_modeling": "masked_modeling", "jepa_predictor_bottleneck": "predictor",
                   "fino_patient_positive_bank": "fino", "zeroinit_residual_cls_branch": "cls",
                   "latephase_cls_jepa_reweight": "schedule", "rot90_d4_invariance": "augmentation",
                   "densecl_xview_corr": "dense_objective", "xscale_pred_width_4x": "predictor",
                   "salient_ibot_protos_capacity": "ibot_proto", "teacher_momentum_horizon_fix": "teacher",
                   "slide_balanced_sampling": "sampling"}
    assert LG.subsystem(n("odd_thing", "mtype=data_sampling")) == "sampling"


# ------------------------------------------------------------------------------------------------ legs
def _node(ctx, store, nid, *, parent=ROOT_NODE_ID, status=NodeStatus.NOT_CONFIRMED, tag="", notes="", cfg=None,
          commit=None, ts=1.0):
    store.put_node(Node(id=nid, task_id=ctx.task_id, commit=commit or f"c{nid}", parent_id=parent,
                        strategy=Strategy.EXPLORE, operator=Operator.DRAFT, family="objective",
                        mechanism_tag=tag or f"mech_{nid}", diff_hash=f"h{nid}", status=status, notes=notes,
                        created_ts=ts, updated_ts=ts))
    if cfg:
        store.set_state(f"config_diff:{nid}", cfg)


def _scored(ctx, store, nid, seeds=(11, 12, 14)):
    for s in seeds:
        st = Stage.CONFIRM if s == 14 else Stage.SCREEN
        store.put_run(Run(id=Run.make_id(nid, st, s), task_id=ctx.task_id, node_id=nid, seed=s, stage=st,
                          status=RunStatus.DONE, scores={"alpha": 0.6, "beta": 0.5, "gamma": 0.4}))


def _post(nid, mean, sd=0.002):
    q = Posterior(mean, sd, 3)
    return NodePosterior(node_id=nid, increment=q, utility=q, axes={}, p_positive=0.9, p_beats_incumbent=0.5,
                         vetoed=False, veto_axes=())


def _world(ctx, store):
    """P (confirmed) <- carrier C = stack of legs a (loss.alpha_w) + b; a menu of candidate legs."""
    store.ensure_root("root0")
    store.set_state("intake.base_config", {"train.lr": 0.001, "train.weight_decay": 0.05, "aug.strength": 0.5})
    _node(ctx, store, "P", status=NodeStatus.CONFIRMED, cfg={"train.lr": 0.002})
    _node(ctx, store, "a", parent="P", tag="salient_masking", cfg={"loss.alpha_w": 0.5})
    _node(ctx, store, "b", parent="P", tag="slide_balanced_sampling")
    _node(ctx, store, "C", parent="P", status=NodeStatus.CONFIRMED, tag="stack_a_b", notes="stack_legs=a,b",
          cfg={"loss.alpha_w": 0.5, "train.lr": 0.0025})
    legs = {"x1": ("sigreg_isotropy", "P", 0.004), "x2": ("rot90_invariance", ROOT_NODE_ID, 0.003),
            "x3": ("rot90_flip_aug", "P", 0.0025),                    # same subsystem as x2, lower P
            "x4": ("salient_masking_v2", "P", 0.005),                # shares 'salient' with carrier leg a
            "x5": ("coverage_x_sampler", "P", 0.004),                # a crossover of b and x1
            "x6": ("teacher_momentum_fix", "C", 0.003),              # built on the carrier itself
            "x7": ("predictor_bottleneck", "P", -0.002)}             # P(own gain) < 0.7
    for nid, (tag, par, _) in legs.items():
        _node(ctx, store, nid, parent=par, tag=tag, notes="xover=b,x1" if nid == "x5" else "")
        _scored(ctx, store, nid)
    posts = {nid: _post(nid, m) for nid, (_, _, m) in legs.items()}
    return SimpleNamespace(nodes={**posts, "C": _post("C", 0.01), "P": _post("P", 0.01)}, incumbent_id="C",
                           sigma={PANEL: 0.004}, own_confirm={})


def test_legs_best_per_subsystem_crossovers_expanded_overlaps_dropped(ctx, store):
    ctx = _on(ctx)
    snap = _world(ctx, store)
    legs, dropped = LG.candidates(ctx, store, snap, "C")
    assert [r["id"] for r in legs] == ["x1", "x2", "x6"]
    why = {r["id"]: r["why"] for r in dropped}
    assert "subsystem augmentation" in why["x3"] and "salient" in why["x4"] and "crossover" in why["x5"]
    assert "x7" not in why and LG.own_legs(store, "C") == ["a", "b"]
    assert LG.shared_knobs(ctx, store, "C") == [{"key": "train.lr", "low": 0.002, "high": 0.0025, "kind": "shared",
                                                 "leg": "", "levers": {}}]
    p = LG.plan(ctx, store, snap)
    assert p["carrier"] == "C" and p["runs"] == 16 and p["seeds"] == 4 and p["expose"] == ["b"]   # 3 + 2 + 1 factors


# ------------------------------------------------------------------------------------------------ the flow
def _sim_world(ctx, store, sim):
    """A sim carrier C (loss.alpha_w 0.5 = leg a; leg b hard-coded; train.lr retuned from P's) and legs x1, x2."""
    from loop.adapters import flatten
    from loop.adapters.sim import BASE_CONFIG
    root = store.ensure_root(sim.root_commit())
    store.set_state("intake.base_config", flatten(BASE_CONFIG))

    def commit(parent, nid, cfg):
        wt = sim.prepare_worktree(parent_commit=parent, name=nid)
        data = yaml.safe_load((Path(wt.path) / "config.yaml").read_text())
        for k, v in cfg.items():
            sec, key = k.split(".")
            data.setdefault(sec, {})[key] = v
        (Path(wt.path) / "config.yaml").write_text(yaml.safe_dump(data, sort_keys=True))
        _, _, cfg_diff = sim.diff(wt)
        sha = sim.commit(wt, message=nid)
        sim.reap_worktree(wt)
        return sha, dict(cfg_diff)
    cp, dp = commit(root.commit, "P", {"train.lr": 0.002})
    _node(ctx, store, "P", status=NodeStatus.CONFIRMED, cfg=dp, commit=cp)
    _node(ctx, store, "a", parent="P", tag="salient_masking", cfg={"loss.alpha_w": 0.5})
    _node(ctx, store, "b", parent="P", tag="slide_balanced_sampling")
    cc, dc = commit(cp, "C", {"train.lr": 0.0025, "loss.alpha_w": 0.5})
    _node(ctx, store, "C", parent="P", status=NodeStatus.CONFIRMED, tag="stack_a_b", notes="stack_legs=a,b",
          cfg=dc, commit=cc)
    for nid, tag in (("x1", "sigreg_isotropy"), ("x2", "rot90_invariance")):
        _node(ctx, store, nid, parent="P", tag=tag)
        _scored(ctx, store, nid)
    snap = SimpleNamespace(nodes={n: _post(n, 0.004) for n in ("x1", "x2", "C", "P")}, incumbent_id="C",
                           sigma={PANEL: 0.004}, own_confirm={})
    return snap, commit


def test_the_branch_the_points_and_the_optimum_go_to_confirm_without_a_screen_gate(ctx, store, sim):
    ctx = _on(ctx)
    snap, commit = _sim_world(ctx, store, sim)
    ids = iter(f"n{i:04d}" for i in range(100, 400))
    new_id = lambda: next(ids)                                                   # noqa: E731
    acts = LG.tick(ctx, store, sim, snap, new_id=new_id, now=1.0)
    assert "leg factorial L1 on carrier C" in acts[0], acts
    assert LG.tick(ctx, store, sim, snap, new_id=new_id, now=1.5, inflight=[]) == []   # not asked yet: waits
    (req,) = LG.requests(ctx, store, snap, limit=2)
    assert req.brief_kind == "legs" and req.parent_ids == ("C",) and "x1" in req.brief and "b slide" in req.brief
    assert LG.requests(ctx, store, snap, limit=2) == []                          # one in flight
    # the researcher's branch: x1 / x2 flags (default 0), leg b's strength multiplier (default 1), identical
    sha, cfg = commit(store.get_node("C").commit, "br", {"loss.leg_x1": 0, "aug.leg_x2": 0, "loss.leg_b": 1.0})
    br = Node(id="n0050", task_id=ctx.task_id, commit=sha, parent_id="C", strategy=req.strategy, operator=req.operator,
              family="objective", mechanism_tag="leg_branch_L1", diff_hash=diff_hash(sha), status=NodeStatus.QUEUED,
              notes=f"request={req.id} brief=legs identical=1")
    store.put_node(LG.adopt(store, req, br))
    store.set_state("config_diff:n0050", cfg)
    assert store.get_node("n0050").status is NodeStatus.REJECTED and design_point(store.get_node("n0050"))
    acts = LG.tick(ctx, store, sim, snap, new_id=new_id, now=2.0, inflight=[])
    cur = LG.book(store)["current"]
    assert cur is not None, (acts, LG.book(store)["batches"])
    assert cur["phase"] == "run" and set(cur["designs"]) == {"legs", "dose"}, acts
    leg = cur["designs"]["legs"]
    assert [f["leg"] or f["key"] for f in leg["factors"]] == ["x1", "x2", "a", "b", "train.lr"]
    assert leg["corner"] == [-1, -1, 1, 1, 1] and cur["designs"]["dose"]["corner"] == [0, 0, 1]
    pts = {i: store.get_node(n) for i, n in leg["points"].items() if n != "C"}
    seeds = [int(LG._note(p, "seed1")) for p in pts.values()]
    assert all(p.parent_id == "C" and design_point(p) for p in pts.values())
    assert len(pts) == 15 and sorted(set(seeds)) == cur["seeds"] and len(cur["seeds"]) == 4   # one row IS C
    assert min(cur["seeds"]) >= ctx.cfg.seeds.fresh_base            # never-used rotation seeds, never a screen block
    blocks = S.assign_blocks(store.nodes(), ctx.cfg.seeds, store.runs())
    assert blocks is not None
    # every point and the carrier run their seed; truth: x1 +0.004, x2 -0.002, a +0.003, b 0, lr -0.001 (panel)
    truth = {"loss.leg_x1": (0, 1, 0.004), "aug.leg_x2": (0, 1, -0.002), "loss.alpha_w": (0.0, 0.5, 0.003),
             "loss.leg_b": (0.0, 1.0, 0.0), "train.lr": (0.002, 0.0025, -0.001)}

    def panel(levels):
        return 0.6 + sum((e / 2) * (1 if levels.get(k, hi) == hi else -1) for k, (lo, hi, e) in truth.items())
    cfg_c = {k: lo if ".leg_x" in k else hi for k, (lo, hi, e) in truth.items()}      # the carrier: new legs off
    for s in cur["seeds"]:
        y = panel(cfg_c) + 0.01 * s % 0.03
        store.put_run(Run(id=Run.make_id("C", Stage.PARENT, s), task_id=ctx.task_id, node_id="C", seed=s,
                          stage=Stage.PARENT, status=RunStatus.DONE, scores={a: y for a in ("alpha", "beta", "gamma")}))
    snap.sigma = {PANEL: 0.0005}     # se floor 2 sigma / sqrt(19): every true effect is credible after shrinkage
    from loop.driver import RunBook
    rb = RunBook(ctx, store, SimpleNamespace(submit=lambda run, commit: None))
    legs_acts = []
    for name, d in sorted(cur["designs"].items(), key=lambda x: x[0] != "legs"):    # legs first
        for i, nid in d["points"].items():
            if nid == "C":
                continue
            p = store.get_node(nid)
            s = int(LG._note(p, "seed1"))
            lv = {**cfg_c, **{f["key"]: (f["high"] if x > 0 else f["low"]) for f, x in zip(d["factors"], d["rows"][int(i)])}}
            y = panel(lv) + 0.01 * s % 0.03
            store.put_run(Run(id=Run.make_id(nid, Stage.SCREEN, s), task_id=ctx.task_id, node_id=nid, seed=s,
                              stage=Stage.SCREEN, status=RunStatus.DONE,
                              scores={a: y for a in ("alpha", "beta", "gamma")}))
            if name == "legs" and LG.stragglers(store) == set():      # 14 open points: LOW, held to design points'
                assert not rb.high(p, Stage.SCREEN, s) and LG.leg_point(p) and not rb._batch_point(p)  # rules, uncapped
            store.update_node(nid, status=NodeStatus.SCREENED_OUT)
            if name == "legs" and 0 < len(open_ := LG.stragglers(store)) <= LG.BLOCK:  # the last block: HIGH
                assert all(rb.high(store.get_node(x), Stage.SCREEN, 0) for x in open_)
        if name == "legs":       # the legs design is analysed (its optimum graduates) while the dose design runs
            legs_acts = LG.tick(ctx, store, sim, snap, new_id=new_id, now=2.5)
            cur = LG.book(store)["current"]
            assert cur is not None and set(cur["result"]) == {"legs"} and cur["result"]["legs"]["node"], legs_acts
            assert "legs design analysed" in LG.lines(store)
            assert LG.tick(ctx, store, sim, snap, new_id=new_id, now=2.6) == []       # analysed once
    acts = legs_acts + LG.tick(ctx, store, sim, snap, new_id=new_id, now=3.0)
    assert LG.book(store)["current"] is None
    res = LG.book(store)["batches"][-1]["result"]
    eff = {e["key"]: round(e["effect"], 5) for e in res["legs"]["effects"]}
    assert eff == {k: round(e, 5) for k, (_, _, e) in truth.items()}, acts
    assert res["legs"]["best"] == {"loss.leg_x1": 1, "aug.leg_x2": 0, "loss.alpha_w": 0.5, "loss.leg_b": 1.0,
                                   "train.lr": 0.002}
    assert res["legs"]["gain"] == pytest.approx(0.004 + 0.001, rel=0.02)       # x1 on, lr back to P's (EB-shrunk)
    assert res["legs"]["raw_gain"] == pytest.approx(0.005) and res["legs"]["tau"] > 0
    best = store.get_node(res["legs"]["node"])
    assert best.parent_id == "C" and best.status is NodeStatus.SCREENING and not design_point(best)
    assert "legs_of=C" in best.notes and any("leg factorial L1 legs design" in a for a in acts)
    # no screen gate: done screen seeds -> CONFIRMING whatever the posterior says
    post = _post(best.id, -0.01)
    blocks = S.assign_blocks(store.nodes(), ctx.cfg.seeds, store.runs())
    scr = S.screen_block(ctx.cfg.seeds, blocks[best.id])
    assert not set(scr) & set(cur["seeds"])                                     # a fresh block
    assert not set(ctx.cfg.seeds.confirm_pool) & set(cur["seeds"])             # confirmed off the factorial's seeds
    runs = [Run(id=Run.make_id(best.id, Stage.SCREEN, s), task_id=ctx.task_id, node_id=best.id, seed=s,
                stage=Stage.SCREEN, status=RunStatus.DONE, scores={"alpha": 0.6}) for s in scr]
    ctl = [Run(id=Run.make_id("C", Stage.PARENT, s), task_id=ctx.task_id, node_id="C", seed=s, stage=Stage.PARENT,
               status=RunStatus.DONE, scores={"alpha": 0.6}) for s in scr]
    dec = S.step(best, own_runs=runs, control_runs=ctl, post=dataclasses.replace(post, p_beats_incumbent=0.0),
                 cfg=ctx.cfg, sigma_panel=0.004, infra_retry=2, screen=scr)
    assert dec.status is NodeStatus.CONFIRMING and "legs_of" in dec.reason


def test_design_points_are_not_exploit_offspring(ctx, store):
    """A leg factorial's 48 points on the incumbent are not research offspring (the exploit penalty)."""
    from loop.select import thompson
    _node(ctx, store, "C")
    _node(ctx, store, "kid", parent="C")
    store.put_node(Node(id="pt", task_id=ctx.task_id, commit="cpt", parent_id="C", strategy=Strategy.DESIGN,
                        operator=None, family="design", mechanism_tag="L1lpt00", diff_hash="hpt",
                        status=NodeStatus.SCREENED_OUT, notes="legs=L1 seed1=611 design x=1"))
    assert design_point(store.get_node("pt")) and LG.leg_point(store.get_node("pt"))
    assert thompson.offspring_counts(store.nodes()) == {ROOT_NODE_ID: 1, "C": 1}


def test_an_effect_se_is_floored_at_the_panel_sigma_implied_se():
    ana = {"n": 16, "effects": [{"estimable": True, "effect": -0.0025, "se": 0.0009, "significant": True},
                                {"estimable": True, "effect": 0.001, "se": 0.003, "significant": False},
                                {"estimable": False, "effect": None, "se": None, "significant": False}]}
    floor = LG.se_floor(ana, 0.002758, 2.0)
    assert floor == pytest.approx(2 * 0.002758 / 4)
    e0, e1, _ = ana["effects"]
    assert e0["se"] == pytest.approx(floor) and e0["se_resid"] == 0.0009 and not e0["significant"]
    assert e1["se"] == 0.003 and "se_resid" not in e1


def test_shrinkage_keeps_the_carrier_when_the_effect_spread_is_pure_noise():
    """L1: 15 effects whose spread equals their se -> tau 0, every P = 0.5, nothing flips, shrunk gain 0."""
    eff = [{"estimable": True, "effect": x, "se": 0.002} for x in (0.003, -0.002, 0.0015, -0.001, 0.0005, 0.002)]
    assert LG.shrink(eff) == 0 and all(e["shrunk"] == 0 and e["p_pos"] == 0.5 for e in eff)
    big = [{"estimable": True, "effect": x, "se": 0.0005} for x in (0.006, -0.005, 0.0002, 0.004)]
    assert LG.shrink(big) > 0.003 and big[0]["p_pos"] > 0.99 and big[1]["p_pos"] < 0.01
    assert 0.3 < big[2]["p_pos"] < 0.8 and abs(big[2]["shrunk"]) < 0.0002


def test_a_legs_graduates_fresh_screen_seeds_count_in_its_confirm_evalue_an_ordinary_nodes_never():
    from loop.world import evalues as EV
    from loop.world.noise import NodeData, Obs

    def nd(notes):
        node = Node(id="g", task_id="t", commit="c", parent_id="C", strategy=Strategy.DESIGN, operator=Operator.DRAFT,
                    family="design", mechanism_tag="L1lbest", diff_hash="h", status=NodeStatus.CONFIRMING, notes=notes)
        obs = [Obs(Stage.SCREEN, "C/s1", {PANEL: 0.004}), Obs(Stage.SCREEN, "C/s2", {PANEL: 0.003}, resumed=True),
               Obs(Stage.CONFIRM, "C/s3", {PANEL: 0.002})]
        return NodeData(node, obs, {}, 0.0, selection={PANEL: [(0.004, 1e-6), (0.003, 1e-6)]}, confirm=[(0.002, 1e-6)])
    assert EV.confirm_increments(nd("legs_of=C legs=L1")) == ([0.002, 0.004, 0.003], [False, False, True])
    assert EV.confirm_increments(nd("")) == ([0.002], [False])
    assert EV.lr_evalue([0.002, 0.004, 0.003], 0.003, 0.003) > EV.lr_evalue([0.002], 0.003, 0.003)


def test_graduation_needs_the_shrunk_gain_over_the_confirm_mde_and_the_raw_optimum_over_the_null_max(ctx, store):
    ctx = _on(ctx)
    snap = SimpleNamespace(sigma={PANEL: 0.002758})
    eff = [{"estimable": True, "effect": x, "se": 0.0014} for x in (0.003, -0.002, 0.0015, 0.001, -0.0025)]
    d = {"corner": [-1, -1, 1, 1, 1]}
    q = LG.null_max_q(eff, d["corner"], 0.95)
    assert 0.003 < q < 0.008                              # picking the better level of 5 pure-noise effects 'gains'
    noise = LG.graduation(ctx, store, snap, d, {"effects": eff, "gain": 0.0008, "raw_gain": 0.0040})
    assert not noise["ok"] and "confirm MDE" in noise["why"] and 0.002 < noise["mde"] < 0.005
    real = LG.graduation(ctx, store, snap, d, {"effects": eff, "gain": 0.006, "raw_gain": 0.012})
    assert real["ok"] and real["why"] == ""


def test_a_shrunk_optimum_over_the_bar_graduates_though_its_raw_misses_the_null_max(ctx, store):
    """Live L4: legs n1030 / n0090 significant, shrunk optimum +0.00418 vs MDE 0.00273, raw +0.00634 <= null q0.95
    +0.00701 -> discarded. Fresh confirmation (not a second selection gate) guards the winner's curse."""
    ctx = _on(ctx)
    snap = SimpleNamespace(sigma={PANEL: 0.002758})
    eff = [{"estimable": True, "effect": x, "se": 0.0014} for x in (0.003, -0.002, 0.0015, 0.001, -0.0025)]
    d = {"corner": [-1, -1, 1, 1, 1]}
    q = LG.null_max_q(eff, d["corner"], 0.95)
    g = LG.graduation(ctx, store, snap, d, {"effects": eff, "gain": 0.0042, "raw_gain": q - 1e-4})
    assert g["ok"] and g["levels"] == "optimum" and "null gate not passed" in g["reason"] and g["bar"] == g["mde"]
    off = dataclasses.replace(ctx, cfg=dataclasses.replace(
        ctx.cfg, promotion=dataclasses.replace(ctx.cfg.promotion, shrunk_grad_mde=0.0)))
    assert not LG.graduation(off, store, snap, d, {"effects": eff, "gain": 0.0042, "raw_gain": q - 1e-4})["ok"]
    # the shrunk optimum fails, the significant legs alone clear the bar: they graduate
    sig = LG.graduation(ctx, store, snap, d, {"effects": eff, "gain": 0.001, "raw_gain": 0.002, "sig_gain": 0.004,
                                              "sig_best": {"a": 1}})
    assert sig["ok"] and sig["levels"] == "significant" and sig["gain"] == 0.004
    assert LG.grad_levels({"best": {"a": 0, "b": None}, "sig_best": {"a": 1, "b": None}}, sig) == {"a": 1}


def test_analyse_builds_the_significant_legs_only_combination(ctx, store, monkeypatch):
    fac = [{"key": k, "low": 0, "high": 1} for k in "abc"]
    eff = [{"estimable": True, "effect": 0.003, "se": 0.001, "significant": True},
           {"estimable": True, "effect": -0.0012, "se": 0.001, "significant": False},
           {"estimable": True, "effect": -0.003, "se": 0.001, "significant": True}]
    monkeypatch.setattr(LG, "_obs", lambda *a: [])
    monkeypatch.setattr(LG.D, "analyze", lambda *a: {"effects": [dict(e) for e in eff], "interactions": [], "n": 0,
                                                    "df": 0})
    ana = LG.analyse(_on(ctx), store, {}, {"factors": fac, "rows": [], "corner": [-1, 1, 1]})
    # a (significant +) goes high, b (not significant) stays at the carrier's high, c (significant -) goes low
    assert ana["sig_best"] == {"a": 1, "b": 1, "c": 0}
    e = ana["effects"]
    assert ana["sig_gain"] == pytest.approx(e[0]["shrunk"] - e[2]["shrunk"]) and ana["sig_gain"] > 0


def test_the_graduation_retro_runs_once_and_reports_only_graduates(ctx, store, monkeypatch):
    from loop import regrad as GR
    calls = []
    monkeypatch.setattr(GR, "retro", lambda *a, **k: calls.append(k) or [
        "retro: leg factorial L4 legs on n0655: node n9 {} (graduate on shrunk ...) -> confirm",
        "retro: variants for n1: best x still no graduate"])
    snap = SimpleNamespace(sigma={PANEL: 0.002758}, nodes={})
    out = GR.once(ctx, store, None, snap, new_id=None, now=1.0)
    assert out == ["retro: leg factorial L4 legs on n0655: node n9 {} (graduate on shrunk ...) -> confirm"]
    assert GR.once(ctx, store, None, snap, new_id=None, now=2.0) == [] and len(calls) == 1
    assert store.get_state(GR.K_RETRO)["lines"][1].endswith("still no graduate")
    # a closed leg batch: retro (dry) re-judges each design that sent no optimum, within the window only
    monkeypatch.undo()
    store.set_state(LG.K_LEGS, {"seq": 2, "current": None, "batches": [
        {"batch": 1, "carrier": "c", "closed": 0.0, "designs": {"legs": {}}, "result": {"legs": {"node": ""}}}]})
    assert GR.retro(ctx, store, None, snap, new_id=None, now=1e9, dry=True) == []        # closed > 48 h ago


def test_dose_levels_never_code_the_carrier_value_as_the_low_level():
    assert LG.dose_levels("dino.jepa_leg_n0365", 1, True) == (0.5, 2.0)      # a multiplier written as int 1
    assert LG.dose_levels("dino.jepa_salient_tau", 1.5, False) == (0.75, 3.0)
    assert LG.dose_levels("data.n_crops", 4, False) == (2, 8)
    assert LG.dose_levels("data.n_crops", 1, False) is None                 # halved == carrier: dropped


def test_a_legs_graduate_needs_one_confirm_run_before_it_is_confirmed_whatever_its_screen_e(ctx):
    node = Node(id="g", task_id=ctx.task_id, commit="c", parent_id="C", strategy=Strategy.DESIGN,
                operator=Operator.DRAFT, family="design", mechanism_tag="L1dbest", diff_hash="h",
                status=NodeStatus.CONFIRMING, notes="legs_of=C legs=L1")
    kw = dict(control_runs=[], post=_post("g", 0.01), cfg=ctx.cfg, sigma_panel=0.003, infra_retry=2)
    dec = S.step(node, own_runs=[], confirm=(1e6, 2), **kw)                  # E from its 2 screen seeds alone
    assert dec.status is None and any(r.stage is Stage.CONFIRM for r in dec.needs)
    seed = dec.needs[0].seed if dec.needs[0].stage is Stage.CONFIRM else next(r.seed for r in dec.needs
                                                                                if r.stage is Stage.CONFIRM)
    run = Run(id=Run.make_id("g", Stage.CONFIRM, seed), task_id=ctx.task_id, node_id="g", seed=seed,
              stage=Stage.CONFIRM, status=RunStatus.DONE, scores={"alpha": 0.6})
    ctl = Run(id=Run.make_id("C", Stage.PARENT, seed), task_id=ctx.task_id, node_id="C", seed=seed,
              stage=Stage.PARENT, status=RunStatus.DONE, scores={"alpha": 0.6})
    dec = S.step(node, own_runs=[run], confirm=(1e6, 3), **{**kw, "control_runs": [ctl]})
    assert dec.status is NodeStatus.CONFIRMED
    plain = dataclasses.replace(node, notes="")
    assert S.step(plain, own_runs=[], confirm=(1e6, 2), **kw).status is NodeStatus.CONFIRMED


def test_a_further_round_is_never_deadlocked_by_the_carrier_legs_eating_the_factor_cap(ctx, store):
    """live 10-05..10-07: max_factors 8 - 4 carrier legs - 1 shared knob left a 3-leg cap under min_new 4: no further
    round on n0655 could ever start. The new-leg cap is never under min_new."""
    ctx = _on(ctx, max_factors=4, min_new=3)                   # 2 carrier legs + 1 shared knob: 4 - 3 = 1 < 3
    snap = _world(ctx, store)
    store.set_state(LG.K_LEGS, {"seq": 1, "current": None, "batches": [{"batch": 1, "carrier": "C", "legs": []}]})
    assert LG.new_cap(ctx, 3) == 3 and LG.new_cap(ctx, 0) == 4 and LG.new_cap(ctx, 15) == 1
    legs, dropped = LG.candidates(ctx, store, snap, "C")
    assert [r["id"] for r in legs] == ["x1", "x2", "x6"] and not any("cap" in r["why"] for r in dropped)
    p = LG.plan(ctx, store, snap)
    assert p is not None and p["round"] == 2 and len(p["legs"]) == 3


def test_well_measured_promising_nodes_are_legs_and_winner_cross_partners_without_being_confirmed(ctx, store):
    """A leg need not be CONFIRMED nor clear pooled P >= min_p: a node with >= legs.fresh_seeds positive fresh confirm
    seeds (here still CONFIRMING), a banked node (and its children: an established parent), and a backfill port onto
    the carrier with a positive own increment there qualify; a port of a leg a round already tried does not."""
    ctx = _on(ctx, max_new=20, max_factors=16)
    snap = _world(ctx, store)
    for nid, tag, par, st, mean, seeds, notes in (
            ("y1", "fino_patient_bank", "P", NodeStatus.CONFIRMING, -0.002, (11, 12, 14), ""),
            ("y2", "contour_modeling", "P", NodeStatus.NOT_CONFIRMED, -0.001, (11, 12, 14), ""),
            ("y3", "densecl_xview", "C", NodeStatus.SCREENED_OUT, 0.0005, (11, 12), "backfill_of=x7"),
            ("y4", "jepa_attmask", "y2", NodeStatus.SCREENED_OUT, 0.004, (11, 12, 14), ""),
            ("y5", "cls_token_bank", "C", NodeStatus.SCREENED_OUT, 0.0005, (11, 12), "backfill_of=x9"),
            ("y6", "fino_subtype", "P", NodeStatus.CONFIRMING, -0.002, (11, 12), "")):
        _node(ctx, store, nid, parent=par, tag=tag, status=st, notes=notes)
        _scored(ctx, store, nid, seeds)
        snap.nodes[nid] = _post(nid, mean)
    snap.own_confirm = {"y1": (0.001, 3), "y6": (0.001, 2)}
    snap.banked = ("y2",)
    store.set_state(LG.K_LEGS, {"seq": 1, "current": None,
                                "batches": [{"batch": 1, "carrier": "C", "legs": [{"id": "x9"}]}]})
    legs, dropped = LG.candidates(ctx, store, snap, "C")
    why = {r["id"]: r["why_leg"] for r in legs}
    assert {"y1", "y2", "y3", "y4"} <= set(why) and "y6" not in why, (why, dropped)
    assert "fresh" in why["y1"] and why["y2"] == "banked" and "port of x7" in why["y3"]
    assert "y5" not in why and "round L1" in next(r["why"] for r in dropped if r["id"] == "y5")
    assert LG.pool(ctx, store, snap) == sorted({"x1", "x2", "x3", "x4", "x5", "x6", "y1", "y2", "y4"})


def test_a_further_round_runs_on_the_cadence_and_a_strong_challenger_gets_its_own_round(ctx, store, sim):
    """3 new legs < min_new 4: no further round on C until legs.round_hours passed since its last; meanwhile a
    CONFIRMED challenger with P(U>U_inc) >= legs.challenger_p gets its first round (2+ legs). Scarce mode starts it."""
    h = 3600.0
    ctx = _on(ctx, round_hours=12, challenger_p=0.9)
    snap = _world(ctx, store)
    store.set_state(LG.K_LEGS, {"seq": 1, "current": None,
                                "batches": [{"batch": 1, "carrier": "C", "legs": [], "closed": 10 * h}]})
    assert LG.plan(ctx, store, snap, now=11 * h) is None
    p = LG.plan(ctx, store, snap, now=23 * h)
    assert p["carrier"] == "C" and p["round"] == 2 and "cadence" in p["why"] and len(p["legs"]) == 3
    _node(ctx, store, "D", status=NodeStatus.CONFIRMED, tag="teacher_ema_fix", cfg={"aug.d": 1})
    snap.nodes["D"] = dataclasses.replace(_post("D", 0.01), p_beats_incumbent=0.95)
    assert LG.carriers(ctx, store, snap) == ["C", "D"]
    p = LG.plan(ctx, store, snap, now=11 * h)
    assert p["carrier"] == "D" and p["round"] == 1 and "challenger" in p["why"] and len(p["legs"]) >= 2
    acts = LG.tick(ctx, store, sim, snap, new_id=lambda: "n9999", now=11 * h, scarce=True)
    assert LG.book(store)["current"]["carrier"] == "D" and "strong challenger" in acts[-1]
    assert LG.hb(store) == f"legs=L2(build on D: {len(p['legs'])} legs)"


def test_the_heartbeat_shows_the_running_round_and_its_runs_else_the_last_rounds_age(ctx, store):
    from loop.driver import hb_line
    assert LG.hb(store) == "legs=none"
    store.set_state(LG.K_LEGS, {"seq": 3, "current": None, "batches": [{"batch": 3, "carrier": "C", "closed": 0.0}]})
    assert LG.hb(store, now=7200.0) == "legs=idle(L3 on C closed 2 h ago)"
    _node(ctx, store, "C")
    _node(ctx, store, "p1", status=NodeStatus.SCREENED_OUT)
    _node(ctx, store, "p2", status=NodeStatus.SCREENING)
    store.set_state(LG.K_LEGS, {"seq": 4, "current": {
        "batch": 4, "carrier": "C", "phase": "run", "legs": [{"id": "x"}] * 6,
        "designs": {"legs": {"points": {"0": "C", "1": "p1", "2": "p2"}}}}, "batches": []})
    assert LG.hb(store) == "legs=L4(run on C: 6 legs, 1/2 runs)"
    assert " legs=L4(run on C: 6 legs, 1/2 runs) " in hb_line(_on(ctx), store, "poll")


def test_the_winner_cross_pool_takes_promising_nodes_crossed_with_the_head_only(ctx, store):
    from loop.select import winner_pairs
    ctx = _on(ctx)
    snap = _world(ctx, store)
    snap.banked = ()
    store.set_state("config_diff:x1", {"reg.sigreg_w": 0.1})
    store.set_state("config_diff:x2", {"aug.rot90": 1})
    assert winner_pairs(store, snap) == []                                     # C contains P: no confirmed pair
    got = winner_pairs(store, snap, pool=["x1", "x2"])
    assert got and all("C" in pr for pr in got) and {x for pr in got for x in pr} == {"C", "x1", "x2"}


def test_a_branch_committed_with_its_smoke_values_still_grafts_every_leg_at_the_contract_levels(ctx, store, sim):
    """live L3 (branch n1434): the researcher's time limit fired before it reset its smoke values -- new flags at 1, a
    carrier multiplier at 2.0 -- and every new leg read 'no key'. The levels come from the contract (flag 0/1,
    multiplier 1) and the branch's base is reset to the carrier's corner, so that row is still the carrier's control."""
    ctx = _on(ctx)
    snap, commit = _sim_world(ctx, store, sim)
    ids = iter(f"n{i:04d}" for i in range(100, 400))
    new_id = lambda: next(ids)                                                   # noqa: E731
    LG.tick(ctx, store, sim, snap, new_id=new_id, now=1.0)
    (req,) = LG.requests(ctx, store, snap, limit=2)
    sha, cfg = commit(store.get_node("C").commit, "br", {"loss.leg_x1": 1, "aug.leg_x2": 1, "loss.leg_b": 2.0})
    br = Node(id="n0050", task_id=ctx.task_id, commit=sha, parent_id="C", strategy=req.strategy, operator=req.operator,
              family="objective", mechanism_tag="leg_branch_L1", diff_hash=diff_hash(sha), status=NodeStatus.QUEUED,
              notes=f"request={req.id} brief=legs identical=1")
    store.put_node(LG.adopt(store, req, br))
    store.set_state("config_diff:n0050", cfg)
    acts = LG.tick(ctx, store, sim, snap, new_id=new_id, now=2.0, inflight=[])
    cur = LG.book(store)["current"]
    assert cur["phase"] == "run" and cur["missing"] == [], acts
    leg = {f["leg"] or f["key"]: (f["low"], f["high"]) for f in cur["designs"]["legs"]["factors"]}
    assert leg == {"x1": (0, 1), "x2": (0, 1), "a": (0.0, 0.5), "b": (0.0, 1.0), "train.lr": (0.002, 0.0025)}
    assert cur["reset"] == {"loss.leg_x1": 0, "aug.leg_x2": 0, "loss.leg_b": 1.0} and "reset" in acts[-1]
    pts = cur["designs"]["legs"]["points"]
    assert list(pts.values()).count("C") == 1 and len(pts) == 16              # the corner row IS the carrier
    dose = {f["leg"] or f["key"]: (f["low"], f["high"]) for f in cur["designs"]["dose"]["factors"]}
    assert dose["b"] == (0.5, 2.0)                                              # around 1, not around the smoke's 2.0
