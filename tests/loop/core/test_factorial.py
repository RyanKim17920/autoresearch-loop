"""loop.core.factorial: the design, main-effects estimation, x*, range extension, knob sources, building points."""

from __future__ import annotations

import itertools
import json
import math
import random
import statistics

from pathlib import Path

import pytest
import yaml

from loop import tick as TK
from loop.adapters.sim import SimAdapter
from loop.core import data as CD
from loop.core import factorial as F
from loop.core import model as M
from loop.core.data import Node

SIGMA, RHO = 0.0032, 0.45                       # the fitted composite noise (design §8)
NOISE = M.Noise({M.COMPOSITE: SIGMA ** 2}, {M.COMPOSITE: RHO})


def _node(nid: str, carrier: str = "c") -> Node:
    return Node(nid, carrier, None, "explore", "f", "explore", carrier, 0.0, "", "")


# -- design ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("k", range(1, 8))
def test_design_sizes_blocks_and_resolution_iv(k):
    rows, blocks = F.matrix(k)
    # k < 3: each replicate of the 2^k rows is its own block (v3-c7a A2); else blocks of 8
    assert len(rows) == (8 if k <= 4 else 16) and len(set(blocks)) == (8 // 2 ** k if k < 3 else len(rows) // 8)
    assert F.n_blocks(k) == len(set(blocks))
    assert len({(r, b) for r, b in zip(rows, blocks)}) == len(rows)                 # no row twice in one block
    assert rows[0] == (-1,) * k
    cols = list(zip(*rows))
    for j, c in enumerate(cols):
        assert sum(c) == 0                                                          # balanced
        for b in set(blocks):                                                       # orthogonal to the block
            assert sum(x for x, bl in zip(c, blocks) if bl == b) == 0
        for a, b in itertools.combinations([i for i in range(k) if i != j], 2):     # never aliased with a 2FI
            assert abs(sum(x * y * z for x, y, z in zip(c, cols[a], cols[b]))) < len(rows)
        for i in range(j):
            assert sum(x * y for x, y in zip(c, cols[i])) == 0 or k < 3             # mains orthogonal (k<3: replicas)
    with pytest.raises(ValueError):
        F.matrix(8)


def test_design_points_take_the_knob_levels_and_one_seed_per_block():
    knobs = [F.Knob(f"k{j}", 1.0, 2.0) for j in range(5)]
    w = F.design("w1", "c", knobs, [501, 502])
    assert len(w.points) == 16 and w.seeds == (501, 502)
    assert w.points[0] == {f"k{j}": 1.0 for j in range(5)}
    assert w.rows() == F.matrix(5)[0]
    assert F.from_json(F.to_json(w)) == w
    with pytest.raises(ValueError):
        F.design("w2", "c", knobs, [501])


# -- planted effects ------------------------------------------------------------------------------------------------
def simulate(rng: random.Random, wave: F.Wave, beta: list[float], *, sigma=SIGMA, rho=RHO, base=0.67,
             missing: int = 0) -> dict[int, float]:
    """y = base + Σ β_j x_j / 2 + b[block seed] + e (the seed effect shared by the block)."""
    b = {s: rng.gauss(0, math.sqrt(rho) * sigma) for s in set(wave.blocks)}
    rows = wave.rows()
    ys = {i: base + sum(bj * x / 2 for bj, x in zip(beta, rows[i])) + b[wave.blocks[i]]
          + rng.gauss(0, math.sqrt(1 - rho) * sigma) for i in range(len(rows))}
    for i in rng.sample(sorted(ys), missing):
        del ys[i]
    return ys


@pytest.mark.parametrize("k,missing", [(4, 0), (7, 0), (6, 2)])
def test_recovers_planted_main_effects_with_calibrated_se(k, missing):
    rng = random.Random(k)
    beta = [0.006, -0.004, 0.0, 0.002, -0.001, 0.003, 0.0][:k]
    wave = F.design("w", "c", [F.Knob(f"k{j}", 0, 1) for j in range(k)], [1, 2])
    zs, est = [], [[] for _ in range(k)]
    for _ in range(400):
        eff = F.estimate(wave, simulate(rng, wave, beta, missing=missing), NOISE, tau2=1.0, nu=None, min_df=99)
        for j, e in enumerate(eff):
            est[j].append(e.beta)
            zs.append((e.beta - beta[j]) / e.se)
    for j in range(k):
        assert statistics.mean(est[j]) == pytest.approx(beta[j], abs=0.0006)       # unbiased
    assert statistics.pstdev(zs) == pytest.approx(1.0, abs=0.1)                     # se from the model's noise
    if not missing:
        n = len(wave.points)
        assert eff[0].se == pytest.approx(math.sqrt(4 * (1 - RHO) * SIGMA ** 2 / n))


def test_se_is_the_larger_of_the_model_and_the_waves_own_residuals_given_4_dfs():
    knobs = [F.Knob(f"k{j}", 0, 1) for j in range(5)]
    wave = F.design("w", "c", knobs, [1, 2])                         # 16 points, 2 blocks, 5 mains: 9 resid dfs
    model_se = math.sqrt(4 * (1 - RHO) * SIGMA ** 2 / 16)
    noisy = simulate(random.Random(1), wave, [0.0] * 5, sigma=4 * SIGMA)
    quiet = simulate(random.Random(1), wave, [0.0] * 5, sigma=SIGMA / 10)
    _, resid, df = F.main_effects(wave.rows(), wave.blocks, noisy)
    assert df == 9
    assert F.estimate(wave, noisy, NOISE, tau2=1.0)[0].se == pytest.approx(math.sqrt(4 * resid / 16))
    assert F.estimate(wave, noisy, NOISE, tau2=1.0)[0].se > 2 * model_se
    assert F.estimate(wave, quiet, NOISE, tau2=1.0)[0].se == pytest.approx(model_se)      # never below the model
    small = F.design("s", "c", knobs[:4], [1])                       # 8 points, 1 block, 4 mains: 3 dfs
    ys = simulate(random.Random(2), small, [0.0] * 4, sigma=4 * SIGMA)
    assert F.estimate(small, ys, NOISE, tau2=1.0)[0].se == pytest.approx(math.sqrt(4 * (1 - RHO) * SIGMA ** 2 / 8))


def test_shrinkage_pulls_small_effects_to_zero_and_keeps_large_ones():
    wave = F.design("w", "c", [F.Knob(f"k{j}", 0, 1) for j in range(4)], [1])
    ys = simulate(random.Random(0), wave, [0.02, 0.0, 0.0, 0.0], sigma=1e-5)
    e = F.estimate(wave, ys, NOISE, tau2=0.002 ** 2)
    assert abs(e[0].shrunk) > 0.8 * abs(e[0].beta)                                  # t prior: a big effect stays
    assert all(abs(x.shrunk) <= abs(x.beta) + 1e-9 for x in e)


def test_main_effects_beat_best_cell_on_fresh_truth():
    """The winner's curse: picking the best observed cell overstates its gain and finds a worse config than x*
    from the main effects (true gain = the expected fresh-seed outcome over the carrier at the all-low corner)."""
    rng = random.Random(7)
    k, tau = 7, 0.002
    knobs = [F.Knob(f"k{j}", 0, 1, "diff") for j in range(k)]
    wave = F.design("w", "c", knobs, [1, 2])
    rows = wave.rows()
    cell_true, cell_opt, x_true, x_opt = [], [], [], []
    for _ in range(600):
        beta = [rng.gauss(0, tau) for _ in range(k)]
        truth = lambda x: sum(b * (xi + 1) / 2 for b, xi in zip(beta, x))     # noqa: E731 - gain over all-low
        ys = simulate(rng, wave, beta)
        best = max(ys, key=ys.get)
        cell_true.append(truth(rows[best]))
        cell_opt.append(ys[best] - ys[0] - truth(rows[best]))
        eff = F.estimate(wave, ys, NOISE, tau2=tau ** 2)
        nxt = F.next_point(wave, eff, {})
        x = [1 if nxt and nxt.levels.get(kb.key) == 1 else -1 for kb in knobs]
        x_true.append(truth(x))
        x_opt.append((nxt.m if nxt else 0.0) - truth(x))
    assert statistics.mean(x_true) > statistics.mean(cell_true)              # a better config on fresh seeds
    assert statistics.mean(cell_opt) > 0.002                                 # the best cell's gain is inflated
    assert abs(statistics.mean(x_opt)) < 0.4 * statistics.mean(cell_opt)     # x*'s shrunk prior is ~calibrated


# -- next step ------------------------------------------------------------------------------------------------------
def test_next_point_moves_only_clear_effects_and_sums_the_shrunk_effects():
    knobs = [F.Knob("a", 0.1, 0.3), F.Knob("b", 1e-4, 4e-4, "log"), F.Knob("n7", 0, 1, "diff"), F.Knob("d", 1, 3)]
    wave = F.design("w", "c", knobs, [9])
    eff = [F.Effect("a", 0.004, 0.001, 0.003, 1e-6), F.Effect("b", -0.003, 0.001, -0.002, 1e-6),
           F.Effect("n7", 0.005, 0.001, 0.004, 1e-6), F.Effect("d", 0.0009, 0.001, 0.0001, 1e-6)]
    carrier = {"a": 0.2, "b": 2e-4, "d": 2}                                  # a, b at the centre
    nxt = F.next_point(wave, eff, carrier)
    assert nxt.levels == {"a": 0.3, "b": 1e-4, "n7": 1}                       # d: |z| = .9 stays at the carrier
    assert nxt.m == pytest.approx(0.003 * 0.5 + 0.002 * 0.5 + 0.004)
    assert nxt.v == pytest.approx(1e-6 * (0.25 + 0.25 + 1))
    eff[2] = F.Effect("n7", -0.005, 0.001, -0.004, 1e-6)                       # the diff hurts: stays off
    assert "n7" not in F.next_point(wave, eff, carrier).levels
    assert F.next_point(wave, [F.Effect(e.key, 0.0, 0.001, 0.0, 1e-6) for e in eff], carrier) is None


def test_a_significant_effect_moves_the_range_one_step_out():
    sig, weak = F.Effect("x", 0.004, 0.001, 0.003, 1e-6), F.Effect("x", 0.001, 0.001, 0.0, 1e-6)
    neg = F.Effect("x", -0.004, 0.001, -0.003, 1e-6)
    assert F.extend(F.Knob("x", 0.1, 0.3), sig) == F.Knob("x", 0.3, pytest.approx(0.5))
    assert F.extend(F.Knob("x", 0.1, 0.3), neg) == F.Knob("x", pytest.approx(-0.1), 0.1)
    assert F.extend(F.Knob("x", 0.1, 0.3), neg, (0.05, 1.0)) == F.Knob("x", 0.05, 0.1)
    assert F.extend(F.Knob("x", 1e-4, 2e-4, "log"), sig) == F.Knob("x", 2e-4, pytest.approx(4e-4), "log")
    assert F.extend(F.Knob("x", 2, 4), sig) == F.Knob("x", 4, 6)
    assert F.extend(F.Knob("x", 0.1, 0.3), weak) == F.Knob("x", 0.1, 0.3)
    assert F.extend(F.Knob("x", 0.1, 0.3), sig, (0.0, 0.3)) is None            # already at the bound
    assert F.extend(F.Knob("n1", 0, 1, "diff"), sig) == F.Knob("n1", 0, 1, "diff")
    wave = F.design("w", "c", [F.Knob("x", 0.1, 0.3), F.Knob("y", 1, 2)], [1, 2])
    assert F.next_knobs(wave, [sig, F.Effect("y", 0.0, 1.0, 0.0, 1.0)]) == (F.Knob("x", 0.3, pytest.approx(0.5)),
                                                                           F.Knob("y", 1, 2))


# -- knob sources ---------------------------------------------------------------------------------------------------
def test_knob_sources_declared_flagged_and_stacking():
    got = F.declared([{"key": "dino.lr", "lo": 1e-4, "hi": 2e-4, "scale": "log"}, {"key": "x", "lo": 1, "hi": 1},
                      {"key": "y", "lo": -1, "hi": 1, "scale": "log"}, {"lo": 1, "hi": 2},
                      {"key": "z", "lo": 0, "hi": 1, "scale": "diff"}, {"key": "w", "lo": 0, "hi": 2}])
    assert got == [F.Knob("dino.lr", 1e-4, 2e-4, "log"), F.Knob("w", 0, 2)]
    reg = {"lr_throttled": [{"key": "dino.lr", "lo": 2e-4, "hi": 4e-4, "scale": "log"}]}
    assert F.flagged(["lr_throttled", "clip_dominated"], reg) == [F.Knob("dino.lr", 2e-4, 4e-4, "log")]
    nodes = {n: _node(n, c) for n, c in (("n1", "c"), ("n2", "c"), ("n3", "other"), ("n4", "c"))}
    assert F.stacking("c", nodes, {"n1": 0.001, "n2": 0.003, "n3": 0.01, "n4": -0.001}) == \
        [F.Knob("n2", 0, 1, "diff"), F.Knob("n1", 0, 1, "diff")]
    picked = F.pick(got, F.flagged(["lr_throttled"], reg), F.stacking("c", nodes, {"n1": 1.0}), k=2)
    assert picked == (F.Knob("dino.lr", 1e-4, 2e-4, "log"), F.Knob("w", 0, 2))       # first per key, capped


def test_the_researcher_finish_declares_knobs_and_its_record_keeps_the_valid_ones(ctx):
    from types import SimpleNamespace
    from loop.agents import researcher as RS
    assert RS.finish_schema(ctx)["properties"]["knobs"]["items"] == F.KNOB_SCHEMA
    fin = {"hypothesis": {}, "knobs": [{"key": "train.lr", "lo": 5e-4, "hi": 2e-3, "scale": "log"},
                                       {"key": "aug.strength", "lo": 0.5, "hi": 0.5}]}
    rec = RS.LLMResearcher._plan(SimpleNamespace(ctx=ctx), None, fin, SimpleNamespace(chosen=None), None)
    assert rec["knobs"] == [{"key": "train.lr", "lo": 5e-4, "hi": 2e-3, "scale": "log"}]
    assert F.declared(rec["knobs"]) == [F.Knob("train.lr", 5e-4, 2e-3, "log")]


def test_the_nanopath_task_registers_knobs_for_the_optimiser_flags(tmp_path):
    from loop.spec import load_context
    repo = Path(__file__).resolve().parents[3]
    reg = load_context(repo / "tasks" / "nanopath_v2.yaml", state_dir=tmp_path / "s",
                       require_measured=False).cfg.factorial.flag_knobs
    assert [kb.key for kb in F.flagged(["lr_throttled"], reg)] == ["dino.layerwise_decay", "dino.patch_embed_lr_mult"]
    assert F.flagged(["update_starved"], reg) == [F.Knob("dino.lr", 2.344e-4, 3.5e-4, "log")]


def test_factorial_tau2_reads_the_factorial_cell_of_the_carrier():
    pts = [M.PriorPoint(("c", "design", "factorial"), d, 1e-6) for d in (0.001, -0.002, 0.0, 0.003)]
    pts += [M.PriorPoint(("c", "x", "explore"), d, 1e-6) for d in (0.01, -0.01, 0.02)]
    prior = M.fit_prior(pts)
    assert F.factorial_tau2(prior, "c") == prior.tau2[("c", "design", "factorial")]
    assert F.factorial_tau2(prior, "unseen") == prior.tau2[()]


def test_node_prior_parses_an_x_star_node():
    assert F.node_prior("factorial_of=w1 prior_m=0.0021 prior_v=4e-07 design a=1") == (0.0021, 4e-07)
    assert F.node_prior("design a=1") is None


# -- building points on the sim adapter -----------------------------------------------------------------------------
def _diff_node(sim, store, nid: str, key: str, value):
    """A node on root whose diff sets ``key`` (a research session's commit): (its parent commit, its commit)."""
    from loop.contracts import Node as StoreNode, NodeStatus, Strategy
    wt = sim.prepare_worktree(parent_commit=sim.root_commit(), name=nid)
    p = Path(wt.path) / "config.yaml"
    cfg = yaml.safe_load(p.read_text())
    sec, leaf = key.split(".")
    cfg[sec][leaf] = value
    p.write_text(yaml.safe_dump(cfg))
    sha = sim.commit(wt, message=nid)
    store.put_node(StoreNode(id=nid, task_id=store.task_id, commit=sha, parent_id="root", strategy=Strategy.EXPLORE,
                             operator=None, family="f", mechanism_tag=nid, diff_hash=nid,
                             status=NodeStatus.SCREENING, created_ts=1.0))
    return sim.root_commit(), sha


def _config(sim, sha):
    return yaml.safe_load(sim._object(sha)["files"]["config.yaml"])


def test_materialize_builds_every_point_with_its_config_and_grafted_diffs(ctx, store, tmp_path):
    sim = SimAdapter(ctx)
    store.ensure_root(sim.root_commit())
    diffs = {"n1": _diff_node(sim, store, "n1", "train.weight_decay", 0.01)}
    wave = F.design("w1", "root", [F.Knob("aug.strength", 0.3, 0.7), F.Knob("n1", 0, 1, "diff")], [41, 42])
    ids = iter(f"p{i}" for i in range(100))
    made, fails = F.materialize(store, sim, wave, diffs, "config.yaml", lambda: next(ids), now=2.0)
    assert not fails and len(made) == 8
    for i, nid in enumerate(made):
        node, pt = store.get_node(nid), wave.points[i]
        cfg = _config(sim, node.commit)
        assert node.parent_id == "root" and node.notes.startswith(
            f"wave=w1 point={i} block={i % 2} seed={41 + i % 2}")
        assert cfg["aug"]["strength"] == pt["aug.strength"]
        assert cfg["train"]["weight_decay"] == (0.01 if pt["n1"] == 1 else 0.05)
    assert F.from_json(store.get_state("wave:w1")) == wave
    data = CD.load(tmp_path / "loop.db", ctx.task_id)
    assert data.nodes["p0"].lane == "factorial"
    assert F.wave_runs("w1", data) == [(nid, 41 + i % 2) for i, nid in enumerate(made)]
    specs = TK.ModuleCore().runs_for(TK.Action("wave", "w1", "factorial", 8.0, 1.0), data, [], {}, ctx.cfg)
    assert specs == [TK.RunSpec(nid, TK.SEED_STAGE, 41 + i % 2) for i, nid in enumerate(made)] + [
        TK.RunSpec("root", TK.Stage.PARENT, s) for s in (41, 42)]                 # the tick's wave hook + controls
    nxt = F.Next({"aug.strength": 0.7, "n1": 1}, 0.002, 1e-6)
    xid, why = F.materialize_next(store, sim, wave, nxt, diffs, "config.yaml", lambda: "x1", now=3.0)
    assert not why and F.node_prior(store.get_node(xid).notes) == (0.002, 1e-6)
    assert _config(sim, store.get_node(xid).commit)["train"]["weight_decay"] == 0.01


def test_a_conflicting_diff_pair_leaves_the_point_out(ctx, store):
    sim = SimAdapter(ctx)
    store.ensure_root(sim.root_commit())
    diffs = {"n1": _diff_node(sim, store, "n1", "train.weight_decay", 0.01),
             "n2": _diff_node(sim, store, "n2", "train.weight_decay", 0.09)}
    knobs = [F.Knob("n1", 0, 1, "diff"), F.Knob("n2", 0, 1, "diff")]
    sha, why = F.build_point(sim, sim.root_commit(), {"n1": 1, "n2": 1}, knobs, diffs, "x", "config.yaml")
    assert sha == "" and why.startswith("n2: conflict")
    assert F.build_point(sim, sim.root_commit(), {"n1": 0, "n2": 0}, knobs, diffs, "y", "config.yaml") == \
        (sim.root_commit(), "")


def test_a_one_knob_wave_has_no_byte_identical_replicates_and_controls_pair_every_point():
    """v3-c7a A2: w013 (one knob) had 4 identical configs per level at ONE seed. Now each replicate is a block with its
    own seed (no (levels, seed) twice), duplicates are detected generally, and the carrier gets one control per block."""
    w = F.design("w", "c", [F.Knob("x", 0.1, 1.0)], [11, 12, 13, 14])
    assert len(w.points) == 8 and sorted(set(w.blocks)) == [0, 1, 2, 3]
    keys = {(json.dumps(p), w.blocks[i]) for i, p in enumerate(w.points)}
    assert len(keys) == 8                                           # every (config, seed) once
    assert F.duplicates([{"x": 1}, {"x": 1}, {"x": 1}, {"x": 2}], [0, 0, 1, 0]) == [1]
    old = F.design("w", "c", [F.Knob("x", 0.1, 1.0)], [11])          # O1: a pre-A2 journal's one seed -> the
    assert len(old.points) == 8 and set(old.blocks) == {0}              # old single-block layout, never a raise
    from loop.core.data import Ledger, Node, Run

    def node(i, notes="", parent="c"):
        return Node(i, parent, None, "design", "design", "factorial", "root", 0.0, "", notes)
    nodes = {"c": node("c", parent=None), **{f"w.p{i}": node(f"w.p{i}", F.point_tag(w, i)) for i in range(8)}}
    runs = (Run("c:screen:11", "c", 11, "screen", "done", 1.0, 0.0, 0.0, "", 0),)
    led = Ledger(nodes, runs, (), frozenset(), ())
    assert F.wave_controls(led, "w") == [("c", 12), ("c", 13), ("c", 14)]     # seed 11: the carrier ran there
