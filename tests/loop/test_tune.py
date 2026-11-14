"""MOONSHOT TUNING (loop/tune.py): the trigger band and the veto rule, the budget classes, one pass per node,
trials built on the moonshot's commit and paired against it, the adaptive phase and its early stop, the tuned
node's fresh-seed verdict, the retro path after a restart, credit and the safety cap.
Run: uv run pytest tests/loop -q --confcutdir=tests/loop"""

from __future__ import annotations

import dataclasses
import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from loop import tune as MT
from loop.agents.researcher import diff_hash, set_yaml_value
from loop.contracts import (DESIGN_FAMILY, PANEL, ROOT_NODE_ID, Candidate, HandoffNote, HypothesisKind,
                            HypothesisRecord, Node, NodePosterior, NodeStatus, Operator, Posterior, Postmortem,
                            ResearchResult, Stage, Strategy, design_point)
from loop.spec import MoonshotTuneCfg, ResearcherCfg, ReviewCfg, load_context
from tests.loop.conftest import InlineExecutor
from tests.loop.test_driver import FAKE, Scripts

AXES = ("alpha", "beta", "gamma")
AUG = {"aug.strength": 0.8}          # a near-null moonshot: stronger augmentation (sim: -0.0017 panel)


def _post(nid: str, p: float, *, veto: tuple[str, ...] = (), axis_mean: float = 0.0) -> NodePosterior:
    """P(gain over the parent > 0) = p (the increment); P(U>0) vs root is ~1, as for a child of an incumbent."""
    from statistics import NormalDist
    inc = Posterior(0.003 * NormalDist().inv_cdf(min(max(p, 1e-6), 1 - 1e-6)), 0.003, 2)
    return NodePosterior(node_id=nid, increment=inc, utility=Posterior(0.01, 0.003, 2),
                         axes={a: Posterior(axis_mean if a in veto else 0.0, 0.003, 2) for a in AXES},
                         p_positive=0.999, p_beats_incumbent=p, vetoed=bool(veto), veto_axes=veto)


def _snap(posts: dict) -> SimpleNamespace:
    return SimpleNamespace(nodes=posts, sigma={PANEL: 0.004, "alpha": 0.005, "beta": 0.006, "gamma": 0.007})


def _with(ctx, **tune):        # the fresh-confirm-seed gate off unless a test sets it (loop.tune.gate)
    return dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, moonshot_tune=MoonshotTuneCfg(
        **{"min_confirm_sigmas": 0.0, **tune})))


def _moonshot(ctx, store, adapter, nid: str, *, status=NodeStatus.SCREENED_OUT, cfg=None, code=True,
              strategy=Strategy.MOONSHOT, notes="brief=moonshot backend=x", ts=1.0) -> Node:
    """A structural node on top of root: config changes plus (``code``) a new function in aug.py."""
    root = store.get_node(ROOT_NODE_ID) or store.ensure_root(adapter.root_commit())
    wt = adapter.prepare_worktree(parent_commit=root.commit, name=nid)
    text = (Path(wt.path) / "config.yaml").read_text()
    for k, v in (cfg if cfg is not None else {"train.lr": 0.002, "loss.temperature": 0.2}).items():
        text = set_yaml_value(text, k, v)
    (Path(wt.path) / "config.yaml").write_text(text)
    if code:
        aug = Path(wt.path) / "aug.py"
        aug.write_text(aug.read_text() + "\n\ndef masked_view(x):\n    return x\n")
    diff_text, _, cfg_diff = adapter.diff(wt)
    commit = adapter.commit(wt, message=nid)
    adapter.reap_worktree(wt)
    p = Path(ctx.state_dir) / "diffs" / f"{nid}.diff"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(diff_text)
    node = Node(id=nid, task_id=ctx.task_id, commit=commit, parent_id=ROOT_NODE_ID, strategy=strategy,
                operator=Operator.DRAFT, family="augmentation", mechanism_tag="jepa_masked_view",
                diff_hash=diff_hash(diff_text), status=status, diff_path=str(p), notes=notes,
                created_ts=ts, updated_ts=ts)
    store.put_node(node)
    store.set_state(f"config_diff:{nid}", dict(cfg_diff))
    return node


# ------------------------------------------------------------------------------ pure rules
@pytest.fixture
def sim(ctx, store):
    from loop.adapters.sim import SimAdapter
    return SimAdapter(ctx)


def test_trigger_band_statuses_and_structural_only(ctx, store, sim):
    ctx = _with(ctx)
    m = _moonshot(ctx, store, sim, "n0001", code=False)
    cls = lambda p, **kw: MT.classify(ctx, store, _snap({m.id: _post(m.id, p, **kw)}), store.get_node(m.id))  # noqa: E731
    assert cls(0.19) is None and cls(0.2) and cls(0.5) and cls(0.85) and cls(0.86) is None
    store.put_postmortem(Postmortem("pm-1", ctx.task_id, m.id, "loss", "e", "m", m.levers, under_tuned=True,
                                    tune_knobs=json.dumps([{"key": "loss.temperature", "low": 0.2, "high": 0.3}])))
    assert "under-tuned" in cls(0.05)["reason"]                      # the postmortem overrides the band
    store.update_node(m.id, status=NodeStatus.NOT_CONFIRMED)
    assert cls(0.9) is not None
    store.update_node(m.id, status=NodeStatus.CRASHED)
    assert cls(0.5) is None
    plain = _moonshot(ctx, store, sim, "n0002", strategy=Strategy.EXPLORE, notes="brief=literature", code=False)
    bank = _moonshot(ctx, store, sim, "n0003", strategy=Strategy.EXPLORE, notes="brief=bank bank=v1-jepa-x", code=False)
    snap = _snap({n.id: _post(n.id, 0.5) for n in (plain, bank)})
    assert MT.classify(ctx, store, snap, plain) is None and MT.classify(ctx, store, snap, bank) is not None


def test_vetoed_only_when_borderline_clearly_terrible_stays_dead(ctx, store, sim):
    ctx = _with(ctx)
    m = _moonshot(ctx, store, sim, "n0001", status=NodeStatus.VETOED, code=False)
    cls = lambda post: MT.classify(ctx, store, _snap({m.id: post}), store.get_node(m.id))  # noqa: E731
    assert cls(_post(m.id, 0.45, veto=("beta",), axis_mean=-0.009)) is not None   # -1.5 sigma on beta, one axis
    assert cls(_post(m.id, 0.25, veto=("beta",), axis_mean=-0.004)) is None        # panel P < 0.3
    assert cls(_post(m.id, 0.45, veto=("beta",), axis_mean=-0.013)) is None        # harm > 2 sigma
    assert cls(_post(m.id, 0.45, veto=("beta", "gamma"), axis_mean=-0.004)) is None   # two axes
    store.put_postmortem(Postmortem("pm-1", ctx.task_id, m.id, "loss", "e", "m", m.levers, under_tuned=True))
    assert cls(_post(m.id, 0.1, veto=("beta",), axis_mean=-0.03)) is None           # under_tuned cannot revive it
    snap = _snap({m.id: _post(m.id, 0.1, veto=("beta",), axis_mean=-0.03)})
    assert MT.tick(ctx, store, sim, snap, new_id=lambda: "nX", now=10.0) == [] and not MT.book(store)["tuned"]


def test_a_major_change_gets_the_major_budget_a_numbers_only_one_the_minor(ctx, store, sim):
    """``major`` is computed from the diff (it was hard-coded False, so the 5-trial budget never ran): a code change,
    an init swap or an objective / architecture mechanism is MAJOR and gets the pass as well as its ladder."""
    ctx = _with(ctx)
    major = _moonshot(ctx, store, sim, "n0001")
    minor = _moonshot(ctx, store, sim, "n0002", cfg={"train.lr": 0.0015}, code=False)
    swap = _moonshot(ctx, store, sim, "n0003", cfg={"train.lr": 0.0015}, code=False,
                     notes="brief=moonshot init=timm/vit_small mtype=optimization")
    snap = _snap({n.id: _post(n.id, 0.4) for n in (major, minor, swap)})
    got = MT.classify(ctx, store, snap, major)
    assert got["major"] and got["rough"] and got["budget"] == 5 and MT.is_major(ctx, store, major)
    assert MT.classify(ctx, store, snap, minor)["budget"] == 2 and not MT.classify(ctx, store, snap, minor)["major"]
    assert MT.classify(ctx, store, snap, swap)["major"] and not MT.is_major(ctx, store, swap)
    gated = _with(ctx, min_confirm_sigmas=1.0)                        # the fresh-confirm-seed gate: numbers-only
    assert MT.gate(gated, store, snap, major) == "" and MT.gate(gated, store, snap, swap) == ""
    assert "own confirm increment" in MT.gate(gated, store, snap, minor)
    store.update_node(major.id, notes=major.notes + " tuning=t1")       # laddered: its pass waits for the ladder
    store.set_state(MT.K_TUNE, {"seq": 1, "current": {"1": {"batch": 1, "base": major.id, "kind": "ladder"}}})
    assert major.id not in {n.id for n, _ in MT.eligible(ctx, store, snap, 1e12)}
    store.set_state(MT.K_TUNE, {"seq": 1, "current": {}, "laddered": [major.id],
                                "batches": [{"batch": 1, "base": major.id, "kind": "ladder", "closed": 1.0}]})
    assert major.id in {n.id for n, _ in MT.eligible(ctx, store, snap, 1e12)}
    assert MT.diff_stats("--- /dev/null\n+++ b/new.py\n+x = 1\n")["new_files"] == 1
    assert [len(MT.rows_for(k, b)) - 1 for k, b in ((1, 5), (2, 5), (3, 5), (2, 2), (3, 2), (2, 1))] == [1, 3, 3, 2, 2, 1]


# ----------------------------------------------------------------------- the pass, end to end
class TuneResearcher:
    """Proposes each tuning variant (a train.lr value on top of the moonshot); every other request fails."""

    def __init__(self, adapter, values) -> None:
        self.adapter, self.values, self.calls = adapter, list(values), []

    def research(self, req, store, snapshot):
        if req.brief_kind != "tune":
            return ResearchResult("failed", None, "not scripted")
        self.calls.append(req)
        base = store.get_node(req.parent_ids[0])
        wt = self.adapter.prepare_worktree(parent_commit=base.commit, name=req.id)
        cfg = Path(wt.path) / "config.yaml"
        cfg.write_text(set_yaml_value(cfg.read_text(), "train.lr", self.values.pop(0)))
        diff_text, paths, cfg_diff = self.adapter.diff(wt)
        commit = self.adapter.commit(wt, message=req.id)
        h = HypothesisRecord(f"h-{req.id}", base.task_id, HypothesisKind.TRIAL, "retuned lr", "lr_tune",
                             "optimization", {}, "no gain")
        return ResearchResult("ok", Candidate(req.id, base.task_id, commit, base.commit, diff_text, diff_hash(diff_text),
                                              tuple(paths), dict(cfg_diff), h,
                                              HandoffNote("", base.task_id, req.id, "set lr", "", "")))


def _driver(tmp_path, values=(), *, slots=40, p_promote=None, **tune):
    from loop.__main__ import build
    from loop.adapters.sim import SimAdapter
    ctx = load_context(FAKE, state_dir=tmp_path / "state")
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(
        ctx.cfg, slots=slots, researcher=ResearcherCfg(max_concurrent=1),
        review=ReviewCfg(stall_evals=1000, fp_every_reviews=0,
                         fp_stall_closes=0),
        moonshot_tune=MoonshotTuneCfg(**{"p_lo": 0.0, "p_hi": 1.0, "veto_min_p": 0.0, "veto_max_sigmas": 100.0,
                                         "rough_min_p": 0.0,
                                         "shared": {"train.warmup": 1.5}, "min_confirm_sigmas": 0.0, **tune})))
    if p_promote is not None:
        ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, promotion=dataclasses.replace(
            ctx.cfg.promotion, p_promote=p_promote)))
    adapter = SimAdapter(ctx)
    adapter.lever_effects["train.lr"] = -0.03          # the variants' lever: a clear effect
    driver, store = build(ctx, llm=Scripts().client(), adapter=adapter, executor=InlineExecutor,
                          rng=random.Random(7))
    driver.sleep = lambda s: None
    driver.researcher = TuneResearcher(adapter, values)
    driver.selector.next_request = lambda store, snap: None           # no unrelated ideation
    return ctx, driver, store, adapter


def _run_until(driver, store, cond, n=120):
    for _ in range(n):
        driver.tick()
        if cond():
            return True
    return False


def test_a_pass_runs_on_the_moonshot_then_its_best_variant_confirms_on_fresh_seeds(tmp_path):
    ctx, driver, store, adapter = _driver(tmp_path, values=(0.00095, 0.0006), max_trials_minor=5)
    driver.tick()
    m = _moonshot(ctx, store, adapter, "n0001", status=NodeStatus.SCREENING, cfg=AUG, code=False)
    assert _run_until(driver, store, lambda: bool(MT.book(store)["batches"]))
    (res,) = MT.book(store)["batches"]
    assert res["base"] == m.id and res["budget"] == 5 and res["phase"] == 2
    trials = [store.get_node(i) for i in res["trials"]]
    assert len(trials) == 5 and [kb["key"] for kb in res["knobs"]] == ["aug.strength", "train.warmup"]
    assert all(design_point(t) and t.parent_id == m.id and t.family == DESIGN_FAMILY for t in trials)
    for t in trials:                                                # every variant is built ON the moonshot
        assert m.commit in adapter._ancestors(t.commit)
    assert len(driver.researcher.calls) == 2 and all(r.brief_kind == "tune" for r in driver.researcher.calls)
    assert "Phase-1 design" in driver.researcher.calls[0].brief and "clears the promote bar" in res["why"]
    seeds = set(res["seeds"])                                       # ONE shared block, paired vs the moonshot
    assert {r.seed for t in trials for r in store.runs(node_id=t.id)} == seeds and len(seeds) == len(ctx.cfg.seeds.screen)
    assert {r.seed for r in store.runs(node_id=m.id, stage=Stage.PARENT)} >= seeds
    assert "tuning=t1" in store.get_node(m.id).notes
    tuned = store.get_node(res["tuned"])
    assert tuned.strategy is Strategy.MOONSHOT and f"tune_of={m.id}" in tuned.notes
    assert tuned.commit == store.get_node(res["best"]).commit and tuned.parent_id == m.parent_id
    assert _run_until(driver, store, lambda: store.get_node(tuned.id).status not in (
        NodeStatus.SCREENING, NodeStatus.PROMOTING, NodeStatus.CONFIRMING), n=150)
    runs = store.runs(node_id=tuned.id)
    assert runs and not {r.seed for r in runs} & seeds             # screen, promote, confirm: all fresh
    assert not {r.seed for r in store.runs(node_id=tuned.id, stage=Stage.CONFIRM)} & seeds
    plan = ctx.cfg.seeds
    assert not seeds & {*plan.promote, *plan.confirm_pool, *plan.audit}
    pull = next(p for p in store.pulls() if p.node_id == tuned.id)  # the pass's GPU-h is the moonshot arm's
    own = sum(r.gpu_h for r in runs)
    assert pull.arm is Strategy.MOONSHOT and pull.gpu_h == pytest.approx(own + res["gpu_h"]) and res["gpu_h"] > 0
    from loop.quality import quality_groups
    arms = {g for t, g, _ in quality_groups(store, driver.snapshot) if t == "arm"}
    assert "moonshot_tuned" in arms and "moonshot" in arms
    assert not MT.eligible(ctx, store, driver.snapshot, 1e12)         # once per moonshot
    assert MT.card(store, m.id).startswith("tuning pass t1") and "t1" in MT.lines(store)


def test_two_consecutive_worse_trials_stop_early_and_charge_the_arm(tmp_path):
    ctx, driver, store, adapter = _driver(tmp_path, values=(0.004, 0.005, 0.006), max_trials_minor=6,
                                          p_promote=0.999)       # lr up: worse; no trial clears the bar
    driver.tick()
    m = _moonshot(ctx, store, adapter, "n0001", status=NodeStatus.SCREENING, cfg=AUG, code=False)
    assert _run_until(driver, store, lambda: bool(MT.book(store)["batches"]))
    (res,) = MT.book(store)["batches"]
    assert "two consecutive" in res["why"] and len(res["trials"]) == 5 < res["budget"] == 6
    assert len(driver.researcher.calls) == 2
    if not res["tuned"]:                                            # no tuned node: a zero-reward moonshot pull
        pull = next(p for p in store.pulls() if p.node_id == f"{m.id}#tune")
        assert pull.arm is Strategy.MOONSHOT and pull.reward == 0.0 and pull.gpu_h == pytest.approx(res["gpu_h"])


def test_minor_budget_is_two_config_trials_and_no_researcher(tmp_path):
    ctx, driver, store, adapter = _driver(tmp_path)                    # a numbers-only node: budget 2
    driver.tick()
    _moonshot(ctx, store, adapter, "n0001", status=NodeStatus.SCREENING, cfg=AUG, code=False)
    assert _run_until(driver, store, lambda: bool(MT.book(store)["batches"]))
    (res,) = MT.book(store)["batches"]
    assert res["budget"] == 2 and len(res["trials"]) == 2 and res["rows"] == [list(r) for r in MT._OFAT2]
    assert driver.researcher.calls == []


def test_a_closed_eligible_moonshot_gets_its_pass_after_a_restart(tmp_path):
    from loop.__main__ import build
    ctx, driver, store, adapter = _driver(tmp_path, enabled=False)
    driver.tick()
    m = _moonshot(ctx, store, adapter, "n0001", status=NodeStatus.SCREENING, cfg=AUG, code=False)
    assert _run_until(driver, store, lambda: store.get_node(m.id).status not in (
        NodeStatus.SCREENING, NodeStatus.PROMOTING, NodeStatus.CONFIRMING))
    assert not MT.book(store)["tuned"]
    store.close()
    on = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, moonshot_tune=dataclasses.replace(
        ctx.cfg.moonshot_tune, enabled=True)))
    driver2, store2 = build(on, llm=Scripts().client(), adapter=adapter, executor=InlineExecutor, rng=random.Random(3))
    driver2.selector.next_request = lambda store, snap: None
    acts = driver2.tick()                                           # no manual step
    assert any(a.startswith(("retro_tune queued n0001", "tune queued n0001")) for a in acts), acts
    assert MT.book(store2)["tuned"] == [m.id] and any(e["node_id"] == m.id for e in
                                                       store2.events("retro_tune") + store2.events("tune"))


def test_tuning_trials_and_their_controls_are_preemptible_only(ctx, store, sim):
    from loop.contracts import Run, RunStatus
    from loop.driver import RunBook
    from loop.spec import PoolCfg, SlurmCfg
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, slurm=SlurmCfg(pools=(
        PoolCfg("primary", slots=2), PoolCfg("idle", slots=4, preemptible=True)))))
    rb = RunBook(ctx, store, sim)
    m = _moonshot(ctx, store, sim, "n0001")
    assert rb.pool(m, Stage.PARENT) == "primary" and rb.high(m, Stage.PARENT)
    m = store.update_node(m.id, notes=m.notes + " tuning=t1")
    assert rb.pool(m, Stage.PARENT) == "idle" and not rb.high(m, Stage.PARENT)
    pt = dataclasses.replace(m, id="n0002", strategy=Strategy.DESIGN, operator=None, status=NodeStatus.SCREENING,
                             mechanism_tag="t1pt01", parent_id=m.id, notes="tune=t1")
    assert rb.pool(pt, Stage.SCREEN) == "idle" and not rb.high(pt, Stage.SCREEN)
    store.put_node(pt)                       # n0333:parent:390: once a trial finished on seed 7 its control there
    store.put_run(Run(id=Run.make_id(pt.id, Stage.SCREEN, 7), task_id=store.task_id, node_id=pt.id, seed=7,
                      stage=Stage.SCREEN, status=RunStatus.DONE))      # blocks a verdict: HIGH, primary allowed
    assert rb.high(m, Stage.PARENT, 7) and rb.pool(m, Stage.PARENT, 7) == "primary"
    assert not rb.high(m, Stage.PARENT, 8) and rb.pool(m, Stage.PARENT, 8) == "idle"


def test_the_safety_cap_is_half_the_preemptible_slots(ctx):
    wide = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, slots=16))
    assert MT.safety_cap(wide) == 8 and MT.safety_cap(ctx) == 2      # 4 slots, no preemptible pool: half


def test_every_phase1_design_tunes_a_mechanism_dose_knob_never_lr_alone(ctx, store, sim):
    """9/20 under-tuned postmortems named the shared lr and the passes all tried lr x1.25: a postmortem's knob
    outside the node's own dose set takes the ONE shared slot, a dose knob always leads, and a node with no usable
    dose knob gets no phase-1 design at all (its researcher variants are told to tune the dose)."""
    import yaml
    from loop.adapters import flatten
    ctx = _with(ctx, shared={"train.warmup": 1.5})
    m = _moonshot(ctx, store, sim, "n0001", cfg=AUG)
    wt = sim.prepare_worktree(parent_commit=m.commit, name="look")
    text = (Path(wt.path) / "config.yaml").read_text()
    base = flatten(yaml.safe_load(text))
    pm = SimpleNamespace(tune_knobs=json.dumps([{"key": "train.lr", "high": base["train.lr"] * 1.25}]))
    knobs = MT.pick_knobs(ctx, store, m, text, base, pm)
    assert [k["key"] for k in knobs] == ["aug.strength", "train.lr"]          # dose first; lr only as the shared one
    assert "aug.strength" in MT.dose_keys(store, m, base)
    code_only = _moonshot(ctx, store, sim, "n0002", cfg={})
    assert MT.pick_knobs(ctx, store, code_only, text, base, pm) == []         # no dose knob: never an lr-only design
    # a level screened out by >= 3 earlier design / tuning points is not tried again (dino.lr=2.344e-4, 9x, then t12)
    hi = base["train.lr"] * 1.25
    for i in range(3):
        store.put_node(Node(id=f"dp{i}", task_id=store.task_id, commit=f"c-dp{i}", parent_id=m.id,
                            strategy=Strategy.DESIGN, operator=None, family="design", mechanism_tag=f"t{i} design",
                            diff_hash=f"h-dp{i}", status=NodeStatus.SCREENED_OUT, created_ts=float(i)))
        store.set_state(f"config_diff:dp{i}", {"train.lr": hi})
        got = [k["key"] for k in MT.pick_knobs(ctx, store, m, text, base, pm)]
        assert got == (["aug.strength", "train.lr"] if i < 2 else ["aug.strength", "train.warmup"])


def test_tuning_needs_a_one_sigma_own_increment_on_fresh_confirm_seeds_and_one_family_at_a_time(ctx, store, sim):
    """Live 2026-10-02: ~70 GPU-h of tuning passes made 6 tuned nodes, all screened out; selection-seed promise is the
    winner's curse. A node is tuned only when its OWN increment on its fresh CONFIRM seeds alone reaches 1 panel
    sigma, and while no other node of its family (normalized tag) is in tuning-or-confirm."""
    ctx = _with(ctx, min_confirm_sigmas=1.0)
    a, b = (_moonshot(ctx, store, sim, nid, status=NodeStatus.NOT_CONFIRMED, ts=t, code=False)
            for nid, t in (("n0001", 1.0), ("n0002", 2.0)))
    snap = _snap({a.id: _post(a.id, 0.9), b.id: _post(b.id, 0.8)})
    assert not MT.eligible(ctx, store, snap, 1e12)                          # no confirm seeds: never
    snap.own_confirm = {a.id: (0.0039, 2), b.id: (0.0045, 3)}
    assert [n.id for n, _ in MT.eligible(ctx, store, snap, 1e12)] == [b.id]  # a: 0.0039 < 1 sigma 0.004
    assert "< 1 sigma" in MT.gate(ctx, store, snap, a)
    snap.own_confirm[a.id] = (0.005, 2)
    assert [n.id for n, _ in MT.eligible(ctx, store, snap, 1e12)] == [a.id]  # one per family: the most promising
    store.update_node(b.id, mechanism_tag="other_mech")
    assert {n.id for n, _ in MT.eligible(ctx, store, snap, 1e12)} == {a.id, b.id}
    _moonshot(ctx, store, sim, "n0003", status=NodeStatus.CONFIRMING)      # its family is confirming now
    assert "already in tuning/confirm (n0003)" in MT.gate(ctx, store, snap, a)
    assert [n.id for n, _ in MT.eligible(ctx, store, snap, 1e12)] == [b.id]
    code = _moonshot(ctx, store, sim, "n0004", status=NodeStatus.NOT_CONFIRMED)
    store.update_node(code.id, mechanism_tag="third_mech")
    assert MT.gate(ctx, store, snap, store.get_node(code.id)) == ""      # a code change: no confirm-seed gate


def test_once_a_running_pass_failing_the_gate_is_withdrawn_its_pending_trials_closed_running_ones_kept(tmp_path):
    """One-time (driver.tune_gate): t13 on n0427 / t10 on n0431 tuned nodes whose fresh-seed own increment is < 1 sigma.
    The pass is withdrawn (no new trial, a zero-reward moonshot pull); a trial no run of which SLURM started is closed
    and its PENDING runs cancelled; a trial with a running run finishes; a proposal coming back is rejected."""
    from loop.contracts import Run, RunStatus, Stage
    from loop.driver import K_TUNE_GATE
    ctx, driver, store, adapter = _driver(tmp_path, min_confirm_sigmas=1.0)
    driver.tick()
    base = _moonshot(ctx, store, adapter, "n0900", status=NodeStatus.NOT_CONFIRMED, code=False)
    for nid in ("n0901", "n0902"):
        store.put_node(Node(id=nid, task_id=store.task_id, commit=base.commit, parent_id=base.id,
                            strategy=Strategy.DESIGN, operator=None, family=DESIGN_FAMILY,
                            mechanism_tag=f"t1pt0{nid[-1]}", diff_hash=nid, status=NodeStatus.SCREENING,
                            notes="tune=t1"))
        store.put_run(Run(id=f"{nid}:screen:500", task_id=store.task_id, node_id=nid, seed=500,
                          stage=Stage.SCREEN, status=RunStatus.RUNNING, job_id=f"j{nid}"))
    store.set_state(MT.K_TUNE, {"seq": 1, "tuned": [base.id], "batches": [], "current": {"1": {
        "batch": 1, "base": base.id, "budget": 5, "trials": ["n0901", "n0902"], "points": {"0": base.id},
        "pending": "tune-t1-3", "fails": 0, "phase": 2, "reason": "x", "knobs": [], "rows": [], "effects": []}}})
    driver.book.pending = {"n0902:screen:500"}                         # SLURM runs n0901's, holds n0902's
    driver.book.adapter.cancel = lambda h: None
    store.set_state(K_TUNE_GATE, False)
    from loop.contracts import K_INFLIGHT, CandidateRequest, Operator
    from loop.spec import encode
    req = CandidateRequest(id="tune-t1-3", task_id=store.task_id, strategy=Strategy.DESIGN, operator=Operator.IMPROVE,
                           island="main", parent_ids=(base.id,), brief="b", brief_kind="tune")
    store.set_state(K_INFLIGHT, [{"id": req.id, "req": encode(req), "side": True}])
    driver._research.clear()
    driver._restore_inflight(acts := [])                               # a restart: the failing pass's proposal is
    assert not driver._research and "not re-issued" in acts[-1]        # not re-issued (deploy: before the gate runs)
    driver._tune_gate(acts := [])
    bk = MT.book(store)
    assert not bk["current"] and bk["batches"][-1]["why"].startswith("withdrawn: own confirm increment")
    assert store.get_node("n0902").status is NodeStatus.SCREENED_OUT
    assert store.get_run("n0902:screen:500").void_type.value == "cancelled"
    assert store.get_node("n0901").status is NodeStatus.SCREENING
    assert store.get_run("n0901:screen:500").status is RunStatus.RUNNING
    assert any(p.node_id == f"{base.id}#tune" and p.reward == 0.0 for p in store.pulls())
    assert "withdrawn" in acts[-1] and store.get_state(K_TUNE_GATE)
    req = SimpleNamespace(brief_kind="tune", id="tune-t1-3")
    late = MT.adopt(store, req, store.get_node("n0901"))
    assert late.status is NodeStatus.REJECTED and late.reject_reason == "tune_pass_closed"


# ------------------------------------------------------------------------------- the dose ladder
def _screened(ctx, store, nid, seeds=(11, 12), y=0.6):
    from loop.contracts import Run, RunStatus
    for s in seeds:
        store.put_run(Run(id=Run.make_id(nid, Stage.SCREEN, s), task_id=store.task_id, node_id=nid, seed=s,
                          stage=Stage.SCREEN, status=RunStatus.DONE, scores={a: y for a in ctx.cfg.axes}))


def test_ladder_eligibility_order_one_per_mechanism_inflight_cap_and_scarce_mode(ctx, store, sim):
    """A closed code change ladders at P(own gain>0) >= ladder_p with NO confirm seed (the 1-sigma confirm gate made
    all 55 screened-out moonshots with P in [.2, .85] permanently ineligible), highest P first, one per mechanism,
    at most max_ladders_inflight at once; scarce mode starts only P >= ladder_scarce_p."""
    ctx = _with(ctx, min_confirm_sigmas=1.0, max_ladders_inflight=5, max_concurrent=0)    # ladders only here
    ps = {"n0001": 0.9, "n0002": 0.8, "n0003": 0.75, "n0004": 0.5, "n0005": 0.35, "n0006": 0.25, "n0007": 0.85}
    for i, nid in enumerate(ps):
        _moonshot(ctx, store, sim, nid, cfg=AUG, ts=float(i + 1))
        store.update_node(nid, mechanism_tag="mech_a" if nid == "n0007" else f"mech_{nid[-1]}x")
        _screened(ctx, store, nid)
    store.update_node("n0001", mechanism_tag="mech_a")
    snap = _snap({nid: _post(nid, p) for nid, p in ps.items()})
    got = [n.id for n, _ in MT.eligible_ladders(ctx, store, snap, 1e12)]
    assert got == ["n0001", "n0002", "n0003", "n0004", "n0005"]           # n0007: mech_a again; n0006: P < 0.3
    assert {n.id for n, _ in MT.eligible(ctx, store, snap, 1e12)} == set(ps) - {"n0001"}   # code changes: passes too
    ids = iter(f"n{i:04d}" for i in range(100, 200))
    acts = MT.tick(ctx, store, sim, snap, new_id=lambda: next(ids), now=1e12, scarce=True)
    assert sorted(c["base"] for c in MT.book(store)["current"].values()) == ["n0001", "n0002", "n0003"], acts
    MT.tick(ctx, store, sim, snap, new_id=lambda: next(ids), now=1e12 + 1)
    assert len(MT.book(store)["current"]) == 5 and MT.book(store)["laddered"][-2:] == ["n0004", "n0005"]
    cur = next(c for c in MT.book(store)["current"].values() if c["base"] == "n0001")
    assert cur["knob"] == "aug.strength" and cur["value"] == 0.8 and sorted(cur["doses"]) == ["+1", "-1"]
    pts = [store.get_node(i) for i in cur["trials"]]
    assert [store.get_state(f"config_diff:{p.id}")["aug.strength"] for p in pts] == [0.4, 1.6]
    from loop import seeds as S
    blocks = S.assign_blocks(store.nodes(), ctx.cfg.seeds, store.runs())
    assert all(design_point(p) and p.parent_id == "n0001" and blocks[p.id] == blocks["n0001"] for p in pts)
    small = _with(ctx, max_ladders_inflight=1)
    assert MT._ladders(small, store, sim, snap, lambda: next(ids), 1e12 + 2, False) == []   # 5 running > cap 1


def test_dose_knob_declared_first_never_lr_or_shared_and_bounded_levels(ctx, store, sim):
    ctx = _with(ctx, shared={"train.warmup": 1.5})
    m = _moonshot(ctx, store, sim, "n0001", cfg={"train.lr": 0.002, "train.warmup": 200, "aug.strength": 0.8})
    import yaml
    from loop.adapters import flatten
    wt = sim.prepare_worktree(parent_commit=m.commit, name="look")
    text = (Path(wt.path) / "config.yaml").read_text()
    base = flatten(yaml.safe_load(text))
    assert MT.dose_knob(ctx, store, m, text, base) == ("aug.strength", 0.8, [0.4, 1.6])   # never lr / a shared knob
    m = store.update_node(m.id, notes=m.notes + " dose=loss.temperature")
    assert MT.dose_knob(ctx, store, m, text, base)[0] == "loss.temperature"                 # the researcher's declared
    assert MT.ladder_levels("mask.ratio", 0.75) == [0.375, 0.875] and MT.ladder_levels("k", 3) == [2, 6]
    assert MT.ladder_levels("k", 1) == [2] and MT.ladder_levels("loss.weight", 0.75) == [0.375, 1.5]


def test_dose_fit_shrinks_curvature_toward_the_population_and_1x_wins_ties(ctx, store):
    """Quadratic in log2-dose through 1x, curvature shrunk toward the mean of every finished ladder (DL tau)."""
    flat = MT.dose_fit([(-1.0, 0.004, 0.002), (1.0, -0.004, 0.002)], (0.0, 0.005))
    assert flat["b"] == pytest.approx(-0.004, abs=2e-4) and abs(flat["c"]) < 1e-4 and "c_raw" in flat
    assert MT.curvature_prior(ctx, store) == (0.0, ctx.cfg.prior.tau0)                      # cold
    store.set_state(MT.K_TUNE, {"batches": [{"kind": "ladder", "base": f"x{i}", "fit": {"c_raw": c, "v_raw": 1e-6}}
                                            for i, c in enumerate((-0.006, -0.005, -0.004, -0.0045))]})
    cbar, tau = MT.curvature_prior(ctx, store)
    assert -0.0055 < cbar < -0.0045 and tau < 0.002                                        # an inverted U, tight
    pts = [(-1.0, 0.003, 0.002), (1.0, 0.003, 0.002)]                                       # raw: convex, +0.003
    raw, post = MT.dose_fit(pts, (0.0, 1.0)), MT.dose_fit(pts, (cbar, tau))
    assert raw["c"] == pytest.approx(0.003, abs=1e-4) and post["c"] < 0.0                 # pulled to the population
    assert post["b"] * 1 + post["c"] < 0.0                                                  # so 1x wins: no tuned node


def test_a_ladder_runs_on_the_nodes_own_screen_seeds_and_its_chosen_dose_confirms_on_fresh_seeds(tmp_path):
    ctx, driver, store, adapter = _driver(tmp_path, ladder_p=0.0, max_concurrent=0)   # the ladder alone
    driver.tick()
    m = _moonshot(ctx, store, adapter, "n0001", status=NodeStatus.SCREENING, cfg=AUG)
    assert _run_until(driver, store, lambda: bool(MT.book(store)["batches"]))
    (res,) = MT.book(store)["batches"]
    assert res["kind"] == "ladder" and res["base"] == m.id and res["knob"] == "aug.strength"
    own = {r.seed for r in store.runs(node_id=m.id, stage=Stage.SCREEN)}
    assert set(res["seeds"]) == own and len(res["trials"]) >= 3   # 1/2x, 2x and (the bottom rung won) 1/4x ...
    for nid in res["trials"]:                                     # ... on the node's OWN screen seeds
        assert {r.seed for r in store.runs(node_id=nid)} == own and store.get_node(nid).parent_id == m.id
    assert not store.runs(node_id=m.id, stage=Stage.PARENT)        # paired vs the node's own runs: no new control
    assert {"-1", "+1", "-2"} <= set(res["rows"]) <= {"-1", "+1", "-2", "-3"} and {"b", "c", "prior", "pred"} <= set(res["fit"])
    assert res["fit"]["pred"]["-1"] > res["fit"]["pred"]["+1"]    # sim truth: weaker augmentation helps
    assert any(e["event"] == "ladder_edge" and e["dose"] == "-2" for e in store.events("ladder_edge"))
    tuned = store.get_node(res["tuned"])
    best = max(res["fit"]["pred"], key=lambda x: (res["fit"]["pred"][x], x))
    assert tuned is not None and f"tune_of={m.id}" in tuned.notes and res["best"] == res["rows"][best][3]
    assert _run_until(driver, store, lambda: store.get_node(tuned.id).status not in (
        NodeStatus.SCREENING, NodeStatus.PROMOTING, NodeStatus.CONFIRMING), n=150)
    assert not {r.seed for r in store.runs(node_id=tuned.id)} & own   # its verdict: seeds the ladder never used
    assert "dose ladder t1" in MT.card(store, m.id) and "ladder t1" in MT.lines(store)
    assert not MT.eligible_ladders(ctx, store, driver.snapshot, 1e12)  # once per node


def test_a_key_the_code_merely_reads_is_never_the_dose(ctx, store, sim):
    """Live backlog: train.global_views / train.batch_size / dino.layerwise_decay were picked because the new code
    named them -- shared knobs, not the mechanism's dose. Only a declared or diff-changed key is."""
    import yaml
    from loop.adapters import flatten
    m = _moonshot(ctx, store, sim, "n0001", cfg={})
    wt = sim.prepare_worktree(parent_commit=m.commit, name="look")
    text = (Path(wt.path) / "config.yaml").read_text()
    base = flatten(yaml.safe_load(text))
    m = store.update_node(m.id, notes=m.notes)
    Path(m.diff_path).write_text(Path(m.diff_path).read_text() + "\n+    s = cfg.aug.strength\n")
    assert "aug.strength" in MT.dose_keys(store, m, base) and MT.dose_knob(ctx, store, m, text, base) is None


def test_a_node_on_an_older_carrier_is_backrun_on_the_incumbent_and_picked_on_gain_over_it(ctx, store, sim):
    """Operator 2026-10-03: promising old moonshots tested on an older carrier are re-run ON THE INCUMBENT within the
    ladder: the change grafted onto the incumbent is the 1x point (screened vs the incumbent's controls), the 1/2x
    and 2x points pair against it, and the dose is picked on gain over the incumbent, 1x included."""
    from loop import seeds as S
    from loop.contracts import Run, RunStatus
    ctx = _with(ctx)
    inc = _moonshot(ctx, store, sim, "n0001", status=NodeStatus.CONFIRMED, cfg={"train.lr": 0.0008}, code=False)
    m = _moonshot(ctx, store, sim, "n0002", cfg=AUG, ts=2.0)                      # on root, not the incumbent
    _screened(ctx, store, m.id)
    snap = _snap({m.id: _post(m.id, 0.8)})
    snap.incumbent_id = inc.id
    ids = iter(f"n{i:04d}" for i in range(100, 200))
    acts = MT.tick(ctx, store, sim, snap, new_id=lambda: next(ids), now=1e12)
    (cur,) = MT.book(store)["current"].values()
    assert cur["on"] == inc.id and sorted(cur["doses"]) == ["+0", "+1", "-1"] and "BACKRUN" in acts[-1], acts
    g = store.get_node(cur["at"])
    assert g.parent_id == inc.id and design_point(g) and f"backrun_of={m.id}" in g.notes
    files = sim._object(g.commit)["files"]                                     # the change, grafted on the incumbent
    assert "masked_view" in files["aug.py"] and sim._config(g.commit)["aug"]["strength"] == 0.8 \
        and sim._config(g.commit)["train"]["lr"] == 0.0008
    pts = [store.get_node(cur["doses"][x]) for x in ("-1", "+1")]
    assert all(p.parent_id == g.id for p in pts)
    blocks = S.assign_blocks(store.nodes(), ctx.cfg.seeds, store.runs())
    assert blocks[pts[0].id] == blocks[pts[1].id] == blocks[g.id] != blocks[m.id]   # fresh: never m's seeds
    seeds = S.screen_block(ctx.cfg.seeds, blocks[g.id])
    for nid, y in ((inc.id, 0.60), (g.id, 0.602), (pts[0].id, 0.606), (pts[1].id, 0.598)):
        for s in seeds:
            st = Stage.PARENT if nid == inc.id else Stage.SCREEN
            store.put_run(Run(id=Run.make_id(nid, st, s), task_id=store.task_id, node_id=nid, seed=s, stage=st,
                              status=RunStatus.DONE, scores={a: y for a in ctx.cfg.axes}))
    for nid in (g.id, pts[0].id, pts[1].id):
        store.update_node(nid, status=NodeStatus.SCREENED_OUT)
    acts = MT.tick(ctx, store, sim, snap, new_id=lambda: next(ids), now=1e12 + 1)
    (cur,) = MT.book(store)["current"].values()            # the bottom rung won: the next rung (1/4x) first
    edge = store.get_node(cur["doses"]["-2"])
    assert "next rung" in " ".join(acts) and edge.parent_id == g.id and sim._config(edge.commit)["aug"]["strength"] == 0.2
    for s in seeds:
        store.put_run(Run(id=Run.make_id(edge.id, Stage.SCREEN, s), task_id=store.task_id, node_id=edge.id, seed=s,
                          stage=Stage.SCREEN, status=RunStatus.DONE, scores={a: 0.600 for a in ctx.cfg.axes}))
    store.update_node(edge.id, status=NodeStatus.SCREENED_OUT)
    acts = MT.tick(ctx, store, sim, snap, new_id=lambda: next(ids), now=1e12 + 2)
    (res,) = [c for c in MT.book(store)["batches"] if c.get("kind") == "ladder"]
    assert res["rows"]["+0"][0] == pytest.approx(0.002) and res["rows"]["-1"][0] == pytest.approx(0.004)
    assert res["fit"]["pred"]["+0"] == pytest.approx(0.002) and res["fit"]["pred"]["-1"] > 0.002
    tuned = store.get_node(res["tuned"])
    assert tuned.parent_id == inc.id and tuned.commit == pts[0].commit and f"tune_of={m.id}" in tuned.notes
    assert "over the incumbent n0001" in res["why"], acts
    assert MT.tune_gpu_h(store, m.id) >= 0.0 and {g.id, *res["trials"]} == set(res["trials"])


def test_a_void_ladder_point_is_retried_once_and_a_ladder_closed_on_void_points_is_reopened(ctx, store, sim):
    """Live 2026-10-03: t17/t31/t34's 1/2x and 2x points all ended void:partner and the ladders closed on the 1x
    alone. A void dose point is re-run ONCE (same commit / config, the 1x block); a ladder closed on void points
    before this existed is re-opened with them retried."""
    from loop import seeds as S
    ctx = _with(ctx, max_concurrent=0)                 # the ladder alone (a code change also gets a pass)
    inc = _moonshot(ctx, store, sim, "n0001", status=NodeStatus.CONFIRMED, cfg={"train.lr": 0.0008}, code=False)
    m = _moonshot(ctx, store, sim, "n0002", cfg=AUG, ts=2.0)
    _screened(ctx, store, m.id)
    snap = _snap({m.id: _post(m.id, 0.8)})
    snap.incumbent_id = inc.id
    ids = iter(f"n{i:04d}" for i in range(100, 200))
    MT.tick(ctx, store, sim, snap, new_id=lambda: next(ids), now=1e12)
    (cur,) = MT.book(store)["current"].values()
    g, pts = cur["at"], [cur["doses"][x] for x in ("-1", "+1")]
    store.update_node(g, status=NodeStatus.SCREENED_OUT)
    for nid in pts:
        store.update_node(nid, status=NodeStatus.VOID)
    acts = MT.tick(ctx, store, sim, snap, new_id=lambda: next(ids), now=1e12 + 1)
    (cur,) = MT.book(store)["current"].values()                        # still running: the points re-run
    new = [cur["doses"][x] for x in ("-1", "+1")]
    assert set(new).isdisjoint(pts) and sum("retried once" in a for a in acts) == 2, acts
    blocks = S.assign_blocks(store.nodes(), ctx.cfg.seeds, store.runs())
    for old, nid in zip(pts, new):
        n, o = store.get_node(nid), store.get_node(old)
        assert n.status is NodeStatus.SCREENING and n.commit == o.commit and n.parent_id == g and design_point(n)
        assert store.get_state(f"config_diff:{nid}") == store.get_state(f"config_diff:{old}")
        assert blocks[nid] == blocks[g]
        store.update_node(nid, status=NodeStatus.VOID)                 # void again: never a second retry
    MT.tick(ctx, store, sim, snap, new_id=lambda: next(ids), now=1e12 + 2)
    assert not MT.book(store)["current"]
    bk = MT.book(store)                                                # a ladder closed before the retry existed
    (closed,) = bk["batches"]
    closed.pop("retried")
    closed.update(doses={**closed["doses"], "-1": pts[0], "+1": pts[1]}, trials=[g, *pts])
    store.set_state(MT.K_TUNE, bk)
    acts = MT.tick(ctx, store, sim, snap, new_id=lambda: next(ids), now=1e12 + 3)
    (cur,) = MT.book(store)["current"].values()
    assert not MT.book(store)["batches"] and any("re-opened" in a for a in acts), acts
    assert all(store.get_node(cur["doses"][x]).status is NodeStatus.SCREENING for x in ("-1", "+1"))
    assert "why" not in cur and "closed" not in cur


def _expose_commit(sim, par, req):
    import yaml
    wt = sim.prepare_worktree(parent_commit=par.commit, name=req.id)
    cfg = yaml.safe_load((Path(wt.path) / "config.yaml").read_text())
    cfg["aug"]["mask_power"] = 0.5
    (Path(wt.path) / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True))
    aug = Path(wt.path) / "aug.py"
    aug.write_text(aug.read_text().replace("return x", "return x ** (cfg.aug.mask_power / 0.5)"))
    diff_text, paths, cfg_diff = sim.diff(wt)
    commit = sim.commit(wt, message=req.id)
    sim.reap_worktree(wt)
    return diff_text, paths, cfg_diff, commit


def _exposure(ctx, store, sim, req, *, identical, nid="n0050"):
    """The researcher's knob exposure on the request's parent: aug.mask_power (default 0.5) read by its code."""
    par = store.get_node(req.parent_ids[0])
    diff_text, _, cfg_diff, commit = _expose_commit(sim, par, req)
    node = Node(id=nid, task_id=ctx.task_id, commit=commit, parent_id=par.id, strategy=req.strategy,
                operator=req.operator, family="augmentation", mechanism_tag="jepa_masked_view",
                diff_hash=diff_hash(diff_text), status=NodeStatus.QUEUED, created_ts=5.0, updated_ts=5.0,
                notes=f"request={req.id} brief=expose dose=aug.mask_power" + (" identical=1" if identical else ""))
    store.put_node(MT.adopt(store, req, node))
    store.set_state(f"config_diff:{nid}", dict(cfg_diff))
    return store.get_node(nid)


@pytest.mark.parametrize("identical", [True, False])
def test_a_ladder_with_no_dose_knob_first_gets_its_hard_coded_strength_exposed(ctx, store, sim, identical):
    """Operator 2026-10-03: ~36 promising old moonshots hard-code their strength (or reweight their data sampling),
    so they got no ladder. A side-slot researcher request exposes the constant as a config key (default = the current
    value) on the 1x commit. Identical at 1x: the 1x measurements stand, the 1/2x / 2x points are built on the exposed
    commit and paired against the measured 1x node; not identical: the exposed commit is a new 1x point on its block."""
    from loop import seeds as S
    ctx = _with(ctx)
    m = _moonshot(ctx, store, sim, "n0001", cfg={})                               # code only: no dose knob
    _screened(ctx, store, m.id)
    snap = _snap({m.id: _post(m.id, 0.8)})
    ids = iter(f"n{i:04d}" for i in range(100, 200))
    acts = MT.tick(ctx, store, sim, snap, new_id=lambda: next(ids), now=1e12)
    (cur,) = MT.book(store)["current"].values()
    assert cur["expose"] == "want" and not cur["trials"] and "knob exposure requested" in acts[-1], acts
    assert MT.requests(ctx, store, snap, limit=0) == []                           # the side-slot cap
    (req,) = MT.requests(ctx, store, snap, limit=2, scarce=True)   # scarce mode too
    assert req.brief_kind == "expose" and req.parent_ids == (m.id,) and "mask ratio" in req.brief
    assert MT.tick(ctx, store, sim, snap, new_id=lambda: next(ids), now=1e12 + 1, inflight=[req.id]) == []
    ex = _exposure(ctx, store, sim, req, identical=identical)
    assert ex.status is NodeStatus.REJECTED and ex.reject_reason == "carrier" and design_point(ex)
    acts = MT.tick(ctx, store, sim, snap, new_id=lambda: next(ids), now=1e12 + 2)
    (cur,) = MT.book(store)["current"].values()
    assert cur["knob"] == "aug.mask_power" and cur["value"] == 0.5 and cur["identical"] is identical, acts
    pts = [store.get_node(cur["doses"][x]) for x in ("-1", "+1")]
    assert all(sim._config(p.commit)["aug"]["mask_power"] in (0.25, 1.0) for p in pts)   # on the exposed commit
    blocks = S.assign_blocks(store.nodes(), ctx.cfg.seeds, store.runs())
    ex = store.get_node(ex.id)
    if identical:                                  # 1x stands: points pair against n0001's own screen runs
        assert ex.status is NodeStatus.REJECTED and "+0" not in cur["doses"] and "at" not in cur
        assert all(p.parent_id == m.id and blocks[p.id] == blocks[m.id] for p in pts)
    else:                                          # re-measured at 1x on n0001's block
        assert ex.status is NodeStatus.SCREENING and cur["at"] == ex.id == cur["doses"]["+0"] and ex.parent_id == ROOT_NODE_ID
        assert all(p.parent_id == ex.id for p in pts) and blocks[ex.id] == blocks[pts[0].id] == blocks[m.id]
    assert sorted(cur["trials"]) == sorted(cur["doses"].values())


def test_a_failed_exposure_is_retried_then_closes_the_ladder_and_frees_its_mechanism(ctx, store, sim):
    ctx = _with(ctx, max_research_fails=2, max_concurrent=0)
    m = _moonshot(ctx, store, sim, "n0001", cfg={})
    _screened(ctx, store, m.id)
    snap = _snap({m.id: _post(m.id, 0.8)})
    ids = iter(f"n{i:04d}" for i in range(100, 200))
    MT.tick(ctx, store, sim, snap, new_id=lambda: next(ids), now=1e12)
    (req,) = MT.requests(ctx, store, snap, limit=1)
    MT.tick(ctx, store, sim, snap, new_id=lambda: next(ids), now=1e12 + 1)        # produced nothing
    (cur,) = MT.book(store)["current"].values()
    assert cur["expose"] == "want" and cur["fails"] == 1
    (req,) = MT.requests(ctx, store, snap, limit=1)
    assert req.id.endswith("-2")
    bad = Node(id="n0050", task_id=ctx.task_id, commit="", parent_id=m.id, strategy=req.strategy, operator=req.operator,
               family="augmentation", mechanism_tag="x", diff_hash="h", status=NodeStatus.REJECTED,
               reject_reason="smoke_failed", notes=f"request={req.id}")
    store.put_node(MT.adopt(store, req, bad))
    acts = MT.tick(ctx, store, sim, snap, new_id=lambda: next(ids), now=1e12 + 2)
    (res,) = MT.book(store)["batches"]
    assert not MT.book(store)["current"] and "knob exposure failed 2x (smoke_failed)" in res["why"], acts
    assert not res["trials"]                                     # a no-knob node never uses up its mechanism



def test_an_exposure_no_ladder_point_can_set_is_retried_with_the_why(ctx, store, sim):
    """Live 2026-10-04: 6 of 14 exposures declared a key their code read with a default (data.get(key, 0.5)) but
    never wrote into the config, under no family: no point could set it, the ladder closed 'no usable dose knob'
    and the carrier was wasted. Such an exposure is a failed one, asked again with the reason in the brief."""
    ctx = _with(ctx)
    m = _moonshot(ctx, store, sim, "n0001", cfg={})
    _screened(ctx, store, m.id)
    snap = _snap({m.id: _post(m.id, 0.8)})
    ids = iter(f"n{i:04d}" for i in range(100, 200))
    MT.tick(ctx, store, sim, snap, new_id=lambda: next(ids), now=1e12)
    (req,) = MT.requests(ctx, store, snap, limit=1)
    assert "MUST be written into the config file" in req.brief and "aug.* (augmentation)" in req.brief
    par = store.get_node(req.parent_ids[0])
    node = Node(id="n0050", task_id=ctx.task_id, commit=par.commit, parent_id=par.id, strategy=req.strategy,
                operator=req.operator, family="augmentation", mechanism_tag="x", diff_hash="h1",
                status=NodeStatus.QUEUED, notes=f"request={req.id} brief=expose dose=aug.code_default identical=1")
    store.put_node(MT.adopt(store, req, node))
    acts = MT.tick(ctx, store, sim, snap, new_id=lambda: next(ids), now=1e12 + 1)
    (cur,) = MT.book(store)["current"].values()
    assert cur["expose"] == "want" and cur["fails"] == 1 and not cur["knob"] and not cur["trials"], acts
    assert store.get_node("n0050").reject_reason == "expose_unusable"       # a real failure, not a carrier
    (req,) = MT.requests(ctx, store, snap, limit=1)
    assert req.id.endswith("-2") and "previous exposure failed: declared dose knob 'aug.code_default'" in req.brief


def test_the_driver_issues_the_exposure_on_a_side_slot_and_ladders_the_exposed_knob(tmp_path):
    ctx, driver, store, adapter = _driver(tmp_path, ladder_p=0.0)

    class Exposer:
        calls: list = []

        def research(self, req, store, snapshot):
            if req.brief_kind != "expose":
                return ResearchResult("failed", None, "not scripted")
            self.calls.append(req)
            par = store.get_node(req.parent_ids[0])
            diff_text, paths, cfg_diff, commit = _expose_commit(adapter, par, req)
            h = HypothesisRecord(f"h-{req.id}", par.task_id, HypothesisKind.TRIAL, "exposed", "jepa_masked_view",
                                 "augmentation", {}, "none")
            return ResearchResult("ok", Candidate(req.id, par.task_id, commit, par.commit, diff_text,
                                                  diff_hash(diff_text), tuple(paths), dict(cfg_diff), h,
                                                  HandoffNote("", par.task_id, req.id, "exposed", "", "")),
                                  dose_knob="aug.mask_power", identical=True)
    driver.researcher = Exposer()
    driver.tick()
    m = _moonshot(ctx, store, adapter, "n0001", status=NodeStatus.SCREENING, cfg={})
    assert _run_until(driver, store, lambda: bool(MT.book(store)["batches"]), n=150)
    (res,) = MT.book(store)["batches"]
    assert len(Exposer.calls) == 1 and res["knob"] == "aug.mask_power" and res["identical"] is True, res
    assert {store.get_node(t).parent_id for t in res["trials"]} == {m.id} and len(res["trials"]) == 2
    ex = store.get_node(res["exposed"])
    assert ex.reject_reason == "carrier" and " identical=1" in ex.notes and "dose=aug.mask_power" in ex.notes


def test_knob_less_ladders_from_before_exposures_are_backfilled_reusing_their_measured_graft(ctx, store, sim):
    """Deploy-time backfill: a running knob-less ladder gets its exposure; a closed one (no knob, never exposed) is
    laddered again, a backrun reusing its finished 1x graft on the same incumbent instead of re-running it."""
    off, on = _with(ctx, expose=False, max_concurrent=0), _with(ctx, max_concurrent=0)
    inc = _moonshot(ctx, store, sim, "n0001", status=NodeStatus.CONFIRMED, cfg={"train.lr": 0.0008}, code=False)
    m = _moonshot(ctx, store, sim, "n0002", cfg={}, ts=2.0)
    _screened(ctx, store, m.id)
    snap = _snap({m.id: _post(m.id, 0.8)})
    snap.incumbent_id = inc.id
    ids = iter(f"n{i:04d}" for i in range(100, 200))
    MT.tick(off, store, sim, snap, new_id=lambda: next(ids), now=1e12)
    (cur,) = MT.book(store)["current"].values()
    g = cur["at"]
    assert not cur["knob"] and "expose" not in cur and cur["trials"] == [g]          # a 1x-only backrun
    store.update_node(g, status=NodeStatus.SCREENED_OUT)
    MT.tick(off, store, sim, snap, new_id=lambda: next(ids), now=1e12 + 1)
    assert not MT.book(store)["current"] and not MT.eligible_ladders(off, store, snap, 1e12)
    assert [n.id for n, _ in MT.eligible_ladders(on, store, snap, 1e12)] == [m.id]   # backfilled
    n_nodes = len(store.nodes())
    MT.tick(on, store, sim, snap, new_id=lambda: next(ids), now=1e12 + 2)
    (cur,) = MT.book(store)["current"].values()
    assert cur["at"] == g and cur["on"] == inc.id and cur["expose"] == "want" and len(store.nodes()) == n_nodes
    bk = MT.book(store)                                              # a running knob-less ladder from before
    bk["current"][str(cur["batch"])].pop("expose")
    store.set_state(MT.K_TUNE, bk)
    MT.tick(on, store, sim, snap, new_id=lambda: next(ids), now=1e12 + 3)
    assert MT.book(store)["current"][str(cur["batch"])]["expose"] == "want"


def test_an_exposure_or_tuning_request_an_outage_ate_is_asked_again_without_a_failure(ctx, store, sim):
    """Live 2026-10-03 22:25-22:55Z: t18/t19/t21/t22 closed 'knob exposure failed 3x (no candidate)' -- every
    request was refused by the account (backend_unavailable). Such a request is given back: no fails count."""
    ctx = _with(ctx, max_research_fails=1)
    m = _moonshot(ctx, store, sim, "n0001", cfg={})
    _screened(ctx, store, m.id)
    snap = _snap({m.id: _post(m.id, 0.8)})
    ids = iter(f"n{i:04d}" for i in range(100, 200))
    MT.tick(ctx, store, sim, snap, new_id=lambda: next(ids), now=1e12)
    for k in range(3):
        (req,) = MT.requests(ctx, store, snap, limit=1)
        assert req.id.endswith("-1") and MT.outage(store, req.id).startswith("t")
        MT.tick(ctx, store, sim, snap, new_id=lambda: next(ids), now=1e12 + 1 + k)
        (cur,) = MT.book(store)["current"].values()
        assert cur["expose"] == "want" and cur["fails"] == 0 and not cur["pending"]
    assert MT.outage(store, "req-unknown") == ""


def test_hparam_like_mechanism_types_never_ladder(ctx, store, sim):
    """moonshot_tune.ladder_skip_types (operator 2026-10-05: deprioritize low-yield ladders): a code change whose
    mechanism type is listed (e.g. optimization) gets no dose ladder; other types still do."""
    ctx = _with(ctx, ladder_skip_types=("optimization",))
    for i, (nid, mtype) in enumerate([("n0001", "optimization"), ("n0002", "loss_objective")]):
        _moonshot(ctx, store, sim, nid, cfg=AUG, ts=float(i + 1), notes=f"brief=moonshot backend=x mtype={mtype}")
        store.update_node(nid, mechanism_tag=f"mech_{nid[-1]}x")
        _screened(ctx, store, nid)
    snap = _snap({nid: _post(nid, 0.9) for nid in ("n0001", "n0002")})
    assert [n.id for n, _ in MT.eligible_ladders(ctx, store, snap, 1e12)] == ["n0002"]
