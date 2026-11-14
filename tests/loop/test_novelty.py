"""The novelty engine (loop.novelty): sweep redirection to the design knob pool, MAP-Elites cells + the
elite as inspiration, the audited novelty bonus on bandit rewards, ideation personas (+ their meta record)
and the web session's prior-work statement.
Run: uv run pytest tests/loop -q --confcutdir=tests/loop"""

from __future__ import annotations

import dataclasses
import random

import pytest
from types import SimpleNamespace

from loop import components as CG, design as D, meta as ML, novelty as NV
from loop.contracts import (ROOT_NODE_ID, ArmPull, Candidate, CandidateRequest, HandoffNote, HypothesisKind,
                            HypothesisRecord, LeverTags, Node, NodeStatus, Operator, ResearchResult, Strategy)
from loop.select import intake
from loop.select.embed import HashingEmbedder
from loop.spec import ComponentsCfg, NoveltyCfg
from tests.loop.test_driver import Scripts, _make, _world

CFG_DIFF = ("diff --git a/config.yaml b/config.yaml\n--- a/config.yaml\n+++ b/config.yaml\n@@ -1,3 +1,3 @@\n"
            " train:\n-  lr: 0.001\n+  lr: 0.0008\n")
CODE_DIFF = ("diff --git a/train.py b/train.py\n--- a/train.py\n+++ b/train.py\n@@ -1 +1,2 @@\n"
             "+loss = loss + 0.1 * koleo_loss(z)  # spread the features\n")


def _on(ctx, **kw):
    return dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, novelty=NoveltyCfg(**kw)))


def _cand(diff=CFG_DIFF, paths=("config.yaml",), cfg=None, family="optimization", claim="lower lr",
          levers=None):
    h = HypothesisRecord(id="", task_id="fake-loop", kind=HypothesisKind.TRIAL, claim=claim, mechanism_tag="t",
                         family=family, predictions={}, falsification="f",
                         levers=LeverTags(levers or {"effective_lr": -1}))
    return Candidate("req-9", "fake-loop", "c", "c0", diff, f"h{hash(diff)}", tuple(paths),
                     {"train.lr": 0.0008} if cfg is None else cfg, h, HandoffNote("", "fake-loop", "req-9", "", "", ""))


def test_sweep_only_is_detected_and_rejected_on_every_arm_but_design_and_debug(ctx, store):
    c = _on(ctx)
    assert NV.sweep_knobs(_cand()) == {"train.lr": 0.0008}
    assert not NV.sweep_knobs(_cand(diff=CFG_DIFF + CODE_DIFF, paths=("config.yaml", "train.py")))   # code
    assert not NV.sweep_knobs(_cand(diff=CFG_DIFF.replace("-  lr: 0.001\n", ""), cfg={"train.lr": 5e-4}))  # new key
    assert not NV.sweep_knobs(_cand(cfg={"train.lr": True}))                                         # a bool
    rule = lambda s, o: intake.diagnose(c, store, _cand(), operator=o, embedder=HashingEmbedder(),  # noqa: E731
                                        strategy=s)[0]
    assert rule(Strategy.EXPLORE, Operator.DRAFT) == "sweep_only" == rule(Strategy.MOONSHOT, Operator.DRAFT)
    fix = intake.diagnose(c, store, _cand(), operator=Operator.DRAFT, embedder=HashingEmbedder(),
                          strategy=Strategy.EXPLORE)[1]
    assert fix.startswith("numeric-only tweak -> introduce a mechanism")
    for s, o in ((Strategy.EXPLOIT, Operator.IMPROVE), (Strategy.CROSSOVER, Operator.CROSSOVER),
                 (Strategy.REVISIT, Operator.IMPROVE)):           # every arm (operator 2026-09-30) ...
        assert rule(s, o) == "sweep_only"
    for s, o in ((Strategy.REVISIT, Operator.DEBUG), (Strategy.DESIGN, None), (Strategy.DESIGN, Operator.IMPROVE)):
        assert rule(s, o) != "sweep_only"                          # ... but a debug fix, a design point, a tuning trial
    assert intake.diagnose(_on(ctx, sweep_redirect=False), store, _cand(), operator=Operator.DRAFT,
                           embedder=HashingEmbedder(), strategy=Strategy.EXPLORE)[0] != "sweep_only"


def test_redirected_sweep_feeds_the_next_design_batch(tmp_path):
    ctx = _on(_world(tmp_path))
    driver, store, _ = _make(ctx, Scripts())
    driver.tick()
    req = CandidateRequest(id="req-77", task_id=ctx.task_id, strategy=Strategy.EXPLORE, operator=Operator.DRAFT,
                           island="main", parent_ids=(ROOT_NODE_ID,), brief="b", brief_kind="literature")
    root = store.get_node(ROOT_NODE_ID)
    cand = dataclasses.replace(_cand(), parent_commit=root.commit)
    out = driver._intake(req, ResearchResult("ok", cand))
    assert out.startswith("redirected") and "train.lr" in out
    node = [n for n in store.nodes() if n.reject_reason == "sweep_only"][0]
    assert node.status is NodeStatus.REJECTED and "mtype=optimization" in node.notes
    pool = {e["key"]: e for e in NV.pool_entries(store)}
    assert pool["train.lr"] | {"ts": 0} == {"key": "train.lr", "high": 0.0008, "levers": {"effective_lr": -1},
                                            "node": node.id, "ts": 0}
    assert store.events("sweep_redirect")[-1]["knobs"] == ["train.lr"]
    # the researcher's check fires first (it then revises): its sweep's knobs reach the pool too (0 events live)
    assert any(e.get("at") == "check" for e in store.events("sweep_redirect"))
    text = "train:\n  lr: 0.001\n  warmup: 10\n"
    base = {"train.lr": 0.001, "train.warmup": 10}
    snap = SimpleNamespace(nodes={})
    got = D.pick_knobs(ctx, store, snap, text, base)
    assert got[0] == {"key": "train.lr", "low": 0.001, "high": 0.0008, "levers": {"effective_lr": -1}}
    assert D.pick_knobs(ctx, store, snap, text, base, auto=False) == []      # the pool rides with the auto pick
    assert "design knob pool" in D.lines(store)
    NV.consume(store, ["train.lr"])
    assert "train.lr" not in {e["key"] for e in NV.pool_entries(store)}


def _plant(store, nid, family, notes, *, diff_path="", tag="t", emb=None, ts=1.0, status=NodeStatus.SCREENED_OUT,
           strategy=Strategy.EXPLORE):
    store.put_node(Node(id=nid, task_id=store.task_id, commit="c", parent_id=ROOT_NODE_ID, strategy=strategy,
                        operator=Operator.DRAFT, family=family, mechanism_tag=tag, diff_hash=f"h{nid}", status=status,
                        notes=notes, diff_path=diff_path, embedding=emb, created_ts=ts))


def test_mechanism_type_cells_target_and_inspiration(ctx, store, tmp_path):
    assert NV.mechanism_type("loss_objective", ("train.py",), CODE_DIFF) == "loss_objective"      # declared, shown
    assert NV.mechanism_type("readout", ("train.py",), CODE_DIFF) == "loss_objective"            # contradicted
    assert NV.mechanism_type("", ("config.yaml",), CFG_DIFF) == "optimization"
    d = tmp_path / "k.diff"
    d.write_text(CODE_DIFF)
    _plant(store, "a", "objective", "", diff_path=str(d), tag="koleo")            # legacy: classified from its diff
    _plant(store, "b", "objective", "mtype=augmentation")
    _plant(store, "r", "objective", "mtype=readout", status=NodeStatus.REJECTED)  # rejected: not in a cell
    c = NV.counts(store, "objective")
    assert c["loss_objective"] == 1 and c["augmentation"] == 1 and c["readout"] == 0
    assert NV.target(store, "objective") not in ("loss_objective", "augmentation")
    assert {NV.target(store, "objective", s) for s in range(12)} == set(NV.TYPES) - {"loss_objective", "augmentation"}
    snap = SimpleNamespace(nodes={"a": SimpleNamespace(utility=SimpleNamespace(n=2, mean=0.004, sd=0.001),
                                                       p_positive=0.9)})
    assert NV.elite(store, snap, "objective", "loss_objective").id == "a"
    block = NV.cell_block(store, snap, "objective", "loss_objective")
    assert "Target cell (MAP-Elites): family objective x type loss_objective" in block and "koleo_loss" in block
    assert "empty" in NV.cell_block(store, snap, "objective", "readout")
    assert NV.scout_block(store, snap).startswith("Mechanism types by nodes tried")


def test_personas_rotate_and_meta_tracks_the_winner(ctx, store):
    assert [NV.persona_for(store, True) for _ in range(4)] == list(NV.PERSONAS)
    assert "literature" not in {NV.persona_for(store, False) for _ in range(6)}
    for i, (p, hit) in enumerate([("theorist", True), ("theorist", True), ("contrarian", False)]):
        _plant(store, f"p{i}", "objective", f"persona={p} mtype=loss_objective")
    snap = SimpleNamespace(nodes={f"p{i}": SimpleNamespace(utility=SimpleNamespace(n=2, mean=0.01 if h else -0.01),
                                                            p_positive=0.9 if h else 0.1, p_gain=0.9 if h else 0.1)
                                  for i, h in enumerate((True, True, False))})
    r = ML.resolve(ctx, store, snap, "persona")
    assert "theorist hit 2/2" in r["headline"] and "leading: theorist" in r["headline"]
    assert "persona" in ML.COMPONENTS and ctx.cfg.meta.persona_bias == 0.0


def test_ideation_request_carries_persona_cell_and_prompt_block(ctx, store):
    from loop.select import PortfolioSelector
    c = _on(ctx)
    sel = PortfolioSelector(c, world=None, rng=random.Random(1))
    store.put_node(Node(id=ROOT_NODE_ID, task_id=store.task_id, commit="c0", parent_id=None,
                        strategy=Strategy.BASELINE, operator=None, family="", mechanism_tag="root", diff_hash="",
                        status=NodeStatus.CONFIRMED))
    snap = SimpleNamespace(nodes={}, incumbent_id=ROOT_NODE_ID, levers=(), hypotheses={})
    req = sel._mint(store, snap, strategy=Strategy.MOONSHOT, operator=Operator.DRAFT, island="main",
                    parent_ids=(ROOT_NODE_ID,), brief_kind="moonshot")
    assert req.persona == "theorist" and req.cell in NV.TYPES
    probe = sel._mint(store, snap, strategy=Strategy.EXPLORE, operator=Operator.IMPROVE, island="main",
                      parent_ids=(ROOT_NODE_ID,), brief_kind="probe", meta_id="m1")
    exploit = sel._mint(store, snap, strategy=Strategy.EXPLOIT, operator=Operator.IMPROVE, island="main",
                        parent_ids=(ROOT_NODE_ID,), brief_kind="frontier")
    assert (probe.persona, probe.cell, exploit.persona) == ("", "", "")
    text = NV.block(c, store, snap, req)
    assert text.startswith("## Novelty") and "Persona: THEORIST" in text and "Target cell" in text \
        and "self-critique" in text
    assert NV.block(c, store, snap, exploit) == ""
    assert "Persona" not in NV.block(c, store, snap, probe) and "redirected" in NV.block(c, store, snap, probe)


def test_novelty_bonus_is_bounded_recorded_and_audited(ctx, store, tmp_path):
    from loop.select import PortfolioSelector
    emb = HashingEmbedder()
    d = tmp_path / "code.diff"
    d.write_text(CODE_DIFF)
    _plant(store, "old", "objective", "", emb=emb.embed("lower the learning rate schedule warmup"), ts=1.0)
    _plant(store, "new", "objective", "", emb=emb.embed("koleo feature spreading regulariser on patch tokens"),
           ts=2.0, diff_path=str(d))
    _plant(store, "knob", "optimization", "", emb=emb.embed("lower the learning rate schedule warmup a bit"), ts=3.0)
    c = _on(ctx, weight=0.5)
    new, knob = store.get_node("new"), store.get_node("knob")
    assert 0.0 < NV.bonus(c, store, new) <= 0.5 and NV.bonus(c, store, knob) == 0.0    # a config-only node: none
    assert NV.distance(store.get_node("old"), store.nodes()) == 1.0                    # nothing before it
    assert NV.bonus(_on(ctx, weight=0.0), store, new) == 0.0
    sel = PortfolioSelector(c, world=None, rng=random.Random(1))
    import loop.select as S
    orig = S.compute_reward
    S.compute_reward = lambda *a, **k: ArmPull(store.task_id, Strategy.EXPLORE, "new", 0.2, 2.0, 5.0)
    try:
        pull = sel.reward(store, None, SimpleNamespace(nodes={}, sigma={}), new)
    finally:
        S.compute_reward = orig
    assert pull.reward == 0.2                                          # the raw pull is untouched
    per = store.get_state(NV.K_BONUS)["new"]
    assert per == round(NV.bonus(c, store, new) / 2.0, 6) and store.events("novelty_bonus")
    got = NV.with_bonus(store, [pull, dataclasses.replace(pull, node_id="knob")])
    assert got[0].reward == 0.2 + per and got[1].reward == 0.2
    store.update_node("new", status=NodeStatus.CRASHED)            # a crash (~0 GPU-h) earns no bonus
    store.set_state(NV.K_BONUS, {})
    S.compute_reward = lambda *a, **k: ArmPull(store.task_id, Strategy.EXPLORE, "new", 0.0, 0.0, 5.0)
    try:
        sel.reward(store, None, SimpleNamespace(nodes={}, sigma={}), store.get_node("new"))
    finally:
        S.compute_reward = orig
    assert not store.get_state(NV.K_BONUS)


def test_arm_choice_seeing_the_bonus_is_an_audited_decision(tmp_path):
    ctx = _world(tmp_path)
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, components=ComponentsCfg(ablation_rate=0.5)))
    driver, store, _ = _make(ctx, Scripts())
    store.set_state(NV.K_BONUS, {"nX": 0.3})
    for _ in range(3):
        driver.tick()
    ev = [e for e in store.events(CG.K_GATE) if e["component"] == "novelty"]
    assert ev and all(e["decision"] == "arm" and e["request"].startswith("req-") for e in ev)
    reqs = {e["request_id"] for e in store.events("research_started")}
    assert {e["request"] for e in ev} <= reqs | {f"req-{len(reqs) + 1:05d}"}


def test_web_session_must_state_prior_work(ctx):
    from loop.agents.researcher import MAX_FINISH_BOUNCES, WorktreeTools, finish_schema
    tools = WorktreeTools(None, SimpleNamespace(path="/tmp", branch="b", base_commit="c"), writable=(),
                          locked=(), checker=lambda **kw: "OK")
    fin = {"hypothesis": {"claim": "x"}}
    assert tools.check_finish(fin) == ""
    tools.need_prior_work = True
    assert "REJECT prior_work" in tools.check_finish(fin)
    assert tools.check_finish({"hypothesis": {"claim": "x", "prior_work": "arXiv:2304.07193; not in train.py"}}) == ""
    tools.bounces = MAX_FINISH_BOUNCES
    assert tools.check_finish(fin) == ""                               # capped like every bounce
    hyp = finish_schema(ctx)["properties"]["hypothesis"]["properties"]
    assert hyp["mechanism_type"]["enum"] == list(NV.TYPES) and "prior_work" in hyp and "dose_knob" in hyp


def test_major_moves_and_substantiated_small_moves_pass_the_sweep_rule(ctx, store):
    c = _on(ctx)
    rule = lambda cand, snap=None: intake.diagnose(c, store, cand, operator=Operator.IMPROVE,  # noqa: E731
                                                   embedder=HashingEmbedder(), strategy=Strategy.EXPLOIT,
                                                   snapshot=snap)[0]
    major = _cand(diff=CFG_DIFF.replace("0.0008", "0.0004"), cfg={"train.lr": 0.0004})     # 2.5x
    assert NV.ratio(0.0, 1.0) == float("inf") and NV.ratio([0.4, 1.0], [0.1, 1.0]) == 4.0
    assert rule(major) != "sweep_only" and rule(_cand()) == "sweep_only"
    # backed by the lever regression (design.lever_p > 0): P(effective_lr effect < 0) = 0.9 >= 0.8
    lp = dataclasses.replace(c, cfg=dataclasses.replace(c.cfg, design=dataclasses.replace(c.cfg.design, lever_p=0.8)))
    lever = lambda mean: SimpleNamespace(nodes={}, sigma={"panel": 0.001}, levers=(SimpleNamespace(  # noqa: E731
        lever="effective_lr", target="panel", p_positive=0.1, mean=mean),))
    snap = lever(-0.01)
    assert intake.diagnose(lp, store, _cand(), operator=Operator.IMPROVE, embedder=HashingEmbedder(),
                           strategy=Strategy.EXPLOIT, snapshot=snap)[0] == ""                  # 0.01 >= 2 x 0.001
    assert intake.diagnose(lp, store, _cand(), operator=Operator.IMPROVE, embedder=HashingEmbedder(),
                           strategy=Strategy.EXPLOIT, snapshot=lever(-0.001))[0] == "below_noise"
    assert rule(_cand(), snap) == "sweep_only"                               # lever_p 0: the regression is off
    # ... or by a designed batch's significant effect in the direction of the move (lower lr was better)
    store.set_state(D.K_DESIGN, {"batches": [{"batch": 1, "knobs": [{"key": "train.lr", "low": 0.001, "high": 0.002}],
                                              "effects": [{"effect": -0.004, "significant": True}]}]})
    why, eff = NV.expected(c, store, None, _cand(), NV.moves(_cand()))
    assert why == "design d1 effect on train.lr" and eff == pytest.approx(0.004 * 0.0002 / 0.001)   # scaled to the move
    assert NV.mde(c, None) == pytest.approx(2 * c.cfg.panel_sd / len(c.cfg.seeds.screen) ** 0.5)
    assert rule(_cand()) == "below_noise"                                    # backed, but a minute move: noise
    store.set_state(D.K_DESIGN, {"batches": [{"batch": 1, "knobs": [{"key": "train.lr", "low": 0.001, "high": 0.002}],
                                              "effects": [{"effect": -10.0, "significant": True}]}]})
    assert rule(_cand()) == ""
    store.set_state(D.K_DESIGN, {"batches": [{"batch": 1, "knobs": [{"key": "train.lr", "low": 0.001, "high": 0.002}],
                                              "effects": [{"effect": 0.004, "significant": True}]}]})
    assert rule(_cand()) == "sweep_only"                                     # the design says the other way


def _plant_cfg(store, nid, cfg, parent=ROOT_NODE_ID, status=NodeStatus.SCREENED_OUT, commit=None, diff_path=""):
    store.put_node(Node(id=nid, task_id=store.task_id, commit=commit or f"c-{nid}", parent_id=parent,
                        strategy=Strategy.EXPLORE, operator=Operator.DRAFT, family="optimization", mechanism_tag=nid,
                        diff_hash=f"h-{nid}", status=status, diff_path=diff_path, created_ts=float(len(store.nodes()))))
    store.set_state(f"config_diff:{nid}", cfg)


def _xover(cfg):
    """A crossover candidate re-rooted on root: its config_diff is the FULL change set vs root."""
    lines = "".join(f"-  {k.rsplit('.', 1)[-1]}: 0\n+  {k.rsplit('.', 1)[-1]}: {v}\n" for k, v in cfg.items())
    diff = f"diff --git a/config.yaml b/config.yaml\n--- a/config.yaml\n+++ b/config.yaml\n@@ -1 +1 @@\n{lines}"
    return dataclasses.replace(_cand(diff=diff, cfg=dict(cfg), family="objective", claim=str(cfg)),
                               parent_commit="c-root")


def test_near_dup_crossovers_are_judged_on_effective_configs(ctx, store, tmp_path):
    """The live n0292 shape: a crossover re-rooted on root carries the whole stack (MAJOR keys), so its small
    tweak (kde x1.33, warmup x1.8 on a key the head never listed, hed_jitter x1.5) hid behind them."""
    c = _on(ctx)
    store.ensure_root("c-root")
    store.set_state(intake.K_BASE, {"train.lr": 0.001, "loss.kde_w": 0.1, "loss.depth": 4, "train.warmup": 0.09,
                                    "aug.local": [0.08, 0.35], "aug.hed": 0.07})
    inc = {"train.lr": 0.0019, "loss.kde_w": 0.0667}
    _plant_cfg(store, "I", inc, status=NodeStatus.CONFIRMED)
    _plant_cfg(store, "N", {"loss.depth": 6}, parent="I", status=NodeStatus.NOT_CONFIRMED)
    full = {**inc, "loss.depth": 6}
    rule = lambda cand: intake.diagnose(c, store, cand, operator=Operator.CROSSOVER,  # noqa: E731
                                        embedder=HashingEmbedder(), strategy=Strategy.CROSSOVER)[0]
    assert rule(_xover(full)) == "near_dup_config"                               # n0363: N's config, re-rooted
    assert rule(_xover({**full, "loss.kde_w": 0.05})) == "sweep_only"            # n0374: + kde x1.33
    assert rule(_xover({**full, "train.warmup": 0.05})) == "sweep_only"          # n0376: + warmup x1.8 (new key)
    assert rule(_xover({**full, "aug.local": [0.16, 0.35]})) == "sweep_only"     # n0369: EXACTLY 2x is minor
    assert NV.minor(2.0, 2.0) and NV.minor(0.16 / 0.08, 2.0) and not NV.minor(2.001, 2.0)
    assert intake.config_twin(c, store, _xover({**full, "aug.hed": 0.035})).id == "N"   # a 2x step is a twin
    crops = {**full, "aug.local": [0.2, 0.35]}
    assert rule(_xover(crops)) == ""                                             # a 2.5x crop move is MAJOR
    rel, gaps, _, _ = intake.nearest(c, store, _xover({**crops, "aug.hed": 0.04667}))
    assert rel.id == "N" and set(gaps) == {"aug.local", "aug.hed"}
    # the queued pair: re-intake keeps the MAJOR one and rejects its small tweak (itself excluded)
    for nid, cfg in (("X9", crops), ("XA", {**crops, "aug.hed": 0.04667})):
        p = tmp_path / f"{nid}.diff"
        p.write_text(_xover(cfg).diff_text)
        _plant_cfg(store, nid, cfg, status=NodeStatus.QUEUED, diff_path=str(p))
        store.update_node(nid, strategy=Strategy.CROSSOVER, operator=Operator.CROSSOVER)
    assert intake.recheck(c, store, None, store.get_node("X9"))[0] == ""
    assert intake.recheck(c, store, None, store.get_node("XA"))[0] == "sweep_only"
    assert NV.pool_entries(store) == []                                          # a plain recheck only reads
    assert intake.recheck(c, store, None, store.get_node("XA"), forward=True)[0] == "sweep_only"
    assert [(e["key"], e["high"], e["node"]) for e in NV.pool_entries(store)] == [("aug.hed", 0.04667, "XA")]


def test_winner_pairs_skip_a_config_twin_of_a_partner_already_crossed_with_the_same_parent(store):
    from loop.select import winner_pairs
    store.ensure_root("c-root")
    store.set_state(intake.K_BASE, {"train.lr": 0.001, "aug.hed": 0.07})
    _plant_cfg(store, "H", {"model.w": 2}, status=NodeStatus.CONFIRMED)
    _plant_cfg(store, "P", {"aug.hed": 0.03}, status=NodeStatus.CONFIRMED)
    _plant_cfg(store, "Q", {"aug.hed": 0.035}, status=NodeStatus.CONFIRMED)          # P's twin (x1.17)
    _plant_cfg(store, "R", {"train.lr": 0.004}, status=NodeStatus.CONFIRMED)
    snap = SimpleNamespace(nodes={}, incumbent_id="H")
    assert {frozenset(p) for p in winner_pairs(store, snap)} >= {frozenset("HQ"), frozenset("HP")}
    pairs = {frozenset(p) for p in winner_pairs(store, snap, {frozenset("HP")})}
    assert frozenset("HQ") not in pairs and frozenset("HR") in pairs


def _plant_mech(store, nid, tag, claim, status=NodeStatus.SCREENING, ts=0.0):
    store.put_hypothesis(HypothesisRecord(id=f"h-{nid}", task_id=store.task_id, kind=HypothesisKind.TRIAL, claim=claim,
                                          mechanism_tag=tag, family="objective", predictions={}, falsification="f"))
    store.put_node(Node(id=nid, task_id=store.task_id, commit=f"c-{nid}", parent_id=ROOT_NODE_ID,
                        strategy=Strategy.MOONSHOT, operator=Operator.DRAFT, family="objective", mechanism_tag=tag,
                        diff_hash=f"h-{nid}", status=status, hypothesis_id=f"h-{nid}", created_ts=ts))


def _fresh(ctx, store, nid, delta, seed=101):
    """A confirm run of ``nid`` and its parent's (root) run at ``seed``: its fresh own increment is ``delta``."""
    from loop.contracts import Run, RunStatus, Stage
    for node, stage, v in ((ROOT_NODE_ID, Stage.BASELINE, 0.5), (nid, Stage.CONFIRM, 0.5 + delta)):
        store.put_run(Run(id=Run.make_id(node, stage, seed), task_id=store.task_id, node_id=node, seed=seed, stage=stage,
                          status=RunStatus.DONE, scores={a: v for a in ctx.cfg.axes}))


def test_a_moonshot_repeating_live_mechanisms_must_answer_them(ctx, store):
    """The live CLS-decorrelation cluster (n0386 / n0392 / n0394): the third copy of a mechanism two live nodes
    already test is rejected near_dup_mechanism unless it names both (no postmortem is waited for)."""
    c, now = _on(ctx), 1e6
    _plant_mech(store, "n0386", "cls_decorr_antidimcollapse", "decorrelate the CLS covariance (Barlow style) "
                "against dimensional collapse of the cls token", ts=now - 3600)
    _plant_mech(store, "n0392", "cls_covariance_decorrelation", "penalise the CLS covariance off-diagonal: "
                "decorrelation against dimensional collapse", ts=now - 1800)
    _plant_mech(store, "n0393", "stain_mixup_views", "mix HED stain statistics across views", ts=now - 60)
    _plant_mech(store, "n0100", "cls_decorr_old", "decorrelate CLS covariance", status=NodeStatus.CRASHED, ts=0.0)
    for i in range(6):                                    # background vocabulary for the IDF
        _plant_mech(store, f"n02{i:02d}", f"other_{i}", f"an unrelated jepa panel change number {i} on tiles", ts=now)
    cand = dataclasses.replace(_cand(diff=CODE_DIFF, paths=("train.py",), cfg={}, family="objective",
                                     claim="Barlow-style CLS covariance decorrelation against dimensional collapse"))
    cand = dataclasses.replace(cand, hypothesis=dataclasses.replace(cand.hypothesis, mechanism_tag="barlow_cls_decorrelation"))
    near = [n.id for n, _ in NV.mech_dups(c, store, cand, now)]
    assert sorted(near) == ["n0386", "n0392"]
    assert "n0393" not in near and "n0100" not in near           # unrelated; closed (not dead) and older than 48 h
    rule = lambda cd, s=Strategy.MOONSHOT: NV.mech_dup_rule(c, store, cd, s, now)   # noqa: E731
    got = rule(cand)
    assert got[0] == "near_dup_mechanism" and "n0386" in got[1] and "postmortem" not in got[1]
    assert rule(cand, Strategy.EXPLORE)[0] == "near_dup_mechanism"                   # every ideation arm
    assert rule(cand, Strategy.CROSSOVER) == ("", "") and rule(cand, Strategy.DESIGN) == ("", "")
    assert NV.mech_dup_rule(c, store, cand, Strategy.EXPLOIT, now, operator=Operator.DEBUG) == ("", "")
    child = dataclasses.replace(cand, parent_commit="c-n0392")                      # its own lineage refines
    assert rule(child, Strategy.EXPLOIT)[0] == "near_dup_mechanism"                  # n0386 is still another one
    store.update_node("n0386", parent_id="n0392")
    assert rule(dataclasses.replace(cand, parent_commit="c-n0386"), Strategy.EXPLOIT) == ("", "")
    store.update_node("n0386", parent_id=ROOT_NODE_ID)
    assert NV.mech_dup_rule(_on(ctx, mech_dup_cos=0.0), store, cand, Strategy.MOONSHOT, now) == ("", "")
    answered = dataclasses.replace(cand, hypothesis=dataclasses.replace(
        cand.hypothesis, claim=cand.hypothesis.claim + "; unlike n0386 and n0392 (too weak a penalty) it whitens"))
    assert rule(answered) == ("", "") and rule(cand)[0] == "near_dup_mechanism"     # named: answered
    assert intake.diagnose(c, store, cand, operator=Operator.DRAFT, embedder=HashingEmbedder(),
                           strategy=Strategy.MOONSHOT)[0] in ("near_dup_mechanism", "near_dup")


def test_a_dead_mechanism_of_any_age_must_be_answered(ctx, store):
    """Live 2026-10-03: n0583 pcgrad_late_block_surgery was the 7th PCGrad node (n0087 ... n0283 all vetoed /
    screened out, weeks old). A DEAD node (screened out / vetoed / not confirmed, or a loss / null postmortem) counts
    at any age: rejected dead_mechanism unless the candidate names it and says what differs (after its postmortem);
    an old CONFIRMED or crashed node does not count."""
    from loop.contracts import Postmortem
    c, now = _on(ctx), 1e7
    _plant_mech(store, "n0087", "pcgrad_conflict_resolution", "PCGrad gradient surgery between conflicting DINO and JEPA gradients",
                status=NodeStatus.VETOED, ts=0.0)
    _plant_mech(store, "n0283", "deep_block_pcgrad", "PCGrad gradient surgery on the deep blocks",
                status=NodeStatus.SCREENED_OUT, ts=10.0)
    _plant_mech(store, "n0300", "pcgrad_cls_surgery", "PCGrad surgery on the CLS head", status=NodeStatus.CONFIRMED, ts=0.0)
    for i in range(6):
        _plant_mech(store, f"n02{i:02d}", f"other_{i}", f"an unrelated jepa panel change number {i} on tiles", ts=now)
    cand = _cand(diff=CODE_DIFF, paths=("train.py",), cfg={}, family="training",
                 claim="PCGrad gradient surgery between the DINO and JEPA gradients on the late blocks")
    cand = dataclasses.replace(cand, hypothesis=dataclasses.replace(cand.hypothesis, mechanism_tag="pcgrad_late_block_surgery"))
    assert NV.dead(store) == {"n0087", "n0283"}
    assert sorted(n.id for n, _ in NV.mech_dups(c, store, cand, now)) == ["n0087", "n0283"]
    cold = NV.mech_dup_rule(c, store, cand, Strategy.EXPLORE, now)      # screen-only failures: COLD, advisory
    assert cold[0] == "" and cold[1].startswith("COLD") and "n0087" in cold[1]
    _fresh(c, store, "n0283", -0.002)                                   # its best fresh-seed evidence <= 0: DEAD
    got = NV.mech_dup_rule(c, store, cand, Strategy.EXPLORE, now)
    assert got[0] == "dead_mechanism" and "n0087" in got[1] and "n0283" in got[1]
    assert NV.mech_dup_rule(c, store, cand, Strategy.MOONSHOT, now)[0] == "dead_mechanism"
    answered = dataclasses.replace(cand, hypothesis=dataclasses.replace(
        cand.hypothesis, claim=cand.hypothesis.claim + "; unlike n0087 / n0283 it projects only the JEPA side"))
    for nid in ("n0087", "n0283"):
        store.put_postmortem(Postmortem(f"pm-{nid}", store.task_id, nid, "loss", "e", "m", LeverTags()))
    assert NV.mech_dup_rule(c, store, answered, Strategy.EXPLORE, now) == ("", "")
    _plant_mech(store, "n0583", "pcgrad_late_block_surgery", cand.hypothesis.claim, status=NodeStatus.QUEUED, ts=now)
    assert intake.recheck(c, store, None, store.get_node("n0583"))[0] == "dead_mechanism"   # the re-intake on deploy


def test_a_mechanism_is_dead_only_on_fresh_seed_evidence(ctx, store):
    """2026-10-06: tissue-coverage sampling was blocked 4x as dead_mechanism while n0685 (not confirmed, E 4.07, all 3
    confirm seeds positive) carried the strongest unported signal. DEAD = best fresh-seed own increment <= 0, or >= 2
    failed fresh-seed confirmations (fresh <= 0); a positive one that missed the bar is not a failure."""
    c = _on(ctx)
    for nid, st in (("n1", NodeStatus.NOT_CONFIRMED), ("n2", NodeStatus.SCREENED_OUT), ("n3", NodeStatus.NOT_CONFIRMED)):
        _plant_mech(store, nid, "tissue_coverage", "c", status=st)
    _fresh(c, store, "n1", +0.003)
    assert NV.mechanism_dead(c, store, {"n1", "n2"})[0] is False             # positive fresh evidence: cold
    assert NV.mechanism_dead(c, store, {"n2"})[0] is False                   # screen-only: cold
    assert NV.mechanism_dead(c, store, {"n1", "n2", "n3"})[0] is False       # n3 failed (no fresh), n1 did not
    _fresh(c, store, "n1", -0.001, seed=102)
    _fresh(c, store, "n1", -0.003, seed=103)                                 # n1's fresh mean now <= 0
    dead, why = NV.mechanism_dead(c, store, {"n1", "n2", "n3"})
    assert dead and "2 failed confirmation(s)" in why


def test_the_researcher_sees_the_most_retried_dead_mechanisms(ctx, store):
    """The context pack lists each retried dead mechanism once (attempts, best own increment, last node), most
    retried first; a confirmed or single dead node is not one."""
    c = _on(ctx)
    for i, (tag, claim) in enumerate([("pcgrad_conflict_resolution", "PCGrad gradient surgery on DINO vs JEPA"),
                                      ("deep_block_pcgrad", "PCGrad gradient surgery on the deep blocks"),
                                      ("pcgrad_cls_surgery", "PCGrad gradient surgery on the CLS head"),
                                      ("stem_lr_unfreeze", "unfreeze the stem learning rate"),
                                      ("raise_stem_lr", "raise the stem learning rate")]):
        _plant_mech(store, f"n00{i}0", tag, claim, status=NodeStatus.SCREENED_OUT, ts=float(i))
    _plant_mech(store, "n0090", "lonely_idea", "something else entirely", status=NodeStatus.VETOED, ts=9.0)
    _plant_mech(store, "n0091", "pcgrad_ok", "PCGrad gradient surgery that worked", status=NodeStatus.CONFIRMED, ts=9.0)
    for i in range(6):
        _plant_mech(store, f"n02{i:02d}", f"other_{i}", f"an unrelated jepa panel change number {i} on tiles", ts=20.0)
    snap = SimpleNamespace(nodes={"n0010": SimpleNamespace(increment=SimpleNamespace(mean=0.0012))})
    rows = NV.dead_block(c, store, snap).splitlines()
    assert len(rows) == 2 and "3 attempts, best own increment +0.0012, last n0020" in rows[0] and "n0091" not in rows[0]
    assert "2 attempts" in rows[1] and "last n0040" in rows[1]
    extra = dataclasses.replace(c, cfg=dataclasses.replace(c.cfg, novelty=dataclasses.replace(
        c.cfg.novelty, dead_extra=("optimizer swaps (Muon / Lion)",))))
    assert NV.dead_block(extra, store, snap).splitlines()[-1] == "- optimizer swaps (Muon / Lion) (operator: low yield, deprioritized)"


def test_tag_spellings_and_families_never_evade_the_mechanism_dedup(ctx, store, monkeypatch):
    """Live: n0420 objective/whitened_jepa_target vs n0430 training/whiten_jepa_target. Tags are normalized
    (lowercase, simple suffixes stemmed, family-agnostic); the same normalized tag is a duplicate (cosine 1),
    and a MOONSHOT is rejected at ONE such node (or one at cosine >= MOON_ONE_COS), else at >= 2 as before."""
    assert NV.norm_tag("whitened_jepa_target") == NV.norm_tag("Whiten-JEPA-targets") == "whiten_jepa_target"
    c, now = _on(ctx), 1e6
    _plant_mech(store, "n0420", "whitened_jepa_target", "ZCA-whiten the teacher targets", ts=now - 60)
    for i in range(6):
        _plant_mech(store, f"n02{i:02d}", f"other_{i}", f"an unrelated jepa panel change number {i} on tiles", ts=now)
    cand = _cand(diff=CODE_DIFF, paths=("train.py",), cfg={}, family="training", claim="decorrelate the predictor goal")
    cand = dataclasses.replace(cand, hypothesis=dataclasses.replace(cand.hypothesis, mechanism_tag="whiten_jepa_target"))
    assert [(n.id, c_) for n, c_ in NV.mech_dups(c, store, cand, now)] == [("n0420", 1.0)]
    assert NV.mech_dup_rule(c, store, cand, Strategy.MOONSHOT, now)[0] == "near_dup_mechanism"
    other = dataclasses.replace(cand, hypothesis=dataclasses.replace(cand.hypothesis, mechanism_tag="zca_teacher",
                                                                     claim="ZCA-whiten the teacher targets per batch"))
    near = NV.mech_dups(c, store, other, now)
    assert len(near) == 1 and near[0][1] >= NV.MOON_ONE_COS                    # one close node: a dup
    assert NV.mech_dup_rule(c, store, other, Strategy.MOONSHOT, now)[0] == "near_dup_mechanism"
    monkeypatch.setattr(NV, "MOON_ONE_COS", 0.9)                               # one merely-near node: the
    assert NV.mech_dup_rule(c, store, other, Strategy.MOONSHOT, now) == ("", "")   # moonshot keeps >= 2
    assert NV.mech_dup_rule(c, store, other, Strategy.EXPLORE, now)[0] == "near_dup_mechanism"


def test_ideation_prompts_carry_the_computed_regime(ctx, store, tmp_path):
    import json
    from loop.contracts import LeverEffect, Run, RunStatus, Stage
    store.ensure_root("c0")
    assert NV.regime_block(store, SimpleNamespace(levers=(), nodes={})) == ""
    for i, ff in enumerate((0.28, 0.287, 0.29)):
        out = tmp_path / f"r{i}"
        out.mkdir()
        (out / "summary.json").write_text(json.dumps({"stop_reason": "max_train_samples", "flop_fraction": ff,
                                                      "sample_fraction": 0.98}))
        store.put_run(Run(id=f"root:screen:{i}", task_id=store.task_id, node_id=ROOT_NODE_ID, seed=i,
                          stage=Stage.SCREEN, status=RunStatus.DONE, output_dir=str(out)))
    _plant_mech(store, "n0112", "schedule_rekey_to_samples", "re-key the schedule", status=NodeStatus.SCREENED_OUT)
    snap = SimpleNamespace(levers=(LeverEffect("effective_lr", "panel", 0.004, 0.001, 0.99, 12),), nodes={},
                           incumbent_id=ROOT_NODE_ID, hypotheses={})
    text = NV.regime_block(store, snap)
    assert "3/3 recent runs stop on max_train_samples at a median 0.287" in text and "71%" in text
    assert "still at 81%" in text and "effective_lr lever measured: +0.00400" in text and "n0112 schedule_rekey_to_samples" in text
    assert "Open directions" not in text and "do not" not in text          # computed facts only, no steering
    req = CandidateRequest(id="r", task_id=store.task_id, strategy=Strategy.MOONSHOT, operator=Operator.DRAFT,
                           island="main", parent_ids=(ROOT_NODE_ID,), brief="b", brief_kind="moonshot")
    assert "Training regime" in NV.block(_on(ctx), store, snap, req)
    assert "Training regime" not in NV.block(_on(ctx), store, snap, dataclasses.replace(req, strategy=Strategy.EXPLOIT))


WSD = """diff --git a/train.py b/train.py
--- a/train.py
+++ b/train.py
@@ -647,7 +647,12 @@ def main():
-                lr = cosine_schedule(dino_cfg["lr"], dino_cfg["lr_min"], frac)
+                WSD_DECAY_FRACTION = 0.2
+                decay_start = 1.0 - WSD_DECAY_FRACTION  # decay over the last fifth
+                if frac < decay_start:
+                    lr = dino_cfg["lr"]
+                else:
+                    lr = cosine_schedule(dino_cfg["lr"], dino_cfg["lr_min"], (frac - decay_start) / WSD_DECAY_FRACTION)
             wd = cosine_schedule(0.04, 0.2, frac)
"""
BETA2 = """diff --git a/train.py b/train.py
--- a/train.py
+++ b/train.py
@@ -651,6 +651,8 @@ def main():
             wd = cosine_schedule(0.04, 0.2, frac)
+            beta2_now = cosine_schedule(0.999, 0.98, frac)
             for group in opt.param_groups:
+                group["betas"] = (0.9, beta2_now)
"""


def test_a_schedule_change_beyond_the_reachable_horizon_is_rejected(ctx, store, tmp_path):
    """Runs stop at FLOP frac ~0.29: n0577's WSD decays from frac 0.8 (never), n0593's beta2 cosine over frac moves
    19% of its range (0.999 -> 0.98 reaches 0.995), n0021's terminal weight decay barely moves; a breakpoint inside
    the horizon, a start-value change or a new mechanism that merely anneals is fine."""
    import json
    from loop.contracts import Run, RunStatus, Stage
    store.ensure_root("c0")
    assert NV.unreachable(store, WSD) == ""                                   # no regime known: no verdict
    for i, ff in enumerate((0.28, 0.287, 0.29)):
        out = tmp_path / f"r{i}"
        out.mkdir()
        (out / "summary.json").write_text(json.dumps({"stop_reason": "max_train_samples", "flop_fraction": ff,
                                                      "sample_fraction": 0.98}))
        store.put_run(Run(id=f"root:screen:{i}", task_id=store.task_id, node_id=ROOT_NODE_ID, seed=i,
                          stage=Stage.SCREEN, status=RunStatus.DONE, output_dir=str(out)))
    why = NV.unreachable(store, WSD)
    assert "FLOP frac 0.8" in why and "0.287" in why and "frac / 0.29" in why
    assert NV.unreachable(store, WSD.replace("0.2\n", "0.75\n")) == ""      # decays from 0.25: reached
    assert "reaches 0.9954" in NV.unreachable(store, BETA2)
    assert NV.unreachable(store, BETA2 + "+            ema = EMA(model, 0.998)\n") == ""   # a mechanism + its anneal
    end = "-            wd = cosine_schedule(0.04, 0.2, frac)\n+            wd = cosine_schedule(0.04, 0.3, frac)\n"
    start = "-            wd = cosine_schedule(0.04, 0.2, frac)\n+            wd = cosine_schedule(0.08, 0.2, frac)\n"
    assert "0.04 -> 0.3" in NV.unreachable(store, end) and NV.unreachable(store, start) == ""
    assert NV.unreachable(store, "+    if sample_fraction > 1.5:\n+        x = 1\n") != ""
    cand = _cand(diff=WSD, paths=("train.py",), cfg={}, family="training", claim="WSD trapezoidal LR")
    assert intake.diagnose(_on(ctx), store, cand, operator=Operator.DRAFT, embedder=HashingEmbedder(),
                           strategy=Strategy.REVISIT)[0] == "unreachable_schedule"


def test_a_dead_node_sharing_one_rare_tag_word_is_not_a_dup_without_a_second(ctx, store):
    """Live 2026-10-03: dropkey_attn_logits was rejected dead_mechanism on attn_entropy_target_controller (one shared
    rare tag word 'attn', half the cosine), tile_loss_truncation_jepa on filter_low_tissue_tiles. A rare word counts
    only with one more shared tag word (decouple_stem_lr vs unthrottle_stem_lr, dino_jepa_handoff vs weighted_dino_jepa)."""
    c, now = _on(ctx), 1e7
    _plant_mech(store, "n0405", "attn_entropy_target_controller", "attention entropy controller on the masking targets",
                status=NodeStatus.NOT_CONFIRMED, ts=0.0)
    _plant_mech(store, "n0256", "unthrottle_stem_lr", "raise the stem learning rate multiplier", status=NodeStatus.SCREENED_OUT)
    for i in range(30):
        _plant_mech(store, f"n02{i:02d}", f"other_{i}", f"an unrelated jepa panel change number {i} on tiles", ts=now)

    def near(tag, claim):
        cand = _cand(diff=CODE_DIFF, paths=("train.py",), cfg={}, family="training", claim=claim)
        cand = dataclasses.replace(cand, hypothesis=dataclasses.replace(cand.hypothesis, mechanism_tag=tag))
        return [n.id for n, _ in NV.mech_dups(c, store, cand, now)]
    assert near("dropkey_attn_logits", "randomly drop attention keys before the softmax on the masking logits") == []
    assert near("attn_dropkey_entropy", "randomly drop attention keys before the softmax on the masking logits") \
        == ["n0405"]                                               # two shared tag words: still a dup
    assert near("decouple_stem_lr", "decouple the stem multiplier from depth decay") == ["n0256"]


def test_the_researcher_check_reports_a_dead_mechanism_before_any_edit_and_finish_bounces_on_it(ctx, store):
    """Live 2026-10-03: 20 sessions ended at intake dead_mechanism; the researcher's check never saw the
    mechanism_tag (nor the handoff), so it said OK. check now runs the SAME rule on the declared tag + claim, even
    before an edit, listing the nodes to name with their postmortems; finish bounces on it in-session."""
    from loop.agents.researcher import LLMResearcher, WorktreeTools
    from loop.contracts import Postmortem
    c, now = _on(ctx), 1e7
    _plant_mech(store, "n0087", "pcgrad_conflict_resolution", "PCGrad gradient surgery between conflicting DINO and JEPA gradients",
                status=NodeStatus.VETOED, ts=0.0)
    store.put_postmortem(Postmortem("pm-n0087", store.task_id, "n0087", "loss", "projection starved the JEPA side", "m",
                                    LeverTags()))
    for i in range(6):
        _plant_mech(store, f"n02{i:02d}", f"other_{i}", f"an unrelated jepa panel change number {i} on tiles", ts=now)
    _fresh(c, store, "n0087", -0.001)
    diff = {"text": ""}
    adapter = SimpleNamespace(diff=lambda wt: (diff["text"], ["train.py"] if diff["text"] else [], {}))
    res = LLMResearcher(c, None, adapter)
    req = CandidateRequest(id="r", task_id=store.task_id, strategy=Strategy.EXPLORE, operator=Operator.DRAFT,
                           island="i", parent_ids=(ROOT_NODE_ID,), brief="")
    wt = SimpleNamespace(path="/tmp", branch="b", base_commit="c0")
    claim = "PCGrad gradient surgery between the DINO and JEPA gradients on the late blocks"
    early = res.check(req, store, wt, family="optimization", claim=claim, mechanism_tag="pcgrad_late_block_surgery")
    assert early.startswith("REJECT dead_mechanism") and "n0087" in early and "projection starved" in early
    diff["text"] = CODE_DIFF
    tools = WorktreeTools(None, wt, writable=(), locked=(),
                          checker=lambda **kw: res.check(req, store, wt, **kw))
    fin = {"hypothesis": {"family": "optimization", "claim": claim, "mechanism_tag": "pcgrad_late_block_surgery",
                          "falsification": "f"}, "handoff": {"actions": "a", "learnings": "l", "concerns": "c"}}
    assert "REJECT dead_mechanism" in tools.check_finish(fin) and tools.bounces == 1
    fin["handoff"]["learnings"] = "unlike n0087 it projects only the JEPA side, answering its postmortem"
    assert tools.check_finish(fin) == ""                                # named in the handoff, as intake reads
