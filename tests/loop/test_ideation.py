"""First-principles ideation (operator 2026-10-06): component discovery + dossiers, the share guard, the
challenge-the-base share, plan-before-code idea slates, multi-variant branched plans, the zero-init exemption.
Run: uv run pytest tests/loop -q --confcutdir=tests/loop"""

from __future__ import annotations

import dataclasses
import json
import random
import subprocess
import time
from pathlib import Path

import pytest

from loop.agents import dossiers as DS
from loop.contracts import (PANEL, LeverTags, Node, NodePosterior, NodeStatus, Operator, Strategy,
                            WorldSnapshot)

TRAIN = '''import json
import torch
import torch.nn.functional as F


# Cosine schedule from start to end.
def cosine_schedule(start, end, frac):
    return end + 0.5 * (start - end) * (1 + frac)


def plumbing_log(x):
    print(x)
    return x


def main(cfg):
    meta = json.loads(open(cfg["data"]["dir"] + "/meta.json").read())
    a = 1
    b = 2
    c = 3
    d = 4
    # Auxiliary factor loss weighted by a constant nobody exposed.
    w = 0.03 * cfg["loss"]["aux_weight"]
    loss = F.mse_loss(torch.zeros(1), torch.zeros(1)) * w
    e = 5
    f = 6
    # Post-hoc whitening fit after training.
    for _ in range(3):
        z = torch.randn(4, 4)
        z = z @ z.t() * 0.85
        g = 7
        h = 8
''' + "\n".join(f"    q{i} = {i}" for i in range(60)) + "\n    return loss\n"
CONFIG = "train:\n  lr: 0.001\n  warmup: 0.1\nloss:\n  aux_weight: 1.0\ndata:\n  dir: {data}\naug:\n  p: 0.5\n"


@pytest.fixture
def repo(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "meta.json").write_text(json.dumps({"discrete": {"subtype": {}, "site": {}, "organ": {}},
                                                "n": {"subtype": 3}}))
    r = tmp_path / "repo"
    r.mkdir()
    (r / "train.py").write_text(TRAIN)
    (r / "config.yaml").write_text(CONFIG.format(data=data))
    (r / "model.py").write_text("class ViT:\n    def __init__(self):\n        self.w = 1\n        self.b = 2\n\n"
                                "    def forward(self, x):\n        y = x * 2\n        return y.mean()\n")
    for cmd in (["init", "-q"], ["add", "."], ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "base"]):
        subprocess.run(["git", "-C", str(r), *cmd], check=True, capture_output=True)
    sha = subprocess.run(["git", "-C", str(r), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    return r, sha


@pytest.fixture
def ictx(ctx, repo):
    DS._CACHE.clear()
    return dataclasses.replace(ctx, spec=ctx.spec.model_copy(update={"repo": str(repo[0])}))


def _node(store, nid, *, tag="m", family="objective", status=NodeStatus.SCREENED_OUT, strategy=Strategy.EXPLORE,
          parent="root", diff="", cfg=None, ts=None, notes=""):
    """``diff``: a diff FILE path."""
    n = Node(id=nid, task_id=store.task_id, commit=f"c-{nid}", parent_id=parent, strategy=strategy,
             operator=Operator.DRAFT, family=family, mechanism_tag=tag, diff_hash=f"h-{nid}", status=status,
             created_ts=ts or time.time(), updated_ts=ts or time.time(), notes=notes, diff_path=diff)
    store.put_node(n)
    if cfg:
        store.set_state(f"config_diff:{nid}", cfg)
    return n


def _snap(posts=(), inc="root"):
    return WorldSnapshot(task_id="fake-loop", n_scored=0, sigma={PANEL: 0.004}, variance={},
                         nodes={p.node_id: p for p in posts}, incumbent_id=inc, family_means={},
                         levers=(), hypotheses={})


# ------------------------------------------------------------------ discovery + dossier
def test_components_are_discovered_from_code_and_config_never_a_hand_list(ictx, repo):
    comps = DS.components(ictx, repo[1])
    ids = [c.id for c in comps]
    assert "train.py::cosine_schedule" in ids
    assert not any("plumbing_log" in i for i in ids)                       # plumbing without a mechanism
    assert not any(i.endswith("ViT.__init__") for i in ids)                # model init: the operator's exception
    assert "model.py::ViT.forward" in ids
    aux = next(c for c in comps if "Auxiliary factor loss" in c.title)
    assert aux.keys == ("loss.aux_weight",) and "0.03" in aux.consts and "objective" in aux.families
    assert any("Post-hoc whitening" in c.title for c in comps)             # a comment-headed block of a long fn
    start = next(c for c in comps if c.id.startswith("train.py::main#main (start)"))
    assert start.data == ("meta.json",)
    assert any(c.id == "config::aug" for c in comps)                       # a config section no unit reads


def test_dossier_lists_history_open_questions_and_the_unused_data_factors(ictx, repo, store):
    store.ensure_root("c0")
    comps = DS.components(ictx, repo[1])
    start = next(c for c in comps if c.id.startswith("train.py::main#main (start)"))
    aux = next(c for c in comps if "Auxiliary factor loss" in c.title)
    _node(store, "n0001", tag="aux_weight_up", cfg={"loss.aux_weight": 2.0})
    hist = DS.history(ictx, store, _snap(), comps)
    assert [n.id for n in hist[aux.id]] == ["n0001"]
    cfg = DS.config_values(ictx, repo[1])
    text = DS.dossier(ictx, store, _snap(), aux, hist[aux.id], cfg, commit=repo[1])
    assert "n0001 aux_weight_up screened_out" in text and "tried at ONE other value only" in text
    assert "hard-coded constants" in text and "0.03" in text
    data = DS.dossier(ictx, store, _snap(), start, hist[start.id], cfg, commit=repo[1])
    assert "meta.json" in data and "organ, site, subtype" in data          # the full list: the worktree can't read it
    assert "meta.json[discrete] offers 3 factors, the recipe uses 0; unused: organ, site, subtype" in data


def test_rotation_prefers_the_under_explored_component(ictx, repo, store):
    store.ensure_root("c0")
    comps = DS.components(ictx, repo[1])
    hist = {c.id: [] for c in comps}
    hist[comps[0].id] = [object()] * 50
    rng = random.Random(1)
    picks = [DS.pick(ictx, store, _snap(), comps, hist, rng).id for _ in range(400)]
    assert picks.count(comps[0].id) < 400 / len(comps) / 4


def test_share_guard_caps_a_theme_unless_the_allocation_signal_backs_it(ctx, store, monkeypatch):
    store.ensure_root("c0")
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(
        ctx.cfg, ideation=dataclasses.replace(ctx.cfg.ideation, share_window=10, max_share=0.35)))
    for i in range(10):
        _node(store, f"n{i:04d}", tag="predictor_jepa_depth" if i < 6 else f"other_thing_{i}", ts=1000.0 + i,
              family="objective" if i % 2 else "augmentation")
    theme, why = DS.share_guard(ctx, store, _snap(), "jepa_target_whiten", "augmentation")
    assert theme == "jepa" and "60%" in why
    assert DS.share_guard(ctx, store, _snap(), "fresh_idea", "optimization") == ("", "")
    monkeypatch.setattr(DS, "_alloc", lambda store, snap: {"jepa": 0.9})
    assert DS.share_guard(ctx, store, _snap(), "jepa_target_whiten", "augmentation") == ("", "")


def test_base_share_draws_a_component_brief_with_its_dossier(ictx, repo, store):
    from loop.agents import briefs
    from loop.select import PortfolioSelector
    from tests.loop.test_select import StubWorld
    store.ensure_root(repo[1])
    ictx = dataclasses.replace(ictx, cfg=dataclasses.replace(
        ictx.cfg, ideation=dataclasses.replace(ictx.cfg.ideation, base_share=1.0)))
    s = PortfolioSelector(ictx, StubWorld(), rng=random.Random(3),
                          brief_fn=lambda kind, st, sn, **kw: briefs.brief(kind, st, sn, ctx=ictx, **kw))
    req = s.next_request(store, _snap())
    assert req.brief_kind == "base" and req.strategy is Strategy.EXPLORE and req.operator is Operator.DRAFT
    assert req.parent_ids == ("root",) and "CHALLENGE-THE-BASE" in req.brief and "COMPONENT `" in req.brief
    comp = store.events("base_challenge")[-1]["component"]
    assert f"`{comp}`" in req.brief and "The other components" in req.brief
    off = dataclasses.replace(ictx, cfg=dataclasses.replace(
        ictx.cfg, ideation=dataclasses.replace(ictx.cfg.ideation, base_share=0.0)))
    s2 = PortfolioSelector(off, StubWorld(), rng=random.Random(3))
    assert all(s2.next_request(store, _snap()).brief_kind != "base" for _ in range(10))


def test_init_compat_admits_child_only_keys_that_are_identity_at_init():
    from loop.adapters import initcompat as IC
    from tests.loop.test_adapters import GOOD
    m = json.loads(json.dumps(GOOD))
    m["keys"]["missing"] = ["readout_gate", "rn_basis"]                    # new tensors the parent never had
    assert not IC.evaluate(m).ok                                           # no identity measurement: fail closed
    m["keys"]["identity"] = {"rel_gap": 0.0, "loss_child": 2.0, "loss_parent": 2.0}
    res = IC.evaluate(m)
    assert res.ok and res.checks["checkpoint_keys"]["identity_at_init"]["ok"]
    m["keys"]["identity"] = {"rel_gap": 0.3, "loss_child": 2.4, "loss_parent": 2.0}   # changes the function
    assert not IC.evaluate(m).ok
    m["keys"]["identity"] = {"rel_gap": 0.0, "loss_child": 2.0, "loss_parent": 2.0}
    m["keys"]["unexpected"] = ["blocks.11.norm1.weight"]                   # a renamed / dropped parent tensor
    assert not IC.evaluate(m).ok
    compile(IC.SCRIPT, "check_init.py", "exec")


def test_intake_applies_the_share_guard_to_new_ideas_only(ctx, store):
    from loop.select import intake
    from tests.loop.test_select import cand
    store.ensure_root("c0")
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(
        ctx.cfg, ideation=dataclasses.replace(ctx.cfg.ideation, share_window=10, max_share=0.35)))
    for i in range(10):
        _node(store, f"n{i:04d}", tag="predictor_depth_sweep" if i < 6 else f"other_{i}", ts=1000.0 + i)
    c = cand(diff="+x\n", paths=("train.py",), family="optimization", claim="deeper predictor")
    c = dataclasses.replace(c, hypothesis=dataclasses.replace(c.hypothesis, mechanism_tag="predictor_width"))
    from loop.select.embed import HashingEmbedder
    emb = HashingEmbedder()
    rule = intake.diagnose(ctx, store, c, operator=Operator.DRAFT, embedder=emb, strategy=Strategy.EXPLORE)
    assert rule[0] == "theme_share" and "predictor" in rule[1]
    assert intake.diagnose(ctx, store, c, operator=Operator.IMPROVE, embedder=emb,
                           strategy=Strategy.EXPLOIT)[0] != "theme_share"


def test_base_challenge_nodes_do_not_kill_their_mechanism_before_a_dose_ladder(ctx, store, monkeypatch):
    from loop import novelty as NV
    from loop import tune
    from loop.select import intake
    monkeypatch.setattr(NV, "mech_dup_rule", lambda *a, **k: ("dead_mechanism", "failed on fresh seeds"))
    from tests.loop.test_select import cand
    store.ensure_root("c0")
    _node(store, "n0001", tag="site_reversal", notes="request=req-1 brief=base")
    c = cand(diff="+x\n", paths=("train.py",), family="optimization", claim="site reversal again")
    c = dataclasses.replace(c, hypothesis=dataclasses.replace(c.hypothesis, mechanism_tag="site_reversal"))
    rule, fix = intake.mech_rule(ctx, store, c, Strategy.EXPLORE, Operator.DRAFT)
    assert rule == "" and "dose ladder has not run" in fix
    bk = tune.book(store)
    bk["batches"].append({"kind": "ladder", "base": "n0001", "doses": {"-1": "a", "+1": "b"}})
    store.set_state(tune.K_TUNE, bk)
    assert intake.mech_rule(ctx, store, c, Strategy.EXPLORE, Operator.DRAFT)[0] == "dead_mechanism"


# ------------------------------------------------------------------ plan before code: the slate
def _req(rid="req-00007", strategy=Strategy.EXPLORE, kind="literature"):
    from loop.contracts import CandidateRequest
    return CandidateRequest(id=rid, task_id="fake-loop", strategy=strategy, operator=Operator.DRAFT, island="main",
                            parent_ids=("root",), brief="b", brief_kind=kind)


def _idea(tag, family, files=(), keys=(), init="unchanged", **kw):
    return {"title": tag.replace("_", " "), "mechanism_tag": tag, "family": family, "files": list(files),
            "config_keys": list(keys), "init": init, "idea": f"do {tag}", "mechanistic_model": "m",
            "bottleneck": "b", "mechanism_prediction": {"quantity": "q", "direction": "up"}, "prior_attempts": "none",
            "citations": ["https://arxiv.org/abs/1"], **kw}


def test_slate_gates_plans_before_code_ranks_and_banks_the_rest(ctx, store):
    from loop.agents import ideas as IDEAS
    from loop.select import bank as B
    store.ensure_root("c0")
    req = _req()
    got, text = IDEAS.propose(ctx, store, _snap(), req, "c0", [_idea("a_b", "objective", keys=["loss.a"])] * 2)
    assert got is None and "at least 3" in text
    slate = [_idea("locked_eval_hack", "objective", files=["eval.py"]),
             _idea("resize_backbone", "architecture", files=["model.py"], init="changes_shapes"),
             _idea("lr_and_loss", "objective", keys=["train.lr", "loss.a"]),
             _idea("new_aux_target", "objective", keys=["loss.aux"], predicted_sigmas=1.0,
                   variants=[{"name": "half", "overrides": {"loss.aux": 0.5}}]),
             _idea("zero_init_adapter", "architecture", files=["model.py"], init="zero_init")]
    got, text = IDEAS.propose(ctx, store, _snap(), req, "c0", slate)
    rules = {r["tag"]: r["rule"] for r in store.get_state(IDEAS.K_SLATES)[req.id]["ranks"]}
    assert rules == {"locked_eval_hack": "surface_locked", "resize_backbone": "init_incompatible",
                     "lr_and_loss": "multi_family", "new_aux_target": "", "zero_init_adapter": ""}
    assert got == 3 and 'idea_id = "gen-req-00007-4"' in text                      # the larger predicted size
    bank = B.pending(store, ctx.cfg)
    assert [e["id"] for e in bank] == ["gen-req-00007-5"] and bank[0]["priority"]   # banked with its plan
    assert bank[0]["plan"]["init"] == "zero_init" and "PLAN" in B.text(bank[0])
    assert store.events("idea_slate")[-1]["rejected"] == {"gen-req-00007-1": "surface_locked",
                                                          "gen-req-00007-2": "init_incompatible",
                                                          "gen-req-00007-3": "multi_family"}


def test_finish_bounces_without_the_slate_the_idea_id_or_the_first_principles_record(ctx):
    from loop.agents.researcher import WorktreeTools
    from loop.contracts import Worktree
    t = WorktreeTools(None, Worktree("/tmp/x", "c0", "b"), writable=("*",), locked=())
    t.request_id, t.need_fp = "req-1", True
    t.planner = lambda ideas: (1, "IMPLEMENT #2")
    fin = {"hypothesis": {"claim": "c", "mechanism_tag": "m", "family": "objective"}, "handoff": {}}
    assert "plan_first" in t.check_finish(fin)
    assert t.t_propose_ideas([{}, {}, {}]) == "IMPLEMENT #2" and t.chosen == 1
    assert "idea_id" in t.check_finish(fin)
    fin["hypothesis"]["idea_id"] = "gen-req-1-2"
    t.bounces = 0
    assert "first_principles" in t.check_finish(fin)
    fin["hypothesis"].update(mechanistic_model="m", bottleneck="b", prior_attempts="p",
                             mechanism_prediction={"quantity": "update ratio of block 0", "direction": "up"})
    t.bounces = 0
    assert t.check_finish(fin) == ""


def _root_runs(ctx, store, seeds=(11, 12), y=0.6):
    from loop.contracts import ROOT_NODE_ID, Run, RunStatus, Stage
    for s in seeds:
        store.put_run(Run(id=Run.make_id(ROOT_NODE_ID, Stage.BASELINE, s), task_id=store.task_id,
                          node_id=ROOT_NODE_ID, seed=s, stage=Stage.BASELINE, status=RunStatus.DONE,
                          scores={a: y for a in ctx.cfg.axes}))


def test_variants_screen_beside_their_idea_branch_on_shrunk_results_and_graduate_on_fresh_seeds(ctx, store):
    from types import SimpleNamespace
    from loop import variants as VR
    from loop.adapters.sim import SimAdapter
    from tests.loop.test_tune import _moonshot, _screened
    sim = SimAdapter(ctx)
    base = _moonshot(ctx, store, sim, "n0010", status=NodeStatus.SCREENING, cfg={"loss.temperature": 0.2},
                     strategy=Strategy.EXPLORE)
    snap = SimpleNamespace(nodes={}, sigma={PANEL: 0.004})
    ids = iter(f"n{i:04d}" for i in range(100, 200))
    plan = {"variants": [{"name": "cool", "overrides": {"loss.temperature": 0.1}, "from": "base"},
                         {"name": "hot", "overrides": {"loss.temperature": 0.4}, "from": "base"}],
            "branches": [{"when": {"metric": "base", "op": "<", "value": -1.0}, "then": [{"name": "never"}]},
                         {"when": {"metric": "hot", "op": ">", "value": 0.5},
                          "then": [{"name": "hotter", "from": "hot", "overrides": {"loss.temperature": 0.8}}]}]}
    assert VR.register(ctx, store, base.id, plan, snap)
    assert VR.tick(ctx, store, sim, snap, new_id=lambda: next(ids)) == []        # waits for the base's screen runs
    _screened(ctx, store, base.id, y=0.60)
    _root_runs(ctx, store, y=0.59)                     # the base's parent: every variant must beat it too
    acts = VR.tick(ctx, store, sim, snap, new_id=lambda: next(ids))
    pts = VR.book(store)[base.id]["points"]
    assert set(pts) == {"cool", "hot"} and "variants for n0010" in acts[0]
    for name, nid in pts.items():
        n = store.get_node(nid)
        assert n.parent_id == base.id and f"block_of={base.id}" in n.notes and n.strategy is Strategy.DESIGN
        assert store.get_state(f"config_diff:{nid}")["loss.temperature"] == {"cool": 0.1, "hot": 0.4}[name]
    for nid, y in ((pts["cool"], 0.595), (pts["hot"], 0.62)):
        _screened(ctx, store, nid, y=y)
        store.update_node(nid, status=NodeStatus.SCREENED_OUT)
    acts = VR.tick(ctx, store, sim, snap, new_id=lambda: next(ids))
    cur = VR.book(store)[base.id]
    assert "branch 1" in acts[0] and "hotter" in cur["points"] and cur["round"] == 1   # base metric unavailable: skip 0
    assert store.get_state(f"config_diff:{cur['points']['hotter']}")["loss.temperature"] == 0.8
    _screened(ctx, store, cur["points"]["hotter"], y=0.63)
    store.update_node(cur["points"]["hotter"], status=NodeStatus.SCREENED_OUT)
    acts = VR.tick(ctx, store, sim, snap, new_id=lambda: next(ids))
    cur = VR.book(store)[base.id]                     # the best (0.8) is the highest tried: one step beyond it
    assert "edge" in acts[0] and store.get_state(f"config_diff:{cur['points']['hotter_edge']}")["loss.temperature"] \
        == 1.2 and cur["status"] == "running", acts  # 0.4 -> 0.8 one more step: a temperature is not bounded by 1
    _screened(ctx, store, cur["points"]["hotter_edge"], y=0.61)
    store.update_node(cur["points"]["hotter_edge"], status=NodeStatus.SCREENED_OUT)
    acts = VR.tick(ctx, store, sim, snap, new_id=lambda: next(ids))
    cur = VR.book(store)[base.id]
    assert cur["status"] == "done" and cur["tuned"], acts
    grad = store.get_node(cur["tuned"])
    assert f"tune_of={base.id}" in grad.notes and grad.strategy is Strategy.EXPLORE   # fresh block, own verdict
    assert grad.commit == store.get_node(cur["points"]["hotter"]).commit
    assert base.id in VR.bases(store)                                                 # no second dose ladder


def test_the_driver_registers_a_finish_plan_and_screens_its_variants_on_the_ideas_own_block(tmp_path):
    from loop import variants as VR
    from loop.agents.researcher import diff_hash, set_yaml_value
    from loop.contracts import Candidate, CandidateRequest, HandoffNote, HypothesisKind, HypothesisRecord, ResearchResult, Stage
    from tests.loop.test_tune import _driver, _run_until
    ctx, driver, store, adapter = _driver(tmp_path)
    driver.tick()
    root = store.get_node("root")
    wt = adapter.prepare_worktree(parent_commit=root.commit, name="idea")
    cfg = Path(wt.path) / "config.yaml"
    cfg.write_text(set_yaml_value(cfg.read_text(), "aug.strength", 0.8))
    aug = Path(wt.path) / "aug.py"
    aug.write_text(aug.read_text() + "\n\ndef new_view(x):\n    return x\n")
    diff_text, paths, cfg_diff = adapter.diff(wt)
    commit = adapter.commit(wt, message="idea")
    adapter.reap_worktree(wt)
    req = CandidateRequest(id="req-09001", task_id=ctx.task_id, strategy=Strategy.EXPLORE, operator=Operator.DRAFT,
                           island="main", parent_ids=("root",), brief="b", brief_kind="literature")
    h = HypothesisRecord("h-x", ctx.task_id, HypothesisKind.TRIAL, "a new view", "new_view_temp", "augmentation", {},
                         "no gain")
    res = ResearchResult("ok", Candidate(req.id, ctx.task_id, commit, root.commit, diff_text, diff_hash(diff_text),
                                         tuple(paths), dict(cfg_diff), h, HandoffNote("", ctx.task_id, req.id, "a", "", "")),
                         plan={"first_principles": {"mechanistic_model": "m"}, "idea_id": "gen-req-09001-1",
                               "variants": [{"name": "lo", "overrides": {"aug.strength": 0.4}, "from": "base"}],
                               "branches": []})
    out = driver._intake(req, res)
    nid = out.split()[1]
    assert out.startswith("queued") and "idea=gen-req-09001-1 fp=1" in store.get_node(nid).notes
    assert store.get_state(f"ideation:{nid}")["idea_id"] == "gen-req-09001-1"
    assert _run_until(driver, store, lambda: VR.book(store)[nid]["status"] == "done", n=150)
    pt = store.get_node(VR.book(store)[nid]["points"]["lo"])
    base_seeds = {r.seed for r in store.runs(node_id=nid, stage=Stage.SCREEN)}
    assert {r.seed for r in store.runs(node_id=pt.id)} == base_seeds                  # the idea's own block
    assert not {r.seed for r in store.runs(node_id=nid, stage=Stage.PARENT)} - base_seeds   # no extra controls


def test_variants_and_branches_are_validated_against_the_config(ctx):
    from loop.agents import ideas as IDEAS
    cfg = {"loss.a": 1.0, "train.lr": 0.001, "eval.seed": 1, "aug.flip": True}
    vs, probs = IDEAS.validate_variants(ctx, cfg, [
        {"name": "lo", "overrides": {"loss.a": 0.5}}, {"name": "lr", "overrides": {"train.lr": 0.002, "eval.seed": 2}},
        {"name": "bad", "overrides": {"nope": 1, "aug.flip": "yes"}}, {"name": "same", "overrides": {"loss.a": 1.0}}],
        limit=4)
    assert [v["name"] for v in vs] == ["lo", "lr"] and vs[1]["overrides"] == {"train.lr": 0.002}
    assert any("eval.seed" in p for p in probs) and any(p.startswith("bad") for p in probs)
    plan = IDEAS.validate_plan([{"when": {"metric": "base", "op": "<", "value": -1}, "then": [{"name": "x"}]},
                                {"when": {"metric": "base", "op": "!=", "value": 0}, "then": []}])
    assert len(plan) == 1 and plan[0]["when"]["op"] == "<"


def test_live_plan_gate_false_rejects_now_pass_on_the_nanopath_v2_families():
    """2026-10-07 live plan-gate rejects (18 sessions since 04:02Z: 19 unmapped_path, 5 surface_not_writable,
    2 multi_family; check tool 5 multi_family, 2 unmapped_path): annotated paths, brand-new keys and a mechanism's
    code + knob in two families pass; two families' knobs still do not."""
    from types import SimpleNamespace

    from loop.agents import ideas as IDEAS
    from loop.select import intake
    from loop.spec import load_context
    ctx = load_context(Path(__file__).resolve().parents[2] / "tasks/nanopath_v2.yaml", require_measured=False)

    def rule(family, files, keys, strategy=Strategy.EXPLORE):
        idea = {"family": family, "files": files, "config_keys": keys, "mechanism_tag": "m"}
        for f in map(IDEAS.plan_path, files):
            if not intake.on_surface(f.partition("::")[0], ctx.spec.surface.writable):
                return "surface_not_writable"
        probe = IDEAS._probe(ctx, SimpleNamespace(id="r"), "c", idea)
        fams, problem = intake.derive_families(ctx, probe)
        return problem or intake.family_rule(ctx, None, None, probe, Operator.DRAFT, strategy, fams)[0]

    assert IDEAS.plan_path("model.py (VIT_VARIANTS, robust_norm flag)") == "model.py"
    assert IDEAS.plan_path("train.py::main training-step (line ~774)") == "train.py::main training-step (line ~774)"
    # init_swap_vitb (n1726's plan): surface_not_writable on the annotation; keys of 3 families (an init swap)
    assert rule("architecture", ["model.py (VIT_VARIANTS, robust_norm flag)", "configs/main.yaml"],
                ["model.type", "train.local_views", "dino.lr"]) == ""
    # sam_sharpness / agc: brand-new dino.* knobs read by train.py were unmapped_path
    assert rule("regularization", ["train.py (step: ascent perturb, re-forward/backward, descent)",
                                   "configs/main.yaml"], ["dino.sam_rho"]) == ""
    assert rule("optimization", ["train.py::main training-step (line ~774 nn.utils.clip_grad_norm_)"],
                ["dino.agc_lambda"]) == ""
    assert rule("data_curation", ["train.py::main#Arrow shards"], ["data.calibration_spread"]) == ""
    # block_layerdrop: model.py code (architecture) + its dino.drop_path_rate knob (regularization) was multi_family
    assert rule("regularization", ["model.py::Block.forward", "model.py::Block.__init__"], ["dino.drop_path_rate"]) == ""
    # still rejected: two families' knobs (not a moonshot / init swap), a family the diff does not touch
    assert rule("optimization", ["train.py"], ["dino.lr", "data.color_jitter"]) == "multi_family"
    assert rule("optimization", ["train.py"], ["dino.lr", "data.color_jitter"], Strategy.MOONSHOT) == ""
    assert rule("augmentation", ["model.py"], []) == "family_mismatch"


def test_a_variant_that_only_beats_a_bad_base_never_graduates(ctx, store):
    """n1729 (2026-10-07): a moonshot ~-0.039 under root; its flip_sigma_hi variant read +0.018 vs the base (credibly
    > 0 by the base-relative gate) but ~-0.021 vs root -- it graduated n1779. A variant must also credibly beat the
    base's PARENT (paired on the parent's runs at the same seeds; else via the base's own increment)."""
    from types import SimpleNamespace
    from loop import variants as VR
    from loop.adapters.sim import SimAdapter
    from loop.contracts import NodePosterior, Posterior
    from tests.loop.test_tune import _moonshot, _screened
    sim = SimAdapter(ctx)
    base = _moonshot(ctx, store, sim, "n0010", status=NodeStatus.SCREENING, cfg={"loss.temperature": 0.2},
                     strategy=Strategy.EXPLORE)
    snap = SimpleNamespace(nodes={}, sigma={PANEL: 0.0033})
    ids = iter(f"n{i:04d}" for i in range(100, 200))
    plan = {"variants": [{"name": n, "overrides": {"loss.temperature": t}, "from": "base"}
                         for n, t in (("cool", 0.1), ("hot", 0.4), ("hotter", 0.8))]}
    VR.register(ctx, store, base.id, plan, snap)
    _screened(ctx, store, base.id, y=0.632)
    _root_runs(ctx, store, y=0.671)
    VR.tick(ctx, store, sim, snap, new_id=lambda: next(ids))
    pts = VR.book(store)[base.id]["points"]
    for name, y in (("cool", 0.611), ("hot", 0.650), ("hotter", 0.608)):
        _screened(ctx, store, pts[name], y=y)
        store.update_node(pts[name], status=NodeStatus.SCREENED_OUT)
    acts = VR.tick(ctx, store, sim, snap, new_id=lambda: next(ids))
    cur = VR.book(store)[base.id]
    assert cur["status"] == "done" and not cur["tuned"], acts
    assert "hot beats n0010" in cur["why"] and "but not its parent" in cur["why"] and "raw -0.0210" in cur["why"]
    # no parent run at the seeds: the base's own increment (world model) + the variant's delta decides
    rows = {"hot": [0.018, 0.0023, 2.0]}
    inc = lambda m: SimpleNamespace(nodes={base.id: NodePosterior(base.id, Posterior(m, 0.002, 2),  # noqa: E731
                                                                  Posterior(m, 0.002, 2), {}, 0.5, 0.0, False)})
    one = {"base": base.id, "points": {"hot": "nX"}}
    assert VR._parent_rows(ctx, store, inc(-0.039), one, rows, 0.0033)["hot"][0] == pytest.approx(-0.021)
    assert VR._parent_rows(ctx, store, inc(0.0), one, rows, 0.0033)["hot"][0] == pytest.approx(0.018)


def test_a_variant_whose_shrunk_delta_clears_the_confirm_mde_graduates_though_its_raw_misses_the_null_max(ctx, store):
    from types import SimpleNamespace
    from loop import legs as LG
    from loop import variants as VR
    from loop.contracts import Posterior
    snap = SimpleNamespace(sigma={PANEL: 0.002758}, nodes={"b": NodePosterior(
        "b", Posterior(0.0, 0.0001, 2), Posterior(0.0, 0.0001, 2), {}, 0.5, 0.0, False)})
    bar = LG.shrunk_bar(ctx, store, snap)
    # 8 variants of se = bar: x reads 1.9 se, under the q0.9 null max of 8 (~2.2 se); the spread is real (tau > 0),
    # so x's shrunk delta (~1.37 se) clears the bar -- vs the base and (increment 0) vs its parent alike
    se = bar
    rows = {"x": [1.9 * se, se, 2.0], **{f"y{i}": [-1.9 * se, se, 2.0] for i in range(7)}}
    eff = VR._shrunk(rows)
    assert eff["x"]["shrunk"] >= bar and rows["x"][0] <= VR.null_max([se] * 8, ctx.cfg.ideation.null_q, seed=8)
    cur = {"base": "b", "best": "x", "points": {k: "n" + k for k in rows}, "tuned": ""}
    VR._finish(ctx, store, snap, cur, rows, eff, None, 0.0, dry=True)
    assert cur["tuned"] == "(dry)" and "null gate not passed" in cur["why"]
    off = dataclasses.replace(ctx, cfg=dataclasses.replace(
        ctx.cfg, promotion=dataclasses.replace(ctx.cfg.promotion, shrunk_grad_mde=0.0)))
    VR._finish(off, store, snap, cur, rows, eff, None, 0.0, dry=True)
    assert cur["tuned"] == ""
