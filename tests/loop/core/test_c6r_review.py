"""v3-c6r (adversarial review of b8c611a): brief_plan has no side effect and a pair is claimed at the session's start
(P7); the pre-screen hides the neighbours' values and is scored as skill over a baseline (S1) with retries (S2); the
holdout audit reads the selection seeds (H1) and LLM-facing numbers the visible composite (H2); calibration in the
prompt's own unit with per-definition base rates (C1); mechanism dedup explanations + the judge (D1); the operator
program's eval-only arms (O1); the test reserve's timeout (M1)."""

from __future__ import annotations

import dataclasses
import json
import re
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from loop.contracts import ROOT_NODE_ID
from loop.core import calibration as CAL
from loop.core import data as CD
from loop.core import ideation as ID
from loop.core import mechanisms as MS
from loop.core import pairs as PR
from loop.core import prescreen as PS

from ..conftest import InlineExecutor
from .test_c6i import _hnode, _icfg
from .test_fnf import _Research, env  # noqa: F401  (the fixture)


def _state(store):
    rows = store._read("SELECT key, value FROM state WHERE task_id=?", (store.task_id,))
    return {r["key"]: r["value"] for r in rows}, len(store.events())


# -- P7 ------------------------------------------------------------------------------------------------------------------
def _pairing(ide, store, v, monkeypatch):
    other = [n for n in v.beliefs if n not in (v.incumbent, ROOT_NODE_ID)][:2]
    pairs = [(v.incumbent, other[0]), (v.incumbent, other[1])]

    def code_pair(view, exclude=()):
        used = store.get_state(PR.K_USED) or {}
        for a, b in pairs:
            if PR.pair_key(a, b) not in used and PR.pair_key(a, b) not in exclude:
                return a, b, f"pred +0.002 {a}x{b}"
        return None
    monkeypatch.setattr(ide, "code_pair", code_pair)
    return pairs


def test_brief_plan_writes_nothing_to_the_store_and_the_start_claims_the_pair(env, monkeypatch):
    ctx, store, led, ide = env
    v = ide.view(led)
    pairs = _pairing(ide, store, v, monkeypatch)
    before = _state(store)
    plan = ide.brief_plan("crossover", v, "crossover")
    ide.brief_plan("crossover", v, "crossover")                       # a re-plan / dry run: still nothing
    assert _state(store) == before and plan["pair"] == list(pairs[0])
    assert ide._claim_pair(plan, "req-1") and store.events("pair_picked")[-1]["request_id"] == "req-1"
    assert not ide._claim_pair(plan, "req-2") and store.events("pair_taken")


def test_the_real_next_pair_has_no_side_effect(env):
    ctx, store, led, ide = env
    v = ide.view(led)
    PR.bind(ctx, store)
    before = _state(store)
    ide.code_pair(v)
    PR.next_pair(v)
    assert _state(store) == before


def test_a_rendered_brief_whose_pair_was_taken_is_re_planned_on_the_next_pair(env, monkeypatch):
    ctx, store, led, ide = env
    ide.ctx = _icfg(ctx, brief_workers=1)
    ide._executor_cls = InlineExecutor
    v = ide.view(led)
    pairs = _pairing(ide, store, v, monkeypatch)
    plan = ide.brief_plan("crossover", v, "crossover")
    fut = Future()
    fut.set_result(ide.brief_text(plan, led, v))
    ide._jobs = [{"fut": fut, "plan": plan, "inc": v.incumbent, "ts": ide.clock(), "tok": ID._view_token(v)}]
    assert PR.claim(store, *pairs[0], "brief", "req-elsewhere")       # another session took it meanwhile
    tick = SimpleNamespace(research=_Research(), dry=False)
    ide._settle(tick, led, v)
    assert tick.research.started == [] and store.events("brief_stale")[-1]["why"] == "pair_taken"
    [j] = ide._jobs
    assert j["plan"]["pair"] == list(pairs[1])                        # re-planned on the next pair
    ide._settle(tick, led, v)
    assert len(tick.research.started) == 1 and PR.pair_key(*pairs[1]) in store.get_state(PR.K_USED)


# -- S1 / S2 -------------------------------------------------------------------------------------------------------------
class _PSLLM:
    def __init__(self, fail=0):
        self.prompts, self.fail = [], fail

    def complete_json(self, role, *, system, prompt, schema):
        self.prompts.append(prompt)
        if self.fail:
            self.fail -= 1
            raise RuntimeError("qwen: TimeoutError: timed out")
        n = len(re.findall(r"(?m)^\d+\. ", prompt.split("Measured neighbours", 1)[-1]))
        return {"p_better": [0.7] * n}


def test_the_prescreen_prompt_shows_rank_order_only_best_first():
    near = [("a", -0.004, "sig a", 0.9), ("b", 0.003, "sig b", 0.8), ("c", 0.0005, "sig c", 0.7)]
    llm = _PSLLM()
    rec = PS.ask(llm, "new sig", "claim", near, m0=-0.001)
    text = llm.prompts[0]
    assert not re.search(r"[+-]0\.\d", text) and "values withheld" in text
    assert text.index("sig b") < text.index("sig c") < text.index("sig a")      # best measured first
    assert [n for n, _ in rec["neighbors"]] == ["b", "c", "a"] and rec["m0"] == -0.001


def test_prescreen_skill_is_over_the_baseline_predictor_and_no_skill_switches_it_off(env):
    ctx, store, led, ide = env
    B = ID.Belief
    # the "p > .5 iff the neighbour lost" predictor: right 3/4 here -- exactly what the baseline (m0 = 0) gets
    store.set_state(PS.K_PS, {
        "a": {"s": 0.0, "p_better": [0.9, 0.1], "neighbors": [["x", -0.002], ["y", 0.002]], "m0": 0.0},
        "b": {"s": 0.0, "p_better": [0.9, 0.1], "neighbors": [["x", -0.002], ["y", 0.001]], "m0": 0.0}})
    PS.score(store, {"a": B(0.0, 1e-6, 3), "b": B(0.003, 1e-6, 3)}, 3, 5.0)
    acc = PS.accuracy(store.get_state(PS.K_PS))
    assert acc["acc"] == 0.75 and acc["base_acc"] == 0.75 and acc["skill"] == 0.0
    c = _icfg(ctx, pairwise_min_scored=2, pairwise_min_acc=0.55, pairwise_weight=0.1)
    assert PS.weight(c, store) == 0.0                                  # accurate but no skill: off


def test_a_failed_prescreen_is_retried_while_the_node_is_unmeasured(env):
    ctx, store, led, ide = env
    ide.ctx = _icfg(ctx, brief_workers=0, pairwise_weight=0.2)
    v = ide.view(led)
    ide._view = (ID._view_token(v), v)
    ide.llm = _PSLLM(fail=1)
    some = [n for n, b in v.beliefs.items() if b.n >= 1][:3]
    for n in some:
        store.set_state(f"ideation:{n}", {"sig": f"objective sig of {n} a mechanism text"})
    ide._prescreen("nNEW", "objective sig of a new mechanism text", "claim", (ROOT_NODE_ID,))
    rec = store.get_state(PS.K_PS)["nNEW"]
    assert "error" in rec and rec["tries"] == 1 and rec["sig"]
    led2 = dataclasses.replace(led, nodes={**led.nodes, "nNEW": led.nodes[some[0]]})
    v2 = dataclasses.replace(v, led=led2)
    ide.clock = lambda: rec["ts"] + PS.RETRY_S + 1
    ide._ps_retry(v2)
    assert "p_better" in store.get_state(PS.K_PS)["nNEW"]            # retried on a later tick, recorded


def test_the_prescreen_role_has_a_short_timeout_that_never_trips_the_qwen_cooldown(monkeypatch):
    from loop.agents import llm as L
    seen = {}

    def chat(system, prompt, temperature, timeout=600.0):
        seen["timeout"] = timeout
        raise L.BackendUnavailable("qwen: TimeoutError: timed out")
    monkeypatch.setattr(L, "qwen_chat", chat)
    c = L.SdkLLMClient(SimpleNamespace(llm_roles={}))
    with pytest.raises(L.BackendUnavailable, match="timed out after 45s"):
        c._ask("prescreen", "s", "p", 0.2)
    assert seen["timeout"] == 45.0 and c._qwen_down_until == 0.0


# -- H1 / H2 ---------------------------------------------------------------------------------------------------------------
def _summary(cls_vis, cls_ho):
    d = {}
    for t, x in (("a", cls_vis), ("h", cls_ho)):
        for h in ("linear", "knn", "fewshot"):
            d[f"final_probe_{t}_{h}_val_f1"] = x
    return d


def test_the_holdout_audit_reads_the_selection_seeds_and_flags_the_test_seeds(tmp_path):
    from loop.core import holdout as HO
    from loop.core.seeds import SeedBlock
    nodes = {"root": CD.Node("root", None, None, "baseline", "baseline", "baseline", "root", 0.0, "", "", "B"),
             "c": CD.Node("c", "root", None, "explore", "f", "explore", "root", 1.0, "", "", "B")}
    runs, obs = [], []
    # selection seeds 1-3: c won on the VISIBLE task only; TEST seeds 11-14: the full-panel verdict saw the holdout too
    for nid in ("root", "c"):
        for s, pool in [(1, CD.SELECT), (2, CD.SELECT), (3, CD.SELECT), (11, CD.TEST), (12, CD.TEST), (13, CD.TEST)]:
            gv, gh = (0.0, 0.0) if nid == "root" else ((0.01, 0.0) if pool == CD.SELECT else (0.006, 0.006))
            d = tmp_path / f"{nid}-{s}"
            d.mkdir()
            (d / "summary.json").write_text(json.dumps(_summary(0.6 + gv, 0.5 + gh)))
            rid = f"{nid}:{pool}:{s}"
            runs.append(CD.Run(rid, nid, s, "screen", "done", 1.0, 0.0, 0.0, str(d), 0, "B"))
            obs.append(CD.Obs(rid, nid, s, "screen", pool, {"cls": 0.55, "seg": .5, "slide": .7, "auc": .6, "surv": .5,
                                                           "rob": .6}, 0.6, 1.0, 0.0, 0, None, "B"))
    led = CD.Ledger(nodes, tuple(runs), tuple(obs), frozenset({11, 12, 13}), ((0.0, "root"),), bases=((0.0, "root", "B"),))
    blk = SeedBlock("t1", CD.TEST, (11, 12, 13), ("c", "root"), "test c", 0.0, 5.0,
                    ({"node": "c", "vs": "root", "kind": "pass", "mean": 0.003},))
    cache = HO.TaskCache(None, 1)
    cache.update(led)
    [row] = HO.audit(led, [blk], cache, frozenset({"cls:h"}))
    assert row["basis"] == "select" and row["n"] == 3
    assert row["visible"] == pytest.approx(0.01) and row["holdout"] == pytest.approx(0.0)    # the overfit signal
    assert row["test_biased"] and row["test_holdout"] == pytest.approx(0.006)
    s = HO.summarize([row] * 5, 2.0, 5)
    assert s["alert"]
    only_test = dict(row, basis="test")
    assert HO.summarize([only_test] * 6, 2.0, 5)["n"] == 0                                    # never alerted on


def test_llm_facing_numbers_read_the_visible_composite(tmp_path):
    from loop.core import holdout as HO
    o = CD.Obs("r", "n", 1, "screen", CD.SELECT, {"cls": 0.5, "seg": 0.5, "slide": 0.5, "auc": 0.5, "surv": 0.5,
                                                    "rob": 0.5}, 0.9, 1.0, 0.0, 0, None, "B")
    led = CD.Ledger({}, (), (o,), frozenset(), ((0.0, "root"),), bases=((0.0, "root", "B"),))
    vp = HO.visible_panel(led)
    assert vp.obs[0].panel == pytest.approx(0.5 * sum(CD.OFFICIAL_WEIGHTS.values()))
    assert HO.panel_offset(led) == pytest.approx(vp.obs[0].panel - 0.9)
    assert "OPERATOR" in HO.report_text.__doc__


def test_the_llm_reviewers_report_has_no_holdout_numbers():
    from loop.core import meta as MT
    rep = MT.MetaReport(ts=1e9, since=0.0, holdout={"tasks": ["cls:h"], "rows": [], "summary": {"n": 0}})
    assert "Holdout audit" in rep.markdown() and "Holdout audit" not in rep.markdown(llm=True)
    assert "cls:h" not in rep.markdown(llm=True)


def test_the_meta_cycle_reads_the_visible_ledger(ctx, tmp_path):
    from ..test_tick_core import _module_tick, _WithTests
    tick, store, _ = _module_tick(ctx, tmp_path, _WithTests())
    marker, seen = {}, {}

    def vis(led):
        marker["v"] = dataclasses.replace(led, obs=tuple(led.obs))
        return marker["v"]
    tick.holdout.visible = vis
    tick.meta = lambda t, data, statuses: seen.setdefault("meta", data) and []
    tick.step()
    assert seen["meta"] is not tick.data and len(seen["meta"].obs) == len(marker["v"].obs)
    tick.close()
    store.close()


# -- C1 ----------------------------------------------------------------------------------------------------------------------
def test_calibration_asks_and_scores_the_visible_composite_with_per_definition_base_rates():
    text = CAL.brief_text(4, 0.006, "the gap")
    assert "VISIBLE SCREENING COMPOSITE" in text and "not the official" in text
    from loop.spec import IdeationCfg
    assert IdeationCfg().calibration_min_seeds == 3
    recs = {"a": {"kind": "explore", "gap_kind": "target", "q": [0.0, 1.0, 0.0], "ls": -0.1},
            "b": {"kind": "explore", "gap_kind": "mde", "q": [0.0, 0.0, 1.0], "ls": -0.1}}
    assert CAL.base_rates(recs, "target")[1] > CAL.base_rates(recs)[1] > CAL.base_rates(recs, "mde")[1]
    assert ID.ref_kind(ID.ref_gap.__doc__ or "") in ("target", "mde")


# -- D1 ----------------------------------------------------------------------------------------------------------------------
def test_differs_from_must_explain_each_cited_node():
    assert not MS.explained(" n0001]", "n0001")
    assert not MS.explained(" n0001, n0002: both]", "n0001")
    assert MS.explained(" n0001: changes the teacher's EMA instead of the student's lr; n0002 x]", "n0001")
    assert not MS.explained(" n0001: changes the teacher's EMA instead of the student's lr; n0002 x]", "n0002")


def test_the_judge_catches_a_paraphrase_the_lexical_embedder_misses(ctx, store):
    from loop.core.mechanisms import dup_rule
    c = _icfg(ctx, mech_dup_cos=0.95, mech_judge_cos=0.2)
    _hnode(store, "n0101", "objective koleo spread term on the patch tokens", "koleo_patch")
    store.set_state("ideation:n0101", {"sig": "objective koleo patch a koleo spread term on the patch tokens"})
    sig = "objective koleo patch spread regulariser over patch embeddings"
    assert dup_rule(c, store, sig, ("root",))[0] == ""                                  # lexical only: passes
    calls = []
    judge = lambda s, near: calls.append(near) or [True] * len(near)                    # noqa: E731
    rule, fix = dup_rule(c, store, sig, ("root",), judge=judge)
    assert rule == "near_dup_signature" and "n0101" in fix and "DOSE" in fix and calls
    assert dup_rule(c, store, sig, ("root",), judge=lambda s, near: None)[0] == ""      # no verdict: no hit
    claim = "x [differs from: n0101: a uniformity loss on CLS tokens, not the patch KoLeo]"
    assert dup_rule(c, store, sig, ("root",), claim, judge=judge)[0] == ""


# -- O1 ----------------------------------------------------------------------------------------------------------------------
PF = '''import torch


class ViT:
    def forward(self, x):
        return x

    def probe_features(self, x):
        cls = x
        if self.pf_fitted:
            typicality = x
            gate = torch.sigmoid(((typicality - self.ct_ms[0]) / self.ct_ms[1]).clamp(-30, 30))
            cls = self.rn_mu[0] + gate[:, None] * (cls - self.rn_mu[0])
        return cls
'''


def _program():
    from pathlib import Path

    import yaml
    from tests.loop.conftest import example_programs
    return example_programs()[0]


def test_the_operator_programs_gate_arms_are_literal_probe_features_edits_and_the_norm_arm_is_full_retrain():
    from loop.core import program as PG
    prog = _program()
    arms = {a["id"]: a for a in prog["arms"]}
    assert set(arms) == {"gate_off", "gate_floor", "cls_norm_matched"}
    for aid in ("gate_off", "gate_floor"):
        assert PG.arm_problem(arms[aid]) == "" and arms[aid]["span"] == "model.py::ViT.probe_features"
        new, why = PG.apply_edits(PF, arms[aid]["span"], arms[aid]["edits"])
        assert new is not None and new != PF, why
        assert new.split("def probe_features")[0] == PF.split("def probe_features")[0]   # nothing outside the span
    assert arms["cls_norm_matched"]["lane"] == "full" and "train.py" in arms["cls_norm_matched"]["note"]
    assert "lane 'full'" in PG.arm_problem(arms["cls_norm_matched"])
    text = ID.program_text(prog)
    assert "gate_off, gate_floor) are built by the loop" in text and "FULL-RETRAIN" in text


def test_eval_only_arms_materialise_as_their_own_eligible_nodes(tmp_path, monkeypatch):
    from loop.contracts import Node, NodeStatus, Operator, Strategy
    from loop.core import program as PG
    from loop.store import SqliteStore
    from tests.loop.test_adapters import git
    from tests.loop.test_c5_evalonly_lane import commit_on
    from tests.loop.test_c5b_evalonly_train_path import VIT, adapter
    from tests.loop.test_evalonly import TRAIN, repo as _repo  # noqa: F401

    r = tmp_path / "base"
    (r / "configs").mkdir(parents=True)
    (r / "configs" / "main.yaml").write_text("train: {seed: 1}\nmodel: {type: vit}\n")
    (r / "model.py").write_text(VIT)
    (r / "train.py").write_text(TRAIN)
    git(r, "init", "-q", "-b", "main")
    git(r, "add", "-A")
    git(r, "commit", "-qm", "base")
    base = git(r, "rev-parse", "HEAD")
    ad = adapter(tmp_path, r)
    arms = [{"id": "half", "lane": "evalonly", "file": "model.py", "span": "model.py::ViT.probe_features",
             "edits": [{"old": "* 1.0", "new": "* 0.5"}], "note": "readout scale"},
            {"id": "trainpath", "lane": "evalonly", "file": "model.py", "span": "model.py::ViT.forward",
             "edits": [{"old": "x = self._suppress(x, self.rn_mu)", "new": "x = self._suppress(x, self.rn_mu) * 0.9"}]},
            {"id": "cfg", "lane": "evalonly", "file": "model.py", "span": "model.py::ViT.probe_features",
             "config": {"model.x": 1}, "edits": [{"old": "* 1.0", "new": "* 0.7"}]},
            {"id": "retrain", "lane": "full", "note": "needs a train.py buffer"}]
    ctx = dataclasses.replace(ad.ctx, cfg=dataclasses.replace(ad.ctx.cfg, ideation=dataclasses.replace(
        ad.ctx.cfg.ideation, operator_programs=({"id": "op-t", "arms": arms},))))
    ad.ctx = ctx
    store = SqliteStore(tmp_path / "s.db", ctx.task_id)
    store.ensure_root(base)
    store.put_node(Node(id="inc", task_id=ctx.task_id, commit=base, parent_id="root", strategy=Strategy.EXPLORE,
                        operator=Operator.DRAFT, family="f", mechanism_tag="inc", diff_hash="inc",
                        status=NodeStatus.CONFIRMED))
    acts = PG.arms(ctx, store, ad, store.get_node("inc"), 10.0)
    book = store.get_state(PG.K_ARMS)["op-t"]
    # the full arm is the session's; v3-c8a B1: each mirrorable eval-only arm's opposite direction is built beside it
    assert set(book) == {"half", "half_opp", "trainpath", "cfg", "cfg_opp"} and acts
    opp = store.get_node(book["half_opp"])
    assert "* 2.0" in ad.file_at(opp.commit, "model.py") and "opposite_of=half" in opp.notes
    nid = book["half"]
    n = store.get_node(nid)
    assert n.status is NodeStatus.QUEUED and n.parent_id == "inc" and "arm=half" in n.notes
    assert ad.changed_files(base, n.commit) == ["model.py"] and ad.eval_only(base, n.commit)[0]
    assert store.get_state(f"config_diff:{nid}") == {}
    assert book["trainpath"].startswith("-: not eval-only eligible")
    assert "changes no config" in book["cfg"]
    assert PG.arms(ctx, store, ad, store.get_node("inc"), 11.0) == []                    # once
    store.close()


# -- M1 ----------------------------------------------------------------------------------------------------------------------
def test_the_test_reserve_is_released_when_owed_runs_cannot_launch(ctx, tmp_path):
    from loop import tick as TK

    from ..test_port import _two_epochs
    sim, store = _two_epochs(ctx, tmp_path)
    t = [1000.0]
    tick = TK.Tick(ctx, store, sim, TK.ModuleCore(), db_path=tmp_path / "loop.db", clock=lambda: t[0])
    acts: list[str] = []
    assert tick._reserve_timeout(4, acts) == 4
    t[0] += 3 * 3600 - 10
    assert tick._reserve_timeout(4, acts) == 4
    t[0] += 20
    tick._tick_ts = t[0]
    assert tick._reserve_timeout(4, acts) == 0 and store.events("test_reserve_released")
    assert any("released" in a for a in acts) and tick._reserve_timeout(4, []) == 0
    assert tick._reserve_timeout(3, []) == 3                                             # the owed count moved: re-armed
    tick.close()
    store.close()
