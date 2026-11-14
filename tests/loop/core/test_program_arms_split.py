"""v3-c10 (operator 10-21): a program with several FULL-RETRAIN arms gets ONE research session per arm; each arm's node
is tagged ``op=<program> arm=<arm>`` like the eval-only arms'; a node touching another arm's keys is rejected at intake
(``cross_arm_bundling``); the stack arm waits for its arms' verdicts; a superseded program is not briefed."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import yaml

from loop.agents.researcher import ResearchResult
from loop.core import ideation as ID
from loop.core import program as PG

from .test_fnf import env  # noqa: F401  (the fixture)

ROOT = Path(__file__).resolve().parents[3]
V2 = "op-dino-forcing-v2"


def _programs():
    from tests.loop.conftest import example_programs
    return example_programs()


def _icfg(ctx, **kw):
    return dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, ideation=dataclasses.replace(ctx.cfg.ideation, **kw)))


def _ide(env):
    ctx, store, led, ide = env
    object.__setattr__(ctx.cfg.ideation, "kind_weight", False)
    ide.ctx = _icfg(ctx, operator_programs=tuple(_programs()), program_share=0.0, program_share_base=0.0, brief_workers=0)
    store.set_state(ID.K_OPPROG, {p["id"]: {"ts": 1.0} for p in _programs() if p["id"] != V2})   # briefed long ago
    return store, led, ide


def _brief(ide, led):
    v = ide.view(led)
    plan = ide.brief_plan("explore", v, "program")
    return v, plan


def test_the_v2_program_is_split_into_one_session_per_arm_and_the_old_one_is_superseded(env):
    store, led, ide = _ide(env)
    old = next(p for p in _programs() if p["id"] == "op-dino-forcing")
    assert old["superseded_by"] == V2
    seen = []
    for want in ("A1", "A2", "B1", "B2"):
        _, plan = _brief(ide, led)
        assert plan["op"] == V2 and plan["op_arm"] == want          # the superseded program is never briefed
        head = plan["head"]
        assert f"YOUR ARM is {want}" in head and "ONE-ARM RULE" in head
        others = {"A1", "A2", "B1", "B2"} - {want}
        assert not any(f"YOUR ARM {o}:" in head for o in others)
        assert ("deepcopy" in head.split(f"YOUR ARM {want}:")[1].split("ONE-ARM")[0]) == want.startswith("A")      # only its own arm's spec
        ide._reserve(plan)
        seen.append(plan["op_arm"])
        req = ide.request(ID.Brief(plan["kind"], "explore", ("root",), plan["head"], arm=ID._arm(plan)))
        assert req.arm == f"program_arm:{V2}:{want}"
    assert seen == ["A1", "A2", "B1", "B2"]
    assert {k for k in store.get_state(ID.K_OPPROG) if k.startswith(V2)} == {f"{V2}/{a}" for a in seen}
    SAMP = "op-sampler-flatten"       # briefed whole on 10-19 (legacy key in K_OPPROG; its session built nothing)
    for want in ("site_cap", "project_cap"):     # live 10-21: the legacy key hid arms B / D for good; one session each
        _, plan = _brief(ide, led)
        assert plan["op"] == SAMP and plan["op_arm"] == want
        ide._reserve(plan)
    _, plan = _brief(ide, led)                                       # the stack arm waits: no arm has a node yet
    assert not plan["op"]                                            # nothing else is due (the gate program: no re-run)


def test_a_dropped_arm_brief_is_given_back_and_the_stack_waits_for_settled_arms(env):
    store, led, ide = _ide(env)
    _, plan = _brief(ide, led)
    ide._reserve(plan)
    ide._unreserve(plan)
    assert _brief(ide, led)[1]["op_arm"] == "A1"
    for a in ("A1", "A2", "B1", "B2"):
        ide._op_mark(f"{V2}/{a}")
    for a in ("site_cap", "project_cap"):
        ide._op_mark(f"op-sampler-flatten/{a}")
    assert ide._op_next() is None or ide._op_next().get("_arm", {}).get("id") != "stack"
    store.set_state(PG.K_ARMS, {V2: {"A1": "n1", "A2": "n2", "B1": "-: x", "B2": "-: y"}})
    ide._statuses = {"n1": "screening", "n2": "dropped"}
    assert ide._op_next() is None                                    # A1 is still screening
    ide._statuses = {"n1": "passed", "n2": "dropped"}
    nxt = ide._op_next()
    assert nxt["_arm"]["id"] == "stack" and "STACK arm" in ID.program_text(nxt)


def test_a_legacy_single_full_arm_program_is_still_one_session(env):
    store, led, ide = _ide(env)
    gate = next(p for p in _programs() if p["id"] == "op-gate-x-globals")
    assert PG.units(gate) == [(gate["id"], None)]


def _cand(commit, **kw):
    from .test_ideation import _cand as c
    return dataclasses.replace(c("a mechanism", commit=commit), **kw)


def _req(ide, arm):
    return ide.request(ID.Brief("program", "explore", ("root",), "b", arm=f"program_arm:{V2}:{arm}"))


def test_intake_tags_the_arm_node_and_books_it(env):
    store, led, ide = _ide(env)
    out = ide.intake(_req(ide, "A1"), ResearchResult("ok", _cand("cA1", config_diff={"dino.init_align_weight": 0.1})))
    assert out.startswith("queued")
    node = store.get_node(out.split()[1])
    assert f"op={V2} arm=A1 stage=A program_arm=1" in node.notes and "brief=program" in node.notes
    assert store.get_state(PG.K_ARMS)[V2]["A1"] == node.id


def test_intake_rejects_a_node_bundling_another_arm_and_rebriefs_it(env):
    store, led, ide = _ide(env)
    bundled = _cand("cX", config_diff={"dino.init_align_weight": 0.1, "dino.selfcond_drop": 0.1})
    out = ide.intake(_req(ide, "A1"), ResearchResult("ok", bundled))
    node = store.get_node(out.split()[1])
    assert out.startswith("rejected") and node.reject_reason.startswith("cross_arm_bundling")
    assert store.get_state(PG.K_ARMS)[V2]["A1"] == {"fails": 1}
    # code-level bundling (no config key): an added line carrying the other arm's identifier
    diff = "+++ b/model.py\n+    cond = self.selfcond_proj(x)\n"
    out = ide.intake(_req(ide, "A2"), ResearchResult("ok", _cand("cY", diff_text=diff, diff_hash="hY")))
    assert out.startswith("rejected") and "selfcond" in store.get_node(out.split()[1]).reject_reason
    # B2 after B1: an operator arm re-runs a measured mechanism, so the dedup does not apply to it
    ok = ide.intake(_req(ide, "B2"), ResearchResult("ok", _cand("cB2", config_diff={"dino.selfcond_drop": 0.1})))
    assert ok.startswith("queued")


def test_cross_arm_lets_the_stack_arm_and_shared_keys_through():
    p = next(p for p in _programs() if p["id"] == V2)
    arms = {a["id"]: a for a in p["arms"]}
    assert PG.cross_arm(p, arms["A1"], ["dino.init_align_layer", "dino.init_align_weight"], "") == ""
    assert PG.cross_arm(p, arms["stack"], ["dino.init_align_weight", "dino.selfcond_drop"], "x selfcond") == ""
    assert PG.cross_arm(p, arms["B1"], ["dino.init_align_weight"], "")


def test_a_per_arm_program_ignores_its_legacy_whole_program_key_but_a_one_session_program_honours_it(env):
    store, led, ide = _ide(env)
    done = store.get_state(ID.K_OPPROG)
    assert "op-sampler-flatten" in done and "op-gate-x-globals" in done
    got = [ide._op_next()]
    assert got[0]["id"] == V2                                        # the gate program (one unit, legacy key) is not re-run
    store.set_state(ID.K_OPPROG, {**done, **{f"{V2}/{a}": {"ts": 1.0} for a in ("A1", "A2", "B1", "B2")}})
    ide._statuses = {}
    assert ide._op_next()["_arm"]["id"] == "site_cap"


def test_an_operator_arm_is_not_rejected_by_the_similarity_dedups(env):
    store, led, ide = _ide(env)
    c1 = _cand("cA1", config_diff={"dino.init_align_weight": 0.1})
    assert ide.intake(_req(ide, "A1"), ResearchResult("ok", c1)).startswith("queued")
    twin = dataclasses.replace(c1, commit="cA2", diff_hash="h-other", config_diff={"dino.init_align_weight": 0.1,
                                                                                    "dino.init_align_layer": 6})
    out = ide.intake(_req(ide, "A2"), ResearchResult("ok", twin))     # a near-copy of A1 (embedding cosine ~1)
    assert out.startswith("queued"), out
    neg = {store.get_state(PG.K_ARMS)[V2]["A1"]: -0.0156}             # a negative closed node: n2923's place
    store.set_state(ID.K_NEGATIVE, neg)
    out = ide.intake(_req(ide, "B1"), ResearchResult("ok", _cand("cB1", config_diff={"dino.selfcond_drop": 0.1})))
    assert out.startswith("queued"), out


def test_the_session_side_checks_skip_the_dedups_for_an_operator_arm():
    from loop.agents.researcher import is_program_arm
    assert is_program_arm(ID_req := type("R", (), {"arm": f"program_arm:{V2}:A1"})())
    assert not is_program_arm(type("R", (), {"arm": "alt_base:n1"})()) and not is_program_arm(type("R", (), {})())
