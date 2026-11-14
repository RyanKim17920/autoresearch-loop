"""v3-c6i (10-13 literature scan + ideation analysis): the verbalized-sampling finish and its calibration scoring,
mechanism-signature dedup (the crossover near-dup fix) + terse gate reasons, the mechanism-taxonomy coverage,
persona lenses + the hidden idea pool, the kuhn kind, clade credit, the triggered diagnostic summary + the readout
kind, the pairwise pre-screen prior, and operator-seeded programs."""

from __future__ import annotations

import dataclasses
import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from loop.agents import researcher as RS
from loop.contracts import (Candidate, HandoffNote, HypothesisKind, HypothesisRecord, Node, NodeStatus, Operator,
                            ResearchResult, ROOT_NODE_ID, Strategy, Worktree)
from loop.core import calibration as CAL
from loop.core import ideation as ID
from loop.core import mechanisms as MS
from loop.core import prescreen as PS
from loop.select import intake as IN
from loop.select.embed import HashingEmbedder, changed_lines, cosine

from ..conftest import InlineExecutor
from .test_fnf import _Research, env  # noqa: F401  (the fixture)


def _icfg(ctx, **kw):
    return dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, ideation=dataclasses.replace(
        ctx.cfg.ideation, **kw)))


def _cands(k=4, gap=0.006):
    mechs = ["zero-initialised register tokens read by the probe head", "stain-aware colour augmentation per slide",
             "a KoLeo spread term on the patch tokens", "a cosine schedule clock keyed to samples seen",
             "multi-resolution global crops at 252 px"]
    return [{"mechanism": mechs[i], "family": "objective", "mechanism_tag": f"m{i}", "p_neg": 0.3, "median": 0.002,
             "p_gap": 0.1, "why": "w"} for i in range(k)]


def _fin(cands, chosen=0, choice="it targets the measured deficit", **hy):
    return {"hypothesis": {"claim": "c", "mechanism_tag": "m0", "family": "objective", "falsification": "f",
                           "mechanism_prediction": {"diagnostic": "lr_end_frac", "direction": 1, "axes": {"cls": 1},
                                                    "budget": [{"axis": "cls", "row": "D-cls", "gain": 0.01}]}, **hy},
            "handoff": {"actions": "a", "learnings": "l", "concerns": "c"}, "candidates": cands, "chosen": chosen,
            "choice": choice}


def _tools(tmp_path, **kw):
    t = RS.WorktreeTools(None, Worktree(str(tmp_path), "b", "c0"), writable=("*",), locked=(),
                         checker=kw.pop("checker", lambda **k: "OK"))
    t.v3, t.rows = True, ("D-cls",)
    for k, x in kw.items():
        setattr(t, k, x)
    return t


# ---------------------------------------------------------------------------------------------- 1. VS + calibration
def test_the_slate_needs_k_distinct_coherent_candidates_and_a_justified_choice():
    assert CAL.problem(_fin(_cands()), 4, 0.006) == ""
    assert "list 4 candidate" in CAL.problem(_fin(_cands(2)), 4, 0.006)
    bad = _cands()
    bad[1] = {**bad[1], "p_neg": 0.7, "p_gap": 0.5}
    assert "p_neg + p_gap must be <= 1" in CAL.problem(_fin(bad), 4, 0.006)
    bad = _cands()
    bad[2] = {**bad[2], "median": -0.004, "p_neg": 0.1}
    assert "contradicts P(Δ<0)" in CAL.problem(_fin(bad), 4, 0.006)
    same = _cands()
    same[3] = dict(same[0])
    assert "are one mechanism" in CAL.problem(_fin(same), 4, 0.006)
    assert "`chosen`" in CAL.problem(_fin(_cands(), chosen=7), 4, 0.006)
    assert "`choice`" in CAL.problem(_fin(_cands(), choice=""), 4, 0.006)


def test_a_malformed_slate_bounces_once_then_is_accepted_and_flagged(tmp_path):
    t = _tools(tmp_path, vs_k=4, ref_gap=0.006, vs_cap=1)
    msg = t.check_finish(_fin(_cands(2)))
    assert msg.startswith("candidate slate:") and "list 4 candidate" in msg
    assert t.check_finish(_fin(_cands(2))) == "" and "list 4" in t.vs_problem
    ok = _tools(tmp_path, vs_k=4, ref_gap=0.006)
    assert ok.check_finish(_fin(_cands())) == "" and ok.vs_problem == ""
    # no size floor: a tiny honest budget is never bounced (the c4i floor is gone from v3 requests: floor 0)
    tiny = _fin(_cands())
    tiny["hypothesis"]["mechanism_prediction"]["budget"] = [{"axis": "cls", "row": "D-cls", "gain": 0.0005}]
    assert _tools(tmp_path, vs_k=4, ref_gap=0.006).check_finish(tiny) == ""


def test_the_checker_compares_the_chosen_candidates_mechanism(tmp_path):
    seen = []
    t = _tools(tmp_path, vs_k=4, ref_gap=0.006, checker=lambda **k: seen.append(k) or "OK")
    t.check_finish(_fin(_cands(), chosen=2))
    assert seen[-1]["mechanism"] == _cands()[2]["mechanism"]


def test_every_candidate_is_kept_and_the_chosen_prediction_rides_on_the_plan(tmp_path, ctx):
    r = RS.LLMResearcher(ctx, None, SimpleNamespace())
    t = _tools(tmp_path, vs_k=4, ref_gap=0.006)
    plan = r._plan(SimpleNamespace(), _fin(_cands(), chosen=1), t, SimpleNamespace(path=str(tmp_path)))
    assert len(plan["candidates"]) == 4 and plan["chosen"] == 1 and plan["mechanism"] == _cands()[1]["mechanism"]
    assert plan["prediction"] == {"p_neg": 0.3, "median": 0.002, "p_gap": 0.1, "gap": 0.006, "gap_kind": "",
                                  "unit": "visible_composite"}
    assert "candidates" in RS.finish_schema(ctx, True)["properties"]
    assert "candidates" not in RS.finish_schema(ctx, False)["properties"]


def test_the_log_score_is_proper_an_honest_forecast_beats_an_overstated_one():
    gap, m, v = 0.006, 0.001, 0.0005 ** 2                       # measured: small positive, well below the gap
    honest = {"p_neg": 0.15, "median": 0.001, "p_gap": 0.02}
    hype = {"p_neg": 0.05, "median": 0.008, "p_gap": 0.7}
    s_h, s_x = CAL.score(honest, m, v, gap), CAL.score(hype, m, v, gap)
    assert s_h["ls"] > s_x["ls"] and s_h["rps"] < s_x["rps"] and s_h["err"] < s_x["err"]
    assert sum(s_h["q"]) == pytest.approx(1.0, abs=1e-3) and sum(s_h["p"]) == pytest.approx(1.0, abs=1e-3)
    recs = {f"a{i}": {"kind": "explore", **CAL.score(honest, m, v, gap)} for i in range(6)}
    recs |= {f"b{i}": {"kind": "program", **CAL.score(hype, m, v, gap)} for i in range(6)}
    sk = CAL.kind_skill(recs, 5.0)
    assert sk["explore"]["skill"] > sk["program"]["skill"] and sk["explore"]["n"] == 6
    w = CAL.reweight({"explore": 0.5, "program": 0.5, "repair": 0.0}, sk, 1.0)
    assert w["explore"] > 0.5 > w["program"] and sum(w.values()) == pytest.approx(1.0)


def test_ideate_scores_due_predictions_and_the_kind_table_follows_calibration(env):
    ctx, store, led, ide = env
    ide.ctx = _icfg(ctx, calibration_min_seeds=2)
    v = ide.view(led)
    nid = next(n for n, b in v.beliefs.items() if b.n >= 2 and n != ROOT_NODE_ID)
    store.set_state(f"ideation:{nid}", {"prediction": {"p_neg": 0.2, "median": 0.001, "p_gap": 0.1, "gap": 0.006}})
    acts = ide._calibrate(v)
    rec = store.get_state(CAL.K_CAL)[nid]
    assert acts and rec["n"] >= 2 and set(rec) >= {"ls", "rps", "p", "q", "kind"}
    assert store.events("calibration_scored")[-1]["node_id"] == nid
    assert ide._calibrate(v) == []                                    # scored once
    # the brief carries the slate and NO size floor
    b = ide.brief_for("explore", led, v, "explore")
    assert "## Candidates, then one" in b.text and "There is no minimum size" in b.text
    assert "bounced once" not in b.text and "at least" not in ID.decisive_text(v)
    req = ide.request(b, v)
    assert req.k_candidates == ctx.cfg.ideation.k_candidates and req.claim_floor == 0.0


# ---------------------------------------------------------------------------------------------- 2. signature dedup
INC = ("The incumbent stacks a zero-initialised readout gate on the probe features with a CLS norm match and four "
       "global crops; it lifted classification on fine-grained sets.")


def _hnode(store, nid, claim, tag, family="objective", parent=ROOT_NODE_ID, second=None, ts=1.0, diff=""):
    store.put_hypothesis(HypothesisRecord(id=f"h-{nid}", task_id=store.task_id, kind=HypothesisKind.TRIAL,
                                          claim=claim, mechanism_tag=tag, family=family, predictions={},
                                          falsification="f"))
    store.put_node(Node(id=nid, task_id=store.task_id, commit=f"c-{nid}", parent_id=parent, strategy=Strategy.EXPLORE,
                        operator=Operator.DRAFT, family=family, mechanism_tag=tag, diff_hash=f"d-{nid}",
                        status=NodeStatus.SCREENING, second_parent_id=second, hypothesis_id=f"h-{nid}",
                        embedding=HashingEmbedder().embed(f"{claim}\n{changed_lines(diff)}"), created_ts=ts))


def test_the_signature_drops_the_ancestors_text_and_node_ids():
    sig = MS.signature("objective", "koleo_patch_spread", f"Stack n2520 onto n1248@065b87f. {INC} Adds a KoLeo "
                       "spread term on patch tokens.", [INC])
    assert "n2520" not in sig and "incumbent stacks" not in sig and sig.startswith("objective koleo patch spread")
    assert MS.first_line("a. b") == "a." and MS.strip_ancestors("x " + INC, [INC]) == "x"


def test_crossovers_onto_one_incumbent_are_not_near_dups_of_each_other_but_a_real_repeat_is(ctx, store):
    """10-13: 43-69% of crossover sessions were rejected near_dup -- the claim + diff embedding of a crossover carries
    the incumbent's change and description (its diff is against the common ancestor)."""
    inc_diff = "".join(f"+gate_line_{i} = probe_features * gate_{i}\n" for i in range(40))
    _hnode(store, "n1000", INC, "readout_gate_globals", diff=inc_diff, ts=1.0)
    _hnode(store, "n1001", "an orthogonality penalty between register tokens", "register_orthogonality", ts=2.0)
    _hnode(store, "n1002", "a cosine schedule clock keyed on samples seen", "schedule_clock_samples", ts=3.0)
    _hnode(store, "n1100", f"Stack n1001 onto n1000. {INC} Adds an orthogonality penalty between register tokens.",
           "stack_register_orthogonality", parent="n1000", second="n1001", ts=4.0,
           diff=inc_diff + "+ortho = reg @ reg.T\n")
    claim = f"Stack n1002 onto n1000. {INC} Adds a cosine schedule clock keyed on samples seen."
    cand_text = f"{claim}\n{changed_lines(inc_diff + '+clock = samples / total\n')}"
    old = cosine(HashingEmbedder().embed(cand_text), store.get_node("n1100").embedding)
    assert old > ctx.cfg.intake.near_dup_cos                       # the old rule: rejected as a dup of n1100
    anc = MS.ancestors(store, ("n1000", "n1002"))
    assert {"n1000", "n1002"} <= set(anc)
    sig = MS.signature("training", "stack_schedule_clock", claim, MS.claims_of(store, anc))
    assert MS.dup_rule(ctx, store, sig, ("n1000", "n1002"), claim) == ("", "")   # its own mechanism: new
    again = MS.signature("objective", "stack_register_orthogonality",
                         f"Stack n1001 onto n1000. {INC} Adds an orthogonality penalty between register tokens.",
                         MS.claims_of(store, MS.ancestors(store, ("n1000",))))
    rule, fix = MS.dup_rule(ctx, store, again, ("n1000",), "")
    assert rule == "near_dup_signature" and "n1100" in fix          # a real repeat (outside the ancestry) is caught
    named = MS.dup_rule(ctx, store, again, ("n1000",), "x [differs from: n1100 uses a learned target]")
    assert named == ("", "")
    _hnode(store, "n1200", "a learned stain lookup table", "stain_lut", ts=5.0)
    small = _icfg(ctx, mech_dup_window=1)                            # only the newest mechanism is compared
    assert MS.dup_rule(small, store, again, ("n1000",), "")[0] == ""


def test_intake_dedups_v3_candidates_on_the_signature_not_the_diff(ctx, store):
    inc_diff = "".join(f"+gate_line_{i} = probe_features * gate_{i}\n" for i in range(40))
    _hnode(store, "n1000", INC, "readout_gate_globals", diff=inc_diff, ts=1.0)
    _hnode(store, "n1100", f"Stack n1001 onto n1000. {INC} Adds an orthogonality penalty.", "stack_ortho",
           parent="n1000", ts=4.0, diff=inc_diff + "+ortho = reg @ reg.T\n")
    h = HypothesisRecord(id="", task_id=store.task_id, kind=HypothesisKind.TRIAL,
                         claim=f"Stack n1002 onto n1000. {INC} Adds a schedule clock.", mechanism_tag="stack_clock",
                         family="objective", predictions={}, falsification="f")
    cand = Candidate("req-v3-1", store.task_id, "cN", "c0", inc_diff + "+clock = samples\n", "dN", ("train.py",), {},
                     h, HandoffNote("", store.task_id, "r", "a", "l", "c"))
    emb = HashingEmbedder()
    assert IN.diagnose(ctx, store, cand, operator=Operator.CROSSOVER, embedder=emb)[0] == "near_dup"
    sig = MS.signature(h.family, h.mechanism_tag, h.claim, MS.claims_of(store, ["n1000"]))
    got = IN.diagnose(ctx, store, cand, operator=Operator.CROSSOVER, embedder=emb, signature=(sig, ("n1000",)))
    assert got[0] not in ("near_dup", "near_dup_signature")


def test_gate_rejections_reach_the_session_terse():
    for rule in ("forbidden_reference", "init_illegal", "disallowed_data", "protected_key"):
        t = IN.session_text(rule, "`open('/data/probe/labels.csv')` matches '(?<![\\w.])open\\s*\\(' ...")
        assert "open(" not in t and "matches" not in t and t
    assert IN.session_text("near_dup", "too close to n1 (a/b)") == "too close to n1 (a/b)"   # dedup reasons stay


# ---------------------------------------------------------------------------------------------- 3. taxonomy
def test_the_taxonomy_maps_family_and_tag_and_briefs_point_at_the_least_used_category(env):
    ctx, store, led, ide = env
    assert MS.category(ctx, "eval_readout", "gate_floor") == "readout_eval"
    assert MS.category(ctx, "objective", "koleo_spread") == "objective"
    assert MS.category(ctx, "optimization", "warmup_shorter") == "optimization"
    assert MS.category(ctx, "x", "zzz") == MS.OTHER
    assert MS.suggest({"a": 3, "b": 1, "c": 1, MS.OTHER: 0}, ["a", "b", "c", MS.OTHER]) == "b"
    v = ide.view(led)
    b = ide.brief_for("explore", led, v, "explore")
    assert "## Mechanism coverage this epoch" in b.text and b.category and f"**{b.category}**" in b.text
    assert ide.request(b, v).category == b.category
    counts = MS.coverage(ctx, store, led)
    assert set(counts) == set(MS.categories(ctx))


# ---------------------------------------------------------------------------------------------- 4. diversity
def test_personas_rotate_and_hidden_pool_kinds_see_only_a_terse_taken_list(env):
    ctx, store, led, ide = env
    object.__setattr__(ctx.cfg.ideation, "kind_weight", False)
    tick = SimpleNamespace(research=_Research(3), pool=InlineExecutor(), dry=False)
    ide.ideate(tick, [SimpleNamespace(kind="idea", lane=l) for l in ("explore", "exploit", "explore")], led, {})
    reqs = tick.research.started
    personas = [r.persona for r in reqs]
    assert len(set(personas)) == 3 and all(p in ctx.cfg.ideation.personas for p in personas)
    assert all(f"## Your lens this session: {r.persona}" in r.brief for r in reqs)
    store.put_node(Node(id="nq01", task_id=store.task_id, commit="cq", parent_id=ROOT_NODE_ID,
                        strategy=Strategy.EXPLORE, operator=Operator.DRAFT, family="objective",
                        mechanism_tag="queued_idea_tag", diff_hash="dq", status=NodeStatus.QUEUED, created_ts=9e9))
    v = ide.view(led)
    ex = ide.brief_for("explore", led, v, "explore")
    assert "## Taken (in flight / queued: do not build these)" in ex.text and "objective/queued_idea_tag" in ex.text
    assert "## Reserved:" not in ex.text and "## Engagement of recent ideas" not in ex.text
    rp = ide.brief_for("exploit", led, v)                            # a repair keeps the full context
    assert rp.kind == "repair" and "## Reserved:" in rp.text and "## Engagement of recent ideas" in rp.text


# ---------------------------------------------------------------------------------------------- 5. kuhn
def test_a_kuhn_finish_needs_a_distant_analogy_once(tmp_path):
    t = _tools(tmp_path, need_analogy=True)
    fin = _fin(_cands())
    assert t.check_finish(fin).startswith("analogy:")
    assert t.check_finish(fin) == ""                                  # bounced once, then accepted
    ok = _tools(tmp_path, need_analogy=True)
    good = _fin(_cands(), analogy={"field": "radio astronomy", "source": "arXiv:1234.5678",
                                   "mapping": "matched filtering of faint sources -> per-dataset gate calibration"})
    assert ok.check_finish(good) == "" and RS.analogy_problem(good["hypothesis"]) == ""
    assert ID.KIND_REQUEST["kuhn"] == (Strategy.MOONSHOT, Operator.DRAFT) and "kuhn" in ID.KINDS


# ---------------------------------------------------------------------------------------------- 6. clade credit
def _cnode(nid, parent, notes="", base="e1"):
    from loop.core import data as D
    return D.Node(nid, parent, None, "explore", "f", "explore", ROOT_NODE_ID, 1.0, "screening", notes, base)


def test_clade_credit_takes_the_best_tuned_descendant_not_the_untuned_screen():
    from loop.core import data as D
    nodes = [_cnode("p1", ROOT_NODE_ID, "brief=program"), _cnode("p1.d1", "p1", "ladder_of=p1"),
             _cnode("p1.d2", "p1.d1", ""), _cnode("x1", ROOT_NODE_ID, "brief=explore"), _cnode("x1.k", "x1", "brief=crossover")]
    led = SimpleNamespace(nodes={n.id: n for n in nodes}, epoch="e1")
    B = ID.Belief
    v = SimpleNamespace(led=led, beliefs={"p1": B(-0.002, 1e-6, 3), "p1.d1": B(0.004, 1e-6, 2),
                                          "p1.d2": B(0.001, 1e-6, 1), "x1": B(0.001, 1e-6, 3),
                                          "x1.k": B(0.01, 1e-6, 3)})
    vals = ID.clade_values(v, min_seeds=2)
    assert vals["p1"] == pytest.approx(0.002)        # -0.002 + 0.004 (d1, 2 seeds); d2 has 1 seed: not counted
    assert vals["x1"] == pytest.approx(0.001)        # x1.k is another session's clade
    ys = ID.kind_yields(v, clade=2)
    assert ys["program"] == [pytest.approx(0.002)] and ys["explore"] == [pytest.approx(0.001)]
    assert ID.kind_yields(v)["program"] == [-0.002, 0.004, 0.001]     # the old per-node attribution
    judged = [SimpleNamespace(node="p1.d1", verdict=SimpleNamespace(mean=0.003)),
              SimpleNamespace(node="p1", verdict=SimpleNamespace(mean=-0.001))]
    v2 = SimpleNamespace(led=SimpleNamespace(nodes=led.nodes, runs=()), judged=judged, beliefs={})
    cr = ID.credit(v2, key=lambda n: ID._brief_tag(n), clade=True)
    assert cr["program"]["gain"] == pytest.approx(0.003) and cr["program"]["tested"] == 1


# ---------------------------------------------------------------------------------------------- 7. diag summary
FEATS = {ds: {"tiles": 32, "cos_head_vs_cls": {"mean": c}, "gates": [{"mean": g, "p10": g / 2, "p50": g, "p90": g * 2}]}
         for ds, g, c in (("bach", 0.35, 0.74), ("bracs", 0.45, 0.78), ("crc", 0.45, 0.81), ("mhist", 0.46, 0.77),
                          ("pcam", 0.10, 0.50), ("wilds", 0.11, 0.49))}


def _features(ctx, inc):
    from loop.agents import diagnose as DX
    p = Path(ctx.state_dir) / DX.SUBDIR / DX._safe(inc)
    p.mkdir(parents=True, exist_ok=True)
    (p / "features.json").write_text(json.dumps({"n_tiles": 32, "datasets": FEATS}))


def test_distortions_flag_shrunk_gates_against_the_median_or_a_reference():
    d = ID.distortions(FEATS, 0.5)
    assert [x["dataset"] for x in d if x["metric"] == "gate"] == ["pcam", "wilds"]
    assert all(x["ratio"] < 0.5 for x in d) and not [x for x in d if x["metric"] == "cos"]
    assert ID.distortions(FEATS, 0.5, reference="crc")[0]["ref_name"] == "crc"
    assert ID.distortions({}, 0.5) == [] and ID.distortions(FEATS, 0.0) == []


def test_the_diagnostic_summary_tops_every_brief_and_triggers_the_readout_kind(env):
    ctx, store, led, ide = env
    object.__setattr__(ctx.cfg.ideation, "kind_weight", False)
    v = ide.view(led)
    assert ide.diag(v)[0] == "" or "largest per-dataset" in ide.diag(v)[0]   # no features yet: no gate lines
    assert not ide._readout_ok(v)
    _features(ctx, v.incumbent)
    ide._diag = None
    for kind in ("explore", "repair", "crossover"):
        b = ide.brief_for({"repair": "exploit"}.get(kind, kind), led, v, None if kind != "explore" else "explore")
        assert b.text.startswith(f"## Diagnostic summary of the incumbent {v.incumbent}")
        assert "pcam 0.10" in b.text.split("\n\n")[0]
    assert ide._readout_ok(v) and ide.kind_for("explore", v) == "readout"
    rb = ide.brief_for("explore", led, v, "readout")
    assert "READOUT (eval-only)" in rb.text and "change no config value" in rb.text
    tick = SimpleNamespace(research=_Research(3), pool=InlineExecutor(), dry=False)
    ide.ideate(tick, [SimpleNamespace(kind="idea", lane="explore")] * 3, led, {})
    kinds = [r.brief_kind for r in tick.research.started]
    assert kinds.count("readout") == ctx.cfg.ideation.readout_max_sessions        # capped per incumbent
    assert store.get_state(ID.K_READOUT)[v.incumbent] == ctx.cfg.ideation.readout_max_sessions


# ---------------------------------------------------------------------------------------------- 8. pre-screen
class _LLM:
    def __init__(self, p=None, fail=False):
        self.calls, self.p, self.fail = [], p, fail

    def complete_json(self, role, *, system, prompt, schema):
        self.calls.append((role, prompt))
        if self.fail:
            raise RuntimeError("backend down")
        import re
        n = len(re.findall(r"(?m)^\d+\. ", prompt.split("Measured neighbours", 1)[-1]))
        return {"p_better": self.p or [0.8] * n}


def test_the_pairwise_prior_only_nudges_screen_options_and_is_one_call_per_node(env):
    ctx, store, led, ide = env
    ide.ctx = _icfg(ctx, brief_workers=0, pairwise_weight=0.2)
    v = ide.view(led)
    ide.llm = _LLM()
    some = [n for n, b in v.beliefs.items() if b.n >= 1][:3]
    for n in some:
        store.set_state(f"ideation:{n}", {"sig": f"objective sig of {n} a mechanism text"})
    ide._prescreen("nNEW", "objective sig of a new mechanism text", "claim", (ROOT_NODE_ID,))
    rec = store.get_state(PS.K_PS)["nNEW"]
    assert len(ide.llm.calls) == 1 and ide.llm.calls[0][0] == PS.ROLE and rec["s"] == pytest.approx(0.6)
    assert len(rec["neighbors"]) == min(ctx.cfg.ideation.pairwise_k, len([b for b in v.beliefs.values() if b.n >= 1]))
    from loop.core.acquire import Option
    opts = [Option("screen", "nNEW", "explore", 0.001, 4e-6, 1e-5, 1.0, 1), Option("seed", "nNEW", "explore", 0.001,
                                                                                  4e-6, 1e-5, 1.0, 1),
            Option("screen", "other", "explore", 0.001, 4e-6, 1e-5, 1.0, 1)]
    got = PS.nudge(opts, store.get_state(PS.K_PS), 0.2)
    assert got[0].m == pytest.approx(0.001 + 0.2 * 0.6 * 0.002) and got[1].m == 0.001 and got[2].m == 0.001
    ide.llm = _LLM(fail=True)
    ide._prescreen("nBAD", "objective sig x", "claim", (ROOT_NODE_ID,))
    assert "error" in store.get_state(PS.K_PS)["nBAD"]


def test_prescreen_accuracy_is_logged_and_a_bad_prior_switches_itself_off(env):
    ctx, store, led, ide = env
    c = _icfg(ctx, pairwise_min_scored=2, pairwise_min_acc=0.6, pairwise_weight=0.1)
    B = ID.Belief
    store.set_state(PS.K_PS, {"a": {"s": 0.6, "p_better": [0.9, 0.9], "neighbors": [["x", 0.001], ["y", 0.002]]},
                              "b": {"s": -0.6, "p_better": [0.1, 0.9], "neighbors": [["x", 0.001], ["y", -0.004]]}})
    acts = PS.score(store, {"a": B(-0.003, 1e-6, 2), "b": B(0.0, 1e-6, 2)}, 2, 5.0)
    assert len(acts) == 2 and store.events("prescreen_scored")
    acc = PS.accuracy(store.get_state(PS.K_PS))
    assert acc == {"n": 2, "pairs": 4, "correct": 2, "acc": 0.5, "base_acc": 1.0, "skill": -0.5}   # a: 0/2, b: 2/2
    assert PS.weight(c, store) == 0.0 and store.get_state(PS.K_ACC)["weight"] == 0.0
    assert PS.weight(_icfg(ctx, pairwise_min_scored=9), store) == 0.1   # too few scored: the configured weight


def test_the_core_has_a_prior_seam_and_the_proxy_is_gone():
    from loop.tick import ModuleCore
    assert ModuleCore.prior is None and not hasattr(PS, "install") and not hasattr(PS, "_Nudged")
    assert PS.nudge([], {"nA": {"s": 1.0}}, 0.1) == []


# ---------------------------------------------------------------------------------------------- 9. operator programs
OP = {"id": "op-test-1", "brief": "Operator text: try the gate arms first.", "stages": ["(A) gate arms", "(B) crops"],
      "components": [{"family": "eval_readout", "mechanism": "gate off"}], "budget_gpu_h": 30,
      "harm_guards": ["rob", "slide"]}


def test_an_operator_program_is_briefed_once_and_given_back_when_dropped(env):
    ctx, store, led, ide = env
    object.__setattr__(ctx.cfg.ideation, "kind_weight", False)
    ide.ctx = _icfg(ctx, operator_programs=(OP,), program_share=0.0, program_share_base=0.0, brief_workers=0)
    v = ide.view(led)
    assert ide.kind_for("explore", v) == "program"
    plan = ide.brief_plan("explore", v, "program")
    assert plan["op"] == "op-test-1" and "Operator text: try the gate arms first." in plan["head"]
    ide._reserve(plan)
    assert "op-test-1" in store.get_state(ID.K_OPPROG) and ide.kind_for("explore", v) != "program"
    ide._unreserve(plan)                                               # a dropped brief gives it back
    assert ide.kind_for("explore", v) == "program"
    tick = SimpleNamespace(research=_Research(2), pool=InlineExecutor(), dry=False)
    ide.ideate(tick, [SimpleNamespace(kind="idea", lane="explore")] * 2, led, {})
    reqs = tick.research.started
    assert [r.brief_kind for r in reqs].count("program") == 1 and "## Operator-seeded program op-test-1" in reqs[0].brief
    assert reqs[0].strategy is Strategy.MOONSHOT and "Harm guards: rob, slide" in reqs[0].brief


def test_the_task_spec_seeds_the_operator_program_and_loads():
    from loop.spec import load_context
    root = Path(__file__).resolve().parents[3]
    import yaml
    raw = yaml.safe_load((root / "tasks" / "nanopath_v2.yaml").read_text())
    from tests.loop.conftest import example_programs
    assert raw["loop"]["ideation"]["operator_programs"] == []          # the shipped spec seeds none
    progs = example_programs()
    assert progs[0]["id"] == "op-gate-x-globals" and progs[0]["stages"] and progs[0]["harm_guards"]
    from loop import spec as SP
    cfg = SP.build(SP.IdeationCfg, {**raw["loop"]["ideation"], "operator_programs": progs}, "loop.ideation")
    assert cfg.operator_programs[0]["budget_gpu_h"] == 30


def test_a_crossover_takes_the_code_picked_pair_when_loop_core_pairs_exists(env, monkeypatch):
    import sys
    import types
    ctx, store, led, ide = env
    v = ide.view(led)
    import loop.core as LC
    from loop.core import pairs as REAL
    real_next = REAL.next_pair
    monkeypatch.setattr(REAL, "next_pair", lambda view, **kw: None)
    fallback = ide.brief_plan("crossover", v, "crossover")            # nothing paired: the stack partner, as before
    monkeypatch.setattr(REAL, "next_pair", real_next)
    REAL.bind(ctx, store)
    real = REAL.next_pair(v, mark=False)                             # the real c6a implementation on this view: callable
    assert real is None or (len(real) == 3 and real[0] != real[1])
    other = next(n for n in v.beliefs if n not in (v.incumbent, ROOT_NODE_ID))
    mod = types.ModuleType("loop.core.pairs")
    mod.next_pair = lambda view, **kw: (view.incumbent, other, "code-picked")
    mod.pair_key = REAL.pair_key
    monkeypatch.setitem(sys.modules, "loop.core.pairs", mod)
    monkeypatch.setattr(LC, "pairs", mod)
    from loop.core import alloc as AL                                  # v3-c7i I5: the LLM brief takes a pair whose
    store.set_state(AL.K_FLAGGED, {REAL.pair_key(v.incumbent, other): {"why": "x: conflict"}})   # code graft conflicted
    plan = ide.brief_plan("crossover", v, "crossover")
    assert plan["kind"] == "crossover" and plan["parents"] == (v.incumbent, other) and other in plan["head"]
    assert plan["pair"] == [v.incumbent, other] and not store.events("pair_picked")   # v3-c6r: logged at the start
    mod.next_pair = lambda view, **kw: None                             # nothing to pair: the fallback
    assert ide.brief_plan("crossover", v, "crossover")["parents"] == fallback["parents"]


def test_held_out_datasets_never_reach_the_diagnose_tables():
    from loop.agents import diagnose as DX
    store = SimpleNamespace(get_state=lambda k: ["cls:spider_colorectal", "seg:pannuke"] if k == "holdout.tasks" else None)
    assert DX.held_names(store) == {"spider_colorectal", "pannuke"}
    rows = [{"dataset": "spider_colorectal", "head": "knn"}, {"dataset": "bach", "head": "knn"}]
    assert DX.unheld(rows, store) == [rows[1]]
    assert "spider" not in DX.features_text({"incumbent": "n1", "datasets": {"spider_colorectal": {"tiles": 3}, "bach": {"tiles": 3}}},
                                            DX.held_names(store))


def test_the_pairwise_prior_is_centred_once_enough_nodes_are_scored():
    """11-09 nanobrain: 32 of 33 queued nodes had s < 0, so every unscored node outranked every scored one; with >=
    CENTER_MIN predictions s is centred on their median (it orders scored nodes AROUND the unscored ones)."""
    from loop.core.acquire import Option
    book = {f"n{i}": {"s": s} for i, s in enumerate([-0.3, -0.2, -0.2, -0.1, 0.3])}
    assert PS.center(book) == pytest.approx(-0.2) and PS.center(dict(list(book.items())[:4])) == 0.0
    opts = [Option("screen", n, "explore", 0.0, 1e-4, 1e-5, 1.0, 1) for n in ("n0", "n4", "unscored")]
    got = {o.node: o.m for o in PS.nudge(opts, book, 0.1, centred=True)}
    assert got["n4"] > got["unscored"] == 0.0 > got["n0"]
    raw = {o.node: o.m for o in PS.nudge(opts, book, 0.1)}          # default off (nanopath): uncentred, unchanged
    assert raw["n0"] < raw["n4"] and raw["n4"] > 0.0 and raw["unscored"] == 0.0
    assert raw["n0"] == pytest.approx(0.1 * -0.3 * math.sqrt(1e-4))
