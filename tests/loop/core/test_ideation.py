"""loop.core.ideation: briefs per lane, the headroom table, engagement, the plateau trigger and review, anti-repeat
with one differs_from retry, test-block-only credit, research results -> nodes; the researcher's v3 finish."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from loop.agents import notes as NT
from loop.agents import researcher as RS
from loop.contracts import (AxisPrediction, Candidate, HandoffNote, HypothesisKind, HypothesisRecord, Node, NodeStatus,
                            Operator, ResearchResult, Strategy, Worktree)
from loop.core import data as D
from loop.core import ideation as ID
from loop.core import status as ST
from loop.core import test as T
from loop.core.model import Noise
from loop.select.embed import HashingEmbedder
from loop.store import SqliteStore

from ..conftest import InlineExecutor
from ..test_tick import _ledger
from .conftest import TASK

README = """# x
### Baselines

| # | Name | Description | final score | classification | segmentation | progression | mutation | survival | robustness |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | FM-A | a | **0.69** | 0.82 | 0.61 | 0.66 | 0.62 | 0.63 | 0.65 |
| 2 | FM-B | b | 0.68 | 0.80 | 0.64 | 0.68 | 0.64 | 0.61 | 0.58 |
| 3 | DINOv2-G/14 | c | 0.60 | 0.90 | 0.90 | 0.90 | 0.90 | 0.90 | 0.90 |

### How to submit
"""


@pytest.fixture
def env(ctx, tmp_path):
    db = _ledger(tmp_path / "ledger.db")
    ctx = dataclasses.replace(ctx, task_id=TASK)
    store = SqliteStore(db, TASK)
    readme = tmp_path / "README.md"
    readme.write_text(README)
    yield ctx, store, D.load(db, TASK), ID.Ideation(ctx, store, readme=readme)
    store.close()


def test_fm_reference_takes_the_best_pathology_baseline_per_axis(tmp_path):
    (p := tmp_path / "README.md").write_text(README)
    fm = ID.fm_reference(p)
    assert fm["cls"] == (0.82, "FM-A") and fm["seg"] == (0.64, "FM-B") and fm["slide"] == (0.68, "FM-B")
    assert fm["rob"] == (0.65, "FM-A")                                      # DINOv2 is not a reference
    assert ID.fm_reference(tmp_path / "missing.md") == {}


def _row(axis, inc, best=None, root=0.0, sigma=0.004):
    return ID.Headroom(axis, D.OFFICIAL_WEIGHTS[axis], sigma, inc, 5, inc - root, best, "nX" if best else "", None, "")


def test_repair_axis_is_the_attainable_headroom_worth_most_in_the_composite():
    w = {"cls": 0.36, "seg": 0.09, "slide": 0.23, "auc": 0.12, "surv": 0.0, "rob": 0.2}
    rows = [_row("cls", 0.73, best=0.735), _row("auc", 0.62, root=0.66), _row("surv", 0.6, best=0.7)]
    assert ID.repair_axis(rows, w) == "auc"                  # .12 x .04 regression beats .36 x .005; surv weighs 0
    assert ID.repair_axis([_row("seg", 0.6)], w) == "cls"   # no headroom anywhere: classification
    text = ID.headroom_text(rows, "n1")
    assert "classification (0.300) TARGET" in text and "mutation" in text and "(-10.0σ)" in text


def test_engagement_z_uses_the_pairs_own_noise_and_is_signed_along_the_prediction():
    """The 10-08 shadow logged ``n1708 gn_late_over_clip did not engage (z +96.3)``: the sd was pooled over every
    node, where a few exploded nodes (178, 369 vs ~2.3) inflate it ~25x past the parent's own run-to-run sd, and the
    z was printed unsigned. A predicted fall that happens (n1732 / n1842) is engaged; an explosion is reversed."""
    parent = [{"node_id": "root", "features": {"gn_late_over_clip": x}} for x in (2.25, 2.39, 2.23, 2.28, 2.33, 2.43)]
    exploded = [{"node_id": f"x{i}", "features": {"gn_late_over_clip": x}} for i, xs in enumerate(((365, 372), (170, 186)))
                for x in xs]
    fell = parent + exploded + [{"node_id": "n1842", "features": {"gn_late_over_clip": x}} for x in (1.40, 1.43)]
    e = ID.engagement(fell, "n1842", "root", "gn_late_over_clip", -1)
    assert e.status == "engaged" and e.z > 5                                    # global pooled sd: z -0.5
    rose = parent + exploded + [{"node_id": "n1708", "features": {"gn_late_over_clip": x}} for x in (178.1, 178.8)]
    e = ID.engagement(rose, "n1708", "root", "gn_late_over_clip", -1)
    assert e.status == "reversed" and e.z < -100


def test_engagement_is_a_z_above_one_in_the_declared_direction():
    rows = [{"node_id": "p", "features": {"lr_end_frac": x}} for x in (0.50, 0.52, 0.48)]
    up = rows + [{"node_id": "c", "features": {"lr_end_frac": 0.60}}]
    e = ID.engagement(up, "c", "p", "lr_end_frac", +1)
    assert e.status == "engaged" and e.z > 1 and e.n == 1                  # decided from the first done seed
    assert ID.engagement(up, "c", "p", "lr_end_frac", -1).status == "reversed"           # it moved the other way
    flat = rows + [{"node_id": "c", "features": {"lr_end_frac": 0.505}}]
    assert ID.engagement(flat, "c", "p", "lr_end_frac", -1).status == "did not engage"
    assert ID.engagement(rows, "c", "p", "lr_end_frac", 1).status == "pending"
    assert ID.engagement(up, "c", "p", "no_such_quantity", 1).status == "pending"
    assert ID.engagement(up, "c", "nobody", "lr_end_frac", 1).status == "unmeasurable"


def _judged(node, ts, switched=True, mean=0.003):
    return ST.Judged(node, "root", "b", ts, T.Verdict(T.Kind.PASS if switched else T.Kind.FAIL, 20.0, 5, mean, ""),
                     switched)


def _runs(spec):
    return tuple(D.Run(f"r{i}", n, i, "screen", "done", h, t - 1, t, "", 0) for i, (n, t, h) in enumerate(spec))


def test_plateau_when_the_gpu_hours_since_the_last_pass_exceed_twice_the_median_between_passes():
    runs = _runs([("a", 10, 5.0), ("a", 20, 5.0), ("b", 30, 10.0), ("c", 40, 21.0)])
    led = D.Ledger({}, runs, (), frozenset(), ())
    p = ID.plateau(led, [_judged("a", 20), _judged("b", 30), _judged("x", 35, switched=False)])
    assert (p.passes, p.median, p.gpu_h, p.active) == (2, 10.0, 21.0, True)        # spans 10 (start..a), 10 (a..b)
    assert not ID.plateau(dataclasses.replace(led, runs=runs[:3]), [_judged("a", 20), _judged("b", 30)]).active
    assert not ID.plateau(led, []).active                                          # no pass yet: no median


def _cand(claim, diff="+loss.weight: 2.0\n", commit="c-new"):
    h = HypothesisRecord(id="", task_id=TASK, kind=HypothesisKind.TRIAL, claim=claim, mechanism_tag="m",
                         family="objective", predictions={"panel": AxisPrediction(1, 0.003)}, falsification="f")
    return Candidate("req-1", TASK, commit, "c0", diff, f"h-{commit}", ("config.yaml",), {"loss.weight": 2.0}, h,
                     HandoffNote("", TASK, "req-1", "a", "l", "c"))


def test_anti_repeat_rejects_a_near_copy_of_a_negative_node_once_then_takes_differs_from(store):
    emb, old = HashingEmbedder(), "raise the jepa loss weight to sharpen patch targets on the tiles"
    store.put_node(Node(id="n0001", task_id=store.task_id, commit="c1", parent_id="root", strategy=Strategy.EXPLORE,
                        operator=Operator.DRAFT, family="objective", mechanism_tag="jepa_weight", diff_hash="d1",
                        status=NodeStatus.QUEUED, embedding=emb.embed(old + "\nloss.weight: 2.0")))
    cand = _cand(old + " a bit more")
    vec = emb.embed(f"{cand.hypothesis.claim}\nloss.weight: 2.0")
    assert ID.repeat_rule(store, cand, vec) == ("", "")                       # inert before the tick writes the set
    store.set_state(ID.K_NEGATIVE, {"n0001": -0.004})
    rule, fix = ID.repeat_rule(store, cand, vec)
    assert rule == "repeat_of_negative" and "n0001" in fix and "differs_from" in fix
    again = dataclasses.replace(cand, hypothesis=dataclasses.replace(
        cand.hypothesis, claim=RS.claim_text({"claim": cand.hypothesis.claim, "differs_from": "n0001 used 2x the dose on the stem"})))
    assert ID.repeat_rule(store, again, vec) == ("", "")                     # the one retry: names the neighbour
    other = emb.embed("swap the optimiser for a per-layer trust ratio\nopt.kind: lamb")
    assert ID.repeat_rule(store, cand, other) == ("", "")


def test_credit_counts_only_test_block_gains_per_gpu_hour():
    nodes = {k: D.Node(k, None if k == "root" else "root", None, s, "f", lane, "root", 0.0, "", f"brief={b}")
             for k, s, lane, b in (("root", "baseline", "baseline", "-"), ("a", "explore", "explore", "explore"),
                                   ("b", "explore", "explore", "explore"), ("c", "crossover", "crossover", "crossover"))}
    runs = _runs([("a", 1, 2.0), ("b", 2, 2.0), ("c", 3, 4.0)])
    led = D.Ledger(nodes, runs, (), frozenset(), ())
    v = ID.View(led, Noise({}, {}), {}, {}, {}, (_judged("a", 5, mean=0.004), _judged("c", 6, False, -0.002)), "a")
    cr = ID.credit(v)
    assert cr["explore/explore/-/-"]["gain"] == pytest.approx(0.004) and cr["explore/explore/-/-"]["gpu_h"] == 4.0
    assert cr["explore/explore/-/-"]["per_gpu_h"] == pytest.approx(0.001)
    assert cr["crossover/crossover/-/-"] == {"nodes": 1, "tested": 1, "gain": 0.0, "gpu_h": 4.0, "per_gpu_h": 0.0}


def test_briefs_per_lane_on_the_ledger_fixture(env):
    ctx, store, led, ide = env
    object.__setattr__(ctx.cfg.ideation, "kind_weight", False)        # the fixed rotation (A3 kind weights off)
    v = ide.view(led)
    assert v.incumbent in led.nodes and v.beliefs and v.mde() > 0
    kinds = {lane: ide.brief_for(lane, led, v) for lane in ("explore", "crossover", "exploit", "factorial")}
    assert kinds["exploit"].kind == "repair" and kinds["exploit"].axis in D.AXES
    assert kinds["crossover"].kind in ("crossover", "explore")                 # explore when no partner is left
    if kinds["crossover"].kind == "crossover":
        assert len(kinds["crossover"].parents) == 2         # the incumbent + a stack partner, or a code-picked pair of legs
    for b in kinds.values():
        assert (b.parents[0] == v.incumbent or b.kind == "crossover") and "## Per-axis headroom" in b.text and "TARGET" in b.text
        assert f"about {v.mde():.4f}" in b.text and f"at least {v.mde():.4f}" not in b.text and "## Literature notes" in b.text
        assert "FM-A" in b.text                                               # the README reference
    lit = ide.brief_for("explore", led, v, kind="literature")
    assert lit.text.startswith("EXPLORE brief (literature)") and "READ at least two papers" in lit.text
    store.set_state(ID.K_PLATEAU, {"active": True, "since": 1.0, "ts": 2.0, "served": 0, "diagnosis": "d",
                                   "hypotheses": [{"claim": "the teacher never freezes", "evidence": "e",
                                                   "decisive_experiment": "reclock the EMA", "prediction": {}}]})
    pl = ide.brief_for("explore", led, v)
    assert pl.kind == "kuhn" and "the teacher never freezes" in pl.text and pl.plateau_index == 0   # v3-c6i
    assert "PARADIGM" in pl.text and "hypothesis.analogy" in pl.text
    review = ide.review_prompt(v, ID.plateau(led, v.judged), {})
    assert review.startswith("PLATEAU REVIEW") and "{" + "headroom}" not in review and "lr_end_frac" in review


def test_literature_every_third_explore_session(env):
    ctx, store, led, ide = env
    got = []
    for i in range(6):
        got.append(ide.kind_for("explore"))
        store.log("research_started", request_id=f"r{i}", brief_kind=got[-1])
    assert got == ["explore", "explore", "literature"] * 2
    assert ide.kind_for("exploit") == "repair" and ide.kind_for("crossover") == "crossover"


class _Research:
    def __init__(self, free=2):
        self.n, self.started, self.researcher = free, [], SimpleNamespace(llm=None)

    def free(self):
        return self.n - len(self.started)

    def start(self, req, *args, **log):
        self.started.append(req)


def test_ideate_writes_the_anti_repeat_set_and_starts_one_session_per_idea(env):
    ctx, store, led, ide = env
    tick = SimpleNamespace(research=_Research(2), pool=InlineExecutor(), dry=False)
    ideas = [SimpleNamespace(kind="idea", lane=l) for l in ("explore", "exploit", "crossover")]
    acts = ide.ideate(tick, ideas, led, {})
    reqs = tick.research.started
    assert len(reqs) == 2 and all(r.id.startswith(ID.REQUEST_PREFIX) for r in reqs)
    assert [r.brief_kind for r in reqs] == ["explore", "repair"] and reqs[1].strategy is Strategy.EXPLOIT
    assert RS.is_v3(reqs[0]) and sum("researcher started" in a for a in acts) == 2
    assert store.get_state(ID.K_NEGATIVE) == ID.negatives(ide.view(led), {})
    dry = SimpleNamespace(research=_Research(0), pool=InlineExecutor(), dry=True)
    assert [a.split()[:2] for a in ide.ideate(dry, ideas[:1], led, {}) if a.startswith("would")] == \
        [["would", "brief"]]


def test_plateau_review_runs_once_per_episode_and_its_hypotheses_become_briefs(env, monkeypatch):
    ctx, store, led, ide = env
    object.__setattr__(ctx.cfg.ideation, "kind_weight", False)        # the fixed rotation (A3 kind weights off)
    v = ide.view(led)
    monkeypatch.setattr(ID, "plateau", lambda led, judged: ID.Plateau(True, 123.0, 50.0, 10.0, 2))
    calls = []

    class LLM:
        def complete_json(self, role, *, system, prompt, schema):
            calls.append(prompt)
            return {"diagnosis": "d", "hypotheses": [{"claim": "c1", "evidence": "e", "decisive_experiment": "x1",
                                                      "prediction": {"diagnostic": "lr_end_frac"}}]}
    ide.llm = LLM()
    tick = SimpleNamespace(research=_Research(0), pool=InlineExecutor(), dry=False)
    assert "review started" in " ".join(ide._plateau(tick, v, {}))
    assert "hypotheses" in " ".join(ide._plateau(tick, v, {}))                 # harvested next tick
    assert ide._plateau(tick, v, {}) == [] and len(calls) == 1                # once per episode
    assert "PLATEAU REVIEW" in calls[0] and "Per-axis headroom" in calls[0]
    assert ide.kind_for("explore") == "kuhn"
    tick = SimpleNamespace(research=_Research(1), pool=InlineExecutor(), dry=False)
    ide.ideate(tick, [SimpleNamespace(kind="idea", lane="explore")], led, {})
    assert tick.research.started[0].brief_kind == "kuhn" and "c1" in tick.research.started[0].brief
    assert tick.research.started[0].strategy is Strategy.MOONSHOT
    assert store.get_state(ID.K_PLATEAU)["served"] == 1 and ide.kind_for("explore") != "kuhn"


def test_intake_makes_a_node_with_its_prediction_and_source_tags(env):
    ctx, store, led, ide = env
    req = ide.request(ID.Brief("explore", "explore", ("root",), "b"))
    plan = {"mechanism_prediction": {"diagnostic": "lr_end_frac", "direction": -1, "axes": {"cls": 1}, "size": 0.004},
            "knobs": [{"key": "train.lr", "lo": 1e-4, "hi": 4e-4, "scale": "log"}]}
    ok = ResearchResult("init_incompatible", _cand("a brand new idea about something", commit="cX"), "x")
    out = ide.intake(req, dataclasses.replace(ok, plan=plan))
    nid = out.split()[1]
    node = store.get_node(nid)
    assert node.status is NodeStatus.REJECTED and node.reject_reason == "init_incompatible"
    assert f"request={req.id} brief=explore lane=explore" in node.notes and " prompt=" in node.notes
    assert store.events("intake")[-1]["claimed_size"] == 0.004 and {k: x for k, x in store.get_state(f"ideation:{nid}").items() if k != "sig"} == plan
    assert ID.prediction(store, nid) == plan["mechanism_prediction"]
    assert ide.intake(req, ResearchResult("backend_unavailable", None, "down")).startswith("research")
    good = ide.intake(req, ResearchResult("ok", _cand("zero-initialised side adapter on the last block", commit="cY")))
    node = store.get_node(good.split()[1])
    assert good.startswith("queued") and node.status is NodeStatus.QUEUED and node.embedding is not None
    assert store.get_state(f"config_diff:{node.id}") == {"loss.weight": 2.0} and node.parent_id == "root"
    assert len([n for n in store.nodes() if n.id == nid]) == 1


def test_researcher_v3_finish_requires_a_measurable_mechanism_prediction(ctx):
    sch = RS.finish_schema(ctx, v3=True)
    hy = sch["properties"]["hypothesis"]
    assert "mechanism_prediction" in hy["required"] and "differs_from" in hy["properties"]
    from loop.core import factorial as F             # one knob schema (diff knobs come from stacking, not a finish)
    assert sch["properties"]["knobs"]["items"] is F.KNOB_SCHEMA
    assert "mechanism_prediction" not in RS.finish_schema(ctx)["properties"]["hypothesis"]["required"]
    t = RS.WorktreeTools(None, Worktree("/tmp/x", "c0", "b"), writable=("*",), locked=())
    t.v3 = t.need_fp = True
    fin = {"hypothesis": {"claim": "c", "mechanistic_model": "m", "bottleneck": "b", "prior_attempts": "p",
                          "mechanism_prediction": {"diagnostic": "the teacher gets better", "direction": 1,
                                                   "axes": {"cls": 1}, "size": 0.003}}, "handoff": {}}
    assert "is not a measured quantity" in t.check_finish(fin) and "lr_end_frac" in t.check_finish(fin)
    fin["hypothesis"]["mechanism_prediction"]["diagnostic"] = "lr_end_frac"
    t.bounces = 0
    assert t.check_finish(fin) == ""
    fin["hypothesis"]["mechanism_prediction"]["diagnostic"] = "b3_upd"                # a raw feature key
    assert t.check_finish(fin) == ""
    assert RS.claim_text({"claim": "c", "prior_work": "p", "differs_from": "n1"}) == \
        "c [prior work: p] [differs from: n1]"


def test_notes_are_filed_by_topic_and_retrieved_by_similarity(ctx, tmp_path):
    r = dataclasses.replace(ctx.cfg.researcher, notes_path=str(tmp_path / "rn" / "notes.jsonl"))
    c = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, researcher=r))
    st = NT.ResearchStage(c, "req-1")
    for url, title, topic in (("https://arxiv.org/abs/2101.00001", "teacher EMA schedules", "Teacher EMA"),
                              ("https://arxiv.org/abs/2101.00002", "stain augmentation", "stain")):
        st.on_web("WebFetch", {"url": url})
        assert st.t_note(url=url, title=title, topic=topic,
                         **{f: f"{title}: a long enough {f} with numbers +1%" for f in NT.FIELDS}).startswith("noted")
    md = (tmp_path / "rn" / "teacher_ema.md").read_text()
    assert "teacher EMA schedules" in md and "- transfer:" in md and (tmp_path / "rn" / "stain.md").exists()
    got = NT.retrieve(NT.notes_path(c), "the EMA teacher momentum schedule", 1)
    assert [n["title"] for n in got] == ["teacher EMA schedules"]


def test_headroom_is_measured_against_the_bar_node_when_configured():
    """Operator 2026-10-12: nothing against the old version -- the 'vs' column is the bar node (n0655), not root."""
    v = ID.View(None, type("N", (), {"sd": staticmethod(lambda a: 0.004)})(), {}, {}, {"n1": {"cls": (0.70, 5)}, "root": {"cls": (0.66, 9)}, "bar": {"cls": (0.69, 40)}},
                (), "n1")
    row = ID.headroom(v, None, "bar")[0]
    assert row.ref == "bar" and abs(row.vs_root - 0.01) < 1e-9
    assert "| vs bar |" in ID.headroom_text([row], "n1")
    # operator 11-10: the root (and any ancestor of the incumbent) is no reference: the column is vs the incumbent
    nr = ID.headroom(v, None)[0]
    assert nr.ref == "n1" and nr.vs_root == 0.0 and nr.gap == 0.0


def test_headroom_bar_node_the_incumbent_descends_from_is_no_reference():
    """A bar node the incumbent has replaced (its ancestor, e.g. after a re-root onto our Labless leader) is no rival:
    the gap carries no 'loss' against it and the column reads vs the incumbent."""
    from types import SimpleNamespace as NS
    led = NS(nodes={"n1": NS(parent="bar", second_parent=None, port_of=""), "bar": NS(parent="root", second_parent=None,
                                                                                   port_of="")})
    v = ID.View(led, type("N", (), {"sd": staticmethod(lambda a: 0.004)})(), {}, {},
                {"n1": {"cls": (0.68, 5)}, "bar": {"cls": (0.69, 40)}}, (), "n1")
    row = ID.headroom(v, None, "bar")[0]
    assert row.ref == "n1" and row.vs_root == 0.0
    assert abs(row.gap - 0.01) < 1e-9 and row.best_node == "bar"     # the gap is only a lead OVER the incumbent
