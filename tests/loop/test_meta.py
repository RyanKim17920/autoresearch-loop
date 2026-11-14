"""META-LEARNING (loop/meta.py): every component's track record computed from a fake store, the record in
that component's prompt, and the per-component bias knob clamped by selftune and passed as 'be more/less X'.
Run: uv run pytest tests/loop -q --confcutdir=tests/loop"""

from __future__ import annotations

import dataclasses
import json

import pytest

from loop import meta as ML
from loop import selftune as T
from loop.contracts import (PANEL, ROOT_NODE_ID, AxisPrediction, CandidateRequest, HypothesisEvidence,
                            HypothesisKind, HypothesisRecord, HypothesisStatus, LeverTags, Node, NodePosterior,
                            NodeStatus, Operator, Posterior, Postmortem, ReviewReport, Strategy, WorldSnapshot)


def _post(nid: str, mean: float, p: float) -> NodePosterior:
    return NodePosterior(node_id=nid, increment=Posterior(mean, 0.002, 2), utility=Posterior(mean, 0.002, 2),
                         axes={}, p_positive=p, p_beats_incumbent=p, vetoed=False)


def _snap(posts=(), hyps=None) -> WorldSnapshot:
    return WorldSnapshot(task_id="t", n_scored=len(posts), sigma={PANEL: 0.004}, variance={},
                         nodes={p.node_id: p for p in posts}, incumbent_id=ROOT_NODE_ID, family_means={},
                         levers=(), hypotheses=hyps or {})


def _node(store, nid: str, *, claim: str = "", pred: float = 0.005, status=NodeStatus.SCREENED_OUT, notes="",
          parent=ROOT_NODE_ID, operator=Operator.DRAFT, strategy=Strategy.EXPLORE, ts=10.0, **kw) -> Node:
    store.put_hypothesis(HypothesisRecord(
        id=f"h-{nid}", task_id=store.task_id, kind=HypothesisKind.TRIAL, claim=claim or f"idea {nid}",
        mechanism_tag=f"{nid}_tag", family="optimization",
        predictions={PANEL: AxisPrediction(1 if pred > 0 else -1, abs(pred))}, falsification="f", created_ts=ts))
    n = Node(id=nid, task_id=store.task_id, commit=f"c{nid}", parent_id=parent, strategy=strategy, operator=operator,
             family="optimization", mechanism_tag=f"{nid}_tag", diff_hash=nid, status=status, hypothesis_id=f"h-{nid}",
             notes=notes, created_ts=ts, updated_ts=ts, **kw)
    store.put_node(n)
    return n


@pytest.fixture
def root(store):
    return store.ensure_root("c0")


def test_researcher_record_calibration_and_examples(ctx, store, root):
    posts = []
    for i, m in enumerate([0.001, -0.002, 0.0005, -0.001, 0.0008]):
        _node(store, f"n{i}", pred=0.006, claim=f"lower lr variant {i}")
        posts.append(_post(f"n{i}", m, 0.8 if m > 0 else 0.2))
    r = ML.resolve(ctx, store, _snap(posts), "researcher")
    assert r["resolved"] == 5 and r["accuracy"] == pytest.approx(0.6) and r["successes"] == 3
    txt = ML.track_record(ctx, store, _snap(posts), "researcher")
    assert "sign right 60% of 5" in txt and "over-predict" in txt and "predicted +0.0060, measured -0.0020" in txt


def test_review_and_first_principles_records(ctx, store, root):
    rv = ReviewReport(id="rv-1", task_id=store.task_id, trigger="cadence", summary="s", findings=(),
                      meta_hypothesis_ids=("rv-1-m1", "rv-1-m2"), probe_briefs=((None, "try x"),), explore_bias={},
                      n_evals_at=3, created_ts=5.0)
    fp = dataclasses.replace(rv, id="rv-2", trigger="first_principles", meta_hypothesis_ids=(), created_ts=50.0)
    store.put_review(rv)
    store.put_review(fp)
    for mid in ("rv-1-m1", "rv-1-m2"):
        store.put_hypothesis(HypothesisRecord(id=mid, task_id=store.task_id, kind=HypothesisKind.META, claim=f"claim {mid}",
                                              mechanism_tag="m", family="cross", predictions={PANEL: AxisPrediction(1, 0.004)},
                                              falsification="f", review_id="rv-1", created_ts=5.0))
    hyps = {"rv-1-m1": HypothesisEvidence("rv-1-m1", 30.0, 0.1, 3, HypothesisStatus.SUPPORTED),
            "rv-1-m2": HypothesisEvidence("rv-1-m2", 0.1, 30.0, 3, HypothesisStatus.REFUTED)}
    _node(store, "p1", notes="request=req-7 brief=probe", ts=6.0)               # served by the log
    _node(store, "p2", notes="request=req-8 brief=moonshot", ts=60.0)           # after rv-2: the fp review's
    store.log("meta_decision", component="review_probe", decision_id="rv-1:0", node_id="", hypothesis_id="",
              review_id="rv-1", payload={"request": "req-7"})
    snap = _snap([_post("p1", 0.003, 0.9), _post("p2", -0.003, 0.1)], hyps)
    r = ML.resolve(ctx, store, snap, "review")
    assert "1 supported / 1 refuted of 2" in r["headline"] and "spawned 1 nodes, hit 1/1" in r["headline"]
    assert any("SUPPORTED" in x for x in r["right"]) and any("REFUTED" in x for x in r["wrong"])
    f = ML.resolve(ctx, store, snap, "first_principles")
    assert "1 reviews" in f["headline"] and "hit 0/1" in f["headline"]


def test_review_selftune_before_after(ctx, store, root):
    store.put_review(ReviewReport(id="rv-1", task_id=store.task_id, trigger="cadence", summary="s", findings=(),
                                  meta_hypothesis_ids=(), probe_briefs=(), explore_bias={}, n_evals_at=1, created_ts=1.0))
    t0 = 1000.0
    for i, ts in enumerate([t0 - 100, t0 + 100, t0 + 200]):
        _node(store, f"n{i}", ts=ts)
    store.log("selftune", knob="review.stall_evals", old=25, new=10, source="rv-1")
    ev = store.events("selftune")[0]
    store._db.execute("UPDATE log SET ts=? WHERE event='selftune'", (t0,))
    assert ev
    r = ML.resolve(ctx, store, _snap([_post("n1", 0.002, 0.9), _post("n2", 0.002, 0.9)]), "review")
    assert "closes/12h 1.0 -> 2.0" in r["headline"] and "hit rate 0% -> 100%" in r["headline"]


def test_design_knob_picks(ctx, store, root):
    from loop.design import K_DESIGN
    store.put_review(ReviewReport(id="rv-1", task_id=store.task_id, trigger="cadence", summary="s", findings=(),
                                  meta_hypothesis_ids=(), probe_briefs=(), explore_bias={}, n_evals_at=1,
                                  design=json.dumps([{"key": "train.lr", "high": 0.002}])))
    eff = lambda k, e, s: {"key": k, "effect": e, "se": 0.001, "significant": s, "estimable": True}  # noqa: E731
    store.set_state(K_DESIGN, {"batches": [{"batch": 1, "points": {}, "stacked": "", "n": 32, "df": 20, "interactions": [],
                                            "knobs": [{"key": k, "low": 1, "high": 2} for k in ("train.lr", "a", "b", "c")],
                                            "effects": [eff("train.lr", 0.004, True), eff("a", 0.0, False),
                                                        eff("b", -0.003, True), eff("c", 0.0005, False)]}]})
    r = ML.resolve(ctx, store, _snap(), "design")
    assert "significant 2/4" in r["headline"] and "review-picked 1/1" in r["headline"] and "auto 1/3" in r["headline"]


def test_crossover_vs_parents(ctx, store, root):
    _node(store, "a")
    _node(store, "b")
    _node(store, "x", notes="xover=a,b", operator=Operator.CROSSOVER, strategy=Strategy.CROSSOVER)
    _node(store, "k", notes="compound=optimization+augmentation stack_from=a", parent="b")
    snap = _snap([_post("a", 0.003, 0.9), _post("b", 0.002, 0.8), _post("x", 0.005, 0.95), _post("k", 0.001, 0.6)])
    r = ML.resolve(ctx, store, snap, "crossover")
    assert "beat best parent 1/2" in r["headline"] and r["right"][0].startswith("crossover x")
    assert r["wrong"][0].startswith("compound k")


def test_scorecard_covers_every_component_and_empty_records(ctx, store, root):
    rows = ML.scorecard(ctx, store, _snap())
    assert [c for c, *_ in rows] == list(ML.COMPONENTS) and not any("error" in t for _, t, _, _ in rows)
    assert "No decisions yet" in ML.track_record(ctx, store, _snap(), "scout")
    assert "meta.scout_bias=+0.00" in ML.scorecard_text(ctx, store, _snap())


# --------------------------------------------------------------------- bias knob + prompts
def test_bias_knob_is_clamped_and_reaches_the_prompt(ctx, store, root):
    acts, props = T.apply(store, ctx.cfg, [{"knob": "meta.scout_bias", "value": 5, "reason": "too timid"},
                                           {"knob": "meta.researcher_bias", "value": -0.6, "reason": "optimistic"}],
                          source="rv-1", now=1.0)
    assert not props and "(asked 5, clamped)" in acts[0]
    cfg = T.effective(ctx.cfg, store)
    assert cfg.meta.scout_bias == 1.0 and cfg.meta.researcher_bias == -0.6
    c2 = dataclasses.replace(ctx, cfg=cfg)
    assert "MORE speculative" in ML.track_record(c2, store, _snap(), "scout")
    assert "MORE conservative" in ML.track_record(c2, store, _snap(), "researcher")
    assert ML.bias_line(c2, "design") == ""                     # 0: no nudge

def test_an_out_of_range_spec_bias_is_refused_at_load(fake_loop_spec, tmp_path):
    from loop.spec import load_context
    bad = tmp_path / "bad.yaml"
    bad.write_text(fake_loop_spec.read_text().replace("\nloop:\n", "\nloop:\n  meta: {scout_bias: 2.0}\n", 1))
    with pytest.raises(ValueError, match="loop.meta"):
        load_context(bad, state_dir=tmp_path / "s")


def test_review_prompts_carry_records(ctx, store, root):
    from loop.agents.analyst import LLMAnalyst
    from loop.agents.llm import FakeLLMClient
    n = _node(store, "n1", status=NodeStatus.SCREENED_OUT)
    snap = _snap([_post("n1", -0.002, 0.2)])
    rv = {"summary": "s", "findings": [], "meta_hypotheses": []}
    llm = FakeLLMClient({"review": [rv, {"summary": "s", "assumptions": []}]})
    an = LLMAnalyst(ctx, llm)
    an.big_review(store, snap, trigger="manual")
    p = llm.calls[-1]["prompt"]
    assert "scorecard (component: headline" in p and "meta.<c>_bias" in p and "[design] " in p and "[review] " in p
    assert "sign right 0% of 1" in p                               # the researcher's line reaches the review
    an.big_review(store, snap, trigger="first_principles")
    fp = llm.calls[-1]["prompt"]
    assert "Your own track record as reviewer" in fp and "\nscout: " in fp and "[first_principles] " in fp


def test_researcher_prompt_carries_its_record_and_crossover_record(ctx, store, root):
    from loop.agents.researcher import LLMResearcher
    _node(store, "n1", pred=0.006)
    snap = _snap([_post("n1", -0.001, 0.3)])
    res = LLMResearcher(ctx, None, None)
    req = CandidateRequest(id="req-1", task_id=ctx.task_id, strategy=Strategy.EXPLORE, operator=Operator.DRAFT,
                           island="main", parent_ids=(ROOT_NODE_ID,), brief="b")
    p = res._prompt(req, store, snap, [store.get_node(ROOT_NODE_ID)], "c0", 0, "")
    assert "Your track record as researcher" in p and "predicted +0.0060, measured -0.0010" in p
    assert "Your crossovers" not in p
    xr = dataclasses.replace(req, operator=Operator.CROSSOVER, strategy=Strategy.CROSSOVER)
    assert "Your crossovers / compounds" in res._record(xr, store, snap)


def test_a_meta_resting_on_fallen_winners_goes_stale_and_stops_steering(ctx, store, root):
    """rv-29's m1 / m3 rested on n0308 / n0404 being bar-clearers, both since not_confirmed: such a meta is marked
    stale once, drops out of the open metas, its probes and crossover links, and the review table says so."""
    import time
    from loop.agents import briefs
    from loop.select import PortfolioSelector
    for nid in ("n0308", "n0404", "n0200"):
        _node(store, nid, status=NodeStatus.PROMOTING)
    store.log("node_transition", node_id="n0308", frm="screening", to="promoting", reason="")
    store.log("node_transition", node_id="n0200", frm="screening", to="screened_out", reason="")
    t = time.time() + 1
    for hid, claim in (("rv-29-m1", "salient masking clears the bar (n0308, n0404)"),
                       ("rv-29-m2", "regularisers hurt (n0200)")):
        store.put_hypothesis(HypothesisRecord(id=hid, task_id=store.task_id, kind=HypothesisKind.META, claim=claim,
                                              mechanism_tag="m", family="cross", predictions={}, falsification="f",
                                              levers=LeverTags({"regularization": 1}), review_id="rv-29", created_ts=t))
    store.put_review(ReviewReport(id="rv-29", task_id=store.task_id, trigger="cadence", n_evals_at=1, summary="s",
                                  findings=(), meta_hypothesis_ids=("rv-29-m1", "rv-29-m2"),
                                  probe_briefs=(("rv-29-m1", "probe a"), ("rv-29-m2", "probe b")), explore_bias={},
                                  created_ts=t))
    snap = _snap([_post("n0308", 0.01, 0.9), _post("n0404", 0.01, 0.9), _post("n0200", -0.01, 0.1)])
    assert ML.mark_stale(ctx, store, snap, now=t) == []                         # nothing fell yet
    store.update_node("n0308", status=NodeStatus.NOT_CONFIRMED)
    store.update_node("n0200", status=NodeStatus.SCREENED_OUT)                 # cited as a loser: no change
    acts = ML.mark_stale(ctx, store, snap, now=t)
    assert len(acts) == 1 and "rv-29-m1 STALE" in acts[0] and "n0308" in acts[0]
    assert ML.mark_stale(ctx, store, snap, now=t) == []                         # once
    assert "rv-29-m1" not in briefs.open_meta(store, snap) and "rv-29-m2" in briefs.open_meta(store, snap)
    assert "STALE (its evidence n0308 fell" in briefs.hypothesis_table(store, snap)
    sel = PortfolioSelector(ctx, None, None)
    assert [m for _, m, _ in sel.pending_probes(store)] == ["rv-29-m2"]


def test_a_hit_is_the_own_increment_not_the_inherited_carrier(store):
    """Live 2026-10-03: a child of n0255 had P(U>0) ~1 on n0255's +0.008 whatever its own change did; a hit is now
    its OWN increment over its measure parent (P(gain>0) > 0.5). The playbook built on the old hit is retired once."""
    from types import SimpleNamespace as NS
    from loop import meta
    snap = NS(nodes={"carried": NS(utility=NS(n=2), p_positive=0.99, p_gain=0.2),
                     "own": NS(utility=NS(n=2), p_positive=0.6, p_gain=0.8)})
    f = meta._Facts(store, snap)
    assert f.hit("carried") is False and f.hit("own") is True and f.hit("unscored") is None
    assert meta.retire_playbook(store, 1.0, "hit redefined") == ""
    meta.save_playbook(store, {"version": 3, "rules": [{"component": "researcher", "rule": "r", "evidence": "1"}]})
    assert meta.retire_playbook(store, 2.0, "hit redefined") == "playbook v4: 1 rule(s) invalidated (hit redefined)"
    assert meta.playbook(store)["rules"] == [] and meta.playbook(store)["retired"][0]["retired_in"] == 4
