"""RESEARCH SCOUT (loop/scout.py), the broadened operator focus, web research in researcher sessions, and the
concise / consolidated meta-learning records (loop/meta.py caps + playbook).
Run: uv run pytest tests/loop -q --confcutdir=tests/loop"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from loop import meta as ML
from loop import scout as SC
from loop import selftune as T
from loop.agents.llm import FakeLLMClient
from loop.contracts import (PANEL, ROOT_NODE_ID, AxisPrediction, HypothesisKind, HypothesisRecord, Node,
                            NodePosterior, NodeStatus, Operator, Posterior, ReviewReport, Strategy, WorldSnapshot)
from loop.spec import load_context

REPO = Path(__file__).resolve().parents[2]


def _snap(posts=()) -> WorldSnapshot:
    return WorldSnapshot(task_id="t", n_scored=len(posts), sigma={PANEL: 0.004}, variance={},
                         nodes={p.node_id: p for p in posts}, incumbent_id=ROOT_NODE_ID, family_means={},
                         levers=(), hypotheses={})


def _post(nid: str, mean: float, p: float) -> NodePosterior:
    return NodePosterior(node_id=nid, increment=Posterior(mean, 0.002, 2), utility=Posterior(mean, 0.002, 2),
                         axes={}, p_positive=p, p_beats_incumbent=p, vetoed=False)


def _node(store, nid: str, *, tag: str = "", notes: str = "", status=NodeStatus.SCREENED_OUT, pred=0.005,
          claim: str = "") -> Node:
    store.put_hypothesis(HypothesisRecord(
        id=f"h-{nid}", task_id=store.task_id, kind=HypothesisKind.TRIAL, claim=claim or f"idea {nid} " + "x" * 300,
        mechanism_tag=tag or f"{nid}_tag", family="optimization",
        predictions={PANEL: AxisPrediction(1 if pred > 0 else -1, abs(pred))}, falsification="f", created_ts=10.0))
    n = Node(id=nid, task_id=store.task_id, commit=f"c{nid}", parent_id=ROOT_NODE_ID, strategy=Strategy.EXPLORE,
             operator=Operator.DRAFT, family="optimization", mechanism_tag=tag or f"{nid}_tag", diff_hash=nid,
             status=status, hypothesis_id=f"h-{nid}", notes=notes, created_ts=10.0, updated_ts=10.0)
    store.put_node(n)
    return n


def _idea(title: str, **kw) -> dict:
    return {"title": title, "mechanism_tag": kw.pop("tag", ""), "family": kw.pop("family", "optimization"),
            "moonshot": kw.pop("moonshot", True), "jepa": kw.pop("jepa", False),
            "idea": kw.pop("idea", f"add {title} to train.py behind train.{title[:5]}"),
            "why_it_might_work": "a mechanism", "source": kw.pop("source", ["https://arxiv.org/abs/2401.00001"])}


@pytest.fixture
def sctx(ctx, tmp_path):
    bank = tmp_path / "bank_v1.jsonl"
    bank.write_text(json.dumps({"id": "v1-old", "title": "Sinkhorn centering of the teacher", "idea": "x"}) + "\n")
    scout = tmp_path / "idea_bank_scout.jsonl"
    cfg = dataclasses.replace(ctx.cfg, idea_banks=(str(bank), str(scout)),
                              scout=dataclasses.replace(ctx.cfg.scout, path=str(scout), max_ideas=4),
                              operator_focus="Bold diverse moonshots: entirely new losses; JEPA is one theme.")
    return dataclasses.replace(ctx, cfg=cfg)


# ------------------------------------------------------------------ operator focus + spec
def test_v2_spec_focus_is_neutral_procedural_and_the_scout_bank_is_served(tmp_path):
    """The focus teaches the first-principles PROCEDURE and the hard constraints and names NO mechanism (a theme
    named once in it became 248 nodes)."""
    import re
    ctx = load_context(REPO / "tasks" / "nanopath_v2.yaml", state_dir=tmp_path / "state")
    focus = ctx.cfg.operator_focus
    for must in ("first principles", "mechanistic model", "binding constraint", "measured quantity", "cite",
                 "1M-sample / 1e18-FLOP", "TCGA-only", "init-compatibility"):
        assert must in focus, must
    assert not re.search(r"jepa|kde|fino|ibot|teacher|ema\b|gradient balanc|token|stain|sampling", focus, re.I)
    assert ctx.cfg.researcher.max_concurrent == 4
    # 10-22: the scout section is gone from the YAML (v2 driver only); the public spec ships no idea banks
    assert ctx.cfg.idea_banks == ()


def test_focus_text_leads_ideation_and_scout_prompts(sctx, store):
    from loop.agents import briefs
    store.ensure_root("c0")
    assert briefs.brief("literature", store, _snap(), ctx=sctx).startswith(
        briefs.FOCUS_HEADING + "\n" + sctx.cfg.operator_focus)
    p = SC.Scout(sctx, None).prompt(store, _snap(), 1790000000.0)
    assert sctx.cfg.operator_focus in p and "v1-old" in p and "2026-" in p


def test_bank_interleaves_families_and_privileges_no_named_theme(tmp_path, store):
    """A theme named in the operator focus gets no priority (a mechanism named once became 248 nodes); ideas
    interleave across families instead."""
    from loop.select import bank as B
    from tests.loop.test_select import make_cfg
    p = tmp_path / "b.jsonl"
    rows = [("j1", "objective"), ("j2", "objective"), ("j3", "objective"), ("o1", "data"), ("o2", "data")]
    p.write_text("".join(json.dumps({"id": i, "title": i, "idea": "x", "jepa": True, "family": f}) + "\n"
                         for i, f in rows))
    got = [e["id"] for e in B.pending(store, make_cfg(idea_banks=(str(p),), operator_focus="JEPA is one theme"))]
    assert got == ["j1", "o1", "j2", "o2", "j3"]


# ------------------------------------------------------------------ scout: write / dedupe / append
def test_scout_writes_valid_jsonl_dedupes_and_appends(sctx, store):
    store.ensure_root("c0")
    _node(store, "n1", tag="barlow_redundancy_loss")
    res = {"trigger": "first", "ts": 1790000000.0, "ok": True, "searches": 7, "fetches": 3, "ideas": [
        _idea("VICReg variance term on patch tokens", tag="vicreg_patch_variance"),
        _idea("Barlow twins redundancy loss", tag="barlow_redundancy_loss"),        # tried mechanism
        _idea("Sinkhorn centering of the teacher"),                                  # bank title near-dup
        _idea("VICReg variance term on the patch tokens"),                           # within-batch near-dup
        _idea("Uncited trick", source=[]),                                           # refused: uncited
        _idea("Score quoting", idea="reached 0.6514 in v1"),                         # refused: quotes a score
        _idea("Token merging for more views", family="not_a_family", jepa=True)]}
    run = SC.write(sctx, store, res)
    lines = [json.loads(l) for l in Path(SC.path(sctx)).read_text().splitlines()]
    assert [l["id"] for l in lines] == ["scout-vicreg_patch_variance",               # the mechanism tag names it
                                        "scout-token_merging_for_more_views"]
    need = {"id", "title", "family", "moonshot", "jepa", "idea", "why_it_might_work", "v1_fate",
            "why_v1_verdict_may_not_transfer", "source"}
    assert all(set(l) == need for l in lines)
    assert lines[0]["v1_fate"].startswith("n/a (scouted 2026-") and lines[0]["why_v1_verdict_may_not_transfer"] == ""
    assert lines[1]["family"] == "" and lines[1]["jepa"] is True and lines[0]["source"][0].startswith("https://")
    assert (run["written"], run["dup"], run["refused"], run["searches"]) == (2, 3, 2, 7)
    # the bank serves them (source list joined), and a second run APPENDS and dedupes against the first
    from loop.select import bank as B
    served = {e["id"]: e for e in B.pending(store, sctx.cfg)}
    assert "scout-token_merging_for_more_views" in served and served["v1-old"]
    assert served["scout-vicreg_patch_variance"]["source"] == "https://arxiv.org/abs/2401.00001"
    SC.write(sctx, store, {**res, "ts": 1790030000.0, "ideas": [_idea("Token merging for more views"),
                                                                 _idea("Stain-mixup curriculum")]})
    ids = [json.loads(l)["id"] for l in Path(SC.path(sctx)).read_text().splitlines()]
    assert ids == ["scout-vicreg_patch_variance", "scout-token_merging_for_more_views",
                   "scout-stain_mixup_curriculum"]
    assert len(SC.book(store)["runs"]) == 2 and ML.decisions(store, "scout")


def test_scout_write_dedupes_a_bank_ideas_mechanism_by_cosine(sctx, store):
    """Live bank: 'whitened_latent_target' then 'whitened_jepa_target' (cos 0.65), slide / stain balanced sampling
    (0.50): the second of each is a duplicate by novelty.mech_cos over tag + title."""
    store.ensure_root("c0")
    res = {"trigger": "first", "ts": 1790000000.0, "ok": True, "ideas": [
        _idea("Whitened latent target for JEPA regression", tag="whitened_latent_target"),
        _idea("Stain diversity balanced sampling of tiles", tag="stain_diversity_balanced_sampling"),
        _idea("Gram anchoring with a lagged self teacher", tag="gram_anchoring_selfteacher"),
        _idea("Whitened JEPA target: ZCA whitening of teacher patch latents", tag="whitened_jepa_target"),
        _idea("Slide balanced sampling of training tiles", tag="slide_balanced_sampling")]}
    run = SC.write(sctx, store, res)
    ids = [json.loads(l)["id"] for l in Path(SC.path(sctx)).read_text().splitlines()]
    assert ids == ["scout-whitened_latent_target", "scout-stain_diversity_balanced_sampling",
                   "scout-gram_anchoring_selfteacher"] and run["dup"] == 2
    op = {**_idea("Whitened JEPA target: ZCA whitening of teacher patch latents", tag="zca_jepa_target_v2"),
          "priority": True}                                  # an operator's deliberate variant passes (ANVIL2 / Muon)
    run = SC.write(sctx, store, {**res, "ideas": [op, {**op, "mechanism_tag": "whitened_jepa_target"}]})
    assert run["written"] == 1 and json.loads(Path(SC.path(sctx)).read_text().splitlines()[-1])["priority"] is True


def test_scout_run_uses_web_and_the_fake_llm(sctx, store):
    store.ensure_root("c0")
    llm = FakeLLMClient({"scout": [{"tool_calls": [("WebSearch", {"query": "new SSL loss 2025"}),
                                                   ("WebSearch", {"query": "pathology FM"}),
                                                   ("WebFetch", {"url": "https://arxiv.org/abs/2401.00001"})],
                                    "finish": {"ideas": [_idea("Kernel dependence regulariser")]}}]})
    res = SC.Scout(sctx, llm).run(store, _snap(), trigger="first", now=1790000000.0)
    call = llm.calls[0]
    assert call["web"] is True and call["role"] == "scout" and call["tools"] == ["notes", "note"]
    assert "Stage 1 -- RESEARCH" in call["prompt"]               # the research stage (loop.agents.notes)
    assert "Hard constraints" in call["prompt"] and "8-15" not in call["prompt"]      # min/max from the cfg
    assert res["ok"] and (res["searches"], res["fetches"]) == (2, 1) and len(res["ideas"]) == 1


# ------------------------------------------------------------------ cadence + self-tune
def test_scout_cadence_failed_retry_stall_and_selftune(sctx, store):
    now = 1790000000.0
    due = lambda t, since=0, marker=0: SC.due(sctx, store, t, evals_since_confirm=since, confirm_marker=marker)  # noqa: E731
    assert due(now) == "first"
    SC.write(sctx, store, {"trigger": "first", "ts": now, "ok": True, "ideas": []})
    assert due(now + 3600) == "" and due(now + 6 * 3600) == "cadence"
    stall = sctx.cfg.review.stall_evals
    assert due(now + 3 * 3600, since=stall, marker=5) == "stall"            # a stall runs one early ...
    SC.write(sctx, store, {"trigger": "stall", "ts": now + 3 * 3600, "ok": True, "ideas": []}, confirm_marker=5)
    assert due(now + 6 * 3600, since=stall + 9, marker=5) == ""             # ... once per stall
    SC.write(sctx, store, {"trigger": "cadence", "ts": now + 9 * 3600, "ok": False, "error": "down", "ideas": []})
    assert due(now + 9.5 * 3600) == "" and due(now + 10 * 3600) == "cadence"   # a failed run retries after 1 h
    # self-tune: scout.every_hours is whitelisted and clamped to 2..24
    T.apply(store, sctx.cfg, [{"knob": "scout.every_hours", "value": 100, "reason": "r"}], source="rv-1", now=now)
    assert T.effective(sctx.cfg, store).scout.every_hours == 24.0
    T.apply(store, sctx.cfg, [{"knob": "scout.every_hours", "value": 0.5, "reason": "r"}], source="rv-2", now=now)
    assert T.effective(sctx.cfg, store).scout.every_hours == 2.0
    assert any(l.startswith("scout    3 runs") for l in SC.lines(sctx, store, now))


def test_driver_runs_the_scout_and_harvests(sctx, store, tmp_path):
    from tests.loop.conftest import InlineExecutor
    from loop.driver import Driver

    class Stub:
        def __init__(self):
            self.calls = []

        def run(self, store, snap, *, trigger, now):
            self.calls.append(trigger)
            return {"trigger": trigger, "ts": now, "ok": True, "searches": 3, "fetches": 0,
                    "ideas": [_idea("Spectral contrastive loss")]}

    class Null:
        def __getattr__(self, name):
            raise AttributeError(name)

    d = Driver.__new__(Driver)                   # only the scout plumbing is exercised
    d.ctx, d.store, d.clock, d.snapshot, d.analyst, d.scout = sctx, store, lambda: 1790000000.0, None, Null(), Stub()
    d._scout, d._consolidation, d._fails, d.pool = None, None, {}, InlineExecutor()
    d._n_evals = lambda: 0
    acts: list[str] = []
    d._meta_jobs(acts)
    d._harvest_meta(acts)
    assert d.scout.calls == ["first"] and any("1 idea(s) written" in a for a in acts)
    assert "scout-spectral_contrastive_loss" in Path(SC.path(sctx)).read_text()


# ------------------------------------------------------------------ web research in researcher sessions
def test_web_split_and_scout_component(ctx, store):
    store.ensure_root("c0")
    posts = []
    for i, (notes, hit) in enumerate([("bank=scout-a web=1 searches=3 fetches=1", 1),
                                      ("bank=scout-b web=1 searches=2 fetches=0", 0),
                                      ("bank=scout-c web=1 searches=4 fetches=2", 1),
                                      ("web=0", 0), ("web=0", 0), ("", 1)]):
        _node(store, f"n{i}", notes=f"request=r{i} brief=literature {notes}")
        posts.append(_post(f"n{i}", 0.002 if hit else -0.002, 0.9 if hit else 0.1))
    SC.write(ctx, store, {"trigger": "first", "ts": 1.0, "ok": True, "ideas": []})
    r = ML.resolve(ctx, store, _snap(posts), "scout")
    assert "1 scout runs" in r["headline"] and "3 became nodes: hit 2/3" in r["headline"]
    assert "web research sessions hit 2/3 (67%) vs no-web 0/2 (0%)" in r["headline"]
    assert "web-researched ideas hit 2/3" in ML.resolve(ctx, store, _snap(posts), "researcher")["headline"]
    assert "scout" in ML.COMPONENTS and "scout" in dict((c, 0) for c, *_ in ML.scorecard(ctx, store, _snap(posts)))


# ------------------------------------------------------------------ concise records
def test_every_component_record_fits_the_prompt_caps(ctx, store):
    store.ensure_root("c0")
    posts = []
    for i in range(12):
        _node(store, f"n{i}", pred=0.02, notes=f"request=r{i} brief=moonshot bank=scout-x{i} web={i % 2} "
                                                "persona=" + "contrarian" * 30)
        posts.append(_post(f"n{i}", 0.003 * (-1) ** i, 0.9 if i % 2 else 0.1))
    snap = _snap(posts)
    biased = dataclasses.replace(ctx, cfg=dataclasses.replace(
        ctx.cfg, meta=dataclasses.replace(ctx.cfg.meta, researcher_bias=0.8, scout_bias=-0.5, design_bias=0.1)))
    for c in ML.COMPONENTS:
        txt = ML.track_record(biased, store, snap, c)
        assert len(txt) <= ML.MAX_RECORD_CHARS and len(txt.splitlines()) <= ML.MAX_RECORD_LINES, (c, txt)
        head = ML.resolve(biased, store, snap, c)["headline"]
        assert txt.startswith(head[:40]) or txt.startswith("No decisions yet"), c   # the headline is never dropped
        assert ("Reviews set your bias" in txt) == (c in ("researcher", "scout")), c  # |b| >= 0.25 only
    rec = ML.track_record(biased, store, snap, "researcher")
    assert "Calibration:" in rec and sum(l.startswith(("Held up", "Did NOT")) for l in rec.splitlines()) <= 2
    card, own = ML.meta_block(biased, store, snap, ("review", "design"))
    assert len(card) + len(own) + 1 <= ML.MAX_META_BLOCK
    assert all(f"{c}:" in card for c in ML.COMPONENTS if c not in ("review", "design"))
    assert len(ML.meta_block(biased, store, snap, ("first_principles",))[0]) <= ML.MAX_META_BLOCK
    assert "Calls that held up" in ML.raw_record(biased, store, snap, "researcher")   # the long form stays


# ------------------------------------------------------------------ consolidation / playbook
def _review(store, rid: str, trigger: str, ts: float) -> None:
    store.put_review(ReviewReport(id=rid, task_id=store.task_id, trigger=trigger, summary="s", findings=("f",),
                                  meta_hypothesis_ids=(), probe_briefs=(), explore_bias={}, n_evals_at=1,
                                  created_ts=ts))


def test_consolidation_replaces_records_validates_numbers_and_retires(ctx, store):
    from loop.agents.analyst import FP_TRIGGER
    store.ensure_root("c0")
    posts = []
    for i, m in enumerate([0.001, -0.002, 0.0005, -0.001, 0.0008]):
        _node(store, f"n{i}", pred=0.006)
        posts.append(_post(f"n{i}", m, 0.8 if m > 0 else 0.2))
    snap = _snap(posts)
    assert not ML.consolidation_due(store, ctx.cfg, FP_TRIGGER)
    _review(store, "rv-1", "cadence", 5.0)
    assert not ML.consolidation_due(store, ctx.cfg, FP_TRIGGER)
    _review(store, "rv-2", FP_TRIGGER, 6.0)
    assert ML.consolidation_due(store, ctx.cfg, FP_TRIGGER)                   # with every FP review
    raw = ML.track_record(ctx, store, snap, "researcher")
    assert "sign right 60% of 5" in raw and "Calibration:" in raw
    llm = FakeLLMClient({"review": [
        {"rules": [{"component": "researcher", "rule": "Predict smaller sizes: you over-predict",
                    "evidence": "sign right 60% of 5 scored"},
                   {"component": "researcher", "rule": "JEPA ideas win 83% of the time",          # invented
                    "evidence": "n=41"},
                   {"component": "design", "rule": "Move less", "evidence": "no numbers"}]},      # uncited
        {"rules": [{"component": "scout", "rule": "Keep scouting new losses", "evidence": "60% of 5"}]}]})
    pb = ML.consolidate(ctx, llm, store, snap, now=10.0)
    assert [r["rule"] for r in pb["rules"]] == ["Predict smaller sizes: you over-predict"]
    assert {d["why"][:8] for d in pb["dropped"]} == {"invented", "cites no"}
    ML.save_playbook(store, pb)
    assert not ML.consolidation_due(store, ctx.cfg, FP_TRIGGER)
    rec = ML.track_record(ctx, store, snap, "researcher")
    assert rec.startswith("Now: sign right 60% of 5") and "Playbook v1: Predict smaller sizes" in rec
    assert "Calibration:" not in rec and "Did NOT hold up" not in rec          # the playbook replaced the record
    assert "Calibration:" in ML.track_record(ctx, store, snap, "scout") or "No decisions" in \
        ML.track_record(ctx, store, snap, "scout")                               # no rule: the concise record
    for i in (3, 4, 5):
        _review(store, f"rv-{i}", "cadence", 10.0 + i)
    assert ML.consolidation_due(store, ctx.cfg, FP_TRIGGER)                   # every consolidate_every reviews
    pb2 = ML.consolidate(ctx, llm, store, snap, now=20.0)
    ML.save_playbook(store, pb2)
    assert pb2["version"] == 2 and [r["rule"] for r in pb2["retired"]] == ["Predict smaller sizes: you over-predict"]
    rec = ML.track_record(ctx, store, snap, "researcher")
    assert "Predict smaller" not in rec and "Calibration:" in rec             # retired: never in a prompt
    assert "Keep scouting" in ML.track_record(ctx, store, snap, "scout")
    assert store.get_state("meta.playbook.v1")["rules"][0]["component"] == "researcher"
    assert "Predict smaller" not in "\n".join(ML.playbook_lines(store))
    assert store.events("playbook_rule_dropped")


def test_rule_number_check():
    corpus = "sign right 60% of 5 scored; predicted +0.0060 vs measured -0.0020; hit 2/3 (67%)"
    assert ML.check_rule({"rule": "shrink sizes", "evidence": "measured -0.002 vs +0.006 over 5"}, corpus) == ""
    assert ML.check_rule({"rule": "hit 67% (2/3)", "evidence": "60%"}, corpus) == ""
    assert ML.check_rule({"rule": "hit 70%", "evidence": "n=5"}, corpus).startswith("invented")
    assert ML.check_rule({"rule": "be bolder", "evidence": "trust me"}, corpus).startswith("cites no")
