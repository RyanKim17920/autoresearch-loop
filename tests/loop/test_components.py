"""Causal component audits (loop.components): seeded ablation gates, with-vs-without outcomes, verdicts and the
automatic down / off / re-probe actions; protected machinery is never a component.
Run: uv run pytest tests/loop -q --confcutdir=tests/loop"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from loop import components as CG
from loop.agents.analyst import FP_TRIGGER
from loop.contracts import OPTIONAL_COMPONENTS, PROTECTED_COMPONENTS, ROOT_NODE_ID, Node, NodeStatus, Strategy
from loop.spec import ComponentsCfg
from tests.loop.test_driver import Scripts, _make, _world


def _ctx(ctx, **kw):
    return dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, components=ComponentsCfg(**kw)))


def _snap(hits):
    """node id -> hit (True / False): a scored posterior."""
    return SimpleNamespace(nodes={i: SimpleNamespace(utility=SimpleNamespace(n=2, mean=0.01 if h else -0.01),
                                                     p_positive=0.9 if h else 0.1,
                                                     p_gain=0.9 if h else 0.1) for i, h in hits.items()})


def _node(store, nid, *, notes="", status=NodeStatus.SCREENED_OUT, bonus=0.0):
    store.put_node(Node(id=nid, task_id=store.task_id, commit="c", parent_id=ROOT_NODE_ID, strategy=Strategy.EXPLORE,
                        operator=None, family="optim", mechanism_tag=f"t{nid}", diff_hash=f"h{nid}", status=status,
                        hypothesis_id=f"h-{nid}", notes=notes, prior_bonus=bonus, created_ts=1.0))


def test_gate_is_seeded_logged_and_refuses_protected(ctx, store):
    c = _ctx(ctx, ablation_rate=0.3)
    for name in PROTECTED_COMPONENTS + ("nope",):
        with pytest.raises(ValueError):
            CG.use(c, store, name, "x", 1)
    CG.take_lines()
    got = [CG.use(c, store, "web", "research", i, request=f"r{i}") for i in range(400)]
    assert 0.2 < 1 - sum(got) / len(got) < 0.4
    ev = store.events(CG.K_GATE)
    assert len(ev) == 400 and [e["used"] for e in ev] == got and ev[0]["request"] == "r0"
    lines = CG.take_lines()
    assert len(lines) == got.count(False) and lines[0].startswith("ablate web research")
    store.set_state(CG.K_DRAWS, 0)                           # same seed, same draw index -> same decisions
    assert [CG.use(c, store, "web", "research", i) for i in range(400)] == got
    assert CG.use(_ctx(ctx, ablation_rate=0.0), store, "lever_prior", "rank", "k") is True    # no experiment
    assert CG.use(_ctx(ctx, enabled={"scout": False}), store, "scout", "run", "k") is False    # operator: off
    s = [CG.use(_ctx(ctx, ablation_rate=0.5), store, "scout", "run", "n1", sticky=True) for _ in range(20)]
    assert len(set(s)) == 1                                  # a node-level decision is drawn once


def test_rates_follow_the_mode(ctx, store):
    c = _ctx(ctx, ablation_rate=0.15, down_rate=0.5, reprobe_rate=0.05, rates={"web": 0.3})
    assert CG.rate(c.cfg, store, "lever_prior") == 0.15 and CG.rate(c.cfg, store, "web") == 0.3
    store.set_state(CG.K_MODES, {"lever_prior": {"mode": "down"}, "scout": {"mode": "off"}})
    assert CG.rate(c.cfg, store, "lever_prior") == 0.5 and CG.rate(c.cfg, store, "scout") == pytest.approx(0.95)


def test_spec_refuses_protected_or_unknown_component(fake_loop_spec, tmp_path):
    from loop.spec import load_context
    for bad in ("{enabled: {held_out_audit: false}}", "{rates: {foo: 0.1}}", "{ablation_rate: 1.5}"):
        p = tmp_path / "bad.yaml"
        p.write_text(fake_loop_spec.read_text().replace("  components: {ablation_rate: 0}",
                                                        f"  components: {bad}", 1))
        with pytest.raises(ValueError, match="loop.components"):
            load_context(p, state_dir=tmp_path / "s")
    assert set(OPTIONAL_COMPONENTS) == set(CG.REGISTRY) and not set(OPTIONAL_COMPONENTS) & set(PROTECTED_COMPONENTS)


def test_audit_verdicts_and_actions(ctx, store):
    """web: bypassed ideas hit as often as the rest -> no value -> down, and off at >= 2x min_ablated;
    review_probes: WITH hits far more -> useful (an off component comes back); scout: WITH never hits -> harmful."""
    c = _ctx(ctx, ablation_rate=0.15, min_ablated=10, equivalence_margin=1.0)
    hits, k = {}, 0

    def decide(comp, used, hit):
        nonlocal k
        k += 1
        nid = f"n{k:04d}"
        _node(store, nid, notes=f"request=req-{k}")
        store.log(CG.K_GATE, component=comp, decision="d", key=str(k), used=used, request=f"req-{k}")
        hits[nid] = hit

    for i in range(24):
        decide("web", True, i % 2 == 0)
        decide("web", False, i % 2 == 0)
    for i in range(12):
        decide("review_probes", True, True)
        decide("review_probes", False, i % 4 == 0)
        decide("scout", True, False)
        decide("scout", False, True)
    store.set_state(CG.K_MODES, {"review_probes": {"mode": "off"}})
    rows = {r["component"]: r for r in CG.evaluate(c, store, _snap(hits))}
    assert rows["web"]["verdict"] == "no_measurable_value" and rows["web"]["n"] == 24 and rows["web"]["next"] == "off"
    assert rows["review_probes"]["verdict"] == "useful" and rows["review_probes"]["next"] == "on"
    assert rows["scout"]["verdict"] == "harmful" and rows["scout"]["next"] == "off"
    assert rows["lever_prior"]["verdict"] == "insufficient" and rows["lever_prior"]["next"] == "on"
    assert rows["web"]["with"]["hits"] == 12 and rows["web"]["without"]["scored"] == 24
    acts = CG.audit(c, store, _snap(hits), 100.0)
    assert any(a.startswith("component web: on -> off") for a in acts)
    assert {k: v["mode"] for k, v in CG.modes(store).items()} == {"web": "off", "review_probes": "on", "scout": "off"}
    assert CG.next_mode("on", "no_measurable_value", 25, 10, randomised=False) == "down"   # natural never off
    assert CG.next_mode("on", "inconclusive", 25, 10) == "on" and CG.next_mode("down", "inconclusive", 25, 10) == "down"
    assert CG.next_mode("on", "no_measurable_value", 12, 10) == "down"      # N <= n < 2N: downweight first
    assert "web: no_measurable_value" in CG.review_block(store)
    assert any(ln.lstrip().startswith("web") for ln in CG.lines(c, store, _snap(hits)))


def test_driver_ablates_logs_and_audits_with_first_principles(tmp_path):
    ctx = _world(tmp_path, review_every=1000)
    ctx = _ctx(ctx, ablation_rate=0.0, rates={"lever_prior": 0.99, "web": 0.99})
    driver, store, _ = _make(ctx, Scripts())
    log = []
    for _ in range(6):
        log += driver.tick()
    gates = store.events(CG.K_GATE)
    comps = {e["component"] for e in gates}
    assert "web" in comps
    assert any(ln.startswith("ablate web research") for ln in log)
    assert [n for n in store.nodes() if n.id != ROOT_NODE_ID]
    web = [e for e in gates if e["component"] == "web" and not e["used"]]
    assert all(" web=0" in n.notes for n in store.nodes() if any(f"request={e['request']} " in n.notes for e in web))
    acts: list[str] = []
    driver.start_review(FP_TRIGGER, acts)
    assert store.get_state(CG.K_LAST)["rows"] and store.events("component_audit")
    driver.drain()


def test_contrast_is_stratified_by_bypass_rate(ctx, store):
    """A mode change (bypass 0.15 -> 0.95) coinciding with a time trend: pooled arms would call the component
    useful (51/54 vs 9/66); within each rate stratum with == without, so no measurable value."""
    c = _ctx(ctx, ablation_rate=0.15, min_ablated=10, equivalence_margin=1.0)
    hits, k = {}, 0
    for rate, used, n, hit in ((0.15, True, 51, True), (0.15, False, 9, True),
                               (0.95, True, 3, False), (0.95, False, 57, False)):
        for _ in range(n):
            k += 1
            _node(store, f"n{k:04d}", notes=f"request=req-{k}")
            store.log(CG.K_GATE, component="web", decision="d", key=str(k), used=used, rate=rate, request=f"req-{k}")
            hits[f"n{k:04d}"] = hit
    row = next(r for r in CG.evaluate(c, store, _snap(hits)) if r["component"] == "web")
    assert row["effect"] == pytest.approx(0.0) and row["verdict"] == "no_measurable_value"
    assert row["with"]["hits"] == 51 and row["without"]["scored"] == 66


def test_equivalence_rule_underpowered_null_is_inconclusive_powered_null_is_off(ctx, store):
    """0.00 +- 0.14 (n=24/arm) cannot rule out a real effect -> inconclusive, mode unchanged; the same null
    from 1000/arm (upper bound < 0.05) is no_measurable_value -> off."""
    c = _ctx(ctx, ablation_rate=0.15, min_ablated=10)
    assert CG.verdict(0.0, 0.144, 24, 10, 0.05) == "inconclusive"
    assert CG.verdict(0.0, 0.0158, 1000, 10, 0.05) == "no_measurable_value"
    assert CG.verdict(0.3, 0.05, 24, 10, 0.05) == "useful" and CG.verdict(-0.3, 0.05, 24, 10, 0.05) == "harmful"
    hits, k = {}, 0
    for used in (True, False):
        for i in range(24):
            k += 1
            _node(store, f"n{k:04d}", notes=f"request=req-{k}")
            store.log(CG.K_GATE, component="web", decision="d", key=str(k), used=used, request=f"req-{k}")
            hits[f"n{k:04d}"] = i % 2 == 0
    row = next(r for r in CG.evaluate(c, store, _snap(hits)) if r["component"] == "web")
    assert row["verdict"] == "inconclusive" and row["next"] == row["mode"] == "on"
    assert CG.audit(c, store, _snap(hits), 100.0) == [] or not any("component web" in a for a in CG.audit(c, store, _snap(hits), 100.0))
    assert CG.mode(c.cfg, store, "web") == "on"
