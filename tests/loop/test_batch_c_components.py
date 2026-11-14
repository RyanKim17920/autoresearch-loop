"""10-22 batch C3: the component_* events stopped 10-08 (ComponentsCfg had 0 live reads). Ported minimal: each optional v3 context
part (web research, parent postmortems) is bypassed on a seeded share of research sessions through ``request.ablated`` (as
fork-and-flush does), and the meta report carries the causal audit. Report-only: the v2 stored modes are not read."""

from __future__ import annotations

import dataclasses
import time

from loop import components as CG
from loop import tick as TK
from loop.contracts import Operator, Strategy
from loop.core import data as CD
from loop.core import ideation as ID

from .test_runner import _node
from .test_set1_harness import _Held


def _c(ctx, **kw):
    return dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, components=dataclasses.replace(ctx.cfg.components, **kw)))


def test_draw_is_seeded_rate_bound_logged_and_ignores_the_v2_modes(ctx, store):
    store.set_state(CG.K_MODES, {"web": {"mode": "off"}})            # the v2 era audited web 'off': the v3 draw must not read it
    always, never = _c(ctx, ablation_rate=1.0), _c(ctx, ablation_rate=0.0)
    assert CG.draw(always, store, "web", "explore", "r1", request="r1") is False
    assert CG.draw(never, store, "web", "explore", "r2", request="r2") is True
    assert [e["request"] for e in store.events(CG.K_GATE)] == ["r1"] and store.events(CG.K_GATE)[0]["v"] == 3   # rate 0: no experiment
    ids = [CG.draw(_c(ctx, ablation_rate=0.15), store, "postmortem", "x", f"k{i}") for i in range(400)]
    assert 0.08 < 1 - sum(ids) / 400 < 0.22                          # ~ the 0.15 default bypass share
    assert CG.draw(_c(ctx, enabled={"web": False}), store, "web", "x", "k", request="k") is False
    try:
        CG.draw(ctx, store, "scout", "x", "k")
        raise AssertionError("scout is not a v3 component")
    except ValueError:
        pass


def test_the_real_ideation_request_ablates_through_request_ablated(ctx, store, tmp_path):
    from loop.core.ideation import Brief
    store.ensure_root("abc123")
    ide = ID.Ideation(_c(ctx, ablation_rate=1.0), store)
    b = Brief(kind="explore", lane="explore", parents=("n1",), text="t")
    req = ide.request(b)
    assert req.strategy is Strategy.EXPLORE and "postmortem" in req.ablated
    from loop.agents.researcher import wants_web
    assert ("web" in req.ablated) == wants_web(dataclasses.replace(req, ablated=()), ide.ctx)   # drawn iff it would use web
    assert {e["component"] for e in store.events(CG.K_GATE)} >= {"postmortem"} and all(e["request"] == req.id for e in store.events(CG.K_GATE))
    flushed = ide.request(dataclasses.replace(b, branch="1.0", ablated=("postmortem", "handoffs")))   # fork-and-flush: exempt
    assert flushed.ablated == ("postmortem", "handoffs") and len(store.events(CG.K_GATE)) == len(
        [e for e in store.events(CG.K_GATE) if e["request"] == req.id])
    off = ID.Ideation(_c(ctx, ablation_rate=0.0), store).request(b)
    assert off.ablated == ()
    root = ID.Ideation(_c(ctx, ablation_rate=1.0), store).request(dataclasses.replace(b, parents=()))
    assert "postmortem" not in root.ablated                         # no parent, no postmortem to bypass
    dry = ID.Ideation(_c(ctx, ablation_rate=1.0), store)
    dry._dry = True
    assert dry.request(b).ablated == ()


def _gate(store, comp, req, used, rate=0.15):
    store.log(CG.K_GATE, component=comp, decision="d", key=req, used=used, rate=rate, request=req, v=3)


def test_the_section_contrasts_with_and_without_and_ignores_v2_and_outage_events(ctx, store):
    nodes, beliefs, gpu = {}, {}, {}
    cfg = _c(ctx, min_ablated=3, ablation_rate=0.15)
    for i in range(12):                                             # used: 8 of 12 hit; bypassed (6): 0 hit
        rid, used = f"req-{i}", i < 6 or i % 2 == 0
        _gate(store, "web", rid, used)
        n = _node(store, f"n{i}")
        store.update_node(n.id, notes=f"request={rid} brief=explore")
        nodes[n.id] = store.get_node(n.id)
        beliefs[n.id], gpu[n.id] = ((0.01 if used else -0.01), 1e-4, 3), 2.0
    store.log(CG.K_GATE, component="web", decision="d", key="old", used=False, rate=0.15, request="req-0")   # v2: no v=3
    sec = CG.v3_section(cfg, store, nodes, gpu, beliefs, {"n0"})
    web = next(r for r in sec["rows"] if r["component"] == "web")
    assert web["with"]["decisions"] == 9 and web["without"]["decisions"] == 3 and web["n"] == 3     # the v2 draw is not counted
    assert web["with"]["hits"] == 9 and web["without"]["hits"] == 0 and web["with"]["conf"] == 1
    assert web["effect"] > 0.4 and web["verdict"] == "useful"
    pm = next(r for r in sec["rows"] if r["component"] == "postmortem")
    assert pm["verdict"] == "insufficient" and sec["applied"] is False
    assert any("web (bypass 0.15): useful" in x for x in CG.v3_text(sec))
    assert CG.v3_section(_c(ctx, min_ablated=30), store, nodes, gpu, beliefs, set())["rows"][0]["verdict"] == "insufficient"


def test_the_ticks_meta_report_carries_the_audit(ctx, store, tmp_path):
    c = _c(dataclasses.replace(ctx, state_dir=str(tmp_path)), min_ablated=1, ablation_rate=0.15)
    tick = TK.Tick(c, store, _Held(), TK.ModuleCore(), db_path=tmp_path / "loop.db", clock=time.time)
    store.ensure_root("abc123")
    _gate(store, "web", "req-1", False)
    led = CD.load(tmp_path / "loop.db", ctx.task_id)
    tick.meta.cycle(led)
    import json
    rep = json.loads(next((tmp_path / "meta").glob("*.json")).read_text())
    assert "components" not in rep["errors"], rep["errors"]      # even with no noise fit (no beliefs): the draws still count
    assert {r["component"] for r in rep["components"]["rows"]} == set(CG.V3_COMPONENTS)
    assert "## Component audit" in next((tmp_path / "meta").glob("*.md")).read_text()
    tick.close()


def _gate2(store, comp, req, used, kind, rate=0.15):
    store.log(CG.K_GATE, component=comp, decision=kind, key=req, used=used, rate=rate, request=req, v=3)


def test_the_audit_is_stratified_by_brief_kind_clusters_by_session_and_counts_unscored_nodes_as_misses(ctx, store):
    """S6: pooled across brief kinds, a bypass drawn mostly on the weak kind looked harmful (Simpson); node-level se
    overstated n; unscored nodes were dropped (bypass-induced attrition looked like a gain)."""
    nodes, beliefs, gpu = {}, {}, {}
    cfg = _c(ctx, min_ablated=3, ablation_rate=0.15)
    k = 0

    def session(kind, used, hits, unscored=0):
        nonlocal k
        rid = f"req-{k}"
        k += 1
        _gate2(store, "web", rid, used, kind)
        for j in range(hits + unscored):
            n = _node(store, f"n{k}x{j}")
            store.update_node(n.id, notes=f"request={rid} brief={kind}")
            nodes[n.id] = store.get_node(n.id)
            beliefs[n.id] = ((0.01 if j < hits else -0.01), 1e-4, 3 if j < hits else 1)     # the last ones: 1 seed = unscored
            gpu[n.id] = 1.0
    for _ in range(8):                      # strong kind: used 8 sessions hit, bypassed 2 sessions hit: no effect inside it
        session("explore", True, 1)
    for _ in range(2):
        session("explore", False, 1)
    for _ in range(2):                      # weak kind: used rarely, bypassed often, both never hit
        session("bold", True, 0, unscored=1)
    for _ in range(8):
        session("bold", False, 0, unscored=1)
    row = next(r for r in CG.v3_section(cfg, store, nodes, gpu, beliefs, set())["rows"] if r["component"] == "web")
    assert abs(row["effect"]) < 1e-6                           # stratified: no effect (pooled it would be strongly positive)
    assert row["without"]["decisions"] == 10 and row["n"] == 2  # n counts ablated SESSIONS with a scored node, not nodes
    assert row["verdict"] == "insufficient"                      # 2 scored units < min_ablated 3
    # intention to treat: sessions whose nodes were all unscored count as a miss (in the contrast), not as missing
    for i in range(3):
        session("explore", False, 0, unscored=2)
    row = next(r for r in CG.v3_section(cfg, store, nodes, gpu, beliefs, set())["rows"] if r["component"] == "web")
    assert row["effect"] > 0.1 and row["without"]["scored"] == 2


def test_the_history_tool_hides_postmortems_from_an_ablated_session(ctx, store):
    from loop.agents import history as HX
    store.ensure_root("r0")
    h = HX.Historian(ctx, store, (), request_id="r", ablated=("postmortem",))
    assert "not available" in h._postmortem("anything")
    assert "not available" not in HX.Historian(ctx, store, (), request_id="r")._postmortem("anything")
    assert any("history tool" in x for x in CG.v3_text({"rows": []}))
