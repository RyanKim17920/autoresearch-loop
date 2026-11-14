"""Waste reduction driven by the predictive model: the screen / lane / revisit value gates, model-driven carrier
allocation and control sharing."""

from __future__ import annotations

import dataclasses
import random
from types import SimpleNamespace

from loop.contracts import ROOT_NODE_ID, Node, NodeStatus, Operator, Run, RunStatus, Stage, Strategy
from loop.driver import Driver
from loop.seeds import RunNeed, screen_block
from loop.select import allocate_carrier, revisit_ok
from loop.spec import SeedPlan
from loop.world import predictive as P


def _cfg(value_p=0.003, explore=0.15, wait=3.0):
    return SimpleNamespace(prior=SimpleNamespace(wm_value_p=value_p, wm_explore=explore),
                           seeds=SeedPlan(control_wait_h=wait))


def _node(nid, parent=ROOT_NODE_ID, status=NodeStatus.QUEUED, notes="", strategy=Strategy.EXPLORE, ts=1.0):
    return Node(id=nid, task_id="t", commit="c", parent_id=parent, strategy=strategy, operator=Operator.DRAFT,
                family="f", mechanism_tag="x", diff_hash=nid, status=status, notes=notes, created_ts=ts)


class StubFit:
    mde = 0.003

    def __init__(self, values):
        self.values = values

    def value(self, node, parent=None, strategy=None):
        return self.values.get((node.id, parent), self.values.get(node.id, 0.0))

    def lane_value(self, members, parent=None, recent=20):
        ms = list(members)
        return (sum(self.value(n) for n in ms) / len(ms), len(ms)) if ms else (1.0, 0)

    def allocation(self, carriers, recent=40):
        return [{"carrier": c, "mtype": "m", "mean": 0.0, "sd": 0.001, "p_mde": self.values.get(c, 0.0), "n": 1}
                for c in carriers]


def test_gate_value_threshold_and_a_deterministic_exploration_slot():
    cfg = _cfg()
    fit = StubFit({})
    assert P.gate_value(None, cfg, "k", 0.0)[0]                        # no model: everything passes
    assert P.gate_value(fit, cfg, "k", 0.01)[0]
    keys = [f"n{i}" for i in range(2000)]
    passed = [k for k in keys if P.gate_value(fit, cfg, k, 0.0)[0]]
    assert 0.12 < len(passed) / len(keys) < 0.18                       # ~15% exploration
    assert passed == [k for k in keys if P.gate_value(fit, cfg, k, 0.0)[0]]   # never flips between ticks
    assert P.gate_value(fit, _cfg(value_p=0.0), "k", 0.0)[0]           # 0 = off


def test_the_screen_gate_rejects_a_low_value_queued_node():
    low = next(f"n{i}" for i in range(100) if not P.explore(f"n{i}", 0.15))
    logs, sets = [], []
    me = SimpleNamespace(snapshot=SimpleNamespace(wm=StubFit({low: 0.0001, "hi": 0.05})), cfg=_cfg(),
                         store=SimpleNamespace(log=lambda ev, **kw: logs.append((ev, kw))),
                         _set_status=lambda n, st, why, acts, **kw: sets.append((n.id, st, kw)))
    me._gate_exempt = lambda wm, node: Driver._gate_exempt(me, wm, node)
    assert not Driver._value_gate(me, _node(low), [])
    assert sets == [(low, NodeStatus.REJECTED, {"reject_reason": "low_value"})]
    assert Driver._value_gate(me, _node("hi"), [])
    assert Driver._value_gate(me, _node(low, notes="legs_of=n1 legs=L1"), [])          # side lanes pass
    assert logs[0][0] == "wm_gate" and logs[0][1]["passed"] is False
    me.snapshot = SimpleNamespace(wm=None)
    assert Driver._value_gate(me, _node(low), [])


def test_the_design_lane_starts_only_when_the_model_expects_value():
    state = {}
    store = SimpleNamespace(get_state=lambda k, d=None: state.get(k, d), set_state=state.__setitem__,
                            log=lambda *a, **k: None)
    pts = [_node(f"d{i}", strategy=Strategy.DESIGN) for i in range(5)]
    key = next(str(i) for i in range(100) if not P.explore(f"design:{i}", 0.15))
    me = SimpleNamespace(snapshot=SimpleNamespace(wm=StubFit({p.id: 1e-6 for p in pts}), incumbent_id="n9"),
                         cfg=_cfg(), store=store)
    assert not Driver._lane_gate(me, "design", pts, key)
    assert state["wm.lane_gate"] == {"design": False}
    me.snapshot.wm = StubFit({p.id: 0.02 for p in pts})
    assert Driver._lane_gate(me, "design", pts, key)
    assert Driver._lane_gate(me, "design", [], key)                       # no history: no verdict


def test_revisit_gate():
    n = _node("n5", status=NodeStatus.SCREENED_OUT)
    nid = next(f"n{i}" for i in range(100) if not P.explore(f"revisit:n{i}", 0.15))
    n = _node(nid, status=NodeStatus.SCREENED_OUT)
    assert not revisit_ok(SimpleNamespace(wm=StubFit({(nid, "c1"): 0.0})), _cfg(), n, "c1")
    assert revisit_ok(SimpleNamespace(wm=StubFit({(nid, "c1"): 0.01})), _cfg(), n, "c1")
    assert revisit_ok(SimpleNamespace(wm=None), _cfg(), n, "c1")


def test_allocation_draws_carriers_in_proportion_to_the_model():
    conf = [_node("c1", status=NodeStatus.CONFIRMED), _node("c2", status=NodeStatus.CONFIRMED)]
    store = SimpleNamespace(nodes=lambda status=None: conf, get_state=lambda k, d=None: d, runs=lambda **kw: [])
    snap = SimpleNamespace(wm=StubFit({ROOT_NODE_ID: 0.05, "c1": 0.0005, "c2": 0.0}), incumbent_id="c1", nodes={})
    rng = random.Random(0)
    picks = [allocate_carrier(store, snap, ["c1", "c2"], rng, _cfg(explore=0.0))[0] for _ in range(400)]
    assert picks.count(ROOT_NODE_ID) > 350 and "c2" not in picks
    pick, why = allocate_carrier(store, snap, ["c1", "c2"], rng, _cfg())
    assert "allocation" in why and "Mechanism types" in why
    assert allocate_carrier(store, SimpleNamespace(wm=None), ["c1"], rng, _cfg()) is None


def test_a_screen_control_waits_for_a_second_child_or_the_wait():
    cfg = _cfg()
    blk = screen_block(cfg.seeds, 0)
    a = _node("a", parent="p", status=NodeStatus.SCREENING)
    need = RunNeed("p", Stage.PARENT, blk[0], "a")
    own = [Run(id="a:screen:11", task_id="t", node_id="a", seed=blk[0], stage=Stage.SCREEN,
               status=RunStatus.RUNNING, submitted_ts=1000.0)]
    me = SimpleNamespace(cfg=cfg, clock=lambda: 1000.0 + 3600.0)
    blocks = {"a": 0, "b": 0}
    assert Driver._hold_control(me, need, a, blocks, {("p", 0): 1}, {"a": own})          # alone, 1 h in: hold
    assert not Driver._hold_control(me, need, a, blocks, {("p", 0): 2}, {"a": own})      # a sibling shares it
    me.clock = lambda: 1000.0 + 4 * 3600.0
    assert not Driver._hold_control(me, need, a, blocks, {("p", 0): 1}, {"a": own})      # waited long enough
    assert not Driver._hold_control(me, RunNeed("a", Stage.SCREEN, blk[0], "a"), a, blocks, {}, {"a": own})
    me.clock = lambda: 1000.0 + 3600.0
    for note in ("legs_of=p legs=L4", "tune_of=q tune=t1"):          # a graduate's own block: nobody will share it
        g = dataclasses.replace(a, notes=note)
        assert not Driver._hold_control(me, need, g, blocks, {("p", 0): 1}, {"a": own})
    me.cfg = _cfg(wait=0.0)
    assert not Driver._hold_control(me, need, a, blocks, {("p", 0): 1}, {"a": own})      # 0 = off


def test_the_incumbent_floor_takes_its_share_and_the_allocation_the_rest():
    conf = [_node("c1", status=NodeStatus.CONFIRMED), _node("c2", status=NodeStatus.CONFIRMED)]
    store = SimpleNamespace(nodes=lambda status=None: conf, get_state=lambda k, d=None: d, runs=lambda **kw: [])
    snap = SimpleNamespace(wm=StubFit({ROOT_NODE_ID: 0.05, "c1": 0.0005, "c2": 0.0}), incumbent_id="c1", nodes={})
    cfg = _cfg(explore=0.0)
    cfg.prior.wm_incumbent_share = 0.5
    rng = random.Random(0)
    picks = [allocate_carrier(store, snap, ["c1", "c2"], rng, cfg)[0] for _ in range(1000)]
    assert 0.45 < picks.count("c1") / 1000 < 0.56 and picks.count(ROOT_NODE_ID) > 400
    pick, why = next((p, w) for p, w in (allocate_carrier(store, snap, ["c1"], rng, cfg) for _ in range(50))
                     if p == "c1")
    assert "incumbent floor" in why


def test_ideation_reads_the_model_allocation_as_shares_and_the_optimiser_flags():
    from loop import diagnostics as DG
    from loop.agents import dossiers as DS
    nodes = {n.id: n for n in (_node("a", ts=1.0), _node("b", ts=2.0))}
    nodes["b"] = Node(**{**nodes["b"].__dict__, "family": "g"})
    fit = StubFit({"a": 0.3, "b": 0.1, ROOT_NODE_ID: 0.2, "c1": 0.2})
    fit.book = SimpleNamespace(nodes=nodes)
    snap = SimpleNamespace(wm=fit, incumbent_id="c1")
    got = DG.allocation(None, snap)
    assert {k: round(v, 9) for k, v in got.items()} == {"m": 1.0, "f": 0.75, "g": 0.25}
    assert DS.allocation(None, snap) == got and DG.allocation(None, SimpleNamespace(wm=None)) == {}
    starved = {"lr_end_frac": 0.9}                                     # the schedule never annealed
    rows = {"p": [{"features": starved}], "c1": []}
    store = SimpleNamespace(diagnostics=lambda node_id=None: rows.get(node_id, []), get_node=lambda i: None)
    assert DG.optimiser_flags(store, ["x", "p"]) == ["schedule_not_annealed"]
    assert DS.optimizer_flagged(store, snap, "p") and not DS.optimizer_flagged(store, snap, "x")


def test_the_screen_gate_exempts_moonshots_init_swaps_and_thin_support_and_readmits_them_once():
    from loop.driver import K_READMIT, READMIT_SINCE
    low = [f"n{i}" for i in range(200) if not P.explore(f"n{i}", 0.15)]
    fit = StubFit({nid: 0.0 for nid in low})
    fit.support = lambda node: 3 if node.family == "new" else 50
    cfg = SimpleNamespace(prior=SimpleNamespace(wm_value_p=0.003, wm_explore=0.15, wm_gate_min_support=10))
    logs, sets = [], []
    me = SimpleNamespace(snapshot=SimpleNamespace(wm=fit), cfg=cfg,
                         store=SimpleNamespace(log=lambda ev, **kw: logs.append((ev, kw))),
                         _set_status=lambda n, st, why, acts, **kw: sets.append((n.id, st, kw)))
    me._gate_exempt = lambda wm, node: Driver._gate_exempt(me, wm, node)
    moon = _node(low[0], strategy=Strategy.MOONSHOT)                 # n1726-like: a moonshot / an init swap
    swap = _node(low[1], notes="brief=moonshot init=https://dl.fbaipublicfiles.com/dinov2/vitb14.pth")
    thin = dataclasses.replace(_node(low[2]), family="new")
    dup = _node(low[3])                                               # a well-measured family / type: still judged
    for n in (moon, swap, thin):
        assert Driver._value_gate(me, n, [])
    assert [kw["why"] for _, kw in logs] == ["exempt: moonshot", "exempt: init swap",
                                            "exempt: thin support (3 < 10 scored nodes share its family / "
                                            "mechanism type / init)"]
    assert not sets and not Driver._value_gate(me, dup, [])
    assert sets == [(dup.id, NodeStatus.REJECTED, {"reject_reason": "low_value"})]
    # the one-shot readmission: low_value rejects since the deploy that the exemption passes are QUEUED again
    sets.clear()
    rej = lambda n, ts: dataclasses.replace(n, status=NodeStatus.REJECTED, reject_reason="low_value", updated_ts=ts)  # noqa: E731
    state = {}
    me.store = SimpleNamespace(log=lambda ev, **kw: logs.append((ev, kw)), set_state=state.__setitem__,
                               nodes=lambda status=None: [rej(moon, READMIT_SINCE + 1), rej(dup, READMIT_SINCE + 1),
                                                          rej(swap, READMIT_SINCE + 9),
                                                          rej(_node("n9", strategy=Strategy.MOONSHOT), READMIT_SINCE - 60)])
    Driver._readmit_low_value(me, [])
    assert sets == [(moon.id, NodeStatus.QUEUED, {"reject_reason": ""}), (swap.id, NodeStatus.QUEUED, {"reject_reason": ""})]
    assert state[K_READMIT] == {"nodes": [moon.id, swap.id]} and logs[-1][0] == "readmit_low_value"
