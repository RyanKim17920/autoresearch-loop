"""FOLLOW THE GRADIENT (loop/extrapolate.py): knob metadata and range edges, the winning-direction ledger (incumbent
lineage, factorial main effects, single config moves), the extrapolation lane on the incumbent (x1 / x2 of the last
winning step, a substitute partner backed off, dedupe, the in-flight cap, the stop rule, continuing a win) and the
computed facts. Run: uv run pytest tests/loop -q --confcutdir=tests/loop"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from loop import extrapolate as EX
from loop.adapters import flatten
from loop.adapters.sim import BASE_CONFIG
from loop.contracts import PANEL, NodePosterior, NodeStatus, Posterior, design_point
from loop.select.intake import K_BASE
from loop.spec import ExtrapolateCfg
from tests.loop.test_tune import _moonshot, _screened

KNOBS = ({"key": "*weight_decay", "scale": "log"}, {"key": "*.lr", "scale": "log", "high": 0.004})


@pytest.fixture
def sim(ctx, store):
    from loop.adapters.sim import SimAdapter
    return SimAdapter(ctx)


@pytest.fixture
def ex(ctx):
    return dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, extrapolate=ExtrapolateCfg(
        enabled=True, max_inflight=4, min_z=1.0, knobs=KNOBS)))


def _p(nid: str, mean: float, sd: float = 0.001, *, vetoed: bool = False) -> NodePosterior:
    return NodePosterior(node_id=nid, increment=Posterior(mean, sd, 2), utility=Posterior(0.01 + mean, sd, 2), axes={},
                         p_positive=0.99, p_beats_incumbent=0.5, vetoed=vetoed)


def _snap(inc: str, posts: dict) -> SimpleNamespace:
    return SimpleNamespace(nodes=posts, sigma={PANEL: 0.004}, incumbent_id=inc)


def _incumbent(ex, store, sim):
    """root -> n0001 (weight decay 0.05 -> 0.1, confirmed): the lineage's winning step, x2 in log."""
    store.set_state(K_BASE, flatten(BASE_CONFIG))
    return _moonshot(ex, store, sim, "n0001", status=NodeStatus.CONFIRMED, cfg={"train.weight_decay": 0.1}, code=False)


def test_meta_scales_bounds_and_the_range_edge(ex):
    assert EX.meta(ex, "train.weight_decay", 0.05)["scale"] == "log"                     # a knobs row
    assert EX.meta(ex, "train.lr", 0.001) == {"scale": "log", "low": None, "high": 0.004}
    # an unknown key in [0, 1]: a probability / fraction leaf stays in [0, 1]; anything else (a loss weight, a
    # strength) is linear from 0 capped at max(1, 4 x the largest value seen) -- live: jepa_xscale_weight "at its bound (1)"
    assert EX.meta(ex, "aug.prob", 0.5) == {"scale": "linear", "low": 0.0, "high": 1.0}
    assert EX.meta(ex, "data.global_crop_scale[0]", 0.4)["high"] == 1.0 and EX.meta(ex, "x.drop_rate", 0.1)["high"] == 1.0
    assert EX.meta(ex, "aug.strength", 0.5) == {"scale": "linear", "low": 0.0, "high": 2.0}
    assert EX.meta(ex, "dino.jepa_xscale_weight", 0.5, [1.0])["high"] == 4.0 and EX.meta(ex, "w", 0.0)["high"] == 1.0
    assert EX.meta(ex, "model.depth", 12) == {"scale": "log", "low": 1, "high": None}
    assert EX.meta(ex, "x.offset", -1.0) is None and EX.meta(ex, "x.flag", True) is None
    assert EX.beyond(ex, "aug.prob", 0.7, [0.5, 0.6]) == 0.8                              # one step past the edge
    assert EX.beyond(ex, "aug.prob", 0.9, [0.5, 0.7]) == 1.0                              # clipped at the bound
    assert EX.beyond(ex, "aug.prob", 1.0, [0.8]) is None and EX.beyond(ex, "aug.prob", 0.6, [0.5, 0.7]) is None
    assert EX.beyond(ex, "aug.strength", 1.0, [0.8]) == 1.2                               # a weight goes past 1
    assert EX.beyond(ex, "train.lr", 0.002, [0.001]) == 0.004 and EX.beyond(ex, "train.lr", 0.004, [0.002]) is None
    assert EX.inside(ex, "aug.prob", 0.3) and not EX.inside(ex, "aug.prob", 1.2, 0.5)     # bounds of the value
    assert EX.inside(ex, "aug.strength", 1.2, 0.5) and not EX.inside(ex, "aug.strength", 2.5, 0.5)


def _launched(ex, store, sim):
    """The incumbent n0001 and its first extrapolation round (n0100 = 0.2, n0101 = 0.4), asserted."""
    inc = _incumbent(ex, store, sim)
    snap = _snap(inc.id, {inc.id: _p(inc.id, 0.003)})
    led = EX.ledger(ex, store, snap)
    e = led["train.weight_decay"]
    assert e["dir"] == 1 and e["z"] == pytest.approx(3.0) and e["step"] == pytest.approx(0.6931, rel=1e-3)
    assert e["last"]["src"] == "lineage n0001 confirmed" and e["value"] == 0.1 and not e["beyond"]
    moves, stopped = EX.plan(ex, store, snap)
    assert [m["levels"] for m in moves] == [{"train.weight_decay": 0.2}, {"train.weight_decay": 0.4}] and not stopped
    ids = iter(f"n{i:04d}" for i in range(100, 200))
    acts = EX.tick(ex, store, sim, snap, new_id=lambda: next(ids), now=1e12)
    assert acts[0].startswith("extrapolate: train.weight_decay 0.1->0.2 (direction from lineage n0001 confirmed")
    for nid, v in (("n0100", 0.2), ("n0101", 0.4)):
        n = store.get_node(nid)
        assert n.status is NodeStatus.SCREENING and n.parent_id == inc.id and not design_point(n)
        assert store.get_state(f"config_diff:{nid}") == {"train.weight_decay": v} and n.family == "regularization"
        assert sim._config(n.commit)["train"]["weight_decay"] == v and "extrapolate=train.weight_decay" in n.notes
    assert [e["levels"] for e in store.events("extrapolate")] == [{"train.weight_decay": 0.2}, {"train.weight_decay": 0.4}]
    assert EX.tick(ex, store, sim, snap, new_id=lambda: next(ids), now=1e12 + 1) == []        # in flight: no dup
    assert "0.2 open (n0100)" in EX.facts(ex, store, snap)
    return inc, snap, ids


def test_the_ledger_reads_the_lineage_and_the_lane_continues_it_on_the_incumbent(ex, store, sim):
    _launched(ex, store, sim)


def _close(ex, store, nid, gain, snap, **kw):
    _screened(ex, store, nid)
    store.update_node(nid, status=NodeStatus.SCREENED_OUT)
    snap.nodes[nid] = _p(nid, gain, **kw)


def test_two_nodes_at_one_value_one_open_do_not_break_the_ledger(ex, store, sim, monkeypatch):
    inc = _incumbent(ex, store, sim)
    snap = _snap(inc.id, {inc.id: _p(inc.id, 0.003)})
    k = "train.weight_decay"
    monkeypatch.setattr(EX, "tested", lambda *_: {frozenset([k]): [
        {"levels": {k: 0.2}, "gain": None, "nid": "n0102", "vetoed": False},     # live 10-08: TypeError None < float
        {"levels": {k: 0.2}, "gain": 0.001, "nid": "n0101", "vetoed": False}]})
    assert [b[2] for b in EX.ledger(ex, store, snap)[k]["beyond"]] == ["n0101", "n0102"]


def test_a_bound_stop_is_logged_once_and_lifted_when_the_bound_moves(ex, store, sim):
    capped = dataclasses.replace(ex, cfg=dataclasses.replace(ex.cfg, extrapolate=dataclasses.replace(
        ex.cfg.extrapolate, knobs=({"key": "*weight_decay", "scale": "log", "high": 0.2}, *KNOBS[1:]))))
    inc = _incumbent(capped, store, sim)
    snap = _snap(inc.id, {inc.id: _p(inc.id, 0.003)})
    ids = iter(f"n{i:04d}" for i in range(100, 200))
    acts = EX.tick(capped, store, sim, snap, new_id=lambda: next(ids), now=1e12)
    assert len(acts) == 1 and "0.1->0.2" in acts[0]                   # x2 clipped onto x1: one node
    _close(capped, store, "n0100", 0.002, snap)
    acts = EX.tick(capped, store, sim, snap, new_id=lambda: next(ids), now=2e12)
    assert acts == ["extrapolate: train.weight_decay stopped on n0001 (at its bound (0.2))"]
    assert EX.tick(capped, store, sim, snap, new_id=lambda: next(ids), now=3e12) == []      # not re-logged
    acts = EX.tick(ex, store, sim, snap, new_id=lambda: next(ids), now=4e12)                # the bound lifted
    assert acts and "0.1->0.4" in acts[0] and "train.weight_decay@n0001" not in EX.book(store)["stopped"]


def test_two_losing_steps_stop_a_direction_and_a_winning_step_continues_further(ex, store, sim):
    inc, snap, ids = _launched(ex, store, sim)
    _close(ex, store, "n0100", 0.002, snap)                    # x1 won, x2 still open: the round fills one step further
    moves, _ = EX.plan(ex, store, snap)
    assert [m["levels"] for m in moves] == [{"train.weight_decay": 0.8}]
    _close(ex, store, "n0101", -0.001, snap)                   # x2 lost once: one more step past it
    moves, stopped = EX.plan(ex, store, snap)
    assert [m["levels"] for m in moves] == [{"train.weight_decay": 0.8}] and not stopped
    acts = EX.tick(ex, store, sim, snap, new_id=lambda: next(ids), now=1e12 + 2)
    assert "0.1->0.8" in acts[0] and store.get_node("n0102").parent_id == inc.id
    _close(ex, store, "n0102", -0.0005, snap)                  # two furthest steps <= 0: stopped for this incumbent
    acts = EX.tick(ex, store, sim, snap, new_id=lambda: next(ids), now=1e12 + 3)
    assert acts and "train.weight_decay stopped on n0001 (two furthest steps shrunk <= 0" in acts[0], acts
    assert EX.plan(ex, store, snap)[0] == [] and "stopped: two furthest" in EX.facts(ex, store, snap)


def test_a_vetoed_step_stops_the_direction(ex, store, sim):
    inc, snap, ids = _launched(ex, store, sim)
    _close(ex, store, "n0100", 0.004, snap, vetoed=True)
    _, stopped = EX.plan(ex, store, snap)
    assert stopped == {"train.weight_decay": "harm veto at 0.2"}


def test_factorial_effects_a_substitute_partner_the_cap_and_dedupe(ex, store, sim):
    from loop import design as D
    inc = _incumbent(ex, store, sim)
    store.set_state(D.K_DESIGN, {"batches": [{"batch": 1, "n": 16, "df": 8, "interactions": [], "points": {}, "knobs": [
        {"key": "train.lr", "low": 0.0005, "high": 0.001}, {"key": "train.weight_decay", "low": 0.05, "high": 0.1},
        {"key": "aug.strength", "low": 0.5, "high": 0.6}, {"key": "loss.temperature", "low": 0.1, "high": 0.2}],
        "effects": [{"effect": e, "se": 0.0005, "estimable": True, "significant": False}
                    for e in (0.004, 0.003, -0.0002, 0.0001)],
        "chains": [{"pairs": [["train.lr", "train.weight_decay"]], "c": -0.002, "se": 0.0005},
                   {"pairs": [["aug.strength", "loss.temperature"], ["train.lr", "x"]], "c": -0.003, "se": 0.0005}]}]})
    assert EX.partners(store, 1.5) == {"train.lr": [("train.weight_decay", -4.0)],
                                       "train.weight_decay": [("train.lr", -4.0)]}   # an aliased chain never counts
    snap = _snap(inc.id, {inc.id: _p(inc.id, 0.003)})
    led = EX.ledger(ex, store, snap)
    assert led["train.lr"]["dir"] == 1 and led["train.lr"]["last"]["src"] == "design d1"
    moves, _ = EX.plan(ex, store, snap, led=led)
    lv = [m["levels"] for m in moves]
    assert lv[:2] == [{"train.weight_decay": 0.2}, {"train.lr": 0.002}]           # every key's first step first
    assert {"train.weight_decay": 0.2, "train.lr": 0.0005} in lv                 # the partner backed off one step
    assert {"train.lr": 0.004} in lv and {"train.lr": 0.002, "train.weight_decay": 0.05} in lv   # x2 at the bound
    one = dataclasses.replace(ex, cfg=dataclasses.replace(ex.cfg, extrapolate=dataclasses.replace(
        ex.cfg.extrapolate, max_inflight=3)))
    ids = iter(f"n{i:04d}" for i in range(100, 200))
    assert len(EX.tick(one, store, sim, snap, new_id=lambda: next(ids), now=1e12)) == 3           # the cap
    assert EX.tick(one, store, sim, snap, new_id=lambda: next(ids), now=1e12 + 1) == []
    planned = [m["levels"] for m in EX.plan(ex, store, snap)[0]]
    assert {"train.weight_decay": 0.2} not in planned and {"train.lr": 0.002} not in planned   # tested: never rebuilt


def test_a_single_config_move_of_any_lane_is_evidence_and_off_means_off(ctx, ex, store, sim):
    inc = _incumbent(ex, store, sim)
    other = _moonshot(ex, store, sim, "n0002", status=NodeStatus.SCREENED_OUT, cfg={"train.lr": 0.0005}, code=False)
    snap = _snap(inc.id, {inc.id: _p(inc.id, 0.0), other.id: _p(other.id, -0.003)})
    e = EX.ledger(ex, store, snap)["train.lr"]                   # lowering LR hurt: the + direction, no winning step
    assert e["dir"] == 1 and e["last"] is None and e["n"] == 1 \
        and [i["src"] for i in EX.evidence(ex, store, snap) if i["key"] == "train.lr"] == ["n0002 on root"]
    assert not [m for m in EX.plan(ex, store, snap)[0] if m["key"] == "train.lr"]
    assert EX.tick(ctx, store, sim, snap, new_id=lambda: "n0100") == []                       # disabled by default
    from loop.agents.briefs import directions
    assert directions(ctx, store, snap) == "" and "train.lr" not in directions(ex, store, snap)
