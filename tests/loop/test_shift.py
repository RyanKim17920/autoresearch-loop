"""loop.shift: an incumbent change's KNOB SHIFT is booked, propagated onto the not-started nodes / queued campaigns on
the previous incumbent, and the closed nulls it may have decided are backfilled onto the new incumbent.
Run: uv run pytest tests/loop -q --confcutdir=tests/loop"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest

from loop import backfill as BF
from loop import campaign as CP
from loop import diagnostics as DG
from loop import shift as SH
from loop.agents.researcher import set_yaml_value
from loop.contracts import (ROOT_NODE_ID, Node, NodePosterior, NodeStatus, Operator, Posterior, Run, RunStatus, Stage,
                            Strategy)
from loop.spec import BackfillCfg, CampaignCfg

T = 1000.0
WD = "train.weight_decay"


@pytest.fixture
def sim(ctx, store):
    from loop.adapters.sim import SimAdapter
    return SimAdapter(ctx)


def _mk(ctx, store, sim, nid, parent, cfg, status, notes="", family="optimization", strategy=Strategy.EXPLORE,
        operator=Operator.DRAFT):
    """A config-only node on ``parent``'s commit (a real sim commit, its config diff stored)."""
    root = store.get_node(ROOT_NODE_ID) or store.ensure_root(sim.root_commit())
    base = store.get_node(parent) or root
    wt = sim.prepare_worktree(parent_commit=base.commit, name=nid)
    text = (Path(wt.path) / "config.yaml").read_text()
    for k, v in cfg.items():
        text = set_yaml_value(text, k, v)
    (Path(wt.path) / "config.yaml").write_text(text)
    _, _, cfg_diff = sim.diff(wt)
    commit = sim.commit(wt, message=nid)
    sim.reap_worktree(wt)
    store.put_node(Node(id=nid, task_id=ctx.task_id, commit=commit, parent_id=base.id, strategy=strategy,
                        operator=operator, family=family, mechanism_tag=f"m{nid}", diff_hash=f"h{nid}", status=status,
                        notes=notes, created_ts=1.0, updated_ts=1.0))
    store.set_state(f"config_diff:{nid}", dict(cfg_diff))
    return store.get_node(nid)


def _post(nid, p_gain, plus):
    q = Posterior(0.001, 0.002, 2)
    return NodePosterior(nid, q, q, {}, 1.0, 0.5, False, gain_p=p_gain, gain_prior=0.2, gain_plus=plus)


def _ctx(ctx, **bf):
    cfg = dataclasses.replace(
        ctx.cfg, backfill=BackfillCfg(**{"enabled": True, "per_tick": 1, "max_active": 4, **bf}),
        campaign=CampaignCfg(input_families=("augmentation",), input_space=({"key": WD, "high": 0.2, "scale": "log"},)))
    return dataclasses.replace(ctx, cfg=cfg)


def _incumbents(ctx, store, sim):
    _mk(ctx, store, sim, "old", ROOT_NODE_ID, {"train.lr": 0.002}, NodeStatus.CONFIRMED)
    _mk(ctx, store, sim, "new", "old", {WD: 0.1}, NodeStatus.SCREENING)     # no run yet: still never re-based


def test_shift_is_detected_once_per_incumbent_change_and_propagated(ctx, store, sim):
    ctx = _ctx(ctx)
    _incumbents(ctx, store, sim)
    q1 = _mk(ctx, store, sim, "q1", "old", {"loss.temperature": 0.2}, NodeStatus.QUEUED)       # re-based
    _mk(ctx, store, sim, "q2", "old", {WD: 0.1}, NodeStatus.QUEUED)                    # the new one holds it
    _mk(ctx, store, sim, "q3", "old", {"loss.temperature": 0.3}, NodeStatus.SCREENING)  # started: left alone
    store.put_run(Run(id=Run.make_id("q3", Stage.SCREEN, 1), task_id=ctx.task_id, node_id="q3", seed=1,
                      stage=Stage.SCREEN, status=RunStatus.RUNNING))
    _mk(ctx, store, sim, "dp", "old", {"loss.temperature": 0.4}, NodeStatus.SCREENING,     # a batch point
        strategy=Strategy.DESIGN, operator=None)
    _mk(ctx, store, sim, "rc", ROOT_NODE_ID, {"loss.temperature": 0.2}, NodeStatus.QUEUED)  # root lane: untouched
    _mk(ctx, store, sim, "sw", "old", {"loss.temperature": 0.5}, NodeStatus.SCREENING, notes="swap_of=q1 swap_off=x")
    _mk(ctx, store, sim, "cb", "old", {"aug.strength": 0.7}, NodeStatus.SCREENED_OUT)        # queued campaign base
    store.set_state("ideation:cb", {"campaign": True, "search_space": [
        {"key": WD, "low": 0.02, "high": 0.08, "scale": "log"}, {"key": "train.lr", "low": 0.001, "high": 0.004}]})
    CP.enqueue(store, "cb", "declared campaign: x")
    ids = iter(f"p{i}" for i in range(9))
    tick = lambda inc, now: SH.tick(ctx, store, sim, SimpleNamespace(incumbent_id=inc), new_id=lambda: next(ids),  # noqa: E731
                                    now=now)
    assert tick("old", T) == [] and store.get_state(SH.K_LAST) == "old"           # the first sight: nothing to compare
    acts = tick("new", T + 1)
    (ev,) = SH.events(store)
    assert ev["n"] == SH.BASE + 1 and ev["keys"] == {WD: [0.05, 0.1]} and ev["rel"] == {WD: 1.0} \
        and ev["root"] == {WD: 0.05} and ev["frm"] == "old" and ev["to"] == "new"
    assert acts[0] == "knob shift old -> new: weight_decay 0.05->0.1 (backfill epoch 101)"
    assert store.events("knob_shift")[0]["keys"] == {WD: [0.05, 0.1]}
    q1 = store.get_node("q1")
    assert q1.parent_id == "new" and "rebased_from=old" in q1.notes and q1.status is NodeStatus.QUEUED
    assert sim.config_at(q1.commit)[WD] == 0.1 and sim.config_at(q1.commit)["loss.temperature"] == 0.2
    q2 = store.get_node("q2")
    assert q2.status is NodeStatus.REJECTED and q2.reject_reason == "already_in_incumbent"
    assert {store.get_node(n).parent_id for n in ("q3", "dp", "sw", "new")} == {"old"} and store.get_node("rc").parent_id == ROOT_NODE_ID
    (q,) = CP.book(store)["queue"]                                         # the queued campaign follows its port
    port = store.get_node(q["base"])
    assert port.parent_id == "new" and "backfill_of=cb" in port.notes and sim.config_at(port.commit)[WD] == 0.1
    rows = store.get_state(f"ideation:{port.id}")["search_space"]
    assert rows[0] == {"key": WD, "low": 0.04, "high": 0.1, "scale": "log"}     # x2, inside the hull with the new 0.1
    assert rows[1] == {"key": "train.lr", "low": 0.001, "high": 0.004}          # not shifted
    assert tick("new", T + 2) == []                                              # once per change


def test_recentre_moves_in_scale_and_never_leaves_the_hull():
    row = {"key": "ld", "low": 0.8, "high": 1.0, "scale": "linear"}
    assert SH.recentre(row, 0.9, 1.0) == {**row, "low": 0.9, "high": 1.0}        # a bound at 1.0 is never crossed
    assert SH.recentre(row, 0.9, 0.85) == {**row, "low": 0.8, "high": 0.95}
    assert SH.recentre({"key": "lr", "low": 1e-4, "high": 4e-4}, 2e-4, 4e-4) == {"key": "lr", "low": 2e-4, "high": 4e-4}
    assert SH.recentre({"key": "lr", "low": 1e-4, "high": 4e-4}, 2e-4, 8e-4) == {"key": "lr", "low": 4e-4, "high": 8e-4}
    assert SH.recentre({"key": "x", "factor": 2.0}, 1, 2) == {"key": "x", "factor": 2.0}     # no range: unchanged
    assert SH.diff({"a": 1.0, "b": "x", "c": 0}, {"a": 1.5, "b": "y", "c": 1}) == (
        {"a": [1.0, 1.5], "b": ["x", "y"], "c": [0, 1]}, {"a": 0.5, "b": None, "c": None})


def test_affected_closed_nulls_are_ported_onto_the_new_incumbent_capped_and_diverse(ctx, store, sim, monkeypatch):
    ctx = _ctx(ctx, shift_max=2, shift_flags={"update_starved": ("weight_decay",)})
    _incumbents(ctx, store, sim)
    mk = lambda nid, parent="old", cfg=None, **kw: _mk(ctx, store, sim, nid, parent, cfg or {"aug.strength": 0.6},  # noqa: E731
                                                     NodeStatus.SCREENED_OUT, **kw)
    mk("a_flag", notes="mtype=optimization")                       # its own runs fired a flag the shift relieves
    mk("a_ut", notes="mtype=optimization")                         # close: under_tuned naming the shifted knob
    mk("a_in", ROOT_NODE_ID, family="augmentation", notes="mtype=augmentation")   # input-side at root's (= old) wd
    mk("a_in2", ROOT_NODE_ID, {WD: 0.2}, family="augmentation", notes="mtype=augmentation")   # ran at another wd
    mk("harm", notes="mtype=optimization")                         # a clear harm is not under-tuned
    mk("late", notes="mtype=optimization")                         # closed after the shift: judged with it
    store.update_node("late", updated_ts=T + 50)
    monkeypatch.setattr(DG, "under_tuned", lambda n, st, sn=None: "close: under_tuned (knob weight_decay) -- x"
                        if n.id in ("a_ut", "harm", "late") else "")
    monkeypatch.setattr(store, "diagnostics", lambda node_id=None: [1] if node_id == "a_flag" else [])
    monkeypatch.setattr(DG, "bottlenecks", lambda st, nid: [{"flag": "update_starved"}] if nid == "a_flag" else [])
    store.set_state(BF.K_SEEN, {str(e.n): 0.0 for e in BF.EPOCHS})          # no static epoch re-opens anything here
    posts = {"a_flag": _post("a_flag", 0.4, 0.003), "a_ut": _post("a_ut", 0.4, 0.002), "a_in": _post("a_in", 0.3, 0.001),
             "a_in2": _post("a_in2", 0.3, 0.004), "harm": _post("harm", 0.01, 0.0), "late": _post("late", 0.4, 0.009)}
    snap = SimpleNamespace(nodes=posts, eb_raw={}, incumbent_id="old")
    ids = iter(f"p{i}" for i in range(9))
    SH.tick(ctx, store, sim, snap, new_id=lambda: next(ids), now=T)
    assert BF.shift_items(ctx, store, snap, {}, set()) == []                      # no shift onto the incumbent yet
    snap.incumbent_id = "new"
    SH.tick(ctx, store, sim, snap, new_id=lambda: next(ids), now=T + 10)
    why = {nid: SH.affected(ctx, store, snap, SH.current(store, snap), store.get_node(nid)) for nid in posts}
    assert why == {"a_flag": "update_starved on its runs (relieved by weight_decay)",
                   "a_ut": "close: under_tuned (knob weight_decay)",
                   "a_in": "input-side (mechanism type augmentation) closed null at weight_decay 0.05", "a_in2": "", "harm": "",
                   "late": "close: under_tuned (knob weight_decay)"}
    q = BF.shift_items(ctx, store, snap, {}, set())        # best of each mechanism type first, then the next; cap 2
    assert [(i.node_id, i.action, i.target, i.epoch) for i in q] == [("a_flag", "port", "new", 101),
                                                                     ("a_in", "port", "new", 101)]
    assert [i.node_id for i in BF.shift_items(ctx, store, snap, {}, {"a_flag"})] == ["a_ut", "a_in"]   # re-done: out
    acts = BF.tick(ctx, store, sim, snap, new_id=lambda: "pa", now=T + 11, reopen=lambda *a: None)
    assert acts == ["backfill: a_flag retest after knob shift (weight_decay 0.05->0.1): update_starved on its runs "
                    "(relieved by weight_decay) -- port onto new E[gain+]=0.0030 (epoch 101)"]
    pa = store.get_node("pa")
    assert pa.parent_id == "new" and "backfill_of=a_flag" in pa.notes and sim.config_at(pa.commit)[WD] == 0.1 \
        and sim.config_at(pa.commit)["aug.strength"] == 0.6
    book = store.get_state(BF.K_BOOK)
    assert book["a_flag"]["epoch"] == 101
    assert [i.node_id for i in BF.shift_items(ctx, store, snap, book, set())] == ["a_ut"]   # booked: 1 left of the cap
