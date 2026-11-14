"""loop.backfill: verdict epochs re-evaluate closed nodes and queue the cheapest corrective action."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from loop import backfill as BF
from loop.contracts import ROOT_NODE_ID, Node, NodePosterior, NodeStatus, Posterior, Run, RunStatus, Stage, Strategy
from loop.world.evalues import CONFIRM_KEY

from tests.loop.test_driver import _cfg

T0 = 1000.0
LIVE_E: dict[str, float] = {}        # node id -> its live confirm E (default: the stored confirm_e)


@pytest.fixture(autouse=True)
def _live_e(monkeypatch):
    LIVE_E.clear()
    monkeypatch.setattr(BF, "live_e", lambda ctx, store, snap: lambda n: LIVE_E.get(n.id, n.confirm_e or 1.0))


def _node(store, nid, status, parent=ROOT_NODE_ID, **kw):
    store.put_node(Node(id=nid, task_id=store.task_id, commit=f"c-{nid}", parent_id=parent, strategy=Strategy.EXPLORE,
                        operator=None, family="f", mechanism_tag=f"m{nid}", diff_hash=f"h{nid}", status=status,
                        created_ts=T0, updated_ts=T0, **kw))
    for seed in (1, 2):
        store.put_run(Run(id=Run.make_id(nid, Stage.SCREEN, seed), task_id=store.task_id, node_id=nid, seed=seed,
                          stage=Stage.SCREEN, status=RunStatus.DONE, scores={"a": 0.5}))


def _post(nid, p_gain, plus, p_pos=1.0):
    q = Posterior(0.001, 0.002, 2)
    return NodePosterior(nid, q, q, {}, p_pos, 0.5, False, gain_p=p_gain, gain_prior=0.2, gain_plus=plus)


def _world(store):
    store.ensure_root("c0")
    _node(store, "inc", NodeStatus.CONFIRMED)
    _node(store, "oth", NodeStatus.CONFIRMED)
    _node(store, "s_pass", NodeStatus.SCREENED_OUT, "inc")       # passes the current gate: promote
    _node(store, "s_ext", NodeStatus.SCREENED_OUT, "inc")        # early-confirmation eligible
    _node(store, "s_dead", NodeStatus.SCREENED_OUT, "inc")       # still a screen-out
    _node(store, "c_off", NodeStatus.NOT_CONFIRMED, "oth")       # off the incumbent's lineage: port
    _node(store, "c_on", NodeStatus.NOT_CONFIRMED, "inc", confirm_e=2.0)   # one more seed can reach 1/alpha
    _node(store, "r_leg", NodeStatus.SCREENED_OUT)               # a root child: port
    store.set_state(CONFIRM_KEY, {"c_on": [0.004, 0.003]})
    posts = {"s_pass": _post("s_pass", 0.7, 0.002), "s_ext": _post("s_ext", 0.3, 0.0004),
             "s_dead": _post("s_dead", 0.01, 0.0), "c_off": _post("c_off", 0.7, 0.001),
             "c_on": _post("c_on", 0.4, 0.0003), "r_leg": _post("r_leg", 0.9, 0.004)}
    raw = {k: ("carrier", 0.004 if k != "s_dead" else -0.001, 4e-6) for k in posts}
    return SimpleNamespace(nodes=posts, eb_raw=raw, incumbent_id="inc")


def _ctx(**backfill):
    from loop.spec import BackfillCfg, PromotionCfg, SeedPlan
    cfg = _cfg(seeds=SeedPlan(promote_stage=False, max_seeds_per_node=5),
               promotion=PromotionCfg(p_promote=0.6, early_confirm=True, voi_gain=0.0007, alpha_confirm=0.2),
               backfill=BackfillCfg(**backfill))
    return SimpleNamespace(cfg=cfg, task_id="t")


def test_plan_routes_each_invalidated_close_to_its_cheapest_action(store):
    snap = _world(store)
    q = {i.node_id: i for i in BF.plan(_ctx(), store, snap, now=T0 + 10)}
    assert {k: i.action for k, i in q.items()} == {"s_pass": "promote", "s_ext": "early_confirm", "c_off": "port",
                                                   "c_on": "confirm+1", "r_leg": "port"}
    assert q["c_off"].target == q["r_leg"].target == "inc"
    assert [i.node_id for i in BF.plan(_ctx(), store, snap, now=T0 + 10)][0] == "r_leg"   # best E[gain+] per run
    # a node that closed AFTER the epoch was first seen was judged by the current rule: not re-evaluated
    store.update_node("s_ext", updated_ts=T0 + 20)
    assert "s_ext" not in {i.node_id for i in BF.plan(_ctx(), store, snap, now=T0 + 30)}
    # a futility-cancelled screen run: no promote; a legacy screen extension (< 3 confirm seeds left): no early_confirm
    store.update_run(Run.make_id("s_pass", Stage.SCREEN, 2), status=RunStatus.VOID)
    assert "s_pass" not in {i.node_id for i in BF.plan(_ctx(), store, snap, now=T0 + 30)}
    store.update_node("s_ext", updated_ts=T0)
    assert BF.plan(_ctx(), store, snap, now=T0 + 30)[-1].node_id == "s_ext"
    for seed in (3, 4):
        store.put_run(Run(id=Run.make_id("s_ext", Stage.SCREEN, seed), task_id=store.task_id, node_id="s_ext",
                          seed=seed, stage=Stage.SCREEN, status=RunStatus.DONE, scores={"a": 0.5}))
    assert "s_ext" not in {i.node_id for i in BF.plan(_ctx(), store, snap, now=T0 + 30)}


def test_tick_rate_limits_books_and_never_repeats(store):
    snap = _world(store)
    moved, made = [], iter(f"n{i}" for i in range(9))
    graft = SimpleNamespace(graft=lambda p, n, o, name: f"g-{n}-{o}")
    reopen = lambda n, st, why: (moved.append((n.id, st)), store.update_node(n.id, status=st))    # noqa: E731
    assert BF.tick(_ctx(), store, graft, snap, new_id=lambda: next(made), now=T0 + 10, reopen=reopen) == []   # off
    ctx = _ctx(enabled=True, max_active=2, per_tick=1)
    a1 = BF.tick(ctx, store, graft, snap, new_id=lambda: next(made), now=T0 + 10, reopen=reopen)
    port = store.get_node("n0")
    assert len(a1) == 1 and port.parent_id == "inc" and port.commit == "g-c-r_leg-c-inc" \
        and port.status is NodeStatus.SCREENING and "backfill_of=r_leg" in port.notes
    BF.tick(ctx, store, graft, snap, new_id=lambda: next(made), now=T0 + 11, reopen=reopen)
    assert moved == [("s_pass", NodeStatus.SCREENING)]
    assert BF.tick(ctx, store, graft, snap, new_id=lambda: next(made), now=T0 + 12, reopen=reopen) == []  # 2 active
    store.update_node("n0", status=NodeStatus.SCREENED_OUT)
    BF.tick(ctx, store, graft, snap, new_id=lambda: next(made), now=T0 + 13, reopen=reopen)
    book = store.get_state(BF.K_BOOK)                     # c_on: E[gain+] 0.0003 / 1 run beats c_off's 0.001 / 5
    assert set(book) == {"r_leg", "s_pass", "c_on"} and {e["action"] for e in store.events("backfill")} == \
        {"port", "promote", "confirm+1"}
    assert moved[-1] == ("c_on", NodeStatus.CONFIRMING) and BF.extra_confirm(store, "c_on") == 1
    assert "r_leg" not in {i.node_id for i in BF.plan(ctx, store, snap, now=T0 + 14)}     # once per epoch


def test_a_failed_graft_is_booked_not_retried(store):
    snap = _world(store)
    nograft = SimpleNamespace(graft=lambda *a: "")
    acts = BF.tick(_ctx(enabled=True, per_tick=1), store, nograft, snap, new_id=lambda: "nX", now=T0 + 10,
                   reopen=lambda *a: None)
    assert "did not apply" in acts[0] and store.get_node("nX") is None
    assert store.events("backfill_graft_failed")[0]["node_id"] == "r_leg"
    assert "r_leg" not in {i.node_id for i in BF.plan(_ctx(), store, snap, now=T0 + 11)}


def test_ports_book_contained_and_conflicts_and_a_conflict_gets_one_port_request(store):
    """A port the incumbent already holds is booked already_in_incumbent; a genuine conflict port_conflict with its
    files / keys, then ONE low-priority researcher port request (backfill.port_requests in flight), never again."""
    snap = _world(store)
    outcomes = {"r_leg": ("", "conflict: dataloader.py"), "c_off": ("", "contained")}
    adapter = SimpleNamespace(graft_port=lambda p, n, o, name: outcomes[name[3:]])
    ctx = _ctx(enabled=True, max_active=6, per_tick=6)
    acts = BF.tick(ctx, store, adapter, snap, new_id=lambda: "nX", now=T0 + 10, reopen=lambda *a: None)
    book = store.get_state(BF.K_BOOK)
    assert book["r_leg"]["outcome"] == "port_conflict" and book["r_leg"]["conflict"] == "dataloader.py"
    assert book["c_off"]["outcome"] == "already_in_incumbent" and store.get_node("nX") is None
    assert any("port_conflict (dataloader.py)" in a for a in acts) and any("already_in_incumbent" in a for a in acts)
    assert {e["node_id"] for e in store.events("backfill_already_in_incumbent")} == {"c_off"}
    asked = []
    sel = SimpleNamespace(port_request=lambda st, sn, nid, onto, why: (asked.append((nid, onto, why)),
                                                                       SimpleNamespace(id=f"req-{nid}"))[1])
    assert BF.port_requests(_ctx(), store, snap, sel, 2, set()) == []                     # backfill off
    reqs = BF.port_requests(ctx, store, snap, sel, 2, set())
    assert [r.id for r in reqs] == ["req-r_leg"] and asked == [("r_leg", "inc", "dataloader.py")]
    assert BF.port_requests(ctx, store, snap, sel, 2, {"req-r_leg"}) == []                 # once per node
    store.set_state(BF.K_BOOK, {**store.get_state(BF.K_BOOK), "c_off": {"epoch": 5, "action": "port", "node": ""}})
    assert "c_off" in {i.node_id for i in BF.plan(ctx, store, snap, now=T0 + 11)}   # a pre-graft_port failure: again
    assert "r_leg" not in {i.node_id for i in BF.plan(ctx, store, snap, now=T0 + 11)}


def test_merge_config_keeps_what_the_incumbent_holds_and_flags_real_conflicts():
    from loop.adapters import MISSING, merge_config, subsumed
    base = {"lr": 1.0, "wd": 0.1, "kde": 0.1, "crop": [0.1, 0.3], "old": 1}
    ours = {"lr": 2.0, "wd": 0.1, "kde": 0.05, "crop": [0.1, 0.3], "old": 1, "salient": 1.5}
    theirs = {"lr": 1.5, "wd": 0.2, "kde": 0.2, "crop": [0.2, 0.3], "salient": 1.5}
    put, held, clash = merge_config(base, theirs, ours)
    assert put == {"wd": 0.2, "crop": [0.2, 0.3], "old": MISSING} and held == ["lr", "salient"] and clash == ["kde"]
    assert merge_config({"lr": 1.0}, {"lr": 3.0}, {"lr": 2.0})[0] == {"lr": 3.0}       # the node's further dose
    assert subsumed(1.0, 1.5, 2.0) and not subsumed(1.0, 1.5, 0.5) and not subsumed(True, False, False)


def test_extra_confirm_seed_past_the_plan():
    from loop import seeds as S
    from tests.loop.test_driver import _node as dnode, _roots, _run
    from loop.spec import PromotionCfg, SeedPlan
    cfg = _cfg(seeds=SeedPlan(confirm_pool=(101, 102, 103, 104), max_seeds_per_node=5),
               promotion=PromotionCfg(confirm_max_seeds=3))
    own = [_run("n1", Stage.SCREEN, 11), _run("n1", Stage.SCREEN, 12), _run("n1", Stage.CONFIRM, 101),
           _run("n1", Stage.CONFIRM, 102), _run("n1", Stage.CONFIRM, 103)]
    step = lambda extra: S.step(dnode(NodeStatus.CONFIRMING), own_runs=own, control_runs=_roots([101, 102, 103, 104]),   # noqa: E731
                                post=None, cfg=cfg, sigma_panel=0.004, infra_retry=2, confirm=(2.0, 3), extra=extra)
    assert step(0).status is NodeStatus.NOT_CONFIRMED
    d = step(1)
    assert d.status is None and [n.run_id for n in d.needs] == ["n1:confirm:104"]
    assert dataclasses.is_dataclass(d)


def test_confirm_plus_one_reads_the_live_e_the_confirm_step_judges_by(store, monkeypatch):
    """n0297 (2026-10-07 06:45:36Z): backfill booked confirm+1 on its STORED E=1.14, the confirm step in the same
    tick computed the live E=0.541 and closed it futile before the seed ran. Both now read loop.world's
    confirm_evalue on the current ledger."""
    snap = _world(store)
    LIVE_E["c_on"] = 0.5                                  # stored confirm_e 2.0 can reach the bar; the live E cannot
    assert "c_on" not in {i.node_id for i in BF.plan(_ctx(), store, snap, now=T0 + 10)}
    LIVE_E["c_on"] = 2.5
    item = next(i for i in BF.plan(_ctx(), store, snap, now=T0 + 10) if i.node_id == "c_on")
    assert item.action == "confirm+1" and "E=2.5 " in item.why
    # the real live_e: ComputedWorldModel.confirm_evalue on one corrected ledger read once
    from loop import world as W
    monkeypatch.undo()
    calls, reads = [], []
    monkeypatch.setattr(W.noise, "read_ledger", lambda ctx, st: reads.append(1) or "raw")
    monkeypatch.setattr(W.noise, "correct", lambda ctx, led, sigma: ("corrected", led, sigma))
    monkeypatch.setattr(W.ComputedWorldModel, "confirm_evalue",
                        lambda self, sn, nid, st, ledger=None: calls.append((nid, ledger)) or (0.7, 2))
    e = BF.live_e(_ctx(), store, SimpleNamespace(sigma="sig"))
    assert e(store.get_node("c_on")) == 0.7 and e(store.get_node("s_ext")) == 0.7
    assert reads == [1] and calls == [("c_on", ("corrected", "raw", "sig")), ("s_ext", ("corrected", "raw", "sig"))]


def test_the_campaign_epoch_queues_recent_difficult_closes_on_few_runs_capped(store):
    """Epoch 7: init / architecture / objective nodes closed in the retro window on <= 2 scored runs are queued for a
    campaign (loop.campaign), best E[gain+] first, at most campaign.retro_max over the epoch; never re-opened."""
    from loop import campaign as CP
    from loop.spec import CampaignCfg
    snap = _world(store)
    now = T0 + 3600.0 * 100
    for nid, ts, plus in (("i1", now - 3600.0 * 10, 0.002), ("i2", now - 3600.0 * 20, 0.003),
                          ("i3", now - 3600.0 * 30, 0.001), ("old", now - 3600.0 * 80, 0.005), ("many", now - 60, 0.009)):
        _node(store, nid, NodeStatus.SCREENED_OUT, "inc", notes="brief=moonshot init=timm/vit_s mtype=architecture_addon")
        store.update_node(nid, updated_ts=ts)
        snap.nodes[nid] = _post(nid, 0.2, plus)
    store.put_run(Run(id=Run.make_id("many", Stage.CONFIRM, 101), task_id=store.task_id, node_id="many", seed=101,
                      stage=Stage.CONFIRM, status=RunStatus.DONE, scores={"a": 0.5}))
    _node(store, "obj", NodeStatus.SCREENED_OUT, "inc", notes="brief=moonshot mtype=loss_objective")   # objective
    store.update_node("obj", updated_ts=now - 60, strategy=Strategy.MOONSHOT)       # code-only: after the init swaps
    snap.nodes["obj"] = _post("obj", 0.2, 0.009)
    ctx = _ctx(enabled=True, max_active=0, per_tick=5)            # max_active never holds a campaign: it only queues
    ctx.cfg = dataclasses.replace(ctx.cfg, campaign=CampaignCfg(retro_max=3))
    q = [i for i in BF.plan(ctx, store, snap, now=now) if i.action == "campaign"]
    assert [i.node_id for i in q] == ["i2", "i1", "i3"] and q[0].epoch == 7 and "init swap" in q[0].why
    ctx.cfg = dataclasses.replace(ctx.cfg, campaign=CampaignCfg(retro_max=2))
    assert [i.node_id for i in BF.plan(ctx, store, snap, now=now) if i.action == "campaign"] == ["i2", "i1"]
    BF.tick(ctx, store, SimpleNamespace(), snap, new_id=lambda: "x", now=now, reopen=lambda *a: None)
    assert [x["base"] for x in CP.book(store)["queue"]] == ["i2", "i1"]
    assert store.get_node("i2").status is NodeStatus.SCREENED_OUT                  # the node itself stays closed
    assert not [i for i in BF.plan(ctx, store, snap, now=now + 1) if i.action == "campaign"]   # cap: 2 per epoch


def test_init_swaps_start_first_even_with_a_tiny_value_and_a_queue_keeps_that_order(store):
    """per_tick=1: the init swap (E[gain+] 0.0005) is queued before the objective node (0.009), and loop.campaign
    starts a queued init swap before a queued objective node."""
    from loop import campaign as CP
    from loop.spec import CampaignCfg
    snap = _world(store)
    now = T0 + 3600.0 * 100
    _node(store, "obj", NodeStatus.SCREENED_OUT, "inc", notes="brief=moonshot mtype=loss_objective")
    store.update_node("obj", updated_ts=now - 3600.0, strategy=Strategy.MOONSHOT)
    snap.nodes["obj"] = _post("obj", 0.2, 0.009)
    _node(store, "swap", NodeStatus.SCREENED_OUT, "inc", notes="brief=moonshot init=timm/vit_s mtype=architecture_addon")
    store.update_node("swap", updated_ts=now - 7200.0)
    snap.nodes["swap"] = _post("swap", 0.2, 0.0005)
    ctx = _ctx(enabled=True, max_active=0, per_tick=1)
    ctx.cfg = dataclasses.replace(ctx.cfg, campaign=CampaignCfg(retro_max=5))
    BF.tick(ctx, store, SimpleNamespace(), snap, new_id=lambda: "x", now=now, reopen=lambda *a: None)
    assert [x["base"] for x in CP.book(store)["queue"]] == ["swap"]
    bk = CP.book(store)
    bk["queue"] = [{"base": "obj", "why": "backfill epoch 7: new objective / loss; closed on 1 run(s)"},
                   {"base": "swap", "why": "backfill epoch 7: init swap (x); closed on 1 run(s)"}]
    assert [n.id for n, _ in CP.pending(ctx, store, snap, bk)] == ["swap", "obj"]
