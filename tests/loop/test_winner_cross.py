"""Standing exploitation of confirmed winners: every untried, non-nested pair of CONFIRMED nodes is
crossed deterministically (outside the bandit), most complementary first; every confirmed node is
audited; Labless has no spacing but a failed try still backs off.
Run: uv run pytest tests/loop -q --confcutdir=tests/loop"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

from loop import labless as L
from loop.contracts import PANEL, ROOT_NODE_ID, Node, NodeStatus, Operator, Posterior, Strategy
from loop.driver import K_WINX, NOISE_KEY
from loop.select import winner_pairs
from loop.spec import LablessCfg
from tests.loop.test_driver import Scripts, _make, _world

SNAP = SimpleNamespace(nodes={})


def _plant(store, nid, family, cfg, *, parent=ROOT_NODE_ID, strategy=Strategy.EXPLORE, notes="",
           second=None, diff_path="", status=NodeStatus.CONFIRMED):
    store.put_node(Node(id=nid, task_id=store.task_id, commit=store.get_node(ROOT_NODE_ID).commit,
                        parent_id=parent, strategy=strategy, operator=None, family=family, mechanism_tag=nid,
                        diff_hash=f"h{nid}", status=status, second_parent_id=second, notes=notes,
                        diff_path=diff_path, created_ts=1.0))
    store.set_state(f"config_diff:{nid}", cfg)


def _live_shape(store, tmp_path):
    """The live c12 shape: A (crops+kde), B (lr stack) ⊂ C (lr stack + kde weight), D = JEPA x (A, C)."""
    code = tmp_path / "d.diff"
    code.write_text("diff --git a/configs/main.yaml b/configs/main.yaml\ndiff --git a/train.py b/train.py\n")
    _plant(store, "A", "augmentation", {"data.jitter": 0.04, "dino.kde": 6.6})
    _plant(store, "B", "optimization", {"dino.lr": 2e-4, "dino.decay": 0.8, "dino.kde": 6.6})
    _plant(store, "C", "optimization", {"dino.lr": 2e-4, "dino.decay": 0.8, "dino.kde": 6.6, "dino.w": 0.07})
    _plant(store, "X", "optimization", {}, status=NodeStatus.SCREENED_OUT, strategy=Strategy.CROSSOVER,
           notes="xover=C,A", second="A")
    _plant(store, "D", "training", {"data.jitter": 0.04, "dino.kde": 6.6, "dino.lr": 2e-4, "dino.decay": 0.8,
                                    "dino.w": 0.07}, strategy=Strategy.CROSSOVER, notes="xover=X,C",
           second="C", diff_path=str(code))


def test_pairs_skip_tried_and_nested_and_order_by_complementarity(store, tmp_path):
    store.ensure_root("c0")
    _live_shape(store, tmp_path)
    assert winner_pairs(store, SNAP) == [("B", "A")]          # A x C tried (X); B ⊂ C; A, C ⊂ D (built from)
    _plant(store, "E", "training", {"train.steps": 2}, diff_path=str(tmp_path / "d.diff"))  # code: never nested
    _plant(store, "F", "optimization", {"dino.lr": 3e-4})
    pairs = winner_pairs(store, SNAP)
    assert pairs[:5] == [("E", "A"), ("E", "B"), ("E", "C"), ("F", "A"), ("F", "E")]   # other family, disjoint keys
    assert pairs[-3:] == [("E", "D"), ("F", "C"), ("F", "B")]                   # same family last, least overlap first
    assert not {frozenset("BC"), frozenset("AC"), frozenset("AD"), frozenset("BD"), frozenset("CD")} & set(map(frozenset, pairs))
    assert ("B", "A") not in winner_pairs(store, SNAP, {frozenset("BA")})


def test_stack_head_is_the_incumbent_unless_a_challenger_beats_it_on_evidence(store):
    """n0292 herding: a null child with a higher confirm-only mean on two lucky seeds must not become
    every banked node's partner; the head moves only when P(U>U_inc) >= head_p."""
    store.ensure_root("c0")
    for nid, fam in (("I", "optimization"), ("N", "objective"), ("K", "augmentation"), ("M", "training")):
        _plant(store, nid, fam, {f"k.{nid}": 1}, status=NodeStatus.CONFIRMED if nid == "I" else NodeStatus.NOT_CONFIRMED)

    def snap(p_n):
        post = lambda m, p: SimpleNamespace(confirm_estimate=Posterior(m, 0.002, 2), utility=Posterior(m, 0.002, 4),  # noqa: E731
                                            p_positive=0.99, p_gain=0.99, p_beats_incumbent=p, vetoed=False)
        return SimpleNamespace(nodes={"I": post(0.004, 0.0), "N": post(0.006, p_n), "K": post(0.003, 0.3),
                                      "M": post(0.002, 0.2)}, incumbent_id="I", banked=("K", "M", "N"))
    sibs = {frozenset("KM"), frozenset("KN"), frozenset("MN")}               # banked root siblings stack too
    head = lambda s: [p for p in winner_pairs(store, s) if frozenset(p) not in sibs]   # noqa: E731
    assert head(snap(0.55)) and all("I" in p for p in head(snap(0.55)))     # N's lead is noise: head stays I
    assert head(snap(0.9)) and all("I" in p for p in head(snap(0.9)))       # banked N never heads (n0292)
    assert sibs <= {frozenset(p) for p in winner_pairs(store, snap(0.55))}
    no_inc = SimpleNamespace(**{**vars(snap(0.55)), "incumbent_id": ROOT_NODE_ID})
    assert {frozenset(p) for p in winner_pairs(store, no_inc)} == sibs      # no head: only sibling stacks
    _plant(store, "C", "regularization", {"k.C": 1})                         # a CONFIRMED challenger heads
    s9 = snap(0.9)
    s9.nodes["C"] = SimpleNamespace(**{**vars(s9.nodes["N"]), "confirm_estimate": Posterior(0.007, 0.002, 2)})
    bank_pairs = [p for p in winner_pairs(store, s9) if set(p) & {"K", "M", "N"} and frozenset(p) not in sibs]
    assert bank_pairs and all("C" in p for p in bank_pairs)
    _plant(store, "Q", "objective", {"k.Q": 1}, parent="I", status=NodeStatus.NOT_CONFIRMED)
    q = SimpleNamespace(**{**vars(snap(0.55)), "banked": ("K", "M", "N", "Q")})
    q.nodes = {**q.nodes, "Q": q.nodes["K"]}
    assert not any("Q" in p and "I" not in p for p in winner_pairs(store, q))   # no banked sibling: head only


def test_a_leg_without_a_real_own_increment_is_never_crossed(store):
    """n0363: confirmed, but the same commit as n0292 re-rooted on root -- its own increment over the
    ancestor it contains is null, so it is no cross leg (P(own gain > 0) must exceed leg_p)."""
    store.ensure_root("c0")
    for nid, fam in (("A", "optimization"), ("B", "objective"), ("C", "augmentation")):
        _plant(store, nid, fam, {f"k.{nid}": 1})
    post = lambda g: SimpleNamespace(confirm_estimate=Posterior(0.01, 0.002, 2), utility=Posterior(0.01, 0.002, 4),  # noqa: E731
                                     p_positive=0.99, p_gain=g, p_beats_incumbent=0.0, vetoed=False)
    snap = SimpleNamespace(nodes={"A": post(0.95), "B": post(0.6), "C": post(0.9)}, incumbent_id="A", banked=())
    assert winner_pairs(store, snap) == [("C", "A")]
    assert winner_pairs(store, snap, leg_p=0.5) and all(len(p) == 2 for p in winner_pairs(store, snap, leg_p=0.5))
    assert {"B"} & {x for p in winner_pairs(store, snap, leg_p=0.5) for x in p}


def test_stacking_requests_hold_their_own_max_stack_slots_never_max_sides(tmp_path):
    """live 10-05..10-07: max_side 1, held by debug fixes / tuning / ports, starved every winner cross and leg branch.
    Stacking holds researcher.max_stack slots of its own; a side request holding max_side does not block it."""
    from loop.spec import ResearcherCfg
    ctx = _world(tmp_path, review_every=1000)
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, researcher=ResearcherCfg(
        max_concurrent=5, max_side=1, max_stack=2)))
    driver, store, _ = _make(ctx, Scripts())
    driver.tick()
    for i, fam in enumerate(("optimization", "augmentation", "regularization", "training", "objective")):
        _plant(store, f"W{i}", fam, {f"k.w{i}": 1})
    assert len(winner_pairs(store, driver.snapshot)) >= 4
    driver._research.clear()
    side = SimpleNamespace(id="debug-1", operator=Operator.DEBUG, parent_ids=("X",))
    driver._research[object()] = side                     # a debug fix holds the one max_side slot
    driver._side.add(side.id)
    assert driver._side_room() == 0 and driver._stack_room() == 2
    got = driver._winner_cross_requests()
    assert len(got) == 2
    for r in got:
        driver._research[object()] = r
        driver._side.add(r.id)
        driver._stack.add(r.id)
    assert driver._stack_room() == 0 and driver._winner_cross_requests() == []
    assert driver._side_room() == 0                        # stacking never counts against max_side either
    driver._research = {k: v for k, v in driver._research.items() if v is not side}
    assert driver._side_room() == 1
    driver._sync_inflight()
    assert sorted(e["id"] for e in store.get_state("driver.inflight") if e["stack"]) == sorted(r.id for r in got)


def test_driver_queues_every_pair_once_outside_the_bandit_and_audits_every_confirmed(tmp_path):
    ctx = _world(tmp_path, review_every=1000)
    driver, store, _ = _make(ctx, Scripts())
    driver.tick()
    _live_shape(store, tmp_path)
    driver.tick()
    started = [e for e in store.events("research_started") if e.get("winner_cross")]
    assert [e["parents"] for e in started] == [["B", "A"]]
    assert [e["parents"] for e in store.events("winner_cross")] == [["B", "A"]]
    assert "B+A" in (store.get_state(K_WINX) or {})
    for _ in range(2):
        driver.tick()
    child = next(n for n in store.nodes() if "xover=B,A" in (n.notes or ""))
    assert child.operator is Operator.CROSSOVER and child.strategy is Strategy.CROSSOVER
    assert len([e for e in store.events("research_started") if e.get("winner_cross")]) == 1
    a = store.get_state("driver.audit")
    assert {"A", "B", "C", "D"} <= set(a["queue"] + a["passed"] + a["failed"] + [(a["current"] or {}).get("node")])
    post = SimpleNamespace(confirm_estimate=Posterior(0.01, 0.002, 4), utility=Posterior(0.01, 0.002, 4))
    driver.snapshot = dataclasses.replace(driver.snapshot, incumbent_id="C", nodes={k: post for k in "ABCD"})
    if not store.get_state(NOISE_KEY):
        store.set_state(NOISE_KEY, {PANEL: {"sigma": 0.003, "df_eff": 20.0}})
    driver._gate_backfilled = True
    try:
        driver._audit([])
    except Exception:  # noqa: BLE001 - only the pick matters here
        pass
    assert store.get_state("driver.audit")["current"]["node"] == "C"   # the incumbent jumps the audit queue


def test_labless_no_spacing_but_a_failed_try_backs_off(ctx, store):
    from tests.loop.test_labless import Fake, _node, _sub
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, labless=LablessCfg(enabled=True, min_hours_between=0)))
    for nid in ("n0001", "n0002"):
        _node(store, nid)
    now, fake = [0.0], Fake([RuntimeError("exit 1"), "sid-2", "sid-1"])
    sub = _sub(ctx, store, fake, now)
    sub.step(["n0001", "n0002"], None), sub.step(["n0001", "n0002"], None)       # n0001 fails
    sub.step(["n0001", "n0002"], None), sub.step(["n0001", "n0002"], None)       # n0002 at once
    assert store.get_state(L.K_LABLESS)["n0002"]["submission_id"] == "sid-2"
    assert sub.step(["n0001", "n0002"], None) == []                             # n0001 backs off
    now[0] += L.MIN_BACKOFF_H * 3600
    sub.step(["n0001", "n0002"], None), sub.step(["n0001", "n0002"], None)
    assert store.get_state(L.K_LABLESS)["n0001"]["submission_id"] == "sid-1" and len(fake.calls) == 3
    now[0] += 100 * 3600
    assert sub.step(["n0001", "n0002"], None) == []                             # never twice


def test_failed_crossover_is_retried_once_for_a_non_backend_reason(tmp_path):
    from loop.contracts import ResearchResult
    ctx = _world(tmp_path, review_every=1000)
    driver, store, _ = _make(ctx, Scripts())
    driver.tick()
    _live_shape(store, tmp_path)
    driver.tick()
    req_id = store.get_state(K_WINX)["B+A"]
    req = next(r for r in driver._research.values() if r.id == req_id)
    driver._research = {f: r for f, r in driver._research.items() if r.id != req_id}
    driver._intake(req, ResearchResult("failed", None, "boom"))
    assert "B+A" not in (store.get_state(K_WINX) or {})                 # freed once
    driver.tick()
    req2_id = (store.get_state(K_WINX) or {}).get("B+A")
    assert req2_id and req2_id != req_id                                 # re-queued
    req2 = next(r for r in driver._research.values() if r.id == req2_id)
    driver._research = {f: r for f, r in driver._research.items() if r.id != req2_id}
    driver._intake(req2, ResearchResult("failed", None, "boom again"))
    assert store.get_state(K_WINX)["B+A"] == req2_id                     # second failure stays booked


def _leg(store, nid, tag, cfg, status=NodeStatus.NOT_CONFIRMED, seeds=(11, 12, 101), parent=ROOT_NODE_ID):
    from loop.contracts import Run, RunStatus, Stage
    _plant(store, nid, "training", cfg, parent=parent, status=status)
    store.update_node(nid, mechanism_tag=tag)
    for s, st in zip(seeds, (Stage.SCREEN, Stage.SCREEN, Stage.CONFIRM, Stage.CONFIRM)):
        store.put_run(Run(id=Run.make_id(nid, st, s), task_id=store.task_id, node_id=nid, seed=s, stage=st,
                          status=RunStatus.DONE, scores={"a": 0.5}))


def _gain(m, sd=0.0015, vetoed=False):
    return SimpleNamespace(increment=Posterior(m, sd, 3), vetoed=vetoed)


def test_unconfirmed_small_signals_stack_on_their_carrier(store):
    """Operator 2026-10-03: a closed node whose own gain is positive over ALL its seeds (P >= 0.7, >= 3 seeds) is a
    stack leg even though it never confirmed; one stack per carrier of its highest-P legs with distinct normalized
    tags and no shared knob; the confirm seeds pool into P; a vetoed, active, 2-seed or built stack never is one."""
    from loop.select import pooled_gain, stack_plan
    store.ensure_root("c0")
    _leg(store, "L1", "slide_balanced_sampling", {})                   # P 0.99
    _leg(store, "L2", "slide_balanced_sampling_v2", {})                # 0.96: another tag
    _leg(store, "L3", "Slide-Balanced-Sampled", {})                     # 0.95: L1's normalized tag
    _leg(store, "L4", "patch_koleo", {"dino.koleo": 0.1})              # 0.9
    _leg(store, "L5", "varjepa", {"dino.koleo": 0.2})                  # 0.85: L4's knob -> a conflict
    _leg(store, "L6", "xview_corr", {})                                # selection +, confirm seeds - -> pooled < 0.7
    _leg(store, "L7", "lonely", {}, seeds=(11, 12))                    # 2 scored seeds
    _leg(store, "L8", "harm", {})                                      # vetoed
    _leg(store, "L9", "live_one", {}, status=NodeStatus.CONFIRMING)    # unresolved
    _leg(store, "M1", "under_a_closed_parent", {}, parent="L1")        # its carrier is no established one
    _leg(store, "L10", "tiny", {})                                     # 0.6: below the bar
    snap = SimpleNamespace(nodes={"L1": _gain(0.0042), "L2": _gain(0.0030), "L3": _gain(0.0025), "L4": _gain(0.0020),
                                  "L5": _gain(0.0016), "L6": _gain(0.0015), "L7": _gain(0.004),
                                  "L8": _gain(0.004, vetoed=True), "L9": _gain(0.004), "M1": _gain(0.004),
                                  "L10": _gain(0.0004)},
                           own_confirm={"L6": (-0.004, 1)}, banked=())
    assert pooled_gain(snap, "L6", 0.003) < 0.7 < pooled_gain(snap, "L4", 0.003)
    plan = stack_plan(store, snap, 0.003)
    assert plan == [("L1", "L2", "L4")]
    assert stack_plan(store, snap, 0.003, {("L1", "L2", "L4")}) == []


def test_a_stack_confirms_only_on_seeds_none_of_its_legs_ran(tmp_path):
    """The stack's verdict never rests on a seed a leg was selected on: its confirm pool is the pool + STACK_SEEDS
    less every leg's scored seed; any other node keeps the plain pool."""
    from loop import seeds as S
    ctx = _world(tmp_path, review_every=1000)
    driver, store, _ = _make(ctx, Scripts())
    driver.tick()
    pool = [int(s) for s in ctx.cfg.seeds.confirm_pool]
    _leg(store, "L1", "a", {}, seeds=(11, 12, pool[0]))
    _leg(store, "L2", "b", {}, seeds=(11, 12, pool[1]))
    _plant(store, "ST", "training", {}, strategy=Strategy.CROSSOVER, status=NodeStatus.CONFIRMING,
           notes="xover=L1,L2 stack_legs=L1,L2")
    got = driver._stack_pool(store.get_node("ST"), driver._runs_by_node())
    assert got == [s for s in (*pool, *S.STACK_SEEDS) if s not in (11, 12, pool[0], pool[1])]
    assert driver._stack_pool(store.get_node("L1"), driver._runs_by_node()) is None


def test_the_driver_queues_each_stack_once_and_caps_stacks_in_flight(tmp_path, monkeypatch):
    import loop.driver as DR
    ctx = _world(tmp_path, review_every=1000)
    driver, store, _ = _make(ctx, Scripts())
    driver.tick()
    _leg(store, "L1", "a", {})
    _leg(store, "L2", "b", {})
    _leg(store, "L3", "c", {})
    plans = [("L1", "L2"), ("L2", "L3"), ("L1", "L3")]
    monkeypatch.setattr(DR, "stack_plan", lambda store, snap, sigma, skip: [p for p in plans if p not in skip])
    driver._research.clear()
    got = driver._stack_requests()
    assert [r.parent_ids for r in got] == [("L1", "L2"), ("L2", "L3")][:min(DR.STACK_MAX, driver._side_room() + 2)]
    assert all("STACK of 2 unconfirmed small signals" in r.brief for r in got)
    for r in got:
        driver._research[object()] = r
    assert driver._stack_requests() == []                                # STACK_MAX in flight / no side room
    assert set(store.get_state(DR.K_STACK)) == {"+".join(r.parent_ids) for r in got}


def test_a_wanted_leg_branch_takes_a_stack_slot_before_the_winner_crosses(tmp_path):
    """live 10-07 17:54-22:00Z: L4 on n0655 sat want=True / pending='' for 4 h while every freed max_stack slot went to
    the next winner cross (they were requested first in the tick). The leg branch is ONE request and is asked first."""
    from loop import legs as LG
    from loop.spec import LegsCfg, ResearcherCfg
    ctx = _world(tmp_path, review_every=1000)
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(
        ctx.cfg, legs=LegsCfg(enabled=True), researcher=ResearcherCfg(max_concurrent=4, max_side=1, max_stack=1)))
    driver, store, _ = _make(ctx, Scripts())
    driver.tick()
    _live_shape(store, tmp_path)
    store.set_state(LG.K_LEGS, {"seq": 4, "batches": [], "current": {
        "batch": 4, "carrier": "D", "round": 4, "legs": [], "dropped": [], "own": [], "shared": [], "phase": "build",
        "want": True, "pending": "", "fails": 0, "branch": "", "started": 0.0, "designs": {}}})
    driver.tick()
    started = store.events("research_started")
    assert [e["brief_kind"] for e in started if e["request_id"].startswith("legs-")] == ["legs"]
    assert not [e for e in started if e.get("winner_cross")]          # the one stack slot went to the leg branch
    assert LG.book(store)["current"]["pending"] == "legs-L4-1"
