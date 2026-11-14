"""v3-c6r (adversarial review of b8c611a): the code-picked crossover planner -- an unread leg is never "has code" (P1),
ranking by the additive shrunk prediction with a winner's-curse correction (P2), old-epoch config moves transferred
only as the same move + sibling dedup (P3), Bonferroni axis legs (P4), capped ports of top pairs only (P5), retryable
build / git failures (P6), no side effect at planning + an atomic claim (P7); the store's stats lock + closed reads
(M2)."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from loop.contracts import Node, NodeStatus, Operator, Strategy
from loop.core import data as CD

from .test_c6a import _pairs_env


def _sd(post, sds):
    belief = post.belief
    post.belief = lambda nid: (belief(nid)[0], sds.get(nid, 0.001) ** 2)


# -- P1: an unread leg ------------------------------------------------------------------------------------------------
def test_an_unread_leg_is_planned_but_never_merged_ported_or_briefed(ctx, store, monkeypatch):
    PR, led, post = _pairs_env(monkeypatch, {
        "g4": ("inc", "E2", 0.0012, 3, {"cls": (0.012, 4.0), "rob": (-0.008, -2.5)}),
        "old": ("x", "E1", 0.0015, 3, {"seg": (0.01, 4.0), "cls": (-0.002, -1.0)}),     # an old-epoch leg
    })
    unread = {"g4": {"train.global_views": 4}, "old": PR.UNREAD}
    got = PR.plan(post, led, ctx, {}, incumbent="inc", inc_cfg={"train.global_views": 2}, config_of=unread.get, used={})
    [p] = got
    assert p.unread and p.needs_port == () and p.config is None and not PR.briefable(p)
    PR.bind(ctx, store, statuses={}, config_of=unread.get, inc_cfg={"train.global_views": 2}, post=post, pairs=got)
    assert PR.next_pair(SimpleNamespace(post=post, led=led, incumbent="inc")) is None      # never briefed
    ported = []
    tick = SimpleNamespace(port_graft=lambda x, led_: ported.append(x) or [f"port {x}"])
    pl = PR.Planner(ctx, store, SimpleNamespace(), lambda: 5.0)
    assert pl._ports(tick, led, got) == [] and not ported                                  # never ported
    # read: it has code -> the pair needs a port (only now)
    code = {"g4": {"train.global_views": 4}, "old": None}
    [p2] = PR.plan(post, led, ctx, {}, incumbent="inc", inc_cfg={"train.global_views": 2}, config_of=code.get, used={})
    assert not p2.unread and p2.needs_port == ("old",)


def test_the_reader_returns_unread_past_the_budget_and_retries_a_failed_git_read(ctx, store):
    from loop.core import pairs as PR
    pl = PR.Planner(ctx, store, SimpleNamespace(), lambda: 100.0)
    calls = []

    def diff(nid):
        calls.append(nid)
        raise RuntimeError("git hiccup")
    pl._config_diff = diff
    book, fails = {"a": None}, {}
    read = pl._reader(book, fails, [1])
    assert read("a") is None                                       # cached: has code
    assert read("b") is PR.UNREAD and "b" not in book              # a raising read: unread, not "has code"
    assert read("c") is PR.UNREAD and calls == ["b"]               # past the budget: unread, no git read
    assert fails["cfg:b"]["n"] == 1 and fails["cfg:b"]["next"] > 100.0
    read2 = pl._reader(book, fails, [5])
    assert read2("b") is PR.UNREAD and calls == ["b"]              # backed off: not retried yet
    fails["cfg:b"]["next"] = 0.0
    fails["cfg:b"]["n"] = 2
    assert read2("b") is None and book["b"] is None                # the 3rd failure: booked as code (bounded)


def test_a_port_is_booked_only_once_its_port_node_exists(ctx, store, monkeypatch):
    from loop.core import pairs as PR
    from loop import cutover as CO
    p = PR.Pair("old", "g4", 0.0, 0.0, 0.0, 0.002, ("seg", "cls"), ("composite", "composite"), None, "why", ("old",))
    led = SimpleNamespace(epoch="E2abcdef", nodes={})
    pl = PR.Planner(ctx, store, SimpleNamespace(), lambda: 5.0)
    tick = SimpleNamespace(port_graft=lambda x, led_: [f"would port {x} (the adapter cannot graft)"])
    pl._ports(tick, led, [p])
    assert "old" not in (store.get_state(PR.K_PORTED) or {})        # nothing made: not booked, asked again later
    store.ensure_root("r0")

    def graft(x, led_):
        store.put_node(Node(id=CO.port_id(x, led_.epoch), task_id=store.task_id, commit="c", parent_id="root",
                            strategy=Strategy.EXPLORE, operator=Operator.DRAFT, family="f", mechanism_tag="t",
                            diff_hash="d", status=NodeStatus.QUEUED))
        return [f"port {x}"]
    pl._ports(SimpleNamespace(port_graft=graft), led, [p])
    assert store.get_state(PR.K_PORTED)["old"]["port"] == CO.port_id("old", "E2abcdef")


# -- P2: ranking --------------------------------------------------------------------------------------------------------
def test_pairs_rank_by_the_curse_corrected_additive_prediction_not_complementarity(ctx, monkeypatch):
    PR, led, post = _pairs_env(monkeypatch, {
        # a huge-coverage pair whose additive prediction is negative: never planned
        "big_cls": ("inc", "E2", 0.0010, 3, {"cls": (0.02, 5.0), "rob": (-0.02, -5.0)}),
        "neg_rob": ("inc", "E2", -0.0016, 3, {"rob": (0.03, 6.0), "cls": (-0.01, -3.0)}),
        # two clean pairs with the same additive prediction: the precise one first
        "p1": ("inc", "E2", 0.0020, 4, {"seg": (0.006, 3.0)}),
        "p2": ("inc", "E2", 0.0020, 4, {"slide": (0.006, 3.0)}),
        "q1": ("inc", "E2", 0.0020, 4, {"auc": (0.006, 3.0)}),
        "lowz": ("inc", "E2", 0.0004, 4, {"surv": (0.01, 3.0)}),                        # z .4 < leg_z 1: no leg
        "old2": ("x", "E1", 0.0030, 2, {"seg": (0.01, 4.0)}),                            # old epoch n=2: no leg
    })
    _sd(post, {"p1": 0.0005, "p2": 0.0005, "q1": 0.0015})
    got = PR.plan(post, led, ctx, {}, incumbent="inc", used={})
    keys = [p.key for p in got]
    assert "big_cls|neg_rob" not in keys                                          # pred -.0006 <= 0
    assert all("lowz" not in k and "old2" not in k for k in keys)
    assert all(p.pred > 0 for p in got)
    assert keys[0] == "p1|p2"                                                     # same pred, smallest sd
    best = got[0]
    # v3 review K5: the stack prior of the two shrunk legs (min(.004, max(.74 x .002, .24 x .004 + .0013))), no curse
    assert best.pred == pytest.approx(0.00226) and best.value == pytest.approx(best.pred + 0.25 * best.coverage)
    assert "pred +0.00226" in best.why and "stack prior" in best.why and "comp " not in best.why


def test_the_plan_log_carries_the_prediction_not_complementarity(ctx, store, monkeypatch):
    PR, led, post = _pairs_env(monkeypatch, {
        "p1": ("inc", "E2", 0.0020, 4, {"seg": (0.006, 3.0)}),
        "p2": ("inc", "E2", 0.0015, 4, {"slide": (0.006, 3.0)}),
    })
    store.ensure_root("r0")
    pl = PR.Planner(ctx, store, SimpleNamespace(), lambda: 5.0)
    monkeypatch.setattr(PR, "plan", lambda *a, **k: [PR.Pair("p1", "p2", 0.9, 0.5, 0.4, 0.0035, ("seg", "slide"),
                                                             ("composite", "composite"), None, "w", (), 0.001)])
    pl.step(SimpleNamespace(dry=True), led, post, {}, "inc", [])
    [top] = store.get_state(PR.K_PLAN)
    assert top["pred"] == 0.0035 and top["value"] == 0.001 and "comp" not in top


# -- P3: config moves -------------------------------------------------------------------------------------------------
def test_only_the_same_move_transfers_and_a_leg_with_nothing_left_is_no_leg():
    from loop.core import pairs as PR
    mv = PR.moves_of({"a": {"from": 2, "to": 4}, "b": {"from": 0.1, "to": 0.05}})
    eff, dropped, held = PR.transfer(mv, {"a": 2, "b": 0.2})
    assert eff == {"a": 4} and dropped == ("b",) and not held                     # b: the incumbent moved it already
    assert PR.transfer(mv, {"a": 3, "b": 0.2})[0] == {}                            # nothing is the same move
    assert PR.transfer(mv, {"a": 4, "b": 0.05})[2]                                 # already held
    assert PR.transfer(PR.moves_of({"a": 4}), {"a": 3})[0] == {"a": 4}             # an unknown parent: as before


def test_an_old_epoch_config_leg_transfers_only_its_unchanged_keys(ctx, monkeypatch):
    PR, led, post = _pairs_env(monkeypatch, {
        "old_cfg": ("x", "E1", 0.0015, 3, {"rob": (0.01, 4.0), "cls": (-0.002, -1.0)}),
        "old_gone": ("x", "E1", 0.0016, 3, {"slide": (0.01, 4.0)}),
        "cur": ("inc", "E2", 0.0012, 3, {"cls": (0.012, 4.0), "rob": (-0.008, -2.5)}),
    })
    moves = {"old_cfg": {"data.local_views": {"from": 14, "to": 9}, "data.jitter": {"from": 0.4, "to": 0.2}},
             "old_gone": {"data.blur": {"from": 0.5, "to": 0.1}},
             "cur": {"data.global_views": {"from": 2, "to": 4}}}
    inc = {"data.local_views": 14, "data.jitter": 0.3, "data.blur": 0.3, "data.global_views": 2}
    got = PR.plan(post, led, ctx, {}, incumbent="inc", inc_cfg=inc, config_of=moves.get, used={})
    assert all("old_gone" not in p.key for p in got)                              # nothing of its move transfers
    p = next(p for p in got if p.key == "cur|old_cfg")
    assert p.config == {"data.local_views": 9, "data.global_views": 4} and "dropped ['data.jitter']" in p.why


def test_config_siblings_count_as_one_mechanism(ctx, monkeypatch):
    PR, led, post = _pairs_env(monkeypatch, {
        "n1214": ("inc", "E2", 0.0015, 3, {"rob": (0.01, 3.0)}),
        "n1216": ("inc", "E2", 0.0018, 3, {"rob": (0.01, 3.5)}),
        "n1218": ("inc", "E2", 0.0012, 3, {"rob": (0.01, 2.5)}),
        "other": ("inc", "E2", 0.0020, 3, {"cls": (0.01, 4.0)}),
    })
    diffs = {"n1214": {"data.hed_jitter": 0.035, "data.color_jitter_saturation": 0.15, "data.aug_hue_local": 0.1},
             "n1216": {"data.hed_jitter": 0.035, "data.color_jitter": 0.15, "data.aug_hue_local": 0.1},
             "n1218": {"data.hed_jitter": 0.035, "data.color_jitter": 0.15, "data.color_jitter_saturation": 0.15,
                       "data.aug_hue_local": 0.1},
             "other": {"train.global_views": 4}}
    info: dict = {}
    got = PR.plan(post, led, ctx, {}, incumbent="inc", inc_cfg={k: 0 for d in diffs.values() for k in d},
                  config_of=diffs.get, used={}, info=info)
    sibs = [k for p in got for k in (p.a, p.b) if k.startswith("n12")]
    assert set(sibs) == {"n1216"} and set(info["siblings"]) == {"n1214", "n1218"}   # the best-z sibling stays


# -- P4: axis legs ------------------------------------------------------------------------------------------------------
def test_axis_legs_need_a_bonferroni_z_three_seeds_and_never_the_survival_axis(ctx, monkeypatch):
    spec = {"c": ("inc", "E2", 0.0015, 3, {"cls": (0.01, 4.0), "rob": (-0.006, -3.0), "surv": (-0.01, -3.0)})}
    spec.update({f"f{i}": ("inc", "E2", -0.0005, 3, {"seg": (0.001, 0.5)}) for i in range(40)})   # many candidates
    spec.update({"ax_ok": ("inc", "E2", -0.0005, 3, {"rob": (0.02, 5.0)}),
                 "ax_weak": ("inc", "E2", -0.0005, 3, {"rob": (0.012, 3.0)}),    # z 3 < Bonferroni over 42 x 5
                 "ax_n2": ("inc", "E2", -0.0005, 2, {"rob": (0.02, 6.0)}),       # only 2 seeds
                 "ax_surv": ("inc", "E2", -0.0005, 3, {"surv": (0.03, 8.0)})})   # survival: excluded
    PR, led, post = _pairs_env(monkeypatch, spec)
    info: dict = {}
    legs = {l.node: l for l in PR.legs(post, led, ctx, {}, incumbent="inc", info=info)}
    assert info["axis_z"] > 3.0
    assert "ax_ok" in legs and legs["ax_ok"].kind == "axis"
    assert not {"ax_weak", "ax_n2", "ax_surv"} & set(legs)


# -- P6: build failures --------------------------------------------------------------------------------------------------
def test_a_failed_build_backs_off_and_gives_up_without_quarantining(ctx, store):
    from loop.core import pairs as PR
    store.ensure_root("r0")
    store.put_node(Node(id="inc", task_id=store.task_id, commit="c-inc", parent_id="root", strategy=Strategy.EXPLORE,
                        operator=Operator.DRAFT, family="f", mechanism_tag="inc", diff_hash="inc",
                        status=NodeStatus.SCREENING))
    t = [0.0]

    def build(c, lv, kn, df, name):
        raise OSError("disk full")
    pl = PR.Planner(ctx, store, SimpleNamespace(config_rel="c.yaml", dry_build=build), lambda: t[0])
    p = PR.Pair("a", "b", 0.0, 0.0, 0.0, 0.002, ("cls", "rob"), ("composite", "composite"), {"k": 1}, "w")
    led = CD.Ledger({}, (), (), frozenset(), ((0.0, "root"),))
    fails: dict = {}
    for i in range(3):
        acts = pl._merges(led, {}, "inc", store.get_node("inc"), [p], fails)
        assert acts and "build failed" in acts[0]
        assert pl._merges(led, {}, "inc", store.get_node("inc"), [p], fails) == []       # backed off
        t[0] = fails["pair:a|b"]["next"] + 1
    assert store.get_state(PR.K_USED)["a|b"]["how"] == "failed"
    # a deterministic "no change" is unbuilt at once
    pl2 = PR.Planner(ctx, store, SimpleNamespace(config_rel="c.yaml", dry_build=lambda *a: ("c-inc", "")), lambda: 1.0)
    p2 = PR.Pair("c", "d", 0.0, 0.0, 0.0, 0.002, ("cls", "rob"), ("composite", "composite"), {"k": 1}, "w")
    pl2._merges(led, {}, "inc", store.get_node("inc"), [p2], {})
    assert store.get_state(PR.K_USED)["c|d"]["how"] == "unbuilt"


# -- P7: no side effect at planning; an atomic claim; the book swap ---------------------------------------------------
def test_the_claim_is_atomic_under_concurrent_sessions(store):
    from loop.core import pairs as PR
    wins = []
    barrier = threading.Barrier(8)

    def go(i):
        barrier.wait()
        wins.append(PR.claim(store, "a", "b", "brief", f"req-{i}"))
    ts = [threading.Thread(target=go, args=(i,)) for i in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert sorted(wins) == [False] * 7 + [True]
    owner = store.get_state(PR.K_USED)["a|b"]["node"]
    PR.release(store, "a", "b", "start failed", "req-not-the-owner")
    assert "a|b" in store.get_state(PR.K_USED)                      # only the owner's claim is given back
    PR.release(store, "a", "b", "start failed", owner)
    assert "a|b" not in store.get_state(PR.K_USED)


def test_the_ideation_side_reads_a_fully_built_book_never_a_reset_one(ctx, store, monkeypatch):
    from loop.core import pairs as PR
    pl = PR.Planner(ctx, store, SimpleNamespace(), lambda: 1.0)
    pl._book = {"a": {"k": {"from": 1, "to": 2}}}
    seen = []

    def plan(*a, config_of=None, **k):
        seen.append(pl.cached("a"))                                 # a brief thread reading mid-step
        config_of("zz")
        return []
    monkeypatch.setattr(PR, "plan", plan)
    pl.step(SimpleNamespace(dry=True), SimpleNamespace(), SimpleNamespace(), {}, "inc", [])
    assert seen == [{"k": {"from": 1, "to": 2}}] and pl._book is not None
    assert pl.cached("never") is PR.UNREAD                          # the off-tick lookup never reads git


# -- M2: the store -----------------------------------------------------------------------------------------------------
def test_a_read_after_close_raises_a_clear_error(ctx, tmp_path):
    from loop.store import SqliteStore, StoreClosed
    s = SqliteStore(tmp_path / "x.db", ctx.task_id)
    s.set_state("k", 1)
    assert s.get_state("k") == 1 and s.lock_stats()["reads"] >= 1
    s.close()
    with pytest.raises(StoreClosed, match="closed"):
        s.get_state("k")


def test_a_combination_is_no_leg_and_old_epoch_legs_count_at_a_discount(ctx, monkeypatch):
    import dataclasses
    PR, led, post = _pairs_env(monkeypatch, {
        "xo": ("root", "E1", 0.0135, 3, {"slide": (0.035, 6.0), "rob": (0.03, 5.0)}),   # an old crossover onto root
        "old": ("x", "E1", 0.0020, 3, {"seg": (0.01, 4.0)}),
        "cur": ("inc", "E2", 0.0020, 3, {"cls": (0.01, 4.0)}),
    })
    led.nodes["xo"] = dataclasses.replace(led.nodes["xo"], notes="request=r brief=frontier xover=n0655,n0155")
    legs = {l.node: l for l in PR.legs(post, led, ctx, {}, incumbent="inc")}
    assert "xo" not in legs
    # v3 review K5: discounted ONCE -- by the posterior's V / OLD_EPOCH_W (post.belief), not again on the mean
    assert legs["old"].m == pytest.approx(0.002) and legs["cur"].m == pytest.approx(0.002)
