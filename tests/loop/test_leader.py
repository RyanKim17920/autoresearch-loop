"""loop.leader: our own Labless submission leads the board -> the bar's leader_median is re-based (max of the configured
number and our leader's estimate); another contributor leading / no rows / a failed fetch change nothing; a node (or
its exact code commit) already on Labless is never resubmitted (11-10: a48ee1990d / n0103 led while the bar stayed
.5471 + .005)."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

from loop import leader as LD
from loop import runner as R
from loop.contracts import RunStatus, Stage
from loop.core.data import AXES, Ledger, Node, Obs

from .conftest import InlineExecutor
from .test_runner import _Pending, _node

BASE = "b" * 40
SHA_OURS, SHA_OTHER, SHA_KID = "9ae54bc3" + "1" * 32, "5f4e121a" + "2" * 32, "c0ffee00" + "3" * 32


def _cfg(ctx, **kw):
    lc = dataclasses.replace(ctx.cfg.labless, **{"enabled": True, "leader_median": 0.5471, "promote_margin": 0.005,
                                                 **kw})
    return dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, labless=lc))


def _row(rid: str, commit: str, leader: bool, updated: str = "2026-11-10T19:00:00Z", **kw) -> dict:
    return {"run_id": rid, "git_commit": commit, "is_leader": leader, "verdict": "replicated" if leader else "promising",
            "updated_at": updated, "metric_value": 0.5618, "metrics": {"open_score": 0.5618}, **kw}


def _n(nid, parent):
    return Node(nid, parent, None, "explore", "f", "explore", "root", 0.0, "", "", BASE, "")


def _ledger(means: dict[str, tuple[float, int, str]]) -> Ledger:
    """node -> (mean, seeds, stage): runs on the one epoch BASE (jitter +-.002 around the mean)."""
    nodes = {"root": _n("root", None), **{k: _n(k, "root") for k in means}}
    obs = [Obs(f"root:screen:{s}", "root", s, "screen", "select", {a: 0.54 for a in AXES}, 0.54 + (0.001 if s % 2 else -0.001),
               1.0, 0.0, 0, None, BASE) for s in range(1, 9)]
    for k, (m, n, stage) in means.items():
        for i in range(n):
            y = m + (0.002 if i % 2 else -0.002)
            obs.append(Obs(f"{k}:{stage}:{100 + i}", k, 100 + i, stage, "test", {a: y for a in AXES}, y, 1.0, 0.0, 0,
                           None, BASE))
    return Ledger(nodes, (), tuple(obs), frozenset(), ((0.0, "root"),), bases=((0.0, "root", BASE),))


def test_our_leader_rebases_the_bar_to_its_local_test_mean(ctx, store):
    ctx = _cfg(ctx)
    _node(store, "n0103", commit=SHA_OURS)
    store.put_board_runs([_row("run_sub_8eb1eb614b", SHA_OTHER, True, "2026-11-10T17:22:02Z"),   # a stale earlier leader
                          _row("run_sub_a48ee1990d", SHA_OURS, True)], ts=1.0)
    led = _ledger({"n0103": (0.558, 4, "confirm")})
    out = LD.sync(ctx, store, led, 100.0)
    assert len(out) == 1 and "re-based leader .5471 -> .5580" in out[0].replace("0.", ".") and "n0103" in out[0]
    st = store.get_state(LD.K_LEADER)
    assert st["node"] == "n0103" and st["run_id"] == "run_sub_a48ee1990d" and "fresh TEST" in st["source"]
    eff = LD.effective(ctx.cfg, store)
    assert abs(eff.labless.leader_median - 0.558) < 1e-9 and eff.labless.promote_margin == 0.005
    assert LD.sync(ctx, store, led, 200.0) == []                  # unchanged: said once
    assert [e["event"] for e in store.events("labless_leader_rebase")] == ["labless_leader_rebase"]


def test_our_leader_never_lowers_the_configured_bar_and_prefers_a_rerun_median(ctx, store):
    ctx = _cfg(ctx)
    _node(store, "n0103", commit=SHA_OURS)
    store.put_board_runs([_row("run_sub_a48ee1990d", SHA_OURS, True)], ts=1.0)
    out = LD.sync(ctx, store, _ledger({"n0103": (0.540, 4, "confirm")}), 100.0)
    assert "does not exceed the configured leader_median" in out[0]
    assert LD.effective(ctx.cfg, store).labless.leader_median == 0.5471
    store.put_board_runs([_row("run_sub_a48ee1990d", SHA_OURS, True, metrics={"rerun_median": 0.5602})], ts=2.0)
    out = LD.sync(ctx, store, _ledger({"n0103": (0.540, 4, "confirm")}), 200.0)
    assert "Labless rerun median" in out[0] and abs(LD.effective(ctx.cfg, store).labless.leader_median - 0.5602) < 1e-9


def test_leader_matched_by_our_submission_id_too(ctx, store):
    ctx = _cfg(ctx)
    _node(store, "n7", commit="")                                   # no usable commit: the book entry matches
    store.set_state(R.K_LABLESS, {"n7": {"submission_id": "a48ee1990d"}})
    store.put_board_runs([_row("run_sub_a48ee1990d", "", True)], ts=1.0)
    LD.sync(ctx, store, _ledger({"n7": (0.56, 4, "confirm")}), 1.0)
    assert store.get_state(LD.K_LEADER)["node"] == "n7"


def test_not_our_leader_is_a_noop_and_drops_an_earlier_rebase(ctx, store):
    ctx = _cfg(ctx)
    _node(store, "n0103", commit=SHA_OURS)
    led = _ledger({"n0103": (0.558, 4, "confirm")})
    store.put_board_runs([_row("run_sub_x", SHA_OTHER, True)], ts=1.0)
    assert LD.sync(ctx, store, led, 1.0) == []
    assert LD.effective(ctx.cfg, store) is ctx.cfg                  # the configured bar, untouched
    store.put_board_runs([_row("run_sub_a48ee1990d", SHA_OURS, True, "2026-11-10T20:00:00Z")], ts=2.0)
    LD.sync(ctx, store, led, 2.0)
    assert abs(LD.effective(ctx.cfg, store).labless.leader_median - 0.558) < 1e-9
    store.put_board_runs([_row("run_sub_a48ee1990d", SHA_OURS, False, "2026-11-10T21:00:00Z"),
                          _row("run_sub_y", SHA_OTHER, True, "2026-11-10T21:00:00Z")], ts=3.0)
    out = LD.sync(ctx, store, led, 3.0)
    assert out and "not ours" in out[0] and LD.effective(ctx.cfg, store).labless.leader_median == 0.5471


def test_no_rows_no_leader_or_unmeasurable_keep_the_last_bar(ctx, store):
    ctx = _cfg(ctx)
    _node(store, "n0103", commit=SHA_OURS)
    assert LD.sync(ctx, store, None, 1.0) == [] and store.get_state(LD.K_LEADER) is None    # never fetched
    store.put_board_runs([_row("run_sub_a48ee1990d", SHA_OURS, True)], ts=1.0)
    LD.sync(ctx, store, _ledger({"n0103": (0.558, 4, "confirm")}), 2.0)
    assert LD.sync(ctx, store, None, 3.0) == []                     # the ledger failed to load: the stored estimate stays
    store.put_board_runs([_row("run_sub_a48ee1990d", SHA_OURS, False, "2026-11-10T22:00:00Z")], ts=4.0)
    assert LD.sync(ctx, store, None, 4.0) == []                     # no leader marked: keep what we know
    assert abs(LD.effective(ctx.cfg, store).labless.leader_median - 0.558) < 1e-9


def test_too_few_seeds_is_not_measurable_and_off_switch(ctx, store):
    ctx = _cfg(ctx)
    _node(store, "n0103", commit=SHA_OURS)
    store.put_board_runs([_row("run_sub_a48ee1990d", SHA_OURS, True)], ts=1.0)
    out = LD.sync(ctx, store, _ledger({"n0103": (0.60, 2, "screen")}), 1.0)
    assert "too few seeds" in out[0] and LD.effective(ctx.cfg, store).labless.leader_median == 0.5471
    off = _cfg(ctx, leader_rebase=False)
    store.set_state(LD.K_LEADER, {"run_id": "r", "node": "n0103", "estimate": 0.6})
    assert LD.sync(off, store, _ledger({"n0103": (0.6, 4, "confirm")}), 2.0) == []
    assert LD.effective(off.cfg, store) is off.cfg
    # the tick passes the SPEC's section: its re-based cfg returns to the configured number when switched off
    rebased = LD.effective(ctx.cfg, store)
    assert LD.effective(rebased, store, off.cfg.labless).labless.leader_median == 0.5471


def test_no_configured_leader_median_is_never_rebased(ctx, store):
    ctx = _cfg(ctx)
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, labless=dataclasses.replace(
        ctx.cfg.labless, leader_median=None)))
    _node(store, "n0103", commit=SHA_OURS)
    store.put_board_runs([_row("run_sub_a48ee1990d", SHA_OURS, True)], ts=1.0)
    assert LD.sync(ctx, store, _ledger({"n0103": (0.558, 4, "confirm")}), 1.0) == []
    assert LD.effective(ctx.cfg, store).labless.leader_median is None


def test_tick_hook_installs_the_rebase_on_the_shared_context(ctx, store):
    from loop.tick import Tick
    ctx = _cfg(ctx)
    _node(store, "n0103", commit=SHA_OURS)
    store.put_board_runs([_row("run_sub_a48ee1990d", SHA_OURS, True)], ts=1.0)
    fake = SimpleNamespace(ctx=ctx, store=store, _spec_labless=ctx.cfg.labless)
    lab = R.IncumbentLabless(ctx, store, InlineExecutor(), clock=lambda: 1000.0, submit=lambda *a: "sub-x")
    out = Tick._leader(fake, _ledger({"n0103": (0.558, 4, "confirm")}), 5.0)
    assert out and abs(ctx.cfg.labless.leader_median - 0.558) < 1e-9
    assert lab.ctx.cfg.labless.leader_median == ctx.cfg.labless.leader_median     # every component reads ctx.cfg


def _done(ctx, store, nid, commit):
    n = _node(store, nid, commit=commit)
    R.RunBook(ctx, store, _Pending()).launch(n, Stage.CONFIRM, 100_000)
    store.update_run(f"{nid}:confirm:100000", status=RunStatus.DONE)
    return n


def test_no_resubmit_of_a_node_already_on_the_board(ctx, store):
    ctx = _cfg(ctx)
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, labless=dataclasses.replace(
        ctx.cfg.labless, leader_median=None)))                      # no bar: only the dedupe can stop it
    calls = []
    lab = R.IncumbentLabless(ctx, store, InlineExecutor(), clock=lambda: 1000.0,
                             submit=lambda *a: calls.append(a) or "sub-1")
    _done(ctx, store, "n0103", SHA_OURS)
    store.put_board_runs([_row("run_sub_a48ee1990d", SHA_OURS, True)], ts=1.0)   # submitted by hand, outside the loop
    out = lab.step("n0103")
    assert out == ["labless: incumbent n0103 not submitted: already on Labless (board run run_sub_a48ee1990d "
                   f"(same commit {SHA_OURS[:10]}))"]
    assert lab.step("n0103") == [] and calls == []                  # said once, never sent
    assert store.events("labless_duplicate")


def test_no_resubmit_of_an_identical_commit_and_a_fresh_node_still_goes(ctx, store):
    ctx = _cfg(ctx)
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, labless=dataclasses.replace(
        ctx.cfg.labless, leader_median=None)))
    calls = []
    lab = R.IncumbentLabless(ctx, store, InlineExecutor(), clock=lambda: 1000.0,
                             submit=lambda *a: calls.append(a) or "sub-2")
    _node(store, "n1", commit=SHA_OURS)
    store.set_state(R.K_LABLESS, {"n1": {"submission_id": "sub-1"}})
    _done(ctx, store, "n2", SHA_OURS)                               # same code under another id
    assert "already on Labless (submission sub-1 of n1" in lab.step("n2")[0] and calls == []
    _done(ctx, store, "n3", SHA_KID)
    assert lab.step("n3") == ["labless: submitting incumbent n3 from n3:confirm:100000"] and len(calls) == 1
    off = R.IncumbentLabless(_cfg(ctx, dedupe_submitted=False, leader_median=None), store, InlineExecutor(),
                             clock=lambda: 1000.0, submit=lambda *a: calls.append(a) or "sub-3")
    _done(ctx, store, "n4", SHA_OURS)
    assert off.step("n4")[0].startswith("labless: submitting incumbent n4")     # the switch


def test_child_of_our_leader_must_beat_the_rebased_bar(ctx, store):
    """After the re-base a child of the leader node needs leader estimate + margin on its OWN fresh seeds."""
    from loop.core import bar as BR
    ctx = _cfg(ctx)
    _node(store, "n0103", commit=SHA_OURS)
    store.put_board_runs([_row("run_sub_a48ee1990d", SHA_OURS, True)], ts=1.0)
    led = _ledger({"n0103": (0.558, 5, "confirm"), "kid": (0.556, 5, "confirm")})
    LD.sync(ctx, store, led, 1.0)
    eff = LD.effective(ctx.cfg, store).labless
    b_stale = BR.beat(led, BR.bar_from(led, ctx.cfg.labless), "kid")
    b_new = BR.beat(led, BR.bar_from(led, eff), "kid")
    assert b_stale.beats and not b_new.beats                        # the stale .5521 bar would have sent a worse child


# ------------------------------------------------------------------ re-root onto our validated leader
def _fake_tick(ctx, store, dry=False):
    from loop.tick import Tick
    t = SimpleNamespace(ctx=ctx, store=store, dry=dry, _blocks=None)
    t.blocks = lambda: list(store.seed_blocks())
    t._update_block = lambda *a, **k: Tick._update_block(t, *a, **k)
    t._put_block = lambda *a, **k: Tick._put_block(t, *a, **k)
    return t, Tick


def _statuses(inc):
    from loop.tick import INCUMBENT
    return {inc: INCUMBENT}


def _validated(rid, commit, **kw):
    return _row(rid, commit, True, validation="leader", raw_validation="validated", **kw)


def test_validated_leader_becomes_the_incumbent_through_the_replay(ctx, store):
    from loop.core import status as ST
    ctx = _cfg(ctx)
    _node(store, "n0103", commit=SHA_OURS)
    store.put_board_runs([_validated("run_sub_a48ee1990d", SHA_OURS)], ts=1.0)
    led = _ledger({"n0012": (0.552, 6, "screen"), "n0103": (0.558, 6, "screen")})
    led = dataclasses.replace(led, nodes={**led.nodes, "n0103": _n("n0103", "n0012")})
    from loop.core.seeds import SeedBlock                           # n0012 became the incumbent by an earlier TEST pass
    store.put_seed_block(SeedBlock("t0", "test", (1, 2), ("n0012", "root"), "test n0012", 5.0, 6.0, (
        {"node": "n0012", "vs": "root", "kind": "pass", "e": 20.0, "n": 2, "mean": 0.012, "reason": "x", "harm": [],
         "sd": 0.003, "legacy": False},)))
    assert ST.chain(led, list(store.seed_blocks()), None)[-1] == "n0012"
    t, Tick = _fake_tick(ctx, store)
    acts: list[str] = []
    assert Tick._leader_reroot(t, led, _statuses("n0012"), 50.0, acts) is True
    assert any("LEADER RE-ROOT over n0012" in a for a in acts)
    blocks = list(store.seed_blocks())
    assert ST.chain(led, blocks, None)[-1] == "n0103"               # status.replay now switches to it
    assert any(b.purpose == "remint n0103" and b.pool == "select" and b.seeds for b in blocks)
    assert store.get_state(LD.K_REROOT)["previous"] == "n0012" and store.events("leader_reroot")
    acts2: list[str] = []                                           # once per leader run, even if the incumbent moves on
    assert Tick._leader_reroot(t, led, _statuses("n0200"), 60.0, acts2) is False and acts2 == []


def test_leader_reroot_noop_when_incumbent_or_ancestor_or_unvalidated_or_not_ours(ctx, store):
    ctx = _cfg(ctx)
    _node(store, "n0103", commit=SHA_OURS)
    led = _ledger({"n0103": (0.558, 4, "confirm"), "kid": (0.56, 4, "confirm")})
    led = dataclasses.replace(led, nodes={**led.nodes, "kid": _n("kid", "n0103")})
    store.put_board_runs([_row("run_sub_a48ee1990d", SHA_OURS, True, validation="unvalidated",
                               raw_validation="unvalidated")], ts=1.0)
    assert LD.reroot_plan(ctx, store, led, "root") is None          # not validated
    store.put_board_runs([_validated("run_sub_a48ee1990d", SHA_OURS)], ts=2.0)
    assert LD.reroot_plan(ctx, store, led, "n0103") is None         # already the incumbent
    assert LD.reroot_plan(ctx, store, led, "kid") is None           # an ancestor of the incumbent
    assert LD.reroot_plan(ctx, store, led, "root")[1] == "n0103"
    assert LD.reroot_plan(_cfg(ctx, leader_reroot=False), store, led, "root") is None      # the switch
    store.put_board_runs([_row("run_sub_a48ee1990d", SHA_OURS, False, "2026-11-11T00:00:00Z"),
                          _validated("run_sub_z", SHA_OTHER, updated="2026-11-11T00:00:00Z")], ts=3.0)
    assert LD.reroot_plan(ctx, store, led, "root") is None          # another contributor leads


def test_leader_reroot_closes_its_own_open_test_block_and_dry_writes_nothing(ctx, store):
    from loop.core.seeds import SeedBlock
    from loop.core import status as ST
    ctx = _cfg(ctx)
    _node(store, "n0103", commit=SHA_OURS)
    store.put_board_runs([_validated("run_sub_a48ee1990d", SHA_OURS)], ts=1.0)
    store.put_seed_block(SeedBlock("t1", "test", (100, 101, 102, 103), ("n0103", "root"), "test n0103", 10.0))
    led = _ledger({"n0103": (0.558, 4, "confirm")})
    t, Tick = _fake_tick(ctx, store, dry=True)
    acts: list[str] = []
    assert Tick._leader_reroot(t, led, _statuses("root"), 50.0, acts) is False and "would LEADER RE-ROOT" in acts[0]
    assert next(b for b in store.seed_blocks() if b.id == "t1").open
    t.dry = False
    assert Tick._leader_reroot(t, led, _statuses("root"), 50.0, [])
    b = next(b for b in store.seed_blocks() if b.id == "t1")
    assert not b.open and b.verdicts[0]["tier"] == "leader" and b.verdicts[0]["vs"] == "root"
    assert ST.chain(led, list(store.seed_blocks()), None)[-1] == "n0103"
    assert any(x.purpose == "remint n0103" and x.seeds == (100, 101, 102, 103) for x in store.seed_blocks())


def test_deficits_are_vs_the_incumbent_never_the_root():
    """Operator 11-10: an axis where the incumbent sits below the epoch root but no other node leads it is no deficit
    (the root is its ancestor; it was a deficit on any root seed count before); a node leading the incumbent is."""
    from loop.core import levers as LV
    from loop.core.data import AXES as AX
    a, b = AX[0], AX[1]
    nodes = {"root": SimpleNamespace(base="E", notes="", parent=None), "inc": SimpleNamespace(base="E", notes="", parent="root"),
             "x": SimpleNamespace(base="E", notes="", parent="root")}
    v = SimpleNamespace(led=SimpleNamespace(epoch="E", base_root="root", nodes=nodes), incumbent="inc",
                        noise=SimpleNamespace(sd=lambda _a: 0.001),
                        levels={"inc": {a: (0.50, 9), b: (0.50, 9)}, "root": {a: (0.60, 1), b: (0.50, 1)},
                                "x": {b: (0.60, 9)}})
    rows = {r["axis"]: r for r in LV.deficits(v, SimpleNamespace(get_state=lambda *_: {}), None)}
    assert rows[a]["weighted"] == 0.0 and rows[a]["vs_root"] < 0          # below root: context only
    assert rows[b]["weighted"] > 0 and rows[b]["best"][0] == "x"           # a lead over the incumbent: a deficit


def test_replay_never_switches_to_a_failed_graft(ctx, store):
    """Live 11-11: a frozen leader-tier PASS onto the failed rescue port n0091@d4b5a0a (n0103's commit, port_failed)
    made it the incumbent; replay skips it and the incumbent stays n0103 -- no DB edit."""
    from loop.core import status as ST
    from loop.core.seeds import SeedBlock
    led = _ledger({"n0103": (0.558, 4, "confirm")})
    bad = dataclasses.replace(_n("n0091@d4b5a0a", "n0103"), notes="port_of=n0091 moonshot_rescue=n0091 port_failed")
    led = dataclasses.replace(led, nodes={**led.nodes, "n0091@d4b5a0a": bad})
    rec = lambda c, vs, tier=None: {"node": c, "vs": vs, "kind": "pass", "e": 9.0, "n": 4, "mean": 0.01, "reason": "x",
                                     "harm": [], "sd": 0.003, "legacy": False, **({"tier": tier} if tier else {})}
    store.put_seed_block(SeedBlock("t1", "test", (1,), ("n0103", "root"), "test n0103", 1.0, 2.0, (rec("n0103", "root"),)))
    store.put_seed_block(SeedBlock("leader-x", "test", (), ("n0091@d4b5a0a", "n0103"), "leader re-root", 3.0, 4.0,
                                   (rec("n0091@d4b5a0a", "n0103", "leader"),)))
    assert ST.no_code(led, "n0091@d4b5a0a") and not ST.no_code(led, "n0103")
    assert ST.chain(led, list(store.seed_blocks()), None)[-1] == "n0103"


def test_our_node_prefers_the_built_original_over_a_port_copy_of_its_commit(ctx, store):
    """11-11: n0091@d4b5a0a (a port of n0091 onto n0103, not yet built: it carries n0103's commit) sorted before n0103
    and became 'our leader'. A port / a node whose commit equals its parent's loses to the built original; then the
    oldest created."""
    from loop.contracts import Node as CN, NodeStatus, Operator, Strategy

    def put(nid, commit, parent, ts, notes=""):
        store.put_node(CN(id=nid, task_id=store.task_id, commit=commit, parent_id=parent, strategy=Strategy.EXPLORE,
                          operator=Operator.DRAFT, family="f", mechanism_tag="m", diff_hash=f"h{nid}",
                          status=NodeStatus.QUEUED, created_ts=ts, notes=notes))
    put("n0012", SHA_OTHER, "root", 1.0)
    put("n0103", SHA_OURS, "n0012", 2.0)
    put("n0091@d4b5a0a", SHA_OURS, "n0103", 3.0, "port_of=n0091 moonshot_rescue=n0091")     # pending, not failed yet
    put("n0050", SHA_OURS, "n0103", 0.5)                                                    # unbuilt copy, older
    row = _row("run_sub_a48ee1990d", SHA_OURS, True)
    assert LD.our_node(ctx, store, row, sids={}) == "n0103"
    put("n0200", SHA_OURS, "n0012", 0.1)                                                    # a second built one, older
    assert LD.our_node(ctx, store, row, sids={}) == "n0200"
    assert LD.our_node(ctx, store, row, sids={"a48ee1990d": "n0091@d4b5a0a"}) == "n0091@d4b5a0a"   # a sid match rules
