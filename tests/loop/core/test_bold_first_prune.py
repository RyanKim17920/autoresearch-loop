"""11-09 nanobrain: bold screens first (``alloc.priority_categories``), stale small-tweak QUEUED nodes of flat
mechanism families pruned (``alloc.prune_stale_family``) and recorded in the tested-mechanism memory. Both are off
without their keys (nanopath unchanged)."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest

from loop.contracts import ROOT_NODE_ID, Node, NodeStatus, Operator, Strategy
from loop.core import acquire as A
from loop.core import alloc as AL
from loop.core import data as D
from loop.core import model as M
from loop.core import tested as TS
from loop.spec import AllocCfg, PruneStaleCfg, load_context

REPO = Path(__file__).resolve().parents[3]
H = 3600.0
NOW = 100 * H
Q = SimpleNamespace(value="queued")
SCR = SimpleNamespace(value="screening")


def _n(nid, parent=ROOT_NODE_ID, notes="", ts=NOW - 10 * H, strategy="explore"):
    return D.Node(nid, parent, None, strategy, "f", "explore", ROOT_NODE_ID, ts, "screening", notes, "B")


def _led(nodes):
    root = D.Node(ROOT_NODE_ID, None, None, "baseline", "", "baseline", ROOT_NODE_ID, 0.0, "baseline", "", "B")
    return SimpleNamespace(nodes={ROOT_NODE_ID: root, **{n.id: n for n in nodes}}, runs=(), obs=(), epoch="B")


def _put(store, nid, tag, *, status=NodeStatus.SCREENING, ts=0.0, notes="", family="objective", parent=ROOT_NODE_ID):
    notes = notes or f"request=r-{nid} brief=explore"
    store.put_node(Node(id=nid, task_id=store.task_id, commit=f"c-{nid}", parent_id=parent, strategy=Strategy.EXPLORE,
                        operator=Operator.DRAFT, family=family, mechanism_tag=tag, diff_hash=f"h-{nid}", status=status,
                        created_ts=ts, notes=notes))
    return _n(nid, parent=parent, notes=notes, ts=ts)


def _post(deltas):
    """A posterior stand-in: node -> raw paired composite delta (n seeds)."""
    return SimpleNamespace(est={k: {M.COMPOSITE: SimpleNamespace(delta=d, n=n, paired=n)} for k, (d, n) in deltas.items()})


def _pctx(ctx, **kw):
    al = dataclasses.replace(ctx.cfg.alloc, family_block=False, prune_stale_family=PruneStaleCfg(**kw),
                             priority_categories=("bold",))
    return dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, alloc=al))


def _world(store, *, best=0.002, child=None, extra=()):
    """A collection-temperature family: 4 measured members (2 seeds each, best ``best``), one QUEUED tweak of it 10 h
    old, a fresh one 1 h old, and 12 unrelated nodes (the IDF corpus). ``child``: a ladder child of t0 with that raw
    delta. ``extra``: (nid, tag, notes, strategy) further QUEUED nodes of the family."""
    nodes, bel, raw = [], {}, {}
    for i, g in enumerate([-0.001, 0.0, 0.001, best]):
        nodes.append(_put(store, f"t{i}", f"collection_temperature_sampling_v{i}", ts=i))
        bel[f"t{i}"], raw[f"t{i}"] = (g, 2), (g, 2)
    nodes.append(_put(store, "qold", "collection_temperature_sampling_v9", status=NodeStatus.QUEUED, ts=NOW - 10 * H))
    nodes.append(_put(store, "qnew", "collection_temperature_sampling_v8", status=NodeStatus.QUEUED, ts=NOW - 1 * H))
    for nid, tag, notes, strat in extra:
        _put(store, nid, tag, status=NodeStatus.QUEUED, ts=NOW - 10 * H, notes=notes)
        nodes.append(_n(nid, notes=notes, ts=NOW - 10 * H, strategy=strat))
    for i, tag in enumerate(["hed_stain_jitter", "patient_cap_volume", "register_tokens", "drop_path_ramp",
                             "teacher_temp_warmup", "global_crop_scale", "ibot_mask_ratio", "koleo_weight",
                             "ema_momentum_cosine", "grad_clip_adaptive", "tissue_filter_floor", "local_crop_count"]):
        nodes.append(_put(store, f"o{i}", tag, family="misc", ts=30 + i))
        bel[f"o{i}"], raw[f"o{i}"] = (0.0, 1), (0.0, 1)
    if child is not None:                              # a tuning-ladder child: no brief tag, so in no cluster itself
        _put(store, "t0k", "collection_temperature_sampling_v0_s1", notes="ladder=t0 struct_tune=other cell=x",
             parent="t0", ts=5)
        nodes.append(_n("t0k", parent="t0", notes="ladder=t0 struct_tune=other cell=x", ts=5))
        bel["t0k"], raw["t0k"] = (child, 3), (child, 3)
    led = _led(nodes)
    st = {n.id: Q if n.id.startswith("q") or n.id.startswith("x") else SCR for n in nodes}
    return led, st, bel, _post(raw)


# ---------------------------------------------------------------------------------------------- bold first (rank)
def _opts():
    return [A.Option("screen", "tweak", "explore", 0.004, 1e-5, 1e-6, 3.0, 3),
            A.Option("screen", "bold1", "explore", 0.001, 1e-5, 1e-6, 3.0, 3),
            A.Option("seed", "other", "explore", 0.003, 1e-5, 1e-6, 1.0, 1)]


def test_rank_puts_preferred_screens_first_and_leaves_other_kinds_in_place():
    base = A.rank(_opts(), 7, 0)
    assert [a.node for a in base] == [a.node for a in A.rank(_opts(), 7, 0, prefer=())]      # off = unchanged
    got = A.rank(_opts(), 7, 0, prefer={"bold1"})
    assert {a.node for a in got} == {a.node for a in base}
    scr = [a.node for a in got if a.kind == "screen"]
    assert scr == ["bold1", "tweak"] and [a.node for a in base if a.kind == "screen"] == ["tweak", "bold1"]
    # with room for ONE screen, the bold one takes it
    one = A.rank(_opts(), 3, 0, prefer={"bold1"})
    assert [a.node for a in one if a.kind == "screen"] == ["bold1"]
    assert [a.node for a in A.rank(_opts(), 3, 0) if a.kind == "screen"] == ["tweak"]


def test_rank_program_priority_still_goes_before_preferred_screens(monkeypatch):
    monkeypatch.setattr(A, "PROGRAM_ARM_SHARE", 0.5)
    opts = _opts() + [A.Option("screen", "arm", "explore", 0.0005, 1e-5, 1e-6, 3.0, 3)]
    got = A.rank(opts, 12, 0, priority={"arm"}, prefer={"bold1"})
    assert [a.node for a in got if a.kind == "screen"][:2] == ["arm", "bold1"]


def test_priority_nodes_match_brief_category_or_strategy_and_skip_program_children():
    nodes = [_n("b", notes="request=r brief=bold cat=objective"), _n("m", strategy="moonshot", notes="request=r brief=x"),
             _n("e", notes="request=r brief=explore bold_cat=pathology"), _n("s", notes="request=r brief=bold"),
             _n("arm", notes="request=r brief=bold program_arm=1")]
    led = _led(nodes)
    st = {"b": Q, "m": Q, "e": Q, "s": SCR, "arm": Q}
    assert AL.priority_nodes(AllocCfg(), led, st) == set()                          # default: off
    # 11-09: the bold brief's nodes carry strategy "moonshot" -- the two words alias each other
    assert AL.priority_nodes(AllocCfg(priority_categories=("bold",)), led, st) == {"b", "m"}
    assert AL.priority_nodes(AllocCfg(priority_categories=("moonshot",)), led, st) == {"b", "m"}
    assert AL.priority_nodes(AllocCfg(priority_categories=("bold", "moonshot")), led, st) == {"b", "m"}
    assert AL.priority_nodes(AllocCfg(priority_categories=("pathology",)), led, st) == {"e"}


def test_priority_nodes_include_struct_tune_children_of_a_bold_node_only():
    """11-09 live: n0104 / n0105 (``ladder=n0030 struct_tune=other``, strategy explore) tune the moonshot n0030."""
    nodes = [_n("bold", strategy="moonshot", notes="request=r brief=bold"), _n("tw", notes="request=r brief=explore"),
             _n("k1", parent="bold", notes="ladder=bold struct_tune=other cell=lr:hi"),
             _n("k2", parent="k1", notes="ladder=k1 struct_tune=other cell=wd:lo"),          # a nested ladder
             _n("k3", parent="tw", notes="ladder=tw struct_tune=other cell=lr:hi"),          # tunes a non-bold node
             _n("dose", parent="bold", notes="ladder=bold dose=x=1")]                          # a program dose: no
    st = {n.id: Q for n in nodes}
    st["bold"] = SCR
    got = AL.priority_nodes(AllocCfg(priority_categories=("bold",)), _led(nodes), st)
    assert got == {"k1", "k2"}
    assert AL.priority_nodes(AllocCfg(), _led(nodes), st) == set()


def test_rank_bold_screen_beats_non_llm_screens_under_the_llm_group_soft_cap():
    """11-09 17:05-17:11Z: 17 LLM runs live against a cap of 8 -> group:llm soft 0; the strict pass skipped every bold
    (LLM) screen while a non-LLM screen took the slots. With ``prefer`` the bold screen goes first."""
    opts = [A.Option("screen", "code1", "explore", 0.004, 1e-5, 1e-6, 3.0, 3),
            A.Option("screen", "bold1", "explore", 0.001, 1e-5, 1e-6, 3.0, 3)]
    groups = {"code1": "code", "bold1": "llm"}
    kw = dict(soft={"group:llm": 0}, groups=groups)
    assert [a.node for a in A.rank(opts, 3, 0, **kw)] == ["code1"]                     # the bug
    assert [a.node for a in A.rank(opts, 3, 0, prefer={"bold1"}, **kw)] == ["bold1"]
    # room for both: the soft cap still orders the rest as before
    assert [a.node for a in A.rank(opts, 6, 0, prefer={"bold1"}, **kw)] == ["bold1", "code1"]


def test_rank_keeps_seed_extensions_and_tests_ahead_of_preferred_screens():
    opts = [A.Option("screen", "bold1", "explore", 0.004, 1e-5, 1e-6, 3.0, 3),
            A.Option("seed", "inflight", "explore", 0.0005, 1e-5, 1e-6, 1.0, 1),
            A.Option("screen", "tweak", "explore", 0.003, 1e-5, 1e-6, 3.0, 3)]
    got = A.rank(opts, 4, 0, prefer={"bold1"}, soft={"group:llm": 0}, groups={"inflight": "llm", "bold1": "llm"})
    assert [a.node for a in got] == ["inflight", "bold1"]


def test_rank_a_preferred_screen_takes_the_last_slots_as_a_partial_before_a_smaller_other_screen():
    opts = [A.Option("screen", "small", "explore", 0.004, 1e-5, 1e-6, 2.0, 2),
            A.Option("screen", "bold1", "explore", 0.001, 1e-5, 1e-6, 3.0, 3)]
    assert [a.node for a in A.rank(opts, 2, 0)] == ["small"]
    got = A.rank(opts, 2, 0, prefer={"bold1"})
    assert [(a.node, a.runs) for a in got] == [("bold1", 2)] and "partial" in got[0].why


def test_rank_relax_pass_orders_seeds_then_preferred_screens(monkeypatch):
    monkeypatch.setattr(A, "FILL_DROP_BELOW", False)
    opts = [A.Option("screen", "tweak", "explore", 0.004, 1e-5, 1e-6, 3.0, 3, below=True),
            A.Option("screen", "bold1", "explore", 0.001, 1e-5, 1e-6, 3.0, 3, below=True),
            A.Option("seed", "s", "explore", 0.0001, 1e-5, 1e-6, 1.0, 1, below=True)]
    got = A.rank(opts, 4, 0, prefer={"bold1"})
    assert [a.node for a in got] == ["s", "bold1"] and all(a.pre for a in got)
    assert [a.node for a in A.rank(opts, 3, 0)] == ["tweak"]


def test_rank_a_stuck_reservation_is_released_in_tier_order():
    """Review: a discovery reservation its node cannot use (tag-capped) is released over A, B, C in order -- never to a
    tier-C screen ahead of an owed test or a bold screen."""
    kw = dict(reserve=(3, {"disc"}), tags={"disc": "T"}, tag_left={"T": 0.0})
    d = A.Option("screen", "disc", "explore", 0.006, 1e-5, 1e-6, 3.0, 3)
    t = A.Option("test", "t1", "explore", 0.005, 1e-5, 1e-6, 3.0, 3)
    b = A.Option("screen", "bold1", "explore", 0.001, 1e-5, 1e-6, 3.0, 3)
    c = A.Option("screen", "code1", "explore", 0.004, 1e-5, 1e-6, 3.0, 3)
    assert [a.node for a in A.rank([d, t, b, c], 3, 0, **kw)] == [a.node for a in A.rank([d, t, b, c], 3, 0,
                                                                                          prefer={"bold1"}, **kw)]
    assert [a.node for a in A.rank([d, b, c], 3, 0, prefer={"bold1"}, **kw)] == ["bold1"]


def test_rank_a_preferred_discovery_screen_may_partial_into_its_own_reservation():
    opts = [A.Option("screen", "code1", "explore", 0.004, 1e-5, 1e-6, 2.0, 2),
            A.Option("screen", "bold1", "explore", 0.001, 1e-5, 1e-6, 3.0, 3)]
    got = A.rank(opts, 2, 0, prefer={"bold1"}, reserve=(2, {"bold1"}))
    assert [(a.node, a.runs) for a in got] == [("bold1", 2)]


def test_rank_program_overflow_stays_ahead_of_preferred_screens(monkeypatch):
    monkeypatch.setattr(A, "PROGRAM_ARM_SHARE", 0.3)
    opts = [A.Option("screen", "arm1", "explore", 0.001, 1e-5, 1e-6, 3.0, 3),
            A.Option("screen", "arm2", "explore", 0.002, 1e-5, 1e-6, 3.0, 3),
            A.Option("screen", "bold1", "explore", 0.004, 1e-5, 1e-6, 3.0, 3)]
    got = [a.node for a in A.rank(opts, 6, 0, priority={"arm1", "arm2"}, prefer={"bold1"})]
    assert sorted(got) == ["arm1", "arm2"]


def test_idle_fill_screens_preferred_queued_nodes_first(sample_db):
    from .test_acquire_floors import _new_epoch, _state
    from .test_epochs import _add_node
    from loop.core import status as ST
    con, inc, t = _new_epoch(sample_db)
    for q in ("q1", "q2", "q3"):
        _add_node(con, q, "p-" + inc, t + 5, commit=q)
    con.commit()
    con.close()
    led, blocks, noise = _state(sample_db)
    st, post = ST.derive(led, blocks, noise), A.Posterior(led, blocks)
    c = lambda acts: [a.node for a in acts if a.why.startswith("idle (c)")]          # noqa: E731
    plain = c(A.idle_fill(led, blocks, st, inc, post, 200, set(), A.Costs()))
    assert set(plain) == {"q1", "q2", "q3"}
    pref = c(A.idle_fill(led, blocks, st, inc, post, 200, set(), A.Costs(), prefer={"q3"}))
    assert pref[0] == "q3" and set(pref) == set(plain)
    one = A.idle_fill(led, blocks, st, inc, post, A.Costs().idea_runs, {n for n in led.nodes if n not in plain},
                      A.Costs(), prefer={"q3"})
    assert [a.node for a in one] == ["q3"] and "priority category" in one[0].why


# ---------------------------------------------------------------------------------------------- stale-family prune
def test_stale_family_prunes_only_the_old_queued_tweak_of_a_flat_family(store, ctx):
    led, st, bel, post = _world(store)
    pc = _pctx(ctx)
    assert AL.stale_family(ctx, store, led, st, post, bel, NOW) == []               # no key: off (nanopath)
    rows = AL.stale_family(pc, store, led, st, post, bel, NOW)
    assert [r["node"] for r in rows] == ["qold"]                                   # qnew is 1 h old < 6 h
    assert rows[0]["measured"] == 4 and rows[0]["best"] == ("t3", pytest.approx(0.002))
    assert AL.stale_family(pc, store, led, st, post, bel, NOW, exempt={"qold"}) == []
    off = dataclasses.replace(pc, cfg=dataclasses.replace(pc.cfg, alloc=dataclasses.replace(
        pc.cfg.alloc, prune_stale_family=PruneStaleCfg(enabled=False))))
    assert AL.stale_family(off, store, led, st, post, bel, NOW) == []


def test_a_family_with_a_member_or_a_tuned_child_above_the_need_is_never_pruned(store, ctx, tmp_path):
    led, st, bel, post = _world(store, best=0.006)
    assert AL.stale_family(_pctx(ctx), store, led, st, post, bel, NOW) == []
    from loop.store import SqliteStore
    s2 = SqliteStore(tmp_path / "b.db", ctx.task_id, weights=ctx.cfg.weights)
    try:
        led, st, bel, post = _world(s2, child=0.0035)          # t0 (-.001) + its tuned child (+.0035): path +.0025
        assert [r["node"] for r in AL.stale_family(_pctx(ctx), s2, led, st, post, bel, NOW)] == ["qold"]
    finally:
        s2.close()
    s3 = SqliteStore(tmp_path / "c.db", ctx.task_id, weights=ctx.cfg.weights)
    try:
        led, st, bel, post = _world(s3, child=0.006)           # path +.005 > .004: the family is not dead
        assert AL.stale_family(_pctx(ctx), s3, led, st, post, bel, NOW) == []
    finally:
        s3.close()


def test_too_few_measured_members_or_single_seed_members_do_not_prune(store, ctx):
    led, st, bel, post = _world(store)
    assert AL.stale_family(_pctx(ctx, min_measured=5), store, led, st, post, bel, NOW) == []
    one = {k: ((m, 1) if k.startswith("t") else (m, n)) for k, (m, n) in bel.items()}
    assert AL.stale_family(_pctx(ctx), store, led, st, post, one, NOW) == []          # min_seeds 2


def test_confirm_root_audit_bold_program_and_ladder_nodes_are_never_pruned(store, ctx):
    tag = "collection_temperature_sampling_v7"
    extra = [("xconf", tag, "request=r brief=local_confirm local_confirm=lc-1 sha=1", "explore"),
             ("xaudit", tag, "request=r brief=explore root_audit=1", "explore"),
             ("xbold", tag, "request=r brief=bold", "moonshot"),
             ("xarm", tag, "request=r brief=program program_arm=1", "explore"),
             ("xrep", tag, "request=r brief=explore root_replicate=1", "explore"),
             ("xbank", tag, "request=r brief=explore bank=nb-recon-3", "explore"),
             ("xbr", tag, "request=r brief=explore branch=1.2", "explore"),
             ("xalt", tag, "request=r brief=explore arm=alt_base alt_of=t1", "explore"),
             ("xretry", tag, "request=r brief=explore retry_of=t2", "explore"),
             ("xbf", tag, "request=r brief=explore backfill_of=t2", "explore"),
             ("xmix", tag, "request=r brief=explore mixed=2", "explore")]
    led, st, bel, post = _world(store, extra=extra)
    pc = _pctx(ctx)
    pc = dataclasses.replace(pc, cfg=dataclasses.replace(pc.cfg, alloc=dataclasses.replace(pc.cfg.alloc,
                                                                                          keep_bank_nodes=True)))
    got = {r["node"] for r in AL.stale_family(pc, store, led, st, post, bel, NOW)}
    assert got == {"qold"}


def test_old_base_or_unpaired_evidence_never_prunes(store, ctx):
    led, st, bel, post = _world(store)
    old = dict(led.nodes)
    for k in ("t0", "t1"):                                   # only 2 of the 4 measured members on the current base
        old[k] = dataclasses.replace(old[k], base="OLD")
    led2 = SimpleNamespace(**{**led.__dict__, "nodes": old})
    assert AL.stale_family(_pctx(ctx), store, led2, st, post, bel, NOW) == []
    unpaired = SimpleNamespace(est={k: {M.COMPOSITE: SimpleNamespace(delta=v[M.COMPOSITE].delta, n=2, paired=0)}
                                    for k, v in post.est.items()})
    assert AL._raw_delta(unpaired, "t0") is None


def test_a_pruned_node_is_restored_once_its_family_beats_the_need(store, ctx):
    led, st, bel, post = _world(store)
    pc = _pctx(ctx)
    AL.prune_stale(store, AL.stale_family(pc, store, led, st, post, bel, NOW), NOW)
    assert store.get_node("qold").status is NodeStatus.REJECTED
    assert AL.revive_pruned(pc, store, led, post, bel, NOW) == []           # still flat
    bel["t3"] = (0.006, 3)
    post.est["t3"][M.COMPOSITE].delta = 0.006
    acts = AL.revive_pruned(pc, store, led, post, bel, NOW + H)
    n = store.get_node("qold")
    assert len(acts) == 1 and n.status is NodeStatus.QUEUED and not n.reject_reason and "pruned_family" not in n.notes
    assert AL.revive_pruned(ctx, store, led, post, bel, NOW) == []          # no key: off


def test_prune_marks_rejected_skips_launched_nodes_and_feeds_the_tested_memory(store, ctx):
    led, st, bel, post = _world(store)
    rows = AL.stale_family(_pctx(ctx), store, led, st, post, bel, NOW)
    acts = AL.prune_stale(store, rows + [{**rows[0], "node": "qnew"}], NOW)
    assert len(acts) == 2                                               # qnew stored QUEUED, no run: written too
    n = store.get_node("qold")
    assert n.status is NodeStatus.REJECTED and n.reject_reason.startswith(AL.PRUNED_REASON)
    assert "pruned_family=" in n.notes and not n.reject_reason.startswith("dropped_backlog")
    assert A.restore_aged(store) == []                                  # a family verdict is not a backlog drop
    mem = {e.id: e for e in TS.node_entries(store, TS.MEASURED)}
    assert "qold" in mem and mem["qold"].kind == "node" and AL.PRUNED_REASON in mem["qold"].result
    assert "qold" not in {e.id for e in TS.node_entries(store, TS.LIVE)}
    assert AL.prune_stale(store, rows, NOW) == []                       # already REJECTED: nothing again


def test_tick_prune_stale_dry_lists_and_never_touches_a_node_being_launched(store, ctx):
    from loop import tick as TK
    led, st, bel, post = _world(store)
    pc = _pctx(ctx)
    beliefs3 = {k: (m, 1e-6, n) for k, (m, n) in bel.items()}
    post.beliefs = lambda: beliefs3
    fake = SimpleNamespace(core=SimpleNamespace(held=()), ctx=pc, store=store, dry=True, post=post)
    got = TK.Tick._prune_stale(fake, led, st, [], NOW)
    assert got and got[0].startswith("backlog: would prune 1") and store.get_node("qold").status is NodeStatus.QUEUED
    fake.dry = False
    assert TK.Tick._prune_stale(fake, led, st, [SimpleNamespace(node="qold", kind="screen")], NOW) == []
    assert store.get_node("qold").status is NodeStatus.QUEUED
    assert len(TK.Tick._prune_stale(fake, led, st, [], NOW)) == 1
    assert store.get_node("qold").status is NodeStatus.REJECTED
    fake.ctx = ctx                                                       # no key: nothing read, nothing written
    assert TK.Tick._prune_stale(fake, led, st, [], NOW) == []


# ---------------------------------------------------------------------------------------------- yaml
def test_nanobrain_enables_the_keys_and_nanopath_keeps_the_defaults(tmp_path):
    nb = load_context(REPO / "tasks" / "nanobrain.yaml", state_dir=tmp_path / "nb", require_measured=False).cfg
    assert nb.alloc.priority_categories == ("bold",)
    assert nb.alloc.prune_stale_family == PruneStaleCfg(age_h=6.0, min_measured=3, max_best_delta=0.004)
    assert nb.ideation.struct_tune and nb.ideation.struct_tune_max == 4 and nb.ideation.struct_tune_flat_floor == -0.005
    np_ = load_context(REPO / "tasks" / "nanopath_v2.yaml", state_dir=tmp_path / "np").cfg
    assert np_.alloc.priority_categories == () and np_.alloc.prune_stale_family is None
    assert np_.ideation.struct_tune_flat_floor is None
