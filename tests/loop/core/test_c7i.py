"""v3-c7i (24-h measurement 10-14): budget allocation by measured yield per GPU-h (loop.core.alloc) -- I1 the kind
bandit (yield / GPU-h, robust softmax, code-made units credited, program share a decaying ceiling), I2 the GPU split +
brief throttle + backlog age-out, I3 mechanism clusters, I4 pre-session dedup + check_idea, I5 the LLM crossover
brief only for conflicting code grafts, I6 fork-and-flush pause + test-only credit, I7 session tooling + max_turns
labelling, I8 knob keys resolved at intake, I9 engagement is no quality signal + early kill."""

from __future__ import annotations

import dataclasses
import json
import sys
import types
from collections import Counter
from types import SimpleNamespace

import pytest

from loop.contracts import Candidate, HandoffNote, HypothesisKind, HypothesisRecord, Node, NodeStatus, \
    Operator, ROOT_NODE_ID, Strategy
from loop.core import acquire as A
from loop.core import alloc as AL
from loop.core import data as D
from loop.core import fnf as FN
from loop.core import ideation as ID
from loop.core import program as PG
from loop.spec import AllocCfg

from ..conftest import InlineExecutor
from .test_fnf import _Research, env  # noqa: F401  (the fixture)


# ---------------------------------------------------------------------------------------------- toy ledgers
def _n(nid, parent=ROOT_NODE_ID, notes="", lane="explore", ts=100.0, base="B"):
    return D.Node(nid, parent, None, "explore", "f", lane, ROOT_NODE_ID, ts, "screening", notes, base)


def _r(nid, gpu=1.0, stage="screen", seed=1, status="done"):
    return D.Run(f"{nid}:{stage}:{seed}", nid, seed, stage, status, gpu, 0.0, 0.0, "", 0)


def _led(nodes, runs=(), obs=()):
    root = D.Node(ROOT_NODE_ID, None, None, "baseline", "", "baseline", ROOT_NODE_ID, 0.0, "baseline", "", "B")
    return SimpleNamespace(nodes={ROOT_NODE_ID: root, **{n.id: n for n in nodes}}, runs=tuple(runs), obs=tuple(obs),
                           epoch="B")


def _bank():
    """explore: 8 sessions, all negative (one -.028 outlier), 2 GPU-h each; a wave of 4 points (one at a lucky +.009)
    whose x* is +.004;
    a code-made pair node +.003; a program node -.02 whose untagged dose child is -.01 on top."""
    nodes, runs, bel = [], [], {}
    for i in range(8):
        nodes.append(_n(f"e{i}", notes=f"request=r{i} brief=explore"))
        runs.append(_r(f"e{i}", 2.0))
        bel[f"e{i}"] = (-0.028 if i == 0 else -0.002, 2)
    for i in range(4):                 # a point's own (lucky-seed) increment never values the wave: x* does (merge)
        nodes.append(_n(f"w{i}", notes=f"wave=w001 point={i} block=0 seed=7", lane="factorial"))
        runs.append(_r(f"w{i}", 1.0))
        bel[f"w{i}"] = (0.009 if i == 0 else -0.001, 1)
    nodes.append(_n("wx", notes="factorial_of=w001 prior_m=0.003", lane="factorial"))
    runs.append(_r("wx", 1.0))
    bel["wx"] = (0.004, 2)
    nodes.append(_n("p1", notes="pair_of=e1,e2 pairs", lane="crossover"))
    runs.append(_r("p1", 2.0))
    bel["p1"] = (0.003, 2)
    nodes.append(_n("g1", notes="request=r9 brief=program"))
    nodes.append(_n("g2", parent="g1", notes="ladder=g1 dose=x=1", lane="factorial"))
    runs += [_r("g1", 3.0), _r("g2", 1.0), _r("g1", 5.0, stage="confirm")]       # a test run is not charged
    bel["g1"], bel["g2"] = (-0.02, 2), (-0.01, 2)
    return _led(nodes, runs), bel


# ---------------------------------------------------------------------------------------------- I1 the kind bandit
def test_units_credit_code_made_nodes_and_untagged_descendants_to_their_kind():
    led, bel = _bank()
    us = AL.units(led, bel)
    assert us["wave:w001"]["kind"] == "wave" and us["wave:w001"]["nodes"] == 5 and us["wave:w001"]["best"] == 0.004
    assert us["wave:w001"]["node"] == "wx" and us["wave:w001"]["gpu_h"] == 5.0
    assert us["p1"]["kind"] == "pair" and us["p1"]["best"] == 0.003              # a code-picked crossover is credited
    assert us["g1"]["kind"] == "program" and us["g1"]["nodes"] == 2               # the ladder dose: what it led to
    assert us["g1"]["best"] == pytest.approx(-0.02) and us["g1"]["gpu_h"] == 4.0  # path value; test runs not charged
    assert AL.groups(led)["w2"] == "code" and AL.groups(led)["e3"] == "llm" and AL.groups(led)["g2"] == "llm"


def test_kind_rates_rank_on_yield_per_gpu_h_and_a_negative_outlier_no_longer_flattens_them():
    led, bel = _bank()
    tab, pooled = AL.kind_rates(led, bel, n0=1.0)
    assert min(tab["wave"]["rate"], tab["pair"]["rate"]) > max(tab["program"]["rate"], tab["explore"]["rate"])
    assert tab["explore"]["gain"] == 0.0 and tab["explore"]["gpu_h"] == 16.0      # a loss is floored at 0, not summed
    w = AL.softmax(tab, ["explore", "program", "wave", "pair"], pooled["temp"])
    assert w["wave"] + w["pair"] > 0.6 and w["explore"] < 0.15                     # not flat (.19-.21 before)
    old = ID.kind_weights({"explore": [b[0] for k, b in bel.items() if k.startswith("e")],
                           "program": [-0.02]}, 5.0, ("explore", "program"))
    assert abs(old["explore"]["w"] - old["program"]["w"]) < 0.25                   # the old exp((y - top) / sd)
    assert pooled["llm_rate"] < pooled["code_rate"]


def test_program_share_is_a_ceiling_that_decays_toward_the_floor_while_program_lags():
    tab = {"program": {"rate": 0.0}, "explore": {"rate": 2e-5}}
    assert AL.program_share(0.25, 0.05, tab, {"llm_rate": 2e-5}) == pytest.approx(0.05)
    assert AL.program_share(0.25, 0.05, {"program": {"rate": 1e-5}}, {"llm_rate": 2e-5}) == pytest.approx(0.15)
    assert AL.program_share(0.25, 0.05, {"program": {"rate": 5e-5}}, {"llm_rate": 2e-5}) == pytest.approx(0.25)
    assert AL.program_share(0.25, 0.05, tab, {"llm_rate": 0.0}) == 0.25            # nothing measured yet


def test_ideation_kind_table_uses_rates_and_the_program_share_follows_them(env, monkeypatch):
    ctx, store, led, ide = env
    v = ide.view(led)
    tab = ide.kind_table(v)
    assert {"rate", "w", "gpu_h"} <= set(tab["explore"]) and "wave" not in tab and set(tab) == set(ID.KINDS)
    assert sum(r["w"] for r in tab.values()) == pytest.approx(1.0)                # the brief kinds only
    assert store.events("kind_weights")                                           # logged when it changes
    monkeypatch.setattr(ID, "target_gap", lambda v: 0.006)
    monkeypatch.setattr(AL, "program_share", lambda c, f, t, p: 0.07)
    assert ide.program_share(v) == 0.07
    ide.ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, alloc=AllocCfg(kind_rate=False)))
    assert ide.program_share(v) == ctx.cfg.ideation.program_share                  # off: the operator's quota


# ---------------------------------------------------------------------------------------------- I2 split + backlog
def test_gpu_split_share_follows_the_measured_rates_within_floor_and_cap():
    tab = {"explore": {"gain": 0.001, "gpu_h": 100.0}, "wave": {"gain": 0.01, "gpu_h": 50.0}}
    sp = AL.gpu_split(tab, 0.4, 0.1)
    assert sp["llm_rate"] == pytest.approx(1e-5) and sp["code_rate"] == pytest.approx(2e-4) and sp["share"] == 0.1
    assert AL.gpu_split({"explore": {"gain": 0.01, "gpu_h": 10.0}, "wave": {"gain": 0.01, "gpu_h": 10.0}},
                        0.4, 0.1)["share"] == 0.4                                   # never above the cap
    assert AL.gpu_split({}, 0.4, 0.1)["share"] == 0.4


def test_rank_soft_caps_llm_unit_runs_so_cheap_code_work_wins_but_idle_slots_are_lifted():
    opts = [A.Option("screen", f"L{i}", "explore", 0.004, 1e-5, 1e-6, 3.0, 3) for i in range(3)] + \
           [A.Option("screen", "C1", "factorial", 0.002, 1e-5, 1e-6, 3.0, 3)]
    groups = {"L0": "llm", "L1": "llm", "L2": "llm", "C1": "code"}
    out = A.rank(opts, 6, 0, soft={"group:llm": 3}, groups=groups)
    assert [a.node for a in out] == ["L0", "C1"]                     # one LLM screen (3 runs), then the code one
    out = A.rank(opts, 12, 0, soft={"group:llm": 3}, groups=groups)
    assert {a.node for a in out} == {"L0", "L1", "L2", "C1"}          # slots nothing else wants: the cap is lifted
    assert [a.node for a in A.rank(opts, 6, 0)] == ["L0", "L1"]       # without groups: as before


def test_backlog_throttles_briefs_and_ages_out_low_prior_nodes_without_deleting(store):
    c = AllocCfg(backlog_max=3, backlog_age_h=6.0, drop_age_h=24.0, backlog_keep=2, drop_max=10)
    now = 100 * 3600.0
    nodes = [_n(f"q{i}", ts=now - h * 3600) for i, h in enumerate([1, 2, 30, 40, 50])]
    led = _led(nodes)
    statuses = {n.id: SimpleNamespace(value="queued") for n in nodes}
    bl = AL.backlog(led, statuses, now)
    assert bl["n"] == 5 and bl["median_age_h"] == pytest.approx(30.0)
    assert "5 queued > 3" in AL.throttle(c, bl) and "median queued age 30.0 h" in AL.throttle(c, bl)
    assert AL.throttle(c, AL.backlog(led, {}, now)) == ""
    rates = {"q0": 1e-5, "q1": 5e-6, "q2": 9e-6, "q3": 1e-7, "q4": 2e-7}         # q4 has no screen option below
    rows = AL.age_out(c, bl, {k: x for k, x in rates.items() if k != "q4"})
    assert [r[0] for r in rows] == ["q3"]            # q2 is in the best 2 by rate; q0 / q1 are young; q4 has no option
    for i, n in enumerate(nodes):
        store.put_node(Node(id=n.id, task_id=store.task_id, commit=f"c{i}", parent_id=ROOT_NODE_ID,
                            strategy=Strategy.EXPLORE, operator=Operator.DRAFT, family="f", mechanism_tag="t",
                            diff_hash=f"h{i}", status=NodeStatus.QUEUED, created_ts=n.created_ts))
    acts = AL.drop(store, rows, c.backlog_keep, now)
    got = store.get_node("q3")
    assert got.status is NodeStatus.REJECTED and got.reject_reason.startswith("dropped_backlog")
    assert store.events("backlog_dropped")[0]["node_id"] == "q3" and "dropped q3" in acts[0]


def test_ideate_starts_no_brief_while_the_backlog_throttles(env, monkeypatch):
    ctx, store, led, ide = env
    monkeypatch.setattr(FN, "SHARE", 0.0)
    monkeypatch.setattr(AL, "backlog", lambda led, st, now, sc=None: {"n": 90, "median_age_h": 12.0, "ages": {}})
    tick = SimpleNamespace(research=_Research(4), pool=InlineExecutor(), dry=False)
    acts = ide.ideate(tick, [SimpleNamespace(kind="idea", lane="explore")] * 3, led, {})
    assert tick.research.started == [] and any("briefs throttled" in a for a in acts)
    assert store.events("brief_throttle")[-1]["queued"] == 90
    monkeypatch.setattr(AL, "backlog", lambda led, st, now, sc=None: {"n": 2, "median_age_h": 1.0, "ages": {}})
    ide.ideate(tick, [SimpleNamespace(kind="idea", lane="explore")] * 2, led, {})
    assert len(tick.research.started) == 2 and store.get_state(AL.K_THROTTLE) == ""


def test_a_throttled_backlog_leaves_a_one_session_trickle_of_literature_and_bold(env, monkeypatch):
    """11-09 nanobrain: ~33 nodes queued unscreened while research kept starting Opus sessions. alloc.backlog_trickle 1:
    while throttled at most ONE session is in flight, an explore-lane slot only, briefed literature then bold."""
    ctx, store, led, ide = env
    monkeypatch.setattr(FN, "SHARE", 0.0)
    monkeypatch.setattr(ide.ctx, "cfg", dataclasses.replace(ide.ctx.cfg, alloc=dataclasses.replace(
        AL.cfg_of(ide.ctx), backlog_max=12, backlog_trickle=1)))
    monkeypatch.setattr(AL, "backlog", lambda led, st, now, sc=None: {"n": 33, "median_age_h": 2.0, "ages": {}})
    tick = SimpleNamespace(research=_Research(4), pool=InlineExecutor(), dry=False)
    tick.research.inflight = tick.research.started
    kinds = []
    monkeypatch.setattr(ide, "_plan", (lambda orig: lambda lane, v, kind, *a, **k: kinds.append(kind) or orig(
        lane, v, kind, *a, **k))(ide._plan))
    slots = [SimpleNamespace(kind="idea", lane="crossover")] + [SimpleNamespace(kind="idea", lane="explore")] * 3
    acts = ide.ideate(tick, slots, led, {})
    assert len(tick.research.started) == 1 and kinds == ["literature"], acts
    assert any("1 trickle (literature/bold" in a for a in acts) and "33 queued > " in store.get_state(AL.K_THROTTLE)
    ide.ideate(tick, slots, led, {})
    assert len(tick.research.started) == 1                     # one in flight: the trickle is full
    tick.research.started.clear()
    ide.ideate(tick, slots, led, {})
    assert len(tick.research.started) == 1 and kinds == ["literature", "bold"]
    assert AL.trickle_room(AllocCfg(), "x", 0) == 0 and AL.trickle_room(AllocCfg(), "", 9) == -1   # default: hard stop


def test_program_and_local_confirm_nodes_do_not_count_toward_the_backlog_max():
    off = AllocCfg(backlog_max=3, backlog_age_h=0.0)                 # default (nanopath): every QUEUED node counts
    assert "5 queued > 3" in AL.throttle(off, {"n": 5, "n_program": 2, "median_age_h": 0.0})
    c = AllocCfg(backlog_max=3, backlog_age_h=0.0, backlog_exclude_program=True)
    assert AL.throttle(c, {"n": 5, "n_program": 2, "median_age_h": 0.0}) == ""
    assert "4 queued > 3" in AL.throttle(c, {"n": 5, "n_program": 1, "median_age_h": 0.0})
    assert A.program_child(SimpleNamespace(notes="brief=local_confirm local_confirm=local-confirm-c1 sha=60df476"))


def test_tick_age_out_drops_after_rank_and_removes_their_actions(store, ctx):
    from loop import tick as TK
    now = 100 * 3600.0
    nodes = [_n(f"q{i}", ts=now - (12 if i == 0 else 48) * 3600) for i in range(3)]   # q0 is the keeper, under drop_age_h
    led = _led(nodes)
    for i, n in enumerate(nodes):
        store.put_node(Node(id=n.id, task_id=store.task_id, commit=f"c{i}", parent_id=ROOT_NODE_ID,
                            strategy=Strategy.EXPLORE, operator=Operator.DRAFT, family="f", mechanism_tag="t",
                            diff_hash=f"h{i}", status=NodeStatus.QUEUED, created_ts=n.created_ts))
    opts = [A.Option("screen", n.id, "explore", 0.004 - i * 0.002, 1e-5, 1e-6, 3.0, 3) for i, n in enumerate(nodes)]
    ctx2 = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, alloc=AllocCfg(backlog_keep=1)))
    fake = SimpleNamespace(core=SimpleNamespace(last_opts=opts, A=A, held=()), ctx=ctx2, store=store, dry=False)
    actions = [SimpleNamespace(node="q2", kind="screen"), SimpleNamespace(node="q0", kind="screen")]
    acts = TK.Tick._age_out(fake, led, {n.id: SimpleNamespace(value="queued") for n in nodes}, actions, now)
    assert len(acts) == 2 and [a.node for a in actions] == ["q0"]
    assert store.get_node("q0").status is NodeStatus.QUEUED and store.get_node("q2").status is NodeStatus.REJECTED
    store.update_node("q1", status=NodeStatus.QUEUED)
    fake.dry = True                                                       # shadow: listed, nothing written
    got = TK.Tick._age_out(fake, led, {"q0": SimpleNamespace(value="queued"), "q1": SimpleNamespace(value="queued")},
                           [], now)
    assert got and got[0].startswith("backlog: would drop 1") and store.get_node("q1").status is NodeStatus.QUEUED


# ---------------------------------------------------------------------------------------------- I3 clusters
def _put(store, nid, tag, family="objective", ts=0.0, notes=""):
    store.put_node(Node(id=nid, task_id=store.task_id, commit=f"c-{nid}", parent_id=ROOT_NODE_ID,
                        strategy=Strategy.EXPLORE, operator=Operator.DRAFT, family=family, mechanism_tag=tag,
                        diff_hash=f"h-{nid}", status=NodeStatus.SCREENED_OUT, created_ts=ts,
                        notes=notes or f"request=r-{nid} brief=explore"))
    return _n(nid, notes=notes or f"request=r-{nid} brief=explore", ts=ts)


def test_clusters_mark_saturated_negative_and_boost_under_visited_positive(store, ctx):
    # the share-cap rule alone: the set-1 family block (negative_family, tests/loop/test_set1_harness.py) is off here
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, alloc=dataclasses.replace(
        ctx.cfg.alloc, family_block=False)))
    nodes, bel = [], {}
    for i in range(10):
        nodes.append(_put(store, f"k{i}", f"kde_uniformity_margin_v{i}", "representation", ts=i))
        bel[f"k{i}"] = (-0.004, 2)
    for i in range(3):
        nodes.append(_put(store, f"s{i}", f"stem_layerwise_lr_decay_{i}", "optimization", ts=20 + i))
        bel[f"s{i}"] = (0.003, 2)
    for i, tag in enumerate(["hed_stain_jitter", "patient_cap_volume", "register_tokens", "drop_path_ramp",
                             "teacher_temp_warmup", "global_crop_scale", "ibot_mask_ratio", "koleo_weight",
                             "ema_momentum_cosine", "grad_clip_adaptive", "tissue_filter_floor", "local_crop_count"]):
        nodes.append(_put(store, f"o{i}", tag, "misc", ts=30 + i))       # unrelated mechanisms (the IDF's corpus)
        bel[f"o{i}"] = (0.0, 1)
    led = _led(nodes)
    cls = AL.clusters(ctx, store, led, bel)
    kde = next(c for c in cls if "k0" in c.members)
    stem = next(c for c in cls if "s0" in c.members)
    assert set(kde.members) >= {f"k{i}" for i in range(10)} and kde.flag == "saturated-negative"
    assert kde.n == 10 and kde.mean == pytest.approx(-0.004) and kde.shrunk < 0
    assert stem.flag == "boost" and stem.shrunk > 0
    text = AL.cluster_text(cls, 5)
    assert "SATURATED-NEGATIVE" in text and "UNDER-VISITED POSITIVE" in text and "| 10 | -0.0040 |" in text
    AL.remember(cls, store, led, 25)
    rule, fix = AL.cluster_rule(ctx, "representation", "kde_uniformity_margin_stronger", "a stronger kde margin")
    assert rule == "saturated_cluster" and kde.id in fix and "differs from" in fix
    assert AL.cluster_rule(ctx, "optimization", "stem_layerwise_lr_decay_more", "x")[0] == ""
    ok = "[differs from: " + "; ".join(f"{m}: it changes the kernel bandwidth schedule entirely" for m in kde.sample) + "]"
    assert AL.cluster_rule(ctx, "representation", "kde_uniformity_margin_stronger", ok, ok)[0] == ""
    ctx2 = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, alloc=AllocCfg(cluster_cap_share=0.95, family_block=False)))
    assert AL.cluster_rule(ctx2, "representation", "kde_uniformity_margin_stronger", "x")[0] == ""   # under its cap


def test_every_brief_carries_the_cluster_table_and_the_nearest_dead_mechanisms(env, monkeypatch):
    ctx, store, led, ide = env
    v = ide.view(led)
    monkeypatch.setattr(AL, "dedup_text", lambda ctx, store, focus, k: "## Dead / in-flight mechanisms nearest X")
    ide._cls = (ID._view_token(v), [AL.Cluster("C1", ["a"], {}, label="kde/margin", category="objective", n=9,
                                               mean=-0.004, flag="saturated-negative")])
    b = ide.brief_for("explore", led, v, "explore")
    assert "## Mechanism clusters" in b.text and "SATURATED-NEGATIVE" in b.text
    assert "## Dead / in-flight mechanisms nearest X" in b.text
    assert "## Dead / in-flight mechanisms nearest X" in ide.brief_for("explore", led, v, "explore", branch="1.1").text


# ---------------------------------------------------------------------------------------------- I4 dedup + check_idea
def test_dedup_rows_rank_dead_and_live_mechanisms_by_similarity_to_the_focus(store, ctx):
    _put(store, "d1", "kde_uniformity_margin")                           # SCREENED_OUT: dead
    _put(store, "d2", "stain_jitter_hed")
    rows = AL.dedup_rows(ctx, store, "a kde uniformity margin on the cls token", 5)
    assert rows[0][0] == "d1" and rows[0][2] == "dead"
    assert "check_idea" in AL.dedup_text(ctx, store, "kde uniformity margin", 5)
    assert AL.dedup_rows(ctx, store, "", 5) == []


def test_check_idea_runs_the_mechanism_rules_before_any_edit(tmp_path):
    from loop.agents import researcher as RS
    seen = []
    wt = SimpleNamespace(path=str(tmp_path), base_commit="b")
    tools = RS.WorktreeTools(None, wt, writable=("*",), locked=(),
                             checker=lambda **kw: seen.append(kw) or ("REJECT dead_mechanism: tried in n0001"
                                                                     if "kde" in kw["mechanism_tag"] else "OK"))
    out = tools("check_idea", {"mechanism_tag": "kde_margin", "family": "objective", "claim": "c"})
    assert out.startswith("REJECT dead_mechanism") and "nothing was built yet" in out
    assert seen[0]["mechanism_only"] is True
    assert tools("check_idea", {"mechanism_tag": "stem_lr"}).startswith("OK")
    assert tools("check_idea", {}).startswith("ERROR")
    assert any(t["name"] == "check_idea" for t in RS.TOOL_DEFS)


# ---------------------------------------------------------------------------------------------- I5 crossover grafts
class _Adapter:
    def __init__(self, conflict=()):
        self.conflict, self.calls = set(conflict), []

    def graft_port(self, parent, commit, onto, name):
        self.calls.append((parent, commit, onto))
        if commit in self.conflict:
            return "", "conflict: train.py"
        return f"{onto}+{commit}", "applied"


def _pairs_env(store, monkeypatch, pairs):
    from loop.core import pairs as PR
    for nid in ("inc", "a", "b", "c", "d"):
        store.put_node(Node(id=nid, task_id=store.task_id, commit=f"sha-{nid}", parent_id=ROOT_NODE_ID if nid != "inc"
                            else None, strategy=Strategy.EXPLORE, operator=Operator.DRAFT, family="objective",
                            mechanism_tag=nid, diff_hash=f"h-{nid}", status=NodeStatus.SCREENING))
    store.put_node(Node(id=ROOT_NODE_ID, task_id=store.task_id, commit="sha-root", parent_id=None,
                        strategy=Strategy.BASELINE, operator=Operator.DRAFT, family="", mechanism_tag="",
                        diff_hash="h-root", status=NodeStatus.BASELINE))
    monkeypatch.setattr(PR, "next_pair", lambda view, mark=False, exclude=(): next(
        (p for p in pairs if PR.pair_key(p[0], p[1]) not in set(exclude)), None))
    return SimpleNamespace(incumbent="inc", led=None, post=None)


def test_a_code_picked_pair_is_grafted_by_code_and_only_a_conflict_goes_to_the_llm(store, ctx, monkeypatch):
    from loop.core import pairs as PR
    view = _pairs_env(store, monkeypatch, [("a", "b", "why1"), ("c", "d", "why2")])
    ad = _Adapter(conflict={"sha-d"})
    acts = AL.graft_pairs(ctx, store, ad, view, {}, 10.0, new_id=lambda: "n9000")
    node = store.get_node("n9000")
    assert node.status is NodeStatus.QUEUED and "pair_of=a,b" in node.notes and AL.GRAFT_TAG in node.notes
    assert node.commit == "sha-inc+sha-a+sha-b" and node.parent_id == "inc"
    assert PR.pair_key("a", "b") in store.get_state(PR.K_USED)
    assert PR.pair_key("c", "d") in AL.flagged(store) and any("CONFLICT" in a for a in acts)
    assert AL.code_kind(node.notes)[0] == "pair"                      # credited to the code-made pair kind
    got = AL.conflict_pair(view, store)
    assert got[:2] == ("c", "d") and "conflicted" in got[2]
    n_calls = len(ad.calls)
    assert AL.graft_pairs(ctx, store, ad, view, {}, 11.0, dry=True) == []   # both handled: nothing left to try
    assert len(ad.calls) == n_calls


def test_the_llm_crossover_brief_falls_back_to_explore_unless_a_conflicting_pair_exists(env, monkeypatch):
    ctx, store, led, ide = env
    monkeypatch.setattr(FN, "SHARE", 0.0)
    monkeypatch.setattr(AL, "conflict_pair", lambda view, store, exclude=(), tries=24: None)
    tick = SimpleNamespace(research=_Research(4), pool=InlineExecutor(), dry=False)
    acts = ide.ideate(tick, [SimpleNamespace(kind="idea", lane="crossover")], led, {})
    assert [r.brief_kind for r in tick.research.started] == ["explore"]       # no LLM stacking: the planner's
    other = next(n for n in ide.view(led).beliefs if n not in (ide.view(led).incumbent, ROOT_NODE_ID))
    monkeypatch.setattr(AL, "conflict_pair", lambda view, store, exclude=(), tries=24:
                        (view.incumbent, other, "conflicted graft"))
    ide.ideate(tick, [SimpleNamespace(kind="idea", lane="crossover")], led, {})
    assert [r.brief_kind for r in tick.research.started] == ["explore", "crossover"]
    assert "conflicted graft" in tick.research.started[1].brief


def test_xover_near_dup_rate_is_measured_from_the_intake_log(store):
    store.log("research_started", request_id="r1", brief_kind="crossover")
    store.log("research_started", request_id="r2", brief_kind="crossover")
    store.log("research_started", request_id="r3", brief_kind="explore")
    store.log("intake", request_id="r1", reason="near_dup_signature: x")
    store.log("intake", request_id="r2", reason="")
    store.log("intake", request_id="r3", reason="near_dup: y")
    got = AL.xover_near_dup(store)
    assert got["all"] == {"sessions": 2, "near_dup": 1, "rate": 0.5}


# ---------------------------------------------------------------------------------------------- I6 fork-and-flush
def test_fnf_pauses_after_closed_rounds_without_a_positive_test_and_resumes_on_evidence():
    led = SimpleNamespace(nodes={"b1": _n("b1", notes="request=r branch=1.1"), "c1": _n("c1", notes="request=s")},
                          runs=(_r("b1"), _r("c1")), obs=(), epoch="B")
    st = {"rounds": [{"nodes": {"1.1": ["b1"]}, "control": ["c1"], "ts": 0, "closed": 10},
                     {"nodes": {"2.1": []}, "control": [], "ts": 10, "closed": 20}]}
    v = SimpleNamespace(led=led, judged=[])
    assert "2 closed rounds" in AL.fnf_pause(v, st, 2)
    assert AL.fnf_pause(v, st, 3) == ""                                   # not enough rounds yet
    v.judged = [SimpleNamespace(node="b1", verdict=SimpleNamespace(mean=0.003))]
    assert AL.fnf_pause(v, st, 2) == ""                                   # a branch node passed: evidence changed
    assert FN.SCREEN_CREDIT is False                                      # branches credited on test gains only


def test_a_paused_fnf_closes_the_open_round_but_forks_no_new_one(env, monkeypatch):
    ctx, store, led, ide = env
    v = ide.view(led)
    monkeypatch.setattr(FN, "trigger", lambda *a: "cadence")
    monkeypatch.setattr(AL, "fnf_pause", lambda v, st, k: "2 closed rounds, no positive test")
    store.set_state(FN.K_FNF, {"round": 1, "open": {"id": 1, "ts": 0.0, "gpu0": 0.0, "cap": 21.0,
                                                   "branches": ["1.1"], "incumbent": v.incumbent,
                                                   "trigger": "start", "sessions": {}}, "rounds": []})
    acts = ide._fnf(SimpleNamespace(dry=False), v)
    st = FN.state(store)
    assert st["open"] is None and len(st["rounds"]) == 1 and any("paused" in a for a in acts)
    assert store.events("fnf_paused")


# ---------------------------------------------------------------------------------------------- I7 sessions
def test_bash_accepts_description_and_read_aliases_reach_read_file(tmp_path):
    from loop.agents import researcher as RS
    (tmp_path / "a.txt").write_text("hello\nworld\n")
    tools = RS.WorktreeTools(None, SimpleNamespace(path=str(tmp_path), base_commit="b"), writable=("*",), locked=())
    assert tools("bash", {"command": "echo hi", "description": "say hi"}).startswith("exit=0")
    assert "hello" in tools("Read", {"file_path": "a.txt"}) and "world" in tools("read_file", {"path": "a.txt"})
    assert "unknown tool" in tools("nope", {})


def test_a_max_turns_ending_is_labelled_max_turns_not_backend_unavailable(monkeypatch):
    from loop.agents import llm as llm_mod
    fake_llm = types.SimpleNamespace(OPUS_TIMEOUT=0, sdk_session=lambda prompt, **kw: {
        "ok": False, "error_kind": "max_turns", "error": "32 turns", "turns": 32})
    fake_agent = types.SimpleNamespace(sdk_tool_server=lambda defs, gate, name: ({}, [], []))
    monkeypatch.setitem(sys.modules, "llm", fake_llm)
    monkeypatch.setitem(sys.modules, "agent", types.SimpleNamespace(llm=fake_agent))
    monkeypatch.setitem(sys.modules, "agent.llm", fake_agent)
    run = llm_mod.SdkLLMClient(types.SimpleNamespace(llm_roles={})).run_agent(
        "researcher", system="s", prompt="p", cwd=".", tools=[], executor=lambda n, a: "",
        finish_schema={"type": "object"}, max_turns=30, timeout_s=60.0)
    assert run.error.startswith("max_turns") and "backend_unavailable" not in run.error and run.turns == 32
    fake_llm.sdk_session = lambda prompt, **kw: {"ok": False, "error_kind": "network", "error": "x"}
    run = llm_mod.SdkLLMClient(types.SimpleNamespace(llm_roles={})).run_agent(
        "researcher", system="s", prompt="p", cwd=".", tools=[], executor=lambda n, a: "",
        finish_schema={"type": "object"}, max_turns=30, timeout_s=60.0)
    assert run.error.startswith("backend_unavailable")                  # a real outage keeps its label


def test_researcher_max_turns_default_is_raised():
    from loop.spec import ResearcherCfg
    assert ResearcherCfg().max_turns >= 40


# ---------------------------------------------------------------------------------------------- I8 knob keys
CFG = {"dino.gram_patch_weight": 0.5, "dino.gram_cls_weight": 0.2, "train.lr": 1e-4, "model.lr": 3.0,
       "data.views": 4}


def test_bare_knob_keys_resolve_by_unique_suffix_and_ambiguity_is_listed():
    plan = {"knobs": [{"key": "views", "lo": 2, "hi": 8}],
            "components": [{"knobs": [{"key": "gram_patch_weight", "lo": 0.1, "hi": 2}],
                            "off": {"gram_patch_weight": 0}},
                           {"knobs": [{"key": "lr", "lo": 1e-5, "hi": 1e-3}], "off": {}}]}
    out, problems = PG.resolve_plan_keys(plan, CFG)
    assert out["knobs"][0]["key"] == "data.views"
    assert out["components"][0]["knobs"][0]["key"] == "dino.gram_patch_weight"
    assert out["components"][0]["off"] == {"dino.gram_patch_weight": 0}
    assert len(problems) == 1 and "ambiguous" in problems[0] and "model.lr" in problems[0] and "train.lr" in problems[0]
    assert plan["knobs"][0]["key"] == "views"                            # the input is not mutated
    _, p2 = PG.resolve_plan_keys({"knobs": [{"key": "gram_patch_wieght"}]}, CFG)
    assert "not a config key" in p2[0] and "dino.gram_patch_weight" in p2[0]
    assert PG.resolve_plan_keys(plan, {}) == (plan, [])


def _cand(commit="sha1", cfg_diff=None):
    h = HypothesisRecord(id="", task_id="t", kind=HypothesisKind.TRIAL, claim="c", mechanism_tag="gram",
                         family="objective", predictions={}, falsification="")
    return Candidate("req-v3-00001", "t", commit, "base", "diff", "dh", ("configs/main.yaml",), dict(cfg_diff or {}),
                     h, HandoffNote("", "t", "req-v3-00001", "", "", ""))


def test_intake_rewrites_resolvable_keys_and_rejects_an_unresolvable_one(env):
    ctx, store, led, ide = env
    ide.adapter = SimpleNamespace(config_at=lambda commit: CFG)
    plan, why = ide._resolve_keys(_cand(), {"components": [{"knobs": [{"key": "gram_cls_weight", "lo": 0.05,
                                                                       "hi": 1}]}]})
    assert why == "" and plan["components"][0]["knobs"][0]["key"] == "dino.gram_cls_weight"
    plan, why = ide._resolve_keys(_cand(), {"knobs": [{"key": "lr", "lo": 1e-5, "hi": 1e-3}]})
    assert "ambiguous" in why
    ide.adapter = SimpleNamespace(config_at=lambda commit: {})          # unreadable: never a verdict
    assert ide._resolve_keys(_cand(), {"knobs": [{"key": "lr"}]})[1] == ""


def test_the_finish_bounces_an_unresolvable_knob_key_and_rewrites_a_bare_one(tmp_path):
    import yaml
    from loop.agents import researcher as RS
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "main.yaml").write_text(yaml.safe_dump(
        {"dino": {"gram_patch_weight": 0.5}, "train": {"lr": 1e-4}, "model": {"lr": 3.0}}))
    tools = RS.WorktreeTools(None, SimpleNamespace(path=str(tmp_path), base_commit="b"), writable=("*",), locked=(),
                             config_rel="configs/main.yaml")
    args = {"components": [{"knobs": [{"key": "gram_patch_weight", "lo": 0.1, "hi": 2}]}]}
    assert tools._keys_problem(args) == "" and args["components"][0]["knobs"][0]["key"] == "dino.gram_patch_weight"
    assert "ambiguous" in tools._keys_problem({"knobs": [{"key": "lr", "lo": 1e-5, "hi": 1e-3}]})


def test_an_n2734_type_ladder_failure_is_retried_and_an_unreadable_config_is_never_booked(store):
    store.set_state(PG.K_LADDERS, {"nX": "-: gram_patch_weight absent from the config; gram_cls_weight absent from "
                                         "the config", "nY": "-: foo absent from the config"})
    store.set_state("config_diff:nX", {"dino.gram_patch_weight": 0.5, "dino.gram_cls_weight": 0.2})
    cleared = PG.retry_bare_key_failures(store, lambda nid: CFG if nid == "nX" else {"a.b": 1})
    assert cleared == ["nX"] and "nX" not in store.get_state(PG.K_LADDERS) and "nY" in store.get_state(PG.K_LADDERS)
    import inspect
    src = inspect.getsource(PG.ladders)
    assert "if unreadable:" in src and src.index("if unreadable:") < src.index('book[nid] = "-: "')


def test_the_researcher_prompt_requires_full_dotted_keys_and_check_idea_first():
    from pathlib import Path
    text = (Path(ID.PROMPTS) / "researcher.md").read_text()
    assert "FULL dotted config key" in text and "check_idea" in text and "BEFORE any edit" in text


# ---------------------------------------------------------------------------------------------- I9 diagnostics
def _obs(node, seed, y):
    from loop.core.data import AXES
    return D.Obs(f"{node}:{seed}", node, seed, "screen", "select", {a: y for a in AXES}, y, 1.0, float(seed), 0)


def test_a_reversed_first_seed_with_a_negative_paired_delta_is_an_early_kill():
    from loop.core.data import AXES
    w = {a: 1.0 / len(AXES) for a in AXES}
    led = _led([_n("x"), _n("y"), _n("z")], obs=[_obs(ROOT_NODE_ID, 1, 0.60), _obs("x", 1, 0.59),
                                                  _obs("y", 1, 0.61), _obs("z", 1, 0.58)])
    E = ID.Engagement
    engs = [E("x", "q", 1, "reversed", -2.5, 1), E("y", "q", 1, "reversed", -2.5, 1),
            E("z", "q", 1, "reversed", -2.5, 2), E(ROOT_NODE_ID, "q", 1, "engaged", 2.0, 1)]
    assert AL.first_seed_delta(led, "x", w) == pytest.approx(-0.01)
    got = AL.early_kills(engs, led, w, {})
    assert [g[0] for g in got] == ["x"]               # y: a positive first seed; z: decided on two seeds
    assert AL.early_kills(engs, led, w, {"x": {}}) == []
    # v3 review K9: z must be < -2 and the delta below -sd_d (one paired seed's noise)
    assert AL.early_kills([E("x", "q", 1, "reversed", -1.5, 1)], led, w, {}) == []
    assert AL.early_kills(engs, led, w, {}, sd_d=0.02) == []
    assert [g[0] for g in AL.early_kills(engs, led, w, {}, sd_d=0.005)] == ["x"]
    later = _led([_n("x")], obs=[_obs(ROOT_NODE_ID, 1, 0.60), _obs("x", 1, 0.59), _obs(ROOT_NODE_ID, 2, 0.60),
                                 _obs("x", 2, 0.612)])                     # a 2nd paired seed: mean +.001
    assert AL.early_lifts(later, w, {"x": {}}) == [("x", pytest.approx(0.001), 2)]
    pr = AL.predictive(engs, {"x": (-0.001, 1), "y": (0.002, 1), "z": (-0.003, 2)})
    assert pr["reversed"] == (1, 3)


def test_rank_drops_seed_and_screen_options_of_early_killed_nodes(store, ctx):
    from loop import tick as TK
    core = TK.ModuleCore()
    core.configure(ctx.cfg, store=store)
    store.set_state(AL.K_EARLY, {"x": {"z": -2.0}})
    assert core._killed() == frozenset({"x"})
    core.alloc_cfg = AllocCfg(early_kill=False)
    assert core._killed() == frozenset()


def test_engagement_is_shown_as_no_quality_signal_and_levers_drop_the_engaged_count():
    import inspect
    from loop.core import levers as LV
    assert "engaged {e}/{m}" not in inspect.getsource(LV.diag_text)
    assert "NOT evidence" in inspect.getsource(ID.Ideation._engagement_lines)


# ---------------------------------------------------------------------------------------------- the meta report
def test_meta_section_reports_kind_weights_split_backlog_and_xover(store, ctx):
    led, bel = _bank()
    post = SimpleNamespace(beliefs=lambda: {n: (m, 1e-6, k) for n, (m, k) in bel.items()})
    store.set_state(AL.K_THROTTLE, "90 queued > 40")
    sec = AL.meta_section(ctx, store, led, post, 0.0, 10.0)
    assert sec["kinds"]["wave"]["rate"] > sec["kinds"]["explore"]["rate"] and "share" in sec["split"]
    text = "\n".join(AL.meta_text(sec))
    assert "kind yield per GPU-h" in text and "GPU split" in text and "90 queued > 40" in text
    assert "LLM crossover briefs: near_dup" in text and "check_idea" in text


def test_a_paused_fnf_line_and_a_waiting_graft_line_are_shown_once_an_hour(store):
    """Live 10-21: 'fnf: paused' 3325 lines and 'xover graft ... waits' 450 identical lines (one per tick)."""
    t = "fnf: paused (3 closed rounds)"
    assert AL.quiet(store, "k", t, 100.0) == [t]
    assert AL.quiet(store, "k", t, 160.0) == []                                     # unchanged: quiet
    assert AL.quiet(store, "k", t + "!", 200.0) == [t + "!"]                        # changed: shown
    assert AL.quiet(store, "k", t + "!", 200.0 + 3601) == [t + "!"]                 # an hour later: shown again


def test_a_graft_queued_for_days_no_longer_holds_an_open_slot(store):
    mk = lambda nid, ts: store.put_node(Node(id=nid, task_id=store.task_id, commit="c", parent_id=None,
                                             strategy=Strategy.CROSSOVER, operator=Operator.DRAFT, family="f",
                                             mechanism_tag=nid, diff_hash="h" + nid, status=NodeStatus.QUEUED,
                                             created_ts=ts, notes=f"pair_of=a,b {AL.GRAFT_TAG}"))
    now = 10 * 86400.0
    mk("n1", now - 3 * 86400.0)
    mk("n2", now - 3600.0)
    st = {"n1": NodeStatus.QUEUED, "n2": NodeStatus.QUEUED}
    assert AL.graft_open(store, st) == 2                                            # no age bound: both
    assert AL.graft_open(store, st, now, 48.0) == 1                                 # the 3-day-old one is stale
