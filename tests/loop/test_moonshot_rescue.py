"""11-10: the MOONSHOT RESCUE (loop.core.rescue) -- catastrophe -> one researcher DEBUG session; a not-clearly-negative
bold node with <= 2 seeds -> ported onto the CURRENT incumbent and tuned by a short ladder (port + <= 4 rungs at 2 seeds
each, the best released); a cap by prior plausibility, a GPU share, a retro pass, the n0082 knob re-judge, and the
adversarial cases: queue flooding, double-enqueue across restarts, futility / close holds mid-ladder, seed pairing vs the
incumbent.

Operator 11-10: n0113 sparse_upcycled_moe_mlp -.052 / n0089 hemi_swap_detect -.046 were judged on one untuned seed, no
bug check; n0082 natinit_sincos_pos3d was rejected bold_knob_only untested."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from loop import cutover as CO
from loop.contracts import ROOT_NODE_ID, Node, NodeStatus, Operator, RunStatus, Strategy
from loop.core import acquire as AQ
from loop.core import bold as BOLD
from loop.core import data as CD
from loop.core import program as PG
from loop.core import rescue as RS
from loop.core import struct_tune as ST

EP = "B"
INC = "n10"
BOLD_NOTES = "request=r brief=bold"


# ---------------------------------------------------------------------------------------------- the world
@pytest.fixture
def rctx(ctx, monkeypatch):
    grid = {"grid": [{"key": "muon_lr_mult", "factor": 1.5}, {"key": "weight_decay", "factor": 2.0}]}
    ic = dataclasses.replace(ctx.cfg.ideation, struct_tune_flat_floor=-0.005, moonshot_rescue_gpu_share=10.0,
                             struct_tune_knobs={"backbone": grid, "other": {"fallback": grid}})
    return dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, panel_sd=0.005, ideation=ic))


def _with(ctx, **kw):
    return dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, ideation=dataclasses.replace(ctx.cfg.ideation, **kw)))


def _cd(nid, parent, notes=BOLD_NOTES, strategy="moonshot", port_of=""):
    return CD.Node(nid, parent, None, strategy, "architecture", "explore", ROOT_NODE_ID, 1.0, "screening", notes,
                   base=EP, port_of=port_of)


def _obs(nid, seed, panel):
    return CD.Obs(f"{nid}:screen:{seed}", nid, seed, "screen", "select", {}, panel, 1.0, 1.0, 0)


def _run(nid, seed, status="done"):
    return CD.Run(f"{nid}:screen:{seed}", nid, seed, "screen", status, 1.0, -1e6, -1e6, f"/out/{nid}/s{seed}", 0)


#: node -> (parent, notes, strategy, {seed: panel} done, [running seeds])
WORLD = {
    INC: (ROOT_NODE_ID, "request=r brief=explore", "explore", {1: 0.61, 2: 0.61, 3: 0.61}, []),
    "a": (ROOT_NODE_ID, BOLD_NOTES, "moonshot", {1: 0.598}, []),                  # -.002 on 1: rescue
    "b": (ROOT_NODE_ID, BOLD_NOTES, "moonshot", {1: 0.55}, []),                   # -.05: catastrophic -> debug
    "c": (ROOT_NODE_ID, BOLD_NOTES, "moonshot", {1: 0.588, 2: 0.588}, []),        # -.012 on 2: clearly negative
    "d": (ROOT_NODE_ID, BOLD_NOTES, "moonshot", {1: 0.6, 2: 0.6, 3: 0.6}, []),    # 3 seeds: the normal path
    "e": (ROOT_NODE_ID, BOLD_NOTES, "moonshot", {1: 0.6}, [2]),                   # a run in flight: not yet
    "x": (ROOT_NODE_ID, "request=r brief=explore", "explore", {1: 0.6}, []),      # not bold / structural
    "g": (INC, BOLD_NOTES, "moonshot", {1: 0.609}, []),                           # on the incumbent: struct_tune's
}


def _world(store, world=None, extra_nodes=(), extra_obs=(), extra_runs=()):
    world = dict(WORLD if world is None else world)
    store.ensure_root("r0")
    nodes, obs, runs = {ROOT_NODE_ID: CD.Node(ROOT_NODE_ID, None, None, "baseline", "", "baseline", ROOT_NODE_ID, 0.0,
                                                "baseline", "", base=EP)}, [], []
    for s in (1, 2, 3):
        obs.append(_obs(ROOT_NODE_ID, s, 0.60))
        runs.append(_run(ROOT_NODE_ID, s))
    for nid, (parent, notes, strat, done, live) in world.items():
        if store.get_node(nid) is None:
            store.put_node(Node(id=nid, task_id=store.task_id, commit=f"c-{nid}", parent_id=parent,
                                strategy=Strategy(strat), operator=Operator.DRAFT, family="architecture",
                                mechanism_tag=f"tag_{nid}", diff_hash=f"h{nid}", status=NodeStatus.SCREENING, notes=notes))
        nodes[nid] = _cd(nid, parent, notes, strat)
        for s, p in done.items():
            obs.append(_obs(nid, s, p))
            runs.append(_run(nid, s))
        runs += [_run(nid, s, "running") for s in live]
    for n in extra_nodes:
        nodes[n.id] = n
    led = CD.Ledger(nodes=nodes, runs=tuple(runs + list(extra_runs)), obs=tuple(obs + list(extra_obs)),
                    test_seeds=frozenset(), carriers=(), bases=((0.0, ROOT_NODE_ID, EP),))
    statuses = {n: ("incumbent" if n == INC else "screening") for n in nodes if n != ROOT_NODE_ID}
    return led, statuses


def _post(beliefs=None, prior=None):
    beliefs = dict(beliefs or {})
    prior = dict(prior or {})
    return SimpleNamespace(est={n: {} for n in beliefs}, by={INC: ()},
                           belief=lambda n: beliefs[n], shift=lambda n, t, to: (0.0, 0.0),
                           cell=lambda n: n, prior=SimpleNamespace(cell=lambda c: (prior.get(c, 0.0), 1e-5, 3)))


CFG = {"model_kwargs.rope_frac": 0.5, "model_kwargs.rope_scale": 1.0, "base_lr": 1e-3, "muon_lr_mult": 1.0,
       "weight_decay": 0.05}


def _adapter():
    built = []
    return SimpleNamespace(config_rel="config.yaml", config_at=lambda c: CFG, built=built,
                           dry_build=lambda commit, levels, knobs, diffs, name: built.append(dict(levels)) or
                           (f"sha-{len(built)}", ""),
                           graft_port=lambda src, old, onto, name: (f"graft-{name}", "applied"))


def _plan(store, nid, median=0.0, knobs=()):
    store.set_state(f"ideation:{nid}", {"prediction": {"median": median},
                                        "components": [{"knobs": [dict(k) for k in knobs]}] if knobs else []})


# ---------------------------------------------------------------------------------------------- 1. verdicts
def test_the_verdict_debugs_a_catastrophe_rescues_a_flat_node_and_closes_a_clearly_negative_one(rctx):
    v = lambda m, n, **k: RS.verdict(rctx, {"n": n, "mean": m}, **k)[0]       # noqa: E731
    assert v(-0.05, 1) == "debug"                         # < -4 x .005 (and < -.02)
    assert v(-0.021, 1) == "debug"                        # < -4 x .005
    assert v(-0.002, 1) == "rescue" and v(-0.009, 1) == "rescue"   # > -2 se (se .005 at 1 seed)
    assert v(-0.012, 2) == "negative"                     # se .0035: -.012 < -.0071
    assert v(None, 0, crashed=True) == "debug" and v(None, 0) == "wait"
    big = dataclasses.replace(rctx, cfg=dataclasses.replace(rctx.cfg, panel_sd=0.05))
    assert RS.verdict(big, {"n": 1, "mean": -0.03})[0] == "debug"         # the absolute floor (moonshot_cata_abs .02)


def test_candidates_are_bold_nodes_with_at_most_two_seeds_and_nothing_in_flight(rctx, store):
    led, st = _world(store)
    got = {c["node"]: c["verdict"] for c in RS.candidates(rctx, store, led, st, _post(), INC)}
    assert got == {"a": "rescue", "b": "debug", "c": "negative"}
    # d: 3 seeds (the normal path), e: a run in flight, x: not bold, g: on the incumbent (struct_tune ladders it)


def test_priority_is_prior_plausibility_never_the_one_seed_delta(rctx, store):
    w = {INC: WORLD[INC], "a": WORLD["a"], "h": (ROOT_NODE_ID, BOLD_NOTES, "moonshot", {1: 0.592}, []),
         "k": (ROOT_NODE_ID, BOLD_NOTES, "moonshot", {1: 0.601}, [])}
    led, st = _world(store, w)
    _plan(store, "h", median=0.004)                       # the worst 1-seed delta, the best expected effect
    _plan(store, "k", median=0.0)                         # the best 1-seed delta
    post = _post(prior={"a": 0.001})
    assert [c["node"] for c in RS.candidates(rctx, store, led, st, post, INC)] == ["h", "a", "k"]
    # tuned already by the GENERIC LR x wd ladder on its old parent: still rescued, after the untuned ones
    sbk = {"h": {"kind": "other", "children": ["hc"]}}
    store.set_state(ST.K_STRUCT, sbk)
    store.set_state("config_diff:hc", {"muon_lr_mult": 1.5})
    led, st = _world(store, w, extra_obs=[_obs("hc", 1, 0.6)])
    assert [c["node"] for c in RS.candidates(rctx, store, led, st, post, INC)] == ["a", "k", "h"]
    # tuned on its OWN knobs already: its ladder judged it, no rescue (11-11 review)
    store.set_state("config_diff:hc", {"model_kwargs.rope_frac": 0.25})
    assert [c["node"] for c in RS.candidates(rctx, store, led, st, post, INC)] == ["a", "k"]


# ---------------------------------------------------------------------------------------------- 2. the rescue ladder
def test_a_rescue_ports_onto_the_incumbent_and_its_ladder_is_the_declared_knobs_at_two_seeds_outside_the_cap(rctx, store):
    led, st = _world(store)
    _plan(store, "a", knobs=[{"key": "model_kwargs.rope_frac", "lo": 0.25, "hi": 1.0, "scale": "lin"},
                             {"key": "model_kwargs.rope_scale", "lo": 0.3, "hi": 3.0, "scale": "log"}])
    ad = _adapter()
    acts = RS.step(rctx, store, ad, led, _post(), st, 100.0)
    rec = RS.book(store)["a"]
    pid = CO.port_id("a", EP)
    assert rec["phase"] == "ported" and rec["port"] == pid and rec["onto"] == INC and rec["src"] == ROOT_NODE_ID
    port = store.get_node(pid)
    assert port.parent_id == INC and port.status is NodeStatus.QUEUED          # paired vs the INCUMBENT
    assert "moonshot_rescue=a" in port.notes and "ladder=a" in port.notes and "struct_retro=1" in port.notes
    assert AQ.program_child(port) and AQ.futility_exempt(port, 2)              # screened first, never futile at 2 seeds
    assert rec["spec"]["points"] == [{"model_kwargs.rope_frac": 0.25}, {"model_kwargs.rope_frac": 1.0},
                                     {"model_kwargs.rope_scale": 0.3}, {"model_kwargs.rope_scale": 3.0}]
    assert any("ported onto n10" in a for a in acts) and store.events("moonshot_rescue")
    # the port screens (1 seed, flat): struct_tune builds the rescue ladder, outside a FULL struct_ladder_max_open
    full = _with(rctx, struct_ladder_max_open=0)
    led2, st2 = _world(store, extra_nodes=[_cd(pid, INC, port.notes, "moonshot", port_of="a")],
                       extra_obs=[_obs(pid, 1, 0.6)], extra_runs=[_run(pid, 1)])
    PG.ladders(full, store, ad, led2, _post({pid: (-0.012, 1e-5)}), {**st2, pid: "screening"}, 200.0)
    sr = ST.book(store)[pid]
    assert sr["rescue"] and sr["seeds"] == 2 and len(sr["children"]) == 4 == sr["need"] and "rescue" in sr["kind"]
    assert ad.built == rec["spec"]["points"]                                   # its declared knobs, not the LR grid


def test_a_rescue_on_the_incumbent_is_tuned_in_place_and_replaces_struct_tunes_hopeless_skip(rctx, store):
    led, st = _world(store)
    store.set_state(ST.K_STRUCT, {"g": ST.SKIP_FIRST + " -.0060 hopeless"})
    RS.step(rctx, store, _adapter(), led, _post(), st, 100.0)
    rec = RS.book(store)["g"]
    assert rec["port"] == "g" and "moonshot_rescue=g" in store.get_node("g").notes
    assert "g" not in ST.book(store)                                           # the skip is gone: it is laddered now


def test_the_caps_never_flood_the_queue_and_a_restart_never_enqueues_twice(rctx, store):
    w = {INC: WORLD[INC], **{f"r{i}": (ROOT_NODE_ID, BOLD_NOTES, "moonshot", {1: 0.6}, []) for i in range(6)},
         **{f"z{i}": (ROOT_NODE_ID, BOLD_NOTES, "moonshot", {1: 0.5}, []) for i in range(4)}}
    led, st = _world(store, w)
    ad = _adapter()
    RS.step(rctx, store, ad, led, _post(), st, 100.0)
    bk = RS.book(store)
    assert sum(1 for r in bk.values() if isinstance(r, dict) and r["phase"] == "ported") == 3   # moonshot_rescue_max
    assert sum(1 for r in bk.values() if isinstance(r, dict) and r["phase"] == "debug") == 2    # moonshot_debug_max_open
    n_nodes = len(store.nodes())
    # a RESTART: a fresh ledger view of the same store -- nothing is re-booked, no second port
    led, st = _world(store, w)
    RS.step(rctx, store, ad, led, _post(), st, 200.0)
    assert RS.book(store) == bk and len(store.nodes()) == n_nodes
    assert CO.port_node(store, ad, led, "r0", 300.0, onto=INC, src=ROOT_NODE_ID).how == "exists"   # idempotent port


def test_no_new_rescue_while_the_gpu_share_is_full_and_the_rescue_nodes_then_wait(rctx, store, monkeypatch):
    led, st = _world(store)
    monkeypatch.setattr(RS, "share_full", lambda ctx, store, led: (True, 9, 9))
    RS.step(rctx, store, _adapter(), led, _post(), st, 100.0)
    bk = RS.book(store)
    assert "a" not in bk and bk["b"]["phase"] == "debug"                       # a debug costs no GPU
    assert isinstance(bk["c"], str) and bk["c"].startswith("-: clearly negative")


# ---------------------------------------------------------------------------------------------- 3. holds / futility
def _ported(rctx, store):
    led, st = _world(store)
    ad = _adapter()
    RS.step(rctx, store, ad, led, _post(), st, 100.0)
    pid = RS.book(store)["a"]["port"]
    port = store.get_node(pid)
    store.set_state(ST.K_STRUCT, {pid: {"kind": "other+rescue", "children": ["k1", "k2"], "ts": 100.0, "rescue": True,
                                        "seeds": 2, "need": 2}})
    return pid, port, ad


def test_rungs_are_held_at_two_seeds_and_the_original_no_longer_screens_on_its_old_parent(rctx, store, monkeypatch):
    pid, port, _ = _ported(rctx, store)
    monkeypatch.setattr(RS, "share_full", lambda ctx, store, led: (False, 1, 9))
    kids = [_cd("k1", pid, f"ladder={pid} struct_tune=other cell=x"), _cd("k2", pid, f"ladder={pid} struct_tune=other")]
    led, _ = _world(store, extra_nodes=[_cd(pid, INC, port.notes, port_of="a"), *kids],
                    extra_obs=[_obs(pid, 1, 0.6), _obs(pid, 2, 0.6), _obs("k1", 1, 0.6)],
                    extra_runs=[_run(pid, 1), _run(pid, 2), _run("k1", 1), _run("k1", 2, "running")])
    held = RS.hold(rctx, store, led)
    assert {"a", pid, "k1"} <= held and "k2" not in held                       # k2 has no seed yet
    monkeypatch.setattr(RS, "share_full", lambda ctx, store, led: (True, 9, 9))
    assert "k2" in RS.hold(rctx, store, led)                                   # the GPU share is full: it waits
    # futility: a rung below 3 done seeds is exempt; struct_tune.held keeps the original / port open (no close, no
    # negative cluster evidence, no early kill) while the ladder is undecided
    assert AQ.futility_exempt(kids[0], 2)
    owed = ST.held(rctx, store, led, now=150.0)
    assert "a" in owed and pid in owed


def test_the_decided_rescue_releases_only_its_best_point_in_the_incumbent_frame(rctx, store):
    pid, port, _ = _ported(rctx, store)
    kids = [_cd("k1", pid, f"ladder={pid} struct_tune=other"), _cd("k2", pid, f"ladder={pid} struct_tune=other")]
    led, st = _world(store, extra_nodes=[_cd(pid, INC, port.notes, port_of="a"), *kids],
                     extra_obs=[_obs(c, s, 0.6) for c in (pid, "k1", "k2") for s in (1, 2)],
                     extra_runs=[_run(c, s) for c in (pid, "k1", "k2") for s in (1, 2)])
    post = _post({pid: (-0.001, 1e-5), "k1": (0.004, 1e-5), "k2": (-0.002, 1e-5)})
    acts = RS.step(rctx, store, _adapter(), led, post, st, 200.0)
    rec = RS.book(store)["a"]
    assert rec["phase"] == "done" and rec["best"] == "k1" and rec["outcome"] == "decided"
    held = RS.hold(rctx, store, led)
    assert "k1" not in held and {pid, "k2", "a"} <= held                       # the best takes the normal screen-to-5
    assert "a" not in RS.owed(store) and any("best k1" in a for a in acts)
    ev = [e for e in store.events("moonshot_rescue") if e.get("phase") == "decided"]
    assert ev and ev[-1]["best"] == "k1"


def test_a_stuck_rescue_is_decided_on_what_it_has_after_the_hold(rctx, store):
    pid, port, _ = _ported(rctx, store)
    led, st = _world(store, extra_nodes=[_cd(pid, INC, port.notes, port_of="a")], extra_obs=[_obs(pid, 1, 0.6)],
                     extra_runs=[_run(pid, 1)])
    assert RS.decide(rctx, store, led, _post({pid: (0.0, 1e-5)}), "a", RS.book(store)["a"], INC, 200.0) is None
    late = 100.0 + 3600.0 * rctx.cfg.ideation.moonshot_rescue_hold_h + 1
    got = RS.decide(rctx, store, led, _post({pid: (0.0, 1e-5)}), "a", RS.book(store)["a"], INC, late)
    assert got["outcome"] == "stale" and got["best"] == pid


# ---------------------------------------------------------------------------------------------- 4. debug sessions
class _Research:
    def __init__(self, free=4):
        self.inflight, self.requeue, self.started, self._free = {}, [], [], free

    def free(self):
        return self._free - len(self.started)

    def start(self, req, snap, lane=""):
        self.started.append((req, lane))
        self.inflight[req.id] = req


def _tick(rctx, store, research, statuses, ts=1000.0):
    return SimpleNamespace(ctx=rctx, store=store, research=research, statuses=statuses, adapter=_adapter(), dry=False,
                           _tick_ts=ts, clock=lambda: ts)


def test_a_catastrophe_gets_one_debug_session_and_no_fix_closes_it(rctx, store):
    led, st = _world(store)
    RS.step(rctx, store, _adapter(), led, _post(), st, 100.0)
    res = _Research()
    acts: list[str] = []
    RS.debug_sessions(_tick(rctx, store, res, st, ts=500.0), led, acts)
    (req, lane), = res.started
    assert req.operator is Operator.DEBUG and req.strategy is Strategy.MOONSHOT and req.parent_ids == ("b",)
    assert req.id == f"{RS.debug_prefix()}b-1" and lane == "repair" and "/out/b/s1" in req.brief
    assert "no bug" in req.brief.lower() and RS.book(store)["b"]["status"] == "started"
    res.inflight.clear()                                                       # the session ended, no node
    RS.debug_sessions(_tick(rctx, store, res, st, ts=900.0), led, acts)
    assert RS.book(store)["b"]["status"] == "no_bug" and not RS.is_open(RS.book(store)["b"])
    RS.debug_sessions(_tick(rctx, store, res, st, ts=1900.0), led, acts)
    assert len(res.started) == 1                                               # never re-issued
    assert [e["outcome"] for e in store.events("moonshot_debug") if e.get("phase") == "done"] == ["no_bug_found"]


def test_a_debug_session_lost_to_infra_is_reissued_once_never_read_as_no_bug_found(rctx, store):
    led, st = _world(store)
    RS.step(rctx, store, _adapter(), led, _post(), st, 100.0)
    res = _Research(free=5)
    acts: list[str] = []
    RS.debug_sessions(_tick(rctx, store, res, st, ts=500.0), led, acts)
    store.log("research_failed", request_id=f"{RS.debug_prefix()}b-1", status="restart")
    res.inflight.clear()
    RS.debug_sessions(_tick(rctx, store, res, st, ts=900.0), led, acts)
    assert [r.id for r, _ in res.started] == [f"{RS.debug_prefix()}b-1", f"{RS.debug_prefix()}b-2"]
    store.log("research_failed", request_id=f"{RS.debug_prefix()}b-2", status="restart")
    res.inflight.clear()
    RS.debug_sessions(_tick(rctx, store, res, st, ts=1900.0), led, acts)
    assert RS.book(store)["b"]["status"] == "no_bug" and len(res.started) == 2      # at most DEBUG_TRIES sessions


def test_a_debug_fix_is_reparented_onto_the_control_and_rescued_and_never_debugged_again(rctx, store):
    led, st = _world(store)
    RS.step(rctx, store, _adapter(), led, _post(), st, 100.0)
    res = _Research()
    acts: list[str] = []
    RS.debug_sessions(_tick(rctx, store, res, st, ts=500.0), led, acts)
    rid = RS.book(store)["b"]["request"]
    store.put_node(Node(id="bfix", task_id=store.task_id, commit="c-bfix", parent_id="b", strategy=Strategy.MOONSHOT,
                        operator=Operator.DEBUG, family="architecture", mechanism_tag="tag_b", diff_hash="hbfix",
                        status=NodeStatus.QUEUED, notes=f"request={rid} brief=repair"))
    res.inflight.clear()
    RS.debug_sessions(_tick(rctx, store, res, st, ts=900.0), led, acts)
    fix = store.get_node("bfix")
    assert fix.parent_id == ROOT_NODE_ID and "debug_of=b" in fix.notes          # judged against b's control
    bk = RS.book(store)
    assert bk["b"]["status"] == "fixed" and bk["b"]["child"] == "bfix" and "bfix" not in bk
    # held (never screened) until the per-tick pass rescues it under the rescue cap / GPU share
    led3, st3 = _world(store, extra_nodes=[_cd("bfix", ROOT_NODE_ID, fix.notes)])
    assert "bfix" in RS.hold(rctx, store, led3)
    RS.step(rctx, store, _adapter(), led3, _post(), {**st3, "bfix": "queued"}, 1000.0)
    bk = RS.book(store)
    assert bk["bfix"]["phase"] == "ported" and store.get_node(bk["bfix"]["port"]).parent_id == INC
    assert bk["bfix"]["src"] == ROOT_NODE_ID                                   # its whole change over b's control
    assert "bfix" in RS.claimed(store) and any("fixed as bfix" in a for a in acts)
    # a still-catastrophic debug child is never debugged a second time
    node = _cd("bfix2", ROOT_NODE_ID, "request=q brief=repair debug_of=b")
    store.put_node(dataclasses.replace(fix, id="bfix2", notes="request=q brief=repair debug_of=b", diff_hash="h2"))
    led2, st2 = _world(store, extra_nodes=[node], extra_obs=[_obs("bfix2", 1, 0.5)], extra_runs=[_run("bfix2", 1)])
    c = next(c for c in RS.candidates(rctx, store, led2, {**st2, "bfix2": "screening"}, _post(), INC)
             if c["node"] == "bfix2")
    assert c["verdict"] == "negative" and "already a debug fix" in c["why"]


# ---------------------------------------------------------------------------------------------- 5. retro + n0082
N0082 = ("--- a/src/natinit.py\n+++ b/src/natinit.py\n@@ -141,3 +141,9 @@ def apply_natural_init(model, cfg):\n"
         "     report = {}\n+    if cfg.get('pos_init', 'random') == 'sincos':\n+        table = sincos_table(model.dim)\n"
         "+        model.pos.copy_(table)\n+\n+def sincos_table(d):\n+    w = torch.arange(d)\n+    return torch.sin(w)\n")


def test_the_retro_pass_restores_a_judge_rejected_code_change_and_rescues_it_onto_the_incumbent(rctx, store, tmp_path):
    p = tmp_path / "n82.diff"
    p.write_text(N0082)
    w = {INC: WORLD[INC]}
    led, st = _world(store, w)
    store.put_node(Node(id="n82", task_id=store.task_id, commit="c-n82", parent_id=ROOT_NODE_ID,
                        strategy=Strategy.MOONSHOT, operator=Operator.DRAFT, family="architecture",
                        mechanism_tag="natinit_sincos_pos3d", diff_hash="h82", status=NodeStatus.REJECTED,
                        diff_path=str(p), notes=BOLD_NOTES,
                        reject_reason="bold_knob_only: the knob judge classified it as a re-weighting / ..."))
    assert BOLD.code_override(rctx, N0082)
    led, st = _world(store, w, extra_nodes=[_cd("n82", ROOT_NODE_ID)])
    acts = RS.step(rctx, store, _adapter(), led, _post(), {**st, "n82": "void"}, 100.0)
    n82 = store.get_node("n82")
    assert "knob_rejudged=1" in n82.notes and n82.status is NodeStatus.REJECTED       # never screened on its old parent
    rec = RS.book(store)["n82"]
    assert rec["phase"] == "ported" and store.get_node(rec["port"]).parent_id == INC
    assert store.get_state(RS.K_RETRO)["rejudged"] == ["n82"] and store.events("moonshot_rescue_retro")
    assert any("re-judged structural" in a for a in acts)
    RS.step(rctx, store, _adapter(), led, _post(), {**st, "n82": "void"}, 200.0)                  # retro: once
    assert len(store.events("moonshot_rescue_retro")) == 1


def test_off_switches(rctx, store):
    led, st = _world(store)
    assert RS.step(_with(rctx, moonshot_rescue=False), store, _adapter(), led, _post(), st, 1.0) == []
    assert RS.hold(_with(rctx, moonshot_rescue=False), store, led) == set()
    got = {c["node"]: c["verdict"] for c in RS.candidates(_with(rctx, moonshot_debug=False), store, led, st, _post(), INC)}
    assert got["b"] == "negative"


# ---------------------------------------------------------------------------------------------- 6. status + wiring
def test_status_lines_and_the_ladders_hook(rctx, store):
    led, st = _world(store)
    acts = PG.ladders(rctx, store, _adapter(), led, _post(), st, 100.0)       # the tick's housekeeping hook
    assert any("moonshot rescue a: ported" in a for a in acts) and any("moonshot debug b" in a for a in acts)
    out = "\n".join(RS.lines(store, rctx))
    assert "1/3 rescues + 1/2 debugs open" in out and "a RESCUE ported" in out and "b DEBUG pending" in out


def test_the_live_tick_wires_the_debug_sessions_and_the_holds():
    import inspect
    from loop import tick as TK
    src = inspect.getsource(TK.Tick)
    assert "debug_sessions(self, data, acts)" in src and "RESCUE.hold(self.ctx, self.store, vis)" in src
    from loop import __main__ as MAIN
    assert "rescue_lines" in inspect.getsource(MAIN._status)


def test_a_crashed_bold_node_is_debugged_fix_and_continue(rctx, store):
    w = {INC: WORLD[INC], "cr": (ROOT_NODE_ID, BOLD_NOTES, "moonshot", {}, [])}
    led, st = _world(store, w, extra_runs=[_run("cr", 1, "crash")])
    got = {c["node"]: c for c in RS.candidates(rctx, store, led, {**st, "cr": "crashed"}, _post(), INC)}
    assert got["cr"]["verdict"] == "debug" and "crashed" in got["cr"]["why"]
    assert RunStatus.CRASH                                                    # (the stored run status the tick derives)


def test_a_fresh_debug_fix_is_held_until_the_rescue_ports_it_and_a_dead_rung_never_blocks_a_candidate(rctx, store):
    led, st = _world(store)
    fresh = _cd("fx", "b", f"request={RS.debug_prefix()}b-1 brief=repair")
    led2, _ = _world(store, extra_nodes=[fresh])
    assert "fx" in RS.hold(rctx, store, led2)          # harvested this tick: never screened against the broken node
    # a finished-or-dead in-place ladder: the node is a candidate again (a dead rung never keeps it waiting)
    store.put_node(Node(id="dk", task_id=store.task_id, commit="c-dk", parent_id="a", strategy=Strategy.EXPLORE,
                        operator=Operator.IMPROVE, family="architecture", mechanism_tag="t", diff_hash="hdk",
                        status=NodeStatus.VOID, notes="ladder=a struct_tune=other"))
    store.set_state(ST.K_STRUCT, {"a": {"kind": "other", "children": ["dk"], "ts": 1.0, "need": 1}})
    store.set_state("config_diff:dk", {"weight_decay": 0.1})                  # the generic grid
    got = {c["node"]: c for c in RS.candidates(rctx, store, led, st, _post(), INC, now=10.0)}
    assert got["a"]["verdict"] == "rescue" and got["a"]["tuned"]


def test_a_dead_store_ladder_child_counts_dead_whatever_its_status_type():
    """11-10 fix: ``struct_tune._nodes_ok`` read str(NodeStatus.VOID) == 'NodeStatus.VOID' as alive."""
    assert not ST._nodes_ok(SimpleNamespace(status=NodeStatus.VOID))
    assert not ST._nodes_ok(SimpleNamespace(status="rejected")) and ST._nodes_ok(SimpleNamespace(status=NodeStatus.QUEUED))


# ---------------------------------------------------------------------------------------------- 11-11 adversarial review
def _decided_world(rctx, store, kid_seeds=2):
    pid, port, _ = _ported(rctx, store)
    for c in ("k1", "k2"):
        store.put_node(Node(id=c, task_id=store.task_id, commit=f"c-{c}", parent_id=pid, strategy=Strategy.EXPLORE,
                            operator=Operator.IMPROVE, family="architecture", mechanism_tag="t", diff_hash=f"h{c}",
                            status=NodeStatus.SCREENING, notes=f"ladder={pid} struct_tune=other"))
    kids = [_cd(c, pid, f"ladder={pid} struct_tune=other") for c in ("k1", "k2")]
    obs = [_obs(pid, s, 0.6) for s in (1, 2)] + [_obs("k1", s, 0.6) for s in range(1, kid_seeds + 1)] \
        + [_obs("k2", s, 0.6) for s in (1, 2)]
    led, st = _world(store, extra_nodes=[_cd(pid, INC, port.notes, port_of="a"), *kids], extra_obs=obs,
                     extra_runs=[_run(o.node, o.seed) for o in obs])
    return pid, led, st


def test_decide_closes_the_non_best_points_for_good_and_shrinks_the_best_for_its_k(rctx, store):
    """MUST-FIX 1: held non-best points sat in PORT_OPEN_MAX forever; now closed, never open ports / futility-exempt."""
    pid, led, st = _decided_world(rctx, store)
    post = _post({pid: (-0.001, 1e-5), "k1": (0.004, 1e-5), "k2": (-0.002, 1e-5)})
    acts = RS.step(rctx, store, _adapter(), led, post, st, 200.0)
    rec = RS.book(store)["a"]
    assert rec["best"] == "k1" and rec["k"] == 3 and any("closed" in a for a in acts)
    for c in (pid, "k2"):
        n = store.get_node(c)
        assert "rescue_closed=1" in n.notes and n.status in (NodeStatus.SCREENED_OUT, NodeStatus.REJECTED)
        assert not AQ.futility_exempt(_cd(c, INC, n.notes), 1)
    best = store.get_node("k1")
    assert "rescue_best_of=3" in best.notes and AQ.beta_for(best) >= 1.0
    assert AQ.struct_k(led, _cd("k1", pid, best.notes), 1) >= 3
    # a rescue port never counts against PORT_OPEN_MAX (its own cap bounds it)
    stx = {n: AQ.Status.SCREENING for n in led.nodes}
    assert AQ.open_ports(led, stx, AQ.Costs()) == 0
    RS.step(rctx, store, _adapter(), led, post, st, 300.0)
    assert sum(1 for e in store.events("moonshot_rescue") if e.get("phase") == "decided") == 1


def test_only_points_at_full_rescue_seeds_compete(rctx, store):
    """SHOULD-FIX 3: a 1-seed rung with the top belief cannot win."""
    pid, led, st = _decided_world(rctx, store, kid_seeds=1)
    store.set_state(ST.K_STRUCT, {pid: {"kind": "other+rescue", "children": ["k1", "k2"], "ts": 100.0, "rescue": True,
                                        "seeds": 2, "need": 2}})
    store.update_node("k1", status=NodeStatus.REJECTED)        # dead: the ladder is over with k1 at one seed
    got = RS.decide(rctx, store, led, _post({pid: (0.0, 1e-5), "k1": (0.01, 1e-5), "k2": (-0.001, 1e-5)}), "a",
                    RS.book(store)["a"], INC, 200.0)
    assert got["best"] == pid and "k1" not in [r[0] for r in got["table"]]


def test_the_rescue_is_never_decided_before_struct_ladders_has_seen_its_port(rctx, store):
    """MUST-FIX 2b: both port seeds in one tick decided the rescue on the port alone (4 rungs then held forever)."""
    pid, _, _ = _ported(rctx, store)
    store.set_state(ST.K_STRUCT, {})
    led, st = _world(store, extra_nodes=[_cd(pid, INC, store.get_node(pid).notes, port_of="a")],
                     extra_obs=[_obs(pid, 1, 0.6), _obs(pid, 2, 0.6)], extra_runs=[_run(pid, 1), _run(pid, 2)])
    post = _post({pid: (0.0, 1e-5)})
    assert RS.decide(rctx, store, led, post, "a", RS.book(store)["a"], INC, 200.0) is None
    store.set_state(ST.K_NONSTRUCT, {pid: "sig"})                # struct_tune judged it: no ladder -> port alone
    assert RS.decide(rctx, store, led, post, "a", RS.book(store)["a"], INC, 200.0)["best"] == pid


def test_a_rescue_point_gets_at_most_its_rescue_seeds_whatever_the_screen_action_asks(rctx, store):
    """MUST-FIX 2a: a 0-run point took a normal idea_runs (3) screen."""
    from loop import tick as TK
    pid, _, _ = _ported(rctx, store)
    led, _ = _world(store, extra_nodes=[_cd(pid, INC, store.get_node(pid).notes, port_of="a")])
    caps = RS.seed_caps(rctx, store, led)
    assert caps[pid] == 2 and caps["k1"] == 2
    core = TK.ModuleCore()
    core.seed_caps = caps
    specs = [TK.RunSpec(pid, TK.Stage.SCREEN, s) for s in (1, 2, 3)] + [TK.RunSpec(INC, TK.Stage.PARENT, 3)]
    core._runs_for = lambda *a, **k: specs
    got = core.runs_for(SimpleNamespace(kind="screen", node=pid, runs=3), led, (), {}, rctx.cfg)
    assert [s.seed for s in got] == [1, 2]                     # its control's seed-3 run dropped with it
    led2, _ = _world(store, extra_nodes=[_cd(pid, INC, store.get_node(pid).notes, port_of="a")],
                     extra_obs=[_obs(pid, 1, 0.6)], extra_runs=[_run(pid, 1), _run(pid, 2, "running")])
    assert RS.seed_caps(rctx, store, led2)[pid] == 0 and pid in RS.hold(rctx, store, led2)


def test_a_node_mid_screen_is_not_captured(rctx, store):
    """SHOULD-FIX 4: n0125 / n0140 were captured between seed 2 and 3 of a normal screen."""
    import time as _t
    w = {INC: WORLD[INC], "a": WORLD["a"]}
    led, st = _world(store, w)
    recent = [CD.Run("a:screen:2", "a", 2, "screen", "done", 1.0, _t.time() - 3600.0, _t.time(), "/o", 0)]
    led2, _ = _world(store, w, extra_runs=recent)
    assert "a" in {c["node"] for c in RS.candidates(rctx, store, led, st, _post(), INC)}
    assert "a" not in {c["node"] for c in RS.candidates(rctx, store, led2, st, _post(), INC)}


def test_rescue_researcher_sessions_take_at_most_their_slots_and_leave_the_reserve(rctx, store):
    """SHOULD-FIX 5: live tick 1 took all 4 researcher slots (3 port sessions + 1 debug)."""
    from loop.tick import Tick
    w = {INC: WORLD[INC], "b": WORLD["b"], "b2": (ROOT_NODE_ID, BOLD_NOTES, "moonshot", {1: 0.5}, [])}
    led, st = _world(store, w)
    RS.step(rctx, store, _adapter(), led, _post(), st, 100.0)
    bk = RS.book(store)
    bk["p1"] = {"phase": "port_session", "queued": True, "onto": INC, "src": ROOT_NODE_ID, "ts": 100.0, "how": "conflict",
                "score": 1.0}
    store.set_state(RS.K_RESCUE, bk)
    store.put_node(Node(id="p1", task_id=store.task_id, commit="c-p1", parent_id=ROOT_NODE_ID, strategy=Strategy.MOONSHOT,
                        operator=Operator.DRAFT, family="f", mechanism_tag="t", diff_hash="hp1",
                        status=NodeStatus.SCREENING, notes=BOLD_NOTES))
    res = _Research(free=4)
    acts: list[str] = []
    RS.debug_sessions(_tick(rctx, store, res, st, ts=500.0), led, acts)
    ps = store.get_state(Tick.K_PORT_SESSIONS) or {}
    assert len(res.started) + len(ps) == 1 and "p1" in ps     # one slot (the best prior first), never all four
    RS.debug_sessions(_tick(rctx, store, res, st, ts=600.0), led, acts)
    assert len(res.started) == 0                               # p1's session holds the slot
    two = _with(rctx, moonshot_rescue_llm_slots=3)
    r3 = _Research(free=3)
    RS.debug_sessions(_tick(two, store, r3, st, ts=700.0), led, acts)
    assert len(r3.started) == 1                                # 3 free, reserve 2: one more, then the reserve is kept
    # a port session never waits for ever
    late = 100.0 + 3600.0 * rctx.cfg.ideation.moonshot_rescue_hold_h + 1
    assert RS._adopt_port_session(store, "p1", RS.book(store)["p1"], acts, late,
                                  3600.0 * rctx.cfg.ideation.moonshot_rescue_hold_h).startswith("-: port session timed out")


def test_the_code_override_reads_the_parents_source(rctx):
    """SHOULD-FIX 6: an identifier the parent already has is no new code."""
    diff = ("--- a/m.py\n+++ b/m.py\n@@ -1,2 +1,3 @@\n     x = cfg.get('a', 1)\n+    y = scale(cfg.get('b', 2))\n"
            "+    z = model.head\n")
    parent = "def scale(v):\n    return v\nx = cfg.get('a', 1)\nmodel.head = 1\n"
    assert BOLD.code_override(rctx, diff, lambda p: parent) == ""
    assert BOLD.code_override(rctx, N0082)                      # n0082's new defs still are code


def test_retro_reruns_on_a_new_code_version_only(rctx, store, monkeypatch):
    led, st = _world(store)
    RS.step(rctx, store, _adapter(), led, _post(), st, 100.0)
    RS.step(rctx, store, _adapter(), led, _post(), st, 200.0)
    assert len(store.events("moonshot_rescue_retro")) == 1
    monkeypatch.setattr(RS, "code_version", lambda: "other")
    RS.step(rctx, store, _adapter(), led, _post(), st, 300.0)
    assert len(store.events("moonshot_rescue_retro")) == 2


def test_the_gpu_share_counts_the_runs_a_booking_will_launch(rctx, store, monkeypatch):
    w = {INC: WORLD[INC], **{f"r{i}": (ROOT_NODE_ID, BOLD_NOTES, "moonshot", {1: 0.6}, []) for i in range(3)}}
    led, st = _world(store, w)
    monkeypatch.setattr(RS, "share_full", lambda ctx, store, led, planned=0: (False, 1, 5))   # 1 in flight, cap 5
    RS.step(rctx, store, _adapter(), led, _post(), st, 100.0)
    assert sum(1 for r in RS.book(store).values() if isinstance(r, dict) and r["phase"] == "ported") == 2   # 1+2+2 <= 5


def test_a_failed_rescue_port_is_never_our_leader_node_nor_a_tagged_rescue_node(rctx, store):
    """11-11 live incident: the failed graft n0091@d4b5a0a carried n0103's commit (cutover.port_node records the
    TARGET's commit on a failure), sorted before n0103, and the leader re-root made it the incumbent."""
    from loop import leader as LD
    sha = "9ae54bc34a" + "0" * 30
    store.ensure_root("r0")
    store.put_node(Node(id="n0103", task_id=store.task_id, commit=sha, parent_id=ROOT_NODE_ID, strategy=Strategy.MOONSHOT,
                        operator=Operator.DRAFT, family="f", mechanism_tag="t", diff_hash="h1",
                        status=NodeStatus.CONFIRMING, notes=BOLD_NOTES))
    store.put_node(Node(id="n0091@d4b5a0a", task_id=store.task_id, commit=sha, parent_id="n0103",
                        strategy=Strategy.MOONSHOT, operator=Operator.DRAFT, family="f", mechanism_tag="t", diff_hash="h2",
                        status=NodeStatus.REJECTED, notes="port_of=n0091 port_failed"))
    assert LD.our_node(rctx, store, {"git_commit": sha}, sids={}) == "n0103"
    # the rescue strips its tags off a failed graft
    led, st = _world(store, {INC: WORLD[INC], "a": WORLD["a"]})
    ad = _adapter()
    ad.graft_port = lambda s, o, t, name: ("", "conflict: model.py")
    RS.step(rctx, store, ad, led, _post(), st, 100.0)
    failed = store.get_node(CO.port_id("a", EP))
    assert failed is not None and "port_failed" in failed.notes and "moonshot_rescue=" not in failed.notes
    assert RS.book(store)["a"]["phase"] == "port_session" and RS.book(store)["a"]["queued"]
