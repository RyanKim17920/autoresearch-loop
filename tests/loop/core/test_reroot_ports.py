"""RE-ROOT PORTS (operator 11-10): after an incumbent switch the old incumbent's best descendants (EB-shrunk paired gain
over the OLD incumbent > 0, P >= p_min, >= min_seeds paired seeds; best per lineage) are grafted onto the new incumbent
(loop.core.altbase.reroot_plan / tick._reroot_ports)."""

from __future__ import annotations

import math
from statistics import NormalDist
from types import SimpleNamespace

from loop.core import altbase as AB
from loop.core.data import AXES, Ledger, Node, Obs

BASE = "c" * 40
SEEDS = (1, 2, 3, 8, 2026)
ROOT = {s: 0.5471 + 0.003 * math.sin(s) for s in SEEDS}
POP = [-0.0026 + 0.002 * NormalDist().inv_cdf((i + 0.5) / 24) for i in range(24)]
JIT = (0.006, -0.006, 0.003, -0.003, 0.0)


def _node(nid, parent, family="f", notes="", second=None, port_of=""):
    return Node(nid, parent, second, "explore", family, "explore", "root", 0.0, "", notes, BASE, port_of)


def _obs(node, seed, y):
    return Obs(f"{node}:screen:{seed}", node, seed, "screen", "select", {a: y for a in AXES}, y, 1.0, 0.0, 0, None,
               BASE)


def world(extra_nodes=(), gains=None):
    """root (old incumbent) -> inc (the new one, 'sampling'); a strong child 'good' (+.0035 on 5/5), a ladder lineage
    'arch' (-.0004 on 1 seed) with points 'arch_hi' (+.003 on 3 seeds) and 'arch_lo' (+.001 on 3), a sampling sibling
    'samp' (+.004), a noise population p00..p23, and a child of the new incumbent."""
    gains = gains or {"inc": [0.005] * 5, "good": [0.0037, 0.0034, 0.0041, 0.0043, 0.0011],
                      "arch": [-0.0004], "arch_hi": [0.0030, 0.0035, 0.0025], "arch_lo": [0.001, 0.0005, 0.0015],
                      "samp": [0.004, 0.0045, 0.0035, 0.004, 0.0042], "kid": [0.006] * 3}
    nodes = {"root": _node("root", None), "inc": _node("inc", "root", "sampling"),
             "good": _node("good", "root", "objective"), "arch": _node("arch", "root", "architecture"),
             "arch_hi": _node("arch_hi", "arch", "architecture", "ladder=arch"),
             "arch_lo": _node("arch_lo", "arch", "architecture", "ladder=arch"),
             "samp": _node("samp", "root", "sampling"), "kid": _node("kid", "inc", "objective")}
    nodes.update({n.id: n for n in extra_nodes})
    obs = [_obs("root", s, y) for s, y in ROOT.items()]
    for nid, ds in gains.items():
        obs += [_obs(nid, s, ROOT[s] + d) for s, d in zip(SEEDS, ds)]
    for i, g in enumerate(POP):
        nid = f"p{i:02d}"
        nodes[nid] = _node(nid, "root")
        obs += [_obs(nid, s, ROOT[s] + g + JIT[(i + j) % 5]) for j, s in enumerate(SEEDS[:2 + i % 3])]
    return Ledger(nodes, (), tuple(obs), frozenset(), ((0.0, "root"), (10.0, "inc")), bases=((0.0, "root", BASE),))


class Store:
    def __init__(self, led):
        self.n = {k: SimpleNamespace(id=k, commit=f"c-{k}", status="screening", family=v.family, notes=v.notes)
                  for k, v in led.nodes.items()}

    def get_node(self, nid):
        return self.n.get(nid)


def test_previous_incumbent_is_the_carrier_before_the_switch():
    led = world()
    assert AB.previous_incumbent(led, "inc") == "root"
    assert AB.previous_incumbent(led, "root") == ""


def test_plan_ports_the_strong_child_and_the_lineage_best_never_the_new_incumbents_subtree():
    led = world()
    picks, skipped = AB.reroot_plan(led, Store(led), {}, "inc", "root", p_min=0.6, min_seeds=3, eb_min_nodes=5)
    ids = [p.node for p in picks]
    assert ids[0] in ("good", "samp")
    assert "arch_hi" in ids and "arch_lo" not in ids and "arch" not in ids     # one per lineage, its best point
    assert next(p for p in picks if p.node == "arch_hi").lineage == "arch"
    assert "kid" not in ids and "inc" not in ids and "root" not in ids
    assert all(p.m > 0 and p.p >= 0.6 and p.n >= 3 for p in picks)
    assert skipped == {}


def test_same_mechanism_is_skipped_and_reported():
    led = world()
    same = lambda nid: "same mechanism" if led.nodes[nid].family == "sampling" else ""   # noqa: E731
    picks, skipped = AB.reroot_plan(led, Store(led), {}, "inc", "root", p_min=0.6, min_seeds=3, eb_min_nodes=5,
                                    same_mech=same)
    assert "samp" not in [p.node for p in picks] and skipped == {"samp": "same mechanism"}


def test_thresholds_seed_count_and_unsound_nodes():
    led = world()
    st = Store(led)
    picks, _ = AB.reroot_plan(led, st, {}, "inc", "root", p_min=0.99, min_seeds=3, eb_min_nodes=5)
    assert picks == []                                                        # nothing that sure after shrinkage
    picks, _ = AB.reroot_plan(led, st, {}, "inc", "root", p_min=0.6, min_seeds=4, eb_min_nodes=5)
    assert "arch_hi" not in [p.node for p in picks]                           # 3 paired seeds < 4
    st.n["good"].status = "rejected"
    picks, _ = AB.reroot_plan(led, st, {}, "inc", "root", p_min=0.6, min_seeds=3, eb_min_nodes=5)
    assert "good" not in [p.node for p in picks]
    # no EB prior (too few paired nodes): nothing
    assert AB.reroot_plan(led, st, {}, "inc", "root", p_min=0.6, min_seeds=3, eb_min_nodes=500) == ([], {})


def test_ports_and_crossovers_are_not_candidates():
    extra = (_node("x", "root", "objective", "pair_of=inc,good", second="good"),
             _node("good@new", "root", "objective", "port_of=good", port_of="good"))
    g = {"inc": [0.005] * 5, "good": [0.0037, 0.0034, 0.0041, 0.0043, 0.0011], "x": [0.008] * 5,
         "good@new": [0.008] * 5}
    led = world(extra, g)
    picks, _ = AB.reroot_plan(led, Store(led), {}, "inc", "root", p_min=0.6, min_seeds=3, eb_min_nodes=5)
    assert [p.node for p in picks] == ["good"]


def test_tick_ports_with_the_old_incumbent_as_anchor_books_and_keeps_screening(monkeypatch):
    from loop import cutover as CO
    from loop import tick as TK
    led = world()
    calls, logs, state = [], [], {}
    port_nodes = {}

    def fake_port(store, adapter, data, old, now, *, onto=None, tag="", src=None, lane_tags=True, **_):
        calls.append((old, onto, tag, src, lane_tags))
        nid = f"{old}@new"
        port_nodes[nid] = SimpleNamespace(id=nid, status=SimpleNamespace(value="queued"))
        return CO.Ported(old, nid, onto, src, "applied", "sha")
    monkeypatch.setattr(CO, "port_node", fake_port)
    st = Store(led)

    class S:
        task_id = "T"

        def get_node(self, nid):
            return port_nodes.get(nid) or st.get_node(nid)

        def get_state(self, k):
            return state.get(k)

        def set_state(self, k, v):
            state[k] = v

        def log(self, ev, **f):
            logs.append(ev)

        def runs(self, node_id=None):
            return []

    class Core:
        costs = SimpleNamespace(idea_runs=3)

        def screen_specs(self, node, parent, k, data, blocks, cfg):
            return [(node, parent, i) for i in range(k)]
    t = SimpleNamespace(ctx=SimpleNamespace(cfg=SimpleNamespace(
        test=SimpleNamespace(reroot_port=True, reroot_port_k=2, reroot_port_p=0.6, reroot_port_min_seeds=3),
        labless=SimpleNamespace(eb_min_nodes=5))), store=S(), adapter=SimpleNamespace(graft_port=lambda *a: None),
        core=Core(), dry=False, blocks=lambda: [])
    t._same_mech = lambda inc, old: (lambda nid: "same mechanism" if nid == "samp" else "")
    statuses = {"inc": SimpleNamespace(name="INCUMBENT")}
    acts = []
    specs = TK.Tick._reroot_ports(t, led, statuses, 100.0, acts)
    assert [c[0] for c in calls] == ["good", "arch_hi"]                       # k=2, best first, samp skipped
    assert all(c[1] == "inc" and c[3] == "root" and c[2] == AB.REROOT_TAG and c[4] is False for c in calls)
    assert len(specs) == 6 and state[AB.K_REROOT]["inc"]["skipped"] == {"samp": "same mechanism"}
    assert "reroot_port" in logs and "reroot_port_skip" in logs
    # next tick: nothing new is ported (k spent), the open ports still get their screens ahead of the queue
    calls.clear()
    specs = TK.Tick._reroot_ports(t, led, statuses, 200.0, acts)
    assert calls == [] and len(specs) == 6
    # off: nothing
    t.ctx.cfg.test.reroot_port = False
    assert TK.Tick._reroot_ports(t, led, statuses, 300.0, acts) == []


def test_a_sibling_under_an_ancestor_the_new_incumbent_shares_is_not_ported():
    """root -> mid -> inc2: a sibling under mid has mid's gain in its gain over root, which inc2 already holds."""
    extra = (_node("mid", "root", "objective"), _node("inc2", "mid", "sampling"), _node("sib", "mid", "objective"))
    g = {"mid": [0.004] * 5, "inc2": [0.006] * 5, "sib": [0.0045] * 5, "good": [0.0037, 0.0034, 0.0041, 0.0043, 0.0011]}
    led = world(extra, g)
    picks, _ = AB.reroot_plan(led, Store(led), {}, "inc2", "root", p_min=0.6, min_seeds=3, eb_min_nodes=5)
    ids = [p.node for p in picks]
    assert "sib" not in ids and "mid" not in ids and "good" in ids


# ---------------------------------------------------------------------------------------------- re-root crossovers
TAGS = {"inc": "cross_subject_replaced_token_detection", "good": "intra_image_token_shuffle_detection",
        "samp": "subject_sampler", "arch_hi": "deeper_blocks_hi"}


class XStore:
    """A store with nodes() / put_node / state / log: enough for _reroot_ports + _reroot_cross(_sessions)."""

    def __init__(self, led, tags=TAGS):
        self.n = {k: SimpleNamespace(id=k, task_id="T", commit=f"c-{k}", status=SimpleNamespace(value="screening"),
                                     family=v.family, notes=v.notes, parent_id=v.parent, second_parent_id=None,
                                     mechanism_tag=tags.get(k, f"tag_{k}"))
                  for k, v in led.nodes.items()}
        self.state, self.logs, self.put = {}, [], []

    def get_node(self, nid):
        return self.n.get(nid)

    def nodes(self):
        return list(self.n.values())

    def put_node(self, node):
        self.n[node.id] = node
        self.put.append(node.id)

    def get_state(self, k):
        return self.state.get(k)

    def set_state(self, k, v):
        import copy
        self.state[k] = copy.deepcopy(v)

    def log(self, ev, **f):
        self.logs.append((ev, f))

    def runs(self, node_id=None):
        return []


class Research:
    def __init__(self, free=1):
        self.inflight, self.requeue, self.started, self._free = {}, [], [], free

    def free(self):
        return self._free - len(self.started)

    def capacity(self):
        return self.free()

    def start(self, req, *_a, **_k):
        self.started.append(req)
        self.inflight[req.id] = req


def xtick(led, store, *, graft="conflict: src/model.py", same=("good", "samp", "arch_hi"), k=2, cross=True,
          port_k=4, research=None):
    import types
    from loop import cutover as CO
    from loop import tick as TK
    grafts = []

    def graft_port(src, dst, onto, name):
        grafts.append((src, dst, onto, name))
        return ("sha-x", "applied") if graft == "applied" else ("", graft)

    class Core:
        costs = SimpleNamespace(idea_runs=3)

        def screen_specs(self, node, parent, n, data, blocks, cfg):
            return [(node, parent, i) for i in range(n)]
    t = SimpleNamespace(
        ctx=SimpleNamespace(task_id="T", cfg=SimpleNamespace(
            test=SimpleNamespace(reroot_port=True, reroot_port_k=port_k, reroot_port_p=0.6, reroot_port_min_seeds=3,
                                 reroot_cross=cross, reroot_cross_k=k),
            labless=SimpleNamespace(eb_min_nodes=5))),
        store=store, adapter=SimpleNamespace(graft_port=graft_port, diff_commits=lambda a, b: f"diff {a}..{b}"),
        core=Core(), dry=False, blocks=lambda: [], statuses={"inc": SimpleNamespace(name="INCUMBENT")},
        research=research or Research(), _tick_ts=1000.0, clock=lambda: 1000.0, llm_state=lambda: "ok",
        REROOT_XOVER_TRIES=TK.Tick.REROOT_XOVER_TRIES, grafts=grafts)
    for m in ("_reroot_cross", "_reroot_cross_brief", "_reroot_cross_sessions", "_missing_prereqs"):
        setattr(t, m, types.MethodType(getattr(TK.Tick, m), t))
    t._missing_prereqs = lambda *a, **kw: None
    why = "same mechanism as inc (family objective; both change src/model.py)"
    t._same_mech = lambda inc, old: (lambda nid: why if nid in same else "")

    def fake_port(st, adapter, data, old, now, *, onto=None, tag="", src=None, lane_tags=True, **_):
        nid = f"{old}@new"
        st.n[nid] = SimpleNamespace(id=nid, status=SimpleNamespace(value="queued"), parent_id=onto,
                                    second_parent_id=None, notes="", commit="c", mechanism_tag="", family="")
        return CO.Ported(old, nid, onto, src, "applied", "sha")
    return t, fake_port


def _run(t, led, now=100.0):
    from loop import tick as TK
    acts = []
    specs = TK.Tick._reroot_ports(t, led, t.statuses, now, acts)
    return specs, acts


def test_true_same_is_a_stem_or_lineage_relation_not_family_plus_file():
    led = world((_node("good_hi", "good", "objective", "ladder=good"),),
                {"inc": [0.005] * 5, "good": [0.004] * 5, "good_hi": [0.004] * 5})
    st = XStore(led, {**TAGS, "good_hi": "intra_image_token_shuffle_detection_muon_lr_mult_hi"})
    assert AB.true_same(led, st, "inc", "good") == ""                         # different aux heads: not the same
    assert AB.true_same(led, st, "inc", "good_hi") == ""
    st.n["good"].mechanism_tag = "cross_subject_replaced_token_detection"     # the same head under another node
    assert "identical mechanism stem" in AB.true_same(led, st, "inc", "good")
    st.n["good_hi"].mechanism_tag = "lr_hi"                                   # a ladder point: its ladder base's stem
    assert "identical mechanism stem" in AB.true_same(led, st, "inc", "good_hi")
    assert "ancestor" in AB.true_same(led, st, "inc", "root") and AB.stem_same("a_b", "a") and not AB.stem_same("ab", "a")


def test_a_family_skip_with_port_evidence_is_crossed_not_dropped(monkeypatch):
    from loop import cutover as CO
    led = world()
    st = XStore(led)
    t, fake = xtick(led, st, same=("samp",))
    monkeypatch.setattr(CO, "port_node", fake)
    specs, acts = _run(t, led)
    ent = st.state[AB.K_REROOT]["inc"]
    assert "samp" in ent["skipped"] and "samp" not in ent["ported"]
    rec = ent["crossed"]["samp"]
    assert rec["status"] == "pending" and rec["request"] == "req-v3-rxover-inc-samp" and rec["lineage"] == "samp"
    assert rec["m"] > 0 and rec["p"] >= 0.6 and rec["n"] >= 3
    assert t.grafts and t.grafts[0][:3] == ("c-root", "c-samp", "c-inc")    # its change over the OLD incumbent, onto inc
    assert any(ev == "reroot_port_cross" for ev, _ in st.logs)
    # the session: a CROSSOVER request with parents (new incumbent, candidate), started once
    acts = []
    t._reroot_cross_sessions(acts)
    (req,) = t.research.started
    assert req.parent_ids == ("inc", "samp") and req.brief_kind == "crossover" and "CROSSOVER (re-root)" in req.brief
    assert "diff c-root..c-samp" in req.brief
    assert st.state[AB.K_REROOT]["inc"]["crossed"]["samp"]["status"] == "started"
    t._reroot_cross_sessions(acts)
    assert len(t.research.started) == 1                                       # in flight: never a second
    # its node lands (intake tags request=<id>): booked, then screened ahead of the queue against the new incumbent
    st.n["n9000"] = SimpleNamespace(id="n9000", task_id="T", commit="c-x", status=SimpleNamespace(value="queued"),
                                    family="sampling", notes="request=req-v3-rxover-inc-samp reroot_cross_of=samp",
                                    parent_id="inc", second_parent_id="samp", mechanism_tag="x")
    t.research.inflight.clear()
    t._reroot_cross_sessions(acts)
    assert st.state[AB.K_REROOT]["inc"]["crossed"]["samp"] | {} and \
        st.state[AB.K_REROOT]["inc"]["crossed"]["samp"]["node"] == "n9000"
    specs, _ = _run(t, led, 200.0)
    assert ("n9000", "inc", 0) in specs and sum(1 for s in specs if s[0] == "n9000") == 3


def test_a_clean_graft_makes_a_code_crossover_node():
    from loop import cutover as CO
    import pytest
    mp = pytest.MonkeyPatch()
    led = world()
    st = XStore(led)
    t, fake = xtick(led, st, graft="applied", same=("samp",))
    mp.setattr(CO, "port_node", fake)
    try:
        specs, _ = _run(t, led)
    finally:
        mp.undo()
    rec = st.state[AB.K_REROOT]["inc"]["crossed"]["samp"]
    node = st.n[rec["node"]]
    assert rec["status"] == "done" and node.parent_id == "inc" and node.second_parent_id == "samp"
    assert node.commit == "sha-x" and "reroot_cross_of=samp" in node.notes and "pair_of=inc,samp" in node.notes
    assert getattr(node.status, "value", node.status) == "queued"
    assert sum(1 for s in specs if s[0] == rec["node"] and s[1] == "inc") == 3


def test_a_true_same_mechanism_is_still_skipped(monkeypatch):
    from loop import cutover as CO
    led = world()
    st = XStore(led, {**TAGS, "samp": "cross_subject_replaced_token_detection_v2"})
    t, fake = xtick(led, st, same=("samp",))
    monkeypatch.setattr(CO, "port_node", fake)
    _run(t, led)
    ent = st.state[AB.K_REROOT]["inc"]
    assert "samp" in ent["skipped"] and ent["crossed"] == {} and t.grafts == []


def test_at_most_k_crossovers_best_gain_first_one_per_lineage(monkeypatch):
    from loop import cutover as CO
    extra = (_node("good2", "good", "objective", "ladder=good"),)
    g = {"inc": [0.005] * 5, "good": [0.0037, 0.0034, 0.0041, 0.0043, 0.0011], "good2": [0.0045] * 5,
         "arch": [-0.0004], "arch_hi": [0.0030, 0.0035, 0.0025], "samp": [0.004, 0.0045, 0.0035, 0.004, 0.0042]}
    led = world(extra, g)
    st = XStore(led)
    t, fake = xtick(led, st, same=("good", "good2", "samp", "arch_hi"), k=2)
    monkeypatch.setattr(CO, "port_node", fake)
    _run(t, led)
    crossed = st.state[AB.K_REROOT]["inc"]["crossed"]
    assert len(crossed) == 2
    assert len({v["lineage"] for v in crossed.values()}) == 2                 # one per lineage
    assert not ({"good", "good2"} <= set(crossed))
    cands = {}
    AB.reroot_plan(led, st, {}, "inc", "root", p_min=0.6, min_seeds=3, eb_min_nodes=5,
                   same_mech=lambda n: "x" if n in ("good", "good2", "samp", "arch_hi") else "", crossable=cands)
    best = {}
    for pk in cands.values():
        if pk.lineage not in best or pk.m > best[pk.lineage].m:
            best[pk.lineage] = pk
    top2 = {pk.node for pk in sorted(best.values(), key=lambda x: -x.m)[:2]}
    assert set(crossed) == top2                                               # the two best lineages by EB gain
    ent = st.state[AB.K_REROOT]["inc"]
    assert set(ent["skipped"]) <= set(crossed) | set(ent["cross_declined"]) | {"arch_hi", "samp", "good", "good2"}
    assert any(v.startswith("lineage") for v in ent["cross_declined"].values())   # the lineage's second: declined
    _run(t, led, 300.0)                                                       # next tick: the cap holds
    assert len(st.state[AB.K_REROOT]["inc"]["crossed"]) == 2 and len(t.grafts) == 2


def test_catch_up_of_a_reroot_made_before_deploy_is_idempotent(monkeypatch):
    """The live 11-10 shape: reroot.ports[inc] has 'skipped' with the coarse reason and no 'crossed' -- the next tick
    crosses them once; another tick, and a restart (a fresh tick object on the same state), make nothing more."""
    from loop import cutover as CO
    led = world()
    st = XStore(led)
    why = "same mechanism as inc (family objective; both change src/model.py)"
    st.state[AB.K_REROOT] = {"inc": {"from": "root", "ts": 1.0, "failed": {}, "skipped": {"samp": why},
                                     "ported": {"good": {"port": "good@new", "how": "applied", "lineage": "good"}}}}
    t, fake = xtick(led, st, same=("samp",), port_k=1)                        # port slots spent: crossing still runs
    monkeypatch.setattr(CO, "port_node", fake)
    _run(t, led)
    crossed = st.state[AB.K_REROOT]["inc"]["crossed"]
    assert list(crossed) == ["samp"] and crossed["samp"]["status"] == "pending"
    assert [e for e, _ in st.logs].count("reroot_port_skip") == 0             # the old skip is not re-logged
    _run(t, led, 200.0)
    t2, _ = xtick(led, st, same=("samp",), port_k=1)                          # a restart
    _run(t2, led, 300.0)
    assert len(t.grafts) == 1 and t2.grafts == []
    assert [e for e, _ in st.logs].count("reroot_port_cross") == 1
    t2._reroot_cross_sessions([])
    t2._reroot_cross_sessions([])
    assert len(t2.research.started) == 1
    # a restart after the session made its node but before it was booked: adopted, not a second session
    st.n["n9001"] = SimpleNamespace(id="n9001", task_id="T", commit="c", status=SimpleNamespace(value="queued"),
                                    family="", notes="request=req-v3-rxover-inc-samp reroot_cross_of=samp",
                                    parent_id="inc", second_parent_id="samp", mechanism_tag="")
    t3, _ = xtick(led, st, same=("samp",), port_k=1)
    t3._tick_ts = 5000.0
    t3._reroot_cross_sessions([])
    assert t3.research.started == [] and st.state[AB.K_REROOT]["inc"]["crossed"]["samp"]["node"] == "n9001"


def test_cross_off_keeps_the_plain_skip(monkeypatch):
    from loop import cutover as CO
    led = world()
    st = XStore(led)
    t, fake = xtick(led, st, same=("samp",), cross=False)
    monkeypatch.setattr(CO, "port_node", fake)
    _run(t, led)
    assert st.state[AB.K_REROOT]["inc"]["crossed"] == {} and t.grafts == []
