"""loop/v3 merge of v3-c7a x v3-c7i: the allocator's rates read paired-only increments (no wave point, no unpaired
node), a c7a probe repair is a REPAIR brief for the kind bandit and waits on the c7i brief throttle, the hb line carries
both pools= (c7a) and alloc= (c7i), and a v3 TEST-pass incumbent switch registers as a carrier (data.load)."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

from loop import tick as TK
from loop.contracts import ROOT_NODE_ID
from loop.core import alloc as AL
from loop.core import data as CD
from loop.core import ideation as ID
from loop.core import model as M

from .core.conftest import TASK
from .core.test_c7i import _led, _n, _r
from .test_c7a import ILL_TAIL
from .test_tick_core import _Means, _module_tick


# -- wiring 1: the allocator reads paired-only numbers --------------------------------------------------------------
def test_alloc_beliefs_drop_unpaired_nodes_for_a_posterior_and_a_view():
    est = {"a": {M.COMPOSITE: M.Estimate(0.004, 1e-6, 2, 2)},     # paired with its parent on both seeds
           "b": {M.COMPOSITE: M.Estimate(0.009, 1e-6, 1, 0)}}     # no shared seed: a raw level minus a mean
    post = SimpleNamespace(est=est, beliefs=lambda: {"a": (0.004, 1e-6, 2), "b": (0.009, 1e-6, 1)})
    assert AL.post_beliefs(post) == {"a": (0.004, 2)}
    view = SimpleNamespace(post=post, beliefs={"a": ID.Belief(0.004, 1e-6, 2), "b": ID.Belief(0.009, 1e-6, 1)})
    assert AL.post_beliefs(view) == {"a": (0.004, 2)}
    assert AL.post_beliefs(SimpleNamespace(beliefs={"b": ID.Belief(0.009, 1e-6, 1)})) == {"b": (0.009, 1)}  # no est


def test_a_lucky_wave_point_never_values_its_wave_only_x_star_does():
    """10-13: w013.p03 'best' .6858 was 4 replicates at one seed. A point's own increment is a draw of the design: it
    counts in the wave's GPU-h and nodes, never as its value; x* (factorial_of=) does."""
    nodes = [_n(f"w{i}", notes=f"wave=w001 point={i} block=0 seed=7", lane="factorial") for i in range(4)]
    runs = [_r(f"w{i}", 1.0) for i in range(4)]
    bel = {"w0": (0.012, 1), **{f"w{i}": (-0.001, 1) for i in range(1, 4)}}
    led = _led(nodes, runs)
    u = AL.units(led, bel)["wave:w001"]
    assert u["best"] is None and u["nodes"] == 4 and u["gpu_h"] == 4.0
    tab, _ = AL.kind_rates(led, bel, n0=1.0)
    assert tab["wave"]["gain"] == 0.0
    led = _led([*nodes, _n("xs", notes="factorial_of=w001 prior_m=0.002")], [*runs, _r("xs", 2.0)])
    u = AL.units(led, {**bel, "xs": (0.002, 2)})["wave:w001"]
    assert u["best"] == 0.002 and u["node"] == "xs" and u["gpu_h"] == 6.0
    sp = AL.gpu_split(AL.kind_rates(led, {**bel, "xs": (0.002, 2)}, n0=1.0)[0], 0.4, 0.1)
    assert sp["code_rate"] == 0.002 / 6.0


# -- wiring 3: a probe repair is a repair brief, throttled like any other ----------------------------------------------
def test_a_probe_repair_is_credited_to_the_repair_kind():
    assert ID.BRIEF_ALIAS["repair_probe"] == "repair"
    assert AL.brief_kind("request=req-v3-fix-n1-1 brief=repair_probe repair_of=n1") == "repair"   # pre-merge nodes
    led = _led([_n("f1", notes="request=req-v3-fix-n1-1 brief=repair repair_of=n1 lane=repair")], [_r("f1", 2.0)])
    # v3 review K1: a probe repair (repair_of=) is its own unit kind, out of the bandit and the GPU split
    assert AL.unit_of(led, "f1") == ("f1", AL.PROBE_REPAIR) and AL.group_of(AL.PROBE_REPAIR) == "other"
    assert ID._brief_tag(led.nodes["f1"]) == "repair"


def test_a_repair_session_waits_on_the_brief_throttle(ctx, tmp_path, monkeypatch):
    from loop.contracts import RunStatus

    from .test_c5_test_ideate import _open_block
    tick, store, t, b = _open_block(ctx, tmp_path, cap=2)
    cand = b.node_ids[0]
    data = CD.load(tick.db_path, tick.ctx.task_id)
    for r in list(store.runs(node_id=cand))[:2]:         # v3 review K4: >= 2 crashes (>= 1/3), positive evidence
        store.update_run(r.id, status=RunStatus.CRASH, log_tail=ILL_TAIL)
    monkeypatch.setattr(tick, "evidence", lambda *a, **k: {"vs": "p", "n": 2, "mean": 0.002, "seeds": {}})
    tick.dry = True
    monkeypatch.setattr(AL, "throttle", lambda c, bl: "backlog 50 > 40")
    acts: list[str] = []
    tick._repair_sessions(data, acts)
    assert store.get_state(tick.K_REPAIR)[cand]["status"] == "pending"            # booked, not started
    assert any(f"repair session for {cand} held: briefs throttled" in a for a in acts)
    assert not any(a.startswith("would start a researcher repair session") for a in acts)
    monkeypatch.setattr(AL, "throttle", lambda c, bl: "")
    acts = []
    tick._repair_sessions(data, acts)
    assert any(a.startswith(f"would start a researcher repair session for {cand}") for a in acts)
    tick.close()
    store.close()


# -- wiring 2: the hb line ---------------------------------------------------------------------------------------------
def test_the_heartbeat_carries_pools_and_the_alloc_split(ctx, tmp_path):
    tick, store, _ = _module_tick(ctx, tmp_path, TK.ModuleCore())
    tick.step()
    tick.core.split = {"share": 0.4, "llm_live": 3, "llm_cap_runs": 6}
    store.set_state("alloc.throttle", "backlog: 50 queued > 40")
    hb = tick.heartbeat(1, 0.5)
    assert " pools=" in hb and " alloc=llm3/6runs share0.40 throttle=backlog" in hb
    tick.close()
    store.close()


# -- fix: a TEST-pass incumbent switch is a carrier --------------------------------------------------------------------
def test_data_load_reads_carrier_events_at_their_close_time(ctx, tmp_path):
    tick, store, _ = _module_tick(ctx, tmp_path, TK.StubCore())
    led = CD.load(tick.db_path, TASK)
    child = next(n for n in led.nodes.values() if n.parent not in (None, ROOT_NODE_ID) and n.parent in led.nodes
                 and n.carrier != n.parent)
    store.log("carrier", node_id=child.parent, previous="x", block="test-x", at=child.created_ts - 1.0,
              source="test_pass")
    led2 = CD.load(tick.db_path, TASK)
    assert (child.created_ts - 1.0, child.parent) in led2.carriers and led2.nodes[child.id].carrier == child.parent
    store.log("carrier", node_id=child.parent, previous="x", block="test-y", at=child.created_ts + 1.0)
    assert CD.load(tick.db_path, TASK).nodes[child.id].carrier == child.parent    # the earliest close counts
    tick.close()
    store.close()


def test_a_test_pass_switch_registers_the_new_incumbent_as_a_carrier_once(ctx, tmp_path):
    """10-14: n1248@065b87f became the incumbent by a v3 TEST PASS, which logged neither an audit nor an incumbent
    event, so its children sat in the root prior cell. The tick now logs a ``carrier`` event at the block's close."""
    tick, store, t = _module_tick(ctx, tmp_path, _Means([0.004]))
    tick.step()
    [opened] = [b for b in store.seed_blocks() if b.id.startswith("test-")]
    t[0] += 60
    rep = tick.step()
    inc = TK.incumbent_of(rep.statuses)
    assert inc == opened.node_ids[0]
    [ev] = [e for e in store.events("carrier") if e["node_id"] == inc]
    closed = next(b for b in store.seed_blocks() if b.id == opened.id).closed_ts
    assert ev["source"] == "test_pass" and ev["block"] == opened.id and ev["at"] == closed
    assert ev["previous"] == opened.node_ids[-1]
    assert (closed, inc) in CD.load(tick.db_path, TASK).carriers
    assert inc in {c for _, c in tick.data.carriers}                     # this tick's ledger already sees it
    assert any("registered as a carrier" in x for x in rep.lines)
    t[0] += 60
    tick.step()
    assert len([e for e in store.events("carrier") if e["node_id"] == inc]) == 1      # once
    tick.close()
    store.close()


def test_the_carrier_sync_is_a_no_op_when_the_incumbent_is_a_carrier_already(ctx, tmp_path):
    tick, store, _ = _module_tick(ctx, tmp_path, TK.ModuleCore())
    rep = tick.step()
    inc = TK.incumbent_of(rep.statuses)
    data = dataclasses.replace(tick.data, carriers=(*tick.data.carriers, (0.0, inc)))
    assert tick._sync_carriers(data, object(), rep.statuses, []) is False
    tick.close()
    store.close()
