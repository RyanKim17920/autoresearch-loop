"""c5 wave fixes: knobs checked against the carrier config (A1: w007 built 0/16 points on one absent key), a wave
that built no point re-armed, stacking legs (>= 2 seeds, one per mechanism) ahead of relax waves; zombie waves (R1:
a voided point never finished its wave, so w003/w005/w006/w008/w010 counted open forever); typed wave levels (R2)."""

from __future__ import annotations

import dataclasses
import math

import pytest

from loop import tick as TK
from loop.contracts import Node, NodeStatus, RunStatus, Strategy, VoidType
from loop.core import data as CD
from loop.core import factorial as F
from loop.core import model as M

from .test_port import _two_epochs
from .test_waves import _Post


# -- R2: coerce_like / check_knobs ----------------------------------------------------------------------------------
def test_coerce_like_keeps_the_carrier_value_type():
    assert F.coerce_like(5.6, 3) == 6 and type(F.coerce_like(5.6, 3)) is int
    assert F.coerce_like(2.5, 3) == 3 and F.coerce_like(-2.5, 3) == -3          # half away from zero
    assert F.coerce_like(4, 3) == 4 and type(F.coerce_like(4.0, 3)) is int
    assert F.coerce_like(1, True) is True and F.coerce_like(0.0, False) is False and F.coerce_like(False, True) is False
    assert F.coerce_like(2, 0.5) == 2.0 and type(F.coerce_like(2, 0.5)) is float
    assert F.coerce_like([0.1, 1], [0.08, 2]) == [0.1, 1]
    assert F.coerce_like(1e-4, "1e-4") == 1e-4 and F.coerce_like("x", None) == "x"      # str / None ref: as is
    for value, ref in ((0.5, True), (True, 3), (math.inf, 3), ("a", 0.1), ([1], [1, 2]), (2, False)):
        with pytest.raises(ValueError):
            F.coerce_like(value, ref)


def _bk(key, lo, hi, scale="lin", source="plateau"):
    return F.BankKnob(F.Knob(key, lo, hi, scale), 0.0, 1e-6, source)


def test_check_knobs_drops_absent_keys_keeps_diffs_and_types_levels():
    cfg = {"dino.jepa_blocks": 3, "dino.flag": True, "dino.lr": 1e-4, "data.crop": [0.08, 0.35]}
    bank = [_bk("n7", 0, 1, "diff", "stacking"),                       # a node diff: not a config key, kept
            _bk("dino.schedule_ref_frac", 0.3, 0.95),                  # w007's knob: the carrier has no such key
            _bk("dino.jepa_blocks", 3, 4.445, "log"),                  # an int key: 4.445 -> 4
            _bk("dino.jepa_blocks", 3, 4.2, "log", "next"),            # -> 3 -> 4 too: the same knob, dropped
            _bk("dino.flag", 1, 0),                                    # bool levels
            _bk("dino.lr", 1e-4, 2e-4, "log"),
            _bk("data.crop", [0.08, 0.35], [0.16, 0.35]),
            _bk("dino.flag", 0.5, 1)]                                  # no bool
    kept, dropped = F.check_knobs(bank, cfg)
    assert [b.knob for b in kept] == [F.Knob("n7", 0, 1, "diff"), F.Knob("dino.jepa_blocks", 3, 4, "log"),
                                      F.Knob("dino.flag", True, False, "lin"), F.Knob("dino.lr", 1e-4, 2e-4, "log"),
                                      F.Knob("data.crop", [0.08, 0.35], [0.16, 0.35])]
    assert type(kept[1].knob.hi) is int and kept[2].knob.lo is True
    why = {b.knob.key + str(b.knob.hi): w for b, w in dropped}
    assert "has no dino.schedule_ref_frac" in why["dino.schedule_ref_frac0.95"]
    assert "same knob" in why["dino.jepa_blocks4"] and "not a bool" in why["dino.flag1"]
    kept2, d2 = F.check_knobs([_bk("dino.jepa_blocks", 3, 3.4)], cfg)      # 3 -> 3: collapses
    assert not kept2 and "collapse" in d2[0][1]
    assert F.check_knobs(bank, {})[0] == bank                          # unreadable config: nothing checked
    wave = F.design("w1", "c", [b.knob for b in kept], (1, 2))        # every point typed like the carrier
    assert all(type(p["dino.jepa_blocks"]) is int and type(p["dino.flag"]) is bool for p in wave.points)


# -- A1: the knob bank and the build ----------------------------------------------------------------------------------
def _child(store, nid, commit, parent, ts, family):
    store.put_node(Node(id=nid, task_id=store.task_id, commit=commit, parent_id=parent, strategy=Strategy.EXPLORE,
                        operator=None, family=family, mechanism_tag=nid, diff_hash=nid, status=NodeStatus.SCREENING,
                        created_ts=ts, updated_ts=ts, notes=""))


class _BankPost(_Post):
    def __init__(self, beliefs, ns, epoch):
        self._b, self.epoch = beliefs, epoch
        self.est = {n: {M.COMPOSITE: M.Estimate(m, 1e-6, ns[n], ns[n])} for n, m in beliefs.items()}

    def belief(self, nid):
        return self._b[nid], 1e-6


def test_carrier_bank_drops_absent_keys_once_and_keeps_independent_multi_seed_legs(ctx, tmp_path):
    from loop.core.ideation import K_PLATEAU
    sim, store = _two_epochs(ctx, tmp_path)
    car = "inc@new"
    commit = store.get_node(car).commit
    for i, fam in enumerate(("fa", "fa", "fb", "fc", "fd")):           # a, b: one mechanism (family fa)
        _child(store, f"s{i}", commit, car, 20.0 + i, fam)
    core = TK.ModuleCore()
    tick = TK.Tick(ctx, store, sim, core, db_path=tmp_path / "loop.db", clock=lambda: 50.0)
    store.set_state(core.K_BANK + car, {"knobs": [], "lines": [], "tau2": 2e-6, "effects": 0, "used": "w000",
                                        "v": core.RETRO_V})
    store.set_state(K_PLATEAU, {"hypotheses": [{"knobs": [
        {"key": "dino.schedule_ref_frac", "lo": 0.3, "hi": 0.95},       # absent from the carrier: dropped
        {"key": "train.lr", "lo": 7e-4, "hi": 1e-3}]}]})
    led = CD.load(tmp_path / "loop.db", ctx.task_id)
    post = _BankPost({"s0": 0.002, "s1": 0.003, "s2": 0.001, "s3": 0.002, "s4": -0.001},
                     {"s0": 3, "s1": 2, "s2": 2, "s3": 1, "s4": 4}, led.epoch)
    acts: list[str] = []
    bank = core._carrier_bank(tick, led, post, car, acts)
    keys = [b.knob.key for b in bank]
    assert [k for k in keys if k.startswith("s")] == ["s1", "s2"]      # s0: s1's mechanism (lower), s3: 1 seed, s4 < 0
    assert "train.lr" in keys and "dino.schedule_ref_frac" not in keys
    [ev] = store.events("knob_dropped")
    assert ev["key"] == "dino.schedule_ref_frac" and "has no" in ev["why"]
    assert any("dino.schedule_ref_frac dropped" in a for a in acts)
    core._carrier_bank(tick, led, post, car, [])
    assert len(store.events("knob_dropped")) == 1                      # logged once
    tick.close()
    store.close()


def test_a_wave_build_drops_a_bad_knob_and_builds_every_point(ctx, tmp_path):
    """w007: one config knob the carrier lacks failed every point (0/16); now it is dropped (marked used, logged)
    and the rest of the wave is built."""
    sim, store = _two_epochs(ctx, tmp_path)
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, slots=16))
    core = TK.ModuleCore()
    core.knobs = {"inc@new": [_bk("train.weight_decay", 0.05, 0.01, "log", "flag"),
                              _bk("dino.schedule_ref_frac", 0.3, 0.95)]}
    tick = TK.Tick(ctx, store, sim, core, db_path=tmp_path / "loop.db", clock=lambda: 50.0)
    led = CD.load(tmp_path / "loop.db", ctx.task_id)
    acts: list[str] = []
    launched = tick._launch([TK.Action("wave", "inc@new", "factorial", 8.0, 1.0, 8)], led, {}, 50.0, acts)
    assert len([r for r in launched if ":screen:" in r]) == 8 and len(launched) == 8 + F.n_blocks(1)
    wave = F.from_json(store.get_state("wave:w001"))
    assert [k.key for k in wave.knobs] == ["train.weight_decay"]
    assert [e["key"] for e in store.events("knob_dropped")] == ["dino.schedule_ref_frac"]
    assert _bk("dino.schedule_ref_frac", 0.3, 0.95).signature() in store.get_state(TK.ModuleCore.K_USED + "inc@new")
    tick.close()
    store.close()


def test_a_wave_that_built_no_point_is_rearmed_once_without_its_bad_knob(ctx, tmp_path):
    sim, store = _two_epochs(ctx, tmp_path)
    car = "inc@new"
    commit = store.get_node(car).commit
    for nid in ("n1", "n2"):
        _child(store, nid, commit, car, 20.0, nid)
    core = TK.ModuleCore()
    tick = TK.Tick(ctx, store, sim, core, db_path=tmp_path / "loop.db", clock=lambda: 50.0)
    bank = [_bk("n1", 0, 1, "diff", "stacking"), _bk("n2", 0, 1, "diff", "stacking"),
            _bk("train.weight_decay", 0.05, 0.01, "log", "flag"), _bk("dino.schedule_ref_frac", 0.3, 0.95)]
    wave = F.design("w001", car, [b.knob for b in bank], (100114,))
    store.set_state("wave:w001", {**F.to_json(wave), "pre": False})     # w007 as the live loop left it: 0 points
    store.set_state("wave:w001:bank", [b.to_json() for b in bank])
    store.set_state(TK.ModuleCore.K_USED + car, sorted(b.signature() for b in bank))
    store.set_state(tick.K_WAVES, 1)
    led = CD.load(tmp_path / "loop.db", ctx.task_id)
    tick.post = _Post()
    acts: list[str] = []
    tick._close_waves(led, acts)
    res = store.get_state("wave:w001:result")
    assert res["void"] == "no point built" and res["rearmed"] == ["n1", "n2", "train.weight_decay"]
    assert [d["key"] for d in res["dropped"]] == ["dino.schedule_ref_frac"]
    assert store.get_state(TK.ModuleCore.K_USED + car) == [bank[3].signature()]   # the bad knob stays consumed
    assert [e["wave"] for e in store.events("wave_rearmed")] == ["w001"]
    assert any("re-armed" in a for a in acts)
    tick._close_waves(led, [])                                         # once
    assert len(store.events("wave_rearmed")) == 1
    # the re-armed knobs built no point again: rearm_max (1) is spent, they stay consumed
    store.set_state("wave:w002", {**F.to_json(dataclasses.replace(wave, id="w002")), "pre": False})
    store.set_state("wave:w002:bank", [b.to_json() for b in bank[:3]])
    store.set_state(TK.ModuleCore.K_USED + car, sorted(b.signature() for b in bank))
    store.set_state(tick.K_WAVES, 2)
    tick._close_waves(led, [])
    assert store.get_state("wave:w002:result")["rearmed"] == [] and len(store.get_state(TK.ModuleCore.K_USED + car)) == 4
    tick.close()
    store.close()


def test_a_stacking_wave_takes_a_relax_waves_place_and_runs_primary():
    core = TK.ModuleCore()
    core.knobs = {"inc": [_bk("n1", 0, 1, "diff", "stacking"), _bk("n2", 0, 1, "diff", "stacking")],
                  "c2": [_bk("a.x", 1, 2, "lin", "declared")]}
    stack = TK.Action("wave", "inc", "factorial", 16.0, 0.001, 8)
    plain = TK.Action("wave", "c2", "factorial", 8.0, 0.002, 8)
    calls = []

    def ranked(opts, relax_wave=None):                  # the relax pass: one wave at most (relax_max_waves 0 -> none)
        calls.append(([o.node for o in opts], relax_wave))
        return [dataclasses.replace(o, pre=True) for o in opts if o.kind == "wave"][:relax_wave or 1]
    out = core._stack_first([plain, stack], ranked([plain, stack]), ranked)
    assert calls[-1] == (["inc"], 1) and [(a.node, a.pre) for a in out] == [("inc", False)]
    assert any("stack priority" in n for n in core.notes)

    def none_relax(opts, relax_wave=None):              # relax_max_waves 0: the stacking wave still gets one
        return [dataclasses.replace(o, pre=True) for o in opts if o.kind == "wave"][:relax_wave or 0]
    assert [a.node for a in core._stack_first([plain, stack], none_relax([plain, stack]), none_relax)] == ["inc"]
    core.stack_priority = False
    assert [a.node for a in core._stack_first([plain, stack], ranked([plain, stack]), ranked)] == ["c2"]


def test_a_slot_starved_stacking_wave_takes_the_slots_the_primary_pass_leaves():
    """10-13 shadow: the 16-point stacking wave on n1248 never ranked at 5 free slots (rank is all-or-nothing per
    action; live gpu_free was 0-2 on most ticks): the 1-run relax seeds took every slot. Now it takes what the
    primary pass leaves; its other points go first on the next ticks."""
    core = TK.ModuleCore()
    core.knobs = {"inc": [_bk("n1", 0, 1, "diff", "stacking"), _bk("n2", 0, 1, "diff", "stacking")]}
    stack = TK.Action("wave", "inc", "factorial", 16.0, 0.006, 16)     # rate 3.75e-4 >= .5 x the screen's 6.7e-4
    screen = TK.Action("screen", "q", "explore", 3.0, 0.002, 3)
    seeds = [TK.Action("seed", f"s{i}", "exploit", 1.0, 1e-6, 1) for i in range(3)]
    idea = TK.Action("idea", None, "explore", 3.0, 0.0001, 3)
    opts = [screen, stack, *seeds, idea]

    def fake(free):
        def ranked(opts, relax_wave=None, slots=None):   # acquire.rank: all-or-nothing, primary then relax
            left, out = free if slots is None else slots, []
            for o in opts:
                if o.kind == "screen" and o.runs <= left:
                    out.append(o)
                    left -= o.runs
            for o in opts:
                if o.kind == "wave" and (sum(a.kind == "wave" for a in out) >= (relax_wave or 0) or o.value <= 0):
                    continue
                if o.kind in ("wave", "seed") and o.runs <= left:
                    out.append(dataclasses.replace(o, pre=True))
                    left -= o.runs
            return out + [o for o in opts if o.kind == "idea"]
        return ranked
    r5 = fake(5)
    assert not any(a.kind == "wave" for a in core._stack_first(opts, r5(opts), r5))     # free_slots unknown: as before
    out = core._stack_first(opts, r5(opts), r5, 5)
    assert [(a.kind, a.node, a.pre) for a in out] == [("screen", "q", False), ("wave", "inc", False),
                                                     ("idea", None, False)]
    assert "2 of 16 points now" in out[1].why and "stacking: primary" in out[1].why
    r3 = fake(3)                                         # the primary pass takes every slot: the wave waits
    assert not any(a.kind == "wave" for a in core._stack_first(opts, r3(opts), r3, 3))
    nil = [screen, dataclasses.replace(stack, value=0.0), *seeds]   # no value at full size: not a slot question
    assert not any(a.kind == "wave" for a in core._stack_first(nil, r5(nil), r5, 5))
    # v3-c5b S1: below the wave floor (WAVE_FLOOR x the best screen's value/GPU-h) a stacking wave takes no
    # primary slot: neither the slot-starved path nor the relax-pass promotion
    low = [screen, dataclasses.replace(stack, value=0.001), *seeds, idea]
    assert not any(a.kind == "wave" for a in core._stack_first(low, r5(low), r5, 5))
    r40 = fake(40)
    assert all(a.pre for a in core._stack_first(low, r40(low), r40, 40) if a.kind == "wave")
    r40 = fake(40)                                       # fits: the usual relax-wave path, all 16 now
    out = core._stack_first(opts, r40(opts), r40, 40)
    assert [a for a in out if a.kind == "wave"][0].why.endswith("[stacking: primary]")


# -- R1: zombie waves -------------------------------------------------------------------------------------------------
def _wave_with_points(ctx, tmp_path, n=8):
    sim, store = _two_epochs(ctx, tmp_path)
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, slots=16))
    core = TK.ModuleCore()
    core.knobs = {"inc@new": [_bk("train.weight_decay", 0.05, 0.01, "log", "flag")]}
    tick = TK.Tick(ctx, store, sim, core, db_path=tmp_path / "loop.db", clock=lambda: 50.0)
    led = CD.load(tmp_path / "loop.db", ctx.task_id)
    launched = tick._launch([TK.Action("wave", "inc@new", "factorial", 8.0, 1.0, 8)], led, {}, 50.0, [])
    pts = [r for r in launched if ":screen:" in r]                  # + one carrier control per block seed (A2)
    assert len(pts) == n and len(launched) == n + F.n_blocks(1)
    assert all(r.startswith("inc@new:parent:") for r in launched if r not in pts)
    return sim, store, tick, pts


def _finish(store, rid, i):
    y = 0.67 + 0.0005 * (i % 3)
    store.update_run(rid, status=RunStatus.DONE, aggregate=y, scores={k: y for k in CD.PROBE_KEYS.values()},
                     gpu_h=1.0)


def _churn(store, rid):
    run = store.get_run(rid)
    store.update_run(rid, status=RunStatus.VOID, void_type=VoidType.CANCELLED)
    store.log("run_cancelled", run_id=rid, node_id=run.node_id,
              reason="idle fill preempted 2x (cap 2), no free primary slot")


def test_a_voided_point_finishes_its_wave_and_a_recorded_result_closes_it(ctx, tmp_path):
    sim, store, tick, runs = _wave_with_points(ctx, tmp_path)
    for i, rid in enumerate(runs[:-1]):
        _finish(store, rid, i)
    led = CD.load(tmp_path / "loop.db", ctx.task_id)
    assert F.open_waves(led) == {"w001": "inc@new"}                    # one point still running
    _churn(store, runs[-1])
    led = CD.load(tmp_path / "loop.db", ctx.task_id)
    assert F.open_waves(led) == {}                                     # a void finishes its point
    assert F.open_waves(dataclasses.replace(led, runs=tuple(r for r in led.runs if r.id != runs[-1]))) \
        == {"w001": "inc@new"}                                         # a point with no run is still open ...
    assert F.open_waves(dataclasses.replace(led, runs=tuple(r for r in led.runs if r.id != runs[-1])),
                        closed=["w001"]) == {}                         # ... unless the wave has its result
    tick.close()
    store.close()


def test_a_churned_point_is_resubmitted_once_then_abandoned_and_the_wave_closes(ctx, tmp_path):
    sim, store, tick, runs = _wave_with_points(ctx, tmp_path)
    for i, rid in enumerate(runs[:-1]):
        _finish(store, rid, i)
    _churn(store, runs[-1])
    led = CD.load(tmp_path / "loop.db", ctx.task_id)
    tick.post = _Post()
    acts: list[str] = []
    tick._close_waves(led, acts)
    assert store.get_state("wave:w001:result") is None                 # waits for its resubmit
    got = tick._resubmit_churned(led, acts)
    assert got == [runs[-1]] and store.get_run(runs[-1]).status in (RunStatus.SUBMITTING, RunStatus.RUNNING)
    assert store.get_state("wave:w001:resub") == {runs[-1]: 1}
    assert [e["run_id"] for e in store.events("wave_point_resubmitted")] == [runs[-1]]
    led = CD.load(tmp_path / "loop.db", ctx.task_id)
    assert tick._resubmit_churned(led, []) == []                       # running: nothing more
    _churn(store, runs[-1])                                            # churned again: its resubmit is spent
    led = CD.load(tmp_path / "loop.db", ctx.task_id)
    assert tick._resubmit_churned(led, []) == []
    acts = []
    tick._close_waves(led, acts)
    res = store.get_state("wave:w001:result")
    assert res is not None and res["observed"] == 7
    assert [e["run_id"] for e in store.events("wave_point_abandoned")] == [runs[-1]]
    assert any("abandoned" in a for a in acts)
    tick.close()
    store.close()


def test_churn_resubmit_off_abandons_at_once(ctx, tmp_path):
    sim, store, tick, runs = _wave_with_points(ctx, tmp_path)
    tick.ctx = dataclasses.replace(tick.ctx, cfg=dataclasses.replace(
        tick.ctx.cfg, factorial=dataclasses.replace(tick.ctx.cfg.factorial, churn_resubmit_max=0)))
    for i, rid in enumerate(runs[:-1]):
        _finish(store, rid, i)
    _churn(store, runs[-1])
    led = CD.load(tmp_path / "loop.db", ctx.task_id)
    tick.post = _Post()
    tick._close_waves(led, [])
    assert store.get_state("wave:w001:result") is not None and store.events("wave_point_abandoned")
    tick.close()
    store.close()
