"""Factorial waves from the knob sources (docs/loop-core-v3.md §5): the retro over the ledger's historical factorials
seeds a carrier's first wave (significant legs as diff knobs, config moves the carrier lacks, the edge rule for moves it
holds), acquisition prices one wave per carrier by its x*'s expected gain, and the tick designs it (old-epoch diff
knobs via their ports onto the carrier), launches its points on a fresh SELECT block and closes it into x*."""

from __future__ import annotations

import dataclasses
import json
import random

import yaml

from loop import tick as TK
from loop.contracts import Run, RunStatus, Stage
from loop.core import data as CD
from loop.core import factorial as F
from loop.core import model as M

from .test_port import _two_epochs

LINEAGE = ("inc", "root")


def _eff(key, lo, hi, beta, se=0.001, wave="d1", carrier="root", scale=None):
    kb = F.Knob(key, lo, hi, scale or ("diff" if lo == 0 and hi == 1 and key.startswith("n") else F.auto_scale(lo, hi)))
    return F.RetroEffect(wave, carrier, kb, beta, se, 16)


def test_retro_bank_maps_significant_effects_onto_the_carrier():
    cfg = {"a.layer": 0.8, "a.kde": 6.667, "a.crop": [0.08, 0.35], "a.mean": 0.5, "a.weak": 1.0, "a.hed": 0.07}
    effects = [
        _eff("n1", 0, 1, +0.004, wave="L4", carrier="inc"),           # a good leg on the incumbent: diff knob
        _eff("n2", 0, 1, -0.004, wave="L4", carrier="inc"),           # a harmful leg: dropped
        _eff("n3", 0, 1, +0.004, wave="L6", carrier="other"),         # not on the carrier's lineage: ignored
        _eff("a.layer", 0.7, 0.8, +0.004), _eff("a.layer", 0.7, 0.8, +0.003, wave="d2"),   # pooled; holds .8: out
        _eff("a.kde", 10.0, 6.667, +0.0035),                          # holds 6.667 (log): one step out -> 4.445
        _eff("a.crop", [0.08, 0.35], [0.16, 0.35], +0.003),           # carrier at the worse level: that range
        _eff("a.hed", 0.07, 0.04667, +0.003),
        _eff("a.mean", 0.1, 0.2, +0.004),                             # carrier at .5: off the tested range
        _eff("a.gone", 1, 2, +0.004),                                 # the carrier has no such key
        _eff("a.weak", 1.0, 2.0, +0.001),                             # |z| 1: not significant
    ]
    bank, lines = F.retro_bank(effects, LINEAGE, cfg, tau2=2.5e-6)
    got = {b.knob.key: b for b in bank}
    assert set(got) == {"n1", "a.layer", "a.kde", "a.crop", "a.hed"}
    assert got["n1"].knob == F.Knob("n1", 0, 1, "diff") and got["n1"].m > 0
    assert got["a.layer"].knob == F.Knob("a.layer", 0.8, 0.9, "lin") and got["a.layer"].m == 0.0
    assert got["a.kde"].knob == F.Knob("a.kde", 6.667, 4.445, "log")
    assert got["a.crop"].knob.lo == [0.08, 0.35] and got["a.crop"].knob.hi == [0.16, 0.35] and got["a.crop"].m > 0
    assert got["a.hed"].knob.lo == 0.07 and got["a.hed"].knob.coded(0.07) == -1.0       # row 0 = the carrier
    assert [b.m for b in bank] == sorted((b.m for b in bank), reverse=True)
    assert any("a.mean" in x and "off the tested range" in x for x in lines)
    assert any("a.gone" in x and "has no a.gone" in x for x in lines)
    assert any("n2" in x and "harmful" in x for x in lines)
    old, _ = F.retro_bank(effects, LINEAGE, cfg, tau2=2.5e-6, w=0.5)                 # an older base: discounted
    assert {b.knob.key: b.m for b in old}["n1"] < got["n1"].m


def test_retro_recovers_planted_main_effects_of_a_historical_wave():
    rng = random.Random(4)
    rows, _ = F.matrix(3)
    beta = {"a.lr": 0.004, "a.wd": 0.0, "n7": -0.003}
    nodes, obs = {}, []
    for i, row in enumerate(rows):
        nid = f"p{i}"
        nodes[nid] = CD.Node(nid, "inc", None, "design", "design", "factorial", "inc", 1.0, "screening", "")
        for seed in (611, 612):
            y = 0.67 + sum(b * x / 2 for b, x in zip(beta.values(), row)) + 0.001 * seed % 3 * 1e-3 \
                + rng.gauss(0, 0.0004)
            obs.append(CD.Obs(f"{nid}:screen:{seed}", nid, seed, "screen", CD.SELECT, {"cls": y}, y, 1.0, 0.0, 0))
    led = CD.Ledger(nodes, (), tuple(obs), frozenset(), ((0.0, "root"),))
    legs = {"batches": [{"batch": 9, "carrier": "inc", "seeds": [611, 612], "designs": {"legs": {
        "factors": [{"key": "a.lr", "low": 1.0, "high": 2.0, "kind": "own"},
                    {"key": "a.wd", "low": 0.1, "high": 0.2, "kind": "own"},
                    {"key": "data.aug_leg_n7", "low": 0, "high": 1, "kind": "new"}],
        "rows": [list(r) for r in rows], "points": {str(i): f"p{i}" for i in range(len(rows))}}}}]}
    [w] = F.hist_waves(legs, None)
    assert w.name == "L9-legs" and w.knobs[2] == F.Knob("n7", 0, 1, "diff") and w.knobs[0].scale == "log"
    eff = {e.knob.key: e for e in F.retro(led, [w], M.Noise({M.COMPOSITE: 4e-7}, {M.COMPOSITE: 0.0}), {"cls": 1.0})}
    for k, b in beta.items():
        assert abs(eff[k].beta - b) < 4 * eff[k].se
    assert eff["a.lr"].z > 4 and eff["n7"].z < -4


def test_forecast_prices_a_wave_by_its_x_stars_expected_gain():
    s2 = (1 - 0.45) * 0.0032 ** 2
    good = [F.BankKnob(F.Knob(f"k{i}", 0, 1, "diff"), 0.003, 1e-6, "retro") for i in range(3)]
    flat = [F.BankKnob(F.Knob(f"k{i}", 0, 1, "diff"), 0.0, 1e-6, "flag") for i in range(3)]
    m, v, big_v, n = F.forecast(good, s2)
    assert n == 8 and 0 < m < 3 * 0.003 and big_v == 3 * 4 * s2 / 8
    assert 0 < F.forecast(flat, s2)[0] < m                 # uncertain knobs: a positive part only
    assert F.forecast(good + flat + good[:1], s2)[3] == 16


def _bank():
    return [F.BankKnob(F.Knob("train.weight_decay", 0.05, 0.01, "log"), 0.002, 1e-6, "retro"),
            F.BankKnob(F.Knob("x", 0, 1, "diff"), 0.001, 1e-6, "retro")]


def _seed(n) -> int:
    return int(F._TAG.search(n.notes).group(4))


class _Post:
    weights = {"cls": 1.0}
    noise = M.Noise({M.COMPOSITE: 1e-5}, {M.COMPOSITE: 0.4})
    prior = M.Prior({(): 0.0}, {(): 4e-6})


def test_tick_designs_launches_and_closes_a_wave_into_x_star(ctx, tmp_path):
    sim, store = _two_epochs(ctx, tmp_path)
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, slots=16))
    core = TK.ModuleCore()
    core.knobs = {"inc@new": _bank()}
    tick = TK.Tick(ctx, store, sim, core, db_path=tmp_path / "loop.db", clock=lambda: 50.0)
    led = CD.load(tmp_path / "loop.db", ctx.task_id)
    acts: list[str] = []
    launched = tick._launch([TK.Action("wave", "inc@new", "factorial", 8.0, 1.0, 8)], led, {}, 50.0, acts)
    port = f"x@{led.epoch[:7]}"
    assert any(f"knob x via its port {port} on inc@new" in a for a in acts)
    pn = store.get_node(port)
    assert pn.parent_id == "inc@new" and "port_of=x" in pn.notes and "for_wave=w001" in pn.notes
    [blk] = [b for b in store.seed_blocks() if b.id == "wave-w001"]
    # 2 knobs: each replicate of the 4 cells is a block with its own fresh seed (v3-c7a A2), + a carrier control each
    assert blk.pool == CD.SELECT and len(blk.seeds) == 2 and min(blk.seeds) >= ctx.cfg.seeds.fresh_base
    pts = {i: store.get_node(f"w001.p{i:02d}") for i in range(8)}
    assert sorted(launched) == sorted([f"w001.p{i:02d}:screen:{_seed(n)}" for i, n in pts.items()]
                                      + [f"inc@new:parent:{s}" for s in blk.seeds])
    assert {(json.dumps(F.from_json(store.get_state("wave:w001")).points[i], sort_keys=True), _seed(n))
            for i, n in pts.items()}.__len__() == 8                    # no config twice at one seed
    wave = F.from_json(store.get_state("wave:w001"))
    for i, n in pts.items():
        cfg = yaml.safe_load(sim._object(n.commit)["files"]["config.yaml"])
        assert cfg["train"]["weight_decay"] == wave.points[i]["train.weight_decay"]
        assert cfg["aug"]["strength"] == (0.3 if wave.points[i][port] == 1 else 0.5)
        assert cfg["train"]["lr"] == 7e-4                                  # on the carrier
    led = CD.load(tmp_path / "loop.db", ctx.task_id)
    assert F.open_waves(led) == {"w001": "inc@new"}
    rng = random.Random(1)
    for i, n in pts.items():                                               # wd .01 helps +.004, the port nothing
        y = 0.67 + (0.004 if wave.points[i]["train.weight_decay"] == 0.01 else 0.0) + rng.gauss(0, 0.0003)
        store.update_run(Run.make_id(n.id, Stage.SCREEN, _seed(n)), status=RunStatus.DONE, aggregate=y,
                         scores={k: y for k in CD.PROBE_KEYS.values()}, gpu_h=1.0)
    led = CD.load(tmp_path / "loop.db", ctx.task_id)
    assert F.open_waves(led) == {}
    tick.post = _Post()
    acts = []
    tick._close_waves(led, acts)
    res = store.get_state("wave:w001:result")
    eff = {e["key"]: e for e in res["effects"]}
    assert eff["train.weight_decay"]["beta"] > 0.003 and abs(eff[port]["beta"]) < 0.002
    x = store.get_node("w001.x")
    assert x is not None and x.parent_id == "inc@new" and F.node_prior(x.notes)[0] > 0
    assert yaml.safe_load(sim._object(x.commit)["files"]["config.yaml"])["train"]["weight_decay"] == 0.01
    assert any(a.startswith("wave w001 closed") for a in acts)
    assert [e["wave"] for e in store.events("wave_result")] == ["w001"]
    nxt = store.get_state("factorial.next:inc@new")
    assert nxt and nxt[0]["key"] == "train.weight_decay" and nxt[0]["lo"] == 0.01      # the edge rule: one step out
    tick._close_waves(led, acts)
    assert len(store.events("wave_result")) == 1                          # analysed once
    tick.close()
    store.close()


def test_an_x_star_build_that_raises_never_stops_the_tick(ctx, tmp_path, monkeypatch):
    """v3-rc ops smoke: build_point raising inside the wave close was unguarded -- every tick died the same way with
    no call site to quarantine (a crash loop). Now: logged, retried next tick, the x* built once."""
    sim, store = _two_epochs(ctx, tmp_path)
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, slots=16))
    core = TK.ModuleCore()
    core.knobs = {"inc@new": _bank()}
    tick = TK.Tick(ctx, store, sim, core, db_path=tmp_path / "loop.db", clock=lambda: 50.0)
    led = CD.load(tmp_path / "loop.db", ctx.task_id)
    tick._launch([TK.Action("wave", "inc@new", "factorial", 8.0, 1.0, 8)], led, {}, 50.0, [])
    [blk] = [b for b in store.seed_blocks() if b.id == "wave-w001"]
    wave = F.from_json(store.get_state("wave:w001"))
    rng = random.Random(1)
    for i in range(8):
        y = 0.67 + (0.004 if wave.points[i]["train.weight_decay"] == 0.01 else 0.0) + rng.gauss(0, 0.0003)
        store.update_run(Run.make_id(f"w001.p{i:02d}", Stage.SCREEN, _seed(store.get_node(f"w001.p{i:02d}"))),
                         status=RunStatus.DONE,
                         aggregate=y, scores={k: y for k in CD.PROBE_KEYS.values()}, gpu_h=1.0)
    led = CD.load(tmp_path / "loop.db", ctx.task_id)
    tick.post = _Post()
    real = F.build_point

    def boom(*a, **k):
        raise OSError("disk full mid-build")
    monkeypatch.setattr(F, "build_point", boom)
    acts: list[str] = []
    tick._close_waves(led, acts)                                           # no raise out of the tick
    assert store.get_state("wave:w001:result") is None and store.get_node("w001.x") is None
    assert [e["key"] for e in store.events("tick_call_error")] == ["wave.close:w001"]
    monkeypatch.setattr(F, "build_point", real)
    tick._close_waves(led, acts)                                           # retried: closes, x* built
    assert store.get_state("wave:w001:result")["next"]["node"] == "w001.x" and store.get_node("w001.x") is not None
    tick.close()
    store.close()


def test_a_wave_consumes_its_knobs_atomically_with_its_creation(ctx, tmp_path):
    sim, store = _two_epochs(ctx, tmp_path)
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, slots=16))
    core = TK.ModuleCore()
    bank = [F.BankKnob(F.Knob("train.weight_decay", 0.05, 0.01, "log"), 0.002, 1e-6, "flag"),
            F.BankKnob(F.Knob("aug.strength", 0.5, 0.3), 0.001, 1e-6, "declared")]
    core.knobs = {"inc@new": bank}
    tick = TK.Tick(ctx, store, sim, core, db_path=tmp_path / "loop.db", clock=lambda: 50.0)
    led = CD.load(tmp_path / "loop.db", ctx.task_id)

    def boom(*a, **k):                                     # a crash inside the wave's transaction
        raise RuntimeError("crash")
    real, tick._put_block = tick._put_block, boom
    try:
        tick._launch([TK.Action("wave", "inc@new", "factorial", 8.0, 1.0, 8)], led, {}, 50.0, [])
    except RuntimeError:
        pass
    assert store.get_state("wave:w001") is None and store.get_node("w001.p00") is None    # all or nothing
    assert store.get_state(TK.ModuleCore.K_USED + "inc@new") is None and not store.seed_blocks()
    tick._put_block = real
    launched = tick._launch([TK.Action("wave", "inc@new", "factorial", 8.0, 1.0, 8)], led, {}, 50.0, [])
    assert len(launched) == 8 + 2 and store.get_state("wave:w001") is not None      # + 2 carrier controls
    used = store.get_state(TK.ModuleCore.K_USED + "inc@new")
    assert sorted(used) == sorted(b.signature() for b in bank)
    assert F.unused(bank, used) == []                      # prepare never offers them on this carrier again
    tick.close()
    store.close()


def test_the_retro_seed_is_rebuilt_trimmed_and_its_wave_comes_alone(ctx, tmp_path, monkeypatch):
    sim, store = _two_epochs(ctx, tmp_path)
    core = TK.ModuleCore()
    tick = TK.Tick(ctx, store, sim, core, db_path=tmp_path / "loop.db", clock=lambda: 50.0)
    led = CD.load(tmp_path / "loop.db", ctx.task_id)
    retro = [F.BankKnob(F.Knob(f"a.k{i}", 0.1, 0.2), 0.001, 1e-6, "retro", "", z) for i, z in
             enumerate((3.5, 2.5, 5.0, 3.2, 3.1, 4.0))]
    store.set_state(core.K_BANK + "inc@new", {"knobs": [b.to_json() for b in retro], "lines": [], "tau2": 2e-6,
                                              "effects": 6, "used": False})               # an old (v1) seed
    monkeypatch.setattr(core, "_retro_seed", lambda *a: {"knobs": [b.to_json() for b in F.retro_trim(retro)],
                                                          "lines": [], "tau2": 2e-6, "effects": 6, "used": False,
                                                          "v": core.RETRO_V})
    monkeypatch.setattr(core, "_config", lambda *a: {f"a.k{i}": 0.1 for i in range(6)})   # keys the carrier holds
    acts: list[str] = []
    bank = core._carrier_bank(tick, led, _Post(), "inc@new", acts)
    assert [b.knob.key for b in bank] == ["a.k2", "a.k5", "a.k0", "a.k3"] and acts       # rebuilt; the retro wave alone
    store.set_state(core.K_BANK + "inc@new", {**store.get_state(core.K_BANK + "inc@new"), "used": "w001"})
    assert all(b.source != "retro" for b in core._carrier_bank(tick, led, _Post(), "inc@new", []))
    tick.close()
    store.close()


def test_a_relax_wave_stays_preemptible_only_when_its_points_resume(ctx, tmp_path):
    """10-13 (w002 held 5 of 8 primary slots): a relax-pass wave's ``pre`` flag is persisted in ``wave:<id>`` and
    every point it launches -- now or resumed by a later tick's open-wave pass -- is fill-only (preemptible)."""
    sim, store = _two_epochs(ctx, tmp_path)
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, slots=3))
    core = TK.ModuleCore()
    core.knobs = {"inc@new": _bank()}
    tick = TK.Tick(ctx, store, sim, core, db_path=tmp_path / "loop.db", clock=lambda: 50.0)
    led = CD.load(tmp_path / "loop.db", ctx.task_id)
    act = dataclasses.replace(TK.Action("wave", "inc@new", "factorial", 8.0, 1.0, 8), pre=True)
    first = tick._launch([act], led, {}, 50.0, [])
    assert store.get_state("wave:w001")["pre"] is True and 0 < len(first) < 8
    assert {(r.split(":")[0], int(r.split(":")[2])) for r in first} <= tick.book.fill_only
    for r in first:                                                        # slots free again
        store.update_run(r, status=RunStatus.DONE, aggregate=0.67, scores={k: 0.67 for k in CD.PROBE_KEYS.values()},
                         gpu_h=1.0)
    led = CD.load(tmp_path / "loop.db", ctx.task_id)
    later = tick._launch([], led, {}, 60.0, [])                            # the open wave's unlaunched points
    assert later and set(later).isdisjoint(first)
    assert {(r.split(":")[0], int(r.split(":")[2])) for r in later} <= tick.book.fill_only
    tick.close()
    store.close()


def test_a_ranked_wave_is_not_fill_only(ctx, tmp_path):
    sim, store = _two_epochs(ctx, tmp_path)
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, slots=16))
    core = TK.ModuleCore()
    core.knobs = {"inc@new": _bank()}
    tick = TK.Tick(ctx, store, sim, core, db_path=tmp_path / "loop.db", clock=lambda: 50.0)
    led = CD.load(tmp_path / "loop.db", ctx.task_id)
    tick._launch([TK.Action("wave", "inc@new", "factorial", 8.0, 1.0, 8)], led, {}, 50.0, [])
    assert store.get_state("wave:w001")["pre"] is False and not tick.book.fill_only
    tick.close()
    store.close()


def test_a_wave_build_past_the_tick_budget_pauses_and_resumes_without_rebuilding(ctx, tmp_path, monkeypatch):
    """10-13: wave builds took 7-10 min of a tick. Past ``heartbeat.tick_budget_s`` the build pauses: the journal keeps
    the points built so far (``built``), the next tick resumes it -- each point built once, the pause no crash try."""
    sim, store = _two_epochs(ctx, tmp_path)
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, slots=16))
    core = TK.ModuleCore()
    core.knobs = {"inc@new": _bank()}
    tick = TK.Tick(ctx, store, sim, core, db_path=tmp_path / "loop.db", clock=lambda: 50.0)
    calls = []
    real = F.build_point

    def counting(*a, **k):
        calls.append(a[5])
        return real(*a, **k)
    monkeypatch.setattr(F, "build_point", counting)
    budget = {"over": False}
    monkeypatch.setattr(tick, "over_budget", lambda: budget["over"] or len(calls) >= 3)
    led = CD.load(tmp_path / "loop.db", ctx.task_id)
    acts: list[str] = []
    assert tick._launch([TK.Action("wave", "inc@new", "factorial", 8.0, 1.0, 8)], led, {}, 50.0, acts) == []
    job = store.get_state(TK.K_WAVE_BUILD)
    assert job["paused"] and len(job["built"]) == 3 and job["tries"] == 0 and store.get_state("wave:w001") is None
    assert any("build paused (tick budget)" in a for a in acts)
    monkeypatch.setattr(tick, "over_budget", lambda: False)
    led = CD.load(tmp_path / "loop.db", ctx.task_id)
    acts = []
    launched = tick._launch([], led, {}, 60.0, acts)
    assert len(launched) == 8 + 2 and store.get_state(TK.K_WAVE_BUILD) is None      # + 2 carrier controls
    assert sorted(calls) == sorted(set(calls)) and len(calls) == 8                  # every point built once
    tick.close()
    store.close()


def test_tick_budget_defers_submits_and_new_waves(ctx, tmp_path, monkeypatch):
    sim, store = _two_epochs(ctx, tmp_path)
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, slots=16))
    core = TK.ModuleCore()
    core.knobs = {"inc@new": _bank()}
    tick = TK.Tick(ctx, store, sim, core, db_path=tmp_path / "loop.db", clock=lambda: 50.0)
    monkeypatch.setattr(tick, "over_budget", lambda: True)
    tick._wpts = 1                                     # this tick already built its min_wave_points (c4j floor)
    led = CD.load(tmp_path / "loop.db", ctx.task_id)
    acts: list[str] = []
    assert tick._launch([TK.Action("wave", "inc@new", "factorial", 8.0, 1.0, 8)], led, {}, 50.0, acts) == []
    assert any("not started (tick budget spent)" in a for a in acts) and store.get_state(TK.K_WAVE_BUILD) is None
    tick.phase_s = {"poll": 11.2, "launch": 400.0, "launch.waves": 390.4, "ideate": 0.1}
    assert tick.timing_text() == " phase_s=poll:11,launch:400,launch.waves:390"
    tick.close()
    store.close()
