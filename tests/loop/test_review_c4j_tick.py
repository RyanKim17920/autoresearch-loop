"""c4j review (M2, S9): the tick time box never starves wave builds or submits (>= heartbeat.min_wave_points points and
>= heartbeat.min_submits submits per tick); a paused build is counted (paused_n) and quarantined after
heartbeat.wave_pause_quarantine paused ticks with no new point; the levers summary.json scan is time-boxed and
disk-cached by mtime."""

from __future__ import annotations

import dataclasses
import json
import os
import time
import types

from loop import tick as TK
from loop.core import data as CD
from loop.core import factorial as F
from loop.core import levers as LV

from .test_port import _two_epochs
from .test_waves import _bank


def _tick(ctx, tmp_path):
    sim, store = _two_epochs(ctx, tmp_path)
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, slots=16))
    core = TK.ModuleCore()
    core.knobs = {"inc@new": _bank()}
    return TK.Tick(ctx, store, sim, core, db_path=tmp_path / "loop.db", clock=lambda: 50.0), store


def test_a_tick_past_its_budget_still_builds_a_wave_point_and_submits(ctx, tmp_path, monkeypatch):
    tick, store = _tick(ctx, tmp_path)
    monkeypatch.setattr(tick, "over_budget", lambda: True)          # the budget was gone before the launch phase
    led = CD.load(tmp_path / "loop.db", ctx.task_id)
    acts: list[str] = []
    assert tick._launch([TK.Action("wave", "inc@new", "factorial", 8.0, 1.0, 8)], led, {}, 50.0, acts) == []
    job = store.get_state(TK.K_WAVE_BUILD)
    assert job["paused"] and len(job["built"]) == 1 and job["stall"] == 0     # the floor: one point, progress
    for k in range(2, 9):                                                     # one point per starved tick ...
        tick._wpts = 0
        led = CD.load(tmp_path / "loop.db", ctx.task_id)
        acts = []
        launched = tick._launch([], led, {}, 50.0 + k, acts)
        if store.get_state(TK.K_WAVE_BUILD) is None:
            break
        assert store.get_state(TK.K_WAVE_BUILD)["paused_n"] == k - 1           # paused tries counted
    assert store.get_state(TK.K_WAVE_BUILD) is None and store.get_state("wave:w001") is not None
    # ... and the submits still go out: >= min_submits (8) although over budget
    assert len(launched) == 8
    tick.close()
    store.close()


def test_a_wave_paused_with_no_progress_is_quarantined(ctx, tmp_path, monkeypatch):
    tick, store = _tick(ctx, tmp_path)
    led = CD.load(tmp_path / "loop.db", ctx.task_id)
    monkeypatch.setattr(tick, "over_budget", lambda: True)
    monkeypatch.setattr(F, "build_point", lambda *a, **k: ("", "boom"))       # every point fails: no progress
    tick._launch([TK.Action("wave", "inc@new", "factorial", 8.0, 1.0, 8)], led, {}, 50.0, [])
    cap = tick.ctx.cfg.heartbeat.wave_pause_quarantine
    acts: list[str] = []
    for k in range(cap + 2):
        if store.get_state(TK.K_WAVE_BUILD) is None:
            break
        tick._wpts = 0
        acts = []
        tick._launch([], CD.load(tmp_path / "loop.db", ctx.task_id), {}, 60.0 + k, acts)
    assert store.get_state(TK.K_WAVE_BUILD) is None
    assert any("QUARANTINED" in a for a in acts)
    ev = store.events("wave_quarantined")
    assert ev and ev[0]["stall"] >= cap and ev[0]["built"] == 0
    tick.close()
    store.close()


def _ctx(tmp_path, scan_s):
    from loop.spec import IdeationCfg
    return types.SimpleNamespace(state_dir=str(tmp_path / "state"),
                                 cfg=types.SimpleNamespace(ideation=dataclasses.replace(IdeationCfg(),
                                                                                         levers_scan_s=scan_s)))


def test_levers_summary_scan_is_disk_cached_by_mtime(tmp_path):
    LV._SCAN.clear()
    d = tmp_path / "run1"
    d.mkdir()
    (d / "summary.json").write_text(json.dumps({"flop_fraction": 0.5}))
    ctx = _ctx(tmp_path, 20.0)
    assert LV.summaries(ctx, [str(d)])[str(d)] == {"flop_fraction": 0.5}
    cache = json.loads((tmp_path / "state" / "cache" / "summaries.json").read_text())
    assert cache[str(d)][2] == {"flop_fraction": 0.5}
    LV._SCAN.clear()                                   # a restart: served from disk, the file not re-read ...
    reads = []
    orig = LV.Path.read_text

    def counting(self, *a, **k):
        reads.append(str(self))
        return orig(self, *a, **k)
    try:
        LV.Path.read_text = counting
        assert LV.summaries(ctx, [str(d)])[str(d)] == {"flop_fraction": 0.5}
        assert not [r for r in reads if r.endswith("run1/summary.json")]
        (d / "summary.json").write_text(json.dumps({"flop_fraction": 0.75, "pad": 1}))   # ... until it changes
        st = os.stat(d / "summary.json")
        os.utime(d / "summary.json", ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
        assert LV.summaries(ctx, [str(d)])[str(d)]["flop_fraction"] == 0.75
    finally:
        LV.Path.read_text = orig


def test_levers_summary_scan_past_its_time_box_serves_the_cache(tmp_path, monkeypatch):
    LV._SCAN.clear()
    dirs = []
    for i in range(3):
        d = tmp_path / f"r{i}"
        d.mkdir()
        (d / "summary.json").write_text(json.dumps({"i": i}))
        dirs.append(str(d))
    ctx = _ctx(tmp_path, 20.0)
    LV.summaries(ctx, dirs[:1])                         # r0 cached
    ctx = _ctx(tmp_path, 1e-9)                          # a time box already spent at the first dir
    clock = iter([0.0] + [10.0] * 10)
    monkeypatch.setattr(LV.time, "monotonic", lambda: next(clock))
    got = LV.summaries(ctx, dirs)
    assert got[dirs[0]] == {"i": 0} and got[dirs[1]] == {} and got[dirs[2]] == {}
