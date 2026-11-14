"""10-22 batch C1: the dashboard page was last written 10-08 (loop.dashboard reachable only from the CLI, ``dashboard.every_ticks``
read nowhere). The real Tick renders it every N ticks in a child process and shows the DERIVED PASSED / INCUMBENT."""

from __future__ import annotations

import dataclasses
import json
import re
import time

from loop import dashboard
from loop import tick as TK
from loop.contracts import NodeStatus

from .test_runner import _node
from .test_set1_harness import _Held


class _Proc:
    def __init__(self, rc=None):
        self.rc = rc

    def poll(self):
        return self.rc


def _tick(ctx, store, tmp_path, **dash):
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, dashboard=dataclasses.replace(ctx.cfg.dashboard, **dash)))
    t = TK.Tick(ctx, store, _Held(), TK.ModuleCore(), db_path=tmp_path / "loop.db", clock=time.time)
    t.spawned = []
    t._dashboard_spawn = lambda cmd: (t.spawned.append(cmd), _Proc())[1]
    return t


def _st(tick, **by):
    S = tick.core.ST.Status
    return {nid: S[name.upper()] for nid, name in by.items()}


def test_the_real_tick_renders_every_n_ticks_one_child_at_a_time(ctx, store, tmp_path):
    tick = _tick(ctx, store, tmp_path, every_ticks=3, min_interval_s=0.0)
    now = time.time()
    outs = [tick._dashboard({}, now + i, []) for i in range(3)]
    assert outs == [False, False, True] and len(tick.spawned) == 1
    cmd = tick.spawned[0]
    assert cmd[1:4] == ["-m", "loop", "dashboard"] and "--task" in cmd and ctx.state_dir in cmd
    assert [tick._dashboard({}, now + 10 + i, []) for i in range(3)] == [False, False, False]   # still rendering
    assert len(tick.spawned) == 1
    tick._dash_proc = _Proc(rc=0)                              # the child finished
    assert tick._dashboard({}, now + 20, []) is True and len(tick.spawned) == 2   # due ticks piled up while it ran
    tick.close()


def test_min_interval_and_off_and_dry(ctx, store, tmp_path):
    tick = _tick(ctx, store, tmp_path, every_ticks=1, min_interval_s=900.0)
    now = time.time()
    assert tick._dashboard({}, now, []) is True and tick._dashboard({}, now + 5, []) is False
    tick._dash_proc = None
    assert tick._dashboard({}, now + 901, []) is True
    off = _tick(ctx, store, tmp_path, every_ticks=0)
    assert off._dashboard({}, now, []) is False and not off.spawned
    tick.close()
    off.close()


def test_a_failed_child_is_reported(ctx, store, tmp_path):
    tick = _tick(ctx, store, tmp_path, every_ticks=1, min_interval_s=0.0)
    tick._dash_proc, acts = _Proc(rc=2), []
    tick._dashboard({}, time.time(), acts)
    assert any("exited 2" in a for a in acts) and store.events("dashboard_error")
    tick.close()


def test_derived_passed_and_incumbent_reach_the_page_but_explicit_acts_do_not(ctx, store, tmp_path):
    tick = _tick(ctx, store, tmp_path, every_ticks=1, min_interval_s=0.0)
    store.ensure_root("abc123")
    for nid in ("p", "i", "q", "conf"):
        _node(store, nid)
    store.update_node("conf", status=NodeStatus.CONFIRMED)
    tick._dashboard(_st(tick, p="passed", i="incumbent", q="queued", conf="passed"), time.time(), [])
    assert store.get_state(dashboard.K_STATUS) == {"p": "passed", "i": "incumbent"}
    page = dashboard.render(ctx, store, None)
    data = json.loads(re.search(r'id="loop-data">(.*?)</script>', page, re.S).group(1))
    st = {n["id"]: n["st"] for n in data["nodes"]}
    assert st == {"p": "passed", "i": "incumbent", "q": "queued", "conf": "confirmed"}
    assert {"passed", "incumbent"} <= set(data["statuses"])
    assert '<option value="passed">' in page
    tick.close()


def test_the_cleared_overrides_leave_the_stored_status(ctx, store):
    store.ensure_root("abc123")
    _node(store, "q")
    assert json.loads(re.search(r'id="loop-data">(.*?)</script>', dashboard.render(ctx, store, None), re.S).group(1)
                      )["nodes"][0]["st"] == "queued"
