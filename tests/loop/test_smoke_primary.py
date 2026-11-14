"""11-08 nanobrain smoke starvation: primary (8 slots) was full of 2 h screens, so the "auto" smoke rule sent every smoke
to the contended idle pool, where it sat PENDING behind other users' jobs for the whole hour (queue timeout, no verdict,
an Opus session held open for that hour, then research idled on ``smokecap=probe``). ``slurm.smoke_pool: primary``
queues smokes on primary and holds new primary submits while one waits; ``smoke_primary_hold`` keeps the reserve slot
there; ``smoke_wait_s`` / ``smoke_session_wait_s`` bound the queue wait (post-session / inside the session's tool)."""

from __future__ import annotations

import dataclasses
import json
from collections import Counter
from types import SimpleNamespace

from loop import runner as R
from loop.agents.researcher import WorktreeTools
from loop.contracts import SmokeResult, Worktree
from loop.spec import PoolCfg, SlurmCfg
from tests.loop.test_adapters import SlurmRunner, base_repo, make_np, np_ctx_with_locks  # noqa: F401

POOLS = (PoolCfg("primary", slots=8, qos="normal"), PoolCfg("idle", slots=56, qos="low", preemptible=True))


def _sctx(ctx, **slurm):
    return dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, slurm=SlurmCfg(pools=POOLS, **slurm)))


def test_defaults_keep_the_old_routing():
    s = SlurmCfg()
    assert s.smoke_pool == "auto" and s.smoke_primary_hold is False
    assert s.smoke_wait_s == 0.0 and s.smoke_session_wait_s == 0.0


def test_auto_sends_a_smoke_to_a_free_idle_pool_primary_mode_to_primary(ctx, store):
    smokes = Counter()
    auto = R.RunBook(_sctx(ctx), store, SimpleNamespace(smokes=smokes))
    auto.pool_states = {"primary": (8, 0), "idle": (40, 0)}
    assert auto.smoke_pool() == "idle"
    prim = R.RunBook(_sctx(ctx, smoke_pool="primary"), store, SimpleNamespace(smokes=smokes))
    prim.pool_states = {"primary": (8, 0), "idle": (0, 9)}                # the live 11-08 state: idle 0r/9p
    assert prim.smoke_on_primary() and prim.smoke_pool() == "primary" and prim._smoke_home() == "primary"
    smokes["primary"] = 2                                                  # smoke_primary_max (2) reached: auto fallback
    assert prim.smoke_pool() == "idle"


def test_primary_mode_needs_a_primary_pool(ctx, store):
    only_idle = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, slurm=SlurmCfg(
        pools=(PoolCfg("idle", slots=56, preemptible=True),), smoke_pool="primary")))
    book = R.RunBook(only_idle, store, SimpleNamespace(smokes=Counter()))
    assert not book.smoke_on_primary() and book.smoke_pool() == "idle"


def test_a_smoke_pending_on_primary_holds_new_primary_submits(ctx, store):
    smokes = Counter({"primary": 1})
    book = R.RunBook(_sctx(ctx, smoke_pool="primary"), store, SimpleNamespace(smokes=smokes))
    book.pool_states = {"primary": (0, 0), "idle": (0, 0)}
    book.smoke_pending = {"primary": 1}
    assert book.used()["primary"] == 1                    # the smoke holds its slot
    assert book.limits()["primary"] == 1                  # no new run submitted there while it waits
    assert "primary" not in book._open(False)
    book.smoke_pending = {}                               # it started: the rest of primary is open again
    assert "primary" in book._open(False)


def test_hold_keeps_the_reserve_slot_on_primary_with_no_smoke_in_flight(ctx, store):
    nohold = R.RunBook(_sctx(ctx, smoke_pool="primary"), store, SimpleNamespace(smokes=Counter()))
    nohold.pool_states = {"primary": (0, 0), "idle": (0, 0)}
    u = nohold.used()
    assert u["primary"] == 0 and u["idle"] == 1           # the c4j rule: never idle a primary slot
    hold = R.RunBook(_sctx(ctx, smoke_pool="primary", smoke_primary_hold=True), store,
                     SimpleNamespace(smokes=Counter()))
    hold.pool_states = {"primary": (0, 0), "idle": (0, 0)}
    u = hold.used()
    assert u["primary"] == 1 and u["idle"] == 0
    hold.adapter.smokes["primary"] = 1                    # the smoke takes the held slot: no second one held
    assert hold.used()["primary"] == 1


def _np(tmp_path, base_repo, clock, **slurm):  # noqa: F811
    runner = SlurmRunner({"squeue": lambda argv: (0, "PENDING\n", "")})
    ctx = np_ctx_with_locks(tmp_path / "sp")
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, slurm=SlurmCfg(pools=POOLS, **slurm)))
    adapter, _ = make_np(ctx, base_repo, runner=runner, clock=lambda: clock["t"])
    return adapter, runner


def _wait(adapter, clock, **kw):
    import loop.adapters.nanopath_v2 as NV

    def tick(_s):
        clock["t"] += 600.0
    orig, NV.time.sleep = NV.time.sleep, tick
    try:
        return adapter._smoke_wait("77", **kw)
    finally:
        NV.time.sleep = orig


def test_queue_wait_is_yaml_bounded_and_never_a_failure(tmp_path, base_repo):  # noqa: F811
    clock = {"t": 7000.0}
    adapter, runner = _np(tmp_path, base_repo, clock)
    assert adapter.smoke_queue_s() == 3600.0                      # default: SMOKE_WAIT_S
    adapter, runner = _np(tmp_path, base_repo, clock, smoke_wait_s=10800)
    assert adapter.smoke_queue_s() == 10800.0 and adapter.smoke_queue_s(900) == 900.0
    t0 = clock["t"]
    rc, why = _wait(adapter, clock, queue_s=900)
    assert rc == 124 and "queue timeout" in why                   # rc 124 = no verdict (smoke_unavailable), not a fail
    assert clock["t"] - t0 <= 1800                                # scancelled after ~15 min, not the hour
    assert runner.argvs("scancel")
    assert [r["kind"] for r in json.loads((adapter.state / "smoke_waits.json").read_text())] == ["timeout"]


class _Ad:
    def __init__(self, wait):
        self.ctx = SimpleNamespace(cfg=SimpleNamespace(slurm=SimpleNamespace(smoke_session_wait_s=wait)))
        self.calls = []

    def smoke(self, wt, **kw):
        self.calls.append(kw)
        return SmokeResult(False, "smoke_unavailable: queue timeout after 900s pending on primary", "", unavailable=True)


def test_the_session_smoke_tool_passes_the_session_wait_and_says_finish(tmp_path):
    wt = Worktree(str(tmp_path), "b", "c")
    for wait, want in ((900, [{"queue_s": 900.0}]), (0, [{}])):
        ad = _Ad(wait)
        tools = WorktreeTools.__new__(WorktreeTools)
        tools.adapter, tools.wt = ad, wt
        out = tools.t_smoke()
        assert ad.calls == want and "smoke NOT RUN" in out and "finish when the code is ready" in out
        assert tools.last_smoke.unavailable                     # the post-session path re-smokes it (no LLM held)
