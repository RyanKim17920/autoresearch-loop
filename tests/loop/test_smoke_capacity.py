"""10-22: four Opus sessions ended ``backend_unavailable`` after their smokes sat PENDING 1 h behind 28 of our own screens
(same QoS, so SLURM ordered them by submit time), and the finished drafts were dropped as 'produced nothing'."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

import dataclasses

import loop.infra as INF
from loop import runner as R
from loop import tick as TK
from loop.contracts import CandidateRequest, Operator, ResearchResult, Strategy
from tests.loop.test_adapters import SlurmRunner, base_repo, make_np, np_ctx_with_locks  # noqa: F401
from loop.spec import PoolCfg, SlurmCfg
from tests.loop.test_set1_harness import _Held

from .test_infra import _req, _research


def _sub(ago_s: float, now: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now - ago_s))


def _adapter(tmp_path, base_repo, rows, now=5000.0):  # noqa: F811
    clock = {"t": now}
    runner = SlurmRunner({"squeue": lambda argv: (0, "\n".join(r() if callable(r) else r for r in rows) + "\n", "")})
    ctx = np_ctx_with_locks(tmp_path / "sp")
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, slurm=SlurmCfg(pools=(
        PoolCfg("primary", slots=8, qos="normal"), PoolCfg("idle", slots=56, qos="low", preemptible=True)))))
    adapter, _ = make_np(ctx, base_repo, runner=runner, clock=lambda: clock["t"])
    return adapter, runner, clock, ctx


def _row(jid, name, nice, sub, qos="low"):
    return f"{jid}|{name}|{nice}|{sub}|{qos}"


def _pool_qos(ctx):
    return {p.qos for p in ctx.cfg.pools if p.preemptible}


def test_smokes_pending_counts_only_our_pending_smokes_per_pool_and_never_touches_nice(tmp_path, base_repo):  # noqa: F811
    now = time.time()
    adapter, runner, _, ctx = _adapter(tmp_path, base_repo, [], now)
    tag, q = adapter._tag, sorted(_pool_qos(ctx))[0]
    rows = [_row(1, f"lp{tag}-n3035-screen", 0, _sub(100, now), q),          # a run, not a smoke
            _row(2, f"lp{tag}-smoke-loop-req-v3-1", 0, _sub(100, now), q),    # our smoke on the idle pool
            _row(3, f"lp{tag}-smoke-loop-req-v3-2", 0, _sub(100, now), q),
            _row(4, f"lp{tag}-smoke-loop-req-v3-3", 0, _sub(100, now), "normal"),   # on primary
            _row(5, "sw-s4-spectra-smoke-x", 0, _sub(100, now), q)]           # someone else's: never
    runner.replies["squeue"] = (0, "\n".join(rows) + "\n", "")
    assert adapter.smokes_pending() == {"idle": 2, "primary": 1}
    assert not runner.argvs("scontrol") and not hasattr(adapter, "renice_runs")   # no Nice, ever (10-02)
    runner.replies["squeue"] = (1, "", "boom")
    assert adapter.smokes_pending() == {}


def test_a_long_pending_smoke_closes_new_sessions_and_a_young_one_does_not(tmp_path, base_repo):  # noqa: F811
    now = time.time()
    adapter, runner, clock, ctx = _adapter(tmp_path, base_repo, [], now)
    tag, q = adapter._tag, sorted(_pool_qos(ctx))[0]
    clock["t"] = now
    runner.replies["squeue"] = (0, _row(9, f"lp{tag}-smoke-loop-req-v3-9", 0, _sub(300, now), q) + "\n", "")
    assert adapter.smoke_capacity() == ("open", "")
    clock["t"] = now + 3600                                  # past the 30 s cache; the same smoke is now an hour old
    state, why = adapter.smoke_capacity()
    assert state == "closed" and "oldest 65m" in why
    runner.replies["squeue"] = (1, "", "boom")               # a monitoring error never blocks sessions
    clock["t"] += 100
    assert adapter.smoke_capacity()[0] == "open"


def test_a_queue_timeout_probes_with_one_session_until_a_smoke_starts(tmp_path, base_repo):  # noqa: F811
    now = 100000.0
    adapter, runner, clock, _ = _adapter(tmp_path, base_repo, [], now)
    runner.replies["squeue"] = (0, "", "")
    adapter._smoke_note("timeout", 3600)
    assert adapter.smoke_capacity()[0] == "probe"
    clock["t"] += 100
    adapter._smoke_note("started", 40)                       # a smoke got a slot after the timeout: open again
    assert adapter.smoke_capacity()[0] == "open"
    clock["t"] += 100
    adapter._smoke_note("timeout", 3600)
    assert adapter.smoke_capacity()[0] == "probe"
    clock["t"] += 3 * 3600 + 200                             # the memory runs out
    assert adapter.smoke_capacity()[0] == "open"
    assert len(json.loads((adapter.state / "smoke_waits.json").read_text())) == 3


def test_smoke_wait_records_a_timeout_in_the_notes(tmp_path, base_repo):  # noqa: F811
    adapter, runner, clock, _ = _adapter(tmp_path, base_repo, [], 7000.0)
    runner.replies["squeue"] = (0, "PENDING\n", "")

    def tick(_s):
        clock["t"] += 2000.0
    import loop.adapters.nanopath_v2 as NV
    orig, NV.time.sleep = NV.time.sleep, tick
    try:
        rc, why = adapter._smoke_wait("77")
    finally:
        NV.time.sleep = orig
    assert rc == 124 and "queue timeout" in why
    assert [r["kind"] for r in json.loads((adapter.state / "smoke_waits.json").read_text())] == ["timeout"]


def test_the_research_runner_starts_no_session_while_the_smoke_queue_is_closed(ctx, store):
    rs = _research(ctx, store)
    rs.researcher = object()
    open_n = rs.capacity()
    assert open_n > 1
    rs.smoke_capacity = lambda: ("closed", "x")
    assert rs.capacity() == 0
    rs.smoke_capacity = lambda: ("probe", "x")
    assert rs.capacity() == 1                                # one probe session at a time
    rs.inflight[object()] = _req()
    assert rs.capacity() == 0


def test_a_smoke_starved_session_keeps_its_draft_and_is_reissued_not_dropped(ctx, store):
    rs = _research(ctx, store)
    h = SimpleNamespace(family="optimization", mechanism_tag="m", claim="the claim")
    res = ResearchResult("backend_unavailable", SimpleNamespace(hypothesis=h, diff_text="DIFF-BODY"),
                         "smoke_unavailable: smoke_unavailable: queue timeout after 3610s pending on idle")
    assert INF.is_smoke_starved(res) and not INF.is_infra_unavailable(res)
    fake = SimpleNamespace(research=rs, intake=lambda *a: pytest.fail("a starved session makes no node"))
    out = TK.Tick._intake(fake, _req(), res)
    assert "re-issued with its diff (1/" in out and "produced nothing" not in out
    q = rs.requeue[0]
    assert "DIFF-BODY" in q.brief and q.brief.startswith("orig brief") and q.id == "req-v3-1"
    assert any(e["status"] == "smoke_starved" and e["reissued"] for e in store.events("research_failed"))
    for _ in range(INF.MAX_STARVE_REISSUES - 1):
        rs.reissue_starved(_req(), res)
    assert "dropped" in rs.reissue_starved(_req(), res)


def test_the_real_tick_wires_the_capacity_into_the_runner_and_the_heartbeat(ctx, store, tmp_path):
    adapter = _Held()
    adapter.smoke_capacity = lambda: ("closed", "2 smoke(s) pending, oldest 31m > 25m")
    from .test_runner import _pools
    ctx = _pools(ctx)
    tick = TK.Tick(ctx, store, adapter, TK.ModuleCore(), db_path=tmp_path / "loop.db", clock=time.time)
    assert tick.research.smoke_capacity is adapter.smoke_capacity
    assert " smokecap=closed(2 smoke(s) pending, oldest 31m > 25m)" in tick.heartbeat(1, 0.5)
    assert " throttle=off " in tick.heartbeat(1, 0.5)                    # 11-09: the brief throttle's state
    store.set_state("alloc.throttle", "33 queued > 12")
    assert " throttle=on(33 queued > 12; trickle 0) " in tick.heartbeat(1, 0.5)
    assert not hasattr(tick, "_renice") and not hasattr(ctx.cfg.slurm, "idle_nice")
    tick.close()
