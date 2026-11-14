"""Operations of the live v3 loop (cutover fix batch C): no zombie (a tick that raises is logged and retried with a
backoff, FATAL_STREAK in a row is fatal), guarded call sites quarantined after repeated failures (a port that always
raises becomes a REJECTED port node), the research re-issue with the REAL ``LLMResearcher.research`` signature, a
BackendUnavailable session released and retried once, lost port sessions retried, a half-built wave resumed (never
re-drawn), the disk guard, the shadow's real LLM pause state, the meta-review deferred under the pause, the loud lock
guard, the systemd watchdog, TEST-stage runs never resumed, and the heartbeat line."""

from __future__ import annotations

import dataclasses
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest

from loop import runner as R
from loop import tick as TK
from loop.adapters.sim import SimAdapter
from loop.agents.llm import BackendUnavailable
from loop.contracts import ROOT_NODE_ID, CandidateRequest, NodeStatus, Operator, Run, Stage, Strategy
from loop.core import data as CD
from loop.core import factorial as F
from loop.core import meta as MT
from loop.store import SqliteStore

from .conftest import InlineExecutor
from .test_port import _two_epochs
from .test_waves import _bank


def _quiet(**kw):
    """Tick hooks that do nothing (no ideation / intake / meta of their own)."""
    return {"ideate": lambda *a: [], "intake": lambda req, res: f"intake {req.id}", "meta": lambda *a: [], **kw}


class _Flaky(TK.StubCore):
    """rank raises on its first ``fail`` calls."""

    def __init__(self, fail: int) -> None:
        self.fail, self.n = fail, 0

    def rank(self, *a, **k):
        self.n += 1
        if self.n <= self.fail:
            raise RuntimeError("rank exploded")
        return []


def _req(rid: str, parent: str = ROOT_NODE_ID) -> CandidateRequest:
    return CandidateRequest(id=rid, task_id="", strategy=Strategy.EXPLORE, operator=Operator.DRAFT, island="main",
                            parent_ids=(parent,), brief="try something")


# ---------------------------------------------------------------------------------------------- 1. no zombie
def test_live_logs_a_failed_tick_with_its_traceback_backs_off_and_goes_on(ctx, tmp_path):
    t, printed = [1e6], []
    db = tmp_path / "loop.db"
    wd = R.Watchdog(env={})
    n = TK.live(ctx, db, _Flaky(2), SimAdapter(ctx), None, max_ticks=4, clock=lambda: t[0],
                sleep=lambda dt: t.__setitem__(0, t[0] + dt), out=printed.append, executor=InlineExecutor,
                watchdog=wd, **_quiet())
    assert n == 4
    poll = ctx.cfg.heartbeat.poll_s
    assert t[0] - 1e6 == pytest.approx(poll * (1 + 2 + 1))                # backoff: poll, 2 poll, then back to poll
    store = SqliteStore(db, ctx.task_id)
    errs = store.events("tick_error")
    assert [e["streak"] for e in errs] == [1, 2] and errs[0]["phase"] == "acquire"
    assert "rank exploded" in errs[0]["traceback"] and "Traceback" in errs[0]["traceback"]
    assert any("tick_error (1 in a row" in p and "Traceback" in p for p in printed)
    hbs = [p for p in printed if " hb tick=" in p]
    assert any("TICK_ERROR RuntimeError in acquire" in h for h in hbs) and len(store.events("hb")) >= 2
    assert all(k in hbs[0] for k in ("gpu_free=", "research=", "open_tests=", "open_waves=", "incumbent=", "llm=",
                                     "lock=", "disk="))
    store.close()


def test_live_gives_up_after_fatal_streak_errors_in_a_row(ctx, tmp_path):
    t = [1e6]
    with pytest.raises(TK.TickFatal, match="10 tick errors in a row"):
        TK.live(ctx, tmp_path / "loop.db", _Flaky(10**6), SimAdapter(ctx), None, clock=lambda: t[0],
                sleep=lambda dt: t.__setitem__(0, t[0] + dt), out=lambda m: None, executor=InlineExecutor,
                watchdog=R.Watchdog(env={}), **_quiet())
    assert t[0] - 1e6 <= TK.TICK_BACKOFF_MAX_S * TK.FATAL_STREAK                # the backoff is capped


def test_cli_run_exits_non_zero_on_a_fatal_tick(tmp_path, monkeypatch, fake_loop_spec):
    import loop.__main__ as M

    def boom(*a, **k):
        raise TK.TickFatal("10 tick errors in a row")
    monkeypatch.setattr(TK, "live", boom)
    monkeypatch.setattr(M, "resolve_adapter", lambda ctx: SimAdapter(ctx))
    assert M.main(["run", "--task", str(fake_loop_spec), "--fake-llm", "--state-dir", str(tmp_path / "st")]) == 1


# ---------------------------------------------------------------------------------------------- 2. guarded call sites
def test_a_port_that_always_raises_is_quarantined_as_a_rejected_port_node(ctx, tmp_path):
    sim, store = _two_epochs(ctx, tmp_path)
    calls = []

    def port(tick, a, data):
        calls.append(a.node)
        raise RuntimeError("graft exploded")
    tick = TK.Tick(ctx, store, sim, TK.ModuleCore(), db_path=tmp_path / "loop.db", clock=lambda: 30.0,
                   executor=InlineExecutor, **_quiet(port=port))
    acts: list[str] = []
    for _ in range(4):
        led = CD.load(tmp_path / "loop.db", ctx.task_id)
        tick._launch([TK.Action("port", "x", "explore", 3.0, 1.0, 3)], led, {}, 30.0, acts)
    assert calls == ["x"] * TK.QUARANTINE_AFTER                            # quarantined: not called a 4th time
    assert [e["n"] for e in store.events("tick_call_error")] == [1, 2, 3]
    assert "port:x" in store.get_state(TK.K_QUARANTINE)
    node = store.get_node(f"x@{led.epoch[:7]}")
    assert node.status is NodeStatus.REJECTED and "port_error" in node.notes and "graft exploded" in node.reject_reason
    assert any(a.startswith("ALERT QUARANTINED port:x") for a in acts)
    tick.close()
    store.close()


def test_a_failing_intake_and_ledger_load_never_stop_the_tick(ctx, tmp_path, monkeypatch):
    sim = SimAdapter(ctx)
    store = SqliteStore(tmp_path / "loop.db", ctx.task_id)

    def intake(req, res):
        raise ValueError("bad candidate")
    tick = TK.Tick(ctx, store, sim, TK.StubCore(), db_path=tmp_path / "loop.db", clock=lambda: 5.0,
                   executor=InlineExecutor, ideate=lambda *a: [], intake=intake, meta=lambda *a: [])
    assert "ERROR intake of r1" in tick._intake(_req("r1"), SimpleNamespace(status="ok"))
    assert store.events("research_failed")[0]["status"] == "intake_error"
    monkeypatch.setattr(TK.CD, "load", lambda *a: (_ for _ in ()).throw(OSError("disk I/O error")))
    rep = tick.step()
    assert any("ERROR load the ledger view" in x for x in rep.lines) and not rep.actions
    assert not store.get_state(TK.K_QUARANTINE)                           # the ledger is never quarantined
    tick.close()
    store.close()


# ---------------------------------------------------------------------------------------------- 3. / 4. research
def test_a_requeued_session_starts_with_the_real_llm_researcher_signature(ctx, tmp_path):
    from loop.__main__ import _fake_llm
    from loop.agents import LLMResearcher
    sim = SimAdapter(ctx)
    store = SqliteStore(tmp_path / "loop.db", ctx.task_id)
    store.ensure_root(sim.root_commit())
    research = R.Research(ctx, store, sim, LLMResearcher(ctx, _fake_llm(), sim), InlineExecutor(), lambda: 1.0)
    research.requeue.append(dataclasses.replace(_req("req-v3-00007"), task_id=ctx.task_id))
    assert research.start_requeued() == ["researcher re-issued after restart: req-v3-00007 explore/draft"]
    acts: list[str] = []
    [(req, res)] = research.harvest(acts)
    assert req.id == "req-v3-00007" and not store.events("research_error")   # no TypeError (missing snapshot)
    assert res.status != "ok" and "fake-llm" in str(res.error)                 # the real research ran (fake LLM)
    store.close()


def test_backend_unavailable_releases_the_slot_and_retries_once(ctx, tmp_path):
    store = SqliteStore(tmp_path / "loop.db", ctx.task_id)

    class Down:
        n = 0

        def research(self, req, store, snapshot):
            Down.n += 1
            raise BackendUnavailable("qwen gateway 502")
    research = R.Research(ctx, store, SimAdapter(ctx), Down(), InlineExecutor(), lambda: 1.0)
    research.start(_req("r1"))
    acts: list[str] = []
    assert research.harvest(acts) == [] and not research.inflight             # the slot is free again
    assert [r.id for r in research.requeue] == ["r1"] and "retried once" in acts[-1]
    assert research.start_requeued() and research.harvest(acts) == []
    assert not research.inflight and not research.requeue and Down.n == 2      # dropped after the second
    assert [e["tries"] for e in store.events("research_backend_unavailable")] == [1, 2]
    assert store.events("research_failed")[0]["status"] == "backend_unavailable"
    store.close()


def test_a_port_session_lost_without_a_node_is_retried_then_given_up(ctx, tmp_path):
    sim, store = _two_epochs(ctx, tmp_path)
    led = CD.load(tmp_path / "loop.db", ctx.task_id)

    class Broken:
        def research(self, req, store, snapshot):
            raise RuntimeError("session crashed")
    t = [30.0]
    tick = TK.Tick(ctx, store, sim, TK.ModuleCore(), db_path=tmp_path / "loop.db", clock=lambda: t[0],
                   researcher=Broken(), executor=InlineExecutor, **_quiet())
    tick.port_graft("z", led)                                             # a conflict: one port session queued
    acts: list[str] = []
    for i in range(5):
        tick._tick_ts = t[0]
        tick.research.harvest(acts)
        tick._port_sessions(acts)
        t[0] += 60.0
    book = store.get_state(TK.Tick.K_PORT_SESSIONS)
    assert book["z"]["status"] == "failed" and book["z"]["tries"] == TK.Tick.PORT_SESSION_TRIES
    assert len(store.events("research_started")) == TK.Tick.PORT_SESSION_TRIES
    assert len(store.events("port_session_retry")) == TK.Tick.PORT_SESSION_TRIES - 1
    assert len(store.events("port_session_failed")) == 1
    tick.close()
    store.close()


# ---------------------------------------------------------------------------------------------- 5. half-built waves
def test_a_wave_build_that_crashes_is_resumed_not_redrawn(ctx, tmp_path, monkeypatch):
    sim, store = _two_epochs(ctx, tmp_path)
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, slots=16))
    core = TK.ModuleCore()
    core.knobs = {"inc@new": _bank()}
    tick = TK.Tick(ctx, store, sim, core, db_path=tmp_path / "loop.db", clock=lambda: 50.0,
                   executor=InlineExecutor, **_quiet())
    real, n = F.build_point, [0]

    class Killed(BaseException):                       # the process dies mid-build (no except clause sees it)
        pass

    def crashy(*a, **k):
        n[0] += 1
        if n[0] == 4:
            raise Killed()
        return real(*a, **k)
    monkeypatch.setattr(F, "build_point", crashy)
    led = CD.load(tmp_path / "loop.db", ctx.task_id)
    acts: list[str] = []
    with pytest.raises(Killed):
        tick._launch([TK.Action("wave", "inc@new", "factorial", 8.0, 1.0, 8)], led, {}, 50.0, acts)
    job = store.get_state(TK.K_WAVE_BUILD)                              # only the journal survives the crash
    assert job["wave"] == "w001" and store.get_state(TK.Tick.K_WAVES) is None
    assert not store.events("wave_started") and not [x for x in store.nodes() if F.is_point(x.notes)]
    assert store.get_state(TK.ModuleCore.K_USED + "inc@new") is None and store.get_state("wave:w001") is None
    monkeypatch.setattr(F, "build_point", real)
    led = CD.load(tmp_path / "loop.db", ctx.task_id)
    launched = tick._launch([], led, {}, 51.0, acts)                       # the next tick resumes it
    [blk] = [b for b in store.seed_blocks() if b.id.startswith("wave-")]
    assert blk.id == "wave-w001" and store.get_state(TK.K_WAVE_BUILD) is None
    pts = [x for x in store.nodes() if F.is_point(x.notes)]
    assert sorted(int(F._TAG.search(x.notes).group(2)) for x in pts) == list(range(8))
    pt_runs = [r for r in launched if ":screen:" in r]                  # each point once, at one of the block seeds
    assert sorted(r.split(":")[0] for r in pt_runs) == sorted(x.id for x in pts)
    assert {int(r.rsplit(":", 1)[1]) for r in pt_runs} == set(blk.seeds)
    assert sorted(r for r in launched if ":screen:" not in r) == sorted(f"inc@new:parent:{s}" for s in blk.seeds)
    assert [e["wave"] for e in store.events("wave_started")] == ["w001"]
    assert any("resuming a half-built wave" in a for a in acts)
    assert store.get_state(TK.Tick.K_WAVES) == 1                           # no w002 on the same bank
    assert sorted(store.get_state(TK.ModuleCore.K_USED + "inc@new")) == sorted(b.signature() for b in _bank())
    tick.close()
    store.close()


# ---------------------------------------------------------------------------------------------- 6. disk, 8. pause
def test_the_disk_guard_holds_new_research_sessions(ctx, tmp_path, monkeypatch):
    store = SqliteStore(tmp_path / "loop.db", ctx.task_id)
    tick = TK.Tick(ctx, store, SimAdapter(ctx), TK.StubCore(), db_path=tmp_path / "loop.db", clock=lambda: 5.0,
                   researcher=object(), executor=InlineExecutor, **_quiet())
    assert tick.research.free() > 0
    monkeypatch.setattr(TK, "DISK_FLOOR_GB", 1e12)
    acts: list[str] = []
    tick._disk_guard(5.0, acts)
    tick._disk_guard(6.0, acts)
    assert tick.research.free() == 0 and len(store.events("disk_low")) == 1 and acts[0].startswith("ALERT disk low")
    assert "held (disk low)" in tick.heartbeat(0, 1.0)
    monkeypatch.setattr(TK, "DISK_FLOOR_GB", 0.0)
    tick._disk_guard(7.0, acts)
    assert tick.research.free() > 0 and len(store.events("disk_ok")) == 1
    tick.close()
    store.close()


def test_the_shadow_reports_the_real_llm_pause(ctx, tmp_path):
    seen = []

    class Core(TK.StubCore):
        def rank(self, data, blocks, statuses, post, free_slots, llm_free):
            seen.append(llm_free)
            return []
    store = SqliteStore(tmp_path / "loop.db", ctx.task_id)
    store.ensure_root("dry-root")
    store.set_state(R.K_LLM_PAUSE, {"active": True, "kind": "operator", "since": 0.0, "until": 9e9, "probes": 0})
    tick = TK.Tick(ctx, store, TK.DryRunAdapter(store), Core(), db_path=tmp_path / "loop.db", clock=lambda: 5.0,
                   dry=True, executor=InlineExecutor, **_quiet())
    rep = tick.step()
    assert seen == [0] and any(x.startswith("LLM paused operator until") for x in rep.lines)
    store.set_state(R.K_LLM_PAUSE, {"active": False, "until": 1.0})       # cleared (until in the past)
    tick.step()
    # v3-c5 A3: every GPU slot idle (> ideation.llm_scale_free_slots) -> the cap scales to ideation.llm_slots_max
    assert tick.research.gpu_free > ctx.cfg.ideation.llm_scale_free_slots
    assert seen[-1] == max(ctx.cfg.researcher.max_concurrent, ctx.cfg.ideation.llm_slots_max)
    tick.close()
    store.close()


def test_the_meta_review_waits_for_the_llm_and_has_its_own_pool(ctx, tmp_path, store):
    from loop.agents.llm import FakeLLMClient
    asked = []
    llm = FakeLLMClient({"review": lambda **call: asked.append(1) or {"diagnosis": "fine", "proposals": []}})
    c = dataclasses.replace(ctx, state_dir=str(tmp_path))
    meta = MT.Meta(c, store, llm=llm, clock=lambda: 2e9, locks=None)
    store.set_state(R.K_LLM_PAUSE, {"active": True, "kind": "quota", "since": 0.0, "until": 9e9, "probes": 0})
    led = CD.load(tmp_path / "loop.db", ctx.task_id)
    lines = meta.cycle(led, llm=llm, pool=InlineExecutor())
    assert any("meta-review deferred: LLM paused" in x for x in lines) and not asked
    assert store.get_state(MT.K_REVIEW_PENDING)
    tick = SimpleNamespace(core=None, research=SimpleNamespace(prompt_dir=0, researcher=None), post=None,
                           pool=InlineExecutor())
    assert not meta(tick, led, {}) or not asked                            # still paused: nothing
    store.set_state(R.K_LLM_PAUSE, {"active": False, "until": 1.0})
    out = meta(tick, led, {})
    assert asked == [1] and any("meta-review started" in x for x in out)
    assert isinstance(meta.review_pool(tick), InlineExecutor) and meta.review_pool(tick) is not tick.pool
    assert store.get_state(MT.K_REVIEW_PENDING) is None


# ---------------------------------------------------------------------------------------------- 9. lock guard
def test_a_disabled_lock_guard_is_loud_and_a_clean_restart_clears_it(ctx, tmp_path, store, capsys):
    off = MT.Meta(dataclasses.replace(ctx, state_dir=str(tmp_path)), store, locks=MT.Locks(tmp_path / "not-a-repo"))
    assert off.guard() is None and off.guard() is None
    assert off.lock_state() == "DISABLED" and len(store.events("lock_guard_disabled")) == 1
    assert capsys.readouterr().err.count("lock guard DISABLED") == 1           # once, not every tick
    assert store.get_state(MT.K_BREAKER)["disabled"]
    from .core.test_meta import _git_repo
    repo = _git_repo(tmp_path / "repo")
    on = MT.Meta(dataclasses.replace(ctx, state_dir=str(tmp_path)), store,      # the redeploy restart
                 locks=MT.Locks(repo, ("loop/core/test.py",)))
    assert on.guard() is None and on.lock_state() == "ok"
    assert "disabled" not in store.get_state(MT.K_BREAKER) and store.events("lock_guard_cleared")


# ---------------------------------------------------------------------------------------------- watchdog, routing
def test_the_watchdog_notifies_systemd_and_writes_the_heartbeat_file(tmp_path):
    addr = str(tmp_path / "notify.sock")
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    srv.bind(addr)
    srv.settimeout(2)
    t = [100.0]
    wd = R.Watchdog(tmp_path / "loop.heartbeat", every_s=10.0, clock=lambda: t[0], env={"NOTIFY_SOCKET": addr})
    assert wd.enabled and wd.notify("READY=1")
    wd.ping("tick 3")
    t[0] += 1
    wd.ping()                                                              # rate-limited: not sent
    assert srv.recv(256) == b"READY=1" and srv.recv(256) == b"WATCHDOG=1\nSTATUS=tick 3"
    assert wd.sent == ["READY=1", "WATCHDOG=1\nSTATUS=tick 3"]
    assert (tmp_path / "loop.heartbeat").read_text().startswith("100 ")
    srv.close()
    assert not R.Watchdog(env={}).notify("WATCHDOG=1")                    # no socket: a no-op


def test_test_stage_runs_are_never_resumed(ctx, tmp_path):
    store = SqliteStore(tmp_path / "loop.db", ctx.task_id)
    sim = SimAdapter(ctx)
    sim.resume_ok = True
    tick = TK.Tick(ctx, store, sim, TK.StubCore(), db_path=tmp_path / "loop.db", executor=InlineExecutor, **_quiet())

    def run(stage):
        return Run(id=Run.make_id("n1", stage, 7), task_id=ctx.task_id, node_id="n1", seed=7, stage=stage,
                   status=R.RunStatus.RUNNING)
    assert sim.no_resume(run(Stage.CONFIRM)) and sim.no_resume(run(Stage.REMINT))
    assert not sim.no_resume(run(Stage.SCREEN)) and not sim.no_resume(run(Stage.BASELINE))
    assert tick.book.high(None, Stage.CONFIRM) and tick.book.high(None, Stage.REMINT)   # primary pool first
    tick.close()
    store.close()


def test_the_prune_view_keeps_what_open_work_references(ctx, tmp_path):
    sim, store = _two_epochs(ctx, tmp_path)
    tick = TK.Tick(ctx, store, sim, TK.StubCore(), db_path=tmp_path / "loop.db", executor=InlineExecutor, **_quiet())
    led = CD.load(tmp_path / "loop.db", ctx.task_id)
    view = tick.prune_view(led, {"x": "FAILED", "z": "VOID", "y": "SCREENING", "inc@new": "INCUMBENT"})
    assert view.incumbent_id == "inc@new" and view.closed == {"x", "z"}
    assert {"inc@new", "root2"} <= view.keep                                 # the incumbent and its parent
    assert "y" in view.screening and "y" not in view.keep                    # SCREENING: loop.prune settles it
    from loop import prune as P
    protected, models, _ = P.classify(ctx, store, view, 1e9)
    assert "x" not in protected and "inc@new" in models
    assert "y" not in protected                                              # nothing holds it a day after its runs
    just_after = max([r.submitted_ts for r in store.runs() if r.node_id == "y"] + [store.get_node("y").updated_ts]) + 60.0
    protected, _, _ = P.classify(ctx, store, view, just_after)
    assert "y" in protected and "root" in protected                          # inside the grace: it and its parent
    tick.close()
    store.close()


def test_the_prune_view_keeps_the_inflight_parents_past_incumbents_and_the_best_alt_base_sources(ctx, tmp_path):
    """10-20 review of 08876d4: a SCREENING node nothing open references is settled, EXCEPT what a later lane still
    needs a checkpoint of: an in-flight research session's parent (its child resumes it eval-only), a past incumbent,
    and the top posterior current-epoch nodes (altbase.buildable needs a checkpoint, else the pick leaves the lane)."""
    from loop.contracts import K_INFLIGHT
    from .test_port import _commit, _node
    sim, store = _two_epochs(ctx, tmp_path)
    _node(store, "q", _commit(sim, store.get_node("root2").commit, "q", {"train.lr": 6e-4}), "root2", 12.0)
    _node(store, "w", _commit(sim, store.get_node("root2").commit, "w", {"train.lr": 5e-4}), "root2", 13.0)
    store.set_state(K_INFLIGHT, [{"id": "r1", "parents": ["y"], "queued": True}])
    tick = TK.Tick(ctx, store, sim, TK.StubCore(), db_path=tmp_path / "loop.db", executor=InlineExecutor, **_quiet())
    tick.post = SimpleNamespace(beliefs=lambda: {"q": (0.01, 1e-6, 5), "w": (-0.02, 1e-6, 5)})
    led = CD.load(tmp_path / "loop.db", ctx.task_id)
    st = {"x": "SCREENING", "y": "SCREENING", "z": "SCREENING", "q": "SCREENING", "w": "SCREENING",
          "inc": "SCREENING", "inc@new": "INCUMBENT"}
    view = tick.prune_view(led, st)
    assert "y" in view.keep and "y" not in view.screening                    # the in-flight session's parent
    assert "q" in view.keep and "q" not in view.screening                    # a standing, sound, current-epoch pick
    assert "w" in view.screening and "w" not in view.keep                    # clearly negative: settles
    assert "x" in view.screening and "z" in view.screening
    tick.close()
    store.close()
