"""10-22: ``meta.review_pending`` was written when the LLM pause deferred a review and never read back, so a restart
during a pause lost the review. ``Meta`` now rebuilds it from the report's JSON twin at start."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

from loop import runner as R
from loop.agents.llm import FakeLLMClient
from loop.core import data as CD
from loop.core import meta as MT

from .conftest import InlineExecutor


def _deferred(ctx, tmp_path, store, now=2e9):
    asked = []
    llm = FakeLLMClient({"review": lambda **call: asked.append(1) or {"diagnosis": "fine", "proposals": []}})
    c = dataclasses.replace(ctx, state_dir=str(tmp_path))
    meta = MT.Meta(c, store, llm=llm, clock=lambda: now, locks=None)
    store.set_state(R.K_LLM_PAUSE, {"active": True, "kind": "quota", "since": 0.0, "until": 9e9, "probes": 0})
    led = CD.load(tmp_path / "loop.db", ctx.task_id)
    assert any("deferred" in x for x in meta.cycle(led, llm=llm, pool=InlineExecutor()))
    return c, llm, led, asked


def test_a_review_deferred_by_the_llm_pause_survives_a_restart(ctx, tmp_path, store):
    c, llm, led, asked = _deferred(ctx, tmp_path, store)
    assert store.get_state(MT.K_REVIEW_PENDING)
    fresh = MT.Meta(c, store, llm=llm, clock=lambda: 2e9 + 60, locks=None)       # the restart: a new process
    assert isinstance(fresh._pending, MT.MetaReport) and fresh._pending.ts == 2e9
    store.set_state(R.K_LLM_PAUSE, {"active": False, "until": 1.0})
    tick = SimpleNamespace(core=None, research=SimpleNamespace(prompt_dir=0, researcher=None), post=None,
                           pool=InlineExecutor())
    out = fresh(tick, led, {})
    assert asked == [1] and any("meta-review started" in x for x in out)
    assert store.get_state(MT.K_REVIEW_PENDING) is None and fresh._pending is None


def test_a_stale_or_unreadable_pending_review_is_dropped_not_fatal(ctx, tmp_path, store):
    c, llm, led, _ = _deferred(ctx, tmp_path, store)
    late = MT.Meta(c, store, llm=llm, clock=lambda: 2e9 + 3 * 86400, locks=None)
    assert late._pending is None and store.get_state(MT.K_REVIEW_PENDING) is None
    assert store.events("meta_review_pending_dropped")[-1]["why"] == "stale"
    store.set_state(MT.K_REVIEW_PENDING, {"ts": 2e9, "report": str(tmp_path / "nope.md")})
    bad = MT.Meta(c, store, llm=llm, clock=lambda: 2e9 + 1, locks=None)
    assert bad._pending is None and store.get_state(MT.K_REVIEW_PENDING) is None
    assert "unreadable" in store.events("meta_review_pending_dropped")[-1]["why"]


def test_the_real_tick_builds_a_meta_that_holds_the_pending_review(ctx, tmp_path, store):
    import time
    from loop import tick as TK
    from .test_set1_harness import _Held
    c, llm, led, _ = _deferred(ctx, tmp_path, store, now=time.time())
    tick = TK.Tick(c, store, _Held(), TK.ModuleCore(), db_path=tmp_path / "loop.db", clock=time.time)
    assert isinstance(tick.meta._pending, MT.MetaReport)             # read back by the Meta the Tick itself constructs
    tick.close()


def test_a_deferred_review_older_than_the_last_report_is_not_replayed(ctx, tmp_path, store):
    """10-22 20:30Z: a restart reloaded the 10-21 11:43 pending report (an empty window) although three newer reports
    had been written, and the review said "this window has no data" about a day with dozens of sessions."""
    c, llm, led, asked = _deferred(ctx, tmp_path, store)
    store.set_state(MT.K_LAST, {**(store.get_state(MT.K_LAST) or {}), "ts": 2e9 + 3 * 3600})
    fresh = MT.Meta(c, store, llm=llm, clock=lambda: 2e9 + 4 * 3600, locks=None)
    assert fresh._pending is None and store.get_state(MT.K_REVIEW_PENDING) is None
    assert store.events("meta_review_pending_dropped")[-1]["why"] == "superseded"


def test_a_new_undeferred_report_clears_a_stored_deferral(ctx, tmp_path, store):
    c, llm, led, asked = _deferred(ctx, tmp_path, store)
    store.set_state(R.K_LLM_PAUSE, {"active": False, "until": 1.0})
    meta = MT.Meta(c, store, llm=llm, clock=lambda: 2e9 + 3600, locks=None)
    assert meta._pending is not None
    meta.cycle(led, llm=llm, pool=InlineExecutor())
    assert meta._pending is None and store.get_state(MT.K_REVIEW_PENDING) is None


def test_dashboard_and_digest_render_without_the_instance_lock(ctx, tmp_path, monkeypatch, capsys):
    """10-22: the render child is read-only; it must neither take nor be refused by ``<db>.lock`` the live loop holds."""
    import fcntl

    from loop import __main__ as M
    from loop.store import SqliteStore
    c = dataclasses.replace(ctx, state_dir=str(tmp_path / "st"))
    SqliteStore(M._db(c), c.task_id, weights=c.cfg.weights).close()
    held = M._lock(c)                                   # the live loop
    assert held is not None and M._lock(c) is None      # a second `run` IS refused ...
    assert "(pid " in capsys.readouterr().err           # ... and the shared log now says who asked
    monkeypatch.setattr("loop.dashboard.write", lambda *a, **k: tmp_path / "index.html")
    (tmp_path / "index.html").write_text("x")
    assert M._dashboard(c, None) == 0
    assert "refused" not in capsys.readouterr().err
    fcntl.flock(held, fcntl.LOCK_UN)
    held.close()
