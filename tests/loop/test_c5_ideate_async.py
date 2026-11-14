"""v3-c5 R4 (research briefs rendered off the tick thread) and A3 (brief kinds weighted by their measured yield)."""

from __future__ import annotations

import dataclasses
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait as fwait
from types import SimpleNamespace

import pytest

from loop import tick as TK
from loop.core import data as D
from loop.core import fnf as FN
from loop.core import ideation as ID
from loop.store import SqliteStore

from .conftest import InlineExecutor
from .core.conftest import TASK
from .core.test_ideation import README
from .test_tick import _ledger


@pytest.fixture
def aenv(ctx, tmp_path, monkeypatch):
    """A real store + ledger, an Ideation on a REAL thread pool, the fork-and-flush arm off (shared briefs only)."""
    db = _ledger(tmp_path / "ledger.db")
    ctx = dataclasses.replace(ctx, task_id=TASK)
    store = SqliteStore(db, TASK)
    readme = tmp_path / "README.md"
    readme.write_text(README)
    monkeypatch.setattr(FN, "SHARE", 0.0)
    ide = ID.Ideation(ctx, store, readme=readme, executor=ThreadPoolExecutor)
    yield ctx, store, D.load(db, TASK), ide
    ide.close(timeout=10)
    store.close()


class _Research:
    def __init__(self, free=2):
        self.n, self.started, self.researcher = free, [], SimpleNamespace(llm=None)

    def free(self):
        return self.n - len(self.started)

    def start(self, req, *args, **log):
        self.started.append(req)


def _tick(free=2):
    return SimpleNamespace(research=_Research(free), pool=InlineExecutor(), dry=False)


class _Gate:
    """A slow renderer: each call blocks until ``open``; ``fail`` makes the next call raise."""

    def __init__(self, ide):
        self.ev, self.calls, self.fail, self.real = threading.Event(), [], 0, ide.brief_text

    def __call__(self, plan, data, v):
        self.calls.append(plan["kind"])
        if not self.ev.wait(30):
            raise TimeoutError("gate never opened")
        if self.fail:
            self.fail -= 1
            raise RuntimeError("render broke")
        return self.real(plan, data, v)


def _drain(ide):
    fwait([j["fut"] for j in ide._jobs], timeout=60)


EXPLORE = SimpleNamespace(kind="idea", lane="explore")
EXPLOIT = SimpleNamespace(kind="idea", lane="exploit")


# ------------------------------------------------------------------------------------------ R4: async briefs
def test_the_tick_returns_without_waiting_on_a_slow_brief(aenv, monkeypatch):
    ctx, store, led, ide = aenv
    gate = _Gate(ide)
    monkeypatch.setattr(ide, "brief_text", gate)
    tick = _tick(2)
    t0 = time.monotonic()
    acts = ide.ideate(tick, [EXPLORE, EXPLOIT], led, {})
    assert time.monotonic() - t0 < 10                                # the renders block 30 s: the tick did not wait
    assert tick.research.started == [] and sum(a.startswith("brief queued") for a in acts) == 2
    assert len(ide._jobs) == 2 and [e["id"] for e in ide.inflight()][-2:] == ["queued-1", "queued-2"]
    gate.ev.set()
    _drain(ide)
    acts = ide.ideate(tick, [EXPLORE, EXPLOIT], led, {})             # the next tick starts the rendered briefs
    assert len(tick.research.started) == 2 and ide._jobs == []
    assert sorted(r.brief_kind for r in tick.research.started) == sorted(gate.calls)
    assert len(store.events("brief_rendered")) == 2


def test_no_double_enqueue_across_ticks(aenv, monkeypatch):
    ctx, store, led, ide = aenv
    gate = _Gate(ide)
    monkeypatch.setattr(ide, "brief_text", gate)
    tick = _tick(2)
    queued = []
    for _ in range(4):                                               # 4 ticks while the renders are still running
        queued += [a for a in ide.ideate(tick, [EXPLORE, EXPLORE, EXPLOIT], led, {}) if a.startswith("brief queued")]
    assert len(queued) == 2 and len(ide._jobs) == 2
    gate.ev.set()
    _drain(ide)
    ide.ideate(tick, [EXPLORE, EXPLORE, EXPLOIT], led, {})
    assert len(tick.research.started) == 2 and ide._jobs == []       # 2 slots: 2 sessions, nothing more queued
    assert len(gate.calls) == 2


def test_a_failed_render_frees_its_slot_and_is_logged(aenv, monkeypatch):
    ctx, store, led, ide = aenv
    gate = _Gate(ide)
    gate.fail = 1
    monkeypatch.setattr(ide, "brief_text", gate)
    tick = _tick(1)
    ide.ideate(tick, [EXPLORE], led, {})
    gate.ev.set()
    _drain(ide)
    acts = ide.ideate(tick, [EXPLORE], led, {})
    err = store.events("brief_error")
    assert len(err) == 1 and "render broke" in err[0]["error"] and any("failed" in a for a in acts)
    assert tick.research.started == []
    assert len(ide._jobs) == 1                                       # the freed slot was enqueued again this tick
    _drain(ide)
    ide.ideate(tick, [EXPLORE], led, {})
    assert len(tick.research.started) == 1


def test_a_raising_start_is_logged_and_never_started_twice(aenv, monkeypatch):
    ctx, store, led, ide = aenv
    tick = _tick(1)
    calls = []

    def boom(req, *a, **k):
        calls.append(req)
        raise RuntimeError("worktree busy")
    tick.research.start = boom
    gate = _Gate(ide)
    gate.ev.set()
    monkeypatch.setattr(ide, "brief_text", gate)
    ide.ideate(tick, [EXPLORE], led, {})
    _drain(ide)
    ide.ideate(tick, [], led, {})
    ide.ideate(tick, [], led, {})
    assert len(calls) == 1 and "worktree busy" in store.events("brief_error")[0]["error"] and ide._jobs == []


def test_a_stale_brief_is_replanned_on_an_incumbent_switch_and_a_plateau_keeps_its_hypothesis(aenv, monkeypatch):
    ctx, store, led, ide = aenv
    object.__setattr__(ctx.cfg.ideation, "kind_weight", False)      # plateau hypotheses go first
    monkeypatch.setattr(ID, "plateau", lambda led, judged: ID.Plateau(True, 123.0, 50.0, 10.0, 2))
    store.set_state(ID.K_PLATEAU, {"active": True, "since": 123.0, "ts": 1e18, "served": 0, "diagnosis": "d",
                                   "inputs": ID.PLATEAU_INPUTS,
                                   "hypotheses": [{"claim": "c0", "decisive_experiment": "x0"},
                                                  {"claim": "c1", "decisive_experiment": "x1"}]})
    gate = _Gate(ide)
    monkeypatch.setattr(ide, "brief_text", gate)
    tick = _tick(1)
    ide.ideate(tick, [EXPLORE], led, {})
    assert ide._jobs[0]["plan"]["kind"] == "kuhn" and ide._jobs[0]["plan"]["idx"] == 0
    assert store.get_state(ID.K_PLATEAU)["served"] == 1             # reserved at enqueue time
    gate.ev.set()
    _drain(ide)
    ide._jobs[0]["inc"] = "a-former-incumbent"                       # the incumbent switched while it rendered
    old = ide._jobs[0]["fut"]
    acts = ide.ideate(tick, [EXPLORE], led, {})
    stale = store.events("brief_stale")
    assert len(stale) == 1 and stale[0]["was"] == "a-former-incumbent"
    if not tick.research.started:
        assert len(ide._jobs) == 1 and ide._jobs[0]["fut"] is not old and ide._jobs[0]["plan"]["idx"] == 0
        _drain(ide)
        ide.ideate(tick, [EXPLORE], led, {})
    r = tick.research.started[0]
    assert r.brief_kind == "kuhn" and r.id and ide._jobs == []        # the SAME hypothesis (idx 0), not the next one
    assert store.get_state(ID.K_PLATEAU)["served"] == 1 and not any(a.startswith("brief queued") for a in acts)


def test_close_cancels_queued_renders_and_waits_for_running_ones(aenv, monkeypatch):
    ctx, store, led, ide = aenv
    gate = _Gate(ide)
    monkeypatch.setattr(ide, "brief_text", gate)
    tick = _tick(3)                                                  # 3 jobs on 2 workers: one waits in the queue
    ide.ideate(tick, [EXPLORE, EXPLORE, EXPLOIT], led, {})
    futs = [j["fut"] for j in ide._jobs]
    deadline = time.monotonic() + 10
    while len(gate.calls) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    threading.Timer(0.3, gate.ev.set).start()
    t0 = time.monotonic()
    assert ide.close(timeout=20) == 3
    assert 0.2 < time.monotonic() - t0 < 15                          # it WAITED for the running renders
    assert all(f.done() for f in futs) and sum(f.cancelled() for f in futs) == 1
    assert ide._jobs == [] and ide._pool is None and len(gate.calls) == 2


def test_tick_close_closes_the_ideation_first(aenv):
    ctx, store, led, ide = aenv
    closed = []
    ide.close = lambda *a, **k: closed.append("ideation") or 0
    pool = SimpleNamespace(shutdown=lambda **k: closed.append("pool"))
    TK.Tick.close(SimpleNamespace(ideate=ide.ideate, pool=pool))
    assert closed == ["ideation", "pool"]
    TK.Tick.close(SimpleNamespace(ideate=lambda *a: [], pool=pool))      # a plain hook: nothing to close


def test_an_inline_pool_or_zero_workers_starts_in_the_same_tick(aenv):
    ctx, store, led, ide = aenv
    inline = ID.Ideation(ctx, store)                                 # no executor: the tick pool's class (inline)
    tick = _tick(2)
    inline.ideate(tick, [EXPLORE, EXPLOIT], led, {})
    assert len(tick.research.started) == 2 and inline._jobs == []
    object.__setattr__(ctx.cfg.ideation, "brief_workers", 0)          # synchronous: no pool at all
    sync = ID.Ideation(ctx, store, executor=ThreadPoolExecutor)
    tick = _tick(1)
    sync.ideate(tick, [EXPLORE], led, {})
    assert len(tick.research.started) == 1 and sync._pool is None


# ------------------------------------------------------------------------------------------ A3: kind weights
def _n(nid, parent, notes="", base="E"):
    return D.Node(nid, parent, None, "explore", "f", "explore", ID.ROOT, 0.0, "", notes, base)


def _view(nodes, beliefs, epoch="E"):
    led = SimpleNamespace(nodes={n.id: n for n in nodes}, epoch=epoch)
    return SimpleNamespace(led=led, beliefs={k: ID.Belief(m, 1e-6, n) for k, (m, n) in beliefs.items()})


def test_kind_yields_credit_tagged_nodes_and_their_untagged_descendants():
    nodes = [_n(ID.ROOT, None), _n("a", ID.ROOT, "request=r1 brief=explore"), _n("a1", "a"), _n("a11", "a1"),
             _n("b", ID.ROOT, "brief=repair"), _n("p", ID.ROOT, "brief=port port_of=x"), _n("p1", "p"),
             _n("u", ID.ROOT), _n("old", ID.ROOT, "brief=explore", base="OLD"), _n("z", ID.ROOT, "brief=literature")]
    bel = {"a": (0.01, 3), "a1": (0.02, 2), "a11": (-0.01, 1), "b": (-0.02, 5), "p": (0.05, 3), "p1": (0.05, 2),
           "u": (0.03, 1), "old": (0.04, 4), "z": (0.0, 0)}
    v = _view(nodes, bel)
    ys = ID.kind_yields(v, descendants=True)
    assert sorted(ys["explore"]) == [-0.01, 0.01, 0.02] and ys["repair"] == [-0.02]
    assert set(ys) == {"explore", "repair"}          # port (+ its child), untagged root child, old epoch, 0 seeds: none
    assert ID.kind_yields(v, descendants=False) == {"explore": [0.01], "repair": [-0.02]}


def test_kind_weights_shrink_to_the_all_kind_mean_and_scale_by_the_node_sd():
    ys = {"explore": [0.01, 0.03], "repair": [-0.02] * 4, "plateau": [-0.05]}
    tab = ID.kind_weights(ys, n0=2.0)
    allv = [0.01, 0.03, -0.02, -0.02, -0.02, -0.02, -0.05]
    mu, sd = sum(allv) / 7, ID.statistics.pstdev(allv)
    assert tab["explore"]["yield"] == pytest.approx((0.04 + 2 * mu) / 4)
    assert tab["repair"]["yield"] == pytest.approx((-0.08 + 2 * mu) / 6)
    assert tab["literature"]["n"] == 0 and tab["literature"]["yield"] == pytest.approx(mu)   # no data: the mean
    assert tab["explore"]["mean"] == pytest.approx(0.02)
    assert sum(r["w"] for r in tab.values()) == pytest.approx(1.0)
    best = max(tab, key=lambda k: tab[k]["w"])
    assert best == "explore"
    for k in tab:
        assert tab[k]["w"] / tab[best]["w"] == pytest.approx(math.exp((tab[k]["yield"] - tab[best]["yield"]) / sd))
    # more pseudo-nodes: everything pulled to the mean (flatter weights)
    flat = ID.kind_weights(ys, n0=100.0)
    assert max(r["w"] for r in flat.values()) < tab["explore"]["w"]
    # too little data: uniform
    assert {r["w"] for r in ID.kind_weights({"explore": [0.1]}, 5.0).values()} == {1 / len(ID.KINDS)}


def test_floored_pins_small_shares_at_the_floor_and_renormalises():
    w = ID.floored({"a": 0.9, "b": 0.07, "c": 0.03}, 0.1)
    assert w["b"] == pytest.approx(0.1) and w["c"] == pytest.approx(0.1) and w["a"] == pytest.approx(0.8)
    assert sum(w.values()) == pytest.approx(1.0)
    w = ID.floored({"a": 10, "b": 1, "c": 0}, 0.2)                   # un-normalised in, cascade: b falls below too
    assert w == pytest.approx({"a": 0.6, "b": 0.2, "c": 0.2})
    assert ID.floored({"a": 1.0, "b": 0.0}, 0.9) == pytest.approx({"a": 0.5, "b": 0.5})   # floor >= 1/n: uniform
    assert ID.floored({}, 0.1) == {}


def test_kind_for_follows_the_weights_and_keeps_the_program_share(aenv, monkeypatch):
    ctx, store, led, ide = aenv
    w = {"explore": 0.6, "literature": 0.25, "repair": 0.1, "program": 0.05, "plateau": 0.0, "crossover": 0.0}
    monkeypatch.setattr(ide, "kind_table", lambda v: {k: {"w": x} for k, x in w.items()})
    monkeypatch.setattr(ide, "program_share", lambda v=None: 0.25)
    v = SimpleNamespace(led=SimpleNamespace(nodes={}, epoch=""))
    got = []
    for i in range(40):
        k = ide.kind_for("explore", v)
        got.append(k)
        store.log("research_started", request_id=f"r{i}", brief_kind=k)
        ide._count_program(k)
    c = {k: got.count(k) for k in set(got)}
    assert c["program"] == 10                                        # exactly the operator's share (quota first)
    rest = {k: c.get(k, 0) for k in ("explore", "literature", "repair")}
    fl = ID.floored({k: w[k] for k in ("explore", "literature", "repair", "program")}, 0.08)
    norm = sum(fl[k] for k in rest)
    for k, n in rest.items():
        assert abs(n / 30 - fl[k] / norm) < 0.07, (k, c)
    assert rest["explore"] > rest["literature"] > rest["repair"] >= 2  # the floor keeps a weak kind alive
    assert ide.kind_for("exploit", v) == "repair" and ide.kind_for("crossover", v) == "crossover"
    # queued (not yet started) briefs count: a queued literature brief pushes the next pick off literature
    monkeypatch.setattr(ide, "program_share", lambda v=None: 0.0)
    nxt = ide.kind_for("explore", v)
    ide._jobs.append({"plan": {"kind": nxt}})
    assert ide.kind_for("explore", v) != nxt or w[nxt] == max(w.values())
    ide._jobs.clear()


def test_kind_table_is_stored_and_logged_once_per_view(aenv):
    ctx, store, led, ide = aenv
    v = ide.view(led)
    t1 = ide.kind_table(v)
    assert set(t1) == set(ID.KINDS) and sum(r["w"] for r in t1.values()) == pytest.approx(1.0)
    ide.kind_table(v)
    assert len(store.events("kind_weights")) == 1
    got = store.get_state(ID.K_KIND_W)                     # v3-c7i: rates per GPU-h are kept to 4 significant digits
    assert got.keys() == t1.keys() and all(got[k][f] == pytest.approx(x, rel=1e-3, abs=6e-3 if f == "w" else 1e-5)
                                           for k, r in t1.items() for f, x in r.items())   # K1: w to 2 decimals


def test_a_rendered_brief_is_replanned_only_when_its_meaning_changed(aenv, monkeypatch):
    """10-22: 62% of rendered briefs were discarded (102 'view', 96 'ttl'; 95 of them two briefs re-rendered every 30 min
    for 48 h while the LLM pause held every slot). A brief waiting for a slot is never re-rendered for age or a moved
    view; a changed epoch or incumbent re-plans it; with a free slot a brief older than BRIEF_TTL_X x ttl whose view
    moved is re-planned; a fresh one on the same view starts."""
    ctx, store, led, ide = aenv
    clock = [1000.0]
    ide.clock = lambda: clock[0]
    ttl = float(ide.opt("brief_ttl_s"))
    tick = _tick(0)                                                  # no free slot: the rendered brief waits
    ide._jobs.append(ide._submit(ide._plan("explore", ide.view(led), "explore", ""), led, ide.view(led)))
    _drain(ide)
    ide._jobs[0]["tok"] = ("another view",)                          # the view moved (every new observation does)
    for hours in (0.2, 1, 6, 48):                                    # ... and the brief waits for 48 h: kept, never re-rendered
        clock[0] = 1000.0 + hours * 3600
        ide.ideate(tick, [], led, {})
        assert not store.events("brief_stale") and len(ide._jobs) == 1, hours
    ide._jobs[0]["epoch"] = "an older base epoch"                    # the base changed under it: its meaning changed
    ide.ideate(tick, [], led, {})
    assert [e["why"] for e in store.events("brief_stale")] == ["epoch"]
    _drain(ide)
    ide._jobs[0]["tok"] = ("another view",)
    tick = _tick(1)                                                  # a slot is free ...
    clock[0] += ttl * (ID.BRIEF_TTL_X - 0.5)                          # ... but the brief is younger than the hard cap: it starts
    ide.ideate(tick, [], led, {})
    assert len(store.events("brief_stale")) == 1 and len(tick.research.started) == 1 and not ide._jobs
    ide._jobs.append(ide._submit(ide._plan("explore", ide.view(led), "explore", ""), led, ide.view(led)))
    _drain(ide)
    ide._jobs[0]["tok"] = ("another view",)
    clock[0] += ttl * (ID.BRIEF_TTL_X + 0.5)
    tick = _tick(1)
    ide.ideate(tick, [], led, {})                                    # a free slot and older than the cap: re-planned
    assert [e["why"] for e in store.events("brief_stale")] == ["epoch", "ttl"]      # (the new render may already have started)


def test_a_dropped_brief_gives_back_its_program_count_and_plateau_hypothesis(aenv, monkeypatch):
    """v3-c5b S7: the program quota count and the plateau hypothesis are reserved at enqueue; a brief that fails to
    render (or is dropped by close) rolls them back -- the hypothesis is served by the next plateau brief."""
    ctx, store, led, ide = aenv
    object.__setattr__(ctx.cfg.ideation, "kind_weight", False)      # plateau hypotheses go first
    monkeypatch.setattr(ID, "plateau", lambda led, judged: ID.Plateau(True, 123.0, 50.0, 10.0, 2))
    store.set_state(ID.K_PLATEAU, {"active": True, "since": 123.0, "ts": 1e18, "served": 0, "diagnosis": "d",
                                   "inputs": ID.PLATEAU_INPUTS,
                                   "hypotheses": [{"claim": "c0", "decisive_experiment": "x0"},
                                                  {"claim": "c1", "decisive_experiment": "x1"}]})
    store.set_state(ID.K_PROGRAM, {"n": 5, "p": 1})
    gate = _Gate(ide)
    gate.fail = 1
    monkeypatch.setattr(ide, "brief_text", gate)
    tick = _tick(2)
    ide.ideate(tick, [EXPLORE, EXPLORE], led, {})
    assert [j["plan"]["idx"] for j in ide._jobs] == [0, 1] and store.get_state(ID.K_PLATEAU)["served"] == 2
    gate.ev.set()
    _drain(ide)
    tick.research.n = 0                                              # nothing starts: only the failure is settled
    ide.ideate(tick, [], led, {})
    st = store.get_state(ID.K_PLATEAU)
    assert len(store.events("brief_error")) == 1
    failed = store.events("brief_error")[0]
    assert st["served"] - len(st.get("returned") or ()) == 1 and ide._plateau_next() is not None   # one given back
    assert ide.close(timeout=10) == 1                                # close: the unstarted one gives back too
    st = store.get_state(ID.K_PLATEAU)
    assert st["served"] - len(st.get("returned") or ()) == 0, (st, failed)
    ide._count_program("program")                                    # the program quota count rolls back too
    assert store.get_state(ID.K_PROGRAM) == {"n": 6, "p": 2}
    ide._unreserve({"kind": "program", "idx": None})
    assert store.get_state(ID.K_PROGRAM) == {"n": 5, "p": 1}
