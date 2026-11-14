"""10-22 fnf deadlock: the pause judges on fresh-seed TEST gains only and branch nodes are never tested, so with no round
running nothing could lift it while judge_fnf said "keep testing". The pause episode now expires after
``ideation.fnf_pause_max_d`` days and grants ONE round."""

from __future__ import annotations

import time
from types import SimpleNamespace

from loop.core import alloc as AL
from loop.core import fnf as FN

from .test_c7i import _n, _r
from .test_fnf import env  # noqa: F401  (the fixture)

DAY = 86400.0


def _paused_store(store, *, since, rounds=2):
    store.set_state(AL.K_FNF_PAUSE, "2 closed rounds, no positive fresh-seed test gain")
    store.log("fnf_paused", why="x", trigger="cadence")
    store._db.execute("UPDATE log SET ts=? WHERE event='fnf_paused'", (since,))
    store.set_state(FN.K_FNF, {"round": rounds, "open": None, "rounds": [
        {"nodes": {f"{i}.1": ["b1"]}, "control": ["c1"], "ts": i, "closed": i + 1} for i in range(rounds)]})


def _view():
    led = SimpleNamespace(nodes={"b1": _n("b1", notes="request=r branch=1.1"), "c1": _n("c1", notes="request=s")},
                          runs=(_r("b1"), _r("c1")), obs=(), epoch="B")
    return SimpleNamespace(led=led, judged=[])


def test_the_pause_holds_until_max_d_then_grants_one_round_with_one_quiet_alert(store):
    now = time.time()
    _paused_store(store, since=now - 2 * DAY)
    assert AL.fnf_pause_expiry(store, now, 3.0) == []                             # 2 d < 3 d: still paused
    assert AL.fnf_pause(_view(), FN.state(store), 2)
    out = AL.fnf_pause_expiry(store, now + 1.5 * DAY, 3.0)                       # 3.5 d
    assert len(out) == 1 and "pause expired" in out[0]
    assert len(store.events("fnf_pause_expired")) == 1
    assert AL.fnf_pause(_view(), FN.state(store), 2) == ""                       # the round may open
    assert AL.fnf_pause_expiry(store, now + 1.6 * DAY, 3.0) == []              # once, not every tick
    assert len(store.events("fnf_pause_expired")) == 1


def test_max_d_zero_never_expires_and_an_unpaused_loop_is_untouched(store):
    now = time.time()
    _paused_store(store, since=now - 30 * DAY)
    assert AL.fnf_pause_expiry(store, now, 0.0) == [] and AL.fnf_pause(_view(), FN.state(store), 2)
    store.set_state(AL.K_FNF_PAUSE, "")
    assert AL.fnf_pause_expiry(store, now, 3.0) == [] and not store.events("fnf_pause_expired")


def test_the_grant_covers_one_round_the_closed_round_starts_a_new_episode(store):
    now = time.time()
    _paused_store(store, since=now - 5 * DAY)
    AL.fnf_pause_expiry(store, now, 3.0)
    st = FN.state(store)
    assert AL.fnf_pause(_view(), st, 2) == ""
    st["open"] = {"id": 3}                                                        # the granted round is running: judged as before
    assert AL.fnf_pause(_view(), st, 2)
    st["open"], st["rounds"] = None, st["rounds"] + [{"nodes": {"3.1": []}, "control": [], "ts": 5, "closed": now}]
    assert AL.fnf_pause(_view(), st, 2)                                           # 3 closed rounds, still no test: paused again
    store.set_state(FN.K_FNF, st)
    store.set_state(AL.K_FNF_PAUSE, "paused again")
    assert AL.fnf_pause_expiry(store, now + 0.1 * DAY, 3.0) == []                 # a fresh episode (n changed)
    assert FN.state(store)["pause_ep"]["granted"] is False


def test_the_real_ideation_forks_one_round_after_the_grant_then_pauses_again(env, monkeypatch):
    ctx, store, led, ide = env
    v = ide.view(led)
    now = time.time()
    monkeypatch.setattr(FN, "trigger", lambda *a: "cadence")
    monkeypatch.setattr(AL, "fnf_pause", AL.fnf_pause)
    monkeypatch.setattr(ide, "clock", lambda: now, raising=False)
    store.set_state(AL.K_FNF_PAUSE, "")
    st = FN.state(store)
    st["rounds"] = [{"id": i, "nodes": {}, "control": [], "ts": i, "closed": i + 1, "credit": {}, "winner": ""} for i in (1, 2)]
    store.set_state(FN.K_FNF, st)
    monkeypatch.setattr(AL, "fnf_pause", lambda v_, s_, k: ("2 closed rounds" if not (s_.get("pause_ep") or {}).get("granted")
                                                           or s_.get("open") else ""))
    assert any("paused" in a for a in ide._fnf(SimpleNamespace(dry=False), v))
    assert FN.state(store).get("open") is None
    store.set_state(AL.K_FNF_PAUSE, "2 closed rounds")
    store.log("fnf_paused", why="x", trigger="cadence")
    store._db.execute("UPDATE log SET ts=?", (now - 4 * DAY,))
    assert AL.fnf_pause_expiry(store, now, 3.0)
    ide._fnf(SimpleNamespace(dry=False), v)
    assert FN.state(store).get("open") is not None                                # exactly the granted round


def test_the_real_tick_runs_the_expiry_before_ideation(ctx, store, tmp_path):
    from loop import tick as TK
    from ..test_set1_harness import _Held
    tick = TK.Tick(ctx, store, _Held(), TK.ModuleCore(), db_path=tmp_path / "loop.db", clock=lambda: time.time() + 10 * DAY)
    _paused_store(store, since=time.time(), rounds=2)
    assert tick.ctx.cfg.ideation.fnf_pause_max_d == 3.0
    out = tick._fnf_pause_expiry()
    assert len(out) == 1 and "pause expired" in out[0]
    import inspect
    src = inspect.getsource(TK.Tick.step)
    assert src.index("_fnf_pause_expiry") < src.index("self.ideate(")
    tick.close()


def test_the_expiry_stops_granting_after_two_consecutive_granted_episodes_with_no_tested_branch_node(store):
    """S8: every grant produced untested branch nodes, so the expiry re-opened the same dead round every max_d days."""
    now = time.time()
    _paused_store(store, since=now - 5 * DAY)
    t = now
    for k in range(1, 4):                                    # episode k: grant, the granted round closes untested
        out = AL.fnf_pause_expiry(store, t, 3.0)
        if k <= 2:
            assert len(out) == 1 and "pause expired" in out[0], k
        else:                                               # the third episode: two dead grants in a row -> no grant
            assert out == [] and FN.state(store)["pause_dead"] == 2
            assert len(store.events("fnf_pause_expiry_stopped")) == 1
            assert len(store.events("fnf_pause_expired")) == 2
        st = FN.state(store)
        st["round"] = st.get("round", 0) + 1
        st["rounds"] = st["rounds"] + [{"nodes": {f"{st['round']}.1": []}, "control": [], "ts": t, "closed": t + 1}]
        store.set_state(FN.K_FNF, st)
        t += 5 * DAY
    AL.fnf_pause_expiry(store, t + 30 * DAY, 3.0)
    assert len(store.events("fnf_pause_expired")) == 2        # still stopped
    store.set_state(AL.K_FNF_PAUSE, "")                       # the pause lifted: the streak is over
    AL.fnf_pause_expiry(store, t, 3.0)
    assert FN.state(store)["pause_dead"] == 0
