"""c5 misc fixes: R2 program dose types, R5 stray_keep, R6 retry after an accepted finish, R7 post-calibrate reload."""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import claude_agent_sdk as cas
import pytest

import llm
from loop import __main__ as M
from loop.agents.llm import ToolGate
from loop.contracts import RunStatus, UnmeasuredLoopNoise
from loop.core import program as P
from loop import prune as PR


# ------------------------------------------------------------------------------------------------ R2
def test_int_knob_doses_are_ints_deduped_and_cur_excluded():
    xs = P.typed_doses(P.doses(1.0, 20.0, 6, "lin", 8), 8)
    assert xs and all(type(x) is int for x in xs) and len(set(xs)) == len(xs) and 8 not in xs
    # rungs that collapse after rounding are deduped
    assert P.typed_doses([20.0, 20.4, 19.8], 5) == [20]


def test_float_and_bool_knobs_keep_their_type():
    assert P.typed_doses([0.5, 1.0], 0.25) == [0.5, 1.0]
    assert all(type(x) is float for x in P.typed_doses([1, 2], 0.25))
    assert P.typed_like(1.0, True) is True and P.typed_like(0, False) is False
    assert P.typed_like(0.5, True) is None          # a bool knob has no dose between


def test_typed_like_rejects_wrong_shapes():
    assert P.typed_like("x", 3) is None
    assert P.typed_like(float("inf"), 3) is None
    assert P.typed_like("a", "b") == "a"
    assert P.typed_like(20.0, None) == 20.0


# ------------------------------------------------------------------------------------------------ R5
def test_stray_keep_ignores_protected(monkeypatch):
    monkeypatch.setattr(PR, "classify", lambda *a: ({"n1", "n2"}, {"root"}, set()))

    class Store:
        def runs(self, status=()):
            return [SimpleNamespace(node_id="n5")]
    assert PR.stray_keep(None, Store(), SimpleNamespace(incumbent_id="n9"), 0.0) == {"n9", "n5"}
    assert PR.stray_keep(None, Store(), None, 0.0) == {"n5"}


# ------------------------------------------------------------------------------------------------ R6
def test_gate_after_finish_refuses_and_sdk_session_does_not_rerun_after_accepted_finish(monkeypatch):
    gate = ToolGate(lambda n, a: "ok", {"type": "object"}, 1e18)
    assert gate("finish", {}).startswith("finish accepted") and gate("bash", {}) == "session already finished; stop now"
    monkeypatch.setattr(llm.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(llm, "_ledger_record", lambda **kw: None)
    monkeypatch.delenv("OVFX_PAUSE_BACKENDS", raising=False)
    calls = {"n": 0}

    async def dies_after_finish(*, prompt, options=None, transport=None):
        calls["n"] += 1
        raise RuntimeError("connection refused (CLI subprocess died)")
        yield  # pragma: no cover

    monkeypatch.setattr(cas, "query", dies_after_finish)
    res = llm.sdk_session("go", role="t", done=lambda: gate.finish is not None)
    assert calls["n"] == 1 and res["ok"] is True          # the accepted finish stands; no re-run into a dead gate
    calls["n"] = 0
    res = llm.sdk_session("go", role="t", done=lambda: False)     # nothing accepted: still the bounded retry
    assert res["ok"] is False and calls["n"] == llm.OPUS_SDK_DEFAULT_RETRIES + 1


# ------------------------------------------------------------------------------------------------ R7
def test_reload_measured_survives_a_yaml_newer_than_the_code(ctx):
    upd = {("loop", "panel_sd"): 0.004, ("noise", "seed_transfer_sd"): 0.003,
           **{("loop", "axis_sd", a): 0.01 for a in ctx.cfg.axes}}

    def bad():
        raise ValueError("loop.labless: unknown key (known: [...])")
    out = M.reload_measured(ctx, bad, upd)
    assert out.cfg.panel_sd == 0.004 and set(out.cfg.axis_sd.values()) == {0.01}
    assert out.noise.seed_transfer_sd == 0.003 and out.task_id == ctx.task_id


def test_reload_measured_passes_through_and_keeps_the_unmeasured_refusal(ctx):
    assert M.reload_measured(ctx, lambda: "fresh", {}) == "fresh"

    def unmeasured():
        raise UnmeasuredLoopNoise("still UNMEASURED")
    with pytest.raises(UnmeasuredLoopNoise):
        M.reload_measured(ctx, unmeasured, {})
