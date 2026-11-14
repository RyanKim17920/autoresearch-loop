"""c4j pre-deploy review fixes: S4 (diagnose: one submit under concurrency, a failed squeue is not death, error
reports are never cached), S6 (program leave-one-out capped and counted, ``off`` restricted to own knobs), S7 (a
prompt-arm patch that no longer applies ends the arm, logged; the arm files are written atomically)."""

from __future__ import annotations

import dataclasses
import json
import threading
from pathlib import Path
from types import SimpleNamespace

from loop.agents import diagnose as DX
from loop.core import data as D
from loop.core import meta as MT
from loop.core import program as PG

from .core.test_c4i import _Runner, _icfg, _prog_env


# ---------------------------------------------------------------------------------------------- S4
def _diag_env(ctx, store, tmp_path, runner):
    run = tmp_path / "run"
    run.mkdir()
    (run / "latest.pt").write_text("x")
    wt = tmp_path / "wt"
    (wt / "submit").mkdir(parents=True)
    (wt / "submit" / "train_1gpu.sbatch").write_text("#!/bin/bash\n#SBATCH --gpus=1\necho\n")
    adapter = SimpleNamespace(runner=runner, config_rel="configs/main.yaml", _node_worktree=lambda n, c: wt,
                              _sbatch=lambda cfg, out, name, pool="": (["x"], {}))
    DX.record(ctx, store, SimpleNamespace(runs=(D.Run("r", "inc", 1, "screen", "done", 1, 0, 0, str(run), 0),)),
              "inc", "c-inc", None)
    return adapter


def test_a_failing_squeue_is_retried_and_unknown_never_counts_as_dead(monkeypatch):
    monkeypatch.setattr(DX, "ALIVE_RETRY_S", 0.0)

    class R:
        def __init__(self, replies):
            self.replies, self.n = list(replies), 0

        def run(self, argv, **kw):
            self.n += 1
            return self.replies.pop(0) if self.replies else (1, "", "slurm_load_jobs error: Socket timed out")
    r = R([(1, "", "Socket timed out"), (0, "RUNNING\n", "")])
    assert DX._alive(SimpleNamespace(runner=r), "5") and r.n == 2                    # retried, then confirmed
    r = R([])
    assert DX._alive(SimpleNamespace(runner=r), "5") and r.n == DX.ALIVE_TRIES          # still failing: unknown = alive
    r = R([(1, "", "slurm_load_jobs error: Invalid job id specified")])
    assert not DX._alive(SimpleNamespace(runner=r), "5") and r.n == 1                   # squeue's own verdict: gone
    assert not DX._alive(SimpleNamespace(runner=R([(0, "", "")])), "5")


def test_concurrent_diagnose_calls_submit_the_feature_job_once(ctx, store, tmp_path):
    runner = _Runner()
    adapter = _diag_env(ctx, store, tmp_path, runner)
    gate = threading.Barrier(4)

    def call():
        gate.wait()
        DX.features(ctx, store, adapter, wait_s=0, sleep=lambda s: None, clock=lambda: 0.0)
    ts = [threading.Thread(target=call) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len([c for c in runner.calls if c[0][0] == "sbatch"]) == 1


def test_an_error_report_is_shown_but_never_cached(ctx, store, tmp_path):
    runner = _Runner()
    adapter = _diag_env(ctx, store, tmp_path, runner)
    DX.features(ctx, store, adapter, wait_s=0, sleep=lambda s: None, clock=lambda: 0.0)
    out_dir = Path(ctx.state_dir) / "diagnose" / "inc"
    (out_dir / DX.ERR_NAME).write_text(json.dumps({"error": "RuntimeError: CUDA busy"}))
    runner.alive = False
    got = DX.features(ctx, store, adapter, wait_s=0, sleep=lambda s: None, clock=lambda: 0.0)
    assert "FAILED" in got and "CUDA busy" in got
    assert len([c for c in runner.calls if c[0][0] == "sbatch"]) == 1
    # the error was not cached: the next call (job dead, no report) resubmits
    assert not (out_dir / "features.json").exists()
    DX.features(ctx, store, adapter, wait_s=0, sleep=lambda s: None, clock=lambda: 0.0)
    n = len([c for c in runner.calls if c[0][0] == "sbatch"])
    assert n == 2 and not (out_dir / DX.ERR_NAME).exists()                             # resubmitted, old error cleared
    # a legacy error report sitting in features.json is dropped, not served
    (out_dir / "features.json").write_text(json.dumps({"error": "boom"}))
    runner.alive = True
    got = DX.features(ctx, store, adapter, wait_s=0, sleep=lambda s: None, clock=lambda: 0.0)
    assert "boom" not in got and not (out_dir / "features.json").exists()


def test_feature_diag_writes_a_failure_beside_the_report_not_as_it(tmp_path, monkeypatch):
    from loop.adapters import feature_diag as FD
    monkeypatch.setattr(FD, "run", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no gpu")))
    out = tmp_path / "features.json"
    assert FD.main(["x", "wt", "cfg", "ckpt", str(out)]) == 1
    assert not out.exists() and "no gpu" in json.loads((tmp_path / DX.ERR_NAME).read_text())["error"]


# ---------------------------------------------------------------------------------------------- S6
def _three(store):
    comps = [{"family": "objective", "mechanism": f"m{i}", "share": s,
              "knobs": [{"key": f"k{i}.x", "lo": 1, "hi": 4}], "off": {f"k{i}.x": 0, "other.key": 9}}
             for i, s in enumerate([0.1, 0.4, 0.2, 0.3])]
    store.set_state("ideation:p1", {"components": comps})


def test_leave_one_out_is_capped_by_share_and_off_names_only_own_knobs(ctx, store, monkeypatch):
    data, post, adapter, built = _prog_env(store, monkeypatch, (0.006, 4e-6))       # +3 sd
    _three(store)
    store.set_state(PG.K_LADDERS, {"p1": []})
    PG.leave_one_out(ctx, store, adapter, data, post, {"p1": "screening"}, 10.0)
    loo = store.get_state(PG.K_LOO)["p1"]
    kids = {i: n for i, n in loo.items() if not n.startswith("-")}
    assert set(kids) == {"1", "3", "2"} and loo["0"].startswith("-: over the LOO budget")   # largest shares first
    for i, n in kids.items():
        assert store.get_state(f"config_diff:{n}") == {f"k{i}.x": 0}               # other.key dropped


def test_leave_one_out_counts_in_the_program_trial_budget(ctx, store, monkeypatch):
    data, post, adapter, built = _prog_env(store, monkeypatch, (0.006, 4e-6))
    _three(store)
    c = _icfg(ctx, program_wave_max_points=3, program_loo_max=2)
    store.set_state(PG.K_LADDERS, {"p1": ["d1", "d2", "d3"]})       # 3 doses already (over a 3-point budget)
    PG.leave_one_out(c, store, adapter, data, post, {"p1": "screening"}, 10.0)
    loo = store.get_state(PG.K_LOO)["p1"]
    # node 1 + doses 3 + LOO <= 3 + 2 -> one LOO child
    assert sum(not n.startswith("-") for n in loo.values()) == 1


# ---------------------------------------------------------------------------------------------- S7
def test_an_arm_whose_patch_no_longer_applies_is_ended_and_logged(store, tmp_path, ctx, monkeypatch):
    base = tmp_path / "base"
    base.mkdir()
    for n in ("ideation.md", "researcher.md"):
        (base / n).write_text(f"{n} line one\nline two\n")
    monkeypatch.setattr(MT, "PROMPTS", base)
    meta = MT.Meta(dataclasses.replace(ctx, state_dir=str(tmp_path / "st")), store, locks=None)
    meta.accept({"diagnosis": "d", "proposals": [{"kind": "prompt", "path": "prompts/ideation.md",
                 "patch": {"find": "line two", "replace": "line two ZZZ"}, "rationale": "r", "predicted_effect": "e"}]},
                None)
    assert (store.get_state(MT.K_AB) or {}).get("arm") == 1 and meta.versions.skipped(1) == []
    (base / "ideation.md").write_text("ideation.md rewritten\n")       # the find text is gone: the patch is skipped
    assert meta.versions.skipped(1) == ["v1:ideation.md"]
    d = meta.arm_for("req-1")
    assert not (store.get_state(MT.K_AB) or {}).get("arm")
    assert next(r for r in store.config_versions("prompt") if r["v"] == 1)["state"] == "stale"
    assert any(e.get("arm") == 1 for e in store.events("meta_ab_stale"))
    assert d is None or "v1" not in str(d)
    assert not list((tmp_path / "st" / "prompts").rglob("*.tmp"))
