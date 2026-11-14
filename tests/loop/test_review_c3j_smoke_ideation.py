"""Pre-deploy review c3j: smoke checkpoint pruning, the ideation per-view caches, the narrowed probe-smoke trigger."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import yaml

import loop.core.ideation as ID
import loop.prune as P
from tests.loop.test_adapters import make_ctx
from tests.loop.test_probe_smoke import FAMS, LAUNCHER_SH, SlurmRunner, edit, np_probe, set_mode  # noqa: F401
from tests.loop.test_adapters import base_repo, research_wt  # noqa: F401 (fixtures)
from loop.adapters.nanopath_v2 import LAUNCHER


# ------------------------------------------------------------------ 9: smoke checkpoints
def _ckpt(root: Path, wt: str, age_h: float, now: float, sub: str = "") -> Path:
    p = root / "smoke" / wt / "out" / sub / "latest.pt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x" * 10)
    os.utime(p, (now - 3600 * age_h,) * 2)
    return p


def test_smoke_plan_picks_only_old_smoke_checkpoints(tmp_path):
    ctx = make_ctx(tmp_path / "state")
    now = 1e9
    old = _ckpt(Path(ctx.state_dir), "a", 7, now, "run1")
    new = _ckpt(Path(ctx.state_dir), "b", 1, now)
    assert [p for p, _ in P.smoke_plan(ctx, now)] == [old]
    assert new.exists()


def test_pruner_deletes_old_smoke_checkpoints_and_respects_dry_run(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path / "state")
    now = 1e9
    old = _ckpt(Path(ctx.state_dir), "a", 7, now)
    logs = []
    store = SimpleNamespace(log=lambda *a, **k: logs.append((a, k)), get_state=lambda k: None)
    monkeypatch.setattr(P, "plan", lambda *a: ([], 0))
    pr = P.Pruner(ctx, store, lambda: now)
    object.__setattr__(ctx.cfg.prune, "dry_run", True)
    pr._pass(None, now)
    assert old.exists()
    object.__setattr__(ctx.cfg.prune, "dry_run", False)
    object.__setattr__(ctx.cfg.prune, "git", False)
    pr._pass(None, now)
    assert not old.exists()


def test_a_new_smoke_start_removes_the_stale_checkpoint(np_probe):  # noqa: F811
    adapter, runner, base = np_probe
    wt = research_wt(adapter, base)
    edit(wt, "train.py", "lr = 0.1", "lr = 0.2")
    (Path(wt.path) / "submit").mkdir()
    (Path(wt.path) / LAUNCHER).write_text(LAUNCHER_SH.replace('touch "$d/latest.pt"', "true"))
    stale = adapter.state / "smoke" / Path(wt.path).name / "out" / "sub" / "latest.pt"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old")
    adapter.smoke(wt)
    assert not stale.exists()


# ------------------------------------------------------------------ 10: levers cache key
def _view(n_obs: int, incumbent: str = "a"):
    led = SimpleNamespace(obs=tuple(SimpleNamespace(ts=float(i)) for i in range(n_obs)), runs=(), nodes={},
                          carriers=(), incumbents=(), audits=(), draws=(), epoch="")
    return SimpleNamespace(led=led, weights={}, beliefs={}, incumbent=incumbent)


def test_levers_cache_is_keyed_by_view_content_not_object_id(monkeypatch):
    from loop.core import levers as LV
    calls = []
    monkeypatch.setattr(LV, "retest", lambda *a: calls.append(1) or [])
    monkeypatch.setattr(LV, "shortlist", lambda *a: ())
    monkeypatch.setattr(LV, "text", lambda *a: f"t{len(calls)}")
    ide = ID.Ideation(SimpleNamespace(spec=None), store=None)
    v1 = _view(3)
    assert ide.levers(v1) == ide.levers(_view(3))          # an equal view: cached (a different object)
    assert len(calls) == 1
    assert ide.levers(_view(4))[0] == "t2"                 # a grown ledger: recomputed even if id() were reused
    assert ID._view_token(_view(3, "a")) != ID._view_token(_view(3, "b"))


# ------------------------------------------------------------------ 11: probe smoke trigger
def test_generic_model_keys_and_non_probe_model_py_edits_do_not_trigger(np_probe):  # noqa: F811
    adapter, _, base = np_probe
    wt = research_wt(adapter, base)
    edit(wt, "model.py", "return 1", "return 2")                      # ViT.build_blocks: not probe-facing
    assert adapter.probe_smoke_why(wt) == ""
    cfg = Path(wt.path) / "configs/main.yaml"
    tree = yaml.safe_load(cfg.read_text())
    tree["model"]["type"] = "vit_b"
    cfg.write_text(yaml.safe_dump(tree))
    assert adapter.probe_smoke_why(wt) == ""                           # generic model.* key
    tree["data"] = {"mean": [0.1], "std": [0.2]}
    cfg.write_text(yaml.safe_dump(tree))
    assert adapter.probe_smoke_why(wt) == "config data.mean changed"
    tree["data"] = {"other": 1}
    cfg.write_text(yaml.safe_dump(tree))
    assert adapter.probe_smoke_why(wt) == ""                           # data.* other than mean / std


def test_probe_facing_model_functions_trigger(np_probe):  # noqa: F811
    adapter, _, base = np_probe
    wt0 = research_wt(adapter, base, "r0")
    edit(wt0, "model.py", "return x\n", "return x + 1\n")             # ViT.encode_image
    assert adapter.probe_smoke_why(wt0) == "model.py changed"
    wt = research_wt(adapter, base, "r2")
    edit(wt, "model.py", "return x * 2", "return x * 3")               # ViT.forward
    assert adapter.probe_smoke_why(wt) == "model.py::ViT.forward changed"   # its eval_readout span
    object.__setattr__(adapter.ctx.cfg.slurm, "smoke_probe_funcs", ("only_this",))
    assert adapter.probe_smoke_why(wt0) == ""                          # the function list is configurable
