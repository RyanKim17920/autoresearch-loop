"""loop.selfcheck (11-08): the LLM-free end-to-end preflight against the real task config -- it would have caught each
defect behind the 1,207 failed nanobrain sessions -- and `loop run`'s startup gate that holds research on a failure."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from loop import selfcheck as SC
from loop.core import bold as BOLD
from loop.core import tested as TS
from loop.spec import load_context

from .test_productive_sessions import REPO, nb  # noqa: F401  (the fixture)


class Thin:
    """The 11-08 history: 5 nodes, 0 clusters."""

    def __init__(self):
        self.by_id = {n: {"id": n, "tag": "x/y", "desc": "d", "cluster": ""} for n in ("root", "n0001", "n0002")}
        self.cluster_ids = {}

    def exists(self, i):
        return i in self.by_id

    def node_cluster(self, i):
        return ""


def test_selfcheck_slate_passes_on_a_thin_history_and_catches_the_11_08_deadlock(nb, monkeypatch):
    ctx, store = nb
    monkeypatch.setattr(SC, "history_table", lambda *a, **k: (Thin(), ""))
    got = SC.check_slate(ctx, store, None)
    assert got.status == SC.PASS and "0 clusters" in got.detail, got
    # the pre-271fd26 rule: '>= 3 distinct clusters' never capped at what the history has
    real = BOLD.Slate._pf
    monkeypatch.setattr(BOLD.Slate, "_pf", lambda self: {k: v for k, v in (real(self) or {}).items()
                                                         if k not in ("available_clusters", "n_ids")} or None)
    monkeypatch.setattr(BOLD, "thin_history", lambda tab, need: False)
    bad = SC.check_slate(ctx, store, None)
    assert bad.status == SC.FAIL and "DISTINCT history clusters" in bad.detail, bad
    assert not store.get_state(TS.K_PROPOSED)                         # the selfcheck wrote nothing


def test_selfcheck_nan_scan_and_breaker(nb, monkeypatch):
    ctx, store = nb
    from loop.adapters.nanobrain import NanobrainAdapter
    a = object.__new__(NanobrainAdapter)
    assert SC.check_nan_scan(a).status == SC.PASS
    monkeypatch.setattr(NanobrainAdapter, "nan_scan", staticmethod(lambda t: "nan" in t))   # the pre-fix scan
    assert SC.check_nan_scan(a).status == SC.FAIL
    assert SC.check_breaker(ctx, store).status == SC.PASS
    store.set_state("research.breaker", {"tripped": "6 consecutive research sessions produced nothing"})
    assert SC.check_breaker(ctx, store).status == SC.FAIL


def test_selfcheck_locks(nb):
    ctx, store = nb
    ok = SimpleNamespace(error="", verify=lambda: [], head="abc")
    adapter = SimpleNamespace(lock_violations=lambda c: [], root_commit=lambda: "r")
    assert SC.check_locks(ctx, adapter, ok).status == SC.PASS
    bad = SimpleNamespace(error="", verify=lambda: ["loop/core/test.py"], head="abc")
    assert SC.check_locks(ctx, adapter, bad).status == SC.FAIL
    assert SC.check_locks(ctx, SimpleNamespace(lock_violations=lambda c: ["probe.py"], root_commit=lambda: "r"),
                          ok).status == SC.FAIL


TRAIN = '''
def main(args):
    assert not (args.native and args.plot_period), "set plot_period 0 with native"
    if args.get("fino3d") and args.fino3d.enabled:
        assert args.native, "fino3d: native training only"
    for x in range(3):
        assert x < 0          # not args-only: never evaluated
'''
FAKE_OMEGACONF = '''
import yaml
class C(dict):
    __getattr__ = lambda self, k: self.get(k)
def _wrap(d):
    return C({k: _wrap(v) for k, v in d.items()}) if isinstance(d, dict) else d
def _merge(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return _wrap(out)
class OmegaConf:
    load = staticmethod(lambda p: _wrap(yaml.safe_load(open(p)) or {}))
    unsafe_merge = staticmethod(_merge)
    @staticmethod
    def from_dotlist(xs):
        out = {}
        for x in xs:
            k, v = x.split("=", 1)
            out[k] = yaml.safe_load(v)
        return _wrap(out)
'''
MODEL = '''
class ViTMAE3D:
    @classmethod
    def from_config(cls, cfg):
        assert cfg.total_steps, "no steps"
        m = cls()
        return m
    def parameters(self):
        return []
'''


def _repo(tmp_path):
    repo = tmp_path / "proj"
    (repo / "src" / "nanobrain" / "config").mkdir(parents=True)
    (repo / "src" / "nanobrain" / "train.py").write_text(TRAIN)
    (repo / "src" / "nanobrain" / "model.py").write_text(MODEL)
    (repo / "src" / "nanobrain" / "__init__.py").write_text("")
    (repo / "src" / "omegaconf.py").write_text(FAKE_OMEGACONF)
    (repo / "src" / "nanobrain" / "config" / "default_pretrain.yaml").write_text(
        "native: false\nplot_period: 100\ntotal_steps: 1000\nseed: 0\n")
    (repo / "src" / "nanobrain" / "config" / "root.yaml").write_text("native: true\nplot_period: 0\ntotal_steps: 22500\n")
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t"}
    for c in (["init", "-q"], ["add", "-A"], ["commit", "-qm", "root"]):
        subprocess.run(["git", "-C", str(repo), *c], check=True, env=env, capture_output=True)
    sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    return repo, sha


@pytest.mark.parametrize("plot_period,status", [(0, SC.PASS), (1_000_000, SC.FAIL)])
def test_selfcheck_train_config_runs_trains_asserts_on_the_smoke_overrides(tmp_path, monkeypatch, plot_period, status):
    """11-08: SMOKE_OVERRIDES plot_period 1_000_000 broke `assert not (native and plot_period)` -- every smoke of the
    native root died rc=1. The probe merges the smoke config like train.py and evaluates that assert."""
    from loop.adapters.nanobrain import NanobrainAdapter
    repo, sha = _repo(tmp_path)
    a = object.__new__(NanobrainAdapter)
    a.repo, a.config_rel = str(repo), "src/nanobrain/config/root.yaml"
    a.root_commit = lambda: sha
    monkeypatch.setattr(NanobrainAdapter, "SMOKE_OVERRIDES", {**NanobrainAdapter.SMOKE_OVERRIDES,
                                                              "plot_period": plot_period})
    ctx = SimpleNamespace(cfg=SimpleNamespace(project=SimpleNamespace(python=sys.executable)))
    got = SC.check_train_config(ctx, a, timeout=120)
    assert got.status == status, got
    if status == SC.FAIL:
        assert "plot_period" in got.detail


def test_train_verdict_rules():
    assert SC.train_verdict({"smoke": {"asserts": [], "model": ""}, "base": {"asserts": [], "model": ""}}).status == SC.PASS
    assert SC.train_verdict({"smoke": {"merge": "bad yaml"}}).status == SC.FAIL
    assert SC.train_verdict({"smoke": {"asserts": [], "model": "boom"}, "base": {"asserts": [], "model": ""}}).status == SC.FAIL
    assert SC.train_verdict({"smoke": {"asserts": [], "model": "no torch"},
                             "base": {"asserts": [], "model": "no torch"}}).status == SC.WARN


def test_the_startup_gate_alerts_and_holds_research(nb, tmp_path):
    ctx, store = nb
    lines: list[str] = []
    fail = lambda *a, **k: [SC.Check("slate", SC.FAIL, "every slate rejected"), SC.Check("breaker", SC.PASS, "ok")]
    why = SC.gate(ctx, store, None, out=lines.append, runner=fail)
    assert why.startswith("selfcheck failed: slate") and "FAIL (slate)" in lines[-1]
    assert [e for e in store.events("ALERT_selfcheck")] and store.get_state(SC.K_SELFCHECK)["ok"] is False
    ok = lambda *a, **k: [SC.Check("slate", SC.PASS, "fine")]
    assert SC.gate(ctx, store, None, out=lines.append, runner=ok) == ""
    Path(ctx.state_dir).mkdir(parents=True, exist_ok=True)
    (Path(ctx.state_dir) / SC.SKIP_FILE).write_text("")
    assert SC.gate(ctx, store, None, out=lines.append, runner=fail) == "" and not (Path(ctx.state_dir) / SC.SKIP_FILE).exists()


def test_live_holds_research_sessions_when_the_selfcheck_fails(monkeypatch, tmp_path):
    from loop import tick as TK
    seen = {}

    class FakeTick:
        def __init__(self, *a, **k):
            self.research = SimpleNamespace(breaker="", inflight=[], draining=False)
            self.selfcheck_hold = ""
            self.pruner = SimpleNamespace(stop=lambda timeout=0: None)
            seen["tick"] = self

        def close(self):
            pass

    monkeypatch.setattr(TK, "Tick", FakeTick)
    monkeypatch.setattr(SC, "gate", lambda *a, **k: "selfcheck failed: slate: x")
    ctx = load_context(REPO / "tasks" / "nanobrain.yaml", state_dir=tmp_path, require_measured=False)
    TK.live(ctx, tmp_path / "x.db", None, None, None, max_ticks=0, selfcheck=True, out=lambda m: None,
            watchdog=SimpleNamespace(notify=lambda *a: None, ping=lambda: None))
    t = seen["tick"]
    assert t.selfcheck_hold.startswith("selfcheck failed") and t.research.breaker == t.selfcheck_hold
