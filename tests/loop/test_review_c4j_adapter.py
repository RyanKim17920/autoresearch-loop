"""c4j review: S3 (eval-only shim script written atomically, once) and S5 (tidy removes STRAY worktrees --
eval-only ``<node>_<c10>`` / ``root_<c10>`` / ``w###.x`` -- only when no live job, keep set or recent use holds them)."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from loop import prune as PR
from loop.adapters.nanopath_v2 import SHIM, WT_RECENT_S
from loop.contracts import RunStatus
from tests.loop.test_adapters import (SlurmRunner, base_repo, git, make_np, np_ctx, run_of)  # noqa: F401

C10 = "da22bf8472"


def _old(path: Path) -> None:
    t = time.time() - 2 * WT_RECENT_S
    for p in (path, path / ".git"):
        if p.exists():
            os.utime(p, (t, t))


def _wts(adapter, base_repo, names):
    head = git(base_repo, "rev-parse", "HEAD")
    out = {}
    for n in names:
        dest = adapter.worktrees / n
        adapter._add_worktree(dest, head, "--detach")
        _old(dest)
        out[n] = dest
    adapter._wt_used.clear()
    return out


def test_shim_script_written_once_atomically_under_threads(np_ctx, base_repo):
    adapter, _ = make_np(np_ctx, base_repo)
    script = adapter.state / "evalonly" / "shim.py"
    seen = []

    def go():
        for _ in range(20):
            p = adapter._shim_script()
            seen.append(p.read_text() == SHIM)
    ts = [threading.Thread(target=go) for _ in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert all(seen) and script.read_text() == SHIM
    assert not [p for p in script.parent.iterdir() if p.name.endswith(".tmp")]
    mtime = script.stat().st_mtime_ns
    adapter._shim_script()
    assert script.stat().st_mtime_ns == mtime                       # same content: never rewritten
    script.write_text("stale")
    assert adapter._shim_script().read_text() == SHIM               # changed content: replaced


def test_tidy_stray_worktrees_rules(np_ctx, base_repo):
    """Removed: an old eval-only / w.x worktree whose node is not kept. Kept: a kept node, a live job's cwd, a
    recently used one; never _research / _graft / root / crn-*. dry_run removes nothing."""
    runner = SlurmRunner()
    adapter, _ = make_np(np_ctx, base_repo, runner=runner)
    names = [f"n0094_{C10}", f"n0095_{C10}", f"n0096_{C10}", f"root_{C10}", "w001.x", "w002.x", "w001.p00"]
    wts = _wts(adapter, base_repo, names)
    for keep in ("_research", "_graft", "root", "crn-abc"):
        (adapter.worktrees / keep).mkdir(parents=True, exist_ok=True)
    runner.replies["squeue -u"] = (0, f"{wts[f'n0096_{C10}']}\n/elsewhere\n", "")
    adapter._wt_used["w002.x"] = time.time()                         # launched just now

    dry = adapter.tidy({}, [], [], dry_run=True, keep={"n0095"})
    assert dry["stray"] == 3 and dry["stray_removed"] == 0 and dry["stray_kept"] == 3
    assert all(p.exists() for p in wts.values())

    n = adapter.tidy({}, [], [], keep={"n0095"})
    assert n["stray"] == 3 and n["stray_removed"] == 3
    gone = {k for k, p in wts.items() if not p.exists()}
    assert gone == {f"n0094_{C10}", f"root_{C10}", "w001.x"}
    assert str(wts[f"n0094_{C10}"]) not in git(base_repo, "worktree", "list")
    for keep in ("_research", "_graft", "root", "crn-abc"):
        assert (adapter.worktrees / keep).exists()


def test_tidy_stray_squeue_failure_or_no_keep_removes_nothing(np_ctx, base_repo):
    runner = SlurmRunner({"squeue -u": (1, "", "slurm down")})
    adapter, _ = make_np(np_ctx, base_repo, runner=runner)
    wts = _wts(adapter, base_repo, [f"n0094_{C10}"])
    n = adapter.tidy({}, [], [], keep=set())
    assert n["stray"] == 0 and n["stray_kept"] == 1 and wts[f"n0094_{C10}"].exists()
    runner.replies["squeue -u"] = (0, "", "")
    assert adapter.tidy({}, [], [])["stray"] == 0                    # old callers (no keep): skipped
    assert adapter.tidy({}, [], [], keep=set())["stray_removed"] == 1


def test_stray_keep_is_incumbent_plus_active_runs(monkeypatch):
    monkeypatch.setattr(PR, "classify", lambda *a: ({"n1"}, {"root", "n7"}, set()))
    seen = {}

    class Store:
        def runs(self, status=()):
            seen["status"] = set(status)
            return [SimpleNamespace(node_id="n5")]
    keep = PR.stray_keep(None, Store(), SimpleNamespace(incumbent_id="n9"), 0.0)
    assert keep == {"n9", "n5"}   # c5: protected nodes no longer keep their stray eval worktrees and seen["status"] == {RunStatus.RUNNING, RunStatus.SUBMITTING}
