"""v3-c8a B5: parallel submits -- worktree adds of different nodes share the adapter's worktree lock (a remove / prune
still excludes them), config-only wave points / key grafts are committed with git plumbing (no worktree)."""

from __future__ import annotations

import dataclasses
import threading
import time

import yaml

from loop import runner as R
from loop.adapters.nanopath_v2 import LAUNCHER, WorktreeLock
from loop.contracts import Stage
from loop.core import factorial as F
from loop.spec import PoolCfg, SlurmCfg

from .test_adapters import SlurmRunner, base_repo, git, make_np, np_ctx  # noqa: F401 - fixtures
from .test_runner import _node

SLOW_S = 1.0


class SlowRunner(SlurmRunner):
    """Real git; every ``git worktree add`` and every sbatch launch sleeps SLOW_S (virtiofs / slurmctld latency);
    the peak number of concurrent sleepers is recorded."""

    def __init__(self):
        super().__init__({LAUNCHER: lambda argv: (0, f"Submitted batch job {self._job()}\n", "")})
        self.live = self.peak = self.n = 0
        self.mu = threading.Lock()

    def _job(self):
        with self.mu:
            self.n += 1
            return str(1000 + self.n)

    def run(self, argv, *, cwd=None, timeout=None, env=None):
        argv = list(argv)
        slow = argv[0] == LAUNCHER or (argv[0] == "git" and "worktree" in argv and "add" in argv)
        if slow:
            with self.mu:
                self.live += 1
                self.peak = max(self.peak, self.live)
            time.sleep(SLOW_S)
            with self.mu:
                self.live -= 1
        return super().run(argv, cwd=cwd, timeout=timeout, env=env)


def _book(np_ctx, store, adapter, workers):
    ctx = dataclasses.replace(np_ctx, cfg=dataclasses.replace(np_ctx.cfg, slurm=SlurmCfg(
        pools=(PoolCfg("primary", slots=32),), submit_workers=workers)))
    return R.RunBook(ctx, store, adapter)


def test_twelve_new_node_submits_run_six_wide_under_a_fake_slow_sbatch(np_ctx, base_repo, store):
    """Live 10-14: ~25 s per new node's worktree add + ~10 s per sbatch, serial under one lock: a 20-job tick spent
    193 s in launch.submit. 12 NEW nodes, each add + sbatch sleeping 1 s: serial >= 24 s; 6 workers with shared
    adds finish in ~2 rounds of 2 s."""
    adapter, runner = make_np(np_ctx, base_repo, runner=SlowRunner())
    head = git(base_repo, "rev-parse", "HEAD")
    book = _book(np_ctx, store, adapter, 6)
    nodes = [_node(store, f"n{i:02d}", commit=head) for i in range(12)]
    rows = [(book.prepare_launch(n, Stage.SCREEN, 11), n) for n in nodes]
    t0 = time.monotonic()
    got = book.submit_prepared(rows)
    dt = time.monotonic() - t0
    assert all(not isinstance(g, BaseException) and g.job_id for g in got), got
    assert runner.peak >= 4                               # adds AND sbatches overlap across nodes
    assert dt < 12 * 2 * SLOW_S / 2, dt                   # >= 2x faster than serial (24 s)
    assert len({g.job_id for g in got}) == 12             # one job per row, receipts intact
    for n in nodes:                                       # every worktree at its commit
        assert git(adapter.worktrees / n.id, "rev-parse", "HEAD") == head


def test_the_worktree_lock_lets_adds_share_and_a_prune_exclude_them():
    lk, order = WorktreeLock(), []
    inside = threading.Barrier(3)

    def add(i):
        with lk.shared():
            inside.wait(timeout=5)                          # all three readers inside at once
            order.append(f"add{i}")
    ts = [threading.Thread(target=add, args=(i,)) for i in range(3)]
    [t.start() for t in ts]
    [t.join(5) for t in ts]
    assert sorted(order) == ["add0", "add1", "add2"]
    with lk:                                                # exclusive: reentrant, may nest a shared section
        with lk:
            with lk.shared():
                pass
    got = []
    with lk.shared():                                       # a shared holder may upgrade when it is alone
        with lk:
            got.append("upgraded")
    assert got == ["upgraded"]
    seen = []
    with lk:
        t = threading.Thread(target=lambda: (lk.shared().__enter__(), seen.append("in")))
        t.start()
        time.sleep(0.1)
        assert seen == []                                   # an add waits for the prune
    t.join(5)
    assert seen == ["in"]


def test_a_config_only_wave_point_is_committed_without_a_worktree(np_ctx, base_repo):
    adapter, runner = make_np(np_ctx, base_repo)
    head = git(base_repo, "rev-parse", "HEAD")
    sha, why = F.build_point(adapter, head, {"train.lr": 3e-4}, [F.Knob("train.lr", 1e-4, 3e-4, "log")], {},
                             "w001-p0", adapter.config_rel)
    assert why == "" and sha and sha != head
    assert not any("worktree" in c["argv"] for c in runner.calls)          # plumbing only
    cfg = yaml.safe_load(git(base_repo, "show", f"{sha}:configs/main.yaml"))
    assert cfg["train"]["lr"] == 3e-4 and cfg["model"]["type"] == "vit"
    assert git(base_repo, "rev-parse", f"{sha}^") == head
    assert git(base_repo, "diff", "--name-only", head, sha) == "configs/main.yaml"
    assert git(base_repo, "status", "--porcelain") == ""                   # the repo's own index untouched
    same, _ = F.build_point(adapter, head, {"train.lr": 1e-4}, [F.Knob("train.lr", 1e-4, 3e-4, "log")], {},
                            "w001-p1", adapter.config_rel)
    assert same == head                                                     # nothing changed: the carrier


def test_graft_keys_writes_the_key_patch_without_a_worktree(np_ctx, base_repo):
    adapter, runner = make_np(np_ctx, base_repo)
    head = git(base_repo, "rev-parse", "HEAD")
    a = adapter.commit_files(head, {"configs/main.yaml": (base_repo / "configs/main.yaml").read_text()
                                    .replace("lr: 0.0001", "lr: 0.0005")}, message="a")
    sha, how = adapter.graft_keys(a, ["train.lr"], head, "k1")
    assert how == "applied" and yaml.safe_load(git(base_repo, "show", f"{sha}:configs/main.yaml"))["train"]["lr"] == 5e-4
    assert git(base_repo, "rev-parse", f"loop/{adapter._tag}-eo-k1") == sha
    assert adapter.graft_keys(a, ["train.lr"], head, "k1") == (sha, "applied")         # reused by its branch
    assert adapter.graft_keys(a, ["train.lr"], sha, "k2") == ("", "contained")
    assert not any("worktree" in c["argv"] for c in runner.calls)
