"""Adapter tests: sim determinism/noise/effects/faults/resume, nanopath argv + status mapping
with a fake runner, init-compat verdicts. No SLURM, no network, no real trial.py."""

from __future__ import annotations

import dataclasses
import fnmatch
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from kernel.model import TaskSpec
from kernel.stats import NoiseModel
from loop.adapters import initcompat
from loop.adapters.nanopath_v2 import (BASE_SHA, LAUNCHER, NanopathV2Adapter, SubprocessRunner,
                                       _span, locked_source)
from loop.adapters.sim import INIT_MARKER, SYNTAX_MARKER, SimAdapter
from loop.contracts import ExecutionAdapter, JobHandle, Run, RunStatus, Stage, SubmissionRejected, VoidType, Worktree
from loop.spec import FamilyRule, HeartbeatCfg, LoopConfig, LoopContext, PoolCfg, SlurmCfg
from loop.select.intake import path_matches

REPO = Path(__file__).resolve().parents[2]
AXES = ("alpha", "beta", "gamma")
WEIGHTS = {"alpha": 0.5, "beta": 0.3, "gamma": 0.2}
AXIS_SD = {"alpha": 0.005, "beta": 0.006, "gamma": 0.007}


def make_ctx(state_dir: Path, **cfg_kw) -> LoopContext:
    spec = TaskSpec.load(REPO / "tasks" / "fake_loop.yaml")
    cfg = LoopConfig(execution_adapter="loop.adapters.sim:SimAdapter", axes=AXES, weights=WEIGHTS,
                     panel_sd=0.004, axis_sd=AXIS_SD, **cfg_kw)
    return LoopContext(spec=spec, cfg=cfg, noise=NoiseModel.from_spec(spec), task_id=spec.name,
                       spec_path=str(REPO / "tasks" / "fake_loop.yaml"), state_dir=str(state_dir))


def run_of(node: str, seed: int, stage: Stage = Stage.SCREEN, attempt: int = 0) -> Run:
    return Run(id=Run.make_id(node, stage, seed), task_id="fake-loop", node_id=node, seed=seed,
               stage=stage, status=RunStatus.SUBMITTING, attempt=attempt)


@pytest.fixture
def ctx(tmp_path):
    return make_ctx(tmp_path / "state")


@pytest.fixture
def sim(ctx):
    return SimAdapter(ctx, duration_ticks=2)


def edit_config(sim: SimAdapter, parent: str, name: str, changes: dict, *, model_py: str | None = None):
    """Worktree from ``parent``, apply flattened config ``changes``, commit. -> (commit, worktree)."""
    wt = sim.prepare_worktree(parent_commit=parent, name=name)
    path = Path(wt.path) / "config.yaml"
    cfg = yaml.safe_load(path.read_text())
    for key, val in changes.items():
        section, leaf = key.split(".")
        cfg[section][leaf] = val
    path.write_text(yaml.safe_dump(cfg))
    if model_py is not None:
        (Path(wt.path) / "model.py").write_text(model_py)
    return sim.commit(wt, message=name), wt


def finish(sim: SimAdapter, handle: JobHandle, max_polls: int = 50):
    for _ in range(max_polls):
        st = sim.poll(handle)
        if st.state is not RunStatus.RUNNING:
            return st
    raise AssertionError("job never finished")


def score_run(sim, commit, seed, node="n"):
    return finish(sim, sim.submit(run=run_of(node, seed), commit=commit)).scores


# ===================================================================== sim


def test_sim_satisfies_protocol(sim):
    assert isinstance(sim, ExecutionAdapter)


def test_sim_worktree_commit_diff(sim):
    root = sim.root_commit()
    child, wt = edit_config(sim, root, "a", {"train.lr": 5e-4})
    assert child != root
    text, paths, cfg_diff = sim.diff(wt)
    assert paths == ("config.yaml",) and "@@" in text and "+  lr: 0.0005" in text
    assert cfg_diff == {"train.lr": 5e-4}
    # identical content -> identical content-hash commit
    again, _ = edit_config(sim, root, "b", {"train.lr": 5e-4})
    assert again == child
    sim.reap_worktree(wt)
    assert not Path(wt.path).exists()


def test_sim_deterministic_and_seed_dependent(ctx, tmp_path):
    a, b = SimAdapter(ctx, duration_ticks=1), SimAdapter(make_ctx(tmp_path / "other"), duration_ticks=1)
    assert a.root_commit() == b.root_commit()
    s1, s2 = score_run(a, a.root_commit(), 11), score_run(b, b.root_commit(), 11)
    assert s1 == s2
    assert score_run(a, a.root_commit(), 12) != s1
    st = finish(a, a.submit(run=run_of("n", 11, attempt=1), commit=a.root_commit()))
    assert st.scores == s1                              # attempt is not part of the noise seed


def test_sim_seed_noise_sd_matches_axis_sd(ctx):
    sim = SimAdapter(ctx, duration_ticks=1)
    rows = [score_run(sim, sim.root_commit(), seed) for seed in range(400)]
    for axis in AXES:
        vals = [r[axis] for r in rows]
        assert statistics.pstdev(vals) == pytest.approx(AXIS_SD[axis], rel=0.15)
        assert statistics.mean(vals) == pytest.approx(sim.base_score(axis), abs=4 * AXIS_SD[axis] / 20)


def test_sim_shared_seed_component_quiets_paired_deltas(tmp_path):
    root = SimAdapter(make_ctx(tmp_path / "r"), duration_ticks=1).root_commit()

    def paired_sd(frac):
        sim = SimAdapter(make_ctx(tmp_path / f"f{frac}"), duration_ticks=1, shared_frac=frac)
        child, _ = edit_config(sim, root, f"c{frac}", {"loss.temperature": 0.2})
        d = [score_run(sim, child, s, "c")["alpha"] - score_run(sim, root, s, "r")["alpha"] for s in range(200)]
        return statistics.pstdev(d)
    assert paired_sd(0.8) < 0.6 * paired_sd(0.0)


def test_sim_planted_effects_recovered(ctx):
    sim = SimAdapter(ctx, duration_ticks=1)
    root = sim.root_commit()
    weaker, _ = edit_config(sim, root, "weak", {"train.lr": 5e-4, "train.weight_decay": 0.025})
    stronger, _ = edit_config(sim, root, "strong", {"train.lr": 2e-3})
    assert sim.true_panel(root) == 0.0
    assert sim.true_panel(weaker) > 0.005          # train weaker helps
    assert sim.true_panel(stronger) < 0            # train stronger hurts
    n = 300
    d = [score_run(sim, weaker, s, "w")["alpha"] - score_run(sim, root, s, "r")["alpha"] for s in range(n)]
    se = AXIS_SD["alpha"] * 2 ** 0.5 / n ** 0.5
    assert statistics.mean(d) == pytest.approx(sim.true_effect(weaker)["alpha"], abs=4 * se)
    # the effect is per axis, hidden, and scaled per axis
    eff = sim.true_effect(weaker)
    assert eff["alpha"] != eff["beta"] and all(v > 0 for v in eff.values())


def test_sim_architecture_change_is_catastrophic_unless_init_fixed(sim):
    root = sim.root_commit()
    broken, wt_b = edit_config(sim, root, "deep", {"model.depth": 24})
    fixed, wt_f = edit_config(sim, root, "deepfix", {"model.depth": 24},
                              model_py=f"# {INIT_MARKER}\ndef build(cfg):\n    return cfg\n")
    assert sim.true_panel(broken) < -0.05
    assert abs(sim.true_panel(fixed)) < 0.005
    bad = sim.init_compat(wt_b, parent_commit=root)
    assert not bad.ok and "model.depth" in bad.reason and bad.checks["init_marker"] is False
    assert sim.init_compat(wt_f, parent_commit=root).ok
    _, wt_lr = edit_config(sim, root, "lr", {"train.lr": 5e-4})
    assert sim.init_compat(wt_lr, parent_commit=root).ok


def test_sim_smoke_fails_on_planted_marker_and_real_syntax_error(sim):
    root = sim.root_commit()
    wt = sim.prepare_worktree(parent_commit=root, name="s")
    assert sim.smoke(wt).ok
    (Path(wt.path) / "train.py").write_text(f"# {SYNTAX_MARKER}\n")
    res = sim.smoke(wt)
    assert not res.ok and "train.py" in res.reason
    (Path(wt.path) / "train.py").write_text("def broken(:\n")
    assert not sim.smoke(wt).ok


def test_sim_merge_three_way_uses_common_ancestor(sim):
    root = sim.root_commit()
    x, _ = edit_config(sim, root, "x", {"train.lr": 5e-4})
    a, _ = edit_config(sim, x, "a", {"train.lr": 4e-4})             # ours moves lr again
    b, _ = edit_config(sim, x, "b", {"aug.strength": 0.3})          # theirs touches something else
    wt = sim.prepare_worktree(parent_commit=a, name="m")
    clean, text = sim.merge(wt, other_commit=b)
    assert clean and text == ""
    _, _, cfg_diff = sim.diff(wt)
    assert cfg_diff == {"train.lr": 4e-4, "aug.strength": 0.3} or cfg_diff["aug.strength"] == 0.3
    conflict_other, _ = edit_config(sim, x, "c", {"train.lr": 3e-4})
    wt2 = sim.prepare_worktree(parent_commit=a, name="m2")
    clean, text = sim.merge(wt2, other_commit=conflict_other)
    assert not clean and "train.lr" in text
    assert yaml.safe_load((Path(wt2.path) / "config.yaml").read_text())["train"]["lr"] == 4e-4


def test_sim_crossover_of_root_children_diffs_both_against_root(sim):
    root = sim.root_commit()
    a, _ = edit_config(sim, root, "a", {"train.lr": 4e-4})
    b, _ = edit_config(sim, root, "b", {"aug.strength": 0.3})
    wt = sim.prepare_worktree(parent_commit=a, name="x")            # as the researcher does
    wt = dataclasses.replace(wt, base_commit=root)                   # judged against the common parent
    clean, _ = sim.merge(wt, other_commit=b)
    _, _, cfg_diff = sim.diff(wt)
    assert clean and cfg_diff == {"train.lr": 4e-4, "aug.strength": 0.3}


def test_sim_polling_advances_one_step_per_poll(ctx):
    sim = SimAdapter(ctx, duration_ticks=3, clock=lambda: 1000.0)
    h = sim.submit(run=run_of("n", 1), commit=sim.root_commit())
    assert [sim.poll(h).state for _ in range(2)] == [RunStatus.RUNNING] * 2
    done = sim.poll(h)
    assert done.state is RunStatus.DONE and done.gpu_h == pytest.approx(0.15)
    assert sim.poll(h) == done                                       # terminal state is stable
    assert done.aggregate == pytest.approx(sum(WEIGHTS[a] * v for a, v in done.scores.items()))


def test_sim_fault_injection(ctx):
    sim = SimAdapter(ctx, duration_ticks=2)
    root = sim.root_commit()
    sim.inject_fault("n:screen:1", "crash")
    sim.inject_fault("n:screen:2", "infra", times=1)
    sim.inject_fault("n:screen:3", "eval", at_poll=2)
    sim.inject_fault("n:screen:4", "hang")
    crash = finish(sim, sim.submit(run=run_of("n", 1), commit=root))
    assert crash.state is RunStatus.CRASH and "injected" in crash.log_tail and not crash.scores
    infra = finish(sim, sim.submit(run=run_of("n", 2), commit=root))
    assert (infra.state, infra.void_type) == (RunStatus.VOID, VoidType.INFRA)
    retry = finish(sim, sim.submit(run=run_of("n", 2, attempt=1), commit=root))   # same run, next attempt
    assert retry.state is RunStatus.DONE                                          # infra fault fired once
    ev = finish(sim, sim.submit(run=run_of("n", 3), commit=root))
    assert (ev.state, ev.void_type) == (RunStatus.VOID, VoidType.EVAL)
    hang = sim.submit(run=run_of("n", 4), commit=root)
    assert all(sim.poll(hang).state is RunStatus.RUNNING for _ in range(6))
    sim.cancel(hang)
    assert sim.poll(hang).void_type is VoidType.INFRA
    with pytest.raises(ValueError):
        sim.inject_fault("*", "bogus")


def test_sim_resume_via_find(ctx, tmp_path):
    a = SimAdapter(ctx, duration_ticks=3)
    run = run_of("n", 7)
    assert a.find(run) is None
    handle = a.submit(run=run, commit=a.root_commit())
    a.poll(handle)
    b = SimAdapter(ctx, duration_ticks=3)                      # "driver restarted"
    found = b.find(run)
    assert found == handle
    assert b.submit(run=run, commit=b.root_commit()) == handle  # idempotent per run.id
    assert b.poll(found).state is RunStatus.RUNNING             # progress survived the restart
    assert b.poll(found).state is RunStatus.DONE
    assert b.find(run_of("n", 7, attempt=1)) is None


def test_sim_unknown_commit_rejected(sim):
    with pytest.raises(KeyError):
        sim.submit(run=run_of("n", 1), commit="deadbeef")


# ============================================================ nanopath_v2

MODEL_PY = '''import torch


class ViT:
    def forward(self, x):
        # eval-time D4 average
        return x * 2

    def encode_image(self, x):
        return x

    def build_blocks(self):
        return 1
'''
TRAIN_PY = '''def main():
    budget = 10
    robust_norm_tiles = 6144
    calibration_tiles = robust_norm_tiles + 1
    train_sample_budget = budget - calibration_tiles
    lr = 0.1
    # Fit scanner-response directions after optimization.
    directions = fit(lr)
    model.ct_ms.copy_(directions)
    return lr
'''
LOCKS = ("probe.py", "model.py::ViT.forward", "model.py::ViT.encode_image",
         "train.py::robust_norm_tiles = 6144...train_sample_budget = budget - calibration_tiles",
         "train.py::# Fit scanner-response directions...model.ct_ms.copy_")


class SlurmRunner:
    """Real git (+ the launcher is a no-op); scripted squeue/sacct/sbatch/scancel/python."""

    def __init__(self, replies=None):
        self.real, self.calls, self.replies = SubprocessRunner(), [], dict(replies or {})

    def run(self, argv, *, cwd=None, timeout=None, env=None):
        argv = list(argv)
        self.calls.append({"argv": argv, "cwd": cwd, "env": env})
        if argv[0] == "git":
            return self.real.run(argv, cwd=cwd, timeout=timeout, env=env)
        joined = " ".join(argv)
        for key, reply in self.replies.items():
            if key in joined:
                return reply(argv) if callable(reply) else reply
        return 0, "", ""

    def argvs(self, head=None):
        return [c["argv"] for c in self.calls if head is None or c["argv"][0] == head]


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                          capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def base_repo(tmp_path):
    repo = tmp_path / "base"
    (repo / "configs").mkdir(parents=True)
    (repo / "configs" / "main.yaml").write_text(yaml.safe_dump(
        {"train": {"seed": 7974, "lr": 1e-4}, "project": {"output_dir": "/x"}, "model": {"type": "vit"}}))
    (repo / "model.py").write_text(MODEL_PY)
    (repo / "train.py").write_text(TRAIN_PY)
    (repo / ".gitignore").write_text(".venv/\nslurm/\n")
    (repo / ".venv" / "bin").mkdir(parents=True)
    git(repo, "init", "-q", "-b", "main")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "base")
    return repo


def np_ctx_with_locks(state: Path, locks=LOCKS, **cfg_kw) -> LoopContext:
    from loop.spec import SlurmCfg
    cfg_kw.setdefault("slurm", SlurmCfg(smoke_probe="readout"))   # nanopath's probe leg (generic default: off, 11-10)
    ctx = make_ctx(state, **cfg_kw)
    spec = ctx.spec.model_copy(update={"surface": ctx.spec.surface.model_copy(update={"locked": locks})})
    return dataclasses.replace(ctx, spec=spec)


@pytest.fixture
def np_ctx(tmp_path):
    return np_ctx_with_locks(tmp_path / "state")


def make_np(np_ctx, base_repo, runner=None, **kw):
    runner = runner or SlurmRunner()
    kw.setdefault("clock", lambda: 5000.0)
    root = Path(np_ctx.state_dir).parent
    adapter = NanopathV2Adapter(np_ctx, runner=runner, repo=str(base_repo),
                                runs_root=str(root / "runs"), worktrees_root=str(root / "wts"), **kw)
    adapter.lock_ref = git(base_repo, "rev-parse", "HEAD")       # the launch-time lock guard's canonical copy
    return adapter, runner


def sbatch_ok(job="777"):
    return {LAUNCHER: (0, f"No interactive stdin; training will run without Labless auto-submit.\n"
                          f"Submitted batch job {job}\n", "")}


def test_nanopath_protocol_and_root_commit_pins_a_full_sha_without_git(np_ctx, base_repo):
    adapter, runner = make_np(np_ctx, base_repo)
    assert isinstance(adapter, ExecutionAdapter)
    assert adapter.root_commit() == BASE_SHA and len(BASE_SHA) == 40 and not runner.calls


def research_wt(adapter, base_repo, name="req1"):
    return adapter.prepare_worktree(parent_commit=git(base_repo, "rev-parse", "HEAD"), name=name)


def test_nanopath_research_worktree_add_is_idempotent(np_ctx, base_repo):
    """A restart re-issues a request under the same name: its stale worktree (registered, or a bare dir a
    killed add left behind) and branch must not fail the add ("already exists" / "resetting branch")."""
    adapter, _ = make_np(np_ctx, base_repo)
    wt = research_wt(adapter, base_repo, "req-7")
    (Path(wt.path) / "junk.txt").write_text("x")
    again = research_wt(adapter, base_repo, "req-7")                    # still registered + dirty
    assert again.path == wt.path and not (Path(again.path) / "junk.txt").exists()
    git(base_repo, "worktree", "remove", "--force", wt.path)
    Path(wt.path).mkdir(parents=True)
    (Path(wt.path) / "partial").write_text("x")                        # a bare dir from a killed add
    assert (Path(research_wt(adapter, base_repo, "req-7").path) / "configs/main.yaml").exists()


def test_nanopath_cancel_smokes_and_a_dest_a_live_job_recreates(np_ctx, base_repo, monkeypatch):
    """A restart orphans the old process's smoke job (lp<tag>-smoke-loop-<id>): it is scancelled by name
    (never another request's or another loop's), and a dest that a dying job keeps recreating fails loudly."""
    adapter, runner = make_np(np_ctx, base_repo)
    tag = adapter._tag
    runner.replies["squeue -h -u"] = (0, f"11|lp{tag}-smoke-loop-req-7\n12|lp{tag}-smoke-loop-req-7-expandable_segments\n"
                                         f"13|lp{tag}-smoke-loop-req-70\n14|lpzzzzzz-smoke-loop-req-7\n", "")
    assert adapter.cancel_smokes("loop-req-7") == ["11", "12"]
    assert ["scancel", "11", "12"] in runner.argvs("scancel")
    assert adapter.cancel_smokes() == ["11", "12", "13"]
    adapter._smoke_names.add(f"lp{tag}-smoke-loop-req-70")      # a live smoke() call waits on it: no orphan
    assert adapter.cancel_smokes() == ["11", "12"]
    import loop.adapters.nanopath_v2 as NP
    monkeypatch.setattr(NP, "REAP_WAIT_S", 0.0)
    real = adapter.reap_worktree
    def reap_then_recreate(wt):
        real(wt)
        (Path(wt.path) / "slurm").mkdir(parents=True, exist_ok=True)      # the live smoke job writes again
    adapter.reap_worktree = reap_then_recreate
    with pytest.raises(RuntimeError, match="still exists"):
        research_wt(adapter, base_repo, "req-8")


def test_a_researchers_own_add_never_stages_the_venv_symlink(np_ctx, base_repo):
    """10-13: a researcher ran `git add -A` in its worktree, staging the .venv symlink -> surface_violation. The
    repo's info/exclude lists the env links (every worktree reads it) and diff / commit unstage them first."""
    adapter, _ = make_np(np_ctx, base_repo)
    wt = research_wt(adapter, base_repo, "req-venv")
    p = Path(wt.path)
    excl = Path(git(p, "rev-parse", "--path-format=absolute", "--git-path", "info/exclude"))
    assert ".venv" in excl.read_text().splitlines()
    research_wt(adapter, base_repo, "req-venv2")
    assert excl.read_text().splitlines().count(".venv") == 1                    # idempotent
    git(p, "add", "-A")                                                          # the researcher's own add
    assert ".venv" not in git(p, "diff", "--cached", "--name-only").splitlines()
    git(p, "add", "-f", ".venv")                                                 # even force-staged
    (p / "train.py").write_text(TRAIN_PY.replace("lr = 0.1", "lr = 0.3"))
    _, paths, _ = adapter.diff(wt)
    assert set(paths) == {"train.py"}
    sha = adapter.commit(wt, message="m")
    assert ".venv" not in git(base_repo, "show", "--name-only", "--format=", sha).splitlines()


def test_nanopath_worktree_commit_diff_real_git(np_ctx, base_repo):
    adapter, _ = make_np(np_ctx, base_repo)
    wt = research_wt(adapter, base_repo)
    p = Path(wt.path)
    assert (p / ".venv").resolve() == (base_repo / ".venv").resolve()   # env shared, not rebuilt
    cfg = yaml.safe_load((p / "configs/main.yaml").read_text())
    cfg["train"]["lr"] = 5e-5
    (p / "configs/main.yaml").write_text(yaml.safe_dump(cfg))
    (p / "dataloader.py").write_text("x = 1\n")                        # untracked file counts
    (p / "train.py").write_text(TRAIN_PY.replace("lr = 0.1", "lr = 0.2"))
    text, paths, cfg_diff = adapter.diff(wt)
    assert set(paths) == {"configs/main.yaml", "dataloader.py", "train.py"}   # never .venv
    assert cfg_diff == {"train.lr": 5e-5} and "lr = 0.2" in text
    sha = adapter.commit(wt, message="m")
    assert git(base_repo, "rev-parse", f"loop/{adapter._tag}-{Path(wt.path).name}") == sha
    assert ".venv" not in git(base_repo, "show", "--name-only", "--format=", sha).split()
    adapter.reap_worktree(wt)
    assert not p.exists()
    assert git(base_repo, "cat-file", "-t", sha) == "commit"                 # branch keeps it alive


@pytest.mark.parametrize("edit,lock", [
    (lambda m, t: (m.replace("return x * 2", "return x * 3"), t), "model.py::ViT.forward"),
    (lambda m, t: (m.replace("return x\n", "return -x\n", 1), t), "model.py::ViT.encode_image"),
    (lambda m, t: (m.replace("    def forward", "    def forwardx"), t), "model.py::ViT.forward"),
    (lambda m, t: (m, t.replace("6144", "1")), LOCKS[3]),
    (lambda m, t: (m, t.replace("fit(lr)", "fit(lr * 2)")), LOCKS[4]),
])
def test_nanopath_diff_reports_a_changed_locked_function_or_span(np_ctx, base_repo, edit, lock):
    adapter, _ = make_np(np_ctx, base_repo)
    wt = research_wt(adapter, base_repo)
    p = Path(wt.path)
    m, t = edit(MODEL_PY, TRAIN_PY)
    (p / "model.py").write_text(m)
    (p / "train.py").write_text(t)
    _, paths, _ = adapter.diff(wt)
    assert lock in paths
    assert any(path_matches(lock, p) for p in np_ctx.spec.surface.locked)  # surface rejects it


def test_nanopath_diff_allows_edits_outside_locks(np_ctx, base_repo):
    adapter, _ = make_np(np_ctx, base_repo)
    wt = research_wt(adapter, base_repo)
    p = Path(wt.path)
    (p / "model.py").write_text(MODEL_PY.replace("# eval-time D4 average", "# reworded")
                                .replace("return 1", "return 2"))
    (p / "train.py").write_text(TRAIN_PY.replace("lr = 0.1", "lr = 0.3"))
    _, paths, _ = adapter.diff(wt)
    assert set(paths) == {"model.py", "train.py"}


def test_nanopath_diff_family_span_edit_reports_the_span_not_the_file(tmp_path, base_repo):
    fams = {"eval_readout": FamilyRule(paths=("model.py::ViT.forward",), combine=True),
            "architecture": FamilyRule(paths=("model.py",))}
    adapter, _ = make_np(np_ctx_with_locks(tmp_path / "s", ("probe.py",), families=fams), base_repo)
    wt = research_wt(adapter, base_repo)
    p = Path(wt.path)
    (p / "model.py").write_text(MODEL_PY.replace("return x * 2", "return x * 3"))
    assert adapter.diff(wt)[1] == ("model.py::ViT.forward",)             # readout only: not architecture
    (p / "model.py").write_text(MODEL_PY.replace("return x * 2", "return x * 3").replace("return 1", "return 2"))
    assert set(adapter.diff(wt)[1]) == {"model.py", "model.py::ViT.forward"}   # plus a trained part


def test_locked_source_nested_def_and_missing():
    src = "def main():\n    if x:\n        def inner(a):\n            return a\n"
    assert locked_source(src, "t.py::main.inner") and locked_source(src, "t.py::main.nope") is None
    assert locked_source("def (", "t.py::main") is None and locked_source(None, "t.py::a...b") is None


def test_nanopath_merge_aborts_on_conflict(np_ctx, base_repo):
    adapter, runner = make_np(np_ctx, base_repo)
    wt = Worktree("/w", "b", "p")
    runner.real = SimpleNamespace(run=lambda argv, **kw: (1, "CONFLICT (content)", "") if "merge" in argv
                                  and "--abort" not in argv else (0, "", ""))
    clean, text = adapter.merge(wt, other_commit="abc")
    assert not clean and "CONFLICT" in text and any("--abort" in a for a in runner.argvs("git"))


def submit(adapter, run=None, sha=None, base_repo=None, **kw):
    run = run or run_of("n1", 11)
    return run, adapter.submit(run=run, commit=sha or git(base_repo, "rev-parse", "HEAD"), **kw)


def test_nanopath_submit_launches_sbatch_in_a_node_worktree(np_ctx, base_repo):
    adapter, runner = make_np(np_ctx, base_repo, runner=SlurmRunner(sbatch_ok()))
    run, handle = submit(adapter, base_repo=base_repo)
    call = next(c for c in runner.calls if c["argv"][0] == LAUNCHER)
    wt = Path(call["cwd"])
    assert wt == adapter.worktrees / "n1" and git(wt, "rev-parse", "HEAD") == git(base_repo, "rev-parse", "HEAD")
    assert (wt / ".venv").is_symlink()
    out = adapter.runs / "n1" / "screen_s11_a0"
    assert call["argv"] == [LAUNCHER, "configs/main.yaml", f"output_dir={out}", "seed=11"]
    assert call["env"]["WANDB_MODE"] == "offline" and call["env"]["SBATCH_JOB_NAME"].endswith("n1-screen_s11_a0")
    assert call["env"]["LABLESS_AUTOSUBMIT_FILE"] == ""                   # never arms Labless autosubmit
    assert handle.job_id == "777" and handle.output_dir == str(out)
    rec = json.loads((Path(np_ctx.state_dir) / "receipts/n1/screen_s11_a0.json").read_text())
    assert rec["job_id"] == "777" and rec["run_id"] == run.id
    assert not any("trial.py" in " ".join(a) or "commit_full" in " ".join(a) for a in runner.argvs())


def test_nanopath_submit_is_idempotent_and_find_resumes(np_ctx, base_repo):
    adapter, runner = make_np(np_ctx, base_repo, runner=SlurmRunner(sbatch_ok()))
    run, h1 = submit(adapter, base_repo=base_repo)
    _, h2 = submit(adapter, run=run, base_repo=base_repo)
    assert h1 == h2 and len(runner.argvs(LAUNCHER)) == 1
    assert adapter.find(run) == h1 and adapter.find(run_of("n1", 12)) is None
    _, h3 = submit(adapter, run=run_of("n1", 11, attempt=1), base_repo=base_repo)   # retry = new dir
    assert h3.output_dir.endswith("screen_s11_a1") and len(runner.argvs(LAUNCHER)) == 2


def test_nanopath_lost_job_id_is_recovered_by_name_never_resubmitted(np_ctx, base_repo):
    """Crash after sbatch but before the job id was recorded (the old pid-null window)."""
    adapter, runner = make_np(np_ctx, base_repo, runner=SlurmRunner({LAUNCHER: (0, "garbled", "")}))
    run = run_of("n1", 11)
    with pytest.raises(RuntimeError, match="launch failed"):
        submit(adapter, run=run, base_repo=base_repo)                   # nothing on the queue
    assert adapter.find(run) is None                                    # -> driver may resubmit
    runner.replies = {"squeue -h -n": (0, "4242\n", "")}                 # ...but now it IS queued
    assert adapter.find(run).job_id == "4242"
    _, h = submit(adapter, run=run, base_repo=base_repo)
    assert h.job_id == "4242" and len(runner.argvs(LAUNCHER)) == 1


def test_nanopath_node_worktree_at_wrong_commit_refuses(np_ctx, base_repo):
    adapter, _ = make_np(np_ctx, base_repo, runner=SlurmRunner(sbatch_ok()))
    submit(adapter, base_repo=base_repo)
    with pytest.raises(RuntimeError, match="not deadbeef|LOCKED-FILE REFUSAL"):  # c4j: the lock guard fails closed first
        submit(adapter, run=run_of("n1", 12), sha="deadbeef", base_repo=base_repo)


def _reroot(adapter, base_repo):
    """A first submit of n1 at the base commit, then a new (lock-neutral) commit: n1's worktree stands at the old one
    (live 11-09: worktrees/<tag>/root at the r12 root 45b541e9 after the re-root to fb 9035f616)."""
    old = git(base_repo, "rev-parse", "HEAD")
    submit(adapter, base_repo=base_repo)
    (base_repo / "NOTES.md").write_text("re-rooted\n")
    git(base_repo, "add", "NOTES.md")
    git(base_repo, "commit", "-qm", "new root")
    return old, git(base_repo, "rev-parse", "HEAD"), adapter.worktrees / "n1"


def test_nanopath_node_worktree_at_a_stale_commit_is_re_checked_out_when_clean_and_unused(np_ctx, base_repo):
    adapter, runner = make_np(np_ctx, base_repo, runner=SlurmRunner(sbatch_ok()))
    old, new, wt = _reroot(adapter, base_repo)
    assert git(wt, "rev-parse", "HEAD") == old
    _, handle = submit(adapter, run=run_of("n1", 12), sha=new, base_repo=base_repo)
    assert git(wt, "rev-parse", "HEAD") == new and handle.job_id == "777"
    assert any(a[:2] == ["squeue", "-u"] for a in runner.argvs("squeue"))     # checked no live job runs in it


def test_nanopath_stale_node_worktree_with_edits_or_a_live_job_is_rejected_once_never_retried(np_ctx, base_repo,
                                                                                              capsys):
    adapter, runner = make_np(np_ctx, base_repo, runner=SlurmRunner(sbatch_ok()))
    old, new, wt = _reroot(adapter, base_repo)
    (wt / "train.py").write_text(TRAIN_PY + "# hand edit\n")
    for seed in (12, 13):
        with pytest.raises(SubmissionRejected, match="modified tracked"):
            submit(adapter, run=run_of("n1", seed), sha=new, base_repo=base_repo)
    assert git(wt, "rev-parse", "HEAD") == old and capsys.readouterr().err.count("ALERT") == 1   # alerted once
    git(wt, "checkout", "-q", "--", "train.py")
    runner.replies["squeue -u"] = (0, f"901 {wt}\n902 /elsewhere\n", "")
    with pytest.raises(SubmissionRejected, match="live job.s. 901"):
        submit(adapter, run=run_of("n1", 14), sha=new, base_repo=base_repo)
    assert git(wt, "rev-parse", "HEAD") == old
    runner.replies["squeue -u"] = (1, "", "slurmctld down")                  # unknown: a plain (retried) error
    with pytest.raises(RuntimeError, match="squeue failed") as e:
        submit(adapter, run=run_of("n1", 15), sha=new, base_repo=base_repo)
    assert not isinstance(e.value, SubmissionRejected)


def _half_created(wt):
    """What a `git worktree add` killed mid-way leaves (live 11-11 n0168): ``.git`` only, an EMPTY gitdir HEAD and
    a ``locked`` "initializing" registration."""
    gitdir = Path((wt / ".git").read_text().split("gitdir:", 1)[1].strip())
    for p in wt.iterdir():
        if p.name != ".git":
            shutil.rmtree(p) if p.is_dir() and not p.is_symlink() else p.unlink()
    (gitdir / "HEAD").write_text("")
    (gitdir / "locked").write_text("initializing")
    return gitdir


def test_nanopath_half_created_node_worktree_is_rebuilt_not_voided(np_ctx, base_repo, capsys):
    adapter, runner = make_np(np_ctx, base_repo, runner=SlurmRunner(sbatch_ok()))
    submit(adapter, base_repo=base_repo)
    wt, sha = adapter.worktrees / "n1", git(base_repo, "rev-parse", "HEAD")
    _half_created(wt)
    assert subprocess.run(["git", "-C", str(wt), "rev-parse", "HEAD"], capture_output=True).returncode != 0
    _, handle = submit(adapter, run=run_of("n1", 12), base_repo=base_repo)      # was: "is at ?, not <sha>"
    assert git(wt, "rev-parse", "HEAD") == sha and handle.job_id == "777" and (wt / "train.py").is_file()
    assert "rebuilt" in capsys.readouterr().err


def test_nanopath_node_worktree_cut_off_mid_checkout_is_rebuilt(np_ctx, base_repo, capsys):
    """Live 11-10 n0128: HEAD written, `locked` still 'initializing', checkout cut off (index.lock, no index, files
    missing): rev-parse matched the commit, so the jobs ran without the package. The lock file alone is the sign."""
    adapter, runner = make_np(np_ctx, base_repo, runner=SlurmRunner(sbatch_ok()))
    submit(adapter, base_repo=base_repo)
    wt, sha = adapter.worktrees / "n1", git(base_repo, "rev-parse", "HEAD")
    gitdir = Path((wt / ".git").read_text().split("gitdir:", 1)[1].strip())
    (gitdir / "locked").write_text("initializing")
    (gitdir / "index").unlink()
    (gitdir / "index.lock").write_text("")
    (wt / "train.py").unlink()
    assert git(wt, "rev-parse", "HEAD") == sha                                 # HEAD alone looks fine
    submit(adapter, run=run_of("n1", 12), base_repo=base_repo)
    assert (wt / "train.py").is_file() and not (gitdir / "locked").exists() and "rebuilt" in capsys.readouterr().err


def test_nanopath_other_rev_parse_failures_never_reap_the_worktree(np_ctx, base_repo, monkeypatch):
    """Only the half-created signature rebuilds: a garbled HEAD or a rev-parse timeout (rc 124) is a plain, retried
    RuntimeError and the tree (jobs' slurm/%j.out live in it) is left alone."""
    adapter, runner = make_np(np_ctx, base_repo, runner=SlurmRunner(sbatch_ok()))
    submit(adapter, base_repo=base_repo)
    wt = adapter.worktrees / "n1"
    (wt / "slurm").mkdir(exist_ok=True)
    (wt / "slurm" / "901.out").write_text("live job log\n")
    real = adapter._git
    monkeypatch.setattr(adapter, "_git", lambda path, *a, **kw: (124, "", "timeout") if a[:1] == ("rev-parse",)
                        and a[1:] == ("HEAD",) and str(path) == str(wt) else real(path, *a, **kw))
    with pytest.raises(RuntimeError, match=r"is at \?") as e:
        submit(adapter, run=run_of("n1", 12), base_repo=base_repo)
    assert not isinstance(e.value, SubmissionRejected) and (wt / "slurm" / "901.out").is_file()
    monkeypatch.setattr(adapter, "_git", real)
    gitdir = Path((wt / ".git").read_text().split("gitdir:", 1)[1].strip())
    (gitdir / "HEAD").write_text("not-a-ref\n")
    with pytest.raises(RuntimeError, match=r"is at \?"):
        submit(adapter, run=run_of("n1", 13), base_repo=base_repo)
    assert (wt / "slurm" / "901.out").is_file() and (wt / "train.py").is_file()


def test_nanopath_half_created_node_worktree_with_a_live_job_is_a_retried_error(np_ctx, base_repo):
    adapter, runner = make_np(np_ctx, base_repo, runner=SlurmRunner(sbatch_ok()))
    submit(adapter, base_repo=base_repo)
    wt = adapter.worktrees / "n1"
    _half_created(wt)
    runner.replies["squeue -u"] = (0, f"901 {wt}\n", "")
    with pytest.raises(RuntimeError, match="not rebuilt: live job.s. 901") as e:
        submit(adapter, run=run_of("n1", 12), base_repo=base_repo)
    assert not isinstance(e.value, SubmissionRejected) and (wt / ".git").exists()


@pytest.fixture
def launched(np_ctx, base_repo):
    runner = SlurmRunner(sbatch_ok())
    clock = {"t": time.mktime(time.strptime("2026-09-28T12:00:00", "%Y-%m-%dT%H:%M:%S"))}
    adapter, _ = make_np(np_ctx, base_repo, runner=runner, clock=lambda: clock["t"])
    run, handle = submit(adapter, base_repo=base_repo)
    Path(handle.output_dir).mkdir(parents=True)
    return SimpleNamespace(adapter=adapter, runner=runner, handle=handle, clock=clock)


SUMMARY = {"alpha": 0.61, "beta": 0.52, "gamma": 0.43, "panel_score": 0.55}


def finish_job(l, state="COMPLETED", elapsed="7200", summary=SUMMARY, err=""):
    l.runner.replies = {"squeue": (1, "", "slurm_load_jobs error: Invalid job id specified"),
                        "sacct": (0, f"{state}|{elapsed}\n", "")}
    if summary is not None:
        (Path(l.handle.output_dir) / "summary.json").write_text(json.dumps(summary))
    if err:
        (Path(l.adapter.worktrees) / "n1" / "slurm").mkdir(exist_ok=True)
        (Path(l.adapter.worktrees) / "n1" / "slurm" / "777.err").write_text(err)
    return l.adapter.poll(l.handle)


def test_nanopath_poll_done_reads_axes_and_gpu_hours(launched):
    st = finish_job(launched)
    assert st.state is RunStatus.DONE and dict(st.scores) == {"alpha": 0.61, "beta": 0.52, "gamma": 0.43}
    assert st.aggregate == 0.55 and st.gpu_h == 2.0


def test_nanopath_stale_summary_from_before_submit_is_never_read(launched):
    out = Path(launched.handle.output_dir) / "summary.json"
    out.write_text(json.dumps(SUMMARY))
    old = launched.handle.submitted_ts - 60
    os.utime(out, (old, old))
    launched.runner.replies = {"squeue": (1, "", "Invalid job id specified"), "sacct": (0, "COMPLETED|60\n", "")}
    st = launched.adapter.poll(launched.handle)
    assert st.state is RunStatus.VOID and st.void_type is VoidType.EVAL


def test_nanopath_paths_and_branches_are_namespaced_by_state_dir(tmp_path, base_repo):
    a, _ = make_np(np_ctx_with_locks(tmp_path / "s1"), base_repo)
    b, _ = make_np(np_ctx_with_locks(tmp_path / "s2"), base_repo)
    assert a._tag != b._tag and a.runs != b.runs and a.worktrees != b.worktrees
    assert a.runs.name == f"lp{a._tag}" and a.worktrees.name == f"lp{a._tag}"
    wa, wb = research_wt(a, base_repo), research_wt(b, base_repo)
    assert wa.path != wb.path and wa.branch != wb.branch and a._tag in wa.branch
    assert str(a.worktrees) in a.command(run_of("n1", 11)) and str(a.runs) in a.command(run_of("n1", 11))


@pytest.mark.parametrize("state,summary,err,expect,vtype", [
    ("COMPLETED", {"alpha": 1.0}, "", RunStatus.VOID, VoidType.EVAL),
    ("TIMEOUT", None, "", RunStatus.VOID, VoidType.TOO_SLOW),         # killed at the time limit
    ("FAILED", None, "Traceback\nValueError: shapes", RunStatus.CRASH, None),
    ("FAILED", None, "Traceback\nRuntimeError: CUDA error: uncorrectable ECC error encountered",
     RunStatus.VOID, VoidType.INFRA),
    ("NODE_FAIL", None, "", RunStatus.VOID, VoidType.INFRA),
    ("CANCELLED", None, "", RunStatus.VOID, VoidType.INFRA),
    ("PREEMPTED", None, "", RunStatus.VOID, VoidType.INFRA),
    ("", None, "", RunStatus.VOID, VoidType.INFRA),
    ("RUNNING", None, "", RunStatus.RUNNING, None),                  # sacct lagging squeue
])
def test_nanopath_poll_types_finished_jobs(launched, state, summary, err, expect, vtype):
    st = finish_job(launched, state=state, summary=summary, err=err)
    assert st.state is expect and st.void_type is vtype


def queue(l, state="RUNNING", start="2026-09-28T12:00:00"):
    l.runner.replies = {"squeue -h -j": (0, f"{state}|{start}\n", "")}
    return l.adapter.poll(l.handle)


def test_nanopath_poll_running_and_pending(launched):
    launched.clock["t"] += 3600
    assert queue(launched, "PENDING", "N/A").state is RunStatus.RUNNING   # queue wait is never a freeze
    launched.clock["t"] -= 3600 - 600
    assert queue(launched).state is RunStatus.RUNNING                    # 10 min after start


def test_nanopath_freeze_is_keyed_to_start_and_heartbeat_then_cancelled(launched):
    t0 = launched.clock["t"]
    launched.clock["t"] = t0 + 3600                                      # submitted an hour ago...
    st = queue(launched, start="2026-09-28T12:50:00")                    # ...started 10 min ago
    assert st.state is RunStatus.RUNNING
    beat = Path(launched.handle.output_dir) / "metrics.jsonl"
    beat.write_text("{}\n")
    os.utime(beat, (t0 + 3000, t0 + 3000))
    launched.clock["t"] = t0 + 3000 + 1100                               # fresh beat: still running
    assert queue(launched, start="2026-09-28T12:00:00").state is RunStatus.RUNNING
    launched.clock["t"] = t0 + 3000 + 1300                               # >20 min without progress
    st = queue(launched, start="2026-09-28T12:00:00")
    assert st.state is RunStatus.VOID and st.void_type is VoidType.INFRA  # retryable, never TIMEOUT
    assert "frozen" in st.detail and ["scancel", "777"] in launched.runner.argvs()


def test_nanopath_freeze_window_is_configurable(np_ctx, base_repo, tmp_path):
    ctx = np_ctx_with_locks(tmp_path / "f", heartbeat=HeartbeatCfg(freeze_min=1))
    assert make_np(ctx, base_repo)[0].freeze_s == 60
    assert make_np(np_ctx, base_repo)[0].freeze_s == 1200


def test_nanopath_squeue_failure_is_not_a_verdict(launched):
    launched.runner.replies = {"squeue": (1, "", "slurm_load_jobs error: Socket timed out")}
    with pytest.raises(RuntimeError, match="squeue failed"):
        launched.adapter.poll(launched.handle)


POOLS = SlurmCfg(pools=(PoolCfg("primary", account="main", qos="high", partition="gpu", slots=1),
                        PoolCfg("idle", account="preempt", qos="low", partition="gpu", slots=2, preemptible=True)))


def test_nanopath_pool_env_and_requeue_only_for_the_preemptible_pool(tmp_path, base_repo):
    adapter, runner = make_np(np_ctx_with_locks(tmp_path / "p", slurm=POOLS), base_repo,
                              runner=SlurmRunner(sbatch_ok()))
    for seed, pool in ((11, "primary"), (12, "idle")):
        submit(adapter, run=dataclasses.replace(run_of("n1", seed), pool=pool), base_repo=base_repo)
    env = [c["env"] for c in runner.calls if c["argv"][0] == LAUNCHER]
    assert (env[0]["SBATCH_ACCOUNT"], env[0]["SBATCH_QOS"], "SBATCH_REQUEUE" in env[0]) == ("main", "high", False)
    assert (env[1]["SBATCH_ACCOUNT"], env[1]["SBATCH_QOS"], env[1]["SBATCH_REQUEUE"]) == ("preempt", "low", "")


def test_nanopath_preempted_job_is_resubmitted_as_a_free_retry(launched):
    """PREEMPTED / NODE_FAIL: scancel (never two copies), relaunch the same run, follow the new job id;
    the driver sees RUNNING throughout (no infra void, no retry counted)."""
    l = launched
    l.runner.replies = {"squeue": (1, "", "Invalid job id specified"), "sacct -n -X -P -j 777": (0, "PREEMPTED|60\n", ""),
                        **sbatch_ok("778")}
    st = l.adapter.poll(l.handle)
    assert st.state is RunStatus.RUNNING and "free retry 1 as job 778" in st.detail
    assert ["scancel", "777"] in l.runner.argvs() and len(l.runner.argvs(LAUNCHER)) == 2
    l.runner.replies = {"squeue -h -j 778": (0, "PENDING|N/A|Priority\n", "")}      # polled by the NEW id
    assert l.adapter.poll(l.handle).state is RunStatus.RUNNING
    l.runner.replies = {"squeue -h -j 778": (0, "PENDING|N/A|JobHeldAdmin\n", ""), **sbatch_ok("780")}
    assert "free retry 2 as job 780" in l.adapter.poll(l.handle).detail   # held after MaxBatchRequeue
    l.adapter.cancel(l.handle)
    assert l.runner.argvs("scancel")[-1] == ["scancel", "780"]
    l.runner.replies = {"squeue": (1, "", "Invalid job id specified"), "sacct": (0, "NODE_FAIL|60\n", ""),
                        **sbatch_ok("779")}
    for _ in range(8):
        assert l.adapter.poll(l.handle).state is RunStatus.RUNNING
    st = l.adapter.poll(l.handle)                                        # 10 free retries spent
    assert st.state is RunStatus.VOID and st.void_type is VoidType.INFRA


def test_nanopath_slurm_requeue_under_the_same_id_is_reported_as_a_preemption(launched):
    """SLURM requeues a preempted job under the SAME id (it only looks PENDING again): RestartCnt rising is
    reported once per new requeue, with the time the instances ran before it (sacct -D); a nice sticks. The
    first reading is the baseline (a receipt from before the telemetry never backfills its old requeues)."""
    l = launched
    l.runner.replies = {"squeue -h -j 777": (0, "RUNNING             |N/A   |None       |0     \n", "")}
    assert l.adapter.poll(l.handle).requeued == 0
    l.runner.replies = {"squeue -h -j 777": (0, "PENDING             |N/A   |Priority   |2     \n", ""),
                        "sacct -n -D": (0, "PREEMPTED|600\nPREEMPTED|120\nPENDING|0\n", "")}
    st = l.adapter.poll(l.handle)
    assert (st.state, st.detail, st.restarts, st.requeued) == (RunStatus.RUNNING, "PENDING", 2, 2)
    assert st.gpu_h == pytest.approx(720 / 3600)
    st = l.adapter.poll(l.handle)                                         # nothing new: counted once
    assert (st.restarts, st.requeued, st.gpu_h) == (2, 0, 0.0)
    rel = l.adapter._rel_of(l.handle)
    rec = l.adapter._receipt(rel)
    rec.pop("restarts")
    rec.pop("lost_s")
    l.adapter._save(rel, rec)                                             # a pre-telemetry receipt, 5 old requeues
    l.runner.replies["squeue -h -j 777"] = (0, "PENDING             |N/A   |Priority   |5     \n", "")
    assert l.adapter.poll(l.handle).requeued == 0 and l.adapter._receipt(rel)["restarts"] == 5
    l.adapter.nice(l.handle, 100)
    assert l.runner.argvs("scontrol")[-1] == ["scontrol", "update", "JobId=777", "Nice=100"]
    l.runner.replies = {"squeue": (1, "", "Invalid job id specified"), "sacct -n -X -P -j 777": (0, "PREEMPTED|60\n", ""),
                        **sbatch_ok("778")}
    assert "free retry 1 as job 778" in l.adapter.poll(l.handle).detail
    assert l.runner.argvs("scontrol")[-1] == ["scontrol", "update", "JobId=778", "Nice=100"]   # kept


def test_nanopath_repool_moves_a_pending_job_to_the_primary_pool(tmp_path, base_repo):
    """scancel the pending preemptible job, relaunch the same run (same receipt / attempt / output
    dir) on the primary association without --requeue; poll and cancel follow the new job id."""
    runner = SlurmRunner(sbatch_ok("777"))
    adapter, _ = make_np(np_ctx_with_locks(tmp_path / "rp", slurm=POOLS), base_repo, runner=runner)
    run, handle = submit(adapter, run=dataclasses.replace(run_of("n1", 11), pool="idle"), base_repo=base_repo)
    runner.replies = sbatch_ok("901")
    moved = adapter.repool(handle, dataclasses.replace(run, pool="primary"))
    assert ["scancel", "777"] in runner.argvs() and moved.job_id == "901"
    assert moved.output_dir == handle.output_dir and moved.submitted_ts == handle.submitted_ts
    env = [c["env"] for c in runner.calls if c["argv"][0] == LAUNCHER][-1]
    assert (env["SBATCH_ACCOUNT"], env["SBATCH_QOS"], "SBATCH_REQUEUE" in env) == ("main", "high", False)
    rec = json.loads((Path(adapter.state) / "receipts/n1/screen_s11_a0.json").read_text())
    assert rec["job_id"] == "901" and rec["pool"] == "primary" and rec["env"] == env
    runner.replies = {"squeue -h -j 901": (0, "PENDING|N/A|Priority\n", "")}
    assert adapter.poll(handle).state is RunStatus.RUNNING
    runner.replies = {"squeue -h -j 901": (0, "", "")}                   # scancel took: squeue confirms it is gone
    adapter.cancel(handle)
    assert runner.argvs("scancel")[-1] == ["scancel", "901"]
    runner.replies = {LAUNCHER: (1, "", "sbatch: error: Socket timed out")}
    with pytest.raises(RuntimeError, match="repool sbatch failed"):
        adapter.repool(handle, dataclasses.replace(run, pool="primary"))
    # the old job's scancel is CONFIRMED before the relaunch: still PENDING after scancel -> raise, no sbatch
    n_sb = len([c for c in runner.calls if c["argv"][0] == LAUNCHER])
    runner.replies = {"squeue -h -j": (0, "PENDING\n", ""), "scancel": (1, "", "scancel: error: Socket timed out")}
    with pytest.raises(RuntimeError, match="not confirmed"):
        adapter.repool(handle, dataclasses.replace(run, pool="primary"))
    assert len([c for c in runner.calls if c["argv"][0] == LAUNCHER]) == n_sb


def test_nanopath_pending_jobs_asks_squeue_for_pending_protected_names(np_ctx, base_repo):
    adapter, runner = make_np(np_ctx, base_repo, runner=SlurmRunner(
        {"squeue": (0, "qwen-serve-low-pri |PENDING |Priority |3\nsomething-else |PENDING |Priority |0\n", "")}))
    assert adapter.pending_jobs(["qwen-serve-low-pri", "other"]) == ["qwen-serve-low-pri"]
    assert runner.argvs("squeue")[-1] == ["squeue", "-h", "-t", "PENDING,REQUEUED,REQUEUE_HOLD,REQUEUE_FED,PREEMPTED,SUSPENDED",
                                          "-n", "qwen-serve-low-pri,other",
                                          "-O", "Name:200|,State:20|,Reason:200|,RestartCnt:8"]
    # it could not start soon anyway: no backoff (2026-09-30: ReqNodeNotAvail, 55 restarts, ~70 slot-h idled)
    for line in ("qwen-serve-low-pri |PENDING |ReqNodeNotAvail, May be reserved for other job |3",
                 "qwen-serve-low-pri |PENDING |Priority |55"):
        runner.replies = {"squeue": (0, line + "\n", "")}
        assert adapter.pending_jobs(["qwen-serve-low-pri"]) == []
    runner.replies = {"squeue": (0, "qwen-serve-low-pri |REQUEUED |BeginTime |4\n", "")}
    assert adapter.pending_jobs(["qwen-serve-low-pri"]) == ["qwen-serve-low-pri"]   # just preempted: it restarts
    runner.replies = {"squeue": (1, "", "slurm_load_jobs error")}
    with pytest.raises(RuntimeError, match="squeue failed"):
        adapter.pending_jobs(["qwen-serve-low-pri"])


def test_nanopath_held_job_with_spent_free_retries_is_scancelled_and_voided(launched):
    l = launched
    l.runner.replies = {"squeue": (0, "PENDING|N/A|JobHeldAdmin\n", ""), **sbatch_ok("790")}
    for _ in range(10):
        assert l.adapter.poll(l.handle).state is RunStatus.RUNNING
    st = l.adapter.poll(l.handle)                                        # held again, no retry left
    assert st.state is RunStatus.VOID and st.void_type is VoidType.INFRA and "held" in st.detail
    assert l.runner.argvs("scancel")[-1] == ["scancel", "790"]


def test_nanopath_smoke_takes_a_counted_slot_in_the_picked_pool(tmp_path, base_repo):
    seen = []
    runner = SlurmRunner({LAUNCHER: (0, "Submitted batch job 57\n", "")})
    adapter, _ = make_np(np_ctx_with_locks(tmp_path / "sp", slurm=POOLS), base_repo, runner=runner)
    adapter.pool_picker = lambda: "idle"
    real = runner.run
    runner.run = lambda argv, **kw: (seen.append(dict(adapter.smokes)), real(argv, **kw))[1]
    adapter.smoke(research_wt(adapter, base_repo))
    call = next(c for c in runner.calls if c["argv"][0] == LAUNCHER)
    assert call["env"]["SBATCH_ACCOUNT"] == "preempt" and "SBATCH_NO_REQUEUE" in call["env"]
    assert {"idle": 1} in seen and adapter.smokes["idle"] == 0          # held while it runs, freed after


def test_nanopath_cancel_scancels_by_job_id(launched):
    launched.adapter.cancel(launched.handle)
    assert launched.runner.argvs("scancel") == [["scancel", "777"]]


def test_nanopath_smoke_is_a_short_one_gpu_slurm_job_through_the_launcher(np_ctx, base_repo):
    runner = SlurmRunner({LAUNCHER: (0, "Submitted batch job 55\n", ""), "sacct": (0, "COMPLETED|0:0\n", "")})
    adapter, _ = make_np(np_ctx, base_repo, runner=runner)
    wt = research_wt(adapter, base_repo)
    (Path(wt.path) / "slurm").mkdir()
    (Path(wt.path) / "slurm" / "55.out").write_text("step 1 loss 6.9\ntraining complete\n")
    res = adapter.smoke(wt)
    assert res.ok and "training complete" in res.log_tail
    call = next(c for c in runner.calls if c["argv"][0] == LAUNCHER)
    assert call["argv"][0] == LAUNCHER and call["cwd"] == wt.path       # never a local GPU process
    env = call["env"]
    assert "SBATCH_WAIT" not in env and int(env["SBATCH_TIMELIMIT"]) <= 30
    assert "SBATCH_NO_REQUEUE" in env and "SBATCH_REQUEUE" not in env     # a preempted smoke never re-queues
    assert env["WANDB_MODE"] == "disabled" and env["LABLESS_AUTOSUBMIT_FILE"] == ""
    cfg = yaml.safe_load(Path(call["argv"][1]).read_text())
    assert cfg["train"]["max_train_samples"] > 6144 and cfg["probe"]["enabled"] is False
    assert not Path(call["argv"][1]).is_relative_to(wt.path)
    (Path(wt.path) / "slurm" / "55.out").write_text("loss nan\n")
    assert "NaN" in adapter.smoke(wt).reason
    runner.replies = {LAUNCHER: (1, "Submitted batch job 56\n", "")}
    assert "rc=1" in adapter.smoke(wt).reason
    runner.replies = {LAUNCHER: (0, "Submitted batch job 56\n", ""), "sacct": (0, "FAILED|3:0\n", "")}
    assert "rc=3" in adapter.smoke(wt).reason
    runner.replies = {LAUNCHER: (124, "", "timeout")}
    res = adapter.smoke(wt)                 # the submission hung: backend unavailable, no verdict
    assert not res.ok and res.unavailable and res.reason.startswith("smoke_unavailable")
    assert ["scancel", "--name", env["SBATCH_JOB_NAME"]] in runner.argvs()
    runner.replies = {LAUNCHER: (1, "Submitted batch job 56\n", "")}
    assert not adapter.smoke(wt).unavailable


def test_nanopath_smoke_measures_flops_per_sample_for_this_code_only(np_ctx, base_repo):
    """An init swap's flop_budget: the smoke's measured_flops_per_step / its batch, valid only while the config /
    model.py / train.py it measured are unchanged; with the config's sample + FLOP budget."""
    runner = SlurmRunner({LAUNCHER: (0, "Submitted batch job 55\n", ""), "sacct": (0, "COMPLETED|0:0\n", "")})
    adapter, _ = make_np(np_ctx, base_repo, runner=runner)
    wt = research_wt(adapter, base_repo)
    assert adapter._flops(wt) is None                                        # no smoke yet: pending
    (Path(wt.path) / "slurm").mkdir()
    (Path(wt.path) / "slurm" / "55.out").write_text("[x] measured_flops_per_step: 11,040,000,000,000\nstep 1\n")
    assert adapter.smoke(wt).ok
    cfg = yaml.safe_load((Path(wt.path) / adapter.config_rel).read_text())
    got = adapter._flops(wt)
    assert got["per_sample"] == 11_040_000_000_000 / 16
    assert (got["max_samples"], got["max_flops"]) == (cfg["train"].get("max_train_samples"), cfg["train"].get("max_train_flops"))
    (Path(wt.path) / "model.py").write_text((Path(wt.path) / "model.py").read_text() + "\n# edited\n")
    assert adapter._flops(wt) is None                                        # the code changed since the smoke


def test_nanopath_smoke_timeout_counts_queue_time_only(np_ctx, base_repo, monkeypatch):
    """Live req-00424: the smoke waited 59m38s, started, and was killed 21 s later by a queue + run timer. Only
    PENDING time counts against SMOKE_WAIT_S; a RUNNING smoke has its own wall (SMOKE_RUN_S)."""
    import loop.adapters.nanopath_v2 as NP
    monkeypatch.setattr(NP, "SMOKE_POLL_S", 0.0)
    t, states = [0.0], []
    def squeue(argv):
        t[0] += 600.0
        return (0, states.pop(0) + "\n", "") if states else (0, "", "")
    runner = SlurmRunner({LAUNCHER: (0, "Submitted batch job 58\n", ""), "squeue -h -j": squeue,
                          "sacct": (0, "COMPLETED|0:0\n", "")})
    adapter, _ = make_np(np_ctx, base_repo, runner=runner, clock=lambda: t[0])
    wt = research_wt(adapter, base_repo)
    states[:] = ["PENDING"] * 5 + ["RUNNING"] * 2       # 55 min queued, then 20 min running: 75 min in all
    assert adapter.smoke(wt).ok and ["scancel", "58"] not in runner.argvs()
    states[:] = ["PENDING"] * 8                         # 80 min queued: scancelled by id, no verdict
    res = adapter.smoke(wt)
    assert res.unavailable and "queue timeout" in res.reason and ["scancel", "58"] in runner.argvs()
    runner.replies["sacct"] = (0, "PREEMPTED|0:0\n", "")
    states[:] = ["RUNNING"]                             # preempted mid-run (no requeue): no verdict either
    assert adapter.smoke(wt).unavailable


def test_nanopath_init_compat_uses_the_worktree_python(tmp_path, base_repo):
    adapter, runner = make_np(np_ctx_with_locks(tmp_path / "d"), base_repo)
    runner.real = SimpleNamespace(run=lambda argv, **kw: (0, "x: 1\n", ""))   # git show succeeds
    adapter.init_compat(Worktree(str(tmp_path / "w"), "b", "p"), parent_commit="p")
    assert runner.calls[-1]["argv"][0] == str(tmp_path / "w" / ".venv/bin/python")


@pytest.mark.skipif(not (Path(os.environ.get("NANOPATH_REPO", "")) / "model.py").is_file(),
                    reason="needs a real nanopath checkout in NANOPATH_REPO (model.py:: locks resolve against its source)")
def test_nanopath_v2_task_maps_every_writable_path_and_locks_eval():
    from loop.spec import load_context
    ctx = load_context(REPO / "tasks/nanopath_v2.yaml", require_measured=False)
    fams = ctx.cfg.families
    for path in ("train.py", "model.py", "dataloader.py"):
        hits = [f for f, r in fams.items() if any(fnmatch.fnmatch(path, g) for g in r.paths)]
        assert len(hits) == 1, (path, hits)
    assert fams["training"].init_check and fams["architecture"].init_check
    assert {"model.py", "train.py"} <= set(ctx.cfg.init_compat.paths)
    surface = ctx.spec.surface
    assert {"probe.py", "benchmarking/", "labless/", "submit/"} <= set(surface.locked)
    # eval-time code is family eval_readout (guarded), no longer locked; the budget charge stays locked
    ro = fams["eval_readout"]
    assert {"model.py::ViT.forward", "model.py::ViT.encode_image", "model.py::ViT.probe_features",
            "model.py::ViT._suppress", "model.py::ViT._forward_tokens"} <= set(ro.paths)
    assert not any(l.startswith("model.py::") for l in surface.locked)
    train_locks = [l for l in surface.locked if l.startswith("train.py::")]
    assert train_locks[:2] == [   # the charge AND the draw
        "train.py::calibration_tiles = robust_norm_tiles + len(site_rows)...train_sample_budget = "
        "max_train_samples - calibration_tiles", "train.py::robust_norm_tiles =...del rows, tables, table, order"]
    # v3 review M2: the FLOP accounting (measured at step 1, charged per step) is pinned too
    assert len(train_locks) == 10 and any("train_flops += step_train_flops" in l for l in train_locks) \
        and any("step_train_flops = measured_flops_per_step" in l for l in train_locks)
    assert ro.combine and ro.init_check and sum(r.combine for r in fams.values()) == 1
    from loop.select.intake import forbidden         # probe dataset names/roots come from the base config
    for bad in ('x = cfg["probe"]["datasets"]', 'r = "/data/thunder-data/bach"', 'for d in ("bach", "crc"):',
                'p = Path("/data/pathorob")', 'y = df["label"]', 'np.load("labels.npy")', 'import probe',
                'from benchmarking.x import y', 's = json.load(open(out / "summary.json"))'):
        assert forbidden(ctx, ("training",), "+" + bad)[0] == "surface", bad    # every family, not only readout
    # frame/introspection, environment/filesystem, and reads beyond the charged calibration draw
    for bad in ('f = sys._getframe(1).f_locals["y_true"]', "import inspect", "m = sys.modules['__main__']",
                'd = os.environ["PROBE_ROOT"]', "xs = glob.glob('/data/x/*')", "os.listdir(root)",
                'p = "/data/foo"', "extra = TileDataset(cfg)", "t = pa.ipc.open_file(x)",
                "loader = DataLoader(ds)", "arr = np.load(fn)", "jpegs = jpegs + more", "site_rows.extend(x)",
                "y = ds.labels", "for r in rows[:99999]:", "b = next(iter(train_loader))"):
        assert forbidden(ctx, ("eval_readout",), "+" + bad)[0], bad
    base = Path(ctx.spec.repo)                       # today's eval code (incl. site LDA `labels`) passes
    for rel in ("model.py", "train.py"):
        src = (base / rel).read_text()
        for entry in (e for e in ro.paths if e.startswith(rel + "::")):
            i, j, _ = _span(src, entry)
            body = "\n".join("+" + l for l in src.splitlines()[i:j + 1])
            assert forbidden(ctx, ("eval_readout",), body) == ("", "", ""), entry
    # the SURFACE guard (every family): model.py/dataloader.py pass whole; train.py only trips on its
    # own probe/labless plumbing lines, which no family may edit
    for rel in ("model.py", "dataloader.py", "train.py"):
        for line in (base / rel).read_text().splitlines():
            if forbidden(ctx, (), "+" + line)[1]:
                assert rel == "train.py" and re.search(r"probe|labless|summary\.json", line, re.I), line
    assert forbidden(ctx, ("training",), "+ds = load_probe_dataset(root)")[0] == "surface"
    assert "probe" in surface.protected_keys and ctx.spec.repo == "/home/user/nanopath/base"
    assert ctx.cfg.execution_adapter == "loop.adapters.nanopath_v2:NanopathV2Adapter"


def test_loop_never_imports_v1_trial_machinery():
    bad = re.compile(r"^\s*(from|import)\s.*(overfitx_tools|trial|worktree|commit_full|job_watch)", re.M)
    for py in (REPO / "loop").rglob("*.py"):
        assert not bad.search(py.read_text()), py
    for py in (REPO / "loop" / "adapters").glob("*.py"):
        text = py.read_text()
        assert not any(w in text for w in ("overfitx_tools", "trial.py", "commit_full", "job_watch")), py


# ============================================================ initcompat


class FakeRunner:
    """Records argv; scripted replies keyed by a substring of the joined argv."""

    def __init__(self, replies=None):
        self.calls, self.replies = [], replies or {}

    def run(self, argv, *, cwd=None, timeout=None, env=None):
        self.calls.append({"argv": list(argv), "cwd": cwd})
        joined = " ".join(argv)
        return next((r for k, r in self.replies.items() if k in joined), (0, "", ""))

GOOD = {"child": {"constructed": True, "finite": True, "loss": 2.0, "feature_rms": 1.0, "params": 22_000_000},
        "parent": {"constructed": True, "finite": True, "loss": 2.1, "feature_rms": 1.1, "params": 22_000_000},
        "keys": {"checked": True, "source": "parent_init", "missing": [], "unexpected": []}}


def variant(**patch):
    out = json.loads(json.dumps(GOOD))
    for path, val in patch.items():
        head, leaf = path.split(".")
        out[head][leaf] = val
    return out


def test_evaluate_accepts_compatible_child():
    res = initcompat.evaluate(GOOD)
    assert res.ok and res.reason == "" and set(initcompat.CHECKS) <= set(res.checks)
    assert all(c["ok"] for c in res.checks.values())


@pytest.mark.parametrize("patch,failed", [
    ({"child.constructed": False, "child.error": "RuntimeError: shape"}, "construct"),
    ({"child.finite": False}, "forward_finite"),
    ({"child.forward_error": "ValueError: nan"}, "forward_finite"),
    ({"child.loss": 40.0}, "initial_loss"),
    ({"child.feature_rms": 100.0}, "initial_loss"),
    ({"child.loss": float("nan")}, "initial_loss"),
    ({"child.params": 90_000_000}, "param_ratio"),
    ({"child.params": 5_000_000}, "param_ratio"),
    ({"keys.missing": ["blocks.3.attn.qkv.weight"]}, "checkpoint_keys"),
    ({"keys.unexpected": ["blocks.12.norm1.weight"]}, "checkpoint_keys"),
    ({"parent.constructed": False}, "construct"),
])
def test_evaluate_rejects_each_broken_property(patch, failed):
    res = initcompat.evaluate(variant(**patch))
    assert not res.ok and not res.checks[failed]["ok"] and failed in res.reason


def test_evaluate_limits_are_configurable():
    m = variant(**{"child.loss": 5.0})
    assert not initcompat.evaluate(m).ok
    assert initcompat.evaluate(m, initcompat.Limits(loss_factor=3.0)).ok


def test_evaluate_fails_closed_on_empty_measurements():
    res = initcompat.evaluate({})
    assert not res.ok and all(not res.checks[c]["ok"] for c in initcompat.CHECKS)


def test_parse_measurements_takes_last_verdict_line():
    noisy = "warn: something\n" + json.dumps({"other": 1}) + "\n" + json.dumps(GOOD) + "\ntrailing text\n"
    assert initcompat.parse_measurements(noisy) == GOOD
    assert initcompat.parse_measurements("no json here") is None


VITB = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth"
DENY = ("bioptimus|h[-_ ]?optimus", "path[-_]?fm", "genbio")


def swap(src=VITB, typ="dinov2_vitb14_reg", flops=None, **keys):
    """An init-swap measurement: ViT-S parent -> ``typ`` / ``src`` (params 4x: the parent-relative checks would fail)."""
    out = variant(**{"child.params": 86_000_000, "child.loss": 40.0})
    out["child"].update(type=typ, source=src)
    out["parent"].update(type="robust_norm_dinov2_vits14_reg", source=VITB.replace("vitb14", "vits14"))
    out["keys"] = {"swap": True, "loader": True, "finite": True, "loads": [{"keys": 176, "n_missing": 0}], **keys}
    out["flops"] = flops
    return out


def test_init_swap_skips_parent_relative_checks_and_passes_a_legal_new_init():
    res = initcompat.evaluate(swap(flops={"per_sample": 6.9e11, "max_samples": 1_000_000, "max_flops": 1e18}),
                              initcompat.Limits(deny=DENY))
    assert res.ok and res.swap and set(res.checks) == {*initcompat.SWAP_CHECKS, "init_swap"}
    assert res.checks["init_swap"]["to"] == {"type": "dinov2_vitb14_reg", "source": VITB}
    assert res.checks["flop_budget"]["projected"] == pytest.approx(0.69)
    pending = initcompat.evaluate(swap(), initcompat.Limits(deny=DENY))       # no smoke yet: not judged
    assert pending.ok and "pending" in pending.checks["flop_budget"]
    rand = initcompat.evaluate(swap(src="", typ="my_new_arch", loader=False, finite=None, loads=None))
    assert rand.ok                                                            # a random init / new architecture


@pytest.mark.parametrize("src", ["hf-hub:bioptimus/H-optimus-0", "/x/OpenMidnight/path-fm-dinov3/teacher.pth",
                                 "GenBio-PathFM"])
def test_init_swap_to_a_pathology_checkpoint_is_init_illegal(src):
    res = initcompat.evaluate(swap(src=src), initcompat.Limits(deny=DENY))
    assert not res.ok and res.rule == "init_illegal" and res.reason.startswith("init_illegal: ")
    assert src in res.reason and "pathology-trained" in res.reason


def test_init_swap_over_the_flop_cap_reports_the_projection():
    res = initcompat.evaluate(swap(flops={"per_sample": 1.3e12, "max_samples": 1_000_000, "max_flops": 1e18}),
                              initcompat.Limits(deny=DENY))
    assert not res.ok and res.rule == "over_flop_cap" and res.checks["flop_budget"]["projected"] == pytest.approx(1.3)
    assert res.reason.startswith("over_flop_cap: projected flop_fraction 1.300 > 0.97 at the full 1,000,000 samples")
    unprinted = initcompat.evaluate(swap(flops={"per_sample": None, "max_samples": 1_000_000, "max_flops": 1e18}))
    assert not unprinted.ok and unprinted.rule == "over_flop_cap" and "measured_flops_per_step" in unprinted.reason


@pytest.mark.parametrize("keys,failed", [({"error": "AssertionError: (['x'], [])"}, "load"),
                                         ({"finite": False}, "load")])
def test_init_swap_that_cannot_load_is_init_incompatible(keys, failed):
    res = initcompat.evaluate(swap(**keys))
    assert not res.ok and res.rule == "init_incompatible" and not res.checks[failed]["ok"]


def test_non_swap_child_keeps_the_five_parent_relative_checks():
    same = variant(**{"child.params": 86_000_000})
    for side in ("child", "parent"):
        same[side].update(type="robust_norm_dinov2_vits14_reg", source=VITB)
    res = initcompat.evaluate({**same, "flops": {"per_sample": 9e12, "max_samples": 1, "max_flops": 1}},
                              initcompat.Limits(deny=("vitb14",)))
    assert not res.ok and not res.swap and set(res.checks) == set(initcompat.CHECKS)
    assert not res.checks["param_ratio"]["ok"] and res.rule == "init_incompatible"


def test_measuring_script_is_valid_python():
    compile(initcompat.SCRIPT, "check_init.py", "exec")


def init_env(tmp_path, model_type="vit_s"):
    wt = tmp_path / "wt"
    (wt / "configs").mkdir(parents=True)
    (wt / "configs" / "main.yaml").write_text(f"model:\n  type: {model_type}\n")
    return wt


def test_run_init_check_extracts_parent_and_parses_verdict(tmp_path):
    wt = init_env(tmp_path, "vit_l")

    runner = FakeRunner(replies={"show parentsha:model.py": (0, "# parent model\n", ""),
                                 "show parentsha:configs/main.yaml": (0, "model:\n  type: vit_s\n", ""),
                                 "check_init.py": (0, "log line\n" + json.dumps(GOOD) + "\n", "")})
    res = initcompat.run_init_check(runner=runner, worktree_path=str(wt), parent_commit="parentsha",
                                    state_dir=str(tmp_path / "state"))
    assert res.ok
    script_call = runner.calls[-1]
    argv = script_call["argv"]
    assert argv[0] == "python" and argv[1].endswith("check_init.py")
    assert script_call["cwd"] == str(wt) and argv[argv.index("--child-dir") + 1] == str(wt)
    parent_dir = Path(argv[argv.index("--parent-dir") + 1])
    assert (parent_dir / "model.py").read_text() == "# parent model\n"
    assert (parent_dir / "configs" / "main.yaml").is_file()


def test_run_init_check_fails_closed(tmp_path):
    wt = init_env(tmp_path)
    missing = FakeRunner(replies={"show": (128, "", "fatal: path 'model.py' does not exist")})
    res = initcompat.run_init_check(runner=missing, worktree_path=str(wt), parent_commit="p", state_dir=str(tmp_path / "s"))
    assert not res.ok and "cannot read model.py" in res.reason
    silent = FakeRunner(replies={"check_init.py": (1, "", "ModuleNotFoundError: torch")})
    res = initcompat.run_init_check(runner=silent, worktree_path=str(wt), parent_commit="p", state_dir=str(tmp_path / "s"))
    assert not res.ok and "no verdict" in res.reason and "torch" in res.reason
    bad = FakeRunner(replies={"check_init.py": (0, json.dumps(variant(**{"child.loss": 99.0})), "")})
    res = initcompat.run_init_check(runner=bad, worktree_path=str(wt), parent_commit="p", state_dir=str(tmp_path / "s"))
    assert not res.ok and "initial_loss" in res.reason


# ============================================================== hygiene


def test_adapters_import_without_root_config():
    code = ("import sys, loop.adapters, loop.adapters.sim, loop.adapters.nanopath_v2, "
            "loop.adapters.initcompat; assert 'config' not in sys.modules, 'config imported'; "
            "assert 'runtime.supervise' not in sys.modules and 'llm' not in sys.modules")
    proc = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_tests_can_never_reach_real_slurm():
    import shutil
    import subprocess as sp
    from loop.adapters.nanopath_v2 import SubprocessRunner
    assert "fake_slurm_bin" in shutil.which("sbatch")
    cp = sp.run(["sbatch", "x"], capture_output=True, text=True)
    assert cp.returncode == 1 and "REAL SLURM CALLED FROM TESTS" in cp.stderr
    with pytest.raises(RuntimeError, match="REAL SLURM CALLED FROM TESTS"):
        SubprocessRunner().run(["./submit/train_1gpu.sbatch", "cfg.yaml"])


def test_nanopath_tidy_tags_node_commits_then_drops_stale_research_branches_and_node_worktrees(np_ctx, base_repo):
    """386 worktrees / 343 branches piled up in the base repo. A node's commit stays reachable (a tag) before
    its research branch goes."""
    adapter, _ = make_np(np_ctx, base_repo)
    wt = research_wt(adapter, base_repo, "loop-req-00001")
    (Path(wt.path) / "train.py").write_text(TRAIN_PY.replace("lr = 0.1", "lr = 0.3"))
    c1 = adapter.commit(wt, message="n1")
    adapter.reap_worktree(wt)
    wt2 = research_wt(adapter, base_repo, "loop-req-00002")                 # still registered (a lost reap)
    live = research_wt(adapter, base_repo, "loop-req-00003")
    node_wt = adapter._node_worktree("n1", c1)
    tag = adapter._tag
    assert adapter.tidy({"n1": c1}, [], ["n1"], dry_run=True)["worktrees"] == 0   # launched within WT_RECENT_S: kept
    adapter._wt_used.clear()                                                # ... an hour ago
    dry = adapter.tidy({"n1": c1}, ["req-00001", "req-00002"], ["n1"], dry_run=True)
    assert dry == {"tagged": 1, "branches": 2, "worktrees": 1, "stray": 0, "stray_removed": 0, "stray_kept": 0} \
        and node_wt.exists()
    assert git(base_repo, "rev-parse", f"refs/tags/loop/{tag}/n1") == c1   # tagged even in a dry run
    n = adapter.tidy({"n1": c1, "n9": "1" * 40}, ["req-00001", "req-00002"], ["n1"])
    assert n == {"tagged": 1, "branches": 2, "worktrees": 1, "stray": 0, "stray_removed": 0, "stray_kept": 0}
    heads = git(base_repo, "for-each-ref", "--format=%(refname:short)", "refs/heads/loop")
    assert heads.split() == [f"loop/{tag}-loop-req-00003"] and Path(live.path).exists()
    assert not Path(wt2.path).exists() and not node_wt.exists()
    assert git(base_repo, "cat-file", "-t", c1) == "commit"                 # still reachable, via the tag
    assert adapter._node_worktree("n1", c1).exists()                        # a later launch re-adds it


def test_nanopath_resumable_run_config_copy_arms_on_first_checkpoint_and_interrupt_resumes(np_ctx, base_repo):
    """slurm.resume: a resumable run launches from a config copy (save_every lowered, wandb off); once latest.pt
    exists the copy gets train.resume, so a requeue / free retry / forced interrupt continues from it; the
    receipt counts the resumed starts."""
    runner = SlurmRunner(sbatch_ok("777"))
    adapter, _ = make_np(np_ctx, base_repo, runner=runner)
    run = run_of("n1", 11)
    adapter.resume_runs = {run.id}
    _, handle = submit(adapter, run=run, base_repo=base_repo)
    call = runner.calls[-1]
    cfg = Path(call["argv"][1])
    assert cfg == adapter.state / "resume/n1/screen_s11_a0.yaml" and call["env"]["WANDB_MODE"] == "disabled"
    tree = yaml.safe_load(cfg.read_text())
    assert tree["train"]["save_every"] == 500 and "resume" not in tree["train"] and tree["train"]["lr"] == 1e-4
    out = Path(handle.output_dir)
    out.mkdir(parents=True)
    runner.replies = {"squeue -h -j": (0, "RUNNING|2026-09-28T12:00:00||0\n", "")}
    adapter.poll(handle)
    assert adapter.resume_state(handle) == {"armed": False, "resumed": 0, "fraction": 0.0, "resumable": True}
    (out / "latest.pt").write_bytes(b"x")
    (out / "metrics.jsonl").write_text('{"sample_fraction": 0.55, "step": 4000}\n{"val_total": 1}\n')
    adapter.poll(handle)
    assert yaml.safe_load(cfg.read_text())["train"]["resume"] == str(out / "latest.pt")
    assert adapter.resume_state(handle) == {"armed": True, "resumed": 0, "fraction": 0.55, "resumable": True}
    runner.replies = {"squeue -h -j": (0, "PENDING|N/A|BeginTime|1\n", "")}
    adapter.poll(handle)                                                    # a SLURM requeue: resumes
    assert adapter.resume_state(handle)["resumed"] == 1
    runner.replies = sbatch_ok("778")
    assert adapter.interrupt(handle) == "778" and ["scancel", "777"] in runner.argvs()
    assert runner.calls[-1]["argv"][1] == str(cfg) and adapter.resume_state(handle)["resumed"] == 2
    plain, _ = make_np(np_ctx, base_repo, runner=SlurmRunner(sbatch_ok()))  # not resumable: the commit's config
    _, h2 = submit(plain, run=run_of("n1", 12), base_repo=base_repo)
    with pytest.raises(RuntimeError, match="no latest.pt"):
        plain.interrupt(h2)


def test_nanopath_activation_checkpointing_rung_launches_a_config_copy_with_the_flag(np_ctx, base_repo):
    """The v3 runner's last OOM rung (loop.runner.ACT_CKPT): a config copy with train.activation_checkpointing on
    (the CLI takes only output_dir / seed), the allocator env kept, wandb untouched; the resumable copy gets the flag
    too. The old driver never asks for it: oom_ladder is unchanged."""
    from loop.adapters.nanopath_v2 import ACT_CKPT, ALLOC_ENV
    runner = SlurmRunner(sbatch_ok("778"))
    adapter, _ = make_np(np_ctx, base_repo, runner=runner)
    assert ACT_CKPT not in adapter.oom_ladder("x")
    submit(adapter, run=dataclasses.replace(run_of("n1", 11), mitigation=ACT_CKPT), base_repo=base_repo)
    call = runner.calls[-1]
    cfg = Path(call["argv"][1])
    assert cfg == adapter.state / "oom/n1/screen_s11_a0.yaml" and call["argv"][-1] == "seed=11"
    tree = yaml.safe_load(cfg.read_text())
    assert tree["train"]["activation_checkpointing"] is True and tree["train"]["lr"] == 1e-4
    assert call["env"]["WANDB_MODE"] == "offline" and all(call["env"][k] == v for k, v in ALLOC_ENV.items())
    run = dataclasses.replace(run_of("n2", 12), mitigation=ACT_CKPT)
    adapter.resume_runs = {run.id}
    submit(adapter, run=run, base_repo=base_repo)
    tree = yaml.safe_load(Path(runner.calls[-1]["argv"][1]).read_text())
    assert tree["train"]["activation_checkpointing"] is True and tree["train"]["save_every"] == 500


def test_tick_retry_of_a_rejected_submit_skips_that_row_and_retries_the_rest():
    """A retry whose submit raises SubmissionRejected (e.g. a stale node worktree it may not heal) ends that row;
    the tick's other owed retries still go (it used to propagate out of Tick._retries)."""
    from loop.tick import Tick
    bad = dataclasses.replace(run_of("root", 3, Stage.BASELINE, attempt=2), status=RunStatus.VOID,
                              void_type=VoidType.INFRA)
    good = dataclasses.replace(bad, id="root:baseline:4", seed=4)
    tried = []

    def retry(run, node):
        tried.append(run.id)
        if run.id == bad.id:
            raise SubmissionRejected("worktree is at a, not b (not healed: 1 modified tracked file(s))")
        return dataclasses.replace(run, status=RunStatus.RUNNING)
    fake = SimpleNamespace(store=SimpleNamespace(runs=lambda status: [bad, good], get_node=lambda nid: object()),
                           book=SimpleNamespace(infra_retry=8, free=lambda: 4, pool=lambda *a: "primary", retry=retry))
    acts = []
    assert Tick._retries(fake, {}, acts) == [good.id] and tried == [bad.id, good.id]
    assert acts[0].startswith(f"retry {bad.id} rejected") and acts[1] == f"retry {good.id} -> running"
