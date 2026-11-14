"""Review M1 (10-13): a crossover could ship a LOCKED file that differs from its first parent.

The drift exemption only filtered locked paths out of the diff vs the parents' common ancestor, so an agent that
restored a locked file to the merge-base content (no base diff left) shipped it. Now: (a) every candidate's locked
paths / spans are judged on its diff vs the FIRST PARENT; (b) check() applies the same ancestry exemption (it never
tells the agent to revert a drifted locked file); (c) bash refuses writes onto locked paths and restores whatever a
command changed there anyway; (d) the adapter refuses to LAUNCH a commit whose locked content differs from the
canonical copy (void, loud log). Real git throughout."""

from __future__ import annotations

import dataclasses
import json
import subprocess
from pathlib import Path

import pytest

from loop.adapters.nanopath_v2 import NanopathV2Adapter
from loop.agents import FakeLLMClient, LLMResearcher
from loop.agents.researcher import WorktreeTools
from loop.contracts import (InitCompat, NodeStatus, Operator, RunStatus, SmokeResult, Stage, SubmissionRejected,
                            Run)
from tests.loop.test_adapters import (LAUNCHER, base_repo, make_np, np_ctx, np_ctx_with_locks,  # noqa: F401
                                      sbatch_ok)
from tests.loop.test_agents import FINISH, MemStore, _ctx, _node, _request, _snapshot


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                          capture_output=True, text=True, check=True).stdout.strip()


MODEL = "LR = 1e-3\n\n\n\nWD = 0.05\n\n\n\nDROP = 0.0\n"


def _commit(repo: Path, files: dict[str, str], msg: str) -> str:
    for rel, text in files.items():
        (repo / rel).write_text(text)
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", msg)
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def xenv(tmp_path):
    """A: the parents' common ancestor (fake_probe.py SCORE = 1); E: a base-epoch commit that changed the locked
    fake_probe.py (SCORE = 2); P1 (on E) and P2 (on A) each change a different line of fake_model.py."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".gitignore").write_text(".venv\nslurm/\n")
    git(repo, "init", "-q", "-b", "main")
    a = _commit(repo, {"fake_model.py": MODEL, "fake_probe.py": "SCORE = 1\n",
                       "fake_config.yaml": "train:\n  lr: 0.001\n"}, "A")
    e = _commit(repo, {"fake_probe.py": "SCORE = 2\n"}, "E: epoch drift of a locked file")
    p1 = _commit(repo, {"fake_model.py": MODEL.replace("WD = 0.05", "WD = 0.02")}, "P1")
    git(repo, "checkout", "-q", "-b", "side", a)
    p2 = _commit(repo, {"fake_model.py": MODEL.replace("DROP = 0.0", "DROP = 0.1")}, "P2")
    git(repo, "checkout", "-q", "main")
    ctx = dataclasses.replace(_ctx(), state_dir=str(tmp_path / "state"))
    ad = NanopathV2Adapter(ctx, repo=str(repo), runs_root=str(tmp_path / "runs"), worktrees_root=str(tmp_path / "wts"))
    ad.root_commit = lambda: a
    ad.lock_ref = e
    ad.smoke = lambda wt, **kw: SmokeResult(ok=True, reason="ok", log_tail="")
    ad.init_compat = lambda wt, **kw: InitCompat(ok=True, reason="")
    store = MemStore()
    store.put_node(_node("a", status=NodeStatus.CONFIRMED, family="optim", commit=p1))
    store.put_node(_node("b", status=NodeStatus.CONFIRMED, family="optim", commit=p2))
    return ad, store, repo, {"A": a, "E": e, "P1": p1, "P2": p2}


EDIT = ("edit_file", {"path": "fake_model.py", "old": "LR = 1e-3", "new": "LR = 5e-4"})


def _crossover(ad, store, calls):
    llm = FakeLLMClient({"researcher": [{"tool_calls": calls, "finish": FINISH}]})
    res = LLMResearcher(ad.ctx, llm, ad).research(_request(Operator.CROSSOVER, parents=("a", "b")), store,
                                                  _snapshot())
    return res, llm.calls[0]["tool_log"]


def test_crossover_drift_is_exempt_and_check_never_says_revert(xenv):
    """The control: the epoch-drifted locked file is in the base diff, not the session's own work -> ok, and the
    in-session check is OK (it no longer tells the agent to revert fake_probe.py)."""
    ad, store, _, c = xenv
    res, log = _crossover(ad, store, [EDIT, ("check", {"family": "optim", "mechanism_tag": "lr_down"})])
    assert res.status == "ok", res.error
    assert "fake_probe.py" not in res.candidate.changed_paths
    chk = next(o for n, _, o in log if n == "check")
    assert "surface_locked" not in chk and "fake_probe.py" not in chk
    assert git(ad.repo, "show", f"{res.candidate.commit}:fake_probe.py") == "SCORE = 2"


def test_the_attack_restoring_a_locked_file_to_merge_base_content_is_refused_by_bash(xenv):
    ad, store, _, c = xenv
    res, log = _crossover(ad, store, [EDIT, ("bash", {"command": f"git show {c['A']}:fake_probe.py > fake_probe.py"})])
    out = next(o for n, _, o in log if n == "bash")
    assert out.startswith("REFUSED (LOCKED)") and "fake_probe.py" in out
    assert res.status == "ok" and git(ad.repo, "show", f"{res.candidate.commit}:fake_probe.py") == "SCORE = 2"


def test_bash_writes_that_slip_past_the_precheck_are_restored(xenv):
    """`git apply -R` / a python one-liner name no locked path: the post-command check restores the file."""
    ad, store, repo, c = xenv
    (repo.parent / "unused").mkdir()
    res, log = _crossover(ad, store, [
        ("bash", {"command": f"git diff {c['E']} {c['A']} -- fake_probe.py > undo.patch && git apply undo.patch"}),
        ("bash", {"command": "python3 -c \"open('fake_probe.py','w').write('SCORE = 1')\""}),
        ("bash", {"command": "rm -f undo.patch"}), EDIT])
    outs = [o for n, _, o in log if n == "bash"]
    assert outs[0].startswith("REFUSED (LOCKED)") and "restored" in outs[0]
    assert outs[1].startswith("REFUSED (LOCKED)") and "restored" in outs[1]
    assert res.status == "ok", res.error
    assert git(ad.repo, "show", f"{res.candidate.commit}:fake_probe.py") == "SCORE = 2"


def test_the_attack_is_caught_by_the_final_check_when_bash_guards_are_bypassed(xenv, monkeypatch):
    """(a): with both bash guards off, the restored-to-merge-base locked file has NO base diff -- the old final
    check exempted it as drift and shipped it. The diff vs the first parent catches it: no commit."""
    ad, store, _, c = xenv
    monkeypatch.setattr(WorktreeTools, "_locked_write", lambda self, cmd: "")
    monkeypatch.setattr(WorktreeTools, "_restore_locked", lambda self, before: [])
    commits = []
    real = ad.commit
    ad.commit = lambda wt, **kw: commits.append(1) or real(wt, **kw)
    res, log = _crossover(ad, store, [EDIT, ("bash", {"command": f"git show {c['A']}:fake_probe.py > fake_probe.py"}),
                                      ("finish", FINISH), ("finish", FINISH)])
    assert res.status != "ok" and not commits
    assert "fake_probe.py" in (res.error or "") or any("fake_probe.py" in o for n, _, o in log if n == "finish")
    # (b) the in-session check names it too (the session's OWN change of a locked file)
    assert any("surface_locked" in o and "fake_probe.py" in o for n, _, o in log if n == "finish")


def test_own_surface_unit(xenv):
    ad, store, _, c = xenv
    wt = ad.prepare_worktree(parent_commit=c["P1"], name="u1")
    wt = dataclasses.replace(wt, base_commit=c["A"])
    r = LLMResearcher(ad.ctx, FakeLLMClient({}), ad)
    assert r._own_surface(wt, c["P1"], ("fake_probe.py", "fake_model.py")) == ("fake_model.py",)   # drift
    (Path(wt.path) / "fake_probe.py").write_text("SCORE = 1\n")                      # back to merge base
    _, base_paths, _ = ad.diff(wt)
    assert "fake_probe.py" not in base_paths                                         # invisible vs the ancestor
    assert "fake_probe.py" in r._own_surface(wt, c["P1"], base_paths)                # visible vs the parent


# ------------------------------------------------------------------ (d) launch-time guard

def _run(node="n1"):
    return Run(id=Run.make_id(node, Stage.SCREEN, 11), task_id="fake-loop", node_id=node, seed=11,
               stage=Stage.SCREEN, status=RunStatus.SUBMITTING)


def test_launch_refuses_a_commit_whose_locked_files_differ(np_ctx, base_repo):
    ad, runner = make_np(np_ctx, base_repo, runner=None)
    runner.replies.update(sbatch_ok())
    good = git(base_repo, "rev-parse", "HEAD")
    assert ad.lock_violations(good) == []
    (base_repo / "probe.py").write_text("SCORE = 9\n")                 # a locked whole file, new
    bad = _commit(base_repo, {}, "touch probe")
    with pytest.raises(SubmissionRejected, match="LOCKED-FILE REFUSAL"):
        ad.submit(run=_run(), commit=bad)
    assert not runner.argvs(LAUNCHER)
    rec = json.loads((Path(np_ctx.state_dir) / "lock_refusals.jsonl").read_text().splitlines()[0])
    assert rec["commit"] == bad and rec["paths"] == ["probe.py"]
    git(base_repo, "checkout", "-q", good)
    span = _commit(base_repo, {"train.py": (base_repo / "train.py").read_text().replace(
        "calibration_tiles = robust_norm_tiles + 1", "calibration_tiles = 0")}, "span")
    assert any(v.startswith("train.py::robust_norm_tiles") for v in ad.lock_violations(span))
    git(base_repo, "checkout", "-q", good)
    ok = _commit(base_repo, {"train.py": (base_repo / "train.py").read_text().replace("lr = 0.1", "lr = 0.2")},
                 "outside the span")
    assert ad.lock_violations(ok) == []
    assert ad.submit(run=_run("n2"), commit=ok).job_id == "777"


def test_launch_guard_fails_closed_on_an_unreadable_ref(np_ctx, base_repo):
    ad, runner = make_np(np_ctx, base_repo)
    ad.lock_ref = "0" * 40
    with pytest.raises(SubmissionRejected):
        ad.submit(run=_run(), commit=git(base_repo, "rev-parse", "HEAD"))
