"""F1 execution integrity of the nanobrain adapter + job script (adversarial ops review): the wall budget across
SLURM instances (C1), the ambiguous-lock refusals (C1), the shadow data root (C2, in the job-script tests of
test_nanobrain_adapter), binary / large / foreign files and the weight-loading deny list (C3), the fabricated eval
dir and the per-task-json validation + re-scoring (C4), resumed-instance counting (H1), the eval continuation after
a TIMEOUT (H2), smoke schedule scaling (H4), the real GUARD (M1), the TERM trap / exit 75 (M3), the CPU heartbeat
(M4), per-job compile caches (M6), TEST runs on the primary pool (L3). Fake runner / fake python; no SLURM / GPU."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path

import pytest
import yaml

from loop.adapters.nanobrain import (JOB_SCRIPT, MAX_FILE_BYTES, SETUP_ALLOWANCE_S, NanobrainAdapter, meta_dir)
from loop.adapters.nanopath_v2 import locked_source
from loop.contracts import RunStatus, Stage, SubmissionRejected, VoidType, Worktree
from tests.loop.test_adapters import SlurmRunner, git
from tests.loop.test_nanobrain_adapter import (CONFIG_REL, MISC_PY, NB_BASE, SUMMARY, TRAIN_PY, _fake_base, _job_env,
                                               _run_job, add_events, commit_change, config_with, events_of, finish,
                                               launched, make_nb, nb_repo, run_of, sbatch, task_json, write_eval)

__all__ = ("launched", "nb_repo")          # fixtures re-used from the adapter tests


# ============================================================ C1: the wall budget
def test_train_wall_over_the_hour_across_instances_is_a_budget_void(launched):
    a, _, h = launched
    write_eval(a, h, events=False)
    out = Path(h.output_dir)
    now = time.time()
    # two instances (a preemption + resume): 2 x 1950 s = 3900 s > 3600 + 2 x 120 s
    add_events(out, {"event": "train_start", "ts": now - 4100, "resume": False},
               {"event": "train_end", "ts": now - 2150, "rc": 0, "stopped": True},
               {"event": "train_start", "ts": now - 2050, "resume": True},
               {"event": "train_end", "ts": now - 100, "rc": 0, "done": True},
               {"event": "train_done", "ts": now - 90}, {"event": "eval_start", "ts": now - 80})
    st = finish(a, h)
    assert st.state is RunStatus.VOID and st.void_type is VoidType.TOO_SLOW and st.detail.startswith("budget:")
    assert "3900 s over 2 instance(s)" in st.detail


def test_setup_allowance_per_instance_and_hard_killed_instances(launched):
    a, _, h = launched
    write_eval(a, h, events=False)
    out = Path(h.output_dir)
    now = time.time()
    # 3600 + 2 x 100 s of import / exit: inside 3600 + 2 x SETUP_ALLOWANCE_S; a hard-killed instance (no end) in the
    # middle is not counted (its unsaved steps were lost with it); the run is resumed twice
    assert SETUP_ALLOWANCE_S == 120.0
    add_events(out, {"event": "train_start", "ts": now - 5000, "resume": False},
               {"event": "train_end", "ts": now - 5000 + 1900, "rc": 0, "stopped": True},
               {"event": "train_start", "ts": now - 3000, "resume": True},          # SIGKILLed: no train_end
               {"event": "train_start", "ts": now - 1900, "resume": True},
               {"event": "train_end", "ts": now - 100, "rc": 0, "done": True},
               {"event": "train_done", "ts": now - 90}, {"event": "eval_start", "ts": now - 80})
    st = finish(a, h)
    assert st.state is RunStatus.DONE, st.detail
    assert st.resumed == 2                                                  # H1: resumed instances reach the record


def test_a_fresh_restart_resets_the_budget_clock(launched):
    """A TEST run (auto_resume=false) re-launched after a node failure trains from step 0 again: train.py's clock
    restarts, so the hour counts from that fresh start."""
    a, _, h = launched
    write_eval(a, h, events=False)
    now = time.time()
    add_events(Path(h.output_dir), {"event": "train_start", "ts": now - 7500, "resume": False},
               {"event": "train_end", "ts": now - 4000, "rc": 0, "stopped": True},
               {"event": "train_start", "ts": now - 3700, "resume": False},
               {"event": "train_end", "ts": now - 100, "rc": 0, "done": True},
               {"event": "train_done", "ts": now - 90}, {"event": "eval_start", "ts": now - 80})
    st = finish(a, h)
    assert st.state is RunStatus.DONE, st.detail
    assert st.resumed == 0


def test_summary_wall_over_the_cap_is_a_budget_void(launched):
    a, _, h = launched
    write_eval(a, h, summary={**SUMMARY, "wall_seconds": 3600 + 61})
    st = finish(a, h)
    assert st.void_type is VoidType.TOO_SLOW and "summary.wall_seconds 3661 s" in st.detail
    write_eval(a, h, summary={**SUMMARY, "wall_seconds": 3607.3})          # the real overshoot (+7 s): fine
    assert finish(a, h).state is RunStatus.DONE


def test_resumed_instances_are_reported_while_running(launched):
    a, runner, h = launched
    add_events(Path(h.output_dir), {"event": "train_start", "ts": 1.0, "resume": False},
               {"event": "train_start", "ts": 2.0, "resume": True})
    runner.replies = {"squeue": (0, "RUNNING|2026-11-08T00:00:00|None|1\n", ""), "sacct": (0, "RUNNING|60\n", "")}
    a.freeze_s = 1e12
    assert a.poll(h).resumed == 1


# ============================================================ C1: ambiguous locks
@pytest.mark.parametrize("files,expect", [
    # a decoy copy of the span's start line above the real one would take the pin: refused
    ({"src/nanobrain/train.py": TRAIN_PY.replace(
        "def train(", "# time_limit = args.time_limit * 60 if args.time_limit else float('inf')\ndef train(", 1)
        .replace("args.time_limit * 60 if", "args.time_limit * 600 if", 2).replace(
            "# time_limit = args.time_limit * 600", "# time_limit = args.time_limit * 60", 1)},
     "src/nanobrain/train.py::time_limit = args.time_limit * 60"),
    # a second def load_model below the locked one replaces it at import
    ({"src/nanobrain/misc.py": MISC_PY + "\n\ndef load_model(args, *a):\n    return 0, -1e9\n"},
     "src/nanobrain/misc.py::load_model"),
    # ... so does a rebinding
    ({"src/nanobrain/misc.py": MISC_PY + "\n\nload_model = lambda *a: (0, -1e9)\n"}, "src/nanobrain/misc.py::load_model"),
    ({"src/nanobrain/misc.py": "from os import getcwd as git_info\n" + MISC_PY}, "src/nanobrain/misc.py::git_info"),
])
def test_ambiguous_locks_refuse_the_launch(tmp_path, nb_repo, files, expect):
    a, runner = make_nb(tmp_path, nb_repo, runner=SlurmRunner(sbatch()))
    sha = commit_change(a, nb_repo, "dup", files)
    assert expect in a.lock_violations(sha)
    with pytest.raises(SubmissionRejected):
        a.submit(run=run_of(), commit=sha)


def test_unambiguous_edits_beside_the_locks_pass(tmp_path, nb_repo):
    a, _ = make_nb(tmp_path, nb_repo)
    sha = commit_change(a, nb_repo, "ok", {"src/nanobrain/misc.py": MISC_PY + "\n\ndef helper():\n"
                                           "    load = 1\n    return load\n\n\nclass Cache:\n"
                                           "    def load_model(self):\n        return None\n"})
    # a METHOD named load_model is a second def of that name anywhere in the module: refused (conservative)
    assert "src/nanobrain/misc.py::load_model" in a.lock_violations(sha)
    sha2 = commit_change(a, nb_repo, "ok2", {"src/nanobrain/misc.py": MISC_PY + "\n\ndef helper():\n    x = 1\n"
                                            "    return x\n"})
    assert a.lock_violations(sha2) == []


def test_span_and_def_resolution_rules():
    src = "a = 1\nSTART x\nmid\nEND\n"
    assert locked_source(src, "f.py::START...END") == "START x\nmid\nEND"
    assert locked_source(src + "START again\n", "f.py::START...END") is None
    code = "def f():\n    return 1\n\n\nclass C:\n    def g(self):\n        f = 2\n        return f\n"
    assert locked_source(code, "m.py::f") is not None              # a LOCAL f in another scope is no rebinding
    assert locked_source(code + "f = 3\n", "m.py::f") is None
    assert locked_source(code + "for f in []:\n    pass\n", "m.py::f") is None
    assert locked_source(code, "m.py::C.g") is not None
    assert locked_source(code.replace("        f = 2", "        g = 2"), "m.py::C.g") is not None
    assert locked_source(code + "    g = None\n", "m.py::C.g") is None   # class-level rebinding of the method


# ============================================================ C3: binary / large / foreign files, weight loading
def test_binary_large_and_foreign_files_refuse_the_launch(tmp_path, nb_repo):
    a, _ = make_nb(tmp_path, nb_repo)
    base = git(nb_repo, "rev-parse", "HEAD")

    def commit_raw(name, rel, data: bytes):
        wt = a.prepare_worktree(parent_commit=base, name=name)
        (Path(wt.path) / rel).parent.mkdir(parents=True, exist_ok=True)
        (Path(wt.path) / rel).write_bytes(data)
        return a.commit(wt, message=name)
    assert "binary:src/nanobrain/init.bin" in a.lock_violations(commit_raw("b", "src/nanobrain/init.bin",
                                                                            b"\x00\x01\x02weights"))
    big = commit_raw("l", "src/nanobrain/table.py", b"W = '" + b"a" * (MAX_FILE_BYTES + 10) + b"'\n")
    assert "large:src/nanobrain/table.py" in a.lock_violations(big)
    assert "foreign:src/brainmarks_smri" in a.lock_violations(commit_raw("f", "src/brainmarks_smri/__init__.py",
                                                                          b"x = 1\n"))
    assert "foreign:src/sitecustomize.py" in a.lock_violations(commit_raw("s", "src/sitecustomize.py", b"x = 1\n"))
    assert a.lock_violations(commit_raw("ok", "src/nanobrain/extra.py", b"x = 1\n")) == []


@pytest.mark.parametrize("line,ok", [
    ("state = torch.load('/x/brain.pt')", False),
    ("from safetensors.torch import load_file", False),
    ("model.load_state_dict(ckpt['model'])", False),
    ("w = np.load('init.npy')", False),
    ("enc = timm.create_model('vit_base', pretrained=True)", False),
    ("root = '/data/other/volumes'", False),
    ("teacher.load_state_dict(student.state_dict())", True),       # an EMA teacher copy
    ("from torch.utils.checkpoint import checkpoint", True),       # activation checkpointing
    ("x = self.blocks[i](x)", True),
])
def test_weight_loading_idioms_are_init_illegal_in_any_file(tmp_path, nb_repo, line, ok):
    a, _ = make_nb(tmp_path, nb_repo)
    base = git(nb_repo, "rev-parse", "HEAD")
    wt = a.prepare_worktree(parent_commit=base, name="w")
    (Path(wt.path) / "src" / "nanobrain" / "data.py").write_text(f"def f():\n    {line}\n")
    ic = a.init_compat(wt, parent_commit=base)
    assert ic.ok is ok, ic.reason


# ============================================================ C4: the eval's provenance
def test_an_eval_dir_written_by_training_is_refused_by_the_job_script(tmp_path):
    base, out = _fake_base(tmp_path), tmp_path / "runs" / "n1" / "r"
    p, log, _ = _run_job(tmp_path, base, out, FAKE_FABRICATE="1")
    assert p.returncode == 0 and "REFUSED" in p.stdout and " brainmarks_smri " not in log
    assert not (out / "eval").exists() and list(out.glob("eval.pre-eval.*"))
    assert [e["event"] for e in events_of(out)][-1] == "eval_fabricated"


def test_a_fabricated_eval_is_an_eval_void(launched):
    a, _, h = launched
    write_eval(a, h)
    add_events(Path(h.output_dir), {"event": "eval_fabricated", "ts": time.time()})
    st = finish(a, h)
    assert st.state is RunStatus.VOID and st.void_type is VoidType.EVAL and st.detail.startswith("eval_fabricated")


@pytest.mark.parametrize("tamper,expect", [
    (lambda t, d: {**d, "config": {"model_kwargs": {"checkpoint": "/elsewhere/checkpoint-last.pth"}}},
     "evaluated checkpoint '/elsewhere"),
    (lambda t, d: {**d, "cwd_git": {"sha": "deadbee"}}, "cwd_git.sha 'deadbee'"),
    (lambda t, d: {**d, "date": "2020-01-01T00:00:00"}, "predates"),
    (lambda t, d: {**d, "task": "ixi_age"} if t == "abide_diagnosis" else d, "identifies"),
    (lambda t, d: {**d, "subset": "test"}, "identifies"),
    # the json's samples changed after aggregate.json was computed: the re-score disagrees with the aggregate
    (lambda t, d: {**d, "eval_samples": [{**s, "dice": 0.99} for s in d["eval_samples"]]} if t == "ixi_age" else d,
     "aggregate.json disagrees with the re-scored task jsons on ['ixi_age']"),
    (lambda t, d: {k: v for k, v in d.items() if k != "eval_samples"}, "cannot be re-scored"),
])
def test_task_jsons_are_validated_and_rescored(launched, tamper, expect):
    a, _, h = launched
    write_eval(a, h, tamper=tamper)
    st = finish(a, h)
    assert st.state is RunStatus.VOID and st.void_type is VoidType.EVAL and expect in st.detail, st.detail


def test_an_aggregate_json_inflated_over_honest_task_jsons_is_refused(launched):
    a, _, h = launched
    write_eval(a, h)
    agg_path = Path(h.output_dir) / "eval" / "aggregate.json"
    agg = json.loads(agg_path.read_text())
    for row in agg["summary"].values():
        row["score"] += 0.05
    agg_path.write_text(json.dumps(agg))
    st = finish(a, h)
    assert st.void_type is VoidType.EVAL and "disagrees with the re-scored task jsons" in st.detail


def test_scores_need_the_job_scripts_eval_start(launched):
    a, _, h = launched
    write_eval(a, h, events=False)
    st = finish(a, h)
    assert st.void_type is VoidType.EVAL and "no eval_start" in st.detail


# ============================================================ H2 / M3: continuations and retries
def test_timeout_after_training_is_an_eval_continuation_twice_then_too_slow(launched):
    a, runner, h = launched
    (Path(h.output_dir) / "summary.json").write_text(json.dumps(SUMMARY))
    add_events(Path(h.output_dir), {"event": "train_start", "ts": time.time() - 3700, "resume": False},
               {"event": "train_end", "ts": time.time() - 100, "rc": 0, "done": True})
    st = finish(a, h, state="TIMEOUT")
    assert st.state is RunStatus.RUNNING and "eval continuation 1" in st.detail
    rec = a._receipt(a._rel_of(h))
    assert rec["eval_continuations"] == 1 and runner.argvs("sbatch")[-1] == rec["argv"]   # same OUT, same argv
    assert finish(a, h, state="TIMEOUT").state is RunStatus.RUNNING
    st = finish(a, h, state="TIMEOUT")
    assert st.state is RunStatus.VOID and st.void_type is VoidType.TOO_SLOW and "3 job(s)" in st.detail


def test_timeout_during_training_stays_too_slow(launched):
    a, _, h = launched
    st = finish(a, h, state="TIMEOUT")
    assert st.void_type is VoidType.TOO_SLOW


def test_exit_75_training_stopped_by_a_signal_is_a_free_retry(launched):
    a, runner, h = launched
    (Path(h.output_dir) / "summary.json").write_text(json.dumps({**SUMMARY, "done": False, "stopped_by_signal": True}))
    add_events(Path(h.output_dir), {"event": "train_start", "ts": 1.0, "resume": False},
               {"event": "train_end", "ts": 2.0, "rc": 0, "done": False, "stopped": True})
    st = finish(a, h, state="FAILED")
    assert st.state is RunStatus.RUNNING and "free retry 1" in st.detail
    rec = a._receipt(a._rel_of(h))
    rec["requeues"] = 10
    a._save(a._rel_of(h), rec)
    st = finish(a, h, state="FAILED")
    assert st.state is RunStatus.VOID and st.void_type is VoidType.INFRA and "free retries spent" in st.detail


def test_a_real_training_crash_is_still_a_crash(launched):
    a, _, h = launched
    add_events(Path(h.output_dir), {"event": "train_start", "ts": 1.0, "resume": False},
               {"event": "train_end", "ts": 2.0, "rc": 1, "done": False, "stopped": False})
    assert finish(a, h, state="FAILED", log="Traceback\nValueError: bad\n").state is RunStatus.CRASH


def test_job_script_forwards_sigterm_so_the_trainer_saves(tmp_path):
    """M3: SIGTERM to the job script (preemption / time limit) reaches the training process, which saves and
    exits 0 with done false; the script records it (stopped) and exits 75."""
    base, out = _fake_base(tmp_path), tmp_path / "runs" / "n1" / "r"
    wt, env = _job_env(tmp_path, FAKE_TRAP="1")
    p = subprocess.Popen(["bash", str(JOB_SCRIPT), "cfg.yaml", str(out), "11", "true", str(base), "full"],
                         cwd=str(tmp_path), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for _ in range(200):
        if (out / "train.started").exists():
            break
        time.sleep(0.05)
    p.send_signal(signal.SIGTERM)
    so, se = p.communicate(timeout=30)
    assert p.returncode == 75, so + se
    assert "SIGTERM, forwarded" in so
    assert json.loads((out / "summary.json").read_text())["stopped_by_signal"] is True
    end = events_of(out)[-1]
    assert end["event"] == "train_end" and end["rc"] == 0 and end["stopped"] is True


def test_job_script_kills_what_the_phase_left_behind(tmp_path):
    base, out = _fake_base(tmp_path), tmp_path / "runs" / "n1" / "r"
    p, _, _ = _run_job(tmp_path, base, out, FAKE_DAEMON="1")
    assert p.returncode == 0
    pid = int((out / "daemon.pid").read_text())
    time.sleep(0.2)
    assert not Path(f"/proc/{pid}").exists() or "Z" in Path(f"/proc/{pid}/stat").read_text().split()[2]


# ============================================================ M4 / M6: heartbeat, compile caches
def test_heartbeat_tracks_cpu_progress_not_mere_liveness(tmp_path):
    base = _fake_base(tmp_path)
    busy = tmp_path / "runs" / "busy"
    p, _, _ = _run_job(tmp_path, base, busy, FAKE_BUSY_S="3", NANOBRAIN_HEARTBEAT_S="1")
    assert p.returncode == 0 and (meta_dir(busy) / "heartbeat").is_file()
    idle = tmp_path / "runs" / "idle"                       # a hung (sleeping / futex-blocked) phase never beats
    p, _, _ = _run_job(tmp_path, base, idle, FAKE_IDLE_S="3", NANOBRAIN_HEARTBEAT_S="1")
    assert p.returncode == 0 and not (meta_dir(idle) / "heartbeat").exists()


def test_adapter_heartbeat_reads_the_job_scripts_beat(launched):
    a, _, h = launched
    out = Path(h.output_dir)
    meta_dir(out).mkdir(parents=True, exist_ok=True)
    (meta_dir(out) / "heartbeat").write_text("1")
    os.utime(meta_dir(out) / "heartbeat", (9e9, 9e9))
    assert a._heartbeat(out, a._receipt(a._rel_of(h)), "777") == 9e9


def test_compile_caches_are_per_job_and_removed(tmp_path):
    base = _fake_base(tmp_path)
    _run_job(tmp_path, base, tmp_path / "runs" / "a")
    _, log, _ = _run_job(tmp_path, base, tmp_path / "runs" / "b")
    caches = re.findall(r"IND=(\S+) GUARD .* nanobrain\.train", log)
    assert len(caches) == 2 and caches[0] != caches[1] and all(c.startswith(str(tmp_path / "tmp")) for c in caches)
    assert not list((tmp_path / "tmp").iterdir())                        # the per-job scratch is gone


# ============================================================ M1: the guard
def test_a_foreign_package_in_the_worktree_src_refuses_the_job(tmp_path):
    base, out = _fake_base(tmp_path), tmp_path / "runs" / "r"
    wt, _ = _job_env(tmp_path)
    (wt / "src" / "brainmarks_smri").mkdir(parents=True)
    p, log, _ = _run_job(tmp_path, base, out)
    assert p.returncode == 2 and "REFUSED: src/brainmarks_smri" in p.stdout and " nanobrain.train " not in log


def _guard() -> str:
    return re.search(r"^GUARD='\n(.*?)^'$", JOB_SCRIPT.read_text(), re.S | re.M).group(1)


@pytest.mark.skipif(not (NB_BASE / ".venv" / "bin" / "python").exists(), reason="nanobrain base venv absent")
def test_real_guard_isolates_the_worktree_from_the_base_src(tmp_path):
    wt = tmp_path / "wt"
    (wt / "src" / "nanobrain").mkdir(parents=True)
    (wt / "src" / "nanobrain" / "probe.py").write_text(
        "import json, sys, nanobrain\n"
        "try:\n    import nanobrain.misc\n    misc = nanobrain.misc.__file__\nexcept ImportError:\n    misc = None\n"
        "print(json.dumps({'path': list(nanobrain.__path__), 'misc': misc, 'argv': sys.argv}))\n")
    (wt / "brainmarks_smri").mkdir()                                     # a planted package in the cwd: dropped
    (wt / "brainmarks_smri" / "__init__.py").write_text("raise SystemExit('planted')\n")
    py = str(NB_BASE / ".venv" / "bin" / "python")
    p = subprocess.run([py, "-I", "-c", _guard(), str(wt), str(NB_BASE), "nanobrain.probe", "x"], cwd=str(wt),
                       capture_output=True, text=True, timeout=120, env={**os.environ, "PYTHONPATH": str(wt)})
    assert p.returncode == 0, p.stdout + p.stderr
    got = json.loads(p.stdout.strip().splitlines()[-1])
    assert got["path"] == [str(wt / "src" / "nanobrain")] and got["misc"] is None     # no base fallback
    assert got["argv"][1:] == ["x"]
    (wt / "src" / "brainmarks_smri").mkdir()
    (wt / "src" / "brainmarks_smri" / "__init__.py").write_text("x = 1\n")
    p = subprocess.run([py, "-I", "-c", _guard(), str(wt), str(NB_BASE), "nanobrain.probe"], cwd=str(wt),
                       capture_output=True, text=True, timeout=120)
    assert p.returncode == 3 and "nanobrain_guard: FAILED brainmarks_smri" in p.stdout


LEADER_PY = Path("/home/user/nanobrain/worktrees/wave1-base/.venv/bin/python")


@pytest.mark.skipif(not LEADER_PY.exists(), reason="leader venv absent")
def test_real_guard_drops_the_editable_src_of_another_checkouts_venv(tmp_path):
    """loop.project.python = a venv whose editable install points at ANOTHER checkout's src (wave1-base): that src
    is dropped too, so nanobrain resolves to the worktree only (live 11-08: every calibration job failed the guard)."""
    wt = tmp_path / "wt"
    (wt / "src" / "nanobrain").mkdir(parents=True)
    (wt / "src" / "nanobrain" / "probe.py").write_text(
        "import json, nanobrain\nprint(json.dumps(list(nanobrain.__path__)))\n")
    venv = LEADER_PY.parent.parent
    p = subprocess.run([str(LEADER_PY), "-I", "-c", _guard(), str(wt), str(NB_BASE), "nanobrain.probe"], cwd=str(wt),
                       capture_output=True, text=True, timeout=120, env={**os.environ, "NB_VENV": str(venv)})
    assert p.returncode == 0, p.stdout + p.stderr
    assert json.loads(p.stdout.strip().splitlines()[-1]) == [str(wt / "src" / "nanobrain")]


# ============================================================ H4 / L3 / L5 / meta
def test_smoke_scales_every_step_counted_key(tmp_path, nb_repo):
    a, _ = make_nb(tmp_path, nb_repo)
    cfg = yaml.safe_load(config_with(total_steps=30000, warmup_steps=9000, decay_steps=6000, checkpoint_steps=[5000, 20000],
                                     mask_anneal_steps=3000, model_kwargs={"depth": 12, "ema_period": 1000}))
    got = a.smoke_config(cfg)
    assert (got["total_steps"], got["warmup_steps"], got["decay_steps"]) == (300, 90, 60)
    assert got["checkpoint_steps"] == [50, 200] and got["mask_anneal_steps"] == 30
    assert got["model_kwargs"]["ema_period"] == 10 and got["checkpoint_period"] == 300 and got["time_limit"] == 3
    # an annealing schedule filling total_steps stays a valid schedule (warmup + decay <= total) after rounding
    got = a.smoke_config(yaml.safe_load(config_with(total_steps=22501, warmup_steps=6751, decay_steps=15750)))
    assert got["warmup_steps"] + got["decay_steps"] <= got["total_steps"] == 300
    assert a.smoke_config(cfg, memory=True)["batch_size"] == cfg["batch_size"]


def test_test_runs_are_pinned_to_the_primary_pool(tmp_path, nb_repo):
    a, runner = make_nb(tmp_path, nb_repo, runner=SlurmRunner(sbatch()))
    a.no_resume = lambda run: run.stage is Stage.CONFIRM
    sha = git(nb_repo, "rev-parse", "HEAD")
    a.submit(run=run_of(stage=Stage.CONFIRM, pool="idle"), commit=sha)
    a.submit(run=run_of(node="n2", pool="idle"), commit=sha)
    envs = [c["env"] for c in runner.calls if c["argv"][0] == "sbatch"]
    assert [e["SBATCH_ACCOUNT"] for e in envs] == ["main", "preempt"]
    assert "SBATCH_REQUEUE" not in envs[0] and "SBATCH_REQUEUE" in envs[1]


def test_a_first_launch_clears_a_leftover_loopmeta(tmp_path, nb_repo):
    a, _ = make_nb(tmp_path, nb_repo, runner=SlurmRunner(sbatch()))
    out = a.runs / "n1" / "screen_s11_a0"
    add_events(out, {"event": "train_start", "ts": 1.0, "resume": True})
    h = a.submit(run=run_of(), commit=git(nb_repo, "rev-parse", "HEAD"))
    assert not meta_dir(out).exists()
    add_events(out, {"event": "train_start", "ts": 2.0, "resume": False})
    a.submit(run=run_of(), commit=git(nb_repo, "rev-parse", "HEAD"))          # idempotent: never cleared again
    assert len(events_of(out)) == 1 and h.output_dir == str(out)


def test_nanopath_resumed_hook_is_the_receipt_count(tmp_path):
    from loop.adapters.nanopath_v2 import NanopathV2Adapter
    from tests.loop.test_nanobrain_adapter import nb_ctx
    a = NanopathV2Adapter(nb_ctx(tmp_path / "s", project=None), runner=SlurmRunner(), repo=str(tmp_path))
    assert a._resumed(tmp_path, {"resumed": 3}) == 3 and a._resumed(tmp_path, {}) == 0


# ============================================================ resume_state (runner idle churn / preempt backoff)
def test_resume_state_reads_the_in_trainer_resume(launched):
    """11-10: the inherited nanopath resume_state (resume config copy / latest.pt) called every nanobrain run "not
    resumable", so the preempt backoff re-ran each 3x-preempted screen from step 0 on primary. nanobrain resumes in
    train.py from OUT/checkpoint-last.pth when the job script got auto_resume=true."""
    a, _, h = launched
    out = Path(h.output_dir)
    st = a.resume_state(h)
    assert st == {"armed": False, "resumed": 0, "fraction": 0.0, "resumable": True}   # not yet saved: never dropped
    (out / "checkpoint-last.pth").write_bytes(b"x")
    (out / "summary.json").write_text(json.dumps({**SUMMARY, "done": False, "stopped_by_signal": True,
                                                  "wall_seconds": 1800.0}))
    add_events(out, {"event": "train_start", "ts": time.time() - 900, "resume": True})
    st = a.resume_state(h)
    assert st["armed"] and st["resumable"] and st["resumed"] == 1 and abs(st["fraction"] - 0.5) < 1e-9
    (out / "summary.json").write_text(json.dumps({**SUMMARY, "done": True}))
    assert a.resume_state(h)["fraction"] == 1.0


def test_resume_state_a_no_resume_run_is_never_armed(tmp_path, nb_repo):
    a, _ = make_nb(tmp_path, nb_repo, runner=SlurmRunner(sbatch("778")))
    a.no_resume = lambda run: True
    h = a.submit(run=run_of(stage=Stage.CONFIRM), commit=git(nb_repo, "rev-parse", "HEAD"))
    out = Path(h.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoint-last.pth").write_bytes(b"x")
    st = a.resume_state(h)
    assert not st["armed"] and not st["resumable"]
