"""loop.adapters.nanobrain (multi-project Stage 1, design §3 / §5 B): the job argv / env per pool, the job script
(PYTHONPATH, base venv, .env, no submit.py, requeue-safe -- run for real against a fake python), TEST runs never
auto-resume, scores from brainmarks' aggregate.json (synthetic 11-task + the REAL 9-task smoke output), the
l-bfgs eval void, the launch-time lock / config guard, the smoke overrides + load check, the init deny list; and
nanopath parity of the class attributes. Fake runner: no SLURM, no GPU, no network."""

from __future__ import annotations

import dataclasses
import datetime
import gzip
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest
import re
import yaml

from kernel.model import TaskSpec
from kernel.stats import NoiseModel
from loop import project as P
from loop.adapters import nanopath_v2 as NP
from loop.adapters.nanobrain import JOB_SCRIPT, LOADCHECK_OK, NanobrainAdapter, meta_dir
from loop.adapters.nanopath_v2 import NanopathV2Adapter, locked_source
from loop.contracts import JobHandle, Run, RunStatus, Stage, SubmissionRejected, VoidType, Worktree
from loop.spec import InitCompatCfg, LoopConfig, LoopContext, PoolCfg, SlurmCfg
from tests.loop.test_adapters import SlurmRunner, git

REPO = Path(__file__).resolve().parents[2]
FIX = REPO / "tests" / "fixtures" / "nanobrain"
NB_BASE = Path("/home/user/nanobrain/base")
AXES = ("nb_cls_mean", "nb_reg_mean", "nb_seg_mean")
CLS = ("abide_diagnosis", "adhd_diagnosis", "cnp_diagnosis", "ucsf_idh", "upenn_survival_1y", "soop_mrs_poor")
REG = ("ixi_age", "openbhb_age")
SEG = ("brats_enhancing_tumor", "brats_whole_tumor", "soop_lesion")
PROJECT = {
    "name": "nanobrain", "base_sha": "440fc92a0f949c5874db64ec6491f397bcbda37c",
    "runs_root": "/data/user/nanobrain/runs", "worktrees_root": "/home/user/nanobrain/worktrees",
    "config_rel": "src/nanobrain/config/default_pretrain.yaml", "aggregation": "weighted",
    "axes": [{"name": "cls", "key": "nb_cls_mean", "weight": 6, "tasks": list(CLS)},
             {"name": "reg", "key": "nb_reg_mean", "weight": 2, "tasks": list(REG)},
             {"name": "seg", "key": "nb_seg_mean", "weight": 3, "tasks": list(SEG)}],
    "target": "cls", "guard": [], "harm_h": {}, "axis_exclude": [],
}
TRAIN_SPANS = ("src/nanobrain/train.py::time_limit = args.time_limit * 60...time_limit = args.time_limit * 60",
               "src/nanobrain/train.py::wall_time = prev_wall_time + time.monotonic() - start_time"
               "...done = step == total_steps or wall_time >= time_limit")
LOCKS = ("scripts/", "pyproject.toml", "uv.lock", "src/nanobrain/config/fomo300k_images.txt",
         "src/nanobrain/select_images.py", *TRAIN_SPANS,
         "src/nanobrain/misc.py::load_model", "src/nanobrain/misc.py::git_info")
PROTECTED = ("time_limit", "train_filelist", "ckpt", "resume", "auto_resume", "output_dir", "name", "seed",
             "device", "debug")
DENY = (r"brainiac|brain[-_]?fm|medicalnet|models?[-_]?genesis|hf_hub_download|from_pretrained|torch\.hub",)
CONFIG_REL = PROJECT["config_rel"]

TRAIN_PY = '''def train(args, start_step, prev_wall_time, start_time, total_steps):
    # time limit on wall time from the start of main, summed over restarts
    time_limit = args.time_limit * 60 if args.time_limit else float("inf")
    step = start_step
    while True:
        step += 1
        wall_time = prev_wall_time + time.monotonic() - start_time
        done = step == total_steps or wall_time >= time_limit
        if done:
            return step
'''
MISC_PY = '''def load_model(args, model_without_ddp, optimizer, loss_scaler):
    start_step, wall_time = 0, 0.0
    return start_step, wall_time


def git_info(path=None):
    return {"sha": "x"}


def backward_step(loss):
    return loss
'''


def full_aggregate(shift: float = 0.0, open_delta: float = 0.0, drop: tuple[str, ...] = ()) -> dict:
    tasks = [t for t in (*CLS, *REG, *SEG) if t not in drop]
    summ = {t: {"score": 0.1 + 0.05 * i + shift, "low": 0.0, "high": 1.0, "n_samples": 100}
            for i, t in enumerate(tasks)}
    if not drop:
        summ["open"] = {"score": sum(summ[t]["score"] for t in tasks) / len(tasks) + open_delta}
    summ["all"] = {"score": sum(summ[t]["score"] for t in tasks) / len(tasks)}
    return {"summary": summ}


@pytest.fixture
def nb_repo(tmp_path):
    repo = tmp_path / "nb"
    cfgdir = repo / "src" / "nanobrain" / "config"
    cfgdir.mkdir(parents=True)
    (repo / "scripts").mkdir()
    (repo / "scripts" / "train.sbatch").write_text("#!/usr/bin/env bash\nuv run python scripts/submit.py x\n")
    (repo / "scripts" / "submit.py").write_text("print('labless')\n")
    (repo / "pyproject.toml").write_text('[project]\nname = "nanobrain"\n'
                                         'brainmarks = { git = "x", rev = "8ef7115" }\n')
    (repo / "uv.lock").write_text("version = 1\n")
    (cfgdir / "default_pretrain.yaml").write_text((FIX / "default_pretrain.yaml").read_text())
    (cfgdir / "fomo300k_images.txt").write_text("a.nii.gz\nb.nii.gz\n")
    (repo / "src" / "nanobrain" / "train.py").write_text(TRAIN_PY)
    (repo / "src" / "nanobrain" / "misc.py").write_text(MISC_PY)
    (repo / "src" / "nanobrain" / "model.py").write_text("class ViTMAE3D:\n    pass\n")
    (repo / "src" / "nanobrain" / "select_images.py").write_text("print(1)\n")
    (repo / ".gitignore").write_text(".venv/\n.env\n*.out\n")
    (repo / ".venv" / "bin").mkdir(parents=True)
    git(repo, "init", "-q", "-b", "main")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "base")
    return repo


def nb_ctx(state: Path, project: dict | None = PROJECT, **cfg_kw) -> LoopContext:
    spec = TaskSpec.load(REPO / "tasks" / "fake_loop.yaml")
    surf = spec.surface.model_copy(update={"locked": LOCKS, "protected_keys": PROTECTED,
                                           "cap_ceilings": {"time_limit": 60}})
    spec = spec.model_copy(update={"surface": surf, "slots": spec.slots.model_copy(update={"score_key": "nb_open"})})
    pools = (PoolCfg("primary", account="main", qos="normal", partition="gpu", slots=8),
             PoolCfg("idle", account="preempt", qos="low", partition="gpu", slots=56, preemptible=True))
    cfg = LoopConfig(execution_adapter="loop.adapters.nanobrain:NanobrainAdapter", axes=AXES,
                     weights={"nb_cls_mean": 6 / 11, "nb_reg_mean": 2 / 11, "nb_seg_mean": 3 / 11},
                     panel_sd=0.01, axis_sd={a: 0.01 for a in AXES},
                     slurm=SlurmCfg(pools=pools, time_limit="01:50:00"),
                     init_compat=InitCompatCfg(deny=DENY),
                     project=P.parse_project(project) if project else None, **cfg_kw)
    return LoopContext(spec=spec, cfg=cfg, noise=NoiseModel.from_spec(spec), task_id="nanobrain",
                       spec_path=str(REPO / "tasks" / "fake_loop.yaml"), state_dir=str(state))


def make_nb(tmp_path, repo, runner=None, **kw):
    runner = runner or SlurmRunner()
    kw.setdefault("clock", lambda: 5000.0)
    a = NanobrainAdapter(nb_ctx(tmp_path / "state"), runner=runner, repo=str(repo), runs_root=str(tmp_path / "runs"),
                         worktrees_root=str(tmp_path / "wts"), **kw)
    a.lock_ref = git(repo, "rev-parse", "HEAD")
    return a, runner


def run_of(node="n1", seed=11, stage=Stage.SCREEN, pool="primary", attempt=0) -> Run:
    return Run(id=Run.make_id(node, stage, seed), task_id="nanobrain", node_id=node, seed=seed, stage=stage,
               status=RunStatus.SUBMITTING, attempt=attempt, pool=pool)


def sbatch(job="777"):
    return {"sbatch": (0, f"Submitted batch job {job}\n", "")}


def commit_change(adapter, repo, name, files: dict[str, str]) -> str:
    wt = adapter.prepare_worktree(parent_commit=git(repo, "rev-parse", "HEAD"), name=name)
    for rel, text in files.items():
        (Path(wt.path) / rel).write_text(text)
    return adapter.commit(wt, message=name)


def config_with(**changes) -> str:
    cfg = yaml.safe_load((FIX / "default_pretrain.yaml").read_text())
    cfg.update(changes)
    return yaml.safe_dump(cfg, sort_keys=False)


# ============================================================ parity (nanopath unchanged)

def test_nanopath_class_attributes_equal_the_former_module_constants(tmp_path):
    A = NanopathV2Adapter
    assert (A.BASE_SHA, A.RUNS_ROOT, A.WORKTREES_ROOT, A.CONFIG_REL) == (NP.BASE_SHA, NP.RUNS_ROOT, NP.WORKTREES_ROOT,
                                                                         NP.CONFIG_REL)
    assert (A.LAUNCHER, A.SMOKE_OVERRIDES, A.FLOPS_RE, A.MEMORY_KEEP) == (NP.LAUNCHER, NP.SMOKE_OVERRIDES, NP.FLOPS_RE,
                                                                          NP.MEMORY_KEEP)
    assert NP.BASE_SHA == "065b87f57b9d1ee1fafe8df9fcdc2b0e4b2b5d8f" and NP.LAUNCHER == "./submit/train_1gpu.sbatch"
    for project in (None, {**PROJECT, "base_sha": "", "runs_root": "", "worktrees_root": "", "config_rel": ""}):
        a = NanopathV2Adapter(nb_ctx(tmp_path / "s", project=project), runner=SlurmRunner(), repo=str(tmp_path))
        assert a.root_commit() == NP.BASE_SHA and a.lock_ref == NP.BASE_SHA and a.config_rel == NP.CONFIG_REL
        assert a.runs.parent == Path(NP.RUNS_ROOT) and a.worktrees.parent == Path(NP.WORKTREES_ROOT)
        assert a._classify(Path("x"), {}, "1", "COMPLETED", 0.0, "", 0.0) is None     # the base split decides


def test_project_block_overrides_base_sha_roots_and_config(tmp_path):
    a = NanobrainAdapter(nb_ctx(tmp_path / "s"), runner=SlurmRunner(), repo=str(tmp_path))
    assert a.root_commit() == PROJECT["base_sha"] == a.lock_ref and a.config_rel == CONFIG_REL
    assert a.runs.parent == Path(PROJECT["runs_root"]) and a.worktrees.parent == Path(PROJECT["worktrees_root"])
    custom = {**PROJECT, "base_sha": "f" * 40, "config_rel": "cfg.yaml"}
    b = NanobrainAdapter(nb_ctx(tmp_path / "t", project=custom), runner=SlurmRunner(), repo=str(tmp_path))
    assert b.root_commit() == "f" * 40 and b.config_rel == "cfg.yaml"
    c = NanobrainAdapter(nb_ctx(tmp_path / "u", project=None), runner=SlurmRunner(), repo=str(tmp_path))
    assert c.root_commit() == NanobrainAdapter.BASE_SHA == PROJECT["base_sha"]   # class fallbacks == the yaml values
    assert c.families() == NanobrainAdapter.DEFAULT_FAMILIES
    assert a.families() == tuple((k, float(w), ts) for k, w, ts in NanobrainAdapter.DEFAULT_FAMILIES)


# ============================================================ submit argv / env

@pytest.mark.parametrize("pool,acct,qos,requeue", [("primary", "main", "normal", False),
                                                   ("idle", "preempt", "low", True)])
def test_submit_argv_and_env_per_pool(tmp_path, nb_repo, pool, acct, qos, requeue):
    a, runner = make_nb(tmp_path, nb_repo, runner=SlurmRunner(sbatch()))
    h = a.submit(run=run_of(pool=pool), commit=git(nb_repo, "rev-parse", "HEAD"))
    call = next(c for c in runner.calls if c["argv"][0] == "sbatch")
    out = a.runs / "n1" / "screen_s11_a0"
    assert call["argv"] == ["sbatch", str(JOB_SCRIPT), CONFIG_REL, str(out), "11", "true", str(nb_repo), "full"]
    assert Path(call["cwd"]) == a.worktrees / "n1" and (Path(call["cwd"]) / "slurm").is_dir()
    env = call["env"]
    assert (env["SBATCH_ACCOUNT"], env["SBATCH_QOS"], env["SBATCH_PARTITION"]) == (acct, qos, "gpu")
    assert env["SBATCH_TIMELIMIT"] == "01:50:00" and env["SBATCH_JOB_NAME"].endswith("n1-screen_s11_a0")
    assert ("SBATCH_REQUEUE" in env) is requeue
    assert h.job_id == "777" and h.output_dir == str(out)
    assert not any("submit.py" in " ".join(c["argv"]) or "train.sbatch" in " ".join(c["argv"]) for c in runner.calls)
    rec = json.loads((a.state / "receipts" / "n1" / "screen_s11_a0.json").read_text())
    assert rec["argv"] == call["argv"] and "resume_cfg" not in rec          # nanobrain resumes in-trainer


def test_test_stage_runs_never_auto_resume(tmp_path, nb_repo):
    a, runner = make_nb(tmp_path, nb_repo, runner=SlurmRunner(sbatch()))
    a.no_resume = lambda run: run.stage is Stage.CONFIRM                  # what tick.py installs for TEST stages
    sha = git(nb_repo, "rev-parse", "HEAD")
    a.submit(run=run_of(stage=Stage.CONFIRM), commit=sha)
    a.submit(run=run_of(node="n2", stage=Stage.SCREEN), commit=sha)
    autos = [c["argv"][5] for c in runner.calls if c["argv"][0] == "sbatch"]
    assert autos == ["false", "true"]
    assert a.oom_ladder(sha) == () and a.probe_smoke_why(Worktree(str(nb_repo), "", sha)) == ""


def test_dry_render_and_command_never_submit(tmp_path, nb_repo):
    a, runner = make_nb(tmp_path, nb_repo)
    r = a.render(run_of(pool="idle"))
    assert r["argv"][:2] == ["sbatch", str(JOB_SCRIPT)] and r["env"]["SBATCH_ACCOUNT"] == "preempt"
    assert "cd " in a.command(run_of()) and not runner.argvs("sbatch")


# ============================================================ the job script

def test_job_script_contract_and_syntax():
    text = JOB_SCRIPT.read_text()
    subprocess.run(["bash", "-n", str(JOB_SCRIPT)], check=True)
    code = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
    assert 'export PYTHONPATH="${WT}/src' in code and 'PY="${NB_PYTHON:-${REPO}/.venv/bin/python}"' in code
    assert 'source "${REPO}/.env"' in code and "export BRAINMARKS_SMRI_DATA" in code
    assert "submit.py" not in code and "uv run" not in code
    assert "run_phase nanobrain.train --config" in code and '"auto_resume=${AUTO_RESUME}"' in code
    assert "run_phase brainmarks_smri vit_mae_3d open" in code and '"output_dir=${OUT}"' in code and "name=eval" in code
    assert 'setsid "${PY}" -I -c "${GUARD}"' in code and "-m nanobrain" not in code   # every python under the guard
    assert "#SBATCH --open-mode=append" in text and "#SBATCH --output=slurm/%j.out" in text


FAKE_PY = r'''#!/usr/bin/env bash
# fake base-venv python: logs each call; "trains" (summary.json) and "evaluates" (eval/aggregate.json). The job script
# calls it as  python -I -c GUARD WT REPO TARGET ARGS...  (and  python -I -c CODE FILE KEY  for the summary flags).
if [[ "$3" == *nanobrain_guard* ]]; then shown="GUARD ${*:4}"; else shown=$(echo "$*" | tr '\n' ' '); fi
echo "CALL PYTHONPATH=$PYTHONPATH DATA=$BRAINMARKS_SMRI_DATA IND=$TORCHINDUCTOR_CACHE_DIR $shown" >> "$LOG"
if [[ "$1" == "-I" && "$2" == "-c" && "$3" == *nanobrain_guard* ]]; then
    target=$6
    shift 6
    if [[ "$target" == "nanobrain.train" ]]; then
        echo "TRAIN-DATA $(ls "$BRAINMARKS_SMRI_DATA" | tr '\n' ' ')" >> "$LOG"
        for a in "$@"; do case "$a" in output_dir=*) out="${a#output_dir=}";; esac; done
        mkdir -p "$out"
        echo "$$" > "$out/train.pid"
        if [[ -n "${FAKE_FABRICATE:-}" ]]; then mkdir -p "$out/eval"; echo '{}' > "$out/eval/abide_diagnosis.json"; fi
        if [[ -n "${FAKE_DAEMON:-}" ]]; then sleep 300 & echo $! > "$out/daemon.pid"; fi
        if [[ -n "${FAKE_BUSY_S:-}" ]]; then timeout "$FAKE_BUSY_S" bash -c 'while :; do :; done' || true; fi
        if [[ -n "${FAKE_IDLE_S:-}" ]]; then sleep "$FAKE_IDLE_S"; fi
        if [[ -n "${FAKE_TRAP:-}" ]]; then
            trap 'echo "{\"done\": false, \"stopped_by_signal\": true, \"step\": 5, \"wall_seconds\": 9}" > "$out/summary.json"; exit 0' TERM
            touch "$out/train.started"
            sleep 30 & wait $!
        fi
        echo "{\"done\": ${TRAIN_DONE:-true}, \"stopped_by_signal\": ${TRAIN_STOPPED:-false}, \"step\": 300, \"wall_seconds\": 60}" > "$out/summary.json"
        echo "fake" > "$out/checkpoint-last.pth"; exit "${TRAIN_RC:-0}"
    fi
    if [[ "$target" == "brainmarks_smri" ]]; then
        for a in "$@"; do case "$a" in output_dir=*) out="${a#output_dir=}";; esac; done
        mkdir -p "$out/eval"; echo '{"summary": {}}' > "$out/eval/aggregate.json"; exit "${EVAL_RC:-0}"
    fi
    if [[ "$target" == *.py ]]; then echo "nanobrain_loadcheck: ok global=(768,)"; exit 0; fi
    exit 9
fi
if [[ "$1" == "-I" && "$2" == "-c" ]]; then shift; exec python3 "$@"; fi
exit 9
'''


def _fake_base(tmp_path) -> Path:
    base = tmp_path / "fakebase"
    (base / ".venv" / "bin").mkdir(parents=True)
    py = base / ".venv" / "bin" / "python"
    py.write_text(FAKE_PY)
    py.chmod(py.stat().st_mode | stat.S_IEXEC)
    (base / ".env").write_text("BRAINMARKS_SMRI_DATA=/data/brainmarks-fake\n")
    return base


def _job_env(tmp_path, **env):
    wt = tmp_path / "wt"
    wt.mkdir(exist_ok=True)
    (tmp_path / "tmp").mkdir(exist_ok=True)
    e = {k: v for k, v in os.environ.items() if k not in ("BRAINMARKS_SMRI_DATA", "PYTHONPATH")}
    e.update(LOG=str(tmp_path / "calls.log"), SLURM_SUBMIT_DIR=str(wt), TMPDIR=str(tmp_path / "tmp"), **env)
    return wt, e


def _run_job(tmp_path, base, out, mode="full", auto="true", **env):
    wt, e = _job_env(tmp_path, **env)
    log = tmp_path / "calls.log"
    p = subprocess.run(["bash", str(JOB_SCRIPT), "cfg.yaml", str(out), "11", auto, str(base), mode],
                       cwd=str(tmp_path), env=e, capture_output=True, text=True, timeout=60)
    return p, (log.read_text() if log.exists() else ""), wt


def events_of(out) -> list[dict]:
    path = meta_dir(Path(out)) / "instances.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []


def test_job_script_runs_train_then_eval_with_the_worktree_on_pythonpath(tmp_path):
    base, out = _fake_base(tmp_path), tmp_path / "runs" / "n1" / "screen_s11_a0"
    p, log, wt = _run_job(tmp_path, base, out)
    assert p.returncode == 0, p.stdout + p.stderr
    train = next(l for l in log.splitlines() if " nanobrain.train " in l)
    assert f"PYTHONPATH={wt}/src " in train and f" {wt} {base} nanobrain.train --config cfg.yaml" in train
    assert f"output_dir={out} name=screen_s11_a0 seed=11 auto_resume=true" in train
    # C2: training sees a shadow data root holding fomo300k only; the eval sees the real root
    shadow = train.split("DATA=", 1)[1].split()[0]
    assert shadow != "/data/brainmarks-fake" and shadow.startswith(str(tmp_path / "tmp")) and shadow.endswith("/data")
    assert next(l for l in log.splitlines() if l.startswith("TRAIN-DATA")).split()[1:] == ["fomo300k"]
    ev = next(l for l in log.splitlines() if " brainmarks_smri " in l)
    assert "DATA=/data/brainmarks-fake " in ev
    assert f"model_kwargs.checkpoint={out}/checkpoint-last.pth output_dir={out} name=eval" in ev
    assert "imports=[nanobrain.model]" in ev and (out / "eval" / "aggregate.json").is_file()
    kinds = [e["event"] for e in events_of(out)]
    assert kinds == ["train_start", "train_end", "train_done", "eval_start", "eval_end"]
    assert events_of(out)[0]["resume"] is False and events_of(out)[1]["done"] is True
    # requeue: the same script again -- the script skips the training it saw done, the eval re-runs
    p2, log2, _ = _run_job(tmp_path, base, out)
    assert p2.returncode == 0 and " nanobrain.train " not in log2.split(log, 1)[-1] and log2.count(" brainmarks_smri ") == 2
    assert [e["event"] for e in events_of(out)][5:] == ["eval_start", "eval_end"]


def test_job_script_no_eval_when_training_stopped_by_a_signal(tmp_path):
    base, out = _fake_base(tmp_path), tmp_path / "o" / "r"
    p, log, _ = _run_job(tmp_path, base, out, TRAIN_DONE="false", TRAIN_STOPPED="true")
    assert p.returncode == 75 and " brainmarks_smri " not in log
    end = events_of(out)[-1]
    assert end["event"] == "train_end" and end["rc"] == 0 and end["stopped"] is True and end["done"] is False
    # a requeued instance resumes (checkpoint-last.pth exists, auto_resume=true): counted as a resumed start
    _run_job(tmp_path, base, out)
    assert [e.get("resume") for e in events_of(out) if e["event"] == "train_start"] == [False, True]


def test_job_script_smoke_mode_load_checks_and_drops_the_checkpoint(tmp_path):
    base, out = _fake_base(tmp_path), tmp_path / "smoke" / "out"
    p, log, _ = _run_job(tmp_path, base, out, mode="smoke", auto="false")
    assert p.returncode == 0 and LOADCHECK_OK in p.stdout
    assert " brainmarks_smri " not in log and not (out / "checkpoint-last.pth").exists()
    assert "loadcheck.py" in log
    assert "auto_resume=false" in log


# ============================================================ scores / poll

@pytest.fixture
def launched(tmp_path, nb_repo):
    a, runner = make_nb(tmp_path, nb_repo, runner=SlurmRunner(sbatch("777")))
    h = a.submit(run=run_of(), commit=git(nb_repo, "rev-parse", "HEAD"))
    Path(h.output_dir).mkdir(parents=True, exist_ok=True)
    return a, runner, h


def add_events(out, *rows) -> None:
    """Append job-script events (dicts) to OUT.loopmeta/instances.jsonl."""
    meta = meta_dir(Path(out))
    meta.mkdir(parents=True, exist_ok=True)
    with (meta / "instances.jsonl").open("a") as f:
        for r in rows:
            f.write(json.dumps({"job": "777", "restart": 0, **r}) + "\n")


def ran(out, *, wall=3500.0, resumes=0, now=None) -> None:
    """The job script's bookkeeping of one honest run: train phases (``1 + resumes`` instances summing ``wall`` s),
    train_done, eval_start (now)."""
    now = float(now if now is not None else int(datetime.datetime.now().timestamp()))
    n = 1 + resumes
    t = now - wall - 60
    for k in range(n):
        add_events(out, {"event": "train_start", "ts": t, "resume": k > 0},
                   {"event": "train_end", "ts": t + wall / n, "rc": 0, "done": k == n - 1, "stopped": k < n - 1})
        t += wall / n + 5
    add_events(out, {"event": "train_done", "ts": now - 30}, {"event": "eval_start", "ts": now - 20})


def task_json(out, commit, task, value, **over) -> dict:
    """A brainmarks task json whose re-scored value is exactly ``value`` (segmentation: the Dice mean)."""
    d = {"model": "vit_mae_3d", "name": "eval", "task": task, "subset": "dev", "type": "segmentation",
         "target": "mask", "config": {"model_kwargs": {"checkpoint": f"{out}/checkpoint-last.pth"}},
         "cwd_git": {"sha": commit[:7], "branch": "x", "dirty": False},
         "date": datetime.datetime.now().isoformat(timespec="seconds"),
         "eval_samples": [{"participant_id": f"p{i}", "session_id": "1", "dice": value} for i in range(4)]}
    d.update(over)
    return d


def write_eval(a, h, *, shift=0.0, drop=(), open_delta=0.0, summary=None, events=True, tamper=None) -> dict:
    """An honest finished run in h.output_dir: summary.json, the job events, 11 task jsons and the aggregate.json
    brainmarks writes from them (``open`` only when all 11 exist). ``tamper(task, json) -> json`` edits a task json
    AFTER the aggregate was computed from the honest one."""
    out = Path(h.output_dir)
    commit = a._receipt(a._rel_of(h))["commit"]
    (out / "summary.json").write_text(json.dumps(summary if summary is not None else SUMMARY))
    if events:
        ran(out)
    (out / "eval").mkdir(exist_ok=True)
    tasks = [t for t in (*CLS, *REG, *SEG) if t not in drop]
    summ = {}
    for i, t in enumerate(tasks):
        v = 0.1 + 0.05 * i + shift
        d = task_json(out, commit, t, v)
        summ[t] = {"score": v, "low": 0.0, "high": 1.0, "n_samples": 4}
        (out / "eval" / f"{t}.json").write_text(json.dumps(tamper(t, d) if tamper else d))
    if not drop:
        summ["open"] = {"score": sum(summ[t]["score"] for t in tasks) / len(tasks) + open_delta}
    summ["all"] = {"score": sum(summ[t]["score"] for t in tasks) / len(tasks)}
    agg = {"summary": summ}
    (out / "eval" / "aggregate.json").write_text(json.dumps(agg))
    return agg


def finish(a, h, state="COMPLETED", log="", jid="777"):
    a.runner.replies = {"squeue": (1, "", "slurm_load_jobs error: Invalid job id specified"),
                        "sacct": (0, f"{state}|3600\n", ""), "sbatch": (0, "Submitted batch job 778\n", "")}
    if log:
        (a.worktrees / "n1" / "slurm" / f"{jid}.out").write_text(log)
    return a.poll(h)


SUMMARY = json.loads((FIX / "summary.json").read_text())
REAL_EVAL = json.loads(gzip.decompress((FIX / "eval_9task.json.gz").read_bytes()))


def test_eleven_task_aggregate_scores_family_means_and_open(launched):
    a, _, h = launched
    agg = write_eval(a, h)
    st = finish(a, h)
    assert st.state is RunStatus.DONE and st.gpu_h == 1.0, st
    s = agg["summary"]
    mean = lambda ts: sum(s[t]["score"] for t in ts) / len(ts)        # noqa: E731
    assert st.scores["nb_cls_mean"] == pytest.approx(mean(CLS)) and st.scores["nb_reg_mean"] == pytest.approx(mean(REG))
    assert st.scores["nb_seg_mean"] == pytest.approx(mean(SEG)) and st.aggregate == pytest.approx(s["open"]["score"])
    assert (6 * st.scores["nb_cls_mean"] + 2 * st.scores["nb_reg_mean"] + 3 * st.scores["nb_seg_mean"]) / 11 \
        == pytest.approx(st.aggregate, abs=1e-9)
    assert st.scores["nb_task_ixi_age"] == pytest.approx(s["ixi_age"]["score"])
    assert len([k for k in st.scores if k.startswith("nb_task_")]) == 11
    assert set(AXES) <= set(st.scores) and st.scores["nb_open"] == st.aggregate
    assert (st.scores["nb_steps"], st.scores["nb_images"]) == (1486.0, 47552.0) and st.scores["nb_wall_s"] > 300
    assert st.resumed == 0


def test_real_nine_task_smoke_aggregate_is_an_incomplete_eval_void(launched):
    a, _, h = launched
    real = json.loads((FIX / "aggregate_9task.json").read_text())
    assert "open" not in real["summary"] and "all" in real["summary"]        # brainmarks: open only when complete
    write_eval(a, h, drop=("brats_whole_tumor", "soop_lesion"))
    (Path(h.output_dir) / "eval" / "aggregate.json").write_text(json.dumps(real))
    st = finish(a, h)
    assert st.state is RunStatus.VOID and st.void_type is VoidType.EVAL
    assert "eval incomplete: 2/11" in st.detail and "brats_whole_tumor" in st.detail and "soop_lesion" in st.detail


def test_real_task_jsons_rescore_to_brainmarks_own_aggregate():
    """C4: the loop's re-implementation of brainmarks' point scores reproduces the REAL aggregate (smoke-r0, 9 tasks:
    6 classification incl. a [False, True] class list, 2 regression, 1 segmentation) to the bit."""
    from loop.adapters.nanobrain import task_score
    real = json.loads((FIX / "aggregate_9task.json").read_text())["summary"]
    assert set(REAL_EVAL) == {k for k in real if k not in ("global", "segmentation", "all")}
    for t, d in REAL_EVAL.items():
        assert task_score(d) == pytest.approx(real[t]["score"], abs=1e-12), t
    assert {d["type"] for d in REAL_EVAL.values()} == {"classification", "regression", "segmentation"}


def test_lbfgs_gets_one_eval_retry_then_one_retrain_then_is_the_verdict(launched, tmp_path, nb_repo):
    """H3: an l-bfgs non-convergence: eval-only retry into the same OUT (no retrain), then -- repeated on that
    checkpoint -- VOID(infra) once (the core's fresh training attempt), then on attempt >= 1 VOID(eval)."""
    a, runner, h = launched
    log = (FIX / "slurm_lbfgs.out").read_text()                 # the real 579012.out excerpt
    write_eval(a, h, drop=("brats_whole_tumor", "soop_lesion"))
    (Path(h.output_dir) / "eval" / "aggregate.json").write_text((FIX / "aggregate_9task.json").read_text())
    st = finish(a, h, state="FAILED", log=log)
    assert st.state is RunStatus.RUNNING and "eval continuation 1 as job 778" in st.detail and "free retry" not in st.detail
    rec = a._receipt(a._rel_of(h))
    assert rec["job_id"] == "778" and rec["lbfgs_evals"] == 1 and rec["argv"] == runner.argvs("sbatch")[-1]
    st = finish(a, h, state="FAILED", log=log, jid="778")                       # repeated on the eval-only retry
    assert st.state is RunStatus.VOID and st.void_type is VoidType.INFRA and "one fresh training attempt" in st.detail
    assert st.detail.startswith("eval_lbfgs: AssertionError: l-bfgs used all 1000 iterations")
    assert "brats_whole_tumor" in st.detail
    # the fresh attempt (a1) repeats it too: the verdict
    h1 = a.submit(run=run_of(attempt=1), commit=git(nb_repo, "rev-parse", "HEAD"))
    Path(h1.output_dir).mkdir(parents=True)
    write_eval(a, h1, drop=("brats_whole_tumor", "soop_lesion"))
    a._save(a._rel_of(h1), {**a._receipt(a._rel_of(h1)), "lbfgs_evals": 1})
    st = finish(a, h1, state="FAILED", log=log, jid=a._receipt(a._rel_of(h1))["job_id"])
    assert st.state is RunStatus.VOID and st.void_type is VoidType.EVAL and "after a fresh training attempt" in st.detail
    notes = [json.loads(l)["verdict"] for l in (a.state / "lbfgs.jsonl").read_text().splitlines()]
    assert notes == ["eval_retry", "retrain", "verdict"]


def test_weighted_family_composite_must_equal_open(launched):
    a, _, h = launched
    write_eval(a, h, open_delta=1e-3)
    st = finish(a, h)
    assert st.state is RunStatus.VOID and st.void_type is VoidType.EVAL and "aggregate.json open" in st.detail
    assert a.eval_report(Path(h.output_dir), 0.0)[0] is None


def test_family_weights_that_do_not_reproduce_open_are_an_eval_void(tmp_path, nb_repo):
    bad = {**PROJECT, "axes": [{**ax, "weight": 1} for ax in PROJECT["axes"]]}
    a = NanobrainAdapter(nb_ctx(tmp_path / "state", project=bad), runner=SlurmRunner(sbatch("777")), repo=str(nb_repo),
                         runs_root=str(tmp_path / "runs"), worktrees_root=str(tmp_path / "wts"), clock=lambda: 5000.0)
    a.lock_ref = git(nb_repo, "rev-parse", "HEAD")
    h = a.submit(run=run_of(), commit=git(nb_repo, "rev-parse", "HEAD"))
    Path(h.output_dir).mkdir(parents=True)
    write_eval(a, h)
    assert "weighted family composite" in a.eval_report(Path(h.output_dir), 0.0)[1]


def test_training_crash_is_a_crash_and_stale_aggregate_is_ignored(launched):
    a, _, h = launched
    st = finish(a, h, state="FAILED", log="Traceback\nRuntimeError: shape mismatch in forward_encoder\n")
    assert st.state is RunStatus.CRASH                                       # no summary.json: training died
    out = Path(h.output_dir)
    write_eval(a, h)
    os.utime(out / "eval" / "aggregate.json", (100.0, 100.0))               # before the submission (ts 5000)
    assert "stale" in a.eval_report(out, h.submitted_ts)[1]


def test_completed_train_without_eval_is_an_eval_void(launched):
    a, _, h = launched
    (Path(h.output_dir) / "summary.json").write_text(json.dumps(SUMMARY))
    st = finish(a, h)
    assert st.state is RunStatus.VOID and st.void_type is VoidType.EVAL and "no eval/aggregate.json" in st.detail


# ============================================================ launch-time lock / config guard

@pytest.mark.parametrize("files,expect", [
    ({CONFIG_REL: config_with(time_limit=90)}, "cap:time_limit=90>60"),
    ({CONFIG_REL: config_with(train_filelist="mine.txt")}, "config:train_filelist"),
    ({CONFIG_REL: config_with(ckpt="/x/brainiac.pth")}, "config:ckpt"),
    ({CONFIG_REL: config_with(auto_resume=False)}, "config:auto_resume"),
    ({"pyproject.toml": '[project]\nname = "nanobrain"\nbrainmarks = { git = "x", rev = "deadbee" }\n'},
     "pyproject.toml"),
    ({"uv.lock": "version = 2\n"}, "uv.lock"),
    ({"src/nanobrain/config/fomo300k_images.txt": "a.nii.gz\n"}, "src/nanobrain/config/fomo300k_images.txt"),
    ({"scripts/train.sbatch": "#!/bin/bash\n"}, "scripts/train.sbatch"),
    ({"src/nanobrain/train.py": TRAIN_PY.replace("args.time_limit * 60", "args.time_limit * 600")},
     "src/nanobrain/train.py::time_limit = args.time_limit * 60"),
    ({"src/nanobrain/train.py": TRAIN_PY.replace("wall_time >= time_limit", "wall_time >= 2 * time_limit")},
     "src/nanobrain/train.py::wall_time = prev_wall_time + time.monotonic() - start_time"),
    ({"src/nanobrain/misc.py": MISC_PY.replace("wall_time = 0, 0.0", "wall_time = 0, -3600.0")},
     "src/nanobrain/misc.py::load_model"),
])
def test_lock_and_config_violations_refuse_the_launch(tmp_path, nb_repo, files, expect):
    a, runner = make_nb(tmp_path, nb_repo, runner=SlurmRunner(sbatch()))
    sha = commit_change(a, nb_repo, "bad", files)
    assert expect in a.lock_violations(sha)
    with pytest.raises(SubmissionRejected):
        a.submit(run=run_of(), commit=sha)
    assert not runner.argvs("sbatch")


def test_free_knobs_pass_the_guard(tmp_path, nb_repo):
    a, _ = make_nb(tmp_path, nb_repo)
    sha = commit_change(a, nb_repo, "ok", {CONFIG_REL: config_with(total_steps=30000, base_lr=7e-4, batch_size=24),
                                           "src/nanobrain/model.py": "class ViTMAE3D:\n    depth = 14\n"})
    assert a.lock_violations(sha) == []


@pytest.mark.skipif(not (NB_BASE / "src" / "nanobrain" / "train.py").is_file(), reason="nanobrain base checkout absent")
def test_lock_spans_resolve_in_the_real_nanobrain_base():
    for span in (*TRAIN_SPANS, "src/nanobrain/misc.py::load_model", "src/nanobrain/misc.py::git_info"):
        src = (NB_BASE / span.split("::", 1)[0]).read_text()
        assert locked_source(src, span) is not None, span
    real = (NB_BASE / CONFIG_REL).read_text()
    assert real == (FIX / "default_pretrain.yaml").read_text() or yaml.safe_load(real)["time_limit"] == 60


@pytest.mark.skipif(not (NB_BASE / "src" / "nanobrain" / "train.py").is_file(), reason="nanobrain base checkout absent")
def test_nanobrain_yaml_locked_spans_resolve_in_the_real_base():
    """package C's tasks/nanobrain.yaml: every ``file::`` lock pins real lines (the right ones) of the base checkout."""
    spec = TaskSpec.load(REPO / "tasks" / "nanobrain.yaml")
    spans = [l for l in spec.surface.locked if "::" in l]
    assert len(spans) >= 4
    for span in spans:
        src = (NB_BASE / span.split("::", 1)[0]).read_text()
        got = locked_source(src, span)
        assert got is not None, span
    train = (NB_BASE / "src/nanobrain/train.py").read_text()
    pinned = "\n".join(locked_source(train, s) for s in spans if s.startswith("src/nanobrain/train.py::"))
    assert 'time_limit = args.time_limit * 60 if args.time_limit else float("inf")' in pinned
    assert "wall_time = prev_wall_time + time.monotonic() - start_time" in pinned
    assert "done = step == total_steps or wall_time >= time_limit" in pinned
    assert spec.slots.score_key == "nb_open" and tuple(spec.slots.vector) == AXES


@pytest.mark.skipif(not (NB_BASE / ".git").exists(), reason="nanobrain base checkout absent")
def test_nanobrain_yaml_root_sha_resolves_locks_and_passes_ceilings():
    """The re-rooted base (loop.project.base_sha, not the checkout's HEAD): every ``file::`` lock resolves AT that sha,
    config_rel is the first writable yaml, and its config is inside every cap ceiling with a wall-clock budget on."""
    raw = yaml.safe_load((REPO / "tasks" / "nanobrain.yaml").read_text())
    proj, surf = raw["loop"]["project"], raw["surface"]
    sha, rel = proj["base_sha"], proj["config_rel"]

    def show(path: str) -> str:
        return subprocess.run(["git", "-C", str(NB_BASE), "show", f"{sha}:{path}"], capture_output=True, text=True,
                              check=True).stdout
    for span in (l for l in surf["locked"] if "::" in l):
        assert locked_source(show(span.split("::", 1)[0]), span) is not None, span
    assert next(w for w in surf["writable"] if w.endswith(".yaml")) == rel
    cfg = yaml.safe_load(show(rel))
    for key, ceiling in surf["cap_ceilings"].items():
        assert cfg[key] is not None and cfg[key] <= ceiling, key
    assert cfg["time_schedule"] is True and cfg["train_filelist"] == "src/nanobrain/config/fomo300k_images_qc.txt"
    # re-root 11-08 PM: the root IS FINO fb (a root replicate run's overrides baked in: no CLI overrides needed)
    f = cfg["fino3d"]
    assert (f["enabled"], f["sex"], f["group"], f["age"], f["weight"]) == (True, 0, 0, 3.0, 0.03)
    assert show("src/nanobrain/fino.py")
    assert Path(proj["python"]).is_file()


# ============================================================ smoke

def test_smoke_applies_overrides_and_requires_the_load_check(tmp_path, nb_repo):
    a, runner = make_nb(tmp_path, nb_repo)
    wt = a.prepare_worktree(parent_commit=git(nb_repo, "rev-parse", "HEAD"), name="s1")
    out = a.state / "smoke" / Path(wt.path).name / "out"
    (out).mkdir(parents=True)
    (out / "summary.json").write_text('{"done": true}')         # a stale earlier smoke: wiped before the launch
    add_events(out, {"event": "train_done", "ts": 1.0})           # ... with its job-script events

    def submitted(argv):
        o = Path(argv[3])
        assert not (o / "summary.json").exists() and not meta_dir(o).exists()
        o.mkdir(parents=True, exist_ok=True)
        (o / "summary.json").write_text(json.dumps({"done": True, "step": 300, "wall_seconds": 120.0}))
        (Path(wt.path) / "slurm" / "900.out").write_text("train ok\nnanobrain_loadcheck: ok global=(768,)\n")
        return 0, "Submitted batch job 900\n", ""
    runner.replies = {"sbatch": submitted, "squeue": (0, "", ""), "sacct": (0, "COMPLETED|0:0\n", "")}
    res = a.smoke(wt)
    assert res.ok, res
    call = next(c for c in runner.calls if c["argv"][0] == "sbatch")
    cfg = yaml.safe_load(Path(call["argv"][2]).read_text())
    for k, v in NanobrainAdapter.SMOKE_OVERRIDES.items():
        assert cfg[k] == v
    assert (cfg["time_limit"], cfg["total_steps"], cfg["warmup_steps"], cfg["checkpoint_period"],
            cfg["num_workers"], cfg["plot_period"]) == (3, 300, 90, 300, 4, 0)   # warmup 6750 x 300/22500
    assert call["argv"][3] == str(out) and call["argv"][5:] == ["false", str(nb_repo), "smoke"]
    env = call["env"]
    assert env["SBATCH_TIMELIMIT"] == "20" and "SBATCH_REQUEUE" not in env and env["HF_HUB_OFFLINE"] == "1"
    tp = json.loads((a.state / "smoke" / Path(wt.path).name / "throughput.json").read_text())
    assert tp["steps_per_s"] == pytest.approx(2.5)
    # memory smoke keeps the point's batch size
    runner.calls.clear()
    assert a.smoke(wt, memory=True).ok
    mem = next(c for c in runner.calls if c["argv"][0] == "sbatch")
    cfg = yaml.safe_load(Path(mem["argv"][2]).read_text())
    assert cfg["batch_size"] == 32 and cfg["total_steps"] == 300
    # L5: its own smoke dir (never the normal smoke's out / throughput)
    assert mem["argv"][3] == str(a.state / "smoke" / f"{Path(wt.path).name}-mem" / "out")
    assert json.loads((a.state / "smoke" / Path(wt.path).name / "throughput.json").read_text())["steps_per_s"] \
        == pytest.approx(2.5)


def test_smoke_fails_without_or_on_a_failed_load_check(tmp_path, nb_repo):
    a, runner = make_nb(tmp_path, nb_repo)
    wt = a.prepare_worktree(parent_commit=git(nb_repo, "rev-parse", "HEAD"), name="s2")
    log = Path(wt.path) / "slurm" / "901.out"

    def sub(text):
        def f(argv):
            log.parent.mkdir(exist_ok=True)
            log.write_text(text)
            return 0, "Submitted batch job 901\n", ""
        return f
    runner.replies = {"sbatch": sub("train ok\n"), "squeue": (0, "", ""), "sacct": (0, "COMPLETED|0:0\n", "")}
    assert "did not report ok" in a.smoke(wt).reason
    runner.replies.update({"sbatch": sub("nanobrain_loadcheck: FAILED KeyError: 'args'\n"),
                           "sacct": (0, "FAILED|1:0\n", "")})
    res = a.smoke(wt)
    assert not res.ok and "load check failed: KeyError" in res.reason
    runner.replies["sbatch"] = sub("nanobrain_job: REFUSED: src/sitecustomize.py is not the nanobrain package\n")
    res = a.smoke(wt)
    assert not res.ok and "job refused the worktree" in res.reason and "sitecustomize" in res.reason


# ============================================================ init compat

def test_init_compat_denies_pretrained_brain_weights_sources(tmp_path, nb_repo):
    a, _ = make_nb(tmp_path, nb_repo)
    base = git(nb_repo, "rev-parse", "HEAD")
    wt = a.prepare_worktree(parent_commit=base, name="i1")
    (Path(wt.path) / "src" / "nanobrain" / "model.py").write_text(
        "import torch\nclass ViTMAE3D:\n    w = torch.hub.load('x/brainiac', 'vit')\n")
    ic = a.init_compat(wt, parent_commit=base)
    assert not ic.ok and ic.rule == "init_illegal" and "torch.hub" in ic.reason
    wt2 = a.prepare_worktree(parent_commit=base, name="i2")
    (Path(wt2.path) / "src" / "nanobrain" / "extra.py").write_text("x = from_pretrained('medicalnet')\n")
    assert not a.init_compat(wt2, parent_commit=base).ok                    # untracked files count
    wt3 = a.prepare_worktree(parent_commit=base, name="i3")
    (Path(wt3.path) / "src" / "nanobrain" / "model.py").write_text("class ViTMAE3D:\n    depth = 14\n")
    assert a.init_compat(wt3, parent_commit=base).ok


def test_handle_roundtrip_find_after_restart(tmp_path, nb_repo):
    a, _ = make_nb(tmp_path, nb_repo, runner=SlurmRunner(sbatch("31")))
    run = run_of()
    h = a.submit(run=run, commit=git(nb_repo, "rev-parse", "HEAD"))
    b, _ = make_nb(tmp_path, nb_repo)
    assert b.find(run) == JobHandle(run.id, "31", h.output_dir, h.submitted_ts)
    assert dataclasses.asdict(b.find(run))["job_id"] == "31"


# ============================================================ anneal_filelist subset check (re-root merge pass)

def _anneal(tmp_path, nb_repo, files):
    a, _ = make_nb(tmp_path, nb_repo)
    return a, commit_change(a, nb_repo, "anneal", files)


@pytest.mark.parametrize("files,expect", [
    ({}, None),                                                                        # unset (null) is fine
    ({CONFIG_REL: config_with(anneal_filelist="src/nanobrain/config/anneal_a.txt"),
      "src/nanobrain/config/anneal_a.txt": "a.nii.gz\n"}, None),                       # a subset of the lists
    ({CONFIG_REL: config_with(anneal_filelist="src/nanobrain/config/anneal_a.txt"),
      "src/nanobrain/config/anneal_a.txt": "a.nii.gz\nevil/ixi_0001.nii.gz\n"}, "1 line(s) outside"),
    ({CONFIG_REL: config_with(anneal_filelist="src/nanobrain/config/missing.txt")}, "does not exist"),
    ({CONFIG_REL: config_with(anneal_filelist="/data/other/list.txt")}, "is not a file under"),
    ({CONFIG_REL: config_with(anneal_filelist="src/nanobrain/config/../x.txt")}, "is not a file under"),
    ({CONFIG_REL: config_with(anneal_filelist="scripts/train.sbatch")}, "is not a file under"),
])
def test_anneal_filelist_must_be_a_subset_of_the_fomo_lists(tmp_path, nb_repo, files, expect):
    a, sha = _anneal(tmp_path, nb_repo, files)
    bad = [v for v in a.lock_violations(sha) if v.startswith("anneal_filelist")]
    if expect is None:
        assert not bad
    else:
        assert len(bad) == 1 and expect in bad[0], bad


def test_anneal_filelist_is_checked_against_either_list(tmp_path, nb_repo):
    """the qc list (leader branch only) joins the universe when the base has it"""
    (nb_repo / "src/nanobrain/config/fomo300k_images_qc.txt").write_text("q.nii.gz\n")
    git(nb_repo, "add", "-A")
    git(nb_repo, "commit", "-qm", "qc list")
    a, sha = _anneal(tmp_path, nb_repo, {CONFIG_REL: config_with(anneal_filelist="src/nanobrain/config/anneal_q.txt"),
                                         "src/nanobrain/config/anneal_q.txt": "q.nii.gz\na.nii.gz\n"})
    assert not [v for v in a.lock_violations(sha) if v.startswith("anneal_filelist")]


# ============================================================ loop.project.python (a re-rooted base with its own venv)

def test_project_python_reaches_the_job_env_and_the_worktree_venv_link(tmp_path, nb_repo):
    alt = tmp_path / "leader" / ".venv"
    (alt / "bin").mkdir(parents=True)
    (alt / "bin" / "python").write_text("")
    a = NanobrainAdapter(nb_ctx(tmp_path / "state", {**PROJECT, "python": str(alt / "bin" / "python")}),
                         runner=SlurmRunner(), repo=str(nb_repo), runs_root=str(tmp_path / "runs"),
                         worktrees_root=str(tmp_path / "wts"), clock=lambda: 5000.0)
    a.lock_ref = git(nb_repo, "rev-parse", "HEAD")
    _, env = a._sbatch("cfg.yaml", tmp_path / "o", "n")
    assert env["NB_PYTHON"] == str(alt / "bin" / "python")
    wt = a.prepare_worktree(parent_commit=a.lock_ref, name="v")
    assert (Path(wt.path) / ".venv").resolve() == alt.resolve()
    # default: no key -> no NB_PYTHON, the worktree .venv is the repo's
    b, _ = make_nb(tmp_path / "b", nb_repo)
    assert "NB_PYTHON" not in b._sbatch("cfg.yaml", tmp_path / "o", "n")[1]
    assert (Path(b.prepare_worktree(parent_commit=b.lock_ref, name="v").path) / ".venv").resolve() == \
        (nb_repo / ".venv").resolve()


def test_job_script_runs_under_nb_python_with_its_own_guard_venv(tmp_path):
    base, out = _fake_base(tmp_path), tmp_path / "runs" / "n1" / "r"
    alt = tmp_path / "leader" / ".venv" / "bin"
    alt.mkdir(parents=True)
    (alt / "python").write_text(FAKE_PY)
    (alt / "python").chmod(0o755)
    (base / ".venv" / "bin" / "python").write_text("#!/bin/sh\nexit 97\n")            # the base venv must NOT be used
    p, log, _ = _run_job(tmp_path, base, out, NB_PYTHON=str(alt / "python"))
    assert p.returncode == 0, p.stdout + p.stderr
    assert " nanobrain.train " in log and (out / "eval" / "aggregate.json").is_file()


def test_project_python_parses_and_unknown_keys_still_fail():
    assert P.parse_project({**PROJECT, "python": "/x/.venv/bin/python"}).python == "/x/.venv/bin/python"
    assert P.parse_project(PROJECT).python == ""


# ============================================================ the yaml's forbid / deny regexes vs the leader's own code

def _nb_rx():
    y = yaml.safe_load((REPO / "tasks" / "nanobrain.yaml").read_text())["loop"]
    from loop.adapters.nanobrain import NB_INIT_DENY
    return ([re.compile(p, re.I) for p in y["forbid"]],
            [re.compile(p, re.I) for p in (*y["init_compat"]["deny"], *NB_INIT_DENY)])


@pytest.mark.parametrize("line", [
    "late_generator = torch.Generator().manual_seed(self.seed + 1)",                  # leader data.py
    "misc.random_seed(args.seed + rank)",                                             # leader train.py
    "threads = [threading.Thread(target=run, args=(k,), daemon=True) for k in range(n)]",   # leader misc.py
    "opt.load_state_dict(sd)", "def load_state_dict(self, state_dict):",              # leader optim.py (Muon)
    "teacher.load_state_dict(student.state_dict())", "import gc; gc.freeze()",
    "from torch.utils.checkpoint import checkpoint", "checkpoint_period: 100000000", "print_freq: 100",
])
def test_regexes_accept_the_leaders_own_code(line):
    forbid, deny = _nb_rx()
    assert not [r.pattern[:40] for r in (*forbid, *deny) if r.search(line)]


@pytest.mark.parametrize("line", [
    "torch.manual_seed(0)", "np.random.seed(1234)", "random.seed(7)", "g = torch.Generator().manual_seed(42)",
    "daemon = os.fork()", "ckpt = torch.load('/x/y.pth')", "model.load_state_dict(sd)", "os.environ['PYTHONHASHSEED']='0'",
])
def test_regexes_still_reject_what_they_are_for(line):
    forbid, deny = _nb_rx()
    assert [r for r in (*forbid, *deny) if r.search(line)]


def test_nanobrain_yaml_cap_ceilings_and_masking_text():
    y = yaml.safe_load((REPO / "tasks" / "nanobrain.yaml").read_text())
    assert y["surface"]["cap_ceilings"] == {"time_limit": 60, "print_freq": 500}
    assert y["loop"]["slurm"]["time_limit"] == "02:10:00"
    assert "80%" not in (REPO / "tasks" / "nanobrain.yaml").read_text()


def _fp_setup(tmp_path, nb_repo, edit):
    from loop.spec import IntakeCfg
    a = NanobrainAdapter(nb_ctx(tmp_path / "state", intake=IntakeCfg(noop_fingerprint=True)), runner=SlurmRunner(),
                         repo=str(nb_repo), runs_root=str(tmp_path / "runs"), worktrees_root=str(tmp_path / "wts"))
    base = git(nb_repo, "rev-parse", "HEAD")
    wt = tmp_path / "wt"
    git(nb_repo, "worktree", "add", "-q", "--detach", str(wt), base)
    edit(wt)
    calls = []

    def fake_run(worktree, config, out, base=""):
        cfg = yaml.safe_load(Path(config).read_text())
        calls.append(cfg)
        lw = (cfg.get("model_kwargs") or {}).get("level_weight", 0.0)
        return {"errors": {}, "lr": [0.0, 1e-4], "loss0": 0.95 + lw, "grad0": 0.29, "param1": 1.0, "loss1": 0.949}
    a._fp_run = fake_run
    return a, Worktree(str(wt), "b", base), calls


def _set(key, value):
    def edit(wt):
        p = wt / "src" / "nanobrain" / "config" / "default_pretrain.yaml"
        cfg = yaml.safe_load(p.read_text())
        set_dotted_local(cfg, key, value)
        p.write_text(yaml.safe_dump(cfg))
    return edit


def set_dotted_local(cfg, key, value):
    *head, last = key.split(".")
    for h in head:
        cfg = cfg.setdefault(h, {})
    cfg[last] = value


def test_noop_fingerprint_bounces_an_inert_config_change_before_any_gpu(tmp_path, nb_repo):
    a, wt, calls = _fp_setup(tmp_path, nb_repo, _set("model_kwargs.hard_gamma", 2.0))
    res = a.smoke(wt)
    assert not res.ok and res.reason.startswith("noop_knob: model_kwargs.hard_gamma") and len(calls) == 2
    assert not a.runner.argvs("sbatch")                        # no GPU job
    v = json.loads((a.state / "fingerprint" / "wt" / "verdict.json").read_text())
    assert v["same"] and v["keys"] == ["model_kwargs.hard_gamma"]


def test_noop_fingerprint_passes_a_real_change_and_skips_blind_keys(tmp_path, nb_repo):
    a, wt, calls = _fp_setup(tmp_path, nb_repo, _set("model_kwargs.level_weight", 0.5))
    assert a.noop_fingerprint(wt, tmp_path / "fp") == ""      # loss moved
    b, wt2, calls2 = _fp_setup(tmp_path / "b", nb_repo, _set("clip_quantiles", [0.01, 0.99]))
    assert b.noop_fingerprint(wt2, tmp_path / "fp2") == "" and calls2 == []   # data-side key: never fingerprinted


def test_noop_fingerprint_skips_code_changes_and_is_off_by_default(tmp_path, nb_repo):
    def code(wt):
        (wt / "src" / "nanobrain" / "model.py").write_text("class ViTMAE3D:\n    x = 1\n")
        _set("model_kwargs.hard_gamma", 2.0)(wt)
    a, wt, calls = _fp_setup(tmp_path, nb_repo, code)
    assert a.noop_fingerprint(wt, tmp_path / "fp") == "" and calls == []
    off = NanobrainAdapter(nb_ctx(tmp_path / "off"), runner=SlurmRunner(), repo=str(nb_repo))
    assert off.noop_fingerprint(wt, tmp_path / "fp3") == ""


def test_fingerprint_comparison_fails_open():
    from loop.adapters import nb_fingerprint as FP
    a = {"lr": [0.0, 1e-4], "loss0": 0.95, "grad0": 0.29, "param1": 1.0, "loss1": 0.94}
    assert FP.same(a, dict(a))[0]
    assert not FP.same(a, {**a, "loss0": 0.95 + 1e-4})[0]
    assert FP.same(a, {**a, "loss0": 0.95 + 1e-9})[0]                    # nondeterminism tolerance
    assert not FP.same({"lr": [1.0]}, {"lr": [1.0]})[0]                  # no forward on either side: not inert
    assert not FP.same(a, {})[0]


# ============================================================ 11-08: smoke under the native root

def test_smoke_never_forces_a_plot_period_under_native(tmp_path, nb_repo):
    """train.py:114 asserts ``not (native and plot_period)``; the smoke forced plot_period 1_000_000, so every smoke of
    the native root (the leader recipe) died at startup rc=1. The smoke's plot period is 0 (off) whatever the config."""
    a, _ = make_nb(tmp_path, nb_repo)
    assert NanobrainAdapter.SMOKE_OVERRIDES["plot_period"] == 0
    for cfg in ({"native": True, "plot_period": 0, "total_steps": 22500}, {"native": False, "plot_period": 5000}):
        assert a.smoke_config(cfg)["plot_period"] == 0


def test_the_step0_nan_throughput_placeholder_is_no_nan_loss():
    from loop.adapters.nanobrain import nonfinite
    step0 = ("[07:21:14] Train:  [         0/1000000000]  eta: 712279 days, 8:18:36  lr: 0.000500  grad: 0.2329 "
             "(0.2329)  loss: 1.3408 (1.3408)  vol/s: nan (nan)  tflop/s: nan (nan)  watts: nan (nan)  time: 61.5409")
    assert not nonfinite(step0) and not nonfinite('{"vol_per_sec": NaN, "tflop_per_sec": nan}')
    assert nonfinite(step0.replace("loss: 1.3408 (1.3408)", "loss: nan (nan)"))
    assert nonfinite(step0.replace("grad: 0.2329", "grad: inf"))


def test_a_smoke_whose_log_has_the_step0_nan_meters_passes(tmp_path, nb_repo):
    """n0001 / n0004 (11-08) trained to done with finite losses and were rejected 'smoke log shows NaN/inf' on the
    step-0 ``vol/s: nan`` meters."""
    a, runner = make_nb(tmp_path, nb_repo)
    wt = a.prepare_worktree(parent_commit=git(nb_repo, "rev-parse", "HEAD"), name="s9")
    line = ("Train:  [0/1000000000]  lr: 0.000500  grad: 0.2329 (0.2329)  loss: 1.3408 (1.3408)  vol/s: nan (nan)  "
            "tflop/s: nan (nan)  watts: nan (nan)")

    def submitted(argv):
        o = Path(argv[3])
        o.mkdir(parents=True, exist_ok=True)
        (o / "summary.json").write_text(json.dumps({"done": True, "step": 300, "wall_seconds": 120.0}))
        (Path(wt.path) / "slurm").mkdir(exist_ok=True)
        (Path(wt.path) / "slurm" / "909.out").write_text(f"{line}\ntrain ok\n{LOADCHECK_OK} global=(768,)\n")
        return 0, "Submitted batch job 909\n", ""
    runner.replies = {"sbatch": submitted, "squeue": (0, "", ""), "sacct": (0, "COMPLETED|0:0\n", "")}
    res = a.smoke(wt)
    assert res.ok, res


GUARD_LOG = ("nanobrain_guard: FAILED nanobrain resolves to [], not only the worktree\n"
             "nanobrain_job: training exited rc=3\n")


def test_guard_failure_on_an_incomplete_checkout_is_infra_not_crash(launched):
    """Live 11-11 n0128: a restart cut its `git worktree add` off before src/ (HEAD valid, locked 'initializing');
    both seeds crashed rc=3 on the job guard and were scored CRASH. src/nanobrain missing while the commit has it
    is INFRA (retried; the submit rebuilds the half-created tree)."""
    a, _, h = launched
    shutil.rmtree(a.worktrees / "n1" / "src")
    st = finish(a, h, state="FAILED", log=GUARD_LOG)
    assert st.state is RunStatus.VOID and st.void_type is VoidType.INFRA and "checkout incomplete" in st.detail


def test_guard_failure_with_the_package_on_disk_or_absent_from_the_commit_stays_a_crash(launched, monkeypatch):
    a, _, h = launched
    assert finish(a, h, state="FAILED", log=GUARD_LOG).state is RunStatus.CRASH         # src/nanobrain is there
    shutil.rmtree(a.worktrees / "n1" / "src")
    real = a._git
    monkeypatch.setattr(a, "_git", lambda path, *args, **kw: (128, "", "missing") if args[:1] == ("cat-file",)
                        else real(path, *args, **kw))
    assert finish(a, h, state="FAILED", log=GUARD_LOG).state is RunStatus.CRASH         # the node's own layout bug
