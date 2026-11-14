"""The probe leg of the nanopath smoke (slurm.smoke_probe; loop.adapters.probe_smoke): a readout that crashes in the
locked probe.py fails the smoke instead of a full run (live n2534: linalg.eigh in probe_features after 2.4 GPU-h)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import loop.adapters.nanopath_v2 as NP
from loop.adapters.nanopath_v2 import LAUNCHER, PROBE_SAVE_EVERY, PROBE_SMOKE_MIN, SMOKE_MIN
from loop.design import CRASHED
from loop.spec import FamilyRule
from tests.loop.test_adapters import (SlurmRunner, base_repo, make_np, np_ctx_with_locks,  # noqa: F401 (fixture)
                                      research_wt)

FIT = "train.py::# Fit scanner-response directions...model.ct_ms.copy_"
FAMS = {"eval_readout": FamilyRule(paths=("model.py::ViT.forward", FIT), combine=True),
        "optim": FamilyRule(paths=("train.py",))}
BASE_PY = Path("/home/user/nanopath/base/.venv/bin/python")
LAUNCHER_SH = """#!/usr/bin/env bash
#SBATCH --gpus-per-task=1
#SBATCH --output=slurm/%j.out
for a in "$@"; do case "$a" in output_dir=*) d="${a#output_dir=}"; mkdir -p "$d"; touch "$d/latest.pt";; esac; done
echo trained
exit "${FAKE_TRAIN_RC:-0}"
"""


def set_mode(adapter, mode):
    object.__setattr__(adapter.ctx.cfg.slurm, "smoke_probe", mode)


@pytest.fixture
def np_probe(tmp_path, base_repo):  # noqa: F811
    runner = SlurmRunner({"sbatch": (0, "Submitted batch job 61\n", ""), "sacct": (0, "COMPLETED|0:0\n", "")})
    adapter, _ = make_np(np_ctx_with_locks(tmp_path / "pp", families=FAMS), base_repo, runner=runner)
    return adapter, runner, base_repo


def edit(wt, rel, old, new):
    p = Path(wt.path) / rel
    text = p.read_text()
    assert old in text
    p.write_text(text.replace(old, new))


def test_gate_readout_mode_fires_only_on_probe_relevant_changes(np_probe):
    adapter, _, base = np_probe
    wt = research_wt(adapter, base)
    assert adapter.probe_smoke_why(wt) == ""                                   # no change: no probe leg
    edit(wt, "train.py", "lr = 0.1", "lr = 0.2")                               # training code outside the fit span
    assert adapter.probe_smoke_why(wt) == ""
    edit(wt, "train.py", "directions = fit(lr)", "directions = fit(lr) * 2")   # the post-training readout fit
    assert "Fit scanner-response" in adapter.probe_smoke_why(wt)
    wt = research_wt(adapter, base)
    edit(wt, "model.py", "return x * 2", "return x * 3")
    assert adapter.probe_smoke_why(wt) == "model.py::ViT.forward changed"   # an eval_readout span (c3j: not every model.py edit)
    wt = research_wt(adapter, base)
    cfg = Path(wt.path) / "configs/main.yaml"
    tree = yaml.safe_load(cfg.read_text())
    tree["train"]["lr"] = 3e-4
    cfg.write_text(yaml.safe_dump(tree))
    assert adapter.probe_smoke_why(wt) == ""                                   # a training key
    tree["model"]["type"] = "vit_b"
    cfg.write_text(yaml.safe_dump(tree))
    assert adapter.probe_smoke_why(wt) == ""                                   # generic model.* key (c3j)
    tree["model"]["type"] = "vit"
    tree["probe"] = {"model_weights": "model"}
    cfg.write_text(yaml.safe_dump(tree))
    assert adapter.probe_smoke_why(wt) == "config probe.model_weights changed"
    git_commit = subprocess.run(["git", "-C", wt.path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam",
                                 "c"], capture_output=True, text=True)
    assert git_commit.returncode == 0 and adapter.probe_smoke_why(wt) != ""   # committed changes count too


def test_gate_modes_off_always_and_unknown(np_probe):
    adapter, _, base = np_probe
    wt = research_wt(adapter, base)
    assert getattr(adapter.ctx.cfg.slurm, "smoke_probe", "readout") == "readout"   # nanopath's mode (np_ctx_with_locks)
    edit(wt, "model.py", "return x * 2", "return x * 3")
    set_mode(adapter, "off")
    assert adapter.probe_smoke_why(wt) == ""
    set_mode(adapter, "always")
    assert adapter.probe_smoke_why(research_wt(adapter, base, "clean")) == "smoke_probe=always"
    set_mode(adapter, "bogus")                                                 # unknown -> the default (readout)
    assert adapter.probe_smoke_why(wt) == "model.py::ViT.forward changed"


def test_smoke_runs_the_probe_leg_in_the_same_job_when_the_readout_changed(np_probe):
    adapter, runner, base = np_probe
    wt = research_wt(adapter, base)
    edit(wt, "model.py", "return x * 2", "return x * 3")
    (Path(wt.path) / "submit").mkdir()
    (Path(wt.path) / LAUNCHER).write_text(LAUNCHER_SH)
    (Path(wt.path) / "slurm").mkdir()
    (Path(wt.path) / "slurm" / "61.out").write_text("trained\nprobe_smoke: ok, 15 readout calls in 70s\n")
    res = adapter.smoke(wt)
    assert res.ok, res.reason
    call = next(c for c in runner.calls if c["argv"][0] == "sbatch")
    wrapper, cfg_path = Path(call["argv"][1]), Path(call["argv"][2])
    assert call["cwd"] == wt.path and call["argv"][3].startswith("output_dir=")
    text = wrapper.read_text()
    assert "#SBATCH --gpus-per-task=1" in text and f"bash {LAUNCHER}" in text      # the launcher's header, in-job
    assert (wrapper.parent / "probe_smoke.py").is_file() and "probe_smoke.py" in text
    assert yaml.safe_load(cfg_path.read_text())["train"]["save_every"] == PROBE_SAVE_EVERY
    assert yaml.safe_load(cfg_path.read_text())["probe"]["enabled"] is False     # never the full probe
    assert int(call["env"]["SBATCH_TIMELIMIT"]) == SMOKE_MIN + PROBE_SMOKE_MIN
    (Path(wt.path) / "slurm" / "61.out").write_text("trained\n")                  # the leg never reported
    res = adapter.smoke(wt)
    assert not res.ok and "did not report ok" in res.reason


def test_a_probe_crash_fails_the_smoke_with_its_traceback(np_probe):
    adapter, runner, base = np_probe
    runner.replies["sacct"] = (0, "FAILED|1:0\n", "")
    wt = research_wt(adapter, base)
    edit(wt, "model.py", "return x * 2", "return x * 3")
    (Path(wt.path) / "slurm").mkdir()
    (Path(wt.path) / "slurm" / "61.out").write_text("trained\nprobe_smoke: start\n")
    (Path(wt.path) / "slurm" / "61.err").write_text(
        'Traceback (most recent call last):\n  File "model.py", line 284, in _cc_descriptor\n'
        "torch._C._LinAlgError: linalg.eigh: (Batch element 0): The algorithm failed to converge\n"
        "probe_smoke: FAILED _LinAlgError: linalg.eigh: (Batch element 0): The algorithm failed to converge\n")
    res = adapter.smoke(wt)
    assert not res.ok and res.reason.startswith("probe smoke failed (model.py::ViT.forward changed): _LinAlgError: linalg.eigh")
    assert CRASHED.search(res.log_tail) and not res.unavailable                 # design.smoke_done: a crash


def test_no_readout_change_keeps_the_plain_launcher_smoke(np_probe):
    adapter, runner, base = np_probe
    wt = research_wt(adapter, base)
    edit(wt, "train.py", "lr = 0.1", "lr = 0.2")
    assert adapter.smoke(wt).ok
    assert not [c for c in runner.calls if c["argv"][0] == "sbatch"]
    call = next(c for c in runner.calls if c["argv"][0] == LAUNCHER)
    assert int(call["env"]["SBATCH_TIMELIMIT"]) == SMOKE_MIN
    assert yaml.safe_load(Path(call["argv"][1]).read_text())["train"]["save_every"] is None


@pytest.mark.parametrize("train_rc,probe_rc,want", [(0, 0, 0), (0, 1, 1), (3, 0, 3)])
def test_the_wrapper_script_trains_then_probes_and_removes_latest_pt(np_probe, train_rc, probe_rc, want):
    """The generated sbatch script itself, under bash: the launcher in-job, then the probe leg on latest.pt;
    the job's exit code is the verdict; latest.pt never outlives the job."""
    adapter, _, base = np_probe
    wt = research_wt(adapter, base)
    (Path(wt.path) / "submit").mkdir()
    (Path(wt.path) / LAUNCHER).write_text(LAUNCHER_SH)
    venv = Path(wt.path) / ".venv" / "bin"
    venv.mkdir(parents=True, exist_ok=True)
    (venv / "python").write_text(f'#!/usr/bin/env bash\necho "probe_smoke: args $*"\n'
                                 f'test -f "$4" || exit 9\nexit {probe_rc}\n')
    (venv / "python").chmod(0o755)
    smoke_dir = Path(adapter.state) / "smoke" / "w"
    smoke_dir.mkdir(parents=True)
    wrapper = adapter._probe_wrapper(wt, smoke_dir, smoke_dir / "smoke.yaml")
    cp = subprocess.run(["bash", str(wrapper), str(smoke_dir / "smoke.yaml"), f"output_dir={smoke_dir / 'out'}"],
                        cwd=wt.path, capture_output=True, text=True,
                        env={**os.environ, "SLURM_SUBMIT_DIR": wt.path, "FAKE_TRAIN_RC": str(train_rc)})
    assert cp.returncode == want, cp.stdout + cp.stderr
    assert ("probe_smoke: args" in cp.stdout) is (train_rc == 0)                # no probe leg after a failed train
    assert not (smoke_dir / "out" / "latest.pt").exists() or train_rc != 0


# ------------------------------------------------- the probe leg itself (needs torch: the nanopath venv)
FAKE_PROBE = '''
import torch
THUNDER_PROBE_SEED = 0
DATASET_ROOTS = {}
THUNDER_V2 = {"classification": {"ds": {"val": {"images": list(range(100)), "labels": [0] * 100}}},
              "segmentation": {}}
def worker_probe_transforms(cfg):
    return None, None
class ClassificationDataset:
    def __init__(self, dataset, split, transform):
        self.n = len(THUNDER_V2["classification"][dataset][split]["labels"])
    def __len__(self):
        return self.n
    def __getitem__(self, i):
        return torch.zeros(3, 8, 8), 0
def embed_classification_dataset(model, mean, std, dataset, split, device, transform):
    ds = ClassificationDataset(dataset, split, transform)
    x = torch.stack([ds[i][0] for i in range(len(ds))])
    return model.probe_features((x - mean) / std).numpy(), None
'''
FAKE_MODEL = '''
import torch
class ViT(torch.nn.Module):
    def __init__(self, variant=""):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(1))
    def probe_features(self, x):
        {body}
    def forward(self, x):
        return {{"cls": x.flatten(1)[:, :4], "patches": x.flatten(2).transpose(1, 2)}}
'''


@pytest.mark.skipif(not BASE_PY.is_file(), reason="needs the nanopath venv (torch)")
@pytest.mark.parametrize("body,ok", [("return x.flatten(1)[:, :6]", True),
                                     ("raise RuntimeError('readout boom')", False),
                                     ("return x.flatten(1)[:, :6] / 0 * 0", False)])   # NaN embeddings
def test_probe_leg_drives_the_readout_through_probe_entry_points(tmp_path, body, ok):
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / "probe.py").write_text(FAKE_PROBE)
    (wt / "model.py").write_text(FAKE_MODEL.format(body=body))
    cfg = {"probe": {"dataset_roots": {}, "model_weights": "ema", "datasets": ["ds"], "segmentation_datasets": []},
           "model": {"type": "t"}, "data": {"mean": [0.5] * 3, "std": [0.2] * 3}}
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
    script = Path(NP.__file__).with_name("probe_smoke.py")
    mk = ("import sys, torch; sys.path.insert(0, sys.argv[1]); from model import ViT; sd = ViT().state_dict(); "
          "torch.save({'model': sd, 'model_ema': sd}, sys.argv[2])")
    subprocess.run([str(BASE_PY), "-c", mk, str(wt), str(tmp_path / "l.pt")], check=True, capture_output=True)
    cp = subprocess.run([str(BASE_PY), str(script), str(wt), str(tmp_path / "c.yaml"), str(tmp_path / "l.pt"),
                         "--device=cpu"], capture_output=True, text=True, timeout=300)
    assert (cp.returncode == 0) is ok, cp.stdout + cp.stderr
    if ok:
        assert "probe_smoke: ok" in cp.stdout and "probe_features ds: (64, 6)" in cp.stdout   # bounded to N_TILES
    else:
        assert "probe_smoke: FAILED" in cp.stderr and "Traceback" in cp.stderr
