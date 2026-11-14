"""Multi-project Labless (package C): ``labless.submit_cmd`` / ``push`` / ``name_into_config`` and tasks/nanobrain.yaml.

Nanopath's argv is byte-identical with the new fields at their defaults; nanobrain pushes the run's commit to the fork
remote BEFORE it submits, puts a <= 20 char name into the run's config.yaml for the submitter and restores the file.
subprocess.run is monkeypatched everywhere: nothing here pushes, submits or launches a job."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from loop import labless as L
from loop.spec import LablessCfg, LoopConfig, load_context

ROOT = Path(__file__).resolve().parents[2]
NB_PY = "${NANOBRAIN_PYTHON:-/path/to/nanobrain/.venv/bin/python}"
SHA = "a" * 40


class Runner:
    """A fake subprocess.run: records every argv, and plays the submitter (writes the receipt) for the submit call."""

    def __init__(self, out: Path, push_rc: int = 0, submit_rc: int = 0, receipt: dict | None = None):
        self.calls: list[dict] = []
        self.out, self.push_rc, self.submit_rc = out, push_rc, submit_rc
        self.receipt = {"submission_id": "abc123def0", "run": {"validation_errors": []}} if receipt is None else receipt

    def __call__(self, argv, **kw):
        cfg_path = self.out / "config.yaml"
        self.calls.append({"argv": list(argv), "kw": kw,
                           "config": cfg_path.read_text() if cfg_path.exists() else None})
        if argv[0] == "git":
            return SimpleNamespace(returncode=self.push_rc, stdout="", stderr="remote: denied" if self.push_rc else "")
        if self.receipt:
            (self.out / "labless_submission.json").write_text(json.dumps(self.receipt))
        return SimpleNamespace(returncode=self.submit_rc, stdout="", stderr="")


def _ctx(labless, repo="/repo"):
    return SimpleNamespace(cfg=SimpleNamespace(labless=labless),
                           spec=SimpleNamespace(repo=repo, surface=SimpleNamespace(writable=["train.py", "configs/main.yaml"])))


def _nb_cfg(**kw):
    base = dict(enabled=True, python=NB_PY, token_file="~/.cache/nanobrain/labless_token.json", timeout_min=30.0,
                submit_cmd=("{python}", "scripts/submit.py", "{out_dir}"), push={"remote": "fork", "ref": "loop/{node}"},
                name_into_config=True)
    return SimpleNamespace(**{**base, **kw})


@pytest.fixture
def out(tmp_path, monkeypatch):
    monkeypatch.delenv("LOOP_FORBID_SLURM", raising=False)
    d = tmp_path / "run"
    d.mkdir()
    (d / "config.yaml").write_text("name: nanobrain\nnotes: null\nbase_lr: 0.0005\n")
    (d / "summary.json").write_text(json.dumps({"git": {"sha": SHA}}))
    return d


def test_nanopath_argv_is_byte_identical(out, monkeypatch):
    r = Runner(out)
    monkeypatch.setattr(L.subprocess, "run", r)
    cfg = LablessCfg(enabled=True)
    assert L.submit_run(_ctx(cfg, "/nano"), str(out), "v3 lr.16 #123", "the note", "the changes") == "abc123def0"
    assert len(r.calls) == 1                                       # no push, nothing else
    expect = ["/nano/.venv/bin/python", "/nano/labless/submit_to_labless.py", f"output_dir={out}",
              "run_name=v3 lr.16 #123", "notes=the note", "changes=the changes", "review_config=configs/main.yaml",
              f"github_token_file={Path('~/.labless_autosubmit.json').expanduser()}"]
    assert r.calls[0]["argv"] == expect
    assert r.calls[0]["kw"]["cwd"] == "/nano" and r.calls[0]["kw"]["env"]["LABLESS_AUTOSUBMIT_FILE"] == ""
    assert (out / "config.yaml").read_text().startswith("name: nanobrain")       # nanopath never touches the config
    r2 = Runner(out)                                               # no `changes` -> no changes= argument
    monkeypatch.setattr(L.subprocess, "run", r2)
    L.submit_run(_ctx(cfg, "/nano"), str(out), "n", "t")
    assert not any(a.startswith("changes=") for a in r2.calls[0]["argv"])


def test_nanobrain_argv_push_then_submit(out, monkeypatch):
    r = Runner(out)
    monkeypatch.setattr(L.subprocess, "run", r)
    assert L.submit_run(_ctx(_nb_cfg(), "/nb"), str(out), "v3 base_lr.001 #12", "note", "ignored", node_id="n0042") \
        == "abc123def0"
    push, submit = r.calls
    assert push["argv"] == ["git", "-C", "/nb", "push", "fork", f"{SHA}:refs/heads/loop/n0042"]     # push FIRST
    assert submit["argv"] == [NB_PY, "scripts/submit.py", str(out)]
    assert submit["kw"]["cwd"] == "/nb"
    assert not any("changes=" in a or "notes=" in a for a in submit["argv"])


def test_push_ref_falls_back_to_the_run_dir_name(out, monkeypatch):
    r = Runner(out)
    monkeypatch.setattr(L.subprocess, "run", r)
    L.submit_run(_ctx(_nb_cfg()), str(out), "n", "t")
    assert r.calls[0]["argv"][-1] == f"{SHA}:refs/heads/loop/run"


def test_a_failed_push_never_submits(out, monkeypatch):
    r = Runner(out, push_rc=1)
    monkeypatch.setattr(L.subprocess, "run", r)
    with pytest.raises(RuntimeError, match="git push fork .*denied"):
        L.submit_run(_ctx(_nb_cfg()), str(out), "n", "t")
    assert [c["argv"][0] for c in r.calls] == ["git"]
    assert (out / "config.yaml").read_text() == "name: nanobrain\nnotes: null\nbase_lr: 0.0005\n"


def test_no_push_remote_means_no_git_call(out, monkeypatch):
    r = Runner(out)
    monkeypatch.setattr(L.subprocess, "run", r)
    L.submit_run(_ctx(_nb_cfg(push={})), str(out), "n", "t")
    assert [c["argv"][0] for c in r.calls] == [NB_PY]


def test_name_is_patched_for_the_submitter_and_restored(out, monkeypatch):
    before = (out / "config.yaml").read_bytes()
    r = Runner(out)
    monkeypatch.setattr(L.subprocess, "run", r)
    long = "[exp] a name that is much longer than twenty characters"
    L.submit_run(_ctx(_nb_cfg()), str(out), long, "why this run")
    seen = yaml.safe_load(r.calls[1]["config"])                     # the config AS THE SUBMITTER saw it
    assert seen["name"] == long[: L.NAME_MAX] and len(seen["name"]) <= 20
    assert seen["notes"].startswith("why this run\n\ncode: ") and seen["base_lr"] == 0.0005
    assert f"loop/run @ {SHA[:12]}" in seen["notes"]                # M5: the pushed ref + commit (+ the fork url)
    assert (out / "config.yaml").read_bytes() == before            # restored byte for byte


def test_config_is_restored_when_the_submitter_fails(out, monkeypatch):
    before = (out / "config.yaml").read_bytes()
    r = Runner(out, submit_rc=1, receipt={})
    monkeypatch.setattr(L.subprocess, "run", r)
    with pytest.raises(RuntimeError, match="exit 1"):
        L.submit_run(_ctx(_nb_cfg()), str(out), "n", "t")
    assert (out / "config.yaml").read_bytes() == before


def test_validation_errors_fail_the_submission(out, monkeypatch):
    monkeypatch.setattr(L.subprocess, "run", Runner(out, receipt={"submission_id": "x", "run": {"validation_errors": ["no open"]}}))
    with pytest.raises(RuntimeError, match="no open"):
        L.submit_run(_ctx(_nb_cfg()), str(out), "n", "t")


def test_name_into_config_off_leaves_the_config_alone(out, monkeypatch):
    r = Runner(out)
    monkeypatch.setattr(L.subprocess, "run", r)
    L.submit_run(_ctx(_nb_cfg(name_into_config=False)), str(out), "n", "t")
    assert r.calls[1]["config"] == "name: nanobrain\nnotes: null\nbase_lr: 0.0005\n"


# ---------------------------------------------------------------------------------------------- the yaml files
@pytest.fixture(scope="module")
def nb_raw():
    return yaml.safe_load((ROOT / "tasks" / "nanobrain.yaml").read_text())


def test_nanobrain_yaml_shape(nb_raw):
    s, loop = nb_raw["slots"], nb_raw["loop"]
    assert nb_raw["name"] == "nanobrain-smri" and nb_raw["repo"] == "${NANOBRAIN_REPO:-/path/to/nanobrain}"
    assert s["vector"] == ["nb_cls_mean", "nb_reg_mean", "nb_seg_mean"]
    w = s["weights"]
    assert w["nb_cls_mean"] / w["nb_reg_mean"] == pytest.approx(3.0) and w["nb_seg_mean"] / w["nb_reg_mean"] == pytest.approx(1.5)
    assert [a["key"] for a in loop["project"]["axes"]] == s["vector"]
    assert [a["weight"] for a in loop["project"]["axes"]] == [6, 2, 3]
    assert nb_raw["surface"]["cap_ceilings"]["time_limit"] == 60
    assert {"time_limit", "train_filelist", "ckpt", "resume", "auto_resume"} <= set(nb_raw["surface"]["protected_keys"])
    assert loop["execution_adapter"] == "loop.adapters.nanobrain:NanobrainAdapter"
    assert [p["slots"] for p in loop["slurm"]["pools"]] == [8, 56] and loop["slurm"]["time_limit"] == "02:10:00"
    assert loop["slurm"]["resume"] == "off" and loop["slurm"]["smoke_probe"] == "off"
    assert len(loop["seeds"]["screen"]) + 3 <= loop["seeds"]["max_seeds_per_node"]       # screen 2 + confirm 3 = the 5-seed cap
    assert all(isinstance(v, float) and v > 0 for v in (nb_raw["noise"]["seed_transfer_sd"], loop["panel_sd"]))  # measured by loop calibrate 2026-11-08


def test_nanobrain_yaml_labless_and_disabled_features(nb_raw):
    lb, loop = nb_raw["loop"]["labless"], nb_raw["loop"]
    assert lb["enabled"] is True and lb["exploratory"] is False and lb["leader_median"] == 0.5471
    assert (lb["promote_margin"], lb["min_test_seeds"], lb["max_per_day"]) == (0.005, 4, 2)
    assert lb["submit_cmd"] == ["{python}", "scripts/submit.py", "{out_dir}"] and lb["python"] == NB_PY
    assert lb["push"]["remote"] == "fork" and lb["name_into_config"] is True
    assert lb["token_file"] == "~/.cache/nanobrain/labless_token.json"
    for sec in ("evalonly", "diagnostics", "databank", "prune", "holdout", "extrapolate"):
        assert loop[sec] == {"enabled": False}, sec
    assert loop["factorial"] == {"flag_knobs": {}} and loop["ideation"]["diagnose_precompute"] is False
    assert all(b.startswith("/home/user/nanobrain/idea_banks/") for b in loop["idea_banks"])   # 11-08 banks
    assert loop["data_bank"] == [] and loop["pairs"]["axis_exclude"] == []
    assert any("neuroimaging domain" == p for p in loop["ideation"]["personas"])
    assert not any("pathology" in p for p in loop["ideation"]["personas"])


def test_nanobrain_deny_list_blocks_brain_weights_and_passes_scratch(nb_raw):
    import re
    deny = re.compile("|".join(nb_raw["loop"]["init_compat"]["deny"]), re.I)
    for bad in ("BrainIAC", "load sam-med3d", "swin_unetr pretrain", "hf_hub_download(x)", "from_pretrained", "torch.hub.load"):
        assert deny.search(bad), bad
    for ok in ("self.blocks = nn.ModuleList()", "rope = Rope3D(dim)", "optimizer = Muon(params)"):
        assert not deny.search(ok), ok


def _xfail_unless_a_landed():
    if not {"project", "domain"} <= {f.name for f in dataclasses.fields(LoopConfig)}:
        pytest.xfail("package A not merged: LoopConfig has no `project` / `domain` fields yet")
    if not {"submit_cmd", "push", "name_into_config"} <= {f.name for f in dataclasses.fields(LablessCfg)}:
        pytest.xfail("package A has not applied pkgC-spec-hunk.md (LablessCfg.submit_cmd / push / name_into_config)")


needs_a = pytest.mark.skipif("project" not in {f.name for f in dataclasses.fields(LoopConfig)},
                             reason="package A (loop.spec: loop.project / loop.domain on LoopConfig) not merged yet")


def test_nanobrain_yaml_loads(tmp_path):
    ctx = load_context(ROOT / "tasks" / "nanobrain.yaml", state_dir=tmp_path, require_measured=False)
    assert ctx.cfg.axes == ("nb_cls_mean", "nb_reg_mean", "nb_seg_mean") and ctx.cfg.capacity == 64
    assert ctx.cfg.labless.leader_median == 0.5471 and ctx.cfg.labless.max_per_day == 2


def test_labless_cfg_carries_the_new_fields(tmp_path):
    lc = load_context(ROOT / "tasks" / "nanobrain.yaml", state_dir=tmp_path, require_measured=False).cfg.labless
    assert tuple(lc.submit_cmd) == ("{python}", "scripts/submit.py", "{out_dir}") and lc.push["remote"] == "fork"
    assert lc.name_into_config is True
    assert LablessCfg().submit_cmd == () and not LablessCfg().push and LablessCfg().name_into_config is False


def test_nanopath_v2_project_block_equals_the_defaults(tmp_path):
    from kernel.model import load_yaml
    raw = load_yaml(ROOT / "tasks" / "nanopath_v2.yaml")       # ${VAR:-default} expanded, as the loop reads it
    pj = raw["loop"]["project"]
    from loop import project as P
    from loop.adapters import nanopath_v2 as V2
    P.reset()
    assert pj["name"] == "nanopath" and pj["base_sha"] == V2.BASE_SHA and pj["runs_root"] == V2.RUNS_ROOT
    assert pj["worktrees_root"] == V2.WORKTREES_ROOT and pj["config_rel"] == V2.CONFIG_REL
    assert [a["name"] for a in pj["axes"]] == list(P.AXES) and [a["key"] for a in pj["axes"]] == raw["slots"]["vector"]
    assert {a["name"]: a["weight"] for a in pj["axes"]} == dict(P.OFFICIAL_WEIGHTS)
    assert {a["name"]: a["key"] for a in pj["axes"]} == dict(P.PROBE_KEYS)
    assert {a["name"]: a["label"] for a in pj["axes"]} == dict(P.AXIS_NAMES)
    assert pj["target"] == P.target() and pj["guard"] == list(P.GUARD_AXES) and pj["harm_h"] == dict(P.HARM_H)
    assert tuple(pj["axis_exclude"]) == P.axis_exclude()
    P.install(pj)                                        # the explicit block installs to exactly the defaults
    try:
        assert (list(P.AXES), dict(P.PROBE_KEYS), dict(P.OFFICIAL_WEIGHTS), dict(P.HARM_H), P.target(), P.guard()) == \
            (["cls", "seg", "slide", "auc", "surv", "rob"], {a["name"]: a["key"] for a in pj["axes"]},
             {a["name"]: a["weight"] for a in pj["axes"]}, {"rob": 0.008}, "cls", "rob")
    finally:
        P.reset()
    if "project" not in {f.name for f in dataclasses.fields(LoopConfig)}:
        pytest.xfail("package A not merged: the strict loader cannot read loop.project yet")
    ctx = load_context(ROOT / "tasks" / "nanopath_v2.yaml", state_dir=tmp_path, require_measured=False)
    assert ctx.cfg.axes == tuple(raw["slots"]["vector"])
