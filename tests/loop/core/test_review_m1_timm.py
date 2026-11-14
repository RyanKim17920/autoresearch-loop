"""Adversarial-review fixes: the timm weights-allowlist bypass (M1), the loader fetch spy (S1), ported arms on the writable
surface (S3), terminal status sync stamps updated_ts (sync), the identity-port control arm (S7)."""

from __future__ import annotations

import dataclasses
import json
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import yaml

from loop.adapters import initcompat as IC
from loop.core import program as PG
from loop.contracts import NodeStatus

from ..test_struct_tune import GOOD, v2ctx  # noqa: F401
from ..test_review_c4j_locks import _commit, git, xenv  # noqa: F401  (fixture)

ALLOW = (r"^(?:hf-hub:)?(?:timm/)?vit_small_patch16_dinov3\.lvd1689m$",
         r"^(?:hf-hub:)?(?:timm/)?(?:eva02_(?:small|base)_patch14_224\.mim_in22k)$")


def _cfg(model: dict) -> str:
    return yaml.safe_dump({"model": model, "train": {"lr": 0.001}})


def _guard_adapter(xenv, v2ctx):
    ad, _, repo, _ = xenv
    surf = SimpleNamespace(locked=(), protected_keys=tuple(v2ctx.spec.surface.protected_keys), cap_ceilings={})
    ic = dataclasses.replace(ad.ctx.cfg.init_compat, allow=tuple(v2ctx.cfg.init_compat.allow),
                             allow_types=tuple(v2ctx.cfg.init_compat.allow_types))
    ad.ctx = SimpleNamespace(spec=SimpleNamespace(surface=surf, repo=str(repo)),
                             cfg=dataclasses.replace(ad.ctx.cfg, init_compat=ic))
    ad.config_rel, ad.locks = "fake_config.yaml", ()
    git(repo, "checkout", "-q", "main")
    return ad, repo


# ------------------------------------------------------------------------------------------------ M1
def test_timm_name_is_a_protected_key_in_the_task_yaml(v2ctx):
    assert "model.timm_name" in v2ctx.spec.surface.protected_keys


def test_the_timm_plus_timm_name_bypass_is_refused_at_launch_and_the_swap_record_is_commit_bound(xenv, v2ctx):
    ad, repo = _guard_adapter(xenv, v2ctx)
    ref = _commit(repo, {"fake_config.yaml": _cfg({"type": "robust_norm_dinov2_vits14_reg"})}, "R")
    ad.lock_ref = ref
    evil = _commit(repo, {"fake_config.yaml": _cfg({"type": "timm", "timm_name": "vit_huge_patch14_clip_224.laion2b"})}, "evil")
    ok = _commit(repo, {"fake_config.yaml": _cfg({"type": "timm:eva02_small_patch14_224.mim_in22k"})}, "ok")
    bad = ad.config_violations(evil)
    assert "config:model.type" in bad and "config:model.timm_name" in bad       # folded type off-list; the key is absent at BASE
    assert ad.config_violations(ok) == []
    # timm_name alone (type unchanged) is still a protected-key violation
    only = _commit(repo, {"fake_config.yaml": _cfg({"type": "robust_norm_dinov2_vits14_reg", "timm_name": "x"})}, "only")
    assert ad.config_violations(only) == ["config:model.timm_name"]
    # a passed swap check records a type mark only for allow_types types, bound to the sha; bare `timm` is never recorded
    ad.state.mkdir(parents=True, exist_ok=True)
    wt = repo.parent / "wt"
    git(repo, "worktree", "add", "-q", str(wt), evil)
    ad._swap_ok_record(str(wt))
    assert ad._swap_ok_urls() == set()
    git(repo, "worktree", "remove", "--force", str(wt))
    git(repo, "worktree", "add", "-q", str(wt), ok)
    ad._swap_ok_record(str(wt))
    assert ad._swap_ok_urls() == {f"type:timm:eva02_small_patch14_224.mim_in22k@{ok}"}


def test_the_swap_record_does_not_accept_a_type_for_another_commit(xenv, v2ctx):
    ad, repo = _guard_adapter(xenv, v2ctx)
    ad.lock_ref = _commit(repo, {"fake_config.yaml": _cfg({"type": "robust_norm_dinov2_vits14_reg"})}, "R")
    t = _commit(repo, {"fake_config.yaml": _cfg({"type": "openmidnight_vitl16"})}, "t1")
    other = _commit(repo, {"fake_config.yaml": _cfg({"type": "openmidnight_vitl16"}) + "# again\n"}, "t2")
    ad.state.mkdir(parents=True, exist_ok=True)
    ad._swap_ok_path().write_text(json.dumps([f"type:openmidnight_vitl16@{t}"]))
    assert ad.config_violations(t) == [] and ad.config_violations(other) == ["config:model.type"]


def test_initcompat_script_folds_the_timm_type_and_python_helper_agrees():
    assert 'def folded_type' in IC.SCRIPT and 'folded_type(cfg["model"])' in IC.SCRIPT
    assert IC.folded_type({"type": "timm", "timm_name": "eva02_base_patch14_224.mim_in22k"}) == \
        "timm:eva02_base_patch14_224.mim_in22k"
    assert IC.folded_type({"type": "dinov3_vits16"}) == "dinov3_vits16" and IC.folded_type({"type": "timm"}) == "timm"


# ------------------------------------------------------------------------------------------------ S1
def _swap(fetches, allow=ALLOW):
    m = {"child": {"constructed": True, "finite": True, "type": "x", "source": "timm/eva02_small_patch14_224.mim_in22k"},
         "parent": {"type": "y", "source": ""},
         "keys": {"swap": True, "loader": True, "finite": True, "loads": [], "fetches": fetches}}
    return IC.evaluate(m, IC.Limits(allow=allow, deny=(r"path[-_]?fm",)))


def test_the_loader_spy_rejects_a_non_allowlisted_repo_id_and_accepts_allowlisted_ones():
    good = _swap(["timm/eva02_small_patch14_224.mim_in22k"])
    assert good.ok and good.checks["init_legal"]["fetches"] == ["timm/eva02_small_patch14_224.mim_in22k"]
    bad = _swap(["owner/some-pathology-model"])
    assert not bad.ok and "allowlist" in bad.reason and bad.checks["init_legal"]["bad_fetch"] == "owner/some-pathology-model"
    assert not _swap(["timm/vit_small_patch16_dinov3.lvd1689m", "bioptimus/h-optimus-0"]).ok
    assert not _swap(["hf-hub:MahmoodLab/PathFM-x"], allow=()).ok                      # deny wins even with no allowlist


def test_spy_fetches_records_hub_timm_and_torch_hub_calls_then_calls_through(monkeypatch):
    ns = {}
    exec(IC.FETCH_SPY, ns)
    calls = []
    hh = types.ModuleType("huggingface_hub")
    hh.hf_hub_download = lambda repo_id, filename=None, **k: calls.append(("hf", repo_id)) or "p"
    hh.snapshot_download = lambda repo_id, **k: calls.append(("snap", repo_id)) or "p"
    tm = types.ModuleType("timm")
    tm.create_model = lambda name, pretrained=False, **k: calls.append(("timm", name, pretrained))
    monkeypatch.setitem(sys.modules, "huggingface_hub", hh)
    monkeypatch.setitem(sys.modules, "timm", tm)
    mod = types.SimpleNamespace(hf_hub_download=hh.hf_hub_download)                      # `from huggingface_hub import ...`
    orig = (hh.hf_hub_download, tm.create_model)
    seen: list[str] = []
    undo = ns["spy_fetches"](seen, mod)
    mod.hf_hub_download("evil/path-fm", "model.safetensors")
    hh.snapshot_download(repo_id="evil/snap")
    tm.create_model("eva02_base_patch14_224.mim_in22k", pretrained=True)
    tm.create_model("vit_tiny", pretrained=False)                                        # no weights: not a fetch
    assert seen == ["evil/path-fm", "evil/snap", "eva02_base_patch14_224.mim_in22k"] and len(calls) == 4   # called through
    for u in undo:
        u()
    assert (hh.hf_hub_download, tm.create_model) == orig and mod.hf_hub_download is orig[0]
    assert "spy_fetches(fetches, mod)" in IC.SCRIPT and "#FETCH_SPY#" not in IC.SCRIPT


# ------------------------------------------------------------------------------------------------ S3 / S7
from .test_ported_arms import Ad, _arm, _icfg, _inc  # noqa: E402
from loop.core import program as PG  # noqa: E402,F811


def test_ported_problem_refuses_a_code_file_off_the_writable_surface():
    writable = ("train.py", "model.py", "dataloader.py", "configs/main.yaml")
    assert PG.ported_problem(_arm(), writable) == ""
    why = PG.ported_problem(_arm(code_files=["model.py", "probe.py"]), writable)
    assert "probe.py" in why and "writable" in why
    assert PG.arm_problem(_arm(code_files=["submit/launch.py"]), writable) and PG.arm_problem(_arm()) == ""


def test_an_off_surface_arm_is_booked_refused_and_never_built(ctx, store):
    inc, ad = _inc(store), Ad(files={("inc", "evil.py"): "E0", ("base-sha", "evil.py"): "E0", ("ref-sha", "evil.py"): "E1"})
    c = _icfg(ctx, operator_programs=({"id": "op-a", "arms": [_arm(code_files=["model.py", "evil.py"])]},))
    PG.arms(c, store, ad, inc, 1.0)
    rec = store.get_state(PG.K_ARMS)["op-a"]["eva"]
    assert rec.startswith("-:") and "evil.py" in rec and not ad.built


def test_the_identity_port_control_arm_is_in_the_task_yaml_and_builds_a_distinct_commit(v2ctx, store):
    from tests.loop.conftest import example_programs
    progs = {p["id"]: p for p in example_programs()}
    arm = next(a for a in progs["op-arch-core"]["arms"] if a["id"] == "port_id")
    assert arm["lane"] == "ported" and arm["config"] == {"model.type": "robust_norm_dinov2_vits14_reg"}
    assert PG.arm_problem(arm, tuple(v2ctx.spec.surface.writable)) == ""
    inc = _inc(store)
    ad = Ad(files={("inc", "configs/main.yaml"): "model:\n  type: robust_norm_dinov2_vits14_reg\n"})
    ad.files[("inc", "model.py")] = "M0"
    arm = {**arm, "code_ref": "ref-sha", "code_base": "base-sha"}
    nid = PG._build_ported_arm(v2ctx, store, ad, inc, {"id": "op"}, arm, 1.0)
    assert nid.startswith("n") and ad.built and store.get_node(nid).status is NodeStatus.QUEUED
    assert yaml.safe_load(ad.built[0][1]["configs/main.yaml"])["model"]["type"] == "robust_norm_dinov2_vits14_reg"
