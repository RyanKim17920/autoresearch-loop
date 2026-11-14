"""Eval-only readout runs (loop.evalonly) and checkpoint pruning (loop.prune): eligibility, graft and launch
plumbing on real git with a fake runner; the null control, the lifecycle and the keep-set on a real store."""

from __future__ import annotations

import dataclasses
import os
import time
from pathlib import Path

import pytest
import yaml

from loop import evalonly as EO
from loop import prune as PR
from loop.contracts import (PANEL, ROOT_NODE_ID, JobHandle, Node, NodeStatus, Run, RunStatus, Stage,
                            Strategy, VoidType)
from loop.driver import RunBook
from loop.spec import EvalOnlyCfg, FamilyRule, PruneCfg
from loop.world import ComputedWorldModel
from tests.loop.test_adapters import (LAUNCHER, SlurmRunner, git, make_np, np_ctx_with_locks, sbatch_ok)

MODEL = '''import torch


class ViT:
    def __init__(self):
        self.register_buffer("rn_mu", torch.zeros(3))

    def forward(self, x):
        return x * 2

    def build_blocks(self):
        return 1
'''
TRAIN = '''def main():
    lr = 0.1
    # Fit scanner-response directions after optimization.
    directions = fit(lr)
    model.ct_ms.copy_(directions)
    return lr
'''
FAMS = {"eval_readout": FamilyRule(paths=("model.py::ViT.forward",
                                          "train.py::# Fit scanner-response directions...model.ct_ms.copy_"),
                                   combine=True),
        "architecture": FamilyRule(paths=("model.py",))}


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "base"
    (r / "configs").mkdir(parents=True)
    (r / "configs" / "main.yaml").write_text(yaml.safe_dump(
        {"train": {"seed": 1, "lr": 1e-4, "resume": None, "save_every": 1000}, "model": {"type": "vit"}}))
    (r / "model.py").write_text(MODEL)
    (r / "train.py").write_text(TRAIN)
    (r / ".gitignore").write_text(".venv/\nslurm/\n")
    (r / ".venv" / "bin").mkdir(parents=True)
    git(r, "init", "-q", "-b", "main")
    git(r, "add", "-A")
    git(r, "commit", "-qm", "base")
    return r


def commit_edit(repo, name, **files):
    base = git(repo, "rev-parse", "main")
    git(repo, "checkout", "-q", "-b", name, base)
    for rel, text in files.items():
        (repo / rel.replace("__", "/")).write_text(text)
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", name)
    sha = git(repo, "rev-parse", "HEAD")
    git(repo, "checkout", "-q", "main")
    return base, sha


def np_adapter(tmp_path, repo, runner=None, enabled=True):
    ctx = np_ctx_with_locks(tmp_path / "state", ("probe.py",), families=FAMS,
                            evalonly=EvalOnlyCfg(enabled=enabled))
    return make_np(ctx, repo, runner=runner)


READOUT = MODEL.replace("return x * 2", "return (x - x.mean()) * 2")
BUFFER = READOUT.replace('torch.zeros(3))\n', 'torch.zeros(3))\n        self.register_buffer("pf_zmu", torch.zeros(3))\n')


@pytest.mark.parametrize("files,ok,shim", [
    ({"model.py": READOUT}, True, False),
    ({"model.py": BUFFER}, True, True),                                       # an added fitted buffer: a shim
    ({"train.py": TRAIN.replace("fit(lr)", "fit(lr) * 0.5")}, True, False),  # the post-training fit span
    ({"model.py": READOUT.replace("# unused", "") + "# a trailing comment\n"}, True, False),
    ({"model.py": BUFFER.replace('"pf_zmu"', '"zmu"')}, False, False),        # load_pretrained would refuse it
    ({"model.py": READOUT.replace("return 1", "return 2")}, False, False),    # trained code
    ({"train.py": TRAIN.replace("lr = 0.1", "lr = 0.2")}, False, False),
    ({"configs__main.yaml": yaml.safe_dump({"train": {"seed": 1, "lr": 2e-4}})}, False, False),
])
def test_eval_only_eligibility(tmp_path, repo, files, ok, shim):
    adapter, _ = np_adapter(tmp_path, repo)
    parent, child = commit_edit(repo, "c", **files)
    got = adapter.eval_only(parent, child)
    assert (got[0], got[2]) == (ok, shim), got


def test_graft_reapplies_the_readout_on_another_model(tmp_path, repo):
    adapter, _ = np_adapter(tmp_path, repo)
    parent, child = commit_edit(repo, "child", **{"model.py": READOUT})
    _, other = commit_edit(repo, "other", **{"model.py": MODEL.replace("return 1", "return 7")})
    sha = adapter.graft(parent, child, other, "n9-on-other")
    text = git(repo, "show", f"{sha}:model.py")
    assert "(x - x.mean()) * 2" in text and "return 7" in text
    assert adapter.graft(parent, child, parent, "same") == child               # its own parent: itself
    moves = git(repo, "reflog", "show", f"loop/{adapter._tag}-eo-n9-on-other")
    assert adapter.graft(parent, child, other, "n9-on-other") == sha             # idempotent: the branch is reused
    assert git(repo, "reflog", "show", f"loop/{adapter._tag}-eo-n9-on-other") == moves   # ... not rebuilt
    _, clash = commit_edit(repo, "clash", **{"model.py": MODEL.replace("return x * 2", "return x * 5")})
    assert adapter.graft(parent, child, clash, "n9-on-clash") == ""            # does not apply: no model


def test_graft_port_merges_the_config_per_key_and_reports_contained_and_conflicts(tmp_path, repo):
    """Live 2026-10-06: 9 of 10 backfill ports onto n0655 failed `git apply --3way` on ADJACENT YAML lines (n0313's
    crop / lr / kde lines next to n0655's lr / layerwise / kde). The config now merges per key: a key the incumbent
    already holds (equal, or moved further the same way) is kept, a new key lands after its predecessor with its
    comment; code still merges 3-way from the node's parent."""
    adapter, _ = np_adapter(tmp_path, repo)
    cfg = (repo / "configs" / "main.yaml").read_text()
    inc_cfg = cfg.replace("lr: 0.0001", "lr: 0.0003").replace("save_every: 1000", "save_every: 500")
    _, inc = commit_edit(repo, "inc", **{"configs__main.yaml": inc_cfg, "model.py": MODEL.replace("return 1", "return 7")})
    node_cfg = cfg.replace("lr: 0.0001", "lr: 0.0002").replace("seed: 1", "seed: 2").replace(
        "  save_every: 1000\n", "  save_every: 1000\n  # a new knob\n  new_knob: 3\n")
    parent, node = commit_edit(repo, "node", **{"configs__main.yaml": node_cfg, "model.py": READOUT})
    sha, how = adapter.graft_port(parent, node, inc, "bf-n1")
    got = yaml.safe_load(git(repo, "show", f"{sha}:configs/main.yaml"))["train"]
    assert how == "applied" and got == {"lr": 3e-4, "save_every": 500, "seed": 2, "new_knob": 3, "resume": None}
    assert "# a new knob" in git(repo, "show", f"{sha}:configs/main.yaml")
    assert "(x - x.mean()) * 2" in git(repo, "show", f"{sha}:model.py") and "return 7" in git(repo, "show", f"{sha}:model.py")
    _, held = commit_edit(repo, "held", **{"configs__main.yaml": cfg.replace("lr: 0.0001", "lr: 0.0002")})
    assert adapter.graft_port(parent, held, inc, "bf-n2") == ("", "contained")            # inc moved lr further
    assert adapter.graft(parent, held, inc, "bf-n2") == inc
    _, down = commit_edit(repo, "down", **{"configs__main.yaml": cfg.replace("lr: 0.0001", "lr: 5.0e-05")})
    sha, how = adapter.graft_port(parent, down, inc, "bf-n3")
    assert sha == "" and how == "conflict: configs/main.yaml:train.lr"                      # moved the other way
    _, clash = commit_edit(repo, "clash", **{"model.py": MODEL.replace("return 1", "return 9")})
    assert adapter.graft_port(parent, clash, inc, "bf-n4") == ("", "conflict: model.py")


def test_graft_reaps_a_stale_locked_worktree_left_by_a_killed_add(tmp_path, repo):
    adapter, _ = np_adapter(tmp_path, repo)
    parent, child = commit_edit(repo, "child", **{"model.py": READOUT})
    _, other = commit_edit(repo, "other", **{"model.py": MODEL.replace("return 1", "return 7")})
    dest = adapter.worktrees / "_graft" / "n9-on-other"                        # SIGTERM mid `worktree add`:
    git(repo, "worktree", "add", "-B", f"loop/{adapter._tag}-eo-n9-on-other", str(dest), other)
    git(repo, "worktree", "lock", "--reason", "initializing", str(dest))       # registered, locked, branch held
    sha = adapter.graft(parent, child, other, "n9-on-other")
    assert sha and "return 7" in git(repo, "show", f"{sha}:model.py") and not dest.exists()


def test_eval_only_submit_resumes_the_source_checkpoint(tmp_path, repo):
    def shim(argv):
        Path(argv[-1]).write_bytes(b"shim")
        return 0, "{}", ""
    adapter, runner = np_adapter(tmp_path, repo, SlurmRunner({**sbatch_ok("9"), "shim.py": shim}))
    parent, child = commit_edit(repo, "c", **{"model.py": BUFFER})
    run = Run(id=Run.make_id("n5", Stage.PROMOTE, EO.EO_SEED), task_id="fake-loop", node_id="n5",
              seed=EO.EO_SEED, stage=Stage.PROMOTE, status=RunStatus.SUBMITTING)
    adapter.plan_eval("n5", Stage.PROMOTE, EO.EO_SEED, {"commit": child, "seed": 13, "shim": True,
                                                        "resume": "/runs/root/baseline_s13_a0/latest.pt",
                                                        "source_run": "root:baseline:13", "model": "root"})
    handle = adapter.submit(run=run, commit=parent)
    call = next(c for c in runner.calls if c["argv"][0] == LAUNCHER)
    out = adapter.runs / "n5" / f"promote_s{EO.EO_SEED}_a0"
    assert call["argv"][2:] == [f"output_dir={out}", "seed=13"]              # the source run's seed
    cfg = yaml.safe_load(Path(call["argv"][1]).read_text())
    assert cfg["train"]["resume"] == str(out / "resume_shim.pt") and cfg["train"]["save_every"] is None
    assert (out / "resume_shim.pt").read_bytes() == b"shim"
    assert call["env"]["WANDB_MODE"] == "disabled" and git(Path(call["cwd"]), "rev-parse", "HEAD") == child
    assert handle.job_id == "9"
    plain = Run(id="n5:promote:900001", task_id="fake-loop", node_id="n5", seed=EO.EO_SEED + 1,
                stage=Stage.PROMOTE, status=RunStatus.SUBMITTING)
    adapter.plan_eval("n5", Stage.PROMOTE, EO.EO_SEED + 1, {"commit": parent, "seed": 11, "shim": False,
                                                            "resume": "/r/latest.pt", "source_run": "x", "model": "root"})
    argv, _, _ = adapter._launch(plain)
    assert yaml.safe_load(Path(argv[1]).read_text())["train"]["resume"] == "/r/latest.pt"


def test_init_compat_is_skipped_only_for_an_enabled_eligible_readout(tmp_path, repo, monkeypatch):
    import loop.adapters.nanopath_v2 as NP
    monkeypatch.setattr(NP, "run_init_check", lambda **kw: NP.InitCompat(False, "checkpoint_keys"))
    for enabled, text, ok in ((True, BUFFER, True), (True, READOUT.replace("return 1", "return 2"), False),
                              (False, BUFFER, False)):
        adapter, _ = np_adapter(tmp_path / str(enabled) / str(ok), repo, enabled=enabled)
        wt = adapter.prepare_worktree(parent_commit=git(repo, "rev-parse", "main"), name=f"r{enabled}{ok}")
        (Path(wt.path) / "model.py").write_text(text)
        adapter.commit(wt, message="m")
        assert adapter.init_compat(wt, parent_commit=wt.base_commit).ok is ok
        adapter.reap_worktree(wt)


# ------------------------------------------------------------------ the lane on a real store

class FakeAdapter:
    def __init__(self, ok=True, shim=False):
        self.ok, self.shim, self.plans, self.grafts = ok, shim, {}, []

    def eval_only(self, parent, commit):
        return self.ok, "" if self.ok else "trained code", self.shim

    def graft(self, parent, commit, onto, name):
        self.grafts.append(name)
        return f"{commit}+{onto}"

    def diff_commits(self, a, b):
        return f"diff {a} {b}\n+readout\n"

    def plan_eval(self, node_id, stage, seed, plan):
        self.plans[Run.make_id(node_id, stage, seed)] = dict(plan)

    def eval_plan(self, run):
        return self.plans.get(run.id)

    def submit(self, *, run, commit):
        return JobHandle(run.id, f"j{len(self.plans)}", f"/out/{run.id}", 0.0)

    def cancel(self, handle):
        pass


def lane_ctx(ctx, **kw):
    return dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, evalonly=EvalOnlyCfg(enabled=True, **kw)))


def node(store, nid, status, parent=ROOT_NODE_ID, family="eval_readout", ts=1.0, **kw):
    n = Node(id=nid, task_id=store.task_id, commit=f"c{nid}", parent_id=parent, strategy=Strategy.EXPLORE,
             operator=None, family=family, mechanism_tag=f"t{nid}", diff_hash=nid, status=status,
             created_ts=ts, updated_ts=ts, **kw)
    store.put_node(n)
    return n


def done(ctx, store, nid, stage, seed, base=0.5, agg=None, out=None, delta=0.0, **axes):
    scores = {a: base + delta + axes.get(a, 0.0) for a in ctx.cfg.axes}
    run = Run(id=Run.make_id(nid, stage, seed), task_id=store.task_id, node_id=nid, seed=seed, stage=stage,
              status=RunStatus.DONE, scores=scores, aggregate=agg if agg is not None else base + delta,
              output_dir=str(out) if out else "")
    if store.get_run(run.id) is None:
        store.put_run(run)
    else:
        store.update_run(run.id, status=RunStatus.DONE, scores=scores, aggregate=run.aggregate)
    return run


def models_with_ckpts(ctx, store, tmp_path, n_root=3, n_inc=3):
    store.ensure_root("croot")
    node(store, "inc", NodeStatus.CONFIRMED)
    for nid, k in ((ROOT_NODE_ID, n_root), ("inc", n_inc)):
        for s in range(11, 11 + k):
            d = tmp_path / "runs" / nid / f"screen_s{s}_a0"
            d.mkdir(parents=True)
            (d / "latest.pt").write_bytes(b"x" * 10)
            (d / "summary.json").write_text("{}")
            done(ctx, store, nid, Stage.BASELINE if nid == ROOT_NODE_ID else Stage.SCREEN, s, out=d)


class Snap:
    incumbent_id, banked = "inc", ()


def lane(ctx, store, adapter=None, **kw):
    ctx = lane_ctx(ctx, **{"max_share": 1.0, **kw})
    adapter = adapter or FakeAdapter()
    book = RunBook(ctx, store, adapter, clock=lambda: 100.0)
    ids = iter(f"n{i:04d}" for i in range(900, 999))
    return ctx, EO.EvalOnly(ctx, store, adapter, book, lambda: next(ids), lambda: 100.0), adapter


def finish_eval(ctx, store, eo_ctx, delta_by_model, phase_stage=None):
    """Mark every running eval-only run done at its source's score + the model's delta."""
    st = store.get_state(EO.K_EO)
    for e in st["nodes"].values():
        for rid, r in e["runs"].items():
            run = store.get_run(rid)
            if run is None or run.status is RunStatus.DONE:
                continue
            src = store.get_run(r["source"])
            nid, stage, seed = rid.split(":")
            done(ctx, store, nid, Stage(stage), int(seed), base=src.scores[ctx.cfg.axes[0]],
                 delta=delta_by_model[r["model"]])


def test_null_control_gates_the_lane(ctx, store, tmp_path):
    models_with_ckpts(ctx, store, tmp_path)
    node(store, "n1", NodeStatus.QUEUED)
    ectx, eo, ad = lane(ctx, store)
    acts = eo.tick(Snap())
    assert store.get_node("n1").status is NodeStatus.EVALUATING and "evalonly=yes" in store.get_node("n1").notes
    null = store.get_state(EO.K_EO)["null"]
    assert null["status"] == "running" and len(null["runs"]) == 2               # incumbent + root
    assert all(r.startswith(("inc:audit:99", "root:audit:99")) for r in null["runs"])
    plan = ad.plans["inc:audit:990000"]
    assert plan["commit"] == "cinc" and plan["seed"] == 11 and plan["resume"].endswith("inc/screen_s11_a0/latest.pt")
    assert not store.runs(node_id="n1")                                          # nothing scored before the null
    for rid, src in null["runs"].items():
        s = store.get_run(src)
        nid, _, seed = rid.split(":")
        done(ctx, store, nid, Stage.AUDIT, int(seed), base=s.scores[ctx.cfg.axes[0]], agg=s.aggregate + 5e-5)
    acts = eo.tick(Snap())
    assert store.get_state(EO.K_EO)["null"]["status"] == "passed", acts
    assert len(store.runs(node_id="n1")) == 3                                    # screening launched


def test_failed_null_control_blocks_scoring_and_returns_nodes_to_the_queue(ctx, store, tmp_path):
    models_with_ckpts(ctx, store, tmp_path)
    node(store, "n1", NodeStatus.QUEUED)
    _, eo, _ = lane(ctx, store)
    eo.tick(Snap())
    for rid, src in store.get_state(EO.K_EO)["null"]["runs"].items():
        s = store.get_run(src)
        nid, _, seed = rid.split(":")
        done(ctx, store, nid, Stage.AUDIT, int(seed), base=s.scores[ctx.cfg.axes[0]], agg=s.aggregate + 1e-3)
    eo.tick(Snap())
    assert store.get_state(EO.K_EO)["null"]["status"] == "failed"
    n = store.get_node("n1")
    assert n.status is NodeStatus.QUEUED and "evalonly=blocked" in n.notes and not store.runs(node_id="n1")
    node(store, "n2", NodeStatus.QUEUED, ts=2.0)
    eo.tick(Snap())
    assert store.get_node("n2").status is NodeStatus.QUEUED                     # never routed again


def passed_null(ctx, store, tmp_path, **kw):
    models_with_ckpts(ctx, store, tmp_path, **kw)
    store.set_state(EO.K_EO, {"null": {"status": "passed"}})


def test_lifecycle_screens_across_models_confirms_on_fresh_checkpoints_and_graduates(ctx, store, tmp_path):
    passed_null(ctx, store, tmp_path)
    node(store, "n1", NodeStatus.QUEUED, hypothesis_id=None)
    _, eo, ad = lane(ctx, store)
    eo.tick(Snap())
    e = store.get_state(EO.K_EO)["nodes"]["n1"]
    screen = {r["source"] for r in e["runs"].values()}
    assert len(e["runs"]) == 3 and {r["model"] for r in e["runs"].values()} == {"inc", ROOT_NODE_ID}
    assert all(store.get_run(rid).stage is Stage.PROMOTE and store.get_run(rid).seed >= EO.EO_SEED for rid in e["runs"])
    finish_eval(ctx, store, None, {"inc": 0.004, ROOT_NODE_ID: 0.003})
    acts = eo.tick(Snap())
    e = store.get_state(EO.K_EO)["nodes"]["n1"]
    assert e["phase"] == "confirm" and len(e["runs"]) == 5, acts
    fresh = {r["source"] for r in e["runs"].values() if r["phase"] == "confirm"}
    assert fresh and not fresh & screen                                          # fresh checkpoints
    assert store.get_node("n1").status is NodeStatus.EVALUATING
    finish_eval(ctx, store, None, {"inc": 0.004, ROOT_NODE_ID: 0.003})
    acts = eo.tick(Snap())
    n1 = store.get_node("n1")
    assert n1.status is NodeStatus.EVAL_PASSED and n1.confirm_e > 1.0 / ctx.cfg.promotion.alpha_confirm, acts
    new = [n for n in store.nodes(status=[NodeStatus.QUEUED]) if "evalonly_from=n1" in n.notes]
    assert len(new) == 1 and new[0].parent_id == "inc" and new[0].commit == "cn1+cinc"
    assert Path(new[0].diff_path).read_text().startswith("diff cinc")
    eo.tick(Snap())
    assert store.get_node(new[0].id).status is NodeStatus.QUEUED                # full retrain: never re-routed
    # the world model never pairs an eval-only run (synthetic seeds, no control)
    snap = ComputedWorldModel(lane_ctx(ctx)).fit(store)
    assert "n1" not in snap.nodes


@pytest.mark.parametrize("deltas,status", [
    ({"inc": 0.004, ROOT_NODE_ID: -0.002}, NodeStatus.SCREENED_OUT),          # does not generalise
    ({"inc": -0.001, ROOT_NODE_ID: -0.001}, NodeStatus.SCREENED_OUT),
])
def test_screen_needs_every_model_positive(ctx, store, tmp_path, deltas, status):
    passed_null(ctx, store, tmp_path)
    node(store, "n1", NodeStatus.QUEUED)
    _, eo, _ = lane(ctx, store)
    eo.tick(Snap())
    finish_eval(ctx, store, None, deltas)
    eo.tick(Snap())
    assert store.get_node("n1").status is status


def test_veto_on_a_catastrophic_axis(ctx, store, tmp_path):
    passed_null(ctx, store, tmp_path)
    node(store, "n1", NodeStatus.QUEUED)
    ectx, eo, _ = lane(ctx, store)
    eo.tick(Snap())
    a = ctx.cfg.axes[-1]
    obs = [("inc", {**{x: 0.004 for x in ctx.cfg.axes}, a: -0.2, PANEL: 0.001})] * 3
    assert EO.vetoed(ectx, obs, 0.0005) == (a,)
    assert EO.vetoed(ectx, [("inc", {**{x: 0.004 for x in ctx.cfg.axes}, PANEL: 0.004})] * 3, 0.0005) == ()


def test_too_few_models_or_ineligible_means_a_full_run(ctx, store, tmp_path):
    store.ensure_root("croot")
    store.set_state(EO.K_EO, {"null": {"status": "passed"}})
    d = tmp_path / "r"
    d.mkdir()
    (d / "latest.pt").write_bytes(b"x")
    done(ctx, store, ROOT_NODE_ID, Stage.BASELINE, 11, out=d)                     # one model only
    node(store, "n1", NodeStatus.QUEUED)
    node(store, "n2", NodeStatus.QUEUED, ts=2.0)
    _, eo, ad = lane(ctx, store)
    eo.tick(Snap())
    assert store.get_node("n1").status is NodeStatus.QUEUED and "evalonly=nockpt" in store.get_node("n1").notes
    ad.ok = False
    node(store, "n3", NodeStatus.QUEUED, ts=3.0)
    eo.tick(Snap())
    assert "evalonly=no" in store.get_node("n3").notes and store.get_node("n3").status is NodeStatus.QUEUED
    node(store, "n4", NodeStatus.QUEUED, ts=4.0, notes="request=r4 fams=eval_readout")      # readout-only:
    node(store, "n5", NodeStatus.QUEUED, ts=5.0, notes="request=r5 fams=eval_readout+training")
    eo.tick(Snap())                                                  # never a full training run (n0585)
    assert store.get_node("n4").status is NodeStatus.REJECTED and store.get_node("n4").reject_reason == "readout_not_evalonly"
    assert store.get_node("n5").status is NodeStatus.QUEUED and "evalonly=no" in store.get_node("n5").notes


def test_init_incompatible_readouts_are_requeued_once(ctx, store, tmp_path):
    passed_null(ctx, store, tmp_path)
    node(store, "n1", NodeStatus.REJECTED, reject_reason="init_incompatible")
    node(store, "n2", NodeStatus.REJECTED, reject_reason="exact_dup")
    _, eo, _ = lane(ctx, store)
    eo.tick(Snap())
    assert store.get_node("n1").status is NodeStatus.EVALUATING and "evalonly_requeue=1" in store.get_node("n1").notes
    assert store.get_node("n2").status is NodeStatus.REJECTED
    assert store.get_state(EO.K_EO)["requeued"] == ["n1"]


def test_eval_only_share_of_slots_is_capped(ctx, store, tmp_path):
    passed_null(ctx, store, tmp_path, n_root=4, n_inc=4)
    for i in range(3):
        node(store, f"n{i}", NodeStatus.QUEUED, ts=float(i))
    _, eo, _ = lane(ctx, store, max_share=0.5)                                   # 4 slots -> 2
    eo.tick(Snap())
    assert sum(EO.is_eval_run(r) for r in store.runs(status=[RunStatus.RUNNING])) == 2


def test_eval_only_crash_and_spent_void_close_the_node(ctx, store, tmp_path):
    passed_null(ctx, store, tmp_path)
    node(store, "n1", NodeStatus.QUEUED)
    ectx, eo, _ = lane(ctx, store)
    eo.tick(Snap())
    rid = next(iter(store.get_state(EO.K_EO)["nodes"]["n1"]["runs"]))
    store.update_run(rid, status=RunStatus.CRASH)
    eo.tick(Snap())
    assert store.get_node("n1").status is NodeStatus.CRASHED


def test_pick_rotates_and_needs_distinct_models():
    runs = {m: [Run(id=f"{m}:{i}", task_id="t", node_id=m, seed=i, stage=Stage.SCREEN, status=RunStatus.DONE)
                for i in range(3)] for m in ("a", "b")}
    from collections import Counter
    got = EO.pick(runs, 3, skip=set(), uses=Counter({"a:0": 2}), min_models=2)
    assert [r.id for _, r in got] == ["a:1", "b:0", "a:2"]
    assert EO.pick({"a": runs["a"]}, 2, skip=set(), uses=Counter(), min_models=2) == []
    assert [m for m, _ in EO.pick(runs, 2, skip=set(), uses=Counter(), min_models=2, prefer_new=["a"])] == ["b", "a"]


def test_sigma_is_the_floored_pooled_spread_across_checkpoints(ctx):
    c = lane_ctx(ctx)
    assert EO.sigma(c, {"n1": [0.001, 0.001, 0.001]}) == pytest.approx(5e-4)
    assert EO.sigma(c, {"n1": [0.0, 0.004], "n2": [0.001]}) == pytest.approx(0.004 / 2 ** 0.5)


# ------------------------------------------------------------------ pruning

def prune_ctx(ctx, **kw):
    import tempfile
    sd = ctx.state_dir or tempfile.mkdtemp(prefix="prune-state-")            # the live prune audit trail needs a state dir
    return dataclasses.replace(ctx, state_dir=sd, cfg=dataclasses.replace(ctx.cfg, prune=PruneCfg(enabled=True, **kw)))


def ck(ctx, store, tmp_path, nid, stage, seed, attempt=0, status=RunStatus.DONE, extra=()):
    d = tmp_path / "runs" / nid / f"{stage.value}_s{seed}_a{attempt}"
    d.mkdir(parents=True)
    for name in ("latest.pt", "summary.json", "metrics.jsonl", *extra):
        (d / name).write_bytes(b"x" * 100)
    if attempt:                                                                   # the earlier attempt's dir
        old = d.with_name(f"{stage.value}_s{seed}_a0")
        old.mkdir()
        (old / "latest.pt.tmp").write_bytes(b"x" * 100)
    run = Run(id=Run.make_id(nid, stage, seed), task_id=store.task_id, node_id=nid, seed=seed, stage=stage,
              status=status, scores={a: 0.5 for a in ctx.cfg.axes}, aggregate=0.5, output_dir=str(d), attempt=attempt)
    store.put_run(run)
    return d


def test_prune_keep_set(ctx, store, tmp_path):
    now = 10 * 86400.0
    old = now - 2 * 86400.0
    store.ensure_root("croot")
    root = [ck(ctx, store, tmp_path, ROOT_NODE_ID, Stage.BASELINE, s) for s in range(11, 18)]   # 7 root runs
    node(store, "inc", NodeStatus.CONFIRMED, ts=old)
    inc = ck(ctx, store, tmp_path, "inc", Stage.SCREEN, 11)
    node(store, "dead", NodeStatus.SCREENED_OUT, ts=old)
    dead = ck(ctx, store, tmp_path, "dead", Stage.SCREEN, 11, attempt=1, extra=("step_1000.pt",))
    node(store, "fresh", NodeStatus.VETOED, ts=now - 3600)                       # inside the grace period
    fresh = ck(ctx, store, tmp_path, "fresh", Stage.SCREEN, 11)
    node(store, "par", NodeStatus.NOT_CONFIRMED, ts=old)                        # parent of a live node
    par = ck(ctx, store, tmp_path, "par", Stage.SCREEN, 11)
    node(store, "kid", NodeStatus.SCREENING, parent="par", ts=old)
    kid = ck(ctx, store, tmp_path, "kid", Stage.SCREEN, 12)
    node(store, "ev", NodeStatus.EVAL_PASSED, ts=old)
    ev = ck(ctx, store, tmp_path, "ev", Stage.PROMOTE, EO.EO_SEED, extra=("resume_shim.pt",))
    store.set_state(EO.K_EO, {"nodes": {"ev": {"runs": {"ev:promote:900000": {"source": "root:baseline:17"}}}}})
    run_ = ck(ctx, store, tmp_path, "kid", Stage.SCREEN, 13, status=RunStatus.RUNNING)
    pctx = prune_ctx(ctx, reserve_per_model=5)
    drop, kept = PR.plan(pctx, store, Snap(), now)
    gone = {p for p, _ in drop}
    assert gone == {dead / "latest.pt", dead / "step_1000.pt",
                    dead.with_name("screen_s11_a0") / "latest.pt.tmp", ev / "latest.pt", ev / "resume_shim.pt"}
    assert all(r / "latest.pt" not in gone for r in root)                         # root: every checkpoint stays
    assert kept == 7 + 1 + 1 + 1 + 1                                              # root x7, inc, fresh, par, kid
    lines = PR.Pruner(prune_ctx(ctx, dry_run=True), store, lambda: now).tick(Snap())
    assert "would be freed" in lines[0] and all(p.exists() for p in gone)
    assert PR.Pruner(prune_ctx(ctx, dry_run=True), store, lambda: now + 60).tick(Snap()) == []   # rate-limited
    lines = PR.Pruner(prune_ctx(ctx, dry_run=False), store, lambda: now + 7200).tick(Snap())
    assert "freed" in lines[0] and not any(p.exists() for p in gone)
    assert (dead / "summary.json").exists() and (dead / "metrics.jsonl").exists()   # never logs / summaries
    assert (run_ / "latest.pt").exists() and (fresh / "latest.pt").exists() and (inc / "latest.pt").exists()
    ev_log = store.events("prune")
    assert ev_log[0]["dry_run"] and ev_log[-1]["freed"] == 500


def test_prune_is_off_unless_enabled(ctx, store):
    assert PR.Pruner(ctx, store, time.time).tick(None) == []


class RaisingAdapter(FakeAdapter):
    def graft(self, parent, commit, onto, name):
        raise RuntimeError("git worktree add failed (rc=-15): Preparing worktree")


def test_a_raising_lane_step_never_reaches_the_driver_and_falls_back_to_a_full_run(ctx, store, tmp_path):
    passed_null(ctx, store, tmp_path)
    node(store, "n1", NodeStatus.QUEUED)
    _, eo, _ = lane(ctx, store, adapter=RaisingAdapter())
    for k in range(1, EO.MAX_ERRORS):
        acts = eo.tick(Snap())
        assert store.get_node("n1").status is NodeStatus.EVALUATING and any("step n1 failed" in a for a in acts)
        assert store.get_state(EO.K_EO)["errors"] == {"n1": k}
    eo.tick(Snap())
    n1 = store.get_node("n1")
    assert n1.status is NodeStatus.QUEUED and "evalonly=error" in n1.notes and not store.runs(node_id="n1")
    assert len(store.events("evalonly_error")) == EO.MAX_ERRORS
    eo.tick(Snap())
    assert store.get_node("n1").status is NodeStatus.QUEUED                     # never routed again


def test_lost_lane_state_readopts_the_plans_its_runs_were_launched_with(ctx, store, tmp_path):
    passed_null(ctx, store, tmp_path)
    node(store, "n1", NodeStatus.QUEUED)
    _, eo, ad = lane(ctx, store)
    eo.tick(Snap())
    rid = sorted(store.get_state(EO.K_EO)["nodes"]["n1"]["runs"])[0]
    ad.plans[rid].update(source_run="root:baseline:13", model=ROOT_NODE_ID)   # what it really resumed
    store.set_state(EO.K_EO, {"null": {"status": "passed"}})                   # the tick died before saving
    eo.tick(Snap())
    r = store.get_state(EO.K_EO)["nodes"]["n1"]["runs"][rid]
    assert (r["source"], r["model"]) == ("root:baseline:13", ROOT_NODE_ID) and len(store.runs(node_id="n1")) == 3


def test_prune_keeps_every_checkpoint_a_launched_plan_resumes(ctx, store, tmp_path):
    now = 10 * 86400.0
    store.ensure_root("croot")
    node(store, "old", NodeStatus.NOT_CONFIRMED, ts=now - 3 * 86400.0)
    old = [ck(ctx, store, tmp_path, "old", Stage.SCREEN, s) for s in range(11, 13)]
    ck(ctx, store, tmp_path, ROOT_NODE_ID, Stage.BASELINE, 11)                 # root: never pruned at all
    plan = Path(ctx.state_dir) / "evalonly" / "n1" / "promote_s900000.json"
    plan.parent.mkdir(parents=True)
    plan.write_text('{"source_run": "old:screen:12"}')                         # no lane state entry at all
    assert [p for p, _ in PR.plan(prune_ctx(ctx, reserve_per_model=1), store, Snap(), now)[0]] == [old[0] / "latest.pt"]
    plan.unlink()
    assert {p for p, _ in PR.plan(prune_ctx(ctx, reserve_per_model=1), store, Snap(), now)[0]} == {o / "latest.pt" for o in old}


def test_git_plan_keeps_live_model_and_in_flight_work(ctx, store):
    from loop.contracts import K_INFLIGHT
    now = time.time() + 3 * 86400.0
    store.ensure_root("croot")
    for nid, st, req in (("dead", NodeStatus.SCREENED_OUT, "req-1"), ("inc", NodeStatus.CONFIRMED, "req-2"),
                         ("live", NodeStatus.SCREENING, "req-3")):
        node(store, nid, st, ts=now - 2 * 86400.0)
        store.update_node(nid, notes=f"request={req} brief=x")
    for req in ("req-1", "req-2", "req-3", "req-4", "req-5"):              # req-4: no node (failed research)
        store.log("research_started", request_id=req)
    store.set_state(K_INFLIGHT, [{"id": "req-5"}])
    ck(ctx, store, Path(ctx.state_dir), "dead", Stage.SCREEN, 11, status=RunStatus.DONE)
    commits, requests, nodes = PR.git_plan(prune_ctx(ctx), store, Snap(), now)
    assert requests == ["req-1", "req-4"] and nodes == ["dead"]
    assert set(commits) == {"root", "dead", "inc", "live"}
    calls = []

    class Tidy:
        def tidy(self, *a, **kw):
            calls.append((a, kw))
            return {"tagged": 4, "branches": 2, "worktrees": 1}
    lines = PR.Pruner(prune_ctx(ctx, dry_run=False), store, lambda: now, Tidy()).tick(Snap())
    assert lines[-1].startswith("prune git: 2 research branches, 1 node worktrees") and calls[0][1]["dry_run"] is False
    # the driver's pruner runs the pass in a background thread; the next tick reports it
    bg = PR.Pruner(prune_ctx(ctx, dry_run=False), store, lambda: now + 86400.0, Tidy(), background=True)
    assert bg.tick(Snap()) == [] and bg._thread is not None
    bg._thread.join(30)
    assert any(l.startswith("prune git: 2 research branches") for l in bg.tick(Snap()))
    bg.stop()
    assert calls[-1][1]["halt"]() is True                                 # a stop ends the pass between git commands
