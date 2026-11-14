"""v3-c4i (09Z deep meta-analysis, 10-13): unflushed computed facts in fork-and-flush briefs, the per-axis claim budget,
programs as staged plans, the researcher's diagnose tool, the new levers (compute headroom, per-dataset gaps, repair
candidates), annotated diagnostics, check without a diff + revert + the clean bounce-limit reject, spread concurrent
sessions, dataset-content near-dups, bounded web + the session clock."""

from __future__ import annotations

import csv
import dataclasses
import json
import math
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from loop import diagnostics as DG
from loop.agents import diagnose as DX
from loop.agents import llm as LLM
from loop.agents import researcher as RS
from loop.contracts import (K_INFLIGHT, CandidateRequest, Node, NodeStatus, Operator, ROOT_NODE_ID, Strategy,
                            Worktree)
from loop.core import data as D
from loop.core import fnf as FN
from loop.core import ideation as ID
from loop.core import levers as LV
from loop.core import model as M
from loop.core import program as PG
from loop.select import intake as IN

from ..conftest import InlineExecutor
from ..test_databank import _ctx as _bank_ctx, _ds
from .test_fnf import _Research, env  # noqa: F401  (the fixture)

BASE_PY = Path("/home/user/nanopath/base/.venv/bin/python")


def _icfg(ctx, **kw):
    return dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, ideation=dataclasses.replace(
        ctx.cfg.ideation, **kw)))


# ---------------------------------------------------------------------------------------------- 1. unflush
def test_a_flushed_branch_keeps_the_computed_levers_and_headroom_but_not_the_shared_story(env, monkeypatch):
    ctx, store, led, ide = env
    assert "headroom_target" not in FN.FLUSH and {"notes", "postmortem", "handoffs", "plateau"} <= set(FN.FLUSH)
    monkeypatch.setattr(ID, "target_gap", lambda v: 0.006)
    br = ide.brief_for("explore", led, ide.view(led), "program", branch="1.2")
    for kept in ("## Per-axis headroom (computed)", "## Untested levers & deficits (computed)",
                 "### (b) Per-axis deficits", "### (c) External headroom", "## Effect size (a per-axis budget)"):
        assert kept in br.text
    for gone in ("## Engagement of recent ideas", "## Computed facts", "## Literature notes (top 5"):
        assert gone not in br.text
    assert br.ablated == FN.FLUSH and br.kind == "program"


# ---------------------------------------------------------------------------------------------- 2. budget
def test_the_claim_is_a_per_axis_budget_tied_to_brief_rows_and_summed_with_the_official_weights():
    budget = [{"axis": "cls", "row": "D-cls", "gain": 0.01}, {"axis": "rob", "row": "H-rob", "gain": -0.004}]
    assert RS.budget_total(budget) == pytest.approx(0.30 * 0.01 - 0.15 * 0.004)
    assert RS.budget_total([]) is None and RS.budget_total([{"axis": "nope", "gain": 1}]) is None
    rows = ("D-cls", "H-rob", "XD-break_his", "C-compute")
    assert RS.budget_problem(budget, rows) == ""
    assert "not a row" in RS.budget_problem([{"axis": "cls", "row": "D-seg9", "gain": .01}], rows)
    assert "the rob row" in RS.budget_problem([{"axis": "cls", "row": "H-rob", "gain": .01}], rows)
    assert "classification dataset" in RS.budget_problem([{"axis": "seg", "row": "XD-break_his", "gain": .01}], rows)
    assert RS.budget_problem(budget, ()) == ""                         # no rows shown: nothing to cite
    mp = RS.MECHANISM_PREDICTION
    assert "budget" in mp["required"] and "size" not in mp["required"]


def test_target_mode_brief_has_no_size_anchor_and_the_request_carries_floor_and_rows(env, monkeypatch):
    ctx, store, led, ide = env
    v = ide.view(led)
    monkeypatch.setattr(ID, "target_gap", lambda v: 0.0062)
    b = ide.brief_for("explore", led, v, "explore")
    assert "Claim the size you actually believe" not in b.text and "mechanism_prediction.budget" in b.text
    assert "[H-cls]" in b.text and "[D-" in b.text
    req = ide.request(b, v)
    # v3-c6i: no claim floor any more -- the slate's prediction is stated against the gap and scored
    assert req.claim_floor == 0.0 and req.ref_gap == pytest.approx(0.0062) and req.k_candidates > 0
    assert "H-cls" in req.rows and req.rows == ID.row_ids(b.text)
    monkeypatch.setattr(ID, "target_gap", lambda v: None)
    assert ide.request(ide.brief_for("explore", led, v, "explore"), v).ref_gap == pytest.approx(v.mde(), abs=1e-6)


def _tools(tmp_path, checker=None, **kw):
    t = RS.WorktreeTools(None, Worktree(str(tmp_path), "b", "c0"), writable=("*",), locked=(),
                         checker=checker or (lambda **k: "OK"))
    t.v3 = True
    for k, x in kw.items():
        setattr(t, k, x)
    return t


def _fin(budget, **extra):
    return {"hypothesis": {"claim": "c", "mechanism_tag": "m", "family": "optim", "falsification": "f",
                           "mechanism_prediction": {"diagnostic": "lr_end_frac", "direction": 1, "axes": {"cls": 1},
                                                    "budget": budget}},
            "handoff": {"actions": "a", "learnings": "l", "concerns": "c"}, **extra}


def test_a_budget_below_the_gap_share_bounces_once_then_is_logged_as_is(tmp_path):
    t = _tools(tmp_path, claim_floor=0.003, rows=("D-cls",), size_cap=1)
    small = _fin([{"axis": "cls", "row": "D-cls", "gain": 0.005}])               # 0.0015 < 0.003
    msg = t.check_finish(small)
    assert msg.startswith("budget below the target share") and "0.0015" in msg
    assert t.check_finish(small) == "" and t.below_floor                         # the second time: as is
    t2 = _tools(tmp_path, claim_floor=0.003, rows=("D-cls",))
    assert t2.check_finish(_fin([{"axis": "cls", "row": "D-cls", "gain": 0.02}])) == "" and not t2.below_floor
    t3 = _tools(tmp_path, claim_floor=0.003, rows=("D-cls",))
    assert "REJECT budget" in t3.check_finish(_fin([{"axis": "cls", "row": "D-rob", "gain": 0.02}]))


def test_a_program_finish_declares_components_targeting_brief_rows(tmp_path):
    t = _tools(tmp_path, rows=("D-cls", "C-compute"), program=True, min_components=2)
    one = [{"family": "optim", "mechanism": "a", "knobs": [{"key": "train.lr", "lo": 1, "hi": 2}],
            "target": "D-cls", "share": 1.0}]
    budget = [{"axis": "cls", "row": "D-cls", "gain": 0.02}]
    assert "REJECT components" in t.check_finish(_fin(budget, components=one))
    two = one + [{"family": "augment", "mechanism": "b", "knobs": [{"key": "aug.p", "lo": 0, "hi": 1}],
                  "target": "Q-nope", "share": 0.5}]
    assert "not rows of the brief" in t.check_finish(_fin(budget, components=two))
    two[1]["target"] = "C-compute"
    assert t.check_finish(_fin(budget, components=two)) == ""


# ---------------------------------------------------------------------------------------------- 3. programs
def test_allocate_spreads_the_dose_budget_over_components_by_share():
    assert PG.allocate([0.5, 0.3, 0.2], 4) == [2, 1, 1]
    assert PG.allocate([1.0, 1.0], 4) == [2, 2]
    assert PG.allocate([0.9, 0.05, 0.05], 2) == [1, 1, 0]               # first one each, largest share first
    assert sum(PG.allocate([0.2, 0.7, 0.1], 4)) == 4 and PG.allocate([], 4) == []
    assert PG.allocate([0.5, 0.5], 0) == [0, 0]


def _prog_env(store, monkeypatch, belief):
    from loop.core import factorial as F
    store.ensure_root("r0")
    store.put_node(Node(id="p1", task_id=store.task_id, commit="c-p1", parent_id=ROOT_NODE_ID,
                        strategy=Strategy.MOONSHOT, operator=Operator.DRAFT, family="objective",
                        mechanism_tag="prog", diff_hash="h-p1", status=NodeStatus.SCREENING))
    comps = [{"family": "objective", "mechanism": "a", "target": "D-cls", "share": 0.75,
              "knobs": [{"key": "loss.w", "lo": 0.25, "hi": 4.0, "scale": "log"}]},
             {"family": "augmentation", "mechanism": "b", "target": "C-compute", "share": 0.25,
              "knobs": [{"key": "aug.v", "lo": 2, "hi": 8}], "off": {"aug.v": 2}}]
    store.set_state("ideation:p1", {"components": comps, "knobs": comps[0]["knobs"]})
    nodes = {"p1": D.Node("p1", ROOT_NODE_ID, None, "moonshot", "objective", "explore", ROOT_NODE_ID, 1.0, "screening",
                          "request=r1 brief=program", base="B")}
    data = SimpleNamespace(nodes=nodes, port=lambda n, b=None: n, epoch="B")
    post = SimpleNamespace(est={"p1": {}}, belief=lambda n: belief)
    built = []
    adapter = SimpleNamespace(config_rel="config.yaml",
                              config_at=lambda c: {"loss": {"w": 1.0}, "aug": {"v": 4}},
                              dry_build=lambda commit, levels, knobs, diffs, name: built.append(dict(levels)) or
                              (f"sha-{len(built)}", ""))
    monkeypatch.setattr("loop.extrapolate._allowed", lambda ctx, key: True)
    return data, post, adapter, built


def test_program_components_get_their_share_of_the_dose_budget(ctx, store, monkeypatch):
    data, post, adapter, built = _prog_env(store, monkeypatch, (0.0005, 4e-6))      # +0.25 sd: no LOO yet
    acts = PG.ladders(ctx, store, adapter, data, post, {"p1": "screening"}, 10.0)
    made = store.get_state(PG.K_LADDERS)["p1"]
    assert len(made) == ctx.cfg.ideation.program_wave_max_points - 1 == 4
    keys = [next(iter(store.get_state(f"config_diff:{n}"))) for n in made]
    assert keys.count("loss.w") == 3 and keys.count("aug.v") == 1                   # 0.75 / 0.25 of 4
    assert all("component=" in store.get_node(n).notes for n in made) and acts
    assert store.get_state(PG.K_LOO) in (None, {})


def test_a_positive_program_gets_one_leave_one_out_child_per_component(ctx, store, monkeypatch):
    data, post, adapter, built = _prog_env(store, monkeypatch, (0.006, 4e-6))       # +3 sd
    store.set_state(PG.K_LADDERS, {"p1": []})                                        # ladders done
    acts = PG.ladders(ctx, store, adapter, data, post, {"p1": "screening"}, 10.0)
    loo = store.get_state(PG.K_LOO)["p1"]
    assert set(loo) == {"0", "1"} and acts
    assert store.get_state(f"config_diff:{loo['1']}") == {"aug.v": 2}               # the declared off value
    assert store.get_state(f"config_diff:{loo['0']}") == {"loss.w": 1.0}            # the parent's value
    assert "loo=0" in store.get_node(loo["0"]).notes and "ladder=p1" in store.get_node(loo["0"]).notes
    assert PG.leave_one_out(ctx, store, adapter, data, post, {"p1": "screening"}, 20.0) == []   # once
    c2 = _icfg(ctx, program_loo=False)
    store.set_state(PG.K_LOO, {})
    PG.ladders(c2, store, adapter, data, post, {"p1": "screening"}, 30.0)
    assert store.get_state(PG.K_LOO) == {}


# ---------------------------------------------------------------------------------------------- 4. diagnose
def _summary(d: Path, **scores):
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps({
        "flop_fraction": 0.3, "sample_fraction": 0.98, "stop_reason": "max_train_samples",
        **{f"final_probe_{k.rsplit('_', 1)[0]}_{k.rsplit('_', 1)[1]}_val_f1": x for k, x in scores.items()}}))
    return str(d)


def _thunder(path: Path):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "model", "task", "adaptation", "metric", "metric_score", "ci_low", "ci_high", "setting"])
        w.writerow(["break_his", "uni2h-base-control_optimized", "linear_probing", "frozen", "f1", "78.7", "", "", ""])
        w.writerow(["break_his", "uni2h-c50-s0_optimized", "linear_probing", "frozen", "f1", "99.0", "", "", ""])
        w.writerow(["break_his", "genbio-base-control", "linear_probing", "frozen", "f1", "99.5", "", "", ""])
        w.writerow(["patch_camelyon", "virchow2-base-control_optimized", "knn", "frozen", "f1", "93.0", "", "", ""])


def _led(tmp_path, runs):
    nodes = {n: D.Node(n, ROOT_NODE_ID, None, "explore", "f", "explore", ROOT_NODE_ID, 1.0, "screening", "", base="B")
             for n in {r[0] for r in runs}}
    rr = tuple(D.Run(f"{n}:screen:{i}", n, i, "screen", "done", 1.0, 0.0, 0.0, d, 0, "B")
               for i, (n, d) in enumerate(runs))
    return SimpleNamespace(nodes=nodes, runs=rr, epoch="B")


def test_per_dataset_breakdown_vs_best_node_and_public_off_the_shelf_rows(ctx, tmp_path):
    _thunder(tmp_path / "res.csv")
    c = _icfg(ctx, external_task_csv=str(tmp_path / "res.csv"), deficit_min_seeds=2)
    runs = [("inc", _summary(tmp_path / f"i{s}", break_his_linear=0.55 + s / 100, pcam_knn=0.85)) for s in range(2)]
    runs += [("o1", _summary(tmp_path / f"o{s}", break_his_linear=0.60, pcam_knn=0.80)) for s in range(2)]
    led = _led(tmp_path, runs)
    rows = DX.breakdown(c, led, "inc")
    bh = next(r for r in rows if r["dataset"] == "break_his")
    assert bh["ours"] == pytest.approx(0.555) and bh["n"] == 2 and bh["best"][:2] == ["o1", pytest.approx(0.60)]
    assert bh["ext"] == ["uni2h-base-control_optimized", pytest.approx(0.787)]      # off-the-shelf, GenBio out
    pc = next(r for r in rows if r["dataset"] == "pcam")
    assert pc["ext"][1] == pytest.approx(0.93) and pc["best"][0] == "o1"            # pcam <- patch_camelyon
    gaps = DX.dataset_gaps(rows)
    assert gaps[0]["dataset"] == "break_his" and gaps[0]["gap"] == pytest.approx(0.787 - 0.555)
    text = DX.breakdown_text(rows, "inc")
    assert "break_his/linear" in text and "mean public lead per head" in text


def test_compute_headroom_row_reads_the_flop_fraction(ctx, store, tmp_path):
    led = _led(tmp_path, [("inc", _summary(tmp_path / "a")), ("inc", _summary(tmp_path / "b"))])
    store.set_state(IN.K_BASE, {"train.global_views": 2, "train.local_size": 96, "train.lr": 1e-3})
    v = SimpleNamespace(led=led, incumbent="inc")
    cp = LV.compute(v, store, ctx)
    assert cp["multiplier"] == pytest.approx(1 / 0.3) and cp["stop"] == {"max_train_samples": 2}
    assert cp["keys"] == {"train.global_views": 2, "train.local_size": 96}            # per-tile compute keys only


class _Runner:
    def __init__(self):
        self.calls, self.alive = [], True

    def run(self, argv, *, cwd=None, timeout=None, env=None):
        self.calls.append((list(argv), dict(env or {})))
        if argv[0] == "sbatch":
            return 0, "Submitted batch job 77\n", ""
        if argv[0] == "squeue":
            return 0, ("RUNNING\n" if self.alive else ""), ""
        return 0, "", ""


def test_feature_diagnostic_job_is_submitted_once_on_primary_and_cached(ctx, store, tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "latest.pt").write_text("x")
    wt = tmp_path / "wt"
    (wt / "submit").mkdir(parents=True)
    (wt / "submit" / "train_1gpu.sbatch").write_text("#!/bin/bash\n#SBATCH --gpus=1\necho\n")
    runner = _Runner()
    adapter = SimpleNamespace(runner=runner, config_rel="configs/main.yaml", _node_worktree=lambda n, c: wt,
                              _sbatch=lambda cfg, out, name, pool="": (["x"], {"SBATCH_ACCOUNT": pool,
                                                                               "SBATCH_REQUEUE": ""}))
    DX.record(ctx, store, SimpleNamespace(runs=(D.Run("r", "inc", 1, "screen", "done", 1, 0, 0, str(run), 0),)),
              "inc", "c-inc", None)
    t = [0.0]
    out = DX.features(ctx, store, adapter, wait_s=30, sleep=lambda s: t.__setitem__(0, t[0] + s), clock=lambda: t[0])
    assert "still pending" in out and store.get_state(DX.K_JOBS)["inc"]["job"] == "77"
    sb = [c for c in runner.calls if c[0][0] == "sbatch"]
    assert len(sb) == 1 and "SBATCH_REQUEUE" not in sb[0][1] and sb[0][1]["SBATCH_TIMELIMIT"] == "15"
    script = Path(sb[0][0][1]).read_text()
    assert "#SBATCH --gpus=1" in script and "feature_diag.py" in script and str(run / "latest.pt") in script
    DX.features(ctx, store, adapter, wait_s=0, sleep=lambda s: None, clock=lambda: 0.0)
    assert len([c for c in runner.calls if c[0][0] == "sbatch"]) == 1               # alive: never resubmitted
    res = Path(ctx.state_dir) / "diagnose" / "inc" / "features.json"
    res.write_text(json.dumps({"n_tiles": 4, "datasets": {"bach": {
        "tiles": 4, "norm_probe_features": {"mean": 3.0}, "norm_probe_features_head": {"mean": 2.0},
        "norm_cls": {"mean": 5.0}, "cos_head_vs_cls": {"mean": 0.4},
        "gates": [{"mean": 0.1, "p10": 0.0, "p50": 0.05, "p90": 0.3}]}}}))
    runner.alive = False
    got = DX.Diagnoser(ctx, store, adapter)("features")
    assert "bach (4 tiles)" in got and "0.1 / 0 / 0.05 / 0.3" in got


@pytest.mark.skipif(not BASE_PY.exists(), reason="the nanopath venv (torch) is not on this machine")
def test_the_feature_diag_job_reports_norms_and_sigmoid_gates_per_dataset(tmp_path):
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / "probe.py").write_text('''
import torch
DATASET_ROOTS = {}
THUNDER_V2 = {"classification": {"a": {"val": {"images": list(range(10)), "labels": [0] * 10}},
                                 "b": {"val": {"images": list(range(3)), "labels": [1] * 3}}}}
def worker_probe_transforms(cfg):
    return (lambda x: x), (lambda x: x)
class ClassificationDataset:
    def __init__(self, ds, split, transform):
        self.n, self.v = len(THUNDER_V2["classification"][ds][split]["images"]), 1.0 if ds == "a" else -1.0
    def __len__(self):
        return self.n
    def __getitem__(self, i):
        return torch.full((3, 4, 4), self.v * (i + 1) / 10), 0
''')
    (wt / "model.py").write_text('''
import torch
class ViT(torch.nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.w = torch.nn.Linear(48, 6)
    def forward(self, x):
        return {"cls": self.w(x.flatten(1))}
    def probe_features(self, x):
        cls = self(x)["cls"]
        gate = torch.sigmoid(cls.sum(-1))
        return torch.cat([gate[:, None] * cls, cls.abs()], -1)
''')
    cfg = tmp_path / "c.yaml"
    cfg.write_text(json.dumps({"probe": {"dataset_roots": {}, "datasets": ["a", "b"], "model_weights": "ema"},
                               "model": {"type": "s"}, "data": {"mean": [0, 0, 0], "std": [1, 1, 1]}}))
    ck = tmp_path / "ck.pt"
    subprocess.run([str(BASE_PY), "-c", f"import sys; sys.path.insert(0, {str(wt)!r}); import torch, model; "
                    f"torch.save({{'model_ema': model.ViT('s').state_dict()}}, {str(ck)!r})"], check=True)
    out = tmp_path / "o.json"
    cp = subprocess.run([str(BASE_PY), str(DX.JOB_SCRIPT), str(wt), str(cfg), str(ck), str(out), "--n=4",
                         "--device=cpu"], capture_output=True, text=True, timeout=300)
    assert cp.returncode == 0, cp.stderr[-2000:]
    rep = json.loads(out.read_text())
    a, b = rep["datasets"]["a"], rep["datasets"]["b"]
    assert a["tiles"] == 4 and b["tiles"] == 3 and a["probe_features_dim"] == 12 and a["cls_dim"] == 6
    assert len(a["gates"]) == 1 and 0.0 <= a["gates"][0]["mean"] <= 1.0 and a["gates"][0]["n"] == 4
    assert "THUNDER_V2" in (wt / "probe.py").read_text()                                # probe.py never rewritten
    text = DX.features_text({"incumbent": "x", **rep})
    assert "- a (4 tiles)" in text and "- b (3 tiles)" in text


def test_the_databank_listing_reads_manifests_and_one_dataset_s_files(ctx, store, tmp_path):
    c = _bank_ctx(ctx, tmp_path / "bank")
    _ds(tmp_path / "bank", "alpha", "ready", px=224)
    _ds(tmp_path / "bank", "beta", "building")
    out = DX.databank(c, store)
    assert "2 dataset dir(s)" in out and "- alpha [ready]" in out and "- beta [" in out
    one = DX.databank(c, store, "alpha")
    assert "manifest.json" in one and '"n_tiles"' in one
    assert "no dataset" in DX.databank(c, store, "../etc")


# ---------------------------------------------------------------------------------------------- 5. levers
def _post_axes(gains: dict[str, dict[str, tuple[float, float]]], comp: dict[str, float], monkeypatch):
    """A posterior double: per node per axis (delta, var), its composite delta."""
    est = {n: {M.COMPOSITE: M.Estimate(comp[n], 1e-6, 4, 4)} for n in gains}
    nz = SimpleNamespace(sigma2={a: 1.0 for a in D.AXES})
    by = {n: [n] for n in gains} | {ROOT_NODE_ID: [ROOT_NODE_ID]}
    monkeypatch.setattr(M, "cuped", lambda child, parent, a, noise, w=None: M.Estimate(
        gains[child[0]][a][0], gains[child[0]][a][1], 4, 4) if a in gains[child[0]] else None)
    return SimpleNamespace(est=est, by=by, fit=lambda base, far=False: (nz, {}))


def test_repair_candidates_are_real_axis_gains_screened_out_by_a_loss(ctx, store, monkeypatch):
    nodes = {n: D.Node(n, ROOT_NODE_ID, None, "explore", "f", "explore", ROOT_NODE_ID, 1.0, "screening", "", base="B")
             for n in ("g1", "g2", "g3", "inc")}
    led = SimpleNamespace(nodes=nodes, epoch="B", carriers=(), runs=())
    post = _post_axes({"g1": {"cls": (0.02, 1e-5), "slide": (-0.1, 1e-4)},     # cls z 6.3, slide z -10: candidate
                       "g2": {"cls": (0.02, 1e-5)},                             # no loss, composite > 0: not screened out
                       "g3": {"rob": (0.004, 1e-5)}, "inc": {}},                # rob z 1.3: no real gain
                      {"g1": -0.01, "g2": 0.004, "g3": -0.002, "inc": 0.0}, monkeypatch)
    v = SimpleNamespace(led=led, incumbent="inc", post=post, judged=[])
    rows = LV.repair_candidates(v, store, ctx, {})
    assert [r["node"] for r in rows] == ["g1"] and rows[0]["axis"] == "cls" and "slide" in rows[0]["losses"]
    assert LV.repair_candidates(v, store, ctx, {"g1": "queued"}) == []       # live: not a candidate
    v.judged = [SimpleNamespace(node="g1", switched=True)]
    assert LV.repair_candidates(v, store, ctx, {}) == []                      # passed a fresh-seed test


# ---------------------------------------------------------------------------------------------- 6. diagnostics
def test_each_diagnostic_flag_shows_the_prior_attempts_at_its_mechanism(ctx, store, monkeypatch):
    store.ensure_root("r0")
    for nid, tag in (("a1", "clip_relax"), ("a2", "late_lr_cooldown"), ("a3", "wider_head")):
        store.put_node(Node(id=nid, task_id=store.task_id, commit=f"c-{nid}", parent_id=ROOT_NODE_ID,
                            strategy=Strategy.EXPLORE, operator=Operator.DRAFT, family="optim", mechanism_tag=tag,
                            diff_hash=f"h-{nid}", status=NodeStatus.SCREENED_OUT))
    store.set_state("ideation:a2", {"mechanism_prediction": {"diagnostic": "gn_late_over_clip", "direction": -1}})
    monkeypatch.setattr(DG, "bottlenecks", lambda st, nid: [{"flag": "clip_dominated", "severity": 0.9,
                                                              "text": "clip-dominated", "trend_quantity":
                                                              "gn_late_over_clip"}])
    monkeypatch.setattr(store, "diagnostics", lambda node_id=None: [{"x": 1}] if node_id == "inc" else [])
    nodes = {n: D.Node(n, ROOT_NODE_ID, None, "explore", "optim", "explore", ROOT_NODE_ID, i, "", "")
             for i, n in enumerate(("a1", "a2", "a3", "inc"))}
    led = SimpleNamespace(nodes=nodes, epoch="B", carriers=())
    v = SimpleNamespace(led=led, incumbent="inc", post=None, judged=[], beliefs={"a1": ID.Belief(-0.002, 1e-6, 3)})
    rows = LV.diag_attempts(v, store, ctx)
    assert rows[0]["flag"] == "clip_dominated" and set(rows[0]["recent"]) == {"a1", "a2"}   # tag word / declared
    assert rows[0]["best"] == ("a1", -0.002)
    text = LV.diag_text(rows)
    assert "[G-clip_dominated]" in text and "PRIOR ATTEMPTS" in text and "2 nodes" in text
    assert LV.shortlist(store, ctx, (), [("optim", "clip_relax")])[0] == "optim|clip_relax"   # pre-checked


# ---------------------------------------------------------------------------------------------- 7. check / revert / bounce
class _StateStore:
    def __init__(self, base):
        self.base, self.state = base, {}

    def __getattr__(self, k):
        return getattr(self.base, k)

    def get_state(self, k, default=None):
        return self.state.get(k, default)

    def set_state(self, k, v):
        self.state[k] = v


def test_check_evaluates_a_mechanism_before_any_edit_and_only_the_finish_needs_a_diff(tmp_path):
    from ..test_agents import DirAdapter, MemStore, _ctx, _request
    tpl = tmp_path / "tpl"
    tpl.mkdir()
    (tpl / "fake_model.py").write_text("LR = 1e-3\n")
    (tpl / "fake_config.yaml").write_text("train:\n  lr: 0.001\n")
    (tmp_path / "wt").mkdir()
    adapter, store = DirAdapter(tpl, tmp_path / "wt"), _StateStore(MemStore())
    r = RS.LLMResearcher(_ctx(), None, adapter)
    req = _request(id="req-v3-00001")
    wt = adapter.prepare_worktree(parent_commit="c0", name="w")
    out = r.check(req, store, wt, family="optim", mechanism_tag="cosine_lr", claim="cosine lr")
    assert out.startswith("OK (no diff yet)") and "cosine_lr" in out
    assert store.state[ID.K_CLAIMS]["req-v3-00001"]["tag"] == "cosine_lr"
    assert r.check(req, store, wt).startswith("OK (no diff yet): nothing to check")
    assert r.check(req, store, wt, mechanism_tag="cosine_lr", final=True).startswith("REJECT empty_diff")
    assert r.check(req, store, wt, mechanism_tag="x", final=True, dataset_only=True).startswith("OK")
    # another in-flight session claimed the same mechanism: reserved
    store.state[K_INFLIGHT] = [{"id": "req-v3-00002"}, {"id": "req-v3-00001"}]
    store.state[ID.K_CLAIMS] = {"req-v3-00002": {"family": "optim", "tag": "cosine_lr", "ts": 1.0}}
    assert r.check(req, store, wt, family="optim", mechanism_tag="cosine_lrs").startswith("REJECT reserved_inflight")
    store.state[K_INFLIGHT] = [{"id": "req-v3-00001"}]                         # it finished: free again
    assert r.check(req, store, wt, family="optim", mechanism_tag="cosine_lrs").startswith("OK")


def test_revert_restores_only_the_research_worktree(tmp_path):
    repo, wt = tmp_path / "repo", tmp_path / "wt"
    repo.mkdir()
    g = lambda *a, cwd=repo: subprocess.run(["git", "-C", str(cwd), "-c", "user.name=t", "-c", "user.email=t@t", *a],  # noqa: E731
                                            capture_output=True, text=True, check=True)
    g("init", "-q")
    (repo / "a.py").write_text("A = 1\n")
    (repo / "b.py").write_text("B = 1\n")
    g("add", "-A")
    g("commit", "-qm", "base")
    g("worktree", "add", "-q", str(wt), "HEAD")
    t = RS.WorktreeTools(SimpleNamespace(repo=str(repo)), Worktree(str(wt), "b", "HEAD"), writable=("*",), locked=())
    (wt / "a.py").write_text("A = 2\n")
    (wt / "b.py").write_text("B = 2\n")
    (wt / "new.py").write_text("N = 1\n")
    assert "a.py" in t.t_revert("a.py") and (wt / "a.py").read_text() == "A = 1\n" and (wt / "b.py").read_text() == "B = 2\n"
    out = t.t_revert()
    assert "b.py" in out and "new.py" in out and not (wt / "new.py").exists() and (wt / "b.py").read_text() == "B = 1\n"
    assert (repo / "a.py").read_text() == "A = 1\n" and t.dirty
    main = RS.WorktreeTools(SimpleNamespace(repo=str(repo)), Worktree(str(repo), "b", "HEAD"), writable=("*",), locked=())
    assert main.t_revert().startswith("REFUSED")                                # never the main repo


def test_the_bounce_limit_ends_a_still_rejected_diff_cleanly_without_smoke_or_commit(tmp_path):
    from loop.agents import FakeLLMClient
    from ..test_agents import FINISH, DirAdapter, MemStore, _ctx, _request
    tpl = tmp_path / "tpl"
    tpl.mkdir()
    (tpl / "fake_model.py").write_text("LR = 1e-3\n")
    (tpl / "fake_config.yaml").write_text("train:\n  lr: 0.001\naug:\n  p: 0.5\n")
    (tmp_path / "wt").mkdir()
    adapter, store = DirAdapter(tpl, tmp_path / "wt"), MemStore()
    edit = [("set_config", {"key": "train.lr", "value": 0.0005}), ("set_config", {"key": "aug.p", "value": 0.9})]
    llm = FakeLLMClient({"researcher": [{"tool_calls": edit, "finish": FINISH}]})
    r = RS.LLMResearcher(_ctx(), llm, adapter)
    r.check = lambda *a, **k: "REJECT multi_family: the config keys span families ['augment', 'optim']"
    res = r.research(_request(), store, None)
    assert res.status == "multi_family" and "bounce_limit_reject" in res.error and adapter.smokes == 0
    assert adapter.commits == 0 and res.candidate is not None and res.candidate.commit == ""


# ---------------------------------------------------------------------------------------------- 8. spread
def test_concurrent_repair_sessions_rotate_axes_and_see_what_is_reserved(env, monkeypatch):
    ctx, store, led, ide = env
    assert ID.rotate_axis(["cls", "auc", "rob"], ["cls", "cls", "auc"]) == "rob"
    assert ID.rotate_axis(["cls", "auc"], []) == "cls" and ID.rotate_axis(["cls", "auc"], ["cls"]) == "auc"
    v = ide.view(led)
    first = ide.brief_for("exploit", led, v, "repair")
    store.set_state(K_INFLIGHT, [{"id": "req-v3-09001", "parents": [v.incumbent],
                                  "req": {"brief_kind": "repair", "family_hint": first.axis}}])
    store.set_state(ID.K_CLAIMS, {"req-v3-09001": {"family": "optim", "tag": "warm_restart", "ts": 1.0}})
    second = ide.brief_for("exploit", led, v, "repair")
    ranked = ID.repair_axes(ID.headroom(v, ide.fm, ide.bar_node), v.weights)
    assert len(ranked) == 1 or second.axis != first.axis
    assert "## Reserved: in flight and queued" in second.text and "building optim/warm_restart" in second.text


def test_a_crossover_partner_already_stacked_by_an_in_flight_session_is_reserved(env):
    ctx, store, led, ide = env
    v = ide.view(led)
    p = ide.stack_partner(v)
    if p is None:
        pytest.skip("no stack partner in the sample ledger")
    assert ide.stack_partner(v, {p}) != p


# ---------------------------------------------------------------------------------------------- 9. dataset near-dup
def test_dataset_near_dup_compares_dataset_content_not_the_key(ctx, store, tmp_path):
    c = _bank_ctx(ctx, tmp_path / "bank")
    _ds(tmp_path / "bank", "hi_a", "ready", spec={"seed": 1})
    _ds(tmp_path / "bank", "hi_b", "ready", spec={"seed": 2})
    _ds(tmp_path / "bank", "hi_a_copy", "ready", spec={"seed": 1})               # same content, other name
    root = str(tmp_path / "bank")
    k = IN.DATA_KEY
    assert not IN.inflight_twin(c, store, {k: f"{root}/hi_a"}, {k: f"{root}/hi_b"})
    assert IN.inflight_twin(c, store, {k: f"{root}/hi_a"}, {k: f"{root}/hi_a_copy"})
    assert IN.inflight_twin(c, store, {"train.lr": 1}, {"train.lr": 2})           # other keys: as before
    assert not IN.inflight_twin(c, store, {k: f"{root}/hi_a"}, {k: f"{root}/hi_a", "train.lr": 1})


# ---------------------------------------------------------------------------------------------- 10. web + clock
def test_v3_repair_and_crossover_sessions_get_bounded_web_research(ctx):
    req = CandidateRequest("req-v3-00009", ctx.task_id, Strategy.EXPLOIT, Operator.DRAFT, "main", ("n1",), "b",
                           brief_kind="repair")
    assert RS.wants_web(req, ctx) and RS.bounded_web(req, ctx) and not RS.wants_web(req)
    assert not RS.wants_web(dataclasses.replace(req, ablated=("web",)), ctx)
    assert not RS.bounded_web(dataclasses.replace(req, id="req1"), ctx)               # not a v3 session
    x = dataclasses.replace(req, strategy=Strategy.CROSSOVER, operator=Operator.CROSSOVER, brief_kind="crossover")
    assert RS.bounded_web(x, ctx) and not RS.bounded_web(x, _icfg(ctx, web_kinds=("repair",)))
    assert "{n}" in RS.WEB_BOUNDED and "at most 3 searches" in RS.WEB_BOUNDED.format(n=3)


def test_many_sessions_in_flight_scale_the_time_limit_and_the_smoke_is_off_the_clock(ctx, tmp_path):
    r = RS.LLMResearcher(ctx, None, None)
    req = CandidateRequest("req-v3-1", ctx.task_id, Strategy.EXPLORE, Operator.DRAFT, "main", ("n1",), "b")
    st = _StateStore(SimpleNamespace())
    st.state[K_INFLIGHT] = [{"id": f"r{i}"} for i in range(5)]
    base = r.limits(req)[1]
    assert r.limits(req, st)[1] == base
    st.state[K_INFLIGHT].append({"id": "r6"})
    assert r.limits(req, st)[1] == pytest.approx(base * 1.5)
    st.state[K_INFLIGHT][-1] = {"id": "r6", "queued": True}                         # a queued re-issue: not in flight
    assert r.limits(req, st)[1] == base

    class Ex:
        off_clock = ("smoke",)

        def __call__(self, name, args):
            time.sleep(0.05)
            return "ok"
    gate = LLM.ToolGate(Ex(), {"type": "object"}, time.monotonic() + 10)
    d0 = gate.deadline
    gate("smoke", {})
    assert gate.deadline - d0 >= 0.05 and gate.off_clock_s >= 0.05
    d1 = gate.deadline
    gate("bash", {"command": "x"})
    assert gate.deadline == d1                                                       # other tools: on the clock


def test_a_program_request_may_span_families_at_intake():
    """(3) a program is a MOONSHOT request, so intake's family rule lets its components span families."""
    from loop.contracts import Strategy
    from loop.core.ideation import KIND_REQUEST
    assert KIND_REQUEST["program"][0] is Strategy.MOONSHOT     # admit(strategy=MOONSHOT) spans families:
    # tests/loop/test_select.py::test_moonshot_may_span_families_but_locks_protected_and_init_compat_hold


# ---------------------------------------------------------------------------------------------- v3-c6a item 4
def test_a_bare_component_key_resolves_to_its_flattened_config_key(ctx, store, monkeypatch):
    """n2734: components declared ``gram_patch_weight`` for ``dino.gram_patch_weight`` -- the ladder was recorded
    permanently failed "absent from the config"; the bare key now maps (the node's config_diff keys first)."""
    assert PG.resolve_key("w", {"loss.w": 1.0, "aug.v": 4}) == "loss.w"
    assert PG.resolve_key("w", {"a.w": 1.0, "b.w": 2.0}) == "w"                           # ambiguous: unchanged
    assert PG.resolve_key("w", {"a.w": 1.0, "b.w": 2.0}, ["b.w"]) == "b.w"                # the node's own diff key
    data, post, adapter, built = _prog_env(store, monkeypatch, (0.0005, 4e-6))
    plan = store.get_state("ideation:p1")
    for c in plan["components"]:
        for k in c["knobs"]:
            k["key"] = k["key"].split(".")[-1]                                            # bare: "w", "v"
    store.set_state("ideation:p1", plan)
    store.set_state("config_diff:p1", {"loss.w": 1.0})
    store.set_state(PG.K_LADDERS, {"p1": "-: w absent from the config; v absent from the config"})   # the old bug
    acts = PG.ladders(ctx, store, adapter, data, post, {"p1": "screening"}, 10.0)
    made = store.get_state(PG.K_LADDERS)["p1"]
    assert isinstance(made, list) and len(made) == 4 and any("re-planning" in a for a in acts)
    keys = [next(iter(store.get_state(f"config_diff:{n}"))) for n in made]
    assert keys.count("loss.w") == 3 and keys.count("aug.v") == 1
    store.set_state(PG.K_LADDERS, {"p1": "-: zzz absent from the config"})                # not resolvable: kept
    assert PG.retry_bare_key_failures(store, lambda n: {"loss.w": 1.0}) == []
