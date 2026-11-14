"""Diagnostic layer tests (loop.diagnostics / loop.diag_ckpt / the postprocess + prune hooks). Offline, stdlib."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from loop import diagnostics as DG
from loop import prune as PR
from loop.contracts import ROOT_NODE_ID, JobStatus, Node, NodeStatus, Operator, Run, RunStatus, Stage, Strategy
from loop.driver import RunBook


def feats(*, grms_e=1.0, grms_l=1.0, upd_e=1.0, upd_l=1.0, lr_end=0.02, clip=3.0, gn_late=2.0, clip_frac=0.1,
          drift_e=0.02, drift_l=0.3, st_e=0.001, st_l=0.01, val_min=10.0, val_end=10.05, val_argmin=0.9, depth=12):
    f = {"n_blocks": float(depth), "lr_end_frac": lr_end, "flop_frac_end": 0.98, "clip_grad": clip, "gn_late": gn_late,
         "gn_late_over_clip": gn_late / clip, "gn_clip_frac": clip_frac,
         "val_total_min": val_min, "val_total_end": val_end, "val_total_rise": val_end - val_min,
         "val_total_argmin_frac": val_argmin}
    for i in range(depth):
        w = "early" if i in DG.blocks(depth, "early") else "late" if i in DG.blocks(depth, "late") else "mid"
        f[f"b{i}_grms"] = {"early": grms_e, "mid": (grms_e + grms_l) / 2, "late": grms_l}[w]
        f[f"b{i}_upd"] = {"early": upd_e, "mid": (upd_e + upd_l) / 2, "late": upd_l}[w]
        f[f"b{i}_tdrift"] = {"early": drift_e, "mid": (drift_e + drift_l) / 2, "late": drift_l}[w]
        f[f"b{i}_st"] = {"early": st_e, "mid": (st_e + st_l) / 2, "late": st_l}[w]
    return f


def mk_node(store, nid, parent=ROOT_NODE_ID):
    n = Node(id=nid, task_id=store.task_id, commit="c" + nid, parent_id=parent, strategy=Strategy.EXPLORE,
             operator=Operator.DRAFT, family="f", mechanism_tag="m", diff_hash="h" + nid, status=NodeStatus.SCREENED_OUT,
             updated_ts=1.0)
    store.put_node(n)
    return n


def mk_run(store, nid, seed, stage=Stage.SCREEN, agg=0.65, out="", features=None, status=RunStatus.DONE):
    r = Run(id=Run.make_id(nid, stage, seed), task_id=store.task_id, node_id=nid, seed=seed, stage=stage,
            status=status, aggregate=agg, scores={"alpha": agg, "beta": agg, "gamma": agg}, output_dir=out)
    store.put_run(r)
    if features is not None:
        store.put_diagnostics(r.id, node_id=nid, seed=seed, stage=stage, features=features)
    return r


@pytest.fixture
def root(store):
    return store.ensure_root("c0")


def flags(store, nid):
    return {d["flag"]: d for d in DG.bottlenecks(store, nid)}


# ------------------------------------------------------------------ flags
def test_healthy_run_fires_nothing(store, root):
    mk_node(store, "a")
    mk_run(store, "a", 1, features=feats(grms_e=0.5, grms_l=1.0, upd_e=1.0, upd_l=1.0))
    assert DG.bottlenecks(store, "a") == []
    assert "no bottleneck flag fired" in DG.summary(store, "a")


def test_no_diagnostics_is_empty(store, root):
    mk_node(store, "a")
    assert DG.bottlenecks(store, "a") == [] and DG.summary(store, "a") == "" and DG.node_features(store, "a") is None


def test_update_starved(store, root):
    mk_node(store, "a")
    mk_run(store, "a", 1, features=feats(grms_e=2.0, grms_l=1.0, upd_e=0.01, upd_l=0.4))
    f = flags(store, "a")["update_starved"]
    assert f["numbers"]["upd_late_over_early"] == pytest.approx(40.0) and 0 < f["severity"] <= 1
    mk_node(store, "b")                                    # gradient absent early: a quiet block is not starved
    mk_run(store, "b", 1, features=feats(grms_e=0.1, grms_l=1.0, upd_e=0.01, upd_l=0.4))
    assert "update_starved" not in flags(store, "b")
    mk_node(store, "c")                                    # update not that much smaller
    mk_run(store, "c", 1, features=feats(grms_e=2.0, grms_l=1.0, upd_e=0.3, upd_l=0.4))
    assert "update_starved" not in flags(store, "c")


def throttled(ld=0.8, **kw):
    """The live incumbent's profile: early gradient 2x the top's, early update = the layer-decay multiplier."""
    f = feats(grms_e=2.0, grms_l=1.0, upd_e=0.4 * ld ** 9, upd_l=0.4, drift_e=0.07, drift_l=0.4, **kw)
    return {**f, "layerwise_decay": ld, "pos_embed_drift": 0.3, "cls_token_drift": 0.1}


def test_lr_throttled_when_the_layer_decay_multiplier_explains_the_starved_update(store, root):
    mk_node(store, "a")
    mk_run(store, "a", 1, features=throttled())
    fl = flags(store, "a")
    f = fl["lr_throttled"]
    assert "update_starved" not in fl                      # the explained special case replaces it
    assert f["numbers"]["lr_mult_early_late"] == pytest.approx(0.8 ** 9, rel=1e-5)
    assert f["numbers"]["upd_ratio_early_late"] == pytest.approx(0.8 ** 9, rel=1e-5)
    assert f["numbers"]["layerwise_decay"] == 0.8 and "knob layerwise_decay" in f["text"]
    lrs = {f"b{i}_lr_end": 0.8 ** (11 - i) for i in range(12)}     # the final per-block LRs win over the formula
    assert DG.quantity({**throttled(ld=0.5), **lrs}, "lr_mult_early_late") == pytest.approx(0.8 ** 9)
    mk_node(store, "b")                                    # the update is far below what the multiplier explains
    mk_run(store, "b", 1, features={**throttled(), "layerwise_decay": 1.0})
    assert "lr_throttled" not in flags(store, "b") and "update_starved" in flags(store, "b")
    mk_node(store, "c")                                    # no decay, a near-even update: nothing
    mk_run(store, "c", 1, features=throttled(ld=1.0))
    assert not {"lr_throttled", "update_starved"} & set(flags(store, "c"))
    text = DG.summary(store, "a")
    assert "blocks early / late: grad RMS x2, update/weight x0.134, LR multiplier x0.134" in text
    assert "pos_embed 0.3, cls_token 0.1" in text and "[lr_throttled" in text


def test_a_low_level_null_closes_under_tuned_on_layer_decay_when_lr_throttled(store, root):
    from types import SimpleNamespace
    mk_node(store, "p")
    mk_run(store, "p", 1, features=throttled())
    for nid, notes in (("a", "init=timm/x"), ("b", "mtype=augmentation"), ("c", "mtype=loss_objective")):
        mk_node(store, nid, parent="p")
        store.update_node(nid, notes=notes)
    assert "under_tuned (knob layerwise_decay)" in DG.under_tuned(store.get_node("a"), store)
    assert "init swap" in DG.under_tuned(store.get_node("a"), store) and "its parent p's" in DG.explain(
        store.get_node("a"), store)
    assert "mechanism type augmentation" in DG.under_tuned(store.get_node("b"), store)
    assert DG.under_tuned(store.get_node("c"), store) == ""                    # not a low-level change
    harm = SimpleNamespace(nodes={"a": SimpleNamespace(gain_p=0.01, p_gain=0.01)})
    assert DG.under_tuned(store.get_node("a"), store, harm) == ""             # a clear harm is not a null
    store.update_node("a", status=NodeStatus.CONFIRMED)
    assert DG.under_tuned(store.get_node("a"), store) == ""
    mk_run(store, "b", 1, features=throttled(ld=1.0))                         # its own runs: not throttled
    assert DG.under_tuned(store.get_node("b"), store) == ""


def test_bank_writes_the_computed_ideas_once_per_incumbent(ctx, store, root):
    from types import SimpleNamespace
    from loop.select import bank as BK
    mk_node(store, "a")
    mk_run(store, "a", 1, features=throttled())
    snap = SimpleNamespace(incumbent_id="a")
    rows = DG.bank_rows(ctx, store, "a")
    assert [r["id"] for r in rows] == ["diag-token_params_lowest_lr-a", "diag-layer_decay_anneal-a"]
    assert all(r["plan"]["campaign"] and r["priority"] for r in rows)
    assert "position embedding 30.0%" in rows[0]["why_it_might_work"] and "class token" not in rows[0][
        "why_it_might_work"]                               # only tokens drifting > TOKEN_DRIFT_K x the early blocks
    acts = DG.bank(ctx, store, snap)
    assert acts and "diag-layer_decay_anneal-a" in acts[0] and DG.bank(ctx, store, snap) == []
    served = {e["id"]: e for e in BK.pending(store, ctx.cfg)}
    assert set(served) >= {r["id"] for r in rows}
    assert served[rows[0]["id"]]["why_it_might_work"]      # nothing scrubbed as a quoted score
    mk_node(store, "b")
    mk_run(store, "b", 1, features=throttled(ld=1.0))
    assert DG.bank(ctx, store, SimpleNamespace(incumbent_id="b")) == []          # no flag, no idea


def test_schedule_not_annealed(store, root):
    mk_node(store, "a")
    mk_run(store, "a", 1, features=feats(lr_end=0.89))
    assert flags(store, "a")["schedule_not_annealed"]["numbers"]["lr_end_frac"] == pytest.approx(0.89)
    mk_node(store, "b")
    mk_run(store, "b", 1, features=feats(lr_end=DG.ANNEAL_MAX_FRAC - 0.01))
    assert "schedule_not_annealed" not in flags(store, "b")


def test_clip_dominated_by_ratio_or_fraction(store, root):
    mk_node(store, "a")
    mk_run(store, "a", 1, features=feats(gn_late=7.0, clip=3.0, clip_frac=0.2))
    assert "clip_dominated" in flags(store, "a")
    mk_node(store, "b")
    mk_run(store, "b", 1, features=feats(gn_late=2.0, clip=3.0, clip_frac=0.9))
    assert "clip_dominated" in flags(store, "b")
    mk_node(store, "c")
    mk_run(store, "c", 1, features=feats(gn_late=2.0, clip=3.0, clip_frac=0.1))
    assert "clip_dominated" not in flags(store, "c")


def test_teacher_lag(store, root):
    mk_node(store, "a")
    mk_run(store, "a", 1, features=feats(drift_l=0.1, st_l=0.05))
    assert flags(store, "a")["teacher_lag"]["numbers"]["st_over_drift_late"] == pytest.approx(0.5)
    mk_node(store, "b")
    mk_run(store, "b", 1, features=feats(drift_l=0.3, st_l=0.01))
    assert "teacher_lag" not in flags(store, "b")


def test_val_loss_rising_late(store, root):
    mk_node(store, "a")
    mk_run(store, "a", 1, features=feats(val_min=10.0, val_end=10.5, val_argmin=0.4))
    assert "val_loss_rising_late" in flags(store, "a")
    mk_node(store, "b")                                    # minimum is late: not "early min then rise"
    mk_run(store, "b", 1, features=feats(val_min=10.0, val_end=10.5, val_argmin=0.95))
    assert "val_loss_rising_late" not in flags(store, "b")
    mk_node(store, "c")                                    # a rise within the margin
    mk_run(store, "c", 1, features=feats(val_min=10.0, val_end=10.05, val_argmin=0.4))
    assert "val_loss_rising_late" not in flags(store, "c")


def test_flags_average_over_runs_and_trend_walks_the_chain(store, root):
    mk_node(store, "a")
    mk_node(store, "b", parent="a")
    mk_run(store, ROOT_NODE_ID, 1, stage=Stage.BASELINE, features=feats(lr_end=0.9))
    mk_run(store, "a", 1, features=feats(lr_end=0.8))
    mk_run(store, "b", 1, features=feats(lr_end=0.95))
    mk_run(store, "b", 2, features=feats(lr_end=0.55))      # mean over b's two runs = 0.75
    f = flags(store, "b")["schedule_not_annealed"]
    assert f["numbers"]["lr_end_frac"] == pytest.approx(0.75) and f["n_runs"] == 2
    assert [n for n, _ in f["trend"]] == [ROOT_NODE_ID, "a", "b"]
    assert [v for _, v in f["trend"]] == pytest.approx([0.9, 0.8, 0.75])
    text = DG.summary(store, "b")
    assert "schedule_not_annealed" in text and "root:0.9 -> a:0.8 -> b:0.75" in text and text.count("\n") >= 1
    assert DG.node_features(store, "b")["lr_end_frac"] == pytest.approx(0.75)
    assert set(DG.node_features(store, "b")) == set(DG.NODE_FEATURES)


def test_quantities_table(store):
    q = DG.quantities()
    for k in ("drift_early", "drift_mid", "drift_late", "upd_early", "upd_late", "grms_early", "grms_late", "st_early",
              "lr_end_frac", "gn_late_over_clip", "upd_ratio_early_late"):
        assert k in q and q[k]
    f = feats(upd_e=0.1, upd_l=0.4)
    assert DG.quantity(f, "upd_ratio_early_late") == pytest.approx(0.25)
    assert DG.quantity(f, "b0_upd") == pytest.approx(0.1) and DG.quantity(f, "nope") is None


# ------------------------------------------------------------------ mechanism_check
def pair_up(store, nid, parent, shifts, *, node_gain=0.0, noise=(0.0,) * 6, parent_stage=Stage.PARENT):
    """Seeds 1..n: parent upd_early 0.10, node 0.10 + shift; panel node = parent + node_gain + noise."""
    for s, (sh, nz) in enumerate(zip(shifts, noise), start=1):
        mk_run(store, parent, s, stage=parent_stage, agg=0.65, features=feats(upd_e=0.10))
        mk_run(store, nid, s, stage=Stage.SCREEN if s % 2 else Stage.PROMOTE, agg=0.65 + node_gain + nz,
               features=feats(upd_e=0.10 + sh))


def test_mechanism_moved_and_gained(store, root):
    mk_node(store, "p")
    mk_node(store, "a", parent="p")
    pair_up(store, "a", "p", [0.05, 0.06, 0.05, 0.055], node_gain=0.01, noise=(0.001, -0.001, 0.0005, -0.0005))
    r = DG.mechanism_check(store, "a", "upd_early", +1)
    assert r["verdict"] == "moved+gained" and r["n"] == 4 and r["moved"] and r["gained"]
    assert r["mean_delta"] == pytest.approx(0.05375) and r["panel_mean"] == pytest.approx(0.01)
    assert DG.mechanism_check(store, "a", "upd_early", -1)["verdict"] == "didn't-move"   # wrong direction


def test_mechanism_moved_no_gain_and_didnt_move(store, root):
    mk_node(store, "p")
    mk_node(store, "a", parent="p")
    pair_up(store, "a", "p", [0.05, 0.06, 0.05, 0.055], node_gain=-0.004, noise=(0.001, -0.001, 0.0005, -0.0005))
    assert DG.mechanism_check(store, "a", "upd_early", +1)["verdict"] == "moved-no-gain"
    mk_node(store, "q")
    mk_node(store, "b", parent="q")
    pair_up(store, "b", "q", [0.001, -0.002, 0.0015, -0.001], node_gain=0.02)
    r = DG.mechanism_check(store, "b", "upd_early", +1)
    assert r["verdict"] == "didn't-move" and not r["moved"]


def test_mechanism_one_pair_uses_relative_threshold_and_root_parent(store, root):
    mk_node(store, "a")                                    # a root child: paired against root's run at the seed
    mk_run(store, ROOT_NODE_ID, 7, stage=Stage.BASELINE, agg=0.65, features=feats(upd_e=0.10))
    mk_run(store, "a", 7, agg=0.66, features=feats(upd_e=0.15))
    r = DG.mechanism_check(store, "a", "upd_early", +1)
    assert r["n"] == 1 and r["se"] is None and r["verdict"] == "moved+gained"   # +50% moved; 0.01 > 0.0033


def test_mechanism_no_data_and_unknown(store, root):
    mk_node(store, "a")
    assert DG.mechanism_check(store, "a", "upd_early", +1)["verdict"] == "no-data"
    mk_run(store, ROOT_NODE_ID, 7, stage=Stage.BASELINE, features=feats())
    mk_run(store, "a", 7, features=feats())
    assert DG.mechanism_check(store, "a", "no_such_quantity", +1)["verdict"] == "unknown-quantity"
    assert DG.mechanism_check(store, "ghost", "upd_early", +1)["verdict"] == "no-data"


def test_mechanism_pairs_need_both_sides_diagnosed(store, root):
    mk_node(store, "p")
    mk_node(store, "a", parent="p")
    mk_run(store, "p", 1, stage=Stage.PARENT, features=feats(upd_e=0.1))
    mk_run(store, "a", 1, features=feats(upd_e=0.2))
    mk_run(store, "p", 2, stage=Stage.PARENT, features=None)          # parent seed 2 has no diagnostics
    mk_run(store, "a", 2, features=feats(upd_e=0.2))
    mk_run(store, "a", 3, features=feats(upd_e=0.2))                   # no parent run at seed 3
    assert DG.mechanism_check(store, "a", "upd_early", +1)["n"] == 1


# ------------------------------------------------------------------ ingest / launch / pending
def write_result(ctx, run_id, features, **extra):
    out, _, _ = DG.paths(ctx, run_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"run_id": run_id, "features": features, "version": 1, "secs": 1.0, **extra}))
    return out


def test_ingest_inserts_finished_json_once(ctx, store, root):
    mk_node(store, "a")
    r = mk_run(store, "a", 1)
    write_result(ctx, r.id, {"lr_end_frac": 0.5})
    write_result(ctx, "a:screen:99", {"lr_end_frac": 0.5})             # a run the store does not know: skipped
    write_result(ctx, "a:screen:98", {})                               # no features: skipped
    (DG.diag_dir(ctx) / "garbage.json").write_text("{not json")
    assert DG.ingest(ctx, store) == 1 and DG.ingest(ctx, store) == 0
    (row,) = store.diagnostics(node_id="a")
    assert row["run_id"] == r.id and row["seed"] == 1 and row["stage"] == "screen" and row["features"] == {"lr_end_frac": 0.5}
    assert DG.ingest(ctx, store) == 0


def test_pending_launched_until_output_or_timeout(ctx, tmp_path):
    r = Run(id="a:screen:1", task_id="t", node_id="a", seed=1, stage=Stage.SCREEN, status=RunStatus.DONE)
    out, marker, _ = DG.paths(ctx, r.id)
    assert not DG.pending(ctx, r)                                       # never launched
    marker.parent.mkdir(parents=True)
    marker.write_text("x")
    assert DG.pending(ctx, r)
    assert not DG.pending(ctx, r, now=time.time() + 3600 * ctx.cfg.diagnostics.pending_h + 5)   # abandoned
    out.write_text("{}")
    assert not DG.pending(ctx, r)                                       # produced


def test_launch_runs_detached_low_priority(ctx, tmp_path, monkeypatch):
    rd = tmp_path / "run"
    rd.mkdir()
    (rd / "latest.pt").write_bytes(b"x")
    r = Run(id="a:screen:1", task_id="t", node_id="a", seed=1, stage=Stage.SCREEN, status=RunStatus.DONE,
            output_dir=str(rd))
    seen = {}

    class P:
        pid = 4242

        def poll(self):
            return None

    def fake_popen(argv, **kw):
        seen.update(argv=argv, **kw)
        return P()
    monkeypatch.setattr(DG.subprocess, "Popen", fake_popen)
    assert DG.launch(ctx, r, "/py/python", "/init.pth") == 4242
    a = seen["argv"]
    assert a[:3] == ["nice", "-n", "19"] and a[3] == "/py/python" and "--run-dir" in a and str(rd) in a
    assert a[a.index("--init") + 1] == "/init.pth" and a[a.index("--run-id") + 1] == r.id
    assert seen["start_new_session"] and int(seen["env"]["OMP_NUM_THREADS"]) <= 4
    assert DG.pending(ctx, r) and DG.launch(ctx, r, "/py/python", "/init.pth") is None   # not launched twice
    DG._PROCS.clear()
    r2 = Run(id="a:screen:2", task_id="t", node_id="a", seed=2, stage=Stage.SCREEN, status=RunStatus.DONE,
             output_dir=str(tmp_path / "gone"))
    assert DG.launch(ctx, r2, "/py/python", "") is None                 # no checkpoint: nothing to extract


def test_drift_init_is_the_runs_own_model_type_checkpoint(tmp_path):
    """An init swap: drift is measured from the checkpoint the node's model.py declares for the run's model.type
    (torch hub cache), never from the root's init file; unknown / uncached -> no drift; no node code -> --init."""
    import contextlib
    import importlib.util
    from types import SimpleNamespace
    spec = importlib.util.spec_from_file_location("diag_ckpt_under_test", DG.SCRIPT)
    dc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dc)
    code = tmp_path / "node"
    code.mkdir()
    (code / "model.py").write_text("class ViT:\n    def __init__(self, variant):\n        self.pretrained_url = "
                                   "{'s': 'https://x/s_reg4.pth', 'b': 'https://x/b_reg4.pth?dl=1'}[variant]\n")
    (tmp_path / "hub" / "checkpoints").mkdir(parents=True)
    (tmp_path / "hub" / "checkpoints" / "b_reg4.pth").write_text("w")
    torch = SimpleNamespace(device=lambda _: contextlib.nullcontext(), hub=SimpleNamespace(get_dir=lambda: str(tmp_path / "hub")))
    cfg = lambda t: {"model": {"type": t}}                               # noqa: E731
    assert dc.resolve_init(torch, str(code), cfg("b"), "/root_init.pth") == str(tmp_path / "hub" / "checkpoints" / "b_reg4.pth")
    assert dc.resolve_init(torch, str(code), cfg("s"), "/root_init.pth") == ""          # declared, not cached
    assert dc.resolve_init(torch, str(code), cfg("unknown"), "/root_init.pth") == ""    # not constructible
    assert dc.resolve_init(torch, str(tmp_path / "gone"), cfg("b"), "/root_init.pth") == "/root_init.pth"
    argv = DG.command("py", tmp_path, "/root_init.pth", tmp_path / "o.json", "r", 2, str(code))
    assert argv[argv.index("--model-dir") + 1] == str(code) and "--model-dir" not in DG.command("py", tmp_path, "", tmp_path, "r", 2)


# ------------------------------------------------------------------ hooks
def test_prune_keeps_a_pending_checkpoint_until_timeout(ctx, store, root):
    mk_node(store, "a")
    d = Path(ctx.state_dir).parent / "runs" / "a"
    d.mkdir(parents=True)
    ckpt = d / "latest.pt"
    ckpt.write_bytes(b"x" * 10)
    r = mk_run(store, "a", 1, out=str(d))
    now = time.time()
    drop, _ = PR.plan(ctx, store, None, now)
    assert [p for p, _ in drop] == [ckpt]                               # nothing pending: it goes
    _, marker, _ = DG.paths(ctx, r.id)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("x")
    drop, kept = PR.plan(ctx, store, None, now)
    assert drop == [] and kept == 1                                     # extraction in flight: kept
    drop, _ = PR.plan(ctx, store, None, now + 3600 * ctx.cfg.diagnostics.pending_h + 60)
    assert [p for p, _ in drop] == [ckpt]                               # bounded: never blocks forever
    write_result(ctx, r.id, {"lr_end_frac": 0.1})
    drop, _ = PR.plan(ctx, store, None, now)
    assert [p for p, _ in drop] == [ckpt]                               # done: it goes


class _Adapter:
    def __init__(self, post=None):
        self.seen = []
        if post is not None:
            self.postprocess = post

    def submit(self, *, run, commit):
        from loop.contracts import JobHandle
        return JobHandle(run.id, "1", f"/o/{run.id}", 0.0)

    def poll(self, handle):
        return JobStatus(RunStatus.DONE, scores={"alpha": 0.6, "beta": 0.6, "gamma": 0.6}, aggregate=0.6, gpu_h=1.0)

    def cancel(self, handle):
        pass


def _launch_one(ctx, store, adapter):
    book = RunBook(ctx, store, adapter)
    n = mk_node(store, "a")
    store.update_node("a", status=NodeStatus.SCREENING)
    assert book.launch(n, Stage.SCREEN, 1) is not None
    return book


def test_postprocess_hook_called_on_done(ctx, store, root):
    calls = []
    book = _launch_one(ctx, store, _Adapter(lambda run: calls.append((run.id, run.status))))
    assert [r.id for r in book.poll()] == ["a:screen:1"]
    assert calls == [("a:screen:1", RunStatus.DONE)]


def test_postprocess_error_is_logged_not_raised(ctx, store, root):
    def boom(run):
        raise RuntimeError("no torch")
    book = _launch_one(ctx, store, _Adapter(boom))
    assert len(book.poll()) == 1
    (ev,) = store.events("diagnostics_error")
    assert ev["run_id"] == "a:screen:1" and "no torch" in ev["error"]


def test_adapter_without_postprocess_is_untouched(ctx, store, root):
    book = _launch_one(ctx, store, _Adapter())
    assert len(book.poll()) == 1 and store.events("diagnostics_error") == []


def test_nanopath_adapter_postprocess_launches_with_cfg_python(monkeypatch, tmp_path):
    import dataclasses
    from loop.adapters.nanopath_v2 import NanopathV2Adapter
    from loop.spec import DiagnosticsCfg, load_context
    repo = Path(__file__).resolve().parents[2]
    c = load_context(repo / "tasks" / "nanopath_v2.yaml", state_dir=tmp_path / "s", require_measured=False)
    assert c.cfg.diagnostics.enabled and c.cfg.diagnostics.python and c.cfg.diagnostics.init
    c = dataclasses.replace(c, cfg=dataclasses.replace(c.cfg, diagnostics=DiagnosticsCfg(enabled=True, python="/p/python", init="/i.pth")))
    ad = NanopathV2Adapter(c, runner=object(), runs_root=str(tmp_path / "r"), worktrees_root=str(tmp_path / "w"))
    launched = []
    monkeypatch.setattr(DG, "launch", lambda ctx, run, py, init, code="": launched.append((run.id, py, init, code)))
    d = tmp_path / "out"
    d.mkdir()
    run = Run(id="a:screen:1", task_id=c.task_id, node_id="a", seed=1, stage=Stage.SCREEN, status=RunStatus.DONE,
              output_dir=str(d))
    ad.postprocess(run)
    assert launched == []                                               # no latest.pt: nothing to do
    (d / "latest.pt").write_bytes(b"x")
    ad.postprocess(run)
    assert launched == [("a:screen:1", "/p/python", "/i.pth", "")]          # no node code: the configured init
    (ad.worktrees / "a").mkdir(parents=True)
    (ad.worktrees / "a" / "model.py").write_text("")
    launched.clear()
    ad.postprocess(run)
    assert launched == [("a:screen:1", "/p/python", "/i.pth", str(ad.worktrees / "a"))]   # its model.type's init
    ad2 = NanopathV2Adapter(dataclasses.replace(c, cfg=dataclasses.replace(c.cfg, diagnostics=DiagnosticsCfg(enabled=False))),
                            runner=object(), runs_root=str(tmp_path / "r"), worktrees_root=str(tmp_path / "w"))
    ad2.postprocess(run)
    assert len(launched) == 1


# ------------------------------------------------------------------ the standalone script
def _metrics(path: Path):
    lines = []
    for i in range(1, 41):
        lines.append({"step": i * 20, "lr": 1e-4 * min(1.0, i / 4) * (1 - 0.1 * i / 40), "grad_norm": 5.0 if i > 30 else 2.0,
                      "flop_fraction": i / 40})
        if i % 5 == 0:
            lines.append({"step": i * 20, "val_total": 10.0 + (0.1 * abs(i - 15)), "val_dino": 9.0})
    lines.append({"event": "probe", "step": 800, "final_score": 0.6})
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")


def test_diag_ckpt_metrics_only_without_a_checkpoint(tmp_path):
    rd = tmp_path / "run"
    rd.mkdir()
    _metrics(rd / "metrics.jsonl")
    out = tmp_path / "o.json"
    rc = subprocess.run([sys.executable, str(DG.SCRIPT), "--run-dir", str(rd), "--out", str(out), "--run-id", "x:y:1"],
                        capture_output=True, text=True).returncode
    doc = json.loads(out.read_text())
    f = doc["features"]
    assert rc == 0 and doc["version"] == 1 and doc["run_id"] == "x:y:1" and "secs" in doc
    assert f["lr_end_frac"] == pytest.approx(0.9 / 1.0, rel=0.02) and f["flop_frac_end"] == 1.0
    assert f["gn_late"] == pytest.approx(5.0) and f["gn_early"] == pytest.approx(2.0)
    assert f["val_total_argmin_frac"] == pytest.approx(15 / 40) and f["val_total_rise"] > 0
    assert "gn_late_over_clip" not in f                                 # no checkpoint -> no clip threshold: omitted


def test_diag_ckpt_on_a_tiny_checkpoint(tmp_path):
    py = os.environ.get("DIAG_TORCH_PYTHON") or "/home/user/nanopath/base/.venv/bin/python"
    if not Path(py).is_file():
        pytest.skip("no torch python")
    prog = tmp_path / "mk.py"
    prog.write_text('''
import math, sys, torch
d, depth, w = sys.argv[1], 4, 8
def sd(scale):
    g = torch.Generator().manual_seed(0)
    out = {}
    for i in range(depth):
        for k, sh in (("attn.qkv.weight", (3 * w, w)), ("attn.proj.weight", (w, w)), ("mlp.fc1.weight", (4 * w, w)), ("mlp.fc2.weight", (w, 4 * w))):
            out[f"blocks.{i}.{k}"] = torch.randn(*sh, generator=g) * (1 + scale)
    out["pos_embed"] = torch.randn(1, 5, w, generator=g) * (1 + scale)
    return out
W0, S, T = sd(0.0), sd(0.05), sd(0.02)
torch.save(W0, d + "/init.pth")
names = [(f"blocks.{i}.{k}") for i in range(depth) for k in ("attn.qkv.weight", "attn.proj.weight", "mlp.fc1.weight", "mlp.fc2.weight")]
ld = 0.7
state, groups, pid = {}, [], 0
for i in range(depth):
    ps = []
    for k in ("attn.qkv.weight", "attn.proj.weight", "mlp.fc1.weight", "mlp.fc2.weight"):
        n = S[f"blocks.{i}.{k}"]
        state[pid] = {"step": torch.tensor(100.0), "exp_avg": torch.full_like(n, 0.01), "exp_avg_sq": torch.full_like(n, 1e-4)}
        ps.append(pid); pid += 1
    lm = ld ** (depth - 1 - i)
    groups.append({"params": ps, "lr_mult": lm, "wd_mult": 1.0, "last_layer": False, "lr": 1e-3 * lm, "betas": (0.9, 0.999), "eps": 1e-8})
groups.append({"params": [pid], "lr_mult": 1.0, "wd_mult": 0.0, "last_layer": False, "lr": 1e-3})
state[pid] = {"step": torch.tensor(100.0), "exp_avg": torch.zeros(3), "exp_avg_sq": torch.zeros(3)}
torch.save({"model": S, "model_ema": T, "opt": {"state": state, "param_groups": groups}, "step": 100, "train_flops": 1.0,
            "examples_seen": 10, "config": {"dino": {"layerwise_decay": ld, "clip_grad": 3.0, "adam_beta2": 0.999, "lr": 1e-3}}}, d + "/latest.pt")
''')
    run = tmp_path / "run"
    run.mkdir()
    _metrics(run / "metrics.jsonl")
    assert subprocess.run([py, str(prog), str(run)], capture_output=True, text=True).returncode == 0
    shutil.move(str(run / "init.pth"), str(tmp_path / "init.pth"))
    out = tmp_path / "o.json"
    p = subprocess.run([py, str(DG.SCRIPT), "--run-dir", str(run), "--init", str(tmp_path / "init.pth"), "--out", str(out),
                        "--threads", "2"], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    f = json.loads(out.read_text())["features"]
    assert f["n_blocks"] == 4 and f["clip_grad"] == 3.0 and f["gn_late_over_clip"] == pytest.approx(5.0 / 3.0)
    for i in range(4):
        assert 0 < f[f"b{i}_tdrift"] < f[f"b{i}_st"] + 0.1 and f[f"b{i}_st"] > 0 and f[f"b{i}_grms"] == pytest.approx((1e-4 / (1 - 0.999 ** 100)) ** 0.5)
    assert f["b3_lr_end"] == pytest.approx(1e-3) and f["b0_lr_end"] == pytest.approx(1e-3 * 0.7 ** 3)
    assert f["b0_upd"] < f["b3_upd"]                                    # layerwise decay: the early block moves less
    assert "b0_qkv_erank_d" in f and "pos_embed_drift" in f
    assert DG.quantity(f, "upd_ratio_early_late") < 1.0
