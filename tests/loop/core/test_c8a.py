"""v3-c8a: eval-only arms / ladders both directions (B1), dead dataset lanes (B2), mixed changes split (B3), ladders
isolate every component (B4), smoke-dir + node-worktree housekeeping (B6)."""

from __future__ import annotations

import dataclasses
import os
import time
from types import SimpleNamespace

import pytest

from loop import prune as PR
from loop.contracts import K_INFLIGHT, Node, NodeStatus, Operator, ROOT_NODE_ID, Run, RunStatus, Stage, Strategy
from loop.core import data as D
from loop.core import datasets as DS
from loop.core import mixed as MX
from loop.core import program as PG

from .test_c6i import _icfg
from .test_ideation import env  # noqa: F401 - fixture


# ---------------------------------------------------------------------------------------------- B1 both directions
def test_mirror_is_x2_x_half_geometric_and_bounded_knobs_stay_inside_0_1():
    assert PG.mirror(1.0, 2.0) == 0.5 and PG.mirror(1.0, 0.5) == 2.0
    assert PG.mirror(10, 14) == pytest.approx(7.14286)
    assert PG.mirror(0.4, 0.1, "dino.mask_ratio") == 0.7          # 1.6 would leave (0, 1): halfway to 1
    assert PG.mirror(0.0, 0.3) == -0.3 and PG.mirror(1.0, 1.0) is None and PG.mirror("a", 1) is None


def test_symmetric_doses_sit_on_both_sides_of_the_nodes_value():
    assert PG.symmetric_doses(8, 14, 4, "lin", 10) == [9.0, 8.0, 12.0, 14.0]
    lo_end = PG.symmetric_doses(4, 8, 4, "lin", 4)                  # the declared range starts AT the value
    assert lo_end and all(4 < x <= 8 for x in lo_end)               # v3 review M5: never mirrored past the declared range
    mid = PG.symmetric_doses(2, 8, 4, "log", 4)                     # a range on both sides: both sides get doses
    assert [x for x in mid if x < 4] and [x for x in mid if x > 4]
    xs = PG.symmetric_doses(0.25, 4.0, 2, "log", 1.0)
    assert xs == [0.25, 4.0]
    assert PG.symmetric_doses(0.25, 4.0, 1, "log", 1.0) == PG.doses(0.25, 4.0, 1, "log", 1.0)   # one rung: as before


def _prog(store, cfgs, belief=(0.0005, 4e-6), comps=None):
    store.ensure_root("r0")
    store.put_node(Node(id="p1", task_id=store.task_id, commit="c-p1", parent_id=ROOT_NODE_ID,
                        strategy=Strategy.MOONSHOT, operator=Operator.DRAFT, family="augmentation",
                        mechanism_tag="prog", diff_hash="h-p1", status=NodeStatus.SCREENING))
    comps = comps or [{"family": "augmentation", "mechanism": "views", "share": 1.0,
                       "knobs": [{"key": "train.local_views", "lo": 8, "hi": 14}]}]
    store.set_state("ideation:p1", {"components": comps, "knobs": comps[0]["knobs"]})
    nodes = {"p1": D.Node("p1", ROOT_NODE_ID, None, "moonshot", "augmentation", "explore", ROOT_NODE_ID, 1.0,
                          "screening", "request=r1 brief=program", base="B")}
    data = SimpleNamespace(nodes=nodes, port=lambda n, b=None: n, epoch="B")
    post = SimpleNamespace(est={"p1": {}}, belief=lambda n: belief)
    built = []
    adapter = SimpleNamespace(config_rel="config.yaml", config_at=lambda c: cfgs.get(c, cfgs["c-p1"]),
                              dry_build=lambda commit, levels, knobs, diffs, name: built.append(dict(levels)) or
                              (f"sha-{len(built)}", ""))
    return data, post, adapter, built


def test_a_program_ladder_doses_both_sides_and_isolates_the_undosed_crop_change(ctx, store, monkeypatch):
    """n2821: its views ladder carried the crop-scale change [.4, 1] -> [.3, 1] in every point (never isolated)."""
    monkeypatch.setattr("loop.extrapolate._allowed", lambda c, k: True)
    cfgs = {"r0": {"train": {"local_views": 14, "global_views": 2, "num_workers": 8},
                   "data": {"global_crop_scale": [0.4, 1.0]}},
            "c-p1": {"train": {"local_views": 10, "global_views": 4, "num_workers": 12},
                     "data": {"global_crop_scale": [0.3, 1.0]}}}
    c = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, intake=dataclasses.replace(
        ctx.cfg.intake, lever_groups={"views": ("train.*_views",), "crops": ("data.*crop_scale",)})))
    data, post, adapter, built = _prog(store, cfgs, belief=(0.01, 4e-6))          # screens positive
    acts = PG.ladders(c, store, adapter, data, post, {"p1": "screening"}, 10.0)
    made = store.get_state(PG.K_LADDERS)["p1"]
    vals = sorted(store.get_state(f"config_diff:{n}")["train.local_views"] for n in made)
    # v3 review S6: the 2 isolation points are part of the ladder's 5-point budget (itself + 2 doses + 2 isolations)
    assert len(made) == 2 and any(v < 10 for v in vals) and any(v > 10 for v in vals), vals
    assert store.get_state(PG.K_ISO) is None                                       # not built with the ladder ...
    acts += PG.ladders(c, store, adapter, data, post, {"p1": "screening"}, 15.0)   # ... but once it screens positive
    iso = store.get_state(PG.K_ISO)["p1"]
    assert set(iso) == {"data.global_crop_scale", "train.global_views"}          # num_workers: neutral
    crop = iso["data.global_crop_scale"]
    assert store.get_state(f"config_diff:{crop}") == {"data.global_crop_scale": [0.4, 1.0]}   # reverted ONLY it
    assert "loo_key=data.global_crop_scale" in store.get_node(crop).notes and any("isolation" in a for a in acts)
    PG.ladders(c, store, adapter, data, post, {"p1": "screening"}, 20.0)
    assert store.get_state(PG.K_ISO)["p1"] == iso                                 # once


def test_isolation_respects_the_loo_cap_largest_change_first(ctx, store, monkeypatch):
    monkeypatch.setattr("loop.extrapolate._allowed", lambda c, k: True)
    cfgs = {"r0": {"a": {"x": 1.0, "y": 1.0, "z": 1.0, "w": 1.0}, "train": {"local_views": 14}},
            "c-p1": {"a": {"x": 8.0, "y": 1.1, "z": 2.0, "w": 4.0}, "train": {"local_views": 10}}}
    data, post, adapter, built = _prog(store, cfgs, belief=(0.01, 4e-6))
    PG.ladders(_icfg(ctx, program_loo_max=2), store, adapter, data, post, {"p1": "screening"}, 10.0)
    PG.ladders(_icfg(ctx, program_loo_max=2), store, adapter, data, post, {"p1": "screening"}, 11.0)
    iso = store.get_state(PG.K_ISO)["p1"]
    kept = [k for k, v in iso.items() if not v.startswith("-")]
    assert set(kept) == {"a.x", "a.w"} and iso["a.z"].startswith("-: over the LOO budget")


def test_a_harmful_one_sided_ladder_dose_gets_its_mirrored_dose_once(ctx, store, monkeypatch):
    monkeypatch.setattr("loop.extrapolate._allowed", lambda c, k: True)
    cfgs = {"c-p1": {"loss": {"w": 1.0}}}
    comps = [{"family": "objective", "mechanism": "w", "share": 1.0,
              "knobs": [{"key": "loss.w", "lo": 2.0, "hi": 4.0, "scale": "log"}]}]
    data, post, adapter, built = _prog(store, cfgs, comps=comps)
    PG.ladders(_icfg(ctx, program_wave_max_points=2, program_ladder_symmetric=False), store, adapter, data, post,
               {"p1": "screening"}, 10.0)
    (kid,) = store.get_state(PG.K_LADDERS)["p1"]
    assert store.get_state(f"config_diff:{kid}") == {"loss.w": 4.0}               # one-sided (symmetry off)
    store.set_state("evalonly.v3", {"harm": {kid: "slide -.09 (tol .01, P=.99)"}})
    acts = PG.ladders(ctx, store, adapter, data, post, {"p1": "screening"}, 20.0)
    opp = store.get_state(PG.K_OPP)["p1"]["loss.w"]
    assert store.get_state(f"config_diff:{opp}") == {"loss.w": 0.25} and f"opposite_of={kid}" in store.get_node(opp).notes
    assert any("opposite" in a for a in acts)
    n = len(store.nodes())
    PG.ladders(ctx, store, adapter, data, post, {"p1": "screening"}, 30.0)
    assert len(store.nodes()) == n                                                 # once


def test_an_arm_without_a_declared_opposite_mirrors_its_numeric_literals_and_a_text_change_has_none():
    arm = {"id": "half", "lane": "evalonly", "file": "m.py", "span": "m.py::f", "edits": [{"old": "x * 1.0", "new": "x * 0.5"}]}
    opp = PG.opposite_arm(arm)
    assert opp["id"] == "half_opp" and opp["edits"] == [{"old": "x * 1.0", "new": "x * 2.0"}] and PG.arm_problem(opp) == ""
    assert PG.opposite_arm({**arm, "edits": [{"old": "g = s(t)", "new": "g = 0.5 + 0.5 * s(t)"}]}) is None
    declared = PG.opposite_arm({**arm, "opposite": {"id": "sharp", "edits": [{"old": "a", "new": "b"}], "note": "n"}})
    assert declared["id"] == "sharp" and declared["opposite_of"] == "half" and declared["span"] == "m.py::f"


def test_with_both_directions_off_the_opposite_arm_is_built_only_once_the_arm_is_harmful(ctx, store):
    built = []

    class Ad:
        config_rel = "config.yaml"

        def prepare_worktree(self, **k):
            raise AssertionError("no worktree")

        commit = reap_worktree = prepare_worktree

        def file_at(self, c, rel):
            return "def f():\n    return x * 1.0\n"

        def commit_files(self, parent, files, *, message, branch=""):
            built.append(files)
            return f"sha{len(built)}"

    arm = {"id": "half", "lane": "evalonly", "file": "m.py", "span": "m.py::f",
           "edits": [{"old": "x * 1.0", "new": "x * 0.5"}]}
    c = _icfg(ctx, operator_programs=({"id": "op", "arms": [arm]},), evalonly_both_directions=False)
    store.ensure_root("r0")
    inc = store.get_node(ROOT_NODE_ID)
    PG.arms(c, store, Ad(), inc, 1.0)
    book = store.get_state(PG.K_ARMS)["op"]
    assert set(book) == {"half"}
    PG.arms(c, store, Ad(), inc, 2.0)
    assert set(store.get_state(PG.K_ARMS)["op"]) == {"half"}                     # not harmful yet
    store.update_node(book["half"], status=NodeStatus.VETOED)
    acts = PG.arms(c, store, Ad(), inc, 3.0)
    book = store.get_state(PG.K_ARMS)["op"]
    assert "half_opp" in book and any("harmful" in a for a in acts) and "x * 2.0" in built[-1]["m.py"]
    PG.arms(c, store, Ad(), inc, 4.0)
    assert len(built) == 2                                                        # once


# ---------------------------------------------------------------------------------------------- B2 dead datasets
def _ds_node(store, nid, path, parent=ROOT_NODE_ID, code=False, extra=None):
    """A dataset node: v3 review M4 -- only a CLEAN single-lever node (config diff = the dataset + neutral keys, no code) can
    kill a dataset, so it has a (config-only) diff file."""
    import tempfile
    d = ("diff --git a/configs/main.yaml b/configs/main.yaml\n--- a/configs/main.yaml\n+++ b/configs/main.yaml\n@@ -1 +1 @@\n"
         "-  dataset_dir: x\n+  dataset_dir: y\n" + ("diff --git a/train.py b/train.py\n--- a/train.py\n+++ b/train.py\n@@ -1 +1 @@\n-a\n+b\n"
                                                      if code else ""))
    f = tempfile.NamedTemporaryFile("w", suffix=f"-{nid}.diff", delete=False)
    f.write(d)
    f.close()
    store.put_node(Node(id=nid, task_id=store.task_id, commit=f"c-{nid}", parent_id=parent, strategy=Strategy.EXPLORE,
                        operator=Operator.DRAFT, family="data_curation", mechanism_tag=f"t{nid}", diff_hash=f"h{nid}",
                        status=NodeStatus.QUEUED, diff_path=f.name))
    store.set_state(f"config_diff:{nid}", {"data.dataset_dir": path, **(extra or {})})


def test_a_dataset_measured_below_its_parent_on_3_seeds_is_dead_unless_differs_from_answers_it(ctx, store):
    store.ensure_root("r0")
    for nid in ("n1", "n2", "n3"):
        _ds_node(store, nid, "/bank/mpp050_448")
    _ds_node(store, "n4", "/bank/other")
    B = SimpleNamespace
    DS.record(ctx, store, {"n1": B(m=-0.0237, n=4), "n2": B(m=-0.01, n=2), "n3": B(m=0.002, n=5),
                           "n4": B(m=-0.003, n=1)})
    tab = store.get_state(DS.K_DATASETS)
    assert set(tab["path:/bank/mpp050_448"]["nodes"]) == {"n1", "n2", "n3"}
    ident = "path:/bank/mpp050_448"
    assert [r[0] for r in DS.dead_nodes(ctx, store, ident)] == ["n1"]             # n2: 2 seeds; n3: positive
    cand = SimpleNamespace(config_diff={"data.dataset_dir": "/bank/mpp050_448"},
                           hypothesis=SimpleNamespace(claim="tile the bank at 448"))
    rule, fix = DS.candidate_rule(ctx, store, cand)
    assert rule == "dead_dataset" and "n1 (-0.0237, 4 seeds)" in fix
    cand.hypothesis.claim = "x [differs from: n1 it trained at 336 px with a different crop, this keeps 224]"
    assert DS.candidate_rule(ctx, store, cand) == ("", "")
    cand.hypothesis.claim = "x [differs from: n1]"                                # named but not explained
    assert DS.candidate_rule(ctx, store, cand)[0] == "dead_dataset"
    assert DS.rule(ctx, store, "", "path:/bank/other") == ("", "")
    assert DS.rule(_icfg(ctx, dead_dataset_min_seeds=0), store, "", ident) == ("", "")
    lines = DS.lines(ctx, store)
    assert lines[0].startswith("- /bank/mpp050_448: 3 node(s), 11 paired seed(s)") and "DEAD" in lines[0]
    assert "DEAD" not in lines[1]


def test_dead_dataset_fires_in_intake_diagnose(ctx, store, monkeypatch):
    from loop.select import intake as IN
    store.ensure_root("r0")
    _ds_node(store, "n1", "/bank/d")
    DS.record(ctx, store, {"n1": SimpleNamespace(m=-0.02, n=3)})
    monkeypatch.setattr(IN, "mech_rule", lambda *a, **k: ("", ""))
    cand = SimpleNamespace(config_diff={"data.dataset_dir": "/bank/d"}, diff_text="+x", changed_paths=("config.yaml",),
                           hypothesis=SimpleNamespace(claim="again", family="data_curation", mechanism_tag="t",
                                                      levers=()), diff_hash="new", commit="", parent_commit="c-root",
                           handoff=SimpleNamespace(actions="", learnings="", concerns=""))
    emb = SimpleNamespace(embed=lambda t: None)
    monkeypatch.setattr(IN, "repeat_rule", lambda *a: ("", ""))
    monkeypatch.setattr(IN, "sweep_rule", lambda *a, **k: ("", ""))
    monkeypatch.setattr(IN, "forbidden", lambda *a, **k: ("", "", ""))
    assert IN.diagnose(ctx, store, cand, operator=Operator.DRAFT, embedder=emb, strategy=Strategy.EXPLORE)[0] \
        == "dead_dataset"


# ---------------------------------------------------------------------------------------------- B3 mixed changes
def _mx(ctx, mode="split"):
    return dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, intake=dataclasses.replace(
        ctx.cfg.intake, mixed_change=mode, lever_groups={
            "dataset": ("data.dataset_dir",), "resolution": ("train.global_size", "train.local_size"),
            "optimizer": ("dino.lr", "dino.warmup_fraction")})))


def test_a_dataset_plus_resolution_change_is_mixed_but_crossovers_joint_claims_and_one_group_are_not(ctx, store):
    c = _mx(ctx)
    cand = SimpleNamespace(config_diff={"data.dataset_dir": "/bank/d", "train.global_size": 336, "train.local_size": 168,
                                        "train.num_workers": 12}, hypothesis=SimpleNamespace(claim="hi-mag"))
    rule, fix = MX.rule(c, store, None, cand, Strategy.MOONSHOT, Operator.DRAFT)
    assert rule == "mixed_change" and "dataset: data.dataset_dir" in fix and "resolution:" in fix
    assert MX.rule(c, store, None, cand, Strategy.CROSSOVER, Operator.CROSSOVER) == ("", "")
    joint = SimpleNamespace(config_diff=cand.config_diff, hypothesis=SimpleNamespace(
        claim="x [joint: 448 px tiles only carry more detail when the crops are resized to 336 px]"))
    assert MX.rule(c, store, None, joint, Strategy.MOONSHOT, Operator.DRAFT) == ("", "")
    one = SimpleNamespace(config_diff={"dino.lr": 2e-4, "dino.warmup_fraction": 0.08, "train.num_workers": 4},
                          hypothesis=SimpleNamespace(claim="lr"))
    assert MX.rule(c, store, None, one, Strategy.EXPLORE, Operator.DRAFT) == ("", "")   # one group (+ neutral)
    assert MX.rule(_mx(ctx, "off"), store, None, cand, Strategy.MOONSHOT, Operator.DRAFT) == ("", "")
    assert MX.neutral(c, "train.num_workers") and not MX.neutral(c, "dino.lr")


def test_admit_returns_mixed_change_only_after_init_compat_passes(ctx, store, monkeypatch):
    from loop.select import intake as IN
    monkeypatch.setattr(IN, "diagnose", lambda *a, **k: ("mixed_change", "fix"))
    monkeypatch.setattr(IN, "needs_init_compat", lambda *a: True)
    monkeypatch.setattr(IN, "derive_families", lambda *a: (("data_curation",), ""))
    cand = SimpleNamespace(init_compat=None)
    assert IN.admit(ctx, store, cand, operator=None, embedder=None) == "init_compat_missing"
    cand.init_compat = SimpleNamespace(ok=True, rule="")
    assert IN.admit(ctx, store, cand, operator=None, embedder=None) == "mixed_change"


def test_a_mixed_node_is_split_into_one_sibling_per_group_and_held_until_each_part_screened(ctx, store):
    c = _mx(ctx)
    store.ensure_root("r0")
    store.put_node(Node(id="inc", task_id=store.task_id, commit="c-inc", parent_id=ROOT_NODE_ID,
                        strategy=Strategy.EXPLORE, operator=Operator.DRAFT, family="f", mechanism_tag="inc",
                        diff_hash="hinc", status=NodeStatus.QUEUED))
    store.put_node(Node(id="mx", task_id=store.task_id, commit="c-mx", parent_id="inc", strategy=Strategy.MOONSHOT,
                        operator=Operator.DRAFT, family="augmentation", mechanism_tag="hires", diff_hash="hmx",
                        status=NodeStatus.QUEUED, notes="request=req-1 brief=program mixed=dataset,resolution"))
    diff = {"data.dataset_dir": "/bank/d", "train.global_size": 336, "train.local_size": 168, "train.num_workers": 12}
    store.set_state("config_diff:mx", diff)
    built = []
    pcfg = {"data": {"dataset_dir": "/data/arrow"}, "train": {"global_size": 224, "local_size": 128, "num_workers": 8}}
    adapter = SimpleNamespace(config_rel="config.yaml", config_at=lambda commit: pcfg,
                              dry_build=lambda commit, levels, knobs, diffs, name: built.append((commit, dict(levels)))
                              or (f"sha-{len(built)}", ""))
    acts: list[str] = []
    st = {"inc": "incumbent", "mx": "queued"}
    held = MX.held(c, store, adapter, SimpleNamespace(obs=()), st, 5.0, acts)
    rec = store.get_state(MX.K_SPLIT)["mx"]
    assert held == {"mx"} and set(rec["parts"]) == {"dataset", "resolution"}
    dpart, rpart = rec["parts"]["dataset"], rec["parts"]["resolution"]
    assert ("c-mx", {"train.global_size": 224, "train.local_size": 128}) in built       # the OTHER group reverted
    assert store.get_state(f"config_diff:{dpart}") == {"data.dataset_dir": "/bank/d", "train.num_workers": 12}
    assert store.get_state(f"config_diff:{rpart}") == {"train.global_size": 336, "train.local_size": 168,
                                                       "train.num_workers": 12}
    p = store.get_node(dpart)
    assert p.parent_id == "inc" and p.status is NodeStatus.QUEUED and "split_of=mx group=dataset" in p.notes
    assert "mixed=" not in p.notes and p.mechanism_tag == "hires_dataset"
    st.update({dpart: "screening", rpart: "queued"})
    obs = tuple(SimpleNamespace(node=dpart, seed=s) for s in (1, 2))
    assert MX.held(c, store, adapter, SimpleNamespace(obs=obs), st, 6.0, []) == {"mx"}   # one part still unscreened
    st[rpart] = "failed"
    acts = []
    assert MX.held(c, store, adapter, SimpleNamespace(obs=obs), st, 7.0, acts) == frozenset()
    assert any("may run" in a for a in acts) and len(built) == 2                        # parts built once


def test_a_split_part_on_a_dead_dataset_is_not_built_and_one_part_left_is_no_split(ctx, store):
    c = _mx(ctx)
    store.ensure_root("r0")
    _ds_node(store, "old", "/bank/d")
    DS.record(c, store, {"old": SimpleNamespace(m=-0.02, n=4)})
    store.put_node(Node(id="mx", task_id=store.task_id, commit="c-mx", parent_id=ROOT_NODE_ID, strategy=Strategy.MOONSHOT,
                        operator=Operator.DRAFT, family="augmentation", mechanism_tag="hires", diff_hash="hmx",
                        status=NodeStatus.QUEUED, notes="mixed=dataset,resolution"))
    store.set_state("config_diff:mx", {"data.dataset_dir": "/bank/d", "train.global_size": 336})
    adapter = SimpleNamespace(config_rel="config.yaml", config_at=lambda c_: {"data": {"dataset_dir": "/a"},
                                                                             "train": {"global_size": 224}},
                              dry_build=lambda commit, levels, knobs, diffs, name: ("sha-x", ""))
    held = MX.held(c, store, adapter, SimpleNamespace(obs=()), {"mx": "queued"}, 1.0, [])
    rec = store.get_state(MX.K_SPLIT)["mx"]
    assert rec["parts"]["dataset"].startswith("-: dead_dataset") and held == frozenset()


# ---------------------------------------------------------------------------------------------- B6 housekeeping
def test_smoke_dirs_of_finished_requests_older_than_the_cap_go_whole_never_an_inflight_ones(ctx, store):
    root = __import__("pathlib").Path(ctx.state_dir) / "smoke"
    old = time.time() - 30 * 3600
    for name in ("loop-req-v3-00001", "loop-req-v3-00002-expandable_segments", "loop-req-v3-00003", "memsmoke-n1",
                 "loop-req-v3-00004"):
        d = root / name / "out" / "labless_source"
        d.mkdir(parents=True)
        (d / "x.py").write_text("x" * 1000)
        for p in (d, d.parent, d.parent.parent):
            os.utime(p, (old, old))
    fresh = root / "loop-req-v3-00004" / "smoke.yaml"
    fresh.write_text("y")                                                      # touched now: too young
    store.set_state(K_INFLIGHT, [{"id": "req-v3-00002"}, {"id": "req-v3-00009", "queued": True}])
    got = {d.name: s for d, s in PR.smoke_dir_plan(ctx, store, time.time())}
    assert set(got) == {"loop-req-v3-00001", "loop-req-v3-00003"} and got["loop-req-v3-00001"] >= 1000
    c = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, prune=dataclasses.replace(
        ctx.cfg.prune, enabled=True, dry_run=False, git=False)))
    lines = PR.Pruner(c, store, time.time)._pass(None, time.time())
    assert not (root / "loop-req-v3-00001").exists() and (root / "loop-req-v3-00002-expandable_segments").exists()
    assert (root / "memsmoke-n1").exists() and (root / "loop-req-v3-00004").exists()
    assert any(l.startswith("prune smoke: 2 finished request smoke dir(s)") for l in lines)


def test_v3_node_worktrees_go_once_nothing_launches_them_whatever_the_checkpoint_view_keeps(ctx, store):
    """10-14: 411 node worktrees, `prune git: 0 node worktrees` every pass -- the v3 keep set (every non-closed node,
    and a SCREENING node stays SCREENING for life) also kept every worktree."""
    store.ensure_root("r0")
    now = 1_000_000.0
    for nid, ts in (("old", now - 3 * 86400), ("busy", now - 3 * 86400), ("young", now - 3600),
                    ("queued", now - 3 * 86400), ("inc", now - 5 * 86400), ("wave", now - 3 * 86400)):
        store.put_node(Node(id=nid, task_id=store.task_id, commit=f"c-{nid}", parent_id=ROOT_NODE_ID,
                            strategy=Strategy.EXPLORE, operator=Operator.DRAFT, family="f", mechanism_tag=nid,
                            diff_hash=f"h{nid}", status=NodeStatus.QUEUED, created_ts=ts))
    store.put_run(Run(id=Run.make_id("busy", Stage.SCREEN, 1), task_id=store.task_id, node_id="busy", seed=1,
                      stage=Stage.SCREEN, status=RunStatus.RUNNING, submitted_ts=now - 2 * 86400))
    store.put_run(Run(id=Run.make_id("old", Stage.SCREEN, 1), task_id=store.task_id, node_id="old", seed=1,
                      stage=Stage.SCREEN, status=RunStatus.DONE, submitted_ts=now - 2 * 86400))
    snap = SimpleNamespace(incumbent_id="inc", banked=(), closed=set(), queued={"queued"}, open={"wave"},
                           keep={"old", "busy", "young", "queued", "inc", "wave"})
    _, _, wts = PR.git_plan(ctx, store, snap, now)
    assert wts == ["old"]
    v2 = SimpleNamespace(incumbent_id="inc", banked=(), closed=set(), keep={"old"})   # no ``queued``: the old rule
    assert "old" not in PR.git_plan(ctx, store, v2, now)[2]


def test_v3_intake_admits_a_mixed_node_tagged_for_the_split_and_rejects_it_in_reject_mode(env, monkeypatch):
    from loop.agents.researcher import ResearchResult
    from loop.core import ideation as ID
    from loop.select import intake as IN

    from .test_ideation import _cand
    ctx, store, led, ide = env
    lg = {"objective": ("loss.*",), "optimizer": ("train.lr",)}
    ide.ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, intake=dataclasses.replace(
        ctx.cfg.intake, lever_groups=lg)))
    monkeypatch.setattr(IN, "admit", lambda *a, **k: "mixed_change")
    req = ide.request(ID.Brief("explore", "explore", ("root",), "b"))
    cand = dataclasses.replace(_cand("two levers at once", commit="cM"), config_diff={"loss.weight": 2.0,
                                                                                       "train.lr": 3e-4})
    out = ide.intake(req, ResearchResult("ok", cand))
    node = store.get_node(out.split()[1])
    assert out.startswith("queued") and "mixed=objective,optimizer" in node.notes
    ide.ctx = dataclasses.replace(ide.ctx, cfg=dataclasses.replace(ide.ctx.cfg, intake=dataclasses.replace(
        ide.ctx.cfg.intake, mixed_change="reject")))
    out = ide.intake(req, ResearchResult("ok", dataclasses.replace(cand, commit="cM2", diff_hash="h-cM2")))
    assert out.startswith("rejected") and store.get_node(out.split()[1]).reject_reason == "mixed_change"


def test_a_ladder_built_before_isolation_is_backfilled_once(ctx, store, monkeypatch):
    monkeypatch.setattr("loop.extrapolate._allowed", lambda c, k: True)
    cfgs = {"r0": {"train": {"local_views": 14}, "data": {"global_crop_scale": [0.4, 1.0]}},
            "c-p1": {"train": {"local_views": 10}, "data": {"global_crop_scale": [0.3, 1.0]}}}
    data, post, adapter, built = _prog(store, cfgs, belief=(0.01, 4e-6))
    PG.ladders(_icfg(ctx, program_iso=False), store, adapter, data, post, {"p1": "screening"}, 10.0)
    assert store.get_state(PG.K_ISO) in (None, {})
    acts = PG.ladders(ctx, store, adapter, data, post, {"p1": "screening"}, 20.0)
    iso = store.get_state(PG.K_ISO)["p1"]
    assert set(iso) == {"data.global_crop_scale"} and any("isolation" in a for a in acts)
    n = len(store.nodes())
    PG.ladders(ctx, store, adapter, data, post, {"p1": "screening"}, 30.0)
    assert len(store.nodes()) == n
