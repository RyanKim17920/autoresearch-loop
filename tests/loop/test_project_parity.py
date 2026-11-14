"""loop.project (multi-project Stage 1): with no ``loop.project`` block every registry value equals the nanopath literal
it replaced; a nanobrain 3-axis install reaches every former import-time snapshot site; the guard-None paths of
status / meta / ideation run."""

from __future__ import annotations

import json
import math
import random
import statistics
from pathlib import Path

import pytest
import yaml

from loop import project as P

REPO = Path(__file__).resolve().parents[2]

# today's literals (loop/core/data.py:29-34, core/ideation.py:71-73, core/test.py:60, core/acquire.py:73), verbatim
OLD_AXES = ("cls", "seg", "slide", "auc", "surv", "rob")
OLD_PROBE_KEYS = {"cls": "final_probe_classification_mean_f1", "seg": "final_probe_seg_mean_f1",
                  "slide": "final_probe_slide_mean_auc", "auc": "final_probe_auc_mean",
                  "surv": "final_probe_survival_mean_cindex", "rob": "final_probe_robustness_mean"}
OLD_WEIGHTS = {"cls": 0.30, "seg": 0.15, "slide": 0.175, "auc": 0.15, "surv": 0.075, "rob": 0.15}
OLD_NAMES = {"cls": "classification", "seg": "segmentation", "slide": "progression", "auc": "mutation",
             "surv": "survival", "rob": "robustness"}

NB_TASKS = {"cls": ["abide_diagnosis", "adhd_diagnosis", "cnp_diagnosis", "ucsf_idh", "upenn_survival_1y",
                    "soop_mrs_poor"],
            "reg": ["ixi_age", "openbhb_age"],
            "seg": ["brats_enhancing_tumor", "brats_whole_tumor", "soop_lesion"]}
NANOBRAIN = {
    "name": "nanobrain", "base_sha": "440fc92a0f949c5874db64ec6491f397bcbda37c",
    "runs_root": "/data/user/nanobrain/runs", "worktrees_root": "/home/user/nanobrain/worktrees",
    "config_rel": "src/nanobrain/config/default_pretrain.yaml", "aggregation": "weighted",
    "axes": [{"name": "cls", "key": "nb_cls_mean", "label": "classification (2*AUROC-1)", "weight": 6,
              "tasks": NB_TASKS["cls"]},
             {"name": "reg", "key": "nb_reg_mean", "label": "regression (R2, brain age)", "weight": 2,
              "tasks": NB_TASKS["reg"]},
             {"name": "seg", "key": "nb_seg_mean", "label": "segmentation (Dice)", "weight": 3,
              "tasks": NB_TASKS["seg"]}],
    "target": "cls", "guard": [], "harm_h": {}, "axis_exclude": [],
    "budget": {"kind": "wall", "minutes": 60, "key": "time_limit"}}


# ---------------------------------------------------------------------------------------------- defaults (parity)
def test_defaults_equal_the_old_literals():
    from loop.core import acquire as A
    from loop.core import data as D
    from loop.core import ideation as ID
    from loop.core import test as T
    assert tuple(P.AXES) == OLD_AXES and list(D.AXES) == list(OLD_AXES) and (*D.AXES,) == OLD_AXES
    assert dict(P.PROBE_KEYS) == OLD_PROBE_KEYS == dict(D.PROBE_KEYS)
    assert dict(P.OFFICIAL_WEIGHTS) == OLD_WEIGHTS == dict(D.OFFICIAL_WEIGHTS)
    assert list(P.OFFICIAL_WEIGHTS) == list(OLD_WEIGHTS)                 # iteration order too
    assert dict(ID.AXIS_NAMES) == OLD_NAMES
    assert tuple(T.GUARD_AXES) == ("rob",) and dict(A.HARM_H) == {"rob": 0.008}
    assert P.target() == "cls" and P.guard() == "rob" and P.axis_exclude() == ("surv",)
    assert P.current() is None and P.domain() is None
    assert P.prompt_path("ideation") == REPO / "loop" / "prompts" / "ideation.md"
    assert P.rule("headroom_footer", "x") == "x"


def test_live_views_behave_like_tuples_and_dicts():
    assert len(P.AXES) == 6 and "rob" in P.AXES and P.AXES.index("surv") == 4 and P.AXES[0] == "cls"
    assert P.AXES == OLD_AXES and P.AXES + ("panel",) == (*OLD_AXES, "panel") and ("x",) + P.AXES == ("x", *OLD_AXES)
    assert P.OFFICIAL_WEIGHTS.get("cls") == 0.30 and P.OFFICIAL_WEIGHTS.get("zz", 0.0) == 0.0
    assert {**P.PROBE_KEYS} == OLD_PROBE_KEYS and json.dumps(list(P.AXES))
    assert math.isclose(sum(P.OFFICIAL_WEIGHTS.values()), 1.0)
    with pytest.raises(TypeError):
        P.OFFICIAL_WEIGHTS["cls"] = 1.0                                  # type: ignore[index]


def test_an_explicit_nanopath_block_installs_the_defaults_exactly():
    block = {"name": "nanopath", "axes": [{"name": a, "key": OLD_PROBE_KEYS[a], "label": OLD_NAMES[a],
                                           "weight": OLD_WEIGHTS[a]} for a in OLD_AXES],
             "target": "cls", "guard": ["rob"], "harm_h": {"rob": 0.008}, "axis_exclude": ["surv"]}
    P.install(block)
    assert tuple(P.AXES) == OLD_AXES and dict(P.PROBE_KEYS) == OLD_PROBE_KEYS and dict(P.AXIS_NAMES) == OLD_NAMES
    assert dict(P.OFFICIAL_WEIGHTS) == OLD_WEIGHTS                       # byte-exact: weights summing to 1 are kept
    assert tuple(P.GUARD_AXES) == ("rob",) and dict(P.HARM_H) == {"rob": 0.008} and P.axis_exclude() == ("surv",)


def test_nanopath_v2_yaml_block_equals_the_in_code_defaults():
    raw = yaml.safe_load((REPO / "tasks" / "nanopath_v2.yaml").read_text())
    block = (raw.get("loop") or {}).get("project")
    if block is None:
        pytest.skip("tasks/nanopath_v2.yaml has no explicit loop.project block yet (package C)")
    P.install(block, (raw.get("loop") or {}).get("domain"))
    assert tuple(P.AXES) == OLD_AXES and dict(P.PROBE_KEYS) == OLD_PROBE_KEYS
    assert dict(P.OFFICIAL_WEIGHTS) == pytest.approx(OLD_WEIGHTS, abs=1e-12)
    assert P.target() == "cls" and tuple(P.GUARD_AXES) == ("rob",) and dict(P.HARM_H) == {"rob": 0.008}
    assert P.axis_exclude() == ("surv",)


# ---------------------------------------------------------------------------------------------- nanobrain install
def test_nanobrain_install_normalises_weights_and_has_no_guard():
    P.install(NANOBRAIN, {"prompt_dir": "tasks/nanobrain/prompts", "rules": {"init": "no brain-MRI weights"}})
    assert tuple(P.AXES) == ("cls", "reg", "seg")
    w = dict(P.OFFICIAL_WEIGHTS)
    assert w == pytest.approx({"cls": 6 / 11, "reg": 2 / 11, "seg": 3 / 11}, abs=1e-15)
    assert P.guard() is None and tuple(P.GUARD_AXES) == () and dict(P.HARM_H) == {} and P.target() == "cls"
    assert dict(P.PROBE_KEYS) == {"cls": "nb_cls_mean", "reg": "nb_reg_mean", "seg": "nb_seg_mean"}
    assert P.AXIS_NAMES["reg"] == "regression (R2, brain age)"
    assert P.current().base_sha.startswith("440fc92") and P.current().axes[0].tasks == tuple(NB_TASKS["cls"])
    assert P.rule("init", "x") == "no brain-MRI weights" and P.rule("data", "dflt") == "dflt"
    P.reset()
    assert tuple(P.AXES) == OLD_AXES and P.guard() == "rob" and P.current() is None


def test_composite_of_the_family_means_is_the_plain_mean_of_the_11_tasks():
    P.install(NANOBRAIN)
    rng = random.Random(7)
    task = {t: rng.uniform(0.0, 0.9) for ts in NB_TASKS.values() for t in ts}
    fam = {a: statistics.fmean(task[t] for t in NB_TASKS[a]) for a in P.AXES}
    composite = sum(P.OFFICIAL_WEIGHTS[a] * fam[a] for a in P.AXES)
    assert composite == pytest.approx(statistics.fmean(task.values()), abs=1e-12)


def test_mean_aggregation_weights_are_equal():
    P.install({**NANOBRAIN, "aggregation": "mean"})
    assert dict(P.OFFICIAL_WEIGHTS) == pytest.approx({"cls": 1 / 3, "reg": 1 / 3, "seg": 1 / 3})


@pytest.mark.parametrize("patch, msg", [
    ({"target": "rob"}, "target"), ({"guard": ["rob"]}, "guard"), ({"harm_h": {"x": 1.0}}, "harm_h"),
    ({"axis_exclude": ["surv"]}, "axis_exclude"), ({"axes": []}, "at least one"), ({"bogus": 1}, "unknown"),
    ({"aggregation": "max"}, "aggregation"),
    ({"axes": NANOBRAIN["axes"] + [{"name": "cls", "key": "dup", "weight": 1}]}, "duplicate"),
])
def test_project_validation(patch, msg):
    with pytest.raises(ValueError, match=msg):
        P.install({**NANOBRAIN, **patch})


def test_prompt_overlay_wins_then_falls_back(tmp_path):
    (tmp_path / "ideation.md").write_text("OVERLAY {x}")
    P.install(None, {"prompt_dir": str(tmp_path)})
    assert P.prompt_path("ideation") == tmp_path / "ideation.md"
    assert P.prompt_path("review") == REPO / "loop" / "prompts" / "review.md"
    from loop.core import ideation as ID
    from loop.core import meta as MT
    assert ID._render("ideation", x=1) == "OVERLAY 1"
    assert MT._prompt("ideation.md") == tmp_path / "ideation.md"
    assert ID.prompt_version("ideation") != (P.reset() or ID.prompt_version("ideation"))


# ---------------------------------------------------------------------------------------------- spec wiring
def _spec_with_project(tmp_path, project):
    raw = yaml.safe_load((REPO / "tasks" / "fake_loop.yaml").read_text())
    raw["loop"]["project"] = project
    raw["loop"]["domain"] = {"prompt_dir": "tasks/nanobrain/prompts", "rules": {"init": "I"}}
    p = tmp_path / "spec.yaml"
    p.write_text(yaml.safe_dump(raw))
    return p


FAKE_PROJECT = {"name": "fake", "axes": [{"name": "a", "key": "alpha", "weight": 0.5},
                                         {"name": "b", "key": "beta", "weight": 0.3},
                                         {"name": "c", "key": "gamma", "weight": 0.2}],
                "target": "a", "guard": []}


def test_load_context_installs_the_project_and_a_plain_spec_resets(tmp_path, fake_loop_spec):
    from loop.spec import load_context
    ctx = load_context(_spec_with_project(tmp_path, FAKE_PROJECT), state_dir=tmp_path / "s")
    assert ctx.cfg.project.name == "fake" and ctx.cfg.domain.rules == {"init": "I"}
    assert tuple(P.AXES) == ("a", "b", "c") and P.guard() is None and ctx.cfg.axes == ("alpha", "beta", "gamma")
    plain = load_context(fake_loop_spec, state_dir=tmp_path / "s2")
    assert plain.cfg.project is None and plain.cfg.domain is None
    assert tuple(P.AXES) == OLD_AXES and P.guard() == "rob"


def test_project_axis_keys_must_equal_slots_vector(tmp_path):
    from loop.spec import load_context
    bad = {**FAKE_PROJECT, "axes": FAKE_PROJECT["axes"][:2]}
    with pytest.raises(ValueError, match="slots.vector"):
        load_context(_spec_with_project(tmp_path, bad), state_dir=tmp_path / "s")


# ---------------------------------------------------------------------------------------------- snapshot sites
def test_import_time_snapshot_sites_read_live_values():
    """acquire.Posterior.TARGETS, model.fit_noise's targets, status.judge's guard, test.GUARD_AXES: all were bound at
    import time; after an install they see the new project."""
    from loop.core import acquire as A
    from loop.core import data as D
    from loop.core import model as M
    from loop.core import test as T
    assert A.Posterior.TARGETS.fget is not None                          # a property, not a class-time tuple
    led_obs = [D.Obs(f"r{i}", "n", i, "screen", D.SELECT, {a: 0.5 for a in P.AXES}, 0.5, 1.0, 0.0, 0)
               for i in range(3)]
    assert set(M.fit_noise(led_obs).sigma2) == {*OLD_AXES, D.PANEL}
    P.install(NANOBRAIN)
    obs = [D.Obs(f"r{i}", "n", i, "screen", D.SELECT, {a: 0.5 for a in P.AXES}, 0.5, 1.0, 0.0, 0) for i in range(3)]
    assert set(M.fit_noise(obs).sigma2) == {"cls", "reg", "seg", D.PANEL}
    assert tuple(T.GUARD_AXES) == () and dict(A.HARM_H) == {} and tuple(D.AXES) == ("cls", "reg", "seg")

    class _Fake:                                                         # TARGETS reads only the class's property
        TARGETS = A.Posterior.TARGETS
    assert _Fake().TARGETS == (M.COMPOSITE,)
    P.reset()
    assert _Fake().TARGETS == (M.COMPOSITE, "rob")


def test_data_load_reads_the_installed_probe_keys(tmp_path):
    """An observation exists only if every PROBE_KEYS value is in runs.scores: the nanopath fixture re-keyed to the
    nanobrain summary keys has observations only once nanobrain is installed."""
    from loop.core import data as D
    from tests.loop.core.conftest import SAMPLE, TASK, build_db
    sample = json.loads(SAMPLE.read_text())
    remap = {"final_probe_classification_mean_f1": "nb_cls_mean", "final_probe_seg_mean_f1": "nb_seg_mean",
             "final_probe_slide_mean_auc": "nb_reg_mean"}
    for r in sample["runs"]:
        if r.get("scores"):
            sc = json.loads(r["scores"])
            r["scores"] = json.dumps({remap[k]: v for k, v in sc.items() if k in remap})
    db = build_db(tmp_path / "l.db", sample)
    assert D.load(db, TASK).obs == ()
    P.install(NANOBRAIN)
    led = D.load(db, TASK)
    assert led.obs and all(set(o.axes) == {"cls", "reg", "seg"} for o in led.obs)


def test_status_judge_with_no_guard_axis():
    from loop.core import data as D
    from loop.core import status as ST
    from loop.core.model import Noise
    P.install(NANOBRAIN)
    rng = random.Random(1)
    runs, obs = [], []
    for nid, mu in ((D.ROOT, 0.0), ("c", 0.01)):
        for s in range(100000, 100006):
            rid = f"{nid}:test:{s}"
            y = mu + rng.gauss(0, 0.003)
            runs.append(D.Run(rid, nid, s, "test", "done", 1.0, 1.0, 2.0, "", 0))
            obs.append(D.Obs(rid, nid, s, "test", D.TEST, {a: y for a in D.AXES}, y, 1.0, 2.0, 0))
    nodes = {n: D.Node(n, None if n == D.ROOT else D.ROOT, None, "explore", "f", "explore", D.ROOT, 0.0,
                       "screening", "") for n in (D.ROOT, "c")}
    led = D.Ledger(nodes, tuple(runs), tuple(obs), frozenset(), ((0.0, D.ROOT),))
    noise = Noise({t: 0.003 ** 2 for t in (*D.AXES, D.PANEL)}, {t: 0.4 for t in (*D.AXES, D.PANEL)})
    v = ST.judge(led, "c", D.ROOT, set(range(100000, 100006)), noise)
    assert v.harm == () or list(v.harm) == []


def test_meta_policy_and_apply_with_no_guard_axis():
    from loop import spec as SP
    from loop.core import backtest as B
    from loop.core import meta as MT

    def point(i, m_c, fresh):
        p = B.Point(f"n{i}", 1e9 + i * 3600, "explore", "f", "root", 3, 4, "confirm", m_c, 1e-5, m_c, m_c, 1e-6,
                    m_c, 1e-6, 0.0, 1, fresh, 2e-6, fresh, 2e-6)
        return MT.EvalPoint(p, 4, (-0.05, 1e-6), 0.0)                    # a disastrous "rob" belief
    pts = [point(i, 0.003, 0.004) for i in range(6)]
    vals = SP.tunable_values()
    assert MT._policy(pts, vals, 1e-5) == []                             # nanopath: the rob harm limit blocks all
    P.install(NANOBRAIN)
    assert len(MT._policy(pts, vals, 1e-5)) == 3                         # no guard: no harm term
    assert MT.metrics(pts, vals, 1e-5)["harm"] == 0.0

    class Core:
        harm = None
    MT.apply({"acquire.harm_h_rob": 0.01}, Core)
    assert "harm_h" not in Core.harm                                     # no guard axis: no harm_h keyword
    P.reset()
    MT.apply({"acquire.harm_h_rob": 0.01}, Core)
    assert Core.harm == {"harm_h": {"rob": 0.01}}                        # nanopath: unchanged


def test_ideation_text_with_and_without_a_guard():
    from loop.core import ideation as ID
    row = {"node": "n1", "cls": 0.01, "cls_z": 2.0, "rob": -0.001, "seg": None, "composite": 0.002, "n": 5,
           "keys": "lr=1"}
    assert ID.trades_text([row]) == ("node | cls Δ (z) | rob Δ | seg Δ | composite Δ | seeds | changed\n"
                                     "n1 | +0.0100 (+2.0) | -0.0010 | - | +0.0020 | 5 | lr=1")
    assert ID.repair_axis([], {}) == "cls" and ID.repair_axes([], {}) == ["cls"]
    P.install(NANOBRAIN, {"rules": {"headroom_footer": "FOOT"}})
    # no guard: every axis but the target as a side column (nanobrain target cls here: reg, seg)
    assert ID.trades_text([row]) == ("node | cls Δ (z) | reg Δ | seg Δ | composite Δ | seeds | changed\n"
                                     "n1 | +0.0100 (+2.0) | - | - | +0.0020 | 5 | lr=1")
    hr = ID.Headroom("seg", 3 / 11, 0.01, 0.5, 5, 0.0, None, "", None, "", "root")
    text = ID.headroom_text([hr], "n1")
    assert "segmentation (Dice)" in text and text.endswith("FOOT") and "GUARD" not in text.splitlines()[1]


def test_closed_pm_iterates_the_installed_axes(monkeypatch):
    from loop.agents import history as HX
    from loop.core import closed_pm as CP
    seen = []
    monkeypatch.setattr(HX, "axis_delta", lambda v, nid, a: seen.append(a) or None)
    P.install(NANOBRAIN)
    try:
        CP.facts(None, None, type("V", (), {"beliefs": {}})(), "n", "", None, (), {})
    except Exception:  # noqa: BLE001 - only the axis loop is under test
        pass
    assert seen == ["cls", "reg", "seg"]
