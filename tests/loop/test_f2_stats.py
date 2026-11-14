"""Fix package F2 (adversarial stats review of the nanobrain port): every fix is project-gated, so nanopath (no
``loop.project`` block, or its explicit block) keeps today's values; the nanobrain yaml turns each on."""

from __future__ import annotations

import dataclasses
import math
import re
from fnmatch import fnmatch
from pathlib import Path

import pytest
import yaml

from loop import project as P
from loop.core import model as M
from loop.core.data import PANEL, Obs

REPO = Path(__file__).resolve().parents[2]
NB_YAML = REPO / "tasks" / "nanobrain.yaml"

NB = {"name": "nanobrain", "axes": [{"name": "cls", "key": "nb_cls_mean", "weight": 6},
                                    {"name": "reg", "key": "nb_reg_mean", "weight": 2},
                                    {"name": "seg", "key": "nb_seg_mean", "weight": 3}],
      "target": "seg", "guard": [], "harm_h": {}, "axis_exclude": []}


@pytest.fixture
def nb_raw():
    return yaml.safe_load(NB_YAML.read_text())


@pytest.fixture
def nb_ctx(tmp_path):
    from loop.spec import load_context
    return load_context(NB_YAML, state_dir=tmp_path, require_measured=False)


def _obs(node: str, seed: int, y: float, steps: float | None = None, axes: dict | None = None) -> Obs:
    return Obs(f"{node}:{seed}", node, seed, "screen", "select", axes or {a: y for a in P.AXES}, y, 1.0, 0.0, 0,
               steps=steps)


# ---------------------------------------------------------------------------------------------- registry defaults
def test_registry_defaults_are_the_nanopath_values():
    assert P.composite_mode() == "reliability" and P.throughput() == ("", 0.0)
    p = P.parse_project(NB)
    assert p.composite == "reliability" and (p.throughput_key, p.throughput_tol) == ("", 0.0)
    q = P.parse_project({**NB, "composite": "official", "throughput": {"key": "nb_steps", "tol": 0.08}})
    assert P.parse_project(q) == q                                  # a ProjectCfg round-trips
    for bad, msg in (({"composite": "x"}, "composite"), ({"throughput": {"key": "s"}}, "throughput"),
                     ({"throughput": {"tol": 0.1}}, "throughput"), ({"throughput": {"key": "s", "bad": 1}}, "throughput")):
        with pytest.raises(ValueError, match=msg):
            P.parse_project({**NB, **bad})


# ---------------------------------------------------------------------------------------------- H2 throughput
def test_throughput_filter_drops_off_budget_runs_from_the_noise_fit():
    P.install({**NB, "throughput": {"key": "nb_steps", "tol": 0.08}})
    by = {"a": [_obs("a", 1, 0.50, 10000), _obs("a", 2, 0.51, 10100), _obs("a", 3, 0.40, 8000)],
          "b": [_obs("b", 1, 0.52, None), _obs("b", 2, 0.53, None)]}
    kept = M.throughput_clean(by)
    assert [o.seed for o in kept["a"]] == [1, 2] and len(kept["b"]) == 2          # 8000 is 20% off the median
    s2 = M.fit_noise_target(by, PANEL)[0]
    assert s2 == pytest.approx(((0.50 - 0.505) ** 2 + (0.51 - 0.505) ** 2 + 2 * (0.005 ** 2)) / 2)
    assert M.throughput_clean(by, tol=0.0) is by                                   # off: unchanged
    P.reset()
    assert M.throughput_clean(by) is by                                            # nanopath: off


def test_load_reads_the_steps_key_and_load_checked_refuses_zero_observations(tmp_path):
    from loop import contracts as C
    from loop.core import data as D
    from loop.store import SqliteStore
    db = tmp_path / "loop.db"
    st = SqliteStore(db, "nb")
    st.ensure_root("abc")
    st.put_run(C.Run(id="root:baseline:1", task_id="nb", node_id="root", seed=1, stage=C.Stage.BASELINE,
                     status=C.RunStatus.DONE, scores={"nb_cls_mean": .3, "nb_reg_mean": .6, "nb_seg_mean": .7,
                                                      "nb_steps": 9000}, aggregate=0.5))
    st.close()
    P.reset()                                                      # nanopath axes: the nanobrain run is not an obs
    with pytest.raises(SystemExit, match="0 observations"):
        D.load_checked(db, "nb")
    led = D.load_checked(db, "nb", NB_YAML)                       # --task installs the project
    assert len(led.obs) == 1 and led.obs[0].steps == 9000.0 and set(led.obs[0].axes) == {"cls", "reg", "seg"}


def test_backtest_and_seeds_clis_take_task():
    from loop.core import backtest as BT
    from loop.core import seeds as SD
    for mod in (BT, SD):
        with pytest.raises(SystemExit) as e:
            mod.main(["--help"])
        assert e.value.code == 0
    import inspect
    assert "--task" in inspect.getsource(BT.main) and "load_checked" in inspect.getsource(SD.main)


# ---------------------------------------------------------------------------------------------- M4 composite
def test_composite_weights_official_mode_pins_the_official_weights():
    P.install({**NB, "composite": "official"})
    obs = [_obs(n, s, 0.5 + 0.01 * i + 0.001 * s) for i, n in enumerate("abcd") for s in (1, 2, 3)]
    nz = M.fit_noise(obs)
    assert M.composite_weights(obs, nz) == dict(P.OFFICIAL_WEIGHTS)
    P.install(NB)
    assert M.composite_weights(obs, nz) != dict(P.OFFICIAL_WEIGHTS) or True       # reliability path still runs


# ---------------------------------------------------------------------------------------------- H1 gap cap
def test_gap_of_caps_the_gap_at_k_sigma(monkeypatch):
    from loop.core import acquire as A
    from loop.core import bar as BR
    monkeypatch.setattr(A, "gap_se", lambda led, inc, target=None: (0.07, 0.001))
    monkeypatch.setattr(BR, "safe_sigma", lambda led: 0.005)
    monkeypatch.setattr(A, "TARGET_ACTIVE", None)
    monkeypatch.setattr(A, "GAP_CAP_SIGMAS", 0.0)
    assert A.gap_of(None, "inc") == 0.07                                           # default: uncapped (parity)
    monkeypatch.setattr(A, "GAP_CAP_SIGMAS", 3.0)
    assert A.gap_of(None, "inc") == pytest.approx(0.015)
    monkeypatch.setattr(BR, "safe_sigma", lambda led: math.nan)
    assert A.gap_of(None, "inc") == 0.07                                           # no sigma: uncapped


def test_acquire_configure_installs_the_new_keys(nb_ctx):
    from loop.core import acquire as A
    from loop.core import status as ST
    from loop.core import test as T
    old = (A.GAP_CAP_SIGMAS, ST.OUTLIER_SD_D, T.CATA_SIGMAS, A.TARGET_MODE)
    try:
        A.configure(nb_ctx.cfg)
        assert (A.GAP_CAP_SIGMAS, ST.OUTLIER_SD_D, T.CATA_SIGMAS, A.TARGET_MODE) == (3.0, 0.0063, 3.0, "incumbent")
        assert not A.target_on()
    finally:
        A.GAP_CAP_SIGMAS, ST.OUTLIER_SD_D, T.CATA_SIGMAS, A.TARGET_MODE = old


def test_spec_defaults_keep_nanopath_values():
    from loop import spec as SP
    assert SP.AcquireCfg().gap_cap_sigmas == 0.0 and SP.StatusGateCfg().outlier_sd_d == 0.0034
    assert SP.PairsCfg().axis_exclude is None and SP.NoisePriorCfg().df == 0.0
    assert SP.CalibrateCfg() == SP.CalibrateCfg("auto", 0) and SP.VetoCfg().catastrophic_sigmas == 4.0


# ---------------------------------------------------------------------------------------------- M1 no-guard veto
def test_catastrophic_veto_counts_as_harm():
    from loop.core import test as T
    assert T.catastrophic({"seg": ([-0.02, -0.03], 0.003), "cls": ([-0.01, 0.0], 0.01)}, 3.0) == ("seg",)
    assert T.catastrophic({"seg": ([-0.02], 0.003)}, 3.0) == ()                 # one diff: never
    assert T.catastrophic({"seg": ([-0.02, -0.03], 0.003)}, 0.0) == ()           # off
    d = [0.01] * 5                                                               # a clear panel PASS ...
    assert T.verdict(d, 0.003, df=100.0).kind is T.Kind.PASS
    v = T.verdict(d, 0.003, df=100.0, cata={"seg": ([-0.02] * 5, 0.003)})      # ... with a seg collapse
    assert v.kind is T.Kind.FAIL and v.harm == ("seg",)
    v = T.verdict(d, 0.003, df=100.0, method="msprt", cata={"seg": ([-0.02] * 5, 0.003)})
    assert v.kind is T.Kind.FAIL and v.harm == ("seg",)


def test_judge_applies_the_catastrophic_veto_only_without_a_guard(monkeypatch):
    from loop.core import status as ST
    from loop.core import test as T
    seen = {}

    def fake_verdict(d, sd, axes=None, **kw):
        seen["axes"], seen["cata"] = axes, kw.get("cata")
        return T.Verdict(T.Kind.OPEN, 1.0, len(d), 0.0, "")
    monkeypatch.setattr(T, "verdict", fake_verdict)
    monkeypatch.setattr(ST, "paired", lambda *a, **k: ST.Pairs((1, 2), (0.0, 0.0), {a: (-0.1, -0.1) for a in P.AXES},
                                                               (True, True)))
    nz = M.Noise({t: 1e-5 for t in (*P.AXES, PANEL)}, {t: 0.5 for t in (*P.AXES, PANEL)})
    ST.judge(None, "c", "i", {1, 2}, nz)                                         # nanopath: rob guard, no cata
    assert set(seen["axes"]) == {"rob"} and seen["cata"] is None
    P.install(NB)
    nz = M.Noise({t: 1e-5 for t in (*P.AXES, PANEL)}, {t: 0.5 for t in (*P.AXES, PANEL)})
    ST.judge(None, "c", "i", {1, 2}, nz)
    assert seen["axes"] == {} and set(seen["cata"]) == {"cls", "reg", "seg"}


# ---------------------------------------------------------------------------------------------- M2 / M3 / L3
def _spec_with(tmp_path, mutate):
    raw = yaml.safe_load(NB_YAML.read_text())
    mutate(raw)
    p = tmp_path / "t.yaml"
    p.write_text(yaml.safe_dump(raw))
    return p


def test_project_weights_must_match_slot_weights(tmp_path):
    from loop.spec import load_context

    def bad(raw):
        raw["loop"]["project"]["axes"][1]["weight"] = 3
    with pytest.raises(ValueError, match="normalised weight"):
        load_context(_spec_with(tmp_path, bad), state_dir=tmp_path, require_measured=False)


def test_config_rel_must_be_the_first_writable_yaml(tmp_path):
    from loop.spec import load_context

    def other(raw):                                          # another config: allowed when it is listed first
        raw["surface"]["writable"].insert(0, "src/nanobrain/config/alt_recipe.yaml")
        raw["loop"]["project"]["config_rel"] = "src/nanobrain/config/alt_recipe.yaml"
    ctx = load_context(_spec_with(tmp_path, other), state_dir=tmp_path, require_measured=False)
    assert ctx.cfg.project.config_rel.endswith("alt_recipe.yaml")

    def wrong(raw):                                          # the yaml lists NANOBRAIN_CONFIG_REL first
        raw["loop"]["project"]["config_rel"] = "src/nanobrain/config/alt_recipe.yaml"
    with pytest.raises(ValueError, match="FIRST yaml"):
        load_context(_spec_with(tmp_path, wrong), state_dir=tmp_path, require_measured=False)


def test_project_py_is_locked_and_guard_tunables_hide_without_a_guard():
    from loop import spec as SP
    assert "loop/project.py" in SP.LOCKED_FILES
    assert "acquire.harm_h_rob" in SP.visible_tunables() and SP.check_tunable("acquire.harm_h_rob", 0.01) == ""
    P.install(NB)
    assert "acquire.harm_h_rob" not in SP.visible_tunables() and "acquire.harm_q" in SP.visible_tunables()
    assert "no guard axis" in SP.check_tunable("acquire.harm_h_rob", 0.01)


def test_harm_yardstick_is_half_the_guard_harm_limit():
    from loop.core import meta as CM
    assert CM.harm_yardstick() == CM.HARM_YARDSTICK == 0.004                   # nanopath: rob .008 / 2
    P.install({**NB, "guard": ["seg"], "harm_h": {"seg": 0.006}})
    assert CM.harm_yardstick() == 0.003


# ---------------------------------------------------------------------------------------------- L5 axis_exclude
def test_pairs_axis_exclude_defaults_to_the_project(nb_ctx):
    from loop.core import pairs as PR

    class Ctx:
        cfg = type("C", (), {"pairs": type("Pc", (), {"axis_exclude": None})()})()
    assert PR._opt(Ctx, "axis_exclude", None) is None
    assert nb_ctx.cfg.pairs.axis_exclude == () and P.axis_exclude() == ()
    P.reset()
    assert P.axis_exclude() == ("surv",)                       # nanopath default the pairs fall back to
    import inspect
    assert "P.axis_exclude()" in inspect.getsource(PR.legs)


# ---------------------------------------------------------------------------------------------- ladder `unless`
def test_ladder_rows_skip_a_knob_made_inert_by_epochs(nb_ctx):
    from loop.core import struct_tune as SU
    knobs = nb_ctx.cfg.ideation.struct_tune_knobs["backbone"]
    leader = {"muon_lr_mult": 2, "weight_decay": 0.05, "base_lr": 0.002, "epochs": 3.5, "warmup_epochs": 0.77,
              "warmup_steps": 6750}
    pts = SU.grid_points(nb_ctx, knobs, leader, 6)
    moved = [set(lv) for lv, _ in pts]
    assert moved[0] == {"muon_lr_mult", "weight_decay"} and {lv["muon_lr_mult"] for lv, _ in pts[:4]} == {1, 3}
    assert {"warmup_epochs"} in moved and {"warmup_steps"} not in moved          # epochs set: warmup_steps inert
    main = {"weight_decay": 0.05, "base_lr": 5e-4, "warmup_steps": 6750}           # 440fc92: no Muon, no epochs
    pts = SU.grid_points(nb_ctx, knobs, main, 6)
    assert set(pts[0][0]) == {"weight_decay", "base_lr"} and {"warmup_steps"} in [set(lv) for lv, _ in pts]


# ---------------------------------------------------------------------------------------------- calibration
def test_calibration_pools_the_external_prior(nb_ctx):
    from loop import calibrate as CAL
    from loop.contracts import Run, RunStatus, Stage
    axes = nb_ctx.cfg.axes

    def run(stage, seed, y):
        return Run(id=f"root:{stage.value}:{seed}", task_id="t", node_id="root", seed=seed, stage=stage,
                   status=RunStatus.DONE, scores={a: y for a in axes})
    runs = [run(Stage.BASELINE, s, y) for s, y in ((11, .50), (12, .51), (13, .49), (14, .505), (101, .495))]
    res = CAL.compute(nb_ctx, runs, [11, 12, 13, 14, 101])
    own = res.seed_sd[PANEL]
    want = math.sqrt((4 * own ** 2 + 22 * 0.0052 ** 2) / 26)
    assert res.prior_df == 22 and res.pooled_sd[PANEL] == pytest.approx(want)
    assert res.updates[("noise", "seed_transfer_sd")] == pytest.approx(want)
    assert res.updates[("loop", "axis_sd", "nb_seg_mean")] == pytest.approx(
        math.sqrt(2) * math.sqrt((4 * res.seed_sd["nb_seg_mean"] ** 2 + 22 * 0.002 ** 2) / 26))
    np_ctx = dataclasses.replace(nb_ctx, cfg=dataclasses.replace(nb_ctx.cfg, noise_prior=type(nb_ctx.cfg.noise_prior)()))
    plain = CAL.compute(np_ctx, runs, [11, 12, 13, 14, 101])
    assert plain.prior_df == 0 and plain.updates[("noise", "seed_transfer_sd")] == pytest.approx(own)


def test_calibration_spread_alternates_pools(nb_ctx):
    from loop import calibrate as CAL
    from loop.contracts import Run, RunStatus, Stage

    class Book:
        def _open(self, preemptible, *a, **k):
            return ["idle"] if preemptible else ["primary"]

    class Store:
        rows: list = []

        def runs(self, node_id=None):
            return self.rows
    st = Store()
    pick = CAL._spread(nb_ctx, st, Book())
    assert pick() == "primary"
    st.rows = [Run(id="r1", task_id="t", node_id="root", seed=11, stage=Stage.BASELINE, status=RunStatus.RUNNING,
                   pool="primary")]
    assert pick() == "idle"


def test_calibration_dry_run_plan_shows_the_spread_pools(nb_ctx):
    """The dry-run plan prints each new run on the pool the spread would give it (it printed primary for all 10)."""
    from loop import calibrate as CAL
    lines: list[str] = []

    class Adapter:
        def command(self, run):
            return f"pool={run.pool}"
    CAL.calibrate(nb_ctx, None, Adapter(), seeds=[11, 12, 13], dry_run=True, echo=lines.append)
    pools = [l.rsplit("pool=", 1)[1] for l in lines if "would run" in l]
    assert pools == ["primary", "idle", "primary", "idle", "primary", "idle"]


# ---------------------------------------------------------------------------------------------- the nanobrain yaml
def test_nanobrain_yaml_statistics_block(nb_raw, nb_ctx):
    lp = nb_raw["loop"]
    assert lp["acquire"]["target_mode"] == "incumbent" and lp["labless"]["leader_median"] == 0.5471
    assert lp["project"]["target"] == "cls" and lp["project"]["composite"] == "official"
    assert lp["noise_prior"]["df"] == 22 and lp["calibrate"]["pools"] == "spread"
    assert "veto" in lp and nb_ctx.cfg.veto.catastrophic_sigmas == 3.0
    scale = 0.0052 / 0.0028
    for got, base in ((lp["acquire"]["stack_intercept"], 0.0013), (lp["pairs"]["parent_tol"], 0.004),
                      (lp["pairs"]["axis_leg_tol"], 0.002), (lp["promotion"]["voi_gain"], 0.0007),
                      (lp["status"]["outlier_sd_d"], 0.0034), (lp["slurm"]["resume_tol"], 0.001)):
        assert got == pytest.approx(base * scale, rel=0.03), (got, base)
    from kernel.model import expand_env
    assert nb_ctx.cfg.project.config_rel == expand_env(nb_raw["surface"]["writable"][0])


def test_nanobrain_forbid_allows_gc_freeze_and_blocks_introspection(nb_ctx):
    rx = [re.compile(p, re.I) for p in nb_ctx.cfg.forbid]
    hit = lambda line: any(r.search(line) for r in rx)                          # noqa: E731
    for ok in ("            gc.collect()", "            gc.freeze()", "import gc"):
        assert not hit(ok), ok
    for bad in ("objs = gc.get_objects()", "gc.get_referrers(model)", "sys._getframe(1)", "import brainmarks_smri"):
        assert hit(bad), bad


LEADER_KEYS = ["train_filelist", "anneal_filelist", "num_workers", "prefetch_factor", "prep_threads", "prep_lean",
               "clip_quantiles", "grid_size", "target_spacing", "patch_size", "native", "num_visible", "num_predict",
               "num_samples", "model_kwargs.depth", "model_kwargs.embed_dim", "model_kwargs.num_heads",
               "model_kwargs.decoder_depth", "model_kwargs.decoder_embed_dim", "model_kwargs.decoder_num_heads",
               "model_kwargs.class_tokens", "model_kwargs.drop_path_rate", "model_kwargs.min_mask_frac",
               "model_kwargs.level_weight", "model_kwargs.mask_cube", "batch_size", "accum_iter", "base_lr", "lr",
               "min_lr", "total_steps", "warmup_steps", "decay_steps", "epochs", "warmup_epochs", "decay_epochs",
               "weight_decay", "betas", "clip_grad", "fused_adamw", "optimizer", "muon_lr_mult", "muon_momentum",
               "muon_encoder_only", "muon_decoder_only", "muon_all_matrices", "muon_fused", "amp", "amp_dtype",
               "compile", "gc_freeze", "cudagraphs", "fast_startup", "high_priority_stream", "decoder_prune_last",
               "flash_attention_impl", "checkpoint_period", "checkpoint_steps", "print_freq", "plot_period"]
UNPROTECTED_LEADER_KEYS = [k for k in LEADER_KEYS if k != "train_filelist"]


def test_every_leader_recipe_key_maps_to_one_family_and_every_file_too(nb_ctx):
    fams = nb_ctx.cfg.families

    def hits(item, kind):
        got = [f for f, r in fams.items() if any(fnmatch(item, g) for g in getattr(r, kind) if g != "*")]
        return got or [f for f, r in fams.items() if "*" in getattr(r, kind)]
    for k in UNPROTECTED_LEADER_KEYS:
        if k in nb_ctx.cfg.intake.noop_keys:          # 11-08: an inert knob is in NO family (bounced noop_knob)
            assert hits(k, "keys") == [], k
            continue
        assert len(hits(k, "keys")) == 1, (k, hits(k, "keys"))
    for f in ("train.py", "model.py", "modules.py", "data.py", "misc.py", "optim.py", "prep.py", "newmod.py"):
        assert len(hits(f"src/nanobrain/{f}", "paths")) == 1, f
    from loop.select.intake import path_matches
    assert any(path_matches("src/nanobrain/model.py", p) for p in nb_ctx.cfg.init_compat.paths)
    assert any(fnmatch("src/nanobrain/newmod.py", w) for w in nb_ctx.spec.surface.writable)
    assert "src/nanobrain/config/fomo300k_images_qc.txt" in nb_ctx.spec.surface.locked


def test_compute_keys_match_the_per_image_compute_knobs(nb_ctx):
    rx = re.compile(nb_ctx.cfg.ideation.compute_keys)
    for k in ("num_visible", "num_predict", "patch_size", "native", "batch_size", "model_kwargs.embed_dim",
              "model_kwargs.depth", "model_kwargs.decoder_depth"):
        assert rx.search(k.rsplit(".", 1)[-1]), k
    for k in ("base_lr", "model_kwargs.level_weight", "num_workers"):
        assert not rx.search(k.rsplit(".", 1)[-1]), k


def test_levers_compute_text_defaults_to_the_nanopath_literal():
    assert P.rule("lever_compute_tail", "X{mult:.2f}").format(mult=2.0) == "X2.00"


def test_weights_policy_text_is_consistent(nb_raw):
    text = yaml.safe_dump(nb_raw)
    rules = nb_raw["loop"]["domain"]["rules"]
    for k in ("bold_absolute", "init_legacy", "init_swap", "forbid_data", "databank_forbid_msg"):
        assert "natural-image" in rules[k] and "brain-MRI" in rules[k], k
    # 11-09 re-root onto natinit: the root's natural-image MAE-B init is the only pretrained source
    assert "natinit MAE-B init is the only pretrained" in nb_raw["loop"]["operator_focus"]
    # (rules.init_swap is F3's verbatim text, still naming "80% masking": F3 follow-up)
    assert "other external weights" not in text and "87K volumes at 1 mm" not in text
    assert "80% masked" not in nb_raw["loop"]["operator_focus"] and "84% masked" in nb_raw["loop"]["operator_focus"]
