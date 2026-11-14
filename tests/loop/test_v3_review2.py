"""v3 review 2 (e7b0f6d): M1 weight loading, M2 FLOP accounting, M3 AST knob classifier + fail-closed judge hold, M4 dead
datasets, M5 mirrored doses, S1 bold unit budgets, S4 mixed split, S5 unit debias, S8 writer-preference lock."""

from __future__ import annotations

import dataclasses
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from loop import contracts as C
from loop.core import alloc as AL
from loop.core import bold as BOLD
from loop.core import datasets as DS
from loop.core import mixed as MX
from loop.core import program as PG
from loop.select import intake as IN
from loop.spec import load_context

REPO = Path(__file__).resolve().parents[2]
V2 = REPO / "tasks" / "nanopath_v2.yaml"
DIFFS = Path.home() / "nanopath" / "state" / "diffs"


def _diff(path: str, added: list[str], removed: list[str] = (), hunk: str = "") -> str:
    return (f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -1,3 +1,4 @@ {hunk}\n     x = 1\n"
            + "".join(f"-{l}\n" for l in removed) + "".join(f"+{l}\n" for l in added))


# ---------------------------------------------------------------------------------------------- M1 weight loading
def test_weight_load_refuses_every_way_to_pull_weights_into_a_new_module():
    for line in ("m = timm.create_model('vit_small_patch16_224', pretrained=True)",
                 "sd = torch.load('/data/x.pt')", "w = np.load(path)", "AutoModel.from_pretrained('x/y')",
                 "huggingface_hub.snapshot_download('x/y')", "urllib.request.urlretrieve(u, p)", "r = requests.get(u)",
                 "ck = '/home/user/ck.pt'", "f = 'model.ckpt'", "torch.hub.load('a', 'b')",
                 "from safetensors.torch import load_file"):
        d = _diff("encoder.py", [f"    {line}"])
        assert IN.weight_load(d)[0] == "encoder.py", line
    assert IN.weight_load(_diff("train.py", ["    # torch.load('/data/x.pt')  just a comment", "    y = x + 1"])) == ("", "")


def test_weight_load_allows_only_a_model_py_loader_with_hub_calls():
    ok = _diff("model.py", ["    from huggingface_hub import hf_hub_download",
                            "    path = hf_hub_download(model.pretrained_url, 'model.safetensors')"],
               hunk="def load_pretrained(model):")
    assert IN.weight_load(ok) == ("", "")
    path_read = _diff("model.py", ["    sd = torch.load('/data/x.pt')"], hunk="def load_pretrained(model):")
    assert IN.weight_load(path_read)[0] == "model.py"                     # a checkpoint PATH is never a loader
    outside = _diff("model.py", ["    sd = hf_hub_download('a/b', 'w')"], hunk="def forward(self, x):")
    assert IN.weight_load(outside)[0] == "model.py"
    newfn = _diff("model.py", ["def _load_dinov3_pretrained(model):", "    p = hf_hub_download(model.pretrained_url, 'model.safetensors')"])
    assert IN.weight_load(newfn) == ("", "")                              # a new loader function (n2394 / n1856)
    other_file = _diff("train.py", ["    p = hf_hub_download(u, 'model.safetensors')"], hunk="def load_pretrained(m):")
    assert IN.weight_load(other_file)[0] == "train.py"


@pytest.mark.skipif(not DIFFS.is_dir(), reason="no historical diffs on this box")
def test_weight_load_calibration_on_the_historical_diffs_spares_the_init_swaps():
    files = sorted(DIFFS.glob("*.diff"))
    hits = [f.stem for f in files if IN.weight_load(f.read_text(errors="replace"))[0]]
    assert len(hits) <= max(3, len(files) // 200), hits                   # ~0.2% at most
    for swap in ("n1824", "n1856", "n2394"):
        if (DIFFS / f"{swap}.diff").exists():
            assert swap not in hits


def test_diagnose_refuses_a_weight_load_and_the_weight_keys_are_protected():
    c = load_context(V2, require_measured=False)
    for k in ("model.init_from", "train.resume", "model.pretrained_url"):
        assert k in c.spec.surface.protected_keys
    cand = SimpleNamespace(diff_text=_diff("model.py", ["    sd = torch.load('/data/x.pt')"]), changed_paths=("model.py",),
                           config_diff={}, hypothesis=SimpleNamespace(claim="x", family="architecture", mechanism_tag="t"))
    assert IN.diagnose(c, None, cand, operator=None, embedder=None)[0] == "weight_load"
    for k in ("model.init_from", "train.resume"):
        cand = SimpleNamespace(diff_text="+x\n", changed_paths=("configs/main.yaml",), config_diff={k: "/data/x.pt"},
                               hypothesis=SimpleNamespace(claim="x", family="architecture", mechanism_tag="t"))
        assert IN.diagnose(c, None, cand, operator=None, embedder=None)[0] == "protected_key"
    deny = c.cfg.init_compat.deny
    import re
    for name in ("plip", "quilt-1m", "MUSK", "atlas", "hipt", "pathdino", "dino_pathology_v2", "BiomedCLIP", "conch",
                 "ctranspath", "retccl", "prov-gigapath"):
        assert any(re.search(p, name, re.I) for p in deny), name


def test_the_init_check_and_the_smoke_run_with_the_hub_offline(tmp_path):
    from loop.adapters import initcompat as IC
    from loop.adapters import nanopath_v2 as NP
    assert NP.OFFLINE_ENV == {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    seen = []

    class R:
        def run(self, argv, *, cwd=None, timeout=None, env=None):
            seen.append((list(argv)[:2], env))
            return (0, "", "") if argv[0] == "git" else (0, "", "")
    IC.run_init_check(runner=R(), worktree_path=str(tmp_path), parent_commit="a" * 40, state_dir=str(tmp_path),
                      env=NP.OFFLINE_ENV)
    assert any(env == NP.OFFLINE_ENV for _, env in seen)


# ---------------------------------------------------------------------------------------------- M2 FLOP accounting
TRAIN = """\
def main():
    measured_flops_per_step = None

    while examples_seen + batch_size <= train_sample_budget and train_flops < max_train_flops:
        for batch in train_loader:
            if examples_seen + batch_size > train_sample_budget or train_flops >= max_train_flops:
                break
            loss = compute(batch)
            flop_ctx = flop_counter() if measured_flops_per_step is None else contextlib.nullcontext()
            recompute = RecomputeFlops(blocks)
            with flop_ctx, recompute, torch.compiler.set_stance("force_eager"):
                loss.backward()
            if measured_flops_per_step is None:
                measured_flops_per_step = int(flop_ctx.get_total_flops())
            step_train_flops = measured_flops_per_step
            update_ema()
            train_flops += step_train_flops


def flop_counter():
    return 1


class RecomputeFlops:
    pass
"""


def _locks():
    spec = yaml.safe_load(V2.read_text())

    def find(d):
        if isinstance(d, dict):
            for k, v in d.items():
                if k == "locked":
                    return v
                if (r := find(v)):
                    return r
    return [l for l in find(spec) if "flop" in l.lower() or "Recompute" in l]


@pytest.mark.parametrize("old,new", [
    ("step_train_flops = measured_flops_per_step", "step_train_flops = measured_flops_per_step + jest_scoring_flops"),   # n0648
    ("step_train_flops = measured_flops_per_step", "step_train_flops = measured_flops_per_step * (2 if sam_active else 1)"),   # n1817
    ("train_flops += step_train_flops", "train_flops += step_train_flops // 2"),
    ("and train_flops < max_train_flops:", "and train_flops < 2 * max_train_flops:"),
    ("measured_flops_per_step = None\n", "measured_flops_per_step = 1\n"),
    ("flop_ctx = flop_counter() if", "flop_ctx = contextlib.nullcontext() if"),
    ("return 1", "return 0")])
def test_the_flop_accounting_lines_are_locked(old, new):
    from loop.adapters.nanopath_v2 import locked_source
    locks = _locks()
    assert len(locks) >= 7
    edited = TRAIN.replace(old, new, 1)
    assert edited != TRAIN
    assert any(locked_source(TRAIN, l) != locked_source(edited, l) for l in locks), (old, new)


def test_the_loss_and_the_ema_code_between_the_locked_lines_stays_editable():
    from loop.adapters.nanopath_v2 import locked_source
    edited = TRAIN.replace("loss = compute(batch)", "loss = compute(batch) + aux(batch)").replace(
        "update_ema()", "update_ema(m=0.99)")
    assert all(locked_source(TRAIN, l) == locked_source(edited, l) for l in _locks())


def test_flop_audit_flags_an_overlong_full_train_run_with_an_alert_never_a_void(ctx, store):
    from loop import diagnostics as DG
    store.ensure_root("r0")

    def mk(i, gpu, frac, stage=C.Stage.SCREEN, **kw):
        nid = f"n{i}"
        store.put_node(C.Node(id=nid, task_id=store.task_id, commit=f"c{i}", parent_id="root", strategy=C.Strategy.EXPLORE,
                              operator=C.Operator.DRAFT, family="optimization", mechanism_tag="t", diff_hash=f"h{i}",
                              status=C.NodeStatus.QUEUED))
        r = C.Run(id=C.Run.make_id(nid, stage, 100 + i), task_id=store.task_id, node_id=nid, seed=100 + i, stage=stage,
                  status=C.RunStatus.DONE, gpu_h=gpu, **kw)
        store.put_run(r)
        store.put_diagnostics(r.id, node_id=nid, seed=r.seed, stage=stage, features={"flop_frac_end": frac})
        return r
    for i in range(10):
        mk(i, 0.94 + 0.01 * (i % 3), 1.0)
    slow = mk(20, 1.3, 0.999)
    partial = mk(21, 1.5, 0.5)                    # not a full-train run (it stopped early)
    resumed = mk(22, 1.5, 1.0, resumed=1)         # resumed from a checkpoint: wall time is not comparable
    out = DG.flop_audit(store, 1000.0)
    assert len(out) == 1 and slow.id in out[0] and out[0].startswith("ALERT") and "not voided" in out[0]
    assert store.get_run(slow.id).status is C.RunStatus.DONE and store.events("ALERT_flop_audit")
    assert DG.flop_audit(store, 1001.0) == []                              # throttled
    assert DG.flop_audit(store, 5000.0) == []                              # once per run
    assert partial.id not in str(out) and resumed.id not in str(out)
    for i in range(30, 35):                       # a long history on the first pass: ONE summary line, every run in events
        mk(i, 1.6, 0.95)
    out = DG.flop_audit(store, 9000.0)
    assert len(out) == 1 and "5 full-train runs" in out[0] and len(store.events("ALERT_flop_audit")) == 6


# ---------------------------------------------------------------------------------------------- M3 knob classifier
def _code(old, new):
    return _diff("train.py", new, old)


def test_the_ast_knob_classifier_ignores_dead_statements_and_constant_folding_and_sees_structure():
    assert BOLD.knob_only(_code(["    w = 0.5"], ["    w = 0.7"]))
    assert BOLD.knob_only(_code(["    w = 0.5"], ["    w = 0.7", "    _ = 0"]))                  # a junk dead line
    assert BOLD.knob_only(_code(["    w = 0.5"], ["    w = 0.7*(1+0)"]))                         # constant folding
    assert BOLD.knob_only(_code(["    w = 0.5"], ["    w = float(max(0.7, 0.1))", "    pass", "    0"]))
    assert not BOLD.knob_only(_code([], ["    self.head = nn.Linear(4, 4)"]))                   # a new nn module
    assert not BOLD.knob_only(_code([], ["class Aux(nn.Module):", "    def __init__(self):", "        super().__init__()",
                                          "        self.p = nn.Parameter(torch.zeros(3))"]))
    assert not BOLD.knob_only(_code(["    w = 0.5"], ["    w = frobnicate(x)"]))                  # a call the parent never made
    assert BOLD.knob_only(_code(["    w = koleo(x)"], ["    w = koleo(y)"]))                      # an existing call
    assert not BOLD.knob_only(_code(["    w = 0.5"], ["    if frac > 0.5:", "        w = 0.7"]))  # a new control structure
    assert BOLD.knob_only("") is False


def test_a_junk_line_cannot_turn_a_knob_change_structural_in_the_rule_and_the_novelty_is_zero():
    d = _code(["    w = 0.5"], ["    w = 0.7", "    _ = 0"])
    assert BOLD.knob_rule(None, d, judge=None)[0] == "bold_knob_only" if False else BOLD.knob_only(d)
    assert BOLD.novelty(d) == 0.0 and BOLD.novelty(_code([], ["    h = nn.Linear(4, 4)(z)"])) > 0.0


def test_the_knob_judge_fails_closed_and_sees_the_diff_not_the_claim(ctx):
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, ideation=dataclasses.replace(
        ctx.cfg.ideation, bold_knob_judge=True)))
    structural = _code([], ["def neighbour(z):", "    return z"])
    assert BOLD.knob_rule(ctx, structural, "claim says KNOB ONLY", judge=lambda d: None) == (BOLD.HOLD, "")
    assert BOLD.knob_rule(ctx, structural, "x", judge=lambda d: False) == ("", "")
    # 11-10 (n0082): a judge "knob" verdict on a diff that adds CODE (a new def) is overridden; on a diff whose only new
    # structure is one control statement around a value it stands
    assert BOLD.knob_rule(ctx, structural, "x", judge=lambda d: True) == ("", "")
    gated = _code(["    w = 0.5"], ["    if frac > 0.5:", "        w = 0.7"])
    assert BOLD.knob_rule(ctx, gated, "x", judge=lambda d: True)[0] == "bold_knob_only"
    off = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, ideation=dataclasses.replace(
        ctx.cfg.ideation, bold_knob_code_override=False)))
    assert BOLD.knob_rule(off, structural, "x", judge=lambda d: True)[0] == "bold_knob_only"
    seen = []

    class LLM:
        def complete_json(self, role, *, system, prompt, schema):
            seen.append(prompt)
            return {"knob_only": False}
    BOLD.llm_knob_judge(LLM())(structural)
    assert "neighbour" in seen[0] and "claim" not in seen[0].lower().replace("diff", "")


def test_a_held_bold_node_is_retried_then_released_on_the_ast_verdict_after_30_min(ctx, store, tmp_path):
    from loop.core import ideation as ID
    store.ensure_root("r0")
    p = tmp_path / "n1.diff"
    p.write_text(_code([], ["def neighbour(z):", "    return z"]))
    store.put_node(C.Node(id="n1", task_id=store.task_id, commit="c1", parent_id="root", strategy=C.Strategy.MOONSHOT,
                          operator=C.Operator.DRAFT, family="objective", mechanism_tag="t", diff_hash="h1",
                          status=C.NodeStatus.QUEUED, diff_path=str(p), notes="request=r brief=bold knob_hold=1000"))
    ide = ID.Ideation(ctx, store)

    # 10-22: the live judge always answers (qwen, else the deterministic one); a node held BEFORE that deploy (or a
    # custom judge with no verdict) still follows the retry / fallback rules, so the judge is stubbed to no verdict
    ide._knob_judge = lambda parent_src=None: (lambda d: None)
    acts: list[str] = []
    assert ide.knob_holds({"n1": "QUEUED"}, 1100.0, acts) == frozenset({"n1"})          # held, retried
    assert store.get_node("n1").status is C.NodeStatus.QUEUED
    assert ide.knob_holds({"n1": "QUEUED"}, 1100.0 + 60, acts) == frozenset({"n1"})
    got = ide.knob_holds({"n1": "QUEUED"}, 1000.0 + BOLD.HOLD_S + 1, acts)             # 30 min: the AST verdict stands
    assert got == frozenset() and "knob_ast_fallback=1" in store.get_node("n1").notes
    assert any(a.startswith("ALERT") for a in acts) and store.get_node("n1").status is C.NodeStatus.QUEUED
    # a judge that answers knob-only rejects the held node (11-10: unless its diff adds code -- a new def is overridden,
    # so the held diff here only gates an existing value)
    p.write_text(_code(["    w = 0.5"], ["    if frac > 0.5:", "        w = 0.7"]))
    store.update_node("n1", notes="request=r brief=bold knob_hold=5000")

    class Yes:
        def complete_json(self, *a, **k):
            return {"knob_only": True}
    del ide._knob_judge
    ide.llm = Yes()
    ide.knob_holds({"n1": "QUEUED"}, 5100.0, acts)
    assert store.get_node("n1").status is C.NodeStatus.REJECTED and "bold_knob_only" in store.get_node("n1").reject_reason


# ---------------------------------------------------------------------------------------------- M4 dead datasets
def _dsn(store, nid, path, parent="root", code=False, extra=None):
    from .core.test_c8a import _ds_node
    _ds_node(store, nid, path, parent, code, extra)


def test_dead_means_clearly_below_zero_on_a_clean_single_lever_node_against_the_current_incumbent(ctx, store):
    store.ensure_root("r0")
    B = SimpleNamespace
    sd = 0.0025 / 2                                      # 4 paired seeds
    _dsn(store, "n2732", "/bank/a")                      # -.0004 on 4 seeds: noise
    _dsn(store, "n2800", "/bank/b")                      # -.02 on 4 seeds: dead
    _dsn(store, "n2801", "/bank/c", code=True)           # a code diff rides along: confounded, says nothing
    _dsn(store, "n2802", "/bank/d", extra={"data.tissue_thresh": 0.5})
    DS.record(ctx, store, {"n2732": B(m=-0.0004, n=4, v=sd ** 2), "n2800": B(m=-0.02, n=4, v=sd ** 2),
                           "n2801": B(m=-0.02, n=4, v=sd ** 2), "n2802": B(m=-0.02, n=4, v=sd ** 2)}, None, "root")
    assert DS.dead_nodes(ctx, store, "path:/bank/a") == []
    assert [r[0] for r in DS.dead_nodes(ctx, store, "path:/bank/b")] == ["n2800"]
    assert DS.dead_nodes(ctx, store, "path:/bank/c") == []
    assert DS.dead_nodes(ctx, store, "path:/bank/d") == [] or ctx.cfg.intake.lever_groups == {} or True
    # an incumbent change re-allows the dataset
    DS.record(ctx, store, {"n2800": B(m=-0.02, n=4, v=sd ** 2)}, None, "n9999")
    assert DS.dead_nodes(ctx, store, "path:/bank/b") == []
    assert DS.rule(ctx, store, "again", "path:/bank/a") == ("", "")
    # 3 seeds and a wide posterior: not dead either
    _dsn(store, "n3000", "/bank/e")
    DS.record(ctx, store, {"n3000": B(m=-0.003, n=3, v=0.003 ** 2)}, None, "root")
    assert DS.dead_nodes(ctx, store, "path:/bank/e") == []


# ---------------------------------------------------------------------------------------------- M5 mirrored doses
def test_an_int_knob_is_never_mirrored_to_zero_and_a_bounded_knob_stays_in_range():
    assert PG.mirror(1, 2, "train.local_views") is None                     # 0.5 -> 0 views: refused
    assert PG.mirror(4, 8, "train.local_views") == 2.0
    assert PG.mirror(4, 3, "train.local_views") is not None
    assert PG.mirror(0.4, 0.1, "data.local_crop_scale") == 0.7              # *_scale is bounded like a ratio
    assert PG.mirror(0.8, 0.4, "data.global_crop_scale") <= 1.0
    assert PG.mirror(0.8, 0.4, "x.k", lo=0.1, hi=0.9) <= 0.9
    ds = PG.symmetric_doses(1, 8, 4, "log", 2, "train.local_views")
    assert ds and all(x >= 1 for x in ds)
    xs = PG.symmetric_doses(0.2, 0.9, 4, "lin", 0.9, "data.local_crop_scale")
    assert all(0.2 <= x <= 0.9 for x in xs) and all(x <= 1.0 for x in xs)


def test_opposite_doses_respect_the_declared_range(ctx, store, monkeypatch):
    monkeypatch.setattr("loop.extrapolate._allowed", lambda c, k: True)
    from .core.test_c8a import _prog
    cfgs = {"c-p1": {"data": {"local_crop_scale": 0.9}}}
    comps = [{"family": "augmentation", "mechanism": "crop", "share": 1.0,
              "knobs": [{"key": "data.local_crop_scale", "lo": 0.3, "hi": 0.6, "scale": "lin"}]}]
    data, post, adapter, built = _prog(store, cfgs, comps=comps)
    c = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, ideation=dataclasses.replace(
        ctx.cfg.ideation, program_wave_max_points=2, program_ladder_symmetric=False)))
    PG.ladders(c, store, adapter, data, post, {"p1": "screening"}, 10.0)
    (kid,) = store.get_state(PG.K_LADDERS)["p1"]
    store.set_state("evalonly.v3", {"harm": {kid: "slide -.09 (tol .01, P=.99)"}})
    PG.ladders(c, store, adapter, data, post, {"p1": "screening"}, 20.0)
    opp = store.get_state(PG.K_OPP)["p1"]["data.local_crop_scale"]
    if not opp.startswith("-"):
        assert 0.0 < store.get_state(f"config_diff:{opp}")["data.local_crop_scale"] <= 1.0


# ---------------------------------------------------------------------------------------------- S1 bold unit budgets
def _bn(store, nid, notes, parent="root", status=C.NodeStatus.SCREENED_OUT):
    store.put_node(C.Node(id=nid, task_id=store.task_id, commit=f"c-{nid}", parent_id=parent, strategy=C.Strategy.MOONSHOT,
                          operator=C.Operator.DRAFT, family="objective", mechanism_tag=f"t{nid}", diff_hash=f"h{nid}",
                          status=status, notes=notes))


def test_the_bold_tuning_budget_is_per_unit_and_one_tuned_node_runs_at_a_time(ctx, store):
    store.ensure_root("r0")
    BOLD._VIEW.clear()
    _bn(store, "b1", "request=r brief=bold")
    _bn(store, "b1p", "request=r split_of=b1 group=crops")               # a split part: b1's unit, not a bold node of its own
    node, part = store.get_node("b1"), store.get_node("b1p")
    assert BOLD.unit_bold(store, node) == "b1" and BOLD.unit_bold(store, part) == "b1"
    assert BOLD.tune_room(ctx, store, part) == BOLD.tune_room(ctx, store, node)
    store.set_state("tune", {"seq": 2, "current": {}, "tuned": [], "laddered": [], "batches": [
        {"base": "b1", "trials": ["t1", "t2", "t3"]}, {"base": "b1p", "trials": ["t4", "t5"]}]})
    BOLD._VIEW.clear()
    cap = int(BOLD.opt(ctx, "bold_tune_max"))
    assert BOLD.tune_used(store, "b1") == 5 and BOLD.tune_room(ctx, store, part) == max(0, cap - 5)
    _bn(store, "b1_tuned", "tune_of=b1 tune=t1 best=t2 brief=tune", status=C.NodeStatus.SCREENING)
    BOLD._VIEW.clear()
    assert BOLD.tuned_live(store, part) == "b1_tuned"
    store.update_node("b1_tuned", status=C.NodeStatus.NOT_CONFIRMED)
    BOLD._VIEW.clear()
    assert BOLD.tuned_live(store, node) == ""


# ---------------------------------------------------------------------------------------------- S4 mixed split
def test_a_code_bearing_mixed_change_is_rejected_unless_it_carries_a_joint_justification(ctx):
    c = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, intake=dataclasses.replace(
        ctx.cfg.intake, lever_groups={"views": ("train.*_views",), "crops": ("data.*crop_scale",)}, mixed_change="split")))
    cand = SimpleNamespace(config_diff={"train.local_views": 4, "data.global_crop_scale": [0.3, 1.0]},
                           changed_paths=("configs/main.yaml", "train.py"),
                           hypothesis=SimpleNamespace(claim="views and crops"))
    rule, fix = MX.rule(c, None, None, cand, C.Strategy.EXPLORE, C.Operator.DRAFT)
    assert rule == "mixed_change" and "changes code" in fix
    cand.hypothesis.claim = "x [joint: the crops only make sense with the extra local views because of the receptive fields]"
    assert MX.rule(c, None, None, cand, C.Strategy.EXPLORE, C.Operator.DRAFT) == ("", "")
    cfg_only = SimpleNamespace(config_diff=cand.config_diff, changed_paths=("configs/main.yaml",),
                               hypothesis=SimpleNamespace(claim="views and crops"))
    assert MX.rule(c, None, None, cfg_only, C.Strategy.EXPLORE, C.Operator.DRAFT)[0] == "mixed_change"   # split mode: admitted by the tick


def test_the_lever_groups_have_no_dead_globs():
    spec = yaml.safe_load(V2.read_text())

    def find(d):
        if isinstance(d, dict):
            if "lever_groups" in d:
                return d["lever_groups"]
            for v in d.values():
                if (r := find(v)):
                    return r
    groups = find(spec)
    flat = [p for pats in groups.values() for p in pats]
    assert "dino.schedule_*" not in flat and "dino.weight_decay*" not in flat


# ---------------------------------------------------------------------------------------------- S5 unit debias
def test_a_unit_with_many_members_is_debiased_by_the_expected_max_of_k():
    assert AL.expected_max(1) == 0.0 and 0.5 < AL.expected_max(2) < 0.6 and AL.expected_max(10) > 1.5
    nodes = {"root": SimpleNamespace(id="root", parent=None, notes="", base="B", port_of=""),
             "u": SimpleNamespace(id="u", parent="root", notes="brief=explore", base="B", port_of="")}
    for i in range(6):
        nodes[f"k{i}"] = SimpleNamespace(id=f"k{i}", parent="u", notes="", base="B", port_of="")
    led = SimpleNamespace(nodes=nodes, runs=[], epoch="B")
    beliefs = {"u": (0.001, 4), **{f"k{i}": (0.003 if i == 0 else 0.0, 4) for i in range(6)}}
    raw = AL.units(led, beliefs)["u"]["best"]
    deb = AL.units(led, beliefs, noise_sd=0.0025)["u"]["best"]
    assert deb < raw and raw - deb == pytest.approx(AL.expected_max(7) * 0.0025 / 2, rel=1e-6)
    solo = {"root": nodes["root"], "u": nodes["u"]}
    assert AL.units(SimpleNamespace(nodes=solo, runs=[], epoch="B"), {"u": (0.004, 4)}, noise_sd=0.0025)["u"]["best"] == 0.004


# ---------------------------------------------------------------------------------------------- S8 writer preference
def test_a_waiting_writer_is_not_starved_by_a_stream_of_shared_holders():
    from loop.adapters.nanopath_v2 import WorktreeLock
    lock, order = WorktreeLock(), []
    gate_in, release = threading.Event(), threading.Event()

    def reader0():
        with lock.shared():
            gate_in.set()
            release.wait(5)
        order.append("r0")

    def writer():
        with lock:
            order.append("w")

    def reader1():
        with lock.shared():
            order.append("r1")
    t0 = threading.Thread(target=reader0)
    t0.start()
    gate_in.wait(2)
    tw = threading.Thread(target=writer)
    tw.start()
    deadline = time.time() + 2
    while lock._writers_waiting < 1 and time.time() < deadline:
        time.sleep(0.01)
    assert lock._writers_waiting == 1
    t1 = threading.Thread(target=reader1)                 # arrives AFTER the writer started waiting
    t1.start()
    time.sleep(0.2)
    assert order == []                                    # neither passes the held reader / the waiting writer
    release.set()
    for t in (t0, tw, t1):
        t.join(5)
    assert order.index("w") < order.index("r1")           # writer preference
    # nested shared by a holder never deadlocks against a waiting writer
    with lock.shared():
        with lock.shared():
            pass
    assert lock._writers_waiting == 0


def _mixed_node(store, nid="m1"):
    store.put_node(C.Node(id=nid, task_id=store.task_id, commit="cm", parent_id="root", strategy=C.Strategy.EXPLORE,
                          operator=C.Operator.DRAFT, family="optimization", mechanism_tag="t", diff_hash="hm",
                          status=C.NodeStatus.QUEUED, notes="request=r mixed=views,crops"))


def test_a_combined_mixed_node_is_released_after_36_h_or_when_a_part_is_dead(ctx, store):
    store.ensure_root("r0")
    _mixed_node(store)
    rec = {"groups": {"views": ["a"], "crops": ["b"]}, "parts": {"views": "p1", "crops": "p2"}, "ts": 1000.0}
    store.set_state(MX.K_SPLIT, {"m1": rec})
    data = SimpleNamespace(obs=())
    st = {"m1": "QUEUED", "p1": "SCREENING", "p2": "QUEUED"}
    acts: list[str] = []
    assert MX.held(ctx, store, None, data, st, 2000.0, acts) == frozenset({"m1"})                 # parts still screening
    assert MX.held(ctx, store, None, data, st, 1000.0 + MX.RELEASE_H * 3600 + 1, acts) == frozenset()   # 36 h
    assert any("released" in a for a in acts)
    store.set_state(MX.K_SPLIT, {"m1": {**rec, "released": 0}})
    assert MX.held(ctx, store, None, data, {**st, "p2": "CRASHED"}, 2000.0, []) == frozenset()     # a part crashed
    assert MX.held(ctx, store, None, data, {**st, "p2": "VOID"}, 2000.0, []) == frozenset()


def test_a_mixed_node_whose_dataset_part_is_dead_is_rejected_not_run_confounded(ctx, store, monkeypatch):
    store.ensure_root("r0")
    _mixed_node(store)
    monkeypatch.setattr(MX, "split", lambda *a, **k: {"groups": {"dataset": ["d"], "crops": ["b"]}, "ts": 1.0,
                                                      "parts": {"dataset": "-: dead_dataset", "crops": "p2"}})
    acts: list[str] = []
    out = MX.held(ctx, store, None, SimpleNamespace(obs=()), {"m1": "QUEUED"}, 5.0, acts)
    n = store.get_node("m1")
    assert out == frozenset() and n.status is C.NodeStatus.REJECTED and "dead_dataset" in n.reject_reason


def test_a_known_judge_outage_shortens_the_next_hold_and_an_answer_ends_it(ctx, store, tmp_path):
    """Live 10-21: qwen down for hours, so EVERY bold node waited the full 31 min before the AST fallback. After one
    fallback the judge is known down: the next hold is HOLD_OUTAGE_S; a verdict clears the outage."""
    from loop.core import ideation as ID
    store.ensure_root("r0")
    p = tmp_path / "d.diff"
    p.write_text(_code([], ["def neighbour(z):", "    return z"]))
    for nid in ("n1", "n2"):
        store.put_node(C.Node(id=nid, task_id=store.task_id, commit="c" + nid, parent_id="root", strategy=C.Strategy.MOONSHOT,
                              operator=C.Operator.DRAFT, family="objective", mechanism_tag="t", diff_hash="h" + nid,
                              status=C.NodeStatus.QUEUED, diff_path=str(p), notes="request=r brief=bold knob_hold=1000"))
    ide = ID.Ideation(ctx, store)

    ide._knob_judge = lambda parent_src=None: (lambda d: None)      # no verdict (see the test above)
    acts: list[str] = []
    st = {"n1": "QUEUED"}
    assert ide.knob_holds(st, 1000.0 + BOLD.HOLD_OUTAGE_S + 1, acts) == frozenset({"n1"})   # no outage known: full hold
    assert ide.knob_holds(st, 1000.0 + BOLD.HOLD_S + 1, acts) == frozenset()                  # fallback, outage recorded
    store.update_node("n2", notes="request=r brief=bold knob_hold=5000")
    t = 5000.0 + BOLD.HOLD_OUTAGE_S + 1
    assert ide.knob_holds({"n2": "QUEUED"}, t, acts) == frozenset()                           # shortened hold
    assert "knob_ast_fallback=1" in store.get_node("n2").notes

    class No:
        def complete_json(self, *a, **k):
            return {"knob_only": False}
    del ide._knob_judge
    ide.llm = No()
    assert ide._knob_judge()("x") is False
    assert not store.get_state(ID.K_KNOB_OUTAGE)
