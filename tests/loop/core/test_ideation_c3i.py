"""v3-c3i (meta-analysis 10-13): ambition from the gap, the computed "untested levers & deficits" brief section, the
program (moonshot) kind, fork-and-flush routing by brief kind + screen credit, the factorial lane's idea value, the
data path (intake, dataset_spec-only finishes, banked datasets run on the incumbent), and the intake pre-check."""

from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace

import pytest
import yaml

from loop import databank as DB
from loop import tick as TK
from loop.agents import researcher as RS
from loop.contracts import (CandidateRequest, Node, NodeStatus, Operator, ROOT_NODE_ID, Strategy)
from loop.core import acquire as A
from loop.core import data as D
from loop.core import factorial as F
from loop.core import fnf as FN
from loop.core import ideation as ID
from loop.core import levers as LV
from loop.core import model as M
from loop.core.status import Status
from loop.select import intake as IN
from loop.spec import FamilyRule

from ..conftest import InlineExecutor
from ..test_databank import SPEC, _ctx, _ds
from .test_fnf import _Research, _led, _node, _round, _run, env  # noqa: F401  (the fixture)


def _ideation_cfg(ctx, **kw):
    return dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, ideation=dataclasses.replace(
        ctx.cfg.ideation, **kw)))


# ---------------------------------------------------------------------------------------------- 1. ambition
def test_target_mode_asks_for_a_share_of_the_gap_not_the_mde(env, monkeypatch):
    ctx, store, led, ide = env
    v = ide.view(led)
    monkeypatch.setattr(ID, "target_gap", lambda v: 0.0062)
    b = ide.brief_for("explore", led, v, "explore")
    # v3-c6i: no size floor at all (claims clustered at whatever minimum the brief named): the gap is the reference
    assert "is +0.0062 above the incumbent" in b.text and "multi-component PROGRAM" in b.text
    assert "at least 0.5 x the gap" not in b.text and f"at least {v.mde():.4f}" not in b.text
    assert "P(Δ >= 0.0062)" in b.text                                  # the candidates' p_gap is stated against it
    assert "## Untested levers & deficits (computed)" in b.text and "### (e) Refuted" in b.text
    monkeypatch.setattr(ID, "target_gap", lambda v: None)
    assert f"about {v.mde():.4f} (2σ_C/√5" in ide.brief_for("explore", led, v, "explore").text
    br = ide.brief_for("explore", led, v, "explore", branch="1.1")      # a flushed branch keeps the computed levers
    assert f"about {v.mde():.4f}" in br.text and "## Untested levers" in br.text   # (c4i: only shared notes go)
    assert "## Per-axis headroom" in br.text and "## Computed facts" not in br.text


# ---------------------------------------------------------------------------------------------- 3. program kind
def test_program_share_in_target_mode_is_a_moonshot_request_with_moonshot_caps(env, monkeypatch):
    ctx, store, led, ide = env
    object.__setattr__(ctx.cfg.ideation, "kind_weight", False)        # the fixed rotation (A3 kind weights off)
    monkeypatch.setattr(ID, "target_gap", lambda v: 0.006)
    monkeypatch.setattr(FN, "SHARE", 0.0)                              # no branches: the shared briefs only
    tick = SimpleNamespace(research=_Research(4), pool=InlineExecutor(), dry=False)
    ide.ideate(tick, [SimpleNamespace(kind="idea", lane="explore")] * 4, led, {})
    reqs = tick.research.started
    assert [r.brief_kind == "program" for r in reqs] == [True, False, False, False]    # .25: 1 of 4, no burst
    prog = reqs[0]
    assert prog.strategy is Strategy.MOONSHOT and prog.operator is Operator.DRAFT and prog.brief.index("PROGRAM (multi-component") < prog.brief.index("## Per-axis headroom")
    assert store.get_state(ID.K_PROGRAM) == {"n": 4, "p": 1}
    r = ctx.cfg.researcher
    assert RS.LLMResearcher(ctx, None, None).limits(prog) == (r.moonshot_max_turns, r.moonshot_timeout_min)
    assert RS.LLMResearcher(ctx, None, None).limits(reqs[1]) == (r.max_turns, r.timeout_min)
    monkeypatch.setattr(ID, "target_gap", lambda v: None)              # outside target mode: program_share_base 0
    assert ide.kind_for("explore", ide.view(led)) != "program"
    ide.ctx = _ideation_cfg(ctx, program_share_base=1.0)
    assert ide.kind_for("explore", ide.view(led)) == "program"


def test_a_program_node_gets_a_dose_ladder_within_the_moonshot_budget_never_a_wave(store, ctx, monkeypatch):
    from loop.core import program as PG
    store.ensure_root("r0")
    _store_node(store, "p1", ROOT_NODE_ID, "multi_part", NodeStatus.SCREENING, family="objective")
    _store_node(store, "x1", ROOT_NODE_ID, "single", NodeStatus.SCREENING, family="objective")
    nodes = {n: D.Node(n, ROOT_NODE_ID, None, "moonshot", "objective", "explore", ROOT_NODE_ID, 1.0, "screening",
                       notes, base="B")
             for n, notes in (("p1", "request=r1 brief=program"), ("x1", "request=r2 brief=explore"))}
    data = SimpleNamespace(nodes=nodes, port=lambda n, b=None: n, epoch="B")
    knobs = [{"key": "loss.w", "lo": 0.25, "hi": 4.0, "scale": "log"}, {"key": "loss.v", "lo": 0, "hi": 1}]
    for n in ("p1", "x1"):
        store.set_state(f"ideation:{n}", {"knobs": knobs})
    post = SimpleNamespace(by={}, est={"p1": {}, "x1": {}}, epoch="B", prior=None, belief=lambda n: (0.002, 4e-6))
    monkeypatch.setattr(F, "factorial_tau2", lambda prior, n: 1e-6)
    st = {"p1": Status.SCREENING, "x1": Status.SCREENING}
    core = TK.ModuleCore()
    core.prepare(SimpleNamespace(ctx=ctx, store=store), data, st, post)
    assert set(core.knobs) == {"x1"}                                     # a program never gets an 8-point wave
    built = []
    adapter = SimpleNamespace(config_rel="config.yaml", config_at=lambda c: {"loss": {"w": 1.0}},
                              dry_build=lambda commit, levels, knobs, diffs, name: built.append(levels) or
                              (f"sha-{len(built)}", ""))
    post.belief = lambda n: (-0.001, 4e-6)                               # -0.5 sd: still laddered
    acts = PG.ladders(ctx, store, adapter, data, post, st, 10.0)
    made = store.get_state(PG.K_LADDERS)["p1"]
    assert "x1" not in store.get_state(PG.K_LADDERS) and len(made) == ctx.cfg.ideation.program_wave_max_points - 1
    # v3-c8a B1: symmetric around the node's own 1.0 (half below, half above, geometric for a log knob)
    assert [store.get_state(f"config_diff:{n}")["loss.w"] for n in made] == [0.5, 0.25, 2.0, 4.0]
    assert all(store.get_node(n).parent_id == "p1" and "ladder=p1" in store.get_node(n).notes for n in made)
    assert D.lane_of("explore", store.get_node(made[0]).notes) == "factorial" and acts
    assert PG.ladders(ctx, store, adapter, data, post, st, 20.0) == []   # once per node
    assert PG.doses(0.25, 4.0, 4, "log", cur=1.0) == [0.25, 0.629961, 1.5874, 4.0]
    assert PG.doses(0.0, 1.0, 3, "lin", cur=0.5) == [0.0, 1.0]           # the node's own dose is no new trial
    store.set_state(PG.K_LADDERS, {})
    post.belief = lambda n: (-0.005, 4e-6)                               # -2.5 sd: clearly negative, no ladder
    assert PG.ladders(ctx, store, adapter, data, post, st, 30.0) == []


# ---------------------------------------------------------------------------------------------- 5. fork-and-flush
def test_fnf_routes_by_brief_kind_so_factorial_lane_explore_briefs_branch_too(env):
    ctx, store, led, ide = env
    object.__setattr__(ctx.cfg.ideation, "kind_weight", False)        # the fixed rotation (A3 kind weights off)
    tick = SimpleNamespace(research=_Research(4), pool=InlineExecutor(), dry=False)
    ide.ideate(tick, [SimpleNamespace(kind="idea", lane="factorial")] * 4, led, {})
    reqs = tick.research.started
    assert {r.brief_kind for r in reqs} <= set(ID.EXPLORE_KINDS)
    assert any(r.branch for r in reqs)                                   # was: only lane == "explore" could branch
    assert sum(FN.state(store)["open"]["sessions"].values()) == 4


def test_close_credits_untested_branch_nodes_on_their_shrunk_screen_posterior(monkeypatch):
    nodes = [_node("x1", "request=r1 branch=1.1"), _node("x2", "request=r2 branch=1.2"),
             _node("x3", "request=r3 branch=1.3")]
    led = _led(nodes, [_run(n, s) for n in ("x1", "x2", "x3") for s in (1, 2)])
    v = SimpleNamespace(led=led, judged=[], beliefs={"x1": ID.Belief(0.004, 1e-6, 2), "x2": ID.Belief(-0.002, 1e-6, 2),
                                                      "x3": ID.Belief(0.010, 1e-6, 1)})   # x3: too few seeds
    assert FN.SCREEN_CREDIT is False                                     # v3-c7i: test gains only by default
    monkeypatch.setattr(FN, "SCREEN_CREDIT", True)                       # the screen-credit rule, still available
    closed = FN.close(_round({}), v, 500.0)
    assert closed["winner"] == "1.1"
    assert closed["credit"]["1.1"]["gain"] == pytest.approx(round(A.FRESH_SHRINK * 0.004, 6))
    assert closed["credit"]["1.3"]["gain"] == 0.0
    monkeypatch.setattr(FN, "SCREEN_CREDIT", False)                      # the old rule: test gains only
    assert FN.close(_round({}), v, 500.0)["winner"] == ""


# ---------------------------------------------------------------------------------------------- 6. lane accounting
def test_the_factorial_lane_buys_idea_sessions_on_the_explore_prior_or_not_at_all(env, monkeypatch):
    ctx, store, led, ide = env
    v = ide.view(led)
    opts = A.ledger_options(v.led, [], {}, v.incumbent, A.Costs(), v.post)
    ideas = {o.lane: o for o in opts if o.kind == "idea"}
    assert (ideas["factorial"].m, ideas["factorial"].v, ideas["factorial"].p) == \
        (ideas["explore"].m, ideas["explore"].v, ideas["explore"].p)
    monkeypatch.setattr(A, "IDEA_LANE_PRIOR", {"factorial": ""})
    lanes = {o.lane for o in A.ledger_options(v.led, [], {}, v.incumbent, A.Costs(), v.post) if o.kind == "idea"}
    assert "factorial" not in lanes and "explore" in lanes


# ---------------------------------------------------------------------------------------------- 4. data path
def _readout_ctx(ctx):
    fams = dict(ctx.cfg.families, data_curation=FamilyRule(keys=("data.*",)),
                eval_readout=FamilyRule(paths=("model.py",), combine=True, forbid=(r"dataset_dir", r"['\"]?/data\b")))
    return dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, families=fams))


def test_intake_judges_a_config_line_as_a_value_not_by_a_readout_code_rule(ctx):
    c = _readout_ctx(ctx)
    diff = ("diff --git a/model.py b/model.py\n--- a/model.py\n+++ b/model.py\n@@ -1 +1 @@\n-x = 0\n+x = 1\n"
            "diff --git a/config.yaml b/config.yaml\n--- a/config.yaml\n+++ b/config.yaml\n@@ -1 +1 @@\n"
            "-  dataset_dir: /data/old\n+  dataset_dir: /data/nanopath-databank/balanced\n")
    cd = {"data.dataset_dir": "/data/nanopath-databank/balanced"}
    assert IN.forbidden(c, ("data_curation", "eval_readout"), diff, cd) == ("", "", "")
    assert IN.forbidden(c, ("data_curation", "eval_readout"), diff, None)[0] == "eval_readout"   # no config diff
    bad = diff.replace("+x = 1", "+x = open(cfg.dataset_dir)")
    assert IN.forbidden(c, ("data_curation", "eval_readout"), bad, cd)[0] == "eval_readout"     # readout CODE


def test_a_dataset_spec_only_finish_points_the_config_at_the_dataset_as_data_curation(ctx, store, tmp_path):
    root = tmp_path / "bank"
    c = _ctx(ctx, root)
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / "config.yaml").write_text("data:\n  dataset_dir: /data/old\n  mean: [0.5]\n")
    r = RS.LLMResearcher(c, None, SimpleNamespace(config_rel="config.yaml"))
    fin = {"dataset_spec": SPEC, "hypothesis": {"family": "objective", "mechanism_tag": "m"}}
    assert r.dataset_only(store, SimpleNamespace(path=str(wt)), fin) == ""
    name = DB.validate(SPEC)[0]["name"]
    assert yaml.safe_load((wt / "config.yaml").read_text())["data"] == {"dataset_dir": str(root / name),
                                                                        "mean": [0.5]}
    assert fin["hypothesis"]["family"] == "data_curation"
    # the same content already in the bank under another name: that dataset
    bk = DB.book(store)
    bk["registered"]["have_it"] = {"path": str(root / "have_it"), "spec_sha": DB.spec_hash(DB.validate(SPEC)[0])}
    store.set_state(DB.K_BANK, bk)
    assert r.dataset_only(store, SimpleNamespace(path=str(wt)), dict(fin)) == ""
    assert yaml.safe_load((wt / "config.yaml").read_text())["data"]["dataset_dir"] == str(root / "have_it")
    assert r.dataset_only(store, SimpleNamespace(path=str(wt)), {"dataset_spec": {"tile_px": 1}, "hypothesis": {}})


def _store_node(store, nid, parent, tag="t", status=NodeStatus.QUEUED, cfg=None, diff="", family="data_curation"):
    store.put_node(Node(id=nid, task_id=store.task_id, commit=f"c-{nid}", parent_id=parent, strategy=Strategy.EXPLORE,
                        operator=Operator.DRAFT, family=family, mechanism_tag=tag, diff_hash=f"h-{nid}",
                        status=status, diff_path=diff))
    if cfg:
        store.set_state(f"config_diff:{nid}", cfg)


def test_banked_datasets_run_once_per_base_on_the_incumbent_smallest_first_one_at_a_time(ctx, store, tmp_path):
    root = tmp_path / "bank"
    c = _ctx(ctx, root)
    for name, gb in (("a", 450), ("b", 30), ("c", 20)):
        d = _ds(root, name, "ready", px=224, spec={"seed": ord(name)})
        m = json.loads((d / "manifest.json").read_text())
        (d / "manifest.json").write_text(json.dumps({**m, "bytes": gb * 1e9}))
    DB.register(c, store)
    store.ensure_root("r0")
    (d := tmp_path / "n0002.diff").write_text("diff --git a/config.yaml b/config.yaml\n+  dataset_dir: x\n")
    _store_node(store, "n0001", ROOT_NODE_ID, "inc", NodeStatus.SCREENING)
    _store_node(store, "n0002", ROOT_NODE_ID, "old_b", cfg={"data.dataset_dir": str(root / "b")}, diff=str(d))
    _store_node(store, "n0003", "n0001", "on_c", NodeStatus.SCREENING, cfg={"data.dataset_dir": str(root / "c")})
    nodes = {n: D.Node(n, None, None, "explore", "f", "explore", ROOT_NODE_ID, 1.0, "queued", "", base=b)
             for n, b in (("n0001", "B"), ("n0002", "A"), ("n0003", "B"))}
    led = SimpleNamespace(epoch="B", nodes=nodes, port=lambda n, b=None: n, runs=())
    built = []
    adapter = SimpleNamespace(config_rel="config.yaml", config_at=lambda commit: {"data": {"dataset_dir": "/x"}},
                              dry_build=lambda commit, levels, knobs, diffs, name: built.append(levels) or
                              (f"sha-{len(built)}", ""))
    acts = DB.run_unrun(c, store, adapter, led, "n0001", 100.0)          # c runs already (n0003); b < a: b first
    runs = store.get_state(DB.K_RUNS)["B"]
    assert set(runs) == {"b"} and len(acts) == 1
    assert store.get_node("n0002").status is NodeStatus.VOID and "re-rooted onto n0001" in \
        store.get_node("n0002").reject_reason and "reroot_of=n0002" in store.get_node(runs["b"]).notes
    assert DB.run_unrun(c, store, adapter, led, "n0001", 150.0) == []    # b still screening: a waits (max 1)
    led.runs = tuple(D.Run(f"r{s}", runs["b"], s, "screen", "done", 1.0, 0.0, 0.0, "", 0) for s in (1, 2, 3))
    assert len(DB.run_unrun(c, store, adapter, led, "n0001", 200.0)) == 1
    runs = store.get_state(DB.K_RUNS)["B"]
    assert set(runs) == {"a", "b"}
    for name, nid in runs.items():
        n = store.get_node(nid)
        assert n.parent_id == "n0001" and n.status is NodeStatus.QUEUED and n.family == "data_curation"
        assert store.get_state(f"config_diff:{nid}")["data.dataset_dir"] == str(root / name)
    led.runs += tuple(D.Run(f"q{s}", runs["a"], s, "screen", "done", 1.0, 0.0, 0.0, "", 0) for s in (1, 2, 3))
    assert DB.run_unrun(c, store, adapter, led, "n0001", 250.0) == []    # once per dataset per base epoch
    off = dataclasses.replace(c, cfg=dataclasses.replace(c.cfg, databank=dataclasses.replace(c.cfg.databank,
                                                                                               run_unrun=False)))
    assert DB.run_unrun(off, store, adapter, SimpleNamespace(epoch="C", nodes=nodes, port=lambda n, b=None: n),
                        "n0001", 300.0) == []


# ---------------------------------------------------------------------------------------------- 2. levers
def _lnode(nid, parent="r", base="B", status="screening", family="f", notes=""):
    return D.Node(nid, parent, None, "explore", family, "explore", "r", 1.0, status, notes, base=base)


def test_deficits_surface_an_axis_the_incumbent_lost_and_a_better_node(ctx, store):
    nodes = {n: _lnode(n, None if n == "r" else "r") for n in ("r", "inc", "o", "few")}
    v = SimpleNamespace(led=SimpleNamespace(nodes=nodes, epoch="B", base_root="r"), incumbent="inc",
                        levels={"inc": {"auc": (0.623, 10), "cls": (0.735, 10)},
                                "r": {"auc": (0.638, 6), "cls": (0.730, 6)},
                                "o": {"cls": (0.760, 5)}, "few": {"cls": (0.9, 2)}},
                        noise=SimpleNamespace(sd=lambda a: 0.004))
    rows = {r["axis"]: r for r in LV.deficits(v, store, ctx)}
    assert rows["auc"]["vs_root"] == pytest.approx(-0.015) and rows["auc"]["vs_root_shrunk"] < 0
    assert rows["auc"]["weighted"] > 0
    assert rows["cls"]["best"][0] == "o"                                 # 'few' has < deficit_min_seeds seeds
    assert rows["cls"]["weighted"] == pytest.approx(0.30 * rows["cls"]["best"][2])


def test_external_headroom_reads_the_configured_table(ctx, tmp_path):
    (p := tmp_path / "ref.csv").write_text(
        "model,group,v2_final,v2_classification,v2_segmentation\n"
        "GenBio-PathFM,baseline,0.70,0.83,0.61\nUNI-2-h,baseline,0.69,0.816,0.632\n"
        "Kaiko-S/16,baseline,0.642,0.774,0.606\nours-old,nanopath,0.65,0.95,0.60\n")
    c = _ideation_cfg(ctx, external_csv=str(p))
    lv = {a: (0.7, 5) for a in D.AXES}
    lv["cls"] = (0.735, 10)
    v = SimpleNamespace(levels={"inc": lv}, incumbent="inc")
    got = {r["axis"]: r for r in LV.external(v, c)}
    assert got["cls"]["best"] == (0.816, "UNI-2-h") and got["cls"]["peer"] == (0.774, "Kaiko-S/16")   # no GenBio
    assert got["panel"]["ours"] == pytest.approx(sum(w * lv[a][0] for a, w in D.OFFICIAL_WEIGHTS.items()))
    assert LV.external(v, ctx) == []                                     # no table configured


def test_refuted_terms_pool_random_effects_and_retest_finds_screened_out_positive_nodes(ctx, store):
    c = _ideation_cfg(ctx, refute_max_term_share=1.0)
    tags = {"k1": "koleo_cls", "k2": "koleo_cls", "k3": "weight_only_koleo", "g1": "good_idea", "s1": "rot_teacher",
            "s2": "fresh_failed", "s3": "not_confirmed_one"}
    for n, t in tags.items():
        _store_node(store, n, ROOT_NODE_ID, t)
    est = {"k1": -0.004, "k2": -0.005, "k3": -0.003, "g1": 0.001, "s1": 0.0031, "s2": 0.004, "s3": 0.005}
    status = {"s1": "screened_out", "s2": "screened_out", "s3": "not_confirmed"}
    nodes = {n: _lnode(n, "r", base="A", status=status.get(n, "screened_out")) for n in est}
    nodes["r"] = _lnode("r", None)
    nodes["inc"] = _lnode("inc", "r")

    def fit(*a):
        raise ValueError("no fit")
    post = SimpleNamespace(est={n: {M.COMPOSITE: M.Estimate(d, 1e-6, 4, 4)} for n, d in est.items()}, fit=fit, by={})
    runs = (D.Run("s2:confirm:9", "s2", 9, "confirm", "done", 1.0, 0.0, 0.0, "", 0),)
    v = SimpleNamespace(led=SimpleNamespace(nodes=nodes, epoch="B", runs=runs), judged=[], incumbent="inc", post=post)
    ref = LV.refuted(v, store, c)
    assert [r["term"] for r in ref] == ["koleo"] and ref[0]["nodes"] == 3 and ref[0]["m"] < 0   # no "weight"/"only"
    store.update_node("k2", mechanism_tag="koleo_spread")             # no whole tag, no shared exact tag: a word
    assert LV.refuted(v, store, c) == []
    store.update_node("k2", mechanism_tag="koleo_cls")
    rt = LV.retest(v, store, c)
    assert [r["node"] for r in rt][:1] == ["s1"] and {r["node"] for r in rt} == {"s1", "g1"}   # s2 fresh, s3 failed
    assert rt[0]["old"] and rt[0]["tag"] == "rot_teacher"
    assert LV.shortlist(store, c, rt)[0] == "f|rot_teacher"            # the re-test candidates first
    text = LV.text(v, store, c, retests=rt)
    assert "- koleo: 3 nodes" in text and "- [T-s1] s1 f/rot_teacher" in text      # c4i: rows carry ids


def test_unrun_lists_registered_datasets_no_node_of_this_base_trains_on(ctx, store, tmp_path):
    root = tmp_path / "bank"
    c = _ctx(ctx, root)
    for name in ("a", "b"):
        _ds(root, name, "ready", px=224, spec={"seed": ord(name)})
    DB.register(c, store)
    store.ensure_root("r0")
    _store_node(store, "n0001", ROOT_NODE_ID, cfg={"data.dataset_dir": str(root / "a")})
    v = SimpleNamespace(led=SimpleNamespace(epoch="B", nodes={"n0001": _lnode("n0001")}))
    assert [n for n, _ in LV.unrun(v, store, c)] == ["b"]
    v.led.nodes = {"n0001": _lnode("n0001", base="A")}                  # on an older base: both unrun here
    assert [n for n, _ in LV.unrun(v, store, c)] == ["a", "b"]


# ---------------------------------------------------------------------------------------------- 7. pre-check
def test_the_researcher_prechecks_the_brief_shortlist_against_the_intake_mechanism_rule(ctx, store):
    store.ensure_root("r0")
    _store_node(store, "n0001", ROOT_NODE_ID, "patch_pool_readout", NodeStatus.QUEUED, family="objective")
    req = CandidateRequest(id="req-v3-00001", task_id=store.task_id, strategy=Strategy.EXPLORE,
                           operator=Operator.DRAFT, island="main", parent_ids=(ROOT_NODE_ID,), brief="b",
                           brief_kind="explore", shortlist=("objective|patch_pool_readout", "objective|brand_new_thing"))
    r = RS.LLMResearcher(ctx, None, None)
    out = r.precheck(req, store, [store.get_node(ROOT_NODE_ID)])
    assert "## Pre-checked mechanisms" in out
    assert "objective/patch_pool_readout: REJECT near_dup_mechanism" in out and "n0001" in out
    assert "objective/brand_new_thing: OK" in out
    assert r.precheck(dataclasses.replace(req, shortlist=()), store, []) == ""


def test_brief_carries_the_shortlist_into_the_request(env):
    ctx, store, led, ide = env
    b = ide.brief_for("explore", led, None, "explore")
    assert isinstance(b.shortlist, tuple) and ide.request(b).shortlist == b.shortlist
    assert json.dumps(list(b.shortlist))                                 # family|tag strings
