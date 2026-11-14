"""v3-c5: the research-session tools -- I1 finish budget rows + budget-vs-panel reconciliation, I2 the diagnose
breakdown precomputed per incumbent, I3 mechanism (dead / near-dup) checks in check() and at note time, I4 the
structural-lever block + the measured tail in program briefs."""

from __future__ import annotations

import pytest

from loop.agents import researcher as RS
from loop.contracts import PANEL, Worktree


def _tools(tmp_path, checker=None, **kw):
    t = RS.WorktreeTools(None, Worktree(str(tmp_path), "b", "c0"), writable=("*",), locked=(),
                         checker=checker or (lambda **k: "OK"))
    t.v3 = True
    for k, x in kw.items():
        setattr(t, k, x)
    return t


def _fin(budget, panel=None, **extra):
    hy = {"claim": "c", "mechanism_tag": "m", "family": "optim", "falsification": "f",
          "mechanism_prediction": {"diagnostic": "lr_end_frac", "direction": 1, "axes": {"cls": 1}, "budget": budget}}
    if panel is not None:
        hy["predictions"] = {PANEL: {"sign": 1 if panel >= 0 else -1, "size": abs(panel)}}
    return {"hypothesis": hy, "handoff": {"actions": "a", "learnings": "l", "concerns": "c"}, **extra}


# ---------------------------------------------------------------------------------------------- I1 rows
@pytest.mark.parametrize("row", ["n2673", "classification-external-UNI-2-h-+0.0819", "guard_robustness_n2646",
                                 "seg_bestother_n2618", "D-nope"])
def test_free_text_budget_rows_are_rejected_even_when_the_brief_shows_no_row_ids(tmp_path, row):
    """10-13 live: n2728 / n2729 / n2732 cited node ids and prose as rows; with no row ids in the brief the check
    was skipped. Now a row must be a well-formed id, and the reject lists the valid ones."""
    t = _tools(tmp_path, rows=())
    msg = t.check_finish(_fin([{"axis": "cls", "row": row, "gain": 0.01}]))
    assert msg.startswith("REJECT budget") and "is not a row id" in msg and "D-cls" in msg and "D-rob" in msg


def test_well_formed_rows_pass_without_brief_rows_and_valid_ids_are_listed_with_them(tmp_path):
    assert RS.budget_problem([{"axis": "cls", "row": "[D-cls]", "gain": .01},
                              {"axis": "cls", "row": "XD-break_his", "gain": .01}], ()) == ""
    t = _tools(tmp_path, rows=("D-cls", "H-rob", "C-compute"))
    msg = t.check_finish(_fin([{"axis": "cls", "row": "the cls deficit", "gain": 0.01}]))
    assert "not a row of the brief" in msg and "D-cls, H-rob, C-compute" in msg
    assert "is not an {axis, row, gain}" in RS.budget_problem(["D-cls"], ("D-cls",))


# ---------------------------------------------------------------------------------------------- I1 reconcile
def test_a_budget_that_disagrees_with_the_panel_claim_bounces_once_then_is_accepted_and_logged(tmp_path):
    logged = []
    t = _tools(tmp_path, rows=("D-cls",), logger=lambda ev, **f: logged.append((ev, f)), request_id="req-v3-1")
    fin = _fin([{"axis": "cls", "row": "D-cls", "gain": 0.004}], panel=0.004)    # weighs to 0.0012, claims 0.004
    msg = t.check_finish(fin)
    assert msg.startswith("budget and panel claim disagree") and "0.3 x +0.0040 (cls)" in msg
    assert "+0.0012" in msg and "+0.0040" in msg and "70%" in msg and "20%" in msg
    assert t.bounces == 0                                          # its own counter: not a REJECT bounce
    assert t.check_finish(fin) == ""                               # the second try goes through ...
    assert t.claim_mismatch == {"total": 0.0012, "claim": 0.004, "rel": 0.7}
    assert logged == [("claim_mismatch", {"request_id": "req-v3-1", "total": 0.0012, "claim": 0.004, "rel": 0.7})]


def test_a_budget_within_the_tolerance_passes_and_the_tolerance_is_configurable(tmp_path):
    t = _tools(tmp_path, rows=("D-cls", "D-slide"))
    ok = _fin([{"axis": "cls", "row": "D-cls", "gain": 0.008}, {"axis": "slide", "row": "D-slide", "gain": 0.005}],
              panel=0.004)                                         # 0.0024 + 0.000875 = 0.0033: 18% off
    assert t.check_finish(ok) == "" and t.claim_mismatch is None
    tight = _tools(tmp_path, rows=("D-cls", "D-slide"), mismatch_tol=0.1)
    assert tight.check_finish(ok).startswith("budget and panel claim disagree")
    no_panel = _fin([{"axis": "cls", "row": "D-cls", "gain": 0.01}])   # no panel claim: nothing to reconcile
    assert _tools(tmp_path, rows=("D-cls",)).check_finish(no_panel) == ""


def test_the_mismatch_rides_on_the_plan(tmp_path):
    from types import SimpleNamespace
    t = _tools(tmp_path, rows=("D-cls",))
    t.claim_mismatch = {"total": 0.0012, "claim": 0.004, "rel": 0.7}
    r = RS.LLMResearcher.__new__(RS.LLMResearcher)
    r.ctx = SimpleNamespace(cfg=SimpleNamespace(ideation=None))
    r.config_rel = lambda: ""
    plan = r._plan(SimpleNamespace(), _fin([{"axis": "cls", "row": "D-cls", "gain": 0.004}], panel=0.004), t,
                   Worktree(str(tmp_path), "b", "c0"))
    assert plan["claim_mismatch"]["rel"] == 0.7 and plan["mechanism_prediction"]["size"] == pytest.approx(0.0012)


# ---------------------------------------------------------------------------------------------- I2 / I5c diagnose
import json  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from loop.agents import diagnose as DX  # noqa: E402

from .core.test_c4i import _Runner, _icfg, _led, _summary, _thunder  # noqa: E402


def test_the_breakdown_is_precomputed_per_incumbent_so_the_first_diagnose_call_has_data(ctx, store, tmp_path):
    """record(rows=None) -- ideation's every-tick call -- now computes the breakdown when the cache is of another
    incumbent (startup, a switch) or of fewer of its runs; before, the tool said 'computed yet' until a levers pass."""
    _thunder(tmp_path / "res.csv")
    c = _icfg(ctx, external_task_csv=str(tmp_path / "res.csv"), deficit_min_seeds=2)
    runs = [("inc", _summary(tmp_path / f"i{s}", break_his_linear=0.55, pcam_knn=0.85)) for s in range(2)]
    runs += [("new", _summary(tmp_path / f"n{s}", break_his_linear=0.65, pcam_knn=0.86)) for s in range(2)]
    led = _led(tmp_path, runs)
    tool = DX.Diagnoser(c, store, None)
    assert "computed yet" in tool("classification")                                   # nothing recorded yet
    DX.record(c, store, led, "inc", "c-inc", None)
    assert store.get_state(DX.K_CLS)["incumbent"] == "inc" and "break_his/linear" in tool("classification")
    ts = store.get_state(DX.K_CLS)["ts"]
    DX.record(c, store, led, "inc", "c-inc", None)
    assert store.get_state(DX.K_CLS)["ts"] == ts                                      # cached: not recomputed
    DX.record(c, store, led, "new", "c-new", None)                                    # the incumbent switches
    got = store.get_state(DX.K_CLS)
    assert got["incumbent"] == "new" and got["n_runs"] == 2
    assert next(r for r in got["rows"] if r["dataset"] == "break_his")["ours"] == pytest.approx(0.65)
    assert "incumbent new" in tool("classification")


def test_a_computed_empty_breakdown_says_so_instead_of_computing(ctx, store, tmp_path):
    led = _led(tmp_path, [("inc", _summary(tmp_path / "a"))])                         # no per-dataset scores
    DX.record(ctx, store, led, "inc", "c", None)
    assert "no per-dataset scores" in DX.Diagnoser(ctx, store, None)("classification")


def _adapter(tmp_path, runner):
    wt = tmp_path / "wt"
    (wt / "submit").mkdir(parents=True, exist_ok=True)
    (wt / "submit" / "train_1gpu.sbatch").write_text("#!/bin/bash\n#SBATCH --gpus=1\necho\n")
    return SimpleNamespace(runner=runner, config_rel="configs/main.yaml", _node_worktree=lambda n, c: wt,
                           _sbatch=lambda cfg, out, name, pool="": (["x"], {"SBATCH_ACCOUNT": pool}))


def _inc_run(tmp_path, name):
    run = tmp_path / f"run-{name}"
    run.mkdir()
    (run / "latest.pt").write_text("x")
    return SimpleNamespace(runs=(SimpleNamespace(node=name, status="done", output_dir=str(run)),), nodes={},
                           epoch="B")


def test_the_feature_job_is_precomputed_once_per_incumbent_without_waiting(ctx, store, tmp_path):
    runner = _Runner()
    adapter = _adapter(tmp_path, runner)
    sb = lambda: [c for c in runner.calls if c[0][0] == "sbatch"]                     # noqa: E731
    assert DX.precompute_features(ctx, store, adapter) == ""                          # no incumbent yet
    DX.record(ctx, store, _inc_run(tmp_path, "inc"), "inc", "c-inc", None)
    assert "job 77" in DX.precompute_features(ctx, store, adapter) and len(sb()) == 1
    assert DX.precompute_features(ctx, store, adapter) == "" and len(sb()) == 1       # once per incumbent
    assert "still pending" in DX.features(ctx, store, adapter, wait_s=0, sleep=lambda s: None)
    assert len(sb()) == 1                                                             # the session reuses the job
    DX.record(ctx, store, _inc_run(tmp_path, "nxt"), "nxt", "c-nxt", None)            # a switch: submitted again
    assert DX.precompute_features(ctx, store, adapter) and len(sb()) == 2
    assert DX.precompute_features(_icfg(ctx, diagnose_precompute=False), store, adapter) == ""
    assert DX.precompute_features(ctx, store, SimpleNamespace()) == ""                # not a cluster adapter


def test_no_checkpoint_yet_is_retried_next_tick_and_cached_features_read_back(ctx, store, tmp_path):
    runner = _Runner()
    adapter = _adapter(tmp_path, runner)
    led = SimpleNamespace(runs=(SimpleNamespace(node="inc", status="done", output_dir=str(tmp_path / "none")),),
                          nodes={}, epoch="B")
    DX.record(ctx, store, led, "inc", "c-inc", None)
    assert "not precomputed" in DX.precompute_features(ctx, store, adapter)
    pre = store.get_state(DX.K_PRE)                                                   # v3-c5b S8: retried, backed off
    assert pre["ok"] is False and pre["fails"] == 1 and pre["next"] > pre["ts"]
    assert DX.precompute_features(ctx, store, adapter) == ""                          # inside the backoff
    assert DX.cached_features(ctx, store) is None
    res = Path(ctx.state_dir) / "diagnose" / "inc" / "features.json"
    res.parent.mkdir(parents=True)
    res.write_text(json.dumps({"n_tiles": 4, "datasets": {"wilds": {"tiles": 4, "gates": [{"mean": 0.06}]},
                                                           "tcga": {"tiles": 4, "gates": [{"mean": 0.56}]}}}))
    assert DX.precompute_features(ctx, store, adapter).endswith("cached") and not runner.calls
    got = DX.cached_features(ctx, store)
    assert got["wilds"]["gates"][0]["mean"] == 0.06 and set(got) == {"wilds", "tcga"}
    assert DX.cached_features(ctx, store, "other") is None
    res.write_text(json.dumps({"error": "oom"}))
    assert DX.cached_features(ctx, store, "inc") is None


def test_precompute_marks_done_only_on_success_and_runs_off_the_tick(ctx, store, tmp_path):
    """v3-c5b S8: a failed sbatch is no 'done' -- retried after a doubling backoff; with an executor the worktree +
    sbatch run off the tick thread and the next call harvests the outcome."""
    from concurrent.futures import ThreadPoolExecutor
    runner = _Runner()
    adapter = _adapter(tmp_path, runner)
    DX.record(ctx, store, _inc_run(tmp_path, "inc"), "inc", "c-inc", None)
    real = runner.run
    runner.run = lambda argv, **k: (1, "", "sbatch: error: down") if argv[0] == "sbatch" else real(argv, **k)
    t = [1000.0]
    got = DX.precompute_features(ctx, store, adapter, clock=lambda: t[0])
    assert "not precomputed" in got and store.get_state(DX.K_PRE)["next"] == 1000.0 + DX.PRE_BACKOFF_S[0]
    t[0] += DX.PRE_BACKOFF_S[0] + 1
    DX.precompute_features(ctx, store, adapter, clock=lambda: t[0])                  # fails again: doubled
    assert store.get_state(DX.K_PRE)["fails"] == 2
    assert store.get_state(DX.K_PRE)["next"] == t[0] + 2 * DX.PRE_BACKOFF_S[0]
    runner.run = real
    t[0] += 3 * DX.PRE_BACKOFF_S[0]
    with ThreadPoolExecutor(1) as ex:
        assert "off the tick" in DX.precompute_features(ctx, store, adapter, ex, clock=lambda: t[0])
        DX._PRE_JOBS[id(store)]["fut"].result(timeout=30)
        assert "job 77" in DX.precompute_features(ctx, store, adapter, ex, clock=lambda: t[0])
    assert store.get_state(DX.K_PRE)["ok"] is True
    assert DX.precompute_features(ctx, store, adapter, clock=lambda: t[0]) == ""


def test_ideation_ideate_precomputes_for_the_incumbent(monkeypatch):
    """The hook: ideation.ideate's every-tick DX.record is followed by precompute_features with the researcher's
    adapter (loop/core/ideation.py, the 'what the researcher's diagnose tool reads' block)."""
    import inspect
    from loop.core import ideation as ID
    src = inspect.getsource(ID.Ideation.ideate)
    assert "DX.precompute_features(" in src and src.index("DX.record(") < src.index("DX.precompute_features(")


# ---------------------------------------------------------------------------------------------- I3 early mechanism checks
import dataclasses  # noqa: E402

from loop.contracts import (CandidateRequest, Node, NodeStatus, Operator, Postmortem, ROOT_NODE_ID,  # noqa: E402
                            Strategy, LeverTags)
from loop.agents import notes as NT  # noqa: E402

from .test_novelty import _fresh, _on, _plant_mech  # noqa: E402

PCGRAD = "PCGrad gradient surgery between the DINO and JEPA gradients on the late blocks"


def _dead_pcgrad(ctx, store):
    c = _on(ctx)
    _plant_mech(store, "n0087", "pcgrad_conflict_resolution",
                "PCGrad gradient surgery between conflicting DINO and JEPA gradients", status=NodeStatus.VETOED)
    store.put_postmortem(Postmortem("pm-n0087", store.task_id, "n0087", "loss", "projection starved the JEPA side",
                                    "m", LeverTags()))
    for i in range(6):
        _plant_mech(store, f"n02{i:02d}", f"other_{i}", f"an unrelated jepa panel change number {i} on tiles", ts=1e7)
    _fresh(c, store, "n0087", -0.001)
    return c


def _session(c, store, tmp_path, diff_text=""):
    adapter = SimpleNamespace(diff=lambda wt: (diff_text, ["train.py"] if diff_text else [], {}))
    res = RS.LLMResearcher(c, None, adapter)
    req = CandidateRequest(id="req-v3-00009", task_id=store.task_id, strategy=Strategy.EXPLORE,
                           operator=Operator.DRAFT, island="i", parent_ids=(ROOT_NODE_ID,), brief="")
    wt = SimpleNamespace(path=str(tmp_path), branch="b", base_commit="c0")
    tools = RS.WorktreeTools(None, wt, writable=(), locked=(), checker=lambda **kw: res.check(req, store, wt, **kw))
    return res, req, wt, tools


def test_a_noted_paper_on_a_dead_mechanism_gets_a_heads_up_at_note_time_without_claiming_it(ctx, store, tmp_path):
    from loop.core.ideation import K_CLAIMS
    c = _dead_pcgrad(ctx, store)
    c = dataclasses.replace(c, cfg=dataclasses.replace(c.cfg, researcher=dataclasses.replace(
        c.cfg.researcher, notes_path=str(tmp_path / "notes" / "notes.jsonl"))))
    _, _, _, tools = _session(c, store, tmp_path)
    tools.research = NT.ResearchStage(c, "req-v3-00009")
    url = "https://arxiv.org/abs/2001.06782"
    tools.research.on_web("WebFetch", {"url": url})
    args = {"url": url, "title": "Gradient Surgery for Multi-Task Learning (PCGrad)", "topic": "pcgrad_late_block_surgery",
            "claim": PCGRAD + " removes conflicting components", "mechanism": "project each task gradient off the other",
            "setting": "multi-task supervised benchmarks", "effect": "+2 to +5 points on MT10 success rate",
            "transfer": "DINO vs JEPA objectives conflict in our late blocks"}
    out = tools("note", args)
    assert out.startswith("noted rn-") and "HEADS-UP" in out and "dead_mechanism" in out and "n0087" in out
    assert not (store.get_state(K_CLAIMS) or {})                    # advisory only: nothing claimed
    ok = tools("note", {**args, "url": url, "topic": "stain_lut", "title": "A stain LUT",
                        "claim": "a learned colour lookup table for H&E stain normalisation of tiles",
                        "mechanism": "a 3D LUT fitted per slide"})
    assert ok.startswith("noted rn-") and "HEADS-UP" not in ok


def test_check_before_any_edit_runs_the_embedding_near_dup_intake_would_reject(ctx, store, tmp_path):
    """n2728 (10-13) was rejected near_dup only at intake; check() without a diff ran the mechanism rules only.
    v3-c6i: a v3 session's near-dup compares MECHANISM signatures (family + tag + one line, loop.core.mechanisms)."""
    from loop.contracts import HypothesisKind, HypothesisRecord
    from loop.select.embed import HashingEmbedder
    c = _on(ctx)
    claim = "anneal the KDE bandwidth on a cosine schedule over the truncated run to sharpen late features"
    res, req, wt, tools = _session(c, store, tmp_path)
    vec = HashingEmbedder().embed(claim + "\n")
    store.put_hypothesis(HypothesisRecord(id="h-n2700", task_id=store.task_id, kind=HypothesisKind.TRIAL, claim=claim,
                                          mechanism_tag="kde_bandwidth_anneal", family="objective", predictions={},
                                          falsification="f"))
    store.put_node(Node(id="n2700", task_id=store.task_id, commit="c-n2700", parent_id=ROOT_NODE_ID,
                        strategy=Strategy.EXPLORE, operator=Operator.DRAFT, family="objective",
                        mechanism_tag="kde_bandwidth_anneal", diff_hash="h-n2700", status=NodeStatus.SCREENING,
                        embedding=vec, hypothesis_id="h-n2700", created_ts=1e7))
    got = tools.t_check(family="objective", claim=claim, mechanism_tag="kde_bandwidth_anneal")
    assert got.startswith("REJECT near_dup") and "n2700" in got
    ok = tools.t_check(family="objective", claim="drop path on the last two blocks only", mechanism_tag="late_droppath")
    assert ok.startswith("OK (no diff yet)")


def test_check_takes_the_falsification_so_naming_the_dead_nodes_passes_as_at_finish(ctx, store, tmp_path):
    c = _dead_pcgrad(ctx, store)
    _, _, _, tools = _session(c, store, tmp_path)
    assert tools.t_check(family="optimization", claim=PCGRAD, mechanism_tag="pcgrad_late").startswith(
        "REJECT dead_mechanism")
    ok = tools.t_check(family="optimization", claim=PCGRAD, mechanism_tag="pcgrad_late",
                       falsification="unlike n0087 only the JEPA side is projected; fails if late-block cos stays < 0")
    assert ok.startswith("OK (no diff yet)")
    assert "falsification" in next(t for t in RS.TOOL_DEFS if t["name"] == "check")["input_schema"]["properties"]


# ---------------------------------------------------------------------------------------------- I4 structural levers
from loop.agents import briefs as BR  # noqa: E402
from loop.core import data as D  # noqa: E402
from loop.core import ideation as ID  # noqa: E402
from loop.select import intake as IN  # noqa: E402

from .core.test_fnf import env  # noqa: E402,F401  (the fixture)


def _node(nid, strategy, base="B", notes="", port_of=""):
    return D.Node(nid, ROOT_NODE_ID, None, strategy, "f", "explore", ROOT_NODE_ID, 1.0, "screening", notes, base=base,
                  port_of=port_of)


def test_the_measured_tail_is_computed_from_this_epochs_screened_nodes(ctx):
    nodes = {"a": _node("a", "explore"), "b": _node("b", "exploit"), "c": _node("c", "explore"),
             "old": _node("old", "explore", base="A"), "w": _node("w", "explore", notes="for_wave=w1"),
             "p": _node("p", "explore", port_of="x"), "m": _node("m", "moonshot"), "u": _node("u", "explore"),
             "x": _node("x", "crossover")}
    beliefs = {"a": ID.Belief(0.005, 1e-6, 2), "b": ID.Belief(0.001, 1e-6, 3), "c": ID.Belief(-0.002, 1e-6, 1),
               "old": ID.Belief(0.02, 1e-6, 4), "w": ID.Belief(0.02, 1e-6, 4), "p": ID.Belief(0.02, 1e-6, 4),
               "m": ID.Belief(0.009, 1e-6, 2), "x": ID.Belief(0.02, 1e-6, 2)}
    v = SimpleNamespace(led=SimpleNamespace(nodes=nodes, epoch="B", base_root="root"), beliefs=beliefs)
    t = BR.measured_tail(v, ctx)
    assert t["inc"] == (1, 3, 0.005) and t["struct"] == (1, 1, 0.009)   # old epoch / wave / port / crossover / unscreened out
    line = BR.tail_line(t)
    assert "1/3 = 33.3%" in line and ">= +0.004" in line and "1/1 reached >= +0.008" in line
    t2 = BR.measured_tail(v, _icfg(ctx, tail_gain=0.0005, tail_min_seeds=2))
    assert t2["inc"] == (2, 2, 0.005)                                     # thresholds are config keys
    empty = BR.tail_line(BR.measured_tail(SimpleNamespace(led=SimpleNamespace(nodes={}, epoch="B", base_root="r"),
                                                          beliefs={}), ctx))
    assert "no incremental node" in empty and "no structural node" in empty


def test_structural_rows_compute_per_tile_headroom_and_per_dataset_panel_equivalents(ctx, store, tmp_path):
    _thunder(tmp_path / "res.csv")
    c = _icfg(ctx, external_task_csv=str(tmp_path / "res.csv"), deficit_min_seeds=2)
    led = _led(tmp_path, [("inc", _summary(tmp_path / f"i{s}", break_his_linear=0.55, pcam_knn=0.85))
                          for s in range(2)])
    store.set_state(IN.K_BASE, {"train.global_views": 2, "train.global_size": 224, "train.lr": 1e-3})
    v = SimpleNamespace(led=led, incumbent="inc", beliefs={})
    rows = BR.structural_rows(v, store, c)
    comp = next(r for r in rows if r.startswith("- [C-compute] per-tile"))
    assert "x3.33 more compute PER TILE" in comp
    assert any("train.global_size=224" in r for r in rows)
    bh = next(r for r in rows if r.startswith("- [XD-break_his]"))
    assert "+0.2370" in bh and f"{0.30 * 0.237 / 2:+.4f}" in bh and "alone >= the structural bar" in bh


def test_program_briefs_carry_the_structural_block_and_other_kinds_do_not(env, monkeypatch):
    ctx, store, led, ide = env
    monkeypatch.setattr(ID, "target_gap", lambda v: 0.006)
    v = ide.view(led)
    prog = ide.brief_for("explore", led, v, "program").text
    assert "## Structural levers (aim for >= +0.008" in prog and "Measured tail (computed from the ledger)" in prog
    assert "## Structural levers" not in ide.brief_for("explore", led, v, "explore").text
    br = ide.brief_for("explore", led, v, "program", branch="1.1").text
    assert "## Structural levers" in br
    t = BR.measured_tail(v, ctx)                                          # the block states exactly the computed tail
    assert BR.tail_line(t) in prog
