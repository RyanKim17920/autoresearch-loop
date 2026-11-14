"""loop.core.fnf — fork-and-flush: branch tagging, per-branch GPU-h caps, the flushed context, selection on
fresh-seed test credit (a "no branch won" round included), the round triggers, the arm-vs-control A/B and its
adoption, and the test-entry p_min TUNABLE."""

from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace

import pytest

from loop import spec as SP
from loop.agents import notes as NT
from loop.agents import researcher as RS
from loop.contracts import (CandidateRequest, HandoffNote, LeverTags, Operator, Postmortem, ResearchResult,
                            ROOT_NODE_ID, Strategy)
from loop.core import acquire as A
from loop.core import data as D
from loop.core import fnf as FN
from loop.core import ideation as ID
from loop.core import meta as MT
from loop.store import SqliteStore

from ..conftest import InlineExecutor
from ..test_tick import _ledger
from .conftest import TASK
from .test_ideation import README, _cand


@pytest.fixture
def env(ctx, tmp_path):
    db = _ledger(tmp_path / "ledger.db")
    ctx = dataclasses.replace(ctx, task_id=TASK)
    store = SqliteStore(db, TASK)
    (readme := tmp_path / "README.md").write_text(README)
    yield ctx, store, D.load(db, TASK), ID.Ideation(ctx, store, readme=readme)
    store.close()


class _Research:
    def __init__(self, free=4):
        self.n, self.started, self.logs, self.researcher = free, [], [], SimpleNamespace(llm=None)

    def free(self):
        return self.n - len(self.started)

    def start(self, req, *args, **log):
        self.started.append(req)
        self.logs.append(log)


# ---------------------------------------------------------------------------------------------- toy ledger / view
def _node(nid, notes="", lane="explore", ts=100.0, family="f"):
    return D.Node(nid, ROOT_NODE_ID, None, "explore", family, lane, ROOT_NODE_ID, ts, "screening", notes)


def _run(nid, seed, gpu=1.0, stage="screen", status="done"):
    return D.Run(f"{nid}:{stage}:{seed}", nid, seed, stage, status, gpu, 0.0, 0.0, "", 0)


def _led(nodes, runs):
    return SimpleNamespace(nodes={n.id: n for n in nodes}, runs=tuple(runs), obs=(), epoch="")


def _view(led, gains):
    judged = [SimpleNamespace(node=n, verdict=SimpleNamespace(mean=g)) for n, g in gains.items()]
    return SimpleNamespace(led=led, judged=judged)


def _round(st, ts=50.0, gpu0=0.0):
    st["open"] = {"id": 1, "ts": ts, "gpu0": gpu0, "cap": 21.0, "branches": ["1.1", "1.2", "1.3", "1.4"],
                  "incumbent": "inc", "trigger": "start", "sessions": {}}
    st["round"] = 1
    return st


# ---------------------------------------------------------------------------------------------- tagging + flush
def test_first_ideate_forks_a_round_and_explore_sessions_split_between_branches_and_control(env):
    ctx, store, led, ide = env
    object.__setattr__(ctx.cfg.ideation, "kind_weight", False)        # the fixed rotation (A3 kind weights off)
    tick = SimpleNamespace(research=_Research(4), pool=InlineExecutor(), dry=False)
    acts = ide.ideate(tick, [SimpleNamespace(kind="idea", lane="explore")] * 4, led, {})
    st = store.get_state(FN.K_FNF)
    assert st["open"]["branches"] == ["1.1", "1.2", "1.3", "1.4"] and st["open"]["trigger"] == "start"
    assert any("fnf round 1 (start)" in a for a in acts) and store.events("fnf_round")
    reqs = tick.research.started
    br = [r for r in reqs if r.branch]
    assert br and len(br) < len(reqs)                          # the arm's share: some sessions stay shared
    for r in br:
        assert r.ablated == FN.FLUSH and r.parent_ids == (ide.view(led).incumbent,) and RS.is_v3(r)
        assert "## Protected incumbent" in r.brief and f"## Your branch {r.branch} so far" in r.brief
        for hidden in ("## Engagement of recent ideas", "## Computed facts", "## Literature notes (top 5"):
            assert hidden not in r.brief
        # c4i: the computed headroom / levers are measurements, not the shared story: a branch keeps them
        assert "## Per-axis headroom" in r.brief and "## Untested levers & deficits" in r.brief
    for r in reqs:
        if not r.branch:
            assert r.ablated == () and "## Per-axis headroom" in r.brief
    assert {lg.get("branch") for lg in tick.research.logs} - {None} == {r.branch for r in br}
    assert sum(st["open"]["sessions"].values()) == 4
    # the branch tag rides on the node made from a branch session
    out = ide.intake(br[0], ResearchResult("ok", _cand("a flushed idea about the patch head", commit="cB")))
    assert f"branch={br[0].branch}" in store.get_node(out.split()[1]).notes


def test_only_explore_lane_sessions_branch_and_never_a_plateau_brief(env, monkeypatch):
    ctx, store, led, ide = env
    object.__setattr__(ctx.cfg.ideation, "kind_weight", False)        # the fixed rotation (A3 kind weights off)
    monkeypatch.setattr(ID, "plateau", lambda led, judged: ID.Plateau(True, 1.0, 50.0, 10.0, 2))
    st = _round(FN.state(store), gpu0=FN.total_gpu(led))
    assert FN.assign(dict(st), led, "exploit") == "" and FN.assign(dict(st), led, "factorial") == ""
    store.set_state(FN.K_FNF, st)
    store.set_state(ID.K_PLATEAU, {"active": True, "since": 1.0, "ts": 2.0, "served": 0, "diagnosis": "d",
                                   "hypotheses": [{"claim": "c", "evidence": "e", "decisive_experiment": "x",
                                                   "prediction": {}}]})
    tick = SimpleNamespace(research=_Research(1), pool=InlineExecutor(), dry=False)
    ide.ideate(tick, [SimpleNamespace(kind="idea", lane="explore")], led, {})
    r = tick.research.started[0]
    assert r.brief_kind == "kuhn" and not r.branch                      # plateau hypotheses go first (kuhn), shared


def test_branch_brief_sees_only_its_own_notes_and_handoffs(env, tmp_path):
    ctx, store, led, ide = env
    v = ide.view(led)
    for branch, title in (("", "shared basin paper"), ("2.1", "branch two one paper"), ("2.2", "other branch")):
        p = NT.notes_path(ctx, branch)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"id": f"rn-{abs(hash(title)) % 10**8:08d}", "title": title, "claim": "c",
                                 "transfer": "t"}) + "\n")
    assert NT.notes_path(ctx, "2.1").parent.name == "branch-2.1"
    b = ide.brief_for("explore", led, v, branch="2.1")
    assert b.branch == "2.1" and b.ablated == FN.FLUSH and "branch two one paper" in b.text
    assert "shared basin paper" not in b.text and "other branch" not in b.text
    shared = ide.brief_for("explore", led, v)
    assert "shared basin paper" in shared.text and "branch two one paper" not in shared.text
    assert ide.brief_for("exploit", led, v, branch="2.1").kind == "explore"     # a branch is explore / literature


def test_researcher_prompt_flush_drops_postmortems_and_the_parents_story_not_its_diff(ctx, store, tmp_path):
    store.ensure_root("c0")
    store.put_handoff(HandoffNote("ho-r", store.task_id, ROOT_NODE_ID, "did A-thing", "learned B-thing", "worry C"))
    store.put_postmortem(Postmortem("pm-1", store.task_id, ROOT_NODE_ID, "loss", "the D-mechanism lost", "m",
                                    LeverTags({})))
    res = RS.LLMResearcher(ctx, None, None)
    base = CandidateRequest(id="req-v3-00001", task_id=ctx.task_id, strategy=Strategy.EXPLORE, operator=Operator.DRAFT,
                            island="main", parent_ids=(ROOT_NODE_ID,), brief="b", brief_kind="explore")
    shared = res._prompt(base, store, None, [store.get_node(ROOT_NODE_ID)], "c0", 0, "")
    assert "learned B-thing" in shared and "the D-mechanism lost" in shared
    flushed = res._prompt(dataclasses.replace(base, ablated=FN.FLUSH, branch="1.1"), store, None,
                          [store.get_node(ROOT_NODE_ID)], "c0", 0, "")
    assert "learned B-thing" not in flushed and "the D-mechanism lost" not in flushed
    assert "## Surface" in flushed                                         # surface / families / locked rules stay
    stage = NT.ResearchStage(ctx, "req-v3-00001", "1.1")
    assert stage.path == NT.notes_path(ctx, "1.1") != NT.notes_path(ctx)


# ---------------------------------------------------------------------------------------------- caps
def test_branch_caps_bound_non_test_actions_in_rank_and_idle_fill():
    tags, left = {"a": "1.1", "b": "1.2"}, {"1.1": 2.5, "1.2": 0.5}
    acts = [A.Action("screen", "a", "explore", 3.0, 1.0, 3), A.Action("seed", "b", "explore", 1.0, 1.0, 1),
            A.Action("test", "b", "explore", 10.0, 1.0, 10)]
    lft = dict(left)
    assert [A.tag_ok(a, tags, lft) for a in acts] == [False, False, True]   # a test is never capped
    lft = {"1.1": 3.5}
    assert A.tag_ok(acts[0], tags, lft) and lft["1.1"] == pytest.approx(0.5)
    opts = [A.Option("screen", "a", "explore", 0.004, 1e-5, 1e-5, 3.0, 3, tail=True),
            A.Option("screen", "c", "explore", 0.001, 1e-5, 1e-5, 3.0, 3, tail=True)]
    got = A.rank(opts, 10, 0, tags=tags, tag_left=left)
    assert [a.node for a in got] == ["c"]                                  # the capped branch node is skipped
    assert [a.node for a in A.rank(opts, 10, 0)] == ["a", "c"]


def test_budgets_are_the_cap_minus_the_branch_nodes_non_test_gpu_h():
    nodes = [_node("x1", "request=r1 branch=1.1"), _node("x2", "request=r2 branch=1.2"), _node("c1", "request=r3")]
    runs = [_run("x1", 1, 4.0), _run("x1", 2, 3.0), _run("x1", 3, 9.0, stage="confirm"), _run("x2", 1, 1.0),
            _run("x2", 2, 5.0, status="running")]
    led = _led(nodes, runs)
    tags, left = FN.budgets(_round({}), led)
    assert tags == {"x1": "1.1", "x2": "1.2"}
    run_h = FN.run_hours(led)
    assert left["1.1"] == pytest.approx(21.0 - 7.0)                         # the test run is not counted
    assert left["1.2"] == pytest.approx(21.0 - 1.0 - run_h) and left["1.3"] == 21.0
    assert FN.budgets({"open": None}, led) == ({}, {})


def test_assign_keeps_the_arm_near_its_share_and_skips_capped_branches():
    led = _led([_node("c1", "request=r1", ts=60.0)], [_run("c1", s, 1.0) for s in range(1, 4)])
    st = _round({})
    got = [FN.assign(st, led, "explore") for _ in range(10)]
    w_arm = sum(st["open"]["sessions"].get(b, 0) for b in st["open"]["branches"])
    assert 2 <= w_arm <= 4 and st["open"]["sessions"]["control"] == 10 - w_arm
    assert len({g for g in got if g}) == w_arm                               # round-robin over the branches
    st["open"]["cap"] = 0.0                                                  # every branch at its cap: control
    assert FN.assign(st, led, "explore") == ""
    old = FN.SHARE
    try:
        FN.SHARE = 0.0
        assert FN.assign(_round({}), led, "explore") == ""
    finally:
        FN.SHARE = old


# ---------------------------------------------------------------------------------------------- selection
def test_selection_is_on_fresh_seed_test_credit_and_the_winner_merges(ctx, store):
    nodes = [_node("x1", "request=r1 branch=1.1"), _node("x2", "request=r2 branch=1.2"),
             _node("x3", "request=r3 branch=1.3"), _node("c1", "request=r4", ts=60.0)]
    runs = [_run(n, s, 1.0) for n in ("x1", "x2", "x3", "c1") for s in range(1, 4)]
    led = _led(nodes, runs)
    # x3 has the best screening belief but no test; x1 / x2 have fresh-seed tests; x2 earns more per GPU-h
    v = _view(led, {"x1": 0.002, "x2": 0.003, "c1": 0.001})
    for b in ("1.1", "1.2"):
        p = NT.notes_path(ctx, b)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"id": f"rn-0000000{b[-1]}", "title": f"paper of {b}", "claim": "c"}) + "\n")
    store.put_handoff(HandoffNote("ho-x2", store.task_id, "x2", "did x2", "learned x2", "worry x2"))
    st = _round({})
    closed = FN.close(st, v, 500.0)
    assert closed["winner"] == "1.2" and st["open"] is None and st["rounds"][-1] is closed
    assert closed["credit"]["1.3"]["gain"] == 0.0 and closed["control"] == ["c1"]
    acts = FN.merge(ctx, store, closed)
    shared = NT.load(NT.notes_path(ctx))
    assert {n["title"] for n in shared} >= {"paper of 1.2"} and "paper of 1.1" not in {n["title"] for n in shared}
    assert any(n.get("claim") == "learned x2" for n in shared) and "branch 1.2 won" in acts[0]
    assert not NT.notes_path(ctx, "1.1").parent.exists()                      # archived
    assert (NT.notes_path(ctx).parent / "archive" / "branch-1.1" / "notes.jsonl").exists()


def test_no_branch_won_is_a_valid_outcome_and_nothing_is_merged(ctx, store):
    nodes = [_node("x1", "request=r1 branch=1.1"), _node("x2", "request=r2 branch=1.2")]
    led = _led(nodes, [_run("x1", 1), _run("x2", 1)])
    v = _view(led, {"x1": -0.004})                          # a failed test is no credit
    st = _round({})
    closed = FN.close(st, v, 500.0)
    assert closed["winner"] == ""
    assert "no branch won" in FN.merge(ctx, store, closed)[0] and NT.load(NT.notes_path(ctx)) == []


def test_fork_round_closes_the_open_round_then_forks_the_next(env):
    ctx, store, led, ide = env
    v = ide.view(led)
    store.set_state(FN.K_FNF, _round(FN.state(store)))
    acts = ide.fork_round(v, "cadence")
    st = store.get_state(FN.K_FNF)
    assert st["round"] == 2 and st["open"]["branches"][0] == "2.1" and len(st["rounds"]) == 1
    assert any("no branch won" in a for a in acts) and store.events("fnf_round_closed")
    old = FN.SHARE
    try:
        FN.SHARE = 0.0                                       # arm off: the open round closes, none opens
        ide.fork_round(v, "cadence")
        assert store.get_state(FN.K_FNF)["open"] is None
    finally:
        FN.SHARE = old


# ---------------------------------------------------------------------------------------------- triggers
def test_triggers_start_cadence_plateau_and_diversity(monkeypatch):
    led = _led([_node("c1", "request=r1")], [_run("c1", s, 100.0) for s in range(1, 3)])   # 200 GPU-h
    monkeypatch.setattr(ID, "plateau", lambda led, judged: ID.Plateau(False, 0.0, 0.0, 1.0, 0))
    assert FN.trigger(led, [], {"round": 0, "open": None, "rounds": []}, None) == "start"
    st = _round({}, ts=10.0, gpu0=0.0)
    assert FN.trigger(led, [], st, None) is None                             # 200 < 350, nothing else
    st["open"]["gpu0"] = -200.0
    assert FN.trigger(led, [], st, None) == "cadence"                        # 400 >= 350
    st["open"]["gpu0"] = 0.0
    div = {"ts": 20.0, "n": 10, "median": 0.7}
    assert FN.trigger(led, [], st, div).startswith("diversity")               # 200 >= .5 x 350
    assert FN.trigger(led, [], st, {**div, "ts": 5.0}) is None               # measured before the round
    assert FN.trigger(led, [], st, {**div, "median": 0.5}) is None
    monkeypatch.setattr(ID, "plateau", lambda led, judged: ID.Plateau(True, 77.0, 500.0, 100.0, 2))
    assert FN.trigger(led, [], st, None) == "plateau"
    assert FN.trigger(led, [], {**st, "last_plateau": 77.0}, None) is None   # once per plateau episode
    st["open"]["gpu0"] = 100.0                                               # 100 < .5 x 350: too soon
    assert FN.trigger(led, [], st, div) is None


def test_nn_similarity_is_the_median_nearest_earlier_idea_per_arm(store):
    from loop.contracts import Node, NodeStatus
    from loop.select.embed import HashingEmbedder
    emb = HashingEmbedder()
    texts = {"n1": ("a teacher momentum schedule for the ema", 0.0, ""),
             "n2": ("a teacher momentum schedule for the ema twice", 100 * 3600.0, ""),
             "n3": ("stain jitter in the hed colour space", 101 * 3600.0, "branch=1.1")}
    nodes = []
    for nid, (t, ts, tag) in texts.items():
        store.put_node(Node(id=nid, task_id=store.task_id, commit="c", parent_id=ROOT_NODE_ID,
                            strategy=Strategy.EXPLORE, operator=Operator.DRAFT, family="f", mechanism_tag=nid,
                            diff_hash=nid, status=NodeStatus.QUEUED, embedding=emb.embed(t)))
        nodes.append(_node(nid, f"request=r-{nid} {tag}", ts=ts))
    ID._EMB.clear()
    d = FN.nn_similarity(store, _led(nodes, []), now=102 * 3600.0)
    assert d["n"] == 2 and d["by_arm"]["shared"]["median"] > 0.8 > d["by_arm"]["flush"]["median"]


# ---------------------------------------------------------------------------------------------- A/B + adoption
def _ab_world(rounds=2, arm_gain=0.004, ctl_gain=0.0005):
    nodes, runs, gains, closed = [], [], {}, []
    for r in range(1, rounds + 1):
        arm = {f"{r}.{k}": [] for k in range(1, 5)}
        ctl = []
        for i in range(8):
            a, c = f"a{r}{i}", f"c{r}{i}"
            nodes += [_node(a, f"request=q{a} branch={r}.{1 + i % 4}", family=f"fa{i}"),
                      _node(c, f"request=q{c}", family="fc")]
            runs += [_run(a, s, 1.0) for s in range(3)] + [_run(c, s, 1.0) for s in range(3)]
            gains[a] = arm_gain * (i % 2)
            gains[c] = ctl_gain * (i % 2)
            arm[f"{r}.{1 + i % 4}"].append(a)
            ctl.append(c)
        closed.append({"id": r, "ts": 0.0, "closed": 86400.0, "winner": "", "nodes": arm, "control": ctl})
    return _view(_led(nodes, runs), gains), {"rounds": closed, "open": None}


def test_ab_adopts_only_after_two_rounds_and_two_bootstrap_se():
    v, st = _ab_world(rounds=1)
    assert FN.ab(v, st)["verdict"] == "insufficient"
    v, st = _ab_world(rounds=2)
    a = FN.ab(v, st)
    assert a["verdict"] == "adopt" and a["diff"] > 2 * a["se"] and a["nodes"] == {"flush": 16, "shared": 16}
    assert a["families_per_day"]["flush"] > a["families_per_day"]["shared"]
    v, st = _ab_world(rounds=3, arm_gain=0.001, ctl_gain=0.001)
    assert FN.ab(v, st)["verdict"] == "keep testing"


def test_meta_adopts_the_arm_as_a_config_version_and_installs_it(ctx, store):
    meta = MT.Meta(ctx, store, locks=None)
    v, st = _ab_world(rounds=2)
    rep = MT.MetaReport(ts=1.0, since=0.0, fnf={"diversity": {"ts": 1.0, "n": 7, "median": 0.6},
                                                "ab": FN.ab(v, st)})
    msg = meta.judge_fnf(rep)
    assert "ADOPTED" in msg and store.get_state(FN.K_DIV)["median"] == 0.6
    assert meta.versions.values()["ideation.fnf_share"] == FN.ADOPT_SHARE and store.events("meta_fnf")
    assert "ADOPTED" not in meta.judge_fnf(rep)                                # once
    old = FN.SHARE
    try:
        MT.apply(meta.versions.values(), None)
        assert FN.SHARE == FN.ADOPT_SHARE
    finally:
        FN.SHARE = old
    assert "Fork-and-flush" in MT.render_markdown(rep)


# ---------------------------------------------------------------------------------------------- test_p_min
def test_test_p_min_is_a_bounded_tunable_outside_the_locked_test_prefix(monkeypatch):
    t = SP.TUNABLE["acquire.test_p_min"]
    assert (t.default, t.lo, t.hi) == (0.7, 0.6, 0.9) and SP.check_tunable("acquire.test_p_min", 0.8) == ""
    assert "outside" in SP.check_tunable("acquire.test_p_min", 0.95)
    assert SP.check_tunable("test.p_min", 0.8).endswith("LOCKED")
    seen = {}
    from loop.core import status as ST
    monkeypatch.setattr(ST, "test_entry_ok", lambda *a, **k: (seen.update(k), (True, ""))[1])
    monkeypatch.setattr(ST, "seed_artifact", lambda *a, **k: (True, ""))
    monkeypatch.setattr(A, "TEST_P_MIN", 0.85)
    A.test_entry_ok("n", SimpleNamespace(near=frozenset(), led=SimpleNamespace(nodes={"n": SimpleNamespace(
        base="", parent="p")}), by={}, fit=lambda b: (None, {})), "inc", 4)
    assert seen["p_min"] == 0.85
    old = A.TEST_P_MIN
    MT.apply(SP.tunable_values({"acquire.test_p_min": 0.65}), None)
    assert A.TEST_P_MIN == 0.65
    A.TEST_P_MIN = old


def test_a_round_records_the_incumbent_after_a_mid_round_switch():
    """10-13: round 1 opened on n0655@065b87f; n1248@065b87f passed mid-round, yet the closed round still recorded
    n0655. The open round follows a switch (``track_incumbent``, from the tick) and close() records the incumbent at
    close, the switches kept."""
    st = _round({})
    assert not FN.track_incumbent(st, "inc", 60.0) and "switches" not in st["open"]
    assert FN.track_incumbent(st, "n1248", 70.0) and st["open"]["incumbent"] == "n1248"
    assert st["open"]["switches"] == [{"from": "inc", "to": "n1248", "ts": 70.0}]
    assert not FN.track_incumbent({"open": None}, "x", 1.0)
    st2 = _round({})
    led = _led([_node("x1", "request=r1 branch=1.1")], [_run("x1", 1)])
    v = SimpleNamespace(led=led, judged=[], incumbent="n1248")
    closed = FN.close(st2, v, 500.0)
    assert closed["incumbent"] == "n1248" and closed["switches"][0]["from"] == "inc"
