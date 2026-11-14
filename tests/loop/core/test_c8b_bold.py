"""v3-c8b: the BOLD structural lane and diverge-then-judge (loop.core.bold) -- the share / floor, the step-0 identity
exemption without any banned-init exemption, the knob-only rejection, the propose_slate ordering, the 5-trial tuning
cap and the meta report's originality mix."""

from __future__ import annotations

import dataclasses
from collections import Counter
from types import SimpleNamespace

import pytest

from loop.adapters import initcompat as ICK
from loop.agents import researcher as RS
from loop.contracts import (CandidateRequest, InitCompat, Node, NodeStatus, Operator, Strategy, Worktree)
from loop.core import alloc as AL
from loop.core import bold as BOLD
from loop.core import data as D
from loop.core import ideation as ID
from loop.spec import IdeationCfg
from loop.store import SqliteStore

from ..test_tick import _ledger
from .conftest import TASK
from .test_ideation import README


def _with(ctx, **ideation):
    base = ctx.cfg.ideation or IdeationCfg()
    return dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, ideation=dataclasses.replace(base, **ideation)))


@pytest.fixture
def benv(ctx, tmp_path):
    db = _ledger(tmp_path / "ledger.db")
    ctx = _with(dataclasses.replace(ctx, task_id=TASK), bold_share=0.3, diverge_explore=True)
    store = SqliteStore(db, TASK)
    readme = tmp_path / "README.md"
    readme.write_text(README)
    yield ctx, store, D.load(db, TASK), ID.Ideation(ctx, store, readme=readme)
    store.close()


# ---------------------------------------------------------------------------------------------- the share / floor
def test_bold_weight_is_the_share_without_bandit_evidence_and_floored_while_n_is_small():
    w = {"explore": 0.6, "literature": 0.3, "repair": 0.1}
    got = BOLD.weight(w, 0, share=0.3, floor=0.2, floor_n=20, n0=5.0)
    assert got["bold"] == pytest.approx(0.3) and sum(got.values()) == pytest.approx(1.0)
    assert got["explore"] / got["literature"] == pytest.approx(2.0)               # the rest keep their proportions
    starved = {**w, "bold": 0.01}                                                 # the bandit says bold is poor ...
    assert BOLD.weight(starved, 10, share=0.3, floor=0.2, floor_n=20, n0=5.0)["bold"] == pytest.approx(0.2)
    # ... and once n is past floor_n the bandit may take it below the floor (blended with n0 pseudo-sessions)
    late = BOLD.weight(starved, 45, share=0.3, floor=0.2, floor_n=20, n0=5.0)["bold"]
    assert late == pytest.approx((5 * 0.3 + 45 * 0.01) / 50) and late < 0.2
    good = BOLD.weight({**w, "bold": 0.7}, 30, share=0.3, floor=0.2, floor_n=20, n0=5.0)["bold"]
    assert good == pytest.approx((5 * 0.3 + 30 * 0.7) / 35)                       # the bandit can also raise it
    assert "bold" not in BOLD.weight({**w, "bold": 0.5}, 0, share=0.0, floor=0.2, floor_n=20, n0=5.0)


def test_kind_for_gives_bold_its_share_and_the_bandit_cannot_starve_it(benv, monkeypatch):
    ctx, store, led, ide = benv
    w = {"explore": 0.6, "literature": 0.25, "repair": 0.1, "program": 0.05, "bold": 0.001}
    monkeypatch.setattr(ide, "kind_table", lambda v: {k: {"w": x} for k, x in w.items()})
    monkeypatch.setattr(ide, "program_share", lambda v=None: 0.0)
    v = SimpleNamespace(led=SimpleNamespace(nodes={}, epoch=""))
    got = []
    for i in range(50):
        k = ide.kind_for("explore", v)
        got.append(k)
        store.log("research_started", request_id=f"r{i}", brief_kind=k)
    c = Counter(got)
    assert 14 <= c["bold"] <= 16, c                    # no bold node measured yet: the operator share .3
    # 10 bold nodes measured (< floor_n 20): the bandit's .001 blended with 5 pseudo-sessions is .10 -> the .2 floor
    bold_nodes = {f"b{i}": SimpleNamespace(base="", notes="brief=bold", created_ts=0.0) for i in range(10)}
    v2 = SimpleNamespace(led=SimpleNamespace(nodes=bold_nodes, epoch=""))
    store2 = []
    for i in range(50):
        k = ide.kind_for("explore", v2)
        store2.append(k)
        store.log("research_started", request_id=f"s{i}", brief_kind=k)
    assert 19 <= Counter(got + store2)["bold"] <= 21   # the bandit's .001 is held at the .2 floor (n_bold < floor_n)
    assert ide.kind_for("exploit", v) == "repair"      # the other lanes keep their kinds


def test_bold_quota_in_the_fixed_rotation_and_the_request_is_a_moonshot_draft(benv):
    ctx, store, led, ide = benv
    ide.ctx = _with(ctx, kind_weight=False)
    got = []
    for i in range(20):
        k = ide.kind_for("explore", None)
        got.append(k)
        store.log("research_started", request_id=f"r{i}", brief_kind=k)
    assert Counter(got)["bold"] == 6                    # .3 of 20
    assert ID.KIND_REQUEST["bold"] == (Strategy.MOONSHOT, Operator.DRAFT) and "bold" in ID.KINDS


def test_the_bold_brief_has_the_taxonomy_and_no_lever_or_budget_anchoring(benv):
    ctx, store, led, ide = benv
    b = ide.brief_for("explore", led, kind="bold")
    assert b.kind == "bold" and not b.shortlist
    for must in ("BOLD STRUCTURAL brief", "pretext", "data", "architecture", "schedule", "context", "pathology",
                 "propose_slate", "bold_knob_only", "STEP-0 IDENTITY IS RELAXED", "GenBio", "max_train_flops",
                 "Structural coverage"):
        assert must in b.text, must
    for never in ("## Untested levers", "(j) Incumbent diagnostics", "## Per-axis headroom", "dose_knob",
                  "## Engagement of recent ideas", "Decisive"):
        assert never not in b.text, never
    req = ide.request(b, ide.view(led))
    assert req.strategy is Strategy.MOONSHOT and req.k_candidates == 0 and req.brief_kind == "bold"
    ex = ide.brief_for("explore", led, kind="explore")             # explore keeps its brief + the slate section
    assert "## Diverge, then judge" in ex.text and ide.k_for("explore") == 0
    assert ide.k_for("crossover") == int(ide.opt("k_candidates"))


# ---------------------------------------------------------------------------------------------- init identity
def _measure(**over):
    m = {"child": {"constructed": True, "finite": True, "loss": 9.0, "feature_rms": 1.0, "params": 110},
         "parent": {"constructed": True, "finite": True, "loss": 2.0, "feature_rms": 1.0, "params": 100},
         "keys": {"checked": True, "missing": ["ctx.mixer.weight"], "unexpected": [],
                  "identity": {"rel_gap": 0.3, "loss_child": 9.0, "loss_parent": 2.0}}}
    for k, v in over.items():
        m[k] = {**m[k], **v}
    return m


def test_relax_identity_passes_a_randomly_initialised_new_module_only():
    ic = ICK.evaluate(_measure())
    assert not ic.ok and ic.rule == "init_incompatible"
    got = BOLD.relax_identity(ic)
    assert got.ok and got.checks["bold_identity_relaxed"]["relaxed"] == ["initial_loss", "checkpoint_keys"]
    renamed = ICK.evaluate(_measure(keys={"unexpected": ["blocks.0.attn.qkv.weight"]}))   # a parent tensor dropped
    assert not BOLD.relax_identity(renamed).ok
    nan = ICK.evaluate(_measure(child={"finite": False}))                                  # no finite forward
    assert not BOLD.relax_identity(nan).ok
    big = ICK.evaluate(_measure(child={"params": 400}))                                    # the param ratio stands
    assert not BOLD.relax_identity(big).ok
    ok = ICK.evaluate(_measure(child={"loss": 2.0}, keys={"missing": []}))
    assert BOLD.relax_identity(ok) is ok


def test_relax_identity_never_exempts_a_banned_init_or_the_flop_cap():
    m = _measure(keys={"swap": True}, child={"source": "hf-hub:MahmoodLab/UNI2-h", "type": "vit_large"})
    banned = ICK.evaluate(m, ICK.Limits(deny=(r"uni2|MahmoodLab",)))
    assert banned.swap and not banned.ok and banned.rule == "init_illegal"
    assert BOLD.relax_identity(banned) is banned
    over = ICK.evaluate({**m, "flops": {"per_sample": 2.0, "max_samples": 10, "max_flops": 10.0}}, ICK.Limits())
    assert over.rule == "over_flop_cap" and BOLD.relax_identity(over) is over


class _Adapter:
    def __init__(self, diff, ic):
        self.d, self.ic = diff, ic

    def diff(self, wt):
        return self.d, ("model.py",), {}

    def init_compat(self, wt, *, parent_commit):
        return self.ic


DIFF = ("diff --git a/model.py b/model.py\n--- a/model.py\n+++ b/model.py\n@@ -1,2 +1,6 @@\n class Net:\n"
        "+    def context_mixer(self, x, neighbours):\n+        return self.mix(torch.cat([x, neighbours], -1))\n"
        "     pass\n")


def _req(kind, n=1):
    return CandidateRequest(id=f"{ID.REQUEST_PREFIX}{n:05d}", task_id=TASK, strategy=Strategy.MOONSHOT,
                            operator=Operator.DRAFT, island="main", parent_ids=("root",), brief="b", brief_kind=kind)


def test_the_researcher_relaxes_identity_for_bold_only_and_keeps_every_banned_init_rule(ctx, tmp_path):
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, init_compat=dataclasses.replace(
        ctx.cfg.init_compat, deny=(r"uni2|MahmoodLab",))))
    store = SqliteStore(tmp_path / "s.db", ctx.task_id)
    wt = Worktree(str(tmp_path), "b", "c0")
    ident = ICK.evaluate(_measure())

    def check(kind, diff=DIFF, ic=ident, n=1):
        r = RS.LLMResearcher(ctx, None, _Adapter(diff, ic))
        return r.check(_req(kind, n), store, wt, family="architecture", mechanism_tag="slide_context_mixer",
                       claim="mix each tile with its neighbours", parent_commit="c0")
    assert check("bold").startswith("OK") and store.events("bold_identity_relaxed")
    assert check("explore", n=2).startswith("REJECT init_incompatible")
    illegal = DIFF.replace("neighbours], -1))", "neighbours], -1))\n+        w = load('hf-hub:MahmoodLab/UNI2-h')")
    assert check("bold", illegal, n=3).startswith("REJECT init_illegal")          # the diff rule: never exempt
    genbio = DIFF.replace("neighbours], -1))", "neighbours], -1))\n+        w = load('genbio-pathfm')")
    assert check("bold", genbio, n=4).startswith("REJECT disallowed_data")
    swap = InitCompat(False, "init_illegal: banned", {"init_legal": {"ok": False}, "init_swap": {"to": {}}})
    assert check("bold", ic=swap, n=5).startswith("REJECT init_illegal")          # a banned init swap stays banned
    store.close()


# ---------------------------------------------------------------------------------------------- knob-only
CFG_ONLY = "--- a/config.yaml\n+++ b/config.yaml\n@@ -3 +3 @@\n-  lr: 1.0e-4\n+  lr: 2.0e-4\n"
CONST = ("--- a/train.py\n+++ b/train.py\n@@ -9,2 +9,2 @@\n-    loss = a + 0.5 * koleo(z)  # weight\n"
         "+    loss = a + 0.25 * koleo(z)  # weight\n-    t = 'cosine'\n+    t = 'linear'\n")
NEWDEF = CONST + "@@ -40 +40,3 @@\n+def neighbour_loss(z, zn):\n+    return (z - zn).pow(2).mean()\n"
NEWFILE = "--- /dev/null\n+++ b/context.py\n@@ -0,0 +1 @@\n+X = 1\n"
DATA = "--- a/config.yaml\n+++ b/config.yaml\n@@ -3 +3 @@\n-  dataset_dir: /bank/a\n+  dataset_dir: /bank/retiled_448\n"
NEWLINE = "--- a/train.py\n+++ b/train.py\n@@ -9 +9,2 @@\n     loss = a\n+    loss = loss + aux(z, neighbours)\n"


@pytest.mark.parametrize("diff,knob", [(CFG_ONLY, True), (CONST, True), (CFG_ONLY + CONST, True), (NEWDEF, False),
                                       (NEWFILE, False), (DATA, False), (NEWLINE, False), ("", False)])
def test_knob_only_is_judged_on_the_diff(diff, knob):
    assert BOLD.knob_only(diff) is knob


def test_bold_rejects_a_knob_only_change_at_check_and_at_intake_with_the_judge(benv, tmp_path):
    ctx, store, led, ide = benv
    assert BOLD.knob_rule(ctx, CONST)[0] == "bold_knob_only"
    assert BOLD.knob_rule(ctx, NEWDEF) == ("", "")
    assert BOLD.knob_rule(ctx, NEWLINE, judge=lambda d: True)[0] == "bold_knob_only"       # the judge says knob
    assert BOLD.knob_rule(ctx, NEWDEF, judge=lambda d: True) == ("", "")    # 11-10: ... but a new def is code (n0082)
    assert BOLD.knob_rule(ctx, NEWDEF, judge=lambda d: False) == ("", "")
    # v3 review M3: no verdict (the backend is down) FAILS CLOSED: held + retried, never a silent pass
    assert BOLD.knob_rule(ctx, NEWDEF, judge=lambda d: None) == (BOLD.HOLD, "")
    assert BOLD.knob_rule(_with(ctx, bold_knob_judge=False), NEWDEF, judge=lambda d: True) == ("", "")

    class LLM:
        def __init__(self, ans):
            self.ans = ans

        def complete_json(self, role, *, system, prompt, schema):
            assert role == "bold_knob_judge" and "Diff" in prompt
            if isinstance(self.ans, Exception):
                raise self.ans
            return {"knob_only": self.ans}
    cand = SimpleNamespace(diff_text=NEWDEF, hypothesis=SimpleNamespace(claim="a neighbour loss"))
    ide.llm = LLM(True)
    knobby = SimpleNamespace(diff_text=NEWLINE, hypothesis=cand.hypothesis)
    assert ide._bold_rule(knobby)[0] == "bold_knob_only" and store.events("bold_knob_judge")
    assert ide._bold_rule(cand) == ("", "")                    # 11-10: a new def overrides the judge's knob verdict
    ide.llm = LLM(RuntimeError("qwen down"))
    # 10-22: qwen down no longer holds: the deterministic judge decides (a new def is structural) and the log says so
    assert ide._bold_rule(cand) == ("", "") and store.events("bold_knob_judge_error")
    assert store.events("bold_knob_judge")[-1]["judge"] == "deterministic"
    ide.llm = None
    assert ide._bold_rule(SimpleNamespace(diff_text=CONST, hypothesis=cand.hypothesis))[0] == "bold_knob_only"
    # the researcher's check bounces a knob-only bold diff in-session (a non-bold session is not judged by it)
    r = RS.LLMResearcher(ctx, None, _Adapter(CONST, None))
    r.adapter.diff = lambda wt: (CONST, ("train.py",), {})
    got = r.check(_req("bold", 7), store, Worktree(str(tmp_path), "b", "c0"), family="optimization",
                  mechanism_tag="koleo_half", claim="halve the koleo weight", parent_commit="c0")
    assert got.startswith("REJECT bold_knob_only")


# ---------------------------------------------------------------------------------------------- the slate
def _cand(i, cat, plaus=0.5, **kw):
    return {"title": f"idea {i} {cat}", "category": cat, "family": "objective", "mechanism_tag": f"m{i}_{cat}",
            "mechanism": f"change number {i} in category {cat}: " + " ".join(f"w{i}x{j}" for j in range(14)),
            "why": "evidence", "plausibility": plaus, "p_neg": 0.4, "median": 0.002, "p_gap": 0.1, **kw}


SLATE = [_cand(i, c) for i, c in enumerate(["pretext", "pretext", "data", "data", "architecture", "schedule",
                                             "context", "context", "pathology", "pathology"])]


def _gated(tmp_path, ctx, store, kind="bold", judge=None):
    t = RS.WorktreeTools(None, Worktree(str(tmp_path), "b", "c0"), writable=("*",), locked=(),
                         checker=lambda **k: "OK")
    t.slate_gate = BOLD.Slate(ctx, store, "req-v3-00001", kind, judge=judge, logger=store.log)
    t.bold = kind == "bold"
    return t


def test_the_slate_must_precede_every_edit_and_the_finish(benv, tmp_path):
    ctx, store, led, ide = benv
    (tmp_path / "train.py").write_text("x = 1\n")
    t = _gated(tmp_path, ctx, store)
    for name, args in (("edit_file", {"path": "train.py", "old": "x = 1", "new": "x = 2"}),
                       ("write_file", {"path": "new.py", "content": "y"}), ("set_config", {"key": "a", "value": 1}),
                       ("bash", {"command": "echo hi > train.py"}), ("bash", {"command": "sed -i s/1/2/ train.py"})):
        assert t(name, args).startswith("REFUSED: diverge first"), name
    assert (tmp_path / "train.py").read_text() == "x = 1\n" and not (tmp_path / "new.py").exists()
    assert not t("read_file", {"path": "train.py"}).startswith("REFUSED")          # reading is never blocked
    assert not t("bash", {"command": "cat train.py 2>/dev/null"}).startswith("REFUSED")
    assert t.check_finish({"hypothesis": {}, "handoff": {}}).startswith("REJECT slate_first")
    bad = t("propose_slate", {"candidates": SLATE[:3]})                             # too few, too few categories
    assert bad.startswith("REJECT slate") and "8-12" in bad and ">= 4 categories" in bad
    crowd = t("propose_slate", {"candidates": [_cand(i, "pretext") for i in range(8)]})
    assert "at most 3 candidates per category" in crowd
    ok = t("propose_slate", {"candidates": SLATE})
    assert "BUILD candidate #" in ok and t.slate_gate.done
    assert t("edit_file", {"path": "train.py", "old": "x = 1", "new": "x = 2"}).startswith("edited") or \
        (tmp_path / "train.py").read_text() == "x = 2\n"
    ev = store.events("slate")
    assert len(ev) == 1 and len(ev[0]["candidates"]) == 10 and ev[0]["chosen"] == t.slate_gate.result["chosen"]
    assert {"originality", "plausibility", "score", "category", "mechanism"} <= set(ev[0]["candidates"][0])
    plan = t.slate_gate.plan()
    assert plan["bold_category"] == SLATE[plan["chosen"]]["category"] and len(plan["candidates"]) == 10
    assert plan["prediction"] == {"p_neg": 0.4, "median": 0.002, "p_gap": 0.1}


def test_the_slate_ranks_originality_x_plausibility_and_rejects_saturated_clusters(benv, monkeypatch):
    ctx, store, led, ide = benv
    cands = [dict(c) for c in SLATE]
    cands[3]["plausibility"] = 0.9                       # the most plausible ...
    sat = AL.Cluster("C9", ["n1"], dict(AL._norm(dict(BOLD._terms(cands[3])))), flag="saturated-negative")
    monkeypatch.setitem(AL.LAST, "clusters", [sat])
    monkeypatch.setitem(AL.LAST, "df", Counter(BOLD._terms(cands[3])))
    monkeypatch.setitem(AL.LAST, "n", 10)
    res = BOLD.rank(ctx, store, cands, docs=[])
    rows = {r["i"]: r for r in res["rows"]}
    assert rows[3]["rejected"].startswith("saturated-negative cluster C9") and rows[3]["score"] == 0.0
    assert res["chosen"] != 3 and res["judge"] == "self"
    for r in res["rows"]:
        if not r["rejected"]:
            assert r["score"] == pytest.approx(r["originality"] * r["plausibility"], abs=1e-4)
    # the judge's plausibility is averaged with the session's own; a raising judge falls back to the session's
    res2 = BOLD.rank(ctx, store, cands, judge=lambda cs: [0.1] * len(cs), docs=[])
    assert res2["judge"] == "qwen" and res2["rows"][0]["plausibility"] == pytest.approx((0.5 + 0.1) / 2)
    res3 = BOLD.rank(ctx, store, cands, judge=lambda cs: 1 / 0, docs=[])
    assert res3["judge"] == "self"
    # a candidate that repeats a ledger node's mechanism text is less original than a fresh one
    monkeypatch.setitem(AL.LAST, "clusters", [])
    others = [BOLD._terms(c) for c in cands[5:9]]
    near = BOLD.originality(ctx, store, cands[0], docs=[BOLD._terms(cands[0]), *others])
    far = BOLD.originality(ctx, store, cands[0], docs=others)
    assert near["originality"] < far["originality"]


def test_explore_sessions_get_the_slate_when_the_flag_is_on(benv):
    ctx, store, led, ide = benv
    assert BOLD.slate_kinds(ctx) == ("bold", "explore", "literature")                  # the repair portfolio is opt-in (c9 fix S3)
    assert BOLD.slate_kinds(_with(ctx, portfolio_repair=True)) == ("bold", "explore", "literature", "repair")
    assert BOLD.slate_kinds(_with(ctx, portfolio=False)) == ("bold", "explore", "literature")
    assert BOLD.slate_kinds(_with(ctx, diverge_explore=False)) == ("bold",)


# ---------------------------------------------------------------------------------------------- tuning cap
def _bold_node(store, nid="n9001", notes="brief=bold"):
    n = Node(id=nid, task_id=store.task_id if hasattr(store, "task_id") else TASK, commit="c", parent_id="root",
             strategy=Strategy.MOONSHOT, operator=Operator.DRAFT, family="objective", mechanism_tag="slide_ctx",
             diff_hash=f"h{nid}", status=NodeStatus.SCREENED_OUT, notes=notes)
    store.put_node(n)
    return n


def test_a_bold_family_gets_at_most_five_tuning_trials(benv):
    from loop import campaign as CP
    from loop import tune as MT
    ctx, store, led, ide = benv
    node = _bold_node(store)
    post = SimpleNamespace(utility=SimpleNamespace(n=2), gain_p=0.5, veto_axes=[],
                           increment=SimpleNamespace(mean=0.001, sd=0.002))
    snap = SimpleNamespace(nodes={node.id: post}, sigma={}, own_confirm={})
    assert BOLD.tune_room(ctx, store, node) == 5
    assert MT.classify(ctx, store, snap, node)["budget"] == 5
    store.set_state(MT.K_TUNE, {"seq": 1, "current": {}, "tuned": [], "laddered": [node.id],
                                "batches": [{"batch": 1, "base": node.id, "kind": "ladder", "trials": ["a", "b", "c"]}]})
    assert BOLD.tune_used(store, node.id) == 3 and MT.classify(ctx, store, snap, node)["budget"] == 2
    store.set_state(MT.K_TUNE, {"seq": 2, "current": {"2": {"batch": 2, "base": node.id, "trials": ["d", "e"]}},
                                "tuned": [], "laddered": [node.id],
                                "batches": [{"batch": 1, "base": node.id, "kind": "ladder", "trials": ["a", "b", "c"]}]})
    assert BOLD.tune_room(ctx, store, node) == 0 and MT.classify(ctx, store, snap, node) is None
    assert MT._edge_rung(ctx, store, None, snap, {"base": node.id, "batch": 1}, None) == []
    other = _bold_node(store, "n9002", notes="brief=explore")
    assert BOLD.tune_room(ctx, store, other) is None
    got = MT.classify(ctx, store, SimpleNamespace(nodes={other.id: post}, sigma={}, own_confirm={}), other)
    mt = ctx.cfg.moonshot_tune                           # a non-bold node keeps loop.tune's own budgets
    assert got is not None and got["budget"] in (mt.max_trials_major, mt.max_trials_minor) and "bold" not in got["reason"]
    assert not CP.claims(ctx, store, node)               # a bold node is never a campaign's
    assert BOLD.tune_room(_with(ctx, bold_tune_max=8), store, node) == 3


# ---------------------------------------------------------------------------------------------- meta report
def test_the_meta_report_has_an_originality_mix_per_kind(benv, tmp_path):
    from loop.core import meta as MM
    ctx, store, led, ide = benv
    rows = [("n9101", "brief=bold bold_cat=context", NEWDEF), ("n9102", "brief=bold bold_cat=data", NEWFILE),
            ("n9103", "brief=explore", CONST), ("n9104", "brief=explore", CFG_ONLY),
            ("n9105", "brief=explore", NEWDEF)]
    for nid, notes, diff in rows:
        p = tmp_path / f"{nid}.diff"
        p.write_text(diff)
        store.put_node(Node(id=nid, task_id=TASK, commit="c" + nid, parent_id="root", strategy=Strategy.EXPLORE,
                            operator=Operator.DRAFT, family="objective", mechanism_tag="neighbour_loss",
                            diff_hash=nid, status=NodeStatus.QUEUED, diff_path=str(p), notes=notes,
                            created_ts=2e9))
    store.log("slate", request_id="r1", kind="bold", judge="self", chosen=0,
              candidates=[{"category": "context", "rejected": ""}, {"category": "data", "rejected": "sat"}])
    m = BOLD.mix(ctx, store, since=1.9e9)
    assert m["kinds"]["bold"]["n"] == 2 and m["kinds"]["bold"]["knob_only"] == 0.0
    assert m["kinds"]["bold"]["cats"] == {"context": 0.5, "data": 0.5}
    assert m["kinds"]["explore"]["n"] == 3 and m["kinds"]["explore"]["knob_only"] == pytest.approx(2 / 3, abs=1e-3)
    assert m["kinds"]["all"]["n"] == 5
    m["slates"] = BOLD.mix(ctx, store, since=0.0)["slates"]          # the slate event carries the wall clock ts
    rep = MM.MetaReport(ts=2e9, since=1.9e9, originality=m)
    text = rep.markdown()
    assert "## Originality mix" in text and "bold | 2 | 0% |" in text and "explore | 3 | 67% |" in text
    assert "slates: 1 sessions, 2 candidates (1 rejected" in text


def test_a_bold_finish_needs_no_budget_rows_and_the_plan_carries_the_slate(benv, tmp_path):
    from loop.contracts import PANEL
    ctx, store, led, ide = benv
    t = _gated(tmp_path, ctx, store)
    t.v3 = True
    t("propose_slate", {"candidates": SLATE})
    hy = {"claim": "c", "mechanism_tag": "m", "family": "objective", "falsification": "f",
          "predictions": {PANEL: {"sign": 1, "size": 0.01}},
          "mechanism_prediction": {"diagnostic": "lr_end_frac", "direction": 1, "axes": {"cls": 1}}}
    assert t.check_finish({"hypothesis": hy, "handoff": {"actions": "a", "learnings": "l", "concerns": "c"}}) == ""
    sch = RS.finish_schema(ctx, True, bold=True)["properties"]["hypothesis"]["properties"]["mechanism_prediction"]
    assert "budget" not in sch["required"] and "budget" in RS.finish_schema(ctx, True)["properties"][
        "hypothesis"]["properties"]["mechanism_prediction"]["required"]
    r = RS.LLMResearcher.__new__(RS.LLMResearcher)
    r.ctx = ctx
    plan = r._plan(_req("bold"), {"hypothesis": hy, "knobs": [{"key": "loss.w", "lo": 0.1, "hi": 1, "scale": "log"}]},
                   t, Worktree(str(tmp_path), "b", "c0"))
    assert plan["bold_category"] and plan["slate"]["rows"] and "knobs" not in plan     # no separate designed wave
    assert plan["prediction"]["unit"] == "visible_composite" and plan["chosen"] == t.slate_gate.result["chosen"]


def test_bold_quota_comes_before_the_program_quota_and_program_still_runs(benv, monkeypatch):
    ctx, store, led, ide = benv
    w = {"explore": 0.6, "literature": 0.25, "repair": 0.1, "program": 0.05}
    monkeypatch.setattr(ide, "kind_table", lambda v: {k: {"w": x} for k, x in w.items()})
    monkeypatch.setattr(ide, "program_share", lambda v=None: 0.9)       # programs would take every slot ...
    monkeypatch.setattr(ide, "_op_next", lambda: {"id": "op-x"})        # ... and one is pending
    v = SimpleNamespace(led=SimpleNamespace(nodes={}, epoch=""))
    got = []
    for i in range(50):
        got.append(k := ide.kind_for("explore", v))
        store.log("research_started", request_id=f"r{i}", brief_kind=k)
    c = Counter(got)
    assert 14 <= c["bold"] <= 16 and c["program"] >= 30, c              # ~30% bold AND program runs
    # shadow (dry): two explore-lane slots per tick -> bold first, then the pending program, every tick
    ide._dry = True
    for _ in range(3):
        ide._tick_kinds = []
        assert [ide.kind_for("explore", v), ide.kind_for("explore", v)][1] == "program"
        assert ide._tick_kinds[0] in ("bold", "program") and "program" in ide._tick_kinds
