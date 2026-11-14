"""v3-c9 C3: the verbalized-sampling PORTFOLIO for explore / literature / repair sessions (loop.core.bold): 6-8 directions
(repair: a light 3-5) with verbalized probabilities as ONE joint distribution, grounded in distinct history clusters, the
chosen direction citing the history nodes / clusters it differs from (existence + explanation length checked at finish)."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from loop.agents import researcher as RS
from loop.contracts import Worktree
from loop.core import bold as BOLD
from loop.core import ideation as ID
from loop.spec import IdeationCfg
from loop.store import SqliteStore

from .core.conftest import TASK
from .core.test_ideation import README
from .test_tick import _ledger
from loop.core import data as D


def _with(ctx, **ideation):
    base = ctx.cfg.ideation or IdeationCfg()
    return dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, ideation=dataclasses.replace(base, **ideation)))


class Tab:
    """A history table stand-in: node -> its cluster id (rows carry a tag and a description, clusters a label)."""

    def __init__(self, nodes):
        self.by = dict(nodes)
        self.by_id = {n: {"id": n, "tag": f"fam{c or 'x'}/tag{n}", "desc": f"node {n} trains a contrastive objective",
                          "cluster": c} for n, c in nodes.items()}
        self.cluster_ids = {c: SimpleNamespace(id=c, label=f"cluster {c} words", category="objective")
                            for c in set(nodes.values()) if c} | {"C9": SimpleNamespace(id="C9", label="cluster C9 words",
                                                                                        category="objective")}

    def exists(self, i):
        return i in self.by or i in self.cluster_ids

    def node_cluster(self, i):
        return self.by.get(i, "")


TAB = Tab({"n0001": "C1", "n0002": "C2", "n0003": "C3", "n0004": "C4", "n0005": ""})
CATS = ["pretext", "data", "architecture", "schedule", "context", "pathology", "pretext", "data"]
PROBS = [0.40, 0.22, 0.12, 0.09, 0.07, 0.05, 0.03, 0.02]
GROUND = ["C1", "C2", "n0003", "C4", "n0001", "C2", "C3", "n0002"]


def _cand(i, cat, prob, ground, plaus=0.5):
    return {"title": f"idea {i} {cat}", "category": cat, "family": "objective", "mechanism_tag": f"m{i}_{cat}",
            "mechanism": f"change number {i} in category {cat}: " + " ".join(f"w{i}x{j}" for j in range(14)),
            "why": "evidence", "plausibility": plaus, "p_neg": 0.4, "median": 0.002, "p_gap": 0.1,
            "probability": prob, "grounded_in": ground,
            "differs_from": [{"id": ground, "how": "pools the patch tokens and gates them with an entropy mask instead"}]}


def _slate(n=7):
    probs = PROBS[:n]
    probs = [p / sum(probs) for p in probs]
    return [_cand(i, CATS[i], probs[i], GROUND[i]) for i in range(n)]


@pytest.fixture
def penv(ctx, tmp_path):
    db = _ledger(tmp_path / "ledger.db")
    ctx = _with(dataclasses.replace(ctx, task_id=TASK), diverge_explore=True, portfolio_repair=True)
    store = SqliteStore(db, TASK)
    (readme := tmp_path / "README.md").write_text(README)
    yield ctx, store, D.load(db, TASK), ID.Ideation(ctx, store, readme=readme)
    store.close()


def _gate(ctx, store, kind="explore", table=TAB):
    return BOLD.Slate(ctx, store, "req-v3-00001", kind, logger=store.log, history=(lambda: table) if table else None)


def _tools(tmp_path, gate):
    t = RS.WorktreeTools(None, Worktree(str(tmp_path), "b", "c0"), writable=("*",), locked=(), checker=lambda **k: "OK")
    t.slate_gate = gate
    return t


# ---------------------------------------------------------------------------------------------- kinds + limits
def test_explore_literature_and_repair_propose_portfolios_and_bold_keeps_its_slate(penv):
    ctx, store, led, ide = penv
    assert BOLD.slate_kinds(ctx) == ("bold", "explore", "literature", "repair")
    for kind in ("explore", "literature"):
        assert BOLD.portfolio_on(ctx, kind) and BOLD.slate_limits(ctx, kind)["k_min"] == 6 and \
            BOLD.slate_limits(ctx, kind)["k_max"] == 8 and BOLD.slate_limits(ctx, kind)["min_clusters"] == 3
    rep = BOLD.slate_limits(ctx, "repair")
    assert (rep["k_min"], rep["k_max"], rep["min_clusters"], rep["min_cats"]) == (3, 5, 2, 2)
    assert not BOLD.portfolio_on(ctx, "bold") and BOLD.slate_limits(ctx, "bold")["k_min"] == 8
    assert BOLD.slate_tool(ctx, "bold") is BOLD.SLATE_TOOL
    d = BOLD.slate_tool(ctx, "explore")["description"]
    assert "6-8 DIRECTIONS" in d and "JOINT PORTFOLIO" in d and "NOT variations of the most typical answer" in d
    assert "3-5 DIRECTIONS" in BOLD.slate_tool(ctx, "repair")["description"]
    assert {"probability", "grounded_in", "differs_from"} <= set(BOLD.CANDIDATE["properties"])
    assert not BOLD.portfolio_on(_with(ctx, portfolio=False), "explore")
    assert BOLD.slate_kinds(_with(ctx, portfolio=False)) == ("bold", "explore", "literature")
    assert BOLD.slate_limits(_with(ctx, portfolio=False), "explore")["k_min"] == 8        # the old slate
    assert BOLD.slate_kinds(_with(ctx, portfolio_repair=False)) == ("bold", "explore", "literature")
    from loop.spec import IdeationCfg as IC
    assert IC().portfolio_repair is False                    # review fix S3: explore / literature first
    assert BOLD.slate_limits(_with(ctx, portfolio_min=5, portfolio_max=6), "explore")["k_max"] == 6


# ---------------------------------------------------------------------------------------------- the portfolio rules
def test_a_well_formed_portfolio_is_accepted_ranked_and_logged(penv):
    ctx, store, led, ide = penv
    g = _gate(ctx, store)
    out = g(_slate(7))
    assert "BUILD candidate #" in out and g.done
    ev = store.events("slate")[0]
    assert len(ev["candidates"]) == 7 and all("probability" in r for r in ev["candidates"])
    plan = g.plan()
    assert plan["portfolio"]["table"] is True and plan["portfolio"]["grounded_in"] in GROUND
    assert set(plan["portfolio"]["clusters"]) >= {"C1", "C2", "C3", "C4"} - {""}
    assert all("probability" in c and "grounded_in" in c for c in plan["candidates"])


@pytest.mark.parametrize("mutate,needle", [
    (lambda s: s[:5], "propose 6-8 candidates (you gave 5)"),
    (lambda s: s + [_cand(7, "data", 0.0, "C1"), _cand(8, "context", 0.0, "C2")], "propose 6-8 candidates (you gave 9)"),
    (lambda s: [{k: v for k, v in c.items() if k != "probability"} for c in s], "`probability` (the chance a typical"),
    (lambda s: [{**c, "probability": 0.5} for c in s], "the probabilities must sum to about 1"),
    (lambda s: [{**c, "probability": 1 / 6} for c in s[:6]], "sample the tails"),
    (lambda s: [{k: v for k, v in c.items() if k != "grounded_in"} for c in s], "`grounded_in` must be a cluster id"),
    (lambda s: [{**c, "grounded_in": "n9999"} if i == 2 else c for i, c in enumerate(s)], "ids not in the history table"),
    (lambda s: [{**c, "grounded_in": "C1" if i % 2 else "n0001"} for i, c in enumerate(s)], "DISTINCT history clusters"),
])
def test_a_malformed_portfolio_comes_back_with_what_to_fix(penv, mutate, needle):
    ctx, store, led, ide = penv
    g = _gate(ctx, store)
    out = g(mutate(_slate(7)))
    assert out.startswith("REJECT slate") and needle in out and not g.done


def test_the_tail_and_sum_tolerances_are_config(penv):
    ctx, store, led, ide = penv
    flat = [{**c, "probability": 1 / 6} for c in _slate(6)]
    assert _gate(ctx, store)(flat).startswith("REJECT slate")
    ok = _gate(_with(ctx, portfolio_tail_p=0.2), store)(flat)               # 1/6 = .167 <= .2 : all are tails
    assert "BUILD candidate" in ok
    off = [{**c, "probability": 0.15} for c in _slate(7)]                    # sum 1.05 ... fine; sum .35 is not
    assert "BUILD" in _gate(ctx, store)(off)
    low = [{**c, "probability": 0.05} for c in _slate(7)]
    assert "sum to about 1" in _gate(ctx, store)(low)
    assert "BUILD" in _gate(_with(ctx, portfolio_sum_tol=0.7), store)(low)


def test_without_a_history_table_the_grounding_is_skipped_but_the_probabilities_still_count(penv):
    ctx, store, led, ide = penv
    bare = [{k: v for k, v in c.items() if k != "grounded_in"} for c in _slate(7)]
    assert "BUILD candidate" in _gate(ctx, store, table=None)(bare)
    noprob = [{k: v for k, v in c.items() if k != "probability"} for c in bare]
    assert "`probability`" in _gate(ctx, store, table=None)(noprob)


def test_a_repair_portfolio_is_light_four_directions_two_clusters(penv):
    ctx, store, led, ide = penv
    g = _gate(ctx, store, "repair")
    four = [_cand(0, "pretext", 0.5, "C1"), _cand(1, "pretext", 0.3, "n0002"), _cand(2, "data", 0.15, "C1"),
            _cand(3, "data", 0.05, "C2")]
    assert "BUILD candidate" in g(four)
    assert _gate(ctx, store, "repair")(_slate(7)).startswith("REJECT slate: propose 3-5 candidates (you gave 7)")
    one_cluster = [{**c, "grounded_in": "C1"} for c in four]
    assert "DISTINCT history clusters (yours cover 1" in _gate(ctx, store, "repair")(one_cluster)


def test_bold_slates_keep_the_old_rules_and_ignore_the_portfolio_fields(penv):
    ctx, store, led, ide = penv
    old = [{k: v for k, v in c.items() if k not in ("probability", "grounded_in")} for c in _slate(8)]
    cats = ["pretext", "pretext", "data", "data", "architecture", "schedule", "context", "pathology"]
    old = [{**c, "category": cats[i]} for i, c in enumerate(old)]
    g = _gate(ctx, store, "bold")
    assert "BUILD candidate" in g(old) and "portfolio" not in g.plan()


def test_a_typical_direction_is_discounted_in_the_ranking(penv, monkeypatch):
    ctx, store, led, ide = penv
    a = _cand(0, "pretext", 0.9, "C1", plaus=0.5)
    b = {**_cand(0, "pretext", 0.9, "C1", plaus=0.5), "probability": 0.02}
    monkeypatch.setattr(BOLD, "originality", lambda *a_, **k: {"originality": 0.8, "cluster": "", "cluster_cos": 0.0,
                                                                "flag": "", "nearest_node_cos": 0.0})
    res = BOLD.rank(ctx, store, [a, b], docs=[], typical_w=0.5)
    rows = {r["i"]: r for r in res["rows"]}
    assert rows[1]["score"] > rows[0]["score"] and res["chosen"] == 1
    assert rows[0]["score"] == pytest.approx(0.8 * 0.5 * (1 - 0.5 * 0.9), abs=1e-4)
    plain = BOLD.rank(ctx, store, [a, b], docs=[])
    assert plain["rows"][0]["score"] == plain["rows"][1]["score"]                  # weight 0: the old ranking


# ---------------------------------------------------------------------------------------------- the citation check
def test_content_chars_ignores_ids_and_punctuation():
    assert BOLD.content_chars("n0001, C7: ---") == 0
    assert BOLD.content_chars("n0001 uses a pooled readout") == len("uses a pooled readout")
    assert BOLD.content_chars("C12 and n1234@065b87f differ!!!") == len("and differ")


@pytest.mark.parametrize("text,exists,needle", [
    ("", None, "must cite >= 1 history node / cluster id"),
    ("it is just different from the rest of them", None, "must cite >= 1 history node"),
    ("n0001", None, "must say HOW"),
    ("n0001 differs", None, "must say HOW"),                                                     # 7 chars < 12
    ("n9999 trains a different head entirely", lambda i: i != "n9999", "n9999, which is not in the history table"),
    ("n0001 and C99 both tune the head", lambda i: i == "n0001", "C99, which is not in the history table"),
    ("n0001 trains a different head entirely", lambda i: True, ""),
    ("C7 trains a different head entirely", lambda i: True, ""),
    ("[differs from: n0001@065b87f pools patches instead of the cls token]", lambda i: True, ""),
])
def test_cite_problem(text, exists, needle):
    got = BOLD.cite_problem(text, exists, min_ids=1, min_chars=12)
    assert (needle in got) if needle else got == ""


def test_cite_problem_min_ids_and_chars_are_config():
    assert "cite >= 2" in BOLD.cite_problem("n0001 trains a different head", None, min_ids=2)
    assert BOLD.cite_problem("n0001 n0002 trains a different head", None, min_ids=2) == ""
    assert "must say HOW" in BOLD.cite_problem("n0001 trains a different head", None, min_chars=40)


def _hy(differs=None):
    hy = {"claim": "c", "mechanism_tag": "m", "family": "objective", "falsification": "f"}
    return {"hypothesis": {**hy, **({"differs_from": differs} if differs is not None else {})},
            "handoff": {"actions": "a", "learnings": "l", "concerns": "c"}}


def _restate(i):
    """The cited node's / cluster's own description, said back as the 'how'."""
    if i in TAB.cluster_ids:
        return f"{TAB.cluster_ids[i].label} {TAB.cluster_ids[i].category}"
    return f"{TAB.by_id[i]['desc']} {TAB.by_id[i]['tag']}"


@pytest.mark.parametrize("mutate,needle", [
    (lambda c: {k: v for k, v in c.items() if k != "differs_from"}, "must carry differs_from"),
    (lambda c: {**c, "differs_from": [{"id": "n7777", "how": "something else entirely different"}]}, "n7777 is not in the history"),
    (lambda c: {**c, "differs_from": [{"id": "n0005", "how": "pools the patch tokens and gates them with an entropy mask"}]},
     "n0005 is unrelated"),                                           # a node of no cluster, other family, other text
    (lambda c: {**c, "differs_from": [{"id": c["grounded_in"], "how": "differs"}]}, "must say what is DIFFERENT"),
    (lambda c: {**c, "differs_from": [{"id": c["grounded_in"], "how": _restate(c["grounded_in"])}]},
     "must say what is DIFFERENT"),                                   # only words of the cited description
])
def test_the_chosen_candidate_must_cite_related_ids_with_new_words_at_propose_time(penv, mutate, needle):
    ctx, store, led, ide = penv
    g = _gate(ctx, store)
    out = g([mutate(c) for c in _slate(7)])                          # whichever candidate the loop picks fails
    assert out.startswith("REJECT slate") and needle in out and "costs one of your slate tries" in out and not g.done
    assert g.tries == 1
    assert "BUILD" in g(_slate(7)) and g.done                         # one retry later it is accepted


def test_a_citation_related_by_family_tag_or_text_similarity_is_accepted(penv):
    ctx, store, led, ide = penv
    base = _cand(0, "pretext", 0.4, "C1")
    tab = TAB
    far = {**base, "grounded_in": "C2"}                                # n0005 is in no cluster
    cite = lambda i: {**far, "differs_from": [{"id": i, "how": "pools the patch tokens and gates them with an entropy mask"}]}
    assert BOLD.differs_problem(cite("n0005"), tab).startswith("n0005 is unrelated")
    assert BOLD.differs_problem({**cite("n0005"), "family": "famx"}, tab) == ""          # same family
    assert BOLD.differs_problem({**cite("n0005"), "mechanism_tag": "tagn0005"}, tab) == ""   # same mechanism tag
    assert BOLD.differs_problem({**cite("n0005"), "title": "node n0005 trains a contrastive objective"}, tab,
                                sim_min=0.3) == ""                                          # text similarity
    assert BOLD.differs_problem(cite("C2"), tab) == ""                 # the grounded cluster itself


def test_a_finish_never_rejects_on_portfolio_cites_it_only_logs(penv, tmp_path):
    ctx, store, led, ide = penv
    g = _gate(ctx, store)
    t = _tools(tmp_path, g)
    assert t.check_finish(_hy("n0001 differs in how it pools")).startswith("REJECT slate_first")     # nothing proposed yet
    t.bounces = 0
    g(_slate(7))
    assert g.done
    for differs in (None, "n7777 does something else entirely", "n0001 differs"):
        t.bounces = 0
        assert t.check_finish(_hy(differs)) == "", differs                  # the finished build goes through ...
    assert store.events("portfolio_cites")                                  # ... and the miss is logged
    t.bounces = RS.MAX_FINISH_BOUNCES
    assert t.check_finish(_hy(None)) == "" and not t.final_reject          # and is never a rejection of the build


def test_bold_and_non_portfolio_finishes_need_no_citation(penv, tmp_path):
    ctx, store, led, ide = penv
    cats = ["pretext", "pretext", "data", "data", "architecture", "schedule", "context", "pathology"]
    old = [{**{k: v for k, v in c.items() if k not in ("probability", "grounded_in")}, "category": cats[i]}
           for i, c in enumerate(_slate(8))]
    t = _tools(tmp_path, _gate(ctx, store, "bold"))
    t.slate_gate(old)
    assert t.check_finish(_hy(None)) == ""
    t2 = _tools(tmp_path, _gate(_with(ctx, portfolio=False), store, "explore"))
    plain = [{**c, "category": cats[i]} for i, c in enumerate(old)]
    t2.slate_gate(plain)
    assert t2.check_finish(_hy(None)) == ""


# ---------------------------------------------------------------------------------------------- the brief
def test_the_brief_asks_for_the_portfolio_and_the_grounding(penv):
    ctx, store, led, ide = penv
    v = ide.view(led)
    expl = ide.brief_text(ide.brief_plan("explore", v, "explore"), led, v).text
    assert "6-8 DIRECTIONS as ONE JOINT PORTFOLIO" in expl and "not variations of the most typical answer" in expl
    assert "DISTINCT clusters/branches" in expl and "differs_from" in expl and ">= 3" in expl
    assert "at most 6 times" in expl                                     # the history-call cap is in the prompt
    rep = ide.brief_text(ide.brief_plan("explore", v, "repair"), led, v).text
    assert "3-5 DIRECTIONS" in rep and ">= 2" in rep
    lit = ide.brief_text(ide.brief_plan("explore", v, "literature"), led, v).text
    assert "6-8 DIRECTIONS" in lit
    bold = ide.brief_text(ide.brief_plan("explore", v, "bold"), led, v).text
    assert "JOINT PORTFOLIO" not in bold and "8-12" in bold
    off = ID.Ideation(_with(ctx, portfolio=False), store)
    old = off.brief_text(off.brief_plan("explore", off.view(led), "explore"), led, off.view(led)).text
    assert "JOINT PORTFOLIO" not in old and "8-12" in old
    assert ide.k_for("repair") == 0 and ide.k_for("kuhn") > 0                            # a slate carries the predictions


def test_the_prompts_tell_researchers_to_use_history_first():
    from loop.agents import briefs
    from pathlib import Path
    root = Path(briefs.__file__).resolve().parents[1] / "prompts"
    for name in ("researcher.md", "researcher_bold.md"):
        text = (root / name).read_text()
        assert "`history`" in text and "history" in text.split("## How to work")[1][:900]


def test_portfolio_kinds_get_sixty_turns_and_the_prompt_states_the_history_cap(penv):
    ctx, store, led, ide = penv
    from loop.contracts import CandidateRequest, Operator, Strategy
    me = SimpleNamespace(ctx=ctx)
    mk = lambda kind, strat=Strategy.EXPLORE: CandidateRequest(      # noqa: E731
        id="req-v3-00001", task_id=ctx.task_id, strategy=strat, operator=Operator.DRAFT, island="main",
        parent_ids=("root",), brief="b", brief_kind=kind)
    assert ctx.cfg.researcher.max_turns == 48 and ctx.cfg.researcher.portfolio_max_turns == 60
    assert RS.LLMResearcher.limits(me, mk("explore"))[0] == 60 and RS.LLMResearcher.limits(me, mk("literature"))[0] == 60
    assert RS.LLMResearcher.limits(me, mk("readout"))[0] == 48 and RS.LLMResearcher.limits(me, mk("bold"))[0] == 48
    assert RS.LLMResearcher.limits(me, mk("explore", Strategy.MOONSHOT))[0] == ctx.cfg.researcher.moonshot_max_turns
    assert RS.LLMResearcher.limits(SimpleNamespace(ctx=_with(ctx, portfolio=False)), mk("explore"))[0] == 48


# ---------------------------------------------------------------------------------------------- 11-08: a thin history
class ThinTab(Tab):
    """The nanobrain ledger of 11-08: root + 4 nodes, all rejected at smoke, none clustered (0 clusters)."""

    def __init__(self, nodes):
        super().__init__(nodes)
        self.cluster_ids = {}


THIN = ThinTab({"root": "", "n0001": "", "n0002": "", "n0003": "", "n0004": ""})
EMPTY = ThinTab({})


def _thin_slate(n=7, ids=("root", "n0001", "n0002", "n0003", "n0004")):
    out = []
    for i, c in enumerate(_slate(n)):
        g = ids[i % len(ids)]
        out.append({**c, "family": f"other{i}", "grounded_in": g,
                    "differs_from": [{"id": g, "how": "pools the patch tokens and gates them with an entropy mask"}]})
    return out


def test_a_history_without_clusters_degrades_the_diversity_rule_and_unlocks_edits(penv, tmp_path):
    """11-08 deadlock: '>= 3 DISTINCT history clusters' on a history with 0 clusters rejected every slate (1207 sessions,
    edits locked forever). With fewer clusters than the rule asks it asks for what exists; node ids ground; a
    citation of any existing node counts as related. The session can then edit and finish."""
    ctx, store, led, ide = penv
    assert BOLD.table_clusters(THIN) == set() and BOLD.thin_history(THIN, 3) and not BOLD.thin_history(TAB, 3)
    g = _gate(ctx, store, table=THIN)
    out = g(_thin_slate())
    assert "BUILD candidate #" in out and g.done and g.tries == 1, out
    t = _tools(tmp_path, g)
    assert not t.slate_refusal("set_config", {"key": "plot_period", "value": 0})   # edits are unlocked
    made_up = [{**c, "grounded_in": f"C{i + 1}"} for i, c in enumerate(_thin_slate())]
    rej = _gate(ctx, store, table=THIN)(made_up)                    # invented cluster ids are still refused, with a hint
    assert "ids not in the history table" in rej and "ground in node ids" in rej


def test_one_cluster_asks_for_one_and_three_or_more_keep_the_rule(penv):
    ctx, store, led, ide = penv
    one = Tab({"root": "", "n0001": "C1", "n0002": "", "n0003": ""})
    one.cluster_ids = {"C1": one.cluster_ids["C1"]}
    g3 = ["C1", "n0002", "root"]
    s = [{**c, "grounded_in": g3[i % 3], "differs_from": [{"id": g3[i % 3], "how": "pools the patch tokens and gates "
                                                           "them with an entropy mask"}]} for i, c in enumerate(_slate(7))]
    assert "BUILD candidate" in _gate(ctx, store, table=one)(s)
    none = [{**c, "grounded_in": "n0002"} for c in s]               # 1 cluster exists: 0 covered is still short
    assert "DISTINCT history clusters (yours cover 0" in _gate(ctx, store, table=one)(none)
    # nanopath parity: >= 3 clusters in the table -> the 3-cluster rule is unchanged (TAB has 5)
    two = [{**c, "grounded_in": "C1" if i % 2 else "C2"} for i, c in enumerate(_slate(7))]
    assert "DISTINCT history clusters (yours cover 2" in _gate(ctx, store)(two)


def test_an_empty_history_needs_no_grounding_or_citation(penv):
    ctx, store, led, ide = penv
    bare = [{k: v for k, v in c.items() if k not in ("grounded_in", "differs_from")} for c in _slate(7)]
    g = _gate(ctx, store, table=EMPTY)
    assert "BUILD candidate" in g(bare) and g.done
