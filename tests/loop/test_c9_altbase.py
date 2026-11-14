"""v3-c9 C2: non-incumbent starting points (loop.core.altbase) -- the quota draw, the sound-parent rules, the three
sources (a non-saturated cluster outside the incumbent's lineage, the previous epoch's best in its port, the top posterior
node outside the incumbent's ancestry), the credit rule (own increment over the CHOSEN parent), the planner / request /
intake wiring, fork-and-flush branches and the meta report's arm-vs-control section."""

from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace

import pytest

from loop.contracts import NodeStatus, Operator, ResearchResult, Strategy
from loop.core import alloc as AL
from loop.core import altbase as AB
from loop.core import data as D
from loop.core import fnf as FN
from loop.core import ideation as ID
from loop.core import model as M
from loop.spec import IdeationCfg
from loop.store import SqliteStore

from .core.conftest import TASK, synth
from .core.test_ideation import README, _cand
from .test_tick import _ledger


def _ctx(ctx=None, **kw):
    from loop.spec import load_context
    import tempfile
    if ctx is None:
        ctx = load_context("tasks/fake_loop.yaml", state_dir=tempfile.mkdtemp())
    base = ctx.cfg.ideation or IdeationCfg()
    kw.setdefault("alt_base_root", False)       # the root source has its own tests (below); the older ones pin the three
    return dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, ideation=dataclasses.replace(base, **kw)))


# ---------------------------------------------------------------------------------------------- the toy world
class MemStore:
    def __init__(self, status=None):
        self.state, self.logged, self.status = {}, [], status or {}

    def get_state(self, k, default=None):
        return self.state.get(k, default)

    def set_state(self, k, v):
        self.state[k] = json.loads(json.dumps(v))

    def log(self, event, **f):
        self.logged.append((event, f))

    def get_node(self, nid):
        if nid not in self.status and nid != "missing":
            self.status[nid] = NodeStatus.SCREENED_OUT
        return None if nid == "missing" else SimpleNamespace(
            id=nid, commit=f"c-{nid}", status=self.status[nid], family="fam", mechanism_tag=f"tag_{nid}")


def _world(spec, *, bases=(("root", "OLD"), ("r2", "NEW")), inc="inc"):
    """spec: id -> dict(parent, base, m, sd=0.002, n=4, port_of='', notes='')."""
    nodes = {}
    for i, (nid, s) in enumerate(spec.items()):
        nodes[nid] = D.Node(nid, s.get("parent"), None, "explore", "fam", "explore", "root", float(i), "screening",
                            s.get("notes", ""), s["base"], s.get("port_of", ""))
    led = D.Ledger(nodes, (), (), frozenset(), ((0.0, "root"),), bases=tuple((float(i), n, b) for i, (n, b) in enumerate(bases)))
    beliefs = {nid: ID.Belief(s["m"], s.get("sd", 0.002) ** 2, s.get("n", 4)) for nid, s in spec.items() if "m" in s}
    return SimpleNamespace(led=led, beliefs=beliefs, post=None, judged=(), incumbent=inc, weights={})


def _clusters(monkeypatch, groups):
    cls = [AL.Cluster(cid, list(mem), {}, label=cid.lower(), category="objective", flag=flag)
           for cid, (mem, flag) in groups.items()]
    monkeypatch.setattr(AL, "clusters", lambda *a, **k: cls)
    return cls


SPEC = {
    "root": dict(parent=None, base="OLD"),
    "r2": dict(parent=None, base="NEW"),
    "b0": dict(parent="r2", base="NEW", m=0.004),                      # the incumbent's parent
    "inc": dict(parent="b0", base="NEW", m=0.003),
    "c1": dict(parent="r2", base="NEW", m=0.006),                     # best of cluster CA
    "c1b": dict(parent="r2", base="NEW", m=0.002),
    "c2": dict(parent="r2", base="NEW", m=0.020),                      # in a SATURATED cluster
    "c3": dict(parent="r2", base="NEW", m=0.015),                      # in a cluster of the incumbent's lineage
    "t1": dict(parent="r2", base="NEW", m=0.0045),                     # not clustered
    "few": dict(parent="r2", base="NEW", m=0.03, n=2),                 # 2 seeds
    "neg": dict(parent="r2", base="NEW", m=-0.01, sd=0.002),           # clearly negative
    "edge": dict(parent="r2", base="NEW", m=-0.002, sd=0.002),         # exactly -1 sd: kept
    "dead": dict(parent="r2", base="NEW", m=0.05),                     # store VOID
    "failed": dict(parent="r2", base="NEW", m=0.05),                   # derived FAILED
    "o1": dict(parent="root", base="OLD", m=0.010),                    # previous epoch, ported
    "o1@": dict(parent="r2", base="NEW", m=0.009, port_of="o1"),
    "o2": dict(parent="root", base="OLD", m=0.020),                    # previous epoch, NOT ported: skipped
    "wave": dict(parent="inc", base="NEW", m=0.04, notes="wave=w1 factorial_of=w1"),
}
GROUPS = {"CA": (["c1", "c1b"], ""), "CB": (["c2"], "saturated-negative"), "CC": (["b0", "c3"], ""),
          "CD": (["inc"], "")}


@pytest.fixture
def world(monkeypatch):
    ctx = _ctx()
    store = MemStore({"dead": NodeStatus.VOID})
    v = _world(SPEC)
    _clusters(monkeypatch, GROUPS)
    return ctx, store, v, {"failed": SimpleNamespace(status=SimpleNamespace(name="FAILED"))}


# ---------------------------------------------------------------------------------------------- the quota
def test_the_draw_is_random_like_converges_to_the_share_and_is_off_at_zero():
    hits = sum(AB.due(n, 0.2) for n in range(5000))
    assert 0.17 < hits / 5000 < 0.23
    assert [AB.due(n, 0.2) for n in range(40)] == [AB.due(n, 0.2) for n in range(40)]       # deterministic
    runs = [AB.due(n, 0.2) for n in range(200)]
    assert not all(runs[i] == runs[i + 5] for i in range(190))                              # not a fixed period
    assert not any(AB.due(n, 0.0) for n in range(100)) and all(AB.due(n, 1.0) for n in range(100))


def test_booking_a_plan_counts_the_session_the_alt_and_the_use_and_rolls_back():
    st = MemStore()
    alt = {"node": "c1", "source": "cluster"}
    AB.count(st, {"alt": alt, "alt_eligible": True})
    AB.count(st, {"alt": None, "alt_eligible": True})
    s = AB.state(st)
    assert (s["n"], s["alt"], s["uses"]) == (2, 1, {"c1": 1})
    AB.count(st, {"alt": alt, "alt_eligible": True}, -1)
    s = AB.state(st)
    assert (s["n"], s["alt"], s["uses"]) == (1, 0, {"c1": 0})
    AB.count(st, {"alt": alt, "branch": "1.4"})                       # a fork-and-flush branch: no quota draw, a use
    assert (AB.state(st)["n"], AB.state(st)["uses"]["c1"]) == (1, 1)
    AB.count(st, {"alt": None, "branch": "1.1"})                       # nothing to book
    assert AB.state(st)["n"] == 1


# ---------------------------------------------------------------------------------------------- sound parents
def test_a_parent_needs_paired_seeds_and_a_not_clearly_negative_gain():
    v = _world({"a": dict(parent=None, base="X", m=0.0, n=3), "b": dict(parent=None, base="X", m=-0.002, sd=0.002, n=5),
                "c": dict(parent=None, base="X", m=-0.0021, sd=0.002, n=5), "d": dict(parent=None, base="X", m=0.01, n=2)})
    assert AB.standing(v, "a", 3, 1.0) == (0.0, 0.002, 3)
    assert AB.standing(v, "b", 3, 1.0) is not None                 # exactly -1 sd: within 1 sd of 0
    assert AB.standing(v, "c", 3, 1.0) is None                     # beyond 1 sd below 0
    assert AB.standing(v, "c", 3, 2.0) is not None                  # the z is config
    assert AB.standing(v, "d", 3, 1.0) is None and AB.standing(v, "d", 2, 1.0) is not None
    assert AB.standing(v, "nope", 3, 1.0) is None
    # PAIRED seeds: the CUPED estimate's count, not the seed count
    est = {"a": {M.COMPOSITE: M.Estimate(0.0, 1e-6, 5, 1)}}
    v2 = SimpleNamespace(**{**v.__dict__, "beliefs": {"a": ID.Belief(0.01, 1e-6, 5)}, "post": SimpleNamespace(est=est)})
    assert AB.standing(v2, "a", 3, 1.0) is None


def test_sound_means_a_commit_and_no_void_crashed_vetoed_rejected_or_failed_status():
    st = MemStore({"v": NodeStatus.VOID, "x": NodeStatus.CRASHED, "z": NodeStatus.VETOED, "r": NodeStatus.REJECTED})
    for bad in "vxzr":
        assert not AB.sound(st, None, bad), bad
    assert AB.sound(st, None, "ok") and not AB.sound(st, None, "missing")
    assert not AB.sound(st, {"ok": SimpleNamespace(status=SimpleNamespace(name="FAILED"))}, "ok")
    assert not AB.sound(st, {"ok": SimpleNamespace(status=SimpleNamespace(name="CRASHED"))}, "ok")
    assert AB.sound(st, {"ok": SimpleNamespace(status=SimpleNamespace(name="PASSED"))}, "ok")


# ---------------------------------------------------------------------------------------------- the sources
def test_cluster_source_skips_saturated_and_lineage_clusters_and_unsound_nodes(world):
    ctx, store, v, statuses = world
    c = AB.candidates(ctx, store, v, statuses)["cluster"]
    assert [p.node for p in c] == ["c1"]                   # CB saturated, CC holds the incumbent's parent, CD the incumbent
    assert c[0].cluster == "CA" and c[0].source == "cluster" and "not saturated" in c[0].why and "CC" in c[0].why


def test_top_outside_is_the_best_sound_node_outside_the_ancestry_without_waves_or_ports(world):
    ctx, store, v, statuses = world
    top = AB.candidates(ctx, store, v, statuses)["top_outside"]
    ids = [p.node for p in top]
    assert ids[:4] == ["c2", "c3", "c1", "t1"] and "edge" in ids
    for gone in ("inc", "b0", "few", "neg", "dead", "failed", "wave", "o1@", "root", "r2"):
        assert gone not in ids, gone
    assert ids == sorted(ids, key=lambda i: (-v.beliefs[i].m, i))


def test_previous_epoch_source_uses_the_port_never_the_old_node_and_skips_the_unported(world):
    ctx, store, v, statuses = world
    prev = AB.candidates(ctx, store, v, statuses)["prev_epoch"]
    assert [p.node for p in prev] == ["o1@"] and prev[0].via == "o1" and "ported" in prev[0].why
    assert prev[0].m == pytest.approx(0.009) and prev[0].rank == pytest.approx(0.010)       # ranked by the old node's gain
    v1 = _world({k: s for k, s in SPEC.items() if k not in ("o1", "o1@", "o2")}, bases=(("r2", "NEW"),))
    assert AB.candidates(ctx, store, v1, statuses)["prev_epoch"] == []                       # one epoch: no previous


def test_the_incumbents_ancestry_and_a_lineage_cluster_never_provide_a_cluster_or_outside_pick(world):
    ctx, store, v, statuses = world
    got = AB.candidates(ctx, store, v, statuses)
    assert "b0" not in {p.node for ps in got.values() for p in ps} and "inc" not in {p.node for ps in got.values() for p in ps}


# ---------------------------------------------------------------------------------------------- choosing
def test_sources_take_turns_and_a_used_up_base_is_skipped(world):
    ctx, store, v, statuses = world
    cands = AB.candidates(ctx, store, v, statuses)
    order = [AB.choose(ctx, store, v, statuses, t, cands=cands) for t in range(3)]
    assert [p.source for p in order] == ["cluster", "prev_epoch", "top_outside"]
    assert AB.choose(ctx, store, v, statuses, 3, cands=cands).source == "cluster"            # the cycle repeats
    store.set_state(AB.K_ALT, {"n": 9, "alt": 3, "uses": {"c1": 3}})                           # alt_base_max_uses = 3
    nxt = AB.choose(ctx, store, v, statuses, 0, cands=cands)
    assert nxt.source == "prev_epoch"                                                         # the cluster source is dry
    assert AB.choose(ctx, store, v, statuses, 2, exclude=["c3", "c1", "t1", "edge"], cands=cands).source != "top_outside"
    none = {s: [] for s in AB.SOURCES}
    assert AB.choose(ctx, store, v, statuses, 0, cands=none) is None


def test_fnf_alt_branches_are_the_last_k():
    rnd = {"branches": ["1.1", "1.2", "1.3", "1.4"]}
    assert AB.fnf_branches(rnd, 1) == ["1.4"] and AB.fnf_branches(rnd, 2) == ["1.3", "1.4"]
    assert AB.fnf_branches(rnd, 0) == [] and AB.fnf_branches(rnd, 9) == rnd["branches"]


def test_the_note_names_the_base_the_credit_rule_and_the_incumbent():
    p = AB.Pick("c1", "cluster", 0.006, 0.002, 4, "the best node of cluster CA")
    t = AB.note(p, "inc", "fam/tag")
    assert "builds on **c1**" in t and "NOT on the incumbent inc" in t and "OWN paired increment over c1" in t
    assert "never c1's gain" in t and "bar" in t


# ---------------------------------------------------------------------------------------------- credit
def test_a_child_is_credited_with_its_own_increment_over_the_chosen_parent_never_the_parents_gain():
    """Parent P is +0.010 over the base; its child C is +0.0105: C's own increment is +0.0005, not +0.0105."""
    import random
    from loop.core.acquire import Posterior
    rng = random.Random(3)
    mu = {"root": 0.0, "P": 0.010, "C": 0.0105, "S": 0.0}
    seeds = {n: [1, 2, 3, 4, 5, 6] for n in mu}
    obs = synth(rng, mu, seeds, sigma=0.002, rho=0.5)
    nodes = {"root": D.Node("root", None, None, "", "", "explore", "root", 0.0, "", "", "B"),
             "P": D.Node("P", "root", None, "explore", "f", "explore", "root", 1.0, "", "", "B"),
             "C": D.Node("C", "P", None, "explore", "f", "explore", "root", 2.0, "", "arm=alt_base alt_of=P", "B"),
             "S": D.Node("S", "root", None, "explore", "f", "explore", "root", 3.0, "", "", "B")}
    led = D.Ledger(nodes, (), tuple(obs), frozenset(), ((0.0, "root"),), bases=((0.0, "root", "B"),))
    post = Posterior(led, [])
    m = {n: b[0] for n, b in post.beliefs().items()}
    assert abs(m["C"]) < 0.004 and m["P"] > 0.004                       # C ~ +0.0005 (not P's +0.01)


# ---------------------------------------------------------------------------------------------- the planner
@pytest.fixture
def env(ctx, tmp_path):
    db = _ledger(tmp_path / "ledger.db")
    ctx = _ctx(dataclasses.replace(ctx, task_id=TASK), alt_base_require_ckpt=False,
              alt_base_gain_z=1e9)     # the toy runs left no checkpoint; its picks are all far below the incumbent
    store = SqliteStore(db, TASK)
    (readme := tmp_path / "README.md").write_text(README)
    yield ctx, store, D.load(db, TASK), ID.Ideation(ctx, store, readme=readme)
    store.close()


def test_an_alt_draw_moves_the_session_off_the_incumbent_and_books_the_quota(env, monkeypatch):
    ctx, store, led, ide = env
    v = ide.view(led)
    monkeypatch.setattr(AB, "due", lambda n, share: True)
    plan = ide.brief_plan("explore", v, "explore")
    assert plan["alt_eligible"] and plan["alt"] and plan["parents"] == (plan["alt"]["node"],) != (v.incumbent,)
    assert plan["head"].startswith("STARTING POINT (measured arm `alt_base`)")
    assert plan["alt"]["node"] not in ID.lineage(v.led, v.incumbent)
    assert AB.state(store)["n"] == 0                                         # planning has no side effect ...
    ide._reserve(plan)                                                       # ... booking does
    s = AB.state(store)
    assert (s["n"], s["alt"], s["uses"][plan["alt"]["node"]]) == (1, 1, 1) and store.events("alt_base_pick")
    ide._unreserve(plan)
    s = AB.state(store)
    assert (s["n"], s["alt"], s["uses"][plan["alt"]["node"]]) == (0, 0, 0)


def test_only_the_configured_kinds_and_a_positive_share_draw_alt_bases(env, monkeypatch):
    ctx, store, led, ide = env
    v = ide.view(led)
    monkeypatch.setattr(AB, "due", lambda n, share: True)
    for kind in ("repair", "literature", "crossover", "program"):
        plan = ide.brief_plan("explore", v, kind)
        assert plan["alt"] is None and not plan["alt_eligible"] or plan["kind"] == "explore", kind
    off = ID.Ideation(_ctx(ctx, alt_base_share=0.0), store, readme=None)
    p = off.brief_plan("explore", off.view(led), "explore")
    assert p["alt"] is None and p["parents"] == (v.incumbent,) and not p["alt_eligible"]
    no_draw = ID.Ideation(ctx, store)
    monkeypatch.setattr(AB, "due", lambda n, share: False)
    p = no_draw.brief_plan("explore", no_draw.view(led), "explore")
    assert p["alt"] is None and p["alt_eligible"] and p["parents"] == (v.incumbent,)


def test_the_alt_brief_request_and_node_carry_the_arm(env, monkeypatch):
    ctx, store, led, ide = env
    v = ide.view(led)
    monkeypatch.setattr(AB, "due", lambda n, share: True)
    plan = ide.brief_plan("explore", v, "explore")
    b = ide.brief_text(plan, led, v)
    assert b.arm.startswith(f"alt_base:{plan['alt']['source']}") and b.parents == plan["parents"]
    assert "STARTING POINT" in b.text and "Your worktree starts from" in b.text
    req = ide.request(b, v)
    assert req.arm == b.arm and req.parent_ids == (plan["alt"]["node"],)
    out = ide.intake(req, ResearchResult("ok", _cand("an idea built away from the incumbent", commit="cALT")))
    node = store.get_node(out.split()[1])
    assert node.parent_id == plan["alt"]["node"]
    assert " arm=alt_base " in f" {node.notes} " and f"alt_of={plan['alt']['node']}" in node.notes
    assert f"alt_src={plan['alt']['source']}" in node.notes
    # alt_dist= needs the ledger sample's diff files, which are not shipped (the sample's diff_path values are scrubbed)
    # an incumbent-based session carries no arm tag
    monkeypatch.setattr(AB, "due", lambda n, share: False)
    plain = ide.request(ide.brief_text(ide.brief_plan("explore", v, "explore"), led, v), v)
    assert plain.arm == "" and plain.parent_ids == (v.incumbent,)
    out2 = ide.intake(plain, ResearchResult("ok", _cand("another idea on the incumbent itself", commit="cINC")))
    assert "arm=alt_base" not in store.get_node(out2.split()[1]).notes


def test_a_bold_alt_session_says_so_in_the_protected_incumbent_section(env, monkeypatch):
    ctx, store, led, ide = env
    v = ide.view(led)
    ide2 = ID.Ideation(_ctx(ctx, alt_base_kinds=("explore", "bold")), store)
    monkeypatch.setattr(AB, "due", lambda n, share: True)
    plan = ide2.brief_plan("explore", v, "bold")
    assert plan["alt"] and plan["kind"] == "bold"
    b = ide2.bold_brief(plan, led, v)
    assert "This session builds on " + plan["parents"][0] + " instead" in b.text and b.parents == plan["parents"]
    assert b.arm.startswith("alt_base:")


def test_the_alt_pick_is_stable_per_view_and_a_failure_falls_back_to_the_incumbent(env, monkeypatch):
    ctx, store, led, ide = env
    v = ide.view(led)
    monkeypatch.setattr(AB, "due", lambda n, share: True)
    assert ide._alt_cands(v) is ide._alt_cands(v)                              # once per view token
    monkeypatch.setattr(AB, "choose", lambda *a, **k: 1 / 0)
    plan = ide.brief_plan("explore", v, "explore")
    assert plan["alt"] is None and plan["parents"] == (v.incumbent,) and store.events("alt_base_error")


def test_a_replan_keeps_the_quota_consistent(env, monkeypatch):
    ctx, store, led, ide = env
    v = ide.view(led)
    monkeypatch.setattr(AB, "due", lambda n, share: True)
    plan = ide.brief_plan("explore", v, "explore")
    ide._reserve(plan)
    ide._count_alt(plan, -1)
    new = {**plan, "alt": None}
    ide._count_alt(new)
    s = AB.state(store)
    assert (s["n"], s["alt"]) == (1, 0) and s["uses"][plan["alt"]["node"]] == 0


# ---------------------------------------------------------------------------------------------- fork-and-flush
class _Research:
    def __init__(self, free=4):
        self.n, self.started, self.logs, self.researcher = free, [], [], SimpleNamespace(llm=None)

    def free(self):
        return self.n - len(self.started)

    def start(self, req, *args, **log):
        self.started.append(req)
        self.logs.append(log)


def test_one_of_k_branches_starts_from_an_alt_base_fixed_for_the_round(env, monkeypatch):
    ctx, store, led, ide = env
    object.__setattr__(ctx.cfg.ideation, "kind_weight", False)
    v = ide.view(led)
    st = FN.state(store)
    st.update(round=1, open={"id": 1, "ts": 50.0, "gpu0": 0.0, "cap": 21.0, "branches": ["1.1", "1.2", "1.3", "1.4"],
                             "incumbent": v.incumbent, "trigger": "start", "sessions": {}})
    store.set_state(FN.K_FNF, st)
    plans = {b: ide._plan("explore", v, "explore", b) for b in ("1.1", "1.2", "1.3", "1.4")}
    assert [b for b, p in plans.items() if p["alt"]] == ["1.4"]                                   # default alt_base_fnf = 1
    base = plans["1.4"]["alt"]["node"]
    assert plans["1.4"]["parents"] == (base,) and all(plans[b]["parents"] == (v.incumbent,) for b in ("1.1", "1.2", "1.3"))
    assert ide._plan("explore", v, "explore", "1.4")["alt"]["node"] == base                       # stable within the round
    b = ide.branch_brief("explore", v, "explore", "1.4", plans["1.4"])
    assert b.parents == (base,) and b.arm.startswith("alt_base:") and "STARTING POINT" in b.text
    assert f"This branch starts from {base} instead" in b.text
    same = ide.branch_brief("explore", v, "explore", "1.1", plans["1.1"])
    assert same.parents == (v.incumbent,) and same.arm == "" and "STARTING POINT" not in same.text
    # off: alt_base_fnf = 0
    off = ID.Ideation(_ctx(ctx, alt_base_fnf=0), store)
    assert off._plan("explore", v, "explore", "1.4")["alt"] is None
    # brief_for builds the same plan
    assert ide.brief_for("explore", led, v, branch="1.4").parents == (base,)


# ---------------------------------------------------------------------------------------------- the meta report
def test_the_meta_section_compares_the_arm_with_incumbent_based_sessions(env):
    ctx, store, led, ide = env
    v = ide.view(led)
    nodes = dict(led.nodes)
    tagged = [n for n, nd in nodes.items() if "request=" in nd.notes and "brief=" in nd.notes][:6]
    for i, n in enumerate(tagged[:3]):
        nodes[n] = dataclasses.replace(nodes[n], notes=nodes[n].notes + f" arm=alt_base alt_of=n0165 alt_src={'cluster' if i else 'prev_epoch'}")
    for n in tagged[3:]:
        nodes[n] = dataclasses.replace(nodes[n], notes=nodes[n].notes.replace("brief=literature", "brief=explore"))
    v2 = dataclasses.replace(v, led=dataclasses.replace(led, nodes=nodes))
    sec = AB.section(ctx, v2.led, store, v2)
    assert sec["alt_base"]["nodes"] == 3 and sec["incumbent"]["nodes"] == 3
    assert sec["sources"] == {"cluster": 2, "prev_epoch": 1}
    for arm in ("alt_base", "incumbent"):
        assert {"tested", "gpu_h", "credit_per_gpu_h", "screen_credit_per_gpu_h", "measured", "mean_m"} <= set(sec[arm])
    txt = "\n".join(AB.report_text(sec))
    assert "alt_base: 3 nodes" in txt and "incumbent: 3 nodes" in txt and "credit/GPU-h" in txt
    assert AB.report_text({}) == ["(no alt-base node yet)"]


def test_the_meta_report_renders_the_alt_base_section(env):
    from loop.core import meta as MT
    ctx, store, led, ide = env
    rep, _ = MT.report(led, store, now=1e9, since=0.0)
    assert "alt_base" in rep.__dict__ and (not rep.alt_base or "alt_base" in rep.alt_base or "alt_base" not in rep.errors)
    rep.alt_base = {"alt_base": {"nodes": 2, "measured": 2, "positive": 1, "mean_m": 0.001, "tested": 1, "gpu_h": 4.0,
                                 "credit_per_gpu_h": 1e-4, "screen_credit_per_gpu_h": 2e-4},
                    "incumbent": {"nodes": 9, "measured": 8, "positive": 3, "mean_m": -0.001, "tested": 2, "gpu_h": 30.0,
                                  "credit_per_gpu_h": 5e-5, "screen_credit_per_gpu_h": 6e-5}, "sources": {"cluster": 2}}
    text = MT.render_markdown(rep)
    assert "## Alt-base arm" in text and "arm - control credit/GPU-h" in text



# ---------------------------------------------------------------------------------------------- review fixes M1 / M2
class DiffStore(MemStore):
    """MemStore with per-node diffs (real files) so the code-distinctness checks run."""

    def __init__(self, tmp_path, diffs, status=None, extra=None):
        super().__init__(status)
        self.tmp, self.diffs, self.extra = tmp_path, diffs, extra or {}

    def get_node(self, nid):
        sn = super().get_node(nid)
        if sn is None:
            return None
        p = self.tmp / f"{nid}.diff"
        if nid in self.diffs and not p.exists():
            p.write_text("\n".join(f"+{ln}" for ln in self.diffs[nid]))
        for k, x in self.extra.get(nid, {}).items():
            setattr(sn, k, x)
        sn.diff_path = str(p) if nid in self.diffs else ""
        return sn


def _L(tag, k):
    return [f"{tag}_statement_number_{i} = compute({i})" for i in range(k)]


M_SPEC = {
    "root": dict(parent=None, base="OLD"),
    "r2": dict(parent=None, base="NEW"),
    "b0": dict(parent="r2", base="NEW", m=0.004),
    "inc": dict(parent="b0", base="NEW", m=0.003),
    "kid": dict(parent="inc", base="NEW", m=0.02),               # a direct child of the incumbent
    "sib": dict(parent="b0", base="NEW", m=0.019),               # a child of a non-root ancestor
    "x2p": dict(parent="r2", base="NEW", m=0.018),               # the incumbent's SECOND parent's branch (crossover)
    "x2": dict(parent="r2", base="NEW", m=0.017),
    "contained": dict(parent="r2", base="NEW", m=0.016),         # 98% of its own diff is in the incumbent's
    "close": dict(parent="r2", base="NEW", m=0.015),             # cumulative diff 95% the incumbent's (crossed with it)
    "far": dict(parent="r2", base="NEW", m=0.014),               # its own code
    "o1": dict(parent="root", base="OLD", m=0.010),
    "o1@": dict(parent="r2", base="NEW", m=0.009, port_of="o1"),   # previous epoch's best, ported -- but in the lineage
}


def _mworld(tmp_path, monkeypatch, status=None, extra=None, **cfg):
    spec = {k: dict(v) for k, v in M_SPEC.items()}
    spec["inc"]["second_parent"] = "x2"
    spec["close"]["second_parent"] = "inc"                            # crosses the incumbent with a small change of its own
    v = _world(spec)
    nodes = {k: (dataclasses.replace(nd, second_parent=spec[k].get("second_parent")) if spec[k].get("second_parent") else nd)
             for k, nd in v.led.nodes.items()}
    v = SimpleNamespace(**{**v.__dict__, "led": dataclasses.replace(v.led, nodes=nodes)})
    inc_code = _L("b", 30) + _L("x", 30)
    diffs = {"b0": _L("b", 30), "inc": [], "x2": _L("x", 30), "kid": _L("kid", 2),
             "sib": _L("sib", 3), "contained": _L("b", 29) + _L("new", 1) + [], "close": _L("z", 3),
             "far": _L("far", 30), "o1@": _L("b", 30)[:5] + _L("o", 20)}
    diffs["x2p"] = _L("p", 4)
    store = DiffStore(tmp_path, diffs, status, extra)
    _clusters(monkeypatch, {"CA": (["kid", "sib", "contained", "close", "far", "x2p"], "")})
    return _ctx(alt_base_require_ckpt=False, alt_base_gain_z=1e9, **cfg), store, v


def test_the_second_parent_is_in_the_ancestry_and_a_port_of_it_is_never_a_prev_epoch_pick(tmp_path, monkeypatch):
    ctx, store, v = _mworld(tmp_path, monkeypatch)
    assert {"inc", "b0", "x2", "r2"} <= AB.ancestry(v.led, v.incumbent)
    assert "x2" not in ID.lineage(v.led, v.incumbent) and "x2" in ID.lineage(v.led, v.incumbent, second=True)
    v.led.nodes["o1"] = dataclasses.replace(v.led.nodes["o1"])
    # make the incumbent's second parent the PORT itself: it is in the ancestry, so it can no longer be picked
    nodes = dict(v.led.nodes)
    nodes["inc"] = dataclasses.replace(nodes["inc"], second_parent="o1@")
    v2 = SimpleNamespace(**{**v.__dict__, "led": dataclasses.replace(v.led, nodes=nodes)})
    assert "o1@" in AB.ancestry(v2.led, "inc")
    assert AB.candidates(ctx, store, v2)["prev_epoch"] == []


def test_a_child_of_the_incumbent_or_of_its_ancestors_is_not_a_diverse_base(tmp_path, monkeypatch):
    ctx, store, v = _mworld(tmp_path, monkeypatch)
    got = AB.candidates(ctx, store, v)
    ids = {p.node for ps in got.values() for p in ps}
    assert "kid" not in ids and "sib" not in ids and "inc" not in ids
    assert "far" in ids                                              # a real branch off the base root stays


def test_a_diff_contained_in_the_incumbents_or_too_close_to_it_is_excluded(tmp_path, monkeypatch):
    ctx, store, v = _mworld(tmp_path, monkeypatch)
    lines = AB.Lines(store, v.led)
    inc = lines.cumulative("inc")
    assert lines.contained("contained", inc) > 0.8 and lines.contained("far", inc) == 0.0
    d = AB.diverse(ctx, store, v, "far", AB.ancestry(v.led, "inc"), inc, lines)
    assert d is not None and d[0] > 0.9 and d[1] == 0.0
    assert AB.diverse(ctx, store, v, "contained", AB.ancestry(v.led, "inc"), inc, lines) is None
    assert lines.distance("close", inc) < 0.1
    assert AB.diverse(ctx, store, v, "close", AB.ancestry(v.led, "inc"), inc, lines) is None
    got = AB.candidates(ctx, store, v)
    ids = {p.node for ps in got.values() for p in ps}
    assert "contained" not in ids and "close" not in ids
    pick = next(p for p in got["top_outside"] if p.node == "far")
    assert pick.dist > 0.9 and pick.contain == 0.0
    assert f"{pick.dist:.2f}" in AB.note(pick, "inc")


def test_a_pick_clearly_worse_than_the_incumbent_is_skipped(tmp_path, monkeypatch):
    ctx, store, v = _mworld(tmp_path, monkeypatch)
    monkeypatch.setattr(AB, "gain_vs_incumbent", lambda v_, nid: (-0.0100, 0.0040) if nid == "far" else (-0.001, 0.002))
    ids = {p.node for ps in AB.candidates(_ctx(alt_base_require_ckpt=False), store, v).values() for p in ps}
    assert "far" not in ids                                          # -0.0100 < -1.28 x 0.0040
    monkeypatch.setattr(AB, "gain_vs_incumbent", lambda v_, nid: (-0.0100, 0.0090))
    ids = {p.node for ps in AB.candidates(_ctx(alt_base_require_ckpt=False), store, v).values() for p in ps}
    assert "far" in ids


def test_eval_only_parents_and_the_real_statuses_are_not_sound(tmp_path, monkeypatch):
    from loop.core.status import Status
    st = MemStore({"e1": NodeStatus.EVALUATING, "e2": NodeStatus.EVAL_PASSED, "a": NodeStatus.AUDIT_FAILED,
                   "n": NodeStatus.NOT_CONFIRMED, "ok": NodeStatus.SCREENED_OUT})
    for bad in ("e1", "e2", "a", "n"):
        assert not AB.sound(st, None, bad), bad
    assert AB.sound(st, None, "ok")
    ev = SimpleNamespace(id="ev", commit="c", status=NodeStatus.SCREENING, family="eval_readout", notes="")
    st.get_node = lambda nid: ev
    assert not AB.sound(st, None, "ev")                                # the eval-only family has no checkpoint of its own
    st2 = MemStore()
    for name in ("FAILED", "VOID", "CRASHED"):                         # the REAL derived statuses (loop.core.status.Status)
        assert not AB.sound(st2, {"ok": Status[name]}, "ok"), name
    for good in ("PASSED", "SCREENING", "TESTING", "QUEUED", "INCUMBENT"):
        assert AB.sound(st2, {"ok": Status[good]}, "ok"), good
    assert AB.sound(st2, {"ok": SimpleNamespace(status=Status.PASSED)}, "ok")


def test_the_commit_object_and_a_checkpoint_must_exist(tmp_path, monkeypatch):
    import subprocess
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    (repo / "f").write_text("x")
    subprocess.run(["git", "-C", str(repo), "add", "f"], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "m"], check=True)
    sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = tmp_path / "run"
    out.mkdir()
    ctx = _ctx(alt_base_require_ckpt=True)
    ctx = SimpleNamespace(cfg=ctx.cfg, spec=SimpleNamespace(repo=str(repo)))
    run = SimpleNamespace(node="n1", output_dir=str(out))
    led = SimpleNamespace(runs=[run])
    sn = {"n1": SimpleNamespace(commit=sha)}
    store = SimpleNamespace(get_node=lambda n: sn.get(n))
    assert not AB.buildable(ctx, store, led, "n1")                      # commit exists, but no checkpoint
    (out / "latest.pt").write_text("w")
    assert AB.buildable(ctx, store, led, "n1")
    sn["n1"] = SimpleNamespace(commit="0" * 40)
    assert not AB.buildable(ctx, store, led, "n1")                      # no such commit object
    assert AB.buildable(_ctx(alt_base_require_ckpt=False), store, led, "n1")


def test_no_diverse_candidate_falls_back_to_the_incumbent_and_says_so(env, monkeypatch):
    ctx, store, led, ide = env
    v = ide.view(led)
    monkeypatch.setattr(AB, "due", lambda n, share: True)
    monkeypatch.setattr(AB, "candidates", lambda *a, **k: {s: [] for s in AB.SOURCES})
    plan = ide.brief_plan("explore", v, "explore")
    assert plan["alt"] is None and plan["parents"] == (v.incumbent,) and plan["alt_eligible"]
    assert any(e.get("note") == "alt_base: no diverse candidate" for e in store.events("alt_base_none"))


def test_the_section_splits_credit_by_distance_bucket(env):
    ctx, store, led, ide = env
    v = ide.view(led)
    nodes = dict(led.nodes)
    tagged = [n for n, nd in nodes.items() if "request=" in nd.notes and "brief=" in nd.notes][:4]
    for n, d in zip(tagged, ("0.150", "0.450", "0.900", None)):
        nodes[n] = dataclasses.replace(nodes[n], notes=nodes[n].notes + " arm=alt_base alt_of=n0165 alt_src=cluster"
                                       + (f" alt_dist={d}" if d else ""))
    v2 = dataclasses.replace(v, led=dataclasses.replace(led, nodes=nodes))
    sec = AB.section(ctx, v2.led, store, v2)
    assert set(sec["by_distance"]) == {"near", "mid", "far", "unmeasured"}
    assert all(sec["by_distance"][b]["nodes"] == 1 for b in sec["by_distance"])
    txt = "\n".join(AB.report_text(sec))
    assert "distance near:" in txt and "distance far:" in txt


# ---------------------------------------------------------------------------------------------- the root source
ROOT_SPEC = {
    "r2": dict(parent=None, base="NEW"),
    "inc": dict(parent="r2", base="NEW", m=0.003),
}


def _rworld(monkeypatch, extra=None, **cfg):
    spec = {**ROOT_SPEC, **(extra or {})}
    v = _world(spec, bases=(("r2", "NEW"),))
    _clusters(monkeypatch, {})
    store = MemStore()
    return _ctx(alt_base_root=True, **cfg), store, v


def test_root_is_picked_when_the_other_sources_are_empty_and_is_far_and_exempt_from_the_ancestry_rules(monkeypatch):
    ctx, store, v = _rworld(monkeypatch)
    cands = AB.candidates(ctx, store, v, {})
    assert [p.node for p in cands["root"]] == ["r2"] and not any(cands[s] for s in AB.SOURCES)
    for turn in range(20):                                   # every alt draw: root, however often it was used
        store.set_state(AB.K_ALT, {"n": 9, "alt": turn, "uses": {"r2": 99}})
        got = AB.choose(ctx, store, v, {}, turn, cands=cands)
        assert got.node == "r2" and got.source == "root" and got.dist == 1.0
    assert ID._arm({"alt": {"source": "root", "dist": 1.0}}) == "alt_base:root:far"      # alt_dist=far
    # r2 is in the incumbent's ancestry -- exempt; the soundness checks still apply
    assert "r2" in AB.ancestry(v.led, "inc")
    store.status["r2"] = NodeStatus.VOID
    assert "root" not in AB.candidates(ctx, store, v, {}) or not AB.candidates(ctx, store, v, {})["root"]
    store.status["r2"] = NodeStatus.SCREENED_OUT
    assert not AB.candidates(_ctx(alt_base_root=False), store, v, {}).get("root")        # config off
    assert not AB.candidates(ctx, store, v, {"r2": SimpleNamespace(status=SimpleNamespace(name="FAILED"))}).get("root")


def test_root_needs_a_checkpoint_when_required(monkeypatch, tmp_path):
    ctx, store, v = _rworld(monkeypatch, alt_base_require_ckpt=True)
    led = SimpleNamespace(runs=[SimpleNamespace(node="r2", output_dir=str(tmp_path))])
    assert not AB.buildable(ctx, store, led, "r2")           # a run dir with no checkpoint
    (tmp_path / "latest.pt").write_text("x")
    assert AB.buildable(ctx, store, led, "r2")


def test_the_root_share_is_respected_when_other_sources_exist(monkeypatch):
    extra = {"t1": dict(parent="r2", base="NEW", m=0.0045)}
    for share, lo, hi in ((0.5, 0.4, 0.6), (0.2, 0.1, 0.3), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)):
        ctx, store, v = _rworld(monkeypatch, extra, alt_base_root_share=share, alt_base_max_uses=10 ** 6)
        cands = AB.candidates(ctx, store, v, {})
        assert cands["top_outside"] and cands["root"]
        picks = [AB.choose(ctx, store, v, {}, t, cands=cands) for t in range(400)]
        frac = sum(p.source == "root" for p in picks) / 400
        assert lo <= frac <= hi, (share, frac)
        assert all(p.source in ("root", "top_outside") for p in picks)


def test_the_root_brief_says_plain_base_increment_then_port_and_shows_the_incumbents_mechanisms():
    p = AB.Pick("r2", "root", 0.0, 0.0, 5, "why", dist=1.0)
    mech = ["- b0: fam/tagB", "- inc: fam/tagI"]
    t = AB.note(p, "inc", "fam/root", mech)
    assert "PLAIN BASE **r2**" in t and "NOT on the incumbent inc" in t and "FIRST as its own paired increment over r2" in t
    assert "ported onto inc afterwards" in t and "- b0: fam/tagB" in t and "- inc: fam/tagI" in t
    assert "Do NOT re-add a mechanism the incumbent already has, unless it is a deliberately different variant" in t
    assert "STARTING POINT" in t and "OWN paired increment over c1" not in t


def test_report_lists_root_separately():
    z = {"nodes": 0, "measured": 0, "positive": 0, "mean_m": None, "tested": 0, "gpu_h": 0.0, "credit_per_gpu_h": 0.0,
         "screen_credit_per_gpu_h": 0.0}
    r = {**z, "nodes": 2, "measured": 2, "positive": 1, "mean_m": 0.002}
    text = "\n".join(AB.report_text({"alt_base": z, "root": r, "incumbent": z, "sources": {"root": 2}}))
    assert "root: 2 nodes" in text and "(no alt-base node yet)" not in text
    assert AB.dist_of(SimpleNamespace(notes="arm=alt_base alt_src=root alt_dist=far")) == 1.0
    assert AB.bucket(SimpleNamespace(notes="alt_dist=far")) == "far"


def test_a_root_childs_credit_is_its_own_increment_over_root_not_over_the_incumbent():
    """I is +0.020 over root; C (built on root) is +0.010 over root: C's belief is +0.010; its bar gain is -0.010."""
    import random
    from loop.core.acquire import Posterior
    rng = random.Random(5)
    mu = {"root": 0.0, "I": 0.020, "C": 0.010}
    obs = synth(rng, mu, {n: [1, 2, 3, 4, 5, 6] for n in mu}, sigma=0.002, rho=0.5)
    nodes = {"root": D.Node("root", None, None, "", "", "explore", "root", 0.0, "", "", "B"),
             "I": D.Node("I", "root", None, "explore", "f", "explore", "root", 1.0, "", "", "B"),
             "C": D.Node("C", "root", None, "explore", "f", "explore", "root", 2.0, "",
                         "arm=alt_base alt_of=root alt_src=root alt_dist=far", "B")}
    led = D.Ledger(nodes, (), tuple(obs), frozenset(), ((0.0, "root"),), bases=((0.0, "root", "B"),))
    post = Posterior(led, [])
    m = {n: b[0] for n, b in post.beliefs().items()}
    assert 0.006 < m["C"] < 0.014                                        # vs root, never vs the incumbent
    g = AB.gain_vs_incumbent(SimpleNamespace(led=led, incumbent="I", post=post, weights=post.weights), "C")
    assert g is not None and -0.014 < g[0] < -0.006                       # the bar gain is vs the incumbent
    assert AB.arm_of(nodes["C"]) == "alt_base" and AB.tags(nodes["C"])["alt_src"] == "root"


# ---------------------------------------------------------------------------------------------- root -> incumbent ports
class FakePost:
    def __init__(self, b):                       # node -> (m, v, n, paired)
        self._b = b
        self.est = {n: {M.COMPOSITE: M.Estimate(0.0, x[1], x[2], x[3])} for n, x in b.items()}

    def beliefs(self):
        return {n: x[:3] for n, x in self._b.items()}


def _pworld(extra_nodes=None, ported=()):
    R = "arm=alt_base alt_of=r2 alt_src=root alt_dist=far"
    nodes = {"r2": D.Node("r2", None, None, "", "", "explore", "root", 0.0, "", "", "NEW"),
             "inc": D.Node("inc", "r2", None, "explore", "f", "explore", "root", 1.0, "", "", "NEW")}
    for i, n in enumerate(["a", "b", "c", "d", "e"] + list(extra_nodes or ())):
        nodes[n] = D.Node(n, "r2", None, "explore", "f", "explore", "root", 2.0 + i, "", R, "NEW")
    nodes["x"] = D.Node("x", "inc", None, "explore", "f", "explore", "root", 9.0, "", "", "NEW")      # not a root child
    for old in ported:
        nodes[f"{old}@NEW"] = D.Node(f"{old}@NEW", "inc", None, "explore", "f", "explore", "root", 20.0, "",
                                     f"port_of={old} {AB.PORT_TAG}", "NEW", old)
    return D.Ledger(nodes, (), (), frozenset(), ((0.0, "r2"),), bases=((0.0, "r2", "NEW"),))


POST = {"a": (0.006, 0.002 ** 2, 5, 5),      # positive, > 1 se: due (best)
        "b": (0.004, 0.002 ** 2, 5, 5),      # positive, > 1 se: due (second)
        "c": (0.001, 0.002 ** 2, 5, 5),      # m < 1 se
        "d": (0.009, 0.002 ** 2, 5, 2),      # < 3 paired seeds
        "e": (-0.004, 0.002 ** 2, 5, 5),     # negative
        "x": (0.02, 0.002 ** 2, 5, 5)}       # not a root child


def test_port_due_picks_positive_root_children_best_first_and_at_most_one_open(monkeypatch):
    ctx = _ctx(alt_base_root=True)
    led, store = _pworld(), MemStore()
    assert AB.port_due(ctx, store, led, FakePost(POST), {}, "inc") == ["a"]                  # room for ONE
    led2 = _pworld(ported=("a",))                                                            # a is ported, still screening
    assert AB.port_due(ctx, store, led2, FakePost(POST), {"a@NEW": SimpleNamespace(name="SCREENING")}, "inc") == []
    # a's port has finished screening: the next best goes
    st = {"a@NEW": SimpleNamespace(name="PASSED")}
    assert AB.port_due(ctx, store, led2, FakePost(POST), st, "inc") == ["b"]
    assert AB.port_due(_ctx(alt_base_root=False), store, led, FakePost(POST), {}, "inc") == []


def test_the_tick_auto_ports_one_root_child_onto_the_incumbent_and_rate_limits_it(monkeypatch):
    from loop import cutover as CO
    from loop import tick as TK
    led, made, logs = _pworld(), [], []
    nodes = {n.id: SimpleNamespace(id=n.id, commit=f"c-{n.id}", parent_id=n.parent, strategy=None, operator=None, family="f",
                                   mechanism_tag="t", hypothesis_id=None, levers=(), notes=n.notes, task_id="T",
                                   status=NodeStatus.SCREENED_OUT)
             for n in led.nodes.values()}

    class St:
        task_id = "T"

        def get_node(self, nid):
            return nodes.get(nid)

        def put_node(self, n):
            made.append(n)
            nodes[n.id] = SimpleNamespace(id=n.id, commit=n.commit, parent_id=n.parent_id, notes=n.notes)

        def log(self, ev, **f):
            logs.append((ev, f))

    class Ad:
        def graft_port(self, parent_commit, commit, onto, name):
            return "sha-port", "applied"

    class Core:
        costs = SimpleNamespace(idea_runs=3)

        def screen_specs(self, node, parent, k, data, blocks, cfg):
            return [("spec", node, parent, k)]

    inc_status = {"inc": TK.Status.INCUMBENT} if hasattr(TK, "Status") else {}
    me = SimpleNamespace(ctx=_ctx(alt_base_root=True), store=St(), adapter=Ad(), core=Core(), dry=False,
                         post=FakePost(POST), blocks=lambda: [], _missing_prereqs=lambda *a, **k: None)
    monkeypatch.setattr(TK, "incumbent_of", lambda s: "inc")
    acts = []
    specs = TK.Tick._alt_root_ports(me, led, {}, 5.0, acts)
    assert [m.id for m in made] == ["a@NEW"] and specs == [("spec", "a@NEW", "inc", 3)]
    n = made[0]
    assert n.parent_id == "inc" and "port_of=a" in n.notes and "port_src=alt_root" in n.notes
    assert n.status.value in ("queued", "QUEUED") or str(n.status).endswith("QUEUED")
    assert any(ev == "alt_root_port" for ev, _ in logs)
    # next tick: the port exists and is still screening -> nothing new; and a shadow tick only lists
    led2 = _pworld(ported=("a",))
    st = {"a@NEW": SimpleNamespace(name="SCREENING")}
    assert TK.Tick._alt_root_ports(me, led2, st, 6.0, []) == [] and len(made) == 1
    me.dry, acts = True, []
    assert TK.Tick._alt_root_ports(me, led, {}, 7.0, acts) == [] and len(made) == 1
    assert acts and "would auto-port root child a" in acts[0]


# ---------------------------------------------------------------------------------------------- kind text of alt sessions
def test_retarget_rewrites_on_and_in_the_incumbent_to_the_starting_point():
    t = AB.retarget("propose ONE mechanism on the incumbent; the incumbent's recipe; in the incumbent", "r2")
    assert "on your starting point (r2)" in t and "in your starting point (r2)" in t and "on the incumbent" not in t
    assert AB.retarget("on the incumbent's recipe", "r2") == "on your starting point (r2's) recipe"


def test_an_alt_session_kind_text_names_the_starting_point_not_the_incumbent(env, monkeypatch):
    ctx, store, led, ide = env
    v = ide.view(led)
    ide2 = ID.Ideation(_ctx(ctx, alt_base_kinds=("explore", "bold")), store)
    monkeypatch.setattr(AB, "due", lambda n, share: True)
    plan = ide2.brief_plan("explore", v, "explore")
    assert plan["alt"]
    kind_line = plan["head"].split("\n\n", 1)[1]
    assert kind_line.startswith("EXPLORE:") and "on the incumbent" not in kind_line
    assert f"on your starting point ({plan['parents'][0]})" in kind_line
    bplan = ide2.brief_plan("explore", v, "bold")
    b = ide2.bold_brief(bplan, led, v)
    assert bplan["alt"]
    kind_head = b.text.split("\n\n", 1)[1].split("## Structural coverage", 1)[0]
    assert "on the incumbent" not in kind_head and "in the incumbent" not in kind_head
    # an incumbent-based session keeps the plain text
    monkeypatch.setattr(AB, "due", lambda n, share: False)
    assert "on the incumbent" in ide2.brief_plan("explore", v, "explore")["head"]
