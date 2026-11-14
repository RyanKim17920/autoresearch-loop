"""loop.select unit tests: in-file doubles, no network, no SLURM."""

from __future__ import annotations

import dataclasses

import random
from types import SimpleNamespace

import pytest

from loop.contracts import (PANEL, ArmPull, Candidate, HandoffNote, HypothesisKind,
                            HypothesisRecord, InitCompat, Node,
                            NodePosterior, NodeStatus, Operator, Posterior, ReviewReport, Run,
                            RunStatus, Stage, Strategy, WorldSnapshot, BANDIT_ARMS)
from loop.spec import (IntakeCfg, FamilyRule, InitCompatCfg, IslandCfg, LoopConfig, BanditCfg)
from loop.select import HashingEmbedder, PortfolioSelector
from loop.select.bandit import Bandit
from loop.select.embed import cosine
from loop.select.islands import IslandBook

TASK = "t1"


# ---- doubles -------------------------------------------------------------------
class FakeStore:
    task_id = TASK

    def __init__(self):
        self._nodes, self._pulls, self._state, self._reviews, self._runs = {}, [], {}, [], []
        self._hyps = []

    def hypotheses(self, *, kind=None):
        return [h for h in self._hyps if kind is None or h.kind is kind]

    def put(self, node):
        self._nodes[node.id] = node
        return node

    def nodes(self, *, status=None, island=None):
        st = set(status) if status is not None else None
        return [n for n in self._nodes.values() if (st is None or n.status in st)
                and (island is None or n.island == island)]

    def get_node(self, i):
        return self._nodes.get(i)

    def find_by_diff_hash(self, h):
        return next((n for n in self._nodes.values() if n.diff_hash == h), None)

    def pulls(self):
        return list(self._pulls)

    def get_state(self, k, default=None):
        return self._state.get(k, default)

    def set_state(self, k, v):
        self._state[k] = v

    def reviews(self):
        return list(self._reviews)

    def runs(self, *, node_id=None, status=None, stage=None):
        return [r for r in self._runs if node_id is None or r.node_id == node_id]


class StubWorld:
    def predictive(self, snapshot, *, family, tag, levers):
        return Posterior(0.0, 0.005)


def make_cfg(**kw):
    base = dict(execution_adapter="x", axes=("a", "b"), weights={"a": .5, "b": .5},
                panel_sd=0.01, axis_sd={"a": .01, "b": .01},
                families={"optim": FamilyRule(keys=("train.lr", "train.wd"), paths=("optim.py",)),
                          "aug": FamilyRule(keys=("data.aug.*",), paths=("aug.py",)),
                          "arch": FamilyRule(keys=("model.*",), paths=("model.py",), init_check=True)})
    base.update(kw)
    return LoopConfig(**base)


def make_ctx(cfg=None):
    surface = SimpleNamespace(writable=("*.py", "configs/"), locked=("eval/",),
                              protected_keys=("train.max_steps",))
    return SimpleNamespace(spec=SimpleNamespace(surface=surface), cfg=cfg or make_cfg(),
                           task_id=TASK, noise=None, spec_path="", state_dir="")


def node(i, family="optim", status=NodeStatus.CONFIRMED, parent="root", strategy=Strategy.EXPLOIT,
         island="main", **kw):
    kw.setdefault("diff_hash", f"h-{i}")
    return Node(id=i, task_id=TASK, commit=f"c-{i}", parent_id=parent,
                strategy=strategy, operator=Operator.IMPROVE, family=family, mechanism_tag=f"tag-{i}",
                status=status, island=island, **kw)


def post(i, mean=0.01, sd=0.003, p=0.9, vetoed=False):
    return NodePosterior(node_id=i, increment=Posterior(mean, sd), utility=Posterior(mean, sd),
                         axes={}, p_positive=p, p_beats_incumbent=0.5, vetoed=vetoed)


def conf(p, est):
    from dataclasses import replace
    return replace(p, confirm_estimate=Posterior(est, 0.003))


def snap(posts=(), n_scored=0, incumbent="root", levers=(), family_means=None):
    return WorldSnapshot(task_id=TASK, n_scored=n_scored, sigma={PANEL: 0.01}, variance={},
                         nodes={p.node_id: p for p in posts}, incumbent_id=incumbent,
                         family_means=family_means or {}, levers=tuple(levers),
                         hypotheses={})


def sel(store=None, cfg=None, seed=0, **kw):
    ctx = make_ctx(cfg)
    return PortfolioSelector(ctx, StubWorld(), rng=random.Random(seed), **kw), \
        (store or FakeStore())


def cand(diff="+lr = 2\n", paths=("optim.py",), cfg_diff=None, family="optim", h="hh", claim="raise lr",
         parent="c-root", task=TASK, ic=None, req="req-1"):
    hyp = HypothesisRecord(id="hy", task_id=task, kind=HypothesisKind.TRIAL, claim=claim,
                           mechanism_tag="m", family=family, predictions={}, falsification="f")
    return Candidate(request_id=req, task_id=task, commit="new", parent_commit=parent, diff_text=diff,
                     diff_hash=h, changed_paths=tuple(paths), config_diff=cfg_diff or {}, hypothesis=hyp,
                     handoff=HandoffNote("h", task, "n", "", "", ""), init_compat=ic)


# ---- bandit ---------------------------------------------------------------------
def test_floor_guarantees_each_eligible_arm():
    b = Bandit(BanditCfg(), random.Random(1))
    pulls = [ArmPull(TASK, Strategy.EXPLOIT, "n", 5.0, 1.0, float(i)) for i in range(50)]
    b.rebuild(pulls)
    counts = {a: 0 for a in BANDIT_ARMS}
    for _ in range(2000):
        counts[b.choose(BANDIT_ARMS)] += 1
    for arm in BANDIT_ARMS:
        assert counts[arm] / 2000 >= 0.08, counts     # floor 0.10 minus 3 sd sampling slack
    assert counts[Strategy.EXPLOIT] > 0.5 * 2000


def test_floor_holds_for_masked_subset():
    b = Bandit(BanditCfg(floor=0.1), random.Random(2)).rebuild([])
    el = [Strategy.EXPLOIT, Strategy.EXPLORE]
    counts = {a: 0 for a in el}
    for _ in range(1000):
        counts[b.choose(el)] += 1
    assert all(c > 100 for c in counts.values())


def test_explore_floor_is_its_own_key():
    b = Bandit(BanditCfg(floor=0.1, explore_floor=0.3), random.Random(4))
    b.rebuild([ArmPull(TASK, Strategy.EXPLOIT, "n", 5.0, 1.0, float(i)) for i in range(50)])
    counts = {a: 0 for a in BANDIT_ARMS}
    for _ in range(4000):
        counts[b.choose(BANDIT_ARMS)] += 1
    assert counts[Strategy.EXPLORE] / 4000 >= 0.27, counts        # its 0.3 floor
    for arm in (Strategy.CROSSOVER, Strategy.REVISIT):
        assert 0.07 <= counts[arm] / 4000 <= 0.14, counts          # the others keep 0.10


def test_discounting_tracks_switching_best_arm():
    pulls, t = [], 0.0
    for phase_best in (Strategy.EXPLORE, Strategy.CROSSOVER):
        for _ in range(60):
            for arm in (Strategy.EXPLORE, Strategy.CROSSOVER):
                t += 1
                pulls.append(ArmPull(TASK, arm, "n", 1.0 if arm == phase_best else 0.0, 1.0, t))
    b = Bandit(BanditCfg(discount=0.9, floor=0.0), random.Random(3)).rebuild(pulls)
    picks = [b.choose([Strategy.EXPLORE, Strategy.CROSSOVER]) for _ in range(200)]
    assert picks.count(Strategy.CROSSOVER) > 170
    stale = Bandit(BanditCfg(discount=1.0, floor=0.0), random.Random(3)).rebuild(pulls)
    assert stale.S[Strategy.EXPLORE] == stale.S[Strategy.CROSSOVER]   # no discount: cannot tell


def test_reward_pays_only_confirmed_uses_confirm_estimates_and_divides_by_gpu_h():
    s, store = sel()
    n = node("n1", status=NodeStatus.CONFIRMED, strategy=Strategy.EXPLORE)
    store.put(n)
    store._runs.append(Run("n1:screen:1", TASK, "n1", 1, Stage.SCREEN, RunStatus.DONE, gpu_h=2.0))
    before = snap([post("n1", 0.0, 0.005)], incumbent="root")
    after = snap([conf(post("n1", 0.05, 0.005), 0.02)])      # selection U 0.05 must be ignored
    pull = s.reward(store, before, after, n)
    assert pull.arm == Strategy.EXPLORE and pull.gpu_h == 2.0
    assert pull.reward >= (0.02 / 0.01) / 2.0 and pull.reward < (0.05 / 0.01) / 2.0
    assert s.reward(store, before, snap([post("n1", 0.05, 0.005)]), n).reward < pull.reward  # no confirm est
    inc_before = snap([conf(post("inc", 0.03, 0.005), 0.018)], incumbent="inc")
    assert s.reward(store, inc_before, after, n).reward < pull.reward       # vs incumbent's confirm est
    lost = s.reward(store, before, after, node("n1", status=NodeStatus.SCREENED_OUT))
    assert lost.reward < pull.reward


# ---- intake ---------------------------------------------------------------------
def root_store():
    store = FakeStore()
    store.put(node("root", family="", status=NodeStatus.BASELINE, parent=None))
    return store


def test_one_family_rule_and_crossover_exemption():
    s, store = sel(store=root_store())
    two = cand(paths=("optim.py", "aug.py"), cfg_diff={"train.lr": 2, "data.aug.p": 1})
    v = s.admit(store, snap(), two)
    assert v == "multi_family"
    assert s.admit(store, snap(), two, Operator.CROSSOVER) == ""


def test_ambiguous_unmapped_and_unknown_family_rejected():
    cfg = make_cfg(families={"a": FamilyRule(paths=("x*.py",)), "b": FamilyRule(paths=("*y.py",))})
    s, store = sel(store=root_store(), cfg=cfg)
    assert s.admit(store, snap(), cand(paths=("xy.py",), family="a")) == "ambiguous_family"
    assert s.admit(store, snap(), cand(paths=("x1.py",), family="b")) == "family_mismatch"
    assert s.admit(store, snap(), cand(paths=("x1.py",), family="a")) == ""
    assert s.admit(store, snap(), cand(paths=("z.py",), family="a")) == "unmapped_path"
    assert s.admit(store, snap(), cand(paths=("x1.py",), family="free text")) == "unknown_family"
    s2, store2 = sel(store=root_store())
    # a brand-new key takes the family of the code that reads it (loop.select.intake.item_families) ...
    assert s2.admit(store2, snap(), cand(cfg_diff={"other.key": 1})) == ""
    # ... and is unmapped only when nothing names one
    assert s2.admit(store2, snap(), cand(paths=("configs/main.yaml",), cfg_diff={"other.key": 1},
                                         family="free text")) == "unmapped_path"
    # config files are judged by their keys, not their path
    ok = cand(paths=("configs/main.yaml",), cfg_diff={"train.lr": 2})
    assert s2.admit(store2, snap(), ok) == ""


def test_catch_all_rule_maps_otherwise_unmapped():
    cfg = make_cfg(families={"a": FamilyRule(paths=("x*.py",)), "misc": FamilyRule(paths=("*",))})
    s, store = sel(store=root_store(), cfg=cfg)
    assert s.admit(store, snap(), cand(paths=("z.py",), family="misc")) == ""
    assert s.admit(store, snap(), cand(paths=("x1.py",), family="a")) == ""   # specific wins


def test_a_re_rooted_copy_of_an_existing_commit_or_tree_is_an_exact_dup():
    """n0363 = n0292's commit (90940a53) re-rooted on root as a crossover: a different diff (vs root, not vs n0255)
    and diff_hash, the same code + config. The commit, or its tree, is the identity -- whatever the parent."""
    from loop.select.intake import K_TREES
    store = root_store()
    s, _ = sel(store=store)
    store.put(node("n1", diff_hash="h1"))
    c = cand(diff="+whole stack vs root\n", h="h9", claim="deep wide predictor x lr up")
    assert s.admit(store, snap(), dataclasses.replace(c, commit="c-n1")) == "exact_dup"
    store.set_state(K_TREES, {"c-n1": "T1", "rebuilt": "T1"})
    assert s.admit(store, snap(), dataclasses.replace(c, commit="rebuilt")) == "exact_dup"
    store.set_state(K_TREES, {"c-n1": "T1", "rebuilt": "T2"})
    assert s.admit(store, snap(), dataclasses.replace(c, commit="rebuilt")) == ""


def test_exact_and_near_duplicate_rejected():
    store = root_store()
    s, _ = sel(store=store)
    c0 = cand(diff="+lr = 2\n+warmup = 5\n", h="h1", claim="raise the lr with warmup")
    store.put(node("n1", embedding=s.embedding_of(c0), diff_hash="h1"))
    v = s.admit(store, snap(), cand(diff="+different\n", h="h1"))
    assert v == "exact_dup"
    v = s.admit(store, snap(), cand(diff="+lr = 2\n+warmup = 5\n", h="h2", claim="raise the lr with warmup"))
    assert v == "near_dup"
    far = cand(diff="+use_cutmix(images)\n", h="h3", claim="cutmix regularisation", family="optim")
    assert s.admit(store, snap(), far) == ""


def test_parent_contains_knob_rejected():
    store = root_store()
    s, _ = sel(store=store)
    assert s.admit(store, snap(), cand(cfg_diff={"train.lr": 1e-4})) == ""   # root config unknown
    store.put(node("n2"))
    store.set_state("config_diff:n2", {"train": {"lr": 3e-4}})
    v = s.admit(store, snap(), cand(cfg_diff={"train.lr": 3e-4}, parent="c-n2", h="x"))
    assert v == "parent_contains_knob"


def test_surface_and_task_guards():
    s, store = sel(store=root_store())
    assert s.admit(store, snap(), cand(paths=("eval/score.py",))) == "surface_locked"
    assert s.admit(store, snap(), cand(paths=("README.md",))) == "surface_not_writable"
    assert s.admit(store, snap(), cand(cfg_diff={"train.max_steps": 5})) == "protected_key"
    assert s.admit(store, snap(), cand(diff="", paths=())) == "empty_diff"


def test_init_compat():
    s, store = sel(store=root_store())
    arch = dict(paths=("model.py",), family="arch", cfg_diff={"model.depth": 4})
    assert s.admit(store, snap(), cand(**arch)) == "init_compat_missing"
    assert s.admit(store, snap(), cand(ic=InitCompat(False, "shape"), **arch)) == "init_incompatible"
    assert s.admit(store, snap(), cand(ic=InitCompat(True), **arch)) == ""
    cfg = make_cfg(init_compat=InitCompatCfg(paths=("optim.py",)))
    s3, _ = sel(store=store, cfg=cfg)
    assert s3.admit(store, snap(), cand()) == "init_compat_missing"
    assert s3.admit(store, snap(), cand(ic=InitCompat(True))) == ""


def test_init_swap_failures_are_typed_and_denied_inits_rejected_on_any_diff():
    """An init swap's failed check names its rule (InitCompat.rule); a denied (pathology-trained) checkpoint in any
    added code line or config value is init_illegal before any check; a generic new init is admitted."""
    cfg = make_cfg(init_compat=InitCompatCfg(deny=("bioptimus|h[-_ ]?optimus", "(?-i:\\bUNI\\b)|mahmoodlab")))
    s, store = sel(store=root_store(), cfg=cfg)
    arch = dict(paths=("model.py",), family="arch", cfg_diff={"model.depth": 4})
    swap = {"init_swap": {"to": {"type": "vit_b"}}}
    illegal = InitCompat(False, "init_illegal: ...", {**swap, "init_legal": {"ok": False}, "load": {"ok": True}})
    over = InitCompat(False, "over_flop_cap: ...", {**swap, "init_legal": {"ok": True}, "flop_budget": {"ok": False}})
    broken = InitCompat(False, "load", {**swap, "init_legal": {"ok": True}, "load": {"ok": False}})
    assert illegal.swap and illegal.rule == "init_illegal" and s.admit(store, snap(), cand(ic=illegal, **arch)) == "init_illegal"
    assert over.rule == "over_flop_cap" and s.admit(store, snap(), cand(ic=over, **arch)) == "over_flop_cap"
    assert broken.rule == "init_incompatible" and s.admit(store, snap(), cand(ic=broken, **arch)) == "init_incompatible"
    assert not InitCompat(False, "shape", {"checkpoint_keys": {"ok": False}}).swap
    ok = InitCompat(True, "", {**swap, "init_legal": {"ok": True}})
    assert s.admit(store, snap(), cand(ic=ok, **arch)) == ""
    hub = dict(paths=("model.py",), family="arch", ic=ok)
    assert s.admit(store, snap(), cand(diff='+    w = hf_hub_download("bioptimus/H-optimus-0")\n', **hub)) == "init_illegal"
    assert s.admit(store, snap(), cand(diff="+x = 1\n", cfg_diff={"model.ckpt": "MahmoodLab/UNI"}, **hub)) == "init_illegal"
    assert s.admit(store, snap(), cand(diff="+    # unlike H-optimus we keep the generic init\n+x = 1\n", **hub)) == ""
    assert s.admit(store, snap(), cand(diff='+    url = "timm/vit_base_patch16_dinov3.lvd1689m"  # uniform\n', **hub)) == ""


def test_hashing_embedder_properties():
    e = HashingEmbedder()
    v = e.embed("lower the learning rate and weight decay")
    assert len(v) == 512 and abs(sum(x * x for x in v) - 1) < 1e-9
    assert cosine(v, e.embed("lower the learning rate and weight decay")) > 0.999
    assert cosine(v, e.embed("cutmix augmentation of tiles")) < 0.5
    assert e.embed("") == tuple([0.0] * 512)



# ---- thompson ranking -----------------------------------------------------------
def test_rank_queue_bonus_bias_caps_in_sigmas_and_lever_counted_once():
    cfg = make_cfg()
    s, store = sel(cfg=cfg, seed=5)
    s.world = SimpleNamespace(predictive=lambda *a, **k: Posterior(0.0, 1e-9))
    good = node("g", status=NodeStatus.QUEUED, prior_bonus=0.005, family="aug")
    bad = node("b", status=NodeStatus.QUEUED, prior_bonus=-0.005)
    store._reviews.append(ReviewReport("r", TASK, "cadence", "", (), (), (), {"aug": 1.0}, n_evals_at=0))
    assert [n.id for n in s.rank_queue(store, snap(n_scored=0), [bad, good])] == ["g", "b"]
    from loop.select.thompson import review_bias
    assert review_bias(store, cfg, snap(n_scored=0), "aug") == pytest.approx(0.005)   # 0.5 * sigma 0.01
    assert review_bias(store, cfg, snap(n_scored=cfg.review.bias_decay_evals), "aug") == 0.0
    import loop.select.thompson as th
    assert not hasattr(th, "lever_term")      # the lever trend is added once, by world.predictive


# ---- arms + islands -------------------------------------------------------------
def test_next_request_masks_ineligible_arms_and_uses_root_fallback():
    s, store = sel(store=root_store())
    seen = set()
    for _ in range(40):
        r = s.next_request(store, snap())
        seen.add(r.strategy)
        assert r.parent_ids == ("root",) or r.strategy == Strategy.EXPLORE
    assert seen == {Strategy.EXPLOIT, Strategy.EXPLORE, Strategy.MOONSHOT}   # moonshot: always eligible


def test_crossover_requires_two_families_and_revisit_needs_candidates():
    store = root_store()
    store.put(node("w1", family="optim"))
    store.put(node("w2", family="optim"))
    s, _ = sel(store=store)
    plans = lambda: {a: s._plans(a, __import__("loop.select.islands", fromlist=["x"]).IslandBook(s.cfg, store),
                                 store.nodes(), snap([post("w1"), post("w2")])) for a in BANDIT_ARMS}
    p = plans()
    assert not p[Strategy.CROSSOVER] and not p[Strategy.REVISIT]
    store.put(node("w3", family="aug"))
    store.put(node("cr", family="aug", status=NodeStatus.CRASHED, parent="root"))
    store.put(node("nm", family="aug", status=NodeStatus.SCREENED_OUT))
    p = plans()
    assert p[Strategy.CROSSOVER]["main"][0] == Operator.CROSSOVER
    a, b = p[Strategy.CROSSOVER]["main"][1]
    assert {store.get_node(a).family, store.get_node(b).family} == {"optim", "aug"}
    assert p[Strategy.REVISIT]["main"][:2] == (Operator.DEBUG, ("cr",))


def test_crossover_pool_admits_strong_screened_nodes_not_vetoed_refuted_or_low_p():
    from dataclasses import replace
    from loop.contracts import HypothesisEvidence, HypothesisStatus, LeverTags
    from loop.select import crossover_meta
    from loop.select.islands import IslandBook
    store = root_store()
    store.put(node("a", family="aug", status=NodeStatus.SCREENED_OUT, levers=LeverTags({"augmentation_strength": -1})))
    store.put(node("a2", family="aug", status=NodeStatus.SCREENED_OUT))         # same family as a
    store.put(node("b", family="optim", status=NodeStatus.NOT_CONFIRMED,
                   levers=LeverTags({"objective_difficulty": -1})))
    store.put(node("v", family="arch", status=NodeStatus.SCREENED_OUT))         # vetoed
    store.put(node("r", family="arch", status=NodeStatus.SCREENED_OUT, hypothesis_id="hr"))  # refuted
    store.put(node("l", family="arch", status=NodeStatus.SCREENED_OUT))         # P below min
    store.put(node("q", family="arch", status=NodeStatus.SCREENING))            # not closed
    store.put(node("x", family="arch", status=NodeStatus.VETOED))
    posts = [post("a", p=0.95), post("a2", p=0.99), post("b", p=0.90), post("v", p=0.999, vetoed=True),
             post("r", p=0.999), post("l", p=0.80), post("q", p=0.999), post("x", p=0.999)]
    sn = replace(snap(posts), hypotheses={"hr": HypothesisEvidence("hr", 0.1, 30.0, 1, HypothesisStatus.REFUTED)})
    s, _ = sel(store=store)
    plan = lambda inflight=(): s._plans(Strategy.CROSSOVER, IslandBook(s.cfg, store), store.nodes(), sn, inflight)
    assert plan()["main"][:2] == (Operator.CROSSOVER, ("a2", "b"))   # best joint P, different families
    store.put(node("c1", family="aug", status=NodeStatus.SCREENED_OUT, strategy=Strategy.CROSSOVER,
                   second_parent_id="b", notes="request=r xover=a2,b"))
    assert plan()["main"][1] == ("a", "b")                          # the tried pair is not repeated
    store.put(replace(store.get_node("a"), mechanism_tag="TAG_a2"))    # live n0431 / n0427: one mechanism,
    assert not plan()                                               # so a x b is the tried a2 x b again
    store.put(replace(store.get_node("a"), mechanism_tag="tag-a"))
    assert not plan([{"id": "q", "strategy": "crossover", "family": "", "parents": ["a", "b"]}])
    s2, _ = sel(store=store, cfg=make_cfg(bandit=BanditCfg(crossover_min_p=0.96)))
    assert not s2._plans(Strategy.CROSSOVER, IslandBook(s2.cfg, store), store.nodes(), sn)  # only a2 left
    # the pair's combined levers pick the review meta-hypothesis that matches them best (m4)
    meta = lambda i, lev: HypothesisRecord(id=i, task_id=TASK, kind=HypothesisKind.META, claim="c",
                                           mechanism_tag="m", family="cross", predictions={},
                                           falsification="f", levers=LeverTags(lev))
    store._hyps = [meta("m1", {"augmentation_strength": -1}), meta("m4", {"augmentation_strength": -1,
                                                                          "objective_difficulty": -1}),
                   meta("m9", {"capacity": 1})]
    assert crossover_meta(store, ("a", "b")) == "m4"
    assert crossover_meta(store, ("v", "l")) is None                # no lever overlap


def test_revisit_near_miss_and_crash_depth_cap():
    store = root_store()
    store.put(node("cr", family="aug", status=NodeStatus.CRASHED, debug_depth=3))
    store.put(node("nm", family="aug", status=NodeStatus.SCREENED_OUT))
    s, _ = sel(store=store)
    from loop.select.islands import IslandBook
    plan = s._plans(Strategy.REVISIT, IslandBook(s.cfg, store), store.nodes(),
                    snap([post("nm", 0.002, 0.004, p=0.5)]))
    assert plan["main"][:2] == (Operator.IMPROVE, ("root", "nm"))   # crash at max_debug is skipped
    # a closed node is never the parent: its idea is rebuilt on the nearest established (confirmed) ancestor
    from loop.select import established
    store.put(node("C", family="aug"))
    store.put(node("x", family="aug", status=NodeStatus.NOT_CONFIRMED, parent="C"))
    store.put(node("nm2", family="aug", status=NodeStatus.SCREENED_OUT, parent="x"))
    assert established(store, store.get_node("nm2")) == "C" and established(store, store.get_node("nm")) == "root"
    plan = s._plans(Strategy.REVISIT, IslandBook(s.cfg, store), store.nodes(),
                    snap([post("nm", 0.002, 0.004, p=0.5), post("nm2", 0.003, 0.004, p=0.6)]))
    assert plan["main"][:2] == (Operator.IMPROVE, ("C", "nm2"))


def test_exploit_parent_penalised_by_offspring():
    from loop.select.thompson import pick_exploit_parent
    sn = snap([post("a", 0.011, 1e-9), post("b", 0.010, 1e-9)])
    rng = random.Random(0)
    assert pick_exploit_parent(sn, ["a", "b"], {}, 0.01, rng) == "a"
    assert pick_exploit_parent(sn, ["a", "b"], {"a": 10}, 0.01, rng) == "b"


def island_store():
    store = root_store()
    for i, fam in enumerate(["optim", "optim", "optim", "aug", "aug", "arch"]):
        store.put(node(f"n{i}", family=fam, status=NodeStatus.SCREENED_OUT))
    return store


def test_island_split_at_threshold_greedy_and_new_family():
    store = island_store()
    s, _ = sel(store=store, cfg=make_cfg(islands=IslandCfg(n=2, split_at=10, reset_every=2)))
    from loop.select.islands import IslandBook
    b = IslandBook(s.cfg, store)
    b.maybe_split(snap(n_scored=9))
    assert b.islands == ["main"]
    b.maybe_split(snap(n_scored=10))
    assert b.islands == ["i0", "i1"]
    assert b.assign["optim"] != b.assign["aug"]                  # 3 vs (2+1): balanced
    assert b.assign["aug"] == b.assign["arch"]
    again = IslandBook(s.cfg, store)                             # persisted
    assert again.assign == b.assign
    store.put(node("z", family="optim", status=NodeStatus.SCREENED_OUT))   # optim island now 4 vs 3
    assert again.island_of("brandnew") == b.assign["aug"]        # new family -> smallest island
    assert IslandBook(s.cfg, store).assign["brandnew"] == b.assign["aug"]


def test_migration_pool_and_weakest_reset():
    store = island_store()
    cfg = make_cfg(islands=IslandCfg(n=2, split_at=5, reset_every=2))
    s, _ = sel(store=store, cfg=cfg)
    from loop.select.islands import IslandBook
    s.on_confirmed(store, snap(n_scored=6), store._nodes["n0"])                # splits the islands
    book = IslandBook(cfg, store)
    home_opt, home_aug = book.assign["optim"], book.assign["aug"]
    store._nodes["n0"] = node("n0", family="optim", status=NodeStatus.CONFIRMED)
    sn = snap([post("n0", 0.02)], n_scored=6, incumbent="n0")
    s.on_confirmed(store, sn, store._nodes["n0"])
    b = IslandBook(cfg, store)
    assert b.frontier(home_aug, sn) == ["n0"]                      # migrant is the other island's parent pool
    assert b.frontier(home_opt, sn) == ["n0"]
    store._nodes["n3"] = node("n3", family="aug", status=NodeStatus.CONFIRMED)
    sn2 = snap([post("n0", 0.02), post("n3", 0.005)], n_scored=8, incumbent="n0")
    s.on_confirmed(store, sn2, store._nodes["n3"])                 # 2nd confirmation -> reset weakest
    b = IslandBook(cfg, store)
    assert b.resets[home_aug]["parent"] == "n0"
    assert b.frontier(home_aug, sn2) == ["n0"]
    assert set(b.frontier(home_opt, sn2)) == {"n0", "n3"}
    nxt = s.next_request(store, sn2)
    assert nxt is not None and nxt.brief_kind != "migration"


def test_probe_pairs_drain_once_with_their_own_meta():
    store = root_store()
    store._reviews.append(ReviewReport("r1", TASK, "cadence", "", (), ("m1", "m2"),
                                       (("m2", "probe two"), (None, "global probe")), {}, 5))
    s, _ = sel(store=store)
    got = [s.next_request(store, snap()) for _ in range(40)]
    probes = [r for r in got if r.brief_kind == "probe"]
    assert [r.meta_hypothesis_id for r in probes] == ["m2", None]       # each once, in order
    assert "probe two" in probes[0].brief and "global probe" in probes[1].brief
    assert all(r.strategy in (Strategy.EXPLORE, Strategy.EXPLOIT) for r in probes)
    assert s.pending_probes(store) == []


def test_bandit_state_rebuilds_from_pulls_and_requests_are_unique():
    store = root_store()
    s, _ = sel(store=store)
    ids = {s.next_request(store, snap()).id for _ in range(10)}
    assert len(ids) == 10
    assert s.next_request(FakeStore(), snap()) is not None


def test_a_debugged_crash_is_not_revisited_again():
    """A DEBUG child hangs off the crashed node's parent; the crash is its second parent."""
    store = root_store()
    store.put(node("cr", family="aug", status=NodeStatus.CRASHED))
    store.put(node("fix", family="aug", status=NodeStatus.QUEUED, strategy=Strategy.REVISIT,
                   second_parent_id="cr", debug_depth=1))
    s, _ = sel(store=store)
    from loop.select.islands import IslandBook
    plan = s._plans(Strategy.REVISIT, IslandBook(s.cfg, store), store.nodes(), snap([]))
    assert not plan                                            # cr already has its debug child
    store.put(node("cr2", family="aug", status=NodeStatus.CRASHED))
    busy = [{"id": "r1", "strategy": "revisit", "family": "aug", "parents": ["cr2"]}]
    assert s._plans(Strategy.REVISIT, IslandBook(s.cfg, store), store.nodes(), snap([]), busy) == {}
    assert s._plans(Strategy.REVISIT, IslandBook(s.cfg, store), store.nodes(), snap([]))  # cr2 is free


def test_diagnose_gives_the_admit_rule_plus_a_one_line_fix():
    from loop.select.intake import diagnose, needs_init_compat
    s, store = sel(store=root_store())
    ctx, emb = make_ctx(), HashingEmbedder()
    cases = [(cand(paths=("eval/score.py",)), "surface_locked", "eval/score.py"),
             (cand(paths=("optim.py", "aug.py"), cfg_diff={"train.lr": 2, "data.aug.p": 1}), "multi_family", "aug"),
             (cand(cfg_diff={"train.max_steps": 5}), "protected_key", "train.max_steps"),
             (cand(family="aug"), "family_mismatch", "optim"),
             (cand(paths=("configs/main.yaml",), cfg_diff={"other.key": 1}, family="x"), "unmapped_path", "other.key")]
    for c, rule, word in cases:
        got, fix = diagnose(ctx, store, c, operator=None, embedder=emb)
        assert got == rule == s.admit(store, snap(), c) and word in fix and "\n" not in fix, (rule, fix)
    assert diagnose(ctx, store, cand(), operator=None, embedder=emb) == ("", "")
    assert needs_init_compat(ctx, cand(paths=("model.py",), family="arch"), ("arch",))
    assert not needs_init_compat(ctx, cand(), ("optim",))


# ---- concurrent-researcher diversity ----------------------------------------------
def test_explore_targets_least_sampled_family_not_already_being_explored():
    from loop.contracts import K_INFLIGHT
    from loop.select.islands import IslandBook
    store = root_store()
    store.put(node("a1", family="arch", status=NodeStatus.QUEUED))
    s, _ = sel(store=store)

    def plan(inflight, sn=None):
        return s._plans(Strategy.EXPLORE, IslandBook(s.cfg, store), store.nodes(), sn or snap(),
                        inflight).get("main")
    got, inflight = [], []
    for i in range(3):                       # 3 concurrent explore requests, each sees the earlier ones
        got.append(plan(inflight)[3])
        inflight.append({"id": f"r{i}", "strategy": "explore", "family": got[-1]})
    assert sorted(got[:2]) == ["aug", "optim"] and got[2] == "arch"    # the queued family comes last
    assert plan(inflight) is None                                     # all taken: the explore arm is masked
    store.set_state(K_INFLIGHT, inflight)
    assert all(s.next_request(store, snap()).strategy is not Strategy.EXPLORE for _ in range(20))
    assert plan([{"id": "x", "strategy": "exploit", "family": "optim"}])[3] == "aug"   # in-flight counts
    fm = {"aug": Posterior(0.0, 0.01), "optim": Posterior(0.0, 0.02)}
    assert plan([], snap(family_means=fm))[3] == "optim"               # tie -> widest posterior


def test_queue_diversity_caps_screening_per_family_and_tag_but_never_idles():
    from dataclasses import replace
    from loop.select.thompson import diversify
    mk = lambda i, fam, tag, st=NodeStatus.QUEUED: replace(node(i, family=fam, status=st), mechanism_tag=tag)  # noqa: E731
    screening = [mk("s", "data", "tissue", NodeStatus.SCREENING)]
    ranked = [mk("a", "data", "tissue"), mk("b", "data", "gentle"), mk("c", "data", "otsu"), mk("d", "optim", "lr")]
    assert [n.id for n in diversify(ranked, screening)] == ["b", "d", "a", "c"]   # capped ones last, not dropped
    store = root_store()
    for n in screening + ranked:
        store.put(n)
    s, _ = sel(store=store)
    head = [n.id for n in s.rank_queue(store, snap(), ranked)]
    assert sorted(head[:2]) in (["b", "d"], ["c", "d"])                # at most one more data node first


def test_value_variant_of_an_unresolved_knob_is_rejected_until_it_resolves():
    from dataclasses import replace
    store = root_store()
    store.put(node("q", family="optim", status=NodeStatus.SCREENING))
    store.set_state("config_diff:q", {"train.lr": 0.5})
    s, _ = sel(store=store)
    variant = cand(diff="-lr: 1\n+lr: 2\n", paths=("configs/c.yaml",), cfg_diff={"train.lr": 2})
    assert s.admit(store, snap(), variant) == "near_dup_inflight"
    other = cand(diff="+wd: 1\n", paths=("configs/c.yaml",), cfg_diff={"train.lr": 2, "train.wd": 1}, h="h2")
    assert s.admit(store, snap(), other) == ""                         # a different knob set is new
    store.put(replace(store.get_node("q"), status=NodeStatus.SCREENED_OUT))
    assert s.admit(store, snap(), variant) == ""                       # a legit follow-up once resolved


def test_near_copy_of_a_live_nodes_config_change_set_is_rejected():
    from dataclasses import replace
    store = root_store()
    store.put(node("q", family="optim", status=NodeStatus.SCREENED_OUT))        # resolved: not in flight
    store.set_state("config_diff:q", {"train.lr": 0.5})
    s, _ = sel(store=store)
    near = cand(diff="-lr: 1\n+lr: 0.6\n", paths=("configs/c.yaml",), cfg_diff={"train.lr": 0.6})
    assert s.admit(store, snap(), near) == "near_dup_config"                    # same keys, within 2x
    far = cand(diff="-lr: 1\n+lr: 2\n", paths=("configs/c.yaml",), cfg_diff={"train.lr": 2}, h="h2")
    assert s.admit(store, snap(), far) == ""                                    # 4x: a MAJOR move
    more = cand(diff="+wd: 1\n", paths=("configs/c.yaml",), cfg_diff={"train.lr": 0.6, "train.wd": 1}, h="h3")
    assert s.admit(store, snap(), more) == ""                                   # another knob set
    store.put(replace(store.get_node("q"), status=NodeStatus.REJECTED))
    assert s.admit(store, snap(), near) == ""                                   # a rejected node is no twin


# ---- moonshot arm -----------------------------------------------------------------
def test_moonshot_arm_capped_by_moonshot_max_inflight_and_its_floor_respected():
    from dataclasses import replace
    s, store = sel(store=root_store())
    assert s.cfg.bandit.moonshot_max_inflight == 2
    plans = s._plans(Strategy.MOONSHOT, IslandBook(s.cfg, store), store.nodes(), snap(), [])
    assert list(plans.values()) == [(Operator.DRAFT, ("root",), ("moonshot",), "")]
    one = [{"id": "r1", "strategy": "moonshot", "family": "", "parents": ["root"]}]
    assert s._plans(Strategy.MOONSHOT, IslandBook(s.cfg, store), store.nodes(), snap(), one)   # 1 < 2: open
    two = one + [{"id": "r2", "strategy": "moonshot", "family": "", "parents": ["root"]}]
    assert s._plans(Strategy.MOONSHOT, IslandBook(s.cfg, store), store.nodes(), snap(), two) == {}
    store.put(node("m1", strategy=Strategy.MOONSHOT, status=NodeStatus.QUEUED))       # a queued moonshot
    assert s._plans(Strategy.MOONSHOT, IslandBook(s.cfg, store), store.nodes(), snap(), one) == {}   # node counts
    store.put(replace(store.get_node("m1"), status=NodeStatus.SCREENING))   # screening (hours on the preemptible
    assert s._plans(Strategy.MOONSHOT, IslandBook(s.cfg, store), store.nodes(), snap(), one)   # pool): frees it
    store.put(replace(store.get_node("m1"), status=NodeStatus.QUEUED, notes="tune_of=m0 tune=t1 brief=tune"))
    assert s._plans(Strategy.MOONSHOT, IslandBook(s.cfg, store), store.nodes(), snap(), one)   # a tuning winner: never
    store.put(replace(store.get_node("m1"), status=NodeStatus.SCREENED_OUT, notes=""))  # closed: does not
    assert s._plans(Strategy.MOONSHOT, IslandBook(s.cfg, store), store.nodes(), snap(), one)
    s1, _ = sel(store=store, cfg=make_cfg(bandit=BanditCfg(moonshot_max_inflight=1)))
    assert s1._plans(Strategy.MOONSHOT, IslandBook(s1.cfg, store), store.nodes(), snap(), one) == {}
    b = Bandit(BanditCfg(floor=0.1, explore_floor=0.3, moonshot_floor=0.2), random.Random(5))
    b.rebuild([ArmPull(TASK, Strategy.EXPLOIT, "n", 5.0, 1.0, float(i)) for i in range(50)])
    counts = {a: 0 for a in BANDIT_ARMS}
    for _ in range(4000):
        counts[b.choose(BANDIT_ARMS)] += 1
    assert counts[Strategy.MOONSHOT] / 4000 >= 0.18, counts          # its 0.2 floor
    assert 0.07 <= counts[Strategy.REVISIT] / 4000 <= 0.14, counts


def test_moonshot_may_span_families_but_locks_protected_and_init_compat_hold():
    s, store = sel(store=root_store())
    two = cand(paths=("optim.py", "aug.py"), cfg_diff={"train.lr": 2, "data.aug.p": 1})
    assert s.admit(store, snap(), two) == "multi_family"
    assert s.admit(store, snap(), two, Operator.DRAFT, strategy=Strategy.MOONSHOT) == ""
    assert s.admit(store, snap(), cand(paths=("optim.py", "aug.py"), cfg_diff={"train.lr": 2, "data.aug.p": 1},
                                       family="arch"), strategy=Strategy.MOONSHOT) == "family_mismatch"
    moon = dict(operator=Operator.DRAFT, strategy=Strategy.MOONSHOT)
    assert s.admit(store, snap(), cand(paths=("optim.py", "eval/score.py"), h="h2"), **moon) == "surface_locked"
    assert s.admit(store, snap(), cand(paths=("optim.py",), cfg_diff={"train.lr": 2, "train.max_steps": 9},
                                       h="h3"), **moon) == "protected_key"
    arch = cand(paths=("optim.py", "model.py"), h="h4", diff="+head = Aux()\n")
    assert s.admit(store, snap(), arch, **moon) == "init_compat_missing"
    bad = InitCompat(False, "shape mismatch")
    assert s.admit(store, snap(), cand(paths=("optim.py", "model.py"), h="h5", diff="+aux\n", ic=bad),
                   **moon) == "init_incompatible"


def test_tagged_review_probe_runs_as_a_moonshot():
    s, store = sel(store=root_store())
    store._reviews.append(ReviewReport("r", TASK, "cadence", "", (), (),
                                       ((None, "[moonshot] PCGrad over the SSL losses"), (None, "halve wd")),
                                       {}, n_evals_at=0))
    got = [s.next_request(store, snap()) for _ in range(40)]
    moon = [r for r in got if "PCGrad" in r.brief]
    plain = [r for r in got if "halve wd" in r.brief]
    assert len(moon) == len(plain) == 1
    assert (moon[0].strategy, moon[0].brief_kind, moon[0].operator) == (Strategy.MOONSHOT, "moonshot", Operator.DRAFT)
    assert plain[0].strategy in (Strategy.EXPLORE, Strategy.EXPLOIT) and plain[0].brief_kind == "probe"


def test_pending_probes_never_starve_the_bandit_floors():
    """40 queued review probes (8 tagged [moonshot]) must not bypass the bandit: over 50 requests
    every eligible arm keeps its floor, a tagged probe only fills a MOONSHOT pull and an untagged one
    only an EXPLORE/EXPLOIT pull, and a tagged probe waits while a moonshot is in flight."""
    from loop.contracts import K_INFLIGHT
    cfg = make_cfg(bandit=BanditCfg(floor=0.1, explore_floor=0.3, moonshot_floor=0.3))
    briefs = tuple((None, f"[moonshot] bold {i}" if i % 5 == 0 else f"probe {i}") for i in range(40))
    total, n_seeds = {a: 0 for a in BANDIT_ARMS}, 8
    for seed in range(n_seeds):
        store = root_store()
        store._reviews.append(ReviewReport("r", TASK, "cadence", "", (), (), briefs, {}, n_evals_at=0))
        s, _ = sel(store=store, cfg=cfg, seed=seed)
        got = [s.next_request(store, snap()) for _ in range(50)]
        counts = {a: sum(r.strategy is a for r in got) for a in BANDIT_ARMS}
        for a in BANDIT_ARMS:
            total[a] += counts[a]
        assert counts[Strategy.MOONSHOT] >= 6 and counts[Strategy.EXPLORE] >= 6, (seed, counts)
        for r in got:
            if "[moonshot] bold" in r.brief:
                assert (r.strategy, r.operator, r.brief_kind) == (Strategy.MOONSHOT, Operator.DRAFT, "moonshot")
            elif r.brief_kind == "probe":
                assert r.strategy in (Strategy.EXPLORE, Strategy.EXPLOIT)
        assert any(r.brief_kind != "probe" and "[moonshot] bold" not in r.brief for r in got)   # bandit ran
    n = 50 * n_seeds
    assert total[Strategy.MOONSHOT] / n >= 0.3 - 0.05, total          # its 0.3 floor
    assert total[Strategy.EXPLORE] / n >= 0.3 - 0.05, total
    assert total[Strategy.EXPLOIT] / n >= 0.1 - 0.03, total
    store = root_store()                                   # a tagged probe waits at moonshot_max_inflight
    store._reviews.append(ReviewReport("r", TASK, "cadence", "", (), (), briefs[:1], {}, n_evals_at=0))
    store.set_state(K_INFLIGHT, [{"id": "x", "strategy": "moonshot", "family": "", "parents": ["root"]}])
    store.put(node("mq", strategy=Strategy.MOONSHOT, status=NodeStatus.QUEUED))
    s, _ = sel(store=store, cfg=cfg)
    assert all(s.next_request(store, snap()).strategy is not Strategy.MOONSHOT for _ in range(30))
    assert len(s.pending_probes(store)) == 1


# ---- eval_readout: a guarded, combinable family --------------------------------------------------

def readout_sel(**kw):
    fams = dict(make_cfg().families, readout=FamilyRule(
        paths=("model.py::ViT.forward", "train.py::# Fit...copy_"), combine=True,
        forbid=(r"benchmarking|['\"]probe['\"]", r"(?<![\w-])bach(?![\w-])", r"['\"]labels?['\"]")))
    s, store = sel(store=root_store(), cfg=make_cfg(families=fams, **kw))
    s.ctx.spec.surface.locked = ("eval/", "probe.py", "train.py::calibration_tiles...budget")
    return s, store


def test_readout_guard_rejects_label_and_probe_data_but_admits_a_training_fit():
    s, store = readout_sel()
    ro = dict(paths=("model.py::ViT.forward",), family="readout")
    for i, line in enumerate(['roots = cfg["probe"]["dataset_roots"]', 'x = load("/data/thunder-data/bach")',
                              'y = df["label"]', "from benchmarking import tasks"]):
        assert s.admit(store, snap(), cand(diff=f"+{line}\n", h=f"g{i}", claim=f"c{i}", **ro)) \
            == "forbidden_reference", line
    legit = "+    cls = self.readout_norm(cls)  # fit on TCGA tiles\n+    labels = np.array(site_rows)\n"
    assert s.admit(store, snap(), cand(diff=legit, h="ok", claim="normalise cls", **ro)) == ""
    assert s.admit(store, snap(), cand(diff=legit, h="ok2", claim="normalise cls", paths=ro["paths"],
                                       family="arch")) == "family_mismatch"     # always filed as readout


def test_readout_combines_with_one_training_family_and_eval_stays_locked():
    s, store = readout_sel()
    head = cand(diff="+self.head = Head()\n", paths=("model.py::ViT.forward", "optim.py"), family="readout",
                h="c1", claim="train a readout head during pretraining")
    assert s.admit(store, snap(), head) == ""                                    # readout + ONE training family
    three = cand(diff="+x\n", paths=("model.py::ViT.forward", "optim.py", "aug.py"), family="readout", h="c2")
    assert s.admit(store, snap(), three) == "multi_family"
    assert s.admit(store, snap(), three, Operator.DRAFT, strategy=Strategy.MOONSHOT) == ""
    for i, locked in enumerate(("probe.py", "eval/score.py", "train.py::calibration_tiles...budget")):
        assert s.admit(store, snap(), cand(diff="+x\n", paths=("model.py::ViT.forward", locked),
                                           family="readout", h=f"l{i}")) == "surface_locked", locked


def test_a_readout_only_moonshot_is_admitted_for_the_eval_only_lane():
    """Readout ideas are never banned: a readout-only moonshot is admitted (the eval-only lane scores it on trained
    checkpoints), so no researcher pads a readout with a token training loss to dodge a ban (n0585
    second_order_dispersion_readout ran as a full training run); readout + a training family passes too."""
    from loop.spec import EvalOnlyCfg
    s, store = readout_sel(evalonly=EvalOnlyCfg(enabled=True, family="readout"))
    ro = cand(diff="+    cls = self.readout_norm(cls)\n", paths=("model.py::ViT.forward",), family="readout",
              h="m1", claim="multi-depth CLS readout")
    assert s.admit(store, snap(), ro, Operator.DRAFT, strategy=Strategy.MOONSHOT) == ""
    assert s.admit(store, snap(), ro) == ""
    head = cand(diff="+self.head = Head()\n", paths=("model.py::ViT.forward", "optim.py"), family="readout",
                h="m2", claim="train a readout head during pretraining")
    assert s.admit(store, snap(), head, Operator.DRAFT, strategy=Strategy.MOONSHOT) == ""


def test_surface_forbid_applies_to_every_diff_and_config_paths_cannot_reach_probe_roots():
    """The probe/label guard is SURFACE-level: a training-side diff, a crossover, a moonshot and a
    design point are all scanned, and no config path may equal, sit under or above a probe root."""
    cfg = make_cfg(forbid=(r"['\"]probe['\"]", r"_getframe|\bf_locals\b"), forbid_roots=("/data/thunder-data/bach",),
                   families=dict(make_cfg().families, data=FamilyRule(keys=("data.dataset_dir",))))
    s, store = sel(store=root_store(), cfg=cfg)
    bad = 'x = cfg["probe"]["dataset_roots"]'
    assert s.admit(store, snap(), cand(diff=f"+{bad}\n")) == "forbidden_reference"      # optim family
    two = cand(diff="+f = sys._getframe(2).f_locals\n", paths=("optim.py", "aug.py"), h="x2")
    assert s.admit(store, snap(), two, Operator.CROSSOVER) == "forbidden_reference"
    assert s.admit(store, snap(), two, Operator.DRAFT, strategy=Strategy.MOONSHOT) == "forbidden_reference"
    assert s.admit(store, snap(), cand(diff=f"+{bad}\n", family="design", h="x3"), None,
                   strategy=Strategy.DESIGN) == "forbidden_reference"
    for i, root in enumerate(("/data/thunder-data", "/data/thunder-data/bach/sub", "/data/../data/thunder-data/bach",
                              "$PROBE_ROOT/x", "/")):
        c = cand(diff=f"+  dataset_dir: {root}\n", paths=("configs/main.yaml",),
                 cfg_diff={"data.dataset_dir": root}, family="data", h=f"r{i}", claim=f"curate {i}")
        assert s.admit(store, snap(), c) == "forbidden_reference", root
    ok = cand(diff="+  dataset_dir: /data/nanopath_curated\n", paths=("configs/main.yaml",),
              cfg_diff={"data.dataset_dir": "/data/nanopath_curated"}, family="data", h="ok", claim="curate")
    assert s.admit(store, snap(), ok) == ""


def test_genbio_data_or_models_are_disallowed_data():
    """Operator 2026-10-05: GenBio data/models are usage-restricted -- a config path to the GenBio-curated subset,
    or any added line (a comment too) naming GenBio, is rejected disallowed_data on every arm."""
    cfg = make_cfg(families=dict(make_cfg().families, data=FamilyRule(keys=("data.dataset_dir",))))
    s, store = sel(store=root_store(), cfg=cfg)
    gb = "/data/nanopath-curated-genbiopathfm/arrow"
    c = cand(diff=f"+  dataset_dir: {gb}\n", paths=("configs/main.yaml",), cfg_diff={"data.dataset_dir": gb},
             family="data", h="g1", claim="curate")
    assert s.admit(store, snap(), c) == "disallowed_data"
    assert s.admit(store, snap(), c, Operator.DRAFT, strategy=Strategy.MOONSHOT) == "disallowed_data"
    emb = cand(diff="+# distill GenBio-PathFM features\n+lr = 2\n", h="g2")
    assert s.admit(store, snap(), emb) == "disallowed_data"


# ---- compound candidates ----------------------------------------------------------
def _compound_setup(p=0.9):
    cfg = make_cfg(families={"a": FamilyRule(paths=("x*.py",), keys=("a.*",)), "b": FamilyRule(paths=("y*.py",), keys=("b.*",)),
                             "c": FamilyRule(paths=("z*.py",), keys=("c.*",))})
    s, store = sel(store=root_store(), cfg=cfg)
    store.put(node("n1", family="b", status=NodeStatus.CONFIRMED))
    store.set_state("config_diff:n1", {"b.w": 0.05})
    return s, store, snap([post("n1", p=p)])


def test_compound_admitted_tagged_and_rejections():
    from loop.select.intake import compound_sources, compound_split
    s, store, sn = _compound_setup()
    two = cand(paths=("x1.py",), cfg_diff={"b.w": 0.05}, family="a")
    assert s.admit(store, sn, two) == ""
    assert compound_split(s.ctx, store, sn, two) == ("b", "a")
    assert compound_sources(s.ctx, store, sn, two, "b") == ("n1",)      # recorded as stack_from=n1 at intake
    assert compound_sources(s.ctx, store, sn, two, "a") == ()
    assert s.admit(store, sn, cand(paths=("x1.py",), cfg_diff={"b.w": 0.05}, family="b")) == "family_mismatch"
    assert s.admit(store, sn, cand(paths=("x1.py",), cfg_diff={"a.v": 1, "b.w": 0.07}, family="a", h="h3")) \
        == "multi_family"                                    # two families' knobs, not a stack
    knobs = cand(paths=("x1.py",), cfg_diff={"a.v": 1, "b.w": 0.05}, family="a", h="h5")
    assert s.admit(store, sn, knobs) == ""                   # a's knob + b's stackable positive knob: a compound
    low_s, low_store, low_sn = _compound_setup(p=0.5)
    assert low_s.admit(low_store, low_sn, knobs) == "multi_family"
    three = cand(paths=("x1.py", "z1.py"), cfg_diff={"a.v": 1, "b.w": 0.05, "c.u": 2}, family="a", h="h4")
    assert s.admit(store, sn, three) == "multi_family"


# ---- idea banks (loop.select.bank) ----------------------------------------------------------------

def _bank_line(i, *, moon=False, jepa=False, family="optim", **kw):
    import json
    row = {"id": i, "title": f"title {i}", "family": family, "moonshot": moon, "jepa": jepa,
           "idea": f"idea {i}", "why_it_might_work": f"why {i}", "v1_fate": f"fate {i}",
           "why_v1_verdict_may_not_transfer": f"transfer {i}", "source": "v1"}
    row.update(kw)
    return json.dumps(row)


def _write_bank(path, lines):
    import os
    path.write_text("\n".join(lines) + "\n")
    os.utime(path, (path.stat().st_mtime + 5,) * 2)          # a fresh mtime: the cache re-reads


def test_bank_order_dedupe_clean_slate_retry_and_reread(tmp_path, store):
    from loop.select import bank as B
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    _write_bank(a, [_bank_line("a1"), _bank_line("j1", jepa=True), "not json",
                    _bank_line("bad", idea="reached 0.6514 on the panel"),
                    _bank_line("s1", v1_fate="scored +0.012 over base", why_it_might_work="fine"),
                    _bank_line("m1", moon=True, jepa=True)])
    _write_bank(b, [_bank_line("a1", idea="duplicate"), _bank_line("b1")])
    cfg = make_cfg(idea_banks=(str(a), str(b), str(tmp_path / "missing.jsonl")))
    got = B.pending(store, cfg)
    assert [e["id"] for e in got] == ["a1", "j1", "s1", "m1", "b1"]             # file order, first id wins
    assert got[0]["idea"] == "idea a1"
    s1 = next(e for e in got if e["id"] == "s1")
    assert s1["v1_fate"] == "" and s1["why_it_might_work"] == "fine"            # the score is stripped
    assert "0.012" not in B.text(s1) and "What happened before" not in B.text(s1)
    assert {(e["bank_id"], tuple(e["fields"])) for e in store.events("bank_refused")} == {("bad", ("idea",))}
    assert [e["bank_id"] for e in store.events("bank_scrubbed")] == ["s1"]
    focus = make_cfg(idea_banks=cfg.idea_banks, operator_focus="Mess around with JEPA predictors")
    assert [e["id"] for e in B.pending(store, focus)] == ["a1", "j1", "s1", "m1", "b1"]   # a focus privileges nothing
    B.mark(store, "a1", "req-1")
    assert "a1" not in [e["id"] for e in B.pending(store, cfg)]                 # served once
    store.log("research_failed", request_id="req-1", status="failed")
    assert "a1" in [e["id"] for e in B.pending(store, cfg)]                     # a failed attempt retries once
    B.mark(store, "a1", "req-2")
    store.log("research_failed", request_id="req-2", status="failed")
    assert "a1" not in [e["id"] for e in B.pending(store, cfg)]                 # ... but only once
    _write_bank(b, [_bank_line("b1"), _bank_line("b2")])                        # re-read on mtime change
    assert [e["id"] for e in B.pending(store, cfg)][-2:] == ["b1", "b2"]
    import os
    _write_bank(b, [_bank_line("b1"), _bank_line("b2"), _bank_line("anvil2", moon=True, priority=True)])
    os.utime(b, (b.stat().st_mtime + 5,) * 2)
    assert [e["id"] for e in B.pending(store, cfg)][:2] == ["anvil2", "j1"]      # the operator's "draw it soon"
    assert B.pending(store, make_cfg(idea_banks=(str(tmp_path / "missing.jsonl"),))) == []


def test_bank_ideas_fill_their_arm_once_respect_the_moonshot_cap_and_alternate_with_probes(tmp_path):
    from loop.contracts import K_INFLIGHT
    from loop.select import bank as B
    path = tmp_path / "bank.jsonl"
    _write_bank(path, [_bank_line("e1"), _bank_line("e2", jepa=True), _bank_line("x9", family="nope"),
                       _bank_line("m1", moon=True), _bank_line("m2", moon=True, jepa=True)])
    cfg = make_cfg(idea_banks=(str(path),), operator_focus="JEPA first",
                   bandit=BanditCfg(floor=0.1, explore_floor=0.3, moonshot_floor=0.3, moonshot_max_inflight=2))
    store = root_store()
    s, _ = sel(store=store, cfg=cfg, seed=3)
    got = [s.next_request(store, snap()) for _ in range(60)]
    bank = [r for r in got if r.brief_kind == "bank"]
    assert sorted(r.bank_id for r in bank) == ["e1", "e2", "m1", "m2", "x9"]     # each exactly once
    for r in bank:
        moon = r.bank_id.startswith("m")
        assert r.strategy is (Strategy.MOONSHOT if moon else Strategy.EXPLORE) and r.operator is Operator.DRAFT
        assert f"idea {r.bank_id}" in r.brief
    assert next(r for r in bank if r.bank_id.startswith("e")).bank_id == "e1"   # file order: no theme first
    assert next(r for r in bank if r.bank_id.startswith("m")).bank_id == "m1"
    assert next(r for r in bank if r.bank_id == "e1").family_hint == "optim"
    assert next(r for r in bank if r.bank_id == "x9").family_hint == ""          # unknown family: any
    assert any(r.brief_kind != "bank" for r in got)                              # the bandit keeps running
    # the moonshot cap: a moonshot bank idea waits while moonshot_max_inflight moonshots are in flight
    store = root_store()
    store.set_state(K_INFLIGHT, [{"id": f"x{i}", "strategy": "moonshot", "family": "", "parents": ["root"]}
                                 for i in range(2)])
    s, _ = sel(store=store, cfg=cfg, seed=4)
    got = [s.next_request(store, snap()) for _ in range(40)]
    assert all(r.strategy is not Strategy.MOONSHOT for r in got)
    assert {e["id"] for e in B.pending(store, cfg)} >= {"m1", "m2"}
    # probes and bank ideas alternate on the EXPLORE arm: neither starves
    only_explore = make_cfg(idea_banks=(str(path),), bandit=BanditCfg(floor=0.0, explore_floor=1.0,
                                                                       moonshot_floor=0.0))
    store = root_store()
    store._reviews.append(ReviewReport("r", TASK, "cadence", "", (), (),
                                       tuple((None, f"probe {i}") for i in range(3)), {}, n_evals_at=0))
    s, _ = sel(store=store, cfg=only_explore, seed=5)
    kinds = [s.next_request(store, snap()).brief_kind for _ in range(6)]
    assert kinds == ["bank", "probe", "bank", "probe", "bank", "probe"], kinds


def test_moonshot_and_explore_build_on_the_incumbent_and_rotate_carriers_on_a_minority_share():
    """Operator 2026-10-03: "if we have a confirmed current best, focus on comparisons with the confirmed best": a
    moonshot / explore parent is the incumbent except on CARRIER_ROTATE (25%) of requests; those are a Thompson draw
    over each established carrier's recent hit rate (child P(own gain>0) > 0.8); the brief says which."""
    from loop.select import carrier_rates, pick_carrier
    store, posts = FakeStore(), []
    store.put(node("root", parent=None))
    store.put(node("I"))
    for k in range(5):
        store.put(node(f"r{k}", status=NodeStatus.SCREENED_OUT, created_ts=float(k)))
        store.put(node(f"i{k}", status=NodeStatus.SCREENED_OUT, parent="I", created_ts=float(k)))
        posts += [dataclasses.replace(post(i, mean=m), utility=Posterior(m, 0.003, 2))
                  for i, m in ((f"r{k}", 0.01), (f"i{k}", -0.01))]
    sn = snap(posts + [post("I")], incumbent="I")
    assert carrier_rates(store, sn, ["root", "I"]) == {"root": (5, 5), "I": (0, 5)}
    rng = random.Random(0)
    picks = [pick_carrier(store, sn, ["I"], rng)[0] for _ in range(400)]
    assert 0.68 < picks.count("I") / 400 < 0.82                 # the incumbent by default, 25% rotated
    assert "the confirmed incumbent" in pick_carrier(store, sn, ["I"], random.Random(1), rotate=0.0)[1]
    picks = [pick_carrier(store, sn, ["I"], rng, rotate=1.0)[0] for _ in range(200)]
    assert picks.count("root") > 190                      # rotated: 5/5 vs 0/5, the incumbent rarely carries
    s, st = sel(store=store)
    why = pick_carrier(store, sn, ["I"], rng, rotate=1.0)[1]
    req = s._mint(st, sn, strategy=Strategy.EXPLORE, operator=Operator.DRAFT, island="main", parent_ids=("root",),
                  brief_kind="literature", carrier=why)
    assert req.brief.endswith(f"CARRIER: {why}") and "root 5/5, I 0/5" in why


def test_new_screens_prefer_carriers_whose_block_controls_already_run_and_two_at_most_when_scarce(ctx, store):
    """Live 2026-10-02: 13 of 19 controls served exactly one child. A new screen's carrier is one whose newest block
    still has room and already has its controls submitted / running / done (siblings share them); in the driver's
    scarce mode at most 2 carriers are used."""
    from loop import seeds as S
    from loop.contracts import Run, RunStatus, Stage
    from loop.contracts import ROOT_NODE_ID
    from loop.select import live_blocks, pick_carrier, prefer_carriers
    store.ensure_root("c0")
    for nid, parent, st in (("n1", ROOT_NODE_ID, NodeStatus.SCREENING), ("A", ROOT_NODE_ID, NodeStatus.CONFIRMED),
                            ("B", ROOT_NODE_ID, NodeStatus.CONFIRMED), ("n2", "A", NodeStatus.SCREENING)):
        store.put_node(dataclasses.replace(node(nid, status=st, parent=parent), task_id=store.task_id,
                                           created_ts=float(len(store.nodes()))))
    blocks = S.assign_blocks(store.nodes(), ctx.cfg.seeds, [])
    assert live_blocks(store, ctx.cfg) == {}                                # no control launched yet
    for s in S.screen_block(ctx.cfg.seeds, blocks["n2"]):
        store.put_run(Run(id=f"A:parent:{s}", task_id=store.task_id, node_id="A", seed=s, stage=Stage.PARENT,
                          status=RunStatus.RUNNING))
    assert live_blocks(store, ctx.cfg) == {"A": ctx.cfg.seeds.block_children - 1}
    assert prefer_carriers(store, ctx.cfg, [ROOT_NODE_ID, "A", "B"]) == ["A"]
    sn = snap([post("A"), post("B"), post("n1"), post("n2")])
    assert {pick_carrier(store, sn, ["A", "B"], random.Random(i), ctx.cfg, rotate=1.0)[0] for i in range(20)} == {"A"}
    store.set_state("runbook.scarce", {"on": True})
    assert prefer_carriers(store, ctx.cfg, [ROOT_NODE_ID, "A", "B"]) == ["A"]
    store.update_run("A:parent:" + str(S.screen_block(ctx.cfg.seeds, blocks["n2"])[0]), status=RunStatus.VOID)
    assert prefer_carriers(store, ctx.cfg, [ROOT_NODE_ID, "A", "B"]) == [ROOT_NODE_ID, "A"]   # most children first
    assert prefer_carriers(store, None, ["A", "B"]) == ["A", "B"]


def test_an_outage_request_spends_no_bank_idea_no_review_probe_and_no_component_decision(tmp_path, store):
    """Live 2026-10-03: ~1500 requests failed backend_unavailable (account refused). Such a request is no attempt:
    its bank idea and review probe are served again, and its component-gate draws are no audit decision."""
    from loop import components as CG
    from loop.contracts import outage_requests
    from loop.select import PortfolioSelector, bank as B
    a = tmp_path / "a.jsonl"
    _write_bank(a, [_bank_line("a1")])
    cfg = make_cfg(idea_banks=(str(a),))
    for k in (1, 2, 3):                                        # three outages in a row: still pending
        B.mark(store, "a1", f"req-{k}")
        store.log("research_started", request_id=f"req-{k}")
        store.log("research_failed", request_id=f"req-{k}", status="backend_unavailable")
        assert "a1" in [e["id"] for e in B.pending(store, cfg)]
    assert outage_requests(store) == {"req-1", "req-2", "req-3"}
    store.log("research_started", request_id="req-3")          # re-issued (same id): no longer an outage
    assert outage_requests(store) == {"req-1", "req-2"}
    store.set_state("select.probes_used", ["rv-1:0", "rv-1:1"])
    store.log("meta_decision", component="review_probe", decision_id="rv-1:1", payload={"request": "req-1"})
    assert PortfolioSelector.unserve_probe(store, "req-1") == ["rv-1:1"]
    assert store.get_state("select.probes_used") == ["rv-1:0"]
    store.log("component_gate", component="web", decision="research", key="req-1", rate=0.15,
              request="req-1", used=True)
    store.log("component_gate", component="web", decision="research", key="req-9", rate=0.15,
              request="req-9", used=True)
    f = __import__("types").SimpleNamespace(nodes={}, hit=lambda i: None, gpu={})
    assert CG.randomized(store, f, 0.0)["web"][True]["decisions"] == 1
