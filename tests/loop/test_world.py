"""loop.world: shrinkage, BP exactness, cold start, levers, e-values, veto, confirm."""

from __future__ import annotations

import dataclasses
import math
import random

import pytest

from loop.contracts import (PANEL, ROOT_NODE_ID, AxisPrediction, HypothesisKind,
                            HypothesisRecord, HypothesisStatus, LeverTags, Node, NodeStatus,
                            Operator, Run, RunStatus, Stage, Strategy, VoidType)
from loop.world import ComputedWorldModel, link_meta
from loop.world import evalues
from loop.world.evalues import (CAP_META, CAP_TRIAL, CONFIRM_KEY, HYP_KEY, calibrate, fold,
                                lr_evalue, phi, sigma_ucb, trial_lr)
from loop.world.predictive import invert
from loop.world.shrinkage import cold_components, refit_components, tree_bp

BASE = {"alpha": 0.60, "beta": 0.50, "gamma": 0.40}
DF0 = 7          # calibration df of fake_loop: seeds 11..14, 101..104 -> 8 - 1


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def with_cfg(ctx, **sections):
    cfg = ctx.cfg
    for name, kw in sections.items():
        cfg = dataclasses.replace(cfg, **{name: dataclasses.replace(getattr(cfg, name), **kw)})
    return dataclasses.replace(ctx, cfg=cfg)


def panel_of(ctx, axes):
    return sum(ctx.cfg.weights[a] * v for a, v in axes.items())


class Lab:
    """Writes a synthetic ledger: root runs + nodes whose per-axis true effect
    over root is known, with seeded paired noise."""

    def __init__(self, ctx, store, seed=0, noise=True):
        self.ctx, self.store, self.rng, self.noise = ctx, store, random.Random(seed), noise
        self.t = 1000.0
        self.root_scores: dict[int, dict[str, float]] = {}
        store.ensure_root("base")

    def tick(self):
        self.t += 1.0
        return self.t

    def root_at(self, seed, stage=Stage.BASELINE):
        if seed not in self.root_scores:
            jitter = self.rng.gauss(0, 0.01) if self.noise else 0.0
            self.root_scores[seed] = {a: v + jitter for a, v in BASE.items()}
            self.run(ROOT_NODE_ID, seed, stage, self.root_scores[seed])
        return self.root_scores[seed]

    def run(self, nid, seed, stage, scores, status=RunStatus.DONE, void_type=None):
        ts = self.tick()
        self.store.put_run(Run(id=Run.make_id(nid, stage, seed), task_id=self.ctx.task_id,
                               node_id=nid, seed=seed, stage=stage, status=status,
                               void_type=void_type, scores=scores, submitted_ts=ts))

    def node(self, nid, family="optimization", tag="lr", parent=ROOT_NODE_ID,
             status=NodeStatus.SCREENED_OUT, levers=None, confirm_alt=None):
        self.store.put_node(Node(
            id=nid, task_id=self.ctx.task_id, commit=f"c-{nid}", parent_id=parent,
            strategy=Strategy.EXPLORE, operator=Operator.DRAFT,
            family=family, mechanism_tag=tag, diff_hash=f"h-{nid}", status=status,
            levers=LeverTags(levers or {}), confirm_alt=confirm_alt, created_ts=self.tick()))

    def measure(self, nid, effect, seeds, stage=Stage.SCREEN):
        for s in seeds:
            base = self.root_at(s)
            scores = {a: base[a] + effect.get(a, 0.0)
                      + (self.rng.gauss(0, self.ctx.cfg.axis_sd[a]) if self.noise else 0.0)
                      for a in BASE}
            self.run(nid, s, stage, scores)


# ---------------------------------------------------------------------------
# tree BP == brute-force joint Gaussian
# ---------------------------------------------------------------------------

def test_bp_equals_joint_gaussian():
    """Families are the exact joint-Gaussian posterior; a node's eps is the joint posterior of the model WITHOUT
    the node's own observation (the cavity: nodes of one family pool, a node never lifts itself)."""
    tau1 = 0.004
    priors = {"fA": (0.0, 0.006 ** 2), "fB": (-0.02, 0.005 ** 2)}
    obs = {"n1": ("fA", 0.010, 1e-5), "n2": ("fA", 0.004, 2e-5), "n3": ("fA", -0.003, 1.5e-5),
           "n4": ("fB", -0.012, 1e-5), "n5": ("fB", -0.020, 3e-5)}
    fit = tree_bp(obs, priors, tau1)

    def joint(obs, without=None):
        fams = list(priors)
        idx = {v: i for i, v in enumerate(fams + list(obs))}
        k = len(idx)
        J = [[0.0] * k for _ in range(k)]
        h = [0.0] * k

        def edge(a, b, var):            # factor N(a; b, var): (a-b)^2 / var
            i, j = idx[a], idx[b]
            J[i][i] += 1 / var; J[j][j] += 1 / var
            J[i][j] -= 1 / var; J[j][i] -= 1 / var

        for f, (m, v) in priors.items():
            J[idx[f]][idx[f]] += 1 / v
            h[idx[f]] += m / v
        for n, (f, e, v) in obs.items():
            edge(n, f, tau1 ** 2)
            if n != without:
                J[idx[n]][idx[n]] += 1 / v
                h[idx[n]] += e / v
        cov = invert(J)
        return idx, [sum(cov[i][j] * h[j] for j in range(k)) for i in range(k)], cov

    idx, mean, cov = joint(obs)
    for f in priors:
        assert fit.families[f][0] == pytest.approx(mean[idx[f]], abs=1e-12)
        assert fit.families[f][1] == pytest.approx(cov[idx[f]][idx[f]], rel=1e-9)
    for n in obs:
        assert fit.eps[n][0] == pytest.approx(mean[idx[n]], abs=1e-12)
        assert fit.eps[n][1] == pytest.approx(cov[idx[n]][idx[n]], rel=1e-9)
        idx2, mean2, cov2 = joint(obs, without=n)         # the cavity prior: the model without n's own observation
        assert fit.eps_prior[n][0] == pytest.approx(mean2[idx2[n]], abs=1e-12)
        assert fit.eps_prior[n][1] == pytest.approx(cov2[idx2[n]][idx2[n]], rel=1e-9)


# ---------------------------------------------------------------------------
# shrinkage on synthetic data
# ---------------------------------------------------------------------------

def test_shrinkage_recovers_known_effects(ctx, store):
    lab = Lab(ctx, store, seed=1)
    rng = random.Random(7)
    truth_tag = {("optimization", "lr"): {"alpha": 0.010, "beta": 0.008, "gamma": 0.004},
                 ("optimization", "warmup"): {"alpha": 0.002, "beta": 0.000, "gamma": 0.001},
                 ("augmentation", "crop"): {"alpha": -0.006, "beta": -0.004, "gamma": -0.002},
                 ("regularization", "wd"): {"alpha": 0.004, "beta": 0.006, "gamma": 0.008}}
    truth_node, raw = {}, {}
    i = 0
    for _ in range(8):              # as the loop screens: blocks of 4 mixed siblings on their own seeds
        for (fam, tag), eff in truth_tag.items():
            i += 1
            nid = f"n{i}"
            jit = rng.gauss(0, 0.002)
            e = {a: v + jit for a, v in eff.items()}
            truth_node[nid] = panel_of(ctx, e)
            lab.node(nid, fam, tag)
            lab.measure(nid, e, (300 + 2 * ((i - 1) // 4), 301 + 2 * ((i - 1) // 4)))
    snap = ComputedWorldModel(ctx).fit(store)
    assert snap.n_scored == 32
    assert not snap.variance[PANEL].cold                  # DL refit ran (>= refit_min)
    assert store.get_state(f"world.variance.{PANEL}")["n_scored_at_fit"] == 32
    for fam in {f for f, _ in truth_tag}:
        want = [panel_of(ctx, eff) for (f, _), eff in truth_tag.items() if f == fam]
        post = snap.family_means[fam]
        assert abs(post.mean - sum(want) / len(want)) < 3 * post.sd + 3e-3, fam
    # shrunk node estimates beat the raw per-node means
    for nid in truth_node:
        runs = [r for r in store.runs(node_id=nid)]
        raw[nid] = sum(panel_of(ctx, {a: r.scores[a] - lab.root_scores[r.seed][a] for a in BASE})
                       for r in runs) / len(runs)
    rmse = lambda est: math.sqrt(sum((est[k] - truth_node[k]) ** 2 for k in truth_node) / len(truth_node))
    shrunk = {k: snap.nodes[k].utility.mean for k in truth_node}
    assert rmse(shrunk) < rmse(raw)
    # the best family's nodes rank above the worst family's
    assert snap.family_means["regularization"].mean > snap.family_means["augmentation"].mean
    # per-axis posteriors exist for every scored node
    assert set(snap.nodes["n1"].axes) == set(ctx.cfg.axes)


def test_cold_start_architecture_prior_negative(ctx, store):
    lab = Lab(ctx, store)
    wm = ComputedWorldModel(ctx)
    snap = wm.fit(store)
    assert snap.incumbent_id == ROOT_NODE_ID and snap.variance[PANEL].cold
    arch = wm.predictive(snap, family="architecture", tag="deeper", levers=LeverTags())
    opt = wm.predictive(snap, family="optimization", tag="lr", levers=LeverTags())
    assert arch.mean == pytest.approx(-0.02) and arch.mean < 0
    assert opt.mean == 0.0 and opt.sd == pytest.approx(ctx.cfg.prior.tau0)
    # a flat architecture trial is still read as harmful-leaning
    lab.node("a1", "architecture", "deeper")
    lab.measure("a1", {}, (11, 12))
    lab.node("o1", "optimization", "lr")
    lab.measure("o1", {}, (11, 12))
    snap = wm.fit(store)
    assert snap.nodes["a1"].utility.mean < snap.nodes["o1"].utility.mean
    assert snap.family_means["architecture"].mean < -0.005


# ---------------------------------------------------------------------------
# lever trend model
# ---------------------------------------------------------------------------

def test_lever_regression_recovers_train_weaker(ctx, store):
    lab = Lab(ctx, store, seed=3)
    rng = random.Random(11)
    levers = ("effective_lr", "regularization", "capacity", "augmentation_strength")
    for i in range(60):
        vec = {l: rng.choice((-1, 0, 1)) for l in levers}
        # planted: lowering effective LR gains +0.006 panel; everything else inert
        gain = -0.006 * vec["effective_lr"]
        lab.node(f"n{i}", "optimization", f"t{i % 4}", levers=vec)
        lab.measure(f"n{i}", {a: gain for a in BASE}, (11, 12))
    snap = ComputedWorldModel(with_cfg(ctx, prior={"wm_min_nodes": 20})).fit(store)
    eff = {(e.lever, e.target): e for e in snap.levers}
    lr = eff[("effective_lr", PANEL)]
    assert lr.mean < 0 and lr.p_positive < 0.01        # "train weaker" is a number
    assert abs(lr.mean + 0.006) < 3 * lr.se and 0 < lr.se < 0.002
    assert lr.n_nonzero > 30
    for other in ("regularization", "capacity"):
        e = eff[(other, PANEL)]
        assert abs(e.mean) < 3 * e.se + 1e-4


# ---------------------------------------------------------------------------
# e-values
# ---------------------------------------------------------------------------

def test_calibrator_integrates_to_one():
    for kappa in (0.25, 0.5, 0.8):
        # substitute p = u**(1/kappa) to remove the singularity at 0
        n = 20000
        total = 0.0
        for i in range(n):
            u = (i + 0.5) / n
            p = u ** (1 / kappa)
            total += calibrate(p, kappa) * (1 / kappa) * u ** (1 / kappa - 1) / n
        assert total == pytest.approx(1.0, abs=1e-6)


def test_single_trial_evidence_is_bounded():
    """The retired calibrator turned one z = -10.6 (a broken readout node, -0.021 vs sd 0.002) into
    E_against = 1e15; a bounded LR moves E by at most the cap, whatever the tail."""
    z = -0.02141 / 0.00202
    assert calibrate(phi(z), 0.5) > 1e12                          # the bug, reproduced
    assert CAP_META <= CAP_TRIAL <= 50 and CAP_META ** 2 >= 10 > CAP_META
    for x in (-1.0, -0.0214, -0.003, 0.0, 0.003, 0.0214, 1.0, math.inf, math.nan):
        for cap in (CAP_META, CAP_TRIAL):
            ef, ea = trial_lr(x, 0.002 ** 2, 0.003, cap)
            assert 0.0 <= ef <= cap and 0.0 <= ea <= cap
    ef, ea = trial_lr(0.0015, 0.002 ** 2, 0.003, CAP_META)         # halfway: no evidence either way
    assert ef == pytest.approx(1.0) and ea == pytest.approx(1.0)


def _sim(mu, n_rep, rng, *, cap, max_trials, d1=0.003, sd=0.002, alpha=0.1):
    out = {s: 0 for s in HypothesisStatus}
    for _ in range(n_rep):
        pairs = [trial_lr(rng.gauss(mu, sd), sd * sd, d1, cap) for _ in range(max_trials)]
        out[fold(pairs, alpha, max_trials)[3]] += 1
    return {s: k / n_rep for s, k in out.items()}


@pytest.mark.parametrize("cap,max_trials", [(CAP_META, 6), (CAP_TRIAL, 1)])
def test_hypothesis_error_rates_and_power(cap, max_trials):
    """Null mu = 0: false SUPPORT <= alpha; null mu = delta_1 (the predicted size is real): false
    REFUTE <= alpha; a planted effect above delta_1 is supported."""
    rng, alpha = random.Random(11), 0.1
    assert _sim(0.0, 4000, rng, cap=cap, max_trials=max_trials)[HypothesisStatus.SUPPORTED] <= alpha
    assert _sim(0.003, 4000, rng, cap=cap, max_trials=max_trials)[HypothesisStatus.REFUTED] <= alpha
    assert _sim(-0.01, 4000, rng, cap=cap, max_trials=max_trials)[HypothesisStatus.SUPPORTED] == 0.0
    if max_trials > 1:
        assert _sim(0.005, 2000, rng, cap=cap, max_trials=max_trials)[HypothesisStatus.SUPPORTED] > 0.8
        assert _sim(-0.002, 2000, rng, cap=cap, max_trials=max_trials)[HypothesisStatus.REFUTED] > 0.8
    assert fold([], alpha, max_trials)[3] == HypothesisStatus.UNTESTED


def test_one_catastrophic_trial_cannot_close_a_meta_hypothesis(ctx, store):
    lab = Lab(ctx, store, noise=False)
    h = hypothesis(ctx, "ro", lab.tick(), {"readout": 1})
    store.put_hypothesis(h)
    lab.node("broken", levers={"readout": 1}, status=NodeStatus.VETOED)
    lab.measure("broken", {a: -0.05 for a in BASE}, (11, 12))
    store.link("ro", "broken", sign=1, auto=False)               # a lever link alone is no evidence
    ev = ComputedWorldModel(ctx).fit(store).hypotheses["ro"]
    assert ev.n_trials == 1 and ev.status == HypothesisStatus.TESTING
    assert ev.e_against == pytest.approx(CAP_META) and ev.e_value < 1


def test_shared_control_siblings_do_not_multiply(ctx, tmp_path):
    """10 NULL siblings paired against one pair of control runs that is 0.004 low: every raw delta
    carries +0.004, so treating them as independent supports a false meta-hypothesis almost every
    time; with the causal control correction it (nearly) never does."""
    from loop.store import SqliteStore
    # the meta's between-implementation spread is the DL tau_node; 10 null siblings give it nothing to estimate, so
    # pin it at the old tau0 level (the test is about the control correction, not the spread estimate)
    ctx = with_cfg(ctx, prior={"tau_floor": ctx.cfg.prior.tau0})
    naive = fixed = 0
    reps = 20
    for k in range(reps):
        st = SqliteStore(tmp_path / f"s{k}.db", ctx.task_id, weights=ctx.cfg.weights)
        st.put_hypothesis(hypothesis(ctx, "m", 0.0))
        effects = {f"n{i}": 0.0 for i in range(10)}
        for nid in effects:
            st.link("m", nid, sign=1, auto=False)
        snap = shared_block(ctx, st, effects, 0.004, seed=100 + k)
        fixed += snap.hypotheses["m"].status == HypothesisStatus.SUPPORTED
        rows = st.get_state(HYP_KEY)["m"]
        assert all(max(r[1], r[2]) <= CAP_META for r in rows)
        sig = snap.sigma[PANEL]
        raw = [panel_of(ctx, {a: sum(r.scores[a] for r in st.runs(node_id=n)) / 2
                              - sum(r.scores[a] for r in st.runs(node_id=ROOT_NODE_ID)) / 2
                              for a in BASE}) for n in effects]
        pairs = [trial_lr(x, sig * sig / 2, rows[0][5], CAP_META) for x in raw]   # independent-sibling view
        naive += fold(pairs, ctx.cfg.hypotheses.alpha, ctx.cfg.hypotheses.max_trials)[3] == \
            HypothesisStatus.SUPPORTED
        st.close()
    assert naive / reps > 0.5
    assert fixed / reps <= 0.1


def test_migration_rederives_legacy_book_and_logs_changes(ctx, store):
    lab = Lab(ctx, store, noise=False)
    h = hypothesis(ctx, "m", lab.tick(), {"readout": 1})
    store.put_hypothesis(h)
    lab.node("broken", levers={"readout": 1}, status=NodeStatus.VETOED)
    lab.measure("broken", {a: -0.05 for a in BASE}, (11, 12))
    store.link("m", "broken", sign=1, auto=False)
    store.set_state(evalues.LEGACY_HYP_KEY, {"m": [["broken", 0.5, 1.06e15]]})   # the old verdict
    wm = ComputedWorldModel(ctx)
    ev = wm.fit(store).hypotheses["m"]
    assert ev.status == HypothesisStatus.TESTING and ev.e_against <= CAP_META
    logged = store.events("hypotheses_rederived")
    assert len(logged) == 1 and logged[0]["changed"]["m"]["old"] == "refuted"
    assert logged[0]["changed"]["m"]["new"] == "testing"
    wm.fit(store)
    assert len(store.events("hypotheses_rederived")) == 1          # once
    assert store.get_state(evalues.LEGACY_HYP_KEY) == {"m": [["broken", 0.5, 1.06e15]]}   # kept


def hypothesis(ctx, hid, created_ts, levers=None, kind=HypothesisKind.META, sign=1):
    return HypothesisRecord(
        id=hid, task_id=ctx.task_id, kind=kind, claim="model is over-regularized",
        mechanism_tag="train_weaker", family="optimization",
        predictions={PANEL: AxisPrediction(sign=sign, size=0.005)},
        falsification="weaker-training trials do not gain", levers=LeverTags(levers or {}),
        source="review", created_ts=created_ts)


def test_meta_hypothesis_linking_and_evidence(ctx, store):
    lab = Lab(ctx, store, seed=5)
    lab.node("early", levers={"effective_lr": -1})              # before the review
    lab.measure("early", {a: 0.01 for a in BASE}, (11, 12))
    h = hypothesis(ctx, "meta1", lab.tick(), {"effective_lr": -1, "regularization": -1})
    store.put_hypothesis(h)
    specs = {"w1": {"effective_lr": -1}, "w2": {"regularization": -1},
             "w3": {"effective_lr": -1, "regularization": -1}, "s1": {"effective_lr": 1},
             "cap": {"capacity": 1}}
    for nid, lv in specs.items():
        lab.node(nid, levers=lv)
        gain = 0.012 if nid.startswith("w") else -0.012 if nid == "s1" else 0.0
        lab.measure(nid, {a: gain for a in BASE}, (11, 12, 13, 14))
    linked = dict(link_meta(store, h))
    assert linked == {"w1": 1, "w2": 1, "w3": 1, "s1": -1}
    assert link_meta(store, h) == []                            # idempotent
    ev = ComputedWorldModel(ctx).fit(store).hypotheses["meta1"]
    assert ev.status == HypothesisStatus.UNTESTED and not ev.trial_ids   # lever links are no evidence (rv-36-m3)
    for nid, sign in linked.items():                             # explicit links (its probes' nodes) are
        store.link("meta1", nid, sign=sign, auto=False)
    snap = ComputedWorldModel(ctx).fit(store)
    ev = snap.hypotheses["meta1"]
    assert "early" not in ev.trial_ids                         # pre-registration
    assert ev.status == HypothesisStatus.SUPPORTED and ev.e_value >= 1 / ctx.cfg.hypotheses.alpha


# ---------------------------------------------------------------------------
# veto and incumbent
# ---------------------------------------------------------------------------

def test_veto_fires_on_harmful_axis_despite_panel_gain(ctx, store):
    lab = Lab(ctx, store, noise=False)
    harm = {"alpha": 0.03, "beta": 0.02, "gamma": -0.035}
    assert panel_of(ctx, harm) > 0
    lab.node("harm", status=NodeStatus.CONFIRMED)
    lab.measure("harm", harm, (11, 12, 13, 14))
    lab.node("good", status=NodeStatus.CONFIRMED, tag="wd", family="regularization")
    lab.measure("good", {a: 0.004 for a in BASE}, (11, 12, 13, 14))
    for nid, eff in (("harm", harm), ("good", {a: 0.004 for a in BASE})):
        lab.measure(nid, eff, (101, 102), stage=Stage.CONFIRM)
    snap = ComputedWorldModel(ctx).fit(store)
    np_ = snap.nodes["harm"]
    assert np_.p_positive > 0.9 and np_.vetoed and np_.veto_axes == ("gamma",)
    assert not snap.nodes["good"].vetoed
    assert snap.incumbent_id == "good"                         # vetoed never incumbent
    assert snap.nodes["harm"].p_beats_incumbent > 0.5


# ---------------------------------------------------------------------------
# confirmation e-value
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("screen_gain", [0.05, -0.05])
def test_confirm_evalue_ignores_screen_seeds(ctx, store, screen_gain):
    lab = Lab(ctx, store, noise=False)
    lab.node("c", status=NodeStatus.CONFIRMING, confirm_alt=0.004)
    lab.measure("c", {a: screen_gain for a in BASE}, (11, 12))
    for s in (101, 102, 103):
        lab.root_at(s)
    lab.measure("c", {a: 0.004 for a in BASE}, (101, 102, 103), stage=Stage.CONFIRM)
    # a void confirm run is never a measurement
    lab.run("c", 104, Stage.CONFIRM, {a: 9.0 for a in BASE}, status=RunStatus.VOID,
            void_type=VoidType.EVAL)
    wm = ComputedWorldModel(ctx)
    snap = wm.fit(store)
    e, n = wm.confirm_evalue(snap, "c", store)
    assert n == 3
    sig = sigma_ucb(ctx.cfg.panel_sd, DF0)
    assert sig > ctx.cfg.panel_sd                               # upper bound, not the estimate
    assert store.get_state(CONFIRM_KEY)["c"] == pytest.approx([0.004, sig])
    assert e == pytest.approx(lr_evalue([0.004] * 3, 0.004, sig))
    est = snap.nodes["c"].confirm_estimate
    assert est is not None and est.n == 3 and 0 < est.mean < 0.004 + 1e-9
    # selection posterior still uses screen seeds only
    assert snap.nodes["c"].utility.n == 2
    assert (snap.nodes["c"].utility.mean > 0) == (screen_gain > 0)


def test_confirm_paired_vs_non_root_parent(ctx, store):
    lab = Lab(ctx, store, noise=False)
    lab.node("p", status=NodeStatus.CONFIRMED)
    lab.measure("p", {a: 0.01 for a in BASE}, (11, 12))
    lab.node("k", parent="p", status=NodeStatus.CONFIRMING, confirm_alt=0.002)
    lab.measure("p", {a: 0.01 for a in BASE}, (21, 22), stage=Stage.PARENT)   # k's own block
    lab.measure("k", {a: 0.013 for a in BASE}, (21, 22))
    for s in (101, 102):
        lab.root_at(s)
        lab.measure("p", {a: 0.01 for a in BASE}, (s,), stage=Stage.PARENT)
        lab.measure("k", {a: 0.012 for a in BASE}, (s,), stage=Stage.CONFIRM)
    wm = ComputedWorldModel(ctx)
    snap = wm.fit(store)
    e, n = wm.confirm_evalue(snap, "k", store)
    sig = sigma_ucb(ctx.cfg.panel_sd, DF0)
    assert n == 2 and e == pytest.approx(lr_evalue([0.002, 0.002], 0.002, sig))
    # the child's increment is paired against the parent's own control runs, and its U
    # is chained through the parent's
    assert snap.nodes["k"].increment.mean == pytest.approx(0.003, abs=1e-3)
    assert snap.nodes["k"].utility.mean > snap.nodes["p"].utility.mean


def test_noise_refit_tightens_sigma(ctx, store):
    ctx = with_cfg(ctx, prior={"noise_min_df": 4})
    lab = Lab(ctx, store, seed=9)
    for i in range(5):
        lab.node(f"n{i}")
        # true within-node sd far below the declared 0.004 panel sd: runs are exact
        for s in (11, 12):
            base = lab.root_at(s)
            lab.run(f"n{i}", s, Stage.SCREEN, {a: base[a] + 0.001 for a in BASE})
    snap = ComputedWorldModel(ctx).fit(store)
    assert snap.sigma[PANEL] < ctx.cfg.panel_sd
    assert store.get_state("world.noise")[PANEL]["df"] == 5


# ---------------------------------------------------------------------------
# review fixes: frozen confirmation (4), lever residuals (6), append-only
# hypothesis evidence (8), registered levers only (9), confirm-only incumbent
# (10), DL fallback (12)
# ---------------------------------------------------------------------------

def test_confirm_sigma_and_delta_frozen_at_confirm_start(ctx, store):
    ctx = with_cfg(ctx, prior={"noise_min_df": 4})
    lab = Lab(ctx, store, noise=False)
    lab.node("c", status=NodeStatus.CONFIRMING)                 # no driver-frozen confirm_alt
    lab.measure("c", {a: 0.004 for a in BASE}, (11, 12))
    wm = ComputedWorldModel(ctx)
    snap = wm.fit(store)                                         # confirm start: freeze
    d1, sig = store.get_state(CONFIRM_KEY)["c"]
    assert sig == pytest.approx(sigma_ucb(ctx.cfg.panel_sd, DF0))
    assert d1 >= ctx.cfg.promotion.delta_min_sigmas * ctx.cfg.panel_sd
    # the node's own confirm runs never feed sigma, and exact replicates elsewhere
    # shrink the refit sigma -- neither moves the frozen values or E
    lab.measure("c", {a: 0.004 for a in BASE}, (101, 102), stage=Stage.CONFIRM)
    for i in range(6):
        lab.node(f"x{i}")
        lab.measure(f"x{i}", {}, (11, 12, 13))
    snap = wm.fit(store)
    assert snap.sigma[PANEL] < 0.7 * ctx.cfg.panel_sd
    assert store.get_state(CONFIRM_KEY)["c"] == [d1, sig]
    e, n = wm.confirm_evalue(snap, "c", store)
    assert n == 2 and e == pytest.approx(lr_evalue([0.004, 0.004], d1, sig))


def test_sigma_ucb_is_conservative_and_tightens_with_df():
    assert sigma_ucb(1.0, 4) > sigma_ucb(1.0, 40) > sigma_ucb(1.0, 4000) > 1.0
    assert sigma_ucb(1.0, 4000) == pytest.approx(1.0, abs=0.03)


def test_hypothesis_evidence_append_only_never_revoked(ctx, store):
    lab = Lab(ctx, store, noise=False)
    h = hypothesis(ctx, "m", lab.tick(), {"effective_lr": -1})
    store.put_hypothesis(h)
    for nid in ("w1", "w2"):
        lab.node(nid, levers={"effective_lr": -1})
        lab.measure(nid, {a: 0.012 for a in BASE}, (11, 12))
    lab.node("pending", levers={"effective_lr": -1}, status=NodeStatus.SCREENING)
    lab.measure("pending", {a: 0.012 for a in BASE}, (11,))
    for nid in ("w1", "w2", "pending"):
        store.link("m", nid, sign=1, auto=False)
    wm = ComputedWorldModel(ctx)
    ev = wm.fit(store).hypotheses["m"]
    assert ev.status == HypothesisStatus.SUPPORTED and "pending" not in ev.trial_ids
    stored = store.get_state(HYP_KEY)["m"]
    # later, contrary data on the SAME trials (extra promote seeds) and a refuting
    # new trial: recorded e-values do not change and the verdict stands
    for nid in ("w1", "w2"):
        lab.measure(nid, {a: -0.05 for a in BASE}, (13, 14), stage=Stage.PROMOTE)
    lab.node("s1", levers={"effective_lr": -1})
    lab.measure("s1", {a: -0.05 for a in BASE}, (11, 12))
    store.link("m", "s1", sign=1, auto=False)
    ev2 = wm.fit(store).hypotheses["m"]
    assert store.get_state(HYP_KEY)["m"][:len(stored)] == stored
    assert (ev2.status, ev2.e_value, ev2.trial_ids) == (ev.status, ev.e_value, ev.trial_ids)


@pytest.mark.parametrize("gain,status", [(0.02, HypothesisStatus.SUPPORTED),
                                         (-0.02, HypothesisStatus.REFUTED),
                                         (0.0025, HypothesisStatus.INCONCLUSIVE),   # delta_1 / 2
                                         (0.0005, HypothesisStatus.REFUTED)])       # << predicted 0.005
def test_trial_hypothesis_closes_on_its_one_trial(ctx, store, gain, status):
    lab = Lab(ctx, store, noise=False)
    h = hypothesis(ctx, "t", lab.tick(), kind=HypothesisKind.TRIAL)
    store.put_hypothesis(h)
    lab.node("n")
    store.link("t", "n", sign=1, auto=False)
    lab.measure("n", {a: gain for a in BASE}, (11, 12, 13, 14))
    ev = ComputedWorldModel(ctx).fit(store).hypotheses["t"]
    assert ev.status == status and ev.n_trials == 1


def test_incumbent_and_p_beats_use_confirm_only_estimates(ctx, store):
    lab = Lab(ctx, store, noise=False)
    # "lucky": great selection seeds, mediocre confirm; "solid": the reverse
    lab.node("lucky", status=NodeStatus.CONFIRMED)
    lab.measure("lucky", {a: 0.03 for a in BASE}, (11, 12, 13, 14))
    lab.measure("lucky", {a: 0.002 for a in BASE}, (101, 102, 103), stage=Stage.CONFIRM)
    lab.node("solid", status=NodeStatus.CONFIRMED, family="regularization", tag="wd")
    lab.measure("solid", {a: 0.006 for a in BASE}, (11, 12, 13, 14))
    lab.measure("solid", {a: 0.006 for a in BASE}, (101, 102, 103), stage=Stage.CONFIRM)
    lab.node("noconf", status=NodeStatus.CONFIRMED, family="augmentation", tag="crop")
    lab.measure("noconf", {a: 0.05 for a in BASE}, (11, 12, 13, 14))
    snap = ComputedWorldModel(ctx).fit(store)
    assert snap.nodes["lucky"].utility.mean > snap.nodes["solid"].utility.mean
    assert snap.incumbent_id == "solid"                         # confirm-only choice
    assert snap.nodes["lucky"].p_beats_incumbent < 0.5          # compared on confirm seeds


def test_dl_unestimable_level_keeps_current_tau(ctx):
    obs = {f"n{i}": (f"f{i}", 0.001 * i, 1e-5) for i in range(4)}                   # 1 per family
    cold = cold_components(ctx)
    got = refit_components(obs, ctx, 4, cold)
    assert got.tau_node == pytest.approx(ctx.cfg.prior.tau0)    # cold -> tau0, not the floor
    warm = dataclasses.replace(cold, tau_node=0.003, cold=False)
    assert refit_components(obs, ctx, 4, warm).tau_node == pytest.approx(0.003)


# ---------------------------------------------------------------------------
# shared control runs (siblings paired against the same control run)
# ---------------------------------------------------------------------------

def shared_block(ctx, store, effects, offset, seed, seeds=(11, 12)):
    """``effects``: node -> true panel effect (on every axis); siblings are distinct ideas (own tag,
    3 families). All pair against the same root runs, each run noisy (sd/sqrt2 per run) and planted
    ``offset`` LOW, i.e. every paired delta of the block carries +offset."""
    lab, rng = Lab(ctx, store, seed=seed, noise=False), random.Random(seed)
    half = {a: ctx.cfg.axis_sd[a] / math.sqrt(2) for a in BASE}
    for s in seeds:
        lab.root_scores[s] = {a: v - offset + rng.gauss(0, half[a]) for a, v in BASE.items()}
        lab.run(ROOT_NODE_ID, s, Stage.BASELINE, lab.root_scores[s])
    for i, (nid, eff) in enumerate(effects.items()):
        lab.node(nid, ("optimization", "regularization", "augmentation")[i % 3], f"t{i}")
        for s in seeds:
            lab.run(nid, s, Stage.SCREEN, {a: v + eff + rng.gauss(0, half[a]) for a, v in BASE.items()})
    return ComputedWorldModel(ctx).fit(store)


def test_shared_control_offset_is_not_evidence(ctx, tmp_path):
    """10 null siblings, one pair of control runs 0.003 low: raw deltas are almost all
    positive, but P(U>0) stays ~uniform instead of 9/10 > 0.5."""
    from loop.store import SqliteStore
    ps, raw_pos = [], 0
    for k in range(30):
        st = SqliteStore(tmp_path / f"r{k}.db", ctx.task_id, weights=ctx.cfg.weights)
        snap = shared_block(ctx, st, {f"n{i}": 0.0 for i in range(10)}, 0.003, seed=k)
        ps += [snap.nodes[f"n{i}"].p_positive for i in range(10)]
        raw_pos += sum(panel_of(ctx, {a: sum(r.scores[a] for r in st.runs(node_id=f"n{i}"))
                                      - sum(r.scores[a] for r in st.runs(node_id=ROOT_NODE_ID))
                                      for a in BASE}) > 0 for i in range(10))
    assert raw_pos / len(ps) > 0.8                      # the offset is real in the raw deltas
    assert 0.3 < sum(p > 0.5 for p in ps) / len(ps) < 0.7
    assert 0.35 < sum(ps) / len(ps) < 0.65


def test_real_effect_among_null_siblings_is_recovered(ctx, store):
    effects = {f"n{i}": 0.0 for i in range(9)} | {"win": 0.012}
    snap = shared_block(ctx, store, effects, 0.003, seed=5)
    win = snap.nodes["win"]
    assert win.p_positive > 0.95
    assert win.increment.mean > max(snap.nodes[n].increment.mean for n in effects if n != "win")
    assert abs(win.increment.mean - 0.012) < 3 * win.increment.sd


def test_shared_control_leave_one_out_and_single_child(ctx):
    from loop.world import noise
    sig, half = {t: 0.004 for t in noise.targets(ctx)}, 0.004 ** 2 / 2
    tau = ctx.cfg.prior.tau0                    # < noise_min_df sibling df: the declared tau0

    def nd(nid, deltas, control="root:baseline:11"):
        n = Node(id=nid, task_id=ctx.task_id, commit="c", parent_id=ROOT_NODE_ID,
                 strategy=Strategy.EXPLORE, operator=Operator.DRAFT, family="optimization",
                 mechanism_tag="lr", diff_hash=nid, status=NodeStatus.SCREENED_OUT)
        obs = [noise.Obs(Stage.SCREEN, control, {PANEL: d}) for d in deltas]
        return n, noise.NodeData(n, obs, {}, 0.0)

    rows = {"a": nd("a", [0.010]), "b": nd("b", [0.002]), "c": nd("c", [-0.004]),
            "solo": nd("solo", [0.005], "p:parent:7")}
    led = noise.correct(ctx, noise.Ledger({k: v[0] for k, v in rows.items()},
                                          {k: v[1] for k, v in rows.items()}), sig)
    s = half + tau ** 2
    for nid, own, others in (("a", 0.010, (0.002, -0.004)), ("b", 0.002, (0.010, -0.004)),
                             ("c", -0.004, (0.010, 0.002))):
        p = 1 / half + 2 / s
        b = sum(others) / s / p
        x, v = led.data[nid].selection[PANEL][0]
        assert x == pytest.approx(own - b) and v == pytest.approx(half + 1 / p)
    # LOO: a child's own delta never moves its own offset estimate
    rows2 = dict(rows, a=nd("a", [0.050]))
    led2 = noise.correct(ctx, noise.Ledger({k: v[0] for k, v in rows2.items()},
                                           {k: v[1] for k, v in rows2.items()}), sig)
    assert led2.data["a"].selection[PANEL][0][0] - 0.050 == pytest.approx(
        led.data["a"].selection[PANEL][0][0] - 0.010)
    # a single child on its control: b_hat = 0 with the full prior variance -> sigma^2
    assert led.data["solo"].selection[PANEL][0] == pytest.approx((0.005, 0.004 ** 2))


def test_a_broken_or_vetoed_sibling_never_estimates_the_control_offset(ctx):
    """Live: n0429 (-0.066, collapsed) on control 412 read n0430's raw +0.0014 as +0.013; vetoed n0437
    (-0.015) lifted n0446 likewise. A sibling beyond OUTLIER_SIGMAS * sigma, or a harm-VETOED one, is left
    out of the offset estimate: the node keeps the offset its other siblings give."""
    from loop.world import noise
    sig = {t: 0.003 for t in noise.targets(ctx)}

    def nd(nid, d, status=NodeStatus.SCREENED_OUT):
        n = Node(id=nid, task_id=ctx.task_id, commit="c", parent_id=ROOT_NODE_ID, strategy=Strategy.EXPLORE,
                 operator=Operator.DRAFT, family="optimization", mechanism_tag=nid, diff_hash=nid, status=status)
        return n, noise.NodeData(n, [noise.Obs(Stage.SCREEN, "root:baseline:412", {PANEL: d})], {}, 0.0)

    def corrected(rows):
        led = noise.correct(ctx, noise.Ledger({k: v[0] for k, v in rows.items()},
                                              {k: v[1] for k, v in rows.items()}), sig)
        return led.data["me"].selection[PANEL][0][0]

    base = {"me": nd("me", 0.0014), "a": nd("a", 0.0026), "b": nd("b", -0.0063)}
    clean = corrected(base)
    assert corrected(base | {"broken": nd("broken", -0.066)}) == pytest.approx(clean)
    assert corrected(base | {"veto": nd("veto", -0.008, NodeStatus.VETOED)}) == pytest.approx(clean)
    assert corrected(base | {"ok": nd("ok", -0.008)}) > clean + 1e-4       # an ordinary sibling still counts
    rows = base | {"broken": nd("broken", -0.066)}
    rise = noise.offset_rise(ctx, noise.Ledger({k: v[0] for k, v in rows.items()},
                                               {k: v[1] for k, v in rows.items()}), sig)
    assert rise["me"] > 0.005 and rise["broken"] == pytest.approx(0.0)   # the one-time regate's selector


def test_near_duplicate_siblings_never_estimate_each_others_control_offset(ctx, store):
    """Live: n0431 (slide_balanced_sampling) on n0436's (slide_balanced_resample) control read it as lucky and
    shrank n0436's gain to ~0. Siblings with the same normalized tag, or a mechanism cosine >= mech_dup_cos,
    are twins: neither estimates the other's offset; an unrelated sibling still does."""
    from loop.world import noise
    sig = {t: 0.003 for t in noise.targets(ctx)}
    rows = {}
    for nid, tag, d in (("me", "whitened_jepa_target", 0.004), ("twin", "Whiten_JEPA_targets", 0.006),
                        ("other", "lower_lr", 0.001)):
        n = Node(id=nid, task_id=ctx.task_id, commit="c", parent_id=ROOT_NODE_ID, strategy=Strategy.EXPLORE,
                 operator=Operator.DRAFT, family="optimization", mechanism_tag=tag, diff_hash=nid,
                 status=NodeStatus.SCREENED_OUT)
        store.put_node(n)
        rows[nid] = (n, noise.NodeData(n, [noise.Obs(Stage.SCREEN, "root:baseline:413", {PANEL: d})], {}, 0.0))
    nodes, data = {k: v[0] for k, v in rows.items()}, {k: v[1] for k, v in rows.items()}
    tw = noise.twins(ctx, store, nodes, data)
    assert tw.get("me") == {"twin"} and tw.get("twin") == {"me"} and "other" not in tw
    led = noise.correct(ctx, noise.Ledger(nodes, data, twins=tw), sig)
    alone = noise.correct(ctx, noise.Ledger({k: nodes[k] for k in ("me", "other")},
                                            {k: data[k] for k in ("me", "other")}), sig)
    assert led.data["me"].selection[PANEL][0] == pytest.approx(alone.data["me"].selection[PANEL][0])


# ---------------------------------------------------------------------------
# meta-hypothesis links (explicit > mechanistic > lever), the trial window, multiplicity
# ---------------------------------------------------------------------------

def _keyed_meta(ctx, hid, created_ts, levers, claim):
    return dataclasses.replace(hypothesis(ctx, hid, created_ts, levers), claim=claim,
                               falsification="the named knobs do not gain")


def test_explicit_and_mechanistic_links_beat_lever_links(ctx, store):
    lab = Lab(ctx, store, noise=False)
    h = _keyed_meta(ctx, "rv-1-x-m4", lab.tick(), {"effective_lr": -1, "regularization": 1},
                    "milder local-crop scale plus weaker KDE are additive")
    lab.node("design_pt", levers={"effective_lr": -1})                    # a factorial point: never linked
    store.update_node("design_pt", strategy=Strategy.DESIGN, operator=None)
    store.set_state("config_diff:design_pt", {"data.local_crop_scale": [0.16, 0.35]})
    store.put_hypothesis(h)
    lab.node("lever_only", levers={"effective_lr": -1})                   # generic LR node
    lab.node("mech_leaf", levers={"capacity": 1})                         # names local_crop_scale
    store.set_state("config_diff:mech_leaf", {"data.local_crop_scale": [0.16, 0.35]})
    lab.node("mech_stem")                                                 # "local-crop" -> local_crop_*
    store.set_state("config_diff:mech_stem", {"data.local_crop_count": 4})
    lab.node("too_generic")                                               # "KDE" alone names no kde_* key
    store.set_state("config_diff:too_generic", {"dino.kde_concentration": 6.667})
    lab.node("probe")                                                     # made from its probe brief
    store.link(h.id, "probe", sign=1, auto=False)
    lab.node("names_it")                                                  # its hypothesis names the meta
    store.put_hypothesis(dataclasses.replace(
        hypothesis(ctx, "h-names", lab.tick(), kind=HypothesisKind.TRIAL),
        claim="stacking both confirms rv-1-x-m4"))
    store.update_node("names_it", hypothesis_id="h-names")
    lab.node("near_miss")                                                 # rv-1-x-m41 is another meta
    store.put_hypothesis(dataclasses.replace(
        hypothesis(ctx, "h-miss", lab.tick(), kind=HypothesisKind.TRIAL), claim="tests rv-1-x-m41"))
    store.update_node("near_miss", hypothesis_id="h-miss")
    tiers = {nid: tier for nid, _, tier in evalues.meta_links(store, h, evalues.LinkIndex(store))}
    assert tiers == {"probe": evalues.EXPLICIT, "names_it": evalues.EXPLICIT,
                     "mech_leaf": evalues.MECHANISTIC, "mech_stem": evalues.MECHANISTIC}
    keys = set(store.get_state(evalues.META_KEYS)[h.id])
    assert {"local_crop_scale", "local_crop_count"} <= keys and "kde_concentration" not in keys
    # no explicit or mechanistic link at all: lever links are the fallback
    g = _keyed_meta(ctx, "rv-1-x-m5", h.created_ts, {"effective_lr": -1}, "train more gently")
    store.put_hypothesis(g)
    assert [(n, t) for n, _, t in evalues.meta_links(store, g, evalues.LinkIndex(store))] == \
        [("lever_only", evalues.LEVER)]
    # a one-word key is named only by its dotted key: "LR" in prose is a lever, not dino.lr
    assert evalues.key_names("LR barely anneals; raise dino.lr_min", {"dino.lr", "dino.lr_min"}) \
        == {"lr_min"}
    assert evalues.key_names("effective_lr- via data_curation+", ()) == set()        # lever names


def test_mechanistic_link_on_code_identifiers(ctx, store, tmp_path):
    lab = Lab(ctx, store, noise=False)
    h = _keyed_meta(ctx, "m", lab.tick(), {"effective_lr": -1},
                    "re-key the schedules to sample_fraction so they anneal fully")
    store.put_hypothesis(h)
    diff = tmp_path / "rekey.diff"
    diff.write_text("--- a/train.py\n+++ b/train.py\n@@ -1 +1 @@\n-    frac = flops / max_flops\n"
                    "+    frac = sample_fraction(examples_seen)\n")
    lab.node("rekey")
    store.update_node("rekey", diff_path=str(diff))
    lab.node("lr_knob", levers={"effective_lr": -1})
    assert [(n, t) for n, _, t in evalues.meta_links(store, h, evalues.LinkIndex(store))] == \
        [("rekey", evalues.MECHANISTIC)]


def test_later_direct_trials_count_past_six(ctx, store):
    """Six earlier lever-only nodes no longer fill the window ahead of the direct tests; and every
    eligible linked trial folds (up to hypotheses.max_trials), not the first six."""
    lab = Lab(ctx, store, noise=False)
    h = _keyed_meta(ctx, "m", lab.tick(), {"effective_lr": -1}, "lower kde_loss_weight helps")
    store.put_hypothesis(h)
    for i in range(6):
        lab.node(f"lev{i}", levers={"effective_lr": -1})
        lab.measure(f"lev{i}", {a: -0.02 for a in BASE}, (11, 12))
    for i in range(3):
        lab.node(f"k{i}")
        store.set_state(f"config_diff:k{i}", {"dino.kde_loss_weight": 0.05})
        lab.measure(f"k{i}", {a: 0.02 for a in BASE}, (11, 12))
    ev = ComputedWorldModel(ctx).fit(store).hypotheses["m"]
    assert ctx.cfg.hypotheses.max_trials >= 12
    assert set(ev.trial_ids) <= {"k0", "k1", "k2"} and ev.status is HypothesisStatus.SUPPORTED
    assert [r[6] for r in store.get_state(HYP_KEY)["m"]] == [evalues.MECHANISTIC] * 3
    # a fold that was flat for six trials still decides on the 7th and 8th
    pairs = [(1.0, 1.0)] * 6 + [(CAP_META, 1 / CAP_META)] * 2
    assert fold(pairs, 0.1, 12)[2:] == (8, HypothesisStatus.SUPPORTED)
    assert fold(pairs, 0.1, 6)[3] is HypothesisStatus.INCONCLUSIVE          # the old window


def _family(mus, rng, *, family, d1=0.003, sd=0.002, alpha=0.1, n=12):
    """Statuses of len(mus) META hypotheses, n capped trials each, decided at family/alpha."""
    return [fold([trial_lr(rng.gauss(mu, sd), sd * sd, d1, CAP_META) for _ in range(n)],
                 alpha, n, family)[3] for mu in mus]


def test_multiplicity_null_family_and_planted_meta():
    """50 metas: at the per-meta 1/alpha bar some null is falsely decided most of the time; at the
    family bar K/alpha the family-wise error is <= alpha for both null directions, and a planted true
    meta among 49 nulls is still supported."""
    rng, alpha, reps, k = random.Random(7), 0.1, 400, 50
    naive = sum(any(s is HypothesisStatus.SUPPORTED for s in _family([0.0] * k, rng, family=1))
                for _ in range(reps // 4)) / (reps // 4)
    assert naive > 0.3
    fwer_for = sum(any(s is HypothesisStatus.SUPPORTED for s in _family([0.0] * k, rng, family=k))
                   for _ in range(reps)) / reps
    fwer_against = sum(any(s is HypothesisStatus.REFUTED for s in _family([0.003] * k, rng, family=k))
                       for _ in range(reps)) / reps
    assert fwer_for <= alpha and fwer_against <= alpha
    hits = [_family([0.008] + [0.0] * (k - 1), rng, family=k) for _ in range(reps // 4)]
    assert sum(s[0] is HypothesisStatus.SUPPORTED for s in hits) / len(hits) > 0.8
    assert sum(any(x is HypothesisStatus.SUPPORTED for x in s[1:]) for s in hits) / len(hits) <= alpha


def test_rows_booked_on_lever_links_fold_out(ctx, store):
    """rv-36-m3 'gradient balance' held E_for 33.9 on 22 LEVER rows from unrelated nodes: on deploy such rows
    (append-only, already booked) stop counting and the meta loses that support."""
    lab = Lab(ctx, store, noise=False)
    h = hypothesis(ctx, "m", lab.tick(), {"effective_lr": -1})
    store.put_hypothesis(h)
    lab.node("w", levers={"effective_lr": -1})
    lab.measure("w", {a: 0.02 for a in BASE}, (11, 12))
    store.set_state(HYP_KEY, {"m": [["x", 4.0, 0.25, 0.01, 1e-5, 0.003, evalues.LEVER]] * 3})
    ev = ComputedWorldModel(ctx).fit(store).hypotheses["m"]
    assert ev.status is HypothesisStatus.UNTESTED and ev.e_value == 1.0 and not ev.trial_ids


def test_family_size_raises_the_meta_bar(ctx, store):
    """The same four winning trials support a lone meta but not one of 50 (K/alpha = 500 > 4^4)."""
    lab = Lab(ctx, store, noise=False)
    h = hypothesis(ctx, "m", lab.tick(), {"effective_lr": -1})
    store.put_hypothesis(h)
    for i in range(4):
        lab.node(f"w{i}", levers={"effective_lr": -1})
        lab.measure(f"w{i}", {a: 0.02 for a in BASE}, (11, 12))
        store.link("m", f"w{i}", sign=1, auto=False)
    wm = ComputedWorldModel(ctx)
    assert wm.fit(store).hypotheses["m"].status is HypothesisStatus.SUPPORTED
    for i in range(49):
        store.put_hypothesis(hypothesis(ctx, f"other{i}", lab.tick(), {"capacity": 1}))
    ev = wm.fit(store).hypotheses["m"]
    assert ev.status is HypothesisStatus.TESTING and ev.e_value < 50 / ctx.cfg.hypotheses.alpha


# ---------------------------------------------------------------------------
# veto on the panel average; confirm-and-condense bank (operator 2026-09-30)
# ---------------------------------------------------------------------------

def test_positive_panel_is_vetoed_only_for_a_catastrophic_axis_drop(ctx, store):
    lab = Lab(ctx, store, noise=False)
    mild = {"alpha": 0.01, "beta": 0.01, "gamma": -0.02}           # gamma -2.9 sigma, panel +0.004
    bad = {"alpha": -0.01, "beta": 0.0, "gamma": -0.02}            # same gamma drop, panel negative
    assert panel_of(ctx, mild) > 0 > panel_of(ctx, bad)
    for nid, eff in (("mild", mild), ("bad", bad)):
        lab.node(nid)
        lab.measure(nid, eff, (11, 12, 13, 14))
    snap = ComputedWorldModel(ctx).fit(store)
    assert not snap.nodes["mild"].vetoed                            # the old per-axis rule vetoed it
    assert snap.nodes["bad"].vetoed                                 # panel negative: the per-axis rule stands
    strict = ComputedWorldModel(with_cfg(ctx, veto={"catastrophic_sigmas": 1.0})).fit(store)
    assert strict.nodes["mild"].veto_axes == ("gamma",)             # the knob: 1 sigma = the old rule


def test_bank_admits_fully_seeded_positive_non_confirmed_nodes_into_winner_pairs(ctx, store):
    from loop.select import winner_pairs
    lab = Lab(ctx, store, noise=False)
    good = {a: 0.006 for a in BASE}
    specs = {"win": (NodeStatus.CONFIRMED, good, 5), "nc": (NodeStatus.NOT_CONFIRMED, good, 5),
             "short": (NodeStatus.NOT_CONFIRMED, good, 4), "neg": (NodeStatus.NOT_CONFIRMED, {a: -0.004 for a in BASE}, 5)}
    for i, (nid, (status, eff, n)) in enumerate(specs.items()):
        lab.node(nid, status=status, tag=f"t{i}", family=("optimization", "regularization")[i % 2])
        store.set_state(f"config_diff:{nid}", {f"k.{nid}": 1})
        seeds = [(Stage.SCREEN, 11), (Stage.SCREEN, 12), (Stage.PROMOTE, 13), (Stage.CONFIRM, 101), (Stage.CONFIRM, 102)]
        for st, s in seeds[:n]:
            lab.measure(nid, eff, (s,), stage=st)
    lab.node("win2", status=NodeStatus.CONFIRMED, tag="t9", family="augmentation")
    store.set_state("config_diff:win2", {"k.win2": 1})
    for st, s in [(Stage.SCREEN, 11), (Stage.SCREEN, 12), (Stage.PROMOTE, 13), (Stage.CONFIRM, 103)]:
        lab.measure("win2", {a: 0.003 for a in BASE}, (s,), stage=st)
    snap = ComputedWorldModel(ctx).fit(store)
    assert snap.banked == ("nc",) and snap.incumbent_id == "win"
    pairs = [set(p) for p in winner_pairs(store, snap)]
    assert pairs == [{"win", "win2"}, {"nc", "win"}]                # banked: crossed with the stack head only, last
    off = ComputedWorldModel(with_cfg(ctx, promotion={"bank_min_seeds": 0})).fit(store)
    assert off.banked == () and [set(p) for p in winner_pairs(store, off)] == [{"win", "win2"}]
    lab.node("lucky", status=NodeStatus.NOT_CONFIRMED, tag="t7", family="training")     # n0423: selection-seed gain only
    lab.measure("lucky", {a: 0.01 for a in BASE}, (11, 12))
    lab.measure("lucky", {a: 0.01 for a in BASE}, (13,), stage=Stage.PROMOTE)
    lab.measure("lucky", {a: -0.003 for a in BASE}, (101, 102), stage=Stage.CONFIRM)
    snap = ComputedWorldModel(ctx).fit(store)
    assert snap.nodes["lucky"].p_gain > 0.8 and "lucky" not in snap.banked     # judged on its confirm seeds alone


def test_p_beats_incumbent_has_no_inherited_carrier_bias(ctx, store):
    """A child of the incumbent is judged on its OWN paired increment, not on its full-seed U (which
    carries the parent's lucky selection seeds) against the incumbent's confirm-only estimate."""
    lab = Lab(ctx, store, noise=False)
    lab.node("inc", status=NodeStatus.CONFIRMED)
    lab.measure("inc", {a: 0.012 for a in BASE}, (11, 12))                     # lucky selection seeds
    lab.measure("inc", {a: 0.012 for a in BASE}, (13,), stage=Stage.PROMOTE)
    lab.measure("inc", {a: 0.004 for a in BASE}, (101, 102), stage=Stage.CONFIRM)
    lab.measure("inc", {a: 0.012 for a in BASE}, (11, 12), stage=Stage.PARENT)
    lab.node("null", parent="inc", status=NodeStatus.SCREENING, tag="null", family="regularization")
    lab.measure("null", {a: 0.012 for a in BASE}, (11, 12))
    snap = ComputedWorldModel(ctx).fit(store)
    assert snap.incumbent_id == "inc"
    null = snap.nodes["null"]
    assert null.p_positive > 0.9                                   # the carrier's U alone looks great
    assert abs(null.p_gain - 0.5) < 0.05 and abs(null.p_beats_incumbent - null.p_gain) < 1e-9
    assert null.p_gate < ctx.cfg.promotion.p_promote               # the screen gate sees its own gain
    lab.node("real", parent="inc", status=NodeStatus.SCREENING, tag="real", family="regularization")
    lab.measure("real", {a: 0.030 for a in BASE}, (13, 14))
    lab.measure("inc", {a: 0.012 for a in BASE}, (13, 14), stage=Stage.PARENT)
    real = ComputedWorldModel(ctx).fit(store).nodes["real"]
    assert real.p_gain > 0.9 and real.p_beats_incumbent == real.p_gain and real.p_gate > 0.9
    # a grandchild: parent's confirm-only (U when unconfirmed) + own increment - incumbent's confirm-only
    lab.measure("real", {a: 0.030 for a in BASE}, (13, 14), stage=Stage.PARENT)
    lab.node("gc", parent="real", status=NodeStatus.SCREENING, tag="gc", family="augmentation")
    lab.measure("gc", {a: 0.030 for a in BASE}, (13, 14))
    snap = ComputedWorldModel(ctx).fit(store)
    gc, par = snap.nodes["gc"], snap.nodes["real"]
    inc = snap.nodes["inc"].confirm_estimate
    m = par.utility.mean + gc.increment.mean - inc.mean
    sd = (par.utility.var + gc.increment.var + inc.var) ** 0.5
    assert gc.p_beats_incumbent == pytest.approx(phi(m / sd), abs=1e-9)


def test_a_node_containing_an_ancestor_is_credited_only_with_its_own_increment(ctx, store):
    """n0363: n0255's whole stack + a predictor tweak re-rooted on root. Its paired deltas (vs root) carry the
    carrier's gain; measured against the contained ancestor its own increment is the tweak's. What it SCORES
    (U, P(U>U_inc)) is unchanged; a dead measure parent falls back to the declared one."""
    from loop.world.noise import MEASURE_KEY
    lab = Lab(ctx, store, seed=3, noise=False)
    gain = {"alpha": 0.02, "beta": 0.02, "gamma": 0.02}
    lab.node("a", status=NodeStatus.CONFIRMED)                                   # established: its gain is credited
    lab.measure("a", gain, (11, 12, 13))
    lab.node("b", tag="b_tweak")
    lab.measure("b", {k: v + 0.0005 for k, v in gain.items()}, (14, 101, 102))
    w = ComputedWorldModel(ctx)
    raw = w.fit(store)
    store.set_state(MEASURE_KEY, {"b": ["a"], "a": []})
    snap = w.fit(store)
    b0, b1 = raw.nodes["b"], snap.nodes["b"]
    assert b0.increment.mean > 0.01 and b0.p_gain > 0.95                        # credited with a's gain
    assert b1.increment.mean < 0.25 * b0.increment.mean and b1.p_gain < b0.p_gain   # its own tweak (+ the
    #                                                   carrier's shrinkage: U_a is a shrunk estimate of a's gain)
    assert snap.measured["b"][0] == "a" and snap.measured["b"][1] > 0.01
    assert b1.utility.mean == pytest.approx(b0.utility.mean, abs=1e-12)          # what it scores: unchanged
    assert b1.p_beats_incumbent == pytest.approx(b0.p_beats_incumbent, abs=1e-12)
    assert snap.nodes["a"].increment.mean == pytest.approx(raw.nodes["a"].increment.mean, abs=1e-12)
    store.update_node("a", status=NodeStatus.NOT_CONFIRMED)                     # an unconfirmed leg: a composition
    assert "b" not in w.fit(store).measured                                      # keeps its declared parent
    store.set_state("driver.audit", {"passed": ["a"]})                           # ... unless it passed the audit
    assert w.fit(store).measured["b"][0] == "a"


def test_a_resumed_run_carries_seed_level_noise_and_stays_out_of_the_noise_fit(ctx, store):
    """slurm.resume=on (operator 2026-10-01): a run resumed from latest.pt is not its seed's exact replicate; its
    paired delta's variance gains resume_var (seed-level), it never feeds sigma, and confirm e-values weigh it so."""
    from loop.world import noise
    ctx = with_cfg(ctx, prior={"noise_min_df": 1})
    lab = Lab(ctx, store, noise=False)
    lab.node("a")
    lab.measure("a", {k: 0.004 for k in BASE}, (11, 12))
    store.update_run("a:screen:12", resumed=1)
    led = noise.read_ledger(ctx, store)
    assert [o.resumed for o in led.data["a"].obs] == [False, True]
    assert all(len(xs) == 1 for xs in led.data["a"].all_seeds.values())          # only the exact one
    sigma = {t: 0.003 for t in noise.targets(ctx)}
    (_, v0), (_, v1) = noise.correct(ctx, led, sigma).data["a"].selection[PANEL]
    rv = noise.resume_var(ctx, sigma, PANEL)
    assert rv > 0 and v1 == pytest.approx(v0 + rv)
    assert evalues.lr_evalue([0.004, 0.004], 0.004, 0.003, [0.0, rv]) < evalues.lr_evalue([0.004, 0.004], 0.004, 0.003)


def test_audit_weighs_a_resumed_seed_with_seed_level_noise(ctx):
    from loop import seeds as S
    cfg = ctx.cfg
    seeds = list(cfg.seeds.audit)
    mk = lambda nid, s, x, r=0: Run(id=Run.make_id(nid, Stage.AUDIT, s), task_id="t", node_id=nid, seed=s,  # noqa: E731
                                    stage=Stage.AUDIT, status=RunStatus.DONE, scores={a: x for a in cfg.axes}, resumed=r)
    node = [mk("n1", s, 0.51) for s in seeds]
    root = [mk(ROOT_NODE_ID, s, 0.50) for s in seeds]
    kw = dict(cfg=cfg, delta=0.005, sigma=0.003, infra_retry=2)
    _, e0, _ = S.audit_step("n1", node, root, **kw, resume_v=1e-5)
    node[0] = mk("n1", seeds[0], 0.51, r=2)
    _, e1, _ = S.audit_step("n1", node, root, **kw, resume_v=1e-5)
    assert e1 < e0


def test_refreeze_confirmed_and_twin_challengers_never_take_the_incumbency_unaudited(ctx, store):
    """n0264 / n0267: confirmed by the one-time refreeze (rule changed after the data) -> no incumbent until the
    held-out audit passes; n0267 is built from n0255 -> a swap needs its audited estimate to beat by 2x the sd."""
    from loop.world import K_AUDIT, K_INC, K_REFROZEN
    lab = Lab(ctx, store, noise=False)
    lab.node("inc", status=NodeStatus.CONFIRMED)
    lab.measure("inc", {a: 0.004 for a in BASE}, (11, 12))
    lab.measure("inc", {a: 0.004 for a in BASE}, (101, 102), stage=Stage.CONFIRM)
    assert ComputedWorldModel(ctx).fit(store).incumbent_id == "inc" and store.get_state(K_INC) == "inc"
    lab.node("kid", status=NodeStatus.CONFIRMED, tag="kid", family="regularization")    # unrelated, higher
    lab.measure("kid", {a: 0.006 for a in BASE}, (11, 12))
    lab.measure("kid", {a: 0.006 for a in BASE}, (103, 104), stage=Stage.CONFIRM)
    store.set_state("config_diff:inc", {"k.inc": 1})
    store.set_state("config_diff:kid", {"k.kid": 1})
    store.set_state(K_REFROZEN, ["kid"])
    assert ComputedWorldModel(ctx).fit(store).incumbent_id == "inc"                   # held back: unaudited
    store.set_state(K_AUDIT, {"passed": ["kid"]})
    assert ComputedWorldModel(ctx).fit(store).incumbent_id == "kid"                   # audited, not a twin
    store.set_state(K_INC, "inc")
    store.set_state("config_diff:kid", {"k.inc": 1, "k.kid": 1})                    # the incumbent's recipe + 1
    assert ComputedWorldModel(ctx).fit(store).incumbent_id == "inc"                   # +0.002 < 2x combined sd
    store.set_state(K_AUDIT, {"passed": ["kid"], "gate": {"kid": {"verdict": "passed", "mean": 0.02, "se": 0.001}}})
    assert ComputedWorldModel(ctx).fit(store).incumbent_id == "kid"                   # the audit beats it clearly


def test_an_audited_incumbent_moves_only_on_the_paired_audit_seeds_whatever_the_relatedness(ctx, store):
    """n0264 replaced n0255 on U +0.0077 vs +0.0076 (unrelated: one 2.0x gap) while losing on every shared audit
    seed: two audited nodes are compared on their paired audit seeds (2-sd rule), an unaudited one never displaces
    an audited incumbent, and a wrong swap already stored is undone by the next fit."""
    from loop.world import K_AUDIT, K_INC, audit_paired
    lab = Lab(ctx, store, noise=False)
    for nid, x, fam in (("inc", 0.004, "optimization"), ("kid", 0.006, "regularization")):
        lab.node(nid, status=NodeStatus.CONFIRMED, tag=nid, family=fam)
        lab.measure(nid, {a: x for a in BASE}, (11, 12))
        lab.measure(nid, {a: x for a in BASE}, (101 if nid == "inc" else 103, 102 if nid == "inc" else 104),
                    stage=Stage.CONFIRM)
        store.set_state(f"config_diff:{nid}", {f"k.{nid}": 1})
    gate = lambda xs, off=0.0: {"verdict": "passed", "mean": sum(xs) / 4 - off, "se": 0.001,   # noqa: E731
                                "deltas": {str(s): x - off for s, x in zip((201, 202, 203, 204), xs)},
                                "offsets": {str(s): off for s in (201, 202, 203, 204)}}
    store.set_state(K_AUDIT, {"passed": ["inc"], "gate": {"inc": gate([0.010, 0.011, 0.015, 0.008])}})
    assert ComputedWorldModel(ctx).fit(store).incumbent_id == "inc"
    assert ComputedWorldModel(ctx).fit(store).incumbent_id == "inc"        # kid: higher confirm U, unaudited
    store.set_state(K_AUDIT, {"passed": ["inc", "kid"], "gate": {"inc": gate([0.010, 0.011, 0.015, 0.008]),
                                                                  "kid": gate([0.007, 0.009, 0.010, 0.007], 0.003)}})
    m, v = audit_paired(store, "kid", "inc")
    assert m == pytest.approx(-0.00275) and v > 0                           # offsets added back: root cancels
    assert ComputedWorldModel(ctx).fit(store).incumbent_id == "inc"
    store.set_state(K_INC, "kid")                                            # the wrong swap, already stored
    assert ComputedWorldModel(ctx).fit(store).incumbent_id == "inc"
    assert store.events("incumbent")[-1]["previous"] == "kid"
    store.set_state(K_AUDIT, {"passed": ["inc", "kid"], "gate": {"inc": gate([0.010, 0.011, 0.015, 0.008]),
                                                                  "kid": gate([0.016, 0.018, 0.020, 0.015])}})
    assert ComputedWorldModel(ctx).fit(store).incumbent_id == "kid"         # +0.0063 paired, sd ~0.001


def test_a_childs_control_reuses_the_parents_own_done_run_at_that_seed():
    """No fresh parent run where the parent already ran the seed (screen / promote / confirm): the same commit and
    seed. Its parent-stage run wins when both exist."""
    from loop.contracts import Node, NodeStatus, Run, RunStatus, Stage, Strategy
    from loop.world import noise
    kid = Node(id="n2", task_id="t", commit="c", parent_id="n1", strategy=Strategy.EXPLORE, operator=None,
               family="f", mechanism_tag="m", diff_hash="h", status=NodeStatus.CONFIRMING)
    run = lambda st, s: Run(id=f"n1:{st.value}:{s}", task_id="t", node_id="n1", seed=s, stage=st,   # noqa: E731
                            status=RunStatus.DONE)
    got = noise.controls(kid, {}, [run(Stage.CONFIRM, 14), run(Stage.PARENT, 14), run(Stage.SCREEN, 11),
                                   run(Stage.AUDIT, 201)])
    assert {s: r.stage for s, r in got.items()} == {14: Stage.PARENT, 11: Stage.SCREEN}


def test_audited_nodes_are_compared_over_every_fresh_seed_both_ran_never_a_selection_seed(ctx, store):
    """n0655 vs n0255: 4 audit seeds alone could not resolve +0.002; every fresh seed both ran (audit + confirm +
    parent) is paired, and a seed either node SELECTED on (screen / promote) never counts."""
    from loop.world import K_AUDIT, fresh_paired
    lab = Lab(ctx, store, noise=False)
    for nid, fam in (("inc", "optimization"), ("kid", "regularization")):
        lab.node(nid, status=NodeStatus.CONFIRMED, tag=nid, family=fam)
        store.set_state(f"config_diff:{nid}", {f"k.{nid}": 1})
    wig = (0.001, -0.002, 0.0015, -0.0005, 0.002, -0.001, 0.0)
    for s, w in zip((101, 102, 103, 104, 201, 202, 203), wig):         # kid = inc + 0.003 (+-0.0002) on 7 seeds
        st = Stage.AUDIT if s > 200 else Stage.CONFIRM
        lab.measure("inc", {a: 0.004 + w for a in BASE}, (s,), stage=st)
        lab.measure("kid", {a: 0.007 + w + (0.0002 if s % 2 else -0.0002) for a in BASE}, (s,), stage=st)
    lab.measure("inc", {a: 0.004 for a in BASE}, (11, 12))
    lab.measure("kid", {a: 0.050 for a in BASE}, (11, 12), stage=Stage.SCREEN)   # selection: never paired
    lab.measure("inc", {a: 0.004 for a in BASE}, (11, 12), stage=Stage.PARENT)
    m, v = fresh_paired(ctx, store, "kid", "inc")
    assert m == pytest.approx(0.003 + 0.0002 / 7, abs=1e-6) and v ** 0.5 < 0.0002
    gate = {"verdict": "passed", "mean": 0.006, "se": 0.003, "deltas": {"201": 0.0, "202": 0.01}, "offsets": {}}
    store.set_state(K_AUDIT, {"passed": ["inc", "kid"], "gate": {"inc": {**gate, "mean": 0.005}, "kid": gate}})
    store.set_state("world.incumbent", "inc")
    assert ComputedWorldModel(ctx).fit(store).incumbent_id == "kid"           # 7 fresh seeds: +0.003 >> 2 sd
    assert fresh_paired(ctx, store, "kid", ROOT_NODE_ID) is None              # < 2 shared fresh seeds


def test_every_audited_challenger_is_tested_not_only_the_best_audited_estimate(ctx, store):
    """n0655 beat n0255 by +0.0021 over 10 fresh seeds (2.1 sd) but was never tested: n0255's own lucky audit seed
    made IT the best audited estimate and n0764 led on confirm. Every audited node is a challenger."""
    from loop.world import K_AUDIT, fresh_paired
    lab = Lab(ctx, store, noise=False)
    for nid, fam in (("inc", "optimization"), ("kid", "regularization"), ("lead", "augmentation")):
        lab.node(nid, status=NodeStatus.CONFIRMED, tag=nid, family=fam)
        store.set_state(f"config_diff:{nid}", {f"k.{nid}": 1})
    wig = (0.001, -0.002, 0.0015, -0.0005, 0.002, -0.001, 0.0)
    for s, w in zip((101, 102, 103, 104, 201, 202, 203), wig):
        st = Stage.AUDIT if s > 200 else Stage.CONFIRM
        lab.measure("inc", {a: 0.004 + w for a in BASE}, (s,), stage=st)
        lab.measure("kid", {a: 0.007 + w + (0.0002 if s % 2 else -0.0002) for a in BASE}, (s,), stage=st)
    for nid, x in (("inc", 0.004), ("kid", 0.007), ("lead", 0.009)):
        lab.measure(nid, {a: x for a in BASE}, (11, 12))
    lab.measure("lead", {a: 0.009 for a in BASE}, (14,), stage=Stage.CONFIRM)     # confirm leader, no shared seed
    gate = lambda m: {"verdict": "passed", "mean": m, "se": 0.003, "deltas": {}, "offsets": {}}   # noqa: E731
    store.set_state(K_AUDIT, {"passed": ["inc", "kid", "lead"],
                              "gate": {"inc": gate(0.010), "kid": gate(0.006), "lead": gate(0.005)}})
    store.set_state("world.incumbent", "inc")
    assert fresh_paired(ctx, store, "kid", "inc")[0] > 0.0029
    assert ComputedWorldModel(ctx).fit(store).incumbent_id == "kid"     # the lowest audited estimate, tested anyway


def test_a_confirmed_carriers_fresh_estimate_counts_every_fresh_seed_paired_with_root(ctx, store):
    """2026-10-06: n0255's confirm-only U read +0.0049 on its ONE confirm seed (tree prior tau_node 0.0019) beside 15
    parent / audit seeds paired with root at +0.0090. Every fresh seed root also ran counts, so the prior's weight
    shrinks with all of them; a selection seed never does."""
    from loop.world import fresh_vs_root, noise
    lab = Lab(ctx, store, noise=False)
    lab.node("car", status=NodeStatus.CONFIRMED)
    lab.measure("car", {a: 0.03 for a in BASE}, (11, 12))                          # selection: never fresh
    lab.measure("car", {a: 0.01 for a in BASE}, (101,), stage=Stage.CONFIRM)
    lab.measure("car", {a: 0.01 for a in BASE}, range(301, 313), stage=Stage.PARENT)
    lab.measure("car", {a: 0.01 for a in BASE}, (201, 202), stage=Stage.AUDIT)
    lab.measure("car", {a: 0.01 for a in BASE}, (12,), stage=Stage.PARENT)        # a selection seed: not fresh
    snap = ComputedWorldModel(ctx).fit(store)
    ledger = noise.correct(ctx, noise.read_ledger(ctx, store), snap.sigma)
    fr = fresh_vs_root(ctx, store, ledger, snap.sigma)["car"]
    assert len(fr) == 14 and all(x == pytest.approx(panel_of(ctx, {a: 0.01 for a in BASE})) for x, _ in fr)
    ce = snap.nodes["car"].confirm_estimate
    assert ce.n == 15 and ce.mean == pytest.approx(panel_of(ctx, {a: 0.01 for a in BASE}), abs=0.1 * ce.mean)
