"""loop.core.model on synthetic data with known mu, seed effects and noise."""

from __future__ import annotations

import random
import statistics as st

import pytest

from loop.core import model as M
from loop.core.data import AXES, PANEL, Obs

from .conftest import synth

SIGMA, RHO = 0.003, 0.4


def _world(rng: random.Random, n_children: int = 150, parent_seeds: int = 40, child_seeds: int = 3):
    """A parent on ``parent_seeds`` seeds; children each on ``child_seeds`` of them (CRN), true mu ~ N(0, .002)."""
    mu = {"p": 0.0} | {f"c{i}": rng.gauss(0, 0.002) for i in range(n_children)}
    seeds = {"p": list(range(parent_seeds))} | {f"c{i}": rng.sample(range(parent_seeds), child_seeds)
                                                 for i in range(n_children)}
    return mu, synth(rng, mu, seeds, sigma=SIGMA, rho=RHO), {c: "p" for c in mu if c != "p"}


def test_fit_noise_recovers_sigma_and_rho():
    rng = random.Random(1)
    _, obs, parents = _world(rng, n_children=300)
    nz = M.fit_noise(obs, [PANEL], parents=parents)
    assert nz.sd(PANEL) == pytest.approx(SIGMA, rel=0.08)
    assert nz.rho[PANEL] == pytest.approx(RHO, abs=0.08)


def test_seed_means_average_duplicates_and_skip_resumed():
    a = Obs("a1", "n", 1, "screen", "select", {x: 0.5 for x in AXES}, 0.5, 1, 0, 0)
    b = Obs("a2", "n", 1, "confirm", "select", {x: 0.7 for x in AXES}, 0.7, 1, 0, 1)
    assert M.seed_means([a, b], PANEL) == {1: pytest.approx(0.6)}
    assert M.seed_means([a, b], PANEL, paired_only=True) == {1: 0.5}


def test_cuped_is_unbiased_variance_reducing_and_its_V_is_right():
    rng = random.Random(2)
    cuped, naive, vs = [], [], []
    for _ in range(1500):
        mu = {"p": 0.0, "c": 0.002}
        ps = list(range(20))
        obs = synth(rng, mu, {"p": ps, "c": rng.sample(ps, 3)}, sigma=SIGMA, rho=RHO)
        child = [o for o in obs if o.node == "c"]
        parent = [o for o in obs if o.node == "p"]
        nz = M.Noise({PANEL: SIGMA ** 2}, {PANEL: RHO})
        est = M.cuped(child, parent, PANEL, nz)
        cuped.append(est.delta)
        vs.append(est.var)
        naive.append(st.mean(o.panel for o in child) - st.mean(o.panel for o in parent))
        assert est.n == 3 and est.paired == 3
    se = st.stdev(cuped) / len(cuped) ** 0.5
    assert abs(st.mean(cuped) - 0.002) < 3 * se                     # unbiased
    assert st.variance(cuped) < 0.92 * st.variance(naive)            # seed luck regressed out (theory .78-.88)
    assert st.variance(cuped) == pytest.approx(st.mean(vs), rel=0.15)  # V is the sampling variance


def test_cuped_unpaired_seed_is_not_adjusted():
    nz = M.Noise({PANEL: SIGMA ** 2}, {PANEL: RHO})
    mk = lambda node, seed, y, resumed=0: Obs(f"{node}:{seed}", node, seed, "screen", "select",
                                              {a: y for a in AXES}, y, 1, 0, resumed)
    parent = [mk("p", 1, 0.60), mk("p", 2, 0.62)]
    est = M.cuped([mk("c", 9, 0.65)], parent, PANEL, nz)
    assert est.paired == 0 and est.delta == pytest.approx(0.65 - 0.61)
    assert M.cuped([mk("c", 1, 0.65, resumed=1)], parent, PANEL, nz).paired == 0
    assert M.cuped([], parent, PANEL, nz) is None


def test_parent_mean_drops_a_selected_parents_own_screens():
    mk = lambda seed, stage, y: Obs(f"p:{stage}:{seed}", "p", seed, stage, "select", {a: y for a in AXES}, y, 1, 0, 0)
    obs = [mk(1, "screen", 0.70), mk(2, "screen", 0.70), mk(3, "parent", 0.60), mk(4, "parent", 0.62)]
    assert M.parent_mean(obs, PANEL) == (pytest.approx(0.61), 2)
    assert M.parent_mean(obs[:3], PANEL)[1] == 3                     # < 2 unbiased seeds: all of them


def test_composite_weights_drop_a_pure_noise_axis():
    rng = random.Random(3)
    obs = []
    for i in range(60):
        mu = rng.gauss(0, 0.004)
        for s in range(4):
            axes = {a: mu + rng.gauss(0, 0.002) for a in AXES} | {"surv": 0.6 + rng.gauss(0, 0.01)}
            obs.append(Obs(f"n{i}:{s}", f"n{i}", s, "screen", "select", axes, 0.0, 1, 0, 0))
    nz = M.Noise({a: 0.002 ** 2 for a in AXES} | {"surv": 0.01 ** 2}, {a: 0.0 for a in AXES})
    w = M.composite_weights(obs, nz)
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["surv"] == 0.0 and w["cls"] > w["seg"] > 0


def _pts(rng, cells: dict[tuple, tuple[float, int]], tau=0.001, v=0.002 ** 2):
    return [M.PriorPoint(c, rng.gauss(m, (tau ** 2 + v) ** 0.5), v) for c, (m, n) in cells.items() for _ in range(n)]


def test_eb_prior_shrinkage_is_sane():
    rng = random.Random(4)
    cells = {("root", "objective", "explore"): (0.004, 200), ("n1", "objective", "explore"): (-0.002, 200),
             ("n1", "training", "exploit"): (-0.002, 3)}
    pts = _pts(rng, cells)
    prior = M.fit_prior(pts)
    big, _, depth = prior.cell(("root", "objective", "explore"))
    assert depth == 3 and big == pytest.approx(0.004, abs=0.0006)    # a big cell keeps its own mean
    small, _, _ = prior.cell(("n1", "training", "exploit"))
    own = M._median([p.delta for p in pts if p.cell[1] == "training"])     # v3 review K7: robust (median) location
    assert prior.means[("n1",)] < small < own                          # a tiny cell leans on its carrier
    unseen, tau2, depth = prior.cell(("n1", "augmentation", "explore"))
    assert depth == 1 and unseen == pytest.approx(prior.means[("n1",)])
    assert tau2 >= M.TAU2_MIN
    assert M.fit_prior([]).cell(("a", "b", "c"))[2] == 0


def test_eb_prior_pools_when_cells_do_not_differ():
    rng = random.Random(5)
    cells = {(f"c{i}", "f", "explore"): (0.001, 30) for i in range(8)}
    prior = M.fit_prior(_pts(rng, cells))
    spread = [prior.means[(f"c{i}",)] for i in range(8)]
    assert max(spread) - min(spread) < 0.0006                         # shrunk together (true spread 0)


def test_val_loss_covariate_slope_is_learned():
    rng = random.Random(6)
    pts = []
    for _ in range(400):
        z = rng.gauss(0, 1)
        pts.append(M.PriorPoint(("root", "f", "explore"), -0.002 * z + rng.gauss(0, 0.002), 0.002 ** 2, z))
    prior = M.fit_prior(pts)
    assert prior.gamma == pytest.approx(-0.002, abs=0.0004)
    assert prior.cell(("root", "f", "explore"), 1.0)[0] < prior.cell(("root", "f", "explore"), -1.0)[0]


def test_posterior_gaussian_and_t():
    est = M.Estimate(0.004, 0.002 ** 2, 3, 3)
    m, v = M.posterior(0.0, 0.002 ** 2, est, nu=None)
    assert m == pytest.approx(0.002) and v == pytest.approx(0.5 * 0.002 ** 2)
    mt, vt = M.posterior(0.0, 0.002 ** 2, est, nu=3.0)
    assert 0.0 < mt < 0.004 and vt > 0
    far = M.Estimate(0.02, 0.002 ** 2, 3, 3)                           # 10 prior sds out
    assert M.posterior(0.0, 0.002 ** 2, far, nu=3.0)[0] > 0.9 * 0.02 > M.posterior(0.0, 0.002 ** 2, far, nu=None)[0]


def test_expected_max_matches_simulation_and_known_values():
    assert M.expected_max(1) == 0.0
    assert M.expected_max(4) == pytest.approx(1.029, abs=0.002)
    assert M.expected_max(16) == pytest.approx(1.766, abs=0.002)
    rng = random.Random(7)
    for k in (2, 4, 16):
        sim = st.mean(max(rng.gauss(0, 1) for _ in range(k)) for _ in range(20000))
        assert M.expected_max(k) == pytest.approx(sim, abs=0.03)


def test_max_of_k_removes_the_winners_curse():
    """Raw estimates (flat prior: m = δ̂, v = V) of K equal candidates: the selected one's m_sel is unbiased."""
    rng = random.Random(8)
    k, sd, gap = 8, 0.002, []
    for _ in range(5000):
        best = max(rng.gauss(0.001, sd) for _ in range(k))
        gap.append(M.max_of_k(best, sd ** 2, k) - 0.001)
    assert abs(st.mean(gap)) < 3 * st.stdev(gap) / len(gap) ** 0.5
    assert M.max_of_k(0.003, sd ** 2, 1) == 0.003


def test_fit_noise_reports_dfs_and_calibrates_the_resumed_factor():
    """A resumed run keeps only part of its seed effect: z_r = sqrt(1-f) z + sqrt(f) z'. The fitted factor is
    Var d(resumed pair) / Var d(clean pair), recovered from resumed-vs-clean same-seed pairs (not 1/(1-rho))."""
    rng = random.Random(5)
    sigma, rho, f = 0.003, 0.9, 0.3
    obs, parents = [], {}
    for i in range(120):
        c = f"c{i}"
        parents[c] = "p"
        for s in range(8):
            b = rng.gauss(0, 1)
            for node in ("p", c) if i == 0 else (c,):
                z = b
                resumed = node == c and s >= 6                    # two resumed seeds per child
                if resumed:
                    z = (1 - f) ** 0.5 * b + f ** 0.5 * rng.gauss(0, 1)
                y = sigma * (rho ** 0.5 * z + (1 - rho) ** 0.5 * rng.gauss(0, 1))
                obs.append(Obs(f"{node}:{s}:{i}", node, s, "screen", "select", {a: y for a in AXES}, y, 1.0, 0.0,
                               int(resumed)))
            if i:
                pass
    # the parent ran every seed once (its seed effect b is shared by construction only for c0); rebuild CRN properly
    rng = random.Random(6)
    obs, bs = [], {s: rng.gauss(0, 1) for s in range(8)}
    for node in ["p", *parents]:
        for s in range(8):
            resumed = node != "p" and s >= 6
            z = (1 - f) ** 0.5 * bs[s] + f ** 0.5 * rng.gauss(0, 1) if resumed else bs[s]
            y = sigma * (rho ** 0.5 * z + (1 - rho) ** 0.5 * rng.gauss(0, 1))
            obs.append(Obs(f"{node}:{s}", node, s, "screen", "select", {a: y for a in AXES}, y, 1.0, 0.0, int(resumed)))
    nz = M.fit_noise(obs, [PANEL], parents=parents)
    assert nz.rho[PANEL] == pytest.approx(rho, abs=0.03)
    assert nz.df[PANEL] == 120 * 5                                      # 6 clean pairs per child -> 5 dfs
    expect = (1 - rho * (1 - f) ** 0.5) / (1 - rho)                  # Var d resumed / clean, ~ 1.55 / .1 -> ~1.5x
    assert nz.resumed_factor(PANEL) == pytest.approx(expect, rel=0.25)
    assert nz.resumed_factor(PANEL) < 1 / (1 - rho) / 2              # far below the uncalibrated 1/(1-rho) = 10
    bare = M.Noise({PANEL: sigma ** 2}, {PANEL: rho})
    assert bare.resumed_factor(PANEL) == pytest.approx(10.0)


def test_transport_resume_moves_the_rho_drop_not_the_ratio():
    pooled = M.Noise({PANEL: 1.0}, {PANEL: 0.4}, {PANEL: 100}, {PANEL: 1.5})      # rho_r = .4 - .6*.5 = .1
    near = M.transport_resume(M.Noise({PANEL: 1.0}, {PANEL: 0.9}, {PANEL: 30}), pooled)
    assert near.resumed_factor(PANEL) == pytest.approx((1 - 0.9 + 0.3) / 0.1)       # 4.0, not 1.5
    assert M.transport_resume(near, M.Noise({PANEL: 1.0}, {PANEL: 0.4}, {}, {PANEL: 9.0})).resume == near.resume


def test_pairs_by_change_near_only_for_one_key_config_children():
    from loop.core.data import ROOT, Node

    def node(nid, parent, port_of=""):
        return Node(nid, parent, None, "explore", "f", "explore", ROOT, 0.0, "", "", "", port_of)
    nodes = {ROOT: node(ROOT, None), "a": node("a", ROOT), "k": node("k", "a"), "c": node("c", "a"),
             "pt": node("pt", ROOT, "old")}
    near, far = M.pairs_by_change(nodes, {"k", "pt"})
    assert near == [("k", "a")]                                         # a port is never near
    assert set(far) == {("a", ROOT), ("c", "a"), ("pt", ROOT), ("k", ROOT), ("c", ROOT)}


def test_cuped_variance_with_a_near_rho_on_a_far_change_is_understated():
    """B3: theta = .94 on a change whose true rho is .4 claims var_s = .12 sigma2, actual 1.13 sigma2."""
    rng = random.Random(9)
    est, vs = [], []
    for _ in range(1200):
        mu = {"p": 0.0, "c": 0.0}
        ps = list(range(20))
        obs = synth(rng, mu, {"p": ps, "c": rng.sample(ps, 3)}, sigma=SIGMA, rho=0.4)
        e = M.cuped([o for o in obs if o.node == "c"], [o for o in obs if o.node == "p"], PANEL,
                    M.Noise({PANEL: SIGMA ** 2}, {PANEL: 0.94}))
        est.append(e.delta)
        vs.append(e.var)
    assert st.variance(est) > 4 * st.mean(vs)                          # the far class's noise is the fix
