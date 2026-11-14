"""loop.world.predictive (the predictive world model) and loop.world.mixture (the gates' nonparametric prior)."""

from __future__ import annotations

import math
import random
from types import SimpleNamespace

import pytest

from loop.contracts import ROOT_NODE_ID, LeverTags, Node, NodePosterior, NodeStatus, Operator, Posterior, Strategy
from loop.select import pooled_gain
from loop.world import ComputedWorldModel, mixture, predictive as P
from loop.world.evalues import phi

from tests.loop.test_world import Lab, with_cfg


# ------------------------------------------------------------------------------------------------ mixture
def _population(rng, n, pi0=0.8, mu=-0.004, tau=0.01, se=0.0023):
    eps = [0.0 if rng.random() < pi0 else rng.gauss(mu, tau) for _ in range(n)]
    return eps, [(x + rng.gauss(0.0, se), se * se) for x in eps]


def _ols(xs, ys):
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return b, mx - my


def test_the_grid_posterior_of_selected_screens_is_unbiased_for_fresh_seeds():
    """The winner's curse: the top 5% of 4000 screens (90% null) over-predict fresh seeds by ~1.7 se raw; the
    grid NPMLE posterior does not, and its E[max(gain, 0)] matches the truth."""
    rng = random.Random(12)
    eps, obs = _population(rng, 4000, pi0=0.9, mu=0.0, tau=0.004)
    m = mixture.fit_grid(obs)
    top = sorted(range(len(obs)), key=lambda i: -obs[i][0])[:200]
    fresh = [eps[i] + rng.gauss(0.0, 0.0023) for i in top]
    b_raw, bias_raw = _ols([obs[i][0] for i in top], fresh)
    b_eb, bias_eb = _ols([mixture.moments(mixture.posterior(m, obs[i]))[0] for i in top], fresh)
    assert bias_raw > 0.003
    assert abs(bias_eb) < 0.0005 and 0.8 < b_eb < 1.25
    gain = sum(mixture.expected_gain(mixture.posterior(m, obs[i])) for i in top) / len(top)
    assert abs(gain - sum(max(eps[i], 0.0) for i in top) / len(top)) < 0.0005


def test_prob_greater_and_moments():
    spike = ((1.0, 0.0, 0.0),)
    assert mixture.prob_greater(spike) == 0.0 and mixture.prob_greater(spike, (0.001, 0.0)) == 1.0     # ties -> 0
    assert mixture.prob_greater(spike, (0.001, 0.001 ** 2)) == pytest.approx(phi(1.0))
    cs = ((0.5, 0.0, 0.0), (0.5, 0.002, 0.001 ** 2))
    assert mixture.prob_greater(cs) == pytest.approx(0.5 * phi(2.0))
    m, v = mixture.moments(cs)
    assert m == pytest.approx(0.001) and v == pytest.approx(0.5 * (1e-6 + 4e-6) - 1e-6)
    assert mixture.expected_gain(((0.5, 0.003, 0.0), (0.5, -0.002, 0.0))) == pytest.approx(0.0015)
    assert mixture.expected_gain(((1.0, 0.0, 1e-6),)) == pytest.approx(1e-3 / (2.0 * math.pi) ** 0.5)


def test_the_grid_keeps_small_positives_under_a_harmful_majority():
    rng = random.Random(4)
    eps = [rng.choice((0.0,) * 7 + (-0.02, -0.01, 0.003)) for _ in range(2000)]
    obs = [(x + rng.gauss(0.0, 0.0023), 0.0023 ** 2) for x in eps]
    g = mixture.fit_grid(obs)
    near = [i for i, (e, _) in enumerate(obs) if abs(e - 0.004) < 0.001]
    truth = sum(max(eps[i], 0.0) for i in near) / len(near)
    gain = lambda m: sum(mixture.expected_gain(mixture.posterior(m, obs[i])) for i in near) / len(near)   # noqa: E731
    assert abs(gain(g) - truth) < 0.2 * truth
    warm = mixture.fit_grid(obs, start=g)
    assert [x for _, x, _ in warm.prior] == [x for _, x, _ in g.prior] and abs(gain(warm) - gain(g)) < 0.1 * truth
    assert mixture.fit_grid(obs[:1]) is None


def test_the_grid_stays_fine_around_zero_whatever_the_outliers():
    rng = random.Random(1)
    obs = [(rng.gauss(0.0, 0.0023), 0.0023 ** 2) for _ in range(500)] + [(-0.178, 0.0023 ** 2)]
    g = mixture.grid_for(obs)
    assert 0.0 in g and min(g) == pytest.approx(-0.178) and min(abs(x - 0.001) for x in g) <= 0.0023 / 8 + 1e-9


# ------------------------------------------------------------------------------------------------ the model
def _feat(gen, fam, mt="opt", lev=(0.0,)):
    return P.Features(gen, {"family": fam, "gen_family": f"{gen}/{fam}", "mtype": mt, "gen_mtype": f"{gen}/{mt}",
                            "parent": ROOT_NODE_ID if gen == "root" else "c1", "strategy": "explore/draft",
                            "brief": "none", "web": "na"},
                      {"structure": (0.0, 0.0, 1.0), "saturation": (0.0, 0.0), "levers": lev,
                       "diagnostics": (0.0,) * (len(P.DIAG_KEYS) + 1)})


def _synthetic(rng, n=600, se=0.0023):
    """Rows whose true own gain is a family effect + a lever effect + noise; families 'good' / 'bad' / 'flat'."""
    eff = {"good": 0.004, "bad": -0.004, "flat": 0.0}
    rows = {}
    for i in range(n):
        fam = rng.choice(list(eff))
        gen = "root" if i % 3 == 0 else "carrier"
        lev = (float(rng.choice((-1, 0, 1))),)
        theta = eff[fam] + 0.002 * lev[0] + rng.gauss(0.0, 0.001)
        screen = (theta + rng.gauss(0.0, se), se * se)
        fresh = (theta + rng.gauss(0.0, se / 2), se * se / 4) if rng.random() < 0.15 else None
        rows[f"n{i:04d}"] = P.NodeRow(f"n{i:04d}", float(i), _feat(gen, fam, lev=lev), screen, screen, screen, fresh,
                                      screen, screen)
    return rows, eff


def test_the_additive_model_recovers_group_effects():
    rows, eff = _synthetic(random.Random(2))
    m = P.Additive(("family", "levers")).fit(P.observations(rows.values(), 0.0))
    good, bad = m.predict(_feat("root", "good"))[0], m.predict(_feat("root", "bad"))[0]
    assert good - bad == pytest.approx(0.008, abs=0.0015)
    up = m.predict(_feat("root", "flat", lev=(1.0,)))[0] - m.predict(_feat("root", "flat", lev=(-1.0,)))[0]
    assert up == pytest.approx(0.004, abs=0.0015)
    _, v_seen = m.predict(_feat("root", "good"))
    _, v_new = m.predict(_feat("root", "never-seen"))
    assert v_new > v_seen                               # an unseen level carries its group's tau^2


def test_the_time_split_keeps_skillful_groups_and_drops_noise():
    rows, _ = _synthetic(random.Random(3))
    ev = P.evaluate(rows, ("family", "levers", "brief", "web"), 0.0, 0.1, 0.0005 ** 2, 15, folds=4, sweeps=15)
    assert "family" in ev["kept"] and "levers" in ev["kept"]
    assert "brief" not in ev["kept"] and "web" not in ev["kept"]          # one level each: no skill, dropped
    assert ev["screen:all"]["rho"] > 0.4 and ev["gate_bins"] in P.GATE_BINS
    assert ev["fresh"]["all"]["n"] > 0 and ev["fresh"]["all"]["rho_post"] > 0.3


def test_transfer_recovers_a_screen_to_fresh_slope():
    rng = random.Random(5)
    rows = []
    for _ in range(400):
        p = rng.gauss(0.0, 0.003)
        d = rng.gauss(0.0, 0.004)
        fresh = 0.0005 + 0.5 * p + 0.25 * d + rng.gauss(0.0, 0.0005)
        rows.append((fresh, 0.0005 ** 2, p, 1.0, d))
    t = P.fit_transfer(rows, 1e-8)
    assert t.c == pytest.approx(0.0005, abs=0.0002) and t.b == pytest.approx(0.5, abs=0.1)
    assert t.kappa == pytest.approx(0.25, abs=0.08) and t.n == 400
    assert P.fit_transfer(rows[:2], 1e-8) is None


def test_fit_outputs_allocation_and_a_plateau_forecast():
    rows, _ = _synthetic(random.Random(7), n=300)
    nodes = {r.nid: Node(id=r.nid, task_id="t", commit="c", parent_id=None if r.feat.gen == "root" else "c1",
                         strategy=Strategy.EXPLORE, operator=Operator.DRAFT, family=r.feat.cats["family"],
                         mechanism_tag="x", diff_hash=r.nid, status=NodeStatus.SCREENED_OUT, created_ts=r.ts)
             for r in rows.values()}
    ctx = SimpleNamespace(cfg=SimpleNamespace(levers=SimpleNamespace(vocabulary=("lr",))))
    book = P.FeatureBook(ctx, None, nodes, lambda n: "opt")
    rows = {k: P.NodeRow(r.nid, r.ts, book.of(nodes[k]), r.screen, r.sel, r.own, r.fresh, r.dec_all, r.own_all)
            for k, r in rows.items()}
    f = P.build(rows, ("gen_family", "family", "parent"), 0.0, 0.0005 ** 2, 15, book, sweeps=15, mde=0.002, gpu_h=2.0)
    cells = f.allocation([ROOT_NODE_ID, "c1"])
    assert {c["carrier"] for c in cells} == {ROOT_NODE_ID, "c1"} and all(0.0 <= c["p_mde"] <= 1.0 for c in cells)
    fc = f.forecast(cells)
    assert [x["n"] for x in fc] == [10.0, 25.0, 50.0, 100.0]
    assert all(a["e_max"] <= b["e_max"] + 1e-12 for a, b in zip(fc, fc[1:]))          # more nodes, higher max
    assert fc[-1]["gpu_h"] == pytest.approx(200.0)
    good = next(n for n in nodes.values() if n.family == "good")
    bad = next(n for n in nodes.values() if n.family == "bad")
    assert f.candidate(good)[0] > f.candidate(bad)[0]                      # the ranking score
    ex = f.explain(good.id)
    assert ex["contributions"] and ex["prior_sd"] > 0
    assert set(f.gate) >= {"all"} and all(isinstance(m, mixture.Mixture) for m in f.gate.values())


def test_features_are_read_from_the_node_record():
    n = Node(id="n2", task_id="t", commit="c", parent_id="n1", strategy=Strategy.EXPLORE, operator=Operator.DRAFT,
             family="sampling", mechanism_tag="x", diff_hash="h", status=NodeStatus.SCREENED_OUT, created_ts=5.0,
             levers=LeverTags({"lr": 1}),
             notes="request=r brief=frontier backend=x web=1 searches=3 fetches=2 stack_legs=a,b mtype=data_sampling")
    p = Node(id="n1", task_id="t", commit="c", parent_id=None, strategy=Strategy.EXPLORE, operator=Operator.DRAFT,
             family="f", mechanism_tag="y", diff_hash="g", status=NodeStatus.CONFIRMED, created_ts=1.0)
    ctx = SimpleNamespace(cfg=SimpleNamespace(levers=SimpleNamespace(vocabulary=("lr", "wd"))))
    book = P.FeatureBook(ctx, None, {"n1": p, "n2": n}, lambda x: "data_sampling",
                         diag=lambda nid: {"lr_end_frac": 0.9} if nid == "n1" else None)
    f = book.of(n)
    assert f.gen == "carrier" and f.cats["parent"] == "n1" and f.cats["brief"] == "frontier"
    assert f.cats["web"] == "1:deep" and f.cats["gen_mtype"] == "carrier/data_sampling"
    assert f.nums["structure"][0] == 2.0 and f.nums["levers"] == (1.0, 0.0)
    assert f.nums["diagnostics"][0] == 0.9 and f.nums["diagnostics"][-1] == 1.0
    assert book.of(n, parent=ROOT_NODE_ID).gen == "root"
    assert f.cats["init"] == "base"                                  # the root's init


def test_init_feature_follows_an_init_swap_down_its_lineage():
    """An init swap (the driver's ``init=`` note) is a creation-time feature of the swap node and every descendant,
    so a lineage on another pretrained model is never pooled with the root's."""
    mk = lambda nid, parent, notes="": Node(                                  # noqa: E731
        id=nid, task_id="t", commit="c", parent_id=parent, strategy=Strategy.MOONSHOT, operator=Operator.DRAFT,
        family="architecture", mechanism_tag=nid, diff_hash=nid, status=NodeStatus.SCREENED_OUT, created_ts=1.0,
        notes=notes)
    vitb = "https://x/dinov2_vitb14_reg4_pretrain.pth"
    nodes = {"s": mk("s", None, f"request=r mtype=architecture init={vitb}"), "c": mk("c", "s"), "g": mk("g", "c"),
             "o": mk("o", None)}
    ctx = SimpleNamespace(cfg=SimpleNamespace(levers=SimpleNamespace(vocabulary=())))
    book = P.FeatureBook(ctx, None, nodes, lambda x: "architecture")
    assert [book.of(nodes[k]).cats["init"] for k in "scgo"] == [vitb, vitb, vitb, "base"]
    assert book.of(nodes["o"], parent="c").cats["init"] == vitb     # a counterfactual parent carries its init


# ------------------------------------------------------------------------------------------------ world wiring
def _lab(ctx, store, n_null=40):
    lab = Lab(ctx, store, seed=5)
    for i in range(n_null):
        lab.node(f"z{i:02d}")
        lab.measure(f"z{i:02d}", {}, (300 + 2 * (i // 4), 301 + 2 * (i // 4)))
    lab.node("w1")
    lab.measure("w1", {"alpha": 0.03, "beta": 0.03, "gamma": 0.03}, (400, 401))
    return lab


def test_world_fit_puts_every_scored_node_through_the_gate_prior(ctx, store):
    _lab(ctx, store)
    tree = ComputedWorldModel(with_cfg(ctx, prior={"model": "tree"})).fit(store, persist=False)
    snap = ComputedWorldModel(with_cfg(ctx, prior={"wm_min_nodes": 20})).fit(store, persist=False)
    assert tree.eb == {} and tree.wm is None and all(p.gain_p is None for p in tree.nodes.values())
    assert snap.wm is not None and "root" in snap.eb and snap.eb_raw["w1"][0] == "root"
    nulls = [k for k in snap.nodes if k.startswith("z")]
    lucky = max(nulls, key=lambda k: snap.eb_raw[k][1])                       # the luckiest null screen
    assert snap.nodes[lucky].p_gain < tree.nodes[lucky].p_gain
    assert snap.nodes["w1"].p_gain > 0.9 and snap.nodes["w1"].p_positive > 0.9     # a real effect survives
    cell = snap.eb_raw["w1"][0]
    assert snap.nodes["w1"].gain_prior == pytest.approx(mixture.prob_greater(snap.eb[cell][0]))
    for k in nulls:                                     # U of a root child = its gate increment; P(U>0) = P(gain>0)
        assert snap.nodes[k].utility.mean == pytest.approx(snap.nodes[k].increment.mean)
        assert snap.nodes[k].p_positive == pytest.approx(snap.nodes[k].p_gain)
    assert ComputedWorldModel(with_cfg(ctx, prior={"wm_min_nodes": 50})).fit(store, persist=False).wm is None
    pred = ComputedWorldModel(with_cfg(ctx, prior={"wm_min_nodes": 20})).predictive(
        snap, family="optimization", tag="lr", levers=LeverTags(), node=store.get_node("w1"))
    assert pred.sd > 0 and math.isfinite(pred.mean)


def test_pooled_gain_reposteriors_the_pooled_raw_increment():
    rng = random.Random(8)
    _, obs = _population(rng, 500, pi0=0.8, mu=0.0, tau=0.005)
    m = mixture.fit_grid(obs)
    q = Posterior(0.0, 0.001, 2)
    node = NodePosterior("n1", q, q, {}, 0.5, 0.5, False, gain_p=0.1)
    snap = SimpleNamespace(nodes={"n1": node}, own_confirm={"n1": (0.004, 4)}, eb={"root": (m.prior, 500)},
                           eb_raw={"n1": ("root", 0.003, 0.002 ** 2)}, incumbent_id=ROOT_NODE_ID)
    w, wc = 1.0 / 0.002 ** 2, 4 / 0.003 ** 2
    pooled = ((w * 0.003 + wc * 0.004) / (w + wc), 1.0 / (w + wc))
    assert pooled_gain(snap, "n1", 0.003) == pytest.approx(mixture.prob_greater(mixture.posterior(m, pooled)))
    assert pooled_gain(snap, "n1", 0.003) > mixture.prob_greater(mixture.posterior(m, (0.003, 0.002 ** 2)))
    assert math.isfinite(pooled_gain(SimpleNamespace(nodes={"n1": node}, own_confirm={}), "n1", 0.003))


def test_explain_and_allocation_text_for_briefs_and_the_explain_hook(ctx, store):
    _lab(ctx, store)
    snap = ComputedWorldModel(with_cfg(ctx, prior={"wm_min_nodes": 20})).fit(store)
    txt = P.explain_text(snap, store, "w1")
    assert "model expected" in txt and "measured" in txt and "P(gain>0)" in txt
    assert P.explain_text(snap, store, "nope") == ""
    assert "allocation" in P.allocation_text(snap) and "Plateau forecast" in P.allocation_text(snap)
    assert P.allocation_text(SimpleNamespace(wm=None)) == ""
    assert store.get_state(P.FORECAST_KEY)["forecast"]                  # logged with the evaluation
