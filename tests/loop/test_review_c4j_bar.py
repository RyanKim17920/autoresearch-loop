"""c4j review S1 / S8 (loop.core.bar): mu's empirical-Bayes prior keeps a true-P=.5 candidate's pass rate near alpha
(the flat prior passed ~19% on 4 seeds); epoch_sigma's ValueError after an epoch rebase degrades (beat / ranked /
summary / IncumbentLabless.below_bar never raise)."""

from __future__ import annotations

import dataclasses
import logging
import math
import random

from loop import runner as R
from loop.contracts import RunStatus, Stage
from loop.core import bar as BR
from loop.core.data import AXES, Ledger, Node, Obs
from loop.spec import LablessCfg

from .conftest import InlineExecutor
from .test_runner import _node, _Pending

OLD, NEW = "a" * 40, "b" * 40
SIGMA, MU0, TAU = 0.0033, 0.678, 0.0024           # the 10-13 ledger: epoch sigma, prior median / scale
T = 0.6810443 + 0.004


def _mu_star(p: float = 0.7) -> float:
    lo, hi = T - 0.05, T + 0.05
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if BR.p_median(mid, SIGMA, T) >= p else (mid, hi)
    return hi


def _pass_rate(mu: float, n: int, prior, draws: int, rng: random.Random) -> float:
    """Monte Carlo: n fresh seeds at true mean mu, the rule P(promote) >= .7 on their mean."""
    return sum(BR.p_promote(sum(rng.gauss(mu, SIGMA) for _ in range(n)) / n, SIGMA, n, T, prior=prior) >= 0.7
               for _ in range(draws)) / draws


def test_eb_prior_pass_rate_near_alpha_at_true_p_half():
    rng, prior = random.Random(20261013), (MU0, TAU ** 2)
    flat = _pass_rate(T, 4, None, 1500, rng)
    eb = _pass_rate(T, 4, prior, 1500, rng)
    assert flat > 0.15                    # the bug: ~.19 analytically
    assert eb <= 0.10                     # ~alpha (test.ALPHA .1): ~.065 analytically
    assert _pass_rate(_mu_star() - 0.0005, 4, prior, 1500, rng) <= 0.18     # just below the true-P=.7 mean
    assert _pass_rate(_mu_star() + 0.002, 4, prior, 1000, rng) >= 0.5       # a real beater still passes (~.66)


def test_eb_need_is_monotone_and_above_flat():
    prior = (MU0, TAU ** 2)
    flat, eb = BR.need_mean(SIGMA, 4, T, 0.7), BR.need_mean(SIGMA, 4, T, 0.7, prior=prior)
    assert eb > flat > T
    assert abs(BR.p_promote(eb, SIGMA, 4, T, prior=prior) - 0.7) < 1e-6
    assert BR.need_mean(SIGMA, 8, T, 0.7, prior=prior) < eb                  # more seeds, less shrinkage


def _n(nid: str, parent: str | None, base: str) -> Node:
    return Node(nid, parent, None, "explore", "f", "explore", "root", 0.0, "", "", base, "")


def _obs(node: str, base: str, mean: float, seeds, pool: str = "select", jit: float = 0.002) -> list[Obs]:
    return [Obs(f"{node}:screen:{s}", node, s, "screen", pool, {a: mean + (jit if i % 2 else -jit) for a in AXES},
                mean + (jit if i % 2 else -jit), 1.0, 0.0, 0, None, base) for i, s in enumerate(seeds)]


def _ledger(cands: dict[str, tuple[float, int]], old_seeds: int = 40) -> Ledger:
    nodes = {"root": _n("root", None, OLD), "new": _n("new", None, NEW), **{c: _n(c, "new", NEW) for c in cands}}
    obs = _obs("root", OLD, 0.668, range(1, old_seeds + 1)) + _obs("new", NEW, 0.663, [700001])
    for c, (m, k) in cands.items():
        obs += _obs(c, NEW, m, range(800001, 800001 + k), "test")
    return Ledger(nodes, (), tuple(obs), frozenset(), ((0.0, "root"),), bases=((0.0, "root", OLD), (1.0, "new", NEW)))


def test_mu_prior_needs_enough_nodes_and_honours_zero():
    many = _ledger({f"c{i}": (0.675 + 0.001 * i, 3) for i in range(6)})
    loc, tau2 = BR.mu_prior(many)
    assert 0.675 < loc < 0.681 and tau2 >= BR.M.TAU2_MIN
    assert BR.mu_prior(many, 0) is None
    assert BR.mu_prior(_ledger({"c0": (0.68, 3)})) is None
    cfg = LablessCfg(leader_median=0.6810443)
    assert BR.bar_from(many, cfg).prior is not None
    assert BR.bar_from(many, dataclasses.replace(cfg, eb_min_nodes=0)).prior is None


def test_epoch_sigma_valueerror_degrades(caplog):
    led = _ledger({"x": (0.69, 1), "y": (0.68, 1)})        # right after a rebase: no new-epoch config has 2 seeds
    try:
        BR.epoch_sigma(led)
        raise AssertionError("epoch_sigma should raise here")
    except ValueError:
        pass
    BR._SIGMA_WARNED.discard(led.epoch)
    with caplog.at_level(logging.WARNING, logger="loop.core.bar"):
        s = BR.safe_sigma(led)
        BR.safe_sigma(led)
    assert math.isfinite(s) and s > 0                       # the whole ledger's pooled sigma stands in
    assert sum("no panel sigma" in r.message for r in caplog.records) == 1     # logged once
    bar = BR.bar_from(led, LablessCfg(leader_median=0.6810443))
    assert BR.beat(led, bar, "x") is not None
    assert isinstance(BR.ranked(led, bar), list)              # never raises (x, y: 1 unpaired seed each: unranked)
    assert BR.summary(led, LablessCfg(leader_median=0.6810443)).startswith("bar=Labless")
    bare = _ledger({"x": (0.69, 1)}, old_seeds=1)          # no config with 2 seeds anywhere: nan, nothing beats
    assert math.isnan(BR.safe_sigma(bare))
    b = BR.beat(bare, BR.bar_from(bare, LablessCfg(leader_median=0.6810443)), "x")
    assert b is not None and not b.beats
    BR.summary(bare, LablessCfg(leader_median=0.6810443))


def test_incumbent_labless_below_bar_survives_a_rebase(ctx, store):
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(
        ctx.cfg, labless=dataclasses.replace(ctx.cfg.labless, enabled=True, leader_median=0.6810443)))
    lab = R.IncumbentLabless(ctx, store, InlineExecutor(), clock=lambda: 1000.0,
                             submit=lambda *a: "sub-1")
    R.RunBook(ctx, store, _Pending()).launch(_node(store, "x"), Stage.CONFIRM, 100_000)
    store.update_run("x:confirm:100000", status=RunStatus.DONE)
    out = lab.step("x", _ledger({"x": (0.69, 1)}, old_seeds=1))
    assert out and "not submitted" in out[0]
