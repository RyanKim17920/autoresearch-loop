"""Nonparametric Gaussian-mixture priors for a node's OWN panel increment (the gates' prior; loop.world.predictive
chooses what it is conditioned on). ``fit_grid`` is an NPMLE (Efron-style g-modeling) on a grid of atoms plus an exact
0 -- the null most ideas are --, EM on the weights; a node's posterior is :func:`posterior` of it given the node's own
increment estimate. Under a grid P(gain>0) is weakly identified (a null and a +0.0003 atom look alike) and
E[max(gain, 0)] (:func:`expected_gain`) is not; the promote gate's value-of-information rule uses that
(loop.seeds.promote_ok).

History: moved from loop/world/eb.py (2026-10-06), whose spike-and-slab EM backfired (one slab fit to every carrier
child became N(-0.013, 0.020): a +0.004 screen read P(gain>0) 0.03) and whose generation split is now one of the
conditionings the predictive model validates out of time."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from loop.world.evalues import phi

Gauss = tuple[float, float]                 # (mean, var)
Comp = tuple[float, float, float]           # (weight, mean, var): var 0 is an atom


@dataclass(frozen=True)
class Mixture:
    """A Gaussian-mixture prior on eps: ``prior`` components (weight, mean, var; var 0 = an atom)."""
    prior: tuple[Comp, ...]
    n: int

    def comps(self) -> tuple[Comp, ...]:
        return self.prior


def _npdf(x: float, v: float) -> float:
    return math.exp(-x * x / (2.0 * v)) / math.sqrt(2.0 * math.pi * v)


GRID_POINTS, TAIL_POINTS = 150, 8


def grid_for(xs: Sequence[Gauss]) -> list[float]:
    """The atoms: a dense core at a quarter of the median observation sd over the 1st..99th percentile (padded by 2
    sd, at most ``GRID_POINTS``), ``TAIL_POINTS`` sparse atoms out to each extreme, and an exact 0 (live 2026-10-06:
    one -0.178 carrier screen stretched a uniform 100-atom grid to 0.0019 steps: no atom for a +0.001 gain)."""
    es, sd = sorted(e for e, _ in xs), sorted(v for _, v in xs)[len(xs) // 2] ** 0.5
    q = lambda f: es[min(len(es) - 1, int(f * len(es)))]       # noqa: E731
    lo, hi = min(q(0.01) - 2.0 * sd, 0.0), max(q(0.99) + 2.0 * sd, 0.0)
    step = max(sd / 4.0, (hi - lo) / GRID_POINTS, 1e-9)
    core = [lo + i * step for i in range(int((hi - lo) / step) + 1)]
    tails = [lo + (es[0] - lo) * k / TAIL_POINTS for k in range(1, TAIL_POINTS + 1)] if es[0] < lo else []
    tails += [hi + (es[-1] - hi) * k / TAIL_POINTS for k in range(1, TAIL_POINTS + 1)] if es[-1] > hi else []
    return sorted({0.0, *(round(x, 9) for x in (*core, *tails))})


def fit_grid(obs: Sequence[Gauss], start: Mixture | None = None, iters: int = 100) -> Mixture | None:
    """NPMLE of the prior on a fixed grid (:func:`grid_for`, plus an exact 0 -- the null): EM on the atom weights,
    the likelihood matrix computed once. Warm-started from ``start`` when it is on the same grid (then a quarter of
    the iterations). None under 2 observations. P(gain>0) under it is weakly identified (a null and a +0.0003 atom
    look alike); E[max(gain, 0)] (:func:`expected_gain`) is not -- the value-of-information gate
    (loop.seeds.promote_ok) uses that."""
    xs = [(e, v) for e, v in obs if v > 0.0 and math.isfinite(e)]
    if len(xs) < 2:
        return None
    grid = grid_for(xs)
    w = [1.0 / len(grid)] * len(grid)
    if start is not None and [m for _, m, _ in start.prior] == grid:     # (atoms are kept even at weight ~0)
        w, iters = [max(c[0], 1e-12) for c in start.prior], max(1, iters // 4)
    lik = [[_npdf(e - g, v) for g in grid] for e, v in xs]
    for _ in range(iters):
        acc = [0.0] * len(grid)
        for row in lik:
            s = sum(a * b for a, b in zip(w, row)) or 1e-300
            for k, a in enumerate(row):
                acc[k] += w[k] * a / s
        w = [a / len(xs) for a in acc]
    return Mixture(tuple((x, g, 0.0) for x, g in zip(w, grid)), len(xs))


def posterior(m: Mixture, g: Gauss) -> tuple[Comp, ...]:
    """The posterior of eps given its increment estimate ``g`` = (e, v): per prior component (weight, mean, var)."""
    e, v = g
    if v <= 0.0:
        return ((1.0, e, 0.0),)
    out = []
    for w, mu, t2 in m.prior:
        if t2 <= 0.0:
            out.append((w * _npdf(e - mu, v), mu, 0.0))
        else:
            pv = 1.0 / (1.0 / t2 + 1.0 / v)
            out.append((w * _npdf(e - mu, t2 + v), pv * (mu / t2 + e / v), pv))
    s = sum(c[0] for c in out)
    if s <= 0.0:                                  # far in a tail: the nearest component takes it all
        k = min(range(len(out)), key=lambda i: abs(e - out[i][1]))
        return tuple((1.0 if i == k else 0.0, c[1], c[2]) for i, c in enumerate(out))
    return tuple((w / s, x, pv) for w, x, pv in out)


def moments(cs: Iterable[Comp]) -> Gauss:
    cs = list(cs)
    m = sum(w * x for w, x, _ in cs)
    return m, max(0.0, sum(w * (v + x * x) for w, x, v in cs) - m * m)


def prob_greater(cs: Iterable[Comp], shift: Gauss = (0.0, 0.0), t: float = 0.0) -> float:
    """P(eps + S > t) for eps ~ the mixture and an independent S ~ N(shift); a point mass gives 0/1 (ties -> 0)."""
    out = 0.0
    for w, x, v in cs:
        m, s2 = x + shift[0], v + shift[1]
        out += w * ((1.0 if m > t else 0.0) if s2 <= 0.0 else phi((m - t) / math.sqrt(s2)))
    return out


def expected_gain(cs: Iterable[Comp]) -> float:
    """E[max(eps, 0)]: P(gain > 0) x the mean gain given one -- the value a confirmation can bank."""
    out = 0.0
    for w, x, v in cs:
        if v <= 0.0:
            out += w * max(x, 0.0)
        else:
            s = math.sqrt(v)
            out += w * (x * phi(x / s) + s * _npdf(x / s, 1.0))
    return out
