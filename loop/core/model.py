"""loop.core.model — the statistical core (docs/loop-core-v3.md §2; LOCKED from self-modification).

Model per target t (an axis, the panel, or the screening composite): ``y[c,s] = mu[c] + b[s] + e[c,s]`` with
``Var b = rho*sigma2`` and ``Var e = (1-rho)*sigma2``.

- :func:`fit_noise`        sigma2 = pooled within-config variance; rho from same-seed paired differences of CLEAN
                           pairs (neither run resumed), with its dfs and the resumed-pair variance factor
- :func:`pairs_by_change`  the (child, comparator) pairs a rho is fitted on: NEAR = one config key apart, config only;
                           FAR = every other change (code, several keys, crossovers, ports) and two-step relatives
- :func:`cuped`            a child's increment over its parent's multi-seed mean, seed luck regressed out
- :func:`composite_weights` official weights x per-axis reliability (screening only)
- :func:`fit_prior`        hierarchical EB running means global > carrier > carrier x family > carrier x family x lane,
                           plus an optional val-loss covariate
- :func:`posterior`        Student-t prior (heavy-tailed effects) x Gaussian likelihood; :func:`max_of_k`

Stdlib only; every function is pure."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Iterable, Mapping, Sequence

from loop import project as P
from loop.core.data import AXES, OFFICIAL_WEIGHTS, PANEL, ROOT, Node, Obs

COMPOSITE = "composite"
SELECTION_STAGES = frozenset({"screen", "promote"})   # a non-root parent's own selection runs: biased upward
TAU2_MIN = 1e-7                                       # (0.0003)^2: a cell is never treated as exactly known
KAPPA_MAX = 1e6
NU = 3.0                                              # Student-t prior dfs (posterior); None = Gaussian
SPREAD = "mad"                                        # how a cell's node-level tau2 is measured (SPREADS)
NEAR_MIN_DF = 20                                      # a near-rho fit needs this many paired dfs, else far is used
RESUME_MIN_N = 8                                      # resumed paired seeds needed to calibrate the resumed factor
RESUME_MAX = 50.0                                     # the calibrated resumed factor is clipped to [1, RESUME_MAX]


@dataclass(frozen=True)
class Noise:
    sigma2: Mapping[str, float]
    rho: Mapping[str, float]
    #: dfs of the same-seed paired-difference variance 2(1-rho)sigma2 (the rho fit); empty = unknown (treated as exact)
    df: Mapping[str, float] = field(default_factory=dict, compare=False)
    #: Var d of a pair with a resumed run / Var d of a clean pair, fitted (empty = not calibrated: 1/(1-rho))
    resume: Mapping[str, float] = field(default_factory=dict, compare=False)

    def sd(self, target: str) -> float:
        return math.sqrt(self.sigma2[target])

    def sd_d(self, target: str) -> float:
        """sd of a clean same-seed paired difference, ``sqrt(2(1-rho)sigma2)``."""
        return math.sqrt(2.0 * (1.0 - self.rho[target]) * self.sigma2[target])

    def resumed_factor(self, target: str) -> float:
        """Variance factor of a pair where either run resumed: calibrated (``resume``: resumed vs clean same-seed
        pairs; it may exceed ``1/(1-rho)`` -- on the 10-12 ledger 2.75 vs 1.74, 10-13 2.8: a resumed run adds noise of its
        own; NOT the 1.27 single-run variance ratio of the 10-13 meta-analysis, a different quantity),
        else ``1/(1-rho)`` (the pair as if it shared no seed effect: Var d = 2 sigma2)."""
        f = self.resume.get(target)
        return 1.0 / (1.0 - self.rho[target]) if f is None else min(max(f, 1.0), RESUME_MAX)


@dataclass(frozen=True)
class Estimate:
    delta: float      # child mean (CUPED-adjusted) minus the parent's multi-seed mean
    var: float        # its sampling variance V
    n: int            # child seeds
    paired: int       # of which CUPED-paired with a parent run at the same seed


# -- cells ------------------------------------------------------------------------------------------------------
def _y(o: Obs, target: str, weights: Mapping[str, float] | None) -> float:
    return o.value(target, weights if target == COMPOSITE else None)


def seed_means(obs: Iterable[Obs], target: str, weights: Mapping[str, float] | None = None,
               *, paired_only: bool = False) -> dict[int, float]:
    """seed -> mean of the target over the runs at that seed (duplicates across stages averaged).
    ``paired_only``: resumed runs left out (a resumed run is an observation, never a CUPED pair)."""
    acc: dict[int, list[float]] = {}
    for o in obs:
        if not (paired_only and o.resumed):
            acc.setdefault(o.seed, []).append(_y(o, target, weights))
    return {s: sum(v) / len(v) for s, v in acc.items()}


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def _var(xs: Sequence[float]) -> float:
    m = _mean(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


# -- noise ------------------------------------------------------------------------------------------------------
def _cells(obs: Sequence[Obs], target: str, weights: Mapping[str, float] | None) -> dict[int, tuple[float, bool]]:
    """seed -> (mean over the CLEAN runs at that seed, else over the resumed ones; True when only resumed)."""
    clean, res = seed_means([o for o in obs if not o.resumed], target, weights), \
        seed_means([o for o in obs if o.resumed], target, weights)
    return {**{s: (y, True) for s, y in res.items()}, **{s: (y, False) for s, y in clean.items()}}


def throughput_clean(by_node: Mapping[str, Sequence[Obs]], tol: float | None = None) -> Mapping[str, Sequence[Obs]]:
    """The noise fit's runs without the THROUGHPUT OUTLIERS (``loop.project.throughput``, wall-clock budgets): a run
    whose step count deviates from its config's median step count by more than ``tol`` (a fraction; default the
    project's, 0 = off: ``by_node`` unchanged) measured a different budget, not seed noise. Within one config the
    step count varies only with the node / load, so the median is a treatment-free reference; runs without a step
    count are kept."""
    tol = P.throughput()[1] if tol is None else tol
    if tol <= 0:
        return by_node
    out: dict[str, Sequence[Obs]] = {}
    for n, obs in by_node.items():
        st = sorted(o.steps for o in obs if o.steps is not None)
        if not st:
            out[n] = obs
            continue
        med = st[len(st) // 2] if len(st) % 2 else (st[len(st) // 2 - 1] + st[len(st) // 2]) / 2
        out[n] = [o for o in obs if o.steps is None or abs(o.steps / med - 1.0) <= tol]
    return out


def fit_noise_target(by_node: Mapping[str, Sequence[Obs]], target: str,
                     weights: Mapping[str, float] | None = None,
                     parents: Mapping[str, str | None] | Iterable[tuple[str, str]] | None = None,
                     ) -> tuple[float, float, float, float | None]:
    """(sigma2, rho, rho dfs, resumed factor) of one target. sigma2: pooled within-config variance over configs with
    >= 2 seeds. rho: a config pair sharing >= 2 CLEAN seeds (neither run resumed) gives d_s = y[c,s] - y[c',s] with
    Var d = 2(1-rho)sigma2 (the shared seed effect cancels). ``parents`` given (a child -> parent map, or (a, b)
    pairs): only those pairs -- e.g. the near / far pairs of :func:`pairs_by_change`; the seed effect is
    config-specific (distant configs share less of it: ledger 10-08 panel rho .38 child/parent, .20 all pairs).
    Resumed factor: on the same pairs, the mean squared deviation of a resumed seed's d from the pair's clean mean
    (minus that mean's own variance) over the clean Var d; None with fewer than ``RESUME_MIN_N`` resumed seeds."""
    by_node = throughput_clean(by_node)
    cells = {n: seed_means(obs, target, weights) for n, obs in by_node.items()}
    ss = df = 0.0
    for ys in cells.values():
        if len(ys) >= 2:
            v = list(ys.values())
            ss, df = ss + _var(v) * (len(v) - 1), df + len(v) - 1
    if df == 0:
        raise ValueError(f"fit_noise: no config with >= 2 seeds for {target!r}")
    sigma2 = ss / df
    if parents is not None:
        items = parents.items() if isinstance(parents, Mapping) else parents
        pairs = sorted({(c, p) for c, p in items if p is not None and c in cells and p in cells and c != p})
    else:
        at_seed: dict[int, list[str]] = {}
        for n, ys in cells.items():
            for s in ys:
                at_seed.setdefault(s, []).append(n)
        pairs = sorted({ab for ns in at_seed.values() for ab in combinations(sorted(ns), 2)})
    flagged = {n: _cells(by_node[n], target, weights) for n in {x for ab in pairs for x in ab}}
    dss = ddf = 0.0
    resumed: list[tuple[float, int]] = []          # (squared deviation from the pair's clean mean, clean n)
    for a, b in pairs:
        ca, cb = flagged[a], flagged[b]
        both = [s for s in ca if s in cb]
        clean = [ca[s][0] - cb[s][0] for s in both if not ca[s][1] and not cb[s][1]]
        if len(clean) >= 2:
            dss, ddf = dss + _var(clean) * (len(clean) - 1), ddf + len(clean) - 1
            m = _mean(clean)
            resumed += [((ca[s][0] - cb[s][0] - m) ** 2, len(clean)) for s in both if ca[s][1] or cb[s][1]]
    rho = min(0.99, max(0.0, 1.0 - dss / ddf / (2 * sigma2))) if ddf else 0.0
    factor = None
    if ddf and len(resumed) >= RESUME_MIN_N:
        vc = 2 * (1 - rho) * sigma2
        vr = _mean([x - vc / n for x, n in resumed])
        factor = max(1.0, vr / vc)
    return sigma2, rho, ddf, factor


def root_seeds(led: object) -> int:
    """Done seeds of the ledger's current base root in its observations (``led``: a core.data Ledger)."""
    root = getattr(led, "base_root", ROOT)
    return len({o.seed for o in getattr(led, "obs", ()) if o.node == root})


def floor_sigma2(target: str, sigma2: float, n_root: int) -> float:
    """``sigma2`` raised to the installed floor (``loop.project.noise_floor_seeds``) while the root has fewer done
    seeds than that; unchanged when off (nanopath), past the seed count, or no floor is declared for ``target``."""
    k, floor = P.noise_floor()
    f = floor.get(target)
    return sigma2 if not k or n_root >= k or f is None else max(sigma2, f * f)


def floor_noise(nz: "Noise", n_root: int) -> "Noise":
    """:func:`floor_sigma2` on every target of ``nz`` (rho / dfs unchanged)."""
    if not P.noise_floor()[0]:
        return nz
    from dataclasses import replace as _replace
    return _replace(nz, sigma2={t: floor_sigma2(t, v, n_root) for t, v in nz.sigma2.items()})


def fit_noise(obs: Iterable[Obs], targets: Iterable[str] | None = None,
              weights: Mapping[str, float] | None = None,
              parents: Mapping[str, str | None] | Iterable[tuple[str, str]] | None = None) -> Noise:
    if targets is None:
        targets = (*AXES, PANEL)                      # call time: the installed project's axes (loop.project)
    by_node: dict[str, list[Obs]] = {}
    for o in obs:
        by_node.setdefault(o.node, []).append(o)
    pairs = list(parents.items()) if isinstance(parents, Mapping) else (None if parents is None else list(parents))
    fits = {t: fit_noise_target(by_node, t, weights, pairs) for t in targets}
    return Noise({t: f[0] for t, f in fits.items()}, {t: f[1] for t, f in fits.items()},
                 {t: f[2] for t, f in fits.items()}, {t: f[3] for t, f in fits.items() if f[3] is not None})


def merge(*fits: Noise) -> Noise:
    """One Noise from several (later fits win per target)."""
    out: list[dict] = [{}, {}, {}, {}]
    for nz in fits:
        for d, x in zip(out, (nz.sigma2, nz.rho, nz.df, nz.resume)):
            d.update(x)
    return Noise(*out)


def transport_resume(nz: Noise, pooled: Noise) -> Noise:
    """``nz`` with each target's uncalibrated resumed factor taken from ``pooled`` (a wider fit) as a DROP in ρ:
    a resumed pair's ρ_r = ρ_pooled - (1 - ρ_pooled)(f_pooled - 1), and ``nz``'s factor = (1 - ρ + drop)/(1 - ρ).
    (A ratio would not transport: the same resume noise costs a ρ .93 pair 3x and a ρ .4 pair 1.2x.)"""
    out = dict(nz.resume)
    for t, f in pooled.resume.items():
        if t not in out and t in nz.rho and t in pooled.rho:
            drop = (1.0 - pooled.rho[t]) * (f - 1.0)
            out[t] = (1.0 - nz.rho[t] + drop) / (1.0 - nz.rho[t])
    return Noise(nz.sigma2, nz.rho, nz.df, out)


def pairs_by_change(nodes: Mapping[str, Node], near: Iterable[str]) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """(near pairs, far pairs) for the rho fits. ``near``: the nodes ONE config key from their parent with no code
    change (the caller classifies, e.g. tick from the adapter's configs + diff). Near = (child, parent) of those.
    Far = (child, parent) of every other child (code changes, several keys, crossovers, ports) plus every two-step
    pair (grandparent): what a test of a non-trivial change, or against a non-parent incumbent, pairs."""
    near = set(near)
    np, fp = [], []
    for k, n in nodes.items():
        if n.parent is None:
            continue
        (np if k in near and not n.port_of else fp).append((k, n.parent))
        if (p := nodes.get(n.parent)) is not None and p.parent is not None:
            fp.append((k, p.parent))
    return np, fp


# -- composite --------------------------------------------------------------------------------------------------
def reliability(by_node: Mapping[str, Sequence[Obs]], axis: str, sigma2: float, min_seeds: int = 3) -> float:
    """1 - (noise variance of a config mean) / (between-config variance of config means), configs >= min_seeds."""
    means, nvar = [], []
    for obs in by_node.values():
        ys = seed_means(obs, axis)
        if len(ys) >= min_seeds:
            means.append(_mean(list(ys.values())))
            nvar.append(sigma2 / len(ys))
    if len(means) < 3:
        return 0.0
    return max(0.0, 1.0 - _mean(nvar) / _var(means))


def composite_weights(obs: Iterable[Obs], noise: Noise, min_seeds: int = 3) -> dict[str, float]:
    """``w_a ∝ w0_a * r_a`` (normalised to sum 1); falls back to the official weights if nothing is reliable.
    ``loop.project.composite: official`` pins it to the official weights (the screening composite IS the score)."""
    if P.composite_mode() == "official":
        return dict(OFFICIAL_WEIGHTS)
    by_node: dict[str, list[Obs]] = {}
    for o in obs:
        by_node.setdefault(o.node, []).append(o)
    raw = {a: OFFICIAL_WEIGHTS[a] * reliability(by_node, a, noise.sigma2[a], min_seeds) for a in AXES}
    tot = sum(raw.values())
    return {a: w / tot for a, w in raw.items()} if tot > 0 else dict(OFFICIAL_WEIGHTS)


# -- parent mean + CUPED ----------------------------------------------------------------------------------------
def parent_mean(obs: Sequence[Obs], target: str, weights: Mapping[str, float] | None = None) -> tuple[float, int]:
    """(mean, seeds) of a parent over its unbiased seeds: a non-root parent's own screen/promote runs (what it
    was SELECTED on) are left out while >= 2 other seeds remain."""
    unbiased = [o for o in obs if o.node == ROOT or o.stage not in SELECTION_STAGES]
    ys = seed_means(unbiased if len(seed_means(unbiased, target, weights)) >= 2 else obs, target, weights)
    if not ys:
        raise ValueError("parent_mean: no parent observation")
    return _mean(list(ys.values())), len(ys)


def cuped(child: Sequence[Obs], parent: Sequence[Obs], target: str, noise: Noise,
          weights: Mapping[str, float] | None = None) -> Estimate | None:
    """``ỹ[c,s] = y[c,s] - θ(y[p,s] - μ̂_p)`` where p ran non-resumed at s (θ = ρ), ``δ̂ = mean ỹ - μ̂_p``,
    ``V = Σ var_s / n² + (1 - θ·paired/n)² σ²/n_p`` with var_s = σ²(1-ρ²) paired, σ² unpaired. None: no data.
    ``noise`` must be the fit of the child's CHANGE CLASS (:func:`pairs_by_change`): with a near ρ (.94) on a code
    change whose true ρ is .4, ``var_s`` = σ²(1+θ²-2θρ) is 1.13σ², not the .12σ² claimed -- V understated 9.6x."""
    if not child or not parent:
        return None
    s2, rho = noise.sigma2[target], noise.rho[target]
    mu_p, n_p = parent_mean(parent, target, weights)
    p_at = seed_means(parent, target, weights, paired_only=True)
    c_at = seed_means(child, target, weights)
    c_pairable = seed_means(child, target, weights, paired_only=True)
    adj, var_sum, paired = [], 0.0, 0
    for s, y in c_at.items():
        if s in p_at and s in c_pairable:
            adj.append(y - rho * (p_at[s] - mu_p))
            var_sum, paired = var_sum + s2 * (1 - rho ** 2), paired + 1
        else:
            adj.append(y)
            var_sum += s2
    n = len(adj)
    return Estimate(_mean(adj) - mu_p, var_sum / n ** 2 + (1 - rho * paired / n) ** 2 * s2 / n_p, n, paired)


# -- hierarchical prior -----------------------------------------------------------------------------------------
#: the prior's hierarchy, coarse to fine (global first). Carrier first: the base a node builds on is the strongest
#: split (ledger 10-08: root children +0.004 fresh, carrier children ~0); the design's lane-first order ranks worse
CELL_ORDER = ("carrier", "family", "lane")


def cell_of(node: Node) -> tuple[str, str, str]:
    return tuple(getattr(node, k) for k in CELL_ORDER)        # type: ignore[return-value]


@dataclass(frozen=True)
class PriorPoint:
    cell: tuple[str, str, str]      # cell_of(node)
    delta: float
    var: float
    z: float | None = None          # val-loss covariate: child - parent val_loss_q25 (None: unavailable)
    w: float = 1.0                  # weight in the cell means: < 1 for discounted (old base epoch) evidence


@dataclass(frozen=True)
class Prior:
    means: Mapping[tuple[str, ...], float]
    tau2: Mapping[tuple[str, ...], float]
    kappa: tuple[float, ...] = ()
    gamma: float = 0.0              # per standard deviation of z
    z_sd: float = 1.0

    def cell(self, cell: tuple[str, str, str], z: float | None = None) -> tuple[float, float, int]:
        """(prior mean, prior variance tau2, depth used) of the finest known level of ``cell``."""
        for depth in (3, 2, 1, 0):
            key = tuple(cell[:depth])
            if key in self.means:
                return self.means[key] + self.gamma * (z or 0.0) / self.z_sd, self.tau2[key], depth
        raise KeyError("prior has no global level")


def _wmean(points: Sequence[PriorPoint]) -> float:
    """Weighted mean of δ̂ (the plain mean when every weight is 1)."""
    return sum(p.w * p.delta for p in points) / sum(p.w for p in points)


def _median(xs: Sequence[float]) -> float:
    ys = sorted(xs)
    return (ys[(len(ys) - 1) // 2] + ys[len(ys) // 2]) / 2


def _wmedian(points: Sequence[PriorPoint]) -> float:
    """The w-weighted median of δ̂ (equal weights: :func:`_median`; an exact half at a boundary averages the two
    neighbours)."""
    ps = sorted(points, key=lambda p: p.delta)
    tot, acc = sum(p.w for p in ps), 0.0
    for i, p in enumerate(ps):
        acc += p.w
        if acc > tot / 2 + 1e-12 * tot:
            return p.delta
        if abs(acc - tot / 2) <= 1e-12 * tot:
            return (p.delta + ps[i + 1].delta) / 2 if i + 1 < len(ps) else p.delta
    return ps[-1].delta


#: the spread of δ̂ in a cell: ``all`` = the sample variance; ``mad`` = (1.4826 MAD)², the spread of the bulk,
#: blind to the crashed / diverged tail that otherwise inflates τ² and under-shrinks every selected node
SPREADS = {"all": lambda xs: _var(xs),
           "mad": lambda xs: (1.4826 * _median([abs(x - _median(xs)) for x in xs])) ** 2}


MED_CHI2_1 = 0.4549364231195724     # the median of chi-square with 1 df


def tau2_robust(ms: Sequence[float], vs: Sequence[float], tau2_min: float = TAU2_MIN) -> float:
    """The between-node scale² of estimates ``ms`` with sampling variances ``vs`` (>= ``model.TAU2_MIN``): the τ² at
    which the MEDIAN of (m_i − median m)² / (τ² + V_i) equals χ²₁'s median (a robust Paule–Mandel; the bisection is on
    a decreasing function). 11-10 nanobrain: the previous (1.4826 MAD)² − MEAN V mixed a robust spread with a
    non-robust noise term -- on a heteroscedastic set (34 of 51 nodes paired on 1-2 seeds, V up to 5x a 5-seed node's)
    the MAD of the mixture is far below its mean variance, so τ² hit the floor (scale .0003) and every node, a node's
    +.0060 on 5/5 paired seeds included, shrank to the median gain (-.0004; P(promote) .50 -> .06), flipping as
    1-seed screens came and went."""
    pts = [(m, v) for m, v in zip(ms, vs) if math.isfinite(m) and math.isfinite(v) and v >= 0]
    if len(pts) < 2:
        return tau2_min
    loc = _median([m for m, _ in pts])

    def stat(t2: float) -> float:
        return _median([(m - loc) ** 2 / max(t2 + v, 1e-300) for m, v in pts])

    if stat(tau2_min) <= MED_CHI2_1:
        return tau2_min
    lo, hi = tau2_min, max(tau2_min, max((m - loc) ** 2 for m, _ in pts) / MED_CHI2_1)
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if stat(mid) > MED_CHI2_1:
            lo = mid
        else:
            hi = mid
    return hi


def fit_prior(points: Sequence[PriorPoint], *, tau2_min: float = TAU2_MIN, k_tau: float = 5.0,
              ridge: float = 10.0, spread: str = SPREAD) -> Prior:
    """Running means shrunk level by level: ``m_g = (n_g d̄_g + κ m_parent)/(n_g + κ)`` (``n_g`` = Σ w and d̄_g
    the w-weighted MEDIAN with the default ``mad`` spread (v3 review K7; the w-weighted mean with ``all``): an
    old-base-epoch point counts w < 1), ``κ = s²/τ²_cell``
    (s² pooled within-cell variance of δ̂, τ²_cell the between-cell variance by moments). Node-level prior
    variance ``τ²_g = max(spread(δ̂ in g) - mean V, τ²_min)``, shrunk to the parent level with ``k_tau``
    pseudo-dfs. ``gamma``: ridge slope of (δ̂ - m_cell) on standardised z (``ridge`` in units of n)."""
    if not points:
        return Prior({(): 0.0}, {(): 1e-4})
    spr = SPREADS[spread]
    # v3 review K7: the cell LOCATION is as robust as its spread -- with the MAD-based τ² the location is the weighted
    # median (a few crashed / diverged δ̂ no longer drag every cell mean while τ² ignores them); "all": the mean
    loc = _wmedian if spread == "mad" else _wmean
    unw = _median if spread == "mad" else _mean

    robust = spread == "mad" and P.cell_tau2() == "robust"

    def tau2_raw(ds: list[float], vs: list[float], fallback: float) -> float:
        if len(ds) < 2:
            return fallback
        # 11-10 generalization: with the robust spread the cell tau2 is the robust Paule-Mandel (tau2_robust), not
        # (1.4826 MAD)^2 - MEAN V -- the mix of a robust spread and a non-robust noise term floored tau2 on a
        # heteroscedastic cell (many 1-2 seed nodes), the defect core.bar fixed for its EB priors
        # (loop.project.cell_tau2: "robust" by default for a declared / derived project, nanopath pins the old one)
        return tau2_robust(ds, vs, tau2_min) if robust else max(spr(ds) - _mean(vs), tau2_min)

    d = [p.delta for p in points]
    means = {(): loc(points)}
    tau2 = {(): tau2_raw(d, [p.var for p in points], 1e-4)}
    kappas = []
    for depth in (1, 2, 3):
        groups: dict[tuple[str, ...], list[PriorPoint]] = {}
        for p in points:
            groups.setdefault(p.cell[:depth], []).append(p)
        ss = df = 0.0
        for g in groups.values():
            if len(g) >= 2:
                ss, df = ss + _var([p.delta for p in g]) * (len(g) - 1), df + len(g) - 1
        s2 = ss / df if df else tau2[()]
        dev = [(unw([p.delta for p in g]) - means[k[:-1]]) ** 2 - s2 / len(g) for k, g in groups.items()]
        between = max(_mean(dev), 0.0) if dev else 0.0
        kappa = min(KAPPA_MAX, s2 / between) if between > 0 else KAPPA_MAX
        kappas.append(kappa)
        for k, g in groups.items():
            n, par, ds = len(g), k[:-1], [p.delta for p in g]
            wn = sum(p.w for p in g)
            means[k] = (wn * loc(g) + kappa * means[par]) / (wn + kappa)
            tau2[k] = ((n - 1) * tau2_raw(ds, [p.var for p in g], tau2[par]) + k_tau * tau2[par]) / (n - 1 + k_tau)
    raw_z = [p.z for p in points if p.z is not None]
    if len(raw_z) < 3 or _var(raw_z) <= 0:
        return Prior(means, tau2, tuple(kappas))
    z_sd = math.sqrt(_var(raw_z))
    zs = [(p.z / z_sd, p.delta - means[p.cell]) for p in points if p.z is not None]
    gamma = sum(z * r for z, r in zs) / (sum(z * z for z, _ in zs) + ridge)
    return Prior(means, tau2, tuple(kappas), gamma, z_sd)


# -- posterior + max-of-K ---------------------------------------------------------------------------------------
def posterior(prior_mean: float, tau2: float, est: Estimate, nu: float | None = NU) -> tuple[float, float]:
    """(m, v). ``nu=None``: the Gaussian prior, ``v = 1/(1/τ² + 1/V)``, ``m = v (m_g/τ² + δ̂/V)``.
    ``nu``: a Student-t prior (location m_g, scale τ, ``nu`` dfs) -- effects are heavy-tailed (rare large wins,
    a harmful tail), so a far-out estimate keeps most of its size while the bulk is shrunk; moments by quadrature."""
    if nu is None:
        v = 1.0 / (1.0 / tau2 + 1.0 / est.var)
        return v * (prior_mean / tau2 + est.delta / est.var), v
    s, sd = math.sqrt(tau2), math.sqrt(est.var)
    lo = min(prior_mean, est.delta) - 8 * max(s, sd)
    h = (max(prior_mean, est.delta) + 8 * max(s, sd) - lo) / 400
    w0 = w1 = w2 = 0.0
    for i in range(401):
        mu = lo + i * h
        p = (1 + ((mu - prior_mean) / s) ** 2 / nu) ** (-(nu + 1) / 2) * math.exp(-0.5 * ((est.delta - mu) / sd) ** 2)
        w0, w1, w2 = w0 + p, w1 + p * mu, w2 + p * mu * mu
    m = w1 / w0
    return m, max(w2 / w0 - m * m, 1e-12)


def _phi(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _Phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def expected_max(k: int, steps: int = 4000) -> float:
    """E[max of k iid N(0,1)] (E_1 = 0, E_4 ≈ 1.029, E_16 ≈ 1.766), Simpson on [-8, 8]."""
    if k <= 1:
        return 0.0
    a, b = -8.0, 8.0
    h = (b - a) / steps
    f = [x * k * _phi(x) * _Phi(x) ** (k - 1) for x in (a + i * h for i in range(steps + 1))]
    return h / 3 * (f[0] + f[-1] + 4 * sum(f[1:-1:2]) + 2 * sum(f[2:-1:2]))


def max_of_k(m: float, v: float, k: int, beta: float = 1.0) -> float:
    """``m_sel = m - β √v E_K``: the best of K siblings competing for one test slot."""
    return m - beta * math.sqrt(v) * expected_max(k)
