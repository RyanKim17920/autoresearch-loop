"""loop.core.test — the sequential fresh-seed test against the incumbent (docs/loop-core-v3.md §4; LOCKED).

Candidate c and the incumbent I run on the same never-used seeds (a TEST block, loop.core.seeds);
``d_s = P[c,s] - P[I,s]`` on the OFFICIAL panel, ``σ_d = sqrt(2(1-ρ)σ²)`` from the fitted panel noise.

Two methods (``test.method``, :data:`METHOD`):

``gs2`` (default) -- a two-look group-sequential one-sided z/t test. Looks at n = K (5) and n = K_MAX (8) paired
seeds; ``Z = Σ(d_s/f_s) / (σ_d √Σ1/f_s)`` (``f_s``: a resumed pair's variance factor). Look 1: PASS at Z >= c1,
FAIL (binding futility) at Z < ``FUTILITY_Z`` (0: a negative mean), else EXTEND by 3 seeds -- the at-most-once
extension IS the second look. Look 2: PASS at Z >= c2, else FAIL. α = .1 is spent by a Lan-DeMets O'Brien-Fleming-type
function (``test.spending`` "obf"; "pocock" selectable): α(t) = 2(1 - Φ(z_{1-α/2}/√t)) at t = 5/8 spends .0375 at
look 1 and the rest at look 2 (df 71: c1 = 1.81, c2 = 1.35 vs the fixed-n t 1.29), so nearly all of the n = 8 power
is kept while a strong candidate still stops at 5. Pocock spends .073 early (c1 = 1.47) and needs c2 = 1.55.
Simulated (σ_d .0033 fitted on 71 dfs, 20k null / 4k alt sims): OBF type-I .100, power +.003 .89 / +.005 1.00,
mean seeds 6.1 / 5.2; Pocock .101, .86 / 1.00, 5.8 / 5.1. The incumbent's own gains were ~+.003, so power there
is the scarce resource and the 0.4 seed per arm Pocock saves is not worth 3 points of it: OBF. c1 / c2 are the
exact boundaries of the bivariate t (σ_d is a pooled estimate with ``df`` dfs, independent of the block: the
candidate's own runs and the block's seeds are left out of its fit) with the futility bound binding, so the simulated type-I is α. σ_d is used as fitted once
its dfs exceed ``UCB_DF_MAX`` (50; live pooled df ~72), else at its 80% upper confidence bound (:func:`sd_for`).
EB blend (``test.sd_prior_df`` ν0 = 4, :func:`blend_sd`): at each look σ²_used = (ν0 σ²_pooled + (n-1) s²_own) /
(ν0 + n - 1), boundaries at t dfs ν0 + k - 1 (look 1) / ν0 + k_max - 1 (look 2), so a candidate pair noisier than
the pool (the pooled pairs are heterogeneous) widens its own σ_d. Simulated at σ_d .0033 / 71 dfs: type-I .093 at
the pooled σ_d, .116 at 1.25x it (no blend: .160); power +.003 .87 / +.005 1.00 / +.009 1.00 (+.003 at 1.25x: .78).

``msprt`` (the 10-12 test) -- a mixture SPRT (anytime-valid: a mean of likelihood ratios is a nonnegative martingale
under δ = 0, so by Ville P(ever E >= 1/α) <= α however often it is looked at). Far too conservative as run (σ_d
at its UCB, the mixture spread over ½-2 MDE): simulated type-I .013, power +.003 .53 at σ_d .0033 / 71 dfs:

    E_n = mean_{δ ∈ {½, 1, 2}·MDE} exp(Σ_s (δ d_s - δ²/2) / σ_d²),   MDE = 2σ_d/√5

:func:`verdict` -> PASS | FAIL | EXTEND | OPEN (see its docstring). :func:`simulate` gives type-I / power at a known
σ_d; :func:`simulate_fit` with σ_d ESTIMATED (``df`` dfs) and mis-specified (``misfit`` = true / pooled σ_d).

msprt's σ_d is a plug-in estimate (``df`` dfs of the paired-difference variance): :func:`sd_used` inflates it to its
``1 - SD_Q`` upper confidence bound (``ν/χ²_{ν,SD_Q}`` on the variance, Wilson-Hilferty), so a small-df fit cannot
under-state the noise the e-value is a likelihood ratio of (live 10-12: the near ρ .94 came from ONE pair, 4 df)."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any, Mapping, Sequence

from loop import project as P

ALPHA = 0.1             # locked
E_LO = 1.0              # n = K and E below this: FAIL
K = 5                   # default seeds
K_MAX = 8               # extension cap (operator: 10 is too many)
MIN_PASS_N = 4          # a PASS needs at least this many paired seeds
SD_Q = 0.2              # σ_d is used at its 80% upper confidence bound (sd_used)
EXTEND_BY = 3
MAX_EXTENDED = 2        # extended tests open at once
HARM_Z = -2.0           # one-sided z of a paired axis mean that counts as harm
#: axes the harm check guards. Not every axis: the panel already prices each axis at its official weight, and on
#: the ledger a node (the recorded incumbent) is significantly worse than a node on surgen AUC (-.013, z -3.0 over 18
#: fresh seeds) while +.017 slide / +.010 robustness / +.0022 panel -- an all-axes guard would veto that trade.
GUARD_AXES = P.GUARD_AXES          # live (loop.project; nanopath: ("rob",))
#: NO GUARD -> NO TRIPWIRE: when the project has no guard axis (loop.project.guard []) every axis gets a catastrophic
#: veto instead: a paired axis mean below -CATA_SIGMAS x that axis's paired sd FAILs the test (``loop.veto.
#: catastrophic_sigmas``, installed by acquire.configure; 0 = off). Never used with a guard (nanopath: rob).
CATA_SIGMAS = 4.0
MIXTURE = (0.5, 1.0, 2.0)
#: the test (``test.method``): "gs2" two-look group-sequential (default) | "msprt" the mixture SPRT
METHOD = "gs2"
#: gs2: the α-spending function (``test.spending``): "obf" (Lan-DeMets O'Brien-Fleming type) | "pocock"
SPENDING = "obf"
#: gs2: binding futility at look 1 (``test.futility_z``): Z below this FAILs (0 = a non-positive mean)
FUTILITY_Z = 0.0
#: gs2: σ_d is used as fitted once its dfs exceed this (``test.ucb_df_max``), else at its UCB (sd_used)
UCB_DF_MAX = 50.0
#: gs2: the pooled σ_d is a prior worth this many dfs (ν0, ``test.sd_prior_df``) blended with the test's own paired
#: variance (:func:`blend_sd`); boundaries at ν0 + n - 1 dfs (look 1: ν0 + k - 1, look 2: ν0 + k_max - 1). 0 = the
#: pooled σ_d alone. On the 10-13 copy per-pair paired variances fit F(d, ν0) with ν0 ≈ 15 over every epoch (MLE;
#: 20 on pairs with >= 3 seeds) but the epoch's likelihood is flat over ~4-30. 4 is chosen for robustness (inside the
#: epoch's support): a pair 1.25x noisier than the pool (≈ the 91st percentile pair at ν0 15) passes nulls .116
#: (ν0 15: .135, no blend: .160) for 1 point of power at +.003 (.873 vs .880 / .881)
SD_PRIOR_DF = 4.0
METHODS = ("gs2", "msprt")
#: the meta report's guard on the simulated type-I under a 1.25x sigma_d misfit (``test.type1_guard_max``; meta)
TYPE1_GUARD_MAX = 0.2


def configure(cfg: Any) -> None:
    """Install the ``test`` section's keys (method, spending, futility_z, ucb_df_max, type1_guard_max) as constants;
    a config without the section keeps the values in force."""
    global METHOD, SPENDING, FUTILITY_Z, UCB_DF_MAX, TYPE1_GUARD_MAX, SD_PRIOR_DF
    t = getattr(cfg, "test", None)
    if t is None:
        return
    if t.method not in METHODS or t.spending not in ("obf", "pocock"):
        raise ValueError(f"test.method {t.method!r} / test.spending {t.spending!r}")
    METHOD, SPENDING, FUTILITY_Z, UCB_DF_MAX = t.method, t.spending, float(t.futility_z), float(t.ucb_df_max)
    TYPE1_GUARD_MAX = float(getattr(t, "type1_guard_max", TYPE1_GUARD_MAX))
    SD_PRIOR_DF = float(getattr(t, "sd_prior_df", SD_PRIOR_DF))


class Kind(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    EXTEND = "extend"
    OPEN = "open"


@dataclass(frozen=True)
class Verdict:
    kind: Kind
    e: float
    n: int
    mean: float
    reason: str
    harm: tuple[str, ...] = ()
    more: int = 0           # seeds to add (EXTEND) / still missing (OPEN)
    sd: float = 0.0         # the σ_d the e-value used (after sd_used)
    z: float = 0.0          # gs2: the weighted z (t) statistic


def sigma_d(sigma2: float, rho: float) -> float:
    """sd of a same-seed paired difference: Var d = 2(1-ρ)σ²."""
    return math.sqrt(2.0 * (1.0 - rho) * sigma2)


def _z(q: float) -> float:
    """Standard normal quantile (Acklam's rational approximation, |error| < 1.2e-9)."""
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00)
    if q < 0.02425:
        t = math.sqrt(-2 * math.log(q))
        return (((((c[0] * t + c[1]) * t + c[2]) * t + c[3]) * t + c[4]) * t + c[5]) / \
            ((((d[0] * t + d[1]) * t + d[2]) * t + d[3]) * t + 1)
    if q > 1 - 0.02425:
        return -_z(1 - q)
    t = (q - 0.5) ** 2
    return (((((a[0] * t + a[1]) * t + a[2]) * t + a[3]) * t + a[4]) * t + a[5]) * (q - 0.5) / \
        (((((b[0] * t + b[1]) * t + b[2]) * t + b[3]) * t + b[4]) * t + 1)


def var_inflation(df: float | None, q: float = SD_Q) -> float:
    """``ν / χ²_{ν, q}``: the factor taking a ν-df variance estimate to its ``1 - q`` upper confidence bound
    (Wilson-Hilferty). 1 for ``df`` None / inf; df 4 -> 2.0, 20 -> 1.37, 100 -> 1.14 (q = .2)."""
    if df is None or math.isinf(df):
        return 1.0
    nu = max(float(df), 1.0)
    h = 2.0 / (9.0 * nu)
    return 1.0 / max(1.0 - h + _z(q) * math.sqrt(h), 0.05) ** 3


def sd_used(sd: float, df: float | None, q: float = SD_Q) -> float:
    """σ_d at its upper confidence bound given the dfs it was fitted on (None / inf: as fitted)."""
    return sd * math.sqrt(var_inflation(df, q))


def sd_for(sd: float, df: float | None, method: str | None = None) -> float:
    """The σ_d the test uses: gs2 with more than ``UCB_DF_MAX`` dfs as fitted (its boundaries already price the
    estimate's dfs), else at its upper confidence bound (:func:`sd_used`)."""
    if (method or METHOD) == "gs2" and df is not None and df > UCB_DF_MAX:
        return sd
    return sd_used(sd, df)


def e_value(d: Sequence[float], sd: float, k: int = K, infl: Sequence[float] | None = None) -> float:
    """The mixture e-value. ``infl``: per-seed variance factors (a resumed run's pair has Var d = 2σ², i.e.
    1/(1-ρ) times the clean pair's): each term is still a likelihood ratio, so the mixture stays valid."""
    mde = 2.0 * sd / math.sqrt(k)
    f = infl or [1.0] * len(d)
    s, w = sum(x / g for x, g in zip(d, f)), sum(1.0 / g for g in f)
    return sum(math.exp(min(700.0, (m * mde * s - w * (m * mde) ** 2 / 2) / sd ** 2)) for m in MIXTURE) / len(MIXTURE)


def catastrophic(axes: Mapping[str, tuple[Sequence[float], float]], k: float | None = None) -> tuple[str, ...]:
    """Axes whose paired mean dropped by more than ``k`` (default ``CATA_SIGMAS``) x the axis's paired sd (a level
    threshold, not a z: catastrophic only). ``axes``: name -> (diffs, paired sd). Needs >= 2 diffs; k <= 0 = off."""
    k = CATA_SIGMAS if k is None else k
    if k <= 0:
        return ()
    return tuple(sorted(a for a, (d, sd) in axes.items() if len(d) >= 2 and sd > 0 and sum(d) / len(d) < -k * sd))


def harmed(axes: Mapping[str, tuple[Sequence[float], float, Sequence[float]]]) -> tuple[str, ...]:
    """Guard axes whose paired mean is significantly negative (z < HARM_Z).
    ``axes``: name -> (diffs, σ_d of the axis, per-seed variance factors)."""
    out = []
    for a, (d, sd, f) in axes.items():
        if len(d) >= 2 and sd > 0 and sum(d) / len(d) / (sd * math.sqrt(sum(f)) / len(d)) < HARM_Z:
            out.append(a)
    return tuple(sorted(out))


# -- gs2: two-look group-sequential boundaries ------------------------------------------------------------------
def _phi(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def spent(alpha: float, t: float, spending: str) -> float:
    """Cumulative one-sided α spent by information fraction ``t``: Lan-DeMets O'Brien-Fleming type
    ``2(1 - Φ(z_{1-α/2}/√t))`` ("obf") or Pocock type ``α ln(1 + (e-1)t)`` ("pocock")."""
    if t >= 1.0:
        return alpha
    if spending == "pocock":
        return alpha * math.log(1.0 + (math.e - 1.0) * t)
    return 2.0 * (1.0 - _cdf(_z(1.0 - alpha / 2.0) / math.sqrt(t)))


def _simpson(f: Any, a: float, b: float, m: int = 80) -> float:
    if b <= a:
        return 0.0
    h = (b - a) / m
    return h / 3.0 * sum((1 if i in (0, m) else 4 if i % 2 else 2) * f(a + i * h) for i in range(m + 1))


def _scale_nodes(df: float | None, m: int = 60) -> list[tuple[float, float]]:
    """(s, weight) quadrature of s = σ̂/σ = √(χ²_ν/ν) (a single node s = 1 for a known σ)."""
    if df is None or math.isinf(df) or df <= 0:
        return [(1.0, 1.0)]
    nu = float(df)
    w = 8.0 / math.sqrt(2.0 * nu)
    lo, hi = max(1e-3, 1.0 - w), 1.0 + w
    logc = math.log(2.0) + (nu / 2.0) * math.log(nu / 2.0) - math.lgamma(nu / 2.0)
    h = (hi - lo) / m
    out = []
    for i in range(m + 1):
        s = lo + i * h
        dens = math.exp(logc + (nu - 1.0) * math.log(s) - nu * s * s / 2.0)
        out.append((s, h / 3.0 * (1 if i in (0, m) else 4 if i % 2 else 2) * dens))
    tot = sum(wt for _, wt in out)
    return [(s, wt / tot) for s, wt in out]


def _bisect(f: Any, lo: float, hi: float, target: float) -> float:
    """x in [lo, hi] with f(x) = target, f decreasing."""
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if f(mid) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


@lru_cache(maxsize=256)
def gs_bounds(alpha: float = ALPHA, t1: float = K / K_MAX, df: float | None = None, spending: str = "obf",
              futility: float = 0.0, df1: float | None = -1.0) -> tuple[float, float]:
    """(c1, c2): the gs2 boundaries on the studentised statistic. Look 1 at information fraction ``t1`` spends
    ``spent(alpha, t1)``; look 2 the rest, with the futility bound (Z1 < ``futility``) binding:
    ``P(T1 >= c1) + P(futility <= T1 < c1, T2 >= c2) = alpha`` where T_k = Z_k / s, corr(Z1, Z2) = √t1 and
    s = √(χ²_df/df) shared by both looks (σ_d estimated once; ``df`` None: σ_d known). ``df1`` (default: ``df``):
    look 1's own dfs when its σ_d has fewer (the EB blend, :func:`blend_sd`: ν0 + k - 1 at look 1, ν0 + k_max - 1
    at look 2): c1 is then its exact marginal t bound and c2 is solved at ``df``."""
    df = None if df is None or math.isinf(df) else float(round(df))
    nodes, r = _scale_nodes(df), math.sqrt(t1)
    q = math.sqrt(1.0 - t1)
    a1 = spent(alpha, t1, spending)
    n1 = nodes if df1 is not None and df1 < 0 else _scale_nodes(None if df1 is None or math.isinf(df1)
                                                                 else float(round(df1)))
    c1 = _bisect(lambda c: sum(w * (1.0 - _cdf(c * s)) for s, w in n1), -10.0, 10.0, a1)

    def p2(c2: float) -> float:
        return sum(w * _simpson(lambda z: _phi(z) * (1.0 - _cdf((c2 * s - r * z) / q)),
                                max(futility * s, -9.0), c1 * s) for s, w in nodes)
    c2 = _bisect(p2, -10.0, 10.0, alpha - a1)
    return round(c1, 6), round(c2, 6)


def blend_sd(d: Sequence[float], sd: float, prior_df: float, infl: Sequence[float] | None = None) -> float:
    """The EB σ_d: the pooled ``sd`` as a prior worth ``prior_df`` dfs (ν0) blended with the test's OWN paired-d
    variance, ``σ² = (ν0 σ²_pooled + (n-1) s²_own) / (ν0 + n - 1)`` (s² precision-weighted by 1/f_s). A pair noisier
    than the pool (the pooled pairs are heterogeneous: per-pair variances ~ F(d, ν0), ν0 ≈ 15-20 on the ledger) then
    widens its own σ_d instead of inheriting the pool's."""
    n = len(d)
    if n < 2 or prior_df <= 0:
        return sd
    f = infl or [1.0] * n
    w = sum(1.0 / g for g in f)
    m = sum(x / g for x, g in zip(d, f)) / w
    s2 = sum((x - m) ** 2 / g for x, g in zip(d, f)) / (n - 1)
    return math.sqrt((prior_df * sd * sd + (n - 1) * s2) / (prior_df + n - 1))


def _z_stat(d: Sequence[float], sd: float, infl: Sequence[float] | None) -> tuple[float, float, float]:
    """(Σ d/f, Σ 1/f, Z) -- the precision-weighted statistic (a resumed pair weighs 1/f)."""
    f = infl or [1.0] * len(d)
    s, w = sum(x / g for x, g in zip(d, f)), sum(1.0 / g for g in f)
    return s, w, (s / (sd * math.sqrt(w)) if w > 0 and sd > 0 else 0.0)


def verdict(d: Sequence[float], sd: float,
            axes: Mapping[str, tuple[Sequence[float], float, Sequence[float]]] | None = None, *,
            infl: Sequence[float] | None = None, extended_open: int = 0,
            prior: tuple[float, float] = (0.0, math.inf), alpha: float = ALPHA,
            k: int = K, k_max: int = K_MAX, e_lo: float = E_LO, planned: int | None = None,
            df: float | None = None, method: str | None = None,
            cata: Mapping[str, tuple[Sequence[float], float]] | None = None) -> Verdict:
    """The test (``method``, default :data:`METHOD`) on the paired panel differences ``d`` seen so far: gs2
    (:func:`verdict_gs2`; ``df`` = the dfs σ_d was fitted on, None = known) or msprt (:func:`verdict_msprt`).
    ``cata`` (axis -> (diffs, paired sd)): axes under the catastrophic veto (:func:`catastrophic`), counted as harm."""
    if (method or METHOD) == "msprt":
        return verdict_msprt(d, sd, axes, infl=infl, extended_open=extended_open, prior=prior, alpha=alpha, k=k,
                             k_max=k_max, e_lo=e_lo, planned=planned, cata=cata)
    return verdict_gs2(d, sd, axes, infl=infl, extended_open=extended_open, prior=prior, alpha=alpha, k=k,
                       k_max=k_max, planned=planned, df=df, cata=cata)


def verdict_gs2(d: Sequence[float], sd: float,
                axes: Mapping[str, tuple[Sequence[float], float, Sequence[float]]] | None = None, *,
                infl: Sequence[float] | None = None, extended_open: int = 0,
                prior: tuple[float, float] = (0.0, math.inf), alpha: float = ALPHA,
                k: int = K, k_max: int = K_MAX, planned: int | None = None, df: float | None = None,
                spending: str | None = None, futility: float | None = None,
                prior_df: float | None = None,
                cata: Mapping[str, tuple[Sequence[float], float]] | None = None) -> Verdict:
    """The two-look group-sequential test (module docstring). ``planned``: the seeds the block holds (done + in
    flight; default ``len(d)``).

    - OPEN: n < planned (running; the extension's seeds included), or n < k.
    - FAIL early: even ``d_s = +2σ_d`` on every seed up to ``k_max`` cannot reach c2 (non-binding: only lowers α).
    - Look 1 (n = planned = k < k_max): PASS at Z >= c1 (n >= ``MIN_PASS_N``, no harmed guard axis); a harmed axis
      FAILs; Z < ``futility`` FAILs (binding, priced into c2), as does a shrunk mean <= 0 when a ``prior`` is given;
      else EXTEND by ``EXTEND_BY`` (cap ``k_max``) -- or OPEN while ``MAX_EXTENDED`` other tests are extended.
    - Look 2 (n >= k_max, or every planned seed of an extended block in): PASS at Z >= c2 without harm, else FAIL.
    ``e``: the mixture e-value, reported only (never decides)."""
    n = len(d)
    planned = n if planned is None else max(n, planned)
    mean = sum(d) / n if n else 0.0
    e = e_value(d, sd, k, infl) if n else 1.0
    harm = tuple(sorted({*harmed(axes or {}), *catastrophic(cata or {})}))
    fut = FUTILITY_Z if futility is None else futility
    nu0 = SD_PRIOR_DF if prior_df is None else prior_df
    if nu0 > 0:                                   # the EB blend: σ_d from the pool AND the test's own pairs
        sd = blend_sd(d, sd, nu0, infl)
        c1, c2 = gs_bounds(alpha, min(1.0, k / k_max), nu0 + k_max - 1, spending or SPENDING, fut, nu0 + k - 1)
    else:
        c1, c2 = gs_bounds(alpha, min(1.0, k / k_max), df, spending or SPENDING, fut)
    sw, w, z = _z_stat(d, sd, infl) if n else (0.0, 0.0, 0.0)

    def out(kind: Kind, reason: str, more: int = 0) -> Verdict:
        return Verdict(kind, e, n, mean, reason, harm, more, sd, z)

    final = n >= k_max or (planned > k and n >= planned)
    if not final and n < k_max:
        rest = k_max - n
        if (sw + 2.0 * sd * rest) / (sd * math.sqrt(w + rest)) < c2:
            return out(Kind.FAIL, f"cannot reach c2={c2:.2f} even at +2 sd per remaining seed")
    if n < planned:
        return out(Kind.OPEN, "running" if planned <= k else "extended, running", planned - n)
    if n < k:
        return out(Kind.OPEN, "running", k - n)
    if final:
        if z >= c2 and n >= MIN_PASS_N and not harm:
            return out(Kind.PASS, f"look 2: Z={z:.2f} >= c2={c2:.2f}")
        return out(Kind.FAIL, f"harm on {', '.join(harm)}" if harm else f"look 2 (n={n}): Z={z:.2f} < c2={c2:.2f}")
    if z >= c1 and n >= MIN_PASS_N and not harm:
        return out(Kind.PASS, f"look 1: Z={z:.2f} >= c1={c1:.2f}")
    if harm:
        return out(Kind.FAIL, f"harm on {', '.join(harm)}")
    m0, t2 = prior
    if z < fut:
        return out(Kind.FAIL, f"futility: Z={z:.2f} < {fut:g}")
    if not math.isinf(t2):
        shrunk = (m0 / t2 + mean * n / sd ** 2) / (1 / t2 + n / sd ** 2)
        if shrunk <= 0:
            return out(Kind.FAIL, f"shrunk mean {shrunk:+.5f} <= 0")
    if extended_open >= MAX_EXTENDED:
        return out(Kind.OPEN, "promising; waiting for an extension slot", 0)
    return out(Kind.EXTEND, f"look 1: c1={c1:.2f} > Z={z:.2f} >= {fut:g}", min(EXTEND_BY, k_max - n))


def verdict_msprt(d: Sequence[float], sd: float,
                  axes: Mapping[str, tuple[Sequence[float], float, Sequence[float]]] | None = None, *,
                  infl: Sequence[float] | None = None, extended_open: int = 0,
                  prior: tuple[float, float] = (0.0, math.inf), alpha: float = ALPHA,
                  k: int = K, k_max: int = K_MAX, e_lo: float = E_LO, planned: int | None = None,
                  cata: Mapping[str, tuple[Sequence[float], float]] | None = None) -> Verdict:
    """The mixture SPRT (``test.method: msprt``) on the paired panel differences ``d`` seen so far (``axes`` for the harm check). ``planned``: the
    seeds the block holds (done + in flight; default ``len(d)``) -- an asynchronous block is never extended again
    while (or after) its extension runs.

    - PASS: E >= 1/α, n >= ``MIN_PASS_N``, no harmed guard axis (``axes``: only the guard axes are passed in).
    - FAIL: even ``d_s = +2σ_d`` on every seed up to ``k_max`` cannot reach 1/α; or n >= ``k_max``; or, every
      planned pair in: a harmed axis, E < ``e_lo``, a shrunk mean <= 0, or an already-extended block (planned > k)
      still without a pass.
    - EXTEND: n = planned = k (never a block already past k), ``e_lo <= E < 1/α``, shrunk mean > 0 and fewer than
      ``MAX_EXTENDED`` other tests extended (``extended_open``): add ``EXTEND_BY`` seeds (cap ``k_max``). With no slot
      free the block waits (OPEN).
    - OPEN: still running (n < planned, or n < k).
    ``prior`` (mean, variance) shrinks the mean for the EXTEND decision only (default: none)."""
    n = len(d)
    planned = n if planned is None else max(n, planned)
    mean = sum(d) / n if n else 0.0
    e = e_value(d, sd, k, infl) if n else 1.0
    harm = tuple(sorted({*harmed(axes or {}), *catastrophic(cata or {})}))

    def out(kind: Kind, reason: str, more: int = 0) -> Verdict:
        return Verdict(kind, e, n, mean, reason, harm, more, sd)

    if e >= 1 / alpha and n >= MIN_PASS_N and not harm:
        return out(Kind.PASS, f"E={e:.3g} >= {1 / alpha:g}")
    if n >= k_max:
        return out(Kind.FAIL, f"n={n} at the cap without a pass")
    if e_value([*d, *[2 * sd] * (k_max - n)], sd, k, [*(infl or [1.0] * n), *[1.0] * (k_max - n)]) < 1 / alpha:
        return out(Kind.FAIL, "cannot reach 1/alpha even at +2 sd per remaining seed")
    if n < planned:
        return out(Kind.OPEN, "running" if planned <= k else "extended, running", planned - n)
    if n < k:
        return out(Kind.OPEN, "running", k - n)
    if harm:
        return out(Kind.FAIL, f"harm on {', '.join(harm)}")
    if planned > k:
        return out(Kind.FAIL, f"extended to n={n} without a pass (E={e:.3g})")
    m0, t2 = prior
    shrunk = mean if math.isinf(t2) else (m0 / t2 + mean * n / sd ** 2) / (1 / t2 + n / sd ** 2)   # (clean-seed V)
    if e < e_lo or shrunk <= 0:
        return out(Kind.FAIL, f"E={e:.3g} < {e_lo:g}" if e < e_lo else f"shrunk mean {shrunk:+.5f} <= 0")
    if extended_open >= MAX_EXTENDED:
        return out(Kind.OPEN, "promising; waiting for an extension slot", 0)
    return out(Kind.EXTEND, f"promising (E={e:.3g}, shrunk {shrunk:+.5f})", min(EXTEND_BY, k_max - n))


# -- simulation -------------------------------------------------------------------------------------------------
def run_one(rng: random.Random, gain: float, sd: float, *, k: int = K, k_max: int = K_MAX,
            true_sd: float | None = None, df: float | None = None, method: str | None = None) -> Verdict:
    """One test with true paired gain ``gain``, looked at after every seed, at σ_d = ``sd`` (fitted on ``df``
    dfs; None = known) while the differences have sd ``true_sd`` (default ``sd``); an EXTEND adds its seeds (an
    extension slot is always free)."""
    d: list[float] = []
    planned = k
    while True:
        d.append(rng.gauss(gain, sd if true_sd is None else true_sd))
        v = verdict(d, sd, k=k, k_max=k_max, planned=planned, df=df, method=method)
        if v.kind in (Kind.PASS, Kind.FAIL):
            return v
        if v.kind is Kind.EXTEND:
            planned += v.more


def simulate(gain: float, sigma: float, rho: float, n_sims: int = 10_000, seed: int = 0,
             method: str | None = None) -> dict[str, float]:
    """Pass rate, mean seeds per arm and extension rate for a true panel gain at panel noise (σ, ρ), σ_d known."""
    rng, sd = random.Random(seed), sigma_d(sigma ** 2, rho)
    vs = [run_one(rng, gain, sd, method=method) for _ in range(n_sims)]
    return {"gain": gain, "method": method or METHOD, "rho": rho, "sigma_d": sd, "mde": 2 * sd / math.sqrt(K),
            "pass": sum(v.kind is Kind.PASS for v in vs) / n_sims,
            "mean_n": sum(v.n for v in vs) / n_sims, "extended": sum(v.n > K for v in vs) / n_sims}


def simulate_fit(gain: float, sd: float, df: float, misfit: float = 1.0, n_sims: int = 10_000, seed: int = 0,
                 q: float = SD_Q, method: str | None = None) -> dict[str, float]:
    """The test as the loop runs it: the pair's true σ_d is ``misfit × sd`` (``sd`` = the pooled σ_d of its class;
    misfit > 1: this pair is noisier than its class), the loop's estimate is ``sd·√(χ²_ν/ν)`` (ν = ``df``), used
    through :func:`sd_for` (msprt: :func:`sd_used` at ``q``; gs2: as fitted past ``UCB_DF_MAX`` dfs, its boundaries
    at ``df``). Pass rate (type-I at ``gain`` 0), mean seeds per arm, extension rate."""
    m = method or METHOD
    rng, nu = random.Random(seed), max(1, int(round(df))) if not math.isinf(df) else 0
    vs = []
    for _ in range(n_sims):
        est = sd * math.sqrt(sum(rng.gauss(0, 1) ** 2 for _ in range(nu)) / nu) if nu else sd
        used = sd_used(est, df, q) if m == "msprt" else sd_for(est, df, m)
        vs.append(run_one(rng, gain, used, true_sd=misfit * sd, df=None if math.isinf(df) else df, method=m))
    return {"gain": gain, "method": m, "sigma_d": sd, "df": df, "misfit": misfit, "mde": 2 * sd / math.sqrt(K),
            "pass": sum(v.kind is Kind.PASS for v in vs) / n_sims,
            "mean_n": sum(v.n for v in vs) / n_sims, "extended": sum(v.n > K for v in vs) / n_sims}


def test_entry_ok(m_sel: float, v: float, sd: float, *, k: int = K, frac: float = 0.5, p_min: float = 0.7
                  ) -> tuple[bool, str]:
    """Is a test worth its 2k runs? Yes when the selection-shrunk belief clears half the test's MDE
    (``m_sel >= frac·2σ_d/√k``) or P(δ > 0) >= ``p_min`` under N(m_sel, v). ``sd``: the σ_d the test would use."""
    mde = 2.0 * sd / math.sqrt(k)
    p = 0.5 * (1.0 + math.erf(m_sel / math.sqrt(2.0 * v))) if v > 0 else float(m_sel > 0)
    return m_sel >= frac * mde or p >= p_min, f"m_sel {m_sel:+.5f} vs MDE/2 {frac * mde:.5f}, P(d>0) {p:.2f}"
