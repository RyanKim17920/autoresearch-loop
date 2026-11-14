"""loop.core.bar — the EXTERNAL bar: Labless's actual acceptance rule (operator 2026-10-13, superseding the 10-12
"mean beats a node's Labless average + .006, one-sided α" rule).

Labless's maintainer reruns a candidate on 3 fresh seeds; it is accepted iff the MEDIAN of those 3 beats the
leader's median (``labless.leader_median`` = .6810443, the median of a node's reproduction seeds .6807748 / .6810443 /
.6824622, record run_sub_6da523428f) by >= ``labless.promote_margin`` (.004): target median T = .68504 on their scale.

The loop's rule: a candidate beats the bar iff

    P(median of 3 fresh seeds >= T) >= ``labless.promote_p`` (.7)

estimated from the candidate's own seed distribution on our infra, with NO credit for a local-vs-Labless offset
(``labless.offset``, default 0, added to our mean: the key exists for later). One fresh seed's panel score is
Y ~ N(μ, σ²); for iid seeds P(median of 3 >= T) = 3q² − 2q³, q = Φ((μ − T)/σ). μ is uncertain: μ ~ N(m + offset,
σ²/n) with m the candidate's mean over its n seeds, so P = E_μ[3q² − 2q³] (numerical quadrature over μ).
EMPIRICAL-BAYES PRIOR (S1, 10-13): the flat prior on μ passed ~19% of candidates at true P = .5 on 4 seeds (a
best-of-many node's mean is winner's-cursed); μ now has core.model's Student-t prior (ν = ``model.NU``) located at the
median of the current epoch's node means, scale² = the robust Paule–Mandel τ² over the node means with V_i = σ²/n_i (:func:`tau2_robust`, ≥ ``model.TAU2_MIN``), when >=
``labless.eb_min_nodes`` (5; 0 = flat) nodes have a done run; μ ~ N(posterior mean + offset, posterior var). MC on the
10-13 ledger (σ .0033, prior .6779 / τ .0024, T .6850): pass rate at true P = .5, n = 4 .191 → .065; at true P = .7
.430 → .208; at true μ = μ*(P=.7) + .002 .852 → .658.
σ = the current epoch's pooled per-run panel σ (within-config, ``model``) blended with the candidate's own per-seed
sd: σ² = (ν₀ σ_pool² + (n − 1) s²) / (ν₀ + n − 1), ν₀ = ``labless.sigma_prior_df`` (8).

The needed local mean (:func:`need_mean`) is the m at which P = promote_p for a given σ and n (bisection); at
p = .5 it is T exactly (the median of a symmetric law). ``acquire.gap_of`` plans with n = ``labless.min_test_seeds``.

candidate = the mean over its seeds on the current epoch (test-block + screening runs, per-seed means) -- the
provisional P_screen; the VERDICT (``Beat.beats``, the meta "real improvement", the hb ``*``, the Labless gate) uses
only its FRESH TEST runs -- a TEST stage, confirm / remint / audit (:func:`fresh_test`; v3 review K8: not any run on a
TEST-pool seed, most of which were screens) -- and >= ``labless.min_test_seeds`` (4) of them (winner's curse guard); a
node whose last TEST verdict is FAIL ranks below every candidate not failed. Back-compat: without ``leader_median``
the old keys are read as leader = ``bar_score`` (or, without it,
``bar_node``'s own local mean) and margin = ``bar_margin``; ``bar_se`` / ``bar_alpha`` are unused. Pure; stdlib only."""

from __future__ import annotations

import dataclasses
import logging
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from loop.core import model as M
from loop.core.data import HELD_OUT_STAGES, PANEL, Ledger, Obs

PROMOTE_P = 0.7         # labless.promote_p
MIN_TEST_SEEDS = 4      # labless.min_test_seeds: the verdict needs at least this many fresh TEST-pool seeds
SIGMA_PRIOR_DF = 8.0    # labless.sigma_prior_df: pseudo-df of the pooled σ in the blend with the candidate's own sd
K_MEDIAN = 3            # the maintainer's fresh seeds (the median of K_MEDIAN)
_GRID = 81              # quadrature points over μ (±6 sd)
EB_MIN_NODES = 5        # labless.eb_min_nodes: nodes needed for μ's empirical-Bayes prior (0 = the flat prior)
_LOG = logging.getLogger(__name__)
_SIGMA_WARNED: set[str] = set()     # epochs whose σ fell back (logged once each)


@dataclass(frozen=True)
class Level:
    mean: float
    se: float
    n: int              # distinct seeds
    sd: float = math.nan    # per-seed sd (nan below 2 seeds)


@dataclass(frozen=True)
class Bar:
    node: str
    leader: float       # the leader's median on Labless's scale (or, back-compat, the bar node's local mean)
    margin: float = 0.0     # the required lead over it (labless.promote_margin)
    lab: bool = True        # leader is Labless's number (not our own runs of the bar node)
    p: float = PROMOTE_P    # the required P(median of 3 >= target)
    offset: float = 0.0     # Labless − local level offset added to our means (labless.offset; 0 = no credit)
    min_test: int = MIN_TEST_SEEDS
    prior_df: float = SIGMA_PRIOR_DF
    prior: tuple[float, float] | None = None    # μ's EB prior (location, scale²) on the local scale; None = flat

    @property
    def tag(self) -> str:
        return "lab" if self.lab else self.node

    @property
    def target(self) -> float:
        """The target median on Labless's scale."""
        return self.leader + self.margin

    @property
    def level(self) -> Level:       # back-compat view (se 0: the leader median is a fixed number)
        return Level(self.leader, 0.0, 0)


@dataclass(frozen=True)
class Beat:
    node: str
    n: int              # its seeds on the current epoch (screen + test): the PROVISIONAL, winner's-curse-prone view
    mean: float
    d: float            # mean − the target median (local scale, no offset credit unless labless.offset)
    sd: float           # the σ used (pooled blended with its own)
    p: float            # P(promote) on all its seeds (P_screen, provisional)
    beats: bool         # the VERDICT: P_test >= promote_p on >= min_test fresh TEST-pool seeds (never the screen P)
    n_test: int = 0     # its seeds on the never-used TEST pool (the sequential test's block)
    p_test: float | None = None     # P(promote) on those seeds alone; None below min_test
    d_test: float | None = None
    need: float = math.nan          # the local mean it needs for P = promote_p (its σ, n = max(n_test, min_test))
    mean_test: float | None = None
    #: the PAIRED POSTERIOR (v3-c7a A1, what :func:`ranked` / the heartbeat / the meta "best" rank on): the incumbent's
    #: level + the node's own paired gain over it (same-seed differences, EB-shrunk), its sd, and P(promote) on it
    post: float = math.nan
    post_sd: float = math.nan
    gain: float = math.nan          # the shrunk paired gain over the incumbent
    n_pair: int = 0                 # seeds paired with the incumbent (directly or through its parent chain)
    p_post: float = math.nan
    inc: str = ""


def level(obs: Iterable[Obs]) -> Level | None:
    ys = list(M.seed_means(obs, PANEL).values())
    if not ys:
        return None
    m = sum(ys) / len(ys)
    if len(ys) < 2:
        return Level(m, math.inf, 1)
    sd = math.sqrt(sum((y - m) ** 2 for y in ys) / (len(ys) - 1))
    return Level(m, sd / math.sqrt(len(ys)), len(ys), sd)


# ------------------------------------------------------------------------------------------------- the rule's math
def p_median(mu: float, sigma: float, target: float, k: int = K_MEDIAN) -> float:
    """P(median of k iid N(mu, sigma²) >= target), k odd: P(at least (k+1)/2 of k exceed it)."""
    if sigma <= 0:
        return float(mu >= target)
    q = 1.0 - M._Phi((target - mu) / sigma)
    need = k // 2 + 1
    return sum(math.comb(k, j) * q ** j * (1.0 - q) ** (k - j) for j in range(need, k + 1))


def p_promote(mean: float, sigma: float, n: int, target: float, offset: float = 0.0, k: int = K_MEDIAN,
              prior: tuple[float, float] | None = None) -> float:
    """P(median of k fresh seeds >= target) with μ ~ N(mean + offset, σ²/n) (the uncertainty in the mean); with
    ``prior`` (location, scale², local scale) μ is first shrunk by core.model's Student-t posterior."""
    if n <= 0 or not math.isfinite(sigma):
        return math.nan
    s = sigma / math.sqrt(n)
    mu0 = mean + offset
    if prior is not None and s > 0:
        pm, pv = M.posterior(prior[0], prior[1], M.Estimate(mean, s * s, n, 0))
        mu0, s = pm + offset, math.sqrt(pv)
    return p_mu(mu0, s, sigma, target, k)


def p_mu(mu0: float, s: float, sigma: float, target: float, k: int = K_MEDIAN) -> float:
    """P(median of k fresh seeds >= target) with μ ~ N(mu0, s²) and one seed's score ~ N(μ, σ²) (quadrature)."""
    if not (math.isfinite(mu0) and math.isfinite(sigma)):
        return math.nan
    if not math.isfinite(s) or s <= 0:
        return p_median(mu0, sigma, target, k)
    tot = wsum = 0.0
    for i in range(_GRID):
        z = -6.0 + 12.0 * i / (_GRID - 1)
        w = math.exp(-0.5 * z * z)
        tot += w * p_median(mu0 + z * s, sigma, target, k)
        wsum += w
    return tot / wsum


def need_mean(sigma: float, n: int, target: float, p: float = PROMOTE_P, offset: float = 0.0,
              k: int = K_MEDIAN, prior: tuple[float, float] | None = None) -> float:
    """The local mean m with p_promote(m, σ, n, target, offset, prior) = p (P is increasing in m; bisection)."""
    if not (0.0 < p < 1.0) or not math.isfinite(sigma) or sigma <= 0 or n <= 0:
        return target - offset
    lo, hi = target - offset - 10 * sigma, target - offset + 10 * sigma
    if prior is not None:           # shrinkage toward a far prior can need more than 10 σ
        far = 4 * abs(target - offset - prior[0])
        lo, hi = lo - far, hi + far
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        if p_promote(mid, sigma, n, target, offset, k, prior) >= p:
            hi = mid
        else:
            lo = mid
    return hi


def blend_sigma(pooled: float, own: Level | None, prior_df: float = SIGMA_PRIOR_DF) -> float:
    """σ² = (ν₀ σ_pool² + (n − 1) s²) / (ν₀ + n − 1): the pooled per-run σ, pulled toward the candidate's own sd."""
    if own is None or own.n < 2 or not math.isfinite(own.sd):
        return pooled
    if not math.isfinite(pooled) or pooled <= 0:
        return own.sd
    df = own.n - 1
    return math.sqrt((prior_df * pooled ** 2 + df * own.sd ** 2) / (prior_df + df))


# ------------------------------------------------------------------------------------------------------- the bar
MED_CHI2_1 = M.MED_CHI2_1               # re-exported: the estimator lives in core.model (fit_prior uses it too)
tau2_robust = M.tau2_robust


def mu_prior(led: Ledger, min_nodes: int = EB_MIN_NODES, sigma: float | None = None) -> tuple[float, float] | None:
    """μ's empirical-Bayes prior (location, scale²) from the current epoch's node means (base root excluded), as
    core.model fits a cell: location = their median, scale² = :func:`tau2_robust` with V_i = σ²/n_i.
    None (the flat prior) below ``min_nodes`` nodes, ``min_nodes`` <= 0, or without a σ."""
    if min_nodes <= 0:
        return None
    lvs = [lv for nid, obs in led.in_epoch().by_node().items() if nid != led.base_root and (lv := level(obs))]
    if len(lvs) < max(min_nodes, 2):
        return None
    s = safe_sigma(led) if sigma is None else sigma
    if not math.isfinite(s) or s <= 0:
        return None
    ms = [lv.mean for lv in lvs]
    return M._median(ms), tau2_robust(ms, [s * s / lv.n for lv in lvs])


def bar_of(led: Ledger, node: str, score: float | None = None, se: float = 0.0, margin: float = 0.0, *,
           p: float = PROMOTE_P, offset: float = 0.0, min_test: int = MIN_TEST_SEEDS,
           prior_df: float = SIGMA_PRIOR_DF, eb_min_nodes: int = EB_MIN_NODES) -> Bar | None:
    """The bar: leader ``score`` (Labless's median) when given, else ``node``'s own local mean; None if it cannot be
    measured. ``margin``: the lead the median of 3 needs over it. ``se`` is ignored (the leader is a fixed number).
    ``eb_min_nodes``: μ's empirical-Bayes prior (:func:`mu_prior`; 0 = flat)."""
    del se
    kw = dict(p=float(p), offset=float(offset), min_test=int(min_test), prior_df=float(prior_df),
              prior=mu_prior(led, int(eb_min_nodes)))
    if score is not None:
        return Bar(node, float(score), float(margin), True, **kw)
    lv = level(led.by_node().get(node, ())) if node and led.nodes.get(node) is not None else None
    return Bar(node, lv.mean, float(margin), False, **kw) if lv is not None else None


def leader_margin(cfg: Any) -> tuple[float | None, float]:
    """(leader, margin) a ``LablessCfg`` names: ``leader_median`` + ``promote_margin``, else (back-compat)
    ``bar_score`` + ``bar_margin``."""
    lm = getattr(cfg, "leader_median", None)
    if lm is not None:
        return float(lm), float(getattr(cfg, "promote_margin", 0.0) or 0.0)
    bs = getattr(cfg, "bar_score", None)
    return (None if bs is None else float(bs)), float(getattr(cfg, "bar_margin", 0.0) or 0.0)


def configured(cfg: Any) -> bool:
    """A bar is configured: a leader median / (legacy) bar score, or a bar node."""
    return cfg is not None and (leader_margin(cfg)[0] is not None or bool(getattr(cfg, "bar_node", "") or ""))


def bar_from(led: Ledger, cfg: Any) -> Bar | None:
    """The bar a ``LablessCfg`` names."""
    leader, margin = leader_margin(cfg)
    return bar_of(led, getattr(cfg, "bar_node", "") or "", leader, 0.0, margin,
                  p=float(getattr(cfg, "promote_p", PROMOTE_P)), offset=float(getattr(cfg, "offset", 0.0) or 0.0),
                  min_test=int(getattr(cfg, "min_test_seeds", MIN_TEST_SEEDS)),
                  prior_df=float(getattr(cfg, "sigma_prior_df", SIGMA_PRIOR_DF)),
                  eb_min_nodes=int(getattr(cfg, "eb_min_nodes", EB_MIN_NODES)))


def epoch_sigma(led: Ledger) -> float:
    """The current epoch's per-run panel σ (pooled within-config over configs with >= 2 seeds)."""
    cur = led.in_epoch()
    return math.sqrt(M.floor_sigma2(PANEL, M.fit_noise_target(cur.by_node(), PANEL)[0], M.root_seeds(cur)))


def safe_sigma(led: Ledger) -> float:
    """:func:`epoch_sigma`, degraded instead of raising (S8): right after an epoch rebase no config of the new epoch has
    2 seeds yet (ValueError) -- the whole ledger's pooled σ (the previous epochs' within-config noise) stands in, else
    nan (P(promote) is then nan: nothing beats the bar). Logged once per epoch."""
    try:
        return epoch_sigma(led)
    except (ValueError, ZeroDivisionError) as exc:
        try:
            s, how = math.sqrt(M.fit_noise_target(led.by_node(), PANEL)[0]), "the whole ledger's pooled sigma"
        except (ValueError, ZeroDivisionError):
            s, how = math.nan, "nan (nothing beats the bar)"
        if led.epoch not in _SIGMA_WARNED:
            _SIGMA_WARNED.add(led.epoch)
            _LOG.warning("bar: epoch %s has no panel sigma yet (%s); using %s", led.epoch[:8], exc, how)
        return s


def plan_need(bar: Bar, sigma: float) -> float:
    """The local mean a candidate with the pooled σ needs on ``bar.min_test`` fresh seeds for P(promote) = bar.p."""
    return need_mean(sigma, bar.min_test, bar.target, bar.p, bar.offset, prior=bar.prior)


def need_of(b: Beat, bar: Bar) -> float:
    """``b.need``, computed when :func:`ranked` left it out: its σ on n = its TEST seeds (or ``min_test``)."""
    if math.isfinite(b.need):
        return b.need
    return need_mean(b.sd, b.n_test if b.p_test is not None else max(bar.min_test, b.n_test), bar.target, bar.p,
                     bar.offset, prior=bar.prior)


# ------------------------------------------------------------------------------------------- the paired posterior
PAIR_PRIOR_DF = 8.0     # pseudo-dfs of the pooled same-seed paired sd in the blend with a node's own
CHAIN_MAX = 3           # a node with no seed shared with the incumbent pairs through at most this many ancestors


@dataclass(frozen=True)
class Paired:
    """The current epoch's paired view (v3-c7a A1): every node's own paired gain over the incumbent ``inc`` --
    the mean of its same-seed panel differences with ``inc`` (else through its parent chain: d(node, parent) +
    the parent's gain) -- EB-shrunk toward the median gain (Student-t, model.NU; scale² = :func:`tau2_robust` of the gains and their V,
    >= TAU2_MIN) when >= ``eb_min_nodes`` nodes pair, flat below. ``gains``: node -> (shrunk m, its var, n paired)."""
    inc: str
    level: Level
    gains: Mapping[str, tuple[float, float, int]]
    sd_d: float
    prior: tuple[float, float] | None = None


def incumbent_of(led: Ledger) -> str:
    """The incumbent the ledger records for the current epoch: the latest ``incumbent`` event's node (its port on the
    epoch), else the base root (callers that know the derived incumbent pass it)."""
    for _, nid, _ in reversed(led.incumbents):
        if (p := led.port(nid)) is not None:
            return p
    return led.base_root


def _point(led: Ledger, nid: str) -> bool:
    """A factorial wave point (``core.factorial.is_point``): a design cell at ONE shared seed, never a candidate."""
    from loop.core.factorial import is_point       # lazy: factorial imports nothing of bar
    n = led.nodes.get(nid)
    return n is not None and is_point(n.notes)


def paired(led: Ledger, inc: str | None = None, *, eb_min_nodes: int = EB_MIN_NODES,
           sigma: float | None = None) -> Paired | None:
    """:class:`Paired` on the current epoch; None when the incumbent has no done run there (nothing pairs)."""
    by = led.in_epoch().by_node()
    inc = inc or incumbent_of(led)
    if (lv := level(by.get(inc, ()))) is None:
        return None
    sm = {n: M.seed_means(o, PANEL) for n, o in by.items()}
    ds = {n: {x: y[x] - sm[inc][x] for x in y.keys() & sm[inc].keys()} for n, y in sm.items() if n != inc}
    ss = df = 0.0
    for d in ds.values():
        if len(d) >= 2:
            v = list(d.values())
            m = sum(v) / len(v)
            ss, df = ss + sum((x - m) ** 2 for x in v), df + len(v) - 1
    if df > 0:
        sd_d = math.sqrt(ss / df)
    else:
        s = safe_sigma(led) if sigma is None else sigma
        sd_d = math.sqrt(2.0) * s if math.isfinite(s) else math.nan

    def var_of(d: Sequence[float]) -> float:
        n = len(d)
        own = sum((x - sum(d) / n) ** 2 for x in d) / (n - 1) if n >= 2 else None
        s2 = sd_d ** 2 if own is None else (PAIR_PRIOR_DF * sd_d ** 2 + (n - 1) * own) / (PAIR_PRIOR_DF + n - 1)
        return s2 / n

    raw: dict[str, tuple[float, float, int]] = {}

    def gain(nid: str, depth: int = 0) -> tuple[float, float, int] | None:
        if nid in raw:
            return raw[nid]
        d = list(ds.get(nid, {}).values())
        if d:
            raw[nid] = (sum(d) / len(d), var_of(d), len(d))
            return raw[nid]
        node = led.nodes.get(nid)
        par = node.parent if node is not None else None
        if depth >= CHAIN_MAX or par is None or par not in sm or nid not in sm:
            return None
        if par == inc:
            return None
        up = gain(par, depth + 1)
        dd = [sm[nid][x] - sm[par][x] for x in sm[nid].keys() & sm[par].keys()]
        if up is None or not dd:
            return None
        raw[nid] = (sum(dd) / len(dd) + up[0], var_of(dd) + up[1], min(len(dd), up[2]))
        return raw[nid]

    for nid in ds:
        if not _point(led, nid) and nid != led.base_root:
            gain(nid)
    cands = {n: g for n, g in raw.items() if not _point(led, n) and n != led.base_root and math.isfinite(g[1])}
    prior = None
    if eb_min_nodes > 0 and len(cands) >= max(eb_min_nodes, 2):
        ms = [g[0] for g in cands.values()]
        prior = (M._median(ms), tau2_robust(ms, [g[1] for g in cands.values()]))
    out: dict[str, tuple[float, float, int]] = {inc: (0.0, 0.0, lv.n)}
    for n, (m, v, k) in cands.items():
        if prior is not None and v > 0:
            pm, pv = M.posterior(prior[0], prior[1], M.Estimate(m, v, k, k))
            out[n] = (pm, pv, k)
        else:
            out[n] = (m, v, k)
    return Paired(inc, lv, out, sd_d, prior)


def _post_fields(pv: Paired | None, node: str, c: Level, bar: Bar, pooled: float) -> dict[str, Any]:
    """The :class:`Beat` paired-posterior fields of ``node`` (empty when it does not pair with the incumbent)."""
    if pv is None or node not in pv.gains:
        return {}
    m, v, k = pv.gains[node]
    se_i = pv.level.se if math.isfinite(pv.level.se) else (pooled if math.isfinite(pooled) else 0.0)
    post, sd = pv.level.mean + m, math.sqrt(max(se_i * se_i + v, 0.0))
    p = p_mu(post + bar.offset, sd, blend_sigma(pooled, c, bar.prior_df), bar.target)
    return dict(post=post, post_sd=sd, gain=m, n_pair=k, p_post=p, inc=pv.inc)


def fresh_test(o: Obs) -> bool:
    """A FRESH TEST observation (v3 review K8): a run of a TEST stage -- confirm / remint / audit (``HELD_OUT_STAGES``:
    the runs a node was never selected on) -- not any run whose seed is in the TEST pool (live 10-14: 517 of the
    epoch's 905 runs on TEST-pool seeds were SCREENS, selected on, and counted as fresh test seeds)."""
    return o.stage in HELD_OUT_STAGES


def last_fail(led: Ledger) -> frozenset[str]:
    """Nodes whose LAST recorded TEST verdict (``test_verdict`` events) is FAIL."""
    last: dict[str, str] = {}
    for _, nid, v in sorted(getattr(led, "verdicts", ()) or ()):
        if v in ("pass", "fail"):
            last[nid] = v
    return frozenset(n for n, v in last.items() if v == "fail")


def beat(led: Ledger, bar: Bar, node: str, *, alpha: float | None = None, sigma: float | None = None,
         with_need: bool = True, pv: Paired | None = None) -> Beat | None:
    """``node`` vs the bar (None: not on the current epoch, the base root itself, or no done run). ``p`` is the
    provisional P(promote) on all its seeds (a best-of-many screened node can look good on it: winner's curse);
    ``beats`` / ``p_test`` use ONLY the node's runs on fresh TEST-pool seeds and need >= ``bar.min_test`` of them.
    ``alpha`` is ignored (back-compat: the rule is ``bar.p``). ``pv``: the epoch's :func:`paired` view (the
    ``post`` / ``p_post`` fields; left nan without it)."""
    del alpha
    n = led.nodes.get(node)
    if n is None or n.base != led.epoch or node == led.base_root:
        return None
    obs = led.in_epoch().by_node().get(node, ())
    c = level(obs)
    if c is None:
        return None
    pooled = safe_sigma(led) if sigma is None else sigma
    ct = level(o for o in obs if fresh_test(o))
    if ct is None or ct.n < bar.min_test:
        s = blend_sigma(pooled, c, bar.prior_df)
        p = p_promote(c.mean, s, c.n, bar.target, bar.offset, prior=bar.prior)
        need = need_mean(s, max(bar.min_test, ct.n if ct else 0), bar.target, bar.p, bar.offset, prior=bar.prior) \
            if with_need \
            else math.nan
        return Beat(node, c.n, c.mean, c.mean - bar.target, s, p, False, ct.n if ct is not None else 0, need=need,
                    mean_test=ct.mean if ct is not None else None, **_post_fields(pv, node, c, bar, pooled))
    s = blend_sigma(pooled, ct, bar.prior_df)
    p = p_promote(c.mean, blend_sigma(pooled, c, bar.prior_df), c.n, bar.target, bar.offset, prior=bar.prior)
    pt = p_promote(ct.mean, s, ct.n, bar.target, bar.offset, prior=bar.prior)
    need = need_mean(s, ct.n, bar.target, bar.p, bar.offset, prior=bar.prior) if with_need else math.nan
    return Beat(node, c.n, c.mean, c.mean - bar.target, s, p, pt >= bar.p, ct.n, pt, ct.mean - bar.target, need,
                ct.mean, **_post_fields(pv, node, c, bar, pooled))


def ranked(led: Ledger, bar: Bar, *, alpha: float | None = None, nodes: Sequence[str] | None = None,
           incumbent: str | None = None, eb_min_nodes: int = EB_MIN_NODES) -> list[Beat]:
    """Every current-epoch CANDIDATE with a done run (or ``nodes``) vs the bar: verdicts first, then the highest
    P(promote) -- P_test on >= ``bar.min_test`` fresh TEST seeds, else P on the PAIRED POSTERIOR (v3-c7a A1: the
    incumbent's level + the node's own shrunk paired gain over it, :func:`paired`). Never a candidate: a factorial
    wave point (one shared seed per block: 10-13, w013.p03 "best" at .6858 was 4 identical replicates at seed 100138)
    or a node without a test verdict that shares no seed with the incumbent (its raw level is a seed offset)."""
    s = safe_sigma(led)
    pv = paired(led, incumbent, eb_min_nodes=eb_min_nodes, sigma=s)
    ids = nodes if nodes is not None else sorted({o.node for o in led.in_epoch().obs})
    out = [b for x in ids if not _point(led, x)
           and (b := beat(led, bar, x, sigma=s, with_need=False, pv=pv)) is not None   # need: lazily
           and (b.p_test is not None or b.n_pair > 0)]
    failed = last_fail(led)                  # v3 review K8: a node whose last TEST verdict is FAIL ranks below every
    return sorted(out, key=lambda b: (b.node in failed, not b.beats,             # candidate not failed (untested ones)
                                      -_num(b.p_test if b.p_test is not None else b.p_post),
                                      -_num(b.post if b.p_test is None else b.mean_test), b.node))


def _num(x: float | None) -> float:
    return x if x is not None and math.isfinite(x) else -1.0


# ------------------------------------------------------------------------------------------------------- display
def _f(x: float, fmt: str = ".4f") -> str:
    """``.6789`` / ``+.0106``: no leading zero."""
    return format(x, fmt).replace("0.", ".", 1)


def bar_text(bar: Bar, sigma: float | None = None) -> str:
    """``bar=Labless median≥.6850 (leader .6810+.004) need local mean ≥ .6862`` (``sigma``: the pooled σ the needed
    mean is planned with, n = min_test seeds); a bar node without a Labless number: ``bar=a node local .6789+.004``."""
    head = (f"bar=Labless median≥{_f(bar.target)} (leader {_f(bar.leader)}{_f(bar.margin, '+.3f')})" if bar.lab
            else f"bar={bar.node} local {_f(bar.leader)}{_f(bar.margin, '+.3f')}→{_f(bar.target)}")
    if bar.offset:
        head += f" offset {_f(bar.offset, '+.4f')}"
    if sigma is not None and math.isfinite(sigma):
        head += f" need local mean ≥ {_f(plan_need(bar, sigma))}"
    return head


def _pt(b: Beat) -> str:
    return _f(b.p_test, ".2f") if b.p_test is not None else "—"


def beat_text(b: Beat, bar: Bar, short: bool = False) -> str:
    """``short``: ``a node mean=.6831 need .6862 P(promote)=.41 test 6s*``; the long form is for the meta report."""
    b = b if math.isfinite(b.need) else dataclasses.replace(b, need=need_of(b, bar))
    if short:
        if b.p_test is not None:
            return (f"{b.node} mean={_f(b.mean_test)} need {_f(b.need)} P(promote)={_pt(b)} test {b.n_test}s"
                    + ("*" if b.beats else ""))
        if math.isfinite(b.post):
            return (f"{b.node} post={_f(b.post)} ({b.inc} {_f(b.gain, '+.4f')} on {b.n_pair} paired s, raw "
                    f"{_f(b.mean)}) need {_f(b.need)} P(promote)={_f(b.p_post, '.2f')} provisional "
                    f"({b.n}s, {b.n_test}/{bar.min_test} test)")
        return (f"{b.node} mean={_f(b.mean)} need {_f(b.need)} P(promote)={_f(b.p, '.2f')} provisional "
                f"({b.n}s, {b.n_test}/{bar.min_test} test)")
    m = b.mean_test if b.p_test is not None else b.mean
    left = b.need - m
    pp = (f"; paired posterior {_f(b.post)} ± {_f(b.post_sd)} (incumbent {b.inc} + its shrunk gain "
          f"{_f(b.gain, '+.4f')} on {b.n_pair} paired seeds), P(promote) on it {_f(b.p_post, '.2f')} (the ranking)"
          if math.isfinite(b.post) and b.p_test is None else "")
    return (f"{b.node} local mean {_f(m)} ({b.n_test if b.p_test is not None else b.n} "
            f"{'fresh TEST' if b.p_test is not None else 'screen+test'} seeds, σ {_f(b.sd)}), needs {_f(b.need)} for "
            f"P(median of 3 ≥ {_f(bar.target)}) ≥ {_f(bar.p, '.2f')}: "
            + (f"{_f(left, '.4f')} short" if left > 0 else "reached")
            + f"; P(promote) provisional (all {b.n} seeds) {_f(b.p, '.2f')}; on fresh TEST seeds ({b.n_test}, needs "
            f">= {bar.min_test}) {_pt(b)}" + pp + (": BEATS THE BAR" if b.beats else ""))


def summary(led: Ledger, cfg: Any, commits: Mapping[str, str] | None = None, incumbent: str | None = None) -> str:
    """For the heartbeat: the bar (target median, leader, the local mean needed), then the best current candidate
    (``*``: it beats the bar on fresh TEST seeds) by :func:`ranked` (the meta report's "best" is the same ranking);
    '' when no bar is configured. ``cfg``: the ``labless`` config; ``incumbent``: the derived one (pairing)."""
    if not configured(cfg):
        return ""
    bar = bar_from(led, cfg)
    if bar is None:
        return f"bar={cfg.bar_node} (not measurable)"
    s = safe_sigma(led)
    head = bar_text(bar, s)
    best = ranked(led, bar, incumbent=incumbent, eb_min_nodes=int(getattr(cfg, "eb_min_nodes", EB_MIN_NODES)))
    if not best:
        return head + "; best=none"
    c = (commits or {}).get(best[0].node, "")
    t = beat_text(best[0], bar, short=True)
    return head + "; best=" + (f"{best[0].node}@{c[:7]}" + t[len(best[0].node):] if c else t)
