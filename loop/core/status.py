"""loop.core.status — statuses are derived, never stored (docs/loop-core-v3.md §2 "Statuses").

Stored facts are ``nodes``, ``runs`` and ``seed_blocks`` (with each closed TEST block's FROZEN verdicts); everything
here is a pure function of them, so recomputing is idempotent and a refitted noise never re-judges a closed test.

- :func:`fresh_seeds`  per node, the seeds it ran on as a member of the TEST block owning them (never selected on)
- :func:`replay`       closed TEST blocks in decision order (:func:`order`). A block's verdicts are FROZEN when it
                       closes (tick ``_decide`` stores :func:`record`: the candidate vs the block's control on the
                       block's own seeds, with the σ_d / ρ / dfs / change class used); replay reads them back and
                       never re-judges, never adds later shared seeds. A PASS switches the incumbent when its control
                       is the then-incumbent; several blocks closing in the same tick that PASS against it: the best
                       (highest mean δ, then E) switches, the others are PASSED and :func:`retest_candidates`.
                       A closed block without frozen verdicts (historical blocks before :func:`freeze`) is judged the
                       legacy way: against the THEN-incumbent over every fresh seed both ran when they share >= 3,
                       else against its own control (PASSED, never a switch); :func:`freeze` stores that once.
- :func:`chain`        the incumbent chain; :func:`derive` a status per node; :func:`decide_open` the verdicts of the
                       still-open TEST blocks (tick closes / extends them).

Base epochs (docs §8): a candidate on a newer base than the then-incumbent is judged against the incumbent's PORT on
that base (``Ledger.port``); old-base and new-base runs never pair. After a base change the incumbent's port is the
INCUMBENT (its original PASSED) -- the port is the same change on the new base, re-measured by the re-baseline."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Sequence

from loop import project as P
from loop.core import test as T
from loop.core.data import AXES, PANEL, ROOT, TEST, Ledger
from loop.core.model import Noise, seed_means
from loop.core.seeds import SeedBlock, owner

MIN_SHARED = 3


class Status(str, Enum):
    QUEUED = "queued"
    SCREENING = "screening"
    TESTING = "testing"
    PASSED = "passed"
    FAILED = "failed"
    INCUMBENT = "incumbent"
    VOID = "void"
    CRASHED = "crashed"


@dataclass(frozen=True)
class Judged:
    node: str
    vs: str
    block: str
    ts: float
    verdict: T.Verdict
    switched: bool
    frozen: bool = False


@dataclass(frozen=True)
class PairFit:
    """The noise a pair is judged with: ``noise`` (σ², ρ, dfs, resumed factor) and its provenance."""
    noise: Noise
    near: bool = False          # the one-config-key ρ (else far)
    label: str = ""             # which fit ("near", "far", "history", ...)


def fresh_seeds(led: Ledger, blocks: Sequence[SeedBlock]) -> dict[str, set[int]]:
    """node -> seeds of its runs that, when submitted, belonged to a TEST block listing the node."""
    out: dict[str, set[int]] = {}
    for r in led.runs:
        b = owner(r.seed, blocks, r.submitted_ts + 1e-6)
        if b is not None and b.pool == TEST and r.node in b.node_ids:
            out.setdefault(r.node, set()).add(r.seed)
    return out


@dataclass(frozen=True)
class Pairs:
    """Same-seed differences c - i (seed order): the panel, each axis, and whether the pair is clean (neither run
    resumed: a resumed run is an observation but shares less of the seed effect)."""
    seeds: tuple[int, ...]
    panel: tuple[float, ...]
    axes: Mapping[str, tuple[float, ...]]
    clean: tuple[bool, ...]

    def infl(self, noise: Noise | float, target: str = PANEL) -> list[float]:
        """Per-seed variance factors: 1 for a clean pair, the fitted resumed factor (``Noise.resumed_factor``)
        otherwise -- or ``1/(1-ρ)`` when given a bare ρ (no calibration)."""
        f = noise.resumed_factor(target) if isinstance(noise, Noise) else 1.0 / (1.0 - noise)
        return [1.0 if ok else f for ok in self.clean]


def _at(led: Ledger, node: str, seeds: set[int], before: float | None) -> tuple[dict[int, dict[str, float]], set[int]]:
    """seed -> target means (non-resumed runs preferred), and the seeds only resumed runs cover."""
    obs = [o for o in led.obs if o.node == node and o.seed in seeds and (before is None or o.ts <= before)]
    clean = {o.seed for o in obs if not o.resumed}
    use = [o for o in obs if not o.resumed or o.seed not in clean]
    vals = {t: seed_means(use, t) for t in (PANEL, *AXES)}
    return {s: {t: vals[t][s] for t in vals} for s in vals[PANEL]}, {o.seed for o in use} - clean


def paired(led: Ledger, c: str, i: str, seeds: set[int], before: float | None = None) -> Pairs:
    (yc, rc), (yi, ri) = _at(led, c, seeds, before), _at(led, i, seeds, before)
    both = tuple(sorted(set(yc) & set(yi)))
    return Pairs(both, tuple(yc[s][PANEL] - yi[s][PANEL] for s in both),
                 {a: tuple(yc[s][a] - yi[s][a] for s in both) for a in AXES},
                 tuple(s not in rc and s not in ri for s in both))


def judge(led: Ledger, c: str, i: str, seeds: set[int], noise: Noise, before: float | None = None,
          extended_open: int = 0, guard: Sequence[str] | None = None, planned: int | None = None) -> T.Verdict:
    """:func:`loop.core.test.verdict` of c against i on ``seeds`` (official panel; harm on the ``guard`` axes).
    σ_d as ``test.sd_for`` uses it for the dfs it was fitted on (``Noise.df``): gs2 as fitted past
    ``test.UCB_DF_MAX`` dfs (its boundaries priced at those dfs), else at its upper confidence bound."""
    if guard is None:
        guard = tuple(P.GUARD_AXES)                   # call time: the installed project's guard (nanopath: rob)
    p = paired(led, c, i, seeds, before)
    df = noise.df.get(PANEL)
    sd = T.sd_for(noise.sd_d(PANEL), df)
    axes = {a: (p.axes[a], T.sd_for(noise.sd_d(a), noise.df.get(a)), p.infl(noise, a)) for a in guard}
    # no guard -> no tripwire: every axis under the catastrophic veto (test.CATA_SIGMAS x its paired sd; 0 = off)
    cata = {a: (p.axes[a], noise.sd_d(a)) for a in AXES if a in noise.sigma2 and a in noise.rho} \
        if not guard and T.CATA_SIGMAS > 0 else None
    return T.verdict(p.panel, sd, axes, infl=p.infl(noise, PANEL), extended_open=extended_open, planned=planned,
                     df=df or None, cata=cata)


# -- frozen verdicts ----------------------------------------------------------------------------------------------
def record(c: str, vs: str, v: T.Verdict, fit: PairFit | Noise | None = None, *, legacy: bool = False,
           ts: float | None = None) -> dict[str, Any]:
    """The frozen form of one judgement (stored in ``seed_blocks.verdicts``)."""
    nz = fit.noise if isinstance(fit, PairFit) else fit
    out: dict[str, Any] = {"node": c, "vs": vs, "kind": v.kind.value, "e": v.e, "n": v.n, "mean": v.mean,
                           "reason": v.reason, "harm": list(v.harm), "sd": v.sd, "legacy": legacy}
    if nz is not None:
        out.update(rho=nz.rho.get(PANEL), df=nz.df.get(PANEL), sigma2=nz.sigma2.get(PANEL),
                   resumed=nz.resumed_factor(PANEL) if PANEL in nz.rho else None)
    if isinstance(fit, PairFit):
        out.update(near=fit.near, fit=fit.label)
    if ts is not None:
        out["ts"] = ts
    return out


def verdict_of(rec: Mapping[str, Any]) -> T.Verdict:
    return T.Verdict(T.Kind(rec["kind"]), float(rec["e"]), int(rec["n"]), float(rec["mean"]), str(rec["reason"]),
                     tuple(rec.get("harm") or ()), 0, float(rec.get("sd") or 0.0))


def order(b: SeedBlock) -> float:
    """A closed block's place in the decision sequence: historical (``hist-``) blocks by draw (as the old loop
    decided them), live blocks by close (the tick that judged them)."""
    if b.id.startswith("hist-") or b.closed_ts is None:
        return b.drawn_ts
    return b.closed_ts


def no_code(led: Ledger, c: str) -> bool:
    """A FAILED GRAFT (notes ``port_failed``: ``cutover.port_node`` records the target's commit on it, it has no code of
    its own) can never become the incumbent: a frozen PASS onto one switches nothing (live 11-11: a leader re-root
    matched a node's Labless submission to the failed rescue port a node by commit and made it the incumbent;
    replay now skips that frozen verdict and the incumbent stays a node -- no DB edit)."""
    node = led.nodes.get(c)
    return node is not None and "port_failed" in f" {node.notes or ''} ".split()


def _here(led: Ledger, c: str, inc: str) -> str:
    node = led.nodes.get(c)
    return (led.port(inc, node.base) if node is not None else None) or inc


def _legacy(led: Ledger, b: SeedBlock, inc: str, fresh: Mapping[str, set[int]], noise: Noise) -> list[Judged]:
    """A closed block without frozen verdicts, judged as the old replay did (see the module docstring)."""
    out = []
    for c in b.node_ids:
        node = led.nodes.get(c)
        here = _here(led, c, inc)
        if c in (inc, here, ROOT) or (node is not None and node.parent is None):
            continue
        with_inc = paired(led, c, here, fresh.get(c, set()) & fresh.get(here, set()), b.closed_ts)
        vs = here if len(with_inc.seeds) >= MIN_SHARED else b.node_ids[-1]
        if vs == c:
            continue
        v = judge(led, c, vs, fresh.get(c, set()) & fresh.get(vs, set()), noise, b.closed_ts)
        if v.kind in (T.Kind.OPEN, T.Kind.EXTEND):         # the block closed undecided
            v = T.Verdict(T.Kind.FAIL, v.e, v.n, v.mean, f"closed undecided ({v.reason})", v.harm, 0, v.sd)
        switched = vs == here and v.kind is T.Kind.PASS and not no_code(led, c)
        out.append(Judged(c, vs, b.id, b.closed_ts or 0.0, v, switched))
        if switched:
            inc = c
    return out


def replay(led: Ledger, blocks: Sequence[SeedBlock], noise: Noise | None) -> list[Judged]:
    """Every judgement made by the closed TEST blocks, in decision order (see the module docstring)."""
    closed = sorted((b for b in blocks if b.pool == TEST and not b.open), key=lambda b: (order(b), b.id))
    fresh: dict[str, set[int]] | None = None
    inc, out, i = ROOT, [], 0
    while i < len(closed):
        b = closed[i]
        if b.verdicts is None:
            i += 1
            if noise is None:
                continue
            fresh = fresh if fresh is not None else fresh_seeds(led, blocks)
            for j in _legacy(led, b, inc, fresh, noise):
                out.append(j)
                inc = j.node if j.switched else inc
            continue
        group = [b]
        while i + len(group) < len(closed) and closed[i + len(group)].verdicts is not None \
                and order(closed[i + len(group)]) == order(b) and not b.id.startswith("hist-"):
            group.append(closed[i + len(group)])
        i += len(group)
        recs = [(blk, r) for blk in group for r in blk.verdicts or ()]
        if any(r.get("legacy") for _, r in recs):          # a frozen legacy block: its members in order
            for blk, r in recs:
                here = _here(led, r["node"], inc)
                sw = r["kind"] == "pass" and r["vs"] in (inc, here) and r["node"] not in (inc, here, ROOT) \
                    and not no_code(led, r["node"])
                out.append(Judged(r["node"], r["vs"], blk.id, blk.closed_ts or 0.0, verdict_of(r), sw, True))
                inc = r["node"] if sw else inc
            continue
        wins = [(blk, r) for blk, r in recs if r["kind"] == "pass" and r["vs"] in (inc, _here(led, r["node"], inc))
                and r["node"] not in (inc, _here(led, r["node"], inc), ROOT) and not no_code(led, r["node"])]
        best = max(wins, key=lambda br: (br[1].get("tier") != "local", br[1]["mean"], br[1]["e"], br[0].id)) \
            if wins else None     # a gs2 PASS outranks a local-tier one (test.local_reroot)
        for blk, r in recs:
            sw = best is not None and blk is best[0] and r is best[1]
            out.append(Judged(r["node"], r["vs"], blk.id, blk.closed_ts or 0.0, verdict_of(r), sw, True))
        if best is not None:
            inc = best[1]["node"]
    return out


def freeze(led: Ledger, blocks: Sequence[SeedBlock], noise: Noise | None) -> dict[str, tuple[dict[str, Any], ...]]:
    """block id -> the frozen verdicts of every closed TEST block that has none yet (historical blocks, or one closed
    outside ``_decide``): its legacy judgements under ``noise``, marked ``legacy``; () for a block that judged
    nothing (a void draw, a root-only audit). Written once by the tick; replay then never re-judges them."""
    todo = {b.id for b in blocks if b.pool == TEST and not b.open and b.verdicts is None}
    if not todo or noise is None:
        return {}
    out: dict[str, list[dict[str, Any]]] = {bid: [] for bid in todo}
    for j in replay(led, blocks, noise):
        if j.block in todo:
            out[j.block].append(record(j.node, j.vs, j.verdict, noise, legacy=True))
    return {bid: tuple(v) for bid, v in out.items()}


def chain(led: Ledger, blocks: Sequence[SeedBlock], noise: Noise | None) -> list[str]:
    """The incumbent chain, root first (switches only; ``Ledger.port`` maps its last onto the current base)."""
    return [ROOT, *(j.node for j in replay(led, blocks, noise) if j.switched)]


def retest_candidates(led: Ledger, blocks: Sequence[SeedBlock], noise: Noise | None,
                      judged: Sequence[Judged] | None = None) -> list[str]:
    """Nodes whose LAST judgement is a PASS that did not switch (its control was a former incumbent, or a better
    simultaneous pass won), that are not the incumbent or a former one, have no open TEST block, and are on the
    current base: re-testable against the current incumbent (acquire offers them a ``test``)."""
    judged = list(replay(led, blocks, noise)) if judged is None else list(judged)
    chained = next((j.node for j in reversed(judged) if j.switched), ROOT)
    inc = led.port(chained) or chained
    former = {j.vs for j in judged if j.switched} | {j.node for j in judged if j.switched}
    last = {j.node: j for j in judged}
    testing = {c for b in blocks if b.pool == TEST and b.open for c in b.node_ids[:-1]}
    out = []
    for nid, j in last.items():
        node = led.nodes.get(nid)
        if j.verdict.kind is T.Kind.PASS and not j.switched and j.vs != inc and nid not in former \
                and nid != inc and nid not in testing and node is not None and node.base == led.epoch:
            out.append(nid)
    return sorted(out)


def test_entry_ok(nid: str, post: Any, incumbent: str, *, near: bool = False, beta: float | None = None,
                  k: int | None = None, frac: float = 0.5, p_min: float = 0.7) -> tuple[bool, str]:
    """Is a test of ``nid`` against ``incumbent`` worth its runs (``test.test_entry_ok``): its composite belief
    (``post.belief`` moved to the incumbent by ``post.shift``; max-of-K shrunk when ``beta``/``k`` are given) against
    the σ_d the test will use -- the near fit only for a one-config-key candidate, else far (``post.fit(base, far)``),
    at its upper confidence bound. ``post``: an ``acquire.Posterior``."""
    from loop.core import model as M
    node = post.led.nodes[nid]
    m, v = post.belief(nid)
    dm, dv = post.shift(nid, M.COMPOSITE, incumbent)
    m, v = m + dm, v + dv
    if beta and k:
        m = M.max_of_k(m, v, k, beta)
    nz = post.fit(node.base, not near)[0]
    sd = T.sd_for(nz.sd_d(M.COMPOSITE), nz.df.get(M.COMPOSITE))
    return T.test_entry_ok(m, v, sd, frac=frac, p_min=p_min)


#: the seed-artifact gate (``loop.status``): the leave-one-seed-out check needs SEED_GATE_MIN paired seeds; from
#: SHARE_MIN_N paired seeds no single seed may be a one-seed OUTLIER of the others (v3 review K6: a studentised
#: test, z > OUTLIER_Z); from TRIM_MIN_N the trimmed mean must keep TRIM_FRAC of the mean. Set by ``acquire.configure``
#: from the config; read at call time. MAX_SEED_SHARE / SHARE_K: the replaced share cap (kept for old configs, unused)
MAX_SEED_SHARE = 0.4
SEED_GATE_MIN = 3
SHARE_MIN_N = 3
SHARE_K = 2.4
OUTLIER_Z = 1.8         # status.outlier_z: (d_s - mean of the others) / (sd_d sqrt(1 + 1/(n-1))) above this blocks
OUTLIER_SD_D = 0.0034   # the outlier test's sd_d when the caller has no fit (the live 10-13 pooled paired sd, 72 dfs)
TRIM_MIN_N = 4          # status.trim_min_n: from this many paired seeds the trimmed mean (max and min out) must
TRIM_FRAC = 0.25        # ... be >= status.trim_frac x the full mean


def seed_artifact(child: Iterable[Any], parent: Iterable[Any], weights: Mapping[str, float] | None, *,
                  target: str = "composite", max_share: float | None = None, min_seeds: int | None = None,
                  share_min_n: int | None = None, trim_min_n: int | None = None,
                  share_k: float | None = None, sd_d: float | None = None,
                  outlier_z: float | None = None) -> tuple[bool, str]:
    """(ok, why not): is the node's paired gain over its control (its parent: the shared control of its screen) more
    than one seed's artifact? Per shared seed d_s = y[c,s] - y[p,s]; ok iff >= ``min_seeds`` pairs, the
    leave-one-seed-out mean is > 0 for EVERY left-out seed, from ``trim_min_n`` pairs the TRIMMED mean (largest and
    smallest d_s out) keeps >= ``TRIM_FRAC`` of the mean, and -- from ``share_min_n`` pairs on -- no seed is a
    one-seed OUTLIER: t_s = (d_s - mean of the other seeds) / (sd_d sqrt(1 + 1/(n-1))) <= ``OUTLIER_Z`` (1.8) for
    every s (``sd_d``: the pooled same-seed paired sd of the composite, the posterior's fit; without one
    ``OUTLIER_SD_D``). v3 review K6: this replaces the n-scaled share cap max(.4, 2.4 / n), whose power fell
    with n. Simulated (d_s ~ N(mu, .0034), sd_d .0034): a true +.004 passes .81 / .82 / .81 at n 3 / 4 / 5 (the
    share cap .81 / .77 / .74), a one-seed +.010 artifact on a null .21 / .17 / .14; a node (10-13: seeds 100098-100
    d +.0006 / +.0007 / +.0086, the far-fit sd_d .00339: t 1.90) is blocked, its test failed. ``max_share`` /
    ``share_k``: the old cap's keys, accepted and ignored."""
    del max_share, share_k
    need = SEED_GATE_MIN if min_seeds is None else min_seeds
    share_n = SHARE_MIN_N if share_min_n is None else share_min_n
    c, p = seed_means(child, target, weights), seed_means(parent, target, weights)
    d = {s: c[s] - p[s] for s in c.keys() & p.keys()}
    if len(d) < max(2, need):
        return False, f"{len(d)} paired seeds < {max(2, need)}"
    tot = sum(d.values())
    loo = min((tot - x) / (len(d) - 1) for x in d.values())
    if loo <= 0:
        return False, f"leave-one-seed-out min {loo:+.5f} <= 0"
    trim_n = TRIM_MIN_N if trim_min_n is None else trim_min_n
    if trim_n > 0 and len(d) >= max(trim_n, 3):
        mid = sorted(d.values())[1:-1]
        if sum(mid) / len(mid) < TRIM_FRAC * tot / len(d):
            return False, (f"trimmed mean (max and min seed out) {sum(mid) / len(mid):+.5f} < {TRIM_FRAC:g} x the mean "
                           f"{tot / len(d):+.5f}")
    if len(d) >= max(2, share_n):
        z = OUTLIER_Z if outlier_z is None else float(outlier_z)
        n = len(d)
        for s, x in sorted(d.items(), key=lambda kv: (-kv[1], kv[0])):
            rest = [y for t, y in d.items() if t != s]
            sd = sd_d if sd_d is not None and math.isfinite(sd_d) and sd_d > 0 else OUTLIER_SD_D
            t = (x - sum(rest) / len(rest)) / (sd * math.sqrt(1 + 1 / (n - 1)))
            if t > z:
                return False, (f"seed {s} is a one-seed outlier: d {x:+.5f} vs the others' mean "
                               f"{sum(rest) / len(rest):+.5f} (t {t:.2f} > {z:g}, sd_d {sd:.5f}; it carries "
                               f"{x / tot:.0%} of the paired gain {tot / n:+.5f})")
            break                                   # the largest d_s is the only candidate outlier
    return True, ""


#: where a test's σ_d comes from (``test.sd_pool``): "pooled" = every clean paired child / comparator pair of the
#: current epoch (near and far together; live 10-13: sd_d .0034 on 72 dfs), else the same pool over every epoch;
#: "class" = the near / far change class (below). Set by ``acquire.configure``; read at call time
SD_POOL = "pooled"


def pair_fit(led: Ledger, c: str | None, drop_seeds: set[int], near: Iterable[str], near_ok: bool,
             fit: Callable[[Ledger, Any], Noise | None]) -> PairFit | None:
    """The noise a test of ``c`` is judged with (docs §4, fix A). ``c``'s own runs and the ``drop_seeds`` (its block)
    are left out of every fit. ``SD_POOL`` "pooled" (default): one fit over every clean pair of the current epoch
    (>= ``model.NEAR_MIN_DF`` dfs), else over every epoch. ``SD_POOL`` "class": the NEAR fit (one-config-key child / parent pairs of the current epoch:
    ``model.pairs_by_change``) only when ``near_ok`` (c is one config key from its comparator, config only) and only
    with >= ``model.NEAR_MIN_DF`` paired dfs; else the FAR fit (every other change + two-step pairs) of the current
    epoch with as many dfs; else the far fit over every epoch (the old base: lower ρ, conservative). The fit's dfs
    then inflate σ_d (``test.sd_used``). ``fit(ledger, pairs)``: e.g. ``backtest.fit_at(...)[0]``, None on failure."""
    from loop.core import model as M

    def ex(x: Ledger) -> Ledger:
        return replace(x, obs=tuple(o for o in x.obs if o.node != c and o.seed not in drop_seeds))

    cur, full = ex(led.in_epoch()), ex(led)
    np_, fp_ = M.pairs_by_change(led.nodes, near)
    tried, got = [], None
    order = ((("pooled", cur, [*np_, *fp_], False), ("pooled-all-epochs", full, [*np_, *fp_], False))
             if SD_POOL == "pooled" else
             (("near", cur, np_, True), ("far", cur, fp_, False), ("far-all-epochs", full, fp_, False)))
    for label, x, pairs, is_near in order:
        if is_near and not near_ok:
            continue
        nz = fit(x, pairs)
        df = nz.df.get(PANEL, 0.0) if nz is not None else 0.0
        tried.append(f"{label} df {df:g}")
        if nz is not None and (df >= M.NEAR_MIN_DF or label.endswith("all-epochs")):
            got = (nz, is_near, label)
            break
    if got is None:
        return None
    nz, is_near, label = got
    if PANEL not in nz.resume:      # resumed pairs too few in the class: the ρ drop of a wider fit (model docstring)
        for x in (cur, full):
            pooled = fit(x, [*np_, *fp_])
            if pooled is not None and PANEL in pooled.resume:
                nz = M.transport_resume(nz, pooled)
                tried.append(f"resumed factor from the {'epoch' if x is cur else 'all-epochs'} pool")
                break
    return PairFit(nz, is_near, f"{label} ({'; '.join(tried)})")


NoiseFor = Callable[..., "PairFit | Noise"]


def _fit_for(noise_for: NoiseFor | None, noise: Noise, c: str, i: str, b: SeedBlock) -> PairFit:
    if noise_for is None:
        return PairFit(noise, False, "given")
    try:
        f = noise_for(c, i, b)
    except TypeError:                                   # an old two-argument noise_for
        f = noise_for(c, i)
    return f if isinstance(f, PairFit) else PairFit(f, False, "given")


def decide_open(led: Ledger, blocks: Sequence[SeedBlock], noise: Noise, incumbent: str,
                noise_for: NoiseFor | None = None, *, k_max: int = T.K_MAX
                ) -> list[tuple[SeedBlock, str, T.Verdict, PairFit]]:
    """(block, candidate, verdict, fit) for every open TEST block's candidates, judged against the block's control
    (its last member) on the block's own seeds only. ``extended_open`` counts the other blocks already past K seeds;
    ``planned`` = the block's seeds (in flight included), so a block is extended at most once and never past
    ``k_max``. ``noise_for(c, i, block)``: the pair's :class:`PairFit` (change class; the candidate's own runs left
    out of the fit), default ``noise``."""
    opened = [b for b in blocks if b.pool == TEST and b.open]
    extended = sum(len(b.seeds) > T.K for b in opened)
    out = []
    for b in sorted(opened, key=lambda b: (b.drawn_ts, b.id)):
        for c in b.node_ids[:-1]:
            i = b.node_ids[-1]
            fit = _fit_for(noise_for, noise, c, i, b)
            v = judge(led, c, i, set(b.seeds), fit.noise, extended_open=extended - (len(b.seeds) > T.K),
                      planned=len(b.seeds))
            if v.kind is T.Kind.EXTEND and (len(b.seeds) > T.K or len(b.seeds) >= k_max):
                v = replace(v, kind=T.Kind.OPEN, more=0, reason=f"{v.reason}; already extended")   # never twice
            out.append((b, c, v, fit))
    return out


@dataclass(frozen=True)
class LocalReroot:
    """The LOCAL-INCUMBENT tier's judgement of one TEST candidate (``test.local_reroot``; operator 11-09 "reroot on
    free wins"): ``ok`` = re-root on it. ``p`` / ``gain`` / ``sd``: its EB-shrunk paired posterior over the control on
    EVERY paired seed of the epoch (``core.bar.paired``; ``n_pair`` seeds), ``n_fresh`` / ``fresh_mean``: the block's
    own fresh TEST pairs."""
    ok: bool
    p: float
    gain: float
    sd: float
    n_pair: int
    n_fresh: int
    fresh_mean: float
    why: str


def local_reroot(led: Ledger, c: str, i: str, seeds: Iterable[int], *, p_min: float = 0.95, min_fresh: int = 4,
                 min_gain: float = 0.002, fresh_z: float = 1.0, sd_fresh: float | None = None,
                 eb_min_nodes: int = 5) -> LocalReroot:
    """Is candidate ``c`` a FREE WIN over the control ``i`` (the incumbent) at its TEST block's look?

    ok iff ALL of:
    (1) FRESH evidence on its own: >= ``min_fresh`` fresh TEST pairs on the block's ``seeds`` whose paired mean is
        >= ``min_gain`` and whose one-sided z (mean / (``sd_fresh`` / sqrt n); ``sd_fresh`` = the σ_d the gs2 test used,
        else the epoch's pooled paired sd) is >= ``fresh_z`` -- the unselected seeds must carry the gain, not just
        agree in sign (review 11-10: sign-only passed ~30% of true-zero candidates selected on hot screens);
    (2) an EB prior exists (>= ``eb_min_nodes`` paired nodes; a flat prior would take a best-of-many screen at face
        value) and the candidate's GAUSSIAN posterior gain (its raw same-seed differences with ``i`` on EVERY seed both
        ran, screens + fresh, shrunk toward the epoch's median gain with the robust Paule-Mandel τ² of
        ``core.bar.paired``) is >= ``min_gain`` with P(gain > 0) >= ``p_min`` (Gaussian, not bar's Student-t: a far
        screen mean must not escape shrinkage);
    (3) no seed artifact over those differences (:func:`seed_artifact`: leave-one-seed-out, trimmed mean, outlier)."""
    from loop.core import bar as BR
    from loop.core import model as M
    fresh = paired(led, c, i, set(seeds)).panel
    fm = sum(fresh) / len(fresh) if fresh else math.nan

    def no(why: str, *rest: Any) -> LocalReroot:
        return LocalReroot(False, *(rest or (math.nan, math.nan, math.nan, 0)), len(fresh), fm, why)
    if c == i:
        return no(f"{c} is the control")
    pv = BR.paired(led, i, eb_min_nodes=eb_min_nodes)
    if pv is None or c not in pv.gains:
        return no(f"{c} does not pair with {i}")
    if pv.prior is None:
        return no(f"no EB prior (< {eb_min_nodes} paired nodes): a flat prior is never a re-root")
    cur = led.in_epoch().by_node()
    yc, yi = seed_means(cur.get(c, ()), PANEL), seed_means(cur.get(i, ()), PANEL)
    d = [yc[s] - yi[s] for s in sorted(yc.keys() & yi.keys())]
    if len(d) < 2:
        return no(f"{len(d)} same-seed pairs with {i}")
    k, raw = len(d), sum(d) / len(d)
    own = sum((x - raw) ** 2 for x in d) / (k - 1)
    v = (BR.PAIR_PRIOR_DF * pv.sd_d ** 2 + (k - 1) * own) / (BR.PAIR_PRIOR_DF + k - 1) / k
    m, pvar = M.posterior(pv.prior[0], pv.prior[1], M.Estimate(raw, v, k, k), nu=None)
    sd = math.sqrt(max(pvar, 0.0))
    p = (1.0 if m > 0 else 0.0) if sd <= 0 else 0.5 * (1.0 + math.erf(m / (sd * math.sqrt(2.0))))
    sdf = sd_fresh if sd_fresh is not None and math.isfinite(sd_fresh) and sd_fresh > 0 else pv.sd_d
    z = fm / (sdf / math.sqrt(len(fresh))) if fresh and sdf > 0 else math.nan
    why = []
    if len(fresh) < min_fresh:
        why.append(f"{len(fresh)} fresh TEST pairs < {min_fresh}")
    else:
        if not fm >= min_gain:
            why.append(f"fresh paired mean {fm:+.5f} < min_gain {min_gain:+.4f}")
        if not z >= fresh_z:
            why.append(f"fresh z {z:.2f} < {fresh_z:g}")
    if not m >= min_gain:
        why.append(f"shrunk gain {m:+.5f} < min_gain {min_gain:+.4f}")
    if not p >= p_min:
        why.append(f"P(gain > 0) {p:.3f} < {p_min:g}")
    art_ok, art = seed_artifact(cur.get(c, ()), cur.get(i, ()), None, target=PANEL, sd_d=pv.sd_d)
    if not art_ok:
        why.append(f"seed artifact: {art}")
    head = (f"shrunk paired gain {m:+.5f} ± {sd:.5f} on {k} paired seeds (raw {raw:+.5f}), P(gain > 0) {p:.3f}; "
            f"fresh TEST {len(fresh)} pairs mean {fm:+.5f} (z {z:.2f})")
    return LocalReroot(not why, p, m, sd, k, len(fresh), fm, head + ("" if not why else "; NO: " + "; ".join(why)))


def derive(led: Ledger, blocks: Sequence[SeedBlock], noise: Noise | None,
           void_statuses: frozenset[str] = frozenset({"rejected", "vetoed", "void"}),
           judged: Iterable[Judged] | None = None) -> dict[str, Status]:
    """A status per node; pure and idempotent."""
    judged = list(replay(led, blocks, noise)) if judged is None else list(judged)
    chained = next((j.node for j in reversed(judged) if j.switched), ROOT)
    inc = led.port(chained) or chained               # after a base change: its port on the current base
    last = {j.node: j for j in judged}
    testing = {c for b in blocks if b.pool == TEST and b.open for c in b.node_ids[:-1]}
    runs: dict[str, list[str]] = {}
    for r in led.runs:
        runs.setdefault(r.node, []).append(r.status)
    former = {j.vs for j in judged if j.switched} | ({chained} - {inc})
    out = {}
    for nid, node in led.nodes.items():
        st = runs.get(nid, [])
        if nid == inc:
            out[nid] = Status.INCUMBENT
        elif nid in testing:
            out[nid] = Status.TESTING
        elif nid in last:
            out[nid] = Status.PASSED if last[nid].verdict.kind is T.Kind.PASS else Status.FAILED
        elif nid in former:
            out[nid] = Status.PASSED
        elif not st:
            out[nid] = Status.VOID if node.status in void_statuses else Status.QUEUED
        elif "done" in st or any(s not in ("crash", "void") for s in st):
            out[nid] = Status.SCREENING
        else:
            out[nid] = Status.CRASHED if "crash" in st else Status.VOID
    return out


def summary(statuses: Mapping[str, Status]) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in statuses.values():
        out[s.value] = out.get(s.value, 0) + 1
    return out
