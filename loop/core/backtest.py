"""loop.core.backtest — time-ordered replay of the posterior against fresh-seed outcomes (docs/loop-core-v3.md §8).

Check (a): for every node that later ran on HELD-OUT stages (confirm / audit / remint: seeds it was never
selected on), compute its posterior from the data available strictly before its first held-out run was
submitted (noise, composite weights, prior and its own estimate all refit on that cut), then compare with the
fresh outcome (CUPED increment over the parent on the held-out runs): Spearman rho against the fresh PANEL,
calibration slope (OLS of fresh on m) and 80% predictive coverage against the fresh value of the posterior's own
target. Gate: rho >= .43 and slope in [.7, 1.3], on the screening (composite) posterior; the panel posterior
is reported next to it.
Check (b): Spearman of the RAW selection-data estimate on the composite vs on the panel against the fresh panel.

Check (c): :func:`check_c` -- the fresh-seed test replayed on the audited nodes' confirm / audit blocks; (d):
:func:`check_d` -- the derived incumbent chain vs the recorded one; plus the test's type-I / power simulations.

``uv run python -m loop backtest --db <copy> [--task-id ID] [--val-loss] [--decide]``"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, replace
from typing import Mapping, Sequence

from loop.core import model as M
from loop.core import status as ST
from loop.core import test as T
from loop.core.data import HELD_OUT_STAGES, PANEL, ROOT, Ledger, Obs, load, load_checked
from loop.core.seeds import SeedBlock, reconstruct

GATE_RHO, GATE_SLOPE = 0.43, (0.7, 1.3)
Z80 = 1.2816


@dataclass(frozen=True)
class Point:
    node: str
    ts: float
    lane: str
    family: str
    carrier: str
    n_sel: int
    n_fresh: int
    stage: str           # the held-out stage that produced the fresh outcome (first of confirm/audit/remint)
    raw: float           # panel CUPED increment on selection data (no prior)
    raw_var: float       # its modelled sampling variance V
    raw_c: float         # same on the screening composite
    m: float             # panel posterior mean
    v: float
    m_c: float           # composite posterior mean
    v_c: float
    prior: float
    depth: int
    fresh: float         # panel CUPED increment on the held-out runs
    fresh_var: float
    fresh_c: float       # composite CUPED increment on the held-out runs
    fresh_c_var: float


# -- statistics -------------------------------------------------------------------------------------------------
def ranks(xs: Sequence[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx, syy = sum((a - mx) ** 2 for a in x), sum((b - my) ** 2 for b in y)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else 0.0


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    return pearson(ranks(x), ranks(y))


def slope(x: Sequence[float], y: Sequence[float]) -> float:
    """OLS slope of y on x (calibration: 1 = the posterior's spread is right, < 1 = over-confident)."""
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sxx = sum((a - mx) ** 2 for a in x)
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / sxx if sxx > 0 else 0.0


# -- replay -----------------------------------------------------------------------------------------------------
def _parents(led: Ledger) -> dict[str, str | None]:
    return {k: n.parent for k, n in led.nodes.items()}


def fit_at(cut: Ledger, parents: Mapping[str, str | None]) -> tuple[M.Noise, dict[str, float]]:
    """Noise (axes, panel, composite) and composite weights from the data in ``cut``."""
    noise = M.fit_noise(cut.obs, parents=parents)
    w = M.composite_weights(cut.obs, noise)
    nc = M.fit_noise(cut.obs, [M.COMPOSITE], w, parents)
    # the root-seed floor (loop.project.noise_floor_seeds; default 5 since 11-10) AFTER the composite weights are fitted
    return M.floor_noise(M.merge(noise, nc), M.root_seeds(cut)), w


def prior_at(cut: Ledger, by_node: Mapping[str, Sequence[Obs]], noise: M.Noise, w: Mapping[str, float],
             target: str, exclude: str, spread: str = M.SPREAD) -> M.Prior:
    pts = []
    for nid, obs in by_node.items():
        node = cut.nodes.get(nid)
        if nid in (ROOT, exclude) or node is None or node.parent not in by_node:
            continue
        est = M.cuped(obs, by_node[node.parent], target, noise, w)
        if est is not None:
            pts.append(M.PriorPoint(M.cell_of(node), est.delta, est.var,
                                    val_z(obs, by_node[node.parent])))
    return M.fit_prior(pts, spread=spread)


def val_z(child: Sequence[Obs], parent: Sequence[Obs]) -> float | None:
    """(child - parent) mean val_loss_q25 (None if either side lacks it)."""
    c = [o.val_loss_q25 for o in child if o.val_loss_q25 is not None]
    p = [o.val_loss_q25 for o in parent if o.val_loss_q25 is not None]
    return sum(c) / len(c) - sum(p) / len(p) if c and p else None


def replay(led: Ledger, *, refit_hours: float = 6.0, nu: float | None = M.NU, spread: str = M.SPREAD) -> list[Point]:
    """One :class:`Point` per node with held-out runs, in time order (see the module docstring)."""
    parents = _parents(led)
    full = led.by_node()
    first: dict[str, tuple[float, str]] = {}
    for r in led.runs:
        if r.stage in HELD_OUT_STAGES and r.node != ROOT and (r.node not in first or r.submitted_ts < first[r.node][0]):
            first[r.node] = (r.submitted_ts, r.stage)
    fits: dict[int, tuple[M.Noise, dict[str, float]]] = {}
    out = []
    for nid, (ts, stage) in sorted(first.items(), key=lambda kv: kv[1][0]):
        node = led.nodes.get(nid)
        if node is None or node.parent is None:
            continue
        cut = led.before(ts)
        by_node = cut.by_node()
        sel, par = by_node.get(nid, []), by_node.get(node.parent, [])
        fresh_obs = [o for o in full.get(nid, []) if o.stage in HELD_OUT_STAGES]
        if not sel or not par or not fresh_obs:
            continue
        bucket = int(ts // (refit_hours * 3600))
        if bucket not in fits:
            fits[bucket] = fit_at(cut, parents)
        noise, w = fits[bucket]
        est, est_c = M.cuped(sel, par, PANEL, noise), M.cuped(sel, par, M.COMPOSITE, noise, w)
        z = val_z(sel, par)
        pm, tau2, depth = prior_at(cut, by_node, noise, w, PANEL, nid, spread).cell(M.cell_of(node), z)
        pc, tau2c, _ = prior_at(cut, by_node, noise, w, M.COMPOSITE, nid, spread).cell(M.cell_of(node), z)
        m, v = M.posterior(pm, tau2, est, nu)
        m_c, v_c = M.posterior(pc, tau2c, est_c, nu)
        fresh = M.cuped(fresh_obs, full[node.parent], PANEL, noise)
        fresh_c = M.cuped(fresh_obs, full[node.parent], M.COMPOSITE, noise, w)
        out.append(Point(nid, ts, node.lane, node.family, node.carrier, est.n, fresh.n, stage, est.delta,
                         est.var, est_c.delta, m, v, m_c, v_c, pm, depth, fresh.delta, fresh.var,
                         fresh_c.delta, fresh_c.var))
    return out


def block(m: Sequence[float], v: Sequence[float], y: Sequence[float], y_var: Sequence[float],
          rank_y: Sequence[float] | None = None) -> dict[str, float]:
    """rho (of m against ``rank_y``, default y), calibration slope of y on m, 80% predictive coverage, bias."""
    n = len(m)
    if n < 3:
        return {"n": n}
    cover = sum(abs(b - a) <= Z80 * math.sqrt(va + vb) for a, va, b, vb in zip(m, v, y, y_var)) / n
    return {"n": n, "rho": spearman(m, rank_y if rank_y is not None else y), "slope": slope(m, y),
            "coverage80": cover, "bias": sum(b - a for a, b in zip(m, y)) / n}


def report(points: Sequence[Point]) -> dict[str, object]:
    """Check (a) twice -- the screening (composite) posterior and the panel posterior -- and check (b).
    (a) composite: rho of m_c against the fresh PANEL (the official outcome it must rank), slope and coverage
    against the fresh COMPOSITE (its own target, same scale). The gate is on the composite posterior."""
    if len(points) < 3:
        return {"n": len(points)}
    P = points
    f = [p.fresh for p in P]
    a_c = block([p.m_c for p in P], [p.v_c for p in P], [p.fresh_c for p in P], [p.fresh_c_var for p in P], f)
    a_p = block([p.m for p in P], [p.v for p in P], f, [p.fresh_var for p in P])
    for a in (a_c, a_p):
        a["gate"] = bool(a["rho"] >= GATE_RHO and GATE_SLOPE[0] <= a["slope"] <= GATE_SLOPE[1])
    b = {"rho_raw_panel": spearman([p.raw for p in P], f), "rho_raw_composite": spearman([p.raw_c for p in P], f),
         "rho_post_panel": a_p["rho"], "rho_post_composite": a_c["rho"],
         "rho_prior_only": spearman([p.prior for p in P], f)}
    by_carrier = {c: block([p.m_c for p in ps], [p.v_c for p in ps], [p.fresh_c for p in ps],
                           [p.fresh_c_var for p in ps], [p.fresh for p in ps])
                  for c in sorted({p.carrier for p in P}) if len(ps := [p for p in P if p.carrier == c]) >= 5}
    return {"a_composite": a_c, "a_panel": a_p, "b": b, "by_carrier": by_carrier}


# -- checks (c) and (d): the fresh-seed test and the derived chain ------------------------------------------------
def sequential(led: Ledger, c: str, i: str, seeds: set[int], noise: M.Noise, before: float | None = None) -> T.Verdict:
    """The test as it would have run: looked at after every paired seed (seed order), stopped at PASS / FAIL; the
    block planned at ``test.K`` seeds, grown by an EXTEND's seeds (gs2: its second look is at the extended n)."""
    seen: set[int] = set()
    v = T.Verdict(T.Kind.OPEN, 1.0, 0, 0.0, "no paired seed")
    planned = T.K
    for s in ST.paired(led, c, i, seeds, before).seeds:
        seen.add(s)
        v = ST.judge(led, c, i, seen, noise, before, planned=max(planned, len(seen)))
        if v.kind in (T.Kind.PASS, T.Kind.FAIL):
            break
        if v.kind is T.Kind.EXTEND:
            planned = len(seen) + v.more
    return v


def epoch_noise(led: Ledger, ts: float, base: str, c: str | None = None) -> M.Noise | None:
    """The decision noise of a far-class test in ``base``'s epoch AS KNOWN at ``ts`` (``status.pair_fit`` on the
    ledger cut at ts, every node far: a historical change's class is unknown), ``c``'s own runs left out."""
    cut = replace(led.before(ts), bases=tuple(b for b in led.bases if b[2] == base) or led.bases)

    def fit(x: Ledger, pairs: object) -> M.Noise | None:
        try:
            return fit_at(x, pairs)[0]                                   # type: ignore[arg-type]
        except (ValueError, ZeroDivisionError):
            return None
    pf = ST.pair_fit(cut, c, set(), (), False, fit)
    return pf.noise if pf is not None else None


def check_c(led: Ledger, noise: M.Noise | None = None) -> dict[str, object]:
    """``verdict()`` replayed, for every audited node, on (1) its held-out blocks against its PARENT -- every
    confirm / audit / remint seed both ran by the audit verdict, looked at seed by seed -- and (2) its audit block
    against root (the old audit's own pairing). False pass = (1) PASS while the recorded audit failed.
    Each audit is judged with ITS OWN epoch's noise as known at the audit (:func:`epoch_noise`: the far class, its
    dfs inflating σ_d), never a later or another epoch's fit; ``noise`` only stands in where that cannot be fitted.
    (The old audit's alternative was the node's CLAIMED gain over root, so (2) passing an audit-failed node only
    says the node beats root, which a node / a node do: they are built on carriers far above root.)"""
    rows = []
    for ts, nid, truth in led.audits:
        if (node := led.nodes.get(nid)) is None:
            continue
        nz = epoch_noise(led, ts, node.base, nid) or noise
        if nz is None:
            continue
        held = {r.seed for r in led.runs if r.node == nid and r.stage in HELD_OUT_STAGES}
        audit = {r.seed for r in led.runs if r.node == nid and r.stage == "audit"}
        vp = sequential(led, nid, node.parent or ROOT, held, nz, ts)
        va = sequential(led, nid, ROOT, audit, nz, ts)
        rows.append({"node": nid, "parent": node.parent, "audit": truth, "vs_parent": vp.kind.value,
                     "e": round(vp.e, 3), "n": vp.n, "mean": round(vp.mean, 5), "reason": vp.reason,
                     "sd": round(vp.sd, 5), "df": nz.df.get(PANEL),
                     "vs_root_audit_block": va.kind.value, "e_root": round(va.e, 1), "n_root": va.n})
    passes = [r for r in rows if r["vs_parent"] == "pass"]
    false = [r["node"] for r in passes if r["audit"] != "passed"]
    return {"rows": rows, "passes": len(passes), "false_passes": false,
            "false_pass_rate": len(false) / len(passes) if passes else 0.0}


def check_d(led: Ledger, blocks: Sequence[SeedBlock], noise: M.Noise) -> dict[str, object]:
    """The derived chain (with each judgement that switched) next to the recorded ``incumbent`` events."""
    judged = ST.replay(led, blocks, noise)
    return {"derived": ST.chain(led, blocks, noise),
            "switches": [{"node": j.node, "previous": j.vs, "block": j.block, "ts": j.ts, "e": round(j.verdict.e, 2),
                          "n": j.verdict.n} for j in judged if j.switched],
            "recorded": [{"node": n, "previous": p, "ts": t} for t, n, p in led.incumbents],
            "statuses": ST.summary(ST.derive(led, blocks, noise))}


def simulations(n_null: int = 10_000, n_alt: int = 4_000, sigma: float = 0.0031) -> list[dict[str, float]]:
    """Type-I (10k null sims) and power at true panel gains .002/.003/.005, at rho = .38 (now) / .65 / .8 (CRN base),
    σ_d known; then as the loop runs it (``test.simulate_fit``, the configured ``test.method``): σ_d fitted on 20 /
    100 dfs and a pair 1.25x noisier than its class; and at the observed σ_d / dfs (:data:`OBSERVED_SD_D`)."""
    out = [T.simulate(g, sigma, rho, n_null if g == 0 else n_alt, seed=1)
           for rho in (0.38, 0.65, 0.8) for g in (0.0, 0.002, 0.003, 0.005)]
    return out + [T.simulate_fit(g, T.sigma_d(sigma ** 2, rho), df, mis, n_null if g == 0 else n_alt, seed=1)
                  for rho in (0.38, 0.8) for df in (20, 100) for mis in (1.0, 1.25) for g in (0.0, 0.002, 0.003)] \
        + [T.simulate_fit(g, OBSERVED_SD_D, OBSERVED_DF, mis, n_null if g == 0 else n_alt, seed=1)
           for mis in (1.0, 1.25) for g in (0.0, 0.003, 0.005, 0.009)]


#: the live pooled decision noise (10-13: clean same-epoch pairs, sd_d .0030-.0037 on ~71 dfs) the simulations report
OBSERVED_SD_D, OBSERVED_DF = 0.0033, 71.0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m loop backtest", description=__doc__.splitlines()[0])
    ap.add_argument("--db", required=True, help="a COPY of the ledger (opened read-only)")
    ap.add_argument("--task-id", default="nanopath-v2-rescore")
    ap.add_argument("--task", default=None, help="the task yaml: installs its loop.project (axes / probe keys) before the ledger is read")
    ap.add_argument("--val-loss", action="store_true", help="read metrics.jsonl for the val-loss covariate")
    ap.add_argument("--points", default=None, help="write the per-node points as JSONL here")
    ap.add_argument("--decide", action="store_true", help="checks (c) and (d) and the test simulations instead")
    args = ap.parse_args(argv)
    if args.decide:
        led = load_checked(args.db, args.task_id, args.task)
        noise, _ = fit_at(led, _parents(led))
        print(json.dumps({"c": check_c(led), "d": check_d(led, reconstruct(led), noise),
                          "simulations": simulations()}, indent=1))
        return 0
    pts = replay(load_checked(args.db, args.task_id, args.task, val_loss=args.val_loss))
    if args.points:
        with open(args.points, "w") as fh:
            fh.writelines(json.dumps(asdict(p)) + "\n" for p in pts)
    print(json.dumps(report(pts), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
