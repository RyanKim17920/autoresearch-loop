"""The PREDICTIVE world model: one hierarchical additive regression of a node's OWN panel increment on what was
known when it was CREATED, refit every world fit and scored out of time on every evaluation.

Why (2026-10-06 audit of the v2 run, time-split forecasts of 1,229 scored nodes): the tag -> family tree prior, the
lever ridge and the EB generation prior forecast fresh outcomes with Spearman rho ~ -0.03; only the family means and
the root-vs-carrier split carried skill. One ridge on creation-time features reached rho 0.38 on first screens and
0.29 on fresh seeds. This module is that model, generalised:

* **Features** (all generic: read from the node record, never from named ideas): the node's generation (a child of
  root vs of a carrier), family, mechanism type (loop.novelty.node_type) and their crossings with the generation,
  parent id, strategy / operator, brief kind, web-research depth, structure (#stack legs, crossover, tree depth),
  carrier saturation (children the parent already had, and in the same parent x mechanism-type cell), the registered
  lever vector, and (when the diagnostic layer has them) the parent's checkpoint diagnostics (``DIAG_KEYS``); the
  INIT the node trains from (the ``init=`` note of it or its nearest ancestor, an init swap; "base" = the root's), so
  a lineage on another pretrained model is never pooled with the root's.
* **Model**: an additive mixed model -- per (generation, target) fixed effects plus one random-effect group per
  feature group, each with its own variance tau_g^2 (EM; a group nothing supports shrinks to 0 by itself) --
  fit by backfitting on (a) every node's first two SCREEN seeds (decided before any of its data: selection-free) and
  (b) its fresh CONFIRM seeds (weighted ``prior.wm_fresh_weight``; their own per-generation offset).
* **Screen -> fresh** (per generation, out-of-fold): ``fresh = c_g + b_g * pred + kappa_g * r * (screen - pred)``:
  how a forecast and a node's own screen deviation transfer to fresh seeds (live: slope 0.81 for root children,
  0.07 for carrier children). A node's prior is ``N(c_g + b_g pred, T_g + k^2 (S + v))``; with its screen seeds
  ``N(c_g + b_g pred + k (e - pred), T_g)``; its fresh seeds then update that as direct measurements.
* **Time split**: forward-chained folds over the newest half of the nodes; logged each evaluation (rho and the
  calibration slope on screens and on fresh seeds, per generation). A feature group whose incremental out-of-time
  rho is below ``prior.wm_min_rho`` drops out of the fit until the next evaluation.
* **Outputs**: per-node prior / posterior (the gates' P(gain>0), E[gain+]), a candidate ranking score
  (:meth:`Fit.candidate`), the ALLOCATION signal (expected P(fresh gain > MDE) per carrier x mechanism type:
  :meth:`Fit.allocation`) and a plateau forecast (expected best fresh gain over the next N nodes vs their GPU cost).

Pure Python (the loop has no numpy): the backfitting sweeps are O(rows x groups)."""

from __future__ import annotations

import bisect
import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from loop.contracts import PANEL, ROOT_NODE_ID, LeverEffect, Node, Stage
from loop.world.evalues import phi
from loop.world.mixture import Mixture, fit_grid, posterior as mix_posterior, prob_greater as mix_p

Gauss = tuple[float, float]
STATE_KEY = "world.wm"                  # {"n": nodes at the last evaluation, "kept": [...], "eval": {...}}
FORECAST_KEY = "world.wm.forecast"      # {"n", "mde", "gpu_h_per_node", "top": cells, "forecast": [...]}
GATE_KEY = "world.wm.gate"              # the last fit's gate priors (warm start): {cell: {"n", "comps"}}
ROOT_GEN, CARRIER_GEN = "root", "carrier"
#: categorical feature groups (name -> how a row's level is built from its raw features)
CAT_GROUPS: tuple[str, ...] = ("family", "gen_family", "mtype", "gen_mtype", "parent", "strategy", "brief", "web",
                               "init")
#: numeric feature groups
NUM_GROUPS: tuple[str, ...] = ("structure", "saturation", "levers", "diagnostics")
GROUPS = CAT_GROUPS + NUM_GROUPS
#: parent checkpoint diagnostics used as features when present (loop.diagnostics quantities)
DIAG_KEYS: tuple[str, ...] = ("lr_end_frac", "upd_ratio_early_late", "gn_late_over_clip", "drift_early", "drift_late",
                              "st_late")
TAU_MAX2, TAU_MIN2, FIXED2 = 0.05 ** 2, 1e-12, 1.0
SWEEPS = 40


# ---------------------------------------------------------------------------------------------- features
def generation(parent: str | None) -> str:
    return ROOT_GEN if (parent or ROOT_NODE_ID) == ROOT_NODE_ID else CARRIER_GEN


def _note(notes: str, key: str) -> str | None:
    m = re.search(rf"(?:^|\s){key}=(\S+)", notes or "")
    return m.group(1) if m else None


def web_level(notes: str) -> str:
    """'na' (no record), '0' (no web), else the research depth bucket from searches + fetches."""
    w = _note(notes, "web")
    if w is None:
        return "na"
    if w != "1":
        return "0"
    depth = sum(int(_note(notes, k) or 0) for k in ("searches", "fetches") if (_note(notes, k) or "0").isdigit())
    return "1:shallow" if depth <= 2 else "1:deep"


def n_legs(notes: str) -> int:
    legs = _note(notes, "stack_legs") or _note(notes, "xover") or ""
    return len([x for x in legs.split(",") if x])


@dataclass(frozen=True)
class Features:
    gen: str
    cats: Mapping[str, str]
    nums: Mapping[str, tuple[float, ...]]


class FeatureBook:
    """Creation-time features of any node (scored or a candidate) against the tree as it is now; a node's
    saturation counts only nodes created before it."""

    def __init__(self, ctx: Any, store: Any, nodes: Mapping[str, Node], mtype: Callable[[Node], str],
                 diag: Callable[[str], Mapping[str, float] | None] | None = None) -> None:
        self.ctx, self.nodes, self.mtype, self.diag = ctx, nodes, mtype, diag or (lambda _: None)
        self.vocab = tuple(ctx.cfg.levers.vocabulary)
        self.kids: dict[str, list[float]] = {}
        self.cell: dict[tuple[str, str], list[float]] = {}
        for n in nodes.values():
            if n.id == ROOT_NODE_ID:
                continue
            p = n.parent_id or ROOT_NODE_ID
            self.kids.setdefault(p, []).append(n.created_ts)
            self.cell.setdefault((p, self._mt(n)), []).append(n.created_ts)
        for v in (*self.kids.values(), *self.cell.values()):
            v.sort()
        self._cache: dict[str, Features] = {}

    def _mt(self, n: Node) -> str:
        try:
            return self.mtype(n) or "none"
        except Exception:           # noqa: BLE001 - a missing diff file is an unknown type, never a failed fit
            return "none"

    def init(self, nid: str) -> str:
        """The init ``nid`` trains from: the ``init=`` note of it or its nearest ancestor, else "base"."""
        seen: set[str] = set()
        while nid and nid in self.nodes and nid not in seen:
            seen.add(nid)
            if got := _note(self.nodes[nid].notes or "", "init"):
                return got
            nid = self.nodes[nid].parent_id or ""
        return "base"

    def depth(self, nid: str) -> int:
        d, seen = 0, set()
        while nid and nid != ROOT_NODE_ID and nid in self.nodes and nid not in seen:
            seen.add(nid)
            nid, d = self.nodes[nid].parent_id or ROOT_NODE_ID, d + 1
        return d

    def of(self, n: Node, *, parent: str | None = None, mtype: str | None = None,
           strategy: str | None = None) -> Features:
        """``n``'s features; ``parent`` / ``mtype`` / ``strategy`` ("strategy/operator") override them (a counterfactual:
        :meth:`Fit.allocation`, :meth:`Fit.value`)."""
        key = f"{n.id}|{parent}|{mtype}|{strategy}"
        if key in self._cache:
            return self._cache[key]
        p = parent if parent is not None else (n.parent_id or ROOT_NODE_ID)
        gen, mt, fam, notes = generation(p), mtype or self._mt(n), n.family or "none", n.notes or ""
        op = getattr(n.operator, "value", "-") if n.operator is not None else "-"
        cats = {"family": fam, "gen_family": f"{gen}/{fam}", "mtype": mt, "gen_mtype": f"{gen}/{mt}", "parent": p,
                "strategy": strategy or f"{getattr(n.strategy, 'value', n.strategy)}/{op}",
                "brief": _note(notes, "brief") or "none",
                "web": web_level(notes), "init": _note(notes, "init") or self.init(p)}
        t = n.created_ts
        before = lambda xs: bisect.bisect_left(xs, t) if t else len(xs)       # noqa: E731
        try:
            lev = tuple(float(x) for x in n.levers.vector(self.vocab))
        except ValueError:
            lev = tuple(float(n.levers.directions.get(v, 0)) for v in self.vocab)
        d = self.diag(p) or {}
        nums = {"structure": (float(n_legs(notes)), float(bool(n.second_parent_id or _note(notes, "xover"))),
                              float(self.depth(p) + 1)),
                "saturation": (math.log1p(before(self.kids.get(p, []))), math.log1p(before(self.cell.get((p, mt), [])))),
                "levers": lev,
                "diagnostics": tuple(float(d[k]) if isinstance(d.get(k), (int, float)) and math.isfinite(d[k]) else 0.0
                                     for k in DIAG_KEYS) + (float(bool(d)),)}
        out = Features(gen, cats, nums)
        self._cache[key] = out
        return out


# ---------------------------------------------------------------------------------------------- the model
@dataclass
class Obs:
    nid: str
    target: str               # "screen" | "fresh"
    y: float
    v: float                  # measurement variance
    w: float = 1.0            # weight multiplier


class Additive:
    """``y = base[gen/target] + sum_g beta_g(x) + u``, u ~ N(0, s2) plus measurement noise; each categorical group
    a random effect with variance tau_g^2, each numeric group standardized columns with a shared tau_g^2 (EM)."""

    def __init__(self, groups: Sequence[str]) -> None:
        self.groups = tuple(g for g in groups if g in GROUPS)
        self.base: dict[str, float] = {}
        self.cat: dict[str, dict[str, float]] = {g: {} for g in self.groups if g in CAT_GROUPS}
        self.catw: dict[str, dict[str, float]] = {g: {} for g in self.groups if g in CAT_GROUPS}
        self.num: dict[str, list[float]] = {}
        self.mu: dict[str, list[float]] = {}
        self.sd: dict[str, list[float]] = {}
        self.numse: dict[str, list[float]] = {}      # posterior variance of each numeric coefficient
        self.tau2: dict[str, float] = {g: 0.003 ** 2 for g in self.groups}
        self.s2 = 0.003 ** 2

    def _numx(self, g: str, f: Features) -> list[float]:
        xs = f.nums.get(g, ())
        return [(x - m) / s if s > 0 else 0.0 for x, m, s in zip(xs, self.mu[g], self.sd[g])]

    def fit(self, rows: Sequence[tuple[Features, Obs]], sweeps: int = SWEEPS) -> "Additive":
        if not rows:
            return self
        for g in self.groups:
            if g in NUM_GROUPS:
                cols = list(zip(*(f.nums.get(g, ()) for f, _ in rows))) if rows[0][0].nums.get(g) else []
                self.mu[g] = [sum(c) / len(c) for c in cols]
                self.sd[g] = [(sum((x - m) ** 2 for x in c) / len(c)) ** 0.5 for c, m in zip(cols, self.mu[g])]
                self.num[g] = [0.0] * len(cols)
        bkey = [f"{f.gen}/{o.target}" for f, o in rows]
        lev = {g: [f.cats.get(g, "none") for f, _ in rows] for g in self.cat}
        X = {g: [self._numx(g, f) for f, _ in rows] for g in self.num}
        y = winsorize([o.y for _, o in rows])
        pred = [0.0] * len(rows)
        for _ in range(sweeps):
            w = [o.w / (o.v + self.s2) for _, o in rows]
            # base (flat prior)
            acc: dict[str, list[float]] = {}
            for i, k in enumerate(bkey):
                a = acc.setdefault(k, [0.0, 0.0])
                a[0] += w[i] * (y[i] - pred[i] + self.base.get(k, 0.0))
                a[1] += w[i]
            for k, (s, ww) in acc.items():
                new = s / (ww + 1.0 / FIXED2)
                d = new - self.base.get(k, 0.0)
                self.base[k] = new
                for i, kk in enumerate(bkey):
                    if kk == k:
                        pred[i] += d
            for g, coef in self.cat.items():
                acc = {}
                for i, lv in enumerate(lev[g]):
                    a = acc.setdefault(lv, [0.0, 0.0])
                    a[0] += w[i] * (y[i] - pred[i] + coef.get(lv, 0.0))
                    a[1] += w[i]
                t2 = self.tau2[g]
                delta = {}
                for lv, (s, ww) in acc.items():
                    new = s / (ww + 1.0 / t2)
                    delta[lv] = new - coef.get(lv, 0.0)
                    coef[lv] = new
                    self.catw[g][lv] = ww
                for i, lv in enumerate(lev[g]):
                    pred[i] += delta[lv]
                L = len(coef)
                self.tau2[g] = min(TAU_MAX2, max(TAU_MIN2, sum(b * b + 1.0 / (self.catw[g][lv] + 1.0 / t2)
                                                                for lv, b in coef.items()) / max(1, L)))
            for g, beta in self.num.items():
                t2 = self.tau2[g]
                post = 0.0
                se2 = self.numse[g] = [0.0] * len(beta)
                for j in range(len(beta)):
                    s = ww = 0.0
                    for i, x in enumerate(X[g]):
                        if x[j]:
                            s += w[i] * x[j] * (y[i] - pred[i] + beta[j] * x[j])
                            ww += w[i] * x[j] * x[j]
                    new = s / (ww + 1.0 / t2) if ww > 0 else 0.0
                    d = new - beta[j]
                    beta[j] = new
                    se2[j] = 1.0 / (ww + 1.0 / t2)
                    post += se2[j]
                    if d:
                        for i, x in enumerate(X[g]):
                            pred[i] += d * x[j]
                if beta:
                    self.tau2[g] = min(TAU_MAX2, max(TAU_MIN2, (sum(b * b for b in beta) + post) / len(beta)))
            scr = [(y[i] - pred[i]) ** 2 - o.v for i, (_, o) in enumerate(rows) if o.target == "screen"]
            if scr:
                self.s2 = max(1e-10, sum(scr) / len(scr))
        return self

    def predict(self, f: Features, target: str = "screen") -> Gauss:
        """(mean, parameter variance) of a node's screen-scale increment; an unseen level adds its group's tau^2."""
        m = self.base.get(f"{f.gen}/{target}", self.base.get(f"{f.gen}/screen", 0.0))
        v = 0.0
        for g, coef in self.cat.items():
            lv = f.cats.get(g, "none")
            if lv in coef:
                m += coef[lv]
                v += 1.0 / (self.catw[g][lv] + 1.0 / self.tau2[g])
            else:
                v += self.tau2[g]
        for g, beta in self.num.items():
            m += sum(b * x for b, x in zip(beta, self._numx(g, f)))
        return m, v


# ---------------------------------------------------------------------------------------------- statistics
WINSOR = 0.02            # targets are clipped to their [2%, 98%] quantiles: a collapsed run is harm, not a scale


def winsorize(ys: Sequence[float], q: float = WINSOR) -> list[float]:
    if len(ys) < 10:
        return list(ys)
    s = sorted(ys)
    lo, hi = s[int(q * (len(s) - 1))], s[int((1.0 - q) * (len(s) - 1))]
    return [min(hi, max(lo, y)) for y in ys]


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 3:
        return float("nan")

    def rank(v: Sequence[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2.0
            i = j + 1
        return r
    return pearson(rank(xs), rank(ys))


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx, sy = sum((x - mx) ** 2 for x in xs), sum((y - my) ** 2 for y in ys)
    return sxy / math.sqrt(sx * sy) if sx > 0 and sy > 0 else float("nan")


def slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    """OLS slope of y on x (calibration: 1 = calibrated, < 1 = over-confident spread)."""
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sx = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sx if sx > 0 else float("nan")


def invert(a: Sequence[Sequence[float]]) -> list[list[float]]:
    """Gauss-Jordan inverse with partial pivoting (no numpy here)."""
    n = len(a)
    m = [list(map(float, row)) + [float(i == j) for j in range(n)] for i, row in enumerate(a)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(m[r][c]))
        if abs(m[piv][c]) < 1e-300:
            raise ZeroDivisionError("singular matrix")
        m[c], m[piv] = m[piv], m[c]
        m[c] = [x / m[c][c] for x in m[c]]
        for r in range(n):
            if r != c and m[r][c]:
                m[r] = [x - m[r][c] * y for x, y in zip(m[r], m[c])]
    return [row[n:] for row in m]


def wls(X: Sequence[Sequence[float]], y: Sequence[float], w: Sequence[float], prior_mean: Sequence[float],
        prior_var: Sequence[float]) -> list[float]:
    """Weighted least squares with independent Gaussian priors on the coefficients."""
    k = len(prior_mean)
    a = [[(1.0 / prior_var[i] if i == j else 0.0) for j in range(k)] for i in range(k)]
    b = [prior_mean[i] / prior_var[i] for i in range(k)]
    for x, yy, ww in zip(X, y, w):
        for i in range(k):
            b[i] += ww * x[i] * yy
            for j in range(k):
                a[i][j] += ww * x[i] * x[j]
    inv = invert(a)
    return [sum(inv[i][j] * b[j] for j in range(k)) for i in range(k)]


@dataclass(frozen=True)
class Transfer:
    """Screen -> fresh map of one generation: fresh = c + b * pred + kappa * r * (screen - pred) + N(0, T)."""
    c: float = 0.0
    b: float = 1.0
    kappa: float = 1.0
    T: float = 0.0            # 0: not fit (identity map; T = the model's own residual variance)
    n: int = 0


def fit_transfer(rows: Sequence[tuple[float, float, float, float, float]], floor2: float) -> Transfer | None:
    """``rows``: (fresh mean, its variance, out-of-fold prediction, r, screen - prediction)."""
    if len(rows) < 3:
        return None
    X = [(1.0, p, r * d) for _, _, p, r, d in rows]
    y = winsorize([f for f, _, _, _, _ in rows])
    T = max(floor2, sum((f - sum(y) / len(y)) ** 2 for f in y) / len(y))
    coef = [0.0, 1.0, 1.0]
    for _ in range(8):
        w = [1.0 / (vf + T) for _, vf, _, _, _ in rows]
        coef = wls(X, y, w, (0.0, 1.0, 1.0), (0.01 ** 2, 1.0, 1.0))
        # a forecast transfers at most 1.5x, a node's own screen deviation at most fully (kappa <= 1: shrinkage)
        coef[1], coef[2] = min(1.5, max(0.0, coef[1])), min(1.0, max(0.0, coef[2]))
        res = [(f - sum(c * x for c, x in zip(coef, xx))) ** 2 - vf for f, (_, vf, *_), xx in zip(y, rows, X)]
        T = max(floor2, sum(res) / len(res))
    return Transfer(coef[0], coef[1], coef[2], T, len(rows))


# ---------------------------------------------------------------------------------------------- the fit
@dataclass
class NodeRow:
    nid: str
    ts: float
    feat: Features
    screen: Gauss | None               # first two screen seeds (selection-free)
    sel: Gauss | None                  # every selection seed, over the declared parent
    own: Gauss | None                  # every selection seed, over the measure parent (the node's own increment)
    fresh: Gauss | None                # fresh confirm seeds, pooled (over the measure parent)
    dec_all: Gauss | None = None       # selection + confirm seeds over the declared parent (what a gate conditions on)
    own_all: Gauss | None = None       # ... over the measure parent


@dataclass
class Fit:
    """One predictive fit: per node prior / posterior on the FRESH scale, the transfer per generation, the
    evaluation it ran on, and helpers for candidates, allocation and the plateau forecast."""
    model: Additive
    transfer: Mapping[str, Transfer]
    rows: Mapping[str, NodeRow]
    book: FeatureBook | None
    kept: tuple[str, ...]
    evaluation: Mapping[str, Any] = field(default_factory=dict)
    mde: float = 0.0
    gpu_h: float = 0.0                 # GPU-h a node costs, median of recent closed nodes
    oof: Mapping[str, float] = field(default_factory=dict)
    gate: Mapping[str, Mixture] = field(default_factory=dict)    # gate cell -> nonparametric prior (loop.world.mixture)
    cuts: Mapping[str, Sequence[float]] = field(default_factory=dict)   # generation -> prediction cuts of its cells

    # -- the gates' prior: a nonparametric prior per CELL (generation, or generation x prediction bin)
    def cell(self, row: NodeRow) -> str:
        """The node's gate cell: its generation, split by its OUT-OF-FOLD prediction's bin when ``cuts`` has some."""
        cuts = self.cuts.get(row.feat.gen) or ()
        if not cuts:
            return row.feat.gen
        p = self.oof[row.nid] if row.nid in self.oof else self.model.predict(row.feat)[0]
        return f"{row.feat.gen}:{sum(p > c for c in cuts)}"

    def gate_prior(self, row: NodeRow) -> tuple[str, Mixture] | None:
        for c in (self.cell(row), row.feat.gen, "all"):
            if c in self.gate:
                return c, self.gate[c]
        return None

    # -- per node
    def _tr(self, gen: str) -> Transfer:
        return self.transfer.get(gen) or self.transfer.get("all") or Transfer(T=self.model.s2)

    def _k(self, f: Features, v: float) -> tuple[float, float, float, Transfer]:
        """(screen-scale prediction p, its prior variance S, shrink k for a screen estimate of variance v, transfer)."""
        p, pv = self.model.predict(f)
        S, tr = self.model.s2 + pv, self._tr(f.gen)
        r = S / (S + v) if v > 0 else 1.0
        return p, S, tr.kappa * r, tr

    def prior(self, f: Features, v: float | None = None) -> Gauss:
        """Fresh-scale N(mean, var) of a node's own gain before its own seeds (``v``: its planned screen variance)."""
        v = self.typical_v if v is None else v
        p, S, k, tr = self._k(f, v)
        T = tr.T if tr.T > 0 else self.model.s2
        return tr.c + tr.b * p, T + k * k * (S + v)

    def posterior(self, nid: str) -> Gauss | None:
        """Fresh-scale posterior of a scored node's own gain (:meth:`posterior_of` its row); None if unscored."""
        row = self.rows.get(nid)
        return None if row is None else self.posterior_of(row)

    def posterior_of(self, row: NodeRow, own: Gauss | None = None) -> Gauss:
        """The prior shifted by the node's screen deviation (its own-increment selection seeds, or ``own``),
        then its fresh seeds as direct measurements."""
        e = own or row.own or row.sel
        if e is None:
            m, V = self.prior(row.feat)
        else:
            p, S, k, tr = self._k(row.feat, e[1])
            m, V = tr.c + tr.b * p + k * (e[0] - p), tr.T if tr.T > 0 else self.model.s2
        if row.fresh is not None and row.fresh[1] > 0:
            w0, w1 = 1.0 / V, 1.0 / row.fresh[1]
            m, V = (w0 * m + w1 * row.fresh[0]) / (w0 + w1), 1.0 / (w0 + w1)
        return m, V

    @property
    def typical_v(self) -> float:
        vs = sorted(r.screen[1] for r in self.rows.values() if r.screen is not None)
        return vs[len(vs) // 2] if vs else 0.0

    def lever_effects(self) -> tuple[LeverEffect, ...]:
        """The 'levers' group's coefficient per unit of each registered lever (beta_j / sd_j of the standardized
        column), on the panel; a group the evaluation dropped (or never fit) is 0 with its prior sd tau / sd_j."""
        m, vocab = self.model, (self.book.vocab if self.book is not None else ())
        fitted = "levers" in m.num and len(m.num["levers"]) == len(vocab)
        out = []
        for j, lever in enumerate(vocab):
            sd = m.sd["levers"][j] if fitted else 0.0
            if fitted and sd > 0:
                mean, se = m.num["levers"][j] / sd, math.sqrt(m.numse["levers"][j]) / sd
            else:
                mean, se = 0.0, math.sqrt(m.tau2.get("levers", 0.003 ** 2)) / (sd if sd > 0 else 1.0)
            nz = sum(1 for r in self.rows.values() if len(xs := r.feat.nums.get("levers", ())) > j and xs[j] != 0)
            out.append(LeverEffect(lever, PANEL, mean, se, phi(mean / se) if se > 0 else float(mean > 0), nz))
        return tuple(out)

    # -- candidates and allocation
    def candidate(self, node: Node) -> Gauss:
        """An unrun (or any) node's fresh-scale prior from its creation-time features: the ranking score."""
        return self.prior(self.book.of(node))

    def value(self, node: Node, *, parent: str | None = None, strategy: str | None = None) -> float:
        """P(fresh own gain > MDE) of ``node`` before its own seeds (``parent`` / ``strategy``: as if built there / so):
        what the screen and lane gates compare with ``prior.wm_value_p``."""
        if self.book is None:
            return 1.0
        return self.p_over(self.prior(self.book.of(node, parent=parent, strategy=strategy)), self.mde)

    def support(self, node: Node) -> int:
        """Scored nodes sharing ``node``'s least-measured level of family, mechanism type and init: how much data its
        prediction rests on (0: a brand-new family / type / init, where the fit only extrapolates)."""
        if self.book is None:
            return 0
        f = self.book.of(node)
        return min(sum(1 for r in self.rows.values() if r.feat.cats.get(g) == f.cats.get(g))
                   for g in ("family", "mtype", "init"))

    def lane_value(self, members: Iterable[Node], parent: str | None = None, recent: int = 20) -> tuple[float, int]:
        """(mean :meth:`value` of a lane's ``recent`` newest members, re-parented to ``parent``, n): what one more
        node of that lane (a design batch, a revisit) is expected to be worth; (1.0, 0) for a lane with no history."""
        ms = sorted(members, key=lambda n: (n.created_ts, n.id))[-recent:]
        if not ms:
            return 1.0, 0
        return sum(self.value(n, parent=parent) for n in ms) / len(ms), len(ms)

    def p_over(self, g: Gauss, t: float) -> float:
        m, v = g
        return (1.0 if m > t else 0.0) if v <= 0 else phi((m - t) / math.sqrt(v))

    def allocation(self, carriers: Iterable[str], recent: int = 40) -> list[dict[str, Any]]:
        """Per carrier x mechanism-type cell: the mean over the ``recent`` newest nodes' feature vectors with their
        parent and mechanism type set to the cell (a counterfactual 'the next node, built here'), of the fresh-scale
        prior mean and P(fresh gain > MDE). Sorted best first."""
        nodes = sorted((n for n in self.book.nodes.values() if n.id != ROOT_NODE_ID),
                       key=lambda n: (n.created_ts, n.id))[-recent:]
        types = sorted({r.feat.cats["mtype"] for r in self.rows.values()} - {"none"}) or ["none"]
        out = []
        for c in dict.fromkeys(carriers):
            for t in types:
                gs = [self.prior(self.book.of(n, parent=c, mtype=t)) for n in nodes] if nodes else []
                if not gs:
                    continue
                n_hist = sum(1 for r in self.rows.values() if r.feat.cats["parent"] == c and r.feat.cats["mtype"] == t)
                out.append({"carrier": c, "mtype": t, "mean": sum(m for m, _ in gs) / len(gs),
                            "sd": (sum(v for _, v in gs) / len(gs)) ** 0.5,
                            "p_mde": sum(self.p_over(g, self.mde) for g in gs) / len(gs), "n": n_hist})
        return sorted(out, key=lambda d: (-d["p_mde"], -d["mean"], d["carrier"], d["mtype"]))

    def forecast(self, cells: Sequence[Mapping[str, Any]], ns: Sequence[int] = (10, 25, 50, 100)) -> list[dict[str, float]]:
        """The plateau forecast: E[max fresh gain] and P(any > MDE) over the next N nodes spread over ``cells`` (the
        allocation, weighted by p_mde: where the loop would spend), and their GPU cost."""
        if not cells:
            return []
        ws = [max(c["p_mde"], 1e-6) for c in cells]
        tot = sum(ws)
        mix = [(w / tot, c["mean"], max(c["sd"], 1e-9)) for w, c in zip(ws, cells)]
        lo = min(m - 6 * s for _, m, s in mix)
        hi = max(m + 6 * s for _, m, s in mix)
        grid = [lo + (hi - lo) * i / 400 for i in range(401)]
        F = [sum(w * phi((x - m) / s) for w, m, s in mix) for x in grid]
        out = []
        for n in ns:
            cdf = [f ** n for f in F]
            emax = grid[0] + sum((1.0 - c) * (grid[i + 1] - grid[i]) for i, c in enumerate(cdf[:-1]))
            fm = sum(w * phi((self.mde - m) / s) for w, m, s in mix)
            out.append({"n": float(n), "e_max": emax, "p_any_mde": 1.0 - fm ** n, "gpu_h": n * self.gpu_h})
        return out

    def explain(self, nid: str) -> dict[str, Any]:
        """Why the model expected what it did for ``nid`` and what came out: the per-group contributions to its
        screen-scale prediction, the prediction, and the residual of its own data (fresh, else screen)."""
        row = self.rows.get(nid)
        if row is None:
            return {}
        parts = {}
        for g, coef in self.model.cat.items():
            parts[f"{g}={row.feat.cats.get(g)}"] = coef.get(row.feat.cats.get(g, "none"), 0.0)
        for g, beta in self.model.num.items():
            parts[g] = sum(b * x for b, x in zip(beta, self.model._numx(g, row.feat)))
        pr = self.prior(row.feat)
        obs = row.fresh or row.own or row.sel
        return {"prior_mean": pr[0], "prior_sd": pr[1] ** 0.5, "contributions": dict(sorted(parts.items(),
                key=lambda kv: -abs(kv[1]))), "observed": obs[0] if obs else None,
                "residual": (obs[0] - pr[0]) if obs else None, "on": "fresh" if row.fresh else "screen"}


def explore(key: str, share: float) -> bool:
    """The randomized exploration slot of the value gates: a deterministic draw per decision ``key`` (a node or a
    lane round), so a decision never flips between ticks; ``share`` of keys pass whatever the model says."""
    return int(hashlib.sha1(f"explore:{key}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < share


def gate_value(fit: Fit | None, cfg: Any, key: str, p: float) -> tuple[bool, str]:
    """(pass, why) of a value gate at P(fresh gain > MDE) = ``p``: passes without a fit, at ``p >=
    prior.wm_value_p``, or on the exploration slot (``prior.wm_explore``)."""
    thr, share = cfg.prior.wm_value_p, cfg.prior.wm_explore
    if fit is None or thr <= 0.0:
        return True, ""
    if p >= thr:
        return True, f"P(fresh gain>{fit.mde:.4f})={p:.4f} >= {thr}"
    if explore(key, share):
        return True, f"P(fresh gain>{fit.mde:.4f})={p:.4f} < {thr}, exploration slot ({share:.0%})"
    return False, f"P(fresh gain>{fit.mde:.4f})={p:.4f} < {thr} (predictive model)"


# ---------------------------------------------------------------------------------------------- building
def node_rows(ledger: Any, own_ledger: Any, book: FeatureBook) -> dict[str, NodeRow]:
    """Every scored node's targets: first two SCREEN seeds, all selection seeds (declared and own parent), fresh."""
    from loop.world.noise import SELECTION_STAGES, pool
    out = {}
    for nid, nd in ledger.data.items():
        if nid == ROOT_NODE_ID or nd.n(PANEL) == 0:
            continue
        sel = list(nd.selection.get(PANEL, ()))
        stages = [o.stage for o in nd.obs if o.stage in SELECTION_STAGES and PANEL in o.delta]
        first = [g for g, s in zip(sel, stages) if s is Stage.SCREEN][:2] or sel[:2]
        od = own_ledger.data.get(nid, nd)
        fresh = pool(od.confirm) if od.confirm else None
        osel = list(od.selection.get(PANEL, ()))
        out[nid] = NodeRow(nid, nd.node.created_ts, book.of(nd.node), pool(first) if first else None,
                           pool(sel) if sel else None, pool(osel) if osel else None, fresh,
                           pool([*sel, *nd.confirm]) if sel else None, pool([*osel, *od.confirm]) if osel else None)
    return out


def observations(rows: Iterable[NodeRow], fresh_weight: float) -> list[tuple[Features, Obs]]:
    out = []
    for r in rows:
        if r.screen is not None:
            out.append((r.feat, Obs(r.nid, "screen", r.screen[0], r.screen[1])))
        if r.fresh is not None and fresh_weight > 0:
            out.append((r.feat, Obs(r.nid, "fresh", r.fresh[0], r.fresh[1], fresh_weight)))
    return out


def _fold(nid: str, k: int) -> int:
    return int(hashlib.sha1(nid.encode()).hexdigest()[:8], 16) % k


def cross_fit(rows: Mapping[str, NodeRow], groups: Sequence[str], fresh_weight: float, k: int = 5,
              sweeps: int = SWEEPS) -> dict[str, Gauss]:
    """node -> out-of-fold screen-scale prediction (mean, param var) from ``k`` folds by node id."""
    out = {}
    for f in range(k):
        tr = [r for r in rows.values() if _fold(r.nid, k) != f]
        m = Additive(groups).fit(observations(tr, fresh_weight), sweeps)
        for r in rows.values():
            if _fold(r.nid, k) == f:
                out[r.nid] = m.predict(r.feat)
    return out


def transfers(rows: Mapping[str, NodeRow], oof: Mapping[str, Gauss], s2: float, floor2: float,
              min_n: int) -> dict[str, Transfer]:
    """generation (and "all") -> the screen -> fresh map, fit on nodes with fresh seeds and an out-of-fold prediction."""
    data: dict[str, list] = {"all": []}
    for r in rows.values():
        e = r.own or r.sel
        if r.fresh is None or e is None or r.nid not in oof:
            continue
        p, pv = oof[r.nid]
        S = s2 + pv
        row = (r.fresh[0], r.fresh[1], p, S / (S + e[1]), e[0] - p)
        data.setdefault(r.feat.gen, []).append(row)
        data["all"].append(row)
    out = {}
    for g, xs in data.items():
        if len(xs) >= min_n and (t := fit_transfer(xs, floor2)) is not None:
            out[g] = t
    return out


def build(rows: Mapping[str, NodeRow], kept: Sequence[str], fresh_weight: float, floor2: float, min_fresh: int,
          book: FeatureBook | None = None, sweeps: int = SWEEPS, gate_bins: int = 0,
          warm: Mapping[str, Mixture] | None = None, iters: int = 100, **kw: Any) -> Fit:
    """The model on ``rows`` with feature groups ``kept``: the additive fit, its out-of-fold predictions and the
    per-generation screen -> fresh transfer fit on them."""
    oof = cross_fit(rows, kept, fresh_weight, sweeps=sweeps)
    model = Additive(kept).fit(observations(rows.values(), fresh_weight), sweeps)
    tr = transfers(rows, oof, model.s2, floor2, min_fresh)
    o = {k: v[0] for k, v in oof.items()}
    gate, cuts = fit_gates(rows, o, gate_bins, warm, iters)
    return Fit(model, tr, rows, book, tuple(kept), oof=o, gate=gate, cuts=cuts, **kw)


GATE_MIN = 30          # nodes a gate cell needs (fewer: its generation's, then the pooled prior)


def fit_gates(rows: Mapping[str, NodeRow], oof: Mapping[str, float], bins: int,
              warm: Mapping[str, Mixture] | None = None, iters: int = 100) -> tuple[dict[str, Mixture], dict[str, list[float]]]:
    """(cell -> nonparametric prior fit on its nodes' own selection-seed increments, generation -> cuts): cells are
    the generation, or (``bins`` > 1) the generation x quantile bin of the out-of-fold prediction; every generation
    and "all" are fit too (the fallbacks)."""
    warm = warm or {}
    cuts: dict[str, list[float]] = {}
    if bins > 1:
        for gen in (ROOT_GEN, CARRIER_GEN):
            ps = sorted(oof[r.nid] for r in rows.values() if r.nid in oof and r.feat.gen == gen)
            if len(ps) >= bins * GATE_MIN:
                cuts[gen] = [ps[len(ps) * i // bins] for i in range(1, bins)]
    xs: dict[str, list[Gauss]] = {}
    for r in rows.values():
        e = r.own or r.sel
        if e is None or e[1] <= 0:
            continue
        keys = ["all", r.feat.gen]
        if r.feat.gen in cuts and r.nid in oof:
            keys.append(f"{r.feat.gen}:{sum(oof[r.nid] > c for c in cuts[r.feat.gen])}")
        for k in keys:
            xs.setdefault(k, []).append(e)
    gate = {}
    for k, v in xs.items():
        if len(v) >= (2 if k == "all" else GATE_MIN) and (m := fit_grid(v, warm.get(k), iters)) is not None:
            gate[k] = m
    return gate, cuts


def _blocks(rows: Mapping[str, NodeRow], folds: int, start: float) -> list[tuple[list[NodeRow], list[NodeRow]]]:
    """Forward-chained (train, test) blocks: the newest ``1 - start`` of the nodes by creation in ``folds`` blocks,
    each trained on every node created before its first node."""
    order = sorted(rows.values(), key=lambda r: (r.ts, r.nid))
    n0 = int(len(order) * start)
    size = max(1, (len(order) - n0 + folds - 1) // folds)
    out = []
    for b in range(n0, len(order), size):
        test = order[b:b + size]
        tr = [r for r in order[:b] if r.ts < test[0].ts]
        if len(tr) >= 30:
            out.append((tr, test))
    return out


def time_split(rows: Mapping[str, NodeRow], groups: Sequence[str], fresh_weight: float, folds: int = 6,
               start: float = 0.3, sweeps: int = SWEEPS) -> dict[str, tuple[float, Gauss, Gauss | None]]:
    """Out-of-time SCREEN-scale predictions (:func:`_blocks`): node -> (prediction, its screen target, its fresh)."""
    out = {}
    for tr, test in _blocks(rows, folds, start):
        m = Additive(groups).fit(observations(tr, fresh_weight), sweeps)
        for r in test:
            if r.screen is not None:
                out[r.nid] = (m.predict(r.feat)[0], r.screen, r.fresh)
    return out


def posterior_split(rows: Mapping[str, NodeRow], groups: Sequence[str], fresh_weight: float, floor2: float,
                    min_fresh: int, folds: int = 6, start: float = 0.3, sweeps: int = SWEEPS,
                    bins: Sequence[int] = (0,), iters: int = 40) -> dict[int, dict[str, tuple[Gauss, tuple, Gauss]]]:
    """Out-of-time forecasts of every test node with fresh seeds, the whole pipeline refit per block: bins -> node ->
    (fresh-scale prior before its own seeds, the gate posterior components on its SELECTION seeds alone, its fresh
    outcome), for each gate conditioning in ``bins`` (0: generation cells)."""
    out: dict[int, dict[str, tuple[Gauss, tuple, Gauss]]] = {b: {} for b in bins}
    for tr, test in _blocks(rows, folds, start):
        f = build({r.nid: r for r in tr}, groups, fresh_weight, floor2, min_fresh, sweeps=sweeps, gate_bins=0,
                  iters=iters)
        for b in bins:
            if b:
                f.gate, f.cuts = fit_gates(f.rows, f.oof, b, iters=iters)
            for r in test:
                e = r.own or r.sel
                if r.fresh is None or e is None:
                    continue
                f.oof = {**f.oof, r.nid: f.model.predict(r.feat)[0]}
                if (g := f.gate_prior(r)) is not None:
                    out[b][r.nid] = (f.prior(r.feat), mix_posterior(g[1], e), r.fresh)
    return out


def fresh_stats(pred: Mapping[str, tuple[Gauss, tuple, Gauss]], rows: Mapping[str, NodeRow]) -> dict[str, Any]:
    """rho / calibration slope of the prior means and the gate posterior means on fresh outcomes, the Brier score of
    the posterior predictive P(fresh mean > 0), and the fresh outcome of the nodes the promote rule would pass
    (P(gain>0) >= 0.6 or E[gain+] >= 0.0007), all and per generation."""
    from loop.world.mixture import expected_gain, moments
    ev = {}
    for gen in ("all", ROOT_GEN, CARRIER_GEN):
        ks = [k for k in pred if gen == "all" or rows[k].feat.gen == gen]
        fy = [pred[k][2][0] for k in ks]
        pm, qm = [pred[k][0][0] for k in ks], [moments(pred[k][1])[0] for k in ks]
        pp = [mix_p(pred[k][1], (0.0, pred[k][2][1])) for k in ks]   # + the fresh mean's own noise
        ps = [k for k in ks if mix_p(pred[k][1]) >= 0.6 or expected_gain(pred[k][1]) >= 0.0007]
        ev[gen] = {"n": len(ks), "rho_prior": spearman(pm, fy), "slope_prior": slope(pm, fy),
                   "rho_post": spearman(qm, fy), "slope_post": slope(qm, fy),
                   "brier_post": sum((p - (y > 0)) ** 2 for p, y in zip(pp, fy)) / len(ks) if ks else float("nan"),
                   "mean_fresh": sum(fy) / len(fy) if fy else float("nan"), "n_pass": len(ps),
                   "pass_fresh": sum(pred[k][2][0] for k in ps) / len(ps) if ps else float("nan"),
                   "pass_pos": sum(pred[k][2][0] > 0 for k in ps) / len(ps) if ps else float("nan")}
    return ev


GATE_BINS = (0, 2)       # gate conditionings the evaluation compares (0: generation; k: generation x k bins)
GATE_MARGIN = 0.002      # Brier a finer conditioning must win by


def evaluate(rows: Mapping[str, NodeRow], groups: Sequence[str], fresh_weight: float, min_rho: float,
             floor2: float = 0.0, min_fresh: int = 15, folds: int = 6, sweeps: int = 25) -> dict[str, Any]:
    """The out-of-time evaluation: rho / calibration slope on first screens (all and per generation) of the full
    model and of the generation-only base; per feature group its INCREMENTAL rho over the base (prediction minus
    base vs outcome minus base) -- groups under ``min_rho`` are dropped --; then the kept model's whole pipeline on
    fresh outcomes (:func:`posterior_split`, :func:`fresh_stats`)."""
    full = time_split(rows, groups, fresh_weight, folds, sweeps=sweeps)
    base = time_split(rows, (), fresh_weight, folds, sweeps=sweeps)
    ev: dict[str, Any] = {"n_test": len(full)}

    def stats(pred: Mapping[str, tuple[float, Gauss, Gauss | None]], tag: str) -> None:
        for gen in ("all", ROOT_GEN, CARRIER_GEN):
            ks = [k for k in pred if gen == "all" or rows[k].feat.gen == gen]
            xs, ys = [pred[k][0] for k in ks], [pred[k][1][0] for k in ks]
            ev[f"{tag}{gen}"] = {"n": len(ks), "rho": spearman(xs, ys), "slope": slope(xs, ys)}
    stats(full, "screen:")
    stats(base, "screen_base:")
    kept, comp = [], {}
    for g in groups:
        one = time_split(rows, (g,), fresh_weight, folds, sweeps=sweeps)
        ks = [k for k in one if k in base]
        xs = [one[k][0] - base[k][0] for k in ks]
        ys = [one[k][1][0] - base[k][0] for k in ks]
        rho = spearman(xs, ys) if any(abs(x) > 1e-12 for x in xs) else float("nan")
        comp[g] = rho
        if rho == rho and rho >= min_rho:
            kept.append(g)
    ev["components"], ev["kept"] = comp, kept
    # fresh seeds are few and mostly early: the fresh blocks start at the oldest tenth
    split = posterior_split(rows, kept, fresh_weight, floor2, min_fresh, 5, 0.1, sweeps, GATE_BINS)
    ev["gate"] = {str(b): fresh_stats(split[b], rows) for b in GATE_BINS}
    best = 0
    for b in GATE_BINS[1:]:
        if ev["gate"][str(b)]["all"]["brier_post"] < ev["gate"][str(best)]["all"]["brier_post"] - GATE_MARGIN:
            best = b
    ev["fresh"], ev["gate_bins"] = ev["gate"][str(best)], best
    return ev


def recent_gpu_h(store: Any, k: int = 100) -> float:
    """Median GPU-h of the ``k`` newest nodes' own runs (what one more node costs)."""
    per: dict[str, float] = {}
    for r in store.runs():
        if r.node_id != ROOT_NODE_ID:
            per[r.node_id] = per.get(r.node_id, 0.0) + float(r.gpu_h or 0.0)
    nodes = sorted((n for n in store.nodes() if n.id in per), key=lambda n: n.created_ts)[-k:]
    xs = sorted(per[n.id] for n in nodes)
    return xs[len(xs) // 2] if xs else 0.0


_CACHE: dict[Any, Fit] = {}


def _fingerprint(rows: Mapping[str, NodeRow], kept: Sequence[str], bins: int) -> int:
    return hash((tuple(kept), bins, tuple(sorted((k, r.screen, r.own, r.sel, r.fresh, r.dec_all, r.own_all,
                                                  r.feat.cats.get("parent")) for k, r in rows.items()))))


def fit(ctx: Any, store: Any, plain: Any, ledger: Any, sigma: Mapping[str, float],
        diag: Callable[[str], Mapping[str, float] | None] | None = None, force_eval: bool = False) -> Fit | None:
    """The predictive fit on the corrected ledgers (``plain``: declared parents; ``ledger``: measure parents).
    The time-split evaluation (feature-group selection) reruns every ``prior.wm_eval_every`` new scored nodes and is
    kept in store state ``world.wm``; between evaluations the kept groups stand. None under ``prior.wm_min_nodes``."""
    from loop.novelty import node_type
    from loop.world.evalues import confirm_delta
    pc = ctx.cfg.prior
    book = FeatureBook(ctx, store, plain.nodes, node_type, diag)
    rows = node_rows(plain, ledger, book)
    if len(rows) < max(10, pc.wm_min_nodes):
        return None
    fw, floor2 = pc.wm_fresh_weight, pc.tau_floor ** 2
    st = store.get_state(STATE_KEY) or {}
    if force_eval or "kept" not in st or len(rows) - int(st.get("n", 0)) >= pc.wm_eval_every:
        ev = evaluate(rows, GROUPS, fw, pc.wm_min_rho, floor2, pc.wm_min_fresh)
        st = {"n": len(rows), "kept": ev["kept"], "eval": ev}
        store.set_state(STATE_KEY, st)
        store.log("wm_eval", n=len(rows), kept=ev["kept"], components=ev["components"], fresh=ev["fresh"],
                  gate_bins=ev["gate_bins"], screen={k: v for k, v in ev.items() if k.startswith("screen")})
    sp = sigma.get(PANEL, ctx.cfg.panel_sd)
    warm = {k: Mixture(tuple(tuple(c) for c in v["comps"]), int(v["n"]))
            for k, v in (store.get_state(GATE_KEY) or {}).items() if isinstance(v, Mapping)}
    kept, bins = tuple(st.get("kept") or ()), int((st.get("eval") or {}).get("gate_bins", 0))
    key = (getattr(store, "task_id", ""), _fingerprint(rows, kept, bins))
    if key in _CACHE:                   # nothing it is fit on changed since the last fit: only the book is new
        f = _CACHE[key]
        f.book, f.evaluation = book, st.get("eval") or {}
        return _publish(store, plain, st, f)
    f = build(rows, kept, fw, floor2, pc.wm_min_fresh, book, gate_bins=bins, warm=warm,
                 evaluation=st.get("eval") or {}, mde=confirm_delta(sp, ctx.cfg.promotion.alpha_confirm,
                                                                    ctx.cfg.promotion.confirm_max_seeds),
                 gpu_h=recent_gpu_h(store))
    _CACHE.clear()
    _CACHE[key] = f
    return _publish(store, plain, st, f)


def _publish(store: Any, plain: Any, st: Mapping[str, Any], f: Fit) -> Fit:
    """Store state of a fit: the plateau forecast (refreshed with each evaluation, logged) and the gate priors."""
    if (store.get_state(FORECAST_KEY) or {}).get("n") != st.get("n"):
        from loop.contracts import NodeStatus
        cells = f.allocation([ROOT_NODE_ID, *(n.id for n in plain.nodes.values() if n.status is NodeStatus.CONFIRMED)])
        fc = {"n": st.get("n"), "mde": f.mde, "gpu_h_per_node": f.gpu_h, "top": cells[:5],
              "forecast": f.forecast(cells[:10])}
        store.set_state(FORECAST_KEY, fc)       # the plateau forecast, refreshed with each evaluation
        store.log("wm_forecast", **fc)
    now = {k: {"n": m.n, "comps": [[round(w, 8), round(x, 9), round(v, 12)] for w, x, v in m.prior]}
           for k, m in f.gate.items()}
    if store.get_state(GATE_KEY) != now:
        store.set_state(GATE_KEY, now)
    return f


def explain_text(snapshot: Any, store: Any, node_id: str) -> str:
    """The COMPUTED explanation of a node's outcome for the driver's ``explain(node)`` hook (and briefs): what the
    predictive model expected from its creation-time features (the largest feature-group contributions), what its own
    seeds measured (fresh where it has them) and the residual, its gate posterior, and (loop.diagnostics) its
    checkpoint bottleneck flags with the root -> node trend. '' when there is nothing to say."""
    wm, out = getattr(snapshot, "wm", None), []
    ex = wm.explain(node_id) if wm is not None else {}
    if ex:
        top = ", ".join(f"{k} {v:+.4f}" for k, v in list(ex["contributions"].items())[:4] if abs(v) >= 5e-5)
        out.append(f"model expected {ex['prior_mean']:+.4f} +- {ex['prior_sd']:.4f} (fresh scale) from its features"
                   + (f" [{top}]" if top else ""))
        if ex["observed"] is not None:
            z = ex["residual"] / ex["prior_sd"] if ex["prior_sd"] > 0 else 0.0
            out.append(f"measured {ex['observed']:+.4f} on its {ex['on']} seeds: residual {ex['residual']:+.4f} "
                       f"({z:+.1f} sd) -- " + ("in line with the forecast" if abs(z) < 1.0 else
                                               "better than its features predicted" if z > 0 else
                                               "worse than its features predicted"))
    p = (getattr(snapshot, "nodes", None) or {}).get(node_id)
    if p is not None and p.gain_p is not None:
        out.append(f"gate posterior: own gain {p.increment.mean:+.4f}, P(gain>0)={p.p_gain:.2f}, "
                   f"E[gain+]={p.gain_plus or 0.0:.4f}")
    try:
        from loop import diagnostics as DG
        if s := DG.summary(store, node_id):
            out.append(s)
    except Exception:                       # noqa: BLE001 - no diagnostics table / rows: the model's part stands
        pass
    return "\n".join(out)


def allocation_text(snapshot: Any, carriers: Sequence[str] = (), k: int = 6) -> str:
    """For briefs: the predictive model's allocation -- the ``k`` best carrier x mechanism-type cells by P(fresh own
    gain > MDE) (root, the incumbent and ``carriers``) -- and the plateau forecast. '' without the model."""
    wm = getattr(snapshot, "wm", None)
    if wm is None:
        return ""
    cells = wm.allocation([ROOT_NODE_ID, getattr(snapshot, "incumbent_id", ROOT_NODE_ID), *carriers])
    lines = [f"Predictive model allocation (P(fresh own gain > {wm.mde:.4f}), mean fresh gain), best cells first:"]
    lines += [f"  {c['carrier']} x {c['mtype']}: P {c['p_mde']:.3f}, mean {c['mean']:+.4f} (n={c['n']})"
              for c in cells[:k]]
    fc = wm.forecast(cells[:10])
    if fc:
        lines.append("Plateau forecast: " + "; ".join(
            f"next {int(x['n'])} nodes: E[best gain] {x['e_max']:+.4f}, P(any > MDE) {x['p_any_mde']:.2f}, "
            f"{x['gpu_h']:.0f} GPU-h" for x in fc))
    return "\n".join(lines)
