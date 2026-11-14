"""loop.core.factorial — the one HP / stacking engine (docs/loop-core-v3.md §5).

A WAVE is a 2-level resolution-IV fractional factorial over k <= 7 knobs on one carrier: 8 points for k <= 4,
16 for k <= 7, in blocks of 8 points that share one fresh SELECT seed (the block is confounded with the AB = CE = FG
2FI alias chain, orthogonal to every main effect). Its result is the main effects only:

    β̂_j = ȳ(+) − ȳ(−) on the screening composite (OLS with one intercept per block: exact in a balanced wave,
          still unbiased when a point is missing)
    se²_j = 4 s² [(XᵀX)⁻¹]_jj  (= 4s²/N balanced), s² = max((1−ρ)σ², the wave's residual variance) when the wave
          has >= 4 residual dfs, else (1−ρ)σ²: σ², ρ the model's fitted composite noise (child/parent ρ, §8)
    β̃_j   = the posterior of β_j under the factorial-lane prior (location 0, scale² τ²_g, Student-t)

The best cell is NEVER picked (winner's curse: it is the max of N noisy cells). The next step x* sets knob j to
sign(β̂_j) where |β̂_j|/se > 1, every other knob stays at the carrier; x* becomes a normal node (lane factorial) whose
prior is Σ_j β̃_j (x*_j − c_j)/2 (c_j the carrier's coded level), and it graduates only through the fresh-seed test.
A significant effect (|β̂|/se >= 2) at a range edge moves that knob's next-wave range one step out.

Knob sources: researcher-declared ``knobs`` (:func:`declared`), diagnostics flags -> YAML-registered keys
(:func:`flagged`), and stacking: one ``diff`` knob per candidate with m > 0 on the carrier (:func:`stacking`; a point
with the knob on carries that node's diff, grafted by the adapter's ``graft_port`` / ``merge_config``).

The retro (:func:`retro`) re-estimates every historical factorial in the ledger (the old loop's ``legs`` leg / dose
designs and ``design`` batches, state rows) by this module's estimator; :func:`retro_bank` turns the effects measured on
the carrier's lineage into the knobs of its first wave: a significant (pooled |z| >= 2) positive leg is a diff knob, a
config move the carrier does not hold is that range, one it holds at the better edge moves one step out (the edge
rule), one the carrier has left is dropped. :func:`forecast` prices a wave: the expected gain of its x*."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Literal, Mapping, Sequence

from loop.core import model as M
from loop.core.data import Ledger, Node

Scale = Literal["lin", "log", "diff"]
MAX_K = 7
BLOCK = 8
Z_NEXT = 1.0            # |β̂|/se above which x* moves the knob
MIN_RESID_DF = 4        # a wave with this many residual dfs may raise its se to its own residual sd
Z_EDGE = 2.0            # |β̂|/se at which the next wave's range moves one step out
FACTORIAL_FAMILY = "design"     # contracts.DESIGN_FAMILY: the family of wave points and their x*
#: extra columns over the base factors A.. (odd words: the all-low row stays all-low); resolution >= IV
GENERATORS = {8: ("ABC",), 16: ("ABC", "BCD", "ACD")}
KNOB_SCHEMA = {"type": "object", "required": ["key", "lo", "hi"], "properties": {
    "key": {"type": "string", "minLength": 1}, "lo": {}, "hi": {}, "scale": {"type": "string", "enum": ["lin", "log"]}}}


@dataclass(frozen=True)
class Knob:
    key: str            # dotted config key, or (scale "diff") the node whose diff is stacked on the carrier
    lo: Any
    hi: Any
    scale: Scale = "lin"

    def level(self, x: int) -> Any:
        return self.hi if x > 0 else self.lo

    def coded(self, value: Any) -> float:
        """``value`` in design units (lo = −1, hi = +1; log scale for ``log``); a diff knob's carrier is −1 (off), a
        non-numeric level (a list) is +1 at hi, else −1."""
        if self.scale == "diff" or value is None:
            return -1.0
        if not (_num(value) and _num(self.lo) and _num(self.hi)):
            return 1.0 if value == self.hi else -1.0
        f = math.log if self.scale == "log" else float
        return 2 * (f(value) - f(self.lo)) / (f(self.hi) - f(self.lo)) - 1


@dataclass(frozen=True)
class Wave:
    id: str
    carrier: str
    knobs: tuple[Knob, ...]
    points: tuple[dict, ...]      # key -> value of every knob
    blocks: tuple[int, ...]       # block of each point
    seeds: tuple[int, ...] = ()   # one fresh SELECT seed per block

    def rows(self) -> list[tuple[int, ...]]:
        return [tuple(1 if p[k.key] == k.hi else -1 for k in self.knobs) for p in self.points]


@dataclass(frozen=True)
class Effect:
    key: str
    beta: float         # ȳ(+) − ȳ(−)
    se: float
    shrunk: float       # posterior mean under the factorial prior
    v: float            # its posterior variance

    @property
    def z(self) -> float:
        return self.beta / self.se if self.se > 0 else 0.0


# -- knob sources ---------------------------------------------------------------------------------------------------
def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def valid(kb: Knob) -> bool:
    if kb.scale == "diff":
        return bool(kb.key)
    if not kb.key or kb.lo == kb.hi or kb.lo is None or kb.hi is None:
        return False
    return not (kb.scale == "log" and not (_num(kb.lo) and _num(kb.hi) and kb.lo > 0 and kb.hi > 0))


def declared(items: Iterable[Mapping[str, Any]] | None) -> list[Knob]:
    """The researcher finish's ``knobs`` (:data:`KNOB_SCHEMA`); malformed ones are dropped."""
    out = []
    for it in items or ():
        try:
            kb = Knob(str(it["key"]), it["lo"], it["hi"], it.get("scale") or "lin")
        except (KeyError, TypeError):
            continue
        if kb.scale in ("lin", "log") and valid(kb):
            out.append(kb)
    return out


def flagged(flags: Iterable[str], registry: Mapping[str, Sequence[Mapping[str, Any]]],
            carrier_cfg: Mapping[str, Any] | None = None) -> list[Knob]:
    """Diagnostics flags (e.g. ``lr_throttled``) -> the knobs the task YAML registers for them. An entry is either
    absolute ``{key, lo, hi, scale}`` or relative to the carrier ``{key, factor, scale}``: lo = the carrier's value,
    hi = lo x factor (``clip_dominated`` -> the grad-clip threshold x3); a relative entry needs a numeric carrier value."""
    out = []
    for f in flags:
        for it in registry.get(f, ()):
            if "factor" in it:
                c = (carrier_cfg or {}).get(str(it.get("key", "")))
                if not _num(c) or not _num(it["factor"]) or c == 0:
                    continue
                it = {"key": it["key"], "lo": c, "hi": float(f"{c * it['factor']:.4g}"), "scale": it.get("scale")}
            out += declared([it])
    return out


def stacking(carrier: str, nodes: Mapping[str, Node], m: Mapping[str, float]) -> list[Knob]:
    """One ``diff`` knob (off = carrier, on = carrier + the node's diff) per candidate built on ``carrier`` with
    posterior m > 0, best first."""
    cands = [n for n, x in m.items() if x > 0 and n != carrier and n in nodes and nodes[n].carrier == carrier]
    return [Knob(n, 0, 1, "diff") for n in sorted(cands, key=lambda n: -m[n])]


def pick(*sources: Iterable[Knob], k: int = MAX_K) -> tuple[Knob, ...]:
    """The first ``k`` valid knobs over the sources in order, one per key."""
    out: dict[str, Knob] = {}
    for kb in (kb for src in sources for kb in src):
        if valid(kb) and kb.key not in out and len(out) < k:
            out[kb.key] = kb
    return tuple(out.values())


# -- design ---------------------------------------------------------------------------------------------------------
def matrix(k: int) -> tuple[list[tuple[int, ...]], list[int]]:
    """(rows, block of each row): 8 rows for k <= 4, 16 for k <= 7 (two blocks split on AB). Row 0 is all-low.
    k < 3 takes the first k columns of the 2³, whose 2^k distinct rows repeat 8 / 2^k times: each REPLICATE is a
    block of its own (v3-c7a A2: w013, one knob, had 4 byte-identical configs per level at ONE seed -- 3 of every 4
    runs measured nothing new); with a fresh seed per block the replicates measure the seed noise instead."""
    if not 1 <= k <= MAX_K:
        raise ValueError(f"a wave takes 1..{MAX_K} knobs, got {k}")
    runs = 8 if k <= 4 else 16
    base = runs.bit_length() - 1
    rows, blocks, seen = [], [], {}
    for r in range(runs):
        b = {chr(65 + j): 1 if r >> (base - 1 - j) & 1 else -1 for j in range(base)}
        cols = list(b.values()) + [math.prod(b[c] for c in w) for w in GENERATORS[runs]]
        row = tuple(cols[:k])
        rows.append(row)
        if k < 3:                                # a replicated full factorial: the i-th copy of a row -> block i
            blocks.append(seen.get(row, 0))
            seen[row] = seen.get(row, 0) + 1
        else:
            blocks.append(0 if runs == BLOCK or b["A"] * b["B"] < 0 else 1)
    return rows, blocks


def n_blocks(k: int) -> int:
    """Seed blocks (one fresh SELECT seed each) of a ``k``-knob wave (:func:`matrix`)."""
    return max(matrix(k)[1]) + 1


def design(wave_id: str, carrier: str, knobs: Sequence[Knob], seeds: Sequence[int],
           blocks: Sequence[int] | None = None) -> Wave:
    """The wave over ``knobs``; ``seeds`` holds one fresh SELECT seed per block (:func:`n_blocks`). Points with
    identical levels in one block (one config at one seed: a duplicate measurement) are left out (:func:`duplicates`).

    ``blocks``: the block of each design row as the build journal recorded it (``factorial.building`` ``blocks``),
    used as given. A journal written before v3-c7a A2 holds no ``blocks`` and ONE seed for a k < 3 wave (live 10-14:
    w017, k 1, seeds [100147]): with fewer seeds than :func:`n_blocks` the old single-block layout of k < 3 is
    rebuilt (every row block 0, replicates kept: the point indices its ``built`` map names), never a ValueError
    (k >= 3 had the same layout before A2)."""
    rows, mblocks = matrix(len(knobs))
    legacy = False
    if blocks is not None and len(blocks) == len(rows):
        mblocks = [int(b) for b in blocks]
    elif len(seeds) < max(mblocks) + 1 and len(knobs) < 3 and seeds:
        mblocks, legacy = [0] * len(rows), True        # the pre-A2 layout of k < 3: one block, one seed
    if max(mblocks) + 1 > len(seeds):
        raise ValueError(f"wave {wave_id}: {max(mblocks) + 1} block seeds needed, got {len(seeds)}")
    pts = tuple({kb.key: kb.level(x) for kb, x in zip(knobs, row)} for row in rows)
    drop = set() if legacy else set(duplicates(pts, mblocks))
    keep = [i for i in range(len(pts)) if i not in drop]
    return Wave(wave_id, carrier, tuple(knobs), tuple(pts[i] for i in keep), tuple(mblocks[i] for i in keep),
                tuple(seeds[:max(mblocks) + 1]))


def duplicates(points: Sequence[Mapping[str, Any]], blocks: Sequence[int]) -> list[int]:
    """Indices of the points whose levels (canonical JSON) repeat an earlier point of the SAME block: the same config
    at the same seed, a byte-identical replicate (``design`` leaves them out; the tick logs them)."""
    seen, out = set(), []
    for i, (p, b) in enumerate(zip(points, blocks)):
        key = (b, json.dumps(p, sort_keys=True, default=str))
        if key in seen:
            out.append(i)
        seen.add(key)
    return out


# -- estimation -----------------------------------------------------------------------------------------------------
def _inv(a: list[list[float]]) -> list[list[float]]:
    n = len(a)
    m = [row[:] + [float(i == j) for j in range(n)] for i, row in enumerate(a)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        if abs(m[p][c]) < 1e-12:
            raise ValueError("singular design")
        m[c], m[p] = m[p], m[c]
        m[c] = [x / m[c][c] for x in m[c]]
        for r in range(n):
            if r != c:
                m[r] = [x - m[r][c] * y for x, y in zip(m[r], m[c])]
    return [row[n:] for row in m]


def main_effects(rows: Sequence[Sequence[int]], blocks: Sequence[int], ys: Mapping[int, float]
                 ) -> tuple[list[tuple[float, float]], float, int]:
    """([(β̂_j, [(XᵀX)⁻¹]_jj)] per knob, residual variance, residual df) from the observed points ``ys`` (row
    index -> value); OLS on one intercept per block + the ±1 main-effect columns. β̂ = 2·coefficient = ȳ(+) − ȳ(−)
    when balanced."""
    idx = sorted(ys)
    bl = sorted({blocks[i] for i in idx})
    k = len(rows[0])
    x = [[float(blocks[i] == b) for b in bl] + [float(rows[i][j]) for j in range(k)] for i in idx]
    p = len(x[0])
    cov = _inv([[sum(r[a] * r[b] for r in x) for b in range(p)] for a in range(p)])
    xty = [sum(r[a] * ys[i] for r, i in zip(x, idx)) for a in range(p)]
    coef = [sum(cov[a][b] * xty[b] for b in range(p)) for a in range(p)]
    df = len(idx) - p
    ss = sum((ys[i] - sum(c * v for c, v in zip(coef, r))) ** 2 for r, i in zip(x, idx))
    return [(2 * coef[len(bl) + j], cov[len(bl) + j][len(bl) + j]) for j in range(k)], ss / df if df else 0.0, df


def estimate(wave: Wave, ys: Mapping[int, float], noise: M.Noise, tau2: float, *,
             target: str = M.COMPOSITE, nu: float | None = M.NU, min_df: int = MIN_RESID_DF,
             far: M.Noise | None = None) -> list[Effect]:
    """Main effects of ``wave`` on ``target`` (point index -> its value), each shrunk by the factorial prior
    (location 0, scale² ``tau2``). ``se² = 4 s² [(XᵀX)⁻¹]_jj`` with s² the larger of the model's ``(1−ρ)σ²`` and,
    when the wave has >= ``min_df`` residual dfs, its own residual variance (leg waves carrying code diffs ran
    1.2-1.55x noisier than the model on the 10-08 ledger). With fewer residual dfs the wave cannot check its own
    noise, so the larger of the near and the ``far`` (several-changes) ρ's ``(1−ρ)σ²`` stands in: a point is up to
    k changes from its carrier, and the near ρ (.94 on the CRN base) would make its se ~2.5x too small."""
    effects, resid, df = main_effects(wave.rows(), wave.blocks, ys)
    s2 = (1 - noise.rho[target]) * noise.sigma2[target]
    if df >= min_df:
        s2 = max(s2, resid)
    elif far is not None and target in far.sigma2:
        s2 = max(s2, (1 - far.rho[target]) * far.sigma2[target])
    out = []
    for kb, (beta, c) in zip(wave.knobs, effects):
        var = 4 * s2 * c
        m, v = M.posterior(0.0, tau2, M.Estimate(beta, var, len(ys), 0), nu)
        out.append(Effect(kb.key, beta, math.sqrt(var), m, v))
    return out


def factorial_tau2(prior: M.Prior, carrier: str) -> float:
    """The factorial-lane prior variance on ``carrier`` (the history of the old HP lanes sets it)."""
    return prior.cell((carrier, FACTORIAL_FAMILY, "factorial"))[1]


# -- next step ------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Next:
    levels: dict[str, Any]   # the knobs x* moves off the carrier -> their level
    m: float                 # prior mean of x* over the carrier: Σ β̃_j (x*_j − c_j)/2
    v: float


def next_point(wave: Wave, effects: Sequence[Effect], carrier_cfg: Mapping[str, Any], *,
               z: float = Z_NEXT) -> Next | None:
    """x*: knob j at sign(β̂_j) where |β̂_j|/se > ``z``, every other knob at the carrier. None: nothing moves."""
    levels, m, v = {}, 0.0, 0.0
    for kb, e in zip(wave.knobs, effects):
        if abs(e.z) <= z:
            continue
        x = 1 if e.beta > 0 else -1
        step = (x - kb.coded(carrier_cfg.get(kb.key))) / 2
        if abs(step) < 1e-9:
            continue
        levels[kb.key] = kb.level(x)
        m, v = m + e.shrunk * step, v + e.v * step * step
    return Next(levels, m, v) if levels else None


def extend(kb: Knob, e: Effect, bounds: tuple[float, float] | None = None, *, z: float = Z_EDGE) -> Knob | None:
    """The knob's next-wave range: one step out past the better edge when the effect is significant (lin: by
    hi − lo; log: by hi / lo), clipped to ``bounds``; None when clipping leaves no range. Diff knobs never move."""
    if kb.scale == "diff" or abs(e.z) < z or not (_num(kb.lo) and _num(kb.hi)):
        return kb
    up = e.beta > 0
    if kb.scale == "log":
        r = kb.hi / kb.lo
        lo, hi = (kb.hi, kb.hi * r) if up else (kb.lo / r, kb.lo)
    else:
        d = kb.hi - kb.lo
        lo, hi = (kb.hi, kb.hi + d) if up else (kb.lo - d, kb.lo)
    if bounds is not None:                       # a descending knob (lo > hi) is clipped the same way
        lo, hi = (min(max(x, bounds[0]), bounds[1]) for x in (lo, hi))
    if isinstance(kb.lo, int) and isinstance(kb.hi, int):
        lo, hi = round(lo), round(hi)
    else:                                        # 4 significant digits: .9, not .9000000000000001
        lo, hi = float(f"{lo:.4g}"), float(f"{hi:.4g}")
    nk = Knob(kb.key, lo, hi, kb.scale)
    return nk if valid(nk) and lo != hi else None


def next_knobs(wave: Wave, effects: Sequence[Effect], bounds: Mapping[str, tuple[float, float]] | None = None
               ) -> tuple[Knob, ...]:
    out = (extend(kb, e, (bounds or {}).get(kb.key)) for kb, e in zip(wave.knobs, effects))
    return tuple(kb for kb in out if kb is not None)


# -- the ledger side ------------------------------------------------------------------------------------------------
_TAG = re.compile(r"(?:^|\s)wave=(\S+) point=(\d+) block=(\d+) seed=(\d+)")
_PRIOR = re.compile(r"(?:^|\s)prior_m=(\S+) prior_v=(\S+)")


def is_point(notes: str) -> bool:
    """A wave point (launched by its wave, never screened on its own)."""
    return bool(_TAG.search(notes or ""))


def point_tag(wave: Wave, i: int) -> str:
    return f"wave={wave.id} point={i} block={wave.blocks[i]} seed={wave.seeds[wave.blocks[i]]}"


def wave_points(data: Ledger, wave_id: str) -> dict[int, tuple[str, int]]:
    """point index -> (node, seed) of the wave's materialised points."""
    out = {}
    for n in data.nodes.values():
        if (t := _TAG.search(n.notes)) and t.group(1) == wave_id:
            out[int(t.group(2))] = (n.id, int(t.group(4)))
    return out


def wave_runs(wave_id: str, data: Ledger) -> list[tuple[str, int]]:
    """(node, seed) of every point run still to submit: the tick's ``runs_for`` of a ``wave`` action."""
    have = {(r.node, r.seed) for r in data.runs}
    return [ns for _, ns in sorted(wave_points(data, wave_id).items()) if ns not in have]


def wave_controls(data: Ledger, wave_id: str) -> list[tuple[str, int]]:
    """(carrier, seed) of the wave's control runs still to submit (v3-c7a A2): ONE run of the carrier at every block
    seed it has no run on, so every point pairs with its parent at its own seed (a point's level alone is a seed
    offset: 10-13, w013.p03 ranked "best" at .6858 with no control at seed 100138). A crashed / voided carrier run
    is no control (v3 review O4): its seed is listed again (the tick re-submits the dead PARENT row)."""
    pts = wave_points(data, wave_id)
    if not pts:
        return []
    carrier = data.nodes[next(iter(pts.values()))[0]].parent
    if carrier is None:
        return []
    have = {r.seed for r in data.runs if r.node == carrier and r.status in CONTROL_OK}   # v3 review O4: a dead
    return [(carrier, s) for s in sorted({s for _, s in pts.values()}) if s not in have]   # run is no control


CONTROL_OK = frozenset({"submitting", "running", "done"})      # a carrier run that is (or will be) a wave's control


def point_values(data: Ledger, wave_id: str, weights: Mapping[str, float], target: str = M.COMPOSITE
                 ) -> dict[int, float]:
    """point index -> the target at the point's block seed (the mean if it ran there twice)."""
    by = data.by_node()
    out = {}
    for i, (node, seed) in wave_points(data, wave_id).items():
        ys = [o.value(target, weights) for o in by.get(node, ()) if o.seed == seed]
        if ys:
            out[i] = sum(ys) / len(ys)
    return out


def node_prior(notes: str) -> tuple[float, float] | None:
    """(m, v) of an x* node (its notes ``prior_m=.. prior_v=..``), None for any other node."""
    t = _PRIOR.search(notes or "")
    return (float(t.group(1)), float(t.group(2))) if t else None


def to_json(wave: Wave) -> dict[str, Any]:
    return asdict(wave)


def from_json(d: Mapping[str, Any]) -> Wave:
    return Wave(d["id"], d["carrier"], tuple(Knob(**k) for k in d["knobs"]), tuple(dict(p) for p in d["points"]),
                tuple(d["blocks"]), tuple(d.get("seeds", ())))


# -- building points ------------------------------------------------------------------------------------------------
def build_point(adapter: Any, carrier_commit: str, levels: Mapping[str, Any], knobs: Sequence[Knob],
                diffs: Mapping[str, tuple[str, str]], name: str, config_rel: str) -> tuple[str, str]:
    """(commit, '') of the carrier at ``levels``, or ('', why). Every diff knob that is on is grafted onto the
    carrier in turn (``adapter.graft_port``: config per key via ``merge_config``, code by 3-way; ``diffs`` node ->
    (its parent commit, its commit)); then the config knobs are written into the YAML and committed."""
    from loop.agents.researcher import set_yaml_value
    if hasattr(adapter, "dry_build"):            # shadow: checked, never built (tick.DryRunAdapter)
        return adapter.dry_build(carrier_commit, levels, knobs, diffs, name)
    cur = carrier_commit
    for kb in knobs:
        if kb.scale == "diff" and levels.get(kb.key) == kb.hi:
            sha, how = adapter.graft_port(*diffs[kb.key], cur, f"{name}-{kb.key}")
            if how.startswith("conflict"):
                return "", f"{kb.key}: {how}"
            cur = sha or cur
    cfg = {kb.key: levels[kb.key] for kb in knobs if kb.scale != "diff" and kb.key in levels}
    if not cfg:
        return cur, ""
    if hasattr(adapter, "commit_files") and hasattr(adapter, "file_at"):   # v3-c8a B5: no worktree (git plumbing)
        text = adapter.file_at(cur, config_rel)
        if text is None:
            return "", f"config: {config_rel} unreadable at {cur[:12]}"
        try:
            for k, v in cfg.items():
                text = set_yaml_value(text, k, v)
        except ValueError as exc:
            return "", f"config: {exc}"
        return adapter.commit_files(cur, {config_rel: text}, message=f"loop wave {name}: {cfg}"), ""
    wt = adapter.prepare_worktree(parent_commit=cur, name=name)
    try:
        path = Path(wt.path) / config_rel
        text = path.read_text()
        for k, v in cfg.items():
            text = set_yaml_value(text, k, v)
        path.write_text(text)
        return adapter.commit(wt, message=f"loop wave {name}: {cfg}"), ""
    except (OSError, ValueError) as exc:
        return "", f"config: {exc}"
    finally:
        adapter.reap_worktree(wt)


@contextmanager
def atomic(store: Any) -> Iterator[None]:
    """One SQLite transaction over the store's writes inside (``BEGIN IMMEDIATE`` .. ``COMMIT``, ROLLBACK on an
    error): a crash leaves all of them or none. A store without a SQLite connection (a test double), or a transaction
    already open, runs the writes as they are."""
    db, lock = getattr(store, "_db", None), getattr(store, "_lock", None)
    if db is None or lock is None or getattr(db, "in_transaction", True):
        yield
        return
    with lock:
        db.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            db.execute("ROLLBACK")
            raise
        db.execute("COMMIT")


def materialize(store: Any, adapter: Any, wave: Wave, diffs: Mapping[str, tuple[str, str]], config_rel: str,
                new_id: Callable[[], str], now: float | None = None) -> tuple[list[str], list[str]]:
    """Every point of ``wave`` as a node (parent the carrier, lane factorial, tagged :func:`point_tag`); the wave
    is kept in state ``wave:<id>``. Returns (node ids, failures); a point that does not build is left out (the
    analysis is OLS on the points that ran). Every point is built first (git only); then the wave state and the
    point nodes are ONE transaction. The tick's own build is ``Tick._finish_wave`` (journal ``factorial.building``,
    then one commit that also holds the seed block, the counter and the bank's ``used`` marks)."""
    carrier = store.get_node(wave.carrier)
    built, fails = [], []
    for i, levels in enumerate(wave.points):
        sha, why = build_point(adapter, carrier.commit, levels, wave.knobs, diffs, f"{wave.id}-p{i}", config_rel)
        if sha:
            built.append((i, sha))
        else:
            fails.append(f"{wave.id} point {i}: {why}")
    with atomic(store):
        store.set_state(f"wave:{wave.id}", to_json(wave))
        ids = [_put(store, new_id(), sha, carrier, now, f"{point_tag(wave, i)} {_levels(wave.points[i])}", None)
               for i, sha in built]
    return ids, fails


def materialize_next(store: Any, adapter: Any, wave: Wave, nxt: Next, diffs: Mapping[str, tuple[str, str]],
                     config_rel: str, new_id: Callable[[], str], now: float | None = None) -> tuple[str, str]:
    """x* as a normal node (Operator draft, lane factorial) carrying its prior; (node id, '') or ('', why)."""
    from loop.contracts import Operator
    carrier = store.get_node(wave.carrier)
    sha, why = build_point(adapter, carrier.commit, nxt.levels, wave.knobs, diffs, f"{wave.id}-next", config_rel)
    if not sha:
        return "", why
    notes = f"factorial_of={wave.id} prior_m={nxt.m:.6g} prior_v={nxt.v:.6g} {_levels(nxt.levels)}"
    return _put(store, new_id(), sha, carrier, now, notes, Operator.DRAFT), ""


def _levels(levels: Mapping[str, Any]) -> str:
    return "design " + ", ".join(f"{k}={json.dumps(v)}" for k, v in levels.items())


def _put(store: Any, nid: str, commit: str, carrier: Any, now: float | None, notes: str, operator: Any) -> str:
    from loop.contracts import Node as StoreNode, NodeStatus, Strategy
    ts = time.time() if now is None else now
    store.put_node(StoreNode(
        id=nid, task_id=carrier.task_id, commit=commit, parent_id=carrier.id, strategy=Strategy.DESIGN,
        operator=operator, family=FACTORIAL_FAMILY, mechanism_tag=notes.split()[0],
        diff_hash=hashlib.sha256(f"{carrier.commit}\n{commit}".encode()).hexdigest(), status=NodeStatus.SCREENING,
        created_ts=ts, updated_ts=ts, notes=notes))
    return nid



# -- retro: the historical factorials of the ledger ----------------------------------------------------------------
LEG = re.compile(r"_leg_(n\d+)$")
LOG_RATIO = 1.5         # a positive range spanning >= this ratio is designed / extended on the log scale


def auto_scale(lo: Any, hi: Any) -> Scale:
    """log for a positive numeric range spanning a ratio >= :data:`LOG_RATIO` (lr 1.25e-4 -> 1.875e-4, kde 10 -> 6.67:
    their next step is x1.5 / x2/3), else lin (layerwise .7 -> .8, a list level)."""
    if _num(lo) and _num(hi) and lo > 0 and hi > 0 and max(lo, hi) / min(lo, hi) >= LOG_RATIO * (1 - 1e-3):
        return "log"
    return "lin"


@dataclass(frozen=True)
class HistWave:
    name: str
    carrier: str
    knobs: tuple[Knob, ...]
    rows: tuple[tuple[int, ...], ...]
    points: Mapping[int, str]           # row -> node
    seeds: frozenset[int] | None        # the wave's block seeds (None: every seed its points ran on)


def hist_waves(legs: Mapping[str, Any] | None, design: Mapping[str, Any] | None) -> list[HistWave]:
    """The old loop's factorials (state ``legs`` / ``design``): a ``new`` leg factor (``*_leg_<node>``, 0/1) is that
    node's diff knob, every other factor a config knob over its (low, high)."""
    out = []
    for b in (legs or {}).get("batches", ()):
        for dn, d in (b.get("designs") or {}).items():
            ks = tuple(Knob(m.group(1), 0, 1, "diff") if f.get("kind") == "new" and (m := LEG.search(f["key"]))
                       else Knob(f["key"], f.get("low"), f.get("high"), auto_scale(f.get("low"), f.get("high")))
                       for f in d.get("factors", ()))
            pts = d.get("points") or {}
            out.append(HistWave(f"L{b['batch']}-{dn}", b.get("carrier") or "root", ks,
                                tuple(tuple(r) for r in d.get("rows", ())),
                                {int(k): v for k, v in (pts.items() if isinstance(pts, Mapping) else enumerate(pts))},
                                frozenset(b.get("seeds") or ()) or None))
    for b in (design or {}).get("batches", ()):
        if not (b.get("points") and b.get("rows")):
            continue
        pts = b["points"]
        ks = tuple(Knob(k["key"], k.get("low"), k.get("high"), auto_scale(k.get("low"), k.get("high")))
                   for k in b.get("knobs", ()))
        out.append(HistWave(f"d{b['batch']}", b.get("carrier") or "root", ks, tuple(tuple(r) for r in b["rows"]),
                            {int(k): v for k, v in (pts.items() if isinstance(pts, Mapping) else enumerate(pts))}, None))
    return out


@dataclass(frozen=True)
class RetroEffect:
    wave: str
    carrier: str
    knob: Knob
    beta: float          # y(hi) - y(lo) on the composite
    se: float
    n: int

    @property
    def z(self) -> float:
        return self.beta / self.se if self.se > 0 else 0.0


def retro(data: Ledger, waves: Sequence[HistWave], noise: M.Noise, weights: Mapping[str, float], *,
          target: str = M.COMPOSITE, min_df: int = MIN_RESID_DF) -> list[RetroEffect]:
    """Main effects of every historical wave, :func:`estimate`'s rules (one intercept per seed block, se from the
    model's (1−ρ)σ² raised to the wave's residual variance given >= ``min_df`` residual dfs). A factor that is
    constant / aliased among the observed points is dropped (the last one first) until the design is estimable."""
    by = data.by_node()
    out = []
    for w in waves:
        xs, blocks, ys = [], [], {}
        for r, node in w.points.items():
            if r >= len(w.rows):
                continue
            for o in by.get(node, ()):
                if w.seeds is None or o.seed in w.seeds:
                    ys[len(xs)] = o.value(target, weights)
                    xs.append(w.rows[r])
                    blocks.append(o.seed)
        keep = list(range(len(w.knobs)))
        if len(ys) < len(keep) + len(set(blocks)) + 2:
            continue
        while keep:
            try:
                eff, resid, df = main_effects([tuple(x[j] for j in keep) for x in xs], blocks, ys)
                break
            except ValueError:
                keep.pop()
        else:
            continue
        s2 = (1 - noise.rho[target]) * noise.sigma2[target]
        s2 = max(s2, resid) if df >= min_df else s2
        out += [RetroEffect(w.name, w.carrier, w.knobs[j], beta, math.sqrt(4 * s2 * c), len(ys))
                for j, (beta, c) in zip(keep, eff)]
    return out


def retro_tau2(effects: Sequence[RetroEffect], floor: float = 1e-8) -> float:
    """The empirical factorial prior: var(β̂) − mean(se²) over the retro's main effects (10-08 copy: τ ≈ .0016)."""
    if len(effects) < 3:
        return floor
    bs = [e.beta for e in effects]
    mb = sum(bs) / len(bs)
    return max(sum((b - mb) ** 2 for b in bs) / len(bs) - sum(e.se ** 2 for e in effects) / len(effects), floor)


@dataclass(frozen=True)
class BankKnob:
    knob: Knob           # oriented: lo = the carrier's level (row 0 = the carrier), hi = the move to test
    m: float             # prior mean of its main effect (the move lo -> hi), composite
    v: float
    source: str          # retro / stacking / flag / declared / next / plateau
    why: str = ""
    z: float = 0.0       # retro: the pooled |β̂|/se it was selected on (0: not measured)

    def to_json(self) -> dict[str, Any]:
        return {"knob": asdict(self.knob), "m": self.m, "v": self.v, "source": self.source, "why": self.why,
                "z": self.z}

    @staticmethod
    def from_json(d: Mapping[str, Any]) -> "BankKnob":
        return BankKnob(Knob(**d["knob"]), float(d["m"]), float(d["v"]), d["source"], d.get("why", ""),
                        float(d.get("z") or 0.0))

    def signature(self) -> str:
        """What a wave consumes: the knob's key and range (a used signature is never offered again on its carrier)."""
        return json.dumps([self.knob.key, self.knob.lo, self.knob.hi], sort_keys=True, default=str)


def _same(a: Any, b: Any) -> bool:
    if _num(a) and _num(b):
        return math.isclose(a, b, rel_tol=1e-3, abs_tol=1e-12)
    return a == b


def coerce_like(value: Any, ref: Any) -> Any:
    """``value`` in the type of ``ref`` (the carrier config's value of the same key). A bool ref takes a bool (or
    0 / 1); an int ref an integral number, rounded half away from zero (``5.6 -> 6``); a float ref any number (as
    float); a non-empty list / tuple ref a list of the same length, element-wise. A str or None ref (YAML reads
    ``1e-4`` as a str) or any other type: ``value`` unchanged. ValueError on a value the ref's type can't take (a
    bool for a number, a non-finite number for an int, a str for a number, a list of another length)."""
    if isinstance(ref, bool):
        if isinstance(value, bool):
            return value
        if _num(value) and value in (0, 1):
            return bool(value)
        raise ValueError(f"{value!r} is not a bool")
    if isinstance(ref, int):
        if not _num(value) or not math.isfinite(value):
            raise ValueError(f"{value!r} is not an int")
        return int(math.copysign(math.floor(abs(value) + 0.5), value))
    if isinstance(ref, float):
        if not _num(value):
            raise ValueError(f"{value!r} is not a number")
        return float(value)
    if isinstance(ref, (list, tuple)) and ref:
        if not isinstance(value, (list, tuple)) or len(value) != len(ref):
            raise ValueError(f"{value!r} is not a list of {len(ref)}")
        return [coerce_like(v, r) for v, r in zip(value, ref)]
    return value


def check_knobs(bank: Sequence[BankKnob], carrier_cfg: Mapping[str, Any]
                ) -> tuple[list[BankKnob], list[tuple[BankKnob, str]]]:
    """(kept, [(dropped, why)]) of ``bank`` on a carrier whose flattened config is ``carrier_cfg``: a ``diff`` knob
    (a node's diff, not a config key) is kept as it is; a config knob whose key the carrier lacks is dropped (10-13:
    w007 built 0 of 16 points -- every point wrote the absent ``dino.schedule_ref_frac``); its levels are coerced to
    the carrier value's type (:func:`coerce_like`: an int key stays int, a bool a bool), and one the carrier's type
    can't take, or whose levels collapse into one (``2.4 -> 2.6`` on an int key), is dropped; two knobs that become
    one signature keep the first. An empty (unreadable) config checks nothing."""
    if not carrier_cfg:
        return list(bank), []
    kept: list[BankKnob] = []
    dropped: list[tuple[BankKnob, str]] = []
    seen: set[str] = set()
    for b in bank:
        kb = b.knob
        if kb.scale != "diff":
            if kb.key not in carrier_cfg:
                dropped.append((b, f"the carrier config has no {kb.key}"))
                continue
            ref = carrier_cfg[kb.key]
            try:
                lo, hi = coerce_like(kb.lo, ref), coerce_like(kb.hi, ref)
            except ValueError as exc:
                dropped.append((b, f"{kb.key} is {type(ref).__name__} on the carrier: {exc}"))
                continue
            if lo == hi:
                dropped.append((b, f"{kb.key} levels {json.dumps(kb.lo)}->{json.dumps(kb.hi)} collapse to "
                                   f"{json.dumps(lo)} as {type(ref).__name__}"))
                continue
            scale = kb.scale if kb.scale != "log" or (_num(lo) and _num(hi) and lo > 0 and hi > 0) else "lin"
            if json.dumps([lo, hi, scale]) != json.dumps([kb.lo, kb.hi, kb.scale]):
                b = BankKnob(Knob(kb.key, lo, hi, scale), b.m, b.v, b.source, b.why, b.z)
        if (sig := b.signature()) in seen:
            dropped.append((b, f"{kb.key}: the same knob as another once coerced"))
            continue
        seen.add(sig)
        kept.append(b)
    return kept, dropped


def retro_bank(effects: Sequence[RetroEffect], lineage: Sequence[str], carrier_cfg: Mapping[str, Any], tau2: float,
               *, z: float = Z_EDGE, nu: float | None = M.NU, w: float = 1.0) -> tuple[list[BankKnob], list[str]]:
    """(knobs, derivation lines) for a wave on a carrier whose old-epoch lineage (its original, then its ancestors)
    is ``lineage``: the effects measured on a lineage carrier, pooled per (key, lo, hi) by inverse variance; those with
    |z| >= ``z`` become knobs (see the module docstring), each with its shrunk effect (Student-t at 0, ``tau2``) as
    the prior -- an edge-extended range has prior mean 0 (no evidence there yet). Best prior first. ``w`` < 1:
    evidence from an older base epoch, shrunk with its variance / w (docs §8: old evidence is a discounted prior)."""
    pooled: dict[tuple, list[RetroEffect]] = {}
    for e in effects:
        if e.carrier in lineage:
            pooled.setdefault((e.knob.key, json.dumps(e.knob.lo), json.dumps(e.knob.hi), e.knob.scale), []).append(e)
    out, lines = [], []
    for (key, _, _, _), es in pooled.items():
        kb, wsum = es[0].knob, sum(1 / e.se ** 2 for e in es)
        beta, se = sum(e.beta / e.se ** 2 for e in es) / wsum, math.sqrt(1 / wsum)
        src = ", ".join(f"{e.wave} {e.beta:+.4f}" for e in es)
        if abs(beta / se) < z:
            continue
        m, v = M.posterior(0.0, tau2, M.Estimate(beta, se * se / w, sum(e.n for e in es), 0), nu)
        head = f"{key} {json.dumps(kb.lo)}->{json.dumps(kb.hi)}: pooled {beta:+.4f} (z {beta / se:+.1f}; {src})"
        if kb.scale == "diff":
            if beta > 0:
                out.append(BankKnob(kb, m, v, "retro", f"{head}: leg on", abs(beta / se)))
                lines.append(f"{head} -> diff knob {key}, prior {m:+.4f}")
            else:
                lines.append(f"{head} -> dropped (harmful leg)")
            continue
        better, worse = (kb.hi, kb.lo) if beta > 0 else (kb.lo, kb.hi)
        c = carrier_cfg.get(key, MISSING_KEY)
        if c is MISSING_KEY:
            lines.append(f"{head} -> dropped (the carrier has no {key})")
        elif _same(c, worse):
            out.append(BankKnob(Knob(key, worse, better, kb.scale), abs(m), v, "retro", f"{head}: carrier at {c}",
                                abs(beta / se)))
            lines.append(f"{head} -> knob {key} {json.dumps(worse)}->{json.dumps(better)} (carrier at the worse level), "
                         f"prior {abs(m):+.4f}")
        elif _same(c, better) and not _num(c):
            lines.append(f"{head} -> dropped (carrier holds the better level {json.dumps(c)}; no step out of a "
                         f"non-numeric level)")
        elif _same(c, better):
            nk = extend(Knob(key, worse, better, kb.scale), Effect(key, abs(beta), se, abs(m), v), z=z)
            if nk is None or (nk.lo, nk.hi) == (worse, better):
                lines.append(f"{head} -> dropped (no step out of {key})")
                continue
            out.append(BankKnob(nk, 0.0, tau2, "retro", f"{head}: carrier at the better edge {c}, one step out",
                                abs(beta / se)))
            lines.append(f"{head} -> knob {key} {nk.lo}->{nk.hi:.4g} (carrier holds the better edge: one step out), "
                         f"prior 0")
        else:
            lines.append(f"{head} -> dropped (carrier at {json.dumps(c)}: off the tested range)")
    out.sort(key=lambda b: (-b.m, b.knob.key))
    return out, lines


MISSING_KEY = object()


def choose(bank: Sequence[BankKnob], k: int = MAX_K) -> list[BankKnob]:
    """The first ``k`` valid knobs of the bank (in its order), one per key."""
    out: dict[str, BankKnob] = {}
    for b in bank:
        if valid(b.knob) and b.knob.key not in out and len(out) < k:
            out[b.knob.key] = b
    return list(out.values())


#: the retro wave: only effects the retro table measured at |z| >= RETRO_Z, at most RETRO_K of them (the largest
#: |z| first) -- an 8-point wave. On the 10-08 copy the 7-knob retro bank priced x* at +.0103 (Σβ of old-epoch
#: effects); what a factorial x* actually delivers is the carrier x factorial cell prior (acquire prices it so).
RETRO_Z = 3.0
RETRO_K = 4


def retro_trim(bank: Sequence[BankKnob], z: float = RETRO_Z, k: int = RETRO_K) -> list[BankKnob]:
    """The retro knobs measured at |z| >= ``z``, the ``k`` largest |z| (then the best prior) first."""
    keep = sorted((b for b in bank if b.z >= z), key=lambda b: (-b.z, -b.m, b.knob.key))
    return choose(keep, k)


def unused(bank: Sequence[BankKnob], used: Iterable[str]) -> list[BankKnob]:
    """The bank without the knobs a wave on this carrier already consumed (:meth:`BankKnob.signature`)."""
    done = set(used)
    return [b for b in bank if b.signature() not in done]


def arrange(knobs: Sequence[BankKnob]) -> list[BankKnob]:
    """Column order for :func:`design`. A 16-point wave's blocks split on AB (alias chain AB = CE = FG), so the two
    highest-prior knobs go to columns A and C: their interaction AC (= BE = DG) is never confounded with the block
    (with the bank's order they were A and B: exactly the block contrast). 8-point waves are one block: unchanged."""
    ks = list(knobs)
    if len(ks) <= 4:
        return ks
    top = sorted(range(len(ks)), key=lambda i: (-abs(ks[i].m), i))[:2]
    out = [ks[i] for i in range(len(ks)) if i not in top]
    out.insert(0, ks[top[0]])
    out.insert(2, ks[top[1]])
    return out


def forecast(knobs: Sequence[BankKnob], s2: float, *, z: float = Z_NEXT) -> tuple[float, float, float, int]:
    """(m, v, V, N) of the wave's x*: N points; knob j's main effect is observed with ``V_j = 4 s² / N`` (s² the
    model's (1−ρ)σ²) and x* takes it when ``β̂_j > z·√V_j``, so its expected gain is
    ``Σ_j E[β_j 1(β̂_j > z√V_j)] = Σ_j m_j Φ(a_j) + v_j/S_j φ(a_j)``, ``S_j = √(v_j + V_j)``, ``a_j = (m_j − z√V_j)/S_j``;
    v = Σ v_j, V = Σ V_j (what the acquisition's EI is computed with)."""
    n = 8 if len(knobs) <= 4 else 16
    big_v = 4 * s2 / n
    return sum(knob_gains(knobs, s2, z=z)), sum(b.v for b in knobs), big_v * len(knobs), n


def knob_gains(knobs: Sequence[BankKnob], s2: float, *, z: float = Z_NEXT) -> list[float]:
    """Each knob's term ``E[β_j 1(β̂_j > z√V_j)]`` of :func:`forecast` (acquire combines the ``diff`` knobs' terms
    with its stacking rule instead of summing them)."""
    n = 8 if len(knobs) <= 4 else 16
    big_v = 4 * s2 / n
    out = []
    for b in knobs:
        sj = math.sqrt(b.v + big_v)
        a = (b.m - z * math.sqrt(big_v)) / sj
        out.append(b.m * M._Phi(a) + b.v / sj * M._phi(a))
    return out


#: a run status that ends a wave point (done / crash; a void -- a cancel, the idle churn's drop, an infra void --
#: too: 10-13, five waves whose churned points were voided counted open forever and starved new waves)
FINISHED = frozenset({"done", "crash", "void"})


def open_waves(data: Ledger, closed: Iterable[str] = ()) -> dict[str, str]:
    """wave id -> carrier of every wave with a point that has no finished (:data:`FINISHED`) run yet; a wave in
    ``closed`` (its ``wave:<id>:result`` is recorded) is never open."""
    finished = {r.node for r in data.runs if r.status in FINISHED}
    done = set(closed)
    out = {}
    for n in data.nodes.values():
        if (t := _TAG.search(n.notes)) and n.id not in finished and t.group(1) not in done:
            out[t.group(1)] = n.parent or ""
    return out
