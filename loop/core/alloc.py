"""loop.core.alloc — v3-c7i: where the research budget goes, priced in measured yield per GPU-h.

Measured over the 24 h before 10-14: the LLM-idea lanes (explore / literature / repair / program / plateau) used ~410
GPU-h (~75%) and produced ~1 of 49 measured nodes at >= +.002; factorial waves and code-picked crossovers used 17% and
produced 13 of 15; 163 nodes sat QUEUED (median age 15.7 h); the kind bandit was flat (.19-.21: exp((yield - top) / sd)
with sd inflated by a few -.028 nodes); code-made nodes were credited to no kind; program_share .25 was a hard quota
for the worst kind. Everything here is computed from the ledger; nothing names a lane, a mechanism or a node.

- UNITS          a brief-tagged node (``brief=<kind>``) and its untagged descendants (dose ladders, LOO children,
                 tuned variants: what the brief led to); a WAVE (``wave= / factorial_of= / for_wave=``: its points and
                 x*); a code-made PAIR (``pair_of=``). A unit's value is its best path value (the sum of own
                 increments from the unit's first node; the first node counts with >= 1 seed, a descendant with >=
                 ``ideation.clade_min_seeds``); its cost the non-test GPU-h of its nodes.
- :func:`kind_rates`  I1: kind -> yield per GPU-h = sum max(0, unit value) / GPU-h, shrunk to the pooled rate with
                 ``alloc.rate_n0`` pseudo-units of the pooled mean cost; w = softmax((rate - best) / T), T = 1.4826 x
                 MAD of the measured kinds' rates (>= ``temp_floor_frac`` x the pooled rate): a few very negative nodes
                 no longer flatten it (a unit's gain is floored at 0 and its spread is robust).
- :func:`program_share`  the program quota is a CEILING that decays toward ``alloc.program_share_floor`` while the
                 program kind's rate is below the pooled rate.
- :func:`gpu_split`  I2: the GPU share LLM-idea units' runs may hold (soft cap in ``acquire.rank``):
                 rate_llm / (rate_llm + rate_code), clamped to [``llm_gpu_floor``, ``llm_gpu_cap``].
- :func:`backlog` / :func:`throttle` / :func:`age_out`  I2: no new brief while the QUEUED backlog is too big / too
                 old; aged QUEUED nodes outside the best ``backlog_keep`` by screen value / GPU-h are dropped (REJECTED
                 'dropped_backlog', logged), never deleted.
- :func:`clusters` / :func:`cluster_rule` / :func:`cluster_text`  I3: mechanism clusters of this epoch's research
                 nodes with a shrunk posterior mean; saturated-negative ones capped, under-visited positive ones boosted.
- :func:`dedup_text`  I4: the dead / in-flight mechanisms nearest a brief's focus (what check_idea rejects).
- :func:`graft_pairs` / :func:`flagged`  I5: a code-picked pair is first GRAFTED by code (both legs' own changes on the
                 incumbent); only a CONFLICTING graft is flagged for the LLM crossover brief.
- :func:`fnf_pause`  I6: no new fork-and-flush round once enough closed rounds show no positive fresh-seed effect.
- :func:`early_kills`  I9: a declared diagnostic that REVERSED after the first seed, with a negative paired first-seed
                 delta, ends that node's screening."""

from __future__ import annotations

import hashlib
import math
import re
import statistics
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Collection, Mapping, Sequence

#: code-made units: what a ``pair_of=`` (a code-picked crossover) / a wave point / x* / wave port is credited to
CODE_KINDS = ("wave", "pair")
_PAIR = re.compile(r"\bpair_of=")
_WAVE = re.compile(r"\b(?:wave|factorial_of|for_wave)=([^\s,]+)")
_BRIEF = re.compile(r"\bbrief=(\S+)")
_REPAIR_OF = re.compile(r"\brepair_of=")
_PREREQ = re.compile(r"\bprereq_of=(\S+)")
_SPLIT = re.compile(r"\bsplit_of=(\S+)")             # v3-c8a B3: a part of a mixed-change node
_OPP_ARM = re.compile(r"\bop=(\S+) arm=(\S+?)(?:\s|$).*\bopposite_of=(\S+)")   # v3-c8a B1: an eval-only opposite arm
#: the unit kind of a c7a probe-ill-conditioning repair (``repair_of=``): neither an LLM brief kind nor code-made, so
#: the kind bandit and the GPU split leave it out (v3 review K1: it inflated the repair brief's yield)
PROBE_REPAIR = "probe_repair"
TEST_STAGES = frozenset({"confirm", "remint", "audit"})
K_RATES = "alloc.kind_rates"          # the last kind table + split (logged when it changes)
K_FLAGGED = "alloc.xover_flagged"     # {pair key: {a, b, why, ts}}: grafts that conflicted (the LLM brief's cases)
K_XOVER_WAIT = "alloc.xover_wait"       # {text, ts}: the 'waits' line last shown (quiet)
K_GRAFTED = "alloc.xover_grafted"     # {pair key: node | "-: why"}: code grafts tried
K_EARLY = "alloc.early_kill"          # {node: {z, delta, ts}}: screening ended early (I9)
K_THROTTLE = "alloc.throttle"         # the last throttle verdict ('' = none), logged on change
K_FNF_PAUSE = "alloc.fnf_pause"       # the last fork-and-flush pause verdict, logged on change
GRAFT_TAG = "graft_xover"
MAD_K = 1.4826
OLD_W = 0.5                           # an older base's measurement in a cluster posterior (acquire.OLD_EPOCH_W)


def cfg_of(ctx: Any) -> Any:
    from loop.spec import AllocCfg
    return getattr(getattr(ctx, "cfg", None), "alloc", None) or AllocCfg()


def llm_kinds() -> tuple[str, ...]:
    from loop.core.ideation import KINDS
    return tuple(KINDS)


# ---------------------------------------------------------------------------------------------- units
def code_kind(notes: str) -> tuple[str, str]:
    """(kind, unit id) of a code-made node ('', '' otherwise): a code-picked pair, a wave's point / x* / port."""
    notes = notes or ""
    if _PAIR.search(notes):
        return "pair", ""
    if m := _WAVE.search(notes):
        return "wave", f"wave:{m.group(1)}"
    return "", ""


def brief_kind(notes: str) -> str:
    """The node's ``brief=<kind>`` (``ideation.BRIEF_ALIAS`` applied: a c7a probe repair, ``brief=repair_probe`` on
    nodes made before the merge, is a repair)."""
    from loop.core.ideation import BRIEF_ALIAS
    m = _BRIEF.search(notes or "")
    return BRIEF_ALIAS.get(m.group(1), m.group(1)) if m else ""


def _opposite_origin(led: Any, node: Any) -> str:
    """The node of the arm an eval-only OPPOSITE arm mirrors (same program ``op=``, ``arm=`` = its ``opposite_of=``)
    -- the opposite arm is the origin's unit, not a new LLM idea ('' otherwise)."""
    m = _OPP_ARM.search(node.notes or "")
    if not m:
        return ""
    pid, _, aid = m.groups()
    for nid, o in led.nodes.items():
        if nid != node.id and f"op={pid} " in (o.notes or "") and re.search(rf"\barm={re.escape(aid)}(?:\s|$)", o.notes or ""):
            return nid
    return ""


def unit_of(led: Any, nid: str, memo: dict[str, tuple[str, str]] | None = None) -> tuple[str, str]:
    """(unit id, kind) of ``nid``: its nearest ancestor-or-self that starts a unit -- a code-made node (its wave, or the
    pair node itself) or a brief-tagged node (kind = its brief, 'other' for a non-brief source such as a port or a
    data-bank session); ('', '') when none (a base root, an untagged hand-made node)."""
    memo = {} if memo is None else memo
    kinds = set(llm_kinds())
    path, cur, got = [], nid, ("", "")
    while cur is not None and cur in led.nodes:
        if cur in memo:
            got = memo[cur]
            break
        path.append(cur)
        node = led.nodes[cur]
        ck, uid = code_kind(node.notes)
        if ck:
            got = (uid or cur, ck)
            break
        origin = getattr(node, "port_of", "") or (m.group(1) if (m := _PREREQ.search(node.notes or "")) else "") \
            or (m.group(1) if (m := _SPLIT.search(node.notes or "")) else "") or _opposite_origin(led, node)
        if origin and origin in led.nodes and origin not in path:    # v3 review K2: a port (a prerequisite port)
            if (o := unit_of(led, origin, memo))[0]:                     # is credited to its ORIGIN's unit
                got = o
                break
        if bk := brief_kind(node.notes):
            # v3 review K1: a probe-ill-conditioning repair (repair_of=) is its own kind, never the repair brief's
            got = (cur, PROBE_REPAIR if _REPAIR_OF.search(node.notes or "") else bk if bk in kinds else "other")
            break
        cur = node.parent
        if cur in path:
            break
    for p in path:
        memo[p] = got
    return got


def group_of(kind: str) -> str:
    """'llm' (an LLM-idea kind), 'code' (a wave / code-made pair) or 'other'."""
    return "code" if kind in CODE_KINDS else "llm" if kind in llm_kinds() else "other"


def groups(led: Any) -> dict[str, str]:
    """node -> its unit's group (this epoch's nodes; :func:`group_of`)."""
    memo: dict[str, tuple[str, str]] = {}
    epoch = getattr(led, "epoch", "") or ""
    return {nid: group_of(unit_of(led, nid, memo)[1]) for nid, n in led.nodes.items()
            if not epoch or n.base == epoch}


def _gpu(led: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    for r in led.runs:
        if r.stage not in TEST_STAGES:
            out[r.node] = out.get(r.node, 0.0) + float(r.gpu_h or 0.0)
    return out


def _design_point(node: Any) -> bool:
    """A wave point / a diff knob ported for a wave (``acquire.wave_node``): one shared seed per block, so its own
    increment is a correlated draw of the design (10-13: w013.p03 "best" .6858 = 4 replicates at one lucky seed), never
    a unit value -- like ``bar.ranked`` (v3-c7a A1), a wave is credited through x* (``factorial_of=``) and what is
    built on it; the points still count in the unit's nodes and GPU-h."""
    from loop.core.acquire import wave_node
    return wave_node(node)


def expected_max(k: int) -> float:
    """E[max of k iid N(0, 1)] (Blom's approximation; 0 for k <= 1)."""
    from statistics import NormalDist
    return 0.0 if k <= 1 else NormalDist().inv_cdf((k - 0.375) / (k + 0.25))


def units(led: Any, beliefs: Mapping[str, tuple[float, int]], min_seeds: int = 2,
          gains: Mapping[str, float] | None = None, xstar_share: float = 0.5,
          noise_sd: float = 0.0) -> dict[str, dict[str, Any]]:
    """unit id -> {kind, best (None: nothing measured yet), node (the best), gpu_h, nodes, credit} over this epoch's
    nodes. ``beliefs``: node -> (own-increment mean, seeds); the best node is the one with the best path value (the
    sum of own increments from the unit's first node). ``gains`` (v3 review K1/K2: node -> its shrunk paired gain over
    the INCUMBENT, ``bar.paired``): the unit's value is then max(0, its best node's gain over the incumbent) -- a cheap
    unit no longer buys yield with an own increment over a weak parent; a best node that does not pair with the
    incumbent is worth 0 -- and a wave's x* value is SPLIT with its carrier's unit (``xstar_share`` stays with the wave,
    the rest is ``credit`` of the unit the carrier belongs to)."""
    epoch = getattr(led, "epoch", "") or ""
    memo: dict[str, tuple[str, str]] = {}
    gpu = _gpu(led)
    val: dict[str, float | None] = {}

    def value(nid: str, uid: str) -> float | None:          # the path value from the unit's first node
        if nid in val:
            return val[nid]
        node, b = led.nodes[nid], beliefs.get(nid)
        if _design_point(node):          # merged c7a x c7i: a point's raw increment is a draw of the design
            b = None
        par = node.parent
        above = 0.0
        if par is not None and par in led.nodes and unit_of(led, par, memo)[0] == uid and \
                (not epoch or led.nodes[par].base == epoch):
            above = value(par, uid) or 0.0
        val[nid] = None if b is None else above + float(b[0])
        return val[nid]

    out: dict[str, dict[str, Any]] = {}
    for nid, node in led.nodes.items():
        if node.parent is None or (epoch and node.base != epoch):
            continue
        uid, kind = unit_of(led, nid, memo)
        if not uid:
            continue
        u = out.setdefault(uid, {"kind": kind, "best": None, "node": "", "gpu_h": 0.0, "nodes": 0, "credit": 0.0})
        u["nodes"] += 1
        u["gpu_h"] += gpu.get(nid, 0.0)
        b = beliefs.get(nid)
        if b is None or _design_point(node):
            continue
        first = node.parent not in led.nodes or unit_of(led, node.parent, memo)[0] != uid
        if int(b[1]) < (1 if first else min_seeds):
            continue
        x = value(nid, uid)
        if x is not None and (u["best"] is None or x > u["best"]):
            u["best"], u["node"] = x, nid
    if noise_sd > 0:                                         # v3 review S5: the best of k members is biased up by ~E[max_k] sd
        for uid, u in out.items():
            if u["best"] is None:
                continue
            members = [(beliefs[n][1]) for n, nd in led.nodes.items() if n in beliefs and not _design_point(nd)
                       and unit_of(led, n, memo)[0] == uid and (not epoch or nd.base == epoch)]
            k = len(members)
            if k > 1:
                sd = noise_sd / math.sqrt(max(1.0, sum(members) / k))
                u["raw_best"], u["k"], u["debias"] = u["best"], k, expected_max(k) * sd
                u["best"] = u["best"] - u["debias"]
    if gains is None:
        return out
    for u in out.values():                                   # the value over the INCUMBENT
        if u["best"] is not None:
            u["own"] = u["best"]
            u["best"] = max(0.0, float(gains.get(u["node"], 0.0)) - float(u.get("debias", 0.0)))
    for uid, u in list(out.items()):                         # x* split with the carrier's unit
        if u["kind"] != "wave" or not u["best"] or u["node"] not in led.nodes:
            continue
        car = led.nodes[u["node"]].parent
        cu = unit_of(led, car, memo)[0] if car in led.nodes else ""
        if not cu or cu == uid or cu not in out:
            continue
        give = (1.0 - xstar_share) * u["best"]
        u["best"] -= give
        out[cu]["credit"] = out[cu].get("credit", 0.0) + give
    return out


def unit_value(u: Mapping[str, Any]) -> float | None:
    """A unit's value: max(0, best) plus any credit given to it (an x* built on it); None: nothing measured."""
    if u.get("best") is None and not u.get("credit"):
        return None
    return max(0.0, float(u.get("best") or 0.0)) + float(u.get("credit") or 0.0)


# ---------------------------------------------------------------------------------------------- I1 the kind bandit
def mad_sd(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    med = statistics.median(xs)
    return MAD_K * statistics.median(abs(x - med) for x in xs)


THOMPSON_DRAWS = 400     # alloc: bootstrap draws behind the kind weights / the GPU split's interval


def _rng(key: Any) -> "random.Random":
    """A deterministic generator per ``key`` (the tick's view token): the same ledger gives the same weights."""
    import random
    return random.Random(int(hashlib.sha256(repr(key).encode()).hexdigest()[:16], 16))


def kind_rates(led: Any, beliefs: Mapping[str, tuple[float, int]], *, n0: float = 3.0, min_seeds: int = 2,
               temp_floor_frac: float = 0.25, kinds: Sequence[str] | None = None,
               gains: Mapping[str, float] | None = None, seed: Any = 0, draws: int = THOMPSON_DRAWS,
               noise_sd: float = 0.0) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    """(kind -> {n, units, gain, gpu_h, mean, rate, yield, w}, pooled {...}) -- v3 review K1: THOMPSON SAMPLING
    replaces the softmax (its T 3.4e-6 sat ~6x under the rates' bootstrap SE 1.9-2.4e-5: the winner flipped between
    snapshots 15 min apart). Per kind, each draw resamples its units (with replacement) and adds ``n0`` pseudo-units
    at the POOLED per-unit value and cost: rate = (sum v + n0 G0) / (sum h + n0 H0); ``w`` = P(the kind's rate is the
    largest) over ``draws`` draws, seeded by ``seed`` (deterministic per tick). A unit's value v: :func:`unit_value`
    (with ``gains``: over the incumbent). ``rate`` is the point estimate (the full sample); ``pooled["draws"]``
    keeps every draw (:func:`thompson` re-weights any subset of kinds); ``temp_floor_frac`` is unused (kept for old
    configs)."""
    del temp_floor_frac
    kinds = tuple(kinds) if kinds is not None else (*llm_kinds(), *CODE_KINDS)
    us = units(led, beliefs, min_seeds, gains, noise_sd=noise_sd)
    agg = {k: {"n": 0, "units": 0, "gain": 0.0, "gpu_h": 0.0, "vals": [], "rows": []} for k in kinds}
    for u in us.values():
        if u["kind"] not in agg:
            continue
        a = agg[u["kind"]]
        a["units"] += 1
        a["gpu_h"] += u["gpu_h"]
        v = unit_value(u)
        if v is not None:
            a["n"] += 1
            a["gain"] += v
            a["vals"].append(v)
        a["rows"].append((v or 0.0, float(u["gpu_h"])))
    G = sum(a["gain"] for a in agg.values())
    H = sum(a["gpu_h"] for a in agg.values())
    U = sum(a["units"] for a in agg.values())
    pooled = G / H if H > 0 else 0.0
    g0, h0 = (G / U, H / U) if U else (0.0, 0.0)
    tab: dict[str, dict[str, float]] = {}
    for k, a in agg.items():
        den = a["gpu_h"] + n0 * h0
        rate = (a["gain"] + n0 * g0) / den if den > 0 else pooled
        tab[k] = {"n": a["n"], "units": a["units"], "gain": a["gain"], "gpu_h": a["gpu_h"],
                  "mean": statistics.fmean(a["vals"]) if a["vals"] else 0.0, "rate": rate, "yield": rate}
    rng = _rng(("kind_rates", seed))
    dr: dict[str, list[float]] = {}
    for k, a in agg.items():
        rows, out = a["rows"], []
        for _ in range(max(1, int(draws))):
            if rows:
                pick = [rows[rng.randrange(len(rows))] for _ in rows]
                gs, hs = sum(x for x, _ in pick), sum(h for _, h in pick)
            else:
                gs = hs = 0.0
            den = hs + n0 * h0
            out.append((gs + n0 * g0) / den if den > 0 else pooled)
        dr[k] = out
    w = thompson({"draws": dr}, list(tab))
    for k in tab:
        tab[k]["w"] = w.get(k, 0.0)

    def grp(name: str) -> float:
        rows = [a for k, a in agg.items() if group_of(k) == name]
        g, h = sum(a["gain"] for a in rows), sum(a["gpu_h"] for a in rows)
        return g / h if h > 0 else pooled
    return tab, {"gain": G, "gpu_h": H, "rate": pooled, "units": U, "temp": 0.0, "llm_rate": grp("llm"),
                 "code_rate": grp("code"), "draws": dr, "rows": {k: a["rows"] for k, a in agg.items()},
                 "seed": seed}


def thompson(pooled: Mapping[str, Any], kinds: Sequence[str]) -> dict[str, float]:
    """P(argmax) over ``kinds`` of :func:`kind_rates`' draws (ties shared): the Thompson weights of a subset."""
    dr = pooled.get("draws") or {}
    ks = [k for k in kinds if k in dr]
    if not ks:
        return {}
    n = min(len(dr[k]) for k in ks)
    win = {k: 0.0 for k in ks}
    for i in range(n):
        top = max(dr[k][i] for k in ks)
        best = [k for k in ks if dr[k][i] >= top - 1e-18]
        for k in best:
            win[k] += 1.0 / len(best)
    return {k: x / n for k, x in win.items()} if n else {k: 1.0 / len(ks) for k in ks}


def softmax(tab: Mapping[str, Mapping[str, float]], kinds: Sequence[str], temp: float,
            pooled: Mapping[str, Any] | None = None) -> dict[str, float]:
    """The weights over ``kinds`` only: Thompson (``pooled`` from :func:`kind_rates`), else (no draws) the old
    exp((rate - the best of them) / temp)."""
    if pooled is not None and pooled.get("draws"):
        return thompson(pooled, kinds)
    ks = [k for k in kinds if k in tab]
    if not ks:
        return {}
    top = max(float(tab[k]["rate"]) for k in ks)
    raw = {k: math.exp(max(-60.0, (float(tab[k]["rate"]) - top) / max(temp, 1e-15))) for k in ks}
    tot = sum(raw.values()) or 1.0
    return {k: x / tot for k, x in raw.items()}


def program_share(ceiling: float, floor: float, tab: Mapping[str, Mapping[str, float]],
                  pooled: Mapping[str, float]) -> float:
    """The program quota as a CEILING: ``floor + (ceiling - floor) x min(1, rate_program / the LLM kinds' pooled
    rate)`` (the ceiling while nothing has a positive rate yet)."""
    floor = min(max(0.0, floor), ceiling)
    r, mean = (tab.get("program") or {}).get("rate"), float(pooled.get("llm_rate", pooled.get("rate")) or 0.0)
    if r is None or mean <= 0:
        return ceiling
    return floor + (ceiling - floor) * min(1.0, max(0.0, float(r) / mean))


def gpu_split(tab: Mapping[str, Mapping[str, float]], cap: float, floor: float,
              pooled: Mapping[str, Any] | None = None, ci: float = 0.9) -> dict[str, float]:
    """{llm_rate, code_rate, share, ci_lo, ci_hi}: the GPU share the LLM-idea units may hold. v3 review K2: the share
    stays PINNED at ``cap`` until the bootstrap ``ci`` interval of rl / (rl + rc) (units resampled within each group,
    :func:`kind_rates`' rows: unit values over the incumbent) lies wholly below it; then the point share, >= ``floor``."""
    def rate(group: str) -> tuple[float, float]:
        rows = [r for k, r in tab.items() if group_of(k) == group]
        g, h = sum(r["gain"] for r in rows), sum(r["gpu_h"] for r in rows)
        return (g / h if h > 0 else 0.0), h
    (rl, hl), (rc, hc) = rate("llm"), rate("code")
    floor = min(max(0.0, floor), cap)
    point = 0.5 if rl + rc <= 0 else rl / (rl + rc)
    lo = hi = point
    rows = (pooled or {}).get("rows") or {}
    if rows:
        rng = _rng(("gpu_split", (pooled or {}).get("seed")))
        grp = {g: [x for k, xs in rows.items() if group_of(k) == g for x in xs] for g in ("llm", "code")}
        sh = []
        for _ in range(THOMPSON_DRAWS):
            r = {}
            for g, xs in grp.items():
                pick = [xs[rng.randrange(len(xs))] for _ in xs] if xs else []
                h = sum(y for _, y in pick)
                r[g] = sum(x for x, _ in pick) / h if h > 0 else 0.0
            sh.append(0.5 if r["llm"] + r["code"] <= 0 else r["llm"] / (r["llm"] + r["code"]))
        sh.sort()
        lo, hi = sh[int((1 - ci) / 2 * (len(sh) - 1))], sh[int((1 + ci) / 2 * (len(sh) - 1))]
    share = cap if hi >= cap or rl + rc <= 0 else min(cap, max(floor, point))
    return {"llm_rate": rl, "code_rate": rc, "llm_gpu_h": hl, "code_gpu_h": hc, "share": share,
            "point": point, "ci_lo": lo, "ci_hi": hi}


def inc_gains(led: Any, incumbent: str | None) -> dict[str, float] | None:
    """node -> its shrunk paired gain over ``incumbent`` (``bar.paired``), cached per ledger view; None without a
    pairable incumbent."""
    from loop.core import bar as BR
    key = (id(led), len(getattr(led, "obs", ())), len(getattr(led, "nodes", {})), incumbent)
    if _GAINS.get("key") == key:
        return _GAINS["val"]
    try:
        pv = BR.paired(led, incumbent or None) if hasattr(led, "in_epoch") else None
    except Exception:  # noqa: BLE001 - no paired view: units on their own increments
        pv = None
    val = None if pv is None else {n: float(g[0]) for n, g in pv.gains.items()}
    _GAINS.update(key=key, val=val)
    return val


_GAINS: dict[str, Any] = {}
_CACHE: dict[str, Any] = {}


def table(ctx: Any, led: Any, beliefs: Mapping[str, tuple[float, int]], token: Any = None,
          incumbent: str | None = None) -> tuple[dict[str, dict[str, float]], dict[str, Any], dict[str, float]]:
    """(kind table, pooled, gpu split) with this context's keys, cached on ``token`` (else the ledger's sizes);
    units valued over ``incumbent`` (v3 review K1/K2; else ``bar.incumbent_of``'s)."""
    from loop.core import bar as BR
    from loop.core.levers import opt
    c = cfg_of(ctx)
    base = token if token is not None else (len(led.nodes), len(led.runs), len(led.obs), len(beliefs),
                                            round(sum(m for m, _ in beliefs.values()), 9))
    if incumbent is None and hasattr(led, "incumbents"):
        incumbent = BR.incumbent_of(led)
    key = (base, incumbent, c.rate_n0, c.llm_gpu_cap, c.llm_gpu_floor)
    if _CACHE.get("key") != key:
        gains = inc_gains(led, incumbent) if incumbent else None
        tab, pooled = kind_rates(led, beliefs, n0=float(c.rate_n0), min_seeds=int(opt(ctx, "clade_min_seeds")),
                                 gains=gains, seed=key, noise_sd=float(getattr(c, "unit_noise_sd", 0.0) or 0.0))
        _CACHE.update(key=key, val=(tab, pooled, gpu_split(tab, float(c.llm_gpu_cap), float(c.llm_gpu_floor),
                                                           pooled)))
    return _CACHE["val"]


def post_beliefs(post: Any) -> dict[str, tuple[float, int]]:
    """node -> (mean, seeds) from an ``acquire.Posterior`` / ideation ``View`` (whichever is given), PAIRED ONLY
    (merged with v3-c7a A1/A2): a node none of whose seeds is CUPED-paired with a parent run at that seed is left out
    (its increment is a raw level minus the parent's mean: a seed offset, not a gain -- the same rule ``bar.ranked``
    applies against the incumbent; c7a's per-block-seed wave controls make the carriers' seeds pairable). Wave points
    are left out in :func:`units`."""
    core = post if hasattr(post, "est") else getattr(post, "post", None)
    est = getattr(core, "est", None)

    def paired(n: str) -> bool:
        if not isinstance(est, Mapping):
            return True                  # no CUPED estimates to read (a bare belief map): kept as given
        from loop.core.model import COMPOSITE
        e = (est.get(n) or {}).get(COMPOSITE)
        return e is not None and int(getattr(e, "paired", 0) or 0) > 0
    b = getattr(post, "beliefs", None)
    if callable(b):
        return {n: (float(m), int(k)) for n, (m, _v, k) in b().items() if paired(n)}
    return {n: (float(x.m), int(x.n)) for n, x in (b or {}).items() if paired(n)}


# ---------------------------------------------------------------------------------------------- I2 backlog
def backlog(led: Any, statuses: Mapping[str, Any], now: float,
            screenable: Collection[str] | None = None) -> dict[str, Any]:
    """{n, median_age_h, ages: {node: hours}, n_screenable} of this epoch's QUEUED nodes (derived status). v3
    review O6: with ``screenable`` (the nodes holding a screen option this tick, :func:`screenable`) the median age
    is over THOSE only -- a node that cannot screen (its parent unmeasured, held for a dataset, early-killed) never
    ages the backlog into a throttle the GPU queue cannot clear; ``n`` stays every QUEUED node. 10-22: operator-intended
    nodes (:func:`acquire.program_child`: program arms, ladder doses and their isolation children) are left out of the
    median too -- they are not idea overproduction, they are screened first on purpose, and 8 dose nodes waiting their
    turn held the throttle ~20 h; ``n_program`` counts them."""
    from loop.core.acquire import program_child
    epoch = getattr(led, "epoch", "") or ""
    ages = {}
    for nid, st in statuses.items():
        name = str(getattr(st, "value", st)).lower()
        node = led.nodes.get(nid)
        if name == "queued" and node is not None and (not epoch or node.base == epoch):
            ages[nid] = max(0.0, (now - float(node.created_ts or now)) / 3600.0)
    prog = {n for n in ages if program_child(led.nodes[n])}
    med = [a for n, a in ages.items() if n not in prog and (screenable is None or n in screenable)]
    return {"n": len(ages), "median_age_h": statistics.median(med) if med else 0.0, "ages": ages,
            "n_screenable": len(med), "n_program": len(prog)}


K_TRICKLE_N = "alloc.trickle_n"      # trickle sessions briefed so far (the kind rotation's position)


def trickle_room(c: Any, why: str, in_flight: int) -> int:
    """Research sessions the throttle still lets start: unlimited (-1) when not throttled, else ``backlog_trickle``
    minus the sessions in flight / being briefed (0 under the default hard stop)."""
    if not why:
        return -1
    return max(0, int(getattr(c, "backlog_trickle", 0) or 0) - max(0, int(in_flight)))


def trickle_kind(c: Any, store: Any) -> str:
    """The next trickle brief's kind: ``backlog_trickle_kinds`` in turn (state :data:`K_TRICKLE_N`)."""
    kinds = [str(k) for k in getattr(c, "backlog_trickle_kinds", ()) or ()]
    if not kinds:
        return ""
    n = int(store.get_state(K_TRICKLE_N) or 0)
    store.set_state(K_TRICKLE_N, n + 1)
    return kinds[n % len(kinds)]


def screenable(core: Any) -> set[str] | None:
    """The nodes with a screen option in the core's last ranking (None: no ranking yet -- every QUEUED node)."""
    opts = getattr(core, "last_opts", None)
    if opts is None:
        return None
    return {o.node for o in opts if getattr(o, "kind", "") == "screen" and getattr(o, "node", None)}


THROTTLE_ALERT_H = 6.0      # briefs throttled this long: an ALERT (the GPU queue is not clearing the backlog)
K_THROTTLE_SINCE = "alloc.throttle_since"


def throttle_alert(store: Any, why: str, now: float) -> str:
    """Track how long briefs have been throttled (state ``alloc.throttle_since``); '' or the ALERT text once the
    throttle has held > :data:`THROTTLE_ALERT_H` (re-raised every further hour, logged ``brief_throttle_alert``)."""
    st = dict(store.get_state(K_THROTTLE_SINCE) or {})
    if not why:
        if st:
            store.set_state(K_THROTTLE_SINCE, None)
        return ""
    if st.get("since") is None:
        store.set_state(K_THROTTLE_SINCE, {"since": now, "alerted": 0.0})
        return ""
    held = (now - float(st["since"])) / 3600.0
    if held <= THROTTLE_ALERT_H or now - float(st.get("alerted") or 0.0) < 3600.0:
        return ""
    store.set_state(K_THROTTLE_SINCE, {**st, "alerted": now})
    store.log("brief_throttle_alert", why=why, held_h=round(held, 2))
    return f"ALERT: briefs throttled {held:.1f} h > {THROTTLE_ALERT_H:g} h ({why})"


def throttle(c: Any, bl: Mapping[str, Any]) -> str:
    """Why new briefs wait ('' = none): the backlog above ``backlog_max`` nodes or its median age above
    ``backlog_age_h`` (screen what is queued before asking for more)."""
    if not getattr(c, "enabled", True):
        return ""
    why = []
    n = bl["n"] - (int(bl.get("n_program") or 0)       # operator-intended nodes are not idea overproduction
                   if getattr(c, "backlog_exclude_program", False) else 0)
    if c.backlog_max > 0 and n > c.backlog_max:
        why.append(f"{n} queued > {c.backlog_max}")
    if c.backlog_age_h > 0 and bl["n"] and bl["median_age_h"] > c.backlog_age_h:
        why.append(f"median queued age {bl['median_age_h']:.1f} h > {c.backlog_age_h:g} h")
    return "; ".join(why)


def age_out(c: Any, bl: Mapping[str, Any], rates: Mapping[str, float], exempt: Sequence[str] = ()
            ) -> list[tuple[str, float, float]]:
    """(node, age h, rate) to drop: QUEUED older than ``drop_age_h`` with a screen option (``rates``: node -> value /
    GPU-h of its screen) outside the best ``backlog_keep`` by that rate, lowest rate first, at most ``drop_max``. A
    node without a screen option this tick (its parent unmeasured, held for a dataset) is never dropped."""
    if not getattr(c, "enabled", True) or c.drop_age_h <= 0:
        return []
    ages, ex = bl["ages"], set(exempt)
    ranked = sorted((n for n in ages if n in rates), key=lambda n: -rates[n])
    keep = set(ranked[:max(0, int(c.backlog_keep))])
    out = [(n, ages[n], rates[n]) for n in ranked if n not in keep and n not in ex and ages[n] > c.drop_age_h]
    out.sort(key=lambda x: (x[2], -x[1]))
    return out[:max(0, int(c.drop_max))]


STARVED_REASON = "dropped_backlog: starved_zero_run"      # distinct from drop()'s reason so restore can tell them apart


def starved(c: Any, bl: Mapping[str, Any], nodes: Mapping[str, Any], exempt: Collection[str] = (),
            touched: Mapping[str, float] | None = None, now: float = 0.0) -> list[tuple[str, float]]:
    """(node, age h) of QUEUED nodes (derived: zero runs) older than ``drop_age_h`` that acquisition never picks (live
    10-22: 10 nodes 17-19 h old held the median queue age over the throttle, because :func:`age_out` only drops past
    ``backlog_keep`` and the queue held 30). Exempt: ``exempt`` (open-block / held / prerequisite nodes), program arms,
    waves, ports, program ladder doses, the root. Age = the later of created and ``touched`` (a restored node gets a fresh window)."""
    if not getattr(c, "enabled", True) or c.drop_age_h <= 0:
        return []
    from loop.core.acquire import ROOT, program_child, wave_node
    ex, out = set(exempt), []
    for nid, age in bl["ages"].items():
        node = nodes.get(nid)
        if node is None or nid in ex or nid == ROOT or program_child(node) or wave_node(node) \
                or node.port_of:
            continue
        if touched and now > 0 and float(touched.get(nid) or 0.0) > 0:
            age = min(age, max(0.0, (now - float(touched[nid])) / 3600.0))
        if age > c.drop_age_h:
            out.append((nid, age))
    out.sort(key=lambda x: -x[1])
    return out[:max(0, int(c.drop_max))]


def drop(store: Any, rows: Sequence[tuple[str, float, float]], keep: int, now: float) -> list[str]:
    """Record the age-out: each node REJECTED with reason ``dropped_backlog: ...`` (kept, never deleted) and logged."""
    from loop.contracts import NodeStatus
    acts = []
    for nid, age, rate in rows:
        why = (f"dropped_backlog: queued {age:.0f} h, screen value/GPU-h {rate:.2e} outside the best {keep} of the "
               "backlog (never screened; not a measured failure)")
        store.update_node(nid, status=NodeStatus.REJECTED, reject_reason=why, updated_ts=now)
        store.log("backlog_dropped", node_id=nid, age_h=round(age, 2), rate=rate, keep=keep)
        acts.append(f"backlog: dropped {nid} ({why})")
    return acts


def drop_starved(store: Any, rows: Sequence[tuple[str, float]], drop_age_h: float, now: float) -> list[str]:
    """The same recoverable ``dropped_backlog`` marking as :func:`drop`, reason ``starved_zero_run``; one log line each."""
    import logging
    from loop.contracts import NodeStatus
    acts = []
    for nid, age in rows:
        why = (f"{STARVED_REASON}: queued {age:.0f} h > {drop_age_h:g} h with zero runs, never picked by acquisition "
               "(never screened; not a measured failure)")
        store.update_node(nid, status=NodeStatus.REJECTED, reject_reason=why, updated_ts=now)
        store.log("backlog_dropped", node_id=nid, age_h=round(age, 2), reason="starved_zero_run", keep=None)
        logging.getLogger("loop").info("backlog: dropped %s (starved_zero_run: queued %.0f h, zero runs)", nid, age)
        acts.append(f"backlog: dropped {nid} (starved_zero_run: queued {age:.0f} h, zero runs)")
    return acts


# ---------------------------------------------------------------------------------------------- I3 clusters
@dataclass
class Cluster:
    id: str
    members: list[str]
    centroid: dict[str, float]
    label: str = ""
    category: str = ""
    n: int = 0                       # measured members (>= 1 seed)
    eff: float = 0.0                 # their weight (an older base's at OLD_W)
    mean: float = 0.0
    shrunk: float = 0.0
    best: tuple[str, float] | None = None
    last: str = ""                   # the newest member
    flag: str = ""                   # saturated-negative | boost | ''
    sample: list[str] = field(default_factory=list)
    cur_n: int = 0                   # measured members of the CURRENT base (v3 review K10: what may saturate it)
    protected: str = ""              # why it is never saturated ('': it may be): a TEST pass / an incumbent ancestor
    tags: dict[str, str] = field(default_factory=dict)    # member -> its family + mechanism tag (the [differs from:] check)
    gains: dict[str, float] = field(default_factory=dict)  # measured member -> its own increment (the family block's evidence)
    neg_family: bool = False         # >= family_block_min_n measured, mean <= family_block_mean, none above +sigma, unprotected


@dataclass
class TermFamily:
    """A mechanism FAMILY by a shared content stem of the measured research nodes' family / tag words (``position`` over
    a node rel_crop_position_head, a node rasa_positional_..., a node crop_box_position_...): the lexical clustering
    scatters such a family over singleton clusters (live 10-21: 'position' 9 nodes mean -.0049, but no cluster of them).
    The same evidence test as a cluster's (:func:`negative_family`)."""
    stem: str
    members: list[str] = field(default_factory=list)
    gains: dict[str, float] = field(default_factory=dict)     # measured member -> own increment
    tags: dict[str, str] = field(default_factory=dict)        # member -> "family tag"
    protected: str = ""
    n: int = 0
    mean: float = 0.0
    best: tuple[str, float] | None = None
    neg_family: bool = False
    id: str = ""

    @property
    def label(self) -> str:
        return self.stem

    category: str = "term family"


def _cos(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(x * b.get(w, 0.0) for w, x in a.items())


def _norm(v: Mapping[str, float]) -> dict[str, float]:
    z = math.sqrt(sum(x * x for x in v.values())) or 1.0
    return {w: x / z for w, x in v.items()}


def _terms(family: str, tag: str, text: str) -> Counter:
    """family + tag words (each 1) + SIG_W x the text's words, stemmed like intake's mechanism rule."""
    from loop.novelty import mech_terms
    out = Counter({w: 1.0 for w in mech_terms("", f"{family} {tag}")})
    for w, c in mech_terms("", text).items():
        out[w] = out.get(w, 0.0) + SIG_W * min(c, 2)
    return out


def research_nodes(store: Any, led: Any, epoch_only: bool = False, snodes: Mapping[str, Any] | None = None
                   ) -> list[Any]:
    """The ledger's research-session idea nodes (``fnf.idea_node``: made by a session, not a port / wave port; a
    brief tag; not code-made, not REJECTED; ``epoch_only``: this base's), oldest first (store nodes). A mechanism's
    evidence spans the bases of this ledger."""
    from loop.contracts import NodeStatus
    from loop.core.fnf import idea_node
    epoch = (getattr(led, "epoch", "") or "") if epoch_only else ""
    out = []
    for nid, node in led.nodes.items():
        if (epoch and node.base != epoch) or not brief_kind(node.notes) or not idea_node(node) \
                or code_kind(node.notes)[0]:
            continue
        sn = snodes.get(nid) if snodes is not None else store.get_node(nid)
        if sn is not None and sn.status is not NodeStatus.REJECTED:
            out.append(sn)
    return sorted(out, key=lambda n: (n.created_ts or 0.0, n.id))


#: the signature's weight next to the tag words (the tag names the mechanism; the signature's first line adds context)
SIG_W = 0.5


def node_text(store: Any, sn: Any) -> Counter:
    """The mechanism terms a node is clustered on: its family and tag words, plus SIG_W x its signature's terms."""
    from loop.core import mechanisms as MS
    return _terms(sn.family, sn.mechanism_tag, MS.node_signature(store, sn.id))


#: the clustering of the last research-node set (v3 review O2): recomputed only when that set changes; the
#: measured statistics are re-read every call
_CL: dict[str, Any] = {}


def _structure(ctx: Any, store: Any, led: Any, snodes: Mapping[str, Any] | None = None
               ) -> tuple[list[Any], dict[str, Counter], list[tuple[list[str], dict[str, float]]]]:
    """(research nodes, their docs, [(members, centroid)]) -- cached on the research-node ids."""
    from loop.novelty import _tfidf
    c = cfg_of(ctx)
    nodes = research_nodes(store, led, snodes=snodes)
    key = (id(store), str(getattr(store, "_path", "")), float(c.cluster_tau),
           tuple((n.id, n.family, n.mechanism_tag) for n in nodes))
    if _CL.get("key") == key:
        return _CL["nodes"], _CL["docs"], _CL["struct"]
    docs = {n.id: node_text(store, n) for n in nodes}
    df = Counter(w for t in docs.values() for w in t)
    vec = {k: _tfidf(t, df, len(docs)) for k, t in docs.items()}
    out: list[tuple[list[str], dict[str, float]]] = []
    for n in nodes:
        v = vec[n.id]
        best, bc = None, float(c.cluster_tau)
        for i, (_, cen) in enumerate(out):
            if (sim := _cos(v, cen)) >= bc:
                best, bc = i, sim
        if best is None:
            out.append(([n.id], dict(v)))
        else:
            mem, cen = out[best]
            k = len(mem)
            mem.append(n.id)
            acc = {w: x * k for w, x in cen.items()}
            for w, x in v.items():
                acc[w] = acc.get(w, 0.0) + x
            out[best] = (mem, _norm(acc))
    _CL.update(key=key, nodes=nodes, docs=docs, struct=out, df=df)
    return nodes, docs, out


def clusters(ctx: Any, store: Any, led: Any, beliefs: Mapping[str, tuple[float, int]], *,
             protect: Mapping[str, str] | None = None, snodes: Mapping[str, Any] | None = None) -> list[Cluster]:
    """Leader clustering (centroid cosine >= ``alloc.cluster_tau``, TF-IDF with IDF over the clustered nodes) of the
    ledger's research nodes (:func:`research_nodes`), oldest first (the structure cached while that set is the same);
    each cluster's posterior: the mean own increment of its measured members shrunk to 0 with ``cluster_n0``
    pseudo-nodes. SATURATED-NEGATIVE (v3 review K10) only on CURRENT-base evidence (>= ``cluster_min_n`` / 2 measured
    members of this epoch) and never a cluster holding a ``protect`` node (node -> why: a TEST pass, an ancestor of
    the incumbent). Largest first."""
    from loop.core import mechanisms as MS
    c = cfg_of(ctx)
    nodes, _docs, struct = _structure(ctx, store, led, snodes)
    out = [Cluster(f"C{i + 1}", list(mem), dict(cen)) for i, (mem, cen) in enumerate(struct)]
    by_id = {n.id: n for n in nodes}
    epoch = getattr(led, "epoch", "") or ""
    protect = protect or {}
    from loop.core import struct_tune as STT
    owed = STT.held(ctx, store, led)       # 10-22: a structural node still owed its tuning ladder adds no NEGATIVE evidence
    for cl in out:
        ms = [(m, float(beliefs[m][0])) for m in cl.members if m in beliefs and int(beliefs[m][1]) >= 1
              and m not in owed]       # symmetric: an owed node adds neither negative nor positive evidence
        # an older base's increment counts at OLD_W (acquire's old-epoch discount): the base it was measured on moved
        w = {m: 1.0 if not epoch or led.nodes[m].base == epoch else OLD_W for m, _ in ms}
        wt = sum(w.values())
        cl.n = len(ms)
        cl.cur_n = sum(1 for m, _ in ms if not epoch or led.nodes[m].base == epoch)
        cl.eff = wt
        cl.mean = sum(w[m] * x for m, x in ms) / wt if wt else 0.0
        cl.shrunk = sum(w[m] * x for m, x in ms) / (wt + float(c.cluster_n0)) if ms else 0.0
        cl.best = max(ms, key=lambda x: x[1]) if ms else None
        cl.gains = dict(ms)
        cl.last = cl.members[-1]
        words = Counter(w for m in cl.members for w in re.findall(r"[a-z0-9]+", (by_id[m].mechanism_tag or "").lower())
                        if len(w) >= 3)
        cl.label = "/".join(w for w, _ in words.most_common(3)) or by_id[cl.members[0]].mechanism_tag
        cats = Counter(MS.category(ctx, by_id[m].family, by_id[m].mechanism_tag) for m in cl.members)
        cl.category = cats.most_common(1)[0][0] if cats else ""
        cl.sample = cl.members[-3:]
        cl.tags = {m: f"{by_id[m].family} {by_id[m].mechanism_tag}" for m in cl.sample}
        cl.protected = next((f"{m}: {protect[m]}" for m in cl.members if m in protect), "")
        cl.neg_family = negative_family(ctx, cl)
        if cl.eff >= int(c.cluster_min_n) and cl.shrunk <= 0 and cl.cur_n >= int(c.cluster_min_n) / 2 \
                and not cl.protected:
            cl.flag = "saturated-negative"
        elif cl.shrunk > 0 and cl.eff < int(c.cluster_min_n) and (cl.best or ("", 0.0))[1] > 0:
            cl.flag = "boost"
    out.sort(key=lambda cl: (-len(cl.members), cl.id))
    if getattr(c, "family_block", False):
        _CL["tfam"] = term_families(ctx, nodes, beliefs, protect)      # rebound, never mutated: remember() reads it
    return out


def generic_stem(ctx: Any, w: str) -> bool:
    """``w`` (a 7-character tag stem) is a GENERIC / polysemous word (``alloc.family_block_stop``, prefix match): it names
    no mechanism, so it never makes a term family."""
    stop = tuple(getattr(cfg_of(ctx), "family_block_stop", ()) or ())
    return any(w.startswith(x[:7]) for x in stop if len(x) >= 3)


def term_families(ctx: Any, nodes: Sequence[Any], beliefs: Mapping[str, tuple[float, int]],
                  protect: Mapping[str, str] | None = None) -> dict[str, TermFamily]:
    """stem -> :class:`TermFamily` for every stem (>= 4 characters) shared by >= ``family_block_min_n`` MEASURED research
    nodes' family-less tag words, but not by more than ``family_block_max_share`` of all research nodes (a word every
    idea has names no family). ``neg_family`` set by :func:`negative_family_of`."""
    from loop.novelty import mech_terms
    c = cfg_of(ctx)
    protect = protect or {}
    by: dict[str, list[Any]] = {}
    for n in nodes:
        for w in {w for w in mech_terms("", n.mechanism_tag or "") if len(w) >= 4 and not generic_stem(ctx, w)}:
            by.setdefault(w, []).append(n)
    cap = float(getattr(c, "family_block_max_share", 0.15)) * max(1, len(nodes))
    out: dict[str, TermFamily] = {}
    for w, ns in by.items():
        ms = [(n.id, float(beliefs[n.id][0])) for n in ns if n.id in beliefs and int(beliefs[n.id][1]) >= 1]
        if len(ms) < int(c.family_block_min_n) or len(ns) > cap:
            continue
        tf = TermFamily(w, [n.id for n in ns], dict(ms), {}, id=f"T:{w}", n=len(ms),
                        mean=sum(x for _, x in ms) / len(ms), best=max(ms, key=lambda x: x[1]))
        tf.protected = next((f"{m}: {protect[m]}" for m in tf.members if m in protect), "")
        tf.neg_family = negative_family_of(ctx, tf)
        if tf.neg_family:
            tf.tags = {n.id: f"{n.family} {n.mechanism_tag}" for n in ns}
        out[w] = tf
    return out


def protected(led: Any, incumbent: str, passed: Sequence[str] = ()) -> dict[str, str]:
    """node -> why its cluster is never saturated: a TEST pass (``passed``, or a ``pass`` test_verdict of the
    ledger) or an ancestor of the incumbent (a port's original included)."""
    out = {n: "TEST pass" for n in passed}
    for _, nid, v in getattr(led, "verdicts", ()) or ():
        if v == "pass":
            out.setdefault(nid, "TEST pass")
    cur, seen = incumbent, set()
    while cur and cur in led.nodes and cur not in seen:
        seen.add(cur)
        out.setdefault(cur, f"ancestor of the incumbent {incumbent}")
        if led.nodes[cur].port_of:
            out.setdefault(led.nodes[cur].port_of, f"ancestor of the incumbent {incumbent}")
        cur = led.nodes[cur].parent
    return out


#: the last computed clusters (process-wide: the researcher's check_idea runs on a session thread of this process)
LAST: dict[str, Any] = {"clusters": [], "df": Counter(), "n": 0, "window": [], "families": {}}


def remember(cls: Sequence[Cluster], store: Any, led: Any, window: int) -> None:
    """Keep ``cls`` for check_idea / intake (the clustering's own docs: no second pass over the nodes, v3 review O2)."""
    if _CL.get("nodes") is not None:
        nodes, df, n = _CL["nodes"], _CL["df"], len(_CL["docs"])
    else:
        nodes = research_nodes(store, led)
        docs = [node_text(store, x) for x in nodes]
        df, n = Counter(w for t in docs for w in t), len(docs)
    LAST.update(clusters=list(cls), df=df, n=n, window=[x.id for x in nodes][-max(1, int(window)):],
                families=dict(_CL.get("tfam") or {}))


def assign(text: Counter, cls: Sequence[Cluster] | None = None, tau: float = 0.3) -> Cluster | None:
    """The cluster a mechanism text falls in (centroid cosine >= ``tau``, the nearest), else None."""
    from loop.novelty import _tfidf
    cls = LAST["clusters"] if cls is None else cls
    if not cls:
        return None
    v = _tfidf(text, LAST["df"] or Counter(text), max(1, int(LAST["n"])))
    best = max(cls, key=lambda cl: _cos(v, cl.centroid))
    return best if _cos(v, best.centroid) >= tau else None


def family_sigma(ctx: Any) -> float:
    """The 'positive above +sigma' bar of the family block: ``alloc.family_block_sigma``, else the measured paired sd."""
    c = cfg_of(ctx)
    s = float(getattr(c, "family_block_sigma", 0.0) or 0.0)
    if s > 0:
        return s
    sd = float(getattr(getattr(ctx, "cfg", None), "panel_sd", 0.0) or 0.0)
    return sd if math.isfinite(sd) and sd > 0 else 0.0028


def negative_family(ctx: Any, cl: Any) -> bool:
    """The family-level negative block's evidence test on a cluster (or term family): >= ``family_block_min_n`` measured
    members, mean own gain <= ``family_block_mean``, no member above +sigma, and not protected (a TEST pass /
    incumbent ancestor)."""
    c = cfg_of(ctx)
    return bool(getattr(c, "family_block", False) and c.enabled and not cl.protected
                and cl.n >= int(c.family_block_min_n) and cl.mean <= float(c.family_block_mean)
                and (cl.best is None or cl.best[1] <= family_sigma(ctx)))


negative_family_of = negative_family

DIFFERS_CLUSTER = "[differs from cluster:"


def family_rule(ctx: Any, family: str, tag: str, text: str, claim: str = "") -> tuple[str, str]:
    """('negative_family', fix) when the mechanism falls in a NEGATIVE FAMILY -- a cluster (:func:`negative_family`) or a
    shared tag stem (:class:`TermFamily`, what the lexical clustering misses) -- unless the claim's
    ``[differs from cluster: ...]`` (the finish field ``differs_from_cluster``) names >= ``family_block_cite`` of its
    measured nodes and says how each differs (:func:`mechanisms.explained`, ``family_block_explain_min`` characters).
    Generic stems (``family_block_stop``) make no term family. No share condition: the evidence alone blocks. The worst family (lowest mean) is
    the one asked about."""
    c = cfg_of(ctx)
    if not (c.enabled and c.clusters and getattr(c, "family_block", False)):
        return "", ""
    hits: list[Any] = []
    cl = assign(_terms(family, tag, text), tau=float(c.cluster_tau))
    if cl is not None and cl.neg_family:
        hits.append(cl)
    from loop.novelty import mech_terms
    fams = LAST.get("families") or {}
    for w in {w for w in mech_terms("", tag or "") if len(w) >= 4 and not generic_stem(ctx, w)}:
        if (tf := fams.get(w)) is not None and tf.neg_family:
            hits.append(tf)
    if not hits:
        return "", ""
    from loop.core.mechanisms import explained
    differs = claim.split(DIFFERS_CLUSTER, 1)[1] if DIFFERS_CLUSTER in claim else ""
    cl = min(hits, key=lambda h: h.mean)
    need = max(1, min(int(c.family_block_cite), cl.n))
    named = [m for m in cl.gains if explained(differs, m, int(c.family_block_explain_min))] if differs else []
    if len(named) >= need:
        return "", ""
    worst = sorted(cl.gains.items(), key=lambda kv: kv[1])[:6]
    why = ("; ".join(f"{m} {g:+.4f}" for m, g in worst))
    lack = []
    if len(named) < need:
        lack.append(f"it must name and explain >= {need} of the family's measured nodes (now {len(named)})")
    what = f"cluster {cl.id} ({cl.label}; {cl.category})" if isinstance(cl, Cluster) \
        else f"the '{cl.stem}' family ({len(cl.members)} nodes share that tag word)"
    return "negative_family", (
        f"this mechanism falls in {what}, a NEGATIVE FAMILY: {cl.n} measured nodes, mean own gain {cl.mean:+.4f} "
        f"(<= {float(c.family_block_mean):+.4f}), best {cl.best[0]} {cl.best[1]:+.4f} (not above "
        f"+{family_sigma(ctx):.4f}); worst: {why}. Bring a mechanism from another family, or state the difference: finish "
        f"with hypothesis.differs_from_cluster = \"<node>: how it differs and why that failure does not apply; "
        f"<node>: ...\" -- {'; '.join(lack)}")


def cluster_rule(ctx: Any, family: str, tag: str, text: str, claim: str = "") -> tuple[str, str]:
    """('saturated_cluster', fix) when the mechanism falls in a SATURATED-NEGATIVE cluster that already holds >=
    ``cluster_cap_share`` of the last ``cluster_window`` research nodes, unless the claim's ``[differs from: ...]``
    names AND explains its newest members; else ('', '')."""
    c = cfg_of(ctx)
    if not (c.enabled and c.clusters):
        return "", ""
    if (fam := family_rule(ctx, family, tag, text, claim))[0]:     # 10-21: the evidence alone blocks (no share gate)
        return fam
    cl = assign(_terms(family, tag, text), tau=float(c.cluster_tau))
    if cl is None or cl.flag != "saturated-negative":
        return "", ""
    win = LAST["window"]
    share = sum(1 for m in win if m in set(cl.members)) / len(win) if win else 0.0
    if share < float(c.cluster_cap_share):
        return "", ""
    from loop.core.mechanisms import DIFFERS, explained
    differs = claim.split(DIFFERS, 1)[1] if DIFFERS in claim else ""
    if differs and all(explained(differs, m) for m in cl.sample) and new_words(tag, cl):
        return "", ""                    # v3 review K10: the escape also needs a mechanism word the cited ones lack
    best = f", best {cl.best[0]} {cl.best[1]:+.4f}" if cl.best else ""
    return "saturated_cluster", (
        f"this mechanism falls in cluster {cl.id} ({cl.label}; {cl.category}): {cl.n} measured nodes, mean own gain "
        f"{cl.mean:+.4f} (shrunk {cl.shrunk:+.4f}){best} -- SATURATED-NEGATIVE, and it already holds {share:.0%} of the "
        f"last {len(win)} new ideas (cap {float(c.cluster_cap_share):.0%}). Bring a mechanism from another cluster "
        f"(the brief's cluster table lists the under-visited positive ones), or put [differs from: ...] in the claim "
        f"naming and explaining each of {', '.join(cl.sample)} (and a mechanism_tag with a word none of them has)")


def new_words(tag: str, cl: Cluster) -> list[str]:
    """Content words (stemmed, intake's mechanism terms) of the new mechanism ``tag`` that none of the cluster's cited
    members' family / tag has: a ``[differs from: ...]`` escape must name something the cited ideas do not."""
    from loop.novelty import mech_terms
    mine = set(mech_terms("", tag or ""))
    theirs = {w for t in cl.tags.values() for w in mech_terms("", t)}
    return sorted(w for w in mine - theirs if len(w) >= 3)


def cluster_text(cls: Sequence[Cluster], rows: int = 10) -> str:
    """The brief's compact cluster table (n measured, mean, best) with the saturated / boosted flags."""
    if not cls:
        return ""
    show = list(cls[:max(1, rows)])
    show += [cl for cl in cls if cl.flag == "boost" and cl not in show][:3]
    lines = ["## Mechanism clusters (computed: research nodes by mechanism tag + signature; n measured, mean own "
             "gain, best)", "cluster | category | mechanism words | nodes | n | mean | best | flag", "---|---|---|---|---|---|---|---"]
    for cl in show:
        best = f"{cl.best[0]} {cl.best[1]:+.4f}" if cl.best else "-"
        flag = {"saturated-negative": "SATURATED-NEGATIVE (capped: do not add to it)",
                "boost": "UNDER-VISITED POSITIVE (preferred)"}.get(cl.flag, "")
        if cl.neg_family:
            flag = (flag + "; " if flag else "") + "NEGATIVE FAMILY (rejected unless differs_from_cluster names and explains >= 2 of its nodes)"
        lines.append(f"{cl.id} | {cl.category} | {cl.label} | {len(cl.members)} | {cl.n} | {cl.mean:+.4f} | {best} | {flag}")
    sat = sum(cl.flag == "saturated-negative" for cl in cls)
    boost = [cl.id for cl in cls if cl.flag == "boost"]
    lines.append(f"{len(cls)} clusters; {sat} saturated-negative (a new idea inside one is rejected once the cluster holds "
                 "its capped share of new ideas)" + (f"; under-visited positive: {', '.join(boost[:6])}" if boost else ""))
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------- I4 pre-session dedup
def dedup_rows(ctx: Any, store: Any, focus: str, k: int = 8, now: float | None = None) -> list[tuple[str, float, str]]:
    """(node, cosine, why) of the dead (``novelty.dead``) and live / recent (``novelty.mech_dup_hours``) mechanisms
    nearest the brief's ``focus`` text (TF-IDF over every node's mechanism text, as intake's mechanism rule), top k."""
    from loop.contracts import CLOSED_STATUSES, NodeStatus
    from loop.novelty import dead, mech_vectors, mech_terms, _tfidf
    if k <= 0 or not focus.strip():
        return []
    now = time.time() if now is None else now
    vec, gone = mech_vectors(store), dead(store)
    df = Counter(w for v in vec.values() for w in v)
    me = _tfidf(mech_terms("", focus), df, max(1, len(vec)))
    hours = float(getattr(getattr(getattr(ctx, "cfg", None), "novelty", None), "mech_dup_hours", 48.0))
    out = []
    for n in store.nodes():
        if n.id not in vec or n.status is NodeStatus.REJECTED:
            continue
        live = n.status not in CLOSED_STATUSES and (n.created_ts or 0.0) >= now - 3600 * hours
        if n.id not in gone and not live:
            continue
        out.append((n.id, _cos(me, vec[n.id]), "dead" if n.id in gone else "in flight / queued"))
    out.sort(key=lambda x: -x[1])
    return [x for x in out[:k] if x[1] > 0]


def dedup_text(ctx: Any, store: Any, focus: str, k: int = 8) -> str:
    rows = dedup_rows(ctx, store, focus, k)
    if not rows:
        return ""
    lines = []
    for nid, c, why in rows:
        n = store.get_node(nid)
        lines.append(f"- {nid} {n.family}/{n.mechanism_tag} ({why}, similarity {c:.2f})")
    return ("## Dead / in-flight mechanisms nearest this brief (computed; intake rejects a repeat as dead_mechanism / "
            "near_dup_mechanism)\nCall `check_idea` with your family, mechanism_tag and claim BEFORE editing: it runs "
            "those intake checks in one turn, so a repeat is caught before the work.\n" + "\n".join(lines))


# ---------------------------------------------------------------------------------------------- I5 crossover grafts
def flagged(store: Any) -> dict[str, Any]:
    return dict(store.get_state(K_FLAGGED) or {})


def graft_open(store: Any, statuses: Mapping[str, Any], now: float = 0.0, max_age_h: float = 0.0) -> int:
    """Code-grafted pair nodes still unscreened (QUEUED) or screening. With ``now`` and ``max_age_h`` > 0 a node CREATED
    longer ago than that does not count: it sat behind the backlog and will not screen (live 10-21: 5 grafts queued 2-7
    days kept ``xover_max_open`` full)."""
    n = 0
    for sn in store.nodes():
        if GRAFT_TAG in (sn.notes or "") and str(getattr(statuses.get(sn.id), "value", statuses.get(sn.id))).lower() \
                in ("queued", "screening"):
            if now > 0 and max_age_h > 0 and (sn.created_ts or 0.0) > 0 and now - sn.created_ts > max_age_h * 3600.0:
                continue
            n += 1
    return n


def quiet(store: Any, key: str, text: str, now: float, every_s: float = 3600.0) -> list[str]:
    """``[text]`` when it differs from the last one shown under ``key`` or ``every_s`` passed, else ``[]``: a status line a
    tick repeats unchanged (live 10-21: 'fnf: paused' 3325 lines, 'xover graft waits' 450) is shown once an hour."""
    last = store.get_state(key) or {}
    if last.get("text") == text and now - float(last.get("ts") or 0.0) < every_s:
        return []
    store.set_state(key, {"text": text, "ts": now})
    return [text]


def graft_pairs(ctx: Any, store: Any, adapter: Any, view: Any, statuses: Mapping[str, Any], now: float, *,
                dry: bool = False, new_id: Callable[[], str] | None = None) -> list[str]:
    """Up to ``xover_grafts_per_tick`` of the code planner's briefable pairs (``pairs.next_pair``: code legs) are
    GRAFTED by code -- leg a's own change onto the incumbent, then leg b's onto that (``adapter.graft_port``). Clean:
    a QUEUED code-made pair node (``pair_of=<a>,<b> graft_xover``) and the pair claimed (how 'graft'); a conflict: the
    pair is FLAGGED for the LLM crossover brief (``K_FLAGGED``). ``dry``: what would be tried, nothing written."""
    c = cfg_of(ctx)
    if not (c.enabled and c.xover_conflict_only) or not hasattr(adapter, "graft_port"):
        return []
    from loop.contracts import Node as StoreNode, NodeStatus, Operator, Strategy
    from loop.core import pairs as PR
    inc = store.get_node(view.incumbent)
    if inc is None:
        return []
    acts, tried, flags = [], [], flagged(store)
    book = dict(store.get_state(K_GRAFTED) or {})
    left = max(0, int(c.xover_max_open) - graft_open(store, statuses, now, float(getattr(c, "xover_open_max_age_h", 0.0))))
    for _ in range(max(0, int(c.xover_grafts_per_tick))):
        got = PR.next_pair(view, exclude=tuple(tried) + tuple(flags) + tuple(book))
        if not got:
            break
        a, b = str(got[0]), str(got[1])
        key = PR.pair_key(a, b)
        tried.append(key)
        if left <= 0:
            acts += quiet(store, K_XOVER_WAIT, f"xover graft {a} x {b}: waits ({c.xover_max_open} grafted pairs still unscreened)", now)
            break
        if dry:
            acts.append(f"xover graft: would graft {a} x {b} onto {inc.id}")
            continue
        sha, why = inc.commit, ""
        for leg in (a, b):
            ln = store.get_node(leg)
            par = store.get_node(ln.parent_id) if ln is not None and ln.parent_id else None
            if ln is None or par is None:
                why = f"{leg}: no parent to take its own change from"
                break
            got_sha, how = adapter.graft_port(par.commit, ln.commit, sha, f"xg-{a}-{b}-{leg}")
            if how == "applied" and got_sha:
                sha = got_sha
            elif how != "contained":
                why = f"{leg}: {how}"
                break
        if why or sha == inc.commit:
            why = why or "both legs already in the incumbent"
            if why.split(": ", 1)[-1].startswith("conflict"):
                flags[key] = {"a": a, "b": b, "why": why[:300], "ts": now}
                store.set_state(K_FLAGGED, flags)
                store.log("xover_flagged", a=a, b=b, why=why[:300])
                acts.append(f"xover graft {a} x {b}: CONFLICT ({why[:160]}) -> flagged for the LLM crossover brief")
            else:
                book[key] = f"-: {why[:200]}"
                store.set_state(K_GRAFTED, book)
                acts.append(f"xover graft {a} x {b}: not built ({why[:160]})")
            continue
        if not PR.claim(store, a, b, "graft", "", now):
            continue
        nid = new_id() if new_id is not None else _next_id(store)
        na, nb = store.get_node(a), store.get_node(b)
        fam = next((x.family for x in (na, nb) if x is not None and x.family), "crossover")
        store.put_node(StoreNode(
            id=nid, task_id=inc.task_id, commit=sha, parent_id=inc.id, strategy=Strategy.CROSSOVER,
            operator=Operator.CROSSOVER, family=fam, mechanism_tag=f"pair_{a}_{b}",
            diff_hash=hashlib.sha256(f"{inc.commit}\n{sha}".encode()).hexdigest(), status=NodeStatus.QUEUED,
            second_parent_id=b, created_ts=now, updated_ts=now,
            notes=f"{PR.TAG}{a},{b} {GRAFT_TAG} pairs why={re.sub(r'[^A-Za-z0-9_.+-]+', '_', str(got[2]))[:120]}"))
        book[key] = nid
        store.set_state(K_GRAFTED, book)
        store.log("xover_grafted", node_id=nid, a=a, b=b)
        left -= 1
        acts.append(f"xover graft {a} x {b}: clean -> code-made pair node {nid} (no LLM session)")
    return acts


def _next_id(store: Any) -> str:
    seq = len(store.nodes())
    while store.get_node(f"n{seq:04d}") is not None:
        seq += 1
    return f"n{seq:04d}"


def conflict_pair(view: Any, store: Any, exclude: Sequence[str] = (), tries: int = 24
                  ) -> tuple[str, str, str] | None:
    """The best briefable pair whose code graft CONFLICTED (:func:`graft_pairs` flagged it), else None."""
    from loop.core import pairs as PR
    flags, skip = flagged(store), list(exclude)
    for _ in range(tries):
        got = PR.next_pair(view, exclude=tuple(skip))
        if not got:
            return None
        key = PR.pair_key(str(got[0]), str(got[1]))
        if key in flags:
            why = str(flags[key].get("why") or "")
            return str(got[0]), str(got[1]), f"{got[2]}; its code graft conflicted ({why[:200]})"
        skip.append(key)
    return None


def xover_near_dup(store: Any, since: float = 0.0) -> dict[str, Any]:
    """The LLM crossover briefs' intake outcomes: {sessions, near_dup, rate} (window since ``since`` and all time)."""
    kind = {e.get("request_id"): e.get("brief_kind") for e in store.events("research_started")}
    out = {}
    for name, t0 in (("window", since), ("all", 0.0)):
        rows = [e for e in store.events("intake") if kind.get(e.get("request_id")) == "crossover"
                and float(e.get("ts") or 0.0) >= t0]
        nd = sum(1 for e in rows if str(e.get("reason") or "").startswith("near_dup"))
        out[name] = {"sessions": len(rows), "near_dup": nd, "rate": round(nd / len(rows), 3) if rows else None}
    return out


# ---------------------------------------------------------------------------------------------- I6 fork-and-flush
def fnf_pause(v: Any, st: Mapping[str, Any], min_rounds: int) -> str:
    """Why no new fork-and-flush round opens ('' = it may): >= ``min_rounds`` closed rounds, no branch node with a
    positive fresh-seed test gain, and the arm's test credit per GPU-h not above the shared control's. Re-computed
    every tick: a branch node that later passes a test lifts it."""
    from loop.core import fnf as FN
    rounds = list(st.get("rounds") or ())
    if min_rounds <= 0 or len(rounds) < min_rounds:
        return ""
    ep = st.get("pause_ep") or {}
    if ep.get("granted") and int(ep.get("n", -1)) == len(rounds) and not st.get("open"):
        return ""                      # the expiry's one round (see :func:`fnf_pause_expiry`): it may open
    gain = {j.node: j.verdict.mean for j in v.judged}
    arm = {n for r in rounds for ns in (r.get("nodes") or {}).values() for n in ns}
    if any(gain.get(n, 0.0) > 0 for n in arm):
        return ""
    ab = FN.ab(v, st, n_boot=50)
    if float(ab.get("diff") or 0.0) > 0:
        return ""
    tested = sum(1 for n in arm if n in gain)
    return (f"{len(rounds)} closed rounds, {len(arm)} branch nodes ({tested} tested): no positive fresh-seed test gain "
            f"(arm - control credit/GPU-h {float(ab.get('diff') or 0.0):+.2e})")


GRANT_MAX_DEAD = 2     # consecutive granted episodes with no tested branch node before the expiry stops granting rounds


def _branch_tested(store: Any, after_round: int) -> bool:
    """Did any branch node of a round opened after ``after_round`` run a fresh-seed TEST-stage run?"""
    from loop.core import fnf as FN
    try:
        for n in store.nodes():
            b = FN.branch_of(n)
            if b and int(b.split(".", 1)[0]) > after_round and any(
                    str(getattr(r.stage, "value", r.stage)) in FN.TEST_STAGES for r in store.runs(node_id=n.id)):
                return True
    except Exception:  # noqa: BLE001 - an unreadable view counts as untested
        return False
    return False


def fnf_pause_expiry(store: Any, now: float, max_d: float) -> list[str]:
    """The pause cannot be the end of the A/B (10-22 deadlock): :func:`fnf_pause` judges on fresh-seed TEST gains only
    (``ideation.fnf_screen_credit`` False) and branch nodes are never tested, so with no round running nothing can ever
    lift it, while ``judge_fnf`` keeps saying "keep testing". "No positive test" over zero tested branch nodes is an
    A/B with no power, not an effect of zero. Chosen over crediting the screen posterior: a screen gain is selected (the
    best of K branch nodes, a shared control's seed offset, the inherited-carrier bias), so a pause lifted on it would
    be driven by the winner's curse, whereas an expiry adds no biased evidence: a pause EPISODE (the closed-round count it
    was judged at) ends after ``max_d`` days (``ideation.fnf_pause_max_d``; 0 = never) by granting ONE round, whose
    close either brings test evidence (the verdict moves) or starts a new episode. Called once per tick before
    ideation; the grant is read back by :func:`fnf_pause` (``pause_ep`` in the fnf state). One act on a grant."""
    from loop.core import fnf as FN
    if max_d <= 0:
        return []
    if not store.get_state(K_FNF_PAUSE):
        if (cur := FN.state(store)).get("pause_dead"):         # the pause lifted: the stop-granting streak is over
            store.set_state(FN.K_FNF, {**cur, "pause_dead": 0})
        return []
    st = FN.state(store)
    n = len(st.get("rounds") or ())
    ep = dict(st.get("pause_ep") or {})
    if ep.get("n") != n:                       # a new episode: the pause as first logged at this closed-round count
        if ep.get("granted"):                  # the granted round closed and the pause still stands: did it test anything?
            dead = 0 if _branch_tested(store, int(ep.get("round") or 0)) else int(st.get("pause_dead") or 0) + 1
            st = {**st, "pause_dead": dead}
            if dead >= GRANT_MAX_DEAD:
                store.log("fnf_pause_expiry_stopped", consecutive=dead, rounds=n)
        closed = max((float(r.get("closed") or 0.0) for r in st.get("rounds") or ()), default=0.0)
        since = max([float(e["ts"]) for e in store.events("fnf_paused")] + [closed], default=now)   # never before the last close
        ep = {"n": n, "since": min(since, now), "granted": False}
    if not ep["granted"] and not st.get("open") and now - float(ep["since"]) >= max_d * 86400.0 \
            and int(st.get("pause_dead") or 0) < GRANT_MAX_DEAD:
        ep["granted"] = True
        ep["round"] = int(st.get("round") or 0)     # branches of later rounds are the grant's
        store.log("fnf_pause_expired", rounds=n, since=ep["since"], days=round((now - ep["since"]) / 86400.0, 2))
        line = (f"fnf: pause expired after {(now - ep['since']) / 86400.0:.1f} d with {n} closed rounds and no tested "
                f"branch node; one round may open")
        store.set_state(FN.K_FNF, {**st, "pause_ep": ep})
        return quiet(store, "ideation.fnf_pause_expired_line", line, now)
    if st.get("pause_ep") != ep or st.get("pause_dead") != (store.get_state(FN.K_FNF) or {}).get("pause_dead"):
        store.set_state(FN.K_FNF, {**st, "pause_ep": ep})
    return []


# ---------------------------------------------------------------------------------------------- I9 early kill
def _comp(o: Any, weights: Mapping[str, float]) -> float | None:
    if not all(a in o.axes for a in weights):
        return None
    return sum(float(w) * float(o.axes[a]) for a, w in weights.items())


def _at_seed(led: Any, nid: str, seed: int, weights: Mapping[str, float]) -> float | None:
    """``nid``'s composite at ``seed``: the mean of its non-TEST runs there (a TEST-stage run only when it has no other
    run at that seed; duplicates averaged); None without one."""
    obs = [o for o in led.obs if o.node == nid and o.seed == seed]
    pick = [o for o in obs if o.stage not in TEST_STAGES] or obs
    ys = [y for o in pick if (y := _comp(o, weights)) is not None]
    return sum(ys) / len(ys) if ys else None


def first_seed_delta(led: Any, nid: str, weights: Mapping[str, float], seed: int | None = None) -> float | None:
    """The node's composite minus its parent's on ONE seed (CRN): ``seed`` (the seed its diagnostic was decided on),
    else its first finished screening seed. Each side: the mean of its non-TEST runs at that seed (v3 review K9: a
    parent's TEST-stage control run or a duplicate is no longer taken at random). None without a pair."""
    node = led.nodes.get(nid)
    if node is None or node.parent is None:
        return None
    if seed is None:
        mine = sorted((o for o in led.obs if o.node == nid and o.stage not in TEST_STAGES), key=lambda o: o.ts)
        if not mine:
            return None
        seed = mine[0].seed
    x, y = _at_seed(led, nid, int(seed), weights), _at_seed(led, node.parent, int(seed), weights)
    return None if x is None or y is None else x - y


def paired_deltas(led: Any, nid: str, weights: Mapping[str, float]) -> dict[int, float]:
    """seed -> the node's composite minus its parent's, over every seed both have a non-TEST run on."""
    node = led.nodes.get(nid)
    if node is None or node.parent is None:
        return {}
    seeds = {o.seed for o in led.obs if o.node == nid and o.stage not in TEST_STAGES} & \
        {o.seed for o in led.obs if o.node == node.parent}
    out = {}
    for s in sorted(seeds):
        if (d := first_seed_delta(led, nid, weights, s)) is not None:
            out[s] = d
    return out


#: I9 (v3 review K9): an early kill needs the reversal at z below this AND a first-seed delta below -sd_d
EARLY_Z = -2.0


def early_kills(engs: Sequence[Any], led: Any, weights: Mapping[str, float], done: Mapping[str, Any],
                sd_d: float | None = None) -> list[tuple[str, float, float]]:
    """(node, z, first-seed delta) of the nodes to stop screening: the engagement verdict 'reversed' decided on ONE
    diagnosed seed at z < ``EARLY_Z`` (-2: strongly the other way) AND a paired delta AT THAT SEED below -``sd_d``
    (one paired seed's noise: a negative draw alone is a coin flip); nodes already stopped are skipped."""
    out = []
    for e in engs:
        if e.status != "reversed" or e.n != 1 or e.node in done or e.z is None or float(e.z) >= EARLY_Z:
            continue
        d = first_seed_delta(led, e.node, weights, getattr(e, "seed", None))
        bar = -abs(float(sd_d)) if sd_d is not None and math.isfinite(sd_d) else 0.0
        if d is not None and d < bar:
            out.append((e.node, float(e.z), d))
    return out


def early_lifts(led: Any, weights: Mapping[str, float], done: Mapping[str, Any]) -> list[tuple[str, float, int]]:
    """(node, mean, seeds) of early-killed nodes to resume (v3 review K9): another finished paired seed brought the
    paired mean to >= 0 (>= 2 paired seeds)."""
    out = []
    for nid in done:
        d = paired_deltas(led, nid, weights)
        if len(d) >= 2 and (m := sum(d.values()) / len(d)) >= 0:
            out.append((nid, m, len(d)))
    return out


def predictive(engs: Sequence[Any], beliefs: Mapping[str, tuple[float, int]]) -> dict[str, Any]:
    """Does engagement predict gain? {status: (positive, measured)} over the nodes with a measured verdict and a
    posterior (the meta report's check; the brief states it)."""
    out: dict[str, list[int]] = {}
    for e in engs:
        if e.status in ("engaged", "did not engage", "reversed") and e.node in beliefs:
            r = out.setdefault(e.status, [0, 0])
            r[0] += beliefs[e.node][0] > 0
            r[1] += 1
    return {k: tuple(x) for k, x in out.items()}


# ---------------------------------------------------------------------------------------------- the meta report
def meta_section(ctx: Any, store: Any, led: Any, post: Any, since: float, now: float) -> dict[str, Any]:
    """The meta report's allocation section (logged ``alloc_meta`` by the meta cycle): the kind table (rate, w),
    the effective program share, the GPU split, the backlog + throttle + age-outs, the clusters, the LLM crossover
    briefs' near-dup rate and the grafts, the fork-and-flush pause, early kills, check_idea vs intake mechanism
    rejects."""
    from loop.core.levers import opt
    c = cfg_of(ctx)
    out: dict[str, Any] = {}
    if post is not None:
        tab, pooled, sp = table(ctx, led, post_beliefs(post))
        out["kinds"] = {k: {"n": int(r["n"]), "gpu_h": round(r["gpu_h"], 1), "gain": round(r["gain"], 5),
                            "rate": float(f"{r['rate']:.4g}"), "w": round(r["w"], 4)}
                        for k, r in sorted(tab.items(), key=lambda kv: -kv[1]["rate"])}
        out["temp"] = float(f"{pooled['temp']:.4g}")
        out["draws"] = len(next(iter((pooled.get("draws") or {}).values()), ()))
        out["program_share"] = round(program_share(float(opt(ctx, "program_share")), float(c.program_share_floor),
                                                   tab, pooled), 4)
        out["split"] = {k: (float(f"{x:.4g}") if isinstance(x, float) else x) for k, x in sp.items()}

    def n_ev(name: str, t0: float = 0.0) -> int:
        return sum(1 for e in store.events(name) if float(e.get("ts") or 0.0) >= t0)
    out["backlog"] = {"throttle": store.get_state(K_THROTTLE) or "", "dropped": [n_ev("backlog_dropped", since),
                                                                                  n_ev("backlog_dropped")]}
    out["clusters"] = dict(store.get_state("alloc.clusters") or {})
    out["xover"] = {**xover_near_dup(store, since), "flagged": len(flagged(store)),
                    "grafted": sum(1 for x in (store.get_state(K_GRAFTED) or {}).values()
                                   if isinstance(x, str) and not x.startswith("-"))}
    out["fnf_pause"] = store.get_state(K_FNF_PAUSE) or ""
    out["early_kills"] = len(store.get_state(K_EARLY) or {})
    ci = [e for e in store.events("check_idea") if float(e.get("ts") or 0.0) >= since]
    late = [e for e in store.events("intake") if float(e.get("ts") or 0.0) >= since
            and str(e.get("reason") or "").startswith(("dead_mechanism", "near_dup_mechanism", "saturated_cluster",
                                                         "negative_family"))]
    out["mechanism_rejects"] = {"check_idea": sum(1 for e in ci if str(e.get("verdict", "")).startswith("REJECT")),
                                "check_idea_calls": len(ci), "at_intake": len(late)}
    return out


def meta_text(a: Mapping[str, Any]) -> list[str]:
    out = []
    if k := a.get("kinds"):
        out.append("kind yield per GPU-h (shrunk) / weight: " + ", ".join(
            f"{name} {r['rate']:.2e} (w {r['w']:.2f}, n {r['n']}, {r['gpu_h']:.0f} GPU-h)" for name, r in k.items())
                   + f"; Thompson P(argmax) over {a.get('draws', 0)} bootstrap draws; program share "
                     f"{a.get('program_share', 0):.3f}")
    if sp := a.get("split"):
        out.append(f"GPU split: LLM-idea units {sp.get('llm_rate', 0):.2e}/GPU-h vs waves + code pairs "
                   f"{sp.get('code_rate', 0):.2e}/GPU-h (over the incumbent) -> LLM share {sp.get('point', 0):.2f}, "
                   f"90% CI [{sp.get('ci_lo', 0):.2f}, {sp.get('ci_hi', 0):.2f}] -> cap {sp.get('share', 0):.2f}")
    b = a.get("backlog") or {}
    out.append(f"backlog: throttle {b.get('throttle') or 'off'}; dropped {b.get('dropped', [0, 0])[0]} in window "
               f"({b.get('dropped', [0, 0])[1]} all)")
    if cl := a.get("clusters"):
        out.append(f"clusters: {cl.get('n', 0)}; saturated-negative {', '.join(cl.get('saturated') or []) or 'none'}; "
                   f"boosted {', '.join((cl.get('boost') or [])[:6]) or 'none'}")
    x = a.get("xover") or {}
    w, al = x.get("window") or {}, x.get("all") or {}
    out.append(f"LLM crossover briefs: near_dup {w.get('near_dup', 0)}/{w.get('sessions', 0)} in window "
               f"({al.get('near_dup', 0)}/{al.get('sessions', 0)} all); code grafts {x.get('grafted', 0)}, "
               f"conflicts flagged for the LLM {x.get('flagged', 0)}")
    m = a.get("mechanism_rejects") or {}
    out.append(f"mechanism rejects: {m.get('check_idea', 0)} at check_idea ({m.get('check_idea_calls', 0)} calls) vs "
               f"{m.get('at_intake', 0)} at intake; early kills {a.get('early_kills', 0)}; fork-and-flush "
               f"{'PAUSED: ' + a['fnf_pause'] if a.get('fnf_pause') else 'not paused'}")
    return out


# ---------------------------------------------------------------------------------------------- 11-09: bold first, stale tweaks out
#: the REJECTED reason a stale-family prune writes (``loop.core.tested`` records these nodes in the tested-mechanism
#: memory, so the mechanism is not re-proposed; ``acquire.restore_aged`` does not restore them: not ``dropped_backlog``)
PRUNED_REASON = "pruned_stale_family"
_CAT_RX = re.compile(r"(?:^|\s)(?:brief|cat|bold_cat)=(\S+)")
#: an operator-intended / confirm / root-replicate node the prune never touches (besides ``acquire.program_child``)
_NO_PRUNE_RX = re.compile(r"(?:^|\s)(?:\w*confirm\w*=|root_audit=1|\w*replicate\w*=|prereq_for=|ladder=|struct_tune=|"
                          r"tune_of=|split_of=|repair_of=|port_of=|branch=|alt_of=|retry_of=|backfill_of=|mixed=)")


#: category words that name the same lane (11-09: the bold brief's nodes carry strategy ``moonshot``, the config key is
#: ``bold``): a node tagged with either matches both
CAT_ALIASES = {"bold": "moonshot", "moonshot": "bold"}
_LADDER_OF_RX = re.compile(r"(?:^|\s)ladder=(\S+)")


def priority_tags(node: Any) -> set[str]:
    """The category words a node is matched on for ``alloc.priority_categories``: its brief kind (``brief=``), its
    mechanism-taxonomy category (``cat=``), its slate category (``bold_cat=``) and its strategy (e.g. ``moonshot``),
    each with its :data:`CAT_ALIASES` twin (``bold`` <-> ``moonshot``)."""
    notes = getattr(node, "notes", "") or ""
    out = {m.group(1).lower() for m in _CAT_RX.finditer(notes)}
    if kind := brief_kind(notes):
        out.add(kind.lower())
    s = str(getattr(node, "strategy", "") or "").lower().split(".")[-1]
    if s:
        out.add(s)
    return out | {CAT_ALIASES[t] for t in out if t in CAT_ALIASES}


def _tuned_of(led: Any, node: Any) -> Any:
    """The node a structural tuning-ladder child (notes ``ladder=<node> struct_tune=``) tunes, else None."""
    notes = getattr(node, "notes", "") or ""
    if " struct_tune=" not in f" {notes}" or not (m := _LADDER_OF_RX.search(notes)):
        return None
    return getattr(led, "nodes", {}).get(m.group(1)) or getattr(led, "nodes", {}).get(getattr(node, "parent", None))


def priority_nodes(c: Any, led: Any, statuses: Mapping[str, Any]) -> set[str]:
    """QUEUED nodes of a priority category (``alloc.priority_categories``; empty = none, nanopath unchanged): their screen
    actions are placed before the other screens in ``acquire.rank`` (``prefer``) and ``acquire.idle_fill``. A structural
    tuning-ladder child (``struct_tune=``) of a priority node (its ladder's node, through nested ladders) is one too
    (11-09: a node / a node tune the moonshot a node but carry no brief tag); other program children keep their own
    (earlier) priority only; a wave point / port is no research idea."""
    cats = {str(x).lower() for x in (getattr(c, "priority_categories", ()) or ())}
    if not cats or not getattr(c, "enabled", True):
        return set()
    from loop.core.acquire import program_child, wave_node
    out = set()
    for nid, node in getattr(led, "nodes", {}).items():
        st = statuses.get(nid)
        if str(getattr(st, "value", st)).lower() != "queued" or wave_node(node):
            continue
        cur, seen = node, set()
        while (up := _tuned_of(led, cur)) is not None and up.id not in seen:   # a tuning child: its tuned node's tags
            seen.add(up.id)
            cur = up
        if cur is node and program_child(node):
            continue
        if (priority_tags(node) | priority_tags(cur)) & cats:
            out.add(nid)
    return out


def _raw_delta(post: Any, nid: str) -> float | None:
    """The unshrunk paired own increment of ``nid`` (composite), None when unmeasured."""
    try:
        from loop.core import model as M
        e = post.est[nid][M.COMPOSITE]
    except Exception:  # noqa: BLE001 - not measured
        return None
    return float(e.delta) if int(getattr(e, "paired", e.n)) >= 1 and math.isfinite(float(e.delta)) else None


def _children(led: Any) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for nid, node in getattr(led, "nodes", {}).items():
        if node.parent:
            out.setdefault(node.parent, []).append(nid)
    return out


def _family_best(post: Any, member: str, beliefs: Mapping[str, tuple[float, int]], min_seeds: int,
                 kids: Mapping[str, Sequence[str]]) -> float | None:
    """The best evidence a measured family member gives: the larger of its posterior and its raw paired own increment,
    and of every measured descendant's (its tuning-ladder children, doses, ...) raw path gain over the member's parent
    (the member's own increment + the descendants' along the chain; an unmeasured link ends that branch). None when
    the member is not measured on >= ``min_seeds`` seeds."""
    b = beliefs.get(member)
    if b is None or int(b[1]) < int(min_seeds):
        return None
    raw = _raw_delta(post, member)
    own = raw if raw is not None else float(b[0])
    best = max(float(b[0]), own)
    stack, seen = [(k, own) for k in kids.get(member, ())], {member}
    while stack:
        nid, acc = stack.pop()
        if nid in seen:
            continue
        seen.add(nid)
        d = _raw_delta(post, nid)
        if d is None:
            continue
        best = max(best, acc + d)
        stack.extend((k, acc + d) for k in kids.get(nid, ()))
    return best


def stale_family(ctx: Any, store: Any, led: Any, statuses: Mapping[str, Any], post: Any,
                 beliefs: Mapping[str, tuple[float, int]], now: float, *, exempt: Collection[str] = (),
                 protect: Mapping[str, str] | None = None, touched: Mapping[str, float] | None = None
                 ) -> list[dict[str, Any]]:
    """``alloc.prune_stale_family`` (11-09 nanobrain: 40 QUEUED small-tweak ideas, median age ~8-9 h, from families whose
    measured nodes gave at most ~+.004 against a ~+.008 need): the QUEUED nodes (derived: zero runs) older than
    ``age_h`` whose mechanism cluster (:func:`clusters`, the I3 clustering) already holds >= ``min_measured`` measured
    members (>= ``min_seeds`` seeds each, a structural node still owed its tuning ladder not counted) and NO member --
    nor any measured descendant of one (tuning-ladder children, doses), by the larger of posterior and raw paired gain --
    above ``max_best_delta``. Never: a program child (arm / ladder dose / manual-run confirm), a confirm / root-audit /
    replicate / prerequisite / tuning / split / repair / port node, a wave point, the root, a priority-category node
    (``alloc.priority_categories``), a protected cluster (a TEST pass / an incumbent ancestor), ``exempt`` (held, open
    seed blocks). Oldest first, at most ``max_per_tick``. [] when the key is absent (nanopath unchanged)."""
    c = cfg_of(ctx)
    p = getattr(c, "prune_stale_family", None)
    if p is None or not getattr(p, "enabled", True) or not c.enabled or post is None:
        return []
    from loop.core.acquire import ROOT, operator_bank_node, program_child, wave_node
    cls = clusters(ctx, store, led, beliefs, protect=protect)
    prio = {str(x).lower() for x in (getattr(c, "priority_categories", ()) or ())}
    epoch = getattr(led, "epoch", "") or ""
    ex, out, kids = set(exempt), [], _children(led)
    for cl in cls:
        if cl.protected:
            continue
        evid = {m: v for m in cl.members if m in cl.gains          # cl.gains: measured, not owed a tuning ladder
                and (v := _family_best(post, m, beliefs, int(p.min_seeds), kids)) is not None}
        cur = sum(1 for m in evid if not epoch or led.nodes[m].base == epoch)   # review: old-base evidence alone never prunes
        if len(evid) < int(p.min_measured) or cur < int(p.min_measured) or max(evid.values()) > float(p.max_best_delta):
            continue
        best = max(evid.items(), key=lambda kv: kv[1])
        for nid in cl.members:
            node = led.nodes.get(nid)
            st = statuses.get(nid)
            if node is None or str(getattr(st, "value", st)).lower() != "queued" or nid in ex or nid == ROOT \
                    or (epoch and node.base != epoch) or program_child(node) or wave_node(node) or node.port_of \
                    or _NO_PRUNE_RX.search(node.notes or "") or (prio and priority_tags(node) & prio) \
                    or (getattr(c, "keep_bank_nodes", False) and operator_bank_node(node)):
                continue
            t = max(float(node.created_ts or 0.0), float((touched or {}).get(nid) or 0.0))
            age = (now - t) / 3600.0 if t > 0 else 0.0
            if age <= float(p.age_h):
                continue
            out.append({"node": nid, "age_h": age, "cluster": cl.id, "label": cl.label, "measured": len(evid),
                        "best": best, "evidence": dict(sorted(evid.items(), key=lambda kv: -kv[1])[:6])})
    out.sort(key=lambda r: -r["age_h"])
    return out[:max(0, int(p.max_per_tick))]


def prune_stale(store: Any, rows: Sequence[Mapping[str, Any]], now: float) -> list[str]:
    """Record the stale-family prune: each node REJECTED with reason ``pruned_stale_family: ...`` (kept, never deleted;
    the tested-mechanism memory lists it so it is not re-proposed), noted ``pruned_family=<cluster>`` and logged. A node
    that launched since the statuses were derived (a run row) or that is no longer stored QUEUED is skipped."""
    from loop.contracts import NodeStatus
    acts = []
    for r in rows:
        nid = r["node"]
        node = store.get_node(nid)
        if node is None or node.status is not NodeStatus.QUEUED:
            continue
        try:
            if store.runs(node_id=nid):
                continue
        except Exception:  # noqa: BLE001 - no run view: the derived QUEUED stands
            pass
        b = r["best"]
        why = (f"{PRUNED_REASON}: queued {r['age_h']:.0f} h unscreened; its mechanism cluster {r['cluster']} "
               f"({r['label']}) has {r['measured']} measured nodes and none above the promote need (best {b[0]} "
               f"{b[1]:+.4f}); not a measured failure of this node")
        store.update_node(nid, status=NodeStatus.REJECTED, reject_reason=why, updated_ts=now,
                          notes=f"{node.notes or ''} pruned_family={r['cluster']}".strip())
        store.log("stale_family_pruned", node_id=nid, age_h=round(r["age_h"], 2), cluster=r["cluster"],
                  label=r["label"], measured=r["measured"], best=[b[0], round(b[1], 5)],
                  evidence={k: round(v, 5) for k, v in r["evidence"].items()})
        acts.append(f"backlog: pruned {nid} ({why})")
    return acts


def revive_pruned(ctx: Any, store: Any, led: Any, post: Any, beliefs: Mapping[str, tuple[float, int]], now: float,
                  protect: Mapping[str, str] | None = None) -> list[str]:
    """Undo a stale-family prune once its family is no longer flat (void != dead): a ``pruned_stale_family`` node whose
    logged evidence members' cluster now holds a member (or a measured descendant: a finished tuning ladder) above
    ``max_best_delta`` goes back to QUEUED with a fresh age window (``queued_restored``). Off with the key."""
    c = cfg_of(ctx)
    p = getattr(c, "prune_stale_family", None)
    if p is None or not getattr(p, "enabled", True) or not c.enabled or post is None:
        return []
    from loop.contracts import NodeStatus
    pruned = [n for n in store.nodes() if n.status is NodeStatus.REJECTED
              and str(n.reject_reason or "").startswith(PRUNED_REASON)]
    if not pruned:
        return []
    ev = {}
    for e in store.events("stale_family_pruned"):
        ev[str(e.get("node_id"))] = list((e.get("evidence") or {}).keys())
    cls, kids = clusters(ctx, store, led, beliefs, protect=protect), _children(led)
    by = {m: cl for cl in cls for m in cl.members}
    acts = []
    for n in pruned:
        mates = {m for x in ev.get(n.id, ()) if (cl := by.get(x)) is not None for m in cl.members} | set(ev.get(n.id, ()))
        vals = [v for m in mates if (v := _family_best(post, m, beliefs, int(p.min_seeds), kids)) is not None]
        if not vals or max(vals) <= float(p.max_best_delta):
            continue
        store.update_node(n.id, status=NodeStatus.QUEUED, reject_reason="", updated_ts=now,
                          notes=re.sub(r"\s*pruned_family=\S+", "", n.notes or "").strip())
        store.log("queued_restored", node_id=n.id, reason="stale_family_revived", best=round(max(vals), 5))
        acts.append(f"backlog: restored {n.id} (its family is no longer flat: best {max(vals):+.4f} > "
                    f"{float(p.max_best_delta):+.4f})")
    return acts
