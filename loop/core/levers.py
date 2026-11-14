"""loop.core.levers — the computed "untested levers & deficits" section of a v3 brief (meta-analysis 10-13).

Why: 43/58 sessions claimed exactly the size the brief asked for and re-proposed the same few mechanisms (matryoshka
x4, FINO x4, patch-pooling readouts x8, LR anneal x5) while whole families had no fresh test on the current base, five
banked datasets never ran on it, the incumbent had lost an axis (SurGen) to its own stack, the only large external
headroom (classification vs a public size-matched FM) was never shown, refuted mechanisms kept coming back, and a
screened-out node with a positive paired gain (k = 4) was never re-tested. Every line here is computed from the
ledger / config; nothing names an idea.

- :func:`untested`   (a) families with 0 fresh tests on the current base epoch; config keys no node of it changed
- :func:`deficits`   (b) per axis (and its probe tasks): the incumbent vs the base root and vs the best other node of
                     the epoch (>= ``deficit_min_seeds`` seeds), each shrunk ``deficit_z`` se toward 0
- :func:`external`   (c) external headroom per axis from ``ideation.external_csv`` (best public row, size-matched peer)
- :func:`unrun`      (d) registered data-bank datasets no node of the current base trains on
- :func:`refuted`    (e) mechanism terms (tag words) whose pooled own gain is <= -``refute_k`` se: "don't repeat"
- :func:`retest`     (f) closed / stranded nodes with a positive paired own gain over >= ``retest_min_seeds`` seeds
- :func:`compute`    (g) compute headroom: the incumbent's FLOP fraction vs the cap (one sample = one tile, so the
                     unused FLOPs are per-tile compute: views / resolution / crops) and its per-tile compute keys
- :func:`dataset_gaps` (h) per classification dataset: the public lead over the incumbent (``loop.agents.diagnose``)
- :func:`repair_candidates` (i) nodes with a significant own gain on one axis (>= ``repair_k`` se) that lost another
                     axis (>= ``repair_loss_k`` se) or the composite: the axis gain is there to keep, the loss to fix
- :func:`diag_attempts` each diagnostic flag of the incumbent with the prior attempts at its implied mechanism (nodes
                     that declared its quantity, or whose tag names its words) and their verdicts
- :func:`features`   (k) per probe dataset, the incumbent's readout as the heads see it (the cached feature diagnostic,
                     ``loop.agents.diagnose.cached_features``): each per-tile gate's mean / spread, |probe features|
                     vs |cls|, both vs the across-dataset median (rows ``[G-feat-<dataset>]``)
- :func:`shortlist`  the mechanisms a brief points at (f + i + the diagnostics' prior attempts + the latest ideas),
                     pre-checked by the researcher (intake)

Every row carries an id (``[D-cls]``, ``[X-cls]``, ``[XD-break_his]``, ``[C-compute]``, ``[R-a node]``, ...): a v3 finish
ties each per-axis prediction of its budget to one (``loop.agents.researcher``).
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from loop import project as P
from loop.core import model as M
from loop.core.data import AXES, OFFICIAL_WEIGHTS


def opt(ctx: Any, name: str) -> Any:
    """``ctx.cfg.ideation.<name>``, else the schema default (``loop.spec.IdeationCfg``)."""
    from loop.spec import IdeationCfg
    ic = getattr(getattr(ctx, "cfg", None), "ideation", None)
    return getattr(ic, name, getattr(IdeationCfg(), name))


def _wave(node: Any) -> bool:
    from loop.core.acquire import wave_node
    return wave_node(node)


def _shrink(x: float, by: float) -> float:
    """``x`` moved ``by`` (>= 0) toward 0, never past it."""
    return math.copysign(max(0.0, abs(x) - by), x)


def _status(statuses: Mapping[str, Any] | None, nid: str) -> str:
    s = (statuses or {}).get(nid)
    s = getattr(s, "value", s)
    return str(s or "").lower()


# ---------------------------------------------------------------------------------------------- (a) untested
def untested(v: Any, store: Any, ctx: Any) -> tuple[list[str], list[str]]:
    """(family lines, key lines): families of the task with 0 fresh-seed tests among the current base's idea nodes;
    per family, the base config's keys no node of the current base changed (protected keys left out)."""
    from loop.adapters import flatten
    from loop.select import intake as IN
    led, ep = v.led, v.led.epoch
    tested = {j.node for j in v.judged}
    on = [n for n in led.nodes.values() if n.base == ep and n.parent is not None and not _wave(n) and not n.port_of]
    fams = sorted(getattr(getattr(ctx, "cfg", None), "families", {}) or {})
    per = {f: [0, 0] for f in fams}
    for n in on:
        if n.family in per:
            per[n.family][0] += 1
            per[n.family][1] += n.id in tested
    fam_lines = [f"{f}: {a} node(s) on this base, none fresh-tested" if a else f"{f}: no node on this base"
                 for f, (a, t) in sorted(per.items(), key=lambda x: (x[1][0], x[0])) if t == 0]
    base = store.get_state(IN.K_BASE) or {}
    changed: set[str] = set()
    for n in on:
        changed |= set(flatten(store.get_state(f"config_diff:{n.id}") or {}))
    prot = tuple(getattr(getattr(getattr(ctx, "spec", None), "surface", None), "protected_keys", ()) or ())
    keys: dict[str, list[str]] = {}
    for k in sorted(base):
        if k in changed or any(k == p or k.startswith(p + ".") for p in prot):
            continue
        f = IN.family_of(ctx, k, "keys") if fams else ""
        if f:
            keys.setdefault(f, []).append(k)
    key_lines = [f"{f}: {len(ks)} key(s) never changed on this base: {', '.join(ks[:8])}{' …' if len(ks) > 8 else ''}"
                 for f, ks in sorted(keys.items(), key=lambda x: -len(x[1]))]
    return fam_lines, key_lines


# ---------------------------------------------------------------------------------------------- (b) deficits
def deficits(v: Any, store: Any, ctx: Any) -> list[dict[str, Any]]:
    """Per axis the incumbent's level vs the best other node of the epoch with >= ``deficit_min_seeds`` seeds (its
    lead minus ``deficit_z`` se: the max over many nodes is optimistic), shrunk; ``weighted`` = official weight x
    that shrunk shortfall. Sorted by it. ``vs_root`` / ``vs_root_shrunk`` (the epoch's base root) stay as context
    fields only: every shortfall is vs the CURRENT incumbent (operator 11-10)."""
    from loop.select import intake as IN
    led, inc, ep = v.led, v.incumbent, v.led.epoch
    lv, root = v.levels.get(inc, {}), led.base_root
    z, mn = float(opt(ctx, "deficit_z")), int(opt(ctx, "deficit_min_seeds"))
    base, tasks = store.get_state(IN.K_BASE) or {}, dict(opt(ctx, "axis_tasks") or {})
    out = []
    for a in AXES:
        if a not in lv:
            continue
        m_i, n_i = lv[a]
        sd = v.noise.sd(a)
        vs_root = shr_root = None
        if root != inc and (r := v.levels.get(root, {}).get(a)):
            vs_root = m_i - r[0]
            shr_root = _shrink(vs_root, z * sd * math.sqrt(1 / n_i + 1 / r[1]))
        best = None
        for nid, l in v.levels.items():
            node = led.nodes.get(nid)
            if nid == inc or node is None or node.base != ep or a not in l or l[a][1] < mn or _wave(node):
                continue
            lead = l[a][0] - m_i
            s = lead - z * sd * math.sqrt(1 / n_i + 1 / l[a][1])
            if best is None or s > best[2]:
                best = (nid, lead, s, l[a][1])
        # the shortfall is vs the CURRENT incumbent only (operator 11-10): the best other node's shrunk lead; the base
        # root is the incumbent's ancestor, so its level is context, never a deficit
        short = max(0.0, best[2] if best else 0.0)
        names = base.get(tasks.get(a, ""))
        out.append({"axis": a, "weight": OFFICIAL_WEIGHTS[a], "inc": m_i, "n": n_i, "root": root,
                    "vs_root": vs_root, "vs_root_shrunk": shr_root, "best": best,
                    "tasks": list(names) if isinstance(names, (list, tuple)) else [], "weighted": OFFICIAL_WEIGHTS[a] * short})
    return sorted(out, key=lambda r: -r["weighted"])


# ---------------------------------------------------------------------------------------------- (c) external
def external(v: Any, ctx: Any) -> list[dict[str, Any]]:
    """Per configured axis (``external_cols``; 'panel' = the official panel) the incumbent's level, the best row of
    ``external_csv`` (group ``external_group``, ``external_exclude`` rows left out) and the best size-matched peer
    (``external_peer`` regex on the model name). [] when no table is configured / readable."""
    path = str(opt(ctx, "external_csv") or "")
    if not path:
        return []
    p = Path(path)
    if not p.is_absolute():
        p = Path(getattr(getattr(ctx, "spec", None), "repo", "") or ".") / p
    try:
        with p.open(newline="") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return []
    mcol, grp = str(opt(ctx, "external_model_col")), str(opt(ctx, "external_group") or "")
    excl = [re.compile(x, re.I) for x in opt(ctx, "external_exclude") or ()]
    peer = re.compile(str(opt(ctx, "external_peer") or r"(?!)"), re.I)
    rows = [r for r in rows if (not grp or r.get("group", grp) == grp)
            and not any(x.search(" ".join(str(c) for c in r.values())) for x in excl)]
    lv = v.levels.get(v.incumbent, {})
    out = []
    for axis, col in dict(opt(ctx, "external_cols") or {}).items():
        if axis == "panel":
            if not set(OFFICIAL_WEIGHTS) <= set(lv):
                continue
            ours = sum(w * lv[a][0] for a, w in OFFICIAL_WEIGHTS.items())
        elif axis in lv:
            ours = lv[axis][0]
        else:
            continue
        vals = []
        for r in rows:
            try:
                vals.append((float(r[col]), str(r.get(mcol, "?"))))
            except (KeyError, TypeError, ValueError):
                continue
        if not vals:
            continue
        best = max(vals)
        peers = [x for x in vals if peer.search(x[1])]
        out.append({"axis": axis, "ours": ours, "best": best, "peer": max(peers) if peers else None})
    return out


# ---------------------------------------------------------------------------------------------- (d) data bank
def unrun(v: Any, store: Any, ctx: Any) -> list[tuple[str, Mapping[str, Any]]]:
    """(name, summary) of the registered data-bank datasets no node of the current base trains on."""
    from loop import databank as DB
    db = getattr(getattr(ctx, "cfg", None), "databank", None)
    if db is None or not db.enabled:
        return []
    used = used_datasets(v.led, store, ctx)
    return [(n, s) for n, s in sorted(DB.book(store)["registered"].items()) if n not in used]


def used_datasets(led: Any, store: Any, ctx: Any, base: str | None = None) -> set[str]:
    """Data-bank datasets the nodes of ``base`` (default: the current epoch) train on (their effective config)."""
    from loop import databank as DB
    ep, cache, out = led.epoch if base is None else base, {}, set()
    for nid, n in led.nodes.items():
        if n.base == ep and (name := DB.dataset_of(ctx, DB.node_dataset(store, nid, cache))):
            out.add(name)
    return out


# ---------------------------------------------------------------------------------------------- (e) refuted
def pool(xs: Sequence[tuple[float, float]]) -> tuple[float, float]:
    """(mean, se) of estimates (d, var) pooled DerSimonian-Laird: between-estimate variance tau^2 from Cochran's Q,
    weights 1 / (var + tau^2) -- fixed-effect pooling of heterogeneous nodes would claim z of -40."""
    w = [1 / v for _, v in xs]
    sw = sum(w)
    m = sum(wi * d for wi, (d, _) in zip(w, xs)) / sw
    q = sum(wi * (d - m) ** 2 for wi, (d, _) in zip(w, xs))
    c = sw - sum(wi * wi for wi in w) / sw
    tau2 = max(0.0, (q - (len(xs) - 1)) / c) if c > 0 else 0.0
    w = [1 / (v + tau2) for _, v in xs]
    sw = sum(w)
    return sum(wi * d for wi, (d, _) in zip(w, xs)) / sw, math.sqrt(1 / sw)


def refuted(v: Any, store: Any, ctx: Any) -> list[dict[str, Any]]:
    """Mechanism TERMS (normalized tag words, ``novelty.tag_words``; words in more than ``refute_max_term_share`` of
    the nodes are too generic, ``refute_stopwords`` never count; the term must be some node's whole tag, or >= 2 of its
    nodes share one exact tag) whose pooled own increment (CUPED vs the parent, composite; random-effects pooling,
    :func:`pool`: the nodes of one term are different implementations) over >= ``refute_min_nodes`` measured nodes is
    <= -``refute_k`` se. Sorted by z."""
    from loop import novelty as NV
    post = v.post
    if post is None:
        return []
    stop = {w for x in opt(ctx, "refute_stopwords") or () for w in (x.lower(), NV.stem(x.lower()))}
    norm = {n.id: NV.norm_tag(n.mechanism_tag) for n in store.nodes()}
    tags = {nid: NV.tag_words(t) - stop for nid, t in norm.items()}
    whole = set(norm.values())                          # a term that IS some node's whole tag names a mechanism
    df = Counter(w for ws in tags.values() for w in ws)
    cap = float(opt(ctx, "refute_max_term_share")) * max(1, len(tags))
    k, mn = float(opt(ctx, "refute_k")), int(opt(ctx, "refute_min_nodes"))
    by: dict[str, list[tuple[float, float, str]]] = {}
    for nid, e in post.est.items():
        node, c = v.led.nodes.get(nid), e.get(M.COMPOSITE)
        if node is None or c is None or c.var <= 0 or node.port_of or _wave(node):
            continue
        for w in tags.get(nid, ()):
            if df[w] <= cap:
                by.setdefault(w, []).append((c.delta, c.var, nid))
    out = []
    for w, xs in by.items():
        if len(xs) < mn:
            continue
        shared = Counter(norm.get(n, "") for _, _, n in xs)
        if w not in whole and not any(c >= 2 for t, c in shared.items() if t):
            continue                                    # a word, not a mechanism: no whole tag, no shared exact tag
        m, se = pool([(d, var) for d, var, _ in xs])
        if m <= -k * se:
            out.append({"term": w, "nodes": len(xs), "m": m, "se": se, "z": m / se,
                        "ids": [n for _, _, n in sorted(xs)[:4]]})
    return sorted(out, key=lambda r: r["z"])


# ---------------------------------------------------------------------------------------------- (f) re-test
#: stored statuses of nodes already judged on fresh seeds (the old loop's confirm / audit): never re-test candidates
FRESH_JUDGED = frozenset({"confirmed", "not_confirmed", "audit_failed"})


def retest(v: Any, store: Any, ctx: Any, statuses: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Nodes outside the incumbent's lineage, never run on a fresh (held-out) seed, with a positive paired own
    increment (composite) over >= ``retest_min_seeds`` screen seeds, that the acquisition no longer works on: SCREENED
    OUT in the store, or stranded on an older base. A node that failed its fresh seeds (not confirmed, a failed test)
    is not a candidate. With each one's largest per-axis increment. Sorted by z."""
    from loop.core.data import HELD_OUT_STAGES
    from loop.core.ideation import lineage
    post = v.post
    if post is None:
        return []
    tested, lin = {j.node for j in v.judged}, set(lineage(v.led, v.incumbent))
    mn, ep = int(opt(ctx, "retest_min_seeds")), v.led.epoch
    live = {"incumbent", "testing", "passed", "queued", "failed"}
    fresh = {r.node for r in v.led.runs if r.stage in HELD_OUT_STAGES}
    out = []
    for nid, e in post.est.items():
        node, c = v.led.nodes.get(nid), e.get(M.COMPOSITE)
        if node is None or c is None or nid in tested or nid in fresh or nid in lin or node.port_of or _wave(node) \
                or c.n < mn or c.delta <= 0 or c.var <= 0 or _status(statuses, nid) in live:
            continue
        if node.status in FRESH_JUDGED or node.status != "screened_out" and node.base == ep:
            continue                                  # judged on fresh seeds / still screening on this base
        axes = {}
        try:
            nz, wz = post.fit(node.base, True)
            for a in AXES:
                if a in nz.sigma2 and (ea := M.cuped(post.by[nid], post.by[node.parent], a, nz, wz)) is not None:
                    axes[a] = ea.delta
        except (KeyError, ValueError, ZeroDivisionError):
            pass
        sn = store.get_node(nid)
        out.append({"node": nid, "tag": getattr(sn, "mechanism_tag", ""), "family": node.family, "m": c.delta,
                    "z": c.delta / math.sqrt(c.var), "n": c.n, "old": node.base != ep, "status": node.status,
                    "axis": max(axes.items(), key=lambda x: x[1]) if axes else None})
    return sorted(out, key=lambda r: -r["z"])


# ---------------------------------------------------------------------------------------------- summary scan
_SCAN: dict[str, dict[str, list]] = {}         # cache file -> {output_dir: [mtime_ns, size, summary]}
SCAN_KEEP = 4000                                # entries the disk cache keeps (the newest)


def _scan_path(ctx: Any) -> Path | None:
    sd = getattr(ctx, "state_dir", None)
    return Path(sd) / "cache" / "summaries.json" if sd else None


def summaries(ctx: Any, dirs: Sequence[str]) -> dict[str, dict[str, Any]]:
    """{output_dir: its summary.json ({} when absent / unreadable)}, time-boxed (``ideation.levers_scan_s``; 0 = none)
    and cached on disk (``<state>/cache/summaries.json``) keyed by the file's mtime + size: an unchanged file is never
    re-read, and past the time box a dir is served from the cache whatever its mtime (else {}). Read errors are never
    cached. Every summary found also warms ``loop.agents.diagnose``'s in-process cache (its breakdown reads them)."""
    budget, t0 = float(opt(ctx, "levers_scan_s") or 0.0), time.monotonic()
    path = _scan_path(ctx)
    key = str(path) if path else ""
    if key not in _SCAN:
        try:
            got = json.loads(path.read_text()) if path and path.is_file() else {}
        except (OSError, ValueError):
            got = {}
        _SCAN[key] = got if isinstance(got, dict) else {}
    cache, out, dirty = _SCAN[key], {}, False
    for d in dict.fromkeys(x for x in dirs if x):
        c = cache.get(d)
        if budget > 0 and time.monotonic() - t0 > budget:
            out[d] = c[2] if c else {}
            continue
        p = Path(d) / "summary.json"
        try:
            st = p.stat()
        except OSError:
            out[d] = {}
            continue
        if c and c[0] == st.st_mtime_ns and c[1] == st.st_size:
            out[d] = c[2]
            continue
        try:
            s = json.loads(p.read_text())
        except (OSError, ValueError):
            out[d] = {}
            continue
        out[d] = s if isinstance(s, dict) else {}
        cache.pop(d, None)
        cache[d], dirty = [st.st_mtime_ns, st.st_size, out[d]], True
    if dirty and path:
        while len(cache) > SCAN_KEEP:
            cache.pop(next(iter(cache)))
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(cache))
            os.replace(tmp, path)
        except OSError:
            pass
    try:
        from loop.agents import diagnose as DX
        for d, s in out.items():
            if s:
                DX._SUMMARY.setdefault(d, s)
    except (ImportError, AttributeError):
        pass
    return out


def _done_dirs(led: Any, nid: str) -> list[str]:
    return [r.output_dir for r in led.runs if r.node == nid and r.status == "done" and r.output_dir]


# ---------------------------------------------------------------------------------------------- (g) compute
def effective_config(store: Any, led: Any, nid: str, ctx: Any = None) -> dict[str, Any]:
    """The node's run config: a done run's own config file (its summary's ``config_path``), else the base config +
    the node's chain of config diffs."""
    import yaml
    from loop.adapters import flatten
    from loop.select import intake as IN
    sums = summaries(ctx, _done_dirs(led, nid))
    for d in _done_dirs(led, nid):
        if (p := sums.get(d, {}).get("config_path")):
            try:
                return flatten(yaml.safe_load(Path(p).read_text()) or {})
            except (OSError, yaml.YAMLError):
                continue
    return {**(store.get_state(IN.K_BASE) or {}), **IN.parent_config(store, nid)}


def compute(v: Any, store: Any, ctx: Any) -> dict[str, Any] | None:
    """The incumbent's compute use from its done runs' summaries: mean ``flop_fraction`` / ``sample_fraction``, the
    stop reasons, the per-sample compute multiplier the FLOP cap still allows at the same sample count (1 / flop
    fraction), and its per-sample compute keys (``ideation.compute_keys``). None without summaries."""
    sums = [s for s in summaries(ctx, _done_dirs(v.led, v.incumbent)).values() if s]
    ff = [float(s["flop_fraction"]) for s in sums if isinstance(s.get("flop_fraction"), (int, float))]
    if not ff:
        return None
    sf = [float(s["sample_fraction"]) for s in sums if isinstance(s.get("sample_fraction"), (int, float))]
    stop = Counter(str(s.get("stop_reason", "?")) for s in sums)
    rx = re.compile(str(opt(ctx, "compute_keys") or r"(?!)"))
    cfg = effective_config(store, v.led, v.incumbent, ctx)
    keys = {k: cfg[k] for k in sorted(cfg) if rx.search(k.rsplit(".", 1)[-1])}
    m = statistics.fmean(ff)
    return {"runs": len(sums), "flop_fraction": m, "sample_fraction": statistics.fmean(sf) if sf else None,
            "stop": dict(stop), "multiplier": 1.0 / m if m > 0 else None, "keys": keys}


# ---------------------------------------------------------------------------------------------- (h) per dataset
def dataset_gaps(v: Any, store: Any, ctx: Any) -> list[dict[str, Any]]:
    """Per classification dataset the public lead over the incumbent (its largest head), from the breakdown
    (``loop.agents.diagnose``); the breakdown is recorded for the researcher's ``diagnose`` tool."""
    from loop.agents import diagnose as DX
    if not str(opt(ctx, "external_task_csv") or ""):
        return []
    summaries(ctx, _done_dirs(v.led, v.incumbent))     # time-boxed, disk-cached: warms the breakdown's reads
    rows = DX.breakdown(ctx, v.led, v.incumbent, DX.held_names(store))
    if hasattr(store, "set_state") and v.incumbent in v.led.nodes:
        sn = store.get_node(v.incumbent) if hasattr(store, "get_node") else None
        DX.record(ctx, store, v.led, v.incumbent, getattr(sn, "commit", "") or "", rows)
    return DX.dataset_gaps(rows)


# ---------------------------------------------------------------------------------------------- (i) repair
def repair_candidates(v: Any, store: Any, ctx: Any, statuses: Mapping[str, Any] | None = None
                      ) -> list[dict[str, Any]]:
    """Nodes outside the incumbent's lineage (not live: queued / testing / passed / incumbent; never a carrier or a
    node that passed a fresh-seed test: those were not screened out) with an own increment (CUPED vs the parent) of
    >= ``repair_k`` se on some axis over >= ``repair_min_seeds`` seeds, that lost another axis by >= ``repair_loss_k``
    se or have a composite own gain <= 0: screened out by the loss, while the gain is real. The current base's nodes
    first, then by the gain's z."""
    from loop.core.ideation import lineage
    post = v.post
    if post is None:
        return []
    k, kl, mn = float(opt(ctx, "repair_k")), float(opt(ctx, "repair_loss_k")), int(opt(ctx, "repair_min_seeds"))
    lin, live = set(lineage(v.led, v.incumbent)), {"incumbent", "testing", "passed", "queued"}
    lin |= {n for _, n in getattr(v.led, "carriers", ())} | {j.node for j in v.judged if j.switched}
    out = []
    for nid, e in post.est.items():
        node, c = v.led.nodes.get(nid), e.get(M.COMPOSITE)
        if node is None or c is None or nid in lin or node.port_of or _wave(node) or c.n < mn \
                or _status(statuses, nid) in live or node.parent not in post.by:
            continue
        try:
            nz, wz = post.fit(node.base, True)
        except (KeyError, ValueError, ZeroDivisionError):
            continue
        axes = {}
        for a in AXES:
            if a in nz.sigma2 and (ea := M.cuped(post.by[nid], post.by[node.parent], a, nz, wz)) is not None \
                    and ea.var > 0:
                axes[a] = (ea.delta, ea.delta / math.sqrt(ea.var))
        gains = {a: x for a, x in axes.items() if x[1] >= k}
        losses = {a: x for a, x in axes.items() if x[1] <= -kl}
        if not gains or not (losses or c.delta <= 0):
            continue
        sn = store.get_node(nid) if hasattr(store, "get_node") else None
        top = max(gains.items(), key=lambda x: x[1][1])
        out.append({"node": nid, "family": node.family, "tag": getattr(sn, "mechanism_tag", ""), "n": c.n,
                    "m": c.delta, "gains": gains, "losses": losses, "axis": top[0], "z": top[1][1],
                    "old": node.base != v.led.epoch})
    return sorted(out, key=lambda r: (r["old"], -r["z"]))


# ---------------------------------------------------------------------------------------------- diagnostics
def diag_attempts(v: Any, store: Any, ctx: Any) -> list[dict[str, Any]]:
    """Each diagnostic flag of the incumbent (``loop.diagnostics.bottlenecks``; the nearest diagnosed node of its
    lineage when it has none) with the PRIOR ATTEMPTS at the mechanism it implies: nodes that declared the flag's
    trend quantity as their mechanism prediction, or whose tag names a word of the flag (a word too common to name a
    mechanism -- ``refute_max_term_share`` -- or a ``refute_stopwords`` word does not count). Their verdicts: measured
    engagement, the pooled own gain (random effects, :func:`pool`), fresh-seed passes."""
    from loop import diagnostics as DG
    from loop import novelty as NV
    from loop.core.ideation import K_ENGAGED, lineage, prediction
    if not hasattr(store, "diagnostics"):
        return []
    src = next((n for n in lineage(v.led, v.incumbent) if store.diagnostics(node_id=n)), None)
    flags = DG.bottlenecks(store, src) if src else []
    if not flags:
        return []
    stop = {w for x in opt(ctx, "refute_stopwords") or () for w in (x.lower(), NV.stem(x.lower()))}
    tags = {n.id: NV.tag_words(n.mechanism_tag) - stop for n in store.nodes()}
    df = Counter(w for ws in tags.values() for w in ws)
    cap = max(NV.RARE_MIN, float(opt(ctx, "refute_max_term_share")) * max(1, len(tags)))
    declared = {nid: str(mp.get("diagnostic") or "") for nid in v.led.nodes
                if (mp := prediction(store, nid)) is not None}
    engaged = store.get_state(K_ENGAGED) or {}
    passed = {j.node for j in v.judged if j.switched}
    post, out = v.post, []
    for f in flags:
        words = {w for w in NV.tag_words(f["flag"]) if w not in stop and df[w] <= cap}
        q = f.get("trend_quantity") or ""
        hit = sorted({n for n, d in declared.items() if q and d == q}
                     | {n for n, ws in tags.items() if words & ws and n in v.led.nodes})
        est = [(e.delta, e.var) for n in hit if post is not None
               and (e := post.est.get(n, {}).get(M.COMPOSITE)) is not None and e.var > 0]
        pm = pool(est) if len(est) >= 2 else ((est[0][0], math.sqrt(est[0][1])) if est else None)
        eng = [engaged[n] for n in hit if n in engaged]
        best = max(((n, v.beliefs[n].m) for n in hit if n in v.beliefs), key=lambda x: x[1], default=None)
        recent = sorted(hit, key=lambda n: -v.led.nodes[n].created_ts)
        out.append({"flag": f["flag"], "severity": f["severity"], "text": f["text"], "quantity": q, "src": src,
                    "words": sorted(words), "attempts": len(hit), "measured": len(est), "pooled": pm,
                    "engaged": (sum(e == "engaged" for e in eng), len(eng)), "passed": sorted(passed & set(hit)),
                    "best": best, "recent": recent[:int(opt(ctx, "precheck_recent"))]})
    return out


def diag_text(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = []
    for r in rows:
        pm = r["pooled"]          # v3-c7i I9: the engagement count is not shown (it does not predict gain)
        lines.append(f"- [G-{r['flag']}] {r['text']} (sev {r['severity']:.2f}, from {r['src']}). PRIOR ATTEMPTS at its "
                     f"implied mechanism (declared {r['quantity'] or '-'}"
                     + (f", or tag words {', '.join(r['words'])}" if r["words"] else "") + f"): {r['attempts']} nodes"
                     + (f", pooled own gain {pm[0]:+.4f} (se {pm[1]:.4f}) over {r['measured']} measured" if pm else
                        ", none measured")
                     + (f", passed a fresh-seed test: {', '.join(r['passed'])}" if r["passed"] else
                        ", none passed a fresh-seed test")
                     + (f"; best {r['best'][0]} {r['best'][1]:+.4f}" if r["best"] else "")
                     + (f"; latest {', '.join(r['recent'][:4])}" if r["recent"] else "")
                     + ". A flag is a symptom, not a mechanism: fixing it is worth a slot only if the attempts above "
                       "did not already answer it.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------- shortlist + text
def shortlist(store: Any, ctx: Any, retests: Sequence[Mapping[str, Any]] = (),
              extra: Sequence[tuple[str, str]] = ()) -> tuple[str, ...]:
    """``family|mechanism_tag`` the brief points at: the re-test candidates, ``extra`` (repair candidates, the
    diagnostics' implied mechanisms), then the latest ``precheck_recent`` research-session ideas (their mechanisms
    are the likeliest repeats). Deduplicated."""
    from loop import novelty as NV
    k = int(opt(ctx, "precheck_recent"))
    items = [(r["family"], r["tag"]) for r in retests[:k] if r.get("tag")] + [(f, t) for f, t in extra if t]
    ideas = sorted((n for n in store.nodes() if "request=" in (n.notes or "") and n.mechanism_tag),
                   key=lambda n: -(n.created_ts or 0.0))
    items += [(n.family, n.mechanism_tag) for n in ideas[:k]]
    seen, out = set(), []
    for f, t in items:
        if (key := NV.norm_tag(t)) and key not in seen:
            seen.add(key)
            out.append(f"{f}|{t}")
    return tuple(out)


# ---------------------------------------------------------------------------------------------- (k) features
def _mean(stats: Any) -> float | None:
    x = stats.get("mean") if isinstance(stats, Mapping) else None
    return float(x) if isinstance(x, (int, float)) and math.isfinite(x) else None


def _n(x: Any) -> str:
    return f"{x:.3f}" if isinstance(x, (int, float)) else "-"


def features(v: Any, store: Any, ctx: Any) -> list[dict[str, Any]]:
    """Per probe dataset (v3-c5 I5): the incumbent's cached feature diagnostic (``diagnose.cached_features``: a
    SLURM job reads it once per incumbent) -- every per-tile gate the readout applies (mean / p10 / p50 / p90), the
    norm ratio |probe features (cls width)| / |cls| (how far the readout scales the token the heads read) and their
    cosine, each gate mean and ratio also vs its median over the datasets. Most-suppressed dataset first; [] when no
    report is cached yet."""
    from loop.agents import diagnose as DX
    fn = getattr(DX, "cached_features", None)
    rep = fn(ctx, store, getattr(v, "incumbent", None)) if fn is not None else None
    if not rep:
        return []
    rows = []
    for ds, r in rep.items():
        gates = [g for g in (r.get("gates") or []) if _mean(g) is not None]
        head, cls = _mean(r.get("norm_probe_features_head")), _mean(r.get("norm_cls"))
        rows.append({"dataset": ds, "tiles": int(r.get("tiles") or 0), "gates": gates,
                     "gate": _mean(gates[0]) if gates else None,
                     "ratio": head / cls if head is not None and cls else None,
                     "cos": _mean(r.get("cos_head_vs_cls"))})
    gs, rs = [r["gate"] for r in rows if r["gate"] is not None], [r["ratio"] for r in rows if r["ratio"] is not None]
    med_g, med_r = (statistics.median(gs) if gs else None), (statistics.median(rs) if rs else None)
    for r in rows:
        r["median_gate"], r["median_ratio"] = med_g, med_r
        r["gate_vs_median"] = r["gate"] / med_g if r["gate"] is not None and med_g else None
        r["ratio_vs_median"] = r["ratio"] / med_r if r["ratio"] is not None and med_r else None
    return sorted(rows, key=lambda r: (r["gate"] is None, r["gate"] or 0.0, r["ratio"] or 0.0, r["dataset"]))


def features_text(rows: Sequence[Mapping[str, Any]], k: int) -> str:
    out = []
    for r in rows[:k]:
        gl = "; ".join(f"gate{i} mean {_n(g.get('mean'))} (p10 {_n(g.get('p10'))} p50 {_n(g.get('p50'))} p90 "
                       f"{_n(g.get('p90'))})" for i, g in enumerate(r["gates"][:2]))
        out.append(f"- [G-feat-{r['dataset']}] {r['dataset']} ({r['tiles']} tiles): " + (gl or "no per-tile gate")
                   + (f", {r['gate_vs_median']:.2f}x the median dataset's {r['median_gate']:.3f}"
                      if r["gate_vs_median"] is not None else "")
                   + (f" | |probe features|/|cls| {r['ratio']:.3f}" if r["ratio"] is not None else "")
                   + (f" ({r['ratio_vs_median']:.2f}x the median {r['median_ratio']:.3f})"
                      if r["ratio_vs_median"] is not None else "")
                   + (f" | cos(head, cls) {r['cos']:.3f}" if r["cos"] is not None else ""))
    return "\n".join(out)


def _f(x: float | None, d: int = 4) -> str:
    return "-" if x is None else f"{x:+.{d}f}"


def text(v: Any, store: Any, ctx: Any, statuses: Mapping[str, Any] | None = None,
         retests: Sequence[Mapping[str, Any]] | None = None, repairs: Sequence[Mapping[str, Any]] | None = None,
         diags: Sequence[Mapping[str, Any]] | None = None) -> str:
    """The brief section (each part computed on its own: one that cannot be computed says so). Every row carries its
    id in brackets: a v3 finish's per-axis budget names the row each prediction draws on."""
    from loop.core.ideation import AXIS_NAMES
    k = int(opt(ctx, "levers_k"))
    parts = []

    def part(title: str, fn: Any) -> None:
        try:
            body = fn()
        except Exception as exc:  # noqa: BLE001 - one part never loses the brief
            body = f"(not computed: {type(exc).__name__})"
        parts.append(f"### {title}\n{body or '(none)'}")

    def a() -> str:
        fl, kl = untested(v, store, ctx)
        return "\n".join([f"- [U-{x.split(':', 1)[0]}] family {x}" for x in fl[:k]] + [f"- keys of {x}" for x in kl[:k]])

    def b() -> str:
        rows = []
        for r in deficits(v, store, ctx):
            bst = r["best"]
            rows.append(f"- [D-{r['axis']}] {AXIS_NAMES[r['axis']]} ({r['weight']:.3f}"
                        f"{'; ' + ', '.join(r['tasks']) if r['tasks'] else ''})"
                        f": incumbent {r['inc']:.4f} (n={r['n']}; every delta here is vs it) | best other node "
                        + (f"{bst[0]} {_f(bst[1])} (shrunk {_f(bst[2])}, n={bst[3]})" if bst else "-")
                        + (f" | DEFICIT, weighted {r['weighted']:.4f}" if r["weighted"] > 0 else ""))
        return "\n".join(rows)

    def c() -> str:
        rows = []
        for r in external(v, ctx):
            name = "official panel" if r["axis"] == "panel" else AXIS_NAMES.get(r["axis"], r["axis"])
            pe = r["peer"]
            rows.append(f"- [X-{r['axis']}] {name}: incumbent {r['ours']:.4f} | best public {r['best'][1]} "
                        f"{r['best'][0]:.4f} ({r['best'][0] - r['ours']:+.4f})"
                        + (f" | size-matched {pe[1]} {pe[0]:.4f} ({pe[0] - r['ours']:+.4f})" if pe else ""))
        return "\n".join(rows)

    def d() -> str:
        from loop import databank as DB
        return "\n".join(f"- {n}: {DB.line(ctx, s)}" for n, s in unrun(v, store, ctx))

    def e() -> str:
        return "\n".join(f"- {r['term']}: {r['nodes']} nodes, pooled own gain {r['m']:+.4f} (se {r['se']:.4f}, "
                         f"z {r['z']:+.1f}), e.g. {', '.join(r['ids'])}" for r in refuted(v, store, ctx)[:k])

    def f() -> str:
        rows = retests if retests is not None else retest(v, store, ctx, statuses)
        return "\n".join(f"- [T-{r['node']}] {r['node']} {r['family']}/{r['tag']}: own gain {r['m']:+.4f} "
                         f"(z {r['z']:+.1f}, {r['n']} seeds; {r['status']}{', older base: port first' if r['old'] else ''})"
                         + (f", largest axis {AXIS_NAMES[r['axis'][0]]} {r['axis'][1]:+.4f}" if r["axis"] else "")
                         for r in rows[:k])

    def g() -> str:
        cp = compute(v, store, ctx)
        if cp is None:
            return ""
        sf = cp["sample_fraction"]
        return (f"- [C-compute] the incumbent's {cp['runs']} run(s) stop on {', '.join(f'{s} x{n}' for s, n in cp['stop'].items())}"
                f" with flop_fraction {cp['flop_fraction']:.3f}" + (f" (sample_fraction {sf:.3f})" if sf is not None else "")
                + (P.rule("lever_compute_tail", ": one sample is one tile, so x{mult:.2f} more compute PER TILE fits "
                          "under the FLOP cap at the same tile count (more / larger views, higher resolution, more "
                          "crops)").format(mult=cp["multiplier"]) if cp["multiplier"] else "")
                + ("\n  " + P.rule("lever_compute_keys", "per-tile compute keys now") + ": "
                   + ", ".join(f"{k2}={v2}" for k2, v2 in cp["keys"].items()) if cp["keys"] else ""))

    def h() -> str:
        rows = dataset_gaps(v, store, ctx)
        return "\n".join(f"- [XD-{r['dataset']}] {r['dataset']}: incumbent {r['head']} {r['ours']:.4f} | public "
                         f"{r['ext'][0]} {r['ext'][1]:.4f} ({r['gap']:+.4f})"
                         + (f" | best other node {r['node_lead'][1]} ({r['node_lead'][0]:+.4f})" if r["node_lead"] else "")
                         for r in rows[:k])

    def i() -> str:
        rows = repairs if repairs is not None else repair_candidates(v, store, ctx, statuses)
        out = []
        for r in rows[:k]:
            gains = ", ".join(f"{AXIS_NAMES[a]} {x[0]:+.4f} (z {x[1]:+.1f})" for a, x in r["gains"].items())
            loss = ", ".join(f"{AXIS_NAMES[a]} {x[0]:+.4f} (z {x[1]:+.1f})" for a, x in r["losses"].items())
            out.append(f"- [R-{r['node']}] {r['node']} {r['family']}/{r['tag']} ({r['n']} seeds"
                       f"{'; older base' if r['old'] else ''}): GAINED {gains}; lost {loss or 'none significant'}; "
                       f"composite {r['m']:+.4f}")
        return "\n".join(out)

    def j() -> str:
        return diag_text(diags if diags is not None else diag_attempts(v, store, ctx))

    part("(a) Untested on this base (families with no fresh test; config keys no node of this base changed). UNTESTED "
         "is not PROMISING: most knob moves on this base measured null, so treat a never-changed key as a place to "
         "look for a MECHANISM, not as a dose to try", a)
    part("(b) Per-axis deficits of the incumbent (shrunk; a deficit is an axis to win back or a stack that cost "
         "one)", b)
    part("(c) External headroom (public models, same probe suite)", c)
    part("(d) Data-bank datasets never run on this base (the loop runs each once as a config-only node on the "
         "incumbent)", d)
    part(f"(e) Refuted on this task -- DON'T REPEAT (pooled own gain <= -{float(opt(ctx, 'refute_k')):g} se) unless "
         "you name the nodes and say what answers their failure", e)
    part(f"(f) Re-test candidates (positive paired own gain over >= {int(opt(ctx, 'retest_min_seeds'))} seeds, "
         "never fresh-tested)", f)
    part("(g) Compute headroom (the FLOP cap vs what the incumbent uses)", g)
    part("(h) Per-dataset classification gaps (the public model's lead on the incumbent's weakest head; "
         "`diagnose(what='classification')` shows every dataset x head)", h)
    part(f"(i) Repair candidates (a real gain on one axis, >= {float(opt(ctx, 'repair_k')):g} se, screened out by a "
         f"loss elsewhere, >= {float(opt(ctx, 'repair_loss_k')):g} se, or a composite <= 0: keep the gain, fix the loss)", i)
    part("(j) Incumbent diagnostics, each with the prior attempts at the mechanism it implies", j)
    part("(k) The incumbent's readout per probe dataset (cached feature diagnostic; most-suppressed first): each "
         "per-tile gate inside probe_features, |probe features| vs |cls|. A dataset whose gate / norm sits far below "
         "the others is read through a shrunk token; the slide and robustness axes depend on feature scale, so a "
         "readout change is judged per axis (readout-only diffs are scored on the parent's own checkpoints)",
         lambda: features_text(features(v, store, ctx), max(k, 16)))
    return "## Untested levers & deficits (computed)\n" + "\n\n".join(parts)
