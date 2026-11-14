"""loop.agents.history — the researcher's ``history`` tool: a per-trajectory metric dashboard with selective retrieval
(v3-c9 C1; adapted from RSI-top, "Escaping Exploration Collapse in Long-Horizon Autoresearch": a dashboard of every
trajectory's metrics that an agent UNFOLDS on demand, with random metric subsets per agent as a diversity mechanism).

About 60% of recent ideas re-parameterised terms the recipe already has, and a session saw only the static brief the
loop pre-selected -- it could not query past work. This tool lets it.

- :func:`build`       one row per ledger node, computed from the store and the VISIBLE ledger the ideator numbers use
                      (the tick's ``vis``: held-out probe tasks removed from every axis; the official ``panel`` field is
                      never read, no test verdict number is shown): the own-increment posterior m +- sd over n paired
                      seeds, the gain over the incumbent and the per-axis deltas (visible composite / axes), the test
                      verdict KIND (pass / fail), the knob-only flag and AST novelty of its diff (``loop.core.bold``),
                      the mechanism cluster (``loop.core.alloc``), the nearest-history similarity, the failure signature
                      (void / crash reason, early kill, intake reject, harm-guard veto) and a one-line description
- :func:`remember`    the tick hands its view over each tick (:class:`Ideation`); tables are cached per view token
- :func:`pick_columns` DIVERSITY MECHANISM 1: each session sees a seeded random subset of the non-key columns
                      (``ideation.history_view_frac``; id / m / n / kind always); sessions and fork-and-flush branches
                      get different subsets, logged per session (``history_view``)
- :class:`Historian`  the session's tool: ``history(action='table' | 'open' | 'clusters', ...)``. Read-only: no file
                      or store writes

Nothing here names an idea or a node."""

from __future__ import annotations

import hashlib
import math
import random
import re
import statistics
import subprocess
import threading
import time
from collections import OrderedDict, deque
from pathlib import Path
from typing import Any, Mapping, Sequence

from loop.agents.briefs import COMPUTED_NOTE, PROSE_NOTE
from loop.core import model as M
from loop.core.data import AXES, ROOT

KEY_COLS = ("id", "m", "sd", "n", "epoch", "kind")
OPT_COLS = ("parent", "tag", "cluster", "gain", "axes", "verdict", "knob", "novelty", "sim", "fail", "desc")
PARTS = ("diff", "handoff", "postmortem", "per_seed", "config")
SORTS = ("shrunk", "m", "gain", "n", "novelty", "sim", "recent", "id")
SHRINK_Z = 1.28                     # the default sort: m - 1.28 sd (a node measured on few seeds does not top the table)
MIN_N = 3                           # default min_n: rows with fewer paired seeds are marked and sorted last
DIFF_CACHE_MAX = 5000
ROWS_HARD_MAX = 40
TOOL_DEF = {"name": "history", "description": (
    "Read-only dashboard of EVERY past trajectory (node) of this search -- query it BEFORE you propose, to find what "
    "was already tried on your target axis / mechanism and what happened. action='table': a compact text table "
    "(filter: {axis: an axis name, only nodes with a measured delta on it; cluster: a cluster id such as C7; kind: a "
    "brief kind or 'all'; mechanism: substring of family / tag / description; min_m / max_m: own-increment posterior "
    "bounds; failed: true = only nodes with a failure signature (void / crash / early kill / intake reject / harm "
    "veto), false = none; epoch: 'current' (default), 'previous', 'all' or e1, e2, ...}, sort: m | gain | n | novelty | sim | recent | "
    "id | shrunk (the default: m - 1.28 sd) | an axis name (ranked by delta/se; prefix '-' for ascending), limit <= 40; "
    "min_n: default 3, rows with fewer paired seeds are marked n<3 and sorted last; epoch defaults to 'current', "
    "epoch='all' includes older base epochs; id, m, sd, n, epoch, kind are always shown, the other columns are a "
    "subset chosen for this session). action='open', node=<id>, part: diff | handoff | postmortem | per_seed | config (optionally "
    "offset=<chars> to page a long text). action='clusters': the mechanism clusters (what is saturated, what is "
    "under-visited). action='tested': the MECHANISM MEMORY the slate gate enforces -- every mechanism already measured "
    "(the operator's manual runs local:<run> with their verdicts, measured ledger nodes, slate candidates bounced as "
    "repeats) and what other sessions are building now. Numbers are on the visible ledger; m is the node's own "
    "increment over its parent."),
    "input_schema": {"type": "object", "required": ["action"], "properties": {
        "action": {"type": "string", "enum": ["table", "open", "clusters", "tested"]},
        "filter": {"type": "object", "properties": {
            "axis": {"type": "string"}, "cluster": {"type": "string"}, "kind": {"type": "string"},
            "mechanism": {"type": "string"}, "min_m": {"type": "number"}, "max_m": {"type": "number"},
            "failed": {"type": "boolean"}, "epoch": {"type": "string"}, "min_n": {"type": "integer"}}},
        "sort": {"type": "string"}, "limit": {"type": "integer"}, "node": {"type": "string"},
        "part": {"type": "string", "enum": list(PARTS)}, "offset": {"type": "integer"}}}}


def opt(ctx: Any, name: str) -> Any:
    from loop.core.levers import opt as o
    return o(ctx, name)


# ---------------------------------------------------------------------------------------------- the view hand-over
_LAST: dict[str, Any] = {"ctx": None, "store": None, "view": None, "statuses": None}
_LOCK = threading.RLock()
_BUILD_LOCK = threading.Lock()            # serialises table BUILDS only; the global _LOCK is never held while building
_TABLES: dict[str, Any] = {}              # {"tok": view token, "tab": Table} -- the last table only
# node -> (knob_only, novelty) of its diff (immutable once written); a bounded LRU (DIFF_CACHE_MAX entries)
_DIFF: "OrderedDict[str, tuple[bool, float]]" = OrderedDict()
_DIFF_LOCK = threading.Lock()
_WARM: dict[str, Any] = {"thread": None}


def remember(ctx: Any, store: Any, v: Any, statuses: Mapping[str, Any] | None = None) -> None:
    """The tick's view (the VISIBLE ledger's), once per tick (``Ideation.ideate``); the table is built lazily on the
    first ``history`` call of the view and cached per view token. Starts the diff-statistics warm-up thread."""
    with _LOCK:
        _LAST.update(ctx=ctx, store=store, view=v, statuses=statuses)
    if bool(opt(ctx, "history")):
        warm(ctx, store, v)


def forget() -> None:
    """Drop the held view / table (a closing store; tests)."""
    with _LOCK:
        _LAST.update(ctx=None, store=None, view=None, statuses=None)
        _TABLES.clear()


def _token(v: Any) -> Any:
    from loop.core.ideation import _view_token
    return _view_token(v)


def current_table(ctx: Any = None, store: Any = None, v: Any = None,
                  statuses: Mapping[str, Any] | None = None) -> "Table | None":
    """The cached :class:`Table` of the remembered view (or of ``v`` when given); None before the first tick."""
    with _LOCK:
        if v is None:
            ctx, store, v, statuses = _LAST["ctx"], _LAST["store"], _LAST["view"], _LAST["statuses"]
        if v is None or store is None:
            return None
        tok = _token(v)
        if _TABLES.get("tok") == tok and _TABLES.get("store") is store:
            return _TABLES["tab"]
    with _BUILD_LOCK:                       # double-checked: the build runs OUTSIDE the global lock (tick-thread calls
        with _LOCK:                         # to remember() / forget() never wait for it)
            if _TABLES.get("tok") == tok and _TABLES.get("store") is store:
                return _TABLES["tab"]
        tab = build(ctx, store, v, statuses)
        with _LOCK:
            _TABLES.update(tok=tok, tab=tab, store=store)
        return tab


# ---------------------------------------------------------------------------------------------- diff statistics
def _read_diff(node: Any) -> str:
    try:
        return Path(node.diff_path).read_text(errors="replace") if getattr(node, "diff_path", "") else ""
    except OSError:
        return ""


def diff_stats(node: Any) -> tuple[bool, float] | None:
    """(knob_only, AST novelty) of a node's stored diff (``loop.core.bold``); None without a readable diff. Cached
    per node (a diff is immutable once written)."""
    with _DIFF_LOCK:
        if node.id in _DIFF:
            _DIFF.move_to_end(node.id)
            return _DIFF[node.id]
    text = _read_diff(node)
    if not text.strip():
        return None
    from loop.core import bold as BOLD
    try:
        got = (bool(BOLD.knob_only(text)), round(float(BOLD.novelty(text)), 4))
    except Exception:  # noqa: BLE001 - an unparseable diff is unknown, not an error
        return None
    with _DIFF_LOCK:
        _DIFF[node.id] = got
        _DIFF.move_to_end(node.id)
        while len(_DIFF) > DIFF_CACHE_MAX:
            _DIFF.popitem(last=False)
    return got


def warm(ctx: Any, store: Any, v: Any) -> None:
    """Analyse the diffs of the ledger's nodes in a daemon thread (newest first) so a table call never waits for it;
    unanalysed nodes show '-'. The cost per diff is not constant (an AST parse of the diff: milliseconds to tens of
    milliseconds) and a full pass over a long ledger is NOT bounded by a fixed time; the thread yields the GIL after
    every diff (``time.sleep(0)``) so the tick thread is not starved, and the cache keeps the newest
    ``DIFF_CACHE_MAX`` diffs only."""
    with _LOCK:
        th = _WARM["thread"]
        n_nodes = len(getattr(getattr(v, "led", None), "nodes", ()) or ())
        if (th is not None and th.is_alive()) or (_WARM.get("n") == (id(store), n_nodes) and th is not None):
            return
        _WARM["n"] = (id(store), n_nodes)               # re-scan the store only when the ledger grew
        try:
            with _DIFF_LOCK:
                nodes = [n for n in store.nodes() if n.id not in _DIFF and getattr(n, "diff_path", "")]
        except Exception:  # noqa: BLE001 - a closing store
            return
        if not nodes:
            return
        nodes.sort(key=lambda n: -(n.created_ts or 0.0))

        def run() -> None:
            for n in nodes:
                if _LAST["store"] is not store:
                    return
                diff_stats(n)
                time.sleep(0)                           # yield the GIL between diffs
        _WARM["thread"] = threading.Thread(target=run, name="history-warm", daemon=True)
        _WARM["thread"].start()


# ---------------------------------------------------------------------------------------------- the rows
class Table:
    """The per-node rows of one view + its cluster list; ``token`` identifies the view."""

    def __init__(self, rows: list[dict[str, Any]], clusters: list[Any], token: Any, incumbent: str, v: Any,
                 epochs: dict[str, str]) -> None:
        self.rows, self.clusters, self.token, self.incumbent, self.view = rows, clusters, token, incumbent, v
        self.by_id = {r["id"]: r for r in rows}
        self.cluster_ids = {c.id: c for c in clusters}
        self.epochs = epochs
        self.current = next((e for e, sha in epochs.items() if sha == getattr(v.led, "epoch", "")), "")

    def node_cluster(self, nid: str) -> str:
        return self.by_id.get(nid, {}).get("cluster", "")

    def exists(self, ident: str) -> bool:
        """A node id or a cluster id of this table."""
        return ident in self.by_id or ident in self.cluster_ids


def kind_of(node: Any) -> str:
    from loop.core import alloc as AL
    ck, _ = AL.code_kind(node.notes)
    bk = AL.brief_kind(node.notes)
    kind = ck or bk or ("port" if node.port_of else (node.lane or "other"))
    return kind + ("/alt" if "arm=alt_base" in (node.notes or "") else "")


def _failure(store: Any, statuses: Mapping[str, Any] | None, snodes: Mapping[str, Any]) -> dict[str, str]:
    """node -> its failure signature: an intake reject reason, a void / crash / harm-veto reason (the last
    ``node_transition``), an early kill, a failed test."""
    why: dict[str, str] = {}
    try:
        for e in store.events("node_transition"):
            to, nid = str(e.get("to") or ""), str(e.get("node_id") or "")
            if to in ("void", "crashed", "vetoed", "not_confirmed", "audit_failed"):
                why[nid] = str(e.get("reason") or "")
    except Exception:  # noqa: BLE001 - no event log: the stored statuses alone
        pass
    try:
        early = {str(k) for k in (store.get_state("alloc.early_kill") or {})}
    except Exception:  # noqa: BLE001
        early = set()
    label = {"void": "void", "crashed": "crashed", "vetoed": "harm-guard veto", "not_confirmed": "test failed",
             "audit_failed": "test failed"}
    out: dict[str, str] = {}
    for nid, sn in snodes.items():
        st = getattr(getattr(sn, "status", None), "value", str(getattr(sn, "status", "")))
        if st == "rejected":
            out[nid] = "rejected: " + (str(sn.reject_reason or "").split(":", 1)[0].strip() or "intake")
        elif st in label:
            out[nid] = label[st] + (f": {why[nid]}" if why.get(nid) else "")
        if nid in early:
            out[nid] = (out[nid] + "; " if nid in out else "") + "early kill"
    for nid, s in (statuses or {}).items():
        name = str(getattr(getattr(s, "status", s), "name", "") or "").upper()
        if name in ("FAILED", "VOID", "CRASHED") and nid not in out:
            out[nid] = {"FAILED": "test failed", "VOID": "void", "CRASHED": "crashed"}[name]
    return {k: re.sub(r"\s+", " ", x).strip()[:70] for k, x in out.items()}


def _verdicts(v: Any) -> dict[str, str]:
    """node -> its last test verdict KIND (pass / fail ...). Never a number: the verdict is judged on the official panel."""
    out = {}
    for j in getattr(v, "judged", ()) or ():
        k = getattr(j.verdict.kind, "value", str(j.verdict.kind))
        if k:
            out[j.node] = str(k)
    for _, nid, kind in getattr(v.led, "verdicts", ()) or ():
        out.setdefault(nid, str(kind))
    return out


def _composite(obs: Sequence[Any], weights: Mapping[str, float]) -> dict[int, float]:
    return M.seed_means(obs, M.COMPOSITE, weights)


def axis_delta(v: Any, nid: str, axis: str) -> tuple[float, float, int] | None:
    """(delta, sd, seeds): the node's own increment over its parent on one VISIBLE axis (CUPED when the tick's
    posterior has the fit, else the same-seed mean difference)."""
    led, post = v.led, v.post
    node = led.nodes.get(nid)
    if node is None or node.parent is None:
        return None
    by = getattr(post, "by", None) or {}
    if post is not None and nid in by and node.parent in by:
        try:
            far = nid not in post.near or bool(node.port_of)
            nz, wz = post.fit(node.base, far)
            if axis in nz.sigma2:
                e = M.cuped(by[nid], by[node.parent], axis, nz, wz)
                if e is not None:
                    return float(e.delta), math.sqrt(max(0.0, e.var)), int(e.n)
        except Exception:  # noqa: BLE001 - fall through to the plain paired mean
            pass
    obs = by or led.by_node()
    a, b = M.seed_means(obs.get(nid, ()), axis), M.seed_means(obs.get(node.parent, ()), axis)
    both = a.keys() & b.keys()
    if not both:
        return None
    d = [a[s] - b[s] for s in both]
    return statistics.fmean(d), (statistics.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else 0.0), len(d)


def build(ctx: Any, store: Any, v: Any, statuses: Mapping[str, Any] | None = None) -> Table:
    """The table of this view (see the module docstring); cached per view by :func:`current_table`, which runs it
    outside the global lock (build time on the live ledger has not been benchmarked: do not rely on a fixed figure)."""
    from loop.core import alloc as AL
    from loop.core import mechanisms as MS
    from loop.novelty import mech_vectors
    led = v.led
    snodes = {n.id: n for n in store.nodes()}
    try:
        hyps = {h.id: h for h in store.hypotheses()}
    except Exception:  # noqa: BLE001
        hyps = {}
    beliefs = {n: (b.m, b.v, b.n) for n, b in v.beliefs.items()}
    passed = [j.node for j in v.judged if str(getattr(j.verdict.kind, "value", j.verdict.kind)) == "pass"]
    cls = AL.clusters(ctx, store, led, {n: (m, k) for n, (m, _v, k) in beliefs.items()},
                      protect=AL.protected(led, v.incumbent, passed), snodes=snodes)
    where = {m: c.id for c in cls for m in c.members}
    epochs = {}
    for i, (_, _nid, sha) in enumerate(sorted(led.bases), start=1):
        epochs[f"e{i}"] = sha
    sha_epoch = {sha: e for e, sha in epochs.items()}
    names = held_names(ctx, store)
    fails, verdicts = _failure(store, statuses, snodes), _verdicts(v)
    fails = {k: sanitize(x, names) for k, x in fails.items()}
    by = (getattr(v.post, "by", None) or led.by_node())
    inc_comp = _composite(by.get(v.incumbent, ()), v.weights)
    try:
        vecs = mech_vectors(store)
    except Exception:  # noqa: BLE001
        vecs = {}
    rows: list[dict[str, Any]] = []
    for nid, node in led.nodes.items():
        sn = snodes.get(nid)
        b = beliefs.get(nid)
        gain = None
        if nid != v.incumbent and node.base == (led.nodes.get(v.incumbent).base if v.incumbent in led.nodes else None):
            mine = _composite(by.get(nid, ()), v.weights)
            both = mine.keys() & inc_comp.keys()
            if both:
                gain = statistics.fmean(mine[s] - inc_comp[s] for s in both)
        h = hyps.get(getattr(sn, "hypothesis_id", None) or "")
        claim = re.sub(r"\s+", " ", sanitize((h.claim if h else "") or "", names)).strip()
        tag = f"{sn.family}/{sn.mechanism_tag}" if sn is not None else node.family
        e = sha_epoch.get(node.base, "")
        rows.append({
            "id": nid, "parent": node.parent or "-", "epoch": e + ("*" if node.base == led.epoch else ""),
            "kind": kind_of(node), "tag": tag, "cluster": where.get(nid, ""),
            "m": b[0] if b else None, "sd": math.sqrt(max(0.0, b[1])) if b else None, "n": int(b[2]) if b else 0,
            "gain": gain, "axes": {}, "verdict": verdicts.get(nid, ""), "knob": None, "novelty": None, "sim": None,
            "fail": fails.get(nid, ""), "desc": claim or tag, "ts": node.created_ts, "base": node.base,
            "status": getattr(getattr(sn, "status", None), "value", ""), "alt": "arm=alt_base" in (node.notes or "")})
    byid = {r["id"]: r for r in rows}
    for c in cls:                       # nearest earlier member of the SAME cluster (tf-idf cosine of the mechanism text)
        mem = sorted((m for m in c.members if m in vecs), key=lambda m: (byid[m]["ts"] if m in byid else 0.0, m))
        for i, m in enumerate(mem):
            if i == 0 or m not in byid:
                continue
            a = vecs[m]
            byid[m]["sim"] = round(max(sum(x * vecs[o].get(w, 0.0) for w, x in a.items()) for o in mem[:i]), 4)
    tab = Table(rows, cls, _token(v), v.incumbent, v, epochs)
    return tab


def _enrich_stats(tab: Table, snodes: Mapping[str, Any], ids: Sequence[str]) -> None:
    for nid in ids:
        r = tab.by_id.get(nid)
        if r is None or r["knob"] is not None:
            continue
        sn = snodes.get(nid)
        if sn is not None and (got := diff_stats(sn)) is not None:
            r["knob"], r["novelty"] = got


# ---------------------------------------------------------------------------------------------- the partial view
_RECENT: deque[tuple[str, ...]] = deque(maxlen=16)


def pick_columns(seed: str, frac: float, recent: Sequence[tuple[str, ...]] | None = None) -> tuple[str, ...]:
    """DIVERSITY MECHANISM 1: a seeded random subset (``frac`` of the optional columns, in their order) for one
    session. Different ``seed`` s (request id + fork-and-flush branch) draw different subsets; a subset equal to one
    of the last sessions' is redrawn (up to 24 times) so concurrent sessions and branches see different dashboards."""
    k = len(OPT_COLS) if frac >= 1 else max(0, min(len(OPT_COLS), int(round(frac * len(OPT_COLS)))))
    rng = random.Random(int(hashlib.sha1(seed.encode()).hexdigest()[:12], 16))
    recent = _RECENT if recent is None else recent
    sub: tuple[str, ...] = ()
    for _ in range(24):
        pick = set(rng.sample(OPT_COLS, k))
        sub = tuple(c for c in OPT_COLS if c in pick)
        if sub not in recent:
            break
    return sub


def session_columns(ctx: Any, request_id: str, branch: str = "") -> tuple[str, ...]:
    with _LOCK:
        sub = pick_columns(f"{request_id}|{branch}", float(opt(ctx, "history_view_frac")))
        _RECENT.append(sub)
    return sub


# ---------------------------------------------------------------------------------------------- rendering
def _f(x: float | None, spec: str = "+.4f") -> str:
    return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else format(x, spec)


def _cell(r: Mapping[str, Any], col: str, axis: str = "", min_n: int = 0) -> str:
    if col == "id":
        return r["id"]
    if col == "m":
        return _f(r["m"])
    if col == "n":
        return str(r["n"]) + (f" n<{min_n}" if r["n"] < min_n else "")
    if col == "epoch":
        return r["epoch"] or "-"
    if col == "kind":
        return r["kind"]
    if col == "sd":
        return _f(r["sd"], ".4f")
    if col == "gain":
        return _f(r["gain"])
    if col == "axes":
        parts = [f"{a}{r['axes'][a][0]:+.4f}" for a in AXES if r["axes"].get(a)]
        return " ".join(parts) or "-"
    if col == "knob":
        return "-" if r["knob"] is None else ("knob" if r["knob"] else "struct")
    if col == "novelty":
        return _f(r["novelty"], ".2f")
    if col == "sim":
        return _f(r["sim"], ".2f")
    if col == "verdict":
        return r["verdict"] or "-"
    if col == "fail":
        return r["fail"] or "-"
    if col == "desc":
        return r["desc"][:90]
    return str(r.get(col) or "-")


def render(rows: Sequence[Mapping[str, Any]], cols: Sequence[str], axis: str = "", min_n: int = 0) -> str:
    head = [*KEY_COLS, *[c for c in cols if c not in KEY_COLS]]
    if axis:
        head.insert(2, f"d_{axis}")
    lines = [" | ".join(head)]
    for r in rows:
        cells = []
        for c in head:
            if c.startswith("d_"):
                d = r["axes"].get(axis)
                cells.append(f"{d[0]:+.4f}+-{d[1]:.4f}" if d else "-")
            else:
                cells.append(_cell(r, c, axis, min_n))
        lines.append(" | ".join(cells))
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------- the filters
def _match(r: Mapping[str, Any], flt: Mapping[str, Any], tab: Table) -> bool:
    kind = str(flt.get("kind") or "")
    if kind != "all":
        if kind:
            if r["kind"].split("/")[0] != kind and r["kind"] != kind:
                return False
        elif r["id"] == ROOT or r["kind"].split("/")[0] in ("wave", "port", "factorial", "other"):
            return False                     # default: research / code-pair nodes (waves, ports, root: kind='all')
    if (c := str(flt.get("cluster") or "")) and r["cluster"] != c:
        return False
    if (s := str(flt.get("mechanism") or "").lower()) and s not in f"{r['tag']} {r['desc']}".lower():
        return False
    if flt.get("min_m") is not None and (r["m"] is None or r["m"] < float(flt["min_m"])):
        return False
    if flt.get("max_m") is not None and (r["m"] is None or r["m"] > float(flt["max_m"])):
        return False
    if flt.get("failed") is not None and bool(r["fail"]) != bool(flt["failed"]):
        return False
    e = str(flt.get("epoch") or "current")        # default: the current epoch; epoch='all' is explicit
    if e != "all" and (e != "current" or tab.current):
        labels = sorted(tab.epochs, key=lambda x: int(x[1:]))
        if e == "current":
            ok = r["epoch"].rstrip("*") == tab.current
        elif e == "previous":
            ok = len(labels) > 1 and tab.current in labels and labels.index(tab.current) > 0 \
                and r["epoch"].rstrip("*") == labels[labels.index(tab.current) - 1]
        elif e in tab.epochs:
            ok = r["epoch"].rstrip("*") == e
        else:
            ok = r["base"].startswith(e)
        if not ok:
            return False
    return True


def _need_axes(tab: Table, axis: str, ids: Sequence[str]) -> None:
    """Fill the per-axis deltas of ``ids`` (one axis, or all when ``axis`` is '') -- computed on demand."""
    for nid in ids:
        r = tab.by_id[nid]
        for a in ([axis] if axis else AXES):
            if a not in r["axes"]:
                r["axes"][a] = axis_delta(tab.view, nid, a)


def select(tab: Table, flt: Mapping[str, Any], sort: str, limit: int, snodes: Mapping[str, Any]
           ) -> tuple[list[dict[str, Any]], int]:
    """(rows after filter + sort + limit, rows matched). Axis deltas are computed for the matched nodes only. Rows with
    fewer than ``min_n`` (default 3) paired seeds sort last; the default sort is the shrunk score m - 1.28 sd; an axis
    sort ranks by delta / se."""
    axis = str(flt.get("axis") or "")
    min_n = MIN_N if flt.get("min_n") is None else max(0, int(flt["min_n"]))
    key = (sort or "shrunk").strip()
    asc = key.startswith("-")
    key = key.lstrip("-")
    if axis and axis not in AXES:
        raise ValueError(f"unknown axis {axis!r}: use one of {', '.join(AXES)}")
    if key not in SORTS and key not in AXES:
        raise ValueError(f"unknown sort {sort!r}: use one of {', '.join(SORTS)} or an axis ({', '.join(AXES)})")
    pre = {k: x for k, x in flt.items() if k not in ("axis", "min_n")}
    cand = [r for r in tab.rows if _match(r, pre, tab)]
    if axis or key in AXES:
        _need_axes(tab, axis or key, [r["id"] for r in cand])
    if axis:
        cand = [r for r in cand if r["axes"].get(axis)]
    if key in ("novelty",) or "knob" in flt:
        _enrich_stats(tab, snodes, [r["id"] for r in cand])

    def val(r: Mapping[str, Any]) -> float:
        if key in AXES:
            d = r["axes"].get(key)
            return d[0] / d[1] if d and d[1] and d[1] > 0 else -math.inf       # delta / se (no se: unranked)
        if key == "shrunk":
            return -math.inf if r["m"] is None or r["sd"] is None else r["m"] - SHRINK_Z * r["sd"]
        x = {"m": r["m"], "gain": r["gain"], "n": float(r["n"]), "novelty": r["novelty"], "sim": r["sim"],
             "recent": r["ts"], "id": None}[key]
        return -math.inf if x is None else float(x)
    def order(rs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if key == "id":
            return sorted(rs, key=lambda r: r["id"], reverse=not asc)
        known = [r for r in rs if val(r) != -math.inf]
        unknown = [r for r in rs if val(r) == -math.inf]
        known.sort(key=lambda r: (val(r), r["id"]), reverse=not asc)
        return known + unknown
    cand = order([r for r in cand if r["n"] >= min_n]) + order([r for r in cand if r["n"] < min_n])
    return cand[:max(1, limit)], len(cand)


# ---------------------------------------------------------------------------------------------- the leak guard
# Free texts (postmortem / handoff / hypothesis / diff / config) are written by agents that saw the OFFICIAL panel and
# the held-out probe numbers; nothing of that may reach an ideator through `open`. Numeric tokens near a held-out task
# name or near panel / official / labless / composite / cindex / auc are removed, the held-out names are redacted, and
# crash-log tail sections are dropped entirely.
NUM_NEAR = 40
_NUM = re.compile(r"(?<![\w.])[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?%?(?![\w])|(?<![\w.])[+-]?\.\d+(?![\w])")
_CTX = re.compile(r"panel|official|labless|composite|c-?index|auc", re.I)
_LOG_HEAD = re.compile(r"^\s*(?:Traceback \(most recent call last\)|[-=#*\s\[]*(?:crash[ _-]?log|log[ _-]?tail|"
                       r"stderr|stdout|tail of|last \d+ lines)\b)", re.I)
_FIELD = re.compile(r"^\s*(?:verdict|explanation|mechanism|suggested next|actions|learnings|concerns|claim|"
                    r"falsification)\s*:", re.I)


def held_variants(names: Sequence[str]) -> list[str]:
    """Every spelling of the held-out task names the texts may use (axis prefix, case, separators, long name parts)."""
    out: set[str] = set()
    for raw in names:
        raw = str(raw).strip()
        if not raw:
            continue
        for t in {raw, raw.partition(":")[2] or raw}:
            if not t:
                continue
            out |= {t, t.replace("_", " "), t.replace("_", "-"), t.replace("_", ""), t.replace("-", "_")}
            out |= {w for w in re.split(r"[_\-\s:]+", t) if len(w) >= 5}
    return sorted((x for x in out if x), key=lambda x: (-len(x), x))


def drop_log_tails(text: str) -> str:
    """Remove crash-log / traceback tail sections: from a log header (or a Traceback) to the next labelled field."""
    out, skip = [], False
    for line in text.splitlines():
        if _FIELD.match(line):
            skip = False
        elif _LOG_HEAD.match(line):
            skip = True
        if not skip:
            out.append(line)
    return "\n".join(out)


def sanitize(text: str, names: Sequence[str] = ()) -> str:
    """``text`` without crash-log tails, without numeric tokens within ``NUM_NEAR`` characters of a held-out task name
    or of panel / official / labless / composite / cindex / auc, and without the held-out names themselves."""
    text = drop_log_tails(str(text or ""))
    var = held_variants(names)
    rx = re.compile("|".join(re.escape(x) for x in var), re.I) if var else None
    for _ in range(6):                  # idempotent: a redaction shortens the text and can bring another number in range
        spans = [m.span() for m in _CTX.finditer(text)]
        if rx is not None:
            spans += [m.span() for m in rx.finditer(text)]

        def near(m: "re.Match[str]") -> str:
            a, b = m.span()
            if any(a - e <= NUM_NEAR and s - b <= NUM_NEAR for s, e in spans):
                return "[n]"
            return m.group(0)
        new = _NUM.sub(near, text)
        if new == text:
            break
        text = new
    if rx is not None:
        text = rx.sub("[held-out task]", text)
    return text


def held_names(ctx: Any, store: Any) -> list[str]:
    """The blind-holdout task names: the store's frozen pick (``holdout.tasks``) and the configured ones."""
    out: list[str] = []
    try:
        out += [str(x) for x in (store.get_state("holdout.tasks") or ())]
    except Exception:  # noqa: BLE001 - a store without state
        pass
    try:
        out += [str(x) for x in (getattr(getattr(getattr(ctx, "cfg", None), "holdout", None), "tasks", ()) or ())]
    except Exception:  # noqa: BLE001
        pass
    return out


# ---------------------------------------------------------------------------------------------- the tool
def _clip(text: str, n: int, offset: int = 0) -> str:
    offset = max(0, int(offset or 0))
    body = text[offset:offset + n]
    if offset + n < len(text):
        cut = body.rfind("\n")
        if cut > n // 2:
            body = body[:cut + 1]
        left = len(text) - offset - len(body)
        body += f"\n... [truncated: {left} more characters ({text.count(chr(10), offset + len(body))} lines); " \
                f"history(action='open', node=..., part=..., offset={offset + len(body)}) continues]"
    return (f"[from character {offset}]\n" if offset else "") + body


class Historian:
    """One session's ``history`` tool (``tools.historian``); ``columns``: the session's partial view."""

    def __init__(self, ctx: Any, store: Any, columns: Sequence[str] | None = None, *, request_id: str = "",
                 branch: str = "", ablated: Sequence[str] = ()) -> None:
        self.ctx, self.store = ctx, store
        self.request_id, self.branch = request_id, branch
        self.ablated = tuple(ablated or ())              # loop.components: a bypassed part is hidden here too, not only in the brief
        self.columns = tuple(OPT_COLS if columns is None else columns)
        self.calls = 0                                  # `history` calls of this session (capped: history_calls_max)

    def __call__(self, action: str = "table", filter: Any = None, sort: str = "", limit: int = 0, node: str = "",
                 part: str = "diff", offset: int = 0) -> str:
        try:
            if action == "tested":                      # 11-08: the mechanism memory (loop.core.tested); no view needed
                from loop.core import tested as TS
                return TS.table_text(self.ctx, self.store, limit)
            tab = current_table()
            if tab is None:
                return "history: no view yet (the loop builds it each tick); try again shortly"
            cap = int(opt(self.ctx, "history_calls_max"))
            if cap > 0 and self.calls >= cap:
                return (f"history: this session's budget of {cap} calls is spent: work from what you have seen "
                        "(your turns are for the slate and the build)")
            self.calls += 1
            if action == "clusters":
                return self.clusters(tab, limit)
            if action == "open":
                return self.open(tab, str(node or "").strip(), str(part or "diff"), offset)
            if action != "table":
                return f"ERROR: unknown action {action!r}: use table, open, clusters or tested"
            return self.table(tab, dict(filter) if isinstance(filter, Mapping) else {}, str(sort or ""), limit)
        except ValueError as exc:
            return f"ERROR: {exc}"
        except Exception as exc:  # noqa: BLE001 - a dashboard fault never kills the session
            return f"ERROR: history failed ({type(exc).__name__}: {str(exc)[:200]})"

    # -- table ------------------------------------------------------------------------------------------------
    def table(self, tab: Table, flt: Mapping[str, Any], sort: str, limit: int) -> str:
        cap = max(1, min(ROWS_HARD_MAX, int(opt(self.ctx, "history_rows_max"))))
        n = cap if not limit else max(1, min(cap, int(limit)))
        snodes = {x.id: x for x in self.store.nodes()}
        rows, matched = select(tab, flt, sort, n, snodes)
        axis = str(flt.get("axis") or "")
        if "axes" in self.columns and not axis:
            _need_axes(tab, "", [r["id"] for r in rows])
        if "knob" in self.columns or "novelty" in self.columns:
            _enrich_stats(tab, snodes, [r["id"] for r in rows])
        min_n = MIN_N if flt.get("min_n") is None else max(0, int(flt["min_n"]))
        pend = sum(1 for x in snodes.values() if getattr(x, "diff_path", "") and x.id not in _DIFF)
        ep = str(flt.get("epoch") or "current")
        key = (sort or "shrunk").strip().lstrip("-")
        head = (f"history (visible ledger; incumbent {tab.incumbent}; current epoch {tab.current or '?'}): "
                f"{len(rows)} of {matched} matching nodes of {len(tab.rows)}"
                + f", sorted by {sort or 'shrunk (m - 1.28 sd)'}" + "; m = own increment over the parent"
                + "; n = paired seeds" + ("; gain = composite vs the incumbent" if "gain" in self.columns else ""))
        notes = []
        if ep == "current" and tab.current:
            notes.append(f"epoch filter: current ({tab.current}); rows of OLDER base epochs are excluded "
                         "(filter {epoch: 'all'} includes them; their numbers are on another base)")
        elif ep == "all":
            notes.append("epoch filter: all (rows of older base epochs are included: not comparable to current ones)")
        notes.append(f"rows with n < {min_n} paired seeds are marked 'n<{min_n}' and sorted last (filter {{min_n}})")
        if key in AXES:
            notes.append(f"WARNING: ranked by {key} delta/se on ONE axis: a single-axis delta is noisy and unadjusted "
                         "for the many axes / nodes compared; read it as a lead, not as a win")
        foot = (f"({pend} diffs not analysed yet: knob/novelty show '-')" if pend and
                any(c in self.columns for c in ("knob", "novelty")) else "")
        return "\n".join(x for x in (head, *notes, render(rows, self.columns, axis, min_n), foot) if x)

    # -- clusters ----------------------------------------------------------------------------------------------
    def clusters(self, tab: Table, limit: int = 0) -> str:
        n = max(1, min(ROWS_HARD_MAX, int(limit or 20)))
        lines = ["cluster | category | mechanism words | nodes | measured | mean own gain | shrunk | best | flag | "
                 "recent members"]
        for c in tab.clusters[:n]:
            best = f"{c.best[0]} {c.best[1]:+.4f}" if c.best else "-"
            flag = {"saturated-negative": "SATURATED-NEGATIVE", "boost": "under-visited positive"}.get(c.flag, "")
            lines.append(f"{c.id} | {c.category} | {c.label} | {len(c.members)} | {c.n} | {c.mean:+.4f} | "
                         f"{c.shrunk:+.4f} | {best} | {flag}{' (protected: ' + c.protected + ')' if c.protected else ''}"
                         f" | {', '.join(c.sample)}")
        sat = sum(c.flag == "saturated-negative" for c in tab.clusters)
        lines.append(f"{len(tab.clusters)} clusters ({sat} saturated-negative); use history(action='table', "
                     "filter={cluster: 'C#'}) to list a cluster's nodes.")
        return "\n".join(lines)

    # -- open --------------------------------------------------------------------------------------------------
    def open(self, tab: Table, node: str, part: str, offset: int = 0) -> str:
        if node in tab.cluster_ids:
            return self.clusters_one(tab, node)
        if node not in tab.by_id:
            near = [i for i in tab.by_id if node and node in i][:5]
            return f"no node {node!r} in the history" + (f" (did you mean {', '.join(near)}?)" if near else "")
        if part not in PARTS:
            return f"ERROR: part must be one of {', '.join(PARTS)}"
        cap = max(500, int(opt(self.ctx, "history_open_chars")))
        r = tab.by_id[node]
        sn = self.store.get_node(node)
        head = f"{node} [{r['kind']}] {r['tag']} (parent {r['parent']}, epoch {r['epoch'] or '?'}, cluster {r['cluster'] or '-'})"
        if part == "diff":
            text = _read_diff(sn) if sn is not None else ""
            if not text.strip() and sn is not None:
                text = self._git_diff(sn)
            text = sanitize(text, held_names(self.ctx, self.store))
            body = _clip(text, cap, offset) if text.strip() else "(no diff recorded for this node)"
        elif part == "handoff":
            body = self._handoff(node, sn)
        elif part == "postmortem":
            body = self._postmortem(node)
        elif part == "config":
            body = self._config(node)
        else:
            body = self._per_seed(tab, node)
        if part not in ("per_seed", "diff"):  # the leak guard: every free text is sanitized before it is clipped / paged
            body = sanitize(body, held_names(self.ctx, self.store))
        return f"{head}\n--- {part} ---\n" + (body if part == "diff" else _clip(body, cap, offset))

    def clusters_one(self, tab: Table, cid: str) -> str:
        c = tab.cluster_ids[cid]
        rows = sorted((tab.by_id[m] for m in c.members if m in tab.by_id), key=lambda r: -(r["m"] or -9))[:ROWS_HARD_MAX]
        return (f"cluster {c.id} ({c.category}; {c.label}): {len(c.members)} nodes, {c.n} measured, mean {c.mean:+.4f}, "
                f"shrunk {c.shrunk:+.4f}, flag {c.flag or '-'}\n" + render(rows, self.columns))

    def _git_diff(self, sn: Any) -> str:
        repo = str(getattr(getattr(self.ctx, "spec", None), "repo", "") or "")
        par = self.store.get_node(sn.parent_id) if sn.parent_id else None
        if not repo or not sn.commit or par is None:
            return ""
        try:
            return subprocess.run(["git", "-C", repo, "diff", par.commit, sn.commit], capture_output=True,
                                  text=True, timeout=20).stdout
        except (OSError, subprocess.SubprocessError):
            return ""

    def _handoff(self, node: str, sn: Any) -> str:
        out = []
        h = self.store.get_hypothesis(sn.hypothesis_id or "") if sn is not None and sn.hypothesis_id else None
        if h is not None:
            out += [f"claim: {h.claim}", f"falsification: {h.falsification}"]
        for ho in self.store.handoffs(node_id=node):
            out += [f"actions: {ho.actions}", f"learnings: {ho.learnings}", f"concerns: {ho.concerns}"]
        return (PROSE_NOTE + "\n" + "\n".join(out)) if out else "(no hypothesis or handoff recorded)"

    def _postmortem(self, node: str) -> str:
        if "postmortem" in getattr(self, "ablated", ()):                 # the component audit bypassed postmortems for this session
            return "(postmortems are not available in this session)"
        from loop.core.closed_pm import is_computed
        pms = sorted(self.store.postmortems(node_id=node), key=lambda p: p.created_ts or 0.0)
        fmt = lambda p: (f"verdict: {p.verdict}{' (under-tuned)' if p.under_tuned else ''}\nexplanation: "      # noqa: E731
                         f"{p.explanation}\nmechanism: {p.mechanism_reasoning}\nsuggested next: {p.suggested_next}")
        comp, prose = [p for p in pms if is_computed(p)], [p for p in pms if not is_computed(p)]
        if not pms:
            return "(no postmortem recorded)"
        return "\n\n".join(([COMPUTED_NOTE + "\n" + "\n\n".join(map(fmt, comp))] if comp else [])
                           + ([PROSE_NOTE + "\n" + "\n\n".join(map(fmt, prose))] if prose else []))

    def _config(self, node: str) -> str:
        cd = self.store.get_state(f"config_diff:{node}")
        if not cd:
            return "(no config change recorded: a code-only node, or none stored)"
        from loop.adapters import flatten
        return "\n".join(f"{k}: {v}" for k, v in sorted(flatten(cd).items()))

    def _per_seed(self, tab: Table, node: str) -> str:
        """Per seed: the node's and its parent's VISIBLE composite (never an axis breakdown, never the official panel)."""
        v = tab.view
        n = v.led.nodes.get(node)
        by = v.led.by_node()
        mine, par = by.get(node, []), by.get(n.parent, []) if n is not None and n.parent else []
        a, b = _composite(mine, v.weights), _composite(par, v.weights)
        pool = {o.seed: o.pool for o in mine}
        if not a:
            return "(no finished run: nothing to show)"
        lines = ["seed | pool | node composite | parent composite | node - parent"]
        for s in sorted(a):
            lines.append(f"{s} | {pool.get(s, '?')} | {a[s]:.4f} | {_f(b.get(s), '.4f')} | "
                         f"{_f(a[s] - b[s]) if s in b else '-'}")
        both = a.keys() & b.keys()
        if both:
            lines.append(f"mean paired difference over {len(both)} shared seeds: "
                         f"{statistics.fmean(a[s] - b[s] for s in both):+.4f}")
        return "\n".join(lines)
