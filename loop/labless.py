"""loop.labless — auto-submit to Labless, once per node per TIER.

Tiers:
  audited      a node that PASSED the held-out audit (book key: the node id), title "[aud] ...";
  exploratory  every CONFIRMED node at once, then ONE representative per CLOSED tested group (operator
               2026-10-05: "reduce labless spam, only send potentially valid / best-of-some-tested-group (even if
               it did bad)"): a dose ladder / tuning pass, a leg-factorial round, a design batch, a mechanism
               cluster once all its members closed -- its best member by own gain, flagged when that is <= 0
               (:func:`closed_groups`; book key "<id>/exp", title "[exp] ..."); skipped for a node already submitted.
Audited first, then confirmed, then groups by best gain; a rolling 24 h cap (max_per_day) on attempts.
No near-duplicate spam (operator 2026-09-30): a confirmed exploratory node with the same code and near-same config
values as a submitted node, or one queued ahead, is skipped unless its gain over it clears labless.sibling_gain
and its noise (:func:`duplicate`).
Success is a non-null ``submission_id`` from an exit-0 submitter with no validation errors, stored
here -- a receipt file alone proves nothing. The submitter runs in the driver's pool: a failure is
logged and retried with backoff, and never blocks the loop."""

from __future__ import annotations

import json
import math
import os
import re
import statistics
import subprocess
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from loop.adapters import flatten, set_dotted
from loop.contracts import PANEL, ROOT_NODE_ID, NodeStatus, Posterior, Run, RunStatus, Stage, WorldSnapshot
from loop.seeds import _panel, audit_deltas, audit_gate, audit_offsets
from loop.spec import LoopContext

K_LABLESS = "labless.submissions"       # node_id -> {submission_id, run_id, tries, last_ts, next_ts, error}
K_LABLESS_TS = "labless.attempts"      # attempt timestamps of the last 24 h (the max_per_day cap)
K_AUDIT = "driver.audit"                # the driver's audit book: {passed, failed, queue, current, gate}
MAX_BACKOFF_H = 48.0
RESUME_KEYS = ("train.resume", "train.save_every")   # all a resume / eval-only config copy changes
CONFIG_PATH_ERR = "must be a repo-relative configs/"  # the submitter's refusal of a config copy's path
MIN_BACKOFF_H = 1.0                  # a FAILED try waits at least this long even at min_hours_between 0
EXP = "/exp"                         # the exploratory tier's book-key suffix
TIER_PREFIX = {"audited": "[aud]", "exploratory": "[exp]"}
LIFECYCLE = (Stage.SCREEN, Stage.PROMOTE, Stage.CONFIRM)


KEY_ALIAS = {"local_crop_scale": "crop", "global_crop_scale": "gcrop", "kde_concentration": "kde",
             "kde_loss_weight": "kdew", "learning_rate": "lr", "base_lr": "lr", "lr": "lr",
             "weight_decay": "wd", "teacher_temp": "tt", "jepa_blocks": "jb", "warmup_ratio": "wu"}
NAME_MAX = 20                                    # the submitter caps a full-tier label at 20 chars
_MISSING = object()


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def short_value(v: Any) -> str:
    """A compact value for a title: 0.16 -> '.16', 6.667 -> '6.7', 1.9e-4 -> '1.9e-4', [a, b] -> a's."""
    if isinstance(v, (list, tuple)):
        return short_value(v[0]) if v else "[]"
    if isinstance(v, bool):
        return "on" if v else "off"
    if not _num(v):
        return str(v)[:6]
    if v == int(v) and abs(v) < 1e4:
        return str(int(v))
    if v == 0 or 1e-3 <= abs(v) < 1e4:
        s = f"{v:.2g}"
        return s.replace("0.", ".", 1) if s.startswith(("0.", "-0.")) else s
    m, e = f"{v:.1e}".split("e")
    return f"{m.replace('.0', '')}e{int(e)}"


def short_key(key: str) -> str:
    leaf = key.rsplit(".", 1)[-1]
    return KEY_ALIAS.get(leaf) or re.sub(r"_(scale|weight|rate|ratio)$", "", leaf).replace("_", "")[:6]


def run_name(node_id: str, config_diff: Mapping[str, Any] | None = None, tag: str = "", prefix: str = "v2") -> str:
    """<= 20 chars, human-readable from the change: 'v2 crop.16+kde6.7' (short key aliases + new
    values; a ' #<id digits>' suffix when it fits), else 'v2 <mechanism tag>', else 'loop-v2-<id>'.
    ``prefix`` replaces 'v2' (the submitter passes the tier: '[aud]' / '[exp]')."""
    parts = [f"{short_key(k)}{short_value(v)}" for k, v in (config_diff or {}).items()]
    body, kept = prefix, []
    for i, part in enumerate(parts):
        cand = "+".join([*kept, part])
        if len(f"{prefix} {cand}") > NAME_MAX:
            break
        kept.append(part)
    if kept:
        body = f"{prefix} " + "+".join(kept)
        if len(kept) < len(parts) and len(f"{body}+{len(parts) - len(kept)}") <= NAME_MAX:
            body += f"+{len(parts) - len(kept)}"
    elif (t := re.sub(r"[^A-Za-z0-9._-]+", "-", tag or "").strip("-")):
        body = f"{prefix} {t}"[:NAME_MAX]
    elif prefix != "v2":
        return f"{prefix} {node_id}"[:NAME_MAX]
    else:
        return f"loop-v2-{node_id}"[:NAME_MAX]
    digits = re.sub(r"\D", "", node_id)
    for suffix in ([f" #{digits[-4:]}", f" #{digits[-3:]}"] if digits else []):
        if len(body) + len(suffix) <= NAME_MAX:
            return body + suffix
    return body


# ------------------------------------------------------------------ what changed
def _safe(fn: Callable[[], Any], default: Any = None) -> Any:
    try:
        return fn()
    except Exception:  # noqa: BLE001 - a description field never fails a submission
        return default


def _fmt(v: Any) -> str:
    return "?" if v is _MISSING else json.dumps(v) if isinstance(v, (list, tuple, dict)) else str(v)


def _git(ctx: LoopContext, *args: str) -> str:
    return subprocess.run(["git", "-C", str(ctx.spec.repo), *args], capture_output=True, text=True,
                          check=True, timeout=60).stdout


def _cfg_rel(ctx: LoopContext) -> str:
    return next((w for w in ctx.spec.surface.writable if w.endswith((".yaml", ".yml"))), "configs/main.yaml")


def actual_changes(ctx: LoopContext, store: Any, node: Any, base: str = "") -> dict[str, tuple[Any, Any]]:
    """key -> (old, new) for every flattened config key whose value differs between the root commit (or
    ``base``) and the node's OWN commit. Read from git, never accumulated from ancestors: a crossover
    parent's change the child did not take is absent."""
    root = base or getattr(store.get_node(ROOT_NODE_ID), "commit", "") or ""
    rel = _cfg_rel(ctx)
    old, new = (flatten(yaml.safe_load(_git(ctx, "show", f"{c}:{rel}")) or {}) for c in (root, node.commit))
    return {k: (old.get(k, _MISSING), new.get(k, _MISSING)) for k in [*new, *(k for k in old if k not in new)]
            if old.get(k, _MISSING) != new.get(k, _MISSING)}


def code_changes(ctx: LoopContext, store: Any, node: Any, base: str = "") -> list[str]:
    """One line per non-config file that differs between the root commit (or ``base``) and the node's own
    commit: file: functions (+added/-removed) from the unified diff's hunk headers."""
    root = base or getattr(store.get_node(ROOT_NODE_ID), "commit", "") or ""
    text = _git(ctx, "diff", root, node.commit)
    files: dict[str, dict[str, Any]] = {}
    cur = None
    for line in text.splitlines():
        if line.startswith("diff --git "):
            path = line.split(" b/", 1)[-1].strip()
            cur = None if path.endswith((".yaml", ".yml")) else files.setdefault(path, {"fn": [], "add": 0, "del": 0})
        elif cur is not None and line.startswith("@@"):
            ctxt = line.split("@@")[-1].strip()
            if m := re.match(r"(?:async\s+)?(?:def|class)\s+(\w+)", ctxt):
                ctxt = m.group(1)
            if ctxt and ctxt not in cur["fn"]:
                cur["fn"].append(ctxt[:40])
        elif cur is not None and line.startswith("+") and not line.startswith("+++"):
            cur["add"] += 1
        elif cur is not None and line.startswith("-") and not line.startswith("---"):
            cur["del"] += 1
    return [f"code: {p}" + (f" :: {', '.join(v['fn'][:6])}" if v["fn"] else "") + f" (+{v['add']}/-{v['del']} lines)"
            for p, v in files.items()]


def lineage(node: Any) -> str:
    tag, notes = node.mechanism_tag or "", getattr(node, "notes", "") or ""
    strategy = getattr(node.strategy, "value", node.strategy)
    if strategy == "design":
        if m := re.match(r"d(\d+)stacked", tag):
            return f"design batch d{m.group(1)} stacked"
        if m := re.match(r"d(\d+)pt(\d+)", tag):
            return f"design batch d{m.group(1)} point {m.group(2)}"
        return "design batch"
    if (m := re.search(r"\bxover=([^,\s]+),(\S+)", notes)):
        return f"crossover of {m.group(1)} x {m.group(2)}"
    if node.second_parent_id or strategy == "crossover":
        return f"crossover of {node.parent_id} x {node.second_parent_id}"
    if m := re.search(r"compound=(\S+)", notes):
        src = re.search(r"stack_from=(\S+)", notes)
        return f"compound ({m.group(1)})" + (f" stacked from {src.group(1)}" if src else "")
    if strategy == "moonshot":
        return "moonshot"
    return f"{strategy} from {node.parent_id}"


def carrier(store: Any, node: Any) -> Any:
    """The node its own change sits on: a crossover's first parent (the merge started from it), else its
    parent; None for a child of root."""
    m = re.search(r"\bxover=([^,\s]+),", getattr(node, "notes", "") or "")
    c = store.get_node(m.group(1) if m else node.parent_id or ROOT_NODE_ID)
    return None if c is None or c.id == ROOT_NODE_ID else c


def own_changes(ctx: LoopContext, store: Any, node: Any) -> tuple[Any, dict[str, tuple[Any, Any]], list[str]]:
    """(carrier, config key -> (old, new), code lines) of the node's OWN change: its commit vs its
    carrier's (vs root for a child of root), so a title is not the inherited stack (22/30 were
    'lr1.9e-4+N', the incumbent's knobs)."""
    c = carrier(store, node)
    base = c.commit if c is not None else ""
    return (c, _safe(lambda: actual_changes(ctx, store, node, base), {}) or {},
            _safe(lambda: code_changes(ctx, store, node, base), []) or [])


def own_line(ctx: LoopContext, store: Any, node: Any) -> str:
    """'on a node (additive_robustness_lr_stack): dino.jepa_pred_depth: 4 -> 6; code: ...'."""
    c, cfg, code = own_changes(ctx, store, node)
    what = "; ".join([*(f"{k}: {_fmt(o)} -> {_fmt(n)}" for k, (o, n) in cfg.items()), *code]) or node.mechanism_tag
    return (f"on {c.id} ({c.mechanism_tag}): " if c is not None else "on the baseline: ") + what


def changes_text(ctx: LoopContext, store: Any, node: Any) -> str:
    """The exact change vs the root baseline, from the node's own commit: a lineage line, one
    ``key: old -> new`` per config key that differs, the code files/functions touched."""
    commit = getattr(store.get_node(ROOT_NODE_ID), "commit", "") or ""
    lines = [f"vs baseline {commit[:7] or '?'} | {_safe(lambda: lineage(node), '?')}"]
    lines += [f"{k}: {_fmt(o)} -> {_fmt(n)}" for k, (o, n) in (_safe(lambda: actual_changes(ctx, store, node), {}) or {}).items()]
    lines += _safe(lambda: code_changes(ctx, store, node), []) or []
    if len(lines) == 1:
        lines.append("no config or code change recorded")
    return "\n".join(lines)


_PRINT: dict[str, tuple[str, dict[str, Any]]] = {}


def fingerprint(ctx: LoopContext, commit: str) -> tuple[str, dict[str, Any]]:
    """(the blob ids of every non-config file, the flattened config) at ``commit`` (cached: immutable)."""
    if commit not in _PRINT:
        code = " ".join(l.split()[2] for l in _git(ctx, "ls-tree", "-r", commit).splitlines()
                        if not l.split("\t")[-1].endswith((".yaml", ".yml", ".json", ".toml")))
        _PRINT[commit] = (code, flatten(yaml.safe_load(_git(ctx, "show", f"{commit}:{_cfg_rel(ctx)}")) or {}))
    return _PRINT[commit]


def near_dup(ctx: LoopContext, a: Any, b: Any) -> bool:
    """Same code and every config value equal or within novelty.major_ratio x or / (a git failure: False)."""
    from loop.novelty import minor, ratio
    try:
        (ca, fa), (cb, fb) = fingerprint(ctx, a.commit), fingerprint(ctx, b.commit)
    except Exception:  # noqa: BLE001 - an unreadable commit never blocks a submission
        return False
    near = lambda x, y: x == y or minor(ratio(x, y), ctx.cfg.novelty.major_ratio)      # noqa: E731
    return ca == cb and all(near(fa.get(k, _MISSING), fb.get(k, _MISSING)) for k in {*fa, *fb})


def mech_rivals(ctx: LoopContext, store: Any, snapshot: WorldSnapshot | None, node: Any, rivals: set[str]) -> str:
    """'' or why not: one exploratory representative per MECHANISM cluster (loop.novelty's tag + claim TF-IDF cosine
    >= novelty.mech_dup_cos; live: a node / a node / a node all went out as attention-saliency masking, a node ~ a node).
    ``rivals``: submitted nodes, kept ones queued ahead, and queued ones with a higher P(own gain>0) (the best wins).
    A later member goes only if its OWN increment beats theirs by max(labless.sibling_gain, 2 x combined sd)."""
    if (thr := ctx.cfg.novelty.mech_dup_cos) <= 0:
        return ""
    from loop.novelty import mech_vectors
    vec = mech_vectors(store)
    me = vec.get(node.id, {})
    twins = [i for i in sorted(rivals) if i != node.id and sum(x * vec.get(i, {}).get(w, 0.0) for w, x in me.items()) >= thr]
    if not twins:
        return ""

    def inc(i: str) -> tuple[float, float]:
        p = snapshot.nodes.get(i) if snapshot is not None else None
        return (0.0, 0.0) if p is None else (p.increment.mean, p.increment.sd)
    (m, sd), (tm, tsd) = inc(node.id), max(map(inc, twins))
    bar = max(ctx.cfg.labless.sibling_gain, 2.0 * (sd * sd + tsd * tsd) ** 0.5)
    return "" if m - tm >= bar else f"same mechanism as {','.join(twins)} (own gain {m - tm:+.4f} < {bar:.4f})"


def duplicate(ctx: LoopContext, store: Any, snapshot: WorldSnapshot | None, node: Any, tier: str,
              ahead: Sequence[str] = (), queued: Sequence[str] = ()) -> str:
    """'' when the node may go to Labless; else why not (operator 2026-09-30: no "lr1.9e-4 + 4", "lr1.9e-4 + 6"
    spam): an exploratory node that is a near-duplicate (:func:`near_dup`) of a submitted node or of one
    queued ahead of it, unless its shrunk panel gain over the best of them clears both labless.sibling_gain
    and its noise (2 x the pair's combined posterior sd). One representative per cluster; a distinct
    mechanism (other code) or a major value move is never a duplicate."""
    if tier == "audited":
        return ""
    book = store.get_state(K_LABLESS, {}) or {}
    subs = {k.removesuffix(EXP) for k, e in book.items() if e.get("submission_id")}
    gain = lambda i: snapshot.nodes[i].p_gain if snapshot is not None and i in snapshot.nodes else 0.0  # noqa: E731
    if why := mech_rivals(ctx, store, snapshot, node, subs | set(ahead) | {q for q in queued if gain(q) > gain(node.id)}):
        return why
    twins = [t.id for i in sorted(subs | set(ahead)) if i != node.id and (t := store.get_node(i)) is not None
             and t.commit and near_dup(ctx, t, node)]
    if not twins:
        return ""

    def est(i: str) -> tuple[float, float]:
        p = snapshot.nodes.get(i) if snapshot is not None else None
        return (0.0, 0.0) if p is None else ((q := p.confirm_estimate or p.utility).mean, q.sd)
    (m, sd), (tm, tsd) = est(node.id), max(map(est, twins))
    bar = max(ctx.cfg.labless.sibling_gain, 2.0 * (sd * sd + tsd * tsd) ** 0.5)
    if m - tm >= bar:
        return ""
    return f"near-duplicate of {','.join(twins)} (gain {m - tm:+.4f} < {bar:.4f})"


# ------------------------------------------------------------------ evidence
def audit_delta(ctx: LoopContext, store: Any, snapshot: WorldSnapshot | None,
                node_id: str) -> tuple[float, float, int] | None:
    """(mean, se, n seeds) of the node's panel delta vs root on the held-out audit seeds, each seed's
    root deviation shrunk away exactly as the audit gate does; None without >= 1 paired seed."""
    by_node: dict[str, list[Run]] = {}
    for r in store.runs(stage=Stage.AUDIT):
        by_node.setdefault(r.node_id, []).append(r)
    idx = lambda rs: {int(r.seed): r for r in rs if r.status is RunStatus.DONE}  # noqa: E731
    mine, roots = idx(by_node.get(node_id, [])), idx(by_node.get(ROOT_NODE_ID, []))
    offsets: dict[int, float] = {}
    if snapshot is not None:
        est = {k: ((q := p.confirm_estimate or p.utility).mean, q.sd ** 2) for k, p in snapshot.nodes.items()}
        offsets = _safe(lambda: audit_offsets(node_id, by_node, ctx.cfg, est,
                                              float(snapshot.sigma.get(PANEL, ctx.cfg.panel_sd))), {}) or {}
    cs = [c - offsets.get(s, 0.0) for s in sorted(mine) if s in roots
          and (c := _panel(ctx.cfg, mine[s], roots[s])) is not None]
    if not cs:
        return None
    se = statistics.stdev(cs) / len(cs) ** 0.5 if len(cs) > 1 else float("nan")
    return statistics.fmean(cs), se, len(cs)


def gate_line(g: Mapping[str, Any]) -> str:
    """The audit verdict's frozen numbers (K_AUDIT["gate"][node]) as one note line."""
    f = lambda v, spec: "?" if v is None else format(v, spec)       # noqa: E731
    se = f" +- {f(g.get('se'), '.4f')} se" if g.get("se") is not None else ""
    alpha = g.get("alpha")
    line = (f"held-out audit (gate): panel delta vs baseline {f(g.get('mean'), '+.4f')}{se} over "
            f"{g.get('n', '?')} audit seeds, E={f(g.get('e'), '.3g')}"
            + (f" vs bar {1 / alpha:.3g} (alpha {alpha:g})" if alpha else "")
            + f" -> {g.get('verdict', '?')}; frozen claim {f(g.get('delta'), '+.5f')}, sigma {f(g.get('sigma'), '.5f')}")
    if g.get("source", "gate") != "gate":
        line += f" [{g['source']} after the verdict: {g.get('why', '')}]".replace(": ]", "]")
    return line


AUDIT_START = re.compile(r"audit started for (\S+) \(claimed gain ([-+0-9.eE]+)\)")


def backfill_audit_gate(ctx: LoopContext, store: Any, snapshot: WorldSnapshot | None,
                        log_path: str | Path | None = None) -> list[str]:
    """Freeze the gate numbers of audits decided before they were stored (a node, a node on
    2026-09-30). Only runs finished by the verdict count. With no other node's audit run finished
    by then, every shared offset was 0 (the prior): the deltas are exact, E comes from the verdict
    event, the claim from the 'audit started' log line, and sigma is solved from E ('reconstructed').
    Otherwise the offsets depend on estimates the gate never stored: re-fit now, sigma unknown
    ('recomputed')."""
    book = store.get_state(K_AUDIT, {}) or {}
    gate, out = dict(book.get("gate") or {}), []
    todo = [n for n in [*book.get("passed", []), *book.get("failed", [])] if n not in gate]
    if not todo:
        return out
    verdicts = {e["node_id"]: e for e in store.events("audit")}
    fin = {e["run_id"]: e["ts"] for e in store.events("run_finished") if e.get("run_id")}
    claims: dict[str, float] = {}
    path = Path(log_path) if log_path else Path(ctx.state_dir) / "loop.log"
    try:
        for m in AUDIT_START.finditer(path.read_text(errors="replace")):
            claims[m.group(1)] = float(m.group(2))
    except OSError:
        pass
    sigma_p = float(snapshot.sigma.get(PANEL, ctx.cfg.panel_sd)) if snapshot is not None else ctx.cfg.panel_sd
    alpha = ctx.cfg.promotion.alpha_audit
    for nid in todo:
        ev = verdicts.get(nid)
        if ev is None:
            continue
        t = float(ev["ts"])
        by_node: dict[str, list[Run]] = {}
        for r in store.runs(stage=Stage.AUDIT):
            if r.status is RunStatus.DONE and fin.get(r.id, float("inf")) <= t:
                by_node.setdefault(r.node_id, []).append(r)
        sibs = [k for k in by_node if k not in (nid, ROOT_NODE_ID)]
        offsets: dict[int, float] = {}
        if sibs and snapshot is not None:
            est = {k: ((q := p.confirm_estimate or p.utility).mean, q.sd ** 2) for k, p in snapshot.nodes.items()}
            offsets = _safe(lambda: audit_offsets(nid, by_node, ctx.cfg, est, sigma_p), {}) or {}
        cs = audit_deltas(by_node.get(nid, []), by_node.get(ROOT_NODE_ID, []), cfg=ctx.cfg, offsets=offsets)
        claim = claims.get(nid)
        delta = None if claim is None else max(claim, ctx.cfg.promotion.delta_min_sigmas * sigma_p)
        e_value, sigma = float(ev["e_value"]), None
        exact = not sibs
        if exact and delta and e_value > 1.0 and cs:
            num = sum(c * delta - delta * delta / 2 for c in cs.values())
            sigma = (num / math.log(e_value)) ** 0.5 if num > 0 else None
        g = audit_gate(list(cs.values()), seeds=list(cs), verdict=ev.get("verdict", "?"), e_value=e_value,
                       delta=delta, sigma=sigma, alpha=alpha, offsets=offsets, ts=t,
                       source="reconstructed" if exact else "recomputed")
        g["why"] = ("no other audit had finished: offsets 0, deltas exact; E from the verdict; claim from the "
                    "log (5 dp); sigma solved from E" if exact else
                    f"offsets from {','.join(sorted(sibs))} re-fit with today's estimates (the gate's were not "
                    f"stored); E from the verdict; claim from the log; sigma not recoverable")
        gate[nid] = g
        store.log("audit_gate_backfilled", node_id=nid, source=g["source"], mean=g["mean"], se=g["se"], n=g["n"])
        out.append(f"audit gate numbers backfilled for {nid} ({g['source']}): {gate_line(g)}")
    if out:
        store.set_state(K_AUDIT, {**book, "gate": gate})
    return out


def own_effect(snapshot: WorldSnapshot | None, node: Any) -> tuple[str, Any, dict[str, tuple[float, float]]]:
    """(measure parent, own panel increment posterior, axis -> (own delta, sd)) of ``node`` over its measure parent
    (the nearest contained ancestor the world model credits it against; else its parent): the node's OWN effect,
    not the stack it inherits (operator 2026-10-01: notes quoting U vs root overstated it ~6x)."""
    p = snapshot.nodes.get(node.id) if snapshot else None
    mp = ((getattr(snapshot, "measured", None) or {}).get(node.id) or (node.parent_id or ROOT_NODE_ID,))[0]
    if p is None:
        return mp, None, {}
    q, zero = snapshot.nodes.get(mp), Posterior(0.0, 0.0)
    return mp, p.increment, {a: (g.mean - (b := (q.axes.get(a) if q else None) or zero).mean,
                                 (g.sd ** 2 + b.sd ** 2) ** 0.5) for a, g in p.axes.items()}


def evidence_lines(ctx: LoopContext, store: Any, snapshot: WorldSnapshot | None, node: Any) -> list[str]:
    p = snapshot.nodes.get(node.id) if snapshot else None
    n_done = lambda st: sum(r.status is RunStatus.DONE for r in store.runs(node_id=node.id, stage=st))  # noqa: E731
    out = []
    out.append(f"screen: P(own gain>0)={p.p_gain:.3f} over {n_done(Stage.SCREEN)} seeds" if p else "screen: ?")
    out.append(f"promote: {n_done(Stage.PROMOTE)} seeds" if n_done(Stage.PROMOTE) else "promote: none")
    conf = f"confirm: {n_done(Stage.CONFIRM)} seeds"
    if node.confirm_e is not None:
        conf += f", e-value {node.confirm_e:.3g}"
    if node.confirm_alt is not None:
        conf += f" (frozen claim {node.confirm_alt:+.4f})"
    out.append(conf)
    gate = ((store.get_state(K_AUDIT, {}) or {}).get("gate") or {}).get(node.id)
    if gate:
        out.append(gate_line(gate))
    else:
        a = _safe(lambda: audit_delta(ctx, store, snapshot, node.id))
        out.append("held-out audit (live, no verdict yet): panel delta vs baseline (total, incl. inherited) "
                   + (f"{a[0]:+.4f} +- {a[1]:.4f} se over {a[2]} audit seeds" if a and a[1] == a[1]
                      else f"{a[0]:+.4f} over {a[2]} audit seed" if a else "?"))
    mp, inc, _ = own_effect(snapshot, node)
    out.append(f"OWN effect: panel increment vs {mp} {inc.mean:+.4f} +- {inc.sd:.4f} (shrunk; this change alone)"
               if inc is not None else "OWN effect: ?")
    est = p and (p.confirm_estimate or p.utility)
    out.append(f"total vs root (incl. the inherited stack): {est.mean:+.4f} +- {est.sd:.4f}, P(U>0)={p.p_positive:.3f}"
               if est else "total vs root: ?")
    return out


def note(store: Any, snapshot: WorldSnapshot | None, node: Any, ctx: LoopContext | None = None,
         tier: str = "") -> str:
    """The submission notes: tier, hypothesis, what changed, evidence, per-axis deltas, provenance, how to
    reproduce. Every number is computed from the store / snapshot; a missing field prints '?'."""
    h = _safe(lambda: store.get_hypothesis(node.hypothesis_id or ""))
    p = snapshot.nodes.get(node.id) if snapshot else None
    changes = _safe(lambda: changes_text(ctx, store, node), "") if ctx is not None else ""
    cl = changes.splitlines()
    mp, _, own = own_effect(snapshot, node)
    axes = "  ".join(f"{a.removeprefix('final_probe_')} {m:+.4f}+-{sd:.4f}" for a, (m, sd) in own.items()) or "?"
    xo = re.search(r"\bxover=([^,\s]+),(\S+)", getattr(node, "notes", "") or "")
    parents = ", ".join(xo.groups() if xo else [x for x in (node.parent_id, node.second_parent_id) if x]) or "root"
    keys = [l for l in cl[1:] if not l.startswith("code:") and ": " in l]
    overrides = " ".join(f"{l.split(': ', 1)[0]}={l.rsplit(' -> ', 1)[-1]}" for l in keys)
    base = cl[0].split()[2] if cl and len(cl[0].split()) > 2 else "?"
    return "\n".join([
        *([f"Tier: {tier}"] if tier else []),
        f"Change: {(_safe(lambda: own_line(ctx, store, node), '?') if ctx is not None else '?')[:600]}",
        f"Hypothesis: {(h.claim if h else '?')[:400]}",
        f"Family: {node.family}/{node.mechanism_tag}",
        "What changed vs the baseline (the whole stack):", *[f"  {l}" for l in (cl[:14] or ["?"])],
        "Evidence:", *[f"  {l}" for l in _safe(lambda: evidence_lines(ctx, store, snapshot, node), ["?"])],
        f"Per-axis OWN deltas vs {mp or '?'} (shrunk): {axes}",
        f"Provenance: loop/v2 auto-research (overfitx); node {node.id}; commit {node.commit[:12]}; "
        f"parent(s) {parents}; base {base}",
        f"Reproduce: python train.py configs/main.yaml" + (f" with {overrides}" if overrides else "")
        + f" (or commit {node.commit[:12]})",
    ])[:6000]


def pick_run(store: Any, node_id: str, stages: Sequence[Stage] = (Stage.AUDIT,)) -> Run | None:
    """The median-scoring finished run of ``stages`` (default the audit seeds; lower median on an even count)."""
    done = [r for st in stages for r in store.runs(node_id=node_id, stage=st)
            if r.status is RunStatus.DONE and r.output_dir]
    if not done:
        return None
    score = lambda r: r.aggregate if r.aggregate is not None else statistics.fmean(r.scores.values())  # noqa: E731
    return sorted(done, key=lambda r: (score(r), r.seed))[(len(done) - 1) // 2]


def review_config(ctx: Any, out_dir: str) -> str:
    """The repo-relative config Labless reviews (the submitter's ``review_config``). A run launched from a
    config copy outside the repo (resume / eval-only: summary.config_path is absolute) left its effective
    config -- seed, output_dir, every override -- as labless_source/configs/<copy>.yaml; it is written over
    the snapshot's configs/main.yaml with the copy's own knobs (RESUME_KEYS) put back to the repo's: the
    exact snapshot a plain launch leaves (train.py overwrites configs/main.yaml with the effective config)."""
    rel = _safe(lambda: _cfg_rel(ctx), "configs/main.yaml")
    src = Path(out_dir) / "labless_source"
    try:
        path = str(json.loads((Path(out_dir) / "summary.json").read_text()).get("config_path") or "")
    except (OSError, ValueError, AttributeError):
        return rel
    eff, cfg = src / "configs" / Path(path).name, src / rel
    if not path or re.search(r"(?:^|/)configs/[^/]+\.ya?ml$", path) or not eff.is_file() or not cfg.is_file():
        return rel
    tree, repo = yaml.safe_load(eff.read_text()) or {}, flatten(yaml.safe_load(cfg.read_text()) or {})
    for key in RESUME_KEYS:
        if key in repo:
            set_dotted(tree, key, repo[key])
        elif isinstance(tree.get(key.split(".")[0]), dict):
            tree[key.split(".")[0]].pop(key.split(".", 1)[1], None)
    cfg.write_text(yaml.safe_dump(tree, sort_keys=False))
    return rel


def _run_sha(out_dir: str) -> str:
    """The commit a run recorded in its summary.json (``git.sha``); '' if unreadable."""
    try:
        git = json.loads((Path(out_dir) / "summary.json").read_text()).get("git") or {}
    except (OSError, ValueError, AttributeError):
        return ""
    return str(git.get("sha") or "")


def push_commit(ctx: LoopContext, out_dir: str, node_id: str = "") -> str:
    """``labless.push``: publish the run's commit to ``push.remote`` as ``push.ref`` ({node} -> the node id, else the
    run directory's name) so the submission's repo remote/commit resolves. A failed push raises."""
    push = dict(getattr(ctx.cfg.labless, "push", None) or {})
    remote = str(push.get("remote") or "")
    if not remote:
        return ""
    sha = _run_sha(out_dir) or "HEAD"
    ref = str(push.get("ref") or "loop/{node}").format(node=node_id or Path(out_dir).name)
    cp = subprocess.run(["git", "-C", str(ctx.spec.repo), "push", remote, f"{sha}:refs/heads/{ref}"],
                        capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=300,
                        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})     # never block on a credential prompt
    if cp.returncode != 0:
        raise RuntimeError(f"git push {remote} {sha[:10]}:{ref} failed (exit {cp.returncode}): "
                           f"{(cp.stderr or cp.stdout)[-300:]}")
    return ref


def remote_url(repo: str, remote: str) -> str:
    """``remote``'s url from ``repo``'s .git/config (no subprocess); the remote name itself when unreadable."""
    try:
        cfg = (Path(repo) / ".git" / "config").read_text()
    except OSError:
        return remote
    m = re.search(rf'\[remote "{re.escape(remote)}"\][^\[]*?^\s*url\s*=\s*(\S+)', cfg, re.M | re.S)
    return m.group(1) if m else remote


def code_note(ctx: LoopContext, out_dir: str, ref: str) -> str:
    """M5: where the submitted code lives (the push remote's url, the pushed ref, the commit) for the notes."""
    remote = str((dict(getattr(ctx.cfg.labless, "push", None) or {})).get("remote") or "")
    sha = _run_sha(out_dir)
    return f"code: {remote_url(str(ctx.spec.repo), remote)} {ref}" + (f" @ {sha[:12]}" if sha else "")


def _named_config(out_dir: str, name: str, text: str) -> bytes | None:
    """``labless.name_into_config``: write ``name`` (<= NAME_MAX) and ``notes`` into the run's config.yaml (the
    submitter reads both from it); the original bytes, to restore."""
    path = Path(out_dir) / "config.yaml"
    raw = path.read_bytes()
    tree = yaml.safe_load(raw) or {}
    tree["name"], tree["notes"] = name[:NAME_MAX], text
    path.write_text(yaml.safe_dump(tree, sort_keys=False))
    return raw


def submit_run(ctx: LoopContext, out_dir: str, name: str, text: str, changes: str = "", node_id: str = "") -> str:
    """Run the repo's submitter non-interactively; the submission_id, or RuntimeError. ``labless.submit_cmd``
    empty = nanopath's argv (byte-identical); else the formatted command ({python} {out_dir} {repo}) after the
    optional push (``labless.push``) and with name/notes in the run's config (``labless.name_into_config``)."""
    if os.environ.get("LOOP_FORBID_SLURM"):          # the test-suite guard: never a real submission
        raise RuntimeError("real Labless submission refused under LOOP_FORBID_SLURM (tests)")
    cfg, repo = ctx.cfg.labless, ctx.spec.repo
    python = cfg.python or f"{repo}/.venv/bin/python"
    submit_cmd = list(getattr(cfg, "submit_cmd", None) or [])
    restore: bytes | None = None
    if submit_cmd:
        if ref := push_commit(ctx, out_dir, node_id):
            text = f"{text}\n\n{code_note(ctx, out_dir, ref)}" if text else code_note(ctx, out_dir, ref)
        argv = [a.format(python=python, out_dir=out_dir, repo=repo) for a in submit_cmd]
        if getattr(cfg, "name_into_config", False):
            restore = _named_config(out_dir, name, text)
    else:
        argv = [python, f"{repo}/labless/submit_to_labless.py",
                f"output_dir={out_dir}", f"run_name={name}", f"notes={text}",
                *([f"changes={changes}"] if changes else []),
                f"review_config={review_config(ctx, out_dir)}",
                f"github_token_file={Path(cfg.token_file).expanduser()}"]
    try:
        cp = subprocess.run(argv, cwd=repo, capture_output=True, text=True, stdin=subprocess.DEVNULL,
                            timeout=60 * cfg.timeout_min, env={**os.environ, "LABLESS_AUTOSUBMIT_FILE": ""})
    finally:
        if restore is not None:
            (Path(out_dir) / "config.yaml").write_bytes(restore)
    try:
        receipt = json.loads((Path(out_dir) / "labless_submission.json").read_text())
    except (OSError, ValueError):
        receipt = {}
    errors = (receipt.get("run") or {}).get("validation_errors")
    sid = receipt.get("submission_id")
    if cp.returncode != 0 or errors or not sid:
        raise RuntimeError(f"exit {cp.returncode}; validation_errors={errors}; "
                           f"{(cp.stderr or cp.stdout)[-400:]}")
    return str(sid)


def landed(out_dir: str, since: float) -> str:
    """The submission_id of a clean ``labless_submission.json`` written at/after ``since``; '' if none."""
    path = Path(out_dir) / "labless_submission.json"
    try:
        if path.stat().st_mtime < since:
            return ""
        receipt = json.loads(path.read_text())
    except (OSError, ValueError):
        return ""
    ok = isinstance(receipt, dict) and not (receipt.get("run") or {}).get("validation_errors")
    return str(receipt.get("submission_id") or "") if ok else ""


class LablessSubmitter:
    """One submission in flight at a time, in ``pool``; state lives in the store (resumable)."""

    def __init__(self, ctx: LoopContext, store: Any, pool: Any, clock: Callable[[], float],
                 submit: Callable[..., str] = submit_run) -> None:
        self.ctx, self.store, self.pool, self.clock, self.submit = ctx, store, pool, clock, submit
        self._fut: tuple[Future, str] | None = None
        self._cleaned = -1e18                            # when stale failed entries were last cleared
        book = store.get_state(K_LABLESS, {}) or {}     # refused for a config copy's path (fixed): retry now
        if stale := sorted(k for k, e in book.items() if not e.get("submission_id")
                           and CONFIG_PATH_ERR in (e.get("error") or "")):
            for k in stale:
                book[k]["next_ts"] = 0.0
            store.set_state(K_LABLESS, book)
            store.log("labless_backoff_cleared", nodes=stale)

    def step(self, passed: Sequence[str], snapshot: WorldSnapshot | None) -> list[str]:
        cfg, book, now = self.ctx.cfg.labless, self.store.get_state(K_LABLESS, {}), self.clock()
        new = {n: {"tries": 0, "error": "", **e} for n, e in cfg.recorded.items()
               if not book.get(n, {}).get("submission_id")}
        if new:                                          # submitted by hand: record, never resubmit
            book.update(new)
            self.store.set_state(K_LABLESS, book)
            self.store.log("labless_recorded", nodes=sorted(new))
        if self._fut is not None:
            fut, nid = self._fut
            if not fut.done():
                return []
            self._fut, e = None, book[nid]
            try:
                e.update(submission_id=fut.result(), error="")
                self.store.log("labless_submitted", node_id=nid.removesuffix(EXP), tier=e.get("tier", "audited"),
                               submission_id=e["submission_id"])
                msg = f"labless: {nid} submitted, submission_id={e['submission_id']}"
            except Exception as exc:  # noqa: BLE001 - logged, retried after backoff
                e["error"] = repr(exc)[:500]
                self.store.log("labless_failed", node_id=nid.removesuffix(EXP), tier=e.get("tier", "audited"),
                               tries=e["tries"], error=e["error"])
                msg = f"labless: {nid} failed (try {e['tries']}), retry after {e['next_ts'] - now:.0f}s"
            self.store.set_state(K_LABLESS, book)
            return [msg]
        out = self._clear_stale(passed, snapshot, now) if cfg.enabled and now - self._cleaned >= 3600.0 else []
        book = self.store.get_state(K_LABLESS, {}) or {}
        last = max((e.get("last_ts", 0.0) for e in book.values()), default=None)
        if not cfg.enabled or (last is not None and now - last < 3600 * cfg.min_hours_between):
            return out
        recent = [t for t in self.store.get_state(K_LABLESS_TS, []) or [] if now - t < 86400.0]
        if cfg.max_per_day > 0 and len(recent) >= cfg.max_per_day:
            return out
        todo = [(nid, nid, "audited", "audited: passed the held-out audit", "") for nid in passed]
        todo += [(nid, nid + EXP, "exploratory", why, gid)
                 for nid, why, gid in exploratory_queue(self.ctx, self.store, snapshot, now)]
        ahead: list[str] = []                    # kept confirmed nodes: one representative per near-dup cluster
        queued = [nid for nid, _, tier, _, gid in todo if tier == "exploratory" and not gid]
        for nid, key, tier, why, gid in todo:
            e = book.get(key, {})
            node = self.store.get_node(nid)
            run = pick_run(self.store, nid, (Stage.AUDIT,) if tier == "audited" else LIFECYCLE)
            if e.get("submission_id") or run is None or node is None \
                    or (tier == "audited" and node.status is not NodeStatus.CONFIRMED) \
                    or (not gid and duplicate(self.ctx, self.store, snapshot, node, tier, ahead, queued)):
                continue                         # a group's representative IS its dedupe: one per group
            ahead += [nid] if tier == "exploratory" and not gid else []
            if now < e.get("next_ts", 0.0):
                continue
            if e.get("last_ts") and (sid := landed(run.output_dir, e["last_ts"])):
                book[key] = {**e, "run_id": run.id, "submission_id": sid, "error": ""}   # a restart
                self.store.set_state(K_LABLESS, book)                     # mid-submission: never twice
                self.store.log("labless_recovered", node_id=nid, tier=tier, submission_id=sid)
                return [f"labless: {key} already submitted before a restart, submission_id={sid}"]
            tries = int(e.get("tries", 0)) + 1     # backoff is set up front: a crash mid-call backs off too
            book[key] = {**e, "tries": tries, "last_ts": now, "run_id": run.id, "submission_id": None, "tier": tier,
                         **({"group": gid} if gid else {}), "next_ts": now + 3600 * min(MAX_BACKOFF_H, max(cfg.min_hours_between, MIN_BACKOFF_H) * 2 ** (tries - 1))}
            self.store.set_state(K_LABLESS, book)
            self.store.set_state(K_LABLESS_TS, [*recent, now])
            self.store.log("labless_submit", node_id=nid, run_id=run.id, tries=tries, tier=tier)
            _, own, code = own_changes(self.ctx, self.store, node)     # the title is the node's OWN change
            cfg_diff = {k: n for k, (_, n) in own.items() if n is not _MISSING and not code}
            self._fut = (self.pool.submit(self.submit, self.ctx, run.output_dir,
                                          run_name(nid, cfg_diff, node.mechanism_tag, TIER_PREFIX[tier]),
                                          note(self.store, snapshot, node, self.ctx, tier=why),
                                          _safe(lambda: own_line(self.ctx, self.store, node) + "\n"
                                                + changes_text(self.ctx, self.store, node), "")), key)
            return out + [f"labless: submitting {key} ({tier}) from {run.id} (try {tries})"]
        return out

    def _clear_stale(self, passed: Sequence[str], snapshot: WorldSnapshot | None, now: float) -> list[str]:
        """Hourly: a FAILED entry (no submission_id) whose node no tier would send any more -- an audited entry of a
        node no longer CONFIRMED, an exploratory one neither confirmed nor its group's representative (old-rule
        per-node entries, a group represented since) -- leaves the book (and ``loop status``); never a recorded one."""
        self._cleaned = now
        book, cfg = self.store.get_state(K_LABLESS, {}) or {}, self.ctx.cfg.labless
        live = {nid for nid in passed if getattr(self.store.get_node(nid), "status", None) is NodeStatus.CONFIRMED}
        live |= {nid + EXP for nid, _, _ in exploratory_queue(self.ctx, self.store, snapshot, now)}
        stale = sorted(k for k, e in book.items() if not e.get("submission_id") and k not in live
                       and k not in cfg.recorded and (self._fut is None or self._fut[1] != k))
        if not stale:
            return []
        self.store.set_state(K_LABLESS, {k: e for k, e in book.items() if k not in stale})
        self.store.log("labless_stale_cleared", keys=stale, errors={k: str(book[k].get("error", ""))[:200] for k in stale})
        return [f"labless: cleared {len(stale)} stale failed entr{'y' if len(stale) == 1 else 'ies'} ({', '.join(stale)})"]


def _gain_of(snapshot: WorldSnapshot | None, nid: str) -> float | None:
    """The node's total panel estimate vs root (confirm estimate, else utility); root 0; None unscored."""
    if nid == ROOT_NODE_ID:
        return 0.0
    p = snapshot.nodes.get(nid) if snapshot is not None else None
    return None if p is None else (p.confirm_estimate or p.utility).mean


def closed_groups(ctx: LoopContext, store: Any, snapshot: WorldSnapshot | None) -> list[dict[str, Any]]:
    """Every CLOSED tested group, its members ranked best first by own gain (operator 2026-10-05: "only send
    potentially valid / best-of-some-tested-group (even if it did bad) instead of spam"):
      ladder / tune pass  a closed dose ladder or tuning pass (loop.tune): the base and its points / trials, gain vs
                          the base's carrier (U - U_carrier: they share it);
      legs               a closed leg-factorial round (loop.legs): its combinations, gain vs the round's carrier;
      design             a closed design batch (loop.design): its points, gain vs their parent (the control row);
      mechanism          the nodes of one normalized mechanism tag (loop.novelty.norm_tag), none in a batch above,
                          all closed: gain = each one's own increment over its measure parent.
    Each: {id, kind, members [(nid, gain)], ref, closed ts}, with >= labless.min_group members; a member needs a
    posterior; the carrier / control is never a member."""
    from loop.contracts import CLOSED_STATUSES
    from loop.novelty import norm_tag
    from loop.design import K_DESIGN
    from loop.legs import K_LEGS
    from loop.tune import K_TUNE
    out: list[dict[str, Any]] = []
    upd = lambda ids: max((getattr(store.get_node(i), "updated_ts", 0.0) or 0.0 for i in ids), default=0.0)  # noqa: E731

    def add(gid: str, kind: str, ids: Sequence[str], ref: str, closed: float | None) -> None:
        r = _gain_of(snapshot, ref) or 0.0
        mem = sorted(((i, g - r) for i in dict.fromkeys(ids) if i not in (ref, ROOT_NODE_ID)
                      and (g := _gain_of(snapshot, i)) is not None), key=lambda m: -m[1])
        if len(mem) >= ctx.cfg.labless.min_group:      # one tested point is not a group
            out.append({"id": gid, "kind": kind, "members": mem, "ref": ref, "closed": closed or upd(ids)})
    batched: set[str] = set()
    for b in (store.get_state(K_TUNE, {}) or {}).get("batches", []):
        base = store.get_node(b.get("base") or "")
        if base is None:
            continue
        ids = [base.id, *(b.get("points") or {}).values(), *(b.get("trials") or [])]
        batched.update(ids[1:])
        c = carrier(store, base)
        add(f"t{b.get('batch')}", "dose ladder" if b.get("kind") == "ladder" else "tuning pass", ids,
            c.id if c is not None else ROOT_NODE_ID, b.get("closed"))
    for b in (store.get_state(K_LEGS, {}) or {}).get("batches", []):
        ids = [i for d in (b.get("designs") or {}).values() for i in d["points"].values()]
        batched.update(ids)
        add(f"L{b.get('batch')}", "leg-factorial round", ids, b.get("carrier") or ROOT_NODE_ID, b.get("closed"))
    for b in (store.get_state(K_DESIGN, {}) or {}).get("batches", []):
        ids = [i for i in (b.get("points") or {}).values() if i != ROOT_NODE_ID]
        batched.update(ids)
        first = store.get_node(ids[0]) if ids else None
        add(f"d{b.get('batch')}", "design batch", ids, getattr(first, "parent_id", "") or ROOT_NODE_ID, b.get("closed"))
    clusters: dict[str, list[Any]] = {}
    for n in store.nodes():
        if n.id != ROOT_NODE_ID and n.id not in batched and (t := norm_tag(n.mechanism_tag)):
            clusters.setdefault(t, []).append(n)
    for tag, ns in clusters.items():
        inc = {n.id: snapshot.nodes[n.id].increment.mean for n in ns if snapshot is not None and n.id in snapshot.nodes}
        if len(inc) < ctx.cfg.labless.min_group or any(n.status not in CLOSED_STATUSES for n in ns):
            continue
        out.append({"id": f"mech:{tag}", "kind": "mechanism cluster", "ref": "",
                    "members": sorted(inc.items(), key=lambda m: -m[1]), "closed": upd([n.id for n in ns])})
    return out


def group_line(g: Mapping[str, Any], nid: str) -> str:
    """The notes' tier line of a group representative: which group, its size, how the representative ranked."""
    mem = g["members"]
    rank = next(i for i, (m, _) in enumerate(mem, 1) if m == nid)
    gain = dict(mem)[nid]
    vs = f"vs {g['ref']}" if g["ref"] else "own increment vs measure parent"
    ranked = ", ".join(f"{m} {x:+.4f}" for m, x in mem[:8]) + (f", +{len(mem) - 8} more" if len(mem) > 8 else "")
    return (f"exploratory: best of closed {g['kind']} {g['id']} ({len(mem)} tested), ranked {rank}/{len(mem)} by "
            f"gain {vs}: {gain:+.4f}" + ("" if gain > 0 else " -- the group did POORLY (best gain <= 0); "
                                         "submitted as the group's representative, not as a win")
            + f"; group ranking: {ranked}")


def exploratory_queue(ctx: LoopContext, store: Any, snapshot: WorldSnapshot | None,
                      now: float | None = None) -> list[tuple[str, str, str]]:
    """(node id, tier line, group id) of the exploratory tier, in submission order: every CONFIRMED node (by
    P(U>U_inc); group ''), then ONE representative per CLOSED tested group (:func:`closed_groups`) -- its best
    member, even when that did poorly -- best gain first, the group closed within labless.group_max_age_h of
    ``now``. A group with a book entry keeps that entry's node (no second submission when the ranking shifts).
    ONE REPRESENTATIVE PER GROUP, enforced: a group any member of which was already submitted (either tier, any
    era -- live: 30 per-node exploratory submissions of not-confirmed nodes before the 10-05 rule) sends nothing more.
    Never a vetoed confirmed node, nor one already submitted in either tier."""
    cfg, book = ctx.cfg.labless, store.get_state(K_LABLESS, {}) or {}
    if not cfg.exploratory:
        return []
    posts = snapshot.nodes if snapshot is not None else {}
    sent = lambda i: any((book.get(k) or {}).get("submission_id") for k in (i, i + EXP))  # noqa: E731
    conf = sorted((-(posts[n.id].p_beats_incumbent if n.id in posts else 0.0), n.id)
                  for n in store.nodes(status=[NodeStatus.CONFIRMED])
                  if n.id != ROOT_NODE_ID and not (n.id in posts and posts[n.id].vetoed) and not sent(n.id))
    out = [(nid, f"exploratory: confirmed on fresh seeds (E={store.get_node(nid).confirm_e or 0:.3g}), "
                 "held-out audit pending", "") for _, nid in conf]
    if snapshot is None:
        return out
    taken = {e["group"]: k.removesuffix(EXP) for k, e in book.items() if e.get("group")}
    subm = {k.removesuffix(EXP) for k, e in book.items() if e.get("submission_id")}
    reps = []
    for g in closed_groups(ctx, store, snapshot):
        if now is not None and cfg.group_max_age_h > 0 and now - g["closed"] > 3600 * cfg.group_max_age_h:
            continue
        nid = taken.get(g["id"]) or g["members"][0][0]
        if nid not in dict(g["members"]) or sent(nid) or subm & set(dict(g["members"])):
            continue                                   # a member went (any tier / era): the group is represented
        reps.append((-dict(g["members"])[nid], g["id"], nid, group_line(g, nid)))
    seen = {nid for nid, _, _ in out}
    for _, gid, nid, why in sorted(reps):
        if nid not in seen:
            seen.add(nid)
            out.append((nid, why, gid))
    return out


def gate_state(ctx: LoopContext, store: Any) -> str:
    """One phrase: where the audit -> Labless gate stands."""
    a, book = store.get_state(K_AUDIT, {}), store.get_state(K_LABLESS, {})
    if not ctx.cfg.labless.enabled:
        return "DISABLED"
    if (cur := a.get("current")):
        return f"auditing {cur['node']}" + (f" (+{len(a['queue']) - 1} queued)" if len(a.get("queue", [])) > 1 else "")
    if (todo := [n for n in a.get("passed", []) if not book.get(n, {}).get("submission_id")
                 and getattr(store.get_node(n), "status", None) is NodeStatus.CONFIRMED]):
        e = book.get(todo[0], {})
        return f"{todo[0]} passed audit, submission " + (f"retrying ({e['error'][:60]})" if e.get("error") else "pending")
    done = [n for n, e in book.items() if e.get("submission_id")]
    return (f"idle, last {done[-1]}={book[done[-1]]['submission_id']}" if done else "idle") + \
        f", {len(a.get('failed', []))} audit(s) failed"


def status_lines(ctx: LoopContext, store: Any) -> list[str]:
    book = store.get_state(K_LABLESS, {})
    head = f"labless  {gate_state(ctx, store)}; " \
           f"{sum(bool(e.get('submission_id')) for e in book.values())} submitted"
    return [head] + [f"  {nid} " + (f"submission_id={e['submission_id']} ({e.get('run_id', '?')})"
                                   if e.get("submission_id") else
                                   f"pending: {e.get('tries', 0)} tries, {e.get('error', '')[:120]}")
                     for nid, e in book.items()]
