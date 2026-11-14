"""loop.quality — ideation quality, COMPUTED from the ledger (never LLM-judged): per brief kind, bandit
arm and LLM backend. ``quality_table`` is for ``loop status`` and the big-review context;
``ideation_lessons`` is the short text the researcher and review prompts are given."""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter
from typing import Any, Callable, Sequence

from loop.contracts import CARRIERS, PANEL, ROOT_NODE_ID, NodeStatus, WorldSnapshot, outage_requests
from loop.select.bandit import brief_of, kind_pulls

_SCREENED = frozenset(NodeStatus(s) for s in (
    "screened_out promoting confirming confirmed not_confirmed audit_failed vetoed").split())
_DUP = ("near_dup", "exact_dup")


def backend_of(node: Any) -> str:
    m = re.search(r"\bbackend=(\S+)", node.notes or "")
    return m.group(1) if m else "?"


def arm_of(node: Any) -> str:
    """The bandit arm, with a moonshot tuning pass (its trials and its tuned node, loop.tune) apart."""
    return "moonshot_tuned" if re.search(r"\btune(_of)?=", node.notes or "") else node.strategy.value


def _rank(xs: Sequence[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0
        i = j + 1
    return r


def _pearson(a: Sequence[float], b: Sequence[float]) -> float:
    try:
        return statistics.correlation(a, b)
    except statistics.StatisticsError:          # a constant series
        return math.nan


def predictions(store: Any, snapshot: WorldSnapshot, nodes: Sequence[Any]) -> list[tuple[float, float]]:
    """(registered predicted panel delta, measured shrunk increment over the parent) per scored node."""
    out = []
    for n in nodes:
        h, p = store.get_hypothesis(n.hypothesis_id or ""), snapshot.nodes.get(n.id)
        pred = (h.predictions.get(PANEL) or next(iter(h.predictions.values()), None)) if h else None
        if pred is not None and p is not None and p.utility.n > 0:
            out.append((pred.sign * pred.size, p.increment.mean))
    return out


def prediction_stats(pairs: Sequence[tuple[float, float]]) -> dict[str, float]:
    """sign accuracy, mean bias (pred - measured), optimism (mean pred / mean measured, if > 0),
    Pearson and Spearman once n >= 5."""
    if not pairs:
        return {"n": 0}
    pred, meas = [p for p, _ in pairs], [m for _, m in pairs]
    mp, mm = statistics.fmean(pred), statistics.fmean(meas)
    out = {"n": len(pairs), "sign_acc": sum((p > 0) == (m > 0) for p, m in pairs) / len(pairs),
           "bias": mp - mm, "mean_pred": mp, "mean_meas": mm,
           "optimism": mp / mm if mm > 0 else math.nan}
    if len(pairs) >= 5:
        out.update(pearson=_pearson(pred, meas), spearman=_pearson(_rank(pred), _rank(meas)))
    return out


def _groups(store: Any) -> tuple[list[Any], dict[str, dict[str, str]]]:
    """Non-root nodes, and the research requests that produced no node (request id -> keys). A knob exposure's
    carrier commit (contracts.CARRIERS) is bookkeeping, not a rejected idea: left out of every count."""
    every = [n for n in store.nodes() if n.id != ROOT_NODE_ID]
    nodes = [n for n in every if not (n.status is NodeStatus.REJECTED and n.reject_reason in CARRIERS)]
    started = {e["request_id"]: e for e in store.events("research_started")}
    with_node = {m.group(1) for n in every if (m := re.search(r"request=(\S+)", n.notes or ""))}
    failed, outage = {}, outage_requests(store)        # an outage is no attempt: not a failed one
    for e in store.events("research_failed") + store.events("research_error"):
        s = started.get(e["request_id"])
        if s is not None and e["request_id"] not in with_node and e["request_id"] not in outage:
            failed[e["request_id"]] = {"brief": s.get("brief_kind") or "?", "arm": s.get("strategy", "?"),
                                       "backend": s.get("backend") or "?",
                                       "reason": e.get("status") or "error"}
    return nodes, failed


def _row(store: Any, snap: WorldSnapshot, nodes: list[Any], failed: list[dict]) -> dict:
    n = len(nodes) + len(failed)
    admitted = [x for x in nodes if x.status is not NodeStatus.REJECTED]
    screened = [x for x in nodes if x.status in _SCREENED]
    conf = [x for x in nodes if x.status is NodeStatus.CONFIRMED]
    rej = Counter([x.reject_reason or "rejected" for x in nodes if x.status is NodeStatus.REJECTED]
                  + [f["reason"] for f in failed])
    gpu = sum(r.gpu_h for x in nodes for r in store.runs(node_id=x.id))
    gain = sum(max(0.0, snap.nodes[x.id].increment.mean) for x in conf if x.id in snap.nodes)
    return {"n": n, "exec": len(admitted) / n if n else math.nan,
            "hits": sum(snap.nodes[x.id].p_gain > 0.5 for x in screened if x.id in snap.nodes),
            "screened": len(screened), "conf": len(conf), "rej": rej,
            "dup": sum(rej[d] for d in _DUP) / n if n else math.nan,
            "fams": len({x.family for x in admitted}), "tags": len({(x.family, x.mechanism_tag) for x in admitted}),
            "gpu": gpu, "gain": gain, "pred": prediction_stats(predictions(store, snap, nodes))}


def _fmt(key: str, r: dict) -> str:
    p = r["pred"]
    pred = "-" if not p["n"] else f"{p['sign_acc']:.0%}/{p['n']} bias{p['bias']:+.4f}" + (
        f" r{p['pearson']:+.2f} rho{p['spearman']:+.2f}" if p.get("pearson", math.nan) == p.get("pearson") else "")
    rej = " ".join(f"{k}:{v}" for k, v in r["rej"].most_common(3)) or "-"
    per = f"{r['gpu'] / (r['gain'] / 0.01):.0f}" if r["gain"] > 0 else "-"
    return (f"  {key:14s} {r['n']:4d} {r['exec']:5.0%} {r['hits']:3d}/{r['screened']:<3d} {r['conf']:3d}  "
            f"{pred:34s} {r['fams']:2d}/{r['tags']:<3d} {r['dup']:4.0%} {r['gpu']:7.1f} {per:>6s}  {rej}")


def quality_groups(store: Any, snapshot: WorldSnapshot) -> list[tuple[str, str, dict]]:
    """(grouping, group, row) per brief kind / bandit arm / LLM backend / shape; a row is ``_row``'s dict."""
    nodes, failed = _groups(store)
    keys: list[tuple[str, Callable[[Any], str], str]] = [
        ("brief", brief_of, "brief"), ("arm", arm_of, "arm"), ("backend", backend_of, "backend"),
        ("shape", lambda x: "compound" if "compound=" in (x.notes or "") else "single", "shape")]
    out = []
    for title, key, fk in keys:
        for g in sorted({key(x) or "?" for x in nodes} | {f.get(fk, "single") for f in failed.values()}):
            out.append((title, g, _row(store, snapshot, [x for x in nodes if (key(x) or "?") == g],
                                       [f for f in failed.values() if f.get(fk, "single") == g])))
    return out


def quality_table(store: Any, snapshot: WorldSnapshot | None) -> str:
    """Per brief kind / bandit arm / LLM backend: proposed, executed (smoke-passed and admitted),
    screened hits (own increment > 0: P(gain>0)>0.5), confirmed, prediction sign accuracy / bias / correlation, novelty
    (families/tags, near-dup rate), GPU-h and GPU-h per +0.01 confirmed shrunk gain, top rejections."""
    if snapshot is None:
        return "quality  (no world fit yet)"
    lines = ["quality  group            n  exec hit/scr conf  pred sign/n bias r rho            "
             "fam/tag dup   GPU-h h/+.01  rejections"]
    last = None
    for title, g, row in quality_groups(store, snapshot):
        if title != last:
            lines.append(f" by {title}")
            last = title
        lines.append(_fmt(g, row))
    kp = kind_pulls(store)
    if kp:
        by: dict[str, list[float]] = {}
        for p in kp:
            by.setdefault(p.arm, []).append(p.reward)
        lines.append(" brief bandit credit/GPU-h: " + ", ".join(
            f"{k} {statistics.fmean(v):.3g} (n={len(v)})" for k, v in sorted(by.items())))
    return "\n".join(lines)


def ideation_lessons(store: Any, snapshot: WorldSnapshot | None, limit: int = 6) -> str:
    """<= ~600 tokens for the researcher / review prompts: what worked, what did not, how well
    brief kinds execute and hit, and how biased past predictions were."""
    if snapshot is None:
        return "(no measurements yet)"
    out = ["IDEATION LESSONS (computed from measurements, not opinions)"]
    fams = sorted(snapshot.family_means.items(), key=lambda kp: kp[1].mean)
    pos = [f"{k} {p.mean:+.4f}±{p.sd:.4f}" for k, p in reversed(fams) if p.mean > 0][:limit]
    neg = [f"{k} {p.mean:+.4f}±{p.sd:.4f}" for k, p in fams if p.mean < 0][:limit]
    out.append("helped (family, shrunk panel delta±se): " + ("; ".join(pos) or "none yet"))
    out.append("hurt: " + ("; ".join(neg) or "none yet"))
    lev = sorted((e for e in snapshot.levers if e.target == PANEL and e.n_nonzero),
                 key=lambda e: -abs(e.mean) / max(e.se, 1e-12))[:limit]
    if lev:
        out.append("levers (panel per +1 step): " + "; ".join(
            f"{e.lever} {e.mean:+.4f}±{e.se:.4f} n={e.n_nonzero}" for e in lev))
    nodes, failed = _groups(store)
    rows = []
    for g in sorted({brief_of(x) or "?" for x in nodes}):
        r = _row(store, snapshot, [x for x in nodes if (brief_of(x) or "?") == g],
                 [f for f in failed.values() if f["brief"] == g])
        rows.append(f"{g}: {r['n']} proposed, {r['exec']:.0%} executed, {r['hits']}/{r['screened']} "
                    f"screened hits, {r['conf']} confirmed" + (
                        f", prediction bias {r['pred']['bias']:+.4f}" if r["pred"]["n"] else ""))
    if rows:
        out.append("brief kinds: " + "; ".join(rows))
    p = prediction_stats(predictions(store, snapshot, nodes))
    if p["n"]:
        opt = (f"{p['optimism']:.1f}x too optimistic" if p["optimism"] > 1 else
               f"{1 / p['optimism']:.1f}x too pessimistic") if p["optimism"] == p["optimism"] and p["optimism"] > 0 \
            else "the measured mean is <= 0"
        out.append(f"your predictions: sign right {p['sign_acc']:.0%} of {p['n']}; predicted "
                   f"{p['mean_pred']:+.4f} on average, measured {p['mean_meas']:+.4f} ({opt}). "
                   f"Size your predictions accordingly.")
    return "\n".join(out)
