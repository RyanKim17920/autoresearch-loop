"""loop.core.prescreen — a pairwise LLM pre-screen as a WEAK prior on the screen queue (v3-c6i, 10-13).

REHEARSE (2607.27687) / FOREAGENT (2601.05930): an LLM ranks a new candidate against measured neighbours better than
it predicts an absolute effect. Before a new node's screen is ranked, ONE short call compares it with its
``ideation.pairwise_k`` nearest screened nodes (nearest by mechanism signature, :mod:`loop.core.mechanisms`; each with
its measured own increment): P(new beats neighbour i). The mean is a predicted rank ``s`` in [-1, 1].

It only ORDERS: a screen option's belief mean moves by ``pairwise_weight x s x sd(prior)`` (:func:`nudge`; default
.1 sd, so it reorders screens of similar value and never makes a dead one live). Once nodes are measured the
pairwise accuracy is logged (``prescreen_scored``; state :data:`K_ACC`) and the weight is 0 while, over >=
``pairwise_min_scored`` nodes, the accuracy is below ``pairwise_min_acc`` or shows NO SKILL over the baseline
predictor (v3-c6r: "the new node lands at the prior mean" -- the neighbours' values are withheld from the prompt,
only their rank order is shown) -- a bounded auto-disable; the meta-cycle reads :data:`K_ACC` to raise / lower
``ideation.pairwise_weight``. Bounded cost: one call per node on the (qwen-routed, 45 s) ``prescreen`` role, on its
own one-worker pool (it never holds a brief worker); a failed call is retried on a later tick while the node is
unmeasured, at most ``RETRIES`` times."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Mapping, Sequence

K_PS = "ideation.prescreen"       # {node: {s, p_better, neighbors: [[id, delta]], ts} | {error}}
K_ACC = "ideation.prescreen_acc"  # {n, pairs, correct, acc, weight}: measured accuracy and the weight in force
ROLE = "prescreen"
RETRIES = 3                       # a failed call is asked again on later ticks (node still unmeasured) this often
RETRY_S = 600.0                   # ... no sooner than this after the failure
SCHEMA = {"type": "object", "required": ["p_better"], "properties": {
    "p_better": {"type": "array", "items": {"type": "number", "minimum": 0, "maximum": 1}},
    "why": {"type": "string"}}}
SYSTEM = ("You compare ML experiment proposals. For each measured neighbour (they are listed best-measured first), "
          "give the probability that the NEW proposal's effect on the screening composite beats that neighbour's "
          "measured effect. Use only the text given.")


def opt(ctx: Any, name: str) -> Any:
    from loop.core.levers import opt as o
    return o(ctx, name)


def neighbors(store: Any, beliefs: Mapping[str, Any], sig: str, parent_ids: Sequence[str], k: int,
              exclude: Sequence[str] = (), embedder: Any = None) -> list[tuple[str, float, str, float]]:
    """(node, measured own increment, its signature, cosine) of the ``k`` screened nodes (>= 1 seed) nearest to
    ``sig``, the candidate's ancestors and ``exclude`` left out."""
    from loop.core import mechanisms as MS
    from loop.select.embed import HashingEmbedder, cosine
    emb = embedder or HashingEmbedder()
    me = emb.embed(sig)
    skip = set(MS.ancestors(store, parent_ids)) | set(exclude)
    out = []
    for nid, b in beliefs.items():
        if nid in skip or getattr(b, "n", 0) < 1 or (theirs := MS.node_signature(store, nid)) == "":
            continue
        out.append((nid, float(b.m), theirs, cosine(me, emb.embed(theirs))))
    return sorted(out, key=lambda x: -x[3])[:max(0, int(k))]


def prompt(sig: str, claim: str, near: Sequence[tuple[str, float, str, float]]) -> str:
    """v3-c6r: the neighbours ranked by their measured own increment, BEST FIRST, with no values (10-13 review: the
    shown deltas made "p > .5 iff d < 0" score well without reading the proposal)."""
    rows = "\n".join(f"{i}. {s[:240]}" for i, (_, _, s, _) in enumerate(near))
    return (f"NEW proposal: {sig[:300]}\nIts claim: {claim[:600]}\n\nMeasured neighbours, ranked from the best "
            f"measured effect to the worst (values withheld):\n{rows}"
            f"\n\nReturn p_better: {len(near)} probabilities, in the neighbours' order.")


def rank_of(p_better: Sequence[float]) -> float:
    """The predicted rank s = 2 x mean P(better) - 1 in [-1, 1] (0 = no information)."""
    ps = [min(1.0, max(0.0, float(p))) for p in p_better]
    return 2 * sum(ps) / len(ps) - 1 if ps else 0.0


def prior_mean(beliefs: Mapping[str, Any]) -> float:
    """The baseline predictor's guess of a new node's own increment: the mean over the measured nodes."""
    ms = [float(b.m) for b in beliefs.values() if getattr(b, "n", 0) >= 1]
    return sum(ms) / len(ms) if ms else 0.0


def ask(llm: Any, sig: str, claim: str, near: Sequence[tuple[str, float, str, float]], m0: float = 0.0
        ) -> dict[str, Any]:
    """The one call: {s, p_better, neighbors, m0} (raises on a backend / schema failure). The neighbours are asked
    best-measured first; ``m0`` is the baseline predictor's guess (the prior mean) the accuracy is scored against."""
    near = sorted(near, key=lambda x: -x[1])
    got = llm.complete_json(ROLE, system=SYSTEM, prompt=prompt(sig, claim, near), schema=SCHEMA)
    p = [float(x) for x in (got.get("p_better") or [])][:len(near)]
    if len(p) != len(near):
        raise ValueError(f"p_better has {len(p)} entries for {len(near)} neighbours")
    return {"s": round(rank_of(p), 4), "p_better": [round(x, 3) for x in p],
            "neighbors": [[n, round(d, 6)] for n, d, _, _ in near], "m0": round(float(m0), 6)}


def record(store: Any, nid: str, rec: Mapping[str, Any]) -> None:
    book = dict(store.get_state(K_PS) or {})
    book[nid] = dict(rec)
    store.set_state(K_PS, book)
    store.log("prescreen", node_id=nid, **{k: v for k, v in rec.items() if k not in ("neighbors", "claim", "sig")})


# ---------------------------------------------------------------------------------------------- accuracy + weight
def score(store: Any, beliefs: Mapping[str, Any], min_seeds: int, now: float) -> list[str]:
    """Measured nodes with a prediction (>= ``min_seeds`` seeds, not yet scored): each comparison is correct when
    P(better) > .5 and the node's own increment beats the neighbour's (or < .5 and it does not; .5 abstains). The
    BASELINE predictor -- "the new node lands at the prior mean m0": better iff the neighbour's increment < m0 -- is
    scored on the same comparisons (the skill is the accuracy over it)."""
    book, acts, changed = dict(store.get_state(K_PS) or {}), [], False
    for nid, rec in book.items():
        b = beliefs.get(nid)
        if b is None or getattr(b, "n", 0) < min_seeds or "scored" in rec or "p_better" not in rec:
            continue
        m0 = float(rec.get("m0") or 0.0)
        pairs = correct = base = 0
        for (other, d), p in zip(rec.get("neighbors") or (), rec["p_better"]):
            if abs(p - 0.5) < 1e-9:
                continue
            pairs += 1
            truth = b.m > float(d)
            correct += (p > 0.5) == truth
            base += (float(d) < m0) == truth
        rec = book[nid] = {**rec, "scored": {"m": round(b.m, 6), "pairs": pairs, "correct": correct, "base": base,
                                             "ts": now}}
        changed = True
        store.log("prescreen_scored", node_id=nid, s=rec.get("s"), m=round(b.m, 6), pairs=pairs, correct=correct,
                  base=base)
        acts.append(f"prescreen {nid}: {correct}/{pairs} pairwise comparisons right (baseline {base}/{pairs})")
    if changed:
        store.set_state(K_PS, book)
    return acts


def accuracy(book: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """{n, pairs, correct, acc, base_acc, skill}: skill = accuracy - the baseline predictor's (records scored before
    the baseline existed count their baseline from the stored neighbours at m0 0)."""
    sc = []
    for r in book.values():
        if not (isinstance(r, Mapping) and isinstance(r.get("scored"), Mapping)):
            continue
        s = dict(r["scored"])
        if "base" not in s:
            m0, truth_m = float(r.get("m0") or 0.0), float(s.get("m") or 0.0)
            s["base"] = sum((float(d) < m0) == (truth_m > float(d))
                            for (_, d), p in zip(r.get("neighbors") or (), r.get("p_better") or ()) if abs(p - 0.5) >= 1e-9)
        sc.append(s)
    pairs, correct, base = (sum(int(s.get(k) or 0) for s in sc) for k in ("pairs", "correct", "base"))
    acc = round(correct / pairs, 4) if pairs else None
    bacc = round(base / pairs, 4) if pairs else None
    return {"n": len(sc), "pairs": pairs, "correct": correct, "acc": acc, "base_acc": bacc,
            "skill": round(acc - bacc, 4) if pairs else None}


def weight(ctx: Any, store: Any) -> float:
    """``ideation.pairwise_weight`` while unmeasured or skilful; 0 once >= ``pairwise_min_scored`` nodes show an
    accuracy below ``pairwise_min_acc`` OR no skill over the baseline predictor (skill <= 0: the stricter of the
    two always applies). Written to :data:`K_ACC` when it changes."""
    base = float(opt(ctx, "pairwise_weight"))
    acc = accuracy(store.get_state(K_PS) or {})
    bad = acc["n"] >= int(opt(ctx, "pairwise_min_scored")) and acc["acc"] is not None \
        and (acc["acc"] < float(opt(ctx, "pairwise_min_acc")) or (acc["skill"] or 0.0) <= 0.0)
    w = 0.0 if bad else base
    cur = {**acc, "weight": w}
    if store.get_state(K_ACC) != cur:
        store.set_state(K_ACC, cur)
        store.log("prescreen_weight", **cur)
    return w


# ---------------------------------------------------------------------------------------------- ordering
CENTER_MIN = 5                    # predictions needed before s is centred on their median (:func:`center`)


def center(book: Mapping[str, Mapping[str, Any]]) -> float:
    """The median predicted rank over the book once it holds >= CENTER_MIN predictions, else 0. 11-09 live (nanobrain):
    32 of 33 queued nodes had s < 0 (median -.17; the neighbours are shown best-first), so an UNSCORED node (a bold /
    program / crossover node, or one whose call failed) outranked every scored one and the prior only ever pushed
    scored nodes back: centred, s orders the scored nodes around the unscored ones."""
    s = sorted(float(r["s"]) for r in book.values() if isinstance(r, Mapping) and isinstance(r.get("s"), (int, float)))
    if len(s) < CENTER_MIN:
        return 0.0
    k = len(s) // 2
    return s[k] if len(s) % 2 else (s[k - 1] + s[k]) / 2.0


def nudge(options: Sequence[Any], book: Mapping[str, Mapping[str, Any]], w: float,
          centred: bool = False) -> list[Any]:
    """Each ``screen`` option of a node with a prediction: m + w x (s - c) x sqrt(v) (the rest unchanged); c is
    :func:`center` with ``centred`` (``ideation.pairwise_center``), else 0."""
    if w == 0 or not book:
        return list(options)
    out, c = [], center(book) if centred else 0.0
    for o in options:
        rec = book.get(getattr(o, "node", "") or "")
        if getattr(o, "kind", "") == "screen" and isinstance(rec, Mapping) and isinstance(rec.get("s"), (int, float)):
            o = replace(o, m=o.m + w * (float(rec["s"]) - c) * math.sqrt(max(o.v, 0.0)))
        out.append(o)
    return out
