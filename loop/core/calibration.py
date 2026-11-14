"""loop.core.calibration — verbalized-sampling finishes and their calibration (v3-c6i, 10-13).

The ideation analysis of 10-13: claimed sizes clustered at the minimum each brief asked for (43/58 sessions claimed
exactly the MDE; after the gap-share floor, the floor), so a claim carried no information (rho -0.09 with outcomes).
Verbalized sampling (arXiv 2510.01171) asks for a DISTRIBUTION over candidates instead of one mode-collapsed answer:

- the finish lists ``ideation.k_candidates`` distinct candidate mechanisms, each with a predicted effect distribution
  {p_neg = P(Δ < 0), median Δ, p_gap = P(Δ >= g)} (g = the request's ``ref_gap``: the gap to the target in target
  mode, else the 5-seed MDE), then picks ONE to implement with a justification (:func:`problem` checks the slate);
- the prediction is for the node's own increment on the VISIBLE screening composite (v3-c6r: the brief says so; the
  reference gap is the visible-composite gap, :func:`loop.core.ideation.target_gap`) -- the number it is scored on;
- once the node has ``ideation.calibration_min_seeds`` (3) screen seeds, the chosen candidate's prediction is scored with
  a proper scoring rule: the expected LOG SCORE of the 3-bin outcome (Δ < 0 / 0 <= Δ < g / Δ >= g) under the node's
  own-increment posterior N(m, v) (:func:`score`; the ranked probability score -- CRPS's discrete form -- and the
  median's error are logged beside it);
- per brief kind, :func:`kind_skill` is the mean log-score gain over the base-rate forecast (the bin frequencies of
  every scored session OF THE SAME GAP DEFINITION: target gap vs MDE), shrunk with ``ideation.calibration_n0`` pseudo-sessions; :func:`reweight` multiplies a kind's
  weight by exp(``ideation.calibration_weight`` x skill).

No size floor anywhere: a session states what it believes, and an overstated belief costs its kind share."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

K_CAL = "ideation.calibration"      # {node: {kind, p: [3], q: [3], ls, rps, err, gap, ts}}: scored predictions
EPS = 0.01                          # a stated probability is floored here (a 0 would make the log score -inf)
TEXT = {"type": "string", "minLength": 1}
PROB = {"type": "number", "minimum": 0, "maximum": 1}
CANDIDATE = {"type": "object", "required": ["mechanism", "family", "p_neg", "median", "p_gap"], "properties": {
    "mechanism": TEXT, "family": {"type": "string"}, "mechanism_tag": {"type": "string"}, "p_neg": PROB,
    "median": {"type": "number"}, "p_gap": PROB, "why": {"type": "string"}}}
FINISH_PROPS = {"candidates": {"type": "array", "items": CANDIDATE}, "chosen": {"type": "integer", "minimum": 0},
                "choice": {"type": "string"}}


def _phi_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# ---------------------------------------------------------------------------------------------- the finish
def problem(fin: Mapping[str, Any], k: int, gap: float, dup_cos: float = 0.9, embedder: Any = None) -> str:
    """'' when the finish carries a well-formed slate, else what to fix (one line per problem)."""
    cands = [c for c in fin.get("candidates") or () if isinstance(c, Mapping)]
    out = []
    if len(cands) < k:
        out.append(f"list {k} candidate mechanisms in `candidates` (you gave {len(cands)})")
    for i, c in enumerate(cands):
        try:
            pn, pg, med = float(c.get("p_neg")), float(c.get("p_gap")), float(c.get("median"))
        except (TypeError, ValueError):
            out.append(f"candidate {i}: p_neg, median and p_gap must be numbers")
            continue
        if not (0 <= pn <= 1 and 0 <= pg <= 1) or pn + pg > 1 + 1e-9:
            out.append(f"candidate {i}: p_neg + p_gap must be <= 1 (Δ < 0 and Δ >= {gap:.4f} cannot both hold)")
        elif (med < 0) != (pn >= 0.5) and abs(pn - 0.5) > 0.05:
            out.append(f"candidate {i}: a median of {med:+.4f} contradicts P(Δ<0) = {pn:.2f}")
        elif gap > 0 and (med >= gap) != (pg >= 0.5) and abs(pg - 0.5) > 0.05:
            out.append(f"candidate {i}: a median of {med:+.4f} contradicts P(Δ >= {gap:.4f}) = {pg:.2f}")
    if len(cands) >= 2:
        from loop.select.embed import HashingEmbedder, cosine
        emb = embedder or HashingEmbedder()
        vecs = [emb.embed(str(c.get("mechanism") or "")) for c in cands]
        same = [(i, j) for i in range(len(vecs)) for j in range(i + 1, len(vecs)) if cosine(vecs[i], vecs[j]) > dup_cos]
        if same:
            out.append(f"candidates {same[0][0]} and {same[0][1]} are one mechanism: make them DISTINCT mechanisms "
                       "(not doses of one)")
    ch = fin.get("chosen")
    if not isinstance(ch, int) or isinstance(ch, bool) or not 0 <= ch < max(1, len(cands)):
        out.append("`chosen` = the index (0-based) of the candidate you implemented")
    if not str(fin.get("choice") or "").strip():
        out.append("`choice` = why you implemented that candidate over the others")
    return "; ".join(out)


def chosen(fin: Mapping[str, Any]) -> dict[str, Any] | None:
    cands = [c for c in fin.get("candidates") or () if isinstance(c, Mapping)]
    ch = fin.get("chosen")
    return dict(cands[ch]) if isinstance(ch, int) and not isinstance(ch, bool) and 0 <= ch < len(cands) else None


def brief_text(k: int, gap: float, what: str) -> str:
    """The brief section asking for the slate (no size floor: the prediction is scored, not bounced)."""
    return (f"## Candidates, then one (verbalized sampling; scored for calibration)\n"
            f"Before you build, lay out {k} DISTINCT candidate mechanisms for this brief (different mechanisms, not "
            "doses of one) and, for each, the effect distribution you actually believe for its OWN INCREMENT on the "
            "VISIBLE SCREENING COMPOSITE (the composite the screen measures -- the per-axis weights of the panel over "
            "the tasks the loop shows you; it is what the prediction is scored against, not the official Labless "
            f"panel): p_neg = P(Δ < 0), median Δ, p_gap = P(Δ >= {gap:.4f}) ({what}). Then implement the ONE you judge best "
            f"and say why. In the finish: `candidates` = [{{mechanism (one line: what changes and why it helps), "
            "family, mechanism_tag, p_neg, median, p_gap, why}], `chosen` = its index, `choice` = the justification. "
            "After screening the chosen prediction is scored with a proper scoring rule (log score of the outcome "
            "under the measured posterior) and each brief kind's calibration steers its share of sessions: an "
            "overstated distribution costs more than an honest small one. There is no minimum size.\n")


# ---------------------------------------------------------------------------------------------- scoring
def probs(pred: Mapping[str, Any]) -> list[float]:
    """The 3-bin forecast (Δ < 0, 0 <= Δ < g, Δ >= g), each floored at EPS and renormalised."""
    pn = min(1.0, max(0.0, float(pred.get("p_neg") or 0.0)))
    pg = min(1.0 - pn, max(0.0, float(pred.get("p_gap") or 0.0)))
    p = [max(EPS, x) for x in (pn, 1.0 - pn - pg, pg)]
    s = sum(p)
    return [x / s for x in p]


def outcome(m: float, v: float, gap: float) -> list[float]:
    """P of each bin under the measured posterior N(m, v)."""
    sd = math.sqrt(max(v, 1e-12))
    a, b = _phi_cdf((0.0 - m) / sd), _phi_cdf((gap - m) / sd)
    return [a, max(0.0, b - a), max(0.0, 1.0 - b)]


def score(pred: Mapping[str, Any], m: float, v: float, gap: float) -> dict[str, Any]:
    """{p, q, ls (expected log score), rps (expected ranked probability score), err (|median - m|)}."""
    p, q = probs(pred), outcome(m, v, gap)
    ls = sum(qb * math.log(pb) for pb, qb in zip(p, q))
    fp, fq, rps = 0.0, 0.0, 0.0
    for k in range(2):                    # E[(F_k - 1{Δ in bins <= k})^2] = F^2 - 2 F Q + Q
        fp, fq = fp + p[k], fq + q[k]
        rps += fp * fp - 2 * fp * fq + fq
    med = pred.get("median")
    return {"p": [round(x, 4) for x in p], "q": [round(x, 4) for x in q], "ls": round(ls, 5), "rps": round(rps, 5),
            "err": round(abs(float(med) - m), 6) if isinstance(med, (int, float)) else None}


def gap_kind(r: Mapping[str, Any]) -> str:
    """A scored record's gap definition (target | mde; '' = recorded before v3-c6r)."""
    return str(r.get("gap_kind") or "")


def base_rates(records: Mapping[str, Mapping[str, Any]], kind: str | None = None) -> list[float]:
    """The bin frequencies of every scored session (Laplace: one pseudo-count per bin); ``kind``: only the records of
    that gap definition (v3-c6r: the third bin "Δ >= g" means a different thing for g = the target gap vs the MDE)."""
    tot = [1.0, 1.0, 1.0]
    for r in records.values():
        if kind is not None and gap_kind(r) != kind:
            continue
        for i, x in enumerate(r.get("q") or ()):
            tot[i] += float(x)
    s = sum(tot)
    return [x / s for x in tot]


def kind_skill(records: Mapping[str, Mapping[str, Any]], n0: float) -> dict[str, dict[str, float]]:
    """kind -> {n, ls, skill}: the mean log score, and its gain over the base-rate forecast shrunk to 0 with ``n0``
    pseudo-sessions (nats per session; > 0 = better calibrated than knowing only the base rates)."""
    refs = {g: base_rates(records, g) for g in {gap_kind(r) for r in records.values()}}
    out: dict[str, dict[str, float]] = {}
    for r in records.values():
        k = str(r.get("kind") or "")
        q = [float(x) for x in r.get("q") or (0, 0, 0)]
        ref = refs[gap_kind(r)]                      # the base rates of its own gap definition
        gain = float(r.get("ls") or 0.0) - sum(qb * math.log(rb) for qb, rb in zip(q, ref))
        s = out.setdefault(k, {"n": 0, "ls": 0.0, "gain": 0.0})
        s["n"], s["ls"], s["gain"] = s["n"] + 1, s["ls"] + float(r.get("ls") or 0.0), s["gain"] + gain
    for s in out.values():
        s["skill"] = s["gain"] / (s["n"] + max(0.0, n0))
        s["ls"] = s["ls"] / s["n"]
        del s["gain"]
    return out


def reweight(w: Mapping[str, float], skill: Mapping[str, Mapping[str, float]], weight: float) -> dict[str, float]:
    """``w`` x exp(weight x skill) per kind, renormalised (a kind never scored keeps its weight)."""
    raw = {k: x * math.exp(weight * float((skill.get(k) or {}).get("skill", 0.0))) for k, x in w.items()}
    tot = sum(raw.values()) or 1.0
    return {k: x / tot * (sum(w.values()) or 1.0) for k, x in raw.items()}


def due(store: Any, beliefs: Mapping[str, Any], min_seeds: int) -> list[str]:
    """Nodes with a stored chosen prediction, >= ``min_seeds`` screen seeds and no score yet."""
    done = store.get_state(K_CAL) or {}
    out = []
    for nid, b in beliefs.items():
        if nid in done or getattr(b, "n", 0) < min_seeds:
            continue
        plan = store.get_state(f"ideation:{nid}") or {}
        if isinstance(plan, Mapping) and isinstance(plan.get("prediction"), Mapping):
            out.append(nid)
    return out


def score_due(store: Any, beliefs: Mapping[str, Any], kinds: Mapping[str, str], min_seeds: int,
              now: float) -> list[str]:
    """Score every due node (:func:`due`) into :data:`K_CAL` (logged ``calibration_scored``); the acts."""
    nids = due(store, beliefs, min_seeds)
    if not nids:
        return []
    book, acts = dict(store.get_state(K_CAL) or {}), []
    for nid in nids:
        plan = store.get_state(f"ideation:{nid}") or {}
        pred, b = plan["prediction"], beliefs[nid]
        gap = float(pred.get("gap") or plan.get("ref_gap") or 0.0)
        rec = {"kind": kinds.get(nid, ""), "gap": gap, "gap_kind": str(pred.get("gap_kind") or ""),
               "m": round(b.m, 6), "n": b.n, "ts": now, **score(pred, b.m, b.v, gap)}
        book[nid] = rec
        store.log("calibration_scored", node_id=nid, **rec)
        acts.append(f"calibration {nid} ({rec['kind']}): log score {rec['ls']:+.3f}, measured {b.m:+.4f}")
    store.set_state(K_CAL, book)
    return acts
