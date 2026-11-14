"""Thompson helpers: queue ranking (D4.5) and parent / family draws for the arms.

The lever trend enters ``rank_queue`` exactly once, inside ``world.predictive``."""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Any, Sequence

from loop.contracts import PANEL, ROOT_NODE_ID, LeverTags, Node, Posterior, WorldSnapshot, current_reviews, design_point
from loop.novelty import norm_tag
from loop.spec import LoopConfig

OFFSPRING_PENALTY_SIGMAS = 1.0     # exploit parent penalty c*log(1+offspring), c in panel sigmas
CAP_SIGMAS = 0.5                   # review-bias cap, in panel sigmas
SCREENING_PER_FAMILY, SCREENING_PER_TAG = 2, 1   # queue diversity: at most this many in SCREENING


def sigma_panel(cfg: LoopConfig, snapshot: WorldSnapshot | None) -> float:
    return snapshot.sigma.get(PANEL, cfg.panel_sd) if snapshot else cfg.panel_sd


def clip(x: float, cap: float) -> float:
    return max(-cap, min(cap, x))


def utility(snapshot: WorldSnapshot, node_id: str) -> Posterior:
    """U_i, or the exact-zero root posterior for an unknown id."""
    p = snapshot.nodes.get(node_id)
    return p.utility if p is not None else Posterior(0.0, 0.0)


def vetoed(snapshot: WorldSnapshot, node_id: str) -> bool:
    return node_id in snapshot.nodes and snapshot.nodes[node_id].vetoed


def draw(p: Posterior, rng: random.Random) -> float:
    return rng.gauss(p.mean, p.sd) if p.sd > 0 else p.mean


def review_bias(store: Any, cfg: LoopConfig, snapshot: WorldSnapshot, family: str) -> float:
    """Sum of each review's explore_bias[family], linearly decayed over
    ``review.bias_decay_evals`` scored evals; each and the total capped at CAP_SIGMAS*sigma. Reviews from
    before the last measurement-epoch bump are ignored (their estimates were corrected)."""
    cap, total = CAP_SIGMAS * sigma_panel(cfg, snapshot), 0.0
    for rv in current_reviews(store):
        age = max(0, snapshot.n_scored - rv.n_evals_at)
        total += clip(rv.explore_bias.get(family, 0.0), cap) * max(
            0.0, 1.0 - age / max(cfg.review.bias_decay_evals, 1))
    return clip(total, cap)


def rank_queue(store: Any, world: Any, cfg: LoopConfig, snapshot: WorldSnapshot,
               queued: Sequence[Node], rng: random.Random, levers: bool = True) -> list[Node]:
    """Order by theta_c = draw(U_parent) + draw(eps_c), best first, ties older first; ``levers=False``
    (loop.components ablation of ``lever_prior``) ranks without the lever-trend term."""
    scored = []
    for c in queued:
        pred = world.predictive(snapshot, family=c.family, tag=c.mechanism_tag,      # the predictive model's
                                levers=c.levers if levers else LeverTags(),           # score when in force
                                **({"node": c} if getattr(snapshot, "wm", None) is not None else {}))
        m = pred.mean + review_bias(store, cfg, snapshot, c.family)     # prior_bonus: the retired tournament's
        theta = draw(utility(snapshot, c.parent_id or ROOT_NODE_ID), rng) + rng.gauss(m, pred.sd)
        scored.append((-theta, c.created_ts, c.id, c))
    return [t[3] for t in sorted(scored, key=lambda t: t[:3])]


def diversify(ranked: Sequence[Node], screening: Sequence[Node]) -> list[Node]:
    """Heads that keep SCREENING within the per-family / per-tag caps first (counting the heads
    taken before them), then the rest in rank order: diversity never leaves a slot idle."""
    fam, tag = Counter(n.family for n in screening), Counter(norm_tag(n.mechanism_tag) for n in screening)
    first, rest = [], []
    for n in ranked:
        if fam[n.family] < SCREENING_PER_FAMILY and tag[norm_tag(n.mechanism_tag)] < SCREENING_PER_TAG:
            fam[n.family] += 1
            tag[norm_tag(n.mechanism_tag)] += 1
            first.append(n)
        else:
            rest.append(n)
    return first + rest


def offspring_counts(nodes: Sequence[Node]) -> dict[str, int]:
    """Children per parent, designed-batch points excepted: a leg factorial's / design batch's points are not
    research offspring (live: L1's 48 points on a node added ~0.55 sigma to its exploit penalty, tilting exploit
    toward a node)."""
    out: dict[str, int] = {}
    for pid in (p for n in nodes if not design_point(n) for p in (n.parent_id, n.second_parent_id) if p):
        out[pid] = out.get(pid, 0) + 1
    return out


def pick_exploit_parent(snapshot: WorldSnapshot, frontier: Sequence[str], offspring: dict[str, int],
                        sigma_p: float, rng: random.Random) -> str:
    """argmax of a Thompson draw of U minus c*log(1+offspring)."""
    c = OFFSPRING_PENALTY_SIGMAS * sigma_p
    return max(frontier, key=lambda i: draw(utility(snapshot, i), rng) - c * math.log1p(offspring.get(i, 0)))


def pick_family(families: Sequence[str], counts: dict[str, int], snapshot: WorldSnapshot,
                rng: random.Random) -> str:
    """The least-sampled family; ties go to the widest posterior (unseen = infinite), then random."""
    if not families:
        return ""
    sd = {f: snapshot.family_means[f].sd if f in snapshot.family_means else math.inf for f in families}
    return min(families, key=lambda f: (counts.get(f, 0), -sd[f], rng.random()))
