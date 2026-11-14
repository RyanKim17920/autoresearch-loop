"""loop.select — strategy bandit, islands, intake, Thompson pick (DESIGN §4). Imports no
agents: brief text comes from an injected ``brief_fn(kind, store, snapshot, **kw)``; ``admit`` writes
nothing (the driver stores nodes, embeddings and bonuses)."""

from __future__ import annotations

import random
import re
from typing import Any, Callable, Mapping, Sequence

from loop.contracts import (BANDIT_ARMS, K_INFLIGHT, K_STALE_META, ROOT_NODE_ID, ArmPull, Candidate, CandidateRequest,
                            HypothesisKind, HypothesisStatus, LeverTags, Node, NodeStatus, Operator, RunStatus,
                            Stage, Strategy, WorldSnapshot, current_reviews, design_point)
from loop import components as CG
from loop import novelty as NV
from loop.spec import LoopContext
from loop.world import held_back, mixture, prob_greater, scored_seeds
from loop.select import bank as B, thompson
from loop.select.bandit import K_KIND_PULLS, Bandit, brief_of, compute_reward, kind_credit, kind_pulls
from loop.select.embed import HashingEmbedder
from loop.select.intake import K_BASE, admit as _admit, candidate_text, config_gaps, parent_config
from loop.select.islands import IslandBook

__all__ = ["PortfolioSelector", "HashingEmbedder"]

_SEQ, _PROBES, _ALT = "select.req_seq", "select.probes_used", "select.bank_turn"
_PROBE_ARMS = (Strategy.EXPLORE, Strategy.EXPLOIT)     # the arms an untagged review probe may fill


def is_moonshot_probe(text: str) -> bool:
    """A review probe brief tagged ``[moonshot]`` runs as a moonshot (bigger caps, may span families)."""
    return text.lstrip().lower().startswith("[moonshot]")


def moonshots_inflight(nodes: Sequence[Node], inflight: Sequence[Mapping[str, Any]]) -> int:
    """Moonshots still in flight: open research requests plus QUEUED moonshot nodes (the
    ``bandit.moonshot_max_inflight`` cap counts both; a tagged probe fills a MOONSHOT pull so it
    respects the same cap). A screening moonshot no longer holds a slot (screens can pend for hours on
    the preemptible pool), nor does a tuning pass's winner (``tune_of=``)."""
    return sum(r.get("strategy") == Strategy.MOONSHOT.value for r in inflight) + sum(
        n.strategy is Strategy.MOONSHOT and n.status is NodeStatus.QUEUED and "tune_of=" not in (n.notes or "")
        for n in nodes)


def crossover_pair(node: Node) -> frozenset:
    """The two parents a crossover node merged (its parent_id is their common ancestor)."""
    m = re.search(r"\bxover=([^,\s]+),(\S+)", node.notes or "")
    return frozenset(m.groups()) if m else frozenset((node.parent_id, node.second_parent_id))


def crossover_levers(store: Any, parent_ids: Sequence[str]) -> LeverTags:
    """Both parents' registered lever vectors, summed and clipped to a direction."""
    tot: dict[str, int] = {}
    for n in filter(None, (store.get_node(i) for i in parent_ids)):
        for k, d in n.levers.directions.items():
            tot[k] = tot.get(k, 0) + d
    return LeverTags({k: (v > 0) - (v < 0) for k, v in tot.items() if v})


def built_from(store: Any, node_id: str) -> set[str]:
    """Every node this one was built from: parent chain, crossover pair, debug parent, compound stack
    sources, transitively (root excluded)."""
    out: set[str] = set()
    todo = [node_id]
    while todo:
        n = store.get_node(todo.pop())
        if n is None:
            continue
        src = re.search(r"\bstack_from=(\S+)", n.notes or "")
        legs = re.search(r"\bstack_legs=(\S+)", n.notes or "")
        for r in {n.parent_id, n.second_parent_id, *crossover_pair(n), *(src.group(1).split(",") if src else ()),
                  *(legs.group(1).split(",") if legs else ())}:
            if r and r != ROOT_NODE_ID and r not in out:
                out.add(r)
                todo.append(r)
    return out


def _touches_code(n: Node) -> bool:
    """The node's own candidate diff touches a non-config file (unreadable diff: assume it does)."""
    if not n.diff_path:
        return False
    try:
        text = open(n.diff_path).read()
    except OSError:
        return True
    return any(not p.endswith((".yaml", ".yml")) for p in re.findall(r"^diff --git a/(\S+)", text, re.M))


def change_set(store: Any, node: Node) -> tuple[dict[str, str], bool]:
    """(config key -> value, touches code) accumulated down the parent chain from root."""
    items, code, n = {}, False, node
    chain = []
    while n is not None and n.id != ROOT_NODE_ID:
        chain.append(n)
        n = store.get_node(n.parent_id) if n.parent_id else None
    for n in reversed(chain):
        items.update({k: repr(v) for k, v in (store.get_state(f"config_diff:{n.id}") or {}).items()})
        code = code or _touches_code(n)
    return items, code


def winner_pairs(store: Any, snap: WorldSnapshot, skip: set = frozenset(), head_p: float = 0.8,
                 major: float = 2.0, leg_p: float = 0.8, pool: Sequence[str] = ()) -> list[tuple[str, str]]:
    """Every untried pair of CONFIRMED non-vetoed nodes whose OWN increment is real (a scored leg needs
    P(own gain > 0) > ``leg_p``: a confirmed carrier copy re-rooted on root, or a null leg over the node it
    contains, is no cross leg), neither containing the other (built from it,
    or its config-only change set a subset of the other's), most complementary first: different
    family, then least changed-key overlap, then the larger P(U>0) product. Higher-P parent first.
    Confirm-and-condense: a banked node (``snap.banked``) is crossed only with the stack HEAD -- the
    incumbent, unless CONFIRMED challengers (never a banked or refreeze-held node: live a node herded every slot)
    beat it with P(U>U_inc) >= ``head_p`` (then the best
    of those by confirm-only estimate) -- or with a banked SIBLING (same parent: the stack of two banked legs' diffs on
    their common parent; live, a node's banked children could cross nothing, each contains the head) -- after
    every confirmed pair, so the stack grows in O(bank) and the head moves only on evidence (pairs are recomputed
    from the current head every call). A banked leg passed its own confirm-seed bar (loop.world.banked), so
    ``leg_p`` (selection seeds included) is the CONFIRMED legs' filter. A partner that is a
    config-only twin (effective config within ``major`` on every key) of one already crossed with the other
    parent is skipped. ``pool``: more non-CONFIRMED legs crossed with the head only (loop.legs.pool: well-measured
    promising; never with a sibling: 39 live pool nodes made 170 pairs, mostly root siblings).
    """
    sibs = set(getattr(snap, "banked", ()))
    banked = sibs | set(pool)
    conf = [n for n in [*store.nodes(status=[NodeStatus.CONFIRMED]), *map(store.get_node, sorted(banked))]
            if n is not None and n.id != ROOT_NODE_ID and not design_point(n) and not thompson.vetoed(snap, n.id)
            and (n.id in banked or n.id not in snap.nodes or snap.nodes[n.id].p_gain > leg_p)]
    if len(conf) < 2:
        return []
    est = lambda n: (snap.nodes[n.id].confirm_estimate or snap.nodes[n.id].utility).mean \
        if n.id in snap.nodes else float("-inf")                                        # noqa: E731
    inc = getattr(snap, "incumbent_id", ROOT_NODE_ID)
    held = held_back(store)            # refreeze-confirmed, not yet audited: a leg, never the head
    chal = [n for n in conf if n.id != inc and n.id not in held and n.id not in banked   # a head is CONFIRMED
            and n.id in snap.nodes and snap.nodes[n.id].p_beats_incumbent >= head_p]
    head = max(chal, key=lambda n: (est(n), n.id)).id if chal else inc if inc in {n.id for n in conf} else None
    tried = {crossover_pair(n) for n in store.nodes() if n.strategy is Strategy.CROSSOVER} | set(skip)
    info = {n.id: (built_from(store, n.id), *change_set(store, n)) for n in conf}
    p_of = lambda n: snap.nodes[n.id].p_positive if n.id in snap.nodes else 0.0   # noqa: E731
    base, eff = store.get_state(K_BASE) or {}, {}

    def cfg_of(i: str) -> tuple[dict, bool]:
        """(effective config, touches code) of node ``i``."""
        if i not in eff:
            n = store.get_node(i)
            eff[i] = ({**base, **parent_config(store, i)}, change_set(store, n)[1] if n is not None else True)
        return eff[i]

    def twin_tried(x: Node, y: Node) -> bool:
        """``y`` is a config-only twin (every effective value within ``major`` x or /) of a partner already
        crossed with ``x``: that merge was already tried in all but a small tweak."""
        return not cfg_of(y.id)[1] and any(
            not cfg_of(q)[1] and all(NV.minor(g, major) for g in config_gaps(cfg_of(q)[0], cfg_of(y.id)[0]).values())
            for pr in tried if x.id in pr and len(pr) == 2 for q in pr - {x.id, y.id})

    def contains(a: Node, b: Node) -> bool:
        anc, items, _ = info[a.id]
        _, b_items, b_code = info[b.id]
        return b.id in anc or (not b_code and b_items.items() <= items.items())

    out = []
    for i, a in enumerate(conf):
        for b in conf[i + 1:]:
            bank = bool(banked & {a.id, b.id})
            if frozenset((a.id, b.id)) in tried or contains(a, b) or contains(b, a) \
                    or (bank and head not in (a.id, b.id) and not ({a.id, b.id} <= sibs and a.parent_id == b.parent_id)) \
                    or twin_tried(a, b) or twin_tried(b, a):
                continue
            ka, kb = set(info[a.id][1]), set(info[b.id][1])
            overlap = len(ka & kb) / max(1, len(ka | kb))
            hi, lo = (a, b) if (p_of(a), a.id) >= (p_of(b), b.id) else (b, a)
            out.append(((bank, a.family == b.family, overlap, -p_of(a) * p_of(b), hi.id, lo.id), (hi.id, lo.id)))
    return [pair for _, pair in sorted(out)]


STACK_P, STACK_MIN_SEEDS, STACK_K, STACK_MAX = 0.7, 3, 4, 2   # leg bar, its scored seeds, legs per stack, in flight


def pooled_gain(snap: WorldSnapshot, nid: str, sigma: float) -> float:
    """P(own gain > 0) on ALL the node's scored seeds: its selection-seed increment posterior updated by its
    confirm seeds' own mean (``snap.own_confirm``, per-seed variance sigma^2). Under the gate prior (``snap.eb``:
    cell -> mixture, loop.world.predictive): the raw selection increment pooled with the confirm mean, then its posterior
    under its cell's mixture."""
    p, c = snap.nodes[nid], (getattr(snap, "own_confirm", None) or {}).get(nid)
    if (r := (getattr(snap, "eb_raw", None) or {}).get(nid)) is not None and r[0] in (mixes := getattr(snap, "eb", None) or {}):
        m, v = r[1], r[2]
        if c and c[1] and v > 0:
            w, wc = 1.0 / v, c[1] / (sigma * sigma)
            m, v = (w * m + wc * c[0]) / (w + wc), 1.0 / (w + wc)
        return mixture.prob_greater(mixture.posterior(mixture.Mixture(*mixes[r[0]]), (m, v)))
    m, v = p.increment.mean, p.increment.var
    if c and c[1] and v > 0:
        w, wc = 1.0 / v, c[1] / (sigma * sigma)
        m, v = (w * m + wc * c[0]) / (w + wc), 1.0 / (w + wc)
    return prob_greater((m, v))


def stack_plan(store: Any, snap: WorldSnapshot, sigma: float, skip: Any = frozenset()) -> list[tuple[str, ...]]:
    """STACKING UNCONFIRMED SMALL SIGNALS (operator 2026-10-03: "even if it fails confirmation, stacking could make
    a small signal larger"): per ESTABLISHED carrier (CONFIRMED, not held back; root only while it is the incumbent:
    root's legs are the early knobs every confirmed stack already carries), its closed children
    (screened out / not confirmed; never vetoed, a design point or a stack) with >= STACK_MIN_SEEDS scored seeds and
    :func:`pooled_gain` >= STACK_P are legs; the highest-P legs with distinct normalized tags and no shared config key
    (a conflict: the same knob) form ONE stack of 2..STACK_K legs. Returns the stacks (leg ids, highest P first) not
    in ``skip`` (sorted leg-id tuples already built), best mean P first."""
    held = held_back(store)
    ok = {ROOT_NODE_ID} if getattr(snap, "incumbent_id", ROOT_NODE_ID) == ROOT_NODE_ID else set()
    ok |= {n.id for n in store.nodes(status=[NodeStatus.CONFIRMED]) if n.id not in held and not thompson.vetoed(snap, n.id)}
    legs: dict[str, list[tuple[float, Node]]] = {}
    for n in store.nodes(status=[NodeStatus.SCREENED_OUT, NodeStatus.NOT_CONFIRMED]):
        if (n.parent_id or ROOT_NODE_ID) not in ok or n.id not in snap.nodes or snap.nodes[n.id].vetoed \
                or design_point(n) or "stack_legs=" in (n.notes or "") or scored_seeds(store, n.id) < STACK_MIN_SEEDS:
            continue
        if (p := pooled_gain(snap, n.id, sigma)) >= STACK_P:
            legs.setdefault(n.parent_id or ROOT_NODE_ID, []).append((p, n))
    out = []
    for carrier, cands in legs.items():
        pick, tags, keys = [], set(), set()
        for p, n in sorted(cands, key=lambda x: (-x[0], x[1].id)):
            tag, mine = NV.norm_tag(n.mechanism_tag), set(store.get_state(f"config_diff:{n.id}") or {})
            if len(pick) < STACK_K and tag not in tags and not keys & mine:
                pick.append((p, n.id))
                tags.add(tag)
                keys |= mine
        if len(pick) >= 2 and tuple(sorted(i for _, i in pick)) not in skip:
            out.append((-sum(p for p, _ in pick) / len(pick), tuple(i for _, i in pick)))
    return [legs_ for _, legs_ in sorted(out)]


def related(store: Any, a_id: str, b_id: str, major: float = 2.0) -> str:
    """'contains' (one is built from the other, or its config-only change set is a subset of the other's), 'twin'
    (config-only both, every effective value within ``major`` x or /), else ''."""
    a, b = store.get_node(a_id), store.get_node(b_id)
    if a is None or b is None or ROOT_NODE_ID in (a_id, b_id):
        return ""
    (ia, ca), (ib, cb) = change_set(store, a), change_set(store, b)
    if b_id in built_from(store, a_id) or a_id in built_from(store, b_id) \
            or (not cb and ib.items() <= ia.items()) or (not ca and ia.items() <= ib.items()):
        return "contains"
    base = store.get_state(K_BASE) or {}
    gaps = config_gaps({**base, **parent_config(store, a_id)}, {**base, **parent_config(store, b_id)})
    return "twin" if not ca and not cb and all(NV.minor(g, major) for g in gaps.values()) else ""


CARRIER_HIT_P, CARRIER_RECENT, CARRIER_PRIOR = 0.8, 20, 4.0
#: share of moonshot / explore requests on a ROTATED carrier (operator 2026-10-03: "if we have a confirmed current best,
#: focus on comparisons with the confirmed best"): the rest build on the incumbent, where a node's own increment IS
#: its gain over the incumbent
CARRIER_ROTATE = 0.25


def carrier_rates(store: Any, snap: WorldSnapshot, carriers: Sequence[str]) -> dict[str, tuple[int, int]]:
    """carrier -> (hits, children) over its ``CARRIER_RECENT`` most recent scored children (design points
    excepted): a hit is a child whose OWN increment has P(gain > 0) > ``CARRIER_HIT_P``."""
    kids: dict[str, list[Node]] = {c: [] for c in carriers}
    for n in sorted(store.nodes(), key=lambda n: (n.created_ts, n.id)):
        pid = n.parent_id or ROOT_NODE_ID
        if n.id != ROOT_NODE_ID and pid in kids and not design_point(n) and n.id in snap.nodes \
                and snap.nodes[n.id].utility.n:
            kids[pid].append(n)
    return {c: (sum(snap.nodes[k.id].p_gain > CARRIER_HIT_P for k in ks[-CARRIER_RECENT:]), len(ks[-CARRIER_RECENT:]))
            for c, ks in kids.items()}


def established(store: Any, node: Node) -> str:
    """The nearest ancestor of ``node`` a new idea may be built on: CONFIRMED (not held back for its audit), else root.
    A closed node (screened out / not confirmed / vetoed / rejected / audit failed) is never one."""
    held, nid = held_back(store), node.parent_id or ROOT_NODE_ID
    while nid != ROOT_NODE_ID:
        n = store.get_node(nid)
        if n is None:
            return ROOT_NODE_ID
        if n.status is NodeStatus.CONFIRMED and nid not in held:
            return nid
        nid = n.parent_id or ROOT_NODE_ID
    return ROOT_NODE_ID


SCARCE_CARRIERS = 2                 # carriers new screens use while the driver's scarce mode is on


def live_blocks(store: Any, cfg: Any) -> dict[str, int]:
    """carrier -> children its newest screen block still takes, for blocks whose control runs (root's baseline /
    the parent's parent-stage runs at the block's seeds) are already submitted, running or done: a child built
    there shares them (live 2026-10-02: 13 of 19 controls served exactly one child)."""
    from loop import seeds as S
    nodes, runs = store.nodes(), store.runs()
    blocks = S.assign_blocks(nodes, cfg.seeds, runs)
    last: dict[str, int] = {}
    kids: dict[tuple[str, int], int] = {}
    for n in sorted(nodes, key=lambda n: (n.created_ts, n.id)):
        if n.id in blocks and not design_point(n):
            pid = n.parent_id or ROOT_NODE_ID
            last[pid] = blocks[n.id]
            kids[(pid, blocks[n.id])] = kids.get((pid, blocks[n.id]), 0) + 1
    live = {(r.node_id, r.stage, r.seed) for r in runs
            if r.status in (RunStatus.SUBMITTING, RunStatus.RUNNING, RunStatus.DONE)}
    out = {}
    for pid, b in last.items():
        stage = Stage.BASELINE if pid == ROOT_NODE_ID else Stage.PARENT
        room = cfg.seeds.block_children - kids[(pid, b)]
        if room > 0 and all((pid, stage, int(x)) in live for x in S.screen_block(cfg.seeds, b)):
            out[pid] = room
    return out


def prefer_carriers(store: Any, cfg: Any, cands: Sequence[str]) -> list[str]:
    """``cands`` narrowed for a new screen: those whose newest block's controls already run or are done
    (:func:`live_blocks`), if any; in the driver's scarce mode at most SCARCE_CARRIERS of them (the most room in a
    live block, then the most children)."""
    if cfg is None or not cands:
        return list(cands)
    live = live_blocks(store, cfg)
    pref = [c for c in cands if c in live] or list(cands)
    if (store.get_state("runbook.scarce") or {}).get("on"):
        n = {}
        for x in store.nodes():
            n[x.parent_id or ROOT_NODE_ID] = n.get(x.parent_id or ROOT_NODE_ID, 0) + 1
        pref = sorted(pref, key=lambda c: (-live.get(c, 0), -n.get(c, 0), c))[:SCARCE_CARRIERS]
    return pref


def revisit_ok(snap: WorldSnapshot, cfg: Any, node: Node, carrier: str) -> bool:
    """The REVISIT value gate: the near miss ``node`` rebuilt on ``carrier`` as a revisit is predicted worth it
    (loop.world.predictive.gate_value), or takes the exploration slot; always without the predictive model."""
    from loop.world.predictive import gate_value
    wm = getattr(snap, "wm", None)
    if wm is None or cfg is None:
        return True
    return gate_value(wm, cfg, f"revisit:{node.id}", wm.value(node, parent=carrier, strategy="revisit/improve"))[0]


def allocate_carrier(store: Any, snap: WorldSnapshot, carriers: Sequence[str], rng: random.Random,
                     cfg: Any) -> tuple[str, str] | None:
    """(parent, why) from the predictive model's ALLOCATION (loop.world.predictive.Fit.allocation): over root, the
    incumbent and the CONFIRMED non-vetoed ``carriers``, each scored by its best mechanism-type cell's P(fresh gain >
    MDE); the incumbent on a ``prior.wm_incumbent_share`` floor of draws, else drawn in proportion to it (a carrier
    whose newest block still has live controls x1.5: sibling sharing), or uniformly on the ``prior.wm_explore`` share. The why names the carrier's most promising mechanism types (the
    brief's target). None without the model."""
    wm = getattr(snap, "wm", None)
    if wm is None or cfg is None:
        return None
    inc = getattr(snap, "incumbent_id", "") or ROOT_NODE_ID
    conf = {n.id for n in store.nodes(status=[NodeStatus.CONFIRMED])} - held_back(store)
    cands = list(dict.fromkeys([ROOT_NODE_ID, inc, *(c for c in carriers if c in conf and not thompson.vetoed(snap, c))]))
    cells = wm.allocation(cands)
    best: dict[str, list[dict]] = {}
    for c in cells:
        best.setdefault(c["carrier"], []).append(c)
    if not best:
        return None
    live = set(prefer_carriers(store, cfg, list(best)))
    score = {c: max(x["p_mde"] for x in xs) * (1.5 if c in live and len(live) < len(best) else 1.0)
             for c, xs in best.items()}
    if inc in best and rng.random() < getattr(cfg.prior, "wm_incumbent_share", 0.0):
        pick, how = inc, f"the incumbent floor, {cfg.prior.wm_incumbent_share:.0%} of draws"
    elif rng.random() < cfg.prior.wm_explore or sum(score.values()) <= 0.0:
        pick, how = rng.choice(sorted(best)), "exploration draw"
    else:
        ks = sorted(score)
        pick, how = rng.choices(ks, weights=[score[k] for k in ks])[0], "drawn in proportion"
    top = sorted(best[pick], key=lambda x: -x["p_mde"])[:3]
    return pick, (f"carrier {pick} ({how} over the predictive model's allocation, P(fresh own gain > {wm.mde:.4f}) of "
                  "the best cell per carrier: " + ", ".join(f"{c} {score[c]:.3f}" for c in sorted(score, key=lambda k: -score[k]))
                  + f"). Mechanism types the model expects most from here: "
                  + ", ".join(f"{x['mtype']} (P {x['p_mde']:.3f}, mean {x['mean']:+.4f})" for x in top)
                  + f". Your node is judged by its OWN gain over {pick}, and by P(U > U_incumbent {inc}).")


def pick_carrier(store: Any, snap: WorldSnapshot, carriers: Sequence[str], rng: random.Random,
                 cfg: Any = None, rotate: float = CARRIER_ROTATE) -> tuple[str, str]:
    """(parent, why) for a moonshot / explore request: the confirmed INCUMBENT (the reference every idea is judged
    against), except on a ``rotate`` share (CARRIER_ROTATE, 25%) of requests: then a Thompson draw over each
    ESTABLISHED carrier's (root, and the CONFIRMED non-vetoed ``carriers``: an island's frontier keeps demoted
    migrants) recent own-increment hit rate (:func:`carrier_rates`), Beta(k r + hits, k (1 - r) + misses) with the
    pooled rate r over k = CARRIER_PRIOR pseudo-children (a carrier without children is drawn near the pool, not at
    1/2; live 2026-10-01: a node's children fit pi0 ~1.0, root's ~0.73). With the predictive model in force, every
    request is :func:`allocate_carrier` instead (2026-10-06: the EB knew carriers were saturated by day 2.75 while 82%
    of nodes kept going to carriers on days 4-6)."""
    if (alloc := allocate_carrier(store, snap, carriers, rng, cfg)) is not None:
        return alloc
    inc = getattr(snap, "incumbent_id", "") or ROOT_NODE_ID
    if rng.random() >= rotate:
        return inc, (f"carrier {inc}: the confirmed incumbent -- your node is judged by its gain over it (its own "
                     f"increment); build on it")
    conf = {n.id for n in store.nodes(status=[NodeStatus.CONFIRMED])} - held_back(store)
    rates = carrier_rates(store, snap, list(dict.fromkeys(
        [ROOT_NODE_ID, *(c for c in carriers if c in conf and not thompson.vetoed(snap, c))])))
    hits, kids = sum(h for h, _ in rates.values()), sum(n for _, n in rates.values())
    r = (hits + 1) / (kids + 2)
    use = prefer_carriers(store, cfg, list(rates))       # sibling sharing; <= 2 carriers while GPUs are scarce
    draw = {c: rng.betavariate(CARRIER_PRIOR * r + h, CARRIER_PRIOR * (1 - r) + n - h)
            for c, (h, n) in rates.items() if c in use}
    pick = max(draw, key=lambda c: (draw[c], c))
    return pick, (f"carrier {pick} (Thompson draw {draw[pick]:.2f} over recent children with P(own gain>0)>"
                  f"{CARRIER_HIT_P}: " + ", ".join(f"{c} {h}/{n}" for c, (h, n) in rates.items())
                  + f"); an exploration carrier ({rotate:.0%} of requests): your node is still judged by "
                  f"P(U > U_incumbent {inc}), never by its gain over this carrier alone")


def crossover_meta(store: Any, parent_ids: Sequence[str]) -> str | None:
    """The review meta-hypothesis whose lever vector M best matches the pair's (largest L.M > 0,
    newest first): the crossover is then that meta's probe and is linked by sign(L.M)."""
    lev = crossover_levers(store, parent_ids).directions
    stale = store.get_state(K_STALE_META) or {}
    best = max(((sum(d * h.levers.directions.get(k, 0) for k, d in lev.items()), h.created_ts, h.id)
                for h in store.hypotheses(kind=HypothesisKind.META) if h.id not in stale), default=(0, 0.0, None))
    return best[2] if best[0] > 0 else None


class PortfolioSelector:
    def __init__(self, ctx: LoopContext, world: Any, embedder: Any = None,
                 rng: random.Random | None = None, *, brief_fn: Callable[..., str] | None = None) -> None:
        self.ctx, self.world = ctx, world
        self.embedder = embedder or HashingEmbedder()
        self.rng = rng or random.Random()
        self.brief_fn = brief_fn or (lambda kind, store, snap, **kw: f"[{kind} brief] {kw}")
        self._xpairs: list = []                  # the last crossover plan's eligible pairs

    @property
    def cfg(self) -> Any:
        return self.ctx.cfg                 # the driver installs self-tune overrides on ctx

    def _mint(self, store: Any, snapshot: WorldSnapshot, *, strategy: Strategy, operator: Operator,
              island: str, parent_ids: tuple[str, ...], brief_kind: str, family: str = "",
              meta_id: str | None = None, bank_id: str = "", carrier: str = "", note: str = "",
              **brief_kw: Any) -> CandidateRequest:
        seq = int(store.get_state(_SEQ, 0)) + 1
        store.set_state(_SEQ, seq)
        rid, ablated = f"req-{seq:05d}", []           # loop.components: web research
        if CG.web_eligible(strategy, operator, brief_kind) and not CG.use(self.ctx, store, "web", "research", rid,
                                                                           request=rid):
            ablated.append("web")
        brief = self.brief_fn(brief_kind, store, snapshot, family=family, parent_ids=parent_ids,
                              **brief_kw, **({"ablated": tuple(ablated)} if ablated else {}))
        brief += (f"\n\nCARRIER: {carrier}" if carrier else "") + (f"\n\n{note}" if note else "")
        persona = cell = ""                            # loop.novelty: the arm's own ideation plan only
        if self.cfg.novelty.enabled and strategy in NV.IDEATION and operator is Operator.DRAFT \
                and not bank_id and meta_id is None:
            persona = NV.persona_for(store, CG.web_eligible(strategy, operator, brief_kind) and "web" not in ablated)
            cell = NV.target(store, family, seq)
        return CandidateRequest(id=rid, task_id=self.ctx.task_id, strategy=strategy,
                                operator=operator, island=island, parent_ids=parent_ids, brief=brief,
                                brief_kind=brief_kind, family_hint=family, meta_hypothesis_id=meta_id,
                                bank_id=bank_id, ablated=tuple(ablated), persona=persona, cell=cell)

    @staticmethod
    def unserve_probe(store: Any, request_id: str) -> list[str]:
        """The review probe(s) a request served that hit an outage were never tried: served again."""
        keys = {e.get("decision_id") for e in store.events("meta_decision") if e.get("component") == "review_probe"
                and (e.get("payload") or {}).get("request") == request_id}
        used = set(store.get_state(_PROBES, []))
        if back := keys & used:
            store.set_state(_PROBES, sorted(used - back))
        return sorted(back)

    def pending_probes(self, store: Any) -> list[tuple[str, str | None, str]]:
        """Unused review probe briefs as (key, meta id | None, text), newest review first (a later
        review has seen more evidence), each review's probes in its own order; never a review's from before the
        last measurement-epoch bump, nor a stale meta's (loop.meta.mark_stale)."""
        used, stale = set(store.get_state(_PROBES, [])), store.get_state(K_STALE_META) or {}
        revs = sorted(current_reviews(store), key=lambda rv: -(rv.created_ts or 0.0))
        return [(k, meta_id, text) for rv in revs for i, (meta_id, text) in enumerate(rv.probe_briefs)
                if (k := f"{rv.id}:{i}") not in used and meta_id not in stale]

    def next_request(self, store: Any, snapshot: WorldSnapshot) -> CandidateRequest | None:
        """The bandit picks the arm FIRST; a pending review probe or idea-bank idea only fills the
        request the arm already asked for: a ``[moonshot]`` probe or a moonshot bank idea a MOONSHOT
        pull (masked at ``bandit.moonshot_max_inflight`` moonshots in flight), any other probe an
        EXPLORE or EXPLOIT pull, any other bank idea an EXPLORE pull. When both a probe and a bank idea
        could fill a pull they alternate. Neither bypasses the bandit, so every arm keeps its floor."""
        from loop.agents.dossiers import base_request
        if (req := base_request(self, store, snapshot)) is not None:          # the challenge-the-base share
            return req
        book = IslandBook(self.cfg, store)
        book.maybe_split(snapshot)
        probes = self.pending_probes(store)
        moon = [p for p in probes if is_moonshot_probe(p[2])]
        plain = [p for p in probes if not is_moonshot_probe(p[2])]
        bank = B.pending(store, self.cfg) if B.paths(self.cfg, store) else []
        bmoon, bplain = [e for e in bank if e["moonshot"]], [e for e in bank if not e["moonshot"]]
        nodes, inflight = store.nodes(), store.get_state(K_INFLIGHT) or []
        plans = {arm: self._plans(arm, book, nodes, snapshot, inflight) for arm in BANDIT_ARMS}
        eligible = [a for a in BANDIT_ARMS if plans[a] or (plain and a in _PROBE_ARMS)
                    or (bplain and a is Strategy.EXPLORE)]
        if not eligible:
            return None
        rid = f"req-{int(store.get_state(_SEQ, 0)) + 1:05d}"     # the request this pull will mint
        pulls = store.pulls()
        if store.get_state(NV.K_BONUS) and CG.use(self.ctx, store, "novelty", "arm", rid, request=rid):
            pulls = NV.with_bonus(store, pulls)             # the arm choice sees the novelty bonuses
        arm = Bandit(self.cfg.bandit, self.rng).rebuild(pulls).choose(eligible)
        island = self.rng.choice(sorted(plans[arm])) if plans[arm] else book.islands[0]
        probe = (moon or [None])[0] if arm is Strategy.MOONSHOT else \
            (plain or [None])[0] if arm in _PROBE_ARMS else None
        idea = (bmoon or [None])[0] if arm is Strategy.MOONSHOT else \
            (bplain or [None])[0] if arm is Strategy.EXPLORE else None
        if idea is not None and probe is not None:          # both waiting: alternate, starve neither
            turn = int(store.get_state(_ALT, 0))
            store.set_state(_ALT, turn + 1)
            idea, probe = (idea, None) if turn % 2 == 0 else (None, probe)
        if idea is not None and not CG.use(self.ctx, store, "scout" if str(idea["id"]).startswith("scout-")
                                           else "idea_banks", "serve", rid, request=rid):
            idea = None                                     # bypassed: the arm's own plan (served later)
        if probe is not None and not CG.use(self.ctx, store, "review_probes", "serve", rid, request=rid):
            probe = None
        carrier, why = pick_carrier(store, snapshot, book.frontier(island, snapshot), self.rng, self.cfg) \
            if arm in (Strategy.MOONSHOT, Strategy.EXPLORE) else (book.best(island, snapshot), "")
        if idea is not None:
            is_moon = arm is Strategy.MOONSHOT
            req = self._mint(store, snapshot, strategy=arm, operator=Operator.DRAFT, island=island,
                             parent_ids=(carrier,), brief_kind="bank", carrier=why,
                             family=idea["family"] if idea["family"] in self.cfg.families else "",
                             bank_id=idea["id"], bank=idea, moonshot=is_moon)
            B.mark(store, idea["id"], req.id)
            return req
        if probe is not None:
            key, meta_id, text = probe
            store.set_state(_PROBES, sorted(set(store.get_state(_PROBES, [])) | {key}))
            is_moon = arm is Strategy.MOONSHOT
            req = self._mint(store, snapshot, strategy=arm,
                             operator=Operator.DRAFT if is_moon else Operator.IMPROVE, island=island,
                             parent_ids=(carrier,), carrier=why,
                             brief_kind="moonshot" if is_moon else "probe",
                             meta_id=meta_id, hypothesis_id=meta_id, probe=text)
            if hasattr(store, "log"):                   # loop.meta.record: which review this request serves
                store.log("meta_decision", component="review_probe", decision_id=key, node_id="",
                          hypothesis_id=meta_id or "", review_id=key.rsplit(":", 1)[0], payload={"request": req.id})
            return req
        if not plans[arm]:                                  # eligible only through a probe/idea now taken
            return None
        operator, parents, kinds, family = plans[arm][island]
        parents = (carrier,) if why else parents         # moonshot / explore: the rotated carrier
        if arm is Strategy.REVISIT and operator is Operator.IMPROVE and len(parents) > 1:
            why = (f"{parents[0]}. REVISIT of closed node {parents[1]} (listed second above, with its diff): rebuild "
                   f"its idea, improved, on carrier {parents[0]}; never on top of {parents[1]} itself")
        if operator is Operator.CROSSOVER and len(self._xpairs) > 1 and not CG.use(
                self.ctx, store, "crossover_choice", "pair", rid, request=rid):
            parents = tuple(n.id for n in self.rng.choice(self._xpairs))    # bypassed: a random eligible pair
        meta = crossover_meta(store, parents) if operator is Operator.CROSSOVER else None
        return self._mint(store, snapshot, strategy=arm, operator=operator, island=island,
                          parent_ids=parents, brief_kind=self.pick_kind(store, kinds), family=family,
                          meta_id=meta, carrier=why)

    def debug_request(self, store: Any, snapshot: WorldSnapshot, node: Node) -> CandidateRequest:
        """The fix-and-continue DEBUG request for a crashed node (the driver's, outside the bandit)."""
        return self._mint(store, snapshot, strategy=Strategy.REVISIT, operator=Operator.DEBUG,
                          island=node.island, parent_ids=(node.id,), brief_kind="failure", family=node.family)

    def winner_cross_request(self, store: Any, snapshot: WorldSnapshot, pair: tuple[str, str]) -> CandidateRequest:
        """A standing crossover of two CONFIRMED winners (the driver's, outside the bandit)."""
        a = store.get_node(pair[0])
        return self._mint(store, snapshot, strategy=Strategy.CROSSOVER, operator=Operator.CROSSOVER,
                          island=IslandBook(self.cfg, store).island_of(a.family), parent_ids=tuple(pair),
                          brief_kind="frontier", meta_id=crossover_meta(store, pair))

    def stack_request(self, store: Any, snapshot: WorldSnapshot, legs: Sequence[str]) -> CandidateRequest:
        """A STACK of unconfirmed small signals (:func:`stack_plan`, the driver's, outside the bandit): a crossover
        of every leg on their shared carrier, confirmed on seeds none of its legs ran (loop.seeds.step)."""
        a = store.get_node(legs[0])
        return self._mint(store, snapshot, strategy=Strategy.CROSSOVER, operator=Operator.CROSSOVER,
                          island=IslandBook(self.cfg, store).island_of(a.family), parent_ids=tuple(legs),
                          brief_kind="frontier", meta_id=crossover_meta(store, legs[:2]),
                          note=(f"STACK of {len(legs)} unconfirmed small signals on carrier {a.parent_id or ROOT_NODE_ID}: "
                                f"{', '.join(legs)} each gained on its own (P(own gain>0) >= {STACK_P} over all its seeds) "
                                "but none confirmed alone. Keep EVERY leg's mechanism exactly as it is (no retuning, no "
                                "dropping a leg): the stack tests whether their small gains add up."))

    def port_request(self, store: Any, snapshot: WorldSnapshot, node_id: str, onto: str, conflict: str) -> CandidateRequest:
        """A backfill PORT whose graft conflicted (loop.backfill.port_requests, the driver's, outside the bandit):
        ``node_id``'s change re-applied by hand on ``onto`` (the incumbent), its parent."""
        node = store.get_node(node_id)
        return self._mint(store, snapshot, strategy=Strategy.CROSSOVER, operator=Operator.CROSSOVER,
                          island=IslandBook(self.cfg, store).island_of(node.family), parent_ids=(onto, node_id),
                          brief_kind="frontier", note=(
                              f"PORT (backfill): re-apply {node_id}'s change -- its diff over its own parent "
                              f"{node.parent_id or ROOT_NODE_ID} -- onto {onto}, keeping {onto} exactly as it is. The "
                              f"automatic graft conflicted ({conflict}): resolve only that by hand. Do not retune or "
                              f"redesign either side; where {onto} already holds part of the change, keep {onto}'s version."))

    def _crossover_parent(self, n: Node, snap: WorldSnapshot) -> bool:
        """CONFIRMED, or closed after a screen (screened_out / not_confirmed failed the PROMOTE bar,
        not the screen evidence) with shrunk P(U>0) >= bandit.crossover_min_p; never vetoed/refuted."""
        post = snap.nodes.get(n.id)
        if thompson.vetoed(snap, n.id):
            return False
        if n.status is NodeStatus.CONFIRMED:
            return True
        ev = snap.hypotheses.get(n.hypothesis_id or "")
        return (n.status in (NodeStatus.SCREENED_OUT, NodeStatus.NOT_CONFIRMED) and post is not None
                and post.p_positive >= self.cfg.bandit.crossover_min_p
                and not (ev is not None and ev.status is HypothesisStatus.REFUTED))

    def pick_kind(self, store: Any, kinds: tuple[str, ...]) -> str:
        """The brief kind, by the same discounted Thompson (with its floor) over kind credits."""
        return kinds[0] if len(kinds) == 1 else \
            Bandit(self.cfg.bandit, self.rng, arms=kinds).rebuild(kind_pulls(store)).choose(kinds)

    def _plans(self, arm: Strategy, book: IslandBook, nodes: Sequence[Node], snap: WorldSnapshot,
               inflight: Sequence[Mapping[str, str]] = ()) -> dict[str, tuple[Operator, tuple[str, ...], tuple[str, ...], str]]:
        """island -> (operator, parent_ids, candidate brief kinds, family_hint); empty = arm masked."""
        live = [n for n in nodes if n.id != ROOT_NODE_ID and n.status is not NodeStatus.REJECTED
                and not design_point(n)]
        rng, out = self.rng, {}
        if arm is Strategy.CROSSOVER:     # confirmed, or strongly screened (closed, not vetoed/refuted)
            tried = {crossover_pair(n) for n in nodes if n.strategy is Strategy.CROSSOVER} | {
                frozenset(r["parents"][:2]) for r in inflight
                if r["strategy"] == Strategy.CROSSOVER.value and len(r.get("parents", ())) > 1}
            tags = {n.id: NV.norm_tag(n.mechanism_tag) or n.id for n in nodes}
            mech = lambda pr: frozenset(tags.get(i, i) for i in pr)   # noqa: E731
            tried_mech = {mech(pr) for pr in tried}     # live a node / a node: a node x a node then x a node, both
            pool = [n for n in live if self._crossover_parent(n, snap)]   # slide_balanced_sampling: one merge
            p_of = lambda n: snap.nodes[n.id].p_positive if n.id in snap.nodes else 0.0   # noqa: E731
            pairs = [(a, b) for i, a in enumerate(pool) for b in pool[i + 1:]
                     if a.family != b.family and frozenset((a.id, b.id)) not in tried
                     and mech((a.id, b.id)) not in tried_mech
                     and a.id not in crossover_pair(b) and b.id not in crossover_pair(a)]
            self._xpairs = pairs
            if pairs:
                a, b = max(pairs, key=lambda p: (p_of(p[0]) * p_of(p[1]), sorted((p[0].id, p[1].id))))
                a, b = (a, b) if p_of(a) >= p_of(b) else (b, a)
                out[book.island_of(a.family)] = (Operator.CROSSOVER, (a.id, b.id), ("frontier",), "")
            return out
        if arm is Strategy.MOONSHOT:      # <= bandit.moonshot_max_inflight bold changes at once, from each island's best
            if moonshots_inflight(nodes, inflight) < self.cfg.bandit.moonshot_max_inflight:
                out = {i: (Operator.DRAFT, (book.best(i, snap),), ("moonshot",), "") for i in book.islands}
            return out
        offspring = thompson.offspring_counts(live)
        revisited = {n.second_parent_id or n.parent_id for n in live if n.strategy is Strategy.REVISIT} | {
            p for r in inflight if r["strategy"] == Strategy.REVISIT.value for p in r.get("parents", ())[-1:]}
        for island in book.islands:
            mine = [n for n in live if not book.split or book.island_of(n.family) == island]
            if arm is Strategy.EXPLOIT:
                p = thompson.pick_exploit_parent(snap, prefer_carriers(book.store, self.cfg, book.frontier(island, snap)), offspring,
                                                 thompson.sigma_panel(self.cfg, snap), rng)
                out[island] = (Operator.IMPROVE, (p,), ("literature",) if p == ROOT_NODE_ID
                               else ("frontier", "literature"), "")
            elif arm is Strategy.EXPLORE:       # one explore researcher per family at a time
                busy = {r["family"] for r in inflight if r["strategy"] == Strategy.EXPLORE.value}
                fams = [f for f in book.families_of(island) if f not in busy]
                if not fams:
                    continue
                counts: dict[str, int] = {}
                for f in [n.family for n in mine] + [r["family"] for r in inflight]:
                    counts[f] = counts.get(f, 0) + 1
                failed = any(n.status in (NodeStatus.SCREENED_OUT, NodeStatus.NOT_CONFIRMED) for n in mine)
                out[island] = (Operator.DRAFT, (book.best(island, snap),),
                               ("literature", "failure") if failed else ("literature",),
                               thompson.pick_family(fams, counts, snap, rng))
            else:                                            # REVISIT: debug a crash, or retry a near miss
                crashes = [n for n in mine if n.status is NodeStatus.CRASHED and n.id not in revisited
                           and n.debug_depth < self.cfg.researcher.max_debug]
                near = [n for n in mine if n.status in (NodeStatus.SCREENED_OUT, NodeStatus.NOT_CONFIRMED)
                        and n.id in snap.nodes and not snap.nodes[n.id].vetoed and n.id not in revisited
                        and 0.3 < snap.nodes[n.id].p_positive < self.cfg.promotion.p_promote
                        and revisit_ok(snap, self.cfg, n, established(book.store, n))]
                if crashes:
                    c = min(crashes, key=lambda n: (n.debug_depth, n.created_ts, n.id))
                    out[island] = (Operator.DEBUG, (c.id,), ("failure",), c.family)
                elif near:                  # a closed node is never a parent: its idea is rebuilt on an established
                    n = max(near, key=lambda n: (snap.nodes[n.id].p_positive, n.id))      # carrier (second parent)
                    out[island] = (Operator.IMPROVE, (established(book.store, n), n.id), ("failure",), n.family)
        return out

    def embedding_of(self, candidate: Candidate) -> tuple[float, ...]:
        return self.embedder.embed(candidate_text(candidate))

    def admit(self, store: Any, snapshot: WorldSnapshot, candidate: Candidate,
              operator: Operator | None = None, verdict: dict | None = None,
              strategy: Strategy | None = None) -> str:
        return _admit(self.ctx, store, candidate, operator=operator, embedder=self.embedder,
                      verdict=verdict, strategy=strategy, snapshot=snapshot)

    def rank_queue(self, store: Any, snapshot: WorldSnapshot, queued: Sequence[Node],
                   levers: bool = True) -> list[Node]:
        return thompson.diversify(thompson.rank_queue(store, self.world, self.cfg, snapshot, queued, self.rng,
                                                      levers=levers),
                                  store.nodes(status=[NodeStatus.SCREENING]))

    def reward(self, store: Any, before: WorldSnapshot, after: WorldSnapshot, node: Node) -> ArmPull:
        sigma_p = thompson.sigma_panel(self.cfg, after)
        pull = compute_reward(store, self.world, self.cfg.bandit, before, after, node, sigma_p=sigma_p)
        if node.status not in (NodeStatus.VOID, NodeStatus.CRASHED, NodeStatus.REJECTED) \
                and (b := NV.bonus(self.ctx, store, node)) > 0:   # never for a near-0-GPU-h failure (/ gpu_h_min)
            per = b / max(pull.gpu_h, self.cfg.bandit.gpu_h_min)       # kept apart: the audited arm choice adds it
            NV.record_bonus(store, node.id, per)
            if hasattr(store, "log"):
                store.log("novelty_bonus", node_id=node.id, arm=node.strategy.value, bonus=round(b, 5),
                          per_gpu_h=round(per, 5))
        kinds = store.get_state(K_KIND_PULLS, [])
        if (kind := brief_of(node)) and node.status is not NodeStatus.VOID \
                and node.id not in {k["node_id"] for k in kinds}:        # an infra void is nobody's idea
            store.set_state(K_KIND_PULLS, kinds + [{"arm": kind, "node_id": node.id, "ts": pull.ts,
                                                    "reward": kind_credit(store, after, node, self.cfg.bandit, sigma_p)}])
        return pull

    def on_confirmed(self, store: Any, snapshot: WorldSnapshot, node: Node) -> None:
        IslandBook(self.cfg, store).on_confirmed(node, snapshot)
