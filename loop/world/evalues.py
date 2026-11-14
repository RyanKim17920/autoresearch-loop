"""E-values (DESIGN §3.7-3.9): the frozen confirmation likelihood ratio, append-only per-hypothesis
trial evidence folded to a verdict, and meta-hypothesis linking (explicit > mechanistic > lever)."""

from __future__ import annotations

import functools
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from loop.contracts import (PANEL, ROOT_NODE_ID, HypothesisEvidence, HypothesisKind,
                            HypothesisRecord, HypothesisStatus, Node, NodePosterior, NodeStatus,
                            Stage, design_point)
from loop.spec import DEFAULT_LEVERS, LoopContext
from loop.world.noise import (SELECTION_STAGES, STATE_KEY, Ledger, offset_ok, pool, resume_var, shared_offset,
                              sibling_groups, sibling_tau)
from loop.world.shrinkage import TargetFit

CONFIRM_KEY = "world.confirm"          # node_id -> [delta_1, sigma_ucb]
# hypothesis_id -> [[node_id, e_for, e_against, x, v, delta_1, tier], ...]  (bounded LR evidence, D3.8;
# tier = how the trial was linked, D3.9)
HYP_KEY = "world.hypotheses.v3"
# the retired lever-linked, first-6, per-meta-alpha book: read once, to log what the re-derivation changed
LEGACY_HYP_KEY = "world.hypotheses.v2"
LEGACY_MAX_TRIALS = 6
# meta id -> sorted config-key / code-identifier names its claim and own probe briefs name, frozen on
# first sight (D3.9 mechanistic links)
META_KEYS = "world.meta_keys"
EXPLICIT, MECHANISTIC, LEVER = "explicit", "mechanistic", "lever"
# Bounded per-trial evidence (D3.8). One trial moves a META hypothesis's E by at most CAP_META, so a
# verdict at the family bar K/alpha needs >= log4(K/alpha) concordant trials (2 for a lone meta, 5 at
# K = 66); a TRIAL hypothesis closes on its one trial, so its cap must exceed 1/alpha. The
# alternative delta_1 (the predicted size) is floored at promotion.delta_min_sigmas and capped at
# DELTA_MAX_SIGMAS target sigmas.
CAP_META = 4.0
CAP_TRIAL = 20.0
DELTA_MAX_SIGMAS = 2.0
SIGMA_UCB_Z = 1.2816                   # one-sided 90%
_SCORED = frozenset({NodeStatus.SCREENED_OUT, NodeStatus.CONFIRMING, NodeStatus.CONFIRMED,
                     NodeStatus.NOT_CONFIRMED, NodeStatus.VETOED, NodeStatus.AUDIT_FAILED})


def phi(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def sigma_ucb(sigma: float, df: float, z: float = SIGMA_UCB_Z) -> float:
    """``sigma * sqrt(df / chi2_lower(df))``, the chi-square quantile by Wilson-Hilferty."""
    k = max(1.0, float(df))
    a = 2.0 / (9.0 * k)
    return sigma * math.sqrt(k / (k * max(1e-6, 1.0 - a - z * math.sqrt(a)) ** 3))


def confirm_delta(sigma: float, alpha: float, n: int) -> float:
    """The confirm alternative ``delta_1 = sigma * sqrt(2 ln(1/alpha) / n)``: the fixed delta that maximises power
    for ``n`` confirm seeds at the bar ``1/alpha`` (E = 1/alpha exactly when the mean confirm increment equals it).
    Data-independent: never the selection-seed estimate (winner's curse), never a floor (live: 45 of 57 confirms
    froze 0.0016, which needs a mean >= 0.0066 on 2 seeds; this one 0.0043, 13% -> 48% power at a true 0.004)."""
    return sigma * math.sqrt(2.0 * math.log(1.0 / alpha) / max(1, n))


def freeze_confirm(ctx: LoopContext, store: Any, node: Node,
                   posts: Mapping[str, NodePosterior]) -> tuple[float, float]:
    """(delta_1, sigma) of a confirming node, frozen on first call. ``delta_1`` is ``node.confirm_alt`` (the
    driver's, frozen at confirm entry) or :func:`confirm_delta` at the node's planned confirm seeds. A control-corrected confirm
    delta has variance ``sigma^2/2 + Var(b_hat) <= sigma^2`` (the shared-control term is at most its
    prior sigma^2/2), so the sigma upper bound also bounds it whatever siblings arrive later."""
    book = dict(store.get_state(CONFIRM_KEY) or {})
    if node.id not in book:
        row = store.get_state(STATE_KEY)[PANEL]
        from loop.seeds import planned_confirm      # loop.seeds imports this package
        sig = sigma_ucb(row["sigma"], row["df_eff"])
        d1 = node.confirm_alt if node.confirm_alt is not None else confirm_delta(
            sig, ctx.cfg.promotion.alpha_confirm, planned_confirm(ctx.cfg, store.runs(node_id=node.id)))
        book[node.id] = [float(d1), sig]
        store.set_state(CONFIRM_KEY, book)
    return tuple(book[node.id])


def prescreened(node: Node) -> bool:
    """A leg-factorial graduate (``legs_of=``, loop.legs): the factorial chose it; its own screen seeds ran after."""
    return " legs_of=" in f" {node.notes or ''}"


def confirm_increments(nd: Any) -> tuple[list[float], list[bool]]:
    """(corrected panel increments, resumed flags) a node's confirm E-value multiplies: its confirm seeds, and for a
    leg-factorial graduate also its screen seeds -- fresh seeds run AFTER the design picked the combination, so
    unselected evidence (3 confirm seeds alone: 60% power at +0.004). An ordinary node's screen seeds selected it
    (winner's curse) and never count."""
    xs = [x for x, _ in nd.confirm]
    rs = [o.resumed for o in nd.obs if o.stage is Stage.CONFIRM and PANEL in o.delta]
    if prescreened(nd.node):
        xs += [x for x, _ in nd.selection.get(PANEL, ())]
        rs += [o.resumed for o in nd.obs if o.stage in SELECTION_STAGES and PANEL in o.delta]
    return xs, rs


def lr_evalue(increments: Sequence[float], delta1: float, sigma: float, extra: Sequence[float] = ()) -> float:
    """Product of ``exp((c*delta1 - delta1^2/2) / (sigma^2 + extra_i))``, computed in logs; ``extra_i``: added
    variance of increment i (a resumed run's seed-level noise, loop.world.noise.resume_var; default 0)."""
    ex = list(extra) + [0.0] * (len(increments) - len(extra))
    log_e = sum((c * delta1 - 0.5 * delta1 * delta1) / (sigma * sigma + v) for c, v in zip(increments, ex))
    return math.exp(min(700.0, log_e))


def calibrate(p: float, kappa: float) -> float:
    """p -> e calibrator ``kappa * p**(kappa-1)``; integrates to 1 over U(0,1). RETIRED for hypothesis
    evidence: it is unbounded as p -> 0, so one Gaussian-tail z of -10 gave E = 1e15 (kept for the
    legacy-book comparison only)."""
    return kappa * min(1.0, max(p, 1e-300)) ** (kappa - 1.0)


def trial_lr(x: float, v: float, delta1: float, cap: float) -> tuple[float, float]:
    """Bounded one-trial likelihood-ratio e-values for an estimate ``x ~ N(mu, v)`` oriented so the
    hypothesis predicts ``mu >= delta1 > 0``: ``LR = exp((delta1*x - delta1^2/2) / v)`` (N(delta1, v)
    over N(0, v)).

    ``e_for = min(cap, LR)`` tests H0: mu <= 0 (E = exp(mu*delta1/v) <= 1); ``e_against =
    min(cap, 1/LR)`` tests H0: mu >= delta1 (E = exp(delta1*(delta1-mu)/v) <= 1). Capping only lowers
    an e-value, so both stay valid, and no single trial moves E by more than ``cap`` whatever its tail
    (a broken node 10 sigma down is one trial of evidence, not 1e15)."""
    if not (v > 0.0 and delta1 > 0.0 and math.isfinite(x)):
        return 1.0, 1.0
    log_lr = max(-700.0, min(700.0, (delta1 * x - 0.5 * delta1 * delta1) / v))
    return min(cap, math.exp(log_lr)), min(cap, math.exp(-log_lr))


def causal_estimate(ledger: Ledger, nid: str, target: str, sigma: float, tau: float,
                    groups: Mapping[tuple[str, str], Sequence[tuple[str, float]]],
                    order: Mapping[str, tuple[float, str]], resume_v: float = 0.0) -> tuple[float, float] | None:
    """(x, v): a node's selection-seed paired deltas pooled, each corrected for its control run's
    deviation using ONLY siblings ordered before it (first submission, then id). Given those
    siblings, ``d - b_hat ~ N(eps, sigma^2/2 + Var(b | earlier))`` exactly, so the per-trial
    e-values are conditionally valid and their product over shared-control siblings stays a test
    supermartingale (a leave-one-out b_hat would use LATER siblings and correlate them)."""
    nd, me = ledger.data[nid], order[nid]
    gs = []
    for o in nd.obs:
        if o.stage not in SELECTION_STAGES or target not in o.delta:
            continue
        sibs = [(y, tau * tau) for j, y in groups.get((o.control, target), ())
                if offset_ok(ledger, nid, j, y, sigma) and order.get(j, (math.inf, j)) < me]
        b, vb = shared_offset(sibs, sigma)
        gs.append((o.delta[target] - b, sigma * sigma / 2.0 + vb + (resume_v if o.resumed else 0.0)))
    off = ledger.offset.get(nid, {}).get(target, (0.0, 0.0))      # a re-measured node: over its measure parent
    return (lambda x, v: (x - off[0], v + off[1]))(*pool(gs)) if gs else None


def fold(pairs: Sequence[Sequence[float]], alpha: float, max_trials: int,
         family: int = 1) -> tuple[float, float, int, HypothesisStatus]:
    """Running products over (e_for, e_against), stopping at the first crossing of
    ``family / alpha`` or at ``max_trials``: deterministic on every prefix. A running product of
    conditionally valid e-values is a test supermartingale, so by Ville P(it ever crosses
    family/alpha under H0) <= alpha/family: over a family of ``family`` hypotheses (each with at most
    one true null of its two) the chance of ANY false verdict is <= alpha (Bonferroni on e-values,
    valid under arbitrary dependence, e.g. metas sharing trials)."""
    bar = max(1, int(family)) / alpha
    e_for = e_against = 1.0
    for n, (a, b) in enumerate(pairs, start=1):
        e_for, e_against = e_for * a, e_against * b
        if e_for >= bar:
            return e_for, e_against, n, HypothesisStatus.SUPPORTED
        if e_against >= bar:
            return e_for, e_against, n, HypothesisStatus.REFUTED
        if n >= max_trials:
            return e_for, e_against, n, HypothesisStatus.INCONCLUSIVE
    status = HypothesisStatus.TESTING if pairs else HypothesisStatus.UNTESTED
    return e_for, e_against, len(pairs), status


def primary_target(h: HypothesisRecord, targets: Sequence[str]) -> tuple[str, int]:
    """(target, predicted sign): PANEL if predicted, else the first known target."""
    for t, p in sorted(h.predictions.items(), key=lambda kv: kv[0] != PANEL):
        if t in targets:
            return t, 1 if p.sign >= 0 else -1
    return PANEL, 1


def delta_one(ctx: LoopContext, h: HypothesisRecord, target: str, sigma: float,
              size: float | None = None) -> float:
    """The pre-specified alternative: the predicted size floored at the
    minimum meaningful effect (``promotion.delta_min_sigmas`` sigma) and capped at ``DELTA_MAX_SIGMAS`` sigma."""
    p = h.predictions.get(target)
    size = max(0.0, size) if size is not None else abs(p.size) if p is not None and p.size else 0.0
    lo = ctx.cfg.promotion.delta_min_sigmas * sigma
    return min(DELTA_MAX_SIGMAS * sigma, max(size, lo))


class LinkIndex:
    """One refit's view of every non-root node's link facts: its TARGETS (the leaf names of its
    changed config keys plus the identifiers on its diff's changed lines) and its registered
    hypothesis's text; plus the config-key vocabulary seen so far and the reviews (D3.9)."""

    def __init__(self, store: Any) -> None:
        hyps = {h.id: h for h in store.hypotheses()}
        self.nodes = [n for n in store.nodes() if n.id != ROOT_NODE_ID]
        self.targets: dict[str, frozenset[str]] = {}
        self.text: dict[str, str] = {}
        known: set[str] = set()
        for n in self.nodes:
            keys = {str(k).lower() for k in (store.get_state(f"config_diff:{n.id}") or {})}
            known |= keys
            self.targets[n.id] = frozenset({_leaf(k) for k in keys}) | _diff_idents(n.diff_path)
            h = hyps.get(n.hypothesis_id or "")
            self.text[n.id] = " ".join((h.claim, h.falsification, h.mechanism_tag)) if h else ""
        self.known = frozenset(known)
        self.reviews = {rv.id: rv for rv in store.reviews()}


_IDENT = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)+")


def _leaf(key: str) -> str:
    return key.rsplit(".", 1)[-1]


@functools.lru_cache(maxsize=8192)
def _diff_idents(path: str) -> frozenset[str]:
    """snake_case identifiers on a diff's changed (+/-) lines, file headers excluded (diffs are
    written once at intake, so caching by path is exact)."""
    try:
        text = Path(path).read_text(errors="replace") if path else ""
    except OSError:
        return frozenset()
    changed = [ln[1:] for ln in text.splitlines()
               if ln[:1] in "+-" and not ln.startswith(("+++", "---"))]
    return frozenset(_IDENT.findall("\n".join(changed).lower()))


def key_names(text: str, known: Iterable[str]) -> set[str]:
    """The config keys / code identifiers a meta-hypothesis names: every snake_case identifier in
    its text, plus each known config key named in prose -- by dotted key, by its snake_case leaf
    (separators normalised, so "local-crop scale" names ``local_crop_scale``) or, for a leaf of three
    or more segments, by its stem ("local-crop floor" names ``local_crop_scale``). A one- or
    two-segment leaf needs its full name: "LR" in prose is a lever, not ``dino.lr``, and "local" or
    "KDE" alone would name every ``local_*`` / ``kde_*`` knob."""
    low = text.lower()
    norm = "_" + re.sub(r"[^a-z0-9]+", "_", low).strip("_") + "_"
    out = {t for t in _IDENT.findall(low) if len(t) >= 5 and t not in DEFAULT_LEVERS}   # a lever is no key
    for key in known:
        leaf = _leaf(key)
        stem = leaf.rsplit("_", 1)[0] if leaf.count("_") >= 2 else ""
        dotted = "." in key and re.search(rf"(?<![a-z0-9_.]){re.escape(key)}(?![a-z0-9_])", low)
        if dotted or ("_" in leaf and (f"_{leaf}_" in norm or (stem and f"_{stem}" in norm))):
            out.add(leaf)
    return out


def meta_keys(store: Any, h: HypothesisRecord, index: LinkIndex) -> frozenset[str]:
    """The key names in the meta's claim, falsification and its OWN probe briefs, frozen the first
    time the meta is seen (the refit right after its review is stored)."""
    book = store.get_state(META_KEYS) or {}
    if h.id not in book:
        rv = index.reviews.get(h.review_id or "")
        probes = [b for mid, b in (rv.probe_briefs if rv else ()) if mid == h.id]
        book = {**book, h.id: sorted(key_names(" ".join((h.claim, h.falsification, *probes)),
                                              index.known))}
        store.set_state(META_KEYS, book)
    return frozenset(book[h.id])


def meta_links(store: Any, h: HypothesisRecord, index: LinkIndex) -> list[tuple[str, int, str]]:
    """(node id, sign, tier) for every node created after the meta (not rejected, not a designed
    batch's point: a factorial point moves several knobs at once on a shared block, so it is evidence
    for the batch's OLS effects, never for one meta's mechanism), strongest tier first (D3.9): EXPLICIT -- the node came from one of the meta's own probe briefs (an ``auto=False``
    link written at intake), or its registered hypothesis is / names the meta id; MECHANISTIC -- its
    targets overlap the meta's key names. LEVER (``L.M != 0``) links are a fallback used only while
    the meta has no explicit or mechanistic link at all. The sign is sign(L.M) when non-zero, else +1
    (a probe moves the meta's levers the claimed way)."""
    m = {l: d for l, d in h.levers.directions.items() if d}
    pinned = dict(store.links(h.id, explicit_only=True))
    named = re.compile(re.escape(h.id) + r"(?![A-Za-z0-9])")
    keys = meta_keys(store, h, index)
    strong, weak = [], []
    for n in index.nodes:
        if n.status is NodeStatus.REJECTED or design_point(n) or not n.created_ts > h.created_ts:
            continue
        dot = sum(d * m.get(l, 0) for l, d in n.levers.directions.items())
        sign = -1 if dot < 0 else 1
        if n.id in pinned or n.hypothesis_id == h.id or named.search(index.text[n.id]):
            strong.append((n.id, pinned.get(n.id, sign), EXPLICIT))
        elif keys & index.targets[n.id]:
            strong.append((n.id, sign, MECHANISTIC))
        elif dot:
            weak.append((n.id, sign, LEVER))
    return strong or weak


def all_evidence(ctx: LoopContext, store: Any, ledger: Ledger,
                 fits: Mapping[str, TargetFit]) -> dict[str, HypothesisEvidence]:
    """Append newly scored linked trials (pre-registered: submitted after the hypothesis was
    created) to each hypothesis's frozen list as bounded LR e-values, then fold (D3.8).

    A trial's estimate is its own selection-seed paired deltas (screen + promote: unselected with
    respect to the hypothesis, and the Gaussian LR is valid under the adaptive promote), each
    corrected causally for the shared control run, with the MEASUREMENT variance
    ``sigma^2/2 + Var(b_hat)`` per seed -- never a shrunk posterior. A META hypothesis adds the
    between-implementation spread ``tau_node^2 + tau_tag^2``: its claim is the MEAN effect of a lever
    direction, and without that term one bad implementation of a true direction would count as
    evidence against it (the LR is only valid for the marginal variance).

    A META hypothesis's trials are its EXPLICIT / MECHANISTIC ``meta_links``: a LEVER-tier link (a node that merely
    moves the meta's levers) is never evidence (live 2026-10-03: rv-36-m3 'gradient balance' reached E_for 33.9 on
    22 lever links from unrelated readout / stain-dropout nodes), and rows booked on one are folded out; its
    fold runs over up to ``hypotheses.max_trials`` of them and decides at ``K / alpha``, K = the number
    of META hypotheses registered (Bonferroni on e-values: the chance that ANY meta shows a false
    verdict is <= alpha). K grows as reviews add metas, so a META status is a report at the current
    family size; the book rows stay append-only. A TRIAL hypothesis is unchanged (its one linked
    trial, 1/alpha). The first fit under this rule re-derives every hypothesis from scratch and logs
    the statuses that changed."""
    legacy = None if store.get_state(HYP_KEY) is not None else store.get_state(LEGACY_HYP_KEY)
    book = dict(store.get_state(HYP_KEY) or {})
    if legacy is not None:              # a TRIAL's rule is unchanged: its frozen row carries over
        book.update({h.id: [[*r[:6], EXPLICIT] for r in legacy.get(h.id, [])]
                     for h in store.hypotheses(kind=HypothesisKind.TRIAL) if legacy.get(h.id)})
    cfg, out, changed = ctx.cfg.hypotheses, {}, store.get_state(HYP_KEY) is None
    groups = sibling_groups(ledger)
    sigma = {t: f.sigma for t, f in fits.items()}
    tau = sibling_tau(ctx, groups, sigma)
    order = {nid: (nd.first_submitted_ts, nid) for nid, nd in ledger.data.items()}
    hyps = store.hypotheses()
    n_meta = sum(h.kind is HypothesisKind.META for h in hyps)
    index = LinkIndex(store) if n_meta else None
    diffs = {}
    for h in hyps:
        if h.kind is HypothesisKind.STACK:      # stack(A,B) (retired: the pairwise model had no skill): no evidence
            continue
        trial = h.kind is HypothesisKind.TRIAL
        target, sign_h = primary_target(h, tuple(fits))
        links = [(nid, s, EXPLICIT) for nid, s in store.links(h.id)] if trial \
            else [lk for lk in meta_links(store, h, index) if lk[2] != LEVER]   # a lever link is no evidence
        d1 = delta_one(ctx, h, target, sigma[target])
        cap = CAP_TRIAL if trial else CAP_META
        comps = fits[target].comps      # a META claim is about a POPULATION of implementations:
        spread = 0.0 if trial else comps.tau_node ** 2 + comps.tau_tag ** 2
        rows = list(book.get(h.id, []))
        seen, fresh = {r[0] for r in rows}, []
        for nid, sign_j, tier in links:
            nd = ledger.data.get(nid)
            if (nid in seen or nd is None or nid not in fits[target].obs
                    or nd.node.status not in _SCORED or not nd.first_submitted_ts > h.created_ts):
                continue
            est = causal_estimate(ledger, nid, target, sigma[target], tau[target], groups, order,
                                  resume_var(ctx, sigma, target))
            if est is None:
                continue
            x = sign_j * sign_h * est[0]
            fresh.append((nd.first_submitted_ts, nid, x, est[1] + spread, tier))
        if fresh:
            rows += [[nid, *trial_lr(x, v, d1, cap), x, v, d1, tier]
                     for _, nid, x, v, tier in sorted(fresh)]
            book[h.id], changed = rows, True
        max_trials = 1 if trial else cfg.max_trials
        ev = [r for r in rows if r[6] != LEVER]          # rows booked on a lever link before 2026-10-03 fold out
        e_for, e_against, n, status = fold([r[1:3] for r in ev], cfg.alpha, max_trials,
                                           1 if trial else n_meta)
        out[h.id] = HypothesisEvidence(h.id, e_for, e_against, n, status,
                                       tuple(r[0] for r in ev[:n]))
        if legacy is not None:
            old_rows = legacy.get(h.id, [])
            old = fold([r[1:3] for r in old_rows], cfg.alpha, 1 if trial else LEGACY_MAX_TRIALS)
            if old[3] is not status or [r[0] for r in old_rows[:old[2]]] != [r[0] for r in rows[:n]]:
                diffs[h.id] = {"old": old[3].value, "new": status.value,
                               "old_e": [old[0], old[1], old[2]], "new_e": [e_for, e_against, n],
                               "old_trials": [r[0] for r in old_rows[:old[2]]],
                               "new_trials": [[r[0], r[6]] for r in rows[:n]]}
    if changed:
        store.set_state(HYP_KEY, book)
    if legacy is not None:
        store.log("hypotheses_rederived", changed=diffs, n_hypotheses=len(out), n_meta=n_meta,
                  rule="explicit > mechanistic > lever links; up to max_trials linked trials; "
                       "META decided at K/alpha (D3.8-3.9)")
    return out


def link_meta(store: Any, hypothesis: HypothesisRecord,
              index: LinkIndex | None = None) -> list[tuple[str, int]]:
    """Record the meta's current ``meta_links`` as auto-links (the dashboard's view; the evidence
    recomputes the tiers itself) and return the newly added (node id, sign) (D3.9)."""
    have = {nid for nid, _ in store.links(hypothesis.id)}
    out = []
    for nid, sign, _tier in meta_links(store, hypothesis, index or LinkIndex(store)):
        if nid not in have:
            store.link(hypothesis.id, nid, sign=sign, auto=True)
            out.append((nid, sign))
    return out
