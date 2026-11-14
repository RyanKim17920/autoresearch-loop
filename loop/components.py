"""loop.components — CAUSAL AUDITS of the loop's OPTIONAL components (DESIGN §6.4).

Each optional component (the registry below) sits behind :func:`use` at every decision it would influence.
With probability = its ablation rate the decision is BYPASSED (a seeded draw, logged ``component_gate`` and
printed ``ablate <component> <decision>``), so its value is measured causally: the outcomes of the nodes a
decision produced (screened hit = own increment > 0, P(gain>0) > 0.5, confirmations, GPU-h per success) WITH vs WITHOUT it.

:func:`audit` (with every first-principles review, and ``python -m loop audit-components``) gives each
component a verdict -- useful / no_measurable_value / harmful / insufficient -- with its evidence (effect on
the hit rate ± se, n ablated decisions) and ACTS: no value after >= ``min_ablated`` decisions -> ``down``
(bypassed at ``down_rate``), after >= 2x that -> ``off``; harmful -> ``off``; useful -> ``on``. An ``off``
component is still used on ``reprobe_rate`` of its decisions, so it can come back. Before any randomised
evidence exists, a component's own decision log could stand in ("natural" evidence; none is registered since
the judge and the tournament were removed, 2026-10-06).

The statistics gates, fresh-seed confirmation, the held-out audit, locks, budget, the eval guard, the
Labless gate and init-compat are NOT components: :func:`use` refuses them (``PROTECTED``)."""

from __future__ import annotations

import hashlib
import math
import re
import time
from collections import defaultdict
from typing import Any, Collection, Mapping, Sequence

from loop.contracts import OPTIONAL_COMPONENTS, PROTECTED_COMPONENTS, NodeStatus, Operator, Strategy, outage_requests

K_GATE, K_ATTACH = "component_gate", "component_attach"   # log events: a drawn decision; units found later
K_MODES = "components.modes"          # {name: {mode, ts, verdict, effect, se, n}}: the audit's actions
K_DRAWS = "components.draws"          # draw counter (the seeded draw's index)
K_STICKY = "components.sticky"        # {"name:key": used}: a node-level decision is drawn once
K_LIVE = "components.live"            # {component: gate key} whose state the queue currently reflects
K_LAST = "components.audit"           # the last audit {ts, rows}
LINEAGE_WAIT_S = 48 * 3600.0          # a lineage decision with no scored descendant resolves (as a miss) after this

#: component -> what it influences (bypass behaviour in brackets)
REGISTRY: Mapping[str, str] = {
    "review_probes": "a pending review probe fills the pull [the arm's own plan]",
    "lever_prior": "lever-trend term in the queue predictive [ranked without it]",
    "design_auto": "auto-picked knobs topping up a designed batch [review knobs only]",
    "crossover_choice": "crossover pair = best P(U>0) product [a random eligible pair]",
    "scout": "research-scout runs and its ideas filling pulls [skipped / the arm's own plan]",
    "web": "web research in ideation sessions [no web tools]",
    "idea_banks": "an operator idea-bank idea fills the pull [the arm's own plan]",
    "novelty": "novelty bonus in the bandit rewards the arm choice sees [raw rewards]",
}
assert tuple(REGISTRY) == OPTIONAL_COMPONENTS
WEB_STRATEGIES = (Strategy.MOONSHOT, Strategy.EXPLORE)
WEB_BRIEFS = ("literature", "bank", "moonshot", "base")
_LINES: list[str] = []


def web_eligible(strategy: Strategy, operator: Operator, brief_kind: str) -> bool:
    """Literature-grounded ideation gets web tools (never a DEBUG fix or a tuning variant)."""
    return operator is not Operator.DEBUG and (strategy in WEB_STRATEGIES or brief_kind in WEB_BRIEFS)


# ------------------------------------------------------------------ the gate
def modes(store: Any) -> dict[str, dict[str, Any]]:
    return dict(store.get_state(K_MODES) or {})


def mode(cfg: Any, store: Any, name: str) -> str:
    if not cfg.components.enabled.get(name, True):
        return "disabled"
    return modes(store).get(name, {}).get("mode", "on")


def rate(cfg: Any, store: Any, name: str) -> float:
    """P(bypass) for the component's next decision."""
    cc = cfg.components
    base = cc.rates.get(name, cc.ablation_rate)
    m = mode(cfg, store, name)
    return 1.0 if m == "disabled" else base if m == "on" else max(base, cc.down_rate) if m == "down" \
        else 1.0 - cc.reprobe_rate


def use(ctx: Any, store: Any, name: str, decision: str, key: Any, *, sticky: bool = False, **units: Any) -> bool:
    """True = use the component for this decision; False = bypass it. ``units`` say which nodes the decision
    produced: ``node``, ``nodes``, ``request`` (nodes researched from it), ``tag`` (mechanism tag), and
    ``lineage=True`` (the node's later descendants, one outcome per decision)."""
    if name not in REGISTRY:
        raise ValueError(f"{name!r} is not an optional component" + (
            " (protected: never ablated or disabled)" if name in PROTECTED_COMPONENTS else ""))
    cfg = ctx.cfg
    if not hasattr(store, "log"):                       # a store double without a log runs no experiment
        return True
    if not cfg.components.enabled.get(name, True):
        return False
    sk = f"{name}:{key}"
    book = (store.get_state(K_STICKY) or {}) if sticky else {}
    if sk in book:
        return bool(book[sk])
    r = rate(cfg, store, name)
    if r <= 0.0:                                       # no experiment running: nothing to log
        return True
    n = int(store.get_state(K_DRAWS, 0) or 0)
    store.set_state(K_DRAWS, n + 1)
    h = hashlib.sha256(f"{cfg.components.seed}|{ctx.task_id}|{name}|{decision}|{key}|{n}".encode()).hexdigest()
    used = int(h[:13], 16) / 16 ** 13 >= r
    store.log(K_GATE, component=name, decision=decision, key=str(key), used=used, rate=round(r, 4), **units)
    if sticky:
        store.set_state(K_STICKY, {**book, sk: used})
    if not used:
        _LINES.append(f"ablate {name} {decision} ({key})")
    return used


def attach(store: Any, name: str, key: Any, nodes: Sequence[str]) -> None:
    """Units found after the draw (the lever prior: the queue heads launched under it)."""
    if nodes:
        store.log(K_ATTACH, component=name, key=str(key), nodes=list(nodes))


def take_lines() -> list[str]:
    out = list(_LINES)
    _LINES.clear()
    return out


# ------------------------------------------------------------------ outcomes
def _arm() -> dict[str, Any]:
    return {"decisions": 0, "scored": 0, "hits": 0, "conf": 0, "gpu": 0.0}


def _contrast(a: Mapping[str, Any], b: Mapping[str, Any]) -> tuple[float, float]:
    """(hit-rate difference with - without, its se); nan when either arm has no scored unit."""
    if not a["scored"] or not b["scored"]:
        return math.nan, math.nan
    p = [(x["hits"] + 0.5) / (x["scored"] + 1.0) for x in (a, b)]      # se never 0 at 0 or 100%
    se = math.sqrt(p[0] * (1 - p[0]) / a["scored"] + p[1] * (1 - p[1]) / b["scored"])
    return a["hits"] / a["scored"] - b["hits"] / b["scored"], se


def randomized(store: Any, f: Any, now: float) -> dict[str, dict[bool, dict[str, Any]]]:
    """component -> {used: arm stats} over every logged draw and the nodes it produced."""
    by_req, by_tag, kids = defaultdict(list), defaultdict(list), defaultdict(list)
    for n in f.nodes.values():
        if m := re.search(r"\brequest=(\S+)", n.notes or ""):
            by_req[m.group(1)].append(n.id)
        by_tag[n.mechanism_tag].append(n.id)
        for p in {n.parent_id, n.second_parent_id} - {None}:
            kids[p].append(n)
    att = defaultdict(list)
    for e in store.events(K_ATTACH):
        att[(e["component"], e["key"])] += e.get("nodes") or []

    def descendants(nid: str, ts: float) -> list[str]:
        out, todo = [], [nid]
        while todo:
            for c in kids.get(todo.pop(), ()):
                if c.id not in out and (c.created_ts or 0.0) >= ts - 1.0:
                    out.append(c.id)
                    todo.append(c.id)
        return out

    res: dict[str, dict[Any, Any]] = {c: {True: _arm(), False: _arm(), "strata": {}} for c in REGISTRY}
    outage = outage_requests(store)                       # a request an outage ate decided nothing

    def add(arms: Sequence[dict[str, Any]], **d: Any) -> None:
        for a in arms:
            for k, v in d.items():
                a[k] += v
    for e in store.events(K_GATE):
        if e.get("component") not in res or (e.get("request") and e["request"] in outage):
            continue
        used, r = bool(e.get("used")), res[e["component"]]   # stratum = the bypass rate in force (mode changes)
        arms = (r[used], r["strata"].setdefault(round(float(e.get("rate") or 0.0), 4), {True: _arm(), False: _arm()})[used])
        add(arms, decisions=1)
        if e.get("lineage") and e.get("node"):            # one outcome per decision: did the lineage hit?
            ds = descendants(e["node"], float(e["ts"]))
            sc = [i for i in ds if f.hit(i) is not None]
            if sc or now - float(e["ts"]) >= LINEAGE_WAIT_S:
                add(arms, scored=1, hits=int(any(f.hit(i) for i in sc)), gpu=sum(f.gpu[i] for i in ds),
                    conf=int(any(f.nodes[i].status in (NodeStatus.CONFIRMED, NodeStatus.AUDIT_FAILED) for i in ds)))
            continue
        ids = set(e.get("nodes") or []) | ({e["node"]} if e.get("node") else set()) \
            | set(by_req.get(e.get("request") or "", ())) | set(by_tag.get(e.get("tag") or "", ())) \
            | set(att.get((e["component"], e["key"]), ()))
        for i in ids & set(f.nodes):
            h = f.hit(i)
            add(arms, scored=int(h is not None), hits=int(bool(h)), gpu=f.gpu[i],
                conf=int(f.nodes[i].status in (NodeStatus.CONFIRMED, NodeStatus.AUDIT_FAILED)))
    return res


def _stratified(strata: Mapping[float, Mapping[bool, Mapping[str, Any]]]) -> tuple[float, float]:
    """Inverse-variance pooled with-minus-without contrast over bypass-rate strata: pooling raw arms across a
    mode change (0.15 -> 0.5 -> 0.95 bypass) would confound the component with time."""
    parts = [(d, se) for s in strata.values() for d, se in [_contrast(s[True], s[False])] if d == d]
    if not parts:
        return math.nan, math.nan
    w = [1.0 / max(se * se, 1e-12) for _, se in parts]
    return sum(wi * d for wi, (d, _) in zip(w, parts)) / sum(w), math.sqrt(1.0 / sum(w))


def _cluster_contrast(w: Sequence[float], wo: Sequence[float]) -> tuple[float, float]:
    """(mean(with) - mean(without), its se) over SESSION-level outcomes: the se comes from the spread BETWEEN sessions
    (clustered by session), never from the nodes inside one. nan when either arm has no session."""
    if not w or not wo:
        return math.nan, math.nan

    def var(x: Sequence[float]) -> float:
        m, n = sum(x) / len(x), len(x)
        s2 = sum((v - m) ** 2 for v in x) / (n - 1) if n > 1 else 0.0
        p = (m * n + 0.5) / (n + 1.0)
        return max(s2, p * (1 - p) if s2 == 0.0 else 0.0) / n     # a zero spread is not a zero se
    return sum(w) / len(w) - sum(wo) / len(wo), math.sqrt(var(w) + var(wo))


def _cluster_stratified(ys: Mapping[Any, Mapping[bool, Sequence[float]]]) -> tuple[float, float]:
    """Inverse-variance pooled :func:`_cluster_contrast` over the (bypass rate, brief kind) strata."""
    parts = [(d, se) for s in ys.values() for d, se in [_cluster_contrast(s[True], s[False])] if d == d]
    if not parts:
        return math.nan, math.nan
    w = [1.0 / max(se * se, 1e-12) for _, se in parts]
    return sum(wi * d for wi, (d, _) in zip(w, parts)) / sum(w), math.sqrt(1.0 / sum(w))


NATURAL: dict[str, Any] = {}      # component -> f(facts) -> natural evidence (see the module docstring)


def verdict(effect: float, se: float, n: int, min_n: int, margin: float | None = None) -> str:
    """useful / harmful at +-2 se; a null is ``no_measurable_value`` only when the effect's upper 95% bound is
    below ``margin`` (equivalence: confidently not helping meaningfully), else ``inconclusive`` (underpowered)."""
    if n < min_n or effect != effect:
        return "insufficient"
    if not se:
        v = "no_measurable_value" if effect == 0 else "useful" if effect > 0 else "harmful"
    else:
        v = "useful" if effect >= 2 * se else "harmful" if effect <= -2 * se else "no_measurable_value"
    if v == "no_measurable_value" and margin is not None and effect + 1.96 * se >= margin:
        return "inconclusive"
    return v


def next_mode(cur: str, v: str, n: int, min_n: int, randomised: bool = True) -> str:
    """Only randomised ablation evidence can turn a component ``off``; natural (non-randomised) evidence moves
    it at most to ``down``; an inconclusive verdict leaves the mode alone (keep collecting)."""
    if cur == "disabled":
        return cur
    if v == "useful":
        return "on"
    if v == "harmful":
        return "off" if randomised or cur == "off" else "down"
    if v == "no_measurable_value":
        if cur == "off":
            return cur
        return "off" if randomised and n >= 2 * min_n else "down"
    return cur


def evaluate(ctx: Any, store: Any, snapshot: Any, now: float | None = None) -> list[dict[str, Any]]:
    """One row per component: arms, effect ± se, n, source, verdict, current and next mode (read-only)."""
    from loop.meta import _Facts
    now = time.time() if now is None else now
    f = _Facts(store, snapshot)
    arms, cc, rows = randomized(store, f, now), ctx.cfg.components, []
    for c in REGISTRY:
        w, wo = arms[c][True], arms[c][False]
        eff, se = _stratified(arms[c]["strata"])
        ev = {"effect": eff, "se": se, "n": wo["decisions"], "source": "ablation"}
        if wo["decisions"] < cc.min_ablated and c in NATURAL and snapshot is not None:
            if (nat := NATURAL[c](f)) is not None and nat["n"] >= cc.min_ablated:
                ev = nat
        v = verdict(ev["effect"], ev["se"], ev["n"], cc.min_ablated, cc.equivalence_margin)
        cur = mode(ctx.cfg, store, c)
        rows.append({"component": c, "with": w, "without": wo, **ev, "verdict": v, "mode": cur,
                     "next": next_mode(cur, v, ev["n"], cc.min_ablated, not str(ev["source"]).startswith("natural"))})
    return rows


def audit(ctx: Any, store: Any, snapshot: Any, now: float, *, apply: bool = True) -> list[str]:
    """Evaluate every component, store the table (``components.audit``) and apply the mode changes."""
    rows = evaluate(ctx, store, snapshot, now)
    acts, book = [], modes(store)
    for r in rows:
        if apply and r["next"] != r["mode"]:
            book[r["component"]] = {"mode": r["next"], "ts": now, "verdict": r["verdict"], "n": r["n"],
                                    "effect": _num(r["effect"]), "se": _num(r["se"]), "source": r["source"]}
            store.log("component_mode", component=r["component"], was=r["mode"], mode=r["next"],
                      verdict=r["verdict"], effect=_num(r["effect"]), se=_num(r["se"]), n=r["n"], source=r["source"])
            acts.append(f"component {r['component']}: {r['mode']} -> {r['next']} ({r['verdict']}, "
                        f"{_fmt(r)}; {r['source']})")
    if apply:
        store.set_state(K_MODES, book)
        store.set_state(K_LAST, {"ts": now, "rows": [_slim(r) for r in rows]})
        store.log("component_audit", rows=[_slim(r) for r in rows])
    return acts


def _num(x: float) -> float | None:
    return None if x != x else round(float(x), 5)


def _slim(r: Mapping[str, Any]) -> dict[str, Any]:
    return {k: (_num(v) if isinstance(v, float) else v) for k, v in r.items()}


def _fmt(r: Mapping[str, Any]) -> str:
    e, s = r["effect"], r["se"]
    return ("effect -" if e is None or e != e else f"effect {e:+.2f}±{(s or 0.0):.2f}") + f", n={r['n']}"


def _arm_txt(a: Mapping[str, Any]) -> str:
    if not a["decisions"]:
        return "-"
    per = f"{a['gpu'] / (a['hits'] + a['conf']):.1f}" if a["hits"] + a["conf"] and a["gpu"] else "-"
    return f"{a['decisions']} dec, hit {a['hits']}/{a['scored']}, conf {a['conf']}, GPU-h/success {per}"


def lines(ctx: Any, store: Any, snapshot: Any) -> list[str]:
    """``loop status``: the live audit table (the modes in force are the last audit's)."""
    last = store.get_state(K_LAST) or {}
    head = ("component audits (verdict on screened hit rate WITH - WITHOUT, se, n ablated | mode -> next | "
            "with / without)" + (f"; last applied {time.strftime('%Y-%m-%d %H:%M', time.gmtime(last['ts']))}"
                                 if last.get("ts") else "; not applied yet (first-principles review / audit-components)"))
    out = [head]
    for r in evaluate(ctx, store, snapshot):
        out.append(f"  {r['component']:16s} {r['verdict']:20s} {_fmt(r)} [{r['source'][:70]}] | {r['mode']}"
                   + (f" -> {r['next']}" if r["next"] != r["mode"] else "")
                   + f" | {_arm_txt(r['with'])} / {_arm_txt(r['without'])}")
    return out


def review_block(store: Any) -> str:
    """Concise, for the review prompts: the last audit's verdicts that say something, and the modes."""
    rows = (store.get_state(K_LAST) or {}).get("rows") or []
    said = [f"{r['component']}: {r['verdict']} ({_fmt(r)}, {r['source'][:60]}) -> {r['next']}"
            for r in rows if r.get("verdict") != "insufficient"]
    quiet = [r["component"] for r in rows if r.get("verdict") == "insufficient"]
    if not rows:
        return "component audit: none yet"
    return "\n".join(["component audit (causal ablation; code-computed, actions automatic):", *said]
                     + ([f"insufficient evidence yet: {', '.join(quiet)}"] if quiet else []))


# ------------------------------------------------------------------ v3: randomized bypass + the meta report's verdict
#: the v3 optional components, each a context part the v3 researcher honours through ``request.ablated`` (``web``:
#: ``researcher.wants_web``; ``postmortem``: the brief's parent-postmortem block). A fork-and-flush BRANCH request is never
#: drawn: ``fnf.FLUSH`` already ablates its whole context as its own A/B (``request.ablated`` is shared, the draws are not).
V3_COMPONENTS = ("web", "postmortem")
#: 11-08: brief sections that exist only when their feature is on (loop.board / loop.root_audit), audited the same way


def v3_components(ctx: Any) -> tuple[str, ...]:
    cfg = getattr(ctx, "cfg", None)
    extra = tuple(c for c in ("board", "root_audit", "local_runs") if getattr(getattr(cfg, c, None), "enabled", False))
    return V3_COMPONENTS + extra

V3_VERSION = 3        # ``component_gate`` events of v3 carry ``v=3``: the v2 era's draws (10-08, other modes / registry) never mix in
V3_ARM_MIN_SEEDS = 2  # screen seeds before a node's own-increment posterior counts as a scored outcome


def draw(ctx: Any, store: Any, name: str, decision: str, key: Any, **units: Any) -> bool:
    """True = use the component for this decision, False = bypass it: a seeded draw at ``components.rates[name]`` /
    ``ablation_rate``, logged ``component_gate`` (v=3). Report-only: the v2 stored modes (``components.modes``: web / novelty
    / postmortem were audited ``off`` on 10-07..08, in another regime) are NOT read, so the v3 audit starts from a clean
    experiment and a verdict never acts on the loop by itself (:func:`v3_section` only reports)."""
    if name not in v3_components(ctx):
        raise ValueError(f"{name!r} is not a v3 optional component {v3_components(ctx)}")
    cc = ctx.cfg.components
    if not cc.enabled.get(name, True):
        return False
    r = float(cc.rates.get(name, cc.ablation_rate))
    if r <= 0.0 or not hasattr(store, "log"):
        return True
    n = int(store.get_state(K_DRAWS, 0) or 0)
    store.set_state(K_DRAWS, n + 1)
    h = hashlib.sha256(f"{cc.seed}|{ctx.task_id}|{name}|{decision}|{key}|{n}".encode()).hexdigest()
    used = int(h[:13], 16) / 16 ** 13 >= r
    store.log(K_GATE, component=name, decision=decision, key=str(key), used=used, rate=round(r, 4), v=V3_VERSION, **units)
    return used


def v3_section(ctx: Any, store: Any, nodes: Mapping[str, Any], gpu: Mapping[str, float],
               beliefs: Mapping[str, Any], passed: Collection[str]) -> dict[str, Any]:
    """The meta report's component audit: per v3 component, the nodes of its drawn sessions (``request=<id>`` in the node
    notes) WITH vs WITHOUT it -- screened hit rate (own-increment posterior mean > 0 with >= ``V3_ARM_MIN_SEEDS`` seeds,
    ``beliefs`` node -> (m, v, n)), test passes, GPU-h -- the with-minus-without difference +- se, and the verdict
    (:func:`verdict`; ``insufficient`` below ``components.min_ablated`` ablated decisions). Read-only."""
    cc, outage = ctx.cfg.components, outage_requests(store)
    by_req: dict[str, list[str]] = defaultdict(list)
    for nid, n in nodes.items():
        if m := re.search(r"\brequest=(\S+)", getattr(n, "notes", "") or ""):
            by_req[m.group(1)].append(nid)
    arms: dict[str, dict[Any, Any]] = {c: {True: _arm(), False: _arm(), "units": {True: 0, False: 0}, "ys": {}}
                                       for c in v3_components(ctx)}
    for e in store.events(K_GATE):
        if e.get("v") != V3_VERSION or e.get("component") not in arms or e.get("request") in outage:
            continue
        used, r = bool(e.get("used")), arms[e["component"]]
        # strata = (the bypass rate in force, the brief kind the draw was made for): the arms are only comparable within one
        # session type (a literature brief and a debug brief do not hit at the same rate)
        ys = r["ys"].setdefault((round(float(e.get("rate") or 0.0), 4), str(e.get("decision") or "")), {True: [], False: []})
        r[used]["decisions"] += 1
        nids = list(by_req.get(e.get("request") or "", ()))
        hits = scored = 0
        for nid in nids:
            b = beliefs.get(nid)
            ok = b is not None and int(b[2]) >= V3_ARM_MIN_SEEDS
            scored += int(ok)
            hits += int(ok and b[0] > 0.0)
            r[used]["conf"] += int(nid in passed)
            r[used]["gpu"] += float(gpu.get(nid, 0.0))
        r[used]["scored"] += scored
        r[used]["hits"] += hits
        r["units"][used] += int(scored > 0)
        # the unit is the SESSION (its nodes share one context and one draw): its outcome is the share of its nodes that
        # hit, an UNSCORED node counting as a non-hit (intention to treat: a bypass that makes nodes die before they can
        # be scored is a harm, not missing data) and a session that produced nothing as 0
        ys[used].append(hits / max(len(nids), 1))
    rows = []
    for c in v3_components(ctx):
        w, wo = arms[c][True], arms[c][False]
        eff, se = _cluster_stratified(arms[c]["ys"])
        n_units = arms[c]["units"][False]               # ablated sessions with >= 1 scored node
        rows.append({"component": c, "rate": float(cc.rates.get(c, cc.ablation_rate)), "enabled": bool(cc.enabled.get(c, True)),
                     "with": w, "without": wo, "effect": _num(eff), "se": _num(se), "n": n_units,
                     "units": dict(arms[c]["units"]),
                     "verdict": verdict(eff, se, n_units, cc.min_ablated, cc.equivalence_margin)})
    return {"rows": rows, "min_ablated": cc.min_ablated, "applied": False}


def v3_text(sec: Mapping[str, Any]) -> list[str]:
    """The markdown lines of :func:`v3_section`."""
    out = [f"bypass draw per session (fork-and-flush branches exempt); report-only, nothing is switched off by it; "
           f"a verdict needs >= {sec.get('min_ablated', '?')} ablated sessions with a scored node; contrast = per-session "
           "share of nodes that hit (unscored = a miss), stratified by brief kind, se clustered by session",
           "NOTE the postmortem ablation hides the parent postmortems from the brief text AND the history tool's "
           "postmortem part; other sections of the brief that quote a closed node's verdict are not scrubbed, so a "
           "postmortem effect is a lower bound"]
    for r in sec.get("rows") or []:
        e = r.get("effect")
        out.append(f"{r['component']} (bypass {float(r.get('rate', 0.0)):.2f}): {r['verdict']}, "
                   + ("effect -" if e is None else f"hit-rate with - without {e:+.2f}+-{float(r.get('se') or 0.0):.2f}")
                   + f", n ablated {r['n']} | with: {_arm_txt(r['with'])} | without: {_arm_txt(r['without'])}")
    for a in sec.get("root_audit_arms") or []:         # loop.root_audit: each banked arm's sessions and nodes
        out.append(f"root-audit arm {a['arm']}: " + (f"served by {', '.join(a['requests'])}; nodes "
                                                      f"{', '.join(a['nodes']) or 'none yet'}" if a["requests"]
                                                      else "not served yet"))
    return out
