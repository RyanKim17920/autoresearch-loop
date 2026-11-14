"""loop.variants — MULTI-VARIANT IDEAS with a BRANCHED PLAN (operator 2026-10-06: "each ideator should produce many
variants so different hparams are tested immediately"; "a small decision tree ... launched automatically without a
new Opus session").

A research session's finish carries ``variants`` (config overrides of keys in its node's config) and ``plan``
(conditional follow-ups), validated by loop.agents.ideas. The driver registers them with the admitted node
(:func:`register`); :func:`tick` then runs, per node (the BASE):

1. once the base's screen runs exist, every initial variant becomes a config point on the base's commit, on the
   base's OWN screen block (``block_of=``): siblings sharing the commit and the base's runs as their control, so each
   point's paired delta is variant-minus-base on the same seeds (no extra seeds, no extra control);
2. when the points and the base's screen are scored, every variant's delta vs the base is SHRUNK (positive-part
   James-Stein, loop.legs.shrink: the spread of k noisy deltas is mostly noise) and the plan's branches are
   evaluated in order on the shrunk / predictive estimates -- ``base`` = the world model's shrunk own gain of the
   base, ``best`` / a variant's name = its shrunk delta, both in panel sigmas, ``diag:<q>`` = the diagnostic layer's
   value (skipped when unavailable); the first branch whose condition holds launches its variants (on top of
   ``base``, ``best`` or a named variant) on the same block. At most ``ideation.branch_depth`` rounds and
   ``ideation.max_points`` points per base; when no branch fires and the best variant (shrunk > 0) sits at the EDGE
   of the values tried for one of its knobs, one more variant goes a step beyond it (:func:`_edge`, hard bounds:
   loop.extrapolate);
3. finally the best variant graduates only if its shrunk delta is > 0 AND its raw delta beats the ``ideation.null_q``
   quantile of the max of k null deltas (selection-aware): the graduate (loop.tune._tuned_node, ``tune_of=``) takes
   screen / promote / confirm / audit on FRESH seeds, so the best-of-k selection never reaches a verdict.

When the diagnostic layer flags the optimiser (loop.agents.dossiers.optimizer_flagged), an optimiser co-tune variant
(the ``moonshot_tune.shared`` knobs at their step factors) is added to an idea whose variants do not touch them --
a general rule, never a named mechanism."""

from __future__ import annotations

import dataclasses
import math
import random
import statistics
import time
from typing import Any, Mapping, Sequence

from loop.contracts import CLOSED_STATUSES, PANEL, ROOT_NODE_ID, NodeStatus, Stage

K_VAR = "variants"          # {base node id: pass}
COTUNE = "optim_cotune"


def book(store: Any) -> dict[str, Any]:
    return dict(store.get_state(K_VAR) or {})


def bases(store: Any, min_points: int = 2) -> set[str]:
    """Nodes whose variant pass built at least ``min_points`` points (their dose was already searched)."""
    return {b for b, p in book(store).items() if len(p.get("points") or {}) >= min_points}


def register(ctx: Any, store: Any, node_id: str, plan: Mapping[str, Any], snapshot: Any = None) -> str:
    """Store the admitted node's variants + branched plan ('' when it has none)."""
    ic = getattr(ctx.cfg, "ideation", None)
    vs, br = list(plan.get("variants") or ()), list(plan.get("branches") or ())
    if ic is None or not ic.enabled or not (vs or br):
        return ""
    from loop.agents import dossiers as DS
    shared = dict(getattr(ctx.cfg.moonshot_tune, "shared", {}) or {})
    node = store.get_node(node_id)
    if shared and DS.optimizer_flagged(store, snapshot, node.parent_id if node is not None else None) and not any(set(v["overrides"]) & set(shared) for v in vs) \
            and len(vs) < ic.max_variants:
        vs.append({"name": COTUNE, "overrides": {}, "shared": True, "from": "base",
                   "rationale": "diagnostics flag the optimiser: co-tune the shared optimiser knobs"})
    bk = book(store)
    bk[node_id] = {"base": node_id, "status": "planned", "variants": vs, "branches": br, "fired": [], "round": 0,
                   "points": {}, "specs": {}, "created": time.time(), "why": "", "tuned": ""}
    store.set_state(K_VAR, bk)
    store.log("variants_registered", node_id=node_id, variants=[v["name"] for v in vs], branches=len(br))
    return f"variants for {node_id}: {[v['name'] for v in vs]} + {len(br)} branch(es)"


def _screen_runs(store: Any, nid: str) -> list[Any]:
    return [r for r in store.runs(node_id=nid) if r.stage is Stage.SCREEN]


def _resolve(ctx: Any, spec: Mapping[str, Any], cur: Mapping[str, Any], base_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """A variant's full override set: its ``from`` variant's overrides, then its own; the co-tune's shared knobs at
    their step factors."""
    src = spec.get("from") or "base"
    if src == "best":
        src = cur.get("best") or "base"
    out = dict(cur["specs"].get(src, {})) if src != "base" else {}
    out.update(spec.get("overrides") or {})
    if spec.get("shared"):
        for k, f in (getattr(ctx.cfg.moonshot_tune, "shared", {}) or {}).items():
            v = base_cfg.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v:
                out[k] = type(v)(v * f) if isinstance(v, int) else float(f"{v * f:.4g}")
    return {k: v for k, v in out.items() if k in base_cfg and v != base_cfg[k]}


def _build(ctx: Any, store: Any, adapter: Any, cur: dict, specs: Sequence[Mapping[str, Any]], new_id: Any) -> list[str]:
    from loop import design as D
    base = store.get_node(cur["base"])
    left = ctx.cfg.ideation.max_points - len(cur["points"])
    made, num = [], "".join(ch for ch in base.id if ch.isdigit()) or base.id
    if left <= 0 or not specs:
        return made
    with D._Tree(ctx, store, adapter, f"var-{base.id}-{cur['round']}", base_id=base.id) as t:
        for spec in list(specs)[:left]:
            name = spec["name"]
            while name in cur["points"]:
                name = f"{name}_r{cur['round']}"
            levels = _resolve(ctx, spec, cur, t.base)
            if not levels:
                store.log("variant_skipped", base=base.id, name=name, why="no effective override")
                continue
            i = len(cur["points"]) + 1
            nid = t.node(levels, knobs=[{"key": k, "low": t.base[k], "high": v, "levers": {}} for k, v in levels.items()],
                         tag=f"v{num}pt{i:02d}", operator=None, new_id=new_id,
                         notes=f"tune=v{num} variants_of={base.id} block_of={base.id} variant={name}")
            if nid and nid != base.id:
                cur["points"][name], cur["specs"][name] = nid, levels
                made.append(f"{name}={nid} {levels}")
            else:
                store.log("variant_skipped", base=base.id, name=name, why="rejected at intake" if not nid else "= base")
    return made


def _rows(ctx: Any, store: Any, cur: Mapping[str, Any], sigma: float) -> dict[str, list[float]]:
    """name -> [mean delta vs base, se, n] over the shared screen seeds (the base's screen runs are the control)."""
    from loop.tune import screen_panels
    ys = screen_panels(ctx, store, cur["base"])
    out = {}
    for name, nid in cur["points"].items():
        ds = [y - ys[s] for s, y in screen_panels(ctx, store, nid).items() if s in ys]
        if ds:
            out[name] = [statistics.fmean(ds), sigma / math.sqrt(len(ds)), float(len(ds))]
    return out


def _parent_rows(ctx: Any, store: Any, snapshot: Any, cur: Mapping[str, Any], rows: Mapping[str, list[float]],
                 sigma: float) -> dict[str, list[float]]:
    """name -> [mean delta vs the BASE'S PARENT, se, n]: paired on the parent's done runs at the variant's screen seeds
    (any stage: the same commit and seed), else the base's own increment (world-model posterior) + its delta vs the
    base. Missing when neither exists."""
    from loop import design as D
    from loop.seeds import REUSE_STAGES
    from loop.tune import screen_panels
    base = store.get_node(cur["base"])
    pid = (base.parent_id if base is not None else None) or ROOT_NODE_ID
    by_seed: dict[int, list[float]] = {}
    for r in store.runs(node_id=pid):
        if r.stage in REUSE_STAGES and (y := D.panel_of(ctx, r)) is not None:
            by_seed.setdefault(int(r.seed), []).append(y)
    par = {s: statistics.fmean(v) for s, v in by_seed.items()}
    post = (getattr(snapshot, "nodes", None) or {}).get(cur["base"])
    out = {}
    for name, nid in cur["points"].items():
        ds = [y - par[s] for s, y in screen_panels(ctx, store, nid).items() if s in par]
        if ds:
            out[name] = [statistics.fmean(ds), sigma / math.sqrt(len(ds)), float(len(ds))]
        elif post is not None and name in rows:
            out[name] = [post.increment.mean + rows[name][0], math.hypot(post.increment.sd, rows[name][1]), 0.0]
    return out


def _shrunk(rows: Mapping[str, list[float]]) -> dict[str, dict[str, float]]:
    from loop.legs import shrink
    eff = {k: {"estimable": True, "effect": r[0], "se": r[1]} for k, r in rows.items()}
    shrink(list(eff.values()))
    return eff


def null_max(ses: Sequence[float], q: float, seed: int = 0, sims: int = 4000) -> float:
    """The ``q`` quantile of the max of k independent N(0, se_j) deltas: what the best of k null variants reads."""
    if not ses:
        return 0.0
    rng = random.Random(seed)
    draws = sorted(max(rng.gauss(0.0, s) for s in ses) for _ in range(sims))
    return draws[min(sims - 1, int(q * sims))]


def _metric(store: Any, snapshot: Any, cur: Mapping[str, Any], eff: Mapping[str, Mapping[str, float]], metric: str,
            sigma: float) -> float | None:
    if metric == "base":
        p = (getattr(snapshot, "nodes", None) or {}).get(cur["base"])
        return None if p is None else p.increment.mean / sigma
    if metric == "best":
        return max(e["shrunk"] for e in eff.values()) / sigma if eff else None
    if metric.startswith("diag:"):
        try:
            from loop import diagnostics as DG
            rows = store.diagnostics(node_id=cur["base"])
            vs = [v for r in rows if (v := DG.quantity(r["features"], metric[5:])) is not None]
            return sum(vs) / len(vs) if vs else None
        except Exception:  # noqa: BLE001 -- optional module
            return None
    e = eff.get(metric)
    return None if e is None else e["shrunk"] / sigma


def _settled(store: Any, cur: Mapping[str, Any]) -> bool:
    """Every point closed and the base's screen scored (or the base closed)."""
    base = store.get_node(cur["base"])
    pts = [store.get_node(i) for i in cur["points"].values()]
    return all(p is not None and p.status in CLOSED_STATUSES for p in pts) and base is not None and (
        base.status in CLOSED_STATUSES or len(screen_panels_any(store, base.id)) >= len(_screen_runs(store, base.id)) > 0)


def screen_panels_any(store: Any, nid: str) -> list[Any]:
    return [r for r in _screen_runs(store, nid) if r.scores]


def tick(ctx: Any, store: Any, adapter: Any, snapshot: Any, *, new_id: Any, now: float | None = None,
         scarce: bool = False) -> list[str]:
    ic = getattr(ctx.cfg, "ideation", None)
    if ic is None or not ic.enabled or snapshot is None:
        return []
    now = time.time() if now is None else now
    bk, acts = book(store), []
    sigma = float(snapshot.sigma.get(PANEL, ctx.cfg.panel_sd))
    for bid, cur in list(bk.items()):
        if cur["status"] == "done":
            continue
        base = store.get_node(bid)
        if base is None or base.status is NodeStatus.REJECTED:
            cur.update(status="done", why="base rejected or missing")
            continue
        try:
            if cur["status"] == "planned":
                if scarce or len(_screen_runs(store, bid)) < len(ctx.cfg.seeds.screen):
                    continue                  # its screen runs exist first: the points pair on them
                made = _build(ctx, store, adapter, cur, cur["variants"], new_id)
                cur["status"] = "running" if cur["points"] else "done"
                cur["why"] = "" if cur["points"] else "no variant point admitted"
                store.log("variants_started", node_id=bid, points=dict(cur["points"]))
                acts.append(f"variants for {bid}: {'; '.join(made) or 'none admitted'}")
                continue
            if not _settled(store, cur):
                continue
            rows = _rows(ctx, store, cur, sigma)
            eff = _shrunk(rows)
            cur["rows"] = {k: [*r, eff[k]["shrunk"]] for k, r in rows.items()}
            if eff:
                cur["best"] = max(eff, key=lambda k: (eff[k]["shrunk"], k))
            fired = None
            if cur["round"] < ic.branch_depth and len(cur["points"]) < ic.max_points:
                for i, b in enumerate(cur["branches"]):
                    if i in cur["fired"]:
                        continue
                    from loop.agents.ideas import OPS
                    v = _metric(store, snapshot, cur, eff, b["when"]["metric"], sigma)
                    if v is not None and OPS[b["when"]["op"]](v, b["when"]["value"]):
                        fired = (i, b, v)
                        break
            if fired is not None:
                i, b, v = fired
                cur["fired"].append(i)
                cur["round"] += 1
                specs = [{"name": str(t.get("name") or f"b{i}_{k}"), "overrides": dict(t.get("overrides") or {}),
                          "from": str(t.get("from") or "base")} for k, t in enumerate(b["then"])]
                made = _build(ctx, store, adapter, cur, specs, new_id)
                store.log("variants_branch", node_id=bid, branch=i, metric=b["when"]["metric"], value=round(v, 4),
                          points=made)
                acts.append(f"variants for {bid}: branch {i} ({b['when']['metric']} {b['when']['op']} "
                            f"{b['when']['value']:g}, measured {v:+.2f}) -> {'; '.join(made) or 'nothing admitted'}")
                if made:
                    continue
            if fired is None and len(cur["points"]) < ic.max_points and (edge := _edge(ctx, store, cur, eff)):
                made = _build(ctx, store, adapter, cur, [edge], new_id)
                store.log("variants_edge", node_id=bid, best=cur["best"], overrides=edge["overrides"], points=made)
                acts.append(f"variants for {bid}: best {cur['best']} sits at the edge of what was tried -> one step "
                            f"beyond {edge['overrides']}: {'; '.join(made) or 'nothing admitted'}")
                if made:
                    continue
            acts.append(_finish(ctx, store, snapshot, cur, rows, eff, new_id, now))
        except Exception as exc:  # noqa: BLE001 -- a git / adapter failure closes the pass, never the loop
            cur.update(status="done", why=f"error: {exc!r}"[:300])
            store.log("variants_error", node_id=bid, error=repr(exc)[:500])
            acts.append(f"variants for {bid} closed on an error: {exc!r}"[:300])
    store.set_state(K_VAR, {**book(store), **bk})
    return acts


def _edge(ctx: Any, store: Any, cur: Mapping[str, Any], eff: Mapping[str, Mapping[str, float]]) -> dict | None:
    """RANGE EDGE: the best variant (shrunk > 0 over the base) one step further on its first numeric override that is
    the extreme of the values tried for it (the base's included), within the knob's hard bounds
    (loop.extrapolate.beyond); None otherwise."""
    from loop import extrapolate as EX
    best, cap = cur.get("best"), getattr(getattr(ctx.cfg, "extrapolate", None), "edge_max", 0)
    if not best or best not in cur["specs"] or eff.get(best, {}).get("shrunk", 0.0) <= 0.0 \
            or sum(n.endswith("_edge") for n in cur["points"]) >= cap:
        return None
    base = EX.View(store).cfg(cur["base"])
    for k, x in sorted(cur["specs"][best].items()):
        tried = [t for t in (base.get(k), *(s.get(k) for s in cur["specs"].values())) if t is not None]
        if (nx := EX.beyond(ctx, k, x, tried)) is not None:
            return {"name": f"{best}_edge", "overrides": {k: nx}, "from": best}
    return None


def _finish(ctx: Any, store: Any, snapshot: Any, cur: dict, rows: Mapping[str, list[float]],
            eff: Mapping[str, Mapping[str, float]], new_id: Any, now: float, dry: bool = False) -> str:
    """The best variant graduates (``dry``: judged only, ``tuned`` '(dry)', nothing logged) when it credibly beats
    the base AND the base's parent: shrunk > 0 with raw over the null max's ``ideation.null_q`` quantile, or shrunk
    alone >= ``legs.shrunk_bar`` (the confirm MDE: fresh confirmation, not a second selection gate, decides)."""
    from loop import legs as LG
    from loop import tune
    bid = cur["base"]
    cur["status"] = "done"
    if not eff:
        cur["why"] = "no variant scored"
        store.log("variants_done", node_id=bid, why=cur["why"])
        return f"variants for {bid}: no variant scored"
    best = cur["best"]
    nq = null_max([e["se"] for e in eff.values()], ctx.cfg.ideation.null_q, seed=len(cur["points"]))
    line = ", ".join(f"{k}: {r[0]:+.4f}±{r[1]:.4f} (shrunk {eff[k]['shrunk']:+.4f})" for k, r in sorted(rows.items()))
    # ... and credibly beats the base's PARENT under the same gate: "less bad than a bad base" never graduates (live
    # 2026-10-07: a node's flip_sigma_hi read +0.0181 vs its base, about -0.021 vs root, and graduated a node)
    prow = _parent_rows(ctx, store, snapshot, cur, rows, float(snapshot.sigma.get(PANEL, ctx.cfg.panel_sd))
                        if snapshot is not None else ctx.cfg.panel_sd)
    peff = _shrunk(prow) if prow else {}
    pq = null_max([e["se"] for e in peff.values()], ctx.cfg.ideation.null_q, seed=len(cur["points"]))
    bar = LG.shrunk_bar(ctx, store, snapshot)

    def credible(e: Mapping[str, float], raw: float, q: float) -> bool:
        return e["shrunk"] > 0.0 and (raw > q or e["shrunk"] >= bar)

    vs_parent = best in peff and credible(peff[best], prow[best][0], pq)
    vs_base = credible(eff[best], rows[best][0], nq)
    reason = "" if rows[best][0] > nq and (not vs_parent or prow[best][0] > pq) else \
        f" (graduate on shrunk {eff[best]['shrunk']:+.4f} >= bar {bar:.4f}, null gate not passed)"
    if dry:
        cur.update(tuned="(dry)" if vs_base and vs_parent else "", why=reason)
        return ""
    if vs_base and not vs_parent:
        cur["why"] = (f"{best} beats {bid} (shrunk {eff[best]['shrunk']:+.4f}) but not its parent: " + (
            f"raw {prow[best][0]:+.4f} (shrunk {peff[best]['shrunk']:+.4f}) vs null q{ctx.cfg.ideation.null_q:g} "
            f"{pq:+.4f}" if best in peff else "no parent evidence"))
    elif vs_base:
        base, pt = store.get_node(bid), store.get_node(cur["points"][best])
        tid = tune._tuned_node(ctx, store, snapshot, base, pt, {"batch": f"v-{bid}"}, eff[best]["shrunk"], new_id, now)
        store.update_node(tid, strategy=base.strategy)
        cur.update(tuned=tid, why=f"{best} shrunk {eff[best]['shrunk']:+.4f} > 0, raw {rows[best][0]:+.4f} vs null "
                                   f"q{ctx.cfg.ideation.null_q:g} {nq:+.4f}{reason}")
    else:
        cur["why"] = (f"no variant credibly beats {bid}: best {best} shrunk {eff[best]['shrunk']:+.4f}, raw "
                      f"{rows[best][0]:+.4f} vs null q{ctx.cfg.ideation.null_q:g} {nq:+.4f}")
    store.log("variants_done", node_id=bid, why=cur["why"], tuned=cur["tuned"], rows=cur.get("rows", {}))
    return f"variants for {bid}: {line} -> {cur['why']}" + (
        f"; graduate {cur['tuned']} -> screen / confirm on FRESH seeds" if cur["tuned"] else "")


def lines(store: Any) -> str:
    out = []
    for bid, c in book(store).items():
        out.append(f"variants {bid} {c['status']}: points {c.get('points')}" + (f"; {c['why']}" if c.get("why") else "")
                   + (f"; graduate {c['tuned']}" if c.get("tuned") else ""))
    return "\n".join(out)
