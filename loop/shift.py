"""loop.shift -- KNOB SHIFT -> PROPAGATE + BACKFILL (operator 2026-10-08: "if the layer decay thing goes up to 1, maybe
we should flash that across everything, and if there was anything backfill that might've been affected, we modify
that accordingly").

Only an INCUMBENT CHANGE triggers (a confirmed challenger still waiting on its audit does nothing until it is the
incumbent). The config keys whose value differs between the previous and the new incumbent are the KNOB SHIFT (any key;
optimiser / schedule keys are the main case): one ``knob_shift`` event (keys old -> new, relative size, root's value),
numbered as a backfill verdict epoch past loop.backfill.EPOCHS (:data:`BASE` + i). Then:

* PROPAGATE (:func:`propagate`): a node not started yet (QUEUED, or SCREENING without a run) built on the previous
  incumbent is RE-BASED in place onto the new one (the adapter's ``graft_port``: its change on the new commit; a
  crossover, paired against its parents' common ancestor, keeps it); one
  the new incumbent already holds is rejected ``already_in_incumbent``, a conflict stays on its carrier (logged). Batch
  points (design / leg / campaign / tuning points, leg swap tests) are left to their batch, eval-only nodes to their
  lane, the new incumbent itself to its own runs. A QUEUED
  campaign whose base is on the previous incumbent follows it (a closed base: ported onto the new one, the queue entry
  moves to the port) and its declared search ranges on a shifted key are re-centred on the new value
  (:func:`recentre`). New nodes already derive from the incumbent and the leg factorial's next round is on the new
  carrier (loop.legs.carriers); root-referenced lanes (root children, design batches vs root) are untouched.
* BACKFILL (loop.backfill.shift_items, through :func:`affected`): a closed null (screened out / not confirmed with
  P(own gain > 0) >= loop.diagnostics.NULL_P_MIN) closed before the shift is affected when its close explanation names
  an under_tuned knob in the shift, when one of its OWN diagnostics flags maps (``backfill.shift_flags``) to a shifted
  key, or when it is input-side (loop.campaign.input_side), the shift moves a ``campaign.input_space`` key and it ran
  at the previous incumbent's value of it. Each is ported onto the new incumbent (a campaign-eligible port gets its
  campaign if it closes null: loop.campaign's own lane)."""

from __future__ import annotations

import math
import re
from typing import Any, Mapping

from loop.contracts import ROOT_NODE_ID, Node, NodeStatus, design_point

K_LAST = "shift.incumbent"      # the incumbent the last shift was computed against
K_EVENTS = "shift.events"       # [{"n", "ts", "frm", "to", "keys": {key: [old, new]}, "rel": {key: r}, "root": {}}]
BASE = 100                      # shift epochs are BASE + 1, ...; loop.backfill.EPOCHS (append-only) stay below
_BATCH = re.compile(r"(?:^|\s)(?:legs_of|seed1|tune|tuning|campaign_of|swap_of)=")


def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def leaf(key: str) -> str:
    return key.rsplit(".", 1)[-1]


def config(ctx: Any, store: Any, adapter: Any, nid: str) -> dict[str, Any]:
    """``nid``'s flattened config: the adapter's ``config_at`` (a git show), else a worktree on its commit."""
    node = store.get_node(nid)
    if hasattr(adapter, "config_at"):
        return dict(adapter.config_at(node.commit))
    from loop import design as D
    with D._Tree(ctx, store, adapter, f"shift-{nid}", base_id=nid) as t:
        return dict(t.base)


def diff(old: Mapping[str, Any], new: Mapping[str, Any]) -> tuple[dict[str, list], dict[str, float | None]]:
    """({key: [old, new]} for every key whose value differs, {key: |new - old| / |old|} (None: not numeric / old 0))."""
    keys = {k: [old.get(k), new.get(k)] for k in sorted(set(old) | set(new)) if old.get(k) != new.get(k)}
    rel = {k: round(abs(b - a) / abs(a), 4) if _num(a) and _num(b) and a else None for k, (a, b) in keys.items()}
    return keys, rel


def events(store: Any) -> list[dict[str, Any]]:
    return list(store.get_state(K_EVENTS) or ())


def current(store: Any, snapshot: Any) -> dict[str, Any] | None:
    """The latest shift if it is onto the current incumbent (an older one was superseded), else None."""
    evs, inc = events(store), getattr(snapshot, "incumbent_id", None) or ROOT_NODE_ID
    return evs[-1] if evs and evs[-1]["to"] == inc and inc != ROOT_NODE_ID else None


def label(ev: Mapping[str, Any], top: int = 4) -> str:
    """``layerwise_decay 0.8->1.0, patch_embed_lr_mult 0.5->1.0`` (+ k more)."""
    ks = list(ev["keys"])
    out = ", ".join(f"{leaf(k)} {a}->{b}" for k, (a, b) in list(ev["keys"].items())[:top])
    return out + (f" +{len(ks) - top} more" if len(ks) > top else "")


def detect(ctx: Any, store: Any, adapter: Any, snapshot: Any, now: float) -> dict[str, Any] | None:
    """The new knob-shift event when the incumbent changed since the last call (booked), else None. The first call
    only records the incumbent (nothing to compare it with)."""
    inc, prev = getattr(snapshot, "incumbent_id", None) or ROOT_NODE_ID, store.get_state(K_LAST)
    if prev == inc:
        return None
    if prev is None or store.get_node(prev) is None or store.get_node(inc) is None:
        store.set_state(K_LAST, inc)
        return None
    old, new = config(ctx, store, adapter, prev), config(ctx, store, adapter, inc)
    root = config(ctx, store, adapter, ROOT_NODE_ID) if store.get_node(ROOT_NODE_ID) is not None else {}
    keys, rel = diff(old, new)
    store.set_state(K_LAST, inc)
    if not keys:
        store.log("knob_shift", node_id=inc, previous=prev, keys={}, rel={})
        return None
    ev = {"n": BASE + len(events(store)) + 1, "ts": now, "frm": prev, "to": inc, "keys": keys, "rel": rel,
          "root": {k: root.get(k) for k in keys}}
    store.set_state(K_EVENTS, events(store) + [ev])
    store.log("knob_shift", node_id=inc, previous=prev, epoch=ev["n"], keys=keys, rel=rel)
    return ev


def recentre(row: Mapping[str, Any], old: Any, new: Any) -> dict[str, Any]:
    """A declared range row {key, low, high, scale} moved by old -> new in its scale (log: x new / old), kept inside
    the hull of the declared range and the new value (a declared range is never left, as in loop.campaign: a bound
    such as layer decay <= 1.0 is never crossed; a value past it widens it to that value only); the row
    unchanged when it has no numeric range or the move would empty it."""
    lo, hi = row.get("low"), row.get("high")
    if not all(map(_num, (lo, hi, old, new))) or old == new:
        return dict(row)
    log = row.get("scale", "log") == "log" and min(lo, hi, old, new) > 0
    a, b = (lo * new / old, hi * new / old) if log else (lo + new - old, hi + new - old)
    a, b = max(a, min(lo, new)), min(b, max(hi, new))
    fmt = (lambda x: int(round(x))) if isinstance(lo, int) and isinstance(hi, int) else (lambda x: float(f"{x:.4g}"))
    return {**row, "low": fmt(a), "high": fmt(b)} if a < b and math.isfinite(a + b) else dict(row)


def _rebase(ctx: Any, store: Any, adapter: Any, node: Node, ev: Mapping[str, Any], now: float) -> str:
    """``node`` (not started, on the previous incumbent) re-based in place onto the new one; the action line."""
    po, pn = store.get_node(ev["frm"]), store.get_node(ev["to"])
    sha, how = adapter.graft_port(po.commit, node.commit, pn.commit, f"rb-{node.id}")
    if sha:
        store.update_node(node.id, commit=sha, parent_id=pn.id, updated_ts=now,
                          notes=f"{node.notes or ''} rebased_from={po.id}".strip())
        store.log("node_rebased", node_id=node.id, frm=po.id, to=pn.id, epoch=ev["n"])
        return f"shift: {node.id} ({node.status.value}) re-based {po.id} -> {pn.id}"
    if how == "contained":
        store.update_node(node.id, status=NodeStatus.REJECTED, reject_reason="already_in_incumbent", updated_ts=now)
        store.log("node_transition", node_id=node.id, frm=node.status.value, to=NodeStatus.REJECTED.value,
                  reason=f"knob shift: the new incumbent {pn.id} already holds its change")
        return f"shift: {node.id} rejected: the new incumbent {pn.id} already holds its change"
    store.log("rebase_conflict", node_id=node.id, frm=po.id, to=pn.id, outcome=how)
    return f"shift: {node.id} stays on {po.id} ({how})"


def propagate(ctx: Any, store: Any, adapter: Any, ev: Mapping[str, Any], *, new_id: Any, now: float) -> list[str]:
    """Re-base the not-started nodes and the queued campaigns on the previous incumbent onto the new one; re-centre
    the queued campaigns' declared ranges on the shifted keys. The action lines."""
    from loop import backfill as BF
    from loop import campaign as CP
    old, inc, acts = ev["frm"], ev["to"], []
    for node in sorted(store.nodes(status=[NodeStatus.QUEUED, NodeStatus.SCREENING]), key=lambda n: n.id):
        if node.parent_id == old and node.id != inc and not node.second_parent_id and not design_point(node) \
                and not _BATCH.search(node.notes or "") and not store.runs(node_id=node.id):
            acts.append(_rebase(ctx, store, adapter, node, ev, now))
    bk = CP.book(store)
    queue = []
    for q in bk["queue"]:
        n = store.get_node(q["base"])
        if n is not None and n.parent_id == old and n.status in CP.CLOSES:
            made, how = BF.port(ctx, store, adapter, n, inc, new_id(), now)
            if made:
                store.set_state(f"ideation:{made}", dict(store.get_state(f"ideation:{n.id}") or {}))
                acts.append(f"shift: queued campaign on {n.id} moves to its port {made} on {inc}")
                q, n = {**q, "base": made}, store.get_node(made)
            else:
                acts.append(f"shift: queued campaign on {n.id} stays on {old} ({how})")
        if n is not None and n.parent_id == inc and (ide := store.get_state(f"ideation:{n.id}") or {}).get("search_space"):
            rows = [recentre(r, *ev["keys"][r["key"]]) if r.get("key") in ev["keys"] else r for r in ide["search_space"]]
            if rows != ide["search_space"]:
                store.set_state(f"ideation:{n.id}", {**ide, "search_space": rows})
                acts.append(f"shift: campaign on {n.id} re-centred " + ", ".join(
                    f"{leaf(r['key'])} {r.get('low')}..{r.get('high')}" for r in rows if r.get("key") in ev["keys"]))
        queue.append(q)
    if queue != bk["queue"]:
        store.set_state(CP.K_CAMP, {**CP.book(store), "queue": queue})
    return acts


def tick(ctx: Any, store: Any, adapter: Any, snapshot: Any, *, new_id: Any, now: float) -> list[str]:
    """Detect a knob shift and propagate it (once per incumbent change); the action lines."""
    if snapshot is None:
        return []
    try:
        ev = detect(ctx, store, adapter, snapshot, now)
    except Exception as exc:  # noqa: BLE001 -- an unreadable config: retried next tick, never the loop
        store.log("knob_shift_error", error=repr(exc)[:500])
        return []
    if ev is None:
        return []
    acts = [f"knob shift {ev['frm']} -> {ev['to']}: {label(ev)} (backfill epoch {ev['n']})"]
    try:
        acts += propagate(ctx, store, adapter, ev, new_id=new_id, now=now)
    except Exception as exc:  # noqa: BLE001
        store.log("knob_shift_error", node_id=ev["to"], error=repr(exc)[:500])
        acts.append(f"knob shift: propagation failed: {exc!r}"[:300])
    return acts


# ------------------------------------------------------------------------------------- the affected set
def value_at(store: Any, node: Node, key: str, root: Mapping[str, Any]) -> Any:
    """``key``'s value in ``node``'s config: the nearest config diff on its lineage that sets it, else root's."""
    nid, seen = node.id, set()
    while nid and nid != ROOT_NODE_ID and nid not in seen and (n := store.get_node(nid)) is not None:
        seen.add(nid)
        if key in (d := store.get_state(f"config_diff:{nid}") or {}):
            return d[key]
        nid = n.parent_id
    return root.get(key)


def affected(ctx: Any, store: Any, snapshot: Any, ev: Mapping[str, Any], node: Node) -> str:
    """Why the closed ``node`` may have closed only because of a knob the shift moved ('' = it was not)."""
    from loop import diagnostics as DG
    from loop.campaign import input_side
    keys = ev["keys"]

    def hit(names: Any) -> list[str]:
        names = set(names)
        return [k for k in keys if k in names or leaf(k) in names]
    p = (getattr(snapshot, "nodes", None) or {}).get(node.id)
    if node.status not in (NodeStatus.SCREENED_OUT, NodeStatus.NOT_CONFIRMED) \
            or (p is not None and p.gain_p is not None and p.p_gain < DG.NULL_P_MIN):
        return ""
    if (m := re.search(r"under_tuned \(knob ([^)\s]+)\)", DG.under_tuned(node, store, snapshot))) and hit([m.group(1)]):
        return f"close: under_tuned (knob {m.group(1)})"
    fmap = getattr(ctx.cfg.backfill, "shift_flags", {}) or {}
    if fmap and store.diagnostics(node_id=node.id):
        for d in DG.bottlenecks(store, node.id):
            if ks := hit(fmap.get(d["flag"], ())):
                return f"{d['flag']} on its runs (relieved by {', '.join(map(leaf, ks))})"
    ks = hit(r["key"] for r in getattr(ctx.cfg.campaign, "input_space", ()))
    at = [k for k in ks if value_at(store, node, k, ev.get("root") or {}) == keys[k][0]]
    if at and (why := input_side(ctx, store, node)):
        return f"input-side ({why}) closed null at {', '.join(f'{leaf(k)} {keys[k][0]}' for k in at)}"
    return ""
