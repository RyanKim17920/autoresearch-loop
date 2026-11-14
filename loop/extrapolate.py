"""loop.extrapolate — FOLLOW THE GRADIENT (operator 2026-10-08: "not sure how we didn't realize that sometimes if
increasing hyperparam x helps, more x helps"). The biggest real win (a node) moved layer decay 0.7 -> 0.8, positive in
every factorial, yet nothing ever tried 0.9 / 1.0 on the incumbent: the dose ladders key on a node's OWN new knob and
the campaigns stop at their range edge.

THE WINNING-DIRECTION LEDGER (:func:`ledger`): for every numeric config key (a list element as ``key[i]``) the
evidence of the direction of improvement, each item a move a -> b of the key in its scale (:func:`meta`: the
``extrapolate.knobs`` rows, else for a value in [0, 1] linear -- bounded by 1 only for a probability / fraction / rate
leaf, otherwise capped at max(1, 4 x the largest value seen) -- else log) with a shrunk gain and its sd:

* LINEAGE -- each step of the incumbent's chain (root .. incumbent) moved some keys; the step's own increment
  posterior (EB-shrunk) is that move's gain (a multi-key step credits every key it moved);
* FACTORIALS -- loop.design batches and loop.legs designs: each knob's main effect low -> high, positive-part
  James-Stein shrunk over its design (loop.legs.shrink);
* SINGLE MOVES -- any config-only node (a campaign / variant / ladder point, an exploit nudge, an extrapolation node)
  whose effective config differs from its reference (its campaign's base, else its parent) on exactly ONE key, same
  code: its own increment posterior (vs the base's for a campaign point).

Per key: direction = the sign of the sum of each item's ``sign(move) x z`` (z = gain / sd clipped at
``max_item_z``), the last winning step (the incumbent chain's latest move first, else the latest positive item), the
evidence strength (that sum) and what was tested beyond the incumbent's value in that direction ON the incumbent
(same code, effective config equal but for the key).

THE EXTRAPOLATION LANE (:func:`tick`): a key with net evidence >= ``min_z`` whose direction is not stopped gets
config-only nodes on the incumbent continuing it -- ``mults`` (x1, x2) of the last winning step past the furthest
value tested there (one step after one loss), clipped to its bounds -- screened like any node (paired vs the
incumbent's runs, promote / confirm / audit on fresh seeds; protected keys and caps through intake). A direction
stops when its two furthest steps on the incumbent read shrunk <= 0 or one is vetoed (harm). A known SUBSTITUTE
(a factorial interaction of the two keys at z <= -``partner_z``, e.g. LR x layer decay) also gets the combined move with
the partner backed off one of its own steps. At most ``max_inflight`` open at once, strongest evidence first, one
round per key at a time; a value already tested on the incumbent (any lane) is never rebuilt. Each node logs
``extrapolate: <key> <old>-><new> (direction from <evidence>)``.

RANGE EDGES (:func:`beyond`, :func:`inside`): a campaign's best point at the edge of its range, a variant pass's best
variant at the extreme of its tried values and a dose ladder's winning top rung are extended one step further by
their lanes, within these hard bounds. :func:`facts` is the ledger as computed facts for the researcher context."""

from __future__ import annotations

import fnmatch
import math
import re
import time
from typing import Any, Mapping, Sequence

from loop.contracts import ACTIVE_STATUSES, CLOSED_STATUSES, PANEL, ROOT_NODE_ID, NodeStatus

K_EXT = "extrapolate"      # {"nodes": {nid: {"incumbent", "keys", "levels", "why"}}, "stopped": {"<key>@<inc>": why}}
_TOUCH: dict[str, bool] = {}   # diff path -> it touches code (the diffs are immutable)
#: an unknown key in [0, 1] whose leaf names one of these is a probability / fraction: bounded to [0, 1]
UNIT_LEAF = re.compile(r"prob|frac|ratio|rate|momentum|beta|thresh|crop_scale|(^|_)p$")
UNBOUNDED_X = 4.0      # any other unknown key in [0, 1]: capped at max(1, this x the largest value seen)
BOUND_STOP = "at its bound"    # a stop on the knob's metadata, not on evidence: re-planned every pass


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _cfg(ctx: Any) -> Any:
    return getattr(ctx.cfg, "extrapolate", None)


def book(store: Any) -> dict[str, Any]:
    return {"nodes": {}, "stopped": {}, **(store.get_state(K_EXT) or {})}


# ------------------------------------------------------------------------------------------- knob metadata
def meta(ctx: Any, key: str, v: Any, seen: Sequence[Any] = ()) -> dict[str, Any] | None:
    """{scale, low, high} of ``key`` at value ``v``: the first ``extrapolate.knobs`` row whose pattern matches (its
    bounds -- authoritative; a linear row without bounds is unbounded), else for a value in [0, 1]: linear in [0, 1]
    when the key's leaf names a probability / fraction / rate (:data:`UNIT_LEAF`), otherwise linear from 0 with a
    generous cap, max(1, ``UNBOUNDED_X`` x the largest of ``v`` and ``seen``) (live 10-08: the old [0, 1] default
    stopped ``dino.jepa_xscale_weight``, a loss weight, "at its bound (1)" while it was still winning); else log
    (> 0; an int >= 1). None for a non-numeric value or a log knob at <= 0."""
    if not _num(v):
        return None
    ex, row = _cfg(ctx), None
    for r in getattr(ex, "knobs", ()) or ():
        if fnmatch.fnmatchcase(key, str(r["key"])):
            row = r
            break
    if row is not None:
        m = {"scale": str(row.get("scale") or "log"), "low": row.get("low"), "high": row.get("high")}
    elif isinstance(v, float) and 0.0 <= v <= 1.0 and UNIT_LEAF.search(key.rsplit(".", 1)[-1].split("[", 1)[0]):
        m = {"scale": "linear", "low": 0.0, "high": 1.0}
    elif isinstance(v, float) and 0.0 <= v <= 1.0:
        top = max([v, *(abs(float(x)) for x in seen if _num(x))])
        m = {"scale": "linear", "low": 0.0, "high": max(1.0, UNBOUNDED_X * top)}
    else:
        m = {"scale": "log", "low": 1 if isinstance(v, int) else None, "high": None}
    return None if m["scale"] == "log" and v <= 0 else m


def to_u(m: Mapping[str, Any], x: float) -> float:
    return math.log(x) if m["scale"] == "log" else float(x)


def from_u(m: Mapping[str, Any], u: float, like: Any) -> Any:
    x = math.exp(u) if m["scale"] == "log" else u
    lo, hi = m.get("low"), m.get("high")
    x = min(x, hi) if hi is not None else x
    x = max(x, lo) if lo is not None else x
    return int(round(x)) if isinstance(like, int) else float(f"{x:.4g}")


def inside(ctx: Any, key: str, x: Any, like: Any = None, seen: Sequence[Any] = ()) -> bool:
    """``x`` lies within ``key``'s hard bounds (:func:`meta` at ``like``, default ``x``, with ``seen``)."""
    m = meta(ctx, key, x if like is None else like, seen)
    return m is not None and _num(x) and (m["scale"] != "log" or x > 0) \
        and (m["low"] is None or x >= m["low"]) and (m["high"] is None or x <= m["high"])


def beyond(ctx: Any, key: str, x: Any, tried: Sequence[Any]) -> Any:
    """One step past ``x`` when ``x`` is the extreme of ``tried`` (the step: to its nearest tried neighbour, in the
    key's scale), clipped to the hard bounds; None when ``x`` is not at an edge or the step is clipped away."""
    m = meta(ctx, key, x, tried)
    if m is None or (m["scale"] == "log" and any(not _num(t) or t <= 0 for t in tried)):
        return None
    u, ts = to_u(m, x), sorted({to_u(m, t) for t in tried if _num(t)} - {to_u(m, x)})
    if not ts or (min(ts) < u < max(ts)):
        return None
    nb = ts[-1] if u > ts[-1] else ts[0]           # the nearest tried value on the inner side
    out = from_u(m, 2 * u - nb, x)
    return None if out == x or out in tried or not inside(ctx, key, out, x, tried) else out


# ------------------------------------------------------------------------------------------- configs
class View:
    """Effective configs (root config + chain + diff, numeric leaves; a list's numbers as ``key[i]``) and code bases
    of the store's nodes, memoised for one pass. A campaign point is its base's config + its own diff."""

    def __init__(self, store: Any) -> None:
        from loop.select.intake import K_BASE
        self.store, self.base = store, store.get_state(K_BASE) or {}
        self._cfg: dict[str, dict[str, Any]] = {}
        self._code: dict[str, str] = {}
        self.nodes = {n.id: n for n in store.nodes()}

    @staticmethod
    def numeric(cfg: Mapping[str, Any]) -> dict[str, Any]:
        out = {}
        for k, v in cfg.items():
            if _num(v):
                out[k] = v
            elif isinstance(v, list):
                out.update({f"{k}[{i}]": x for i, x in enumerate(v) if _num(x)})
        return out

    def ref(self, nid: str) -> str:
        """The node a config point was written on: its campaign's base (``campaign_of=``), else its parent."""
        from loop.campaign import _note
        n = self.nodes.get(nid)
        if n is None:
            return ""
        return _note(n, "campaign_of") if " tune=" in f" {n.notes or ''}" and _note(n, "campaign_of") \
            else (n.parent_id or "")

    def cfg(self, nid: str) -> dict[str, Any]:
        if nid not in self._cfg:
            from loop.adapters import flatten
            n, self._cfg[nid] = self.nodes.get(nid), {}
            own = flatten(self.store.get_state(f"config_diff:{nid}") or {})
            up = self.ref(nid) if n is not None and nid != ROOT_NODE_ID else ""
            self._cfg[nid] = {**(self.cfg(up) if up else self.numeric(self.base)), **self.numeric(own)}
        return self._cfg[nid]

    def touches(self, nid: str) -> bool:
        """The node's own diff changes code (loop.select._touches_code; a diff file never changes: read once per
        process, :data:`_TOUCH`)."""
        n = self.nodes.get(nid)
        if n is None:
            return False
        if (key := n.diff_path or f"-{nid}") not in _TOUCH:
            from loop.select import _touches_code
            _TOUCH[key] = _touches_code(n)
        return _TOUCH[key]

    def code(self, nid: str) -> str:
        if nid not in self._code:
            n = self.nodes.get(nid)
            self._code[nid] = ROOT_NODE_ID
            if n is not None and n.id != ROOT_NODE_ID:
                up = self.ref(nid)
                self._code[nid] = n.id if self.touches(nid) or not up else self.code(up)
        return self._code[nid]

    def moved(self, a: str, b: str) -> dict[str, tuple[Any, Any]]:
        ca, cb = self.cfg(a), self.cfg(b)
        return {k: (ca.get(k), cb.get(k)) for k in set(ca) | set(cb) if ca.get(k) != cb.get(k)}


def _post(snapshot: Any, nid: str) -> Any:
    return (getattr(snapshot, "nodes", None) or {}).get(nid)


def _gain(snapshot: Any, nid: str, vs: str = "") -> tuple[float, float] | None:
    """(mean, sd) of ``nid``'s own increment posterior (minus ``vs``'s increment: a campaign point vs its base)."""
    p = _post(snapshot, nid)
    if p is None or not p.utility.n:
        return None
    if not vs:
        return p.increment.mean, p.increment.sd
    q = _post(snapshot, vs)
    return None if q is None else (p.increment.mean - q.increment.mean, math.hypot(p.increment.sd, q.increment.sd))


# ------------------------------------------------------------------------------------------- evidence
def _item(ctx: Any, key: str, a: Any, b: Any, gain: float, sd: float, src: str, order: float) -> dict | None:
    m = meta(ctx, key, a if _num(a) else b)
    if m is None or not (_num(a) and _num(b)) or a == b or (m["scale"] == "log" and min(a, b) <= 0):
        return None
    zmax = getattr(_cfg(ctx), "max_item_z", 3.0)
    z = max(-zmax, min(zmax, gain / sd)) if sd > 0 else 0.0
    return {"key": key, "a": a, "b": b, "u": to_u(m, b) - to_u(m, a), "gain": gain, "sd": sd, "z": z, "src": src,
            "order": order}


def _factorials(store: Any) -> list[tuple[str, list[dict], list[dict], list[dict], set[str]]]:
    """(label, factors, shrunk effects, 2FI chains, point ids) of every analysed design / leg design."""
    from loop import design as D
    from loop import legs as LG
    out = []
    for b in (store.get_state(D.K_DESIGN) or {}).get("batches") or ():
        if b.get("effects"):
            out.append((f"design d{b['batch']}", b["knobs"], b["effects"], b.get("chains") or [],
                        set((b.get("points") or {}).values())))
    for b in LG.book(store).get("batches") or ():
        for name, res in (b.get("result") or {}).items():
            d = (b.get("designs") or {}).get(name) or {}
            if res.get("effects") and d.get("factors"):
                out.append((f"legs L{b['batch']}{name[0]} on {b.get('carrier', '?')}", d["factors"], res["effects"],
                            [], set((d.get("points") or {}).values())))
    for i, (_, _, eff, _, _) in enumerate(out):
        eff = [{**e, "estimable": e.get("estimable", e.get("effect") is not None)} for e in eff]
        LG.shrink(eff)
        out[i] = (out[i][0], out[i][1], eff, out[i][3], out[i][4])
    return out


def evidence(ctx: Any, store: Any, snapshot: Any, view: View | None = None) -> list[dict[str, Any]]:
    """Every evidence item (module doc): lineage steps, factorial main effects, single config moves."""
    view, items = view or View(store), []
    inc = getattr(snapshot, "incumbent_id", "") or ROOT_NODE_ID
    chain, n = [], view.nodes.get(inc)
    while n is not None and n.id != ROOT_NODE_ID and n.id not in chain:
        chain.append(n.id)
        n = view.nodes.get(n.parent_id or "")
    for depth, nid in enumerate(chain):                              # the incumbent lineage, latest first
        node, g = view.nodes[nid], _gain(snapshot, nid)
        if g is None:
            continue
        tag = f"lineage {nid}" + (" confirmed" if node.status is NodeStatus.CONFIRMED else "")
        for k, (a, b) in view.moved(node.parent_id or ROOT_NODE_ID, nid).items():
            if it := _item(ctx, k, a, b, g[0], g[1], tag, 1e12 - depth):
                items.append({**it, "lineage": True})
    points: set[str] = set()
    for i, (label, fac, eff, _, pts) in enumerate(_factorials(store)):
        points |= pts
        for f, e in zip(fac, eff):
            if not e.get("estimable", e.get("effect") is not None) or "shrunk" not in e:
                continue
            lo, hi = f["low"], f["high"]
            moves = {f["key"]: (lo, hi)} if _num(lo) else {
                f"{f['key']}[{j}]": (x, y) for j, (x, y) in enumerate(zip(lo, hi)) if x != y} \
                if isinstance(lo, list) and isinstance(hi, list) and len(lo) == len(hi) else {}
            if len(moves) == 1:
                ((k, (a, b)),) = moves.items()
                if it := _item(ctx, k, a, b, e["shrunk"], e.get("shrunk_sd") or e["se"], label, 1e6 + i):
                    items.append(it)
    for nid, node in view.nodes.items():                             # single config moves
        ref = view.ref(nid)
        if nid == ROOT_NODE_ID or nid in points or not ref or view.touches(nid) or nid in chain \
                or node.status is NodeStatus.REJECTED:
            continue
        mv = view.moved(ref, nid)
        if len(mv) != 1 or (g := _gain(snapshot, nid, ref if ref != node.parent_id else "")) is None:
            continue
        ((k, (a, b)),) = mv.items()
        if it := _item(ctx, k, a, b, g[0], g[1], f"{nid} on {ref}", node.created_ts):
            items.append(it)
    return items


def partners(store: Any, z: float) -> dict[str, list[tuple[str, float]]]:
    """key -> [(partner, interaction z)] of factorial 2FIs at z <= -``z`` (substitutes: both moves help, together
    less than the sum); an aliased chain (several pairs) never counts."""
    out: dict[str, list[tuple[str, float]]] = {}
    for _, _, _, chains, _ in _factorials(store):
        for c in chains:
            if len(c.get("pairs") or ()) == 1 and c.get("se") and c["c"] / c["se"] <= -z:
                a, b = c["pairs"][0]
                out.setdefault(a, []).append((b, c["c"] / c["se"]))
                out.setdefault(b, []).append((a, c["c"] / c["se"]))
    return out


def tested(view: View, snapshot: Any, inc: str) -> dict[frozenset, list[dict[str, Any]]]:
    """Moved key set -> [{levels, nid, gain | None (open), vetoed}] of every node with the incumbent's code whose
    effective config differs from the incumbent's on 1 or 2 keys (any lane): what was tested ON the incumbent."""
    out: dict[frozenset, list[dict[str, Any]]] = {}
    code = view.code(inc)
    for nid, n in view.nodes.items():
        if nid == inc or n.status in (NodeStatus.REJECTED, NodeStatus.VOID, NodeStatus.CRASHED) \
                or view.code(nid) != code or not 1 <= len(mv := view.moved(inc, nid)) <= 2:
            continue
        p, g = _post(snapshot, nid), None
        if n.status not in ACTIVE_STATUSES and p is not None and p.utility.n:
            q = _post(snapshot, inc)
            g = p.increment.mean if n.parent_id == inc else p.utility.mean - (q.utility.mean if q else 0.0)
        out.setdefault(frozenset(mv), []).append({"levels": {k: b for k, (_, b) in mv.items()}, "nid": nid,
                                                  "gain": g, "open": n.status in ACTIVE_STATUSES or (
                                                      g is None and n.status not in CLOSED_STATUSES),
                                                  "vetoed": bool(p is not None and p.vetoed)})
    return out


def ledger(ctx: Any, store: Any, snapshot: Any, view: View | None = None) -> dict[str, dict[str, Any]]:
    """key -> {dir, z, n, pos, step (scale units), last, value (the incumbent's), scale, beyond [(value, gain | None,
    nid)] on the incumbent in the direction, sorted by distance} for every key with evidence, strongest first."""
    view = view or View(store)
    inc = getattr(snapshot, "incumbent_id", "") or ROOT_NODE_ID
    icfg, by = view.cfg(inc), {}
    for it in evidence(ctx, store, snapshot, view):
        by.setdefault(it["key"], []).append(it)
    done = tested(view, snapshot, inc)
    out = {}
    for k, its in by.items():
        v, net = icfg.get(k), sum((1 if it["u"] > 0 else -1) * it["z"] for it in its)
        m = meta(ctx, k, v)
        if m is None or not net:
            continue
        d = 1 if net > 0 else -1
        win = [it for it in its if (it["u"] > 0) == (d > 0) and it["gain"] > 0]
        last = max(win, key=lambda it: it["order"], default=None)
        rows = list(((to_u(m, r["levels"][k]) - to_u(m, v)) * d, r["levels"][k], r["gain"], r["nid"], r["vetoed"])
                      for r in done.get(frozenset([k]), ())
                      if _num(r["levels"][k]) and (m["scale"] != "log" or r["levels"][k] > 0))
        # by distance, then node id: two nodes at one value (one open: gain None) must not compare their gains
        # (live 10-08: TypeError on a node's re-plan)
        rows.sort(key=lambda t: (t[0], t[3]))
        out[k] = {"key": k, "dir": d, "z": abs(net), "n": len(its), "pos": len(win), "value": v, "scale": m["scale"],
                  "step": abs(last["u"]) if last else 0.0, "last": last,
                  "beyond": [(x, g, nid, vt) for dist, x, g, nid, vt in rows if dist > 0]}
    return dict(sorted(out.items(), key=lambda kv: -abs(kv[1]["z"])))


# ------------------------------------------------------------------------------------------- the lane
def _allowed(ctx: Any, key: str) -> bool:
    from loop.select.intake import family_of
    root = key.split("[", 1)[0]
    fam = family_of(ctx, root)
    return bool(fam) and not ctx.cfg.families[fam].init_check and not any(
        root == pk or root.startswith(pk + ".") for pk in ctx.spec.surface.protected_keys)


def _fmt(x: Any) -> str:
    return f"{x:.4g}" if isinstance(x, float) else str(x)


def _why(e: Mapping[str, Any]) -> str:
    it = e["last"]
    return (f"{it['src']}: {_fmt(it['a'])}->{_fmt(it['b'])} {it['gain']:+.4f}; net {e['z']:.1f} sigma over "
            f"{e['n']} item(s), {e['pos']} winning")


def plan(ctx: Any, store: Any, snapshot: Any, view: View | None = None,
         led: Mapping[str, Mapping[str, Any]] | None = None) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """(moves, stopped): the next extrapolation nodes on the incumbent, strongest key first ({key, levels, why,
    combo}), and key -> why for the directions that stop (two furthest steps shrunk <= 0, or a vetoed one)."""
    ex, view = _cfg(ctx), view or View(store)
    inc = getattr(snapshot, "incumbent_id", "") or ROOT_NODE_ID
    led = ledger(ctx, store, snapshot, view) if led is None else led
    stop = {k.rsplit("@", 1)[0] for k, why in book(store)["stopped"].items()
            if k.endswith(f"@{inc}") and not str(why).startswith(BOUND_STOP)}     # a bound stop is re-planned
    done, pairs, icfg = tested(view, snapshot, inc), partners(store, ex.partner_z), view.cfg(inc)
    moves, stopped = [], {}
    for k, e in led.items():
        if k in stop or e["z"] < ex.min_z or not e["step"] or not _allowed(ctx, k):
            continue
        closed = [b for b in e["beyond"] if b[1] is not None]           # an open step: its round is in flight
        if vt := [x for x, _, _, v in closed if v]:
            stopped[k] = f"harm veto at {_fmt(vt[0])}"
            continue
        trail = 0
        for _, g, _, _ in reversed(closed):
            if g > 0:
                break
            trail += 1
        if trail >= 2:
            stopped[k] = "two furthest steps shrunk <= 0: " + ", ".join(
                f"{_fmt(x)} {g:+.4f} ({nid})" for x, g, nid, _ in closed[-2:])
            continue
        m = meta(ctx, k, e["value"], [x for x, *_ in e["beyond"]])
        f =to_u(m, closed[-1][0]) if closed else to_u(m, e["value"])
        have = {x for x, *_ in e["beyond"]} | {e["value"]}
        steps = [from_u(m, f + e["dir"] * e["step"] * mult, e["value"]) for mult in ex.mults[:1 if trail else None]]
        targets = [x for x in dict.fromkeys(steps) if x not in have]
        if not targets:
            if len(closed) == len(e["beyond"]) and all(x == from_u(m, f, e["value"]) for x in steps):
                stopped[k] = f"{BOUND_STOP} ({_fmt(from_u(m, f, e['value']))})"
            continue
        why = _why(e)
        group = [{"key": k, "levels": {k: x}, "why": why, "combo": ""} for x in targets]
        for j, z in pairs.get(k, ()) if not e["beyond"] else ():
            pj = led.get(j)
            if pj is None or not pj["step"] or j not in icfg or not _allowed(ctx, j) or frozenset([k, j]) in done:
                continue
            mj = meta(ctx, j, icfg[j])
            back = from_u(mj, to_u(mj, icfg[j]) - pj["dir"] * pj["step"], icfg[j])
            if back != icfg[j]:
                group.append({"key": k, "levels": {k: targets[0], j: back}, "combo": j,
                              "why": f"{why}; substitute {j} (interaction z {z:+.1f}) backed off one step"})
        moves.append(group)
    rr = [g[i] for i in range(max(map(len, moves), default=0)) for g in moves if i < len(g)]   # every key's 1st first
    return rr, stopped


def tick(ctx: Any, store: Any, adapter: Any, snapshot: Any, *, new_id: Any, now: float | None = None,
         scarce: bool = False) -> list[str]:
    """Build the planned extrapolation nodes on the incumbent while fewer than ``max_inflight`` are open (none in
    scarce mode); record the directions that stop."""
    ex = _cfg(ctx)
    if ex is None or not ex.enabled or snapshot is None or scarce:
        return []
    now = time.time() if now is None else now
    bk, acts = book(store), []
    inc = getattr(snapshot, "incumbent_id", "") or ROOT_NODE_ID
    open_ = [nid for nid in bk["nodes"] if (n := store.get_node(nid)) is not None and n.status not in CLOSED_STATUSES]
    room, seen = ex.max_inflight - len(open_), [inc, len(open_), len(bk["nodes"])]
    if room <= 0 or (bk.get("seen") == seen and now < bk.get("at", 0.0) + 60.0 * ex.every_min):
        return []                       # re-planned when its incumbent / open nodes change, else every every_min
    bk.update(seen=seen, at=now)
    view = View(store)
    moves, stopped = plan(ctx, store, snapshot, view)
    for k in {mv["key"] for mv in moves}:    # a bound stop lifted (its bounds changed): the direction goes on
        if str(bk["stopped"].get(f"{k}@{inc}", "")).startswith(BOUND_STOP):
            del bk["stopped"][f"{k}@{inc}"]
    for k, why in stopped.items():
        if bk["stopped"].get(f"{k}@{inc}") == why:
            continue                         # a bound stop re-found on a re-plan: logged once
        bk["stopped"][f"{k}@{inc}"] = why
        store.log("extrapolate_stopped", key=k, incumbent=inc, why=why)
        acts.append(f"extrapolate: {k} stopped on {inc} ({why})")
    moves = moves[:room]
    if moves:
        acts += _build(ctx, store, adapter, bk, view, inc, moves, new_id, now)
    store.set_state(K_EXT, bk)
    return acts


def _build(ctx: Any, store: Any, adapter: Any, bk: dict, view: View, inc: str, moves: Sequence[Mapping[str, Any]],
           new_id: Any, now: float) -> list[str]:
    from loop import campaign as CP
    from loop import design as D
    from loop.contracts import AxisPrediction, HypothesisKind, HypothesisRecord, Operator
    from loop.select.intake import family_of
    acts, icfg = [], view.cfg(inc)
    with D._Tree(ctx, store, adapter, f"extrap-{inc}", base_id=inc) as t:
        for mv in moves:
            real = CP.expand(t.base, mv["levels"])
            if not all(D._knob_ok(ctx, t.text, t.base, {"key": k, "low": t.base[k], "high": v}) for k, v in real.items()):
                store.log("extrapolate_skipped", key=mv["key"], levels=mv["levels"], why="not settable")
                continue
            move = ", ".join(f"{k} {_fmt(icfg.get(k))}->{_fmt(x)}" for k, x in mv["levels"].items())
            fam = family_of(ctx, mv["key"].split("[", 1)[0]) or "optimization"
            hyp = HypothesisRecord(
                id="", task_id=ctx.task_id, kind=HypothesisKind.TRIAL, family=fam, mechanism_tag=f"extrapolate_{mv['key']}",
                claim=f"the winning direction of {mv['key']} continues on the incumbent {inc}: {move} "
                      f"(direction from {mv['why']})", predictions={PANEL: AxisPrediction(1, 1e-3)}, created_ts=now,
                falsification=f"{move} does not beat {inc}")
            tag = "extrap_" + "_".join(k.rsplit(".", 1)[-1] for k in mv["levels"])
            nid = t.node(real, knobs=[{"key": k, "low": t.base[k], "high": v, "levers": {}} for k, v in real.items()],
                         tag=tag, operator=Operator.DRAFT, new_id=new_id, hyp=hyp,
                         notes=f"brief=extrapolate extrapolate={mv['key']}" + (f" partner={mv['combo']}" if mv["combo"] else ""))
            if not nid or nid == inc:
                store.log("extrapolate_skipped", key=mv["key"], levels=mv["levels"],
                          why="rejected at intake" if not nid else "= the incumbent")
                continue
            store.update_node(nid, family=fam)
            bk["nodes"][nid] = {"incumbent": inc, "keys": list(mv["levels"]), "levels": dict(mv["levels"]),
                                "why": mv["why"], "ts": now}
            store.log("extrapolate", node_id=nid, key=mv["key"], incumbent=inc, levels=mv["levels"], why=mv["why"])
            acts.append(f"extrapolate: {move} (direction from {mv['why']}) -> {nid} on {inc}")
    return acts


# ------------------------------------------------------------------------------------------- visibility
def facts(ctx: Any, store: Any, snapshot: Any, limit: int = 8) -> str:
    """The ledger's strongest keys as computed facts, e.g. ``dino.layerwise_decay: + direction (net +6.1 sigma, 9
    items); last winning step 0.7->0.8 +0.0030 (lineage a node confirmed); on the incumbent a node (0.8): 0.9 / 1.0
    untested``; '' when there is none."""
    if snapshot is None:
        return ""
    try:
        view = View(store)
        led = ledger(ctx, store, snapshot, view)
        moves, _ = plan(ctx, store, snapshot, view, led)
    except Exception:  # noqa: BLE001 - analytics never blocks a prompt
        return ""
    inc, bk, out = getattr(snapshot, "incumbent_id", "") or ROOT_NODE_ID, book(store), []
    for k, e in list(led.items())[:limit]:
        if not e["last"]:
            continue
        it = e["last"]
        nxt = [mv["levels"][k] for mv in moves if mv["key"] == k and not mv["combo"]]
        seen = "; ".join(f"{_fmt(x)} " + ("open" if g is None else f"{g:+.4f}") + f" ({nid})"
                         for x, g, nid, _ in e["beyond"]) or "nothing beyond it tested"
        stop = bk["stopped"].get(f"{k}@{inc}")
        out.append(f"- {k} ({e['scale']}): {'+' if e['dir'] > 0 else '-'} direction (net {e['z']:.1f} sigma, "
                   f"{e['n']} items, {e['pos']} winning); last winning step {_fmt(it['a'])}->{_fmt(it['b'])} "
                   f"{it['gain']:+.4f} ({it['src']}); on the incumbent {inc} ({_fmt(e['value'])}): {seen}"
                   + (f"; {' / '.join(map(_fmt, nxt))} untested" if nxt else "")
                   + (f"; stopped: {stop}" if stop else ""))
    return "\n".join(out)


def lines(store: Any) -> str:
    bk = book(store)
    out = [f"extrapolate {nid}: {', '.join(f'{k}={_fmt(v)}' for k, v in r['levels'].items())} on {r['incumbent']}"
           for nid, r in bk["nodes"].items()]
    out += [f"extrapolate stopped {k}: {why}" for k, why in bk["stopped"].items()]
    return "\n".join(out)
