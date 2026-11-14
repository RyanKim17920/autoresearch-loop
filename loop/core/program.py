"""loop.core.program — a PROGRAM node as a STAGED PLAN (moonshot-tuning-budget rule: a promising structural change
gets <= 5 tuning trials before it is judged; a factorial wave is >= 8 points, so a program never gets one).

v3-c4i (09Z MA, 10-13: program a node = 2 knobs of one family, +.003; one ladder over the first knob tuned one part of
a multi-part change): a program's finish declares ``components`` (family, mechanism, knobs, the brief row it targets,
its predicted share of the gain, optionally the config values that switch it off).

- :func:`ladders`        a SCREENING ``brief=program`` node of the current base with a belief above
                         ``-ideation.program_wave_min_z`` sd gets its dose budget -- ``program_wave_max_points - 1``
                         trials (the node is the remaining one) -- ALLOCATED ACROSS its components by their declared
                         shares (largest remainder; each component at least one while the budget lasts): each a
                         config-only child moving that component's first knob over its [lo, hi] (log-spaced for a
                         ``log`` knob), the node's own value left out. Without components: one ladder over the first
                         declared ``knobs`` entry (the c3i rule). Children are QUEUED (``ladder=<node>``: lane
                         factorial) and screened by the acquisition like any QUEUED node. Once per node
                         (``K_LADDERS``).
- :func:`leave_one_out`  once the full program screens positive (belief > ``program_loo_min_z`` sd), one config-only
                         child per component with that component switched off (its ``off`` values, else the parent's
                         values of its knobs; a component with neither is skipped): which part carries the gain
                         (``ideation.program_loo``; ``K_LOO``). Its children are attribution, not tuning: at most
                         ``program_loo_max`` (3) of them (largest declared share first), and they COUNT in the
                         program's trial budget: node + doses + LOO <= ``program_wave_max_points`` +
                         ``program_loo_max``. A component's ``off`` may only name its OWN declared knob keys (any
                         other key is dropped: switching one part off never moves another part's knob).

v3-c8a (operator program 10-14: both eval-only gate arms only WEAKENED the gate -- off / floor .5 -- and both lost
slide -.092 / -.056 for cls +.0136 / +.0115; the stronger-gate direction was never tried):

- ladders are SYMMETRIC (``ideation.program_ladder_symmetric``): a component's doses sit on BOTH sides of the node's
  own value (half below, half above; a declared range that ends at the value is mirrored past it, :func:`mirror`:
  x2 <-> x0.5, a bounded knob stays inside (0, 1)); and a dose measured HARMFUL (:func:`harmed`) whose knob has no
  dose on the other side gets that mirrored dose, once (:func:`opposite_doses`, ``K_OPP``).
- an operator program's eval-only arm gets its OPPOSITE arm (:func:`opposite_arm`: the arm's declared ``opposite``
  literal edits, else its numeric literals mirrored): built beside it by default (``ideation.evalonly_both_
  directions``), else once the arm is measured harmful.
- B4 ISOLATION (``ideation.program_iso``; a node's views ladder carried its crop-scale change in every point): every
  config key the node changed vs its parent that no ladder component doses gets its own leave-one-out child (that key
  reverted to the parent's value only; lever-group-neutral keys such as num_workers excepted), largest relative
  change first, at most ``program_loo_max`` (``K_ISO``); they count against the component LOO budget.

Nothing is named."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import replace
from typing import Any, Mapping, Sequence

K_LADDERS = "ideation.program_ladders"     # {program node: [ladder nodes] | "-: why not"}
K_LOO = "ideation.program_loo"             # {program node: {component index: node | "-: why not"}}
K_ARMS = "ideation.program_arms"           # {operator program id: {arm id: node | "-: why not" | {"fails": n}}}
K_OPP = "ideation.program_opposite"        # {program node: {knob key: mirrored dose node | "-: why not"}}
K_ISO = "ideation.program_iso"             # {program node: {config key: leave-that-key-out node | "-: why not"}}
K_ISO_ROOM = "ideation.program_iso_room"   # {program node: ladder points reserved for its isolation children (v3 review S6)}
OPP_SUFFIX = "_opp"                        # an arm's opposite arm id: <arm id>_opp
BOUNDED = re.compile(r"(ratio|prob|frac|fraction|momentum|ema|beta|rate|decay|_p|_scale)$")   # as loop.tune._BOUNDED + *_scale (crops)
_NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w.])")
ARM_FAILS = 3                              # a raising arm build is retried on later ticks this often
TAG = " brief=program"


def doses(lo: float, hi: float, k: int, scale: str, cur: Any = None) -> list[float]:
    """``k`` doses spanning [lo, hi] (geometric for ``log`` with positive ends), the current value left out."""
    if k <= 0:
        return []
    if k == 1:
        xs = [hi]
    elif scale == "log" and lo > 0 and hi > 0:
        xs = [lo * (hi / lo) ** (i / (k - 1)) for i in range(k)]
    else:
        xs = [lo + (hi - lo) * i / (k - 1) for i in range(k)]
    out = [float(f"{x:.6g}") for x in xs]
    return [x for x in out if not (isinstance(cur, (int, float)) and math.isclose(x, cur, rel_tol=1e-6))]


def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def mirror(cur: Any, x: Any, key: str = "", lo: Any = None, hi: Any = None) -> float | None:
    """The dose on the OTHER side of ``cur`` at the same distance: geometric (``cur^2 / x``: x2 <-> x0.5) for positive
    values, else linear (``2 cur - x``); a bounded knob (ratio / prob / momentum / *_scale ... with ``cur`` in (0, 1))
    stays inside (0, 1) (halfway to the bound instead of past it). Clamped to the knob's declared [``lo``, ``hi``] when
    given (v3 review M5). An int knob (both ``cur`` and ``x`` ints) mirrors to an int >= 1 that is not ``cur`` (a 2 ->
    1 views ladder never mirrors to 0). None when there is none (not numbers, x == cur, a zero or a sign flip)."""
    if not _num(cur) or not _num(x) or x == cur:
        return None
    y = cur * cur / x if cur > 0 and x > 0 else 2 * cur - x
    if 0 < cur < 1 and BOUNDED.search(key.rsplit(".", 1)[-1].split("[", 1)[0]):
        y = (1 + cur) / 2 if y >= 1 else cur / 2 if y <= 0 else y
    if _num(hi) and y > hi:
        y = hi
    if _num(lo) and y < lo:
        y = lo
    if y == 0 or (cur != 0 and (y > 0) != (cur > 0)):
        return None
    if isinstance(cur, int) and isinstance(x, int) and not isinstance(cur, bool):
        if (r := int(round(y))) < 1 or r == cur:
            return None
    return float(f"{y:.6g}")


def _ok_dose(ctx: Any, key: str, y: Any, cur: Any) -> bool:
    """v3 review M5: a (typed) dose ``y`` is usable: a number other than ``cur``, an int knob's >= 1, and inside the
    knob's declared hard bounds (:func:`loop.extrapolate.inside`: the yaml rows, a unit-interval leaf stays in [0, 1])."""
    if not _num(y) or y == cur:
        return False
    if isinstance(cur, int) and not isinstance(cur, bool) and y < 1:
        return False
    try:
        from loop.extrapolate import inside
        return bool(inside(ctx, key, y, cur))
    except Exception:  # noqa: BLE001 - no metadata: the dose is judged by the checks above only
        return True


def isclose_any(x: float, ends: Sequence[float]) -> bool:
    return any(math.isclose(x, e, rel_tol=1e-6) for e in ends)


def symmetric_doses(lo: float, hi: float, k: int, scale: str, cur: Any, key: str = "") -> list[float]:
    """``k`` doses on BOTH sides of the node's own value ``cur`` (v3-c8a B1): ``k // 2`` below, the rest above, each
    side spaced toward its end of [lo, hi] (geometric for ``log``); a range that ends at / before ``cur`` on one side
    is mirrored past it (:func:`mirror`). One side impossible: all on the other. ``k < 2`` or a non-numeric ``cur``:
    :func:`doses`."""
    if k < 2 or not _num(cur):
        return doses(lo, hi, k, scale, cur)
    top = hi if hi > cur else mirror(cur, lo, key, lo, hi)
    bot = lo if lo < cur else mirror(cur, hi, key, lo, hi)
    top = top if top is not None and top > cur else None
    bot = bot if bot is not None and bot < cur else None
    below = (k // 2 if top is not None else k) if bot is not None else 0
    above = k - below if top is not None else 0

    def side(end: float | None, n: int) -> list[float]:
        if end is None or n <= 0:
            return []
        if scale == "log" and cur > 0 and end > 0:
            return [cur * (end / cur) ** (i / n) for i in range(1, n + 1)]
        return [cur + (end - cur) * i / n for i in range(1, n + 1)]
    xs = [float(f"{x:.6g}") for x in side(bot, below) + side(top, above)]
    xs = [x for x in xs if lo <= x <= hi or isclose_any(x, (lo, hi))]       # v3 review M5: never past the declared range
    return [x for x in xs if not math.isclose(x, cur, rel_tol=1e-6)] or doses(lo, hi, k, scale, cur)


def harmed(store: Any, post: Any, nid: str, z: float = 1.0) -> str:
    """Why node ``nid`` was measured HARMFUL ('' = not): held by the eval-only harm guard (an axis lost past its
    tolerance), VETOED (an axis veto), or a posterior own gain below ``-z`` sd."""
    eo = ((store.get_state("evalonly.v3") or {}).get("harm") or {})
    if nid in eo:
        return str(eo[nid])[:200]
    node = store.get_node(nid)
    if node is not None and getattr(node.status, "value", node.status) == "vetoed":
        return "vetoed"
    if post is not None and nid in getattr(post, "est", {}):
        m, v = post.belief(nid)
        if m < -z * math.sqrt(max(v, 0.0)):
            return f"own gain {m:+.4f} < -{z:g} sd"
    return ""


def typed_like(x: Any, ref: Any) -> Any:
    """``x`` in the type of the config value ``ref`` it replaces (so ``local_views = 20.0`` never reaches an int
    knob; the same job as ``loop.core.factorial.coerce_like``, kept local until that exists): an int knob gets the
    rounded int, a float knob a float, a bool stays a bool (0 / 1 only). ``None`` = not representable."""
    if isinstance(ref, bool):
        return x if isinstance(x, bool) else bool(x) if x in (0, 1) else None
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return x if not isinstance(ref, (int, float)) else None
    if isinstance(ref, int):
        return int(round(x)) if math.isfinite(x) else None
    return float(x) if isinstance(ref, float) else x


def typed_doses(xs: Sequence[float], cur: Any) -> list[Any]:
    """``xs`` in ``cur``'s type, the rungs that collapsed (20.0 / 20.4 -> 20) deduped, ``cur`` itself left out."""
    out: list[Any] = []
    for x in xs:
        v = typed_like(x, cur)
        if v is not None and v != cur and v not in out:
            out.append(v)
    return out


def allocate(shares: Sequence[float], budget: int) -> list[int]:
    """Doses per component: ``budget`` split by the (normalised) shares, largest remainder; every component gets one
    first while the budget lasts (largest share first), so no declared part goes untuned."""
    n = len(shares)
    if n == 0 or budget <= 0:
        return [0] * n
    w = [max(0.0, float(s)) for s in shares]
    tot = sum(w)
    w = [x / tot for x in w] if tot > 0 else [1.0 / n] * n
    out = [0] * n
    for i in sorted(range(n), key=lambda i: (-w[i], i))[:min(n, budget)]:
        out[i] = 1
    rest = budget - sum(out)
    if rest > 0:
        want = [max(0.0, w[i] * budget - out[i]) for i in range(n)]
        base = [int(math.floor(x)) for x in want]
        while sum(base) > rest:
            base[max(range(n), key=lambda i: base[i])] -= 1
        left = rest - sum(base)
        for i in sorted(range(n), key=lambda i: (-(want[i] - base[i]), -w[i], i))[:left]:
            base[i] += 1
        out = [a + b for a, b in zip(out, base)]
    return out


def components(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The plan's well-formed components (each with >= 1 declared knob, ``loop.core.factorial.declared``), in order;
    none: the plan's ``knobs`` as ONE component (the c3i single ladder)."""
    from loop.core import factorial as F
    out = []
    for c in plan.get("components") or ():
        if isinstance(c, Mapping) and (kn := F.declared(c.get("knobs"))):
            out.append({**dict(c), "_knobs": kn})
    if out:
        return out
    kn = F.declared(plan.get("knobs"))
    return [{"share": 1.0, "_knobs": kn[:1], "mechanism": "", "target": ""}] if kn else []


def resolve_key(key: str, cfg: Mapping[str, Any], prefer: Sequence[str] = ()) -> str:
    """A component knob's key as the flattened config spells it: ``key`` itself when the config has it; else the
    UNIQUE key ending in ``.key`` -- among ``prefer`` (the node's own ``config_diff`` keys) first, then the config
    (a node declared bare ``gram_patch_weight`` for ``dino.gram_patch_weight``: its ladder failed "absent from the
    config"). Ambiguous or unknown: ``key`` unchanged."""
    if key in cfg or not key:
        return key
    for pool in (list(prefer), list(cfg)):
        hits = sorted({k for k in pool if k.endswith("." + key)})
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            return key
    return key


def key_candidates(key: str, cfg: Mapping[str, Any], prefer: Sequence[str] = ()) -> list[str]:
    """Every flattened key ``key`` could mean: itself when the config has it, else the keys ending in ``.key`` (the
    node's own ``prefer`` keys first when any of them match); [] when none."""
    if not key:
        return []
    if key in cfg:
        return [key]
    for pool in (list(prefer), list(cfg)):
        hits = sorted({k for k in pool if k.endswith("." + key)})
        if hits:
            return hits
    return []


def resolve_plan_keys(plan: Mapping[str, Any], cfg: Mapping[str, Any], prefer: Sequence[str] = ()
                      ) -> tuple[dict[str, Any], list[str]]:
    """v3-c7i I8: a finish / plan whose knob keys (``knobs``, each component's ``knobs`` and ``off``) are spelled bare
    gets them resolved by UNIQUE suffix match against the flattened config (:func:`key_candidates`): (the plan with
    full dotted keys, the problems -- an ambiguous or unknown key, with its candidates). ``cfg`` empty: unchanged."""
    import copy
    import difflib
    out, problems = copy.deepcopy(dict(plan)), []
    if not cfg:
        return out, problems

    def fix(key: Any, where: str) -> Any:
        k = str(key or "")
        if not k or k in cfg:
            return key
        hits = key_candidates(k, cfg, prefer)
        if len(hits) == 1:
            return hits[0]
        if hits:
            problems.append(f"{where} key {k!r} is ambiguous: {', '.join(hits[:8])} -- write the full dotted key")
        else:
            near = difflib.get_close_matches(k, list(cfg), n=4, cutoff=0.5)
            problems.append(f"{where} key {k!r} is not a config key" + (f" (nearest: {', '.join(near)})" if near else "")
                            + " -- write the full dotted key of an existing (or your newly added) config key")
        return key
    for i, kb in enumerate(out.get("knobs") or ()):
        if isinstance(kb, dict) and "key" in kb:
            kb["key"] = fix(kb["key"], f"knobs[{i}]")
    for i, c in enumerate(out.get("components") or ()):
        if not isinstance(c, dict):
            continue
        for j, kb in enumerate(c.get("knobs") or ()):
            if isinstance(kb, dict) and "key" in kb:
                kb["key"] = fix(kb["key"], f"components[{i}].knobs[{j}]")
        if isinstance(c.get("off"), Mapping):
            c["off"] = {fix(k, f"components[{i}].off"): x for k, x in c["off"].items()}
    return out, problems


#: the "-: why" ladder records the bare-key bug wrote (a node): cleared once so they are re-planned
BARE_KEY_WHY = "absent from the config"


def retry_bare_key_failures(store: Any, cfg_of: Any) -> list[str]:
    """Permanently-failed ladder records (``K_LADDERS`` "-: ...") whose every reason was a bare component key that
    :func:`resolve_key` now maps: removed so :func:`ladders` plans the node again. ``cfg_of(node id)`` -> its
    flattened config ({} when unknown). Returns the cleared node ids."""
    book = dict(store.get_state(K_LADDERS) or {})
    out = []
    for nid, rec in list(book.items()):
        if not (isinstance(rec, str) and rec.startswith("-: ")):
            continue
        whys = [w.strip() for w in rec[3:].split(";") if w.strip()]
        keys = [w[:-len(BARE_KEY_WHY)].strip() for w in whys if w.endswith(BARE_KEY_WHY)]
        if not whys or len(keys) != len(whys):
            continue
        cfg = cfg_of(nid) or {}
        own = list((store.get_state(f"config_diff:{nid}") or {}).keys())
        if all(resolve_key(k, cfg, own) != k for k in keys):
            del book[nid]
            out.append(nid)
    if out:
        store.set_state(K_LADDERS, book)
        store.log("program_ladder_retry", nodes=out, why="bare component key resolved")
    return out


def _new_id(store: Any) -> str:
    seq = len(store.nodes())
    while store.get_node(f"n{seq:04d}") is not None:
        seq += 1
    return f"n{seq:04d}"


def _child(store: Any, sn: Any, sha: str, now: float, tag: str, notes: str, cfg: Mapping[str, Any]) -> str:
    from loop.contracts import Node as StoreNode, NodeStatus, Operator, Strategy
    cid = _new_id(store)
    store.put_node(StoreNode(id=cid, task_id=sn.task_id, commit=sha, parent_id=sn.id, strategy=Strategy.EXPLORE,
                             operator=Operator.IMPROVE, family=sn.family, mechanism_tag=f"{sn.mechanism_tag}_{tag}",
                             diff_hash=hashlib.sha256(f"{sn.commit}\n{sha}".encode()).hexdigest(),
                             status=NodeStatus.QUEUED, hypothesis_id=sn.hypothesis_id, created_ts=now,
                             updated_ts=now, notes=notes))
    store.set_state(f"config_diff:{cid}", dict(cfg))
    return cid


def _eligible(led: Any, post: Any, statuses: Mapping[str, Any], nid: str, node: Any) -> bool:
    st = getattr(statuses.get(nid), "value", statuses.get(nid))
    return TAG in f" {node.notes}" and node.base == led.epoch and str(st).lower() == "screening" \
        and nid in getattr(post, "est", {})


def ladders(ctx: Any, store: Any, adapter: Any, led: Any, post: Any, statuses: Mapping[str, Any], now: float
            ) -> list[str]:
    """Build the dose ladders owed (see the module docstring), then the leave-one-out children owed; returns the
    tick's action lines."""
    from loop.adapters import flatten
    from loop.core import factorial as F
    from loop.core.levers import opt
    from loop.extrapolate import _allowed
    if post is None or not getattr(led, "epoch", ""):
        return []
    z, pts = float(opt(ctx, "program_wave_min_z")), int(opt(ctx, "program_wave_max_points"))

    def cfg_of(nid: str) -> dict[str, Any]:
        sn = store.get_node(nid)
        try:
            return flatten(adapter.config_at(sn.commit)) if sn is not None and hasattr(adapter, "config_at") else {}
        except Exception:  # noqa: BLE001 - unreadable config: not resolvable
            return {}
    retried = retry_bare_key_failures(store, cfg_of)
    book, acts = dict(store.get_state(K_LADDERS) or {}), []
    if retried:
        acts.append(f"program ladder: re-planning {retried} (their bare component keys now resolve)")
    iso_done = store.get_state(K_ISO) or {}
    for nid, node in led.nodes.items():
        if nid in book and isinstance(book[nid], list) and book[nid] and nid not in iso_done \
                and opt(ctx, "program_iso") and _eligible(led, post, statuses, nid, node) \
                and post.belief(nid)[0] > float(opt(ctx, "program_loo_min_z")) * math.sqrt(max(post.belief(nid)[1], 0.0)):
            acts += _backfill_iso(ctx, store, adapter, nid, node, book[nid], now, pts)   # B4, once it screens positive
        if nid in book or not _eligible(led, post, statuses, nid, node):
            continue
        comps = components(store.get_state(f"ideation:{nid}") or {})
        if not comps:
            continue
        m, v = post.belief(nid)
        if m <= -z * math.sqrt(max(v, 0.0)):
            from loop.core import struct_tune as ST       # 10-22: an ENGAGED but negative program node most needs its
            if ST.steer(ctx, store, led, post, nid) != "engaged_negative":    # lower-dose rungs: it stays eligible
                continue
        sn = store.get_node(nid)
        try:
            cfg, unreadable = flatten(adapter.config_at(sn.commit)) if hasattr(adapter, "config_at") else {}, False
        except Exception:  # noqa: BLE001 - unreadable config: nothing can be dosed safely
            cfg, unreadable = {}, True
        if unreadable:                                # v3-c7i I8: an unreadable config is no verdict: never booked
            acts.append(f"program ladder {nid}: its config is unreadable now; re-planned on a later tick")
            continue
        usable, why = [], []
        own = list((store.get_state(f"config_diff:{nid}") or {}).keys())
        for i, c in enumerate(comps):
            kb = c["_knobs"][0]
            if (full := resolve_key(kb.key, cfg, own)) != kb.key:   # a bare component key -> its flattened key
                kb = replace(kb, key=full)
            if cfg.get(kb.key) is None:               # the key is absent from the config: nothing to dose
                cand = key_candidates(kb.key, cfg, own)
                why.append(f"{kb.key} absent from the config" + (f" (ambiguous: {', '.join(cand[:6])})"
                                                                   if len(cand) > 1 else ""))
            elif not _allowed(ctx, kb.key):           # protected keys / init-check families, as extrapolate does
                why.append(f"{kb.key} is protected or has no tunable family")
            else:
                usable.append((i, c, kb, cfg.get(kb.key)))
        if not usable:
            book[nid] = "-: " + "; ".join(why or ["no dosable knob"])
            store.set_state(K_LADDERS, book)
            acts.append(f"program ladder {nid}: {book[nid][3:]}")
            continue
        reserve = 0                                   # v3 review S6: the isolation points are part of the ladder's budget
        if opt(ctx, "program_iso"):
            from loop.extrapolate import _allowed as _ok
            _p, ikeys = iso_keys(ctx, store, adapter, sn, node, cfg, {kb.key for _, _, kb, _ in usable})
            reserve = min(max(0, int(opt(ctx, "program_loo_max"))), sum(1 for k in ikeys if k in _p and _ok(ctx, k)),
                          max(0, pts - 1 - len(usable)))
        if reserve:
            store.set_state(K_ISO_ROOM, {**(store.get_state(K_ISO_ROOM) or {}), nid: reserve})
        alloc = allocate([float(c.get("share") or 0.0) for _, c, _, _ in usable], pts - 1 - reserve)
        made: list[str] = []
        book[nid] = made                              # the SAME list: saved after each node, never rebuilt
        sym = bool(opt(ctx, "program_ladder_symmetric"))
        for (i, c, kb, cur), k in zip(usable, alloc):
            span = (lambda n, kb=kb, cur=cur: symmetric_doses(float(kb.lo), float(kb.hi), n, kb.scale, cur, kb.key)) \
                if sym else (lambda n, kb=kb, cur=cur: doses(float(kb.lo), float(kb.hi), n, kb.scale, cur))
            xs = [x for x in typed_doses(span(k), cur) if _ok_dose(ctx, kb.key, x, cur)]
            if len(xs) < k:                           # the node's own value / a collapsed rung: spend the slot elsewhere
                xs = [x for x in typed_doses(span(k + 1), cur) if _ok_dose(ctx, kb.key, x, cur)][:k]
            for j, x in enumerate(xs):
                sha, err = F.build_point(adapter, sn.commit, {kb.key: x}, [F.Knob(kb.key, cur, x, "lin")], {},
                                         f"ladder-{nid}-{i}-{j}", getattr(adapter, "config_rel", "config.yaml"))
                if not sha or sha == sn.commit:
                    acts.append(f"program ladder {nid}: {kb.key}={x} not built ({err or 'no change'})")
                    continue
                made.append(_child(store, sn, sha, now, "dose", f"ladder={nid} dose={kb.key}={json.dumps(x)}"
                                   + (f" component={i}" if len(comps) > 1 else ""), {kb.key: x}))
                store.set_state(K_LADDERS, book)      # a failure mid-ladder leaves the built doses booked
        book[nid] = made or "-: nothing built"
        store.log("program_ladder", node_id=nid, components=len(comps), alloc=alloc, nodes=made)
        acts.append(f"program ladder {nid}: {len(made)} dose(s) over {len(usable)} component(s) "
                    f"{[c.get('mechanism') or kb.key for _, c, kb, _ in usable]} (alloc {alloc}) {made}")
        # v3-c8a B4 isolation is NOT built here (v3 review S6): its children are built once the program node screens
        # positive (the loop below, as the component LOO), and they are charged to this ladder's points budget
    if acts:
        store.set_state(K_LADDERS, book)
    acts += opposite_doses(ctx, store, adapter, post, cfg_of, now)
    acts += leave_one_out(ctx, store, adapter, led, post, statuses, now) if opt(ctx, "program_loo") else []
    # 10-22: structural nodes (a backbone swap, a new loss, a bold / moonshot change) get their own tuning ladder before
    # they can count as negative (loop.core.struct_tune); old-epoch backbone swaps / the operator's list are ported for it
    from loop.core import struct_tune as ST
    try:                    # 11-10: the moonshot rescue first (debug / port onto the incumbent), then the ladders it tags
        from loop.core import rescue as RS
        acts += RS.step(ctx, store, adapter, led, post, statuses, now)
    except Exception as exc:  # noqa: BLE001 - the rescue never stops the structural ladders
        store.log("moonshot_rescue_error", error=repr(exc)[:500])
        acts.append(f"moonshot rescue: error {exc!r}"[:300])
    return (acts + ST.struct_ladders(ctx, store, adapter, led, post, statuses, now) + ST.retro(ctx, store, adapter, led, now)
            + ST.report(ctx, store, led, post, now))


def _backfill_iso(ctx: Any, store: Any, adapter: Any, nid: str, node: Any, kids: Sequence[str], now: float,
                  pts: int = 0) -> list[str]:
    """B4: the isolation children of a ladder whose program node screens positive: its dosed keys are the keys its dose
    children set; they are charged to the ladder's points budget (``program_wave_max_points`` minus the node itself and its
    doses; 0 = no budget left)."""
    from loop.adapters import flatten
    sn = store.get_node(nid)
    try:
        cfg = flatten(adapter.config_at(sn.commit)) if sn is not None and hasattr(adapter, "config_at") else {}
    except Exception:  # noqa: BLE001 - unreadable: retried on a later tick
        return []
    if not cfg:
        return []
    dosed = {d[0] for c in kids if (cn := store.get_node(c)) is not None and (d := _dose_note(cn.notes))}
    res = (store.get_state(K_ISO_ROOM) or {}).get(nid)
    room = int(res) if res is not None else None      # a ladder built before the reservation: capped by program_loo_max only
    return isolate(ctx, store, adapter, sn, node, cfg, dosed, now, room)


def rel_change(a: Any, b: Any) -> float:
    """How far a config value moved: |log2(b / a)| for same-signed non-zero numbers, else |b - a| / max(|a|, 1e-9);
    equal-length lists: the largest element move; anything else that changed: 1."""
    if a == b:
        return 0.0
    if _num(a) and _num(b):
        if a and b and (a > 0) == (b > 0):
            return abs(math.log2(b / a))
        return abs(b - a) / max(abs(a), 1e-9)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == len(b) and a:
        return max(rel_change(x, y) for x, y in zip(a, b))
    return 1.0


def iso_keys(ctx: Any, store: Any, adapter: Any, sn: Any, node: Any, cfg: Mapping[str, Any], dosed: set[str]
             ) -> tuple[dict[str, Any], list[str]]:
    """(the parent's flattened config, the changed keys no ladder component doses, largest change first, neutral keys
    left out): what :func:`isolate` may build, and what the ladder reserves points for (v3 review S6)."""
    from loop.adapters import flatten
    from loop.core import mixed as MX
    parent = store.get_node(node.parent or sn.parent_id or "")
    try:
        pcfg = flatten(adapter.config_at(parent.commit)) if parent is not None and hasattr(adapter, "config_at") else {}
    except Exception:  # noqa: BLE001 - no parent config: nothing to isolate against
        return {}, []
    if not pcfg:
        return {}, []
    changed = {k: rel_change(pcfg.get(k), cfg.get(k)) for k in set(pcfg) | set(cfg)
               if json.dumps(pcfg.get(k), sort_keys=True, default=str) != json.dumps(cfg.get(k), sort_keys=True, default=str)}
    return pcfg, [k for k in sorted(changed, key=lambda k: (-changed[k], k)) if k not in dosed and not MX.neutral(ctx, k)]


def isolate(ctx: Any, store: Any, adapter: Any, sn: Any, node: Any, cfg: Mapping[str, Any], dosed: set[str],
            now: float, room: int | None = None) -> list[str]:
    """v3-c8a B4: the config keys program node ``sn`` changed vs its parent (effective configs) that no ladder
    component doses -- each its own config-only child with ONLY that key reverted to the parent's value
    (``ladder=<node> loo_key=<key>``), largest :func:`rel_change` first, at most ``program_loo_max``; a key the
    lever map calls neutral (:func:`loop.core.mixed.neutral`: num_workers, logging ...) or a protected / new key
    (no parent value) is skipped. Once per node (``K_ISO``)."""
    from loop.adapters import flatten
    from loop.core import factorial as F
    from loop.core import mixed as MX
    from loop.core.levers import opt
    from loop.extrapolate import _allowed
    book = dict(store.get_state(K_ISO) or {})
    if sn.id in book:
        return []
    pcfg, keys = iso_keys(ctx, store, adapter, sn, node, cfg, dosed)
    if not pcfg:
        return []
    made: dict[str, str] = {}
    book[sn.id] = made
    cap = max(0, int(opt(ctx, "program_loo_max")))
    if room is not None:                        # v3 review S6: the ladder's points budget also holds the isolation points
        cap = min(cap, room)
    for k in keys:
        if sum(isinstance(v, str) and not v.startswith("-") for v in made.values()) >= cap:
            made[k] = f"-: over the LOO budget (program_loo_max {cap}, ladder points left {room})"
            continue
        if k not in pcfg:
            made[k] = "-: a new key (no parent value to revert to)"
            continue
        if not _allowed(ctx, k):
            made[k] = "-: protected or has no tunable family"
            continue
        v = typed_like(pcfg[k], cfg.get(k)) if _num(pcfg[k]) else pcfg[k]
        sha, err = F.build_point(adapter, sn.commit, {k: v}, [F.Knob(k, cfg.get(k), v, "lin")], {},
                                 f"iso-{sn.id}-{len(made)}", getattr(adapter, "config_rel", "config.yaml"))
        if not sha or sha == sn.commit:
            made[k] = f"-: not built ({err or 'no change'})"
            continue
        made[k] = _child(store, sn, sha, now, f"iso{len(made)}", f"ladder={sn.id} loo_key={k}", {k: v})
        store.set_state(K_ISO, book)
    store.set_state(K_ISO, book)
    store.log("program_iso", node_id=sn.id, children=made, dosed=sorted(dosed))
    return [f"program isolation {sn.id}: " + ", ".join(f"{k} -> {v}" for k, v in made.items())] if made else []


def _dose_note(notes: str) -> tuple[str, Any] | None:
    m = re.search(r"\bdose=([^\s=]+)=(\S+)", notes or "")
    if not m:
        return None
    try:
        return m.group(1), json.loads(m.group(2))
    except ValueError:
        return None


def _bounds(ctx: Any, key: str, cur: Any) -> tuple[Any, Any]:
    """The knob's declared hard [low, high] (loop.extrapolate.meta; None = open)."""
    try:
        from loop.extrapolate import meta
        m = meta(ctx, key, cur) or {}
        return m.get("low"), m.get("high")
    except Exception:  # noqa: BLE001
        return None, None


def opposite_doses(ctx: Any, store: Any, adapter: Any, post: Any, cfg_of: Any, now: float) -> list[str]:
    """v3-c8a B1: a program's ladder dose measured HARMFUL (:func:`harmed`) whose knob has no dose on the OTHER side
    of the program's own value gets the mirrored dose (:func:`mirror`), once per (program, knob) (``K_OPP``)."""
    from loop.core import factorial as F
    from loop.core.levers import opt
    ladders_ = store.get_state(K_LADDERS) or {}
    book, acts = dict(store.get_state(K_OPP) or {}), []
    z = float(opt(ctx, "program_wave_min_z"))
    for nid, kids in ladders_.items():
        if not isinstance(kids, list) or not kids:
            continue
        by: dict[str, list[tuple[str, Any]]] = {}
        for c in kids:
            if (cn := store.get_node(c)) is not None and (d := _dose_note(cn.notes)) is not None:
                by.setdefault(d[0], []).append((c, d[1]))
        mine = dict(book.get(nid) or {})
        for key, pts in by.items():
            if key in mine:
                continue
            bad = [(c, x, w) for c, x in pts if (w := harmed(store, post, c, z))]
            if not bad:
                continue
            sn = store.get_node(nid)
            cur = (cfg_of(nid) or {}).get(key) if sn is not None else None
            if not _num(cur):
                mine[key] = "-: the program's own value is unreadable"
            else:
                c, x, why = bad[0]
                if any(_num(y) and (y - cur) * (x - cur) < 0 for _, y in pts):
                    mine[key] = f"-: {key} already has a dose on the other side of {cur}"
                elif (y := mirror(cur, x, key, *_bounds(ctx, key, cur))) is None or (y := typed_like(y, cur)) is None \
                        or y == cur or not _ok_dose(ctx, key, y, cur):
                    mine[key] = f"-: no mirrored dose of {x} around {cur} inside the knob's range"
                else:
                    sha, err = F.build_point(adapter, sn.commit, {key: y}, [F.Knob(key, cur, y, "lin")], {},
                                             f"opp-{nid}-{key.rsplit('.', 1)[-1]}"[:60],
                                             getattr(adapter, "config_rel", "config.yaml"))
                    if not sha or sha == sn.commit:
                        mine[key] = f"-: not built ({err or 'no change'})"
                    else:
                        mine[key] = _child(store, sn, sha, now, "opp", f"ladder={nid} dose={key}={json.dumps(y)} "
                                           f"opposite_of={c}", {key: y})
                        acts.append(f"program ladder {nid}: dose {key}={x} ({c}) harmful ({why}); the opposite "
                                    f"dose {key}={y} queued as {mine[key]}")
            book[nid] = mine
            store.set_state(K_OPP, book)
            store.log("program_opposite", node_id=nid, key=key, result=mine[key])
    return acts


def leave_one_out(ctx: Any, store: Any, adapter: Any, led: Any, post: Any, statuses: Mapping[str, Any], now: float
                  ) -> list[str]:
    """One config-only child per component of a positive-screening program (see the module docstring)."""
    from loop.adapters import flatten
    from loop.core import factorial as F
    from loop.core.levers import opt
    from loop.extrapolate import _allowed
    if post is None or not getattr(led, "epoch", ""):
        return []
    z = float(opt(ctx, "program_loo_min_z"))
    pts, cap = int(opt(ctx, "program_wave_max_points")), max(0, int(opt(ctx, "program_loo_max")))
    ladder_book = store.get_state(K_LADDERS) or {}
    book, acts = dict(store.get_state(K_LOO) or {}), []
    for nid, node in led.nodes.items():
        if nid in book or not _eligible(led, post, statuses, nid, node):
            continue
        plan = store.get_state(f"ideation:{nid}") or {}
        comps = [c for c in plan.get("components") or () if isinstance(c, Mapping)]
        if len(comps) < 2:
            continue
        m, v = post.belief(nid)
        if m <= z * math.sqrt(max(v, 0.0)):
            continue
        sn, parent = store.get_node(nid), store.get_node(node.parent or "")
        try:
            pcfg = flatten(adapter.config_at(parent.commit)) if parent is not None and hasattr(adapter, "config_at") \
                else {}
            ncfg = flatten(adapter.config_at(sn.commit)) if hasattr(adapter, "config_at") else {}
        except Exception:  # noqa: BLE001
            pcfg, ncfg = {}, {}
        made: dict[str, str] = {}
        book[nid] = made
        dosed = ladder_book.get(nid)
        iso = {k: v for k, v in ((store.get_state(K_ISO) or {}).get(nid) or {}).items()
               if isinstance(v, str) and not v.startswith("-")}         # v3-c8a B4: its single-key LOO points
        left = min(cap - len(iso), pts + cap - 1 - (len(dosed) if isinstance(dosed, list) else 0))   # the trial budget
        order = sorted(range(len(comps)), key=lambda i: (-float(comps[i].get("share") or 0.0), i))
        for i in order:
            c = comps[i]
            if left <= 0:
                made[str(i)] = f"-: over the LOO budget (program_loo_max {cap})"
                continue
            mine = list((store.get_state(f"config_diff:{nid}") or {}).keys())
            own = [resolve_key(kb.key, ncfg or pcfg, mine) for kb in F.declared(c.get("knobs"))]
            off = {k2: x for k, x in (c.get("off") or {}).items()
                   if (k2 := resolve_key(k, ncfg or pcfg, mine)) in own} if isinstance(c.get("off"), Mapping) else {}
            for k in own:
                if k not in off and k in pcfg:
                    off[k] = pcfg[k]
            off = {k: x for k, x in off.items() if _allowed(ctx, k)}
            off = {k: v for k, x in off.items()          # the value in the type of the config key it replaces
                   if (v := typed_like(x, ncfg.get(k, pcfg.get(k)))) is not None}
            if not off:
                made[str(i)] = "-: no off value (a new key without a declared off)"
                continue
            if len(off) == 1 and next(iter(off)) in iso:
                made[str(i)] = f"-: its only key is already isolated by {iso[next(iter(off))]}"
                continue
            sha, err = F.build_point(adapter, sn.commit, off, [F.Knob(k, None, x, "lin") for k, x in off.items()], {},
                                     f"loo-{nid}-{i}", getattr(adapter, "config_rel", "config.yaml"))
            if not sha or sha == sn.commit:
                made[str(i)] = f"-: not built ({err or 'no change'})"
                continue
            made[str(i)] = _child(store, sn, sha, now, f"loo{i}", f"ladder={nid} loo={i}", off)
            left -= 1
            store.set_state(K_LOO, book)
        store.set_state(K_LOO, book)
        store.log("program_loo", node_id=nid, children=made)
        acts.append(f"program leave-one-out {nid}: {made}")
    return acts


# ---------------------------------------------------------------------------------------------- operator-program arms
# v3-c10 (operator 10-21: a node bundled both DINO-forcing arms into ONE node and lost -.0157; nobody knows which arm
# hurt -- the a node sampler/patch-embed bundling again): a program with >= 2 FULL-RETRAIN arms gets ONE research session
# per arm (:func:`units`), each told to build exactly that arm; its node is tagged ``op=<program> arm=<arm>`` like the
# eval-only arms and K_ARMS books its id; intake refuses a node touching another arm's ``keys`` (:func:`cross_arm`).
ARM_REQ = "program_arm:"                   # CandidateRequest.arm of a per-arm session: program_arm:<program id>:<arm id>


def full_arms(prog: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The program's FULL-RETRAIN arms (``lane: full``, with an id), in declared order."""
    return [a for a in prog.get("arms") or () if isinstance(a, Mapping) and str(a.get("lane") or "") == "full"
            and str(a.get("id") or "").strip()]


def units(prog: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any] | None]]:
    """The program's research-session units as (book key, arm): one per full arm (key ``<program>/<arm>``) when it has
    two or more, else the legacy single unit (key ``<program>``, no arm: the one session builds the first stage)."""
    pid, fa = str(prog.get("id") or ""), full_arms(prog)
    return [(f"{pid}/{a['id']}", a) for a in fa] if len(fa) >= 2 else [(pid, None)]


def arm_keys(arm: Mapping[str, Any]) -> tuple[str, ...]:
    """The config keys / code identifiers the arm owns (``keys``: substrings, e.g. ``init_align``)."""
    return tuple(str(k) for k in arm.get("keys") or () if str(k).strip())


def foreign_keys(prog: Mapping[str, Any], arm: Mapping[str, Any]) -> tuple[str, ...]:
    """Keys of the program's OTHER full arms that this arm does not itself own (a stack arm owns every arm's)."""
    if arm.get("after") or arm.get("stack"):
        return ()
    own = set(arm_keys(arm))
    return tuple(dict.fromkeys(k for a in full_arms(prog) if a.get("id") != arm.get("id")
                               for k in arm_keys(a) if k not in own))


def cross_arm(prog: Mapping[str, Any], arm: Mapping[str, Any], config_keys: Sequence[str], added: str) -> str:
    """'' when a per-arm node stays inside its arm, else why not: a config key or an added code line carrying another
    arm's ``keys`` (the intake rule ``cross_arm_bundling``)."""
    for k in foreign_keys(prog, arm):
        if (hit := next((c for c in config_keys if k in c), "")):
            return f"config key `{hit}` belongs to another arm of {prog.get('id')} (`{k}`); build only arm {arm.get('id')}"
        if (line := next((l.strip() for l in added.splitlines() if k in l), "")):
            return (f"added code `{line[:100]}` carries another arm's `{k}` (program {prog.get('id')}, you are arm "
                    f"{arm.get('id')}); each arm is its own node: revert it")
    return ""


def ported_problem(arm: Mapping[str, Any], writable: Sequence[str] | None = None) -> str:
    """'' when ``arm`` is a well-formed PORTED arm (``lane: ported``, no LLM session): a prebuilt code port -- ``code_ref``
    (the commit holding the files), ``code_files`` (repo paths taken from it) and ``config`` ({dotted key: value} set on the
    incumbent's config); optional ``after_done`` (arm ids, any program: build only once their nodes are terminal or have 3 finished seeds), ``code_base`` (the commit the port was cut from: the arm is refused once the incumbent's
    copies of the files no longer match it), ``stage`` and ``after`` (arm ids that must be booked first). With ``writable``
    (``surface.writable``) every ``code_files`` entry must be on that surface (v3 review S3; the lock post-check alone
    would let a ported arm rewrite any unlocked file)."""
    aid = arm.get("id")
    if not str(arm.get("code_ref") or "").strip():
        return f"arm {aid}: a ported arm needs a code_ref (the commit holding its files)"
    files = arm.get("code_files")
    if not isinstance(files, Sequence) or isinstance(files, str) or not files \
            or not all(isinstance(f, str) and f.strip() for f in files):
        return f"arm {aid}: a ported arm needs code_files (a list of repo paths)"
    if writable:
        from loop.select.intake import on_surface
        if off := [f for f in files if not on_surface(f, writable)]:
            return f"arm {aid}: code_files {off} are not on surface.writable ({list(writable)})"
    cfg = arm.get("config")
    if not isinstance(cfg, Mapping) or not cfg or not all(isinstance(k, str) and k.strip() for k in cfg):
        return f"arm {aid}: a ported arm needs config ({{dotted key: value}})"
    return ""


def arm_problem(arm: Mapping[str, Any], writable: Sequence[str] | None = None) -> str:
    """'' when ``arm`` is a well-formed EVAL-ONLY arm of an operator program (``ideation.operator_programs[*].arms``):
    an ``id``, ``lane: evalonly``, the ``file`` and the readout ``span`` ("file::Class.method") its LITERAL ``edits``
    ([{old, new}], each ``old`` exactly once inside the span) touch, and NO config change (``config`` / a train-time
    buffer fit is a full-retrain arm: ``lane: full``, built by the program's research session instead)."""
    if not isinstance(arm, Mapping) or not str(arm.get("id") or "").strip():
        return "an arm needs an id"
    if str(arm.get("lane") or "") == "ported":
        return ported_problem(arm, writable)
    if str(arm.get("lane") or "") != "evalonly":
        return f"arm {arm.get('id')}: lane {arm.get('lane')!r} is not evalonly (a full-retrain arm is the session's)"
    if arm.get("config"):
        return f"arm {arm['id']}: an eval-only arm changes no config value (make it lane: full)"
    span, file = str(arm.get("span") or ""), str(arm.get("file") or "")
    if not file or not span.startswith(file + "::"):
        return f"arm {arm['id']}: span {span!r} must lie in file {file!r}"
    edits = arm.get("edits") or ()
    if not edits or not all(isinstance(e, Mapping) and str(e.get("old") or "") and "new" in e for e in edits):
        return f"arm {arm['id']}: needs literal edits [{{old, new}}]"
    return ""


def mirror_edit(old: str, new: str) -> str | None:
    """``new`` with each numeric literal that differs from ``old``'s mirrored around it (:func:`mirror`: ``x0.5`` for a
    ``x2`` move); None unless the two differ ONLY in numeric literals (same text around them, same count)."""
    if _NUM.sub("#", old) != _NUM.sub("#", new):
        return None
    a, b = _NUM.findall(old), _NUM.findall(new)
    if len(a) != len(b) or a == b:
        return None
    out, parts = [], _NUM.split(old)
    for x, y in zip(a, b):
        if x == y:
            out.append(x)
            continue
        m = mirror(float(x), float(y))
        if m is None:
            return None
        out.append(str(int(m)) if m.is_integer() and "." not in x and "e" not in x.lower() else repr(m))
    return "".join(p + (out[i] if i < len(out) else "") for i, p in enumerate(parts))


def opposite_arm(arm: Mapping[str, Any]) -> dict[str, Any] | None:
    """The arm moving the OTHER way (v3-c8a B1): its declared ``opposite`` ({edits, note[, id]}: literal edits of the
    same span), else every edit's numeric literals mirrored (:func:`mirror_edit`); id ``<arm id>_opp`` unless declared.
    None when neither exists (a non-numeric change with no declared opposite)."""
    base = {k: v for k, v in arm.items() if k not in ("opposite", "edits", "note", "id")}
    aid = str(arm.get("id") or "")
    if isinstance(opp := arm.get("opposite"), Mapping) and opp.get("edits"):
        return {**base, "id": str(opp.get("id") or aid + OPP_SUFFIX), "edits": list(opp["edits"]),
                "note": str(opp.get("note") or f"the opposite direction of {aid}"), "opposite_of": aid}
    edits = []
    for e in arm.get("edits") or ():
        if not isinstance(e, Mapping) or (m := mirror_edit(str(e.get("old") or ""), str(e.get("new") or ""))) is None:
            return None
        edits.append({"old": e["old"], "new": m})
    return {**base, "id": aid + OPP_SUFFIX, "edits": edits, "note": f"the opposite direction of {aid} (mirrored "
            f"numeric literals)", "opposite_of": aid} if edits else None


def apply_edits(source: str, span: str, edits: Sequence[Mapping[str, Any]]) -> tuple[str | None, str]:
    """(``source`` with each edit applied INSIDE ``span`` only, '') or (None, why): an ``old`` must occur exactly
    once in the span's lines (never anywhere else in the file is touched)."""
    from loop.adapters.nanopath_v2 import _span
    got = _span(source, span)
    if got is None:
        return None, f"{span} not found"
    i, j, _ = got
    lines = source.splitlines(keepends=True)
    block = "".join(lines[i:j + 1])
    for e in edits:
        old, new = str(e["old"]), str(e["new"])
        if (k := block.count(old)) != 1:
            return None, f"`{old[:60]}` occurs {k}x inside {span} (needs exactly 1)"
        block = block.replace(old, new)
    return "".join(lines[:i]) + block + "".join(lines[j + 1:]), ""


AFTER_DONE_SEEDS = 3


def after_done_ready(store: Any, book: Mapping[str, Any], ids: Sequence[Any]) -> bool:
    """10-22 ops review: an arm's ``after_done`` (arm ids, looked up in EVERY program's booked arms): True once each listed
    arm's node is terminal (closed in the store) or has >= ``AFTER_DONE_SEEDS`` finished (non-eval) runs; an arm given up
    (booked ``-:``) counts as done, an arm not booked yet does not. ``after`` only waits for the BOOKING, which is one
    tick, so it cannot order the builds of two programs."""
    from loop.contracts import CLOSED_STATUSES, RunStatus
    from loop.evalonly import is_eval_run
    closed = {s.value for s in CLOSED_STATUSES}
    for aid in ids:
        recs = [mine[str(aid)] for mine in book.values() if isinstance(mine, Mapping) and str(aid) in mine]
        if not recs:
            return False
        for rec in recs:
            if isinstance(rec, str) and rec.startswith("-"):
                continue
            node = store.get_node(rec) if isinstance(rec, str) else None
            if node is None:
                return False
            if getattr(node.status, "value", str(node.status)) in closed:
                continue
            if sum(1 for r in store.runs(node_id=node.id, status=[RunStatus.DONE]) if not is_eval_run(r)) \
                    < AFTER_DONE_SEEDS:
                return False
    return True


def arms(ctx: Any, store: Any, adapter: Any, inc: Any, now: float) -> list[str]:
    """v3-c6r O1: each EVAL-ONLY arm of an operator program (:func:`arm_problem`) materialised ONCE as its own QUEUED
    node on the incumbent: its literal edits applied inside the readout span, committed, and kept only when the
    adapter's eval-only check (``adapter.eval_only``: every change inside the readout-only spans, no config change)
    passes -- the eval-only lane then scores it on trained checkpoints. A full-retrain arm (``lane: full``) is left
    to the program's research session. Returns the tick's action lines."""
    from loop.core.levers import opt
    progs = [p for p in opt(ctx, "operator_programs") or () if isinstance(p, Mapping) and p.get("arms")]
    if not progs or inc is None or not inc.commit:
        return []
    if not all(hasattr(adapter, a) for a in ("prepare_worktree", "commit", "reap_worktree")) \
            and not all(hasattr(adapter, a) for a in ("commit_files", "file_at")):
        return []
    book, acts = dict(store.get_state(K_ARMS) or {}), []
    both = bool(opt(ctx, "evalonly_both_directions"))
    for p in progs:
        pid = str(p.get("id") or "")
        mine = dict(book.get(pid) or {})
        todo = [a for a in p.get("arms") or () if isinstance(a, Mapping)]
        for arm in list(todo):                            # v3-c8a B1: each eval-only arm's OPPOSITE direction
            if str(arm.get("lane") or "") != "evalonly" or (opp := opposite_arm(arm)) is None:
                continue
            if any(str(a.get("id") or "") == opp["id"] for a in todo) or isinstance(mine.get(opp["id"]), str):
                continue                                  # the program declares it itself, or built / given up
            if isinstance(mine.get(opp["id"]), Mapping):  # a build that raised: retried like any arm
                todo.append(opp)
                continue
            done = mine.get(str(arm.get("id") or ""))
            why = harmed(store, None, done) if isinstance(done, str) and not done.startswith("-") else ""
            if both or why:
                todo.append(opp)
                if why:
                    acts.append(f"program arm {pid}/{arm.get('id')}: measured harmful ({why}); queueing its "
                                f"opposite {opp['id']}")
        for arm in todo:
            aid = str((arm or {}).get("id") or "")
            lane = str((arm or {}).get("lane") or "")
            if lane not in ("evalonly", "ported"):
                continue                                  # a full-retrain arm: the program session builds it
            rec = mine.get(aid)
            if isinstance(rec, str):
                continue                                  # built, or given up with a reason
            if lane == "ported" and arm.get("after") and not all(isinstance(mine.get(str(a)), str) for a in arm["after"]):
                continue                                  # waits for those arms to be booked (a stage letter is only a label)
            if arm.get("after_done") and not after_done_ready(store, book, arm["after_done"]):
                continue                                  # waits for those arms' nodes to be terminal / screened (3 seeds)
            if why := arm_problem(arm, tuple(getattr(getattr(getattr(ctx, "spec", None), "surface", None), "writable", ()) or ())):
                mine[aid] = f"-: {why}"
            else:
                try:
                    mine[aid] = (_build_ported_arm if lane == "ported" else _build_arm)(ctx, store, adapter, inc, p, arm, now)
                except Exception as exc:  # noqa: BLE001 - retried on later ticks, bounded
                    n = int((rec or {}).get("fails", 0)) + 1 if isinstance(rec, Mapping) else 1
                    mine[aid] = {"fails": n} if n < ARM_FAILS else f"-: build raised {n}x: {exc!r}"[:300]
            book[pid] = mine
            store.set_state(K_ARMS, book)
            r = mine[aid]
            store.log("program_arm", program=pid, arm=aid, result=r)
            acts.append(f"program arm {pid}/{aid}: " + (f"{'ported' if lane == 'ported' else 'eval-only'} node {r}"
                                                         if isinstance(r, str) and not r.startswith("-") else str(r)))
    return acts


def _build_arm(ctx: Any, store: Any, adapter: Any, inc: Any, prog: Mapping[str, Any], arm: Mapping[str, Any],
               now: float) -> str:
    from pathlib import Path

    from loop.contracts import HypothesisKind, HypothesisRecord, Node as StoreNode, NodeStatus, Operator, Strategy
    pid, aid, rel = str(prog["id"]), str(arm["id"]), str(arm["file"])
    msg = f"loop operator program {pid} arm {aid}: {str(arm.get('note') or '')[:200]}"
    if hasattr(adapter, "commit_files") and hasattr(adapter, "file_at"):      # v3-c8a B5: no worktree
        src = adapter.file_at(inc.commit, rel)
        if src is None:
            return f"-: {rel} not found at {inc.commit[:12]}"
        new, why = apply_edits(src, str(arm["span"]), list(arm["edits"]))
        if new is None:
            return f"-: {why}"
        sha = adapter.commit_files(inc.commit, {rel: new}, message=msg)
    else:
        wt = adapter.prepare_worktree(parent_commit=inc.commit, name=f"arm-{aid}"[:60])
        try:
            path = Path(wt.path) / rel
            new, why = apply_edits(path.read_text(), str(arm["span"]), list(arm["edits"]))
            if new is None:
                return f"-: {why}"
            path.write_text(new)
            sha = adapter.commit(wt, message=msg)
        finally:
            adapter.reap_worktree(wt)
    if not sha or sha == inc.commit:
        return "-: no change"
    if hasattr(adapter, "dry_build"):            # shadow (tick.DryRunAdapter): a fake commit, the edits were checked
        pass
    elif hasattr(adapter, "changed_files") and list(adapter.changed_files(inc.commit, sha)) != [rel]:
        return f"-: the commit changes more than {rel}"
    if hasattr(adapter, "eval_only") and not hasattr(adapter, "dry_build"):
        ok, why, _ = adapter.eval_only(inc.commit, sha)
        if not ok:
            return f"-: not eval-only eligible ({why})"[:300]
    nid, tag = _new_id(store), re.sub(r"[^A-Za-z0-9]+", "_", f"{pid}_{aid}").strip("_")[:80]
    fam = str(arm.get("family") or "eval_readout")
    claim = f"Operator program {pid}, eval-only arm {aid}: {str(arm.get('note') or '')}".strip()
    hid = f"h-{nid}"
    if store.get_hypothesis(hid) is None:
        store.put_hypothesis(HypothesisRecord(id=hid, task_id=inc.task_id, kind=HypothesisKind.TRIAL, claim=claim[:1000],
                                              mechanism_tag=tag, family=fam, predictions={},
                                              falsification="its own paired increment on the eval-only lane"))
    store.put_node(StoreNode(id=nid, task_id=inc.task_id, commit=sha, parent_id=inc.id, strategy=Strategy.EXPLORE,
                             operator=Operator.DRAFT, family=fam, mechanism_tag=tag,
                             diff_hash=hashlib.sha256(f"{inc.commit}\n{sha}".encode()).hexdigest(),
                             status=NodeStatus.QUEUED, hypothesis_id=hid, created_ts=now, updated_ts=now,
                             notes=f"brief=program op={pid} arm={aid} stage={arm.get('stage') or 'A'} program_arm=1"
                             + (f" opposite_of={arm['opposite_of']}" if arm.get("opposite_of") else "")))
    store.set_state(f"config_diff:{nid}", {})
    return nid


def _build_ported_arm(ctx: Any, store: Any, adapter: Any, inc: Any, prog: Mapping[str, Any], arm: Mapping[str, Any],
                      now: float) -> str:
    """A ``lane: ported`` arm: the incumbent's commit with ``code_files`` taken from ``code_ref`` and the arm's ``config``
    keys set in its YAML (:func:`loop.agents.researcher.set_yaml_value`), committed without a worktree, refused when it
    breaks the lock surface (or the files' base drifted from ``code_base``), then noded like :func:`_build_arm`: QUEUED on
    the incumbent, notes ``... program_arm=1 arch_arm=1`` (the structural ladder tunes it before it can close)."""
    from loop.agents.researcher import set_yaml_value
    from loop.contracts import HypothesisKind, HypothesisRecord, Node as StoreNode, NodeStatus, Operator, Strategy
    if not all(hasattr(adapter, a) for a in ("commit_files", "file_at")):
        return "-: the adapter cannot commit files without a worktree"
    pid, aid, ref = str(prog["id"]), str(arm["id"]), str(arm["code_ref"])
    rels = [str(r) for r in arm["code_files"]]
    base = str(arm.get("code_base") or "")
    files: dict[str, str] = {}
    for r in rels:
        text = adapter.file_at(ref, r)
        if text is None:
            return f"-: {r} missing at code_ref {ref[:12]}"
        if base and adapter.file_at(inc.commit, r) != adapter.file_at(base, r):
            return f"-: code_ref base drifted: {r} at the incumbent {inc.commit[:12]} differs from code_base {base[:12]}"
        files[r] = text
    cfg_rel = str(getattr(adapter, "config_rel", "config.yaml"))
    text = adapter.file_at(inc.commit, cfg_rel)
    if text is None:
        return f"-: {cfg_rel} not found at {inc.commit[:12]}"
    try:
        for k, v in arm["config"].items():
            text = set_yaml_value(text, str(k), v)
    except ValueError as exc:
        return f"-: config: {exc}"[:300]
    files[cfg_rel] = text
    sha = adapter.commit_files(inc.commit, files, message=f"loop operator program {pid} arm {aid}: port of {ref[:12]} "
                                                          f"({', '.join(rels)}) + {dict(arm['config'])}"[:400])
    if not sha or sha == inc.commit:
        return "-: no change"
    if hasattr(adapter, "lock_violations") and (bad := list(adapter.lock_violations(sha))):
        return f"-: lock violation: {'; '.join(str(b) for b in bad)[:200]}"
    nid, tag = _new_id(store), re.sub(r"[^A-Za-z0-9]+", "_", f"{pid}_{aid}").strip("_")[:80]
    fam = str(arm.get("family") or "architecture")
    claim = f"Operator program {pid}, ported arm {aid}: {str(arm.get('note') or '')} config {dict(arm['config'])}".strip()
    hid = f"h-{nid}"
    if store.get_hypothesis(hid) is None:
        store.put_hypothesis(HypothesisRecord(id=hid, task_id=inc.task_id, kind=HypothesisKind.TRIAL, claim=claim[:1000],
                                              mechanism_tag=tag, family=fam, predictions={},
                                              falsification="its own paired increment after its lr x layer-decay tuning ladder"))
    store.put_node(StoreNode(id=nid, task_id=inc.task_id, commit=sha, parent_id=inc.id, strategy=Strategy.EXPLORE,
                             operator=Operator.DRAFT, family=fam, mechanism_tag=tag,
                             diff_hash=hashlib.sha256(f"{inc.commit}\n{sha}".encode()).hexdigest(),
                             status=NodeStatus.QUEUED, hypothesis_id=hid, created_ts=now, updated_ts=now,
                             notes=f"brief=program op={pid} arm={aid} stage={arm.get('stage') or 'A'} program_arm=1 arch_arm=1"))
    store.set_state(f"config_diff:{nid}", dict(arm["config"]))
    return nid


# ---------------------------------------------------------------------------------------------- program shadows explore
# 10-21: four sessions overlapped on project / site sampler flattening (a node, a node, a node, a node) while the operator
# program's own arms covered it. While an operator-program arm is in flight, an explore / literature proposal NEAR its
# mechanism is rejected ``shadowed_by_program``: the arm decides that mechanism on its own paired increment.
SHADOW_BRIEFS = ("explore", "literature")
DECIDED = frozenset({"screened_out", "not_confirmed", "confirmed", "audit_failed", "vetoed", "eval_passed", "baseline"})
KNOB_KEY = re.compile(r"\bkey:\s*([A-Za-z_][\w.]*)")
GENERIC = frozenset({"level", "head", "loss", "train", "model", "token", "block", "layer", "weight", "scale", "cls", "patch",
                     "batch", "tile", "image", "stage", "rate", "norm", "input", "output", "global", "local", "crop"})


#: plain-English / boilerplate words of an arm's note and a program brief ('dose', 'knob', 'default', 'only', 'never', ...): they
#: say nothing about the mechanism, so two of them shared with a proposal's tag must not shadow it (replay 10-21 over the
#: explore / literature nodes: 'loosen_grad_clip_major_dose' matched an arm on 'dose' + 'grad')
FILLER = frozenset({"dose", "knob", "default", "grad", "change", "code", "count", "only", "never", "from", "that", "when", "same",
                    "step", "last", "taken", "instead", "through", "every", "enter", "initial", "added", "pass", "plus", "final",
                    "full", "were", "between", "session", "program", "eval", "retrain", "source", "measur", "arms", "ladder",
                    "null", "operato", "type", "remov", "with", "this", "each", "than", "then", "into", "over", "also",
                    "mechani", "stack", "arm", "keys", "value", "values", "inject", "select", "other", "their", "which",
                    "dino", "student", "feature", "predict"})   # ... and the repo's own namespace: every tag says dino / student


def arm_node_ids(store: Any, pid: str, aid: str, nodes: Sequence[Any] | None = None) -> list[str]:
    """Nodes tagged ``op=<pid> arm=<aid>`` (an eval-only arm's node, a full arm's session node). ``nodes``: the store's
    nodes already read (a check_idea call asks for every arm: one read, not one per arm)."""
    rx = re.compile(rf"(?<!\S)op={re.escape(pid)}(?!\S).*?(?<!\S)arm={re.escape(aid)}(?!\S)")
    return [n.id for n in (store.nodes() if nodes is None else nodes) if rx.search(n.notes or "")]


def arm_stems(arm: Mapping[str, Any]) -> set[str]:
    """The distinctive content stems of an arm (its id, note, components, ladder, ``keys``): what a proposal's tag is
    compared with. Generic stems (:data:`GENERIC`) and words under 4 characters never count."""
    from loop.novelty import mech_terms
    text = " ".join([str(arm.get("id") or ""), str(arm.get("note") or ""), str(arm.get("ladder") or ""),
                     *[str(c) for c in arm.get("components") or ()], *arm_keys(arm)])
    return {w for w in mech_terms("", text) if len(w) >= 4 and w not in GENERIC and w not in FILLER}


def arm_knobs(arm: Mapping[str, Any]) -> tuple[str, ...]:
    """Config keys the arm owns: the knob keys its components name (``key: dino.init_align_weight``) + its ``keys``."""
    out = [m.group(1) for c in arm.get("components") or () for m in KNOB_KEY.finditer(str(c))]
    return tuple(dict.fromkeys([*out, *arm_keys(arm)]))


def inflight_arms(ctx: Any, store: Any, statuses: Mapping[str, Any] | None = None, now: float = 0.0) -> list[dict[str, Any]]:
    """The operator-program arms still IN FLIGHT: of every non-superseded program, each non-stack arm without a DECIDED
    node (screened out / confirmed / ...; a queued / screening / rejected-and-retried / not-yet-built arm is open),
    younger than ``alloc.shadow_max_h`` from its first node (or its booking, else now). [{program, arm, knobs, stems,
    nodes, cluster_members}]."""
    from loop.core.alloc import cfg_of
    from loop.core.levers import opt
    c = cfg_of(ctx)
    if not getattr(c, "shadow", True):
        return []
    now = now or __import__("time").time()
    booked = store.get_state("ideation.operator_programs") or {}
    snodes = store.nodes()
    by_id = {n.id: n for n in snodes}
    out = []
    for p in opt(ctx, "operator_programs") or ():
        if not isinstance(p, Mapping) or not p.get("id") or p.get("superseded_by"):
            continue
        pid = str(p["id"])
        for arm in p.get("arms") or ():
            if not isinstance(arm, Mapping) or not str(arm.get("id") or "").strip() or arm.get("stack") \
                    or arm.get("after"):
                continue
            aid = str(arm["id"])
            ids = arm_node_ids(store, pid, aid, snodes)
            st = [getattr(getattr(by_id.get(n), "status", None), "value", "") for n in ids]
            if ids and all(x in DECIDED for x in st):
                continue                                   # decided on its own paired increment
            t0 = min([float(by_id[n].created_ts or now) for n in ids]
                     + [float((booked.get(f"{pid}/{aid}") or booked.get(pid) or {}).get("ts") or now)])
            if float(c.shadow_max_h) > 0 and now - t0 > float(c.shadow_max_h) * 3600.0:
                continue                                   # open too long: it no longer shadows (it may never run)
            out.append({"program": pid, "arm": aid, "knobs": arm_knobs(arm), "stems": arm_stems(arm), "nodes": ids})
    return out


def shadow_rule(ctx: Any, store: Any, family: str, tag: str, text: str = "", config_keys: Sequence[str] = (),
                now: float = 0.0) -> tuple[str, str]:
    """('shadowed_by_program', fix) when an explore / literature proposal (family, mechanism_tag, ``text``, the config
    keys of its diff) is NEAR an in-flight operator-program arm: it changes a config key the arm owns (an arm knob
    is a substring of a diff key), falls in the same mechanism cluster as the arm's node (``alloc.assign``), or its
    tag shares >= ``alloc.shadow_min_terms`` distinctive stems with the arm's description. LLM-free. The caller exempts
    repair / port sessions and the program arms' own sessions."""
    from loop.core.alloc import cfg_of
    c = cfg_of(ctx)
    if not getattr(c, "shadow", True):
        return "", ""
    arms = inflight_arms(ctx, store, None, now)
    if not arms:
        return "", ""
    from loop.novelty import mech_terms
    mine = {w for w in mech_terms("", tag or "") if len(w) >= 4 and w not in GENERIC and w not in FILLER}
    keys = [str(k) for k in config_keys]
    cl_members: set[str] = set()
    try:
        from loop.core import alloc as AL
        if (cl := AL.assign(AL._terms(family, tag, text or tag), tau=float(c.cluster_tau))) is not None:
            cl_members = set(cl.members)
    except Exception:  # noqa: BLE001 - no cluster verdict: the other two signals stand
        pass
    for a in arms:
        why = ""
        if (hit := next((k for k in keys for kn in a["knobs"] if kn and kn in k), "")):
            why = f"it changes config key `{hit}`, which the arm owns"
        elif cl_members and cl_members & set(a["nodes"]):
            why = f"it falls in the same mechanism cluster as the arm's node {sorted(cl_members & set(a['nodes']))[0]}"
        elif len(shared := sorted(mine & a["stems"])) >= int(c.shadow_min_terms):
            why = f"its mechanism_tag shares {', '.join(shared)} with the arm's description"
        if why:
            return "shadowed_by_program", (
                f"operator program {a['program']} arm {a['arm']} is in flight on this mechanism ({why}). The arm "
                f"decides it on its own paired increment (its session builds it and its dose ladder); a second "
                f"proposal on the same mechanism only duplicates the screen. Bring a different mechanism; it can be "
                f"proposed again once the arm is decided (screened out / confirmed) or after {float(c.shadow_max_h):g} h")
    return "", ""
