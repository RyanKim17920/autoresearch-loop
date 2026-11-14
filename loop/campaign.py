"""loop.campaign — DIFFICULT-TUNE CAMPAIGNS (operator 2026-10-07: "why didn't we do some more testing on moonshots?
... give 'difficult-tune-moonshots' a lot more flexibility"). An init / architecture swap, a new objective or loss
(or any node whose researcher declared ``campaign: true``) is screened at its parent's hyperparameters on <= 2 seeds
and closed (live 72 h: 189 / 212 closed moonshots were judged on <= 2 runs; the init swaps a node / a node at the
incumbent's LR). Such a change usually needs its OWN optimiser settings, so a closed one gets ONE campaign: a computed
hyperparameter search, no LLM call per round (the researcher is often paused).

* the SEARCH SPACE: the researcher's declared ``search_space`` (knobs with ranges and a scale), else the default
  (:data:`DEFAULT_SPACE`: LR on a log scale, warmup, weight decay, layer decay, teacher momentum, batch size /
  gradient accumulation) -- only knobs present and settable in the node's config;
* INPUT-SIDE changes (:func:`input_side`: an init swap, a data-sampling / augmentation mechanism, a node of / a changed
  key in a ``campaign.input_families`` family, a changed diff line matching ``campaign.input_code``) also search
  ``campaign.input_space`` on top of ``max_knobs`` (the lower-layer LR throttles: a new input is fit by the lowest
  blocks, which layer-wise LR decay throttles);
* ROUND 1: a spread of max(``campaign.round1_points``, 2 x knobs + 2) points (at most ``round1_max``; the budget
  scales with it) -- the first knob log-spaced over its range (LR x1/4..x4
  around the parent's value), the others coordinate-wise at their low / high -- each a config point on the node's
  commit, ONE seed (``seed1=``), alternating over the node's own screen seeds and paired against the node's PARENT
  run there (its control: no new control run);
* ROUNDS 2..``rounds``: successive halving -- the best shrunk configs (2, then 1) get their next seed, the round-1
  winners are combined, and the best is refined between its value and its nearest tried neighbours on each knob
  (at the EDGE of what was tried: one step beyond it, within the knob's hard bounds -- loop.extrapolate);
  every config's delta vs the parent is SHRUNK (loop.variants' positive-part James-Stein) -- no new statistics;
* FOLLOW-UP: at a round boundary, knobs the node is eligible for now but its space lacks (``input_space`` added after
  it started, a knob new in the config) join it with ONE extra round at their low / high around the current best
  config (:func:`_followup`, logged ``campaign_followup``; it does not use up a round);
* no futility cut and no harm veto on a point (points are design points: screen only); a crashed / void point is
  simply void, an OOM one is rebuilt ONCE with the batch halved (and gradient accumulation doubled where such a knob
  exists); a node that itself OOMed gets that fix on every point;
* budget ``campaign.budget_gpu_h`` per campaign (a round is trimmed to what is left), ``campaign.max_concurrent``
  campaigns at once; points never count toward ``bandit.moonshot_max_inflight`` (design points, not queued moonshots);
* the END: the best config graduates only when its shrunk delta vs the parent is > 0 AND its raw delta beats the
  ``ideation.null_q`` quantile of the max of k null deltas (loop.variants.null_max: selection-aware); the graduate
  (``tune_of=`` / ``campaign_of=``) is judged ONLY on fresh CONFIRM seeds (``promotion.confirm_max_seeds``, within the
  5-seed cap) vs its parent through the normal confirm path -- and must also beat the incumbent (loop.seeds.step);
  the harm veto applies there. A summary line is logged (``campaign_done``).

New closes start campaigns automatically (those closed after the lane first ran); older ones only through the
backfill's campaign epoch (loop.backfill, capped). ``campaign.incumbent`` entries put a config-only node at their
``levels`` on each new incumbent with a declared campaign over their ``search_space`` (:func:`seed_incumbent`)."""

from __future__ import annotations

import dataclasses
import json
import math
import re
import statistics
import time
from typing import Any, Mapping, Sequence

from loop import design as D
from loop import tune as MT
from loop import variants as VR
from loop.contracts import (CLOSED_STATUSES, PANEL, ROOT_NODE_ID, ArmPull, Node, NodeStatus, RunStatus, Stage,
                            Strategy, design_point, is_oom)

K_CAMP = "campaign"         # {"seq", "since", "queue": [{"base", "why"}], "current": {base: cur}, "done": [cur]}
CLOSES = (NodeStatus.SCREENED_OUT, NodeStatus.NOT_CONFIRMED, NodeStatus.VETOED, NodeStatus.CRASHED)
#: (config-key leaf regex, scale, range factor): the default search space, in priority order
DEFAULT_SPACE = ((r"lr", "log", 4.0), (r"warmup\w*", "log", 2.0), (r"weight_decay\w*|wd", "log", 2.0),
                 (r"layer\w*decay", "unit", 2.0), (r"\w*(momentum|ema)\w*", "unit", 2.0),
                 (r"batch_size", "log", 2.0), (r"grad\w*accum\w*|accum\w*steps", "log", 2.0))
_SKIP = re.compile(r"(?:^|\s)(?:tune_of|tune|legs|legs_of|campaign_of)=")
_IDX = re.compile(r"(.+)\[(\d+)\]")        # ``data.global_crop_scale[0]``: one element of a list value
INPUT_TYPES = ("data_sampling", "augmentation")     # loop.novelty mechanism types that change what the network sees
K_INC = "campaign.incumbent"  # {"<tag>@<incumbent>": node id | "-: why"}: campaign.incumbent entries already seeded


def book(store: Any) -> dict[str, Any]:
    return {"seq": 0, "since": 0.0, "queue": [], "current": {}, "done": [], **(store.get_state(K_CAMP) or {})}


def _note(node: Node, key: str) -> str:
    m = re.search(rf"(?:^|\s){key}=(\S+)", node.notes or "")
    return m.group(1) if m else ""


def oom_crashed(store: Any, node: Node) -> bool:
    return any(r.status is RunStatus.CRASH and is_oom(r.log_tail) for r in store.runs(node_id=node.id))


def why(ctx: Any, store: Any, node: Node) -> str:
    """Why ``node`` is a difficult-to-tune change ('' = it is not): the researcher declared a campaign, an init swap,
    or a code change whose mechanism is a new objective / loss or an architecture change."""
    if node.id == ROOT_NODE_ID or design_point(node) or _SKIP.search(node.notes or ""):
        return ""
    if (store.get_state(f"ideation:{node.id}") or {}).get("campaign"):
        return "declared campaign"
    if init := _note(node, "init"):
        return f"init swap ({init.rsplit('/', 1)[-1][:48]})"
    from loop.novelty import node_type
    t = node_type(node)
    bold = node.strategy in (Strategy.MOONSHOT, Strategy.EXPLORE) or bool(_note(node, "bank"))   # not a cross / tweak
    if bold and t in MT.MAJOR_TYPES and MT.is_major(ctx, store, node):
        return "new objective / loss" if t == "loss_objective" else "architecture change"
    return ""


def input_side(ctx: Any, store: Any, node: Node) -> str:
    """Why ``node`` changes the network's INPUT or its lowest layers ('' = it does not): an init swap, an input-side
    mechanism type (:data:`INPUT_TYPES`), its family or a changed config key's family in ``campaign.input_families``,
    or a changed diff line matching ``campaign.input_code`` (``ctx`` None: the first two only)."""
    from loop.novelty import node_type
    from loop.select.intake import family_of
    cc = getattr(getattr(ctx, "cfg", None), "campaign", None)
    fams = set(getattr(cc, "input_families", ()))
    if _note(node, "init"):
        return "init swap"
    if (t := node_type(node)) in INPUT_TYPES:
        return f"mechanism type {t}"
    if node.family in fams:
        return f"family {node.family}"
    if fams and (k := next((k for k in store.get_state(f"config_diff:{node.id}") or {}
                            if family_of(ctx, k) in fams), "")):
        return f"changes {k}"
    if code := getattr(cc, "input_code", ""):
        try:
            text = open(node.diff_path).read() if node.diff_path else ""
        except OSError:
            text = ""
        if re.search(rf"^[+-](?![+-]).*(?:{code})", text, re.M):
            return "input-side code"
    return ""


def rank(reason: str) -> int:
    """Priority of a campaign reason (lower first): declared, init swap, then objective / architecture."""
    reason = re.sub(r"^backfill epoch \d+: ", "", reason)          # a queued reason carries the backfill's prefix
    return 0 if reason.startswith("declared") else 1 if reason.startswith("init swap") else 2


def campaigned(store: Any, bk: Mapping[str, Any] | None = None) -> set[str]:
    bk = book(store) if bk is None else bk
    return {*bk["current"], *(c["base"] for c in bk["done"]), *(q["base"] for q in bk["queue"])}


def claims(ctx: Any, store: Any, node: Node) -> bool:
    """The campaign lane owns ``node`` (loop.tune's pass and dose ladder leave it alone): it has a campaign, or it is
    a difficult-to-tune change closed since the lane first ran."""
    cc = getattr(ctx.cfg, "campaign", None)
    if cc is None or not cc.enabled:
        return False
    from loop.core import bold as BOLD
    if BOLD.is_bold(node):                 # v3-c8b: a bold node's tuning is loop.tune's, within ideation.bold_tune_max
        return False
    bk = book(store)
    return node.id in campaigned(store, bk) or (
        bool(bk["since"]) and node.updated_ts >= bk["since"] and bool(why(ctx, store, node))
        and startable(ctx, store, node))


def enqueue(store: Any, node_id: str, reason: str) -> bool:
    """Queue ``node_id`` for a campaign (the backfill's retro epoch); False when it already has one."""
    bk = book(store)
    if node_id in campaigned(store, bk):
        return False
    bk["queue"] = bk["queue"] + [{"base": node_id, "why": reason}]
    store.set_state(K_CAMP, bk)
    store.log("campaign_queued", node_id=node_id, why=reason)
    return True


def seeds_of(ctx: Any, store: Any, node: Node) -> list[int]:
    """The node's own screen seeds (its parent's control runs are there): the campaign's points run on them."""
    return sorted({int(r.seed) for r in store.runs(node_id=node.id) if r.stage is Stage.SCREEN})


def startable(ctx: Any, store: Any, node: Node | None) -> bool:
    return node is not None and node.status in CLOSES and bool(seeds_of(ctx, store, node)) \
        and (node.status is not NodeStatus.CRASHED or oom_crashed(store, node))


def pending(ctx: Any, store: Any, snapshot: Any, bk: Mapping[str, Any]) -> list[tuple[Node, str]]:
    """Campaigns to start, in order: the queue (retro epoch), then difficult-to-tune nodes closed since the lane
    first ran, most promising (E[max(own gain, 0)]) first. Never a node a tuning pass / ladder is running on."""
    busy = {c["base"] for c in MT.book(store)["current"].values()}
    done = {*bk["current"], *(c["base"] for c in bk["done"])}
    from loop.core import bold as BOLD     # v3-c8b: never a bold node (loop.tune tunes it, within bold_tune_max)
    out = [(n, q["why"]) for q in sorted(bk["queue"], key=lambda q: rank(q["why"]))      # init swaps first (stable)
           if (n := store.get_node(q["base"])) is not None and n.id not in done | busy and startable(ctx, store, n)
           and not BOLD.is_bold(n)]
    auto = []
    for n in store.nodes(status=list(CLOSES)) if bk["since"] else ():
        if n.updated_ts >= bk["since"] and n.id not in done | busy | {x.id for x, _ in out} and not BOLD.is_bold(n) \
                and (w := why(ctx, store, n)) and startable(ctx, store, n):
            post = (getattr(snapshot, "nodes", None) or {}).get(n.id)
            auto.append((rank(w), -(getattr(post, "gain_plus", None) or 0.0), n.created_ts, n.id, n, w))
    return out + [(n, w) for *_, n, w in sorted(auto, key=lambda x: x[:4])]


# ------------------------------------------------------------------------------------ the search space
def value(base: Mapping[str, Any], key: str) -> Any:
    """``base[key]``; an index knob ``key[i]`` reads element i of a list value (None: absent)."""
    if (m := _IDX.fullmatch(key)) and isinstance(v := base.get(m.group(1)), list) and int(m.group(2)) < len(v):
        return v[int(m.group(2))]
    return base.get(key)


def expand(base: Mapping[str, Any], levels: Mapping[str, Any]) -> dict[str, Any]:
    """``levels`` with every index knob folded into its list (the other elements: ``base``'s)."""
    out: dict[str, Any] = {}
    for k, v in levels.items():
        if m := _IDX.fullmatch(k):
            lst = list(out.get(m.group(1), base[m.group(1)]))
            lst[int(m.group(2))] = v
            out[m.group(1)] = lst
        else:
            out[k] = v
    return out


def _fmt(x: float, like: Any) -> Any:
    return int(round(x)) if isinstance(like, int) else float(f"{x:.4g}")


def _to(kb: Mapping[str, Any], v: float) -> float:
    return math.log(v) if kb["scale"] == "log" else math.log(1.0 - v) if kb["scale"] == "unit" else float(v)


def _from(kb: Mapping[str, Any], u: float) -> Any:
    x = math.exp(u) if kb["scale"] == "log" else 1.0 - math.exp(u) if kb["scale"] == "unit" else u
    return _fmt(x, kb["value"])


def _knob(key: str, v: Any, lo: Any, hi: Any, scale: str, declared: bool) -> dict[str, Any] | None:
    num = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool)    # noqa: E731
    if scale == "unit" and not (num(v) and 0 < v < 1 and 0 <= lo < hi < 1):
        scale = "log"
    if not (num(v) and num(lo) and num(hi) and lo < hi) or scale == "log" and (lo <= 0 or v <= 0):
        return None
    return {"key": key, "value": v, "low": _fmt(lo, v), "high": _fmt(hi, v), "scale": scale, "declared": declared}


def resolve(rows: Sequence[Mapping[str, Any]], base: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Declared range rows against ``base``: {key, low, high, scale} or {key, factor, scale} (value / f .. value x f);
    without ``low``, half the upward span below the value in the knob's scale. Keys ``base`` lacks (or at / above a
    ``high`` with no ``low``) are dropped."""
    num = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool)    # noqa: E731
    out = []
    for r in rows:
        v, sc = value(base, k := str(r.get("key") or "")), str(r.get("scale") or "log")
        if not num(v):
            continue
        lo, hi = (v / float(r["factor"]), v * float(r["factor"])) if r.get("factor") else (r.get("low"), r.get("high"))
        if lo is None and num(hi) and v < hi and (sc != "log" or v > 0):
            lo = v * math.sqrt(v / hi) if sc == "log" else v - (hi - v) / 2
        if num(lo) and num(hi) and lo < hi:
            out.append({"key": k, "low": _fmt(lo, v), "high": _fmt(hi, v), "scale": sc})
    return out


def space(ctx: Any, store: Any, node: Node, text: str, base: Mapping[str, Any],
          dropped: list[dict[str, str]] | None = None) -> list[dict[str, Any]]:
    """The campaign's knobs: the researcher's declared ``search_space`` ({key, low, high, scale: log | linear}),
    else :data:`DEFAULT_SPACE` (log range value / f .. value x f; ``unit``: 1 - value scaled likewise) over keys of the
    node's config; only settable ones (design._knob_ok on each range end that MOVES the value: a node whose value
    sits at a range end keeps the knob -- live c8: a node at layer decay 1.0 / patch-embed mult 1.0 lost both declared
    knobs, its range end equal to its value read as "no move"), at most ``campaign.max_knobs``; an :func:`input_side`
    node also gets ``campaign.input_space`` (:func:`resolve`; it replaces a knob on the same key). ``dropped``
    collects each declared / input knob left out, with the reason."""
    drop = dropped if dropped is not None else []
    out: list[dict[str, Any]] = []
    for r in (store.get_state(f"ideation:{node.id}") or {}).get("search_space") or ():
        if not isinstance(r, Mapping):
            continue
        if value(base, k := str(r.get("key") or "")) is None:
            drop.append({"key": k, "why": "not in the node's config"})
            continue
        kb = _knob(k, value(base, k), r.get("low"), r.get("high"), str(r.get("scale") or "log"), True)
        if kb is None:
            drop.append({"key": k, "why": f"invalid range {r.get('low')}..{r.get('high')} at {value(base, k)}"})
        out.append(kb or {})
    if not [kb for kb in out if kb]:
        out = []
        for pat, scale, f in DEFAULT_SPACE:
            k = next((k for k in sorted(base) if re.fullmatch(pat, k.rsplit(".", 1)[-1])
                      and not any(kb["key"] == k for kb in out if kb)), "")
            v = base.get(k)
            if k and isinstance(v, (int, float)) and not isinstance(v, bool) and v:
                d = 1.0 - v
                out.append((_knob(k, v, 1.0 - d * f, 1.0 - d / f, "unit", False) if scale == "unit" and 0 < v < 1
                            and d * f < 1 else None) or _knob(k, v, v / f, v * f, "log", False) or {})
    def ok(kb: Mapping[str, Any]) -> bool:
        ends = [x for x in (kb["low"], kb["high"]) if x != kb["value"]]      # the node's own value is no move
        good = bool(ends) and all(D._knob_ok(ctx, text, base, {"key": k, "low": base[k], "high": v})
                                  for x in ends for k, v in expand(base, {kb["key"]: x}).items())
        if not good and kb.get("declared"):
            drop.append({"key": kb["key"], "why": "not settable in the node's config"})
        return good

    settable = lambda kbs: [kb for kb in kbs if kb and ok(kb)]                                   # noqa: E731
    extra = settable([_knob(r["key"], value(base, r["key"]), r["low"], r["high"], r["scale"], True)
                      for r in resolve(ctx.cfg.campaign.input_space, base)] if input_side(ctx, store, node) else [])
    keys = {kb["key"] for kb in extra}
    own = [kb for kb in settable(out) if kb["key"] not in keys]
    drop += [{"key": kb["key"], "why": f"over campaign.max_knobs ({ctx.cfg.campaign.max_knobs})"}
             for kb in own[ctx.cfg.campaign.max_knobs:] if kb.get("declared")]
    return own[:ctx.cfg.campaign.max_knobs] + extra


def spread(kb: Mapping[str, Any], n: int) -> list[Any]:
    """``n`` levels evenly spaced over the knob's range in its scale (the node's own value dropped; a value AT a range
    end spreads n + 1 levels, so n still move it: 1.0 over 0.8..1.0 -> 0.8, 0.9)."""
    a, b = _to(kb, kb["low"]), _to(kb, kb["high"])
    n += kb["value"] in (kb["low"], kb["high"])
    lv = [_from(kb, a + (b - a) * i / (n - 1)) for i in range(n)] if n > 1 else [kb["high"]]
    return [x for x in dict.fromkeys(lv) if x != kb["value"]]


def r1_size(ctx: Any, knobs: Sequence[Any]) -> int:
    """Round 1's size: ``round1_points``, or 2 per knob + 2 when that is more (every added knob gets its low / high),
    at most ``round1_max``."""
    cc = ctx.cfg.campaign
    return max(cc.round1_points, min(cc.round1_max, 2 * len(knobs) + 2))


def round1(ctx: Any, knobs: Sequence[Mapping[str, Any]], seeds: Sequence[int]) -> list[dict[str, Any]]:
    """The first knob at up to 4 log-spaced levels, then every other knob at its low / high, one at a time, until
    :func:`r1_size`; point i runs on seed i mod len(seeds)."""
    n, specs = r1_size(ctx, knobs), []
    for j, kb in enumerate(knobs):
        for x in spread(kb, min(4, n) if j == 0 else 2):
            if len(specs) < n:
                specs.append({"levels": {kb["key"]: x}})
    return [{**s, "seed": seeds[i % len(seeds)]} for i, s in enumerate(specs)]


def _between(kb: Mapping[str, Any], x: Any, tried: Sequence[Any], ctx: Any = None) -> list[Any]:
    """Midpoints (in the knob's scale) between ``x`` and its nearest tried neighbour on each side; at an EDGE of the
    tried values (the best point at its range's edge) one full step further out, within the knob's hard bounds
    (loop.extrapolate.inside; without ``ctx`` a declared range is never left)."""
    u, ts = _to(kb, x), sorted({_to(kb, t) for t in tried} - {_to(kb, x)})
    lo, hi = [t for t in ts if t < u], [t for t in ts if t > u]
    out = [(u + lo[-1]) / 2] if lo else []
    out += [(u + hi[0]) / 2] if hi else []
    if (lo or hi) and not (lo and hi):
        out.append(u + (u - (lo[-1] if lo else hi[0])))
    vals = [_from(kb, w) for w in out]
    if ctx is not None:
        from loop.extrapolate import inside
        vals = [v for v in vals if inside(ctx, kb["key"], v, kb["value"])]
    elif kb["declared"]:
        vals = [v for v in vals if min(kb["low"], kb["high"]) <= v <= max(kb["low"], kb["high"])]
    return [v for v in dict.fromkeys(vals) if v != x and (kb["scale"] == "linear" or v > 0)]


def oom_fix(ctx: Any, text: str, base: Mapping[str, Any]) -> dict[str, Any]:
    """An OOM's config fix: the batch size halved (gradient accumulation doubled where such a key exists); {} when
    the config has no settable batch-size key."""
    out: dict[str, Any] = {}
    for k in sorted(base):
        leaf, v = k.rsplit(".", 1)[-1], base[k]
        if not isinstance(v, int) or isinstance(v, bool):
            continue
        if leaf == "batch_size" and v >= 2 and not any(x.endswith("batch_size") for x in out):
            out[k] = v // 2
        elif re.fullmatch(r"grad\w*accum\w*|accum\w*steps", leaf) and v >= 1:
            out[k] = v * 2
    out = {k: v for k, v in out.items() if D._knob_ok(ctx, text, base, {"key": k, "low": base[k], "high": v})}
    return out if any(k.endswith("batch_size") for k in out) else {}


def label(levels: Mapping[str, Any]) -> str:
    return ",".join(f"{k}={v}" for k, v in sorted(levels.items())) or "1x"


# ------------------------------------------------------------------------------------------- rounds
def tick(ctx: Any, store: Any, adapter: Any, snapshot: Any, *, new_id: Any, now: float | None = None,
         scarce: bool = False) -> list[str]:
    """Advance every running campaign; start queued / newly closed ones while fewer than ``max_concurrent`` run
    (none in scarce mode)."""
    cc = getattr(ctx.cfg, "campaign", None)
    if cc is None or not cc.enabled or snapshot is None:
        return []
    now = time.time() if now is None else now
    bk, acts = book(store), []
    if not bk["since"]:
        bk["since"] = now
        store.set_state(K_CAMP, bk)
    if not scarce:
        try:
            acts += seed_incumbent(ctx, store, adapter, snapshot, new_id, now)
        except Exception as exc:  # noqa: BLE001 -- retried next tick, never the loop
            store.log("campaign_error", node_id="incumbent", error=repr(exc)[:500])
        bk = book(store)
    for base in list(bk["current"]):
        cur = bk["current"][base]
        try:
            acts += _advance(ctx, store, adapter, snapshot, bk, cur, new_id, now)
        except Exception as exc:  # noqa: BLE001 -- a git / adapter failure closes the campaign, never the loop
            _close(store, bk, cur, now, why=f"error: {exc!r}"[:300], tuned="")
            store.log("campaign_error", node_id=base, error=repr(exc)[:500])
            acts.append(f"campaign {cur['id']} for {base} closed on an error: {exc!r}"[:300])
        store.set_state(K_CAMP, bk)
    for node, reason in [] if scarce else pending(ctx, store, snapshot, bk)[:max(0, cc.max_concurrent - len(bk["current"]))]:
        bk["queue"] = [q for q in bk["queue"] if q["base"] != node.id]
        try:
            acts += _start(ctx, store, adapter, snapshot, bk, node, reason, new_id, now)
        except Exception as exc:  # noqa: BLE001
            store.log("campaign_error", node_id=node.id, error=repr(exc)[:500])
            acts.append(f"campaign for {node.id} could not start: {exc!r}"[:300])
            if (cur := bk["current"].get(node.id)) is not None:
                _close(store, bk, cur, now, why=f"error: {exc!r}"[:300], tuned="")
        store.set_state(K_CAMP, bk)
    return acts


def seed_incumbent(ctx: Any, store: Any, adapter: Any, snapshot: Any, new_id: Any, now: float) -> list[str]:
    """Each ``campaign.incumbent`` entry once per incumbent: a config-only node on it at the entry's ``levels`` (it
    screens like any node) whose declared campaign over the entry's resolved ``search_space`` is queued (it runs
    once the node closes; a node that passes goes on through promote / confirm itself)."""
    from loop.contracts import AxisPrediction, HypothesisKind, HypothesisRecord, Operator
    from loop.select.intake import family_of
    inc =getattr(snapshot, "incumbent_id", None) or ROOT_NODE_ID
    done, acts = dict(store.get_state(K_INC) or {}), []
    for e in ctx.cfg.campaign.incumbent:
        if (slot := f"{e['tag']}@{inc}") in done:
            continue
        with D._Tree(ctx, store, adapter, f"camp-inc-{e['tag']}", base_id=inc) as t:
            levels = {k: v for k, v in e["levels"].items() if k in t.base}
            sp = resolve(e.get("search_space") or (), {**t.base, **levels})
            fam = next((f for k in levels if (f := family_of(ctx, k))), "")
            hyp = HypothesisRecord(
                id="", task_id=ctx.task_id, kind=HypothesisKind.TRIAL, family=fam, mechanism_tag=e["tag"],
                claim=f"the incumbent {inc} at {label(levels)}; a campaign over {[r['key'] for r in sp]} tunes it "
                      f"before it is judged", predictions={PANEL: AxisPrediction(1, 1e-3)}, created_ts=now,
                falsification=f"no tuned config at {label(levels)} beats {inc} on fresh confirm seeds")
            nid = t.node(levels, knobs=[{"key": k, "low": t.base[k], "high": v, "levers": {}}
                                        for k, v in levels.items()], tag=e["tag"], operator=Operator.DRAFT,
                         new_id=new_id, hyp=hyp, notes="brief=incumbent_campaign")
        if not nid or nid == inc:
            done[slot] = "-: " + ("rejected at intake" if not nid else "= the incumbent's config")
        else:
            store.update_node(nid, family=fam or "optimization")
            store.set_state(f"ideation:{nid}", {"campaign": True, "search_space": sp})
            enqueue(store, nid, f"declared campaign: {e['tag']} on the incumbent {inc}")
            done[slot] = nid
            acts.append(f"campaign: {e['tag']} node {nid} on the incumbent {inc} at {label(levels)}, declared campaign "
                        f"over {[r['key'] for r in sp]}")
        store.set_state(K_INC, done)
        store.log("campaign_incumbent", tag=e["tag"], incumbent=inc, node_id=done[slot])
    return acts


def gpu_h(store: Any, cur: Mapping[str, Any]) -> float:
    return sum(r.gpu_h for nid in cur["points"] for r in store.runs(node_id=nid))


def _est(ctx: Any, store: Any, cur: Mapping[str, Any]) -> float:
    """GPU-h one point is expected to cost: its finished points', else the node's own screen runs', else 1."""
    xs = [r.gpu_h for nid in cur["points"] for r in store.runs(node_id=nid) if r.status is RunStatus.DONE]
    xs = xs or [r.gpu_h for r in store.runs(node_id=cur["base"], stage=Stage.SCREEN) if r.status is RunStatus.DONE]
    return max(statistics.fmean(xs) if xs else 1.0, 1e-3)


def budget(ctx: Any, cur: Mapping[str, Any]) -> float:
    """``campaign.budget_gpu_h`` scaled by round 1's size over ``round1_points`` (more knobs, more budget)."""
    cc = ctx.cfg.campaign
    return cc.budget_gpu_h * max(1.0, r1_size(ctx, cur["space"]) / cc.round1_points)


def _room(ctx: Any, store: Any, cur: Mapping[str, Any]) -> int:
    return max(0, int((budget(ctx, cur) - gpu_h(store, cur)) // _est(ctx, store, cur)))


def _start(ctx: Any, store: Any, adapter: Any, snapshot: Any, bk: dict, node: Node, reason: str, new_id: Any,
           now: float) -> list[str]:
    bk["seq"] += 1
    cid, seeds = f"c{bk['seq']}", seeds_of(ctx, store, node)
    plan = ctx.cfg.seeds
    if clash := set(seeds) & {int(s) for s in (*plan.promote, *plan.confirm_pool, *plan.audit)}:
        raise ValueError(f"campaign seeds {sorted(clash)} overlap promote/confirm/audit seeds: no fresh-seed verdict")
    cur = {"id": cid, "base": node.id, "parent": node.parent_id or ROOT_NODE_ID, "why": reason, "seeds": seeds,
           "round": 1, "space": [], "fix": {}, "points": {}, "oom": [], "started": now}
    bk["current"][node.id] = cur
    with D._Tree(ctx, store, adapter, f"camp-{cid}-r1", base_id=node.id) as t:
        cur["space"] = space(ctx, store, node, t.text, t.base, dropped := [])
        for d in dropped:                            # never silently: each left-out knob, with its reason
            store.log("campaign_knob_dropped", campaign=cid, node_id=node.id, key=d["key"], why=d["why"])
        if node.status is NodeStatus.CRASHED:
            cur["fix"] = oom_fix(ctx, t.text, t.base)
        specs = round1(ctx, cur["space"], seeds)[:_room(ctx, store, cur)] \
            if node.status is not NodeStatus.CRASHED or cur["fix"] else []
        made = _build(ctx, store, t, cur, specs, new_id)
    keys = [kb["key"] for kb in cur["space"]]
    store.log("campaign_started", node_id=node.id, campaign=cid, why=reason, space=cur["space"], fix=cur["fix"],
              points=dict(cur["points"]), seeds=seeds)
    if not made:
        _close(store, bk, cur, now, why="no point admitted" + ("" if keys else " (no settable knob)"), tuned="")
        return [f"campaign {cid} for {node.id}: no point admitted (knobs {keys or 'none'})"]
    return [f"campaign {cid} for {node.id} ({node.mechanism_tag}; {reason}): round 1, {len(made)} point(s) over "
            f"{keys} on its seeds {seeds}, 1 seed each, paired vs {cur['parent']}"
            + (f"; OOM fix {cur['fix']} on every point" if cur["fix"] else "")
            + f"; budget {budget(ctx, cur):g} GPU-h"
            + ("; dropped " + ", ".join("{key} ({why})".format(**d) for d in dropped) if dropped else "")]


def _build(ctx: Any, store: Any, t: Any, cur: dict, specs: Sequence[Mapping[str, Any]], new_id: Any) -> list[str]:
    """Config points on the node's commit (``t``), parented on the node's PARENT, each on its one seed; a ``copy``
    spec re-runs an existing point's config on another seed (a fresh node: intake would call it a duplicate)."""
    made, tried = [], {(p["cfg"], p["seed"]) for p in cur["points"].values()}
    for spec in specs:
        if "copy" in spec:
            src, p = store.get_node(spec["copy"]), cur["points"][spec["copy"]]
            if (p["cfg"], spec["seed"]) in tried or src is None:
                continue
            nid = new_id()
            store.put_node(dataclasses.replace(
                src, id=nid, status=NodeStatus.SCREENING, reject_reason="", created_ts=time.time(),
                updated_ts=time.time(), notes=re.sub(r"\bseed1=\S+", f"seed1={spec['seed']}", src.notes or "")))
            store.set_state(f"config_diff:{nid}", dict(store.get_state(f"config_diff:{src.id}") or {}))
            store.log("node_transition", node_id=nid, frm="new", to=NodeStatus.SCREENING.value,
                      reason=f"campaign {cur['id']}: {p['cfg']} on seed {spec['seed']}")
            levels = p["levels"]
        else:
            levels = {**cur["fix"], **spec["levels"]}
            if (label(levels), spec["seed"]) in tried:
                continue
            real = expand(t.base, levels)
            nid = t.node(real, knobs=[{"key": k, "low": t.base[k], "high": v, "levers": {}} for k, v in real.items()],
                         tag=f"{cur['id']}pt{len(cur['points']) + 1:02d}", operator=None, new_id=new_id,
                         notes=f"tune={cur['id']} campaign_of={cur['base']} seed1={spec['seed']}")
            if not nid or nid == cur["base"]:
                store.log("campaign_point_skipped", campaign=cur["id"], levels=levels,
                          why="rejected at intake" if not nid else "= the node")
                continue
            store.update_node(nid, parent_id=cur["parent"])
        cur["points"][nid] = {"cfg": label(levels), "levels": dict(levels), "seed": int(spec["seed"]),
                              "round": cur["round"]}
        tried.add((label(levels), spec["seed"]))
        made.append(nid)
    return made


def rows(ctx: Any, store: Any, snapshot: Any, cur: Mapping[str, Any]) -> dict[str, list[float]]:
    """config label -> [mean delta vs the parent, se, n] over its scored points (the node itself: ``1x``)."""
    sig = float(snapshot.sigma.get(PANEL, ctx.cfg.panel_sd)) if snapshot is not None else ctx.cfg.panel_sd
    by: dict[str, list[float]] = {}
    for nid, p in cur["points"].items():
        by.setdefault(p["cfg"], []).extend(MT._vs_control(ctx, store, nid, cur["parent"]))
    if not cur["fix"] and (ds := MT._vs_control(ctx, store, cur["base"], cur["parent"])):
        by["1x"] = ds
    return {k: [statistics.fmean(v), sig / math.sqrt(len(v)), float(len(v))] for k, v in by.items() if v}


def configs(cur: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {"1x": {}, **{p["cfg"]: {k: v for k, v in p["levels"].items() if k not in cur["fix"]}
                         for p in cur["points"].values()}}


def refine(ctx: Any, cur: Mapping[str, Any], eff: Mapping[str, Mapping[str, float]]) -> list[dict[str, Any]]:
    """The next round's specs (at most ``campaign.refine_points``): the best 2 (then 1) configs on their next seed,
    after round 1 the combination of each knob's best single-knob level that beat the node, then midpoints around
    the best config on each knob."""
    cfgs, out = configs(cur), []
    ranked = sorted(eff, key=lambda k: (-eff[k]["shrunk"], k))
    for k in ranked[:2 if cur["round"] == 1 else 1]:
        have = {p["seed"] for p in cur["points"].values() if p["cfg"] == k}
        src = next((nid for nid, p in cur["points"].items() if p["cfg"] == k), "")
        if src and (nxt := next((s for s in cur["seeds"] if s not in have), None)) is not None:
            out.append({"copy": src, "seed": nxt})
    bar = eff["1x"]["shrunk"] if "1x" in eff else 0.0
    if cur["round"] == 1:
        combo = {}
        for kb in cur["space"]:
            one = [k for k in ranked if set(cfgs.get(k, {})) == {kb["key"]} and eff[k]["shrunk"] > bar]
            if one:
                combo[kb["key"]] = cfgs[one[0]][kb["key"]]
        if len(combo) >= 2:
            out.append({"levels": combo})
    best, own = (cfgs.get(ranked[0], {}) if ranked else {}), {kb["key"]: kb["value"] for kb in cur["space"]}
    for kb in cur["space"]:
        tried = [kb["value"], *(c[kb["key"]] for c in cfgs.values() if kb["key"] in c)]
        for x in _between(kb, best.get(kb["key"], kb["value"]), tried, ctx):
            out.append({"levels": {k: v for k, v in {**best, kb["key"]: x}.items() if v != own.get(k)}})
    seen, specs = set(cfgs) - {"1x"}, []
    for s in out:
        if "levels" in s and (not s["levels"] or label({**cur["fix"], **s["levels"]}) in seen):
            continue
        seen.add(label({**cur["fix"], **s["levels"]}) if "levels" in s else f"copy:{s['copy']}")
        specs.append(s)
    n = ctx.cfg.campaign.refine_points
    return [{**s, "seed": s.get("seed", cur["seeds"][i % len(cur["seeds"])])} for i, s in enumerate(specs[:n])]


def _retry_oom(ctx: Any, store: Any, adapter: Any, cur: dict, new_id: Any) -> list[str]:
    """A point that crashed out of memory is rebuilt ONCE with the OOM fix (its config, the same seed)."""
    bad = [nid for nid in cur["points"] if nid not in cur["oom"] and (n := store.get_node(nid)) is not None
           and n.status is NodeStatus.CRASHED and oom_crashed(store, n)]
    if not bad:
        return []
    cur["oom"] = cur["oom"] + bad
    with D._Tree(ctx, store, adapter, f"camp-{cur['id']}-oom", base_id=cur["base"]) as t:
        specs = [{"levels": {**lv, **fix}, "seed": cur["points"][nid]["seed"]} for nid in bad
                 for lv in [cur["points"][nid]["levels"]] if (fix := oom_fix(ctx, t.text, {**t.base, **lv}))]
        made = _build(ctx, store, t, cur, specs, new_id)
    cur["oom"] = cur["oom"] + made                    # once: a rebuilt point is never rebuilt again
    store.log("campaign_oom", campaign=cur["id"], node_id=cur["base"], crashed=bad, made=made)
    return [f"campaign {cur['id']} for {cur['base']}: {len(bad)} point(s) out of memory -> "
            + (f"rebuilt with the batch halved: {made}" if made else "void (no batch-size knob to halve)")]


def missing(ctx: Any, store: Any, cur: Mapping[str, Any], text: str, base: Mapping[str, Any]) -> list[dict]:
    """Knobs the campaign's node is eligible for NOW (:func:`space` on its config: ``campaign.input_space`` for an
    input-side node, a knob added to the config since the campaign started) that its space lacks."""
    have = {kb["key"] for kb in cur["space"]}
    return [kb for kb in space(ctx, store, store.get_node(cur["base"]), text, base) if kb["key"] not in have]


def _followup(ctx: Any, store: Any, adapter: Any, snapshot: Any, cur: dict, new_id: Any) -> list[str]:
    """At a round boundary, :func:`missing` knobs join the space and ONE follow-up round sweeps each at its low /
    high around the current best config (2 points per knob; the budget grows with the space, :func:`budget`) before
    the rounds go on / the graduate's fresh-confirm judgment."""
    with D._Tree(ctx, store, adapter, f"camp-{cur['id']}-fu", base_id=cur["base"]) as t:
        if not (new := missing(ctx, store, cur, t.text, t.base)):
            return []
        eff = VR._shrunk(rs) if (rs := rows(ctx, store, snapshot, cur)) else {}
        best = max(eff, key=lambda k: (eff[k]["shrunk"], k)) if eff else "1x"
        lv = configs(cur).get(best, {})
        cur["space"] = cur["space"] + new
        specs = [{"levels": {**lv, kb["key"]: x}} for kb in new for x in spread(kb, 2)]
        specs = [{**sp, "seed": cur["seeds"][i % len(cur["seeds"])]} for i, sp in enumerate(specs)]
        made = _build(ctx, store, t, cur, specs[:_room(ctx, store, cur)], new_id)
    keys = [kb["key"] for kb in new]
    cur["followup"] = cur.get("followup", []) + keys
    store.log("campaign_followup", campaign=cur["id"], node_id=cur["base"], knobs=keys, best=best, made=made)
    return [f"campaign {cur['id']} for {cur['base']}: follow-up round over {keys} (missing from its space) around its "
            f"best {best}: {len(made)} point(s); budget {budget(ctx, cur):g} GPU-h"] if made else []


def _advance(ctx: Any, store: Any, adapter: Any, snapshot: Any, bk: dict, cur: dict, new_id: Any,
             now: float) -> list[str]:
    if any(n is None or n.status not in CLOSED_STATUSES for n in map(store.get_node, cur["points"])):
        return []
    if acts := _retry_oom(ctx, store, adapter, cur, new_id):
        return acts
    if acts := _followup(ctx, store, adapter, snapshot, cur, new_id):
        return acts
    rs = rows(ctx, store, snapshot, cur)
    eff = VR._shrunk(rs) if rs else {}
    if cur["round"] < ctx.cfg.campaign.rounds and eff and (room := _room(ctx, store, cur)) > 0 \
            and (specs := refine(ctx, cur, eff)[:room]):
        cur["round"] += 1
        with D._Tree(ctx, store, adapter, f"camp-{cur['id']}-r{cur['round']}", base_id=cur["base"]) as t:
            made = _build(ctx, store, t, cur, specs, new_id)
        best = max(eff, key=lambda k: (eff[k]["shrunk"], k))
        store.log("campaign_round", campaign=cur["id"], node_id=cur["base"], round=cur["round"], made=made,
                  rows={k: [*r, eff[k]["shrunk"]] for k, r in rs.items()})
        if made:
            return [f"campaign {cur['id']} for {cur['base']}: round {cur['round'] - 1} scored {len(rs)} config(s), "
                    f"best {best} shrunk {eff[best]['shrunk']:+.4f} vs {cur['parent']} -> round {cur['round']}: "
                    f"{len(made)} point(s)"]
    return _finish(ctx, store, snapshot, bk, cur, rs, eff, new_id, now)


def _close(store: Any, bk: dict, cur: dict, now: float, **result: Any) -> None:
    bk["current"].pop(cur["base"], None)
    bk["done"] = bk["done"] + [{**cur, **result, "closed": now}]
    store.set_state(K_CAMP, bk)


def _finish(ctx: Any, store: Any, snapshot: Any, bk: dict, cur: dict, rs: Mapping[str, list[float]],
            eff: Mapping[str, Mapping[str, float]], new_id: Any, now: float) -> list[str]:
    """Graduate the best config when its shrunk delta vs the parent is > 0 and its raw delta beats the null max's
    ``ideation.null_q`` quantile, or its shrunk delta alone clears ``legs.shrunk_bar`` (the confirm MDE); the
    graduate goes straight to fresh confirm seeds."""
    spent, n_pts = gpu_h(store, cur), len(cur["points"])
    why, tuned, best, head = graduate(ctx, store, snapshot, cur, rs, eff, new_id, now)
    if not tuned:
        store.record_pull(ArmPull(task_id=ctx.task_id, arm=store.get_node(cur["base"]).strategy,
                                  node_id=f"{cur['base']}#campaign", reward=0.0, gpu_h=spent, ts=now))
    line = head + (f" {why}" if why.startswith("->") else f"; {why}")
    _close(store, bk, cur, now, why=why, tuned=tuned, best=best, gpu_h=spent, summary=line,
           rows={k: [*r, eff[k]["shrunk"]] for k, r in rs.items()})
    store.log("campaign_done", node_id=cur["base"], campaign=cur["id"], summary=line, tuned=tuned, best=best,
              rounds=cur["round"], points=n_pts, gpu_h=spent)
    return [line]


def graduate(ctx: Any, store: Any, snapshot: Any, cur: Mapping[str, Any], rs: Mapping[str, list[float]],
             eff: Mapping[str, Mapping[str, float]], new_id: Any, now: float, dry: bool = False
             ) -> tuple[str, str, str, str]:
    """The graduation judgment of a finished campaign (and its graduate's node unless ``dry``; ``tuned`` is then
    '(dry)'): (why, tuned, best, head)."""
    from loop import legs as LG
    from loop import seeds as S
    from loop.world.evalues import confirm_delta, sigma_ucb
    from loop.world.noise import STATE_KEY as NOISE_KEY
    base, q = store.get_node(cur["base"]), ctx.cfg.ideation.null_q
    cand = {k: e for k, e in eff.items() if k != "1x"}
    best = max(cand, key=lambda k: (cand[k]["shrunk"], k)) if cand else ""
    nq = VR.null_max([e["se"] for e in eff.values()], q, seed=len(cur["points"]))
    bar = LG.shrunk_bar(ctx, store, snapshot)
    spent, n_pts = gpu_h(store, cur), len(cur["points"])
    head = (f"campaign {cur['id']} {cur['base']}: {cur['round']} round(s), {n_pts} points, {spent:.1f} GPU-h, ")
    if not best:
        return "no point scored", "", best, head
    s, raw = eff[best]["shrunk"], rs[best][0]
    head += (f"best {best} shrunk {s:+.4f} vs {cur['parent']} (raw {raw:+.4f}, null q{q:g} {nq:+.4f}"
             + (f"; the node at its own hparams: shrunk {eff['1x']['shrunk']:+.4f}" if "1x" in eff else "") + ")")
    null_ok, shrunk_ok = s > 0.0 and raw > nq, s > 0.0 and s >= bar
    if not (null_ok or shrunk_ok):
        return "no config credibly beats the parent" + (f" (shrunk {s:+.4f} < bar {bar:.4f})" if bar < math.inf
                                                        else ""), "", best, head
    if "1x" in eff and eff["1x"]["shrunk"] >= s:
        return "the node's own hparams are the best: nothing to retune", "", best, head
    reason = "" if null_ok else f" (graduate on shrunk {s:+.4f} >= bar {bar:.4f}, null gate not passed)"
    if dry:
        return f"-> fresh confirm (dry){reason}", "(dry)", best, head
    pt = store.get_node(next(nid for nid, p in cur["points"].items() if p["cfg"] == best
                             and MT.screen_panels(ctx, store, nid)))
    tuned = MT._tuned_node(ctx, store, snapshot, base, pt, {"batch": cur["id"]}, s, new_id, now)
    row = (store.get_state(NOISE_KEY) or {}).get(PANEL)
    sig = sigma_ucb(row["sigma"], row["df_eff"]) if row else ctx.cfg.panel_sd
    n = S.planned_confirm(ctx.cfg, [])
    node = store.get_node(tuned)
    store.update_node(tuned, status=NodeStatus.CONFIRMING, strategy=base.strategy,
                      confirm_alt=confirm_delta(sig, ctx.cfg.promotion.alpha_confirm, n),
                      notes=f"{node.notes} campaign_of={base.id}")
    inc = getattr(snapshot, "incumbent_id", "") or ROOT_NODE_ID
    store.log("node_transition", node_id=tuned, frm=NodeStatus.SCREENING.value, to=NodeStatus.CONFIRMING.value,
              reason=f"campaign {cur['id']} graduate of {base.id}: judged only on {n} fresh confirm seeds")
    why = f"-> fresh confirm {tuned} on {n} seeds vs {cur['parent']}" + (
        f" and the incumbent {inc}" if inc != cur["parent"] else " (the incumbent)") + reason
    return why, tuned, best, head


def campaign_point(node: Node | None) -> bool:
    """A campaign's config point (``campaign_of=`` on a design point; its graduate is not one)."""
    return node is not None and node.strategy is Strategy.DESIGN and node.operator is None \
        and "campaign_of=" in (node.notes or "")


def stragglers(store: Any) -> set[str]:
    """A running campaign's open points once at most max(2, half its current round's points) are open: the round's
    analysis waits on every one of them, so the driver runs them HIGH (primary allowed, moved off a pending
    preemptible slot) like a leg factorial's stragglers (live 10-08: c1-c7's 80 points ran LOW on idle only, niced;
    round-3 points waited 5 h for a first submit, round-1 points were preempted 1-2x each, and no round closed for
    5 h -- c1's round 3 waited on its last point alone)."""
    out: set[str] = set()
    for cur in book(store)["current"].values():
        n_round = sum(p.get("round") == cur["round"] for p in cur["points"].values())
        open_ = [nid for nid in cur["points"]
                 if (n := store.get_node(nid)) is not None and n.status not in CLOSED_STATUSES]
        if open_ and len(open_) <= max(2, n_round // 2):
            out.update(open_)
    return out


# ------------------------------------------------------------------------------------------ visibility
def note(store: Any) -> str:
    """``campaigns=2(17pts) `` while campaigns run (the heartbeat), else ''."""
    cur = book(store)["current"]
    return f"campaigns={len(cur)}({sum(len(c['points']) for c in cur.values())}pts) " if cur else ""


def lines(store: Any) -> str:
    bk, out = book(store), []
    for c in bk["done"]:
        out.append(c.get("summary") or f"campaign {c['id']} {c['base']}: {c.get('why', '')}")
    for c in bk["current"].values():
        out.append(f"campaign {c['id']} {c['base']} running ({c['why']}): round {c['round']}, {len(c['points'])} "
                   f"point(s) over {[kb['key'] for kb in c['space']]}")
    if bk["queue"]:
        out.append(f"campaign queue: {', '.join(q['base'] for q in bk['queue'])}")
    return "\n".join(out)
