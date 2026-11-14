"""loop.selftune — bounded self-tuning of OPERATIONAL knobs (DESIGN §6.1).

A review may ask ``[{knob, value, reason}]``. A whitelisted knob is clamped to its hard bounds and
stored as a runtime override in store state (the spec file is untouched); the driver reads
``effective(spec cfg, overrides)`` every tick. Any other knob (budget, locks, eval, seeds cap, alphas,
labless, init-compat, SLURM, ...) is never tuned: it becomes an operator proposal instead.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from typing import Any, Mapping, Sequence

from loop.contracts import BANDIT_ARMS, Strategy

K_OVERRIDES = "selftune.overrides"      # {knob: {value, spec, reason, source, ts}}
K_ACKED = "selftune.acked"              # proposal ids the operator has dealt with (`loop ack`)
K_PENDING = "runbook.pending"           # run ids SLURM held PENDING at the driver's last poll (`status`)
K_EPOCH = "world.measurement_epoch"     # loop.contracts.K_EPOCH: {"epoch", "ts", ...} of the last measurement bump
K_VERDICT_SEEN = "backfill.seen"        # loop.backfill.K_SEEN: {verdict epoch: ts the loop first saw it}
# The version of the preemption metric definition (preempt_stats). Bump it whenever the definition changes: an
# override tuned on an older version expires (stale). v2: lost GPU-h credits resume-from-latest.pt and useful
# GPU-h credits runs that started on a preemptible pool and finished on primary (rv-35 / rv-36 cut
# slurm.preemptible_slots 32 -> 24 -> 16 citing the v1 108.6 / 124.9 lost GPU-h; v2: ~11% waste).
PREEMPT_METRIC_V = 2
METRIC_V: Mapping[str, int] = {"slurm.preemptible_slots": PREEMPT_METRIC_V}   # knob -> the metric it is tuned on
FLOOR_SUM_MAX = 0.9                     # summed arm floors: >= 10% of pulls stay Thompson argmax

BOUNDS: Mapping[str, tuple[float, float]] = {
    "bandit.floor": (0.05, 0.4),
    "bandit.explore_floor": (0.05, 0.4),
    "bandit.moonshot_floor": (0.05, 0.6),
    "bandit.moonshot_max_inflight": (1, 6),
    "bandit.crossover_min_p": (0.75, 0.95),
    "researcher.max_concurrent": (1, 24),   # operator 2026-10-04: scales with the 64-GPU cluster
    "slurm.preemptible_slots": (4, 56),     # operator 2026-10-04: the whole cluster (64 GPUs) when idle; 8 stay primary
    "review.stall_evals": (5, 40),
    "promotion.p_promote": (0.2, 0.9),
    "seeds.block_children": (2, 8),
    "design.stall_closes": (0, 60),
    "design.z": (1.5, 3.0),
    "scout.every_hours": (2.0, 24.0),              # the research scout's cadence (loop.scout): see its record
    # loop.meta: each LLM component's bias ('be more / less X' in its prompt): see the meta-learning scorecard
    **{f"meta.{c}_bias": (-1.0, 1.0) for c in ("researcher", "review",
                                                "first_principles", "design", "crossover", "scout", "persona")},
}
MAX_CONCURRENT = int(BOUNDS["researcher.max_concurrent"][1])
# knob -> the spec field holding the OPERATOR's minimum (operator focus beats self-tuning)
OPERATOR_MIN: Mapping[str, str] = {"bandit.moonshot_floor": "bandit.moonshot_floor_min",
                                   "bandit.moonshot_max_inflight": "bandit.moonshot_max_inflight_min"}


def _get(cfg: Any, knob: str) -> Any:
    if knob == "slurm.preemptible_slots" and cfg.slurm.preemptible_slots < 0:    # the declared total
        return sum(p.slots for p in cfg.pools if p.preemptible)
    for part in knob.split("."):
        cfg = getattr(cfg, part)
    return cfg


def _set(cfg: Any, knob: str, value: Any) -> Any:
    head, _, rest = knob.partition(".")
    return dataclasses.replace(cfg, **{head: _set(getattr(cfg, head), rest, value) if rest else value})


def floor_sum(cfg: Any) -> float:
    b = cfg.bandit
    own = {Strategy.EXPLORE: b.explore_floor, Strategy.MOONSHOT: b.moonshot_floor}
    return sum(b.floor if own.get(a) is None else own[a] for a in BANDIT_ARMS)


def operator_floors(cfg: Any, *, rescale: bool = True) -> Any:
    """Lift every knob to its operator minimum, then, if the arm floors now sum past FLOOR_SUM_MAX,
    scale the OTHER arms' floors down proportionally (the moonshot floor is the operator's). A spec with
    no operator minimum is returned unchanged."""
    if not cfg.bandit.moonshot_floor_min and not cfg.bandit.moonshot_max_inflight_min:
        return cfg
    for knob, mk in OPERATOR_MIN.items():
        lo = _get(cfg, mk)
        cur = cfg.bandit.floor if knob == "bandit.moonshot_floor" and cfg.bandit.moonshot_floor is None \
            else _get(cfg, knob)
        if lo and cur < lo:
            cfg = _set(cfg, knob, lo)
    total = floor_sum(cfg)
    if rescale and cfg.bandit.moonshot_floor_min and total > FLOOR_SUM_MAX + 1e-9:
        b = cfg.bandit
        moon = b.floor if b.moonshot_floor is None else b.moonshot_floor
        s = max(0.0, FLOOR_SUM_MAX - moon) / max(total - moon, 1e-12)
        cfg = dataclasses.replace(cfg, bandit=dataclasses.replace(
            b, floor=b.floor * s, moonshot_floor=moon,
            explore_floor=None if b.explore_floor is None else b.explore_floor * s))
    return cfg


def clamp(knob: str, value: Any) -> Any:
    """A stored override inside today's bounds (bounds may have tightened since it was stored)."""
    lo, hi = BOUNDS[knob]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    v = min(hi, max(lo, value))
    return int(round(v)) if isinstance(value, int) else float(v)


def stale(spec_cfg: Any, knob: str, o: Mapping[str, Any]) -> bool:
    """An override tuned against a spec value the operator has since changed is stale (ignored): the
    operator's new value wins. So is one tuned on an older definition of the metric it cites (:data:`METRIC_V`;
    none recorded = v1). A legacy override without its spec value is stale on an operator-floor knob."""
    if knob in METRIC_V and int(o.get("metric_v", 1)) != METRIC_V[knob]:
        return True
    if "spec" not in o:
        return knob in OPERATOR_MIN
    a, b = o["spec"], _get(spec_cfg, knob)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) > 1e-9 * max(1.0, abs(b))
    return a != b


def epoch_ts(store: Any) -> float:
    """When the latest measurement epoch (loop.world) or verdict epoch (loop.backfill) bump was first seen."""
    e, seen = store.get_state(K_EPOCH) or {}, store.get_state(K_VERDICT_SEEN) or {}
    return max([float(e.get("ts") or 0.0), *(float(v) for v in seen.values())])


def expired(o: Mapping[str, Any], since: float) -> bool:
    """An override tuned before the latest epoch bump expired: it was argued from numbers that bump redefined."""
    return float(o.get("ts") or 0.0) < since


def live(spec_cfg: Any, book: Mapping[str, Any], since: float = 0.0) -> dict[str, Any]:
    """knob -> override value, stale, expired (tuned before ``since``) and unknown knobs dropped."""
    return {k: o["value"] for k, o in sorted(book.items()) if k in BOUNDS and not stale(spec_cfg, k, o)
            and not expired(o, since)}


def effective(spec_cfg: Any, store: Any) -> Any:
    """The spec's LoopConfig with every live stored override applied, clamped ON READ to the bounds. Every
    override EXPIRES at a measurement / verdict epoch bump (:func:`epoch_ts`): reviews re-tune on the new numbers."""
    cfg = spec_cfg
    for knob, v in live(spec_cfg, store.get_state(K_OVERRIDES) or {}, epoch_ts(store)).items():
        cfg = _set(cfg, knob, clamp(knob, v))
    return operator_floors(cfg)


def apply(store: Any, spec_cfg: Any, items: Sequence[Mapping[str, Any]], *, source: str,
          now: float) -> tuple[list[str], list[str]]:
    """Clamp and store whitelisted changes; return (log lines, operator proposals for the rest)."""
    book, acts, proposals = dict(store.get_state(K_OVERRIDES) or {}), [], []
    for it in items:
        knob, value, reason = str(it.get("knob", "")), it.get("value"), str(it.get("reason", ""))[:300]
        if knob not in BOUNDS:
            proposals.append(f"set {knob} = {value!r}: {reason} (not self-tunable)")
            store.log("selftune_refused", knob=knob, value=value, reason=reason, source=source)
            continue
        lo, hi = BOUNDS[knob]
        spec_v = _get(spec_cfg, knob)
        try:
            v = min(hi, max(lo, float(value)))
        except (TypeError, ValueError):
            store.log("selftune_refused", knob=knob, value=value, reason="not a number", source=source)
            continue
        v = int(round(v)) if isinstance(spec_v, int) and not isinstance(spec_v, bool) else v
        if knob in OPERATOR_MIN:
            v = max(v, type(v)(_get(spec_cfg, OPERATOR_MIN[knob])))    # never below the operator's minimum
        before = after = spec_cfg
        cur = live(spec_cfg, book, epoch_ts(store))
        for k, x in cur.items():
            before = _set(before, k, x)
        for k, x in {**cur, knob: v}.items():
            after = _set(after, k, x)
        # refuse only a change that pushes the floors past the cap (an operator floor may already exceed
        # it; effective() then scales the other arms down)
        before, after = operator_floors(before, rescale=False), operator_floors(after, rescale=False)
        if floor_sum(after) > FLOOR_SUM_MAX + 1e-9 and floor_sum(after) > floor_sum(before) + 1e-9:
            store.log("selftune_refused", knob=knob, value=v, source=source,
                      reason=f"arm floors would sum to {floor_sum(after):.2f} > {FLOOR_SUM_MAX}")
            continue
        old = cur.get(knob, spec_v)
        book[knob] = {"value": v, "spec": spec_v, "reason": reason, "source": source, "ts": now,
                      **({"metric_v": METRIC_V[knob]} if knob in METRIC_V else {})}
        store.log("selftune", knob=knob, old=old, new=v, asked=value, spec=spec_v, reason=reason, source=source)
        acts.append(f"selftune {knob}: {old} -> {v}" + (f" (asked {value}, clamped)" if v != value else ""))
    store.set_state(K_OVERRIDES, book)
    return acts, proposals


def open_proposals(store: Any) -> list[tuple[str, str]]:
    """(id, text) of review proposals the operator has not acked."""
    acked = set(store.get_state(K_ACKED) or [])
    return [(pid, text) for rv in store.reviews() for i, text in enumerate(rv.proposals)
            if (pid := f"{rv.id}#{i}") not in acked]


def describe(cfg: Any) -> str:
    """The whitelist with bounds and the value in force (for review prompts)."""
    mins = {k: _get(cfg, m) for k, m in OPERATOR_MIN.items()}
    return "\n".join(f"- {k}: {_get(cfg, k)} (bounds {lo:g}..{hi:g}"
                     + (f"; OPERATOR minimum {mins[k]:g}, never tuned below it" if mins.get(k) else "") + ")"
                     for k, (lo, hi) in BOUNDS.items()) + \
        f"\n- constraint: the five arms' floors (floor for any arm without its own) sum to <= {FLOOR_SUM_MAX}"


def lines(store: Any, spec_cfg: Any = None) -> list[str]:
    """``loop status``: preemption, active (or stale) overrides and open operator proposals."""
    since = epoch_ts(store)
    out = [f"selftune {k} = {clamp(k, o['value']) if k in BOUNDS else o['value']} (spec {o.get('spec')}; "
           f"{o.get('source', '')}: {o.get('reason', '')[:120]})"
           + (" EXPIRED: tuned before the latest epoch bump" if expired(o, since) else "")
           + ((f" STALE, ignored: tuned on metric v{int(o.get('metric_v', 1))}, now v{METRIC_V[k]}"
               if k in METRIC_V and int(o.get("metric_v", 1)) != METRIC_V[k] else
               f" STALE, ignored: spec is now {_get(spec_cfg, k)}") if spec_cfg is not None and k in BOUNDS
              and stale(spec_cfg, k, o) else "")
           for k, o in sorted((store.get_state(K_OVERRIDES) or {}).items())]
    props = open_proposals(store)
    out.append(f"proposals {len(props)} awaiting operator" + (" (close with: python -m loop ack --task T <id>...)"
                                                               if props else ""))
    rep = store.get_state("redundancy.proposal_repeats") or {}       # loop.redundancy: later reviews restating it
    out += [f"  {pid}: " + (f"(restated {rep[pid]}x since) " if rep.get(pid) else "") + text[:300] for pid, text in props]
    return out


def arm_floor(cfg: Any, arm: Strategy) -> float:
    b = cfg.bandit
    own = {Strategy.EXPLORE: b.explore_floor, Strategy.MOONSHOT: b.moonshot_floor}.get(arm)
    return b.floor if own is None else own


def floors_line(spec_cfg: Any, store: Any) -> str:
    """Arm floors, spec -> effective (after live overrides and the operator floors)."""
    eff = effective(spec_cfg, store)
    return "floors   " + ", ".join(f"{a.value} {arm_floor(spec_cfg, a):.2f}->{arm_floor(eff, a):.2f}"
                                   for a in BANDIT_ARMS) + \
        f" (spec->effective); moonshot_max_inflight {spec_cfg.bandit.moonshot_max_inflight}->{eff.bandit.moonshot_max_inflight}"


# ------------------------------------------------ preemption telemetry (preemptible pools)
DAY = 86400.0


BACKFILL_S = 120.0


def real_preemptions(events: Sequence[Mapping[str, Any]], pools: Any) -> list[Mapping[str, Any]]:
    """The run_preempted events of ``pools`` minus the telemetry's first-poll backfill: before a receipt's
    restart baseline was seeded (``seeded`` events), the first poll after the requeue telemetry shipped logged
    every requeue the job had EVER had at once (106 events live, a fake '17%' preemption rate)."""
    ev = [e for e in events if e.get("pool") in pools]
    req = [e["ts"] for e in ev if str(e.get("detail", "")).startswith("slurm requeue")]
    t0 = min(req, default=0.0)
    return [e for e in ev if e.get("seeded") or not str(e.get("detail", "")).startswith("slurm requeue")
            or e["ts"] > t0 + BACKFILL_S]


RESUME_STEP_S = 0.24     # train step seconds: slurm.resume_save_every 500 steps ~2 min (loop.spec)


def lost_gpu_h(ctx: Any, e: Mapping[str, Any], resumed: bool) -> float:
    """GPU-h a preemption lost: the elapsed time before the requeue, but a requeue that resumed from the run's
    latest.pt kept its progress and lost only the time since the last checkpoint (<= one save interval)."""
    g, every = float(e.get("gpu_h") or 0.0), ctx.cfg.slurm.resume_save_every
    return min(g, every * RESUME_STEP_S / 3600) if resumed and every > 0 else g


def resumed_events(events: Sequence[Mapping[str, Any]], runs: Mapping[str, Any]) -> set[int]:
    """ids of the preemption events whose requeue resumed: the event's ``resumed`` flag, else (events from before
    it) the run's last ``run.resumed`` preemptions (a run resumes only once its first latest.pt exists)."""
    per: dict[Any, list[Mapping[str, Any]]] = {}
    for e in events:
        per.setdefault(e.get("run_id"), []).append(e)
    out: set[int] = set()
    for rid, evs in per.items():
        k = getattr(runs.get(rid), "resumed", 0) or 0
        out |= {id(e) for i, e in enumerate(evs) if (e["resumed"] if "resumed" in e else i >= len(evs) - k)}
    return out


def preempt_stats(ctx: Any, store: Any, now: float) -> dict[str, float] | None:
    """Last-24 h preemption of the preemptible pools (None: no such pool). Launches = first
    submissions on those pools plus every requeue; lost GPU-h = :func:`lost_gpu_h` per preemption
    (``requeue_gpu_h``: the elapsed time before each, resumed progress not credited)."""
    names = {p.name for p in ctx.cfg.pools if p.preemptible}
    if not names:
        return None
    ev = real_preemptions(store.events("run_preempted"), names)
    day = [e for e in ev if e["ts"] >= now - DAY]
    moved = {e.get("run_id") for e in store.events("run_repooled") if e.get("frm") in names}
    runs = [r for r in store.runs() if (r.pool or ctx.cfg.pools[0].name) in names or r.id in moved]  # v2
    per, res = Counter(e.get("run_id") for e in ev), resumed_events(ev, {r.id: r for r in store.runs()})
    done = [r for r in runs if r.status.value == "done"]
    launches = sum((r.submitted_ts or 0.0) >= now - DAY for r in runs) + len(day)
    live = [r.id for r in runs if r.status.value == "running" and r.id not in moved]
    held = set(store.get_state(K_PENDING) or ())
    return {"n": len(day), "pending": sum(i in held for i in live), "running": sum(i not in held for i in live), "launches": launches, "rate": len(day) / launches if launches else 0.0,
            "lost_gpu_h": sum(lost_gpu_h(ctx, e, id(e) in res) for e in day),
            "requeue_gpu_h": sum(float(e.get("gpu_h") or 0.0) for e in day),
            "useful_gpu_h": sum(r.gpu_h or 0.0 for r in done if (r.submitted_ts or 0.0) >= now - DAY),
            "requeues_per_done": sum(per[r.id] for r in done) / len(done) if done else 0.0}


def preempt_line(ctx: Any, store: Any, now: float) -> str:
    s = preempt_stats(ctx, store, now)
    return "" if s is None else (f"preempt 24h: {s['n']:.0f} ({s['rate']:.0%}), lost {s['lost_gpu_h']:.1f} vs useful "
                                 f"{s['useful_gpu_h']:.1f} GPU-h (metric v{PREEMPT_METRIC_V}); preemptible now "
                                 f"{s['running']} running, {s['pending']} pending")


PREEMPT_ADVICE = ("Grow slurm.preemptible_slots while lost GPU-h is small relative to useful GPU-h; "
                   "shrink it when preemption waste dominates.")


def preempt_block(ctx: Any, store: Any, now: float) -> str:
    """Evidence for reviews: pick the preemptible pool size from it (slurm.preemptible_slots)."""
    s = preempt_stats(ctx, store, now)
    return "no preemptible pool" if s is None else PREEMPT_ADVICE + "\n" + (
        f"preemptible pool, last 24 h: {s['n']:.0f} preemptions in {s['launches']:.0f} launches ({s['rate']:.0%}); "
        f"lost {s['lost_gpu_h']:.1f} GPU-h (time since the last checkpoint; {s['requeue_gpu_h']:.1f} GPU-h elapsed "
        f"before the requeues, the rest kept by resume-from-latest.pt) vs {s['useful_gpu_h']:.1f} useful GPU-h (finished "
        f"runs, incl. those that started preemptible and finished on primary); "
        f"mean requeues per completed run {s['requeues_per_done']:.2f}; now {s['running']} running, {s['pending']} "
        f"PENDING (metric v{PREEMPT_METRIC_V}: useful GPU-h includes runs repooled to primary; an override cites it)")
