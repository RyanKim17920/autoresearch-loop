"""loop.diagnostics -- the DIAGNOSTIC LAYER: what a finished run's checkpoint and metrics say about HOW it trained.

``loop/diag_ckpt.py`` (a standalone torch script) turns a DONE run's ``latest.pt`` + ``metrics.jsonl`` into ~110
floats. This module (stdlib only) launches it detached at low priority, ingests the JSON into the ``diagnostics``
table, and reads the table back three ways, all GENERAL (no node, idea or mechanism is named anywhere):

* :func:`bottlenecks` -- rule flags computed from the numbers (thresholds are the module constants below), each with
  the key quantity's trend root -> ancestors -> node;
* :func:`summary` -- the same as compact text for an LLM brief;
* :func:`mechanism_check` -- a node's DECLARED mechanism prediction ("quantity Q moves in direction d") checked on
  same-seed paired runs node-vs-parent: moved+gained / moved-no-gain / didn't-move.

CLI: ``python -m loop.diagnostics backfill --db DB --task TASK --state-dir DIR`` extracts every DONE run that still
has a checkpoint (read-only on the DB; resumable)."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

VERSION = 1
SCRIPT = Path(__file__).with_name("diag_ckpt.py")
SUBDIR = "diagnostics"
DEFAULT_PENDING_H = 2.0

# ---- rule thresholds (named, not buried) ------------------------------------------------------------------------
STARVE_GRAD_K = 1.0            # early-block grad RMS >= late-block grad RMS x this ...
STARVE_UPD_K = 5.0             # ... while early update/weight <= late / this: gradient present, update starved
STARVE_SEV_FULL = 100.0        # a late/early update ratio of this = severity 1
THROTTLE_UPD_MAX = 0.5         # early update/weight at most this fraction of late's (gradient >= late's) ...
THROTTLE_LR_TOL = 2.0          # ... and within this factor of the early / late LR multiplier = LR-throttled
THROTTLE_KNOB = "layerwise_decay"   # the per-block LR multiplier's config leaf (loop/diag_ckpt.py records it)
NULL_P_MIN = 0.05              # a close with P(own gain > 0) >= this is a null, not a harm (loop.explain)
ANNEAL_MAX_FRAC = 0.5          # final LR above this fraction of its peak = the schedule never annealed
CLIP_RATIO = 1.5               # late grad-norm / clip threshold above this ...
CLIP_FRAC = 0.5                # ... or a fraction of steps with grad_norm > clip above this = clip-dominated
TEACHER_LAG_REL = 0.10         # late-block student-teacher distance / late-block teacher drift above this = lag
VAL_EARLY_MIN_FRAC = 0.7       # val-loss minimum reached before this fraction of the run ...
VAL_RISE_REL = 0.02            # ... and the final loss above that minimum by more than this fraction of it
# ---- mechanism_check -----------------------------------------------------------------------------------------------
MOVE_SE = 2.0                  # moved = mean delta in the declared direction beyond this many standard errors ...
MOVE_REL = 0.10                # ... or beyond this fraction of the parent's own value
GAIN_SE = 1.0                  # gained = mean panel delta above this many standard errors (and > 0)
PANEL_SD_FALLBACK = 0.0033     # one pair has no se: the pooled paired-delta sd stands in (spec loop.panel_sd)
STAGE_PREF = ("parent", "confirm", "promote", "screen", "baseline", "calibrate", "remint", "tiebreak", "audit")


# ====================================================================== derived quantities
def _f(f: Mapping[str, float], k: str) -> float | None:
    v = f.get(k)
    return float(v) if isinstance(v, (int, float)) and math.isfinite(v) else None


def depth(f: Mapping[str, float]) -> int:
    d = _f(f, "n_blocks")
    if d:
        return int(d)
    idx = [int(m.group(1)) for k in f if (m := re.match(r"b(\d+)_", k))]
    return max(idx) + 1 if idx else 0


def blocks(n: int, which: str) -> range:
    """The early / mid / late third-ish of ``n`` blocks (a quarter at the ends, the middle third between)."""
    q = max(1, n // 4)
    return {"early": range(0, q), "mid": range(n // 3, n - n // 3), "late": range(n - q, n)}[which]


def _grp(kind: str, which: str) -> Callable[[Mapping[str, float]], float | None]:
    def fn(f: Mapping[str, float]) -> float | None:
        vals = [v for i in blocks(depth(f), which) if (v := _f(f, f"b{i}_{kind}")) is not None]
        return sum(vals) / len(vals) if vals else None
    return fn


def _ratio(a: str, b: str) -> Callable[[Mapping[str, float]], float | None]:
    def fn(f: Mapping[str, float]) -> float | None:
        x, y = Q[a][0](f), Q[b][0](f)
        return x / y if x is not None and y else None
    return fn


def _key(k: str) -> Callable[[Mapping[str, float]], float | None]:
    return lambda f: _f(f, k)


#: name -> (extractor over one run's features, what it is). Callers name quantities from this table.
Q: dict[str, tuple[Callable[[Mapping[str, float]], float | None], str]] = {}
for _w in ("early", "mid", "late"):
    Q[f"drift_{_w}"] = (_grp("tdrift", _w), f"{_w} blocks: teacher drift from init ||W-W0||/||W0||")
    Q[f"st_{_w}"] = (_grp("st", _w), f"{_w} blocks: student-teacher distance / ||W0||")
for _w in ("early", "mid", "late"):
    Q[f"grms_{_w}"] = (_grp("grms", _w), f"{_w} blocks: grad RMS (sqrt of Adam v_hat)")
    Q[f"upd_{_w}"] = (_grp("upd", _w), f"{_w} blocks: per-step update/weight ratio (lr * |m/sqrt(v)|_rms / |W|_rms)")
    Q[f"lr_{_w}"] = (_grp("lr_end", _w), f"{_w} blocks: final learning rate")


def _lr_mult(f: Mapping[str, float]) -> float | None:
    """Early / late blocks' LR multiplier: their final LRs' ratio, else layerwise_decay^(mean depth gap)."""
    e, l = Q["lr_early"][0](f), Q["lr_late"][0](f)
    if e and l:
        return e / l
    ld, n = _f(f, THROTTLE_KNOB), depth(f)
    if ld is None or ld <= 0 or not n:
        return None
    gap = sum(blocks(n, "late")) / len(blocks(n, "late")) - sum(blocks(n, "early")) / len(blocks(n, "early"))
    return ld ** gap


Q["lr_mult_early_late"] = (_lr_mult, "LR multiplier early / late blocks (layer-wise decay)")
Q["grms_early_over_late"] = (_ratio("grms_early", "grms_late"), "grad RMS early / late blocks")
Q["upd_ratio_early_late"] = (_ratio("upd_early", "upd_late"), "update/weight ratio early / late blocks")
Q["st_over_drift_late"] = (_ratio("st_late", "drift_late"), "late blocks: student-teacher distance / teacher drift")
for _k, _d in (("lr_end_frac", "final LR as a fraction of its peak (1 = never annealed)"),
               ("lr_peak", "peak learning rate"),
               ("flop_frac_end", "final flop_fraction of the schedule clock"),
               ("gn_late", "mean grad-norm over the last 25% of logged steps"),
               ("gn_early", "mean grad-norm over the first 10% of logged steps"),
               ("gn_late_over_clip", "late grad-norm / the clip threshold"),
               ("gn_clip_frac", "fraction of logged steps with grad_norm above the clip"),
               ("gn_clip_frac_late", "fraction of the last 25% of logged steps clipped"),
               ("patch_embed_drift", "teacher patch-embed drift from init"),
               ("pos_embed_drift", "teacher pos-embed drift from init"),
               ("cls_token_drift", "teacher cls-token drift from init"),
               ("register_tokens_drift", "teacher register-token drift from init"),
               (THROTTLE_KNOB, "the run's layer-wise LR decay"),
               ("val_total_rise", "final val_total above its minimum"),
               ("val_total_argmin_frac", "run fraction at which val_total was lowest"),
               ("val_total_end", "final val_total"),
               ("val_dino_rise", "final val_dino above its minimum"),
               ("val_jepa_rise", "final val_jepa above its minimum"),
               ("step", "optimizer steps at the checkpoint"),
               ("examples_seen", "tiles seen at the checkpoint"),
               ("train_flops", "training FLOPs at the checkpoint")):
    Q[_k] = (_key(_k), _d)
for _b in ("first", "last"):
    Q[f"qkv_erank_d_{_b}"] = ((lambda f, _b=_b: _f(f, "b0_qkv_erank_d" if _b == "first"
                                                     else f"b{depth(f) - 1}_qkv_erank_d")),
                              f"{_b} block: teacher qkv effective-rank change vs init")
    Q[f"fc2_erank_d_{_b}"] = ((lambda f, _b=_b: _f(f, "b0_fc2_erank_d" if _b == "first"
                                                     else f"b{depth(f) - 1}_fc2_erank_d")),
                              f"{_b} block: teacher fc2 effective-rank change vs init")


def quantities() -> dict[str, str]:
    """Every named derived quantity (``mechanism_check`` also accepts a raw feature key such as ``b3_upd``)."""
    return {k: d for k, (_, d) in Q.items()}


def quantity(features: Mapping[str, float], name: str) -> float | None:
    """One run's value of a named quantity (or of a raw feature key)."""
    return Q[name][0](features) if name in Q else _f(features, name)


_qval = quantity                                         # mechanism_check's `quantity` parameter shadows the name


def _node_values(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Each quantity averaged over a node's diagnosed runs."""
    out: dict[str, float] = {}
    for name in Q:
        vs = [v for r in rows if (v := quantity(r["features"], name)) is not None]
        if vs:
            out[name] = sum(vs) / len(vs)
    return out


NODE_FEATURES = ("lr_end_frac", "upd_ratio_early_late", "gn_late_over_clip", "drift_early", "drift_late", "st_late")


def node_features(store: Any, node_id: str) -> dict[str, float] | None:
    """The derived quantities a predictive model reads (``NODE_FEATURES``) averaged over the node's diagnosed runs;
    None when the node has no diagnostics."""
    rows = store.diagnostics(node_id=node_id)
    if not rows:
        return None
    v = _node_values(rows)
    return {k: v[k] for k in NODE_FEATURES if k in v}


# ====================================================================== flags
def _chain(store: Any, node_id: str) -> list[str]:
    """root -> ... -> node, following parent_id (cycle-safe)."""
    out, seen, cur = [], set(), node_id
    while cur and cur not in seen:
        seen.add(cur)
        out.append(cur)
        n = store.get_node(cur)
        cur = n.parent_id if n is not None else None
    return out[::-1]


def _trend(store: Any, node_id: str, name: str, vals: dict[str, dict[str, float]]) -> list[list[Any]]:
    path = []
    for nid in _chain(store, node_id):
        if nid not in vals:
            vals[nid] = _node_values(store.diagnostics(node_id=nid))
        if name in vals[nid]:
            path.append([nid, round(vals[nid][name], 6)])
    return path


def _sev(x: float) -> float:
    return round(max(0.0, min(1.0, x)), 3)


def _rules(v: Mapping[str, float]) -> list[tuple[str, float, dict[str, float], str, str]]:
    """(flag, severity, numbers, trend quantity, text) per fired rule over one node's mean quantities."""
    out = []
    ge, gl, ue, ul = (v.get(k) for k in ("grms_early", "grms_late", "upd_early", "upd_late"))
    lm = v.get("lr_mult_early_late")
    throttled = None not in (ge, gl, ue, ul, lm) and gl > 0 and ul > 0 and ue > 0 and lm > 0 \
        and ge >= gl * STARVE_GRAD_K and ue / ul <= THROTTLE_UPD_MAX \
        and 1 / THROTTLE_LR_TOL <= ue / ul / lm <= THROTTLE_LR_TOL
    if throttled:                                        # update_starved, explained by the per-block LR multiplier
        de, dl, ld = v.get("drift_early"), v.get("drift_late"), v.get(THROTTLE_KNOB)
        out.append(("lr_throttled", _sev(1.0 - ue / ul), {
            "grms_early_over_late": ge / gl, "upd_ratio_early_late": ue / ul, "lr_mult_early_late": lm,
            **({"drift_early": de, "drift_late": dl} if None not in (de, dl) else {}),
            **({THROTTLE_KNOB: ld} if ld is not None else {})}, "upd_ratio_early_late",
            f"lower blocks LR-throttled: grad RMS x{ge / gl:.2f} of the top blocks' yet update/weight x{ue / ul:.3f} "
            f"~ their LR multiplier x{lm:.3f}" + (f" ({THROTTLE_KNOB} {ld:g})" if ld is not None else "")
            + (f", teacher drift {de:.1%} vs {dl:.1%}" if None not in (de, dl) else "")
            + f": movement is set by the LR multiplier, not the gradient (knob {THROTTLE_KNOB})"))
    elif None not in (ge, gl, ue, ul) and gl > 0 and ul > 0 and ue >= 0 \
            and ge >= gl * STARVE_GRAD_K and ue <= ul / STARVE_UPD_K:
        ratio = ul / max(ue, 1e-30)
        out.append(("update_starved", _sev(math.log(ratio / STARVE_UPD_K + 1.0) / math.log(STARVE_SEV_FULL)),
                    {"grms_early": ge, "grms_late": gl, "upd_early": ue, "upd_late": ul, "upd_late_over_early": ratio},
                    "upd_ratio_early_late",
                    f"gradient present but update starved: early-block grad RMS {ge:.3g} vs late {gl:.3g} "
                    f"(x{ge / gl:.2f}) yet early update/weight {ue:.3g} is {ratio:.0f}x below late {ul:.3g}"))
    lr = v.get("lr_end_frac")
    if lr is not None and lr > ANNEAL_MAX_FRAC:
        n = {"lr_end_frac": lr, **({"flop_frac_end": v["flop_frac_end"]} if "flop_frac_end" in v else {})}
        out.append(("schedule_not_annealed", _sev(lr),
                    n, "lr_end_frac",
                    f"schedule not annealed: the run ended at {lr:.0%} of peak LR"
                    + (f" (flop_fraction {v['flop_frac_end']:.2f})" if "flop_frac_end" in v else "")))
    r, cf = v.get("gn_late_over_clip"), v.get("gn_clip_frac")
    if (r is not None and r > CLIP_RATIO) or (cf is not None and cf > CLIP_FRAC):
        out.append(("clip_dominated", _sev(max(cf or 0.0, (r or 0.0) / (2 * CLIP_RATIO))),
                    {k: v[k] for k in ("gn_late", "gn_late_over_clip", "gn_clip_frac", "gn_clip_frac_late") if k in v},
                    "gn_late_over_clip",
                    f"clip-dominated: late grad-norm is {r if r is not None else float('nan'):.2f}x the clip, "
                    f"{(cf or 0.0):.0%} of steps clipped (the update direction is normalised, not scaled by the loss)"))
    lag = v.get("st_over_drift_late")
    if lag is not None and lag > TEACHER_LAG_REL:
        out.append(("teacher_lag", _sev(lag / (4 * TEACHER_LAG_REL)),
                    {"st_late": v["st_late"], "drift_late": v["drift_late"], "st_over_drift_late": lag},
                    "st_over_drift_late",
                    f"teacher lag: late-block student-teacher distance is {lag:.0%} of the teacher's drift "
                    f"(student {v['st_late']:.3g} vs teacher drift {v['drift_late']:.3g})"))
    rise, am, end = v.get("val_total_rise"), v.get("val_total_argmin_frac"), v.get("val_total_end")
    if None not in (rise, am, end) and am <= VAL_EARLY_MIN_FRAC and end - rise > 0 \
            and rise / (end - rise) > VAL_RISE_REL:
        rel = rise / (end - rise)
        out.append(("val_loss_rising_late", _sev(rel / (5 * VAL_RISE_REL)),
                    {"val_total_rise": rise, "val_total_argmin_frac": am, "rise_rel": rel},
                    "val_total_rise",
                    f"val loss rising late: minimum at {am:.0%} of the run, final {rel:.1%} above it"))
    return out


def bottlenecks(store: Any, node_id: str) -> list[dict]:
    """Generic rule flags over a node's diagnosed runs (quantities averaged across them), strongest first. Each:
    {flag, severity 0..1, numbers{...}, trend [[node_id, value], ...] root -> ancestors -> node of the key quantity,
    text}. [] when the node has no diagnostics or no rule fires."""
    rows = store.diagnostics(node_id=node_id)
    if not rows:
        return []
    mine = _node_values(rows)
    cache: dict[str, dict[str, float]] = {node_id: mine}
    flags = [{"flag": flag, "severity": sev, "numbers": {k: round(x, 6) for k, x in nums.items()},
              "trend": _trend(store, node_id, tq, cache), "trend_quantity": tq, "n_runs": len(rows), "text": text}
             for flag, sev, nums, tq, text in _rules(mine)]
    return sorted(flags, key=lambda d: -d["severity"])


def summary(store: Any, node_id: str) -> str:
    """Compact multi-line text for an LLM brief: flags with numbers and the root -> node trend; '' when the node has
    no diagnostics."""
    rows = store.diagnostics(node_id=node_id)
    if not rows:
        return ""
    v = _node_values(rows)
    head = ", ".join(f"{k}={v[k]:.3g}" for k in ("lr_end_frac", "gn_late_over_clip", "upd_early", "upd_late",
                                                  "drift_early", "drift_late", "st_late") if k in v)
    lines = [f"run diagnostics ({len(rows)} run(s)): {head}"]
    if blk := ", ".join(f"{n} x{v[k]:.3g}" for n, k in (("grad RMS", "grms_early_over_late"), (
            "update/weight", "upd_ratio_early_late"), ("LR multiplier", "lr_mult_early_late")) if k in v):
        tok = ", ".join(f"{t} {v[k]:.3g}" for t, k in (("patch_embed", "patch_embed_drift"), (
            "pos_embed", "pos_embed_drift"), ("cls_token", "cls_token_drift"), ("registers", "register_tokens_drift"))
                        if k in v)
        lines.append(f"blocks early / late: {blk}" + (f"; embedding drift {tok}" if tok else ""))
    flags = bottlenecks(store, node_id)
    for d in flags:
        tr = " -> ".join(f"{n}:{x:.3g}" for n, x in d["trend"]) if len(d["trend"]) > 1 else ""
        lines.append(f"- [{d['flag']} sev {d['severity']:.2f}] {d['text']}" + (f"; {d['trend_quantity']} trend {tr}" if tr else ""))
    if not flags:
        lines.append("- no bottleneck flag fired")
    return "\n".join(lines)


# ====================================================================== hooks other modules call
#: the flags that are about the optimisation itself (loop.variants adds an optimiser co-tune when one fires)
OPTIMISER_FLAGS = frozenset({"update_starved", "lr_throttled", "schedule_not_annealed", "clip_dominated"})


def optimiser_flags(store: Any, node_ids: Sequence[str]) -> list[str]:
    """The OPTIMISER_FLAGS firing on the first of ``node_ids`` that has diagnostics ([] when none has)."""
    for nid in node_ids:
        if store.diagnostics(node_id=nid):
            return [d["flag"] for d in bottlenecks(store, nid) if d["flag"] in OPTIMISER_FLAGS]
    return []


def explain(node: Any, store: Any, snapshot: Any = None) -> str | None:
    """loop.explain's hook (why a node closed the way it did): the predictive model's forecast vs measured residual
    and gate posterior plus this module's summary (loop.world.predictive.explain_text), and :func:`under_tuned`'s
    verdict. None when nothing to say."""
    from loop.world.predictive import explain_text
    return "\n".join(x for x in (explain_text(snapshot, store, node.id), under_tuned(node, store, snapshot)) if x) \
        or None


def under_tuned(node: Any, store: Any, snapshot: Any = None) -> str:
    """'' or the close verdict ``under_tuned (knob layerwise_decay)``: a low-level change (loop.campaign.input_side
    without a config: an init swap, a data-sampling / augmentation mechanism) that closed a NULL (screened out / not
    confirmed, P(own gain > 0) >= NULL_P_MIN) while ``lr_throttled`` fires on its runs (else its parent's: it
    inherits the multiplier) -- its lowest blocks, the ones a new input needs, barely moved."""
    from loop.campaign import input_side
    st = getattr(node.status, "value", node.status)
    p = (getattr(snapshot, "nodes", None) or {}).get(node.id)
    if st not in ("screened_out", "not_confirmed") or (p is not None and p.gain_p is not None
                                                       and p.p_gain < NULL_P_MIN):
        return ""
    try:
        why = input_side(None, store, node)
    except Exception:                                   # noqa: BLE001 - an unreadable node is not low-level
        why = ""
    src = next((nid for nid in (node.id, node.parent_id) if nid and store.diagnostics(node_id=nid)), "")
    f = next((d for d in bottlenecks(store, src) if d["flag"] == "lr_throttled"), None) if why and src else None
    if f is None:
        return ""
    whose = "its" if src == node.id else f"its parent {src}'s"
    return (f"close: under_tuned (knob {THROTTLE_KNOB}) -- a low-level change ({why}) closed a null while {whose} "
            f"lower blocks were LR-throttled (update/weight x{f['numbers']['upd_ratio_early_late']:.3f} of the top "
            f"blocks')")


K_BANKED = "diagnostics.banked"   # ids of the bank ideas :func:`bank` wrote (once per incumbent each)
TOKEN_DRIFT_K = 2.0            # a token parameter drifting more than this x the early blocks' = it trains faster


def bank_rows(ctx: Any, store: Any, node_id: str) -> list[dict]:
    """Idea-bank rows (loop.select.bank's generated format, campaign-eligible) COMPUTED from ``node_id``'s flags:
    with ``lr_throttled`` firing, (1) when token parameters (position embedding, class / register tokens) drift more
    than TOKEN_DRIFT_K x the early blocks -- no LR-multiplier branch matched them, so they train at the full LR --
    give them the lowest block's multiplier; (2) anneal the layer decay from 1.0 to its configured value over
    training, so the lowest blocks move early and are protected late. [] when the flag does not fire."""
    f = next((d for d in bottlenecks(store, node_id) if d["flag"] == "lr_throttled"), None)
    if f is None:
        return []
    v, num = _node_values(store.diagnostics(node_id=node_id)), f["numbers"]
    fam = next((n for n, r in getattr(ctx.cfg, "families", {}).items()
                if any(g.rsplit(".", 1)[-1] == THROTTLE_KNOB for g in r.keys)), "")
    lm, de = num["lr_mult_early_late"], v.get("drift_early")
    seen = f"diagnostics of {node_id}: {f['text']}"
    row = lambda tag, title, idea, why: {                                    # noqa: E731
        "id": f"diag-{tag}-{node_id}", "title": title, "family": fam, "moonshot": False, "priority": True,
        "idea": idea, "why_it_might_work": why, "source": f"loop.diagnostics ({node_id}, computed)",
        "plan": {"mechanism_tag": tag, "component": "optimizer parameter groups (per-parameter LR multiplier)",
                 "campaign": True, "search_space": [], "from_request": f"diagnostics:{node_id}"}}
    out = []
    tok = {t: v[k] for t, k in (("position embedding", "pos_embed_drift"), ("class token", "cls_token_drift"),
                                ("register tokens", "register_tokens_drift")) if k in v}
    fast = {t: x for t, x in tok.items() if de and x > TOKEN_DRIFT_K * de}
    if fast:
        out.append(row("token_params_lowest_lr", "Token parameters get the lowest block's LR multiplier",
                       f"Give the backbone's token parameters (position embedding, class / register / mask tokens) "
                       f"the LR multiplier of the lowest block, as DINOv2 does (layer id 0), instead of the full LR "
                       f"the parameter-group code gives every backbone parameter outside the blocks and the patch "
                       f"embedding. A small change to the optimizer's parameter groups; no other change.",
                       f"{seen}. Token drift " + ", ".join(f"{t} {x:.1%}" for t, x in fast.items())
                       + f" vs the early blocks' {de:.1%}: the tokens feeding block 0 move "
                       f"x{max(fast.values()) / de:.1f} as fast as it while it trains at x{lm:.3f} of the top blocks' LR, so the lowest block keeps "
                       f"re-fitting a moving input."))
    out.append(row("layer_decay_anneal", "Layer-wise LR decay annealed from 1.0 to its configured value",
                   f"Schedule the layer-wise LR decay instead of fixing it: start at 1.0 (every block at the full LR) "
                   f"and anneal to the configured {THROTTLE_KNOB} over a fraction of training (a new config key for "
                   f"that fraction, 0 = today's fixed decay), recomputing each block's LR multiplier every step.",
                   f"{seen}. The lowest blocks get the largest gradient but the smallest step: early in training "
                   f"they never adapt to the data; annealing keeps the protection late (when it stabilises the "
                   f"features) without freezing them early."))
    return out


def bank(ctx: Any, store: Any, snapshot: Any) -> list[str]:
    """Write :func:`bank_rows` of the incumbent into the loop's generated idea bank (loop.agents.ideas.write_bank),
    each id once; the lines to log."""
    inc = getattr(snapshot, "incumbent_id", "") or "root"
    done = set(store.get_state(K_BANKED) or ())
    rows = [r for r in bank_rows(ctx, store, inc) if r["id"] not in done] if store.diagnostics(node_id=inc) else []
    if not rows:
        return []
    from loop.agents.ideas import write_bank
    write_bank(ctx, store, rows)
    store.set_state(K_BANKED, sorted(done | {r["id"] for r in rows}))
    store.log("diagnostics_banked", node_id=inc, ids=[r["id"] for r in rows])
    return [f"diagnostics: banked {', '.join(r['id'] for r in rows)} (computed from {inc}'s lr_throttled flag)"]


def allocation(store: Any, snapshot: Any, recent: int = 80) -> dict[str, float]:
    """The predictive model's allocation as SHARES for ideation (loop.agents.dossiers): per mechanism type
    (loop.novelty.TYPES) its share of the summed P(fresh gain > MDE) over the cells on root + the incumbent
    (loop.world.predictive.Fit.allocation), and per family its share of the summed mean P(fresh gain > MDE) of its
    ``recent`` newest nodes rebuilt on the incumbent (Fit.value); a name that is both keeps the larger. A share is
    what the model would spend on that key, compared with its share of recent draws. {} without the model."""
    wm = getattr(snapshot, "wm", None)
    if wm is None or getattr(wm, "book", None) is None:
        return {}
    inc = getattr(snapshot, "incumbent_id", "") or "root"
    by_t: dict[str, float] = {}
    for c in wm.allocation(list(dict.fromkeys(["root", inc]))):
        by_t[c["mtype"]] = by_t.get(c["mtype"], 0.0) + c["p_mde"]
    by_f: dict[str, list[float]] = {}
    for n in sorted((n for n in wm.book.nodes.values() if n.id != "root" and n.family),
                    key=lambda n: (n.created_ts, n.id))[-recent:]:
        by_f.setdefault(n.family, []).append(wm.value(n, parent=inc))
    fam = {f: sum(v) / len(v) for f, v in by_f.items()}
    out: dict[str, float] = {}
    for d in (by_t, fam):
        tot = sum(d.values())
        for k, v in d.items():
            if tot > 0:
                out[k] = max(out.get(k, 0.0), v / tot)
    return out


# ====================================================================== mechanism check
def _pick(runs: Mapping[str, Any]) -> Any:
    """The run of the best stage in ``runs`` (stage -> run), else None."""
    return next((runs[st] for st in STAGE_PREF if st in runs), None)


def _mean_se(xs: Sequence[float]) -> tuple[float, float | None]:
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, None
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
    return m, sd / math.sqrt(len(xs))


def paired_runs(store: Any, node_id: str) -> list[tuple[Any, Any]]:
    """(node run, parent run) per seed where both are DONE and diagnosed (the parent's run at that seed may be of any
    stage; a node's own stage preference is the same)."""
    node = store.get_node(node_id)
    if node is None:
        return []
    parent_id = node.parent_id or "root"
    have = {r["run_id"] for r in store.diagnostics()}
    sides = []
    for nid in (node_id, parent_id):
        by_seed: dict[int, dict[str, Any]] = {}
        for r in store.runs(node_id=nid):
            if getattr(r.status, "value", r.status) == "done" and r.id in have:
                by_seed.setdefault(r.seed, {})[getattr(r.stage, "value", r.stage)] = r
        sides.append(by_seed)
    out = []
    for seed in sorted(set(sides[0]) & set(sides[1])):
        a, b = (_pick(s[seed]) for s in sides)
        if a is not None and b is not None:
            out.append((a, b))
    return out


def _panel_delta(store: Any, a: Any, b: Any) -> float | None:
    if a.aggregate is not None and b.aggregate is not None:
        return a.aggregate - b.aggregate
    try:
        from kernel.stats import paired_delta
        d = paired_delta(a.scores, b.scores, getattr(store, "_weights", None) or {})
        return d.aggregate if d.deltas else None
    except Exception:                                    # noqa: BLE001
        return None


def mechanism_check(store: Any, node_id: str, quantity: str, direction: int) -> dict:
    """Did a node's DECLARED mechanism prediction ("``quantity`` moves in ``direction`` (+1 up / -1 down)") come true,
    and did the panel gain? Uses same-seed pairs node-vs-parent (both runs diagnosed; any stage on either side).
    ``verdict``: 'moved+gained' | 'moved-no-gain' | "didn't-move" | 'no-data' (no pair) | 'unknown-quantity'."""
    direction = 1 if direction > 0 else -1
    res: dict[str, Any] = {"node_id": node_id, "quantity": quantity, "direction": direction, "n": 0,
                           "verdict": "no-data"}
    by_id = {r["run_id"]: r["features"] for r in store.diagnostics()}
    pairs = []
    for a, b in paired_runs(store, node_id):
        qa, qb = _qval(by_id[a.id], quantity), _qval(by_id[b.id], quantity)
        if qa is None or qb is None:
            continue
        pairs.append({"seed": a.seed, "node_run": a.id, "parent_run": b.id, "node": qa, "parent": qb,
                      "dq": qa - qb, "dpanel": _panel_delta(store, a, b)})
    if not pairs:
        if quantity not in Q and not any(quantity in f for f in by_id.values()):
            res["verdict"] = "unknown-quantity"
        return res
    dq = [p["dq"] for p in pairs]
    mean, se = _mean_se(dq)
    pm = sum(p["parent"] for p in pairs) / len(pairs)
    signed = direction * mean
    moved = signed > 0 and ((se is not None and se > 0 and signed > MOVE_SE * se)
                            or (pm != 0 and signed / abs(pm) > MOVE_REL))
    panel = [p["dpanel"] for p in pairs if p["dpanel"] is not None]
    pmean, pse = _mean_se(panel) if panel else (None, None)
    gained = bool(panel) and pmean > 0 and pmean > GAIN_SE * (pse if pse is not None else PANEL_SD_FALLBACK)
    res.update(n=len(pairs), mean_delta=mean, se=se, parent_mean=pm, node_mean=sum(p["node"] for p in pairs) / len(pairs),
               rel_delta=(mean / abs(pm)) if pm else None, moved=moved, gained=gained, panel_mean=pmean,
               panel_se=pse, n_panel=len(panel), pairs=pairs,
               verdict=("moved+gained" if gained else "moved-no-gain") if moved else "didn't-move")
    return res


# ====================================================================== launch / ingest / pending
def diag_dir(ctx: Any) -> Path:
    return Path(ctx.state_dir) / SUBDIR


def safe_name(run_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", run_id)


def paths(ctx: Any, run_id: str) -> tuple[Path, Path, Path]:
    """(result json, launch marker, log) of one run's extraction."""
    d, n = diag_dir(ctx), safe_name(run_id)
    return d / f"{n}.json", d / f"{n}.launched", d / f"{n}.log"


def _cfg(ctx: Any) -> Any:
    return getattr(getattr(ctx, "cfg", None), "diagnostics", None)


_PROCS: list[subprocess.Popen] = []
_EMPTY: dict[str, float] = {}                              # file -> mtime of an output with no features (read once)


def pending(ctx: Any, run: Any, now: float | None = None) -> bool:
    """True while a launched extraction of ``run`` has produced no output and is not older than ``pending_h``
    (after which it is abandoned: the prune pass never waits on it forever)."""
    out, marker, _ = paths(ctx, run.id)
    if out.exists() or not marker.exists():
        return False
    hours = getattr(_cfg(ctx), "pending_h", DEFAULT_PENDING_H)
    try:
        return (time.time() if now is None else now) - marker.stat().st_mtime < 3600.0 * hours
    except OSError:
        return False


def run_dir_of(run: Any) -> Path | None:
    return Path(run.output_dir) if run.output_dir and (Path(run.output_dir) / "latest.pt").is_file() else None


def command(python: str, run_dir: Path, init_path: str, out: Path, run_id: str, threads: int,
            model_dir: str = "") -> list[str]:
    """``model_dir``: the node's code, so the drift features use the run's OWN init (loop/diag_ckpt.resolve_init)."""
    return ["nice", "-n", "19", python, str(SCRIPT), "--run-dir", str(run_dir), "--init", init_path,
            "--out", str(out), "--run-id", run_id, "--threads", str(threads)] + (
        ["--model-dir", model_dir] if model_dir else [])


def _env(threads: int) -> dict[str, str]:
    t = str(max(1, min(4, int(threads))))
    return {**os.environ, "OMP_NUM_THREADS": t, "MKL_NUM_THREADS": t, "OPENBLAS_NUM_THREADS": t}


def launch(ctx: Any, run: Any, python: str, init_path: str, model_dir: str = "") -> int | None:
    """Start the extraction of ``run`` detached at ``nice 19`` (<= 4 threads); None when there is nothing to do (no
    checkpoint, already launched / done). Writes ``<state_dir>/diagnostics/<run>.json`` when it finishes."""
    rd = run_dir_of(run)
    out, marker, log = paths(ctx, run.id)
    if rd is None or out.exists() or pending(ctx, run):
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    threads = int(getattr(_cfg(ctx), "threads", 4))
    marker.write_text(str(time.time()))
    with open(log, "w") as lf:
        p = subprocess.Popen(command(python, rd, init_path, out, run.id, threads, model_dir), stdout=lf,
                             stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=_env(threads), start_new_session=True)
    _PROCS.append(p)
    return p.pid


def ingest(ctx: Any, store: Any) -> int:
    """Insert every finished extraction JSON whose run has no diagnostics row yet; return how many."""
    _PROCS[:] = [p for p in _PROCS if p.poll() is None]    # reap finished children
    d = diag_dir(ctx)
    if not d.is_dir():
        return 0
    have = {safe_name(r) for r in store.diagnosed_ids()}
    n = 0
    for f in sorted(d.glob("*.json")):
        if f.stem in have:
            continue
        try:
            mt = f.stat().st_mtime
            if _EMPTY.get(str(f)) == mt:
                continue
            doc = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        feats = doc.get("features") if isinstance(doc, dict) else None
        if not feats:
            _EMPTY[str(f)] = mt
            continue
        run = store.get_run(doc.get("run_id") or "")
        if run is None:
            continue
        store.put_diagnostics(run.id, node_id=run.node_id, seed=run.seed, stage=run.stage, features=feats,
                              version=int(doc.get("version", VERSION)), ts=mt)
        n += 1
    return n


# ====================================================================== backfill
def backfill(db: str, task: str, state_dir: str, *, python: str, init: str, threads: int = 4,
             limit: int = 0, log: Callable[[str], None] = print) -> tuple[int, int, int]:
    """Extract every DONE run of ``task`` whose output dir still has ``latest.pt`` and has no JSON yet, one at a time
    under nice 19. Reads the DB READ-ONLY. Returns (done, failed, skipped)."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=30)
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT id, output_dir FROM runs WHERE task_id=? AND status='done' ORDER BY rowid DESC",
                       (task,)).fetchall()
    con.close()
    out_dir = Path(state_dir) / SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = bad = skip = 0
    for r in rows:
        out = out_dir / f"{safe_name(r['id'])}.json"
        rd = Path(r["output_dir"] or "")
        if out.exists() or not (rd / "latest.pt").is_file():
            skip += 1
            continue
        if limit and ok + bad >= limit:
            break
        t0 = time.time()
        try:
            rc = subprocess.run(command(python, rd, init, out, r["id"], threads), env=_env(threads),
                                stdin=subprocess.DEVNULL, capture_output=True, text=True).returncode
        except OSError as exc:
            rc = -1
            log(f"{r['id']}: {exc!r}")
        if rc == 0 and out.exists():
            ok += 1
        else:
            bad += 1
            out.unlink(missing_ok=True)                    # a failed run stays un-extracted (retried next time)
        log(f"{time.strftime('%H:%M:%S')} {r['id']} rc={rc} {time.time() - t0:.1f}s (done {ok}, failed {bad})")
    return ok, bad, skip


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m loop.diagnostics")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backfill", help="extract features of every DONE run that still has a checkpoint")
    b.add_argument("--db", required=True)
    b.add_argument("--task", required=True)
    b.add_argument("--state-dir", required=True)
    b.add_argument("--python", default=sys.executable)
    b.add_argument("--init", default="")
    b.add_argument("--threads", type=int, default=4)
    b.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    ok, bad, skip = backfill(a.db, a.task, a.state_dir, python=a.python, init=a.init, threads=a.threads,
                             limit=a.limit, log=lambda s: print(s, flush=True))
    print(f"backfill finished: {ok} extracted, {bad} failed, {skip} skipped", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ====================================================================== FLOP accounting audit (v3 review M2)
K_FLOP_AUDIT = "audit.flop"          # {"ts": last pass, "alerted": {run_id: ts}}
FLOP_AUDIT_RATIO = 1.15              # gpu_h above this x the stage median ...
FLOP_AUDIT_FRAC = 1.0                # ... while the reported flop_fraction is at most this: compute the charge did not see
FLOP_AUDIT_EVERY_S = 600.0
FLOP_AUDIT_MIN = 8                   # stage runs needed for a median
FLOP_FULL_FRAC = 0.9                 # a FULL-train run reached this flop_fraction (the cap, or its sample budget)


def flop_audit(store: Any, now: float, *, ratio: float = FLOP_AUDIT_RATIO, every: float = FLOP_AUDIT_EVERY_S) -> list[str]:
    """ALERT lines (never a void) for FULL-train runs that cost far more GPU time than their stage's median although
    their reported flop_fraction is <= 1.0: FLOPs are measured at step 1 and charged per step, so an uncharged later
    cost (a scoring pass, a second step) shows only in wall time. Medians over the stage's finished, non-resumed,
    first-attempt runs (live: 0.94 GPU-h, p90 1.03); once per run (state ``audit.flop``); throttled."""
    from loop.contracts import RunStatus
    st = dict(store.get_state(K_FLOP_AUDIT) or {})
    if now - float(st.get("ts") or 0.0) < every:
        return []
    alerted = dict(st.get("alerted") or {})
    by_stage: dict[str, list[Any]] = {}
    for r in store.runs(status=[RunStatus.DONE]):
        if r.gpu_h and r.gpu_h > 0 and not r.resumed and not r.attempt:
            by_stage.setdefault(getattr(r.stage, "value", str(r.stage)), []).append(r)
    out: list[str] = []
    for stage, rs in by_stage.items():
        if len(rs) < FLOP_AUDIT_MIN:
            continue
        gs = sorted(r.gpu_h for r in rs)
        med = gs[len(gs) // 2] if len(gs) % 2 else (gs[len(gs) // 2 - 1] + gs[len(gs) // 2]) / 2
        for r in rs:
            if r.gpu_h <= ratio * med or r.id in alerted:
                continue
            feats = next((d["features"] for d in store.diagnostics(node_id=r.node_id) if d["run_id"] == r.id), None)
            frac = (feats or {}).get("flop_frac_end")
            if frac is None or not FLOP_FULL_FRAC <= frac <= FLOP_AUDIT_FRAC:
                continue
            alerted[r.id] = now
            store.log("ALERT_flop_audit", run_id=r.id, node_id=r.node_id, stage=stage, gpu_h=round(r.gpu_h, 3),
                      stage_median=round(med, 3), flop_fraction=round(float(frac), 4))
            out.append(f"ALERT flop audit: {r.id} took {r.gpu_h:.2f} GPU-h = {r.gpu_h / med:.2f}x the {stage} median "
                       f"{med:.2f} with reported flop_fraction {float(frac):.3f} <= {FLOP_AUDIT_FRAC:g}: compute the "
                       f"FLOP charge may not see (flagged, not voided)")
    store.set_state(K_FLOP_AUDIT, {"ts": now, "alerted": alerted})
    if len(out) > 3:                         # the first pass over a long history: one line, every run in the events
        out = [f"ALERT flop audit: {len(out)} full-train runs took > {ratio:g}x their stage median GPU-h with a reported "
               f"flop_fraction <= {FLOP_AUDIT_FRAC:g} (flagged, not voided; each in an ALERT_flop_audit event): "
               + ", ".join(x.split()[3] for x in out[:3]) + ", ..."]
    return out
