"""loop.core.struct_tune -- the STRUCTURAL TUNING LADDER (operator 2026-10-22: "tune structural moonshots before judging
them"; backbone swaps a node ViT-B -.0085 / its retune -.0113 / a node DINOv3-S -.011 were each judged on about one
untuned shot).

The v2 lanes that tuned a closed moonshot (loop.tune's pass, loop.campaign) are not wired into the v3 tick, and the one
v3 ladder (loop.core.program) covers only ``brief=program`` nodes with declared components. This module is the v3 form,
built on the program-ladder machinery (``program._child`` config-only children tagged ``ladder=<node>``: program-child
priority in ``acquire.program_nodes``, never aged out or dropped as starved, left out of the brief throttle's queue age).

WHICH NODES (:func:`kind_of`): a node of the CURRENT base that is screening with >= 1 paired seed and

* ``backbone``  changes the backbone: ``model.type`` / ``model.pretrained_url`` in its diff, an ``init=`` note, or
  another ``model.type`` than its parent's. ALWAYS laddered (a new backbone's hyperparameters were set for the old one);
* ``loss``      adds / moves a loss weight (a numeric own key whose leaf matches ``struct_tune_knobs.loss.dose_rx``);
* ``other``     any other bold / moonshot structural node (``brief=bold``, strategy moonshot, a retro port): its own new
  numeric config keys, else the lr x layer-decay grid.
* an operator program's ``arch_arm=1`` node (``program._build_ported_arm``) is ``backbone`` when its diff swaps the
  backbone, else ``other`` (a config-only port such as ``natinit.block_lr_mult``); either way its points are the arm's
  ``tune`` dict, and its ladder is OUTSIDE ``struct_ladder_max_open`` (never waits behind the auto ladders).

A non-backbone node is laddered while its belief is above -``struct_tune_min_z`` sd. ``brief=program`` nodes keep
program.ladders' component ladder (a backbone program without one gets this ladder).

THE LADDER (:func:`struct_ladders`): <= ``struct_tune_max`` (5) config-only children on the node's commit, built once
(``K_STRUCT``), tagged ``ladder=<node> struct_tune=<kind> cell=<what>``:

* ``backbone``  a 2 x 2 over lr x layer decay (``struct_tune_knobs.backbone.grid``: lr value/2, value*2; layer decay
  value -/+ .15 clipped to [.5, 1]) plus one warmup child (``extra``). lr and layer decay are INCLUDED here although
  ``moonshot_tune.shared`` leaves them out;
* ``loss`` / ``other``  doses of the node's own new knobs at the yaml factors, STEERED by the engagement diagnostic
  (:func:`steer`): engaged but negative -> lower doses first; did not engage -> higher first; reversed -> a sign flip
  first when the knob is signed, then lower doses; unknown -> alternate. A node with no own numeric knob gets the
  backbone grid. The units' budget is shared with ``bold.tune_used`` (a bold family's 5 trials).

CLOSE RULE (:func:`held`): a structural node is not CLOSED as negative (it adds no negative evidence to its mechanism
cluster, :func:`loop.core.alloc.clusters`, and is not early-killed) until >= ``struct_min_points`` (3) of its tuned
children finished, unless its first point is CATASTROPHIC (:func:`catastrophic`: worse than -``struct_cata_z`` sd on
>= ``struct_cata_seeds`` seeds with the mechanism engaged).

JUDGEMENT: the children are screened like any node; a child is the best tuned point only through the usual fresh-seed
TEST, and its test entry is max-of-K shrunk with at least ``struct_tune_beta`` (:func:`loop.core.acquire.beta_for`):
the raw best of <= 5 is never compared with the incumbent.

RETRO (:func:`retro`): old-epoch backbone swaps (auto) and the operator's explicit ``ideation.struct_retro_nodes``
are ported onto the current base (``loop.cutover.port_node``), the port tagged ``ladder=<old> struct_retro=1`` (program
priority), and ladder as any structural node once it screens. At most ``struct_retro_max`` ports are open at once.

INTAKE (:func:`untunable`): a bold / moonshot candidate whose diff adds no numeric config key and is no backbone swap
must declare ``components`` ({key, lo, hi, scale}) in its finish, else ``untunable_moonshot``."""

from __future__ import annotations

import json
import math
import re
import time
from typing import Any, Mapping, Sequence

K_STRUCT = "ideation.struct_tune"     # {node: {kind, children, steer, ts} | "-: why not"}
K_RECHECK = "ideation.struct_recheck"   # {node: belief at its last flat-floor recheck}
SKIP_FIRST = "-: skipped: first point"   # the booking of a hopeless untuned first shot (re-checked under a flat floor)
K_RETRO = "ideation.struct_retro"     # {old node: {port, ts, why} | "-: why not"}
K_CATA = "ideation.struct_cata"       # {node: why}: a catastrophic first point (no ladder; the close is allowed)
TAG = " struct_tune="
RETRO_TAG = " struct_retro=1"
ARCH_TAG = " arch_arm=1"           # an operator architecture arm (a prebuilt code port): laddered like a backbone swap
BACKBONE_KEYS = ("model.type", "model.pretrained_url")
SIGNED = re.compile(r"(sign|direction|polarity|sgn|reverse)", re.I)
DEFAULT_FACTORS = (0.3, 3.0, 0.1, 0.03, 10.0)      # rungs spanning ~1.5 orders of magnitude (not just x1/2, x2)


def opt(ctx: Any, name: str) -> Any:
    from loop.core.levers import opt as o
    return o(ctx, name)


def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _notes(node: Any) -> str:
    return f" {getattr(node, 'notes', '') or ''}"


def is_child(node: Any) -> bool:
    """A ladder child of this module (never a ladder node itself)."""
    return TAG in _notes(node)


def knob_spec(ctx: Any, kind: str) -> dict[str, Any]:
    spec = opt(ctx, "struct_tune_knobs") or {}
    return dict(spec.get(kind) or {}) if isinstance(spec, Mapping) else {}


# ---------------------------------------------------------------------------------------------- classification
def own_changes(store: Any, nid: str, cfg: Mapping[str, Any], pcfg: Mapping[str, Any]) -> dict[str, Any]:
    """The config keys this node changed or introduced: its config vs its parent's when both are readable, else its
    stored ``config_diff``."""
    if cfg and pcfg:
        return {k: v for k, v in cfg.items() if json.dumps(v, sort_keys=True, default=str)
                != json.dumps(pcfg.get(k), sort_keys=True, default=str)}
    return dict(store.get_state(f"config_diff:{nid}") or {})


def is_backbone(node: Any, changes: Mapping[str, Any]) -> bool:
    return any(k in changes for k in BACKBONE_KEYS) or bool(re.search(r"(?:^|\s)init=\S", _notes(node)))


def dose_keys(ctx: Any, changes: Mapping[str, Any], cfg: Mapping[str, Any], rx: str, store: Any = None
              ) -> list[str]:
    """The node's own numeric config keys that a dose ladder may move (settable, not a backbone key, not a lever-group
    neutral key such as num_workers, leaf matching ``rx``), largest relative change first (a key the parent lacked: 1)."""
    from loop.core import mixed as MX
    from loop.extrapolate import _allowed
    out = []
    for k, v in changes.items():
        base = k.split("[", 1)[0]
        if k in BACKBONE_KEYS or not _num(v) or v == 0 or not _num(cfg.get(k)) or not _allowed(ctx, base) \
                or MX.neutral(ctx, k) or not re.search(rx, k.rsplit(".", 1)[-1]):
            continue
        out.append(k)
    return sorted(out, key=lambda k: (k.rsplit(".", 1)[-1] in ("seed",), k))


def kind_of(ctx: Any, store: Any, node: Any, cfg: Mapping[str, Any], pcfg: Mapping[str, Any]
            ) -> tuple[str, dict[str, Any]]:
    """('backbone' | 'loss' | 'other' | '', own changes) -- '' = not a structural node this module ladders."""
    changes = own_changes(store, node.id, cfg, pcfg)
    if is_child(node):
        return "", changes
    if is_backbone(node, changes):
        return "backbone", changes
    if ARCH_TAG in f"{_notes(node)} ":                    # 11-10: an operator arm with no backbone key (config-only port,
        return "other", changes                           # e.g. natinit.block_lr_mult) is still laddered (its `tune`)
    if " brief=program" in _notes(node):                   # a program keeps program.ladders' component ladder
        return "", changes
    s = str(getattr(node, "strategy", "") or "").lower().split(".")[-1]
    forced = RETRO_TAG in _notes(node)
    if not (s == "moonshot" or re.search(r"(?:^|\s)brief=bold(?:\s|$)", _notes(node)) or forced):
        return "", changes
    lrx = str(knob_spec(ctx, "loss").get("dose_rx") or r"(weight|coef|lambda|gamma)$")
    if dose_keys(ctx, changes, cfg, lrx):
        return "loss", changes
    return "other", changes


def structural_hint(node: Any) -> bool:
    """Cheap, config-free guess used by :func:`held` for a node with no ladder booked yet: a moonshot / bold / backbone
    (``init=``) / retro-port node that is no ladder child, program arm or program-ladder dose."""
    n = _notes(node)
    if is_child(node):
        return False
    if ARCH_TAG in f"{n} " and "program_arm=1" in n:      # an architecture arm of an operator program: held until tuned
        return True
    if "program_arm=1" in n or re.search(r"(?:^|\s)ladder=[A-Za-z]", n) and RETRO_TAG not in n:
        return False
    s = str(getattr(node, "strategy", "") or "").lower().split(".")[-1]
    return (s == "moonshot" and " brief=program" not in n) or bool(re.search(r"(?:^|\s)(?:brief=bold|init=\S|struct_retro=1)", n))


# ---------------------------------------------------------------------------------------------- ladder design
def _typed(x: float, cur: Any) -> Any:
    from loop.core.program import typed_like
    return typed_like(float(f"{x:.4g}") if isinstance(cur, float) else x, cur)


def _inert(row: Mapping[str, Any], cfg: Mapping[str, Any]) -> bool:
    """A knob row's ``unless: <key>`` names a config key that, when set (not null / 0 / false), makes the row's key a
    silent no-op (nanobrain: ``epochs`` overrides warmup_steps / decay_steps / total_steps): the row is skipped."""
    other = str(row.get("unless") or "")
    return bool(other) and cfg.get(other) not in (None, 0, 0.0, False, "")


def _axis(ctx: Any, row: Mapping[str, Any], cfg: Mapping[str, Any]) -> tuple[str, Any, Any, Any] | None:
    """(key, value, low, high) of one grid row against ``cfg``: ``factor`` -> value / f, value x f; ``step`` -> value -/+
    step clipped to [lo, hi] (at an edge the two levels both sit on the open side). None: not settable."""
    from loop.extrapolate import _allowed
    key = str(row.get("key") or "")
    cur = cfg.get(key)
    if not key or not _num(cur) or not _allowed(ctx, key.split("[", 1)[0]) or _inert(row, cfg):
        return None
    if row.get("factor"):
        f = float(row["factor"])
        if f <= 1 or cur <= 0:
            return None
        lo, hi = cur / f, cur * f
    else:
        st = float(row.get("step") or 0.0)
        rlo, rhi = float(row.get("lo", -math.inf)), float(row.get("hi", math.inf))
        if st <= 0:
            return None
        lo, hi = max(cur - st, rlo), min(cur + st, rhi)
        if hi <= cur + 1e-12:                        # at the top edge: both levels below
            lo, hi = max(cur - 2 * st, rlo), max(cur - st, rlo)
        elif lo >= cur - 1e-12:                      # at the bottom edge: both levels above
            lo, hi = min(cur + st, rhi), min(cur + 2 * st, rhi)
    lo, hi = _typed(lo, cur), _typed(hi, cur)
    if lo == hi or (lo == cur and hi == cur):
        return None
    return key, cur, lo, hi


def grid_points(ctx: Any, spec: Mapping[str, Any], cfg: Mapping[str, Any], budget: int
                ) -> list[tuple[dict[str, Any], str]]:
    """(levels, cell label) of the backbone-style design within ``budget`` trials: the first two settable ``grid`` rows
    as a 2 x 2 (corner order (hi, hi), (lo, lo), (hi, lo), (lo, hi): a budget of 2 is the diagonal), then one child per
    ``extra`` row (that key alone moved)."""
    axes = [a for r in (spec.get("grid") or ()) if isinstance(r, Mapping) and (a := _axis(ctx, r, cfg))][:2]
    out: list[tuple[dict[str, Any], str]] = []
    if len(axes) == 2:
        (k1, _, l1, h1), (k2, _, l2, h2) = axes
        for i, j in ((1, 1), (0, 0), (1, 0), (0, 1)):
            out.append(({k1: (l1, h1)[i], k2: (l2, h2)[j]}, f"{k1.rsplit('.', 1)[-1]}:{'lo hi'.split()[i]},"
                        f"{k2.rsplit('.', 1)[-1]}:{'lo hi'.split()[j]}"))
    elif len(axes) == 1:
        k1, _, l1, h1 = axes[0]
        out = [({k1: h1}, f"{k1.rsplit('.', 1)[-1]}:hi"), ({k1: l1}, f"{k1.rsplit('.', 1)[-1]}:lo")]
    out = out[:max(0, budget)]
    from loop.extrapolate import _allowed
    for r in (spec.get("extra") or ()):
        if len(out) >= budget:
            break
        key, cur = str(r.get("key") or ""), cfg.get(str(r.get("key") or ""))
        f = float(r.get("factor") or 0.0)
        if _num(cur) and cur > 0 and f > 0 and f != 1 and _allowed(ctx, key.split("[", 1)[0]) and not _inert(r, cfg):
            v = _typed(cur * f, cur)
            if v != cur:
                out.append(({key: v}, f"{key.rsplit('.', 1)[-1]}:x{f:g}"))
    return out


_ENG: list[Any] = []            # [(store, led, n_obs, n_nodes, monotonic ts, engagements)]: ONE entry
ENG_TTL_S = 30.0


def tick_engagements(store: Any, led: Any) -> list[Any]:
    """ONE :func:`loop.core.ideation.engagements` per ledger view (10-22 ops review: ``steer`` recomputed it per candidate,
    46 x 2.3 s on the live ledger, and ``catastrophic`` / ``held`` per node). Keyed on the store and ledger OBJECTS (a
    held reference, so an id cannot be reused) and their sizes; a view older than ``ENG_TTL_S`` is recomputed."""
    from loop.core.ideation import engagements
    key = (len(getattr(led, "obs", ()) or ()), len(getattr(led, "nodes", ()) or ()), engagements)
    now = time.monotonic()
    if _ENG and _ENG[0][0] is store and _ENG[0][1] is led and _ENG[0][2:5] == key and now - _ENG[0][5] < ENG_TTL_S:
        return _ENG[0][6]
    got = engagements(store, led)
    _ENG[:] = [(store, led, *key, now, got)]
    return got


def steer(ctx: Any, store: Any, led: Any, post: Any, nid: str) -> str:
    """How the engagement diagnostic steers the doses: ``reversed`` | ``not_engaged`` | ``engaged_negative`` | ``mixed``."""
    try:
        eng = [e for e in tick_engagements(store, led) if e.node == nid and e.status in
               ("engaged", "did not engage", "reversed")]
    except Exception:  # noqa: BLE001 - no diagnostics: no steering
        eng = []
    if not eng:
        return "mixed"
    status = {e.status for e in eng}
    if "reversed" in status:
        return "reversed"
    if "did not engage" in status:
        return "not_engaged"
    try:
        m, _ = post.belief(nid)
    except Exception:  # noqa: BLE001
        m = 0.0
    return "engaged_negative" if m < 0 else "mixed"


def ordered_factors(factors: Sequence[float], mode: str) -> list[float]:
    """The dose factors in trial order for a steering ``mode``."""
    fs = [float(f) for f in factors if _num(f) and f > 0 and f != 1] or list(DEFAULT_FACTORS)
    lower, upper = sorted((f for f in fs if f < 1), reverse=True), sorted(f for f in fs if f > 1)
    if mode in ("engaged_negative", "reversed"):
        return lower + upper
    if mode == "not_engaged":
        return upper + lower
    out = []                                         # alternate: .3, 3, .1, 10, ...
    for i in range(max(len(lower), len(upper))):
        out += lower[i:i + 1] + upper[i:i + 1]
    return out


def dose_points(ctx: Any, spec: Mapping[str, Any], keys: Sequence[str], cfg: Mapping[str, Any], budget: int, mode: str
                ) -> list[tuple[dict[str, Any], str]]:
    """Doses of the node's own knobs: the budget split across <= 2 keys (the first gets the extra one), each at the
    steered factors; a ``reversed`` node's signed knob gets its sign flip first."""
    from loop.core.program import _ok_dose, allocate
    fs = ordered_factors(spec.get("factors") or DEFAULT_FACTORS, mode)
    keys = [k for k in keys if _num(cfg.get(k))][:2]
    if not keys:
        return []
    shares = allocate([1.0] * len(keys), budget)
    out: list[tuple[dict[str, Any], str]] = []
    for key, n in zip(keys, shares):
        cur, got = cfg[key], []
        if mode == "reversed" and (SIGNED.search(key.rsplit(".", 1)[-1]) or cur < 0) and n > 0:
            flip = _typed(-cur, cur)
            if flip != cur:
                got.append(({key: flip}, f"{key.rsplit('.', 1)[-1]}:flip"))
        for f in fs:
            if len(got) >= n:
                break
            v = _typed(cur * f, cur)
            if v != cur and _ok_dose(ctx, key, v, cur) and all(v != next(iter(g[0].values())) for g in got):
                got.append(({key: v}, f"{key.rsplit('.', 1)[-1]}:x{f:g}"))
        out += got[:n]
    if len(keys) >= 2 and budget >= 4 and len(out) >= budget:   # COUPLED knobs: the diff's own keys moved together once
        f = fs[0]
        both = {k: _typed(cfg[k] * f, cfg[k]) for k in keys}
        if all(v != cfg[k] for k, v in both.items()):
            out = out[:budget - 1] + [(both, "+".join(k.rsplit(".", 1)[-1] for k in keys) + f":x{f:g}")]
    return out[:budget]


def explicit_points(spec: Mapping[str, Any], cfg: Mapping[str, Any], budget: int) -> list[tuple[dict[str, Any], str]]:
    """The operator's per-node ladder (``struct_retro_nodes: [{node, knobs: {key: [values]}}]``): one child per listed value,
    round-robin over the keys, typed like the config's value; a key the node's config lacks is skipped."""
    if spec.get("points"):                                   # multi-key points: [{key: value, ...}], one child each
        out = []
        for p in spec["points"]:
            if not isinstance(p, Mapping) or not p:
                continue
            lv = {k: _typed(v, cfg[k]) for k, v in p.items() if _num(v) and _num(cfg.get(k))}
            if len(lv) == len(p) and any(lv[k] != cfg[k] for k in lv):
                out.append((lv, "+".join(f"{k.rsplit('.', 1)[-1]}={lv[k]:g}" for k in lv)))
        return out[:budget]
    cols = [(k, [v for v in vals if _num(v)]) for k, vals in (spec.get("knobs") or {}).items()
            if isinstance(vals, Sequence) and _num(cfg.get(k))]
    out: list[tuple[dict[str, Any], str]] = []
    for i in range(max((len(v) for _, v in cols), default=0)):
        for k, vals in cols:
            if i < len(vals) and len(out) < budget and (v := _typed(vals[i], cfg[k])) != cfg[k]:
                out.append(({k: v}, f"{k.rsplit('.', 1)[-1]}={v:g}"))
    return out


# ---------------------------------------------------------------------------------------------- the ladder
def book(store: Any) -> dict[str, Any]:
    return dict(store.get_state(K_STRUCT) or {})


def spent(store: Any, nid: str) -> int:
    """Struct children already built on ``nid`` (booked)."""
    rec = book(store).get(nid)
    return len(rec.get("children") or ()) if isinstance(rec, Mapping) else 0


def catastrophic(ctx: Any, store: Any, led: Any, post: Any, nid: str) -> str:
    """Why the node's FIRST point is catastrophic ('' = it is not): its paired mean is below -``struct_cata_z`` sd on >=
    ``struct_cata_seeds`` seeds AND the mechanism engaged (an unengaged node is an unproven mechanism, not a failed one)."""
    from loop.core import model as M
    z, k = float(opt(ctx, "struct_cata_z")), int(opt(ctx, "struct_cata_seeds"))
    try:
        m, v = post.belief(nid)
        n = int(getattr(post.est[nid].get(M.COMPOSITE), "n", 0))
    except Exception:  # noqa: BLE001
        return ""
    if n < k or v <= 0 or m >= -z * math.sqrt(v):
        return ""
    try:
        engaged = any(e.node == nid and e.status == "engaged" for e in tick_engagements(store, led))
    except Exception:  # noqa: BLE001
        engaged = False
    return f"first point {m:+.4f} < -{z:g} sd on {n} seeds with the mechanism engaged" if engaged else ""


def cell_tag(cell: str, i: int) -> str:
    """The mechanism-tag suffix of ladder point ``i``: its cell (``block_lr_mult=0.5`` -> ``block_lr_mult_0p5``), so
    the points' tags say what each varies (11-10: a node-a node were ``<arm>_s0..s3``, read as replicates of the arm);
    ``s<i>`` when the cell is empty."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", re.sub(r"(?<=\d)\.(?=\d)", "p", str(cell or ""))).strip("_")
    return slug[:60] or f"s{i}"


def build(ctx: Any, store: Any, adapter: Any, sn: Any, kind: str, cfg: Mapping[str, Any],
          points: Sequence[tuple[dict[str, Any], str]], now: float) -> list[str]:
    from loop.core import factorial as F
    from loop.core.program import _child
    made: list[str] = []
    for i, (levels, cell) in enumerate(points):
        knobs = [F.Knob(k, cfg.get(k), v, "lin") for k, v in levels.items()]
        sha, err = F.build_point(adapter, sn.commit, levels, knobs, {}, f"struct-{sn.id}-{i}",
                                 getattr(adapter, "config_rel", "config.yaml"))
        if not sha or sha == sn.commit:
            store.log("struct_ladder_skip", node_id=sn.id, cell=cell, why=err or "no change")
            continue
        made.append(_child(store, sn, sha, now, cell_tag(cell, i), f"ladder={sn.id}{TAG[:-1]}={kind} cell={cell}",
                           dict(levels)))
    return made


K_NONSTRUCT = "ideation.struct_nonstruct"   # {node: signature}: nodes already judged "not a structural node" (never re-read)
K_WAIT = "ideation.struct_wait"             # {node: first ts}: structural nodes waiting for a ladder slot (held() reads it)
CFG_CACHE_MAX = 256


def _cfg_at(adapter: Any, commit: str) -> dict[str, Any]:
    """The flattened config at ``commit`` (a commit's config never changes), memoised on the adapter (a small FIFO)."""
    from loop.adapters import flatten
    cache = getattr(adapter, "_struct_cfg", None)
    if cache is None:
        cache = {}
        try:
            adapter._struct_cfg = cache
        except AttributeError:
            pass
    if commit not in cache:
        if len(cache) >= CFG_CACHE_MAX:
            cache.pop(next(iter(cache)))
        cache[commit] = flatten(adapter.config_at(commit))
    return cache[commit]


def _signature(node: Any) -> str:
    """What :func:`kind_of` reads of a node besides its configs (notes, strategy): the memo key of a negative verdict."""
    import hashlib
    return hashlib.sha1(f"{_notes(node)}|{getattr(node, 'strategy', '')}".encode()).hexdigest()[:16]


def maybe_structural(store: Any, node: Any) -> bool:
    """Cheap pre-filter run BEFORE any config is read (10-22 ops review: 520 derived-SCREENING nodes each cost two
    ``git show`` calls before :func:`kind_of` said "not structural"). False only when :func:`kind_of` cannot say anything
    else: no retro / arch tag, no ``init=`` note, not bold, not a moonshot, and a stored non-empty ``config_diff`` that
    names no backbone key. A missing or empty stored diff is unknown: the config decides."""
    n = f"{_notes(node)} "
    if RETRO_TAG in n or ARCH_TAG in n or re.search(r"(?:^|\s)(?:init=\S|brief=bold(?:\s|$))", n):
        return True
    if str(getattr(node, "strategy", "") or "").lower().split(".")[-1] == "moonshot":
        return True
    ch = store.get_state(f"config_diff:{node.id}")
    return not ch or any(k in ch for k in BACKBONE_KEYS)


def ladder_state(store: Any, rec: Mapping[str, Any], by: Mapping[str, Any], need: int, seeds: int, hold_s: float,
                 now: float) -> tuple[int, int, bool]:
    """(finished tuned points, wanted, still running) of a booked ladder: running = fewer than ``want`` points finished,
    some child neither finished nor dead, and not past ``struct_hold_h`` since its first child started. A moonshot
    RESCUE ladder (``rescue``: loop.core.rescue) counts a point at its own ``seeds`` and wants ALL its points."""
    kids = rec.get("children") or ()
    if rec.get("rescue"):
        seeds, need = int(rec.get("seeds") or seeds), int(rec.get("need") or need)
    fin = sum(1 for c in kids if len(by.get(c) or ()) >= seeds)
    dead = sum(1 for c in kids if (n := store.get_node(c)) is not None and not _nodes_ok(n))
    want = min(need, len(kids))
    t0 = _first_start(store, kids) or float(rec.get("ts") or 0.0)
    return fin, want, fin < want and fin + dead < len(kids) and not now - t0 > hold_s


def operator_arm(node: Any) -> bool:
    """An operator program's architecture arm (``arch_arm=1``): its ladder is the operator's own and is never capped.
    A moonshot rescue (``moonshot_rescue=``, loop.core.rescue) is not capped here either: its own cap
    (``moonshot_rescue_max``) bounds it."""
    return node is not None and (ARCH_TAG in f"{_notes(node)} " or rescued(node))


RESCUE_TAG = " moonshot_rescue="     # a node the moonshot rescue tunes (loop.core.rescue): its own ladder rules


def rescued(node: Any) -> bool:
    return node is not None and RESCUE_TAG in _notes(node)


def open_ladders(ctx: Any, store: Any, led: Any, now: float) -> int:
    """Booked ladders still running (:func:`ladder_state`): the count ``struct_ladder_max_open`` caps. An operator arm's
    ladder (:func:`operator_arm`) is outside the cap and not counted (11-10: a node's ladder waited behind six bold ones)."""
    by = led.by_node() if hasattr(led, "by_node") else {}
    need, seeds, hold_s = int(opt(ctx, "struct_min_points")), int(opt(ctx, "struct_min_seeds")), \
        float(opt(ctx, "struct_hold_h")) * 3600.0
    return sum(1 for nid, rec in book(store).items() if isinstance(rec, Mapping) and (rec.get("children") or ())
               and not operator_arm(led.nodes.get(nid) if hasattr(led, "nodes") else store.get_node(nid))
               and ladder_state(store, rec, by, need, seeds, hold_s, now)[2])


def _priority(node: Any, m: float) -> tuple[int, float]:
    """Waiting order for a ladder slot: operator architecture arms, then retro ports, then the rest by posterior."""
    n = f"{_notes(node)} "
    return (0 if ARCH_TAG in n else 1 if RETRO_TAG in n else 2, -float(m))


def struct_ladders(ctx: Any, store: Any, adapter: Any, led: Any, post: Any, statuses: Mapping[str, Any], now: float
                   ) -> list[str]:
    """Build the structural ladders owed (module docstring); returns the tick's action lines. At most
    ``struct_ladder_max_open`` ladders run at once: a structural node past the cap WAITS (``K_WAIT``; unbooked, its
    ``held`` clock stopped) and is laddered in priority order (:func:`_priority`) as slots free. An operator arm
    (:func:`operator_arm`) is outside the cap: laddered at once, taking no slot. A config-only arm's untuned first shot
    is still skipped when hopeless (<= -``struct_tune_min_z`` sd and <= ``struct_tune_flat_floor``, no engagement)."""
    from loop.core.acquire import program_child, wave_node
    from loop.core.program import K_LADDERS
    if not opt(ctx, "struct_tune") or post is None or not getattr(led, "epoch", ""):
        return []
    bk, acts, prog = book(store), [], store.get_state(K_LADDERS) or {}
    total, min_z = int(opt(ctx, "struct_tune_max")), float(opt(ctx, "struct_tune_min_z"))
    flat = opt(ctx, "struct_tune_flat_floor")        # 11-09: an absolute 'flat-but-not-dead' floor (None: the z rule)
    flat = None if flat is None else float(flat)
    memo, mdirty = dict(store.get_state(K_RECHECK) or {}) if flat is not None else {}, False
    nons, ndirty = dict(store.get_state(K_NONSTRUCT) or {}), False
    from loop.core import rescue as RS
    claimed = RS.claimed(store)                      # 11-10: the moonshot rescue decides these (port / debug first)
    ready: list[tuple[Any, ...]] = []
    for nid, node in led.nodes.items():
        st = getattr(statuses.get(nid), "value", statuses.get(nid))
        notes = f"{_notes(node)} "
        forced = RETRO_TAG in notes or ARCH_TAG in notes
        closed = str(st).lower() in ("screened_out", "not_confirmed") and RETRO_TAG in notes   # a retro-tuned closed moonshot
        recheck = flat is not None and isinstance(bk.get(nid), str) and bk[nid].startswith(SKIP_FIRST)
        if recheck and nid in getattr(post, "est", {}):     # review: re-read configs only when the belief moved
            m0 = round(float(post.belief(nid)[0]), 6)
            if memo.get(nid) == m0:
                continue
            memo[nid], mdirty = m0, True
        if (nid in bk and not recheck) or node.base != led.epoch or (str(st).lower() != "screening" and not closed) \
                or nid not in getattr(post, "est", {}) or is_child(node) or program_child(node) and not forced \
                or wave_node(node) or isinstance(prog.get(nid), list) or (nid in claimed and not rescued(node)):
            continue
        sig = _signature(node)
        if nons.get(nid) == sig and not forced:
            continue                                   # judged not structural already: no config is read again (an
            #   operator arm / retro port is always re-read: 11-10 a node was memoised "not structural" by an older rule)
        if not maybe_structural(store, node):
            nons[nid], ndirty = sig, True
            continue
        sn = store.get_node(nid)
        if sn is None or not hasattr(adapter, "config_at"):
            continue
        try:
            cfg = _cfg_at(adapter, sn.commit)
            par = store.get_node(node.parent or sn.parent_id or "")
            pcfg = _cfg_at(adapter, par.commit) if par is not None else {}
        except Exception:  # noqa: BLE001 - unreadable config: re-planned on a later tick, never booked
            acts.append(f"struct ladder {nid}: its config is unreadable now; re-planned on a later tick")
            continue
        kind, changes = kind_of(ctx, store, node, cfg, pcfg)
        if not kind:
            if not forced:
                nons[nid], ndirty = sig, True
            continue
        m, v = post.belief(nid)
        mode = steer(ctx, store, led, post, nid) if kind != "backbone" else "mixed"
        # review: the floor only RELAXES the z rule (a noisy -.006 first shot is still laddered, as before)
        hopeless = m <= -min_z * math.sqrt(max(v, 0.0)) and (flat is None or m <= flat)
        if kind != "backbone" and mode == "mixed" and hopeless and not rescued(node):    # a rescue judged it already
            # no diagnostic to say the mechanism engaged: a hopeless first shot is not laddered. BOOKED so the node is no
            # longer "owed" a ladder (held() would otherwise keep it open forever)
            if recheck:                                # still hopeless: its earlier booking stands (no log per tick)
                continue
            bound = f"-{min_z:g} sd" + (f" and <= {flat:+.4f} (struct_tune_flat_floor)" if flat is not None else "")
            bk[nid] = (f"{SKIP_FIRST} {m:+.4f} <= {bound} with no engagement diagnostic; "
                       "a hopeless first shot is not laddered")
            store.set_state(K_STRUCT, bk)
            store.log("struct_ladder", node_id=nid, kind=kind, why=bk[nid])
            acts.append(f"struct ladder {nid} ({kind}): {bk[nid][3:]}")
            continue
        if recheck:                                    # passed: unbooked, so held() owes it a ladder while it waits
            bk.pop(nid, None)
            store.set_state(K_STRUCT, bk)
        if why := catastrophic(ctx, store, led, post, nid):
            bk[nid] = f"-: catastrophic: {why}"
            store.set_state(K_CATA, {**(store.get_state(K_CATA) or {}), nid: why})
            store.set_state(K_STRUCT, bk)
            store.log("struct_ladder", node_id=nid, kind=kind, why=bk[nid])
            acts.append(f"struct ladder {nid} ({kind}): {why}: closed without a ladder")
            continue
        ready.append((nid, node, sn, kind, changes, cfg, mode, m, v))
    if ndirty:
        store.set_state(K_NONSTRUCT, nons)
    if mdirty:
        store.set_state(K_RECHECK, memo)
    waiting: list[str] = []
    slots = (int(opt(ctx, "struct_ladder_max_open")) - open_ladders(ctx, store, led, now)) if ready else 0
    for nid, node, sn, kind, changes, cfg, mode, m, v in sorted(ready, key=lambda r: (_priority(r[1], r[7]), r[0])):
        capped = not operator_arm(node)                # an operator arm's ladder never waits for a slot nor takes one
        if capped and slots <= 0:
            waiting.append(nid)
            continue
        room, resc = total, rescued(node)
        if resc:                                       # a rescue's own budget (port + rungs <= 5), not the bold unit's
            room = int(opt(ctx, "moonshot_rescue_rungs"))
        else:
            try:
                from loop.core import bold as BOLD
                if (r := BOLD.tune_room(ctx, store, sn)) is not None:
                    room = min(room, r)
            except Exception:  # noqa: BLE001 - no bold view: the ladder's own cap
                pass
        points: list[tuple[dict[str, Any], str]] = []
        rspec = retro_spec(ctx, store, node)
        if rspec and (points := explicit_points(rspec, cfg, room)):
            kind = f"{kind}+{'rescue' if resc else 'operator'}"
        elif rspec:                                    # review 11-10: say when the operator's own points were not usable
            acts.append(f"struct ladder {nid}: its operator tune {dict(rspec)} gives no buildable point "
                        f"(keys absent / unchanged / room {room}); the {kind} default ladder is used")
        elif kind == "backbone":
            points = grid_points(ctx, knob_spec(ctx, "backbone"), cfg, room)
        else:
            spec = knob_spec(ctx, kind)
            rx = str(spec.get("dose_rx") or ".")
            keys = dose_keys(ctx, changes, cfg, rx)
            points = dose_points(ctx, spec, keys, cfg, room, mode) if keys else []
            if not points:                             # no own knob to dose: lr x layer decay for the new structure
                points = grid_points(ctx, dict(spec.get("fallback") or knob_spec(ctx, "backbone")), cfg, room)
        if not points:
            bk[nid] = "-: no tunable knob (no dose key, lr / layer decay not settable)"
            store.set_state(K_STRUCT, bk)
            acts.append(f"struct ladder {nid} ({kind}): {bk[nid][3:]}")
            continue
        made = build(ctx, store, adapter, sn, kind, cfg, points, now)
        bk[nid] = {"kind": kind, "children": made, "steer": mode, "ts": now, "need": min(
            int(opt(ctx, "struct_min_points")), len(made))} if made else "-: nothing built"
        if made and resc:                              # every rung at the rescue's own screen seeds (ladder_state)
            bk[nid].update(rescue=True, seeds=int(opt(ctx, "moonshot_rescue_seeds")), need=len(made))
        store.set_state(K_STRUCT, bk)
        store.log("struct_ladder", node_id=nid, kind=kind, steer=mode, nodes=made, cells=[c for _, c in points])
        acts.append(f"struct ladder {nid} ({kind}, steer {mode}): {len(made)} tuning trial(s) {made}"
                    + ("" if capped else " (operator arm: outside struct_ladder_max_open)"))
        if made and capped:
            slots -= 1
    old = store.get_state(K_WAIT) or {}
    wait = {n: float(old.get(n) or now) for n in waiting}
    if wait != {k: float(x) for k, x in old.items()}:
        store.set_state(K_WAIT, wait)
        if waiting:
            acts.append(f"struct ladder: {len(waiting)} structural node(s) wait for a ladder slot "
                        f"(struct_ladder_max_open {int(opt(ctx, 'struct_ladder_max_open'))}): {waiting[:12]}")
    return acts


K_PM = "ideation.struct_postmortem"   # {node: postmortem id}


def report(ctx: Any, store: Any, led: Any, post: Any, now: float) -> list[str]:
    """The v3 POSTMORTEM, computed (no LLM): postmortems stopped on 10-06 because the Opus postmortem was removed
    (3b748f8) and v3 never wrote one. When a structural ladder has finished (``struct_min_points`` points at
    ``struct_min_seeds`` seeds, or every point resolved), write ONE ``postmortems`` row for the node: each tuned point's
    paired increment, the best point shrunk for the max of K (``struct_tune_beta``) and the verdict on that best TUNED
    point (never the raw max); ``under_tuned`` is False once the ladder is in."""
    from loop.contracts import LeverTags, Postmortem
    from loop.core import model as M
    if not hasattr(led, "by_node"):
        return []
    done, acts = dict(store.get_state(K_PM) or {}), []
    by = led.by_node()
    need, seeds = int(opt(ctx, "struct_min_points")), int(opt(ctx, "struct_min_seeds"))
    for nid, rec in book(store).items():
        if nid in done or not isinstance(rec, Mapping) or not (kids := rec.get("children") or ()):
            continue
        k_seeds, k_need = (int(rec.get("seeds") or seeds), int(rec.get("need") or need)) if rec.get("rescue") \
            else (seeds, need)                         # a rescue ladder: its own seeds per point, every point
        fin = [c for c in kids if len(by.get(c) or ()) >= k_seeds]
        dead = [c for c in kids if (n := store.get_node(c)) is not None and not _nodes_ok(n)]
        if len(fin) < min(k_need, len(kids)) and len(fin) + len(dead) < len(kids):
            continue
        inc = ""
        try:
            from loop.core import bar as BAR
            inc = BAR.incumbent_of(led)
        except Exception:  # noqa: BLE001 - no incumbent view: the over-parent frame only
            inc = ""

        def shifted(c: str, m: float, v: float) -> tuple[float, float]:
            """A ladder child's belief is its increment over its (untuned) PARENT; the verdict is in the incumbent frame
            (parent - incumbent added, as loop.core.acquire.ledger_options does)."""
            try:
                dm, dv = post.shift(c, M.COMPOSITE, inc) if inc else (0.0, 0.0)
            except Exception:  # noqa: BLE001 - not shiftable (no incumbent run / no posterior): parent frame
                dm, dv = 0.0, 0.0
            return m + dm, v + dv

        rows = []                  # (id, over-parent m, sd, notes, over-incumbent m, sd)
        for c in fin:
            try:
                m, v = post.belief(c)
            except Exception:  # noqa: BLE001 - a point with no posterior yet
                continue
            mi, vi = shifted(c, m, v)
            rows.append((c, m, math.sqrt(max(v, 0.0)), (store.get_node(c).notes or ""), mi, math.sqrt(max(vi, 0.0))))
        try:
            m0, v0 = post.belief(nid)
        except Exception:  # noqa: BLE001
            m0, v0 = 0.0, 0.0
        best = max(rows, key=lambda r: r[4], default=None)
        beta = float(opt(ctx, "struct_tune_beta"))
        sel = M.max_of_k(best[4], best[5] ** 2, max(len(rows), len(kids)), beta) if best else None
        verdict = "null" if best is None or best[4] >= -best[5] else "loss"
        cell = lambda notes: (re.search(r"cell=(\S+)", notes) or [None, "?"])[1]        # noqa: E731
        table = "; ".join(f"{c} [{cell(nt)}] over untuned {m:+.4f} +- {sd:.4f} | over incumbent {mi:+.4f} +- {sdi:.4f}"
                          for c, m, sd, nt, mi, sdi in rows)
        text = (f"structural ladder ({rec.get('kind')}, steer {rec.get('steer')}) of {nid}: untuned point {m0:+.4f} +- "
                f"{math.sqrt(max(v0, 0.0)):.4f} (over its parent); {len(rows)} tuned point(s) finished ({len(dead)} dead); "
                f"each point is shown over the UNTUNED node and over the INCUMBENT, and the verdict uses the incumbent "
                f"frame: {table or 'none'}. "
                + (f"Best tuned point {best[0]} {best[4]:+.4f} over the incumbent (selection-shrunk for the max of "
                   f"{max(len(rows), len(kids))}: {sel:+.4f}); it is judged on fresh test seeds, never on this raw best."
                   if best else "No tuned point has a posterior."))
        store.put_postmortem(Postmortem(
            id=f"pm-struct-{nid}", task_id=store.task_id, node_id=nid, verdict=verdict, explanation=text,
            mechanism_reasoning="computed (loop.core.struct_tune.report): no LLM postmortem since 10-06 (3b748f8)",
            levers=LeverTags(), suggested_next=(f"test {best[0]} on fresh seeds" if best and best[4] > 0 else
                                           "closed after its tuning ladder"), created_ts=now, under_tuned=False,
            tune_knobs=json.dumps([{"key": k} for c in fin for k in (store.get_state(f"config_diff:{c}") or {})][:8])))
        done[nid] = f"pm-struct-{nid}"
        store.set_state(K_PM, done)
        store.log("struct_postmortem", node_id=nid, verdict=verdict, tuned=len(rows), dead=len(dead))
        acts.append(f"struct postmortem {nid}: {verdict} after {len(rows)} tuned point(s)")
    return acts


# ---------------------------------------------------------------------------------------------- the close rule
def _ts(store: Any, nid: str, node: Any) -> float:
    sn = store.get_node(nid) if hasattr(store, "get_node") else None
    return float(getattr(node, "created_ts", 0.0) or getattr(sn, "created_ts", 0.0) or 0.0)


def _eligible(store: Any, nid: str, node: Any, post: Any = None) -> bool:
    """May this unbooked node still get a ladder (:func:`struct_ladders`)? Its status is not terminal (a closed node
    that was never laddered is not owed one, except a retro-tuned closed moonshot), and, when the posterior is known, it
    has a posterior (``post.est``)."""
    from loop.contracts import CLOSED_STATUSES
    sn = store.get_node(nid) if hasattr(store, "get_node") else None
    st = getattr(sn, "status", None) or getattr(node, "status", "")
    st = str(getattr(st, "value", st) or "").lower()
    if st in {s.value for s in CLOSED_STATUSES} and RETRO_TAG not in _notes(node):
        return False
    return post is None or nid in getattr(post, "est", {})


def _first_start(store: Any, kids: Sequence[str]) -> float:
    """When the first ladder child was SUBMITTED (0.0 = none has started)."""
    ts = []
    for c in kids:
        try:
            ts += [float(r.submitted_ts) for r in store.runs(node_id=c) if getattr(r, "submitted_ts", None)]
        except Exception:  # noqa: BLE001 - a store view without runs
            continue
    return min(ts, default=0.0)


def held(ctx: Any, store: Any, led: Any, now: float | None = None, post: Any = None) -> dict[str, str]:
    """node -> why it may not be CLOSED as negative yet: a structural node of the current base with no ladder booked
    (owed) that is still ladder-eligible (:func:`_eligible`: not terminal, in ``post.est`` when ``post`` is given), or with
    fewer than ``struct_min_points`` tuned points FINISHED (a point counts at >= ``struct_min_seeds`` finished seeds: one
    run's SE is ~2.3 mDelta). Released by a catastrophic first point (re-evaluated on EVERY call when ``post`` is given, so
    a node that later reaches ``struct_cata_seeds`` seeds exits), a ladder that could not be built or was skipped (booked
    ``-:``), enough finished points, every point resolved (a crashed / void point is dead, not tuned), or ``struct_hold_h``
    hours: after the node was created for an UNBOOKED node, after the first ladder child STARTED (its build time while none
    has) for a booked ladder (a stuck ladder never holds a node forever)."""
    try:
        if not opt(ctx, "struct_tune") or not getattr(led, "epoch", ""):
            return {}
        bk, by = book(store), led.by_node()
        need, seeds, now = int(opt(ctx, "struct_min_points")), int(opt(ctx, "struct_min_seeds")), \
            time.time() if now is None else now
        hold_s = float(opt(ctx, "struct_hold_h")) * 3600.0
        wait = store.get_state(K_WAIT) or {}              # waiting for a ladder slot: the hold clock does not run
        out = {}
        for nid, node in led.nodes.items():
            if node.base != led.epoch:
                continue
            rec = bk.get(nid)
            if rec is None:
                waits = nid in wait
                if not ((structural_hint(node) or waits) and _nodes_ok(node) and _eligible(store, nid, node, post)):
                    continue
                if not waits and (t0 := _ts(store, nid, node)) > 0 and now - t0 > hold_s:
                    continue                                    # never booked, never will be: not owed forever
                if post is not None and catastrophic(ctx, store, led, post, nid):
                    continue
                out[nid] = "ladder waiting for a ladder slot (struct_ladder_max_open)" if waits else "ladder owed"
            elif isinstance(rec, Mapping):
                fin, want, running = ladder_state(store, rec, by, need, seeds, hold_s, now)
                if running and not (post is not None and catastrophic(ctx, store, led, post, nid)):
                    out[nid] = f"ladder {fin}/{want} tuned points finished (>= {rec.get('seeds') or seeds} seeds each)"
        from loop.core import rescue as RS               # 11-10: a moonshot under rescue (debug / port / ladder)
        for nid, why in RS.owed(store).items():
            if nid in led.nodes and nid not in out:
                out[nid] = why
        return out
    except Exception:  # noqa: BLE001 - a view without the fields: nothing is held
        return {}


def _nodes_ok(node: Any) -> bool:
    """Not a dead node. 11-10 fix: a STORE node's status is a NodeStatus enum whose str() is 'NodeStatus.VOID', so a
    rejected / void ladder child was never counted dead (ladder_state / report waited out struct_hold_h)."""
    st = getattr(node, "status", "") or ""
    return str(getattr(st, "value", st)).lower() not in ("rejected", "void", "crashed")


# ---------------------------------------------------------------------------------------------- retro
def retro_items(ctx: Any) -> list[tuple[str, dict[str, Any]]]:
    """The operator's ``struct_retro_nodes`` as (node id, spec): an entry is a node id, or {node, knobs: {key: [values]}}
    (the ladder's exact values for that node)."""
    out = []
    for it in opt(ctx, "struct_retro_nodes") or ():
        if isinstance(it, Mapping) and it.get("node"):
            out.append((str(it["node"]), dict(it)))
        elif isinstance(it, str) and it:
            out.append((it, {}))
    return out


def retro_spec(ctx: Any, store: Any, node: Any) -> dict[str, Any]:
    """The operator's per-node ladder spec of a retro port (its ``ladder=<old>`` note names the original), or of an
    architecture arm (``arch_arm=1``: its operator program arm's ``tune`` dict, ``{points: [...]}`` / ``{knobs: {...}}``)."""
    n = f"{_notes(node)} "
    if ARCH_TAG in n:
        m = re.search(r"(?:^|\s)op=(\S+) arm=(\S+)", n)
        for p in (opt(ctx, "operator_programs") or ()) if m else ():
            if isinstance(p, Mapping) and str(p.get("id")) == m.group(1):
                arm = next((a for a in p.get("arms") or () if isinstance(a, Mapping) and str(a.get("id")) == m.group(2)), {})
                return dict(arm.get("tune") or {})
        return {}
    if RETRO_TAG not in _notes(node):
        return {}
    if rescued(node):                                    # a moonshot rescue: the knobs its researcher declared
        from loop.core import rescue as RS
        return RS.spec_of(store, node)
    m = re.search(r"(?:^|\s)ladder=(\S+)", _notes(node))
    rec = (store.get_state(K_RETRO) or {}).get(m.group(1)) if m else None
    return dict(rec.get("spec") or {}) if isinstance(rec, Mapping) else {}


def retro_targets(ctx: Any, store: Any, led: Any) -> list[tuple[str, str]]:
    """(old node, why): the operator's explicit ``ideation.struct_retro_nodes`` first, then old-epoch backbone swaps with
    >= 1 finished paired run (never run / rejected ones are no measured negative)."""
    by = led.by_node()
    out = [(n, "operator list") for n, _ in retro_items(ctx)]
    seen = {n for n, _ in out}
    for nid, node in led.nodes.items():
        if nid in seen or node.base == led.epoch or node.port_of or not by.get(nid):
            continue
        ch = store.get_state(f"config_diff:{nid}") or {}
        if is_backbone(node, ch) and str(node.status or "").lower() not in ("rejected", "void", "crashed"):
            out.append((nid, "backbone swap judged on its old base"))
    return out


def retro_open(store: Any, led: Any) -> int:
    """Retro ports whose ladder is not finished (not yet laddered, or its children still running)."""
    by, bk, n = led.by_node(), book(store), 0
    for rec in (store.get_state(K_RETRO) or {}).values():
        if not isinstance(rec, Mapping):
            continue
        sr = bk.get(rec.get("port"))
        if isinstance(sr, str):
            continue
        if not isinstance(sr, Mapping) or any(not by.get(c) for c in sr.get("children") or ()):
            n += 1
    return n


def retro(ctx: Any, store: Any, adapter: Any, led: Any, now: float) -> list[str]:
    """Port the retro targets onto the current base (``cutover.port_node``), tagged for program priority; the ladder
    itself is built by :func:`struct_ladders` once the port screens."""
    if not opt(ctx, "struct_retro") or not getattr(led, "epoch", "") or not hasattr(adapter, "graft_port"):
        return []
    from loop import cutover as CO
    rb, acts, specs = dict(store.get_state(K_RETRO) or {}), [], dict(retro_items(ctx))
    room = int(opt(ctx, "struct_retro_max")) - retro_open(store, led)
    for old, why in retro_targets(ctx, store, led):
        if old in rb or room <= 0:
            continue
        if old not in led.nodes:
            rb[old] = "-: not in the ledger"
            continue
        if led.nodes[old].base == led.epoch:                 # already on the current base: tune it in place, no port
            pid = old
            if RETRO_TAG not in _notes(led.nodes[old]) and (on := store.get_node(old)) is not None:
                store.update_node(old, notes=f"{on.notes or ''} ladder={old}{RETRO_TAG}".strip())
            if (on := store.get_node(old)) is not None and on.status.value == "rejected":
                rb[old] = f"-: node rejected ({on.reject_reason})"
                acts.append(f"struct retro {old}: {rb[old][3:]}")
                continue
            rb[old] = {"port": old, "ts": now, "why": why, **({"spec": specs[old]} if specs.get(old) else {})}
            room -= 1
            store.log("struct_retro", node_id=old, port=old, why=why)
            acts.append(f"struct retro {old}: on the current base already; tuned in place ({why})")
            continue
        if (existing := led.port(old)) is not None:          # ported already (by a tick): tag it, never re-port
            pid = existing
        else:
            got = CO.port_node(store, adapter, led, old, now)
            if not getattr(got, "node", ""):
                rb[old] = f"-: no port ({getattr(got, 'how', '')})"
                acts.append(f"struct retro {old}: {rb[old][3:]}")
                continue
            pid = got.node
        pn = store.get_node(pid)
        if pn is not None and RETRO_TAG not in _notes(pn):
            store.update_node(pid, notes=f"{pn.notes or ''} ladder={old}{RETRO_TAG}".strip())
        if pn is not None and pn.status.value == "rejected":
            rb[old] = f"-: port rejected ({pn.reject_reason})"
            acts.append(f"struct retro {old}: {rb[old][3:]}")
            continue
        rb[old] = {"port": pid, "ts": now, "why": why, **({"spec": specs[old]} if specs.get(old) else {})}
        room -= 1
        store.log("struct_retro", node_id=old, port=pid, why=why)
        acts.append(f"struct retro {old}: ported as {pid} ({why}); its tuning ladder follows once it screens")
    store.set_state(K_RETRO, rb)
    return acts


# ---------------------------------------------------------------------------------------------- intake
def untunable(ctx: Any, cfg_diff: Mapping[str, Any], declared: Sequence[Any] | None, notes: str = "") -> tuple[str, str]:
    """('untunable_moonshot', fix) for a bold / moonshot candidate that gives the ladder nothing to tune: its diff adds /
    changes no numeric config key, it is no backbone swap and its finish declares no ``components`` / ``knobs``."""
    if declared or any(k in cfg_diff for k in BACKBONE_KEYS) or re.search(r"(?:^|\s)init=\S", notes or ""):
        return "", ""
    if any(_num(v) and v != 0 for k, v in cfg_diff.items() if k not in BACKBONE_KEYS):
        return "", ""
    return ("untunable_moonshot",
            "a structural change is tuned before it is judged (up to 5 trials over its knobs), but your diff adds no "
            "numeric config key (loss weight, ratio, temperature, dropout ...). Expose its strength as a config key "
            "(default = the value you built), or declare `components` in finish: [{family, mechanism, share, knobs: "
            "[{key, lo, hi, scale}]}] naming the config keys the ladder may move")
