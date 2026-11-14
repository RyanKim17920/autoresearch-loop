"""loop.novelty — the NOVELTY ENGINE (DESIGN §4.7), against a loop that drifts into hyperparameter sweeps.

* SWEEP REDIRECTION (operator separation, AIRA / AIDE; operator 2026-09-30: no "lr + one small tweak" spam):
  on EVERY request that makes a candidate (explore, moonshot, exploit, revisit, crossover, compound, winner
  cross, bank; never a debug fix, a design point or a tuning trial) a diff that only changes existing config
  keys to other NUMBERS by less than ``novelty.major_ratio`` (x or /) is not an idea unless :func:`expected`
  backs it (the lever regression or a designed batch's significant effect) with an expected effect >= the
  screen's minimum detectable effect (:func:`mde`; "minute changes fall to noise"): intake rejects it
  ``sweep_only`` (unbacked) or ``below_noise`` (the researcher's ``check`` says so early) and its knobs +
  levels join the design batch's knob pool (:func:`forward`, used by loop.design.pick_knobs when lever-backed). A MAJOR move (> the ratio,
  to / from 0, a sign flip, a categorical / bool / schedule switch) is a candidate in any arm.
* MAP-ELITES CELLS (AlphaEvolve / OpenEvolve): every node gets a mechanism type (:data:`TYPES`, the
  researcher's declared one validated against the diff, else inferred); a cell is family x type. Plain
  explore / moonshot requests and the scout target the emptiest cell; the cell's best node is shown as the
  inspiration.
* NOVELTY BONUS (novelty search / ShinkaEvolve): a closed mechanism node (its diff touches code) earns
  ``novelty.weight`` x (1 - max cosine to every earlier node's claim + diff embedding), per GPU-h, on top of
  its arm's bandit reward; the audited component ``novelty`` decides per request whether the arm choice sees it.
* PERSONAS (co-scientist generate -> critique -> evolve): plain explore / moonshot requests rotate theorist,
  contrarian, cross_domain, literature (web sessions only); every ideation session self-critiques before
  finish. The persona is in the node's notes; loop.meta tracks which one wins.
* MECHANISM DEDUP: an explore / exploit / revisit candidate whose mechanism text (tag + claim, TF-IDF over every
  node's; the same normalized tag, :func:`norm_tag`, is cosine 1) is within ``novelty.mech_dup_cos`` of >= 1 live /
  recent node -- or DEAD node of any age (:func:`dead`: screened out / vetoed / not confirmed, or a loss / null
  postmortem) -- outside its lineage (a MOONSHOT: >= 2, or one at >= MOON_ONE_COS) is
  rejected ``dead_mechanism`` (a dead one among them) / ``near_dup_mechanism`` unless it names each of them (it says what differs and why they failed) (:func:`mech_dup_rule`). The same cosine dedups the scout bank on write
  (loop.scout) and the Labless exploratory tier (one representative per mechanism cluster, loop.labless).
* REGIME FACTS: ideation prompts carry the computed truncated-training regime (:func:`regime_block`)."""

from __future__ import annotations

import dataclasses
import json
import math
import re
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from loop.contracts import PANEL, ROOT_NODE_ID, NodeStatus, Operator, Strategy

TYPES = ("loss_objective", "data_sampling", "augmentation", "architecture_addon", "training_dynamics",
         "readout", "regularization", "optimization")
PERSONAS = ("theorist", "contrarian", "cross_domain", "literature")
IDEATION = (Strategy.EXPLORE, Strategy.MOONSHOT)
K_POOL = "novelty.knob_pool"          # [{key, high, levers, node, ts}]: redirected sweeps for the next design batch
K_BONUS = "novelty.bonus"             # {node id: novelty bonus per GPU-h} added to its arm's reward
K_PERSONA = "novelty.persona_seq"
POOL_MAX = 40
SWEEP_FIX = ("numeric-only tweak -> introduce a mechanism (new code / loss / objective / data or training "
             "procedure), or make a MAJOR move (> 2x, to/from 0, or a categorical / schedule switch) with its "
             "mechanism stated; an unbacked small tweak goes to a design batch (its knob and level are kept for one)")
_CONFIG = (".yaml", ".yml", ".json", ".toml")
_HINTS = {t: re.compile(p, re.I) for t, p in {
    "loss_objective": r"loss|objective|criterion|contrastive|reconstruct|distill|koleo|kde|jepa|ibot|fino",
    "data_sampling": r"sampl|dataloader|dataset|curat|balanc|tissue_thresh|shard|data\.(mean|std)\b",
    "augmentation": r"augment|\baug|crop|jitter|flip|blur|stain|\bhed|solari|mixup|cutmix|views|global_size|local_size",
    "architecture_addon": r"nn\.module|class \w+\(|head|adapter|register|projector|block|attention|model\.",
    "training_dynamics": r"schedul|warmup|\bema\b|momentum|teacher|curriculum|freeze|temperature|progressive",
    "readout": r"readout|pool|cls|patch_concat|jbu|robustnorm|lda",
    "regularization": r"dropout|drop_path|weight_decay|regulari|stochastic_depth|smoothing",
    "optimization": r"\blr\b|lr_|learning_rate|optimi[sz]er|adam|sgd|lion|lars|beta2|clip_grad|batch_size",
}.items()}
PERSONA_TEXT = {
    "theorist": "THEORIST: derive the change from first principles -- name the failure mode of the current "
                "objective / data / optimisation that the evidence points at, and the minimal mechanism that removes it.",
    "contrarian": "CONTRARIAN: break one recipe assumption on purpose; your change tests whether it really holds.",
    "cross_domain": "CROSS-DOMAIN: transfer a technique proven in NLP, natural-image vision or RL self-supervision "
                    "(say where it comes from) that this recipe does not use.",
    "literature": "LITERATURE: search first, then implement the core idea of a recent published method (cite it).",
}
CRITIQUE = ("Before finish, self-critique in handoff.concerns (one line each): is it a real mechanism, not a "
            "value change? is it already in the recipe (grep the code)? is it init-compatible? Revise if not.")


# ------------------------------------------------------------------ sweep redirection
def redirectable(strategy: Strategy | None, operator: Operator | None) -> bool:
    """An ideation request (the novelty prompt block: personas, cells)."""
    return strategy in IDEATION and operator not in (Operator.DEBUG, Operator.CROSSOVER)


def guarded(strategy: Strategy | None, operator: Operator | None) -> bool:
    """A request the sweep_only rule applies to: every arm but a design point / tuning trial and a debug fix."""
    return strategy is not None and strategy is not Strategy.DESIGN and operator is not Operator.DEBUG


def _numeric(v: Any) -> bool:
    return (isinstance(v, (int, float)) and not isinstance(v, bool)) or (
        isinstance(v, (list, tuple)) and bool(v) and all(_numeric(x) for x in v))


def _level(text: str) -> Any:
    try:
        v = yaml.safe_load(text)
    except yaml.YAMLError:
        return None
    try:
        return float(v) if isinstance(v, str) else v        # '1e-6' is a str in YAML 1.1
    except ValueError:
        return v


def ratio(old: Any, new: Any) -> float:
    """The largest x-or-/ move between two numeric levels (element-wise for lists); inf to / from 0, across 0,
    or for a changed shape."""
    if isinstance(old, (list, tuple)) and isinstance(new, (list, tuple)) and len(old) == len(new):
        return max((ratio(a, b) for a, b in zip(old, new)), default=1.0)
    if not (_numeric(old) and _numeric(new)) or isinstance(old, (list, tuple)) or isinstance(new, (list, tuple)):
        return math.inf
    if old == new:
        return 1.0
    if old == 0 or new == 0 or (old > 0) != (new > 0):
        return math.inf
    return max(abs(old), abs(new)) / min(abs(old), abs(new))


def minor(r: float, major: float) -> bool:
    """A move of x-or-/ size ``r`` is MINOR (a sweep / twin step) up to and INCLUDING ``major``: an exact doubling
    (local_crop_scale 0.16 -> 0.35 vs 0.08 -> 0.175, jepa_block_scale 0.1 -> 0.2) is a dose step, not a new idea."""
    return r <= major * (1 + 1e-9)


def moves(candidate: Any, major: float = 2.0) -> dict[str, tuple[Any, Any]]:
    """{key: (old, new)} when the diff is config-only and every change sets an EXISTING key (its leaf is on a
    removed line) to a number at most ``major`` x or / from the old one; {} otherwise (code, a new key, a
    string / bool / structural value, or one MAJOR numeric move)."""
    keys, paths = candidate.config_diff, candidate.changed_paths
    if not keys or not paths or not all(p.endswith(_CONFIG) for p in paths):
        return {}
    old = olds(candidate)
    if all(_numeric(v) and k in old and minor(ratio(old[k], v), major) for k, v in keys.items()):
        return {k: (old[k], v) for k, v in keys.items()}
    return {}


def olds(candidate: Any) -> dict[str, Any]:
    """{key: the value the diff replaced} for the changed keys whose leaf is on a removed line."""
    removed = "\n".join(l[1:] for l in candidate.diff_text.splitlines() if l.startswith("-") and not l.startswith("---"))
    return {k: _level(m.group(1)) for k in candidate.config_diff
            if (m := re.search(rf"^\s*{re.escape(k.rsplit('.', 1)[-1])}\s*:\s*(\S.*)$", removed, re.M))}


def sweep_knobs(candidate: Any, major: float = 2.0) -> dict[str, Any]:
    """{key: new value} of a minor numbers-only diff (:func:`moves`); {} otherwise."""
    return {k: new for k, (_, new) in moves(candidate, major).items()}


def mde(ctx: Any, snapshot: Any) -> float:
    """The screen's minimum detectable panel effect: 2 x the paired panel sigma in force / sqrt(screen seeds)."""
    sigma = float((getattr(snapshot, "sigma", None) or {}).get(PANEL, ctx.cfg.panel_sd))
    return 2.0 * sigma / max(1, len(ctx.cfg.seeds.screen)) ** 0.5


def expected(ctx: Any, store: Any, snapshot: Any, candidate: Any, mv: Mapping[str, tuple[Any, Any]]) -> tuple[str, float]:
    """(what backs a minor numbers-only move, its expected panel effect); ('', 0) = nothing backs it. Backing:
    the declared levers the world model's lever regression supports (P >= design.lever_p in that direction;
    lever_p 0 = never; effect = sum of direction x slope), or a designed batch's significant effect on a moved
    key in the direction of the move (effect scaled linearly to the move's size)."""
    from loop.design import K_DESIGN, _num, _sign, backed
    out = ("", 0.0)
    lev = {l: d for l, d in candidate.hypothesis.levers.directions.items() if d in (-1, 1)}
    if ctx.cfg.design.lever_p > 0 and lev and backed(ctx, snapshot, lev):
        beta = {e.lever: e.mean for e in getattr(snapshot, "levers", ()) or () if e.target == PANEL}
        out = ("lever regression", sum(d * beta.get(l, 0.0) for l, d in lev.items()))
    for b in (store.get_state(K_DESIGN) or {}).get("batches", []):
        for kb, e in zip(b.get("knobs", ()), b.get("effects", ()) or ()):
            m = mv.get(kb["key"])
            if m and e.get("significant") and all(map(_num, (kb["low"], kb["high"], *m))) \
                    and _sign(m[1] - m[0]) == _sign(kb["high"] - kb["low"]) * _sign(e["effect"]) != 0 \
                    and (x := abs(e["effect"] * (m[1] - m[0]) / (kb["high"] - kb["low"]))) > out[1]:
                out = (f"design d{b.get('batch', '?')} effect on {kb['key']}", x)
    return out


def forward(store: Any, candidate: Any, node_id: str, now: float | None = None, major: float = 2.0,
            mv: Mapping[str, tuple[Any, Any]] | None = None) -> list[str]:
    """Add a redirected sweep's knobs (level = its value, levers = its hypothesis') to the design knob pool
    (``mv``: the moves intake judged, loop.select.intake.relative_moves)."""
    knobs = {k: new for k, (_, new) in mv.items()} if mv else sweep_knobs(candidate, major)
    lev = {k: v for k, v in candidate.hypothesis.levers.directions.items() if v in (-1, 1)}
    pool = [e for e in pool_entries(store) if e["key"] not in knobs]
    pool += [{"key": k, "high": v, "levers": lev, "node": node_id, "ts": time.time() if now is None else now}
             for k, v in knobs.items()]
    store.set_state(K_POOL, pool[-POOL_MAX:])
    return sorted(knobs)


def pool_entries(store: Any) -> list[dict[str, Any]]:
    return list(store.get_state(K_POOL) or [])


def consume(store: Any, keys: Sequence[str]) -> None:
    """A batch used these pool knobs."""
    if (pool := pool_entries(store)) and set(keys) & {e["key"] for e in pool}:
        store.set_state(K_POOL, [e for e in pool if e["key"] not in set(keys)])


# ------------------------------------------------------------------ MAP-Elites cells
def infer_type(text: str) -> str:
    """The mechanism type with the most hint matches in ``text`` ('' when none)."""
    score = {t: len(rx.findall(text)) for t, rx in _HINTS.items()}
    best = max(TYPES, key=lambda t: score[t])
    return best if score[best] else ""


def mechanism_type(declared: str, paths: Sequence[str], diff_text: str, tag: str = "", family: str = "") -> str:
    """The researcher's declared type when the diff shows any sign of it (or of nothing), else the inferred one
    (hint matches over paths, changed lines, tag and family name)."""
    from loop.select.embed import changed_lines
    text = "\n".join([*paths, changed_lines(diff_text), tag, family])
    if declared in TYPES and (_HINTS[declared].search(text) or not infer_type(text)):
        return declared
    return infer_type(text) or (declared if declared in TYPES else
                                "optimization" if all(p.endswith(_CONFIG) for p in paths) else "architecture_addon")


_TYPE_CACHE: dict[str, str] = {}


def node_type(n: Any) -> str:
    """``mtype=`` from the notes; older nodes are classified from their diff file + tag (cached)."""
    if m := re.search(r"\bmtype=(\w+)", n.notes or ""):
        return m.group(1)
    if n.id not in _TYPE_CACHE:
        try:
            text = open(n.diff_path).read() if n.diff_path else ""
        except OSError:
            text = ""
        _TYPE_CACHE[n.id] = mechanism_type("", re.findall(r"^diff --git a/(\S+)", text, re.M), text, n.mechanism_tag,
                                           n.family)
    return _TYPE_CACHE[n.id]


def _members(store: Any) -> list[Any]:
    return [n for n in store.nodes() if n.id != ROOT_NODE_ID and n.strategy is not Strategy.DESIGN
            and n.status is not NodeStatus.REJECTED]


def counts(store: Any, family: str = "") -> dict[str, int]:
    """Nodes per mechanism type (within ``family`` when given)."""
    out = dict.fromkeys(TYPES, 0)
    for n in _members(store):
        if (not family or n.family == family) and (t := node_type(n)) in out:
            out[t] += 1
    return out


def target(store: Any, family: str = "", salt: int = 0) -> str:
    """The emptiest mechanism type (of ``family``'s cells, or over all families); ties rotate with ``salt``."""
    c = counts(store, family)
    low = [t for t in TYPES if c[t] == min(c.values())]
    return low[salt % len(low)]


def elite(store: Any, snapshot: Any, family: str, mtype: str) -> Any:
    """The best scored node in the cell (family x type; any family when ''), or None."""
    best = None
    for n in _members(store):
        post = snapshot.nodes.get(n.id) if snapshot is not None else None
        if post is None or not post.utility.n or (family and n.family != family) or node_type(n) != mtype:
            continue
        if best is None or post.utility.mean > best[0]:
            best = (post.utility.mean, n)
    return best[1] if best else None


def cell_block(store: Any, snapshot: Any, family: str, mtype: str) -> str:
    """Target cell, its fill counts and the elite as inspiration (researcher prompt)."""
    from loop.agents.briefs import node_line, read_text
    from loop.select.embed import changed_lines
    c = counts(store, family)
    where = f"family {family} x type {mtype}" if family else f"type {mtype} (any family)"
    out = [f"Target cell (MAP-Elites): {where}, {c[mtype]} node(s) so far. Fill: "
           + ", ".join(f"{t} {c[t]}" for t in TYPES)
           + f". If {mtype} cannot fit, take the emptiest type that can; declare hypothesis.mechanism_type."]
    if (e := elite(store, snapshot, family, mtype)) is not None:
        h = store.get_hypothesis(e.hypothesis_id or "")
        out.append(f"Inspiration (the cell's best; beat it, don't copy it): {node_line(snapshot, e)}"
                   + (f" -- {h.claim[:200]}" if h else "")
                   + (f"\n{d}" if (d := changed_lines(read_text(e.diff_path, 4000))[:800]) else ""))
    else:
        out.append("The cell is empty: a first working entry opens it.")
    return "\n".join(out)


def scout_block(store: Any, snapshot: Any) -> str:
    c = counts(store)
    order = sorted(TYPES, key=lambda t: (c[t], t))
    best = {t: elite(store, snapshot, "", t) for t in order[:4]}
    return ("Mechanism types by nodes tried (MAP-Elites cells; aim most ideas at the emptiest): "
            + ", ".join(f"{t} {c[t]}" + (f" (best {best[t].id} {best[t].mechanism_tag})" if best.get(t) else "")
                        for t in order))


# ------------------------------------------------------------------ personas
def persona_for(store: Any, web: bool) -> str:
    """The next persona in rotation (literature only for a web session)."""
    seq = int(store.get_state(K_PERSONA, 0) or 0)
    store.set_state(K_PERSONA, seq + 1)
    ring = PERSONAS if web else tuple(p for p in PERSONAS if p != "literature")
    return ring[seq % len(ring)]


def assumptions(store: Any, limit: int = 3) -> list[str]:
    """The latest first-principles review's recipe assumptions, broken / unknown first."""
    for rv in sorted(store.reviews(), key=lambda r: -(r.created_ts or 0.0)):
        rows = [f for f in rv.findings if re.match(r"\[\w+/(broken|unknown|holds)\]", f)]
        if rows:
            return sorted(rows, key=lambda f: ("/broken]" not in f, "/unknown]" not in f))[:limit]
    return []


def persona_block(store: Any, persona: str) -> str:
    text = PERSONA_TEXT.get(persona, "")
    if persona == "contrarian":
        rows = assumptions(store)
        text += ("\n" + "\n".join(f"- {r[:240]}" for r in rows)) if rows else \
            " (No first-principles review yet: pick an assumption the recipe hard-codes.)"
    return f"Persona: {text}" if text else ""


def block(ctx: Any, store: Any, snapshot: Any, request: Any) -> str:
    """The researcher prompt's novelty section ('' outside ideation)."""
    if not ctx.cfg.novelty.enabled or not redirectable(request.strategy, request.operator):
        return ""
    out = ["## Novelty", "A small numbers-only change to existing config keys is NOT an idea here: unless "
                         "substantiated it is redirected to a design batch. Bring a mechanism."]
    if request.persona:
        from loop.meta import bias_line
        out += [x for x in (persona_block(store, request.persona), bias_line(ctx.cfg, "persona")) if x]
    if request.cell:
        out.append(cell_block(store, snapshot, request.family_hint, request.cell))
    if request.strategy in IDEATION and (rb := regime_block(store, snapshot)):
        out.append(rb)
    out.append(CRITIQUE)
    return "\n".join(out) + "\n\n"


# ------------------------------------------------------------------ novelty bonus
def distance(node: Any, others: Sequence[Any]) -> float:
    """1 - the largest cosine between ``node``'s embedding and every earlier node's (1 when none)."""
    from loop.select.embed import cosine
    prior = [o.embedding for o in others if o.id != node.id and o.embedding is not None
             and (o.created_ts or 0.0) <= (node.created_ts or 0.0)]
    return max(0.0, min(1.0, 1.0 - max((cosine(node.embedding, e) for e in prior), default=0.0)))


def bonus(ctx: Any, store: Any, node: Any) -> float:
    """``novelty.weight`` x distance for a mechanism node (its diff touches code); 0 otherwise."""
    from loop.select import _touches_code
    w = ctx.cfg.novelty.weight
    if not ctx.cfg.novelty.enabled or w <= 0 or node.embedding is None or node.strategy is Strategy.DESIGN \
            or not _touches_code(node):
        return 0.0
    return w * distance(node, store.nodes())


def record_bonus(store: Any, node_id: str, per_gpu_h: float) -> None:
    store.set_state(K_BONUS, {**(store.get_state(K_BONUS) or {}), node_id: round(per_gpu_h, 6)})


def with_bonus(store: Any, pulls: Sequence[Any]) -> list[Any]:
    """The pulls with each node's novelty bonus added to its reward."""
    book: Mapping[str, float] = store.get_state(K_BONUS) or {}
    return [dataclasses.replace(p, reward=p.reward + book[p.node_id]) if p.node_id in book else p for p in pulls]


# ------------------------------------------------------------------ mechanism dedup
_WORD = re.compile(r"[a-z][a-z0-9]{3,}")
_SUFFIX = re.compile(r"(?<=[a-z0-9]{4})(?:ing|ed|es|(?<![su])s)$")
MOON_ONE_COS = 0.5                    # a moonshot this near (or with the same normalized tag) to ONE node is a dup


def stem(word: str) -> str:
    """One simple suffix off (-ing / -ed / -es / -s, keeping >= 4 chars): whitened -> whiten, views -> view."""
    return _SUFFIX.sub("", word.lower())


def norm_tag(tag: str) -> str:
    """A mechanism tag, family-agnostic, lowercased and stemmed per word: ``whitened_jepa_target`` and
    ``whiten_jepa_target`` are one mechanism (live: a node objective/ vs a node training/)."""
    return "_".join(stem(w) for w in re.findall(r"[a-z0-9]+", (tag or "").lower()))
_NODE_ID = re.compile(r"\bn\d{4}\b")
RARE_MIN, RARE_FRAC = 3, 0.02         # a tag word in at most max(RARE_MIN, RARE_FRAC x nodes) tags names a mechanism


def tag_words(tag: str) -> set[str]:
    """The normalized tag's words of >= 4 characters (``pcgrad``, ``stem``, ``stain``, ``beta2``)."""
    return {w for w in norm_tag(tag).split("_") if len(w) >= 4}


def mech_terms(tag: str, claim: str) -> Counter:
    """A mechanism description's terms: the tag (twice: it names the mechanism) + the claim, 7-char stems."""
    t = f"{tag} {tag} {claim}".replace("_", " ").lower()
    return Counter(stem(w)[:7] for w in _WORD.findall(t))


def _tfidf(terms: Counter, df: Counter, n: int) -> dict[str, float]:
    v = {w: (1.0 + math.log(c)) * math.log((n + 1) / (df[w] + 1)) for w, c in terms.items()}
    z = math.sqrt(sum(x * x for x in v.values())) or 1.0
    return {w: x / z for w, x in v.items()}


def mech_cos(me: Counter, docs: Sequence[Counter]) -> list[float]:
    """Cosine of mechanism terms ``me`` to each of ``docs`` (TF-IDF, IDF over ``docs`` + ``me``)."""
    df = Counter(w for t in (*docs, me) for w in t)
    v = _tfidf(me, df, len(docs) + 1)
    return [sum(x * u.get(w, 0.0) for w, x in v.items()) for u in (_tfidf(t, df, len(docs) + 1) for t in docs)]


_MV: dict[str, Any] = {}
_DEAD: dict[str, Any] = {}


def mech_vectors(store: Any) -> dict[str, dict[str, float]]:
    """node id -> TF-IDF vector of its mechanism text (tag + claim; IDF over every node's). v3 review O2: cached on
    the (node id, tag, hypothesis, claim) set -- a tick asked for it once per brief (12x, ~1 s each)."""
    hyps = {h.id: h for h in store.hypotheses()}
    rows = tuple((n.id, n.mechanism_tag, getattr(hyps.get(n.hypothesis_id or ""), "claim", ""))
                 for n in store.nodes() if n.id != ROOT_NODE_ID)
    key = (id(store), rows)
    if _MV.get("key") != key:
        docs = {i: mech_terms(tag, claim) for i, tag, claim in rows}
        df = Counter(w for t in docs.values() for w in t)
        _MV.update(key=key, val={k: _tfidf(t, df, len(docs)) for k, t in docs.items()})
    return _MV["val"]


DEAD_STATUSES = frozenset({NodeStatus.SCREENED_OUT, NodeStatus.VETOED, NodeStatus.NOT_CONFIRMED})


def dead(store: Any) -> set[str]:
    """Nodes whose mechanism failed: screened out / vetoed / not confirmed, or closed (not confirmed / passed /
    rejected / crashed / void) with a loss / null postmortem."""
    from loop.contracts import CLOSED_STATUSES
    pm = {p.node_id for p in store.postmortems() if p.verdict in ("loss", "null")} if hasattr(store, "postmortems") else set()
    other = CLOSED_STATUSES - {NodeStatus.CONFIRMED, NodeStatus.EVAL_PASSED, NodeStatus.REJECTED, NodeStatus.CRASHED,
                               NodeStatus.VOID}
    skip = struct_exempt(store)
    return {n.id for n in store.nodes() if n.id not in skip
            and (n.status in DEAD_STATUSES or (n.id in pm and n.status in other))}


def struct_exempt(store: Any) -> set[str]:
    """Nodes whose failure is no evidence about their mechanism yet (10-22 adversarial review S4): the children of a
    structural tuning ladder (``struct_tune=``: a point on a dose / lr grid, not a mechanism) and a node with a BOOKED
    ladder that has not written its computed postmortem yet (it is judged after its tuning, loop.core.struct_tune).
    Left out of :func:`dead`, as ``intake.base_unladdered`` is for the base-challenge ladders."""
    try:
        from loop.core.struct_tune import K_PM, K_STRUCT
        booked, done = store.get_state(K_STRUCT) or {}, store.get_state(K_PM) or {}
    except Exception:  # noqa: BLE001 - no state view: nothing is exempt
        booked, done = {}, {}
    open_ = {n for n, r in booked.items() if isinstance(r, Mapping) and n not in done}
    return {n.id for n in store.nodes() if " struct_tune=" in f" {n.notes or ''}" or n.id in open_}


def mech_dups(ctx: Any, store: Any, candidate: Any, now: float | None = None) -> list[tuple[Any, float]]:
    """(node, cosine) of every live (not closed), recent (``novelty.mech_dup_hours``) or DEAD (:func:`dead`, any
    age: live 2026-10-03, a node was the 7th PCGrad node) non-rejected node whose
    mechanism text is within ``novelty.mech_dup_cos`` of the candidate's (IDF over every node's, so shared
    vocabulary like 'jepa' / 'panel' weighs little), closest first; a tuning pass's graduate (``tune_of=``) is
    its source's mechanism and never counts twice. A dead node at half the cosine counts when its tag shares a RARE
    word AND one more word (dino_jepa, stem_lr, kde_anneal) with the candidate's: one shared word alone was loose (live 2026-10-03: dropkey_attn vs
    attn_entropy, tile_loss_truncation vs filter_low_tissue_tiles, lookahead_slow vs slow_ema_jepa)."""
    from loop.contracts import CLOSED_STATUSES
    if (thr := ctx.cfg.novelty.mech_dup_cos) <= 0:
        return []
    now = time.time() if now is None else now
    hyps, gone = {h.id: h for h in store.hypotheses()}, dead(store)
    docs = [(n, mech_terms(n.mechanism_tag, getattr(hyps.get(n.hypothesis_id or ""), "claim", "")))
            for n in store.nodes() if n.id != ROOT_NODE_ID]
    df = Counter(w for _, t in docs for w in t)
    h = candidate.hypothesis
    me, out, tag = _tfidf(mech_terms(h.mechanism_tag, h.claim), df, len(docs)), [], norm_tag(h.mechanism_tag)
    tdf = Counter(w for n, _ in docs for w in tag_words(n.mechanism_tag))
    rare = {w for w in tag_words(h.mechanism_tag) if tdf[w] <= max(RARE_MIN, RARE_FRAC * len(docs))}
    words = {w for w in tag.split("_") if len(w) >= 2}
    for n, terms in docs:
        if n.status is NodeStatus.REJECTED or "tune_of=" in (n.notes or "") or " struct_tune=" in f" {n.notes or ''}" or (
                n.status in CLOSED_STATUSES and n.id not in gone
                and (n.created_ts or 0.0) < now - 3600 * ctx.cfg.novelty.mech_dup_hours):
            continue
        v = _tfidf(terms, df, len(docs))
        c = 1.0 if tag and norm_tag(n.mechanism_tag) == tag else sum(x * v.get(w, 0.0) for w, x in me.items())
        theirs = set(norm_tag(n.mechanism_tag).split("_"))   # a dead node sharing a RARE tag word (pcgrad, stem,
        if n.id in gone and c >= thr / 2 and rare & theirs and len(words & theirs) >= 2:   # coverage) + one more
            c = max(c, thr)                                    # tag word and half the cosine
        if c >= thr:
            out.append((n, c))
    return sorted(out, key=lambda x: -x[1])


_DEAD_LIST: dict[tuple, list[tuple[str, list[str]]]] = {}


def dead_clusters(ctx: Any, store: Any, limit: int = 15) -> list[tuple[str, list[str]]]:
    """[(normalized tag, member ids oldest first)] of the dead mechanisms most often retried: a greedy cover of the
    dead nodes (:func:`dead`; never a tuning pass's graduate) by their mechanism neighbourhoods (same normalized tag
    or cosine >= ``novelty.mech_dup_cos``, the intake rule's), largest first, >= 2 members (cached by the dead set)."""
    gone, thr = dead(store), ctx.cfg.novelty.mech_dup_cos or 1.0
    nodes = {n.id: n for n in store.nodes() if n.id in gone and "tune_of=" not in (n.notes or "")}
    if (key := (tuple(sorted(nodes)), thr, limit)) not in _DEAD_LIST:
        vec, tags = mech_vectors(store), {i: norm_tag(n.mechanism_tag) for i, n in nodes.items()}
        cos = lambda a, b: sum(x * vec[b].get(w, 0.0) for w, x in vec[a].items())   # noqa: E731
        ids = [i for i in nodes if i in vec]
        nb = {a: {b for b in ids if tags[a] == tags[b] or cos(a, b) >= thr} for a in ids}
        left, out = set(ids), []
        while left and len(out) < limit:
            c = max(left, key=lambda i: (len(nb[i] & left), i))
            if len(m := nb[c] & left) < 2:
                break
            out.append((tags[c], sorted(m, key=lambda i: (nodes[i].created_ts, i))))
            left -= m
        _DEAD_LIST.clear()
        _DEAD_LIST[key] = out
    return _DEAD_LIST[key]


def dead_block(ctx: Any, store: Any, snapshot: Any, limit: int = 15) -> str:
    """The researcher's dead-mechanism list: per retried dead mechanism, attempts, the best OWN increment and the
    last node ("" when none)."""
    rows = []
    for tag, ids in dead_clusters(ctx, store, limit):
        inc = [p.increment.mean for i in ids if (p := (getattr(snapshot, "nodes", None) or {}).get(i)) is not None]
        rows.append(f"- {tag}: {len(ids)} attempts, best own increment "
                    + (f"{max(inc):+.4f}" if inc else "-") + f", last {ids[-1]} ({', '.join(ids[-6:])})")
    rows += [f"- {x} (operator: low yield, deprioritized)" for x in getattr(ctx.cfg.novelty, "dead_extra", ())]
    return "\n".join(rows)


def fresh_own(ctx: Any, store: Any, nid: str) -> float | None:
    """The node's mean own panel increment over its parent on its FRESH seeds (confirm / audit-stage runs paired with
    the parent's DONE run at the same seed); None without any."""
    from loop.contracts import RunStatus, Stage
    node = store.get_node(nid)
    if node is None:
        return None
    panel = lambda r: sum(ctx.cfg.weights[a] * r.scores[a] for a in ctx.cfg.axes) \
        if all(a in (r.scores or {}) for a in ctx.cfg.axes) else None        # noqa: E731
    ctl = {}
    for r in store.runs(node_id=node.parent_id or ROOT_NODE_ID, status=[RunStatus.DONE]):
        if (v := panel(r)) is not None:
            ctl.setdefault(r.seed, v)
    ds = [v - ctl[r.seed] for r in store.runs(node_id=nid, status=[RunStatus.DONE])
          if r.stage in (Stage.CONFIRM, Stage.AUDIT) and r.seed in ctl and (v := panel(r)) is not None]
    return sum(ds) / len(ds) if ds else None


def mechanism_dead(ctx: Any, store: Any, ids: Any) -> tuple[bool, str]:
    """(dead, evidence) for a mechanism's dead nodes ``ids``: DEAD only on fresh-seed evidence -- its best fresh
    (confirm / audit) own increment is <= 0, or it failed >= 2 fresh-seed confirmations (not confirmed / audit
    failed with a fresh own increment <= 0: a positive one that missed the e-value bar is too little evidence, not a
    failure); screen-only failures leave it COLD (advisory). Live 2026-10-06: tissue-coverage sampling was blocked
    4x while a node carried the strongest unported signal (confirm E 4.07, all 3 confirm seeds positive)."""
    fresh = {i: f for i in ids if (f := fresh_own(ctx, store, i)) is not None}
    failed = [i for i in ids if (n := store.get_node(i)) is not None and fresh.get(i, 0.0) <= 0.0
              and n.status in (NodeStatus.NOT_CONFIRMED, NodeStatus.AUDIT_FAILED)]
    best = max(fresh.items(), key=lambda kv: kv[1]) if fresh else None
    why = (f"best fresh-seed own increment {best[1]:+.4f} ({best[0]})" if best else "no fresh-seed evidence") + \
        f", {len(failed)} failed confirmation(s)"
    return (best is not None and best[1] <= 0.0) or len(failed) >= 2, why


MECH_ARMS = (Strategy.EXPLORE, Strategy.EXPLOIT, Strategy.REVISIT, Strategy.MOONSHOT)
SHOW = 8                              # mech_dup_rule names (and asks to be answered) at most this many near nodes


def mech_dup_rule(ctx: Any, store: Any, candidate: Any, strategy: Strategy | None,
                  now: float | None = None, operator: Operator | None = None) -> tuple[str, str]:
    """('dead_mechanism' | 'near_dup_mechanism', fix) for an explore / exploit / revisit candidate near >= 1 live /
    recent / dead mechanism outside its own lineage (a moonshot: >= 2, or one at cosine >= MOON_ONE_COS / the same
    normalized tag), that does not answer them: its claim / falsification / handoff must name every near node shown
    (the SHOW closest). ``dead_mechanism`` when a dead node (:func:`dead`) is among them AND the mechanism is dead on
    fresh seeds (:func:`mechanism_dead`); when every near node is a screen-only failure the mechanism is COLD:
    ('', advisory). Never a debug fix (the same mechanism, repaired)
    nor a crossover (two mechanisms). Live: a node / a node both fixed the truncated teacher EMA, a node / a node both
    site/slide-balanced sampling; a node PCGrad after six dead PCGrad nodes, a node the 5th stem-LR node."""
    if strategy not in MECH_ARMS or operator in (Operator.DEBUG, Operator.CROSSOVER):
        return "", ""
    nodes = {n.id: n for n in store.nodes()}
    line, n = set(), next((x for x in nodes.values() if candidate.parent_commit and x.commit == candidate.parent_commit), None)
    while n is not None and n.id not in line:        # its own lineage refines, never duplicates, its mechanism
        line.add(n.id)
        n = nodes.get(n.parent_id or "")
    near, gone = [(n, c) for n, c in mech_dups(ctx, store, candidate, now) if n.id not in line], dead(store)
    if not near or (strategy is Strategy.MOONSHOT and len(near) < 2 and near[0][1] < MOON_ONE_COS):
        return "", ""
    h, ho, near = candidate.hypothesis, candidate.handoff, near[:SHOW]
    named = set(_NODE_ID.findall(" ".join((h.claim, h.falsification, ho.actions, ho.learnings, ho.concerns))))
    if all(n.id in named for n, _ in near):
        return "", ""
    from loop.explain import why as explain           # computed (loop.diagnostics) / stored: never an LLM call
    why = {n.id: explain(n, store) for n, _ in near}
    what = "; ".join(f"{n.id} ({n.mechanism_tag}, cos {c:.2f}, {n.status.value}"
                     + (f": {why[n.id][:200]})" if why[n.id] else ")") for n, c in near)
    ids = ", ".join(n.id for n, _ in near)
    if gone := gone & {n.id for n, _ in near}:
        dead_, why_d = mechanism_dead(ctx, store, gone)
        if not dead_ and len(gone) == len(near):   # every near node a screen-only failure: COLD, advisory only
            return "", (f"COLD mechanism (advisory, not a reject: {why_d}): {what}. Say in hypothesis.claim what differs "
                        f"from {ids}")
        if dead_:
            return "dead_mechanism", (f"this mechanism was already tried and failed on fresh seeds ({why_d}; "
                                      f"{len(gone)} dead node(s)): {what}. Either bring another mechanism, or name "
                                      f"EVERY one of {ids} in hypothesis.claim and state what is MATERIALLY different "
                                      "this time and why that answers their failure")
    return "near_dup_mechanism", (f"this mechanism is already being tried: {what}. Either bring another mechanism, or "
                                  f"name EVERY one of {ids} in hypothesis.claim and say what is different and why it "
                                  "failed")


# ------------------------------------------------------------------ truncated-regime facts
_REGIME: dict[tuple, dict[str, Any]] = {}
REKEY = re.compile(r"re_?key|sample_frac", re.I)


def regime(store: Any, limit: int = 40) -> dict[str, Any]:
    """How runs actually end, from the last ``limit`` finished runs' summary.json (cached by their ids):
    {n, stop: Counter of stop_reason, flop_frac / sample_frac: medians}; {} when none is readable."""
    from loop.contracts import RunStatus
    runs = [r for r in store.runs(status=[RunStatus.DONE]) if r.output_dir][-limit:]
    if (key := tuple(r.id for r in runs)) not in _REGIME:
        rows = []
        for r in runs:
            try:
                rows.append(json.loads((Path(r.output_dir) / "summary.json").read_text()))
            except (OSError, ValueError):
                continue
        num = lambda k: [float(x[k]) for x in rows if isinstance(x.get(k), (int, float))]   # noqa: E731
        ff, sf = num("flop_fraction"), num("sample_fraction")
        _REGIME.clear()
        _REGIME[key] = {"n": len(rows), "stop": Counter(str(x.get("stop_reason")) for x in rows),
                        "flop_frac": statistics.median(ff), "sample_frac": statistics.median(sf or [math.nan])} if ff else {}
    return _REGIME[key]


PROGRESS = {"frac": "FLOP frac", "sample_fraction": "sample_fraction"}   # train.py's schedule clocks
_OPND = r"(-?\d+(?:\.\d*)?(?:e-?\d+)?|[A-Za-z_]\w*)"
_P = "|".join(PROGRESS)
_BREAKS = (re.compile(rf"\b({_P})\s*(?:<=|>=|<|>)\s*{_OPND}"), re.compile(rf"{_OPND}\s*(?:<=|>=|<|>)\s*({_P})\b"),
           re.compile(rf"\(\s*({_P})\s*-\s*{_OPND}\s*\)"))
_COSINE = re.compile(rf"cosine_schedule\(\s*([^,()]+),\s*([^,()]+),\s*({_P})\s*\)")
COSINE_MIN = 0.25                     # a cosine over the whole clock that moves less than this share of its range


def _num(text: str, env: Mapping[str, float]) -> float | None:
    """An arithmetic expression of numbers and ``env`` names; None otherwise."""
    import ast
    import operator as op
    ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.USub: op.neg}

    def ev(n: Any) -> float:
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return float(n.value)
        if isinstance(n, ast.Name) and n.id in env:
            return env[n.id]
        if isinstance(n, ast.BinOp) and type(n.op) in ops:
            return ops[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in ops:
            return ops[type(n.op)](ev(n.operand))
        raise ValueError
    try:
        return ev(ast.parse(text.strip(), mode="eval").body)
    except (SyntaxError, ValueError, ZeroDivisionError, TypeError):
        return None


def unreachable(store: Any, diff_text: str) -> str:
    """Why a diff's new schedule never takes effect ('' = it does, or nothing is known): runs stop at a median FLOP
    frac f (:func:`regime`; samples at 1), and every breakpoint its ADDED code lines put on a schedule clock
    (``frac < x``, ``x <= frac``, ``(frac - x)``; x a number or a name the added lines assign) lies beyond f; or (no
    breakpoint reachable) an added cosine_schedule over the bare clock changes by f less than COSINE_MIN of what it
    changes over the whole clock -- against the removed cosine it replaces (same target), else, in a diff that only
    schedules, against its own start value. Live: a node's WSD decays from frac 0.8 (never); a node's beta2 cosine
    over frac reaches only 0.995 of 0.999 -> 0.98; a node moved the terminal weight decay."""
    rg = regime(store)
    if not rg or not diff_text:
        return ""
    reach = {"frac": rg["flop_frac"], "sample_fraction": 1.0}
    side = lambda c: [re.sub(r"#.*", "", l[1:]) for l in diff_text.splitlines()   # noqa: E731
                      if l.startswith(c) and not l.startswith(c * 3)]
    lines, removed = side("+"), side("-")
    env: dict[str, float] = {}
    for l in lines:
        if (m := re.match(r"\s*([A-Za-z_]\w*)\s*=(?!=)\s*(.+)$", l)) and (v := _num(m.group(2), env)) is not None:
            env[m.group(1)] = v
    text = "\n".join(lines)
    br = []
    for rx in _BREAKS:
        for m in rx.finditer(text):
            clock, x = (m.group(1), m.group(2)) if m.group(1) in PROGRESS else (m.group(2), m.group(1))
            if (v := _num(x, env)) is not None and 0.0 < v:
                br.append((clock, v))
    if any(v <= reach[c] for c, v in br):
        return ""
    share = lambda c: (1.0 - math.cos(math.pi * min(1.0, reach[c]))) / 2.0   # noqa: E731
    lhs = lambda l: (m.group(1) if (m := re.match(r"\s*(.+?)\s*=\s*cosine_schedule\(", l)) else "")   # noqa: E731
    old = {lhs(l): m for l in removed if lhs(l) and (m := _COSINE.search(l))}
    sched = {m.group(1) for l in lines if re.search(rf"\b({_P})\b", l) and (m := re.match(r"\s*([A-Za-z_]\w*)\s*=", l))}
    only = all(not l.strip() or re.search(rf"\b({'|'.join([*PROGRESS, *sched])})\b", l) or l.strip() in ("else:", "pass")
               or (re.match(r"\s*([A-Za-z_]\w*)\s*=", l) and re.match(r"\s*([A-Za-z_]\w*)", l).group(1) in env)
               for l in lines)
    slow = []
    for l in lines:
        if not (m := _COSINE.search(l)):
            continue
        c, a, b, o = m.group(3), _num(m.group(1), env), _num(m.group(2), env), old.get(lhs(l))
        if o is not None and o.group(3) == c:            # it replaces a cosine: how much of the change is reached
            a0, b0 = _num(o.group(1), {}), _num(o.group(2), {})
            if None in (a, b, a0, b0) or max(abs(a - a0), abs(b - b0)) == 0:
                continue
            got = max(abs(a - a0), abs((a - a0) * (1 - share(c)) + (b - b0) * share(c)))
            if got < COSINE_MIN * max(abs(a - a0), abs(b - b0)):
                slow.append((c, a, b))
        elif only and share(c) < COSINE_MIN:             # a new schedule and nothing else
            slow.append((c, a, b))
    if not br and not slow:
        return ""
    why = [f"breakpoint(s) {', '.join(f'{PROGRESS[c]} {v:g}' for c, v in br)} all lie beyond it"] if br else []
    why += [f"a cosine over the whole {PROGRESS[c]} moves only {share(c):.0%} of its range by then"
            + (f" ({a:g} -> {b:g} reaches {b + (a - b) * (1 - share(c)):.4g})" if a is not None and b is not None else "")
            for c, a, b in slow]
    return (f"runs stop at FLOP frac {reach['frac']:.3f} (median of the last {rg['n']} runs; sample_fraction 1): "
            + "; ".join(why) + ", so the new schedule never takes effect. Define it over the REACHABLE horizon "
            f"(e.g. frac / {reach['frac']:.2f}), or bring another mechanism")


def regime_block(store: Any, snapshot: Any) -> str:
    """The ideation prompt's regime section ('' when nothing is known): how runs stop, what that does to the
    FLOP-keyed schedules, the LR lever's measured effect and every re-keying node's verdict -- computed facts
    only, never a suggested direction (operator 2026-10-06: no hardcoded idea steering)."""
    rg, out = regime(store), []
    if rg:
        (reason, k), f = rg["stop"].most_common(1)[0], rg["flop_frac"]
        out += [f"- {k}/{rg['n']} recent runs stop on {reason} at a median {f:.3f} of the FLOP budget "
                f"({rg['sample_frac']:.3f} of the samples): ~{1 - f:.0%} of the FLOPs are never spent.",
                f"- train.py keys LR / WD / teacher temp / EMA / KDE on frac = train_flops / max_train_flops, so every "
                f"FLOP-keyed schedule stops at frac {f:.3f} and never anneals (a cosine is still at "
                f"{0.5 * (1 + math.cos(math.pi * f)):.0%} of its range)."]
    lev = next((e for e in getattr(snapshot, "levers", ()) or () if e.lever == "effective_lr" and e.target == PANEL), None)
    if lev is not None:
        out.append(f"- effective_lr lever measured: {lev.mean:+.5f} per step on the panel "
                   f"(se {lev.se:.5f}, P(>0) {lev.p_positive:.2f}, {lev.n_nonzero} trials).")
    if rk := [n for n in store.nodes() if REKEY.search(n.mechanism_tag or "")]:
        posts = getattr(snapshot, "nodes", None) or {}
        out.append("- Re-keying schedules to samples was measured: " + ", ".join(
            f"{n.id} {n.mechanism_tag} {n.status.value}" + (f" U={posts[n.id].utility.mean:+.4f}" if n.id in posts else "")
            for n in rk[:6]) + ".")
    return "\n".join(["Training regime (computed from the runs):", *out]) if out else ""
