"""Candidate intake (D4.4): ``admit`` runs the rules in order; the first to fire wins.

Reasons: empty_diff, surface_locked, surface_not_writable, protected_key, sweep_only / below_noise (a diff of
any arm but design / tuning / debug that only moves existing config numbers by at most novelty.major_ratio (2x included),
unbacked by the world model / a design batch, or backed with an expected effect below the screen's MDE;
loop.novelty forwards it to the design knob pool), exact_dup,
near_dup, dead_mechanism / near_dup_mechanism (an explore / exploit / revisit within novelty.mech_dup_cos of >= 1
live / recent / dead -- any age -- mechanism outside its lineage, a moonshot of >= 2, that
does not answer them, loop.novelty.mech_dup_rule), unreachable_schedule (every new schedule breakpoint lies
beyond the horizon runs reach, loop.novelty.unreachable), noop_knob (``intake.noop_keys``), parent_contains_knob, config_noop, unmapped_path, ambiguous_family,
unknown_family, disallowed_data (any added line or changed config value referencing GenBio data/models,
operator 2026-10-05: usage restrictions), init_illegal (an added non-comment line or changed config value matching
``init_compat.deny``: a pathology-trained checkpoint), forbidden_reference (checked right after protected_key on EVERY diff: an added line
matches the surface ``loop.forbid`` or a touched family's ``forbid``, or a changed config value reaches a
probe data root: no code may read labels or probe data), multi_family, family_mismatch (a
crossover or a moonshot may span families; a moonshot must still declare one family it touches; a
``combine`` family may pair with ONE other family and is always the declared one; a readout-only diff of ANY arm,
a moonshot's too, is admitted and scored by the eval-only lane, loop.evalonly), near_dup_inflight (a value variant of an unresolved
node's knobs), near_dup_config (a numbers-only candidate whose EFFECTIVE config -- root config + chain + diff --
is within novelty.major_ratio of a live node's on every key, same code), init_compat_missing, init_incompatible /
init_illegal / over_flop_cap (the failed init-compat check's ``InitCompat.rule``),
irrelevant, dead_dataset (v3-c8a B2, loop.core.datasets: a data.dataset_dir already measured below its parent on >=
``ideation.dead_dataset_min_seeds`` paired seeds, unless [differs from: ...] answers those nodes), mixed_change (v3-c8a
B3, loop.core.mixed: config keys spanning >= 2 ``intake.lever_groups``; the LAST rule -- v3 intake splits such a node
instead when ``intake.mixed_change: split``). sweep_only / below_noise judge the move against the NEAREST live relative by effective config
(root included), not the declared parent, so a crossover re-rooted on a common ancestor cannot hide a small tweak
behind inherited MAJOR keys.

A designed batch's node (strategy ``design``, family ``design``) skips near_dup / near_dup_inflight
and the family rules (it may span families); a design POINT also skips exact_dup (a point is a run on
its batch's block, not a new idea), and no candidate is an exact dup of a point. The rest applies.

``diagnose`` is the deterministic part (everything before init-compat and the judge) and
also returns a one-line fix, so the researcher's ``check`` tool runs the SAME rules.
"""

from __future__ import annotations

import json
import math
import os
import re
from fnmatch import fnmatch
from types import SimpleNamespace
from typing import Any, Sequence

from loop.adapters import flatten
from loop.contracts import (ACTIVE_STATUSES, ROOT_NODE_ID, Candidate, NodeStatus, Operator, Strategy,
                            design_point)
from loop import novelty as NV
from loop.spec import LoopContext
from loop.select.embed import changed_lines, cosine

_CONFIG_SUFFIXES = (".yaml", ".yml", ".json", ".toml")


def path_matches(path: str, pat: str) -> bool:
    """Exact, glob, or directory prefix. A ``file::Qual.name`` lock matches only itself."""
    path, d = path.removeprefix("./"), pat.rstrip("/")
    return fnmatch(path, pat) or path == d or path.startswith(d + "/")


def on_surface(path: str, pats: Any) -> bool:
    """``path`` (a ``file::`` span counts as its file) matches one of ``pats``."""
    return any(path_matches(path.split("::", 1)[0], pat) for pat in pats)


DISALLOWED_DATA = re.compile(r"genbio", re.I)   # GenBio data/models (e.g. GenBio-PathFM): usage-restricted, never used


def disallowed_data(diff_text: str, config_diff: Any = None) -> str:
    """The first added diff line (comments too) or changed config value that references disallowed data, or ''."""
    added = [l[1:] for l in diff_text.splitlines() if l.startswith("+") and not l.startswith("+++")]
    vals = [f"{k}: {v}" for k, v in (config_diff or {}).items()]
    return next((x.strip() for x in added + vals if DISALLOWED_DATA.search(x)), "")


def illegal_init(ctx: LoopContext, diff_text: str, config_diff: Any = None) -> tuple[str, str]:
    """(line, pattern) of the first added non-comment line or changed config value matching an
    ``init_compat.deny`` pattern (a pathology-trained checkpoint), or ('', '')."""
    deny = tuple(getattr(ctx.cfg.init_compat, "deny", ()) or ())
    added = [l[1:] for l in diff_text.splitlines()
             if l.startswith("+") and not l.startswith("+++") and not l[1:].lstrip().startswith("#")]
    vals = [f"{k}: {v}" for k, v in (config_diff or {}).items()]
    return next(((x.strip(), p) for x in added + vals for p in deny if re.search(p, x, re.I)), ("", ""))


#: v3 review M1: every way a diff can pull WEIGHTS (or a hub download) into the run. An added non-comment line that matches
#: is refused ``weight_load`` unless it sits inside a model.py weight-source loader (``load*pretrained*``) and matches only
#: the hub-loader subset (:data:`LOADER_OK`); the loader's source is then judged by ``init_compat`` (deny + offline construct).
WEIGHT_LOAD = re.compile(
    r"from_pretrained|hf_hub|huggingface_hub|snapshot_download|torch\.hub|load_state_dict_from_url|urlopen|urlretrieve"
    r"|requests\.|wget|safetensors.*load_file|torch\.load|np\.load|\.(pt|pth|ckpt|safetensors)['\"]|['\"]/(data|admin)"
    r"|pretrained\s*=\s*True")
LOADER_OK = re.compile(
    r"from_pretrained|hf_hub|huggingface_hub|snapshot_download|torch\.hub|load_state_dict_from_url|safetensors.*load_file"
    r"|['\"]model\.safetensors['\"]|pretrained\s*=\s*True")
LOADER_FN = re.compile(r"load\w*pretrained")
_HUNK = re.compile(r"^@@ .*? @@ ?(.*)$")
_TOP = re.compile(r"^(?:async\s+)?(?:def|class)\s+(\w+)")


_QUOTED = re.compile(r"""(['"])(.*?)\1""")
_URL = re.compile(r"https?://[^\s'\"]+")
_SOURCE_ARG = re.compile(r"""(?:create_model|from_pretrained|hf_hub_download|snapshot_download|load_state_dict_from_url"""
                         r"""|hf_hub_url)\(\s*(?:repo_id\s*=\s*|model_name\s*=\s*|url\s*=\s*)?(['"])(.*?)\1""")


def on_allowlist(source: str, allow: Sequence[str]) -> bool:
    return any(re.search(p, source.strip(), re.I) for p in allow)


def strip_allowed(code: str, allow: Sequence[str]) -> str:
    """``code`` with every quoted string that is exactly a WEIGHTS-ALLOWLIST source (``init_compat.allow``) blanked: a
    line that names an allowlisted URL / timm id is not a weight load by that name alone."""
    if not allow:
        return code
    return _QUOTED.sub(lambda m: "''" if on_allowlist(m.group(2), allow) else m.group(0), code)


def off_allowlist(code: str, allow: Sequence[str]) -> str:
    """The first weights source on this line outside the allowlist: any URL literal, or the first string argument of a
    weight-source call (create_model / from_pretrained / hub download); '' when there is none (or no allowlist)."""
    if not allow:
        return ""
    for m in _URL.finditer(code):
        if not on_allowlist(m.group(0), allow):
            return m.group(0)
    for m in _SOURCE_ARG.finditer(code):
        if not on_allowlist(m.group(2), allow):
            return m.group(2)
    return ""


def weight_load(diff_text: str, allow: Sequence[str] = ()) -> tuple[str, str]:
    """(file, line) of the first added non-comment line that loads weights / downloads / reads a checkpoint path outside
    the one allowed place (a model.py ``load*pretrained*`` function, hub loaders only), or ('', ''). ``allow``
    (``init_compat.allow``, the operator's WEIGHTS ALLOWLIST): a quoted string that is exactly an allowlisted source is
    not itself a weight load, and in model.py any other weights source (a URL, a create_model / from_pretrained / hub
    download argument) is refused."""
    cur, func = "", ""
    for raw in (diff_text or "").splitlines():
        if raw.startswith("+++ "):
            cur, func = raw[4:].strip().removeprefix("b/"), ""
            continue
        if raw.startswith(("--- ", "diff --git", "index ")):
            continue
        if m := _HUNK.match(raw):
            mm = re.search(r"(?:def|class)\s+(\w+)", m.group(1))
            func = mm.group(1) if mm else ""
            continue
        if not raw or raw[0] not in "+ ":
            continue
        text = raw[1:]
        if text and not text[0].isspace() and not text.startswith(("#", ")", "]", "}")):
            mm = _TOP.match(text)
            func = mm.group(1) if mm else ""
        if raw[0] != "+" or text.lstrip().startswith("#") or cur.endswith(_CONFIG_SUFFIXES + (".md", ".txt")):
            continue
        code = re.sub(r"\s+#.*$", "", text)
        if cur == "model.py" and off_allowlist(code, allow):
            return cur, text.strip()
        code_chk = strip_allowed(code, allow) if cur == "model.py" else code
        if not WEIGHT_LOAD.search(code_chk):
            continue
        if cur == "model.py" and LOADER_FN.search(func or "") and not WEIGHT_LOAD.search(LOADER_OK.sub(" ", code_chk)):
            continue
        return cur, text.strip()
    return "", ""


def _added_lines(diff_text: str) -> list[tuple[str, str]]:
    """(file, line) of every added non-comment line of a unified diff."""
    out, cur = [], ""
    for l in diff_text.splitlines():
        if l.startswith("+++ "):
            cur = l[4:].strip().removeprefix("b/")
        elif l.startswith("+") and not l[1:].lstrip().startswith("#"):
            out.append((cur, l[1:]))
    return out


def forbidden(ctx: LoopContext, fams: tuple[str, ...], diff_text: str,
              config_diff: Any = None) -> tuple[str, str, str]:
    """(scope, pattern, line) of the first hit: an added non-comment line matching the surface-level
    ``loop.forbid`` (every diff) or a touched family's ``forbid``, or a changed config value that reaches
    a forbidden data root (equal, under or above one, env vars expanded; ``$`` in a path is refused).
    A family's ``forbid`` guards that family's CODE: with a ``config_diff`` the config file's lines are judged as
    values (the surface patterns and the data roots), never by another family's code rules (10-13: a
    ``data.dataset_dir: /data/nanopath-databank/..`` line beside a readout edit was refused as an eval_readout read)."""
    pairs = _added_lines(diff_text)
    added = [l for _, l in pairs]
    code = [l for f, l in pairs if not (config_diff and f.endswith(_CONFIG_SUFFIXES))]
    scopes = [("surface", ctx.cfg.forbid, added)] + [(f, ctx.cfg.families[f].forbid, code) for f in fams
                                                      if f in ctx.cfg.families]
    for scope, pats, lines in scopes:
        for pat in pats:
            if hit := next((l for l in lines if re.search(pat, l, re.I)), None):
                return scope, pat, hit.strip()
    roots = [os.path.realpath(r) for r in ctx.cfg.forbid_roots]
    for key, val in (config_diff or {}).items():
        if not isinstance(val, str) or "/" not in val and "$" not in val:
            continue
        if "$" in val:
            return "surface", "$ in a config path", f"{key}: {val}"
        p = os.path.realpath(os.path.expanduser(val))
        if r := next((r for r in roots if os.path.commonpath([p, r]) in (p, r)), None):
            return "surface", f"data root {r}", f"{key}: {val}"
    return "", "", ""


#: v3-c6i (reward-hacking lit 2609.28614): the eval / lock gates' rejections go back to a research session TERSE --
#: the rule and a generic fix, never the pattern, the matched line or the protected names the gate holds
GATE_TEXT = {
    "forbidden_reference": "the change references something the training / readout code may not read (labels, probe "
                           "or benchmark data, the environment / filesystem); remove that reference",
    "init_illegal": "the change initialises or guides the model from a disallowed checkpoint; use another init",
    "disallowed_data": "the change uses data or models that may not be used; use the training tiles or the data bank",
    "protected_key": "the change sets a protected config key (budgets / eval identity); restore its value",
    "weight_load": "the change loads weights or reads a checkpoint / download source; models start from the declared init only",
}


def session_text(rule: str, fix: str) -> str:
    """What a research session is told about a rejection: the gate rules terse (:data:`GATE_TEXT`), the rest as is."""
    return GATE_TEXT.get(rule, fix)


def candidate_text(c: Candidate) -> str:
    return f"{c.hypothesis.claim}\n{changed_lines(c.diff_text)}"


def parent_config(store: Any, parent_id: str | None) -> dict[str, Any]:
    """Flattened ``config_diff:<id>`` state of each ancestor, root first (unknown, not empty,
    for knobs no ancestor changed)."""
    chain, n = [], store.get_node(parent_id) if parent_id else None
    while n is not None and n.id not in {c.id for c in chain}:
        chain.append(n)
        n = store.get_node(n.parent_id) if n.parent_id else None
    cfg: dict[str, Any] = {}
    for node in reversed(chain):
        cfg.update(flatten(store.get_state(f"config_diff:{node.id}") or {}))
    return cfg


K_BASE = "intake.base_config"     # the root commit's flattened config (the driver stores it from the adapter)
K_TREES = "intake.trees"          # commit -> git tree id of every node commit (the driver stores them from the adapter)
_MISSING = object()


def _norm(v: Any) -> str:
    return json.dumps(v, sort_keys=True, default=str)


def _lv(v: Any) -> Any:
    return NV._level(v) if isinstance(v, str) else v


def config_gaps(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    """key -> x-or-/ size of each differing value of two EFFECTIVE configs (inf: a key one side lacks,
    or a non-numeric / structural change)."""
    out = {}
    for k in set(a) | set(b):
        x, y = a.get(k, _MISSING), b.get(k, _MISSING)
        if x is _MISSING or y is _MISSING:
            out[k] = math.inf
        elif _norm(x) != _norm(y):
            out[k] = NV.ratio(_lv(x), _lv(y))
    return out


def code_base(store: Any, n: Any, memo: dict | None = None) -> str:
    """The node's nearest code-changing ancestor (itself included; root when none)."""
    from loop.select import _touches_code
    memo = {} if memo is None else memo
    seen = []
    while n is not None and n.id != ROOT_NODE_ID and n.id not in memo and not _touches_code(n):
        seen.append(n.id)
        n = store.get_node(n.parent_id) if n.parent_id else None
    out = ROOT_NODE_ID if n is None or n.id == ROOT_NODE_ID else memo.get(n.id, n.id)
    memo.update({i: out for i in seen})
    return out


def nearest(ctx: LoopContext, store: Any, candidate: Candidate, exclude: Any = frozenset()) -> tuple[Any, dict, dict, dict]:
    """(relative, key gaps, the candidate's effective config, the relative's) for the live node (not rejected /
    crashed, not a design point; root counts) with the candidate's code (the same nearest code-changing
    ancestor) whose EFFECTIVE config is closest: one within novelty.major_ratio on every key first, then the
    fewest differing keys, then the smallest largest move. An effective config is the root config
    (``K_BASE``; keys it lacks from the diff's removed lines) + the node's chain + its diff, so a key one
    side never listed is the base value on that side."""
    parent = next((n for n in store.nodes() if n.commit == candidate.parent_commit), None)
    pcfg = parent_config(store, parent.id if parent else None)
    base = {**{k: v for k, v in NV.olds(candidate).items() if k not in pcfg}, **(store.get_state(K_BASE) or {})}
    mine, memo, best = {**base, **pcfg, **flatten(candidate.config_diff)}, {}, None
    cb = code_base(store, parent, memo)
    if parent is None and ROOT_NODE_ID not in exclude:     # an unknown parent: the diff's own base
        gaps = config_gaps(base, mine)
        top = max(gaps.values(), default=1.0)
        best = ((not NV.minor(top, ctx.cfg.novelty.major_ratio), len(gaps), top, ROOT_NODE_ID),
                store.get_node(ROOT_NODE_ID) or SimpleNamespace(id=ROOT_NODE_ID), gaps, base)
    for n in store.nodes():
        if n.id in exclude or n.status in (NodeStatus.REJECTED, NodeStatus.CRASHED) or design_point(n) \
                or code_base(store, n, memo) != cb:
            continue
        theirs = {**base, **parent_config(store, n.id)}
        gaps = config_gaps(theirs, mine)
        top = max(gaps.values(), default=1.0)
        key = (not NV.minor(top, ctx.cfg.novelty.major_ratio), len(gaps), top, n.id)
        if best is None or key < best[0]:
            best = (key, n, gaps, theirs)
    return (best[1], best[2], mine, best[3]) if best else (None, {}, mine, {})


def _config_only(candidate: Candidate) -> bool:
    return bool(candidate.config_diff) and all(p.endswith(_CONFIG_SUFFIXES) for p in candidate.changed_paths)


def relative_moves(ctx: LoopContext, store: Any, candidate: Candidate, exclude: Any = frozenset(),
                   near: tuple | None = None) -> dict[str, tuple[Any, Any]]:
    """{key: (the nearest relative's value, the candidate's)} when the config-only candidate's EFFECTIVE config
    differs from its nearest live relative (:func:`nearest`, root included) only by numeric moves of at most
    novelty.major_ratio: judged on the whole config, not on the diff against its declared parent, so
    inherited MAJOR keys (a crossover re-rooted on a common ancestor) cannot hide a small tweak. {} otherwise."""
    if not _config_only(candidate) or not candidate.changed_paths:
        return {}
    rel, gaps, mine, theirs = near or nearest(ctx, store, candidate, exclude)
    if rel is None or not gaps or not NV.minor(max(gaps.values()), ctx.cfg.novelty.major_ratio) \
            or not all(NV._numeric(_lv(mine[k])) for k in gaps):
        return {}
    return {k: (_lv(theirs[k]), _lv(mine[k])) for k in gaps}


def config_twin(ctx: LoopContext, store: Any, candidate: Candidate, exclude: Any = frozenset(),
                near: tuple | None = None) -> Any:
    """A live non-root node (not rejected / crashed, not a design point) with the numbers-only candidate's
    code whose EFFECTIVE config (see :func:`nearest`) is equal or within novelty.major_ratio x or / on every
    key -- a near-duplicate of "a shared stack + one small tweak", whatever keys each diff happened to list."""
    rel, gaps, _, _ = near or nearest(ctx, store, candidate, {*exclude, ROOT_NODE_ID})
    return rel if rel is not None and rel.id != ROOT_NODE_ID \
        and all(NV.minor(g, ctx.cfg.novelty.major_ratio) for g in gaps.values()) else None


def sweep_rule(ctx: LoopContext, store: Any, snapshot: Any, candidate: Candidate, strategy: Strategy | None,
               operator: Operator | None, exclude: Any = frozenset(), near: tuple | None = None) -> tuple[str, str]:
    """('sweep_only' | 'below_noise', fix) for a minor numbers-only move off the nearest relative (any arm but
    design / tuning / debug) that the world model / a design batch does not back above the screen MDE."""
    cfg = ctx.cfg
    if not (cfg.novelty.enabled and cfg.novelty.sweep_redirect and NV.guarded(strategy, operator)) \
            or not (mv := relative_moves(ctx, store, candidate, exclude, near)):
        return "", ""
    why, eff = NV.expected(ctx, store, snapshot, candidate, mv)
    if not why:
        return "sweep_only", NV.SWEEP_FIX
    if abs(eff) < (floor := NV.mde(ctx, snapshot)):
        return "below_noise", (f"expected panel effect {eff:+.4f} ({why}) < the screen's detectable {floor:.4f}: "
                               "a minute change falls to noise; make it a MAJOR move or bring a mechanism "
                               "(the knob goes to a design batch, whose pooled factorial resolves small effects)")
    return "", ""


def redirect(ctx: LoopContext, store: Any, candidate: Any, source: str, exclude: Any = frozenset(),
             now: float | None = None, **fields: Any) -> list[str]:
    """A sweep_only / below_noise move's knobs + levels -> the design knob pool (loop.novelty.forward), logged
    'sweep_redirect'. Called wherever the rule fires: intake, the researcher's ``check`` (where it nearly always
    fires first: the researcher then revises, so an intake-only redirect never fed the pool) and the re-intake."""
    keys = NV.forward(store, candidate, source, now, ctx.cfg.novelty.major_ratio,
                      mv=relative_moves(ctx, store, candidate, exclude))
    store.log("sweep_redirect", node_id=source, knobs=keys, **fields)
    return keys


class _Before:
    """``store`` without the ``later`` nodes: a re-check sees the nodes that existed when its node was admitted."""

    def __init__(self, store: Any, later: set[str]) -> None:
        self._store, self._later = store, later

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)

    def nodes(self, *args: Any, **kw: Any) -> list[Any]:
        return [n for n in self._store.nodes(*args, **kw) if n.id not in self._later]


def recheck(ctx: LoopContext, store: Any, snapshot: Any, node: Any, forward: bool = False) -> tuple[str, str]:
    """The config near-duplicate rules (sweep_only / below_noise, near_dup_config), the mechanism rule
    (dead_mechanism / near_dup_mechanism) and unreachable_schedule re-run on an already admitted QUEUED node against the nodes that existed
    when it was admitted (itself and later ones excluded; their CURRENT statuses): (rule, fix) or ('', '').
    ``forward``: a sweep's knobs go to the design knob pool."""
    try:
        text = open(node.diff_path).read() if node.diff_path else ""
    except OSError:
        return "", ""
    parent = store.get_node(node.parent_id or ROOT_NODE_ID)
    h = store.get_hypothesis(node.hypothesis_id or "")
    ho = next(iter(store.handoffs(node_id=node.id)), None) or SimpleNamespace(actions="", learnings="", concerns="")
    cand = SimpleNamespace(config_diff=store.get_state(f"config_diff:{node.id}") or {}, diff_text=text,
                           changed_paths=tuple(re.findall(r"^diff --git a/(\S+)", text, re.M)), handoff=ho,
                           parent_commit=parent.commit if parent else "",
                           hypothesis=h if h is not None else SimpleNamespace(levers=node.levers))
    if node.operator is Operator.DEBUG or design_point(node):
        return "", ""
    later = {n.id for n in store.nodes() if (n.created_ts, n.id) >= (node.created_ts, node.id)}
    if _config_only(cand):
        near = nearest(ctx, store, cand, later)
        rule = sweep_rule(ctx, store, snapshot, cand, node.strategy, node.operator, near=near)
        if rule[0]:
            if forward:
                redirect(ctx, store, cand, node.id, later, request_id="reintake")
            return rule
        if (twin := config_twin(ctx, store, cand, later)) is not None:
            return "near_dup_config", (f"{twin.id} ({twin.mechanism_tag}) already has this config (same code, every "
                                       f"value within {ctx.cfg.novelty.major_ratio:g}x)")
    if h is not None and (rule := NV.mech_dup_rule(ctx, _Before(store, later), cand, node.strategy,
                                                   operator=node.operator))[0]:
        return rule
    return ("unreachable_schedule", why) if (why := NV.unreachable(store, text)) else ("", "")


def common_ancestor(store: Any, a: str, b: str) -> str:
    """The nearest node that is ``a``, ``b`` or an ancestor of both (root when none): a
    crossover's diff and its paired control are both taken against it."""
    def chain(i: str | None) -> list[str]:
        out: list[str] = []
        while i and i not in out:
            out.append(i)
            n = store.get_node(i)
            i = n.parent_id if n is not None else None
        return out
    theirs = set(chain(b))
    return next((i for i in chain(a) if i in theirs), ROOT_NODE_ID)


def _family_hits(ctx: LoopContext, item: str, kind: str) -> list[str]:
    rules = ctx.cfg.families
    return [f for f, r in rules.items() if any(fnmatch(item, g) for g in getattr(r, kind) if g != "*")] \
        or [f for f, r in rules.items() if "*" in getattr(r, kind)]


def item_families(ctx: LoopContext, candidate: Candidate, item: str, kind: str) -> list[str]:
    """The families of a changed path or config key. A brand-new key no rule names (``dino.sam_rho``) is not
    unmapped: it takes the declared family when its top-level section is one of that family's, else the family of
    the code that reads it (the candidate's changed code paths, when they map to one), else its section's one
    family, else the declared family."""
    if (hit := _family_hits(ctx, item, kind)) or kind != "keys":
        return hit
    rules, declared = ctx.cfg.families, candidate.hypothesis.family
    sect = [f for f, r in rules.items() if any(g.startswith(item.split(".")[0] + ".") for g in r.keys)]
    code = {h[0] for p, k in _family_items(candidate) if k == "paths" and len(h := _family_hits(ctx, p, k)) == 1}
    for fams in ([declared] if declared in sect else [], sorted(code) if len(code) == 1 else [],
                 sect if len(sect) == 1 else [], [declared] if declared in rules else []):
        if fams:
            return fams
    return []


def _family_items(candidate: Candidate) -> list[tuple[str, str]]:
    paths = [p for p in candidate.changed_paths
             if not (candidate.config_diff and p.endswith(_CONFIG_SUFFIXES))]
    return [(k, "keys") for k in candidate.config_diff] + [(p, "paths") for p in paths]


def derive_families(ctx: LoopContext, candidate: Candidate) -> tuple[tuple[str, ...], str]:
    """(families touched, problem). Every changed config key and path must map to exactly
    one rule (:func:`item_families`: a new key takes a family); a ``*`` glob is a catch-all used only when nothing
    else matches. Config files are judged by their changed keys when ``config_diff`` is given."""
    found: set[str] = set()
    for item, kind in _family_items(candidate):
        hit = item_families(ctx, candidate, item, kind)
        if len(hit) != 1:
            return tuple(sorted(found)), "ambiguous_family" if hit else "unmapped_path"
        found.add(hit[0])
    return tuple(sorted(found)), ""


def family_of(ctx: LoopContext, item: str, kind: str = "keys") -> str:
    """The one family a config key (kind='keys') or path maps to; '' when none or ambiguous."""
    hit = _family_hits(ctx, item, kind)
    return hit[0] if len(hit) == 1 else ""


def stackable(ctx: LoopContext, store: Any, snapshot: Any) -> dict[str, tuple[Any, float, str]]:
    """config key -> (value, shrunk P(U>0), node id) of the already-positive knobs a compound may stack:
    a scored, un-vetoed, non-rejected node with P(U>0) >= ``intake.compound_min_p``."""
    if snapshot is None:
        return {}
    out: dict[str, tuple[Any, float, str]] = {}
    for n in store.nodes():
        post = snapshot.nodes.get(n.id)
        if (post is None or n.status is NodeStatus.REJECTED or post.vetoed or n.id == ROOT_NODE_ID
                or post.p_positive < ctx.cfg.intake.compound_min_p):
            continue
        for k, v in (store.get_state(f"config_diff:{n.id}") or {}).items():
            if k not in out or post.p_positive > out[k][1]:
                out[k] = (v, post.p_positive, n.id)
    return out


def compound_split(ctx: LoopContext, store: Any, snapshot: Any, candidate: Candidate) -> tuple[str, str] | None:
    """(stacked family, new family) when a 2-family diff is a compound: one family changes only config keys
    that equal already-positive knobs; the other family is the new change. None otherwise."""
    fams, problem = derive_families(ctx, candidate)
    if problem or len(fams) != 2:
        return None
    stack = stackable(ctx, store, snapshot)
    norm = lambda v: json.dumps(v, sort_keys=True, default=str)   # noqa: E731
    for f, new in (fams, fams[::-1]):
        items = [(i, k) for i, k in _family_items(candidate) if f in item_families(ctx, candidate, i, k)]
        if items and all(k == "keys" and i in stack and norm(stack[i][0]) == norm(candidate.config_diff[i])
                         for i, k in items):
            return f, new
    return None


def compound_sources(ctx: LoopContext, store: Any, snapshot: Any, candidate: Candidate,
                     stacked_family: str) -> tuple[str, ...]:
    """The node ids that supplied the stacked family's knobs (best P(U>0) per key), first-seen order."""
    stack = stackable(ctx, store, snapshot)
    out: list[str] = []
    for i, k in _family_items(candidate):
        if k == "keys" and stacked_family in item_families(ctx, candidate, i, k) and i in stack and stack[i][2] not in out:
            out.append(stack[i][2])
    return tuple(out)


def needs_init_compat(ctx: LoopContext, candidate: Candidate, fams: tuple[str, ...] = ()) -> bool:
    """An init_check family, or any change under ``init_compat.paths``."""
    cfg = ctx.cfg
    return any(cfg.families[f].init_check for f in fams if f in cfg.families) or any(
        path_matches(p, pat) for p in candidate.changed_paths for pat in cfg.init_compat.paths)


INIT_SWAP_URL = "model.pretrained_url"
SWAP_KEYS = ("model.type", "model.pretrained_url")   # an init swap's keys (loop.adapters.initcompat): may span families
SHARE_ARMS = (Strategy.EXPLORE, Strategy.MOONSHOT)   # new ideas the theme share guard applies to


def family_rule(ctx: LoopContext, store: Any, snapshot: Any, candidate: Candidate, operator: Operator | None,
                strategy: Strategy | None, fams: tuple[str, ...]) -> tuple[str, str]:
    """The declared-family rules on mapped families ``fams``: the declared family is the diff's PRIMARY family and
    must be one it touches. Its code may span families (a mechanism's code lives where it lives: layerdrop in
    model.py with its dino.drop_path_rate knob); its config KEYS may span families only for a crossover, a moonshot,
    an init swap (:data:`SWAP_KEYS`) or a 2-family compound stacking positive knobs (declared as its NEW family).
    A ``combine`` family it touches is the declared one and pairs with ONE other."""
    cfg, declared = ctx.cfg, candidate.hypothesis.family
    if operator is Operator.CROSSOVER:
        return "", ""
    lead = next((f for f in fams if cfg.families[f].combine), None)
    if lead:
        if declared != lead:
            return "family_mismatch", f"the diff touches {lead} code; declare family {lead!r}"
        if len(fams) > 2 and strategy is not Strategy.MOONSHOT:
            return "multi_family", f"the diff spans {list(fams)}; {lead} may pair with ONE other family, revert the rest"
        return "", ""
    split = compound_split(ctx, store, snapshot, candidate) if len(fams) == 2 else None
    if split and declared != split[1]:    # a compound is declared as its NEW family
        return "family_mismatch", f"stacking {split[0]!r}'s positive knobs on a {split[1]!r} change; declare {split[1]!r}"
    if declared not in fams:
        return "family_mismatch", (f"the diff spans {list(fams)}; declare the main one of them, not {declared!r}"
                                   if len(fams) > 1 else
                                   f"the diff is family {list(fams)}, not {declared!r}; declare {fams[0] if fams else '?'}")
    kf = sorted({f for i, k in _family_items(candidate) if k == "keys" for f in item_families(ctx, candidate, i, k)})
    if len(kf) > 1 and not split and strategy is not Strategy.MOONSHOT \
            and not any(k in SWAP_KEYS for k in candidate.config_diff):
        return "multi_family", (f"the config keys span families {kf}; keep ONE family's knobs and revert the rest (or "
                                f"stack only knobs listed as stackable positive knobs, P(U>0) >= {cfg.intake.compound_min_p})")
    return "", ""


def base_unladdered(store: Any) -> set[str]:
    """Challenge-the-base nodes (``brief=base``) whose 3-point dose search (a dose ladder, or >= 2 variant points,
    loop.variants) has not run yet: EXEMPT from making their
    mechanism dead (operator 2026-10-06: base components got 1-2 untuned shots each, then dead_mechanism)."""
    from loop import tune
    from loop.variants import bases
    ran = {c["base"] for c in tune.book(store)["batches"] if c.get("kind") == "ladder" and len(c.get("doses") or {}) >= 2}
    ran |= bases(store, 2)                    # its variants already searched >= 2 more doses
    return {n.id for n in store.nodes() if " brief=base" in f" {n.notes or ''}" and n.id not in ran}


def mech_rule(ctx: LoopContext, store: Any, candidate: Candidate, strategy: Strategy | None,
              operator: Operator | None) -> tuple[str, str]:
    """loop.novelty.mech_dup_rule, except that a dead_mechanism whose every dead near node is an un-laddered
    base-challenge node (:func:`base_unladdered`) passes (advisory)."""
    rule, fix = NV.mech_dup_rule(ctx, store, candidate, strategy, operator=operator)
    if rule == "dead_mechanism" and (exempt := base_unladdered(store)):
        gone = NV.dead(store) & {n.id for n, _ in NV.mech_dups(ctx, store, candidate)}
        if gone and gone <= exempt:
            return "", f"(advisory) near base-challenge node(s) {sorted(gone)} whose dose ladder has not run yet"
    return rule, fix


def repeat_rule(store: Any, candidate: Candidate, vec: Any) -> tuple[str, str]:
    """``repeat_of_negative`` (loop.core.ideation): the claim + changed lines sit within the anti-repeat cosine of a
    node with a negative measured effect, unless ``differs_from`` names it. Inert until the v3 tick writes the set."""
    from loop.core import ideation
    return ideation.repeat_rule(store, candidate, vec)


def diagnose(ctx: LoopContext, store: Any, candidate: Candidate, *, operator: Operator | None,
             embedder: Any, strategy: Strategy | None = None, snapshot: Any = None,
             signature: tuple[str, Sequence[str]] | None = None, judge: Any = None,
             dup_exempt: bool = False) -> tuple[str, str]:
    """(rule, one-line fix) of the first deterministic rule that fires; ('', '') = passes. ``dup_exempt`` (an operator
    program arm, run on purpose): the similarity dedups (embedding near_dup, signature, repeat_of_negative, mechanism)
    never fire -- arm A2 IS a near-copy of A1, and the arms re-run a node's mechanism; exact_dup still does.
    Init-compat and relevance are checked by ``admit`` only. ``signature`` (v3-c6i: the mechanism signature text and
    the parent ids, :mod:`loop.core.mechanisms`): the claim + diff ``near_dup`` is replaced by the signature rule
    ``near_dup_signature`` (a crossover's diff carries the incumbent's whole change)."""
    cfg, surface = ctx.cfg, ctx.spec.surface
    paths, keys = candidate.changed_paths, candidate.config_diff
    if not candidate.diff_text.strip() and not paths:
        return "empty_diff", "nothing changed yet; make your edit (set_config / edit_file) first"
    if bad := [p for p in paths if any(path_matches(p, pat) for pat in surface.locked)]:
        return "surface_locked", f"you changed locked {bad}; revert those edits and work around them"
    if surface.writable and (bad := [p for p in paths if not on_surface(p, surface.writable)]):
        return "surface_not_writable", f"{bad} are not editable; revert them (editable: {list(surface.writable)})"
    if bad := [k for k in keys for pk in surface.protected_keys if (k == pk or k.startswith(pk + "."))
               and k != INIT_SWAP_URL]:   # model.pretrained_url: an init swap, judged by init_compat (init_legal)
        return "protected_key", f"{bad} are protected; restore their original values"
    if hit := disallowed_data(candidate.diff_text, keys):   # operator 2026-10-05: usage-restricted data/models
        return "disallowed_data", (f"`{hit[:120]}` references GenBio data or models, which may not be used "
                                   f"(usage restrictions); use the training tiles or another data-bank subset")
    if (hit := illegal_init(ctx, candidate.diff_text, keys))[0]:   # competition rule: no pathology FM init / guide
        return "init_illegal", (f"`{hit[0][:120]}` matches {hit[1]!r} (init_compat.deny): such checkpoints may not "
                                f"initialise or guide the model; any other init is allowed")
    if (url := (keys or {}).get(INIT_SWAP_URL)) is not None and (allow := tuple(getattr(
            ctx.cfg.init_compat, "allow", ()) or ())) and not on_allowlist(str(url), allow):
        return "init_illegal", (f"`{INIT_SWAP_URL}: {str(url)[:100]}` is not on the weights allowlist: allowlisted generic "
                                f"weights: see init_compat.allow")
    if (hit := weight_load(candidate.diff_text, tuple(getattr(ctx.cfg.init_compat, "allow", ()) or ())))[0]:   # v3 review M1: no weights / downloads / checkpoint paths
        return "weight_load", (f"`{hit[1][:120]}` in {hit[0]} loads weights or reads a checkpoint / download source; "
                               f"models start from the declared init only (a new init goes through the model.type "
                               f"init swap, loaded in model.py::load_pretrained); revert it")
    scope, pat, line = forbidden(ctx, derive_families(ctx, candidate)[0], candidate.diff_text, keys)
    if pat:                           # every diff, before any family rule: no route to probe data or labels
        return "forbidden_reference", (f"{scope} code may not read labels, probe datasets/paths/results, "
                                       f"benchmarking data, the environment/filesystem or introspect callers "
                                       f"(and eval_readout may not read tiles beyond the charged calibration "
                                       f"draw): `{line[:120]}` matches {pat!r}; fit on the charged training tiles only")
    noop = dict(getattr(cfg.intake, "noop_keys", None) or {})      # 11-08: a known-inert knob costs no GPU
    if keys and noop and (dead := [k for k in keys if k in noop]):
        return "noop_knob", ("; ".join(f"{k} does nothing in this recipe ({noop[k]})" for k in dead)
                             + ": drop it and change a key that the code reads")
    if (rule := sweep_rule(ctx, store, snapshot, candidate, strategy, operator))[0]:   # a minor numbers-only move
        return rule
    design = strategy is Strategy.DESIGN
    if not (design and operator is None) and (dup := next((
            n for n in store.nodes() if n.diff_hash == candidate.diff_hash and not design_point(n)), None)):
        return "exact_dup", f"this exact diff was already tried as {dup.id}; try a different change"
    trees = store.get_state(K_TREES) or {}
    mine = trees.get(candidate.commit)
    if not (design and operator is None) and candidate.commit and (dup := next((
            n for n in store.nodes() if not design_point(n) and n.id != ROOT_NODE_ID
            and (n.commit == candidate.commit or mine and trees.get(n.commit) == mine)), None)):
        return "exact_dup", (f"this exact code + config (commit tree) already is {dup.id} ({dup.mechanism_tag}), "
                             f"whatever its parent: a re-rooted copy would be credited with {dup.id}'s gain; "
                             "change something")
    vec = None if design else embedder.embed(candidate_text(candidate))
    near = None if design or signature is not None or dup_exempt else next((
        n for n in store.nodes() if n.embedding is not None and n.status is not NodeStatus.REJECTED
        and cosine(vec, n.embedding) > cfg.intake.near_dup_cos), None)
    if signature is not None and not design and not dup_exempt:
        from loop.core import mechanisms as MS
        if (rule := MS.dup_rule(ctx, store, signature[0], tuple(signature[1]), candidate.hypothesis.claim,
                                judge=judge))[0]:
            return rule
    if near is not None:
        return "near_dup", (f"too close to {near.id} ({near.family}/{near.mechanism_tag}); change the "
                            f"mechanism or dose, or reword the claim to what is actually new")
    if vec is not None and not dup_exempt and (rule := repeat_rule(store, candidate, vec))[0]:
        return rule
    if not dup_exempt and (rule := mech_rule(ctx, store, candidate, strategy, operator))[0]:
        return rule
    if not design:                    # v3-c8a B2: a dataset already measured below its parent on >= k seeds
        from loop.core import datasets as DS
        if (rule := DS.candidate_rule(ctx, store, candidate))[0]:
            return rule
    if strategy is not None and not design and (why := NV.unreachable(store, candidate.diff_text)):
        return "unreachable_schedule", why
    parent = next((n for n in store.nodes() if n.commit == candidate.parent_commit), None)
    if keys:
        pcfg = parent_config(store, parent.id if parent else None)
        norm = lambda v: json.dumps(v, sort_keys=True, default=str)   # noqa: E731
        if all(k in pcfg and norm(pcfg[k]) == norm(v) for k, v in keys.items()):
            return "parent_contains_knob", f"the parent already has {dict(keys)}; pick a different value or knob"
    elif paths and all(p.endswith(_CONFIG_SUFFIXES) for p in paths):
        return "config_noop", "the config file changed but no value did; set a real value (set_config)"
    # one family per diff (a crossover or a moonshot may span several), and it must be the declared one
    fams, problem = derive_families(ctx, candidate)
    if problem:
        item = next(i for i, k in _family_items(candidate) if len(item_families(ctx, candidate, i, k)) != 1)
        return problem, (f"{item!r} maps to no family; revert it or use a key/path that belongs to one"
                         if problem == "unmapped_path" else
                         f"{item!r} maps to several families {_family_hits(ctx, item, 'keys' if item in keys else 'paths')}; "
                         "revert it")
    declared = candidate.hypothesis.family
    if design:
        return "", ""
    if declared not in cfg.families:
        return "unknown_family", f"family {declared!r} is not one of {sorted(cfg.families)}"
    if (rule := family_rule(ctx, store, snapshot, candidate, operator, strategy, fams))[0]:
        return rule
    if strategy in SHARE_ARMS and operator is Operator.DRAFT:      # the diversity guard (loop.agents.dossiers)
        from loop.agents.dossiers import share_guard, under_explored
        if (got := share_guard(ctx, store, snapshot, candidate.hypothesis.mechanism_tag, declared))[0]:
            return "theme_share", got[1] + f" (under-explored families: {', '.join(under_explored(ctx, store, 4))})"
    if keys and all(p.endswith(_CONFIG_SUFFIXES) for p in paths) and (twin := next((
            n for n in store.nodes(status=[NodeStatus.QUEUED, *ACTIVE_STATUSES])
            if n.family == declared and inflight_twin(ctx, store, keys, store.get_state(f"config_diff:{n.id}") or {})),
            None)):
        return "near_dup_inflight", (f"{twin.id} ({twin.mechanism_tag}) already tests {sorted(keys)} and is "
                                     f"unresolved; a value variant waits for its verdict -- try another mechanism"
                                     + (" (the same dataset content)" if DATA_KEY in keys else ""))
    if keys and all(p.endswith(_CONFIG_SUFFIXES) for p in paths) and operator is not Operator.DEBUG \
            and (twin := config_twin(ctx, store, candidate)) is not None:
        return "near_dup_config", (f"{twin.id} ({twin.mechanism_tag}) already has this effective config (same code, "
                                   f"every value within {ctx.cfg.novelty.major_ratio:g}x): bring a mechanism or a MAJOR move")
    from loop.core import mixed as MX   # v3-c8a B3: LAST (admit runs init-compat after it; the split admits)
    return MX.rule(ctx, store, snapshot, candidate, strategy, operator)


DATA_KEY = "data.dataset_dir"     # the config key a data-bank dataset is trained through (loop.databank)


def dataset_identity(ctx: LoopContext, store: Any, path: Any) -> str:
    """What a ``data.dataset_dir`` value trains on: a data-bank dataset's CONTENT hash (its registered manifest's
    spec_sha, a pending request's spec hash, or the manifest on disk), else the normalised path."""
    if not isinstance(path, str) or not path:
        return f"value:{json.dumps(path, sort_keys=True, default=str)}"
    db = getattr(ctx.cfg, "databank", None)
    if db is not None and db.enabled:
        from loop import databank as DB
        if name := DB.dataset_of(ctx, path):
            bk = DB.book(store)
            if sha := (bk["registered"].get(name) or {}).get("spec_sha"):
                return f"sha:{sha}"
            if (req := bk["requests"].get(name)) and isinstance(req.get("spec"), dict):
                return f"sha:{DB.spec_hash(req['spec'])}"
            try:
                if sha := DB.summary(ctx, name).get("spec_sha"):
                    return f"sha:{sha}"
            except Exception:  # noqa: BLE001 - unreadable manifest: the path stands in
                pass
    return f"path:{os.path.normpath(path)}"


def inflight_twin(ctx: LoopContext, store: Any, keys: Any, theirs: Any) -> bool:
    """near_dup_inflight's twin test: the same changed keys -- and, when the data-bank dataset key is one of them,
    the same dataset CONTENT (10-13: dataset_spec-only finishes onto different datasets were rejected as value
    variants of each other: a node / a node / a node / a node)."""
    theirs = dict(theirs or {})
    if set(theirs) != set(keys):
        return False
    if DATA_KEY in keys:
        return dataset_identity(ctx, store, keys[DATA_KEY]) == dataset_identity(ctx, store, theirs.get(DATA_KEY))
    return True


def admit(ctx: LoopContext, store: Any, candidate: Candidate, *, operator: Operator | None,
          embedder: Any, verdict: dict | None = None, strategy: Strategy | None = None,
          snapshot: Any = None, signature: tuple[str, Sequence[str]] | None = None, judge: Any = None,
          dup_exempt: bool = False) -> str:
    """'' when admitted, else the name of the rule that fired (``judge``: the mechanism-dedup judge,
    :func:`loop.core.mechanisms.dup_rule`)."""
    rule, _ = diagnose(ctx, store, candidate, operator=operator, embedder=embedder, strategy=strategy, snapshot=snapshot,
                       signature=signature, judge=judge, dup_exempt=dup_exempt)
    if rule and rule != "mixed_change":   # mixed_change is diagnose's last rule: the init-compat checks still apply
        return rule
    if needs_init_compat(ctx, candidate, derive_families(ctx, candidate)[0]):
        if candidate.init_compat is None:
            return "init_compat_missing"
        if not candidate.init_compat.ok:
            return candidate.init_compat.rule
    return rule
