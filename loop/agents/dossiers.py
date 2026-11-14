"""loop.agents.dossiers — the base recipe's COMPONENTS and one DOSSIER per component (challenge-the-base, operator
2026-10-06: "question everything in the base system").

Nothing here names a mechanism. Components are DISCOVERED on the base commit (:func:`components`):

* every top-level function / class method of the surface's writable Python files, and inside a long function each
  comment-headed block of statements (``# Fit typicality ...`` starts a block) -- the code says what it does;
* the config keys each unit reads (a leaf name as ``["leaf"]`` / ``'leaf'`` / ``.leaf``; a section read whole
  counts for each of its keys), every config section left unread by any unit as a config-only component;
* data files a unit names (``*.json`` under a configured directory) are summarised (their factor / column lists),
  so a researcher sees what the data offers beyond what the recipe uses (the worktree tools never read /data).

Model-init code (``__init__`` / ``*init*`` / ``*pretrain*`` units of the init-checked files) is out of scope: the
operator's one exception. Locked spans and units that only touch the surface ``forbid`` plumbing are skipped.

A dossier (:func:`dossier`) gives a component its code span(s), what the code says it does, its knobs with current
values, its hard-coded constants, its legality family, every past node that touched it (status, own increment,
per-axis shrunk deltas), and OPEN QUESTIONS computed from that history (never-moved knobs, one-dose attempts,
unexposed constants, unused data factors). :func:`pick` rotates components by UNDER-EXPLORATION (weight 1 / (1 +
recent attempts)), tilted by the world model's allocation signal when one exists -- never by what anyone happened to
mention. :func:`themes` / :func:`share_guard` are the diversity guard: no theme (a mechanism-tag word or a family)
may exceed ``ideation.max_share`` of the recent research draws unless the allocation signal backs it."""

from __future__ import annotations

import ast
import dataclasses
import json
import logging
import math
import re
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from loop.contracts import ROOT_NODE_ID, NodeStatus, Operator, Strategy, design_point

logger = logging.getLogger(__name__)

BLOCK_MIN = 60                  # a function this long is split into its comment-headed blocks
UNIT_MIN = 2                    # a function shorter than this is plumbing
BLOCK_LINES = 3                 # ... a block of a long function shorter than this
K_SEQ = "ideation.base_seq"     # rotation draw counter
_ML = re.compile(r"torch|\bF\.|\bnn\.|loss|grad|np\.|random|tensor|\.mean\(|softmax|normali[sz]|@|matmul|einsum", re.I)
_PLUMB = re.compile(r"(\b|_)log|save|checkpoint|console|wandb|artifact|summary|flush|print|config|resume|snapshot|compile|"
                    r"probe|milestone|diagnostic|throughput|worker|reader|identifier|seed/output|backend|output_dir|wipe|"
                    r"prefetch|fused|bf16|eval_every|val_batches|persistent", re.I)
_DUNDER = re.compile(r"\.__(?!init__|call__)\w+__$")
_INIT = re.compile(r"__init__$|(^|_)init|pretrain|load_state|state_dict|embed|stem|prepare_tokens|patchify", re.I)  # init, input stem
_JSON = re.compile(r"""['"]/?([\w.\-]+\.json)['"]""")
_NUM = re.compile(r"(?<![\w.])(\d+\.\d+(?:e-?\d+)?|\d+e-?\d+)(?![\w.])")
_GENERIC = frozenset({"loss", "weight", "schedule", "tune", "tuned", "auxiliary", "head", "with", "late", "early",
                      "more", "less", "ramp", "dose", "scale", "both", "only", "stack", "plus", "fix", "fixed",
                      "full", "half", "base", "version", "variant", "ladder", "expose", "carrier", "point", "trial"})
_SCHEMA: dict[str, tuple[float, dict]] = {}
_CACHE: dict[str, list["Component"]] = {}
RESEARCH = (Strategy.EXPLORE, Strategy.MOONSHOT, Strategy.EXPLOIT, Strategy.REVISIT)


@dataclass(frozen=True)
class Component:
    id: str                                     # file::Unit or file::Unit#block-title, or config::section
    title: str
    file: str = ""
    start: int = 0                              # 1-based inclusive line span on the base commit
    end: int = 0
    doc: str = ""                               # what the code says (docstring / leading comments)
    keys: tuple[str, ...] = ()                  # config keys it reads
    consts: tuple[str, ...] = ()                # hard-coded numeric constants in it
    data: tuple[str, ...] = ()                  # data files it names
    families: tuple[str, ...] = ()
    lines: frozenset = frozenset()              # its meaningful source lines (history matches diffs on content)


# ------------------------------------------------------------------ discovery
def _git_show(repo: str, commit: str, rel: str) -> str:
    try:
        cp = subprocess.run(["git", "-C", repo, "show", f"{commit}:{rel}"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return cp.stdout if cp.returncode == 0 else ""


def _flatten(d: Any, pre: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in (d or {}).items() if isinstance(d, dict) else ():
        key = f"{pre}{k}"
        if isinstance(v, dict) and v:
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def _leading_comments(lines: Sequence[str], i: int) -> list[str]:
    """The comment lines directly above line index ``i`` (0-based), top first."""
    out = []
    j = i - 1
    while j >= 0 and lines[j].strip().startswith("#"):
        out.append(lines[j].strip().lstrip("#").strip())
        j -= 1
    return out[::-1]


def _units(rel: str, text: str) -> list[tuple[str, str, int, int, str]]:
    """(id, title, start, end, doc) of every top-level function, class method, and comment-headed block of a long
    function (including its nested defs); 1-based spans."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    lines, out = text.splitlines(), []

    def doc_of(node: Any) -> str:
        d = ast.get_docstring(node) or ""
        return " ".join((d or " ".join(_leading_comments(lines, node.lineno - 1 - len(getattr(node, "decorator_list", ()))))).split())

    def title_of(head: list[str]) -> str:
        """The LAST comment sentence group above a statement (stacked comments describe several things)."""
        groups: list[list[str]] = [[]]
        for h in head:
            groups[-1].append(h)
            if h.endswith("."):
                groups.append([])
        return " ".join(next((g for g in reversed(groups) if g), []))

    def children(st: Any) -> list[Any]:
        kids = [*getattr(st, "body", ()), *getattr(st, "orelse", ()), *getattr(st, "finalbody", ())]
        for h in getattr(st, "handlers", ()):
            kids += h.body
        return [k for k in kids if isinstance(k, ast.stmt)]

    def split(stmts: Sequence[Any], name: str, default: str, depth: int = 0) -> list[tuple[str, int, int, str]]:
        blocks: list[list[Any]] = []            # [title, start, end, stmts, doc]
        for st in stmts:
            s0 = st.lineno - len(getattr(st, "decorator_list", ()))
            head = _leading_comments(lines, s0 - 1)
            if head or not blocks:
                blocks.append([title_of(head) or default, s0 - len(head), st.end_lineno, [st], " ".join(head)])
            else:
                blocks[-1][2] = st.end_lineno
                blocks[-1][3].append(st)
        res: list[tuple[str, int, int, str]] = []
        for title, s0, e0, sts, doc in blocks:
            inner = [k for st in sts for k in (children(st) or [st])]
            if e0 - s0 + 1 > BLOCK_MIN and depth < 6 and len(inner) > len(sts) and any(
                    _leading_comments(lines, k.lineno - 1) for k in inner[1:]):
                res += split(inner, name, title, depth + 1)
            else:
                res.append((title, s0, e0, doc))
        return res

    def add_fn(fn: Any, name: str) -> None:
        start, end = fn.lineno, getattr(fn, "end_lineno", fn.lineno)
        if end - start + 1 < UNIT_MIN:
            return
        if end - start + 1 < BLOCK_MIN:
            out.append((f"{rel}::{name}", name, start, end, doc_of(fn)))
            return
        for title, s0, e0, doc in split(fn.body, name, f"{name} (start)"):
            if e0 - s0 + 1 >= BLOCK_LINES:
                short = re.sub(r"\s+", " ", title)[:90]
                out.append((f"{rel}::{name}#{short[:48]}", f"{name}: {short}", s0, e0, doc or title))

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add_fn(node, node.name)
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    add_fn(sub, f"{node.name}.{sub.name}")
    return out


def _reads(src: str, key: str, sections: Mapping[str, list[str]]) -> bool:
    """``src`` reads config ``key``: its leaf as a subscript / quoted name, as an attribute (not a method call, >= 6
    chars), or its whole (small) section subscripted."""
    leaf, top = key.rpartition(".")[2], key.split(".")[0]
    if re.search(rf"""\[\s*['"]{re.escape(leaf)}['"]\s*\]|['"]{re.escape(leaf)}['"]""", src):
        return True
    if len(leaf) >= 6 and re.search(rf"\.{re.escape(leaf)}\b(?!\s*\()", src):
        return True
    return bool(re.search(rf"""\[\s*['"]{re.escape(top)}['"]\s*\]\s*(?!\[)""", src)) and len(sections.get(top, ())) <= 8


def _locked_spans(ctx: Any) -> list[str]:
    return [l for l in ctx.spec.surface.locked if "::" in l]


def _families(ctx: Any, rel: str, unit_name: str, keys: Iterable[str], src: str = "") -> tuple[str, ...]:
    """The families a change to the unit maps to: its config keys', and its file's -- a family SPAN of the file
    when the span names the unit (``file::Class.method``) or starts inside it (``file::<first line>...<last>``)."""
    from loop.select.intake import _family_hits
    fams = set()
    for k in keys:
        fams.update(_family_hits(ctx, k, "keys"))
    spans = [f for f, r in ctx.cfg.families.items() for p in r.paths
             if "::" in p and p.split("::")[0] == rel and (
                 (m := p.split("::")[1].split("...")[0].strip()) in (unit_name, "")
                 or ("..." in p and m and src and m.lstrip("# ") in src))]
    fams.update(spans or _family_hits(ctx, rel, "paths"))
    return tuple(sorted(fams))


def components(ctx: Any, commit: str, repo: str = "") -> list[Component]:
    """The base components on ``commit`` (cached per commit); [] when the code cannot be read."""
    repo = repo or str(getattr(ctx.spec, "repo", "") or "")
    key = f"{repo}@{commit}"
    if key in _CACHE:
        return _CACHE[key]
    surface = ctx.spec.surface
    cfg_rel = next((w for w in surface.writable if w.endswith((".yaml", ".yml"))), "")
    try:
        config = _flatten(yaml.safe_load(_git_show(repo, commit, cfg_rel)) or {}) if cfg_rel and repo else {}
    except yaml.YAMLError:
        config = {}
    protected = tuple(surface.protected_keys)
    forbid = [re.compile(p) for p in getattr(ctx.cfg, "forbid", ())]
    keys = [k for k in config if not any(k == p or k.startswith(p + ".") for p in protected)
            and not any(rx.search(k) for rx in forbid)]
    sections: dict[str, list[str]] = {}
    for k in keys:
        sections.setdefault(k.split(".")[0], []).append(k)
    init_paths = tuple(getattr(ctx.cfg.init_compat, "paths", ()))
    locked = _locked_spans(ctx)
    out: list[Component] = []
    read: set[str] = set()
    for rel in [w for w in surface.writable if w.endswith(".py")]:
        text = _git_show(repo, commit, rel) if repo else ""
        lines = text.splitlines()
        ranges = []                             # family spans ``file::<first line>...<last line>`` -> line ranges
        for fam, r in ctx.cfg.families.items():
            for p in r.paths:
                if p.startswith(rel + "::") and "..." in p:
                    a_, _, b_ = p.split("::", 1)[1].partition("...")
                    a = next((i + 1 for i, l in enumerate(lines) if a_.strip() and a_.strip() in l), 0)
                    b = next((i + 1 for i, l in enumerate(lines) if i + 1 >= a and b_.strip() and b_.strip() in l), 0)
                    if a and b:
                        ranges.append((fam, a, b))
        for uid, title, s0, e0, doc in _units(rel, text):
            name = uid.split("::", 1)[1].split("#")[0]
            src = "\n".join(lines[s0 - 1:e0])
            if any(l.split("::")[0] == rel and name.endswith(l.split("::")[1].split("...")[0].strip()) and
                   "..." not in l for l in locked):
                continue                        # a locked unit: never changeable
            if any(fnmatch(rel, p) for p in init_paths) and _INIT.search(name.split(".")[-1]) and "#" not in uid:
                continue                        # model init: the operator's one exception
            body = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
            code = [l for l in body.splitlines() if l.strip()]
            bad = sum(any(rx.search(l) for rx in forbid) for l in code)
            if _DUNDER.search(name) or any(rx.search(name) for rx in forbid) \
                    or bad and (bad >= 0.3 * len(code) or _PLUMB.search(title)):
                continue                        # probe / labless plumbing (no change may touch it), dunders
            mine = tuple(k for k in keys if _reads(body, k, sections))
            if _PLUMB.search(title) or not _ML.search(body) and not mine and len(body.splitlines()) < UNIT_MIN:
                continue                        # infrastructure / plumbing without a mechanism
            read.update(mine)
            consts = tuple(dict.fromkeys(m for m in _NUM.findall(body) if m not in ("0.0", "1.0", "0.5", "2.0")))[:12]
            data = tuple(dict.fromkeys(_JSON.findall(src)))
            out.append(Component(uid, title, rel, s0, e0, doc[:400], mine, consts, data,
                                 tuple(sorted({*_families(ctx, rel, name, mine, src),
                                               *(f for f, a, b in ranges if s0 <= b and e0 >= a)})),
                                 frozenset(_meaningful(src.splitlines()))))
    for sec, ks in sorted(sections.items()):
        rest = tuple(k for k in ks if k not in read and not _PLUMB.search(k.rpartition(".")[2]))
        if rest and sec not in ("project",):
            out.append(Component(f"config::{sec}", f"config section {sec} (keys no code unit reads by name)",
                                 cfg_rel, 0, 0, "", rest, (), (), _families(ctx, cfg_rel, "", rest)))
    _CACHE[key] = out
    return out


def config_values(ctx: Any, commit: str, repo: str = "") -> dict[str, Any]:
    repo = repo or str(getattr(ctx.spec, "repo", "") or "")
    cfg_rel = next((w for w in ctx.spec.surface.writable if w.endswith((".yaml", ".yml"))), "")
    try:
        return _flatten(yaml.safe_load(_git_show(repo, commit, cfg_rel)) or {}) if cfg_rel and repo else {}
    except yaml.YAMLError:
        return {}


def data_schema(path: str) -> dict[str, Any]:
    """A JSON data file's shape: top-level keys, and for each dict-of-dicts / dict value its key list (cached by
    mtime). {} when unreadable."""
    p = Path(path)
    try:
        mt = p.stat().st_mtime
    except OSError:
        return {}
    if path in _SCHEMA and _SCHEMA[path][0] == mt:
        return _SCHEMA[path][1]
    try:
        obj = json.loads(p.read_text())
    except (OSError, ValueError):
        return {}
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out[k] = sorted(v)[:60] if isinstance(v, dict) else type(v).__name__
    _SCHEMA[path] = (mt, out)
    return out


def _schemas(comp: Component, config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    dirs = [str(v) for v in config.values() if isinstance(v, str) and v.startswith("/") and Path(v).is_dir()]
    out = {}
    for name in comp.data:
        for d in dirs:
            if (s := data_schema(str(Path(d) / name))):
                out[str(Path(d) / name)] = s
                break
    return out


# ------------------------------------------------------------------ history
_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@", re.M)


def touched_spans(diff_text: str) -> dict[str, list[tuple[int, int]]]:
    """file -> old-side line ranges of each hunk (1-based, inclusive)."""
    out: dict[str, list[tuple[int, int]]] = {}
    for part in re.split(r"^diff --git ", diff_text or "", flags=re.M)[1:]:
        m = re.match(r"a/(\S+)", part)
        if not m:
            continue
        for h in _HUNK.finditer(part):
            a, n = int(h.group(1)), int(h.group(2) or 1)
            out.setdefault(m.group(1), []).append((a, a + max(n, 1) - 1))
    return out


def _read(path: str) -> str:
    try:
        return Path(path).read_text(errors="replace") if path else ""
    except OSError:
        return ""


def _meaningful(lines: Iterable[str]) -> list[str]:
    """Stripped code lines that identify a place (not comments, not short / bracket-only lines)."""
    return [t for t in (l.strip() for l in lines) if len(t) >= 16 and not t.startswith("#")]


def hunks(diff_text: str) -> list[tuple[str, list[str]]]:
    """(file, the hunk's OLD-side context + removed lines, meaningful only) per hunk."""
    out = []
    for part in re.split(r"^diff --git ", diff_text or "", flags=re.M)[1:]:
        m = re.match(r"a/(\S+)", part)
        if not m:
            continue
        for body in re.split(r"^@@[^\n]*@@[^\n]*\n", part, flags=re.M)[1:]:
            old = [l[1:] for l in body.splitlines() if l[:1] in (" ", "-")]
            out.append((m.group(1), _meaningful(old)))
    return out


def node_components(store: Any, node: Any, comps: Sequence[Component]) -> list[str]:
    """The components a node touched: a hunk of its diff whose old-side lines (context + removed) occur in the
    component's source -- >= 2 of them, or its only one (CONTENT, so line drift across commits does not matter) --
    or its config diff moving one of the component's keys."""
    hs = hunks(_read(node.diff_path))
    keys = set(store.get_state(f"config_diff:{node.id}") or {})
    out = []
    for c in comps:
        hit = keys & set(c.keys) or any(f == c.file and old and len(set(old) & c.lines) >= min(2, len(set(old)))
                                        for f, old in hs)
        if hit:
            out.append(c.id)
    return out


def history(ctx: Any, store: Any, snapshot: Any, comps: Sequence[Component]) -> dict[str, list[Any]]:
    """component id -> the non-rejected, non-design nodes that touched it, oldest first."""
    out: dict[str, list[Any]] = {c.id: [] for c in comps}
    for n in sorted(store.nodes(), key=lambda n: (n.created_ts, n.id)):
        if n.id == ROOT_NODE_ID or n.status is NodeStatus.REJECTED or design_point(n):
            continue
        for cid in node_components(store, n, comps):
            out[cid].append(n)
    return out


def _alloc(store: Any, snapshot: Any) -> dict[str, float]:
    """The predictive world model's allocation as {mechanism type / family: share} (loop.diagnostics.allocation;
    {} without the model)."""
    from loop import diagnostics as DG
    try:
        return DG.allocation(store, snapshot)
    except Exception as exc:  # noqa: BLE001 -- analytics never blocks ideation
        logger.warning("diagnostics.allocation failed: %r", exc)
        return {}


def allocation(store: Any, snapshot: Any) -> dict[str, float]:
    return _alloc(store, snapshot)


def diag_summary(store: Any, node_id: str) -> str:
    """The diagnostic layer's summary of a node's runs: numbers, bottleneck flags, root -> node trend ('' when none)."""
    from loop import diagnostics as DG
    try:
        return DG.summary(store, node_id)[:3000]
    except Exception:  # noqa: BLE001 -- no diagnostics table / rows
        return ""


def optimizer_flagged(store: Any, snapshot: Any, node_id: str | None = None) -> bool:
    """The diagnostic layer flags the optimisation itself (loop.diagnostics.OPTIMISER_FLAGS: update starvation, a
    schedule that never annealed, clip-dominated steps) on ``node_id``'s runs, else the incumbent's: multi-variant
    grids then co-tune the shared optimiser knobs."""
    from loop import diagnostics as DG
    inc = getattr(snapshot, "incumbent_id", "") or ROOT_NODE_ID
    try:
        return bool(DG.optimiser_flags(store, [x for x in (node_id, inc) if x]))
    except Exception:  # noqa: BLE001
        return False


# ------------------------------------------------------------------ rotation + dossier
def research_nodes(store: Any, window: int) -> list[Any]:
    """The last ``window`` research-created nodes (the arms' own ideas: no design point, ladder / tuning / leg /
    compute side product), newest last."""
    keep = [n for n in store.nodes() if n.id != ROOT_NODE_ID and n.strategy in RESEARCH and not design_point(n)
            and not re.search(r"\b(tune_of|legs_of|backrun_of|compute)=", n.notes or "")]
    return sorted(keep, key=lambda n: (n.created_ts, n.id))[-window:]


def pick(ctx: Any, store: Any, snapshot: Any, comps: Sequence[Component], hist: Mapping[str, list[Any]],
         rng: Any) -> Component | None:
    """A component drawn with weight 1 / (1 + attempts) -- under-exploration -- x (1 + its allocation weight) when the
    world model's allocation signal names it (by component id, file::unit, or one of its families)."""
    if not comps:
        return None
    alloc = _alloc(store, snapshot)

    def w(c: Component) -> float:
        a = max([alloc.get(c.id, 0.0), alloc.get(c.id.split("#")[0], 0.0), *(alloc.get(f, 0.0) for f in c.families)])
        return (1.0 + max(0.0, a)) / (1.0 + len(hist.get(c.id, ())))
    ws = [w(c) for c in comps]
    x, acc = rng.random() * sum(ws), 0.0
    for c, wi in zip(comps, ws):
        acc += wi
        if x <= acc:
            return c
    return comps[-1]


def _fmt(v: Any, n: int = 60) -> str:
    t = v if isinstance(v, str) else json.dumps(v, default=str)
    return t if len(t) <= n else t[:n - 3] + "..."


def open_questions(ctx: Any, store: Any, comp: Component, nodes: Sequence[Any], config: Mapping[str, Any],
                   schemas: Mapping[str, Mapping[str, Any]]) -> list[str]:
    moved: dict[str, set[str]] = {}
    for n in nodes:
        for k, v in (store.get_state(f"config_diff:{n.id}") or {}).items():
            moved.setdefault(k, set()).add(json.dumps(v, default=str, sort_keys=True))
    out = []
    never = [k for k in comp.keys if k not in moved]
    if never:
        out.append("never varied by any node: " + ", ".join(f"{k}={_fmt(config.get(k))}" for k in never[:10]))
    one = [k for k in comp.keys if len(moved.get(k, ())) == 1]
    if one:
        out.append("tried at ONE other value only (no dose response): " + ", ".join(one[:8]))
    if comp.consts:
        out.append("hard-coded constants, not exposed as config (an assumption nobody tested): "
                   + ", ".join(comp.consts[:10]))
    closed = [n for n in nodes if n.status in (NodeStatus.SCREENED_OUT, NodeStatus.NOT_CONFIRMED, NodeStatus.VETOED)]
    if closed and len(closed) == len(nodes):
        out.append(f"all {len(closed)} attempt(s) closed, most after a single untuned shot: was the dose or the "
                   "placement wrong, or the mechanism?")
    if not nodes:
        out.append("never touched by any node: every assumption in it is untested")
    used = json.dumps([config.get(k) for k in comp.keys], default=str)
    for path, sch in schemas.items():
        shown: list[set] = []
        for part, names in sorted(sch.items(), key=lambda kv: -len(kv[1]) if isinstance(kv[1], list) else 0):
            if isinstance(names, list) and names and not any(set(names) <= s for s in shown):
                shown.append(set(names))
                unused = [x for x in names if not re.search(rf"\b{re.escape(str(x))}\b", used)]
                if unused and len(unused) < len(names) + 1:
                    out.append(f"{Path(path).name}[{part}] offers {len(names)} factors, the recipe uses "
                               f"{len(names) - len(unused)}; unused: {', '.join(map(str, unused))}")
    return out


def dossier(ctx: Any, store: Any, snapshot: Any, comp: Component, nodes: Sequence[Any], config: Mapping[str, Any],
            commit: str = "", repo: str = "", max_nodes: int = 14) -> str:
    """The component's dossier as brief text."""
    repo = repo or str(getattr(ctx.spec, "repo", "") or "")
    lines = [f"COMPONENT `{comp.id}` -- {comp.title}"]
    if comp.start:
        lines.append(f"Code: {comp.file} lines {comp.start}-{comp.end} on the incumbent's commit "
                     "(read it with read_file before anything else)")
    if comp.doc:
        lines.append(f"What the code says it does: {comp.doc[:400]}")
    if comp.start and repo and commit:
        src = _git_show(repo, commit, comp.file).splitlines()[comp.start - 1:comp.end]
        if src and len(src) <= 40:
            lines.append("Source:\n" + "\n".join(f"  {comp.start + i:5d} {l}" for i, l in enumerate(src)))
    if comp.keys:
        lines.append("Knobs (current values): " + ", ".join(f"{k}={_fmt(config.get(k))}" for k in comp.keys[:16]))
    lines.append(f"Legality: family {' / '.join(comp.families) or '(maps to none: changing it is rejected)'}")
    schemas = _schemas(comp, config)
    for path, sch in schemas.items():
        lines.append(f"Data it reads, {path} (the worktree cannot read /data: this is the full list):")
        lines += [f"  {k}: {', '.join(map(str, v)) if isinstance(v, list) else v}" for k, v in sch.items()]
    posts = getattr(snapshot, "nodes", None) or {}
    if nodes:
        lines.append(f"Every past node that touched it ({len(nodes)}; own increment over its parent, per-axis shrunk "
                     "deltas vs root):")
        for n in list(nodes)[-max_nodes:]:
            p = posts.get(n.id)
            cd = store.get_state(f"config_diff:{n.id}") or {}
            ax = " ".join(f"{a.split('_')[2] if a.count('_') >= 2 else a}={q.mean:+.4f}"
                          for a, q in (p.axes.items() if p is not None else ()))
            lines.append(f"  {n.id} {n.mechanism_tag[:40]} {n.status.value}"
                         + (f" own {p.increment.mean:+.4f}±{p.increment.sd:.4f}" if p is not None else " (unscored)")
                         + (f" | {ax}" if ax else "") + (f" | cfg {_fmt(cd, 80)}" if cd else ""))
        if len(nodes) > max_nodes:
            lines.append(f"  ... {len(nodes) - max_nodes} older omitted")
    else:
        lines.append("No node has touched it yet.")
    if oq := open_questions(ctx, store, comp, nodes, config, schemas):
        lines.append("Open questions (computed from the history above):")
        lines += [f"  - {q}" for q in oq]
    return "\n".join(lines)


def overview(comps: Sequence[Component], hist: Mapping[str, list[Any]], limit: int = 60) -> str:
    """One line per component: id, attempts so far (least explored first)."""
    rows = sorted(comps, key=lambda c: (len(hist.get(c.id, ())), c.id))[:limit]
    return "\n".join(f"- {c.id} ({len(hist.get(c.id, ()))} node(s)){' [' + '/'.join(c.families) + ']' if c.families else ''}"
                     for c in rows)


# ------------------------------------------------------------------ the challenge-the-base family
def incumbent_commit(store: Any, snapshot: Any) -> tuple[str, str]:
    inc = getattr(snapshot, "incumbent_id", "") or ROOT_NODE_ID
    node = store.get_node(inc)
    return inc, (node.commit if node is not None else "")


def base_request(selector: Any, store: Any, snapshot: Any) -> Any:
    """With probability ``ideation.base_share`` the bandit's research draw becomes a ``base`` brief (strategy explore,
    DRAFT on the incumbent) attacking ONE component drawn by :func:`pick`; None otherwise (or when no component
    can be discovered on the incumbent's commit)."""
    ctx, ic = selector.ctx, selector.cfg.ideation
    if not ic.enabled or ic.base_share <= 0 or selector.rng.random() >= ic.base_share:
        return None
    inc, commit = incumbent_commit(store, snapshot)
    comps = components(ctx, commit) if commit else []
    comp = pick(ctx, store, snapshot, comps, history(ctx, store, snapshot, comps), selector.rng) if comps else None
    if comp is None:
        return None
    from loop.select.islands import IslandBook
    fam = comp.families[0] if len(comp.families) == 1 else ""
    req = selector._mint(store, snapshot, strategy=Strategy.EXPLORE, operator=Operator.DRAFT,
                         island=IslandBook(selector.cfg, store).island_of(fam) if fam else IslandBook(selector.cfg, store).islands[0],
                         parent_ids=(inc,), brief_kind="base", family=fam, component=comp.id)
    if hasattr(store, "log"):
        store.log("base_challenge", request_id=req.id, component=comp.id, incumbent=inc)
    return dataclasses.replace(req, persona="", cell="")        # the component is the plan, not a MAP-Elites cell


def base_brief(ctx: Any, store: Any, snapshot: Any, component_id: str) -> str:
    from loop.agents.briefs import render
    inc, commit = incumbent_commit(store, snapshot)
    comps = components(ctx, commit) if commit else []
    comp = next((c for c in comps if c.id == component_id), None)
    if comp is None:
        return f"CHALLENGE-THE-BASE brief: component {component_id or '?'} could not be read on {inc}'s commit."
    hist = history(ctx, store, snapshot, comps)
    return render("brief_base", incumbent=inc, n_components=len(comps), component=comp.id,
                  dossier=dossier(ctx, store, snapshot, comp, hist[comp.id], config_values(ctx, commit), commit=commit),
                  overview=overview([c for c in comps if c.id != comp.id], hist, limit=40))


# ------------------------------------------------------------------ diversity guard
def node_themes(node: Any) -> set[str]:
    """A node's themes: its normalized mechanism-tag words (stop words out) and its family (``family:<f>``)."""
    from loop.novelty import tag_words
    return {w for w in tag_words(node.mechanism_tag) if w not in _GENERIC} | {f"family:{node.family}"}


def text_themes(tag: str, family: str) -> set[str]:
    from loop.novelty import tag_words
    return {w for w in tag_words(tag) if w not in _GENERIC} | ({f"family:{family}"} if family else set())


def themes(ctx: Any, store: Any) -> dict[str, float]:
    """theme -> share of the recent research nodes carrying it."""
    ic = ctx.cfg.ideation
    recent = research_nodes(store, ic.share_window)
    if not recent:
        return {}
    c = Counter(t for n in recent for t in node_themes(n))
    return {t: k / len(recent) for t, k in c.items()}


def share_guard(ctx: Any, store: Any, snapshot: Any, tag: str, family: str) -> tuple[str, str]:
    """('', '') or (theme, why): the declared mechanism carries a theme above ``ideation.max_share`` of the recent
    research draws (family themes: twice that, families are coarse) that the allocation signal does not back
    (its weight for the theme / family below the theme's share). Inactive until the window is half full."""
    ic = ctx.cfg.ideation
    if not ic.enabled or ic.max_share <= 0 or len(research_nodes(store, ic.share_window)) < ic.share_window // 2:
        return "", ""
    sh, alloc = themes(ctx, store), _alloc(store, snapshot)
    for t in sorted(text_themes(tag, family), key=lambda t: -sh.get(t, 0.0)):
        cap = ic.max_share * (2.0 if t.startswith("family:") else 1.0)
        s = sh.get(t, 0.0)
        if s > cap and alloc.get(t.removeprefix("family:"), 0.0) < s:
            return t, (f"theme {t!r} is {s:.0%} of the last {ic.share_window} research draws (cap {cap:.0%}) and the "
                       "world model's allocation signal does not back more of it; pick an under-explored subsystem")
    return "", ""


def under_explored(ctx: Any, store: Any, limit: int = 8) -> list[str]:
    """Families least present among the recent research draws (the share guard's suggestion)."""
    sh = themes(ctx, store)
    return sorted(ctx.cfg.families, key=lambda f: (sh.get(f"family:{f}", 0.0), f))[:limit]
