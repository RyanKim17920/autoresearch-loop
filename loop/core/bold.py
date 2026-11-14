"""loop.core.bold — the BOLD STRUCTURAL LANE and DIVERGE-THEN-JUDGE (v3-c8b, operator 2026-10-14: the ideas are
"too boring / not unique").

The audit behind it (the last 40 researcher nodes): ~60% re-parameterised terms the incumbent already has (4
layer-wise-LR variants, KDE variants, 3 FINO variants, temperatures, weight-decay multipliers), ~25% were new and even
those DINO add-ons, 0 were new pretext tasks, data sources / curation or schedule structures. The prompts pushed that
way: a mandatory dose knob, per-axis budgets tied to existing bracketed rows, "avoid confidently negative mechanisms",
G-* flags and lever lists anchoring the idea, a dedup check cheapest to pass by renaming an old lever, in-place edit
framing and the step-0 identity rule (zero-init bolt-ons only).

- the ``bold`` brief kind  ONE structural change from :data:`TAXONOMY`; no dose knob, no per-axis budget rows (one
                           overall predicted distribution), no diagnostic / lever anchoring, knob / weight / temperature /
                           LR-schedule changes of existing terms out of scope; multi-file components allowed. Its share:
                           :func:`weight` (``ideation.bold_share`` blended with the kind bandit, floored at
                           ``ideation.bold_floor`` while its n is small).
- step-0 identity RELAXED  :func:`relax_identity`: warm-start from the parent's weights, new modules may be randomly
                           initialised (the init-compat check's identity parts only -- the banned-init checks, the
                           locked caps, eval identity and the locked files stay absolute).
- knob-only rejection      :func:`knob_only` (the diff: only config values / constants of existing code changed) plus a
                           qwen yes/no judge (:func:`knob_rule`) -> intake rule ``bold_knob_only``.
- tuning budget            :func:`tune_room`: <= ``ideation.bold_tune_max`` (5) tuning trials over the node's whole
                           tuned family (loop.tune's dose ladder + tuning pass; never a campaign); it is judged on the best
                           of them through the usual fresh-seed TEST (which accounts for the best-of-<=5 curse).
- diverge-then-judge       the ``propose_slate`` tool (:data:`SLATE_TOOL`, :class:`Slate`): ~10 one-paragraph candidates
                           over >= 4 categories (<= 3 each) BEFORE any edit; each scored originality (distance from the
                           ledger's mechanism clusters and signatures) x plausibility (the session's own score, averaged
                           with a cheap qwen judge when it answers); a candidate in a saturated-negative cluster is
                           rejected; the session builds the top one. Every candidate and score is logged (``slate``).
- the meta report          :func:`mix` / :func:`mix_text`: the share of new nodes per taxonomy category and the share that
                           are knob / weight changes of existing terms (diff-based), per kind ("Originality mix").

Nothing here names an idea: the taxonomy is a set of CATEGORIES, never mechanisms."""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from loop import project as P
from loop.contracts import InitCompat

KIND = "bold"
#: the structural taxonomy: category -> what it covers (shown to the session; the slate's categories)
TAXONOMY: dict[str, str] = {
    "pretext": "a new pretext task / training objective (a new prediction target, a new self-supervised task, a new "
               "loss on new targets -- not a re-weighting of the existing losses)",
    "data": "the data source, curation or tiling (which tiles are trained on, how they are drawn, re-tiled at another "
            "size or magnification, filtered or balanced by a computed signal; the data bank builds new datasets)",
    "architecture": "an architecture component (a new module, token type, attention pattern or branch; may be "
                    "randomly initialised)",
    "schedule": "the training-schedule STRUCTURE (multi-stage training, a curriculum over views / resolution / "
                "difficulty, redistributing the fixed compute across the run -- not an LR / warmup / decay value)",
    "context": "slide / spatial context (neighbouring tiles, several tiles of one slide as one sample, multi-instance "
               "objectives, position within the slide)",
    "pathology": "pathology-specific structure (stain physics and stain variation, magnification, cell / nucleus / "
                 "tissue relations, morphology)",
}
OTHER, KNOB = "other", "knob"


def taxonomy() -> dict[str, str]:
    """TAXONOMY with the installed project's domain rules (taxonomy_<category>) over the defaults (call time)."""
    return {k: P.rule(f"taxonomy_{k}", v) for k, v in TAXONOMY.items()}


def label(k: str) -> str:
    """The category's DISPLAY name (rule ``taxonomy_label_<id>``; default the id): what the session is shown. The id
    stays the internal / ledger key (``bold_cat=`` notes, the schema); a displayed label is accepted back by :func:`canon`."""
    return P.rule(f"taxonomy_label_{k}", k)


def canon(c: Any) -> str:
    """A category as the session wrote it -> the internal id (a displayed label maps back; anything else unchanged)."""
    c = str(c or "")
    if c in TAXONOMY:
        return c
    back = {label(k).lower(): k for k in TAXONOMY}
    return back.get(c.strip().lower(), c)


def category_rx() -> dict[str, str]:
    """CATEGORY_RX with the project's rules (category_rx_<category>) over the defaults (call time)."""
    return {k: P.rule(f"category_rx_{k}", v) for k, v in CATEGORY_RX.items()}
#: classifying an EXISTING node's mechanism into the taxonomy (most specific first; first match wins)
CATEGORY_RX: dict[str, str] = {
    "pathology": r"stain|\bhed\b|macenko|reinhard|magnif|nucle|\bcell|tissue.?(type|relation|graph|class|seg)|morpholog|"
                 r"gland|mitos",
    "context": r"neighbo|spatial|slide.?level|\bslide\b|context|multi.?instance|\bmil\b|\bwsi\b|region|adjacent",
    "data": r"data_curation|data_sampling|dataset|curat|retil|re.?tile|tiling|sampl|filter|balanc|databank|source",
    "schedule": r"stage|curricul|phase|progressive|schedule_clock|compute.?redistrib|multi.?res|resolution.?sched",
    "pretext": r"pretext|objective|predict|reconstruct|jepa|ibot|mask|contrast|distill|cluster|target|loss",
    "architecture": r"architect|adapter|register|attention|block|layer|token|module|branch|moe|conv|head",
}
SATURATED = "saturated-negative"
SLATE_EVENT = "slate"


def opt(ctx: Any, name: str) -> Any:
    from loop.core.levers import opt as o
    return o(ctx, name)


def is_bold_kind(kind: str) -> bool:
    return kind == KIND


def is_bold(node: Any) -> bool:
    """A node made by a bold session (notes ``brief=bold``)."""
    return bool(re.search(r"(?:^|\s)brief=bold(?:\s|$)", getattr(node, "notes", "") or ""))


def slate_kinds(ctx: Any) -> tuple[str, ...]:
    """The brief kinds whose sessions must call ``propose_slate`` before editing: bold always; explore / literature
    while ``ideation.diverge_explore`` (default on)."""
    return (KIND, *(("explore", "literature", *(("repair",) if opt(ctx, "portfolio_repair") and opt(ctx, "portfolio")
                                                 else ())) if opt(ctx, "diverge_explore") else ()))


def portfolio_on(ctx: Any, kind: str) -> bool:
    """v3-c9 C3: does a ``kind`` session propose its directions as a verbalized-sampling PORTFOLIO (6-8 directions with
    probabilities, grounded in distinct history clusters; a repair: a light 3-5)? Not the bold lane (its slate keeps
    the taxonomy-spread form)."""
    return bool(opt(ctx, "portfolio")) and kind in slate_kinds(ctx) and kind != KIND


def slate_limits(ctx: Any, kind: str) -> dict[str, int]:
    """{k_min, k_max, min_cats, per_cat, min_clusters} of a ``kind`` session's slate."""
    if portfolio_on(ctx, kind) and kind == "repair":
        return {"k_min": int(opt(ctx, "portfolio_repair_min")), "k_max": int(opt(ctx, "portfolio_repair_max")),
                "min_cats": min(int(opt(ctx, "slate_min_categories")), 2), "per_cat": int(opt(ctx, "slate_per_category")),
                "min_clusters": int(opt(ctx, "portfolio_repair_clusters"))}
    if portfolio_on(ctx, kind):
        return {"k_min": int(opt(ctx, "portfolio_min")), "k_max": int(opt(ctx, "portfolio_max")),
                "min_cats": int(opt(ctx, "slate_min_categories")), "per_cat": int(opt(ctx, "slate_per_category")),
                "min_clusters": int(opt(ctx, "portfolio_min_clusters"))}
    return {"k_min": int(opt(ctx, "slate_min")), "k_max": int(opt(ctx, "slate_max")),
            "min_cats": int(opt(ctx, "slate_min_categories")), "per_cat": int(opt(ctx, "slate_per_category")),
            "min_clusters": 0}


# ---------------------------------------------------------------------------------------------- the share
def weight(w: Mapping[str, float], n_bold: int, *, share: float, floor: float, floor_n: int, n0: float
           ) -> dict[str, float]:
    """``w`` (the kind bandit's floored shares over the eligible kinds, ``bold`` among them or not yet measured) with
    bold's share = the operator share blended with the bandit's (``n0`` pseudo-sessions at ``share``, ``n_bold``
    measured) -- ``share`` when the bandit has no bold row -- held >= ``floor`` while ``n_bold < floor_n`` (the bandit
    cannot starve it before it has evidence); the other kinds share the rest in their proportions."""
    share, floor = min(max(0.0, share), 1.0), min(max(0.0, floor), 1.0)
    if share <= 0:
        return {k: x for k, x in w.items() if k != KIND}
    bandit = w.get(KIND)
    wb = share if bandit is None else (n0 * share + n_bold * float(bandit)) / (n0 + n_bold) if n0 + n_bold > 0 \
        else share
    if n_bold < floor_n:
        wb = max(wb, floor)
    rest = {k: max(0.0, float(x)) for k, x in w.items() if k != KIND}
    tot = sum(rest.values())
    if not rest:
        return {KIND: 1.0}
    wb = min(wb, 1.0)
    out = {k: (x / tot * (1.0 - wb) if tot > 0 else (1.0 - wb) / len(rest)) for k, x in rest.items()}
    out[KIND] = wb
    return out


def n_bold(led: Any) -> int:
    """Bold nodes of this epoch (the bandit's evidence on the kind)."""
    epoch = getattr(led, "epoch", "") or ""
    return sum(1 for n in getattr(led, "nodes", {}).values()
               if (not epoch or getattr(n, "base", "") == epoch) and is_bold(n))


#: the brief kinds an explore-lane slot can get (what the bold quota is a share of)
LANE_KINDS = ("explore", "literature", "repair", "program", "kuhn", "plateau", "readout", KIND)
K_SINCE = "ideation.bold_since"


def behind(store: Any, pending_kinds: Iterable[str], share: float, kinds: Iterable[str], since: float = 0.0) -> bool:
    """The bold quota: fewer bold sessions than ``share`` of the explore-lane sessions (``research_started`` of
    ``kinds`` + bold since ``since``) plus the queued briefs, the next one included."""
    ks = {*kinds, KIND}
    got = [e.get("brief_kind") for e in store.events("research_started")
           if e.get("brief_kind") in ks and float(e.get("ts") or 0.0) >= since]
    got += [k for k in pending_kinds if k in ks]
    return share > 0 and got.count(KIND) < share * (len(got) + 1)


# ---------------------------------------------------------------------------------------------- init identity
#: the init-compat checks that encode STEP-0 IDENTITY with the parent (a bold node is exempt from these only)
IDENTITY_CHECKS = ("initial_loss", "checkpoint_keys")


def relax_identity(ic: InitCompat | None) -> InitCompat | None:
    """A bold node's init-compat verdict with the step-0 IDENTITY parts relaxed: the initial loss / feature scale may
    differ from the parent's (new modules randomly initialised) and parameters the parent never had may be missing
    from its checkpoint (the parent's weights still warm-start every tensor they have: an UNEXPECTED key -- a renamed,
    resized or dropped parent tensor -- still fails). Nothing else is relaxed: construction, a finite forward, the
    parameter ratio, and an init swap's checks (``init_legal``: no pathology-trained / GenBio checkpoint, the FLOP
    budget) stand as they are; intake's diff rules (init_illegal, disallowed_data, protected keys, locked files) are
    separate and untouched. Returns ``ic`` unchanged when it passed, is a swap, or fails on anything else."""
    if ic is None or ic.ok or ic.swap:
        return ic
    checks = {k: dict(v) if isinstance(v, Mapping) else v for k, v in (ic.checks or {}).items()}
    failed = [n for n, c in checks.items() if isinstance(c, Mapping) and c.get("ok") is False]
    relaxed = []
    il = checks.get("initial_loss")
    if isinstance(il, Mapping) and il.get("ok") is False and "loss_ratio" in il \
            and all(isinstance(il.get(k), (int, float)) and math.isfinite(il[k]) for k in ("loss_ratio",)):
        relaxed.append("initial_loss")
    ck = checks.get("checkpoint_keys")
    if isinstance(ck, Mapping) and ck.get("ok") is False and "missing" in ck and not ck.get("unexpected"):
        relaxed.append("checkpoint_keys")
    if not relaxed or [n for n in failed if n not in relaxed]:
        return ic
    for n in relaxed:
        checks[n] = {**checks[n], "ok": True, "relaxed": "bold: step-0 identity not required"}
    checks["bold_identity_relaxed"] = {"ok": True, "relaxed": relaxed, "was": str(ic.reason)[:600]}
    return InitCompat(True, "", checks)


# ---------------------------------------------------------------------------------------------- knob-only
_NUM = re.compile(r"(?<![A-Za-z_0-9.])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?(?![A-Za-z_])")
_STR = re.compile(r"'[^'\n]*'|\"[^\"\n]*\"")
_CFG = (".yaml", ".yml", ".json", ".toml", ".cfg", ".ini")
#: a config value that changes WHAT is trained on (a data source) is structural, not a knob
STRUCTURAL_KEY = re.compile(r"dataset|data_dir|tiles?_dir|source|manifest|tiling", re.I)
_NEW_CODE = re.compile(r"^\s*(def|class|import|from\s+\S+\s+import|@)\b")


def _norm(line: str) -> str:
    s = _STR.sub("S", line)
    s = s.split("#", 1)[0]
    s = _NUM.sub("N", s)
    for w in ("True", "False", "None"):
        s = re.sub(rf"\b{w}\b", "N", s)
    return re.sub(r"\s+", " ", s).strip()


def parse_diff(diff_text: str) -> list[dict[str, Any]]:
    """Per file: {path, new, deleted, added, removed, context, add_runs, rem_runs} (content lines, no +++ / --- headers;
    ``*_runs`` = the contiguous runs of added / removed lines, the units :func:`structure` parses)."""
    out: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None
    old = ""
    run_a: list[str] | None = None
    run_r: list[str] | None = None

    def close() -> None:
        nonlocal run_a, run_r
        run_a, run_r = None, None

    for line in (diff_text or "").splitlines():
        if line.startswith("diff --git"):
            cur = None
            close()
            continue
        if line.startswith("--- "):
            old = line[4:].strip()
            continue
        if line.startswith("+++ "):
            path = line[4:].strip().removeprefix("b/")
            cur = {"path": path if path != "/dev/null" else old.removeprefix("a/"), "new": old == "/dev/null",
                   "deleted": path == "/dev/null", "added": [], "removed": [], "context": [], "add_runs": [],
                   "rem_runs": []}
            out.append(cur)
            close()
            continue
        if cur is None:
            continue
        if line.startswith("+"):
            cur["added"].append(line[1:])
            if run_a is None:
                run_a = []
                cur["add_runs"].append(run_a)
            run_a.append(line[1:])
            run_r = None
        elif line.startswith("-"):
            cur["removed"].append(line[1:])
            if run_r is None:
                run_r = []
                cur["rem_runs"].append(run_r)
            run_r.append(line[1:])
            run_a = None
        else:
            if line.startswith(" "):
                cur["context"].append(line[1:])
            close()
    return out


# ---- the AST structure comparison (v3 review M3) ----------------------------------------------------------------
import ast  # noqa: E402
import textwrap  # noqa: E402

_CONST_FUNCS = frozenset({"float", "int", "bool", "str", "abs", "min", "max", "round", "pow", "sum", "len", "sqrt", "exp",
                          "log", "log2", "log10", "ceil", "floor", "cos", "sin", "tanh"})
_MODULE_ROOTS = frozenset({"nn", "torch.nn"})
_MODULE_NAMES = frozenset({"Parameter", "Module", "ModuleList", "ModuleDict", "Sequential", "register_buffer",
                           "register_parameter"})
_CTL = (ast.If, ast.For, ast.While, ast.With, ast.Try, ast.AsyncFor, ast.AsyncWith)
_IDENT = re.compile(r"[A-Za-z_]\w*")


def _dotted(n: ast.AST) -> str:
    if isinstance(n, ast.Name):
        return n.id
    if isinstance(n, ast.Attribute):
        return f"{_dotted(n.value)}.{n.attr}" if _dotted(n.value) else n.attr
    return ""


def _const_expr(n: ast.AST) -> bool:
    """A literal or an expression over literals only (0.7*(1+0), float(2), max(0, 1), -3): constant folding."""
    if isinstance(n, ast.Constant):
        return True
    if isinstance(n, ast.UnaryOp):
        return _const_expr(n.operand)
    if isinstance(n, ast.BinOp):
        return _const_expr(n.left) and _const_expr(n.right)
    if isinstance(n, (ast.Tuple, ast.List, ast.Set)):
        return all(_const_expr(e) for e in n.elts)
    if isinstance(n, ast.IfExp):
        return _const_expr(n.test) and _const_expr(n.body) and _const_expr(n.orelse)
    if isinstance(n, ast.Compare):
        return _const_expr(n.left) and all(_const_expr(c) for c in n.comparators)
    if isinstance(n, ast.Call) and not n.keywords:
        f = _dotted(n.func)
        return f.rsplit(".", 1)[-1] in _CONST_FUNCS and (f.count(".") == 0 or f.startswith("math.")) \
            and all(_const_expr(a) for a in n.args)
    return False


def _items_ast(tree: ast.AST) -> Counter:
    out: Counter = Counter()

    def visit(n: ast.AST) -> None:
        if isinstance(n, ast.expr) and _const_expr(n):
            return
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[("def", n.name)] += 1
        elif isinstance(n, ast.ClassDef):
            out[("class", n.name)] += 1
        elif isinstance(n, ast.Import):
            for a in n.names:
                out[("import", a.name)] += 1
        elif isinstance(n, ast.ImportFrom):
            for a in n.names:
                out[("import", f"{n.module}.{a.name}")] += 1
        elif isinstance(n, _CTL):
            out[("ctl", type(n).__name__)] += 1
        elif isinstance(n, ast.Call):
            f = _dotted(n.func)
            last = f.rsplit(".", 1)[-1] if f else ""
            if last in _MODULE_NAMES or f.split(".", 1)[0] == "nn" or f.startswith("torch.nn."):
                out[("module", f or last)] += 1
            elif last:
                out[("call", last)] += 1
        elif isinstance(n, ast.Attribute):
            out[("attr", n.attr)] += 1
        for c in ast.iter_child_nodes(n):
            visit(c)

    visit(tree)
    return out


def _items_regex(line: str) -> Counter:
    """The fallback for a line that is no complete statement (a fragment of a multi-line one)."""
    out: Counter = Counter()
    t = _STR.sub("S", line.split("#", 1)[0])
    if m := re.match(r"\s*(?:async\s+)?def\s+(\w+)", t):
        out[("def", m.group(1))] += 1
    if m := re.match(r"\s*class\s+(\w+)", t):
        out[("class", m.group(1))] += 1
    if m := re.match(r"\s*(?:import|from)\s+(\S+)", t):
        out[("import", m.group(1))] += 1
    if re.match(r"\s*(if|elif|for|while|with|try)\b", t):
        out[("ctl", t.strip().split()[0].rstrip(":"))] += 1
    for m in re.finditer(r"(?<![\w.])(\w[\w.]*)\s*\(", t):
        f = m.group(1)
        last = f.rsplit(".", 1)[-1]
        if last in ("if", "elif", "while", "for", "with", "def", "class", "return", "in", "and", "or", "not", "print"):
            continue
        if last in _CONST_FUNCS and "." not in f:
            continue
        out[("module", f) if last in _MODULE_NAMES or f.startswith("nn.") else ("call", last)] += 1
    for m in re.finditer(r"@(\w[\w.]*)", t):
        out[("call", m.group(1).rsplit(".", 1)[-1])] += 1
    return out


def _parse(lines: Sequence[str]) -> ast.AST | None:
    body = textwrap.dedent("\n".join(lines))
    for src in (body, "if 1:\n" + textwrap.indent(body, " ")):
        try:
            return ast.parse(src)
        except SyntaxError:
            continue
    return None


def run_items(lines: Sequence[str]) -> Counter:
    """The non-constant structure items of one contiguous run of changed lines (comments dropped): the run parsed as
    code, else line by line, else the regex fallback."""
    code = [l for l in lines if l.strip() and not l.lstrip().startswith("#")]
    if not code:
        return Counter()
    if (tree := _parse(code)) is not None:
        return _items_ast(tree)
    out: Counter = Counter()
    for l in code:
        t = _parse([l])
        out += _items_ast(t) if t is not None else _items_regex(l)
    return out


def structure(diff_text: str, parent_src: Callable[[str], str | None] | None = None) -> dict[str, Any]:
    """{n, items, files}: the NON-CONSTANT structure a diff adds to code. Per code file the items of the added runs
    minus those of the removed runs (an edit in place cancels); a def / class / import / nn module or Parameter /
    control statement counts, a call or attribute name counts only when the PARENT never used it (its source via
    ``parent_src(path)``, else the diff's removed + context lines). Constants, constant-folded expressions and dead
    statements (``_ = 0``) have no items. A new / deleted code file is structural."""
    n, kept, files = 0, [], []
    for f in parse_diff(diff_text):
        if f["path"].endswith(_CFG):
            continue
        if f["new"] or f["deleted"]:
            n += 1
            files.append(f["path"])
            continue
        add = sum((run_items(r) for r in f["add_runs"]), Counter())
        rem = sum((run_items(r) for r in f["rem_runs"]), Counter())
        new = add - rem
        src = None
        try:
            src = parent_src(f["path"]) if parent_src is not None else None
        except Exception:  # noqa: BLE001 - the diff's own context stands
            src = None
        known = set(_IDENT.findall(src if src is not None else "\n".join([*f["removed"], *f["context"]])))
        for (kind, name), c in new.items():
            if kind in ("call", "attr") and name.rsplit(".", 1)[-1] in known:
                continue
            n += c
            kept.append(f"{kind}:{name}")
            files.append(f["path"])
    return {"n": n, "items": kept[:12], "files": sorted(set(files))}


def knob_only(diff_text: str, parent_src: Callable[[str], str | None] | None = None) -> bool:
    """True when the diff only changes config VALUES or CONSTANTS inside existing code (v3 review M3: an AST comparison,
    :func:`structure`): no new / deleted code file and no added non-constant structure; a config value naming a data
    source (:data:`STRUCTURAL_KEY`) or a new config key counts as structural. An empty diff is not knob-only."""
    files = parse_diff(diff_text)
    if not files:
        return False
    for f in files:
        if f["new"] or f["deleted"]:
            return False
        if f["path"].endswith(_CFG):
            code = lambda xs: [x for x in xs if x.strip() and not x.lstrip().startswith(("#", "//"))]  # noqa: E731
            add, rem = code(f["added"]), code(f["removed"])
            keys_add = {m.group(1) for x in add if (m := re.match(r"\s*([\w.\-]+)\s*:", x))}
            keys_rem = {m.group(1) for x in rem if (m := re.match(r"\s*([\w.\-]+)\s*:", x))}
            if any(re.compile(P.rule('structural_key', STRUCTURAL_KEY.pattern), re.I).search(k) for k in keys_add | keys_rem) or keys_add - keys_rem:
                return False
    return structure(diff_text, parent_src)["n"] == 0


NOV_SCALE = 5.0            # novelty = 1 - exp(-points / NOV_SCALE): 1 point .18, 3 .45, 5 .63, 10 .86, 20 .98
NOV_W = {"def": 1.0, "class": 1.0, "module": 2.0, "loss": 1.5, "other": 0.2}   # points per new item of each kind
NOV_OTHER_MAX = 10         # the "other" new calls / attributes / imports / control statements count at most this many
_CLASS = re.compile(r"^\s*class\s+(\w+)\s*(?:\(([^)]*)\))?\s*:")
_DEFINED = re.compile(r"^\s*(?:async\s+)?(?:def|class)\s+(\w+)", re.M)
_LOSSY = re.compile(r"loss|criterion", re.I)


def novelty_parts(diff_text: str, parent_src: Callable[[str], str | None] | None = None) -> dict[str, Any]:
    """What a diff adds, measured AGAINST THE PARENT (``parent_src(path)`` = the parent's file, else the diff's removed +
    context lines): {defs, classes, modules, losses, other, points, knob_only}. ``defs`` / ``classes``: new function /
    class definitions the parent never defined (an edit in place of an existing def is none); ``modules``: the new
    classes that subclass an ``nn.Module`` (counted in ``classes`` too, weighted by their own :data:`NOV_W`); ``losses``:
    new identifiers naming a loss term (``*loss*`` / ``*criterion*``) the parent never used; ``other``: the remaining new
    non-constant items (calls, attributes, imports, control statements, module / parameter constructors), capped at
    :data:`NOV_OTHER_MAX`. A knob-only diff (:func:`knob_only` semantics: no non-constant structure) scores no parts."""
    zero = {"defs": 0, "classes": 0, "modules": 0, "losses": 0, "other": 0, "points": 0.0, "knob_only": True}
    st = structure(diff_text, parent_src)
    if st["n"] == 0:
        return zero
    defs, classes, mods, losses, other = set(), set(), set(), set(), 0
    for f in parse_diff(diff_text):
        if f["path"].endswith(_CFG):
            continue
        src = None
        try:
            src = parent_src(f["path"]) if parent_src is not None else None
        except Exception:  # noqa: BLE001 - the diff's own context stands
            src = None
        if f["deleted"]:
            continue
        base = src if src is not None else "\n".join([*f["removed"], *f["context"]])
        have, known = set(_DEFINED.findall(base)), set(_IDENT.findall(base))
        add = sum((run_items(r) for r in f["add_runs"]), Counter())
        rem = sum((run_items(r) for r in f["rem_runs"]), Counter()) if not f["new"] else Counter()
        for (kind, name), c in (add - rem).items():
            if kind == "def" and name not in have:
                defs.add((f["path"], name))
            elif kind == "class" and name not in have:
                classes.add((f["path"], name))
            elif kind in ("call", "attr") and name.rsplit(".", 1)[-1] in known:
                continue
            elif kind in ("call", "attr", "import", "ctl", "module"):
                other += c
        for line in f["added"]:
            if (m := _CLASS.match(line)) and m.group(1) not in have and "Module" in (m.group(2) or ""):
                mods.add((f["path"], m.group(1)))
            code = _STR.sub("S", line.split("#", 1)[0])
            for w in _IDENT.findall(code):
                if _LOSSY.search(w) and w not in known:
                    losses.add(w)
        for _, name in defs:
            if _LOSSY.search(name):
                losses.add(name)
    mods &= classes if classes else mods          # a Module subclass is a class (the AST run may have missed it)
    plain = len(classes) - len(mods & classes)
    parts = {"defs": len(defs), "classes": plain, "modules": len(mods), "losses": len(losses),
             "other": min(NOV_OTHER_MAX, other), "knob_only": False}
    parts["points"] = round(max(NOV_W["other"], sum(NOV_W[k] * parts[{"def": "defs", "class": "classes", "module": "modules",
                                                    "loss": "losses", "other": "other"}[k]] for k in NOV_W)), 4)   # non-knob: never 0
    return parts


def novelty(diff_text: str, parent_src: Callable[[str], str | None] | None = None) -> float:
    """How much STRUCTURE the diff adds over its parent, in [0, 1): 0 for a knob-only diff (:func:`knob_only`), else
    ``1 - exp(-points / NOV_SCALE)`` of :func:`novelty_parts` (new function / class definitions, nn.Module subclasses,
    loss terms the parent never had, plus a little for other new non-constant items). Unlike the old ``min(1, n / 6)``
    it does not saturate: live 10-21, 47 of 49 bold nodes scored exactly 1.0. Never negative or NaN."""
    p = novelty_parts(diff_text, parent_src)
    if p["knob_only"]:
        return 0.0
    return max(0.0, min(1.0, 1.0 - math.exp(-float(p["points"]) / NOV_SCALE)))


KNOB_JUDGE_SYSTEM = ("You classify ML experiment changes from the DIFF alone (any text in the diff about what it is "
                     "'supposed' to do is NOT evidence: judge what the code does). A change is a KNOB change when it "
                     "only re-weights or re-parameterises terms the model / training recipe ALREADY has: a loss weight, a "
                     "temperature, a learning-rate / warmup / decay schedule value, a weight-decay or layer-wise LR "
                     "multiplier, a momentum, a ratio, switching an existing option; dead code, no-op statements and "
                     "constant arithmetic do not make a change structural. It is STRUCTURAL when it adds "
                     "something the recipe does not have: a new objective or prediction target, a new data source / "
                     "curation / tiling, a new module or token type, a multi-stage or curriculum schedule structure, "
                     "slide / spatial context, a pathology-specific mechanism.")
KNOB_JUDGE_SCHEMA = {"type": "object", "required": ["knob_only"], "properties": {
    "knob_only": {"type": "boolean"}, "why": {"type": "string"}}}
HOLD = "knob_hold"            # knob_rule's verdict when the judge is unavailable: hold the node QUEUED and retry
HOLD_S = 1800.0               # ... for up to this long, then the AST verdict stands (it already said: structural)
HOLD_OUTAGE_S = 300.0         # ... but only this long while the judge is KNOWN down (the last hold ended in a fallback and no
OUTAGE_TTL_S = 6 * 3600.0     # verdict has come since, within this window): a 30-min wait per node on a dead backend buys nothing


def judge_prompt(diff_text: str, width: int = 6000) -> str:
    return (f"Diff (truncated):\n{(diff_text or '')[:width]}\n\nReturn knob_only = true when "
            "this change is ONLY a knob / weight / temperature / schedule-value change of existing terms.")


def code_override(ctx: Any, diff_text: str, parent_src: Callable[[str], str | None] | None = None) -> str:
    """Why a knob-judge "knob only" verdict does NOT stand ('' = it stands): the diff adds CODE structure the parent
    lacks (:func:`novelty_parts`: a new def / class / nn.Module / loss term, or >= ``bold_knob_code_min_other`` other
    new non-constant items). 11-10: a node natinit_sincos_pos3d (3 new defs: a 3D sin-cos position-table init) was
    rejected ``bold_knob_only`` by the judge although it changes code, not a knob. ``bold_knob_code_override`` off: ''."""
    if not opt(ctx, "bold_knob_code_override"):
        return ""
    try:
        p = novelty_parts(diff_text or "", parent_src)
    except Exception:  # noqa: BLE001 - an unparsable diff: the judge's verdict stands
        return ""
    if p.get("knob_only"):
        return ""
    core = int(p["defs"]) + int(p["classes"]) + int(p["modules"]) + int(p["losses"])
    if core >= 1 or int(p["other"]) >= int(opt(ctx, "bold_knob_code_min_other")):
        return (f"the diff adds code ({p['defs']} def(s), {p['classes'] + p['modules']} class(es), {p['losses']} loss "
                f"term(s), {p['other']} other new item(s)): not a knob-only change")
    return ""


def knob_rule(ctx: Any, diff_text: str, claim: str = "", judge: Callable[[str], bool | None] | None = None,
              parent_src: Callable[[str], str | None] | None = None) -> tuple[str, str]:
    """('bold_knob_only', fix) for a bold candidate that only re-parameterises existing terms: the AST classifier
    (:func:`knob_only`) -- final when it says knob; else the yes/no judge when given (it sees the DIFF, never the claim).
    FAILS CLOSED: no verdict from the judge (the backend is down) is ``(HOLD, '')`` -- the caller holds the node QUEUED and
    retries for :data:`HOLD_S`, then the AST verdict (structural) stands. Never a silent pass. ``claim`` is unused."""
    why = ""
    if knob_only(diff_text, parent_src):
        why = "the diff only changes config values / constants of existing code"
    elif judge is not None and opt(ctx, "bold_knob_judge"):
        try:
            got = judge(diff_text)
        except Exception:  # noqa: BLE001 - no verdict
            got = None
        if got is None:
            return HOLD, ""
        if got and not code_override(ctx, diff_text, parent_src):
            why = "the knob judge classified it as a re-weighting / re-parameterisation of existing terms"
    if not why:
        return "", ""
    return "bold_knob_only", (f"{why}. The bold lane is for ONE structural change ("
                              + P.rule("knob_scope", "a new pretext task / objective, a "
                                       "data source / curation / tiling, an architecture component, a training-schedule "
                                       "structure, slide / spatial context, a pathology-specific structure")
                              + "); knob / weight / "
                              "temperature / LR-schedule changes of existing terms are out of scope here")


def llm_knob_judge(llm: Any, role: str = "bold_knob_judge") -> Callable[[str], bool | None]:
    def judge(diff_text: str) -> bool | None:
        got = llm.complete_json(role, system=P.rule("knob_judge_system", KNOB_JUDGE_SYSTEM),
                                prompt=judge_prompt(diff_text), schema=KNOB_JUDGE_SCHEMA)
        return bool(got.get("knob_only")) if isinstance(got, Mapping) and "knob_only" in got else None
    return judge


DET_OTHER_MAX = 0          # the deterministic judge: ANY new non-constant item (call / attribute / branch / ...) is structural
_TARGETY = re.compile(r"(target|tgt|label|teacher|pseudo|loss|criterion|gt|pred)", re.I)
_ASSIGN = re.compile(r"^\s*([A-Za-z_][\w.]*)\s*=(?!=)\s*(.+?)\s*$")


def _retargeted(diff_text: str) -> str:
    """Why a diff re-points a loss / prediction target ('' = it does not): a removed and an added assignment to the same
    target-ish name or one whose old value came from a teacher / target / label expression (target / label / teacher / pseudo
    / loss / criterion / gt / pred, or ``t`` / ``y``) with a different, non-numeric
    right-hand side. Built only from names the parent already has, such a diff adds no new item yet changes what is
    learned (``t = self.teacher(x)`` -> ``t = self.student(x)``)."""
    num = re.compile(r"^[-+]?[\d.eE_+-]+$")
    norm = lambda x: re.sub(r"(?<![\w.])[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?", "N", _STR.sub("S", x)).replace(" ", "")  # noqa: E731
    for f in parse_diff(diff_text):
        if f["path"].endswith(_CFG):
            continue
        rem = {m.group(1): m.group(2) for l in f["removed"] if (m := _ASSIGN.match(l.split("#", 1)[0]))}
        for l in f["added"]:
            m = _ASSIGN.match(l.split("#", 1)[0])
            if m and m.group(1) in rem and (_TARGETY.search(m.group(1).rsplit(".", 1)[-1]) or _TARGETY.search(rem[m.group(1)])
                                          or m.group(1) in ("t", "y")) \
                    and norm(m.group(2)) != norm(rem[m.group(1)]) and not num.match(m.group(2)):
                return m.group(1)
    return ""


def det_knob_verdict(diff_text: str, parent_src: Callable[[str], str | None] | None = None) -> tuple[bool, str]:
    """(knob_only, why): the DETERMINISTIC knob judge (10-22: the qwen judge answered 3 of 186 calls in 7 days, so every
    other bold node sat held for HOLD_S and then fell back to the bare AST verdict). From the diff alone, against the
    parent: a diff is a KNOB change when :func:`knob_only` says so, or when :func:`novelty_parts` finds no new function /
    class / nn.Module / loss term and no other new non-constant item (:data:`DET_OTHER_MAX` = 0; names the parent
    already has do not count as new). Anything with a new def, class, module, loss term, call, attribute or control
    statement (a curriculum branch, a new head) is STRUCTURAL, as is a diff that re-points a loss / prediction target
    through names the parent already has (:func:`_retargeted`): fail-open, because a wrongly structural diff only costs a
    bold slot while a wrongly knob-only one is rejected from the structural lane."""
    try:
        p = novelty_parts(diff_text, parent_src)
    except Exception as exc:  # noqa: BLE001 - an unparseable diff: structural (the AST rule's own fail-closed side)
        return False, f"novelty parts unavailable ({type(exc).__name__}): structural"
    if tgt := _retargeted(diff_text):
        return False, f"re-points the target / loss assignment {tgt!r} (fail-open: structural)"
    if p["knob_only"]:
        return True, "no non-constant structure added"
    new = p["defs"] + p["classes"] + p["modules"] + p["losses"]
    if new == 0 and p["other"] <= DET_OTHER_MAX:
        return True, f"no new def / class / module / loss and only {p['other']} other new item(s)"
    return False, (f"adds {p['defs']} def, {p['classes']} class, {p['modules']} module, {p['losses']} loss term, "
                   f"{p['other']} other item(s) (novelty points {p['points']})")


def chained_knob_judge(llm_judge: Callable[[str], bool | None] | None,
                       parent_src: Callable[[str], str | None] | None = None,
                       log: Callable[..., Any] | None = None) -> Callable[[str], bool]:
    """The knob judge that always answers: the LLM judge when it is up (qwen's verdict stands), else the deterministic
    one (:func:`det_knob_verdict`). ``log(event, **fields)`` gets ``bold_knob_judge`` with ``judge`` = 'qwen' |
    'deterministic' and ``det`` (the deterministic verdict beside qwen's, to measure their agreement) on every call."""
    def judge(diff_text: str) -> bool:
        det, why = det_knob_verdict(diff_text, parent_src)
        got = None
        if llm_judge is not None:
            try:
                got = llm_judge(diff_text)
            except Exception as exc:  # noqa: BLE001 - qwen down: the deterministic judge decides
                if log is not None:
                    log("bold_knob_judge_error", error=repr(exc)[:300])
        who, out = ("qwen", bool(got)) if got is not None else ("deterministic", det)
        judge.last = who
        if log is not None:
            log("bold_knob_judge", knob_only=out, judge=who, det=det, det_why=why[:200])
        return out
    judge.last = ""                       # who decided the last call
    return judge


# ---------------------------------------------------------------------------------------------- tuning budget
_VIEW: dict[int, tuple[float, "_StoreView"]] = {}
VIEW_TTL_S = 20.0      # a unit view of the whole node table is O(nodes): reused for a tick's worth of budget questions


def store_view(store: Any) -> "_StoreView":
    import time
    hit = _VIEW.get(id(store))
    if hit is not None and time.monotonic() - hit[0] < VIEW_TTL_S:
        return hit[1]
    v = _StoreView(store)
    _VIEW[id(store)] = (time.monotonic(), v)
    return v


class _StoreView:
    """A ledger-shaped view of the store's nodes (``.nodes`` id -> notes / parent / port_of / base) so
    :func:`loop.core.alloc.unit_of` can resolve a node's unit without a ledger."""

    def __init__(self, store: Any) -> None:
        from types import SimpleNamespace
        self.epoch = ""
        self.nodes = {n.id: SimpleNamespace(id=n.id, notes=n.notes or "", parent=n.parent_id, port_of="", base="")
                      for n in store.nodes()}
        self.memo: dict[str, tuple[str, str]] = {}


def unit_root(store: Any, nid: str, view: "_StoreView | None" = None) -> str:
    """The unit (loop.core.alloc.unit_of: split parts, opposite arms, ports and ladder children belong to their origin's
    unit) a node belongs to; the node itself when it has none (v3 review S1: bold budgets count per UNIT, not per node id)."""
    from loop.core import alloc as AL
    view = view or store_view(store)
    try:
        return AL.unit_of(view, nid, view.memo)[0] or nid
    except Exception:  # noqa: BLE001 - no unit: the node's own budget
        return nid


def unit_bold(store: Any, node: Any, view: "_StoreView | None" = None) -> str:
    """The bold unit root of ``node`` ('' when it is not part of a bold unit): the node itself when bold, else the root of
    its unit when that is a bold node (a split part, an opposite arm, a ladder child of a bold node)."""
    if is_bold(node):
        return unit_root(store, node.id, view)
    root = unit_root(store, node.id, view)
    r = store.get_node(root) if root != node.id else None
    return root if r is not None and is_bold(r) else ""


def tune_used(store: Any, nid: str) -> int:
    """Tuning trials already spent on node ``nid``'s UNIT (loop.tune: every ladder / pass batch whose base belongs to the
    same unit, running or closed)."""
    from loop.tune import book
    bk = book(store)
    view = store_view(store)
    root = unit_root(store, nid, view)
    legacy = sum(len(c.get("trials") or ()) for c in [*bk["batches"], *bk["current"].values()]
                 if c.get("base") and unit_root(store, c["base"], view) == root)
    from loop.core.struct_tune import K_STRUCT      # 10-22: the v3 structural ladder's children share the unit budget
    mine = sum(len(r.get("children") or ()) for n, r in (store.get_state(K_STRUCT) or {}).items()
               if isinstance(r, dict) and unit_root(store, n, view) == root)
    return legacy + mine


def tune_room(ctx: Any, store: Any, node: Any) -> int | None:
    """Tuning trials a BOLD unit may still get (``ideation.bold_tune_max`` minus those spent over the whole unit); None for
    any node outside a bold unit (its own budgets apply)."""
    view = store_view(store)
    root = unit_bold(store, node, view)
    if not root:
        return None
    return max(0, int(opt(ctx, "bold_tune_max")) - tune_used(store, root))


def tuned_live(store: Any, node: Any) -> str:
    """v3 review S1: the id of a tuned node (``tune_of=``) of ``node``'s bold unit still on its way to / through its fresh
    TEST (screening, promoting, confirming), else '' -- at most ONE per bold family at a time."""
    from loop.contracts import ACTIVE_STATUSES, NodeStatus
    view = store_view(store)
    root = unit_bold(store, node, view)
    if not root:
        return ""
    live = ACTIVE_STATUSES | {NodeStatus.QUEUED}
    for n in store.nodes():
        m = re.search(r"\btune_of=(\S+)", n.notes or "")
        if m and n.status in live and (b := store.get_node(m.group(1))) is not None and unit_root(store, b.id, view) == root:
            return n.id
    return ""


# ---------------------------------------------------------------------------------------------- the slate
CANDIDATE = {"type": "object", "required": ["title", "category", "mechanism", "why", "plausibility"], "properties": {
    "title": {"type": "string", "minLength": 1},
    "category": {"type": "string", "enum": list(TAXONOMY)},
    "mechanism": {"type": "string", "minLength": 1},
    "family": {"type": "string"}, "mechanism_tag": {"type": "string"},
    "why": {"type": "string", "minLength": 1},
    "plausibility": {"type": "number", "minimum": 0, "maximum": 1},
    "p_neg": {"type": "number", "minimum": 0, "maximum": 1}, "median": {"type": "number"},
    "p_gap": {"type": "number", "minimum": 0, "maximum": 1},
    "probability": {"type": "number", "minimum": 0, "maximum": 1},      # v3-c9 C3: the portfolio's verbalized probability
    "grounded_in": {"type": "string"},                                   # v3-c9 C3: a history cluster / node id
    "differs_from": {"type": "array", "items": {                         # review fix M3: what it differs from, checked now
        "type": "object", "required": ["id", "how"], "properties": {"id": {"type": "string"}, "how": {"type": "string"}}}},
    # 11-08 (loop.core.tested): a candidate matching an already-tested / in-flight mechanism is bounced unless this
    # names that entry (local:<run>, a node id, slate:<req>#<i>) and says what is concretely different
    "differs_from_tested": {"type": "array", "items": {
        "type": "object", "required": ["id", "how"], "properties": {"id": {"type": "string"}, "how": {"type": "string"}}}}}}
SLATE_TOOL = {"name": "propose_slate", "description": (
    "DIVERGE, THEN JUDGE -- required BEFORE any edit (edit_file / write_file / set_config / a writing bash command are "
    "refused until it is accepted). Propose ~10 one-paragraph candidate ideas spread across the structural taxonomy "
    "(>= 4 categories, at most 3 per category): each {title, category, mechanism (one paragraph: what changes, where, "
    "and why it should help this incumbent), family, mechanism_tag, why (the plausibility argument: evidence / "
    "literature / mechanism), plausibility (0-1, your honest probability that it beats the incumbent), p_neg, median, "
    "p_gap (its effect distribution on the visible composite)}. The loop scores each on originality (distance from "
    "the ledger's mechanism clusters) x plausibility (yours, averaged with an independent judge), rejects candidates "
    "in saturated-negative clusters and names the ONE to build. A candidate that repeats a mechanism of the brief's "
    "'Already tested' list or another session's in-flight work is bounced with that result unless it carries "
    "differs_from_tested: [{id, how}] (what is concretely different)."),
    "input_schema": {"type": "object", "required": ["candidates"], "properties": {
        "candidates": {"type": "array", "items": CANDIDATE}}}}


def slate_tool(ctx: Any, kind: str = KIND) -> dict[str, Any]:
    """The ``propose_slate`` tool definition of a ``kind`` session (the bold lane's, or a v3-c9 portfolio's: its
    own counts, the verbalized ``probability`` and the ``grounded_in`` history id)."""
    return _relabelled(_slate_tool(ctx, kind))


def _relabelled(tool: dict[str, Any]) -> dict[str, Any]:
    """The tool with its category enum showing the DISPLAY labels (identity when no taxonomy_label_* rule is set)."""
    labs = [label(k) for k in TAXONOMY]
    if labs == list(TAXONOMY):
        return tool
    item = {**CANDIDATE, "properties": {**CANDIDATE["properties"], "category": {"type": "string", "enum": labs}}}
    sch = tool["input_schema"]
    return {**tool, "input_schema": {**sch, "properties": {**sch["properties"], "candidates": {"type": "array", "items": item}}}}


def _slate_tool(ctx: Any, kind: str = KIND) -> dict[str, Any]:
    if not portfolio_on(ctx, kind):
        return SLATE_TOOL
    lim = slate_limits(ctx, kind)
    return {**SLATE_TOOL, "description": (
        f"DIVERGE, THEN JUDGE -- required BEFORE any edit (edits are refused until it is accepted). Propose "
        f"{lim['k_min']}-{lim['k_max']} DIRECTIONS as ONE JOINT PORTFOLIO sampled from the full distribution of what "
        "you could do here -- NOT variations of the most typical answer. Each: {title, category, mechanism (one "
        "paragraph: what changes, where, and why it should help), family, mechanism_tag, why, plausibility (0-1: your "
        "honest probability that it beats its base), probability (0-1: the chance a typical researcher given this "
        "brief would propose THIS direction; the probabilities of the portfolio sum to about 1 and at least "
        f"{int(opt(ctx, 'portfolio_tail_min'))} of them are <= {float(opt(ctx, 'portfolio_tail_p')):g}, the tails), "
        f"grounded_in (a cluster id like C7 or a node id from the `history` tool the direction builds away from; the "
        f"portfolio covers >= {lim['min_clusters']} DISTINCT history clusters), p_neg, median, p_gap}}. Spread over >= "
        f"{lim['min_cats']} taxonomy categories (at most {lim['per_cat']} each). The loop scores each on originality x "
        "plausibility, rejects candidates in saturated-negative clusters and names the ONE to build; the finish must "
        "carry differs_from: [{id, how}] -- the history node(s) / cluster(s) it differs from and how (checked here). "
        "A candidate that repeats a mechanism of the brief's 'Already tested' list (manual runs local:<run>, measured "
        "nodes) or another session's in-flight work is bounced with that result unless it carries "
        "differs_from_tested: [{id, how}] saying what is concretely different.")}


def content_chars(text: str) -> int:
    """Characters of explanation in ``text``: its words, with node / cluster ids and punctuation taken out."""
    t = re.sub(r"(?<![\w@])(?:n\d{3,}(?:@\w+)?|C\d+)(?![\w@])", " ", str(text or ""))
    return len(re.sub(r"\s+", " ", re.sub(r"[^A-Za-z0-9 ]+", " ", t)).strip())


CITE_ID = re.compile(r"(?<![\w@])(n\d{3,}(?:@\w+)?|C\d+)(?![\w@])")


def cite_problem(differs: str, exists: Callable[[str], bool] | None, *, min_ids: int = 1, min_chars: int = 12) -> str:
    """'' when ``differs`` (hypothesis.differs_from) cites >= ``min_ids`` history node / cluster ids, every cited id
    exists (``exists``: the history table's; None = not checkable: only the format is), and the explanation (the text
    minus the ids) has >= ``min_chars`` content characters; else what to fix."""
    ids = list(dict.fromkeys(CITE_ID.findall(str(differs or ""))))
    if len(ids) < min_ids:
        return (f"the chosen direction must cite >= {min_ids} history node / cluster id(s) it differs from (a node id "
                "like n1234 or a cluster id like C7 from the `history` tool) in hypothesis.differs_from, with HOW it differs")
    if exists is not None and (gone := [i for i in ids if not exists(i)]):
        return (f"hypothesis.differs_from cites {', '.join(gone)}, which is not in the history table "
                "(use ids returned by the `history` tool)")
    if content_chars(differs) < min_chars:
        return (f"hypothesis.differs_from must say HOW the direction differs from {', '.join(ids)}: >= {min_chars} "
                "characters of explanation besides the ids")
    return ""


STOP = frozenset("""that this with from have been into than then them they their there which would could should about
    over under more most less also only each both same other such these those where when while because without within
    using used uses via per not but and the for are was were has had its our can will may""".split())


def _words(text: str) -> set[str]:
    t = re.sub(r"(?<![\w@])(?:n\d{3,}(?:@\w+)?|C\d+)(?![\w@])", " ", str(text or "").lower())
    return {w for w in re.findall(r"[a-z]{4,}", t) if w not in STOP}


def differs_problem(cand: Mapping[str, Any], tab: Any, *, min_chars: int = 12, sim_min: float = 0.3, min_ids: int = 1,
                    min_novel: int = 2, thin: bool = False) -> str:
    """'' when the candidate's ``differs_from`` ([{id, how}]) is sound against the history table ``tab``: >= ``min_ids``
    entries; every id exists; every id is RELATED to the candidate (same cluster as its ``grounded_in``, same family or
    mechanism tag, or text similarity > ``sim_min`` -- a citation of an unrelated node proves nothing); and every ``how``
    has >= ``min_chars`` content characters and >= ``min_novel`` content words absent from the cited node's description
    (a differs-from that restates the description does not say how it differs). Else what to fix. ``thin`` (11-08:
    the history holds fewer clusters than the portfolio rule asks, :func:`thin_history`): any existing id counts as
    related (a 4-node history of unrelated rejects has nothing related to cite), and an EMPTY table needs no citation."""
    from loop.select.embed import HashingEmbedder, cosine
    if thin and not (getattr(tab, "by_id", None) or getattr(tab, "cluster_ids", None)):
        return ""
    raw = cand.get("differs_from")
    items = [x for x in raw if isinstance(x, Mapping)] if isinstance(raw, (list, tuple)) else []
    if len(items) < min_ids:
        return (f"candidate must carry differs_from: [{{id, how}}] with >= {min_ids} history node / cluster id(s) it "
                "differs from (ids from the `history` tool) and HOW")
    emb = HashingEmbedder()
    ctext = f"{cand.get('title', '')} {cand.get('mechanism', '')} {cand.get('family', '')} {cand.get('mechanism_tag', '')}"
    cvec = emb.embed(ctext)
    g = str(cand.get("grounded_in") or "").strip()
    gcl = (tab.cluster_ids[g].id if g in tab.cluster_ids else tab.node_cluster(g)) if g else ""
    out = []
    for it in items:
        i, how = str(it.get("id") or "").strip(), str(it.get("how") or "")
        if not tab.exists(i):
            out.append(f"{i or '?'} is not in the history table (use ids returned by the `history` tool)")
            continue
        if i in tab.cluster_ids:
            c = tab.cluster_ids[i]
            desc, rel = f"{c.label} {c.category}", i == gcl
            fam = tag = ""
        else:
            r = tab.by_id[i]
            fam, _, tag = str(r["tag"]).partition("/")
            desc = f"{r['desc']} {r['tag']}"
            rel = bool(gcl) and r["cluster"] == gcl
        rel = rel or bool(fam and fam == str(cand.get("family") or "")) \
            or bool(tag and tag == str(cand.get("mechanism_tag") or "")) \
            or cosine(cvec, emb.embed(desc)) > sim_min or thin
        if not rel:
            out.append(f"{i} is unrelated to the candidate (a different cluster / family / mechanism, text similarity "
                       f"<= {sim_min:g}): cite the history node or cluster it actually departs from")
        elif content_chars(how) < min_chars or len(_words(how) - _words(desc)) < min_novel:
            out.append(f"`how` for {i} must say what is DIFFERENT: >= {min_chars} characters and >= {min_novel} content "
                       f"words that {i}'s own description ({desc[:80].strip()!r}) does not contain")
    return "; ".join(out)


def slate_problems(cands: Sequence[Mapping[str, Any]], *, k_min: int, k_max: int, min_cats: int, per_cat: int,
                   portfolio: Mapping[str, Any] | None = None) -> list[str]:
    """What is wrong with a proposed slate (empty: well-formed). ``portfolio`` (v3-c9 C3): {sum_tol, tail_p,
    tail_min, min_clusters, cluster_of (id -> cluster id | '' | None when unknown), exists (id -> bool) | None}:
    each candidate's verbalized ``probability`` (sum within sum_tol of 1; >= tail_min of them <= tail_p) and its
    ``grounded_in`` history id (it exists; >= min_clusters distinct clusters)."""
    out = []
    if not k_min <= len(cands) <= k_max:
        out.append(f"propose {k_min}-{k_max} candidates (you gave {len(cands)})")
    cats = Counter(str(c.get("category") or "") for c in cands)
    bad = sorted(c for c in cats if c not in TAXONOMY)
    if bad:
        out.append(f"unknown categories {bad}: use one of {', '.join(TAXONOMY)}")
    if len([c for c in cats if c in TAXONOMY]) < min_cats:
        out.append(f"spread the slate over >= {min_cats} categories (you used {len(cats)}: {dict(cats)})")
    if over := sorted(c for c, n in cats.items() if n > per_cat):
        out.append(f"at most {per_cat} candidates per category (too many in {', '.join(over)})")
    for i, c in enumerate(cands):
        if len(str(c.get("mechanism") or "").split()) < 12:
            out.append(f"candidate {i}: `mechanism` must be a paragraph (what changes, where, why), not a phrase")
        try:
            p = float(c.get("plausibility"))
            if not 0 <= p <= 1:
                raise ValueError
        except (TypeError, ValueError):
            out.append(f"candidate {i}: plausibility must be a number in [0, 1]")
    from loop.select.embed import HashingEmbedder, cosine
    emb = HashingEmbedder()
    vecs = [emb.embed(f"{c.get('title', '')} {c.get('mechanism', '')}") for c in cands]
    same = [(i, j) for i in range(len(vecs)) for j in range(i + 1, len(vecs)) if cosine(vecs[i], vecs[j]) > 0.9]
    if same:
        out.append(f"candidates {same[0][0]} and {same[0][1]} are one idea: make every candidate a DISTINCT mechanism")
    if portfolio:
        out += portfolio_problems(cands, portfolio)
    return out


def table_clusters(tab: Any) -> set[str]:
    """The distinct cluster ids a candidate can be grounded in on the history table ``tab`` (its cluster list + the
    clusters its node rows carry); empty for no table."""
    if tab is None:
        return set()
    rows = getattr(tab, "by_id", {}) or {}
    return {str(c) for c in (getattr(tab, "cluster_ids", {}) or {})} \
        | {str(r.get("cluster")) for r in rows.values() if isinstance(r, Mapping) and r.get("cluster")}


def thin_history(tab: Any, need: int) -> bool:
    """11-08: does the history table hold FEWER distinct clusters than ``need``? (A fresh ledger: nanobrain ran 1207
    sessions against a 4-node history whose nodes were all rejected at smoke and never clustered; ">= 3 distinct
    clusters" could never be met, so the slate was never accepted and edits stayed locked.) The diversity rules then
    degrade to what the history can carry; with >= ``need`` clusters nothing changes."""
    return tab is not None and len(table_clusters(tab)) < int(need)


def portfolio_problems(cands: Sequence[Mapping[str, Any]], pf: Mapping[str, Any]) -> list[str]:
    out = []
    ps: list[float] = []
    for i, c in enumerate(cands):
        try:
            p = float(c.get("probability"))
            if not 0 <= p <= 1:
                raise ValueError
            ps.append(p)
        except (TypeError, ValueError):
            out.append(f"candidate {i}: `probability` (the chance a typical researcher would propose it) must be a "
                       "number in [0, 1]")
    if len(ps) == len(cands) and cands:
        tol = float(pf.get("sum_tol", 0.35))
        if abs(sum(ps) - 1.0) > tol:
            out.append(f"the portfolio is ONE joint distribution: the probabilities must sum to about 1 (yours sum to "
                       f"{sum(ps):.2f}; tolerance {tol:g})")
        tail_p, tail_min = float(pf.get("tail_p", 0.15)), int(pf.get("tail_min", 2))
        if sum(p <= tail_p for p in ps) < tail_min:
            out.append(f"sample the tails: at least {tail_min} directions must have probability <= {tail_p:g} (not "
                       "variations of the most typical answer)")
    exists, cluster_of = pf.get("exists"), pf.get("cluster_of")
    if exists is not None or cluster_of is not None:
        clusters, bad = set(), []
        n_ids = pf.get("n_ids")
        for i, c in enumerate(cands):
            g = str(c.get("grounded_in") or "").strip()
            if n_ids == 0:                               # 11-08: an EMPTY history: nothing to ground in
                continue
            if not g:
                out.append(f"candidate {i}: `grounded_in` must be a cluster id (C7) or node id from the `history` tool")
                continue
            if exists is not None and not exists(g):
                bad.append(f"{i}: {g}")
                continue
            cl = cluster_of(g) if cluster_of is not None else None
            if cl:
                clusters.add(cl)
        avail = pf.get("available_clusters")
        if bad:
            out.append("`grounded_in` ids not in the history table (candidate: id): " + ", ".join(bad)
                       + (f" (the history has {avail} cluster(s): ground in node ids from history(action='table'), "
                          "e.g. root)" if avail is not None and avail < int(pf.get("min_clusters", 0)) else ""))
        need = int(pf.get("min_clusters", 0))
        if avail is not None:                            # 11-08: a thin history: as many clusters as it has
            need = min(need, int(avail))
        if need and cluster_of is not None and len(clusters) < need and not bad:
            out.append(f"ground the portfolio in >= {need} DISTINCT history clusters (yours cover {len(clusters)}: "
                       f"{sorted(clusters)}); `history(action='clusters')` lists them")
    return out


def _terms(c: Mapping[str, Any]) -> Counter:
    from loop.core import alloc as AL
    return AL._terms(str(c.get("family") or ""), str(c.get("mechanism_tag") or c.get("title") or ""),
                     f"{c.get('title', '')} {c.get('mechanism', '')}")


def originality(ctx: Any, store: Any, c: Mapping[str, Any], docs: Sequence[Counter] | None = None
                ) -> dict[str, Any]:
    """{originality, cluster, cluster_cos, flag, nearest_node_cos}: 1 - the larger of the candidate's cosine to its
    nearest mechanism CLUSTER centroid (loop.core.alloc's leader clusters, as the tick last computed them) and to its
    nearest research node's mechanism text (family + tag + signature: ``docs``). ``flag`` = the cluster's flag when the
    candidate falls in it (cosine >= ``alloc.cluster_tau``)."""
    from loop.core import alloc as AL
    from loop.novelty import _tfidf, mech_cos
    me = _terms(c)
    out: dict[str, Any] = {"cluster": "", "cluster_cos": 0.0, "flag": "", "nearest_node_cos": 0.0}
    cls = AL.LAST.get("clusters") or []
    if cls:
        v = _tfidf(me, AL.LAST.get("df") or Counter(me), max(1, int(AL.LAST.get("n") or 0)))
        best = max(cls, key=lambda cl: AL._cos(v, cl.centroid))
        cos = AL._cos(v, best.centroid)
        tau = float(getattr(AL.cfg_of(ctx), "cluster_tau", 0.3))
        out.update(cluster=best.id, cluster_cos=round(cos, 4), flag=best.flag if cos >= tau else "",
                   cluster_label=best.label)
    if docs:
        out["nearest_node_cos"] = round(max(mech_cos(me, list(docs)), default=0.0), 4)
    out["originality"] = round(max(0.0, 1.0 - max(out["cluster_cos"], out["nearest_node_cos"])), 4)
    return out


def ledger_docs(store: Any, k: int = 200) -> list[Counter]:
    """The mechanism texts of the ``k`` newest research nodes (loop.core.mechanisms.window)."""
    from loop.core import alloc as AL
    from loop.core import mechanisms as MS
    try:
        return [AL.node_text(store, n) for n in MS.window(store, k)]
    except Exception:  # noqa: BLE001 - no ledger texts: the clusters alone
        return []


JUDGE_SYSTEM = ("You are a skeptical senior researcher in self-supervised learning for computational pathology. For "
                "each proposed change to a ViT foundation-model training recipe (fixed sample and FLOP budget), give the "
                "probability (0-1) that it improves a linear-probe benchmark panel over the current recipe. Judge "
                "mechanism plausibility and implementation risk; be calibrated, most ideas fail.")
JUDGE_SCHEMA = {"type": "object", "required": ["scores"], "properties": {
    "scores": {"type": "array", "items": {"type": "number"}}, "why": {"type": "string"}}}


def judge_slate_prompt(cands: Sequence[Mapping[str, Any]]) -> str:
    rows = "\n".join(f"{i}. [{c.get('category')}] {c.get('title')}: {str(c.get('mechanism') or '')[:700]}"
                     for i, c in enumerate(cands))
    return f"Candidates:\n{rows}\n\nReturn `scores`: {len(cands)} probabilities in order."


def llm_slate_judge(llm: Any, role: str = "slate_judge") -> Callable[[Sequence[Mapping[str, Any]]], list[float] | None]:
    """The cheap plausibility judge over ``llm.complete_json`` (``role`` is qwen-routed by default)."""
    def judge(cands: Sequence[Mapping[str, Any]]) -> list[float] | None:
        got = llm.complete_json(role, system=P.rule("judge_system", JUDGE_SYSTEM), prompt=judge_slate_prompt(cands),
                                schema=JUDGE_SCHEMA)
        xs = [min(1.0, max(0.0, float(x))) for x in (got.get("scores") or [])]
        return xs if len(xs) == len(cands) else None
    return judge


def rank(ctx: Any, store: Any, cands: Sequence[Mapping[str, Any]],
         judge: Callable[[Sequence[Mapping[str, Any]]], list[float] | None] | None = None,
         docs: Sequence[Counter] | None = None, typical_w: float = 0.0,
         blocked: Mapping[int, str] | None = None) -> dict[str, Any]:
    """Score every candidate: originality (:func:`originality`) x plausibility (the session's own, averaged with the
    judge's when it answers); a candidate in a saturated-negative cluster is rejected. {rows, chosen (index or None),
    judge ('qwen' | 'self')}: rows in the candidates' order."""
    docs = ledger_docs(store) if docs is None else docs
    scores = None
    if judge is not None:
        try:
            scores = judge(cands)
        except Exception:  # noqa: BLE001 - backend down / schema: the session's own plausibility
            scores = None
    rows = []
    for i, c in enumerate(cands):
        o = originality(ctx, store, c, docs)
        own = min(1.0, max(0.0, float(c.get("plausibility") or 0.0)))
        plaus = (own + scores[i]) / 2.0 if scores else own
        rej = "saturated-negative cluster " + o["cluster"] if o["flag"] == SATURATED else ""
        if blocked and i in blocked:                   # 11-08: a repeat of tested / in-flight work (loop.core.tested)
            rej = blocked[i]
        # v3-c9 C3: a direction a typical researcher would propose (high verbalized probability) is discounted
        typ = (1.0 - typical_w * min(1.0, max(0.0, float(c["probability"])))) \
            if typical_w > 0 and _isnum(c.get("probability")) else 1.0
        rows.append({"i": i, "title": str(c.get("title") or "")[:120], "category": str(c.get("category") or ""),
                     "originality": o["originality"], "plausibility": round(plaus, 4), "self": round(own, 4),
                     **({"judge": round(scores[i], 4)} if scores else {}),
                     **({"probability": round(float(c["probability"]), 4)} if _isnum(c.get("probability")) else {}),
                     "cluster": o["cluster"], "cluster_cos": o["cluster_cos"], "node_cos": o["nearest_node_cos"],
                     "score": 0.0 if rej else round(o["originality"] * plaus * typ, 5), "rejected": rej})
    ok = [r for r in rows if not r["rejected"]]
    chosen = max(ok, key=lambda r: (r["score"], r["plausibility"], -r["i"]))["i"] if ok else None
    return {"rows": rows, "chosen": chosen, "judge": "qwen" if scores else "self"}


def _isnum(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def note(r: Mapping[str, Any]) -> str:
    if r.get("rejected"):
        return f"REJECTED: {r['rejected']}"
    return f"nearest cluster {r['cluster']} (cos {r['cluster_cos']:.2f})" if r.get("cluster") else ""


def ranking_text(res: Mapping[str, Any], cands: Sequence[Mapping[str, Any]]) -> str:
    rows = sorted(res["rows"], key=lambda r: (-r["score"], r["i"]))
    lines = ["Slate scored (originality x plausibility; plausibility = "
             + ("your own averaged with the judge's" if res["judge"] == "qwen" else "your own: the judge did not answer")
             + "):", "rank | # | category | title | originality | plausibility | score | note", "---|---|---|---|---|---|---|---"]
    for k, r in enumerate(rows, start=1):
        lines.append(f"{k} | {r['i']} | {r['category']} | {r['title'][:60]} | {r['originality']:.2f} | "
                     f"{r['plausibility']:.2f} | {r['score']:.3f} | {note(r)}")
    ch = res["chosen"]
    if ch is None:
        lines.append("Every candidate was rejected (a saturated-negative cluster, or a repeat of tested / in-flight "
                     "work: see the notes): propose a new slate of mechanisms that are NOT on the tested list "
                     "(call propose_slate again).")
    else:
        c = cands[ch]
        lines.append(f"BUILD candidate #{ch}: [{c.get('category')}] {c.get('title')}. Implement it now (edits are "
                     "unlocked). Declare its family / mechanism_tag in finish; the slate is logged with your node.")
    return "\n".join(lines)


def built_category(diff_text: str, ctx: Any = None) -> str:
    """The taxonomy category the BUILT diff falls in (:data:`CATEGORY_RX` over its changed paths and added code lines;
    'knob' for a knob-only diff, 'other' when none matches) -- compared with the slate candidate it was meant to build."""
    if knob_only(diff_text):
        return KNOB
    text = " ".join([f["path"] for f in parse_diff(diff_text)]
                    + [l for f in parse_diff(diff_text) for l in f["added"] if not l.lstrip().startswith("#")]).replace("_", " ")
    rx = dict(opt(ctx, "bold_category_rx") or {}) if ctx is not None else {}
    for c, p in (rx or category_rx()).items():
        if re.search(p, text, re.I):
            return c
    return OTHER


class Slate:
    """The session-side ``propose_slate`` protocol: bounded re-proposals (at most :data:`MAX_TRIES`), the ranking, the event log."""

    MAX_TRIES = 3      # v3 review S3: a session that cannot produce an acceptable slate in 3 tries gets no edits

    def __init__(self, ctx: Any, store: Any, request_id: str, kind: str, *, judge: Any = None,
                 logger: Callable[..., Any] | None = None, history: Callable[[], Any] | None = None) -> None:
        self.ctx, self.store, self.request_id, self.kind = ctx, store, request_id, kind
        self.judge, self.logger = judge, logger
        self.cands: list[dict[str, Any]] = []
        self.result: dict[str, Any] | None = None
        self.tries = 0
        #: v3-c9 C3: a verbalized-sampling portfolio (explore / literature / repair); ``history``: () -> the history
        #: table (loop.agents.history.Table) or None -- grounding and citations are checked against its ids
        self.portfolio = portfolio_on(ctx, kind)
        self.history = history

    def _table(self) -> Any:
        try:
            return self.history() if self.history is not None else None
        except Exception:  # noqa: BLE001 - no table: the checks that need it are skipped (logged on the plan)
            return None

    def _pf(self) -> dict[str, Any] | None:
        if not self.portfolio:
            return None
        tab = self._table()
        return {"sum_tol": float(opt(self.ctx, "portfolio_sum_tol")), "tail_p": float(opt(self.ctx, "portfolio_tail_p")),
                "tail_min": int(opt(self.ctx, "portfolio_tail_min")),
                "min_clusters": slate_limits(self.ctx, self.kind)["min_clusters"],
                # 11-08: the history's own cluster count caps the diversity rule (a fresh ledger has none)
                **({"available_clusters": len(table_clusters(tab)),
                    "n_ids": len(getattr(tab, "by_id", {}) or {}) + len(getattr(tab, "cluster_ids", {}) or {})}
                   if tab is not None else {}),
                "exists": tab.exists if tab is not None else None,
                "cluster_of": (lambda i: tab.cluster_ids[i].id if i in tab.cluster_ids else tab.node_cluster(i))
                if tab is not None else None}

    def note_finish(self, hypothesis: Mapping[str, Any]) -> str:
        """The finish of a portfolio session: ``hypothesis.differs_from`` was checked at propose_slate (per candidate);
        here the old format check only LOGS (``portfolio_cites``) and never rejects a finished build."""
        why = self.cite_problem(hypothesis)
        if why and self.logger is not None:
            try:
                self.logger("portfolio_cites", request_id=self.request_id, kind=self.kind, why=why[:300])
            except Exception:  # noqa: BLE001
                pass
        return why

    def cite_problem(self, hypothesis: Mapping[str, Any]) -> str:
        """The finish-time check of a portfolio session (C3): the chosen direction cites >= ``portfolio_cite_min``
        history node / cluster ids it differs from (``hypothesis.differs_from``), they exist, and it says how (>=
        ``differs_min_chars`` content characters). '' for a session that is not a portfolio / has no table."""
        if not self.portfolio or not self.done:
            return ""
        tab = self._table()
        return cite_problem(str(hypothesis.get("differs_from") or ""), tab.exists if tab is not None else None,
                            min_ids=int(opt(self.ctx, "portfolio_cite_min")),
                            min_chars=int(opt(self.ctx, "differs_min_chars")))

    @property
    def done(self) -> bool:
        return self.result is not None and self.result.get("chosen") is not None

    @property
    def chosen(self) -> dict[str, Any] | None:
        return dict(self.cands[self.result["chosen"]]) if self.done else None

    def __call__(self, candidates: Any = None) -> str:
        if self.done:
            return f"ERROR: the slate is settled: build candidate #{self.result['chosen']} ({self.chosen.get('title')})"
        if self.tries >= self.MAX_TRIES:
            return (f"ERROR: the slate budget is spent ({self.MAX_TRIES} tries) without an accepted slate: edits stay "
                    "locked; finish without building anything")
        cands = [dict(c) for c in (candidates or []) if isinstance(c, Mapping)]
        for c in cands:                                  # a displayed label (taxonomy_label_<id>) -> the internal id
            c["category"] = canon(c.get("category"))
        self.tries += 1
        lim = slate_limits(self.ctx, self.kind)
        why = slate_problems(cands, k_min=lim["k_min"], k_max=lim["k_max"], min_cats=lim["min_cats"],
                             per_cat=lim["per_cat"], portfolio=self._pf())
        if why:
            return "REJECT slate: " + "; ".join(why) + ". Fix it and call propose_slate again (nothing is edited yet)."
        from loop.core import tested as TS             # 11-08: the mechanism memory (tested / in-flight repeats)
        try:
            hits = TS.screen(self.ctx, self.store, cands, self.request_id)
        except Exception as exc:  # noqa: BLE001 - a broken memory never blocks a slate (logged)
            hits = [None] * len(cands)
            if self.logger is not None:
                try:
                    self.logger("tested_error", request_id=self.request_id, error=repr(exc)[:300])
                except Exception:  # noqa: BLE001
                    pass
        blocked = {i: TS.bounce_text(i, c, h) for i, (c, h) in enumerate(zip(cands, hits)) if h is not None}
        res = rank(self.ctx, self.store, cands, self.judge,
                   typical_w=float(opt(self.ctx, "portfolio_typical_w")) if self.portfolio else 0.0,
                   blocked=blocked)
        if blocked:
            try:
                TS.record_slate(self.ctx, self.store, self.request_id, cands, hits, None)
            except Exception:  # noqa: BLE001
                pass
            if self.logger is not None:
                try:
                    self.logger("slate_repeats", request_id=self.request_id, kind=self.kind, tries=self.tries,
                                n=len(cands), repeats=[{"i": i, "title": str(cands[i].get("title") or "")[:100],
                                                        "prior": hits[i].entry.id, "kind": hits[i].kind,
                                                        "sim": hits[i].sim} for i in blocked])
                except Exception:  # noqa: BLE001
                    pass
        if res.get("chosen") is None and blocked:
            return ("REJECT slate: " + "; ".join(blocked[i] for i in sorted(blocked)) + ". Every other candidate "
                    "was rejected too (see the ranking below). Propose mechanisms that are NOT on the tested / "
                    "in-flight lists and call propose_slate again (nothing is edited yet; this cost one slate try).\n"
                    + ranking_text(res, cands))
        if self.portfolio and (tab := self._table()) is not None and (res.get("chosen") is not None):
            ch = cands[int(res["chosen"])]            # review fix M3: the chosen candidate cites what it differs from NOW
            if why := differs_problem(ch, tab, min_chars=int(opt(self.ctx, "differs_min_chars")),
                                      sim_min=float(opt(self.ctx, "differs_sim_min")),
                                      min_ids=int(opt(self.ctx, "portfolio_cite_min")),
                                      thin=thin_history(tab, slate_limits(self.ctx, self.kind)["min_clusters"])):
                return (f"REJECT slate: the loop picked candidate #{res['chosen']} ({ch.get('title')}) to build, but its "
                        f"differs_from is not sound: {why}. Give EVERY candidate a differs_from [{{id, how}}] and call "
                        "propose_slate again (nothing is edited yet; this costs one of your slate tries).")
        self.cands, self.result = cands, res
        try:                                            # the proposals + this session's in-flight mechanism
            TS.record_slate(self.ctx, self.store, self.request_id, cands, hits, int(res["chosen"]))
        except Exception:  # noqa: BLE001
            pass
        if self.logger is not None:
            try:
                self.logger(SLATE_EVENT, request_id=self.request_id, kind=self.kind, judge=res["judge"],
                            chosen=res["chosen"], tries=self.tries,
                            candidates=[{**r, "mechanism": str(cands[r["i"]].get("mechanism") or "")[:600]}
                                        for r in res["rows"]])
            except Exception:  # noqa: BLE001 - a log fault never blocks the session
                pass
        return ranking_text(res, cands)

    def plan(self, diff_text: str = "", parent_src: Callable[[str], str | None] | None = None) -> dict[str, Any]:
        """What rides on the node's plan (``ideation:<node>``): the scored slate, the chosen candidate, its category
        and its predicted distribution (the calibration score reads ``prediction``). With the BUILT ``diff_text``
        (v3 review S3): its category is checked against the chosen candidate's (a mismatch is logged ``slate_mismatch``
        and kept on the plan) and its originality is scored on the diff too (:func:`novelty`: the share of new
        call / attribute names), averaged with the text originality of the candidate."""
        if not self.done:
            return {}
        ch = self.chosen
        out = {"slate": {"judge": self.result["judge"], "chosen": self.result["chosen"], "rows": self.result["rows"]},
               "bold_category": str(ch.get("category") or ""),
               "candidates": [{"mechanism": f"{c.get('title', '')}: {c.get('mechanism', '')}"[:400],
                               "family": c.get("family", ""), "mechanism_tag": c.get("mechanism_tag", ""),
                               **{k: c[k] for k in ("p_neg", "median", "p_gap", "probability", "grounded_in") if k in c}}
                              for c in self.cands],
               "chosen": int(self.result["chosen"]),
               "mechanism": f"{ch.get('title', '')}: {ch.get('mechanism', '')}"[:300]}
        if all(k in ch for k in ("p_neg", "median", "p_gap")):
            out["prediction"] = {k: ch[k] for k in ("p_neg", "median", "p_gap")}
        if self.portfolio:
            tab = self._table()
            grounded = [str(c.get("grounded_in") or "") for c in self.cands]
            out["portfolio"] = {
                "probability": ch.get("probability"), "grounded_in": ch.get("grounded_in"),
                "clusters": sorted({(tab.cluster_ids[g].id if g in tab.cluster_ids else tab.node_cluster(g))
                                    for g in grounded if g and tab is not None} - {""}),
                "table": tab is not None}
        if diff_text.strip():
            built, want = built_category(diff_text, self.ctx), str(ch.get("category") or "")
            row = self.result["rows"][self.result["chosen"]]
            nov = novelty(diff_text, parent_src)
            out["built"] = {"category": built, "novelty": round(nov, 4), "matches": built == want,
                            "originality": round((float(row.get("originality") or 0.0) + nov) / 2.0, 4)}
            if built != want and self.logger is not None:
                try:
                    self.logger("slate_mismatch", request_id=self.request_id, chosen=want, built=built,
                                title=str(ch.get("title") or "")[:120])
                except Exception:  # noqa: BLE001
                    pass
        return out


# ---------------------------------------------------------------------------------------------- the meta report
def category_of(ctx: Any, store: Any, node: Any, diff: str | None = None) -> str:
    """A node's structural category: a slate session's chosen category (notes ``bold_cat=``), else 'knob' when its diff
    (``diff``, else read from ``diff_path``) is knob-only (:func:`knob_only`: a koleo-weight tweak is not a new
    pretext task, whatever its tag says), else the first :data:`CATEGORY_RX` match over family / tag / mechanism type /
    signature, else 'other'."""
    if m := re.search(r"\bbold_cat=(\S+)", getattr(node, "notes", "") or ""):
        return m.group(1)
    if knob_only(_diff(node) if diff is None else diff):
        return KNOB
    from loop.core import mechanisms as MS
    rx = dict(opt(ctx, "bold_category_rx") or {}) or category_rx()
    try:
        sig = MS.node_signature(store, node.id)
    except Exception:  # noqa: BLE001
        sig = ""
    mt = m.group(1) if (m := re.search(r"\bmtype=(\w+)", getattr(node, "notes", "") or "")) else ""
    text = f"{node.family} {mt} {node.mechanism_tag} {sig}".replace("_", " ") + f" {node.family}"
    for c, p in rx.items():
        if re.search(p, text, re.I):
            return c
    return OTHER


def _diff(node: Any) -> str:
    try:
        return Path(node.diff_path).read_text(errors="replace") if getattr(node, "diff_path", "") else ""
    except OSError:
        return ""


def mix(ctx: Any, store: Any, since: float = 0.0, epoch: str = "") -> dict[str, Any]:
    """The ORIGINALITY MIX of the research nodes created since ``since`` (brief-tagged, not rejected, no port / wave
    point): per kind {n, cats: {category: share}, knob_only: share of knob-only diffs, knob_n}, plus 'all'; and the
    slate events' candidate category shares (what the sessions considered)."""
    from loop.core.alloc import brief_kind
    from loop.core import mechanisms as MS
    by: dict[str, list[tuple[str, bool]]] = {}
    for n in store.nodes():
        if (n.created_ts or 0.0) < since or MS._skip(n) or not (k := brief_kind(n.notes)):
            continue
        if epoch and getattr(n, "base", "") not in ("", epoch):
            continue
        d = _diff(n)
        ko = knob_only(d) if d else False          # no readable diff: not judged knob-only
        by.setdefault(k, []).append((category_of(ctx, store, n, d), ko))
    out: dict[str, Any] = {"kinds": {}}
    allrows = [x for xs in by.values() for x in xs]
    for k, xs in sorted(by.items()) + ([("all", allrows)] if allrows else []):
        c = Counter(cat for cat, _ in xs)
        out["kinds"][k] = {"n": len(xs), "cats": {cat: round(m / len(xs), 3) for cat, m in c.most_common()},
                           "knob_only": round(sum(ko for _, ko in xs) / len(xs), 3), "knob_n": sum(ko for _, ko in xs)}
    sl = [e for e in store.events(SLATE_EVENT) if float(e.get("ts") or 0.0) >= since] \
        if hasattr(store, "events") else []
    cc = Counter(str(r.get("category")) for e in sl for r in e.get("candidates") or ())
    tot = sum(cc.values())
    out["slates"] = {"n": len(sl), "candidates": tot, "cats": {k: round(v / tot, 3) for k, v in cc.most_common()}
                     if tot else {}, "rejected": sum(bool(r.get("rejected")) for e in sl for r in e.get("candidates") or ()),
                     "judge_qwen": sum(e.get("judge") == "qwen" for e in sl)}
    return out


def mix_text(m: Mapping[str, Any]) -> list[str]:
    """The meta report's "Originality mix" lines."""
    kinds = m.get("kinds") or {}
    if not kinds and not (m.get("slates") or {}).get("n"):
        return ["(no new research nodes in the window)"]
    cats = [*TAXONOMY, KNOB, OTHER]
    lines = ["kind | n | knob-only | " + " | ".join(label(c) for c in cats), "---|---|---|" + "---|" * len(cats)]
    for k, r in kinds.items():
        lines.append(f"{k} | {r['n']} | {r['knob_only']:.0%} | "
                     + " | ".join(f"{r['cats'].get(c, 0.0):.0%}" for c in cats))
    s = m.get("slates") or {}
    if s.get("n"):
        lines.append(f"slates: {s['n']} sessions, {s['candidates']} candidates ({s['rejected']} rejected in "
                     f"saturated-negative clusters; qwen judged {s['judge_qwen']}), candidate categories "
                     + ", ".join(f"{label(k)} {v:.0%}" for k, v in s["cats"].items()))
    return lines


# ---------------------------------------------------------------------------------------------- the brief
def taxonomy_text() -> str:
    return "\n".join(f"- **{label(k)}**: {v}" for k, v in taxonomy().items())


def coverage_line(ctx: Any, store: Any, led: Any) -> str:
    """This epoch's research nodes per structural category (computed)."""
    epoch = getattr(led, "epoch", "") or ""
    c: Counter = Counter({k: 0 for k in [*TAXONOMY, KNOB, OTHER]})
    for nid, ln in getattr(led, "nodes", {}).items():
        if (not epoch or ln.base == epoch) and "brief=" in (ln.notes or "") and (n := store.get_node(nid)) is not None:
            c[category_of(ctx, store, n)] += 1
    tot = sum(c.values()) or 1
    return ", ".join(f"{label(k)} {n} ({n / tot:.0%})" for k, n in c.items()) + (
        f". {KNOB} = a diff that only changes config values / constants of existing code" if c[KNOB] else "")


def brief_head(ctx: Any) -> str:
    from loop.core.ideation import _render
    return _render("brief_bold", taxonomy=taxonomy_text(), suggestion="", constraints="(see the hard constraints in "
                   "the Surface section below)", init_rule=P.rule('bold_absolute', INIT_RULE), slate_rule=slate_rule(ctx))


INIT_RULE = ("STEP-0 IDENTITY IS RELAXED for this lane: the run warm-starts from the SAME parent weights (every parent "
             "tensor must still load by name and shape), and NEW modules may be randomly initialised -- a new head, "
             "token type or branch does not have to be zero-initialised and the initial loss may differ from the "
             "parent's. Still absolute: no pathology-trained checkpoint and no GenBio data / model / derivative may "
             "initialise or guide training (an init swap is checked as one); the training budget caps "
             "(train.max_train_samples, train.max_train_flops), eval identity and the locked files.")


def portfolio_rule(ctx: Any, kind: str) -> str:
    lim = slate_limits(ctx, kind)
    return (f"FIRST call `propose_slate` with {lim['k_min']}-{lim['k_max']} DIRECTIONS as ONE JOINT PORTFOLIO, \"not "
            "variations of the most typical answer\": imagine the whole distribution of what a researcher could do "
            "here and sample from it, tails included. Give each direction a verbalized `probability` (the chance a "
            "typical researcher given this brief would propose it; the portfolio sums to about 1, at least "
            f"{int(opt(ctx, 'portfolio_tail_min'))} directions at or below {float(opt(ctx, 'portfolio_tail_p')):g}) and a "
            "`plausibility` (your honest chance that it beats its base). GROUND the portfolio in the history: "
            "`grounded_in` = a cluster id (C7) or node id from the `history` tool for each direction, covering >= "
            f"{lim['min_clusters']} DISTINCT clusters/branches of it (query `history` first: action clusters, then "
            f"tables of the clusters and axes you care about). Spread over >= {lim['min_cats']} categories (at most "
            f"{lim['per_cat']} per category). Edits are refused until it is accepted; the loop scores originality x "
            "plausibility and names the one to build. EVERY direction carries `differs_from`: [{id, how}], >= "
            f"{int(opt(ctx, 'portfolio_cite_min'))} history node / cluster id(s) it differs from; each id must be RELATED to "
            "the direction (same cluster as its grounded_in, same family or mechanism tag, or similar text) and each "
            f"`how` (>= {int(opt(ctx, 'differs_min_chars'))} characters) must use words the cited node's description "
            "does not. The loop checks the CHOSEN direction's citations when you propose the slate (a failure costs one "
            "slate try, never the build). You may call `history` at most "
            f"{int(opt(ctx, 'history_calls_max'))} times in the session: spend them on the clusters you ground in.")


def slate_rule(ctx: Any, kind: str = KIND) -> str:
    if portfolio_on(ctx, kind):
        return portfolio_rule(ctx, kind)
    return (f"FIRST call `propose_slate` with {int(opt(ctx, 'slate_min'))}-{int(opt(ctx, 'slate_max'))} one-paragraph "
            f"candidates over >= {int(opt(ctx, 'slate_min_categories'))} categories (at most "
            f"{int(opt(ctx, 'slate_per_category'))} per category). Edits are refused until it is accepted; the loop "
            "scores originality x plausibility and names the one to build.")


def slate_section(ctx: Any, kind: str = "explore") -> str:
    """The diverge-then-judge section an explore / literature / repair brief carries when its session must propose a
    slate (v3-c9: a portfolio)."""
    return ("## Diverge, then judge (required before any edit)\n" + slate_rule(ctx, kind) + " Spread the candidates over the "
            "structural taxonomy below, then build the one the loop names (you may still build a careful incremental "
            "idea if it ranks first).\n" + taxonomy_text())
