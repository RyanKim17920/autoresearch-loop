"""loop.core.mechanisms — what a research node's MECHANISM is, apart from where it sits (v3-c6i, 10-13).

The ideation analysis of 10-13 found 43-69% of crossover sessions rejected ``near_dup``: intake embedded the claim +
every changed line of the diff, and a crossover's diff is taken against the parents' common ancestor, so it carries the
incumbent's whole change (and its claim restates the incumbent) -- every crossover onto one incumbent looked like every
earlier one (a node's own crossover first). DAPS (2609.00077) / ShinkaEvolve dedup on the MECHANISM instead:

- :func:`signature`      family + mechanism tag + ONE line of mechanism text, with every ancestor's own claim text and
                         node ids taken out of it (the parent / incumbent description is not the mechanism)
- :func:`dup_rule`       cosine of the signature to the last ``ideation.mech_dup_window`` signatures (the candidate's
                         ancestors left out) > ``ideation.mech_dup_cos`` -> ``near_dup_signature``, unless the claim's
                         ``[differs from: ...]`` names AND explains each of them; v3-c6r: the top-3 lexical neighbours
                         above ``ideation.mech_judge_cos`` are also judged by a qwen yes/no "same mechanism?" call
                         (paraphrases the hashing embedder misses)
- :func:`category` / :func:`coverage` / :func:`suggest`
                         a computed coarse taxonomy (``ideation.taxonomy``: category -> regex over "family mechanism
                         type tag"), its usage this epoch, and the category a new brief is pointed at (the least used,
                         in-flight briefs counted)

Nothing here names an idea: categories are config, counts are the ledger's."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

from loop.select.embed import HashingEmbedder, cosine

K_SIG = "sig"                      # the signature text in a node's ``ideation:<id>`` plan (written at intake)
NODE_ID = re.compile(r"\bn\d{3,}(?:@\w+)?\b")
_SENT = re.compile(r"(?<=[.;!?])\s+|\n+")
OTHER = "other"
DIFFERS = "[differs from:"


def opt(ctx: Any, name: str) -> Any:
    from loop.core.levers import opt as o
    return o(ctx, name)


# ---------------------------------------------------------------------------------------------- signatures
def first_line(text: str, width: int = 220) -> str:
    """The first sentence / line of ``text`` (at most ``width`` characters)."""
    for part in _SENT.split(str(text or "")):
        if part.strip():
            return part.strip()[:width]
    return ""


def strip_ancestors(text: str, ancestors: Iterable[str]) -> str:
    """``text`` without node ids and without any ancestor text it repeats verbatim (each ancestor claim whole, then
    each of its sentences of >= 20 characters)."""
    out = str(text or "")
    for a in sorted({str(x) for x in ancestors if x and str(x).strip()}, key=len, reverse=True):
        out = out.replace(a.strip(), " ")
        for s in _SENT.split(a):
            if len(s := s.strip()) >= 20:
                out = out.replace(s, " ")
    return re.sub(r"\s+", " ", NODE_ID.sub(" ", out)).strip()


def signature(family: str, tag: str, mechanism: str, ancestors: Iterable[str] = ()) -> str:
    """family + tag (words) + the first line of ``mechanism`` with the ancestors' own text removed."""
    words = re.sub(r"[_\-]+", " ", str(tag or "")).strip()
    return " ".join(x for x in (str(family or ""), words, first_line(strip_ancestors(mechanism, ancestors))) if x)


def ancestors(store: Any, parent_ids: Sequence[str], depth: int = 64) -> list[str]:
    """Every ancestor of ``parent_ids`` (parents included; first and second parents, ports' originals), nearest first."""
    out, todo = [], [p for p in parent_ids if p]
    while todo and len(out) < depth:
        cur = todo.pop(0)
        if cur in out:
            continue
        out.append(cur)
        n = store.get_node(cur) if hasattr(store, "get_node") else None
        if n is None:
            continue
        todo += [p for p in (n.parent_id, n.second_parent_id) if p]
        if m := re.search(r"\bport_of=(\S+)", n.notes or ""):
            todo.append(m.group(1))
    return out


def claims_of(store: Any, ids: Iterable[str]) -> list[str]:
    out = []
    for nid in ids:
        n = store.get_node(nid) if hasattr(store, "get_node") else None
        h = store.get_hypothesis(n.hypothesis_id or "") if n is not None and n.hypothesis_id else None
        if h is not None and h.claim:
            out.append(h.claim)
    return out


_SIG_CACHE: dict[tuple[int, str], str] = {}


def node_signature(store: Any, nid: str) -> str:
    """The node's stored signature (its plan's ``sig``), else derived from its hypothesis (cached per store)."""
    key = (id(store), nid)
    if key in _SIG_CACHE:
        return _SIG_CACHE[key]
    plan = store.get_state(f"ideation:{nid}") or {} if hasattr(store, "get_state") else {}
    sig = str(plan.get(K_SIG) or "") if isinstance(plan, Mapping) else ""
    n = store.get_node(nid)
    if not sig and n is not None:
        h = store.get_hypothesis(n.hypothesis_id or "") if n.hypothesis_id else None
        anc = ancestors(store, [p for p in (n.parent_id, n.second_parent_id) if p])
        sig = signature(n.family, n.mechanism_tag, h.claim if h else "", claims_of(store, anc))
    _SIG_CACHE[key] = sig
    return sig


def _skip(n: Any) -> bool:
    from loop.contracts import ROOT_NODE_ID, NodeStatus, design_point
    from loop.core.factorial import is_point
    return n.id == ROOT_NODE_ID or n.status is NodeStatus.REJECTED or design_point(n) or is_point(n.notes or "") \
        or "port_of=" in (n.notes or "")


def window(store: Any, k: int, exclude: Iterable[str] = ()) -> list[Any]:
    """The ``k`` newest mechanism nodes (no root, rejected, design / wave point or port), ``exclude`` left out."""
    ex = set(exclude)
    nodes = [n for n in store.nodes() if n.id not in ex and not _skip(n)]
    return sorted(nodes, key=lambda n: -(n.created_ts or 0.0))[:max(0, int(k))]


def dup_hits(ctx: Any, store: Any, sig: str, parent_ids: Sequence[str], embedder: Any = None,
             exclude: Iterable[str] = (), thr: float | None = None) -> list[tuple[Any, float]]:
    """(node, cosine) of the window's nodes whose signature is above ``thr`` (default ``ideation.mech_dup_cos``),
    nearest first."""
    thr = float(opt(ctx, "mech_dup_cos")) if thr is None else float(thr)
    if thr <= 0 or not sig.strip():
        return []
    emb = embedder or HashingEmbedder()
    me = emb.embed(sig)
    anc = ancestors(store, parent_ids)
    hits = []
    for n in window(store, int(opt(ctx, "mech_dup_window")), [*anc, *exclude]):
        if (theirs := node_signature(store, n.id)) and (c := cosine(me, emb.embed(theirs))) > thr:
            hits.append((n, c))
    return sorted(hits, key=lambda h: -h[1])


EXPLAIN_MIN = 12          # a [differs from: ...] entry needs this many characters of explanation after the id


def explained(differs: str, nid: str, min_chars: int | None = None) -> bool:
    """``differs`` (the text after ``[differs from:``) names ``nid`` AND says what differs: at least EXPLAIN_MIN
    characters (``min_chars``, default) of words between it and the next node id / the closing bracket (v3-c6r: a bare
    id list passed)."""
    text = differs.split("]", 1)[0] if "]" in differs else differs
    for m in re.finditer(rf"(?<![\w@]){re.escape(nid)}(?![\w@])", text):
        rest = text[m.end():]
        nxt = NODE_ID.search(rest)
        expl = re.sub(r"[^A-Za-z0-9 ]+", " ", rest[:nxt.start()] if nxt else rest)
        if len(re.sub(r"\s+", " ", expl).strip()) >= (EXPLAIN_MIN if min_chars is None else min_chars):
            return True
    return False


def dup_rule(ctx: Any, store: Any, sig: str, parent_ids: Sequence[str], claim: str = "", embedder: Any = None,
             exclude: Iterable[str] = (), judge: Any = None) -> tuple[str, str]:
    """('near_dup_signature', fix) when the signature repeats a recent mechanism (outside the ancestry) and the claim's
    ``[differs from: ...]`` does not name AND explain each such node; else ('', '').

    v3-c6r (review D1: the hashing embedder is lexical -- paraphrases pass, dose variants block): with ``judge``
    (``(sig, [(id, their signature)]) -> [bool] | None``: a qwen yes/no "same mechanism?" call) the top-3 LEXICAL
    neighbours between ``ideation.mech_judge_cos`` and the dup threshold are judged too, and a "same" verdict counts
    as a hit (no verdict -- backend down -- is no hit). A dose variant of a recent mechanism counts as the same
    mechanism: it belongs on that node's dose ladder (the program / ladder path), not in a new node."""
    hits = dup_hits(ctx, store, sig, parent_ids, embedder, exclude)[:3]
    jc = float(opt(ctx, "mech_judge_cos") or 0.0)
    if judge is not None and jc > 0:
        seen = {n.id for n, _ in hits}
        near = [(n, c) for n, c in dup_hits(ctx, store, sig, parent_ids, embedder, exclude, thr=jc)
                if n.id not in seen][:3]
        if near:
            try:
                same = judge(sig, [(n.id, node_signature(store, n.id)) for n, _ in near])
            except Exception:  # noqa: BLE001 - no verdict: lexical only
                same = None
            if same and len(same) == len(near):
                hits += [(n, c) for (n, c), y in zip(near, same) if y]
    differs = claim.split(DIFFERS, 1)[1] if DIFFERS in claim else ""
    if not hits or all(explained(differs, n.id) for n, _ in hits):
        return "", ""
    near_t = "; ".join(f"{n.id} ({n.family}/{n.mechanism_tag}, cos {c:.2f})" for n, c in hits)
    return "near_dup_signature", (f"the mechanism repeats a recent one: {near_t}. Bring a different mechanism (another "
                                  "DOSE of the same mechanism is not a new one: it belongs on that node's dose ladder), "
                                  "or set hypothesis.differs_from naming EACH of these nodes followed by what is "
                                  "materially different about it (\"n123: ...\"; your one-line mechanism, not the "
                                  "parent's description, is compared)")


JUDGE_SYSTEM = ("You judge whether ML experiment proposals share a MECHANISM. Answer from the text only. Two proposals "
                "are the SAME mechanism when they change the same thing in the same direction for the same reason, "
                "even if worded differently; a different DOSE (amount, rate, count) of the same change is the SAME "
                "mechanism. Different components, objectives or data are DIFFERENT mechanisms.")
JUDGE_SCHEMA = {"type": "object", "required": ["same"], "properties": {
    "same": {"type": "array", "items": {"type": "boolean"}}, "why": {"type": "string"}}}


def judge_prompt(sig: str, near: Sequence[tuple[str, str]]) -> str:
    rows = "\n".join(f"{i}. {s[:300]}" for i, (_, s) in enumerate(near))
    return (f"NEW proposal's mechanism: {sig[:400]}\n\nRecent proposals:\n{rows}\n\nReturn `same`: {len(near)} "
            "booleans in order, true when that proposal is the same mechanism as the NEW one.")


def llm_judge(llm: Any, role: str = "mech_judge") -> Any:
    """A :func:`dup_rule` judge over ``llm.complete_json`` (``role`` is qwen-routed by default; 30 s)."""
    def judge(sig: str, near: Sequence[tuple[str, str]]) -> list[bool] | None:
        got = llm.complete_json(role, system=JUDGE_SYSTEM, prompt=judge_prompt(sig, near), schema=JUDGE_SCHEMA)
        same = [bool(x) for x in (got.get("same") or [])]
        return same if len(same) == len(near) else None
    return judge


# ---------------------------------------------------------------------------------------------- taxonomy
def categories(ctx: Any) -> list[str]:
    return [*dict(opt(ctx, "taxonomy") or {}), OTHER]


_RX: dict[tuple, list[tuple[str, re.Pattern]]] = {}


def category(ctx: Any, family: str, tag: str, mtype: str = "") -> str:
    """The first taxonomy category whose regex matches "family mechanism_type tag" ('other' when none)."""
    tax = dict(opt(ctx, "taxonomy") or {})
    key = tuple(sorted(tax.items()))
    if key not in _RX:
        _RX[key] = [(c, re.compile(p, re.I)) for c, p in tax.items()]
    text = f"{family} {mtype} {tag}".replace("_", " ") + f" {tag}"
    return next((c for c, rx in _RX[key] if rx.search(text)), OTHER)


def node_category(ctx: Any, store: Any, nid: str) -> str:
    n = store.get_node(nid)
    if n is None:
        return OTHER
    if m := re.search(r"\bcat=(\S+)", n.notes or ""):
        return m.group(1)
    mt = m.group(1) if (m := re.search(r"\bmtype=(\w+)", n.notes or "")) else ""   # no diff reads (cheap, uncached)
    return category(ctx, n.family, n.mechanism_tag, mt)


def coverage(ctx: Any, store: Any, led: Any, held: Iterable[str] = ()) -> Counter:
    """category -> this epoch's research nodes (notes ``brief=``) + ``held`` (in-flight / queued briefs' categories)."""
    epoch = getattr(led, "epoch", "") or ""
    c: Counter = Counter({k: 0 for k in categories(ctx)})
    for nid, node in getattr(led, "nodes", {}).items():
        if (not epoch or node.base == epoch) and "brief=" in (node.notes or ""):
            c[node_category(ctx, store, nid)] += 1
    for h in held:
        if h:
            c[h] += 1
    return c


def suggest(counts: Mapping[str, int], cats: Sequence[str]) -> str:
    """The least used category (taxonomy order breaks ties; 'other' only when it is the only one)."""
    named = [c for c in cats if c != OTHER] or list(cats)
    return min(named, key=lambda c: (counts.get(c, 0), named.index(c))) if named else OTHER


def coverage_text(counts: Mapping[str, int], suggested: str) -> str:
    tot = sum(counts.values()) or 1
    low = sorted((c for c in counts if c != OTHER), key=lambda c: counts[c])[:3]
    return ("## Mechanism coverage this epoch (computed: research nodes + in-flight briefs per category)\n"
            + ", ".join(f"{c} {n} ({n / tot:.0%})" for c, n in counts.items())
            + f"\nLeast explored: {', '.join(low)}. This session is pointed at **{suggested}**: prefer a mechanism of "
              "that category when the evidence allows it (any category is allowed; declare the one you built).")
