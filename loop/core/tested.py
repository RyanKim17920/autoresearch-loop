"""loop.core.tested -- the MECHANISM MEMORY every research session is held to (11-08).

11-08 nanobrain: ~30% of 30,580 slate proposals re-proposed mechanisms the operator had already measured by hand (MNI
affine resampling 1,641 times, registers 1.2k, drop-path, JEPA / latent objectives), and one mechanism appeared
under 5-30 reworded tags across concurrent sessions. Root causes: the imported manual runs (loop.localruns) were only
prose in the brief -- opaque run names and config dumps, truncated at ``local_runs.block_chars`` before the flat runs
(registers / drop-path / latent prediction) and stripped from 15% of briefs by the component audit -- and nothing
ENFORCED them: the slate's originality and ``check_idea`` compare with LEDGER nodes only (5 nodes, 0 clusters), the
``history`` tool showed only those, and a bounced slate's candidates were never recorded (only accepted slates are
logged), so the next session proposed them again.

This module is the memory and its gate:

- :func:`memory` -- what was TESTED: closed ledger nodes with a measurement, every imported manual-run group (its
  description from the knowledge file, loop.knowledge, else derived from its name / branch / config keys / new files;
  its verdict and per-axis deltas), and the slate candidates the gate bounced earlier (they cite the original prior).
- :func:`inflight` -- what is BEING BUILT: the chosen slate candidates of other live sessions and the QUEUED / running
  nodes; :func:`recent` -- what other sessions proposed in the last ``tested.recent_hours`` (listed, not blocking).
- :func:`similarity` / :func:`nearest` -- fuzzy matching on title + tag + mechanism across reworded tags: the larger
  of a TF-IDF cosine of the full texts and an IDF-weighted overlap of the key terms (title + tag), on 7-char stems of
  words >= 3 characters (``mni``, ``ema``, ``vit`` count).
- :func:`screen` -- per slate candidate: the tested / in-flight entry it repeats (``tested.sim`` /
  ``tested.inflight_sim``) unless its ``differs_from_tested`` names that entry and says HOW it differs.
- :func:`record_slate` -- remember a slate: bounced candidates (with the prior they repeat) and the proposals.
- :func:`brief_block` (every v3 brief), :func:`session_block` (at session start), :func:`check_text` (check_idea).
"""

from __future__ import annotations

import math
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from loop.novelty import stem

K_REJECTS = "tested.rejects"      # [{id, title, mechanism, tag, family, prior, prior_result, request_id, ts}]
K_PROPOSED = "tested.proposed"    # [{request_id, i, title, mechanism, tag, family, chosen, ts}]
HEADING = "## Already tested -- do NOT re-propose (the loop bounces repeats)"
SESSION_HEADING = "## In flight now / proposed recently -- do NOT duplicate"
_LOCK = threading.Lock()
#: closed nodes whose MEASURED verdict was not a win: a repeat of one is bounced (a confirmed winner is not: building on
#: a winner is the point -- the ledger's own rules handle its doses)
MEASURED = frozenset({"screened_out", "not_confirmed", "vetoed", "audit_failed"})
LIVE = frozenset({"queued", "screening", "promoting", "confirming", "evaluating"})
_WORD = re.compile(r"[a-z][a-z0-9]{2,}")
STOP = frozenset("""the and for are was were has had its our can will may not but with from that this into than then
    them they their there which would could should about over under more most less also only each both same other such
    these those where when while because without within using used uses via per new one two all any add adds added
    adding use make makes made keep keeps kept run runs ran set sets get gets let how why what who too very much many
    just like does did done doe each every own off out onto upon its it's we us you your current incumbent recipe
    idea ideas change changes changed propose proposed proposal candidate session loop already instead rather""".split())
def _stem(w: str) -> str:
    """loop.novelty's suffix stem, untruncated (``contrast`` vs ``contrastive`` stay apart), plus -ion
    (``prediction`` -> ``predict``)."""
    w = stem(w)
    return w[:-3] if len(w) > 7 and w.endswith("ion") else w


#: words of any training-change description: present in nearly every candidate, they name no mechanism
GENERIC = frozenset(_stem(w) for w in """train training trained model models loss losses encoder decoder token
    tokens feature features representation representations learn learning learned improve improves better help helps
    step steps weight weights layer layers head heads embedding embeddings objective method approach scale scaled
    volume volumes image images data dataset input inputs output outputs network pretraining pretrain recipe
    performance evaluation eval probe probes frozen linear axis axes task tasks gain gains signal""".split())


def cfg_of(ctx: Any) -> Any:
    return getattr(getattr(ctx, "cfg", None), "tested", None)


def enabled(ctx: Any) -> bool:
    c = cfg_of(ctx)
    return bool(getattr(c, "enabled", False)) if c is not None else False


def _opt(ctx: Any, name: str, default: Any) -> Any:
    return getattr(cfg_of(ctx), name, default)


# ---------------------------------------------------------------------------------------------- text
def words(text: str) -> list[str]:
    """Stems of the words (>= 3 chars; ``_ - / . :`` split words) minus stop words."""
    t = re.sub(r"[_\-/.:=,;()\[\]{}'\"`]+", " ", str(text or "").lower())
    return [_stem(w) for w in _WORD.findall(t) if w not in STOP]



def key_terms(*parts: str) -> set[str]:
    return {w for p in parts for w in words(p)} - GENERIC


@dataclass(frozen=True)
class Entry:
    """One remembered mechanism. ``id``: ``local:<run>`` / a node id / ``slate:<request>#<i>``; ``kind``: local | node
    | reject | inflight | proposed; ``title``: what it is (one line); ``key``: its naming text (title + tag);
    ``text``: everything matched on; ``result``: the verdict line cited back; ``names``: ids it also answers to."""
    id: str
    kind: str
    title: str
    key: str
    text: str
    result: str = ""
    names: tuple[str, ...] = ()
    ts: float = 0.0
    request_id: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)


def cand_text(c: Mapping[str, Any]) -> tuple[str, str]:
    """(key text, full text) of a slate candidate / a check_idea probe."""
    key = f"{c.get('title', '')} {c.get('mechanism_tag', '')}"
    full = f"{key} {c.get('family', '')} {c.get('mechanism', '')} {c.get('claim', '')}"
    return key, full


def core_terms(e: "Entry") -> list[set[str]]:
    """The entry's CANONICAL NAME(s) as word sets: the knowledge file's head phrase (before the first ':'), its
    ' / '-separated aliases (``drop-path / stochastic depth``); none for a derived / ledger / slate entry."""
    head = str(e.extra.get("core") or "") if e.extra else ""
    out = []
    for alt in re.split(r"\s+/\s+|\s*;\s*", head):
        ws = set(words(alt))
        if ws - GENERIC:
            out.append(ws)
    return out


def _idf(docs: Sequence[Iterable[str]]) -> dict[str, float]:
    df: Counter = Counter()
    for d in docs:
        df.update(set(d))
    n = len(docs)
    return {w: math.log((n + 1) / (c + 0.5)) + 0.1 for w, c in df.items()}


def _vec(ws: Sequence[str], idf: Mapping[str, float], default: float) -> dict[str, float]:
    tf = Counter(ws)
    v = {w: (1.0 + math.log(c)) * idf.get(w, default) for w, c in tf.items() if w not in GENERIC}
    z = math.sqrt(sum(x * x for x in v.values())) or 1.0
    return {w: x / z for w, x in v.items()}


class Matcher:
    """Similarity of probes to a fixed set of entries (IDF over the entries + a probe set)."""

    def __init__(self, entries: Sequence[Entry], probes: Sequence[Mapping[str, Any]] = ()) -> None:
        self.entries = list(entries)
        docs = [words(e.text) for e in self.entries] + [words(cand_text(c)[1]) for c in probes]
        self.idf = _idf(docs) if docs else {}
        self.default = math.log((len(docs) + 1) / 0.5) + 0.1
        self.vecs = [_vec(words(e.text), self.idf, self.default) for e in self.entries]
        self.keys = [key_terms(e.key) for e in self.entries]
        self.cores = [core_terms(e) for e in self.entries]

    def _w(self, w: str) -> float:
        return self.idf.get(w, self.default)

    def score(self, i: int, key: set[str], vec: Mapping[str, float], raw: set[str] | None = None) -> float:
        cos = sum(x * self.vecs[i].get(w, 0.0) for w, x in vec.items())
        core = 0.0                       # the entry's canonical name ("register tokens") contained in the probe's title
        for alt in self.cores[i]:
            have = alt & (raw or key)
            if not have - GENERIC:
                continue
            cover = sum(self._w(w) for w in have) / sum(self._w(w) for w in alt)
            # the whole name (every non-generic word) present = a match; a partial name only counts half
            core = max(core, 1.0 if not (alt - GENERIC) - have else 0.5 * cover)
        ek = self.keys[i]
        both = key & ek
        ov = 0.0
        if both and (len(both) >= 2 or min(len(key), len(ek)) == 1):
            ov = sum(self._w(w) for w in both) / min(sum(self._w(w) for w in key), sum(self._w(w) for w in ek))
        return max(cos, ov, core)

    def nearest(self, c: Mapping[str, Any], kinds: Iterable[str] | None = None) -> tuple[Entry | None, float]:
        k, full = cand_text(c)
        key, vec, raw = key_terms(k), _vec(words(full), self.idf, self.default), set(words(k))
        want = set(kinds) if kinds is not None else None
        best, bs, bc = None, 0.0, -1.0
        for i, e in enumerate(self.entries):
            if want is not None and e.kind not in want:
                continue
            s = self.score(i, key, vec, raw)
            cos = sum(x * self.vecs[i].get(w, 0.0) for w, x in vec.items())   # ties: the closer full text
            if s > bs or (s == bs and s > 0 and cos > bc):
                best, bs, bc = e, s, cos
        return best, round(bs, 4)


def similarity(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    """Similarity of two candidates (title / mechanism_tag / mechanism) on their own two-document IDF."""
    k, full = cand_text(b)
    e = Entry("b", "proposed", str(b.get("title") or ""), k, full)
    return Matcher([e], [a]).nearest(a)[1]


# ---------------------------------------------------------------------------------------------- the memory
def _f(x: Any, sign: bool = True) -> str:
    return f"{x:+.4f}" if sign and isinstance(x, (int, float)) and math.isfinite(x) else \
        (f"{x:.4f}" if isinstance(x, (int, float)) and math.isfinite(x) else "?")


def _ax(k: str) -> str:
    return k.replace("nb_", "").replace("_mean", "").replace("final_probe_", "")


_GENERIC_FILES = {"train", "model", "data", "misc", "modules", "optim", "prep", "utils", "__init__", "config"}
_NOISE_KEYS = re.compile(r"(^|\.)(epochs|decay_epochs|warmup_epochs|cudagraphs|time_limit|total_steps|checkpoint_\w+|"
                         r"fast_startup|gc_freeze|high_priority_stream|decoder_prune_last|prep_threads|distributed|"
                         r"flash_attention_impl|time_schedule|time_decay_frac)$")


def derived_title(g: Mapping[str, Any]) -> str:
    """A group's mechanism words when the knowledge file does not describe it: the run name (seed suffix and lane
    prefix off), the branch, the changed config keys (and string values) and the new source files."""
    names = [re.sub(r"-s\d+$", "", str(n)) for n in g.get("names") or ()]
    parts = list(dict.fromkeys(names))
    br = str(g.get("branch") or "").rsplit("/", 1)[-1]
    if br:
        parts.append(br)
    cd = g.get("config_delta") or {}
    for k, v in list(cd.items())[:12]:
        if _NOISE_KEYS.search(k):
            continue
        new = v[1] if isinstance(v, (list, tuple)) and len(v) == 2 else v
        parts.append(f"{k.rsplit('.', 1)[-1]}={new if isinstance(new, (str, int, float, bool)) else '...'}")
    files = [str(p).rsplit("/", 1)[-1].removesuffix(".py") for p in (g.get("code_numstat") or {})]
    parts += [f for f in files if f not in _GENERIC_FILES]
    return " ".join(parts)


def group_result(g: Mapping[str, Any], axes: Sequence[str] = ()) -> str:
    v = str(g.get("verdict") or "?")
    d, z, a = g.get("delta") or {}, g.get("z") or {}, g.get("abs") or {}
    if d:
        zp = z.get("panel")
        out = f"{v}: panel {_f(d.get('panel'))}" + (f" ({zp:+.1f} sd)" if isinstance(zp, (int, float)) else "")
        out += "".join(f", {_ax(k)} {_f(d.get(k))}" + (f" ({z[k]:+.1f})" if isinstance(z.get(k), (int, float)) else "")
                       for k in axes if k in d)
        return out + f" [n={g.get('n_paired', len(g.get('names') or ()))}]"
    return f"{v}: panel {_f(a.get('panel'), False)} (absolute)"


def local_entries(ctx: Any, store: Any) -> list[Entry]:
    """One entry per imported manual-run group (loop.localruns), described by the knowledge file when it can be."""
    from loop import knowledge as KN
    from loop import localruns as LR
    if not LR.enabled(ctx) or not hasattr(store, "local_runs"):
        return []
    rows = store.local_runs()
    if not rows:
        return []
    d = LR.compute(ctx, rows)
    desc = KN.descriptions(ctx)
    out = []
    for g in d.get("groups") or ():
        names = tuple(str(n) for n in g.get("names") or ())
        if not names:
            continue
        auto = derived_title(g)
        human = KN.describe(names, desc)
        title = human or auto
        base = re.sub(r"-s\d+$", "", names[0])
        core = human.split(":", 1)[0] if human and ":" in human else ""
        out.append(Entry(id=f"local:{names[0]}", kind="local", title=title[:240], key=f"{title} {base}",
                         text=f"{title} {auto}", result=group_result(g, d.get("axes") or ()),
                         names=tuple(f"local:{n}" for n in names) + names + (f"local:{base}", base),
                         extra={"core": core} if core else {}))
    return out


def _status(n: Any) -> str:
    s = getattr(n, "status", "")
    return str(getattr(s, "value", s) or "").lower()


def _pruned(n: Any) -> bool:
    """A QUEUED node pruned unscreened because its mechanism family measured flat (``alloc.prune_stale_family``): it is
    in the memory like a measured node, so the mechanism is not re-proposed."""
    from loop.core.alloc import PRUNED_REASON
    return _status(n) == "rejected" and str(getattr(n, "reject_reason", "") or "").startswith(PRUNED_REASON)


def _claim(store: Any, n: Any) -> str:
    try:
        h = store.get_hypothesis(n.hypothesis_id) if getattr(n, "hypothesis_id", None) else None
    except Exception:  # noqa: BLE001
        h = None
    return str(getattr(h, "claim", "") or "")


def _table_row(nid: str) -> Mapping[str, Any] | None:
    try:
        from loop.agents.history import current_table
        tab = current_table()
    except Exception:  # noqa: BLE001 - no history view yet: status only
        return None
    return (tab.by_id.get(nid) if tab is not None else None) or None


def node_entries(store: Any, statuses: frozenset[str]) -> list[Entry]:
    out = []
    try:
        nodes = list(store.nodes())
    except Exception:  # noqa: BLE001
        return []
    for n in nodes:
        pruned = statuses is MEASURED and _pruned(n)       # 11-09: a stale-family prune is a family verdict
        if n.id == "root" or (_status(n) not in statuses and not pruned):
            continue
        tag = f"{getattr(n, 'family', '')}/{getattr(n, 'mechanism_tag', '')}"
        claim = _claim(store, n)
        r = _table_row(n.id)
        res = _status(n) if not pruned else str(getattr(n, "reject_reason", "") or "pruned")[:260]
        if r is not None and r.get("m") is not None:
            res += f": own increment m {_f(r['m'])} (sd {_f(r.get('sd'), False)}, n={r.get('n')})"
        title = (claim[:200] or tag)
        out.append(Entry(id=n.id, kind="node" if statuses is MEASURED else "inflight", title=title,
                         key=f"{getattr(n, 'mechanism_tag', '')} {claim[:160]}", text=f"{tag} {claim}", result=res,
                         ts=float(getattr(n, "created_ts", 0.0) or 0.0)))
    return out


def _state(store: Any, key: str) -> list[dict[str, Any]]:
    try:
        v = store.get_state(key) if hasattr(store, "get_state") else None
    except Exception:  # noqa: BLE001
        v = None
    return [x for x in v if isinstance(x, Mapping)] if isinstance(v, list) else []


def reject_entries(store: Any) -> list[Entry]:
    out = []
    for r in _state(store, K_REJECTS):
        k = f"{r.get('title', '')} {r.get('tag', '')}"
        out.append(Entry(id=str(r.get("id") or ""), kind="reject", title=str(r.get("title") or "")[:200], key=k,
                         text=f"{k} {r.get('family', '')} {str(r.get('mechanism') or '')[:600]}",
                         result=f"bounced at a slate as a repeat of {r.get('prior')} ({str(r.get('prior_result') or '')[:200]})",
                         names=(str(r.get("prior") or ""),), ts=float(r.get("ts") or 0.0),
                         request_id=str(r.get("request_id") or ""),
                         extra={"prior": r.get("prior"), "prior_result": r.get("prior_result")}))
    return out


def memory(ctx: Any, store: Any) -> list[Entry]:
    """Everything TESTED (or bounced as a repeat of a tested thing): local-run groups, measured closed nodes, rejects."""
    return local_entries(ctx, store) + node_entries(store, MEASURED) + reject_entries(store)


def _live_ids(store: Any) -> set[str]:
    try:
        from loop.contracts import K_INFLIGHT
        return {str(e.get("id", "")) for e in store.get_state(K_INFLIGHT) or [] if not e.get("queued")}
    except Exception:  # noqa: BLE001
        return set()


def _proposal_entry(p: Mapping[str, Any], kind: str) -> Entry:
    k = f"{p.get('title', '')} {p.get('tag', '')}"
    rid = str(p.get("request_id") or "")
    return Entry(id=f"slate:{rid}#{p.get('i', 0)}", kind=kind, title=str(p.get("title") or "")[:200], key=k,
                 text=f"{k} {p.get('family', '')} {str(p.get('mechanism') or '')[:600]}",
                 result=("being built by session " + rid) if kind == "inflight" else f"proposed by {rid}",
                 ts=float(p.get("ts") or 0.0), request_id=rid)


def inflight(ctx: Any, store: Any, request_id: str = "") -> list[Entry]:
    """What other sessions are building now (their chosen slate candidates) + the QUEUED / running nodes."""
    live = _live_ids(store) - {request_id}
    out = [_proposal_entry(p, "inflight") for p in _state(store, K_PROPOSED)
           if p.get("chosen") and str(p.get("request_id") or "") in live]
    return out + node_entries(store, LIVE)


def recent(ctx: Any, store: Any, request_id: str = "", now: float | None = None) -> list[Entry]:
    """Slate candidates other sessions proposed (not chosen) in the last ``tested.recent_hours``, newest first."""
    now = time.time() if now is None else now
    h = float(_opt(ctx, "recent_hours", 12.0)) * 3600.0
    ps = [p for p in _state(store, K_PROPOSED) if not p.get("chosen") and str(p.get("request_id") or "") != request_id
          and now - float(p.get("ts") or 0.0) <= h]
    return [_proposal_entry(p, "proposed") for p in sorted(ps, key=lambda p: -float(p.get("ts") or 0.0))]


# ---------------------------------------------------------------------------------------------- the gate
def _how_ok(how: str, e: Entry, min_words: int) -> bool:
    return len(set(words(how)) - set(words(e.text)) - GENERIC) >= min_words


def _differs(c: Mapping[str, Any]) -> list[tuple[str, str]]:
    raw = c.get("differs_from_tested")
    if isinstance(raw, Mapping):
        raw = [raw]
    if isinstance(raw, str):
        return [("", raw)]
    return [(str(x.get("id") or "").strip(), str(x.get("how") or "")) for x in raw or () if isinstance(x, Mapping)]


def _names(e: Entry) -> set[str]:
    out = {e.id.lower(), *(n.lower() for n in e.names if n)}
    return out | {n.removeprefix("local:") for n in out}


def excused(c: Mapping[str, Any], e: Entry, min_words: int) -> bool:
    """The candidate's ``differs_from_tested`` names ``e`` (any of its ids; a bounced candidate's prior too) and says
    how it differs: >= ``min_words`` content words the entry's text does not contain."""
    ids = _names(e)
    return any(i.lower() in ids and _how_ok(how, e, min_words) for i, how in _differs(c))


@dataclass(frozen=True)
class Hit:
    entry: Entry
    sim: float
    kind: str           # tested | inflight

    def cite(self) -> str:
        e = self.entry
        what = "repeats tested" if self.kind == "tested" else "duplicates in-flight"
        return f"{what} {e.id} ({e.title[:110]}) -- {e.result[:260]} [similarity {self.sim:.2f}]"


def screen(ctx: Any, store: Any, cands: Sequence[Mapping[str, Any]], request_id: str = "",
           mem: Sequence[Entry] | None = None, live: Sequence[Entry] | None = None) -> list[Hit | None]:
    """Per candidate: the tested / in-flight entry it repeats (None = clear). An excused match (its
    ``differs_from_tested`` names the entry and says how) is clear."""
    if not enabled(ctx) or not cands:
        return [None] * len(cands)
    mem = memory(ctx, store) if mem is None else list(mem)
    live = inflight(ctx, store, request_id) if live is None else list(live)
    if not mem and not live:
        return [None] * len(cands)
    m = Matcher([*mem, *live], cands)
    tau, tau_live = float(_opt(ctx, "sim", 0.5)), float(_opt(ctx, "inflight_sim", 0.5))
    mw = int(_opt(ctx, "how_min_words", 3))
    out: list[Hit | None] = []
    for c in cands:
        hit = None
        for kinds, tau_k, label in ((("local", "node", "reject"), tau, "tested"), (("inflight",), tau_live, "inflight")):
            e, s = m.nearest(c, kinds)
            if e is not None and s >= tau_k and not excused(c, e, mw):
                hit = Hit(e, s, label)
                break
        out.append(hit)
    return out


def bounce_text(i: int, c: Mapping[str, Any], h: Hit) -> str:
    return (f"candidate {i} ({str(c.get('title') or '')[:80]}) {h.cite()}. Drop it, or keep it with "
            f"differs_from_tested: [{{id: '{h.entry.id}', how: '<what is concretely different and why that answers "
            f"the result above>'}}]")


def record_slate(ctx: Any, store: Any, request_id: str, cands: Sequence[Mapping[str, Any]],
                 hits: Sequence[Hit | None], chosen: int | None, *, now: float | None = None) -> None:
    """Remember a slate: every bounced candidate (``K_REJECTS``, citing the ORIGINAL prior) and every clear candidate
    as a proposal (``K_PROPOSED``; the chosen one marks the session's in-flight work)."""
    if not enabled(ctx) or not hasattr(store, "set_state"):
        return
    now = time.time() if now is None else now
    rej, prop = [], []
    for i, c in enumerate(cands):
        base = {"title": str(c.get("title") or "")[:200], "mechanism": str(c.get("mechanism") or "")[:600],
                "tag": str(c.get("mechanism_tag") or "")[:120], "family": str(c.get("family") or "")[:60],
                "request_id": request_id, "i": i, "ts": now}
        h = hits[i] if i < len(hits) else None
        if h is not None and h.kind == "tested":
            e = h.entry
            prior = str(e.extra.get("prior") or e.id) if e.kind == "reject" else e.id
            pres = str(e.extra.get("prior_result") or e.result) if e.kind == "reject" else e.result
            rej.append({**base, "id": f"slate:{request_id}#{i}", "prior": prior, "prior_result": pres[:300]})
        elif h is None:
            prop.append({**base, "chosen": chosen == i})
    with _LOCK:
        if rej:
            cur = _state(store, K_REJECTS)
            have = {r.get("id") for r in cur}
            cur += [r for r in rej if r["id"] not in have]
            store.set_state(K_REJECTS, cur[-int(_opt(ctx, "keep_rejects", 400)):])
        if prop:
            cur = [p for p in _state(store, K_PROPOSED)
                   if not (p.get("request_id") == request_id and any(p.get("i") == q["i"] for q in prop))]
            if chosen is not None:          # a re-proposed slate of this session replaces its earlier chosen mark
                for p in cur:
                    if p.get("request_id") == request_id:
                        p["chosen"] = False
            cur += prop
            store.set_state(K_PROPOSED, cur[-int(_opt(ctx, "keep_proposed", 300)):])


def check_text(ctx: Any, store: Any, request_id: str, *, tag: str = "", family: str = "", claim: str = "",
               title: str = "") -> str:
    """check_idea's memory rule: 'REJECT already_tested: ...' / 'REJECT inflight_duplicate: ...' or ''."""
    if not enabled(ctx) or not (tag.strip() or claim.strip() or title.strip()):
        return ""
    probe = {"title": title or tag.replace("_", " "), "mechanism_tag": tag, "family": family, "mechanism": claim}
    h = screen(ctx, store, [probe], request_id)[0]
    if h is None:
        return ""
    rule = "already_tested" if h.kind == "tested" else "inflight_duplicate"
    return (f"REJECT {rule}: this mechanism {h.cite()}. Pick another mechanism, or (in the slate) carry "
            f"differs_from_tested: [{{id: '{h.entry.id}', how}}] saying what is concretely different")


# ---------------------------------------------------------------------------------------------- the brief
def brief_block(ctx: Any, store: Any) -> str:
    """Every v3 brief: the memory, one line per mechanism (manual runs first, then measured nodes, then the bounced
    repeats grouped by the prior they repeated). '' when off / empty."""
    if not enabled(ctx):
        return ""
    try:
        loc = local_entries(ctx, store)
        nodes = sorted(node_entries(store, MEASURED), key=lambda e: -e.ts)
        rej = reject_entries(store)
    except Exception as exc:  # noqa: BLE001 - a broken memory never blocks a brief
        return f"{HEADING}\n(unavailable: {type(exc).__name__})"
    if not (loc or rej):            # measured ledger nodes alone: the history tool / clusters already brief them
        return ""                    # (nanopath's briefs stay as they were)
    cap = int(_opt(ctx, "brief_max", 80))
    lines = [f"- {e.id}: {e.title[:160]} -> {e.result[:200]}" for e in loc]
    room = max(0, min(cap - len(lines), int(_opt(ctx, "brief_nodes", 20))))
    lines += [f"- {e.id}: {e.title[:160]} -> {e.result[:200]}" for e in nodes[:room]]
    more = len(loc) + len(nodes) - len(lines)
    if more > 0:
        lines.append(f"- ... {more} more measured nodes (history(action='tested') lists all)")
    if rej:
        by = Counter(str(e.extra.get("prior") or "") for e in rej)
        lines.append("Bounced as repeats so far (prior: times): " + ", ".join(f"{p} x{n}" for p, n in by.most_common(12)))
    return (f"{HEADING}\nEvery mechanism below was already MEASURED (single-seed manual runs `local:` = the operator's; "
            "node ids = the loop's ledger). The slate gate and check_idea match your candidates against this list by "
            "meaning (reworded tags match too) and bounce a repeat with its result. To build on one anyway, the "
            "candidate must carry differs_from_tested: [{id, how}] saying what is concretely different and why that "
            "answers the measured result.\n" + "\n".join(lines))


def session_block(ctx: Any, store: Any, request_id: str = "") -> str:
    """At session start: what other sessions are building now and what they proposed recently ('' when none)."""
    if not enabled(ctx):
        return ""
    try:
        live = inflight(ctx, store, request_id)
        rec = recent(ctx, store, request_id)
    except Exception:  # noqa: BLE001
        return ""
    if not (live or rec):
        return ""
    cap = int(_opt(ctx, "recent_max", 30))
    lines = [f"- IN FLIGHT {e.id}: {e.title[:150]}" for e in live][:cap]
    seen: list[Entry] = []
    for e in rec:                                   # one line per distinct mechanism
        if len(lines) >= cap:
            break
        if any(similarity({"title": e.title, "mechanism": e.text}, {"title": s.title, "mechanism": s.text}) >= 0.6
               for s in seen):
            continue
        seen.append(e)
        lines.append(f"- proposed {e.id}: {e.title[:150]}")
    return (f"\n\n{SESSION_HEADING}\nOther sessions are building the IN FLIGHT mechanisms now (a near-duplicate is "
            "bounced) and proposed the others in the last "
            f"{float(_opt(ctx, 'recent_hours', 12.0)):g} h. Spend your slate on mechanisms NOT on this list.\n"
            + "\n".join(lines))


def table_text(ctx: Any, store: Any, limit: int = 0) -> str:
    """history(action='tested'): the whole memory + what is in flight."""
    mem, live = memory(ctx, store), inflight(ctx, store)
    n = max(1, int(limit or 200))
    rows = [f"{e.id} | {e.kind} | {e.title[:120]} | {e.result[:200]}" for e in [*mem, *live]][:n]
    return ("tested mechanism memory (local = the operator's manual runs; node = measured ledger nodes; reject = slate "
            "candidates bounced as repeats; inflight = being built now)\nid | kind | what | result\n"
            + ("\n".join(rows) if rows else "(empty)"))
