"""loop.meta — META-LEARNING for every LLM decision-maker (DESIGN §6.3). Each component's decisions are
resolved against what the loop later MEASURED (code, no LLM call), summarised as a short track record
(accuracy / calibration / precision, GPU-h per success, concrete right and wrong calls, a calibration note)
and injected into that component's own prompt, next to a review-tunable bias knob in [-1, 1] rendered as
'be more / less X'. The scorecard (one line per component) goes to ``loop status``, the dashboard and the
big / first-principles review evidence, so the reviews can move the biases (loop.selftune).

Decisions come from records the loop already keeps (hypotheses, postmortems, reviews, the design and
tuning books) plus a small decision log (``meta_decision`` events) for what nothing else stores: which
review probe a request served.

CONCISE in prompts (operator 2026-09-30): a record is <= MAX_RECORD_CHARS / MAX_RECORD_LINES (headline, one
calibration note, <= 2 one-line examples; examples dropped first, never the headline), a review's combined meta
block <= MAX_META_BLOCK. CONSOLIDATION (``consolidate``): with each first-principles review or every
``meta.consolidate_every`` reviews an Opus call condenses everything into a versioned PLAYBOOK of <= 10 one-line
rules whose numbers must appear in the computed records (else dropped); prompts then carry the component's
rules + its live headline, retired rules stay in state only. The raw records stay in status / dashboard."""

from __future__ import annotations

import json
import math
import re
import time
from collections import defaultdict
from typing import Any, Callable, Mapping, Sequence

from loop.contracts import (CLOSED_STATUSES, K_EPOCH, K_STALE_META, PANEL, ROOT_NODE_ID, HypothesisKind, HypothesisStatus, NodeStatus,
                            Operator, Strategy, current_reviews)
from loop.quality import _pearson, _rank, prediction_stats, predictions

K_DECISION = "meta_decision"          # log event: component, decision_id, node_id, hypothesis_id, review_id, payload
DAY = 86400.0
WINDOW = 12 * 3600.0                  # a self-tune change: closes / hit rate this long before vs after it
#: component -> (what 'be MORE ...' says, what 'be LESS ...' says); its bias knob is ``meta.<c>_bias``
COMPONENTS: Mapping[str, tuple[str, str]] = {
    "researcher": ("MORE bold: larger changes, larger predicted effects",
                   "MORE conservative: smaller changes, smaller predicted effects"),
    "review": ("MORE exploratory: bolder meta-hypotheses, more moonshot probes",
               "MORE conservative: meta-hypotheses and probes close to the measured evidence"),
    "first_principles": ("MORE willing to call an assumption broken and propose structural change",
                         "MORE sceptical: call an assumption broken only on direct evidence"),
    "design": ("MORE aggressive in designed-batch knobs: bigger low->high moves",
               "MORE conservative in designed-batch knobs: smaller moves"),
    "crossover": ("MORE willing to combine / stack parents", "MORE selective about which parents to combine"),
    "scout": ("MORE speculative: far-field ideas, new paradigms, ideas from outside the task's own field",
              "MORE grounded: ideas close to the loop's measured evidence"),
    "persona": ("MORE in character: push the persona's angle further",
                "LESS in character: let the evidence override the persona's angle"),
}
#: prompt budgets (operator 2026-09-30: meta-learning stays CONCISE in prompts)
MAX_RECORD_CHARS = 500                # one component's record in its own prompt
MAX_RECORD_LINES = 5                  # headline + calibration note + <= 2 one-line examples (+ the bias line)
MAX_EXAMPLES = 2
EXAMPLE_CHARS = 160
MAX_META_BLOCK = 1500                 # a prompt with several components' records (reviews), combined


def bias_knob(component: str) -> str:
    return f"meta.{component}_bias"


def bias(cfg: Any, component: str) -> float:
    obj: Any = cfg
    for part in bias_knob(component).split("."):
        obj = getattr(obj, part, 0.0)
    return max(-1.0, min(1.0, float(obj or 0.0)))


def bias_line(cfg: Any, component: str) -> str:
    """'' below |0.25|; else the review-set nudge in words."""
    b = bias(cfg, component)
    if abs(b) < 0.25:
        return ""
    more, less = COMPONENTS[component]
    return f"Reviews set your bias to {b:+.2f}: be {more if b > 0 else less} than your default."


# ------------------------------------------------------------------ the decision log
def record(store: Any, component: str, decision_id: str, *, node: str = "", hypothesis: str = "",
           review: str = "", **payload: Any) -> None:
    store.log(K_DECISION, component=component, decision_id=decision_id, node_id=node, hypothesis_id=hypothesis,
              review_id=review, payload=payload)


def decisions(store: Any, component: str | None = None) -> list[dict[str, Any]]:
    return [e for e in store.events(K_DECISION) if component is None or e.get("component") == component]


# ------------------------------------------------------------------ shared facts
class _Facts:
    """One pass over the store per resolution: nodes, GPU-h per node, scored posteriors."""

    def __init__(self, store: Any, snapshot: Any, clock: Callable[[], float] = time.time) -> None:
        self.store, self.snap, self.clock = store, snapshot, clock
        self.nodes = {n.id: n for n in store.nodes() if n.id != ROOT_NODE_ID}
        self.gpu: dict[str, float] = defaultdict(float)
        self._claims: dict[str, str] = {}
        for r in store.runs():
            self.gpu[r.node_id] += r.gpu_h or 0.0

    def post(self, nid: str) -> Any:
        p = self.snap.nodes.get(nid) if self.snap is not None else None
        return p if p is not None and p.utility.n > 0 else None

    def hit(self, nid: str) -> bool | None:
        """Screened hit: its OWN increment over its measure parent is positive (P(gain>0) > 0.5), never the U it
        inherits from its carrier (live 2026-10-03: every child of a node 'hit' on a node's +0.008); None while
        unscored."""
        p = self.post(nid)
        return None if p is None else p.p_gain > 0.5

    def claim(self, nid: str, n: int = 110) -> str:
        if nid not in self._claims:
            node = self.nodes.get(nid)
            h = self.store.get_hypothesis(node.hypothesis_id or "") if node is not None else None
            self._claims[nid] = " ".join((h.claim if h else node.mechanism_tag if node else "?").split())
        return self._claims[nid][:n]

    def tag(self, nid: str) -> str:
        n = self.nodes.get(nid)
        return f"{nid} ({n.family}/{n.mechanism_tag[:40]})" if n else nid

    def request_of(self, nid: str) -> str:
        m = re.search(r"\brequest=(\S+)", self.nodes[nid].notes or "")
        return m.group(1) if m else ""


def _res(n: int, resolved: int, headline: str, right: Sequence[str] = (), wrong: Sequence[str] = (),
         note: str = "", gpu_h: float = 0.0, successes: int = 0, **extra: Any) -> dict[str, Any]:
    return {"n": n, "resolved": resolved, "headline": headline, "right": list(right), "wrong": list(wrong),
            "note": note, "gpu_h": gpu_h, "successes": successes, **extra}


def _rate(a: int, b: int) -> str:
    return f"{a}/{b}" + (f" ({a / b:.0%})" if b else "")


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    return _pearson(_rank(xs), _rank(ys)) if len(xs) >= 5 else math.nan


_STOP = frozenset("that this with from into then than more less their there which while would should could "
                  "about after before over under when where what does each other only also just".split())


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z_][a-z0-9_.]{3,}", (text or "").lower()) if w not in _STOP}


# ------------------------------------------------------------------ resolvers
def _researcher(ctx: Any, f: _Facts) -> dict[str, Any]:
    nodes = [n for n in f.nodes.values() if n.strategy is not Strategy.DESIGN and n.hypothesis_id]
    rows = [(n.id, p, m) for n in nodes for p, m in predictions(f.store, f.snap, [n])]
    s = prediction_stats([(p, m) for _, p, m in rows])
    hits = [n.id for n in nodes if f.hit(n.id)]
    gpu = sum(f.gpu[n.id] for n in nodes)
    if not s["n"]:
        return _res(len(nodes), 0, f"{len(nodes)} ideas, none scored yet", gpu_h=gpu)
    ex = lambda nid, p, m: f"{f.tag(nid)}: predicted {p:+.4f}, measured {m:+.4f} -- {f.claim(nid)}"  # noqa: E731
    right = [ex(*r) for r in sorted((r for r in rows if (r[1] > 0) == (r[2] > 0)), key=lambda r: abs(r[1] - r[2]))]
    wrong = [ex(*r) for r in sorted((r for r in rows if (r[1] > 0) != (r[2] > 0)), key=lambda r: -abs(r[1]))]
    opt = s["optimism"]
    note = ("too few scored predictions to judge yet." if s["n"] < 5 else
            f"you over-predict: measured effects average {s['mean_meas']:+.4f} vs your {s['mean_pred']:+.4f}; "
            "predict smaller sizes." if s["mean_meas"] <= 0 or (opt == opt and opt > 1.5) else
            f"you under-predict ({1 / opt:.1f}x): your ideas do better than you say." if opt == opt and opt < 0.67
            else f"sizes roughly calibrated (bias {s['bias']:+.4f}).")
    if s["n"] >= 5 and s["sign_acc"] < 0.5:
        note += f" Your sign is right only {s['sign_acc']:.0%} of the time: question the mechanism before the dose."
    rho = s.get("spearman", math.nan)
    return _res(len(nodes), s["n"], f"sign right {s['sign_acc']:.0%} of {s['n']} scored; predicted "
                f"{s['mean_pred']:+.4f} vs measured {s['mean_meas']:+.4f} on average"
                + (f", rank corr {rho:+.2f}" if rho == rho else "") + f"; screened hits {len(hits)}"
                + _web_clause(f), right, wrong, note, gpu, len(hits), accuracy=s["sign_acc"])


def web_split(f: _Facts) -> dict[str, tuple[int, int, int]]:
    """'web' / 'noweb' -> (research sessions, scored, screened hits) over nodes whose notes record web use
    (``web=1 searches=N`` / ``web=0``; older nodes carry neither and are left out)."""
    out = {"web": [0, 0, 0], "noweb": [0, 0, 0]}
    for n in f.nodes.values():
        m = re.search(r"\bweb=([01])\b", n.notes or "")
        if m is None or n.strategy is Strategy.DESIGN:
            continue
        row = out["web" if m.group(1) == "1" else "noweb"]
        h = f.hit(n.id)
        row[0] += 1
        row[1] += h is not None
        row[2] += bool(h)
    return {k: tuple(v) for k, v in out.items()}


def _web_clause(f: _Facts) -> str:
    w = web_split(f)
    if not w["web"][0]:
        return ""
    searches = sum(int(x) for n in f.nodes.values() for x in re.findall(r"\bsearches=(\d+)", n.notes or ""))
    return (f"; web-researched ideas hit {_rate(w['web'][2], w['web'][1])} vs no-web {_rate(w['noweb'][2], w['noweb'][1])}"
            f" ({searches} searches over {w['web'][0]} web sessions)")


def _closes(f: _Facts, lo: float, hi: float) -> tuple[int, int, int]:
    """Nodes closed (not rejected) in [lo, hi): (closed, scored, hits)."""
    ids = [n.id for n in f.nodes.values() if n.status in CLOSED_STATUSES and n.status is not NodeStatus.REJECTED
           and lo <= (n.updated_ts or 0.0) < hi]
    sc = [i for i in ids if f.hit(i) is not None]
    return len(ids), len(sc), sum(bool(f.hit(i)) for i in sc)


def _review(ctx: Any, f: _Facts, fp: bool, now: float) -> dict[str, Any]:
    revs = [rv for rv in f.store.reviews() if (rv.trigger == "first_principles") == fp]
    rids = {rv.id for rv in revs}
    metas = [h for h in f.store.hypotheses(kind=HypothesisKind.META) if h.review_id in rids]
    st = lambda h: f.snap.hypotheses[h.id].status if f.snap is not None and h.id in f.snap.hypotheses \
        else HypothesisStatus.UNTESTED  # noqa: E731
    sup = [h for h in metas if st(h) is HypothesisStatus.SUPPORTED]
    ref = [h for h in metas if st(h) is HypothesisStatus.REFUTED]
    # probe-spawned nodes: the serve log (request -> review), else the latest review before the node
    by_req = {(e.get("payload") or {}).get("request"): e.get("review_id") for e in decisions(f.store, "review_probe")}
    allrev = sorted(f.store.reviews(), key=lambda rv: rv.created_ts or 0.0)
    spawned: dict[str, str] = {}
    for n in f.nodes.values():
        if not re.search(r"\bbrief=(probe|moonshot)\b", n.notes or ""):
            continue
        rid = by_req.get(f.request_of(n.id)) or next(
            (rv.id for rv in reversed(allrev) if (rv.created_ts or 0.0) <= n.created_ts and rv.probe_briefs), "")
        if rid in rids:
            spawned[n.id] = rid
    sc = [i for i in spawned if f.hit(i) is not None]
    ph = [i for i in sc if f.hit(i)]
    _, all_sc, all_h = _closes(f, 0.0, math.inf)
    tunes = [e for e in f.store.events("selftune") if e.get("source") in rids]
    tb = ta = hb = ha = 0.0
    done = [e for e in tunes if now >= e["ts"] + WINDOW]
    for e in done:
        b, a = _closes(f, e["ts"] - WINDOW, e["ts"]), _closes(f, e["ts"], e["ts"] + WINDOW)
        tb, ta = tb + b[0], ta + a[0]
        hb, ha = hb + (b[2] / b[1] if b[1] else 0.0), ha + (a[2] / a[1] if a[1] else 0.0)
    right = [f"meta {h.id} SUPPORTED: {h.claim[:100]}" for h in sup] + \
        [f"probe node {f.tag(i)} HIT -- {f.claim(i, 90)}" for i in ph]
    wrong = [f"meta {h.id} REFUTED: {h.claim[:100]}" for h in ref] + \
        [f"probe node {f.tag(i)} missed -- {f.claim(i, 90)}" for i in sc if not f.hit(i)]
    avg = all_h / all_sc if all_sc else math.nan
    notes = []
    if len(sc) >= 3 and avg == avg and len(ph) / len(sc) < avg - 0.1:
        notes.append(f"your probes hit less often ({len(ph) / len(sc):.0%}) than the loop's average ({avg:.0%}): "
                     "ground them in the measured lever trends.")
    if len(ref) > len(sup) and len(ref) >= 2:
        notes.append("more of your meta-hypotheses were refuted than supported: claim less, predict sharper.")
    if done and ta < tb:
        notes.append("the self-tune changes you made were followed by LOWER throughput: undo or reverse them.")
    headline = (f"{len(revs)} reviews: metas {len(sup)} supported / {len(ref)} refuted of {len(metas)}; probes "
                f"spawned {len(spawned)} nodes, hit {_rate(len(ph), len(sc))} (loop avg "
                + ("-" if avg != avg else f"{avg:.0%}") + ")"
                + (f"; self-tune {len(tunes)} changes, {len(done)} judged: closes/12h {tb / len(done):.1f} -> "
                   f"{ta / len(done):.1f}, hit rate {hb / len(done):.0%} -> {ha / len(done):.0%}" if done else
                   f"; self-tune {len(tunes)} changes (none 12 h old yet)" if tunes else ""))
    return _res(len(revs), len(sup) + len(ref) + len(sc) + len(done), headline, right, wrong,
                " ".join(notes) or ("too little resolved to judge yet." if len(sc) + len(sup) + len(ref) < 3
                                    else "no systematic miss found."),
                sum(f.gpu[i] for i in spawned), len(ph) + len(sup))


def _design(ctx: Any, f: _Facts) -> dict[str, Any]:
    from loop.design import K_DESIGN
    asked = set()
    for rv in f.store.reviews():
        try:
            asked |= {str(k.get("key")) for k in json.loads(rv.design or "[]") if isinstance(k, dict)}
        except ValueError:
            pass
    right, wrong, gpu = [], [], 0.0
    counts = {"review": [0, 0], "auto": [0, 0]}
    batches = [b for b in (f.store.get_state(K_DESIGN, {}) or {}).get("batches", []) if "effects" in b]
    for b in batches:
        gpu += sum(f.gpu.get(i, 0.0) for i in b.get("points", {}).values())
        for kb, e in zip(b["knobs"], b["effects"]):
            if not e.get("estimable", True) or e.get("effect") is None:
                continue
            src = "review" if kb["key"] in asked else "auto"
            counts[src][0] += 1
            counts[src][1] += bool(e["significant"])
            txt = f"d{b['batch']} {kb['key']}: {kb.get('low')!r} -> {kb.get('high')!r} ({src}) effect {e['effect']:+.5f}±{e['se']:.5f}"
            (right if e["significant"] else wrong).append(txt + (" significant" if e["significant"] else " null"))
    k, s = counts["review"][0] + counts["auto"][0], counts["review"][1] + counts["auto"][1]
    note = ("too few designed knobs to judge yet." if k < 4 else
            "most designed knobs were null: pick knobs with prior evidence and move them further." if s / k < 0.25
            else f"{s / k:.0%} of designed knobs were significant.")
    return _res(k, k, f"{len(batches)} batches, {k} knobs: significant {_rate(s, k)}; review-picked "
                f"{_rate(counts['review'][1], counts['review'][0])}, auto {_rate(counts['auto'][1], counts['auto'][0])}",
                right, wrong, note, gpu, s, precision=s / k if k else None)


def _crossover(ctx: Any, f: _Facts) -> dict[str, Any]:
    rows = []
    for n in f.nodes.values():
        m = re.search(r"\bxover=(\S+),(\S+)", n.notes or "")
        comp = re.search(r"\bcompound=(\S+)", n.notes or "")
        if not (m or comp or n.operator is Operator.CROSSOVER):
            continue
        parents = [*m.groups()] if m else [n.parent_id or ""] + (
            (re.search(r"\bstack_from=(\S+)", n.notes or "") or [None, ""])[1].split(",") if comp else [])
        rows.append((n.id, "compound" if comp else "crossover", [p for p in parents if p and p != ROOT_NODE_ID]))
    right, wrong, sc, beat, hits = [], [], 0, 0, 0
    for nid, kind, ps in rows:
        p = f.post(nid)
        if p is None:
            continue
        sc += 1
        pu = [q.utility.mean for i in ps if (q := f.post(i)) is not None]
        best = max(pu, default=0.0)
        b = p.utility.mean > best
        beat += b
        hits += p.p_gain > 0.5
        (right if b else wrong).append(f"{kind} {f.tag(nid)} of {','.join(ps) or 'root'}: U {p.utility.mean:+.4f} vs "
                                       f"best parent {best:+.4f} -- {f.claim(nid, 80)}")
    note = ("too few combinations measured to judge yet." if sc < 3 else
            f"only {beat}/{sc} combinations beat their best parent: combine only parents whose gains act on "
            "different mechanisms." if beat / sc < 0.3 else f"{beat}/{sc} combinations beat their best parent.")
    return _res(len(rows), sc, f"{len(rows)} crossover/compound nodes; beat best parent {_rate(beat, sc)}; "
                f"screened hits {hits}", right, wrong, note, sum(f.gpu[i] for i, _, _ in rows), beat)


def _scout(ctx: Any, f: _Facts) -> dict[str, Any]:
    """The research scout (loop.scout): its runs and ideas; outcome = hit rate / confirmations of the nodes
    spawned from its ideas (notes ``bank=scout-...``), against the loop's average; plus web vs no-web."""
    from loop.scout import K_SCOUT
    runs = (f.store.get_state(K_SCOUT) or {}).get("runs", [])
    written = sum(int(r.get("written", 0)) for r in runs)
    spawned = [n for n in f.nodes.values() if re.search(r"\bbank=scout-", n.notes or "")]
    sc = [n.id for n in spawned if f.hit(n.id) is not None]
    hits = [i for i in sc if f.hit(i)]
    conf = [n.id for n in spawned if n.status in (NodeStatus.CONFIRMED, NodeStatus.AUDIT_FAILED)]
    _, all_sc, all_h = _closes(f, 0.0, math.inf)
    avg = all_h / all_sc if all_sc else math.nan
    w = web_split(f)
    note = ("too few scouted ideas measured to judge yet." if len(sc) < 3 else
            f"your ideas hit less often ({len(hits) / len(sc):.0%}) than the loop's average ({avg:.0%}): ground "
            "them in the loop's measured evidence and the lessons." if avg == avg and len(hits) / len(sc) < avg - 0.1
            else f"your ideas hit {len(hits) / len(sc):.0%} (loop average " + ("-" if avg != avg else f"{avg:.0%}")
            + "): keep the mix.")
    headline = (f"{len(runs)} scout runs wrote {written} ideas; {len(spawned)} became nodes: hit {_rate(len(hits), len(sc))}"
                f", confirmed {len(conf)} (loop avg hit " + ("-" if avg != avg else f"{avg:.0%}") + ")"
                + (f"; web research sessions hit {_rate(w['web'][2], w['web'][1])} vs no-web "
                   f"{_rate(w['noweb'][2], w['noweb'][1])}" if w["web"][0] else ""))
    return _res(len(runs) + len(spawned), len(sc), headline,
                [f"{f.tag(i)} HIT -- {f.claim(i, 90)}" for i in hits],
                [f"{f.tag(i)} missed -- {f.claim(i, 90)}" for i in sc if not f.hit(i)], note,
                sum(f.gpu[n.id] for n in spawned), len(hits) + len(conf))


def _persona(ctx: Any, f: _Facts) -> dict[str, Any]:
    """Ideation personas (loop.novelty): per persona, sessions -> screened hits / confirmations; the best one."""
    rows: dict[str, list[str]] = defaultdict(list)
    for n in f.nodes.values():
        if m := re.search(r"\bpersona=(\w+)", n.notes or ""):
            rows[m.group(1)].append(n.id)
    if not rows:
        return _res(0, 0, "no persona sessions yet")
    stat = {p: ([i for i in ids if f.hit(i) is not None], [i for i in ids if f.hit(i)],
                [i for i in ids if f.nodes[i].status in (NodeStatus.CONFIRMED, NodeStatus.AUDIT_FAILED)])
            for p, ids in rows.items()}
    rate = lambda p: (len(stat[p][1]) + 0.5) / (len(stat[p][0]) + 1.0)   # noqa: E731
    best = max(stat, key=lambda p: (rate(p), p))
    sc = sum(len(v[0]) for v in stat.values())
    head = "; ".join(f"{p} hit {_rate(len(h), len(s))}, conf {len(c)}" for p, (s, h, c) in sorted(stat.items()))
    return _res(sum(map(len, rows.values())), sc, f"{head}; leading: {best}",
                [f"{f.tag(i)} [{p}] HIT -- {f.claim(i, 90)}" for p, (_, h, _) in stat.items() for i in h],
                [f"{f.tag(i)} [{p}] missed -- {f.claim(i, 90)}" for p, (s, h, _) in stat.items() for i in s if i not in h],
                "too few scored persona ideas to judge yet." if sc < 8 else f"{best} leads on screened hit rate.",
                sum(f.gpu[i] for ids in rows.values() for i in ids), sum(len(v[1]) + len(v[2]) for v in stat.values()))


RESOLVERS: Mapping[str, Callable[..., dict[str, Any]]] = {
    "researcher": _researcher,
    "review": lambda ctx, f: _review(ctx, f, False, f.clock()),
    "first_principles": lambda ctx, f: _review(ctx, f, True, f.clock()),
    "design": _design, "crossover": _crossover, "scout": _scout, "persona": _persona}


def resolve(ctx: Any, store: Any, snapshot: Any, component: str, facts: _Facts | None = None) -> dict[str, Any]:
    return RESOLVERS[component](ctx, facts or _Facts(store, snapshot))


def _one(text: str, n: int) -> str:
    t = " ".join(str(text).split())
    return t if len(t) <= n else t[:max(0, n - 3)].rstrip() + "..."


def _fit(required_head: str, optional: Sequence[str], tail: str = "", *, max_chars: int = MAX_RECORD_CHARS,
         max_lines: int = MAX_RECORD_LINES) -> str:
    """The headline always, then ``optional`` lines in priority order while they fit (a line that does not fit
    whole is dropped unless >= 80 chars of it fit -- the first, the calibration note, from 40), then ``tail``.
    The budget counts every character incl. newlines; the headline is shortened only if it alone overflows."""
    tail_len = len(tail) + 1 if tail else 0
    head = _one(required_head, max(40, max_chars - tail_len))
    out, used = [head], len(head) + tail_len
    for k, line in enumerate(x for x in optional if x):
        if len(out) + (1 if tail else 0) >= max_lines:
            break
        room = max_chars - used - 1
        if len(line) > room:
            if room >= (40 if k == 0 else 80):         # the note, or an example cut to what is left
                line = _one(line, room)
            else:
                continue
        out.append(line)
        used += len(line) + 1
    return "\n".join(out + ([tail] if tail else []))


def raw_record(ctx: Any, store: Any, snapshot: Any, component: str, examples: int = 3,
               facts: _Facts | None = None) -> str:
    """The long form (status / dashboard / consolidation input): headline, GPU-h per success, up to
    ``examples`` right and wrong calls, the calibration note and the bias line. Never goes into a prompt."""
    r = resolve(ctx, store, snapshot, component, facts)
    if not r["n"]:
        out = ["No decisions yet."]
    else:
        out = [r["headline"]] + ([f"GPU-h per success: {r['gpu_h'] / r['successes']:.1f}"]
                                 if r["successes"] and r["gpu_h"] else [])
        good, bad = r.get("labels", ("Calls that held up", "Calls that did NOT hold up"))
        out += [f"{good}: {x}" for x in r["right"][:examples]] + [f"{bad}: {x}" for x in r["wrong"][:examples]]
        out.append(f"Calibration: {r['note']}")
    if b := bias_line(ctx.cfg, component):
        out.append(b)
    return "\n".join(out)


def track_record(ctx: Any, store: Any, snapshot: Any, component: str, examples: int = MAX_EXAMPLES,
                 facts: _Facts | None = None, max_chars: int = MAX_RECORD_CHARS) -> str:
    """The component's record for ITS OWN prompt, CONCISE: <= ``max_chars`` / MAX_RECORD_LINES lines. With a
    consolidated playbook: its playbook rules + the live headline; else the headline (+ GPU-h per success), a
    calibration note and at most 2 one-line examples (a wrong call first: most informative). Examples are
    dropped first, never the headline. The bias line only when |bias| >= 0.25."""
    r = resolve(ctx, store, snapshot, component, facts)
    tail = bias_line(ctx.cfg, component)
    head = r["headline"] if r["n"] else "No decisions yet: this is your first."
    if r["n"] and r["successes"] and r["gpu_h"] and "GPU-h per" not in head:
        head += f"; GPU-h per success {r['gpu_h'] / r['successes']:.1f}"
    pb = playbook(store)
    rules = [x for x in pb.get("rules", []) if x.get("component") == component]
    if rules:                                   # the playbook REPLACES the raw record; the headline stays live
        return _fit(f"Now: {head}", [f"Playbook v{pb['version']}: " + _one(
            f"{x['rule']} [{x.get('evidence', '')}]", EXAMPLE_CHARS + 60) for x in rules], tail, max_chars=max_chars)
    if not r["n"]:
        return _fit(head, [], tail, max_chars=max_chars)
    good, bad = r.get("labels", ("Held up", "Did NOT hold up"))
    wrong = [f"{bad}: {_one(x, EXAMPLE_CHARS)}" for x in r["wrong"]]
    right = [f"{good}: {_one(x, EXAMPLE_CHARS)}" for x in r["right"]]
    ex = [x for pair in zip(wrong, right) for x in pair] + wrong[len(right):] + right[len(wrong):]
    return _fit(head, [f"Calibration: {r['note']}", *ex[:max(0, min(examples, MAX_EXAMPLES))]], tail,
                max_chars=max_chars)


def records(ctx: Any, store: Any, snapshot: Any, components: Sequence[str], examples: int = 2,
            clock: Callable[[], float] = time.time) -> dict[str, str]:
    """component -> track record over one shared pass of the store ('' for a failing resolver)."""
    facts, out = _Facts(store, snapshot, clock), {}
    for c in components:
        try:
            out[c] = track_record(ctx, store, snapshot, c, examples, facts)
        except Exception:  # noqa: BLE001 - a missing record never blocks a decision
            out[c] = ""
    return out


def scorecard(ctx: Any, store: Any, snapshot: Any, clock: Callable[[], float] = time.time) -> list[tuple[str, str, str, float]]:
    """(component, headline + note, GPU-h per success, bias) for every component; a failing resolver says so."""
    facts, out = _Facts(store, snapshot, clock), []
    for c in COMPONENTS:
        try:
            r = resolve(ctx, store, snapshot, c, facts)
            text = r["headline"] + (f" | {r['note']}" if r["n"] and r["note"] else "")
            per = f"{r['gpu_h'] / r['successes']:.1f}" if r["successes"] and r["gpu_h"] else "-"
        except Exception as exc:  # noqa: BLE001 - one analytics bug never hides the others
            text, per = f"(error: {type(exc).__name__}: {str(exc)[:120]})", "-"
        out.append((c, text, per, bias(ctx.cfg, c)))
    return out


def scorecard_text(ctx: Any, store: Any, snapshot: Any) -> str:
    """For ``loop status`` (the full, raw form): one line per component, its bias knob in force."""
    if snapshot is None:
        return "meta     (no world fit yet)"
    return "\n".join(["meta-learning scorecard (component: record | GPU-h/success | bias knob)"] + [
        f"  {c:16s} {text} | {per} | {bias_knob(c)}={b:+.2f}" for c, text, per, b in scorecard(ctx, store, snapshot)])


def meta_block(ctx: Any, store: Any, snapshot: Any, own: Sequence[str],
               max_chars: int = MAX_META_BLOCK, clock: Callable[[], float] = time.time) -> tuple[str, str]:
    """(scorecard, own records) for a prompt carrying several components (the reviews), COMBINED <=
    ``max_chars``: the reviewer's own records first (each <= 350 chars, examples dropped first), then ONE short
    line per other component (headline only, cut to share what is left; the bias shown only when |b| >= 0.25)."""
    if snapshot is None:
        return "(no world fit yet)", "(no world fit yet)"
    facts = _Facts(store, snapshot, clock)
    per_own = min(350, max_chars // (2 * max(1, len(own))))
    mine = []
    for c in own:
        try:
            mine.append(f"[{c}] " + track_record(ctx, store, snapshot, c, facts=facts, max_chars=per_own - len(c) - 3))
        except Exception as exc:  # noqa: BLE001
            mine.append(f"[{c}] (record unavailable: {type(exc).__name__})")
    own_txt = "\n".join(mine)
    head = "scorecard (component: headline; bias knob meta.<c>_bias)"
    others = [c for c in COMPONENTS if c not in own]
    room = max_chars - len(own_txt) - len(head) - 2
    width = max(40, room // max(1, len(others)) - 1)
    lines = [head]
    for c in others:
        try:
            r = resolve(ctx, store, snapshot, c, facts)
            text = r["headline"] if r["n"] else "no decisions yet"
        except Exception as exc:  # noqa: BLE001
            text = f"(error: {type(exc).__name__})"
        b = bias(ctx.cfg, c)
        lines.append(_one(f"{c}: {text}" + (f" [bias {b:+.2f}]" if abs(b) >= 0.25 else ""), width))
    card = "\n".join(lines)
    while len(card) + len(own_txt) + 1 > max_chars and len(lines) > 1:     # never over budget
        lines.pop()
        card = "\n".join(lines)
    return card, own_txt


# ------------------------------------------------------------------ CONSOLIDATION: the versioned playbook
K_PLAYBOOK = "meta.playbook"          # {version, ts, reviews_at, rules: [{component, rule}], retired: [...]}
PLAYBOOK_MAX_RULES = 10
RETIRED_KEEP = 30
_NUM = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?%?(?![\w])")
CONSOLIDATE_SCHEMA: dict[str, Any] = {
    "type": "object", "required": ["rules"],
    "properties": {"rules": {"type": "array", "items": {
        "type": "object", "required": ["component", "rule", "evidence"],
        "properties": {"component": {"type": "string", "enum": list(COMPONENTS)},
                       "rule": {"type": "string", "minLength": 1},
                       "evidence": {"type": "string", "minLength": 1}}}}}}


def playbook(store: Any) -> dict[str, Any]:
    return store.get_state(K_PLAYBOOK) or {}


def _norm(tok: str) -> str:
    t = tok.lstrip("+-").rstrip("%")
    return t.rstrip("0").rstrip(".") if "." in t else t


def numbers(text: str) -> set[str]:
    """The numbers in ``text``, normalised (sign, trailing % and trailing zeros dropped)."""
    return {_norm(m) for m in _NUM.findall(text or "")}


def check_rule(rule: Mapping[str, Any], evidence_corpus: str) -> str:
    """'' when the rule cites computed evidence: at least one number, and EVERY number in the rule and its
    evidence appears in the records it was condensed from; else why it is dropped."""
    text = f"{rule.get('rule', '')} {rule.get('evidence', '')}"
    got = numbers(text)
    if not got:
        return "cites no number from the records"
    bad = sorted(got - numbers(evidence_corpus))
    return f"invented number(s) {bad[:4]}" if bad else ""


def consolidation_due(store: Any, cfg: Any, fp_trigger: str) -> bool:
    """With every first-principles review, or every ``meta.consolidate_every`` reviews, since the last one."""
    pb, revs = playbook(store), store.reviews()
    last_ts, last_n = float(pb.get("ts", -1.0)), int(pb.get("reviews_at", 0))
    if len(revs) <= last_n:
        return False
    fp = any(rv.trigger == fp_trigger and (rv.created_ts or 0.0) > last_ts for rv in revs)
    every = int(getattr(cfg.meta, "consolidate_every", 0) or 0)
    return fp or (every > 0 and len(revs) - last_n >= every)


def consolidation_input(ctx: Any, store: Any, snapshot: Any) -> str:
    """Every component's RAW record + ideation lessons + recent review findings (the numbers a rule may cite)."""
    from loop.agents import briefs
    facts = _Facts(store, snapshot)
    recs = []
    for c in COMPONENTS:
        try:
            recs.append(f"### {c}\n" + raw_record(ctx, store, snapshot, c, examples=3, facts=facts))
        except Exception as exc:  # noqa: BLE001
            recs.append(f"### {c}\n(unavailable: {type(exc).__name__})")
    revs = current_reviews(store)[-3:]                # never findings built on since-corrected estimates
    return "\n\n".join([
        "## Component records (computed)", *recs,
        "## Ideation lessons (computed)", briefs.optional_block("ideation_lessons", store, snapshot) or "(none)",
        "## Recent review findings", "\n".join(f"- [{rv.id}] {f[:300]}" for rv in revs for f in rv.findings[:6])
        or "(none)"])


def consolidate(ctx: Any, llm: Any, store: Any, snapshot: Any, *, now: float) -> dict[str, Any]:
    """ONE Opus call (role 'review') condenses the records into <= PLAYBOOK_MAX_RULES one-line rules, each
    citing its computed evidence. Rules whose numbers are not in the input are dropped (logged); the previous
    rules that did not survive are RETIRED (kept in state, never shown in prompts). Returns the new playbook
    (the driver persists it with :func:`save_playbook`)."""
    from loop.agents import briefs
    evidence = consolidation_input(ctx, store, snapshot)
    old = playbook(store)
    current = "\n".join(f"- [{r['component']}] {r['rule']} ({r.get('evidence', '')})" for r in old.get("rules", [])) \
        or "(none: this is the first playbook)"
    prompt = briefs.render("consolidate", evidence=evidence, current=current, max_rules=PLAYBOOK_MAX_RULES,
                           components=", ".join(COMPONENTS))
    out = llm.complete_json("review", system="You condense measured track records into a short playbook. "
                                             "Copy numbers exactly from the records; never invent or round them.",
                            prompt=prompt, schema=CONSOLIDATE_SCHEMA)
    rules, dropped, seen = [], [], set()
    for r in out.get("rules") or []:
        rule = {"component": r["component"], "rule": _one(r["rule"], 200), "evidence": _one(r["evidence"], 160)}
        key = (rule["component"], rule["rule"].lower())
        why = check_rule(rule, evidence) or ("duplicate" if key in seen else "")
        if why:
            dropped.append({**rule, "why": why})
            continue
        seen.add(key)
        rules.append(rule)
        if len(rules) >= PLAYBOOK_MAX_RULES:
            break
    kept = {x["rule"].lower() for x in rules}
    version = int(old.get("version", 0)) + 1
    retired = list(old.get("retired", [])) + [{**x, "retired_in": version} for x in old.get("rules", [])
                                              if x["rule"].lower() not in kept]
    return {"version": version, "ts": now, "reviews_at": len(store.reviews()), "rules": rules,
            "retired": retired[-RETIRED_KEEP:], "dropped": dropped}


def save_playbook(store: Any, pb: Mapping[str, Any]) -> None:
    """The current playbook (prompts read it) and its immutable version ``meta.playbook.v<N>``."""
    store.set_state(f"{K_PLAYBOOK}.v{pb['version']}", dict(pb))
    store.set_state(K_PLAYBOOK, dict(pb))
    for d in pb.get("dropped", []):
        store.log("playbook_rule_dropped", version=pb["version"], component=d["component"], rule=d["rule"][:200],
                  why=d["why"])


def playbook_lines(store: Any) -> list[str]:
    """``loop status`` / dashboard: the current playbook, one line per rule (retired rules are not shown)."""
    pb = playbook(store)
    if not pb.get("version"):
        return []
    return [f"playbook v{pb['version']} ({len(pb.get('rules', []))} rules, {len(pb.get('retired', []))} retired)"] + [
        f"  [{r['component']}] {r['rule']} ({r.get('evidence', '')})" for r in pb.get("rules", [])]


# ------------------------------------------------------------------ MEASUREMENT EPOCH: corrected estimates
_NODE_ID = re.compile(r"\bn\d{4}\b")


def bump_epoch(store: Any, reason: str, *, now: float) -> int:
    """Record that the world model's estimates were CORRECTED (a measurement fix): the next tick's
    :func:`on_epoch` withdraws what was reasoned on the old numbers."""
    e = store.get_state(K_EPOCH) or {}
    n = int(e.get("epoch", 0)) + 1
    store.set_state(K_EPOCH, {"epoch": n, "ts": now, "reason": reason[:500], "handled": int(e.get("handled", 0))})
    store.log("measurement_epoch", epoch=n, reason=reason[:500])
    return n


def retire_playbook(store: Any, now: float, why: str, note: str = "") -> str:
    """Retire every playbook rule (the next consolidation regenerates them from the computed records; meanwhile
    prompts see the live records); '' when there was none."""
    pb = playbook(store)
    if not pb.get("rules"):
        return ""
    v = int(pb.get("version", 0)) + 1
    save_playbook(store, {"version": v, "ts": now, "reviews_at": len(store.reviews()), "rules": [],
                          "retired": (list(pb.get("retired", [])) + [{**r, "retired_in": v} for r in pb["rules"]])
                          [-RETIRED_KEEP:], "dropped": [], "invalidated": (note or why)[:300]})
    return f"playbook v{v}: {len(pb['rules'])} rule(s) invalidated ({why})"


def on_epoch(ctx: Any, store: Any, snapshot: Any, *, now: float) -> list[str]:
    """Once per epoch bump: the playbook is invalidated (its rules retired; the next consolidation regenerates it
    from computed records, after the first-principles review the driver starts). Every self-tune override expires
    at the bump on its own (loop.selftune.epoch_ts). Reviews from before the bump stop steering
    (``current_reviews``). [] when nothing is due."""
    e = store.get_state(K_EPOCH) or {}
    if int(e.get("epoch", 0)) <= int(e.get("handled", 0)):
        return []
    acts = [x for x in [retire_playbook(store, now, f"measurement epoch {e['epoch']}",
                                        f"epoch {e['epoch']}: {e.get('reason', '')}")] if x]
    store.set_state(K_EPOCH, {**e, "handled": int(e["epoch"])})
    return acts


_FELL = (NodeStatus.NOT_CONFIRMED, NodeStatus.SCREENED_OUT, NodeStatus.VETOED)
_WINNING = {s.value for s in (NodeStatus.PROMOTING, NodeStatus.CONFIRMING, NodeStatus.CONFIRMED)}
_CHECKED: set[tuple[str, tuple[str, ...]]] = set()


def mark_stale(ctx: Any, store: Any, snapshot: Any, *, now: float) -> list[str]:
    """A META hypothesis resting on a node being a winner -- a node its claim / falsification / predictions cite
    that was promoting / confirming / confirmed when the hypothesis was made -- goes STALE once that node fell:
    closed not_confirmed / screened_out / vetoed, or its OWN increment is now under 1 sigma (live: rv-29's m1 / m3
    on a node / a node as "bar-clearers", both since not_confirmed). Like the epoch rule, a stale meta stops steering
    (no probes, no crossover link, not among the open metas the researchers see) and the next review sees it
    marked. Each is marked once (K_STALE_META)."""
    if snapshot is None:
        return []
    book = dict(store.get_state(K_STALE_META) or {})
    sigma = float(snapshot.sigma.get(PANEL, ctx.cfg.panel_sd))

    def fell(nid: str) -> bool:
        n, p = store.get_node(nid), snapshot.nodes.get(nid)
        return n is not None and (n.status in _FELL or (p is not None and p.increment.mean < sigma))
    cited = {h.id: (h, [i for i in sorted(set(_NODE_ID.findall(" ".join((h.claim, h.falsification, json.dumps(
        h.predictions or {}, default=str)))))) if fell(i)]) for h in store.hypotheses(kind=HypothesisKind.META)
        if h.id not in book}
    if not (cited := {k: v for k, v in cited.items() if v[1] and (k, tuple(v[1])) not in _CHECKED}):
        return []
    _CHECKED.update((k, tuple(v[1])) for k, v in cited.items())     # each fallen set is judged once per process
    hist: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for e in store.events("node_transition"):
        hist[e.get("node_id", "")].append((float(e["ts"]), str(e.get("to", ""))))
    was_winning = lambda nid, ts: (st := [to for t, to in hist[nid] if t <= ts]) and st[-1] in _WINNING  # noqa: E731
    acts = []
    for hid, (h, ids) in cited.items():
        if nodes := [i for i in ids if was_winning(i, h.created_ts or 0.0)]:
            book[hid] = {"nodes": nodes, "ts": now}
            store.log("meta_stale", hypothesis_id=hid, nodes=nodes)
            acts.append(f"meta-hypothesis {hid} STALE: its evidence {', '.join(nodes)} fell (not_confirmed / "
                        "screened_out / vetoed or own increment < 1 sigma); it stops steering")
    if acts:
        store.set_state(K_STALE_META, book)
    return acts
