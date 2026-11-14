"""loop.core.ideation — what a research session is asked, and what is learned from its answer (docs/loop-core-v3.md §6).

Audit 10-12 (the reason for each piece): every real gain came from an LLM idea (explore a node, crossovers a node /
a node); claimed sizes were uncalibrated (ρ −0.09); 41% of late ideas were near-repeats, mostly of closed nodes; 2.6%
of falsification clauses named a measurable signal; nobody targeted classification (weight .30, the only FM gap) or
the regressed axes; 23 research notes in all; the plateau was never diagnosed.

- :func:`view`        one computed picture per tick: each node's own-increment posterior (composite, :mod:`model`),
                      per-axis levels, the replayed test judgements (:mod:`status`), the incumbent
- :func:`headroom`    per axis: incumbent vs the bar node (a node; root if no bar) vs the best other node vs external FM references (task README)
- :func:`engagement`  did the declared ``mechanism_prediction.diagnostic`` move (vs the parent, z > 1 in the declared
                      direction; z < -1: reversed) once the node has a done, diagnosed seed
- :func:`plateau`     no PASS for > 2x the median GPU-h between past PASSes -> one plateau review (``plateau.md``)
- :func:`repeats`     anti-repeat: embedding neighbours among nodes with a negative posterior (intake rule
                      ``repeat_of_negative``; one retry with ``hypothesis.differs_from`` naming them)
- :func:`credit`      per source (lane / brief / model / prompt version; or any key, e.g. a fork-and-flush branch):
                      test-block gains per GPU-h, nothing else
- :class:`Ideation`   the tick's ``ideate`` / ``intake`` hooks, ``brief_for(lane, data, post)`` and ``fork_round``
                      (fork-and-flush, :mod:`loop.core.fnf`: K flushed branches from the incumbent)

Brief kinds: explore / crossover (stack a positive node onto the incumbent) / repair (one axis) / plateau / literature /
program (10-13: a multi-component moonshot -- the researcher's moonshot turn / time caps and sizing, several families
allowed -- for ``ideation.program_share`` of the explore-kind sessions in target mode). In target mode the decisive
size is ``ideation.claim_gap_frac`` x the gap, not the test's MDE (10-13: 43/58 sessions claimed exactly the MDE asked
for). Every brief but a flushed branch carries the computed "untested levers & deficits" (:mod:`loop.core.levers`).
The lane comes from acquisition's ``idea`` actions; nothing here names an idea."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
import re
import statistics
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from loop import diagnostics as DG
from loop.agents import notes as NT
from loop.agents.briefs import structural_block  # v3-c5 I4: program briefs
from loop.contracts import Candidate, CandidateRequest, Node, NodeStatus, Operator, Strategy
from loop import project as P
from loop.core import model as M
from loop.core import status as ST
from loop.core.data import AXES, OFFICIAL_WEIGHTS, ROOT, Ledger
from loop.core.seeds import SeedBlock, reconstruct
from loop.select.embed import HashingEmbedder, changed_lines, cosine

logger = logging.getLogger(__name__)

#: v3-c6i: ``kuhn`` (a plateau-triggered paradigm-shift conjecture) replaces ``plateau`` for new briefs (``plateau`` stays
#: a kind of the record: its nodes' yields); ``readout`` (eval-only readout changes) while the features show distortion
KINDS = ("explore", "crossover", "repair", "plateau", "literature", "program", "kuhn", "readout", "bold")
LANE_KIND = {"explore": "explore", "crossover": "crossover", "exploit": "repair", "factorial": "explore"}
#: a program is a MOONSHOT request: the researcher's moonshot turn / time caps and sizing, research stage, and intake
#: lets its config keys span families
KIND_REQUEST = {"explore": (Strategy.EXPLORE, Operator.DRAFT), "literature": (Strategy.EXPLORE, Operator.DRAFT),
                "plateau": (Strategy.EXPLORE, Operator.DRAFT), "repair": (Strategy.EXPLOIT, Operator.DRAFT),
                "crossover": (Strategy.CROSSOVER, Operator.CROSSOVER), "program": (Strategy.MOONSHOT, Operator.DRAFT),
                "kuhn": (Strategy.MOONSHOT, Operator.DRAFT), "readout": (Strategy.EXPLORE, Operator.DRAFT),
                "bold": (Strategy.MOONSHOT, Operator.DRAFT)}      # v3-c8b: the bold structural lane (loop.core.bold)
#: the kinds an explore-lane slot may get -- what fork-and-flush assigns to its branches (10-13: assignment keyed on
#: the acquisition LANE missed the 45/65 explore briefs bought by the factorial lane)
EXPLORE_KINDS = ("explore", "literature", "program")
K_PROGRAM = "ideation.program"        # {n: explore-kind sessions started, p: of which programs} since the deploy
#: the ideation TARGET / GUARD axes are read at call time from the project registry: P.target() (nanopath cls:
#: weight .30, reliable, the only FM gap) and P.guard() (nanopath rob; None = no guard axis). AXIS_NAMES: live view
AXIS_NAMES = P.AXIS_NAMES
LITERATURE_EVERY = 3                  # every 3rd explore-lane session reads papers first
TAU_REP = 0.70                        # cosine of claim + changed lines to a negative node = a repeat
PLATEAU_FACTOR = 2.0
BRIEF_TTL_X = 4.0                    # a rendered brief waiting for a free slot is re-planned (view moved) after this x brief_ttl_s
PLATEAU_IDLE_H = 72.0                # ... or no PASS for this many hours of ledger time ...
PLATEAU_IDLE_GPU_X = 0.5             # ... with at least this x the median GPU-h between passes spent since (not a stopped loop)
REVIEW_EVERY_H = 6.0                  # a plateau review at most this often (re-asked once its hypotheses are served)
MIN_LEVEL_SEEDS = 3
ENGAGED_Z = 1.0
MEASURED = ("engaged", "did not engage", "reversed")     # engagement verdicts (the rest: pending / unmeasurable)
DIFFERS = "[differs from:"
K_NEGATIVE = "ideation.negative"      # {node: m}: the anti-repeat set (written each tick, read by intake)
K_KNOB_OUTAGE = "bold.knob_outage"   # ts of the last hold that ended on the AST fallback (0 / absent: the judge answered since)
KNOB_RETRY_S = 120.0     # v3 review M3: a held bold node's judge retry period
K_PLATEAU = "ideation.plateau"        # the last plateau review {since, ts, diagnosis, hypotheses, served}
K_ENGAGED = "ideation.engaged"        # {node: status}: engagement verdicts already logged
K_NEXT = "ideation.next_request"
K_KIND_W = "ideation.kind_weights"    # {kind: {n, mean, yield, w}}: the measured brief-kind yields (A3)
K_CLAIMS = "ideation.claims"          # {request id: {family, tag, ts}}: the mechanism an in-flight session's check named
K_PERSONA = "ideation.persona_seq"    # v3-c6i: the persona lens round-robin counter
K_READOUT = "ideation.readout"        # v3-c6i: {incumbent: readout sessions briefed}
K_OPPROG = "ideation.operator_programs"   # v3-c6i: {program id: {ts, kind}}: operator-seeded programs briefed
REQUEST_PREFIX = "req-v3-"            # marks a v3 session for the researcher (schema, prompt)
REPAIR_PREFIX = REQUEST_PREFIX + "fix-"  # req-v3-fix-<node>-<try>: a repair of an ill-conditioned probe crash (A4)
#: a request brief_kind -> the kind its node is tagged / credited with (``brief=<kind>``: the kind bandit's units,
#: ``loop.core.alloc``): c7a's probe-ill-conditioning repair session (brief_kind repair_probe) is a REPAIR
BRIEF_ALIAS = {"repair_probe": "repair"}
PORT_PREFIX = REQUEST_PREFIX + "port-"   # req-v3-port-<old id>: the one session porting a node whose graft conflicted
#: req-v3-rxover-<new incumbent>-<candidate>: the one session crossing a re-root port candidate skipped as the new
#: incumbent's family (tick._reroot_cross) into the new incumbent
REROOT_XOVER_PREFIX = REQUEST_PREFIX + "rxover-"


def cluster_exempt(req: Any) -> bool:
    """A repair (of a probe ill-conditioning or a failed idea) or a port session: it re-implements a measured node's
    mechanism on purpose, so the saturated-cluster cap (``alloc.cluster_rule``) never rejects it (v3 review K4 / K10)."""
    rid, kind = str(getattr(req, "id", "") or ""), str(getattr(req, "brief_kind", "") or "")
    return rid.startswith((REPAIR_PREFIX, PORT_PREFIX, REROOT_XOVER_PREFIX)) or kind in ("repair", "repair_probe", "port") \
        or str(getattr(req, "arm", "") or "").startswith("program_arm:")    # v3-c10: an operator arm is run on purpose


PROMPTS = Path(__file__).resolve().parents[1] / "prompts"


# ---------------------------------------------------------------------------------------------- the computed view
@dataclass(frozen=True)
class Belief:
    m: float                                  # composite posterior of the node's own increment over its parent
    v: float
    n: int


@dataclass(frozen=True)
class View:
    led: Ledger
    noise: M.Noise
    weights: Mapping[str, float]
    beliefs: Mapping[str, Belief]
    levels: Mapping[str, Mapping[str, tuple[float, int]]]     # node -> axis -> (mean over its seeds, seeds)
    judged: tuple[ST.Judged, ...]
    incumbent: str
    post: Any = None                                          # the core posterior it was read from

    def mde(self) -> float:
        """The decisive size: ``2σ_C/√5``, the smallest composite gain a 5-seed test resolves."""
        return 2 * self.noise.sd(M.COMPOSITE) / math.sqrt(5)


def _view_token(v: View) -> tuple:
    """What identifies a ledger view for the per-view caches (``id(v)`` is reused once a view is freed): the
    ledger's sizes, its newest run / observation / node timestamps, the incumbent and the weights."""
    led = v.led
    return (len(led.obs), len(led.runs), len(led.nodes), max((o.ts for o in led.obs), default=0.0),
            max((n.created_ts for n in led.nodes.values()), default=0.0), len(led.carriers), len(led.incumbents),
            len(led.audits), len(led.draws), v.incumbent, getattr(led, "epoch", ""), tuple(sorted(v.weights.items())),
            tuple(sorted((n, round(b.m, 9)) for n, b in v.beliefs.items())))


def view(led: Ledger, blocks: Sequence[SeedBlock], post: Any = None) -> View:
    """The CORE posterior (``acquire.Posterior``: screening data, open test blocks hidden, base epochs, ρ by change
    size) as ideation reads it -- the tick's own when given; levels from every run; the incumbent as ``status``
    derives it (after a base change: its port)."""
    from loop.core.acquire import Posterior
    post = post if isinstance(post, Posterior) else Posterior(led, blocks)
    beliefs = {n: Belief(m, v, k) for n, (m, v, k) in post.beliefs().items()}
    levels = {}
    for n, obs in led.by_node().items():
        levels[n] = {a: (statistics.fmean(ys.values()), len(ys)) for a in AXES if (ys := M.seed_means(obs, a))}
    judged = tuple(ST.replay(led, blocks, post.history()))
    inc = next((j.node for j in reversed(judged) if j.switched), ROOT)
    return View(led, post.noise, post.weights, beliefs, levels, judged, led.port(inc) or inc, post)


def _laps(tick: Any) -> Any:
    """v3-c6a: ``lap(part)`` adds the seconds since the previous lap to ``tick.phase_s["ideate.<part>"]`` (through
    ``tick._time``; a no-op for a tick without it) -- where ideate's time goes (view / record / register / engagement /
    plateau / fnf / settle), next to the store's lock wait (``store.lock_wait``)."""
    add, t = getattr(tick, "_time", None), [time.monotonic()]

    def lap(part: str) -> None:
        if callable(add):
            add(f"ideate.{part}", t[0])
        t[0] = time.monotonic()
    return lap


def blocks_of(store: Any, led: Ledger) -> list[SeedBlock]:
    return list(store.seed_blocks()) or reconstruct(led)


# ---------------------------------------------------------------------------------------------- headroom
@dataclass(frozen=True)
class Headroom:
    axis: str
    weight: float                     # official weight
    sigma: float                      # one run's sd on this axis
    inc: float
    inc_n: int
    vs_root: float                    # incumbent - the reference (a bar node outside its lineage; else 0: itself)
    best: float | None                # the best other node with >= MIN_LEVEL_SEEDS seeds
    best_node: str
    fm: float | None                  # the best external FM reference (task README), None if unknown
    fm_name: str
    ref: str = "root"                 # the reference node the ``vs_root`` column is measured against (the incumbent
                                      # itself when the bar / root is one of its ancestors)

    @property
    def gap(self) -> float:
        """Attainable headroom inside this setup: the best other node's lead, or the loss against root."""
        return max(0.0, (self.best or self.inc) - self.inc, -self.vs_root)


#: README baselines never shown as a reference: operator ban on GenBio (no GenBio data, models or derivatives)
EXCLUDED_FM = ("genbio",)


def fm_reference(readme: str | Path) -> dict[str, tuple[float, str]]:
    """axis -> (best value, model) over the pathology baselines of the README's ``### Baselines`` table (DINOv2
    rows left out: a non-pathology init, not a reference; :data:`EXCLUDED_FM` rows left out: banned). {} if the
    table is absent."""
    try:
        text = Path(readme).read_text()
    except OSError:
        return {}
    part = text.split("### Baselines", 1)[1] if "### Baselines" in text else ""
    rows = [[c.strip().strip("*") for c in l.strip().strip("|").split("|")] for l in part.splitlines()
            if l.startswith("|") and not set(l) <= set("|-: ")]
    if not rows:
        return {}
    head = [h.lower() for h in rows[0]]
    col = {a: head.index(name) for a, name in AXIS_NAMES.items() if name in head}
    out: dict[str, tuple[float, str]] = {}
    for r in rows[1:]:
        name = r[1] if len(r) > 1 else ""
        if name.lower().startswith("dinov2") or len(r) < len(head) or any(
                x in " ".join(r[1:3]).lower() for x in EXCLUDED_FM):
            continue
        for a, i in col.items():
            try:
                x = float(r[i])
            except ValueError:
                continue
            if a not in out or x > out[a][0]:
                out[a] = (x, name)
    return out


def headroom(v: View, fm: Mapping[str, tuple[float, str]] | None = None, bar: str = "") -> list[Headroom]:
    """``bar``: the external bar node (``labless.bar_node``, a node): the incumbent's level is shown against ITS level
    (operator 2026-10-12: nothing against the old version); root only when no bar is configured / measurable."""
    inc = v.levels.get(v.incumbent, {})
    ref_name = bar if bar and v.levels.get(bar) else ROOT
    # operator 11-10 ("every improvement against the current incumbent"): a reference the incumbent descends from (the
    # root, an earlier incumbent, a bar node it has replaced) is no rival -- the column is then vs the incumbent itself
    # (0) and adds no 'loss' to the gap; only a bar node OUTSIDE the incumbent's lineage stays a reference
    from loop.leader import lineage
    if ref_name == ROOT or ref_name in lineage(v.led, v.incumbent):
        ref_name = v.incumbent
    root = v.levels.get(ref_name, {})
    out = []
    for a in AXES:
        if a not in inc:
            continue
        others = [(lv[a][0], n) for n, lv in v.levels.items()
                  if n != v.incumbent and a in lv and lv[a][1] >= MIN_LEVEL_SEEDS]
        best = max(others, default=(None, ""))
        ref = (fm or {}).get(a)
        out.append(Headroom(a, OFFICIAL_WEIGHTS[a], v.noise.sd(a), inc[a][0], inc[a][1],
                            inc[a][0] - root[a][0] if a in root else 0.0, best[0], best[1],
                            ref[0] if ref else None, ref[1] if ref else "", ref_name))
    return out


def repair_axis(rows: Sequence[Headroom], weights: Mapping[str, float]) -> str:
    """The axis whose attainable headroom is worth most in the screening composite (classification by default)."""
    score = {r.axis: weights.get(r.axis, 0.0) * r.gap for r in rows}
    best = max(score, key=score.get, default=P.target())
    return best if score.get(best, 0.0) > 0 else P.target()


def repair_axes(rows: Sequence[Headroom], weights: Mapping[str, float]) -> list[str]:
    """The axes with attainable headroom, worth most first ([classification] when none)."""
    score = {r.axis: weights.get(r.axis, 0.0) * r.gap for r in rows}
    return [a for a in sorted(score, key=lambda a: -score[a]) if score[a] > 0] or [P.target()]


def rotate_axis(ranked: Sequence[str], held: Sequence[str]) -> str:
    """The repair axis of a new session: the best-ranked axis held by the fewest in-flight repair sessions (10-13 09Z
    MA: 4 concurrent repair sessions all worked on classification while mutation / robustness / slide had headroom)."""
    from collections import Counter
    n = Counter(held)
    return min(ranked, key=lambda a: (n[a], list(ranked).index(a)))


def headroom_text(rows: Sequence[Headroom], incumbent: str) -> str:
    def sig(x: float | None, r: Headroom) -> str:
        return "-" if x is None else f"{x:.4f} ({(x - r.inc) / r.sigma:+.1f}σ)"
    lines = [f"axis (weight) | incumbent {incumbent} (seeds) | vs {rows[0].ref if rows else 'root'} | best other node (>= {MIN_LEVEL_SEEDS} seeds) "
             "| best external FM"]
    for r in rows:
        tag = " TARGET" if r.axis == P.target() else " GUARD" if r.axis == P.guard() else ""
        lines.append(f"[H-{r.axis}] {AXIS_NAMES[r.axis]} ({r.weight:.3f}){tag} | {r.inc:.4f} (n={r.inc_n}) | "
                     f"{r.vs_root:+.4f} ({r.vs_root / r.sigma:+.1f}σ) | "
                     f"{sig(r.best, r) + ' ' + r.best_node if r.best is not None else '-'} | "
                     f"{sig(r.fm, r) + ' ' + r.fm_name if r.fm is not None else '-'}")
    return "\n".join(lines + [P.rule("headroom_footer",
                                     "σ = one run's sd per axis. TARGET: classification carries .30 of the panel and "
                                     "is the only axis where external FMs lead clearly. GUARD: a change must not cost "
                                     "robustness. The best other node is the max over many nodes (optimistic).")])


# ---------------------------------------------------------------------------------------------- engagement
@dataclass(frozen=True)
class Engagement:
    node: str
    diagnostic: str
    direction: int
    status: str                       # engaged | did not engage | reversed | pending | unmeasurable
    z: float | None = None            # along the prediction: (mean_c - mean_p) / se x direction
    n: int = 0
    seed: int | None = None           # the seed of its first diagnosed run (what a one-seed verdict was decided on)


def prediction(store: Any, node_id: str) -> dict[str, Any] | None:
    """The node's declared mechanism_prediction: the v3 form (``diagnostic``, ``direction`` ±1), or an older free-text
    ``quantity`` that names a diagnostics quantity (``direction`` up / down). None if it names no measurable one."""
    plan = store.get_state(f"ideation:{node_id}") or {}
    mp = plan.get("mechanism_prediction") or (plan.get("first_principles") or {}).get("mechanism_prediction") or {}
    if not isinstance(mp, Mapping):
        return None
    if mp.get("diagnostic"):
        return dict(mp)
    text = str(mp.get("quantity") or "")
    name = next((q for q in sorted(DG.quantities(), key=len, reverse=True) if re.search(rf"\b{q}\b", text)), None)
    return {"diagnostic": name, "direction": -1 if mp.get("direction") == "down" else 1} if name else None


def diag_values(rows: Sequence[Mapping[str, Any]], q: str) -> dict[str, list[float]]:
    """node -> the quantity over its diagnosed runs."""
    out: dict[str, list[float]] = {}
    for r in rows:
        if (x := DG.quantity(r["features"], q)) is not None and math.isfinite(x):
            out.setdefault(r["node_id"], []).append(x)
    return out


def pooled_sd(vals: Mapping[str, Sequence[float]]) -> float | None:
    ss = df = 0.0
    for xs in vals.values():
        if len(xs) >= 2:
            m = statistics.fmean(xs)
            ss, df = ss + sum((x - m) ** 2 for x in xs), df + len(xs) - 1
    return math.sqrt(ss / df) if df and ss > 0 else None


def engagement(rows: Sequence[Mapping[str, Any]], node: str, parent: str, diagnostic: str, direction: int) -> Engagement:
    """``z = direction · (mean_c − mean_p) / (s √(1/n_c + 1/n_p))``, signed ALONG the prediction; engaged when
    z > 1, reversed when z < −1 (it moved the other way), else did not engage. Decided from the first done, diagnosed
    seed. s is the pair's own pooled within-node sd (the parent usually has many runs); the quantity's sd pooled over
    every node stands in only when the pair has none. That global sd is NOT the run-to-run noise of a quantity whose
    scale differs across nodes: on the 10-08 ledger a few exploded nodes (gn_late_over_clip 178, 369 vs ~2.3) put it at
    2.5 against the parent's .09, so a node / a node (down 8 / 13 sd, as predicted) logged "did not engage", and a node
    (up 2,200 sd against a predicted fall) logged "did not engage (z +96.3)"."""
    d = 1 if direction > 0 else -1
    vals = diag_values(rows, diagnostic)
    c, p = vals.get(node, []), vals.get(parent, [])
    if not c:
        return Engagement(node, diagnostic, d, "pending")
    s = pooled_sd({node: c, parent: p}) or pooled_sd(vals) if p else None
    if s is None:
        return Engagement(node, diagnostic, d, "unmeasurable", n=len(c))
    z = d * (statistics.fmean(c) - statistics.fmean(p)) / (s * math.sqrt(1 / len(c) + 1 / len(p)))
    status = "engaged" if z > ENGAGED_Z else "reversed" if z < -ENGAGED_Z else "did not engage"
    seed = next((int(r["seed"]) for r in rows if r.get("node_id") == node and r.get("seed") is not None
                 and (x := DG.quantity(r["features"], diagnostic)) is not None and math.isfinite(x)), None)
    return Engagement(node, diagnostic, d, status, z, len(c), seed)


def engagements(store: Any, led: Ledger, rows: Sequence[Mapping[str, Any]] | None = None) -> list[Engagement]:
    rows = store.diagnostics() if rows is None else rows
    out = []
    for nid, node in led.nodes.items():
        if (mp := prediction(store, nid)) is not None:
            out.append(engagement(rows, nid, node.parent or ROOT, str(mp["diagnostic"]), int(mp.get("direction") or 1)))
    return out


# ---------------------------------------------------------------------------------------------- plateau
@dataclass(frozen=True)
class Plateau:
    active: bool
    since: float                      # the last PASS (or the first run)
    gpu_h: float                      # spent since
    median: float                     # median GPU-h between past PASSes
    passes: int
    idle_h: float = 0.0               # hours from the last PASS to the newest finished run
    why: str = ""                     # "gpu" (spend past factor x median) | "idle" (no pass for PLATEAU_IDLE_H) | ""


def plateau(led: Ledger, judged: Sequence[ST.Judged], factor: float = PLATEAU_FACTOR) -> Plateau:
    """Active when no PASS has come for more than ``factor`` x the median GPU-h between past PASSes ("gpu"), or -- 10-22 --
    when the best has not changed for PLATEAU_IDLE_H hours of ledger time and at least PLATEAU_IDLE_GPU_X x that median
    has been spent since ("idle"). The spend rule alone stayed off for 9 days of a flat best (1338 GPU-h since the last
    pass against a threshold of 1851: the median is dominated by the long early gaps, and the one late pass, +.0016, reset
    the clock). The idle clock is the newest finished run, not the wall clock: a stopped loop is not a plateau."""
    passes = sorted(j.ts for j in judged if j.switched)
    spent = [(r.done_ts, r.gpu_h) for r in led.runs]
    edges = [-math.inf, *passes]
    spans = [sum(h for t, h in spent if a < t <= b) for a, b in zip(edges, edges[1:])]
    cur = sum(h for t, h in spent if t > edges[-1])
    med = statistics.median(spans) if spans else math.inf
    newest = max((t for t, _ in spent), default=0.0)
    idle = max(0.0, newest - passes[-1]) / 3600.0 if passes else 0.0
    why = "gpu" if spans and cur > factor * med else \
        "idle" if spans and passes and idle >= PLATEAU_IDLE_H and cur >= PLATEAU_IDLE_GPU_X * med else ""
    return Plateau(bool(why), passes[-1] if passes else 0.0, cur, med, len(passes), round(idle, 2), why)


REVIEW_SCHEMA = {"type": "object", "required": ["diagnosis", "hypotheses"], "properties": {
    "diagnosis": {"type": "string", "minLength": 1},
    "hypotheses": {"type": "array", "minItems": 1, "items": {
        "type": "object", "required": ["claim", "evidence", "decisive_experiment", "prediction"], "properties": {
            "claim": {"type": "string"}, "evidence": {"type": "string"}, "decisive_experiment": {"type": "string"},
            "prediction": {"type": "object"}, "knobs": {"type": "array", "items": {"type": "object"}}}}}}}


# ---------------------------------------------------------------------------------------------- computed facts
# Shown in every brief and in the plateau review (fix batch B, 10-12): what the ledger already measured about the
# trade-offs between axes, the regions nobody could measure, what the carrier's lineage has tried, and what the harm
# limit holds back. Computed, never hand-written: no idea is named here.
TRADE_K = 15
COVERAGE_K = 20
PLATEAU_INPUTS = 2          # the plateau review's input version: a stored review on older inputs is redone at tick 0


def lineage(led: Ledger, nid: str, second: bool = False) -> list[str]:
    """``nid`` and its ancestors (a port: then its original's ancestors too), nearest first. ``second``: a crossover's
    second parent too, transitively (the ancestry the alt-base arm excludes)."""
    out, todo = [], [nid]
    while todo:
        cur = todo.pop(0)
        while cur is not None and cur in led.nodes and cur not in out:
            out.append(cur)
            if led.nodes[cur].port_of:
                todo.append(led.nodes[cur].port_of)
            if second and led.nodes[cur].second_parent:
                todo.append(led.nodes[cur].second_parent)
            cur = led.nodes[cur].parent
    return out


def changed(store: Any, led: Ledger, nid: str, width: int = 70) -> str:
    """What the node changed over its parent: its config diff, a wave point's / x*'s levels, a port's original's, else
    its mechanism tag (a code change)."""
    from loop.adapters import flatten
    cd = store.get_state(f"config_diff:{nid}") if hasattr(store, "get_state") else None
    node = led.nodes.get(nid)
    if cd:
        text = ", ".join(f"{k}={v}" for k, v in flatten(cd).items())
    elif node is not None and " design " in f" {node.notes}":
        text = node.notes.split("design ", 1)[1]
    elif node is not None and node.port_of:
        return f"port of {node.port_of}: " + changed(store, led, node.port_of, width)
    else:
        text = _tag(store, nid) if hasattr(store, "get_node") else nid
    return text if len(text) <= width else text[:width - 1] + "…"


def trades(v: View, store: Any, k: int = TRADE_K) -> list[dict[str, Any]]:
    """The ``k`` nodes with the largest |own classification increment| (CUPED over the parent) among the current base's
    nodes and the old-epoch nodes built on the incumbent's lineage, with their robustness / segmentation / panel
    increments: which mechanisms bought classification with another axis, and which did not."""
    from loop.core.acquire import wave_node
    post, led = v.post, v.led
    if post is None:
        return []
    lin = set(lineage(led, v.incumbent))
    rel = [n for n in post.est if not wave_node(led.nodes[n])
           and (led.nodes[n].base == post.epoch or led.nodes[n].parent in lin or led.nodes[n].carrier in lin)]

    def inc(n: str, t: str) -> M.Estimate | None:
        node = led.nodes[n]
        nz, wz = post.fit(node.base, bool(node.port_of))
        return M.cuped(post.by[n], post.by[node.parent], t, nz, wz) if t in nz.sigma2 else None

    tg = P.target()                                   # nanopath: cls
    cls = {n: e for n in rel if (e := inc(n, tg)) is not None}
    out = []
    for n in sorted(cls, key=lambda n: -abs(cls[n].delta))[:k]:
        row = {"node": n, tg: cls[n].delta, f"{tg}_z": cls[n].delta / math.sqrt(cls[n].var) if cls[n].var > 0 else 0.0,
               "n": cls[n].n, "keys": changed(store, led, n)}
        for t in _trade_side():
            e = inc(n, t)
            row[t] = e.delta if e is not None else None
        row[PANEL_T] = post.est[n][M.COMPOSITE].delta       # c6a: the VISIBLE composite (holdout tasks out), never the official panel
        out.append(row)
    return out


PANEL_T = "composite"       # trades() column key: the visible composite (c6a holdout: ideators never see held tasks)


def _trade_side() -> list[str]:
    """trades() side columns: the guard axis and segmentation, those the project has (nanopath: rob, seg); a
    project without a guard axis shows every axis but the target (nanobrain target seg: cls, reg)."""
    if P.guard() is None:
        return [a for a in AXES if a != P.target()]
    return [a for a in (P.guard(), "seg") if a in AXES and a != P.target()]


def trades_text(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return "(no measured node yet)"
    f = lambda x: "-" if x is None else f"{x:+.4f}"    # noqa: E731
    tg, side = P.target(), _trade_side()
    return "\n".join([" | ".join(["node", f"{tg} Δ (z)", *(f"{a} Δ" for a in side), "composite Δ", "seeds",
                                   "changed"])] + [
        " | ".join([f"{r['node']}", f"{r[tg]:+.4f} ({r[f'{tg}_z']:+.1f})", *(f(r.get(a)) for a in side), f(r[PANEL_T]),
                    f"{r['n']}", f"{r['keys']}"]) for r in rows])


def _outcomes(led: Ledger) -> dict[str, tuple[int, int]]:
    """node -> (done runs, crashed runs)."""
    out: dict[str, tuple[int, int]] = {}
    for r in led.runs:
        d, c = out.get(r.node, (0, 0))
        out[r.node] = (d + (r.status == "done"), c + (r.status == "crash"))
    return out


def _configs(store: Any, led: Ledger) -> dict[str, dict[str, Any]]:
    from loop.adapters import flatten
    out = {}
    for nid in led.nodes:
        if (cd := store.get_state(f"config_diff:{nid}")) and isinstance(cd, Mapping):
            out[nid] = flatten(cd)
    return out


def censored(store: Any, led: Ledger, configs: Mapping[str, Mapping[str, Any]] | None = None,
             min_nodes: int = 2) -> list[str]:
    """Config regions every node of which crashed (OOM included) without a done run: UNMEASURED, not negative. A
    numeric key: the values beyond the largest (smallest) value that ever finished; any other key: the values."""
    configs = _configs(store, led) if configs is None else configs
    outc = _outcomes(led)
    per: dict[str, dict[str, list]] = {}
    for nid, cfg in configs.items():
        d, c = outc.get(nid, (0, 0))
        if not d and not c:
            continue
        for k, val in cfg.items():
            rec = per.setdefault(k, {}).setdefault(json.dumps(val, sort_keys=True, default=str), [val, 0, 0])
            rec[1], rec[2] = rec[1] + (d > 0), rec[2] + (d == 0 and c > 0)
    out = []
    for k, vals in sorted(per.items()):
        ok = [v for v, d, _ in vals.values() if d]
        bad = [(v, c) for v, d, c in vals.values() if not d and c]
        if not bad or sum(c for _, c in bad) < min_nodes:
            continue
        if all(_isnum(v) for v, _ in bad) and ok and all(_isnum(v) for v in ok):
            hi, lo = max(ok), min(ok)
            for side, sel, edge in ((">", [x for x in bad if x[0] > hi], hi), ("<", [x for x in bad if x[0] < lo], lo)):
                if sum(c for _, c in sel) >= min_nodes:
                    out.append(f"{k} {side} {edge:g}: tried {sorted({v for v, _ in sel})} on {sum(c for _, c in sel)} "
                               f"nodes, every run crashed / OOM'd -- unmeasured, not negative")
        else:
            out.append(f"{k} in {sorted({json.dumps(v, default=str) for v, _ in bad})}: {sum(c for _, c in bad)} nodes, "
                       "every run crashed / OOM'd -- unmeasured, not negative")
    return out


def _isnum(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def coverage(store: Any, led: Ledger, incumbent: str, configs: Mapping[str, Mapping[str, Any]] | None = None,
             k: int = COVERAGE_K) -> list[str]:
    """Per config key changed by a child of the incumbent's lineage: each value tried (nodes / done seeds), most
    tried keys first -- what the carrier chain has and has not explored."""
    configs = _configs(store, led) if configs is None else configs
    lin = lineage(led, incumbent)
    on = set(lin)
    outc = _outcomes(led)
    per: dict[str, dict[str, list[int]]] = {}
    for n in led.nodes.values():
        if n.parent in on and n.id in configs:
            for key, val in configs[n.id].items():
                rec = per.setdefault(key, {}).setdefault(json.dumps(val, default=str), [0, 0])
                rec[0], rec[1] = rec[0] + 1, rec[1] + outc.get(n.id, (0, 0))[0]
    rows = sorted(per.items(), key=lambda kv: (-sum(r[0] for r in kv[1].values()), kv[0]))[:k]
    head = f"lineage {' > '.join(lin[:6])}{' > …' if len(lin) > 6 else ''}"
    return [head] + [f"{key}: " + ", ".join(f"{v} ({a} nodes / {s} seeds)" for v, (a, s) in
                                            sorted(vals.items(), key=lambda x: -x[1][0])[:6])
                     for key, vals in rows] if rows else []


def harm_lines(v: View, statuses: Mapping[str, Any] | None, harm: Mapping[str, Any] | None = None) -> list[str]:
    """Nodes the harm limit holds back (a positive composite belief, a robustness loss beyond h with P > q) and test
    blocks that FAILED on harm."""
    from loop.core import acquire as A
    out = []
    if v.post is not None:
        st = {n: s for n, s in statuses.items() if isinstance(s, ST.Status)} if statuses else None
        for n, m, rm, rs in A.harm_blocked(v.led, st, v.incumbent, v.post, **dict(harm or {}))[:10]:
            out.append(f"{n}: composite {m:+.4f} vs the incumbent, robustness {rm:+.4f} (sd {rs:.4f}): test blocked")
    for j in v.judged:
        if j.verdict.harm:
            out.append(f"{j.node}: test {j.block} FAILED on harm ({', '.join(j.verdict.harm)}; panel "
                       f"{j.verdict.mean:+.4f} over {j.verdict.n} seeds)")
    return out[-15:]


@dataclass(frozen=True)
class FamilyEI:
    family: str
    nodes: int
    tested: int          # fresh-seed tests (replayed judgements) of its nodes
    m: float             # mean own gain of its measured nodes (the lane prior's mean when none)
    v: float             # prior variance used (x acquire.UNTESTED_VAR_MULT when untested)
    ei: float            # EI over the gap (Student-t, ν = 3)


def distance(v: View) -> tuple[float, float, list[FamilyEI]] | None:
    """Target mode (``acquire.target_on``): (gap, best shrunk candidate in hand, families by EI over the gap) when the
    gap exceeds what the material in hand can deliver (FRESH_SHRINK x the best own gain moved to the incumbent);
    else None. A family's prior: its measured nodes' mean own gain and spread (+ their mean posterior variance), the
    explore-lane prior when none is measured; untested families (no fresh test) get UNTESTED_VAR_MULT x the variance
    -- an unmeasured heavy tail is worth most when the gap is far."""
    from loop.core import acquire as A
    post = v.post
    if post is None or not A.target_on() or (gap := A.gap_of(v.led, v.incumbent)) is None:
        return None
    inc, best = v.incumbent, 0.0
    for n, b in v.beliefs.items():
        node = v.led.nodes.get(n)
        if node is not None and node.base == v.led.epoch and n != inc and inc in post.by and node.parent in post.by:
            best = max(best, b.m + post.shift(n, M.COMPOSITE, inc)[0])
    best *= A.FRESH_SHRINK
    if gap <= best:
        return None
    tested = {j.node for j in v.judged}
    lm, lv, _ = post.lanes.cell((inc, "explore", ""))
    s2, rho = post.noise.sigma2[M.COMPOSITE], post.noise.rho[M.COMPOSITE]
    big_v = s2 * (1 - rho ** 2) / 3
    fams: dict[str, list[str]] = {}
    for n, node in v.led.nodes.items():
        if node.family not in A.GENERIC_FAMILIES and node.parent is not None and not node.port_of:
            fams.setdefault(node.family, []).append(n)
    out = []
    for f, ns in fams.items():
        ms = [v.beliefs[n] for n in ns if n in v.beliefs]
        t = sum(n in tested for n in ns)
        if ms:
            m = statistics.fmean(b.m for b in ms)
            var = (statistics.pvariance([b.m for b in ms]) if len(ms) > 1 else 0.0) + statistics.fmean(b.v for b in ms)
            var = max(var, lv)
        else:
            m, var = lm, lv
        var *= A.UNTESTED_VAR_MULT if t == 0 else 1.0
        out.append(FamilyEI(f, len(ns), t, m, var, A.ei_t(m, var, big_v, gap)))
    out.sort(key=lambda r: -r.ei)
    return gap, best, out


def target_label() -> str:
    """Names the external target in the briefs: Labless's acceptance rule (median of 3 fresh seeds >= leader median +
    margin), the numbers from ``acquire.TARGET`` (never hardcoded)."""
    from loop.core import acquire as A
    t = A.TARGET
    if t is None or t.score is None:
        return "external target"
    return (f"external target (the Labless promotion rule: median of 3 fresh seeds >= {t.score + t.margin:.4f} = leader "
            f"median {t.score:.4f} + {t.margin:g}, needed with P >= {t.p:g}; the gap is the local mean needed minus the "
            "incumbent's level)")


def distance_text(v: View, k: int = 10) -> str:
    """The brief section of :func:`distance` ('' when the target is off or within reach of the material in hand)."""
    from loop.core import acquire as A
    try:
        got = distance(v)
    except (KeyError, ValueError, ZeroDivisionError):
        return ""
    if got is None:
        return ""
    gap, best, rows = got
    s2, rho = v.post.noise.sigma2[M.COMPOSITE], v.post.noise.rho[M.COMPOSITE]
    lm, lv, _ = v.post.lanes.cell((v.incumbent, "explore", ""))
    new = A.ei_t(lm, lv * A.UNTESTED_VAR_MULT, s2 * (1 - rho ** 2) / 3, gap)
    # 10-13 09Z MA: the per-family numbers (mean own gain ~.003, EI) were what sessions sized their claims by; the
    # RANK stays, the numbers go (the per-axis budget against the gap is the size reference now)
    rank = sorted(rows, key=lambda r: -r.ei)
    above = sum(r.ei > new for r in rank)
    lines = [f"{i}. {r.family}: {r.nodes} nodes, {r.tested} fresh-tested"
             + (" (UNTESTED: widened prior)" if r.tested == 0 else "") for i, r in enumerate(rank[:k], 1)]
    return ("## Distance to the target (computed)\n"
            f"The {target_label()} is {gap:+.4f} above the incumbent; the best candidate in hand, shrunk to what fresh "
            "seeds replicate, cannot close it: stacking or tuning what exists does not reach the target. Mechanism "
            "families ranked by expected improvement OVER THE GAP (heavy-tailed prior; families never fresh-tested get "
            f"a widened prior). A family not listed at all ranks after the first {above} at the explore prior:\n"
            + "\n".join(lines))


def facts_text(v: View, store: Any, statuses: Mapping[str, Any] | None = None,
               harm: Mapping[str, Any] | None = None) -> str:
    configs = _configs(store, v.led)
    cen, cov, hb = censored(store, v.led, configs), coverage(store, v.led, v.incumbent, configs), harm_lines(
        v, statuses, harm)
    return "\n\n".join([
        f"### Axis trades (top {TRADE_K} by |own classification increment|; CUPED vs the parent; current base + the "
        f"incumbent's lineage)\n" + trades_text(trades(v, store)),
        "### Crash-censored regions (UNMEASURED, NOT NEGATIVE: a different implementation may fit)\n"
        + ("\n".join(f"- {x}" for x in cen) or "(none)"),
        "### Per-key coverage on the carrier lineage (values its children tried: nodes / done seeds)\n"
        + ("\n".join(f"- {x}" for x in cov) or "(none)"),
        "### " + P.rule("harm_blocked_header", "Harm-blocked (catastrophic robustness loss only; the composite prices every axis)")
        + "\n"
        + ("\n".join(f"- {x}" for x in hb) or "(none)")])


# ---------------------------------------------------------------------------------------------- anti-repeat
_EMB: dict[str, tuple[float, ...]] = {}


def node_vector(store: Any, nid: str, embedder: Any) -> tuple[float, ...] | None:
    """A node's intake embedding (claim + changed lines), computed for nodes stored without one."""
    if nid in _EMB:
        return _EMB[nid]
    node = store.get_node(nid)
    if node is None:
        return None
    vec = node.embedding
    if vec is None:
        h = store.get_hypothesis(node.hypothesis_id or "")
        try:
            diff = Path(node.diff_path).read_text(errors="replace") if node.diff_path else ""
        except OSError:
            diff = ""
        vec = embedder.embed(f"{h.claim if h else node.mechanism_tag}\n{changed_lines(diff)}")
    _EMB[nid] = tuple(vec)
    return _EMB[nid]


def negatives(v: View, statuses: Mapping[str, Any]) -> dict[str, float]:
    """Nodes whose own-increment posterior is negative (incumbent / passed / testing excepted)."""
    keep = {"INCUMBENT", "PASSED", "TESTING"}
    return {n: round(b.m, 6) for n, b in v.beliefs.items() if b.m < 0 and _name(statuses.get(n)) not in keep}


def repeats(store: Any, vec: Sequence[float], negative: Mapping[str, float], tau: float = TAU_REP,
            embedder: Any = None, k: int = 3) -> list[tuple[str, float, float]]:
    """(node, cosine, m) of the negative nodes at cosine >= tau, nearest first."""
    emb = embedder or HashingEmbedder()
    hits = [(n, c, m) for n, m in negative.items()
            if (nv := node_vector(store, n, emb)) is not None and (c := cosine(vec, nv)) >= tau]
    return sorted(hits, key=lambda h: -h[1])[:k]


def repeat_rule(store: Any, candidate: Candidate, vec: Sequence[float], tau: float = TAU_REP) -> tuple[str, str]:
    """intake's anti-repeat rule: ('repeat_of_negative', fix) unless ``differs_from`` names every neighbour."""
    neg = store.get_state(K_NEGATIVE) or {}
    hits = repeats(store, vec, neg, tau) if neg else []
    claim = candidate.hypothesis.claim
    differs = claim.split(DIFFERS, 1)[1] if DIFFERS in claim else ""
    from loop.core.mechanisms import explained
    if not hits or all(explained(differs, n) for n, _, _ in hits):      # v3-c6r: each id named AND explained
        return "", ""
    near = "; ".join(f"{n} (cos {c:.2f}, own gain {m:+.4f}, {_tag(store, n)})" for n, c, m in hits)
    return "repeat_of_negative", (f"this repeats closed idea(s) with a negative measured effect: {near}. Change the "
                                  "mechanism, or set hypothesis.differs_from naming each of these nodes and what is "
                                  "materially different (one retry)")


def _tag(store: Any, nid: str) -> str:
    n = store.get_node(nid)
    return f"{n.family}/{n.mechanism_tag}" if n else "?"


def clusters(store: Any, negative: Mapping[str, float], tau: float = 0.6, top: int = 8) -> list[str]:
    """Leader clustering of the negative nodes (largest first): one line per cluster."""
    emb, groups = HashingEmbedder(), []
    for n in sorted(negative, key=negative.get)[:400]:          # the 400 most negative
        if (vec := node_vector(store, n, emb)) is None:
            continue
        g = next((g for g in groups if cosine(vec, g[0]) >= tau), None)
        (g[1].append(n) if g else groups.append((vec, [n])))
    groups.sort(key=lambda g: -len(g[1]))
    return [f"{len(ns)} nodes, mean own gain {statistics.fmean(negative[n] for n in ns):+.4f}: "
            + ", ".join(f"{n} {_tag(store, n)}" for n in ns[:4]) for _, ns in groups[:top]]


# ---------------------------------------------------------------------------------------------- credit
def source_of(node: Any) -> str:
    """lane / brief / model / prompt version, from the node's notes tags."""
    tags = dict(t.split("=", 1) for t in (node.notes or "").split() if "=" in t)
    return "/".join((node.lane, tags.get("brief", "-"), tags.get("backend", "-"), tags.get("prompt", "-")))


def credit(v: View, key: Callable[[Any], str] = source_of, screen: tuple[float, int] | None = None,
           clade: bool = False) -> dict[str, dict[str, float]]:
    """Per source (``key(node)``; '' = not counted): ``Σ max(0, δ_test) / GPU-h`` -- δ_test the node's last test-block
    judgement (official panel). ``screen`` = (shrink, min seeds): a node WITHOUT a test judgement earns ``shrink x
    max(0, its own-increment posterior)`` once it has ``min seeds`` screen seeds (fork-and-flush rounds close long before
    a branch node reaches a test; ``screen`` is that part of ``gain``). Default: test gains only. ``clade`` (v3-c6i):
    a brief-tagged node's test gain is the best over its clade (itself and its untagged descendants,
    :func:`clade_of`), counted once at the tagged node."""
    gain = {j.node: max(0.0, j.verdict.mean) for j in v.judged}
    if clade:
        roll: dict[str, float] = {}
        for n, g in gain.items():
            r = clade_of(v.led, n)
            roll[r] = max(roll.get(r, 0.0), g)
        gain = roll
    gpu: dict[str, float] = {}
    for r in v.led.runs:
        gpu[r.node] = gpu.get(r.node, 0.0) + r.gpu_h
    out: dict[str, dict[str, float]] = {}
    for nid, node in v.led.nodes.items():
        if nid == ROOT or not (k := key(node)):
            continue
        s = out.setdefault(k, {"nodes": 0, "tested": 0, "gain": 0.0, "gpu_h": 0.0})
        s["nodes"] += 1
        s["tested"] += nid in gain
        s["gain"] += gain.get(nid, 0.0)
        s["gpu_h"] += gpu.get(nid, 0.0)
        if screen is not None and nid not in gain and (b := (getattr(v, "beliefs", None) or {}).get(nid)) \
                is not None and b.n >= screen[1]:
            s["screen"] = s.get("screen", 0.0) + screen[0] * max(0.0, b.m)
            s["gain"] += screen[0] * max(0.0, b.m)
    for s in out.values():
        s["per_gpu_h"] = s["gain"] / s["gpu_h"] if s["gpu_h"] else 0.0
    return out


def clade_of(led: Ledger, nid: str) -> str:
    """The nearest brief-tagged ancestor-or-self of ``nid`` (``nid`` when there is none)."""
    cur, seen = nid, set()
    while cur is not None and cur in led.nodes and cur not in seen:
        seen.add(cur)
        if _brief_tag(led.nodes[cur]):
            return cur
        cur = led.nodes[cur].parent
    return nid


# ---------------------------------------------------------------------------------------------- kind yield (A3)
def _brief_tag(node: Any) -> str:
    got = next((t[6:] for t in (node.notes or "").split() if t.startswith("brief=")), "")
    return BRIEF_ALIAS.get(got, got)


def clade_values(v: View, min_seeds: int = 2) -> dict[str, float]:
    """v3-c6i CLADE CREDIT (HGM 2510.21614): per brief-tagged node of this epoch, the best value in its clade -- the
    node and its untagged descendants (dose ladder points, leave-one-out children, tuned variants) -- each valued as
    its gain over the tagged node's PARENT (the sum of the own increments along the path); a descendant counts with
    >= ``min_seeds`` seeds, the node itself with >= 1. A moonshot is credited with what its tuning found, not with its
    own untuned screen."""
    led = v.led
    epoch = getattr(led, "epoch", "") or ""
    kids: dict[str, list[str]] = {}
    for nid, node in led.nodes.items():
        if node.parent is not None:
            kids.setdefault(node.parent, []).append(nid)
    out: dict[str, float] = {}
    for nid, node in led.nodes.items():
        if nid == ROOT or not _brief_tag(node) or (epoch and node.base != epoch) or \
                (b := v.beliefs.get(nid)) is None or b.n < 1:
            continue
        best, todo = b.m, [(c, b.m) for c in kids.get(nid, ())]
        while todo:
            cur, above = todo.pop()
            cn = led.nodes[cur]
            if _brief_tag(cn) or (epoch and cn.base != epoch):        # another session's clade
                continue
            cb = v.beliefs.get(cur)
            val = above + (cb.m if cb is not None else 0.0)
            if cb is not None and cb.n >= min_seeds:
                best = max(best, val)
            todo += [(c, val) for c in kids.get(cur, ())]
        out[nid] = best
    return out


def kind_yields(v: View, descendants: bool = True, clade: int | None = None) -> dict[str, list[float]]:
    """brief kind -> the own-increment posterior means (``View.beliefs``, >= 1 seed) of THIS epoch's nodes its
    research sessions produced (notes ``brief=<kind>``). ``descendants``: an untagged node (a wave point, a ladder
    dose, a tuned variant) counts for its nearest tagged ancestor's kind -- what the brief led to. A node tagged with
    a non-brief source (port, databank, ...) attributes nothing, nor do its untagged descendants. ``clade`` (min
    seeds; v3-c6i): ONE value per tagged node instead, its clade's best (:func:`clade_values`)."""
    led, memo = v.led, {}
    epoch = getattr(led, "epoch", "") or ""
    if clade is not None:
        out_c: dict[str, list[float]] = {}
        for nid, val in clade_values(v, clade).items():
            if (k := _brief_tag(led.nodes[nid])) in KINDS:
                out_c.setdefault(k, []).append(val)
        return out_c

    def kind_of(nid: str) -> str:
        path, cur, k = [], nid, ""
        while cur is not None and cur in led.nodes and cur not in memo and cur not in path:
            path.append(cur)
            if (t := _brief_tag(led.nodes[cur])) or not descendants:
                k = t if t in KINDS else ""
                cur = None
                break
            cur = led.nodes[cur].parent
        if cur is not None and cur in memo:
            k = memo[cur]
        for p in path:
            memo[p] = k
        return k

    out: dict[str, list[float]] = {}
    for nid, node in led.nodes.items():
        if nid == ROOT or (epoch and node.base != epoch) or (b := v.beliefs.get(nid)) is None or b.n < 1:
            continue
        if k := kind_of(nid):
            out.setdefault(k, []).append(b.m)
    return out


def kind_weights(ys: Mapping[str, Sequence[float]], n0: float, kinds: Sequence[str] = KINDS
                 ) -> dict[str, dict[str, float]]:
    """kind -> {n, mean, yield, w}: ``yield`` the kind's mean shrunk to the all-kind mean with ``n0`` pseudo-nodes,
    ``w`` = exp((yield - best yield) / sd) normalised over ``kinds`` (sd: the spread of every node's value -- a kind one
    node-sd below the best gets e^-1 of its share). Too little data (< 2 values): uniform."""
    allv = [x for k in kinds for x in ys.get(k, ())]
    if len(allv) < 2:
        return {k: {"n": len(ys.get(k, ())), "mean": 0.0, "yield": 0.0, "w": 1.0 / len(kinds)} for k in kinds}
    mu, sd = statistics.fmean(allv), statistics.pstdev(allv) or 1e-9
    out = {}
    for k in kinds:
        xs = list(ys.get(k, ()))
        out[k] = {"n": len(xs), "mean": statistics.fmean(xs) if xs else mu,
                  "yield": (sum(xs) + n0 * mu) / (len(xs) + n0) if len(xs) + n0 > 0 else mu}
    top = max(r["yield"] for r in out.values())
    raw = {k: math.exp((r["yield"] - top) / sd) for k, r in out.items()}
    tot = sum(raw.values())
    for k in out:
        out[k]["w"] = raw[k] / tot
    return out


def floored(w: Mapping[str, float], floor: float) -> dict[str, float]:
    """``w`` renormalised with each entry at least ``floor`` of the total (a floor >= 1/len gives uniform)."""
    if not w:
        return {}
    floor = min(max(0.0, floor), 1.0 / len(w))
    tot = sum(w.values()) or 1.0
    p = {k: x / tot for k, x in w.items()}
    low = {k for k, x in p.items() if x < floor}
    while True:                                   # pin the low ones at the floor, share the rest by weight
        rest = sum(p[k] for k in p if k not in low) or 1.0
        free = 1.0 - floor * len(low)
        q = {k: floor if k in low else p[k] / rest * free for k in p}
        more = {k for k, x in q.items() if x < floor - 1e-12} - low
        if not more:
            return q
        low |= more


def engagement_rates(results: Sequence[Engagement], led: Ledger) -> dict[str, tuple[int, int]]:
    """source -> (engaged, measured): the mechanism-prediction hit rate for the meta-report."""
    out: dict[str, tuple[int, int]] = {}
    for e in results:
        if e.status in MEASURED and e.node in led.nodes:
            k = source_of(led.nodes[e.node])
            a, b = out.get(k, (0, 0))
            out[k] = (a + (e.status == "engaged"), b + 1)
    return out


# ---------------------------------------------------------------------------------------------- briefs
@dataclass(frozen=True)
class Brief:
    kind: str
    lane: str
    parents: tuple[str, ...]
    text: str
    axis: str = ""
    plateau_index: int | None = None
    branch: str = ""                  # fork-and-flush branch (loop.core.fnf); '' = the shared context
    ablated: tuple[str, ...] = ()     # the flushed context components (fnf.FLUSH for a branch)
    shortlist: tuple[str, ...] = ()   # family|mechanism the brief points at: the researcher pre-checks them (intake)
    persona: str = ""                 # v3-c6i: the session's discipline lens
    category: str = ""                # v3-c6i: the taxonomy category it was pointed at
    arm: str = ""                     # v3-c9 C2: "alt_base:<source>" -- starts from a non-incumbent base (core.altbase)


def _op_key(plan: Mapping[str, Any]) -> str:
    return f"{plan['op']}/{plan['op_arm']}" if plan.get("op_arm") else str(plan["op"])


def _arm(plan: Mapping[str, Any]) -> str:
    if plan.get("op") and plan.get("op_arm"):               # v3-c10: a per-arm operator-program session
        return f"program_arm:{plan['op']}:{plan['op_arm']}"
    a = plan.get("alt") or {}
    d = a.get("dist") if a else None
    if a and a.get("source") == "root":                  # root is far by construction: alt_dist=far
        return "alt_base:root:far"
    return f"alt_base:{a.get('source') or '?'}" + (f":{d:.3f}" if d is not None else "") if a else ""


def _name(status: Any) -> str:
    s = getattr(status, "status", status)
    return str(getattr(s, "name", None) or getattr(s, "value", None) or s or "").upper()


def _prompt(name: str) -> Path:
    """``<name>.md``: the project's domain overlay (loop.project.prompt_path) first, else :data:`PROMPTS`."""
    p = P.prompt_path(name)
    return p if p.parent != P.PROMPTS else PROMPTS / p.name


def _render(name: str, **values: Any) -> str:
    text = _prompt(name).read_text()
    return re.sub(r"\{([A-Za-z_][A-Za-z0-9_]*)\}",
                  lambda m: str(values[m.group(1)]) if m.group(1) in values else m.group(0), text)


def prompt_version(*names: str) -> str:
    h = hashlib.sha1()
    for n in names:
        h.update(_prompt(n).read_bytes())
    return h.hexdigest()[:8]


KIND_TEXT = {
    "explore": ("EXPLORE: propose ONE new mechanism on the incumbent, derived from its measured bottleneck. Aim for a "
                "change the recipe cannot express today (new signal, new sampling, new structure), not a dose of a term "
                "it has; read earlier sessions' postmortems as guesses and the measured tables as facts, and before "
                "you build, say which operator finding or computed fact your mechanism rests on."),
    "crossover": ("CROSSOVER (stack): {other} has a positive measured own effect and is not in the incumbent. Make "
                  "both mechanisms work together on the incumbent; resolve interactions, do not just pick one side."),
    "repair": ("REPAIR {axis_name}: the incumbent has headroom on this axis (table below). Find why it is held back "
               "and change that, without costing the guard axis."),
    "plateau": "PLATEAU EXPERIMENT from the plateau review: run its decisive experiment.\n{hypothesis}",
    "program": ("PROGRAM (multi-component moonshot): design ONE change on the incumbent that combines two to four "
                "mechanisms, each aimed at a measured deficit or untested lever below (they may span families), built "
                "to work TOGETHER and sized to close a large share of the gap to the target. Name every component in "
                "hypothesis.claim (component: mechanism, the deficit it targets, its expected share of the gain) and "
                "declare the doses that matter in `knobs`: a promising program gets its declared knobs tuned before it "
                "is judged. You have the moonshot turn and time budget."),
    # v3-c6i (2608.16951): the plateau kind's successor -- the review's hypotheses are anomalies, not the experiment
    "kuhn": ("KUHN (paradigm shift): the search has plateaued ({plateau}). Incremental moves on the incumbent's recipe "
             "stopped paying. (1) State the PARADIGM: the assumptions every recent incumbent change kept (objective, "
             "data, views / compute, readout, optimisation). (2) Name the ANOMALIES the computed facts below show that "
             "it does not explain. (3) Conjecture ONE paradigm-shifting change on the incumbent that resolves them, "
             "derived by ANALOGY from a DISTANT literature (outside {distant}: read it on "
             "the web and cite it in hypothesis.analogy = {{field, source, mapping}}). It may span families; you have "
             "the moonshot turn and time budget; declare the doses that matter in `knobs`.{review}"),
    # v3-c6i: discovered from the feature diagnostic (computed trigger), routed to the eval-only lane
    "readout": ("READOUT (eval-only): the incumbent's probe features are distorted on some datasets (the diagnostic "
                "summary above). Propose ONE readout-only change that corrects the distortion without retraining: "
                "edit ONLY inside {spans}, and change no config value -- such a change is scored by the eval-only "
                "lane on trained checkpoints (minutes, several models). A change that also alters training is "
                "retrained as a normal node."),
}


def target_gap(v: View) -> float | None:
    """The gap to the external target when target mode is in force (``acquire.gap_of``), else None -- on the VISIBLE
    composite (v3-c6r H2: this number is shown in briefs and is the calibration reference gap; the incumbent's
    official-panel level carries the held-out tasks): the incumbent's level is its visible-composite mean
    (``holdout.visible_panel``), a target stated as an official-panel score is moved by the epoch's mean visible-vs-
    official offset. The acquisition and the TEST bar keep the official panel."""
    from loop.core import acquire as A
    from loop.core import holdout as HO
    try:
        if not A.target_on():
            return None
        got = A.gap_se(HO.visible_panel(v.led), v.incumbent)
        if got is None:
            return None
        t = A.TARGET
        gap = got[0] + (HO.panel_offset(v.led) if t is not None and t.score is not None else 0.0)
        on = gap > 0 if A.TARGET_ACTIVE is None else A.TARGET_ACTIVE
        return gap if on and gap > 0 else None
    except (KeyError, ValueError, ZeroDivisionError):
        return None


def ref_gap(v: View, gap: float | None = None) -> tuple[float, str]:
    """(the gain a candidate's p_gap is stated against, what it is): the gap to the target in target mode, else the
    5-seed MDE -- both on the visible screening composite (:func:`target_gap`)."""
    gap = target_gap(v) if gap is None else gap
    if gap is not None and gap > 0:
        return gap, "the gap to the external target, on the visible screening composite"
    return v.mde(), "the smallest gain a 5-seed fresh test resolves"


def ref_kind(what: str) -> str:
    """The definition of a reference gap (:func:`ref_gap`'s text): target | mde (calibration base rates per kind)."""
    return "target" if "target" in what else "mde"


def decisive_text(v: View, claim_gap_frac: float = 0.0, gap: float | None = None) -> str:
    """The brief's "Effect size" (v3-c6i: NO size floor -- 10-13, claims clustered at whatever minimum the brief asked
    for, the MDE and then the gap share): the reference sizes, and the per-axis budget tied to the brief's row ids.
    ``claim_gap_frac`` is kept for the call sites; it no longer sets a floor."""
    gap = target_gap(v) if gap is None else gap
    w = ", ".join(f"{a} {x:g}" for a, x in OFFICIAL_WEIGHTS.items())
    budget = ("Budget it per axis: in mechanism_prediction.budget, one entry per axis you move -- the axis, its "
              "expected level gain, and the bracketed row id ([D-..], [H-..], [X-..], [XD-..], [C-compute], [R-..], "
              f"[U-..], [T-..], [G-..]) of the deficit / headroom it draws on; the panel claim is the sum of official "
              f"weight x gain ({w}); a gain should fit inside the headroom its row shows. State the size you believe: "
              "it is scored for calibration against the measured effect, never bounced for being small.")
    if gap is None or gap <= 0:
        return (f"## Effect size\nA 5-seed fresh test resolves a composite gain of about {v.mde():.4f} (2σ_C/√5). "
                + budget)
    return (f"## Effect size (a per-axis budget)\nThe {target_label()} is {gap:+.4f} above the incumbent; a large "
            "share of it usually takes a strong mechanism or a multi-component PROGRAM, not a dose. " + budget)


# ---------------------------------------------------------------------------------------------- v3-c6i brief parts
def distortions(feats: Mapping[str, Mapping[str, Any]] | None, ratio: float, reference: str = ""
                ) -> list[dict[str, Any]]:
    """Per-dataset readout distortion in the incumbent's feature diagnostic (``diagnose.cached_features``): a dataset
    whose per-tile gate mean (the first gate inside probe_features) or head-vs-cls cosine is below ``ratio`` x the
    reference -- the ``reference`` dataset's value when it is in the report, else the median over the datasets.
    Most distorted first."""
    if not feats or ratio <= 0:
        return []

    def val(d: Mapping[str, Any], metric: str) -> float | None:
        x = ((d.get("gates") or [{}])[0] if metric == "gate" else d.get("cos_head_vs_cls") or {}).get("mean")
        return float(x) if isinstance(x, (int, float)) and math.isfinite(x) else None
    out = []
    for metric in ("gate", "cos"):
        vals = {ds: x for ds, d in feats.items() if (x := val(d, metric)) is not None}
        if len(vals) < 3:
            continue
        ref = vals.get(reference) if reference else None
        ref, ref_name = (ref, reference) if ref is not None else (statistics.median(vals.values()), "median")
        if ref <= 0:
            continue
        for ds, x in vals.items():
            if ds != reference and x < ratio * ref:
                out.append({"dataset": ds, "metric": metric, "value": x, "ref": ref, "ref_name": ref_name,
                            "ratio": x / ref})
    return sorted(out, key=lambda r: r["ratio"])


METRIC_NAME = {"gate": "per-tile gate mean in probe_features", "cos": "cos(probe head, cls)"}


def diag_summary(ctx: Any, store: Any, incumbent: str, k: int = 3, ratio: float = 0.5, reference: str = ""
                 ) -> tuple[str, list[dict[str, Any]]]:
    """(the 3-5 line computed diagnostic summary at the top of every brief, the distortions behind it); ('', []) when
    neither a feature report nor a per-dataset breakdown exists for the incumbent (AIDE^2 2609.26457: triggered)."""
    from loop.agents import diagnose as DX
    try:
        feats = DX.cached_features(ctx, store, incumbent) if getattr(ctx, "state_dir", "") else None
    except Exception:  # noqa: BLE001 - no report: no lines
        feats = None
    dist = distortions(feats, ratio, reference)
    lines = []
    by: dict[str, list[dict[str, Any]]] = {}
    for d in dist:
        by.setdefault(d["metric"], []).append(d)
    for metric, ds in by.items():
        lines.append(f"- {METRIC_NAME[metric]}: " + ", ".join(f"{d['dataset']} {d['value']:.2f}" for d in ds[:4])
                     + f" vs {ds[0]['ref_name']} {ds[0]['ref']:.2f} over {len(feats or {})} datasets "
                       f"(x{min(d['ratio'] for d in ds):.2f} at worst): the readout shrinks / distorts these sets")
    cls = store.get_state(DX.K_CLS) or {} if hasattr(store, "get_state") else {}
    if cls.get("incumbent") == incumbent and isinstance(cls.get("rows"), list) and cls["rows"]:
        cls = {**cls, "rows": DX.unheld(cls["rows"], store)}        # c6a: the held-out dataset is never shown
        gaps = DX.dataset_gaps(cls["rows"])[:k]
        if gaps:
            lines.append("- largest per-dataset public gaps: " + ", ".join(
                f"{g['dataset']}/{g['head']} {-g['gap']:+.3f} vs {g['ext'][0].split('-')[0]}" for g in gaps))
        mine = [r for r in cls["rows"] if r.get("best") and r["best"][1] - r["ours"] > 0]
        lead = sorted(mine, key=lambda r: -(r["best"][1] - r["ours"]))[:k]
        if lead:
            lines.append("- largest per-dataset leads of other nodes: " + ", ".join(
                f"{r['dataset']}/{r['head']} {r['best'][0]} {r['best'][1] - r['ours']:+.3f}" for r in lead))
    if not lines:
        return "", dist
    return (f"## Diagnostic summary of the incumbent {incumbent} (computed; `diagnose` for the full tables)\n"
            + "\n".join(lines[:5])), dist


def persona_text(persona: str) -> str:
    return (f"## Your lens this session: {persona}\nThink as a researcher in {persona} would: frame the incumbent's "
            "bottleneck in that discipline's terms and derive the change from its methods. The other sessions use "
            "other lenses.") if persona else ""


def bank_text(e: Mapping[str, Any]) -> str:
    """A bank idea as a brief head addition (v3, ``ideation.bank_v3``)."""
    from loop.select import bank as BK
    if e.get("plan"):
        return BK.text(dict(e))
    rows = [("Idea", e.get("idea")), ("Why it might work", e.get("why_it_might_work")),
            ("Evidence", e.get("evidence")), ("Axis it should move", e.get("axis")),
            ("Knobs", json.dumps(e["knobs"], default=str) if isinstance(e.get("knobs"), (dict, list)) else e.get("knobs")),
            ("Status so far", e.get("v1_fate"))]
    return "\n".join([f"## Idea to build (bank `{e['id']}`: {e.get('title')}; source: {e.get('source')})",
                       "Build THIS idea (adapt it to the code as it is now; if the code or the ledger shows it is "
                       "already in the recipe or measured dead, build its closest untested variant and say why in "
                       "handoff.learnings):"] + [f"- {k}: {v}" for k, v in rows if v])


def taken_text(live: Sequence[Mapping[str, Any]], queued: Sequence[Any], k: int = 24) -> str:
    """The terse reservation list of the hidden-pool kinds: mechanism names only, no descriptions."""
    names = [f"{c.get('family')}/{c.get('tag')}" for e in live if (c := e.get("claim") or {}).get("tag")]
    names += [f"{n.family}/{n.mechanism_tag}" for n in queued]
    names = list(dict.fromkeys(names))[:k]
    return ("## Taken (in flight / queued: do not build these)\n" + ", ".join(names)) if names else ""


def program_text(p: Mapping[str, Any]) -> str:
    """An operator-seeded program's brief block (``ideation.operator_programs``; operator text, shown as given)."""
    def items(key: str) -> list[str]:
        return [x if isinstance(x, str) else json.dumps(x, default=str) for x in p.get(key) or ()]
    if isinstance(p.get("_arm"), Mapping):                # v3-c10: ONE arm of the program, its own session
        return arm_text(p, p["_arm"])
    out = [f"## Operator-seeded program {p.get('id')}", str(p.get("brief") or "").strip()]
    if st := items("stages"):
        out.append("Stages:\n" + "\n".join(f"- {x}" for x in st))
    if co := items("components"):
        out.append("Components:\n" + "\n".join(f"- {x}" for x in co))
    if p.get("budget_gpu_h"):
        out.append(f"Budget: ~{p['budget_gpu_h']} GPU-h.")
    if hg := items("harm_guards"):
        out.append(f"Harm guards: {', '.join(hg)}; each arm is decided on its own paired increment.")
    eo = [a for a in p.get("arms") or () if isinstance(a, Mapping) and a.get("lane") == "evalonly"]
    if eo:                                   # v3-c6r O1: the loop builds the eval-only arms itself (one node each)
        out.append("The eval-only arms (" + ", ".join(str(a.get("id")) for a in eo) + ") are built by the loop "
                   "itself, one eval-only node each (literal edits inside the readout span, no config change): do "
                   "NOT build them. Build the first FULL-RETRAIN stage as your node (its components in `components`, "
                   "the later stages' doses in `knobs` / the handoff) on the incumbent as it is; a gate arm that "
                   "passes composes with it later (it is readout-only). The loop's intake, screens, tests and harm "
                   "guards judge it as any program.")
    else:
        out.append("Build the FIRST stage as your node (its components in `components`, the later stages' doses in "
                   "`knobs` / the handoff); the loop's intake, screens, tests and harm guards judge it as any program.")
    return "\n".join(x for x in out if x)


def arm_text(p: Mapping[str, Any], arm: Mapping[str, Any]) -> str:
    """v3-c10: the brief block of ONE full-retrain arm of an operator program (``program.units``): the program's brief,
    this arm's own spec, and the one-arm rule. The other arms' components / stages are not shown."""
    from loop.core import program as PG
    aid = str(arm.get("id"))
    others = [a for a in PG.full_arms(p) if str(a.get("id")) != aid and not a.get("after")]
    out = [f"## Operator-seeded program {p.get('id')}: YOUR ARM is {aid} (stage {arm.get('stage') or '?'})",
           str(p.get("brief") or "").strip(), f"YOUR ARM {aid}: {str(arm.get('note') or '').strip()}"]
    if comps := [x if isinstance(x, str) else json.dumps(x, default=str) for x in arm.get("components") or ()]:
        out.append("Components of your arm:\n" + "\n".join(f"- {x}" for x in comps))
    if arm.get("ladder"):
        out.append(f"Dose ladder for your arm: {arm['ladder']}")
    if arm.get("after"):
        out.append(f"This is the STACK arm: build it only from the arms {', '.join(map(str, arm['after']))} whose own "
                   "node screened POSITIVE on its own paired increment (read the ledger rows); stack just those. "
                   "If fewer than two screened positive, make no stack: say so in the finish.")
    elif others:
        fk = PG.foreign_keys(p, arm)
        out.append(
            f"ONE-ARM RULE: the other arms ({', '.join(str(a.get('id')) for a in others)}) are separate sessions and "
            f"separate nodes. Implement EXACTLY arm {aid} and nothing from them: no code, config key, flag or stub of "
            f"theirs" + (f" (their identifiers: {', '.join(fk)})" if fk else "") + ". The operator's bundled node of "
            "these arms lost and nobody could tell which part hurt; a node touching another arm's keys is rejected at "
            "intake (cross_arm_bundling). The program's other stages are NOT yours.")
    out.append(f"Declare `components` in your finish (one component per tunable part of YOUR arm, each knob "
               f"{{key, lo, hi, scale}}): the loop then runs your arm's dose ladder (up to ~5 trials) before judging "
               f"it. Your node is the arm's first point; its own paired increment decides it.")
    if p.get("budget_gpu_h"):
        out.append(f"Program budget: ~{p['budget_gpu_h']} GPU-h over all arms.")
    if hg := [str(x) for x in p.get("harm_guards") or ()]:
        out.append(f"Harm guards: {', '.join(hg)}; your arm is decided on its own paired increment.")
    return "\n".join(x for x in out if x)


class Ideation:
    """The tick's ``ideate`` and ``intake`` hooks. ``brief_for(lane, data, post)`` is pure given the store."""

    def __init__(self, ctx: Any, store: Any, *, llm: Any = None, embedder: Any = None,
                 clock: Callable[[], float] = time.time, readme: str | Path | None = None,
                 executor: Callable[..., Any] | None = None) -> None:
        self.ctx, self.store, self.llm, self.clock = ctx, store, llm, clock
        self.embedder = embedder or HashingEmbedder()
        repo = getattr(getattr(ctx, "spec", None), "repo", "") or ""
        self.fm = fm_reference(readme if readme is not None else Path(repo) / "README.md") if (readme or repo) else {}
        self._view: tuple[int, View] | None = None
        self._review: Any = None                       # the plateau review future
        self._facts: tuple[int, str] | None = None     # (id of the view, its computed-facts text)
        self._statuses: Mapping[str, Any] | None = None   # this tick's derived statuses (ideate sets them)
        self._harm: Mapping[str, Any] = {}             # the core's harm keywords (the meta-cycle's TUNABLE values)
        self._started = False                          # the first ideate of this process (tick 0)
        self._tick_kinds: list[str] = []               # explore-lane kinds given this tick (c8 merge: quota order)
        self._dry = False
        # R4: briefs rendered off the tick thread (ideation.brief_workers): one job per enqueued slot, started by a
        # later tick once rendered; _lock guards the job list and the per-view caches the workers fill
        self._pool: ThreadPoolExecutor | None = None
        # the brief pool's class: the Tick's executor (an inline one in tests: rendered at once, started this tick);
        # None = the class of the calling tick's pool (``tick.pool``), else a thread pool
        self._executor_cls, self._tick_pool = executor, None
        self._jobs: list[dict[str, Any]] = []          # {fut, plan, lane, ts}: rendering, or rendered + unstarted
        self._trickle = False                          # 11-09: this tick's admitted slots are throttle-trickle slots
        self._ingest: Future | None = None             # the diagnostics ingest running off the tick
        self._lock = threading.RLock()
        self._kw: tuple[tuple, dict[str, dict[str, float]]] | None = None   # (view token, kind_weights)
        self._kw_pool: dict[str, float] = {}           # v3-c7i: the rate table's pooled row (loop.core.alloc)
        self.adapter: Any = None                       # v3-c7i: the tick's adapter (intake resolves knob keys)

    @property
    def bar_node(self) -> str:
        """The external bar (``labless.bar_node``): the reference the headroom table is measured against."""
        return getattr(getattr(getattr(self.ctx, "cfg", None), "labless", None), "bar_node", "") or ""

    # -- the computed picture ----------------------------------------------------------------------------------
    def view(self, data: Ledger, post: Any = None) -> View:
        """``post``: a View, or the tick's core posterior (``Tick.post``); else the core posterior is computed. The
        target (``loop.labless``) and acquisition keys are installed from this ideation's own config first
        (``acquire.configure``): the distance section never depends on a tick having ranked before."""
        if (cfg := getattr(self.ctx, "cfg", None)) is not None:
            from loop.core import acquire as A
            from loop.core import fnf as FN
            A.configure(cfg)
            FN.SCREEN_CREDIT = bool(self.opt("fnf_screen_credit"))
            FN.SCREEN_MIN_SEEDS = int(self.opt("fnf_screen_min_seeds"))
        if isinstance(post, View):
            return post
        if self._view is None or self._view[0] != id(data):
            self._view = (id(data), view(data, blocks_of(self.store, data), post))
        return self._view[1]

    def opt(self, name: str) -> Any:
        """``ideation.<name>`` of this ideation's config (the schema default without one)."""
        from loop.core.levers import opt
        return opt(self.ctx, name)

    def program_share(self, v: View | None = None) -> float:
        """``ideation.program_share`` in target mode, else ``program_share_base`` -- v3-c7i: a CEILING, decayed toward
        ``alloc.program_share_floor`` while the program kind's yield per GPU-h is below the LLM kinds' pooled rate
        (:func:`loop.core.alloc.program_share`; with a view and ``alloc.kind_rate``)."""
        from loop.core import acquire as A
        on = target_gap(v) is not None if v is not None else bool(A.target_on() and A.TARGET_ACTIVE)
        ceiling = float(self.opt("program_share" if on else "program_share_base"))
        ac = self._alloc()
        if ceiling <= 0 or v is None or not (ac.enabled and ac.kind_rate):
            return ceiling
        from loop.core import alloc as AL
        tab = self.kind_table(v)
        return AL.program_share(ceiling, float(ac.program_share_floor), tab, self._kw_pool)

    def _alloc(self) -> Any:
        from loop.core import alloc as AL
        return AL.cfg_of(self.ctx)

    def weighted(self) -> bool:
        return bool(self.opt("kind_weight"))

    def kind_for(self, lane: str, v: View | None = None) -> str:
        """The brief kind of a slot: the crossover / exploit lanes' own kinds; an explore-lane slot gets a program
        while programs are behind ``program_share`` (the operator's share), then -- ``ideation.kind_weight`` with a
        view -- the eligible kind (explore, literature, repair, program, plateau while its review has an unserved
        hypothesis) furthest behind its yield-weighted share of this epoch's sessions (:meth:`kind_table`); else
        the fixed rotation (plateau first, every LITERATURE_EVERY-th a literature brief)."""
        kind = LANE_KIND.get(lane, "explore")
        if kind != "explore":
            return kind
        k = self._explore_kind(v)
        self._tick_kinds.append(k)
        return k

    def _explore_kind(self, v: View | None) -> str:
        """An explore-lane slot's kind (c8 merge order, operator 10-14: bold is a ~30% lane): the bold quota FIRST
        (``bold_share`` blended with the bandit, ``bold_floor``), then -- an operator program pending and none yet
        this tick, after a bold slot -- a program (never starved), then the program quota, then the bandit / rotation."""
        weighted = self.weighted() and v is not None
        if not weighted and self._plateau_next() is not None:
            return "kuhn"
        from loop.core import bold as BOLD              # v3-c8b: the bold lane's quota, counted from its deploy
        share = self.program_share(v)
        tk = self._tick_kinds
        if (wb := self.bold_share(v if weighted else None, share > 0)) > 0 and BOLD.behind(
                self.store, [j["plan"]["kind"] for j in self._pending()] + (list(tk) if self._dry else []), wb,
                BOLD.LANE_KINDS, self._bold_since()):
            return BOLD.KIND
        if self._op_next() is not None:                   # v3-c6i: an operator-seeded program, once each
            return "program"
        st = self.store.get_state(K_PROGRAM) or {}        # counted from the deploy: no catch-up burst
        if share > 0 and int(st.get("p", 0)) < share * (int(st.get("n", 0)) + 1):
            return "program"
        if weighted:
            return self._weighted_kind(v, share > 0)
        if v is not None and self._readout_ok(v):          # v3-c6i: a computed readout distortion, few per incumbent
            return "readout"
        n = sum(e.get("brief_kind") in ("explore", "literature") for e in self.store.events("research_started"))
        n += sum(j["plan"]["kind"] in ("explore", "literature") for j in self._pending())
        return "literature" if n % LITERATURE_EVERY == LITERATURE_EVERY - 1 else "explore"

    def kind_table(self, v: View) -> dict[str, dict[str, float]]:
        """kind -> {n, mean, yield, w} measured on this epoch (:func:`kind_yields` / :func:`kind_weights`), once per
        view; a changed table is stored in state ``ideation.kind_weights`` and logged ``kind_weights``."""
        tok = _view_token(v)
        with self._lock:
            if self._kw is not None and self._kw[0] == tok:
                return self._kw[1]
        ac = self._alloc()
        if ac.enabled and ac.kind_rate:                # v3 review K1: yield per GPU-h over the incumbent, Thompson
            from loop.core import alloc as AL
            tab, self._kw_pool, split = AL.table(self.ctx, v.led, AL.post_beliefs(v), tok, v.incumbent)
            tab = {k: dict(r) for k, r in tab.items() if k in KINDS}      # the brief kinds; code kinds: meta
            w = AL.thompson(self._kw_pool, list(tab))                     # v3 review K1: P(argmax) among them
            for k, r in tab.items():
                r["w"] = w.get(k, 0.0)
        else:
            clade = int(self.opt("clade_min_seeds")) if self.opt("clade_credit") else None     # v3-c6i: clade credit
            tab = kind_weights(kind_yields(v, bool(self.opt("kind_yield_descendants")), clade),
                               float(self.opt("kind_yield_n0")))
            self._kw_pool = {}
        from loop.core import calibration as CAL       # v3-c6i: calibrated kinds get more of the share
        skill = CAL.kind_skill(self.store.get_state(CAL.K_CAL) or {}, float(self.opt("calibration_n0")))
        if skill and float(self.opt("calibration_weight")) != 0:
            w = CAL.reweight({k: r["w"] for k, r in tab.items()}, skill, float(self.opt("calibration_weight")))
            for k in tab:
                tab[k]["w"], tab[k]["skill"] = w[k], round(float((skill.get(k) or {}).get("skill", 0.0)), 5)
        with self._lock:
            self._kw = (tok, tab)
        rounded = {k: {f: float(f"{x:.4g}") if self._kw_pool else round(x, 5) for f, x in r.items()}
                   for k, r in tab.items()}                # rates per GPU-h are ~1e-5: significant digits
        if self._kw_pool:                                  # v3 review K1: Thompson weights move in ~1/400 steps
            for r in rounded.values():
                r["w"] = round(float(r.get("w", 0.0)), 2)
        if self.store.get_state(K_KIND_W) != rounded:
            self.store.set_state(K_KIND_W, rounded)
            self.store.log("kind_weights", **rounded)
        return tab

    def _eligible(self, v: View, program: bool) -> list[str]:
        return ["explore", "literature", "repair", *(["program"] if program else []),
                *(["kuhn"] if self._plateau_next() is not None else []),
                *(["readout"] if self._readout_ok(v) else [])]

    def bold_share(self, v: View | None = None, program: bool = False) -> float:
        """v3-c8b: the bold lane's share of the explore-lane sessions (loop.core.bold.weight): ``ideation.bold_share``
        blended with the kind bandit's weight for ``bold`` (its rate row vs the eligible kinds', floored like them)
        over ``bold_n0`` pseudo-sessions, held >= ``bold_floor`` while it has < ``bold_floor_n`` nodes; without a view
        (the fixed rotation) the operator share."""
        from loop.core import bold as BOLD
        share = float(self.opt("bold_share"))
        if share <= 0 or v is None:
            return max(0.0, share)
        tab = self.kind_table(v)
        ks = [*self._eligible(v, program), BOLD.KIND]
        if self._kw_pool:
            from loop.core import alloc as AL
            raw = AL.softmax(tab, ks, float(self._kw_pool.get("temp") or 0.0))
        else:
            raw = {k: tab[k]["w"] for k in ks if k in tab}
        w = floored(raw, float(self.opt("kind_weight_floor")))
        return BOLD.weight(w, BOLD.n_bold(v.led), share=share, floor=float(self.opt("bold_floor")),
                           floor_n=int(self.opt("bold_floor_n")), n0=float(self.opt("bold_n0")))[BOLD.KIND]

    def _bold_since(self) -> float:
        """When the bold lane started counting (state ``ideation.bold_since``, set on first use): its quota is a share
        of the sessions SINCE then -- no catch-up burst over the epoch's earlier sessions."""
        from loop.core import bold as BOLD
        got = self.store.get_state(BOLD.K_SINCE)
        if got is None:
            got = float(self.clock())
            self.store.set_state(BOLD.K_SINCE, got)
        return float(got)

    def _weighted_kind(self, v: View, program: bool) -> str:
        eligible = self._eligible(v, program)
        tab = self.kind_table(v)
        if self._kw_pool:                              # v3-c7i: the softmax over the ELIGIBLE kinds' rates only
            from loop.core import alloc as AL
            raw = AL.softmax(tab, eligible, float(self._kw_pool.get("temp") or 0.0), self._kw_pool)   # Thompson
            if any("skill" in tab[k] for k in eligible):
                from loop.core import calibration as CAL
                skill = CAL.kind_skill(self.store.get_state(CAL.K_CAL) or {}, float(self.opt("calibration_n0")))
                raw = CAL.reweight(raw, skill, float(self.opt("calibration_weight"))) if skill else raw
        else:
            raw = {k: tab[k]["w"] for k in eligible if k in tab}
        w = floored(raw, float(self.opt("kind_weight_floor")))
        nodes = [n for n in v.led.nodes.values() if n.base == getattr(v.led, "epoch", None)]
        t0 = min((n.created_ts for n in nodes), default=0.0)       # this epoch's sessions only
        c = {k: 0 for k in eligible}
        for e in self.store.events("research_started"):
            if e.get("brief_kind") in c and float(e.get("ts") or 0.0) >= t0:
                c[e["brief_kind"]] += 1
        for j in self._pending():
            if j["plan"]["kind"] in c:
                c[j["plan"]["kind"]] += 1
        n = sum(c.values()) + 1
        return max(eligible, key=lambda k: (w[k] * n - c[k], w[k]))

    # -- v3-c6i: operator programs, the readout trigger, personas, coverage ------------------------------------
    def _op_next(self) -> dict[str, Any] | None:
        """The first operator-seeded program (``ideation.operator_programs``) not briefed yet (state K_OPPROG)."""
        from loop.core import program as PG
        done = self.store.get_state(K_OPPROG) or {}
        book = self.store.get_state(PG.K_ARMS) or {}
        for p in self.opt("operator_programs") or ():
            if not isinstance(p, Mapping) or not p.get("id") or p.get("superseded_by") or p.get("sessions") is False:
                continue                                    # superseded by a newer program / built by the loop, no session
            if str(p["id"]) in done and len(PG.full_arms(p)) < 2:
                continue                                    # briefed whole (legacy key). A program rewritten into per-arm
                                                            # sessions ignores it: its whole-program session built the arms
                                                            # the operator since removed (10-21 sampler: arms B / D never ran)
            for key, arm in PG.units(p):
                if key in done:
                    continue
                if arm is not None and arm.get("after") and not self._arms_settled(book.get(str(p["id"])) or {},
                                                                                    arm["after"]):
                    continue                                # a stack arm waits for its arms' own verdicts
                return {**dict(p), **({"_arm": dict(arm)} if arm is not None else {})}
        return None

    def _arms_settled(self, mine: Mapping[str, Any], ids: Sequence[Any]) -> bool:
        """Every arm of ``ids`` has its node booked (``K_ARMS``) and past the queue / screen (or was given up)."""
        st = getattr(self, "_statuses", None) or {}
        for a in ids:
            rec = mine.get(str(a))
            if isinstance(rec, str) and rec.startswith("-"):
                continue
            if not isinstance(rec, str) or _name(st.get(rec)) in ("", "QUEUED", "SCREENING"):
                return False
        return True

    def _op_find(self, pid: str, aid: str = "") -> dict[str, Any] | None:
        """The configured program ``pid`` (with its arm ``aid`` as ``_arm``), for a re-plan / intake."""
        from loop.core import program as PG
        p = next((dict(p) for p in self.opt("operator_programs") or ()
                  if isinstance(p, Mapping) and str(p.get("id")) == pid and p.get("sessions") is not False), None)
        if p is None or not aid:
            return p
        arm = next((dict(a) for a in PG.full_arms(p) if str(a.get("id")) == aid), None)
        return {**p, "_arm": arm} if arm is not None else p

    def _op_mark(self, pid: str, sign: int = 1) -> None:
        """Book (or give back) a unit: ``pid`` (a legacy whole program) or ``pid/arm`` (one arm's session)."""
        st = dict(self.store.get_state(K_OPPROG) or {})
        if sign > 0:
            st[pid] = {"ts": self.clock()}
        else:
            st.pop(pid, None)
        self.store.set_state(K_OPPROG, st)

    def diag(self, v: View) -> tuple[str, list[dict[str, Any]]]:
        """The diagnostic summary (:func:`diag_summary`) of the view's incumbent, once per view."""
        tok = _view_token(v)
        if (got := self.__dict__.get("_diag")) is not None and got[0] == tok:
            return got[1]
        out = ("", [])
        if self.opt("diag_summary"):
            out = _safe(lambda: diag_summary(self.ctx, self.store, v.incumbent, int(self.opt("diag_summary_k")),
                                             float(self.opt("diag_ratio")), str(self.opt("readout_reference") or "")),
                        ("", []))
        self._diag = (tok, out)
        return out

    def _readout_ok(self, v: View) -> bool:
        """The readout kind is eligible: the features show a distorted dataset and fewer than
        ``ideation.readout_max_sessions`` readout sessions were briefed for this incumbent."""
        if not isinstance(v, View) or self.opt("training_only"):   # 11-08: training-side changes only (nanobrain)
            return False
        n = int((self.store.get_state(K_READOUT) or {}).get(v.incumbent, 0))     # booked at enqueue (_reserve)
        return n < int(self.opt("readout_max_sessions")) and bool(self.diag(v)[1])

    def _count_readout(self, kind: str, inc: str, sign: int = 1) -> None:
        if kind == "readout" and inc:
            st = dict(self.store.get_state(K_READOUT) or {})
            st[inc] = max(0, int(st.get(inc, 0)) + sign)
            self.store.set_state(K_READOUT, st)

    def _persona(self, peek: bool = False) -> str:
        """The next lens of the ``ideation.personas`` round-robin (``peek``: without advancing it)."""
        ring = [str(p) for p in self.opt("personas") or () if str(p).strip()]
        if not ring:
            return ""
        n = int(self.store.get_state(K_PERSONA) or 0)
        if not peek:
            self.store.set_state(K_PERSONA, n + 1)
        return ring[n % len(ring)]

    def _category(self, v: View, live: Sequence[Mapping[str, Any]]) -> tuple[str, Any]:
        """(the taxonomy category a new brief is pointed at, the coverage counts it was chosen from)."""
        from loop.core import mechanisms as MS
        tok = _view_token(v)
        if (got := self.__dict__.get("_cov")) is None or got[0] != tok:
            got = (tok, _safe(lambda: MS.coverage(self.ctx, self.store, v.led), None))
            self._cov = got
        if got[1] is None:
            return "", None
        counts = got[1].copy()
        for e in live:
            if e.get("category"):
                counts[e["category"]] += 1
        return MS.suggest(counts, MS.categories(self.ctx)), counts

    def _count_program(self, kind: str, sign: int = 1) -> None:
        """The explore-kind session count (and its programs); ``sign`` -1 rolls one back (an unstarted brief dropped)."""
        if kind in EXPLORE_KINDS:
            st = dict(self.store.get_state(K_PROGRAM) or {})
            st.update(n=max(0, int(st.get("n", 0)) + sign), p=max(0, int(st.get("p", 0)) + sign * (kind == "program")))
            self.store.set_state(K_PROGRAM, st)

    def _count_alt(self, plan: Mapping[str, Any], sign: int = 1) -> None:
        """v3-c9 C2: a booked plan counts as an eligible session (an alt-based one also as an alt session and a use of
        its base); -1 rolls it back."""
        from loop.core import altbase as AB
        try:
            AB.count(self.store, plan, sign)
            if sign > 0 and plan.get("alt"):
                self.store.log("alt_base_pick", node=plan["alt"]["node"], source=plan["alt"]["source"],
                               kind=plan.get("kind"), branch=plan.get("branch", ""), m=plan["alt"]["m"],
                               n=plan["alt"]["n"], via=plan["alt"].get("via", ""))
        except Exception as exc:  # noqa: BLE001 - accounting never stops a brief
            self.store.log("alt_base_error", error=repr(exc)[:300])

    def _serve(self, idx: int | None) -> None:
        """Reserve plateau hypothesis ``idx`` (a returned one is taken back, else ``served`` moves past it)."""
        if idx is None:
            return
        st = dict(self.store.get_state(K_PLATEAU) or {})
        back = [int(i) for i in st.get("returned") or ()]
        if idx in back:
            back.remove(idx)
            st["returned"] = back
        else:
            st["served"] = max(int(st.get("served") or 0), idx + 1)
        self.store.set_state(K_PLATEAU, st)

    def _unreserve(self, plan: Mapping[str, Any]) -> None:
        """v3-c5b S7: a brief dropped before its session started gives back what its enqueue counted -- the program
        quota count and its plateau hypothesis (``returned``: served first by the next plateau brief)."""
        self._count_program(plan.get("kind", ""), -1)
        self._count_alt(plan, -1)
        self._count_readout(plan.get("kind", ""), plan.get("inc", ""), -1)
        if plan.get("op"):
            self._op_mark(_op_key(plan), -1)
        if (idx := plan.get("idx")) is not None:
            st = dict(self.store.get_state(K_PLATEAU) or {})
            if int(st.get("served") or 0) == idx + 1:
                st["served"] = idx
            elif idx not in (back := [int(i) for i in st.get("returned") or ()]):
                st["returned"] = sorted([*back, idx])
            self.store.set_state(K_PLATEAU, st)

    def brief_for(self, lane: str, data: Ledger, post: Any = None, kind: str | None = None, branch: str = ""
                  ) -> Brief:
        """``branch``: a fork-and-flush branch session -- the flushed brief (:meth:`branch_brief`)."""
        v = self.view(data, post)
        if branch:
            plan = self._plan(lane, v, kind or self.kind_for(lane, v), branch)
            return self.branch_brief(lane, v, plan["kind"], branch, plan)
        return self.brief_text(self.brief_plan(lane, v, kind or self.kind_for(lane, v)), data, v)

    def brief_plan(self, lane: str, v: View, kind: str, plateau_at: int | None = None,
                   keep: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """The DECISIONS of a brief (cheap; on the tick thread): its kind after fallbacks, parents (a crossover's
        partner), repair axis, plateau hypothesis index (``plateau_at``: one reserved already), head text, the
        headroom rows and the in-flight list it was decided against. :meth:`brief_text` renders it. v3-c6i: the
        persona lens, the taxonomy category, an operator program -- ``keep`` (a re-plan): the plan whose persona /
        operator program it keeps. No side effects: :meth:`_reserve` books what a queued plan holds."""
        inc = v.incumbent
        rows = headroom(v, self.fm, self.bar_node)
        live = self.inflight()
        axis = rotate_axis(repair_axes(rows, v.weights), [e["axis"] for e in live if e["kind"] == "repair"])
        parents, head, idx, pair_why, pair = (inc,), "", None, "", None
        if kind == "crossover":
            got = self.code_pair(v, self._held_pairs())  # v3-c6i: a code-picked pair (loop.core.pairs) when present
            if got is not None:                          # two legs, merged from their common ancestor (not the incumbent)
                parents, other = (got[0], got[1]), got[1]
                pair_why, pair = got[2], [got[0], got[1]]
            elif self._xover_conflict_only() or \
                    (other := self.stack_partner(v, self.reserved_partners(live))) is None:
                kind = "explore"                         # v3-c7i I5: no conflicting pair -> stacking is the planner's
            else:
                parents = (inc, other)
            if kind == "crossover":
                b = v.beliefs.get(other)
                head = KIND_TEXT["crossover"].format(other=f"{other} ({_tag(self.store, other)}"
                                                           + (f", own gain {b.m:+.4f})" if b else ")"))
                if pair_why:                             # a code-picked pair: both legs are named, neither is the incumbent
                    head = (f"CROSSOVER (pair): merge {parents[0]} ({_tag(self.store, parents[0])}) and {parents[1]} "
                            f"({_tag(self.store, parents[1])}), the best complementary pair the measurements found: "
                            f"{pair_why}. Make both mechanisms work together; resolve interactions, do not just pick one side.")
        if kind in ("plateau", "kuhn"):                 # v3-c6i: the plateau's hypotheses brief a kuhn session
            st = self.store.get_state(K_PLATEAU) or {}
            hyps = st.get("hypotheses") or []
            nxt = self._plateau_next() if plateau_at is None else \
                ((plateau_at, hyps[plateau_at]) if 0 <= plateau_at < len(hyps) else None)
            if nxt is None:
                kind = "explore"
            else:
                idx, kind = nxt[0], "kuhn"
                p = plateau(v.led, v.judged)
                review = (f"\n\nThe plateau review (its hypotheses are INCREMENTAL anomalies to explain, not the "
                          f"experiment):\ndiagnosis: {str(st.get('diagnosis') or '')[:1200]}\n" + "\n".join(
                              f"- {str(h.get('claim') or '')[:300]} (evidence: {str(h.get('evidence') or '')[:200]})"
                              for h in hyps[:4]))
                head = KIND_TEXT["kuhn"].format(
                    plateau=f"{p.gpu_h:.0f} GPU-h since the last pass, median {p.median:.0f} between passes",
                    review=review, distant=P.rule("kuhn_distant", "pathology and self-supervised vision"))
        if kind == "repair":
            head = KIND_TEXT["repair"].format(axis_name=AXIS_NAMES[axis])
        if kind == "literature":
            head = _render("brief_literature", family=f"the {AXIS_NAMES[axis]} axis or any bottleneck the table shows")
        if kind == "readout":
            head = KIND_TEXT["readout"].format(spans=self._readout_spans())
        if kind == "bold":                              # v3-c8b: the bold structural lane (loop.core.bold)
            from loop.core import bold as BOLD
            head = BOLD.brief_head(self.ctx)
        if keep is not None:                             # a re-plan keeps its reserved operator program
            op = self._op_find(str(keep["op"]), str(keep.get("op_arm") or "")) if keep.get("op") else None
        else:
            op = self._op_next() if kind == "program" else None
        if op is not None and kind == "program":         # v3-c6i: an operator-seeded program
            head = KIND_TEXT["program"] + "\n\n" + program_text(op)
        else:
            op = None
        category, _ = self._category(v, live) if kind in EXPLORE_KINDS + ("kuhn", "repair", "crossover") else ("", None)
        head = head or KIND_TEXT[kind]
        bank = self._bank_pick() if kind in ("explore", "literature") and op is None and keep is None else None
        if bank is not None:                            # 11-08: an operator / scout / root-audit idea bank idea leads
            head = head + "\n\n" + bank_text(bank)
        alt, eligible = None, False                      # v3-c9 C2: a non-incumbent starting point (core.altbase)
        if kind in tuple(self.opt("alt_base_kinds") or ()) and parents == (inc,) and not pair \
                and float(self.opt("alt_base_share")) > 0:
            eligible = True
            alt = self._alt_pick(v)
        if alt is not None:
            from loop.core import altbase as AB
            parents, head = (alt.node,), AB.brief_note(v.led, self.store, alt, inc, _tag(self.store, alt.node)) \
                + "\n\n" + AB.retarget(head, alt.node)
        return {"kind": kind, "lane": lane, "parents": parents, "axis": axis, "idx": idx,
                "head": head, "rows": rows, "live": live, "inc": inc,
                "persona": str((keep or {}).get("persona") or "") or self._persona(peek=True), "category": category,
                "op": str(op["id"]) if op else "",
                "op_arm": str((op.get("_arm") or {}).get("id") or "") if op else "", "pair": pair, "pair_why": pair_why,
                "alt": dataclasses.asdict(alt) if alt is not None else None, "alt_eligible": eligible,
                **({"bank": bank["id"]} if bank is not None else {})}

    def _alt_cands(self, v: View) -> dict[str, Any]:
        """The alternate bases of this view (``altbase.candidates``), once per view token."""
        from loop.core import altbase as AB
        tok = _view_token(v)
        got = self.__dict__.get("_altc")
        if got is None or got[0] != tok:
            with self.__dict__.setdefault("_altc_lock", threading.Lock()):
                got = self.__dict__.get("_altc")
                if got is None or got[0] != tok:
                    got = self._altc = (tok, AB.candidates(self.ctx, self.store, v, getattr(self, "_statuses", None)))
        return got[1]

    def _alt_pick(self, v: View) -> Any:
        """The alternate base of the next eligible session when its quota draw says alt-based (``alt_base_share``),
        else None. No side effect: :meth:`_reserve` books the draw."""
        from loop.core import altbase as AB
        try:
            st = AB.state(self.store)
            if not AB.due(int(st["n"]), float(self.opt("alt_base_share"))):
                return None
            pick = AB.choose(self.ctx, self.store, v, getattr(self, "_statuses", None), int(st["alt"]),
                             cands=self._alt_cands(v))
            if pick is None:                              # review fix M2: say so, never pretend the session is diverse
                self.store.log("alt_base_none", note=AB.NO_DIVERSE)
            return pick
        except Exception as exc:  # noqa: BLE001 - an alt-base fault falls back to the incumbent
            self.store.log("alt_base_error", error=repr(exc)[:300])
            return None

    def _branch_alt(self, v: View, br: str) -> Any:
        """A fork-and-flush branch that starts from an alt base (the LAST ``alt_base_fnf`` of the round's branches):
        its base, fixed for the round (chosen on the branch's first session); None for the other branches."""
        from loop.core import altbase as AB
        from loop.core import fnf as FN
        k = int(self.opt("alt_base_fnf"))
        rnd = (FN.state(self.store) or {}).get("open")
        if k <= 0 or not rnd or br not in AB.fnf_branches(rnd, k) or float(self.opt("alt_base_share")) <= 0:
            return None
        try:
            book = dict(self.store.get_state(AB.K_ALT_BRANCH) or {})
            mine = dict(book.get(str(rnd.get("id"))) or {})
            statuses = getattr(self, "_statuses", None)
            if (rec := mine.get(br)) and AB.sound(self.store, statuses, rec.get("node", "")):
                return AB.Pick(**rec)
            taken = [r.get("node") for b, r in mine.items() if b != br and r]
            pick = AB.choose(self.ctx, self.store, v, statuses, AB.fnf_branches(rnd, k).index(br),
                             exclude=taken, cands=self._alt_cands(v))
            if pick is not None:
                mine[br] = dataclasses.asdict(pick)
                self.store.set_state(AB.K_ALT_BRANCH, {str(rnd.get("id")): mine})
            else:
                self.store.log("alt_base_none", note=AB.NO_DIVERSE, branch=br)
            return pick
        except Exception as exc:  # noqa: BLE001
            self.store.log("alt_base_error", error=repr(exc)[:300])
            return None

    def bank_paths(self) -> tuple[str, ...]:
        """11-08 (``ideation.bank_v3``): the banks a v3 brief serves -- ``loop.idea_banks``, the v3 scout's bank
        (``scout.v3``) and the root-audit bank (loop.root_audit)."""
        from loop import localruns as LR
        from loop import root_audit as RA
        from loop import scout as SC
        cfg = self.ctx.cfg
        return tuple(dict.fromkeys([*(cfg.idea_banks or ()),
                                    *([SC.path(self.ctx)] if getattr(cfg.scout, "v3", False) else []),
                                    *([RA.bank_path(self.ctx)] if RA.enabled(self.ctx) else []),
                                    *([LR.bank_path(self.ctx)] if LR.banks(self.ctx) else [])]))

    def _bank_pick(self) -> dict[str, Any] | None:
        """The next unserved bank idea (priority first, families interleaved) not held by a rendered-but-unstarted
        brief; None when ``ideation.bank_v3`` is off or every bank is drained."""
        if not self.opt("bank_v3"):
            return None
        from types import SimpleNamespace
        from loop.select import bank as BK
        held = {str(j["plan"].get("bank")) for j in self._pending() if j.get("plan", {}).get("bank")}
        for e in BK.pending(self.store, SimpleNamespace(idea_banks=self.bank_paths())):
            if e["id"] not in held:
                return e
        return None

    def _reserve(self, plan: Mapping[str, Any]) -> None:
        """What a briefed plan holds (v3-c6i; :meth:`_unreserve` gives it back): the program quota count, the
        plateau hypothesis, an operator program, a readout session of the incumbent, the persona's turn."""
        self._count_program(plan.get("kind", ""))
        self._count_alt(plan)
        self._serve(plan.get("idx"))
        if plan.get("op"):
            self._op_mark(_op_key(plan))
        self._count_readout(plan.get("kind", ""), plan.get("inc", ""))
        if plan.get("persona"):
            self._persona()

    def code_pair(self, v: View, exclude: Sequence[str] = ()) -> tuple[str, str, str] | None:
        """``loop.core.pairs.next_pair(view) -> (a, b, why) | None`` (the tick binds its context each tick); None --
        the stack partner below -- when it returns nothing or raises. NO side effect (v3-c6r: a dry run / shadow /
        re-plan never burns a pair): the session's start claims it (:meth:`_claim_pair`). ``exclude``: pair keys
        queued briefs hold."""
        from loop.core import pairs as PR
        try:
            ac = self._alloc()
            if ac.enabled and ac.xover_conflict_only:      # v3-c7i I5: only a pair whose code graft conflicted
                from loop.core import alloc as AL
                got = AL.conflict_pair(v, self.store, tuple(exclude))
            else:
                got = PR.next_pair(v, mark=False, exclude=tuple(exclude))
        except Exception as exc:  # noqa: BLE001 - a pairing fault falls back to the stack partner
            logger.warning("pairs.next_pair raised (stack partner instead): %r", exc)
            return None
        if not got or len(got) < 2 or not got[0] or not got[1]:
            return None
        return str(got[0]), str(got[1]), str(got[2] if len(got) > 2 else "")

    def _held_pairs(self) -> list[str]:
        """Pair keys the queued briefs were planned on (not claimed yet: claimed at their session's start)."""
        from loop.core import pairs as PR
        return [PR.pair_key(*j["plan"]["pair"]) for j in self._jobs if (j.get("plan") or {}).get("pair")]

    def _claim_pair(self, plan: Mapping[str, Any], req_id: str) -> bool:
        """At the session's START (v3-c6r): the plan's code-picked pair claimed atomically under the store's write
        lock (``pairs.claim``); False when another session took it first (the plan is re-planned on the next pair).
        A plan without a pair: True."""
        pair = (plan or {}).get("pair")
        if (plan or {}).get("kind") == "literature" and not pair and self.opt("require_evidence"):
            from loop.agents.briefs import K_EVIDENCE
            book, n = list(self.store.get_state(K_EVIDENCE) or []), 0
            for e in book:
                if not e.get("served") and n < 5:
                    e["served"], n = req_id, n + 1
            if n:
                self.store.set_state(K_EVIDENCE, book)
        if (plan or {}).get("bank") and not pair:
            from loop.select import bank as BK
            BK.mark(self.store, str(plan["bank"]), req_id)
            self.store.log("bank_served", bank_id=str(plan["bank"]), request_id=req_id, kind=str(plan.get("kind") or ""))
        if not pair:
            return True
        from loop.core import pairs as PR
        ok = PR.claim(self.store, pair[0], pair[1], "brief", req_id, self.clock())
        if ok:
            self.store.log("pair_picked", a=pair[0], b=pair[1], request_id=req_id, why=str(plan.get("pair_why") or "")[:300])
        else:
            self.store.log("pair_taken", a=pair[0], b=pair[1], request_id=req_id)
        return ok

    def _release_pair(self, plan: Mapping[str, Any], req_id: str, why: str) -> None:
        pair = (plan or {}).get("pair")
        if pair:
            from loop.core import pairs as PR
            PR.release(self.store, pair[0], pair[1], why, req_id)

    def _readout_spans(self) -> str:
        cfg = getattr(self.ctx, "cfg", None)
        fam = getattr(getattr(cfg, "evalonly", None), "family", "") or "eval_readout"
        rule = (getattr(cfg, "families", None) or {}).get(fam)
        paths = list(getattr(rule, "paths", ()) or ())
        train = list(getattr(getattr(cfg, "evalonly", None), "train_spans", ()) or ())
        return ((f"the {fam} family's spans ({', '.join(paths[:8])})" if paths else f"the {fam} family's spans")
                + (f"; inside {', '.join(train)} (training runs them too) only under `if not self.training`"
                   if train else ""))

    def head_sections(self, plan: Mapping[str, Any], v: View) -> list[str]:
        """v3-c6i: what goes ABOVE the task head -- the triggered diagnostic summary -- and right below it: the
        persona lens and the mechanism coverage."""
        from loop.agents import briefs as BR      # 10-22: the operator's focus leads every brief (it was never rendered)
        return [x for x in (BR.focus_block(self.ctx).strip(), self.diag(v)[0],
                            *BR.research_blocks(self.ctx, self.store, kind=str(plan.get("kind") or ""))) if x]

    def lens_sections(self, plan: Mapping[str, Any], v: View) -> list[str]:
        from loop.core import mechanisms as MS
        out = [persona_text(str(plan.get("persona") or ""))]
        if plan.get("category"):
            _, counts = self._category(v, plan.get("live") or ())
            if counts is not None:
                out.append(MS.coverage_text(counts, plan["category"]))
        if (k := self.k_for(plan["kind"])) > 0:
            g, what = ref_gap(v)
            from loop.core import calibration as CAL
            out.append(CAL.brief_text(k, g, what))
        from loop.core import bold as BOLD
        if plan["kind"] in BOLD.slate_kinds(self.ctx) and plan["kind"] != BOLD.KIND:   # v3-c8b diverge-then-judge
            out.append(BOLD.slate_section(self.ctx, plan["kind"]))
        from loop.agents import briefs as BR
        return [x for x in (BR.findings_block(self.ctx), *out) if x]   # the operator's manual experiments lead

    def k_for(self, kind: str) -> int:
        """Candidates a session of ``kind`` lists (``ideation.k_candidates``; none for a port; v3-c8b: none for a kind
        whose session proposes its slate through ``propose_slate`` -- the slate carries the predictions)."""
        from loop.core import bold as BOLD
        return int(self.opt("k_candidates")) if kind in KINDS and kind not in BOLD.slate_kinds(self.ctx) else 0

    def hidden(self, kind: str) -> bool:
        """v3-c6i: a kind that sees no in-flight / recent idea pool (``ideation.hidden_pool_kinds``)."""
        return kind in tuple(self.opt("hidden_pool_kinds") or ())

    def brief_text(self, plan: Mapping[str, Any], data: Ledger, v: View) -> Brief:
        """The brief of ``plan`` (:meth:`brief_plan`): the computed sections (levers ~4-12 s once per view, engagement,
        facts, data bank, notes). Thread-safe against the tick: reads the store, fills per-view caches. v3-c6i: the
        diagnostic summary on top, the persona lens / coverage / candidate slate below the head; a hidden-pool kind
        gets the terse 'taken' list instead of the reserved section, no engagement of recent ideas, and only the
        re-test candidates pre-checked."""
        kind, lane, parents, axis, idx, head, rows, live = (plan[k] for k in (
            "kind", "lane", "parents", "axis", "idx", "head", "rows", "live"))
        inc = v.incumbent
        if kind == "bold":
            return self.bold_brief(plan, data, v)
        query = " ".join([head, AXIS_NAMES[axis], *(_tag(self.store, p) for p in parents)])
        levers, short = self.levers(v)
        hide = self.hidden(kind)
        if hide:
            short = self.__dict__.get("_short_rt", ())
            res = taken_text(live, self._queued())
        else:
            res = self.reserved_text(live)
        sections = [*self.head_sections(plan, v), head, *self.lens_sections(plan, v),
                    *[x for x in (self.cluster_section(), self.dedup_section(plan)) if x],     # v3-c7i I3 / I4
                    f"## Per-axis headroom (computed)\n{headroom_text(rows, inc)}",
                    decisive_text(v, float(self.opt("claim_gap_frac"))),
                    *([d] if (d := distance_text(v)) else []),
                    levers,
                    *([structural_block(v, self.store, self.ctx)] if kind == "program" else []),   # v3-c5 I4
                    *([res] if res else []),
                    *([] if hide else ["## Engagement of recent ideas (computed: did the declared diagnostic move "
                                       "after the first seed; not a quality signal)\n" + self._engagement_lines(data)]),
                    "## Computed facts (the ledger's measurements)\n" + self.facts(v),
                    "## Data bank (registered datasets, from their manifests)\n" + self._bank_lines(),
                    "## Literature notes (top 5 by similarity; cite by id, read more with `notes`)\n"
                    + self._notes_lines(query)]
        return Brief(kind, lane, parents, "\n\n".join(sections) + "\n", axis if kind == "repair" else "", idx,
                     shortlist=short, persona=str(plan.get("persona") or ""), category=str(plan.get("category") or ""),
                     arm=_arm(plan))

    def bold_brief(self, plan: Mapping[str, Any], data: Ledger, v: View) -> Brief:
        """v3-c8b: the BOLD brief (loop.core.bold) -- the task head with the structural taxonomy and the slate rule,
        this epoch's category coverage, the protected incumbent, the mechanism clusters (what is saturated), the data
        bank and the literature notes. Deliberately NO lever lists, G-* diagnostic flags, per-axis headroom rows,
        decisive-size or engagement anchoring, and no shortlist to pre-check."""
        from loop.core import bold as BOLD
        inc, head = v.incumbent, plan["head"]
        lv = v.levels.get(inc, {})
        n = min((k for _, k in lv.values()), default=0)
        score = sum(OFFICIAL_WEIGHTS[a] * lv[a][0] for a in OFFICIAL_WEIGHTS) if set(lv) >= set(OFFICIAL_WEIGHTS) \
            else None
        from loop.agents import briefs as BR
        sections = [BR.focus_block(self.ctx).strip(), *BR.research_blocks(self.ctx, self.store, kind=BOLD.KIND),
                    head, BR.findings_block(self.ctx),
                    "## Structural coverage this epoch (computed: research nodes per category)\n"
                    + _safe(lambda: BOLD.coverage_line(self.ctx, self.store, v.led), "(not computed)"),
                    f"## Protected incumbent\n{inc} is the incumbent"
                    + (f" (visible composite {score:.4f} over {n} seeds)" if score is not None else "")
                    + (f". This session builds on {plan['parents'][0]} instead (the starting point above); your node is "
                       f"credited with its own increment over {plan['parents'][0]} and judged against {inc} for the bar."
                       if plan.get("alt") else ". Build your change on it; your node is judged against it on fresh seeds."),
                    *[x for x in (self.cluster_section(),) if x],
                    "## Data bank (registered datasets, from their manifests)\n" + self._bank_lines(),
                    "## Literature notes (top 5 by similarity; cite by id, read more with `notes`)\n"
                    + self._notes_lines(head[:600])]
        return Brief(BOLD.KIND, plan["lane"], plan["parents"], "\n\n".join(x for x in sections if x) + "\n", "", None,
                     persona=str(plan.get("persona") or ""), arm=_arm(plan))

    def _queued(self) -> list[Any]:
        return sorted(self.store.nodes(status=[NodeStatus.QUEUED]) if hasattr(self.store, "nodes") else (),
                      key=lambda n: -(n.created_ts or 0.0))

    def levers(self, v: View) -> tuple[str, tuple[str, ...]]:
        """(the computed "untested levers & deficits" section, the mechanism shortlist the researcher pre-checks),
        once per view (~4 s on the 10-13 ledger)."""
        with self.__dict__.setdefault("_levers_lock", threading.Lock()):     # brief workers: computed once
            return self._levers_of(v)

    def _levers_of(self, v: View) -> tuple[str, tuple[str, ...]]:
        from loop.core import levers as LV
        if (got := self.__dict__.get("_levers")) is not None and got[0] == _view_token(v):
            return got[1]
        try:
            rt = LV.retest(v, self.store, self.ctx, self._statuses)
        except (KeyError, ValueError, ZeroDivisionError):
            rt = []
        rc = _safe(lambda: LV.repair_candidates(v, self.store, self.ctx, self._statuses), [])
        da = _safe(lambda: LV.diag_attempts(v, self.store, self.ctx), [])
        k = int(self.opt("precheck_recent"))
        extra = [(r["family"], r["tag"]) for r in rc[:k]]
        extra += [(sn.family, sn.mechanism_tag) for d in da for n in d["recent"][:3]
                  if (sn := self.store.get_node(n)) is not None]
        try:
            short = LV.shortlist(self.store, self.ctx, rt, extra)
        except Exception:  # noqa: BLE001 - no shortlist: no pre-check
            short = ()
        try:                                   # v3-c6i: the hidden-pool kinds pre-check the re-test candidates only
            self._short_rt = LV.shortlist(self.store, self.ctx, rt, [])
        except Exception:  # noqa: BLE001
            self._short_rt = ()
        out = (LV.text(v, self.store, self.ctx, self._statuses, rt, rc, da), short)
        self._levers = (_view_token(v), out)
        return out

    # -- fork-and-flush (loop.core.fnf) --------------------------------------------------------------------------
    def branch_brief(self, lane: str, v: View, kind: str, branch: str, plan: Mapping[str, Any] | None = None
                     ) -> Brief:
        """A FLUSHED brief: the task head, a protected-incumbent notice (its score), the decisive size, the COMPUTED
        per-axis headroom table and "untested levers & deficits" section (deficits, external / per-dataset headroom,
        compute, repair candidates: measurements, not the shared search's story -- 10-13 09Z MA: flushing them left
        program sessions blind, a node = 2 knobs of one family, +.003), the data bank, and only this branch's own
        history and notes. No shared notes, engagement, plateau hypotheses or lineage facts (``fnf.FLUSH``; the
        researcher drops the postmortems and the parent's handoffs). A plateau / crossover / repair kind falls back
        to explore."""
        from loop.core import fnf as FN
        kind = kind if kind in EXPLORE_KINDS else "explore"
        inc = v.incumbent
        head = _render("brief_literature", family="any bottleneck you find in the incumbent") \
            if kind == "literature" else KIND_TEXT[kind]
        lv = v.levels.get(inc, {})
        n = min((k for _, k in lv.values()), default=0)
        score = sum(OFFICIAL_WEIGHTS[a] * lv[a][0] for a in OFFICIAL_WEIGHTS) if set(lv) >= set(OFFICIAL_WEIGHTS) \
            else None
        alt = (plan or {}).get("alt")
        notice = (f"## Protected incumbent\n{inc} is the incumbent"
                  + (f" (visible composite {score:.4f} over {n} seeds)" if score is not None else "")
                  + (f". It is protected. This branch starts from {plan['parents'][0]} instead (see the starting point "
                     f"above): build your change on that; your node is credited with its own increment over it and "
                     f"judged against {inc} for the bar." if alt else
                     ". It is protected: build your change on it; your node is judged against it on fresh seeds."))
        if alt:
            from loop.core import altbase as AB
            head = AB.brief_note(v.led, self.store, AB.Pick(**alt), inc, _tag(self.store, alt["node"])) \
                + "\n\n" + AB.retarget(head, alt["node"])
        own = sorted((nid for nid, node in v.led.nodes.items() if FN.branch_of(node) == branch),
                     key=lambda x: v.led.nodes[x].created_ts)
        hist = []
        for nid in own:
            b = v.beliefs.get(nid)
            hist.append(f"- {nid} {_tag(self.store, nid)}: "
                        + (f"own gain {b.m:+.4f} over {b.n} seed(s)" if b else "not measured yet"))
            hist += [f"  handoff: did {ho.actions[:300]} | learned {ho.learnings[:300]} | concerns {ho.concerns[:300]}"
                     for ho in (self.store.handoffs(node_id=nid) if hasattr(self.store, "handoffs") else ())]
        query = " ".join([head, *(_tag(self.store, x) for x in own[-3:])])
        got = NT.retrieve(NT.notes_path(self.ctx, branch), query, 5, self.embedder)
        notes = "\n".join(f"- {x['id']} {x.get('title', '')[:90]}: {x.get('claim', '')[:200]} | transfer: "
                          f"{x.get('transfer', '')[:160]}" for x in got) or "(no notes in this branch yet)"
        levers, short = self.levers(v)
        plan = {"kind": kind, "persona": "", **dict(plan or {})}
        plan.pop("category", None)                    # a flushed branch is not pointed at the shared coverage
        sections = [*self.head_sections(plan, v), head, *self.lens_sections(plan, v), notice,
                    *[x for x in (self.dedup_section({**plan, "head": head}),) if x],          # v3-c7i I4
                    f"## Per-axis headroom (computed)\n{headroom_text(headroom(v, self.fm, self.bar_node), inc)}",
                    decisive_text(v, float(self.opt("claim_gap_frac"))),
                    *([d] if (d := distance_text(v)) else []),
                    levers,
                    *([structural_block(v, self.store, self.ctx)] if kind == "program" else []),   # v3-c5 I4
                    "## Data bank (registered datasets, from their manifests)\n" + self._bank_lines(),
                    f"## Your branch {branch} so far (its own sessions only)\n"
                    + ("\n".join(hist) or "(the first session of this branch)"),
                    f"## Literature notes of branch {branch} (top 5 by similarity)\n" + notes]
        return Brief(kind, lane, tuple(plan.get("parents") or (inc,)), "\n\n".join(sections) + "\n", "", None, branch,
                     FN.FLUSH, short, persona=str(plan.get("persona") or ""), arm=_arm(plan))

    def fork_round(self, v: View, why: str = "manual", open_new: bool = True) -> list[str]:
        """Close the open fork-and-flush round (selection on fresh-seed test credit; the winner's notes / handoffs
        merged, every branch dir archived) and fork K new branches from the current incumbent (none while the arm's
        share is 0, nor with ``open_new`` False: v3-c7i, the A/B shows no effect)."""
        from loop.core import fnf as FN
        st, now, acts = FN.state(self.store), self.clock(), []
        if st.get("open"):
            closed = FN.close(st, v, now)
            acts += FN.merge(self.ctx, self.store, closed)
            self.store.log("fnf_round_closed", round=closed["id"], winner=closed["winner"], credit=closed["credit"],
                           nodes=closed["nodes"], control=len(closed["control"]))
        if FN.SHARE > 0 and open_new:
            rnd = FN.open_round(st, v.led, v.judged, v.incumbent, why, now)
            self.store.log("fnf_round", round=rnd["id"], trigger=why, branches=rnd["branches"], cap=rnd["cap"],
                           incumbent=v.incumbent)
            acts.append(f"fnf round {rnd['id']} ({why}): {FN.K} flushed branches from {v.incumbent}, "
                        f"{rnd['cap']:.0f} GPU-h each")
        self.store.set_state(FN.K_FNF, st)
        return acts

    def _fnf(self, tick: Any, v: View) -> list[str]:
        from loop.core import alloc as AL
        from loop.core import fnf as FN
        st = FN.state(self.store)
        why = FN.trigger(v.led, v.judged, st, self.store.get_state(FN.K_DIV))
        if why is None:
            return []
        ac = self._alloc()                             # v3-c7i I6: rounds only while the A/B can still show an effect
        pause = AL.fnf_pause(v, st, int(ac.fnf_pause_rounds)) if ac.enabled else ""
        if pause:
            if self.store.get_state(AL.K_FNF_PAUSE) != pause:
                self.store.set_state(AL.K_FNF_PAUSE, pause)
                self.store.log("fnf_paused", why=pause, trigger=why)
            line = AL.quiet(self.store, "ideation.fnf_pause_line", f"fnf: paused ({pause})", float(self.clock()))   # 1/h (live 10-21: 3325 identical lines)
            if getattr(tick, "dry", False):
                return [f"fnf: paused ({pause}); would close the open round only" if st.get("open") else
                        f"fnf: paused ({pause})"]
            return (self.fork_round(v, why, open_new=False) if st.get("open") else []) + line
        if self.store.get_state(AL.K_FNF_PAUSE):
            self.store.set_state(AL.K_FNF_PAUSE, "")
            self.store.log("fnf_unpaused", trigger=why)
        if getattr(tick, "dry", False):
            return [f"fnf: would fork a round ({why})"]
        return self.fork_round(v, why)

    def facts(self, v: View) -> str:
        """The computed-facts block (axis trades, crash-censored regions, lineage coverage, harm-blocked), once per view."""
        if self._facts is None or self._facts[0] != _view_token(v):
            try:
                text = facts_text(v, self.store, self._statuses, self._harm)
            except (KeyError, ValueError, ZeroDivisionError) as exc:      # too little data for a fit: say so
                text = f"(not computed: {exc!r})"[:200]
            self._facts = (_view_token(v), text)
        return self._facts[1]

    def _bank_lines(self) -> str:
        from loop import databank as DB
        lines = DB.bank_lines(self.ctx, self.store) if getattr(self.ctx, "cfg", None) is not None else []
        if not lines:
            return "(none registered)"
        from loop.select.intake import family_of
        from loop.core import datasets as DS
        fam = family_of(self.ctx, "data.dataset_dir") or "the data family"
        measured = DS.lines(self.ctx, self.store) if hasattr(self.store, "get_state") else []
        return "\n".join(f"- {x}" for x in lines) + (
            "\nMeasured datasets (computed: every node that set data.dataset_dir, its own paired gain over its parent "
            "/ seeds; a DEAD dataset is rejected at intake unless [differs from: ...] answers its nodes):\n"
            + "\n".join(measured) if measured else "") + (
            f"\nTo train on one, set data.dataset_dir to its path (a config-only change of family {fam}; the path "
            "under the data-bank root is allowed). For a new one, declare dataset_spec in the finish: with no other "
            f"change the loop points data.dataset_dir at it and files the node as {fam}; it waits until the dataset "
            "is built.")

    def stack_partner(self, v: View, reserved: set[str] | frozenset[str] = frozenset()) -> str | None:
        """The best idea node (explore / exploit lane, >= 2 seeds, posterior mean > 1 sd above 0) outside the
        incumbent's ancestry, not yet stacked on it and not ``reserved`` (an in-flight crossover's or a queued
        node's second parent)."""
        anc, cur = set(), v.incumbent
        while cur is not None and cur not in anc:
            anc.add(cur)
            cur = v.led.nodes[cur].parent if cur in v.led.nodes else None
        done = {n.second_parent for n in v.led.nodes.values() if n.parent == v.incumbent and n.second_parent}
        cands = [n for n, b in v.beliefs.items() if b.m > math.sqrt(b.v) and b.n >= 2 and n not in anc | done | reserved
                 and v.led.nodes[n].lane in ("explore", "exploit")]   # an idea's own increment: no stack / point
        return max(cands, key=lambda n: (self.stack_value(v, n), v.beliefs[n].m, n)) if cands else None

    @staticmethod
    def stack_value(v: View, other: str) -> float:
        """The prior gain of stacking ``other`` onto the incumbent (``acquire.stack_over_parent``: sub-additive, no
        credit for a leg of the incumbent's mechanism -- so a cross-family partner wins over a stronger same-family
        one); an old-epoch partner: its shrunk own gain if it adds a mechanism, else 0."""
        from loop.core import acquire as A
        post, inc = v.post, v.incumbent
        if post is None:
            return v.beliefs[other].m
        try:
            if (sm := A.stack_over_parent(v.led, post, inc, other)) is not None:
                return sm
            return A.FRESH_SHRINK * v.beliefs[other].m if A.adds_mechanism(v.led, inc, other) else 0.0
        except (KeyError, ValueError, ZeroDivisionError):
            return v.beliefs[other].m

    # -- spreading concurrent sessions (v3-c4i) -----------------------------------------------------------------
    def inflight(self) -> list[dict[str, Any]]:
        """The research sessions in flight (``driver.inflight``; queued restart re-issues left out): id, brief kind,
        axis (a repair's family hint), parents, and the mechanism its ``check`` claimed (``K_CLAIMS``)."""
        from loop.contracts import K_INFLIGHT
        claims = self.store.get_state(K_CLAIMS) or {}
        out = []
        for e in self.store.get_state(K_INFLIGHT) or []:
            if e.get("queued"):
                continue
            r = e.get("req") or {}
            out.append({"id": e.get("id", ""), "kind": r.get("brief_kind", "") if isinstance(r, Mapping) else "",
                        "axis": (r.get("family_hint") if isinstance(r, Mapping) else "") or e.get("family", ""),
                        "parents": list(e.get("parents") or ()), "claim": claims.get(e.get("id", "")),
                        "category": (r.get("category") or "") if isinstance(r, Mapping) else ""})
        for i, j in enumerate(self.__dict__.get("_jobs") or ()):     # R4: enqueued briefs hold their decisions
            p = j["plan"]
            out.append({"id": f"queued-{i + 1}", "kind": p["kind"], "axis": p["axis"] if p["kind"] == "repair" else "",
                        "parents": list(p["parents"]), "claim": None, "category": p.get("category", "")})
        return out

    def reserved_partners(self, live: Sequence[Mapping[str, Any]]) -> set[str]:
        """Second parents an in-flight crossover session or a queued node already stacks."""
        out = {p for e in live if e["kind"] == "crossover" for p in e["parents"][1:]}
        for n in self.store.nodes(status=[NodeStatus.QUEUED]) if hasattr(self.store, "nodes") else ():
            if n.second_parent_id:
                out.add(n.second_parent_id)
        return out

    def reserved_text(self, live: Sequence[Mapping[str, Any]], k: int = 12) -> str:
        """The brief section listing what in-flight sessions and queued nodes already work on ('' when nothing):
        mechanisms (``check`` claims, queued nodes' tags), axes, stack partners and datasets."""
        from loop import databank as DB
        lines = []
        for e in live:
            c = e.get("claim") or {}
            lines.append(f"- in flight {e['id']}: {e['kind'] or '?'}"
                         + (f" on {AXIS_NAMES.get(e['axis'], e['axis'])}" if e["kind"] == "repair" and e["axis"] else "")
                         + (f", stacking {', '.join(e['parents'][1:])}" if e["kind"] == "crossover" else "")
                         + (f", building {c.get('family')}/{c.get('tag')}" if c.get("tag") else ""))
        queued = sorted(self.store.nodes(status=[NodeStatus.QUEUED]) if hasattr(self.store, "nodes") else (),
                        key=lambda n: -(n.created_ts or 0.0))
        for n in queued[:k]:
            ds = DB.dataset_of(self.ctx, (self.store.get_state(f"config_diff:{n.id}") or {}).get("data.dataset_dir")) \
                if getattr(self.ctx, "cfg", None) is not None and getattr(self.ctx.cfg, "databank", None) is not None \
                else ""
            lines.append(f"- queued {n.id}: {n.family}/{n.mechanism_tag}" + (f" (dataset {ds})" if ds else ""))
        if not lines:
            return ""
        return ("## Reserved: in flight and queued (computed; another session already works on these -- pick a "
                "different mechanism, axis, partner or dataset; `check` rejects a mechanism an in-flight session "
                "claimed)\n" + "\n".join(lines[:2 * k]))

    def _engagement_lines(self, data: Ledger, k: int = 8) -> str:
        res = [e for e in self._engagements(data) if e.status in MEASURED]
        res.sort(key=lambda e: -data.nodes[e.node].created_ts)
        lines = [f"- {e.node} {_tag(self.store, e.node)}: {e.diagnostic} {'up' if e.direction > 0 else 'down'} -> "
                 f"{e.status} (z {e.z:+.1f} along the prediction, {e.n} seed(s))" for e in res[:k]]
        if not res:
            return "(no measured prediction yet)"
        # v3-c7i I9: engagement is NOT a quality signal (10-14: engaged 11% positive vs not engaged 40%); a reversal
        # after the first seed is an early-kill signal -- the computed check, each verdict's share of positive nodes
        from loop.core import alloc as AL
        pr = AL.predictive(res, {n: (b.m, b.n) for n, b in (self._view[1].beliefs.items() if self._view else ())})
        check = "; ".join(f"{k} {a}/{b} positive" for k, (a, b) in sorted(pr.items()))
        return "\n".join(lines + [f"Does engagement predict gain? {check or 'not measured yet'}. It is NOT evidence that an "
                                   "idea works: aim at the panel, not at moving the diagnostic. A reversed diagnostic "
                                   "after the first seed with a negative first seed ends the node's screening."])

    def _notes_lines(self, query: str) -> str:
        got = NT.retrieve(NT.notes_path(self.ctx), query, 5, self.embedder)
        return "\n".join(f"- {n['id']} {n.get('title', '')[:90]}: {n.get('claim', '')[:200]} | transfer: "
                         f"{n.get('transfer', '')[:160]}" for n in got) or "(no notes yet)"

    # -- plateau review ----------------------------------------------------------------------------------------
    def _first_kind(self, ahead: int = 0) -> str | None:
        """'plateau' while the plateau review has an unserved hypothesis ``ahead`` places on: its briefs take the
        first LLM slots, whatever lane the slot was ranked for (None: the lane decides)."""
        st = self.store.get_state(K_PLATEAU) or {}
        n, served = len(st.get("hypotheses") or []), int(st.get("served") or 0)
        back = len(st.get("returned") or ())
        return "kuhn" if st.get("active") and served - back + ahead < n else None

    def _plateau_next(self) -> tuple[int, dict] | None:
        st = self.store.get_state(K_PLATEAU) or {}
        hyps, served = st.get("hypotheses") or [], int(st.get("served") or 0)
        back = [int(i) for i in st.get("returned") or () if 0 <= int(i) < len(hyps)]
        if st.get("active") and back:                  # a hypothesis whose brief was dropped unstarted (S7)
            return back[0], hyps[back[0]]
        return (served, hyps[served]) if st.get("active") and served < len(hyps) else None

    def review_prompt(self, v: View, p: Plateau, statuses: Mapping[str, Any]) -> str:
        lanes: dict[str, list[float]] = {}
        for n, b in v.beliefs.items():
            lanes.setdefault(v.led.nodes[n].lane, []).append(b.m)
        cr = credit(v)
        from loop.agents import briefs as BR
        return BR.focus_block(self.ctx) + "".join(x + "\n\n" for x in BR.research_blocks(self.ctx, self.store)) + _render(
            "plateau", task_id=getattr(self.ctx, "task_id", ""), incumbent=v.incumbent,
            plateau=f"no PASS for {p.gpu_h:.0f} GPU-h and {p.idle_h:.0f} h (median between past passes {p.median:.0f}; "
                    f"{p.passes} passes; trigger {p.why or 'manual'})",
            headroom=headroom_text(headroom(v, self.fm, self.bar_node), v.incumbent),
            diagnostics=DG.summary(self.store, v.incumbent) or "(none)",
            lanes="\n".join(f"- {l}: {len(ms)} nodes, mean own gain {statistics.fmean(ms):+.4f}, "
                            f"{sum(m > 0 for m in ms)} positive" for l, ms in sorted(lanes.items())),
            credit="\n".join(f"- {s}: {c['tested']:.0f}/{c['nodes']:.0f} tested, gain {c['gain']:.4f}, "
                             f"{c['gpu_h']:.0f} GPU-h" for s, c in sorted(cr.items(), key=lambda x: -x[1]["gain"])[:12]),
            clusters="\n".join(f"- {c}" for c in clusters(self.store, negatives(v, statuses))) or "(none)",
            engagement=self._engagement_lines(v.led, 12), mde=f"{v.mde():.4f}",
            quantities=", ".join(sorted(DG.quantities())), facts=self.facts(v))

    def _plateau(self, tick: Any, v: View, statuses: Mapping[str, Any]) -> list[str]:
        p, st, now = plateau(v.led, v.judged), self.store.get_state(K_PLATEAU) or {}, self.clock()
        if self._review is not None and self._review.done():
            fut, self._review = self._review, None
            try:
                out = dict(fut.result())
            except Exception as exc:  # noqa: BLE001 - a failed review is retried after REVIEW_EVERY_H
                self.store.log("plateau_review_error", error=repr(exc)[:500])
                return [f"plateau review failed: {exc!r}"[:200]]
            self.store.set_state(K_PLATEAU, {"active": True, "since": p.since, "ts": now, "served": 0,
                                             "inputs": PLATEAU_INPUTS, **out})
            self.store.log("plateau_review", since=p.since, n=len(out.get("hypotheses") or []))
            return [f"plateau review: {len(out.get('hypotheses') or [])} hypotheses"]
        if not p.active:
            if st.get("active"):
                self.store.set_state(K_PLATEAU, {**st, "active": False})
            return []
        llm = self.llm
        # a review on older inputs (no computed facts) is redone at once -- at tick 0 after a deploy
        fresh = st.get("since") == p.since and st.get("inputs") == PLATEAU_INPUTS and (
            self._plateau_next() is not None or now - float(st.get("ts") or 0) < REVIEW_EVERY_H * 3600)
        if self._review is not None or fresh or llm is None or getattr(tick, "dry", False):
            return []
        prompt = self.review_prompt(v, p, statuses)
        self._review = tick.pool.submit(llm.complete_json, "review", system=(
            "You diagnose why an automated ML search stopped improving. Every number you cite must appear in the "
            "input."), prompt=prompt, schema=REVIEW_SCHEMA)
        self.store.set_state(K_PLATEAU, {**st, "since": p.since, "ts": now, "inputs": PLATEAU_INPUTS})
        return [f"plateau ({p.why}): {p.gpu_h:.0f} GPU-h and {p.idle_h:.0f} h since the last pass (median {p.median:.0f}); "
                "review started"]

    # -- the hooks ---------------------------------------------------------------------------------------------
    def ideate(self, tick: Any, ideas: Sequence[Any], data: Ledger, statuses: Mapping[str, Any]) -> list[str]:
        acts: list[str] = []
        lap = _laps(tick)              # v3-c6a hook: sub-phase seconds -> tick.phase_s["ideate.<part>"]
        self.llm = self.llm or getattr(getattr(tick.research, "researcher", None), "llm", None)
        self._tick_pool = getattr(tick, "pool", None)
        if getattr(getattr(self.ctx, "cfg", None), "diagnostics", None) is not None and self.ctx.cfg.diagnostics.enabled:
            if self._async(tick):                                    # R4: off the tick, one at a time
                if self._ingest is None or self._ingest.done():
                    self._ingest = self._executor().submit(self._ingest_job)
            else:
                self._ingest_job()
        lap("ingest")
        try:
            v = self.view(data, getattr(tick, "post", None))
        except (ValueError, KeyError, ZeroDivisionError) as exc:     # too little data to fit the noise yet
            return acts + [f"ideation: no view yet ({exc})"[:200]]
        lap("view")
        try:                                                         # v3-c9 C1: the history tool's view (visible ledger)
            from loop.agents import history as HX
            HX.remember(self.ctx, self.store, v, statuses)
        except Exception as exc:  # noqa: BLE001 - never stops ideation
            self.store.log("history_error", error=repr(exc)[:300])
        try:                                                         # what the researcher's diagnose tool reads
            from loop.agents import diagnose as DX
            sn = self.store.get_node(v.incumbent)
            DX.record(self.ctx, self.store, data, v.incumbent, getattr(sn, "commit", "") or "", None)
            # v3-c5 I2/I5c: the breakdown is precomputed in record; the feature job is submitted once per incumbent
            if got := DX.precompute_features(self.ctx, self.store,
                                             getattr(getattr(tick.research, "researcher", None), "adapter", None),
                                             self._executor() if self._async(tick) else None):   # S8: off the tick
                acts.append(got)
        except Exception as exc:  # noqa: BLE001 - never stops ideation
            self.store.log("diagnose_error", error=repr(exc)[:300])
        lap("record")
        neg = negatives(v, statuses)
        if self.store.get_state(K_NEGATIVE) != neg:
            self.store.set_state(K_NEGATIVE, neg)
        try:                                                         # v3-c8a B2: the per-dataset table intake reads
            from loop.core import datasets as DS
            if self.__dict__.get("_ds_tok") != (tok := _view_token(v)):     # v3 review N1: once per view
                DS.record(self.ctx, self.store, v.beliefs, v.led, v.incumbent)
                self._ds_tok = tok
        except Exception as exc:  # noqa: BLE001 - never stops ideation
            self.store.log("datasets_error", error=repr(exc)[:300])
        self._statuses, self._harm = statuses, dict(getattr(getattr(tick, "core", None), "harm", None) or {})
        try:                                                         # the data bank the briefs list (cheap reads)
            from loop import databank as DB
            acts += DB.register(self.ctx, self.store) if getattr(self.ctx, "cfg", None) is not None else []
        except Exception as exc:  # noqa: BLE001 - the bank never stops ideation
            self.store.log("databank_error", error=repr(exc)[:500])
        lap("register")
        acts += self._log_engagement(data)
        try:                                                         # v3-c7i I9: reversed + negative first seed
            acts += self._early_kill(tick, data, v)
        except Exception as exc:  # noqa: BLE001 - never stops ideation
            self.store.log("early_kill_error", error=repr(exc)[:300])
        lap("engagement")
        try:                                                         # 10-22: a computed postmortem per closed node
            from loop.core import closed_pm as CPM
            acts += CPM.write(self.ctx, self.store, v, self._engagements(data), self.clock())
        except Exception as exc:  # noqa: BLE001 - never stops ideation
            self.store.log("closed_pm_error", error=repr(exc)[:300])
        for part in (lambda: self._calibrate(v), lambda: self._ps_tick(tick, v)):     # v3-c6i
            try:
                acts += part()
            except Exception as exc:  # noqa: BLE001 - scoring never stops ideation
                self.store.log("calibration_error", error=repr(exc)[:300])
        acts += self._plateau(tick, v, statuses)
        lap("plateau")
        try:                                                         # fork-and-flush rounds (loop.core.fnf)
            acts += self._fnf(tick, v)
        except Exception as exc:  # noqa: BLE001 - the arm never stops ideation
            self.store.log("fnf_error", error=repr(exc)[:500])
            acts.append(f"fnf: error {exc!r}"[:200])
        lap("fnf")
        try:                                                         # v3-c7i: clusters, throttle, crossover cases
            acts += self._clusters(v)
            ideas = self._admit_ideas(tick, ideas, data, v, statuses, acts)
        except Exception as exc:  # noqa: BLE001 - allocation never stops ideation
            self.store.log("alloc_error", error=repr(exc)[:500])
            acts.append(f"alloc: error {exc!r}"[:200])
        lap("alloc")
        self._started = True
        self._tick_kinds, self._dry = [], bool(getattr(tick, "dry", False))
        first = (lambda i: None) if self.weighted() else self._first_kind   # weighted: plateau by its yield
        over = getattr(tick, "over_budget", None) or (lambda: False)      # v3 review O2: the tick's time box
        if getattr(tick, "dry", False):                              # shadow: what it would ask, nothing started
            for i, a in enumerate(ideas):
                if over():
                    acts.append(f"ideation: tick budget spent, {len(ideas) - i} brief(s) not rendered")
                    break
                b = self.brief_for(a.lane, data, v, first(i))
                acts.append(f"would brief {b.kind} on {','.join(b.parents)}")
            return acts
        if self._async(tick):                    # settle, enqueue, then start what already rendered (fast / inline)
            acts += self._settle(tick, data, v)
            acts += self._enqueue(tick, ideas, data, v)
            acts += self._settle(tick, data, v)
            lap("settle")
            return acts
        for a in list(ideas)[:tick.research.free()]:
            if over():                                               # the rest: next tick (re-ranked)
                acts.append("ideation: tick budget spent, the remaining briefs wait for the next tick")
                break
            kind = (self._kind(a.lane, v) if self._trickle     # 11-09: a throttle-trickle slot's kind
                    else first(0) or self.kind_for(a.lane, v))  # unweighted: the plateau's hypotheses go first
            plan = self._plan(a.lane, v, kind, self._branch(kind, data))
            b = self.branch_brief(plan["lane"], v, plan["kind"], plan["branch"], plan) if plan.get("branch") \
                else self.brief_text(plan, data, v)
            self._reserve(plan)
            req = self.request(b, v)
            if not self._claim_pair(plan, req.id):           # v3-c6r: taken since it was planned: no pair this time
                self._unreserve(plan)
                continue
            tick.research.start(req, None, lane=b.lane, axis=b.axis, **({"branch": b.branch} if b.branch else {}))
            acts.append(f"researcher started: {req.id} {b.kind} on {','.join(b.parents)}"
                        + (f" (fnf branch {b.branch}, flushed)" if b.branch else ""))
        return acts

    # -- v3-c7i: budget allocation (loop.core.alloc) -----------------------------------------------------------
    def _clusters(self, v: View) -> list[str]:
        """The mechanism clusters of this view (I3), once per view: kept for the briefs and for ``check_idea`` /
        intake (``alloc.LAST``)."""
        ac = self._alloc()
        if not (ac.enabled and ac.clusters):
            return []
        from loop.core import alloc as AL
        tok = _view_token(v)
        if (got := self.__dict__.get("_cls")) is not None and got[0] == tok:
            return []
        passed = [j.node for j in v.judged if str(getattr(j.verdict.kind, "value", j.verdict.kind)) == "pass"]
        cls = AL.clusters(self.ctx, self.store, v.led, {n: (b.m, b.n) for n, b in v.beliefs.items()},
                          protect=AL.protected(v.led, v.incumbent, passed))
        AL.remember(cls, self.store, v.led, int(ac.cluster_window))
        self._cls = (tok, cls)
        sat = [c.id for c in cls if c.flag == "saturated-negative"]
        key = {"n": len(cls), "saturated": sat, "boost": [c.id for c in cls if c.flag == "boost"][:10]}
        if self.store.get_state("alloc.clusters") != key:
            self.store.set_state("alloc.clusters", key)
            self.store.log("clusters", n=len(cls), saturated=[(c.id, c.label, len(c.members), round(c.mean, 5))
                                                               for c in cls if c.flag == "saturated-negative"][:20])
            return [f"clusters: {len(cls)} ({len(sat)} saturated-negative: {', '.join(sat[:8])})"]
        return []

    def cluster_section(self) -> str:
        got = self.__dict__.get("_cls")
        if not got:
            return ""
        from loop.core import alloc as AL
        return AL.cluster_text(got[1], int(self._alloc().cluster_rows))

    def dedup_section(self, plan: Mapping[str, Any]) -> str:
        """I4: the dead / in-flight mechanisms nearest this brief's focus (its head, category, axis, persona)."""
        ac = self._alloc()
        if not ac.enabled or int(ac.dedup_k) <= 0:
            return ""
        from loop.core import alloc as AL
        focus = " ".join(str(x) for x in (plan.get("category") or "", AXIS_NAMES.get(str(plan.get("axis") or ""), ""),
                                          plan.get("persona") or "", str(plan.get("head") or "")[:400],
                                          *(_tag(self.store, p) for p in plan.get("parents") or ()
                                            if p != plan.get("inc"))))
        return _safe(lambda: AL.dedup_text(self.ctx, self.store, focus, int(ac.dedup_k)), "")

    def _xover_conflict_only(self) -> bool:
        """I5: the LLM crossover brief is only for a code-picked pair whose code graft conflicted
        (``alloc.xover_conflict_only``); without one the slot briefs explore -- plain stacking is the planner's."""
        ac = self._alloc()
        return bool(ac.enabled and ac.xover_conflict_only)

    def _admit_ideas(self, tick: Any, ideas: Sequence[Any], data: Ledger, v: View, statuses: Mapping[str, Any],
                     acts: list[str]) -> list[Any]:
        """I2: the idea slots that may brief a session now -- none while the QUEUED backlog throttles issuance
        (logged on change), or (11-09, ``alloc.backlog_trickle``) a trickle: explore-lane slots while fewer than
        ``backlog_trickle`` sessions are in flight / being briefed, each briefed as the next ``backlog_trickle_kinds``
        kind (:meth:`_kind`)."""
        from loop.core import alloc as AL
        ac = self._alloc()
        self._trickle = False
        if not ac.enabled or not ideas:
            return list(ideas)
        bl = AL.backlog(data, statuses, self.clock(), AL.screenable(getattr(tick, "core", None)))
        why = AL.throttle(ac, bl)
        if self.store.get_state(AL.K_THROTTLE) != why:
            self.store.set_state(AL.K_THROTTLE, why)
            self.store.log("brief_throttle", why=why, queued=bl["n"], median_age_h=round(bl["median_age_h"], 2),
                           screenable=bl.get("n_screenable"))
        if alert := AL.throttle_alert(self.store, why, self.clock()):       # v3 review O6
            acts.append(alert)
        if why:
            busy = len(getattr(getattr(tick, "research", None), "inflight", None) or ()) + len(self._jobs)
            room = AL.trickle_room(ac, why, busy)
            got = [a for a in ideas if LANE_KIND.get(a.lane, "explore") == "explore"][:room]
            self._trickle = bool(got)
            acts.append(f"briefs throttled ({why}): {len(ideas) - len(got)} idea slot(s) left to the GPU queue"
                        + (f", {len(got)} trickle ({'/'.join(ac.backlog_trickle_kinds)}; "
                           f"<= {ac.backlog_trickle} in flight, {busy} now)" if ac.backlog_trickle else ""))
            return got
        return list(ideas)

    def _kind(self, lane: str, v: View, i: int = 0) -> str:
        """A slot's brief kind: the throttle trickle's next kind (:func:`alloc.trickle_kind`) on a trickle tick, else
        the weighted / rotation kind."""
        if self._trickle:
            from loop.core import alloc as AL
            if k := AL.trickle_kind(self._alloc(), self.store):
                return k
        return self.kind_for(lane, v) if self.weighted() else (self._first_kind(i) or self.kind_for(lane, v))

    def _early_kill(self, tick: Any, data: Ledger, v: View) -> list[str]:
        """I9: a node whose declared diagnostic REVERSED after its first seed (z < -2) and whose paired delta at that
        seed is below -sd_d (v3 review K9) gets no more screen seeds; a later finished paired seed that brings its
        paired mean to >= 0 lifts the kill (state ``alloc.early_kill``, read by the core's rank) and its running
        screen runs are cancelled (``tick.book.cancel_run``; shadow: listed only)."""
        ac = self._alloc()
        if not (ac.enabled and ac.early_kill):
            return []
        from loop.core import alloc as AL
        from loop.core import model as M
        done = dict(self.store.get_state(AL.K_EARLY) or {})
        acts, dry = [], getattr(tick, "dry", False)
        for nid, m, k in AL.early_lifts(data, v.weights, done):    # v3 review K9: a later paired seed lifts it
            if dry:
                acts.append(f"early kill: would lift {nid} (paired mean {m:+.4f} on {k} seeds)")
                continue
            done.pop(nid, None)
            self.store.set_state(AL.K_EARLY, done)
            self.store.log("early_kill_lifted", node_id=nid, mean=m, seeds=k)
            acts.append(f"early kill lifted: {nid} (paired mean {m:+.4f} on {k} seeds >= 0)")
        try:
            sd_d = v.post.fit(data.epoch, True)[0].sd_d(M.COMPOSITE)
        except Exception:  # noqa: BLE001 - no noise fit: one seed's noise unknown, a negative delta suffices
            sd_d = None
        rows = AL.early_kills(self._engagements(data), data, v.weights, done, sd_d)
        from loop.core import struct_tune as STT           # 10-22: no close (not even an early one) before the ladder
        owed = STT.held(self.ctx, self.store, data, post=getattr(v, "post", None))
        rows = [r for r in rows if r[0] not in owed]
        for nid, z, d in rows:
            if dry:
                acts.append(f"early kill: would stop {nid} (diagnostic reversed, z {z:+.1f}; first-seed delta {d:+.4f})")
                continue
            done[nid] = {"z": round(z, 3), "delta": round(d, 6), "ts": self.clock()}
            self.store.set_state(AL.K_EARLY, done)
            self.store.log("early_kill", node_id=nid, z=z, delta=d)
            cancelled = 0
            book = getattr(tick, "book", None)
            if book is not None and hasattr(book, "cancel_run") and hasattr(self.store, "runs"):
                from loop.contracts import RunStatus, Stage
                for r in self.store.runs(node_id=nid, status=[RunStatus.RUNNING]):
                    if r.stage in (Stage.SCREEN, Stage.PROMOTE) and book.cancel_run(r, f"early kill: {nid} reversed"):
                        cancelled += 1
            acts.append(f"early kill {nid}: diagnostic reversed (z {z:+.1f}) and first-seed delta {d:+.4f}; no more "
                        f"screen seeds ({cancelled} running cancelled)")
        return acts

    # -- R4: briefs rendered off the tick thread ----------------------------------------------------------------
    def _async(self, tick: Any) -> bool:
        return not getattr(tick, "dry", False) and int(self.opt("brief_workers")) > 0

    def _executor(self) -> ThreadPoolExecutor:
        if self._pool is None:
            n = max(1, int(self.opt("brief_workers")))
            cls = self._executor_cls or (type(self._tick_pool) if hasattr(self._tick_pool, "submit")
                                         else ThreadPoolExecutor)
            self._pool = ThreadPoolExecutor(max_workers=n, thread_name_prefix="brief") \
                if cls is ThreadPoolExecutor else cls(max_workers=n)
        return self._pool

    def _ingest_job(self) -> None:
        try:
            DG.ingest(self.ctx, self.store)
        except Exception as exc:  # noqa: BLE001 - diagnostics never stop the loop
            self.store.log("diagnostics_error", error=repr(exc)[:500])

    def _pending(self) -> list[dict[str, Any]]:
        """The enqueued briefs (rendering, or rendered and waiting for a session slot)."""
        return list(self._jobs)

    def _branch(self, kind: str, data: Ledger) -> str:
        """The fork-and-flush branch an explore-kind slot is assigned to ('' = the shared context)."""
        from loop.core import fnf as FN
        if kind not in EXPLORE_KINDS:
            return ""
        st = FN.state(self.store)
        if not st.get("open"):
            return ""
        br = FN.assign(st, data, "explore")
        self.store.set_state(FN.K_FNF, st)
        return br

    def _plan(self, lane: str, v: View, kind: str, br: str, plateau_at: int | None = None,
              keep: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if br:                                    # a flushed branch brief: its kind falls back as branch_brief does
            alt = self._branch_alt(v, br)         # v3-c9 C2: an alt-base branch starts from its round's alternate base
            return {"kind": kind if kind in EXPLORE_KINDS else "explore", "lane": lane, "branch": br,
                    "parents": (alt.node,) if alt is not None else (v.incumbent,), "axis": "", "idx": None,
                    "inc": v.incumbent, "alt": dataclasses.asdict(alt) if alt is not None else None,
                    "persona": str((keep or {}).get("persona") or "") or self._persona(peek=True)}
        return self.brief_plan(lane, v, kind, plateau_at, keep)

    def _render_job(self, plan: Mapping[str, Any], data: Ledger, v: View) -> Brief:
        t0 = time.monotonic()
        b = self.branch_brief(plan["lane"], v, plan["kind"], plan["branch"], plan) if plan.get("branch") \
            else self.brief_text(plan, data, v)
        self.store.log("brief_rendered", kind=b.kind, lane=b.lane, branch=b.branch, s=round(time.monotonic() - t0, 2))
        return b

    def _enqueue(self, tick: Any, ideas: Sequence[Any], data: Ledger, v: View) -> list[str]:
        """One render job per free session slot not already held by an enqueued brief (never two for one slot across
        ticks). The decisions are made here, in order (kind, branch, crossover partner, repair axis, plateau
        hypothesis -- reserved at once: ``served`` moves now), so the jobs can render in parallel."""
        acts = []
        free = tick.research.free() - len(self._jobs)
        for a in list(ideas)[:max(0, free)]:
            kind = self._kind(a.lane, v)
            plan = self._plan(a.lane, v, kind, self._branch(kind, data))
            self._reserve(plan)
            self._jobs.append(self._submit(plan, data, v))
            acts.append(f"brief queued: {plan['kind']} on {','.join(plan['parents'])}"
                        + (f" (fnf branch {plan['branch']})" if plan.get("branch") else ""))
        return acts

    def _submit(self, plan: Mapping[str, Any], data: Ledger, v: View) -> dict[str, Any]:
        return {"fut": self._executor().submit(self._render_job, plan, data, v), "plan": plan, "inc": v.incumbent,
                "ts": self.clock(), "tok": _view_token(v), "epoch": getattr(v.led, "epoch", "")}

    def _settle(self, tick: Any, data: Ledger, v: View) -> list[str]:
        """The rendered briefs start while session slots are free (oldest first; the rest wait for the next tick);
        a failed render is logged and its slot freed; one rendered for a former incumbent is re-planned on this
        view (same kind; a plateau keeps its reserved hypothesis)."""
        acts, free = [], tick.research.free()
        ttl = float(self.opt("brief_ttl_s") or 0.0)
        tok, now, epoch = (_view_token(v) if ttl > 0 else None), self.clock(), getattr(v.led, "epoch", "")
        for j in list(self._jobs):
            fut, plan = j["fut"], j["plan"]
            if not fut.done():
                continue
            if fut.cancelled() or (exc := fut.exception()) is not None:
                self._drop(j, plan, "cancelled" if fut.cancelled() else repr(exc)[:500], acts)
                continue
            b = fut.result()
            # A rendered brief is re-planned only when its MEANING changed: the incumbent switched, or the base epoch did
            # (10-22: 198 of 319 rendered briefs, 62%, were discarded in 7 days -- 102 "view" (the view token moves on every
            # new observation) and 96 "ttl"; 95 of those were two briefs re-rendered every 30 min for 48 h while the
            # operator's LLM pause held every session slot). A brief that is merely WAITING for a session slot is never
            # re-rendered: its numbers move by noise, not by meaning. Once a slot is free, a brief older than
            # BRIEF_TTL_X x ideation.brief_ttl_s whose view moved is re-planned (stale numbers, not a stale meaning).
            age = now - float(j.get("ts") or now)
            why = ("incumbent" if j["inc"] != v.incumbent else
                   "epoch" if j.get("epoch", epoch) != epoch else
                   "ttl" if ttl > 0 and free > 0 and age > ttl * BRIEF_TTL_X and j.get("tok") != tok else "")
            if why:                                    # re-planned without itself in the reserved list (a partner)
                self._jobs.remove(j)
                self.store.log("brief_stale", kind=b.kind, was=j["inc"], incumbent=v.incumbent, why=why,
                               age_s=round(age, 1))
                try:
                    newplan = self._plan(plan["lane"], v, plan["kind"], plan.get("branch", ""), plan["idx"], plan)
                    nj = self._submit(newplan, data, v)
                except Exception as exc:  # noqa: BLE001 - a failed re-plan frees the slot, logged
                    self._drop(None, plan, f"re-plan: {exc!r}"[:500], acts)
                    continue
                self._count_alt(plan, -1)                  # the alt-base quota follows the plan that holds the slot
                self._count_alt(newplan)
                self._jobs.append(nj)
                continue
            if free <= 0:
                continue
            self._jobs.remove(j)                       # removed BEFORE the start: a raising start never starts twice
            req = None
            try:
                req = self.request(b, v)
                if not self._claim_pair(plan, req.id):     # v3-c6r: another session took its pair: the next pair
                    self.store.log("brief_stale", kind=b.kind, was=j["inc"], incumbent=v.incumbent, why="pair_taken",
                                   age_s=round(age, 1))
                    newplan = self._plan(plan["lane"], v, plan["kind"], plan.get("branch", ""), plan["idx"], plan)
                    self._count_alt(plan, -1)
                    self._count_alt(newplan)
                    self._jobs.append(self._submit(newplan, data, v))
                    continue
                tick.research.start(req, None, lane=b.lane, axis=b.axis, **({"branch": b.branch} if b.branch else {}))
            except Exception as exc:  # noqa: BLE001 - one bad brief never stops the tick; logged, slot freed
                if req is not None:
                    self._release_pair(plan, req.id, f"start: {exc!r}")
                self._drop(None, plan, f"start: {exc!r}"[:500], acts)
                continue
            free -= 1
            acts.append(f"researcher started: {req.id} {b.kind} on {','.join(b.parents)}"
                        + (f" (fnf branch {b.branch}, flushed)" if b.branch else ""))
        return acts

    def _drop(self, j: dict[str, Any] | None, plan: Mapping[str, Any], err: str, acts: list[str]) -> None:
        """A failed brief: logged ``brief_error`` and its slot freed (the next tick enqueues a fresh one)."""
        if j is not None and j in self._jobs:
            self._jobs.remove(j)
        try:
            self._unreserve(plan)
        except Exception as exc:  # noqa: BLE001 - accounting never stops the tick
            err = f"{err}; unreserve: {exc!r}"[:600]
        self.store.log("brief_error", kind=plan["kind"], lane=plan["lane"], error=err)
        acts.append(f"brief {plan['kind']} failed: {err}"[:200])

    def close(self, timeout: float = 120.0) -> int:
        """Stop the brief workers BEFORE the store closes: queued renders are cancelled, running ones (and the
        diagnostics ingest) are waited for up to ``timeout`` s -- a worker mid-query on a connection the store closes
        crashes the interpreter (SIGSEGV, seen in tests). Unstarted briefs are dropped (the next process's first tick
        re-enqueues their slots); returns how many."""
        pool, jobs, ingest = self._pool, list(self._jobs), self._ingest
        self._pool, self._jobs, self._ingest = None, [], None
        try:                                            # v3-c9 C1: the history tool's view of a store that is closing
            from loop.agents import history as HX
            HX.forget()
        except Exception:  # noqa: BLE001
            pass
        if (ps := self.__dict__.pop("_ps_pool", None)) is not None:     # the pre-screen's own worker
            ps.shutdown(wait=False, cancel_futures=True)
        for j in jobs:                                  # S7: unstarted briefs give their reservations back
            try:
                self._unreserve(j["plan"])
            except Exception:  # noqa: BLE001 - a closing store: nothing to give back to
                pass
        if pool is None:
            return len(jobs)
        pool.shutdown(wait=False, cancel_futures=True)
        running = [f for f in [j["fut"] for j in jobs] + [ingest] if f is not None and not f.done()]
        if running:
            wait(running, timeout=timeout)
        return len(jobs)

    def _engagements(self, data: Ledger) -> list[Engagement]:
        """v3 review O2: ONE :func:`engagements` per ledger view, shared by the early kill, the engagement log and the
        briefs (each recomputed it over every diagnostics row)."""
        key = (id(data), len(data.obs), len(data.nodes))
        got = self.__dict__.get("_eng")
        if got is None or got[0] != key:
            got = self._eng = (key, engagements(self.store, data))
        return got[1]

    def _log_engagement(self, data: Ledger) -> list[str]:
        seen, acts = dict(self.store.get_state(K_ENGAGED) or {}), []
        for e in self._engagements(data):
            if e.status in MEASURED and seen.get(e.node) != e.status:     # a corrected verdict is logged again
                seen[e.node] = e.status
                self.store.log("engagement", node_id=e.node, diagnostic=e.diagnostic, direction=e.direction,
                               status=e.status, z=e.z, n=e.n)
                acts.append(f"engagement {e.node}: {e.diagnostic} {'up' if e.direction > 0 else 'down'} predicted, "
                            f"{e.status} (z {e.z:+.1f} along the prediction)")
        if acts:
            self.store.set_state(K_ENGAGED, seen)
        return acts

    def request(self, b: Brief, v: View | None = None) -> CandidateRequest:
        """The session's request: the brief's row ids (what a budget entry may cite), the candidate slate's size and
        its reference gain (v3-c6i; the gap-share claim floor is gone: ``claim_floor`` 0), persona and category."""
        n = int(self.store.get_state(K_NEXT) or 0) + 1
        self.store.set_state(K_NEXT, n)
        strategy, op = KIND_REQUEST[b.kind]
        # v3-c6i: no claim floor (the slate's prediction is scored instead); the slate's size and reference gain
        g, what = ref_gap(v) if v is not None else (0.0, "")
        return self._ablate_components(b, CandidateRequest(id=f"{REQUEST_PREFIX}{n:05d}", task_id=self.ctx.task_id, strategy=strategy, operator=op,
                                island="main", parent_ids=b.parents, brief=b.text, brief_kind=b.kind,
                                family_hint=b.axis, ablated=tuple(b.ablated), branch=b.branch,
                                shortlist=tuple(b.shortlist), claim_floor=0.0, rows=row_ids(b.text),
                                persona=b.persona, k_candidates=self.k_for(b.kind), ref_gap=round(g, 6),
                                ref_kind=ref_kind(what) if what else "",
                                category=b.category, arm=b.arm))

    def _ablate_components(self, b: Brief, req: CandidateRequest) -> CandidateRequest:
        """10-22 (loop.components): each optional v3 context part is BYPASSED on ``components.ablation_rate`` of the sessions
        that would use it, a seeded draw logged ``component_gate`` (``request=<id>``); the meta report's component audit reads
        the outcomes of those sessions' nodes. The bypass rides ``request.ablated`` (as a fork-and-flush branch's flush does:
        a branch request is never drawn, its whole context is already its own A/B). Never in a dry run."""
        if b.branch or getattr(self, "_dry", False) or not hasattr(self.store, "log"):
            return req
        from loop import components as CG
        from loop.agents.researcher import wants_web
        out = list(req.ablated)
        if wants_web(req, self.ctx) and not CG.draw(self.ctx, self.store, "web", req.brief_kind, req.id, request=req.id):
            out.append("web")
        if req.parent_ids and not CG.draw(self.ctx, self.store, "postmortem", req.brief_kind, req.id, request=req.id):
            out.append("postmortem")
        brief = req.brief                               # 11-08: the board digest / root ledger, audited like the rest
        for comp, heading in (("board", "## Competition board digest"), ("root_audit", "## Root assumption ledger"),
                              ("local_runs", "## Already measured locally")):
            if comp in CG.v3_components(self.ctx) and heading in brief \
                    and not CG.draw(self.ctx, self.store, comp, req.brief_kind, req.id, request=req.id):
                out.append(comp)
                from loop.agents import briefs as BR
                brief = BR.strip_section(brief, heading)
        if len(out) == len(req.ablated):
            return req
        return dataclasses.replace(req, ablated=tuple(out), **({"brief": brief} if brief != req.brief else {}))

    def _bank_tag(self, req_id: str) -> str:
        """`` bank=<id>`` (+ `` root_audit=1``) when ``ideation.bank_v3`` served this request a bank idea."""
        if not self.opt("bank_v3"):
            return ""
        from loop.select.bank import K_BANK_USED
        for bid, reqs in (self.store.get_state(K_BANK_USED) or {}).items():
            if req_id in (reqs or ()):
                return f" bank={bid}" + (" root_audit=1" if str(bid).startswith("root_audit-") else "")
        return ""

    def _park(self, req: CandidateRequest, cand: Any, why: str) -> None:
        """A structural finish that never gave its evidence: its claim joins the evidence backlog the next literature
        brief researches (``briefs.research_blocks``)."""
        from loop.agents.briefs import K_EVIDENCE
        hyp = getattr(cand, "hypothesis", None)
        claim = f"{getattr(hyp, 'mechanism_tag', '')}: {getattr(hyp, 'claim', '')}".strip(": ")
        book = list(self.store.get_state(K_EVIDENCE) or [])[-49:]
        book.append({"id": req.id, "kind": req.brief_kind, "claim": claim[:600], "why": why[:300], "ts": self.clock()})
        self.store.set_state(K_EVIDENCE, book)
        self.store.log("evidence_parked", request_id=req.id, kind=req.brief_kind, why=why[:300])

    def intake(self, req: CandidateRequest, res: Any) -> str:
        """A finished session -> a node (QUEUED when the intake rules admit it, else REJECTED with the rule)."""
        from loop.select import intake as IN
        cand, store = getattr(res, "candidate", None), self.store
        if req.id in (claims := dict(store.get_state(K_CLAIMS) or {})):     # the session is over: its claim goes
            claims.pop(req.id)
            store.set_state(K_CLAIMS, claims)
        ds = (getattr(res, "plan", None) or {}).get("dataset_spec") if res.status != "backend_unavailable" else None
        if cand is None or res.status == "backend_unavailable":
            store.log("research_failed", request_id=req.id, status=res.status, error=str(res.error)[:500])
            bank = f"; {got}" if ds is not None and (got := self._dataset(req.id, ds)) else ""
            return f"research {req.id} produced nothing ({res.status}){bank}"
        plan0, key_problem = self._resolve_keys(cand, dict(getattr(res, "plan", None) or {}))   # v3-c7i I8
        sig = self.signature_of(req, cand, plan0)       # v3-c6i: the mechanism signature intake dedups on
        parm = self._program_arm(req)                    # v3-c10: (program, arm) of a per-arm operator session
        reason = IN.admit(self.ctx, store, cand, operator=req.operator, embedder=self.embedder,
                          strategy=req.strategy,       # an operator arm re-runs a measured mechanism on purpose: no dedup
                          signature=(sig, tuple(req.parent_ids)) if sig and parm is None else None,
                          judge=self._mech_judge() if sig else None, dup_exempt=parm is not None) \
            if res.status == "ok" else res.status
        if not reason and key_problem:
            reason = f"unresolved_knob_key: {key_problem}"
        if not reason and parm is not None:              # v3-c10: one arm per node (a node bundled both DINO arms)
            from loop.core import program as PG
            why = PG.cross_arm(parm[0], parm[1], list(cand.config_diff or {}),
                               "\n".join(l for _, l in IN._added_lines(cand.diff_text or "")))
            if why:
                reason = f"cross_arm_bundling: {why}"
        if not reason and req.id.startswith(REQUEST_PREFIX) and not cluster_exempt(req) \
                and (got := self._cluster_rule(cand, sig))[0]:
            reason = f"{got[0]}: {got[1]}"               # v3-c7i I3: a saturated-negative cluster past its cap
        if not reason and req.id.startswith(REQUEST_PREFIX) and not cluster_exempt(req) and parm is None \
                and req.brief_kind in ("explore", "literature") and (got := self._shadow_rule(cand))[0]:
            reason = f"{got[0]}: {got[1]}"               # 10-21: an operator-program arm is in flight on this mechanism
        if not reason and ds is not None and getattr(self.ctx, "cfg", None) is not None:
            reason = self._dead_spec(cand, ds)           # v3-c8a B2: a dataset_spec of an already-dead dataset
        mixed = ""
        if reason == "mixed_change":                     # v3-c8a B3: split by the tick (loop.core.mixed), not rejected
            from loop.core import mixed as MX
            if MX.mode(self.ctx) == "split":
                reason, mixed = "", ",".join(MX.groups(self.ctx, cand.config_diff))
        knob_hold = ""
        if not reason and req.brief_kind == "bold" and (got := self._bold_rule(cand))[0]:
            if got[0] == "knob_hold":                    # v3 review M3: judge down -> held QUEUED, retried (fails closed)
                knob_hold = f"{self.clock():.0f}"
            else:
                reason = f"{got[0]}: {got[1]}"           # v3-c8b: a knob / weight change of an existing term
        if not reason and not knob_hold and parm is None and req.brief_kind in ("bold", "kuhn") \
                and self.opt("struct_tune"):             # 10-22: a structural change must give the ladder something to tune
            from loop.core import struct_tune as STT
            got = STT.untunable(self.ctx, dict(cand.config_diff or {}), plan0.get("components") or plan0.get("knobs"))
            if got[0]:
                reason = f"{got[0]}: {got[1]}"
        if not reason and self.opt("require_evidence") and plan0.get("evidence_missing"):
            reason = f"no_evidence: {str(plan0['evidence_missing'])[:300]}"     # 11-08: parked, not dropped
            self._park(req, cand, str(plan0["evidence_missing"]))
        seq = len(store.nodes())
        while store.get_node(f"n{seq:04d}") is not None:
            seq += 1
        nid, now, hyp, plan = f"n{seq:04d}", self.clock(), cand.hypothesis, dict(plan0)
        if store.get_hypothesis(hyp.id or f"h-{nid}") is None:
            store.put_hypothesis(replace(hyp, id=hyp.id or f"h-{nid}", created_ts=hyp.created_ts or now))
        diff_path = ""
        if cand.diff_text and getattr(self.ctx, "state_dir", ""):
            p = Path(self.ctx.state_dir) / "diffs" / f"{nid}.diff"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(cand.diff_text)
            diff_path = str(p)
        bk = BRIEF_ALIAS.get(req.brief_kind, req.brief_kind)    # merged c7a x c7i: a probe repair is kind repair
        tags = (f"request={req.id} brief={bk} lane={_lane(req)} backend={_backend(self.llm)}"
                f" prompt={prompt_version('researcher', 'ideation')}"
                + (f" web=1 searches={res.web_searches} fetches={res.web_fetches}" if getattr(res, "web", False) else "")
                + self._bank_tag(req.id))
        if parm is not None:                                                # same tags as an eval-only arm's node
            tags += f" op={parm[0]['id']} arm={parm[1]['id']} stage={parm[1].get('stage') or 'A'} program_arm=1"
        if req.brief_kind == "port" and req.id.startswith(PORT_PREFIX):   # a researcher port of a conflicting graft
            tags += f" port_of={req.id[len(PORT_PREFIX):]}"
        if req.id.startswith(REROOT_XOVER_PREFIX) and len(req.parent_ids) > 1:   # a re-root crossover (tick)
            tags += f" reroot_cross_of={req.parent_ids[1]}"
        if req.brief_kind == "repair_probe" and req.parent_ids:            # v3-c7a A4: the repaired node
            tags += f" repair_of={req.parent_ids[0]}"
        if getattr(req, "branch", ""):                                      # a fork-and-flush branch (loop.core.fnf)
            tags += f" branch={req.branch}"
        if str(getattr(req, "arm", "") or "").startswith("alt_base"):       # v3-c9 C2: the measured alt-base arm
            _, _, rest = req.arm.partition(":")
            src, _, dist = rest.partition(":")
            tags += f" arm=alt_base alt_of={req.parent_ids[0]} alt_src={src or '?'}" + (f" alt_dist={dist}" if dist else "")
        if mixed:                                                           # v3-c8a B3: its parts are built + screened first
            tags += f" mixed={mixed}"
        if sig:                                     # v3-c6i: lens, built category, the brief's suggested category
            from loop.core import mechanisms as MS
            tags += (f" cat={MS.category(self.ctx, hyp.family, hyp.mechanism_tag, getattr(res, 'mechanism_type', ''))}"
                     + (f" persona={re.sub(r'[^A-Za-z0-9]+', '_', req.persona).strip('_')}" if req.persona else "")
                     + (f" lens_cat={req.category}" if getattr(req, "category", "") else ""))
            plan["sig"] = sig
        if knob_hold and not reason:
            tags += f" knob_hold={knob_hold}"
        if plan.get("bold_category"):                # v3-c8b: the slate's chosen structural category
            tags += f" bold_cat={plan['bold_category']}"
        store.put_node(Node(id=nid, task_id=self.ctx.task_id, commit=cand.commit, parent_id=req.parent_ids[0],
                            strategy=req.strategy, operator=req.operator, family=hyp.family,
                            mechanism_tag=hyp.mechanism_tag, diff_hash=cand.diff_hash,
                            status=NodeStatus.REJECTED if reason else NodeStatus.QUEUED,
                            second_parent_id=req.parent_ids[1] if len(req.parent_ids) > 1 else None,
                            diff_path=diff_path, hypothesis_id=hyp.id or f"h-{nid}", levers=hyp.levers,
                            embedding=None if reason else self.embedder.embed(IN.candidate_text(cand)),
                            reject_reason=reason, created_ts=now, updated_ts=now, notes=tags))
        if not reason and cand.config_diff:
            store.set_state(f"config_diff:{nid}", dict(cand.config_diff))
        if parm is not None:
            self._book_arm(parm, nid, reason)
        if plan:                                     # mechanism_prediction, knobs, claimed size (calibration only)
            store.set_state(f"ideation:{nid}", plan)
        store.put_handoff(replace(cand.handoff, id=f"ho-{nid}", node_id=nid, created_ts=cand.handoff.created_ts or now))
        mp = plan.get("mechanism_prediction") or {}
        store.log("intake", node_id=nid, request_id=req.id, admitted=not reason, reason=reason,
                  claimed_size=mp.get("size"), budget=mp.get("budget"), claim_floor=getattr(req, "claim_floor", 0.0),
                  below_floor=bool(plan.get("claim_below_floor")), components=len(plan.get("components") or ()))
        if plan.get("candidates"):
            store.log("candidates", node_id=nid, request_id=req.id, k=len(plan["candidates"]), chosen=plan.get("chosen"),
                      prediction=plan.get("prediction"), malformed=plan.get("vs_problem", ""))
        if not reason and sig:                      # v3-c6i: the pairwise pre-screen (one short call, off the tick)
            self._prescreen(nid, sig, hyp.claim, tuple(req.parent_ids))
        # its dataset_spec is built for it (loop.databank); the node waits in the queue until the dataset is ready
        bank = f"; {got}" if ds is not None and not reason and (got := self._dataset(nid, ds)) else ""
        return (f"rejected {nid} from {req.id}: {reason}" if reason else
                f"queued {nid} ({hyp.family}/{hyp.mechanism_tag}) from {req.id}") + bank

    def _program_arm(self, req: CandidateRequest) -> tuple[dict[str, Any], dict[str, Any]] | None:
        """(program, arm) of a per-arm operator session (``req.arm`` = ``program_arm:<program>:<arm>``), else None."""
        a = str(getattr(req, "arm", "") or "")
        if not a.startswith("program_arm:"):
            return None
        _, _, rest = a.partition(":")
        pid, _, aid = rest.partition(":")
        p = self._op_find(pid, aid)
        return (p, p["_arm"]) if p is not None and p.get("_arm") else None

    def _book_arm(self, parm: tuple[dict[str, Any], dict[str, Any]], nid: str, reason: str) -> None:
        """``K_ARMS`` records the arm's node (like an eval-only arm's); a bundling rejection gives the arm's session
        back (``{"fails": n}``) up to twice, then the arm is given up with the reason."""
        from loop.core import program as PG
        pid, aid = str(parm[0]["id"]), str(parm[1]["id"])
        book = dict(self.store.get_state(PG.K_ARMS) or {})
        mine = dict(book.get(pid) or {})
        if not reason:
            mine[aid] = nid
        elif reason.startswith("cross_arm_bundling"):
            n = int((mine.get(aid) or {}).get("fails", 0)) + 1 if isinstance(mine.get(aid), Mapping) else 1
            mine[aid] = {"fails": n} if n < 3 else f"-: {reason}"[:300]
            if n < 3:
                self._op_mark(f"{pid}/{aid}", -1)        # re-briefed on a later tick
        else:
            mine[aid] = f"-: {reason}"[:300]
        book[pid] = mine
        self.store.set_state(PG.K_ARMS, book)

    def _resolve_keys(self, cand: Candidate, plan: dict[str, Any]) -> tuple[dict[str, Any], str]:
        """v3-c7i I8: the plan's knob keys (``knobs``, components' ``knobs`` / ``off``) resolved by unique suffix
        against the candidate commit's flattened config (``adapter.config_at``); ('' = fine) or the problems, which
        reject the node at intake (``unresolved_knob_key``) so no ladder ever fails on a bare key later."""
        from loop.core import program as PG
        if not plan or not (plan.get("knobs") or plan.get("components")) or cand is None:
            return plan, ""
        adapter = getattr(self, "adapter", None)
        try:
            cfg = dict(adapter.config_at(cand.commit)) if adapter is not None and hasattr(adapter, "config_at") \
                and cand.commit else {}
        except Exception as exc:  # noqa: BLE001 - unreadable: the ladder resolves (or waits) later
            self.store.log("resolve_keys_error", error=repr(exc)[:300])
            cfg = {}
        if not cfg:
            return plan, ""
        out, problems = PG.resolve_plan_keys(plan, cfg, list((cand.config_diff or {}).keys()))
        if out != plan:
            self.store.log("knob_keys_resolved", candidate=getattr(cand, "request_id", ""), problems=problems[:6])
        return out, "; ".join(problems[:6])

    def _dead_spec(self, cand: Candidate, spec: Any) -> str:
        """v3-c8a B2: 'dead_dataset: ...' when a finish's ``dataset_spec`` builds a dataset (same content hash) already
        measured below its parent (loop.core.datasets.rule), else ''."""
        from loop import databank as DB
        from loop.core import datasets as DS
        try:
            got, why = DB.validate(spec)
        except Exception:  # noqa: BLE001 - an invalid spec is refused by the databank request itself
            return ""
        if why or not got:
            return ""
        rule, fix = DS.rule(self.ctx, self.store, str(cand.hypothesis.claim or ""), f"sha:{DB.spec_hash(got)}")
        return f"{rule}: {fix}"[:600] if rule else ""

    def _shadow_rule(self, cand: Candidate) -> tuple[str, str]:
        """program.shadow_rule on the candidate: its family / tag / claim and the config keys of its diff."""
        if cand is None or not getattr(self.ctx, "cfg", None):
            return "", ""
        from loop.core import program as PG
        h = cand.hypothesis
        return _safe(lambda: PG.shadow_rule(self.ctx, self.store, h.family, h.mechanism_tag, h.claim,
                                            list(cand.config_diff or {}), self.clock()), ("", ""))

    def _cluster_rule(self, cand: Candidate, sig: str) -> tuple[str, str]:
        ac = self._alloc()
        if not (ac.enabled and ac.clusters) or cand is None:
            return "", ""
        from loop.core import alloc as AL
        h = cand.hypothesis
        return _safe(lambda: AL.cluster_rule(self.ctx, h.family, h.mechanism_tag, sig or h.claim, h.claim), ("", ""))

    def _parent_src(self, commit: str) -> Any:
        """``path -> the parent's source`` for the AST knob comparison (None without an adapter that can show a commit)."""
        ad, repo = getattr(self, "adapter", None), None
        show = getattr(ad, "_show", None)
        repo = getattr(ad, "repo", None)
        if not (commit and show and repo):
            return None
        return lambda rel: show(repo, commit, rel)

    def _knob_judge(self, parent_src: Any = None) -> Any:
        """The knob judge over the DIFF alone: the qwen judge when it is up, else the deterministic AST / novelty judge
        (loop.core.bold.chained_knob_judge; 10-22: qwen answered 3 of 186 calls in 7 days, the node was held 30 min and
        then released on the bare AST verdict). It always answers, so no node is held for a knob verdict; every call logs
        ``bold_knob_judge`` with the judge that decided."""
        from loop.core import bold as BOLD
        store = self.store
        inner = BOLD.llm_knob_judge(self.llm) if self.llm is not None and hasattr(self.llm, "complete_json") else None
        judge = BOLD.chained_knob_judge(inner, parent_src, store.log)

        def wrapped(diff_text: str) -> bool:
            got = judge(diff_text)
            if judge.last == "qwen" and store.get_state(K_KNOB_OUTAGE):
                store.set_state(K_KNOB_OUTAGE, 0.0)              # qwen answered: the outage is over
            return got
        return wrapped

    def _bold_rule(self, cand: Candidate) -> tuple[str, str]:
        """v3-c8b / v3 review M3: ('bold_knob_only', fix) for a bold candidate that only re-parameterises existing terms
        (the AST structure comparison, then the qwen judge on the DIFF). ('knob_hold', '') when the judge is wired but gave
        no verdict: the node is admitted QUEUED with a ``knob_hold=<ts>`` tag and the tick retries (:meth:`knob_holds`)."""
        from loop.core import bold as BOLD
        return _safe(lambda: BOLD.knob_rule(self.ctx, cand.diff_text or "", "", self._knob_judge(
            self._parent_src(getattr(cand, "parent_commit", ""))), self._parent_src(getattr(cand, "parent_commit", ""))),
            ("", ""))

    def knob_holds(self, statuses: Mapping[str, Any], now: float, acts: list[str]) -> frozenset[str]:
        """v3 review M3: the QUEUED bold nodes held for a knob verdict (``knob_hold=<ts>``). Each tick (at most every
        ``KNOB_RETRY_S`` per node, <= 2 judge calls) the judge is asked again: knob -> REJECTED ``bold_knob_only``;
        structural -> released; still no verdict -> held until ``bold.HOLD_S``, then the AST verdict (structural) stands and the
        node is released with ``knob_ast_fallback`` (logged + an action line). Never raises."""
        from loop.core import bold as BOLD
        from loop.tick import status_name
        out, calls = set(), 0
        try:
            judge = self._knob_judge()
            down = float(self.store.get_state(K_KNOB_OUTAGE) or 0.0)
            outage = down > 0 and now - down < BOLD.OUTAGE_TTL_S
            for nid, st in statuses.items():
                if status_name(st) != "QUEUED":
                    continue
                node = self.store.get_node(nid)
                m = re.search(r"\bknob_hold=(\d+(?:\.\d+)?)", getattr(node, "notes", "") or "") if node else None
                if m is None:
                    continue
                since = float(m.group(1))
                last = float((self.store.get_state(f"bold.knob_try:{nid}") or 0.0))
                got = None
                if judge is not None and calls < 1 and now - last >= KNOB_RETRY_S:
                    self.store.set_state(f"bold.knob_try:{nid}", now)
                    calls += 1
                    try:
                        got = judge(Path(node.diff_path).read_text() if node.diff_path else "")
                    except Exception:  # noqa: BLE001
                        got = None
                tag = m.group(0)
                if got is True and _safe(lambda: BOLD.code_override(          # against the PARENT's own source
                        self.ctx, Path(node.diff_path).read_text() if node.diff_path else "", self._parent_src(
                            getattr(self.store.get_node(node.parent_id or ""), "commit", ""))), ""):
                    got = False                          # 11-10 (a node): the diff adds code -- not a knob change
                if got is True:
                    self.store.update_node(nid, status=NodeStatus.REJECTED, reject_reason="bold_knob_only: the knob judge "
                                           "classified it as a re-weighting of existing terms",
                                           notes=node.notes.replace(tag, "knob_judged=1"))
                    self.store.log("bold_knob_judge", node_id=nid, knob_only=True, held_s=round(now - since))
                    acts.append(f"bold knob hold {nid}: judged knob-only -> rejected")
                elif got is False:
                    self.store.update_node(nid, notes=node.notes.replace(tag, "knob_judged=0"))
                    acts.append(f"bold knob hold {nid}: judged structural -> released")
                elif now - since >= (BOLD.HOLD_OUTAGE_S if outage else BOLD.HOLD_S):
                    self.store.set_state(K_KNOB_OUTAGE, now)           # live 10-21: qwen down 6 h, every bold node waited 31 min
                    self.store.update_node(nid, notes=node.notes.replace(tag, "knob_ast_fallback=1"))
                    self.store.log("bold_knob_fallback", node_id=nid, held_s=round(now - since))
                    acts.append(f"ALERT: bold knob judge unavailable for {nid} for {(now - since) / 60:.0f} min: "
                                "released on the AST verdict (structural)")
                else:
                    out.add(nid)
        except Exception as exc:  # noqa: BLE001 - a hold never stops the tick
            self.store.log("bold_knob_hold_error", error=repr(exc)[:300])
        return frozenset(out)

    def _mech_judge(self) -> Any:
        """v3-c6r: the qwen yes/no "same mechanism?" judge of the signature dedup (``ideation.mech_judge_cos`` > 0 and
        an LLM); a failed call is no verdict (logged)."""
        if self.llm is None or not hasattr(self.llm, "complete_json") or float(self.opt("mech_judge_cos") or 0) <= 0:
            return None
        from loop.core import mechanisms as MS
        inner, store = MS.llm_judge(self.llm), self.store

        def judge(sig: str, near: Sequence[tuple[str, str]]) -> list[bool] | None:
            try:
                got = inner(sig, near)
            except Exception as exc:  # noqa: BLE001 - backend down / schema: lexical only
                store.log("mech_judge_error", error=repr(exc)[:300])
                return None
            store.log("mech_judge", near=[n for n, _ in near], same=got)
            return got
        return judge

    def signature_of(self, req: CandidateRequest, cand: Candidate, plan: Mapping[str, Any]) -> str:
        """v3-c6i: a v3 session's mechanism signature (:func:`loop.core.mechanisms.signature`: family + tag + the
        chosen candidate's one-line mechanism, else the claim, with the ancestors' own text removed); '' otherwise."""
        if not req.id.startswith(REQUEST_PREFIX) or cand is None:
            return ""
        from loop.core import mechanisms as MS
        try:
            anc = MS.ancestors(self.store, req.parent_ids)
            return MS.signature(cand.hypothesis.family, cand.hypothesis.mechanism_tag,
                                str(plan.get("mechanism") or "") or cand.hypothesis.claim, MS.claims_of(self.store, anc))
        except Exception as exc:  # noqa: BLE001 - no signature: the claim + diff near_dup stays
            self.store.log("signature_error", request_id=req.id, error=repr(exc)[:300])
            return ""

    # -- v3-c6i: the pairwise pre-screen (loop.core.prescreen) -------------------------------------------------
    def _prescreen(self, nid: str, sig: str, claim: str, parents: tuple[str, ...], tries: int = 0) -> None:
        """Ask the one pre-screen call for a new node (off the tick, on its OWN one-worker pool: it never holds a
        brief worker); skipped without an LLM, at pairwise_weight 0 or without measured neighbours."""
        from loop.core import prescreen as PS
        if self.llm is None or float(self.opt("pairwise_weight")) <= 0 or self._view is None:
            return
        beliefs = self._view[1].beliefs
        try:
            near = PS.neighbors(self.store, beliefs, sig, parents, int(self.opt("pairwise_k")), (nid,), self.embedder)
        except Exception as exc:  # noqa: BLE001
            self.store.log("prescreen_error", node_id=nid, error=repr(exc)[:300])
            return
        if not near:
            return
        m0 = PS.prior_mean(beliefs)
        retry = {"sig": sig, "claim": claim[:800], "parents": list(parents), "tries": tries + 1}

        def job() -> dict[str, Any]:
            return PS.ask(self.llm, sig, claim, near, m0)
        jobs = self.__dict__.setdefault("_ps_jobs", {})
        if int(self.opt("brief_workers")) > 0:
            jobs[nid] = (self._ps_executor().submit(job), retry)
        else:
            self._ps_record(nid, job, retry)

    def _ps_executor(self) -> ThreadPoolExecutor:
        if self.__dict__.get("_ps_pool") is None:
            self._ps_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prescreen")
        return self._ps_pool

    def _ps_record(self, nid: str, job: Callable[[], dict[str, Any]], retry: Mapping[str, Any] | None = None) -> None:
        from loop.core import prescreen as PS
        try:
            rec = {**job(), "ts": self.clock()}
        except Exception as exc:  # noqa: BLE001 - retryable: asked again on a later tick (bounded, PS.RETRIES)
            rec = {"error": repr(exc)[:200], "ts": self.clock(), **dict(retry or {})}
        PS.record(self.store, nid, rec)

    def _ps_retry(self, v: View) -> None:
        """Failed pre-screen calls asked again (v3-c6r): a node still UNMEASURED (no screen seed: the prediction
        stays a prediction), at most ``prescreen.RETRIES`` tries, ``RETRY_S`` after the last failure."""
        from loop.core import prescreen as PS
        book, jobs, now = self.store.get_state(PS.K_PS) or {}, self.__dict__.setdefault("_ps_jobs", {}), self.clock()
        for nid, rec in book.items():
            if not isinstance(rec, Mapping) or "error" not in rec or nid in jobs or not rec.get("sig"):
                continue
            if int(rec.get("tries") or 1) >= PS.RETRIES or now - float(rec.get("ts") or 0.0) < PS.RETRY_S:
                continue
            if getattr(v.beliefs.get(nid), "n", 0) >= 1 or nid not in v.led.nodes:
                continue
            self._prescreen(nid, str(rec["sig"]), str(rec.get("claim") or ""), tuple(rec.get("parents") or ()),
                            int(rec.get("tries") or 1))

    def _ps_tick(self, tick: Any, v: View) -> list[str]:
        """Harvest finished pre-screen calls, retry failed ones, score measured ones, and hand the core's rank the
        prior (core.prior -> prescreen.nudge)."""
        from loop.core import prescreen as PS
        jobs = self.__dict__.setdefault("_ps_jobs", {})
        for nid, (fut, retry) in list(jobs.items()):
            if fut.done():
                jobs.pop(nid)
                self._ps_record(nid, fut.result, retry)
        if not getattr(tick, "dry", False):
            self._ps_retry(v)
        acts = PS.score(self.store, v.beliefs, int(self.opt("calibration_min_seeds")), self.clock())
        w = PS.weight(self.ctx, self.store)
        core = getattr(tick, "core", None)
        if core is not None and w > 0 and not getattr(tick, "dry", False):
            store = self.store
            if hasattr(core, "prior"):
                core.prior = lambda: (store.get_state(PS.K_PS) or {}, PS.weight(self.ctx, store),
                                      bool(self.opt("pairwise_center")))
        elif core is not None and hasattr(core, "prior") and w == 0:
            core.prior = None                              # switched off: no stale prior keeps nudging
        return acts

    def _calibrate(self, v: View) -> list[str]:
        """Score every due chosen prediction (loop.core.calibration)."""
        from loop.core import calibration as CAL
        kinds = {n: _brief_tag(node) for n, node in v.led.nodes.items()}
        return CAL.score_due(self.store, v.beliefs, kinds, int(self.opt("calibration_min_seeds")), self.clock())

    def _dataset(self, by: str, spec: Any) -> str:
        """``loop.databank.request`` of a finish's ``dataset_spec`` (by the node, else the request): queued, deduped
        or refused -- logged against ``by``."""
        from loop import databank as DB
        try:
            return DB.request(self.ctx, self.store, by, spec)
        except Exception as exc:  # noqa: BLE001 - a malformed request never loses the node
            self.store.log("databank_error", node_id=by, error=repr(exc)[:500])
            return f"databank request of {by} failed: {exc!r}"[:200]


def _safe(fn: Callable[[], Any], default: Any) -> Any:
    try:
        return fn()
    except Exception:  # noqa: BLE001 - a computed list that cannot be computed is empty (its brief part says so)
        return default


ROW_ID = re.compile(r"\[((?:H|D|X|XD|C|R|U|T|G)-[A-Za-z0-9_.@-]+)\]")


def row_ids(text: str) -> tuple[str, ...]:
    """The bracketed row ids a brief shows (``[D-cls]``, ``[XD-break_his]``, ...): what a per-axis budget may cite."""
    return tuple(dict.fromkeys(ROW_ID.findall(text)))


def _backend(llm: Any) -> str:
    try:
        return str(llm.tier("researcher"))
    except Exception:  # noqa: BLE001 - unknown backend
        return "?"


def _lane(req: CandidateRequest) -> str:
    return {Strategy.CROSSOVER: "crossover", Strategy.EXPLOIT: "exploit"}.get(req.strategy, "explore")
