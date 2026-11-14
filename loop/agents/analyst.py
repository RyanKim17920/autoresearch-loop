"""loop.agents.analyst — big reviews and the playbook consolidation (DESIGN §9). Code computes the evidence, the LLM
writes the explanation; levers are filtered to the vocabulary and links to known ids. Records are
returned, never persisted here: the driver is the one writer (the only write is the
``probe_dropped`` log event for a probe brief the constraint check rejects)."""

from __future__ import annotations

import json
import math
import re
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from loop.agents import briefs
from loop.contracts import (CLOSED_STATUSES, PANEL, ROOT_NODE_ID, AxisPrediction, HypothesisKind,
                            HypothesisRecord, LeverTags, Node, NodeStatus, ReviewReport, RunStatus, VoidType,
                            WorldSnapshot, current_reviews)
from loop.spec import LoopContext
from loop.explain import block as explain_block
from loop.select import is_moonshot_probe

_TXT = {"type": "string"}
_LEVERS = {"type": "object", "additionalProperties": {"type": "integer", "enum": [-1, 0, 1]}}
_META = {"type": "object", "required": ["claim", "mechanism_tag", "levers", "predicted_sign", "falsification"],
         "properties": {"claim": {"type": "string", "minLength": 1}, "mechanism_tag": _TXT,
                        "family": _TXT, "levers": _LEVERS, "predicted_sign": {"type": "integer", "enum": [-1, 1]},
                        "predicted_size": {"type": "number"}, "falsification": {"type": "string", "minLength": 1},
                        "probes": {"type": "array", "items": _TXT}}}
_TUNE = {"type": "object", "required": ["knob", "value", "reason"],
         "properties": {"knob": _TXT, "value": {"type": "number"}, "reason": _TXT}}
LENSES = ("measurement", "ml_problem", "search", "constraints", "loop_system")
REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object", "required": ["summary", "findings", "meta_hypotheses"],
    "properties": {"summary": {"type": "string", "minLength": 1}, "findings": {"type": "array", "items": _TXT},
                   "meta_hypotheses": {"type": "array", "items": _META},
                   "probe_briefs": {"type": "array", "items": _TXT},
                   "design": {"type": "array", "items": {"type": "object", "required": ["key", "high"],
                                                          "properties": {"key": _TXT, "levers": _LEVERS}}},
                   "explore_bias": {"type": "object", "additionalProperties": {"type": "number"}},
                   "selftune": {"type": "array", "items": _TUNE}}}
FP_TRIGGER = "first_principles"
FP_SCHEMA: dict[str, Any] = {
    "type": "object", "required": ["summary", "assumptions"],
    "properties": {"summary": {"type": "string", "minLength": 1},
                   "assumptions": {"type": "array", "items": {
                       "type": "object", "required": ["lens", "assumption", "status", "evidence"],
                       "properties": {"lens": {"type": "string", "enum": list(LENSES)}, "assumption": _TXT,
                                      "status": {"type": "string", "enum": ["holds", "broken", "unknown"]},
                                      "evidence": _TXT}}},
                   "meta_hypotheses": {"type": "array", "items": _META},
                   "moonshots": {"type": "array", "items": _TXT},
                   "assumption_probes": {"type": "array", "items": _TXT},
                   "selftune": {"type": "array", "items": _TUNE},
                   "proposals": {"type": "array", "items": {"type": "object", "required": ["change"],
                                                            "properties": {"change": _TXT, "rationale": _TXT}}}}}


def lever_dict(raw: Mapping[str, Any], vocab: tuple[str, ...]) -> LeverTags:
    """Only vocabulary levers with a -1/+1 direction."""
    return LeverTags({k: v for k, v in raw.items() if k in vocab and v in (-1, 1) and not isinstance(v, bool)})


# A probe asking for more training can never pass intake (the budget keys are protected).
_MORE = (r"(?:more|longer|extra|additional|increase[sd]?|increasing|raise[sd]?|double[sd]?|doubling|triple|"
         r"extend(?:ed|s)?|\+\s*\d+(?:\.\d+)?\s*%)")
_BUDGET = (r"(?:(?<!warmup )(?<!warm-up )steps?|epochs?|flops?|(?:training|train|optimi[sz]er)\s+"
           r"(?:samples?|budget|length|time|steps?|iterations?|schedule)|samples?\s+seen|sample\s+budget|compute\s+budget)")
_MORE_TRAINING = re.compile(
    rf"(?<!\w){_MORE}(?!\w)[^.;\n]{{0,40}}?\b{_BUDGET}\b|\b(?:train(?:ing)?\s+(?:for\s+)?longer|longer\s+training|"
    rf"longer\s+schedule|max_train_\w+)\b", re.IGNORECASE)


def probe_violation(text: str, protected_keys: tuple[str, ...]) -> str:
    """Why a probe brief can never be implemented ('' if it can): it names a protected config
    key, or it asks for more steps / samples / FLOPs / epochs / longer training."""
    low = text.lower()
    for key in protected_keys:
        k = key.lower()
        names = [k] + ([k.rsplit(".", 1)[1]] if "." in k else [])
        for name in names:
            # a bare segment ("probe") only counts as a dotted path ("probe.datasets"); a
            # distinctive leaf ("max_train_samples", "split_seed") counts on its own
            pat = rf"(?<![\w.]){re.escape(name)}(?:\.[\w.]+)?(?![\w])" if "_" in name or "." in name \
                else rf"(?<![\w.]){re.escape(name)}\.[\w.]+"
            if re.search(pat, low):
                return f"names protected key {key}"
    if m := _MORE_TRAINING.search(text):
        return f"asks for more training ({m.group(0)!r}); the training budget is protected"
    return ""


def _meta_block(ctx: LoopContext, store: Any, snapshot: Any, own: tuple[str, ...]) -> tuple[str, str]:
    """(scorecard, own records) for a review, combined <= loop.meta.MAX_META_BLOCK chars."""
    from loop import meta
    try:
        return meta.meta_block(ctx, store, snapshot, own)
    except Exception as exc:  # noqa: BLE001
        return f"(scorecard unavailable: {type(exc).__name__})", ""


class LLMAnalyst:
    """``big_review`` and ``consolidate`` (role 'review', Opus-class)."""

    def __init__(self, ctx: LoopContext, llm: Any) -> None:
        self.ctx, self.llm, self.vocab = ctx, llm, ctx.cfg.levers.vocabulary

    def consolidate(self, store: Any, snapshot: WorldSnapshot) -> dict[str, Any]:
        """The meta-learning PLAYBOOK (loop.meta.consolidate; Opus, role 'review'); returned, not persisted."""
        from loop import meta
        return meta.consolidate(self.ctx, self.llm, store, snapshot, now=time.time())

    def big_review(self, store: Any, snapshot: WorldSnapshot, *,
                   trigger: str) -> tuple[ReviewReport, list[HypothesisRecord]]:
        """A big review; ``trigger == 'first_principles'`` runs the first-principles review instead
        (assumptions tested in five lenses, extra evidence, operator proposals)."""
        from loop import selftune                         # loop.design / loop.driver import loop.agents
        from loop.design import lines as design_lines
        from loop.tune import lines as tune_lines           # tuning passes and dose ladders
        cfg, previous, fp = self.ctx.cfg, store.reviews(), trigger == FP_TRIGGER
        fresh = {r.id for r in current_reviews(store)}
        card, own = _meta_block(self.ctx, store, snapshot, ("first_principles",) if fp else ("review", "design"))
        try:                                              # the causal component audit (loop.components)
            from loop.components import review_block
            card += "\n\n" + review_block(store)
        except Exception as exc:  # noqa: BLE001
            card += f"\n\n(component audit unavailable: {type(exc).__name__})"
        common = dict(
            task_id=self.ctx.task_id, n_scored=snapshot.n_scored, lever_vocab=list(self.vocab),
            levers=briefs.lever_table(snapshot), constraints=briefs.hard_constraints(self.ctx),
            hypotheses=briefs.hypothesis_table(store, snapshot, limit=120),
            selftune=selftune.describe(cfg) + "\n\n" + selftune.preempt_block(self.ctx, store, time.time()),
            postmortems=explain_block(store, sorted(store.nodes(status=list(CLOSED_STATUSES)),
                                                    key=lambda n: (n.updated_ts or 0.0, n.id)), snapshot,
                                      limit=30 if fp else 60),
            previous="\n".join(("" if r.id in fresh else "[STALE: built on estimates a measurement-epoch fix has since "
                                "corrected; re-derive, do not carry its conclusions] ")
                               + f"- {r.id} ({r.trigger}, at {r.n_evals_at} evals): {r.summary[:500]} "
                               f"| meta: {list(r.meta_hypothesis_ids)}" for r in previous[-5:])
            or "(first review)", table=briefs.review_node_table(store, snapshot, cfg.axes, max_rows=500),
            families=sorted(cfg.families),
            quality_table=briefs.optional_block("quality_table", store, snapshot) or "(not available)",
            lessons="\n".join(filter(None, (design_lines(store),
                                             tune_lines(store), briefs.optional_block(
                "ideation_lessons", store, snapshot)))) or "(not available)",
            meta=card, record=own)
        if fp:
            prompt = briefs.render("first_principles", why=f"{len(previous)} reviews so far", **common,
                                   throughput=throughput_block(self.ctx, store, time.time()),
                                   variance=variance_block(self.ctx, store, snapshot),
                                   run_metrics=run_metrics_block(store))
        else:
            frontiers: dict[str, list[str]] = {}
            for n in store.nodes(status=[NodeStatus.CONFIRMED]):
                frontiers.setdefault(n.island, []).append(briefs.node_line(snapshot, n))
            prompt = briefs.render(
                "review", n_review=len(previous) + 1, trigger=trigger, bias_cap="half a panel sigma", **common,
                axis_levers="\n\n".join(briefs.lever_table(snapshot, target=a) for a in cfg.axes),
                frontiers="\n".join(f"island {i}: " + "; ".join(v) for i, v in frontiers.items())
                or f"(no confirmed nodes; incumbent {snapshot.incumbent_id})")
        if facts := briefs.findings_block(self.ctx):      # the operator's manual experiments outrank past prose
            prompt = facts + "\n" + prompt
        if focus := briefs.focus_block(self.ctx):
            prompt = (focus + "Bias your probe briefs, moonshots and meta-hypotheses toward this focus "
                      "(still grounded in the evidence below).\n\n" + prompt)
        out = self.llm.complete_json(
            "review", system="You are the principal investigator of an automated research loop. "
                             + ("Question the loop's own assumptions from first principles. " if fp else "")
                             + "Generalise across experiments; cite evidence; never invent numbers.",
            prompt=prompt, schema=FP_SCHEMA if fp else REVIEW_SCHEMA)
        now, review_id = time.time(), f"rv-{len(previous) + 1}-{uuid.uuid4().hex[:6]}"
        metas: list[HypothesisRecord] = []
        probes: list[tuple[str | None, str]] = [(None, b) for b in out.get("probe_briefs") or []]
        probes += [(None, b if is_moonshot_probe(b) else f"[moonshot] {b}") for b in out.get("moonshots") or []]
        probes += [(None, b) for b in out.get("assumption_probes") or []]
        for i, m in enumerate(out.get("meta_hypotheses") or []):
            levers = lever_dict(m["levers"], self.vocab)
            if not levers.directions:
                continue          # no lever vector: it could never auto-link a trial (D3.9)
            h = HypothesisRecord(
                id=f"{review_id}-m{i + 1}", task_id=self.ctx.task_id, kind=HypothesisKind.META,
                claim=m["claim"], mechanism_tag=m.get("mechanism_tag") or "meta",
                family=m.get("family") or "cross", falsification=m["falsification"], levers=levers,
                predictions={PANEL: AxisPrediction(int(m["predicted_sign"]),
                                                   abs(float(m.get("predicted_size") or 0)) or cfg.panel_sd)},
                source="review", review_id=review_id, created_ts=now)
            metas.append(h)
            probes += [(h.id, p) for p in m.get("probes") or []]
        probes = self._valid_probes(store, review_id, probes)
        findings = [f"[{a['lens']}/{a['status']}] {a['assumption']} -- {a['evidence']}"
                    for a in out.get("assumptions") or []] + list(out.get("findings") or ())
        return ReviewReport(
            id=review_id, task_id=self.ctx.task_id, trigger=trigger, summary=out["summary"],
            findings=tuple(findings), meta_hypothesis_ids=tuple(h.id for h in metas),
            probe_briefs=tuple(probes), explore_bias=dict(out.get("explore_bias") or {}),
            n_evals_at=snapshot.n_scored, created_ts=now,
            design=json.dumps(out["design"], default=str) if out.get("design") else "",
            selftune=json.dumps(out["selftune"], default=str) if out.get("selftune") else "",
            proposals=tuple(f"{p['change']} -- {p.get('rationale', '')}" for p in out.get("proposals") or [])), metas

    def _valid_probes(self, store: Any, review_id: str,
                      probes: list[tuple[str | None, str]]) -> list[tuple[str | None, str]]:
        """Drop (and log ``probe_dropped``) briefs that could never pass intake."""
        keep, protected = [], tuple(self.ctx.spec.surface.protected_keys)
        for meta_id, text in probes:
            if why := probe_violation(text, protected):
                store.log("probe_dropped", review_id=review_id, meta_id=meta_id, brief=text[:500], reason=why)
            else:
                keep.append((meta_id, text))
        return keep


# ------------------------------------------------ first-principles evidence (code-computed)
def _sd(xs: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip().startswith("{")]
    except (OSError, ValueError):
        return []


def _root_done(store: Any) -> list[Any]:
    return sorted((r for r in store.runs(node_id=ROOT_NODE_ID, status=[RunStatus.DONE]) if r.output_dir),
                  key=lambda r: r.submitted_ts or 0.0)


def throughput_block(ctx: LoopContext, store: Any, now: float) -> str:
    """Slots, GPU hours by outcome, void types, and the last 24 h's pace."""
    from loop import selftune
    from loop.driver import pool_use                  # loop.driver imports loop.agents
    runs, nodes = store.runs(), store.nodes()
    gpu, n = Counter(), Counter()
    for r in runs:                                    # the loop's own cancels are not voids
        k = "cancelled" if r.void_type is VoidType.CANCELLED else r.status.value
        gpu[k] += r.gpu_h or 0.0
        n[k] += 1
    done = sorted(r.gpu_h for r in runs if r.status is RunStatus.DONE and r.gpu_h)
    day = now - 86400
    closed = [x for x in nodes if x.status in CLOSED_STATUSES and (x.updated_ts or 0) >= day]
    return "\n".join([
        f"slots: {ctx.cfg.capacity} [{pool_use(ctx, runs)}]",
        *filter(None, [selftune.preempt_line(ctx, store, now)]),
        "runs by status (n, gpu_h): " + ", ".join(f"{k} {n[k]} ({gpu[k]:.1f})" for k in sorted(n)),
        "void types: " + (", ".join(f"{k} {v}" for k, v in Counter(
            r.void_type.value for r in runs if r.void_type and r.void_type is not VoidType.CANCELLED).most_common())
                                 or "none"),
        f"median gpu_h per finished run: {done[len(done) // 2]:.2f}" if done else "no finished run",
        f"last 24 h: {sum((r.submitted_ts or 0) >= day for r in runs)} runs submitted, {len(closed)} nodes closed "
        f"({Counter(x.status.value for x in closed).most_common()}), "
        f"{sum((x.created_ts or 0) >= day for x in nodes)} nodes created"])


def variance_block(ctx: LoopContext, store: Any, snapshot: WorldSnapshot) -> str:
    """Per target: declared vs current paired sigma and the raw sd over root's seeds; per dataset:
    the sd over root's seeds of every key of the last ``probe`` event in its metrics.jsonl."""
    root = _root_done(store)
    out = ["target | declared paired sd | current sigma | raw sd over root seeds (n)"]
    for t in (PANEL, *ctx.cfg.axes):
        xs = [r.aggregate if t == PANEL else r.scores.get(t) for r in root]
        xs = [x for x in xs if x is not None]
        out.append(f"{t} | {ctx.cfg.sigma(t):.5f} | {snapshot.sigma.get(t, float('nan')):.5f} | "
                   f"{_sd(xs):.5f} ({len(xs)})")
    per: dict[str, list[float]] = {}
    for r in root:
        ev = [e for e in _jsonl(Path(r.output_dir) / "metrics.jsonl") if e.get("event") == "probe"]
        for k, v in (ev[-1] if ev else {}).items():
            if isinstance(v, (int, float)) and not re.search(r"step|flops|fraction|seconds|epochs|repeats", k):
                per.setdefault(k, []).append(float(v))
    rows = sorted(((k, sum(v) / len(v), _sd(v), len(v)) for k, v in per.items() if len(v) > 1),
                  key=lambda x: -x[2] if math.isfinite(x[2]) else 0)
    out += ["", "per-dataset key | mean | sd over root seeds | n"] + [
        f"{k} | {m:.4f} | {sd:.4f} | {c}" for k, m, sd, c in rows[:60]] if rows else ["", "(no per-dataset metrics)"]
    return "\n".join(out)


def run_metrics_block(store: Any) -> str:
    """The latest finished root run's summary scalars and its first/last training records (final
    lr / wd / momentum, the schedule fractions reached at the stop)."""
    root = _root_done(store)
    if not root:
        return "(no finished root run)"
    d = Path(root[-1].output_dir)
    try:
        summ = json.loads((d / "summary.json").read_text())
    except (OSError, ValueError):
        summ = {}
    scal = {k: v for k, v in summ.items() if isinstance(v, (int, float, str)) and len(str(v)) < 80
            and not k.startswith("final_probe_")}          # per-dataset scores: the variance block
    train = [e for e in _jsonl(d / "metrics.jsonl") if "lr" in e and "event" not in e]
    return "\n".join([f"run {root[-1].id} ({d})", f"summary: {json.dumps(scal)[:3000]}" if scal else "summary: (none)",
                      *(f"{w} training record: {json.dumps(e)[:1500]}" for w, e in
                        (("first", train[0]), ("last", train[-1])) if train)])
