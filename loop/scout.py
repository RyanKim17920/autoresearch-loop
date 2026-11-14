"""loop.scout — the RESEARCH SCOUT: a periodic Opus session with web tools (every ``scout.every_hours``,
self-tunable 2..24 h; also once per stall) that does broad background research and writes 8-15 concrete,
CITED ideas to an append-only JSONL idea bank (``scout.path``, listed in ``loop.idea_banks``), in the v1
bank schema. The LLM session runs in the driver's pool (:meth:`Scout.run`); the driver alone writes
(:func:`write`): validation, dedupe against tried mechanisms and every existing bank id, append, book.
Its outcomes (hit rate / confirmations of nodes spawned from its ideas) are the ``scout`` component in
loop.meta."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Mapping

from loop.contracts import ROOT_NODE_ID

K_SCOUT = "scout.book"            # {"runs": [{ts, trigger, ok, ideas, written, dup, refused, searches, ...}], ...}
RETRY_FAILED_S = 3600.0           # a failed session is retried after this long (not every_hours)
RUNS_KEEP = 200
_STOP = frozenset("with from into that this their using based via for and the of to in on a an".split())


def slug(text: str, n: int = 60) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", (text or "").lower())).strip("_")[:n].strip("_")


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]{3,}", (text or "").lower()) if w not in _STOP}


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def path(ctx: Any) -> str:
    return ctx.cfg.scout.path or str(Path(ctx.state_dir) / "idea_bank_scout.jsonl")


def book(store: Any) -> dict[str, Any]:
    return {"runs": [], "stall_at": None, **(store.get_state(K_SCOUT) or {})}


def due(ctx: Any, store: Any, now: float, *, evals_since_confirm: int, confirm_marker: Any) -> str:
    """'' or the trigger: 'first' (never ran), 'cadence' (every_hours since the last good run; a failed one
    is retried after RETRY_FAILED_S) or 'stall' (review.stall_evals closes without a confirmation: once per
    stall, and not within stall_min_hours of the last run)."""
    sc = ctx.cfg.scout
    if not sc.enabled:
        return ""
    b = book(store)
    runs = b["runs"]
    if not runs:
        return "first"
    last = runs[-1]
    gap = now - float(last.get("ts", 0.0))
    if gap >= (3600.0 * sc.every_hours if last.get("ok") else min(RETRY_FAILED_S, 3600.0 * sc.every_hours)):
        return "cadence"
    stall = ctx.cfg.review.stall_evals
    if stall > 0 and evals_since_confirm >= stall and b.get("stall_at") != confirm_marker \
            and gap >= 3600.0 * sc.stall_min_hours:
        return "stall"
    return ""


def finish_schema(ctx: Any) -> dict[str, Any]:
    txt = {"type": "string", "minLength": 1}
    return {"type": "object", "required": ["ideas"], "properties": {"ideas": {
        "type": "array", "minItems": 1, "items": {
            "type": "object", "required": ["title", "family", "moonshot", "jepa", "idea", "why_it_might_work", "source"],
            "properties": {"title": txt, "mechanism_tag": {"type": "string"}, "family": {"type": "string"},
                           "moonshot": {"type": "boolean"}, "jepa": {"type": "boolean"}, "idea": txt,
                           "why_it_might_work": txt, "source": {"type": "array", "items": {"type": "string"}}}}}}}


def ledger_summary(store: Any, snapshot: Any, limit: int = 160) -> str:
    """What has been tried: per family, the counts by outcome, then every distinct mechanism tag with its
    outcome (shrunk U when scored), newest first."""
    from collections import Counter, defaultdict
    nodes = [n for n in store.nodes() if n.id != ROOT_NODE_ID and n.strategy.value != "design"]
    fam: dict[str, Counter] = defaultdict(Counter)
    for n in nodes:
        fam[n.family][n.status.value] += 1
    lines = [f"{f}: " + ", ".join(f"{k} {v}" for k, v in c.most_common()) for f, c in sorted(fam.items())]
    seen, tags = set(), []
    for n in sorted(nodes, key=lambda n: -(n.created_ts or 0.0)):
        if n.mechanism_tag in seen:
            continue
        seen.add(n.mechanism_tag)
        post = snapshot.nodes.get(n.id) if snapshot is not None else None
        tags.append(f"{n.family}/{n.mechanism_tag} {n.status.value}"
                    + (f" U={post.utility.mean:+.4f} P(U>0)={post.p_positive:.2f}" if post is not None else ""))
    more = f"\n... {len(tags) - limit} older tags omitted" if len(tags) > limit else ""
    return ("Outcomes by family:\n" + ("\n".join(lines) or "(nothing yet)")
            + "\n\nMechanisms tried (family/tag outcome):\n" + ("\n".join(tags[:limit]) or "(none)") + more)


class _NoTools:
    """The scout has no worktree tools: only the web, the research notes (loop.agents.notes) and ``finish``."""

    def __init__(self, research: Any = None) -> None:
        self.research = research

    def on_web(self, name: str, args: Any) -> None:
        if self.research is not None:
            self.research.on_web(name, args)

    def check_finish(self, args: Mapping[str, Any]) -> str:
        return self.research.missing(args) if self.research is not None else ""

    def __call__(self, name: str, args: Mapping[str, Any]) -> str:
        if self.research is not None and name in ("notes", "note"):
            return getattr(self.research, f"t_{name}")(**dict(args))
        return f"ERROR: no tool {name!r} here; use WebSearch / WebFetch, notes / note, then finish"


class Scout:
    def __init__(self, ctx: Any, llm: Any) -> None:
        self.ctx, self.llm = ctx, llm

    def prompt(self, store: Any, snapshot: Any, now: float) -> str:
        from loop import meta, novelty as NV
        from loop.agents import briefs
        from loop.design import lines as design_lines
        from loop.select.bank import load
        cfg = self.ctx.cfg
        bank = load(tuple(cfg.idea_banks) + (path(self.ctx),), store)
        try:
            record = meta.track_record(self.ctx, store, snapshot, "scout")
        except Exception as exc:  # noqa: BLE001 - analytics never blocks the scout
            record = f"(track record unavailable: {type(exc).__name__})"
        return briefs.render(
            "scout", task_id=self.ctx.task_id, date=time.strftime("%Y-%m-%d", time.gmtime(now)),
            min_ideas=cfg.scout.min_ideas, max_ideas=cfg.scout.max_ideas,
            focus=briefs.focus_block(self.ctx) + "".join(x + "\n\n" for x in briefs.research_blocks(self.ctx, store)),
            ledger=ledger_summary(store, snapshot), bank=", ".join(e["id"] for e in bank) or "(none)",
            lessons="\n".join(filter(None, (design_lines(store), briefs.optional_block(
                "ideation_lessons", store, snapshot)))) or "(nothing yet)",
            record=record, constraints=briefs.hard_constraints(self.ctx), families=sorted(cfg.families),
            cells=NV.scout_block(store, snapshot) if cfg.novelty.enabled else "")

    def run(self, store: Any, snapshot: Any, *, trigger: str, now: float) -> dict[str, Any]:
        """The web session (no store writes): {trigger, ts, ok, error, ideas, searches, fetches}."""
        from loop.agents import notes as NT
        sc, rid = self.ctx.cfg.scout, f"scout-{int(now)}"
        res = self.llm.run_agent(
            "scout", system="You are a research scout for an ML research loop. Search the web, read methods, "
                            "and end with `finish`. Never invent citations.",
            prompt=self.prompt(store, snapshot, now) + "\n\n" + NT.prompt(self.ctx, rid).replace(
                "hypothesis.builds_on names", "each idea names"), cwd=str(self.ctx.state_dir or "."),
            tools=list(NT.tool_defs()), executor=_NoTools(NT.ResearchStage(self.ctx, rid)),
            finish_schema=finish_schema(self.ctx), max_turns=sc.max_turns,
            timeout_s=60.0 * sc.timeout_min, web=True)
        return {"trigger": trigger, "ts": now, "ok": bool(res.ok and res.finish), "error": res.error[:300],
                "ideas": list((res.finish or {}).get("ideas") or []), "searches": res.web_searches,
                "fetches": res.web_fetches, "transcript": res.transcript_path}


def _clean(ctx: Any, raw: Mapping[str, Any], day: str) -> tuple[dict[str, Any] | None, str]:
    """(bank line, '') or (None, why refused)."""
    from loop.select.bank import SCORE
    title, idea = " ".join(str(raw.get("title", "")).split()), " ".join(str(raw.get("idea", "")).split())
    if not title or not idea:
        return None, "empty"
    if SCORE.search(title) or SCORE.search(idea):
        return None, "quotes a score"
    src = [str(u).strip() for u in raw.get("source") or [] if str(u).strip().startswith("http")]
    if not src:
        return None, "uncited"
    fam = str(raw.get("family", ""))
    return {"id": f"scout-{slug(str(raw.get('mechanism_tag') or ''), 48) or slug(title, 48)}", "title": title[:200],
            "family": fam if fam in ctx.cfg.families else "", "moonshot": bool(raw.get("moonshot")),
            "jepa": bool(raw.get("jepa")), "idea": idea,
            "why_it_might_work": " ".join(str(raw.get("why_it_might_work", "")).split()),
            "v1_fate": f"n/a (scouted {day})", "why_v1_verdict_may_not_transfer": "", "source": src[:6],
            **({"priority": True} if raw.get("priority") else {})}, ""


def write(ctx: Any, store: Any, result: Mapping[str, Any], *, confirm_marker: Any = None) -> dict[str, Any]:
    """Validate, dedupe (bank ids of every bank, tried mechanism tags, near-duplicate titles, a bank idea's mechanism
    within novelty.mech_dup_cos: loop.novelty.mech_cos over tag + title -- not for an operator's priority idea, a
    deliberate variant of a related one: ANVIL2 vs the bank's Muon idea) and APPEND the
    run's ideas (at most scout.max_ideas); record the run in the book and the meta decision log."""
    from loop import meta, novelty as NV
    from loop.select.bank import load
    p = path(ctx)
    day = time.strftime("%Y-%m-%d", time.gmtime(float(result.get("ts") or time.time())))
    bank = load(tuple(ctx.cfg.idea_banks) + (p,), store)
    ids = {e["id"] for e in bank}
    titles = [_words(e["title"]) for e in bank]
    tags = {NV.norm_tag(n.mechanism_tag) for n in store.nodes() if n.mechanism_tag}
    mechs = [NV.mech_terms(e["id"].removeprefix("scout-"), e["title"]) for e in bank]
    thr = ctx.cfg.novelty.mech_dup_cos
    out, dup, refused = [], 0, []
    for raw in result.get("ideas") or []:
        if not isinstance(raw, Mapping):
            continue
        line, why = _clean(ctx, raw, day)
        if line is None:
            refused.append(why)
            continue
        w = _words(line["title"])
        mtag = NV.norm_tag(str(raw.get("mechanism_tag") or ""))
        me = NV.mech_terms(line["id"].removeprefix("scout-"), line["title"])  # live: whitened target x2, 5 rebalances
        if line["id"] in ids or (mtag and mtag in tags) or NV.norm_tag(line["title"]) in tags \
                or any(_jaccard(w, t) >= 0.6 for t in titles) \
                or (thr > 0 and not line.get("priority") and max(NV.mech_cos(me, mechs), default=0.0) >= thr):
            dup += 1
            continue
        if len(out) >= ctx.cfg.scout.max_ideas:
            refused.append("over max_ideas")
            continue
        ids.add(line["id"])
        titles.append(w)
        mechs.append(me)
        out.append(line)
    if out:
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a") as fh:                    # append-only: the bank's served-once book keys on ids
            fh.write("".join(json.dumps(x) + "\n" for x in out))
    run = {k: result.get(k) for k in ("trigger", "ts", "ok", "error", "searches", "fetches", "transcript")}
    run.update(ideas=len(result.get("ideas") or []), written=len(out), dup=dup, refused=len(refused),
               ids=[x["id"] for x in out])
    b = book(store)
    b["runs"] = (b["runs"] + [run])[-RUNS_KEEP:]
    if result.get("trigger") == "stall":
        b["stall_at"] = confirm_marker
    store.set_state(K_SCOUT, b)
    store.log("scout_run", **{k: v for k, v in run.items() if k not in ("ids", "ts")}, reasons=refused[:10])
    meta.record(store, "scout", f"scout-{int(float(result.get('ts') or 0))}", ideas=run["ids"],
                searches=run["searches"], fetches=run["fetches"])
    return run


def lines(ctx: Any, store: Any, now: float) -> list[str]:
    """``loop status`` / dashboard: scout runs, ideas written, the last run and when the next is due."""
    b = book(store)
    runs = b["runs"]
    if not ctx.cfg.scout.enabled:
        return ["scout    disabled"]
    if not runs:
        return ["scout    no runs yet (the first runs at the next tick)"]
    last = runs[-1]
    nxt = float(last.get("ts", 0.0)) + 3600.0 * ctx.cfg.scout.every_hours - now
    return [f"scout    {len(runs)} runs ({sum(bool(r.get('ok')) for r in runs)} ok), "
            f"{sum(int(r.get('written') or 0) for r in runs)} ideas written "
            f"({sum(int(r.get('dup') or 0) for r in runs)} dup, {sum(int(r.get('refused') or 0) for r in runs)} refused); "
            f"last {time.strftime('%Y-%m-%d %H:%M', time.gmtime(float(last.get('ts', 0.0))))} ({last.get('trigger')}, "
            + (f"{last.get('written')} written, {last.get('searches')} searches" if last.get("ok")
               else f"FAILED: {str(last.get('error'))[:80]}")
            + f"); next in {max(0.0, nxt) / 3600:.1f} h (every {ctx.cfg.scout.every_hours:g} h)"]
