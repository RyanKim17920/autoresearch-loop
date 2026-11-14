"""loop.select.bank — operator idea banks (``loop.idea_banks``): JSONL ideas served as briefs.

Each line: {id, title, family, moonshot, idea, why_it_might_work, v1_fate, why_v1_verdict_may_not_transfer,
source, priority?, plan?}. A ``priority`` idea is served before the rest (the operator's "draw it soon"); within each
group ideas interleave across families. ``plan`` (the loop's generated bank only) carries a research session's
first-principles record and branched variant plan, so the idea is implemented later without re-researching.
An idea is served ONCE (one retry if its research attempt failed). CLEAN SLATE: a field quoting an old score (0.6xxx, +0.0xx) is stripped; if the idea or title
itself quotes one, the line is refused. A missing file is an empty bank; files re-read on mtime change.
"""

from __future__ import annotations

import json
import re
from itertools import zip_longest
from pathlib import Path
from typing import Any

GENERATED = "idea_bank_generated.jsonl"   # the loop's own bank (loop.agents.ideas), in its state dir
K_GENERATED = "select.bank_generated"     # the generated bank's path (set when the loop first writes it)
K_BANK_USED = "select.bank_used"          # {bank id: [request ids that served it]}
TEXT_FIELDS = ("title", "idea", "why_it_might_work", "v1_fate", "why_v1_verdict_may_not_transfer")
EXTRA_FIELDS = ("evidence", "axis", "knobs")   # optional; rendered by :func:`text` when present
REQUIRED = ("title", "idea")               # a score in one of these refuses the whole line
SCORE = re.compile(r"(?<![\w.])0\.[5-7]\d{2,}(?!\d)|[+\-−±]\s?0?\.0\d+")
_CACHE: dict[str, tuple[float, list[dict], list[tuple[str, str, list[str]]]]] = {}


def _source(v: Any) -> str:
    """A bank line's source: a string, or a list of URLs (the scout's) joined with '; '."""
    return "; ".join(str(x) for x in v if x) if isinstance(v, (list, tuple)) else str(v or "")


def _parse(path: str) -> tuple[list[dict], list[tuple[str, str, list[str]]]]:
    """(clean entries, problems as (event, id, fields))."""
    out, problems = [], []
    for i, line in enumerate(Path(path).read_text(errors="replace").splitlines()):
        try:
            row = json.loads(line) if line.strip() else None
        except json.JSONDecodeError:
            problems.append(("bank_bad_line", f"{Path(path).name}:{i + 1}", []))
            continue
        if not isinstance(row, dict) or not str(row.get("id") or "").strip() or not row.get("idea"):
            continue
        e = {"id": str(row["id"]).strip(), "family": str(row.get("family") or ""),
             "moonshot": bool(row.get("moonshot")), "priority": bool(row.get("priority")),
             "plan": row["plan"] if isinstance(row.get("plan"), dict) else {},
             "source": _source(row.get("source")) or Path(path).name,
             **{f: str(row.get(f) or "") for f in TEXT_FIELDS},
             # optional research-agent fields (nanobrain banks): kept as given, not scrubbed (evidence may cite numbers)
             **{f: (row[f] if isinstance(row[f], (dict, list)) else str(row[f]))
                for f in EXTRA_FIELDS if row.get(f) not in (None, "", [], {})}}
        dirty = [f for f in TEXT_FIELDS if SCORE.search(e[f])]
        if any(f in REQUIRED for f in dirty):
            problems.append(("bank_refused", e["id"], dirty))
            continue
        if dirty:
            problems.append(("bank_scrubbed", e["id"], dirty))
            e.update({f: "" for f in dirty})
        out.append(e)
    return out, problems


def load(paths: Any, store: Any = None) -> list[dict]:
    """Every bank's clean ideas in file order, first id wins; a (re)parse logs its problems."""
    seen, out = set(), []
    for p in paths or ():
        p = str(Path(p).expanduser())
        try:
            mtime = Path(p).stat().st_mtime
        except OSError:
            continue
        if p not in _CACHE or _CACHE[p][0] != mtime:
            try:
                entries, problems = _parse(p)
            except OSError:
                continue
            _CACHE[p] = (mtime, entries, problems)
            for event, bid, fields in problems:
                if hasattr(store, "log"):
                    store.log(event, bank_id=bid, fields=fields, path=p)
        for e in _CACHE[p][1]:
            if e["id"] not in seen:
                seen.add(e["id"])
                out.append(e)
    return out


def _failed(store: Any) -> set[str]:
    if not hasattr(store, "events"):
        return set()
    return {e.get("request_id") for e in store.events("research_failed") + store.events("research_error")}


def _used(store: Any) -> dict[str, list[str]]:
    """bank id -> the requests that served it, an outage's (no attempt at all) left out."""
    from loop.contracts import outage_requests
    used = store.get_state(K_BANK_USED) or {}
    out = outage_requests(store) if used else set()
    return {k: [r for r in v if r not in out] for k, v in used.items()}


def paths(cfg: Any, store: Any = None) -> tuple[str, ...]:
    """The configured banks plus the loop's own GENERATED bank (``K_GENERATED``: the ideas a research session
    proposed but did not implement, loop.agents.ideas, in its state dir)."""
    gen = (store.get_state(K_GENERATED) or "",) if store is not None and hasattr(store, "get_state") else ()
    return tuple(dict.fromkeys(p for p in (*(cfg.idea_banks or ()), *gen) if p))


def interleave(entries: list[dict]) -> list[dict]:
    """Round-robin over families (first-seen order, file order within each): no single theme drains the bank
    first. No mechanism is privileged by name (operator 2026-10-06: a theme named once became 248 nodes)."""
    groups: dict[str, list[dict]] = {}
    for e in entries:
        groups.setdefault(e["family"] or "-", []).append(e)
    return [e for row in zip_longest(*groups.values()) for e in row if e is not None]


def pending(store: Any, cfg: Any) -> list[dict]:
    """Unserved ideas (or served once by a failed research attempt): priority ideas first, each group interleaved
    across families (:func:`interleave`)."""
    used = _used(store)
    failed = _failed(store) if any(len(v) == 1 for v in used.values()) else set()
    out = [e for e in load(paths(cfg, store), store)
           if not (reqs := used.get(e["id"])) or (len(reqs) == 1 and reqs[0] in failed)]
    return interleave([e for e in out if e["priority"]]) + interleave([e for e in out if not e["priority"]])


def mark(store: Any, bank_id: str, request_id: str) -> None:
    used = dict(store.get_state(K_BANK_USED) or {})
    used[bank_id] = list(used.get(bank_id, [])) + [request_id]
    store.set_state(K_BANK_USED, used)


def text(e: dict) -> str:
    """The idea as brief text: never a number from the old project (scrubbed at load)."""
    rows = [("Idea", e["idea"]), ("Why it might work", e["why_it_might_work"]),
            ("What happened before", e["v1_fate"]),
            ("Why that verdict may not transfer", e["why_v1_verdict_may_not_transfer"])]
    if e.get("plan"):                       # the loop's own generated idea: implement the researched plan
        return "\n".join([f"IDEA BANK idea `{e['id']}` (researched earlier, ranked {e['plan'].get('rank', '?')} of its "
                          f"slate): {e['title']} (sources: {e['source']})"]
                         + [f"- {k}: {v}" for k, v in rows[:2] if v]
                         + ["Its PLAN (files, keys, init stance, first-principles record, variants, branched plan) -- "
                            "implement it as planned, re-check the code it names, and carry the record, variants and "
                            "plan into finish (no new slate):", json.dumps(e["plan"], default=str)[:6000]])
    extra = [(k.capitalize(), e[k] if isinstance(e[k], str) else json.dumps(e[k], default=str))
             for k in EXTRA_FIELDS if e.get(k)]
    return "\n".join([f"IDEA BANK idea `{e['id']}`: {e['title']} (source: {e['source']})"]
                     + [f"- {k}: {v}" for k, v in rows + extra if v]
                     + ["Its earlier verdict came from a DIFFERENT eval suite: test it from a clean slate here, "
                        "with no score expectations carried over."])
