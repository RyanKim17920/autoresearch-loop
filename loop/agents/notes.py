"""loop.agents.notes — the RESEARCH STAGE of moonshot / literature / scout ideation (operator 2026-10-06: "way way
deeper moonshot research -- you can't ideate good moonshots without good research"; audit: 286 of 311 web
sessions never fetched a page, typically one search).

Two stages in one session, enforced mechanically from the session's own tool log:

1. RESEARCH: search the shared notes first (``notes``: what earlier sessions read, so nothing is read twice), then
   WebSearch (>= ``researcher.min_searches``) and READ papers in full (WebFetch of the paper's HTML / full text, an
   arXiv abs + its method sections -- never a search snippet), and record one structured note per paper (``note``:
   claim, mechanism, setting, reported effect, and why it might / might not transfer to our regime). A note's URL
   must have been fetched in THIS session. No topical filter: a share of sessions (``researcher.distant_share``) is
   asked to range into distant fields (other modalities, physics, signal processing, optimization theory,
   neuroscience, ...); the ``transfer`` field is where relevance is argued.
2. IDEATE: the finish cites the note ids it builds on and how (``hypothesis.builds_on``; the scout: in its ideas).

``finish`` is bounced ONCE with what is missing unless the session ran >= min_searches searches, recorded >=
``researcher.min_new_papers`` new notes, and its new + cited notes cover >= ``researcher.min_papers`` distinct
papers, and the finish cites >= 1 note. Notes are appended to ``<state_dir>/research_notes/notes.jsonl`` (the index)
and, append-only and human-readable, to ``<state_dir>/research_notes/<topic>.md``; every later session reuses them and
each v3 brief carries the top-k by embedding similarity (:func:`retrieve`, loop.core.ideation)."""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Mapping

from loop import project as P

NOTE_ID = re.compile(r"\brn-[0-9a-f]{8}\b")
ARXIV = re.compile(r"(\d{4}\.\d{4,5})")
FIELDS = ("claim", "mechanism", "setting", "effect", "transfer")
REGIME = ("DINOv2-initialised ViT pathology SSL, a fixed 1M-sample / 1e18-FLOP budget (truncated, FLOP-keyed "
          "schedules), 224px TCGA tiles, a saturated carrier (the incumbent already stacks many legs)")


def regime() -> str:
    """The task's training regime in one line (``domain.rules.regime``; default: nanopath's)."""
    return P.rule("regime", REGIME)


def _tool(name: str, desc: str, required: tuple[str, ...] = (), **props: str) -> dict[str, Any]:
    return {"name": name, "description": desc, "input_schema": {
        "type": "object", "required": list(required), "properties": {k: {"type": v} for k, v in props.items()}}}


def tool_defs() -> tuple[dict[str, Any], ...]:
    """The notes tools, built at call time (the regime text is the INSTALLED task's, loop.project.rule)."""
    return (
        _tool("notes", "Search the shared research notes (papers earlier sessions READ): keywords -> matching notes with "
                       "their ids. Search here first; a note you build on is cited by id, not re-read.", ("query",),
              query="string"),
        _tool("note", "Record ONE paper you have READ in this session (WebFetch of its full text / HTML / arXiv abs + "
                      "method sections; the url must be one you fetched): its claim, mechanism, setting, reported effect "
                      f"(numbers), and transfer: why it might or might not transfer to our regime ({regime()}); topic: a short "
                      "snake_case subject the note is filed under (e.g. teacher_ema, stain_invariance). Returns its note id "
                      "(rn-...).", ("url", "title", "topic", *FIELDS), url="string", title="string", topic="string",
              **{f: "string" for f in FIELDS}),
    )


def key(url: str) -> str:
    """A URL's identity: its arXiv id when it has one, else host + path (no scheme / www / query / trailing /)."""
    if m := ARXIV.search(url or ""):
        return f"arxiv:{m.group(1)}"
    u = re.sub(r"^[a-z]+://", "", (url or "").strip().lower())
    return re.sub(r"^www\.", "", u.split("#")[0].split("?")[0]).rstrip("/")


def notes_path(ctx: Any, branch: str = "") -> Path:
    """The shared notes index; a fork-and-flush branch's own: ``<notes dir>/branch-<id>/notes.jsonl`` (loop.core.fnf)."""
    p = getattr(ctx.cfg.researcher, "notes_path", "") or ""
    path = Path(p) if p else Path(ctx.state_dir or ".") / "research_notes" / "notes.jsonl"
    return path.parent / f"branch-{branch}" / path.name if branch else path


def topic_slug(topic: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (topic or "").lower()).strip("_")[:48] or "misc"


def append_md(path: Path, note: Mapping[str, Any]) -> None:
    """Append one note to ``<notes dir>/<topic>.md`` (append-only; the jsonl index is the machine copy)."""
    md = path.parent / f"{topic_slug(str(note.get('topic') or ''))}.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    when = time.strftime("%Y-%m-%d", time.gmtime(float(note.get("ts") or time.time())))
    with md.open("a") as f:
        f.write(f"\n## {note['id']} {note.get('title', '')}\n<{note.get('url', '')}> ({note.get('request', '')}, {when})\n"
                + "".join(f"- {x}: {note.get(x, '')}\n" for x in FIELDS))


def text_of(note: Mapping[str, Any]) -> str:
    return " ".join(str(note.get(k) or "") for k in ("title", "topic", "claim", "mechanism", "transfer"))


def retrieve(path: Path, query: str, k: int = 5, embedder: Any = None) -> list[dict[str, Any]]:
    """The ``k`` notes most similar to ``query`` (cosine of hashed n-gram embeddings; loop.select.embed)."""
    from loop.select.embed import HashingEmbedder, cosine
    emb = embedder or HashingEmbedder()
    q = emb.embed(query)
    rows = load(path)
    return sorted(rows, key=lambda n: -cosine(q, emb.embed(text_of(n))))[:k] if rows and query.strip() else []


def load(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip().startswith("{")]
    except OSError:
        return []


def distant(request_id: str, share: float) -> bool:
    """Whether this session is one asked to range into distant fields (a stable ``share`` of request ids)."""
    return share > 0 and int(hashlib.sha1(request_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < share


def prompt(ctx: Any, request_id: str) -> str:
    r = ctx.cfg.researcher
    far = (f"\nTHIS SESSION ranges FAR: spend at least half of your reading OUTSIDE {P.rule('analogy_field', 'pathology / SSL')} -- other modalities, "
           "physics, signal processing, optimization theory, neuroscience, statistics, ... -- mechanisms sometimes "
           "transfer unexpectedly; argue the transfer in each note.\n") if distant(request_id, r.distant_share) else ""
    return ("## Stage 1 -- RESEARCH (enforced: finish is bounced until it is done)\n"
            f"1. `notes` first: search what earlier sessions already read (reuse a note by citing its id).\n"
            f"2. WebSearch at least {r.min_searches} times, then READ at least {r.min_new_papers} new papers in full with "
            "WebFetch (the paper's HTML / full text, or its arXiv abs page AND its method section) -- a search snippet "
            "is not reading. Any field is fair game; relevance is argued in the note, not assumed.\n"
            f"3. After reading each paper call `note` (claim, mechanism, setting, reported effect with numbers, "
            f"transfer: why it might or might not work in OUR regime: {regime()}).\n"
            f"In all you must build on at least {r.min_papers} distinct papers (your new notes + earlier notes you "
            "cite).\n" + far +
            "## Stage 2 -- IDEATE from the notes\n"
            "Design the change from what you read. In the finish, hypothesis.builds_on names the note ids (rn-...) it "
            "builds on and HOW each one shaped the change; cite the papers in hypothesis.claim / prior_work too.\n")


class ResearchStage:
    """The session's research ledger: the ``notes`` / ``note`` tools, the live web log (``on_web``), and the finish
    gate (:meth:`missing`)."""

    def __init__(self, ctx: Any, request_id: str, branch: str = "") -> None:
        self.ctx, self.request_id, self.path = ctx, request_id, notes_path(ctx, branch)
        self.searches: list[str] = []
        self.fetched: set[str] = set()
        self.new: list[str] = []
        self.bounced = False

    def on_web(self, name: str, args: Mapping[str, Any] | None) -> None:
        args = dict(args or {})
        if name == "WebSearch":
            self.searches.append(str(args.get("query") or ""))
        elif name == "WebFetch" and args.get("url"):
            self.fetched.add(key(str(args["url"])))

    def t_notes(self, query: str = "") -> str:
        words = {w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 2}
        rows = load(self.path)
        hits = sorted(((sum(w in json.dumps(n).lower() for w in words), n) for n in rows), key=lambda x: -x[0])
        hits = [n for s, n in hits if s > 0][:8]
        if not hits:
            return f"no notes match {query!r} ({len(rows)} notes in all)"
        return "\n".join(f"{n['id']} {n.get('title', '')[:90]} <{n.get('url', '')}>\n  claim: {n.get('claim', '')[:220]}"
                         f"\n  transfer: {n.get('transfer', '')[:220]}" for n in hits)

    def t_note(self, url: str = "", title: str = "", topic: str = "", **fields: Any) -> str:
        k = key(url)
        if k not in self.fetched:
            return (f"ERROR: {url!r} was not fetched in this session. READ it first (WebFetch the paper's full text / "
                    "HTML / arXiv abs + method section), then note it")
        if bad := [f for f in FIELDS if len(str(fields.get(f) or "").strip()) < 20]:
            return f"ERROR: say more in {bad} (claim, mechanism, setting, effect with numbers, transfer to our regime)"
        nid = "rn-" + hashlib.sha1(k.encode()).hexdigest()[:8]
        if nid not in {n["id"] for n in load(self.path)}:
            row = {"id": nid, "url": url, "key": k, "title": title, "topic": topic_slug(topic),
                   **{x: str(fields.get(x) or "") for x in FIELDS}, "request": self.request_id, "ts": time.time()}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            append_md(self.path, row)
        if nid not in self.new:
            self.new.append(nid)
        return f"noted {nid} ({len(self.new)} new this session)"

    def missing(self, finish: Mapping[str, Any]) -> str:
        """'' when the research stage is done and the finish builds on it; else what is missing (bounces ONCE)."""
        if self.bounced:
            return ""
        r = self.ctx.cfg.researcher
        known = {n["id"] for n in load(self.path)}
        cited = set(NOTE_ID.findall(json.dumps(finish, default=str))) & known
        papers = set(self.new) | cited
        gaps = [*([f"{len(self.searches)}/{r.min_searches} WebSearch calls"] if len(self.searches) < r.min_searches else []),
                *([f"{len(self.new)}/{r.min_new_papers} new papers read (WebFetch) and noted"]
                  if len(self.new) < r.min_new_papers else []),
                *([f"{len(papers)}/{r.min_papers} distinct papers built on (new notes + cited earlier notes)"]
                  if len(papers) < r.min_papers else []),
                *(["no note id (rn-...) cited in the finish (hypothesis.builds_on: which notes, and how)"]
                  if not cited else [])]
        if not gaps:
            return ""
        self.bounced = True
        return ("RESEARCH STAGE incomplete: " + "; ".join(gaps) + ". Do the research (notes -> WebSearch -> WebFetch the "
                "full text -> note each paper), then finish citing the note ids you build on")
