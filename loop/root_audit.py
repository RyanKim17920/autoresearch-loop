"""loop.root_audit — the ROOT ASSUMPTION LEDGER (``loop.root_audit``; default off).

What the operator did by hand on 2026-11-08 (fasttrack/assumptions.md): list every choice baked into the root recipe
(each config key and each hard-coded constant of the training code), say who on the board tested it, with how many
seeds and with what per-axis verdict, and turn the untested / weakly tested ones into ranked audit arms. Here:

- :func:`due`: the audit key -- the root commit + its config path (+ the board digest's ``material`` hash when
  ``rerun_on_board``) -- differs from the last audit's.
- :func:`inputs`: the root config (``git show <base_sha>:<config_rel>``), the hard-coded constants of the root's source
  files (module-level ``NAME = <number>`` and numeric keyword defaults), and the board digest (loop.board).
- :class:`RootAudit` runs ONE review-tier ``complete_json`` call in the tick's pool (the schema below); :func:`accept`
  stores the ledger (state ``root_audit.ledger``) and writes its arms -- priority ideas tagged ``root_audit`` -- to the
  loop's own idea bank ``<state_dir>/idea_bank_root_audit.jsonl``, which the v3 briefs serve (``ideation.bank_v3``) to
  research sessions that build them through the normal intake gates.
- :func:`block`: the ledger as a brief section; :func:`outcomes`: per arm, its node and what it measured (the meta
  report's component audit reads it).
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from loop import board as BD

K_LEDGER = "root_audit.ledger"     # {key, ts, summary, assumptions: [...], arms: [ids]}
K_BOOK = "root_audit.book"         # {key, started_ts, error}
BANK = "idea_bank_root_audit.jsonl"
VERDICTS = ("real_gain", "real_harm", "null", "noise_only", "untested", "our_fork")
ARM_KINDS = ("ablate", "sweep_up", "sweep_down", "swap", "none")

SYSTEM = ("You audit the ROOT recipe of an ML research loop: enumerate the choices baked into it, join each with the "
          "competition board's evidence, and propose the audit experiments the evidence leaves open. Be exhaustive "
          "about choices and strict about evidence: a single run is one sample of seed noise. Never invent evidence.")

SCHEMA: dict[str, Any] = {"type": "object", "required": ["summary", "root_assumptions"], "properties": {
    "summary": {"type": "string"},
    "root_assumptions": {"type": "array", "items": {
        "type": "object", "required": ["choice", "where", "value", "verdict", "evidence", "arm"],
        "properties": {
            "choice": {"type": "string", "minLength": 1}, "where": {"type": "string"}, "value": {"type": "string"},
            "tested_by": {"type": "string"}, "seeds": {"type": "integer"},
            "verdict": {"type": "string", "enum": list(VERDICTS)},
            "evidence": {"type": "string"},
            "arm": {"type": "object", "required": ["kind"], "properties": {
                "kind": {"type": "string", "enum": list(ARM_KINDS)},
                "change": {"type": "string"}, "config_diff": {"type": "object"},
                "mechanism": {"type": "string"}, "axis": {"type": "string"},
                "priority": {"type": "number"}}}}}}}}


def cfg_of(ctx: Any) -> Any:
    return getattr(getattr(ctx, "cfg", None), "root_audit", None)


def enabled(ctx: Any) -> bool:
    return bool(getattr(cfg_of(ctx), "enabled", False))


def bank_path(ctx: Any) -> str:
    return str(Path(getattr(ctx, "state_dir", "") or ".") / BANK)


def audit_key(ctx: Any, store: Any) -> str:
    repo, sha, rel = BD.root_ref(ctx)
    mat = BD.digest(store).get("material", "") if getattr(cfg_of(ctx), "rerun_on_board", True) else ""
    return f"{sha[:12]}|{rel}|{mat}"


def due(ctx: Any, store: Any) -> bool:
    if not enabled(ctx):
        return False
    _, sha, _ = BD.root_ref(ctx)
    if not sha:
        return False
    if BD.enabled(ctx) and not BD.digest(store):        # wait for the first board digest (the audit joins it)
        return False
    return (store.get_state(K_LEDGER) or {}).get("key") != audit_key(ctx, store)


_CONST = re.compile(r"^([A-Z][A-Z0-9_]{2,})\s*(?::[^=]+)?=\s*(-?[\d.]+(?:e-?\d+)?|True|False)\s*(?:#.*)?$")
_KWDEF = re.compile(r"\b([a-z_][a-z0-9_]*)\s*(?::\s*[\w\[\], .|]+)?=\s*(-?\d+\.\d*(?:e-?\d+)?|-?\d+e-?\d+|-?\d{2,})\b")


def constants(text: str, path: str, limit: int) -> list[str]:
    """Hard-coded numbers of one source file: module-level UPPER constants and numeric keyword defaults of defs."""
    out: list[str] = []
    for i, line in enumerate(text.splitlines(), 1):
        if (m := _CONST.match(line)) is not None:
            out.append(f"{path}:{i} {m.group(1)} = {m.group(2)}")
        elif (s := line.strip()).startswith("def "):
            out += [f"{path}:{i} {s.split('(')[0][4:]}({k}={v})" for k, v in _KWDEF.findall(s)]
        if len(out) >= limit:
            break
    return out


def source_files(ctx: Any) -> list[str]:
    rc = cfg_of(ctx)
    globs = [g for g in (getattr(rc, "source_globs", ()) or ()) if g]
    if globs:
        return globs
    out: list[str] = []
    for fam in (getattr(getattr(ctx, "cfg", None), "families", {}) or {}).values():
        out += [p for p in getattr(fam, "paths", ()) or () if p.endswith(".py") and "*" not in p]
    return list(dict.fromkeys(out))


def inputs(ctx: Any, store: Any) -> dict[str, str]:
    repo, sha, rel = BD.root_ref(ctx)
    lim = int(getattr(cfg_of(ctx), "constants_max", 120) or 120)
    cfg_text = (BD._git_show(repo, sha, rel) or "") if repo and sha and rel else ""
    consts: list[str] = []
    for p in source_files(ctx):
        text = BD._git_show(repo, sha, p) if repo and sha else None
        if text:
            consts += constants(text, p, lim - len(consts))
        if len(consts) >= lim:
            break
    return {"root": f"{sha[:12]} ({rel})", "config": cfg_text[:12000], "constants": "\n".join(consts) or "(none found)",
            "board": BD.block(ctx, store) or "(no board digest: the board ingest is off or has not run)",
            "local": _local_block(ctx, store)}


def _local_block(ctx: Any, store: Any) -> str:
    from loop import localruns as LR
    try:
        return LR.block(ctx, store)
    except Exception:  # noqa: BLE001 - the audit never waits on the local import
        return ""


def prompt(ctx: Any, store: Any) -> str:
    from loop.agents import briefs as BR
    x = inputs(ctx, store)
    n = int(getattr(cfg_of(ctx), "max_arms", 6) or 6)
    return (BR.focus_block(ctx)
            + f"## The root recipe: {x['root']}\nIts config file:\n```yaml\n{x['config']}\n```\n\n"
            f"## Hard-coded constants in the root's training source (file:line name = value)\n{x['constants']}\n\n"
            f"{x['board']}\n\n"
            + (f"{x['local']}\n(A root choice one of these runs varied is NOT untested: cite `local:<name>` and its "
               "single-seed verdict instead of proposing the same arm again.)\n\n" if x.get("local") else "")
            + "## Your task\n"
            "1. Enumerate EVERY baked-in choice of this root: each config key that sets behaviour (skip pure bookkeeping: "
            "names, paths, logging) and each hard-coded constant above that a training run depends on.\n"
            "2. For each, join the board evidence: who varied it (contributor / run titles), with how many seeds, and the "
            "per-axis verdict (REAL beyond 2 SE vs noise). verdict: real_gain / real_harm (the board measured a real "
            "effect of changing it), null (varied with enough seeds, no effect), noise_only (varied only in single runs "
            "inside seed noise), untested (nobody varied it), our_fork (our own change over the board run, never scored "
            "as such). evidence: the run titles / one-key contrasts / reference-model gaps you rely on, or 'none'.\n"
            f"3. For the untested / noise_only / our_fork choices most likely to matter, propose an audit arm (at most {n} "
            "with kind != none): ablate (switch it off / to the neutral value), sweep_up / sweep_down (one step, give the "
            "value), or swap (an alternative). Give config_diff when it is a config key (flattened dotted keys), else "
            "`change` (the code edit in words), the mechanism (why it could matter for the frozen-feature evaluation), "
            "the axis it should move, and a priority in [0, 1] (expected value of the information). Arms change the "
            "TRAINING recipe only. Everything else gets arm.kind none.\n")


def _slug(text: str, n: int = 40) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", (text or "").lower())).strip("_")[:n] or "x"


def accept(ctx: Any, store: Any, res: Mapping[str, Any], key: str, now: float) -> dict[str, Any]:
    """Store the ledger and append its arms (new ids only) to the root-audit idea bank."""
    rc = cfg_of(ctx)
    rows = [a for a in res.get("root_assumptions") or [] if isinstance(a, Mapping)]
    arms = sorted((a for a in rows if (a.get("arm") or {}).get("kind") not in (None, "none")),
                  key=lambda a: -float((a.get("arm") or {}).get("priority") or 0.0))[:int(rc.max_arms)]
    p = Path(bank_path(ctx))
    have = set()
    if p.exists():
        have = {json.loads(l).get("id") for l in p.read_text().splitlines() if l.strip()}
    lines, ids = [], []
    for a in arms:
        arm = a["arm"]
        cd = arm.get("config_diff") if isinstance(arm.get("config_diff"), Mapping) else {}
        aid = f"root_audit-{_slug(str(a['choice']))}-{arm['kind']}"
        ids.append(aid)
        if aid in have:
            continue
        what = (f"set {json.dumps(cd, default=str)}" if cd else str(arm.get("change") or ""))
        lines.append({"id": aid, "title": f"root audit: {arm['kind']} {a['choice']}"[:200], "family": "",
                      "moonshot": False, "jepa": False, "priority": True,
                      "idea": (f"ROOT AUDIT ARM ({arm['kind']}) of the root's baked-in choice `{a['choice']}` "
                               f"(at {a.get('where') or '?'}, root value {a.get('value') or '?'}): {what}. Change ONLY "
                               "this (one arm = one choice), on the incumbent."),
                      "why_it_might_work": (f"{arm.get('mechanism') or ''} Board verdict: {a['verdict']}; evidence: "
                                            f"{a.get('evidence') or 'none'}; tested by: {a.get('tested_by') or 'nobody'}. "
                                            f"Axis: {arm.get('axis') or '?'}.").strip(),
                      "v1_fate": f"root audit {time.strftime('%Y-%m-%d', time.gmtime(now))}: {a['verdict']}",
                      "source": "loop.root_audit", "evidence": str(a.get("evidence") or ""),
                      "axis": str(arm.get("axis") or ""), "knobs": dict(cd)})
    if lines:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a") as fh:
            fh.write("".join(json.dumps(x, default=str) + "\n" for x in lines))
    led = {"key": key, "ts": now, "summary": str(res.get("summary") or "")[:2000], "assumptions": rows[:200], "arms": ids}
    store.set_state(K_LEDGER, led)
    store.log("root_audit", key=key, assumptions=len(rows), arms=len(ids), written=len(lines))
    return led


class RootAudit:
    """The tick's handle: one audit call at a time in ``pool`` (sync without one)."""

    def __init__(self, ctx: Any, store: Any, llm_of: Callable[[], Any], pool: Any = None,
                 clock: Callable[[], float] = time.time) -> None:
        self.ctx, self.store, self.llm_of, self.pool, self.clock = ctx, store, llm_of, pool, clock
        self._fut: Any = None

    def _call(self, key: str) -> dict[str, Any]:
        llm = self.llm_of()
        if llm is None or not hasattr(llm, "complete_json"):
            raise RuntimeError("no LLM client")
        return dict(llm.complete_json("review", system=SYSTEM, prompt=prompt(self.ctx, self.store), schema=SCHEMA))

    def step(self, acts: list[str]) -> None:
        now = self.clock()
        if self._fut is not None and self._fut[0].done():
            (fut, key), self._fut = self._fut, None
            try:
                led = accept(self.ctx, self.store, fut.result(), key, now)
            except Exception as exc:  # noqa: BLE001 - retried after an hour
                self.store.set_state(K_BOOK, {"key": key, "started_ts": now, "error": repr(exc)[:300]})
                self.store.log("root_audit_error", key=key, error=repr(exc)[:500])
                acts.append(f"root audit: failed ({type(exc).__name__}); retry in 1 h")
            else:
                acts.append(f"root audit: {len(led['assumptions'])} baked-in choices, {len(led['arms'])} audit arm(s) "
                            "banked (priority)")
        if self._fut is not None or not due(self.ctx, self.store):
            return
        key = audit_key(self.ctx, self.store)
        bk = self.store.get_state(K_BOOK) or {}
        if bk.get("key") == key and bk.get("error") and now - float(bk.get("started_ts") or 0) < 3600.0:
            return
        self.store.set_state(K_BOOK, {"key": key, "started_ts": now, "error": ""})
        if self.pool is None:
            from concurrent.futures import Future
            f: Future = Future()
            try:
                f.set_result(self._call(key))
            except Exception as exc:  # noqa: BLE001
                f.set_exception(exc)
            self._fut = (f, key)
            self.step(acts)
            return
        self._fut = (self.pool.submit(self._call, key), key)
        acts.append(f"root audit started ({key})")


def block(ctx: Any, store: Any, k: int = 40) -> str:
    """The ledger as a brief section ('' when off / never run)."""
    if not enabled(ctx):
        return ""
    led = store.get_state(K_LEDGER) or {}
    rows = led.get("assumptions") or []
    if not rows:
        return ""
    order = {v: i for i, v in enumerate(("untested", "our_fork", "noise_only", "real_harm", "real_gain", "null"))}
    rows = sorted(rows, key=lambda a: order.get(str(a.get("verdict")), 9))
    out = [f"- {a.get('choice')} = {str(a.get('value'))[:40]} [{a.get('verdict')}]"
           + (f" by {a.get('tested_by')}" if a.get("tested_by") else "")
           + (f": {str(a.get('evidence'))[:160]}" if a.get("evidence") else "") for a in rows[:k]]
    return ("## Root assumption ledger (every baked-in choice of the root and its board evidence; LLM-joined, check "
            f"the digest's numbers; untested / noise-only first)\n{str(led.get('summary') or '')[:600]}\n"
            + "\n".join(out) + (f"\n... {len(rows) - k} more" if len(rows) > k else ""))


def outcomes(store: Any) -> list[dict[str, Any]]:
    """Per banked arm: the requests that served it and the nodes researched from them (``request=<id>`` notes)."""
    from loop.select.bank import K_BANK_USED
    led = store.get_state(K_LEDGER) or {}
    used = store.get_state(K_BANK_USED) or {}
    by_req: dict[str, list[str]] = {}
    for n in store.nodes() if hasattr(store, "nodes") else ():
        if m := re.search(r"\brequest=(\S+)", n.notes or ""):
            by_req.setdefault(m.group(1), []).append(f"{n.id}:{n.status.value}")
    return [{"arm": a, "requests": list(used.get(a) or []),
             "nodes": [x for r in used.get(a) or [] for x in by_req.get(r, [])]} for a in led.get("arms") or []]


def digest_key(ctx: Any, store: Any) -> str:
    return hashlib.sha1(audit_key(ctx, store).encode()).hexdigest()[:10]
