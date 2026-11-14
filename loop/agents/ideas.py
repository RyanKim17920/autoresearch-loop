"""loop.agents.ideas — PLAN BEFORE CODE and the multi-idea slate (operator 2026-10-06).

307 nodes were rejected at intake after a full session (near_dup 72, dead_mechanism 51, multi_family 26,
unmapped_path 15, init_incompatible 75, ...). A research session (moonshot / explore / challenge-the-base) therefore
first calls ``propose_ideas`` with 3-5 DISTINCT ideas, each a short JSON PLAN (mechanism tag, family, files / spans,
config keys, init stance, dose knob, first-principles record, variants, branched plan) -- no code yet:

* :func:`gate` runs every computed intake rule a plan can answer before code exists (surface, protected keys,
  disallowed data, family mapping / multi-family, dead-or-cold / near-dup mechanism with the base-challenge
  exemption, init-compat incl. the zero-init stance, the theme share guard, duplicates within the slate);
* :func:`rank` scores the passing plans with the world model's allocation signal (loop.diagnostics, when present),
  the under-exploration of their theme, their novelty against every node's mechanism text and their predicted
  size; the share guard is a hard gate, not a weight;
* the best is implemented in the session; ranks 2..``ideation.implement_top`` go to the GENERATED idea bank as
  PRIORITY (served next), the rest as normal ideas -- each with its plan and citations, so it is implemented later
  without re-researching (:func:`bank_rows`).

:func:`validate_variants` / :func:`validate_plan` turn a finish's variants and branched plan into the canonical
form loop.variants runs (keys that exist in the node's config, never a protected key, values of the same type)."""

from __future__ import annotations

import dataclasses
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from loop.databank import SPEC_SCHEMA as DATASET_SCHEMA
from loop.contracts import (Candidate, HandoffNote, HypothesisKind, HypothesisRecord, Operator, Strategy)

K_SLATES = "ideation.slates"          # {request id: {ts, chosen, ranks: [{id, tag, family, rule, score, ...}]}}
RESEARCH_BRIEFS = ("literature", "moonshot", "base", "failure", "probe")
INIT_STANCES = ("unchanged", "zero_init", "changes_shapes", "new_init")
OPS = {"<": lambda a, b: a < b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b, ">=": lambda a, b: a >= b}

_txt = {"type": "string"}
_ovr = {"type": "object", "additionalProperties": {}}
VARIANT_SCHEMA = {"type": "object", "required": ["name", "overrides"], "properties": {
    "name": _txt, "overrides": _ovr, "rationale": _txt, "from": _txt}}
PLAN_SCHEMA = {"type": "object", "required": ["when", "then"], "properties": {
    "when": {"type": "object", "required": ["metric", "op", "value"], "properties": {
        "metric": _txt, "op": {"type": "string", "enum": list(OPS)}, "value": {"type": "number"}}},
    "then": {"type": "array", "items": VARIANT_SCHEMA}, "rationale": _txt}}
SPACE_SCHEMA = {"type": "array", "description": "optional: the hyperparameter search space a campaign explores "
                                                "(knobs of the config with ranges); else a default one",
                "items": {"type": "object", "required": ["key", "low", "high"], "properties": {
                    "key": _txt, "low": {"type": "number"}, "high": {"type": "number"},
                    "scale": {"type": "string", "enum": ["log", "linear"]}}}}
CAMPAIGN_SCHEMA = {"type": "boolean", "description": "true: a closed result gets a computed hyperparameter campaign"}
FP_FIELDS = ("mechanistic_model", "bottleneck", "derivation", "prior_attempts")
PREDICTION_SCHEMA = {"type": "object", "required": ["quantity", "direction"], "properties": {
    "quantity": _txt, "direction": {"type": "string", "enum": ["up", "down"]}, "read_from": _txt}}
IDEA_SCHEMA = {"type": "object", "required": ["title", "mechanism_tag", "family", "files", "mechanistic_model",
                                              "bottleneck", "mechanism_prediction", "idea"],
               "properties": {
                   "title": _txt, "mechanism_tag": _txt, "family": _txt, "component": _txt,
                   "files": {"type": "array", "items": _txt,
                             "description": "files (or file::Class.method spans) the change edits"},
                   "config_keys": {"type": "array", "items": _txt,
                                   "description": "config keys it adds or changes (new keys included)"},
                   "init": {"type": "string", "enum": list(INIT_STANCES),
                            "description": "parent parameters/buffers: unchanged; zero_init (new ones only, the "
                                           "model identical to the parent at step 0); changes_shapes; new_init "
                                           "(another model.type / checkpoint / architecture: an init swap)"},
                   "dose_knob": _txt, "idea": _txt, **{f: _txt for f in FP_FIELDS},
                   "mechanism_prediction": PREDICTION_SCHEMA,
                   "predicted_sigmas": {"type": "number", "description": "expected panel gain in panel sigmas"},
                   "citations": {"type": "array", "items": _txt},
                   "variants": {"type": "array", "items": VARIANT_SCHEMA}, "plan": {"type": "array", "items": PLAN_SCHEMA},
                   "campaign": CAMPAIGN_SCHEMA, "search_space": SPACE_SCHEMA, "dataset_spec": DATASET_SCHEMA}}
PROPOSE_TOOL = {"name": "propose_ideas",
                "description": "PLAN BEFORE CODE: submit your DISTINCT ideas as plans (no code yet). The loop runs "
                               "the intake rules on each plan, ranks the passing ones and answers which ONE to "
                               "implement now; the others are banked with their plans (served later, no re-research).",
                "input_schema": {"type": "object", "required": ["ideas"], "properties": {
                    "ideas": {"type": "array", "items": IDEA_SCHEMA}}}}


def is_slate_session(request: Any) -> bool:
    """A research-type session proposes a slate: explore / moonshot DRAFT on a research brief (never a served bank
    idea, a debug fix, a tuning / exposure / leg / clock / compute side request)."""
    return request.operator is Operator.DRAFT and request.strategy in (Strategy.EXPLORE, Strategy.MOONSHOT) \
        and request.brief_kind in RESEARCH_BRIEFS


def is_fp_session(request: Any) -> bool:
    """A session that must state the first-principles record: every ideation session (draft / improve / revisit
    of an idea); never a debug fix, a crossover / stack, or a mechanical side request."""
    return request.operator in (Operator.DRAFT, Operator.IMPROVE) and request.strategy is not Strategy.DESIGN \
        and request.brief_kind not in ("expose", "legs", "clock", "tune", "compute")


def missing_fp(hy: Mapping[str, Any]) -> list[str]:
    miss = [f for f in ("mechanistic_model", "bottleneck", "prior_attempts") if not str(hy.get(f) or "").strip()]
    mp = hy.get("mechanism_prediction") or {}
    if not isinstance(mp, Mapping) or not str(mp.get("quantity") or "").strip():
        miss.append("mechanism_prediction.quantity")
    return miss


# ------------------------------------------------------------------ the plan gate
def plan_path(f: Any) -> str:
    """A plan's file without its annotation: ``model.py (VIT_VARIANTS, robust_norm flag)`` -> ``model.py``; a
    ``file::span`` keeps its span."""
    rel, sep, span = str(f).strip().partition("::")
    return re.split(r"[\s(]", rel.strip(), maxsplit=1)[0] + sep + span


def _probe(ctx: Any, request: Any, base_commit: str, idea: Mapping[str, Any]) -> Candidate:
    spans = {p for r in ctx.cfg.families.values() for p in r.paths if "::" in p}
    files = [f if f in spans else f.split("::")[0] for f in map(plan_path, idea.get("files") or ())]   # a family's span
    keys = {str(k): None for k in idea.get("config_keys") or ()}
    paths = tuple(dict.fromkeys(files))
    h = HypothesisRecord(id="", task_id=ctx.task_id, kind=HypothesisKind.TRIAL,
                         claim=f"{idea.get('title', '')}. {idea.get('idea', '')} {idea.get('prior_attempts', '')}",
                         mechanism_tag=str(idea.get("mechanism_tag") or "").strip(), family=str(idea.get("family") or ""),
                         predictions={}, falsification="")
    return Candidate(request.id, ctx.task_id, "", base_commit, "", "", paths, keys, h,
                     HandoffNote("", ctx.task_id, request.id, "", "", ""))


def gate(ctx: Any, store: Any, snapshot: Any, request: Any, base_commit: str, idea: Mapping[str, Any],
         others: Sequence[Mapping[str, Any]] = ()) -> tuple[str, str]:
    """(rule, fix) of the first intake rule the PLAN already fails; ('', '') = implement it."""
    from loop.agents import dossiers as DS
    from loop.select import intake
    surface, cfg = ctx.spec.surface, ctx.cfg
    tag = str(idea.get("mechanism_tag") or "").strip()
    if not tag:
        return "no_mechanism", "name the mechanism (mechanism_tag)"
    if any(str(o.get("mechanism_tag") or "").strip() == tag for o in others):
        return "slate_duplicate", "two ideas of the slate share a mechanism tag: make them DISTINCT mechanisms"
    files = [plan_path(f) for f in idea.get("files") or ()]
    if not files and not idea.get("config_keys"):
        return "empty_plan", "name the files (or file::span) and config keys the change touches"
    for f in files:
        rel, _, span = f.partition("::")
        if any(intake.path_matches(rel, pat) for pat in surface.locked if "::" not in pat) \
                or any(l.split("::")[0] == rel and span and l.split("::")[1].split("...")[0].strip() in span
                       for l in surface.locked if "::" in l):
            return "surface_locked", f"{f} is locked; work around it"
        if surface.writable and not intake.on_surface(rel, surface.writable):
            return "surface_not_writable", f"{rel} is not editable (editable: {list(surface.writable)})"
    keys = [str(k) for k in idea.get("config_keys") or ()]
    if bad := [k for k in keys for pk in surface.protected_keys if k == pk or k.startswith(pk + ".")]:
        return "protected_key", f"{bad} are protected (budget / eval identity)"
    text = json.dumps(idea, default=str)
    if intake.DISALLOWED_DATA.search(text):
        return "disallowed_data", "GenBio data / models may not be used (usage restrictions)"
    probe = _probe(ctx, request, base_commit, idea)
    fams, problem = intake.derive_families(ctx, probe)
    if problem:
        item = next(i for i, k in intake._family_items(probe) if len(intake.item_families(ctx, probe, i, k)) != 1)
        return problem, f"{item!r} maps to {'no' if problem == 'unmapped_path' else 'several'} family; " \
                        "use keys / paths that belong to exactly one"
    declared = probe.hypothesis.family
    if declared not in cfg.families:
        return "unknown_family", f"family {declared!r} is not one of {sorted(cfg.families)}"
    if (rule := intake.family_rule(ctx, store, snapshot, probe, request.operator, request.strategy, fams))[0]:
        return rule
    if (rule := intake.mech_rule(ctx, store, probe, request.strategy, request.operator))[0]:
        return rule
    if intake.needs_init_compat(ctx, probe, fams) and str(idea.get("init") or "") not in ("unchanged", "zero_init",
                                                                                          "new_init"):
        return "init_incompatible", ("this plan touches init-checked code: keep every parent parameter's name and "
                                     "shape (init 'unchanged'), add only new zero-initialised parameters so the "
                                     "model is identical to the parent at step 0 (init 'zero_init'), or change the "
                                     "init itself (init 'new_init': another model.type / checkpoint, checked as an "
                                     "init swap)")
    if (got := DS.share_guard(ctx, store, snapshot, tag, declared))[0]:
        return "theme_share", got[1] + f" (under-explored families: {', '.join(DS.under_explored(ctx, store, 4))})"
    return "", ""


# ------------------------------------------------------------------ ranking
def novelty(store: Any, idea: Mapping[str, Any]) -> float:
    """1 - the max mechanism-text cosine to any existing node (loop.novelty's TF-IDF over tags + claims)."""
    from loop import novelty as NV
    hyps = {h.id: h for h in store.hypotheses()}
    docs = [NV.mech_terms(n.mechanism_tag, getattr(hyps.get(n.hypothesis_id or ""), "claim", ""))
            for n in store.nodes() if n.id != "root"]
    if not docs:
        return 1.0
    me = NV.mech_terms(str(idea.get("mechanism_tag") or ""), f"{idea.get('title', '')} {idea.get('idea', '')}")
    return max(0.0, 1.0 - max(NV.mech_cos(me, docs), default=0.0))


def score(ctx: Any, store: Any, snapshot: Any, idea: Mapping[str, Any], alloc: Mapping[str, float],
          shares: Mapping[str, float]) -> dict[str, float]:
    """The idea's rank score: (1 + allocation weight of its family / component / themes) x under-exploration
    1 / (1 + 4 x the largest share of its themes among recent draws) x (0.5 + 0.5 novelty) x (1 + 0.25 min(2,
    predicted sigmas)) -- all bounded, none names a mechanism."""
    from loop.agents import dossiers as DS
    themes = DS.text_themes(str(idea.get("mechanism_tag") or ""), str(idea.get("family") or ""))
    a = max([alloc.get(str(idea.get("family") or ""), 0.0), alloc.get(str(idea.get("component") or ""), 0.0),
             *(alloc.get(t.removeprefix("family:"), 0.0) for t in themes)])
    under = 1.0 / (1.0 + 4.0 * max((shares.get(t, 0.0) for t in themes), default=0.0))
    nov = novelty(store, idea)
    pred = float(idea.get("predicted_sigmas") or 0.0)
    pred = 0.0 if not math.isfinite(pred) else max(0.0, min(2.0, pred))
    s = (1.0 + max(0.0, a)) * under * (0.5 + 0.5 * nov) * (1.0 + 0.25 * pred)
    return {"score": round(s, 4), "alloc": round(a, 3), "under": round(under, 3), "novelty": round(nov, 3),
            "pred": pred}


def rank(ctx: Any, store: Any, snapshot: Any, request: Any, base_commit: str,
         ideas: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Every idea with its gate verdict and score; passing ideas first, best score first."""
    from loop.agents import dossiers as DS
    alloc, shares = DS.allocation(store, snapshot), DS.themes(ctx, store)
    rows = []
    for i, idea in enumerate(ideas):
        rule, fix = gate(ctx, store, snapshot, request, base_commit, idea, ideas[:i])
        rows.append({"i": i, "id": f"gen-{request.id}-{i + 1}", "tag": str(idea.get("mechanism_tag") or ""),
                     "family": str(idea.get("family") or ""), "rule": rule, "fix": fix,
                     **score(ctx, store, snapshot, idea, alloc, shares)})
    return sorted(rows, key=lambda r: (bool(r["rule"]), -r["score"], r["i"]))


def _scrub(text: str) -> str:
    from loop.select.bank import SCORE
    return SCORE.sub("[n]", str(text or ""))


def bank_rows(ctx: Any, request: Any, ideas: Sequence[Mapping[str, Any]], ranked: Sequence[Mapping[str, Any]],
              chosen: int) -> list[dict[str, Any]]:
    """The generated-bank lines of every PASSING idea but the chosen one: ranks up to ``implement_top`` priority."""
    top = ctx.cfg.ideation.implement_top
    passing = [r for r in ranked if not r["rule"]]
    out = []
    for k, r in enumerate(passing):
        if r["i"] == chosen:
            continue
        idea = ideas[r["i"]]
        mp = idea.get("mechanism_prediction") or {}
        out.append({
            "id": r["id"], "title": _scrub(idea.get("title")), "family": r["family"],
            "moonshot": request.strategy is Strategy.MOONSHOT, "priority": k < top,
            "idea": _scrub(f"{idea.get('idea', '')} Files: {', '.join(map(str, idea.get('files') or ()))}. "
                           f"Keys: {', '.join(map(str, idea.get('config_keys') or ())) or '-'}. Init: "
                           f"{idea.get('init') or 'unchanged'}. Dose knob: {idea.get('dose_knob') or '-'}."),
            "why_it_might_work": _scrub(f"Model: {idea.get('mechanistic_model', '')} Bottleneck: "
                                        f"{idea.get('bottleneck', '')} Predicts: {mp.get('quantity', '')} "
                                        f"{mp.get('direction', '')}. Earlier attempts: {idea.get('prior_attempts', '')}"),
            "source": "; ".join(map(str, idea.get("citations") or ())) or f"research session {request.id}",
            "plan": {"mechanism_tag": r["tag"], "component": str(idea.get("component") or ""),
                     "files": list(idea.get("files") or ()), "config_keys": list(idea.get("config_keys") or ()),
                     "init": str(idea.get("init") or ""), "dose_knob": str(idea.get("dose_knob") or ""),
                     **{f: str(idea.get(f) or "") for f in FP_FIELDS}, "mechanism_prediction": dict(mp),
                     "variants": list(idea.get("variants") or ()), "plan": list(idea.get("plan") or ()),
                     "campaign": bool(idea.get("campaign")), "search_space": list(idea.get("search_space") or ()),
                     **({"dataset_spec": dict(idea["dataset_spec"])} if isinstance(idea.get("dataset_spec"), Mapping)
                        else {}),
                     "from_request": request.id, "rank": k + 1, "score": r["score"]}})
    return out


def write_bank(ctx: Any, store: Any, rows: Sequence[Mapping[str, Any]]) -> str:
    """Append ``rows`` to ``<state_dir>/idea_bank_generated.jsonl`` and register it (loop.select.bank serves it)."""
    from loop.select.bank import GENERATED, K_GENERATED
    if not rows or not getattr(ctx, "state_dir", ""):
        return ""
    path = Path(ctx.state_dir) / GENERATED
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    if hasattr(store, "set_state") and store.get_state(K_GENERATED) != str(path):
        store.set_state(K_GENERATED, str(path))
    return str(path)


def propose(ctx: Any, store: Any, snapshot: Any, request: Any, base_commit: str,
            ideas: Sequence[Mapping[str, Any]]) -> tuple[int | None, str]:
    """The ``propose_ideas`` tool: (chosen idea index or None, the answer text). Banks the other passing ideas
    and records the slate (``ideation.slates``)."""
    ic = ctx.cfg.ideation
    ideas = [i for i in ideas if isinstance(i, Mapping)][:max(ic.max_ideas, 1)]
    if len(ideas) < ic.min_ideas:
        return None, (f"REJECT slate: propose at least {ic.min_ideas} DISTINCT ideas (different mechanisms or "
                      f"components; at most {ic.max_ideas}); you sent {len(ideas)}")
    ranked = rank(ctx, store, snapshot, request, base_commit, ideas)
    passing = [r for r in ranked if not r["rule"]]
    lines = ["Plan gate + ranking (computed; score = allocation x under-exploration x novelty x predicted size):"]
    lines += [f"  #{r['i'] + 1} {r['tag']} [{r['family']}] " + (f"REJECT {r['rule']}: {r['fix']}" if r["rule"] else
              f"score {r['score']} (alloc {r['alloc']}, under {r['under']}, novelty {r['novelty']})") for r in ranked]
    slates = dict(store.get_state(K_SLATES) or {}) if hasattr(store, "get_state") else {}
    if not passing:
        slates[request.id] = {"ts": time.time(), "chosen": None, "ranks": ranked}
        store.set_state(K_SLATES, slates)
        return None, "\n".join(lines + ["No plan passed: fix the rejected ones (or replace them) and call "
                                         "propose_ideas again."])
    chosen = passing[0]["i"]
    rows = bank_rows(ctx, request, ideas, ranked, chosen)
    path = write_bank(ctx, store, rows)
    slates[request.id] = {"ts": time.time(), "chosen": chosen, "ranks": ranked, "banked": [r["id"] for r in rows]}
    store.set_state(K_SLATES, slates)
    if hasattr(store, "log"):
        store.log("idea_slate", request_id=request.id, chosen=passing[0]["id"], n=len(ideas), passing=len(passing),
                  banked=[r["id"] for r in rows], rejected={r["id"]: r["rule"] for r in ranked if r["rule"]})
    lines.append(f"IMPLEMENT idea #{chosen + 1} ({passing[0]['tag']}) now, exactly as planned (its files, keys and "
                 f"init stance); declare hypothesis.idea_id = \"{passing[0]['id']}\" and carry its first-principles "
                 "record, variants and plan into finish.")
    if rows:
        lines.append(f"Banked for later (no re-research needed): {', '.join(r['id'] for r in rows)}"
                     + (f" ({path})" if path else ""))
    return chosen, "\n".join(lines)


# ------------------------------------------------------------------ variants / branched plan
def _same_type(old: Any, new: Any) -> bool:
    if isinstance(old, bool) or isinstance(new, bool):
        return isinstance(old, bool) and isinstance(new, bool)
    if isinstance(old, (int, float)):
        return isinstance(new, (int, float))
    return type(old) is type(new) or (isinstance(old, (list, tuple)) and isinstance(new, (list, tuple)))


def validate_variants(ctx: Any, config: Mapping[str, Any], variants: Any, limit: int,
                      taken: Sequence[str] = ()) -> tuple[list[dict[str, Any]], list[str]]:
    """(clean variants, problems): each {name, overrides, rationale, from}; an override key must exist in the
    node's config, never be protected, keep its value's type and change it; empty / duplicate variants dropped."""
    prot = tuple(ctx.spec.surface.protected_keys)
    out, problems, names = [], [], set(taken)
    for v in variants if isinstance(variants, (list, tuple)) else ():
        if not isinstance(v, Mapping):
            continue
        name = re.sub(r"[^\w.\-]+", "_", str(v.get("name") or f"v{len(out) + 1}"))[:40]
        ov, bad = {}, []
        for k, val in (v.get("overrides") or {}).items() if isinstance(v.get("overrides"), Mapping) else ():
            if k not in config or any(k == p or k.startswith(p + ".") for p in prot) or not _same_type(config[k], val):
                bad.append(k)
            elif val != config[k]:
                ov[k] = val
        if bad:
            problems.append(f"{name}: dropped keys {bad} (unknown, protected or wrong type)")
        if not ov and not v.get("from"):
            problems.append(f"{name}: no effective override")
            continue
        if name in names:
            name = f"{name}_{len(out) + 1}"
        names.add(name)
        out.append({"name": name, "overrides": ov, "rationale": str(v.get("rationale") or "")[:300],
                    "from": str(v.get("from") or "base")})
        if len(out) >= limit:
            break
    return out, problems


def validate_space(config: Mapping[str, Any], space: Any, limit: int = 6) -> list[dict[str, Any]]:
    """A declared campaign search space: {key, low, high, scale} over NUMERIC keys of the node's config, low < high
    (> 0 on a log scale); anything else dropped."""
    out = []
    for r in space if isinstance(space, (list, tuple)) else ():
        if not isinstance(r, Mapping) or (k := str(r.get("key") or "")) not in config:
            continue
        lo, hi, sc = r.get("low"), r.get("high"), str(r.get("scale") or "log")
        if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in (config[k], lo, hi)) and lo < hi \
                and sc in ("log", "linear") and (sc == "linear" or lo > 0):
            out.append({"key": k, "low": lo, "high": hi, "scale": sc})
    return out[:limit]


def validate_plan(plan: Any, limit: int = 6) -> list[dict[str, Any]]:
    out = []
    for b in plan if isinstance(plan, (list, tuple)) else ():
        w = b.get("when") if isinstance(b, Mapping) else None
        if not isinstance(w, Mapping) or w.get("op") not in OPS or not isinstance(w.get("value"), (int, float)) \
                or not str(w.get("metric") or "").strip() or not isinstance(b.get("then"), (list, tuple)):
            continue
        out.append({"when": {"metric": str(w["metric"]).strip(), "op": w["op"], "value": float(w["value"])},
                    "then": [dict(t) for t in b["then"] if isinstance(t, Mapping)][:4],
                    "rationale": str(b.get("rationale") or "")[:300]})
    return out[:limit]
