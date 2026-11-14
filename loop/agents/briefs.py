"""loop.agents.briefs — prompt payloads and ideation briefs (DESIGN §9): literature, failure,
frontier (literature until a confirmed node exists), probe (kw: hypothesis_id, probe) and moonshot
(kw: ctx for the hard constraints, probe for a review's tagged suggestion)."""

from __future__ import annotations

import importlib
import json
import logging
import math
import re
import threading
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Iterable, Mapping

from loop import databank as DB
from loop import project as P
from loop.contracts import (ACTIVE_STATUSES, K_INFLIGHT, K_STALE_META, PANEL, ROOT_NODE_ID, HypothesisEvidence,
                            HypothesisKind, HypothesisStatus, NodeStatus, RunStatus, WorldSnapshot)

logger = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"
BRIEF_KINDS = ("literature", "failure", "frontier", "probe", "moonshot", "bank", "base")
FOCUS_HEADING = "## Operator focus (priority)"


def focus_block(ctx: Any) -> str:
    """The operator's focus, verbatim under its heading ('' when unset)."""
    focus = str(getattr(getattr(ctx, "cfg", None), "operator_focus", "") or "").strip()
    return f"{FOCUS_HEADING}\n{focus}\n\n" if focus else ""
_FIELD = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


_ARM = threading.local()


TRAINING_ONLY = ("## Scope: TRAINING-SIDE changes only (operator rule)\n"
                 "Change how the encoder is TRAINED (objective, masking, data, schedule, optimizer, architecture). "
                 "Eval-time readout / embedding tricks (pooling, feature concatenation, test-time normalisation or "
                 "augmentation, anything that only changes what the frozen evaluator reads) and eval-only diagnostics "
                 "are OUT of scope: an eval-side observation may only motivate a training change.\n")
EVIDENCE_HEADING = "## Ideas parked for missing evidence (research these first)"
K_EVIDENCE = "ideation.evidence_backlog"     # [{id, claim, kind, ts, served}] (loop.agents.researcher evidence gate)


def research_blocks(ctx: Any, store: Any, *, kind: str = "") -> list[str]:
    """The v3 research context every brief carries when its feature is on ('' parts dropped; all OFF by default):
    the training-only scope (``ideation.training_only``), the competition board digest (loop.board), the root
    assumption ledger (loop.root_audit), and -- on a literature brief -- the ideas parked by the evidence gate."""
    from loop import board as BD, localruns as LR, root_audit as RA
    ic = getattr(getattr(ctx, "cfg", None), "ideation", None)
    out = [TRAINING_ONLY if getattr(ic, "training_only", False) else ""]
    try:
        out.append(BD.block(ctx, store))
    except Exception as exc:  # noqa: BLE001 - a broken digest never blocks a brief
        out.append(f"## Competition board digest\n(unavailable: {type(exc).__name__})")
    try:
        out.append(LR.block(ctx, store))
    except Exception as exc:  # noqa: BLE001
        out.append(f"{LR.HEADING}\n(unavailable: {type(exc).__name__})")
    try:
        out.append(RA.block(ctx, store))
    except Exception as exc:  # noqa: BLE001
        out.append(f"## Root assumption ledger\n(unavailable: {type(exc).__name__})")
    # 11-08: the project knowledge file + the bar (loop.knowledge) and the mechanism memory (loop.core.tested): never
    # ablated -- the memory is what the slate gate enforces, a brief without it would be bounced blind
    from loop import knowledge as KN
    from loop.core import tested as TS
    try:
        out.append(KN.block(ctx, store))
    except Exception as exc:  # noqa: BLE001
        out.append(f"{KN.HEADING}\n(unavailable: {type(exc).__name__})")
    try:
        out.append(TS.brief_block(ctx, store))
    except Exception as exc:  # noqa: BLE001
        out.append(f"{TS.HEADING}\n(unavailable: {type(exc).__name__})")
    if getattr(ic, "require_evidence", False) and kind in tuple(getattr(ic, "evidence_kinds", ()) or ()):
        out.append("## Evidence required (loop rule)\nA structural proposal must carry hypothesis.evidence -- a paper "
                   "you read (URL / arXiv id / doi), a board run id from the digest, or a reference-model fact -- and "
                   "hypothesis.readout_path (how it changes what the frozen-feature evaluation reads). Research before "
                   "you build: a finish without them is bounced, then parked for a literature session.")
    if kind == "literature" and getattr(ic, "require_evidence", False) and hasattr(store, "get_state"):
        parked = [e for e in store.get_state(K_EVIDENCE) or [] if not e.get("served")][:5]
        if parked:
            out.append(EVIDENCE_HEADING + "\n" + "\n".join(
                f"- {e.get('id')}: {str(e.get('claim') or '')[:400]}" for e in parked)
                + "\nThese structural proposals were built without evidence. Search for prior work (papers, the board, "
                  "the reference models) that supports or refutes each; build the best-supported one with its evidence, "
                  "or say in handoff.learnings why none holds up.")
    return [x.strip() for x in out if x and x.strip()]


def strip_section(text: str, heading_prefix: str) -> str:
    """``text`` without the markdown section whose heading starts with ``heading_prefix`` (up to the next '## ')."""
    i = text.find(heading_prefix)
    if i < 0:
        return text
    j = text.find("\n## ", i + len(heading_prefix))
    return text[:i] + (text[j + 1:] if j >= 0 else "")


def in_prompt_dir(prompt_dir: Path | None, fn: Any, *args: Any) -> Any:
    """``fn(*args)`` with :func:`render` reading ``prompt_dir`` first (a prompt version / A/B arm of loop.core.meta;
    None: the repo's prompts). Thread-local: one research session per pool thread."""
    old, _ARM.dir = getattr(_ARM, "dir", None), prompt_dir
    try:
        return fn(*args)
    finally:
        _ARM.dir = old


def render(name: str, **values: Any) -> str:
    """Substitute ``{field}`` for known keys only (JSON braces and unknown fields survive)."""
    arm = getattr(_ARM, "dir", None)
    path = Path(arm) / f"{name}.md" if arm is not None else None
    text = (path if path is not None and path.exists() else P.prompt_path(name)).read_text()
    return _FIELD.sub(lambda m: str(values[m.group(1)]) if m.group(1) in values else m.group(0), text)


def read_text(path: str, n: int) -> str:
    """First ``n`` chars of a file (n > 0) or its last ``-n``; '' if unreadable."""
    try:
        text = Path(path).read_text(errors="replace") if path else ""
    except OSError:
        return ""
    return text[:n] if n > 0 else text[n:]


def lever_table(snapshot: WorldSnapshot, *, target: str = PANEL) -> str:
    rows = sorted((e for e in snapshot.levers if e.target == target),
                  key=lambda e: -abs(e.mean) / max(e.se, 1e-12))
    if not rows:
        return "(no lever effects fitted yet)"
    return "\n".join([f"lever | effect on {target} per +1 step | se | P(>0) | n trials tagged"] + [
        f"{e.lever} | {e.mean:+.5f} | {e.se:.5f} | {e.p_positive:.2f} | {e.n_nonzero}" for e in rows])


def hypothesis_table(store: Any, snapshot: WorldSnapshot, *, kind: HypothesisKind | None = None,
                     status: Iterable[HypothesisStatus] | None = None, limit: int = 60) -> str:
    out = ["id | kind | status | E_for | E_against | n | family/tag | claim | levers"]
    stale = store.get_state(K_STALE_META) or {}
    for h in store.hypotheses(kind=kind)[-limit:]:
        ev = snapshot.hypotheses.get(h.id) or HypothesisEvidence(h.id, 1.0, 1.0, 0, HypothesisStatus.UNTESTED)
        st = ev.status.value
        if status is not None and (ev.status not in status or h.id in stale):    # a stale meta steers nothing
            continue
        if h.id in stale:
            st = f"STALE (its evidence {','.join(stale[h.id].get('nodes', []))} fell; re-derive, do not carry it)"
        lev = ",".join(f"{k}{'+' if v > 0 else '-'}" for k, v in h.levers.directions.items() if v)
        out.append(f"{h.id} | {h.kind.value} | {st} | {ev.e_value:.2f} | {ev.e_against:.2f} | "
                   f"{ev.n_trials} | {h.family}/{h.mechanism_tag} | {h.claim[:160]} | {lev or '-'}")
    return "\n".join(out) if len(out) > 1 else "(none)"


def open_meta(store: Any, snapshot: WorldSnapshot) -> str:
    return hypothesis_table(store, snapshot, kind=HypothesisKind.META,
                            status=[HypothesisStatus.UNTESTED, HypothesisStatus.TESTING])


FINDINGS_HEADING = ("## Operator's manual experiments (MEASURED by the operator on paired seeds: facts; they outrank "
                    "every postmortem, handoff, review and mechanism rationale below, all written by earlier LLM "
                    "sessions)")
PROSE_NOTE = ("(Written by earlier LLM sessions: HYPOTHESES, not facts. Trust the measured numbers (the tables, the "
              "operator's manual experiments) over this prose; a confident causal story here is a guess until a "
              "paired increment backs it.)")


def findings_block(ctx: Any) -> str:
    """``ideation.operator_findings`` under its heading ('' when unset): the operator's manual results as facts."""
    items = [str(x).strip() for x in getattr(getattr(getattr(ctx, "cfg", None), "ideation", None),
                                              "operator_findings", ()) or () if str(x).strip()]
    if not items:
        return ""
    return (f"{FINDINGS_HEADING}\n" + "\n".join(f"- {x}" for x in items)
            + "\nDo not re-propose a DEAD / NULL line (or its near variant); each OPEN line is a place the evidence "
              "leaves untested. These supersede the diagnostic flags of section (j) they test.\n")


COMPUTED_NOTE = ("(COMPUTED facts: typed from the ledger's own numbers by loop.core.closed_pm, no LLM. Trust them as "
                 "measurements; the outcome type is a rule, not a story.)")


def postmortem_text(pms: Iterable[Any], limit: int = 20, explain_chars: int = 300) -> str:
    """Postmortems split by origin: the COMPUTED rows (loop.core.closed_pm / struct_tune) as measured facts, the
    older LLM-written ones under the 'hypotheses, not facts' disclaimer. The ``next:`` suggestion of an LLM row is left
    out (it steered later sessions toward one idea family; ``history`` still opens it); a computed row's typed hint stays."""
    from loop.core.closed_pm import is_computed
    rows = list(pms)[-limit:]
    comp = [f"- [{p.node_id} {p.verdict}] {p.explanation[:explain_chars + 200]}"
            + (f" -> {p.suggested_next[:160]}" if p.suggested_next else "") for p in rows if is_computed(p)]
    prose = [f"- [{p.node_id} {p.verdict}] {p.explanation[:explain_chars]}" for p in rows if not is_computed(p)]
    out = ([COMPUTED_NOTE, *comp] if comp else []) + ([PROSE_NOTE, *prose] if prose else [])
    return "\n".join(out)


def postmortem_block(pms: Iterable[Any], limit: int = 20) -> str:
    """The last ``limit`` postmortems (:func:`postmortem_text`); '(none)' without any."""
    return postmortem_text(pms, limit) or "(none)"


def node_line(snapshot: WorldSnapshot, n: Any) -> str:
    post = snapshot.nodes.get(n.id)
    u = f"U={post.utility.mean:+.4f}±{post.utility.sd:.4f}" if post else "U=?"
    return f"{n.id} [{n.family}/{n.mechanism_tag}] {n.status.value} {u} commit={n.commit[:10]}"


def brief(kind: str, store: Any, snapshot: WorldSnapshot, **kw: Any) -> str:
    """Render one ideation brief; ValueError for an unknown kind. Every kind but frontier (the
    exploit brief) leads with the operator focus."""
    if kind not in BRIEF_KINDS:
        raise ValueError(f"unknown brief kind {kind!r}; expected one of {BRIEF_KINDS}")
    text = _brief(kind, store, snapshot, **kw)
    return text if kind == "frontier" else focus_block(kw.get("ctx")) + text


LEGACY_INIT = ("- Keeping the current init: its pretrained weights must still load: new modules are heads / auxiliary "
               "branches, or residual branches initialised to zero so the parent's function is unchanged at step 0. "
               "The init-compat check runs on every model.py / train.py change.\n- Changing the init or architecture "
               "is an init swap: any non-pathology, non-GenBio source; it must reach the full sample budget within "
               "the FLOP cap.")


def _brief(kind: str, store: Any, snapshot: WorldSnapshot, **kw: Any) -> str:
    from loop.select.bank import text as bank_text
    common = dict(family=kw.get("family") or "(any)")   # ledger/levers/meta: the context pack
    confirmed = sorted((n for n in store.nodes(status=[NodeStatus.CONFIRMED]) if n.id != ROOT_NODE_ID),
                       key=lambda n: -snapshot.nodes[n.id].utility.mean if n.id in snapshot.nodes else 0)
    h = store.get_hypothesis(kw.get("hypothesis_id") or "")
    if kind == "base" and kw.get("ctx") is not None:
        from loop.agents.dossiers import base_brief
        return base_brief(kw["ctx"], store, snapshot, kw.get("component") or "")
    if kind == "bank" and not kw.get("moonshot"):
        return f"{bank_text(kw['bank'])}\n\n" + render("brief_literature", **common)
    if kind in ("moonshot", "bank"):
        # v3-c8b: the old moonshot brief is retired into the bold brief (loop.core.bold): the structural taxonomy, no
        # lever table / review anchoring. This legacy path has no propose_slate tool and keeps the zero-init rule.
        from loop.core import bold as BOLD
        return render("brief_bold", taxonomy=BOLD.taxonomy_text(), slate_rule="", init_rule=legacy_init(),
                      suggestion=f"\n{bank_text(kw['bank'])}\n" if kind == "bank"
                      else f"\nThe big review suggested: {kw['probe']}\n" if kw.get("probe") else "",
                      constraints=hard_constraints(kw["ctx"]) if kw.get("ctx") is not None
                      else "(see the context pack: locked files, protected keys)")
    if kind == "probe" and h is not None:
        pred = h.predictions.get(PANEL) or next(iter(h.predictions.values()))
        return render("brief_probe", hypothesis_id=h.id, claim=h.claim, probe=kw.get("probe", "(none)"),
                      sign="+" if pred.sign > 0 else "-", falsification=h.falsification,
                      directions=", ".join(
                          f"{k} {'UP' if v > 0 else 'DOWN'}" for k, v in h.levers.directions.items() if v))
    if kind == "probe":
        return f"PROBE from a big review: {kw.get('probe', '(none)')}\n\n" + render("brief_literature", **common)
    if kind == "failure":
        bad = store.nodes(status=[NodeStatus.SCREENED_OUT, NodeStatus.NOT_CONFIRMED, NodeStatus.CRASHED,
                                  NodeStatus.VOID, NodeStatus.VETOED])
        return render("brief_failure", **common,
                      failures="\n".join(node_line(snapshot, n) for n in bad[-40:]) or "(none)",
                      postmortems=postmortem_block([] if "postmortem" in kw.get("ablated", ()) else store.postmortems()),
                      refuted=hypothesis_table(store, snapshot, status=[HypothesisStatus.REFUTED]))
    if kind == "frontier" and confirmed:
        notes = [h for n in confirmed[:5] for h in store.handoffs(node_id=n.id)]
        return render("brief_frontier", **common, incumbent=snapshot.incumbent_id,
                      frontier="\n".join(node_line(snapshot, n) for n in confirmed),
                      handoffs="\n".join(f"- [{h.node_id}] learned: {h.learnings[:240]} | concerns: "
                                         f"{h.concerns[:200]}" for h in notes) or "(none)")
    return render("brief_literature", **common)


# ---------------------------------------------------------------------------
# The researcher's context pack (first user message): compact, token-budgeted
# ---------------------------------------------------------------------------

PACK_TOKENS = 2800
CHARS_PER_TOKEN = 3.5          # conservative for tables of ids and numbers

LEVER_HELP = {
    "training_strength": "how hard the model is trained (LR x steps, fewer = weaker)",
    "regularization": "weight decay, drop-path, clipping, anything that restrains fitting",
    "effective_lr": "the size of the optimizer step (lr, warmup, schedule, layer decay)",
    "augmentation_strength": "how strongly views are distorted (crops, colour, HED)",
    "capacity": "model/head size and depth",
    "objective_difficulty": "how hard the SSL task is (masking, loss weights, targets)",
    "data_curation": "which tiles are trained on (dataset, filtering, normalisation stats)",
    "readout": "how features are pooled/normalised/exposed for probing (the eval_readout family)",
}
FAMILY_HELP = {
    "optimization": "optimizer, LR schedule, batch size",
    "regularization": "weight decay, drop-path",
    "augmentation": "view generation: crops, colour/HED jitter, view counts/sizes",
    "architecture": "model and head shapes, the init model itself (needs init-compat, or an init swap's check)",
    "objective": "SSL loss terms and their weights (KDE, JEPA, FINO)",
    "training": "the training loop and its mechanics (train.py)",
    "data_curation": "which data is trained on and its normalisation",
    "eval_readout": ("eval-time feature code (forward/encode_image/probe_features/_suppress) and the "
                     "post-training fit on TRAINING tiles; may pair with ONE training-side family for a "
                     "readout trained during pretraining; never labels or probe data"),
}
NO_SWEEPS = ("NO HYPERPARAMETER-SWEEP SPAM (operator): propose a MECHANISM (new code: loss / objective / data / "
             "augmentation / architecture / training procedure). A numbers-only change must be substantiated: a "
             "MAJOR move (> 2x, to/from 0, or a categorical / schedule switch, e.g. an LR-schedule truncation fix) "
             "with its mechanism stated, or a small move the lever regression or a designed batch backs. An "
             "unbacked small tweak is rejected (sweep_only), a backed one whose expected effect is below the screen's "
             "detectable effect too (below_noise: minute changes fall to noise), and a near-copy of an existing node's change set (same "
             "code, every value within 2x, 2x included: 'the same stack + one small tweak') is rejected (near_dup_config); small "
             "sweeps belong to designed batches.")
CARRIED_LESSON = ("Architecture changes that break model init (parent weights no longer load, "
                  "changed shapes/names) score MUCH worse: do not make one unless it stays "
                  "init-compatible; the harness checks it and rejects failures.")
INIT_SWAP = ("Changing the init itself is open (operator 2026-10-07): another pretrained model or size, another "
             "architecture, new model code or a random init. A node with another model.type / checkpoint "
             "(model.pretrained_url) than its parent is an INIT SWAP, checked before any GPU instead of the "
             "parent-relative init check: it constructs as model.py's ViT(variant=model.type) (probe.py builds and "
             "strict-loads it the same way), its load_pretrained runs, its forward is finite (else "
             "init_incompatible); no pathology-trained or GenBio checkpoint may initialise or guide it (init_illegal); "
             "the smoke's measured FLOPs per sample x train.max_train_samples must stay within the FLOP cap "
             "(over_flop_cap, with the projected fraction). The recipe's hyperparameters were set for the current "
             "init: variants / a branched plan in finish can retune them beside it.")
GENBIO_BAN = ("- NO change may use GenBio data or models (GenBio-PathFM embeddings, the GenBio-curated subset "
              "/data/nanopath-curated-genbiopathfm): usage restrictions; any reference is rejected disallowed_data.")
FIT_DATA = "charged calibration tiles (jpegs, site_rows)"
DATABANK_HEADER = ("DATA BANK (data_curation: set data.dataset_dir to one; TCGA tile datasets in the training Arrow "
                   "format): ")
#: call-time domain text (loop.project rules; no overlay: the literal defaults above, byte-identical)
_LEVER_RULES = {k: f"lever_{k}" for k in LEVER_HELP}
_FAMILY_RULES = {k: f"family_{k}" for k in FAMILY_HELP}


def legacy_init() -> str:
    return P.rule("init_legacy", LEGACY_INIT)


def init_swap() -> str:
    return P.rule("init_swap", INIT_SWAP)


def carried_lesson() -> str:
    return P.rule("carried_lesson", CARRIED_LESSON)


def no_sweeps() -> str:
    return P.rule("no_sweeps", NO_SWEEPS)


def lever_help() -> dict[str, str]:
    return {k: P.rule(_LEVER_RULES[k], v) if k in _LEVER_RULES else v for k, v in LEVER_HELP.items()}


def family_help() -> dict[str, str]:
    return {k: P.rule(_FAMILY_RULES[k], v) if k in _FAMILY_RULES else v for k, v in FAMILY_HELP.items()}


def references(ctx: Any) -> tuple[str, ...]:
    """``init_compat.references`` resolved against the task spec's directory: read-only reference code."""
    base = Path(getattr(ctx, "spec_path", "") or ".").parent
    return tuple(str((base / r).resolve()) for r in getattr(ctx.cfg.init_compat, "references", ()) or ())


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


def optional_block(name: str, store: Any, snapshot: Any) -> str:
    """``loop.quality.<name>(store, snapshot)`` when that module exists; '' otherwise or on error."""
    try:
        quality = importlib.import_module("loop.quality")           # optional: another module owns it
    except ImportError:
        return ""
    try:
        return str(getattr(quality, name)(store, snapshot) or "").strip()
    except Exception as e:  # noqa: BLE001 — an analytics bug must never block a prompt
        logger.warning("quality.%s failed: %s: %s", name, type(e).__name__, str(e)[:200])
        return ""


def _val(v: Any, n: int = 40) -> str:
    t = v if isinstance(v, str) else json.dumps(v, default=str)
    return t if len(t) <= n else t[:n - 3] + "..."


def _entry(p: str) -> str:
    """A path or ``file::`` entry, a source span shown by its first marker."""
    f, _, t = p.partition("::")
    return p if not t else f"{f}::{t}" if "..." not in t else f"{f} span from `{t.split('...')[0].strip()[:40]}`"


def stackable_line(ctx: Any, store: Any, snapshot: Any) -> str:
    """The already-positive knobs (key=value, shrunk P(U>0)) a 2-family compound may stack."""
    from loop.select.intake import stackable
    st = stackable(ctx, store, snapshot)
    return (f"Stackable positive knobs (P(U>0) >= {ctx.cfg.intake.compound_min_p}; a normal candidate may span 2 "
            "families if ALL changed keys of one family are exactly these; declare the OTHER family): "
            + ("; ".join(f"{k}={v[0]} (P={v[1]:.2f}, {v[2]})" for k, v in sorted(st.items())) or "none yet"))


def directions(ctx: Any, store: Any, snapshot: Any) -> str:
    """The winning-direction ledger as computed facts (loop.extrapolate.facts; '' while the lane is off)."""
    if not getattr(getattr(ctx.cfg, "extrapolate", None), "enabled", False):
        return ""
    from loop.extrapolate import facts
    return facts(ctx, store, snapshot)


def legs_line(store: Any, snapshot: Any) -> str:
    """The incumbent's legs (id[subsystem]) a change may declare in handoff.stacks_with / conflicts_with ('' when none)."""
    inc = getattr(snapshot, "incumbent_id", ROOT_NODE_ID)
    if inc == ROOT_NODE_ID:
        return ""
    from loop.legs import own_legs, subsystem
    legs = own_legs(store, inc)
    return f"Incumbent {inc} legs: " + (", ".join(f"{i}[{subsystem(n) if (n := store.get_node(i)) else 'other'}]"
                                                  for i in legs) or "none") if legs else ""


def guard_line(ctx: Any) -> list[str]:
    """What a ``forbid``-guarded family may not read (it is checked mechanically at intake)."""
    fams = [f for f, r in ctx.cfg.families.items() if r.forbid]
    return [P.rule("forbid_data", GENBIO_BAN)] + \
        ([f"- NO change (any family) may read labels, probe datasets/paths/config/results, benchmarking "
             "or labless data, introspect frames/modules, or point a config path at probe data (a static "
             "check rejects the diff)."] if ctx.cfg.forbid else []) + \
        ([f"- {', '.join(fams)} changes also may not read the environment/filesystem or any data beyond the "
          f"{P.rule('forbid_fit_data', FIT_DATA)}; fit on those only."] if fams else [])


def family_lines(ctx: Any, config: Mapping[str, Any] | None) -> list[str]:
    """One line per family: meaning, its config keys with CURRENT values, its paths."""
    out, fh = [], family_help()
    for fam, rule in sorted(ctx.cfg.families.items()):
        keys = [k for k in (config or {}) if any(fnmatch(k, g) for g in rule.keys if g != "*")]
        vals = ", ".join(f"{k}={_val(config[k])}" for k in keys[:16]) if config else ", ".join(rule.keys)
        paths = [_entry(p) for p in rule.paths if p != "*"] + (["any other file"] if "*" in rule.paths else [])
        out.append(f"- {fam}{' [init-check]' if rule.init_check else ''}{' [combines]' if rule.combine else ''}"
                   f"{' — ' + fh[fam] if fam in fh else ''}: {vals or '-'}"
                   f"{'; files ' + ', '.join(paths) if paths else ''}")
    return out


def compact_ledger(store: Any, snapshot: Any, budget_chars: int) -> str:
    """Newest first: id | family/tag | levers | status | shrunk U (mean±sd vs root)."""
    nodes = [n for n in store.nodes() if n.id != ROOT_NODE_ID]
    counts: dict[str, int] = {}
    for n in nodes:
        counts[n.status.value] = counts.get(n.status.value, 0) + 1
    head = [f"{len(nodes)} trials: " + (", ".join(f"{v} {k}" for k, v in sorted(counts.items())) or "none yet"),
            "id | family/tag | levers | status | U"]
    rows, used = [], sum(len(h) + 1 for h in head)
    for n in sorted(nodes, key=lambda n: (n.created_ts, n.id), reverse=True):
        post = snapshot.nodes.get(n.id) if snapshot is not None else None
        u = f"{post.utility.mean:+.4f}±{post.utility.sd:.4f}" if post else "."
        lev = ",".join(f"{k}{'+' if v > 0 else '-'}" for k, v in sorted(n.levers.directions.items()) if v) or "-"
        st = n.status.value + (f"({n.reject_reason})" if n.reject_reason else "")
        row = f"{n.id} | {n.family}/{n.mechanism_tag[:40]} | {lev} | {st} | {u}"
        if used + len(row) + 1 > budget_chars - 40:          # room for the "omitted" line
            rows.append(f"... {len(nodes) - len(rows)} older trials omitted")
            break
        rows.append(row)
        used += len(row) + 1
    return "\n".join(head + rows)


def in_progress(store: Any, request_id: str = "") -> str:
    """Other researchers' assignments, then queued / active nodes with their config values."""
    rows = [f"- a researcher is drafting ({r['strategy']}) in {'family ' + r['family'] if r['family'] else 'any family'}"
            for r in store.get_state(K_INFLIGHT) or [] if r["id"] != request_id]
    for n in store.nodes(status=[NodeStatus.QUEUED, *ACTIVE_STATUSES]):
        cd = store.get_state(f"config_diff:{n.id}") or {}
        rows.append(f"- {n.id} {n.status.value}: {n.family}/{n.mechanism_tag}"
                    + (f" ({', '.join(f'{k}={_val(v)}' for k, v in cd.items())})" if cd else ""))
    return "\n".join(rows) or "(nothing)"


def example_finish(ctx: Any) -> str:
    """ONE worked example of a good finish record, valid for this task's families/levers/axes."""
    fams = sorted(ctx.cfg.families, key=lambda f: (f != "optimization", not any(
        "lr" in k for k in ctx.cfg.families[f].keys), f))
    fam = fams[0] if fams else "?"
    axis = next(iter(ctx.cfg.axes), PANEL)
    sd = float(getattr(ctx.cfg, "panel_sd", 0.004) or 0.004)
    sd = sd if math.isfinite(sd) else 0.004
    rec = {"hypothesis": {
        "claim": "At this fixed sample budget the peak LR over-trains the backbone; a 20% lower peak LR "
                 "keeps late features less collapsed and lifts the panel.",
        "mechanism_tag": "lower_peak_lr", "family": fam,
        "predictions": {PANEL: {"sign": 1, "size": round(0.5 * sd, 4)},
                        axis: {"sign": 1, "size": round(0.7 * sd, 4)}},
        "falsification": f"panel paired delta <= 0 on both screen seeds, or {axis} falls by more than one sigma"},
        "levers": {"effective_lr": -1, "training_strength": -1},
        "handoff": {"actions": "lowered the peak LR by 20% (config only)", "learnings": "warmup is a fraction of total steps",
                    "concerns": "a smaller LR may just need more steps, which the budget forbids"}}
    return json.dumps(rec, indent=None)


def context_pack(ctx: Any, store: Any, snapshot: Any, *, config: Mapping[str, Any] | None = None,
                 budget_tokens: int = PACK_TOKENS, request_id: str = "") -> str:
    """The researcher's compact world view. Fixed sections first; lessons, meta and the
    ledger are trimmed so the whole pack stays within ``budget_tokens``."""
    surface, vocab = ctx.spec.surface, ctx.cfg.levers.vocabulary
    lh = lever_help()
    files = [l for l in surface.locked if "::" not in l]
    funcs = [_entry(l) for l in surface.locked if "::" in l]
    fixed = [
        "## Context pack",
        f"Editable files: {', '.join(surface.writable)}",
        f"LOCKED files: {', '.join(files) or '-'}",
        *([f"LOCKED code (must stay byte-identical): {'; '.join(funcs)}"] if funcs else []),
        f"Protected config keys (never change): {', '.join(surface.protected_keys) or '-'}",
        "",
        "### Families (declare exactly one; a diff touching a [combines] family may also touch ONE other "
        "family and declares the [combines] one; config keys shown with the parent's CURRENT values)",
        *family_lines(ctx, config),
        "Config keys not listed under a family map to none: changing them is rejected.",
        *([P.rule("databank_header", DATABANK_HEADER) + "; ".join(bank)] if (bank := DB.bank_lines(ctx, store)) else []),
        *(["MEASURED DATASETS (own paired gain over the parent / seeds; DEAD = rejected unless [differs from: ...] "
           "answers its nodes):", *got] if hasattr(store, "get_state") and (got := _measured(ctx, store)) else []),
        *guard_line(ctx),
        f"- {no_sweeps()}",
        stackable_line(ctx, store, snapshot),
        "",
        "### Levers (tag each lever your change moves, -1 or +1)",
        "; ".join(f"{l}: {lh.get(l, '')}".rstrip(": ") for l in vocab),
        "",
        f"### Carried lesson\n{carried_lesson()} (keeping the current init)\n{init_swap()}"
        + (f" Read-only reference code (bash cat / sed; never merged): {', '.join(refs)}." if (refs := references(ctx))
           else ""),
        "",
        "### Example of a good finish record",
        example_finish(ctx),
    ]
    budget = int(budget_tokens * CHARS_PER_TOKEN)
    left = budget - len("\n".join(fixed))
    from loop.design import lines as design_lines
    lessons = "\n".join(filter(None, (design_lines(store), optional_block("ideation_lessons", store, snapshot)))
                         )[:max(0, min(2400, left // 3))]
    legs = legs_line(store, snapshot)
    meta = open_meta(store, snapshot) if snapshot is not None else "(none)"
    meta = meta[:max(0, min(1200, left // 5))]
    busy = in_progress(store, request_id)[:max(0, min(1500, left // 4))]
    from loop.novelty import dead_block
    dead = dead_block(ctx, store, snapshot)[:max(0, min(1500, left // 5))] \
        if getattr(getattr(ctx.cfg, "novelty", None), "enabled", False) else ""
    tail = ["", "### Already in progress -- do NOT propose these or close variants (another value of the "
            "same knob waits for their verdict)", busy,
            *(["", "### Dead mechanisms (tried and failed, most retried first; a new idea near one is rejected "
               "dead_mechanism unless it names those nodes and says what is MATERIALLY different)", dead] if dead else []),
            *(["", "### INCUMBENT LEGS (state handoff.stacks_with / conflicts_with for them)", legs] if legs else []),
            "", "### What we've learned so far", lessons or "(nothing yet)",
            *(["", "### Winning directions per config key (computed: incumbent lineage, factorials, single moves; "
               "loop.extrapolate continues each on the incumbent itself -- do not propose those values)", dirs]
              if (dirs := directions(ctx, store, snapshot)[:max(0, min(1500, left // 5))]) else []),
            "", "### Open meta-hypotheses from reviews (a probe that moves their levers is valuable)", meta]
    tail += ["", "### Ledger (newest first; U = shrunk panel utility vs root)"]
    left -= len("\n".join(tail)) + 1
    tail.append(compact_ledger(store, snapshot, max(200, left)))
    return "\n".join(fixed + tail)


# ---------------------------------------------------------------------------
# Big-review payloads: the CORRECTED per-node table and the hard constraints
# ---------------------------------------------------------------------------

RAW_LABEL = "raw, uncorrected, single-seed — do not cite as effects"


def review_node_table(store: Any, snapshot: WorldSnapshot, axes: Iterable[str], *, max_rows: int = 500) -> str:
    """One line per node, the world model's CORRECTED shrunk estimate first (U mean±sd vs root,
    P(U>0), status, shrunk per-axis means); the raw same-seed paired panel delta rides last and
    is labelled as such. Scored nodes are sorted by U; unscored nodes follow."""
    axes = tuple(axes)
    root: dict[int, Any] = {r.seed: r for r in store.runs(node_id=ROOT_NODE_ID) if r.status is RunStatus.DONE}
    nodes = [n for n in store.nodes() if n.id != ROOT_NODE_ID]
    nodes.sort(key=lambda n: (n.id not in snapshot.nodes,
                              -(snapshot.nodes[n.id].utility.mean if n.id in snapshot.nodes else 0.0)))
    inc = snapshot.incumbent_id
    out = [f"# U = corrected shrunk panel utility vs root (shared-control corrected, hierarchical shrinkage).",
           f"# P(U>inc) = P(the node beats the confirmed incumbent {inc}): the decision quantity (a child of {inc}: "
           f"its own increment).",
           f"# The last column is {RAW_LABEL}.",
           "id | parent | family/tag | levers | status | n done | U mean±sd | P(U>0) | P(U>inc) | vetoed | "
           f"shrunk axes ({' '.join(axes)}) | raw panel ({RAW_LABEL})"]
    for n in nodes[:max_rows]:
        post = snapshot.nodes.get(n.id)
        done = [r for r in store.runs(node_id=n.id) if r.status is RunStatus.DONE]
        raw = [r.aggregate - root[r.seed].aggregate for r in done
               if r.seed in root and None not in (r.aggregate, root[r.seed].aggregate)]
        lev = ",".join(f"{k}{'+' if v > 0 else '-'}" for k, v in sorted(n.levers.directions.items()) if v) or "-"
        st = n.status.value + (f"({n.reject_reason[:40]})" if n.reject_reason else "")
        if post is None:
            est = ["not scored", ".", ".", ".", "."]
        else:
            est = [f"{post.utility.mean:+.4f}±{post.utility.sd:.4f}", f"{post.p_positive:.2f}",
                   "inc" if n.id == inc else f"{post.p_beats_incumbent:.2f}",
                   ",".join(post.veto_axes) if post.vetoed else "-",
                   " ".join(f"{post.axes[a].mean:+.4f}" if a in post.axes else "." for a in axes)]
        out.append(" | ".join([n.id, n.parent_id or "-", f"{n.family}/{n.mechanism_tag[:40]}", lev, st,
                               str(len(done)), *est,
                               f"{sum(raw) / len(raw):+.4f} (n={len(raw)})" if raw else "."]))
    if len(nodes) > max_rows:
        out.append(f"# ... {len(nodes) - max_rows} more nodes omitted")
    return "\n".join(out) if nodes else "(no trials yet)"


def hard_constraints(ctx: Any) -> str:
    """What no probe can change: locked files/code, protected keys (incl. the training budget),
    the seed cap and the init-compatibility lesson."""
    surface, seeds = ctx.spec.surface, ctx.cfg.seeds
    files = [l for l in surface.locked if "::" not in l]
    funcs = [_entry(l) for l in surface.locked if "::" in l]
    cap = seeds.max_seeds_per_node
    return "\n".join([
        f"- Editable files (the ONLY surface a probe can change): {', '.join(surface.writable) or '-'}",
        f"- LOCKED files (never edited): {', '.join(files) or '-'}",
        *([f"- LOCKED code inside editable files (must stay byte-identical): {'; '.join(funcs)}"] if funcs else []),
        f"- Protected config keys (never changed; a diff touching one is rejected at intake): "
        f"{', '.join(surface.protected_keys) or '-'}",
        P.rule("budget_fixed", "- The TRAINING BUDGET is fixed: training longer, more steps / samples / FLOPs / epochs, or a "
               "longer schedule is impossible. Only the use of the fixed budget can change."),
        f"- Seeds: at most {cap} seeds per node (screen + promote + confirm); a probe cannot ask for more "
        "replicates." if cap else "- Seeds: the seed plan is fixed by the loop; a probe cannot ask for more replicates.",
        *guard_line(ctx),
        f"- Init compatibility (keeping the current init): {carried_lesson()}",
        f"- Init swap: {init_swap()}" + (f" Read-only reference code (read it with bash cat / sed; never merged): "
                                      f"{', '.join(refs)}." if (refs := references(ctx)) else ""),
        f"- {no_sweeps()} Probe briefs name a mechanism or a substantiated MAJOR move, not a small value tweak.",
        "- Every probe brief must be ONE change implementable within the editable files and "
        "non-protected keys above. Briefs that name a protected key or ask for more training are "
        "dropped mechanically."])


# ---------------------------------------------------------------------------
# v3-c5 I4: program / moonshot briefs -- structural levers and the measured tail
# ---------------------------------------------------------------------------

INCREMENTAL = ("explore", "exploit", "revisit", "debug")     # node strategies of one-mechanism (incremental) ideas
STRUCTURAL = ("moonshot",)                                  # programs / moonshots


def measured_tail(v: Any, ctx: Any) -> dict[str, Any]:
    """COMPUTED from the ledger view (never hardcoded): of this epoch's screened incremental nodes (strategy in
    :data:`INCREMENTAL`, no wave point / port / base root, >= ``ideation.tail_min_seeds`` seeds) the share whose own
    posterior increment over its parent is >= ``ideation.tail_gain``; the same for the structural nodes at
    ``ideation.structural_gain``. {inc: (k, n, best), struct: (k, n, best), tail_gain, structural_gain}."""
    from loop.core.acquire import wave_node
    from loop.core.levers import opt
    led = v.led
    tg, sg, mn = float(opt(ctx, "tail_gain")), float(opt(ctx, "structural_gain")), int(opt(ctx, "tail_min_seeds"))
    xs: dict[str, list[float]] = {"inc": [], "struct": []}
    for nid, node in led.nodes.items():
        if node.base != led.epoch or nid == led.base_root or node.port_of or wave_node(node):
            continue
        b = v.beliefs.get(nid)
        if b is None or b.n < mn:
            continue
        kind = "inc" if node.strategy in INCREMENTAL else "struct" if node.strategy in STRUCTURAL else ""
        if kind:
            xs[kind].append(b.m)
    stat = lambda ms, thr: (sum(m >= thr for m in ms), len(ms), max(ms) if ms else None)   # noqa: E731
    return {"inc": stat(xs["inc"], tg), "struct": stat(xs["struct"], sg), "tail_gain": tg, "structural_gain": sg,
            "min_seeds": mn}


def tail_line(t: Mapping[str, Any]) -> str:
    (k, n, best), (k2, n2, best2) = t["inc"], t["struct"]
    if not n:
        inc = f"no incremental node of this epoch is screened yet (>= {t['min_seeds']} seed)"
    else:
        inc = (f"{k}/{n} = {k / n:.1%} of this epoch's screened incremental nodes reached an own gain >= "
               f"+{t['tail_gain']:g} (posterior mean over their seeds; the best {best:+.4f})")
    st = (f"; structural (moonshot / program) nodes: {k2}/{n2} reached >= +{t['structural_gain']:g} (best "
          f"{best2:+.4f})" if n2 else "; no structural node of this epoch is screened yet")
    return f"Measured tail (computed from the ledger): {inc}{st}."


def structural_rows(v: Any, store: Any, ctx: Any) -> list[str]:
    """The computed structural levers, each with its brief row id and, where it can be computed, its panel headroom:
    per-tile compute under the FLOP cap ([C-compute], resolution / views keys), registered data-bank datasets never
    run on this base (retiles), the per-dataset public gaps ([XD-..], panel-equivalent = cls weight x gap / the
    number of dataset x head cells the cls axis averages) and the public panel lead ([X-panel])."""
    from loop.agents import diagnose as DX
    from loop.core import levers as LV
    from loop.core.data import OFFICIAL_WEIGHTS
    sg, rows = float(LV.opt(ctx, "structural_gain")), []

    def safe(fn: Any) -> Any:
        try:
            return fn()
        except Exception:  # noqa: BLE001 - one lever never loses the block
            return None
    if cp := safe(lambda: LV.compute(v, store, ctx)):
        if cp.get("multiplier"):
            rows.append("- [C-compute] " + P.rule(
                "struct_compute_head", "per-tile compute: the incumbent runs at flop_fraction {flop_fraction:.3f}, "
                "so x{multiplier:.2f}").format(flop_fraction=cp["flop_fraction"], multiplier=cp["multiplier"])
                        + P.rule(
                            "struct_compute_tail",
                            " more compute PER TILE fits under the FLOP cap at the same tile "
                            "count -- more / larger global views, more local crops, a higher input resolution. No node "
                            "of this epoch measured it at this scale: untested headroom"))
        res = {k: x for k, x in (cp.get("keys") or {}).items() if re.search(r"resolution|(global|local|crop|img|image|input)_size$|patch_size", k)}
        if res:
            rows.append("- [C-compute] input resolution / view size keys now: "
                        + ", ".join(f"{k}={x}" for k, x in res.items()))
    if un := safe(lambda: LV.unrun(v, store, ctx)):
        rows.append(f"- data bank: {len(un)} registered dataset(s) never run on this base ("
                    + ", ".join(n for n, _ in un[:8]) + (" ..." if len(un) > 8 else "")
                    + P.rule("struct_databank_tail", "): a retile (another magnification / tile size / curation) is one key, "
                                          "data.dataset_dir; declare dataset_spec for a new one"))
    gaps = safe(lambda: LV.dataset_gaps(v, store, ctx)) or []
    cls_rows = (store.get_state(DX.K_CLS) or {}).get("rows") if hasattr(store, "get_state") else None
    cls_rows = DX.unheld(cls_rows, store) if cls_rows else cls_rows
    n_ds = len(cls_rows or ()) or len(gaps) or 1       # the cls axis: a mean over dataset x head cells
    w = OFFICIAL_WEIGHTS.get(P.target(), 0.0)
    for g in gaps[:8]:
        pe = w * g["gap"] / n_ds
        rows.append(f"- [XD-{g['dataset']}] {g['dataset']} ({g['head']}): public {g['ext'][0]} leads by "
                    f"{g['gap']:+.4f}; closing it on that head is worth {pe:+.4f} of the panel ({w:g} x gap / {n_ds} "
                    f"dataset-head cells of the {P.target()} axis)"
                    + (" -- alone >= the structural bar" if pe >= sg else ""))
    for r in safe(lambda: LV.external(v, ctx)) or []:
        if r.get("axis") == "panel" and r.get("best"):
            lead = r["best"][0] - r["ours"]
            rows.append(f"- [X-panel] the best public model {r['best'][1]} leads the official panel by {lead:+.4f}"
                        + (f" ({lead / sg:.1f}x the structural bar)" if sg > 0 else ""))
    return rows


def structural_block(v: Any, store: Any, ctx: Any) -> str:
    """The program / moonshot brief section: build for a single-shot gain >= ``ideation.structural_gain`` from a
    STRUCTURAL lever, with the measured tail of incremental ideas this epoch (:func:`measured_tail`)."""
    from loop.core.levers import opt
    sg = float(opt(ctx, "structural_gain"))
    try:
        tail = tail_line(measured_tail(v, ctx))
    except Exception as exc:  # noqa: BLE001
        tail = f"Measured tail: not computed ({type(exc).__name__})."
    rows = structural_rows(v, store, ctx)
    return (f"## Structural levers (aim for >= +{sg:g} on the panel in ONE shot)\n{tail}\nAn incremental tweak is "
            "very unlikely to move the incumbent; this session is for a STRUCTURAL change -- "
            + P.rule("struct_levers", "per-tile compute, input resolution, the training data itself, or closing a "
                     "measured per-dataset gap") + " -- sized from the headroom "
            "below (cite its row id in the budget).\n"
            + ("\n".join(rows) if rows else "(no structural headroom could be computed for this incumbent)"))


def _measured(ctx: Any, store: Any) -> list[str]:
    """v3-c8a B2: loop.core.datasets.lines (the per-dataset measured table), [] when unavailable."""
    from loop.core import datasets as DS
    try:
        return DS.lines(ctx, store)
    except Exception:  # noqa: BLE001 - a brief never fails on the table
        return []
